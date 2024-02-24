use ndarray::{prelude::*, ViewRepr};
use rknpu2_rs::*;
use rknpu2_rs::{RKNNContext, RKNNContextPack, RKNNOutput};
use std::fs;
use std::os::raw::c_void;
use std::time::Instant;

#[derive(Debug, Default, Clone)]
struct DetectResult {
    pub id: u32,
    pub conf: f32,
    pub rect: (u32, u32, u32, u32),
}

fn clip<T: PartialOrd>(n: T, low: T, high: T) -> T {
    if n > high {
        return high;
    } else if n < low {
        return low;
    } else {
        return n;
    }
}

fn qnt_f32_to_affine(n: f32, zp: i32, scale: f32) -> i8 {
    let dst_val = (n / scale) + zp as f32;
    let res = clip(dst_val, -128f32, 127f32) as i8;
    return res;
}

fn deqnt_affine_to_f32(qnt: i8, zp: i32, scale: f32) -> f32 {
    return (qnt as f32 - zp as f32) * scale;
}

fn ptr_to_arrayviewmut(ptr: *mut c_void, shape: &[u32]) -> ArrayBase<ViewRepr<&mut i8>, IxDyn> {
    let arr: ArrayBase<ViewRepr<&mut i8>, IxDyn>;
    let shape = shape.iter().map(|&x| x as usize).collect::<Vec<usize>>();

    unsafe {
        let _arr = ArrayViewMut::from_shape_ptr(shape, ptr as *mut i8);
        arr = _arr.into_dyn();
    }
    return arr;
}

fn calc_iou(rect_a: (u32, u32, u32, u32), rect_b: (u32, u32, u32, u32)) -> f32 {
    let (ax, ay, aw, ah) = rect_a;
    let (bx, by, bw, bh) = rect_b;

    let ax_max = ax + aw;
    let ay_max = ay + ah;
    let bx_max = bx + bw;
    let by_max = by + bh;

    let inter_x_min = ax.max(bx);
    let inter_y_min = ay.max(by);
    let inter_x_max = ax_max.min(bx_max);
    let inter_y_max = ay_max.min(by_max);

    // make sure width and height is positive
    let inter_width = if inter_x_max > inter_x_min {
        inter_x_max - inter_x_min
    } else {
        0
    };
    let inter_height = if inter_y_max > inter_y_min {
        inter_y_max - inter_y_min
    } else {
        0
    };

    // calc intersection area
    let inter_area = inter_width * inter_height;

    let area_a = aw * ah;
    let area_b = bw * bh;

    // calc union area
    let union_area = area_a + area_b - inter_area;

    // 计算IoU，转换为浮点数进行除法运算
    if union_area == 0 {
        0.0 // 避免除以0
    } else {
        inter_area as f32 / union_area as f32
    }
}

fn nms(detect_results: Vec<DetectResult>, iou_thresh: f32) -> Vec<DetectResult> {
    let mut oust_sentinel = vec![0; detect_results.len()];
    let mut new_detect_results: Vec<DetectResult> = Vec::new();
    let n = detect_results.len();
    for i in 0..n {
        let curr_res = detect_results.get(i).unwrap();
        // current value is not ousted
        if *oust_sentinel.get(i).unwrap() == 0 {
            // curr_res will be used to calc iou later, so need clone here.
            new_detect_results.push(std::mem::take(&mut curr_res.clone()));
        } else {
            continue;
        }

        // shadow curr_res from mut to immut
        let curr_res = detect_results.get(i).unwrap();
        for j in i + 1..n {
            let next_res = detect_results.get(j).unwrap();
            if *oust_sentinel.get(j).unwrap() == 1 || next_res.id != curr_res.id {
                continue;
            }

            let iou = calc_iou(curr_res.rect, next_res.rect);
            if iou > iou_thresh {
                *oust_sentinel.get_mut(j).unwrap() = 1;
            }
        }
    }
    return new_detect_results;
}

fn post_process(
    ctx: RKNNContextPack,
    outputs: Vec<RKNNOutput>,
    conf_thresh: f32,
    iou_thresh: f32,
) -> Vec<DetectResult> {
    #![allow(unused_variables)]
    // input information
    let input_num = ctx.io_info.n_input;
    let output_num = ctx.io_info.n_output;
    let input_w = ctx.input_shape.get(1).unwrap().clone();
    let input_h = ctx.input_shape.get(2).unwrap().clone();

    // branch info
    let output_0 = ctx.output_info.get(0).unwrap();
    let dfl_len = output_0.dims[1] / 4;
    let output_per_branch = output_num / 3;

    let mut detect_results: Vec<DetectResult> = Vec::new();

    // large, medium, small target branches
    for i in 0..3usize {
        let box_idx = i * output_per_branch as usize;
        let score_idx = i * output_per_branch as usize + 1;
        let score_sum_idx = i * output_per_branch as usize + 2;

        // process score sum
        let score_sum_zp = ctx.output_info.get(score_sum_idx).unwrap().zp;
        let score_sum_scale = ctx.output_info.get(score_sum_idx).unwrap().scale;
        let socre_sum_output_dims = &ctx.output_info.get(score_sum_idx).unwrap().dims[0..4];
        let _score_sum_output = outputs.get(score_sum_idx).unwrap();
        // [1,1,anchor,anchor]
        // for fast filter
        let score_sum_output = ptr_to_arrayviewmut(_score_sum_output.buf, socre_sum_output_dims);

        // process score
        let score_zp = ctx.output_info.get(score_idx).unwrap().zp;
        let score_scale = ctx.output_info.get(score_idx).unwrap().scale;
        let score_output_dims = &ctx.output_info.get(score_idx).unwrap().dims[0..4];
        let _score_output = outputs.get(score_idx).unwrap();
        // [1,class,anchor,anchor]
        let score_output = ptr_to_arrayviewmut(_score_output.buf, score_output_dims);
        let class_num = score_output_dims[1] as usize;

        // process box
        let box_zp = ctx.output_info.get(box_idx).unwrap().zp;
        let box_scale = ctx.output_info.get(box_idx).unwrap().scale;
        let box_output_dims = &ctx.output_info.get(box_idx).unwrap().dims[0..4];
        let _box_output = outputs.get(box_idx).unwrap();
        // [1,4,anchor,anchor]
        let box_output = ptr_to_arrayviewmut(_box_output.buf, box_output_dims);

        // grid info
        let grid_h = ctx.output_info.get(box_idx).unwrap().dims[2] as usize;
        let grid_w = ctx.output_info.get(box_idx).unwrap().dims[3] as usize;
        let stride = input_h as usize / grid_h;

        let score_thresh_i8 = qnt_f32_to_affine(conf_thresh, score_zp, score_scale);
        let score_sum_thresh_i8 = qnt_f32_to_affine(conf_thresh, score_sum_zp, score_sum_scale);

        // iterate through anchors
        for i in 0..grid_h {
            for j in 0..grid_w {
                // fast filter
                if score_sum_output[[0, 0, i, j]] < score_sum_thresh_i8 {
                    continue;
                }

                // find most likely class to this anchor box
                let mut max_score_i8 = score_zp as i8;
                let mut max_class_id = -1i32;
                for class_id in 0..class_num {
                    let curr_class_conf: i8 = score_output[[0, class_id, i, j]];
                    if curr_class_conf > score_thresh_i8 && curr_class_conf > max_score_i8 {
                        max_score_i8 = curr_class_conf;
                        max_class_id = class_id as i32;
                    }
                }

                if max_class_id == -1 {
                    continue;
                }

                // compute box
                let mut curr_box = vec![0f32; 4];
                if max_score_i8 > score_thresh_i8 {
                    // todo!("DFL when dfl > 1");
                    curr_box[0] = deqnt_affine_to_f32(box_output[[0, 0, i, j]], box_zp, box_scale);
                    curr_box[1] = deqnt_affine_to_f32(box_output[[0, 1, i, j]], box_zp, box_scale);
                    curr_box[2] = deqnt_affine_to_f32(box_output[[0, 2, i, j]], box_zp, box_scale);
                    curr_box[3] = deqnt_affine_to_f32(box_output[[0, 3, i, j]], box_zp, box_scale);
                }

                // process (x1,y1,x2,y2) -> (x,y,w,h)
                let x1 = (-curr_box[0] + j as f32 + 0.5) * stride as f32;
                let y1 = (-curr_box[1] as f32 + i as f32 + 0.5) * stride as f32;
                let x2 = (curr_box[2] as f32 + j as f32 + 0.5) * stride as f32;
                let y2 = (curr_box[3] as f32 + i as f32 + 0.5) * stride as f32;
                let w = x2 - x1;
                let h = y2 - y1;
                let curr_box = (x1 as u32, y1 as u32, w as u32, h as u32);

                detect_results.push(DetectResult {
                    id: u32::try_from(max_class_id).unwrap(),
                    conf: deqnt_affine_to_f32(max_score_i8, score_zp, score_scale),
                    rect: curr_box,
                })
            }
        }
    }

    // todo: result something
    if detect_results.len() == 0 {
        return detect_results;
    }

    // b cmp a = reverse order
    detect_results.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap());

    // apply nms
    let detect_results = nms(detect_results, iou_thresh);

    return detect_results;
}

fn t_rknn_init() -> RKNNContext {
    let model = fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/yolov6.rknn")).unwrap();
    return rknn_init(model, 0, std::ptr::null_mut()).unwrap();
}

fn image_to_array_view<'a>(
    img_buffer: &'a mut ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> ArrayViewMut<u8, IxDyn> {
    let (w, h) = (img_buffer.width(), img_buffer.height());
    unsafe {
        let arr =
            ArrayViewMut::from_shape_ptr((w as usize, h as usize, 3), img_buffer.as_mut_ptr());
        let arr = arr.into_dyn();
        return arr;
    }
}

use image::{imageops, ImageBuffer, Rgb};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};

fn main() {
    // init rknn context
    let ctx: RKNNContext = t_rknn_init();
    let io_info = get_input_output_number(ctx);
    let io_info = io_info.unwrap();
    let ctx_pack = make_rknn_context_pack(ctx).unwrap();

    // read image and convert image to ArrayView
    let img_path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/bus.jpg");
    let img = image::open(img_path).expect("Failed to open image.");
    let mut img_buffer: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> = img.to_rgb8();
    img_buffer = imageops::resize(&img_buffer, 640, 640, imageops::FilterType::Lanczos3);
    let img_array_view = image_to_array_view(&mut img_buffer);

    // setup rknn input
    let rknn_inputs = make_rknn_image_input(img_array_view);
    let _ret = rknn_inputs_set(ctx, io_info.n_input, rknn_inputs);

    // run rknn
    let start = Instant::now();
    let _ret = rknn_run(ctx);
    dbg!(start.elapsed());

    // extract rknn outputs
    let rknn_outputs = rknn_outputs_get(ctx, io_info.n_output);
    let rknn_outputs = rknn_outputs.unwrap();
    let res = post_process(ctx_pack, rknn_outputs, 0.5, 0.5);

    // load font
    let font_data = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/font.ttf"));
    let font = Font::try_from_bytes(font_data as &[u8]).expect("Error constructing Font");

    // load class list
    let class_list = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/coco_80_labels_list.txt"
    ));
    let class_list: Vec<_> = class_list.split("\n").collect();

    for detect_res in res {
        let (x, y, w, h) = detect_res.rect;
        let x = i32::try_from(x).unwrap();
        let y = i32::try_from(y).unwrap();

        let rect = Rect::at(x, y).of_size(w, h);
        let color: Rgb<u8> = Rgb([255, 0, 0]);

        draw_hollow_rect_mut(&mut img_buffer, rect, color);

        // text style
        let scale = Scale { x: 20.0, y: 20.0 };
        let text_color: Rgb<u8> = Rgb([0, 255, 0]);

        let mut text = String::from(class_list.get(detect_res.id as usize).unwrap().to_owned());
        text.push_str(" ");
        text.push_str(&detect_res.conf.to_string());

        // draw text
        draw_text_mut(&mut img_buffer, text_color, x, y, scale, &font, &text);
    }

    // save image to file.
    img_buffer
        .save(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/result.jpg"))
        .expect("Failed to save image");
}
