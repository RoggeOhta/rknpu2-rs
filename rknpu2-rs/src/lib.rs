#![allow(unused_mut)]
#![allow(unused)]

use ndarray::prelude::*;
use rknpu2_sys;
use std::ffi::c_void;
use std::fmt::Error;
use std::fs;
use std::mem;
use std::ptr;

pub type RKNNContext = u64;

pub fn rknn_init(
    ctx: *mut RKNNContext,
    model: Vec<u8>,
    flag: u32,
    rknn_init_extend: *mut rknpu2_sys::rknn_init_extend,
) -> Result<i32, i32> {
    let model_len = model.len() as u32;
    let mut model = model;
    let mut model: *mut c_void = model.as_mut_ptr() as *mut c_void;
    unsafe {
        let res = rknpu2_sys::rknn_init(ctx, model, model_len, flag, rknn_init_extend);
        if res == 0 {
            return Ok(0);
        } else {
            return Err(res);
        }
    }
}

pub type RKNNInputOutputNumber = rknpu2_sys::rknn_input_output_num;

pub fn get_input_output_number(ctx: RKNNContext) -> Result<RKNNInputOutputNumber, i32> {
    let cmd = rknpu2_sys::_rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM;
    let mut input: rknpu2_sys::rknn_input_output_num = RKNNInputOutputNumber {
        n_input: 0,
        n_output: 0,
    };
    let size = mem::size_of::<rknpu2_sys::rknn_input_output_num>() as u32;
    let input_ptr = &mut input as *mut _ as *mut c_void;

    unsafe {
        let ret = rknpu2_sys::rknn_query(ctx, cmd, input_ptr, size);
        if ret == 0 {
            return Ok(input);
        } else {
            return Err(ret);
        }
    }
}

pub type RKNNTensorAttr = rknpu2_sys::rknn_tensor_attr;
pub fn get_model_input_info(ctx: RKNNContext, input_num: u32) -> Result<Vec<RKNNTensorAttr>, i32> {
    let mut input_attrs: Vec<RKNNTensorAttr> = Vec::with_capacity(input_num as usize);

    // metset(input, 0, sizeof(input));
    unsafe {
        for _ in 0..input_attrs.capacity() {
            input_attrs.push(mem::zeroed());
        }
    }

    // for (int i = 0; i < io_num.n_input; i++){
    //     input_attrs[i].index = i;
    //     ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
    //     sizeof(rknn_tensor_attr));
    // }
    let cmd = rknpu2_sys::_rknn_query_cmd_RKNN_QUERY_INPUT_ATTR;
    let size = mem::size_of::<RKNNTensorAttr>() as u32;
    for i in 0..input_num as usize {
        let mut entry = input_attrs.get_mut(i).unwrap();
        entry.index = i as u32;
        let entry_ptr = entry as *mut _ as *mut c_void;
        unsafe {
            let ret = rknpu2_sys::rknn_query(ctx, cmd, entry_ptr, size);
            if ret != 0 {
                return Err(ret);
            }
        }
    }
    return Ok(input_attrs);
}

pub fn get_model_output_info(
    ctx: RKNNContext,
    output_num: u32,
) -> Result<Vec<RKNNTensorAttr>, i32> {
    let output_num = output_num as usize;
    // rknn_tensor_attr output_attrs[io_num.n_output];
    let mut output_attrs: Vec<RKNNTensorAttr> = Vec::with_capacity(output_num);

    // memset(output_attrs, 0, sizeof(output_attrs));
    unsafe {
        for _ in 0..output_attrs.capacity() {
            output_attrs.push(mem::zeroed());
        }
    }

    // for (int i = 0; i < io_num.n_output; i++){
    //     output_attrs[i].index = i;
    //     ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR,
    //         &(output_attrs[i]), sizeof(rknn_tensor_attr));
    // }
    let cmd = rknpu2_sys::_rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR;
    let size = mem::size_of::<RKNNTensorAttr>() as u32;
    for i in 0..output_num {
        let entry = output_attrs.get_mut(i).unwrap();
        let entry_ptr = entry as *mut _ as *mut c_void;
        entry.index = i as u32;
        unsafe {
            let ret = rknpu2_sys::rknn_query(ctx, cmd, entry_ptr, size);
            if ret != 0 {
                return Err(ret);
            }
        }
    }

    return Ok(output_attrs);
}

pub type RKNNInput = rknpu2_sys::rknn_input;
pub type RKNNOutput = rknpu2_sys::rknn_output;
pub fn make_rknn_image_input(mut image_array_view: ArrayViewMut<u8, IxDyn>) -> Vec<RKNNInput> {
    // RKNNInput tensor_inputs;
    #[allow(invalid_value)]
    let mut tensor_inputs: RKNNInput = unsafe { mem::MaybeUninit::uninit().assume_init() };
    // memset(tensor_inputs, 0, sizeof(tensor_inputs));
    unsafe { ptr::write_bytes((&mut tensor_inputs) as *mut _, 0, 1) }

    tensor_inputs.index = 0;
    tensor_inputs.type_ = rknpu2_sys::_rknn_tensor_type_RKNN_TENSOR_UINT8;
    tensor_inputs.fmt = rknpu2_sys::_rknn_tensor_format_RKNN_TENSOR_NHWC;
    tensor_inputs.size = image_array_view.len() as u32;
    tensor_inputs.buf = image_array_view.as_mut_ptr() as *mut _ as *mut c_void;

    return vec![tensor_inputs];
}

pub fn rknn_inputs_set(
    ctx: RKNNContext,
    input_num: u32,
    mut inputs: Vec<RKNNInput>,
) -> Result<i32, i32> {
    let inputs_ptr = inputs.as_mut_ptr() as *mut RKNNInput;
    unsafe {
        let ret = rknpu2_sys::rknn_inputs_set(ctx, input_num, inputs_ptr);
        if ret == 0 {
            return Ok(ret);
        } else {
            return Err(ret);
        }
    }
}

pub fn rknn_run(ctx: RKNNContext) -> Result<i32, i32> {
    unsafe {
        let ret = rknpu2_sys::rknn_run(ctx, ptr::null_mut());
        if ret == 0 {
            return Ok(ret);
        } else {
            return Err(ret);
        }
    }
}

pub fn rknn_outputs_get(ctx: RKNNContext, output_num: u32) -> Result<Vec<RKNNOutput>, i32> {
    let mut outputs: Vec<RKNNOutput> = Vec::with_capacity(output_num as usize);
    // memset(outputs, 0, sizeof(outputs));
    unsafe {
        for _ in 0..outputs.capacity() {
            outputs.push(mem::zeroed());
        }
    }
    let outputs_ptr = outputs.as_mut_ptr();
    unsafe {
        let ret = rknpu2_sys::rknn_outputs_get(ctx, output_num, outputs_ptr, ptr::null_mut());
        if ret == 0 {
            return Ok(outputs);
        } else {
            return Err(ret);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_mut)]
    #![allow(unused)]

    use crate::*;
    use rknpu2_sys::_rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM;
    use std::env::current_dir;
    use std::ffi::c_void;
    use std::fs;
    use std::ptr;
    use std::time::Duration;
    use std::time::Instant;

    use image::imageops;
    use image::{io::Reader as ImageReader, ImageBuffer};
    use ndarray::prelude::*;

    #[test]
    fn test_rknn_init() {
        let mut ctx: RKNNContext = 0;
        let mut ctx: *mut RKNNContext = &mut ctx;
        let mut model =
            fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/yolov8.rknn")).unwrap();

        assert!(rknn_init(ctx, model, 0, std::ptr::null_mut()).is_ok());
    }

    #[test]
    fn test_get_input_output_number() {
        let mut ctx: RKNNContext = 0;
        let mut ctx_ptr: *mut RKNNContext = &mut ctx;
        let mut model =
            fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/yolov8.rknn")).unwrap();

        assert!(rknn_init(ctx_ptr, model, 0, std::ptr::null_mut()).is_ok());

        let io_info = get_input_output_number(ctx);
        assert!(io_info.is_ok());
        dbg!(io_info.unwrap());
    }

    #[test]
    fn test_get_model_input_info() {
        let mut ctx: RKNNContext = 0;
        let mut ctx_ptr: *mut RKNNContext = &mut ctx;
        let mut model =
            fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/yolov8.rknn")).unwrap();

        assert!(rknn_init(ctx_ptr, model, 0, std::ptr::null_mut()).is_ok());

        let io_info = get_input_output_number(ctx);
        assert!(io_info.is_ok());
        let io_info = io_info.unwrap();

        let model_input_info = get_model_input_info(ctx, io_info.n_input);
        dbg!(model_input_info.clone().unwrap());
        assert!(model_input_info.is_ok());
    }

    #[test]
    fn test_get_model_output_info() {
        let mut ctx: RKNNContext = 0;
        let mut ctx_ptr: *mut RKNNContext = &mut ctx;
        let mut model =
            fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/yolov8.rknn")).unwrap();

        assert!(rknn_init(ctx_ptr, model, 0, std::ptr::null_mut()).is_ok());

        let io_info = get_input_output_number(ctx);
        assert!(io_info.is_ok());
        let io_info = io_info.unwrap();

        let model_input_info = get_model_output_info(ctx, io_info.n_output);
        assert!(model_input_info.is_ok());
        dbg!(model_input_info.unwrap());
    }

    fn read_image(img_path: String) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        let img = ImageReader::open(img_path).unwrap().decode().unwrap();
        let img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> = img.to_rgb8();
        return img;
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

    #[test]
    fn test_read_image() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/bus.jpg");
        let mut img_buffer = read_image(path.into());
        let img_array_view = image_to_array_view(&mut img_buffer);
        dbg!(img_array_view.shape());
    }

    #[test]
    fn test_make_rknn_image_input() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/bus.jpg");
        let mut img_buffer = read_image(path.into());
        let img_array_view = image_to_array_view(&mut img_buffer);
        dbg!(make_rknn_image_input(img_array_view));
    }

    #[test]
    fn test_rknn_set_inputs() {
        let mut ctx: RKNNContext = 0;
        let mut ctx_ptr: *mut RKNNContext = &mut ctx;
        let mut model =
            fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/yolov8.rknn")).unwrap();

        assert!(rknn_init(ctx_ptr, model, 0, std::ptr::null_mut()).is_ok());

        let io_info = get_input_output_number(ctx);
        assert!(io_info.is_ok());
        let io_info = io_info.unwrap();

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/bus.jpg");
        let mut img_buffer = read_image(path.into());
        let img_array_view = image_to_array_view(&mut img_buffer);
        let rknn_inputs = make_rknn_image_input(img_array_view);
        let ret = rknn_inputs_set(ctx, io_info.n_input, rknn_inputs);
        assert!(ret.is_ok());
    }

    #[test]
    fn test_rknn_run() {
        let mut ctx: RKNNContext = 0;
        let mut ctx_ptr: *mut RKNNContext = &mut ctx;
        let mut model =
            fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/yolov8.rknn")).unwrap();

        assert!(rknn_init(ctx_ptr, model, 0, std::ptr::null_mut()).is_ok());

        let io_info = get_input_output_number(ctx);
        assert!(io_info.is_ok());
        let io_info = io_info.unwrap();

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/bus.jpg");
        let mut img_buffer = read_image(path.into());
        let img_array_view = image_to_array_view(&mut img_buffer);
        let rknn_inputs = make_rknn_image_input(img_array_view);
        let ret = rknn_inputs_set(ctx, io_info.n_input, rknn_inputs);
        assert!(ret.is_ok());

        let ret = rknn_run(ctx);
        assert!(ret.is_ok());
    }

    #[test]
    fn test_rknn_outputs_get() {
        let mut ctx: RKNNContext = 0;
        let mut ctx_ptr: *mut RKNNContext = &mut ctx;
        let mut model =
            fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/yolov8.rknn")).unwrap();

        assert!(rknn_init(ctx_ptr, model, 0, std::ptr::null_mut()).is_ok());

        let io_info = get_input_output_number(ctx);
        assert!(io_info.is_ok());
        let io_info = io_info.unwrap();

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/bus.jpg");
        let mut img_buffer = read_image(path.into());
        let img_array_view = image_to_array_view(&mut img_buffer);
        let rknn_inputs = make_rknn_image_input(img_array_view);
        let ret = rknn_inputs_set(ctx, io_info.n_input, rknn_inputs);
        assert!(ret.is_ok());

        let start = Instant::now();
        let ret = rknn_run(ctx);
        dbg!(start.elapsed());
        assert!(ret.is_ok());

        let rknn_outputs = rknn_outputs_get(ctx, io_info.n_output);
        assert!(rknn_outputs.is_ok());
        dbg!(rknn_outputs);
    }
}
