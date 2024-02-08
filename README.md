# rknpu2-rs
Unofficial Rockchip RKNN framework rust binding.

## Getting Started

1. Deploy `librknnrt.so` to system library folder `/lib`

    This shared library can be found in [rknpu2 official repo](https://github.com/rockchip-linux/rknpu2), Common path is `runtime/<CHIP>/Linux/librknn_api/<ARCH>/librknnrt.so`

    For more information please refer to [official documentations](https://github.com/rockchip-linux/rknpu2/tree/master/doc).

2. Clone this repository & build.

```shell
git clone https://github.com/RoggeOhta/rknpu2-rs.git
cd rknpu2-rs
cargo build
cargo test
```

## Note

This project requires aarch64 system and RKNN NPU to run.

For now only tested on RK3588 aarch64.