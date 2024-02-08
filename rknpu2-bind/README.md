

# Note
Only support and tested on RK3588 AArch64 v1.5.2 for now. 
If you need to specify the chip or arch you using, edit `Cargo.toml` feature.
If you need to specify the rknpu2 version, edit build.rs `VERSION` const.
If download from build.rs interrupted you'll have to delete `runtime` folder and run it again due to potential file incompletion.