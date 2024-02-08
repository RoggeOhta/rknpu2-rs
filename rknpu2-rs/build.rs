fn main() {
    // For test purpose
    println!("cargo:rustc-link-search=rknpu2/runtime/RK3588/Linux/librknn_api/aarch64");
    println!("cargo:rustc-link-lib=rknnrt");
    println!("cargo:rustc-link-lib=rknn_api");
}
