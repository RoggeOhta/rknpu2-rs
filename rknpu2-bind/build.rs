#![allow(dead_code)]

use anyhow::{Context, Result};
use reqwest::blocking::{Client, ClientBuilder};

use std::collections::HashMap;
use std::env;
use std::fmt::Display;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const VERSION: &str = "v1.5.2";

#[cfg(feature = "rk3588")]
const CHIP: Chip = Chip::RK3588;

#[cfg(feature = "rk356x")]
const CHIP: Chip = Chip::RK356X;

#[cfg(feature = "rv1106")]
const CHIP: Chip = Chip::RV1106;

#[cfg(feature = "aarch64")]
const ARCH: Arch = Arch::Aarch64;

#[cfg(feature = "armhf")]
const ARCH: Arch = Arch::Armhf;

fn main() {
    let lib_dir = PathBuf::from(env::var("OUT_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("runtime")
        .join(VERSION)
        .join(CHIP.to_string())
        .join(ARCH.to_string());
    download_runtime(&lib_dir);

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={}", lib_dir.display());
    println!("cargo:rustc-link-lib=rknnrt");
    println!("cargo:rustc-link-lib=rknn_api");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("headers/wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn features_check() {
    if cfg!(feature = "rv1106") && cfg!(feature = "rk356x") {
        panic!("Only one of the features 'rv1106' and 'rk356x' can be enabled at the same time");
    }

    if cfg!(feature = "rv1106") && cfg!(feature = "rk3588") {
        panic!("Only one of the features 'rv1106' and 'rk3588' can be enabled at the same time");
    }

    if cfg!(feature = "rk356x") && cfg!(feature = "rk3588") {
        panic!("Only one of the features 'rk356x' and 'rk3588' can be enabled at the same time");
    }

    if cfg!(feature = "aarch64") && cfg!(feature = "armhf") {
        panic!("Only one of the features 'aarch64' and 'armhf' can be enabled at the same time");
    }

    if !cfg!(feature = "rv1106") && !cfg!(feature = "rk356x") && !cfg!(feature = "rk3588") {
        panic!("One of the features 'rv1106', 'rk356x' and 'rk3588' must be enabled");
    }

    if !cfg!(feature = "aarch64") && !cfg!(feature = "armhf") {
        panic!("One of the features 'aarch64' and 'armhf' must be enabled");
    }
}

fn download_runtime(download_dir: &PathBuf) {
    let mut runtime = HashMap::new();

    runtime.insert("librknnrt.so", format!("https://github.com/rockchip-linux/rknpu2/raw/{VERSION}/runtime/{CHIP}/Linux/librknn_api/{ARCH}/librknnrt.so"));
    runtime.insert("rknn_matmul_api.h", format!("https://github.com/rockchip-linux/rknpu2/raw/{VERSION}/runtime/{CHIP}/Linux/librknn_api/include/rknn_matmul_api.h"));
    runtime.insert("rknn_api.h", format!("https://github.com/rockchip-linux/rknpu2/raw/{VERSION}/runtime/{CHIP}/Linux/librknn_api/include/rknn_api.h"));

    #[cfg(feature = "mirror")]
    {
        for (_file, url) in runtime.iter_mut() {
            *url = format!("https://mirror.ghproxy.com/{url}");
        }
    }

    if !download_dir.exists() {
        std::fs::create_dir_all(&download_dir).unwrap();
    }

    let symlink_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("runtime");

    if !symlink_dir.exists() {
        std::fs::create_dir_all(&symlink_dir).unwrap();
    }

    for (file, url) in runtime.iter() {
        download_file(url, download_dir.join(file)).unwrap();
        symlink(download_dir.join(file), symlink_dir.join(file)).unwrap();
    }
    symlink(
        download_dir.join("librknnrt.so"),
        download_dir.join("librknn_api.so"),
    )
    .unwrap();
}

fn download_file<P: AsRef<Path>>(url: &str, path: P) -> Result<()> {
    let path_ref = path.as_ref();
    if path_ref.exists() {
        return Ok(());
    }

    static CLIENT: OnceLock<Client> = OnceLock::new();
    let client = CLIENT.get_or_init(|| {
        let proxy = system_proxy::env::from_curl_env();
        let mut builder = ClientBuilder::new();

        if let Some(proxy) = proxy.http {
            builder = builder.proxy(reqwest::Proxy::http(proxy).unwrap());
        } else if let Some(proxy) = proxy.https {
            builder = builder.proxy(reqwest::Proxy::https(proxy).unwrap());
        }

        builder.build().unwrap()
    });

    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("Failed to send request to {}", url))?;

    if response.status().is_client_error() {
        return Err(anyhow::anyhow!("Failed to download file from {}", url));
    }

    let mut file = File::create(path_ref)
        .with_context(|| format!("Failed to create file at {:?}", path_ref))?;

    response
        .copy_to(&mut file)
        .with_context(|| "Failed to write response to file")?;

    Ok(())
}

fn symlink<P1: AsRef<Path>, P2: AsRef<Path>>(path: P1, symlink_path: P2) -> Result<()> {
    let path_ref = path.as_ref();
    let symlink_path = symlink_path.as_ref();
    if symlink_path.exists() {
        std::fs::remove_file(&symlink_path)
            .with_context(|| format!("Failed to remove file at {:?}", &symlink_path))?;
    }
    std::os::unix::fs::symlink(path_ref, &symlink_path)
        .with_context(|| format!("Failed to create symlink at {:?}", symlink_path))?;
    Ok(())
}

enum Chip {
    RV1106,
    RK356X,
    RK3588,
}

impl Display for Chip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Chip::RV1106 => write!(f, "RV1106"),
            Chip::RK356X => write!(f, "RK356X"),
            Chip::RK3588 => write!(f, "RK3588"),
        }
    }
}

enum Arch {
    Aarch64,
    Armhf,
}

impl Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Arch::Aarch64 => write!(f, "aarch64"),
            Arch::Armhf => write!(f, "armhf"),
        }
    }
}
