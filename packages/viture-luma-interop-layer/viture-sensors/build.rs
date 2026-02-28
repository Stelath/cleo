use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let include_dir = manifest_dir.join("viture-sdk").join("include");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let bindings_out = out_dir.join("viture_bindings.rs");

    println!(
        "cargo:rerun-if-changed={}",
        include_dir.join("viture_glasses_provider.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        include_dir.join("viture_device.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        include_dir.join("viture_device_carina.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        include_dir.join("viture_protocol_public.h").display()
    );

    // Keep bindgen integrated so header/API drift is caught in CI,
    // but do not overwrite hand-curated FFI at src/sdk/ffi.rs.
    let wrapper_path = out_dir.join("viture_wrapper.hpp");
    let wrapper = r#"
        #include <stdint.h>
        #include "viture_glasses_provider.h"
        #include "viture_device.h"
        #include "viture_device_carina.h"
        #include "viture_protocol_public.h"
    "#;
    let _ = fs::write(&wrapper_path, wrapper);

    let builder = bindgen::Builder::default()
        .header(wrapper_path.to_string_lossy())
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg(format!("-I{}", include_dir.to_string_lossy()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("xr_device_provider_.*")
        .allowlist_type("XR.*")
        .allowlist_var("VITURE_.*")
        .generate_comments(false);
    match builder.generate() {
        Ok(bindings) => {
            let _ = bindings.write_to_file(&bindings_out);
            println!("cargo:warning=bindgen generated {}", bindings_out.display());
        }
        Err(err) => {
            println!("cargo:warning=bindgen generation failed: {err}");
        }
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if target_os == "macos" && target_arch == "aarch64" {
        let lib_dir = manifest_dir.join("viture-sdk").join("aarch64");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=glasses");
        println!("cargo:rustc-link-lib=dylib=carina_vio");
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            lib_dir.to_string_lossy()
        );
    }
}
