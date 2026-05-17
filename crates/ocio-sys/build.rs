use std::{env, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .ancestors()
        .nth(2)
        .expect("ocio-sys must live under crates/ocio-sys");
    let ocio_src = workspace_root.join("third_party/OpenColorIO");
    let inherited_cxxflags = env::var("CXXFLAGS").unwrap_or_default();
    let cxxflags = if inherited_cxxflags.is_empty() {
        "-include cstdint".to_string()
    } else {
        format!("-include cstdint {inherited_cxxflags}")
    };
    unsafe {
        env::set_var("CXXFLAGS", cxxflags);
    }

    println!("cargo:rerun-if-changed=src/shim.cpp");
    println!("cargo:rerun-if-changed=src/shim.h");
    println!(
        "cargo:rerun-if-changed={}",
        ocio_src.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        ocio_src.join("include/OpenColorIO/OpenColorIO.h").display()
    );

    let dst = cmake::Config::new(&ocio_src)
        .define("BUILD_SHARED_LIBS", "ON")
        .define("OCIO_BUILD_APPS", "OFF")
        .define("OCIO_BUILD_TESTS", "OFF")
        .define("OCIO_BUILD_GPU_TESTS", "OFF")
        .define("OCIO_BUILD_PYTHON", "OFF")
        .define("OCIO_BUILD_JAVA", "OFF")
        .define("OCIO_BUILD_DOCS", "OFF")
        .define("OCIO_BUILD_OPENFX", "OFF")
        .define("OCIO_BUILD_NUKE", "OFF")
        .define("OCIO_INSTALL_EXT_PACKAGES", "MISSING")
        .define("yaml-cpp_CXX_FLAGS", "-include cstdint")
        .cxxflag("-include")
        .cxxflag("cstdint")
        .build();

    let include_dir = dst.join("include");
    let lib_dir = if dst.join("lib64").exists() {
        dst.join("lib64")
    } else {
        dst.join("lib")
    };

    cc::Build::new()
        .cpp(true)
        .file("src/shim.cpp")
        .include(&include_dir)
        .include(ocio_src.join("include"))
        .flag_if_supported("-std=c++17")
        .compile("ocio_sys_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=OpenColorIO");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
