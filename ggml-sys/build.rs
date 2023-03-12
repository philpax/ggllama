use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

fn main() {
    let llama_path = Path::new("../vendor/llama.cpp");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    for filename in ["ggml.c", "ggml.h"] {
        println!(
            "cargo:rerun-if-changed={}",
            llama_path.join(filename).to_str().unwrap()
        );
    }

    // Configure the build.
    let build_target = build_target::target().unwrap();

    let supported_features: HashSet<_> = std::env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap()
        .split(',')
        .map(|s| s.to_string())
        .collect();

    let supports_fma = supported_features.contains("fma");
    let supports_avx = supported_features.contains("avx");
    let supports_avx2 = supported_features.contains("avx2");
    let supports_f16c = supported_features.contains("f16c");
    let supports_sse3 = supported_features.contains("sse3");

    let mut build = cc::Build::new();
    build
        .include(llama_path)
        .cpp(true)
        .file(llama_path.join("ggml.c"));
    // TODO: Apple Silicon support
    if [build_target::Arch::X86, build_target::Arch::X86_64].contains(&build_target.arch) {
        use build_target::Os;
        match build_target.os {
            Os::FreeBSD | Os::Haiku | Os::iOs | Os::MacOs | Os::Linux => {
                if supports_fma {
                    build.flag("-mfma");
                }
                if supports_avx {
                    build.flag("-mavx");
                }
                if supports_avx2 {
                    build.flag("-mavx2");
                }
                if supports_f16c {
                    build.flag("-mf16c");
                }
                if supports_sse3 {
                    build.flag("-msse3");
                }
            }
            Os::Windows => match (supports_avx2, supports_avx) {
                (true, _) => {
                    build.flag("/arch:AVX2");
                }
                (_, true) => {
                    build.flag("/arch:AVX");
                }
                _ => {}
            },
            _ => {}
        }
    }
    build.compile("ggml");

    bindgen::Builder::default()
        .header(llama_path.join("ggml.h").to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .unwrap()
        .write_to_file(out_dir.join("bindings.rs"))
        .unwrap();
}
