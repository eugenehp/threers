//! Sets `metal_backend` where the native Metal backend builds: the `metal`
//! feature on macOS, iOS, tvOS or visionOS.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let apple = matches!(os.as_str(), "macos" | "ios" | "tvos" | "visionos");
    if apple && std::env::var_os("CARGO_FEATURE_METAL").is_some() {
        println!("cargo::rustc-cfg=metal_backend");
    }
}
