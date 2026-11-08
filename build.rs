//! Sets `metal_backend` where the native Metal backend builds: the `metal`
//! feature on macOS, iOS, tvOS or visionOS. Sets `winit_window` where the
//! `sketch` feature brings in winit.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let apple = matches!(os.as_str(), "macos" | "ios" | "tvos" | "visionos");
    if apple && std::env::var_os("CARGO_FEATURE_METAL").is_some() {
        println!("cargo::rustc-cfg=metal_backend");
    }
    // The winit window renderer and `sketch`: the `sketch` feature, wherever
    // winit builds (not wasm, tvOS or visionOS).
    if std::env::var_os("CARGO_FEATURE_SKETCH").is_some()
        && os != "tvos"
        && os != "visionos"
        && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("wasm32")
    {
        println!("cargo::rustc-cfg=winit_window");
    }
}
