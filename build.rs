//! Sets `metal_backend` where the native Metal backend builds: the `metal`
//! feature on macOS, iOS, tvOS or visionOS, and `metal_core` where it or
//! `metal-fluid` does. Sets `winit_window` where the
//! `sketch` feature brings in winit.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let apple = matches!(os.as_str(), "macos" | "ios" | "tvos" | "visionos");
    let metal = std::env::var_os("CARGO_FEATURE_METAL").is_some();
    if apple && metal {
        println!("cargo::rustc-cfg=metal_backend");
    }
    // `metal_core`: the device, the Objective-C glue and the fluid — what
    // `metal-fluid` builds without the scene renderer or wgpu.
    if apple && (metal || std::env::var_os("CARGO_FEATURE_METAL_FLUID").is_some()) {
        println!("cargo::rustc-cfg=metal_core");
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
