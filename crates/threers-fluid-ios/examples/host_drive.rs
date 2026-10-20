//! Drive the iOS entry points on a development Mac.
//!
//!     cargo run --release -p threers-fluid-ios --example host_drive
//!
//! The same `CAMetalLayer`, the same wgpu surface and the same Metal
//! drawables an iPad uses — everything between the C ABI and the screen,
//! short of the screen. What it cannot tell you is how fast an A-series GPU
//! is: this is an M-series Mac and it will flatter every number below.
//!
//! What it *can* tell you is what each memory budget buys, which is the
//! decision an app actually has to make before it ships.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this harness needs macOS — on a device, call the C ABI from the app");
}

#[cfg(target_os = "macos")]
fn main() {
    use objc2_core_foundation::CGSize;
    use objc2_quartz_core::CAMetalLayer;
    use std::ffi::c_void;
    use threers_fluid_ios::*;

    let (width, height) = (1280u32, 720u32);
    let layer = CAMetalLayer::new();
    layer.setDrawableSize(CGSize {
        width: width as f64,
        height: height as f64,
    });
    let ptr = objc2::rc::Retained::as_ptr(&layer) as *mut c_void;

    println!("threers fluid, through the iOS C ABI, at {width}x{height}");
    println!("a budget of 0 takes the conservative default\n");
    println!(
        "  {:>7}  {:>8}  {:>8}  {:>9}  {:>8}  {:>6}",
        "budget", "solver", "dye", "allocated", "gpu solve", "fps"
    );

    for budget in [32u32, 64, 128, 192, 256, 512] {
        let fluid = unsafe { threers_fluid_create(ptr, width, height, budget) };
        if fluid.is_null() {
            println!("  {budget:>5} MB  — no Metal adapter for this layer");
            continue;
        }
        unsafe {
            // A plume, then a drag and a pinch, as a finger would.
            threers_fluid_set(fluid, Key::Splat as u32, 6.0, 0.0, 0.0);
            for i in 0..90 {
                if i == 30 {
                    threers_fluid_set(fluid, Key::Touch as u32, 6.0, -2.0, 1.0);
                }
                if i == 60 {
                    threers_fluid_set(fluid, Key::Pinch as u32, 1.1, 0.0, 0.0);
                }
                threers_fluid_frame(fluid, 1.0 / 60.0);
            }
            let mut info = FluidInfo {
                solver: 0,
                dye: 0,
                megabytes: 0,
                fps: 0.0,
                solver_ms: 0.0,
            };
            threers_fluid_info(fluid, &mut info);
            println!(
                "  {budget:>5} MB  {:>7}³  {:>7}³  {:>6} MB  {:>6.2}ms  {:>6.0}",
                info.solver, info.dye, info.megabytes, info.solver_ms, info.fps
            );
            threers_fluid_destroy(fluid);
        }
    }

    // And the speed knob, at the suggested budget.
    let fluid = unsafe { threers_fluid_create(ptr, width, height, 0) };
    if fluid.is_null() {
        return;
    }
    println!("\n  speed knob, at the default budget:");
    unsafe {
        threers_fluid_set(fluid, Key::Splat as u32, 6.0, 0.0, 0.0);
        for speed in [0.0f32, 1.0, 2.5, 4.0] {
            threers_fluid_set(fluid, Key::Speed as u32, speed, 0.0, 0.0);
            for _ in 0..60 {
                threers_fluid_frame(fluid, 1.0 / 60.0);
            }
            let mut info = FluidInfo {
                solver: 0,
                dye: 0,
                megabytes: 0,
                fps: 0.0,
                solver_ms: 0.0,
            };
            threers_fluid_info(fluid, &mut info);
            println!(
                "    {speed:>4.1}x real time   {:>6.2}ms in the solver, {:>5.0} fps",
                info.solver_ms, info.fps
            );
        }
        // A resize, which is what rotating the device does.
        threers_fluid_set(fluid, Key::Resize as u32, 2048.0, 1536.0, 0.0);
        let drew = threers_fluid_frame(fluid, 1.0 / 60.0);
        println!("\n  resized to 2048x1536 and drew: {drew}");
        threers_fluid_destroy(fluid);
    }
}
