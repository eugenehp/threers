//! The volumetric fluid as a C ABI, for iOS and iPadOS.
//!
//! UIKit has no window handle to hand a graphics library. A view that draws
//! with Metal has a `CAMetalLayer`, and that layer *is* the surface — so the
//! whole interface here is: give me your layer, tell me when a finger moves,
//! and call me once a frame.
//!
//! ```swift
//! final class FluidView: UIView {
//!     override class var layerClass: AnyClass { CAMetalLayer.self }
//!     private var fluid: OpaquePointer?
//!
//!     override func didMoveToWindow() {
//!         let layer = Unmanaged.passUnretained(self.layer).toOpaque()
//!         let scale = window?.screen.nativeScale ?? 2
//!         fluid = threers_fluid_create(
//!             layer,
//!             UInt32(bounds.width * scale),
//!             UInt32(bounds.height * scale),
//!             threers_fluid_suggested_budget_mb())
//!     }
//!
//!     // From a CADisplayLink
//!     @objc func tick(_ link: CADisplayLink) {
//!         threers_fluid_frame(fluid, Float(link.targetTimestamp - link.timestamp))
//!     }
//! }
//! ```
//!
//! # Why the budget is an argument
//!
//! A volume costs the cube of its side, and on a phone the penalty for asking
//! for too much is not a slow frame, it is the watchdog killing the app. So
//! nothing here picks a resolution from a quality name: it picks the largest
//! volume that fits the number of megabytes you are willing to spend, through
//! [`threers::fluid::within_memory`], whose estimate is held to what actually
//! gets allocated by `the_memory_estimate_is_the_memory`.
//!
//! [`threers_fluid_suggested_budget_mb`] is a starting point, not an answer —
//! only the app knows what else is resident.

use std::ffi::c_void;

use threers::controls::{OrbitControls, PointerEvent};
use threers::fluid::{within_memory, FluidConfig3d, FluidSim3d};
use threers::gpu::GpuContext;
use threers::{PerspectiveCamera, Vector3};

/// Everything one view needs. Opaque to the caller.
pub struct Fluid {
    gpu: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    sim: FluidSim3d,
    camera: PerspectiveCamera,
    controls: OrbitControls,
    elapsed: f32,
    /// Smoothed whole-frame time, present included.
    frame_ms: f32,
}

/// What the fluid came out as, for a debug overlay.
#[repr(C)]
pub struct FluidInfo {
    pub solver: u32,
    pub dye: u32,
    pub megabytes: u32,
    /// Frames a second as the display delivers them. With `Fifo` present —
    /// which is what this uses, because racing a phone's display link only
    /// heats the device — this is the refresh rate until the fluid cannot
    /// keep up, and is therefore *not* the number to tune against.
    pub fps: f32,
    /// Milliseconds the **GPU** spends in the solver, from timestamp
    /// queries. Zero where the device has none.
    ///
    /// This is the one to tune against, and it has to come from the GPU's
    /// clock rather than the CPU's. With `Fifo` present, acquiring the next
    /// drawable blocks until the display is ready — so a wall-clock timer
    /// around the frame measures the refresh rate and nothing else, and
    /// reports the same number for a volume of 56³ and one of 144³, and for
    /// a speed of zero.
    ///
    /// The ray march is not included; see `FluidSim3d::stage_ms`.
    pub solver_ms: f32,
}

/// A memory budget to start from, in megabytes, when `create` is given zero.
///
/// Deliberately conservative. An app that knows it has the device to itself
/// should pass more; one with a camera feed and a map behind it should pass
/// less. There is no API that tells you how much you may have — the number
/// you get away with depends on everything else running.
const SUGGESTED_BUDGET_MB: u32 = 192;

/// Create a fluid that draws into `layer`. `budget_mb` of 0 takes a
/// conservative default.
///
/// Returns null if there is no Metal device, the layer is not usable, or the
/// budget is too small for even the smallest volume.
///
/// # Safety
///
/// `layer` must be a live `CAMetalLayer` that outlives the returned handle —
/// the view owns it, so keep the view alive. `width` and `height` are in
/// *pixels*, so multiply the view's bounds by the screen's `nativeScale`.
#[no_mangle]
pub unsafe extern "C" fn threers_fluid_create(
    layer: *mut c_void,
    width: u32,
    height: u32,
    budget_mb: u32,
) -> *mut Fluid {
    if layer.is_null() {
        return std::ptr::null_mut();
    }
    let (width, height) = (width.max(1), height.max(1));
    let Ok((gpu, surface)) = pollster::block_on(unsafe { GpuContext::from_metal_layer(layer) })
    else {
        return std::ptr::null_mut();
    };

    let caps = surface.get_capabilities(&gpu.adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        // Fifo everywhere: a phone's display link is the clock, and a fluid
        // that races it only heats the device.
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Srgb,
    };
    surface.configure(&gpu.device, &config);

    // Resolved against the configuration this is about to build, not against
    // what the adapter would allow: the packed dye is off by default, and
    // predicting with it on halves the estimate for the biggest field.
    let base = FluidConfig3d::default();
    let formats = threers::fluid::FluidFormats::for_adapter(&gpu.adapter)
        .unwrap_or_default()
        .resolve(base.compact_dye);
    let budget_mb = if budget_mb == 0 {
        SUGGESTED_BUDGET_MB
    } else {
        budget_mb.max(4)
    };
    let (solver, dye) = within_memory((budget_mb as u64) << 20, 2, formats);
    let sim = match FluidSim3d::with_formats(
        gpu.device.clone(),
        gpu.queue.clone(),
        FluidConfig3d {
            resolution: solver,
            dye_resolution: dye,
            buoyancy: 46.0,
            curl: 26.0,
            density_dissipation: 0.2,
            // A phone marches fewer samples than a desktop and the empty
            // space skip is what makes that affordable.
            march_steps: 64,
            ..base
        },
        formats,
    ) {
        Ok(sim) => sim,
        Err(_) => return std::ptr::null_mut(),
    };

    // Ask for the GPU's own clock. It is the only way to see what the fluid
    // costs behind a vsync, and it is free where the device has it.
    let mut sim = sim;
    sim.profile();

    let mut camera = PerspectiveCamera::new(40.0, width as f32 / height as f32, 0.05, 20.0);
    camera.position = Vector3::new(1.1, 0.18, 1.24);
    camera.look_at(Vector3::new(0.0, -0.02, 0.0));
    let mut controls = OrbitControls::new(&camera);
    controls.min_distance = 0.7;
    controls.max_distance = 6.0;

    Box::into_raw(Box::new(Fluid {
        gpu,
        surface,
        config,
        sim,
        camera,
        controls,
        elapsed: 0.0,
        frame_ms: 16.0,
    }))
}

/// What [`threers_fluid_set`] is setting.
///
/// One entry point rather than five: the five were five calling conventions
/// to get wrong across a language boundary, and a key with three floats is
/// one. It is the same shape the browser build has used all along —
/// `fluid.set(name, value)` — which is not a coincidence; a small ABI is
/// easier to keep honest than a wide one.
#[repr(u32)]
pub enum Key {
    /// The view resized. `a` = width, `b` = height, in pixels. From
    /// `layoutSubviews`, remembering `nativeScale`.
    Resize = 0,
    /// A finger moved. `a` = dx, `b` = dy in points, `c` non-zero while it
    /// is still touching. One finger orbits; there is no pan, because the
    /// volume is one object in the middle of the screen.
    Touch = 1,
    /// A pinch, `a` = scale since the last call (1.0 is no change).
    Pinch = 2,
    /// Add `a` puffs of dye at random places, up to 32. A tap, or a shake.
    Splat = 3,
    /// How fast the fluid runs, `a` = a multiple of real time. Zero holds it
    /// still; four is about as far as the advection stays believable.
    Speed = 4,
    /// Average the ray march over `a` frames, 1 to 32. One is off. It resets
    /// whenever the camera moves, so it costs nothing while a finger is down
    /// and sharpens the picture the moment it lifts.
    Accumulate = 5,
    /// Draw the frame rate over the picture, `a` non-zero for on. Drawn in
    /// the shader, so it needs no view of its own and no font.
    ShowFps = 6,
}

/// Set one thing. See [`Key`] for what `a`, `b` and `c` mean.
///
/// An unknown key does nothing, which is the right answer for an ABI that
/// will outlive one version of the app linking it.
///
/// # Safety
///
/// `handle` must have come from [`threers_fluid_create`] and not yet been
/// destroyed. Null is fine.
#[no_mangle]
pub unsafe extern "C" fn threers_fluid_set(handle: *mut Fluid, key: u32, a: f32, b: f32, c: f32) {
    let Some(f) = (unsafe { handle.as_mut() }) else {
        return;
    };
    let size = (f.config.width as f32, f.config.height as f32);
    match key {
        k if k == Key::Resize as u32 => {
            f.config.width = (a as u32).max(1);
            f.config.height = (b as u32).max(1);
            f.surface.configure(&f.gpu.device, &f.config);
            f.camera.aspect = f.config.width as f32 / f.config.height as f32;
        }
        k if k == Key::Touch as u32 => f.controls.update(
            PointerEvent {
                dx: a,
                dy: b,
                wheel: 0.0,
                rotating: c != 0.0,
                panning: false,
            },
            &mut f.camera,
            size,
        ),
        k if k == Key::Pinch as u32 => f.controls.update(
            PointerEvent {
                dx: 0.0,
                dy: 0.0,
                // A pinch out is a zoom in, and the control wants a wheel.
                wheel: (a - 1.0) * 4.0,
                rotating: false,
                panning: false,
            },
            &mut f.camera,
            size,
        ),
        k if k == Key::Splat as u32 => f.sim.multiple_splats((a.max(0.0) as u32).min(32)),
        k if k == Key::Speed as u32 => {
            let mut cfg = *f.sim.config();
            cfg.time_scale = a.clamp(0.0, 8.0);
            f.sim.set_config(cfg);
        }
        k if k == Key::ShowFps as u32 => {
            let mut cfg = *f.sim.config();
            cfg.show_fps = a != 0.0;
            f.sim.set_config(cfg);
        }
        k if k == Key::Accumulate as u32 => {
            let mut cfg = *f.sim.config();
            cfg.temporal_samples = (a.max(1.0) as u32).clamp(1, 32);
            f.sim.set_config(cfg);
        }
        _ => {}
    }
}

/// Step the fluid by `dt` seconds and draw it. Call once per display link.
///
/// Returns false if the drawable could not be acquired, which on iOS means
/// the view is off screen — the caller should stop the display link rather
/// than spin.
///
/// # Safety
///
/// As [`threers_fluid_resize`].
#[no_mangle]
pub unsafe extern "C" fn threers_fluid_frame(handle: *mut Fluid, dt: f32) -> bool {
    let Some(f) = (unsafe { handle.as_mut() }) else {
        return false;
    };
    let started = std::time::Instant::now();
    // A clamped step, not the measured one: a frame that took a quarter of a
    // second because the app was backgrounded should not be simulated as a
    // quarter of a second of fluid.
    let dt = dt.clamp(1.0 / 240.0, 1.0 / 20.0);
    f.elapsed += dt;

    // A plume that keeps feeding itself, so there is something to look at
    // without touching the screen.
    if f.elapsed > 0.08 {
        f.elapsed = 0.0;
        let a = started.elapsed().as_secs_f32() + f.frame_ms;
        f.sim.splat(
            [0.5 + 0.05 * a.cos(), 0.12, 0.5 + 0.05 * a.sin()],
            [0.0, 300.0, 0.0],
            [1.0, 0.45, 0.3],
        );
    }
    // `update`, not `step`: the speed knob is applied there.
    f.sim.update(dt);

    // Suboptimal is still presentable — it means the drawable no longer
    // matches the layer, which is the frame in the middle of a rotation.
    let frame = match f.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        _ => {
            f.surface.configure(&f.gpu.device, &f.config);
            return false;
        }
    };
    let view = frame.texture.create_view(&Default::default());
    let size = (f.config.width, f.config.height);
    f.sim
        .render_to_view(&f.camera, &view, f.config.format, size);

    f.gpu.queue.present(frame);

    let ms = started.elapsed().as_secs_f32() * 1000.0;
    f.frame_ms += (ms - f.frame_ms) * 0.1;
    true
}

/// What the fluid actually allocated, and how fast it is going.
///
/// # Safety
///
/// As [`threers_fluid_resize`]; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn threers_fluid_info(handle: *mut Fluid, out: *mut FluidInfo) {
    let (Some(f), false) = (unsafe { handle.as_ref() }, out.is_null()) else {
        return;
    };
    let (solver, dye) = f.sim.grids();
    unsafe {
        *out = FluidInfo {
            solver,
            dye,
            megabytes: (f.sim.memory_bytes() >> 20) as u32,
            fps: 1000.0 / f.frame_ms.max(0.01),
            solver_ms: f.sim.stage_ms().iter().sum(),
        };
    }
}

/// Free it. The handle is invalid afterwards.
///
/// # Safety
///
/// `handle` must have come from [`threers_fluid_create`] and must not be used
/// again. Passing null is fine.
#[no_mangle]
pub unsafe extern "C" fn threers_fluid_destroy(handle: *mut Fluid) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// Every entry point has to survive a null handle.
    ///
    /// The caller is an Objective-C runtime that will happily hand back a nil
    /// from a failed `create` and keep calling — so a crash here is a crash
    /// in someone's app, at a point a long way from the mistake.
    #[test]
    fn null_handles_do_nothing() {
        unsafe {
            for key in 0..6 {
                threers_fluid_set(std::ptr::null_mut(), key, 1.0, 1.0, 1.0);
            }
            assert!(!threers_fluid_frame(std::ptr::null_mut(), 1.0 / 60.0));
            threers_fluid_info(std::ptr::null_mut(), std::ptr::null_mut());
            threers_fluid_destroy(std::ptr::null_mut());
            assert_eq!(
                threers_fluid_create(std::ptr::null_mut(), 64, 64, 64),
                std::ptr::null_mut()
            );
        }
    }

    /// The whole path, on a real `CAMetalLayer`.
    ///
    /// Not a device, but the same class, the same wgpu surface and the same
    /// Metal drawables — which is everything between the C ABI and the
    /// screen. What this cannot tell you is how fast an A-series GPU is; what
    /// it can tell you is whether the thing an app links will come up, draw,
    /// resize and shut down.
    #[test]
    fn a_layer_comes_up_and_draws() {
        use objc2_core_foundation::CGSize;
        use objc2_quartz_core::CAMetalLayer;

        let layer = CAMetalLayer::new();
        layer.setDrawableSize(CGSize {
            width: 256.0,
            height: 256.0,
        });
        let ptr = objc2::rc::Retained::as_ptr(&layer) as *mut c_void;

        let fluid = unsafe { threers_fluid_create(ptr, 256, 256, 48) };
        if fluid.is_null() {
            eprintln!("skipping: no Metal adapter for a bare layer");
            return;
        }
        let mut info = FluidInfo {
            solver: 0,
            dye: 0,
            megabytes: 0,
            fps: 0.0,
            solver_ms: 0.0,
        };
        unsafe { threers_fluid_info(fluid, &mut info) };
        eprintln!(
            "  {}³ solver, {}³ dye, {} MB within a 48 MB budget",
            info.solver, info.dye, info.megabytes
        );
        assert!(
            info.solver >= 16,
            "the budget bought nothing: {}",
            info.solver
        );
        assert!(
            info.megabytes <= 48,
            "{} MB over a 48 MB budget",
            info.megabytes
        );

        unsafe {
            threers_fluid_set(fluid, Key::Splat as u32, 4.0, 0.0, 0.0);
            threers_fluid_set(fluid, Key::Touch as u32, 12.0, 4.0, 1.0);
            threers_fluid_set(fluid, Key::Pinch as u32, 1.2, 0.0, 0.0);
            // An unknown key is a no-op, not a crash.
            threers_fluid_set(fluid, 999, 1.0, 1.0, 1.0);
            let mut drawn = 0;
            for _ in 0..8 {
                if threers_fluid_frame(fluid, 1.0 / 60.0) {
                    drawn += 1;
                }
            }
            assert!(drawn > 0, "no frame reached the layer");
            threers_fluid_set(fluid, Key::Resize as u32, 320.0, 200.0, 0.0);
            assert!(
                threers_fluid_frame(fluid, 1.0 / 60.0),
                "resize broke the surface"
            );
            threers_fluid_info(fluid, &mut info);
            eprintln!("  drew {drawn} frames, {:.0} fps after a resize", info.fps);
            threers_fluid_destroy(fluid);
        }
    }
}

#[cfg(test)]
mod header {
    /// The hand-written header declares exactly what the library exports.
    ///
    /// A header is a second source of truth and they drift: a function added
    /// here and forgotten there is a link error, which is survivable, but a
    /// *signature* that drifts is a crash in someone's app with no symptom
    /// near the cause. Generating it would avoid that and cost a build step
    /// and a dependency; checking it costs this.
    #[test]
    fn the_header_matches_the_library() {
        let rust = include_str!("lib.rs");
        let header = include_str!("../include/threers_fluid.h");
        let exported: Vec<&str> = rust
            .lines()
            .filter_map(|l| l.trim().strip_prefix("pub "))
            .filter_map(|l| {
                let l = l.strip_prefix("unsafe ").unwrap_or(l);
                l.strip_prefix("extern \"C\" fn ")
            })
            .filter_map(|l| l.split('(').next())
            .collect();
        // Five: create, set, frame, info, destroy. If this trips because
        // the number fell, the scan broke; if it trips because the number
        // rose, ask whether the new entry point could have been a `Key`.
        assert_eq!(
            exported.len(),
            5,
            "the ABI is {} functions: {exported:?}",
            exported.len()
        );
        for name in &exported {
            assert!(
                header.contains(&format!("{name}(")),
                "`{name}` is exported but not declared in threers_fluid.h"
            );
        }
        // …and nothing declared that is not exported.
        for line in header.lines() {
            let Some(rest) = line.split_once("threers_fluid_") else {
                continue;
            };
            // Skip the prose and the include guard.
            if line.trim_start().starts_with("//") || line.contains("THREERS_FLUID_H") {
                continue;
            }
            let name: String = std::iter::once("threers_fluid_")
                .chain(std::iter::once(
                    rest.1.split(['(', ' ', ';']).next().unwrap_or(""),
                ))
                .collect();
            assert!(
                exported.contains(&name.as_str()),
                "threers_fluid.h declares `{name}`, which the library does not export"
            );
        }
    }
}
