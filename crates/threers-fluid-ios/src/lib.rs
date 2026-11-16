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

// A `CAMetalLayer` exists only on Apple's platforms, and so does Metal.
// Elsewhere the crate is empty, so the workspace still builds there.
#![cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]

use std::ffi::c_void;

use threers::controls::{OrbitControls, PointerEvent};
use threers::fluid::FluidConfig3d;
use threers::metal::objc::{get_sel, msg0, msg1, AutoreleasePool, Id, Owned};
use threers::metal::{MetalDevice, MetalFluid3d};
use threers::{PerspectiveCamera, Vector3};

/// `MTLPixelFormatBGRA8Unorm`, what the layer is set to: what a
/// `CAMetalLayer` is by default, and unsigned rather than sRGB: the march
/// writes display values.
const LAYER_FORMAT: u64 = 80;

/// `CGSize`, by value, for `setDrawableSize:`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CGSize {
    width: f64,
    height: f64,
}

/// Everything one view needs. Opaque to the caller.
pub struct Fluid {
    device: MetalDevice,
    /// The view's `CAMetalLayer`, retained for as long as the handle lives.
    layer: Owned,
    size: (u32, u32),
    sim: MetalFluid3d,
    camera: PerspectiveCamera,
    controls: OrbitControls,
    elapsed: f32,
    /// Smoothed whole-frame time, present included.
    frame_ms: f32,
    /// Where the stroke being painted was last, in uv; `None` between strokes.
    last_paint: Option<[f32; 2]>,
    /// The paint's colour, or `None` to walk the hue wheel stroke by stroke.
    paint_colour: Option<[f32; 3]>,
    pushes: u32,
    /// The plume that feeds itself, so there is something to watch untouched.
    ambient: bool,
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
    /// The ray march is not included: the step is a command buffer of its
    /// own, and this is that buffer's time on the GPU.
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
#[unsafe(no_mangle)]
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
    let _pool = AutoreleasePool::new();
    let Ok(device) = MetalDevice::new() else {
        return std::ptr::null_mut();
    };
    let Some(layer) = (unsafe { Owned::retain(layer.cast()) }) else {
        return std::ptr::null_mut();
    };
    unsafe {
        let _: () = msg1(layer.id(), get_sel("setDevice:\0"), device.device_id());
        let _: () = msg1(layer.id(), get_sel("setPixelFormat:\0"), LAYER_FORMAT);
        // The march writes the drawable and nothing samples it.
        let _: () = msg1(layer.id(), get_sel("setFramebufferOnly:\0"), true);
        let _: () = msg1(
            layer.id(),
            get_sel("setDrawableSize:\0"),
            CGSize {
                width: width as f64,
                height: height as f64,
            },
        );
    }

    let budget_mb = if budget_mb == 0 {
        SUGGESTED_BUDGET_MB
    } else {
        budget_mb.max(4)
    };
    let (solver, dye) = MetalFluid3d::within_memory((budget_mb as u64) << 20, 2);
    let sim = match MetalFluid3d::new(
        &device,
        FluidConfig3d {
            resolution: solver,
            dye_resolution: dye,
            buoyancy: 46.0,
            curl: 26.0,
            density_dissipation: 0.2,
            // A phone marches fewer samples than a desktop and the empty
            // space skip is what makes that affordable.
            march_steps: 64,
            ..FluidConfig3d::default()
        },
    ) {
        Ok(sim) => sim,
        Err(_) => return std::ptr::null_mut(),
    };

    let mut camera = PerspectiveCamera::new(40.0, width as f32 / height as f32, 0.05, 20.0);
    camera.position = Vector3::new(1.1, 0.18, 1.24);
    camera.look_at(Vector3::new(0.0, -0.02, 0.0));
    let mut controls = OrbitControls::new(&camera);
    controls.min_distance = 0.7;
    controls.max_distance = 6.0;

    Box::into_raw(Box::new(Fluid {
        device,
        layer,
        size: (width, height),
        sim,
        camera,
        controls,
        elapsed: 0.0,
        frame_ms: 16.0,
        last_paint: None,
        paint_colour: None,
        pushes: 0,
        ambient: true,
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
    /// Paint into the volume under a point: `a`, `b` in uv of the view (0 to
    /// 1, origin top left). `c` zero starts a stroke — a tap is a stroke of
    /// one — and non-zero continues it, pushing the dye the way the finger
    /// moved since the last call. The dye lands on the plane through the
    /// volume's centre facing the camera, so it stays under the finger
    /// however the camera has orbited.
    Paint = 7,
    /// The paint's colour, `a`, `b`, `c` linear RGB. All three zero walks the
    /// hue wheel instead, a new hue a stroke.
    PaintColor = 8,
    /// The plume that feeds itself, `a` non-zero for on (the default).
    Ambient = 9,
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid_set(handle: *mut Fluid, key: u32, a: f32, b: f32, c: f32) {
    let Some(f) = (unsafe { handle.as_mut() }) else {
        return;
    };
    let size = (f.size.0 as f32, f.size.1 as f32);
    match key {
        k if k == Key::Resize as u32 => {
            f.size = ((a as u32).max(1), (b as u32).max(1));
            let drawable = CGSize {
                width: f.size.0 as f64,
                height: f.size.1 as f64,
            };
            unsafe {
                let _: () = msg1(f.layer.id(), get_sel("setDrawableSize:\0"), drawable);
            }
            f.camera.aspect = f.size.0 as f32 / f.size.1 as f32;
        }
        k if k == Key::Touch as u32 => f.controls.update(
            PointerEvent {
                dx: a,
                dy: b,
                wheel: 0.0,
                rotating: c != 0.0,
                panning: false,
                cursor: (0.0, 0.0),
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
                // The pinch comes without a position: zoom on the centre.
                cursor: (0.0, 0.0),
            },
            &mut f.camera,
            size,
        ),
        k if k == Key::Splat as u32 => {
            let _ = f
                .sim
                .multiple_splats(&f.device, (a.max(0.0) as u32).min(32));
        }
        k if k == Key::Speed as u32 => {
            let mut cfg = *f.sim.config();
            cfg.time_scale = a.clamp(0.0, 8.0);
            let _ = f.sim.set_config(&f.device, cfg);
        }
        k if k == Key::ShowFps as u32 => {
            let mut cfg = *f.sim.config();
            cfg.show_fps = a != 0.0;
            let _ = f.sim.set_config(&f.device, cfg);
        }
        k if k == Key::Paint as u32 => {
            let at = [a.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
            let previous = if c != 0.0 { f.last_paint } else { None };
            if previous.is_none() {
                f.pushes = f.pushes.wrapping_add(1);
            }
            let drag = previous.map_or([0.0, 0.0], |p| [at[0] - p[0], at[1] - p[1]]);
            f.last_paint = Some(at);
            let colour = f.paint_colour.unwrap_or_else(|| {
                let c = hsv_to_rgb((f.pushes as f32 * 0.137).fract());
                [c[0] * 1.2, c[1] * 1.2, c[2] * 1.2]
            });
            let force = f.sim.config().splat_force;
            let _ = f
                .sim
                .impulse_at_screen(&f.device, &f.camera, at, drag, force, colour);
        }
        k if k == Key::PaintColor as u32 => {
            f.paint_colour = if a == 0.0 && b == 0.0 && c == 0.0 {
                None
            } else {
                Some([a, b, c])
            };
        }
        k if k == Key::Ambient as u32 => f.ambient = a != 0.0,
        k if k == Key::Accumulate as u32 => {
            let mut cfg = *f.sim.config();
            cfg.temporal_samples = (a.max(1.0) as u32).clamp(1, 32);
            let _ = f.sim.set_config(&f.device, cfg);
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
#[unsafe(no_mangle)]
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

    let _pool = AutoreleasePool::new();
    // A plume that keeps feeding itself, so there is something to look at
    // without touching the screen.
    if f.ambient && f.elapsed > 0.08 {
        f.elapsed = 0.0;
        let a = started.elapsed().as_secs_f32() + f.frame_ms;
        let _ = f.sim.splat(
            &f.device,
            [0.5 + 0.05 * a.cos(), 0.12, 0.5 + 0.05 * a.sin()],
            [0.0, 300.0, 0.0],
            [1.0, 0.45, 0.3],
        );
    }
    // `update`, not `step`: the speed knob is applied there. Its own command
    // buffer, so its GPU time is the solver's alone (`solver_ms`).
    if f.sim.update(&f.device, dt).is_err() {
        return false;
    }

    // Blocks until the display has a drawable free: with the layer's three,
    // that is the frame pacing. Null means the view is off screen.
    let drawable: Id = unsafe { msg0(f.layer.id(), get_sel("nextDrawable\0")) };
    if drawable.is_null() {
        return false;
    }
    let texture: Id = unsafe { msg0(drawable, get_sel("texture\0")) };
    let cmd = f.device.command_buffer();
    if f.sim
        .render_on(
            &f.device,
            cmd,
            &f.camera,
            texture,
            LAYER_FORMAT as _,
            f.size,
        )
        .is_err()
    {
        return false;
    }
    unsafe {
        let _: () = msg1(cmd, get_sel("presentDrawable:\0"), drawable);
        let _: () = msg0(cmd, get_sel("commit\0"));
    }

    let ms = started.elapsed().as_secs_f32() * 1000.0;
    f.frame_ms += (ms - f.frame_ms) * 0.1;
    true
}

/// What the fluid actually allocated, and how fast it is going.
///
/// # Safety
///
/// As [`threers_fluid_resize`]; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid_info(handle: *mut Fluid, out: *mut FluidInfo) {
    let (Some(f), false) = (unsafe { handle.as_mut() }, out.is_null()) else {
        return;
    };
    let (solver, dye) = f.sim.grids();
    unsafe {
        *out = FluidInfo {
            solver,
            dye,
            megabytes: (f.sim.memory_bytes() >> 20) as u32,
            fps: 1000.0 / f.frame_ms.max(0.01),
            solver_ms: f.sim.solver_ms(),
        };
    }
}

/// Free it. The handle is invalid afterwards.
///
/// # Safety
///
/// `handle` must have come from [`threers_fluid_create`] and must not be used
/// again. Passing null is fine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid_destroy(handle: *mut Fluid) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// A fully saturated colour at hue `h` (0 to 1).
fn hsv_to_rgb(h: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (q, t) = (1.0 - f, f);
    match (i as i32).rem_euclid(6) {
        0 => [1.0, t, 0.0],
        1 => [q, 1.0, 0.0],
        2 => [0.0, 1.0, t],
        3 => [0.0, q, 1.0],
        4 => [t, 0.0, 1.0],
        _ => [1.0, 0.0, q],
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
            for key in 0..10 {
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
    /// Not a device, but the same class and the same
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
            // A tap, then a stroke in a colour of its own, with the plume off.
            threers_fluid_set(fluid, Key::Ambient as u32, 0.0, 0.0, 0.0);
            threers_fluid_set(fluid, Key::Paint as u32, 0.5, 0.5, 0.0);
            threers_fluid_set(fluid, Key::PaintColor as u32, 0.2, 0.8, 1.0);
            threers_fluid_set(fluid, Key::Paint as u32, 0.4, 0.5, 0.0);
            threers_fluid_set(fluid, Key::Paint as u32, 0.45, 0.48, 1.0);
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
