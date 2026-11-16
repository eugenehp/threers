//! The threers 2D fluid as a C ABI, on Metal and nothing else.
//!
//! An app that already draws with Metal owns an `MTLDevice`, a command queue
//! and a frame. This hands it the fluid on *its* device, and leaves the frame
//! to it: either the fluid commits its own work on its own queue (the plain
//! calls), or everything goes into the app's command buffer (the `_on`
//! calls), in call order, and the fluid commits nothing and waits on nothing.
//!
//! The app can draw the fluid itself — [`threers_fluid2d_dye_texture`] is the
//! dye, `rgba16Float` — or have threers composite it with
//! [`threers_fluid2d_render_on`].
//!
//! ```swift
//! var cfg = ThreersFluid2dConfig()
//! threers_fluid2d_config_default(&cfg)
//! let fluid = threers_fluid2d_create(Unmanaged.passUnretained(device).toOpaque(),
//!                                    UInt32(size.width), UInt32(size.height), &cfg)
//!
//! // per frame, into the app's own buffer
//! let cmd = queue.makeCommandBuffer()!
//! let raw = Unmanaged.passUnretained(cmd).toOpaque()
//! splats.withUnsafeBufferPointer { threers_fluid2d_splats_on(fluid, raw, $0.baseAddress, UInt32($0.count)) }
//! threers_fluid2d_step_on(fluid, raw, dt)
//! // … sample threers_fluid2d_dye_texture(fluid) in the app's own pass …
//! cmd.commit()
//! ```
//!
//! Built with `metal-fluid`: no wgpu, no scene renderer. The header is
//! `include/threers_fluid2d.h`; `the_header_declares_every_export` holds it
//! to this file.

// Metal exists only on Apple's platforms. Elsewhere the crate is empty, so
// the workspace still builds there.
#![cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]

use std::ffi::c_void;

use threers::fluid::{Advection, FluidConfig, PressureSolver};
use threers::metal::{fluid_shader_source, FluidSplat, MetalDevice, MetalFieldFormats, MetalFluid};

/// Mirrors `ThreersFluid2dConfig` in the header. Fill it with
/// [`threers_fluid2d_config_default`] and change what you need: a field added
/// at the end in a later version is then set for you.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThreersFluid2dConfig {
    pub sim_resolution: u32,
    pub dye_resolution: u32,
    pub pressure_iterations: u32,
    pub density_dissipation: f32,
    pub velocity_dissipation: f32,
    pub pressure: f32,
    pub curl: f32,
    /// Variance of a splat from `threers_fluid2d_splat_random`, percent of the canvas.
    pub splat_radius: f32,
    /// Non-zero: threers' own shading of the dye in `render_on`.
    pub shading: u32,
    /// Non-zero: second-order (MacCormack) advection — sharper dye.
    pub maccormack: u32,
    /// Non-zero: a multigrid V-cycle instead of Jacobi sweeps.
    pub multigrid: u32,
    /// Non-zero: velocity and pressure in two and one channels. Fixed at
    /// creation; later changes to it are ignored.
    pub narrow_formats: u32,
}

impl Default for ThreersFluid2dConfig {
    fn default() -> Self {
        let c = FluidConfig::default();
        Self {
            sim_resolution: c.sim_resolution,
            dye_resolution: c.dye_resolution,
            pressure_iterations: c.pressure_iterations,
            density_dissipation: c.density_dissipation,
            velocity_dissipation: c.velocity_dissipation,
            pressure: c.pressure,
            curl: c.curl,
            splat_radius: c.splat_radius,
            shading: c.shading as u32,
            maccormack: (c.advection == Advection::MacCormack) as u32,
            multigrid: (c.pressure_solver == PressureSolver::Multigrid) as u32,
            narrow_formats: 0,
        }
    }
}

impl ThreersFluid2dConfig {
    fn fluid(&self) -> FluidConfig {
        FluidConfig {
            sim_resolution: self.sim_resolution.max(8),
            dye_resolution: self.dye_resolution.max(8),
            pressure_iterations: self.pressure_iterations.clamp(1, 64),
            density_dissipation: self.density_dissipation.max(0.0),
            velocity_dissipation: self.velocity_dissipation.max(0.0),
            pressure: self.pressure,
            curl: self.curl,
            splat_radius: self.splat_radius,
            shading: self.shading != 0,
            advection: if self.maccormack != 0 {
                Advection::MacCormack
            } else {
                Advection::SemiLagrangian
            },
            pressure_solver: if self.multigrid != 0 {
                PressureSolver::Multigrid
            } else {
                PressureSolver::Jacobi
            },
            // The app composites; threers' post passes are for `render_on`
            // callers who want the original's look, and cost a frame's
            // bandwidth for everyone else.
            bloom: false,
            sunrays: false,
            colorful: false,
            show_fps: false,
            time_scale: 1.0,
            ..FluidConfig::default()
        }
    }
}

/// Mirrors `ThreersFluid2dSplat` in the header.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThreersFluid2dSplat {
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub radius: f32,
    pub erase: u32,
}

/// One fluid. Opaque to the caller.
pub struct ThreersFluid2d {
    device: MetalDevice,
    fluid: MetalFluid,
    last_error: std::ffi::CString,
}

impl ThreersFluid2d {
    /// 0, or -1 with the message kept for [`threers_fluid2d_last_error`].
    fn status<E: std::fmt::Debug>(&mut self, r: Result<(), E>) -> i32 {
        match r {
            Ok(()) => 0,
            Err(e) => {
                self.last_error =
                    std::ffi::CString::new(format!("{e:?}").replace('\0', " ")).unwrap_or_default();
                -1
            }
        }
    }
}

fn splats(list: &[ThreersFluid2dSplat]) -> Vec<FluidSplat> {
    list.iter()
        .map(|s| FluidSplat {
            x: s.x,
            y: s.y,
            dx: s.dx,
            dy: s.dy,
            color: [s.r, s.g, s.b],
            radius: s.radius,
            erase: s.erase != 0,
        })
        .collect()
}

/// # Safety
/// `out` is null or points at a config to fill.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_config_default(out: *mut ThreersFluid2dConfig) {
    if let Some(out) = unsafe { out.as_mut() } {
        *out = ThreersFluid2dConfig::default();
    }
}

/// A fluid on `device` (an `id<MTLDevice>`), `width` × `height` pixels. Null
/// on failure. `cfg` may be null for the defaults.
///
/// # Safety
/// `device` is a live `MTLDevice`; `cfg` is null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_create(
    device: *mut c_void,
    width: u32,
    height: u32,
    cfg: *const ThreersFluid2dConfig,
) -> *mut ThreersFluid2d {
    if device.is_null() {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { cfg.as_ref() }.copied().unwrap_or_default();
    let Ok(device) =
        (unsafe { MetalDevice::adopt_with_shader_source(device.cast(), &fluid_shader_source()) })
    else {
        return std::ptr::null_mut();
    };
    let formats = if cfg.narrow_formats != 0 {
        MetalFieldFormats::Narrow
    } else {
        MetalFieldFormats::Uniform
    };
    match MetalFluid::with_formats(&device, width, height, cfg.fluid(), formats) {
        Ok(fluid) => Box::into_raw(Box::new(ThreersFluid2d {
            device,
            fluid,
            last_error: Default::default(),
        })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `h` is null or from `threers_fluid2d_create`; invalid afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_destroy(h: *mut ThreersFluid2d) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// The last failure's message, or an empty string. Valid until the next call
/// on `h`.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_last_error(
    h: *mut ThreersFluid2d,
) -> *const std::ffi::c_char {
    match unsafe { h.as_ref() } {
        Some(f) => f.last_error.as_ptr(),
        None => c"".as_ptr(),
    }
}

/// The `id<MTLCommandQueue>` the plain calls commit on. Borrowed.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_queue(h: *mut ThreersFluid2d) -> *mut c_void {
    unsafe { h.as_ref() }.map_or(std::ptr::null_mut(), |f| f.device.queue_id().cast())
}

/// The dye, an `id<MTLTexture>` (`rgba16Float`). Borrowed; a different
/// texture after every splat, step or clear.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_dye_texture(h: *mut ThreersFluid2d) -> *mut c_void {
    unsafe { h.as_ref() }.map_or(std::ptr::null_mut(), |f| f.fluid.dye_texture().cast())
}

/// New settings. The grids are rebuilt, the picture resampled into them,
/// only if a resolution changed.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`; `cfg` is valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_configure(
    h: *mut ThreersFluid2d,
    cfg: *const ThreersFluid2dConfig,
) -> i32 {
    let (Some(f), Some(c)) = (unsafe { h.as_mut() }, unsafe { cfg.as_ref() }) else {
        return -1;
    };
    let r = f.fluid.set_config(&f.device, c.fluid());
    f.status(r)
}

/// The canvas resized; the picture is resampled into the new grids.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_resize(
    h: *mut ThreersFluid2d,
    width: u32,
    height: u32,
) -> i32 {
    let Some(f) = (unsafe { h.as_mut() }) else {
        return -1;
    };
    let r = f.fluid.resize(&f.device, width, height);
    f.status(r)
}

/// The solver and dye grids as they came out, in texels. Any out pointer may
/// be null.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_grid_sizes(
    h: *mut ThreersFluid2d,
    sim_w: *mut u32,
    sim_h: *mut u32,
    dye_w: *mut u32,
    dye_h: *mut u32,
) {
    let Some(f) = (unsafe { h.as_ref() }) else {
        return;
    };
    let ((dw, dh), (sw, sh)) = f.fluid.grid_sizes();
    for (p, v) in [(sim_w, sw), (sim_h, sh), (dye_w, dw), (dye_h, dh)] {
        if let Some(p) = unsafe { p.as_mut() } {
            *p = v;
        }
    }
}

// -- committing on the fluid's queue ---------------------------------------

/// `count` splats.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`; `list` points at `count` splats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_splats(
    h: *mut ThreersFluid2d,
    list: *const ThreersFluid2dSplat,
    count: u32,
) -> i32 {
    let Some(f) = (unsafe { h.as_mut() }) else {
        return -1;
    };
    if list.is_null() || count == 0 {
        return 0;
    }
    let list = splats(unsafe { std::slice::from_raw_parts(list, count as usize) });
    let r = f.fluid.splat_batch(&f.device, &list);
    f.status(r)
}

/// `count` splats at random places and hues — a burst, for a tap.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_splat_random(h: *mut ThreersFluid2d, count: u32) {
    if let Some(f) = unsafe { h.as_mut() } {
        f.fluid.multiple_splats(&f.device, count.min(256));
    }
}

/// One solver step of `dt` seconds.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_step(h: *mut ThreersFluid2d, dt: f32) -> i32 {
    let Some(f) = (unsafe { h.as_mut() }) else {
        return -1;
    };
    let r = f.fluid.step(&f.device, dt);
    f.status(r)
}

/// Empty the canvas.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_clear(h: *mut ThreersFluid2d) -> i32 {
    let Some(f) = (unsafe { h.as_mut() }) else {
        return -1;
    };
    let r = f.fluid.clear(&f.device);
    f.status(r)
}

// -- into the caller's command buffer --------------------------------------

/// As `threers_fluid2d_splats`, encoded into `cmd`.
///
/// # Safety
/// `h` is from `threers_fluid2d_create`; `cmd` is an uncommitted
/// `id<MTLCommandBuffer>` on the fluid's device; `list` points at `count` splats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_splats_on(
    h: *mut ThreersFluid2d,
    cmd: *mut c_void,
    list: *const ThreersFluid2dSplat,
    count: u32,
) -> i32 {
    let Some(f) = (unsafe { h.as_mut() }) else {
        return -1;
    };
    if cmd.is_null() {
        return -1;
    }
    if list.is_null() || count == 0 {
        return 0;
    }
    let list = splats(unsafe { std::slice::from_raw_parts(list, count as usize) });
    let r = f.fluid.splat_batch_on(cmd.cast(), &list);
    f.status(r)
}

/// As `threers_fluid2d_step`, encoded into `cmd`.
///
/// # Safety
/// As `threers_fluid2d_splats_on`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_step_on(
    h: *mut ThreersFluid2d,
    cmd: *mut c_void,
    dt: f32,
) -> i32 {
    let Some(f) = (unsafe { h.as_mut() }) else {
        return -1;
    };
    if cmd.is_null() {
        return -1;
    }
    let r = f.fluid.step_on(cmd.cast(), dt);
    f.status(r)
}

/// As `threers_fluid2d_clear`, encoded into `cmd`.
///
/// # Safety
/// As `threers_fluid2d_splats_on`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_clear_on(h: *mut ThreersFluid2d, cmd: *mut c_void) -> i32 {
    let Some(f) = (unsafe { h.as_mut() }) else {
        return -1;
    };
    if cmd.is_null() {
        return -1;
    }
    let r = f.fluid.clear_on(cmd.cast());
    f.status(r)
}

/// threers' own composite of the dye into `target` (an `id<MTLTexture>` of
/// `pixel_format`, an `MTLPixelFormat`, `width` × `height`), encoded into `cmd`.
///
/// # Safety
/// As `threers_fluid2d_splats_on`; `target` is a live render-target texture.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_fluid2d_render_on(
    h: *mut ThreersFluid2d,
    cmd: *mut c_void,
    target: *mut c_void,
    pixel_format: u64,
    width: u32,
    height: u32,
) -> i32 {
    let Some(f) = (unsafe { h.as_mut() }) else {
        return -1;
    };
    if cmd.is_null() || target.is_null() {
        return -1;
    }
    let r = f.fluid.render_on(
        &f.device,
        cmd.cast(),
        target.cast(),
        pixel_format as _,
        (width, height),
    );
    f.status(r)
}

#[cfg(test)]
mod tests {
    /// Every `extern "C"` function here is declared in the header, and the
    /// header declares nothing that is not here.
    #[test]
    fn the_header_declares_every_export() {
        let src = include_str!("lib.rs");
        let header = include_str!("../include/threers_fluid2d.h");
        let names = |text: &str, marker: &str| -> Vec<String> {
            text.match_indices(marker)
                .map(|(i, _)| {
                    let rest = &text[i + marker.len()..];
                    rest[..rest.find('(').unwrap()]
                        .trim()
                        .trim_start_matches('*')
                        .to_string()
                })
                .collect()
        };
        let mut exported = names(src, "extern \"C\" fn ");
        exported.retain(|n| n.starts_with("threers_fluid2d_"));
        let declared: Vec<String> = header
            .lines()
            .filter(|l| {
                l.contains("threers_fluid2d_")
                    && l.contains('(')
                    && !l.trim_start().starts_with("//")
            })
            .map(|l| {
                let i = l.find("threers_fluid2d_").unwrap();
                l[i..l[i..].find('(').unwrap() + i].to_string()
            })
            .collect();
        let mut a = exported.clone();
        let mut b = declared.clone();
        a.sort();
        b.sort();
        a.dedup();
        b.dedup();
        assert_eq!(a, b, "header and library disagree");
        assert_eq!(
            a.len(),
            17,
            "a function was added or removed: update the header and this count"
        );
    }

    #[test]
    fn the_config_layout_matches_the_header() {
        // Twelve four-byte fields, no padding.
        assert_eq!(std::mem::size_of::<super::ThreersFluid2dConfig>(), 48);
        assert_eq!(std::mem::size_of::<super::ThreersFluid2dSplat>(), 36);
    }

    /// The whole C path on a real device: create, splat into a caller's
    /// buffer, step, read the dye texture back, clear, destroy.
    #[test]
    fn a_fluid_comes_up_paints_and_clears() {
        use super::*;
        unsafe extern "C" {
            fn MTLCreateSystemDefaultDevice() -> *mut c_void;
        }
        let dev = unsafe { MTLCreateSystemDefaultDevice() };
        if dev.is_null() {
            eprintln!("skipping: no Metal device");
            return;
        }
        unsafe {
            let h = threers_fluid2d_create(dev, 256, 256, std::ptr::null());
            assert!(!h.is_null());
            let splat = ThreersFluid2dSplat {
                x: 0.5,
                y: 0.5,
                dx: 100.0,
                r: 1.0,
                g: 0.5,
                b: 0.2,
                radius: 0.003,
                ..Default::default()
            };
            assert_eq!(threers_fluid2d_splats(h, &splat, 1), 0);
            assert_eq!(threers_fluid2d_step(h, 1.0 / 60.0), 0);
            assert!(!threers_fluid2d_dye_texture(h).is_null());
            assert!(!threers_fluid2d_queue(h).is_null());
            let (mut sw, mut sh) = (0, 0);
            threers_fluid2d_grid_sizes(
                h,
                &mut sw,
                &mut sh,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            assert_eq!((sw, sh), (128, 128));
            assert_eq!(threers_fluid2d_clear(h), 0);
            assert_eq!(
                threers_fluid2d_step_on(h, std::ptr::null_mut(), 0.1),
                -1,
                "a null buffer is refused"
            );
            threers_fluid2d_destroy(h);
        }
    }
}
