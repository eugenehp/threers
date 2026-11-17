//! Candles on a table, as a C ABI: threers' [`Candle`]s — wax, wick, a
//! flame, the light it throws — in a dark room, drawn by the Metal scene
//! renderer into a view's `CAMetalLayer`.
//!
//! The app owns what the candles *mean*: where they stand, what colour each
//! burns. This draws them, moves them, recolours them, and answers the two
//! questions a touch asks — where on the table is this point of the screen,
//! and which candle is under it.
//!
//! ```swift
//! let candles = threers_candles_create(Unmanaged.passUnretained(layer).toOpaque(),
//!                                      UInt32(w), UInt32(h), 1)   // 1: realistic flames
//! var x: Float = 0, z: Float = 0
//! if threers_candles_table_point(candles, u, v, &x, &z) != 0 {
//!     let id = threers_candles_add(candles, x, z)
//!     threers_candles_set_color(candles, id, 0.3, 0.6, 1.0, 1.0)
//! }
//! // From a CADisplayLink:
//! threers_candles_frame(candles, dt)
//! ```
//!
//! The header is `include/threers_candles.h`; `the_header_declares_every_export`
//! holds it to this file.

#![cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]

use std::ffi::c_void;

use threers::metal::objc::Id;
use threers::prelude::Color;

mod fire;
mod table;

pub use table::ThreersCandles;

/// Fill a pair of out pointers (either may be null); 1, for "filled".
///
/// # Safety
/// Each pointer is null or valid for a write.
unsafe fn write_pair(a: *mut f32, b: *mut f32, (x, y): (f32, f32)) -> u32 {
    unsafe {
        if let Some(a) = a.as_mut() {
            *a = x;
        }
        if let Some(b) = b.as_mut() {
            *b = y;
        }
    }
    1
}

// -- the table ----------------------------------------------------------------

/// A table drawing into `layer` (a `CAMetalLayer`), `width` × `height`
/// pixels. `realistic` non-zero gives the volumetric flame, zero the cheap
/// one — for many candles on an older device. Null on failure.
///
/// # Safety
/// `layer` is a live `CAMetalLayer` that outlives the handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_create(
    layer: *mut c_void,
    width: u32,
    height: u32,
    realistic: u32,
) -> *mut ThreersCandles {
    match unsafe { ThreersCandles::new(layer, width, height, realistic != 0) } {
        Some(table) => Box::into_raw(Box::new(table)),
        None => std::ptr::null_mut(),
    }
}

/// # Safety
/// `h` is null or from `threers_candles_create`; invalid afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_destroy(h: *mut ThreersCandles) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// The view resized, in pixels.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_resize(h: *mut ThreersCandles, width: u32, height: u32) {
    if let Some(c) = unsafe { h.as_mut() } {
        c.resize(width, height);
    }
}

// -- candles --------------------------------------------------------------------

/// A candle at `x`, `z` on the table (metres from its centre). Its id, or 0
/// when the point is off the table.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_add(h: *mut ThreersCandles, x: f32, z: f32) -> u32 {
    unsafe { h.as_mut() }.and_then(|c| c.add(x, z)).unwrap_or(0)
}

/// Move candle `id` to `x`, `z`, kept on the table.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_move(h: *mut ThreersCandles, id: u32, x: f32, z: f32) {
    if let Some(c) = unsafe { h.as_mut() } {
        c.move_to(id, x, z);
    }
}

/// Take candle `id` off the table.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_remove(h: *mut ThreersCandles, id: u32) {
    if let Some(c) = unsafe { h.as_mut() } {
        c.remove(id);
    }
}

/// Candles on the table.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_count(h: *mut ThreersCandles) -> u32 {
    unsafe { h.as_ref() }.map_or(0, |c| c.count() as u32)
}

/// Candle `id` burns in `r`, `g`, `b` (linear), `amount` of the way from its
/// own colour: the flame, the glow in the wax and its light.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_set_color(
    h: *mut ThreersCandles,
    id: u32,
    r: f32,
    g: f32,
    b: f32,
    amount: f32,
) {
    if let Some(c) = unsafe { h.as_mut() } {
        c.set_color(id, Color::new(r, g, b), amount);
    }
}

/// `bodies` non-zero: flames on candles (wax, wick, the wax's glow), the
/// default; zero: the flames alone, burning just above the table. Rebuilds
/// the candles already placed, keeping their ids and colours.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_set_bodies(h: *mut ThreersCandles, bodies: u32) {
    if let Some(c) = unsafe { h.as_mut() } {
        c.set_bodies(bodies != 0);
    }
}

/// `fire` non-zero: the flames burn as one volume of gas — threers' 3D fluid,
/// fed at every flame's foot with its colour, rising, curling and fading — in
/// place of each flame's own mesh; flames that stand close feed the same gas,
/// so their plumes run together and their colours mix. Zero: each flame its
/// own mesh again (the default). 1 when the fire burns, 0 when it is off or
/// could not be made.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_set_fire(h: *mut ThreersCandles, fire: u32) -> u32 {
    unsafe { h.as_mut() }.is_some_and(|c| c.set_fire(fire != 0)) as u32
}

/// How much fire, `amount` 0–1: how bright the gas burns and how high it
/// rises. Kept while the fire is off, for when it comes back.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_set_fire_amount(h: *mut ThreersCandles, amount: f32) {
    if let Some(c) = unsafe { h.as_mut() } {
        c.set_fire_amount(amount);
    }
}

/// The sweep at `turns` around the table (0 the far side, clockwise seen
/// from above, as the app's player counts), `strength` 0–1; 0 hides it.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_set_sweep(
    h: *mut ThreersCandles,
    turns: f32,
    strength: f32,
) {
    if let Some(c) = unsafe { h.as_mut() } {
        c.set_sweep(turns, strength);
    }
}

/// Candle `id` flares, `amount` 0–1: thrown up brighter for a moment, as
/// it is played. Shows in the fire.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_flare(h: *mut ThreersCandles, id: u32, amount: f32) {
    if let Some(c) = unsafe { h.as_mut() } {
        c.flare(id, amount);
    }
}

// -- the screen and the table ---------------------------------------------------

/// Where on the table the screen point `u`, `v` (0–1, origin top left)
/// looks: 1 and `x`, `z` filled when it is on the table, else 0.
///
/// # Safety
/// `h` is from `threers_candles_create`; the out pointers are valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_table_point(
    h: *mut ThreersCandles,
    u: f32,
    v: f32,
    x: *mut f32,
    z: *mut f32,
) -> u32 {
    match unsafe { h.as_ref() }.and_then(|c| c.table_point(u, v)) {
        Some(p) => unsafe { write_pair(x, z, p) },
        None => 0,
    }
}

/// The candle under the screen point `u`, `v`, or 0: the nearest whose foot
/// is within a few centimetres of where the point meets the table.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_pick(h: *mut ThreersCandles, u: f32, v: f32) -> u32 {
    unsafe { h.as_ref() }
        .and_then(|c| c.pick(u, v))
        .unwrap_or(0)
}

/// Where candle `id`'s flame is on the screen, in uv (0–1, origin top left),
/// for drawing labels over it. 0 if there is no such candle.
///
/// # Safety
/// `h` is from `threers_candles_create`; the out pointers are valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_screen_point(
    h: *mut ThreersCandles,
    id: u32,
    u: *mut f32,
    v: *mut f32,
) -> u32 {
    match unsafe { h.as_ref() }.and_then(|c| c.flame_on_screen(id)) {
        Some(p) => unsafe { write_pair(u, v, p) },
        None => 0,
    }
}

/// Where the table point `x`, `y`, `z` (metres) is on screen: 1 and `u`,
/// `v` (0–1, origin top left), or 0 behind the camera.
///
/// # Safety
/// `h` is from `threers_candles_create`; the out pointers are valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_project(
    h: *mut ThreersCandles,
    x: f32,
    y: f32,
    z: f32,
    u: *mut f32,
    v: *mut f32,
) -> u32 {
    match unsafe { h.as_ref() }.and_then(|c| c.project(x, y, z)) {
        Some(p) => unsafe { write_pair(u, v, p) },
        None => 0,
    }
}

/// Turn the view: `dx`, `dy` a drag in points.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_orbit(h: *mut ThreersCandles, dx: f32, dy: f32) {
    if let Some(c) = unsafe { h.as_mut() } {
        c.orbit(dx, dy, 0.0);
    }
}

/// Zoom: `scale` since the last call, as a pinch reports it (1 is none).
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_zoom(h: *mut ThreersCandles, scale: f32) {
    if let Some(c) = unsafe { h.as_mut() } {
        c.orbit(0.0, 0.0, (scale - 1.0) * 4.0);
    }
}

// -- the frame ------------------------------------------------------------------

/// Flicker by `dt` seconds and draw. False when no drawable came back —
/// the view is off screen; stop the display link.
///
/// # Safety
/// `h` is from `threers_candles_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_frame(h: *mut ThreersCandles, dt: f32) -> bool {
    unsafe { h.as_mut() }.is_some_and(|c| c.frame(dt, None).is_some())
}

/// As [`threers_candles_frame`], and a copy of the frame into `texture` (an
/// `id<MTLTexture>` the size of the view, in the layer's pixel format —
/// `bgra8Unorm_srgb` — on the system default device): a movie's frame. The
/// layer must not be `framebufferOnly`. Returns the copy's
/// `id<MTLCommandBuffer>`, retained (+1) and *not committed*: add a
/// completion handler — the texture is written when it runs — then commit
/// it, which the caller must do, and release it. Null when nothing was drawn
/// (off screen; nothing to commit).
///
/// # Safety
/// `h` is from `threers_candles_create`; `texture` is a live `MTLTexture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_candles_frame_record(
    h: *mut ThreersCandles,
    dt: f32,
    texture: *mut c_void,
) -> *mut c_void {
    let Some(c) = (unsafe { h.as_mut() }) else {
        return std::ptr::null_mut();
    };
    if texture.is_null() {
        return std::ptr::null_mut();
    }
    match c.frame(dt, Some(texture as Id)) {
        Some(Some(copy)) => {
            let id = copy.id();
            // The +1 goes to the caller.
            std::mem::forget(copy);
            id as *mut c_void
        }
        _ => std::ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::{room, sweep};
    use threers::prelude::*;

    #[test]
    fn the_header_declares_every_export() {
        let src = include_str!("lib.rs");
        let header = include_str!("../include/threers_candles.h");
        let mut exported: Vec<String> = src
            .match_indices("extern \"C\" fn ")
            .map(|(i, m)| {
                let rest = &src[i + m.len()..];
                rest[..rest.find('(').unwrap()].trim().to_string()
            })
            .filter(|n| n.starts_with("threers_candles_"))
            .collect();
        let mut declared: Vec<String> = header
            .lines()
            .filter(|l| {
                l.contains("threers_candles_")
                    && l.contains('(')
                    && !l.trim_start().starts_with("//")
            })
            .map(|l| {
                let i = l.find("threers_candles_").unwrap();
                l[i..i + l[i..].find('(').unwrap()].to_string()
            })
            .collect();
        exported.sort();
        declared.sort();
        exported.dedup();
        declared.dedup();
        assert_eq!(exported, declared);
        assert_eq!(exported.len(), 21);
    }

    /// A recorded frame: the copy's command buffer comes back uncommitted,
    /// runs once committed, and the texture holds the table — not black.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_frame_records_into_a_texture() {
        use objc2::msg_send;
        use objc2::runtime::{AnyClass, AnyObject};
        use objc2_core_foundation::CGSize;
        use objc2_quartz_core::CAMetalLayer;
        let (w, h) = (160usize, 120usize);
        let layer = CAMetalLayer::new();
        layer.setDrawableSize(CGSize {
            width: w as f64,
            height: h as f64,
        });
        layer.setFramebufferOnly(false);
        let ptr = objc2::rc::Retained::as_ptr(&layer) as *mut c_void;
        unsafe {
            let t = threers_candles_create(ptr, w as u32, h as u32, 1);
            if t.is_null() {
                eprintln!("skipping: no Metal device");
                return;
            }
            // A shared BGRA sRGB texture the CPU can read back.
            let device = (*t).renderer.device().device_id() as *mut AnyObject;
            let class = AnyClass::get(c"MTLTextureDescriptor").unwrap();
            let desc: *mut AnyObject = msg_send![class, texture2DDescriptorWithPixelFormat: 81usize, width: w, height: h, mipmapped: false];
            let _: () = msg_send![desc, setStorageMode: 0usize];
            let texture: *mut AnyObject = msg_send![device, newTextureWithDescriptor: desc];
            assert!(!texture.is_null());
            let mut cmd: *mut AnyObject = std::ptr::null_mut();
            for _ in 0..3 {
                let next = threers_candles_frame_record(t, 1.0 / 60.0, texture as *mut c_void);
                assert!(!next.is_null(), "a frame was drawn and copied");
                let next = next as *mut AnyObject;
                let _: () = msg_send![next, commit];
                if !cmd.is_null() {
                    let _: () = msg_send![cmd, release];
                }
                cmd = next;
            }
            let _: () = msg_send![cmd, waitUntilCompleted];
            let status: usize = msg_send![cmd, status];
            assert_eq!(status, 4, "the copy completed");
            let _: () = msg_send![cmd, release];
            #[repr(C)]
            struct Region([usize; 6]);
            unsafe impl objc2::Encode for Region {
                const ENCODING: objc2::Encoding = objc2::Encoding::Struct(
                    "?",
                    &[
                        objc2::Encoding::Struct(
                            "?",
                            &[usize::ENCODING, usize::ENCODING, usize::ENCODING],
                        ),
                        objc2::Encoding::Struct(
                            "?",
                            &[usize::ENCODING, usize::ENCODING, usize::ENCODING],
                        ),
                    ],
                );
            }
            let mut px = vec![0u8; w * h * 4];
            let region = Region([0, 0, 0, w, h, 1]);
            let _: () = msg_send![texture, getBytes: px.as_mut_ptr() as *mut c_void, bytesPerRow: w * 4,
                                  fromRegion: region, mipmapLevel: 0usize];
            assert!(
                px.chunks(4).any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0),
                "the frame is in the texture"
            );
            let _: () = msg_send![texture, release];
            threers_candles_set_fire_amount(t, 0.3);
            assert_eq!(threers_candles_set_fire(t, 1), 1);
            assert!(threers_candles_frame(t, 1.0 / 60.0));
            threers_candles_destroy(t);
        }
    }

    /// The sweep's shader draws on Metal: its hand bright on the table, the
    /// table across from it dark.
    #[test]
    fn the_sweep_draws_its_hand() {
        use threers::metal::MetalHeadlessRenderer;
        let size = 256;
        let Ok(mut metal) = MetalHeadlessRenderer::builder().size(size, size).build() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut scene = Scene::new();
        room(&mut scene);
        let (id, material) = sweep(&mut scene);
        scene.get_mut(id).unwrap().visible = true;
        // The hand a quarter turn round: pointing right (+x).
        material.set_uniform(0, [0.25, 1.0, 0.0, 0.0]);
        let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.02, 20.0);
        camera.position = Vector3::new(0.0, 1.6, 0.001);
        camera.look_at(Vector3::new(0.0, 0.0, 0.0));
        let rgba = metal.render_to_rgba(&mut scene, &camera).unwrap();
        if std::env::var_os("CANDLE_PNG").is_some() {
            std::fs::create_dir_all("out").unwrap();
            std::fs::write(
                "out/candle_sweep.png",
                threers::encode_png(size, size, &rgba),
            )
            .unwrap();
        }
        let at = |x: f32, z: f32| {
            let (u, v) = {
                let p = Vector3::new(x, 0.0, z)
                    .project(&camera.view_matrix(), &camera.projection_matrix());
                ((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5)
            };
            let i = ((v * size as f32) as usize * size as usize + (u * size as f32) as usize) * 4;
            rgba[i] as u32 + rgba[i + 1] as u32 + rgba[i + 2] as u32
        };
        let (hand, across) = (at(0.35, 0.0), at(-0.35, 0.0));
        assert!(
            hand > across + 60,
            "the hand shows: {hand} against {across}"
        );
    }

    /// The fire: flames that stand close run into one plume of both colours,
    /// a flame far off burns alone in its own.
    #[cfg(target_os = "macos")]
    #[test]
    fn close_flames_mix_in_the_fire() {
        use objc2_core_foundation::CGSize;
        use objc2_quartz_core::CAMetalLayer;
        let (w, h) = (320u32, 240u32);
        let layer = CAMetalLayer::new();
        layer.setDrawableSize(CGSize {
            width: w as f64,
            height: h as f64,
        });
        let ptr = objc2::rc::Retained::as_ptr(&layer) as *mut c_void;
        unsafe {
            let t = threers_candles_create(ptr, w, h, 1);
            if t.is_null() {
                eprintln!("skipping: no Metal device");
                return;
            }
            assert_eq!(threers_candles_set_fire(t, 1), 1, "the fire comes up");
            let red = threers_candles_add(t, -0.025, 0.0);
            let blue = threers_candles_add(t, 0.025, 0.0);
            let green = threers_candles_add(t, 0.35, 0.0);
            threers_candles_set_color(t, red, 1.0, 0.0, 0.0, 1.0);
            threers_candles_set_color(t, blue, 0.0, 0.0, 1.0, 1.0);
            threers_candles_set_color(t, green, 0.0, 1.0, 0.0, 1.0);
            for _ in 0..90 {
                assert!(threers_candles_frame(t, 1.0 / 60.0));
            }
            let c = &mut *t;
            let rgba = {
                let fire = c.fire.as_mut().unwrap();
                fire.sim
                    .capture_rgba(&fire.device.clone(), &c.camera, w, h)
                    .unwrap()
            };
            if std::env::var_os("CANDLE_PNG").is_some() {
                std::fs::create_dir_all("out").unwrap();
                // Over black, as on the table at night.
                let shown: Vec<u8> = rgba
                    .chunks(4)
                    .flat_map(|p| [p[0], p[1], p[2], 255])
                    .collect();
                std::fs::write("out/candle_fire.png", threers::encode_png(w, h, &shown)).unwrap();
                // And from the side, at the table's height.
                let mut side = PerspectiveCamera::new(38.0, w as f32 / h as f32, 0.02, 20.0);
                side.position = Vector3::new(0.1, 0.12, 0.8);
                side.look_at(Vector3::new(0.1, 0.12, 0.0));
                let fire = c.fire.as_mut().unwrap();
                let rgba = fire
                    .sim
                    .capture_rgba(&fire.device.clone(), &side, w, h)
                    .unwrap();
                let shown: Vec<u8> = rgba
                    .chunks(4)
                    .flat_map(|p| [p[0], p[1], p[2], 255])
                    .collect();
                std::fs::write(
                    "out/candle_fire_side.png",
                    threers::encode_png(w, h, &shown),
                )
                .unwrap();
            }
            // Summed over a box of the screen around where a point in the
            // air projects.
            let around = |x: f32, y: f32, z: f32| {
                let (u, v) = c.screen_point(x, y, z);
                let (px, py) = ((u * w as f32) as i32, (v * h as f32) as i32);
                let mut sum = [0u32; 3];
                for dy in -14..=4 {
                    for dx in -8..=8 {
                        let (x, y) = (px + dx, py + dy);
                        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                            continue;
                        }
                        let i = (y as usize * w as usize + x as usize) * 4;
                        for k in 0..3 {
                            sum[k] += rgba[i + k] as u32;
                        }
                    }
                }
                sum
            };
            let pair = around(0.0, 0.07, 0.0);
            let lone = around(0.35, 0.07, 0.0);
            assert!(
                pair[0] > 2000 && pair[2] > 2000,
                "the pair's plume holds both colours: {pair:?}"
            );
            assert!(lone[1] > 2000, "the lone flame burns: {lone:?}");
            assert!(
                lone[0] < lone[1] / 4 && lone[2] < lone[1] / 4,
                "in its own colour alone: {lone:?}"
            );
            assert_eq!(threers_candles_set_fire(t, 0), 0);
            assert!(threers_candles_frame(t, 1.0 / 60.0));
            threers_candles_destroy(t);
        }
    }

    /// The whole path on a real `CAMetalLayer`: candles placed through the
    /// screen, picked, moved, recoloured, removed, drawn.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_table_comes_up_and_takes_candles() {
        use objc2_core_foundation::CGSize;
        use objc2_quartz_core::CAMetalLayer;
        let layer = CAMetalLayer::new();
        layer.setDrawableSize(CGSize {
            width: 320.0,
            height: 240.0,
        });
        let ptr = objc2::rc::Retained::as_ptr(&layer) as *mut c_void;
        unsafe {
            let h = threers_candles_create(ptr, 320, 240, 1);
            if h.is_null() {
                eprintln!("skipping: no Metal device");
                return;
            }
            let (mut x, mut z) = (0.0, 0.0);
            assert_eq!(
                threers_candles_table_point(h, 0.5, 0.6, &mut x, &mut z),
                1,
                "the middle is the table"
            );
            let id = threers_candles_add(h, x, z);
            assert_ne!(id, 0);
            assert_eq!(
                threers_candles_pick(h, 0.5, 0.6),
                id,
                "the candle is under the point it was put at"
            );
            let (mut u, mut v) = (0.0, 0.0);
            assert_eq!(threers_candles_screen_point(h, id, &mut u, &mut v), 1);
            assert!(
                (u - 0.5).abs() < 0.05 && v < 0.6,
                "its flame is above its foot: {u}, {v}"
            );
            let second = threers_candles_add(h, x + 0.2, z);
            assert_ne!(second, id);
            threers_candles_set_color(h, id, 0.2, 0.5, 1.0, 1.0);
            threers_candles_move(h, second, 5.0, 0.0);
            assert_eq!(threers_candles_add(h, 5.0, 0.0), 0, "off the table");
            // The sweep shows and hides; a flare is taken; points project.
            threers_candles_set_sweep(h, 0.25, 1.0);
            threers_candles_flare(h, id, 1.0);
            assert!(threers_candles_frame(h, 1.0 / 60.0));
            threers_candles_set_sweep(h, 0.0, 0.0);
            let (mut pu, mut pv) = (0.0, 0.0);
            assert_eq!(
                threers_candles_project(h, 0.0, 0.0, 0.0, &mut pu, &mut pv),
                1
            );
            assert!(
                (pu - 0.5).abs() < 0.02,
                "the table's middle is in the middle: {pu}"
            );
            threers_candles_orbit(h, 20.0, 5.0);
            threers_candles_zoom(h, 1.1);
            let mut drawn = 0;
            for _ in 0..6 {
                if threers_candles_frame(h, 1.0 / 60.0) {
                    drawn += 1;
                }
            }
            assert!(drawn > 0, "no frame reached the layer");
            // The flames alone: same candles, flames lower.
            let (mut u1, mut v1) = (0.0, 0.0);
            threers_candles_screen_point(h, id, &mut u1, &mut v1);
            threers_candles_set_bodies(h, 0);
            assert_eq!(threers_candles_count(h), 2);
            assert_eq!(threers_candles_pick(h, 0.5, 0.6), id, "kept its id");
            let (mut u2, mut v2) = (0.0, 0.0);
            assert_eq!(threers_candles_screen_point(h, id, &mut u2, &mut v2), 1);
            assert!(
                v2 > v1,
                "without its candle the flame sits lower: {v2} vs {v1}"
            );
            assert!(threers_candles_frame(h, 1.0 / 60.0));
            threers_candles_remove(h, id);
            assert_eq!(threers_candles_count(h), 1);
            assert_eq!(threers_candles_pick(h, 0.5, 0.6), 0, "removed");
            threers_candles_resize(h, 400, 300);
            assert!(threers_candles_frame(h, 1.0 / 60.0));
            threers_candles_destroy(h);
        }
    }
}
