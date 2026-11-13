//! threers apps on iPhone, iPad and Apple Vision Pro, written in Rust and
//! drawn with Metal: one [`App`] in five places, with a Swift shell that
//! is the same for every app (`app/Harness.swift`) and a build script that
//! needs no Xcode project (`app/build.sh`).
//!
//! - iOS, **3D** — a window with an orbit camera ([`View`], [`Mode::Window`]);
//! - iOS, **AR** — over the camera, ARKit run from Rust (`ar`, [`Mode::Ar`]);
//! - visionOS, **window** — the orbit view in a window;
//! - visionOS, **volume** — RealityKit draws the scene in the shared space,
//!   beside other apps, no immersion ([`volume`]);
//! - visionOS, **immersive** — both eyes in the room (`immersive`).
//!
//! Every mode draws the scene with threers' own Metal renderer
//! ([`MetalRenderer`](threers::metal::MetalRenderer)): no wgpu, no shader
//! translation at run time. The app's UI is threers-ui ([`App::ui`]),
//! painted on the CPU where it changed and blended over the frame, put on a
//! panel in the room, or handed to the shell as an image beside a volume.
//!
//! The app switches between the modes itself — from its UI, say — with
//! [`request`]: the window and AR are this crate's to switch, the immersive
//! space the shell's, which polls for it. It moves the window's camera with
//! [`camera`].
//!
//! ```ignore
//! struct MyApp { /* … */ }
//! impl threers_ios::App for MyApp { /* setup, update, ui, ui_event */ }
//! threers_ios::app!(MyApp);
//! ```
//!
//! ```sh
//! path/to/threers-ios/app/build.sh path/to/my-app            # iPhone simulator
//! path/to/threers-ios/app/build.sh path/to/my-app vision     # Vision Pro simulator
//! ```

#![cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "visionos"))]

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

pub use threers;

pub mod app;
pub mod bake;
#[cfg(all(target_os = "ios", target_arch = "aarch64"))]
pub mod ar;
#[cfg(target_os = "visionos")]
pub mod immersive;
mod surface_ui;
pub mod view;
pub mod volume;

pub use app::{App, PanelSpec, Place, Surface};
pub use view::{Key, Mode, View};

/// What an app can ask for while it runs.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Request {
    /// The flat view, with its orbit camera.
    Window = 1,
    /// Over the camera (iOS: where [`ar_supported`]).
    Ar = 2,
    /// Into the room: the shell opens the immersive space (visionOS).
    EnterImmersive = 3,
    /// Out of it.
    ExitImmersive = 4,
}

/// What an app can ask of the window's camera ([`camera`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraRequest {
    /// Move in (below 1) or out (above 1): the distance times this.
    Zoom(f32),
    /// Back to framing the app's focus, as at the start.
    Home,
    /// Turn about the focus on its own, or stop.
    Spin(bool),
}

/// For the view: window or AR.
static VIEW_REQUEST: AtomicU32 = AtomicU32::new(0);
/// For the shell: the immersive space.
static SHELL_REQUEST: AtomicU32 = AtomicU32::new(0);
static IMMERSIVE: AtomicBool = AtomicBool::new(false);
static MODE: AtomicU32 = AtomicU32::new(Mode::Window as u32);
static SPINNING: AtomicBool = AtomicBool::new(false);
static CAMERA: Mutex<Vec<CameraRequest>> = Mutex::new(Vec::new());
static TAPS: Mutex<Vec<(f32, f32)>> = Mutex::new(Vec::new());
static PRESSES: Mutex<Vec<u64>> = Mutex::new(Vec::new());
static TOUCHES: Mutex<Vec<(u32, u32, f32, f32, std::time::Instant)>> = Mutex::new(Vec::new());
static VOLUME_SHOWN: AtomicBool = AtomicBool::new(false);
static TEXTS: Mutex<Vec<(u64, String)>> = Mutex::new(Vec::new());

/// Ask for a mode; it happens at the next frame (the view's), or when the
/// shell next polls (the immersive space). The last request wins.
pub fn request(r: Request) {
    match r {
        Request::Window | Request::Ar => VIEW_REQUEST.store(r as u32, Ordering::Relaxed),
        Request::EnterImmersive | Request::ExitImmersive => SHELL_REQUEST.store(r as u32, Ordering::Relaxed),
    }
}

pub(crate) fn take_view_request() -> Option<Request> {
    match VIEW_REQUEST.swap(0, Ordering::Relaxed) {
        1 => Some(Request::Window),
        2 => Some(Request::Ar),
        _ => None,
    }
}

/// Move the window's camera; it happens at the next frame. In AR and the
/// room there is no such camera, and this does nothing.
pub fn camera(c: CameraRequest) {
    if let CameraRequest::Spin(on) = c {
        SPINNING.store(on, Ordering::Relaxed);
    }
    if let Ok(mut q) = CAMERA.lock() {
        q.push(c);
    }
}

pub(crate) fn take_camera_requests() -> Vec<CameraRequest> {
    CAMERA.lock().map(|mut q| std::mem::take(&mut *q)).unwrap_or_default()
}

/// Whether the window's camera is turning on its own ([`CameraRequest::Spin`]).
pub fn spinning() -> bool {
    SPINNING.load(Ordering::Relaxed)
}

/// Whether this device can show the app in AR over its camera.
pub fn ar_supported() -> bool {
    #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
    return ar::supported();
    #[allow(unreachable_code)]
    false
}

/// Whether there is a room to go into (Apple Vision Pro).
pub fn immersive_supported() -> bool {
    cfg!(target_os = "visionos")
}

/// Whether the immersive space is open now (the shell says so).
pub fn immersive_open() -> bool {
    IMMERSIVE.load(Ordering::Relaxed)
}

/// The view's mode now.
pub fn current_mode() -> Mode {
    if MODE.load(Ordering::Relaxed) == Mode::Ar as u32 { Mode::Ar } else { Mode::Window }
}

pub(crate) fn set_current_mode(mode: Mode) {
    MODE.store(mode as u32, Ordering::Relaxed);
}

/// `[layer setOpaque:]`.
///
/// # Safety
/// `layer` null or a live `CALayer`.
pub(crate) unsafe fn set_layer_opaque(layer: *mut c_void, opaque: bool) {
    if let Some(layer) = unsafe { (layer as *const objc2::runtime::AnyObject).as_ref() } {
        let _: () = unsafe { objc2::msg_send![layer, setOpaque: opaque] };
    }
}

/// The shell: a tap on the volume's UI image at `x`, `y` of its pixels.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_tap(x: f32, y: f32) {
    if let Ok(mut t) = TAPS.lock() {
        t.push((x, y));
    }
}

pub(crate) fn take_taps() -> Vec<(f32, f32)> {
    TAPS.lock().map(|mut t| std::mem::take(&mut *t)).unwrap_or_default()
}

/// The system's appearance as the shell last reported it: 0 not yet, 1
/// light, 2 dark.
static APPEARANCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The shell: the system's appearance is dark (or light), at launch and
/// whenever it changes (`traitCollectionDidChange`). Every app's UI follows
/// it while its theme is `Theme::System` (the default).
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_set_dark(dark: bool) {
    APPEARANCE.store(if dark { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

/// The system's appearance, if the shell has said.
pub fn system_appearance() -> Option<threers::ui::states::Appearance> {
    use threers::ui::states::Appearance;
    match APPEARANCE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Some(Appearance::Light),
        2 => Some(Appearance::Dark),
        _ => None,
    }
}

/// The shell: a press on the native view over the volume UI's control
/// `key` (`threers_vol_controls`). It reaches the app as a click there.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_press(key: u64) {
    if let Ok(mut p) = PRESSES.lock() {
        p.push(key);
    }
}

/// The shell: a finger (a look and a pinch) on the volume's UI surface
/// `which` (`volume::SURFACES`) at `x`, `y` of its image's pixels, `phase`
/// 0 down, 1 moved, 2 up, 3 cancelled. It reaches the app as an
/// [`Event::Touch`](threers::ui::input::Event::Touch), so sliders drag and
/// lists scroll and fling.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_touch(which: u32, phase: u32, x: f32, y: f32) {
    if let Ok(mut t) = TOUCHES.lock() {
        t.push((which, phase, x, y, std::time::Instant::now()));
    }
}

pub(crate) fn take_touches(which: u32) -> Vec<(u32, f32, f32, std::time::Instant)> {
    let Ok(mut t) = TOUCHES.lock() else { return vec![] };
    let (mine, rest): (Vec<_>, Vec<_>) = t.drain(..).partition(|e| e.0 == which);
    *t = rest;
    mine.into_iter().map(|(_, p, x, y, at)| (p, x, y, at)).collect()
}

/// The presses on controls `mine` says are its; the rest stay queued.
pub(crate) fn take_presses_for(mine: impl Fn(u64) -> bool) -> Vec<u64> {
    let Ok(mut p) = PRESSES.lock() else { return vec![] };
    let (taken, rest): (Vec<_>, Vec<_>) = p.drain(..).partition(|&k| mine(k));
    *p = rest;
    taken
}

/// The shell: the native text field over the volume UI's text control
/// `key` (a search box, `threers_vol_controls`'s `edit`) holds `text`
/// (UTF-8, `len` bytes) now. The control is focused and given it.
///
/// # Safety
/// `text` must point to `len` readable bytes (or be null with `len` 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_ios_text(key: u64, text: *const u8, len: usize) {
    let bytes = if text.is_null() || len == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(text, len) } };
    if let Ok(mut t) = TEXTS.lock() {
        t.push((key, String::from_utf8_lossy(bytes).into_owned()));
    }
}

/// The texts for controls `mine` says are its; the rest stay queued.
pub(crate) fn take_texts_for(mine: impl Fn(u64) -> bool) -> Vec<(u64, String)> {
    let Ok(mut t) = TEXTS.lock() else { return vec![] };
    let (taken, rest): (Vec<_>, Vec<_>) = t.drain(..).partition(|(k, _)| mine(*k));
    *t = rest;
    taken
}

/// The shell: the volume's window came on screen or went (visionOS).
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_volume_shown(shown: bool) {
    VOLUME_SHOWN.store(shown, Ordering::Relaxed);
}

/// Whether the volume's window is on screen, with its UI as native
/// controls beside it. Windows stay in view in the room, so an app can
/// leave the room's panel out then (`App::ui` for `Place::Panel` → `None`):
/// one set of controls, the system's look highlight on them.
pub fn volume_shown() -> bool {
    VOLUME_SHOWN.load(Ordering::Relaxed)
}

/// The shell's poll: what the app asked of it since the last call — 0
/// nothing, 3 enter the immersive space, 4 leave it ([`Request`]).
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_take_request() -> u32 {
    SHELL_REQUEST.swap(0, Ordering::Relaxed)
}

/// Web pages the app asked to open (a paper's DOI, its PDF), oldest first.
static URLS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Open `url` with the system (Safari; on Vision Pro, a window of its
/// own): the shell takes it ([`threers_ios_take_url`]).
pub fn open_url(url: &str) {
    if let Ok(mut u) = URLS.lock() {
        u.push(url.to_owned());
    }
}

/// The shell's poll for [`open_url`]: the oldest URL asked for, written to
/// `out` (up to `cap` bytes, UTF-8); its length, 0 when there is none (or
/// it does not fit, when it is dropped).
///
/// # Safety
/// `out` must be writable for `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_ios_take_url(out: *mut u8, cap: usize) -> usize {
    let Ok(mut u) = URLS.lock() else { return 0 };
    if u.is_empty() || out.is_null() {
        return 0;
    }
    let url = u.remove(0);
    if url.len() > cap {
        return 0;
    }
    unsafe { std::ptr::copy_nonoverlapping(url.as_ptr(), out, url.len()) };
    url.len()
}

/// The shell telling the app the immersive space opened or closed.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_immersive(open: bool) {
    IMMERSIVE.store(open, Ordering::Relaxed);
}

/// Whether the immersive space is open: the volume hides its model while
/// the room shows it.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_immersive_open() -> bool {
    IMMERSIVE.load(Ordering::Relaxed)
}

/// Makes an [`App`] the app: exports the C functions `app/Harness.swift`
/// calls.
///
/// ```c
/// void *threers_view_create(void *metal_layer, uint32_t w, uint32_t h, uint32_t mode); // 0 window, 1 AR; NULL on failure
/// void  threers_view_set(void *view, uint32_t key, float a, float b, float c);         // threers_ios::Key
/// void  threers_view_ar_camera(void *view, const float world[16], const float projection[16]); // AR's camera, where this crate has no ARKit
/// bool  threers_view_frame(void *view);                                                // once per display refresh
/// void  threers_view_destroy(void *view);
/// const uint8_t *threers_view_controls(void *view, size_t *len);                      // its UI's controls, JSON; NULL: unchanged
///
/// // a RealityKit volume (visionOS): see `volume`
/// void *threers_vol_create(void);
/// uint32_t threers_vol_frame(void *vol);                                              // meshes now
/// bool  threers_vol_object(void *vol, uint32_t i, uint64_t keys[3], float floats[32]);
/// bool  threers_vol_geometry(void *vol, uint32_t i, const void *ptrs[4], uint32_t counts[3]);
/// const float *threers_vol_uvs(void *vol, uint32_t i, size_t *len);                 // a mesh's texture coordinates
/// const uint8_t *threers_vol_texture(void *vol, uint32_t i, uint32_t *w, uint32_t *h, size_t *len); // a sprite's RGBA8, rows from the top
/// void  threers_vol_root(void *vol, float out[17]);
/// void  threers_vol_turn(void *vol, float radians);
/// const uint8_t *threers_vol_ui(void *vol, uint32_t which, uint32_t w, uint32_t h, float scale, size_t *len); // 0 card, 1 toolbar; NULL: unchanged, len 0: hidden
/// const uint8_t *threers_vol_controls(void *vol, uint32_t which, size_t *len); // its controls, JSON; NULL: unchanged
/// void  threers_vol_destroy(void *vol);
///
/// void  threers_app_run(void *layer_renderer);                                         // visionOS: the immersive space
/// ```
///
/// Every view and volume function takes a null handle and does nothing.
#[macro_export]
macro_rules! app {
    ($app:ty) => {
        /// # Safety
        /// `layer` must be a live `CAMetalLayer` that outlives the view.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_view_create(layer: *mut ::core::ffi::c_void, width: u32, height: u32, mode: u32) -> *mut ::core::ffi::c_void {
            unsafe { $crate::view::create::<$app>(layer, width, height, mode) }
        }

        /// # Safety
        /// `view` null or from `threers_view_create`, not destroyed.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_view_set(view: *mut ::core::ffi::c_void, key: u32, a: f32, b: f32, c: f32) {
            if let Some(v) = unsafe { $crate::view::view::<$app>(view) } {
                v.set(key, a, b, c);
            }
        }

        /// # Safety
        /// As `threers_view_set`; each matrix null or sixteen floats.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_view_ar_camera(view: *mut ::core::ffi::c_void, world: *const f32, projection: *const f32) {
            let (Some(v), Some(w), Some(p)) = (unsafe { $crate::view::view::<$app>(view) }, unsafe { $crate::view::matrix(world) }, unsafe { $crate::view::matrix(projection) }) else {
                return;
            };
            v.set_ar_camera(w, p);
        }

        /// # Safety
        /// As `threers_view_set`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_view_frame(view: *mut ::core::ffi::c_void) -> bool {
            unsafe { $crate::view::view::<$app>(view) }.is_some_and(|v| v.frame())
        }

        /// # Safety
        /// `view` null or from `threers_view_create`; not used again.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_view_destroy(view: *mut ::core::ffi::c_void) {
            unsafe { $crate::view::destroy::<$app>(view) }
        }

        /// # Safety
        /// `view` null or from `threers_view_create`; `len` null or writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_view_controls(view: *mut ::core::ffi::c_void, len: *mut usize) -> *const u8 {
            unsafe { $crate::view::controls::<$app>(view, len) }
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn threers_vol_create() -> *mut ::core::ffi::c_void {
            $crate::volume::create::<$app>()
        }
        /// # Safety
        /// `vol` null or from `threers_vol_create`; not used again.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_destroy(vol: *mut ::core::ffi::c_void) {
            unsafe { $crate::volume::destroy::<$app>(vol) }
        }
        /// # Safety
        /// `vol` null or from `threers_vol_create`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_frame(vol: *mut ::core::ffi::c_void) -> u32 {
            unsafe { $crate::volume::frame::<$app>(vol) }
        }
        /// # Safety
        /// As `threers_vol_frame`; 3 keys, 32 floats writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_object(vol: *mut ::core::ffi::c_void, i: u32, keys: *mut u64, floats: *mut f32) -> bool {
            unsafe { $crate::volume::object::<$app>(vol, i, keys, floats) }
        }
        /// # Safety
        /// As `threers_vol_frame`; 4 pointers, 3 counts writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_geometry(vol: *mut ::core::ffi::c_void, i: u32, ptrs: *mut *const ::core::ffi::c_void, counts: *mut u32) -> bool {
            unsafe { $crate::volume::geometry::<$app>(vol, i, ptrs, counts) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_uvs(vol: *mut ::core::ffi::c_void, i: u32, len: *mut usize) -> *const f32 {
            unsafe { $crate::volume::uvs::<$app>(vol, i, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `w`, `h`, `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_texture(vol: *mut ::core::ffi::c_void, i: u32, w: *mut u32, h: *mut u32, len: *mut usize) -> *const u8 {
            unsafe { $crate::volume::texture::<$app>(vol, i, w, h, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`; 17 floats writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_root(vol: *mut ::core::ffi::c_void, out: *mut f32) {
            unsafe { $crate::volume::root::<$app>(vol, out) }
        }
        /// # Safety
        /// As `threers_vol_frame`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_turn(vol: *mut ::core::ffi::c_void, radians: f32) {
            unsafe { $crate::volume::turn::<$app>(vol, radians) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_ui(vol: *mut ::core::ffi::c_void, which: u32, width: u32, height: u32, scale: f32, len: *mut usize) -> *const u8 {
            unsafe { $crate::volume::ui::<$app>(vol, which, width, height, scale, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_controls(vol: *mut ::core::ffi::c_void, which: u32, len: *mut usize) -> *const u8 {
            unsafe { $crate::volume::controls::<$app>(vol, which, len) }
        }

        /// The immersive space: run on the shell's render thread until it
        /// closes.
        ///
        /// # Safety
        /// `layer_renderer` must be the `CompositorLayer`'s layer renderer.
        #[cfg(target_os = "visionos")]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_app_run(layer_renderer: *mut ::core::ffi::c_void) {
            unsafe { $crate::immersive::run::<$app>(layer_renderer) }
        }
    };
}
