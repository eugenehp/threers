//! threers apps on iPhone, iPad, Apple Vision Pro and the Mac, written in
//! Rust and drawn with Metal or RealityKit: one [`App`] in eight places, with
//! a Swift shell that is the same for every app (`app/Harness.swift`,
//! `app/Reality.swift`, `app/MacHarness.swift`) and a build script that
//! needs no Xcode project (`app/build.sh`).
//!
//! - iOS, **3D** — a window with an orbit camera ([`View`], [`Mode::Window`]);
//! - iOS, **AR** — over the camera, ARKit run from Rust (`ar`, [`Mode::Ar`]);
//! - iOS, **RealityKit** — a `RealityView`, on the screen or over the camera
//!   (`ios-reality` builds);
//! - visionOS, **window** — the orbit view in a window;
//! - visionOS, **volume** — in the shared space, beside other apps, no
//!   immersion ([`volume`]): threers' picture or RealityKit's entities;
//! - visionOS, **immersive** — both eyes in the room (`immersive`);
//! - macOS — a RealityKit window (`mac` builds).
//!
//! The Metal modes draw the scene with threers' own Metal renderer
//! ([`MetalRenderer`](threers::metal::MetalRenderer)): no wgpu, no shader
//! translation at run time. The RealityKit ones hand it over entity by
//! entity ([`volume`], [`reality`]). The app's UI is threers-ui ([`App::ui`]),
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

#![cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

pub use threers;

pub mod app;
#[cfg(all(target_os = "ios", target_arch = "aarch64"))]
pub mod ar;
pub mod bake;
#[cfg(target_os = "visionos")]
pub mod immersive;
pub mod reality;
mod surface_ui;
pub mod view;
pub mod volume;

pub use app::{App, PanelSpec, Place, Surface};
pub use reality::{RealityLighting, VolumeDrawing};
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
    /// A backdrop round you (visionOS): a dark sky in place of the room you
    /// are in, the app's windows in front of it — what is small and bright
    /// (a neuron's branches) reads against it. The room closes for it.
    EnterBackdrop = 5,
    /// Back to your surroundings.
    ExitBackdrop = 6,
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
static BACKDROP: AtomicBool = AtomicBool::new(false);
static MODE: AtomicU32 = AtomicU32::new(Mode::Window as u32);
static SPINNING: AtomicBool = AtomicBool::new(false);
static CAMERA: Mutex<Vec<CameraRequest>> = Mutex::new(Vec::new());
static TAPS: Mutex<Vec<(f32, f32)>> = Mutex::new(Vec::new());
static PRESSES: Mutex<Vec<u64>> = Mutex::new(Vec::new());
/// A finger on a volume's UI surface: the surface, its phase, where, when.
type Touch = (u32, u32, f32, f32, std::time::Instant);
static TOUCHES: Mutex<Vec<Touch>> = Mutex::new(Vec::new());
static VOLUME_SHOWN: AtomicBool = AtomicBool::new(false);
static TEXTS: Mutex<Vec<(u64, String)>> = Mutex::new(Vec::new());

/// Ask for a mode; it happens at the next frame (the view's), or when the
/// shell next polls (the immersive space). The last request wins.
pub fn request(r: Request) {
    match r {
        Request::Window | Request::Ar => VIEW_REQUEST.store(r as u32, Ordering::Relaxed),
        Request::EnterImmersive
        | Request::ExitImmersive
        | Request::EnterBackdrop
        | Request::ExitBackdrop => SHELL_REQUEST.store(r as u32, Ordering::Relaxed),
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
    CAMERA
        .lock()
        .map(|mut q| std::mem::take(&mut *q))
        .unwrap_or_default()
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

/// Whether the backdrop is round you now (the shell says so).
pub fn backdrop_open() -> bool {
    BACKDROP.load(Ordering::Relaxed)
}

/// The view's mode now.
pub fn current_mode() -> Mode {
    if MODE.load(Ordering::Relaxed) == Mode::Ar as u32 {
        Mode::Ar
    } else {
        Mode::Window
    }
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
    TAPS.lock()
        .map(|mut t| std::mem::take(&mut *t))
        .unwrap_or_default()
}

/// The system's appearance as the shell last reported it: 0 not yet, 1
/// light, 2 dark.
static APPEARANCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The shell: the system's appearance is dark (or light), at launch and
/// whenever it changes (`traitCollectionDidChange`). Every app's UI follows
/// it while its theme is `Theme::System` (the default).
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_set_dark(dark: bool) {
    APPEARANCE.store(
        if dark { 2 } else { 1 },
        std::sync::atomic::Ordering::Relaxed,
    );
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
    let Ok(mut t) = TOUCHES.lock() else {
        return vec![];
    };
    let (mine, rest): (Vec<_>, Vec<_>) = t.drain(..).partition(|e| e.0 == which);
    *t = rest;
    mine.into_iter()
        .map(|(_, p, x, y, at)| (p, x, y, at))
        .collect()
}

/// The presses on controls `mine` says are its; the rest stay queued.
pub(crate) fn take_presses_for(mine: impl Fn(u64) -> bool) -> Vec<u64> {
    let Ok(mut p) = PRESSES.lock() else {
        return vec![];
    };
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
    let bytes = if text.is_null() || len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(text, len) }
    };
    if let Ok(mut t) = TEXTS.lock() {
        t.push((key, String::from_utf8_lossy(bytes).into_owned()));
    }
}

/// The texts for controls `mine` says are its; the rest stay queued.
pub(crate) fn take_texts_for(mine: impl Fn(u64) -> bool) -> Vec<(u64, String)> {
    let Ok(mut t) = TEXTS.lock() else {
        return vec![];
    };
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

thread_local! {
    static REALITYKIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the scene the app is updating now is drawn by RealityKit (the
/// volume), not threers' Metal renderer (a window, AR, the room): no
/// shader of the app's runs there — a flat
/// [`ShaderMaterial`](threers::materials::ShaderMaterial) mesh is baked
/// into a texture ([`bake`]), anything else of one is left out — so an
/// app can build what RealityKit can draw instead. Asked inside
/// [`App::update`](app::App::update).
pub fn drawn_by_realitykit() -> bool {
    REALITYKIT.with(|f| f.get())
}

pub(crate) fn set_drawn_by_realitykit(on: bool) {
    REALITYKIT.with(|f| f.set(on));
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
    unsafe { hand_over(URLS.lock().ok().and_then(|mut u| (!u.is_empty()).then(|| u.remove(0))), out, cap) }
}

/// What the app hands the shell through a poll (a URL, a path, text):
/// `text` written to `out` (up to `cap` bytes, UTF-8); its length, 0 when
/// there is none, or it does not fit (when it is dropped).
///
/// # Safety
/// `out` must be writable for `cap` bytes.
unsafe fn hand_over(text: Option<String>, out: *mut u8, cap: usize) -> usize {
    let Some(text) = text else { return 0 };
    if out.is_null() || text.len() > cap {
        return 0;
    }
    unsafe { std::ptr::copy_nonoverlapping(text.as_ptr(), out, text.len()) };
    text.len()
}

/// A file asked for ([`pick_file`]), and those picked, not yet taken.
static PICK: AtomicBool = AtomicBool::new(false);
static PICKED: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Ask the person for a file or a folder of theirs: the shell shows the
/// system's document picker, and what they choose arrives by
/// [`take_picked`] as a path the app may read (a folder's, while the app runs).
pub fn pick_file() {
    PICK.store(true, Ordering::Relaxed);
}

/// The paths picked since the last call, oldest first.
pub fn take_picked() -> Vec<String> {
    PICKED.lock().map(|mut p| std::mem::take(&mut *p)).unwrap_or_default()
}

/// The shell's poll for [`pick_file`]: whether to show the picker now.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_take_pick() -> bool {
    PICK.swap(false, Ordering::Relaxed)
}

/// The shell: the person picked `path` (`len` bytes, UTF-8).
///
/// # Safety
/// `path` must be readable for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_ios_picked(path: *const u8, len: usize) {
    if path.is_null() {
        return;
    }
    let bytes = unsafe { std::slice::from_raw_parts(path, len) };
    if let (Ok(s), Ok(mut p)) = (std::str::from_utf8(bytes), PICKED.lock()) {
        p.push(s.to_owned());
    }
}

/// Text the app put on the clipboard since the shell last looked.
static COPIED: Mutex<Option<String>> = Mutex::new(None);

/// Put `text` on the system's clipboard (the shell does, at its next poll).
pub fn copy_text(text: &str) {
    if let Ok(mut c) = COPIED.lock() {
        *c = Some(text.to_owned());
    }
}

/// The shell's poll for [`copy_text`]: the text, written to `out` (up to
/// `cap` bytes, UTF-8); its length, 0 when there is none (or it does not
/// fit, when it is dropped).
///
/// # Safety
/// `out` must be writable for `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_ios_take_copy(out: *mut u8, cap: usize) -> usize {
    unsafe { hand_over(COPIED.lock().ok().and_then(|mut c| c.take()), out, cap) }
}

/// A turn being recorded ([`record_turn`]): frames to make, made, the rate;
/// and the frame waiting for the shell (BGRA, its size, its index, whether
/// it is the last).
pub(crate) struct Recording {
    pub frames: u32,
    pub made: u32,
    pub fps: u32,
}
pub(crate) static RECORDING: Mutex<Option<Recording>> = Mutex::new(None);
/// A recorded frame: BGRA, width, height, index, whether the last.
pub(crate) type RecFrame = (Vec<u8>, u32, u32, u32, bool);
pub(crate) static FRAME: Mutex<Option<RecFrame>> = Mutex::new(None);

/// Record a full turn of the window's view about its centre, `seconds`
/// long at `fps`, as a video: each frame drawn offscreen (the scene, no UI)
/// and handed to the shell, which writes an MP4 and offers it to share.
/// Nothing in AR, the room or a volume.
pub fn record_turn(seconds: f32, fps: u32) {
    let fps = fps.clamp(1, 60);
    if let Ok(mut r) = RECORDING.lock() {
        *r = Some(Recording { frames: ((seconds * fps as f32).round() as u32).max(2), made: 0, fps });
    }
}

/// Whether a turn is being recorded.
pub fn recording() -> bool {
    RECORDING.lock().is_ok_and(|r| r.is_some())
}

/// The shell's poll for a recorded frame: whether one waits, and its
/// width, height, index, rate and whether it is the last.
///
/// # Safety
/// The pointers must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_ios_rec_peek(w: *mut u32, h: *mut u32, index: *mut u32, fps: *mut u32, last: *mut bool) -> bool {
    let Ok(f) = FRAME.lock() else { return false };
    let Some((_, fw, fh, i, l)) = f.as_ref() else { return false };
    let rate = RECORDING.lock().ok().and_then(|r| r.as_ref().map(|r| r.fps)).unwrap_or(30);
    unsafe {
        *w = *fw;
        *h = *fh;
        *index = *i;
        *fps = rate;
        *last = *l;
    }
    true
}

/// The shell takes the waiting frame: its BGRA bytes into `out` (`cap`
/// bytes); whether they fitted. The slot is free for the next one then.
///
/// # Safety
/// `out` must be writable for `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_ios_rec_take(out: *mut u8, cap: usize) -> bool {
    let Ok(mut f) = FRAME.lock() else { return false };
    let Some((bytes, _, _, _, last)) = f.take() else { return false };
    if last && let Ok(mut r) = RECORDING.lock() {
        *r = None;
    }
    if out.is_null() || bytes.len() > cap {
        return false;
    }
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len()) };
    true
}

/// A picture of the window's view asked for ([`snapshot`]).
static SNAPSHOT: AtomicBool = AtomicBool::new(false);

/// Ask for a picture of the view as it is: at the next frame the window's
/// is drawn once more, offscreen, at its size — or a volume's frame read
/// back, square, clear round the model — and handed to [`App::snapshot`].
/// Nothing in AR or the room.
pub fn snapshot() {
    SNAPSHOT.store(true, Ordering::Relaxed);
}

pub(crate) fn take_snapshot() -> bool {
    SNAPSHOT.swap(false, Ordering::Relaxed)
}

/// Files the app made for the person to keep or send (a figure, a table),
/// for the shell's share sheet, oldest first.
static SHARES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Offer the file at `path` (one the app wrote, under
/// [`std::env::temp_dir`] say) through the system's share sheet: saved to
/// Files or Photos, sent, or opened in another app.
pub fn share(path: &str) {
    if let Ok(mut s) = SHARES.lock() {
        s.push(path.to_owned());
    }
}

/// The shell's poll for [`share`]: the oldest path asked for, written to
/// `out` (up to `cap` bytes, UTF-8); its length, 0 when there is none (or
/// it does not fit, when it is dropped).
///
/// # Safety
/// `out` must be writable for `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_ios_take_share(out: *mut u8, cap: usize) -> usize {
    unsafe { hand_over(SHARES.lock().ok().and_then(|mut s| (!s.is_empty()).then(|| s.remove(0))), out, cap) }
}

/// The view request ([`Request::Window`], [`Request::Ar`]) for a shell that
/// draws with RealityKit rather than [`View`]: 0 none, else the request.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_take_view_request() -> u32 {
    take_view_request().map_or(0, |r| r as u32)
}

/// The RealityKit shell switched to AR over the camera (`ar`) or back to
/// its own camera: [`current_mode`] says so.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_reality_mode(ar: bool) {
    set_current_mode(if ar { Mode::Ar } else { Mode::Window });
}

/// Files the app asked to show in AR Quick Look, oldest first.
static QUICK_LOOKS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Show the `.usdz` at `path` in the system's AR Quick Look (iOS, iPadOS),
/// in its own window in the shared space (visionOS) or in Preview (the
/// Mac): RealityKit drawing it, placed in the room, scaled with the
/// fingers. [`write_usdz`] writes a scene for it.
pub fn quick_look(path: &str) {
    if let Ok(mut q) = QUICK_LOOKS.lock() {
        q.push(path.to_owned());
    }
}

/// `scene` as a `.usdz` RealityKit opens (its meshes, materials and maps,
/// packed), written to `path`: what [`quick_look`] shows, or a file to
/// share. The package is checked against what RealityKit will open in one
/// ([`arkit_issues`](threers::loaders::usd::arkit_issues)); a
/// problem with it is an error.
pub fn write_usdz(scene: &mut threers::scene::Scene, path: &std::path::Path) -> std::io::Result<()> {
    scene.update_world();
    let bytes = threers::loaders::usd::UsdExport::scene(&scene.arena, &[scene.root]).usdz();
    let issues = threers::loaders::usd::arkit_issues(&bytes);
    if let Some(issue) = issues.first() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, issue.to_string()));
    }
    std::fs::write(path, bytes)
}

/// The shell's poll for [`quick_look`]: the oldest path, written to `out`
/// (up to `cap` bytes, UTF-8); its length, 0 when there is none (or it does
/// not fit, when it is dropped).
///
/// # Safety
/// `out` must be writable for `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_ios_take_quick_look(out: *mut u8, cap: usize) -> usize {
    let Ok(mut q) = QUICK_LOOKS.lock() else { return 0 };
    if q.is_empty() || out.is_null() {
        return 0;
    }
    let path = q.remove(0);
    if path.len() > cap {
        return 0;
    }
    unsafe { std::ptr::copy_nonoverlapping(path.as_ptr(), out, path.len()) };
    path.len()
}

/// What the app asked the shell's player to do since it last looked:
/// `play\t<at>\t<url>` (from `at` seconds, or where it is when negative)
/// or `pause`; taken by [`threers_ios_take_audio`].
static AUDIO: Mutex<Option<String>> = Mutex::new(None);
/// Where the shell's player is (seconds, its bits; NaN: none), and whether
/// it is playing.
static AUDIO_TIME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0x7ff8_0000_0000_0000);
static AUDIO_PLAYING: AtomicBool = AtomicBool::new(false);

/// What the app asks of its sound (a story an fMRI run heard, its view's
/// clock).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Audio<'a> {
    /// Play `url` from `at` seconds (from where it is: `None`); another URL
    /// replaces the one loaded.
    Play { url: &'a str, at: Option<f64> },
    /// Pause, keeping the place.
    Pause,
}

/// Ask the shell's player: [`audio_state`] says what it is doing.
pub fn audio(cmd: Audio) {
    let s = match cmd {
        Audio::Play { url, at } => format!("play\t{}\t{url}", at.unwrap_or(-1.0)),
        Audio::Pause => "pause".into(),
    };
    if let Ok(mut a) = AUDIO.lock() {
        *a = Some(s);
    }
}

/// Where the sound is (seconds) and whether it is playing, once one is
/// loaded.
pub fn audio_state() -> Option<(f64, bool)> {
    Some(f64::from_bits(AUDIO_TIME.load(Ordering::Relaxed)))
        .filter(|t| t.is_finite())
        .map(|t| (t, AUDIO_PLAYING.load(Ordering::Relaxed)))
}

/// The shell's poll for [`audio`]: −1 nothing asked; else the request's
/// length, written to `out` (up to `cap` bytes, UTF-8).
///
/// # Safety
/// `out` must be writable for `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn threers_ios_take_audio(out: *mut u8, cap: usize) -> isize {
    let Ok(mut a) = AUDIO.lock() else { return -1 };
    let Some(cmd) = a.take() else { return -1 };
    if cmd.len() > cap || out.is_null() {
        return -1;
    }
    unsafe { std::ptr::copy_nonoverlapping(cmd.as_ptr(), out, cmd.len()) };
    cmd.len() as isize
}

/// The shell telling the app where its player is (seconds) and whether it
/// is playing.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_audio_time(t: f64, playing: bool) {
    AUDIO_TIME.store(t.to_bits(), Ordering::Relaxed);
    AUDIO_PLAYING.store(playing, Ordering::Relaxed);
}

/// The shell telling the app the immersive space opened or closed.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_immersive(open: bool) {
    IMMERSIVE.store(open, Ordering::Relaxed);
}

/// The shell telling the app the backdrop opened or closed.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_backdrop(open: bool) {
    BACKDROP.store(open, Ordering::Relaxed);
}

/// Whether the backdrop is open.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_backdrop_open() -> bool {
    BACKDROP.load(Ordering::Relaxed)
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
/// // RealityKit: a volume (visionOS), a view on an iPhone, an iPad or a Mac: see `volume`
/// void *threers_vol_create(void);
/// uint32_t threers_vol_frame(void *vol);                                              // meshes now
/// bool  threers_vol_object(void *vol, uint32_t i, uint64_t keys[4], float floats[48]);
/// bool  threers_vol_geometry(void *vol, uint32_t i, const void *ptrs[5], uint32_t counts[3]);
/// const uint8_t *threers_vol_map(void *vol, uint32_t i, uint32_t slot, uint32_t *w, uint32_t *h, size_t *len); // a lit mesh's map, RGBA8
/// const float *threers_vol_instances(void *vol, uint32_t i, size_t *len);           // an instanced mesh's matrices
/// const float *threers_vol_lights(void *vol, size_t *len);                          // 16 floats a light
/// uint32_t threers_vol_lighting(void *vol);                                          // `RealityLighting`
/// uint32_t threers_vol_drawing(void *vol);                                           // `VolumeDrawing`
/// void  threers_vol_background(void *vol, float out[4]);                             // linear RGB, alpha
/// const float *threers_vol_environment(void *vol, uint32_t width, size_t *len);     // equirect RGBA floats; NULL: unchanged, len 0: none
/// bool  threers_vol_ui_hit(void *vol, uint32_t which, float x, float y, float scale); // the UI is under (x, y)
/// void  threers_vol_ray(void *vol, const float ray[6]);                              // a select along a ray in the scene
/// void  threers_vol_insets(void *vol, float top, float bottom, float left, float right); // the screen's safe area, points
/// const float *threers_vol_uvs(void *vol, uint32_t i, size_t *len);                 // a mesh's texture coordinates
/// const uint8_t *threers_vol_texture(void *vol, uint32_t i, uint32_t *w, uint32_t *h, size_t *len); // a sprite's RGBA8, rows from the top
/// void  threers_vol_root(void *vol, float out[17]);
/// void  threers_vol_turn(void *vol, float radians);
/// void  threers_vol_tap(void *vol, float u, float v);                                // a tap on the quad (0..1, from its foot up)
/// bool  threers_vol_draw(void *vol, float qx, float qy, float qz, float qw, uint32_t size, void *mtl_texture, void *mtl_command_buffer); // the whole scene, drawn by threers, into a quad's texture
/// const uint8_t *threers_vol_ui(void *vol, uint32_t which, uint32_t w, uint32_t h, float scale, size_t *len); // 0 card, 1 toolbar, 2 screen; NULL: unchanged, len 0: hidden
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
        pub unsafe extern "C" fn threers_view_create(
            layer: *mut ::core::ffi::c_void,
            width: u32,
            height: u32,
            mode: u32,
        ) -> *mut ::core::ffi::c_void {
            unsafe { $crate::view::create::<$app>(layer, width, height, mode) }
        }

        /// # Safety
        /// `view` null or from `threers_view_create`, not destroyed.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_view_set(
            view: *mut ::core::ffi::c_void,
            key: u32,
            a: f32,
            b: f32,
            c: f32,
        ) {
            if let Some(v) = unsafe { $crate::view::view::<$app>(view) } {
                v.set(key, a, b, c);
            }
        }

        /// # Safety
        /// As `threers_view_set`; each matrix null or sixteen floats.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_view_ar_camera(
            view: *mut ::core::ffi::c_void,
            world: *const f32,
            projection: *const f32,
        ) {
            let (Some(v), Some(w), Some(p)) = (
                unsafe { $crate::view::view::<$app>(view) },
                unsafe { $crate::view::matrix(world) },
                unsafe { $crate::view::matrix(projection) },
            ) else {
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
        pub unsafe extern "C" fn threers_view_controls(
            view: *mut ::core::ffi::c_void,
            len: *mut usize,
        ) -> *const u8 {
            unsafe { $crate::view::controls::<$app>(view, len) }
        }

        /// The view's mode requests (the window, AR): on the main thread.
        ///
        /// # Safety
        /// `view` null or from `threers_view_create`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_view_requests(view: *mut ::core::ffi::c_void) {
            unsafe { $crate::view::requests::<$app>(view) }
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
        /// As `threers_vol_frame`; 4 keys, 48 floats writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_object(
            vol: *mut ::core::ffi::c_void,
            i: u32,
            keys: *mut u64,
            floats: *mut f32,
        ) -> bool {
            unsafe { $crate::volume::object::<$app>(vol, i, keys, floats) }
        }
        /// # Safety
        /// As `threers_vol_frame`; 5 pointers, 3 counts writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_geometry(
            vol: *mut ::core::ffi::c_void,
            i: u32,
            ptrs: *mut *const ::core::ffi::c_void,
            counts: *mut u32,
        ) -> bool {
            unsafe { $crate::volume::geometry::<$app>(vol, i, ptrs, counts) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_uvs(
            vol: *mut ::core::ffi::c_void,
            i: u32,
            len: *mut usize,
        ) -> *const f32 {
            unsafe { $crate::volume::uvs::<$app>(vol, i, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `w`, `h`, `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_texture(
            vol: *mut ::core::ffi::c_void,
            i: u32,
            w: *mut u32,
            h: *mut u32,
            len: *mut usize,
        ) -> *const u8 {
            unsafe { $crate::volume::texture::<$app>(vol, i, w, h, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_tap(vol: *mut ::core::ffi::c_void, u: f32, v: f32) {
            unsafe { $crate::volume::tap::<$app>(vol, u, v) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `texture` a live `MTLTexture` (`size`
        /// square, RGBA8 sRGB) and `command_buffer` a live, uncommitted
        /// `MTLCommandBuffer`, on the system's default device.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_draw(
            vol: *mut ::core::ffi::c_void,
            qx: f32,
            qy: f32,
            qz: f32,
            qw: f32,
            size: u32,
            texture: *mut ::core::ffi::c_void,
            command_buffer: *mut ::core::ffi::c_void,
        ) -> bool {
            unsafe {
                $crate::volume::draw::<$app>(vol, [qx, qy, qz, qw], size, texture, command_buffer)
            }
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
        pub unsafe extern "C" fn threers_vol_ui(
            vol: *mut ::core::ffi::c_void,
            which: u32,
            width: u32,
            height: u32,
            scale: f32,
            len: *mut usize,
        ) -> *const u8 {
            unsafe { $crate::volume::ui::<$app>(vol, which, width, height, scale, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_controls(
            vol: *mut ::core::ffi::c_void,
            which: u32,
            len: *mut usize,
        ) -> *const u8 {
            unsafe { $crate::volume::controls::<$app>(vol, which, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `w`, `h`, `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_map(
            vol: *mut ::core::ffi::c_void,
            i: u32,
            slot: u32,
            w: *mut u32,
            h: *mut u32,
            len: *mut usize,
        ) -> *const u8 {
            unsafe { $crate::volume::map::<$app>(vol, i, slot, w, h, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_instances(
            vol: *mut ::core::ffi::c_void,
            i: u32,
            len: *mut usize,
        ) -> *const f32 {
            unsafe { $crate::volume::instances::<$app>(vol, i, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_lights(
            vol: *mut ::core::ffi::c_void,
            len: *mut usize,
        ) -> *const f32 {
            unsafe { $crate::volume::lights::<$app>(vol, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_lighting(vol: *mut ::core::ffi::c_void) -> u32 {
            unsafe { $crate::volume::lighting::<$app>(vol) }
        }
        /// # Safety
        /// As `threers_vol_frame`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_drawing(vol: *mut ::core::ffi::c_void) -> u32 {
            unsafe { $crate::volume::drawing::<$app>(vol) }
        }
        /// # Safety
        /// As `threers_vol_frame`; 4 floats writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_background(vol: *mut ::core::ffi::c_void, out: *mut f32) {
            unsafe { $crate::volume::background::<$app>(vol, out) }
        }
        /// # Safety
        /// As `threers_vol_frame`; `len` writable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_environment(
            vol: *mut ::core::ffi::c_void,
            width: u32,
            len: *mut usize,
        ) -> *const f32 {
            unsafe { $crate::volume::environment::<$app>(vol, width, len) }
        }
        /// # Safety
        /// As `threers_vol_frame`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_ui_hit(
            vol: *mut ::core::ffi::c_void,
            which: u32,
            x: f32,
            y: f32,
            scale: f32,
        ) -> bool {
            unsafe { $crate::volume::ui_hit::<$app>(vol, which, x, y, scale) }
        }
        /// # Safety
        /// As `threers_vol_frame`.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_insets(
            vol: *mut ::core::ffi::c_void,
            top: f32,
            bottom: f32,
            left: f32,
            right: f32,
        ) {
            unsafe { $crate::volume::insets::<$app>(vol, top, bottom, left, right) }
        }
        /// # Safety
        /// As `threers_vol_frame`; 6 floats readable.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn threers_vol_ray(vol: *mut ::core::ffi::c_void, ray: *const f32) {
            unsafe { $crate::volume::ray::<$app>(vol, ray) }
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
