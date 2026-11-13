//! ARKit on iOS, from Rust: a world-tracking session, its camera feed shown
//! behind the app's layer, and the camera's pose and projection each frame.
//!
//! The feed is ARKit's own view (`ARSCNView`, with an empty scene): it runs
//! the session and draws the camera image the way ARKit means it to be
//! drawn — cropped, rotated and timed — with nothing copied. It goes into
//! the layer's view's superview, beneath it, so the shell needs a container
//! view around the layer's view, which `app/Harness.swift` has.
//!
//! The Objective-C runtime is called directly (objc2's `msg_send!`), with two
//! exceptions: ARKit's camera returns `simd_float4x4`, which Apple's ARM64
//! convention passes back in `q0`–`q3`, and Rust has no type FFI returns
//! that way (threers' `metal::visionos` explains). Those two calls are made
//! as C makes them, by a branch in assembly that reads the registers back.

use std::ffi::c_void;

use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyClass, AnyObject, Bool};
use objc2::{msg_send, sel};
use objc2_core_foundation::CGRect;

#[link(name = "ARKit", kind = "framework")]
unsafe extern "C" {}
#[link(name = "UIKit", kind = "framework")]
unsafe extern "C" {}
#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    fn objc_msgSend();
}

/// `UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight`
const FLEXIBLE_SIZE: usize = 2 | 16;
/// `UIInterfaceOrientationPortrait`
const PORTRAIT: isize = 1;

/// Whether this device can track the world (not the simulator, not the
/// oldest devices).
pub fn supported() -> bool {
    let Some(class) = AnyClass::get(c"ARWorldTrackingConfiguration") else { return false };
    let yes: Bool = unsafe { msg_send![class, isSupported] };
    yes.as_bool()
}

/// A running session and the view showing its camera.
pub struct ArSession {
    /// The `ARSCNView`.
    view: Retained<AnyObject>,
}

impl ArSession {
    /// Starts world tracking, its camera feed beneath the view whose layer
    /// is `layer`. `None` where there is no world tracking, or the layer's
    /// view has no superview to put the feed in.
    ///
    /// # Safety
    /// `layer` must be a live `CAMetalLayer` backing a `UIView`; call on the
    /// main thread.
    pub unsafe fn start(layer: *mut c_void) -> Option<Self> {
        if !supported() || layer.is_null() {
            return None;
        }
        unsafe {
            let layer = &*(layer as *const AnyObject);
            // a view's backing layer has the view as its delegate
            let view: Option<Retained<AnyObject>> = msg_send![layer, delegate];
            let view = view?;
            let host: Option<Retained<AnyObject>> = msg_send![&*view, superview];
            let host = host?;
            let bounds: CGRect = msg_send![&*host, bounds];
            let feed: Allocated<AnyObject> = msg_send![AnyClass::get(c"ARSCNView")?, alloc];
            let feed: Option<Retained<AnyObject>> = msg_send![feed, initWithFrame: bounds];
            let feed = feed?;
            let _: () = msg_send![&*feed, setAutoresizingMask: FLEXIBLE_SIZE];
            let _: () = msg_send![&*feed, setAutomaticallyUpdatesLighting: false];
            let _: () = msg_send![&*host, insertSubview: &*feed, belowSubview: &*view];
            let config: Option<Retained<AnyObject>> = msg_send![AnyClass::get(c"ARWorldTrackingConfiguration")?, new];
            let config = config?;
            // +Y against gravity, the origin where the session starts
            let _: () = msg_send![&*config, setWorldAlignment: 0isize];
            let session: Retained<AnyObject> = msg_send![&*feed, session];
            let _: () = msg_send![&*session, runWithConfiguration: &*config];
            Some(ArSession { view: feed })
        }
    }

    /// The camera now: its camera-to-world transform and its projection,
    /// column-major, for the interface's orientation and a drawable of
    /// `size` pixels. `None` before ARKit's first frame.
    pub fn camera(&self, size: (u32, u32), near: f32, far: f32) -> Option<([f32; 16], [f32; 16])> {
        unsafe {
            let session: Retained<AnyObject> = msg_send![&*self.view, session];
            let frame: Option<Retained<AnyObject>> = msg_send![&*session, currentFrame];
            let frame = frame?;
            let camera: Retained<AnyObject> = msg_send![&*frame, camera];
            let orientation = self.orientation();
            let cam = Retained::as_ptr(&camera) as usize;
            let view = call_int(cam, sel_bits(sel!(viewMatrixForOrientation:)), orientation as usize);
            let projection = call_projection(
                cam,
                sel_bits(sel!(projectionMatrixForOrientation:viewportSize:zNear:zFar:)),
                orientation as usize,
                [size.0.max(1) as f64, size.1.max(1) as f64, near as f64, far as f64],
            );
            let world = threers::math::Matrix4 { elements: flat(view) }.invert().elements;
            Some((world, flat(projection)))
        }
    }

    /// The window scene's interface orientation, portrait when unknown.
    fn orientation(&self) -> isize {
        unsafe {
            let window: Option<Retained<AnyObject>> = msg_send![&*self.view, window];
            let Some(window) = window else { return PORTRAIT };
            let scene: Option<Retained<AnyObject>> = msg_send![&*window, windowScene];
            let Some(scene) = scene else { return PORTRAIT };
            let o: isize = msg_send![&*scene, interfaceOrientation];
            if o == 0 { PORTRAIT } else { o }
        }
    }
}

impl Drop for ArSession {
    fn drop(&mut self) {
        unsafe {
            let session: Retained<AnyObject> = msg_send![&*self.view, session];
            let _: () = msg_send![&*session, pause];
            let _: () = msg_send![&*self.view, removeFromSuperview];
        }
    }
}

/// A selector as the register holds it (a `SEL` is a pointer).
fn sel_bits(sel: objc2::runtime::Sel) -> usize {
    // SAFETY: `Sel` is a non-null pointer, `repr(transparent)`.
    unsafe { std::mem::transmute::<objc2::runtime::Sel, usize>(sel) }
}

fn flat(columns: [[f32; 4]; 4]) -> [f32; 16] {
    let mut m = [0.0; 16];
    for (c, col) in columns.iter().enumerate() {
        m[c * 4..c * 4 + 4].copy_from_slice(col);
    }
    m
}

/// `objc_msgSend(x0, x1, x2)` for a method returning a `simd_float4x4`.
unsafe fn call_int(x0: usize, x1: usize, x2: usize) -> [[f32; 4]; 4] {
    use std::arch::aarch64::float32x4_t;
    let (c0, c1, c2, c3): (float32x4_t, float32x4_t, float32x4_t, float32x4_t);
    // SAFETY: the method takes up to three integer arguments and returns
    // the four columns in v0–v3; the rest of what C may clobber is declared.
    unsafe {
        std::arch::asm!(
            "blr {f}",
            f = in(reg) objc_msgSend as unsafe extern "C" fn(),
            inout("x0") x0 => _,
            inout("x1") x1 => _,
            inout("x2") x2 => _,
            out("v0") c0,
            out("v1") c1,
            out("v2") c2,
            out("v3") c3,
            clobber_abi("C"),
        );
        std::mem::transmute::<[float32x4_t; 4], [[f32; 4]; 4]>([c0, c1, c2, c3])
    }
}

/// `objc_msgSend(x0, x1, x2, d0, d1, d2, d3)` — an object, a selector, an
/// integer, a `CGSize` and two `CGFloat`s, all floats in `d0`–`d3` — for a
/// method returning a `simd_float4x4`.
unsafe fn call_projection(x0: usize, x1: usize, x2: usize, d: [f64; 4]) -> [[f32; 4]; 4] {
    use std::arch::aarch64::float32x4_t;
    // each double in the low lane of its vector register
    let [mut c0, mut c1, mut c2, mut c3] = d.map(|x| unsafe { std::mem::transmute::<[f64; 2], float32x4_t>([x, 0.0]) });
    // SAFETY: as `call_int`, with the four doubles in d0–d3 (the low halves
    // of v0–v3, read before the results are written back over them).
    unsafe {
        std::arch::asm!(
            "blr {f}",
            f = in(reg) objc_msgSend as unsafe extern "C" fn(),
            inout("x0") x0 => _,
            inout("x1") x1 => _,
            inout("x2") x2 => _,
            inlateout("v0") c0,
            inlateout("v1") c1,
            inlateout("v2") c2,
            inlateout("v3") c3,
            clobber_abi("C"),
        );
        std::mem::transmute::<[float32x4_t; 4], [[f32; 4]; 4]>([c0, c1, c2, c3])
    }
}
