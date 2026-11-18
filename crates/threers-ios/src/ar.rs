//! ARKit on iOS, from Rust: a world-tracking session, its camera feed shown
//! behind the app's layer, and the camera's pose and projection each frame.
//!
//! The feed is ARKit's own view (`ARSCNView`, with an empty scene): it runs
//! the session and draws the camera image the way ARKit means it to be
//! drawn — cropped, rotated and timed — with nothing copied. It goes into
//! the layer's view's superview, beneath it, so the shell needs a container
//! view around the layer's view, which `app/Harness.swift` has.
//!
//! Two devices can share one room (ARKit's collaborative session): each
//! hands the other its [collaboration data](ArSession::take_collaboration)
//! over any link, and takes the other's in; once their maps of the room
//! merge, an anchor one [pins](ArSession::pin) shows up in the other's
//! [`anchors`](ArSession::anchors), where it is in *its* world.
//!
//! The Objective-C runtime is called directly (objc2's `msg_send!`), with two
//! exceptions: ARKit's camera returns `simd_float4x4`, which Apple's ARM64
//! convention passes back in `q0`–`q3`, and Rust has no type FFI returns
//! that way (threers' `metal::visionos` explains). Those two calls are made
//! as C makes them, by a branch in assembly that reads the registers back.

use std::ffi::c_void;

use std::sync::{Arc, Mutex};

use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyClass, AnyObject, Bool, NSObject, NSObjectProtocol};
use objc2::{AnyThread, DefinedClass, define_class, msg_send, sel};
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
/// `ARPlaneDetectionHorizontal`
const HORIZONTAL_PLANES: usize = 1;
/// `ARRaycastTargetExistingPlaneGeometry`, `…EstimatedPlane`
const FOUND_PLANE: isize = 0;
const ESTIMATED_PLANE: isize = 2;
/// `ARRaycastTargetAlignmentHorizontal`
const HORIZONTAL: isize = 0;

/// Whether this device can track the world (not the simulator, not the
/// oldest devices).
pub fn supported() -> bool {
    let Some(class) = AnyClass::get(c"ARWorldTrackingConfiguration") else {
        return false;
    };
    let yes: Bool = unsafe { msg_send![class, isSupported] };
    yes.as_bool()
}

/// `ARCollaborationDataPriorityCritical`
const CRITICAL: isize = 0;

/// Collaboration data the session has made, archived, and whether each must
/// arrive (else a newer will do).
type Outbox = Arc<Mutex<Vec<(Vec<u8>, bool)>>>;

define_class!(
    // SAFETY: NSObject has no subclassing requirements; no Drop.
    #[unsafe(super(NSObject))]
    #[ivars = Outbox]
    struct SessionDelegate;

    unsafe impl NSObjectProtocol for SessionDelegate {}

    // ARSessionDelegate
    impl SessionDelegate {
        #[unsafe(method(session:didOutputCollaborationData:))]
        fn collaboration(&self, _session: &AnyObject, data: &AnyObject) {
            let critical: isize = unsafe { msg_send![data, priority] };
            if let Some(bytes) = archive(data)
                && let Ok(mut out) = self.ivars().lock()
            {
                // Optional data comes every frame or so: keep a backlog short.
                if out.len() > 64 {
                    out.retain(|(_, critical)| *critical);
                }
                out.push((bytes, critical == CRITICAL));
            }
        }
    }
);

/// An object archived, as `NSKeyedArchiver` with secure coding archives it.
fn archive(object: &AnyObject) -> Option<Vec<u8>> {
    unsafe {
        let mut error: *mut AnyObject = std::ptr::null_mut();
        let data: Option<Retained<AnyObject>> = msg_send![
            AnyClass::get(c"NSKeyedArchiver")?,
            archivedDataWithRootObject: object,
            requiringSecureCoding: Bool::YES,
            error: &mut error
        ];
        let data = data?;
        let bytes: *const c_void = msg_send![&*data, bytes];
        let length: usize = msg_send![&*data, length];
        (!bytes.is_null()).then(|| std::slice::from_raw_parts(bytes as *const u8, length).to_vec())
    }
}

/// An object of `class` from `bytes` [`archive`] made.
fn unarchive(class: &AnyClass, bytes: &[u8]) -> Option<Retained<AnyObject>> {
    unsafe {
        let data: Retained<AnyObject> = msg_send![
            AnyClass::get(c"NSData")?,
            dataWithBytes: bytes.as_ptr() as *const c_void,
            length: bytes.len()
        ];
        let mut error: *mut AnyObject = std::ptr::null_mut();
        msg_send![
            AnyClass::get(c"NSKeyedUnarchiver")?,
            unarchivedObjectOfClass: class,
            fromData: &*data,
            error: &mut error
        ]
    }
}

fn ns_string(text: &str) -> Option<Retained<AnyObject>> {
    unsafe {
        let made: Allocated<AnyObject> = msg_send![AnyClass::get(c"NSString")?, alloc];
        msg_send![
            made,
            initWithBytes: text.as_ptr() as *const c_void,
            length: text.len(),
            encoding: 4usize
        ]
    }
}

fn rust_string(ns: &AnyObject) -> String {
    unsafe {
        let utf8: *const std::ffi::c_char = msg_send![ns, UTF8String];
        if utf8.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(utf8).to_string_lossy().into_owned()
    }
}

/// A named anchor in the session: where it is (its anchor-to-world
/// transform, column-major) and whether this device placed it.
#[derive(Debug, Clone, PartialEq)]
pub struct Pin {
    pub name: String,
    pub transform: [f32; 16],
    pub mine: bool,
}

/// A running session and the view showing its camera.
pub struct ArSession {
    /// The `ARSCNView`.
    view: Retained<AnyObject>,
    config: Retained<AnyObject>,
    delegate: Retained<SessionDelegate>,
    collaborating: bool,
    /// The anchors this device pinned, by name.
    pinned: Vec<(String, Retained<AnyObject>)>,
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
            let config: Option<Retained<AnyObject>> =
                msg_send![AnyClass::get(c"ARWorldTrackingConfiguration")?, new];
            let config = config?;
            // +Y against gravity, the origin where the session starts
            let _: () = msg_send![&*config, setWorldAlignment: 0isize];
            // Tables and floors, for `surface`.
            let _: () = msg_send![&*config, setPlaneDetection: HORIZONTAL_PLANES];
            let session: Retained<AnyObject> = msg_send![&*feed, session];
            let delegate = SessionDelegate::alloc().set_ivars(Outbox::default());
            let delegate: Retained<SessionDelegate> = msg_send![super(delegate), init];
            let _: () = msg_send![&*session, setDelegate: &*delegate];
            let _: () = msg_send![&*session, runWithConfiguration: &*config];
            Some(ArSession {
                view: feed,
                config,
                delegate,
                collaborating: false,
                pinned: Vec::new(),
            })
        }
    }

    fn session(&self) -> Retained<AnyObject> {
        unsafe { msg_send![&*self.view, session] }
    }

    /// Share the room with other devices, or stop: the session runs on,
    /// keeping what it knows.
    pub fn collaborate(&mut self, on: bool) {
        if on == self.collaborating {
            return;
        }
        self.collaborating = on;
        unsafe {
            let _: () = msg_send![&*self.config, setCollaborationEnabled: Bool::new(on)];
            let _: () = msg_send![&*self.session(), runWithConfiguration: &*self.config];
        }
        if !on && let Ok(mut out) = self.delegate.ivars().lock() {
            out.clear();
        }
    }

    /// The collaboration data made since last asked, for the other devices,
    /// each with whether it must arrive.
    pub fn take_collaboration(&mut self) -> Vec<(Vec<u8>, bool)> {
        self.delegate
            .ivars()
            .lock()
            .map(|mut out| std::mem::take(&mut *out))
            .unwrap_or_default()
    }

    /// Another device's collaboration data, as its `take_collaboration`
    /// gave it.
    pub fn give_collaboration(&mut self, bytes: &[u8]) {
        if !self.collaborating {
            return;
        }
        let Some(class) = AnyClass::get(c"ARCollaborationData") else { return };
        if let Some(data) = unarchive(class, bytes) {
            unsafe {
                let _: () = msg_send![&*self.session(), updateWithCollaborationData: &*data];
            }
        }
    }

    /// Put an anchor named `name` at `transform` (anchor-to-world,
    /// column-major), in place of one this device pinned before by that name.
    /// Collaborating, the other devices see it too.
    pub fn pin(&mut self, name: &str, transform: [f32; 16]) {
        let session = self.session();
        unsafe {
            if let Some(i) = self.pinned.iter().position(|(n, _)| n == name) {
                let (_, old) = self.pinned.remove(i);
                let _: () = msg_send![&*session, removeAnchor: &*old];
            }
            let (Some(class), Some(label)) = (AnyClass::get(c"ARAnchor"), ns_string(name)) else {
                return;
            };
            let made: Allocated<AnyObject> = msg_send![class, alloc];
            // `init` consumes it: handed on, not released here.
            let raw = Allocated::as_ptr(&made) as usize;
            std::mem::forget(made);
            let columns: [[f32; 4]; 4] = std::array::from_fn(|c| std::array::from_fn(|r| transform[c * 4 + r]));
            let anchor = call_with_matrix(
                raw,
                sel_bits(sel!(initWithName:transform:)),
                Retained::as_ptr(&label) as usize,
                columns,
            );
            // SAFETY: an `init` returns its object retained, or nil.
            let Some(anchor) = Retained::from_raw(anchor as *mut AnyObject) else { return };
            let _: () = msg_send![&*session, addAnchor: &*anchor];
            self.pinned.push((name.to_owned(), anchor));
        }
    }

    /// The named anchors in the session now, this device's and the others',
    /// and how many other devices it is sharing the room with.
    pub fn anchors(&self) -> (Vec<Pin>, usize) {
        let mut pins = Vec::new();
        let mut others = 0;
        unsafe {
            let session = self.session();
            let frame: Option<Retained<AnyObject>> = msg_send![&*session, currentFrame];
            let Some(frame) = frame else { return (pins, 0) };
            let me: Option<Retained<AnyObject>> = msg_send![&*session, identifier];
            let anchors: Retained<AnyObject> = msg_send![&*frame, anchors];
            let count: usize = msg_send![&*anchors, count];
            let participant = AnyClass::get(c"ARParticipantAnchor");
            for i in 0..count {
                let anchor: Retained<AnyObject> = msg_send![&*anchors, objectAtIndex: i];
                if let Some(class) = participant {
                    let is: Bool = msg_send![&*anchor, isKindOfClass: class];
                    if is.as_bool() {
                        others += 1;
                        continue;
                    }
                }
                let name: Option<Retained<AnyObject>> = msg_send![&*anchor, name];
                let Some(name) = name else { continue };
                let from: Option<Retained<AnyObject>> = msg_send![&*anchor, sessionIdentifier];
                let mine = match (&from, &me) {
                    (Some(from), Some(me)) => {
                        let same: Bool = msg_send![&**from, isEqual: &**me];
                        same.as_bool()
                    }
                    // No session named: added here.
                    (None, _) => true,
                    _ => false,
                };
                let m = call_int(Retained::as_ptr(&anchor) as usize, sel_bits(sel!(transform)), 0);
                pins.push(Pin {
                    name: rust_string(&name),
                    transform: flat(m),
                    mine,
                });
            }
        }
        (pins, others)
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
            let view = call_int(
                cam,
                sel_bits(sel!(viewMatrixForOrientation:)),
                orientation as usize,
            );
            let projection = call_projection(
                cam,
                sel_bits(sel!(projectionMatrixForOrientation:viewportSize:zNear:zFar:)),
                orientation as usize,
                [
                    size.0.max(1) as f64,
                    size.1.max(1) as f64,
                    near as f64,
                    far as f64,
                ],
            );
            let world = threers::math::Matrix4 {
                elements: flat(view),
            }
            .invert()
            .elements;
            Some((world, flat(projection)))
        }
    }

    /// Where a horizontal surface (a table, the floor) is under the view's
    /// middle: a point in the world, metres. A plane ARKit has found if the
    /// middle is on one, else its estimate; `None` before it has either.
    pub fn surface(&self) -> Option<[f32; 3]> {
        unsafe {
            let bounds: CGRect = msg_send![&*self.view, bounds];
            let middle = objc2_core_foundation::CGPoint {
                x: bounds.origin.x + bounds.size.width / 2.0,
                y: bounds.origin.y + bounds.size.height / 2.0,
            };
            let session: Retained<AnyObject> = msg_send![&*self.view, session];
            for target in [FOUND_PLANE, ESTIMATED_PLANE] {
                let query: Option<Retained<AnyObject>> = msg_send![
                    &*self.view,
                    raycastQueryFromPoint: middle,
                    allowingTarget: target,
                    alignment: HORIZONTAL
                ];
                let Some(query) = query else { continue };
                let hits: Retained<AnyObject> = msg_send![&*session, raycast: &*query];
                let first: Option<Retained<AnyObject>> = msg_send![&*hits, firstObject];
                let Some(first) = first else { continue };
                let hit = Retained::as_ptr(&first) as usize;
                let m = call_int(hit, sel_bits(sel!(worldTransform)), 0);
                return Some([m[3][0], m[3][1], m[3][2]]);
            }
            None
        }
    }

    /// The window scene's interface orientation, portrait when unknown.
    fn orientation(&self) -> isize {
        unsafe {
            let window: Option<Retained<AnyObject>> = msg_send![&*self.view, window];
            let Some(window) = window else {
                return PORTRAIT;
            };
            let scene: Option<Retained<AnyObject>> = msg_send![&*window, windowScene];
            let Some(scene) = scene else { return PORTRAIT };
            let o: isize = msg_send![&*scene, interfaceOrientation];
            if o == 0 {
                PORTRAIT
            } else {
                o
            }
        }
    }
}

impl Drop for ArSession {
    fn drop(&mut self) {
        unsafe {
            let session: Retained<AnyObject> = msg_send![&*self.view, session];
            let none: *mut AnyObject = std::ptr::null_mut();
            let _: () = msg_send![&*session, setDelegate: none];
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

/// `objc_msgSend(x0, x1, x2, v0–v3)` for a method taking an object and a
/// `simd_float4x4` (its columns in `v0`–`v3`) and returning an object.
unsafe fn call_with_matrix(x0: usize, x1: usize, x2: usize, m: [[f32; 4]; 4]) -> usize {
    use std::arch::aarch64::float32x4_t;
    let [c0, c1, c2, c3] = unsafe { std::mem::transmute::<[[f32; 4]; 4], [float32x4_t; 4]>(m) };
    let out: usize;
    // SAFETY: the columns go in v0–v3 as C passes a homogeneous vector
    // aggregate; the object comes back in x0.
    unsafe {
        std::arch::asm!(
            "blr {f}",
            f = in(reg) objc_msgSend as unsafe extern "C" fn(),
            inout("x0") x0 => out,
            inout("x1") x1 => _,
            inout("x2") x2 => _,
            inout("v0") c0 => _,
            inout("v1") c1 => _,
            inout("v2") c2 => _,
            inout("v3") c3 => _,
            clobber_abi("C"),
        );
    }
    out
}

/// `objc_msgSend(x0, x1, x2, d0, d1, d2, d3)` — an object, a selector, an
/// integer, a `CGSize` and two `CGFloat`s, all floats in `d0`–`d3` — for a
/// method returning a `simd_float4x4`.
unsafe fn call_projection(x0: usize, x1: usize, x2: usize, d: [f64; 4]) -> [[f32; 4]; 4] {
    use std::arch::aarch64::float32x4_t;
    // each double in the low lane of its vector register
    let [mut c0, mut c1, mut c2, mut c3] =
        d.map(|x| unsafe { std::mem::transmute::<[f64; 2], float32x4_t>([x, 0.0]) });
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
