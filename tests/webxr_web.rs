//! WebXR's session in a real browser, on a headset the page fakes: what
//! `xr::web::XrSession` asks for, and what it reads each frame — the eyes,
//! the controller, hit tests, anchors, a buzz, the session's visibility, a
//! recentring, the frame rate and foveation.
//!
//! ```sh
//! FEATURES=xr TEST=webxr_web scripts/wasm-browser-test.sh
//! ```
//!
//! No browser here has a headset, so the page puts its own `navigator.xr`,
//! `XRWebGLLayer` and `XRRigidTransform` in place first: as small as the
//! session uses, and as the WebXR spec has them (matrices as
//! `Float32Array`s, column-major).
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;

use threers::xr::web::XrSession;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen(inline_js = r#"
const at = (x, y, z) => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]);
const proj = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,-1.0002,-1, 0,0,-0.20002,0]);
class Space extends EventTarget { constructor(m) { super(); this.m = m; } }
class Session extends EventTarget {
  constructor(mode, init) {
    super();
    this.mode = mode; this.init = init;
    this.environmentBlendMode = mode === 'immersive-ar' ? 'alpha-blend' : 'opaque';
    this.visibilityState = 'visible';
    this.supportedFrameRates = new Float32Array([72, 90]);
    this.frameRate = 72;
    this.queued = []; this.handle = 0; this.spaces = {}; this.pulses = []; this.renderState = {};
    const pulses = this.pulses;
    this.inputSources = [{
      handedness: 'right', targetRayMode: 'tracked-pointer', profiles: ['generic-trigger'],
      targetRaySpace: new Space(at(0.2, 1.4, -0.3)), gripSpace: new Space(at(0.2, 1.35, -0.25)),
      gamepad: { axes: [0, 0, 0.5, -0.25], buttons: [{ pressed: true, value: 1 }],
                 hapticActuators: [{ pulse(i, ms) { pulses.push([i, ms]); return Promise.resolve(true); } }] },
    }];
  }
  requestReferenceSpace(kind) { const s = new Space(at(0, 0, 0)); this.spaces[kind] = s; return Promise.resolve(s); }
  updateRenderState(s) { Object.assign(this.renderState, s); }
  requestAnimationFrame(cb) { this.queued.push(cb); return ++this.handle; }
  cancelAnimationFrame() {}
  end() { this.dispatchEvent(new Event('end')); return Promise.resolve(); }
  requestHitTestSource(o) { this.fromView = { space: o.space }; return Promise.resolve(this.fromView); }
  requestHitTestSourceForTransientInput(o) { this.fromTaps = { profile: o.profile }; return Promise.resolve(this.fromTaps); }
  updateTargetFrameRate(r) { this.frameRate = r; return Promise.resolve(); }
}
class Frame {
  constructor(session) { this.session = session; }
  getViewerPose(space) {
    return { transform: { matrix: at(0, 1.6, 0) }, views: [
      { eye: 'left', projectionMatrix: proj, transform: { matrix: at(-0.032, 1.6, 0) } },
      { eye: 'right', projectionMatrix: proj, transform: { matrix: at(0.032, 1.6, 0) } },
    ] };
  }
  getPose(space, base) { return space && space.m ? { transform: { matrix: space.m } } : null; }
  fillPoses() { return false; }
  getHitTestResults(source) {
    // the floor two metres ahead, then a table nearer behind it
    return source === this.session.fromView
      ? [{ getPose: () => ({ transform: { matrix: at(0, 0, -2) } }) }, { getPose: () => ({ transform: { matrix: at(0, 0.7, -3) } }) }]
      : [];
  }
  getHitTestResultsForTransientInput(source) {
    if (source !== this.session.fromTaps) return [];
    return [{ inputSource: this.session.inputSources[0], results: [{ getPose: () => ({ transform: { matrix: at(0.5, 0, -1) } }) }] }];
  }
  createAnchor(transform, space) {
    const anchor = { anchorSpace: new Space(transform.matrix), deleted: false, delete() { this.deleted = true; } };
    this.session.anchors = (this.session.anchors || []).concat([anchor]);
    return Promise.resolve(anchor);
  }
}
let current = null;
export function install_fake_xr() {
  Object.defineProperty(navigator, 'xr', { configurable: true, value: {
    isSessionSupported: () => Promise.resolve(true),
    requestSession: (mode, init) => { current = new Session(mode, init); return Promise.resolve(current); },
  } });
  globalThis.XRWebGLLayer = class {
    constructor(session, gl, init) { this.framebuffer = null; this.framebufferWidth = 200; this.framebufferHeight = 100; this.fixedFoveation = 0; current.layer = this; }
    getViewport(view) { return { x: view.eye === 'left' ? 0 : 100, y: 0, width: 100, height: 100 }; }
  };
  globalThis.XRRigidTransform = class {
    constructor(p, q) { this.position = p; this.orientation = q; this.matrix = at(p.x, p.y, p.z); }
  };
}
export function fake_frame(t) {
  const cbs = current.queued; current.queued = [];
  const frame = new Frame(current);
  for (const cb of cbs) cb(t, frame);
}
export function fake_session() { return current; }
export function fake_reset() { current.spaces['local-floor'].dispatchEvent(new Event('reset')); }
"#)]
extern "C" {
    fn install_fake_xr();
    fn fake_frame(t: f64);
    fn fake_session() -> JsValue;
    fn fake_reset();
}

fn get(o: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(o, &key.into()).unwrap()
}

async fn settle() {
    // promises the session awaits, resolved before the next frame
    for _ in 0..4 {
        let _ = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL)).await;
    }
}

#[derive(Default)]
struct Seen {
    head: Option<[f32; 16]>,
    views: usize,
    hits: Vec<[f32; 16]>,
    taps: Vec<(Option<usize>, Vec<[f32; 16]>)>,
    anchors: Vec<(u32, Option<[f32; 16]>)>,
    thumbstick: (f32, f32),
    buzzed: bool,
    anchored: Option<u32>,
}

#[wasm_bindgen_test]
async fn an_ar_session_finds_surfaces_and_keeps_anchors() {
    install_fake_xr();
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas: web_sys::HtmlCanvasElement = document.create_element("canvas").unwrap().unchecked_into();
    let overlay = document.create_element("div").unwrap();
    let session = XrSession::request_with_overlay("immersive-ar", canvas, "local-floor", 1.0, &[], Some(&overlay))
        .await
        .expect("a session");
    assert_eq!(session.mode(), "immersive-ar");
    // what it asked for: hit tests, anchors and the overlay, on the page's element
    let fake = fake_session();
    let init = get(&fake, "init");
    let asked: Vec<String> = js_sys::Array::from(&get(&init, "optionalFeatures")).iter().filter_map(|f| f.as_string()).collect();
    for f in ["local-floor", "hand-tracking", "hit-test", "anchors", "dom-overlay"] {
        assert!(asked.iter().any(|a| a == f), "{f} in {asked:?}");
    }
    assert!(get(&get(&init, "domOverlay"), "root") == JsValue::from(overlay.clone()), "the overlay's root");
    session.start_hit_test().await.expect("hit tests from the view");
    session.start_tap_hit_test("generic-touchscreen").await.expect("hit tests from taps");

    let seen = Rc::new(RefCell::new(Seen::default()));
    let (s2, seen2) = (session.clone(), seen.clone());
    session.set_frame_callback(Some(Box::new(move |_, _| {
        let mut seen = seen2.borrow_mut();
        seen.head = s2.head();
        seen.views = s2.views().len();
        seen.hits = s2.hits();
        seen.taps = s2.tap_hits();
        if seen.anchored.is_none() {
            if let Some(&floor) = seen.hits.first() {
                seen.anchored = Some(s2.add_anchor(floor).expect("an anchor, in a frame"));
            }
        }
        seen.anchors = s2.anchors();
        let input = threers::xr::XrSource::inputs(&s2).into_iter().next();
        seen.thumbstick = input.map_or((0.0, 0.0), |i| i.thumbstick());
        if !seen.buzzed {
            seen.buzzed = s2.pulse(0, 0.5, 30.0);
        }
    })));
    for t in 1..=3 {
        fake_frame(t as f64 * 16.0);
        settle().await;
    }
    let seen = seen.borrow();
    assert_eq!(seen.head.unwrap()[13], 1.6, "the head");
    assert_eq!(seen.views, 2, "both eyes");
    assert_eq!(seen.hits.len(), 2, "the floor, then the table");
    assert_eq!((seen.hits[0][12], seen.hits[0][13], seen.hits[0][14]), (0.0, 0.0, -2.0));
    assert_eq!(seen.taps.len(), 1);
    assert_eq!(seen.taps[0].0, Some(0), "the tap's input, by its slot");
    assert_eq!(seen.taps[0].1[0][12], 0.5);
    // made in the first frame, there with its pose by the next
    assert_eq!(seen.anchors.len(), 1);
    let pose = seen.anchors[0].1.expect("its pose");
    assert_eq!((pose[12], pose[13], pose[14]), (0.0, 0.0, -2.0), "where the floor was hit");
    assert_eq!(seen.thumbstick, (0.5, -0.25));
    assert!(seen.buzzed);
    let pulses = js_sys::Array::from(&get(&fake, "pulses"));
    assert_eq!(pulses.length(), 1, "one buzz");
    assert_eq!(js_sys::Array::from(&pulses.get(0)).get(0).as_f64(), Some(0.5));
    // let go: the browser's anchor is deleted
    session.remove_anchor(seen.anchored.unwrap());
    let anchors = js_sys::Array::from(&get(&fake, "anchors"));
    assert_eq!(get(&anchors.get(0), "deleted").as_bool(), Some(true));
    assert!(session.anchors().is_empty());
}

#[wasm_bindgen_test]
async fn a_session_says_when_it_is_covered_recentred_and_how_fast() {
    install_fake_xr();
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas: web_sys::HtmlCanvasElement = document.create_element("canvas").unwrap().unchecked_into();
    let session = XrSession::request("immersive-vr", canvas, "local-floor", 1.0, &[]).await.expect("a session");
    let fake = fake_session();
    // a VR session asks for neither hit tests nor anchors
    let asked: Vec<String> = js_sys::Array::from(&get(&get(&fake, "init"), "optionalFeatures")).iter().filter_map(|f| f.as_string()).collect();
    assert!(!asked.iter().any(|a| a == "hit-test" || a == "anchors"), "{asked:?}");
    assert_eq!(session.visibility(), "visible");
    js_sys::Reflect::set(&fake, &"visibilityState".into(), &"visible-blurred".into()).unwrap();
    assert_eq!(session.visibility(), "visible-blurred", "a system menu over it");
    assert!(!session.take_recentered());
    fake_reset();
    assert!(session.take_recentered(), "recentred");
    assert!(!session.take_recentered(), "once");
    assert_eq!(session.frame_rates(), vec![72.0, 90.0]);
    assert!(session.set_frame_rate(90.0));
    assert_eq!(get(&fake, "frameRate").as_f64(), Some(90.0));
    session.set_foveation(0.75);
    assert_eq!(get(&get(&fake, "layer"), "fixedFoveation").as_f64(), Some(0.75));
    // no hits before they are started, nor outside a frame
    assert!(session.hits().is_empty() && session.tap_hits().is_empty());
    assert!(session.add_anchor([0.0; 16]).is_err(), "an anchor outside a frame");
}
