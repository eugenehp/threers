//! An [`App`] in a visionOS immersive space, both eyes, drawn by threers'
//! Metal renderer ([`ImmersiveRenderer`]) into the compositor's textures.
//!
//! The space is the reference space: the floor under the viewer where it
//! opened, metres, +Y up. The head comes from ARKit's world tracking; a
//! look-and-pinch (the shell's `onSpatialEvent`, through
//! [`threers_ios_spatial_event`]) is a transient pointer along the ray it
//! selected, for as long as it lasts, as on the web (WebXR's
//! `transient-pointer`). The room is mixed: the scene is drawn over it,
//! with no background ([`Frame::see_through`]).
//!
//! The app's UI ([`App::ui`] for [`Place::Panel`]) stands on a panel ahead
//! of the viewer ([`App::panel`]): painted where it changed, put on a quad
//! in the scene, and pinched as a touch screen is touched.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use threers::core::{Mesh, Object3D, ObjectId, ObjectKind};
use threers::geometries::PlaneGeometry;
use threers::materials::{BasicMaterial, Material};
use threers::math::{Color, Matrix4, Vector3};
use threers::metal::visionos::ImmersiveRenderer;
use threers::scene::Scene;
use threers::textures::{Texture, TextureFormat};
use threers::ui::input::{Event, TouchPhase};
use threers::xr::{Frame, InputSlots, Panel, PanelInput, TargetRayMode, XrEvent, XrEventKind, XrInput, ray_pose};

use crate::surface_ui::SurfaceUi;
use crate::{App, Place, Surface};

/// A look-and-pinch from the shell: which one, its phase (0 began, 1
/// active — a start the first time an id is seen —, 2 ended, 3
/// cancelled), and its selection ray in the space.
#[derive(Debug, Clone, Copy)]
struct Spatial {
    id: u64,
    phase: u32,
    origin: [f32; 3],
    direction: [f32; 3],
}

static SPATIAL: Mutex<Vec<Spatial>> = Mutex::new(Vec::new());

/// From the shell's `onSpatialEvent`: a pinch `id` began (`phase` 0), moved
/// (1), ended (2) or was cancelled (3), with its selection ray (`origin`,
/// `direction`) in the immersive space. Any thread.
#[unsafe(no_mangle)]
pub extern "C" fn threers_ios_spatial_event(id: u64, phase: u32, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) {
    if let Ok(mut q) = SPATIAL.lock() {
        q.push(Spatial { id, phase, origin: [ox, oy, oz], direction: [dx, dy, dz] });
    }
}

/// Pinches as inputs that last while they do.
#[derive(Default)]
struct Pinches {
    slots: InputSlots<u64>,
    rays: HashMap<u64, [f32; 16]>,
    /// Ended last frame: disconnected at the start of this one.
    leaving: Vec<u64>,
}

impl Pinches {
    /// The pinches since the last frame: the inputs now and what they did.
    fn take(&mut self) -> (Vec<XrInput>, Vec<XrEvent>) {
        let mut events = Vec::new();
        for id in std::mem::take(&mut self.leaving) {
            if let Some(slot) = self.slots.disconnect(&id) {
                events.push(XrEvent { kind: XrEventKind::Disconnected, slot });
            }
            self.rays.remove(&id);
        }
        for e in SPATIAL.lock().map(|mut q| std::mem::take(&mut *q)).unwrap_or_default() {
            let (o, d) = (e.origin, e.direction);
            self.rays.insert(e.id, ray_pose(Vector3::new(o[0], o[1], o[2]), Vector3::new(d[0], d[1], d[2])));
            // SwiftUI says only active, ended or cancelled: an id's first
            // active is its start
            let starts = e.phase == 0 || (e.phase == 1 && self.slots.slot_of(&e.id).is_none());
            if starts {
                let slot = self.slots.connect(e.id);
                events.push(XrEvent { kind: XrEventKind::Connected, slot });
                events.push(XrEvent { kind: XrEventKind::SelectStart, slot });
            } else if e.phase >= 2 {
                if let Some(slot) = self.slots.slot_of(&e.id) {
                    events.push(XrEvent { kind: XrEventKind::SelectEnd, slot });
                    if e.phase == 2 {
                        events.push(XrEvent { kind: XrEventKind::Select, slot });
                    }
                }
                self.leaving.push(e.id);
            }
        }
        let inputs = self
            .rays
            .iter()
            .filter_map(|(id, ray)| {
                let slot = self.slots.slot_of(id)?;
                Some(XrInput { slot, target_ray_mode: TargetRayMode::TransientPointer, ray: Some(*ray), ..Default::default() })
            })
            .collect();
        (inputs, events)
    }
}

/// The UI's panel in the room: where it stands, its pixels, and the quad
/// that shows them.
struct RoomPanel {
    panel: Panel,
    ui: SurfaceUi,
    quad: Option<ObjectId>,
    /// Device pixels a UI pixel on its texture.
    scale: f32,
}

impl RoomPanel {
    fn new(spec: crate::PanelSpec) -> Self {
        let mut panel = Panel::new(spec.width, spec.height);
        panel.metres_per_pixel = spec.metres_per_pixel;
        panel.distance = spec.distance;
        panel.drop = spec.drop;
        panel.side = spec.side;
        // below the eyes, turned up to face them
        panel.tilt = spec.drop.atan2(spec.distance.hypot(spec.side));
        RoomPanel { panel, ui: SurfaceUi::default(), quad: None, scale: 2.0 }
    }

    /// Pinches on it to the UI, then its pixels onto its quad. Whether a
    /// pinch landed on it.
    fn step<A: App>(&mut self, app: &mut A, scene: &mut Scene, frame: &Frame, origin: &Matrix4) -> bool {
        let (inputs, _pointers) = self.panel.step(frame);
        let mut took = false;
        for i in inputs {
            let (phase, id, x, y) = match i {
                PanelInput::TouchStart(id, x, y) => (TouchPhase::Start, id, x, y),
                PanelInput::TouchMove(id, x, y) => (TouchPhase::Move, id, x, y),
                PanelInput::TouchEnd(id, x, y) => (TouchPhase::End, id, x, y),
                PanelInput::TouchCancel(id, x, y) => (TouchPhase::Cancel, id, x, y),
                // no mouse or controllers on visionOS
                _ => continue,
            };
            app.ui_event(Place::Panel, Event::Touch { id, phase, x, y, time: frame.time }, frame.time);
            took = true;
        }
        let (w, h) = self.panel.size;
        let surface = Surface { place: Place::Panel, width: w, height: h, scale: self.scale, insets: [0.0; 4], see_through: true };
        let (shown, changed) = self.ui.update(app, &surface);
        let plane = self.panel.plane();
        let (Some(plane), true) = (plane, shown) else {
            if let Some(q) = self.quad.and_then(|q| scene.get_mut(q)) {
                q.visible = false;
            }
            return took;
        };
        let quad = *self.quad.get_or_insert_with(|| {
            let mut m = BasicMaterial::new(Color::new(1.0, 1.0, 1.0));
            m.transparent = true;
            m.side = 2;
            let mut o = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.0, 1.0), Material::Basic(m)));
            o.name = "threers-ios panel".into();
            // drawn after the scene, over what is behind it
            o.render_order = 1000;
            scene.add(o)
        });
        let Some(o) = scene.get_mut(quad) else { return took };
        o.visible = true;
        // the plane's axes, sized: right and up span it, it faces back along their cross
        let (r, u) = (plane.right * (plane.half.0 * 2.0), plane.up * (plane.half.1 * 2.0));
        let n = plane.right.cross(plane.up).normalize();
        let c = plane.center;
        let m = origin.multiply(&Matrix4 { elements: [r.x, r.y, r.z, 0.0, u.x, u.y, u.z, 0.0, n.x, n.y, n.z, 0.0, c.x, c.y, c.z, 1.0] });
        let (p, q, s) = m.decompose();
        (o.position, o.quaternion, o.scale) = (p, q, s);
        if changed && let ObjectKind::Mesh(mesh) = &mut o.kind {
            let (tw, th) = self.ui.pixels.size();
            let texture = Texture::new(tw, th, TextureFormat::Rgba8UnormSrgb, self.ui.pixels.pixels().to_vec());
            let mut m = BasicMaterial::new(Color::new(1.0, 1.0, 1.0));
            m.map = Some(Arc::new(texture));
            // the UI's own colours, not the scene's light curve
            m.tone_mapped = false;
            m.transparent = true;
            m.side = 2;
            mesh.material = Material::Basic(m).into();
        }
        took
    }
}

/// Runs `A` in the immersive space whose layer renderer is
/// `layer_renderer` (a `cp_layer_renderer_t`) until the space closes. Call
/// it on a thread of its own (it waits for each frame), as the shell does.
///
/// # Safety
/// `layer_renderer` must be the `CompositorLayer`'s live layer renderer.
pub unsafe fn run<A: App>(layer_renderer: *mut c_void) {
    let mut renderer = match unsafe { ImmersiveRenderer::new(layer_renderer) } {
        Ok(r) => r,
        Err(e) => {
            eprintln!("threers: no immersive space: {e}");
            return;
        }
    };
    let mut scene = Scene::new();
    let mut app = A::setup(&mut scene);
    let lens = app.lens();
    renderer.set_depth_range(lens.near, lens.far);
    let origin = app.origin();
    renderer.set_origin(origin);
    let mut panel = RoomPanel::new(app.panel());
    let mut pinches = Pinches::default();
    let start = Instant::now();
    let mut last = 0.0;
    loop {
        let (tone, exposure) = app.tone_mapping();
        renderer.renderer_mut().set_tone_mapping(tone, exposure);
        let drawn = renderer.render_frame_with(&mut scene, |scene, f| {
            let time = start.elapsed().as_secs_f64();
            let (inputs, events) = pinches.take();
            let frame = Frame { time, dt: (time - last).clamp(0.0, 0.25), head: f.head.map(|h| h.elements), inputs, events, see_through: true };
            last = time;
            // a pinch on the panel is the UI's, not the scene's
            let on_panel = panel.step(&mut app, scene, &frame, &origin);
            let frame = if on_panel { Frame { inputs: vec![], events: vec![], ..frame } } else { frame };
            app.update(scene, &frame);
            // The room shows through: clear to nothing. The compositor takes
            // the drawable as premultiplied, so a colour at alpha 0 would
            // still be added over the passthrough, as a veil.
            scene.background = threers::math::Color::new(0.0, 0.0, 0.0);
            scene.background_alpha = 0.0;
        });
        match drawn {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => {
                eprintln!("threers: immersive frame: {e}");
                break;
            }
        }
    }
}
