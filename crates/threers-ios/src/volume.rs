//! An [`App`] in a visionOS **volume**: RealityKit draws it, in the shared
//! space beside other apps — no immersive space, no CompositorServices.
//!
//! RealityKit has no C API, so the shell builds the entities
//! (`app/Harness.swift`) and this tells it what to build: the app runs here
//! as it does everywhere ([`App::setup`], [`App::update`] each frame, no
//! head), and after each frame its scene is read out as a list of meshes —
//! each one's geometry (positions, normals, vertex colours, indices), its
//! material (colour, opacity, roughness, metalness, sides) and its world
//! matrix, with keys that change when the geometry or the material does,
//! so the shell rebuilds only what changed.
//!
//! The app's own 2D UI comes out as an image ([`Volume::ui`]): its
//! [`App::ui`] for [`Place::Image`], painted on the CPU, which the shell
//! shows beside the volume, sending the taps on it back
//! ([`crate::threers_ios_tap`]). Its controls come out too
//! ([`Volume::controls`]: where each one is, its shape, role, name and
//! state, from `threers_ui::native`), for the shell to lay a native view
//! over each — the system's look highlight and VoiceOver need its own
//! views — and a press on one comes back by its key
//! ([`crate::threers_ios_press`]).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use threers::core::{LineSegments as ObjectLines, ObjectId, ObjectKind, Points as ObjectPoints};
use threers::materials::Material;
use threers::math::{Matrix4, Vector3};
use threers::scene::Scene;
use threers::ui::input::{Event, PointerButton, TouchPhase};
use threers::ui::native::{self, Control};
use threers::xr::Frame;

use crate::surface_ui::SurfaceUi;
use crate::{App, Place, Surface};

/// One mesh, as the shell reads it: 32 floats.
///
/// `[0..16]` the world matrix (column-major, the volume's frame: see
/// [`Volume::root`]), `[16..20]` the colour (sRGB, straight alpha),
/// `[20]` roughness, `[21]` metalness, `[22]` flags: 1 visible, 2 vertex
/// colours, 4 blended, 8 double-sided; the rest zero.
pub const OBJECT_FLOATS: usize = 32;

/// A geometry's identity for the shell: which one, and its version — an
/// edit in place (`Arc::make_mut`: another surface, repainted colours)
/// keeps the pointer, and must still rebuild.
fn geometry_key(g: &Arc<threers::core::BufferGeometry>) -> u64 {
    (Arc::as_ptr(g) as u64) ^ (u64::from(g.geometry_version) << 40).rotate_left(7)
}

/// A mesh's place in the list, kept across frames.
struct Item {
    id: ObjectId,
    key: u64,
}

/// Interleaved for the shell: what [`Volume::geometry`] points into.
#[derive(Default)]
struct Geometry {
    positions: Vec<f32>,
    normals: Vec<f32>,
    colors: Vec<f32>,
    /// floats a colour (3 or 4), or 0 without
    color_size: u32,
    indices: Vec<u32>,
    /// a sprite's pixels, for [`Volume::texture`]
    texture: Vec<u8>,
    /// texture coordinates (2 a vertex), for [`Volume::uvs`]
    uvs: Vec<f32>,
}

pub struct Volume<A: App> {
    scene: Scene,
    app: A,
    items: Vec<Item>,
    keys: HashMap<ObjectId, u64>,
    next_key: u64,
    geometry: Geometry,
    /// Turned about the vertical by the shell's drags, radians.
    yaw: f32,
    /// The app's UI beside the volume ([`Place::Image`], the card) and
    /// under it ([`Place::Toolbar`]): [`SURFACES`] in order.
    surfaces: [Strip; 2],
    start: Instant,
    last: f64,
    /// Shader materials RealityKit cannot run, drawn by threers for it
    /// ([`bake`](crate::bake)), by mesh, while their key holds.
    baker: crate::bake::Baker,
    baked: HashMap<ObjectId, crate::bake::Baked>,
}

/// Where the volume's UI surfaces are shown, by the index the shell
/// names them with: the card beside the volume, the toolbar under it.
pub const SURFACES: [Place; 2] = [Place::Image, Place::Toolbar];

/// One of the volume's UI surfaces, as last handed to the shell.
struct Strip {
    ui: SurfaceUi,
    /// Its image has not been handed over yet (or it was hidden).
    fresh: bool,
    /// The shell has been told it is hidden (an empty image), since it
    /// was last shown.
    told_hidden: bool,
    /// Its controls as last laid out, and the version of them last handed
    /// over (with their JSON, kept for the shell to read).
    controls: Vec<Control>,
    sent: Option<u64>,
    json: String,
}

impl Default for Strip {
    fn default() -> Self {
        Strip { ui: SurfaceUi::default(), fresh: true, told_hidden: false, controls: vec![], sent: None, json: String::new() }
    }
}

impl<A: App> Default for Volume<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: App> Volume<A> {
    pub fn new() -> Self {
        let mut scene = Scene::new();
        let app = A::setup(&mut scene);
        Volume {
            scene,
            app,
            items: vec![],
            keys: HashMap::new(),
            next_key: 1,
            geometry: Geometry::default(),
            yaw: 0.0,
            surfaces: [Strip::default(), Strip::default()],
            start: Instant::now(),
            last: 0.0,
            baker: Default::default(),
            baked: HashMap::new(),
        }
    }

    pub fn app(&mut self) -> &mut A {
        &mut self.app
    }

    pub fn scene(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// Turn the model about the volume's vertical by `radians` (a drag).
    pub fn turn(&mut self, radians: f32) {
        if radians.is_finite() {
            self.yaw += radians;
        }
    }

    /// Update the app and list the meshes it shows: how many.
    pub fn frame(&mut self) -> usize {
        let time = self.start.elapsed().as_secs_f64();
        let dt = (time - self.last).clamp(0.0, 0.25);
        self.last = time;
        // the app asked for its home view: undo the shell's turning too
        if crate::take_camera_requests().iter().any(|r| matches!(r, crate::CameraRequest::Home)) {
            self.yaw = 0.0;
        }
        let frame = Frame { time, dt, head: None, inputs: vec![], events: vec![], see_through: true };
        self.app.update(&mut self.scene, &frame);
        self.scene.update_world();
        self.items.clear();
        for (id, o) in self.scene.arena.nodes.iter() {
            // meshes; line sets (as thin tubes) and sprites (as billboards)
            if !matches!(o.kind, ObjectKind::Mesh(_) | ObjectKind::LineSegments(_) | ObjectKind::Sprite(_) | ObjectKind::Points(_)) {
                continue;
            }
            let next = &mut self.next_key;
            let key = *self.keys.entry(id).or_insert_with(|| {
                *next += 1;
                *next - 1
            });
            self.items.push(Item { id, key });
        }
        self.items.sort_by_key(|i| i.key);
        // the bakes of meshes gone
        let scene = &self.scene;
        self.baked.retain(|id, _| scene.get(*id).is_some());
        self.items.len()
    }

    /// The model's frame in the volume, column-major: the app's
    /// [`focus`](App::focus) at the volume's centre, turned by the drags,
    /// and the radius it has (the shell scales it to fit).
    pub fn root(&self) -> ([f32; 16], f32) {
        let (centre, radius) = self.app.focus().unwrap_or((Vector3::ZERO, 1.0));
        let turn = Matrix4::from_quaternion(threers::math::Euler::new(0.0, self.yaw, 0.0).to_quaternion());
        (turn.multiply(&Matrix4::translation(-centre)).elements, radius)
    }

    /// Mesh `i`: its key, its geometry's and material's identities (they
    /// change when the shell should rebuild them), and its 32 floats
    /// ([`OBJECT_FLOATS`]).
    pub fn object(&mut self, i: usize) -> Option<([u64; 3], [f32; OBJECT_FLOATS])> {
        let item = self.items.get(i)?;
        let o = self.scene.get(item.id)?;
        // flags beyond the mesh's: 16 a line set (segments, unlit), 64 a
        // point cloud (points, unlit),
        // 32 a sprite (a quad facing the viewer, its texture `texture`)
        let (geometry, material, kind) = match &o.kind {
            ObjectKind::Mesh(m) => (geometry_key(&m.geometry), &m.material, 0u32),
            ObjectKind::LineSegments(l) => (geometry_key(&l.geometry), &l.material, 16),
            ObjectKind::Sprite(s) => (0, &s.material, 32),
            ObjectKind::Points(p) => (geometry_key(&p.geometry), &p.material, 64),
            _ => return None,
        };
        let mut out = [0.0f32; OBJECT_FLOATS];
        out[..16].copy_from_slice(&o.matrix_world.elements);
        let (color, opacity, roughness, metalness, blended, side) = match &**material {
            Material::Standard(s) => (s.color, s.opacity, s.roughness, s.metalness, s.transparent.unwrap_or(s.opacity < 1.0), s.side),
            Material::Physical(p) => (p.color, p.opacity, p.roughness, p.metalness, p.transparent.unwrap_or(p.opacity < 1.0), p.side),
            Material::Basic(b) => (b.color, b.opacity, 1.0, 0.0, b.transparent || b.opacity < 1.0, b.side),
            Material::Line(l) => (l.color, l.opacity, 1.0, 0.0, l.transparent.unwrap_or(l.opacity < 1.0), 2),
            Material::Sprite(sp) => (sp.color, sp.opacity, 1.0, 0.0, true, 2),
            Material::Points(pm) => (pm.color, pm.opacity, 1.0, 0.0, pm.transparent.unwrap_or(pm.opacity < 1.0), 2),
            // baked: its own alpha is in its pixels
            Material::Shader(sm) => (threers::math::Color::new(1.0, 1.0, 1.0), 1.0, 1.0, 0.0, sm.transparent, sm.side),
            _ => (threers::math::Color::new(0.7, 0.7, 0.7), 1.0, 0.6, 0.0, false, 0),
        };
        // a standard material's glow: its emissive colour, as the window's
        let emissive = match &**material {
            Material::Standard(s) => Some((s.emissive, s.emissive_intensity)),
            Material::Physical(p) => Some((p.emissive, p.emissive_intensity)),
            _ => None,
        };
        let srgb = |c: f32| if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
        out[16..20].copy_from_slice(&[srgb(color.r), srgb(color.g), srgb(color.b), opacity]);
        out[20] = roughness;
        out[21] = metalness;
        let visible = self.visible(item.id);
        let colors = match &o.kind {
            ObjectKind::Mesh(m) => m.geometry.get_attribute("color").is_some(),
            ObjectKind::LineSegments(l) => l.geometry.get_attribute("color").is_some(),
            ObjectKind::Points(p) => p.geometry.get_attribute("color").is_some(),
            _ => false,
        };
        // 128: a mesh with a texture (a volume's slice): its pixels from
        // `texture`, its corners' uvs from `geometry`
        // (or a shader material, baked: see `crate::bake`)
        let shader = match (&o.kind, &**material) {
            (ObjectKind::Mesh(mesh), Material::Shader(sm)) => Some(crate::bake::key(sm, mesh) ^ self.view_key(mesh)),
            _ => None,
        };
        let textured = (matches!(&**material, Material::Basic(b) if b.map.is_some()) || shader.is_some()) as u32;
        out[22] = (visible as u32 | (colors as u32) << 1 | (blended as u32) << 2 | ((side == 2) as u32) << 3 | kind | textured << 7) as f32;
        if let Some((e, k)) = emissive {
            out[23..27].copy_from_slice(&[srgb(e.r), srgb(e.g), srgb(e.b), k]);
        }
        // a shader material is what it draws; the geometry with it, as the
        // uvs are the bake's
        let m = shader.unwrap_or(Arc::as_ptr(material) as u64);
        let geometry = shader.map_or(geometry, |k| geometry ^ k.rotate_left(17));
        Some(([item.key, geometry, m], out))
    }

    /// Mesh `i`'s texture coordinates, 2 floats a vertex (empty without);
    /// held until the next call.
    pub fn uvs(&mut self, i: usize) -> &[f32] {
        self.geometry.uvs.clear();
        if let Some(b) = self.bake(i) {
            let uvs = b.uvs.clone();
            self.geometry.uvs = uvs;
            return &self.geometry.uvs;
        }
        if let Some(item) = self.items.get(i)
            && let Some(o) = self.scene.get(item.id)
            && let ObjectKind::Mesh(m) = &o.kind
            && let Some(a) = m.geometry.get_attribute("uv")
        {
            self.geometry.uvs.clone_from(&a.array);
        }
        &self.geometry.uvs
    }

    /// Sprite `i`'s texture, or a textured mesh's: RGBA8, rows from the
    /// top, and its size; held until the next call. `None` for anything
    /// else.
    pub fn texture(&mut self, i: usize) -> Option<(&[u8], u32, u32)> {
        if let Some(b) = self.bake(i) {
            let (w, h) = (b.width, b.height);
            let rgba = b.rgba.clone();
            self.geometry.texture = rgba;
            return Some((&self.geometry.texture, w, h));
        }
        let item = self.items.get(i)?;
        let o = self.scene.get(item.id)?;
        let t = match &o.kind {
            ObjectKind::Sprite(s) => match &*s.material {
                Material::Sprite(m) => m.map.as_ref()?,
                _ => return None,
            },
            ObjectKind::Mesh(m) => match &*m.material {
                Material::Basic(b) => b.map.as_ref()?,
                _ => return None,
            },
            _ => return None,
        };
        let (w, h) = (t.width, t.height);
        let row = w as usize * 4;
        let data = t.data.get(..row * h as usize)?;
        // as drawn: a flipped texture's rows run bottom up
        self.geometry.texture = if t.flip_y { data.chunks_exact(row).rev().flatten().copied().collect() } else { data.to_vec() };
        Some((&self.geometry.texture, w, h))
    }

    /// Mesh `i`'s shader material as baked — flat, as it is; otherwise as
    /// an impostor from the viewer's side — baking it when its key moved
    /// on; `None` for anything else.
    fn bake(&mut self, i: usize) -> Option<&crate::bake::Baked> {
        let id = self.items.get(i)?.id;
        let o = self.scene.get(id)?;
        let ObjectKind::Mesh(mesh) = &o.kind else { return None };
        let Material::Shader(sm) = &*mesh.material else { return None };
        let key = crate::bake::key(sm, mesh) ^ self.view_key(mesh);
        if self.baked.get(&id).is_none_or(|b| b.key != key) {
            let flat = mesh.geometry.get_attribute("position").is_some_and(|p| crate::bake::plane(&p.array).is_some());
            let b = if flat {
                self.baker.bake(mesh)?
            } else {
                // the viewer stands in front of the volume (its +z), which
                // the drags turn: that direction in the mesh's own frame
                let turn = Matrix4::from_quaternion(threers::math::Euler::new(0.0, self.yaw, 0.0).to_quaternion());
                let toward = Vector3::new(0.0, 0.0, 1.0).apply_matrix4(&turn.invert());
                let inv = o.matrix_world.invert();
                let local = Vector3::new(toward.x, toward.y, toward.z).apply_matrix4(&inv) - Vector3::ZERO.apply_matrix4(&inv);
                self.baker.impostor(mesh, local, key)?
            };
            self.baked.insert(id, b);
        }
        self.baked.get(&id)
    }

    /// Which way a mesh that is not flat is seen from, as part of its key:
    /// the volume's turn, in 5° steps (0 for a flat one).
    fn view_key(&self, mesh: &threers::core::Mesh) -> u64 {
        let flat = mesh.geometry.get_attribute("position").is_some_and(|p| crate::bake::plane(&p.array).is_some());
        if flat {
            return 0;
        }
        let step = (self.yaw.to_degrees() / 5.0).round() as i64;
        (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
    }

    /// Whether `id` and every parent of it is visible.
    fn visible(&self, mut id: ObjectId) -> bool {
        loop {
            let Some(o) = self.scene.get(id) else { return false };
            if !o.visible {
                return false;
            }
            match o.parent {
                Some(p) => id = p,
                None => return true,
            }
        }
    }

    /// Mesh `i`'s geometry, held until the next call: positions and normals
    /// (3 floats a vertex), colours (`color_size` floats a vertex, or none),
    /// triangle corners. Unindexed geometry gets indices 0, 1, 2, …
    pub fn geometry(&mut self, i: usize) -> Option<(&[f32], &[f32], &[f32], u32, &[u32])> {
        // a mesh standing in as its impostor's quad
        if let Some(quad) = self.bake(i).and_then(|b| b.quad.clone()) {
            let g = &mut self.geometry;
            g.positions = quad;
            g.normals = [0.0, 0.0, 1.0].repeat(4);
            (g.colors, g.color_size) = (vec![], 0);
            g.indices = vec![0, 1, 2, 0, 2, 3];
            return Some((&g.positions, &g.normals, &g.colors, g.color_size, &g.indices));
        }
        let item = self.items.get(i)?;
        let o = self.scene.get(item.id)?;
        let geometry = &mut self.geometry;
        match &o.kind {
            ObjectKind::Mesh(mesh) => {
                let g = &mesh.geometry;
                let positions = g.get_attribute("position")?;
                let n = positions.count();
                geometry.positions.clone_from(&positions.array);
                geometry.normals = match g.get_attribute("normal") {
                    Some(a) if a.count() == n => a.array.clone(),
                    _ => vec![0.0, 1.0, 0.0].repeat(n),
                };
                (geometry.colors, geometry.color_size) = match g.get_attribute("color") {
                    Some(a) if a.count() == n && (a.item_size == 3 || a.item_size == 4) => (a.array.clone(), a.item_size as u32),
                    _ => (vec![], 0),
                };
                geometry.indices = g.index.clone().unwrap_or_else(|| (0..n as u32).collect());
            }
            ObjectKind::LineSegments(ObjectLines { geometry: g, .. }) | ObjectKind::Points(ObjectPoints { geometry: g, .. }) => {
                // the segments (a pair of vertices each) or the points as
                // they are: the shell draws them with RealityKit's line or
                // point topology, in their vertex colours if they have them
                let positions = g.get_attribute("position")?;
                let n = positions.count();
                geometry.positions.clone_from(&positions.array);
                geometry.normals = [0.0, 1.0, 0.0].repeat(n);
                (geometry.colors, geometry.color_size) = match g.get_attribute("color") {
                    Some(a) if a.count() == n && (a.item_size == 3 || a.item_size == 4) => (a.array.clone(), a.item_size as u32),
                    _ => (vec![], 0),
                };
                geometry.indices = g.index.clone().unwrap_or_else(|| (0..n as u32).collect());
            }
            ObjectKind::Sprite(_) => {
                // a unit quad facing +z; the shell turns it to the viewer
                geometry.positions = vec![-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0];
                geometry.normals = [0.0, 0.0, 1.0].repeat(4);
                (geometry.colors, geometry.color_size) = (vec![], 0);
                geometry.indices = vec![0, 1, 2, 0, 2, 3];
            }
            _ => return None,
        }
        Some((&geometry.positions, &geometry.normals, &geometry.colors, geometry.color_size, &geometry.indices))
    }

    /// The app's UI surface `which` ([`SURFACES`]: 0 the card, 1 the
    /// toolbar), `width × height` pixels at `scale` pixels a UI pixel:
    /// RGBA8, sRGB, straight alpha, rows from the top (transparent where it
    /// draws nothing). `None` when it has not changed since the last call;
    /// empty when it is not shown now (hide it) — once, until it is again.
    /// Taps ([`crate::threers_ios_tap`], on the card), presses on controls
    /// ([`crate::threers_ios_press`]) and touches
    /// ([`crate::threers_ios_touch`]) since the last call reach the app
    /// first.
    pub fn ui(&mut self, which: usize, width: u32, height: u32, scale: f32) -> Option<&[u8]> {
        let place = *SURFACES.get(which)?;
        let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        let surface = Surface {
            place,
            width: width.max(1) as f32 / scale,
            height: height.max(1) as f32 / scale,
            scale,
            insets: [0.0; 4],
            see_through: true,
        };
        let strip = &mut self.surfaces[which];
        // lay it out first: the input lands on what is shown (and what
        // changed by itself — the room opening, the data arriving — counts)
        let (_, before) = strip.ui.update(&mut self.app, &surface);
        let button = PointerButton::Primary;
        // a press on a control's native view: a click at its middle
        let pressed: Vec<(f32, f32)> =
            crate::take_presses_for(|k| native::find(&strip.controls, k).is_some()).into_iter().filter_map(|k| native::find(&strip.controls, k)).map(|c| c.centre()).collect();
        let taps: Vec<(f32, f32)> = if which == 0 { crate::take_taps().into_iter().map(|(x, y)| (x / scale, y / scale)).collect() } else { vec![] };
        for (x, y) in pressed.into_iter().chain(taps) {
            let time = self.last;
            self.app.ui_event(place, Event::PointerMove { x, y }, time);
            self.app.ui_event(place, Event::PointerDown { x, y, button }, time);
            self.app.ui_event(place, Event::PointerUp { x, y, button }, time);
        }
        // typed into a native text field: its control focused (a click on
        // it), then given the text
        let strip = &self.surfaces[which];
        let typed: Vec<((f32, f32), String)> = crate::take_texts_for(|k| native::find(&strip.controls, k).is_some())
            .into_iter()
            .filter_map(|(k, t)| native::find(&strip.controls, k).map(|c| (c.centre(), t)))
            .collect();
        for ((x, y), text) in typed {
            let time = self.last;
            self.app.ui_event(place, Event::PointerDown { x, y, button }, time);
            self.app.ui_event(place, Event::PointerUp { x, y, button }, time);
            self.app.ui_text(place, text);
        }
        // a finger dragged on it: as one on a phone's screen
        for (phase, x, y, at) in crate::take_touches(which as u32) {
            let phase = match phase {
                0 => TouchPhase::Start,
                1 => TouchPhase::Move,
                2 => TouchPhase::End,
                _ => TouchPhase::Cancel,
            };
            let time = at.saturating_duration_since(self.start).as_secs_f64();
            self.app.ui_event(place, Event::Touch { id: 0, phase, x: x / scale, y: y / scale, time }, time);
        }
        let strip = &mut self.surfaces[which];
        let (shown, changed) = strip.ui.update(&mut self.app, &surface);
        let changed = changed || before;
        strip.controls = match (shown, self.app.ui(&surface)) {
            (true, Some(ui)) => native::controls(ui),
            _ => vec![],
        };
        if !shown {
            // hidden: say so once, and paint it whole when it comes back
            strip.fresh = true;
            let tell = !std::mem::replace(&mut strip.told_hidden, true);
            return tell.then_some(&[][..]);
        }
        strip.told_hidden = false;
        let fresh = std::mem::replace(&mut strip.fresh, false);
        (changed || fresh).then(|| strip.ui.pixels.pixels())
    }

    /// The controls of surface `which` as [`ui`](Self::ui) last drew them,
    /// as JSON (`threers_ui::native::to_json`; rectangles in UI pixels,
    /// which are the image's points), when they changed since the last
    /// call: lay a native view over each, and send a press on one back by
    /// its key ([`crate::threers_ios_press`]). `None` while they are as they
    /// were.
    pub fn controls(&mut self, which: usize) -> Option<&str> {
        let strip = self.surfaces.get_mut(which)?;
        let v = native::version(&strip.controls);
        if strip.sent == Some(v) {
            return None;
        }
        strip.sent = Some(v);
        strip.json = native::to_json(&strip.controls);
        Some(&strip.json)
    }
}

// --- the C side, for `app!` -------------------------------------------------

#[doc(hidden)]
/// # Safety
/// `v` null or from [`create`]; `len` null or writable.
pub unsafe fn controls<A: App>(v: *mut c_void, which: u32, len: *mut usize) -> *const u8 {
    let Some(json) = (unsafe { get::<A>(v) }).and_then(|v| v.controls(which as usize)) else { return std::ptr::null() };
    if !len.is_null() {
        unsafe { *len = json.len() };
    }
    json.as_ptr()
}

/// # Safety
/// `v` null or from [`create`]; not destroyed.
unsafe fn get<'a, A: App>(v: *mut c_void) -> Option<&'a mut Volume<A>> {
    unsafe { v.cast::<Volume<A>>().as_mut() }
}

#[doc(hidden)]
pub fn create<A: App>() -> *mut c_void {
    Box::into_raw(Box::new(Volume::<A>::new())).cast()
}

/// # Safety
/// `v` null or from [`create`]; not used again.
#[doc(hidden)]
pub unsafe fn destroy<A: App>(v: *mut c_void) {
    if !v.is_null() {
        drop(unsafe { Box::from_raw(v.cast::<Volume<A>>()) });
    }
}

/// # Safety
/// As [`get`].
#[doc(hidden)]
pub unsafe fn frame<A: App>(v: *mut c_void) -> u32 {
    unsafe { get::<A>(v) }.map_or(0, |v| v.frame() as u32)
}

/// # Safety
/// As [`get`]; `keys` 3 writable u64s, `floats` [`OBJECT_FLOATS`] f32s.
#[doc(hidden)]
pub unsafe fn object<A: App>(v: *mut c_void, i: u32, keys: *mut u64, floats: *mut f32) -> bool {
    let Some((k, f)) = unsafe { get::<A>(v) }.and_then(|v| v.object(i as usize)) else { return false };
    if keys.is_null() || floats.is_null() {
        return false;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(k.as_ptr(), keys, 3);
        std::ptr::copy_nonoverlapping(f.as_ptr(), floats, OBJECT_FLOATS);
    }
    true
}

/// # Safety
/// As [`get`]; `ptrs` 4 writable pointers (positions, normals, colours,
/// indices), `counts` 3 u32s (vertices, floats a colour, indices). The
/// pointers hold until the next call.
#[doc(hidden)]
pub unsafe fn geometry<A: App>(v: *mut c_void, i: u32, ptrs: *mut *const c_void, counts: *mut u32) -> bool {
    let Some((p, n, c, cs, ix)) = unsafe { get::<A>(v) }.and_then(|v| v.geometry(i as usize)) else { return false };
    if ptrs.is_null() || counts.is_null() {
        return false;
    }
    unsafe {
        *ptrs = p.as_ptr().cast();
        *ptrs.add(1) = n.as_ptr().cast();
        *ptrs.add(2) = if c.is_empty() { std::ptr::null() } else { c.as_ptr().cast() };
        *ptrs.add(3) = ix.as_ptr().cast();
        *counts = (p.len() / 3) as u32;
        *counts.add(1) = cs;
        *counts.add(2) = ix.len() as u32;
    }
    true
}

/// # Safety
/// As [`get`]; `len` writable. The floats hold until the next call.
#[doc(hidden)]
pub unsafe fn uvs<A: App>(v: *mut c_void, i: u32, len: *mut usize) -> *const f32 {
    let Some(v) = (unsafe { get::<A>(v) }) else { return std::ptr::null() };
    let u = v.uvs(i as usize);
    if len.is_null() || u.is_empty() {
        return std::ptr::null();
    }
    unsafe { *len = u.len() };
    u.as_ptr()
}

/// # Safety
/// As [`get`]; `w`, `h`, `len` writable. The bytes hold until the next call.
#[doc(hidden)]
pub unsafe fn texture<A: App>(v: *mut c_void, i: u32, w: *mut u32, h: *mut u32, len: *mut usize) -> *const u8 {
    let Some((bytes, tw, th)) = unsafe { get::<A>(v) }.and_then(|v| v.texture(i as usize)) else { return std::ptr::null() };
    if w.is_null() || h.is_null() || len.is_null() {
        return std::ptr::null();
    }
    unsafe {
        (*w, *h, *len) = (tw, th, bytes.len());
    }
    bytes.as_ptr()
}

/// # Safety
/// As [`get`]; `out` 17 writable floats: the matrix, then the radius.
#[doc(hidden)]
pub unsafe fn root<A: App>(v: *mut c_void, out: *mut f32) {
    let (Some(v), false) = (unsafe { get::<A>(v) }, out.is_null()) else { return };
    let (m, r) = v.root();
    unsafe {
        std::ptr::copy_nonoverlapping(m.as_ptr(), out, 16);
        *out.add(16) = r;
    }
}

/// # Safety
/// As [`get`].
#[doc(hidden)]
pub unsafe fn turn<A: App>(v: *mut c_void, radians: f32) {
    if let Some(v) = unsafe { get::<A>(v) } {
        v.turn(radians);
    }
}

/// # Safety
/// As [`get`]; `len` writable. The bytes hold until the next call.
#[doc(hidden)]
pub unsafe fn ui<A: App>(v: *mut c_void, which: u32, width: u32, height: u32, scale: f32, len: *mut usize) -> *const u8 {
    let Some(bytes) = (unsafe { get::<A>(v) }).and_then(|v| v.ui(which as usize, width, height, scale)) else { return std::ptr::null() };
    if !len.is_null() {
        unsafe { *len = bytes.len() };
    }
    bytes.as_ptr()
}
