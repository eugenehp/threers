//! An [`App`] where RealityKit draws it: a visionOS **volume**, in the
//! shared space beside other apps (no immersive space, no
//! CompositorServices), or a `RealityView` on an iPhone, an iPad or a Mac.
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

use threers::core::{ObjectId, ObjectKind};
use threers::materials::Material;
use threers::math::{Matrix4, Vector3};
use threers::scene::Scene;
use threers::ui::input::{Event, PointerButton, TouchPhase};
use threers::ui::native::{self, Control};
use threers::xr::Frame;

use crate::reality;
use crate::surface_ui::SurfaceUi;
use crate::{App, Place, Surface};

/// One mesh, as the shell reads it: 48 floats.
///
/// `[0..16]` the world matrix (column-major, the volume's frame: see
/// [`Volume::root`]), `[16..20]` the colour (sRGB, straight alpha),
/// `[20]` roughness, `[21]` metalness, `[22]` flags: 1 visible, 2 vertex
/// colours, 4 blended, 8 double-sided, 16 lines, 32 a sprite, 64 points,
/// 128 textured, 256 no depth write, 512 casts a shadow, 1024 instanced
/// ([`Volume::instances`]), 2048 posed (skinned or morphed: its vertices
/// move, its counts stay), 4096 tangents with its geometry; `[23..27]` the
/// emissive colour (sRGB) and its intensity; `[27]` clearcoat, `[28]` its
/// roughness; `[29]` the maps it has (a bit per
/// [`Volume::map`] slot), `[30]` the normal map's scale, `[31]` the
/// occlusion map's intensity, `[32..35]` the sheen's colour (sRGB, black
/// for none), `[35]` the alpha test (0 none), `[36]` the instances, `[37]`
/// 1 when its maps repeat; the rest zero.
pub const OBJECT_FLOATS: usize = 48;

/// The keys [`Volume::object`] hands over with each mesh: its own, its
/// geometry's, its material's, and its instances' or pose's.
pub const OBJECT_KEYS: usize = 4;

/// A geometry's identity for the shell: which one, and its version — an
/// edit in place (`Arc::make_mut`: another surface, repainted colours)
/// keeps the pointer, and must still rebuild.
fn geometry_key(g: &Arc<threers::core::BufferGeometry>) -> u64 {
    (Arc::as_ptr(g) as u64) ^ (u64::from(g.geometry_version) << 40).rotate_left(7)
}

/// How much of the focus's radius the volume's quad shows, to its edges.
const FILL: f32 = 1.0;

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
    /// tangents (4 a vertex: the bitangent's sign last), or none
    tangents: Vec<f32>,
    /// instance matrices (16 floats each), for [`Volume::instances`]
    instances: Vec<f32>,
}

/// A mesh's arrays as [`Volume::geometry`] hands them over: positions,
/// normals, colours, floats a colour, triangle corners, tangents.
pub type MeshArrays<'a> = (&'a [f32], &'a [f32], &'a [f32], u32, &'a [u32], &'a [f32]);

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
    /// under it ([`Place::Toolbar`]), or over the screen ([`Place::Screen`]):
    /// [`SURFACES`] in order.
    surfaces: [Strip; 3],
    start: Instant,
    last: f64,
    /// Shader materials RealityKit cannot run, drawn by threers for it
    /// ([`bake`](crate::bake)), by mesh, while their key holds.
    baker: crate::bake::Baker,
    baked: HashMap<ObjectId, crate::bake::Baked>,
    /// The whole scene drawn by threers' own renderer, for the shell's
    /// quad ([`Volume::draw_into`]), made on first use.
    quad: Option<threers::metal::MetalHeadlessRenderer>,
    /// the quad's last orientation in the volume, and the taps on it since
    /// the last frame (rays in the scene)
    quad_q: [f32; 4],
    taps: Vec<(Vector3, Vector3)>,
    /// The scene's lights as last read ([`reality::LIGHT_FLOATS`] each).
    lights: Vec<f32>,
    /// The screen's safe area, points: top, bottom, left, right.
    insets: [f32; 4],
    /// The environment's key as last handed over, and its image.
    environment: Option<u64>,
    equirect: Vec<f32>,
}

/// Where the volume's UI surfaces are shown, by the index the shell
/// names them with: the card beside the volume, the toolbar under it.
/// The third, the screen ([`Place::Screen`]), is the UI over RealityKit's
/// view on an iPhone, an iPad or a Mac.
pub const SURFACES: [Place; 3] = [Place::Image, Place::Toolbar, Place::Screen];

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
        Strip {
            ui: SurfaceUi::default(),
            fresh: true,
            told_hidden: false,
            controls: vec![],
            sent: None,
            json: String::new(),
        }
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
            surfaces: Default::default(),
            start: Instant::now(),
            last: 0.0,
            baker: Default::default(),
            baked: HashMap::new(),
            quad: None,
            quad_q: [0.0, 0.0, 0.0, 1.0],
            taps: vec![],
            lights: vec![],
            insets: [0.0; 4],
            environment: None,
            equirect: vec![],
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
        if radians.is_finite() && !self.app.turn(radians) {
            self.yaw += radians;
        }
    }

    /// Update the app and list the meshes it shows: how many.
    pub fn frame(&mut self) -> usize {
        let time = self.start.elapsed().as_secs_f64();
        let dt = (time - self.last).clamp(0.0, 0.25);
        self.last = time;
        // the app asked for its home view: undo the shell's turning too
        if crate::take_camera_requests()
            .iter()
            .any(|r| matches!(r, crate::CameraRequest::Home))
        {
            self.yaw = 0.0;
        }
        // the taps on the quad since, as transient pointers that select
        let (mut inputs, mut events) = (vec![], vec![]);
        for (slot, (from, dir)) in std::mem::take(&mut self.taps).into_iter().enumerate() {
            use threers::xr::{ray_pose, TargetRayMode, XrEvent, XrEventKind, XrInput};
            inputs.push(XrInput {
                slot,
                target_ray_mode: TargetRayMode::TransientPointer,
                ray: Some(ray_pose(from, dir)),
                ..Default::default()
            });
            events.extend(
                [
                    XrEventKind::Connected,
                    XrEventKind::SelectStart,
                    XrEventKind::SelectEnd,
                    XrEventKind::Select,
                    XrEventKind::Disconnected,
                ]
                .map(|kind| XrEvent { kind, slot }),
            );
        }
        let frame = Frame {
            time,
            dt,
            head: None,
            inputs,
            events,
            see_through: true,
        };
        crate::set_drawn_by_realitykit(self.quad.is_none());
        self.app.update(&mut self.scene, &frame);
        crate::set_drawn_by_realitykit(false);
        self.scene.update_world();
        self.items.clear();
        for (id, o) in self.scene.arena.nodes.iter() {
            // meshes (skinned and instanced too); line sets, points and
            // sprites (as billboards)
            if !matches!(
                o.kind,
                ObjectKind::Mesh(_)
                    | ObjectKind::SkinnedMesh(_)
                    | ObjectKind::InstancedMesh(_)
                    | ObjectKind::LineSegments(_)
                    | ObjectKind::Sprite(_)
                    | ObjectKind::Points(_)
            ) {
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
        self.lights = reality::lights(&self.scene, |id| self.visible(id));
        self.items.len()
    }

    /// The scene's directional, point and spot lights as of the last
    /// [`frame`](Self::frame), [`reality::LIGHT_FLOATS`] each, in the
    /// scene's frame (the meshes').
    pub fn lights(&self) -> &[f32] {
        &self.lights
    }

    /// How the app asks RealityKit to light it.
    pub fn lighting(&self) -> reality::RealityLighting {
        self.app.reality_lighting()
    }

    /// The scene's background, linear RGB and its alpha: what RealityKit
    /// shows behind it on a screen (nothing in a volume or AR).
    pub fn background(&self) -> [f32; 4] {
        let c = self.scene.background;
        [c.r, c.g, c.b, self.scene.background_alpha]
    }

    /// The light round the scene (its environment map, its ambient and hemisphere lights: `width` ×
    /// `width / 2` linear RGBA floats) when it changed since the last call;
    /// empty when the scene has none (RealityKit's own light then).
    pub fn environment(&mut self, width: u32) -> Option<&[f32]> {
        let key = reality::environment_key(&self.scene);
        if self.environment == Some(key) {
            return None;
        }
        self.environment = Some(key);
        self.equirect = if reality::has_environment(&self.scene) {
            reality::equirect(&self.scene, width)
        } else {
            vec![]
        };
        Some(&self.equirect)
    }

    /// Instanced mesh `i`'s instances, 16 floats each (column-major, under
    /// its own matrix); held until the next call. Empty for anything else.
    pub fn instances(&mut self, i: usize) -> &[f32] {
        self.geometry.instances.clear();
        if let Some(item) = self.items.get(i)
            && let Some(o) = self.scene.get(item.id)
            && let ObjectKind::InstancedMesh(m) = &o.kind
        {
            self.geometry.instances.extend(m.transforms.iter().flat_map(|t| t.elements));
        }
        &self.geometry.instances
    }

    /// Lit mesh `i`'s map in `slot` ([`reality::MAP_SLOTS`]): RGBA8, its
    /// size; held until the next call.
    pub fn map(&mut self, i: usize, slot: usize) -> Option<(&[u8], u32, u32)> {
        let item = self.items.get(i)?;
        let o = self.scene.get(item.id)?;
        let (px, w, h) = reality::map_pixels(material_of(&o.kind)?, slot)?;
        self.geometry.texture = px;
        Some((&self.geometry.texture, w, h))
    }

    /// The UI surface at `place`, `width × height` pixels at `scale` (1
    /// when that is not a scale): see-through, the screen's within its safe
    /// area.
    fn surface(&self, place: Place, width: u32, height: u32, scale: f32) -> Surface {
        let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        Surface {
            place,
            width: width.max(1) as f32 / scale,
            height: height.max(1) as f32 / scale,
            scale,
            insets: if place == Place::Screen { self.insets } else { [0.0; 4] },
            see_through: true,
        }
    }

    /// The screen's safe area (its UI's points: top, bottom, left, right),
    /// for [`Place::Screen`].
    pub fn set_insets(&mut self, insets: [f32; 4]) {
        self.insets = insets.map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 });
    }

    /// Whether the screen's UI ([`Place::Screen`], surface 2) has something
    /// at `(x, y)` (its image's pixels at `scale`): a finger there is the
    /// UI's, elsewhere RealityKit's (the camera, a tap on the scene).
    pub fn ui_hit(&mut self, which: usize, x: f32, y: f32, scale: f32) -> bool {
        let Some(&place) = SURFACES.get(which) else { return false };
        let strip = &self.surfaces[which];
        if !strip.ui.shown {
            return false;
        }
        let (w, h) = strip.ui.pixels.size();
        let surface = self.surface(place, w, h, scale);
        let scale = surface.scale;
        self.app.ui(&surface).is_some_and(|ui| {
            let root = ui.root();
            ui.hit_test(x / scale, y / scale).is_some_and(|n| n != root)
        })
    }

    /// A ray through the scene (its frame: the meshes'), as the shell casts
    /// it through a tap on the screen: a select along it, for the app's next
    /// frame.
    pub fn ray(&mut self, origin: Vector3, direction: Vector3) {
        if origin.x.is_finite() && direction.length() > 0.0 {
            self.taps.push((origin, direction));
        }
    }

    /// The model's frame in the volume, column-major: the app's
    /// [`focus`](App::focus) at the volume's centre, turned by the drags,
    /// and the radius it has (the shell scales it to fit).
    pub fn root(&self) -> ([f32; 16], f32) {
        let (centre, radius) = self.app.focus().unwrap_or((Vector3::ZERO, 1.0));
        let turn =
            Matrix4::from_quaternion(threers::math::Euler::new(0.0, self.yaw, 0.0).to_quaternion());
        (
            turn.multiply(&Matrix4::translation(-centre)).elements,
            radius,
        )
    }

    /// Mesh `i`: its key, its geometry's and material's identities (they
    /// change when the shell should rebuild them), and its 32 floats
    /// ([`OBJECT_FLOATS`]).
    pub fn object(&mut self, i: usize) -> Option<([u64; OBJECT_KEYS], [f32; OBJECT_FLOATS])> {
        let item = self.items.get(i)?;
        let o = self.scene.get(item.id)?;
        // flags beyond the mesh's: 16 a line set (segments, unlit), 64 a
        // point cloud (points, unlit),
        // 32 a sprite (a quad facing the viewer, its texture `texture`),
        // 1024 instanced, 2048 posed; and the key of the pose or instances
        let (geometry, material, kind, motion) = match &o.kind {
            ObjectKind::Mesh(m) if morphed(m) => (
                geometry_key(&m.geometry),
                &m.material,
                2048,
                reality::pose_key(None, &m.morph_influences),
            ),
            ObjectKind::Mesh(m) => (geometry_key(&m.geometry), &m.material, 0u32, 0),
            ObjectKind::SkinnedMesh(m) => {
                let palette = m.skinning_palette(&self.scene.arena, &o.matrix_world);
                let pose = reality::pose_key(palette.as_deref(), &m.morph_influences);
                (geometry_key(&m.geometry), &m.material, 2048, pose)
            }
            ObjectKind::InstancedMesh(m) => (
                geometry_key(&m.geometry),
                &m.material,
                1024,
                reality::instances_key(&m.transforms, m.instance_version),
            ),
            ObjectKind::LineSegments(l) => (geometry_key(&l.geometry), &l.material, 16, 0),
            ObjectKind::Sprite(s) => (0, &s.material, 32, 0),
            ObjectKind::Points(p) => (geometry_key(&p.geometry), &p.material, 64, 0),
            _ => return None,
        };
        let mut out = [0.0f32; OBJECT_FLOATS];
        out[..16].copy_from_slice(&o.matrix_world.elements);
        let (color, opacity, roughness, metalness, blended, side) = match &**material {
            Material::Standard(s) => (
                s.color,
                s.opacity,
                s.roughness,
                s.metalness,
                s.transparent.unwrap_or(s.opacity < 1.0),
                s.side,
            ),
            Material::Physical(p) => (
                p.color,
                p.opacity,
                p.roughness,
                p.metalness,
                p.transparent.unwrap_or(p.opacity < 1.0),
                p.side,
            ),
            Material::Basic(b) => (
                b.color,
                b.opacity,
                1.0,
                0.0,
                b.transparent || b.opacity < 1.0,
                b.side,
            ),
            Material::Line(l) => (
                l.color,
                l.opacity,
                1.0,
                0.0,
                l.transparent.unwrap_or(l.opacity < 1.0),
                2,
            ),
            Material::Sprite(sp) => (sp.color, sp.opacity, 1.0, 0.0, true, 2),
            Material::Points(pm) => (
                pm.color,
                pm.opacity,
                1.0,
                0.0,
                pm.transparent.unwrap_or(pm.opacity < 1.0),
                2,
            ),
            // baked: its own alpha is in its pixels
            Material::Shader(sm) => (
                threers::math::Color::new(1.0, 1.0, 1.0),
                1.0,
                1.0,
                0.0,
                sm.transparent,
                sm.side,
            ),
            _ => (
                threers::math::Color::new(0.7, 0.7, 0.7),
                1.0,
                0.6,
                0.0,
                false,
                0,
            ),
        };
        // a standard material's glow: its emissive colour, as the window's
        let emissive = match &**material {
            Material::Standard(s) => Some((s.emissive, s.emissive_intensity)),
            Material::Physical(p) => Some((p.emissive, p.emissive_intensity)),
            _ => None,
        };
        let srgb = |c: f32| {
            if c <= 0.003_130_8 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            }
        };
        out[16..20].copy_from_slice(&[srgb(color.r), srgb(color.g), srgb(color.b), opacity]);
        out[20] = roughness;
        out[21] = metalness;
        // a shader RealityKit cannot run, on a mesh threers cannot bake
        let unbaked = match (&o.kind, &**material) {
            (ObjectKind::Mesh(mesh), Material::Shader(_)) => !mesh
                .geometry
                .get_attribute("position")
                .is_some_and(|p| crate::bake::plane(&p.array).is_some()),
            _ => false,
        };
        let visible = self.visible(item.id) && !unbaked;
        let colors = geometry_of(&o.kind).is_some_and(|g| g.get_attribute("color").is_some());
        // 128: a mesh with a texture (a volume's slice): its pixels from
        // `texture`, its corners' uvs from `geometry`
        // (or a shader material, baked: see `crate::bake`)
        let shader = match (&o.kind, &**material) {
            (ObjectKind::Mesh(mesh), Material::Shader(sm)) => Some(crate::bake::key(sm, mesh)),
            _ => None,
        };
        let textured = (matches!(&**material, Material::Basic(b) if b.map.is_some())
            || shader.is_some()) as u32;
        // 256: writes no depth (a volume's rays, blended over what is
        // behind and in front alike)
        let no_depth = matches!(&**material, Material::Shader(sm) if sm.render_state.depth_write == Some(false))
            as u32;
        // a lit material's maps, and the tangents a normal map needs
        let maps = reality::maps(material);
        let tangents = maps & reality::NORMAL_MAP != 0
            && geometry_of(&o.kind).is_some_and(|g| g.get_attribute("uv").is_some());
        out[22] = (visible as u32
            | (colors as u32) << 1
            | (blended as u32) << 2
            | ((side == 2) as u32) << 3
            | kind
            | textured << 7
            | no_depth << 8
            | (o.cast_shadow as u32) << 9
            | (tangents as u32) << 12) as f32;
        if let Some((e, k)) = emissive {
            out[23..27].copy_from_slice(&[srgb(e.r), srgb(e.g), srgb(e.b), k]);
        }
        match &**material {
            Material::Physical(p) => {
                out[27] = p.clearcoat;
                out[28] = p.clearcoat_roughness;
                if p.sheen > 0.0 {
                    let c = p.sheen_color;
                    let k = p.sheen.min(1.0);
                    out[32..35].copy_from_slice(&[srgb(c.r * k), srgb(c.g * k), srgb(c.b * k)]);
                }
                out[30] = p.normal_scale.x;
                out[31] = p.ao_intensity;
                out[35] = p.alpha_test;
            }
            Material::Standard(s) => {
                out[30] = s.normal_scale.x;
                out[31] = s.ao_intensity;
                out[35] = s.alpha_test;
            }
            _ => {}
        }
        out[29] = maps as f32;
        out[37] = reality::maps_repeat(material) as u32 as f32;
        if let ObjectKind::InstancedMesh(m) = &o.kind {
            out[36] = m.transforms.len() as f32;
        }
        // a shader material is what it draws; the geometry with it, as the
        // uvs are the bake's
        let m = shader.unwrap_or(Arc::as_ptr(material) as u64 ^ reality::maps_key(material));
        let geometry = shader.map_or(geometry, |k| geometry ^ k.rotate_left(17));
        Some(([item.key, geometry, m, motion], out))
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
            && !matches!(o.kind, ObjectKind::LineSegments(_) | ObjectKind::Points(_))
            && let Some(g) = geometry_of(&o.kind)
        {
            // the set and the transform of its maps' (one for all of them)
            let map = material_of(&o.kind).and_then(reality::uv_map);
            let set = if map.is_some_and(|t| t.channel == 1) { "uv1" } else { "uv" };
            if let Some(a) = g.get_attribute(set).or_else(|| g.get_attribute("uv")) {
                self.geometry.uvs.clone_from(&a.array);
            }
            if let Some(t) = map {
                let [r0, r1] = t.uv_transform_rows();
                if [r0, r1] != [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
                    for uv in self.geometry.uvs.as_chunks_mut::<2>().0 {
                        let [u, v] = *uv;
                        *uv = [r0[0] * u + r0[1] * v + r0[2], r1[0] * u + r1[1] * v + r1[2]];
                    }
                }
            }
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
        // as drawn: a flipped texture's rows run bottom up (one channel
        // or floats as RGBA8)
        self.geometry.texture = reality::realitykit_rows(t)?;
        Some((&self.geometry.texture, t.width, t.height))
    }

    /// Mesh `i`'s shader material as baked, baking it when its key moved
    /// on; `None` for anything else, or a mesh that is not flat.
    fn bake(&mut self, i: usize) -> Option<&crate::bake::Baked> {
        let id = self.items.get(i)?.id;
        let ObjectKind::Mesh(mesh) = &self.scene.get(id)?.kind else {
            return None;
        };
        let Material::Shader(sm) = &*mesh.material else {
            return None;
        };
        let key = crate::bake::key(sm, mesh);
        if self.baked.get(&id).is_none_or(|b| b.key != key) {
            // many (a volume's stack of slabs) share the texture budget
            let shaders = self.items.iter().filter(|it| matches!(self.scene.get(it.id).map(|o| &o.kind), Some(ObjectKind::Mesh(m)) if matches!(&*m.material, Material::Shader(_)))).count();
            let longest = if shaders > 8 { 256.0 } else { 768.0 };
            let b = self.baker.bake(mesh, longest)?;
            self.baked.insert(id, b);
        }
        self.baked.get(&id)
    }

    /// The whole scene as the window draws it — threers' own Metal
    /// renderer: glass, clip planes, every shader — into `texture`
    /// (`size` square, RGBA8 sRGB, RealityKit's `LowLevelTexture`), the
    /// copy encoded into `command_buffer`. Seen along the quad that shows
    /// it: `q` its orientation in the volume (x, y, z, w; it faces the
    /// viewer along its +z), turned with the drags; the model's focus
    /// fills it ([`FILL`]). Nothing behind: clear.
    ///
    /// # Safety
    /// As [`MetalHeadlessRenderer::copy_into`](threers::metal::MetalHeadlessRenderer::copy_into).
    pub unsafe fn draw_into(
        &mut self,
        q: [f32; 4],
        size: u32,
        texture: *mut c_void,
        command_buffer: *mut c_void,
    ) -> bool {
        // SAFETY: the caller's live texture and command buffer.
        let Some(r) = self.draw_quad(q, size) else { return false };
        let copied = unsafe { r.copy_into(texture as _, command_buffer as _) }.is_ok();
        // a picture asked for ([`crate::snapshot`]): this frame, read back
        // (clear round the model, as the volume shows it)
        if crate::take_snapshot() {
            match r.read_rgba_processed() {
                Ok(rgba) => self.app.snapshot(size, size, rgba),
                Err(e) => eprintln!("threers-ios: the volume's snapshot: {e:?}"),
            }
        }
        copied
    }

    /// What [`draw_into`](Self::draw_into) draws, read back: `size`
    /// square RGBA8, rows from the top (for tests).
    pub fn quad_picture(&mut self, q: [f32; 4], size: u32) -> Option<Vec<u8>> {
        self.draw_quad(q, size)?.read_rgba_processed().ok()
    }

    /// The scene drawn along the quad, on the quad's renderer.
    fn draw_quad(
        &mut self,
        q: [f32; 4],
        size: u32,
    ) -> Option<&mut threers::metal::MetalHeadlessRenderer> {
        self.quad_q = q;
        // where everything is now (the camera fits it; the render would
        // update them only after)
        self.scene.update_world();
        let camera = self.quad_camera();
        let size = size.max(16);
        let renderer = match &mut self.quad {
            Some(r) => r,
            slot => match threers::metal::MetalHeadlessRenderer::builder()
                .size(size, size)
                .msaa(4)
                .build()
            {
                Ok(r) => slot.insert(r),
                Err(e) => {
                    eprintln!("threers-ios: the volume's quad: {e}");
                    return None;
                }
            },
        };
        renderer.resize(size, size).ok()?;
        // the app's tone mapping, as the window and the room apply it
        let (tone, exposure) = self.app.tone_mapping();
        renderer.renderer_mut().set_tone_mapping(tone, exposure);
        // the folds shaded as the app asks, for this frame's camera
        renderer.set_ssao(self.app.ambient_occlusion().map(|(radius, strength)| threers::metal::SsaoSettings {
            strength,
            radius,
            near: camera.near,
            far: camera.far,
            tan_half_fov: (camera.fov * 0.5).tan(),
            aspect: 1.0,
        }));
        let alpha = self.scene.background_alpha;
        self.scene.background_alpha = 0.0;
        let drawn = renderer.render(&mut self.scene, &camera).is_ok();
        self.scene.background_alpha = alpha;
        drawn.then_some(renderer)
    }

    /// The camera the quad is drawn from: along the quad (`quad_q`, its
    /// orientation in the volume, turned with the drags) at what is shown,
    /// the size the app frames (its zoom with it) filling it ([`FILL`]).
    fn quad_camera(&self) -> threers::cameras::PerspectiveCamera {
        let q = self.quad_q;
        let focus = self.app.focus().unwrap_or((Vector3::ZERO, 1.0));
        // a point the app chose to frame (a scene's, a double tap's): there;
        // else all that is shown, zoomed as the user zoomed — its resting
        // focus may sit off the model (a window's sits above its toolbar)
        // and be smaller than it, which here would cut the model at the
        // quad's edges
        let (centre, radius) = match self.shown() {
            Some((c, r)) if !self.app.framing() => (c, r / self.app.zoom().max(0.1)),
            _ => focus,
        };
        let rot = threers::math::Quaternion {
            x: q[0],
            y: q[1],
            z: q[2],
            w: q[3],
        };
        // the volume's frame is the scene's turned by the drags: undo that
        let unturn = Matrix4::from_quaternion(
            threers::math::Euler::new(0.0, -self.yaw, 0.0).to_quaternion(),
        );
        let toward = Vector3::new(0.0, 0.0, 1.0)
            .apply_quaternion(rot)
            .apply_matrix4(&unturn);
        let up = Vector3::new(0.0, 1.0, 0.0)
            .apply_quaternion(rot)
            .apply_matrix4(&unturn);
        let half_fov = 15f32.to_radians();
        let d = radius * FILL / half_fov.tan();
        let mut camera = threers::cameras::PerspectiveCamera::new(
            half_fov.to_degrees() * 2.0,
            1.0,
            d * 0.02,
            d + radius * 6.0,
        );
        camera.position = centre + toward * d;
        camera.up = up;
        camera.look_at(centre);
        camera
    }

    /// A tap on the quad at `(u, v)` (0..1 across, from its foot up): a
    /// select along the ray through that point of the picture, for the
    /// app's next frame — as a pinch in the room.
    pub fn tap(&mut self, u: f32, v: f32) {
        use threers::cameras::Camera;
        let camera = self.quad_camera();
        let (view, projection) = (camera.view_matrix(), camera.projection_matrix());
        let (x, y) = (2.0 * u - 1.0, 2.0 * v - 1.0);
        let near = Vector3::new(x, y, 0.0).unproject(&view, &projection);
        let far = Vector3::new(x, y, 1.0).unproject(&view, &projection);
        self.taps.push((near, far - near));
    }

    /// A sphere round what is visible (meshes — volumes' boxes among them —
    /// lines, points), in the scene's frame: centred on their bounds, out
    /// to the farthest of their own spheres. The quad's picture fits it, so
    /// whichever way the quad turns nothing is cut, by the picture's edges
    /// or the volume's walls (the sphere is inside both).
    fn shown(&self) -> Option<(Vector3, f32)> {
        let spheres: Vec<(Vector3, f32)> = self
            .scene
            .arena
            .nodes
            .iter()
            .filter_map(|(id, o)| {
                let geometry = match &o.kind {
                    ObjectKind::Mesh(m) => &m.geometry,
                    ObjectKind::LineSegments(l) => &l.geometry,
                    ObjectKind::Points(p) => &p.geometry,
                    _ => return None,
                };
                // stored or computed (and cached): every one counts
                let sphere = geometry.bounding_sphere_cached();
                if sphere.radius < 0.0 || !self.visible(id) {
                    return None;
                }
                let e = &o.matrix_world.elements;
                let scale = (0..3)
                    .map(|c| Vector3::new(e[c * 4], e[c * 4 + 1], e[c * 4 + 2]).length())
                    .fold(0.0, f32::max);
                Some((
                    sphere.center.apply_matrix4(&o.matrix_world),
                    sphere.radius * scale,
                ))
            })
            .filter(|(c, r)| c.x.is_finite() && r.is_finite())
            .collect();
        let (mut lo, mut hi) = (
            Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        );
        for &(c, r) in &spheres {
            lo = Vector3::new(lo.x.min(c.x - r), lo.y.min(c.y - r), lo.z.min(c.z - r));
            hi = Vector3::new(hi.x.max(c.x + r), hi.y.max(c.y + r), hi.z.max(c.z + r));
        }
        let centre = (lo + hi) * 0.5;
        let radius = spheres
            .iter()
            .map(|&(c, r)| (c - centre).length() + r)
            .fold(0.0, f32::max);
        (radius > 0.0).then_some((centre, radius))
    }

    /// Whether `id` and every parent of it is visible.
    fn visible(&self, mut id: ObjectId) -> bool {
        loop {
            let Some(o) = self.scene.get(id) else {
                return false;
            };
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
    pub fn geometry(&mut self, i: usize) -> Option<MeshArrays<'_>> {
        let item = self.items.get(i)?;
        let o = self.scene.get(item.id)?;
        let geometry = &mut self.geometry;
        geometry.tangents.clear();
        // as drawn now: morphed, skinned
        let posed = match &o.kind {
            ObjectKind::Mesh(m) if morphed(m) => Some(m.posed_geometry()),
            ObjectKind::SkinnedMesh(m) => Some(m.posed_geometry(&o.matrix_world)),
            _ => None,
        };
        if let ObjectKind::Sprite(_) = o.kind {
            // a unit quad facing +z; the shell turns it to the viewer
            geometry.positions = vec![-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0];
            geometry.normals = [0.0, 0.0, 1.0].repeat(4);
            (geometry.colors, geometry.color_size) = (vec![], 0);
            geometry.indices = vec![0, 1, 2, 0, 2, 3];
        } else {
            let g = posed.as_deref().or(geometry_of(&o.kind).map(|g| &**g))?;
            // line sets and points: the segments (a pair of vertices each) or
            // the points as they are, which the shell draws with RealityKit's
            // line or point topology, unlit
            let unlit = matches!(o.kind, ObjectKind::LineSegments(_) | ObjectKind::Points(_));
            let positions = g.get_attribute("position")?;
            let n = positions.count();
            geometry.positions.clone_from(&positions.array);
            geometry.normals = match g.get_attribute("normal") {
                Some(a) if a.count() == n && !unlit => a.array.clone(),
                _ => [0.0, 1.0, 0.0].repeat(n),
            };
            (geometry.colors, geometry.color_size) = match g.get_attribute("color") {
                Some(a) if a.count() == n && (a.item_size == 3 || a.item_size == 4) => {
                    (a.array.clone(), a.item_size as u32)
                }
                _ => (vec![], 0),
            };
            geometry.indices = g.index.clone().unwrap_or_else(|| (0..n as u32).collect());
            // a normal map's tangents, where it has the uvs for them
            let maps = material_of(&o.kind).map_or(0, reality::maps);
            if !unlit && maps & reality::NORMAL_MAP != 0 && g.get_attribute("uv").is_some() {
                let mut t = g.clone();
                if t.get_attribute("normal").is_none() {
                    threers::utils::compute_vertex_normals(&mut t);
                }
                threers::utils::compute_tangents(&mut t);
                if let Some(a) = t.get_attribute("tangent") {
                    geometry.tangents.clone_from(&a.array);
                }
            }
        }
        if geometry.tangents.len() != geometry.positions.len() / 3 * 4 {
            geometry.tangents.clear();
        }
        Some((
            &geometry.positions,
            &geometry.normals,
            &geometry.colors,
            geometry.color_size,
            &geometry.indices,
            &geometry.tangents,
        ))
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
        let surface = self.surface(place, width, height, scale);
        let scale = surface.scale;
        let strip = &mut self.surfaces[which];
        // lay it out first: the input lands on what is shown (and what
        // changed by itself — the room opening, the data arriving — counts)
        let (_, before) = strip.ui.update(&mut self.app, &surface);
        let button = PointerButton::Primary;
        // a press on a control's native view: a click at its middle
        let pressed: Vec<(f32, f32)> =
            crate::take_presses_for(|k| native::find(&strip.controls, k).is_some())
                .into_iter()
                .filter_map(|k| native::find(&strip.controls, k))
                .map(|c| c.centre())
                .collect();
        let taps: Vec<(f32, f32)> = if which == 0 {
            crate::take_taps()
                .into_iter()
                .map(|(x, y)| (x / scale, y / scale))
                .collect()
        } else {
            vec![]
        };
        for (x, y) in pressed.into_iter().chain(taps) {
            let time = self.last;
            self.app.ui_event(place, Event::PointerMove { x, y }, time);
            self.app
                .ui_event(place, Event::PointerDown { x, y, button }, time);
            self.app
                .ui_event(place, Event::PointerUp { x, y, button }, time);
        }
        // typed into a native text field: its control focused (a click on
        // it), then given the text
        let strip = &self.surfaces[which];
        let typed: Vec<((f32, f32), String)> =
            crate::take_texts_for(|k| native::find(&strip.controls, k).is_some())
                .into_iter()
                .filter_map(|(k, t)| native::find(&strip.controls, k).map(|c| (c.centre(), t)))
                .collect();
        for ((x, y), text) in typed {
            let time = self.last;
            self.app
                .ui_event(place, Event::PointerDown { x, y, button }, time);
            self.app
                .ui_event(place, Event::PointerUp { x, y, button }, time);
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
            self.app.ui_event(
                place,
                Event::Touch {
                    id: 0,
                    phase,
                    x: x / scale,
                    y: y / scale,
                    time,
                },
                time,
            );
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

/// Whether a mesh's morph targets move it now.
fn morphed(m: &threers::core::Mesh) -> bool {
    m.morph_influences.iter().any(|&w| w != 0.0)
}

/// The geometry an object draws (a sprite's is the shell's quad).
fn geometry_of(kind: &ObjectKind) -> Option<&Arc<threers::core::BufferGeometry>> {
    match kind {
        ObjectKind::Mesh(m) => Some(&m.geometry),
        ObjectKind::SkinnedMesh(m) => Some(&m.geometry),
        ObjectKind::InstancedMesh(m) => Some(&m.geometry),
        ObjectKind::LineSegments(l) => Some(&l.geometry),
        ObjectKind::Points(p) => Some(&p.geometry),
        _ => None,
    }
}

/// The material an object draws with.
fn material_of(kind: &ObjectKind) -> Option<&Material> {
    match kind {
        ObjectKind::Mesh(m) => Some(&m.material),
        ObjectKind::SkinnedMesh(m) => Some(&m.material),
        ObjectKind::InstancedMesh(m) => Some(&m.material),
        ObjectKind::LineSegments(l) => Some(&l.material),
        ObjectKind::Points(p) => Some(&p.material),
        ObjectKind::Sprite(s) => Some(&s.material),
        _ => None,
    }
}

// --- the C side, for `app!` -------------------------------------------------

#[doc(hidden)]
/// # Safety
/// `v` null or from [`create`]; `len` null or writable.
pub unsafe fn controls<A: App>(v: *mut c_void, which: u32, len: *mut usize) -> *const u8 {
    let Some(json) = (unsafe { get::<A>(v) }).and_then(|v| v.controls(which as usize)) else {
        return std::ptr::null();
    };
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
/// As [`get`]; `keys` [`OBJECT_KEYS`] writable u64s, `floats`
/// [`OBJECT_FLOATS`] f32s.
#[doc(hidden)]
pub unsafe fn object<A: App>(v: *mut c_void, i: u32, keys: *mut u64, floats: *mut f32) -> bool {
    let Some((k, f)) = unsafe { get::<A>(v) }.and_then(|v| v.object(i as usize)) else {
        return false;
    };
    if keys.is_null() || floats.is_null() {
        return false;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(k.as_ptr(), keys, OBJECT_KEYS);
        std::ptr::copy_nonoverlapping(f.as_ptr(), floats, OBJECT_FLOATS);
    }
    true
}

/// # Safety
/// As [`get`]; `ptrs` 5 writable pointers (positions, normals, colours,
/// indices, tangents: 4 floats a vertex, or null), `counts` 3 u32s
/// (vertices, floats a colour, indices). The pointers hold until the next
/// call.
#[doc(hidden)]
pub unsafe fn geometry<A: App>(
    v: *mut c_void,
    i: u32,
    ptrs: *mut *const c_void,
    counts: *mut u32,
) -> bool {
    let Some((p, n, c, cs, ix, t)) = unsafe { get::<A>(v) }.and_then(|v| v.geometry(i as usize))
    else {
        return false;
    };
    if ptrs.is_null() || counts.is_null() {
        return false;
    }
    unsafe {
        *ptrs = p.as_ptr().cast();
        *ptrs.add(1) = n.as_ptr().cast();
        *ptrs.add(2) = if c.is_empty() {
            std::ptr::null()
        } else {
            c.as_ptr().cast()
        };
        *ptrs.add(3) = ix.as_ptr().cast();
        *ptrs.add(4) = if t.is_empty() {
            std::ptr::null()
        } else {
            t.as_ptr().cast()
        };
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
    let Some(v) = (unsafe { get::<A>(v) }) else {
        return std::ptr::null();
    };
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
pub unsafe fn texture<A: App>(
    v: *mut c_void,
    i: u32,
    w: *mut u32,
    h: *mut u32,
    len: *mut usize,
) -> *const u8 {
    let Some((bytes, tw, th)) = unsafe { get::<A>(v) }.and_then(|v| v.texture(i as usize)) else {
        return std::ptr::null();
    };
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
    let (Some(v), false) = (unsafe { get::<A>(v) }, out.is_null()) else {
        return;
    };
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
pub unsafe fn ui<A: App>(
    v: *mut c_void,
    which: u32,
    width: u32,
    height: u32,
    scale: f32,
    len: *mut usize,
) -> *const u8 {
    let Some(bytes) =
        (unsafe { get::<A>(v) }).and_then(|v| v.ui(which as usize, width, height, scale))
    else {
        return std::ptr::null();
    };
    if !len.is_null() {
        unsafe { *len = bytes.len() };
    }
    bytes.as_ptr()
}

/// # Safety
/// As [`Volume::draw_into`]; `v` from [`create`].
pub unsafe fn draw<A: App>(
    v: *mut c_void,
    q: [f32; 4],
    size: u32,
    texture: *mut c_void,
    command_buffer: *mut c_void,
) -> bool {
    match unsafe { get::<A>(v) } {
        Some(v) => unsafe { v.draw_into(q, size, texture, command_buffer) },
        None => false,
    }
}

/// # Safety
/// `v` from [`create`].
pub unsafe fn tap<A: App>(v: *mut c_void, u: f32, vv: f32) {
    if let Some(v) = unsafe { get::<A>(v) } {
        v.tap(u, vv);
    }
}

/// # Safety
/// As [`get`]; `len` writable. The floats hold until the next frame.
#[doc(hidden)]
pub unsafe fn lights<A: App>(v: *mut c_void, len: *mut usize) -> *const f32 {
    let (Some(v), false) = (unsafe { get::<A>(v) }, len.is_null()) else {
        return std::ptr::null();
    };
    let l = v.lights();
    unsafe { *len = l.len() };
    l.as_ptr()
}

/// # Safety
/// As [`get`].
#[doc(hidden)]
pub unsafe fn lighting<A: App>(v: *mut c_void) -> u32 {
    unsafe { get::<A>(v) }.map_or(0, |v| v.lighting() as u32)
}

/// # Safety
/// As [`get`].
#[doc(hidden)]
pub unsafe fn drawing<A: App>(v: *mut c_void) -> u32 {
    unsafe { get::<A>(v) }.map_or(0, |v| v.app.volume_drawing() as u32)
}

/// # Safety
/// As [`get`]; `out` 4 writable floats.
#[doc(hidden)]
pub unsafe fn background<A: App>(v: *mut c_void, out: *mut f32) {
    if let (Some(v), false) = (unsafe { get::<A>(v) }, out.is_null()) {
        unsafe { std::ptr::copy_nonoverlapping(v.background().as_ptr(), out, 4) };
    }
}

/// # Safety
/// As [`get`]; `len` writable. Null while unchanged; `len` 0 for none. The
/// floats hold until the next call.
#[doc(hidden)]
pub unsafe fn environment<A: App>(v: *mut c_void, width: u32, len: *mut usize) -> *const f32 {
    let (Some(v), false) = (unsafe { get::<A>(v) }, len.is_null()) else {
        return std::ptr::null();
    };
    let Some(e) = v.environment(width) else {
        return std::ptr::null();
    };
    unsafe { *len = e.len() };
    e.as_ptr()
}

/// # Safety
/// As [`get`]; `len` writable. The floats hold until the next call.
#[doc(hidden)]
pub unsafe fn instances<A: App>(v: *mut c_void, i: u32, len: *mut usize) -> *const f32 {
    let (Some(v), false) = (unsafe { get::<A>(v) }, len.is_null()) else {
        return std::ptr::null();
    };
    let m = v.instances(i as usize);
    unsafe { *len = m.len() };
    m.as_ptr()
}

/// # Safety
/// As [`get`]; `w`, `h`, `len` writable. The bytes hold until the next call.
#[doc(hidden)]
pub unsafe fn map<A: App>(
    v: *mut c_void,
    i: u32,
    slot: u32,
    w: *mut u32,
    h: *mut u32,
    len: *mut usize,
) -> *const u8 {
    if w.is_null() || h.is_null() || len.is_null() {
        return std::ptr::null();
    }
    let Some((bytes, tw, th)) = unsafe { get::<A>(v) }.and_then(|v| v.map(i as usize, slot as usize)) else {
        return std::ptr::null();
    };
    unsafe { (*w, *h, *len) = (tw, th, bytes.len()) };
    bytes.as_ptr()
}

/// # Safety
/// As [`get`].
#[doc(hidden)]
pub unsafe fn ui_hit<A: App>(v: *mut c_void, which: u32, x: f32, y: f32, scale: f32) -> bool {
    unsafe { get::<A>(v) }.is_some_and(|v| v.ui_hit(which as usize, x, y, scale))
}

/// # Safety
/// As [`get`]; `ray` 6 readable floats: the origin, then the direction.
#[doc(hidden)]
pub unsafe fn ray<A: App>(v: *mut c_void, ray: *const f32) {
    if let (Some(v), false) = (unsafe { get::<A>(v) }, ray.is_null()) {
        let r = unsafe { std::slice::from_raw_parts(ray, 6) };
        v.ray(Vector3::new(r[0], r[1], r[2]), Vector3::new(r[3], r[4], r[5]));
    }
}

/// # Safety
/// As [`get`].
#[doc(hidden)]
pub unsafe fn insets<A: App>(v: *mut c_void, top: f32, bottom: f32, left: f32, right: f32) {
    if let Some(v) = unsafe { get::<A>(v) } {
        v.set_insets([top, bottom, left, right]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::core::{Bone, InstancedMesh, Mesh, Object3D, Skeleton, SkinnedMesh};
    use threers::geometries::BoxGeometry;
    use threers::lights::{DirectionalLight, HemisphereLight};
    use threers::materials::{PhysicalMaterial, StandardMaterial};
    use threers::math::Color;
    use threers::textures::{Texture, TextureFormat};

    /// A box with a normal map, an instanced box, a box skinned to one bone
    /// that the app moves each frame, a light and a sky.
    struct Shapes {
        bone: ObjectId,
        t: f32,
    }

    impl App for Shapes {
        fn setup(scene: &mut Scene) -> Self {
            let normal = Texture::new(2, 2, TextureFormat::Rgba8Unorm, [128, 128, 255, 255].repeat(4));
            let lit = StandardMaterial {
                normal_map: Some(Arc::new(normal)),
                normal_scale: threers::math::Vector2::new(0.5, 0.5),
                ..Default::default()
            };
            scene.add(Object3D::mesh(Mesh::new(BoxGeometry::new(1.0, 1.0, 1.0), Material::Standard(lit))));
            let coat = PhysicalMaterial {
                clearcoat: 0.75,
                sheen: 1.0,
                sheen_color: Color::new(1.0, 0.0, 0.0),
                ..Default::default()
            };
            let mut many = InstancedMesh::new(BoxGeometry::new(0.1, 0.1, 0.1), Material::Physical(coat), 3);
            for (k, t) in many.transforms.iter_mut().enumerate() {
                *t = Matrix4::translation(Vector3::new(k as f32, 0.0, 0.0));
            }
            scene.add(Object3D::instanced_mesh(many));
            let bone = scene.add(Object3D::group());
            let skin = SkinnedMesh::from_unweighted(
                BoxGeometry::new(1.0, 1.0, 1.0),
                Material::Standard(StandardMaterial::default()),
                Skeleton::new(vec![Bone { node: bone, inverse_bind: Matrix4::identity() }]),
            );
            let mut o = Object3D::skinned_mesh(skin);
            o.cast_shadow = true;
            scene.add(o);
            scene.add_light(DirectionalLight::new(Color::new(1.0, 1.0, 1.0), 1.0));
            scene.add_light(HemisphereLight::new(Color::new(0.5, 0.5, 1.0), Color::new(0.2, 0.1, 0.0), 1.0));
            Shapes { bone, t: 0.0 }
        }

        fn update(&mut self, scene: &mut Scene, _frame: &Frame) {
            self.t += 1.0;
            scene.get_mut(self.bone).unwrap().position = Vector3::new(0.0, self.t, 0.0);
        }
    }

    fn by_kind(v: &mut Volume<Shapes>, flag: u32) -> usize {
        (0..v.items.len())
            .find(|&i| v.object(i).is_some_and(|(_, f)| f[22] as u32 & flag != 0))
            .expect("an object with that flag")
    }

    #[test]
    fn a_normal_map_brings_its_tangents_and_its_pixels() {
        let mut v = Volume::<Shapes>::new();
        assert_eq!(v.frame(), 3);
        let i = by_kind(&mut v, 4096);
        let (_, f) = v.object(i).unwrap();
        assert_eq!(f[29] as u32, 2, "a normal map, nothing else");
        assert_eq!(f[30], 0.5);
        let (p, _, _, _, _, t) = v.geometry(i).unwrap();
        assert_eq!(t.len(), p.len() / 3 * 4, "a tangent a vertex");
        // a unit vector and the bitangent's sign
        assert!((t[0] * t[0] + t[1] * t[1] + t[2] * t[2] - 1.0).abs() < 1e-3 && t[3].abs() == 1.0);
        let (px, w, h) = v.map(i, 1).unwrap();
        assert_eq!((px.len(), w, h), (16, 2, 2));
        assert!(v.map(i, 0).is_none());
    }

    #[test]
    fn instances_come_out_with_their_matrices_and_a_coat() {
        let mut v = Volume::<Shapes>::new();
        v.frame();
        let i = by_kind(&mut v, 1024);
        let (k, f) = v.object(i).unwrap();
        assert_eq!(f[36], 3.0);
        assert_eq!((f[27], f[28]), (0.75, 0.0));
        assert!((f[32] - 1.0).abs() < 1e-5 && f[33] == 0.0 && f[34] == 0.0, "the sheen, sRGB: {:?}", &f[32..35]);
        let m = v.instances(i).to_vec();
        assert_eq!(m.len(), 48);
        assert_eq!(m[16 + 12], 1.0, "the second one along x");
        // the same next frame: the same key
        v.frame();
        assert_eq!(v.object(i).unwrap().0[3], k[3]);
    }

    #[test]
    fn a_skinned_mesh_moves_its_vertices_and_its_pose_key() {
        let mut v = Volume::<Shapes>::new();
        v.frame();
        let i = by_kind(&mut v, 2048);
        let (k0, f) = v.object(i).unwrap();
        assert_ne!(f[22] as u32 & 512, 0, "it casts a shadow");
        let y0 = v.geometry(i).unwrap().0[1];
        v.frame();
        let (k1, _) = v.object(i).unwrap();
        assert_eq!(k0[1], k1[1], "its geometry is the same");
        assert_ne!(k0[3], k1[3], "its pose is not");
        let y1 = v.geometry(i).unwrap().0[1];
        assert!((y1 - y0 - 1.0).abs() < 1e-4, "the bone rose one: {y0} → {y1}");
    }

    #[test]
    fn lights_and_the_sky_are_read_once() {
        let mut v = Volume::<Shapes>::new();
        v.frame();
        assert_eq!(v.lights().len(), reality::LIGHT_FLOATS);
        assert_eq!(v.lighting(), reality::RealityLighting::Auto);
        let sky = v.environment(32).unwrap();
        assert_eq!(sky.len(), 32 * 16 * 4);
        assert!(sky[2] > sky[0], "blue above");
        assert!(v.environment(32).is_none(), "unchanged: nothing");
    }
}
