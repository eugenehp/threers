//! Fur by shells: a mesh drawn as a stack of offset copies of itself, each
//! cut down to the cross-sections of procedural strands.
//!
//! The technique real-time engines use for coats, plush, carpet and moss. The
//! surface is drawn [`FurMaterial::shells`] + 1 times in one instanced draw:
//! instance 0 is the skin, and instance `k` is pushed out along the normal to
//! `k / shells` of the fur's length, drooped by gravity (and wind, and its own
//! inertia), and kept only where a strand passes through it.
//!
//! ```
//! use threers::fur::{fur_object, FurGroom, FurMaterial};
//! use threers::prelude::*;
//!
//! let mut scene = Scene::new();
//! let fur = FurMaterial {
//!     groom: FurGroom { length: 0.15, density: 120.0, ..Default::default() },
//!     root_color: Color::from_hex(0x3a2614),
//!     tip_color: Color::from_hex(0xc8a070),
//!     ..Default::default()
//! };
//! scene.add(fur_object(SphereGeometry::new(1.0, 64, 32), &fur));
//! ```
//!
//! **Where strands grow** ([`FurPlacement`]). Over the mesh's UVs by default —
//! a grid fitted to the UV stretch, which tiles across a UV seam but follows
//! the UV layout, poles and all. [`FurPlacement::Surface`] scatters them over
//! the surface itself in object space instead, with no UVs needed and no seams
//! or poles, for meshes that do not deform (on a skinned mesh, posing would
//! slide the surface under the strands, so it keeps to UVs).
//!
//! **Motion.** Gravity is a world-space droop, so it stays down however the
//! object turns. Wind ([`FurMaterial::wind`]) comes in gusts that travel across
//! the coat downwind; each strand also flutters on its own phase
//! ([`FurMaterial::flutter`]). A [`FurAnimator`] adds inertia — fur that trails
//! a moving object and springs back when it stops ([`FurDynamics`]) — and
//! carries fur on a skinned mesh through its animation: call
//! [`FurAnimator::update`] once a frame.
//!
//! Shading is Kajiya-Kay: strands are lit along their tangent rather than
//! across a surface normal, with a shifted specular highlight, and darkened
//! towards the root. Shadows from every shadow-casting light fall on the fur.
//!
//! **Painting the groom.** A geometry with a `color` attribute tints the fur
//! by its RGB and scales its length by its alpha: alpha 0 is bald, 0.5 is
//! half length. Without one the fur is uniform.
//!
//! **What it runs on.** WebGPU, WebGL2 and native wgpu, through
//! [`ShaderMaterial`](crate::ShaderMaterial), and the Metal renderer through
//! its own port of the same shader (`tests/fur.rs`, `tests/fur_metal.rs` and
//! the browser suites).
//!
//! **Strand hair** ([`strands`]) takes the same [`FurGroom`] and
//! [`FurMaterial`] to grow real curves from the surface instead, drawn as
//! camera-facing ribbons — for manes, long hair and close-ups.

pub mod strands;

use std::sync::Arc;

use crate::core::{
    BufferAttribute, BufferGeometry, InstancedMesh, Mesh, Object3D, ObjectId, ObjectKind,
};
use crate::materials::{Material, ShaderMaterial};
use crate::math::{Color, Matrix4, Vector3};
use crate::scene::Scene;

/// Where strands are rooted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FurPlacement {
    /// On a grid over the mesh's UVs, fitted to their stretch:
    /// [`FurGroom::density`] strands per UV unit along each axis.
    Uv,
    /// Scattered over the surface in object space, one strand per `cell × cell`
    /// of area (object units). Needs no UVs and has no seams or poles; for
    /// meshes that do not deform.
    Surface { cell: f32 },
}

/// How the fur grows: what both shell fur and (later) strand hair are
/// grown from.
#[derive(Debug, Clone, PartialEq)]
pub struct FurGroom {
    /// Full strand length, in world units.
    pub length: f32,
    /// Strands per UV unit along each axis, for [`FurPlacement::Uv`].
    pub density: f32,
    pub placement: FurPlacement,
    /// Root radius, as a fraction of a strand's cell (0.5 touches the
    /// neighbours).
    pub thickness: f32,
    /// Tip radius as a fraction of the root's: 0 comes to a point, 1 is a
    /// cylinder.
    pub taper: f32,
    /// How far down the tips droop, in world units per unit of length: the
    /// strand bends quadratically from straight out at the root.
    pub gravity: Vector3,
    /// Turns each strand makes around its own root over its length.
    pub curl: f32,
    /// 0..1: how far tips are drawn together into clumps (tufts).
    pub clumping: f32,
    /// Clump size, in strands.
    pub clump_size: f32,
    /// 0..1: how much shorter than `length` an individual strand may be.
    pub length_variation: f32,
    /// Changes the scatter, not its statistics.
    pub seed: u32,
}

impl Default for FurGroom {
    fn default() -> Self {
        Self {
            length: 0.1,
            density: 80.0,
            placement: FurPlacement::Uv,
            thickness: 0.35,
            taper: 0.1,
            gravity: Vector3::new(0.0, -0.3, 0.0),
            curl: 0.0,
            clumping: 0.3,
            clump_size: 4.0,
            length_variation: 0.4,
            seed: 0,
        }
    }
}

/// How fur answers motion, for a [`FurAnimator`]: a damped spring per object
/// (rigid meshes) or per vertex (skinned ones) that the fur's tips trail by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FurDynamics {
    /// Spring rate, 1/s²: how fast the fur returns. ~100 is a short coat,
    /// ~20 a long, lazy one.
    pub stiffness: f32,
    /// 0..1, of critical damping: 1 settles without overshoot, lower sways.
    pub damping: f32,
    /// How strongly motion moves the tips: 1 is a free mass on the spring,
    /// so a steady acceleration `a` holds the tips `a / stiffness` world units
    /// behind; 0 is fur that ignores motion.
    pub response: f32,
    /// The furthest the tips are swung, as a fraction of the strand length.
    pub max_lag: f32,
}

impl Default for FurDynamics {
    fn default() -> Self {
        Self {
            stiffness: 80.0,
            damping: 0.35,
            response: 1.0,
            max_lag: 1.2,
        }
    }
}

/// Shell fur's look: the groom, its colour and sheen, and how it moves.
#[derive(Debug, Clone, PartialEq)]
pub struct FurMaterial {
    pub groom: FurGroom,
    /// Shells above the skin. More is smoother and costs a draw of the mesh
    /// each; 16–48 is the usual range.
    pub shells: u32,
    pub root_color: Color,
    pub tip_color: Color,
    /// 0..1: how much darker the root is than the tip, from self-shadowing.
    pub occlusion: f32,
    /// Strength of the Kajiya-Kay highlight.
    pub specular: f32,
    /// Sharpness of the highlight (a Phong-style exponent along the strand).
    pub specular_exponent: f32,
    /// Moves the highlight along the strand, towards the tip when positive.
    pub specular_shift: f32,
    /// The wind, world units (metres) a second. It pulls the strands by drag
    /// — as the square of the air's speed past them, so a coat moving
    /// through still air streams back too (with a [`FurAnimator`]) — and
    /// carries its gusts downwind at its own speed.
    pub wind: Vector3,
    /// Gusts a second: eddies `|wind| / wind_frequency` across, carried past
    /// a point at the wind's speed.
    pub wind_frequency: f32,
    /// Air drag on the strands: the pull of air going by at 1 unit a second.
    /// At the default 0.15, 4 m/s pulls about as hard as gravity's default
    /// pulls down; fine, light fur more, coarse fur less.
    pub drag: f32,
    /// The gusts' strength: the turbulent velocity as a share of the wind's
    /// (0.35; 0 for a steady wind).
    pub turbulence: f32,
    /// How far each strand flutters on its own in the flow, as a fraction of
    /// a strand's cell — faster in a faster wind, and none without one.
    pub flutter: f32,
    pub dynamics: FurDynamics,
    /// How far up the coat its shadow reaches, as a share of the fur's
    /// length: the skin pushed out this far (and drooped as the fur is) is
    /// drawn, unseen, into the shadow maps. 0 casts the skin's shadow alone
    /// on Metal and none elsewhere. Strand hair ignores it (see
    /// [`StrandOptions::shadow_strands`](strands::StrandOptions::shadow_strands)).
    pub shadow_height: f32,
    /// Draw fewer shells where the fur is small on screen — two per pixel of
    /// its length (one leaves gaps at the silhouette, where shells are seen
    /// edge-on), at least four — so a distant coat costs what it shows.
    /// On by default; it needs the geometry's bounds, so it applies to
    /// [`fur_object`] and [`shader_material_for`](Self::shader_material_for),
    /// not [`shader_material`](Self::shader_material).
    pub lod: bool,
}

impl Default for FurMaterial {
    fn default() -> Self {
        Self {
            groom: FurGroom::default(),
            shells: 32,
            root_color: Color::from_hex(0x2b1d10),
            tip_color: Color::from_hex(0xb08a5a),
            occlusion: 0.6,
            specular: 0.25,
            specular_exponent: 60.0,
            specular_shift: 0.1,
            wind: Vector3::ZERO,
            wind_frequency: 0.6,
            drag: 0.15,
            turbulence: 0.35,
            flutter: 0.0,
            dynamics: FurDynamics::default(),
            shadow_height: 0.6,
            lod: true,
        }
    }
}

// Shader data slots (`u_data.data[i]`).
const TIME: (usize, usize) = (5, 2);
const LAG_IN_TANGENT: (usize, usize) = (8, 3);
const OBJECT_LAG: usize = 9;
/// The placement slot; [`HAIR_PLACEMENT`] there marks strand hair, which the
/// fragment stage draws without the shells' cut-out.
const HAIR: (usize, usize) = (8, 0);
const HAIR_PLACEMENT: f32 = 2.0;

impl FurMaterial {
    /// The shader material that draws it, for an [`InstancedMesh`] of
    /// [`shells`](Self::shells) + 1 identity instances ([`fur_object`] builds
    /// that), assuming the mesh's UVs are as long in `u` as in `v`.
    /// [`shader_material_for`](Self::shader_material_for) measures them.
    pub fn shader_material(&self) -> ShaderMaterial {
        self.shader_material_with_aspect(1.0)
    }

    /// [`shader_material`](Self::shader_material) with the strand grid fitted
    /// to `geometry`'s UVs, so strands come out round and evenly spaced in
    /// world space however the UVs are stretched — a sphere's `u` runs round
    /// twice the distance its `v` does.
    pub fn shader_material_for(&self, geometry: &BufferGeometry) -> ShaderMaterial {
        let mut m = self.shader_material_with_aspect(uv_aspect(geometry));
        if self.lod {
            if let Some((c, r)) = bounds(geometry) {
                m.data[FUR_LOD] = [c.x, c.y, c.z, r];
            }
        }
        m
    }

    /// The shader's uniform slots — also what the Metal port reads.
    pub(crate) fn data(&self, aspect: f32) -> Vec<[f32; 4]> {
        let g = &self.groom;
        // Whole cells per UV unit on each axis, so the strands tile across a
        // UV seam; the same count overall as `density²`.
        let a = aspect.clamp(1.0 / 64.0, 64.0).sqrt();
        let density = g.density.max(1.0);
        let cells = [
            (density * a).round().max(1.0),
            (density / a).round().max(1.0),
        ];
        let clumps = [
            (cells[0] / g.clump_size.max(1.0)).round().max(1.0),
            (cells[1] / g.clump_size.max(1.0)).round().max(1.0),
        ];
        let (placement, cell) = match g.placement {
            FurPlacement::Uv => (0.0, 1.0),
            FurPlacement::Surface { cell } => (1.0, cell.max(1e-5)),
        };
        let rgb = |c: Color| [c.r, c.g, c.b];
        let [rr, rg, rb] = rgb(self.root_color);
        let [tr, tg, tb] = rgb(self.tip_color);
        vec![
            [
                self.shells.max(1) as f32,
                g.length,
                density,
                g.thickness.clamp(0.0, 0.5),
            ],
            [
                g.gravity.x,
                g.gravity.y,
                g.gravity.z,
                g.taper.clamp(0.0, 1.0),
            ],
            [rr, rg, rb, self.occlusion.clamp(0.0, 1.0)],
            [tr, tg, tb, self.specular.max(0.0)],
            [
                g.curl,
                g.clumping.clamp(0.0, 1.0),
                g.length_variation.clamp(0.0, 1.0),
                (g.seed % 65536) as f32,
            ],
            [
                self.specular_exponent.max(1.0),
                self.specular_shift,
                0.0,
                g.clump_size.max(1.0),
            ],
            [self.wind.x, self.wind.y, self.wind.z, self.wind_frequency],
            [cells[0], cells[1], clumps[0], clumps[1]],
            [placement, cell, self.flutter.max(0.0), 0.0],
            [0.0; 4],
            // The shadow caster's layout, when there is one.
            [0.0; 4],
            // The level of detail's bounding sphere (local centre, radius);
            // a negative radius turns it off.
            [0.0, 0.0, 0.0, -1.0],
            // The air: drag, turbulence.
            [self.drag.max(0.0), self.turbulence.max(0.0), 0.0, 0.0],
            // A hand in the coat (`FurAnimator::stroke`): where and how far
            // (none: radius 0), and its push and parting.
            [0.0; 4],
            [0.0; 4],
        ]
    }

    fn shader_material_with_aspect(&self, aspect: f32) -> ShaderMaterial {
        ShaderMaterial {
            fragment: FUR_WGSL.to_string(),
            vertex: Some(FUR_VERTEX_WGSL.to_string()),
            data: self.data(aspect),
            // Shells cut, not blend: each is opaque where a strand passes and
            // discarded elsewhere, so they need no sorting.
            transparent: false,
            side: 2,
            // Strand edges and strands thinner than a pixel by coverage —
            // natively; a browser's per-sample shading costs too much (see
            // `FUR_PER_SAMPLE` in the shader).
            alpha_to_coverage: !cfg!(target_arch = "wasm32"),
            // What a backend without custom WGSL draws if it has no fur of
            // its own: a silhouette in the coat's overall colour.
            fallback_color: self.root_color.lerp(self.tip_color, 0.6),
            ..Default::default()
        }
    }
}

/// `geometry` with `fur` grown on it: an object to add to a scene.
/// It casts shadows (see [`FurMaterial::shadow_height`]) and receives them —
/// the coat shadowing itself, and anything else on it; turn
/// [`receive_shadow`](Object3D::receive_shadow) off to light it flat.
pub fn fur_object(geometry: BufferGeometry, fur: &FurMaterial) -> Object3D {
    let mut object = Object3D::instanced_mesh(fur_mesh(geometry, fur));
    object.receive_shadow = true;
    object.cast_shadow = fur.shadow_height > 0.0;
    object
}

/// The [`InstancedMesh`] behind [`fur_object`]: one identity instance for the
/// skin and one per shell.
///
/// With a [`shadow_height`](FurMaterial::shadow_height) the geometry gains a
/// second group after its own — the coat's shadow caster — drawn with an
/// invisible material, and the mesh draws its groups.
pub fn fur_mesh(geometry: BufferGeometry, fur: &FurMaterial) -> InstancedMesh {
    let mut material = fur.shader_material_for(&geometry);
    let mut geometry = geometry;
    let count = fur.shells.max(1) as usize + 1;
    let vertices = geometry.get_attribute("position").map_or(0, |p| p.count());
    if fur.shadow_height <= 0.0 || vertices == 0 {
        return InstancedMesh::from_arc(
            Arc::new(geometry),
            Arc::new(Material::Shader(material)),
            count,
        );
    }
    // The caster's groups need the skin indexed.
    let index = geometry
        .index
        .clone()
        .unwrap_or_else(|| (0..vertices as u32).collect());
    let indices = index.len();
    geometry.set_index(index);
    material.data[FUR_SHADOW] = [vertices as f32, indices as f32, fur.shadow_height, 0.0];
    refresh_fur_shadow(&mut geometry, &material.data, &Matrix4::identity());
    let fur_material = Arc::new(Material::Shader(material));
    let mut mesh = InstancedMesh::from_arc(Arc::new(geometry), fur_material.clone(), count);
    mesh.materials = vec![fur_material, Arc::new(Material::Basic(invisible_caster()))];
    mesh
}

/// A material drawn into the shadow maps and nowhere else.
pub(crate) fn invisible_caster() -> crate::materials::BasicMaterial {
    let mut caster = crate::materials::BasicMaterial::default();
    caster.render_state.color_write = false;
    caster.render_state.depth_write = Some(false);
    caster
}

/// The material data slot of fur's level of detail: the geometry's local
/// bounding sphere, or a negative radius for none.
const FUR_LOD: usize = 11;

/// A sphere round `geometry`'s positions: its box's centre and the furthest
/// position from it.
fn bounds(geometry: &BufferGeometry) -> Option<(Vector3, f32)> {
    let p = &geometry.get_attribute("position")?.array;
    if p.len() < 3 {
        return None;
    }
    let (mut lo, mut hi) = (
        Vector3::new(p[0], p[1], p[2]),
        Vector3::new(p[0], p[1], p[2]),
    );
    for q in p.chunks_exact(3) {
        lo = Vector3::new(lo.x.min(q[0]), lo.y.min(q[1]), lo.z.min(q[2]));
        hi = Vector3::new(hi.x.max(q[0]), hi.y.max(q[1]), hi.z.max(q[2]));
    }
    let c = (lo + hi) * 0.5;
    let r = p
        .chunks_exact(3)
        .map(|q| (Vector3::new(q[0], q[1], q[2]) - c).length())
        .fold(0.0, f32::max);
    Some((c, r))
}

/// The material data slot of fur's shadow caster: the skin's vertex and
/// index counts, and the share of the length the caster reaches. (Strand
/// hair keeps its own caster's layout in the same slot.)
const FUR_SHADOW: usize = 10;

/// Rebuild fur's shadow caster: the skin (the geometry's first `vertices`
/// vertices and `indices` indices) pushed out along its normals to
/// `shadow_height` of each vertex's fur length and drooped by gravity, wind
/// and inertia as `FUR_VERTEX_WGSL` droops a shell at that height, for an
/// object at `model`. Appended after the skin as group 1; every other
/// attribute padded.
fn refresh_fur_shadow(geometry: &mut BufferGeometry, data: &[[f32; 4]], model: &Matrix4) {
    if let Some(caster) = fur_caster(geometry, data, model) {
        append_caster(geometry, &caster);
    }
}

// ---- The air on the CPU: the shaders' (`fur_noise3`, `fur_mean_pull`,
// `fur_turbulence`, `fur_flutter_noise`), so the shadow casters bend as the
// drawn strands do ------------------------------------------------------------

/// `fur_hash3`: the PCG3D hash, three numbers in [0, 1).
fn hash3(p: [i32; 3]) -> [f32; 3] {
    let mut v = p.map(|x| (x as u32).wrapping_mul(1664525).wrapping_add(1013904223));
    let mix = |v: &mut [u32; 3]| {
        v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
        v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
        v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    };
    mix(&mut v);
    v = v.map(|x| x ^ (x >> 16));
    mix(&mut v);
    v.map(|x| (x >> 8) as f32 * (1.0 / 16_777_216.0))
}

/// `fur_noise3`: smooth value noise, a vector in [-1, 1]³.
fn noise3(p: Vector3) -> Vector3 {
    let i = [p.x.floor(), p.y.floor(), p.z.floor()];
    let f = [p.x - i[0], p.y - i[1], p.z - i[2]];
    let u = f.map(|t| t * t * (3.0 - 2.0 * t));
    let b = i.map(|x| x as i32);
    let h = |x: i32, y: i32, z: i32| {
        let v = hash3([b[0] + x, b[1] + y, b[2] + z]);
        Vector3::new(v[0], v[1], v[2])
    };
    let mix = |a: Vector3, c: Vector3, t: f32| a + (c - a) * t;
    let x00 = mix(h(0, 0, 0), h(1, 0, 0), u[0]);
    let x10 = mix(h(0, 1, 0), h(1, 1, 0), u[0]);
    let x01 = mix(h(0, 0, 1), h(1, 0, 1), u[0]);
    let x11 = mix(h(0, 1, 1), h(1, 1, 1), u[0]);
    let v = mix(mix(x00, x10, u[1]), mix(x01, x11, u[1]), u[2]);
    v * 2.0 - Vector3::new(1.0, 1.0, 1.0)
}

fn slot3(data: &[[f32; 4]], i: usize) -> Vector3 {
    Vector3::new(data[i][0], data[i][1], data[i][2])
}

/// `fur_mean_pull`.
pub(crate) fn mean_pull(data: &[[f32; 4]]) -> Vector3 {
    if data[9][3] > 0.5 {
        return slot3(data, 9);
    }
    let w = slot3(data, 6);
    slot3(data, 1) + w * (data[12][0] * w.length())
}

/// `fur_turbulence` at world point `p`.
pub(crate) fn turbulence(data: &[[f32; 4]], p: Vector3) -> Vector3 {
    let w = slot3(data, 6);
    let wl = w.length();
    if wl <= 0.0 || data[12][1] <= 0.0 {
        return Vector3::ZERO;
    }
    let eddy = (wl / data[6][3].max(1e-3)).max(0.05);
    noise3((p - w * data[5][2]) * (1.0 / eddy)) * (2.0 * data[12][0] * data[12][1] * wl * wl)
}

/// `fur_brush` at world point `p`.
pub(crate) fn brush(data: &[[f32; 4]], p: Vector3) -> Vector3 {
    let Some(b) = data.get(13) else {
        return Vector3::ZERO;
    };
    if b[3] <= 0.0 {
        return Vector3::ZERO;
    }
    let off = p - Vector3::new(b[0], b[1], b[2]);
    let d = off.length();
    if d >= b[3] {
        return Vector3::ZERO;
    }
    let f = 1.0 - d / b[3];
    let w = f * f * (3.0 - 2.0 * f);
    (slot3(data, 14) + off * (data[14][3] / d.max(1e-4))) * w
}

/// `fur_flutter_noise` for a strand's random `r`.
pub(crate) fn flutter_noise(data: &[[f32; 4]], r: f32) -> Vector3 {
    let wl = slot3(data, 6).length();
    let amp = data[8][2] * wl.min(1.0);
    if amp <= 0.0 {
        return Vector3::ZERO;
    }
    let rate = 2.0 + 4.0 * wl;
    noise3(Vector3::new(r * 97.0, data[5][2] * rate, r * 31.0)) * amp
}

/// Up a strand of unit length from its root normal `n` to height `h` under
/// `pull`, at constant speed (the shaders' walk).
pub(crate) fn walk(n: Vector3, pull: Vector3, h: f32) -> Vector3 {
    let mut w = Vector3::ZERO;
    for k in 0..6 {
        let d = n + pull * (2.0 * h * (k as f32 + 0.5) / 6.0);
        w = w + d * (1.0 / d.length().max(1e-6));
    }
    w * (h / 6.0)
}

/// The pushed-out skin of [`refresh_fur_shadow`], on its own, in the
/// geometry's local space.
fn fur_caster(geometry: &BufferGeometry, data: &[[f32; 4]], model: &Matrix4) -> Option<Caster> {
    let &[vertices, indices, height, _] = data.get(FUR_SHADOW)?;
    let (vertices, indices) = (vertices as usize, indices as usize);
    if vertices == 0 {
        return None;
    }
    let (pos, nrm) = (
        geometry.get_attribute("position")?,
        geometry.get_attribute("normal")?,
    );
    let index = geometry.index.as_ref()?;
    if pos.array.len() < vertices * 3 || nrm.array.len() < vertices * 3 || index.len() < indices {
        return None;
    }
    let alpha = geometry.get_attribute("color").filter(|c| c.item_size >= 4);
    let lags = geometry
        .get_attribute("tangent")
        .filter(|t| t.item_size >= 4 && data[LAG_IN_TANGENT.0][LAG_IN_TANGENT.1] > 0.5);
    let v3 = |a: &[f32], i: usize, k: usize| Vector3::new(a[i * k], a[i * k + 1], a[i * k + 2]);
    let to_local = model.invert();
    let mean = mean_pull(data);
    let mut position = Vec::with_capacity(vertices * 3);
    for v in 0..vertices {
        let wp = v3(&pos.array, v, 3).apply_matrix4(model);
        let n = direction(model, v3(&nrm.array, v, 3));
        let n = n * (1.0 / n.length().max(1e-9));
        let len =
            data[0][1] * alpha.map_or(1.0, |c| c.array.get(v * 4 + 3).copied().unwrap_or(1.0));
        let mut pull = mean + turbulence(data, wp) + brush(data, wp);
        if let Some(t) = lags {
            if t.array.len() >= v * 4 + 3 {
                pull = pull + direction(model, v3(&t.array, v, 4));
            }
        }
        let q = (wp + walk(n, pull, height) * len).apply_matrix4(&to_local);
        position.extend_from_slice(&[q.x, q.y, q.z]);
    }
    Some(Caster {
        after_vertices: vertices,
        after_indices: indices,
        position,
        normal: nrm.array[..vertices * 3].to_vec(),
        index: index[..indices].to_vec(),
    })
}

/// A shadow caster's vertices — positions, normals, triangles — in the
/// local space of the object it shadows for, and how much of that object's
/// geometry comes before it when it is kept inside it.
pub(crate) struct Caster {
    pub(crate) after_vertices: usize,
    pub(crate) after_indices: usize,
    pub(crate) position: Vec<f32>,
    pub(crate) normal: Vec<f32>,
    pub(crate) index: Vec<u32>,
}

impl Caster {
    /// The caster as a geometry of its own.
    fn geometry(self) -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(self.position, 3));
        g.set_attribute("normal", BufferAttribute::new(self.normal, 3));
        g.set_index(self.index);
        g
    }
}

/// Put `caster` after the drawn part of `geometry`, as group 1: positions and
/// normals rewritten, every other attribute padded, the index and groups
/// rebuilt only when their size changes.
fn append_caster(geometry: &mut BufferGeometry, caster: &Caster) {
    let (keep, keep_index) = (caster.after_vertices, caster.after_indices);
    let total = keep + caster.position.len() / 3;
    let (Some(pos), Some(nrm)) = (
        geometry.get_attribute("position"),
        geometry.get_attribute("normal"),
    ) else {
        return;
    };
    let mut position = pos.array[..(keep * 3).min(pos.array.len())].to_vec();
    position.resize(keep * 3, 0.0);
    position.extend_from_slice(&caster.position);
    let mut normal = nrm.array[..(keep * 3).min(nrm.array.len())].to_vec();
    normal.resize(keep * 3, 0.0);
    normal.extend_from_slice(&caster.normal);
    geometry.set_attribute("position", BufferAttribute::new(position, 3));
    geometry.set_attribute("normal", BufferAttribute::new(normal, 3));
    pad_attributes(geometry, keep, total);
    let have = geometry.index.as_ref().map_or(0, |i| i.len());
    if have != keep_index + caster.index.len() {
        let mut index: Vec<u32> = geometry
            .index
            .as_ref()
            .map_or_else(Vec::new, |i| i[..keep_index.min(i.len())].to_vec());
        index.extend(caster.index.iter().map(|&k| k + keep as u32));
        geometry.set_index(index);
        geometry.groups.clear();
        geometry.add_group(0, keep_index as u32, 0);
        geometry.add_group(keep_index as u32, caster.index.len() as u32, 1);
    }
}

/// Every attribute but `position` and `normal` cut to its first `keep`
/// vertices and padded with zeros to `total` — where a shadow caster's
/// vertices follow the drawn ones.
pub(crate) fn pad_attributes(geometry: &mut BufferGeometry, keep: usize, total: usize) {
    let names: Vec<String> = geometry
        .attributes
        .keys()
        .filter(|n| *n != "position" && *n != "normal")
        .cloned()
        .collect();
    for name in names {
        let Some(a) = geometry.get_attribute(&name) else {
            continue;
        };
        let size = a.item_size;
        if a.array.len() != total * size {
            let mut v = a.array[..(keep * size).min(a.array.len())].to_vec();
            v.resize(total * size, 0.0);
            geometry.set_attribute(&name, BufferAttribute::new(v, size));
        }
    }
}

/// Whether `material` is fur drawn by this module.
pub(crate) fn is_fur(material: &ShaderMaterial) -> bool {
    matches!(
        material.vertex.as_deref(),
        Some(FUR_VERTEX_WGSL | strands::HAIR_VERTEX_WGSL)
    )
}

/// The fur shader material on `object`, if it is a fur object.
fn fur_material_mut(object: &mut Object3D) -> Option<&mut ShaderMaterial> {
    let material = match &mut object.kind {
        ObjectKind::InstancedMesh(im) => match im.materials.first_mut() {
            Some(first) => first,
            None => &mut im.material,
        },
        // Hair with a shadow caster draws its groups: the hair is the first.
        ObjectKind::Mesh(m) => match m.materials.first_mut() {
            Some(first) => first,
            None => &mut m.material,
        },
        _ => return None,
    };
    match Arc::make_mut(material) {
        Material::Shader(m) if is_fur(m) => Some(m),
        _ => None,
    }
}

/// Advance a fur object's clock — its wind and flutter — to `seconds`. A
/// no-op on anything that is not fur. ([`FurAnimator::update`] does this.)
pub fn set_fur_time(object: &mut Object3D, seconds: f32) {
    if let Some(m) = fur_material_mut(object) {
        if let Some(slot) = m.data.get_mut(TIME.0) {
            slot[TIME.1] = seconds;
        }
    }
}

/// How much farther a step in `u` goes over the surface than the same step
/// in `v`: the area-weighted mean of |∂P/∂u| over |∂P/∂v| across the
/// triangles. 1 without UVs, or when they are degenerate.
fn uv_aspect(geometry: &BufferGeometry) -> f32 {
    let (Some(pos), Some(uv)) = (
        geometry.get_attribute("position"),
        geometry.get_attribute("uv"),
    ) else {
        return 1.0;
    };
    let p = |i: usize| Vector3::new(pos.array[i * 3], pos.array[i * 3 + 1], pos.array[i * 3 + 2]);
    let t = |i: usize| (uv.array[i * 2], uv.array[i * 2 + 1]);
    let tris: Vec<[usize; 3]> = match &geometry.index {
        Some(idx) => idx
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect(),
        None => (0..pos.count() / 3)
            .map(|k| [3 * k, 3 * k + 1, 3 * k + 2])
            .collect(),
    };
    let (mut lu, mut lv) = (0.0f64, 0.0f64);
    for [a, b, c] in tris {
        let (e1, e2) = (p(b) - p(a), p(c) - p(a));
        let ((u0, v0), (u1, v1), (u2, v2)) = (t(a), t(b), t(c));
        let (du1, dv1, du2, dv2) = (u1 - u0, v1 - v0, u2 - u0, v2 - v0);
        let det = du1 * dv2 - du2 * dv1;
        if det.abs() < 1e-12 {
            continue;
        }
        let dpdu = (e1 * dv2 - e2 * dv1) * (1.0 / det);
        let dpdv = (e2 * du1 - e1 * du2) * (1.0 / det);
        let area = e1.cross(e2).length() as f64;
        lu += dpdu.length() as f64 * area;
        lv += dpdv.length() as f64 * area;
    }
    if lu > 0.0 && lv > 0.0 {
        (lu / lv) as f32
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Animation
// ---------------------------------------------------------------------------

/// A damped spring a fur tip trails its root by: `lag'' = -k·lag - c·lag' -
/// r·acceleration`.
#[derive(Debug, Clone, Copy)]
/// The strands' pull, per unit of length, as a damped oscillator: it rests
/// where gravity and the drag of the air going by put it, and the object's
/// acceleration swings it off that (inertia) — so a gust or a stop eases in
/// and overshoots rather than snapping.
struct Spring {
    pull: Vector3,
    velocity: Vector3,
    started: bool,
}

impl Default for Spring {
    fn default() -> Self {
        Self {
            pull: Vector3::ZERO,
            velocity: Vector3::ZERO,
            started: false,
        }
    }
}

impl Spring {
    /// Advance by `dt` towards `rest`, forced by `accel` on strands `len`
    /// long, keeping within `d.max_lag` of the rest.
    fn step(&mut self, rest: Vector3, accel: Vector3, len: f32, d: &FurDynamics, dt: f32) {
        if !self.started {
            self.pull = rest;
            self.started = true;
        }
        let k = d.stiffness.max(0.0);
        let c = 2.0 * d.damping.clamp(0.0, 4.0) * k.sqrt();
        // Substeps keep a stiff spring stable at a 30 Hz frame.
        let steps = ((dt * k.sqrt() / 0.3).ceil() as usize).clamp(1, 16);
        let h = dt / steps as f32;
        for _ in 0..steps {
            let force =
                (rest - self.pull) * k - self.velocity * c - accel * (d.response / len.max(1e-6));
            self.velocity = self.velocity + force * h;
            self.pull = self.pull + self.velocity * h;
        }
        let off = self.pull - rest;
        let max = d.max_lag.max(0.0);
        if off.length() > max && off.length() > 0.0 {
            self.pull = rest + off * (max / off.length());
            self.velocity = self.velocity * 0.5;
        }
    }
}

/// Where the air puts a strand moving at `velocity`: gravity, and drag as
/// the square of the air's speed past it — wind, less its own motion.
fn write_hand(m: &mut ShaderMaterial, hand: &Hand) {
    if m.data.len() > 14 {
        let [a, b] = hand.slots();
        m.data[13] = a;
        m.data[14] = b;
    }
}

fn rest_pull(air: &Air, velocity: Vector3) -> Vector3 {
    let rel = air.wind - velocity;
    air.gravity + rel * (air.drag * rel.length())
}

/// A hand in the coat: where (world), how far it reaches, its push (a pull,
/// along its motion) and its parting (a pull away from it).
#[derive(Clone, Copy)]
struct Hand {
    at: Vector3,
    radius: f32,
    push: Vector3,
    part: f32,
}

impl Default for Hand {
    fn default() -> Self {
        Self {
            at: Vector3::ZERO,
            radius: 0.0,
            push: Vector3::ZERO,
            part: 0.0,
        }
    }
}

impl Hand {
    /// A hand let go fades in about a fifth of a second, and the strands
    /// spring back; once too faint to see, it is gone.
    fn fade(&mut self, dt: f32) {
        let k = (-dt / 0.2).exp();
        self.push = self.push * k;
        self.part *= k;
        if self.push.length() + self.part.abs() < 0.03 {
            self.radius = 0.0;
        }
    }

    fn slots(&self) -> [[f32; 4]; 2] {
        [
            [self.at.x, self.at.y, self.at.z, self.radius],
            [self.push.x, self.push.y, self.push.z, self.part],
        ]
    }
}

/// What the animator needs of a fur material: gravity, the wind, the drag.
#[derive(Clone, Copy)]
struct Air {
    gravity: Vector3,
    wind: Vector3,
    drag: f32,
}

impl Air {
    fn of(object: &Object3D) -> Option<Self> {
        let m = match &object.kind {
            ObjectKind::InstancedMesh(m) => m.materials.first().unwrap_or(&m.material),
            ObjectKind::Mesh(m) => m.materials.first().unwrap_or(&m.material),
            _ => return None,
        };
        let Material::Shader(m) = &**m else {
            return None;
        };
        if !is_fur(m) || m.data.len() <= 12 {
            return None;
        }
        Some(Self {
            gravity: slot3(&m.data, 1),
            wind: slot3(&m.data, 6),
            drag: m.data[12][0],
        })
    }
}

/// A point's motion over the last frames, for its acceleration.
#[derive(Debug, Clone, Copy)]
struct Track {
    position: Vector3,
    velocity: Vector3,
    /// Frames seen: acceleration needs two velocities, so three positions.
    seen: u8,
}

impl Default for Track {
    fn default() -> Self {
        Self {
            position: Vector3::ZERO,
            velocity: Vector3::ZERO,
            seen: 0,
        }
    }
}

impl Track {
    /// The acceleration since the last call; zero until there are two
    /// velocities to compare, so a first frame is not a jump from rest.
    fn accel(&mut self, p: Vector3, dt: f32) -> Vector3 {
        let v = (p - self.position) * (1.0 / dt);
        let a = (v - self.velocity) * (1.0 / dt);
        self.position = p;
        self.seen = self.seen.saturating_add(1);
        let out = if self.seen >= 3 { a } else { Vector3::ZERO };
        if self.seen >= 2 {
            self.velocity = v;
        }
        out
    }
}

struct Tracked {
    fur: ObjectId,
    /// The skinned mesh the fur is grown on, re-posed every frame.
    skinned: Option<ObjectId>,
    dynamics: FurDynamics,
    /// The strand length, world units: the shader takes lag per unit of it.
    length: f32,
    track: Track,
    spring: Spring,
    vertex_tracks: Vec<Track>,
    vertex_springs: Vec<Spring>,
    /// Strand hair on a skinned mesh: its strands, rigged to the bind pose.
    /// (The springs above are then one per strand, at its root.)
    hair: Option<strands::HairRig>,
    /// A hand in the coat ([`FurAnimator::stroke`]), fading.
    hand: Hand,
    /// The shadow caster, once moved out of the object into a child of its
    /// own, and the inputs it was last bent for.
    caster: Option<ObjectId>,
    caster_sig: u64,
}

/// Animates fur: its clock (wind, flutter), its inertia, and fur grown on
/// skinned meshes. Call [`update`](Self::update) once a frame, after moving
/// things and before rendering.
///
/// ```
/// use threers::fur::{fur_object, FurAnimator, FurMaterial};
/// use threers::prelude::*;
///
/// let mut scene = Scene::new();
/// let fur = FurMaterial::default();
/// let ball = scene.add(fur_object(SphereGeometry::new(1.0, 32, 16), &fur));
/// let mut animator = FurAnimator::new();
/// animator.track(ball, &fur);
/// for frame in 0..3 {
///     scene.get_mut(ball).unwrap().position.x = frame as f32 * 0.1;
///     animator.update(&mut scene, frame as f32 / 60.0, 1.0 / 60.0);
/// }
/// ```
#[derive(Default)]
pub struct FurAnimator {
    tracked: Vec<Tracked>,
}

impl FurAnimator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Animate the fur object `fur`: its clock, and its inertia as a whole.
    pub fn track(&mut self, fur: ObjectId, material: &FurMaterial) {
        self.tracked.push(Tracked {
            fur,
            skinned: None,
            dynamics: material.dynamics,
            length: material.groom.length.max(1e-6),
            track: Track::default(),
            spring: Spring::default(),
            vertex_tracks: Vec::new(),
            vertex_springs: Vec::new(),
            hair: None,
            hand: Hand::default(),
            caster: None,
            caster_sig: 0,
        });
    }

    /// Grow fur on the skinned mesh `skinned` and animate it: a fur object
    /// added as the mesh's child, given the mesh's pose every frame, each
    /// vertex trailing its own motion. `None` if `skinned` is not a skinned
    /// mesh. Strands are rooted by UV (see [`FurPlacement`]).
    pub fn grow_on_skinned(
        &mut self,
        scene: &mut Scene,
        skinned: ObjectId,
        material: &FurMaterial,
    ) -> Option<ObjectId> {
        let ObjectKind::SkinnedMesh(sm) = &scene.get(skinned)?.kind else {
            return None;
        };
        let mut geometry = (*sm.geometry).clone();
        for name in ["joint", "weight"] {
            geometry.attributes.remove(name);
        }
        let mut uv_material = material.clone();
        uv_material.groom.placement = FurPlacement::Uv;
        let mut object = fur_object(geometry, &uv_material);
        if let Some(m) = fur_material_mut(&mut object) {
            m.data[LAG_IN_TANGENT.0][LAG_IN_TANGENT.1] = 1.0;
        }
        let fur = scene.add_to(skinned, object);
        self.tracked.push(Tracked {
            fur,
            skinned: Some(skinned),
            dynamics: material.dynamics,
            length: material.groom.length.max(1e-6),
            track: Track::default(),
            spring: Spring::default(),
            vertex_tracks: Vec::new(),
            vertex_springs: Vec::new(),
            hair: None,
            hand: Hand::default(),
            caster: None,
            caster_sig: 0,
        });
        Some(fur)
    }

    /// Grow strand hair ([`strands`]) on the skinned mesh `skinned` and
    /// animate it: a hair object added as the mesh's child, every strand
    /// carried by its triangle through the mesh's animation — moved with it,
    /// turned as the surface turns — and trailing its root's own motion on a
    /// spring. `None` if `skinned` is not a skinned mesh.
    pub fn grow_hair_on_skinned(
        &mut self,
        scene: &mut Scene,
        skinned: ObjectId,
        material: &FurMaterial,
        options: &strands::StrandOptions,
    ) -> Option<ObjectId> {
        self.hair_on_skinned(scene, skinned, material, options, |scene, hair| {
            scene.add_to(skinned, hair)
        })
    }

    /// [`grow_hair_on_skinned`](Self::grow_hair_on_skinned), with the hair
    /// put into the scene by `insert` — for a caller that links the graph
    /// itself (the JS scene graph). The hair must end up the skinned mesh's
    /// child, with no transform of its own.
    pub(crate) fn hair_on_skinned(
        &mut self,
        scene: &mut Scene,
        skinned: ObjectId,
        material: &FurMaterial,
        options: &strands::StrandOptions,
        insert: impl FnOnce(&mut Scene, Object3D) -> ObjectId,
    ) -> Option<ObjectId> {
        let ObjectKind::SkinnedMesh(sm) = &scene.get(skinned)?.kind else {
            return None;
        };
        let bind = sm.geometry.clone();
        let (grown, roots) =
            strands::grow_rooted(&bind, &material.groom, options.count, options.segments);
        let mut object = strands::hair_object_from(&grown, material, options);
        if let Some(m) = fur_material_mut(&mut object) {
            m.data[LAG_IN_TANGENT.0][LAG_IN_TANGENT.1] = 1.0;
        }
        let rig = strands::HairRig::new(&bind, grown, roots);
        let hair = insert(scene, object);
        self.tracked.push(Tracked {
            fur: hair,
            skinned: Some(skinned),
            dynamics: material.dynamics,
            length: material.groom.length.max(1e-6),
            track: Track::default(),
            spring: Spring::default(),
            vertex_tracks: Vec::new(),
            vertex_springs: Vec::new(),
            hair: Some(rig),
            hand: Hand::default(),
            caster: None,
            caster_sig: 0,
        });
        Some(hair)
    }

    /// Put a hand in the coat of `fur` (tracked): within `radius` of `at`
    /// (world), its strands are pushed by `push` — a pull, per unit of their
    /// length, along the hand's motion — and parted away from it by `part`,
    /// fading to nothing at the edge. Call it as the hand moves; let go and
    /// it fades over a fifth of a second, the strands springing back.
    pub fn stroke(&mut self, fur: ObjectId, at: Vector3, radius: f32, push: Vector3, part: f32) {
        if let Some(t) = self.tracked.iter_mut().find(|t| t.fur == fur) {
            t.hand = Hand {
                at,
                radius: radius.max(0.0),
                push,
                part,
            };
        }
    }

    /// Stop animating `fur` (it keeps its last pose).
    pub fn untrack(&mut self, fur: ObjectId) {
        self.tracked.retain(|t| t.fur != fur);
    }

    /// Advance everything tracked to `time` seconds, `dt` after the last call.
    /// Updates the scene's world matrices first.
    pub fn update(&mut self, scene: &mut Scene, time: f32, dt: f32) {
        if self.tracked.is_empty() {
            return;
        }
        let dt = dt.max(1e-4);
        scene.update_world();
        for t in &mut self.tracked {
            let Some(object) = scene.get(t.fur) else {
                continue;
            };
            let world = object.matrix_world;
            let Some(air) = Air::of(object) else {
                continue;
            };
            let accel = t.track.accel(translation(&world), dt);
            t.hand.fade(dt);
            let rest = rest_pull(&air, t.track.velocity);
            t.spring.step(rest, accel, t.length, &t.dynamics, dt);

            // A skinned mesh's pose, and every vertex's own pull — its own
            // motion through the air, its own inertia — into the fur's
            // geometry: positions and normals posed, the pull (in the mesh's
            // local frame) where fur keeps no tangents.
            let posed = t.skinned.and_then(|s| match &scene.get(s)?.kind {
                ObjectKind::SkinnedMesh(sm) => Some(sm.posed_geometry(&scene.get(s)?.matrix_world)),
                _ => None,
            });
            let (dynamics, length) = (t.dynamics, t.length);
            let point =
                |i: usize, wp: Vector3, tracks: &mut Vec<Track>, springs: &mut Vec<Spring>| {
                    let a = tracks[i].accel(wp, dt);
                    let rest = rest_pull(&air, tracks[i].velocity);
                    springs[i].step(rest, a, length, &dynamics, dt);
                    springs[i].pull
                };
            if let (Some(rig), Some(posed)) = (&t.hair, &posed) {
                // Strand hair: carry every strand on its triangle, with a
                // spring at each root.
                let (strands, roots) = rig.pose(posed);
                t.vertex_tracks.resize(roots.len(), Track::default());
                t.vertex_springs.resize(roots.len(), Spring::default());
                let to_local = world.invert();
                let pulls: Vec<Vector3> = roots
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        let p = point(
                            i,
                            r.apply_matrix4(&world),
                            &mut t.vertex_tracks,
                            &mut t.vertex_springs,
                        );
                        direction(&to_local, p)
                    })
                    .collect();
                let Some(object) = scene.get_mut(t.fur) else {
                    continue;
                };
                if let ObjectKind::Mesh(mesh) = &mut object.kind {
                    strands::HairRig::write(&strands, &pulls, Arc::make_mut(&mut mesh.geometry));
                }
                if let Some(m) = fur_material_mut(object) {
                    m.data[TIME.0][TIME.1] = time;
                    write_hand(m, &t.hand);
                    // Every strand on its own spring: nothing for the whole.
                    m.data[OBJECT_LAG] = [0.0, 0.0, 0.0, 1.0];
                }
                update_caster(scene, t, &world);
                continue;
            }
            let pulls = posed.as_ref().and_then(|g| {
                let p = &g.get_attribute("position")?.array;
                let n = p.len() / 3;
                t.vertex_tracks.resize(n, Track::default());
                t.vertex_springs.resize(n, Spring::default());
                let to_local = world.invert();
                let mut out = vec![0.0f32; n * 4];
                for v in 0..n {
                    let wp =
                        Vector3::new(p[v * 3], p[v * 3 + 1], p[v * 3 + 2]).apply_matrix4(&world);
                    let l = direction(
                        &to_local,
                        point(v, wp, &mut t.vertex_tracks, &mut t.vertex_springs),
                    );
                    out[v * 4..v * 4 + 3].copy_from_slice(&[l.x, l.y, l.z]);
                }
                Some(out)
            });

            let Some(object) = scene.get_mut(t.fur) else {
                continue;
            };
            if let (Some(posed), ObjectKind::InstancedMesh(im)) = (&posed, &mut object.kind) {
                let g = Arc::make_mut(&mut im.geometry);
                for name in ["position", "normal"] {
                    if let Some(a) = posed.get_attribute(name) {
                        g.set_attribute(name, a.clone());
                    }
                }
                if let Some(pulls) = &pulls {
                    g.set_attribute("tangent", BufferAttribute::new(pulls.clone(), 4));
                }
            }
            if let Some(m) = fur_material_mut(object) {
                m.data[TIME.0][TIME.1] = time;
                write_hand(m, &t.hand);
                // The coat's pull, for the shader to bend by (`w`: it is the
                // animator's). On a skinned mesh each vertex carries its own.
                let p = if pulls.is_some() {
                    Vector3::ZERO
                } else {
                    t.spring.pull
                };
                m.data[OBJECT_LAG] = [p.x, p.y, p.z, 1.0];
            }
            update_caster(scene, t, &world);
        }
    }
}

/// Keep a tracked object's shadow caster bent to how it is drawn now.
///
/// The first time, the caster moves out of the object's geometry into a
/// child mesh of its own: from then on, rebending it re-uploads its few
/// vertices rather than the whole fur or hair, and fur's is drawn once
/// rather than once a shell. After that it is rebent only when what bends
/// it changed — the object's world matrix, gravity, wind, inertia, the clock
/// (if anything moves with it), the drawn geometry's pose — and re-attached
/// should a scene graph have dropped it.
fn update_caster(scene: &mut Scene, t: &mut Tracked, world: &Matrix4) {
    let Some(object) = scene.get_mut(t.fur) else {
        return;
    };
    let casts = object.cast_shadow;
    let (geometry, material, materials) = match &mut object.kind {
        ObjectKind::Mesh(m) => (&mut m.geometry, &mut m.material, &mut m.materials),
        ObjectKind::InstancedMesh(m) => (&mut m.geometry, &mut m.material, &mut m.materials),
        _ => return,
    };
    let shader = materials.first().unwrap_or(material).clone();
    let Material::Shader(m) = &*shader else {
        return;
    };
    if !is_fur(m) || m.data.get(FUR_SHADOW).is_none_or(|c| c[0] <= 0.0) {
        return;
    }
    let hair = m.data[HAIR.0][HAIR.1] == HAIR_PLACEMENT;
    // What bends the caster. The clock only when something moves with it.
    let sig = {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x0100_0000_01b3);
        world.elements.iter().for_each(|v| mix(v.to_bits() as u64));
        for row in [1, 6, 8, 9, 13, 14] {
            m.data[row].iter().for_each(|v| mix(v.to_bits() as u64));
        }
        let moving = m.data[6][..3].iter().any(|&v| v != 0.0) || m.data[8][2] != 0.0;
        if moving {
            mix(m.data[TIME.0][TIME.1].to_bits() as u64);
        }
        mix(geometry.geometry_version as u64);
        mix(Arc::as_ptr(geometry) as usize as u64);
        h
    };
    let build = |g: &BufferGeometry| {
        if hair {
            strands::hair_caster(g, &m.data, world)
        } else {
            fur_caster(g, &m.data, world)
        }
    };
    match t.caster {
        None => {
            // Still inside: take it out. What is drawn keeps its own part.
            if materials.len() < 2 {
                return;
            }
            let Some(caster) = build(geometry) else {
                return;
            };
            let g = Arc::make_mut(geometry);
            let names: Vec<String> = g.attributes.keys().cloned().collect();
            for name in names {
                if let Some(a) = g.get_attribute(&name) {
                    let size = a.item_size;
                    let keep =
                        a.array[..(caster.after_vertices * size).min(a.array.len())].to_vec();
                    g.set_attribute(&name, BufferAttribute::new(keep, size));
                }
            }
            if let Some(index) = &g.index {
                let keep = index[..caster.after_indices.min(index.len())].to_vec();
                g.set_index(keep);
            }
            g.groups.clear();
            *material = materials[0].clone();
            materials.clear();
            let mut child = Object3D::mesh(Mesh::from_arc(
                Arc::new(caster.geometry()),
                Arc::new(Material::Basic(invisible_caster())),
            ));
            child.cast_shadow = casts;
            child.receive_shadow = false;
            t.caster = Some(scene.add_to(t.fur, child));
            // Bent for the geometry as it now is: the version moved.
            t.caster_sig = 0;
        }
        Some(id) => {
            if t.caster_sig == sig {
                return;
            }
            let caster = build(geometry);
            let Some(child) = scene.get_mut(id) else {
                t.caster = None;
                return;
            };
            child.cast_shadow = casts;
            if let (Some(c), ObjectKind::Mesh(mesh)) = (caster, &mut child.kind) {
                let g = Arc::make_mut(&mut mesh.geometry);
                g.set_attribute("position", BufferAttribute::new(c.position, 3));
                g.set_attribute("normal", BufferAttribute::new(c.normal, 3));
                if g.index.as_ref().map_or(0, |i| i.len()) != c.index.len() {
                    g.set_index(c.index);
                }
            }
            t.caster_sig = sig;
        }
    }
    // A scene graph that sets the object's children whole drops a child it
    // does not know of: put it back.
    if let Some(id) = t.caster {
        if scene.get(id).is_some_and(|c| c.parent != Some(t.fur)) {
            scene.arena.add_child(t.fur, id);
        }
    }
}

fn translation(m: &Matrix4) -> Vector3 {
    Vector3::new(m.elements[12], m.elements[13], m.elements[14])
}

/// `v` through `m`'s linear part (no translation).
fn direction(m: &Matrix4, v: Vector3) -> Vector3 {
    let e = &m.elements;
    Vector3::new(
        e[0] * v.x + e[4] * v.y + e[8] * v.z,
        e[1] * v.x + e[5] * v.y + e[9] * v.z,
        e[2] * v.x + e[6] * v.y + e[10] * v.z,
    )
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

/// The vertex stage: the standard instanced transform, then the shell's
/// offset. The shell's height goes to the fragment in `proj_uv`, the strand's
/// direction in `world_tangent` — fields fur has no other use for.
pub(crate) const FUR_VERTEX_WGSL: &str = r#"
// How many shells the fur needs now: two per pixel of its length on screen,
// measured at the near side of its bounding sphere — at least four, at most
// all of them. The same for every vertex of the object, so shells stay whole.
fn fur_lod_shells(shells : f32) -> f32 {
    let b = u_data.data[11];
    if (b.w < 0.0) {
        return shells;
    }
    let m = mesh.model;
    let scale = max(length(m[0].xyz), max(length(m[1].xyz), length(m[2].xyz)));
    let c = (m * vec4<f32>(b.xyz, 1.0)).xyz;
    let to_cam = frame.camera_position.xyz - c;
    let d = length(to_cam);
    let near = c + to_cam / max(d, 1e-6) * min(b.w * scale, d * 0.9);
    let up = vec3<f32>(frame.view[0][1], frame.view[1][1], frame.view[2][1]);
    let a = frame.view_proj * vec4<f32>(near, 1.0);
    let e = frame.view_proj * vec4<f32>(near + up, 1.0);
    if (a.w <= 1e-4 || e.w <= 1e-4) {
        return shells;
    }
    let px_per_unit = abs(e.y / e.w - a.y / a.w) * frame.viewport_size.y * 0.5;
    return clamp(ceil(2.0 * u_data.data[0].y * px_per_unit), min(4.0, shells), shells);
}

@vertex
fn vs_custom(in : InstancedCustomIn, @builtin(vertex_index) vid : u32,
             @builtin(instance_index) iid : u32) -> VsOut {
    var out = standard_vertex_instanced(instanced_in(in), vid, iid);
    let shells = fur_lod_shells(max(u_data.data[0].x, 1.0));
    // A shell beyond what the fur's size on screen needs: nothing to draw.
    if (f32(iid) > shells) {
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let h = f32(iid) / shells;
    let len = u_data.data[0].y * in.color.a;
    let n = normalize(out.world_normal);
    // What bends the strands here: the coat's pull (gravity, drag, inertia),
    // the gusts going by, and — on a skinned mesh — this vertex's own pull,
    // which the animator writes where fur has no tangents.
    var pull = fur_mean_pull() + fur_turbulence(out.world_pos) + fur_brush(out.world_pos);
    if (u_data.data[8].w > 0.5) {
        pull = pull + (mesh.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz;
    }
    // Up the strand to this shell's height, at constant speed: it bends, it
    // does not stretch.
    var walk = vec3<f32>(0.0);
    for (var k = 0; k < 6; k = k + 1) {
        let d = n + pull * (2.0 * h * (f32(k) + 0.5) / 6.0);
        walk = walk + d / max(length(d), 1e-6);
    }
    out.world_pos = out.world_pos + walk * (len * h / 6.0);
    out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);
    out.clip_zw = out.clip_pos.zw;
    out.view_z = -(frame.view * vec4<f32>(out.world_pos, 1.0)).z;
    // The strand's direction at this height.
    let along = n + pull * (2.0 * h);
    out.world_tangent = vec4<f32>(select(n, normalize(along), length(along) > 1e-6), 1.0);
    // (height, length share, one shell's slab of height, —)
    out.proj_uv = vec4<f32>(h, in.color.a, 1.0 / shells, 1.0);
    return out;
}
"#;

/// The fragment stage: which strand (if any) passes through this point of
/// this shell, then Kajiya-Kay lighting along it.
pub(crate) const FUR_WGSL: &str = r#"
// PCG2D/3D (Jarzynski & Olano 2020): exact integer arithmetic, so the scatter
// is the same on every GPU — a `fract(sin(x))` hash loses its low bits at the
// cell counts fur needs and draws the grid back in.
fn fur_hash2(p : vec2<i32>) -> vec2<f32> {
    var v = bitcast<vec2<u32>>(p) * 1664525u + 1013904223u;
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    return vec2<f32>(v >> vec2<u32>(8u)) * (1.0 / 16777216.0);
}

fn fur_hash3(p : vec3<i32>) -> vec3<f32> {
    var v = bitcast<vec3<u32>>(p) * 1664525u + 1013904223u;
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    return vec3<f32>(v >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

fn fur_seed() -> i32 {
    return i32(u_data.data[4].w);
}

// A grid cell wrapped into the UV square's `n` cells, with the seed mixed in:
// the same strand on both sides of a UV seam.
//
// Wrapped in floats, not integers: cells are whole numbers, so
// `c − n·floor(c / n)` is exact, and an integer `%` costs a GPU a division.
fn fur_cell_key(cell : vec2<f32>, n : vec2<f32>, salt : i32) -> vec2<i32> {
    let w = cell - n * floor(cell / n);
    return vec2<i32>(w) + vec2<i32>(fur_seed() * 7919 + salt, fur_seed() * 104729 - salt);
}

// A strand's own flutter: a small circle on its own phase, growing towards
// the tip, while the clock runs.
fn fur_flutter(r : f32, h : f32) -> vec2<f32> {
    return fur_flutter_noise(r).xy * (h * h);
}

// ---- The air, shared by the shells' and hair's vertex stages ----------------
//
// Everything here is a *pull*: per unit of strand length, what bends a strand
// from its root's normal — a strand's direction `u` of the way up is the
// normal plus `2u` times the pull, walked at constant speed so it never
// stretches. Gravity pulls; air pulls by drag, as the square of its speed
// past the strand (`drag`, `u_data.data[12].x`).

// Smooth value noise, a vector in [-1, 1]³: the hash on the integer lattice,
// blended by smoothstep.
// (Unrolled, with `mix`: a loop of integer bit tricks here broke the
// WebGL2 pipeline outright.)
fn fur_noise3(p : vec3<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let b = vec3<i32>(i);
    let c000 = fur_hash3(b);
    let c100 = fur_hash3(b + vec3<i32>(1, 0, 0));
    let c010 = fur_hash3(b + vec3<i32>(0, 1, 0));
    let c110 = fur_hash3(b + vec3<i32>(1, 1, 0));
    let c001 = fur_hash3(b + vec3<i32>(0, 0, 1));
    let c101 = fur_hash3(b + vec3<i32>(1, 0, 1));
    let c011 = fur_hash3(b + vec3<i32>(0, 1, 1));
    let c111 = fur_hash3(b + vec3<i32>(1, 1, 1));
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    let v = mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
    return v * 2.0 - 1.0;
}

// The pull on the coat as a whole: the animator's, when one drives it (its
// spring brings gravity, the drag of the air past the moving object and the
// object's acceleration together, with inertia — `data[9]`, `w` set); else
// gravity and the wind's mean drag.
fn fur_mean_pull() -> vec3<f32> {
    if (u_data.data[9].w > 0.5) {
        return u_data.data[9].xyz;
    }
    let w = u_data.data[6].xyz;
    return u_data.data[1].xyz + w * (u_data.data[12].x * length(w));
}

// Gusts: turbulence frozen into the air and carried downwind at the wind's
// own speed (Taylor's hypothesis), eddies `|wind| / wind_frequency` across so
// one passes a point that often; its velocity `turbulence` of the mean wind's,
// pulling as the drag's first-order change, `2 · drag · |wind| · u'`.
fn fur_turbulence(p : vec3<f32>) -> vec3<f32> {
    let w = u_data.data[6].xyz;
    let wl = length(w);
    if (wl <= 0.0 || u_data.data[12].y <= 0.0) {
        return vec3<f32>(0.0);
    }
    let eddy = max(wl / max(u_data.data[6].w, 1e-3), 0.05);
    let n = fur_noise3((p - w * u_data.data[5].z) / eddy);
    return n * (2.0 * u_data.data[12].x * u_data.data[12].y * wl * wl);
}

// A hand in the coat (`FurAnimator::stroke`): within `radius` of where it
// is, strands are pushed along its motion and parted away from it, fading
// smoothly to nothing at the edge.
fn fur_brush(p : vec3<f32>) -> vec3<f32> {
    let b = u_data.data[13];
    if (b.w <= 0.0) {
        return vec3<f32>(0.0);
    }
    let off = p - b.xyz;
    let d = length(off);
    if (d >= b.w) {
        return vec3<f32>(0.0);
    }
    let f = 1.0 - d / b.w;
    let w = f * f * (3.0 - 2.0 * f);
    return (u_data.data[14].xyz + off / max(d, 1e-4) * u_data.data[14].w) * w;
}

// A strand's own flutter in the flow — vortices shed off it — faster the
// faster the air, and none in still air: noise in time, on the strand's own
// random `r`.
fn fur_flutter_noise(r : f32) -> vec3<f32> {
    let wl = length(u_data.data[6].xyz);
    let amp = u_data.data[8].z * min(wl, 1.0);
    if (amp <= 0.0) {
        return vec3<f32>(0.0);
    }
    let rate = 2.0 + 4.0 * wl;
    return fur_noise3(vec3<f32>(r * 97.0, u_data.data[5].z * rate, r * 31.0)) * amp;
}

// Whether the pipeline takes a per-sample coverage — written as the sample
// mask, or as alpha for alpha-to-coverage. Without either (one sample, or a
// browser, where both cost a per-sample shading the fur cannot afford) a
// fragment tests the pixel's centre and covers all its samples or none.
const FUR_PER_SAMPLE : bool = THREERS_SAMPLES > 1u && (THREERS_SAMPLE_MASK || THREERS_COVERAGE);

// How far a strand can stray from its root at height `h`, in cells: its
// curl's circle, its flutter, its pull into a clump (capped at 0.95 cells)
// and its own radius. A root further than this from a point cannot cover it.
fn fur_reach(h : f32) -> f32 {
    let curl = 0.3 * h * min(abs(u_data.data[4].x), 1.0);
    let flutter = u_data.data[8].z * h * h * 1.4143;
    // The nearest clump point lies within 1.5 clump cells per axis of the
    // root (2.12 diagonally), the strand within its curl and flutter of that.
    let pull = min(0.95, u_data.data[4].y * h * (2.25 * u_data.data[5].w + 2.0));
    return curl + flutter + pull + u_data.data[0].w + 0.0001;
}

// Which of the pixel's samples about `uv` the strands rooted in the UV cells
// near it cover at height `h` — `jx`, `jy` the pixel's steps in cells. With
// one sample, its centre: whether a strand covers that.
//
// Each strand: a root jittered in its cell; a length of its own; a curl
// round the root growing with height; its flutter; and a pull towards the
// nearest clump point — jittered points on a coarser grid, Worley-style, so
// clumps have no grid lines between them — at most 0.95 cells, growing with
// height. Its radius tapers to the tip.
//
// The point's own cell is tried first, where most hits are, and the search
// stops once every sample is covered. A root too far away to reach
// it, or a strand too short, is passed over before its clump is looked for.
// (Caching the clump points neighbouring strands share measured slower: the
// array spills out of registers.)
// How far `o` (a strand centre's offset from the point) is from the strand
// swept along `d` — a capsule, the segment `[-d/2, d/2]` about the centre.
fn fur_capsule(o : vec2<f32>, d : vec2<f32>) -> f32 {
    let dd = dot(d, d);
    var t = 0.0;
    if (dd > 1e-12) {
        t = clamp(-dot(o, d) / dd, -0.5, 0.5);
    }
    return length(o + d * t);
}

// `slab`: how far the line of sight runs across one shell's slab of height,
// in cells — each strand drawn swept that far, so a strand seen at a slant
// is one column through the shells rather than a stack of dots.
fn fur_mask_uv(uv : vec2<f32>, jx : vec2<f32>, jy : vec2<f32>, h : f32, slab : vec2<f32>) -> u32 {
    let cells = u_data.data[7].xy;
    let clumps = u_data.data[7].zw;
    let q = uv * cells;
    let base = floor(q);
    let full = (1u << THREERS_SAMPLES) - 1u;
    // The samples sit up to 0.4 pixels from the centre: as far again, in
    // cells, a strand may be and still cover one.
    // How far, in cells, the samples sit from the centre at most.
    let spread = select(0.0, 0.4 * (length(jx) + length(jy)), FUR_PER_SAMPLE);
    let reach = fur_reach(h) + spread + 0.5 * length(slab);
    var mask = 0u;
    for (var k = 0; k < 9; k = k + 1) {
        let cell = base + vec2<f32>(f32((k + 4) % 3 - 1), f32((k + 4) / 3 % 3 - 1));
        let r = fur_hash2(fur_cell_key(cell, cells, 0));
        let root = cell + 0.15 + 0.7 * r;
        if (length(q - root) > reach) {
            continue;
        }
        let r2 = fur_hash2(fur_cell_key(cell, cells, 31337));
        let own_len = 1.0 - u_data.data[4].z * r2.x;
        if (h > own_len) {
            continue;
        }
        var c = root;
        // Curl: circle the root, the radius growing with height.
        let angle = (u_data.data[4].x * h + r2.y) * 6.2831853;
        c = c + vec2<f32>(cos(angle), sin(angle)) * (0.3 * h * min(abs(u_data.data[4].x), 1.0));
        c = c + fur_flutter(r.x, h);
        // Clumping: towards the nearest of the four clump points about it.
        let rc = root / cells * clumps;
        let cb = floor(rc - 0.5);
        var best = vec2<f32>(0.0);
        var best_d = 1e9;
        for (var j = 0; j <= 1; j = j + 1) {
            for (var i = 0; i <= 1; i = i + 1) {
                let kk = cb + vec2<f32>(f32(i), f32(j));
                let point = kk + fur_hash2(fur_cell_key(kk, clumps, 7));
                let d = dot(point - rc, point - rc);
                if (d < best_d) {
                    best_d = d;
                    best = point;
                }
            }
        }
        var pull = (best / clumps * cells - c) * (u_data.data[4].y * h);
        let pl = length(pull);
        if (pl > 0.95) {
            pull = pull * (0.95 / pl);
        }
        c = c + pull;
        let radius = u_data.data[0].w * mix(1.0, u_data.data[1].w, h / max(own_len, 1e-3));
        // One sample: its centre, straight — a constant the compiler folds,
        // so the single-sampled shader carries no loop (and stays small
        // enough to keep its occupancy).
        let d = fur_capsule(q - c, slab);
        if (!FUR_PER_SAMPLE) {
            if (d < radius) {
                mask = full;
            }
        } else if (d + spread < radius) {
            // Well inside the strand: every sample is.
            mask = full;
        } else if (d - spread < radius) {
            // Its edge crosses the pixel: sample by sample.
            for (var i = 0u; i < THREERS_SAMPLES; i = i + 1u) {
                let o = threers_sample_offset(i);
                if (fur_capsule(q + jx * o.x + jy * o.y - c, slab) < radius) {
                    mask = mask | (1u << i);
                }
            }
        }
        if (mask == full) {
            return mask;
        }
    }
    return mask;
}

// Surface placement: strands rooted at jittered points of a 3D grid of
// `cell`-sized cells, kept where the point lies within half a cell of the
// skin, measured in the skin's tangent plane at `p` (object space).
fn fur_mask_surface(p : vec3<f32>, n : vec3<f32>, h : f32, jx : vec3<f32>, jy : vec3<f32>, slab3 : vec3<f32>) -> u32 {
    let q = p / u_data.data[8].y;
    let full = (1u << THREERS_SAMPLES) - 1u;
    var mask = 0u;
    let base = floor(q);
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.8);
    let t1 = normalize(cross(n, helper));
    let t2 = cross(n, t1);
    let csize = u_data.data[5].w;
    // As for UV placement: a strand strays from its root by its curl, its
    // flutter and its clump pull — towards a clump centre at most √3 × 0.75
    // clump cells away, and never more than 0.95 cells — plus its radius, so
    // a root further across the skin than that is passed over.
    let curl_r = 0.3 * h * min(abs(u_data.data[4].x), 1.0);
    let pull_r = min(0.95, u_data.data[4].y * h * (1.3 * csize + 0.01));
    // The sweep across the slab, in the skin's tangent plane, as 2D.
    let slab = vec2<f32>(dot(slab3, t1), dot(slab3, t2));
    let reach = curl_r + u_data.data[8].z * h * h * 1.4143 + pull_r + u_data.data[0].w + 0.0001
        + 0.5 * length(slab)
        + select(0.0, 0.4 * (length(jx) + length(jy)) / u_data.data[8].y, FUR_PER_SAMPLE);
    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let cell = base + vec3<f32>(f32(x), f32(y), f32(z));
                let key = vec3<i32>(cell) + vec3<i32>(fur_seed() * 7919, fur_seed() * 104729, 31);
                let r = fur_hash3(key);
                let root = cell + r;
                let off = root - q;
                let dn = dot(off, n);
                if (abs(dn) > 0.5) {
                    continue;
                }
                let r2 = fur_hash3(key + vec3<i32>(977, 31337, 4099));
                let own_len = 1.0 - u_data.data[4].z * r2.x;
                if (h > own_len) {
                    continue;
                }
                // Where the strand is carried from its root at this height.
                var s = vec3<f32>(0.0);
                if (length(off - n * dn) > reach) {
                    continue;
                }
                let angle = (u_data.data[4].x * h + r2.y) * 6.2831853;
                s = s + (t1 * cos(angle) + t2 * sin(angle)) * (0.3 * h * min(abs(u_data.data[4].x), 1.0));
                let fl = fur_flutter(r2.z, h);
                s = s + t1 * fl.x + t2 * fl.y;
                // Clumping towards the jittered centre of the root's clump cell.
                let cp = (floor(root / csize) + 0.25 + 0.5 * fur_hash3(vec3<i32>(floor(root / csize)) + 17)) * csize;
                var pull = cp - root;
                pull = (pull - n * dot(pull, n)) * (u_data.data[4].y * h);
                let pl = length(pull);
                if (pl > 0.95) {
                    pull = pull * (0.95 / pl);
                }
                s = s + pull;
                let radius = u_data.data[0].w * mix(1.0, u_data.data[1].w, h / max(own_len, 1e-3));
                if (!FUR_PER_SAMPLE) {
                    let o2 = off - n * dot(off, n) + s;
                    if (fur_capsule(vec2<f32>(dot(o2, t1), dot(o2, t2)), slab) < radius) {
                        mask = full;
                    }
                } else {
                    for (var i = 0u; i < THREERS_SAMPLES; i = i + 1u) {
                        let o = threers_sample_offset(i);
                        let at = off - (jx * o.x + jy * o.y) / u_data.data[8].y;
                        let o2 = at - n * dot(at, n) + s;
                        if (fur_capsule(vec2<f32>(dot(o2, t1), dot(o2, t2)), slab) < radius) {
                            mask = mask | (1u << i);
                        }
                    }
                }
                if (mask == full) {
                    return mask;
                }
            }
        }
    }
    return mask;
}

// The highlight's exponent, eased where the strand's direction turns across
// the pixel (see `fs_main`), so it does not sparkle.
var<private> fur_spec_exp : f32;

fn fur_kajiya(t : vec3<f32>, n : vec3<f32>, v : vec3<f32>, l : vec3<f32>, albedo : vec3<f32>, h : f32) -> vec3<f32> {
    let ndl = dot(n, l);
    if (h <= 0.0) {
        return albedo * max(ndl, 0.0);
    }
    // Diffuse: strands catch light from every side, dimmer facing away from
    // it — the surface term wrapped rather than cut at the terminator.
    let tl = dot(t, l);
    let diffuse = sqrt(max(1.0 - tl * tl, 0.0)) * clamp(ndl * 0.75 + 0.25, 0.0, 1.0);
    let ts = normalize(t + n * u_data.data[5].y);
    let th = dot(ts, normalize(l + v));
    let spec = pow(sqrt(max(1.0 - th * th, 0.0)), fur_spec_exp) * u_data.data[3].w
        * clamp(ndl * 0.5 + 0.5, 0.0, 1.0) * h;
    return albedo * diffuse + vec3<f32>(spec);
}

// Where the line of sight meets the strand across one shell's slab of
// height, as a step along the skin in world space. Up one slab (`len /
// shells` along the normal) the ray moves `v_t / v·n` sideways and the
// strand — leaning as it is bent — `s_t / s·n`; what the slab hides between
// two shells' slices of a strand is the difference. (Grazing angles are
// capped: there the shells give out anyway.)
fn fur_slab_world(in : VsOut) -> vec3<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(frame.camera_position.xyz - in.world_pos);
    let st = normalize(in.world_tangent.xyz);
    let vn = dot(v, n);
    let sn = max(dot(st, n), 0.2);
    let ray = (v - n * vn) / (sign(vn) * max(abs(vn), 0.2));
    let strand = (st - n * dot(st, n)) / sn;
    let slab = u_data.data[0].y * in.proj_uv.y * in.proj_uv.z;
    return (ray - strand) * slab;
}

// A world step along the surface in the coordinates whose screen
// derivatives are `ax`, `ay`, given the world position's `wx`, `wy`: the
// step as a mix of the two screen steps (least squares), then mapped.
fn fur_world_to(step : vec3<f32>, wx : vec3<f32>, wy : vec3<f32>) -> vec2<f32> {
    let a = dot(wx, wx);
    let b = dot(wx, wy);
    let c = dot(wy, wy);
    let det = a * c - b * b;
    if (abs(det) < 1e-20) {
        return vec2<f32>(0.0);
    }
    let rx = dot(wx, step);
    let ry = dot(wy, step);
    return vec2<f32>(c * rx - b * ry, a * ry - b * rx) / det;
}

fn fur_slab_cells(in : VsOut, wx : vec3<f32>, wy : vec3<f32>, jx : vec2<f32>, jy : vec2<f32>) -> vec2<f32> {
    let m = fur_world_to(fur_slab_world(in), wx, wy);
    let d = jx * m.x + jy * m.y;
    // At most a few cells: past that the shells are seen edge-on.
    let l = length(d);
    return select(d, d * (3.0 / l), l > 3.0);
}

fn fur_slab_cells3(in : VsOut, wx : vec3<f32>, wy : vec3<f32>, px : vec3<f32>, py : vec3<f32>) -> vec3<f32> {
    let m = fur_world_to(fur_slab_world(in), wx, wy);
    let d = px * m.x + py * m.y;
    let l = length(d) / u_data.data[8].y;
    return select(d, d * (3.0 / l), l > 3.0);
}

@fragment
fn fs_main(in : VsOut) -> ThreersFragOut {
    let h = in.proj_uv.x;
    // Derivatives before any branch: they need every fragment of the quad.
    // The skin's normal in object space, for surface placement; the pixel's
    // steps in UV cells, in object space and across a ribbon, for where its
    // samples lie; how far the strand's direction turns across it.
    let px = dpdx(in.local_pos);
    let py = dpdy(in.local_pos);
    let fn_raw = cross(px, py);
    let jx = dpdx(in.uv * u_data.data[7].xy);
    let jy = dpdy(in.uv * u_data.data[7].xy);
    let wx = dpdx(in.world_pos);
    let wy = dpdy(in.world_pos);
    let sx = dpdx(in.proj_uv.z);
    let sy = dpdy(in.proj_uv.z);
    let dt = fwidth(in.world_tangent.xyz);
    fur_spec_exp = u_data.data[5].x / (1.0 + u_data.data[5].x * dot(dt, dt) * 0.25);
    // Which of the pixel's samples this fragment covers — under MSAA each
    // tested against the strands where it lies, so strand edges and strands
    // thinner than a pixel resolve as real geometry does; else the centre.
    // Shells are cut down to strands; hair (placement 2) to its true width
    // within the ribbon drawn at least 1.5 pixels wide.
    let full = (1u << THREERS_SAMPLES) - 1u;
    var mask = full;
    if (u_data.data[8].x > 1.5) {
        mask = 0u;
        if (!FUR_PER_SAMPLE) {
            if (abs(in.proj_uv.z) < in.proj_uv.w) {
                mask = full;
            }
        } else {
            for (var i = 0u; i < THREERS_SAMPLES; i = i + 1u) {
                let o = threers_sample_offset(i);
                if (abs(in.proj_uv.z + sx * o.x + sy * o.y) < in.proj_uv.w) {
                    mask = mask | (1u << i);
                }
            }
        }
    } else if (h > 0.0) {
        if (in.proj_uv.y <= 0.001) {
            discard;
        }
        if (u_data.data[8].x > 0.5) {
            let fl = length(fn_raw);
            mask = 0u;
            if (fl > 0.0) {
                let sl = fur_slab_cells3(in, wx, wy, px, py) / u_data.data[8].y;
                mask = fur_mask_surface(in.local_pos, fn_raw / fl, h, px, py, sl);
            }
        } else {
            mask = fur_mask_uv(in.uv, jx, jy, h, fur_slab_cells(in, wx, wy, jx, jy));
        }
    }
    if (mask == 0u) {
        discard;
    }
    // Where the samples cannot be written directly (WebGL2), their share as
    // the alpha, for alpha-to-coverage to place.
    var alpha = 1.0;
    if (!THREERS_SAMPLE_MASK && THREERS_SAMPLES > 1u) {
        let m = mask;
        alpha = f32((m & 1u) + ((m >> 1u) & 1u) + ((m >> 2u) & 1u) + ((m >> 3u) & 1u)) / f32(THREERS_SAMPLES);
    }
    var n = normalize(in.world_normal);
    let t = normalize(in.world_tangent.xyz);
    let v = normalize(frame.camera_position.xyz - in.world_pos);
    // Shadows where the object receives them (`receive_shadow`, which fur
    // and hair objects turn on); unshadowed light otherwise.
    if ((mesh.flags.z & FLAG_RECEIVE_SHADOW) != 0u) {
        compute_shadows(in.world_pos, n);
    } else {
        for (var r = 0u; r < 16u; r = r + 1u) {
            shadow_cache[r] = 1.0;
        }
    }
    let tint = mix(u_data.data[2].rgb, u_data.data[3].rgb, h) * in.vertex_color.rgb;
    let ao = mix(1.0 - u_data.data[2].w, 1.0, h);
    let albedo = tint * ao;

    let RECIP_PI = 0.3183098861837907;
    var col = albedo * frame.ambient.rgb * RECIP_PI;
    for (var k = 0u; k < frame.light_counts.w; k = k + 1u) {
        if (k >= 8u) { break; }
        col = col + albedo * shade_hemi(n, frame.hemi_lights[k]) * RECIP_PI;
    }
    for (var k = 0u; k < frame.light_counts.x; k = k + 1u) {
        if (k >= 8u) { break; }
        let l = -normalize(frame.dir_lights[k].direction.xyz);
        col = col + frame.dir_lights[k].color.rgb * shadow_factor_dir(k) * fur_kajiya(t, n, v, l, albedo, h);
    }
    for (var k = 0u; k < frame.light_counts.y; k = k + 1u) {
        if (k >= 8u) { break; }
        let pl = frame.point_lights[k];
        let to_light = pl.position.xyz - in.world_pos;
        let d = length(to_light);
        let att = punctual_attenuation(d, pl.params.x, pl.params.y) * shadow_factor_point(k);
        col = col + pl.color.rgb * att * fur_kajiya(t, n, v, to_light / max(d, 1e-4), albedo, h);
    }
    for (var k = 0u; k < frame.light_counts.z; k = k + 1u) {
        if (k >= 8u) { break; }
        let sl = frame.spot_lights[k];
        let to_light = sl.position.xyz - in.world_pos;
        let d = length(to_light);
        let l = to_light / max(d, 1e-4);
        let cos_angle = dot(-l, normalize(sl.direction.xyz));
        var cone = 0.0;
        if (cos_angle > sl.params.z) {
            cone = smoothstep(sl.params.z, sl.params.w, cos_angle);
        }
        let att = punctual_attenuation(d, sl.params.x, sl.params.y) * cone * shadow_factor_spot(k);
        col = col + sl.color.rgb * att * fur_kajiya(t, n, v, l, albedo, h);
    }
    return frag_out(vec4<f32>(framebuffer_encode(apply_tone_mapping(apply_fog(col, in.view_z))), alpha), mask);
}
"#;
