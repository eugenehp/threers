//! Strand hair: curves grown from a mesh's surface, drawn as camera-facing
//! ribbons — for manes, long hair and close-ups, where shells run out.
//!
//! [`grow_strands`] roots `count` strands over the surface (spread evenly by
//! area, from a seed) and grows each from the same [`FurGroom`] as shell fur:
//! out along the normal, curled around itself, shortened at random, and drawn
//! into clumps around guide strands. Gravity, wind and inertia bend them in
//! the shader, in world space, exactly as they bend shell fur — so a
//! [`FurAnimator`](super::FurAnimator) animates hair too, and
//! [`grow_hair_on_skinned`](super::FurAnimator::grow_hair_on_skinned) carries
//! it through a skinned mesh's animation.
//!
//! [`hair_object`] draws them: each strand a ribbon of [`StrandOptions::width`]
//! that turns to face the camera, tapered by [`FurGroom::taper`] and shaded
//! like fur (Kajiya-Kay along the strand, root to tip colours, the
//! highlight). Vertex colours on the source mesh tint the strands and scale
//! their length, as for fur.
//!
//! For a renderer that needs real geometry — the path tracer, an exporter —
//! [`HairStrands::tube_geometry`] turns the same curves into triangle tubes.
//!
//! ```
//! use threers::fur::strands::{hair_object, StrandOptions};
//! use threers::fur::{FurGroom, FurMaterial};
//! use threers::prelude::*;
//!
//! let mut scene = Scene::new();
//! let hair = FurMaterial {
//!     groom: FurGroom { length: 0.8, curl: 0.5, clumping: 0.6, ..Default::default() },
//!     ..Default::default()
//! };
//! let options = StrandOptions { count: 2000, ..Default::default() };
//! scene.add(hair_object(&SphereGeometry::new(1.0, 32, 16), &hair, &options));
//! ```
//!
//! Hair runs where fur does: WebGPU, WebGL2, native wgpu and Metal.
//!
//! # Shadows
//!
//! A ribbon has width only towards the camera, so it cannot cast from a
//! light. Hair carries a second, invisible part for that: thin three-sided
//! tubes around a share of its strands ([`StrandOptions::shadow_strands`],
//! each thicker by as much as the strands it stands for), drawn into the
//! shadow maps only. They are bent on the CPU exactly as the shader bends the
//! ribbons, so the shadow lies where the hair does — on the body, the ground,
//! and the hair itself, which receives shadows and so darkens inside.
//!
//! The tubes are bent when the hair is made, for an object with no
//! transform, and again by [`FurAnimator::update`](super::FurAnimator::update)
//! for its real place, clock and inertia. Hair that is turned, or has wind,
//! wants an animator for its shadow to follow; nothing else needs one.
//! Hair casts and receives by default (`cast_shadow`, `receive_shadow`);
//! `shadow_strands: 0.0` leaves the tubes out.

use std::sync::Arc;

use super::{FurGroom, FurMaterial};
use crate::core::{BufferAttribute, BufferGeometry, Mesh, Object3D};
use crate::materials::{Material, ShaderMaterial};
use crate::math::Matrix4;
use crate::math::Vector3;

/// How many strands, and how they are drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrandOptions {
    pub count: usize,
    /// Straight pieces per strand: more bends more smoothly.
    pub segments: usize,
    /// Ribbon width at the root, world units.
    pub width: f32,
    /// The share of strands that cast shadows, 0 to 1: each one stands for
    /// `1 / shadow_strands` of them, made that much thicker, so the shadow is
    /// as dense as the hair. 0 casts none. (See [the shadows](self#shadows).)
    pub shadow_strands: f32,
}

impl Default for StrandOptions {
    fn default() -> Self {
        Self {
            count: 4000,
            segments: 8,
            width: 0.006,
            shadow_strands: 0.25,
        }
    }
}

/// Strands grown on a surface, in the surface's object space.
#[derive(Debug, Clone, PartialEq)]
pub struct HairStrands {
    /// Straight pieces per strand: each strand is `segments + 1` points.
    pub segments: usize,
    /// Every strand's points, root first, strand after strand.
    pub points: Vec<Vector3>,
    /// Each strand's tint (the mesh's vertex colour at its root).
    pub tints: Vec<[f32; 3]>,
    /// Each strand's own random number in 0..1 (its flutter's phase).
    pub randoms: Vec<f32>,
    /// Each strand's surface normal at its root: the way it grew, which the
    /// shader bends from, and what shades it as a volume.
    pub normals: Vec<Vector3>,
    /// Each strand's length, as grown (after variation and paint).
    pub lengths: Vec<f32>,
}

/// A small deterministic generator (xorshift*), so a seed regrows the same
/// hair on every platform.
struct Rng(u64);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(0x9e37_79b9_7f4a_7c15 ^ (seed as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9) | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// A unit vector perpendicular to `n`.
fn perpendicular(n: Vector3) -> Vector3 {
    let helper = if n.x.abs() > 0.8 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    n.cross(helper).normalize()
}

/// Grow `count` strands of `segments` pieces from `geometry` by `groom`.
/// Roots are spread evenly over the area. A vertex colour's alpha scales a
/// strand's length (0: no strand there), its RGB tints it. Gravity is not
/// baked in: it is applied when drawn, in world space.
pub fn grow_strands(
    geometry: &BufferGeometry,
    groom: &FurGroom,
    count: usize,
    segments: usize,
) -> HairStrands {
    grow_rooted(geometry, groom, count, segments).0
}

/// Where a strand is rooted: a triangle of the source mesh and the
/// barycentric weights in it — what carries the strand when the mesh deforms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StrandRoot {
    tri: [usize; 3],
    bary: [f32; 3],
}

/// The triangles of `geometry`, as vertex indices.
fn triangles(geometry: &BufferGeometry, vertices: usize) -> Vec<[usize; 3]> {
    match &geometry.index {
        Some(idx) => idx
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect(),
        None => (0..vertices / 3)
            .map(|k| [3 * k, 3 * k + 1, 3 * k + 2])
            .collect(),
    }
}

/// The point at `root` on a surface with these positions and (optional)
/// normals, and a frame there: across the surface (`t1`, `t2`) and its
/// normal — the vertex normals blended, else the face's.
fn surface_frame(
    pos: &[f32],
    normals: Option<&[f32]>,
    root: &StrandRoot,
) -> (Vector3, [Vector3; 3]) {
    let v = |a: &[f32], i: usize| Vector3::new(a[i * 3], a[i * 3 + 1], a[i * 3 + 2]);
    let [a, b, c] = root.tri;
    let [wa, wb, wc] = root.bary;
    let (pa, pb, pc) = (v(pos, a), v(pos, b), v(pos, c));
    let at = pa * wa + pb * wb + pc * wc;
    let face = (pb - pa).cross(pc - pa);
    let face = if face.length() > 1e-12 {
        face.normalize()
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let n = normals
        .map(|nn| v(nn, a) * wa + v(nn, b) * wb + v(nn, c) * wc)
        .filter(|m| m.length() > 1e-6)
        .map_or(face, |m| m.normalize());
    // Along the triangle's first edge, so the frame turns with the surface.
    let e = pb - pa;
    let t1 = e - n * e.dot(n);
    let t1 = if t1.length() > 1e-9 {
        t1.normalize()
    } else {
        perpendicular(n)
    };
    (at, [t1, n.cross(t1), n])
}

/// [`grow_strands`], and where each strand is rooted.
pub(crate) fn grow_rooted(
    geometry: &BufferGeometry,
    groom: &FurGroom,
    count: usize,
    segments: usize,
) -> (HairStrands, Vec<StrandRoot>) {
    let segments = segments.max(1);
    let mut rooted = Vec::new();
    let mut out = HairStrands {
        segments,
        points: Vec::new(),
        tints: Vec::new(),
        randoms: Vec::new(),
        normals: Vec::new(),
        lengths: Vec::new(),
    };
    let Some(pos) = geometry.get_attribute("position") else {
        return (out, rooted);
    };
    let normals = geometry.get_attribute("normal");
    let colors = geometry.get_attribute("color");
    let p = |i: usize| Vector3::new(pos.array[i * 3], pos.array[i * 3 + 1], pos.array[i * 3 + 2]);
    let tris = triangles(geometry, pos.count());
    // Cumulative area, for roots spread evenly whatever the tessellation.
    let mut cumulative = Vec::with_capacity(tris.len());
    let mut total = 0.0f64;
    for &[a, b, c] in &tris {
        total += (p(b) - p(a)).cross(p(c) - p(a)).length() as f64 * 0.5;
        cumulative.push(total);
    }
    if total <= 0.0 {
        return (out, rooted);
    }
    struct Root {
        at: Vector3,
        n: Vector3,
        tint: [f32; 3],
        length: f32,
        place: StrandRoot,
    }
    let mut rng = Rng::new(groom.seed);
    let mut roots = Vec::with_capacity(count);
    for _ in 0..count {
        let target = rng.next() as f64 * total;
        let t = cumulative
            .partition_point(|&c| c < target)
            .min(tris.len() - 1);
        let [a, b, c] = tris[t];
        // Uniform in the triangle.
        let (r1, r2) = (rng.next().sqrt(), rng.next());
        let (wa, wb, wc) = (1.0 - r1, r1 * (1.0 - r2), r1 * r2);
        let place = StrandRoot {
            tri: [a, b, c],
            bary: [wa, wb, wc],
        };
        let (at, [_, _, n]) =
            surface_frame(&pos.array, normals.map(|nn| nn.array.as_slice()), &place);
        let (tint, scale) = match colors {
            Some(cc) if cc.item_size >= 3 => {
                let k = cc.item_size;
                let ch = |i: usize, j: usize| cc.array.get(i * k + j).copied().unwrap_or(1.0);
                let mix = |j: usize| ch(a, j) * wa + ch(b, j) * wb + ch(c, j) * wc;
                ([mix(0), mix(1), mix(2)], if k >= 4 { mix(3) } else { 1.0 })
            }
            _ => ([1.0; 3], 1.0),
        };
        let own = 1.0 - groom.length_variation.clamp(0.0, 1.0) * rng.next();
        roots.push(Root {
            at,
            n,
            tint,
            length: groom.length * scale * own,
            place,
        });
    }

    // The unclumped shape of each strand: out along its normal, curling
    // round it on its own phase.
    let shape = |r: &Root, phase: f32| -> Vec<Vector3> {
        let t1 = perpendicular(r.n);
        let t2 = r.n.cross(t1);
        let radius = 0.12 * r.length * groom.curl.abs().min(1.0);
        (0..=segments)
            .map(|k| {
                let s = k as f32 / segments as f32;
                let angle = (groom.curl * s + phase) * std::f32::consts::TAU;
                r.at + r.n * (r.length * s) + (t1 * angle.cos() + t2 * angle.sin()) * (radius * s)
            })
            .collect()
    };
    // Clumps: the first strands are guides, and every strand's tip is drawn
    // towards its nearest guide's.
    let guides =
        ((count as f32 / groom.clump_size.max(1.0).powi(2)).ceil() as usize).clamp(1, count.max(1));
    let phases: Vec<f32> = (0..roots.len()).map(|_| rng.next()).collect();
    let guide_shapes: Vec<Vec<Vector3>> = roots
        .iter()
        .take(guides)
        .zip(&phases)
        .map(|(r, &ph)| shape(r, ph))
        .collect();
    for (i, r) in roots.iter().enumerate() {
        if r.length <= 1e-6 {
            continue;
        }
        let mut pts = shape(r, phases[i]);
        if groom.clumping > 0.0 && i >= guides {
            let g = (0..guides)
                .min_by(|&x, &y| {
                    let dx = (roots[x].at - r.at).length_sq();
                    let dy = (roots[y].at - r.at).length_sq();
                    dx.total_cmp(&dy)
                })
                .unwrap_or(0);
            for (k, q) in pts.iter_mut().enumerate() {
                let s = k as f32 / segments as f32;
                // Towards the guide's point at the same height, relative to
                // the guide's own root: the clump gathers, it does not move.
                let target = guide_shapes[g][k] - roots[g].at + r.at;
                let pull = (target - *q) * (groom.clumping.clamp(0.0, 1.0) * s);
                // And a little towards the guide itself, but never far: a
                // guide across a gap must not drag a tip out into the air.
                let mut toward = guide_shapes[g][k] - *q;
                let reach = 0.5 * r.length;
                if toward.length() > reach {
                    toward = toward.normalize() * reach;
                }
                *q = *q + pull + toward * (groom.clumping.clamp(0.0, 1.0) * s * s * 0.5);
            }
        }
        out.points.extend(pts);
        out.tints.push(r.tint);
        out.randoms.push(rng.next());
        out.normals.push(r.n);
        out.lengths.push(r.length);
        rooted.push(r.place);
    }
    (out, rooted)
}

/// A strand's direction at point `k`: towards the next point (from the
/// previous one at the tip).
fn direction_at(pts: &[Vector3], k: usize) -> Vector3 {
    let dir = if k + 1 < pts.len() {
        pts[k + 1] - pts[k]
    } else {
        pts[k] - pts[k - 1]
    };
    if dir.length() > 1e-9 {
        dir.normalize()
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    }
}

impl HairStrands {
    /// How many strands.
    pub fn count(&self) -> usize {
        self.tints.len()
    }

    /// Strand `i`'s points, root first.
    pub fn strand(&self, i: usize) -> &[Vector3] {
        let n = self.segments + 1;
        &self.points[i * n..(i + 1) * n]
    }

    /// The ribbons [`hair_object`] draws: two vertices at every point (`uv.x`
    /// 0 and 1 — the shader moves them apart, facing the camera), the
    /// strand's direction in `normal`, `uv.y` the height along it, the tint
    /// in `color.rgb` and the strand's random in `color.a`, and the root's
    /// normal and the strand's length in `tangent`.
    pub fn ribbon_geometry(&self) -> BufferGeometry {
        let n = self.segments + 1;
        let verts = self.count() * n * 2;
        let (mut pos, mut nrm, mut uv, mut col, mut tan) = (
            Vec::with_capacity(verts * 3),
            Vec::with_capacity(verts * 3),
            Vec::with_capacity(verts * 2),
            Vec::with_capacity(verts * 4),
            Vec::with_capacity(verts * 4),
        );
        let mut index = Vec::with_capacity(self.count() * self.segments * 6);
        for i in 0..self.count() {
            let pts = self.strand(i);
            let base = (i * n * 2) as u32;
            for k in 0..n {
                let d = direction_at(pts, k);
                let s = k as f32 / self.segments as f32;
                for side in 0..2 {
                    pos.extend_from_slice(&[pts[k].x, pts[k].y, pts[k].z]);
                    nrm.extend_from_slice(&[d.x, d.y, d.z]);
                    uv.extend_from_slice(&[side as f32, s]);
                    let t = self.tints[i];
                    col.extend_from_slice(&[t[0], t[1], t[2], self.randoms[i]]);
                    let r = self.normals[i];
                    tan.extend_from_slice(&[r.x, r.y, r.z, self.lengths[i]]);
                }
            }
            for k in 0..self.segments as u32 {
                let (a, b, c, d) = (
                    base + 2 * k,
                    base + 2 * k + 1,
                    base + 2 * k + 2,
                    base + 2 * k + 3,
                );
                index.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(pos, 3));
        g.set_attribute("normal", BufferAttribute::new(nrm, 3));
        g.set_attribute("uv", BufferAttribute::new(uv, 2));
        g.set_attribute("color", BufferAttribute::new(col, 4));
        g.set_attribute("tangent", BufferAttribute::new(tan, 4));
        g.set_index(index);
        g
    }

    /// The ribbon attributes that move when the strands do — `position`,
    /// `normal` (direction) and `tangent` (root normal, length) — as
    /// [`ribbon_geometry`](Self::ribbon_geometry) lays them out.
    fn ribbon_motion(&self) -> [Vec<f32>; 3] {
        let n = self.segments + 1;
        let verts = self.count() * n * 2;
        let (mut pos, mut nrm, mut tan) = (
            Vec::with_capacity(verts * 3),
            Vec::with_capacity(verts * 3),
            Vec::with_capacity(verts * 4),
        );
        for i in 0..self.count() {
            let pts = self.strand(i);
            let r = self.normals[i];
            for k in 0..n {
                let d = direction_at(pts, k);
                for _ in 0..2 {
                    pos.extend_from_slice(&[pts[k].x, pts[k].y, pts[k].z]);
                    nrm.extend_from_slice(&[d.x, d.y, d.z]);
                    tan.extend_from_slice(&[r.x, r.y, r.z, self.lengths[i]]);
                }
            }
        }
        [pos, nrm, tan]
    }

    /// Every strand as a tube of `sides` sides, `radius` at the root and
    /// `radius × taper` at the tip — real triangles, for the path tracer or
    /// an exporter. Open at both ends. Gravity is not applied: these are the
    /// grown shapes (bend them yourself, or pose with [`Self::points`]).
    pub fn tube_geometry(&self, sides: usize, radius: f32, taper: f32) -> BufferGeometry {
        let sides = sides.max(3);
        let n = self.segments + 1;
        let (mut pos, mut nrm, mut index) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..self.count() {
            let pts = self.strand(i);
            let base = (pos.len() / 3) as u32;
            let mut frame_t1: Option<Vector3> = None;
            for k in 0..n {
                let dir = if k + 1 < n {
                    pts[k + 1] - pts[k]
                } else {
                    pts[k] - pts[k - 1]
                };
                let d = if dir.length() > 1e-9 {
                    dir.normalize()
                } else {
                    Vector3::new(0.0, 1.0, 0.0)
                };
                // A frame carried along the strand, so the tube does not twist.
                let t1 = match frame_t1 {
                    Some(prev) => {
                        let p = prev - d * prev.dot(d);
                        if p.length() > 1e-6 {
                            p.normalize()
                        } else {
                            perpendicular(d)
                        }
                    }
                    None => perpendicular(d),
                };
                frame_t1 = Some(t1);
                let t2 = d.cross(t1);
                let s = k as f32 / self.segments as f32;
                let r = radius * (1.0 + (taper - 1.0) * s);
                for j in 0..sides {
                    let a = j as f32 / sides as f32 * std::f32::consts::TAU;
                    let out = t1 * a.cos() + t2 * a.sin();
                    let q = pts[k] + out * r;
                    pos.extend_from_slice(&[q.x, q.y, q.z]);
                    nrm.extend_from_slice(&[out.x, out.y, out.z]);
                }
            }
            for k in 0..self.segments as u32 {
                for j in 0..sides as u32 {
                    let j1 = (j + 1) % sides as u32;
                    let (a, b) = (base + k * sides as u32 + j, base + k * sides as u32 + j1);
                    let (c, d) = (a + sides as u32, b + sides as u32);
                    index.extend_from_slice(&[a, c, b, b, c, d]);
                }
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(pos, 3));
        g.set_attribute("normal", BufferAttribute::new(nrm, 3));
        g.set_index(index);
        g
    }
}

impl FurMaterial {
    /// The shader material [`hair_object`] draws with: fur's shading and
    /// motion, ribbons `width` wide at the root instead of shells.
    pub fn hair_shader_material(&self, width: f32) -> ShaderMaterial {
        let mut data = self.data(1.0);
        data[0][3] = width.max(0.0);
        data[super::HAIR.0][super::HAIR.1] = super::HAIR_PLACEMENT;
        ShaderMaterial {
            fragment: super::FUR_WGSL.to_string(),
            vertex: Some(HAIR_VERTEX_WGSL.to_string()),
            data,
            transparent: false,
            side: 2,
            alpha_to_coverage: !cfg!(target_arch = "wasm32"),
            fallback_color: self.root_color.lerp(self.tip_color, 0.6),
            ..Default::default()
        }
    }
}

/// Hair grown on `geometry` by `material`: an object to add to a scene.
pub fn hair_object(
    geometry: &BufferGeometry,
    material: &FurMaterial,
    options: &StrandOptions,
) -> Object3D {
    let strands = grow_strands(geometry, &material.groom, options.count, options.segments);
    hair_object_from(&strands, material, options)
}

/// [`hair_object`] for strands already grown (or posed by hand); `options`
/// gives the width and the shadow, its `count` and `segments` are the
/// strands' own.
pub fn hair_object_from(
    strands: &HairStrands,
    material: &FurMaterial,
    options: &StrandOptions,
) -> Object3D {
    let mut shader = material.hair_shader_material(options.width);
    let mut geometry = strands.ribbon_geometry();
    let stride = shadow_stride(options.shadow_strands);
    if stride == 0 || strands.count() == 0 {
        let mut object = Object3D::mesh(Mesh::from_arc(
            Arc::new(geometry),
            Arc::new(Material::Shader(shader)),
        ));
        object.receive_shadow = true;
        return object;
    }
    // A 3-sided tube is about 1.6 radii across from any side: as wide as
    // the `stride` ribbons it stands for.
    let radius = options.width * stride as f32 / 1.6;
    if shader.data.len() <= SHADOW {
        shader.data.resize(SHADOW + 1, [0.0; 4]);
    }
    shader.data[SHADOW] = [
        strands.segments as f32,
        strands.count() as f32,
        stride as f32,
        radius,
    ];
    refresh_hair_shadow(&mut geometry, &shader.data, &Matrix4::identity());
    let hair = Arc::new(Material::Shader(shader));
    // Drawn into the shadow maps only: no colour, no depth on screen.
    let caster = super::invisible_caster();
    let mut mesh = Mesh::from_arc(Arc::new(geometry), hair.clone());
    mesh.materials = vec![hair, Arc::new(Material::Basic(caster))];
    let mut object = Object3D::mesh(mesh);
    object.cast_shadow = true;
    object.receive_shadow = true;
    object
}

// ---------------------------------------------------------------------------
// Shadows
// ---------------------------------------------------------------------------

/// The material data slot holding the shadow caster's layout: segments,
/// strands, stride (every `stride`-th strand casts) and tube radius.
pub(crate) const SHADOW: usize = 10;

/// Every how many strands one casts, for a share `fraction`; 0 for none.
fn shadow_stride(fraction: f32) -> usize {
    if fraction.is_nan() || fraction <= 0.0 {
        0
    } else {
        (1.0 / fraction.min(1.0)).round().max(1.0) as usize
    }
}

/// A strand's point at height `s`, bent as `HAIR_VERTEX_WGSL` bends it:
/// `wp` the grown point in world space, `root_n` the root's normal and `len`
/// the strand's length (world), `r` its random, `lag` its own inertia (world,
/// per unit length).
#[allow(clippy::too_many_arguments)]
fn bend(
    data: &[[f32; 4]],
    wp: Vector3,
    root_n: Vector3,
    len: f32,
    s: f32,
    r: f32,
    lag: Vector3,
) -> Vector3 {
    let root = wp - root_n * (len * s);
    let pull = super::mean_pull(data)
        + super::turbulence(data, root)
        + super::flutter_noise(data, r) * 0.3
        + super::brush(data, wp)
        + lag;
    root + super::walk(root_n, pull, s) * len
}

/// Rebuild the shadow caster of a hair geometry: tubes around every
/// `stride`-th strand, bent by the material's `data` (gravity, wind, time,
/// flutter, inertia) for an object at `model` — what its ribbons draw. The
/// ribbons come first in the geometry, the tubes after; attributes the
/// ribbons alone set are padded to the whole.
pub(crate) fn refresh_hair_shadow(
    geometry: &mut BufferGeometry,
    data: &[[f32; 4]],
    model: &Matrix4,
) {
    if let Some(caster) = hair_caster(geometry, data, model) {
        super::append_caster(geometry, &caster);
    }
}

/// The tubes of [`refresh_hair_shadow`], on their own: bent from the
/// ribbons at the front of `geometry`, in its local space.
pub(crate) fn hair_caster(
    geometry: &BufferGeometry,
    data: &[[f32; 4]],
    model: &Matrix4,
) -> Option<super::Caster> {
    let &[segments, strands, stride, radius] = data.get(SHADOW)?;
    let (segments, strands, stride) = (segments as usize, strands as usize, stride as usize);
    if stride == 0 || strands == 0 || segments == 0 {
        return None;
    }
    let n = segments + 1;
    let ribbon = strands * n * 2;
    let attr = |name: &str| geometry.get_attribute(name).map(|a| &a.array[..]);
    let (pos, tan, col) = (attr("position")?, attr("tangent")?, attr("color")?);
    if pos.len() < ribbon * 3 || tan.len() < ribbon * 4 || col.len() < ribbon * 4 {
        return None;
    }
    let per_strand_lag = data[8][3] > 0.5;
    let (uv1, uv2) = (attr("uv1"), attr("uv2"));
    let to_local = model.invert();
    // A shadow needs no more than six bends a strand: every `step`-th point,
    // and the tip.
    let step = segments.div_ceil(6);
    let picks: Vec<usize> = (0..segments)
        .step_by(step)
        .chain(std::iter::once(segments))
        .collect();
    let mut bent = HairStrands {
        segments: picks.len() - 1,
        points: Vec::with_capacity(strands.div_ceil(stride) * picks.len()),
        tints: Vec::new(),
        randoms: Vec::new(),
        normals: Vec::new(),
        lengths: Vec::new(),
    };
    for i in (0..strands).step_by(stride) {
        let v0 = i * n * 2;
        let t = &tan[v0 * 4..v0 * 4 + 4];
        let rn = super::direction(model, Vector3::new(t[0], t[1], t[2]));
        let scale = rn.length().max(1e-6);
        let root_n = rn * (1.0 / scale);
        let len = t[3] * scale;
        let r = col[v0 * 4 + 3];
        let lag = match (per_strand_lag, uv1, uv2) {
            (true, Some(a), Some(b)) if a.len() > v0 * 2 + 1 && b.len() > v0 * 2 => {
                super::direction(model, Vector3::new(a[v0 * 2], a[v0 * 2 + 1], b[v0 * 2]))
            }
            _ => Vector3::ZERO,
        };
        for &k in &picks {
            let v = v0 + 2 * k;
            let p = Vector3::new(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]);
            let s = k as f32 / segments as f32;
            bent.points
                .push(bend(data, p.apply_matrix4(model), root_n, len, s, r, lag));
        }
        bent.tints.push([1.0; 3]);
        bent.randoms.push(r);
        bent.normals.push(root_n);
        bent.lengths.push(len);
    }
    let tubes = bent.tube_geometry(3, radius, data[1][3]);
    let (tp, tnm) = (
        tubes.get_attribute("position")?,
        tubes.get_attribute("normal")?,
    );
    let mut position = Vec::with_capacity(tp.array.len());
    for q in tp.array.chunks_exact(3) {
        let l = Vector3::new(q[0], q[1], q[2]).apply_matrix4(&to_local);
        position.extend_from_slice(&[l.x, l.y, l.z]);
    }
    let mut normal = Vec::with_capacity(tnm.array.len());
    for q in tnm.array.chunks_exact(3) {
        let l = super::direction(&to_local, Vector3::new(q[0], q[1], q[2]));
        let l = l * (1.0 / l.length().max(1e-9));
        normal.extend_from_slice(&[l.x, l.y, l.z]);
    }
    Some(super::Caster {
        after_vertices: ribbon,
        after_indices: strands * segments * 6,
        position,
        normal,
        index: tubes.index.clone().unwrap_or_default(),
    })
}

/// Hair grown on a mesh that deforms: every strand held in its root's frame
/// on the bind pose, so a posed mesh carries it — moved with its triangle,
/// turned as the surface there turns.
pub(crate) struct HairRig {
    grown: HairStrands,
    roots: Vec<StrandRoot>,
    /// Every point, in its root's bind-pose frame (`t1`, `t2`, `n`).
    local: Vec<Vector3>,
}

impl HairRig {
    /// Strands grown on `bind` (and rooted by `roots`), rigged to follow it.
    pub(crate) fn new(bind: &BufferGeometry, grown: HairStrands, roots: Vec<StrandRoot>) -> Self {
        let pos = bind
            .get_attribute("position")
            .map_or(&[][..], |a| &a.array[..]);
        let normals = bind.get_attribute("normal").map(|a| a.array.as_slice());
        let mut local = Vec::with_capacity(grown.points.len());
        for (i, root) in roots.iter().enumerate() {
            let (at, [t1, t2, n]) = surface_frame(pos, normals, root);
            local.extend(grown.strand(i).iter().map(|&p| {
                let d = p - at;
                Vector3::new(d.dot(t1), d.dot(t2), d.dot(n))
            }));
        }
        Self {
            grown,
            roots,
            local,
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.roots.len()
    }

    /// The strands on `posed` (the mesh posed, same vertices as the bind
    /// pose), and each root's position there.
    pub(crate) fn pose(&self, posed: &BufferGeometry) -> (HairStrands, Vec<Vector3>) {
        let mut out = self.grown.clone();
        let mut roots = Vec::with_capacity(self.count());
        let (Some(pos), normals) = (
            posed.get_attribute("position"),
            posed.get_attribute("normal").map(|a| a.array.as_slice()),
        ) else {
            return (out, roots);
        };
        let n = self.grown.segments + 1;
        for (i, root) in self.roots.iter().enumerate() {
            let (at, [t1, t2, nn]) = surface_frame(&pos.array, normals, root);
            for k in 0..n {
                let l = self.local[i * n + k];
                out.points[i * n + k] = at + t1 * l.x + t2 * l.y + nn * l.z;
            }
            out.normals[i] = nn;
            roots.push(at);
        }
        (out, roots)
    }

    /// Write `posed` strands, and each strand's lag (`lags`, local units per
    /// unit length), into the hair's ribbon geometry: the lag into `uv1`
    /// (x, y) and `uv2` (z), where the hair shader reads it.
    pub(crate) fn write(posed: &HairStrands, lags: &[Vector3], geometry: &mut BufferGeometry) {
        let [pos, nrm, tan] = posed.ribbon_motion();
        let per = (posed.segments + 1) * 2;
        let (mut uv1, mut uv2) = (
            Vec::with_capacity(lags.len() * per * 2),
            Vec::with_capacity(lags.len() * per * 2),
        );
        for l in lags {
            for _ in 0..per {
                uv1.extend_from_slice(&[l.x, l.y]);
                uv2.extend_from_slice(&[l.z, 0.0]);
            }
        }
        geometry.set_attribute("position", BufferAttribute::new(pos, 3));
        geometry.set_attribute("normal", BufferAttribute::new(nrm, 3));
        geometry.set_attribute("tangent", BufferAttribute::new(tan, 4));
        geometry.set_attribute("uv1", BufferAttribute::new(uv1, 2));
        geometry.set_attribute("uv2", BufferAttribute::new(uv2, 2));
    }
}

/// The ribbon vertex stage. Each point is the grown shape's, with its
/// straight part — `length × s` along the root normal — swapped for the same
/// length bent by gravity, wind, flutter and inertia: the direction at `u` is
/// the root normal plus `2u` times the pull, as shell fur's droop is, but
/// walked at constant speed so a strand bends without stretching. Then the
/// point moves half the ribbon's width across the strand and the line of
/// sight, so the ribbon faces the camera. Lit with the root's normal, so a
/// head of hair shades as a volume.
pub(crate) const HAIR_VERTEX_WGSL: &str = r#"
@vertex
fn vs_custom(in : VsIn, @builtin(vertex_index) vid : u32) -> VsOut {
    var out = standard_vertex(in);
    let s = in.uv.y;
    let side = in.uv.x * 2.0 - 1.0;
    let rn4 = (mesh.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz;
    let scale = max(length(rn4), 1e-6);
    let root_n = rn4 / scale;
    let len = in.tangent.w * scale;
    // What bends it: the coat's pull (gravity, drag, inertia), the gusts at
    // its root, its own flutter in the flow, and — on a skinned mesh — its
    // root's own pull, from uv1 and uv2.
    let r = in.color.a;
    let across0 = select(normalize(cross(root_n, vec3<f32>(0.0, 1.0, 0.0))),
        normalize(cross(root_n, vec3<f32>(1.0, 0.0, 0.0))), abs(root_n.y) > 0.9);
    let root = out.world_pos - root_n * (len * s);
    var pull = fur_mean_pull() + fur_turbulence(root) + fur_flutter_noise(r) * 0.3
        + fur_brush(out.world_pos);
    if (u_data.data[8].w > 0.5) {
        pull = pull + (mesh.model * vec4<f32>(out.uv12.xyz, 0.0)).xyz;
    }
    var walk = vec3<f32>(0.0);
    for (var k = 0; k < 6; k = k + 1) {
        // Where the pull exactly cancels the root's direction the strand
        // has turned back on itself: no direction, so no step.
        let d = root_n + pull * (2.0 * s * (f32(k) + 0.5) / 6.0);
        walk = walk + d / max(length(d), 1e-6);
    }
    let p = out.world_pos - root_n * (len * s) + walk * (len * s / 6.0);
    let b = root_n + pull * (2.0 * s);
    let bent = select(normalize(pull + root_n * 1e-3), b / max(length(b), 1e-6), length(b) > 1e-4);
    // The grown direction (curl, clumps) turned as the straight one was.
    let grown = normalize(out.world_normal);
    let tt_dir = grown - root_n + bent;
    let t_dir = select(bent, tt_dir / max(length(tt_dir), 1e-6), length(tt_dir) > 1e-4);
    let to_eye = normalize(frame.camera_position.xyz - p);
    var across = cross(t_dir, to_eye);
    if (length(across) < 1e-5) {
        across = across0;
    }
    across = normalize(across);
    // Never drawn narrower than 1.5 pixels: a thinner ribbon breaks up into
    // dots. What it is wider than it should be goes to the fragment as a
    // share of coverage, so thin hair fades rather than thickens.
    let width = u_data.data[0].w * mix(1.0, u_data.data[1].w, s);
    let up = vec3<f32>(frame.view[0][1], frame.view[1][1], frame.view[2][1]);
    let ca = frame.view_proj * vec4<f32>(p, 1.0);
    let ce = frame.view_proj * vec4<f32>(p + up, 1.0);
    var px_per_unit = 1e6;
    if (ca.w > 1e-4 && ce.w > 1e-4) {
        px_per_unit = abs(ce.y / ce.w - ca.y / ca.w) * frame.viewport_size.y * 0.5;
    }
    let width_px = width * px_per_unit;
    let drawn = max(width_px, 1.5) / max(px_per_unit, 1e-6);
    out.world_pos = p + across * (side * drawn * 0.5);
    out.world_normal = root_n;
    out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);
    out.clip_zw = out.clip_pos.zw;
    out.view_z = -(frame.view * vec4<f32>(out.world_pos, 1.0)).z;
    out.world_tangent = vec4<f32>(t_dir, 1.0);
    // (s, —, across, the share of the ribbon the hair's true width fills)
    out.proj_uv = vec4<f32>(s, 1.0, side, clamp(width_px / max(width_px, 1.5), 0.0, 1.0));
    out.vertex_color = vec4<f32>(in.color.rgb, 1.0);
    return out;
}
"#;
