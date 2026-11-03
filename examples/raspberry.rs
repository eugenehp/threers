//! A raspberry, procedurally: 117 drupelets, dried styles, and fuzz you can
//! count — lit by nothing but a studio built out of three soft boxes.
//!
//! ```text
//! cargo run --release --example raspberry
//! cargo run --release --example raspberry -- --size 1600x1200 --fuzz 98000
//! ```
//!
//! | Flag | Effect |
//! |------|--------|
//! | `--size WxH` | output resolution (default 1280x1600) |
//! | `--fuzz N` | number of hairs (default 48000; 0 for none) |
//! | `--drupelets N` | number of drupelets (default 117) |
//! | `--supersample N` | render at N× and resolve (default 3) |
//! | `--no-styles` | skip the dried styles at the apex |
//! | `--lamp` | switch on the small lamp inside the hollow core |
//! | `--turntable N` | render N frames of a turntable instead of one still |
//!
//! Writes `out/raspberry.png`. Nothing is downloaded and nothing is baked: the
//! geometry, the studio and every map are built in this file.
//!
//! # Why a berry is hard
//!
//! A raspberry is not one surface. It is a hundred-odd little sacs of juice
//! stuck to a hollow cone, each a few millimetres across, each translucent
//! enough that light entering the lit side comes out somewhere else. Render it
//! as one opaque mesh with a red albedo and you get a red golf ball. Three
//! things have to be right, and they are the three things this example is
//! about.
//!
//! **1. The drupelets are separate bodies.** They interpenetrate, and the
//! crevices between them are where almost all of the shading interest lives —
//! the darkening, the saturation, the velvet. So the mesh is 117 overlapping
//! spheres with the faces nobody can see removed, not a smooth hull. Looking
//! into a deep crevice you can end up behind another drupelet's shell; that is
//! the inside of the fruit, and the shader says so rather than leaving a hole.
//!
//! **2. Each drupelet scatters as a bead, not as a surface.** Subsurface
//! scattering with a single blurred irradiance lookup gives "soft", which is
//! not the same as "translucent": the give-away is that the terminator stays
//! put. A real drupelet lets light in on the lit side and glows past the
//! terminator, and from the limb you can see the soft boxes *through* it. Both
//! need the bead's centre and its local radius, so both are computed from the
//! geometry rather than faked with a rim term — `vertex_color.a` carries the
//! drupelet index and a storage buffer carries the 117 centres.
//!
//! **3. The fuzz is most of the silhouette.** The hairs are about ten microns
//! across, which at any sane resolution is a fraction of a pixel. Drawn as
//! geometry at their true width they alias into sparkle; left out, the berry
//! gets a hard edge and reads as plastic. They are drawn as camera-facing
//! ribbons that never go below a pixel wide — widening and fading instead —
//! and the part too fine to draw at all is accounted for by a sheen term on
//! the skin.
//!
//! # The studio
//!
//! There are no lights in the scene. [`Scene::environment`] is a cube map
//! prefiltered from an equirectangular sky that this file generates: ambient
//! with a vertical gradient, plus three rectangular soft boxes with a
//! super-elliptic falloff — a key, a broad fill and a tall strip behind for the
//! rim. The same three directions are handed to the shaders as analytic lights
//! for the translucency, because what a bead does with a soft box is not
//! something a band-limited environment can tell you.
//!
//! The sky is also projected onto nine spherical harmonics on the CPU. That is
//! what the diffuse term integrates, and it is why the creases stay red instead
//! of going grey: ambient occlusion is applied per channel against the albedo,
//! so the channel the flesh does not absorb is the one that survives in a
//! crease.
//!
//! # Units
//!
//! One unit is one centimetre and the berry is 2.3 units tall, because the
//! scattering distances are physical: the subsurface radii are 2.2 mm in red,
//! 0.22 mm in green and 0.13 mm in blue, and a Beer–Lambert chord only means
//! something if the mesh is the size of the real thing.

use std::f32::consts::{PI, TAU};
use std::sync::Arc;

use threers::{
    encode_png, BufferAttribute, BufferGeometry, Color, CubeTexture, HeadlessRenderer, Material,
    PerspectiveCamera, PmremGenerator, Scene, ShaderMaterial, ToneMapping, Vector3,
};

// ---------------------------------------------------------------------------
// The berry, in centimetres
// ---------------------------------------------------------------------------

/// Apex height. The base rim sits at y = 0.
pub const BERRY_H: f32 = 2.30;
/// Radius at the widest point.
pub const BERRY_R: f32 = 1.11;
/// Default drupelet count.
pub const DRUPELETS: usize = 117;
/// Longitude/latitude segments per drupelet shell.
pub const DRUP_U: usize = 24;
pub const DRUP_V: usize = 16;

/// Linear base colours, measured off ripe fruit.
const ALBEDO_RIPE: [f32; 3] = [0.36, 0.0025, 0.012];
const ALBEDO_BRIGHT: [f32; 3] = [0.47, 0.006, 0.017];
const ALBEDO_DARK: [f32; 3] = [0.20, 0.001, 0.012];
/// The cavity lining, which is pale and matte where the flesh tore.
const ALBEDO_INNER: [f32; 3] = [0.74, 0.24, 0.23];
/// Mean free path per channel, in centimetres.
const SSS_RADIUS: [f32; 3] = [0.22, 0.022, 0.0132];
/// Hair colour, linear.
const FUZZ_COLOR: [f32; 3] = [0.92, 0.86, 0.84];
const SHEEN: f32 = 0.12;
/// What fraction of the light meeting a hair leaves it towards the camera.
/// A single scattering event off a fibre, not a Lambertian surface's half.
const FUZZ_SCATTER: f32 = 0.30;
const IOR: f32 = 1.38;
const ROUGHNESS_LO: f32 = 0.19;
const ROUGHNESS_HI: f32 = 0.40;

/// Equirectangular resolution of the generated studio.
pub const SKY_W: usize = 1024;
pub const SKY_H: usize = 512;
/// Cube face size the studio is prefiltered to.
pub const CUBE: u32 = 256;

// ---------------------------------------------------------------------------
// Little maths helpers
// ---------------------------------------------------------------------------

fn v3(x: f32, y: f32, z: f32) -> Vector3 {
    Vector3::new(x, y, z)
}

fn dot(a: Vector3, b: Vector3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn len(a: Vector3) -> f32 {
    dot(a, a).sqrt()
}

fn norm(a: Vector3) -> Vector3 {
    let l = len(a);
    if l > 1e-20 {
        v3(a.x / l, a.y / l, a.z / l)
    } else {
        v3(0.0, 1.0, 0.0)
    }
}

fn add(a: Vector3, b: Vector3) -> Vector3 {
    v3(a.x + b.x, a.y + b.y, a.z + b.z)
}

fn sub(a: Vector3, b: Vector3) -> Vector3 {
    v3(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn mul(a: Vector3, s: f32) -> Vector3 {
    v3(a.x * s, a.y * s, a.z * s)
}

fn cross(a: Vector3, b: Vector3) -> Vector3 {
    v3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if (e1 - e0).abs() < 1e-20 {
        return if x < e0 { 0.0 } else { 1.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Deterministic hash in [0, 1). Everything random in this file comes from here,
/// so the same flags always produce the same berry.
fn hash1(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x >> 8) as f32 / 16_777_216.0
}

fn hash2(seed: u32, i: u32) -> f32 {
    hash1(seed.wrapping_mul(0x9e37_79b9) ^ i.wrapping_mul(0x85eb_ca6b))
}

// ---------------------------------------------------------------------------
// The receptacle: the cone the drupelets are stuck to
// ---------------------------------------------------------------------------

/// Radius of the receptacle at height fraction `v`, as a fraction of [`BERRY_R`].
///
/// A raspberry is a thimble rather than a ball: the base is nearly cylindrical
/// where it let go of the plant, the widest point sits below the middle, and
/// the apex is a rounded point. Those three facts are the whole function.
pub fn profile(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    (1.0 - v.powf(4.5)).max(0.0).powf(0.40) * (0.80 + 0.20 * smoothstep(0.0, 0.30, v))
}

/// The camera the shot is composed for: three quarters on, slightly above, far
/// enough back that the berry is not cropped by its own fuzz.
pub fn studio_camera(aspect: f32) -> PerspectiveCamera {
    let mut camera = PerspectiveCamera::new(34.0, aspect, 0.05, 60.0);
    camera.position = v3(3.25, 2.60, 4.90);
    camera.target = v3(0.0, BERRY_H * 0.48, 0.0);
    camera
}

/// Point on the receptacle in direction `d` from the berry's own centre.
///
/// The shape is star-convex about [`berry_center`], so a bisection on the ray
/// parameter is exact enough and cannot miss.
fn receptacle_hit(d: Vector3) -> Vector3 {
    let c = berry_center();
    let d = norm(d);
    // `inside` is the signed-ish test: positive when the point is still inside.
    let inside = |t: f32| -> f32 {
        let p = add(c, mul(d, t));
        let v = p.y / BERRY_H;
        if v <= 0.0 || v >= 1.0 {
            return -1.0;
        }
        let r = (p.x * p.x + p.z * p.z).sqrt();
        BERRY_R * profile(v) - r
    };
    let mut lo = 0.0_f32;
    let mut hi = 3.0_f32;
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if inside(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    add(c, mul(d, 0.5 * (lo + hi)))
}

fn berry_center() -> Vector3 {
    v3(0.0, BERRY_H * 0.42, 0.0)
}

/// Outward normal of the receptacle at a surface point, by finite differences
/// on the two surface parameters.
fn receptacle_normal(p: Vector3) -> Vector3 {
    let c = berry_center();
    let d = norm(sub(p, c));
    let t = norm(if d.y.abs() > 0.99 {
        cross(v3(1.0, 0.0, 0.0), d)
    } else {
        cross(v3(0.0, 1.0, 0.0), d)
    });
    let b = norm(cross(d, t));
    let e = 0.012;
    let pu = receptacle_hit(add(d, mul(t, e)));
    let pv = receptacle_hit(add(d, mul(b, e)));
    let n = cross(sub(pu, p), sub(pv, p));
    let n = norm(n);
    // The cross product's sign depends on the local frame; point it outward.
    if dot(n, d) < 0.0 {
        mul(n, -1.0)
    } else {
        n
    }
}

// ---------------------------------------------------------------------------
// Drupelet packing
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Drupelet {
    /// Centre of the sac, model space.
    pub center: Vector3,
    /// Radius of the sac.
    pub radius: f32,
    /// Outward normal of the receptacle under it.
    pub outward: Vector3,
    /// Per-drupelet random, in [0, 1).
    pub random: f32,
}

/// Pack `n` drupelets over the receptacle.
///
/// Fibonacci directions give an even spread on a *sphere*; this shape is not
/// one, so they are projected onto it and then relaxed. Relaxation is what
/// produces the hexagonal neighbourhoods a real berry has — a direct Fibonacci
/// map leaves visible spiral seams, and on a berry a spiral seam reads
/// immediately as computer graphics.
pub fn pack_drupelets(n: usize) -> Vec<Drupelet> {
    let golden = PI * (3.0 - 5.0_f32.sqrt());
    let mut dirs: Vec<Vector3> = (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let a = golden * i as f32;
            // Squashed towards the equator: the receptacle is taller than it is
            // wide, so equal angles are not equal areas.
            norm(v3(r * a.cos(), y * 1.25, r * a.sin()))
        })
        .collect();

    // Repulsion on the surface: push apart, re-project, repeat. Forty passes is
    // where the mean nearest-neighbour distance stops moving.
    for pass in 0..40 {
        let pts: Vec<Vector3> = dirs.iter().map(|d| receptacle_hit(*d)).collect();
        let step = 0.55 * (1.0 - pass as f32 / 40.0) + 0.05;
        let mut moved = Vec::with_capacity(n);
        for i in 0..n {
            let mut push = v3(0.0, 0.0, 0.0);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = sub(pts[i], pts[j]);
                let l = len(d);
                if l < 1e-6 {
                    continue;
                }
                // 1/r² falls off fast enough that only the ring of neighbours
                // matters, which is what keeps the packing local.
                push = add(push, mul(d, 1.0 / (l * l * l)));
            }
            let target = add(pts[i], mul(push, step * 0.004));
            moved.push(norm(sub(target, berry_center())));
        }
        dirs = moved;
    }

    let pts: Vec<Vector3> = dirs.iter().map(|d| receptacle_hit(*d)).collect();

    // Radius from the local spacing, so drupelets near the apex — which are
    // packed tighter — come out smaller, as they do on the fruit.
    (0..n)
        .map(|i| {
            let mut d: Vec<f32> = (0..n)
                .filter(|j| *j != i)
                .map(|j| len(sub(pts[i], pts[j])))
                .collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let spacing = d[..6.min(d.len())].iter().sum::<f32>() / 6.0_f32.min(d.len() as f32);
            let random = hash2(7, i as u32);
            // Two neighbours a spacing apart with radius = spacing/1.9 barely
            // touch, which is what leaves a groove between them. Go much above
            // that and the domes merge into one lumpy surface — the first
            // version of this used 1.70 and rendered a blackberry.
            let radius = spacing / 1.90 * (0.88 + 0.26 * random);
            let outward = receptacle_normal(pts[i]);
            // Sunk only a fifth of the way in, so what shows is most of a
            // sphere rather than a shallow cap.
            let center = sub(pts[i], mul(outward, radius * 0.20));
            Drupelet {
                center,
                radius,
                outward,
                random,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The skin mesh
// ---------------------------------------------------------------------------

pub struct Mask {
    /// How exposed this vertex is: 1 on an open dome, 0 buried in a neighbour.
    pub free: f32,
    /// How far into the fruit: 0 on the outside, 1 on the cavity lining.
    pub depth: f32,
}

/// `free`/`depth` for a point on drupelet `i`.
pub fn mask_at(p: Vector3, i: usize, drups: &[Drupelet]) -> Mask {
    let mut clearance = f32::INFINITY;
    for (j, d) in drups.iter().enumerate() {
        if j == i {
            continue;
        }
        clearance = clearance.min(len(sub(p, d.center)) - d.radius);
    }
    let r = drups[i].radius;
    let free = smoothstep(-0.02 * r, 0.30 * r, clearance);
    // Depth is measured against the receptacle, not against the neighbours: it
    // is what the shader uses for "how much flesh is between the hollow and
    // here", so it has to mean a distance into the body.
    let surf = receptacle_hit(sub(p, berry_center()));
    let out = dot(sub(p, surf), receptacle_normal(surf));
    let depth = smoothstep(0.10, -0.26, out);
    Mask { free, depth }
}

/// One drupelet's shell, with the faces nobody can see dropped.
///
/// Returns interleaved vertices and a local index list.
#[allow(clippy::type_complexity)]
fn drupelet_shell(
    i: usize,
    drups: &[Drupelet],
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[f32; 4]>, Vec<u32>) {
    let d = drups[i];
    // A sac is not a ball: it is flattened against the receptacle and very
    // slightly egg-shaped, more so the closer it is to the apex.
    let flatten = 0.86 + 0.10 * d.random;
    let t = norm(if d.outward.y.abs() > 0.99 {
        cross(v3(1.0, 0.0, 0.0), d.outward)
    } else {
        cross(v3(0.0, 1.0, 0.0), d.outward)
    });
    let b = norm(cross(d.outward, t));

    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uvs = Vec::new();
    let mut col = Vec::new();
    // Which vertices are on the half of the sphere that faces into the fruit.
    let mut inward = Vec::new();
    for iv in 0..=DRUP_V {
        // `theta` from the outward pole, so the dome is the v = 0 end.
        let fv = iv as f32 / DRUP_V as f32;
        let theta = fv * PI;
        for iu in 0..=DRUP_U {
            let fu = iu as f32 / DRUP_U as f32;
            let phi = fu * TAU;
            // Unit sphere in the drupelet's own frame.
            let local = v3(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            );
            // A shallow dimple at the pole and a little lumpiness, both tiny:
            // a drupelet is smooth, it is the packing that is not.
            let lump = 1.0
                + 0.018 * (hash2(i as u32 * 31 + 3, iu as u32 * 7 + iv as u32 * 131) - 0.5)
                - 0.030 * smoothstep(0.35, 0.0, theta);
            let dir = add(
                add(mul(t, local.x), mul(b, local.y)),
                mul(d.outward, local.z * flatten),
            );
            let dir = norm(dir);
            let p = add(d.center, mul(dir, d.radius * lump));
            let m = mask_at(p, i, drups);
            pos.push([p.x, p.y, p.z]);
            nrm.push([dir.x, dir.y, dir.z]);
            uvs.push([fu, fv]);
            col.push([m.free, m.depth, d.random, i as f32]);
            inward.push(theta > 1.75);
        }
    }

    // Keep a quad if any of its corners is not buried.
    let stride = DRUP_U + 1;
    // Three quarters of every sphere is inside something. A vertex is hidden if
    // it is inside a *neighbour*, or if it is on the half facing into the fruit
    // and has passed below the receptacle — the second clause is what removes
    // the back of each sac, and the two have to be an `or`. Asking for both at
    // once, which is where this started, culls almost nothing: a vertex deep
    // inside a neighbour is usually still outside the receptacle. The mistake
    // does not show in the picture, because what it fails to remove is by
    // definition hidden; it just doubles the triangle count and leaves
    // coincident shells arguing about depth down in the crevices.
    let buried = |k: usize| -> bool { col[k][0] <= 0.001 || (inward[k] && col[k][1] > 0.25) };
    let mut idx = Vec::new();
    for iv in 0..DRUP_V {
        for iu in 0..DRUP_U {
            let a = iv * stride + iu;
            let bb = a + 1;
            let c = a + stride;
            let e = c + 1;
            if buried(a) && buried(bb) && buried(c) && buried(e) {
                continue;
            }
            idx.extend_from_slice(&[a as u32, c as u32, bb as u32]);
            idx.extend_from_slice(&[bb as u32, c as u32, e as u32]);
        }
    }
    (pos, nrm, uvs, col, idx)
}

/// The whole skin: every drupelet shell plus a shell closing the hollow core.
pub fn build_skin(drups: &[Drupelet]) -> BufferGeometry {
    let mut pos: Vec<f32> = Vec::new();
    let mut nrm: Vec<f32> = Vec::new();
    let mut uvs: Vec<f32> = Vec::new();
    let mut col: Vec<f32> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();

    for i in 0..drups.len() {
        let (p, n, u, c, ix) = drupelet_shell(i, drups);
        let base = (pos.len() / 3) as u32;
        for v in &p {
            pos.extend_from_slice(v);
        }
        for v in &n {
            nrm.extend_from_slice(v);
        }
        for v in &u {
            uvs.extend_from_slice(v);
        }
        for v in &c {
            col.extend_from_slice(v);
        }
        idx.extend(ix.iter().map(|k| k + base));
    }

    // The cavity lining. Without it the berry is a shell of spheres with gaps
    // at the base, and the camera can see straight through the bottom.
    let (p, n, u, c, ix) = cavity_lining(drups);
    let base = (pos.len() / 3) as u32;
    for v in &p {
        pos.extend_from_slice(v);
    }
    for v in &n {
        nrm.extend_from_slice(v);
    }
    for v in &u {
        uvs.extend_from_slice(v);
    }
    for v in &c {
        col.extend_from_slice(v);
    }
    idx.extend(ix.iter().map(|k| k + base));

    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    g.set_attribute("normal", BufferAttribute::new(nrm, 3));
    g.set_attribute("uv", BufferAttribute::new(uvs, 2));
    g.set_attribute("color", BufferAttribute::new(col, 4));
    g.set_index(idx);
    g.compute_bounding_box();
    g.compute_bounding_sphere();
    g
}

/// A closed surface just inside the drupelets, facing inward.
#[allow(clippy::type_complexity)]
fn cavity_lining(
    drups: &[Drupelet],
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[f32; 4]>, Vec<u32>) {
    const NU: usize = 64;
    const NV: usize = 40;
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uvs = Vec::new();
    let mut col = Vec::new();
    for iv in 0..=NV {
        let fv = iv as f32 / NV as f32;
        // Stop a little short of the poles so the lining is a capsule rather
        // than a sphere with a pinch.
        let v = mix(0.02, 0.985, fv);
        for iu in 0..=NU {
            let fu = iu as f32 / NU as f32;
            let phi = fu * TAU;
            let r = BERRY_R * profile(v) - 0.13;
            let p = v3(r.max(0.02) * phi.cos(), v * BERRY_H, r.max(0.02) * phi.sin());
            // Inward-facing: this is seen from inside a crevice.
            let n = norm(mul(sub(p, berry_center()), -1.0));
            let near = nearest(p, drups);
            pos.push([p.x, p.y, p.z]);
            nrm.push([n.x, n.y, n.z]);
            uvs.push([fu, fv]);
            col.push([0.0, 1.0, drups[near].random, near as f32]);
        }
    }
    let stride = NU + 1;
    let mut idx = Vec::new();
    for iv in 0..NV {
        for iu in 0..NU {
            let a = (iv * stride + iu) as u32;
            let b = a + 1;
            let c = a + stride as u32;
            let e = c + 1;
            // Wound the other way round: the visible side is the inside.
            idx.extend_from_slice(&[a, b, c]);
            idx.extend_from_slice(&[b, e, c]);
        }
    }
    (pos, nrm, uvs, col, idx)
}

fn nearest(p: Vector3, drups: &[Drupelet]) -> usize {
    let mut best = 0;
    let mut bd = f32::INFINITY;
    for (i, d) in drups.iter().enumerate() {
        let l = len(sub(p, d.center));
        if l < bd {
            bd = l;
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// The studio
// ---------------------------------------------------------------------------

/// One rectangular soft box.
#[derive(Clone, Copy)]
pub struct SoftBox {
    /// Direction from the berry towards the light.
    pub dir: Vector3,
    /// Colour × radiance.
    pub radiance: [f32; 3],
    /// Half-angle the box subtends, as a tangent.
    pub tan: f32,
    /// > 1 makes a tall strip out of a square box.
    pub stretch: f32,
}

pub fn studio_lights() -> [SoftBox; 3] {
    [
        // Key: high, front left, large and slightly warm.
        SoftBox {
            dir: norm(v3(-0.62, 0.74, 0.52)),
            radiance: [120.0, 113.0, 104.0],
            tan: 0.40,
            stretch: 1.25,
        },
        // Fill: low, front right, broad and cool. Opens up the shadow side
        // without putting a second highlight on every dome.
        SoftBox {
            dir: norm(v3(0.86, 0.16, 0.46)),
            radiance: [14.0, 15.0, 18.0],
            tan: 0.62,
            stretch: 0.85,
        },
        // Rim: behind and slightly above, a tall narrow strip. This is the one
        // that does the work — it is what the drupelets glow with, and a strip
        // is used rather than a box because a narrow source is what makes a
        // translucent body read as translucent instead of merely bright.
        SoftBox {
            dir: norm(v3(0.18, 0.46, -0.87)),
            radiance: [320.0, 250.0, 225.0],
            tan: 0.15,
            stretch: 3.6,
        },
    ]
}

/// Ambient radiance and its vertical gradient: a grey room, brighter at the top.
/// The room the boxes stand in. A studio is not a black void, but it is a long
/// way below the panels: the ratio here is about a thousand to one, which is
/// what leaves room for a specular highlight to be the brightest thing in the
/// frame. At thirty to one — where this started — there is no highlight at all,
/// because prefiltering averages a panel that dim down to the ambient.
const AMBIENT: [f32; 3] = [0.105, 0.102, 0.115];
const AMBIENT_GRADIENT: f32 = 0.60;
/// How boxy the soft boxes are: 0 = elliptical, 1 = almost rectangular.
const SQUARE: f32 = 0.70;

/// Radiance arriving from direction `d`.
pub fn studio_radiance(d: Vector3, lights: &[SoftBox; 3]) -> [f32; 3] {
    let d = norm(d);
    let mut col = [
        AMBIENT[0] * (1.0 + AMBIENT_GRADIENT * d.y),
        AMBIENT[1] * (1.0 + AMBIENT_GRADIENT * d.y),
        AMBIENT[2] * (1.0 + AMBIENT_GRADIENT * d.y),
    ];
    let p = mix(2.0, 8.0, SQUARE);
    for l in lights.iter() {
        let c = dot(d, l.dir);
        if c <= 0.0 {
            continue;
        }
        let up = if l.dir.y.abs() > 0.99 {
            v3(1.0, 0.0, 0.0)
        } else {
            v3(0.0, 1.0, 0.0)
        };
        let t = norm(cross(up, l.dir));
        let b = cross(l.dir, t);
        let qx = (dot(d, t) / c / l.tan).abs();
        let qy = (dot(d, b) / l.stretch / c / l.tan).abs();
        let m = (qx.powf(p) + qy.powf(p)).powf(1.0 / p);
        let box_f = 1.0 - smoothstep(0.92, 1.08, m);
        // Slightly hotter in the middle, like a real diffuser over a lamp.
        let f = box_f * (1.0 - 0.25 * (m * m).min(1.0));
        for (c, r) in col.iter_mut().zip(l.radiance.iter()) {
            *c += r * f;
        }
    }
    col
}

/// The studio as an equirectangular HDR image, plus its nine-coefficient
/// spherical-harmonic projection.
pub fn build_studio(lights: &[SoftBox; 3]) -> (Vec<f32>, [[f32; 3]; 9]) {
    let mut img = vec![0.0f32; SKY_W * SKY_H * 3];
    let mut sh = [[0.0f32; 3]; 9];
    for y in 0..SKY_H {
        let theta = (y as f32 + 0.5) / SKY_H as f32 * PI;
        let (st, ct) = (theta.sin(), theta.cos());
        // Solid angle of one texel at this latitude.
        let dw = (PI / SKY_H as f32) * (TAU / SKY_W as f32) * st;
        for x in 0..SKY_W {
            let phi = (x as f32 + 0.5) / SKY_W as f32 * TAU - PI;
            let d = v3(st * phi.sin(), ct, -st * phi.cos());
            let c = studio_radiance(d, lights);
            let o = (y * SKY_W + x) * 3;
            img[o] = c[0];
            img[o + 1] = c[1];
            img[o + 2] = c[2];
            let basis = sh_basis(d);
            for (i, bi) in basis.iter().enumerate() {
                for k in 0..3 {
                    sh[i][k] += c[k] * bi * dw;
                }
            }
        }
    }
    (img, sh)
}

/// The nine real spherical harmonics, in the order the shader reconstructs in.
fn sh_basis(d: Vector3) -> [f32; 9] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        0.282_095,
        0.488_603 * y,
        0.488_603 * z,
        0.488_603 * x,
        1.092_548 * x * y,
        1.092_548 * y * z,
        0.315_392 * (3.0 * z * z - 1.0),
        1.092_548 * x * z,
        0.546_274 * (x * x - y * y),
    ]
}

/// Irradiance at a surface with normal `n`, from the nine coefficients. The
/// shader has the same function; the CPU one is used to light the fuzz.
pub fn sh_irradiance(sh: &[[f32; 3]; 9], n: Vector3) -> [f32; 3] {
    const C1: f32 = 0.429_043;
    const C2: f32 = 0.511_664;
    const C3: f32 = 0.743_125;
    const C4: f32 = 0.886_227;
    const C5: f32 = 0.247_708;
    let (x, y, z) = (n.x, n.y, n.z);
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        out[k] = C4 * sh[0][k]
            + 2.0 * C2 * (sh[1][k] * y + sh[2][k] * z + sh[3][k] * x)
            + 2.0 * C1 * (sh[4][k] * x * y + sh[5][k] * y * z + sh[7][k] * x * z)
            + C3 * sh[6][k] * z * z
            - C5 * sh[6][k]
            + C1 * sh[8][k] * (x * x - y * y);
        out[k] = out[k].max(0.0);
    }
    out
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

pub struct Args {
    pub w: u32,
    pub h: u32,
    pub fuzz: usize,
    pub drupelets: usize,
    pub supersample: u32,
    pub exposure: f32,
    pub styles: bool,
    pub lamp: bool,
    pub turntable: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            w: 1280,
            h: 1600,
            fuzz: 48_000,
            drupelets: DRUPELETS,
            supersample: 3,
            exposure: 0.058,
            styles: true,
            lamp: false,
            turntable: 0,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--size" if i + 1 < argv.len() => {
                if let Some((w, h)) = argv[i + 1].split_once('x') {
                    a.w = w.parse().unwrap_or(a.w);
                    a.h = h.parse().unwrap_or(a.h);
                }
                i += 1;
            }
            "--fuzz" if i + 1 < argv.len() => {
                a.fuzz = argv[i + 1].parse().unwrap_or(a.fuzz);
                i += 1;
            }
            "--drupelets" if i + 1 < argv.len() => {
                a.drupelets = argv[i + 1].parse().unwrap_or(a.drupelets).max(8);
                i += 1;
            }
            "--supersample" if i + 1 < argv.len() => {
                a.supersample = argv[i + 1].parse().unwrap_or(a.supersample).clamp(1, 4);
                i += 1;
            }
            "--turntable" if i + 1 < argv.len() => {
                a.turntable = argv[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            "--exposure" if i + 1 < argv.len() => {
                a.exposure = argv[i + 1].parse().unwrap_or(a.exposure);
                i += 1;
            }
            "--no-styles" => a.styles = false,
            "--lamp" => a.lamp = true,
            other => eprintln!("ignoring unknown flag {other}"),
        }
        i += 1;
    }
    a
}

fn main() {
    let args = parse_args();
    std::fs::create_dir_all("out").expect("create out/");

    let t0 = std::time::Instant::now();
    let lights = studio_lights();
    let (equirect, sh) = build_studio(&lights);
    // Two steps, and both are load-bearing. `from_equirect_f32` resamples to a
    // cube and keeps the f32 faces, so a 300-radiance panel is still 300 rather
    // than a tone-mapped 1.0. `generate_pmrem` is what actually prefilters it
    // into the roughness chain and builds the CubeUV atlas the renderer samples
    // — without it `sample_env_cube` reads one unblurred level and roughness
    // means nothing.
    let cube = PmremGenerator::from_equirect_f32(&equirect, SKY_W as u32, SKY_H as u32, CUBE);
    let cube = PmremGenerator::generate_pmrem(&cube, CUBE);
    println!(
        "studio: {}x{} equirect, prefiltered to {CUBE}² — {:.1?}",
        SKY_W,
        SKY_H,
        t0.elapsed()
    );

    let t1 = std::time::Instant::now();
    let drups = pack_drupelets(args.drupelets);
    let skin = build_skin(&drups);
    let tris = skin.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
    println!(
        "skin: {} drupelets, {} triangles — {:.1?}",
        drups.len(),
        tris,
        t1.elapsed()
    );

    let mut camera = studio_camera(args.w as f32 / args.h as f32);

    let mut renderer = HeadlessRenderer::builder()
        .size(args.w, args.h)
        .supersample(args.supersample)
        .build()
        .expect("headless renderer");
    // Tone mapping is applied inside the materials, so the renderer must not
    // apply a second curve on the way out.
    renderer
        .renderer()
        .set_tone_mapping(ToneMapping::None, 1.0);
    renderer.set_film_grade(0.55, 0.004);

    let frames = args.turntable.max(1);
    for f in 0..frames {
        let spin = if args.turntable > 0 {
            f as f32 / args.turntable as f32 * TAU
        } else {
            0.0
        };
        if args.turntable > 0 {
            let r = 5.3_f32;
            camera.position = v3(r * spin.sin(), 2.35, r * spin.cos());
        }
        // The camera has to be settled before the scene is built: the hairs are
        // flat ribbons turned to face it, and their width is chosen so that
        // none of them lands under a pixel.
        let mut scene = build_scene(
            &drups,
            &skin,
            &sh,
            &lights,
            &args,
            &camera,
            Arc::new(cube.clone()),
        );
        let rgba = renderer.render_to_rgba_resolved(&mut scene, &camera);
        let path = if args.turntable > 0 {
            format!("out/raspberry_{f:03}.png")
        } else {
            "out/raspberry.png".to_string()
        };
        std::fs::write(&path, encode_png(args.w, args.h, &rgba)).expect("write png");
        println!("wrote {path}");
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_scene(
    drups: &[Drupelet],
    skin: &BufferGeometry,
    sh: &[[f32; 3]; 9],
    lights: &[SoftBox; 3],
    args: &Args,
    camera: &PerspectiveCamera,
    env: Arc<CubeTexture>,
) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.environment = Some(env);

    // The backdrop first, so everything else sorts in front of it.
    let mut back = ShaderMaterial::new(format!("{COMMON_WGSL}{BACKDROP_FRAGMENT}"));
    back.data = vec![[args.exposure, 0.0, 0.0, 0.0]];
    back.storage1 = studio_storage(sh, lights, args);
    scene.add_mesh(backdrop_geometry(), Material::Shader(back));

    let mut mat = ShaderMaterial::new(format!("{COMMON_WGSL}{SKIN_FRAGMENT}"));
    mat.data = skin_uniforms(args);
    mat.storage0 = drupelet_storage(drups);
    mat.storage1 = studio_storage(sh, lights, args);
    scene.add_mesh(skin.clone(), Material::Shader(mat));

    if args.styles {
        // The common block rides with the *vertex* source: the module is
        // assembled preamble, then vertex, then fragment, and WGSL wants a
        // function declared before it is called.
        let mut sm = ShaderMaterial::new(STYLE_FRAGMENT)
            .with_vertex(format!("{COMMON_WGSL}{STYLE_VERTEX}"));
        sm.data = vec![
            [args.exposure, 1.0, 1.0, 0.0],
            [0.0, BERRY_H * 0.42, 0.0, 0.0],
        ];
        sm.storage1 = studio_storage(sh, lights, args);
        sm.transparent = true;
        sm.side = 2;
        scene.add_mesh(build_styles(drups), Material::Shader(sm));
    }

    if args.fuzz > 0 {
        let mut fm = ShaderMaterial::new(format!("{COMMON_WGSL}{FUZZ_FRAGMENT}"));
        fm.data = vec![[args.exposure, 0.0, 0.0, 0.0]];
        fm.storage1 = studio_storage(sh, lights, args);
        fm.transparent = true;
        // Both sides: a ribbon turned edge-on to the camera by a hair's own
        // curvature would otherwise vanish.
        fm.side = 2;
        scene.add_mesh(
            build_fuzz(drups, sh, lights, args, camera),
            Material::Shader(fm),
        );
    }
    scene
}

// ---------------------------------------------------------------------------
// The backdrop
// ---------------------------------------------------------------------------

/// A single quad a long way behind the berry. Its shader puts a soft pool of
/// light behind the subject and lets the corners fall to black, which is what
/// separates a dark red silhouette from a dark background without lighting the
/// background enough to compete with it.
fn backdrop_geometry() -> BufferGeometry {
    let (w, h, z) = (26.0f32, 20.0f32, -7.0f32);
    let pos = vec![
        -w, -h * 0.35, z, w, -h * 0.35, z, w, h * 0.9, z, -w, h * 0.9, z,
    ];
    let nrm = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let uv = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let col = vec![1.0; 16];
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    g.set_attribute("normal", BufferAttribute::new(nrm, 3));
    g.set_attribute("uv", BufferAttribute::new(uv, 2));
    g.set_attribute("color", BufferAttribute::new(col, 4));
    g.set_index(vec![0, 1, 2, 0, 2, 3]);
    g.compute_bounding_box();
    g.compute_bounding_sphere();
    g
}

// ---------------------------------------------------------------------------
// The dried styles
// ---------------------------------------------------------------------------

/// The wiry remains of the styles, which stick out of the crevices.
///
/// They are the detail that stops a berry reading as a pile of spheres: they
/// break the silhouette, they catch the rim light, and they are the first thing
/// the eye finds because they are the only hard edges on the whole subject.
///
/// Built as flat ribbons along each strand, two vertices a segment. The ribbon
/// is turned to face the camera and widened to at least a pixel in the vertex
/// shader — a style is about a tenth of a millimetre across and tapers to
/// nothing, so drawn as a solid tube its tip falls under a pixel and breaks
/// into dashes. `color.a` carries the radius at each point for that.
fn build_styles(drups: &[Drupelet]) -> BufferGeometry {
    const SEGS: usize = 24;
    let mut pos: Vec<f32> = Vec::new();
    let mut nrm: Vec<f32> = Vec::new();
    let mut uvs: Vec<f32> = Vec::new();
    let mut col: Vec<f32> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();

    for (i, d) in drups.iter().enumerate() {
        // Up to three per drupelet, thinning towards the base where they have
        // rubbed off. A real berry carries a few hundred of these.
        let up = (d.center.y / BERRY_H).clamp(0.0, 1.0);
        let budget = 0.45 + 2.6 * smoothstep(0.10, 0.85, up);
        for k in 0..3u32 {
            if (k as f32) >= budget && hash2(21 + k, i as u32) > budget - k as f32 {
                continue;
            }
            let seed = i as u32 * 7 + k;
            let r0 = hash2(22, seed);
            let r1 = hash2(23, seed);
            let r2 = hash2(24, seed);
            let dryness = 0.25 + 0.7 * r2;
            // One to two and a half millimetres, and about a tenth of a
            // millimetre at the root.
            let length = 0.085 + 0.165 * r0;
            let radius = 0.0060 * (0.7 + 0.6 * r1);

            let t = norm(if d.outward.y.abs() > 0.99 {
                cross(v3(1.0, 0.0, 0.0), d.outward)
            } else {
                cross(v3(0.0, 1.0, 0.0), d.outward)
            });
            let b = norm(cross(d.outward, t));
            let a = hash2(25, seed) * TAU;
            let lean = add(mul(t, a.cos()), mul(b, a.sin()));
            let root_dir = norm(add(d.outward, mul(lean, 0.75 + 0.5 * r1)));
            let root = add(d.center, mul(root_dir, d.radius * 0.99));
            // Away from the berry, then bending over and drooping: a style is a
            // cantilever that dried while it bent.
            let curl = add(mul(lean, 0.75 + 0.9 * r0), v3(0.0, -0.55 - 0.7 * r2, 0.0));
            let wobble = hash2(26, seed) * TAU;

            let base = (pos.len() / 3) as u32;
            for seg in 0..=SEGS {
                let ft = seg as f32 / SEGS as f32;
                let mut dir = norm(add(root_dir, mul(curl, ft * ft)));
                // A touch of kink, so twenty of them are not twenty copies of
                // one curve.
                let kink = (wobble + ft * 7.0).sin() * 0.06 * ft;
                dir = norm(add(dir, mul(t, kink)));
                let p = add(root, mul(dir, length * ft));
                // Tapered to a point, with a slight swelling at the base.
                let rad = radius * (1.0 - ft).powf(0.55) * (1.0 + 0.5 * (1.0 - ft).powi(6));
                for side in [-1.0f32, 1.0] {
                    pos.extend_from_slice(&[p.x, p.y, p.z]);
                    // The tangent rides in the normal slot; the vertex stage
                    // needs it to build the ribbon's frame.
                    nrm.extend_from_slice(&[dir.x, dir.y, dir.z]);
                    uvs.extend_from_slice(&[side, ft]);
                    col.extend_from_slice(&[ft, dryness, r2, rad]);
                }
            }
            for seg in 0..SEGS {
                let a0 = base + (seg * 2) as u32;
                let a1 = a0 + 1;
                let b0 = a0 + 2;
                let b1 = a0 + 3;
                idx.extend_from_slice(&[a0, b0, a1, a1, b0, b1]);
            }
        }
    }

    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    g.set_attribute("normal", BufferAttribute::new(nrm, 3));
    g.set_attribute("uv", BufferAttribute::new(uvs, 2));
    g.set_attribute("color", BufferAttribute::new(col, 4));
    g.set_index(idx);
    g.compute_bounding_box();
    g.compute_bounding_sphere();
    g
}

// ---------------------------------------------------------------------------
// The fuzz
// ---------------------------------------------------------------------------

/// Hairs, as camera-facing ribbons with their lighting baked per vertex.
///
/// Two things here are not obvious. The first is the width rule: a hair is
/// about ten microns across and the berry is two centimetres, so at any
/// sensible resolution a hair covers a fraction of a pixel. Drawn at its true
/// width it is sampled or missed depending on where the pixel centre lands,
/// which is a definition of aliasing — so a hair thinner than a pixel is
/// *widened* to one pixel and its alpha dropped by the same factor. Coverage is
/// preserved and the sparkle is not.
///
/// The second is that the lighting is computed here rather than in the shader.
/// A `ShaderMaterial` replaces the fragment stage and not the vertex stage, so
/// there is nowhere to put per-strand work on the GPU — but a hair is lit per
/// vertex anyway, because nothing across four control points varies fast enough
/// to need more. The result travels in `color`, and the fragment does little
/// more than apply the round profile.
pub fn build_fuzz(
    drups: &[Drupelet],
    sh: &[[f32; 3]; 9],
    lights: &[SoftBox; 3],
    args: &Args,
    camera: &PerspectiveCamera,
) -> BufferGeometry {
    const PTS: usize = 4;
    let n = args.fuzz;
    let cam = camera.position;
    // Pixels per world unit at one unit of depth, at the resolution actually
    // rendered — supersampling included, because that is where the sampling
    // happens.
    let vp_h = (args.h * args.supersample) as f32;
    let min_px = 1.15f32;

    let mut pos: Vec<f32> = Vec::with_capacity(n * PTS * 2 * 3);
    let mut nrm: Vec<f32> = Vec::with_capacity(n * PTS * 2 * 3);
    let mut uvs: Vec<f32> = Vec::with_capacity(n * PTS * 2 * 2);
    let mut col: Vec<f32> = Vec::with_capacity(n * PTS * 2 * 4);
    let mut idx: Vec<u32> = Vec::with_capacity(n * (PTS - 1) * 6);

    for hair in 0..n {
        let hu = hair as u32;
        let di = (hash2(31, hu) * drups.len() as f32) as usize % drups.len();
        let d = drups[di];
        // A direction on the drupelet, biased towards the exposed crown: the
        // fuzz grows where there is room for it.
        let z = 1.0 - 1.35 * hash2(32, hu).powf(0.75);
        let r = (1.0 - z * z).max(0.0).sqrt();
        let a = hash2(33, hu) * TAU;
        let t = norm(if d.outward.y.abs() > 0.99 {
            cross(v3(1.0, 0.0, 0.0), d.outward)
        } else {
            cross(v3(0.0, 1.0, 0.0), d.outward)
        });
        let b = norm(cross(d.outward, t));
        let root_n = norm(add(
            add(mul(t, r * a.cos()), mul(b, r * a.sin())),
            mul(d.outward, z),
        ));
        let root = add(d.center, mul(root_n, d.radius));
        // Skip the ones that would start inside a neighbour.
        let m = mask_at(root, di, drups);
        if m.free < 0.35 || hash2(34, hu) > m.free {
            continue;
        }

        let kind = hash2(35, hu);
        // Three to eight tenths of a millimetre, and about fifteen microns
        // across. Both numbers matter more than they look: at four times the
        // length this renders a dandelion clock, and at twice the width the
        // hairs stop being sub-pixel, the alpha goes to one, and the fuzz
        // becomes an opaque white fleece instead of a haze.
        let length = 0.030 + 0.050 * kind * kind;
        let root_radius = 0.00052 * (0.65 + 0.8 * hash2(36, hu));
        // Out along the normal, then bending over: a hair is a cantilever.
        let droop = v3(
            (hash2(37, hu) - 0.5) * 1.5,
            -0.35 - 0.9 * hash2(38, hu),
            (hash2(39, hu) - 0.5) * 1.5,
        );

        let mut p = [v3(0.0, 0.0, 0.0); PTS];
        for (k, slot) in p.iter_mut().enumerate() {
            let ft = k as f32 / (PTS - 1) as f32;
            let dir = norm(add(root_n, mul(droop, ft * ft)));
            *slot = add(root, mul(dir, length * ft));
        }

        let base = (pos.len() / 3) as u32;
        for k in 0..PTS {
            let ft = k as f32 / (PTS - 1) as f32;
            let tangent = norm(if k == 0 {
                sub(p[1], p[0])
            } else if k == PTS - 1 {
                sub(p[PTS - 1], p[PTS - 2])
            } else {
                sub(p[k + 1], p[k - 1])
            });
            let to_eye = sub(cam, p[k]);
            let dist = len(to_eye).max(1e-4);
            let view = mul(to_eye, 1.0 / dist);
            let mut side = cross(tangent, view);
            if len(side) < 1e-5 {
                side = norm(cross(tangent, v3(0.0, 1.0, 0.0)));
            } else {
                side = norm(side);
            }

            let radius = root_radius * (1.0 - 0.8 * ft.powf(1.5));
            // `PerspectiveCamera::pixels_per_unit` rather than the formula by
            // hand: `fov` is radians and `new` takes degrees, so the obvious
            // hand-rolled version calls `.to_radians()` on something that
            // already is, is wrong by a factor of 57, and still draws a
            // picture.
            let px_per_unit = camera.pixels_per_unit(dist, vp_h);
            let w_px = 2.0 * radius * px_per_unit;
            let w = w_px.max(min_px);
            let alpha = (w_px / w).clamp(0.0, 1.0);
            let half = 0.5 * w / px_per_unit;

            let shade = hair_shade(p[k], tangent, root_n, d, ft, m.free, view, sh, lights);
            for sgn in [-1.0f32, 1.0] {
                let q = add(p[k], mul(side, sgn * half));
                pos.extend_from_slice(&[q.x, q.y, q.z]);
                nrm.extend_from_slice(&[root_n.x, root_n.y, root_n.z]);
                // `uv.x` carries the across-the-ribbon coordinate the fragment
                // needs for its round profile.
                uvs.extend_from_slice(&[sgn, ft]);
                col.extend_from_slice(&[shade[0], shade[1], shade[2], alpha]);
            }
        }
        for k in 0..PTS - 1 {
            let a0 = base + (k * 2) as u32;
            let a1 = a0 + 1;
            let b0 = a0 + 2;
            let b1 = a0 + 3;
            idx.extend_from_slice(&[a0, b0, a1, a1, b0, b1]);
        }
    }

    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    g.set_attribute("normal", BufferAttribute::new(nrm, 3));
    g.set_attribute("uv", BufferAttribute::new(uvs, 2));
    g.set_attribute("color", BufferAttribute::new(col, 4));
    g.set_index(idx);
    g.compute_bounding_box();
    g.compute_bounding_sphere();
    g
}

/// Radiance leaving one point on a hair.
///
/// A fibre has no normal — it has a tangent and a whole circle of normals — so
/// the shading normal is the one in that circle which faces the camera, mixed
/// back towards the skin's normal because a hair lying against the berry is lit
/// by what the berry is lit by. Forward scattering is the term that matters:
/// it is what makes the fuzz a halo when the rim light is behind it.
#[allow(clippy::too_many_arguments)]
fn hair_shade(
    p: Vector3,
    tangent: Vector3,
    root_n: Vector3,
    d: Drupelet,
    t: f32,
    free: f32,
    view: Vector3,
    sh: &[[f32; 3]; 9],
    lights: &[SoftBox; 3],
) -> [f32; 3] {
    let fibre_n = norm(sub(view, mul(tangent, dot(view, tangent))));
    let nh = norm(add(mul(root_n, 0.55), mul(fibre_n, 0.45)));
    // Tips stick out of the crease the root sits in.
    let occ = mix(mix(0.25, 1.0, free), 1.0, t * 0.6);
    let mut irr = sh_irradiance(sh, nh);
    for v in irr.iter_mut() {
        *v /= PI;
    }
    let mut back = [0.0f32; 3];
    let outward = d.outward;
    for l in lights.iter() {
        let reach = smoothstep(-0.20, 0.35, dot(outward, l.dir));
        let wrap = (dot(nh, l.dir) * 0.6 + 0.4).clamp(0.0, 1.0);
        let solid = PI * l.tan * l.tan * l.stretch;
        let f = dot(mul(view, -1.0), l.dir).clamp(0.0, 1.0).powi(3);
        let sin_tl = (1.0 - dot(tangent, l.dir).powi(2)).max(0.0).sqrt();
        let through = f * sin_tl * smoothstep(-0.4, 0.3, dot(root_n, l.dir)) * reach;
        for k in 0..3 {
            irr[k] += l.radiance[k] * solid * (wrap * wrap * reach / PI);
            back[k] += l.radiance[k] * solid * through;
        }
    }
    // Pale at the tip, picking up the berry's red at the root.
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        let root_tint = [1.0f32, 0.72, 0.70][k];
        let tint = FUZZ_COLOR[k] * mix(1.0, root_tint, (1.0 - t) * 0.6);
        out[k] = tint * (irr[k] * occ + back[k] * 0.05 * mix(0.6, 1.0, occ)) * FUZZ_SCATTER;
    }
    let _ = p;
    out
}

/// Concatenated into each material: a `ShaderMaterial` is its own WGSL module,
/// so the SH reconstruction and the tone curve have to be in every one of them
/// or the four stop agreeing with each other.
const COMMON_WGSL: &str = include_str!("raspberry_common.wgsl");
const BACKDROP_FRAGMENT: &str = include_str!("raspberry_backdrop.wgsl");
const STYLE_VERTEX: &str = include_str!("raspberry_style_vs.wgsl");
const STYLE_FRAGMENT: &str = include_str!("raspberry_style.wgsl");
const FUZZ_FRAGMENT: &str = include_str!("raspberry_fuzz.wgsl");

/// The sixteen `vec4` slots the skin shader reads. Keep in step with the block
/// of `let`s at the top of `raspberry_skin.wgsl` — they are the same list.
fn skin_uniforms(args: &Args) -> Vec<[f32; 4]> {
    let f0 = ((IOR - 1.0) / (IOR + 1.0)).powi(2);
    vec![
        // 0: subsurface, bead glow, see-through, seed
        [0.92, 2.2, 1.35, 0.9],
        // 1: specular, F0, roughness scale, roughness bias
        [1.0, f0, 1.0, 0.0],
        // 2: detail strength, detail tile (cm), sheen, ao power
        [0.22, 0.26, SHEEN, 1.7],
        // 3: env scale, saturation, exposure, reflection
        [1.0, 1.04, args.exposure, 1.0],
        // 4: berry centre, cm of flesh between the hollow and the skin
        [0.0, BERRY_H * 0.42, 0.0, 0.22],
        // 5: the inner lamp's position
        [0.0, BERRY_H * 0.40, 0.0, 0.0],
        // 6: tint, the depth the transmission colour is quoted at
        [1.0, 1.0, 1.0, 0.10],
        // 7: backdrop radiance seen through a drupelet
        [0.012, 0.010, 0.012, if args.lamp { 1.0 } else { 0.0 }],
        // 8: roughness low/high
        [ROUGHNESS_LO, ROUGHNESS_HI, 0.0, 0.0],
        // 9: which term to show on its own. Shading this involved is mostly
        // debugged by looking at one addend at a time, so the switch is part of
        // the example rather than something to add back each time.
        [debug_mode(), 0.0, 0.0, 0.0],
    ]
}

/// `RASPBERRY_DEBUG=diffuse|specular|sheen|glow|ao|dome|free|env|normal`
fn debug_mode() -> f32 {
    match std::env::var("RASPBERRY_DEBUG").unwrap_or_default().as_str() {
        "diffuse" => 1.0,
        "specular" => 2.0,
        "sheen" => 3.0,
        "glow" => 4.0,
        "ao" => 5.0,
        "dome" => 6.0,
        "free" => 7.0,
        "env" => 8.0,
        "normal" => 9.0,
        _ => 0.0,
    }
}

/// Per-drupelet record: centre, radius, outward normal, random. Eight floats so
/// the shader can index it with a shift.
fn drupelet_storage(drups: &[Drupelet]) -> Vec<f32> {
    let mut out = Vec::with_capacity(drups.len() * 8);
    for d in drups {
        out.extend_from_slice(&[
            d.center.x,
            d.center.y,
            d.center.z,
            d.radius,
            d.outward.x,
            d.outward.y,
            d.outward.z,
            d.random,
        ]);
    }
    out
}

/// SH9, the three soft-box directions and radiances, the extinction
/// coefficients and the albedos — everything the shader reads once per pixel
/// and nothing it needs to tune.
fn studio_storage(sh: &[[f32; 3]; 9], lights: &[SoftBox; 3], args: &Args) -> Vec<f32> {
    let mut out = Vec::new();
    for c in sh.iter() {
        out.extend_from_slice(c);
    }
    for l in lights.iter() {
        out.extend_from_slice(&[l.dir.x, l.dir.y, l.dir.z]);
    }
    for l in lights.iter() {
        // What the analytic term wants is irradiance, not the radiance of the
        // panel: a 0.42-tangent box is a big piece of sky, a 0.17 strip is not.
        let solid = PI * l.tan * l.tan * l.stretch;
        out.extend_from_slice(&[
            l.radiance[0] * solid,
            l.radiance[1] * solid,
            l.radiance[2] * solid,
        ]);
    }
    // Extinction per centimetre, from the mean free path per channel.
    out.extend_from_slice(&[
        1.0 / SSS_RADIUS[0],
        1.0 / SSS_RADIUS[1],
        1.0 / SSS_RADIUS[2],
    ]);
    out.extend_from_slice(&ALBEDO_RIPE);
    out.extend_from_slice(&ALBEDO_BRIGHT);
    out.extend_from_slice(&ALBEDO_DARK);
    out.extend_from_slice(&ALBEDO_INNER);
    out.extend_from_slice(&FUZZ_COLOR);
    // The inner lamp, if it is on.
    let lamp = if args.lamp {
        [2.4f32, 0.55, 0.42]
    } else {
        [0.0, 0.0, 0.0]
    };
    out.extend_from_slice(&lamp);
    out
}

// ---------------------------------------------------------------------------
// The skin shader
// ---------------------------------------------------------------------------

const SKIN_FRAGMENT: &str = include_str!("raspberry_skin.wgsl");
