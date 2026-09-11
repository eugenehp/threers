//! Part of the `simcity` example; see `mod.rs`.
//!
//! Baked light transport, by raycasting.
//!
//! The rasteriser answers "is this point lit by the sun?" with a shadow map and
//! "how much ambient reaches it?" with a constant. The second answer is the
//! expensive one to be wrong about: it means the inside of a courtyard, the
//! strip of pavement under an awning and the middle of an open plaza all
//! receive exactly the same fill, and the city reads as flat however good the
//! direct lighting is.
//!
//! So compute the real answer once, offline, with rays. For every vertex of
//! the static city, integrate the incoming radiance over the hemisphere about
//! its normal: rays that escape collect sky, rays that hit collect what that
//! surface is re-emitting — its own albedo times its own sky access. That is
//! one bounce of global illumination, and it is where the colour comes from: a
//! wall opposite a brick facade picks up the brick.
//!
//! The result is folded into vertex colour, which the shader already
//! multiplies into base colour. That is the standard compromise and it is
//! worth naming: it darkens *direct* sun as well as ambient, so a sunlit
//! courtyard wall comes out slightly too dark. The alternative needs a second
//! vertex attribute and a shader change; against that, a courtyard that reads
//! as a courtyard is worth a few percent of error on the lit faces.
//!
//! None of this depends on the clock. Sky visibility is a property of the
//! geometry, so one bake serves sunrise, noon and midnight alike.
#![allow(dead_code)]

use super::*;
#[cfg(feature = "mesh-bvh")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "mesh-bvh")]
use threers::mesh_bvh::{BuildOptions, MeshBvh, SAH};
#[cfg(feature = "mesh-bvh")]
use threers::Ray;

#[derive(Default)]
pub(crate) struct BakeReport {
    pub(crate) verts: usize,
    /// Triangles in the occluder BVH.
    pub(crate) occluders: usize,
    /// Triangles after the receivers were subdivided.
    pub(crate) receivers: usize,
    pub(crate) rays: usize,
    pub(crate) build_millis: u128,
    pub(crate) millis: u128,
}

impl BakeReport {
    pub(crate) fn ran(&self) -> bool {
        self.rays > 0
    }

    pub(crate) fn summary(&self) -> String {
        if !self.ran() {
            return if cfg!(feature = "mesh-bvh") {
                "light bake: skipped".into()
            } else {
                "light bake: off (build with --features mesh-bvh)".into()
            };
        }
        format!(
            "light bake: {} rays over {} verts against {} occluders \
             ({} receiver tris) in {:.2}s ({:.2}s building the tree)",
            self.rays,
            self.verts,
            self.occluders,
            self.receivers,
            self.millis as f32 / 1000.0,
            self.build_millis as f32 / 1000.0
        )
    }
}

/// Without the crate's raycaster there is nothing to trace against, so the
/// city ships with flat ambient — the way it looked before this existed.
#[cfg(not(feature = "mesh-bvh"))]
pub(crate) fn bake_light(_b: &mut Batches, _strength: f32) -> BakeReport {
    BakeReport::default()
}

/// Longest edge a receiver triangle may keep.
///
/// Ground is built as one quad per block, and a quad has four vertices — so
/// per-vertex shading on it can only ever be a bilinear ramp between its
/// corners. A tower's shadow falling across the middle of a road cell has
/// nowhere to live. Subdividing is cheap here in a way it would not be in a
/// renderer that paid per triangle: this one pays per *draw*, and the whole
/// ground is one draw either way.
const MAX_EDGE: f32 = 3.4;

/// Rays per vertex in the gather pass.
const VERTEX_RAYS: usize = 40;
/// Rays per triangle in the pass that works out what each surface re-emits.
const TRI_RAYS: usize = 10;

/// Radiance of the sky in `dir`, up to a scale.
///
/// A deliberately neutral dome rather than the clock's sky: the bake has to
/// stay valid at every hour, so what it captures is the *shape* of the
/// occlusion, not one moment's colour. Slightly blue up, pale at the horizon,
/// dark below it — enough that a north wall and a floor differ.
#[cfg(feature = "mesh-bvh")]
fn sky(dir: Vector3) -> [f32; 3] {
    let up = dir.y.clamp(-1.0, 1.0);
    if up >= 0.0 {
        let t = up.powf(0.6);
        [
            mix(0.86, 0.55, t),
            mix(0.88, 0.66, t),
            mix(0.90, 0.92, t),
        ]
    } else {
        // Bounce off the ground plane beyond the city, which is what fills the
        // underside of an awning or a bridge soffit.
        let t = (-up).powf(0.5);
        [mix(0.86, 0.26, t), mix(0.88, 0.25, t), mix(0.90, 0.20, t)]
    }
}

/// Mean sky radiance over the upper hemisphere, cosine weighted. The reference
/// a fully open vertex is normalised against.
#[cfg(feature = "mesh-bvh")]
fn open_sky() -> [f32; 3] {
    let mut acc = [0.0f32; 3];
    const N: usize = 512;
    for i in 0..N {
        let (d, _) = cosine_dir(Vector3::new(0.0, 1.0, 0.0), i as u32, N as u32, 0);
        let s = sky(d);
        for c in 0..3 {
            acc[c] += s[c];
        }
    }
    [
        acc[0] / N as f32,
        acc[1] / N as f32,
        acc[2] / N as f32,
    ]
}

/// A cosine-weighted direction about `n`, from a stratified low-discrepancy
/// sequence offset by `seed`.
///
/// The seed is hashed from the vertex *position*, not from a counter. Two
/// vertices at the same place — and subdivision makes a great many of those —
/// then draw the same rays and bake to the same value, so shared edges stay
/// seamless instead of showing a crack of Monte Carlo noise down them.
#[cfg(feature = "mesh-bvh")]
fn cosine_dir(n: Vector3, i: u32, count: u32, seed: u32) -> (Vector3, f32) {
    // Van der Corput in base 2 for one axis, a stratified sweep for the other.
    let mut bits = i.wrapping_add(seed);
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    let u1 = bits as f32 * 2.328_306_4e-10;
    let u2 = (i as f32 + (seed & 0xffff) as f32 * 1.525_878_9e-5).fract() / count as f32
        + i as f32 / count as f32;
    let u2 = u2.fract();

    let r = u1.sqrt();
    let phi = TAU * u2;
    let (x, z) = (r * phi.cos(), r * phi.sin());
    let y = (1.0 - u1).max(0.0).sqrt();

    // Build a frame about n without a branch on its dominant axis.
    let sign = if n.y >= 0.0 { 1.0f32 } else { -1.0 };
    let a = -1.0 / (sign + n.y);
    let b = n.x * n.z * a;
    let t1 = Vector3::new(1.0 + sign * n.x * n.x * a, -sign * n.x, sign * b);
    let t2 = Vector3::new(b, -n.z, sign + n.z * n.z * a);
    (
        Vector3::new(
            t1.x * x + n.x * y + t2.x * z,
            t1.y * x + n.y * y + t2.y * z,
            t1.z * x + n.z * y + t2.z * z,
        )
        .normalize(),
        y,
    )
}

#[cfg(feature = "mesh-bvh")]
fn hash_pos(p: [f32; 3]) -> u32 {
    // Quantise to a millimetre so duplicated vertices agree exactly.
    let q = |v: f32| (v * 1000.0).round() as i32 as u32;
    let mut h = 2_166_136_261u32;
    for v in [q(p[0]), q(p[1]), q(p[2])] {
        h ^= v;
        h = h.wrapping_mul(16_777_619);
    }
    h
}

/// Split every triangle whose longest edge exceeds `max_edge`, repeatedly.
///
/// Vertices are emitted per triangle rather than shared. That duplicates a
/// great many of them, which costs memory and nothing else — and position-
/// hashed sampling means the duplicates bake identically, so there are no
/// seams to pay for it with.
#[cfg(feature = "mesh-bvh")]
fn tessellate(mb: &mut MeshBuilder, max_edge: f32) {
    if mb.idx.is_empty() {
        return;
    }
    let vert = |i: u32| -> ([f32; 3], [f32; 3], [f32; 2], [f32; 3]) {
        let i = i as usize;
        (
            [mb.pos[i * 3], mb.pos[i * 3 + 1], mb.pos[i * 3 + 2]],
            [mb.nrm[i * 3], mb.nrm[i * 3 + 1], mb.nrm[i * 3 + 2]],
            [mb.uv[i * 2], mb.uv[i * 2 + 1]],
            [mb.col[i * 3], mb.col[i * 3 + 1], mb.col[i * 3 + 2]],
        )
    };
    type V = ([f32; 3], [f32; 3], [f32; 2], [f32; 3]);
    let mid = |a: &V, b: &V| -> V {
        let m3 = |x: [f32; 3], y: [f32; 3]| {
            [
                (x[0] + y[0]) * 0.5,
                (x[1] + y[1]) * 0.5,
                (x[2] + y[2]) * 0.5,
            ]
        };
        (
            m3(a.0, b.0),
            m3(a.1, b.1),
            [(a.2[0] + b.2[0]) * 0.5, (a.2[1] + b.2[1]) * 0.5],
            m3(a.3, b.3),
        )
    };
    let long = |a: &V, b: &V| -> f32 {
        let d = [a.0[0] - b.0[0], a.0[1] - b.0[1], a.0[2] - b.0[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };

    let mut work: Vec<[V; 3]> = mb
        .idx
        .chunks_exact(3)
        .map(|t| [vert(t[0]), vert(t[1]), vert(t[2])])
        .collect();
    let mut out: Vec<[V; 3]> = Vec::with_capacity(work.len() * 2);
    // Bounded: each pass halves the longest edge, and a city block is not
    // large enough for this to run away.
    for _ in 0..7 {
        let mut split_any = false;
        out.clear();
        for t in work.drain(..) {
            let e = [
                long(&t[0], &t[1]),
                long(&t[1], &t[2]),
                long(&t[2], &t[0]),
            ];
            let (k, &m) = e
                .iter()
                .enumerate()
                .max_by(|a, c| a.1.partial_cmp(c.1).unwrap())
                .unwrap();
            if m <= max_edge {
                out.push(t);
                continue;
            }
            split_any = true;
            // Split the longest edge; the opposite vertex joins the midpoint.
            let (a, b, c) = match k {
                0 => (t[0], t[1], t[2]),
                1 => (t[1], t[2], t[0]),
                _ => (t[2], t[0], t[1]),
            };
            let m = mid(&a, &b);
            out.push([a, m, c]);
            out.push([m, b, c]);
        }
        std::mem::swap(&mut work, &mut out);
        if !split_any {
            break;
        }
    }

    mb.pos.clear();
    mb.nrm.clear();
    mb.uv.clear();
    mb.col.clear();
    mb.idx.clear();
    mb.pos.reserve(work.len() * 9);
    for t in &work {
        let base = (mb.pos.len() / 3) as u32;
        for v in t {
            mb.pos.extend_from_slice(&v.0);
            mb.nrm.extend_from_slice(&v.1);
            mb.uv.extend_from_slice(&v.2);
            mb.col.extend_from_slice(&v.3);
        }
        mb.idx.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

#[cfg(feature = "mesh-bvh")]
fn tri_list(mb: &MeshBuilder, tris: &mut Vec<[Vector3; 3]>, albedo: &mut Vec<[f32; 3]>) {
    for t in mb.idx.chunks_exact(3) {
        let p = |i: u32| {
            let i = i as usize;
            Vector3::new(mb.pos[i * 3], mb.pos[i * 3 + 1], mb.pos[i * 3 + 2])
        };
        tris.push([p(t[0]), p(t[1]), p(t[2])]);
        let c = |i: u32| {
            let i = i as usize;
            [mb.col[i * 3], mb.col[i * 3 + 1], mb.col[i * 3 + 2]]
        };
        let (a, b, d) = (c(t[0]), c(t[1]), c(t[2]));
        albedo.push([
            (a[0] + b[0] + d[0]) / 3.0,
            (a[1] + b[1] + d[1]) / 3.0,
            (a[2] + b[2] + d[2]) / 3.0,
        ]);
    }
}

/// Bake sky access and one bounce into the static city's vertex colours.
///
/// `strength` is how much of the result to apply, 0 to 1 — it exists so the
/// effect can be dialled back without recomputing anything.
#[cfg(feature = "mesh-bvh")]
pub(crate) fn bake_light(b: &mut Batches, strength: f32) -> BakeReport {
    let t0 = std::time::Instant::now();

    // --- Occluders. Everything solid, before subdivision: splitting a
    // triangle does not move the surface, so the smaller list traces faster
    // for exactly the same answer.
    let mut tris: Vec<[Vector3; 3]> = Vec::new();
    let mut albedo: Vec<[f32; 3]> = Vec::new();
    for mb in b.facades.iter() {
        tri_list(mb, &mut tris, &mut albedo);
    }
    // Note what is NOT here: `litter`. Tens of thousands of leaf quads lying
    // flat on the ground occlude nothing and doubled the time this spent
    // tracing. They are receivers below, not occluders.
    for mb in [&b.trim, &b.foliage, &b.pads, &b.road, &b.grass] {
        tri_list(mb, &mut tris, &mut albedo);
    }
    if tris.is_empty() {
        return BakeReport::default();
    }
    // The crate's own raycaster, over one geometry holding every occluder.
    // `SAH` costs more to build and pays it back immediately: this is millions
    // of rays against half a million triangles.
    let mut geom = BufferGeometry::new();
    let mut flat = Vec::with_capacity(tris.len() * 9);
    for t in &tris {
        for v in t {
            flat.extend_from_slice(&[v.x, v.y, v.z]);
        }
    }
    geom.set_attribute("position", BufferAttribute::new(flat, 3));
    geom.set_index((0..tris.len() as u32 * 3).collect());
    let t_build = std::time::Instant::now();
    let Some(bvh) = MeshBvh::build(
        &geom,
        BuildOptions {
            strategy: SAH,
            max_leaf_tris: 8,
            ..Default::default()
        },
    ) else {
        return BakeReport {
            verts: 0,
            occluders: tris.len(),
            receivers: 0,
            rays: 0,
            build_millis: t_build.elapsed().as_millis(),
            millis: t0.elapsed().as_millis(),
        };
    };
    let build_millis = t_build.elapsed().as_millis();
    let open = open_sky();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // --- Pass one: what each surface re-emits. A coarse sky-access term at
    // each triangle's centroid, times its own albedo. This is the only reason
    // the bounce carries colour rather than being a grey lift.
    let mut emit = vec![[0.0f32; 3]; tris.len()];
    {
        let chunk = tris.len().div_ceil(threads);
        let slices: Vec<&mut [[f32; 3]]> = emit.chunks_mut(chunk).collect();
        std::thread::scope(|s| {
            for (ci, out) in slices.into_iter().enumerate() {
                let (bvh, tris, albedo) = (&bvh, &tris, &albedo);
                s.spawn(move || {
                    for (k, e) in out.iter_mut().enumerate() {
                        let ti = ci * chunk + k;
                        let t = tris[ti];
                        let c = Vector3::new(
                            (t[0].x + t[1].x + t[2].x) / 3.0,
                            (t[0].y + t[1].y + t[2].y) / 3.0,
                            (t[0].z + t[1].z + t[2].z) / 3.0,
                        );
                        let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize();
                        let o = c + n * 0.02;
                        let seed = hash_pos([o.x, o.y, o.z]);
                        let mut vis = 0.0f32;
                        for i in 0..TRI_RAYS {
                            let (d, _) = cosine_dir(n, i as u32, TRI_RAYS as u32, seed);
                            let ray = Ray {
                                origin: o,
                                direction: d,
                            };
                            if bvh.raycast_first(&ray, 1e-3, 1.0e4, false).is_none() {
                                vis += 1.0;
                            }
                        }
                        let vis = vis / TRI_RAYS as f32;
                        for ch in 0..3 {
                            e[ch] = albedo[ti][ch] * vis * open[ch];
                        }
                    }
                });
            }
        });
    }

    // --- Pass two: gather at every vertex of every receiver.
    let occluders = tris.len();
    let rays = AtomicUsize::new(0);
    let mut verts = 0usize;
    let mut receivers = 0usize;

    let mut receiver_mbs: Vec<&mut MeshBuilder> = Vec::new();
    {
        let Batches {
            facades,
            trim,
            foliage,
            pads,
            road,
            grass,
            paint,
            puddle,
            litter,
            ..
        } = b;
        for mb in facades.iter_mut() {
            receiver_mbs.push(mb);
        }
        receiver_mbs.extend([trim, foliage, pads, road, grass, paint, puddle, litter]);
    }

    for mb in receiver_mbs {
        if mb.idx.is_empty() {
            continue;
        }
        tessellate(mb, MAX_EDGE);
        receivers += mb.idx.len() / 3;
        let n_verts = mb.pos.len() / 3;
        verts += n_verts;
        let pos = std::mem::take(&mut mb.pos);
        let nrm = std::mem::take(&mut mb.nrm);
        let mut col = std::mem::take(&mut mb.col);

        // Subdivision emits vertices per triangle, so a shared corner appears
        // six times over and the gather would run six times for an identical
        // answer. Key on the quantised position *and* normal — same place,
        // different facing, genuinely different result — and trace each
        // distinct one once. On an eight-block city that is 1.4M vertices
        // collapsing to a few hundred thousand.
        let mut slot: Vec<u32> = Vec::with_capacity(n_verts);
        let mut uniq: Vec<([f32; 3], Vector3)> = Vec::new();
        let mut seen: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::with_capacity(n_verts / 3);
        for vi in 0..n_verts {
            let p = [pos[vi * 3], pos[vi * 3 + 1], pos[vi * 3 + 2]];
            let n = Vector3::new(nrm[vi * 3], nrm[vi * 3 + 1], nrm[vi * 3 + 2]);
            let n = if n.length() < 1e-4 {
                Vector3::new(0.0, 1.0, 0.0)
            } else {
                n.normalize()
            };
            let key = (
                hash_pos(p),
                hash_pos([n.x * 100.0, n.y * 100.0, n.z * 100.0]),
            );
            let id = *seen.entry(key).or_insert_with(|| {
                uniq.push((p, n));
                (uniq.len() - 1) as u32
            });
            slot.push(id);
        }

        let mut mult = vec![[1.0f32; 3]; uniq.len()];
        let chunk = uniq.len().div_ceil(threads).max(1);
        {
            let slices: Vec<&mut [[f32; 3]]> = mult.chunks_mut(chunk).collect();
            std::thread::scope(|s| {
                for (ci, out) in slices.into_iter().enumerate() {
                    let (bvh, emit, uniq, rays) = (&bvh, &emit, &uniq, &rays);
                    s.spawn(move || {
                        let mut local = 0usize;
                        for (k, m_out) in out.iter_mut().enumerate() {
                            let (p, n) = uniq[ci * chunk + k];
                            let o = Vector3::new(p[0], p[1], p[2]) + n * 0.035;
                            let seed = hash_pos(p);
                            let mut acc = [0.0f32; 3];
                            // The same rays with nothing in the way. Dividing
                            // by this rather than by a fixed constant is what
                            // makes the result pure occlusion: a wall in an
                            // open street comes out at 1.0 like the roof above
                            // it, instead of at the 0.55 a vertical surface
                            // scores against an up-facing reference — which
                            // darkened every facade in the city by half.
                            let mut reference = [0.0f32; 3];
                            for i in 0..VERTEX_RAYS {
                                let (d, _) = cosine_dir(n, i as u32, VERTEX_RAYS as u32, seed);
                                let sk = sky(d);
                                for ch in 0..3 {
                                    reference[ch] += sk[ch];
                                }
                                let ray = Ray {
                                    origin: o,
                                    direction: d,
                                };
                                match bvh.raycast_first(&ray, 1e-3, 1.0e4, false) {
                                    None => {
                                        for ch in 0..3 {
                                            acc[ch] += sk[ch];
                                        }
                                    }
                                    Some(h) => {
                                        let e = emit[h.face_index];
                                        for ch in 0..3 {
                                            acc[ch] += e[ch];
                                        }
                                    }
                                }
                            }
                            local += VERTEX_RAYS;
                            for ch in 0..3 {
                                let m = (acc[ch] / reference[ch].max(1e-4)).clamp(0.0, 1.15);
                                // Deepen the shading without touching what is
                                // already open: an exponent leaves 1.0 at 1.0
                                // and pulls everything below it down. One
                                // bounce under-darkens crevices anyway, so this
                                // is buying back some of what the truncated
                                // series lost rather than inventing contrast.
                                let m = m.powf(1.7);
                                // Never fully black: a closed corner still sees
                                // the light that got there by paths this single
                                // bounce does not carry.
                                m_out[ch] = mix(1.0, m.max(0.22), strength);
                            }
                        }
                        rays.fetch_add(local, Ordering::Relaxed);
                    });
                }
            });
        }
        for (vi, s) in slot.iter().enumerate() {
            let m = mult[*s as usize];
            for ch in 0..3 {
                col[vi * 3 + ch] *= m[ch];
            }
        }
        mb.pos = pos;
        mb.nrm = nrm;
        mb.col = col;
    }

    BakeReport {
        verts,
        occluders,
        receivers,
        rays: rays.load(Ordering::Relaxed),
        build_millis,
        millis: t0.elapsed().as_millis(),
    }
}
