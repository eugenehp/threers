//! Browser-side Cornell box: first-hit G-buffer + a cheap multi-bounce probe.
//!
//! Matches the hostile diffuse Cornell layout in [`crate::generate`] so the
//! trained hops/NRC weights see the same room. No `threers` dependency — the
//! demo wasm must stay small and wasm32-safe.

use crate::pack::{self, PlaneExtras};

const ROOM: f32 = 2.0;
const HALF: f32 = ROOM * 0.5;

#[derive(Clone, Copy)]
struct V3 {
    x: f32,
    y: f32,
    z: f32,
}

impl V3 {
    const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
    fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    fn norm(self) -> Self {
        let l = self.len().max(1e-8);
        self.mul(1.0 / l)
    }
    fn hadamard(self, o: Self) -> Self {
        Self::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }
}

#[derive(Clone, Copy)]
struct Tri {
    a: V3,
    b: V3,
    c: V3,
    n: V3,
    albedo: V3,
    emissive: V3,
}

fn quad(p0: V3, p1: V3, p2: V3, p3: V3, albedo: V3, emissive: V3, out: &mut Vec<Tri>) {
    let n = (p1.sub(p0)).cross(p2.sub(p0)).norm();
    out.push(Tri {
        a: p0,
        b: p1,
        c: p2,
        n,
        albedo,
        emissive,
    });
    out.push(Tri {
        a: p0,
        b: p2,
        c: p3,
        n,
        albedo,
        emissive,
    });
}

fn box_tris(center: V3, hx: f32, hy: f32, hz: f32, yaw: f32, albedo: V3, out: &mut Vec<Tri>) {
    let (sy, cy) = yaw.sin_cos();
    let rot = |v: V3| V3::new(v.x * cy + v.z * sy, v.y, -v.x * sy + v.z * cy);
    let corners = [
        V3::new(-hx, -hy, -hz),
        V3::new(hx, -hy, -hz),
        V3::new(hx, -hy, hz),
        V3::new(-hx, -hy, hz),
        V3::new(-hx, hy, -hz),
        V3::new(hx, hy, -hz),
        V3::new(hx, hy, hz),
        V3::new(-hx, hy, hz),
    ]
    .map(|c| rot(c).add(center));
    let faces: [[usize; 4]; 6] = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [3, 2, 6, 7],
        [0, 3, 7, 4],
        [1, 5, 6, 2],
    ];
    for f in faces {
        quad(
            corners[f[0]],
            corners[f[1]],
            corners[f[2]],
            corners[f[3]],
            albedo,
            V3::new(0.0, 0.0, 0.0),
            out,
        );
    }
}

fn scene() -> Vec<Tri> {
    let mut t = Vec::with_capacity(40);
    let white = V3::new(0.73, 0.73, 0.73);
    let red = V3::new(0.65, 0.05, 0.05);
    let green = V3::new(0.12, 0.45, 0.15);
    let black = V3::new(0.0, 0.0, 0.0);
    let grey = V3::new(0.75, 0.75, 0.72);

    // Floor (+Y up).
    quad(
        V3::new(-HALF, 0.0, HALF),
        V3::new(HALF, 0.0, HALF),
        V3::new(HALF, 0.0, -HALF),
        V3::new(-HALF, 0.0, -HALF),
        white,
        black,
        &mut t,
    );
    // Ceiling with a hole for the lamp (avoids shadow rays dying on the roof).
    let ls = 0.3;
    let c = ROOM;
    // four strips: -Z, +Z, -X, +X
    quad(
        V3::new(-HALF, c, -HALF),
        V3::new(HALF, c, -HALF),
        V3::new(HALF, c, -ls),
        V3::new(-HALF, c, -ls),
        white,
        black,
        &mut t,
    );
    quad(
        V3::new(-HALF, c, ls),
        V3::new(HALF, c, ls),
        V3::new(HALF, c, HALF),
        V3::new(-HALF, c, HALF),
        white,
        black,
        &mut t,
    );
    quad(
        V3::new(-HALF, c, -ls),
        V3::new(-ls, c, -ls),
        V3::new(-ls, c, ls),
        V3::new(-HALF, c, ls),
        white,
        black,
        &mut t,
    );
    quad(
        V3::new(ls, c, -ls),
        V3::new(HALF, c, -ls),
        V3::new(HALF, c, ls),
        V3::new(ls, c, ls),
        white,
        black,
        &mut t,
    );
    // Lamp (faces −Y).
    let lamp_e = V3::new(1.0, 0.92, 0.78).mul(22.0);
    quad(
        V3::new(-ls, c - 0.001, -ls),
        V3::new(ls, c - 0.001, -ls),
        V3::new(ls, c - 0.001, ls),
        V3::new(-ls, c - 0.001, ls),
        black,
        lamp_e,
        &mut t,
    );
    // Back (-Z).
    quad(
        V3::new(-HALF, 0.0, -HALF),
        V3::new(HALF, 0.0, -HALF),
        V3::new(HALF, ROOM, -HALF),
        V3::new(-HALF, ROOM, -HALF),
        white,
        black,
        &mut t,
    );
    // Left (-X) red.
    quad(
        V3::new(-HALF, 0.0, HALF),
        V3::new(-HALF, 0.0, -HALF),
        V3::new(-HALF, ROOM, -HALF),
        V3::new(-HALF, ROOM, HALF),
        red,
        black,
        &mut t,
    );
    // Right (+X) green.
    quad(
        V3::new(HALF, 0.0, -HALF),
        V3::new(HALF, 0.0, HALF),
        V3::new(HALF, ROOM, HALF),
        V3::new(HALF, ROOM, -HALF),
        green,
        black,
        &mut t,
    );
    // Lamp already added with the ceiling hole.
    box_tris(
        V3::new(-0.35, 0.6, -0.3),
        0.3,
        0.6,
        0.3,
        0.3,
        grey,
        &mut t,
    );
    box_tris(
        V3::new(0.35, 0.3, 0.35),
        0.3,
        0.3,
        0.3,
        -0.3,
        grey,
        &mut t,
    );
    t
}

struct Hit {
    t: f32,
    p: V3,
    n: V3,
    albedo: V3,
    emissive: V3,
}

fn intersect(origin: V3, dir: V3, tris: &[Tri]) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    for tri in tris {
        let e1 = tri.b.sub(tri.a);
        let e2 = tri.c.sub(tri.a);
        let pvec = dir.cross(e2);
        let det = e1.dot(pvec);
        if det.abs() < 1e-8 {
            continue;
        }
        let inv = 1.0 / det;
        let tvec = origin.sub(tri.a);
        let u = tvec.dot(pvec) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let qvec = tvec.cross(e1);
        let v = dir.dot(qvec) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = e2.dot(qvec) * inv;
        if t <= 1e-4 {
            continue;
        }
        if best.as_ref().map(|h| t < h.t).unwrap_or(true) {
            let mut n = tri.n;
            if n.dot(dir) > 0.0 {
                n = n.mul(-1.0);
            }
            best = Some(Hit {
                t,
                p: origin.add(dir.mul(t)),
                n,
                albedo: tri.albedo,
                emissive: tri.emissive,
            });
        }
    }
    best
}

struct Rng(u32);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / 16_777_216.0
    }
}

fn cosine_hemisphere(n: V3, rng: &mut Rng) -> V3 {
    let r1 = rng.next_f32();
    let r2 = rng.next_f32();
    let phi = std::f32::consts::TAU * r1;
    let r = r2.sqrt();
    let x = phi.cos() * r;
    let y = phi.sin() * r;
    let z = (1.0 - r2).sqrt();
    let w = n;
    let a = if w.x.abs() > 0.1 {
        V3::new(0.0, 1.0, 0.0)
    } else {
        V3::new(1.0, 0.0, 0.0)
    };
    let u = a.cross(w).norm();
    let v = w.cross(u);
    u.mul(x).add(v.mul(y)).add(w.mul(z)).norm()
}

#[allow(clippy::too_many_arguments)]
fn camera_ray(
    origin: V3,
    target: V3,
    fov_deg: f32,
    side: usize,
    x: usize,
    y: usize,
    jx: f32,
    jy: f32,
) -> V3 {
    let forward = target.sub(origin).norm();
    let right = forward.cross(V3::new(0.0, 1.0, 0.0)).norm();
    let up = right.cross(forward);
    let half = (fov_deg.to_radians() * 0.5).tan();
    let u = ((x as f32 + jx) / side as f32 * 2.0 - 1.0) * half;
    let v = (1.0 - (y as f32 + jy) / side as f32 * 2.0) * half;
    right.mul(u).add(up.mul(v)).add(forward).norm()
}

fn trace_probe(origin: V3, dir: V3, tris: &[Tri], max_bounces: u32, rng: &mut Rng) -> V3 {
    let mut o = origin;
    let mut d = dir;
    let mut throughput = V3::new(1.0, 1.0, 1.0);
    let mut colour = V3::new(0.0, 0.0, 0.0);
    let lamp_c = V3::new(0.0, ROOM - 0.01, 0.0);
    let lamp_extent = 0.28;
    for bounce in 0..max_bounces {
        let Some(hit) = intersect(o, d, tris) else {
            break;
        };
        colour = colour.add(throughput.hadamard(hit.emissive));
        if bounce + 1 == max_bounces {
            break;
        }
        // Next-event: sample a point on the ceiling lamp.
        let lx = (rng.next_f32() * 2.0 - 1.0) * lamp_extent;
        let lz = (rng.next_f32() * 2.0 - 1.0) * lamp_extent;
        let lp = V3::new(lx, lamp_c.y, lz);
        let to_l = lp.sub(hit.p);
        let dist2 = to_l.dot(to_l).max(1e-6);
        let ldir = to_l.mul(1.0 / dist2.sqrt());
        let cos_surf = hit.n.dot(ldir).max(0.0);
        // Lamp normal is −Y; incoming to the lamp is −ldir, so cos = (−Y)·(−ldir) = ldir.y.
        let cos_lamp = ldir.y.max(0.0);
        if cos_surf > 0.0 && cos_lamp > 0.0 {
            let shadow_o = hit.p.add(hit.n.mul(1e-4));
            let blocked = match intersect(shadow_o, ldir, tris) {
                Some(h) if h.emissive.x + h.emissive.y + h.emissive.z > 1e-3 => false,
                Some(h) if h.t * h.t < dist2 * 0.98 => true,
                _ => false,
            };
            if !blocked {
                let area = (2.0 * lamp_extent).powi(2);
                let lamp_e = V3::new(1.0, 0.92, 0.78).mul(22.0);
                let pdf = dist2 / (area * cos_lamp).max(1e-6);
                let contrib = throughput
                    .hadamard(hit.albedo)
                    .hadamard(lamp_e)
                    .mul(cos_surf / (std::f32::consts::PI * pdf).max(1e-6));
                colour = colour.add(contrib);
            }
        }
        throughput = throughput.hadamard(hit.albedo);
        let p = throughput.x.max(throughput.y).max(throughput.z);
        if bounce > 1 && rng.next_f32() > p {
            break;
        }
        if bounce > 1 && p > 1e-4 {
            throughput = throughput.mul(1.0 / p);
        }
        o = hit.p.add(hit.n.mul(1e-4));
        d = cosine_hemisphere(hit.n, rng);
    }
    colour
}

/// Pack 22 input planes for one camera view of the canonical Cornell box.
pub fn pack_view(
    side: usize,
    cam_pos: [f32; 3],
    cam_target: [f32; 3],
    fov_deg: f32,
    spp: u32,
    bounces: u32,
) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(side >= 8 && side.is_multiple_of(8), "side must be a multiple of 8");
    let tris = scene();
    let origin = V3::new(cam_pos[0], cam_pos[1], cam_pos[2]);
    let target = V3::new(cam_target[0], cam_target[1], cam_target[2]);
    let n = side * side;
    let mut probe = vec![0.0f32; n * 4];
    let mut albedo = vec![[0.0f32; 3]; n];
    let mut normal = vec![[0.0f32; 3]; n];
    let mut depth = vec![f32::INFINITY; n];
    let mut world = vec![[f32::NAN; 3]; n];

    let spp = spp.max(1);
    let bounces = bounces.max(1);
    for y in 0..side {
        for x in 0..side {
            let i = y * side + x;
            let mut sum = V3::new(0.0, 0.0, 0.0);
            for s in 0..spp {
                let jx = if spp == 1 {
                    0.5
                } else {
                    // Per-pixel, per-sample jitter — independent of batch order.
                    let tag = ((x as u32 * 1973) ^ (y as u32 * 9277) ^ 0x00C0_FFEE) | 1;
                    let mut j = Rng(tag.wrapping_add(s.wrapping_mul(0x26699)));
                    j.next_f32()
                };
                let jy = if spp == 1 {
                    0.5
                } else {
                    let tag = ((x as u32 * 1973) ^ (y as u32 * 9277) ^ 0x00C0_FFEE) | 1;
                    let mut j = Rng(tag.wrapping_add(s.wrapping_mul(0x26699)).wrapping_add(1));
                    j.next_f32()
                };
                let dir = camera_ray(origin, target, fov_deg, side, x, y, jx, jy);
                let mut sample_rng =
                    Rng(((x as u32 * 1973) ^ (y as u32 * 9277) ^ 0x00C0_FFEE ^ s.wrapping_mul(0x9E37_79B9)) | 1);
                sum = sum.add(trace_probe(origin, dir, &tris, bounces, &mut sample_rng));
            }
            let inv = 1.0 / spp as f32;
            let c = sum.mul(inv);
            probe[i * 4] = c.x;
            probe[i * 4 + 1] = c.y;
            probe[i * 4 + 2] = c.z;
            probe[i * 4 + 3] = 1.0;

            let dir = camera_ray(origin, target, fov_deg, side, x, y, 0.5, 0.5);
            if let Some(hit) = intersect(origin, dir, &tris) {
                albedo[i] = [hit.albedo.x, hit.albedo.y, hit.albedo.z];
                normal[i] = [hit.n.x, hit.n.y, hit.n.z];
                depth[i] = hit.t;
                world[i] = [hit.p.x, hit.p.y, hit.p.z];
            }
        }
    }

    let scale = ROOM;
    pack::planes_input(
        side,
        side,
        &probe,
        &albedo,
        &normal,
        &depth,
        scale,
        Some(PlaneExtras { world: &world }),
    )
}

/// Default orbit camera: spherical coords around the room centre.
pub fn orbit_camera(yaw: f32, pitch: f32, distance: f32) -> ([f32; 3], [f32; 3]) {
    let target = [0.0, 1.0, 0.0];
    let pitch = pitch.clamp(-1.2, 1.2);
    let cy = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let dist = distance.clamp(1.6, 6.0);
    let pos = [
        target[0] + sy * cp * dist,
        target[1] + sp * dist,
        target[2] + cy * cp * dist,
    ];
    (pos, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IN_CHANNELS;

    #[test]
    fn packs_canonical_view() {
        let tris = scene();
        let origin = V3::new(0.0, 1.0, 3.9);
        let target = V3::new(0.0, 1.0, 0.0);
        let side = 16usize;
        let mut hits = 0usize;
        for y in 0..side {
            for x in 0..side {
                let dir = camera_ray(origin, target, 40.0, side, x, y, 0.5, 0.5);
                if intersect(origin, dir, &tris).is_some() {
                    hits += 1;
                }
            }
        }
        assert!(hits > 100, "first-hit coverage too low: {hits}/256");

        let input = pack_view(16, [0.0, 1.0, 3.9], [0.0, 1.0, 0.0], 40.0, 2, 3).expect("pack");
        assert_eq!(input.len(), IN_CHANNELS * 16 * 16);
        let lit: usize = input[..16 * 16]
            .iter()
            .filter(|&&v| v > 0.01)
            .count();
        assert!(lit > 40, "probe should light the room, lit={lit} hits={hits}");
    }
}
