//! three.js's `RoomEnvironment`, as an HDR cube — the neutral studio light
//! most three.js scenes are lit with (`pmrem.fromScene(new RoomEnvironment())`).
//!
//! three.js builds it as a scene — a room, six boxes, a point light, six
//! emissive panels — and rasterises that into a cube with a `CubeCamera`. No
//! rasteriser is needed to reproduce it: every surface is Lambertian, nothing
//! casts a shadow, and the camera sits at the origin, so each texel is one ray
//! cast against thirteen boxes and the nearest hit shaded with three.js's own
//! point-light falloff. The result is the cube three.js renders, texel for
//! texel, without a GPU and without depending on this crate's renderer.
//!
//! ```no_run
//! let mut scene = threers::Scene::new();
//! scene.environment = Some(threers::extras::RoomEnvironment::pmrem(256));
//! scene.environment_intensity = 0.5;
//! ```
//!
//! Values are linear and unbounded (the ceiling panel is 100), as three.js's
//! half-float capture holds them — so the cube goes through
//! [`CubeTexture::new_f32`], whose faces survive prefiltering unclamped. An
//! 8-bit cube would flatten every panel to the walls' white and take the
//! directionality of the light with it.

use std::f32::consts::PI;
use std::sync::Arc;

use super::PmremGenerator;
use crate::math::Vector3;
use crate::textures::CubeTexture;

/// `RoomEnvironment` sets `position.y = -3.5` on itself: everything moves down.
const DROP: f32 = 3.5;

/// The point light: `PointLight(0xffffff, 900, 28, 2)`.
const LIGHT_POS: [f32; 3] = [0.418, 16.199 - DROP, 0.300];
const LIGHT_INTENSITY: f32 = 900.0;
const LIGHT_CUTOFF: f32 = 28.0;

/// A unit box scaled, rotated about Y and placed — three.js's `BoxGeometry()`
/// under an `Object3D` transform.
struct Box3 {
    centre: [f32; 3],
    half: [f32; 3],
    /// rotation about Y, radians
    yaw: f32,
    /// `None` for the white Lambertian walls and boxes; `Some(e)` for an
    /// emissive panel (`color: 0x000000, emissive: 0xffffff, emissiveIntensity: e`).
    emissive: Option<f32>,
}

const fn bx(p: [f32; 3], s: [f32; 3], yaw: f32, emissive: Option<f32>) -> Box3 {
    Box3 {
        centre: [p[0], p[1] - DROP, p[2]],
        half: [s[0] * 0.5, s[1] * 0.5, s[2] * 0.5],
        yaw,
        emissive,
    }
}

/// The room, seen from inside (`side: BackSide`).
const ROOM: Box3 = bx([-0.757, 13.219, 0.717], [31.713, 28.305, 28.591], 0.0, None);

/// The six boxes, then the six light panels, as `RoomEnvironment` lists them.
const SOLIDS: [Box3; 12] = [
    bx([-10.906, 2.009, 1.846], [2.328, 7.905, 4.651], -0.195, None),
    bx([-5.607, -0.754, -0.758], [1.970, 1.534, 3.955], 0.994, None),
    bx([6.167, 0.857, 7.803], [3.927, 6.285, 3.687], 0.561, None),
    bx([-2.017, 0.018, 6.124], [2.002, 4.566, 2.064], 0.333, None),
    bx([2.291, -0.756, -2.621], [1.546, 1.552, 1.496], -0.286, None),
    bx([-2.193, -0.369, -5.547], [3.875, 3.487, 2.986], 0.516, None),
    // -x right, -x left, +x, +z, -z, +y
    bx([-16.116, 14.37, 8.208], [0.1, 2.428, 2.739], 0.0, Some(50.0)),
    bx([-16.109, 18.021, -8.207], [0.1, 2.425, 2.751], 0.0, Some(50.0)),
    bx([14.904, 12.198, -1.832], [0.15, 4.265, 6.331], 0.0, Some(17.0)),
    bx([-0.462, 8.89, 14.520], [4.38, 5.441, 0.088], 0.0, Some(43.0)),
    bx([3.235, 11.486, -12.541], [2.5, 2.0, 0.1], 0.0, Some(20.0)),
    bx([0.0, 20.0, 0.0], [1.0, 0.1, 1.0], 0.0, Some(100.0)),
];

impl Box3 {
    /// Ray `o + t·d` against this box: `(t_enter, t_exit, normal at enter,
    /// normal at exit)`, normals in world space and pointing out of the box.
    fn slab(&self, o: [f32; 3], d: [f32; 3]) -> Option<(f32, f32, [f32; 3], [f32; 3])> {
        // into the box's frame: undo the translation, then the yaw
        let (s, c) = self.yaw.sin_cos();
        let rel = [o[0] - self.centre[0], o[1] - self.centre[1], o[2] - self.centre[2]];
        // R_y(θ)ᵀ applied to (x, z)
        let lo = [c * rel[0] - s * rel[2], rel[1], s * rel[0] + c * rel[2]];
        let ld = [c * d[0] - s * d[2], d[1], s * d[0] + c * d[2]];
        let (mut t0, mut t1) = (f32::NEG_INFINITY, f32::INFINITY);
        let (mut n0, mut n1) = ([0.0f32; 3], [0.0f32; 3]);
        for a in 0..3 {
            if ld[a].abs() < 1e-12 {
                if lo[a].abs() > self.half[a] {
                    return None;
                }
                continue;
            }
            let inv = 1.0 / ld[a];
            let mut near = (-self.half[a] - lo[a]) * inv;
            let mut far = (self.half[a] - lo[a]) * inv;
            // the face crossed first faces against the ray
            let mut nn = [0.0f32; 3];
            nn[a] = -ld[a].signum();
            let mut nf = [0.0f32; 3];
            nf[a] = ld[a].signum();
            if near > far {
                std::mem::swap(&mut near, &mut far);
            }
            if near > t0 {
                t0 = near;
                n0 = nn;
            }
            if far < t1 {
                t1 = far;
                n1 = nf;
            }
            if t0 > t1 {
                return None;
            }
        }
        // back to world: R_y(θ)
        let world = |n: [f32; 3]| [c * n[0] + s * n[2], n[1], -s * n[0] + c * n[2]];
        Some((t0, t1, world(n0), world(n1)))
    }
}

/// three.js's `getDistanceAttenuation` for a point light with a cutoff.
fn falloff(dist: f32) -> f32 {
    let base = 1.0 / dist.powi(2).max(0.01);
    let w = (1.0 - (dist / LIGHT_CUTOFF).powi(4)).clamp(0.0, 1.0);
    base * w * w
}

/// three.js's `RoomEnvironment`: the neutral studio light most three.js
/// scenes are lit with (`pmrem.fromScene(new RoomEnvironment())`), as an HDR
/// cube.
pub struct RoomEnvironment;

impl RoomEnvironment {
    /// Linear radiance seen from the origin along `dir` (any length).
    pub fn radiance(dir: Vector3) -> [f32; 3] {
        let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt().max(1e-12);
        let d = [dir.x / len, dir.y / len, dir.z / len];
        let o = [0.0f32; 3];

        // The room is seen from inside: the hit is where the ray leaves it,
        // and the lit side faces back in.
        let Some((_, t_room, _, n_out)) = ROOM.slab(o, d) else {
            return [0.0; 3];
        };
        let mut t = t_room;
        let mut normal = [-n_out[0], -n_out[1], -n_out[2]];
        let mut emissive = None;
        for b in &SOLIDS {
            if let Some((t0, _, n0, _)) = b.slab(o, d) {
                // the camera is outside every solid, so t0 > 0 is a front face
                if t0 > 0.0 && t0 < t {
                    t = t0;
                    normal = n0;
                    emissive = b.emissive;
                }
            }
        }
        if let Some(e) = emissive {
            return [e; 3];
        }
        // white Lambertian (MeshStandardMaterial defaults: roughness 1,
        // metalness 0) under the one point light, unshadowed
        let p = [d[0] * t, d[1] * t, d[2] * t];
        let l = [LIGHT_POS[0] - p[0], LIGHT_POS[1] - p[1], LIGHT_POS[2] - p[2]];
        let dist = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        let ndl = ((normal[0] * l[0] + normal[1] * l[1] + normal[2] * l[2]) / dist).max(0.0);
        let v = LIGHT_INTENSITY * falloff(dist) * ndl / PI;
        [v; 3]
    }

    /// The room as a linear HDR cube of `size`² faces (+X, −X, +Y, −Y, +Z, −Z).
    ///
    /// Stored MIRRORED IN X, the way any image cubemap is: three.js samples a
    /// regular `CubeTexture` with `flipEnvMap = -1` (and so does
    /// [`PmremGenerator`]), because cube images are authored for a viewer
    /// inside the cube. three.js's own capture is a render target, read
    /// unflipped, so the room has to be written pre-flipped here to come out
    /// the same way round. Unflipped, every box swaps sides in a reflection.
    pub fn cube(size: u32) -> CubeTexture {
        let n = size.max(1);
        let faces: [Vec<f32>; 6] = std::array::from_fn(|face| {
            let mut px = Vec::with_capacity((n * n * 4) as usize);
            for y in 0..n {
                for x in 0..n {
                    let sc = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                    let tc = (y as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                    let d = face_direction(face, sc, tc);
                    let rgb = Self::radiance(Vector3 { x: -d.x, ..d });
                    px.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 1.0]);
                }
            }
            px
        });
        CubeTexture::new_f32(n, faces)
    }

    /// [`cube`](Self::cube) prefiltered for `scene.environment` — three.js's
    /// `pmrem.fromScene(new RoomEnvironment())`, whose default size is 256.
    /// Smaller sizes cost less to build and lose only the sharpest
    /// reflections; diffuse light comes from the coarsest levels either way.
    pub fn pmrem(size: u32) -> Arc<CubeTexture> {
        Arc::new(PmremGenerator::generate_pmrem(&Self::cube(size), size))
    }
}

/// The direction of face texel `(sc, tc)` in `[-1, 1]²` — the inverse of the
/// face selection the cube samplers use.
fn face_direction(face: usize, sc: f32, tc: f32) -> Vector3 {
    let (x, y, z) = match face {
        0 => (1.0, -tc, -sc),
        1 => (-1.0, -tc, sc),
        2 => (sc, 1.0, tc),
        3 => (sc, -1.0, -tc),
        4 => (sc, -tc, 1.0),
        _ => (-sc, -tc, -1.0),
    };
    Vector3 { x, y, z }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }

    /// Looking straight at a panel returns its emissive value; the panels are
    /// thin boxes standing proud of the walls, so a miss here means the box
    /// transform or the slab test is wrong.
    #[test]
    fn panels_are_seen_at_their_intensity() {
        // +y panel straight up
        assert_eq!(RoomEnvironment::radiance(v(0.0, 1.0, 0.0)), [100.0; 3]);
        // +z panel: centre (-0.462, 8.89 - 3.5, 14.52)
        let p = SOLIDS[9].centre;
        assert_eq!(RoomEnvironment::radiance(v(p[0], p[1], p[2])), [43.0; 3]);
        // +x panel
        let p = SOLIDS[8].centre;
        assert_eq!(RoomEnvironment::radiance(v(p[0], p[1], p[2])), [17.0; 3]);
    }

    /// The floor straight below, shaded by hand with three.js's formulas:
    /// floor at y = 13.219 − 28.305/2 − 3.5, light 17.14 above it, so
    /// L = 900 · (1/d²) · (1 − (d/28)⁴)² · n·l / π.
    #[test]
    fn surfaces_are_shaded_as_three_shades_them() {
        let floor_y = 13.219 - 28.305 / 2.0 - DROP;
        let l = [LIGHT_POS[0], LIGHT_POS[1] - floor_y, LIGHT_POS[2]];
        let d = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        let want = 900.0 / (d * d) * (1.0 - (d / 28.0).powi(4)).powi(2) * (l[1] / d) / PI;
        let got = RoomEnvironment::radiance(v(0.0, -1.0, 0.0))[0];
        assert!((got - want).abs() < 1e-4, "floor below: {got}, three.js gives {want}");
        // every lit surface stays between black and the dimmest panel
        for dir in [v(0.3, 1.0, 0.4), v(0.0, 0.0, -1.0), v(-1.0, 0.2, 0.0), v(0.05, -1.0, 0.6)] {
            let r = RoomEnvironment::radiance(dir)[0];
            assert!(r > 0.0 && r < 17.0, "{dir:?}: {r}");
        }
    }

    /// Boxes turn the way three.js's `rotation.y` turns them: a point near
    /// a corner of box 3 (yaw 0.561) placed with three's R_y(θ) is inside it,
    /// and the same corner placed with the opposite turn is not.
    #[test]
    fn boxes_are_rotated_the_way_three_rotates_them() {
        let b = &SOLIDS[2];
        let inside = |yaw: f32| {
            let (s, c) = yaw.sin_cos();
            let (lx, lz) = (b.half[0] * 0.9, b.half[2] * 0.9);
            // three's R_y(θ): x' = c·x + s·z, z' = −s·x + c·z
            let p = [b.centre[0] + c * lx + s * lz, b.centre[1], b.centre[2] - s * lx + c * lz];
            let dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let d = [p[0] / dist, p[1] / dist, p[2] / dist];
            b.slab([0.0; 3], d).is_some_and(|(t0, t1, _, _)| t0 < dist && dist < t1)
        };
        assert!(inside(b.yaw), "a point inside the box as three places it is missed");
        assert!(!inside(-b.yaw), "the box is turned the wrong way");
    }

    /// The cube's faces read back through the sampler's own face selection:
    /// the +y panel lands on +Y, the +z panel on +Z at its x-mirrored place.
    #[test]
    fn cube_faces_are_oriented_as_the_samplers_read_them() {
        // even sizes put no texel on the face axis; 64 lands the nearest
        // ones 0.26 off it, inside the 1×1 ceiling panel
        let n = 64;
        let cube = RoomEnvironment::cube(n);
        let faces = cube.faces_f32.as_ref().expect("hdr faces");
        // the ceiling panel subtends ~3.5°, so it is somewhere on +Y and
        // nowhere on −Y
        let peak = |face: usize| faces[face].chunks(4).fold(0.0f32, |m, p| m.max(p[0]));
        assert_eq!(peak(2), 100.0, "the ceiling panel is not on the +Y face");
        assert!(peak(3) < 17.0, "an emissive panel is drawn on the −Y face");
        // the brightest texel of +Z is the +z panel, below the horizon (y < 0
        // in face terms is the upper half, since t runs down)
        let pz = &faces[4];
        let (mut best, mut at) = (0.0f32, 0usize);
        for i in 0..(n * n) as usize {
            if pz[i * 4] > best {
                best = pz[i * 4];
                at = i;
            }
        }
        assert_eq!(best, 43.0);
        let (x, y) = (at as u32 % n, at as u32 / n);
        let d = face_direction(4, (x as f32 + 0.5) / n as f32 * 2.0 - 1.0, (y as f32 + 0.5) / n as f32 * 2.0 - 1.0);
        // stored mirrored in x (see `cube`)
        let c = SOLIDS[9].centre;
        let p = [-c[0], c[1], c[2]];
        let dot = (d.x * p[0] + d.y * p[1] + d.z * p[2])
            / ((d.x * d.x + d.y * d.y + d.z * d.z).sqrt() * (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
        assert!(dot > 0.97, "the +z panel is drawn away from where it is ({dot})");
    }

    /// Prefiltered at roughness 1 — what a matte surface reads its diffuse
    /// light from — the room must match a Gaussian blur of itself at the
    /// chain's final sigma (three.js's `EXTRA_LOD_SIGMA` ends at 0.582), in
    /// every direction. Regression for the CPU atlas sampler's texel
    /// convention: off by up to half a texel per read, the 18 blur passes
    /// lost a third of the light and left +Z at 0.70 where this asks for 2.19.
    #[test]
    fn prefiltered_room_is_a_gaussian_of_the_room() {
        let env = RoomEnvironment::pmrem(128);
        let atlas = env.cube_uv_atlas.as_ref().expect("cube-uv atlas");
        const N: usize = 100_000;
        let samples: Vec<([f32; 3], f32)> = (0..N)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f32 + 0.5) / N as f32;
                let r = (1.0 - y * y).sqrt();
                let a = i as f32 * 2.399_963;
                let d = [r * a.cos(), y, r * a.sin()];
                (d, RoomEnvironment::radiance(v(d[0], d[1], d[2]))[0])
            })
            .collect();
        let sigma = 0.582f32;
        for dir in [[0.0f32, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
            let (mut w_sum, mut l_sum) = (0.0f64, 0.0f64);
            for (d, l) in &samples {
                let th = (d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2]).clamp(-1.0, 1.0).acos();
                let w = (-(th * th) / (2.0 * sigma * sigma)).exp() as f64;
                w_sum += w;
                l_sum += w * *l as f64;
            }
            let want = (l_sum / w_sum) as f32;
            let got = crate::extras::cube_uv::sample_cube_uv_env(atlas, dir, 1.0)[0];
            assert!(
                (got / want - 1.0).abs() < 0.05,
                "{dir:?}: prefiltered {got}, a σ={sigma} blur of the room gives {want}"
            );
        }
    }

    /// The prefiltered atlas survives the trip out and back — what a worker
    /// computing it, or a cache holding it, relies on: every roughness level
    /// samples the same, and a wrong-sized buffer is refused.
    #[test]
    fn a_prefiltered_atlas_round_trips() {
        let env = RoomEnvironment::pmrem(64);
        let data = env.cube_uv_atlas.as_ref().and_then(|a| a.pixels_f32.as_ref()).expect("f32 atlas").to_vec();
        let back = PmremGenerator::from_atlas_f32(64, data.clone()).expect("a 64 atlas");
        let (a, b) = (env.cube_uv_atlas.as_ref().unwrap(), back.cube_uv_atlas.as_ref().unwrap());
        for dir in [[0.0f32, 1.0, 0.0], [0.6, -0.2, 0.77], [-1.0, 0.0, 0.0]] {
            for r in [0.0f32, 0.35, 0.7, 1.0] {
                let (x, y) = (
                    crate::extras::cube_uv::sample_cube_uv_env(a, dir, r),
                    crate::extras::cube_uv::sample_cube_uv_env(b, dir, r),
                );
                assert_eq!(x, y, "{dir:?} at roughness {r}");
            }
        }
        assert!(PmremGenerator::from_atlas_f32(128, data).is_none(), "a 64 atlas taken for a 128 one");
    }
}
