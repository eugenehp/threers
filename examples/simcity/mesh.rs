//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Mesh builder — accumulates triangles into one BufferGeometry.
// ---------------------------------------------------------------------------

/// How a primitive's UVs are laid out.
#[derive(Clone, Copy)]
pub(crate) enum Uv {
    /// 0..1 across every face. For untextured batches.
    Unit,
    /// World-space projection: `u` tiles per metre horizontally, `v` per metre
    /// vertically. Because it reads world position rather than a per-mesh
    /// frame, window rows line up across a building's setbacks — and across
    /// its neighbours — with no bookkeeping.
    World { u: f32, v: f32 },
}

#[derive(Default)]
pub(crate) struct MeshBuilder {
    /// When set, every primitive in this batch is world-projected at this
    /// scale whatever UV mode it was given. The ground batches need it: each
    /// quad is a different size, so 0..1 per quad would put a 40 m road cell
    /// and a 3 m paint stripe at the same texture scale.
    pub(crate) uv_override: Option<Uv>,
    pub(crate) pos: Vec<f32>,
    pub(crate) nrm: Vec<f32>,
    pub(crate) uv: Vec<f32>,
    pub(crate) col: Vec<f32>,
    pub(crate) idx: Vec<u32>,
}

impl MeshBuilder {
    pub(crate) fn is_empty(&self) -> bool {
        self.idx.is_empty()
    }

    pub(crate) fn tri_count(&self) -> usize {
        self.idx.len() / 3
    }

    pub(crate) fn project(uv: Uv, p: [f32; 3], n: [f32; 3], fallback: [f32; 2]) -> [f32; 2] {
        match uv {
            Uv::Unit => fallback,
            Uv::World { u, v } => {
                if n[1].abs() > 0.5 {
                    [p[0] * u, p[2] * u]
                } else if n[0].abs() > 0.5 {
                    [p[2] * u, p[1] * v]
                } else {
                    [p[0] * u, p[1] * v]
                }
            }
        }
    }

    /// Quad wound counter-clockwise seen from `+n` (the renderer culls back
    /// faces with `FrontFace::Ccw`). Corners must run `c`, `c+u`, `c+u+v`,
    /// `c+v` for some `u x v = n`.
    pub(crate) fn quad(&mut self, v: [[f32; 3]; 4], n: [f32; 3], uv: Uv, c: Color) {
        let base = (self.pos.len() / 3) as u32;
        const UNIT: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let uv = self.uv_override.unwrap_or(uv);
        for k in 0..4 {
            self.pos.extend_from_slice(&v[k]);
            self.nrm.extend_from_slice(&n);
            let t = Self::project(uv, v[k], n, UNIT[k]);
            self.uv.extend_from_slice(&t);
            self.col.extend_from_slice(&[c.r, c.g, c.b]);
        }
        self.idx
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub(crate) fn tri(&mut self, v: [[f32; 3]; 3], n: [f32; 3], uv: [[f32; 2]; 3], c: [Color; 3]) {
        let base = (self.pos.len() / 3) as u32;
        for k in 0..3 {
            self.pos.extend_from_slice(&v[k]);
            self.nrm.extend_from_slice(&n);
            self.uv.extend_from_slice(&uv[k]);
            self.col.extend_from_slice(&[c[k].r, c[k].g, c[k].b]);
        }
        self.idx.extend_from_slice(&[base, base + 1, base + 2]);
    }

    /// Axis-aligned box spanning `min`..`max`.
    pub(crate) fn add_box(&mut self, min: Vector3, max: Vector3, c: Color, uv: Uv) {
        let (x0, y0, z0) = (min.x, min.y, min.z);
        let (x1, y1, z1) = (max.x, max.y, max.z);
        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            // +X: u=+Y, v=+Z
            ([1.0, 0.0, 0.0], [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]),
            // -X: u=+Z, v=+Y
            ([-1.0, 0.0, 0.0], [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]),
            // +Y: u=+Z, v=+X
            ([0.0, 1.0, 0.0], [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]]),
            // -Y: u=+X, v=+Z
            ([0.0, -1.0, 0.0], [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
            // +Z: u=+X, v=+Y
            ([0.0, 0.0, 1.0], [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
            // -Z: u=+Y, v=+X
            ([0.0, 0.0, -1.0], [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]]),
        ];
        for (n, v) in faces {
            self.quad(v, n, uv, c);
        }
    }

    /// Horizontal quad at `y` spanning `x0..x1` by `z0..z1`, facing up.
    pub(crate) fn add_slab(&mut self, x0: f32, z0: f32, x1: f32, z1: f32, y: f32, c: Color, uv: Uv) {
        self.quad(
            [[x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0]],
            [0.0, 1.0, 0.0],
            uv,
            c,
        );
    }

    /// Append another builder's geometry, moved, turned about Y and repainted.
    ///
    /// The point is static copies: a parked car never moves, so it belongs in
    /// a merged batch rather than in an instanced mesh — no draw call, no
    /// per-frame transform, and it can share a batch with everything else made
    /// of painted metal.
    ///
    /// `paint` is applied only to vertices that are *white*, which is the
    /// convention the vehicle bodies already follow: paint is left white so a
    /// material colour can tint it, while glass and rubber carry their own
    /// colours per vertex. Multiplying everything would give a red car red
    /// windows.
    pub(crate) fn append_at(
        &mut self,
        src: &MeshBuilder,
        pos: Vector3,
        yaw: f32,
        scale: f32,
        paint: Color,
    ) {
        let base = (self.pos.len() / 3) as u32;
        let (sn, cs) = (yaw.sin(), yaw.cos());
        let verts = src.pos.len() / 3;
        for i in 0..verts {
            let (x, y, z) = (
                src.pos[i * 3] * scale,
                src.pos[i * 3 + 1] * scale,
                src.pos[i * 3 + 2] * scale,
            );
            self.pos.extend_from_slice(&[
                pos.x + x * cs + z * sn,
                pos.y + y,
                pos.z - x * sn + z * cs,
            ]);
            let (nx, ny, nz) = (src.nrm[i * 3], src.nrm[i * 3 + 1], src.nrm[i * 3 + 2]);
            self.nrm
                .extend_from_slice(&[nx * cs + nz * sn, ny, -nx * sn + nz * cs]);
            self.uv
                .extend_from_slice(&[src.uv[i * 2], src.uv[i * 2 + 1]]);
            let (r, g, bl) = (src.col[i * 3], src.col[i * 3 + 1], src.col[i * 3 + 2]);
            if r.min(g).min(bl) > 0.85 {
                self.col
                    .extend_from_slice(&[r * paint.r, g * paint.g, bl * paint.b]);
            } else {
                self.col.extend_from_slice(&[r, g, bl]);
            }
        }
        self.idx.extend(src.idx.iter().map(|i| i + base));
    }

    /// A box rotated about Y.
    ///
    /// `add_box` is axis-aligned, which is fine for buildings on a grid and
    /// useless for anything a landscape architect drew: benches facing a pond,
    /// jetty planks, pergola beams, a goal at the end of a pitch. Faces are
    /// emitted from an origin corner plus two edge vectors chosen so their
    /// cross product is the outward normal.
    pub(crate) fn add_yaw_box(
        &mut self,
        center: Vector3,
        half: Vector3,
        yaw: f32,
        c: Color,
        uv: Uv,
    ) {
        let (s, co) = (yaw.sin(), yaw.cos());
        let ax = [co * half.x, 0.0, s * half.x];
        let ay = [0.0, half.y, 0.0];
        let az = [-s * half.z, 0.0, co * half.z];
        let add3 = |a: [f32; 3], b: [f32; 3]| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        let sub3 = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let mul3 = |a: [f32; 3], k: f32| [a[0] * k, a[1] * k, a[2] * k];
        let ctr = [center.x, center.y, center.z];
        let norm = |v: [f32; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
            [v[0] / l, v[1] / l, v[2] / l]
        };
        // (origin corner, u, v, outward normal) for each of the six faces.
        let faces = [
            (sub3(sub3(add3(ctr, ax), ay), az), ay, az, ax),
            (sub3(sub3(sub3(ctr, ax), az), ay), az, ay, mul3(ax, -1.0)),
            (sub3(sub3(add3(ctr, ay), az), ax), az, ax, ay),
            (sub3(sub3(sub3(ctr, ay), ax), az), ax, az, mul3(ay, -1.0)),
            (sub3(sub3(add3(ctr, az), ax), ay), ax, ay, az),
            (sub3(sub3(sub3(ctr, az), ay), ax), ay, ax, mul3(az, -1.0)),
        ];
        for (o, u, v, n) in faces {
            let u = mul3(u, 2.0);
            let v = mul3(v, 2.0);
            self.quad(
                [o, add3(o, u), add3(add3(o, u), v), add3(o, v)],
                norm(n),
                uv,
                c,
            );
        }
    }

    /// A flat quad on the ground from four `(x, z)` corners given in order.
    ///
    /// `add_slab` can only make axis-aligned rectangles, which is why every
    /// park in the city used to be a cross of two straight paths. Paths that
    /// curve, cut a diagonal or run round a pond all need this.
    ///
    /// Winding follows the same rule as everything else here: `(p1 - p0) x
    /// (p3 - p0)` has to come out along `+Y` or the quad is back-facing and
    /// simply disappears.
    pub(crate) fn add_ground_quad(&mut self, c: [(f32, f32); 4], y: f32, col: Color, uv: Uv) {
        self.quad(
            [
                [c[0].0, y, c[0].1],
                [c[1].0, y, c[1].1],
                [c[2].0, y, c[2].1],
                [c[3].0, y, c[3].1],
            ],
            [0.0, 1.0, 0.0],
            uv,
            col,
        );
    }

    /// Sweep a polyline into a flat path of constant width.
    ///
    /// A ribbon along a polyline, with mitred joins.
    ///
    /// This used to lay one quad per segment and patch each joint with a small
    /// axis-aligned square, on the reasoning that "at path widths of a metre
    /// or two against a corner that never turns sharply the patch is
    /// invisible". That was true of the park paths it was written for and
    /// false of everything since: an eleven-metre ring road with a fifteen
    /// metre verge, on a diagonal, gets a fifteen-by-fifteen square dropped at
    /// every vertex square to the *world* rather than to the path — so every
    /// bend grew a notch on the outside and a lump on the inside.
    ///
    /// Mitring is the fix and it is not expensive: offset each vertex along
    /// the bisector of its two segments, lengthened by `1/cos(θ/2)` so the
    /// outer edges of both segments meet exactly on it. The scale is clamped,
    /// because that factor goes to infinity as a joint approaches a hairpin
    /// and an unclamped mitre fires a spike across the map.
    ///
    /// A closed ring (first point equal to last) is mitred at the seam too,
    /// which is otherwise the one joint in a loop that keeps the old notch.
    pub(crate) fn add_ground_path(
        &mut self,
        pts: &[(f32, f32)],
        width: f32,
        y: f32,
        col: Color,
        uv: Uv,
    ) {
        if pts.len() < 2 {
            return;
        }
        let hw = width * 0.5;
        // Left normal of each segment, unit length.
        let mut n: Vec<(f32, f32)> = Vec::with_capacity(pts.len() - 1);
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (dx, dz) = (b.0 - a.0, b.1 - a.1);
            let len = (dx * dx + dz * dz).sqrt();
            if len < 1e-4 {
                n.push((0.0, 0.0));
            } else {
                n.push((-dz / len, dx / len));
            }
        }
        let closed = {
            let (a, b) = (pts[0], pts[pts.len() - 1]);
            (a.0 - b.0).abs() < 1e-3 && (a.1 - b.1).abs() < 1e-3
        };
        // Offset at each vertex: the bisector of the segments meeting there.
        let offset = |i: usize| -> (f32, f32) {
            let last = n.len() - 1;
            let (prev, next) = if i == 0 {
                if closed {
                    (n[last], n[0])
                } else {
                    (n[0], n[0])
                }
            } else if i >= n.len() {
                if closed {
                    (n[last], n[0])
                } else {
                    (n[last], n[last])
                }
            } else {
                (n[i - 1], n[i])
            };
            let (mx, mz) = (prev.0 + next.0, prev.1 + next.1);
            let ml = (mx * mx + mz * mz).sqrt();
            if ml < 1e-4 {
                // Doubled back on itself; the mitre is undefined, so take the
                // segment normal and accept the notch.
                return (next.0 * hw, next.1 * hw);
            }
            let (ux, uz) = (mx / ml, mz / ml);
            // 1/cos(theta/2), clamped so a near-hairpin cannot fire a spike.
            let scale = (hw / (ux * next.0 + uz * next.1).abs().max(0.30)).min(hw * 3.2);
            (ux * scale, uz * scale)
        };
        for i in 0..n.len() {
            if n[i].0 == 0.0 && n[i].1 == 0.0 {
                continue;
            }
            let (a, b) = (pts[i], pts[i + 1]);
            let oa = offset(i);
            let ob = offset(i + 1);
            self.add_ground_quad(
                [
                    (a.0 + oa.0, a.1 + oa.1),
                    (b.0 + ob.0, b.1 + ob.1),
                    (b.0 - ob.0, b.1 - ob.1),
                    (a.0 - oa.0, a.1 - oa.1),
                ],
                y,
                col,
                uv,
            );
        }
    }

    /// A flat disc, as a triangle fan. `wobble` perturbs the radius by a hash
    /// of the segment index, which is what turns a circular pond into one with
    /// a natural-looking edge.
    pub(crate) fn add_ground_disc(
        &mut self,
        cx: f32,
        cz: f32,
        r: f32,
        segments: usize,
        wobble: f32,
        y: f32,
        col: Color,
    ) {
        let n = segments.max(3);
        let rad = |i: usize| -> f32 {
            let i = i % n;
            r * (1.0 + wobble * (city_hash2(i as i32 * 13, 7) - 0.5) * 2.0)
        };
        // Theta runs backwards: taking it forwards winds the fan face-down.
        for i in 0..n {
            let (a, bb) = (
                i as f32 / n as f32 * TAU,
                (i + 1) as f32 / n as f32 * TAU,
            );
            let (ra, rb) = (rad(i), rad(i + 1));
            self.tri(
                [
                    [cx, y, cz],
                    [cx + rb * bb.cos(), y, cz + rb * bb.sin()],
                    [cx + ra * a.cos(), y, cz + ra * a.sin()],
                ],
                [0.0, 1.0, 0.0],
                [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                [col; 3],
            );
        }
    }

    /// The wall of an irregular basin: a skirt dropped from the disc's rim to
    /// `y_low`, so a pond reads as sunk into the ground rather than painted on.
    pub(crate) fn add_ground_disc_wall(
        &mut self,
        cx: f32,
        cz: f32,
        r: f32,
        segments: usize,
        wobble: f32,
        y_top: f32,
        y_low: f32,
        col: Color,
    ) {
        let n = segments.max(3);
        let rad = |i: usize| -> f32 {
            let i = i % n;
            r * (1.0 + wobble * (city_hash2(i as i32 * 13, 7) - 0.5) * 2.0)
        };
        for i in 0..n {
            let (a, bb) = (
                i as f32 / n as f32 * TAU,
                (i + 1) as f32 / n as f32 * TAU,
            );
            let (ra, rb) = (rad(i), rad(i + 1));
            let (p0, p1) = (
                (cx + ra * a.cos(), cz + ra * a.sin()),
                (cx + rb * bb.cos(), cz + rb * bb.sin()),
            );
            // Inward-facing: the bank is only ever seen from outside the pond.
            let nx = -(a.cos() + bb.cos()) * 0.5;
            let nz = -(a.sin() + bb.sin()) * 0.5;
            self.quad(
                [
                    [p0.0, y_top, p0.1],
                    [p0.0, y_low, p0.1],
                    [p1.0, y_low, p1.1],
                    [p1.0, y_top, p1.1],
                ],
                [nx, 0.25, nz],
                Uv::Unit,
                col,
            );
        }
    }

    /// Gable roof: a triangular prism ridged along the longer horizontal axis.
    pub(crate) fn add_gable(&mut self, min: Vector3, max: Vector3, rise: f32, c: Color) {
        let (x0, y0, z0) = (min.x, min.y, min.z);
        let (x1, z1) = (max.x, max.z);
        let yr = y0 + rise;
        let end = scale_color(c, 0.92);
        const UV3: [[f32; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        if (x1 - x0) >= (z1 - z0) {
            // Ridge along X. A pitch rises `rise` over `half`, so its normal is
            // (0, half, rise) — NOT (0, half, half), which is a 45-degree roof
            // whatever the pitch actually is.
            let zm = (z0 + z1) * 0.5;
            let half = (z1 - z0) * 0.5;
            let n = Vector3::new(0.0, half, rise).normalize();
            self.quad(
                [[x0, y0, z1], [x1, y0, z1], [x1, yr, zm], [x0, yr, zm]],
                [0.0, n.y, n.z],
                Uv::Unit,
                c,
            );
            self.quad(
                [[x1, y0, z0], [x0, y0, z0], [x0, yr, zm], [x1, yr, zm]],
                [0.0, n.y, -n.z],
                Uv::Unit,
                c,
            );
            self.tri(
                [[x1, y0, z1], [x1, y0, z0], [x1, yr, zm]],
                [1.0, 0.0, 0.0],
                UV3,
                [end; 3],
            );
            self.tri(
                [[x0, y0, z0], [x0, y0, z1], [x0, yr, zm]],
                [-1.0, 0.0, 0.0],
                UV3,
                [end; 3],
            );
        } else {
            let xm = (x0 + x1) * 0.5;
            let half = (x1 - x0) * 0.5;
            let n = Vector3::new(rise, half, 0.0).normalize();
            self.quad(
                [[x1, y0, z1], [x1, y0, z0], [xm, yr, z0], [xm, yr, z1]],
                [n.x, n.y, 0.0],
                Uv::Unit,
                c,
            );
            self.quad(
                [[x0, y0, z0], [x0, y0, z1], [xm, yr, z1], [xm, yr, z0]],
                [-n.x, n.y, 0.0],
                Uv::Unit,
                c,
            );
            self.tri(
                [[x0, y0, z1], [x1, y0, z1], [xm, yr, z1]],
                [0.0, 0.0, 1.0],
                UV3,
                [end; 3],
            );
            self.tri(
                [[x1, y0, z0], [x0, y0, z0], [xm, yr, z0]],
                [0.0, 0.0, -1.0],
                UV3,
                [end; 3],
            );
        }
    }

    /// Vertical cylinder / truncated cone standing on `base`.
    pub(crate) fn add_cylinder(
        &mut self,
        base: Vector3,
        r_bottom: f32,
        r_top: f32,
        height: f32,
        segs: usize,
        c: Color,
        caps: bool,
        uv: Uv,
    ) {
        let segs = segs.max(3);
        let slope = (r_bottom - r_top) / height.max(1e-4);
        let top = base.y + height;
        for s in 0..segs {
            let a0 = s as f32 / segs as f32 * TAU;
            let a1 = (s + 1) as f32 / segs as f32 * TAU;
            let (c0, s0) = (a0.cos(), a0.sin());
            let (c1, s1) = (a1.cos(), a1.sin());
            let n0 = Vector3::new(c0, slope, s0).normalize();
            let n1 = Vector3::new(c1, slope, s1).normalize();
            let nm = Vector3::new((n0.x + n1.x) * 0.5, n0.y, (n0.z + n1.z) * 0.5).normalize();
            let p00 = [base.x + c0 * r_bottom, base.y, base.z + s0 * r_bottom];
            let p10 = [base.x + c1 * r_bottom, base.y, base.z + s1 * r_bottom];
            let p11 = [base.x + c1 * r_top, top, base.z + s1 * r_top];
            let p01 = [base.x + c0 * r_top, top, base.z + s0 * r_top];
            // c, c+u, c+u+v, c+v with u = UP and v = around. The other
            // order — around then up — has a cross product pointing into the
            // cylinder, and every pole and trunk gets back-face culled.
            self.quad([p00, p01, p11, p10], [nm.x, nm.y, nm.z], uv, c);
        }
        if !caps {
            return;
        }
        // Seen from +Y the CCW basis is (u = +Z, v = +X), so a cap fan has to
        // run clockwise in the (cos, sin) parameter to face upward.
        let cap = scale_color(c, 1.03);
        for s in 0..segs {
            let a0 = s as f32 / segs as f32 * TAU;
            let a1 = (s + 1) as f32 / segs as f32 * TAU;
            if r_top > 1e-4 {
                self.tri(
                    [
                        [base.x, top, base.z],
                        [base.x + a1.cos() * r_top, top, base.z + a1.sin() * r_top],
                        [base.x + a0.cos() * r_top, top, base.z + a0.sin() * r_top],
                    ],
                    [0.0, 1.0, 0.0],
                    [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                    [cap; 3],
                );
            }
            if r_bottom > 1e-4 {
                self.tri(
                    [
                        [base.x, base.y, base.z],
                        [base.x + a0.cos() * r_bottom, base.y, base.z + a0.sin() * r_bottom],
                        [base.x + a1.cos() * r_bottom, base.y, base.z + a1.sin() * r_bottom],
                    ],
                    [0.0, -1.0, 0.0],
                    [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                    [cap; 3],
                );
            }
        }
    }

    /// A tapered cylinder between two arbitrary points.
    ///
    /// `add_cylinder` only stands things up the Y axis, which is why every
    /// limb in this file used to be a box: a swung leg cannot be axis-aligned.
    /// This is the primitive that fixes that, and it does branches too.
    pub(crate) fn add_limb(
        &mut self,
        a: Vector3,
        b: Vector3,
        r_a: f32,
        r_b: f32,
        segs: usize,
        c: Color,
        caps: bool,
    ) {
        self.add_limb_flat(a, b, r_a, r_b, 1.0, segs, c, caps);
    }

    /// As `add_limb`, but with an elliptical cross-section: `flatten` scales
    /// the first axis of the ring.
    ///
    /// A torso is about twice as wide as it is deep. Drawn round it reads as a
    /// length of pipe, which is most of what made the first pass at these
    /// figures look like toys.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_limb_flat(
        &mut self,
        a: Vector3,
        b: Vector3,
        r_a: f32,
        r_b: f32,
        flatten: f32,
        segs: usize,
        c: Color,
        caps: bool,
    ) {
        let axis = b - a;
        let len = axis.length();
        if len < 1e-5 {
            return;
        }
        let segs = segs.max(3);
        let w = axis * (1.0 / len);
        // Any vector not parallel to the axis, so the frame is well defined
        // whichever way the limb points.
        let helper = if w.y.abs() < 0.9 {
            Vector3::UP
        } else {
            Vector3::RIGHT
        };
        let u = helper.cross(w).normalize();
        let v = w.cross(u);
        let slope = (r_a - r_b) / len;

        let ring = |ang: f32, r: f32, base: Vector3| -> [f32; 3] {
            let d = u * (ang.cos() * flatten) + v * ang.sin();
            let p = base + d * r;
            [p.x, p.y, p.z]
        };
        for s in 0..segs {
            let a0 = s as f32 / segs as f32 * TAU;
            let a1 = (s + 1) as f32 / segs as f32 * TAU;
            let mid = (a0 + a1) * 0.5;
            // Normal of an ellipse: the axis scaling is inverted, not applied.
            let radial = u * (mid.cos() / flatten.max(0.05)) + v * mid.sin();
            let n = (radial.normalize() + w * slope).normalize();
            // c, c+around, c+around+along, c+along: `around x along` points
            // out of the surface, which is the order the renderer's
            // counter-clockwise front face wants.
            self.quad(
                [
                    ring(a0, r_a, a),
                    ring(a1, r_a, a),
                    ring(a1, r_b, b),
                    ring(a0, r_b, b),
                ],
                [n.x, n.y, n.z],
                Uv::Unit,
                c,
            );
        }
        if !caps {
            return;
        }
        for s in 0..segs {
            let a0 = s as f32 / segs as f32 * TAU;
            let a1 = (s + 1) as f32 / segs as f32 * TAU;
            if r_b > 1e-4 {
                self.tri(
                    [[b.x, b.y, b.z], ring(a1, r_b, b), ring(a0, r_b, b)],
                    [w.x, w.y, w.z],
                    [[0.5, 0.5]; 3],
                    [c; 3],
                );
            }
            if r_a > 1e-4 {
                self.tri(
                    [[a.x, a.y, a.z], ring(a0, r_a, a), ring(a1, r_a, a)],
                    [-w.x, -w.y, -w.z],
                    [[0.5, 0.5]; 3],
                    [c; 3],
                );
            }
        }
    }

    /// Squashed UV sphere.
    pub(crate) fn add_ellipsoid(
        &mut self,
        center: Vector3,
        rx: f32,
        ry: f32,
        rz: f32,
        rings: usize,
        sectors: usize,
        c: Color,
    ) {
        self.add_blob(center, rx, ry, rz, rings, sectors, 0.0, 0, 0.55, c);
    }

    /// A squashed UV sphere with its radius jittered per vertex.
    ///
    /// `jitter` is the fraction the radius may wander. A perfectly smooth
    /// ovoid never reads as foliage however it is coloured — the silhouette
    /// gives it away — and this is the cheapest thing that breaks it, since it
    /// costs no extra vertices.
    ///
    /// The jitter is hashed on `(ring, sector)` and NOT on position, so the
    /// duplicated seam vertices agree and the poles, where every sector
    /// collapses to one point, stay closed.
    ///
    /// `floor` is how dark the underside goes: foliage is darkest where least
    /// light reaches it, which is a cheap stand-in for the occlusion inside a
    /// crown.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_blob(
        &mut self,
        center: Vector3,
        rx: f32,
        ry: f32,
        rz: f32,
        rings: usize,
        sectors: usize,
        jitter: f32,
        seed: i32,
        floor: f32,
        c: Color,
    ) {
        let bump = |ring: usize, sec: usize| -> f32 {
            if jitter <= 0.0 {
                return 1.0;
            }
            // Poles share one point, so their jitter must not depend on the
            // sector or the tip tears open.
            let s = if ring == 0 || ring == rings {
                0
            } else {
                sec % sectors
            };
            let h = city_hash2(seed + ring as i32 * 71, s as i32 * 131 + seed * 17);
            1.0 + (h - 0.5) * 2.0 * jitter
        };
        let point = |ring: usize, sec: usize| -> ([f32; 3], [f32; 3]) {
            let phi = ring as f32 / rings as f32 * PI;
            let theta = sec as f32 / sectors as f32 * TAU;
            let (sp, cp) = (phi.sin(), phi.cos());
            let (st, ct) = (theta.sin(), theta.cos());
            let n = Vector3::new(sp * ct, cp, sp * st);
            let k = bump(ring, sec);
            (
                [
                    center.x + n.x * rx * k,
                    center.y + n.y * ry * k,
                    center.z + n.z * rz * k,
                ],
                [n.x, n.y, n.z],
            )
        };
        for r in 0..rings {
            for s in 0..sectors {
                let (a, na) = point(r, s);
                let (b, _) = point(r, s + 1);
                let (cc, _) = point(r + 1, s + 1);
                let (d, _) = point(r + 1, s);
                // Darken with depth into the crown. The sphere normal is used
                // for shading even where the radius wandered: smooth-shading
                // the facets is what keeps a jittered blob from looking
                // crystalline.
                let t = (na[1] * 0.5 + 0.5).clamp(0.0, 1.0);
                self.quad(
                    [a, b, cc, d],
                    na,
                    Uv::Unit,
                    scale_color(c, floor + (1.17 - floor) * t),
                );
            }
        }
    }


    /// Flat fan facing up, bright at the middle and black at the rim — an
    /// unlit stand-in for the pool of light under a street lamp.
    pub(crate) fn add_glow_disc(&mut self, center: Vector3, radius: f32, segs: usize, c: Color) {
        // Two rings, not one fan: a single fan falls off linearly from the
        // middle to the rim, which reads as a hard-edged sticker. An inner
        // ring held near full brightness gives the pool a core.
        let mid = radius * 0.42;
        let inner = scale_color(c, 0.88);
        let at = |a: f32, r: f32| [center.x + a.cos() * r, center.y, center.z + a.sin() * r];
        for s in 0..segs {
            let a0 = s as f32 / segs as f32 * TAU;
            let a1 = (s + 1) as f32 / segs as f32 * TAU;
            self.tri(
                [[center.x, center.y, center.z], at(a1, mid), at(a0, mid)],
                [0.0, 1.0, 0.0],
                [[0.5, 0.5]; 3],
                [c, inner, inner],
            );
            self.quad_shaded(
                [at(a0, mid), at(a0, radius), at(a1, radius), at(a1, mid)],
                [0.0, 1.0, 0.0],
                [inner, Color::BLACK, Color::BLACK, inner],
            );
        }
    }

    /// A quad with a colour per corner. `quad` takes one colour for all four.
    pub(crate) fn quad_shaded(&mut self, v: [[f32; 3]; 4], n: [f32; 3], c: [Color; 4]) {
        let base = (self.pos.len() / 3) as u32;
        for k in 0..4 {
            self.pos.extend_from_slice(&v[k]);
            self.nrm.extend_from_slice(&n);
            self.uv.extend_from_slice(&[0.5, 0.5]);
            self.col.extend_from_slice(&[c[k].r, c[k].g, c[k].b]);
        }
        self.idx
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub(crate) fn build(self) -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(self.pos, 3));
        g.set_attribute("normal", BufferAttribute::new(self.nrm, 3));
        g.set_attribute("uv", BufferAttribute::new(self.uv, 2));
        g.set_attribute("color", BufferAttribute::new(self.col, 3));
        g.set_index(self.idx);
        g
    }
}
