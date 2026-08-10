//! OpenSCAD-style solid modeling front end (Cargo feature `openscad`).
//!
//! Two ways in, both producing a [`Solid`] CSG tree:
//! - the **`.scad` interpreter** (`scad::parse_scad`) — the OpenSCAD language
//!   (primitives, transforms, booleans, extrusions, `hull`/`minkowski`/`offset`,
//!   modules, functions, comprehensions, `surface`, `import`, …), and
//! - the **Rust DSL** ([`dsl`]) — the `Solid` builder, the `scad!` macro, and the
//!   `union!`/`difference!`/`intersection!`/`hull!` combinators.
//!
//! [`Solid::to_geometry_exact`] evaluates each boolean with the **watertight
//! mesh-arrangement kernel** ([`crate::exact_csg`]) — which resolves genuinely
//! curved∧curved crossings — and falls back to the float [`CsgEvaluator`] only
//! where the never-wrong manifold gate can't verify a result. Meshes go in and out
//! via `scad::import` / [`export`] (STL/OBJ/OFF/3MF/AMF, glTF-GLB, DXF/SVG).
//!
//! Conventions are threers-flavoured, not OpenSCAD-literal: primitives are
//! centred on the origin (three.js style) and cylinders run along **Y**. Use the
//! transform builders to place them.
//!
//! ```ignore
//! use threers::{cube, cylinder, sphere};
//! use std::f32::consts::FRAC_PI_2;
//!
//! let part = cube([40.0, 24.0, 4.0])
//!     .difference(cylinder(20.0, 2.5).rotate_x(FRAC_PI_2).translate([-12.0, 0.0, 0.0]))
//!     .union(sphere(5.0).translate([0.0, 0.0, 2.0]));
//! let geometry = part.to_geometry();          // a normal BufferGeometry
//! ```

use crate::exact_csg::{boolean as exact_boolean, BooleanOutcome, Op as ExactOp};
use crate::math::{Matrix4, Vector3};
use crate::{
    compute_vertex_normals, BoxGeometry, BufferAttribute, BufferGeometry, CsgBrush, CsgEvaluator,
    CylinderGeometry, SphereGeometry, ADDITION, INTERSECTION, SUBTRACTION,
};

/// OpenSCAD language front end (`.scad` → [`Solid`]).
pub mod scad;

/// A Rust DSL (`scad!` macro + fluent builders) for authoring [`Solid`] trees.
pub mod dsl;

/// Mesh exporters — OBJ / OFF / 3MF / glTF-GLB (counterpart to `scad::import`).
pub mod export;

/// Default facet count for curved primitives when `$fn` is unspecified.
pub const DEFAULT_FN: usize = 32;

/// A constructive-solid-geometry expression tree.
///
/// Build leaves with [`cube`], [`sphere`], [`cylinder`], [`cone`],
/// [`polyhedron`]; combine with [`Solid::union`] / [`Solid::difference`] /
/// [`Solid::intersection`]; place with the transform builders. Evaluate with
/// [`Solid::to_geometry`].
#[derive(Debug, Clone)]
pub enum Solid {
    /// A primitive, already baked to its identity pose.
    Leaf(BufferGeometry),
    /// A transform applied to a child (baked into vertices at evaluation time).
    Transform { matrix: Matrix4, child: Box<Solid> },
    /// `head ∪ rest…`
    Union(Vec<Solid>),
    /// `head − rest…`
    Difference(Vec<Solid>),
    /// `head ∩ rest…`
    Intersection(Vec<Solid>),
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// Axis-aligned box of the given `[x, y, z]` size, centred on the origin.
pub fn cube(size: [f32; 3]) -> Solid {
    Solid::Leaf(BoxGeometry::new(size[0], size[1], size[2]))
}

/// Sphere of radius `r` at the default facet count.
pub fn sphere(r: f32) -> Solid {
    sphere_fn(r, DEFAULT_FN)
}

/// Sphere of radius `r` with `fn_` facets (`$fn`).
pub fn sphere_fn(r: f32, fn_: usize) -> Solid {
    let ws = fn_.max(3);
    let hs = (fn_ / 2).max(2);
    Solid::Leaf(SphereGeometry::new(r, ws, hs))
}

/// Cylinder of height `h` and radius `r`, centred on the origin, axis = Y.
pub fn cylinder(h: f32, r: f32) -> Solid {
    frustum(h, r, r, DEFAULT_FN)
}

/// Truncated cone of height `h`, bottom radius `r1`, top radius `r2`
/// (OpenSCAD `cylinder(h, r1, r2)` argument order). `r2 = 0.0` gives a cone.
pub fn cone(h: f32, r1: f32, r2: f32) -> Solid {
    frustum(h, r1, r2, DEFAULT_FN)
}

/// [`cone`] with an explicit facet count (`$fn`).
pub fn frustum(h: f32, r1: f32, r2: f32, fn_: usize) -> Solid {
    // CylinderGeometry takes (radius_top, radius_bottom, ...); r1 is the bottom.
    Solid::Leaf(CylinderGeometry::new(
        r2,
        r1,
        h,
        fn_.max(3),
        1,
        false,
        0.0,
        std::f32::consts::TAU,
    ))
}

/// Arbitrary polyhedron from explicit `points` and `faces` (each face a list of
/// point indices, fan-triangulated). Points are used verbatim (no sphere
/// projection). Winding is **auto-oriented outward** (so either face ordering
/// yields a solid with outward normals — a correct CSG operand).
pub fn polyhedron(points: &[[f32; 3]], faces: &[Vec<u32>]) -> Solid {
    let mut positions: Vec<f32> = Vec::new();
    for face in faces {
        if face.len() < 3 {
            continue;
        }
        for k in 1..face.len() - 1 {
            for &vi in &[face[0], face[k], face[k + 1]] {
                let p = points[vi as usize];
                positions.extend_from_slice(&[p[0], p[1], p[2]]);
            }
        }
    }
    // Auto-orient: if the signed volume is negative the faces are inward-wound,
    // so flip every triangle to make the solid outward-facing.
    let mut vol = 0.0f64;
    for t in positions.chunks_exact(9) {
        let f = |i: usize| t[i] as f64;
        vol += f(0) * (f(4) * f(8) - f(5) * f(7)) + f(1) * (f(5) * f(6) - f(3) * f(8))
            + f(2) * (f(3) * f(7) - f(4) * f(6));
    }
    if vol < 0.0 {
        for t in positions.chunks_exact_mut(9) {
            for k in 0..3 {
                t.swap(3 + k, 6 + k);
            }
        }
    }
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(positions, 3));
    compute_vertex_normals(&mut g);
    Solid::Leaf(g)
}

/// Extrude a 2D polygon (`[x, y]` outer ring) along **Z** by `height`, base at
/// `z = 0` (OpenSCAD `linear_extrude(height) polygon(points)`). The ring is
/// normalized to counter-clockwise, so either winding is accepted. Holes are not
/// yet supported (M0). Built directly from the exact vertices — corners are not
/// chamfered (unlike the arc-length-sampled `ExtrudeGeometry`).
pub fn linear_extrude(height: f32, outline: &[[f32; 2]]) -> Solid {
    Solid::Leaf(extrude_polygon(outline, height))
}

/// [`linear_extrude`] with holes: each entry of `holes` is an inner ring removed
/// from the outer `outline` (OpenSCAD `polygon(paths=…)`). Implemented as the
/// outer prism minus a taller prism per hole — the hole prisms overhang both
/// faces (offset by `height`) so no caps are coplanar with the plate.
pub fn linear_extrude_holes(height: f32, outline: &[[f32; 2]], holes: &[Vec<[f32; 2]>]) -> Solid {
    let mut solid = linear_extrude(height, outline);
    let margin = (height.abs()).max(0.01);
    for hole in holes {
        if hole.len() < 3 {
            continue;
        }
        let punch = linear_extrude(height + 2.0 * margin, hole).translate([0.0, 0.0, -margin]);
        solid = solid.difference(punch);
    }
    solid
}

/// Revolve a 2D profile (`[radius, height]` points) `angle` radians around the
/// **Y** axis (OpenSCAD `rotate_extrude`, threers Y-up convention). Full `TAU`
/// of a closed profile yields a closed manifold suitable for CSG; partial angles
/// leave open ends and should not be used as boolean operands.
pub fn rotate_extrude(angle: f32, profile: &[[f32; 2]]) -> Solid {
    rotate_extrude_fn(angle, profile, DEFAULT_FN)
}

/// [`rotate_extrude`] with an explicit segment count (`$fn`).
pub fn rotate_extrude_fn(angle: f32, profile: &[[f32; 2]], segments: usize) -> Solid {
    let pts: Vec<crate::math::Vector2> = profile
        .iter()
        .map(|p| crate::math::Vector2::new(p[0], p[1]))
        .collect();
    Solid::Leaf(crate::LatheGeometry::new(
        &pts,
        segments.max(3),
        0.0,
        angle.clamp(0.0, std::f32::consts::TAU),
    ))
}

/// Signed area of a 2D ring (>0 for counter-clockwise).
fn signed_area(pts: &[[f32; 2]]) -> f32 {
    let n = pts.len();
    let mut a = 0.0;
    for i in 0..n {
        let p = pts[i];
        let q = pts[(i + 1) % n];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

/// Build a closed prism from an exact polygon ring, extruded along Z by `height`.
/// Caps are earcut-triangulated; every face is wound outward (positive signed
/// volume) so the CSG kernel classifies inside/outside correctly.
fn extrude_polygon(outline: &[[f32; 2]], height: f32) -> BufferGeometry {
    if outline.len() < 3 {
        return BufferGeometry::new();
    }
    let mut ring = outline.to_vec();
    if signed_area(&ring) < 0.0 {
        ring.reverse(); // normalize to CCW for earcut + outward side normals
    }
    let pts2: Vec<crate::math::Vector2> = ring
        .iter()
        .map(|p| crate::math::Vector2::new(p[0], p[1]))
        .collect();
    let n = pts2.len();
    let cap = crate::curves::earcut::earcut(&pts2, &[]);

    let mut positions: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for p in &pts2 {
        positions.extend_from_slice(&[p.x, p.y, 0.0]);
        normals.extend_from_slice(&[0.0, 0.0, -1.0]);
    }
    for p in &pts2 {
        positions.extend_from_slice(&[p.x, p.y, height]);
        normals.extend_from_slice(&[0.0, 0.0, 1.0]);
    }
    for t in cap.chunks_exact(3) {
        indices.extend_from_slice(&[t[0], t[2], t[1]]); // bottom reversed → -Z
        let off = n as u32;
        indices.extend_from_slice(&[off + t[0], off + t[1], off + t[2]]); // top → +Z
    }

    for i in 0..n {
        let a = pts2[i];
        let b = pts2[(i + 1) % n];
        let (ex, ey) = (b.x - a.x, b.y - a.y);
        let len = (ex * ex + ey * ey).sqrt().max(1e-8);
        let (nx, ny) = (ey / len, -ex / len); // outward for a CCW ring
        let base = (positions.len() / 3) as u32;
        positions.extend_from_slice(&[a.x, a.y, 0.0]);
        positions.extend_from_slice(&[b.x, b.y, 0.0]);
        positions.extend_from_slice(&[b.x, b.y, height]);
        positions.extend_from_slice(&[a.x, a.y, height]);
        for _ in 0..4 {
            normals.extend_from_slice(&[nx, ny, 0.0]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(positions, 3));
    g.set_attribute("normal", BufferAttribute::new(normals, 3));
    g.set_index(indices);
    g
}

// ---------------------------------------------------------------------------
// Transforms & booleans (builder API)
// ---------------------------------------------------------------------------

impl Solid {
    /// Translate by `[x, y, z]`.
    pub fn translate(self, t: [f32; 3]) -> Solid {
        self.transformed(Matrix4::translation(Vector3::new(t[0], t[1], t[2])))
    }

    /// Non-uniform scale by `[x, y, z]` (negative components mirror).
    pub fn scale(self, s: [f32; 3]) -> Solid {
        self.transformed(Matrix4::scale(Vector3::new(s[0], s[1], s[2])))
    }

    /// Rotate `rad` radians about the X axis.
    pub fn rotate_x(self, rad: f32) -> Solid {
        self.transformed(rot_x(rad))
    }

    /// Rotate `rad` radians about the Y axis.
    pub fn rotate_y(self, rad: f32) -> Solid {
        self.transformed(rot_y(rad))
    }

    /// Rotate `rad` radians about the Z axis.
    pub fn rotate_z(self, rad: f32) -> Solid {
        self.transformed(rot_z(rad))
    }

    /// Apply an arbitrary affine matrix.
    pub fn transform(self, m: Matrix4) -> Solid {
        self.transformed(m)
    }

    /// Rotate by Euler angles `[x, y, z]` in **degrees** — OpenSCAD's
    /// `rotate([x, y, z])` (about X, then Y, then Z). Friendlier than the
    /// radian-based [`rotate_x`](Self::rotate_x)/`_y`/`_z` for DSL use.
    pub fn rotate(self, deg: [f32; 3]) -> Solid {
        self.rotate_x(deg[0].to_radians())
            .rotate_y(deg[1].to_radians())
            .rotate_z(deg[2].to_radians())
    }

    /// Rotate `deg` degrees about an arbitrary axis `[x, y, z]` — OpenSCAD's
    /// `rotate(a, v)`. A zero-length axis is a no-op.
    pub fn rotate_axis(self, deg: f32, axis: [f32; 3]) -> Solid {
        let (x, y, z) = (axis[0], axis[1], axis[2]);
        let len = (x * x + y * y + z * z).sqrt();
        if len < 1e-9 {
            return self;
        }
        let (x, y, z) = (x / len, y / len, z / len);
        let (s, c) = deg.to_radians().sin_cos();
        let t = 1.0 - c;
        let mut m = Matrix4::identity();
        // Rodrigues rotation, column-major (index = col*4 + row).
        m.elements = [
            t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0.0, // col 0
            t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0.0, // col 1
            t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0.0, // col 2
            0.0, 0.0, 0.0, 1.0, // col 3
        ];
        self.transformed(m)
    }

    /// Mirror across the plane through the origin with normal `[x, y, z]` —
    /// OpenSCAD's `mirror(v)`. A zero-length normal is a no-op.
    pub fn mirror(self, normal: [f32; 3]) -> Solid {
        let (x, y, z) = (normal[0], normal[1], normal[2]);
        let len = (x * x + y * y + z * z).sqrt();
        if len < 1e-9 {
            return self;
        }
        let (nx, ny, nz) = (x / len, y / len, z / len);
        let mut m = Matrix4::identity();
        // Reflection R = I − 2·n̂·n̂ᵀ; symmetric, so column- and row-major agree.
        m.elements = [
            1.0 - 2.0 * nx * nx, -2.0 * nx * ny, -2.0 * nx * nz, 0.0,
            -2.0 * nx * ny, 1.0 - 2.0 * ny * ny, -2.0 * ny * nz, 0.0,
            -2.0 * nx * nz, -2.0 * ny * nz, 1.0 - 2.0 * nz * nz, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        self.transformed(m)
    }

    fn transformed(self, m: Matrix4) -> Solid {
        match self {
            // Collapse nested transforms: the new one is applied outermost.
            Solid::Transform { matrix, child } => Solid::Transform {
                matrix: m.multiply(&matrix),
                child,
            },
            other => Solid::Transform {
                matrix: m,
                child: Box::new(other),
            },
        }
    }

    /// Union with `other` (flattens chained unions into one n-ary node).
    pub fn union(self, other: Solid) -> Solid {
        match self {
            Solid::Union(mut xs) => {
                xs.push(other);
                Solid::Union(xs)
            }
            s => Solid::Union(vec![s, other]),
        }
    }

    /// Subtract `other` (`self − other`; flattens chained subtractions).
    pub fn difference(self, other: Solid) -> Solid {
        match self {
            Solid::Difference(mut xs) => {
                xs.push(other);
                Solid::Difference(xs)
            }
            s => Solid::Difference(vec![s, other]),
        }
    }

    /// Intersect with `other` (flattens chained intersections).
    pub fn intersection(self, other: Solid) -> Solid {
        match self {
            Solid::Intersection(mut xs) => {
                xs.push(other);
                Solid::Intersection(xs)
            }
            s => Solid::Intersection(vec![s, other]),
        }
    }

    /// Evaluate to a `BufferGeometry` with a fresh evaluator.
    pub fn to_geometry(self) -> BufferGeometry {
        run_csg(move || {
            let mut ev = CsgEvaluator::new();
            self.evaluate(&mut ev)
        })
    }

    /// Evaluate to a `BufferGeometry`, reusing an existing evaluator.
    pub fn evaluate(self, ev: &mut CsgEvaluator) -> BufferGeometry {
        eval_solid(self, ev)
    }

    /// Evaluate using the robust arrangement kernel ([`crate::exact_csg`]) for
    /// each boolean, falling back to the float `CsgEvaluator` per-operation when
    /// the arrangement can't verify a watertight result. The result is always
    /// valid: the arrangement where it's confident, the float kernel otherwise.
    pub fn to_geometry_exact(self) -> BufferGeometry {
        run_csg(move || eval_exact(self))
    }

    /// Evaluate (via [`Solid::to_geometry_exact`]) and encode as **binary STL** —
    /// the printable-mesh output an OpenSCAD-style tool exists to produce.
    pub fn to_stl(self) -> Vec<u8> {
        geometry_to_stl(&self.to_geometry_exact())
    }

    /// Evaluate and encode as **Wavefront OBJ** (text).
    pub fn to_obj(self) -> String {
        export::geometry_to_obj(&self.to_geometry_exact())
    }
    /// Evaluate and encode as **Geomview OFF** (text).
    pub fn to_off(self) -> String {
        export::geometry_to_off(&self.to_geometry_exact())
    }
    /// Evaluate and encode as a **3MF** package (bytes) a slicer can open.
    pub fn to_3mf(self) -> Vec<u8> {
        export::geometry_to_3mf(&self.to_geometry_exact())
    }
    /// Evaluate and encode as binary **glTF 2.0** (`.glb`, bytes).
    pub fn to_glb(self) -> Vec<u8> {
        export::geometry_to_glb(&self.to_geometry_exact())
    }
}

/// Encode a triangle mesh as binary STL (80-byte header, `u32` count, then per
/// facet: normal + 3 vertices + `u16` attribute). Handles indexed or soup input.
pub fn geometry_to_stl(g: &BufferGeometry) -> Vec<u8> {
    let pos = match g.get_attribute("position") {
        Some(a) => &a.array,
        None => return Vec::new(),
    };
    let vert = |i: usize| [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
    let mut tris: Vec<[[f32; 3]; 3]> = Vec::new();
    if let Some(idx) = &g.index {
        for t in idx.chunks_exact(3) {
            tris.push([vert(t[0] as usize), vert(t[1] as usize), vert(t[2] as usize)]);
        }
    } else {
        for k in 0..pos.len() / 9 {
            tris.push([vert(k * 3), vert(k * 3 + 1), vert(k * 3 + 2)]);
        }
    }

    let mut out = Vec::with_capacity(84 + tris.len() * 50);
    out.extend_from_slice(&[0u8; 80]); // header (must not start with "solid")
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for t in &tris {
        let (a, b, c) = (t[0], t[1], t[2]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let mut n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 0.0 {
            n = [n[0] / len, n[1] / len, n[2] / len];
        }
        for x in n {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for vtx in t {
            for x in vtx {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

/// Identity pass-through — the [`scad!`](crate::scad) escape hatch for splicing a
/// pre-built [`Solid`] value (a variable, or the result of Rust control flow)
/// into a declarative tree: `scad! { union() { solid(part); cube([2.,2.,2.]); } }`.
pub fn solid(s: Solid) -> Solid {
    s
}

/// n-ary union of a list of solids.
pub fn union_all(solids: Vec<Solid>) -> Solid {
    Solid::Union(solids)
}

/// **Convex hull** of one or more solids (OpenSCAD `hull()`): the smallest convex
/// solid enclosing all their vertices. Falls back to the plain union if a hull
/// can't be formed (fewer than 4 non-coplanar points).
pub fn hull(solids: Vec<Solid>) -> Solid {
    let mut pts = Vec::new();
    for s in &solids {
        pts.extend(scad::solid_points(s));
    }
    scad::hull3d(&pts).unwrap_or_else(|| Solid::Union(solids))
}

/// n-ary difference: `head − rest…`.
pub fn difference_all(solids: Vec<Solid>) -> Solid {
    Solid::Difference(solids)
}

/// n-ary intersection of a list of solids.
pub fn intersection_all(solids: Vec<Solid>) -> Solid {
    Solid::Intersection(solids)
}

// ---------------------------------------------------------------------------
// Lowering: Solid tree -> BufferGeometry via the float CsgEvaluator
// ---------------------------------------------------------------------------

/// Install (once) a panic hook that stays silent for the CSG worker thread, so
/// controlled fallbacks don't spew a scary panic message while the default hook
/// still reports real panics on every other thread.
#[cfg(not(target_arch = "wasm32"))]
fn install_quiet_csg_hook() {
    use std::sync::Once;
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if std::thread::current().name() != Some("threers-csg") {
                prev(info);
            }
        }));
    });
}

/// Run a CSG evaluation on a large-stack worker thread that can't take down the
/// process. The float `CsgEvaluator` (three-bvh-csg port) recurses deeply on
/// curved-boolean fallbacks — a 1 GB stack lets legitimate deep cases finish,
/// and a `catch_unwind` turns any residual panic (e.g. a degenerate traversal
/// that the depth guard cut short) into an empty result instead of a crash.
/// On wasm (no threads) it runs inline.
#[cfg(not(target_arch = "wasm32"))]
fn run_csg<T: Send + 'static + Default>(f: impl FnOnce() -> T + Send + 'static) -> T {
    install_quiet_csg_hook();
    std::thread::Builder::new()
        .name("threers-csg".into())
        .stack_size(1024 * 1024 * 1024)
        .spawn(move || std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_default())
        .expect("spawn csg worker thread")
        .join()
        .unwrap_or_default()
}
#[cfg(target_arch = "wasm32")]
fn run_csg<T: Default>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> T {
    std::panic::catch_unwind(f).unwrap_or_default()
}

/// One float boolean that can't crash the process: if the (imperfect) fallback
/// kernel panics, degrade to the un-combined accumulator rather than aborting.
/// (A weld/heal repair does not help the curved-seam cracks the float kernel
/// leaves — see `exact_csg::repair_geometry` — so it is not applied here; the
/// real fix is exact-arithmetic construction.)
fn float_boolean(acc: BufferGeometry, cg: BufferGeometry, op: u8) -> BufferGeometry {
    let fallback = acc.clone();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let mut a = CsgBrush::new(acc);
        a.prepare_geometry();
        let mut b = CsgBrush::new(cg);
        b.prepare_geometry();
        CsgEvaluator::new().evaluate(&mut a, &mut b, op)
    }))
    .unwrap_or(fallback)
}

fn eval_solid(s: Solid, ev: &mut CsgEvaluator) -> BufferGeometry {
    match s {
        Solid::Leaf(g) => g,
        Solid::Transform { matrix, child } => {
            let mut g = eval_solid(*child, ev);
            bake_matrix(&mut g, &matrix);
            g
        }
        Solid::Union(xs) => fold(xs, ev, ADDITION),
        Solid::Difference(xs) => fold(xs, ev, SUBTRACTION),
        Solid::Intersection(xs) => fold(xs, ev, INTERSECTION),
    }
}

/// Left-to-right fold: seed = first child, each subsequent child combined into
/// the accumulator with `op`. This matches OpenSCAD's n-ary boolean semantics.
fn fold(children: Vec<Solid>, ev: &mut CsgEvaluator, op: u8) -> BufferGeometry {
    let mut it = children.into_iter();
    let seed = match it.next() {
        Some(s) => eval_solid(s, ev),
        None => return BufferGeometry::new(),
    };
    let mut acc = CsgBrush::new(seed);
    acc.prepare_geometry();
    for child in it {
        let mut b = CsgBrush::new(eval_solid(child, ev));
        b.prepare_geometry();
        let out = ev.evaluate(&mut acc, &mut b, op);
        acc = CsgBrush::new(out);
        acc.prepare_geometry();
    }
    acc.geometry
}

fn eval_exact(s: Solid) -> BufferGeometry {
    match s {
        Solid::Leaf(g) => g,
        Solid::Transform { matrix, child } => {
            let mut g = eval_exact(*child);
            bake_matrix(&mut g, &matrix);
            g
        }
        Solid::Union(xs) => fold_exact(xs, ExactOp::Union, ADDITION),
        Solid::Difference(xs) => fold_exact(xs, ExactOp::Difference, SUBTRACTION),
        Solid::Intersection(xs) => fold_exact(xs, ExactOp::Intersection, INTERSECTION),
    }
}

/// Like [`fold`], but try the arrangement kernel for each step first and fall
/// back to the (crash-safe) float evaluator when it returns `NeedsArrangement`.
fn fold_exact(children: Vec<Solid>, exact_op: ExactOp, float_op: u8) -> BufferGeometry {
    let mut it = children.into_iter();
    let mut acc = match it.next() {
        Some(s) => eval_exact(s),
        None => return BufferGeometry::new(),
    };
    for child in it {
        let cg = eval_exact(child);
        acc = match exact_boolean(&acc, &cg, exact_op) {
            BooleanOutcome::Exact(g) => g,
            BooleanOutcome::NeedsArrangement => float_boolean(acc, cg, float_op),
        };
    }
    acc
}

/// Bake an affine matrix into a geometry's positions and recompute normals.
/// Mirror transforms (negative determinant) flip triangle winding so normals
/// stay outward. This keeps arbitrary rotation/scale off the parity-tuned CSG
/// world-matrix path (which is translation-only).
pub(crate) fn bake_matrix(geom: &mut BufferGeometry, m: &Matrix4) {
    if let Some(attr) = geom.attributes.get_mut("position") {
        for v in attr.array.chunks_exact_mut(3) {
            let p = Vector3::new(v[0], v[1], v[2]).apply_matrix4(m);
            v[0] = p.x;
            v[1] = p.y;
            v[2] = p.z;
        }
    }
    if m.determinant() < 0.0 {
        reverse_winding(geom);
    }
    geom.bounding_box = None;
    geom.bounding_sphere = None;
    // Recompute normals from the transformed positions (handles rot/scale/mirror
    // without an inverse-transpose). Also bumps geometry_version via set_attribute.
    compute_vertex_normals(geom);
}

fn reverse_winding(geom: &mut BufferGeometry) {
    if let Some(idx) = geom.index.as_mut() {
        for t in idx.chunks_exact_mut(3) {
            t.swap(0, 2);
        }
    } else if let Some(attr) = geom.attributes.get_mut("position") {
        // Non-indexed soup: swap the first and third vertex (3 floats each).
        for tri in attr.array.chunks_exact_mut(9) {
            for k in 0..3 {
                tri.swap(k, 6 + k);
            }
        }
    }
}

// Column-major rotation matrices (three.js `elements` layout: cols 0..3, with
// col 3 = translation). Verified against Vector3::apply_matrix4.
fn rot_x(a: f32) -> Matrix4 {
    let (s, c) = a.sin_cos();
    let mut m = Matrix4::identity();
    m.elements = [
        1.0, 0.0, 0.0, 0.0, // col 0
        0.0, c, s, 0.0, // col 1
        0.0, -s, c, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ];
    m
}

fn rot_y(a: f32) -> Matrix4 {
    let (s, c) = a.sin_cos();
    let mut m = Matrix4::identity();
    m.elements = [
        c, 0.0, -s, 0.0, // col 0
        0.0, 1.0, 0.0, 0.0, // col 1
        s, 0.0, c, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ];
    m
}

fn rot_z(a: f32) -> Matrix4 {
    let (s, c) = a.sin_cos();
    let mut m = Matrix4::identity();
    m.elements = [
        c, s, 0.0, 0.0, // col 0
        -s, c, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ];
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_leaf_has_positions() {
        let g = cube([2.0, 2.0, 2.0]).to_geometry();
        assert!(g.attributes.get("position").unwrap().count() > 0);
    }

    #[test]
    fn difference_produces_geometry() {
        // Sphere fully inside the box -> a hollow; result must be non-empty.
        let g = cube([3.0, 3.0, 3.0]).difference(sphere(1.0)).to_geometry();
        assert!(g.draw_count() > 0);
        assert_eq!(g.draw_count() % 3, 0);
    }

    #[test]
    fn union_and_intersection_nonempty() {
        let u = cube([1.0, 1.0, 1.0])
            .union(cube([1.0, 1.0, 1.0]).translate([0.5, 0.0, 0.0]))
            .to_geometry();
        assert!(u.draw_count() > 0);
        let i = cube([2.0, 2.0, 2.0])
            .intersection(sphere(1.3))
            .to_geometry();
        assert!(i.draw_count() > 0);
    }

    #[test]
    fn translate_moves_bounds() {
        let mut a = cube([2.0, 2.0, 2.0]).to_geometry();
        let mut b = cube([2.0, 2.0, 2.0]).translate([10.0, 0.0, 0.0]).to_geometry();
        let ca = a.compute_bounding_box().center();
        let cb = b.compute_bounding_box().center();
        assert!((cb.x - ca.x - 10.0).abs() < 1e-3, "translate should shift +10 in x");
    }

    #[test]
    fn chained_booleans_flatten() {
        let s = cube([1.0, 1.0, 1.0])
            .union(cube([1.0, 1.0, 1.0]))
            .union(cube([1.0, 1.0, 1.0]));
        match s {
            Solid::Union(ref xs) => assert_eq!(xs.len(), 3),
            _ => panic!("expected a flat 3-ary union"),
        }
    }

    /// Signed volume via the divergence theorem; positive iff faces wind outward.
    fn signed_volume(g: &BufferGeometry) -> f32 {
        let pos = &g.attributes.get("position").unwrap().array;
        let read = |i: u32| {
            let j = i as usize * 3;
            Vector3::new(pos[j], pos[j + 1], pos[j + 2])
        };
        let mut vol = 0.0;
        let mut tri = |a: Vector3, b: Vector3, c: Vector3| vol += a.dot(b.cross(c));
        if let Some(idx) = &g.index {
            for t in idx.chunks_exact(3) {
                tri(read(t[0]), read(t[1]), read(t[2]));
            }
        } else {
            for k in 0..pos.len() / 9 {
                let b = (k * 3) as u32;
                tri(read(b), read(b + 1), read(b + 2));
            }
        }
        vol / 6.0
    }

    #[test]
    fn linear_extrude_square_is_a_box() {
        // Unit square centred on origin, extruded z=0..2 → volume 2, wound outward.
        let sq = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        let mut g = linear_extrude(2.0, &sq).to_geometry();
        assert!((signed_volume(&g) - 2.0).abs() < 1e-4, "outward winding + volume");
        let bb = g.compute_bounding_box();
        assert!((bb.min.z - 0.0).abs() < 1e-6 && (bb.max.z - 2.0).abs() < 1e-6);
    }

    #[test]
    fn linear_extrude_accepts_clockwise_input() {
        // Same square wound CW; normalization must still give positive volume.
        let cw = [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]];
        let g = linear_extrude(1.0, &cw).to_geometry();
        assert!(signed_volume(&g) > 0.0);
    }

    #[test]
    fn linear_extrude_works_as_csg_operand() {
        let sq = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]];
        let g = linear_extrude(3.0, &sq)
            .difference(cylinder(10.0, 1.0).rotate_x(std::f32::consts::FRAC_PI_2))
            .to_geometry();
        assert!(g.draw_count() > 0 && g.draw_count() % 3 == 0);
    }

    #[test]
    fn linear_extrude_with_hole_has_correct_volume() {
        // 4×4 outer, 2×2 hole, height 2 → volume (16−4)·2 = 24, wound outward.
        let outer = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]];
        let hole = vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let g = linear_extrude_holes(2.0, &outer, &[hole]).to_geometry();
        assert!((signed_volume(&g) - 24.0).abs() < 1e-3, "outward winding + hole volume");
    }

    #[test]
    fn linear_extrude_with_two_holes_volume_and_operand() {
        // Two separate holes; volume = (outer − hole1 − hole2)·height, and the
        // result must still be usable as a further CSG operand.
        let outer = [[-4.0, -2.0], [4.0, -2.0], [4.0, 2.0], [-4.0, 2.0]]; // area 32
        let h1 = vec![[-3.0, -1.0], [-1.0, -1.0], [-1.0, 1.0], [-3.0, 1.0]]; // area 4
        let h2 = vec![[1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [1.0, 1.0]]; // area 4
        let plate = linear_extrude_holes(2.0, &outer, &[h1, h2]);
        let g = plate.clone().to_geometry();
        assert!((signed_volume(&g).abs() - (32.0 - 4.0 - 4.0) * 2.0).abs() < 1e-3);
        // Still a valid operand: union with a boss stays non-empty.
        let combined = plate.union(sphere(1.0).translate([0.0, 0.0, 1.0])).to_geometry();
        assert!(combined.draw_count() > 0 && combined.draw_count() % 3 == 0);
    }

    #[test]
    fn rotate_extrude_full_is_nonempty() {
        // Square profile offset from the axis, revolved 360° → a square torus.
        let profile = [[2.0, -0.5], [3.0, -0.5], [3.0, 0.5], [2.0, 0.5]];
        let g = rotate_extrude(std::f32::consts::TAU, &profile).to_geometry();
        assert!(g.draw_count() > 0);
    }

    #[test]
    fn polyhedron_tetrahedron() {
        let pts = [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ];
        let faces = vec![
            vec![0, 1, 2],
            vec![0, 3, 1],
            vec![0, 2, 3],
            vec![1, 3, 2],
        ];
        let g = polyhedron(&pts, &faces).to_geometry();
        // 4 triangular faces -> 12 vertices of position data.
        assert_eq!(g.attributes.get("position").unwrap().count(), 12);
    }
}

#[cfg(test)]
mod exact_integration {
    use super::*;
    use crate::BufferGeometry;

    fn svol(g: &BufferGeometry) -> f64 {
        let pos = &g.attributes.get("position").unwrap().array;
        let v = |i: usize| [pos[i * 3] as f64, pos[i * 3 + 1] as f64, pos[i * 3 + 2] as f64];
        let tri = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| {
            a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
        };
        let mut s = 0.0;
        if let Some(idx) = &g.index {
            for t in idx.chunks_exact(3) {
                s += tri(v(t[0] as usize), v(t[1] as usize), v(t[2] as usize));
            }
        } else {
            for k in 0..pos.len() / 9 {
                s += tri(v(k * 3), v(k * 3 + 1), v(k * 3 + 2));
            }
        }
        s / 6.0
    }

    #[test]
    fn exact_path_matches_float_on_oblique_boxes() {
        // The arrangement resolves oblique boxes, so the exact path's volume must
        // match the float path for all three ops.
        let a = || cube([2.0, 2.0, 2.0]);
        let b = || cube([2.0, 2.0, 2.0]).rotate_x(0.6).rotate_y(0.4).translate([0.7, 0.5, 0.3]);
        for (name, exact, float) in [
            ("diff", a().difference(b()).to_geometry_exact(), a().difference(b()).to_geometry()),
            ("union", a().union(b()).to_geometry_exact(), a().union(b()).to_geometry()),
            ("inter", a().intersection(b()).to_geometry_exact(), a().intersection(b()).to_geometry()),
        ] {
            let (ve, vf) = (svol(&exact).abs(), svol(&float).abs());
            assert!((ve - vf).abs() / vf < 1e-2, "{name}: exact {ve:.4} vs float {vf:.4}");
        }
    }

    fn tri_count(g: &BufferGeometry) -> usize {
        match &g.index {
            Some(idx) => idx.len() / 3,
            None => g.attributes.get("position").map(|a| a.count() / 3).unwrap_or(0),
        }
    }

    #[test]
    fn stl_export_roundtrips_through_the_loader() {
        // Export a Solid to binary STL, parse it back with the repo's StlLoader,
        // and confirm triangle count and bounds survive.
        let solid = cube([3.0, 2.0, 1.0]).difference(sphere(0.7));
        let mut geom = solid.to_geometry();
        let stl = geometry_to_stl(&geom);
        assert_eq!(stl.len(), 84 + tri_count(&geom) * 50, "binary STL size");

        let mut back = crate::StlLoader::parse_binary(&stl);
        assert_eq!(tri_count(&geom), tri_count(&back), "triangle count survives");

        let (ba, bb) = (geom.compute_bounding_box(), back.compute_bounding_box());
        for k in 0..3 {
            let (amin, bmin) = ([ba.min.x, ba.min.y, ba.min.z][k], [bb.min.x, bb.min.y, bb.min.z][k]);
            let (amax, bmax) = ([ba.max.x, ba.max.y, ba.max.z][k], [bb.max.x, bb.max.y, bb.max.z][k]);
            assert!((amin - bmin).abs() < 1e-4 && (amax - bmax).abs() < 1e-4, "bounds axis {k}");
        }
    }

    #[test]
    fn exact_path_handles_the_full_bracket() {
        // Hybrid path (arrangement + float fallback) must yield a valid non-empty
        // mesh for the full bracket, whatever each step chooses.
        let g = cube([40.0, 24.0, 4.0])
            .difference(cylinder(20.0, 2.5).rotate_x(std::f32::consts::FRAC_PI_2).translate([-12.0, 0.0, 0.0]))
            .difference(cylinder(20.0, 2.5).rotate_x(std::f32::consts::FRAC_PI_2).translate([12.0, 0.0, 0.0]))
            .union(sphere(5.0).translate([0.0, 0.0, 2.0]))
            .to_geometry_exact();
        assert!(g.draw_count() > 0 && g.draw_count() % 3 == 0);
    }
}

/// Build a parametric Cartesian 3D printer as a list of coloured parts. Each part
/// is a self-contained primitive (no cross-part boolean), so it renders instantly
/// and parts may freely overlap to look connected — a machine is a *scene* of
/// bodies, not one welded solid. Shared by `examples/printer_assembly.rs` and the
/// wasm/browser viewer. Lengths in mm; `gantry_z`/`carriage`/`bed_pos` pose Z/X/Y.
#[allow(clippy::too_many_arguments)]
pub fn build_printer(
    bed_x: f32,
    bed_y: f32,
    z_travel: f32,
    ext: f32,
    gantry_z: f32,
    carriage: f32,
    bed_pos: f32,
) -> Vec<(BufferGeometry, [f32; 3])> {
    use std::f32::consts::{FRAC_PI_2, PI};

    let c_ext = [0.75, 0.77, 0.81];
    let c_motor = [0.10, 0.10, 0.12];
    let c_metal = [0.85, 0.87, 0.90];
    let c_lead = [0.80, 0.66, 0.28];
    let c_bed = [0.14, 0.15, 0.19];
    let c_part = [0.20, 0.55, 0.95];
    let c_belt = [0.05, 0.05, 0.06];
    let c_print = [0.93, 0.49, 0.18];
    let c_box = [0.18, 0.20, 0.25];
    let c_dark = [0.12, 0.13, 0.16];

    // Primitive helpers. `beam` runs +X (centred in Y,Z); `cz`/`taper` run +Z from
    // 0..h (the library cylinder/cone are Y-axis & centred).
    let beam = |l: f32, w: f32| cube([l, w, w]).translate([l / 2.0, 0.0, 0.0]);
    let cz = |h: f32, r: f32| frustum(h, r, r, 48).rotate_x(FRAC_PI_2).translate([0.0, 0.0, h / 2.0]);
    let taper = |h: f32, r1: f32, r2: f32| cone(h, r1, r2).rotate_x(FRAC_PI_2).translate([0.0, 0.0, h / 2.0]);

    let mut parts: Vec<(BufferGeometry, [f32; 3])> = Vec::new();
    macro_rules! p {
        ($s:expr, $c:expr) => {
            parts.push(($s.to_geometry(), $c));
        };
    }
    // A NEMA 17 = body + boss + 5 mm shaft + 4 bolt heads + rear cap, positioned by
    // `place` (a transform applied to each sub-solid pointing +Z before placement).
    macro_rules! nema {
        ($len:expr, $place:expr) => {{
            let (b, len, place) = (42.3f32, $len, $place);
            p!(place(cube([b, b, len]).translate([0.0, 0.0, len / 2.0])), c_motor);
            p!(place(cube([b - 4.0, b - 4.0, 2.0]).translate([0.0, 0.0, len + 1.0])), c_dark); // face plate
            p!(place(cz(2.5, 11.0).translate([0.0, 0.0, len + 2.0])), c_metal); // boss
            p!(place(cz(22.0, 2.5).translate([0.0, 0.0, len + 4.0])), c_metal); // shaft
            for sx in [-1.0f32, 1.0] {
                for sy in [-1.0f32, 1.0] {
                    p!(place(cz(3.0, 2.2).translate([sx * 15.5, sy * 15.5, len + 2.0])), c_metal); // bolt heads
                }
            }
            p!(place(cz(4.0, 15.0).translate([0.0, 0.0, -4.0])), c_dark); // rear cap
        }};
    }

    let (xl, xr, yb) = (-30.0f32, bed_x + 30.0, bed_y + 30.0);
    let (fw, fd, z0) = (xr - xl, yb + 30.0, ext / 2.0);
    let yf = yb - ext; // gantry / motor-mount plane (front face of the uprights)

    // ---- Frame ----
    p!(beam(fw, ext).translate([xl, -30.0, z0]), c_ext); // base front
    p!(beam(fw, ext).translate([xl, yb, z0]), c_ext); // base back
    p!(beam(fd, ext).rotate_z(FRAC_PI_2).translate([xl, -30.0, z0]), c_ext); // base left
    p!(beam(fd, ext).rotate_z(FRAC_PI_2).translate([xr, -30.0, z0]), c_ext); // base right
    p!(beam(z_travel, ext).rotate_y(-FRAC_PI_2).translate([xl, yb, z0]), c_ext); // upright L
    p!(beam(z_travel, ext).rotate_y(-FRAC_PI_2).translate([xr, yb, z0]), c_ext); // upright R
    p!(beam(fw, ext).translate([xl, yb, z_travel]), c_ext); // top cross-bar
    for (fx, fy) in [(xl, -30.0), (xr, -30.0), (xl, yb), (xr, yb)] {
        p!(cz(6.0, ext * 0.55).translate([fx, fy, -6.0]), c_dark); // feet
    }

    // ---- Z axis: motor (bottom-left) + coupler + lead screw, smooth rod right ----
    nema!(40.0, |s: Solid| s.translate([xl, yf, ext + 2.0]));
    p!(cz(12.0, 5.0).translate([xl, yf, ext + 46.0]), c_metal); // coupler
    p!(cz((z_travel - ext - 60.0).max(10.0), 3.5).translate([xl, yf, ext + 58.0]), c_lead); // lead screw
    p!(cz((z_travel - ext).max(10.0), 3.0).translate([xr, yf, ext]), c_metal); // smooth rod (right)

    // ---- Gantry (X rail) + X-ends + carriage + hot-end + X/E motors ----
    p!(beam(fw, ext).translate([xl, yf, gantry_z]), c_ext); // X rail
    p!(cube([ext + 10.0, ext + 12.0, 26.0]).translate([xl, yf, gantry_z]), c_part); // left X-end
    p!(cube([ext + 10.0, ext + 12.0, 26.0]).translate([xr, yf, gantry_z]), c_part); // right X-end
    p!(cube([fw - 24.0, 2.0, 5.0]).translate([bed_x / 2.0, yf - 12.0, gantry_z + ext / 2.0 - 2.0]), c_belt); // X belt
    let cx = xl + carriage.clamp(0.0, bed_x);
    p!(cube([32.0, 26.0, 30.0]).translate([cx, yf - 12.0, gantry_z]), c_part); // carriage
    p!(cz(6.0, 5.0).rotate_y(FRAC_PI_2).translate([xl - 7.0, yf, gantry_z]), c_metal); // idler pulley (left end)
    p!(cz(16.0, 8.0).translate([cx, yf - 18.0, gantry_z - 24.0]), c_metal); // hot-end heatsink
    p!(cube([16.0, 14.0, 12.0]).translate([cx, yf - 18.0, gantry_z - 26.0]), c_box); // heater block
    p!(taper(9.0, 4.0, 1.2).rotate_x(PI).translate([cx, yf - 18.0, gantry_z - 34.0]), c_metal); // nozzle
    nema!(34.0, |s: Solid| s.rotate_y(-FRAC_PI_2).translate([xr + 10.0, yf, gantry_z])); // X motor
    nema!(28.0, |s: Solid| s.rotate_x(-FRAC_PI_2).translate([cx, yf - 2.0, gantry_z + 8.0])); // extruder motor

    // ---- Bed + Y carriage + Y motor ----
    let yc = bed_y / 2.0 + bed_pos;
    p!(cube([bed_x + 10.0, bed_y + 10.0, 3.0]).translate([bed_x / 2.0, yc, ext + 10.0]), c_bed); // heated bed
    p!(cube([bed_x * 0.72, bed_y * 0.72, 4.0]).translate([bed_x / 2.0, yc, ext + 5.0]), c_metal); // Y carriage
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            p!(cz(7.0, 4.0).translate([bed_x / 2.0 + sx * bed_x * 0.33, yc + sy * bed_y * 0.33, ext - 1.0]), c_metal); // levelling knobs
        }
    }
    p!(cube([2.0, fd - 24.0, 5.0]).translate([bed_x / 2.0, (fd - 24.0) / 2.0 - 30.0, ext + 2.0]), c_belt); // Y belt
    nema!(38.0, |s: Solid| s.rotate_x(-FRAC_PI_2).translate([bed_x / 2.0, -30.0, ext + 2.0])); // Y motor

    // ---- Electronics box + filament spool on a holder ----
    p!(cube([70.0, 40.0, 46.0]).translate([xr + 34.0, bed_y / 2.0, 24.0]), c_box); // control box
    p!(cz(16.0, 42.0).rotate_y(FRAC_PI_2).translate([bed_x / 2.0 + 60.0, yb + 6.0, z_travel + 30.0]), c_print); // spool
    p!(cz(80.0, 6.0).rotate_y(FRAC_PI_2).translate([bed_x / 2.0 - 30.0, yb + 6.0, z_travel + 30.0]), c_metal); // spool rod
    p!(beam(46.0, ext).rotate_y(-FRAC_PI_2).translate([bed_x / 2.0 - 30.0, yb, z_travel]), c_ext); // spool arm

    // ---- The in-progress print on the bed ----
    p!(cz(26.0, 15.0).translate([bed_x / 2.0, yc, ext + 12.0]), c_print);

    parts
}

/// Build a detailed **NEMA 17 stepper motor** as a list of coloured parts (all
/// primitives — no boolean): the two aluminium end-caps, the inset black
/// lamination stack that leaves the four corner tie-rods proud, the front boss and
/// 5 mm shaft, four countersunk mounting holes, and a side wiring connector with
/// four coloured leads. Shaft points +Z; body spans z = 0..42 mm. Shared by the
/// native example and the wasm/browser viewer.
pub fn build_nema17() -> Vec<(BufferGeometry, [f32; 3])> {
    use std::f32::consts::FRAC_PI_2;

    let c_cap = [0.80, 0.82, 0.86];
    let c_stack = [0.09, 0.09, 0.11];
    let c_rod = [0.55, 0.57, 0.61];
    let c_metal = [0.88, 0.90, 0.93];
    let c_hole = [0.04, 0.04, 0.05];
    let c_conn = [0.10, 0.10, 0.12];
    let wires = [[0.85, 0.15, 0.15], [0.15, 0.70, 0.25], [0.20, 0.45, 0.92], [0.85, 0.80, 0.20]];

    // +Z cylinder from 0..h (library cylinder is Y-axis, centred).
    let cz = |h: f32, r: f32| frustum(h, r, r, 48).rotate_x(FRAC_PI_2).translate([0.0, 0.0, h / 2.0]);
    // A cylinder along −Y of length h, starting at y0 (for the wire leads).
    let wy = |h: f32, r: f32, x: f32, y0: f32, z: f32| frustum(h, r, r, 18).translate([x, y0 - h / 2.0, z]);

    let mut parts: Vec<(BufferGeometry, [f32; 3])> = Vec::new();
    macro_rules! p {
        ($s:expr, $c:expr) => {
            parts.push(($s.to_geometry(), $c));
        };
    }

    let b = 42.3; // body cross-section
    let cap = 8.0; // aluminium end-cap thickness
    let body = 42.0; // overall body length
    let inset = 2.2; // how far the lamination stack is set in from the caps

    // End-caps + inset lamination stack (the stack sits back so the corner tie-rods
    // and the chamfer show, as on a real motor).
    p!(cube([b, b, cap]).translate([0.0, 0.0, cap / 2.0]), c_cap); // rear cap
    p!(cube([b - 2.0 * inset, b - 2.0 * inset, body - 2.0 * cap + 6.0]).translate([0.0, 0.0, body / 2.0]), c_stack);
    p!(cube([b, b, cap]).translate([0.0, 0.0, body - cap / 2.0]), c_cap); // front cap

    // Four corner through-bolt tie-rods.
    let tr = b / 2.0 - 3.3;
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            p!(cz(body, 2.0).translate([sx * tr, sy * tr, 0.0]), c_rod);
        }
    }

    // Four countersunk M3 mounting holes on the front face (31 mm square).
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            p!(cz(3.5, 2.6).translate([sx * 15.5, sy * 15.5, body - 3.0]), c_hole);
        }
    }

    // Front boss + 5 mm output shaft.
    p!(cz(2.5, 11.0).translate([0.0, 0.0, body]), c_metal);
    p!(cz(24.0, 2.5).translate([0.0, 0.0, body + 2.0]), c_metal);

    // Side wiring connector + four coloured leads.
    p!(cube([15.0, 5.0, 8.0]).translate([0.0, -b / 2.0 - 1.5, cap + 2.0]), c_conn);
    for (i, wc) in wires.iter().enumerate() {
        let x = (i as f32 - 1.5) * 3.2;
        p!(wy(22.0, 1.0, x, -b / 2.0 - 3.5, cap + 2.0), *wc);
    }

    parts
}
