//! Bindings split out of `wasm.rs` by area (see the `mod` list there): the
//! geometry generators the shim's `LatheGeometry`, `ConvexGeometry` / `ConvexHull`, `DecalGeometry`, `RoundedBoxGeometry`,
//! `TeapotGeometry`, `EdgesGeometry` and `WireframeGeometry` wrap.
//!
//! Parameters are `f64` because JavaScript's numbers are: the generators
//! compute in `f64` as three.js does, which is what makes their buffers match
//! r165's bit for bit.
#![allow(unused_imports)]

use super::*;
use crate::geometries::{
    ConvexGeometry, ConvexHull, DecalGeometry, EdgesGeometry, EulerOrder, LatheGeometry,
    RoundedBoxGeometry, TeapotGeometry, TeapotOptions, WireframeGeometry,
};

fn web(g: crate::core::BufferGeometry) -> WebGeometry {
    WebGeometry { inner: Arc::new(g) }
}

fn triples(flat: &[f64]) -> Vec<[f64; 3]> {
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

#[wasm_bindgen]
impl WebGeometry {
    /// A three.js primitive by class name, from its constructor's arguments
    /// as JavaScript passed them (`openEnded` as 0 or 1), built in f64 so the
    /// buffers are r165's bit for bit (`tests/primitives_upstream.rs`).
    /// Arguments left off take three.js's defaults.
    #[wasm_bindgen(js_name = threePrimitive)]
    pub fn three_primitive(name: &str, args: &[f64]) -> Result<WebGeometry, JsValue> {
        use crate::geometries::{
            BoxGeometry, CapsuleGeometry, CircleGeometry, ConeGeometry, CylinderGeometry,
            DodecahedronGeometry, IcosahedronGeometry, OctahedronGeometry, PlaneGeometry,
            RingGeometry, SphereGeometry, TetrahedronGeometry, TorusGeometry, TorusKnotGeometry,
        };
        use std::f64::consts::PI;
        let a = |i: usize, default: f64| args.get(i).copied().filter(|v| !v.is_nan()).unwrap_or(default);
        // three.js `Math.floor` on every segment count.
        let n = |i: usize, default: f64| a(i, default).floor().max(0.0) as usize;
        let g = match name {
            "BoxGeometry" => BoxGeometry::from_f64(
                a(0, 1.0), a(1, 1.0), a(2, 1.0), n(3, 1.0) as u32, n(4, 1.0) as u32, n(5, 1.0) as u32,
            ),
            "SphereGeometry" => SphereGeometry::from_f64(
                a(0, 1.0), n(1, 32.0), n(2, 16.0), a(3, 0.0), a(4, PI * 2.0), a(5, 0.0), a(6, PI),
            ),
            "CylinderGeometry" => CylinderGeometry::from_f64(
                a(0, 1.0), a(1, 1.0), a(2, 1.0), n(3, 32.0), n(4, 1.0), a(5, 0.0) != 0.0, a(6, 0.0), a(7, PI * 2.0),
            ),
            "ConeGeometry" => ConeGeometry::from_f64(
                a(0, 1.0), a(1, 1.0), n(2, 32.0), n(3, 1.0), a(4, 0.0) != 0.0, a(5, 0.0), a(6, PI * 2.0),
            ),
            "PlaneGeometry" => PlaneGeometry::from_f64(a(0, 1.0), a(1, 1.0), n(2, 1.0), n(3, 1.0)),
            "CircleGeometry" => CircleGeometry::from_f64(a(0, 1.0), n(1, 32.0), a(2, 0.0), a(3, PI * 2.0)),
            "RingGeometry" => RingGeometry::from_f64(a(0, 0.5), a(1, 1.0), n(2, 32.0), n(3, 1.0), a(4, 0.0), a(5, PI * 2.0)),
            "TorusGeometry" => TorusGeometry::from_f64(a(0, 1.0), a(1, 0.4), n(2, 12.0), n(3, 48.0), a(4, PI * 2.0)),
            "TorusKnotGeometry" => TorusKnotGeometry::from_f64(
                a(0, 1.0), a(1, 0.4), n(2, 64.0), n(3, 8.0), n(4, 2.0) as u32, n(5, 3.0) as u32,
            ),
            "TetrahedronGeometry" => TetrahedronGeometry::from_f64(a(0, 1.0), n(1, 0.0)),
            "OctahedronGeometry" => OctahedronGeometry::from_f64(a(0, 1.0), n(1, 0.0)),
            "IcosahedronGeometry" => IcosahedronGeometry::from_f64(a(0, 1.0), n(1, 0.0)),
            "DodecahedronGeometry" => DodecahedronGeometry::from_f64(a(0, 1.0), n(1, 0.0)),
            "CapsuleGeometry" => CapsuleGeometry::with_f64(a(0, 1.0), a(1, 1.0), n(2, 4.0), n(3, 8.0)),
            other => return Err(JsValue::from_str(&format!("threePrimitive: no primitive '{other}'"))),
        };
        Ok(web(g))
    }

    /// three.js `LatheGeometry(points, segments, phiStart, phiLength)`;
    /// `points` flattened `[x0, y0, x1, y1, …]`.
    #[wasm_bindgen(js_name = lathe)]
    pub fn lathe(points: &[f64], segments: usize, phi_start: f64, phi_length: f64) -> WebGeometry {
        let pts: Vec<[f64; 2]> = points.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
        web(LatheGeometry::with_f64(&pts, segments, phi_start, phi_length))
    }

    /// three.js `ConvexGeometry(points)`, `points` flattened xyz.
    #[wasm_bindgen(js_name = convex)]
    pub fn convex(points: &[f64]) -> WebGeometry {
        web(ConvexGeometry::from_f64(&triples(points)))
    }

    /// three.js `DecalGeometry(mesh, position, orientation, size)`: the mesh as
    /// its geometry and `matrixWorld` (16 numbers, column-major), the
    /// orientation as Euler angles plus their order (`"XYZ"`…).
    #[wasm_bindgen(js_name = decal)]
    pub fn decal(
        source: &WebBufferGeometry,
        matrix_world: &[f64],
        position: &[f64],
        orientation: &[f64],
        order: &str,
        size: &[f64],
    ) -> Result<WebGeometry, JsValue> {
        let three = |a: &[f64], what: &str| -> Result<[f64; 3], JsValue> {
            a.get(..3)
                .map(|s| [s[0], s[1], s[2]])
                .ok_or_else(|| JsValue::from_str(&format!("decal: {what} needs 3 numbers")))
        };
        let mw: [f64; 16] = matrix_world
            .try_into()
            .map_err(|_| JsValue::from_str("decal: matrixWorld needs 16 numbers"))?;
        let g = &source.inner;
        let pos = g
            .get_attribute("position")
            .ok_or_else(|| JsValue::from_str("decal: the mesh has no position attribute"))?;
        let empty = Vec::new();
        let nrm = g.get_attribute("normal").map(|a| &a.array).unwrap_or(&empty);
        Ok(web(DecalGeometry::project(
            &pos.array,
            nrm,
            g.index.as_deref(),
            &mw,
            three(position, "position")?,
            three(orientation, "orientation")?,
            EulerOrder::parse(order),
            three(size, "size")?,
        )))
    }

    /// three.js addon `RoundedBoxGeometry(width, height, depth, segments, radius)`.
    #[wasm_bindgen(js_name = roundedBox)]
    pub fn rounded_box(width: f64, height: f64, depth: f64, segments: usize, radius: f64) -> WebGeometry {
        web(RoundedBoxGeometry::new(width, height, depth, segments, radius))
    }

    /// three.js addon `TeapotGeometry(size, segments, bottom, lid, body, fitLid, blinn)`.
    #[wasm_bindgen(js_name = teapot)]
    pub fn teapot(
        size: f64,
        segments: usize,
        bottom: bool,
        lid: bool,
        body: bool,
        fit_lid: bool,
        blinn: bool,
    ) -> WebGeometry {
        web(TeapotGeometry::new(TeapotOptions {
            size,
            segments,
            bottom,
            lid,
            body,
            fit_lid,
            blinn,
        }))
    }

    /// three.js `BoxGeometry` with segments, exactly as r165 builds it:
    /// indexed, six groups, triangles (no quad polygons).
    #[wasm_bindgen(js_name = threeBox)]
    pub fn three_box(
        width: f64,
        height: f64,
        depth: f64,
        width_segments: usize,
        height_segments: usize,
        depth_segments: usize,
    ) -> WebGeometry {
        web(crate::geometries::three_box_geometry(
            width,
            height,
            depth,
            [width_segments, height_segments, depth_segments],
        ))
    }

    /// three.js `EdgesGeometry(geometry, thresholdAngle)`, the angle in
    /// degrees. A geometry carrying polygons never gets an edge inside one.
    #[wasm_bindgen(js_name = edges)]
    pub fn edges(source: &WebBufferGeometry, threshold_angle: f64) -> WebGeometry {
        web(EdgesGeometry::with_threshold_degrees(&source.inner, threshold_angle))
    }

    /// three.js `WireframeGeometry(geometry)`; a geometry carrying polygons
    /// draws their outlines instead of the triangulation.
    #[wasm_bindgen(js_name = wireframe)]
    pub fn wireframe(source: &WebBufferGeometry) -> WebGeometry {
        web(WireframeGeometry::new(&source.inner))
    }
}

#[wasm_bindgen]
impl WebBufferGeometry {
    /// three.js `groups`, flattened `[start, count, materialIndex, …]`;
    /// replaces any there were.
    #[wasm_bindgen(js_name = setGroups)]
    pub fn set_groups(&mut self, groups: &[u32]) {
        let g = Arc::make_mut(&mut self.inner);
        g.clear_groups();
        for c in groups.chunks_exact(3) {
            g.add_group(c[0], c[1], c[2]);
        }
    }
}

/// three.js addon `ConvexHull` (quickhull3d), the hull itself; the shim's
/// `ConvexHull` class reads its faces back into three.js-shaped objects.
#[wasm_bindgen]
pub struct WebConvexHull {
    inner: ConvexHull,
}

#[wasm_bindgen]
impl WebConvexHull {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebConvexHull {
        WebConvexHull {
            inner: ConvexHull::new(),
        }
    }

    /// `setFromPoints`, `points` flattened xyz. Fewer than four points leave
    /// the hull unchanged, as in three.js.
    #[wasm_bindgen(js_name = setFromPoints)]
    pub fn set_from_points(&mut self, points: &[f64]) {
        self.inner.set_from_points(&triples(points));
    }

    #[wasm_bindgen(js_name = makeEmpty)]
    pub fn make_empty(&mut self) {
        self.inner.make_empty();
    }

    #[wasm_bindgen(getter)]
    pub fn tolerance(&self) -> f64 {
        self.inner.tolerance
    }

    /// Each face's corners as input-point indices, flattened as
    /// `[n, i0, …, i(n-1), n, …]`, in three.js's face order and edge order.
    #[wasm_bindgen(js_name = faceVertexIndices)]
    pub fn face_vertex_indices(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for f in self.inner.face_vertex_indices() {
            out.push(f.len() as u32);
            out.extend(f.iter().map(|&i| i as u32));
        }
        out
    }

    /// Per face `[nx, ny, nz, mx, my, mz, constant, area]`: normal, midpoint,
    /// plane constant and area.
    #[wasm_bindgen(js_name = faceData)]
    pub fn face_data(&self) -> Vec<f64> {
        self.inner
            .faces()
            .flat_map(|f| {
                [
                    f.normal[0],
                    f.normal[1],
                    f.normal[2],
                    f.midpoint[0],
                    f.midpoint[1],
                    f.midpoint[2],
                    f.constant,
                    f.area,
                ]
            })
            .collect()
    }

    #[wasm_bindgen(js_name = containsPoint)]
    pub fn contains_point(&self, x: f64, y: f64, z: f64) -> bool {
        self.inner.contains_point([x, y, z])
    }

    /// `intersectRay`: the distance along the (unit) direction to the hull,
    /// or `undefined` when the ray misses.
    #[wasm_bindgen(js_name = intersectRay)]
    pub fn intersect_ray(&self, ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64) -> Option<f64> {
        self.inner.intersect_ray([ox, oy, oz], [dx, dy, dz])
    }

    /// The hull as three.js's `ConvexGeometry` would build it.
    #[wasm_bindgen(js_name = toGeometry)]
    pub fn to_geometry(&self) -> WebGeometry {
        web(ConvexGeometry::from_hull(&self.inner))
    }
}
