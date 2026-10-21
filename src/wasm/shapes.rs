//! Bindings split out of `wasm.rs` by area (see the `mod` list there): the
//! shim's `Path`, `Shape`, `ShapePath`, `ShapeUtils`, `ShapeGeometry`,
//! `ExtrudeGeometry`, `Font` and `TextGeometry` — three.js r165's 2D shape
//! stack from [`crate::curves::shapes`], computed in `f64` as JavaScript does.
#![allow(unused_imports)]

use super::*;
use crate::curves::shapes::{self, shape_utils, ShapeCurve, Vec2d};
use crate::geometries::{ExtrudeFace, ExtrudeGeometry, ExtrudeOptions, ExtrudePath, ShapeGeometry};

fn pairs(flat: &[f64]) -> Vec<Vec2d> {
    flat.chunks_exact(2)
        .map(|c| Vec2d::new(c[0], c[1]))
        .collect()
}

fn flat(points: &[Vec2d]) -> Vec<f64> {
    points.iter().flat_map(|p| [p.x, p.y]).collect()
}

/// A three.js `Path` (the shim's `Path`, and the outline of its `Shape`).
#[wasm_bindgen]
#[derive(Clone)]
pub struct WebShapeCurvePath {
    pub(crate) inner: shapes::Path,
}

#[wasm_bindgen]
impl WebShapeCurvePath {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebShapeCurvePath {
        WebShapeCurvePath {
            inner: shapes::Path::new(),
        }
    }

    /// `new Path(points)` / `setFromPoints`, points flattened `[x0, y0, …]`.
    #[wasm_bindgen(js_name = setFromPoints)]
    pub fn set_from_points(&mut self, points: &[f64]) {
        self.inner.set_from_points(&pairs(points));
    }
    #[wasm_bindgen(js_name = moveTo)]
    pub fn move_to(&mut self, x: f64, y: f64) {
        self.inner.move_to(x, y);
    }
    #[wasm_bindgen(js_name = lineTo)]
    pub fn line_to(&mut self, x: f64, y: f64) {
        self.inner.line_to(x, y);
    }
    #[wasm_bindgen(js_name = quadraticCurveTo)]
    pub fn quadratic_curve_to(&mut self, cpx: f64, cpy: f64, x: f64, y: f64) {
        self.inner.quadratic_curve_to(cpx, cpy, x, y);
    }
    #[wasm_bindgen(js_name = bezierCurveTo)]
    pub fn bezier_curve_to(&mut self, c1x: f64, c1y: f64, c2x: f64, c2y: f64, x: f64, y: f64) {
        self.inner.bezier_curve_to(c1x, c1y, c2x, c2y, x, y);
    }
    /// `splineThru(pts)`, points flattened.
    #[wasm_bindgen(js_name = splineThru)]
    pub fn spline_thru(&mut self, points: &[f64]) {
        self.inner.spline_thru(&pairs(points));
    }
    pub fn arc(&mut self, x: f64, y: f64, radius: f64, start: f64, end: f64, clockwise: bool) {
        self.inner.arc(x, y, radius, start, end, clockwise);
    }
    pub fn absarc(&mut self, x: f64, y: f64, radius: f64, start: f64, end: f64, clockwise: bool) {
        self.inner.absarc(x, y, radius, start, end, clockwise);
    }
    pub fn ellipse(
        &mut self,
        x: f64,
        y: f64,
        x_radius: f64,
        y_radius: f64,
        start: f64,
        end: f64,
        clockwise: bool,
        rotation: f64,
    ) {
        self.inner
            .ellipse(x, y, x_radius, y_radius, start, end, clockwise, rotation);
    }
    pub fn absellipse(
        &mut self,
        x: f64,
        y: f64,
        x_radius: f64,
        y_radius: f64,
        start: f64,
        end: f64,
        clockwise: bool,
        rotation: f64,
    ) {
        self.inner
            .absellipse(x, y, x_radius, y_radius, start, end, clockwise, rotation);
    }
    #[wasm_bindgen(js_name = closePath)]
    pub fn close_path(&mut self) {
        self.inner.close_path();
    }

    /// `CurvePath.add(curve)` for each curve type, the pen untouched.
    #[wasm_bindgen(js_name = addLineCurve)]
    pub fn add_line_curve(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) {
        self.inner.add(ShapeCurve::Line {
            v1: Vec2d::new(x1, y1),
            v2: Vec2d::new(x2, y2),
        });
    }
    #[wasm_bindgen(js_name = addQuadraticCurve)]
    pub fn add_quadratic_curve(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) {
        self.inner.add(ShapeCurve::QuadraticBezier {
            v0: Vec2d::new(x0, y0),
            v1: Vec2d::new(x1, y1),
            v2: Vec2d::new(x2, y2),
        });
    }
    #[wasm_bindgen(js_name = addCubicCurve)]
    pub fn add_cubic_curve(
        &mut self,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        x3: f64,
        y3: f64,
    ) {
        self.inner.add(ShapeCurve::CubicBezier {
            v0: Vec2d::new(x0, y0),
            v1: Vec2d::new(x1, y1),
            v2: Vec2d::new(x2, y2),
            v3: Vec2d::new(x3, y3),
        });
    }
    #[wasm_bindgen(js_name = addSplineCurve)]
    pub fn add_spline_curve(&mut self, points: &[f64]) {
        self.inner.add(ShapeCurve::Spline {
            points: pairs(points),
        });
    }
    #[wasm_bindgen(js_name = addEllipseCurve)]
    pub fn add_ellipse_curve(
        &mut self,
        x: f64,
        y: f64,
        x_radius: f64,
        y_radius: f64,
        start: f64,
        end: f64,
        clockwise: bool,
        rotation: f64,
    ) {
        self.inner.add(ShapeCurve::Ellipse {
            x,
            y,
            x_radius,
            y_radius,
            start_angle: start,
            end_angle: end,
            clockwise,
            rotation,
        });
    }

    #[wasm_bindgen(js_name = curveCount)]
    pub fn curve_count(&self) -> usize {
        self.inner.curves.len()
    }

    /// Curve `i` as `[kind, …parameters]`: 0 line `x1 y1 x2 y2`, 1 quadratic
    /// `x0 y0 x1 y1 x2 y2`, 2 cubic `x0 y0 … x3 y3`, 3 spline `x0 y0 …`,
    /// 4 ellipse `x y xRadius yRadius start end clockwise rotation`.
    #[wasm_bindgen(js_name = curveData)]
    pub fn curve_data(&self, i: usize) -> Option<Vec<f64>> {
        Some(match self.inner.curves.get(i)? {
            ShapeCurve::Line { v1, v2 } => vec![0.0, v1.x, v1.y, v2.x, v2.y],
            ShapeCurve::QuadraticBezier { v0, v1, v2 } => {
                vec![1.0, v0.x, v0.y, v1.x, v1.y, v2.x, v2.y]
            }
            ShapeCurve::CubicBezier { v0, v1, v2, v3 } => {
                vec![2.0, v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y]
            }
            ShapeCurve::Spline { points } => {
                let mut v = vec![3.0];
                v.extend(flat(points));
                v
            }
            ShapeCurve::Ellipse {
                x,
                y,
                x_radius,
                y_radius,
                start_angle,
                end_angle,
                clockwise,
                rotation,
            } => vec![
                4.0,
                *x,
                *y,
                *x_radius,
                *y_radius,
                *start_angle,
                *end_angle,
                if *clockwise { 1.0 } else { 0.0 },
                *rotation,
            ],
        })
    }

    #[wasm_bindgen(js_name = clearCurves)]
    pub fn clear_curves(&mut self) {
        self.inner.curves.clear();
    }

    #[wasm_bindgen(getter, js_name = autoClose)]
    pub fn auto_close(&self) -> bool {
        self.inner.auto_close
    }
    #[wasm_bindgen(setter, js_name = autoClose)]
    pub fn set_auto_close(&mut self, v: bool) {
        self.inner.auto_close = v;
    }
    #[wasm_bindgen(getter, js_name = currentX)]
    pub fn current_x(&self) -> f64 {
        self.inner.current_point.x
    }
    #[wasm_bindgen(getter, js_name = currentY)]
    pub fn current_y(&self) -> f64 {
        self.inner.current_point.y
    }
    #[wasm_bindgen(js_name = setCurrentPoint)]
    pub fn set_current_point(&mut self, x: f64, y: f64) {
        self.inner.current_point = Vec2d::new(x, y);
    }

    /// `getPoints(divisions)`, flattened.
    #[wasm_bindgen(js_name = getPoints)]
    pub fn get_points(&self, divisions: usize) -> Vec<f64> {
        flat(&self.inner.get_points(divisions))
    }
    /// `getSpacedPoints(divisions)`, flattened.
    #[wasm_bindgen(js_name = getSpacedPoints)]
    pub fn get_spaced_points(&self, divisions: usize) -> Vec<f64> {
        flat(&self.inner.get_spaced_points(divisions))
    }
    #[wasm_bindgen(js_name = getLength)]
    pub fn get_length(&self) -> f64 {
        self.inner.get_length()
    }
    #[wasm_bindgen(js_name = getCurveLengths)]
    pub fn get_curve_lengths(&self) -> Vec<f64> {
        self.inner.get_curve_lengths()
    }
    /// `getLengths(divisions)` of the whole path.
    #[wasm_bindgen(js_name = getLengths)]
    pub fn get_lengths(&self, divisions: usize) -> Vec<f64> {
        self.inner.get_lengths(divisions)
    }
    /// `getPoint(t)` as `[x, y]`, or `undefined` where three.js has `null`.
    #[wasm_bindgen(js_name = getPoint)]
    pub fn get_point(&self, t: f64) -> Option<Vec<f64>> {
        self.inner.get_point(t).map(|p| vec![p.x, p.y])
    }
    #[wasm_bindgen(js_name = getPointAt)]
    pub fn get_point_at(&self, u: f64) -> Option<Vec<f64>> {
        self.inner.get_point_at(u).map(|p| vec![p.x, p.y])
    }
    #[wasm_bindgen(js_name = getTangent)]
    pub fn get_tangent(&self, t: f64) -> Vec<f64> {
        let p = self.inner.get_tangent(t);
        vec![p.x, p.y]
    }
    #[wasm_bindgen(js_name = getTangentAt)]
    pub fn get_tangent_at(&self, u: f64) -> Vec<f64> {
        let p = self.inner.get_tangent_at(u);
        vec![p.x, p.y]
    }

    #[wasm_bindgen(js_name = clone)]
    pub fn clone_path(&self) -> WebShapeCurvePath {
        self.clone()
    }
}

/// Shapes with their holes, as a list — what the shape geometries take and
/// `ShapePath.toShapes` / `Font.generateShapes` return.
#[wasm_bindgen]
#[derive(Default)]
pub struct WebShapeList {
    pub(crate) shapes: Vec<shapes::Shape>,
}

#[wasm_bindgen]
impl WebShapeList {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebShapeList {
        WebShapeList::default()
    }
    /// Append a shape whose outline is (a copy of) `outline`.
    #[wasm_bindgen(js_name = addShape)]
    pub fn add_shape(&mut self, outline: &WebShapeCurvePath) {
        self.shapes
            .push(shapes::Shape::from_path(outline.inner.clone()));
    }
    /// Append a hole to the last shape.
    #[wasm_bindgen(js_name = addHole)]
    pub fn add_hole(&mut self, hole: &WebShapeCurvePath) {
        if let Some(s) = self.shapes.last_mut() {
            s.holes.push(hole.inner.clone());
        }
    }
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.shapes.len()
    }
    pub fn outline(&self, i: usize) -> Option<WebShapeCurvePath> {
        self.shapes.get(i).map(|s| WebShapeCurvePath {
            inner: s.path.clone(),
        })
    }
    #[wasm_bindgen(js_name = holeCount)]
    pub fn hole_count(&self, i: usize) -> usize {
        self.shapes.get(i).map_or(0, |s| s.holes.len())
    }
    pub fn hole(&self, i: usize, j: usize) -> Option<WebShapeCurvePath> {
        self.shapes
            .get(i)
            .and_then(|s| s.holes.get(j))
            .map(|h| WebShapeCurvePath { inner: h.clone() })
    }

    /// `ShapePath.toShapes(isCCW)`, taking this list's outlines as the
    /// subpaths (their holes are ignored).
    #[wasm_bindgen(js_name = toShapes)]
    pub fn to_shapes(&self, is_ccw: bool) -> WebShapeList {
        let subs: Vec<shapes::Path> = self.shapes.iter().map(|s| s.path.clone()).collect();
        WebShapeList {
            shapes: shapes::to_shapes(&subs, is_ccw),
        }
    }
}

/// three.js `ShapeUtils.area(contour)`, points flattened.
#[wasm_bindgen(js_name = shapeUtilsArea)]
pub fn shape_utils_area(contour: &[f64]) -> f64 {
    shape_utils::area(&pairs(contour))
}

/// three.js `ShapeUtils.triangulateShape(contour, holes)`. Holes are
/// concatenated in `holes`, `hole_lengths[i]` points each. Returns
/// `[contourTrimmed, hole0Trimmed, …, a0, b0, c0, a1, …]`: first one flag
/// per contour (1 where three.js would pop a repeated closing point from the
/// caller's array), then the triangles.
#[wasm_bindgen(js_name = shapeUtilsTriangulate)]
pub fn shape_utils_triangulate(contour: &[f64], holes: &[f64], hole_lengths: &[u32]) -> Vec<u32> {
    let mut c = pairs(contour);
    let all = pairs(holes);
    let mut hs = Vec::with_capacity(hole_lengths.len());
    let mut at = 0usize;
    for &n in hole_lengths {
        let n = n as usize;
        hs.push(all[at.min(all.len())..(at + n).min(all.len())].to_vec());
        at += n;
    }
    let before: Vec<usize> = std::iter::once(c.len())
        .chain(hs.iter().map(|h| h.len()))
        .collect();
    let tris = shape_utils::triangulate_shape(&mut c, &mut hs);
    let after = std::iter::once(c.len()).chain(hs.iter().map(|h| h.len()));
    let mut out: Vec<u32> = before
        .iter()
        .zip(after)
        .map(|(&b, a)| (b != a) as u32)
        .collect();
    out.extend(tris.iter().flatten());
    out
}

/// What [`WebGeometry::extrude`] built: the geometry, and the raw unindexed
/// vertices plus the face sequence for a caller with its own UV generator.
#[wasm_bindgen]
pub struct WebExtrudeResult {
    geometry: Arc<crate::core::BufferGeometry>,
    positions: Vec<f64>,
    faces: Vec<u8>,
}

#[wasm_bindgen]
impl WebExtrudeResult {
    /// The finished geometry (`position`, `uv`, `normal`, groups).
    pub fn geometry(&self) -> WebBufferGeometry {
        WebBufferGeometry {
            inner: self.geometry.clone(),
        }
    }
    /// The vertices in JavaScript precision, unindexed — what three.js hands
    /// a `UVGenerator` as `vertices`.
    pub fn positions(&self) -> Vec<f64> {
        self.positions.clone()
    }
    /// One entry per generated face in order: 3 for a cap triangle
    /// (`generateTopUV`), 4 for a side quad (`generateSideWallUV`, six
    /// vertices).
    #[wasm_bindgen(js_name = faceKinds)]
    pub fn face_kinds(&self) -> Vec<u8> {
        self.faces.clone()
    }
}

#[wasm_bindgen]
impl WebGeometry {
    /// three.js `ShapeGeometry(shapes, curveSegments)`; `grouped` is whether
    /// three.js was given an array (which adds one group per shape).
    #[wasm_bindgen(js_name = shapeGeometry)]
    pub fn shape_geometry(
        shapes: &WebShapeList,
        curve_segments: usize,
        grouped: bool,
    ) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(ShapeGeometry::build(
                &shapes.shapes,
                curve_segments,
                grouped,
            )),
        }
    }

    /// three.js `ExtrudeGeometry(shapes, options)` without an `extrudePath`.
    /// `bevel_size` NaN means three.js's default (`bevelThickness - 0.1`).
    #[wasm_bindgen(js_name = extrude)]
    pub fn extrude(
        shapes: &WebShapeList,
        curve_segments: usize,
        steps: usize,
        depth: f64,
        bevel_enabled: bool,
        bevel_thickness: f64,
        bevel_size: f64,
        bevel_offset: f64,
        bevel_segments: usize,
    ) -> WebExtrudeResult {
        extrude_with(
            shapes,
            ExtrudeOptions {
                curve_segments,
                steps,
                depth,
                bevel_enabled,
                bevel_thickness,
                bevel_size: if bevel_size.is_nan() {
                    None
                } else {
                    Some(bevel_size)
                },
                bevel_offset,
                bevel_segments,
                extrude_path: None,
            },
        )
    }

    /// three.js `ExtrudeGeometry(shapes, options)` with `options.extrudePath`
    /// — any curve, a JavaScript `Curve` subclass included — sampled as
    /// three.js samples it (`getSpacedPoints(steps)`,
    /// `computeFrenetFrames(steps, false)`).
    #[wasm_bindgen(js_name = extrudeAlongCurve)]
    pub fn extrude_along_curve(
        shapes: &WebShapeList,
        curve_segments: usize,
        steps: usize,
        bevel_thickness: f64,
        path: &super::curves::WebCurve,
    ) -> Result<WebExtrudeResult, JsValue> {
        let extrude_path =
            super::curves::checked_curve(path, |c| ExtrudePath::from_curve(c, steps))?;
        Ok(extrude_with(
            shapes,
            ExtrudeOptions {
                curve_segments,
                steps,
                bevel_enabled: false,
                bevel_thickness,
                extrude_path: Some(extrude_path),
                ..ExtrudeOptions::default()
            },
        ))
    }
}

fn extrude_with(shapes: &WebShapeList, options: ExtrudeOptions) -> WebExtrudeResult {
    let built = ExtrudeGeometry::build(&shapes.shapes, &options);
    let positions = built.positions.clone();
    let faces = built
        .faces
        .iter()
        .map(|f| match f {
            ExtrudeFace::Top => 3,
            ExtrudeFace::Side => 4,
        })
        .collect();
    WebExtrudeResult {
        geometry: Arc::new(built.into_geometry()),
        positions,
        faces,
    }
}

/// A typeface.js font (`FontLoader.parse` / `new Font(json)`).
#[wasm_bindgen]
pub struct WebFont {
    inner: crate::loaders::typeface::Font,
}

#[wasm_bindgen]
impl WebFont {
    /// Parse typeface.js JSON text.
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str) -> Result<WebFont, JsValue> {
        crate::loaders::typeface::Font::parse(json)
            .map(|inner| WebFont { inner })
            .map_err(|e| JsValue::from_str(&e))
    }

    /// `font.generateShapes(text, size)`.
    #[wasm_bindgen(js_name = generateShapes)]
    pub fn generate_shapes(&self, text: &str, size: f64) -> WebShapeList {
        WebShapeList {
            shapes: self.inner.generate_shapes(text, size),
        }
    }
}
