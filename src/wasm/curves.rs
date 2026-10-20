//! Bindings split out of `wasm.rs` by area (see the `mod` list there): the
//! shim's `Curve` hierarchy — three.js r165's `Curve`, `CurvePath` and every
//! built-in curve — over [`crate::curves::three`], in `f64` as JavaScript
//! computes it.
//!
//! One [`WebCurve`] backs one JavaScript curve object. The shim keeps the
//! three.js-visible properties (`v1`, `points`, `closed`, …) on the JS object
//! and pushes them here before each call; the arc-length tables and their
//! `needsUpdate` live here, cached exactly as three.js caches them.
//!
//! A JavaScript `Curve` subclass that overrides `getPoint` becomes a custom
//! curve: the base-class algorithms run here and call back into that
//! `getPoint` (a `js_sys::Function`), so `getPointAt`, `getSpacedPoints`,
//! `computeFrenetFrames` and the Tube/Extrude sweeps all work on it.
#![allow(unused_imports)]

use super::*;
use crate::curves::shapes::{ShapeCurve, Vec2d};
use crate::curves::three::{
    CatmullRomType, CurveDef, CustomCurve, CustomRole, PointFn, ThreeCurve, Vec3d,
};
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    /// The exception a JavaScript `getPoint` threw during the current
    /// evaluation; further callbacks are skipped until it is reported.
    static CALLBACK_ERROR: RefCell<Option<JsValue>> = const { RefCell::new(None) };
}

/// A JS `(t) => boolean` that writes the point into `out` (false = null).
fn js_point_fn(f: js_sys::Function, out: js_sys::Float64Array) -> PointFn {
    Rc::new(move |t| {
        if CALLBACK_ERROR.with(|e| e.borrow().is_some()) {
            return None;
        }
        match f.call1(&JsValue::NULL, &JsValue::from_f64(t)) {
            Ok(v) if v.is_truthy() => Some([out.get_index(0), out.get_index(1), out.get_index(2)]),
            Ok(_) => None,
            Err(e) => {
                CALLBACK_ERROR.with(|s| *s.borrow_mut() = Some(e));
                None
            }
        }
    })
}

/// Run `f`, rethrowing anything a JS callback threw during it (and dropping
/// the caches it may have filled with NaN).
fn checked<T>(node: &ThreeCurve, f: impl FnOnce() -> T) -> Result<T, JsValue> {
    let outer = CALLBACK_ERROR.with(|e| e.borrow_mut().take());
    let r = f();
    let err = CALLBACK_ERROR.with(|e| std::mem::replace(&mut *e.borrow_mut(), outer));
    match err {
        Some(e) => {
            node.clear_caches();
            Err(e)
        }
        None => Ok(r),
    }
}

fn v2(a: &[f64], i: usize) -> Vec2d {
    Vec2d::new(a[i], a[i + 1])
}

fn v3(a: &[f64], i: usize) -> Vec3d {
    [a[i], a[i + 1], a[i + 2]]
}

fn flat_points(points: Vec<Option<Vec3d>>) -> Vec<f64> {
    points
        .into_iter()
        .flat_map(|p| p.unwrap_or([f64::NAN; 3]))
        .collect()
}

fn bad_len(what: &str, n: usize) -> JsValue {
    js_sys::TypeError::new(&format!("{what}: unexpected {n} numbers")).into()
}

/// One three.js curve object (see the module docs).
#[wasm_bindgen]
pub struct WebCurve {
    node: Rc<ThreeCurve>,
}

impl WebCurve {
    pub(crate) fn node(&self) -> &Rc<ThreeCurve> {
        &self.node
    }
}

#[wasm_bindgen]
impl WebCurve {
    /// A curve to be defined by one of the `set*` calls (a zero-length
    /// `LineCurve3` until then).
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebCurve {
        WebCurve {
            node: ThreeCurve::line3([0.0; 3], [0.0; 3]),
        }
    }

    /// A `Curve` subclass with its own `getPoint`: `get_point(t)` writes the
    /// point to `out[0..3]` and returns whether there was one. An overridden
    /// `getTangent` comes the same way. `role`: 0 plain, 1 line, 2 ellipse,
    /// 3 spline (of `spline_points` points) — what the object's three.js flags
    /// say it is.
    pub fn custom(
        get_point: js_sys::Function,
        out: js_sys::Float64Array,
        get_tangent: Option<js_sys::Function>,
        dims: u8,
        role: u8,
        spline_points: usize,
    ) -> WebCurve {
        let role = match role {
            1 => CustomRole::Line,
            2 => CustomRole::Ellipse,
            3 => CustomRole::Spline(spline_points),
            _ => CustomRole::Plain,
        };
        WebCurve {
            node: ThreeCurve::new(CurveDef::Custom(CustomCurve {
                get_point: js_point_fn(get_point, out.clone()),
                get_tangent: get_tangent.map(|f| js_point_fn(f, out)),
                dims,
                role,
            })),
        }
    }

    /// Update a custom curve's role (a spline subclass's point count moves).
    #[wasm_bindgen(js_name = setCustomRole)]
    pub fn set_custom_role(&self, dims: u8, role: u8, spline_points: usize) {
        self.node.with_def_mut(|d| {
            if let CurveDef::Custom(c) = d {
                c.dims = dims;
                c.role = match role {
                    1 => CustomRole::Line,
                    2 => CustomRole::Ellipse,
                    3 => CustomRole::Spline(spline_points),
                    _ => CustomRole::Plain,
                };
            }
        });
    }

    /// `LineCurve` (`[x1, y1, x2, y2]`) or `LineCurve3` (six numbers).
    #[wasm_bindgen(js_name = setLine)]
    pub fn set_line(&self, v: &[f64]) -> Result<(), JsValue> {
        let def = match v.len() {
            4 => CurveDef::Planar(ShapeCurve::Line {
                v1: v2(v, 0),
                v2: v2(v, 2),
            }),
            6 => CurveDef::Line3 {
                v1: v3(v, 0),
                v2: v3(v, 3),
            },
            n => return Err(bad_len("setLine", n)),
        };
        self.node.set_def(def);
        Ok(())
    }

    /// `QuadraticBezierCurve` (six numbers) or `QuadraticBezierCurve3` (nine).
    #[wasm_bindgen(js_name = setQuadraticBezier)]
    pub fn set_quadratic_bezier(&self, v: &[f64]) -> Result<(), JsValue> {
        let def = match v.len() {
            6 => CurveDef::Planar(ShapeCurve::QuadraticBezier {
                v0: v2(v, 0),
                v1: v2(v, 2),
                v2: v2(v, 4),
            }),
            9 => CurveDef::QuadraticBezier3 {
                v0: v3(v, 0),
                v1: v3(v, 3),
                v2: v3(v, 6),
            },
            n => return Err(bad_len("setQuadraticBezier", n)),
        };
        self.node.set_def(def);
        Ok(())
    }

    /// `CubicBezierCurve` (eight numbers) or `CubicBezierCurve3` (twelve).
    #[wasm_bindgen(js_name = setCubicBezier)]
    pub fn set_cubic_bezier(&self, v: &[f64]) -> Result<(), JsValue> {
        let def = match v.len() {
            8 => CurveDef::Planar(ShapeCurve::CubicBezier {
                v0: v2(v, 0),
                v1: v2(v, 2),
                v2: v2(v, 4),
                v3: v2(v, 6),
            }),
            12 => CurveDef::CubicBezier3 {
                v0: v3(v, 0),
                v1: v3(v, 3),
                v2: v3(v, 6),
                v3: v3(v, 9),
            },
            n => return Err(bad_len("setCubicBezier", n)),
        };
        self.node.set_def(def);
        Ok(())
    }

    /// `SplineCurve`, points flattened `[x0, y0, …]`.
    #[wasm_bindgen(js_name = setSpline)]
    pub fn set_spline(&self, points: &[f64]) {
        let points = points.chunks_exact(2).map(|c| Vec2d::new(c[0], c[1])).collect();
        self.node.set_def(CurveDef::Planar(ShapeCurve::Spline { points }));
    }

    /// `EllipseCurve` / `ArcCurve`.
    #[wasm_bindgen(js_name = setEllipse)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_ellipse(
        &self,
        x: f64,
        y: f64,
        x_radius: f64,
        y_radius: f64,
        start_angle: f64,
        end_angle: f64,
        clockwise: bool,
        rotation: f64,
    ) {
        self.node.set_def(CurveDef::Planar(ShapeCurve::Ellipse {
            x,
            y,
            x_radius,
            y_radius,
            start_angle,
            end_angle,
            clockwise,
            rotation,
        }));
    }

    /// `CatmullRomCurve3`, points flattened `[x0, y0, z0, …]`.
    #[wasm_bindgen(js_name = setCatmullRom)]
    pub fn set_catmull_rom(&self, points: &[f64], closed: bool, curve_type: &str, tension: f64) {
        self.node.set_def(CurveDef::CatmullRom3 {
            points: points.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            closed,
            curve_type: CatmullRomType::parse(curve_type),
            tension,
        });
    }

    /// Make this a `CurvePath` with no curves yet (keeping its caches);
    /// `addCurve` then lists them.
    #[wasm_bindgen(js_name = setPath)]
    pub fn set_path(&self, auto_close: bool) {
        self.node.set_def(CurveDef::Path {
            curves: Vec::new(),
            auto_close,
        });
    }

    /// Append to a `CurvePath` (the same curve object, shared).
    #[wasm_bindgen(js_name = addCurve)]
    pub fn add_curve(&self, curve: &WebCurve) {
        let child = curve.node.clone();
        self.node.with_def_mut(|d| {
            if let CurveDef::Path { curves, .. } = d {
                curves.push(child);
            }
        });
    }

    #[wasm_bindgen(getter, js_name = arcLengthDivisions)]
    pub fn arc_length_divisions(&self) -> usize {
        self.node.arc_length_divisions()
    }

    #[wasm_bindgen(setter, js_name = arcLengthDivisions)]
    pub fn set_arc_length_divisions(&self, n: usize) {
        self.node.set_arc_length_divisions(n);
    }

    #[wasm_bindgen(getter, js_name = needsUpdate)]
    pub fn needs_update(&self) -> bool {
        self.node.needs_update()
    }

    #[wasm_bindgen(setter, js_name = needsUpdate)]
    pub fn set_needs_update(&self, v: bool) {
        self.node.set_needs_update(v);
    }

    /// `cacheArcLengths`, or undefined before the first `getLengths`.
    #[wasm_bindgen(js_name = cacheArcLengths)]
    pub fn cache_arc_lengths(&self) -> Option<Vec<f64>> {
        self.node.cache_arc_lengths().map(|c| c.to_vec())
    }

    /// 2 for a 2D curve (its points are `Vector2`s), 3 otherwise.
    #[wasm_bindgen(getter)]
    pub fn dims(&self) -> u8 {
        self.node.dims()
    }

    /// `getPoint(t)` as `[x, y, z]`, empty for null.
    #[wasm_bindgen(js_name = getPoint)]
    pub fn get_point(&self, t: f64) -> Result<Vec<f64>, JsValue> {
        checked(&self.node, || self.node.get_point(t).map_or(Vec::new(), |p| p.to_vec()))
    }

    /// `getPointAt(u)` as `[x, y, z]`, empty for null.
    #[wasm_bindgen(js_name = getPointAt)]
    pub fn get_point_at(&self, u: f64) -> Result<Vec<f64>, JsValue> {
        checked(&self.node, || self.node.get_point_at(u).map_or(Vec::new(), |p| p.to_vec()))
    }

    /// `getPoints(divisions)`, flattened xyz.
    #[wasm_bindgen(js_name = getPoints)]
    pub fn get_points(&self, divisions: usize) -> Result<Vec<f64>, JsValue> {
        checked(&self.node, || flat_points(self.node.get_points(divisions)))
    }

    /// `getSpacedPoints(divisions)`, flattened xyz.
    #[wasm_bindgen(js_name = getSpacedPoints)]
    pub fn get_spaced_points(&self, divisions: usize) -> Result<Vec<f64>, JsValue> {
        checked(&self.node, || flat_points(self.node.get_spaced_points(divisions)))
    }

    #[wasm_bindgen(js_name = getLength)]
    pub fn get_length(&self) -> Result<f64, JsValue> {
        checked(&self.node, || self.node.get_length())
    }

    #[wasm_bindgen(js_name = getLengths)]
    pub fn get_lengths(&self, divisions: usize) -> Result<Vec<f64>, JsValue> {
        checked(&self.node, || self.node.get_lengths(divisions).to_vec())
    }

    #[wasm_bindgen(js_name = updateArcLengths)]
    pub fn update_arc_lengths(&self) -> Result<(), JsValue> {
        checked(&self.node, || self.node.update_arc_lengths())
    }

    /// `getUtoTmapping(u, distance)`.
    #[wasm_bindgen(js_name = getUtoTmapping)]
    pub fn get_u_to_t_mapping(&self, u: f64, distance: Option<f64>) -> Result<f64, JsValue> {
        checked(&self.node, || self.node.get_u_to_t_mapping(u, distance))
    }

    /// `getTangent(t)` as `[x, y, z]`.
    #[wasm_bindgen(js_name = getTangent)]
    pub fn get_tangent(&self, t: f64) -> Result<Vec<f64>, JsValue> {
        checked(&self.node, || self.node.get_tangent(t).to_vec())
    }

    /// `Curve.prototype.getTangent(t)`: the base class's (or a line's),
    /// never an override — what `super.getTangent` reaches.
    #[wasm_bindgen(js_name = getBaseTangent)]
    pub fn get_base_tangent(&self, t: f64) -> Result<Vec<f64>, JsValue> {
        checked(&self.node, || self.node.get_tangent_base(t).to_vec())
    }

    /// `getTangentAt(u)` as `[x, y, z]`.
    #[wasm_bindgen(js_name = getTangentAt)]
    pub fn get_tangent_at(&self, u: f64) -> Result<Vec<f64>, JsValue> {
        checked(&self.node, || self.node.get_tangent_at(u).to_vec())
    }

    /// `computeFrenetFrames(segments, closed)`: the tangents, then the
    /// normals, then the binormals, each `segments + 1` flattened xyz.
    #[wasm_bindgen(js_name = computeFrenetFrames)]
    pub fn compute_frenet_frames(&self, segments: usize, closed: bool) -> Result<Vec<f64>, JsValue> {
        checked(&self.node, || {
            let f = self.node.compute_frenet_frames(segments, closed);
            f.tangents
                .iter()
                .chain(f.normals.iter())
                .chain(f.binormals.iter())
                .flatten()
                .copied()
                .collect()
        })
    }

    /// `CurvePath.getCurveLengths()`.
    #[wasm_bindgen(js_name = getCurveLengths)]
    pub fn get_curve_lengths(&self) -> Result<Vec<f64>, JsValue> {
        checked(&self.node, || self.node.get_curve_lengths().to_vec())
    }
}

impl Default for WebCurve {
    fn default() -> Self {
        Self::new()
    }
}

/// `new TubeGeometry(path, …)`: the geometry and the frames three.js keeps
/// on it.
#[wasm_bindgen]
pub struct WebTubeResult {
    geometry: Option<WebGeometry>,
    frames: Vec<f64>,
}

#[wasm_bindgen]
impl WebTubeResult {
    /// The geometry (once).
    pub fn geometry(&mut self) -> Option<WebGeometry> {
        self.geometry.take()
    }

    /// Tangents, normals, binormals: each `tubularSegments + 1` xyz.
    pub fn frames(&self) -> Vec<f64> {
        self.frames.clone()
    }
}

#[wasm_bindgen]
impl WebGeometry {
    /// three.js r165 `new TubeGeometry(path, tubularSegments, radius,
    /// radialSegments, closed)`; `path` may be a custom (JavaScript) curve.
    #[wasm_bindgen(js_name = tubeFromCurve)]
    pub fn tube_from_curve(
        path: &WebCurve,
        tubular_segments: usize,
        radius: f64,
        radial_segments: usize,
        closed: bool,
    ) -> Result<WebTubeResult, JsValue> {
        let (g, f) = checked(&path.node, || {
            crate::geometries::TubeGeometry::from_curve(
                &path.node,
                tubular_segments,
                radius,
                radial_segments,
                closed,
            )
        })?;
        Ok(WebTubeResult {
            geometry: Some(WebGeometry { inner: Arc::new(g) }),
            frames: f
                .tangents
                .iter()
                .chain(f.normals.iter())
                .chain(f.binormals.iter())
                .flatten()
                .copied()
                .collect(),
        })
    }
}

/// `curve.getPoint` etc. errors a JS callback raised, for bindings elsewhere.
pub(crate) fn checked_curve<T>(curve: &WebCurve, f: impl FnOnce(&ThreeCurve) -> T) -> Result<T, JsValue> {
    checked(&curve.node, || f(&curve.node))
}
