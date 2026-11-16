//! DXF drawings: a [`DxfDocument`] of lines, polylines (with bulges), circles
//! and arcs on named layers, written as AutoCAD R12 ASCII ([`write_dxf`]),
//! the version every CAD and CAM program reads. [`crate::loaders::parse_dxf`]
//! reads them back, from R12 and from later versions' `LWPOLYLINE`s.
//!
//! DXF is 2D in practice — cutting outlines, flat patterns, profiles — so the
//! helpers take 2D points and curves; entities still carry `z`, and a
//! polyline whose vertices are not all at one height is written as a 3D one.
//!
//! ```
//! use threers::exporters::{write_dxf, DxfDocument, DxfUnits};
//!
//! let mut doc = DxfDocument::new(DxfUnits::Millimeters);
//! doc.layer("CUT", 1);
//! doc.polyline_2d("CUT", &[[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]], true);
//! doc.circle("CUT", [20.0, 10.0], 4.0);
//! let text = write_dxf(&doc);
//! assert!(text.starts_with("0\nSECTION\n2\nHEADER\n"));
//! assert!(text.ends_with("0\nEOF\n"));
//! ```

use std::fmt::Write;

use crate::curves::Curve2;

/// Drawing units: DXF's `$INSUNITS` codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DxfUnits {
    #[default]
    Unitless,
    Inches,
    Feet,
    Millimeters,
    Centimeters,
    Meters,
}

impl DxfUnits {
    pub fn code(self) -> i32 {
        match self {
            DxfUnits::Unitless => 0,
            DxfUnits::Inches => 1,
            DxfUnits::Feet => 2,
            DxfUnits::Millimeters => 4,
            DxfUnits::Centimeters => 5,
            DxfUnits::Meters => 6,
        }
    }

    /// Codes this does not name read as unitless.
    pub fn from_code(code: i32) -> Self {
        match code {
            1 => DxfUnits::Inches,
            2 => DxfUnits::Feet,
            4 => DxfUnits::Millimeters,
            5 => DxfUnits::Centimeters,
            6 => DxfUnits::Meters,
            _ => DxfUnits::Unitless,
        }
    }
}

/// A layer and its AutoCAD Color Index (1 red … 7 white/black, up to 255).
#[derive(Debug, Clone, PartialEq)]
pub struct DxfLayer {
    pub name: String,
    pub color: u8,
}

/// What a drawing holds. Angles are degrees, counter-clockwise from +X.
#[derive(Debug, Clone, PartialEq)]
pub enum DxfEntity {
    Line {
        layer: String,
        start: [f64; 3],
        end: [f64; 3],
    },
    /// Through `points` in order, back to the first when `closed`. `bulges[i]`
    /// curves the segment leaving point `i` into an arc: the tangent of a
    /// quarter of its sweep, positive counter-clockwise. Empty means straight
    /// throughout.
    Polyline {
        layer: String,
        points: Vec<[f64; 3]>,
        bulges: Vec<f64>,
        closed: bool,
    },
    Circle {
        layer: String,
        center: [f64; 3],
        radius: f64,
    },
    Arc {
        layer: String,
        center: [f64; 3],
        radius: f64,
        start_angle: f64,
        end_angle: f64,
    },
}

impl DxfEntity {
    pub fn layer(&self) -> &str {
        match self {
            DxfEntity::Line { layer, .. }
            | DxfEntity::Polyline { layer, .. }
            | DxfEntity::Circle { layer, .. }
            | DxfEntity::Arc { layer, .. } => layer,
        }
    }

    /// The entity as points, arcs and bulges divided finely enough that no
    /// chord strays more than `tolerance` from the true curve. A closed shape
    /// does not repeat its first point.
    pub fn flatten(&self, tolerance: f64) -> Vec<[f64; 3]> {
        match self {
            DxfEntity::Line { start, end, .. } => vec![*start, *end],
            DxfEntity::Circle { center, radius, .. } => {
                let n = arc_divisions(*radius, std::f64::consts::TAU, tolerance);
                let mut pts = arc_points(*center, *radius, 0.0, std::f64::consts::TAU, n);
                pts.pop();
                pts
            }
            DxfEntity::Arc {
                center,
                radius,
                start_angle,
                end_angle,
                ..
            } => {
                let a0 = start_angle.to_radians();
                let mut sweep = (end_angle - start_angle).rem_euclid(360.0).to_radians();
                if sweep == 0.0 {
                    sweep = std::f64::consts::TAU;
                }
                let n = arc_divisions(*radius, sweep, tolerance);
                arc_points(*center, *radius, a0, sweep, n)
            }
            DxfEntity::Polyline {
                points,
                bulges,
                closed,
                ..
            } => {
                let n = points.len();
                let segments = if *closed { n } else { n.saturating_sub(1) };
                let mut out = vec![];
                for i in 0..segments {
                    let (a, b) = (points[i], points[(i + 1) % n]);
                    out.push(a);
                    let bulge = bulges.get(i).copied().unwrap_or(0.0);
                    if bulge != 0.0 {
                        out.extend(bulge_points(a, b, bulge, tolerance));
                    }
                }
                if !*closed && let Some(last) = points.last() {
                    out.push(*last);
                }
                out
            }
        }
    }
}

fn arc_divisions(radius: f64, sweep: f64, tolerance: f64) -> usize {
    // The chord of angle θ strays r(1 − cos θ/2) from its arc.
    let tol = tolerance.max(1e-9).min(radius.abs().max(1e-9));
    let step = 2.0 * (1.0 - tol / radius.abs().max(1e-9)).acos();
    ((sweep.abs() / step.max(1e-6)).ceil() as usize).clamp(1, 100_000)
}

fn arc_points(c: [f64; 3], r: f64, a0: f64, sweep: f64, n: usize) -> Vec<[f64; 3]> {
    (0..=n)
        .map(|i| {
            let a = a0 + sweep * i as f64 / n as f64;
            [c[0] + r * a.cos(), c[1] + r * a.sin(), c[2]]
        })
        .collect()
}

/// The points strictly between `a` and `b` on the arc a bulge makes.
fn bulge_points(a: [f64; 3], b: [f64; 3], bulge: f64, tolerance: f64) -> Vec<[f64; 3]> {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let chord = (dx * dx + dy * dy).sqrt();
    if chord == 0.0 {
        return vec![];
    }
    let sweep = 4.0 * bulge.atan();
    let radius = chord / (2.0 * (sweep / 2.0).sin()).abs();
    // The centre sits off the chord's midpoint, on the side the bulge turns.
    let offset = radius * (sweep / 2.0).cos() * sweep.signum();
    let (mx, my) = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0);
    let c = [mx - dy / chord * offset, my + dx / chord * offset, a[2]];
    let a0 = (a[1] - c[1]).atan2(a[0] - c[0]);
    let n = arc_divisions(radius, sweep, tolerance);
    let mut pts = arc_points(c, radius, a0, sweep, n);
    pts.pop();
    pts.remove(0);
    pts
}

/// A drawing: its units, its layers, and what is drawn on them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DxfDocument {
    pub units: DxfUnits,
    /// Layers declared with a colour. Entities may name others; the writer
    /// declares those white.
    pub layers: Vec<DxfLayer>,
    pub entities: Vec<DxfEntity>,
}

impl DxfDocument {
    pub fn new(units: DxfUnits) -> Self {
        Self {
            units,
            ..Default::default()
        }
    }

    /// Declares `name` with `color`, or recolours it.
    pub fn layer(&mut self, name: &str, color: u8) -> &mut Self {
        match self.layers.iter_mut().find(|l| l.name == name) {
            Some(l) => l.color = color,
            None => self.layers.push(DxfLayer {
                name: name.into(),
                color,
            }),
        }
        self
    }

    pub fn add(&mut self, entity: DxfEntity) -> &mut Self {
        self.entities.push(entity);
        self
    }

    pub fn line(&mut self, layer: &str, start: [f64; 2], end: [f64; 2]) -> &mut Self {
        self.add(DxfEntity::Line {
            layer: layer.into(),
            start: [start[0], start[1], 0.0],
            end: [end[0], end[1], 0.0],
        })
    }

    pub fn polyline_2d(&mut self, layer: &str, points: &[[f64; 2]], closed: bool) -> &mut Self {
        self.add(DxfEntity::Polyline {
            layer: layer.into(),
            points: points.iter().map(|p| [p[0], p[1], 0.0]).collect(),
            bulges: vec![],
            closed,
        })
    }

    pub fn circle(&mut self, layer: &str, center: [f64; 2], radius: f64) -> &mut Self {
        self.add(DxfEntity::Circle {
            layer: layer.into(),
            center: [center[0], center[1], 0.0],
            radius,
        })
    }

    pub fn arc(
        &mut self,
        layer: &str,
        center: [f64; 2],
        radius: f64,
        start_angle: f64,
        end_angle: f64,
    ) -> &mut Self {
        self.add(DxfEntity::Arc {
            layer: layer.into(),
            center: [center[0], center[1], 0.0],
            radius,
            start_angle,
            end_angle,
        })
    }

    /// A curve — a `Shape`'s outline, a spline, an ellipse — as a polyline
    /// through `divisions + 1` of its points.
    pub fn curve(
        &mut self,
        layer: &str,
        curve: &dyn Curve2,
        divisions: usize,
        closed: bool,
    ) -> &mut Self {
        let mut pts: Vec<[f64; 2]> = curve
            .get_points(divisions.max(1))
            .into_iter()
            .map(|p| [p.x as f64, p.y as f64])
            .collect();
        if closed && pts.len() > 1 && pts.first() == pts.last() {
            pts.pop();
        }
        self.polyline_2d(layer, &pts, closed)
    }

    /// The smallest box around every entity, as `(min, max)`; `None` when
    /// there are none. Arcs and bulges are measured on their flattened shape.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut b: Option<([f64; 3], [f64; 3])> = None;
        for e in &self.entities {
            for p in e.flatten(1e-3) {
                let (lo, hi) = b.get_or_insert((p, p));
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        b
    }
}

/// The drawing as R12 ASCII DXF.
pub fn write_dxf(doc: &DxfDocument) -> String {
    let mut out = String::new();
    let mut code = |c: i32, v: &str| {
        let _ = write!(out, "{c}\n{v}\n");
    };
    let num = |v: f64| {
        let v = if v == 0.0 { 0.0 } else { v };
        format!("{v}")
    };

    code(0, "SECTION");
    code(2, "HEADER");
    code(9, "$ACADVER");
    code(1, "AC1009");
    code(9, "$INSUNITS");
    code(70, &doc.units.code().to_string());
    if let Some((lo, hi)) = doc.bounds() {
        code(9, "$EXTMIN");
        code(10, &num(lo[0]));
        code(20, &num(lo[1]));
        code(30, &num(lo[2]));
        code(9, "$EXTMAX");
        code(10, &num(hi[0]));
        code(20, &num(hi[1]));
        code(30, &num(hi[2]));
    }
    code(0, "ENDSEC");

    let mut layers = doc.layers.clone();
    for e in &doc.entities {
        if !layers.iter().any(|l| l.name == e.layer()) {
            layers.push(DxfLayer {
                name: e.layer().into(),
                color: 7,
            });
        }
    }
    if !layers.iter().any(|l| l.name == "0") {
        layers.insert(
            0,
            DxfLayer {
                name: "0".into(),
                color: 7,
            },
        );
    }
    code(0, "SECTION");
    code(2, "TABLES");
    code(0, "TABLE");
    code(2, "LAYER");
    code(70, &layers.len().to_string());
    for l in &layers {
        code(0, "LAYER");
        code(2, &l.name);
        code(70, "0");
        code(62, &l.color.max(1).to_string());
        code(6, "CONTINUOUS");
    }
    code(0, "ENDTAB");
    code(0, "ENDSEC");

    code(0, "SECTION");
    code(2, "ENTITIES");
    for e in &doc.entities {
        match e {
            DxfEntity::Line { layer, start, end } => {
                code(0, "LINE");
                code(8, layer);
                for (k, v) in start.iter().enumerate() {
                    code(10 + 10 * k as i32, &num(*v));
                }
                for (k, v) in end.iter().enumerate() {
                    code(11 + 10 * k as i32, &num(*v));
                }
            }
            DxfEntity::Circle {
                layer,
                center,
                radius,
            } => {
                code(0, "CIRCLE");
                code(8, layer);
                for (k, v) in center.iter().enumerate() {
                    code(10 + 10 * k as i32, &num(*v));
                }
                code(40, &num(*radius));
            }
            DxfEntity::Arc {
                layer,
                center,
                radius,
                start_angle,
                end_angle,
            } => {
                code(0, "ARC");
                code(8, layer);
                for (k, v) in center.iter().enumerate() {
                    code(10 + 10 * k as i32, &num(*v));
                }
                code(40, &num(*radius));
                code(50, &num(*start_angle));
                code(51, &num(*end_angle));
            }
            DxfEntity::Polyline {
                layer,
                points,
                bulges,
                closed,
            } => {
                let z0 = points.first().map(|p| p[2]).unwrap_or(0.0);
                let flat = points.iter().all(|p| p[2] == z0);
                // R12 has no bulges on 3D polylines.
                let three_d = !flat && bulges.iter().all(|b| *b == 0.0);
                let mut flags = if *closed { 1 } else { 0 };
                if three_d {
                    flags |= 8;
                }
                code(0, "POLYLINE");
                code(8, layer);
                code(66, "1");
                code(10, "0");
                code(20, "0");
                code(30, &num(if three_d { 0.0 } else { z0 }));
                code(70, &flags.to_string());
                for (i, p) in points.iter().enumerate() {
                    code(0, "VERTEX");
                    code(8, layer);
                    code(10, &num(p[0]));
                    code(20, &num(p[1]));
                    code(30, &num(if three_d { p[2] } else { z0 }));
                    let bulge = bulges.get(i).copied().unwrap_or(0.0);
                    if bulge != 0.0 {
                        code(42, &num(bulge));
                    }
                    if three_d {
                        code(70, "32");
                    }
                }
                code(0, "SEQEND");
                code(8, layer);
            }
        }
    }
    code(0, "ENDSEC");
    code(0, "EOF");
    out
}
