//! Reads ASCII DXF into a [`DxfDocument`]: the units, the layer table, and
//! the `LINE`, `CIRCLE`, `ARC`, `LWPOLYLINE` and `POLYLINE` entities of the
//! `ENTITIES` section, from R12 onward. What it does not read — splines,
//! text, hatches, block inserts — is counted in [`DxfRead::skipped`] rather
//! than dropped silently.
//!
//! ```
//! use threers::exporters::{write_dxf, DxfDocument, DxfUnits};
//! use threers::loaders::parse_dxf;
//!
//! let mut doc = DxfDocument::new(DxfUnits::Millimeters);
//! doc.circle("HOLES", [5.0, 5.0], 2.5);
//! let read = parse_dxf(&write_dxf(&doc)).unwrap();
//! assert_eq!(read.document.entities, doc.entities);
//! assert_eq!(read.document.units, DxfUnits::Millimeters);
//! ```

use std::collections::BTreeMap;

use crate::exporters::{DxfDocument, DxfEntity, DxfLayer, DxfUnits};

#[derive(Debug, Clone, PartialEq)]
pub enum DxfError {
    /// Group codes and values must alternate line by line; this one's code is
    /// not an integer.
    BadGroupCode { line: usize, text: String },
    /// A number where the group code requires one did not parse.
    BadNumber { line: usize, text: String },
    /// Binary DXF, which this does not read.
    Binary,
}

impl std::fmt::Display for DxfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DxfError::BadGroupCode { line, text } => {
                write!(f, "line {line}: expected a group code, found {text:?}")
            }
            DxfError::BadNumber { line, text } => {
                write!(f, "line {line}: expected a number, found {text:?}")
            }
            DxfError::Binary => write!(f, "binary DXF is not supported"),
        }
    }
}

impl std::error::Error for DxfError {}

/// What [`parse_dxf`] found.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DxfRead {
    pub document: DxfDocument,
    /// Entity types in `ENTITIES` that were not read, and how many of each.
    pub skipped: BTreeMap<String, usize>,
}

struct Pair<'a> {
    code: i32,
    value: &'a str,
    line: usize,
}

impl Pair<'_> {
    fn f64(&self) -> Result<f64, DxfError> {
        self.value.trim().parse().map_err(|_| DxfError::BadNumber {
            line: self.line,
            text: self.value.into(),
        })
    }
    fn i32(&self) -> Result<i32, DxfError> {
        // Some writers put integers as reals ("8.0").
        self.f64().map(|v| v as i32)
    }
}

fn pairs(text: &str) -> Result<Vec<Pair<'_>>, DxfError> {
    if text.starts_with("AutoCAD Binary DXF") {
        return Err(DxfError::Binary);
    }
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::with_capacity(lines.len() / 2);
    let mut i = 0;
    while i + 1 < lines.len() {
        let c = lines[i].trim();
        if c.is_empty() && i + 2 >= lines.len() {
            break;
        }
        let code = c.parse().map_err(|_| DxfError::BadGroupCode {
            line: i + 1,
            text: c.into(),
        })?;
        out.push(Pair {
            code,
            value: lines[i + 1].trim_end_matches('\r').trim_start(),
            line: i + 2,
        });
        i += 2;
    }
    Ok(out)
}

/// The entity starting at `pairs[0]` (a `0` pair) runs up to the next `0`.
fn entity_end(pairs: &[Pair<'_>]) -> usize {
    pairs[1..]
        .iter()
        .position(|p| p.code == 0)
        .map(|k| k + 1)
        .unwrap_or(pairs.len())
}

fn point(body: &[Pair<'_>], base: i32) -> Result<[f64; 3], DxfError> {
    let mut p = [0.0; 3];
    for g in body {
        for (k, slot) in p.iter_mut().enumerate() {
            if g.code == base + 10 * k as i32 {
                *slot = g.f64()?;
            }
        }
    }
    Ok(p)
}

fn number(body: &[Pair<'_>], code: i32) -> Result<Option<f64>, DxfError> {
    body.iter()
        .find(|g| g.code == code)
        .map(|g| g.f64())
        .transpose()
}

fn layer_of(body: &[Pair<'_>]) -> String {
    body.iter()
        .find(|g| g.code == 8)
        .map(|g| g.value.to_string())
        .unwrap_or_else(|| "0".into())
}

pub fn parse_dxf(text: &str) -> Result<DxfRead, DxfError> {
    let pairs = pairs(text)?;
    let mut read = DxfRead::default();
    let mut section = "";
    let mut i = 0;
    while i < pairs.len() {
        let p = &pairs[i];
        if p.code == 0 && p.value == "SECTION" {
            section = pairs
                .get(i + 1)
                .filter(|n| n.code == 2)
                .map(|n| n.value)
                .unwrap_or("");
            i += 2;
            continue;
        }
        if p.code == 0 && p.value == "ENDSEC" {
            section = "";
            i += 1;
            continue;
        }
        match section {
            "HEADER" if p.code == 9 && p.value == "$INSUNITS" => {
                if let Some(v) = pairs.get(i + 1) {
                    read.document.units = DxfUnits::from_code(v.i32()?);
                }
                i += 2;
            }
            "TABLES" if p.code == 0 && p.value == "LAYER" => {
                let end = i + entity_end(&pairs[i..]);
                let body = &pairs[i + 1..end];
                if let Some(name) = body.iter().find(|g| g.code == 2) {
                    let color = number(body, 62)?.unwrap_or(7.0).abs() as u8;
                    read.document.layers.push(DxfLayer {
                        name: name.value.into(),
                        color,
                    });
                }
                i = end;
            }
            "ENTITIES" if p.code == 0 => {
                let end = i + entity_end(&pairs[i..]);
                let body = &pairs[i + 1..end];
                let layer = layer_of(body);
                match p.value {
                    "LINE" => read.document.entities.push(DxfEntity::Line {
                        layer,
                        start: point(body, 10)?,
                        end: point(body, 11)?,
                    }),
                    "CIRCLE" => read.document.entities.push(DxfEntity::Circle {
                        layer,
                        center: point(body, 10)?,
                        radius: number(body, 40)?.unwrap_or(0.0),
                    }),
                    "ARC" => read.document.entities.push(DxfEntity::Arc {
                        layer,
                        center: point(body, 10)?,
                        radius: number(body, 40)?.unwrap_or(0.0),
                        start_angle: number(body, 50)?.unwrap_or(0.0),
                        end_angle: number(body, 51)?.unwrap_or(360.0),
                    }),
                    "LWPOLYLINE" => {
                        let z = number(body, 38)?.unwrap_or(0.0);
                        let closed = number(body, 70)?.unwrap_or(0.0) as i32 & 1 == 1;
                        let mut points: Vec<[f64; 3]> = vec![];
                        let mut bulges: Vec<f64> = vec![];
                        for g in body {
                            match g.code {
                                10 => {
                                    points.push([g.f64()?, 0.0, z]);
                                    bulges.push(0.0);
                                }
                                20 => {
                                    if let Some(p) = points.last_mut() {
                                        p[1] = g.f64()?;
                                    }
                                }
                                42 => {
                                    if let Some(b) = bulges.last_mut() {
                                        *b = g.f64()?;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if bulges.iter().all(|b| *b == 0.0) {
                            bulges.clear();
                        }
                        read.document.entities.push(DxfEntity::Polyline {
                            layer,
                            points,
                            bulges,
                            closed,
                        });
                    }
                    "POLYLINE" => {
                        let flags = number(body, 70)?.unwrap_or(0.0) as i32;
                        // Polyface and polygon meshes (64, 16) are not outlines.
                        if flags & (16 | 64) != 0 {
                            *read.skipped.entry("POLYLINE (mesh)".into()).or_default() += 1;
                            i = end;
                            continue;
                        }
                        let elevation = number(body, 30)?.unwrap_or(0.0);
                        let three_d = flags & 8 != 0;
                        let mut points = vec![];
                        let mut bulges = vec![];
                        let mut j = end;
                        while j < pairs.len() && pairs[j].code == 0 && pairs[j].value == "VERTEX" {
                            let vend = j + entity_end(&pairs[j..]);
                            let vbody = &pairs[j + 1..vend];
                            let mut v = point(vbody, 10)?;
                            if !three_d {
                                v[2] = elevation;
                            }
                            points.push(v);
                            bulges.push(number(vbody, 42)?.unwrap_or(0.0));
                            j = vend;
                        }
                        if j < pairs.len() && pairs[j].code == 0 && pairs[j].value == "SEQEND" {
                            j += entity_end(&pairs[j..]);
                        }
                        if bulges.iter().all(|b| *b == 0.0) {
                            bulges.clear();
                        }
                        read.document.entities.push(DxfEntity::Polyline {
                            layer,
                            points,
                            bulges,
                            closed: flags & 1 == 1,
                        });
                        i = j;
                        continue;
                    }
                    "VERTEX" | "SEQEND" => {}
                    other => *read.skipped.entry(other.to_string()).or_default() += 1,
                }
                i = end;
            }
            _ => i += 1,
        }
    }
    Ok(read)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exporters::write_dxf;

    #[test]
    fn round_trip() {
        let mut doc = DxfDocument::new(DxfUnits::Millimeters);
        doc.layer("FRONT", 7).layer("TEMPLE", 5);
        doc.polyline_2d("FRONT", &[[0.0, 0.0], [10.0, 0.0], [10.0, 5.5]], true);
        doc.line("TEMPLE", [1.0, 2.0], [-3.25, 4.0]);
        doc.arc("TEMPLE", [0.0, 0.0], 3.0, 10.0, 200.0);
        doc.add(DxfEntity::Polyline {
            layer: "FRONT".into(),
            points: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            bulges: vec![1.0, 0.0],
            closed: false,
        });
        doc.add(DxfEntity::Polyline {
            layer: "3D".into(),
            points: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]],
            bulges: vec![],
            closed: false,
        });
        let read = parse_dxf(&write_dxf(&doc)).unwrap();
        assert_eq!(read.document.entities, doc.entities);
        assert!(read.skipped.is_empty());
        let names: Vec<_> = read.document.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["0", "FRONT", "TEMPLE", "3D"]);
        assert_eq!(read.document.layers[2].color, 5);
    }

    #[test]
    fn lwpolyline_with_bulge_is_a_semicircle() {
        let text = "0\nSECTION\n2\nENTITIES\n0\nLWPOLYLINE\n8\nCUT\n90\n2\n70\n0\n\
                    10\n0.0\n20\n0.0\n42\n1.0\n10\n2.0\n20\n0.0\n0\nSPLINE\n8\nCUT\n\
                    0\nENDSEC\n0\nEOF\n";
        let read = parse_dxf(text).unwrap();
        assert_eq!(read.skipped.get("SPLINE"), Some(&1));
        let pts = read.document.entities[0].flatten(1e-4);
        assert_eq!(pts.first(), Some(&[0.0, 0.0, 0.0]));
        assert_eq!(pts.last(), Some(&[2.0, 0.0, 0.0]));
        for p in &pts {
            let r = ((p[0] - 1.0).powi(2) + p[1].powi(2)).sqrt();
            assert!((r - 1.0).abs() < 1e-9, "on the unit circle about (1, 0)");
            assert!(p[1] <= 1e-12, "a positive bulge turns counter-clockwise: below");
        }
        assert!(pts.len() > 20);
    }

    #[test]
    fn flattening_keeps_to_the_tolerance() {
        let c = DxfEntity::Circle {
            layer: "0".into(),
            center: [0.0, 0.0, 0.0],
            radius: 10.0,
        };
        let pts = c.flatten(0.01);
        let n = pts.len();
        for i in 0..n {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            let mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0);
            let sag = 10.0 - (mid.0 * mid.0 + mid.1 * mid.1).sqrt();
            assert!(sag <= 0.01 + 1e-12);
        }
    }

    #[test]
    fn rejects_garbage() {
        assert!(matches!(parse_dxf("SECTION\n0\n"), Err(DxfError::BadGroupCode { line: 1, .. })));
        assert_eq!(parse_dxf("AutoCAD Binary DXF\r\n"), Err(DxfError::Binary));
    }
}
