//! GeoJSON (RFC 7946) in, mark columns out.
//!
//! [`FeatureCollection::parse`] reads a collection, a single feature or a bare
//! geometry. [`FeatureCollection::shapes`] then projects every coordinate and
//! sorts the geometry into what each mark takes — points, paths, polygons and
//! polygon outlines — remembering which feature every row came from, so a
//! property can colour, size or filter it:
//!
//! ```
//! use threers::marks::geojson::FeatureCollection;
//! use threers::marks::{geo, PolygonMarks};
//!
//! let text = r#"{"type": "FeatureCollection", "features": [
//!   {"type": "Feature", "properties": {"name": "São Paulo", "pop": 12.3},
//!    "geometry": {"type": "Polygon", "coordinates":
//!      [[[-46.8, -23.7], [-46.4, -23.7], [-46.4, -23.4], [-46.8, -23.4], [-46.8, -23.7]]]}}
//! ]}"#;
//! let fc = FeatureCollection::parse(text).unwrap();
//! assert_eq!(fc.features[0].string("name"), Some("São Paulo"));
//!
//! let shapes = fc.shapes(geo::lon_lat_to_world);
//! let mut fills = PolygonMarks::new();
//! fills.set_polygons(&shapes.polygons, 0.0);
//! // Population per feature, spread to each feature's polygons.
//! let pop: Vec<f32> = fc.numbers("pop").iter().map(|p| p.unwrap_or(0.0) as f32).collect();
//! fills.set_elevations(&shapes.per_polygon(&pop));
//! ```
//!
//! Coordinates are `[longitude, latitude]` with an optional altitude. Lines
//! and polygons crossing the antimeridian are not split.

use std::collections::BTreeMap;

use super::geo::world_to_ground;
use super::PolygonFeature;
use crate::loaders::json::{self, Value};

/// A coordinate: longitude, latitude, altitude (0 where absent).
pub type Position = [f64; 3];

#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    Point(Position),
    MultiPoint(Vec<Position>),
    LineString(Vec<Position>),
    MultiLineString(Vec<Vec<Position>>),
    /// The outer ring, then holes.
    Polygon(Vec<Vec<Position>>),
    MultiPolygon(Vec<Vec<Vec<Position>>>),
    GeometryCollection(Vec<Geometry>),
}

/// A property value. Arrays and objects are kept as their JSON text.
#[derive(Debug, Clone, PartialEq)]
pub enum Property {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Json(String),
}

impl Property {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Property::Number(n) => Some(*n),
            Property::Bool(b) => Some(f64::from(u8::from(*b))),
            // Numbers in CSV-born GeoJSON often arrive quoted.
            Property::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Property::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Feature {
    /// `None` for a feature with a `null` geometry, which the spec allows.
    pub geometry: Option<Geometry>,
    pub properties: BTreeMap<String, Property>,
    pub id: Option<Property>,
}

impl Feature {
    pub fn number(&self, key: &str) -> Option<f64> {
        self.properties.get(key).and_then(Property::as_f64)
    }
    pub fn string(&self, key: &str) -> Option<&str> {
        self.properties.get(key).and_then(Property::as_str)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FeatureCollection {
    pub features: Vec<Feature>,
}

/// Geometry sorted by the mark that draws it, in world coordinates, with the
/// feature each row came from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GeoShapes {
    /// For [`PointMarks`](super::PointMarks).
    pub points: Vec<[f64; 3]>,
    pub point_features: Vec<usize>,
    /// For [`SegmentMarks::set_paths`](super::SegmentMarks::set_paths).
    pub paths: Vec<Vec<[f64; 3]>>,
    pub path_features: Vec<usize>,
    /// For [`PolygonMarks::set_polygons`](super::PolygonMarks::set_polygons),
    /// one per polygon (a multipolygon gives several).
    pub polygons: Vec<PolygonFeature>,
    pub polygon_features: Vec<usize>,
    /// Every polygon ring, closed, as paths at the ring's own height — the
    /// strokes around the fills.
    pub outlines: Vec<Vec<[f64; 3]>>,
    pub outline_features: Vec<usize>,
}

fn per_row<T: Copy>(rows: &[usize], per_feature: &[T]) -> Vec<T> {
    rows.iter()
        .filter_map(|&f| per_feature.get(f).copied())
        .collect()
}

impl GeoShapes {
    /// A value per feature, as one per point.
    pub fn per_point<T: Copy>(&self, per_feature: &[T]) -> Vec<T> {
        per_row(&self.point_features, per_feature)
    }
    /// A value per feature, as one per path — what
    /// [`set_path_colors`](super::SegmentMarks::set_path_colors) and
    /// [`set_path_widths`](super::SegmentMarks::set_path_widths) take.
    pub fn per_path<T: Copy>(&self, per_feature: &[T]) -> Vec<T> {
        per_row(&self.path_features, per_feature)
    }
    /// A value per feature, as one per polygon.
    pub fn per_polygon<T: Copy>(&self, per_feature: &[T]) -> Vec<T> {
        per_row(&self.polygon_features, per_feature)
    }
    /// A value per feature, as one per outline.
    pub fn per_outline<T: Copy>(&self, per_feature: &[T]) -> Vec<T> {
        per_row(&self.outline_features, per_feature)
    }
}

impl FeatureCollection {
    /// Read a `FeatureCollection`, a `Feature`, or a bare geometry (which
    /// becomes one feature with no properties).
    pub fn parse(text: &str) -> Result<Self, String> {
        let v = json::parse(text).map_err(|e| format!("GeoJSON: {e}"))?;
        let kind = member_str(&v, "type")?;
        let features = match kind {
            "FeatureCollection" => v
                .get("features")
                .and_then(Value::as_array)
                .ok_or("GeoJSON: a FeatureCollection without a \"features\" array")?
                .iter()
                .enumerate()
                .map(|(i, f)| feature(f).map_err(|e| format!("feature {i}: {e}")))
                .collect::<Result<_, _>>()?,
            "Feature" => vec![feature(&v)?],
            _ => vec![Feature {
                geometry: Some(geometry(&v)?),
                ..Default::default()
            }],
        };
        Ok(Self { features })
    }

    /// `key` of every feature as a number, `None` where it is missing or not
    /// numeric.
    pub fn numbers(&self, key: &str) -> Vec<Option<f64>> {
        self.features.iter().map(|f| f.number(key)).collect()
    }

    /// `key` of every feature as a string.
    pub fn strings(&self, key: &str) -> Vec<Option<&str>> {
        self.features.iter().map(|f| f.string(key)).collect()
    }

    /// Every geometry, projected to world coordinates by `project` — usually
    /// [`geo::lon_lat_to_world`](super::geo::lon_lat_to_world) — and sorted
    /// by the mark that draws it.
    pub fn shapes(&self, project: impl Fn(Position) -> [f64; 3]) -> GeoShapes {
        let mut out = GeoShapes::default();
        for (k, f) in self.features.iter().enumerate() {
            if let Some(g) = &f.geometry {
                collect(g, k, &project, &mut out);
            }
        }
        out
    }
}

fn collect(g: &Geometry, k: usize, project: &impl Fn(Position) -> [f64; 3], out: &mut GeoShapes) {
    let line = |l: &[Position]| -> Vec<[f64; 3]> { l.iter().map(|&p| project(p)).collect() };
    let polygon = |rings: &[Vec<Position>], out: &mut GeoShapes| {
        let Some(outer) = rings.first() else { return };
        let ground = |r: &[Position]| -> Vec<[f64; 2]> {
            r.iter().map(|&p| world_to_ground(project(p)).0).collect()
        };
        out.polygons.push(PolygonFeature {
            outer: ground(outer),
            holes: rings[1..].iter().map(|h| ground(h)).collect(),
        });
        out.polygon_features.push(k);
        for r in rings {
            let mut ring = line(r);
            if ring.len() > 1 && ring.first() != ring.last() {
                ring.push(ring[0]);
            }
            out.outlines.push(ring);
            out.outline_features.push(k);
        }
    };
    match g {
        Geometry::Point(p) => {
            out.points.push(project(*p));
            out.point_features.push(k);
        }
        Geometry::MultiPoint(ps) => {
            for p in ps {
                out.points.push(project(*p));
                out.point_features.push(k);
            }
        }
        Geometry::LineString(l) => {
            out.paths.push(line(l));
            out.path_features.push(k);
        }
        Geometry::MultiLineString(ls) => {
            for l in ls {
                out.paths.push(line(l));
                out.path_features.push(k);
            }
        }
        Geometry::Polygon(rings) => polygon(rings, out),
        Geometry::MultiPolygon(polys) => {
            for rings in polys {
                polygon(rings, out);
            }
        }
        Geometry::GeometryCollection(gs) => {
            for g in gs {
                collect(g, k, project, out);
            }
        }
    }
}

fn member_str<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("GeoJSON: no \"{key}\" string"))
}

fn property(v: &Value) -> Property {
    match v {
        Value::Null => Property::Null,
        Value::Bool(b) => Property::Bool(*b),
        Value::Number(n) => Property::Number(*n),
        Value::String(s) => Property::String(s.clone()),
        other => Property::Json(other.to_json()),
    }
}

fn feature(v: &Value) -> Result<Feature, String> {
    if member_str(v, "type")? != "Feature" {
        return Err("not a Feature".into());
    }
    let geometry = match v.get("geometry") {
        None | Some(Value::Null) => None,
        Some(g) => Some(geometry(g)?),
    };
    let properties = v
        .get("properties")
        .and_then(Value::as_object)
        .map(|o| o.iter().map(|(k, v)| (k.clone(), property(v))).collect())
        .unwrap_or_default();
    Ok(Feature {
        geometry,
        properties,
        id: v.get("id").map(property),
    })
}

fn position(v: &Value) -> Result<Position, String> {
    let a = v.as_array().ok_or("a position is not an array")?;
    let n = |i: usize| a.get(i).and_then(Value::as_number);
    match (n(0), n(1)) {
        (Some(x), Some(y)) => Ok([x, y, n(2).unwrap_or(0.0)]),
        _ => Err("a position needs two numbers".into()),
    }
}

fn array(v: &Value) -> Result<&Vec<Value>, String> {
    v.as_array()
        .ok_or_else(|| "coordinates are not an array".into())
}

fn positions(v: &Value) -> Result<Vec<Position>, String> {
    array(v)?.iter().map(position).collect()
}

fn rings(v: &Value) -> Result<Vec<Vec<Position>>, String> {
    array(v)?.iter().map(positions).collect()
}

fn geometry(v: &Value) -> Result<Geometry, String> {
    let kind = member_str(v, "type")?;
    if kind == "GeometryCollection" {
        let gs = v
            .get("geometries")
            .and_then(Value::as_array)
            .ok_or("a GeometryCollection without \"geometries\"")?;
        return Ok(Geometry::GeometryCollection(
            gs.iter().map(geometry).collect::<Result<_, _>>()?,
        ));
    }
    let c = v
        .get("coordinates")
        .ok_or_else(|| format!("a {kind} without \"coordinates\""))?;
    Ok(match kind {
        "Point" => Geometry::Point(position(c)?),
        "MultiPoint" => Geometry::MultiPoint(positions(c)?),
        "LineString" => Geometry::LineString(positions(c)?),
        "MultiLineString" => Geometry::MultiLineString(rings(c)?),
        "Polygon" => Geometry::Polygon(rings(c)?),
        "MultiPolygon" => {
            Geometry::MultiPolygon(array(c)?.iter().map(rings).collect::<Result<_, _>>()?)
        }
        other => return Err(format!("unknown geometry type {other:?}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
      "type": "FeatureCollection",
      "features": [
        {"type": "Feature", "id": 7,
         "properties": {"name": "Zürich", "pop": "415000", "tags": ["a", 1]},
         "geometry": {"type": "Point", "coordinates": [8.54, 47.37, 408]}},
        {"type": "Feature", "properties": {"pop": 2},
         "geometry": {"type": "MultiPolygon", "coordinates": [
           [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]], [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]],
           [[[10, 10], [11, 10], [11, 11], [10, 10]]]
         ]}},
        {"type": "Feature", "properties": null, "geometry": null},
        {"type": "Feature", "properties": {"pop": 3},
         "geometry": {"type": "GeometryCollection", "geometries": [
           {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
           {"type": "MultiPoint", "coordinates": [[5, 5], [6, 6]]}
         ]}}
      ]
    }"#;

    #[test]
    fn parses_every_kind_and_property() {
        let fc = FeatureCollection::parse(SAMPLE).unwrap();
        assert_eq!(fc.features.len(), 4);
        let f = &fc.features[0];
        assert_eq!(f.string("name"), Some("Zürich"));
        assert_eq!(
            f.number("pop"),
            Some(415_000.0),
            "quoted numbers read as numbers"
        );
        assert_eq!(f.properties["tags"], Property::Json(r#"["a",1]"#.into()));
        assert_eq!(f.id, Some(Property::Number(7.0)));
        assert_eq!(f.geometry, Some(Geometry::Point([8.54, 47.37, 408.0])));
        assert!(fc.features[2].geometry.is_none());
        assert_eq!(
            fc.numbers("pop"),
            vec![Some(415_000.0), Some(2.0), None, Some(3.0)]
        );
    }

    #[test]
    fn shapes_sort_by_mark_and_remember_their_feature() {
        let fc = FeatureCollection::parse(SAMPLE).unwrap();
        // Identity on the ground: lon, lat become x, y.
        let s = fc.shapes(|p| super::super::geo::ground_to_world([p[0], p[1]], p[2]));
        assert_eq!(s.point_features, vec![0, 3, 3]);
        assert_eq!(s.points[0], [8.54, 408.0, -47.37]);
        assert_eq!(s.path_features, vec![3]);
        assert_eq!(s.polygon_features, vec![1, 1]);
        assert_eq!(s.polygons[0].holes.len(), 1);
        assert_eq!(s.polygons[0].outer[2], [4.0, 4.0]);
        // Three rings across the two polygons, each closed.
        assert_eq!(s.outline_features, vec![1, 1, 1]);
        assert!(s.outlines.iter().all(|r| r.first() == r.last()));
        // Per-feature values spread to rows.
        assert_eq!(s.per_polygon(&['a', 'b', 'c', 'd']), vec!['b', 'b']);
        assert_eq!(s.per_point(&[1, 2, 3, 4]), vec![1, 4, 4]);
    }

    #[test]
    fn bare_geometry_and_errors() {
        let fc =
            FeatureCollection::parse(r#"{"type": "LineString", "coordinates": [[0, 0], [1, 2]]}"#)
                .unwrap();
        assert_eq!(fc.features.len(), 1);
        let err = FeatureCollection::parse(
            r#"{"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Blob", "coordinates": []}}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("feature 0") && err.contains("Blob"), "{err}");
        assert!(FeatureCollection::parse("{").is_err());
    }
}
