//! A `Geometry` node as a `BufferGeometry`.
//!
//! # How FBX stores a mesh
//!
//! `Vertices` is a flat array of control points. `PolygonVertexIndex` walks the
//! polygons, and the **last index of each polygon is stored bitwise-negated** —
//! that is the only thing marking where one face ends and the next begins, so a
//! reader that treats the array as triangles gets a different mesh.
//!
//! Everything else — normals, UVs, colours, material assignment — is a *layer
//! element*, and each says for itself how it is addressed:
//!
//! | `MappingInformationType` | indexed by |
//! |---|---|
//! | `ByPolygonVertex` | position in `PolygonVertexIndex` |
//! | `ByVertice` / `ByVertex` | control point |
//! | `ByPolygon` | polygon |
//! | `AllSame` | nothing; one value for the mesh |
//!
//! crossed with `ReferenceInformationType`, which is either `Direct` (the value
//! is at that index) or `IndexToDirect` (that index looks up an index array,
//! which then looks up the value). Reading a `ByVertice` layer as
//! `ByPolygonVertex` produces a mesh with plausible-looking wrong normals, so
//! the combination is honoured rather than assumed.
//!
//! Corners are emitted de-indexed — one vertex per polygon corner — because a
//! per-corner normal cannot survive being shared between faces, and a cube's
//! whole point is that its corners have three normals each.
//!
//! Polygons are fanned into triangles, and which triangles were one polygon is
//! kept in [`BufferGeometry::polygons`].

use super::tree::{Node, Property};
use crate::core::{BufferAttribute, BufferGeometry};

/// How a layer element is addressed.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mapping {
    PolygonVertex,
    Vertex,
    Polygon,
    AllSame,
}

impl Mapping {
    fn parse(name: Option<&str>) -> Self {
        match name {
            Some("ByVertice") | Some("ByVertex") | Some("ByVertices") => Mapping::Vertex,
            Some("ByPolygon") => Mapping::Polygon,
            Some("AllSame") | Some("AllSames") => Mapping::AllSame,
            // `ByPolygonVertex` is both the common case and the default.
            _ => Mapping::PolygonVertex,
        }
    }
}

/// One layer element, resolved enough to be asked for a value.
struct Layer {
    data: Vec<f32>,
    indices: Vec<i32>,
    mapping: Mapping,
    /// How many numbers make one value — 3 for a normal, 2 for a UV.
    stride: usize,
}

impl Layer {
    /// Read a layer element out of a `Geometry` node.
    ///
    /// `data_name` and `index_name` differ per element (`Normals`/
    /// `NormalsIndex`, `UV`/`UVIndex`), which is why they are passed rather
    /// than derived.
    fn read(node: &Node, data_name: &str, index_name: &str, stride: usize) -> Option<Self> {
        let data = node.value(data_name)?.as_f32_array()?;
        if data.is_empty() {
            return None;
        }
        let reference = node
            .value("ReferenceInformationType")
            .and_then(Property::as_str)
            .unwrap_or("Direct");
        // `Direct` addresses the data straight; anything else goes through the
        // index array. A file that says `IndexToDirect` and has no index array
        // is treated as direct rather than as empty.
        let indices = if reference == "Direct" {
            Vec::new()
        } else {
            node.value(index_name)
                .and_then(Property::as_i32_array)
                .unwrap_or_default()
        };
        Some(Layer {
            data,
            indices,
            mapping: Mapping::parse(
                node.value("MappingInformationType").and_then(Property::as_str),
            ),
            stride,
        })
    }

    /// The value for one polygon corner.
    fn at(&self, corner: usize, control_point: usize, polygon: usize) -> Vec<f32> {
        let slot = match self.mapping {
            Mapping::PolygonVertex => corner,
            Mapping::Vertex => control_point,
            Mapping::Polygon => polygon,
            Mapping::AllSame => 0,
        };
        let slot = if self.indices.is_empty() {
            slot
        } else {
            match self.indices.get(slot) {
                Some(&i) if i >= 0 => i as usize,
                // A negative index in a layer means "no value here", which FBX
                // uses for holes in a material assignment.
                _ => return vec![0.0; self.stride],
            }
        };
        let at = slot * self.stride;
        self.data
            .get(at..at + self.stride)
            .map(<[f32]>::to_vec)
            .unwrap_or_else(|| vec![0.0; self.stride])
    }
}

/// The per-polygon material slot, for a mesh split across several materials.
pub fn material_slots(node: &Node) -> Vec<i32> {
    node.child("LayerElementMaterial")
        .and_then(|l| l.value("Materials").and_then(Property::as_i32_array))
        .unwrap_or_default()
}

/// Whether the material layer maps one material to the whole mesh
/// (`AllSame`) — three.js `FBXLoader` makes geometry groups only otherwise.
pub fn material_mapping_is_all_same(node: &Node) -> bool {
    node.child("LayerElementMaterial")
        .and_then(|l| l.value("MappingInformationType"))
        .and_then(Property::as_str)
        .is_none_or(|m| m == "AllSame")
}

/// The control point behind each vertex [`build`] emits, in the same fan
/// order — what a blend shape, stored per control point, expands through.
pub fn corner_control_points(node: &Node) -> Vec<usize> {
    let corners = node
        .value("PolygonVertexIndex")
        .and_then(Property::as_i32_array)
        .unwrap_or_default();
    let mut out = Vec::new();
    let mut polygon: Vec<usize> = Vec::with_capacity(4);
    for &raw in &corners {
        polygon.push(if raw < 0 { !raw as usize } else { raw as usize });
        if raw >= 0 {
            continue;
        }
        for i in 1..polygon.len().saturating_sub(1) {
            out.extend_from_slice(&[polygon[0], polygon[i], polygon[i + 1]]);
        }
        polygon.clear();
    }
    out
}

/// Build a `BufferGeometry` from a `Geometry` node.
///
/// Returns the geometry and, for each emitted triangle, which material slot the
/// polygon it came from asked for — the caller needs that to split a
/// multi-material mesh.
pub fn build(node: &Node) -> Option<(BufferGeometry, Vec<i32>)> {
    let points = node.value("Vertices")?.as_f32_array()?;
    if points.is_empty() {
        return None;
    }
    let corners = node
        .value("PolygonVertexIndex")
        .and_then(Property::as_i32_array)
        .unwrap_or_default();

    let normals = node
        .child("LayerElementNormal")
        .and_then(|l| Layer::read(l, "Normals", "NormalsIndex", 3));
    let uvs = node
        .child("LayerElementUV")
        .and_then(|l| Layer::read(l, "UV", "UVIndex", 2));
    let colours = node
        .child("LayerElementColor")
        .and_then(|l| Layer::read(l, "Colors", "ColorIndex", 4));
    let slots = material_slots(node);

    let mut position = Vec::new();
    let mut normal = Vec::new();
    let mut uv = Vec::new();
    let mut colour = Vec::new();
    let mut triangle_slots = Vec::new();
    // Triangles per polygon, in polygon order.
    let mut polygons: Vec<u32> = Vec::new();

    // One polygon at a time. `corners` holds the control-point index of each
    // corner, with the last of each polygon bitwise-negated.
    let mut polygon: Vec<(usize, usize)> = Vec::with_capacity(4);
    let mut polygon_index = 0usize;
    for (corner, &raw) in corners.iter().enumerate() {
        let control_point = if raw < 0 { !raw as usize } else { raw as usize };
        polygon.push((corner, control_point));
        if raw >= 0 {
            continue;
        }
        // Fan from the first corner. FBX polygons are planar and convex in
        // practice; a concave one would need a real triangulator, and getting
        // that wrong is visible as a spike rather than as a silent error.
        if polygon.len() >= 3 {
            polygons.push(polygon.len() as u32 - 2);
        }
        for i in 1..polygon.len().saturating_sub(1) {
            for &(corner, point) in &[polygon[0], polygon[i], polygon[i + 1]] {
                let at = point * 3;
                match points.get(at..at + 3) {
                    Some(p) => position.extend_from_slice(p),
                    // An index past the end of the point array: emit the origin
                    // rather than dropping one corner of a triangle, which
                    // would shift every following face.
                    None => position.extend_from_slice(&[0.0, 0.0, 0.0]),
                }
                if let Some(layer) = &normals {
                    normal.extend_from_slice(&layer.at(corner, point, polygon_index));
                }
                if let Some(layer) = &uvs {
                    uv.extend_from_slice(&layer.at(corner, point, polygon_index));
                }
                if let Some(layer) = &colours {
                    colour.extend_from_slice(&layer.at(corner, point, polygon_index));
                }
            }
            triangle_slots.push(slots.get(polygon_index).copied().unwrap_or(0));
        }
        polygon.clear();
        polygon_index += 1;
    }

    // No polygons at all: a point cloud is still geometry.
    if position.is_empty() {
        position = points;
    }

    let mut geometry = BufferGeometry::new();
    geometry.set_attribute("position", BufferAttribute::new(position, 3));
    if !normal.is_empty() {
        geometry.set_attribute("normal", BufferAttribute::new(normal, 3));
    }
    if !uv.is_empty() {
        geometry.set_attribute("uv", BufferAttribute::new(uv, 2));
    }
    if !colour.is_empty() {
        // FBX colours are RGBA; this crate's vertex colours are RGB.
        let rgb: Vec<f32> = colour
            .chunks_exact(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect();
        geometry.set_attribute("color", BufferAttribute::new(rgb, 3));
    }
    if geometry.get_attribute("normal").is_none() {
        crate::compute_vertex_normals(&mut geometry);
    }
    // Rejected, harmlessly, for the point-cloud case above.
    geometry.set_polygons(polygons);
    Some((geometry, triangle_slots))
}
