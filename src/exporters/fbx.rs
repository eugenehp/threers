//! Binary FBX 7.4 — what Blender, Maya, 3ds Max, Unity, Unreal and Rhino
//! read: each mesh object as a `Model` with its `Geometry` (world-space
//! vertices, triangles, per-corner normals) and, when given a colour, a
//! `Material`, in a Y-up scene.
//!
//! The file is the FBX SDK's node tree: `GlobalSettings` (axes and the unit),
//! `Definitions`, `Objects` and the `Connections` tying geometry and material
//! to model and model to the scene root. Arrays are written uncompressed,
//! which every reader accepts.

use super::{apply_matrix4, attr3, count, normal_matrix, transform_normal, ExportKind, ExportObject};

/// What [`write_fbx_binary`] writes beyond the geometry.
#[derive(Debug, Clone)]
pub struct FbxOptions {
    /// Centimetres per scene unit (FBX's own unit is the centimetre): 0.1 for
    /// a scene in millimetres, 100 for metres.
    pub unit_scale_factor: f64,
    /// A diffuse colour per object, linear RGB in 0 to 1, by index; objects
    /// past the end get no material.
    pub colors: Vec<[f32; 3]>,
}

impl Default for FbxOptions {
    fn default() -> Self {
        Self {
            unit_scale_factor: 1.0,
            colors: vec![],
        }
    }
}

const VERSION: u32 = 7400;
const FOOTER_ID: [u8; 16] = [
    0xfa, 0xbc, 0xab, 0x09, 0xd0, 0xc8, 0xd4, 0x66, 0xb1, 0x76, 0xfb, 0x83, 0x1c, 0xf7, 0x26, 0x7e,
];
const FOOTER_MAGIC: [u8; 16] = [
    0xf8, 0x5a, 0x8c, 0x6a, 0xde, 0xf5, 0xd9, 0x7e, 0xec, 0xe9, 0x0c, 0xe3, 0x75, 0x8f, 0x29, 0x0b,
];

/// A property of a node, as FBX types them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Prop {
    Bool(bool),
    I32(i32),
    I64(i64),
    F64(f64),
    Str(String),
    F64s(Vec<f64>),
    I32s(Vec<i32>),
}

/// A node: a name, its properties, and the nodes nested in it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node {
    pub name: String,
    pub props: Vec<Prop>,
    pub children: Vec<Node>,
}

fn node(name: &str, props: Vec<Prop>, children: Vec<Node>) -> Node {
    Node {
        name: name.into(),
        props,
        children,
    }
}

fn s(v: &str) -> Prop {
    Prop::Str(v.into())
}

/// A `P` entry of a `Properties70` block.
fn p70(name: &str, kind: &str, label: &str, values: Vec<Prop>) -> Node {
    let mut props = vec![s(name), s(kind), s(label), s("")];
    props.extend(values);
    node("P", props, vec![])
}

impl Prop {
    fn write(&self, out: &mut Vec<u8>) {
        let array = |out: &mut Vec<u8>, code: u8, n: usize, bytes: Vec<u8>| {
            out.push(code);
            out.extend((n as u32).to_le_bytes());
            out.extend(0u32.to_le_bytes()); // not compressed
            out.extend((bytes.len() as u32).to_le_bytes());
            out.extend(bytes);
        };
        match self {
            Prop::Bool(v) => out.extend([b'C', *v as u8]),
            Prop::I32(v) => {
                out.push(b'I');
                out.extend(v.to_le_bytes());
            }
            Prop::I64(v) => {
                out.push(b'L');
                out.extend(v.to_le_bytes());
            }
            Prop::F64(v) => {
                out.push(b'D');
                out.extend(v.to_le_bytes());
            }
            Prop::Str(v) => {
                out.push(b'S');
                out.extend((v.len() as u32).to_le_bytes());
                out.extend(v.as_bytes());
            }
            Prop::F64s(v) => array(out, b'd', v.len(), v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            Prop::I32s(v) => array(out, b'i', v.len(), v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        }
    }
}

impl Node {
    /// The node record at `out.len()`: end offset, property count and
    /// length, name, properties, then the nested nodes closed by a null
    /// record.
    fn write(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend([0u8; 12]);
        out.push(self.name.len() as u8);
        out.extend(self.name.as_bytes());
        let props_start = out.len();
        for p in &self.props {
            p.write(out);
        }
        let props_len = out.len() - props_start;
        if !self.children.is_empty() {
            for c in &self.children {
                c.write(out);
            }
            out.extend([0u8; 13]);
        }
        let end = out.len() as u32;
        out[start..start + 4].copy_from_slice(&end.to_le_bytes());
        out[start + 4..start + 8].copy_from_slice(&(self.props.len() as u32).to_le_bytes());
        out[start + 8..start + 12].copy_from_slice(&(props_len as u32).to_le_bytes());
    }
}

/// The mesh objects in `objects` as a binary FBX file. Lines and points are
/// left out: FBX has no plain line or point-cloud geometry every reader takes.
pub fn write_fbx_binary(objects: &[ExportObject], options: &FbxOptions) -> Vec<u8> {
    let meshes: Vec<(usize, &ExportObject)> = objects
        .iter()
        .enumerate()
        .filter(|(_, o)| o.kind == ExportKind::Mesh && count(&o.geometry, "position").is_some())
        .collect();
    let mut nodes = vec![];
    let mut connections = vec![];
    let mut materials = 0;
    for (k, o) in &meshes {
        let model_id = 1_000_000 + *k as i64 * 10;
        let (geometry_id, material_id) = (model_id + 1, model_id + 2);
        let name = if o.name.is_empty() { format!("mesh{k}") } else { o.name.clone() };
        nodes.push(geometry(geometry_id, &name, o));
        nodes.push(node(
            "Model",
            vec![Prop::I64(model_id), s(&format!("{name}\0\u{1}Model")), s("Mesh")],
            vec![
                node("Version", vec![Prop::I32(232)], vec![]),
                node("Properties70", vec![], vec![]),
                node("Shading", vec![Prop::Bool(true)], vec![]),
                node("Culling", vec![s("CullingOff")], vec![]),
            ],
        ));
        connections.push(node("C", vec![s("OO"), Prop::I64(model_id), Prop::I64(0)], vec![]));
        connections.push(node("C", vec![s("OO"), Prop::I64(geometry_id), Prop::I64(model_id)], vec![]));
        if let Some(c) = options.colors.get(*k) {
            materials += 1;
            let rgb = c.map(|v| Prop::F64(v as f64)).to_vec();
            nodes.push(node(
                "Material",
                vec![Prop::I64(material_id), s(&format!("{name}\0\u{1}Material")), s("")],
                vec![
                    node("Version", vec![Prop::I32(102)], vec![]),
                    node("ShadingModel", vec![s("phong")], vec![]),
                    node("MultiLayer", vec![Prop::I32(0)], vec![]),
                    node(
                        "Properties70",
                        vec![],
                        vec![
                            p70("DiffuseColor", "Color", "", rgb.clone()),
                            p70("Diffuse", "Vector3D", "Vector", rgb),
                        ],
                    ),
                ],
            ));
            connections.push(node("C", vec![s("OO"), Prop::I64(material_id), Prop::I64(model_id)], vec![]));
        }
    }
    let definition = |kind: &str, n: usize| node("ObjectType", vec![s(kind)], vec![node("Count", vec![Prop::I32(n as i32)], vec![])]);
    let top = vec![
        node(
            "FBXHeaderExtension",
            vec![],
            vec![
                node("FBXHeaderVersion", vec![Prop::I32(1003)], vec![]),
                node("FBXVersion", vec![Prop::I32(VERSION as i32)], vec![]),
                node("Creator", vec![s("threers")], vec![]),
            ],
        ),
        node(
            "GlobalSettings",
            vec![],
            vec![
                node("Version", vec![Prop::I32(1000)], vec![]),
                node(
                    "Properties70",
                    vec![],
                    vec![
                        p70("UpAxis", "int", "Integer", vec![Prop::I32(1)]),
                        p70("UpAxisSign", "int", "Integer", vec![Prop::I32(1)]),
                        p70("FrontAxis", "int", "Integer", vec![Prop::I32(2)]),
                        p70("FrontAxisSign", "int", "Integer", vec![Prop::I32(1)]),
                        p70("CoordAxis", "int", "Integer", vec![Prop::I32(0)]),
                        p70("CoordAxisSign", "int", "Integer", vec![Prop::I32(1)]),
                        p70("UnitScaleFactor", "double", "Number", vec![Prop::F64(options.unit_scale_factor)]),
                        p70("OriginalUnitScaleFactor", "double", "Number", vec![Prop::F64(options.unit_scale_factor)]),
                    ],
                ),
            ],
        ),
        node(
            "Definitions",
            vec![],
            vec![
                node("Version", vec![Prop::I32(100)], vec![]),
                node("Count", vec![Prop::I32((meshes.len() * 2 + materials) as i32)], vec![]),
                definition("Model", meshes.len()),
                definition("Geometry", meshes.len()),
                definition("Material", materials),
            ],
        ),
        node("Objects", vec![], nodes),
        node("Connections", vec![], connections),
    ];

    let mut out = b"Kaydara FBX Binary  \0\x1a\0".to_vec();
    out.extend(VERSION.to_le_bytes());
    for n in &top {
        n.write(&mut out);
    }
    out.extend([0u8; 13]);
    out.extend(FOOTER_ID);
    out.extend([0u8; 4]);
    let pad = match (16 - out.len() % 16) % 16 {
        0 => 16,
        p => p,
    };
    out.extend(vec![0u8; pad]);
    out.extend(VERSION.to_le_bytes());
    out.extend([0u8; 120]);
    out.extend(FOOTER_MAGIC);
    out
}

/// A `Geometry` of triangles: world-space vertices, each triangle's corner
/// indices (the last one complemented, as FBX ends a polygon) and a normal
/// per corner.
fn geometry(id: i64, name: &str, o: &ExportObject) -> Node {
    let g = &*o.geometry;
    let n = count(g, "position").unwrap_or(0);
    let mut vertices = Vec::with_capacity(n * 3);
    for i in 0..n {
        vertices.extend(apply_matrix4(&o.matrix_world, attr3(g, "position", i)));
    }
    let index: Vec<u32> = g.index.clone().unwrap_or_else(|| (0..n as u32).collect());
    let has_normals = count(g, "normal") == Some(n);
    let nm = normal_matrix(&o.matrix_world);
    let mut polygon = Vec::with_capacity(index.len());
    let mut normals = Vec::with_capacity(index.len() * 3);
    for t in index.as_chunks::<3>().0 {
        polygon.extend([t[0] as i32, t[1] as i32, !(t[2] as i32)]);
        if has_normals {
            for &i in t {
                normals.extend(transform_normal(&nm, attr3(g, "normal", i as usize)));
            }
        }
    }
    let mut children = vec![
        node("Vertices", vec![Prop::F64s(vertices)], vec![]),
        node("PolygonVertexIndex", vec![Prop::I32s(polygon)], vec![]),
        node("GeometryVersion", vec![Prop::I32(124)], vec![]),
    ];
    let mut layer = vec![node("Version", vec![Prop::I32(100)], vec![])];
    if has_normals {
        children.push(node(
            "LayerElementNormal",
            vec![Prop::I32(0)],
            vec![
                node("Version", vec![Prop::I32(101)], vec![]),
                node("Name", vec![s("")], vec![]),
                node("MappingInformationType", vec![s("ByPolygonVertex")], vec![]),
                node("ReferenceInformationType", vec![s("Direct")], vec![]),
                node("Normals", vec![Prop::F64s(normals)], vec![]),
            ],
        ));
        layer.push(node(
            "LayerElement",
            vec![],
            vec![
                node("Type", vec![s("LayerElementNormal")], vec![]),
                node("TypedIndex", vec![Prop::I32(0)], vec![]),
            ],
        ));
    }
    children.push(node(
        "LayerElementMaterial",
        vec![Prop::I32(0)],
        vec![
            node("Version", vec![Prop::I32(101)], vec![]),
            node("Name", vec![s("")], vec![]),
            node("MappingInformationType", vec![s("AllSame")], vec![]),
            node("ReferenceInformationType", vec![s("IndexToDirect")], vec![]),
            node("Materials", vec![Prop::I32s(vec![0])], vec![]),
        ],
    ));
    layer.push(node(
        "LayerElement",
        vec![],
        vec![
            node("Type", vec![s("LayerElementMaterial")], vec![]),
            node("TypedIndex", vec![Prop::I32(0)], vec![]),
        ],
    ));
    children.push(node("Layer", vec![Prop::I32(0)], layer));
    node("Geometry", vec![Prop::I64(id), s(&format!("{name}\0\u{1}Geometry")), s("Mesh")], children)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::BoxGeometry;

    /// Reads the node tree back as a reader walks it, by the end offsets.
    fn read(bytes: &[u8]) -> Vec<Node> {
        fn props(b: &[u8], mut at: usize, n: usize) -> Vec<Prop> {
            let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
            let mut out = vec![];
            for _ in 0..n {
                let code = b[at];
                at += 1;
                match code {
                    b'C' => {
                        out.push(Prop::Bool(b[at] != 0));
                        at += 1;
                    }
                    b'I' => {
                        out.push(Prop::I32(i32::from_le_bytes(b[at..at + 4].try_into().unwrap())));
                        at += 4;
                    }
                    b'L' => {
                        out.push(Prop::I64(i64::from_le_bytes(b[at..at + 8].try_into().unwrap())));
                        at += 8;
                    }
                    b'D' => {
                        out.push(Prop::F64(f64::from_le_bytes(b[at..at + 8].try_into().unwrap())));
                        at += 8;
                    }
                    b'S' => {
                        let len = u32_at(at) as usize;
                        out.push(Prop::Str(String::from_utf8(b[at + 4..at + 4 + len].to_vec()).unwrap()));
                        at += 4 + len;
                    }
                    b'd' | b'i' => {
                        let (n, enc, len) = (u32_at(at) as usize, u32_at(at + 4), u32_at(at + 8) as usize);
                        assert_eq!(enc, 0);
                        let data = &b[at + 12..at + 12 + len];
                        out.push(if code == b'd' {
                            Prop::F64s(data.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
                        } else {
                            Prop::I32s(data.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
                        });
                        assert_eq!(len, n * if code == b'd' { 8 } else { 4 });
                        at += 12 + len;
                    }
                    other => panic!("property type {other}"),
                }
            }
            out
        }
        fn list(b: &[u8], mut at: usize, end: usize) -> (Vec<Node>, usize) {
            let mut nodes = vec![];
            while at + 13 <= end {
                let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
                let (node_end, n, plen) = (u32_at(at), u32_at(at + 4), u32_at(at + 8));
                if node_end == 0 {
                    return (nodes, at + 13);
                }
                let name_len = b[at + 12] as usize;
                let name = String::from_utf8(b[at + 13..at + 13 + name_len].to_vec()).unwrap();
                let props_at = at + 13 + name_len;
                let children = if props_at + plen < node_end {
                    let (c, after) = list(b, props_at + plen, node_end);
                    assert_eq!(after, node_end, "{name}'s children end where it says");
                    c
                } else {
                    vec![]
                };
                nodes.push(Node {
                    name,
                    props: props(b, props_at, n),
                    children,
                });
                at = node_end;
            }
            (nodes, at)
        }
        assert!(bytes.starts_with(b"Kaydara FBX Binary  \0\x1a\0"));
        assert!(bytes.ends_with(&FOOTER_MAGIC));
        list(bytes, 27, bytes.len()).0
    }

    #[test]
    fn a_box_reads_back() {
        let g = BoxGeometry::new(2.0, 2.0, 2.0);
        let mut o = ExportObject::mesh("box", g);
        // Moved 10 along x: the vertices are written in world space.
        o.matrix_world[12] = 10.0;
        let bytes = write_fbx_binary(&[o], &FbxOptions { unit_scale_factor: 0.1, colors: vec![[1.0, 0.5, 0.0]] });
        let top = read(&bytes);
        let names: Vec<&str> = top.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(names, ["FBXHeaderExtension", "GlobalSettings", "Definitions", "Objects", "Connections"]);
        let objects = &top[3].children;
        let geometry = objects.iter().find(|n| n.name == "Geometry").unwrap();
        let get = |n: &str| geometry.children.iter().find(|c| c.name == n).unwrap().props[0].clone();
        let Prop::F64s(v) = get("Vertices") else { panic!() };
        let Prop::I32s(idx) = get("PolygonVertexIndex") else { panic!() };
        assert_eq!(idx.len(), 36, "twelve triangles");
        assert_eq!(idx.iter().filter(|&&i| i < 0).count(), 12, "each closed by a complemented index");
        let xs: Vec<f64> = v.chunks(3).map(|c| c[0]).collect();
        assert!(xs.iter().all(|&x| (x - 9.0).abs() < 1e-6 || (x - 11.0).abs() < 1e-6));
        assert_eq!(objects.iter().filter(|n| n.name == "Material").count(), 1);
        assert_eq!(top[4].children.len(), 3, "model to root, geometry and material to model");
    }
}
