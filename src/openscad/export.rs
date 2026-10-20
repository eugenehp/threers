//! Mesh **exporters** for the OpenSCAD front end — the counterparts to the
//! importers in `scad::import`.
//!
//! All operate on an evaluated [`BufferGeometry`] (indexed or triangle-soup) and
//! are pure Rust with no dependencies:
//! - `geometry_to_obj` — Wavefront **OBJ** (text; positions, normals, faces).
//! - `geometry_to_off` — Geomview **OFF** (text).
//! - `geometry_to_ply` — Stanford **PLY** (ASCII).
//! - `geometry_to_3mf` — **3MF** (a proper OPC/ZIP package a slicer can open).
//! - `geometry_to_glb` — binary **glTF 2.0** (`.glb`; single self-contained file).
//!
//! OBJ, OFF and PLY write a face per polygon the geometry remembers (quads as
//! quads, see [`BufferGeometry::faces`]); 3MF, STL and glTF are triangle-only
//! formats and write triangles.
//!
//! OBJ, PLY and STL are thin wrappers over the general writers in
//! [`crate::exporters`] — the same code the browser's `OBJExporter`,
//! `PLYExporter` and `STLExporter` run — with the compact `f32` number format.
//!
//! FreeCAD `.FCStd` lives in [`crate::openscad::freecad`].
//!
//! [`Solid`](crate::Solid) grows `to_obj`/`to_off`/`to_3mf`/`to_glb`/`to_fcstd`
//! convenience methods that evaluate with the exact-where-confident kernel first.

use crate::core::BufferGeometry;

/// Normalise a geometry to `(vertices, optional per-vertex normals, triangle
/// indices)`. A non-indexed soup gets sequential indices; normals are dropped
/// unless there is exactly one per vertex.
type TriData = (Vec<[f32; 3]>, Option<Vec<[f32; 3]>>, Vec<u32>);

fn tri_data(g: &BufferGeometry) -> Option<TriData> {
    let pos = &g.get_attribute("position")?.array;
    let verts: Vec<[f32; 3]> = pos.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    if verts.len() < 3 {
        return None;
    }
    let normals = g
        .get_attribute("normal")
        .map(|a| {
            a.array
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect::<Vec<_>>()
        })
        .filter(|n: &Vec<[f32; 3]>| n.len() == verts.len());
    let indices = g
        .index
        .clone()
        .unwrap_or_else(|| (0..verts.len() as u32).collect());
    Some((verts, normals, indices))
}

/// The faces a polygon-capable format writes: [`BufferGeometry::faces`] —
/// one face per polygon the geometry remembers and whose outline is
/// recoverable, its triangles otherwise — the same rule the USD exporter
/// follows.
fn faces(g: &BufferGeometry) -> Vec<Vec<u32>> {
    g.faces().collect()
}

/// Export as **Wavefront OBJ** (text). Emits `v`, optional `vt` and `vn`, and
/// 1-indexed `f` lines (`v/vt/vn`, `v//vn` or `v` as available).
///
/// A face is written per polygon the geometry remembers
/// ([`BufferGeometry::polygons`]), with as many corners as it had — a quad is
/// one `f` line of four — and per triangle where there is no grouping or a
/// polygon's outline cannot be recovered. [`crate::ObjLoader`] reads the
/// grouping back.
pub fn geometry_to_obj(g: &BufferGeometry) -> String {
    if tri_data(g).is_none() {
        return String::new();
    }
    let mut s = String::from("# threers OBJ export\n");
    s.push_str(&crate::exporters::write_obj(
        &[crate::exporters::ExportObject::mesh("model", g.clone())],
        &crate::exporters::ObjOptions {
            polygons: true,
            numbers: crate::exporters::NumberFormat::Float32,
        },
    ));
    s
}

/// Export as **Geomview OFF** (text), one face per polygon as
/// [`geometry_to_obj`] does.
pub fn geometry_to_off(g: &BufferGeometry) -> String {
    let Some((verts, _, _)) = tri_data(g) else {
        return String::from("OFF\n0 0 0\n");
    };
    let faces = faces(g);
    let mut s = format!("OFF\n{} {} 0\n", verts.len(), faces.len());
    for v in &verts {
        s.push_str(&format!("{} {} {}\n", v[0], v[1], v[2]));
    }
    for face in &faces {
        s.push_str(&face.len().to_string());
        for i in face {
            s.push_str(&format!(" {i}"));
        }
        s.push('\n');
    }
    s
}

/// Export as ASCII **Stanford PLY**: `x y z`, `nx ny nz` and `s t` per vertex
/// where the geometry has one per vertex, and one `vertex_indices` list per
/// polygon as [`geometry_to_obj`] does. [`crate::PlyLoader`] reads it back,
/// grouping included.
pub fn geometry_to_ply(g: &BufferGeometry) -> String {
    let bytes = crate::exporters::write_ply(
        &[crate::exporters::ExportObject::mesh("model", g.clone())],
        &crate::exporters::PlyOptions {
            polygons: true,
            numbers: crate::exporters::NumberFormat::Float32,
            comment: Some("threers PLY export".into()),
            ..Default::default()
        },
    );
    // `None` only for an unindexed vertex count that is not a whole number
    // of triangles: write the vertices alone, which is still a valid PLY.
    let bytes = bytes.or_else(|| {
        crate::exporters::write_ply(
            &[crate::exporters::ExportObject::mesh("model", g.clone())],
            &crate::exporters::PlyOptions {
                exclude_attributes: vec!["index".into()],
                numbers: crate::exporters::NumberFormat::Float32,
                comment: Some("threers PLY export".into()),
                ..Default::default()
            },
        )
    });
    String::from_utf8(bytes.unwrap_or_default()).expect("ASCII PLY is UTF-8")
}

// --- 3MF (OPC/ZIP package) --------------------------------------------------

/// Export as **3MF** — a spec-shaped OPC package (`[Content_Types].xml`,
/// `_rels/.rels`, `3D/3dmodel.model`) zipped with stored (uncompressed) entries
/// and correct CRC-32s, so real slicers open it.
pub fn geometry_to_3mf(g: &BufferGeometry) -> Vec<u8> {
    let (verts, _, indices) = tri_data(g).unwrap_or_default();
    let mut model = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <model unit=\"millimeter\" xml:lang=\"en-US\" \
         xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n\
         <resources><object id=\"1\" type=\"model\"><mesh><vertices>",
    );
    for v in &verts {
        model.push_str(&format!(
            "<vertex x=\"{}\" y=\"{}\" z=\"{}\"/>",
            v[0], v[1], v[2]
        ));
    }
    model.push_str("</vertices><triangles>");
    for t in indices.chunks_exact(3) {
        model.push_str(&format!(
            "<triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>",
            t[0], t[1], t[2]
        ));
    }
    model.push_str(
        "</triangles></mesh></object></resources><build><item objectid=\"1\"/></build></model>",
    );

    let content_types = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
        <Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\
        <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\
        <Default Extension=\"model\" ContentType=\"application/vnd.ms-package.3dmanufacturing-3dmodel+xml\"/>\
        </Types>";
    let rels = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
        <Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
        <Relationship Target=\"/3D/3dmodel.model\" Id=\"rel0\" \
        Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel\"/></Relationships>";

    let mut zip = Zip::new();
    zip.add("[Content_Types].xml", content_types.as_bytes());
    zip.add("_rels/.rels", rels.as_bytes());
    zip.add("3D/3dmodel.model", model.as_bytes());
    zip.finish()
}

/// Minimal single-purpose ZIP writer: stored (method 0) entries with correct
/// CRC-32s and a central directory. Enough for a valid 3MF / FCStd package.
pub(crate) struct Zip {
    out: Vec<u8>,
    dir: Vec<u8>,
    count: u16,
}
impl Zip {
    pub(crate) fn new() -> Self {
        Self {
            out: Vec::new(),
            dir: Vec::new(),
            count: 0,
        }
    }
    pub(crate) fn add(&mut self, name: &str, data: &[u8]) {
        let crc = crc32(data);
        let (nl, dl) = (name.len() as u16, data.len() as u32);
        let offset = self.out.len() as u32;
        // local file header
        self.out.extend_from_slice(b"PK\x03\x04");
        self.out.extend_from_slice(&[20, 0, 0, 0, 0, 0]); // version, flags, method=store
        self.out.extend_from_slice(&[0, 0, 0, 0]); // mod time/date
        self.out.extend_from_slice(&crc.to_le_bytes());
        self.out.extend_from_slice(&dl.to_le_bytes());
        self.out.extend_from_slice(&dl.to_le_bytes());
        self.out.extend_from_slice(&nl.to_le_bytes());
        self.out.extend_from_slice(&[0, 0]); // extra len
        self.out.extend_from_slice(name.as_bytes());
        self.out.extend_from_slice(data);
        // central directory record
        self.dir.extend_from_slice(b"PK\x01\x02");
        self.dir.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0]); // versions, flags, method
        self.dir.extend_from_slice(&[0, 0, 0, 0]); // mod time/date
        self.dir.extend_from_slice(&crc.to_le_bytes());
        self.dir.extend_from_slice(&dl.to_le_bytes());
        self.dir.extend_from_slice(&dl.to_le_bytes());
        self.dir.extend_from_slice(&nl.to_le_bytes());
        self.dir.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // extra, comment, disk#
        self.dir.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // int/ext attrs
        self.dir.extend_from_slice(&offset.to_le_bytes());
        self.dir.extend_from_slice(name.as_bytes());
        self.count += 1;
    }
    pub(crate) fn finish(mut self) -> Vec<u8> {
        let cd_off = self.out.len() as u32;
        let cd_size = self.dir.len() as u32;
        self.out.extend_from_slice(&self.dir);
        self.out.extend_from_slice(b"PK\x05\x06");
        self.out.extend_from_slice(&[0, 0, 0, 0]); // disk numbers
        self.out.extend_from_slice(&self.count.to_le_bytes());
        self.out.extend_from_slice(&self.count.to_le_bytes());
        self.out.extend_from_slice(&cd_size.to_le_bytes());
        self.out.extend_from_slice(&cd_off.to_le_bytes());
        self.out.extend_from_slice(&[0, 0]); // comment len
        self.out
    }
}

pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

// --- glTF 2.0 (binary .glb) -------------------------------------------------

/// Export as binary **glTF 2.0** (`.glb`) — a single self-contained file with a
/// JSON chunk and a BIN chunk (indices + positions + normals). Opens in any glTF
/// viewer (three.js, Blender, Windows 3D Viewer, …).
pub fn geometry_to_glb(g: &BufferGeometry) -> Vec<u8> {
    let Some((verts, normals, indices)) = tri_data(g) else {
        return Vec::new();
    };

    // Binary buffer: [indices u32][positions f32×3][normals f32×3?]. Each section
    // is a multiple of 4 bytes, so bufferView offsets stay 4-aligned automatically.
    let mut bin = Vec::new();
    for &i in &indices {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    let idx_len = bin.len();
    let pos_off = bin.len();
    for v in &verts {
        for c in v {
            bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    let pos_len = bin.len() - pos_off;
    let (nrm_off, nrm_len) = match &normals {
        Some(n) => {
            let off = bin.len();
            for v in n {
                for c in v {
                    bin.extend_from_slice(&c.to_le_bytes());
                }
            }
            (off, bin.len() - off)
        }
        None => (0, 0),
    };
    let buf_len = bin.len();

    // Position accessor requires min/max (glTF validators enforce it).
    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
    for v in &verts {
        for k in 0..3 {
            mn[k] = mn[k].min(v[k]);
            mx[k] = mx[k].max(v[k]);
        }
    }
    let arr = |a: [f32; 3]| format!("[{},{},{}]", a[0], a[1], a[2]);

    // bufferViews / accessors, with NORMAL only when present.
    let mut views = format!(
        "{{\"buffer\":0,\"byteOffset\":0,\"byteLength\":{idx_len},\"target\":34963}},\
         {{\"buffer\":0,\"byteOffset\":{pos_off},\"byteLength\":{pos_len},\"target\":34962}}"
    );
    let mut accessors = format!(
        "{{\"bufferView\":0,\"componentType\":5125,\"count\":{},\"type\":\"SCALAR\"}},\
         {{\"bufferView\":1,\"componentType\":5126,\"count\":{},\"type\":\"VEC3\",\"min\":{},\"max\":{}}}",
        indices.len(),
        verts.len(),
        arr(mn),
        arr(mx),
    );
    let attributes = if normals.is_some() {
        views.push_str(&format!(
            ",{{\"buffer\":0,\"byteOffset\":{nrm_off},\"byteLength\":{nrm_len},\"target\":34962}}"
        ));
        accessors.push_str(&format!(
            ",{{\"bufferView\":2,\"componentType\":5126,\"count\":{},\"type\":\"VEC3\"}}",
            verts.len()
        ));
        "\"POSITION\":1,\"NORMAL\":2"
    } else {
        "\"POSITION\":1"
    };

    let json = format!(
        "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"threers\"}},\
         \"scene\":0,\"scenes\":[{{\"nodes\":[0]}}],\"nodes\":[{{\"mesh\":0}}],\
         \"meshes\":[{{\"primitives\":[{{\"attributes\":{{{attributes}}},\"indices\":0,\"mode\":4}}]}}],\
         \"buffers\":[{{\"byteLength\":{buf_len}}}],\
         \"bufferViews\":[{views}],\"accessors\":[{accessors}]}}"
    );

    // GLB container: 12-byte header + JSON chunk (space-padded) + BIN chunk (0-padded).
    let mut json_bytes = json.into_bytes();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(total);
    glb.extend_from_slice(b"glTF"); // magic
    glb.extend_from_slice(&2u32.to_le_bytes()); // version
    glb.extend_from_slice(&(total as u32).to_le_bytes());
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json_bytes);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);
    glb
}

/// Export colored OpenSCAD [`crate::openscad::ScadPart`]s as a multi-mesh **glTF 2.0** `.glb`.
pub fn parts_to_glb(parts: &[crate::openscad::ScadPart]) -> Vec<u8> {
    use crate::openscad::ScadPart;

    let mut bin: Vec<u8> = Vec::new();
    let mut buffer_views: Vec<String> = Vec::new();
    let mut accessors: Vec<String> = Vec::new();
    let mut meshes: Vec<String> = Vec::new();
    let mut nodes: Vec<String> = Vec::new();
    let mut materials: Vec<String> = Vec::new();
    let mut mat_map: Vec<(u32, u32, u32, u32)> = Vec::new();

    let default_rgba = [0.91_f32, 0.93, 0.95, 1.0];
    let pbr = |rgba: [f32; 4]| -> (f32, f32) {
        let (r, g, b) = (rgba[0], rgba[1], rgba[2]);
        // Warm glass tag
        if r > 0.85 && g > 0.75 && b < 0.55 {
            return (0.55, 0.06);
        }
        // Solar cells — blue-black
        if r < 0.08 && g < 0.08 && b < 0.12 {
            return (0.08, 0.32);
        }
        // Dark MLI / logos
        if r < 0.15 && g < 0.16 && b < 0.20 {
            return (0.12, 0.74);
        }
        // Docking marker red
        if r > 0.65 && g < 0.20 && b < 0.25 {
            return (0.55, 0.35);
        }
        // Docking marker blue
        if r < 0.15 && g > 0.22 && b > 0.55 {
            return (0.55, 0.35);
        }
        // Gold EVA
        if r > 0.65 && g > 0.55 && b < 0.45 {
            return (0.90, 0.24);
        }
        // White quilted MLI (warm off-white)
        if r > 0.85 && g > 0.82 && b < 0.92 && r - b > 0.02 {
            return (0.04, 0.86);
        }
        // Near-neutral white hull
        if r > 0.92 && g > 0.92 && b > 0.92 {
            return (0.32, 0.48);
        }
        // Silver trim (blue-shifted)
        if b > r + 0.03 && b > g && r > 0.55 {
            return (0.82, 0.18);
        }
        // Hull fallback
        (0.32, 0.48)
    };

    let mut mat_idx = |rgba: [f32; 4]| -> usize {
        let key = (
            (rgba[0] * 255.0).round() as u32,
            (rgba[1] * 255.0).round() as u32,
            (rgba[2] * 255.0).round() as u32,
            (rgba[3] * 255.0).round() as u32,
        );
        if let Some(i) = mat_map.iter().position(|k| *k == key) {
            return i;
        }
        let i = materials.len();
        mat_map.push(key);
        let (metal, rough) = pbr(rgba);
        materials.push(format!(
            "{{\"pbrMetallicRoughness\":{{\"baseColorFactor\":[{},{},{},{}],\"metallicFactor\":{},\"roughnessFactor\":{}}}}}",
            rgba[0], rgba[1], rgba[2], rgba[3], metal, rough
        ));
        i
    };

    let arr3 = |a: [f32; 3]| format!("[{},{},{}]", a[0], a[1], a[2]);

    for part in parts {
        let ScadPart { geometry, color } = part;
        let Some((verts, normals, indices)) = tri_data(geometry) else {
            continue;
        };
        if indices.is_empty() {
            continue;
        }

        let rgba = color.unwrap_or(default_rgba);
        let mi = mat_idx(rgba);

        let idx_off = bin.len();
        for &i in &indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let idx_len = bin.len() - idx_off;
        let idx_bv = buffer_views.len();
        buffer_views.push(format!(
            "{{\"buffer\":0,\"byteOffset\":{idx_off},\"byteLength\":{idx_len},\"target\":34963}}"
        ));
        let idx_acc = accessors.len();
        accessors.push(format!(
            "{{\"bufferView\":{idx_bv},\"componentType\":5125,\"count\":{},\"type\":\"SCALAR\"}}",
            indices.len()
        ));

        let pos_off = bin.len();
        for v in &verts {
            for c in v {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let pos_len = bin.len() - pos_off;
        let pos_bv = buffer_views.len();
        buffer_views.push(format!(
            "{{\"buffer\":0,\"byteOffset\":{pos_off},\"byteLength\":{pos_len},\"target\":34962}}"
        ));

        let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
        for v in &verts {
            for k in 0..3 {
                mn[k] = mn[k].min(v[k]);
                mx[k] = mx[k].max(v[k]);
            }
        }
        let pos_acc = accessors.len();
        accessors.push(format!(
            "{{\"bufferView\":{pos_bv},\"componentType\":5126,\"count\":{},\"type\":\"VEC3\",\"min\":{},\"max\":{}}}",
            verts.len(),
            arr3(mn),
            arr3(mx)
        ));

        let nrm_acc = if let Some(n) = &normals {
            let nrm_off = bin.len();
            for v in n {
                for c in v {
                    bin.extend_from_slice(&c.to_le_bytes());
                }
            }
            let nrm_len = bin.len() - nrm_off;
            let nrm_bv = buffer_views.len();
            buffer_views.push(format!(
                "{{\"buffer\":0,\"byteOffset\":{nrm_off},\"byteLength\":{nrm_len},\"target\":34962}}"
            ));
            let acc = accessors.len();
            accessors.push(format!(
                "{{\"bufferView\":{nrm_bv},\"componentType\":5126,\"count\":{},\"type\":\"VEC3\"}}",
                n.len()
            ));
            Some(acc)
        } else {
            None
        };

        let attrs = if let Some(na) = nrm_acc {
            format!("\"POSITION\":{pos_acc},\"NORMAL\":{na}")
        } else {
            format!("\"POSITION\":{pos_acc}")
        };
        let mesh_idx = meshes.len();
        meshes.push(format!(
            "{{\"primitives\":[{{\"attributes\":{{{attrs}}},\"indices\":{idx_acc},\"material\":{mi},\"mode\":4}}]}}"
        ));
        nodes.push(format!("{{\"mesh\":{mesh_idx}}}"));
    }

    if meshes.is_empty() {
        return Vec::new();
    }

    let node_list: String = (0..nodes.len()).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
    let json = format!(
        "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"threers\"}},\
         \"scene\":0,\"scenes\":[{{\"nodes\":[{node_list}]}}],\
         \"nodes\":[{}],\"meshes\":[{}],\"materials\":[{}],\
         \"buffers\":[{{\"byteLength\":{}}}],\"bufferViews\":[{}],\"accessors\":[{}]}}",
        nodes.join(","),
        meshes.join(","),
        materials.join(","),
        bin.len(),
        buffer_views.join(","),
        accessors.join(","),
    );

    let mut json_bytes = json.into_bytes();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(total);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total as u32).to_le_bytes());
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json_bytes);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);
    glb
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cube, geometry_to_obj as _reexport_check};

    fn tetra() -> BufferGeometry {
        // A unit-ish tetra via the exact kernel (has a normal attribute).
        cube([10.0, 10.0, 10.0]).to_geometry_exact()
    }

    #[test]
    fn obj_roundtrips_through_objloader() {
        let _ = _reexport_check; // ensure the crate re-export exists
        let g = tetra();
        let obj = geometry_to_obj(&g);
        let vlines = obj.lines().filter(|l| l.starts_with("v ")).count();
        let flines = obj.lines().filter(|l| l.starts_with("f ")).count();
        assert!(vlines > 0 && flines > 0, "obj empty");
        // Re-import via the public OBJ loader and compare triangle counts.
        let back = crate::ObjLoader::parse(&obj);
        let n_in = g.index.as_ref().map(|i| i.len() / 3).unwrap_or(vlines / 3);
        let n_out = back.index.as_ref().map(|i| i.len() / 3).unwrap_or_else(|| {
            back.get_attribute("position")
                .map(|a| a.count() / 3)
                .unwrap_or(0)
        });
        assert_eq!(n_out, n_in, "obj triangle count changed on round-trip");
    }

    /// A geometry from corner loops, fanned from corner 0 as every loader
    /// here does, with the grouping set.
    fn from_faces(points: &[[f32; 3]], faces: &[&[u32]]) -> BufferGeometry {
        let mut index = Vec::new();
        let mut counts = Vec::new();
        for f in faces {
            for k in 1..f.len() - 1 {
                index.extend_from_slice(&[f[0], f[k], f[k + 1]]);
            }
            counts.push(f.len() as u32 - 2);
        }
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            crate::core::BufferAttribute::new(points.iter().flatten().copied().collect(), 3),
        );
        g.set_index(index);
        g.set_polygons(counts);
        g
    }

    fn unit_quad() -> BufferGeometry {
        from_faces(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            &[&[0, 1, 2, 3]],
        )
    }

    /// A triangle, a quad and a pentagon sharing edges.
    fn mixed() -> BufferGeometry {
        from_faces(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.5, 0.0],
                [0.5, -1.0, 0.0],
                [-0.5, -0.8, 0.0],
                [-0.7, 0.0, 0.0],
            ],
            &[&[1, 4, 2], &[0, 1, 2, 3], &[0, 7, 6, 5, 1]],
        )
    }

    /// Every polygon as the positions of its corners, in loop order.
    fn polygon_positions(g: &BufferGeometry) -> Vec<Vec<[u32; 3]>> {
        let p = &g.get_attribute("position").unwrap().array;
        g.faces()
            .map(|f| {
                f.iter()
                    .map(|&i| {
                        let at = i as usize * 3;
                        [p[at].to_bits(), p[at + 1].to_bits(), p[at + 2].to_bits()]
                    })
                    .collect()
            })
            .collect()
    }

    fn assert_same_polygons(back: &BufferGeometry, g: &BufferGeometry, what: &str) {
        assert_eq!(back.polygon_counts(), g.polygon_counts(), "{what}: grouping");
        assert_eq!(polygon_positions(back), polygon_positions(g), "{what}: corners");
    }

    fn face_corner_counts(lines: impl Iterator<Item = usize>) -> Vec<usize> {
        lines.collect()
    }

    #[test]
    fn obj_writes_a_quad_as_one_face_and_reads_back() {
        let g = unit_quad();
        let obj = geometry_to_obj(&g);
        let faces: Vec<&str> = obj.lines().filter(|l| l.starts_with("f ")).collect();
        assert_eq!(faces, vec!["f 1 2 3 4"]);
        assert_same_polygons(&crate::ObjLoader::parse(&obj), &g, "obj quad");
    }

    #[test]
    fn obj_ply_off_keep_triangles_quads_and_pentagons() {
        let g = mixed();
        let obj = geometry_to_obj(&g);
        let corners = face_corner_counts(
            obj.lines()
                .filter(|l| l.starts_with("f "))
                .map(|l| l.split_whitespace().count() - 1),
        );
        assert_eq!(corners, vec![3, 4, 5]);
        assert_same_polygons(&crate::ObjLoader::parse(&obj), &g, "obj mixed");

        let ply = geometry_to_ply(&g);
        assert_same_polygons(&crate::PlyLoader::parse(ply.as_bytes()), &g, "ply mixed");

        let off = geometry_to_off(&g);
        let mut lines = off.lines().skip(1);
        let header: Vec<usize> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(header, vec![8, 3, 0]);
        let corners = face_corner_counts(
            off.lines()
                .skip(2 + 8)
                .map(|l| l.split_whitespace().next().unwrap().parse().unwrap()),
        );
        assert_eq!(corners, vec![3, 4, 5]);
    }

    /// A box: 24 split vertices with normals and UVs, six quads. Both text
    /// formats bring back the same six quads, corner for corner, with the
    /// per-vertex normals and UVs intact.
    #[test]
    fn a_box_round_trips_as_six_quads() {
        let g = crate::BoxGeometry::new(1.0, 2.0, 3.0);
        assert_eq!(g.polygon_counts(), Some(&[2u32; 6][..]));
        let obj = geometry_to_obj(&g);
        assert_eq!(obj.lines().filter(|l| l.starts_with("f ")).count(), 6);
        assert!(obj.lines().any(|l| l.starts_with("vt ")), "uvs written");
        let ply = geometry_to_ply(&g);
        for (back, what) in [
            (crate::ObjLoader::parse(&obj), "obj box"),
            (crate::PlyLoader::parse(ply.as_bytes()), "ply box"),
        ] {
            assert_same_polygons(&back, &g, what);
            // Same vertex attributes per polygon corner.
            for name in ["normal", "uv"] {
                let per_corner = |geo: &BufferGeometry| -> Vec<Vec<u32>> {
                    let a = geo.get_attribute(name).unwrap();
                    geo.faces()
                        .map(|f| {
                            f.iter()
                                .flat_map(|&i| {
                                    let at = i as usize * a.item_size;
                                    a.array[at..at + a.item_size].iter().map(|x| x.to_bits())
                                })
                                .collect()
                        })
                        .collect()
                };
                assert_eq!(per_corner(&back), per_corner(&g), "{what}: {name}");
            }
        }
    }

    /// No grouping, or a polygon whose outline cannot be recovered, writes
    /// triangles — nothing is dropped.
    #[test]
    fn ungrouped_and_unrecoverable_polygons_fall_back_to_triangles() {
        let mut g = unit_quad();
        g.polygons = None;
        let obj = geometry_to_obj(&g);
        assert_eq!(obj.lines().filter(|l| l.starts_with("f ")).count(), 2);
        let back = crate::ObjLoader::parse(&obj);
        assert_eq!(back.polygon_counts(), Some(&[1u32, 1][..]));

        // Two triangles that only touch at a corner: grouped, but not a loop.
        let mut bow = from_faces(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            &[&[0, 1, 2], &[0, 3, 4]],
        );
        bow.set_polygons(vec![2]);
        let obj = geometry_to_obj(&bow);
        assert_eq!(obj.lines().filter(|l| l.starts_with("f ")).count(), 2);
    }

    /// The written OBJ and PLY, read by an independent tool: `assimp info`
    /// counts the faces it read, and a quad mesh must read as quads rather
    /// than as twice as many triangles. Skipped (with a note) when `assimp`
    /// is not installed.
    #[test]
    fn assimp_reads_the_quads() {
        let Ok(probe) = std::process::Command::new("assimp").arg("version").output() else {
            eprintln!("assimp not installed; skipping the reference check");
            return;
        };
        if !probe.status.success() {
            eprintln!("assimp not runnable; skipping the reference check");
            return;
        }
        let g = mixed();
        let dir = std::env::temp_dir().join(format!("threers-quad-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (ext, text) in [("obj", geometry_to_obj(&g)), ("ply", geometry_to_ply(&g))] {
            let path = dir.join(format!("mixed.{ext}"));
            std::fs::write(&path, text).unwrap();
            // `--raw`: no post-processing, so assimp's own triangulation step
            // does not run and polygons stay polygons.
            let out = std::process::Command::new("assimp")
                .arg("info")
                .arg(&path)
                .arg("--raw")
                .output()
                .unwrap();
            let report = String::from_utf8_lossy(&out.stdout);
            let faces = report
                .lines()
                .find(|l| l.trim_start().starts_with("Faces:"))
                .and_then(|l| l.split(':').nth(1))
                .map(|n| n.trim().parse::<usize>().unwrap())
                .unwrap_or_else(|| panic!("no face count in assimp's report:\n{report}"));
            assert_eq!(faces, 3, "{ext}: assimp should read 3 faces\n{report}");
            let vertices = report
                .lines()
                .find(|l| l.trim_start().starts_with("Vertices:"))
                .and_then(|l| l.split(':').nth(1))
                .map(|n| n.trim().parse::<usize>().unwrap())
                .unwrap();
            // OBJ is imported one vertex per face corner (3 + 4 + 5); PLY
            // keeps the shared vertex list.
            let expected = if ext == "obj" { 12 } else { 8 };
            assert_eq!(vertices, expected, "{ext}: vertices\n{report}");
            assert!(report.contains("n-polygons"), "{ext}: polygons\n{report}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn off_has_matching_counts() {
        let g = tetra();
        let off = geometry_to_off(&g);
        let mut lines = off.lines();
        assert_eq!(lines.next(), Some("OFF"));
        let counts: Vec<usize> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let (nv, nf) = (counts[0], counts[1]);
        // Faces are written as the polygons the geometry remembers (the
        // exact cube's six quads), so count every face line, not `3 ` ones.
        let face_lines: Vec<&str> = off.lines().skip(2 + nv).collect();
        assert_eq!(face_lines.len(), nf, "face count");
        for f in &face_lines {
            let n: usize = f.split_whitespace().next().unwrap().parse().unwrap();
            assert_eq!(f.split_whitespace().count(), n + 1, "{f}");
        }
        // vertex lines = total - header(2) - faces
        assert_eq!(off.lines().count() - 2 - nf, nv, "vertex count");
    }

    #[test]
    fn threemf_is_valid_zip_with_the_mesh() {
        let g = tetra();
        let bytes = geometry_to_3mf(&g);
        // Central-directory + local headers present.
        assert!(bytes.starts_with(b"PK\x03\x04"), "not a zip");
        assert!(bytes.windows(4).any(|w| w == b"PK\x05\x06"), "no EOCD");
        // Stored (uncompressed) entries → the model XML is verbatim in the bytes.
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("3dmodel.model"), "missing model part name");
        let n_v = g.get_attribute("position").unwrap().count(); // count() = vertices
        assert_eq!(text.matches("<vertex ").count(), n_v, "3mf vertex count");
    }

    #[test]
    fn glb_header_and_json_are_wellformed() {
        let g = tetra();
        let glb = geometry_to_glb(&g);
        assert!(glb.starts_with(b"glTF"), "bad magic");
        let total = u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize;
        assert_eq!(total, glb.len(), "declared length != actual");
        // JSON chunk.
        let jlen = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        assert_eq!(&glb[16..20], b"JSON");
        let json = std::str::from_utf8(&glb[20..20 + jlen]).unwrap();
        assert!(json.contains("\"version\":\"2.0\""));
        assert!(json.contains("\"POSITION\":1"));
        let n_v = g.get_attribute("position").unwrap().count(); // count() = vertices
        assert!(
            json.contains(&format!("\"count\":{n_v}")),
            "position accessor count"
        );
        // Alignment: both chunks 4-byte aligned.
        assert_eq!(jlen % 4, 0);
        // BIN chunk present after the JSON chunk.
        let bin_hdr = 20 + jlen;
        assert_eq!(&glb[bin_hdr + 4..bin_hdr + 8], b"BIN\0");
    }
}
