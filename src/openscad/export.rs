//! Mesh **exporters** for the OpenSCAD front end — the counterparts to the
//! importers in `scad::import`.
//!
//! All operate on an evaluated [`BufferGeometry`] (indexed or triangle-soup) and
//! are pure Rust with no dependencies:
//! - `geometry_to_obj` — Wavefront **OBJ** (text; positions, normals, faces).
//! - `geometry_to_off` — Geomview **OFF** (text).
//! - `geometry_to_3mf` — **3MF** (a proper OPC/ZIP package a slicer can open).
//! - `geometry_to_glb` — binary **glTF 2.0** (`.glb`; single self-contained file).
//!
//! [`Solid`](crate::Solid) grows `to_obj`/`to_off`/`to_3mf`/`to_glb` convenience
//! methods that evaluate with the exact-where-confident kernel first.

use crate::core::BufferGeometry;

/// Normalise a geometry to `(vertices, optional per-vertex normals, triangle
/// indices)`. A non-indexed soup gets sequential indices; normals are dropped
/// unless there is exactly one per vertex.
fn tri_data(g: &BufferGeometry) -> Option<(Vec<[f32; 3]>, Option<Vec<[f32; 3]>>, Vec<u32>)> {
    let pos = &g.get_attribute("position")?.array;
    let verts: Vec<[f32; 3]> = pos.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    if verts.len() < 3 {
        return None;
    }
    let normals = g
        .get_attribute("normal")
        .map(|a| a.array.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect::<Vec<_>>())
        .filter(|n: &Vec<[f32; 3]>| n.len() == verts.len());
    let indices = g.index.clone().unwrap_or_else(|| (0..verts.len() as u32).collect());
    Some((verts, normals, indices))
}

/// Export as **Wavefront OBJ** (text). Emits `v`, optional `vn`, and 1-indexed
/// triangle `f` lines (with `v//vn` references when normals are present).
pub fn geometry_to_obj(g: &BufferGeometry) -> String {
    let Some((verts, normals, indices)) = tri_data(g) else {
        return String::new();
    };
    let mut s = String::from("# threers OBJ export\no model\n");
    for v in &verts {
        s.push_str(&format!("v {} {} {}\n", v[0], v[1], v[2]));
    }
    if let Some(n) = &normals {
        for v in n {
            s.push_str(&format!("vn {} {} {}\n", v[0], v[1], v[2]));
        }
    }
    for t in indices.chunks_exact(3) {
        let (a, b, c) = (t[0] + 1, t[1] + 1, t[2] + 1); // OBJ is 1-indexed
        if normals.is_some() {
            s.push_str(&format!("f {a}//{a} {b}//{b} {c}//{c}\n"));
        } else {
            s.push_str(&format!("f {a} {b} {c}\n"));
        }
    }
    s
}

/// Export as **Geomview OFF** (text).
pub fn geometry_to_off(g: &BufferGeometry) -> String {
    let Some((verts, _, indices)) = tri_data(g) else {
        return String::from("OFF\n0 0 0\n");
    };
    let nf = indices.len() / 3;
    let mut s = format!("OFF\n{} {} 0\n", verts.len(), nf);
    for v in &verts {
        s.push_str(&format!("{} {} {}\n", v[0], v[1], v[2]));
    }
    for t in indices.chunks_exact(3) {
        s.push_str(&format!("3 {} {} {}\n", t[0], t[1], t[2]));
    }
    s
}

// --- 3MF (OPC/ZIP package) --------------------------------------------------

/// Export as **3MF** — a spec-shaped OPC package (`[Content_Types].xml`,
/// `_rels/.rels`, `3D/3dmodel.model`) zipped with stored (uncompressed) entries
/// and correct CRC-32s, so real slicers open it.
pub fn geometry_to_3mf(g: &BufferGeometry) -> Vec<u8> {
    let (verts, _, indices) = match tri_data(g) {
        Some(t) => t,
        None => (Vec::new(), None, Vec::new()),
    };
    let mut model = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <model unit=\"millimeter\" xml:lang=\"en-US\" \
         xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n\
         <resources><object id=\"1\" type=\"model\"><mesh><vertices>",
    );
    for v in &verts {
        model.push_str(&format!("<vertex x=\"{}\" y=\"{}\" z=\"{}\"/>", v[0], v[1], v[2]));
    }
    model.push_str("</vertices><triangles>");
    for t in indices.chunks_exact(3) {
        model.push_str(&format!("<triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>", t[0], t[1], t[2]));
    }
    model.push_str("</triangles></mesh></object></resources><build><item objectid=\"1\"/></build></model>");

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
/// CRC-32s and a central directory. Enough for a valid 3MF package.
struct Zip {
    out: Vec<u8>,
    dir: Vec<u8>,
    count: u16,
}
impl Zip {
    fn new() -> Self {
        Self { out: Vec::new(), dir: Vec::new(), count: 0 }
    }
    fn add(&mut self, name: &str, data: &[u8]) {
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
    fn finish(mut self) -> Vec<u8> {
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

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
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
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    while bin.len() % 4 != 0 {
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
            back.get_attribute("position").map(|a| a.count() / 3).unwrap_or(0)
        });
        assert_eq!(n_out, n_in, "obj triangle count changed on round-trip");
    }

    #[test]
    fn off_has_matching_counts() {
        let g = tetra();
        let off = geometry_to_off(&g);
        let mut lines = off.lines();
        assert_eq!(lines.next(), Some("OFF"));
        let counts: Vec<usize> = lines.next().unwrap().split_whitespace().map(|s| s.parse().unwrap()).collect();
        let (nv, nf) = (counts[0], counts[1]);
        assert_eq!(off.lines().filter(|l| l.starts_with("3 ")).count(), nf, "face count");
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
        assert!(json.contains(&format!("\"count\":{n_v}")), "position accessor count");
        // Alignment: both chunks 4-byte aligned.
        assert_eq!(jlen % 4, 0);
        // BIN chunk present after the JSON chunk.
        let bin_hdr = 20 + jlen;
        assert_eq!(&glb[bin_hdr + 4..bin_hdr + 8], b"BIN\0");
    }
}
