//! STL — three.js r165 `STLExporter.parse`, ASCII and binary.

use super::{apply_matrix4, attr3, count, normalize, ExportKind, ExportObject, NumberFormat};

/// Every mesh triangle in world space, in order, as three.js's `writeFace`
/// sees them. Lines and points are not part of an STL.
fn triangles(objects: &[ExportObject]) -> impl Iterator<Item = [[f64; 3]; 3]> + '_ {
    objects
        .iter()
        .filter(|o| o.kind == ExportKind::Mesh)
        .flat_map(|o| {
            let g = &*o.geometry;
            let corners: Vec<[usize; 3]> = match (&g.index, count(g, "position")) {
                (_, None) => Vec::new(),
                (Some(index), Some(_)) => index
                    .chunks_exact(3)
                    .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
                    .collect(),
                (None, Some(n)) => (0..n / 3).map(|t| [t * 3, t * 3 + 1, t * 3 + 2]).collect(),
            };
            corners.into_iter().map(move |t| {
                t.map(|i| apply_matrix4(&o.matrix_world, attr3(g, "position", i)))
            })
        })
}

/// three.js's `writeNormal`: `(c - b) × (a - b)`, normalised — twice, as it
/// does (`cb.cross(ab).normalize()` then `normal.copy(cb).normalize()`).
fn face_normal([a, b, c]: &[[f64; 3]; 3]) -> [f64; 3] {
    let cb = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let n = [
        cb[1] * ab[2] - cb[2] * ab[1],
        cb[2] * ab[0] - cb[0] * ab[2],
        cb[0] * ab[1] - cb[1] * ab[0],
    ];
    normalize(normalize(n))
}

/// ASCII STL: `solid exported`, a tab-indented `facet` per triangle,
/// `endsolid exported`.
pub fn write_stl_ascii(objects: &[ExportObject], numbers: NumberFormat) -> String {
    let mut out = String::from("solid exported\n");
    let xyz = |out: &mut String, v: [f64; 3]| {
        for (k, c) in v.iter().enumerate() {
            if k > 0 {
                out.push(' ');
            }
            numbers.write(out, *c);
        }
        out.push('\n');
    };
    for tri in triangles(objects) {
        out.push_str("\tfacet normal ");
        xyz(&mut out, face_normal(&tri));
        out.push_str("\t\touter loop\n");
        for v in tri {
            out.push_str("\t\t\tvertex ");
            xyz(&mut out, v);
        }
        out.push_str("\t\tendloop\n\tendfacet\n");
    }
    out.push_str("endsolid exported\n");
    out
}

/// Binary STL: an 80-byte zero header, the triangle count, then per triangle
/// the normal and three corners as little-endian `f32` and a zero attribute
/// word.
pub fn write_stl_binary(objects: &[ExportObject]) -> Vec<u8> {
    let tris: Vec<[[f64; 3]; 3]> = triangles(objects).collect();
    let mut out = Vec::with_capacity(84 + tris.len() * 50);
    out.extend_from_slice(&[0u8; 80]);
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for tri in &tris {
        for c in face_normal(tri) {
            out.extend_from_slice(&(c as f32).to_le_bytes());
        }
        for v in tri {
            for c in v {
                out.extend_from_slice(&(*c as f32).to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}
