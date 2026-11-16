//! Rhino `.3dm` (openNURBS archive version 7): each mesh object as an
//! `ON_Mesh` with its name and colour, in the model's units.
//!
//! An openNURBS archive is a 32-byte header and a tree of chunks: a type
//! code, a 64-bit length and the payload, or for a "short" chunk the value in
//! place of the length. A chunk whose type code has the CRC bit carries a
//! CRC-32 at its end, of the bytes written directly into it — not those of the
//! chunks nested in it.
//!
//! Everything but the objects — the properties, the settings, the empty
//! tables and the default layer — is the same in every file, so it comes from
//! an empty model openNURBS wrote (`rhino_template.3dm`), with its unit
//! changed and the object table filled. Mesh arrays are written stored rather
//! than zlib-compressed, which openNURBS reads as readily.

use super::{apply_matrix4, attr3, count, normal_matrix, transform_normal, ExportKind, ExportObject};
use crate::utils::png::crc32;

/// The model's length unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThreeDmUnits {
    #[default]
    Millimeters,
    Centimeters,
    Meters,
    Inches,
}

impl ThreeDmUnits {
    /// openNURBS's `ON::LengthUnitSystem` code and metres per unit.
    fn code(self) -> (i32, f64) {
        match self {
            ThreeDmUnits::Millimeters => (2, 0.001),
            ThreeDmUnits::Centimeters => (3, 0.01),
            ThreeDmUnits::Meters => (4, 1.0),
            ThreeDmUnits::Inches => (8, 0.0254),
        }
    }
}

/// What [`write_3dm`] writes beyond the geometry.
#[derive(Debug, Clone, Default)]
pub struct ThreeDmOptions {
    pub units: ThreeDmUnits,
    /// A display colour per object (sRGB, 0 to 255), by index; objects past
    /// the end take their layer's.
    pub colors: Vec<[u8; 3]>,
}

const TEMPLATE: &[u8] = include_bytes!("rhino_template.3dm");

const TCODE_SETTINGS_TABLE: u32 = 0x1000_0015;
const TCODE_OBJECT_TABLE: u32 = 0x1000_0013;
const TCODE_ENDOFFILE: u32 = 0x0000_7fff;
const TCODE_ENDOFTABLE: u32 = 0xffff_ffff;
const TCODE_SHORT: u32 = 0x8000_0000;
const TCODE_CRC: u32 = 0x8000;
const TCODE_OBJECT_RECORD: u32 = 0x2000_8070;
const TCODE_OBJECT_RECORD_TYPE: u32 = 0x8200_0071;
const TCODE_OBJECT_RECORD_ATTRIBUTES: u32 = 0x0200_8072;
const TCODE_OBJECT_RECORD_END: u32 = 0x8200_007f;
const TCODE_OPENNURBS_CLASS: u32 = 0x0002_7ffa;
const TCODE_OPENNURBS_CLASS_UUID: u32 = 0x0002_fffb;
const TCODE_OPENNURBS_CLASS_DATA: u32 = 0x0002_fffc;
const TCODE_OPENNURBS_CLASS_END: u32 = 0x8002_7fff;
const TCODE_ANONYMOUS_CHUNK: u32 = 0x4000_8000;
/// `ON::mesh_object`.
const MESH_OBJECT: i64 = 32;
/// `ON_Mesh`'s class id, as openNURBS stores it.
const ON_MESH_CLASS: [u8; 16] = [
    0xe4, 0xd4, 0xd7, 0x4e, 0x47, 0xe9, 0xd3, 0x11, 0xbf, 0xe5, 0x00, 0x10, 0x83, 0x01, 0x22, 0xf0,
];
/// `ON_UNSET_VALUE`.
const UNSET: f64 = -1.234_321_012_343_21e308;
/// The mesh's texture-mapping tag, as an untextured mesh has it: no mapping
/// and the identity transform. A whole chunk, header and CRC included, as
/// openNURBS writes it.
const NO_MAPPING_TAG: [u8; 176] = [
    0x00, 0x80, 0x00, 0x40, 0xa4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x69, 0x3c, 0x98, 0xa3,
];

/// A chunk's parts: bytes written into it directly, and whole chunks nested
/// in it.
enum Part {
    Direct(Vec<u8>),
    Nested(Vec<u8>),
}

fn chunk(typecode: u32, parts: Vec<Part>) -> Vec<u8> {
    let mut payload = vec![];
    let mut direct = vec![];
    for p in parts {
        match p {
            Part::Direct(b) => {
                direct.extend_from_slice(&b);
                payload.extend(b);
            }
            Part::Nested(b) => payload.extend(b),
        }
    }
    if typecode & TCODE_CRC != 0 {
        payload.extend(crc32(&direct).to_le_bytes());
    }
    let mut out = Vec::with_capacity(12 + payload.len());
    out.extend(typecode.to_le_bytes());
    out.extend((payload.len() as i64).to_le_bytes());
    out.extend(payload);
    out
}

fn short(typecode: u32, value: i64) -> Vec<u8> {
    let mut out = typecode.to_le_bytes().to_vec();
    out.extend(value.to_le_bytes());
    out
}

/// openNURBS's compressed buffer, stored: size, CRC, method 0, the bytes.
/// An empty one is its size alone.
fn buffer(data: &[u8]) -> Vec<u8> {
    let mut out = (data.len() as u32).to_le_bytes().to_vec();
    if !data.is_empty() {
        out.extend(crc32(data).to_le_bytes());
        out.push(0);
        out.extend_from_slice(data);
    }
    out
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The objects' meshes as a Rhino model.
pub fn write_3dm(objects: &[ExportObject], options: &ThreeDmOptions) -> Vec<u8> {
    let records: Vec<u8> = objects
        .iter()
        .enumerate()
        .filter(|(_, o)| o.kind == ExportKind::Mesh && count(&o.geometry, "position").is_some_and(|n| n > 0))
        .flat_map(|(k, o)| record(o, k, options.colors.get(k)))
        .collect();

    let mut out = TEMPLATE[..32].to_vec();
    let mut at = 32;
    while at + 12 <= TEMPLATE.len() {
        let typecode = u32::from_le_bytes(TEMPLATE[at..at + 4].try_into().unwrap());
        let value = i64::from_le_bytes(TEMPLATE[at + 4..at + 12].try_into().unwrap());
        let end = if typecode & TCODE_SHORT != 0 { at + 12 } else { at + 12 + value as usize };
        match typecode {
            TCODE_SETTINGS_TABLE => {
                let mut payload = TEMPLATE[at + 12..end].to_vec();
                set_units(&mut payload, options.units);
                out.extend(typecode.to_le_bytes());
                out.extend((payload.len() as i64).to_le_bytes());
                out.extend(payload);
            }
            TCODE_OBJECT_TABLE => {
                let mut payload = records.clone();
                payload.extend(short(TCODE_ENDOFTABLE, 0));
                out.extend(typecode.to_le_bytes());
                out.extend((payload.len() as i64).to_le_bytes());
                out.extend(payload);
            }
            TCODE_ENDOFFILE => {
                // The file's whole length, this chunk's 20 bytes included.
                let length = (out.len() + 20) as u64;
                out.extend(typecode.to_le_bytes());
                out.extend(8i64.to_le_bytes());
                out.extend(length.to_le_bytes());
            }
            _ => out.extend_from_slice(&TEMPLATE[at..end]),
        }
        at = end;
    }
    out
}

/// The units record is the settings table's first: its unit code 4 bytes
/// into the payload, metres per unit at 40, and its CRC after 52.
fn set_units(settings: &mut [u8], units: ThreeDmUnits) {
    let (code, meters) = units.code();
    let p = 12;
    settings[p + 4..p + 8].copy_from_slice(&code.to_le_bytes());
    settings[p + 40..p + 48].copy_from_slice(&meters.to_le_bytes());
    let crc = crc32(&settings[p..p + 52]);
    settings[p + 52..p + 56].copy_from_slice(&crc.to_le_bytes());
}

/// An object id: unique per object and per call, shaped as a version-4 UUID.
fn object_id(k: usize, name: &str) -> [u8; 16] {
    use std::hash::{BuildHasher, Hasher};
    let mut id = [0u8; 16];
    for (half, salt) in [(0usize, 0x5bd1_e995u64), (8, 0x2545_f491)] {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_usize(k);
        h.write(name.as_bytes());
        h.write_u64(salt);
        id[half..half + 8].copy_from_slice(&h.finish().to_le_bytes());
    }
    id[7] = (id[7] & 0x0f) | 0x40;
    id[8] = (id[8] & 0x3f) | 0x80;
    id
}

fn record(o: &ExportObject, k: usize, color: Option<&[u8; 3]>) -> Vec<u8> {
    let class = chunk(
        TCODE_OPENNURBS_CLASS,
        vec![
            Part::Nested(chunk(TCODE_OPENNURBS_CLASS_UUID, vec![Part::Direct(ON_MESH_CLASS.to_vec())])),
            Part::Nested(mesh(o)),
            Part::Nested(short(TCODE_OPENNURBS_CLASS_END, 0)),
        ],
    );
    // ON_3dmObjectAttributes: a version, the id, the layer, then items.
    let mut attributes = vec![0x2d];
    attributes.extend(object_id(k, &o.name));
    attributes.extend(0i32.to_le_bytes());
    if !o.name.is_empty() {
        let units: Vec<u16> = o.name.encode_utf16().collect();
        attributes.push(1);
        attributes.extend((units.len() as i32 + 1).to_le_bytes());
        for u in units {
            attributes.extend(u.to_le_bytes());
        }
        attributes.extend([0, 0]);
    }
    if let Some(c) = color {
        attributes.extend([6, c[0], c[1], c[2], 0]);
        // The colour comes from the object, not its layer.
        attributes.extend([0x0d, 1]);
    }
    attributes.push(0);
    chunk(
        TCODE_OBJECT_RECORD,
        vec![
            Part::Nested(short(TCODE_OBJECT_RECORD_TYPE, MESH_OBJECT)),
            Part::Nested(class),
            Part::Nested(chunk(TCODE_OBJECT_RECORD_ATTRIBUTES, vec![Part::Direct(attributes)])),
            Part::Nested(short(TCODE_OBJECT_RECORD_END, 0)),
        ],
    )
}

/// `ON_Mesh::Write`, version 3.8: counts, unset domains and boxes (openNURBS
/// recomputes those), the faces, then the vertices and normals as floats,
/// the mapping tag and the vertices again as doubles.
fn mesh(o: &ExportObject) -> Vec<u8> {
    let g = &*o.geometry;
    let n = count(g, "position").unwrap_or(0);
    let world: Vec<[f64; 3]> = (0..n).map(|i| apply_matrix4(&o.matrix_world, attr3(g, "position", i))).collect();
    let index: Vec<u32> = g.index.clone().unwrap_or_else(|| (0..n as u32).collect());
    let faces: Vec<[u32; 3]> = index.as_chunks::<3>().0.to_vec();

    let mut d = vec![0x38];
    d.extend((n as i32).to_le_bytes());
    d.extend((faces.len() as i32).to_le_bytes());
    for _ in 0..8 {
        d.extend(UNSET.to_le_bytes());
    }
    d.extend([0u8; 16]);
    let empty_box = floats(&[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);
    d.extend(&empty_box);
    d.extend(&empty_box);
    d.extend(floats(&[1.0, 1.0, -1.0, -1.0]));
    d.extend((-1i32).to_le_bytes());
    d.extend([0u8; 5]);
    // Faces: four corners each, a triangle repeating its third.
    let size: usize = if n < 256 { 1 } else if n < 65536 { 2 } else { 4 };
    d.extend((size as i32).to_le_bytes());
    for f in &faces {
        for i in [f[0], f[1], f[2], f[2]] {
            d.extend(&i.to_le_bytes()[..size]);
        }
    }
    let v32: Vec<f32> = world.iter().flat_map(|p| p.map(|c| c as f32)).collect();
    d.extend(buffer(&floats(&v32)));
    let normals: Vec<f32> = if count(g, "normal") == Some(n) {
        let nm = normal_matrix(&o.matrix_world);
        (0..n).flat_map(|i| transform_normal(&nm, attr3(g, "normal", i)).map(|c| c as f32)).collect()
    } else {
        vec![]
    };
    d.extend(buffer(&floats(&normals)));
    // Texture coordinates, curvatures, colours and the rest: none.
    for _ in 0..9 {
        d.extend(0u32.to_le_bytes());
    }

    let mut doubles = vec![1, 0, 0, 0, 0, 0, 0, 0];
    doubles.extend((n as i32).to_le_bytes());
    let dv: Vec<u8> = world.iter().flatten().flat_map(|c| c.to_le_bytes()).collect();
    doubles.extend(buffer(&dv));
    let mut tail = vec![];
    for _ in 0..6 {
        tail.extend(UNSET.to_le_bytes());
    }
    chunk(
        TCODE_OPENNURBS_CLASS_DATA,
        vec![
            Part::Direct(d),
            Part::Nested(NO_MAPPING_TAG.to_vec()),
            Part::Direct(vec![0, 0, 0, 0, 1]),
            Part::Nested(chunk(TCODE_ANONYMOUS_CHUNK, vec![Part::Direct(doubles)])),
            Part::Direct(tail),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::BoxGeometry;

    /// Every table and object record read back by its length, and the CRC
    /// of every chunk this writer makes checked by the rule openNURBS writes
    /// it by. Counts the object records.
    fn walk(b: &[u8], lo: usize, hi: usize, ours: bool, records: &mut usize) {
        let mut at = lo;
        while at + 12 <= hi {
            let tc = u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
            let v = i64::from_le_bytes(b[at + 4..at + 12].try_into().unwrap());
            if tc & TCODE_SHORT != 0 || tc == TCODE_ENDOFTABLE {
                at += 12;
                continue;
            }
            let end = at + 12 + v as usize;
            assert!(end <= hi, "chunk {tc:08x} at {at} runs past its parent");
            let crc = if tc & TCODE_CRC != 0 { 4 } else { 0 };
            if tc == TCODE_OBJECT_RECORD {
                *records += 1;
                walk(b, at + 12, end - crc, true, records);
            } else if tc & 0xf000_0000 == 0x1000_0000 || tc == TCODE_OPENNURBS_CLASS {
                walk(b, at + 12, end - crc, ours, records);
            } else if crc == 4 && (ours && tc != TCODE_OPENNURBS_CLASS_DATA || tc == 0x2000_8031) {
                // Leaves: the CRC is of the whole payload. (The mesh's own
                // excludes its nested chunks; openNURBS checks that one.)
                let stored = u32::from_le_bytes(b[end - 4..end].try_into().unwrap());
                assert_eq!(crc32(&b[at + 12..end - 4]), stored, "CRC of {tc:08x} at {at}");
            }
            at = end;
        }
    }

    #[test]
    fn a_box_is_one_object_and_the_file_holds_together() {
        let mut o = ExportObject::mesh("box", BoxGeometry::new(2.0, 2.0, 2.0));
        o.matrix_world[12] = 5.0;
        let bytes = write_3dm(&[o], &ThreeDmOptions { units: ThreeDmUnits::Meters, colors: vec![[200, 100, 0]] });
        assert!(bytes.starts_with(b"3D Geometry File Format "));
        let mut records = 0;
        walk(&bytes, 32, bytes.len(), false, &mut records);
        assert_eq!(records, 1);
        let length = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap());
        assert_eq!(length, bytes.len() as u64, "the end says how long the file is");
    }
}
