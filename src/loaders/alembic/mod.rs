//! Alembic (`.abc`) — baked geometry caches, as VFX pipelines write them.
//!
//! The natural neighbour to the USD work: same audience, same kind of file, and
//! the same sampling model — an Alembic property holds a *sequence* of samples
//! against a time sampling, which is USD's `timeSamples` by another name.
//!
//! # How a file is laid out
//!
//! [`ogawa`] is the container: groups of 64-bit references, and data blocks.
//! Everything above it is convention about which child means what:
//!
//! | group | children |
//! |---|---|
//! | archive root | version, file version, the top object, metadata, time samplings, indexed metadata |
//! | object | its properties group, then child objects, then a data block naming them |
//! | property | its samples or sub-properties, then a data block describing them |
//!
//! Two details are easy to miss and wreck everything after them: an object
//! header block ends with **32 bytes of hashes** that are not part of the
//! headers, and every array sample begins with a **16-byte key** that is not
//! part of the data.
//!
//! # What is read
//!
//! The object hierarchy, and `PolyMesh` geometry from it — points, face counts
//! and face indices, which is what a baked cache is. `Xform` transforms,
//! velocities, UVs and normals are named in the property tree and not yet
//! turned into a scene; subdivision meshes, curves and points are not read.

pub mod ogawa;

use ogawa::{Child, Ogawa};

#[derive(Debug, PartialEq)]
pub enum AlembicError {
    /// Not an Ogawa-backed Alembic file. The older HDF5 backing is a different
    /// container entirely and is not read here.
    NotAlembic,
    /// The writer did not finish: the completion byte is clear, so the offsets
    /// in the file cannot be trusted.
    Incomplete,
    Truncated,
    Corrupt,
}

impl std::fmt::Display for AlembicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlembicError::NotAlembic => write!(f, "not an Ogawa Alembic file"),
            AlembicError::Incomplete => write!(f, "Alembic file was not finished being written"),
            AlembicError::Truncated => write!(f, "truncated Alembic file"),
            AlembicError::Corrupt => write!(f, "corrupt Alembic file"),
        }
    }
}

impl std::error::Error for AlembicError {}

/// What kind of thing a property holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropertyKind {
    Compound,
    Scalar,
    Array,
}

/// The numeric type of a property's values, in Alembic's numbering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pod {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    F32,
    F64,
    String,
    WString,
}

impl Pod {
    fn from_code(code: u8) -> Option<Pod> {
        Some(match code {
            0 => Pod::Bool,
            1 => Pod::U8,
            2 => Pod::I8,
            3 => Pod::U16,
            4 => Pod::I16,
            5 => Pod::U32,
            6 => Pod::I32,
            7 => Pod::U64,
            8 => Pod::I64,
            9 => Pod::F16,
            10 => Pod::F32,
            11 => Pod::F64,
            12 => Pod::String,
            13 => Pod::WString,
            _ => return None,
        })
    }

    pub fn width(self) -> usize {
        match self {
            Pod::Bool | Pod::U8 | Pod::I8 => 1,
            Pod::U16 | Pod::I16 | Pod::F16 => 2,
            Pod::U32 | Pod::I32 | Pod::F32 => 4,
            Pod::U64 | Pod::I64 | Pod::F64 => 8,
            Pod::String | Pod::WString => 0,
        }
    }
}

/// One property, as its header describes it.
#[derive(Debug, Clone)]
pub struct PropertyHeader {
    pub name: String,
    pub kind: PropertyKind,
    pub pod: Option<Pod>,
    /// How many numbers make one value — 3 for a point, 1 for an index.
    pub extent: u8,
    /// How many samples the property has.
    pub sample_count: u32,
}

/// An object in the hierarchy.
#[derive(Debug, Clone)]
pub struct Object {
    pub name: String,
    pub full_name: String,
    pub children: Vec<Object>,
    pub properties: Vec<PropertyHeader>,
    /// Where the object's property group is, for reading samples later.
    property_group: Option<usize>,
}

/// A polygon mesh sample, which is what a baked cache holds.
#[derive(Debug, Clone, Default)]
pub struct PolyMesh {
    pub name: String,
    /// `P` — three floats a point.
    pub points: Vec<f32>,
    /// `.faceCounts` — how many corners each face has.
    pub face_counts: Vec<i32>,
    /// `.faceIndices` — the corners, face after face.
    pub face_indices: Vec<i32>,
}

/// An opened archive.
pub struct Archive<'a> {
    file: Ogawa<'a>,
    pub root: Object,
    /// The archive-level metadata string, which carries the frame rate and the
    /// writing application.
    pub metadata: String,
}

/// Read an integer of 1, 2 or 4 bytes, as the header's size hint says.
///
/// Alembic packs these to the smallest width a file needs, and the width is
/// declared once per property rather than per field — so a reader that assumes
/// four bytes walks off into the name.
fn hinted(buf: &[u8], hint: u32, at: &mut usize) -> Option<u32> {
    let value = match hint {
        0 => *buf.get(*at)? as u32,
        1 => u16::from_le_bytes(buf.get(*at..*at + 2)?.try_into().ok()?) as u32,
        2 => u32::from_le_bytes(buf.get(*at..*at + 4)?.try_into().ok()?),
        _ => return None,
    };
    *at += match hint {
        0 => 1,
        1 => 2,
        _ => 4,
    };
    Some(value)
}

/// Parse a property-header block.
///
/// The leading 32-bit word is a bitfield: property kind in bits 0-1, a size hint
/// in 2-3, the POD type in 4-7, flags in 8-11, the extent in 12-19 and a
/// metadata index in 20-27.
fn property_headers(buf: &[u8]) -> Vec<PropertyHeader> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 4 <= buf.len() {
        let info = u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
        at += 4;
        let kind = match info & 3 {
            0 => PropertyKind::Compound,
            1 => PropertyKind::Scalar,
            _ => PropertyKind::Array,
        };
        let hint = (info & 0x0c) >> 2;
        let mut pod = None;
        let mut extent = 0u8;
        let mut sample_count = 0u32;
        if kind != PropertyKind::Compound {
            pod = Pod::from_code(((info & 0xf0) >> 4) as u8);
            extent = ((info & 0xff000) >> 12) as u8;
            let Some(next) = hinted(buf, hint, &mut at) else {
                break;
            };
            sample_count = next;
            // The first/last changed indices are present only when the flag is.
            if info & 0x200 != 0 {
                if hinted(buf, hint, &mut at).is_none() {
                    break;
                }
                if hinted(buf, hint, &mut at).is_none() {
                    break;
                }
            }
            if info & 0x100 != 0 && hinted(buf, hint, &mut at).is_none() {
                break;
            }
        }
        let Some(name_size) = hinted(buf, hint, &mut at) else {
            break;
        };
        let Some(name) = buf.get(at..at + name_size as usize) else {
            break;
        };
        at += name_size as usize;
        // A metadata index of 0xff means the string follows inline.
        if (info & 0xff00000) >> 20 == 0xff {
            let Some(size) = hinted(buf, hint, &mut at) else {
                break;
            };
            at += size as usize;
        }
        out.push(PropertyHeader {
            name: String::from_utf8_lossy(name).into_owned(),
            kind,
            pod,
            extent,
            sample_count,
        });
    }
    out
}

/// Parse an object-header block.
///
/// The block ends with 32 bytes of hashes, which are not headers — reading them
/// as one yields a name of several gigabytes.
fn object_headers(buf: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let Some(buf) = buf.len().checked_sub(32).and_then(|n| buf.get(..n)) else {
        return out;
    };
    let mut at = 0usize;
    while at + 4 <= buf.len() {
        let size = u32::from_le_bytes(buf[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        if size == 0 || at + size + 1 > buf.len() {
            break;
        }
        out.push(String::from_utf8_lossy(&buf[at..at + size]).into_owned());
        at += size;
        let metadata_index = buf[at];
        at += 1;
        if metadata_index == 0xff {
            let Some(bytes) = buf.get(at..at + 4) else {
                break;
            };
            let size = u32::from_le_bytes(bytes.try_into().unwrap()) as usize;
            at += 4 + size;
        }
    }
    out
}

impl<'a> Archive<'a> {
    pub fn open(bytes: &'a [u8]) -> Result<Self, AlembicError> {
        let file = Ogawa::open(bytes)?;
        let children = file.group(file.root)?;
        // The archive root: version, file version, the top object, metadata,
        // time samplings, indexed metadata.
        let metadata = file
            .child_data(&children, 3)
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default();
        let top = match children.get(2) {
            Some(Child::Group(at)) => *at,
            _ => return Err(AlembicError::Corrupt),
        };
        let root = read_object(&file, top, "", "", 0)?;
        Ok(Archive {
            file,
            root,
            metadata,
        })
    }

    /// Every `PolyMesh` in the archive, at sample zero.
    pub fn poly_meshes(&self) -> Vec<PolyMesh> {
        let mut out = Vec::new();
        collect_meshes(&self.file, &self.root, &mut out);
        out
    }
}

/// How deep an object tree this will follow. Real caches nest a handful; a
/// cycle in the offsets would otherwise recurse until the stack ran out.
const MAX_DEPTH: usize = 32;

fn read_object(
    file: &Ogawa<'_>,
    at: usize,
    name: &str,
    parent: &str,
    depth: usize,
) -> Result<Object, AlembicError> {
    if depth > MAX_DEPTH {
        return Err(AlembicError::Corrupt);
    }
    let children = file.group(at)?;
    let full_name = if name.is_empty() {
        String::new()
    } else {
        format!("{parent}/{name}")
    };

    // The last child names this object's children; the first is its properties.
    let names = children
        .last()
        .and_then(|c| match c {
            Child::Data(at) => file.data_at(*at).ok(),
            _ => None,
        })
        .map(object_headers)
        .unwrap_or_default();

    let property_group = match children.first() {
        Some(Child::Group(at)) => Some(*at),
        _ => None,
    };
    let properties = property_group
        .map(|at| read_property_headers(file, at))
        .unwrap_or_default();

    // Children sit between the properties group and the header block.
    let mut kids = Vec::new();
    for (i, child) in children.iter().enumerate().skip(1) {
        if i + 1 >= children.len() {
            break;
        }
        if let Child::Group(child_at) = child {
            let child_name = names.get(i - 1).cloned().unwrap_or_default();
            if let Ok(object) = read_object(file, *child_at, &child_name, &full_name, depth + 1) {
                kids.push(object);
            }
        }
    }
    Ok(Object {
        name: name.to_string(),
        full_name,
        children: kids,
        properties,
        property_group,
    })
}

fn read_property_headers(file: &Ogawa<'_>, at: usize) -> Vec<PropertyHeader> {
    let Ok(children) = file.group(at) else {
        return Vec::new();
    };
    children
        .last()
        .and_then(|c| match c {
            Child::Data(at) => file.data_at(*at).ok(),
            _ => None,
        })
        .map(property_headers)
        .unwrap_or_default()
}

/// A property's sub-group by name, for walking into a compound.
fn property_child(file: &Ogawa<'_>, group: usize, name: &str) -> Option<usize> {
    let children = file.group(group).ok()?;
    let headers = read_property_headers(file, group);
    let index = headers.iter().position(|h| h.name == name)?;
    match children.get(index) {
        Some(Child::Group(at)) => Some(*at),
        _ => None,
    }
}

/// Sample zero of an array property, with the 16-byte key skipped.
fn sample_bytes<'a>(file: &Ogawa<'a>, group: usize) -> Option<&'a [u8]> {
    let children = file.group(group).ok()?;
    let data = children.iter().find_map(|c| match c {
        Child::Data(at) => file.data_at(*at).ok(),
        _ => None,
    })?;
    // Every array sample begins with a key identifying its contents, which is
    // not part of the data. Reading it as values shifts everything by four
    // floats and yields a mesh that is almost right.
    data.get(16..)
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn ints(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn collect_meshes(file: &Ogawa<'_>, object: &Object, out: &mut Vec<PolyMesh>) {
    if let Some(group) = object.property_group {
        // A PolyMesh keeps its geometry in a compound called `.geom`.
        if let Some(geom) = property_child(file, group, ".geom") {
            let read_floats = |name: &str| {
                property_child(file, geom, name)
                    .and_then(|g| sample_bytes(file, g))
                    .map(floats)
                    .unwrap_or_default()
            };
            let read_ints = |name: &str| {
                property_child(file, geom, name)
                    .and_then(|g| sample_bytes(file, g))
                    .map(ints)
                    .unwrap_or_default()
            };
            let points = read_floats("P");
            let face_counts = read_ints(".faceCounts");
            let face_indices = read_ints(".faceIndices");
            if !points.is_empty() && !face_counts.is_empty() {
                out.push(PolyMesh {
                    name: object.name.clone(),
                    points,
                    face_counts,
                    face_indices,
                });
            }
        }
    }
    for child in &object.children {
        collect_meshes(file, child, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCENE: &[u8] = include_bytes!("testdata/scene.abc");

    /// The container opens and the archive metadata reads.
    #[test]
    fn an_archive_opens() {
        let archive = Archive::open(SCENE).expect("opens");
        assert!(
            archive.metadata.contains("FramesPerTimeUnit"),
            "{}",
            archive.metadata
        );
        assert!(archive.metadata.contains("Alembic 1.8"), "written by Alembic");
    }

    /// The object hierarchy is the one the file holds.
    ///
    /// Blender nests each mesh inside an `Xform` of the same name, and parents
    /// the second object under the first — so the tree is
    /// `Cube(Xform) → {Cube(PolyMesh), Ball(Xform) → Sphere(PolyMesh)}`.
    #[test]
    fn the_object_hierarchy_reads() {
        let archive = Archive::open(SCENE).expect("opens");
        let top = &archive.root.children;
        assert_eq!(top.len(), 1, "one root object");
        assert_eq!(top[0].name, "Cube");
        let names: Vec<&str> = top[0].children.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["Cube", "Ball"], "the mesh and the child xform");
        assert_eq!(top[0].children[1].children[0].name, "Sphere");
        assert_eq!(
            top[0].children[1].children[0].full_name,
            "/Cube/Ball/Sphere",
            "full names are built from the walk"
        );
    }

    /// Property headers decode: kinds, POD types and extents.
    ///
    /// The header is a packed bitfield with a size hint that decides how wide
    /// the following integers are, so a wrong hint reads the name as a number
    /// and everything after it is lost. `.childBnds` being six doubles — a
    /// bounding box — is that having worked.
    #[test]
    fn property_headers_decode() {
        let archive = Archive::open(SCENE).expect("opens");
        let bounds = &archive.root.properties[0];
        assert_eq!(bounds.name, ".childBnds");
        assert_eq!(bounds.kind, PropertyKind::Scalar);
        assert_eq!(bounds.pod, Some(Pod::F64));
        assert_eq!(bounds.extent, 6, "a bounding box is two corners");

        let xform = &archive.root.children[0];
        let names: Vec<&str> = xform.properties.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, [".xform", "visible"]);
        assert_eq!(xform.properties[0].kind, PropertyKind::Compound);
        assert_eq!(xform.properties[1].pod, Some(Pod::I8));
    }

    /// The geometry is the mesh the file holds, to the vertex.
    ///
    /// These counts are not "plausible" — they are forced. A cube has 8 points
    /// and 6 quads. An 8-segment, 6-ring UV sphere has 8×5+2 = 42 points, 8×6 =
    /// 48 faces, and 32 quads plus 16 pole triangles = 176 corners. A stride
    /// slip or a missed sample key gives none of those numbers.
    #[test]
    fn poly_mesh_geometry_reads() {
        let archive = Archive::open(SCENE).expect("opens");
        let meshes = archive.poly_meshes();
        assert_eq!(meshes.len(), 2, "a cube and a sphere");

        let cube = meshes.iter().find(|m| m.name == "Cube").expect("the cube");
        assert_eq!(cube.points.len() / 3, 8);
        assert_eq!(cube.face_counts, vec![4; 6], "six quads");
        assert_eq!(cube.face_indices.len(), 24);
        // Every corner of a 2-unit cube is at ±1.
        for v in &cube.points {
            assert!((v.abs() - 1.0).abs() < 1e-5, "{v} is not a cube corner");
        }

        let sphere = meshes.iter().find(|m| m.name == "Sphere").expect("the sphere");
        assert_eq!(sphere.points.len() / 3, 42);
        assert_eq!(sphere.face_counts.len(), 48);
        assert_eq!(sphere.face_indices.len(), 176);
        assert_eq!(
            sphere.face_counts.iter().filter(|c| **c == 3).count(),
            16,
            "two pole rings of triangles"
        );
        // And it is a unit sphere, which a mis-strided read would not be.
        for p in sphere.points.chunks_exact(3) {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 1.0).abs() < 1e-4, "{p:?} is at radius {r}");
        }
    }

    /// The face indices address real points, and the counts account for them
    /// all.
    #[test]
    fn the_topology_is_consistent() {
        let archive = Archive::open(SCENE).expect("opens");
        for mesh in archive.poly_meshes() {
            let points = mesh.points.len() / 3;
            assert_eq!(
                mesh.face_counts.iter().sum::<i32>() as usize,
                mesh.face_indices.len(),
                "{}: counts and indices disagree",
                mesh.name
            );
            for &i in &mesh.face_indices {
                assert!(
                    i >= 0 && (i as usize) < points,
                    "{}: index {i} past {points} points",
                    mesh.name
                );
            }
        }
    }

    /// An unfinished file is refused rather than read from untrustworthy
    /// offsets.
    ///
    /// Ogawa clears a byte in the header until the writer completes, which is
    /// how a cache still being written announces itself — and reading its
    /// offsets anyway is how a loader hangs on a file that is merely early.
    #[test]
    fn an_unfinished_file_is_refused() {
        let mut bytes = SCENE.to_vec();
        bytes[5] = 0x00;
        assert!(matches!(
            Archive::open(&bytes),
            Err(AlembicError::Incomplete)
        ));
    }

    /// Something else entirely is refused, and truncation never panics.
    #[test]
    fn a_non_alembic_file_is_refused_and_truncation_is_safe() {
        assert!(matches!(
            Archive::open(b"not alembic at all"),
            Err(AlembicError::NotAlembic)
        ));
        for cut in (0..SCENE.len()).step_by(37) {
            let _ = Archive::open(&SCENE[..cut]).map(|a| a.poly_meshes());
        }
    }

    /// Bytes flipped anywhere do not panic either.
    #[test]
    fn corruption_does_not_panic() {
        let mut bytes = SCENE.to_vec();
        for step in 0..150 {
            let at = (step * 61 + 17) % bytes.len();
            let was = bytes[at];
            bytes[at] = was.wrapping_add(0x7f);
            let _ = Archive::open(&bytes).map(|a| a.poly_meshes());
            bytes[at] = was;
        }
    }
}
