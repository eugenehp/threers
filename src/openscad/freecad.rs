//! FreeCAD **`.FCStd`** mesh import/export — ZIP of `Document.xml` plus
//! `MeshKernel*.bms` binaries, with no FreeCAD/OCC link.
//!
//! Analytic PartDesign / BREP trees are out of scope (use STEP for that). This
//! is the FreeCAD-native twin of STL/OBJ/3MF: triangle meshes as
//! `Mesh::Feature` objects.
//!
//! # Quick path
//!
//! ```ignore
//! use threers::{cube, geometry_to_fcstd, FcstdLoader};
//!
//! let bytes = geometry_to_fcstd(&cube([10.0, 10.0, 10.0]).to_geometry_exact(), "Cube");
//! let doc = FcstdLoader::parse(&bytes)?;
//! assert_eq!(doc.meshes.len(), 1);
//! ```
//!
//! # Writer / loader
//!
//! [`FcstdWriter`] builds multi-object documents with placements and metadata.
//! [`FcstdLoader`] returns a typed [`FcstdDocument`] with a skip report — same
//! “decline rather than approximate” rule as STEP: missing sidecars and bad
//! kernels are named, not silently dropped without a record.

use crate::core::{BufferAttribute, BufferGeometry};
use crate::openscad::export::Zip;
use crate::openscad::{polyhedron, ScadPart, Solid};
use std::collections::HashMap;
use std::fmt;

/// Normalise a geometry to `(vertices, triangle indices)`.
fn tri_data(g: &BufferGeometry) -> Option<(Vec<[f32; 3]>, Vec<u32>)> {
    let pos = &g.get_attribute("position")?.array;
    let verts: Vec<[f32; 3]> = pos.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    if verts.len() < 3 {
        return None;
    }
    let indices = g
        .index
        .clone()
        .unwrap_or_else(|| (0..verts.len() as u32).collect());
    if indices.len() < 3 {
        return None;
    }
    Some((verts, indices))
}

// --- public types -----------------------------------------------------------

/// Rigid placement: translation + quaternion `(x, y, z, w)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FcstdPlacement {
    pub position: [f32; 3],
    /// Quaternion as FreeCAD writes it (`Q0..Q3` = x,y,z,w).
    pub rotation: [f32; 4],
}

impl Default for FcstdPlacement {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl FcstdPlacement {
    pub fn identity() -> Self {
        Self::default()
    }

    pub fn translate(position: [f32; 3]) -> Self {
        Self {
            position,
            ..Self::default()
        }
    }

    /// Axis-angle rotation (axis need not be unit; angle in radians) then translate.
    pub fn from_axis_angle(axis: [f32; 3], angle: f32, position: [f32; 3]) -> Self {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if len < 1e-20 || angle.abs() < 1e-20 {
            return Self::translate(position);
        }
        let (x, y, z) = (axis[0] / len, axis[1] / len, axis[2] / len);
        let half = angle * 0.5;
        let s = half.sin();
        Self {
            position,
            rotation: [x * s, y * s, z * s, half.cos()],
        }
    }

    pub fn is_identity(self) -> bool {
        self == Self::identity()
    }

    pub fn transform_point(self, p: [f32; 3]) -> [f32; 3] {
        apply_placement(p, self.position, self.rotation)
    }
}

/// Document-level metadata written into `Document.xml`.
#[derive(Debug, Clone, PartialEq)]
pub struct FcstdDocumentMeta {
    pub label: String,
    pub comment: String,
    pub created_by: String,
    pub company: String,
    /// UUID string. Empty / `"auto"` → content-addressed deterministic UUID on write.
    pub uid: String,
}

impl Default for FcstdDocumentMeta {
    fn default() -> Self {
        Self {
            label: "Unnamed".into(),
            comment: "exported by threers".into(),
            created_by: "threers".into(),
            company: String::new(),
            uid: "auto".into(),
        }
    }
}

/// Options for [`FcstdWriter`].
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct FcstdWriteOptions {
    pub meta: FcstdDocumentMeta,
    /// When true, weld coincident vertices and write facet neighbour indices
    /// (better topology for FreeCAD mesh tools). Default: false.
    pub neighbours: bool,
    /// Weld coincident vertices even when `neighbours` is false. Default: false.
    pub weld: bool,
}


/// One mesh to include in an export.
#[derive(Debug, Clone)]
pub struct FcstdMeshSpec {
    /// FreeCAD internal object name (sanitized on write).
    pub name: String,
    /// Display label; defaults to `name` when `None`.
    pub label: Option<String>,
    pub geometry: BufferGeometry,
    pub placement: FcstdPlacement,
    /// Optional display colour (linear RGBA `0..=1`), round-tripped via Document.xml.
    pub color: Option<[f32; 4]>,
}

impl FcstdMeshSpec {
    pub fn new(name: impl Into<String>, geometry: BufferGeometry) -> Self {
        Self {
            name: name.into(),
            label: None,
            geometry,
            placement: FcstdPlacement::identity(),
            color: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_placement(mut self, placement: FcstdPlacement) -> Self {
        self.placement = placement;
        self
    }

    pub fn with_color(mut self, rgba: [f32; 4]) -> Self {
        self.color = Some(rgba);
        self
    }

    /// Build specs from colored OpenSCAD parts (names `Part1…`, colors preserved).
    pub fn from_parts(parts: &[ScadPart]) -> Vec<Self> {
        parts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut s = Self::new(format!("Part{}", i + 1), p.geometry.clone());
                if let Some(c) = p.color {
                    s = s.with_color(c).with_label(format!(
                        "Part{} #{:02X}{:02X}{:02X}",
                        i + 1,
                        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
                        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
                        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
                    ));
                }
                s
            })
            .collect()
    }
}

/// Options for [`FcstdLoader`].
#[derive(Debug, Clone)]
pub struct FcstdReadOptions {
    /// Bake `PropertyPlacement` into vertex positions (default `true`).
    pub apply_placement: bool,
    /// Require `Document.xml`; if false, fall back to loading every `.bms`.
    pub require_document: bool,
}

impl Default for FcstdReadOptions {
    fn default() -> Self {
        Self {
            apply_placement: true,
            require_document: false,
        }
    }
}

/// Why a mesh object could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FcstdSkipped {
    /// `Document.xml` named a sidecar that is missing from the ZIP.
    MissingFile { object: String, file: String },
    /// Sidecar present but not a valid `MeshKernel.bms`.
    InvalidKernel { object: String, file: String },
    /// Geometry had too few vertices/facets.
    EmptyMesh { object: String },
}

impl fmt::Display for FcstdSkipped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingFile { object, file } => {
                write!(f, "object {object}: missing mesh file {file}")
            }
            Self::InvalidKernel { object, file } => {
                write!(f, "object {object}: invalid MeshKernel in {file}")
            }
            Self::EmptyMesh { object } => write!(f, "object {object}: empty mesh"),
        }
    }
}

/// Import / export failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FcstdError {
    /// Bytes are not a ZIP archive (or have no EOCD).
    NotZip,
    /// `Document.xml` was required and missing or unreadable.
    MissingDocument,
    /// Archive opened but no usable mesh was found.
    Empty,
    /// A mesh spec had no triangle data.
    EmptyGeometry { name: String },
    /// Filesystem failure (message only — keeps the error `Eq`).
    Io(String),
}

impl fmt::Display for FcstdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotZip => write!(f, "not a ZIP / FCStd archive"),
            Self::MissingDocument => write!(f, "Document.xml missing or unreadable"),
            Self::Empty => write!(f, "FCStd contains no usable meshes"),
            Self::EmptyGeometry { name } => write!(f, "mesh {name}: empty geometry"),
            Self::Io(msg) => write!(f, "I/O: {msg}"),
        }
    }
}

impl std::error::Error for FcstdError {}

impl From<std::io::Error> for FcstdError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// One mesh object loaded from an `.FCStd` document.
#[derive(Debug, Clone)]
pub struct FcstdMesh {
    /// FreeCAD object name.
    pub name: String,
    /// Display label from `Document.xml` (falls back to `name`).
    pub label: String,
    pub geometry: BufferGeometry,
    /// Placement as stored in the document (identity if absent).
    pub placement: FcstdPlacement,
    /// Optional colour from Document.xml (`DiffuseColor`), if present.
    pub color: Option<[f32; 4]>,
    /// Vertex count before any placement bake.
    pub points: usize,
    /// Triangle count.
    pub facets: usize,
}

impl FcstdMesh {
    pub fn triangle_count(&self) -> usize {
        self.facets
    }

    /// Axis-aligned bounds of the (possibly placement-baked) geometry.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        crate::openscad::geometry_bounds(&self.geometry)
    }
}

/// Parsed FreeCAD document.
#[derive(Debug, Clone)]
pub struct FcstdDocument {
    pub meta: FcstdDocumentMeta,
    pub meshes: Vec<FcstdMesh>,
    /// Objects that could not be carried — reported, not approximated.
    pub skipped: Vec<FcstdSkipped>,
    /// Whether mesh geometries already include their placements (from load options).
    pub placement_applied: bool,
}

impl FcstdDocument {
    pub fn label(&self) -> &str {
        &self.meta.label
    }

    pub fn is_complete(&self) -> bool {
        self.skipped.is_empty() && !self.meshes.is_empty()
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    pub fn total_facets(&self) -> usize {
        self.meshes.iter().map(|m| m.facets).sum()
    }

    pub fn total_points(&self) -> usize {
        self.meshes.iter().map(|m| m.points).sum()
    }

    /// Look up a mesh by FreeCAD object name.
    pub fn mesh(&self, name: &str) -> Option<&FcstdMesh> {
        self.meshes.iter().find(|m| m.name == name)
    }

    /// Look up by display label (first match).
    pub fn mesh_by_label(&self, label: &str) -> Option<&FcstdMesh> {
        self.meshes.iter().find(|m| m.label == label)
    }

    /// Union of all mesh AABBs, or `None` if empty.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut iter = self.meshes.iter().map(FcstdMesh::bounds);
        let (mut lo, mut hi) = iter.next()?;
        for (a, b) in iter {
            for i in 0..3 {
                lo[i] = lo[i].min(a[i]);
                hi[i] = hi[i].max(b[i]);
            }
        }
        Some((lo, hi))
    }

    /// Merge every mesh into one indexed [`BufferGeometry`].
    pub fn to_geometry(&self) -> Option<BufferGeometry> {
        merge_meshes(&self.meshes)
    }

    /// Merge into a [`Solid`] leaf (for CSG / `import()`).
    pub fn to_solid(&self) -> Option<Solid> {
        merge_solid(&self.meshes)
    }

    /// Re-export this document (names, labels, colors preserved).
    ///
    /// If placements were baked into vertices on load, they are written as
    /// identity so FreeCAD does not transform twice.
    pub fn to_fcstd(
        &self,
        opts: &FcstdWriteOptions,
    ) -> Result<(Vec<u8>, FcstdWriteReport), FcstdError> {
        let specs: Vec<FcstdMeshSpec> = self
            .meshes
            .iter()
            .map(|m| {
                let place = if self.placement_applied {
                    FcstdPlacement::identity()
                } else {
                    m.placement
                };
                let mut s = FcstdMeshSpec::new(m.name.clone(), m.geometry.clone())
                    .with_label(m.label.clone())
                    .with_placement(place);
                if let Some(c) = m.color {
                    s = s.with_color(c);
                }
                s
            })
            .collect();
        let mut o = opts.clone();
        if o.meta.label == "Unnamed" || o.meta.label.is_empty() {
            o.meta = self.meta.clone();
        }
        FcstdWriter::write(&specs, &o)
    }
}

/// Write report: what went into the archive.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FcstdWriteReport {
    pub objects: usize,
    pub points: usize,
    pub facets: usize,
    pub neighbours: bool,
}

// --- MeshKernel.bms ---------------------------------------------------------

fn facet_neighbours(indices: &[u32]) -> Vec<[u32; 3]> {
    // Edge (min,max) → (facet, local edge slot). Side e joins verts e and (e+1)%3.
    let n = indices.len() / 3;
    let mut edge_owner: HashMap<(u32, u32), (u32, u8)> = HashMap::with_capacity(n * 3);
    let mut neigh = vec![[0xFFFF_FFFFu32; 3]; n];
    for (fi, t) in indices.chunks_exact(3).enumerate() {
        let fi = fi as u32;
        for e in 0u8..3 {
            let a = t[e as usize];
            let b = t[((e + 1) % 3) as usize];
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some((other, oe)) = edge_owner.remove(&key) {
                neigh[fi as usize][e as usize] = other;
                neigh[other as usize][oe as usize] = fi;
            } else {
                edge_owner.insert(key, (fi, e));
            }
        }
    }
    neigh
}

/// Weld vertices that agree to ~1e-5 so neighbour indices can connect faces that
/// only share positions (typical of three.js-style box meshes).
fn weld_mesh(verts: &[[f32; 3]], indices: &[u32]) -> (Vec<[f32; 3]>, Vec<u32>) {
    const Q: f32 = 1e5;
    let mut map: HashMap<(i64, i64, i64), u32> = HashMap::with_capacity(verts.len());
    let mut out_verts = Vec::with_capacity(verts.len());
    let mut remap = vec![0u32; verts.len()];
    for (i, v) in verts.iter().enumerate() {
        let key = (
            (v[0] * Q).round() as i64,
            (v[1] * Q).round() as i64,
            (v[2] * Q).round() as i64,
        );
        let id = *map.entry(key).or_insert_with(|| {
            let id = out_verts.len() as u32;
            out_verts.push(*v);
            id
        });
        remap[i] = id;
    }
    let out_idx: Vec<u32> = indices.iter().map(|&i| remap[i as usize]).collect();
    (out_verts, out_idx)
}

/// FreeCAD `MeshKernel::Write` binary (`.bms`).
fn mesh_kernel_bms(verts: &[[f32; 3]], indices: &[u32], neighbours: bool, weld: bool) -> Vec<u8> {
    let (verts, indices) = if neighbours || weld {
        weld_mesh(verts, indices)
    } else {
        (verts.to_vec(), indices.to_vec())
    };
    let n_pts = verts.len() as u32;
    let n_fts = (indices.len() / 3) as u32;
    let mut out = Vec::with_capacity(8 + 256 + 8 + verts.len() * 12 + (indices.len() / 3) * 24 + 24);
    out.extend_from_slice(&0xA0B0_C0D0u32.to_le_bytes());
    out.extend_from_slice(&0x0001_0000u32.to_le_bytes());

    let banner = b"MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-\
MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-\
MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-MESH-\
MESH-MESH-MESH-\n";
    debug_assert_eq!(banner.len(), 256);
    out.extend_from_slice(banner);

    out.extend_from_slice(&n_pts.to_le_bytes());
    out.extend_from_slice(&n_fts.to_le_bytes());

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in &verts {
        for i in 0..3 {
            min[i] = min[i].min(v[i]);
            max[i] = max[i].max(v[i]);
        }
        out.extend_from_slice(&v[0].to_le_bytes());
        out.extend_from_slice(&v[1].to_le_bytes());
        out.extend_from_slice(&v[2].to_le_bytes());
    }
    if verts.is_empty() {
        min = [0.0; 3];
        max = [0.0; 3];
    }

    let neigh = if neighbours {
        facet_neighbours(&indices)
    } else {
        vec![[0xFFFF_FFFFu32; 3]; n_fts as usize]
    };
    for (ti, t) in indices.chunks_exact(3).enumerate() {
        out.extend_from_slice(&t[0].to_le_bytes());
        out.extend_from_slice(&t[1].to_le_bytes());
        out.extend_from_slice(&t[2].to_le_bytes());
        let n = neigh[ti];
        out.extend_from_slice(&n[0].to_le_bytes());
        out.extend_from_slice(&n[1].to_le_bytes());
        out.extend_from_slice(&n[2].to_le_bytes());
    }

    for v in [min[0], max[0], min[1], max[1], min[2], max[2]] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Parse a FreeCAD `MeshKernel.bms` blob into vertices and triangle indices.
pub fn parse_mesh_kernel(bms: &[u8]) -> Option<(Vec<[f32; 3]>, Vec<u32>)> {
    parse_mesh_kernel_full(bms).map(|(v, i, _)| (v, i))
}

/// A parsed mesh kernel: vertices, triangle indices, and per-facet neighbours.
pub type MeshKernel = (Vec<[f32; 3]>, Vec<u32>, Vec<[u32; 3]>);

/// Like [`parse_mesh_kernel`], also returning per-facet neighbour indices.
pub fn parse_mesh_kernel_full(bms: &[u8]) -> Option<MeshKernel> {
    if bms.len() < 8 + 256 + 8 {
        return None;
    }
    let magic = u32::from_le_bytes(bms[0..4].try_into().ok()?);
    let version = u32::from_le_bytes(bms[4..8].try_into().ok()?);
    let be = magic.swap_bytes() == 0xA0B0_C0D0 && version.swap_bytes() == 0x0001_0000;
    let le = magic == 0xA0B0_C0D0 && version == 0x0001_0000;
    if !le && !be {
        return None;
    }
    let u32_at = |o: usize| -> Option<u32> {
        let b: [u8; 4] = bms.get(o..o + 4)?.try_into().ok()?;
        Some(if be {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    };
    let f32_at = |o: usize| -> Option<f32> {
        let b: [u8; 4] = bms.get(o..o + 4)?.try_into().ok()?;
        Some(if be {
            f32::from_be_bytes(b)
        } else {
            f32::from_le_bytes(b)
        })
    };

    let mut o = 8 + 256;
    let n_pts = u32_at(o)? as usize;
    o += 4;
    let n_fts = u32_at(o)? as usize;
    o += 4;

    let need = o + n_pts * 12 + n_fts * 24 + 24;
    if bms.len() < need {
        return None;
    }

    let mut verts = Vec::with_capacity(n_pts);
    for _ in 0..n_pts {
        verts.push([f32_at(o)?, f32_at(o + 4)?, f32_at(o + 8)?]);
        o += 12;
    }
    let mut indices = Vec::with_capacity(n_fts * 3);
    let mut neighbours = Vec::with_capacity(n_fts);
    for _ in 0..n_fts {
        let a = u32_at(o)?;
        let b = u32_at(o + 4)?;
        let c = u32_at(o + 8)?;
        let n0 = u32_at(o + 12)?;
        let n1 = u32_at(o + 16)?;
        let n2 = u32_at(o + 20)?;
        o += 24;
        if (a as usize) < n_pts && (b as usize) < n_pts && (c as usize) < n_pts {
            indices.extend_from_slice(&[a, b, c]);
            neighbours.push([n0, n1, n2]);
        }
    }
    (verts.len() >= 3 && indices.len() >= 3).then_some((verts, indices, neighbours))
}

// --- ZIP --------------------------------------------------------------------

fn zip_has_eocd(z: &[u8]) -> bool {
    (0..z.len().saturating_sub(21))
        .rev()
        .any(|i| z[i..].starts_with(b"PK\x05\x06"))
}

fn zip_read_entry(z: &[u8], name_exact: Option<&str>, suffix: Option<&str>) -> Option<Vec<u8>> {
    let u16 = |o: usize| -> Option<usize> {
        Some(u16::from_le_bytes(z.get(o..o + 2)?.try_into().ok()?) as usize)
    };
    let u32 = |o: usize| -> Option<usize> {
        Some(u32::from_le_bytes(z.get(o..o + 4)?.try_into().ok()?) as usize)
    };
    let eocd = (0..z.len().saturating_sub(21))
        .rev()
        .find(|&i| z[i..].starts_with(b"PK\x05\x06"))?;
    let count = u16(eocd + 10)?;
    let mut p = u32(eocd + 16)?;
    for _ in 0..count {
        if !z.get(p..p + 4)?.starts_with(b"PK\x01\x02") {
            break;
        }
        let method = u16(p + 10)?;
        let comp_size = u32(p + 20)?;
        let name_len = u16(p + 28)?;
        let extra_len = u16(p + 30)?;
        let comment_len = u16(p + 32)?;
        let local_off = u32(p + 42)?;
        let name = std::str::from_utf8(z.get(p + 46..p + 46 + name_len)?).ok()?;
        let matched = match (name_exact, suffix) {
            (Some(exact), _) => name == exact || name.ends_with(&format!("/{exact}")),
            (_, Some(suf)) => name.ends_with(suf),
            _ => false,
        };
        if matched {
            if !z.get(local_off..local_off + 4)?.starts_with(b"PK\x03\x04") {
                return None;
            }
            let l_name = u16(local_off + 26)?;
            let l_extra = u16(local_off + 28)?;
            let start = local_off + 30 + l_name + l_extra;
            let data = z.get(start..start + comp_size)?;
            return match method {
                0 => Some(data.to_vec()),
                8 => crate::loaders::deflate::inflate_raw(data).ok(),
                _ => None,
            };
        }
        p += 46 + name_len + extra_len + comment_len;
    }
    None
}

fn zip_list_names(z: &[u8]) -> Vec<String> {
    let mut names = Vec::new();
    let u16 = |o: usize| -> Option<usize> {
        Some(u16::from_le_bytes(z.get(o..o + 2)?.try_into().ok()?) as usize)
    };
    let u32 = |o: usize| -> Option<usize> {
        Some(u32::from_le_bytes(z.get(o..o + 4)?.try_into().ok()?) as usize)
    };
    let Some(eocd) = (0..z.len().saturating_sub(21))
        .rev()
        .find(|&i| z[i..].starts_with(b"PK\x05\x06"))
    else {
        return names;
    };
    let Some(count) = u16(eocd + 10) else {
        return names;
    };
    let Some(mut p) = u32(eocd + 16) else {
        return names;
    };
    for _ in 0..count {
        if !z.get(p..).is_some_and(|s| s.starts_with(b"PK\x01\x02")) {
            break;
        }
        let Some(name_len) = u16(p + 28) else { break };
        let Some(extra_len) = u16(p + 30) else { break };
        let Some(comment_len) = u16(p + 32) else { break };
        if let Some(name) = z
            .get(p + 46..p + 46 + name_len)
            .and_then(|b| std::str::from_utf8(b).ok())
        {
            names.push(name.to_string());
        }
        p += 46 + name_len + extra_len + comment_len;
    }
    names
}

// --- Document.xml -----------------------------------------------------------

fn xml_attr_str<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("{name}=");
    let mut s = tag;
    loop {
        let i = s.find(&key)?;
        let whole = i == 0 || s.as_bytes()[i - 1].is_ascii_whitespace();
        let after = s[i + key.len()..].trim_start();
        if whole {
            let b = after.as_bytes();
            if let Some(&q) = b.first().filter(|&&c| c == b'"' || c == b'\'') {
                let v = &after[1..];
                if let Some(end) = v.find(q as char) {
                    return Some(&v[..end]);
                }
            }
        }
        s = &s[i + key.len()..];
    }
}

fn xml_attr_f32(tag: &str, name: &str) -> Option<f32> {
    xml_attr_str(tag, name)?.parse().ok()
}

fn xml_string_prop(block: &str, prop_name: &str) -> Option<String> {
    let needle = format!("name=\"{prop_name}\"");
    let mut rest = block;
    while let Some(i) = rest.find(&needle) {
        let after = &rest[i..];
        if let Some(sv) = after.find("<String value=\"") {
            let v = &after[sv + 15..];
            if let Some(end) = v.find('"') {
                return Some(v[..end].to_string());
            }
        }
        rest = &rest[i + needle.len()..];
    }
    None
}

fn parse_document_meta(xml: &str) -> FcstdDocumentMeta {
    let head = xml.split("<ObjectData").next().unwrap_or(xml);
    let mut meta = FcstdDocumentMeta::default();
    if let Some(v) = xml_string_prop(head, "Label") {
        meta.label = v;
    }
    if let Some(v) = xml_string_prop(head, "Comment") {
        meta.comment = v;
    }
    if let Some(v) = xml_string_prop(head, "CreatedBy") {
        meta.created_by = v;
    }
    if let Some(v) = xml_string_prop(head, "Company") {
        meta.company = v;
    }
    // Uid is <Uuid value="…"/> not String.
    if let Some(i) = head.find("name=\"Uid\"") {
        let after = &head[i..];
        if let Some(sv) = after.find("<Uuid value=\"") {
            let v = &after[sv + 13..];
            if let Some(end) = v.find('"') {
                meta.uid = v[..end].to_string();
            }
        }
    }
    meta
}

fn parse_color_prop(block: &str) -> Option<[f32; 4]> {
    // <PropertyColor r="…" g="…" b="…" a="…"/> or value="#RRGGBBAA"
    for frag in block.split("<PropertyColor").skip(1) {
        let Some(gt) = frag.find('>') else { continue };
        let tag = &frag[..gt];
        if let (Some(r), Some(g), Some(b)) = (
            xml_attr_f32(tag, "r"),
            xml_attr_f32(tag, "g"),
            xml_attr_f32(tag, "b"),
        ) {
            let a = xml_attr_f32(tag, "a").unwrap_or(1.0);
            return Some([r, g, b, a]);
        }
        if let Some(v) = xml_attr_str(tag, "value") {
            let hex = v.trim().trim_start_matches('#');
            if hex.len() >= 6 {
                let parse = |i| u8::from_str_radix(&hex[i..i + 2], 16).ok().map(|x| x as f32 / 255.0);
                if let (Some(r), Some(g), Some(b)) = (parse(0), parse(2), parse(4)) {
                    let a = if hex.len() >= 8 {
                        parse(6).unwrap_or(1.0)
                    } else {
                        1.0
                    };
                    return Some([r, g, b, a]);
                }
            }
        }
    }
    None
}

/// One mesh object referenced by `Document.xml`.
#[derive(Debug, Clone)]
struct DocMesh {
    name: String,
    label: String,
    bms_file: String,
    placement: FcstdPlacement,
    color: Option<[f32; 4]>,
}

fn parse_document_meshes(xml: &str) -> Vec<DocMesh> {
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(i) = rest.find("<Object") {
        let after = &rest[i + 7..];
        let delim = after
            .chars()
            .next()
            .is_some_and(|c| c.is_whitespace() || c == '>');
        let Some(gt) = after.find('>') else { break };
        if !delim {
            rest = &after[gt + 1..];
            continue;
        }
        let open = &after[..gt];
        if open.contains("type=") || open.trim_end().ends_with('/') {
            rest = &after[gt + 1..];
            continue;
        }
        let name = xml_attr_str(open, "name").unwrap_or("Mesh").to_string();
        let content = &after[gt + 1..];
        let Some(end) = content.find("</Object>") else {
            break;
        };
        let block = &content[..end];
        rest = &content[end + 9..];

        let mut bms_file = None;
        for frag in block.split("<Mesh").skip(1) {
            let Some(gt2) = frag.find('>') else { continue };
            let tag = &frag[..gt2];
            if let Some(f) = xml_attr_str(tag, "file") {
                if !f.is_empty() {
                    bms_file = Some(f.to_string());
                    break;
                }
            }
        }
        let Some(bms_file) = bms_file else { continue };

        let label = xml_string_prop(block, "Label").unwrap_or_else(|| name.clone());
        let color = parse_color_prop(block);
        let mut placement = FcstdPlacement::identity();
        for frag in block.split("<PropertyPlacement").skip(1) {
            let Some(gt2) = frag.find('>') else { continue };
            let tag = &frag[..gt2];
            placement = FcstdPlacement {
                position: [
                    xml_attr_f32(tag, "Px").unwrap_or(0.0),
                    xml_attr_f32(tag, "Py").unwrap_or(0.0),
                    xml_attr_f32(tag, "Pz").unwrap_or(0.0),
                ],
                rotation: [
                    xml_attr_f32(tag, "Q0").unwrap_or(0.0),
                    xml_attr_f32(tag, "Q1").unwrap_or(0.0),
                    xml_attr_f32(tag, "Q2").unwrap_or(0.0),
                    xml_attr_f32(tag, "Q3").unwrap_or(1.0),
                ],
            };
            break;
        }
        out.push(DocMesh {
            name,
            label,
            bms_file,
            placement,
            color,
        });
    }
    out
}

fn apply_placement(p: [f32; 3], t: [f32; 3], q: [f32; 4]) -> [f32; 3] {
    if t == [0.0; 3] && q == [0.0, 0.0, 0.0, 1.0] {
        return p;
    }
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let [vx, vy, vz] = p;
    let ix = w * vx + y * vz - z * vy;
    let iy = w * vy + z * vx - x * vz;
    let iz = w * vz + x * vy - y * vx;
    let iw = -x * vx - y * vy - z * vz;
    let rx = ix * w + iw * -x + iy * -z - iz * -y;
    let ry = iy * w + iw * -y + iz * -x - ix * -z;
    let rz = iz * w + iw * -z + ix * -y - iy * -x;
    [rx + t[0], ry + t[1], rz + t[2]]
}

fn geometry_from_mesh(verts: &[[f32; 3]], indices: &[u32]) -> BufferGeometry {
    let mut positions = Vec::with_capacity(verts.len() * 3);
    for v in verts {
        positions.extend_from_slice(v);
    }
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(positions, 3));
    g.index = Some(indices.to_vec());
    crate::compute_vertex_normals(&mut g);
    g
}

fn merge_meshes(meshes: &[FcstdMesh]) -> Option<BufferGeometry> {
    if meshes.is_empty() {
        return None;
    }
    if meshes.len() == 1 {
        return Some(meshes[0].geometry.clone());
    }
    let mut verts = Vec::new();
    let mut faces = Vec::new();
    for m in meshes {
        let Some((v, idx)) = tri_data(&m.geometry) else {
            continue;
        };
        let base = verts.len() as u32;
        verts.extend_from_slice(&v);
        for t in idx.chunks_exact(3) {
            faces.push(vec![base + t[0], base + t[1], base + t[2]]);
        }
    }
    (!verts.is_empty() && !faces.is_empty()).then(|| match polyhedron(&verts, &faces) {
        Solid::Leaf(g) => g,
        other => other.to_geometry_exact(),
    })
}

fn merge_solid(meshes: &[FcstdMesh]) -> Option<Solid> {
    if meshes.is_empty() {
        return None;
    }
    let mut verts = Vec::new();
    let mut faces = Vec::new();
    for m in meshes {
        let Some((v, idx)) = tri_data(&m.geometry) else {
            continue;
        };
        let base = verts.len() as u32;
        verts.extend_from_slice(&v);
        for t in idx.chunks_exact(3) {
            faces.push(vec![base + t[0], base + t[1], base + t[2]]);
        }
    }
    (verts.len() >= 3 && !faces.is_empty()).then(|| polyhedron(&verts, &faces))
}

// --- loader -----------------------------------------------------------------

/// FreeCAD `.FCStd` reader — counterpart to [`FcstdWriter`] / [`StlLoader`](crate::StlLoader).
pub struct FcstdLoader;

impl FcstdLoader {
    /// Parse with default options (`apply_placement`, Document.xml optional).
    pub fn parse(bytes: &[u8]) -> Result<FcstdDocument, FcstdError> {
        Self::parse_with(bytes, &FcstdReadOptions::default())
    }

    /// Read an `.FCStd` file from disk.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<FcstdDocument, FcstdError> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::parse(&bytes)
    }

    pub fn load_with(
        path: impl AsRef<std::path::Path>,
        opts: &FcstdReadOptions,
    ) -> Result<FcstdDocument, FcstdError> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::parse_with(&bytes, opts)
    }

    pub fn parse_with(bytes: &[u8], opts: &FcstdReadOptions) -> Result<FcstdDocument, FcstdError> {
        if !zip_has_eocd(bytes) {
            return Err(FcstdError::NotZip);
        }
        let doc_xml = zip_read_entry(bytes, Some("Document.xml"), None)
            .and_then(|b| String::from_utf8(b).ok());
        if opts.require_document && doc_xml.is_none() {
            return Err(FcstdError::MissingDocument);
        }
        let meta = doc_xml
            .as_deref()
            .map(parse_document_meta)
            .unwrap_or_default();
        let refs = doc_xml
            .as_deref()
            .map(parse_document_meshes)
            .unwrap_or_default();

        let mut meshes = Vec::new();
        let mut skipped = Vec::new();

        if !refs.is_empty() {
            for r in refs {
                let bms = zip_read_entry(bytes, Some(&r.bms_file), None)
                    .or_else(|| zip_read_entry(bytes, None, Some(&r.bms_file)));
                let Some(bms) = bms else {
                    skipped.push(FcstdSkipped::MissingFile {
                        object: r.name.clone(),
                        file: r.bms_file.clone(),
                    });
                    continue;
                };
                let Some((mut verts, indices)) = parse_mesh_kernel(&bms) else {
                    skipped.push(FcstdSkipped::InvalidKernel {
                        object: r.name.clone(),
                        file: r.bms_file.clone(),
                    });
                    continue;
                };
                if verts.len() < 3 || indices.len() < 3 {
                    skipped.push(FcstdSkipped::EmptyMesh {
                        object: r.name.clone(),
                    });
                    continue;
                }
                let points = verts.len();
                let facets = indices.len() / 3;
                if opts.apply_placement && !r.placement.is_identity() {
                    for v in &mut verts {
                        *v = r.placement.transform_point(*v);
                    }
                }
                meshes.push(FcstdMesh {
                    name: r.name,
                    label: r.label,
                    geometry: geometry_from_mesh(&verts, &indices),
                    placement: r.placement,
                    color: r.color,
                    points,
                    facets,
                });
            }
        } else {
            for name in zip_list_names(bytes) {
                if !name.to_ascii_lowercase().ends_with(".bms") {
                    continue;
                }
                let Some(bms) = zip_read_entry(bytes, Some(&name), None) else {
                    continue;
                };
                let Some((verts, indices)) = parse_mesh_kernel(&bms) else {
                    skipped.push(FcstdSkipped::InvalidKernel {
                        object: name.clone(),
                        file: name.clone(),
                    });
                    continue;
                };
                let label = name
                    .rsplit('/')
                    .next()
                    .unwrap_or(&name)
                    .trim_end_matches(".bms")
                    .to_string();
                let points = verts.len();
                let facets = indices.len() / 3;
                meshes.push(FcstdMesh {
                    name: label.clone(),
                    label,
                    geometry: geometry_from_mesh(&verts, &indices),
                    placement: FcstdPlacement::identity(),
                    color: None,
                    points,
                    facets,
                });
            }
        }

        if meshes.is_empty() {
            return Err(FcstdError::Empty);
        }
        Ok(FcstdDocument {
            meta,
            meshes,
            skipped,
            placement_applied: opts.apply_placement,
        })
    }
}

// --- writer -----------------------------------------------------------------

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn sanitize_name(raw: &str, fallback: &str) -> String {
    let mut out = String::new();
    for (i, c) in raw.chars().enumerate() {
        let ok = if i == 0 {
            c.is_ascii_alphabetic()
        } else {
            c.is_ascii_alphanumeric() || c == '_'
        };
        if ok {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('_') {
            out.push('_');
        }
    }
    if out.is_empty() || !out.chars().next().unwrap().is_ascii_alphabetic() {
        format!("{fallback}{out}")
    } else {
        out
    }
}

struct MeshEntry {
    name: String,
    label: String,
    bms_name: String,
    bms: Vec<u8>,
    placement: FcstdPlacement,
    color: Option<[f32; 4]>,
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Content-addressed UUID (version-4 style bits, deterministic — no clock/RNG).
fn content_uid(entries: &[MeshEntry], meta: &FcstdDocumentMeta) -> String {
    let mut h = fnv1a64(meta.label.as_bytes());
    h ^= fnv1a64(meta.comment.as_bytes());
    for e in entries {
        h ^= fnv1a64(e.name.as_bytes());
        h ^= fnv1a64(e.label.as_bytes());
        h ^= fnv1a64(&e.bms);
        for &f in e
            .placement
            .position
            .iter()
            .chain(e.placement.rotation.iter())
        {
            h ^= fnv1a64(&f.to_le_bytes());
        }
        if let Some(c) = e.color {
            for &f in &c {
                h ^= fnv1a64(&f.to_le_bytes());
            }
        }
    }
    let a = (h >> 32) as u32;
    let b = ((h >> 16) & 0xffff) as u16;
    let c = (h & 0x0fff) | 0x4000; // version 4
    let d = (((h >> 48) & 0x3fff) | 0x8000) as u16; // variant 10
    let e = h & 0xffff_ffff_ffff;
    format!("{a:08x}-{b:04x}-{c:04x}-{d:04x}-{e:012x}")
}

fn build_document(meta: &FcstdDocumentMeta, meshes: &[MeshEntry]) -> String {
    let n = meshes.len();
    let uid = if meta.uid.is_empty() || meta.uid == "auto" {
        content_uid(meshes, meta)
    } else {
        meta.uid.clone()
    };
    let mut xml = String::from(
        "<?xml version='1.0' encoding='utf-8'?>\n\
         <Document SchemaVersion=\"4\" ProgramVersion=\"threers\" FileVersion=\"1\">\n\
         <Properties Count=\"6\">\n",
    );
    xml.push_str(&format!(
        "<Property name=\"Comment\" type=\"App::PropertyString\"><String value=\"{}\"/></Property>\n\
         <Property name=\"Company\" type=\"App::PropertyString\"><String value=\"{}\"/></Property>\n\
         <Property name=\"CreatedBy\" type=\"App::PropertyString\"><String value=\"{}\"/></Property>\n\
         <Property name=\"Label\" type=\"App::PropertyString\"><String value=\"{}\"/></Property>\n\
         <Property name=\"Uid\" type=\"App::PropertyUUID\"><Uuid value=\"{}\"/></Property>\n\
         <Property name=\"ShowHidden\" type=\"App::PropertyBool\"><Bool value=\"false\"/></Property>\n\
         </Properties>\n",
        xml_escape(&meta.comment),
        xml_escape(&meta.company),
        xml_escape(&meta.created_by),
        xml_escape(&meta.label),
        xml_escape(&uid),
    ));

    xml.push_str(&format!("<Objects Count=\"{n}\">\n"));
    for (i, m) in meshes.iter().enumerate() {
        xml.push_str(&format!(
            "<Object type=\"Mesh::Feature\" name=\"{}\" id=\"{}\" />\n",
            xml_escape(&m.name),
            i + 1
        ));
    }
    xml.push_str("</Objects>\n");

    xml.push_str(&format!("<ObjectData Count=\"{n}\">\n"));
    for m in meshes {
        let p = m.placement;
        let prop_count = 4 + usize::from(m.color.is_some());
        xml.push_str(&format!(
            "<Object name=\"{}\">\n\
             <Properties Count=\"{prop_count}\">\n\
             <Property name=\"Label\" type=\"App::PropertyString\"><String value=\"{}\"/></Property>\n\
             <Property name=\"Mesh\" type=\"Mesh::PropertyMeshKernel\"><Mesh file=\"{}\"/></Property>\n\
             <Property name=\"Placement\" type=\"App::PropertyPlacement\">\
             <PropertyPlacement Px=\"{}\" Py=\"{}\" Pz=\"{}\" Q0=\"{}\" Q1=\"{}\" Q2=\"{}\" Q3=\"{}\"/></Property>\n\
             <Property name=\"Visibility\" type=\"App::PropertyBool\"><Bool value=\"true\"/></Property>\n",
            xml_escape(&m.name),
            xml_escape(&m.label),
            xml_escape(&m.bms_name),
            p.position[0],
            p.position[1],
            p.position[2],
            p.rotation[0],
            p.rotation[1],
            p.rotation[2],
            p.rotation[3],
        ));
        if let Some(c) = m.color {
            xml.push_str(&format!(
                "<Property name=\"DiffuseColor\" type=\"App::PropertyColor\">\
                 <PropertyColor r=\"{}\" g=\"{}\" b=\"{}\" a=\"{}\"/></Property>\n",
                c[0], c[1], c[2], c[3]
            ));
        }
        xml.push_str("</Properties>\n</Object>\n");
    }
    xml.push_str("</ObjectData>\n</Document>\n");
    xml
}

/// FreeCAD `.FCStd` writer.
pub struct FcstdWriter;

impl FcstdWriter {
    /// Write a multi-object document. Returns bytes + a count report.
    pub fn write(
        meshes: &[FcstdMeshSpec],
        opts: &FcstdWriteOptions,
    ) -> Result<(Vec<u8>, FcstdWriteReport), FcstdError> {
        if meshes.is_empty() {
            return Err(FcstdError::Empty);
        }
        let mut entries = Vec::with_capacity(meshes.len());
        let mut report = FcstdWriteReport {
            neighbours: opts.neighbours,
            ..Default::default()
        };
        let mut used_names = HashMap::<String, usize>::new();

        for (i, spec) in meshes.iter().enumerate() {
            let (verts, indices) = tri_data(&spec.geometry).ok_or_else(|| {
                FcstdError::EmptyGeometry {
                    name: spec.name.clone(),
                }
            })?;
            let mut name = sanitize_name(&spec.name, &format!("Mesh{i}"));
            let n = used_names.entry(name.clone()).or_insert(0);
            if *n > 0 {
                name = format!("{name}_{n}");
            }
            *n += 1;
            let label = spec.label.clone().unwrap_or_else(|| name.clone());
            let bms_name = if i == 0 {
                "MeshKernel.bms".to_string()
            } else {
                format!("MeshKernel{i}.bms")
            };
            let bms = mesh_kernel_bms(&verts, &indices, opts.neighbours, opts.weld);
            // Report post-weld counts when welding ran.
            let (rp, rf) = if opts.neighbours || opts.weld {
                let (v, idx) = weld_mesh(&verts, &indices);
                (v.len(), idx.len() / 3)
            } else {
                (verts.len(), indices.len() / 3)
            };
            report.points += rp;
            report.facets += rf;
            entries.push(MeshEntry {
                name,
                label,
                bms,
                bms_name,
                placement: spec.placement,
                color: spec.color,
            });
        }
        report.objects = entries.len();

        let mut meta = opts.meta.clone();
        if meta.label.is_empty() {
            meta.label = "Unnamed".into();
        }
        let doc = build_document(&meta, &entries);
        let mut zip = Zip::new();
        zip.add("Document.xml", doc.as_bytes());
        for m in &entries {
            zip.add(&m.bms_name, &m.bms);
        }
        Ok((zip.finish(), report))
    }

    /// Single-mesh convenience.
    pub fn write_geometry(
        g: &BufferGeometry,
        name: &str,
        opts: &FcstdWriteOptions,
    ) -> Result<(Vec<u8>, FcstdWriteReport), FcstdError> {
        Self::write(&[FcstdMeshSpec::new(name, g.clone())], opts)
    }

    /// Write to a filesystem path.
    pub fn save(
        path: impl AsRef<std::path::Path>,
        meshes: &[FcstdMeshSpec],
        opts: &FcstdWriteOptions,
    ) -> Result<FcstdWriteReport, FcstdError> {
        let (bytes, report) = Self::write(meshes, opts)?;
        if let Some(parent) = path.as_ref().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path.as_ref(), bytes)?;
        Ok(report)
    }
}

// --- convenience wrappers (stable shorthand) --------------------------------

/// Load every mesh (placements applied). Empty archive → empty vec.
pub fn fcstd_meshes(bytes: &[u8]) -> Vec<FcstdMesh> {
    FcstdLoader::parse(bytes)
        .map(|d| d.meshes)
        .unwrap_or_default()
}

/// Merge all meshes in an `.FCStd` into one [`BufferGeometry`].
pub fn fcstd_to_geometry(bytes: &[u8]) -> Option<BufferGeometry> {
    FcstdLoader::parse(bytes).ok()?.to_geometry()
}

/// Parse an `.FCStd` document into a [`Solid`] (merged mesh), for `import()`.
pub fn fcstd_to_solid(bytes: &[u8]) -> Option<Solid> {
    FcstdLoader::parse(bytes).ok()?.to_solid()
}

/// Export as a FreeCAD **`.FCStd`** document (ZIP bytes) with one `Mesh::Feature`.
pub fn geometry_to_fcstd(g: &BufferGeometry, object_name: &str) -> Vec<u8> {
    let mut opts = FcstdWriteOptions::default();
    opts.meta.label = object_name.into();
    FcstdWriter::write_geometry(g, object_name, &opts)
        .map(|(b, _)| b)
        .unwrap_or_default()
}

/// Export with explicit options; returns an error instead of an empty archive.
pub fn geometry_to_fcstd_with(
    g: &BufferGeometry,
    object_name: &str,
    opts: &FcstdWriteOptions,
) -> Result<(Vec<u8>, FcstdWriteReport), FcstdError> {
    let mut o = opts.clone();
    if o.meta.label.is_empty() || o.meta.label == "Unnamed" {
        o.meta.label = object_name.into();
    }
    FcstdWriter::write_geometry(g, object_name, &o)
}

/// Export colored OpenSCAD [`ScadPart`]s as a multi-mesh FreeCAD **`.FCStd`**.
pub fn parts_to_fcstd(parts: &[ScadPart], doc_label: &str) -> Vec<u8> {
    parts_to_fcstd_with(parts, doc_label, &FcstdWriteOptions::default())
        .map(|(b, _)| b)
        .unwrap_or_default()
}

/// [`parts_to_fcstd`] with options.
pub fn parts_to_fcstd_with(
    parts: &[ScadPart],
    doc_label: &str,
    opts: &FcstdWriteOptions,
) -> Result<(Vec<u8>, FcstdWriteReport), FcstdError> {
    let specs = FcstdMeshSpec::from_parts(parts);
    let mut o = opts.clone();
    o.meta.label = doc_label.into();
    FcstdWriter::write(&specs, &o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cone, cube, cylinder, linear_extrude, sphere, sphere_fn};
    use std::f32::consts::{FRAC_PI_2, PI};

    fn find_zip_payload(zip: &[u8], name: &str) -> Option<Vec<u8>> {
        zip_read_entry(zip, Some(name), None)
    }

    fn signed_volume(g: &BufferGeometry) -> f64 {
        let pos = &g.get_attribute("position").unwrap().array;
        let read = |i: u32| {
            let j = i as usize * 3;
            [pos[j] as f64, pos[j + 1] as f64, pos[j + 2] as f64]
        };
        let mut vol = 0.0;
        let mut tri = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| {
            vol += a[0] * (b[1] * c[2] - b[2] * c[1])
                + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        };
        if let Some(idx) = &g.index {
            for t in idx.chunks_exact(3) {
                tri(read(t[0]), read(t[1]), read(t[2]));
            }
        } else {
            let n = pos.len() / 9;
            for k in 0..n {
                let b = (k * 3) as u32;
                tri(read(b), read(b + 1), read(b + 2));
            }
        }
        vol / 6.0
    }

    fn roundtrip_volume(solid: Solid, rel_tol: f64) {
        let g = solid.to_geometry_exact();
        let v0 = signed_volume(&g).abs();
        let bytes = geometry_to_fcstd(&g, "Part");
        let doc = FcstdLoader::parse(&bytes).expect("parse");
        assert!(doc.is_complete());
        let back = doc.to_geometry().expect("geom");
        let v1 = signed_volume(&back).abs();
        let scale = v0.max(1e-6);
        assert!(
            (v0 - v1).abs() / scale < rel_tol,
            "volume {v0} → {v1} (rel {})",
            (v0 - v1).abs() / scale
        );
    }

    #[test]
    fn fcstd_is_zip_with_document_and_bms() {
        let g = cube([10.0, 10.0, 10.0]).to_geometry_exact();
        let bytes = geometry_to_fcstd(&g, "Cube");
        assert!(bytes.starts_with(b"PK\x03\x04"), "not a zip");
        assert!(bytes.windows(4).any(|w| w == b"PK\x05\x06"), "no EOCD");

        let doc = find_zip_payload(&bytes, "Document.xml").expect("Document.xml");
        let text = String::from_utf8(doc).unwrap();
        assert!(text.contains("Mesh::Feature"));
        assert!(text.contains("MeshKernel.bms"));
        assert!(text.contains("name=\"Cube\""));

        let bms = find_zip_payload(&bytes, "MeshKernel.bms").expect("MeshKernel.bms");
        let magic = u32::from_le_bytes(bms[0..4].try_into().unwrap());
        let version = u32::from_le_bytes(bms[4..8].try_into().unwrap());
        assert_eq!(magic, 0xA0B0_C0D0);
        assert_eq!(version, 0x0001_0000);

        let (verts, indices) = tri_data(&g).unwrap();
        let n_pts = u32::from_le_bytes(bms[8 + 256..8 + 256 + 4].try_into().unwrap());
        let n_fts = u32::from_le_bytes(bms[8 + 256 + 4..8 + 256 + 8].try_into().unwrap());
        assert_eq!(n_pts as usize, verts.len());
        assert_eq!(n_fts as usize, indices.len() / 3);
    }

    #[test]
    fn parts_to_fcstd_emits_one_bms_per_part() {
        let a = cube([4.0, 4.0, 4.0]).color([1.0, 0.0, 0.0]);
        let b = cube([2.0, 2.0, 2.0])
            .color([0.0, 0.0, 1.0])
            .translate([10.0, 0.0, 0.0]);
        let parts = a.union(b).parts();
        assert!(parts.len() >= 2, "expected colored parts");
        let bytes = parts_to_fcstd(&parts, "TwoCubes");
        let doc = String::from_utf8(find_zip_payload(&bytes, "Document.xml").unwrap()).unwrap();
        let n = parts.len();
        assert!(doc.contains(&format!("Count=\"{n}\"")));
        assert!(find_zip_payload(&bytes, "MeshKernel.bms").is_some());
        if n > 1 {
            assert!(find_zip_payload(&bytes, "MeshKernel1.bms").is_some());
        }
    }

    #[test]
    fn fcstd_roundtrips_through_import() {
        let g = cube([10.0, 10.0, 10.0]).to_geometry_exact();
        let (verts, indices) = tri_data(&g).unwrap();
        let bytes = geometry_to_fcstd(&g, "Cube");

        let meshes = fcstd_meshes(&bytes);
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].name, "Cube");
        let (back_v, back_i) = tri_data(&meshes[0].geometry).unwrap();
        assert_eq!(back_v.len(), verts.len());
        assert_eq!(back_i.len(), indices.len());
        for (a, b) in verts.iter().zip(back_v.iter()) {
            assert!((a[0] - b[0]).abs() < 1e-5);
            assert!((a[1] - b[1]).abs() < 1e-5);
            assert!((a[2] - b[2]).abs() < 1e-5);
        }

        let solid = fcstd_to_solid(&bytes).expect("solid");
        let sg = solid.to_geometry_exact();
        let tris = sg.index.as_ref().map(|i| i.len() / 3).unwrap_or_else(|| {
            sg.get_attribute("position").map(|a| a.count() / 3).unwrap_or(0)
        });
        assert_eq!(tris, indices.len() / 3);
    }

    #[test]
    fn multi_part_fcstd_imports_each_mesh() {
        let a = cube([4.0, 4.0, 4.0]).color([1.0, 0.0, 0.0]);
        let b = cube([2.0, 2.0, 2.0])
            .color([0.0, 0.0, 1.0])
            .translate([10.0, 0.0, 0.0]);
        let parts = a.union(b).parts();
        let bytes = parts_to_fcstd(&parts, "Two");
        let meshes = fcstd_meshes(&bytes);
        assert_eq!(meshes.len(), parts.len());
    }

    #[test]
    fn loader_rejects_garbage() {
        assert!(matches!(
            FcstdLoader::parse(b"not a zip"),
            Err(FcstdError::NotZip)
        ));
        assert!(matches!(
            FcstdLoader::parse(b"PK\x03\x04"),
            Err(FcstdError::NotZip)
        ));
    }

    #[test]
    fn writer_report_counts_match_geometry() {
        let g = cube([5.0, 6.0, 7.0]).to_geometry_exact();
        let (verts, indices) = tri_data(&g).unwrap();
        let (bytes, report) =
            geometry_to_fcstd_with(&g, "Box", &FcstdWriteOptions::default()).unwrap();
        assert_eq!(report.objects, 1);
        assert_eq!(report.points, verts.len());
        assert_eq!(report.facets, indices.len() / 3);
        let doc = FcstdLoader::parse(&bytes).unwrap();
        assert_eq!(doc.total_facets(), report.facets);
        assert_eq!(doc.label(), "Box");
    }

    #[test]
    fn neighbours_roundtrip_on_closed_cube() {
        let g = cube([2.0, 2.0, 2.0]).to_geometry_exact();
        let opts = FcstdWriteOptions {
            neighbours: true,
            ..Default::default()
        };
        let (bytes, report) = geometry_to_fcstd_with(&g, "Cube", &opts).unwrap();
        assert!(report.neighbours);
        let bms = find_zip_payload(&bytes, "MeshKernel.bms").unwrap();
        let (_, _, neigh) = parse_mesh_kernel_full(&bms).unwrap();
        // A watertight cube mesh should have no open edges after neighbour fill.
        let open = neigh
            .iter()
            .flat_map(|n| n.iter())
            .filter(|&&x| x == 0xFFFF_FFFF)
            .count();
        assert_eq!(open, 0, "cube facets should all have three neighbours");
    }

    #[test]
    fn placement_translate_roundtrips() {
        let g = cube([2.0, 2.0, 2.0]).to_geometry_exact();
        let spec = FcstdMeshSpec::new("Moved", g).with_placement(FcstdPlacement::translate([
            10.0, -3.0, 5.0,
        ]));
        let (bytes, _) = FcstdWriter::write(&[spec], &FcstdWriteOptions::default()).unwrap();
        let doc = FcstdLoader::parse(&bytes).unwrap();
        let m = &doc.meshes[0];
        assert!((m.placement.position[0] - 10.0).abs() < 1e-5);
        let (lo, hi) = crate::geometry_bounds(&m.geometry);
        assert!((lo[0] - 9.0).abs() < 1e-4 && (hi[0] - 11.0).abs() < 1e-4);
    }

    #[test]
    fn placement_rotation_around_z() {
        // Non-cubic box so a 90° Z rotation swaps X/Y extents.
        let g = cube([4.0, 2.0, 2.0]).to_geometry_exact();
        let place = FcstdPlacement::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2, [0.0; 3]);
        let spec = FcstdMeshSpec::new("Spun", g).with_placement(place);
        let (bytes, _) = FcstdWriter::write(&[spec], &FcstdWriteOptions::default()).unwrap();

        let opts = FcstdReadOptions {
            apply_placement: false,
            ..Default::default()
        };
        let raw = FcstdLoader::parse_with(&bytes, &opts).unwrap();
        let (lo0, hi0) = crate::geometry_bounds(&raw.meshes[0].geometry);
        assert!((hi0[0] - lo0[0] - 4.0).abs() < 1e-3, "unbaked X extent");
        assert!((hi0[1] - lo0[1] - 2.0).abs() < 1e-3, "unbaked Y extent");

        let baked = FcstdLoader::parse(&bytes).unwrap();
        let (lo, hi) = crate::geometry_bounds(&baked.meshes[0].geometry);
        assert!((hi[0] - lo[0] - 2.0).abs() < 1e-3, "baked X extent {}", hi[0] - lo[0]);
        assert!((hi[1] - lo[1] - 4.0).abs() < 1e-3, "baked Y extent {}", hi[1] - lo[1]);
    }

    #[test]
    fn volume_roundtrip_cube_sphere_cylinder_cone() {
        roundtrip_volume(cube([10.0, 10.0, 10.0]), 1e-5);
        roundtrip_volume(sphere_fn(5.0, 24), 1e-4);
        roundtrip_volume(cylinder(8.0, 3.0), 1e-4);
        roundtrip_volume(cone(10.0, 4.0, 0.0), 1e-3);
    }

    #[test]
    fn volume_roundtrip_extrude_and_csg() {
        let outline = [[-5.0, -3.0], [5.0, -3.0], [5.0, 3.0], [-5.0, 3.0]];
        roundtrip_volume(linear_extrude(4.0, &outline), 1e-5);

        let bracket = cube([40.0, 24.0, 4.0])
            .difference(
                cylinder(20.0, 2.5)
                    .rotate_x(FRAC_PI_2)
                    .translate([-12.0, 0.0, 0.0]),
            )
            .difference(
                cylinder(20.0, 2.5)
                    .rotate_x(FRAC_PI_2)
                    .translate([12.0, 0.0, 0.0]),
            )
            .union(sphere(5.0).translate([0.0, 0.0, 2.0]));
        roundtrip_volume(bracket, 2e-3);
    }

    #[test]
    fn e2e_multipart_assembly_with_placements() {
        let deck = cube([40.0, 20.0, 4.0]).to_geometry_exact();
        let pinion = cylinder(6.0, 3.0)
            .rotate_x(FRAC_PI_2)
            .to_geometry_exact();
        let wheel = cylinder(4.0, 8.0)
            .rotate_x(FRAC_PI_2)
            .to_geometry_exact();

        let specs = [
            FcstdMeshSpec::new("Deck", deck).with_label("Base plate"),
            FcstdMeshSpec::new("Pinion", pinion)
                .with_placement(FcstdPlacement::translate([-10.0, 0.0, 5.0])),
            FcstdMeshSpec::new("Wheel", wheel)
                .with_placement(FcstdPlacement::translate([12.0, 0.0, 5.0])),
        ];
        let mut opts = FcstdWriteOptions::default();
        opts.meta.label = "GearPair".into();
        opts.meta.comment = "e2e assembly".into();
        opts.neighbours = true;
        let (bytes, report) = FcstdWriter::write(&specs, &opts).unwrap();
        assert_eq!(report.objects, 3);

        let doc = FcstdLoader::parse(&bytes).unwrap();
        assert_eq!(doc.label(), "GearPair");
        assert_eq!(doc.mesh_count(), 3);
        assert!(doc.skipped.is_empty());
        assert_eq!(doc.meshes[1].label, "Pinion");
        assert!((doc.meshes[1].placement.position[0] + 10.0).abs() < 1e-5);

        // Merged solid still has positive volume.
        let solid = doc.to_solid().unwrap();
        let v = signed_volume(&solid.to_geometry_exact()).abs();
        assert!(v > 100.0, "assembly volume {v}");
    }

    #[test]
    fn e2e_scad_import_roundtrip_file() {
        let dir = std::env::temp_dir().join("threers_fcstd_e2e");
        std::fs::create_dir_all(&dir).unwrap();
        let solid = cube([12.0, 8.0, 6.0]).difference(sphere(2.5));
        let g = solid.clone().to_geometry_exact();
        let v0 = signed_volume(&g).abs();
        let bytes = solid.to_fcstd();
        let path = dir.join("part.FCStd");
        std::fs::write(&path, &bytes).unwrap();
        std::fs::write(dir.join("part.scad"), "import(\"part.FCStd\");").unwrap();
        let back = crate::parse_scad_file(dir.join("part.scad"))
            .unwrap()
            .to_geometry_exact();
        let v1 = signed_volume(&back).abs();
        assert!(
            (v0 - v1).abs() / v0.max(1.0) < 0.02,
            "scad import vol {v0} → {v1}"
        );
    }

    #[test]
    fn missing_sidecar_is_reported_not_silent() {
        // Build a valid doc, then strip one .bms from the zip by rewriting
        // Document.xml alone into a new archive.
        let g = cube([3.0, 3.0, 3.0]).to_geometry_exact();
        let bytes = geometry_to_fcstd(&g, "Only");
        let doc_xml = find_zip_payload(&bytes, "Document.xml").unwrap();
        // Corrupt: Document.xml references MeshKernel.bms but we don't include it.
        let mut zip = Zip::new();
        zip.add("Document.xml", &doc_xml);
        let broken = zip.finish();
        let err = FcstdLoader::parse(&broken).unwrap_err();
        assert_eq!(err, FcstdError::Empty);

        // With a second good mesh plus a missing one: report skip.
        let a = cube([2.0, 2.0, 2.0]).to_geometry_exact();
        let b = cube([1.0, 1.0, 1.0]).to_geometry_exact();
        let (good, _) = FcstdWriter::write(
            &[
                FcstdMeshSpec::new("A", a),
                FcstdMeshSpec::new("B", b),
            ],
            &FcstdWriteOptions::default(),
        )
        .unwrap();
        let xml = String::from_utf8(find_zip_payload(&good, "Document.xml").unwrap()).unwrap();
        let a_bms = find_zip_payload(&good, "MeshKernel.bms").unwrap();
        // Drop MeshKernel1.bms
        let mut zip = Zip::new();
        zip.add("Document.xml", xml.as_bytes());
        zip.add("MeshKernel.bms", &a_bms);
        let partial = zip.finish();
        let doc = FcstdLoader::parse(&partial).unwrap();
        assert_eq!(doc.mesh_count(), 1);
        assert_eq!(doc.skipped.len(), 1);
        assert!(!doc.is_complete());
        match &doc.skipped[0] {
            FcstdSkipped::MissingFile { object, file } => {
                assert_eq!(object, "B");
                assert!(file.contains("MeshKernel"));
            }
            other => panic!("unexpected skip {other:?}"),
        }
    }

    #[test]
    fn solid_to_fcstd_parts_path() {
        let solid = cube([10.0, 10.0, 4.0])
            .color_named("steelblue")
            .union(sphere(3.0).translate([0.0, 0.0, 4.0]).color_named("tomato"));
        let parts = solid.parts();
        let (bytes, report) =
            parts_to_fcstd_with(&parts, "Colored", &FcstdWriteOptions::default()).unwrap();
        assert_eq!(report.objects, parts.len());
        let doc = FcstdLoader::parse(&bytes).unwrap();
        assert_eq!(doc.mesh_count(), parts.len());
    }

    #[test]
    fn sanitize_and_duplicate_names() {
        let g = cube([1.0, 1.0, 1.0]).to_geometry_exact();
        let (bytes, _) = FcstdWriter::write(
            &[
                FcstdMeshSpec::new("1bad", g.clone()),
                FcstdMeshSpec::new("same", g.clone()),
                FcstdMeshSpec::new("same", g),
            ],
            &FcstdWriteOptions::default(),
        )
        .unwrap();
        let doc = FcstdLoader::parse(&bytes).unwrap();
        let names: Vec<_> = doc.meshes.iter().map(|m| m.name.as_str()).collect();
        assert!(names[0].starts_with("Mesh") || names[0].chars().next().unwrap().is_alphabetic());
        assert_eq!(names.len(), 3);
        assert_ne!(names[1], names[2]);
    }

    #[test]
    fn require_document_option() {
        // BMS-only archive (no Document.xml).
        let g = cube([2.0, 2.0, 2.0]).to_geometry_exact();
        let (verts, indices) = tri_data(&g).unwrap();
        let bms = mesh_kernel_bms(&verts, &indices, false, false);
        let mut zip = Zip::new();
        zip.add("orphan.bms", &bms);
        let bytes = zip.finish();

        let doc = FcstdLoader::parse(&bytes).unwrap();
        assert_eq!(doc.mesh_count(), 1);

        let opts = FcstdReadOptions {
            require_document: true,
            ..Default::default()
        };
        assert!(matches!(
            FcstdLoader::parse_with(&bytes, &opts),
            Err(FcstdError::MissingDocument)
        ));
    }

    #[test]
    fn gear_profile_extrude_roundtrip() {
        // Tiny spur outline (8 teeth trapezoid) — geometry-heavy E2E.
        let n = 8u32;
        let m = 2.0f32;
        let r = m * n as f32 / 2.0;
        let tip = r + m;
        let root = r - 1.25 * m;
        let p = 360.0 / n as f32;
        let mut outline = Vec::new();
        for i in 0..n {
            let i = i as f32;
            for (rad, ang) in [
                (root, (i - 0.25) * p),
                (tip, (i - 0.07) * p),
                (tip, (i + 0.07) * p),
                (root, (i + 0.25) * p),
            ] {
                let a = ang * PI / 180.0;
                outline.push([rad * a.cos(), rad * a.sin()]);
            }
        }
        roundtrip_volume(linear_extrude(5.0, &outline), 1e-4);
    }

    #[test]
    fn export_is_byte_deterministic() {
        let g = cube([3.0, 4.0, 5.0]).to_geometry_exact();
        let mut opts = FcstdWriteOptions::default();
        opts.meta.label = "Det".into();
        opts.meta.comment = "same every time".into();
        let (a, _) = FcstdWriter::write_geometry(&g, "Box", &opts).unwrap();
        let (b, _) = FcstdWriter::write_geometry(&g, "Box", &opts).unwrap();
        assert_eq!(a, b, "identical inputs must produce identical FCStd bytes");
    }

    #[test]
    fn content_uid_stable_and_meta_roundtrips() {
        let g = cube([2.0, 2.0, 2.0]).to_geometry_exact();
        let mut opts = FcstdWriteOptions::default();
        opts.meta.label = "UidDoc".into();
        opts.meta.company = "threers".into();
        opts.meta.comment = "meta check".into();
        opts.meta.uid = "auto".into();
        let (bytes, _) = FcstdWriter::write_geometry(&g, "Box", &opts).unwrap();
        let doc = FcstdLoader::parse(&bytes).unwrap();
        assert_eq!(doc.meta.label, "UidDoc");
        assert_eq!(doc.meta.company, "threers");
        assert_eq!(doc.meta.comment, "meta check");
        assert!(
            doc.meta.uid.len() == 36 && doc.meta.uid.contains('-'),
            "uid={}",
            doc.meta.uid
        );
        // Same content → same uid.
        let (bytes2, _) = FcstdWriter::write_geometry(&g, "Box", &opts).unwrap();
        let doc2 = FcstdLoader::parse(&bytes2).unwrap();
        assert_eq!(doc.meta.uid, doc2.meta.uid);
    }

    #[test]
    fn color_roundtrips_on_mesh() {
        let g = cube([1.0, 1.0, 1.0]).to_geometry_exact();
        let spec = FcstdMeshSpec::new("Red", g).with_color([0.8, 0.1, 0.05, 1.0]);
        let (bytes, _) = FcstdWriter::write(&[spec], &FcstdWriteOptions::default()).unwrap();
        let doc = FcstdLoader::parse(&bytes).unwrap();
        let c = doc.meshes[0].color.expect("color");
        assert!((c[0] - 0.8).abs() < 1e-5);
        assert!((c[1] - 0.1).abs() < 1e-5);
        assert!((c[2] - 0.05).abs() < 1e-5);
    }

    #[test]
    fn document_lookup_and_bounds() {
        let a = cube([2.0, 2.0, 2.0]).to_geometry_exact();
        let b = cube([1.0, 1.0, 1.0]).to_geometry_exact();
        let specs = [
            FcstdMeshSpec::new("Alpha", a).with_placement(FcstdPlacement::translate([10.0, 0.0, 0.0])),
            FcstdMeshSpec::new("Beta", b).with_label("Secondary"),
        ];
        let (bytes, _) = FcstdWriter::write(&specs, &FcstdWriteOptions::default()).unwrap();
        let doc = FcstdLoader::parse(&bytes).unwrap();
        assert!(doc.mesh("Alpha").is_some());
        assert!(doc.mesh_by_label("Secondary").is_some());
        let (lo, hi) = doc.bounds().unwrap();
        assert!(lo[0] < 9.5 && hi[0] > 10.5);
    }

    #[test]
    fn document_reexport_preserves_volume() {
        let solid = cube([8.0, 6.0, 4.0]).difference(sphere(2.0));
        let g = solid.to_geometry_exact();
        let v0 = signed_volume(&g).abs();
        let bytes = geometry_to_fcstd(&g, "Cut");
        let doc = FcstdLoader::parse(&bytes).unwrap();
        let (bytes2, _) = doc.to_fcstd(&FcstdWriteOptions::default()).unwrap();
        let doc2 = FcstdLoader::parse(&bytes2).unwrap();
        let v1 = signed_volume(&doc2.to_geometry().unwrap()).abs();
        assert!((v0 - v1).abs() / v0.max(1.0) < 1e-4);
    }

    #[test]
    fn path_roundtrip_save_load() {
        let dir = std::env::temp_dir().join("threers_fcstd_path");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("box.FCStd");
        let g = cube([5.0, 5.0, 5.0]).to_geometry_exact();
        let mut opts = FcstdWriteOptions::default();
        opts.meta.label = "PathBox".into();
        FcstdWriter::save(&path, &[FcstdMeshSpec::new("Box", g)], &opts).unwrap();
        let doc = FcstdLoader::load(&path).unwrap();
        assert_eq!(doc.label(), "PathBox");
        assert_eq!(doc.mesh_count(), 1);
    }

    #[test]
    fn parts_from_parts_carry_colors() {
        let a = cube([3.0, 3.0, 3.0]).color([1.0, 0.0, 0.0]);
        let b = cube([2.0, 2.0, 2.0])
            .color([0.0, 1.0, 0.0])
            .translate([6.0, 0.0, 0.0]);
        let parts = a.union(b).parts();
        let (bytes, _) =
            parts_to_fcstd_with(&parts, "RGB", &FcstdWriteOptions::default()).unwrap();
        let doc = FcstdLoader::parse(&bytes).unwrap();
        assert_eq!(doc.mesh_count(), parts.len());
        assert!(doc.meshes.iter().any(|m| m.color.is_some()));
    }
}
