//! Discrete cuboct voxels — six face parts, inner and intervoxel joints.
//!
//! The continuum [`Lattice`](super::Lattice) is the bulk material. This is the
//! construction system: injection-molded (or cut) **faces** assemble into a
//! voxel, voxels join face-to-face. Joints are designed stiffer than beams, so
//! the assembled lattice behaves like the beam network.
//!
//! ```
//! use threers::{Cuboct, CuboctAssembly, CuboctJointKind};
//!
//! let asm = CuboctAssembly::new(Cuboct::Rigid)
//!     .pitch(20.0)
//!     .cells([2, 2, 2])
//!     .explode(0.25);
//! assert_eq!(asm.parts().len(), 8 * 6);
//! assert!(asm.joints().iter().any(|j| j.kind == CuboctJointKind::Inner));
//! assert!(asm.joints().iter().any(|j| j.kind == CuboctJointKind::Inter));
//! ```

use super::cuboct::{uv_beams, FACE_NODES_UV};
use super::mesher::IsoGrid;
use super::strut::segment_distance;
use super::{ChiralRule, Cuboct, Hand};
use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;
use crate::utils::merge_geometries;
use std::collections::HashMap;

/// A fastener between two face parts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CuboctJoint {
    /// Inner (same voxel) or inter (neighbouring voxels).
    pub kind: CuboctJointKind,
    /// Fastener location in world units.
    pub at: Vector3,
    /// The two parts, as `(voxel_flat_index, face_id)` with face_id in `0..6`.
    pub parts: [(usize, usize); 2],
}

/// Where a joint sits in the assembly hierarchy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CuboctJointKind {
    /// Two faces of the same voxel, at a cube-edge midpoint.
    Inner,
    /// Two voxels sharing a cube face, at a vertex of that face.
    Inter,
}

/// Six molded (or cut) faces per voxel, tiled and optionally exploded.
/// Decides what occupies each cell of the lattice, by integer cell coordinate.
pub type CuboctProgram<'a> = Box<dyn Fn(i32, i32, i32) -> Cuboct + Sync + Send + 'a>;

pub struct CuboctAssembly<'a> {
    kind: Cuboct,
    pitch: f32,
    cells: [usize; 3],
    /// In-plane beam width.
    beam: f32,
    /// Plate thickness, measured inward from the cube face.
    plate: f32,
    /// Fastener pad radius in the face plane.
    pad: f32,
    /// Pull each face out along its outward normal, in units of pitch.
    explode: f32,
    shape: f32,
    resolution: usize,
    program: Option<CuboctProgram<'a>>,
    chiral_rule: Option<ChiralRule>,
    /// Draft angle (degrees) for mold-ready faces.
    draft_deg: f32,
    /// Blind rivet shank diameter (paper: 3/32 in ≈ 2.38 mm).
    rivet_diameter: f32,
    /// Cut rivet holes and apply draft taper.
    mold_ready: bool,
}

impl<'a> CuboctAssembly<'a> {
    /// A single rigid voxel of pitch 1.
    pub fn new(kind: Cuboct) -> Self {
        let p = 1.0f32;
        Self {
            kind,
            pitch: p,
            cells: [1, 1, 1],
            beam: 0.08 * p,
            plate: 0.045 * p,
            pad: 0.11 * p,
            explode: 0.0,
            shape: kind.default_shape(),
            resolution: 36,
            program: None,
            chiral_rule: None,
            draft_deg: 3.0,
            rivet_diameter: 2.38125,
            mold_ready: false,
        }
    }

    pub fn grid_cells(&self) -> [usize; 3] {
        self.cells
    }

    pub fn assembly_pitch(&self) -> f32 {
        self.pitch
    }

    pub(crate) fn voxel_flat(&self, ix: i32, iy: i32, iz: i32) -> usize {
        let [nx, ny, _] = self.cells;
        (iz as usize * ny + iy as usize) * nx + ix as usize
    }

    pub(crate) fn voxel_origin(&self, ix: i32, iy: i32, iz: i32) -> Vector3 {
        let [nx, ny, nz] = self.cells;
        let p = self.pitch;
        Vector3::new(
            (ix as f32 - (nx as f32) * 0.5) * p,
            (iy as f32 - (ny as f32) * 0.5) * p,
            (iz as f32 - (nz as f32) * 0.5) * p,
        )
    }

    pub(crate) fn kind_at(&self, ix: i32, iy: i32, iz: i32) -> Cuboct {
        match &self.program {
            Some(f) => f(ix, iy, iz),
            None => self.kind,
        }
    }

    /// Cube side length of one voxel.
    pub fn pitch(mut self, pitch: f32) -> Self {
        let s = pitch.abs().max(1e-4);
        let ratio = s / self.pitch;
        self.beam *= ratio;
        self.plate *= ratio;
        self.pad *= ratio;
        self.pitch = s;
        self
    }

    /// Voxel count along each axis.
    pub fn cells(mut self, cells: [usize; 3]) -> Self {
        self.cells = [cells[0].max(1), cells[1].max(1), cells[2].max(1)];
        self
    }

    /// In-plane beam width. Independent of [`pitch`](Self::pitch) once set.
    pub fn beam(mut self, beam: f32) -> Self {
        self.beam = beam.abs().max(1e-5);
        self
    }

    /// Face-plate thickness, inward from the cube face.
    pub fn plate(mut self, plate: f32) -> Self {
        self.plate = plate.abs().max(1e-5);
        self
    }

    /// Node-pad radius in the face plane.
    pub fn pad(mut self, pad: f32) -> Self {
        self.pad = pad.abs().max(1e-5);
        self
    }

    /// Pull faces apart along their outward normals, as a fraction of pitch.
    /// `0` is assembled; `0.25` is a readable exploded view.
    pub fn explode(mut self, explode: f32) -> Self {
        self.explode = explode.max(0.0);
        self
    }

    /// Cuboct beam path (`a/P`, `d/P`, or `r/P`). Same meaning as
    /// [`Lattice::shape`](super::Lattice::shape).
    pub fn shape(mut self, shape: f32) -> Self {
        self.shape = shape.max(0.0);
        self
    }

    /// Samples across a face. Cost is per part, cubic only in this 2.5D slab.
    pub fn resolution(mut self, samples: usize) -> Self {
        self.resolution = samples.max(8);
        self
    }

    /// Per-voxel part type, `(i, j, k)` in cell index.
    pub fn program(mut self, program: impl Fn(i32, i32, i32) -> Cuboct + Sync + Send + 'a) -> Self {
        self.program = Some(Box::new(program));
        self
    }

    /// Orient chiral faces by Jenett et al.'s column rules.
    pub fn chiral_rule(mut self, rule: ChiralRule) -> Self {
        self.chiral_rule = Some(rule);
        self
    }

    /// Enable mold-ready geometry: draft taper and rivet through-holes.
    pub fn mold_ready(mut self, on: bool) -> Self {
        self.mold_ready = on;
        self
    }

    /// Parting-line draft angle in degrees (typical injection molding: 2–5°).
    pub fn draft(mut self, degrees: f32) -> Self {
        self.draft_deg = degrees.abs().min(15.0);
        self
    }

    /// Fastener hole diameter (paper uses 3/32 in blind aluminum rivets).
    pub fn rivet_diameter(mut self, diameter: f32) -> Self {
        self.rivet_diameter = diameter.max(0.0);
        self
    }

    /// One mesh per face part, in world coordinates.
    pub fn parts(&self) -> Vec<BufferGeometry> {
        let mut cache: HashMap<(u8, i32, u8), BufferGeometry> = HashMap::new();
        let mut out = Vec::with_capacity(self.voxel_count() * 6);
        let n = self.cells[0] as i32;
        for iz in 0..self.cells[2] as i32 {
            for iy in 0..self.cells[1] as i32 {
                for ix in 0..self.cells[0] as i32 {
                    let kind = self.kind_at(ix, iy, iz);
                    let origin = self.voxel_origin(ix, iy, iz);
                    for face in 0..6 {
                        let (axis, low) = face_axis_low(face);
                        let hand = self.hand_at(kind, axis, ix, iy, iz, low, n);
                        let proto = cache
                            .entry(cache_key(kind, self.shape, hand))
                            .or_insert_with(|| {
                                mesh_face(
                                    kind,
                                    self.shape,
                                    hand,
                                    self.pitch,
                                    self.beam,
                                    self.plate,
                                    self.pad,
                                    self.resolution,
                                    self.draft_deg,
                                    self.rivet_diameter,
                                    self.mold_ready,
                                )
                            });
                        out.push(place_face(proto, origin, self.pitch, face, self.explode));
                    }
                }
            }
        }
        out
    }

    /// Every face part, merged into one mesh.
    pub fn build(&self) -> BufferGeometry {
        merge_geometries(&self.parts()).unwrap_or_default()
    }

    /// Like [`build`](Self::build), but each face gets a unique vertex color.
    pub fn build_colored(&self) -> BufferGeometry {
        let parts = self.parts();
        let n = parts.len().max(1);
        let tinted: Vec<BufferGeometry> = parts
            .into_iter()
            .enumerate()
            .map(|(i, mut g)| {
                tint_geometry(&mut g, part_rgb(i, n));
                g
            })
            .collect();
        merge_geometries(&tinted).unwrap_or_default()
    }

    /// Parts plus rivet markers at [`joints`](Self::joints).
    pub fn build_with_joints(&self) -> BufferGeometry {
        let mut geoms = self.parts();
        geoms.push(self.joint_geometry());
        merge_geometries(&geoms).unwrap_or_default()
    }

    /// Sphere markers at inner (large) and inter (small) joints.
    pub fn joint_geometry(&self) -> BufferGeometry {
        use crate::geometries::SphereGeometry;
        let mut geoms = Vec::new();
        for j in self.joints() {
            let r = match j.kind {
                CuboctJointKind::Inner => self.pad * 0.55,
                CuboctJointKind::Inter => self.pad * 0.38,
            };
            let mut s = SphereGeometry::new(r, 10, 8);
            translate_geometry(&mut s, j.at);
            geoms.push(s);
        }
        merge_geometries(&geoms).unwrap_or_default()
    }

    /// Write one binary STL per face part (`part_0000.stl`, …).
    pub fn export_stl_dir(&self, dir: &std::path::Path) -> std::io::Result<usize> {
        std::fs::create_dir_all(dir)?;
        for (i, geom) in self.parts().iter().enumerate() {
            let path = dir.join(format!("part_{i:04}.stl"));
            std::fs::write(path, geometry_to_stl_bytes(geom))?;
        }
        Ok(self.voxel_count() * 6)
    }

    /// Write one Wavefront OBJ per face part.
    pub fn export_obj_dir(&self, dir: &std::path::Path) -> std::io::Result<usize> {
        std::fs::create_dir_all(dir)?;
        for (i, geom) in self.parts().iter().enumerate() {
            let path = dir.join(format!("part_{i:04}.obj"));
            std::fs::write(path, geometry_to_obj_text(geom))?;
        }
        Ok(self.voxel_count() * 6)
    }

    /// OpenSCAD source: one translated module per face plus rivet cylinders.
    pub fn to_scad(&self) -> String {
        scad_assembly(self)
    }

    /// Mold tooling OpenSCAD: top/bottom halves with draft and holes.
    pub fn to_scad_mold(&self) -> String {
        scad_mold(self)
    }

    /// Gantry / robot assembly sequence.
    pub fn assembly_plan(&self) -> super::CuboctAssemblyPlan {
        super::CuboctAssemblyPlan::from_assembly(self)
    }

    /// Inner and intervoxel fasteners.
    pub fn joints(&self) -> Vec<CuboctJoint> {
        let mut joints = Vec::new();
        let [nx, ny, nz] = self.cells;
        let p = self.pitch;

        for iz in 0..nz as i32 {
            for iy in 0..ny as i32 {
                for ix in 0..nx as i32 {
                    let v = self.voxel_flat(ix, iy, iz);
                    let origin = self.voxel_origin(ix, iy, iz);
                    for (mid, faces) in INNER_EDGES {
                        joints.push(CuboctJoint {
                            kind: CuboctJointKind::Inner,
                            at: origin + Vector3::new(mid[0], mid[1], mid[2]) * p,
                            parts: [(v, faces[0]), (v, faces[1])],
                        });
                    }
                    if ix + 1 < nx as i32 {
                        let v2 = self.voxel_flat(ix + 1, iy, iz);
                        for node in FACE_NODES_UV {
                            joints.push(CuboctJoint {
                                kind: CuboctJointKind::Inter,
                                at: origin + Vector3::new(1.0, node[0], node[1]) * p,
                                parts: [(v, 1), (v2, 0)],
                            });
                        }
                    }
                    if iy + 1 < ny as i32 {
                        let v2 = self.voxel_flat(ix, iy + 1, iz);
                        for node in FACE_NODES_UV {
                            joints.push(CuboctJoint {
                                kind: CuboctJointKind::Inter,
                                at: origin + Vector3::new(node[0], 1.0, node[1]) * p,
                                parts: [(v, 3), (v2, 2)],
                            });
                        }
                    }
                    if iz + 1 < nz as i32 {
                        let v2 = self.voxel_flat(ix, iy, iz + 1);
                        for node in FACE_NODES_UV {
                            joints.push(CuboctJoint {
                                kind: CuboctJointKind::Inter,
                                at: origin + Vector3::new(node[0], node[1], 1.0) * p,
                                parts: [(v, 5), (v2, 4)],
                            });
                        }
                    }
                }
            }
        }
        joints
    }

    /// 2D cutting profile of one face, in millimetres (or whatever [`Self::pitch`] is).
    /// Suitable for laser or water-jet; tabs still have to be folded/assembled.
    pub fn svg(&self) -> String {
        face_svg(
            self.kind,
            self.shape,
            self.kind.hand_on_face(2, 0, 0, 0, None, 0),
            self.pitch,
            self.beam,
            self.pad,
        )
    }

    fn voxel_count(&self) -> usize {
        self.cells[0] * self.cells[1] * self.cells[2]
    }

    #[allow(clippy::too_many_arguments)]
    fn hand_at(
        &self,
        kind: Cuboct,
        axis: usize,
        ix: i32,
        iy: i32,
        iz: i32,
        low: bool,
        n: i32,
    ) -> Hand {
        let (i, j, k) = if low {
            (ix, iy, iz)
        } else {
            match axis {
                0 => (ix + 1, iy, iz),
                1 => (ix, iy + 1, iz),
                _ => (ix, iy, iz + 1),
            }
        };
        kind.hand_on_face(axis, i, j, k, self.chiral_rule, n)
    }
}

/// Face 0 = −X, 1 = +X, 2 = −Y, 3 = +Y, 4 = −Z, 5 = +Z.
pub(crate) fn face_axis_low(face: usize) -> (usize, bool) {
    (face / 2, face.is_multiple_of(2))
}

fn cache_key(kind: Cuboct, shape: f32, hand: Hand) -> (u8, i32, u8) {
    let k = match kind {
        Cuboct::Rigid => 0,
        Cuboct::Compliant => 1,
        Cuboct::Auxetic => 2,
        Cuboct::ChiralCw => 3,
        Cuboct::ChiralCcw => 4,
    };
    let h = match hand {
        Hand::Cw => 0,
        Hand::Ccw => 1,
    };
    (k, (shape * 1000.0).round() as i32, h)
}

/// Cube-edge midpoints and the two faces of this voxel that meet there.
/// Face ids: 0=−X 1=+X 2=−Y 3=+Y 4=−Z 5=+Z.
const INNER_EDGES: [([f32; 3], [usize; 2]); 12] = [
    ([0.5, 0.0, 0.0], [2, 4]),
    ([0.5, 1.0, 0.0], [3, 4]),
    ([0.5, 0.0, 1.0], [2, 5]),
    ([0.5, 1.0, 1.0], [3, 5]),
    ([0.0, 0.5, 0.0], [0, 4]),
    ([1.0, 0.5, 0.0], [1, 4]),
    ([0.0, 0.5, 1.0], [0, 5]),
    ([1.0, 0.5, 1.0], [1, 5]),
    ([0.0, 0.0, 0.5], [0, 2]),
    ([1.0, 0.0, 0.5], [1, 2]),
    ([0.0, 1.0, 0.5], [0, 3]),
    ([1.0, 1.0, 0.5], [1, 3]),
];

// The dimension set below — pitch, beam, plate, pad, draft, rivet, mold_ready —
// travels together through this whole family and would read better as one
// struct. Left flat for now: it is threaded through working geometry code, and
// the change is worth doing on its own rather than folded into a lint sweep.
#[allow(clippy::too_many_arguments)]
fn mesh_face(
    kind: Cuboct,
    shape: f32,
    hand: Hand,
    pitch: f32,
    beam: f32,
    plate: f32,
    pad: f32,
    resolution: usize,
    draft_deg: f32,
    rivet_diameter: f32,
    mold_ready: bool,
) -> BufferGeometry {
    let beams = uv_beams(kind, shape, hand);
    let tab = 0.1 * pitch;
    let half_b = beam * 0.5;
    let margin = pad.max(half_b) + pitch / resolution as f32 * 2.0;
    let nu = resolution.max(8);
    let nv = nu;
    let w_span = plate + tab + margin;
    let nw = ((w_span / pitch) * nu as f32).ceil() as usize + 4;
    let nw = nw.max(8);
    let origin = Vector3::new(-margin, -margin, -margin * 0.25);
    let step = Vector3::new(
        (pitch + 2.0 * margin) / (nu - 1) as f32,
        (pitch + 2.0 * margin) / (nv - 1) as f32,
        w_span / (nw - 1) as f32,
    );
    let mut values = vec![0.0f32; nu * nv * nw];
    let center = Vector3::new(0.5 * pitch, 0.5 * pitch, 0.0);
    for k in 0..nw {
        for j in 0..nv {
            for i in 0..nu {
                let p = Vector3::new(
                    origin.x + i as f32 * step.x,
                    origin.y + j as f32 * step.y,
                    origin.z + k as f32 * step.z,
                );
                values[(k * nv + j) * nu + i] = -face_field(
                    p,
                    &beams,
                    pitch,
                    half_b,
                    plate,
                    pad,
                    tab,
                    center,
                    draft_deg,
                    rivet_diameter,
                    mold_ready,
                );
            }
        }
    }
    IsoGrid {
        dims: [nu, nv, nw],
        origin,
        step,
        values,
    }
    .triangulate()
}

#[allow(clippy::too_many_arguments)]
fn face_field(
    p: Vector3,
    beams: &[[[f32; 2]; 2]],
    pitch: f32,
    half_b: f32,
    plate: f32,
    pad: f32,
    tab: f32,
    center: Vector3,
    draft_deg: f32,
    rivet_diameter: f32,
    mold_ready: bool,
) -> f32 {
    let draft_scale = if mold_ready && draft_deg > 0.0 {
        let t = (p.z / plate.max(1e-6)).clamp(0.0, 1.0);
        1.0 + draft_deg.to_radians().tan() * t
    } else {
        1.0
    };
    let half_eff = half_b * draft_scale;

    let mut d_beam = f32::MAX;
    for seg in beams {
        let a = Vector3::new(seg[0][0] * pitch, seg[0][1] * pitch, 0.0);
        let b = Vector3::new(seg[1][0] * pitch, seg[1][1] * pitch, 0.0);
        let q = Vector3::new(p.x, p.y, 0.0);
        d_beam = d_beam.min(segment_distance(q, a, b));
    }
    d_beam -= half_eff;
    let mut d_pad = f32::MAX;
    for node in FACE_NODES_UV {
        let n = Vector3::new(node[0] * pitch, node[1] * pitch, 0.0);
        let dx = p.x - n.x;
        let dy = p.y - n.y;
        d_pad = d_pad.min((dx * dx + dy * dy).sqrt() - pad);
    }
    let d_xy = d_beam.min(d_pad);
    let d_z = (p.z - plate * 0.5).abs() - plate * 0.5;
    let mut d = extrude_sdf(d_xy, d_z);

    for node in FACE_NODES_UV {
        let a = Vector3::new(node[0] * pitch, node[1] * pitch, 0.0);
        let to_c = Vector3::new(center.x - a.x, center.y - a.y, 0.0);
        let len = to_c.length().max(1e-6);
        let b = a + (to_c * (1.0 / len) + Vector3::new(0.0, 0.0, 1.0)).normalize() * tab;
        d = d.min(segment_distance(p, a, b) - half_eff * 0.85);
    }

    if mold_ready && rivet_diameter > 0.0 {
        let r = rivet_diameter * 0.5;
        for node in FACE_NODES_UV {
            let cx = node[0] * pitch;
            let cy = node[1] * pitch;
            let d_cyl = ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt() - r;
            let d_hole = extrude_sdf(d_cyl, (p.z - plate * 0.5).abs() - plate * 0.55);
            d = d.max(-d_hole);
        }
    }
    d
}

fn extrude_sdf(d_xy: f32, d_z: f32) -> f32 {
    if d_xy < 0.0 && d_z < 0.0 {
        d_xy.max(d_z)
    } else if d_xy > 0.0 && d_z > 0.0 {
        (d_xy * d_xy + d_z * d_z).sqrt()
    } else {
        d_xy.max(d_z)
    }
}

fn place_face(
    proto: &BufferGeometry,
    voxel_origin: Vector3,
    pitch: f32,
    face: usize,
    explode: f32,
) -> BufferGeometry {
    let (axis, low) = face_axis_low(face);
    let (u_hat, v_hat, n_in) = face_basis(axis, low);
    let plane = if low {
        Vector3::ZERO
    } else {
        match axis {
            0 => Vector3::new(pitch, 0.0, 0.0),
            1 => Vector3::new(0.0, pitch, 0.0),
            _ => Vector3::new(0.0, 0.0, pitch),
        }
    };
    let n_out = n_in * -1.0;
    let shift = voxel_origin + plane + n_out * (explode * pitch);

    let mut geom = proto.clone();
    if let Some(pos) = geom.get_attribute("position") {
        let mut out = pos.array.clone();
        for v in out.chunks_exact_mut(3) {
            let local = Vector3::new(v[0], v[1], v[2]);
            let w = shift + u_hat * local.x + v_hat * local.y + n_in * local.z;
            v[0] = w.x;
            v[1] = w.y;
            v[2] = w.z;
        }
        geom.set_attribute("position", BufferAttribute::new(out, 3));
    }
    if let Some(nor) = geom.get_attribute("normal") {
        let mut out = nor.array.clone();
        for v in out.chunks_exact_mut(3) {
            let local = Vector3::new(v[0], v[1], v[2]);
            let w = (u_hat * local.x + v_hat * local.y + n_in * local.z).normalize();
            v[0] = w.x;
            v[1] = w.y;
            v[2] = w.z;
        }
        geom.set_attribute("normal", BufferAttribute::new(out, 3));
    }
    geom
}

fn face_basis(axis: usize, low: bool) -> (Vector3, Vector3, Vector3) {
    // Inward normal: from the cube face into the voxel [0,1]³.
    match (axis, low) {
        (0, true) => (
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 0.0),
        ),
        (0, false) => (
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(-1.0, 0.0, 0.0),
        ),
        (1, true) => (
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
        (1, false) => (
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, -1.0, 0.0),
        ),
        (_, true) => (
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ),
        (_, false) => (
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        ),
    }
}

fn face_svg(kind: Cuboct, shape: f32, hand: Hand, pitch: f32, beam: f32, pad: f32) -> String {
    let beams = uv_beams(kind, shape, hand);
    let mut out = String::new();
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {pitch} {pitch}\" \
         width=\"{pitch}mm\" height=\"{pitch}mm\">\n"
    ));
    out.push_str(&format!(
        "  <g fill=\"none\" stroke=\"#222\" stroke-width=\"{beam}\" stroke-linecap=\"round\">\n"
    ));
    for seg in &beams {
        out.push_str(&format!(
            "    <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" />\n",
            seg[0][0] * pitch,
            seg[0][1] * pitch,
            seg[1][0] * pitch,
            seg[1][1] * pitch
        ));
    }
    out.push_str("  </g>\n");
    for node in FACE_NODES_UV {
        out.push_str(&format!(
            "  <circle cx=\"{}\" cy=\"{}\" r=\"{pad}\" fill=\"#222\" />\n",
            node[0] * pitch,
            node[1] * pitch
        ));
    }
    out.push_str("</svg>\n");
    out
}

fn tint_geometry(geom: &mut BufferGeometry, rgb: [f32; 3]) {
    let pos = geom.get_attribute("position").expect("position");
    let n = pos.array.len() / 3;
    let mut colors = vec![0.0f32; n * 3];
    for (i, chunk) in colors.chunks_exact_mut(3).enumerate() {
        let _ = i;
        chunk.copy_from_slice(&rgb);
    }
    geom.set_attribute("color", BufferAttribute::new(colors, 3));
}

fn translate_geometry(geom: &mut BufferGeometry, t: Vector3) {
    if let Some(pos) = geom.get_attribute("position") {
        let mut out = pos.array.clone();
        for v in out.chunks_exact_mut(3) {
            v[0] += t.x;
            v[1] += t.y;
            v[2] += t.z;
        }
        geom.set_attribute("position", BufferAttribute::new(out, 3));
    }
}

fn part_rgb(i: usize, n: usize) -> [f32; 3] {
    let h = (i as f32 / n.max(1) as f32) * 0.92;
    let s = 0.55;
    let v = 0.88;
    let c = v * s;
    let hp = h * 6.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

fn scad_assembly(asm: &CuboctAssembly<'_>) -> String {
    let pitch = asm.pitch;
    let [nx, ny, nz] = asm.cells;
    let mut out = String::from("// Cuboct assembly — Jenett et al. Sci. Adv. 2020\n");
    out.push_str(&format!("pitch = {pitch}; plate = {pl}; beam = {beam}; pad = {pad};\n",
        pl = asm.plate, beam = asm.beam, pad = asm.pad));
    out.push_str("module cuboct_face(kind) {\n");
    out.push_str("  // Placeholder: import exported STL/OBJ or extrude from cuboct-face.svg\n");
    out.push_str("  color(\"SteelBlue\") linear_extrude(height=plate) square(pitch, center=false);\n");
    out.push_str("}\n\n");
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let origin = asm.voxel_origin(ix as i32, iy as i32, iz as i32);
                let kind = asm.kind_at(ix as i32, iy as i32, iz as i32);
                for face in 0..6 {
                    let (axis, low) = face_axis_low(face);
                    let plane = if low {
                        Vector3::ZERO
                    } else {
                        match axis {
                            0 => Vector3::new(pitch, 0.0, 0.0),
                            1 => Vector3::new(0.0, pitch, 0.0),
                            _ => Vector3::new(0.0, 0.0, pitch),
                        }
                    };
                    let at = origin + plane;
                    out.push_str(&format!(
                        "translate([{:.3}, {:.3}, {:.3}]) // voxel ({ix},{iy},{iz}) face {face} {}\n",
                        at.x, at.y, at.z, kind.name()
                    ));
                    out.push_str("  cuboct_face();\n\n");
                }
            }
        }
    }
    if asm.mold_ready {
        out.push_str(&format!(
            "// Rivet holes Ø{:.3} mm at joint nodes\n",
            asm.rivet_diameter
        ));
        for j in asm.joints() {
            out.push_str(&format!(
                "translate([{:.3}, {:.3}, {:.3}]) cylinder(h={:.3}, d={:.3}, $fn=16);\n",
                j.at.x, j.at.y, j.at.z,
                asm.plate * 1.2,
                asm.rivet_diameter
            ));
        }
    }
    out
}

fn scad_mold(asm: &CuboctAssembly<'_>) -> String {
    let pitch = asm.pitch;
    let draft = asm.draft_deg;
    let plate = asm.plate;
    let kind = asm.kind;
    let hand = kind.hand_on_face(2, 0, 0, 0, asm.chiral_rule, 0);
    let svg_hint = face_svg(kind, asm.shape, hand, pitch, asm.beam, asm.pad);
    let mut out = String::from("// Mold tooling — cuboct face cavity + core\n");
    out.push_str(&format!(
        "pitch={pitch}; plate={plate}; draft={draft}; rivet_d={:.3};\n\n",
        asm.rivet_diameter
    ));
    out.push_str("// 2D profile (also exported as cuboct-face.svg):\n// ");
    out.push_str(svg_hint.lines().next().unwrap_or(""));
    out.push('\n');
    out.push_str("module cavity() {\n");
    out.push_str(&format!(
        "  linear_extrude(height=plate, scale=1+tan({draft})*1) // draft on pull direction\n"
    ));
    out.push_str("    square(pitch, center=false);\n");
    out.push_str("}\nmodule core() { translate([0,0,plate]) mirror([0,0,1]) cavity(); }\n");
    out.push_str("difference() { cube([pitch*1.4, pitch*1.4, plate*2.2], center=true); cavity(); }\n");
    if asm.mold_ready {
        for node in FACE_NODES_UV {
            out.push_str(&format!(
                "translate([{:.3}, {:.3}, 0]) cylinder(h=plate*1.5, d={:.3}, $fn=20, center=true);\n",
                node[0] * pitch,
                node[1] * pitch,
                asm.rivet_diameter
            ));
        }
    }
    out
}

fn geometry_to_stl_bytes(g: &BufferGeometry) -> Vec<u8> {
    let pos = match g.get_attribute("position") {
        Some(a) => &a.array,
        None => return Vec::new(),
    };
    let vert = |i: usize| [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
    let mut tris: Vec<[[f32; 3]; 3]> = Vec::new();
    if let Some(idx) = &g.index {
        for t in idx.chunks_exact(3) {
            tris.push([
                vert(t[0] as usize),
                vert(t[1] as usize),
                vert(t[2] as usize),
            ]);
        }
    } else {
        for k in 0..pos.len() / 9 {
            tris.push([vert(k * 3), vert(k * 3 + 1), vert(k * 3 + 2)]);
        }
    }
    let mut out = Vec::with_capacity(84 + tris.len() * 50);
    out.extend_from_slice(&[0u8; 80]);
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for t in &tris {
        let (a, b, c) = (t[0], t[1], t[2]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let mut n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 0.0 {
            n = [n[0] / len, n[1] / len, n[2] / len];
        }
        for x in n {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for vtx in t {
            for x in vtx {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

fn geometry_to_obj_text(g: &BufferGeometry) -> String {
    let pos = match g.get_attribute("position") {
        Some(a) => &a.array,
        None => return String::new(),
    };
    let mut s = String::from("# threers cuboct part\no part\n");
    for c in pos.chunks_exact(3) {
        s.push_str(&format!("v {} {} {}\n", c[0], c[1], c[2]));
    }
    if let Some(idx) = &g.index {
        for t in idx.chunks_exact(3) {
            s.push_str(&format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1));
        }
    } else {
        for k in 0..pos.len() / 9 {
            let b = (k * 3 + 1) as u32;
            s.push_str(&format!("f {} {} {}\n", b, b + 1, b + 2));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_voxel_has_six_faces() {
        let asm = CuboctAssembly::new(Cuboct::Rigid).pitch(10.0).resolution(16);
        assert_eq!(asm.parts().len(), 6);
        let geom = asm.build();
        assert!(geom.index.as_ref().unwrap().len() / 3 > 80);
    }

    #[test]
    fn inner_and_inter_joints() {
        let asm = CuboctAssembly::new(Cuboct::Rigid)
            .pitch(8.0)
            .cells([2, 1, 1])
            .resolution(12);
        let j = asm.joints();
        let inner = j
            .iter()
            .filter(|j| j.kind == CuboctJointKind::Inner)
            .count();
        let inter = j
            .iter()
            .filter(|j| j.kind == CuboctJointKind::Inter)
            .count();
        assert_eq!(inner, 24, "12 edges × 2 voxels");
        assert_eq!(inter, 4, "one shared face, four nodes");
    }

    #[test]
    fn svg_has_beams_and_pads() {
        let svg = CuboctAssembly::new(Cuboct::Auxetic)
            .pitch(20.0)
            .shape(0.2)
            .svg();
        assert!(svg.contains("<line"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("20mm"));
    }

    #[test]
    fn explode_moves_faces_apart() {
        let packed = CuboctAssembly::new(Cuboct::Rigid)
            .pitch(10.0)
            .resolution(14)
            .explode(0.0)
            .build();
        let exploded = CuboctAssembly::new(Cuboct::Rigid)
            .pitch(10.0)
            .resolution(14)
            .explode(0.4)
            .build();
        let max_abs = |g: &BufferGeometry| {
            let pos = g.get_attribute("position").unwrap();
            pos.array.iter().fold(0.0f32, |m, v| m.max(v.abs()))
        };
        assert!(
            max_abs(&exploded) > max_abs(&packed) + 2.0,
            "packed {} exploded {}",
            max_abs(&packed),
            max_abs(&exploded)
        );
    }

    #[test]
    fn programmed_column_still_has_six_faces_per_voxel() {
        let m = 2i32;
        let asm = CuboctAssembly::new(Cuboct::ChiralCcw)
            .pitch(6.0)
            .cells([1, 1, 2])
            .shape(0.12)
            .chiral_rule(ChiralRule::R1)
            .program(move |_, _, k| Cuboct::column_half(k, m))
            .resolution(14);
        assert_eq!(asm.parts().len(), 12);
    }

    #[test]
    fn colored_build_has_vertex_colors() {
        let geom = CuboctAssembly::new(Cuboct::Rigid)
            .pitch(8.0)
            .resolution(12)
            .build_colored();
        assert!(geom.get_attribute("color").is_some());
    }

    #[test]
    fn mold_ready_adds_holes_without_crashing() {
        let geom = CuboctAssembly::new(Cuboct::Rigid)
            .pitch(10.0)
            .mold_ready(true)
            .draft(4.0)
            .rivet_diameter(2.38)
            .resolution(14)
            .build();
        assert!(geom.index.as_ref().unwrap().len() > 100);
    }

    #[test]
    fn scad_and_plan_nonempty() {
        let asm = CuboctAssembly::new(Cuboct::Auxetic)
            .pitch(20.0)
            .cells([1, 1, 1])
            .mold_ready(true);
        assert!(asm.to_scad().contains("cuboct_face"));
        assert!(asm.to_scad_mold().contains("cavity"));
        assert!(asm.assembly_plan().to_csv().contains("pick"));
    }
}
