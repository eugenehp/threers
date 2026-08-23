//! Kirigami Expanded Miura — plate-lattice corrugations.
//!
//! Implements the construction from Parra Rubio et al., *Kirigami
//! Corrugations: Strong, Modular, and Programmable Plate Lattices*
//! (ASME IDETC/CIE 2023 / CBA):
//!
//! 1. Folded Miura-ori supports from `(ℓₓ, ℓᵧ, L, h)` — eq. (1)
//! 2. Expand each support into a `bₓ × bᵧ` rectangular pad — eqs. (2)–(3)
//! 3. Keep top / bottom pads and the inclined rectangles along the
//!    *yz*-corrugations; drop the zig-zag parallelograms (the kirigami cuts)
//! 4. Custom tops (§2.3.2): per-support heights `hᵢⱼ`, normals `nᵢⱼ`, and
//!    inclination `ψ`, with top corners from line–plane intersection
//! 5. Development into 2D strip nets (§2.4) and discrete origami cells (§2.7)
//!
//! Output is a plate mesh (`BufferGeometry`), not a marching-cubes lattice:
//! every face stays planar (planar sandwich / single curvature) and crease
//! topology is retained.
//!
//! ```
//! use threers::KirigamiExpandedMiura;
//!
//! let geom = KirigamiExpandedMiura::new(4, 5)
//!     .cell(28.5, 29.5, 68.0)
//!     .height(50.0)
//!     .base(11.0, 11.0)
//!     .build();
//! assert!(geom.index.is_some());
//! ```

use crate::core::{BufferAttribute, BufferGeometry};
use crate::geometries::lattice::{Cuboct, Infill, Lattice, LatticeKind, Strut, Tpms};
use crate::math::{Box3, Vector3};
use crate::utils::merge_geometries;

/// Role of a face in the expanded pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KirigamiFaceKind {
    /// Pad in the lower sandwich plane (even `j`).
    Bottom,
    /// Pad on the upper surface (odd `j`).
    Top,
    /// Rectangle spanning a *yz*-corrugation between pads `(i, j)` and `(i, j+1)`.
    Inclined,
}

/// One planar quad of the corrugation.
///
/// Corner order is canonical, not display winding:
/// pads are `++`, `+-`, `--`, `-+`; inclined faces are
/// `(-+, ++)[j]` then `(+-, --)[j+1]`.
#[derive(Debug, Clone, Copy)]
pub struct KirigamiFace {
    pub kind: KirigamiFaceKind,
    /// Column index `i` in `0..nx`.
    pub i: usize,
    /// For pads: support row `j`. For inclined faces: the lower support row.
    pub j: usize,
    pub verts: [[f64; 3]; 4],
}

impl KirigamiFace {
    /// Bilinear sample on the quad, `u,v` in `[0, 1]`.
    pub fn sample(&self, u: f64, v: f64) -> [f64; 3] {
        let [a, b, c, d] = self.verts;
        let ab = lerp(a, b, u);
        let dc = lerp(d, c, u);
        lerp(ab, dc, v)
    }

    /// Interior rivet sites on a pad, `n×n` grid inset from the border.
    pub fn rivet_holes(&self, n: usize) -> Vec<[f64; 3]> {
        let n = n.max(1);
        let mut out = Vec::with_capacity(n * n);
        let inset = 0.25;
        for iy in 0..n {
            for ix in 0..n {
                let u = inset + (1.0 - 2.0 * inset) * (ix as f64 + 0.5) / n as f64;
                let v = inset + (1.0 - 2.0 * inset) * (iy as f64 + 0.5) / n as f64;
                out.push(self.sample(u, v));
            }
        }
        out
    }
}

/// Evaluated folded state: every plate of the Kirigami Expanded Miura.
#[derive(Debug, Clone)]
pub struct KirigamiMesh {
    pub faces: Vec<KirigamiFace>,
    pub nx: usize,
    pub ny: usize,
    /// Plate thickness. `0` emits mid-surface quads; `> 0` extrudes each plate
    /// into a closed prism (paper §2.8 / 1.2 mm Hylite).
    pub thickness: f64,
}

impl KirigamiMesh {
    pub fn bottoms(&self) -> impl Iterator<Item = &KirigamiFace> {
        self.faces
            .iter()
            .filter(|f| f.kind == KirigamiFaceKind::Bottom)
    }

    pub fn tops(&self) -> impl Iterator<Item = &KirigamiFace> {
        self.faces
            .iter()
            .filter(|f| f.kind == KirigamiFaceKind::Top)
    }

    pub fn inclined(&self) -> impl Iterator<Item = &KirigamiFace> {
        self.faces
            .iter()
            .filter(|f| f.kind == KirigamiFaceKind::Inclined)
    }

    pub fn face(&self, kind: KirigamiFaceKind, i: usize, j: usize) -> Option<&KirigamiFace> {
        self.faces
            .iter()
            .find(|f| f.kind == kind && f.i == i && f.j == j)
    }

    /// Triangulated `BufferGeometry` of every plate (flat-shaded quads or prisms).
    pub fn to_geometry(&self) -> BufferGeometry {
        faces_to_geometry(self.faces.iter().copied(), self.thickness)
    }

    /// Separate geometries for bottom / top / inclined plates — handy when
    /// colouring roles differently in a viewer.
    pub fn to_geometry_by_kind(&self) -> (BufferGeometry, BufferGeometry, BufferGeometry) {
        let t = self.thickness;
        (
            faces_to_geometry(self.bottoms().copied(), t),
            faces_to_geometry(self.tops().copied(), t),
            faces_to_geometry(self.inclined().copied(), t),
        )
    }

    /// One discrete origami cell per bottom pad that has a following top
    /// (§2.7): bottom + inclined wall + top, folded on its own then riveted.
    pub fn discrete_cells(&self) -> Vec<KirigamiCell> {
        let mut cells = Vec::new();
        for i in 0..self.nx {
            for j in (0..self.ny).step_by(2) {
                if j + 1 >= self.ny {
                    break;
                }
                let mut faces = Vec::with_capacity(3);
                if let Some(f) = self.face(KirigamiFaceKind::Bottom, i, j) {
                    faces.push(*f);
                }
                if let Some(f) = self.face(KirigamiFaceKind::Inclined, i, j) {
                    faces.push(*f);
                }
                if let Some(f) = self.face(KirigamiFaceKind::Top, i, j + 1) {
                    faces.push(*f);
                }
                if !faces.is_empty() {
                    cells.push(KirigamiCell { i, j, faces });
                }
            }
        }
        cells
    }

    /// Unroll every *yz* strip independently and pack them one above the next.
    ///
    /// Always valid, including doubly-curved tops (§2.4). Planar sandwiches can
    /// later be joined across strips along the bottom pads; that is a net
    /// layout choice, not a different 3D mesh.
    pub fn develop(&self) -> KirigamiNet {
        let mut net = KirigamiNet {
            panels: Vec::new(),
            creases: Vec::new(),
        };
        let mut y_off = 0.0;
        let gap = 8.0;
        for i in 0..self.nx {
            let strip = develop_strip(self, i);
            let (min_x, _, min_y, max_y) = net_bounds(&strip);
            let h = (max_y - min_y).max(0.0);
            let shift = [-min_x, y_off - min_y];
            let base = net.panels.len();
            for mut p in strip.panels {
                p.verts = p.verts.map(|q| [q[0] + shift[0], q[1] + shift[1]]);
                net.panels.push(p);
            }
            for mut c in strip.creases {
                c.a += base;
                c.b += base;
                net.creases.push(c);
            }
            y_off += h + gap;
        }
        net
    }

    /// Axis-aligned bounds of every vertex.
    pub fn aabb(&self) -> ([f64; 3], [f64; 3]) {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for f in &self.faces {
            for v in &f.verts {
                for k in 0..3 {
                    min[k] = min[k].min(v[k]);
                    max[k] = max[k].max(v[k]);
                }
            }
        }
        if !min[0].is_finite() {
            return ([0.0; 3], [0.0; 3]);
        }
        (min, max)
    }

    /// Discrete cells pulled apart along their centroid from the patch centre.
    pub fn exploded(&self, gap: f64) -> KirigamiMesh {
        let (min, max) = self.aabb();
        let origin = [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ];
        let mut faces = Vec::new();
        for cell in self.discrete_cells() {
            let mut c = [0.0; 3];
            let mut n = 0.0;
            for f in &cell.faces {
                for v in &f.verts {
                    c = add(c, *v);
                    n += 1.0;
                }
            }
            if n > 0.0 {
                c = scale(c, 1.0 / n);
            }
            let mut d = sub(c, origin);
            let len = len(d);
            if len > 1e-9 {
                d = scale(d, gap / len);
            }
            for mut f in cell.faces {
                f.verts = f.verts.map(|v| add(v, d));
                faces.push(f);
            }
        }
        KirigamiMesh {
            faces,
            nx: self.nx,
            ny: self.ny,
            thickness: self.thickness,
        }
    }

    /// One mesh, vertex-coloured by discrete cell (for exploded / assembly views).
    pub fn to_geometry_cells(&self) -> BufferGeometry {
        let cells = self.discrete_cells();
        let n = cells.len().max(1) as f64;
        let tinted = cells.into_iter().enumerate().flat_map(|(k, cell)| {
            let hue = k as f64 / n;
            let rgb = hsl(hue, 0.42, 0.58);
            cell.faces.into_iter().map(move |f| (f, Some(rgb)))
        });
        faces_to_geometry_tinted(tinted, self.thickness)
    }

    /// Line-list of every plate edge (creases + pad borders).
    pub fn crease_geometry(&self) -> BufferGeometry {
        let mut positions = Vec::new();
        for f in &self.faces {
            for i in 0..4 {
                let a = f.verts[i];
                let b = f.verts[(i + 1) % 4];
                positions.extend_from_slice(&[
                    a[0] as f32,
                    a[1] as f32,
                    a[2] as f32,
                    b[0] as f32,
                    b[1] as f32,
                    b[2] as f32,
                ]);
            }
        }
        let mut geom = BufferGeometry::new();
        if !positions.is_empty() {
            geom.set_attribute("position", BufferAttribute::new(positions, 3));
        }
        geom
    }

    /// Rivet sites on every top and bottom pad.
    pub fn rivet_points(&self, n: usize) -> Vec<[f64; 3]> {
        self.bottoms()
            .chain(self.tops())
            .flat_map(|f| f.rivet_holes(n))
            .collect()
    }

    /// Translate every face vertex by `d`.
    pub fn translated(&self, d: [f64; 3]) -> Self {
        let faces = self
            .faces
            .iter()
            .map(|f| {
                let mut nf = *f;
                nf.verts = nf.verts.map(|v| add(v, d));
                nf
            })
            .collect();
        Self {
            faces,
            nx: self.nx,
            ny: self.ny,
            thickness: self.thickness,
        }
    }

    /// Concatenate two evaluated patches into one mesh.
    pub fn merged(&self, other: &Self) -> Self {
        let mut faces = self.faces.clone();
        faces.extend_from_slice(&other.faces);
        Self {
            faces,
            nx: self.nx + other.nx,
            ny: self.ny.max(other.ny),
            thickness: self.thickness.max(other.thickness),
        }
    }

    /// Tile this patch on a grid. `pitch_*` is the repeat period along each axis.
    pub fn array(&self, count_x: usize, count_y: usize, pitch_x: f64, pitch_y: f64, gap: f64) -> Self {
        let count_x = count_x.max(1);
        let count_y = count_y.max(1);
        let step_x = pitch_x + gap;
        let step_y = pitch_y + gap;
        let mut faces = Vec::with_capacity(self.faces.len() * count_x * count_y);
        for ix in 0..count_x {
            for iy in 0..count_y {
                let d = [ix as f64 * step_x, iy as f64 * step_y, 0.0];
                faces.extend(self.translated(d).faces);
            }
        }
        Self {
            faces,
            nx: self.nx * count_x,
            ny: self.ny * count_y,
            thickness: self.thickness,
        }
    }

    /// Stack identical layers along +Z with optional brick-style stagger on even layers.
    pub fn stack(&self, layers: usize, gap_z: f64, stagger: [f64; 2]) -> Self {
        let layers = layers.max(1);
        let (min, max) = self.aabb();
        let layer_h = (max[2] - min[2]).max(1.0);
        let mut faces = Vec::with_capacity(self.faces.len() * layers);
        for layer in 0..layers {
            let dz = layer as f64 * (layer_h + gap_z);
            let st = if layer % 2 == 1 { stagger } else { [0.0, 0.0] };
            faces.extend(
                self.translated([st[0], st[1], dz])
                    .faces,
            );
        }
        Self {
            faces,
            nx: self.nx,
            ny: self.ny * layers,
            thickness: self.thickness,
        }
    }

    /// Periodic lattice infill clipped to the corrugation bounding box.
    pub fn lattice_core(&self, kind: KirigamiCoreLattice, relative_density: f32) -> BufferGeometry {
        let (min, max) = self.aabb();
        if !min[0].is_finite() {
            return BufferGeometry::new();
        }
        let center = Vector3::new(
            0.5 * (min[0] + max[0]) as f32,
            0.5 * (min[1] + max[1]) as f32,
            0.5 * (min[2] + max[2]) as f32,
        );
        let size = Vector3::new(
            ((max[0] - min[0]) * 0.88).max(4.0) as f32,
            ((max[1] - min[1]) * 0.88).max(4.0) as f32,
            ((max[2] - min[2]) * 0.72).max(4.0) as f32,
        );
        let cell = (size.x.min(size.y).min(size.z) * 0.32).clamp(4.0, 14.0);
        // Keep sampling modest: hybrid gallery tiles share one GPU buffer, and
        // wgpu's default max buffer size is 256 MiB.
        let resolution = if cfg!(target_arch = "wasm32") {
            (size.x.max(size.y).max(size.z) * 1.2).clamp(20.0, 32.0) as usize
        } else {
            (size.x.max(size.y).max(size.z) * 1.5).clamp(24.0, 40.0) as usize
        };

        Lattice::new(kind.into())
            .bounds(Box3::from_center_and_size(center, size))
            .cell_size(Vector3::new(cell, cell, cell))
            .fit_relative_density(relative_density.clamp(0.05, 0.35))
            .resolution(resolution)
            .max_samples(600_000)
            .build()
    }

    /// Plate corrugation merged with a lattice core.
    pub fn to_geometry_with_core(
        &self,
        kind: KirigamiCoreLattice,
        relative_density: f32,
    ) -> BufferGeometry {
        let plates = self.to_geometry();
        let core = self.lattice_core(kind, relative_density);
        merge_geometries(&[plates, core]).unwrap_or_else(|| self.to_geometry())
    }

    /// Like [`develop`](Self::develop), but pack *yz* strips side-by-side for
    /// planar sandwiches (paper §2.4 strip joining along bottom pads).
    pub fn develop_joined(&self) -> KirigamiNet {
        let mut net = KirigamiNet {
            panels: Vec::new(),
            creases: Vec::new(),
        };
        let mut x_off = 0.0;
        let gap = 6.0;
        for i in 0..self.nx {
            let strip = develop_strip(self, i);
            let (min_x, max_x, min_y, _) = net_bounds(&strip);
            let w = (max_x - min_x).max(0.0);
            let shift = [x_off - min_x, -min_y];
            let base = net.panels.len();
            for mut p in strip.panels {
                p.verts = p.verts.map(|q| [q[0] + shift[0], q[1] + shift[1]]);
                net.panels.push(p);
            }
            for mut c in strip.creases {
                c.a += base;
                c.b += base;
                net.creases.push(c);
            }
            x_off += w + gap;
        }
        net
    }
}

/// Lattice infill family for hybrid kirigami–lattice cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KirigamiCoreLattice {
    Gyroid,
    Octet,
    Honeycomb,
    Cuboct,
}

impl KirigamiCoreLattice {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gyroid => "gyroid",
            Self::Octet => "octet",
            Self::Honeycomb => "honeycomb",
            Self::Cuboct => "cuboct",
        }
    }
}

impl From<KirigamiCoreLattice> for LatticeKind {
    fn from(k: KirigamiCoreLattice) -> Self {
        match k {
            KirigamiCoreLattice::Gyroid => LatticeKind::Tpms(Tpms::Gyroid),
            KirigamiCoreLattice::Octet => LatticeKind::Strut(Strut::Octet),
            KirigamiCoreLattice::Honeycomb => LatticeKind::Infill(Infill::Honeycomb),
            KirigamiCoreLattice::Cuboct => LatticeKind::Cuboct(Cuboct::Rigid),
        }
    }
}

/// Multi-patch kirigami layout — tiles, stacks, and merges.
#[derive(Debug, Clone, Default)]
pub struct KirigamiAssembly {
    parts: Vec<KirigamiMesh>,
}

impl KirigamiAssembly {
    pub fn new() -> Self {
        Self { parts: Vec::new() }
    }

    pub fn push(mut self, mesh: KirigamiMesh) -> Self {
        self.parts.push(mesh);
        self
    }

    /// `count_x × count_y` grid of one unit patch.
    pub fn tile(unit: &KirigamiMesh, count_x: usize, count_y: usize, gap: f64) -> Self {
        let (min, max) = unit.aabb();
        let px = (max[0] - min[0]).max(1.0);
        let py = (max[1] - min[1]).max(1.0);
        Self::new().push(unit.array(count_x, count_y, px, py, gap))
    }

    /// `layers` identical patches stacked with optional brick stagger.
    pub fn stack(unit: &KirigamiMesh, layers: usize, gap_z: f64, stagger: [f64; 2]) -> Self {
        Self::new().push(unit.stack(layers, gap_z, stagger))
    }

    pub fn evaluate(&self) -> KirigamiMesh {
        self.parts
            .first()
            .cloned()
            .map(|mut m| {
                for p in self.parts.iter().skip(1) {
                    m = m.merged(p);
                }
                m
            })
            .unwrap_or_else(|| KirigamiMesh {
                faces: Vec::new(),
                nx: 0,
                ny: 0,
                thickness: 0.0,
            })
    }
}

#[derive(Debug, Clone)]
pub struct KirigamiCell {
    pub i: usize,
    pub j: usize,
    pub faces: Vec<KirigamiFace>,
}

impl KirigamiCell {
    pub fn to_geometry(&self, thickness: f64) -> BufferGeometry {
        faces_to_geometry(self.faces.iter().copied(), thickness)
    }
}

/// Mountain / valley / cut in a developed net.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KirigamiCreaseKind {
    Mountain,
    Valley,
    Boundary,
}

/// A crease between two developed panels (indices into [`KirigamiNet::panels`]).
#[derive(Debug, Clone, Copy)]
pub struct KirigamiCrease {
    pub a: usize,
    pub b: usize,
    pub kind: KirigamiCreaseKind,
}

/// One face of a 2D net.
#[derive(Debug, Clone, Copy)]
pub struct KirigamiNetPanel {
    pub kind: KirigamiFaceKind,
    pub i: usize,
    pub j: usize,
    pub verts: [[f64; 2]; 4],
}

/// Developed crease pattern: isometric unrolling of the plates.
#[derive(Debug, Clone)]
pub struct KirigamiNet {
    pub panels: Vec<KirigamiNetPanel>,
    pub creases: Vec<KirigamiCrease>,
}

impl KirigamiNet {
    /// Axis-aligned bounds of the packed net.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        net_bounds(self)
    }

    /// SVG document in millimetre user units (paper fabrication scale).
    pub fn to_svg(&self) -> String {
        net_to_svg(self)
    }

    /// Plates laid in the XY plane — same topology as the SVG, for 3D viewers.
    pub fn to_geometry(&self, thickness: f64) -> BufferGeometry {
        let faces = self.panels.iter().map(|p| KirigamiFace {
            kind: p.kind,
            i: p.i,
            j: p.j,
            verts: p.verts.map(|q| [q[0], q[1], 0.0]),
        });
        faces_to_geometry(faces, thickness)
    }

    pub fn crease_geometry(&self) -> BufferGeometry {
        let mut positions = Vec::new();
        for c in &self.creases {
            if let (Some(a), Some(b)) = (self.panels.get(c.a), self.panels.get(c.b)) {
                if let Some((p0, p1)) = shared_edge_2d(&a.verts, &b.verts) {
                    positions.extend_from_slice(&[
                        p0[0] as f32,
                        p0[1] as f32,
                        0.05,
                        p1[0] as f32,
                        p1[1] as f32,
                        0.05,
                    ]);
                }
            }
        }
        let mut geom = BufferGeometry::new();
        if !positions.is_empty() {
            geom.set_attribute("position", BufferAttribute::new(positions, 3));
        }
        geom
    }
}

/// Named folded states used by the gallery, orbit demo, and wasm bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KirigamiPreset {
    /// Planar sandwich, paper §3 cell sizes.
    Planar,
    /// Same grid, shallower core and steeper *yz* walls.
    Steep,
    /// Single curvature: height varies along *y*.
    Cylinder,
    /// Doubly curved saddle on the top pads.
    Saddle,
    /// Discrete origami cells pulled apart.
    Exploded,
    /// Low core, gentle wall angle.
    Shallow,
    /// Single curvature along *x* (arch / barrel).
    Arch,
    /// Dome cap — radial height field on the top pads.
    Dome,
    /// Doubly curved ripple (`sin x · sin y`).
    Ripple,
    /// Linear height ramp along *y* (morphing-wing style).
    Wing,
    /// Coarser unit cell (`ℓₓ`, `ℓᵧ`, `L` scaled up).
    WideCell,
    /// Finer unit cell (scaled down pads).
    FineCell,
    /// Inverted saddle (valley surface).
    Valley,
    /// 2×2 tiled planar patch.
    Tiled,
    /// Two layers with brick stagger.
    Stacked,
    /// Planar corrugation with gyroid core infill.
    GyroidCore,
    /// Curved top with octet-truss core.
    OctetCore,
    /// Honeycomb infill inside a shallow core.
    HoneycombCore,
    /// Helical height field on the top pads.
    Helix,
    /// Core height graded along *x*.
    Gradient,
}

/// Wasm / UI slot reserved for the developed 2D net (not a folded preset).
pub const KIRIGAMI_NET_VARIANT: u32 = 5;

impl KirigamiPreset {
    pub const fn all() -> [KirigamiPreset; 20] {
        [
            Self::Planar,
            Self::Steep,
            Self::Cylinder,
            Self::Saddle,
            Self::Exploded,
            Self::Shallow,
            Self::Arch,
            Self::Dome,
            Self::Ripple,
            Self::Wing,
            Self::WideCell,
            Self::FineCell,
            Self::Valley,
            Self::Tiled,
            Self::Stacked,
            Self::GyroidCore,
            Self::OctetCore,
            Self::HoneycombCore,
            Self::Helix,
            Self::Gradient,
        ]
    }

    pub const fn count() -> usize {
        Self::all().len()
    }

    /// Map a UI / wasm variant index to a preset. Index [`KIRIGAMI_NET_VARIANT`]
    /// is reserved for the developed net and returns `None`.
    pub fn from_variant(v: u32) -> Option<Self> {
        if v == KIRIGAMI_NET_VARIANT {
            return None;
        }
        let i = if v > KIRIGAMI_NET_VARIANT {
            v - 1
        } else {
            v
        } as usize;
        Some(Self::all()[i % Self::count()])
    }

    pub fn from_index(i: u32) -> Self {
        Self::all()[(i as usize) % Self::count()]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Planar => "planar sandwich",
            Self::Steep => "steep walls",
            Self::Cylinder => "single curvature",
            Self::Saddle => "saddle",
            Self::Exploded => "discrete cells",
            Self::Shallow => "shallow core",
            Self::Arch => "arch",
            Self::Dome => "dome",
            Self::Ripple => "ripple",
            Self::Wing => "wing ramp",
            Self::WideCell => "wide cell",
            Self::FineCell => "fine cell",
            Self::Valley => "valley",
            Self::Tiled => "tiled 2×2",
            Self::Stacked => "stacked layers",
            Self::GyroidCore => "gyroid core",
            Self::OctetCore => "octet core",
            Self::HoneycombCore => "honeycomb core",
            Self::Helix => "helix",
            Self::Gradient => "gradient",
        }
    }

    /// Default support rows — curved tops need an extra zig-zag period.
    pub fn default_ny(self) -> usize {
        match self {
            Self::Cylinder
            | Self::Saddle
            | Self::Arch
            | Self::Dome
            | Self::Ripple
            | Self::Wing
            | Self::Valley
            | Self::Helix
            | Self::Gradient
            | Self::OctetCore => 8,
            Self::Tiled | Self::Stacked => 6,
            _ => 6,
        }
    }

    pub fn default_nx(self) -> usize {
        match self {
            Self::Tiled => 3,
            Self::Stacked | Self::GyroidCore | Self::HoneycombCore => 4,
            Self::OctetCore | Self::Helix | Self::Gradient => 5,
            _ => 5,
        }
    }

    pub fn uses_cell_colours(self) -> bool {
        self == Self::Exploded
    }

    pub fn uses_lattice_core(self) -> bool {
        matches!(
            self,
            Self::GyroidCore | Self::OctetCore | Self::HoneycombCore
        )
    }

    pub fn core_lattice(self) -> Option<KirigamiCoreLattice> {
        match self {
            Self::GyroidCore => Some(KirigamiCoreLattice::Gyroid),
            Self::OctetCore => Some(KirigamiCoreLattice::Octet),
            Self::HoneycombCore => Some(KirigamiCoreLattice::Honeycomb),
            _ => None,
        }
    }

    /// Build display geometry for a preset mesh.
    pub fn to_geometry(self, mesh: &KirigamiMesh) -> BufferGeometry {
        if let Some(core) = self.core_lattice() {
            mesh.to_geometry_with_core(core, 0.20)
        } else if self.uses_cell_colours() {
            mesh.to_geometry_cells()
        } else {
            mesh.to_geometry()
        }
    }

    /// Specimen-scale millimetres, `nx×ny` supports.
    pub fn evaluate(self, nx: usize, ny: usize, thickness: f64) -> KirigamiMesh {
        let (lx, ly, span) = match self {
            Self::WideCell => (40.0, 42.0, 90.0),
            Self::FineCell => (18.0, 19.0, 45.0),
            _ => (28.5, 29.5, 68.0),
        };
        let (bx, by) = match self {
            Self::FineCell => (7.0, 7.0),
            Self::WideCell => (14.0, 14.0),
            _ => (11.0, 11.0),
        };
        let base = KirigamiExpandedMiura::new(nx, ny)
            .cell(lx, ly, span)
            .base(bx, by)
            .thickness(thickness);

        let x_max = lx * nx.saturating_sub(1).max(1) as f64;
        let y_max = span * 0.5 * ny.saturating_sub(1).max(1) as f64 + ly;
        let cx = x_max * 0.5;
        let cy = y_max * 0.5;

        match self {
            Self::Planar => base.height(50.0).evaluate(),
            Self::Steep => base
                .height(28.0)
                .inclination(80.0_f64.to_radians())
                .evaluate(),
            Self::Shallow => base
                .height(32.0)
                .inclination(55.0_f64.to_radians())
                .evaluate(),
            Self::Cylinder => base
                .height(40.0)
                .inclination(70.0_f64.to_radians())
                .surface(move |_x, y| {
                    26.0 + 24.0 * (y / y_max.max(1.0) * std::f64::consts::PI * 0.55).sin()
                })
                .evaluate(),
            Self::Arch => base
                .height(40.0)
                .inclination(70.0_f64.to_radians())
                .surface(move |x, _y| {
                    let t = (x - cx) / (cx.max(1.0));
                    32.0 + 20.0 * (1.0 - t * t).max(0.0)
                })
                .evaluate(),
            Self::Saddle => base
                .height(40.0)
                .inclination(70.0_f64.to_radians())
                .surface(move |x, y| {
                    let u = (x - cx) / (cx.max(1.0) * 0.85);
                    let v = (y - cy) / (cy.max(1.0) * 0.85);
                    34.0 + 16.0 * (u * u - 0.7 * v * v)
                })
                .evaluate(),
            Self::Valley => base
                .height(40.0)
                .inclination(70.0_f64.to_radians())
                .surface(move |x, y| {
                    let u = (x - cx) / (cx.max(1.0) * 0.85);
                    let v = (y - cy) / (cy.max(1.0) * 0.85);
                    34.0 + 16.0 * (v * v - u * u)
                })
                .evaluate(),
            Self::Dome => base
                .height(38.0)
                .inclination(70.0_f64.to_radians())
                .surface(move |x, y| {
                    let u = (x - cx) / (cx.max(1.0) * 0.9);
                    let v = (y - cy) / (cy.max(1.0) * 0.9);
                    let r2 = (u * u + v * v).min(1.0);
                    30.0 + 22.0 * (1.0 - r2).sqrt()
                })
                .evaluate(),
            Self::Ripple => base
                .height(36.0)
                .inclination(68.0_f64.to_radians())
                .surface(move |x, y| {
                    34.0
                        + 10.0
                            * (x / lx * 0.55).sin()
                            * (y / (span * 0.5).max(1.0) * 0.55).sin()
                })
                .evaluate(),
            Self::Wing => base
                .height(34.0)
                .inclination(70.0_f64.to_radians())
                .surface(move |_x, y| 28.0 + 0.11 * y)
                .evaluate(),
            Self::WideCell => base.height(55.0).evaluate(),
            Self::FineCell => base.height(35.0).evaluate(),
            Self::Exploded => base.height(50.0).evaluate().exploded(22.0),
            Self::Tiled => {
                let unit = base.height(50.0).evaluate();
                let (min, max) = unit.aabb();
                unit.array(2, 2, (max[0] - min[0]).max(1.0), (max[1] - min[1]).max(1.0), 10.0)
            }
            Self::Stacked => {
                let unit = base.height(45.0).evaluate();
                let (min, max) = unit.aabb();
                let stagger = [0.5 * (max[0] - min[0]), 0.0];
                unit.stack(2, 8.0, stagger)
            }
            Self::GyroidCore => base.height(48.0).evaluate(),
            Self::OctetCore => base
                .height(38.0)
                .inclination(68.0_f64.to_radians())
                .surface(move |x, y| {
                    let u = (x - cx) / cx.max(1.0);
                    let v = (y - cy) / cy.max(1.0);
                    32.0 + 14.0 * (1.0 - u * u - v * v).max(0.0).sqrt()
                })
                .evaluate(),
            Self::HoneycombCore => base
                .height(36.0)
                .inclination(62.0_f64.to_radians())
                .evaluate(),
            Self::Helix => base
                .height(40.0)
                .inclination(70.0_f64.to_radians())
                .surface(move |x, y| {
                    let t = y / y_max.max(1.0) * std::f64::consts::TAU * 1.25;
                    30.0 + 16.0 * t.sin() + 6.0 * (x / lx * 0.4).cos()
                })
                .evaluate(),
            Self::Gradient => base
                .height(40.0)
                .inclination(70.0_f64.to_radians())
                .heights({
                    let cols = nx;
                    move |i, j| {
                        if j % 2 == 0 {
                            return 0.0;
                        }
                        let t = i as f64 / (cols.saturating_sub(1).max(1) as f64);
                        26.0 + 28.0 * t
                    }
                })
                .evaluate(),
        }
    }

    /// Evaluate using each preset's default grid size.
    pub fn evaluate_default(self, thickness: f64) -> KirigamiMesh {
        self.evaluate(self.default_nx(), self.default_ny(), thickness)
    }
}

/// Builder for a Kirigami Expanded Miura corrugation.
///
/// Units are caller-defined (the paper's specimens use millimetres). Defaults
/// match the mechanical-test unit cell in §3 of the paper, with a representative
/// core height.
#[derive(Debug, Clone)]
pub struct KirigamiExpandedMiura {
    nx: usize,
    ny: usize,
    lx: f64,
    ly: f64,
    /// Distance `L` between neighbouring *xy*-corrugations of the same crease
    /// assignment.
    span: f64,
    height: f64,
    bx: f64,
    by: f64,
    /// Inclination `ψ` of the *yz* rectangles, from the xy-plane. `None`
    /// derives `atan2(h, L/2 − bᵧ)` so a planar sandwich matches eqs. (2)–(3).
    psi: Option<f64>,
    thickness: f64,
    heights: Option<Vec<f64>>,
    normals: Option<Vec<[f64; 3]>>,
}

impl KirigamiExpandedMiura {
    /// `nx × ny` support grid (columns × rows). Needs `nx ≥ 1`, `ny ≥ 2` for
    /// at least one inclined face.
    pub fn new(nx: usize, ny: usize) -> Self {
        Self {
            nx: nx.max(1),
            ny: ny.max(1),
            lx: 28.5,
            ly: 29.5,
            span: 68.0,
            height: 50.0,
            bx: 11.0,
            by: 11.0,
            psi: None,
            thickness: 0.0,
            heights: None,
            normals: None,
        }
    }

    /// Zig-zag cell sizes `(ℓₓ, ℓᵧ)` and corrugation pitch `L`.
    pub fn cell(mut self, lx: f64, ly: f64, span: f64) -> Self {
        self.lx = lx.max(1e-9);
        self.ly = ly.max(1e-9);
        self.span = span.max(1e-9);
        self
    }

    /// Distance `h` between the two sandwich planes (used when no height field
    /// is set, and as the fallback for derived `ψ`).
    pub fn height(mut self, h: f64) -> Self {
        self.height = h.max(0.0);
        self
    }

    /// Pad sizes `(bₓ, bᵧ)`. Must satisfy `0 < bₓ < ℓₓ` and `0 < bᵧ < ℓᵧ`.
    pub fn base(mut self, bx: f64, by: f64) -> Self {
        self.bx = bx.clamp(1e-9, self.lx - 1e-9);
        self.by = by.clamp(1e-9, self.ly - 1e-9);
        self
    }

    /// Inclination `ψ` of the *yz* rectangles, in radians from the xy-plane.
    /// The paper's specimens use `70°`.
    pub fn inclination(mut self, psi: f64) -> Self {
        self.psi = Some(psi.clamp(1e-3, std::f64::consts::PI - 1e-3));
        self
    }

    /// Plate thickness. `0` (default) is a mid-surface; the paper's Hylite
    /// specimens are `1.2`.
    pub fn thickness(mut self, t: f64) -> Self {
        self.thickness = t.max(0.0);
        self
    }

    /// Per-support heights `hᵢⱼ`. Even `j` is ignored (`rⱼ = 0` in eq. (1));
    /// odd `j` sets the top-plane offset.
    pub fn heights(mut self, f: impl Fn(usize, usize) -> f64) -> Self {
        let mut g = vec![self.height; self.nx * self.ny];
        for i in 0..self.nx {
            for j in 0..self.ny {
                g[self.idx(i, j)] = f(i, j).max(0.0);
            }
        }
        self.heights = Some(g);
        self
    }

    /// Per-support normals of the top pads (odd `j`). Missing entries, or a
    /// call skipped entirely, are estimated from neighbouring heights.
    pub fn normals(mut self, f: impl Fn(usize, usize) -> [f64; 3]) -> Self {
        let mut g = vec![[0.0, 0.0, 1.0]; self.nx * self.ny];
        for i in 0..self.nx {
            for j in 0..self.ny {
                g[self.idx(i, j)] = norm(f(i, j));
            }
        }
        self.normals = Some(g);
        self
    }

    /// Sample a height field `z = f(x, y)` at each top support. Bottoms stay
    /// on `z = 0`. Normals are estimated from the sampled heights.
    pub fn surface(self, f: impl Fn(f64, f64) -> f64) -> Self {
        let lx = self.lx;
        let ly = self.ly;
        let span = self.span;
        self.heights(move |i, j| {
            if j % 2 == 0 {
                return 0.0;
            }
            let x = lx * i as f64;
            let y = span * 0.5 * j as f64 + r_parity(i + 1) * ly;
            f(x, y)
        })
    }

    pub fn nx(&self) -> usize {
        self.nx
    }

    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Folded supports `Vᵢⱼ` — eq. (1), with `h` possibly varying per site.
    pub fn support(&self, i: usize, j: usize) -> [f64; 3] {
        [
            self.lx * i as f64,
            self.span * 0.5 * j as f64 + r_parity(i + 1) * self.ly,
            r_parity(j) * self.h_at(i, j),
        ]
    }

    /// Four corners of the axis-aligned expansion at `(i, j)` — eqs. (2)–(3).
    ///
    /// Order: `++`, `+-`, `--`, `-+`. For odd `j` the custom construction in
    /// [`evaluate`](Self::evaluate) replaces this with the §2.3.2 intersection.
    pub fn pad_corners(&self, i: usize, j: usize) -> [[f64; 3]; 4] {
        self.expand_support(self.support(i, j))
    }

    /// Every plate in the folded state.
    pub fn evaluate(&self) -> KirigamiMesh {
        let mut pads = vec![[[0.0; 3]; 4]; self.nx * self.ny];
        for i in 0..self.nx {
            for j in 0..self.ny {
                pads[self.idx(i, j)] = if j % 2 == 0 {
                    self.pad_corners(i, j)
                } else {
                    self.top_pad(i, j)
                };
            }
        }

        let mut faces = Vec::with_capacity(self.nx * self.ny + self.nx * self.ny.saturating_sub(1));
        for i in 0..self.nx {
            for j in 0..self.ny {
                let kind = if j % 2 == 0 {
                    KirigamiFaceKind::Bottom
                } else {
                    KirigamiFaceKind::Top
                };
                faces.push(KirigamiFace {
                    kind,
                    i,
                    j,
                    verts: pads[self.idx(i, j)],
                });
            }
        }
        for i in 0..self.nx {
            for j in 0..self.ny.saturating_sub(1) {
                let a = pads[self.idx(i, j)];
                let b = pads[self.idx(i, j + 1)];
                // +y edge of pad j (`-+`, `++`) → −y edge of pad j+1 (`+-`, `--`).
                faces.push(KirigamiFace {
                    kind: KirigamiFaceKind::Inclined,
                    i,
                    j,
                    verts: [a[3], a[0], b[1], b[2]],
                });
            }
        }

        KirigamiMesh {
            faces,
            nx: self.nx,
            ny: self.ny,
            thickness: self.thickness,
        }
    }

    /// Convenience: `evaluate().to_geometry()`.
    pub fn build(&self) -> BufferGeometry {
        self.evaluate().to_geometry()
    }

    /// Convenience: `evaluate().to_geometry_by_kind()`.
    pub fn build_by_kind(&self) -> (BufferGeometry, BufferGeometry, BufferGeometry) {
        self.evaluate().to_geometry_by_kind()
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.ny + j
    }

    fn h_at(&self, i: usize, j: usize) -> f64 {
        match &self.heights {
            Some(g) => g[self.idx(i, j)],
            None => self.height,
        }
    }

    fn psi(&self) -> f64 {
        if let Some(p) = self.psi {
            return p;
        }
        let run = (self.span * 0.5 - self.by).max(1e-9);
        (self.height / run).atan()
    }

    fn expand_support(&self, v: [f64; 3]) -> [[f64; 3]; 4] {
        let hx = self.bx * 0.5;
        let hy = self.by * 0.5;
        [
            [v[0] + hx, v[1] + hy, v[2]],
            [v[0] + hx, v[1] - hy, v[2]],
            [v[0] - hx, v[1] - hy, v[2]],
            [v[0] - hx, v[1] + hy, v[2]],
        ]
    }

    fn normal_at(&self, i: usize, j: usize) -> [f64; 3] {
        if let Some(ns) = &self.normals {
            return ns[self.idx(i, j)];
        }
        if j.is_multiple_of(2) {
            return [0.0, 0.0, 1.0];
        }
        let h = |ii: usize, jj: usize| self.h_at(ii, jj);
        let dx = if i + 1 < self.nx && i > 0 {
            (h(i + 1, j) - h(i - 1, j)) / (2.0 * self.lx)
        } else if i + 1 < self.nx {
            (h(i + 1, j) - h(i, j)) / self.lx
        } else if i > 0 {
            (h(i, j) - h(i - 1, j)) / self.lx
        } else {
            0.0
        };
        let dy = if j + 2 < self.ny && j >= 2 {
            (h(i, j + 2) - h(i, j - 2)) / (2.0 * self.span)
        } else if j + 2 < self.ny {
            (h(i, j + 2) - h(i, j)) / self.span
        } else if j >= 2 {
            (h(i, j) - h(i, j - 2)) / self.span
        } else {
            0.0
        };
        norm([-dx, -dy, 1.0])
    }

    /// Top pad at odd `j` via §2.3.2: shoot `v±`-parallel lines from adjacent
    /// lower-pad corners into the top plane `(Vᵢⱼ, nᵢⱼ)`.
    ///
    /// The paper's index on those lower corners is written `i−1` / `i+1`; the
    /// adjacent *lower* faces of a top at `j` are the even rows `j−1` and
    /// `j+1` on the same strip, which is what the *yz* inclination `ψ` requires.
    fn top_pad(&self, i: usize, j: usize) -> [[f64; 3]; 4] {
        let v = self.support(i, j);
        let n = self.normal_at(i, j);
        let psi = self.psi();
        let vp = [0.0, psi.cos(), psi.sin()];
        let vm = [0.0, -psi.cos(), psi.sin()];

        let lower = self.pad_corners(i, j - 1);
        // −y edge of the top: from the +y edge of the previous bottom.
        let mm = plane_line(lower[3], vp, v, n);
        let pm = plane_line(lower[0], vp, v, n);

        
        if j + 1 < self.ny {
            let upper = self.pad_corners(i, j + 1);
            let mp = plane_line(upper[2], vm, v, n);
            let pp = plane_line(upper[1], vm, v, n);
            [pp, pm, mm, mp]
        } else {
            // No next bottom: complete a parallelogram in the top plane.
            let ey = scale(norm(project_plane([0.0, 1.0, 0.0], n)), self.by);
            let mp = add(mm, ey);
            let pp = add(pm, ey);
            [pp, pm, mm, mp]
        }
    }
}

fn r_parity(k: usize) -> f64 {
    (k % 2) as f64
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn len(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn norm(a: [f64; 3]) -> [f64; 3] {
    let n = len(a);
    if n < 1e-15 {
        [0.0, 0.0, 1.0]
    } else {
        scale(a, 1.0 / n)
    }
}

fn lerp(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    add(a, scale(sub(b, a), t))
}

fn project_plane(v: [f64; 3], n: [f64; 3]) -> [f64; 3] {
    sub(v, scale(n, dot(v, n)))
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    len(sub(a, b))
}

/// Intersect the line `p0 + t dir` with the plane through `point` with normal `n`.
fn plane_line(p0: [f64; 3], dir: [f64; 3], point: [f64; 3], n: [f64; 3]) -> [f64; 3] {
    let denom = dot(dir, n);
    if denom.abs() < 1e-12 {
        return sub(p0, scale(n, dot(sub(p0, point), n)));
    }
    let t = dot(sub(point, p0), n) / denom;
    add(p0, scale(dir, t))
}

fn face_normal(verts: &[[f64; 3]; 4]) -> [f64; 3] {
    norm(cross(sub(verts[1], verts[0]), sub(verts[2], verts[0])))
}

fn emit_winding(kind: KirigamiFaceKind, verts: [[f64; 3]; 4]) -> [[f64; 3]; 4] {
    match kind {
        KirigamiFaceKind::Bottom => [verts[0], verts[3], verts[2], verts[1]],
        KirigamiFaceKind::Top => {
            if face_normal(&verts)[2] >= 0.0 {
                verts
            } else {
                [verts[0], verts[3], verts[2], verts[1]]
            }
        }
        KirigamiFaceKind::Inclined => verts,
    }
}

fn faces_to_geometry(faces: impl Iterator<Item = KirigamiFace>, thickness: f64) -> BufferGeometry {
    faces_to_geometry_tinted(faces.map(|f| (f, None)), thickness)
}

fn faces_to_geometry_tinted(
    faces: impl Iterator<Item = (KirigamiFace, Option<[f32; 3]>)>,
    thickness: f64,
) -> BufferGeometry {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    let mut any_color = false;

    for (face, tint) in faces {
        if tint.is_some() {
            any_color = true;
        }
        let verts = emit_winding(face.kind, face.verts);
        let n = face_normal(&verts);
        if thickness > 1e-12 {
            emit_prism(
                verts,
                n,
                thickness,
                tint,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut colors,
                &mut indices,
            );
        } else {
            emit_quad(
                verts,
                n,
                tint,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut colors,
                &mut indices,
            );
        }
    }

    let mut geom = BufferGeometry::new();
    if positions.is_empty() {
        return geom;
    }
    geom.set_attribute("position", BufferAttribute::new(positions, 3));
    geom.set_attribute("normal", BufferAttribute::new(normals, 3));
    geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
    if any_color {
        geom.set_attribute("color", BufferAttribute::new(colors, 3));
    }
    geom.set_index(indices);
    geom
}

fn push_color(colors: &mut Vec<f32>, tint: Option<[f32; 3]>) {
    let c = tint.unwrap_or([1.0, 1.0, 1.0]);
    colors.extend_from_slice(&c);
}

#[allow(clippy::too_many_arguments)]
fn emit_quad(
    verts: [[f64; 3]; 4],
    n: [f64; 3],
    tint: Option<[f32; 3]>,
    positions: &mut Vec<f32>,
    normals: &mut Vec<f32>,
    uvs: &mut Vec<f32>,
    colors: &mut Vec<f32>,
    indices: &mut Vec<u32>,
) {
    let base = (positions.len() / 3) as u32;
    for (k, v) in verts.iter().enumerate() {
        positions.extend_from_slice(&[v[0] as f32, v[1] as f32, v[2] as f32]);
        normals.extend_from_slice(&[n[0] as f32, n[1] as f32, n[2] as f32]);
        let u = match k {
            0 | 3 => 0.0,
            _ => 1.0,
        };
        let vv = match k {
            0 | 1 => 0.0,
            _ => 1.0,
        };
        uvs.extend_from_slice(&[u, vv]);
        push_color(colors, tint);
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[allow(clippy::too_many_arguments)]
fn emit_prism(
    verts: [[f64; 3]; 4],
    n: [f64; 3],
    thickness: f64,
    tint: Option<[f32; 3]>,
    positions: &mut Vec<f32>,
    normals: &mut Vec<f32>,
    uvs: &mut Vec<f32>,
    colors: &mut Vec<f32>,
    indices: &mut Vec<u32>,
) {
    let h = thickness * 0.5;
    let top = verts.map(|v| add(v, scale(n, h)));
    let bot = verts.map(|v| add(v, scale(n, -h)));
    emit_quad(top, n, tint, positions, normals, uvs, colors, indices);
    emit_quad(
        [bot[0], bot[3], bot[2], bot[1]],
        scale(n, -1.0),
        tint,
        positions,
        normals,
        uvs,
        colors,
        indices,
    );
    for i in 0..4 {
        let j = (i + 1) % 4;
        let side = [bot[i], bot[j], top[j], top[i]];
        let sn = face_normal(&side);
        emit_quad(side, sn, tint, positions, normals, uvs, colors, indices);
    }
}

fn hsl(h: f64, s: f64, l: f64) -> [f32; 3] {
    let hue2rgb = |p: f64, q: f64, mut t: f64| {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    [
        hue2rgb(p, q, h + 1.0 / 3.0) as f32,
        hue2rgb(p, q, h) as f32,
        hue2rgb(p, q, h - 1.0 / 3.0) as f32,
    ]
}

fn develop_strip(mesh: &KirigamiMesh, i: usize) -> KirigamiNet {
    let mut chain: Vec<&KirigamiFace> = Vec::new();
    for j in 0..mesh.ny {
        if let Some(f) = mesh.face(
            if j % 2 == 0 {
                KirigamiFaceKind::Bottom
            } else {
                KirigamiFaceKind::Top
            },
            i,
            j,
        ) {
            chain.push(f);
        }
        if j + 1 < mesh.ny {
            if let Some(f) = mesh.face(KirigamiFaceKind::Inclined, i, j) {
                chain.push(f);
            }
        }
    }
    if chain.is_empty() {
        return KirigamiNet {
            panels: Vec::new(),
            creases: Vec::new(),
        };
    }

    let eps = 1e-6;
    let mut panels = Vec::with_capacity(chain.len());
    let mut creases = Vec::new();
    let mut prev2 = flatten_quad(&chain[0].verts);
    panels.push(KirigamiNetPanel {
        kind: chain[0].kind,
        i: chain[0].i,
        j: chain[0].j,
        verts: prev2,
    });

    for k in 1..chain.len() {
        let next2 = attach_quad(&prev2, &chain[k - 1].verts, &chain[k].verts, eps);
        panels.push(KirigamiNetPanel {
            kind: chain[k].kind,
            i: chain[k].i,
            j: chain[k].j,
            verts: next2,
        });
        // Along a strip the first crease is valley (bottom → wall), then they
        // alternate — the Miura *yz* assignment.
        let kind = if (k - 1) % 2 == 0 {
            KirigamiCreaseKind::Valley
        } else {
            KirigamiCreaseKind::Mountain
        };
        creases.push(KirigamiCrease {
            a: k - 1,
            b: k,
            kind,
        });
        prev2 = next2;
    }

    KirigamiNet { panels, creases }
}

fn flatten_quad(v: &[[f64; 3]; 4]) -> [[f64; 2]; 4] {
    let p0 = [0.0, 0.0];
    let p1 = [dist3(v[0], v[1]), 0.0];
    let p2 = pick_side(p0, dist3(v[0], v[2]), p1, dist3(v[1], v[2]), 1.0);
    let p3 = pick_side(p0, dist3(v[0], v[3]), p1, dist3(v[1], v[3]), 1.0);
    [p0, p1, p2, p3]
}

fn attach_quad(
    placed: &[[f64; 2]; 4],
    placed3: &[[f64; 3]; 4],
    next3: &[[f64; 3]; 4],
    eps: f64,
) -> [[f64; 2]; 4] {
    let mut hits = Vec::new();
    for (ia, a) in placed3.iter().enumerate() {
        for (ib, b) in next3.iter().enumerate() {
            if dist3(*a, *b) < eps {
                hits.push((ia, ib));
            }
        }
    }
    if hits.len() < 2 {
        return flatten_quad(next3);
    }
    let mut out = [[0.0; 2]; 4];
    let mut known = [false; 4];
    for &(ia, ib) in &hits {
        out[ib] = placed[ia];
        known[ib] = true;
    }
    let known_idx: Vec<usize> = (0..4).filter(|&i| known[i]).collect();
    let a = known_idx[0];
    let b = known_idx[1];
    let placed_c = centroid2(placed);
    let side = -edge_side(out[a], out[b], placed_c);
    for i in 0..4 {
        if known[i] {
            continue;
        }
        out[i] = pick_side(
            out[a],
            dist3(next3[i], next3[a]),
            out[b],
            dist3(next3[i], next3[b]),
            side,
        );
    }
    out
}

fn centroid2(v: &[[f64; 2]; 4]) -> [f64; 2] {
    [
        (v[0][0] + v[1][0] + v[2][0] + v[3][0]) * 0.25,
        (v[0][1] + v[1][1] + v[2][1] + v[3][1]) * 0.25,
    ]
}

fn edge_side(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    let s = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if s >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn pick_side(a: [f64; 2], da: f64, b: [f64; 2], db: f64, side: f64) -> [f64; 2] {
    let d = ((b[0] - a[0]).hypot(b[1] - a[1])).max(1e-15);
    let x = (d * d + da * da - db * db) / (2.0 * d);
    let h = (da * da - x * x).max(0.0).sqrt();
    let ux = (b[0] - a[0]) / d;
    let uy = (b[1] - a[1]) / d;
    let px = a[0] + x * ux;
    let py = a[1] + x * uy;
    // (ux, uy) × (+h n) with n = (-uy, ux) is the +CCW candidate.
    let c1 = [px - uy * h, py + ux * h];
    let c2 = [px + uy * h, py - ux * h];
    if edge_side(a, b, c1) == side.signum() || side == 0.0 {
        c1
    } else {
        c2
    }
}

fn net_bounds(net: &KirigamiNet) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in &net.panels {
        for v in &p.verts {
            min_x = min_x.min(v[0]);
            max_x = max_x.max(v[0]);
            min_y = min_y.min(v[1]);
            max_y = max_y.max(v[1]);
        }
    }
    if !min_x.is_finite() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    (min_x, max_x, min_y, max_y)
}

fn net_to_svg(net: &KirigamiNet) -> String {
    let (min_x, max_x, min_y, max_y) = net.bounds();
    let margin = 10.0;
    let w = (max_x - min_x + 2.0 * margin).max(1.0);
    let h = (max_y - min_y + 2.0 * margin).max(1.0);
    let sx = |x: f64| x - min_x + margin;
    let sy = |y: f64| max_y - y + margin; // SVG y-down

    let mut out = String::new();
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.3}mm\" height=\"{h:.3}mm\" \
         viewBox=\"0 0 {w:.3} {h:.3}\">\n"
    ));
    out.push_str(
        "  <style>polygon{stroke:#222;stroke-width:0.35;stroke-linejoin:round}\
         .m{stroke:#111;stroke-width:0.45;fill:none}\
         .v{stroke:#111;stroke-width:0.45;fill:none;stroke-dasharray:2 1.4}</style>\n",
    );

    for p in &net.panels {
        let fill = match p.kind {
            KirigamiFaceKind::Bottom => "#e8dcc8",
            KirigamiFaceKind::Top => "#c5d4e0",
            KirigamiFaceKind::Inclined => "#8fb89a",
        };
        let pts: String = p
            .verts
            .iter()
            .map(|v| format!("{:.3},{:.3}", sx(v[0]), sy(v[1])))
            .collect::<Vec<_>>()
            .join(" ");
        out.push_str(&format!("  <polygon fill=\"{fill}\" points=\"{pts}\"/>\n"));
    }

    for c in &net.creases {
        if let (Some(a), Some(b)) = (net.panels.get(c.a), net.panels.get(c.b)) {
            if let Some((p0, p1)) = shared_edge_2d(&a.verts, &b.verts) {
                let class = match c.kind {
                    KirigamiCreaseKind::Mountain => "m",
                    KirigamiCreaseKind::Valley => "v",
                    KirigamiCreaseKind::Boundary => "m",
                };
                out.push_str(&format!(
                    "  <line class=\"{class}\" x1=\"{:.3}\" y1=\"{:.3}\" x2=\"{:.3}\" y2=\"{:.3}\"/>\n",
                    sx(p0[0]),
                    sy(p0[1]),
                    sx(p1[0]),
                    sy(p1[1])
                ));
            }
        }
    }

    out.push_str("</svg>\n");
    out
}

fn shared_edge_2d(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> Option<([f64; 2], [f64; 2])> {
    let mut hits = Vec::new();
    for pa in a {
        for pb in b {
            if (pa[0] - pb[0]).hypot(pa[1] - pb[1]) < 1e-4 {
                hits.push(*pa);
            }
        }
    }
    if hits.len() >= 2 {
        Some((hits[0], hits[1]))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planar() -> KirigamiExpandedMiura {
        KirigamiExpandedMiura::new(3, 4)
            .cell(28.5, 29.5, 68.0)
            .height(50.0)
            .base(11.0, 11.0)
    }

    fn quad_area(v: &[[f64; 3]; 4]) -> f64 {
        0.5 * (len(cross(sub(v[1], v[0]), sub(v[2], v[0])))
            + len(cross(sub(v[2], v[0]), sub(v[3], v[0]))))
    }

    fn quad_area2(v: &[[f64; 2]; 4]) -> f64 {
        let c = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
        0.5 * (c(sub2(v[1], v[0]), sub2(v[2], v[0])) + c(sub2(v[2], v[0]), sub2(v[3], v[0]))).abs()
    }

    fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [a[0] - b[0], a[1] - b[1]]
    }

    #[test]
    fn support_matches_equation_one() {
        let k = planar();
        let v = k.support(1, 2);
        // i=1, j=2: (ℓₓ, L + r₂ ℓᵧ, 0) with r₂=0 → (28.5, 68, 0)
        assert!((v[0] - 28.5).abs() < 1e-9);
        assert!((v[1] - 68.0).abs() < 1e-9);
        assert!(v[2].abs() < 1e-9);
        let top = k.support(0, 1);
        assert!((top[2] - 50.0).abs() < 1e-9);
        // r_{0+1}=r₁=1 → y = L/2 + ℓᵧ
        assert!((top[1] - (34.0 + 29.5)).abs() < 1e-9);
    }

    #[test]
    fn face_counts() {
        let mesh = planar().evaluate();
        let (nx, ny) = (3usize, 4usize);
        assert_eq!(mesh.faces.len(), nx * ny + nx * (ny - 1));
        assert_eq!(mesh.bottoms().count(), nx * ny.div_ceil(2));
        assert_eq!(mesh.tops().count(), nx * (ny / 2));
        assert_eq!(mesh.inclined().count(), nx * (ny - 1));
    }

    #[test]
    fn pads_are_axis_aligned_rectangles() {
        let mesh = planar().evaluate();
        for face in mesh.bottoms().chain(mesh.tops()) {
            let z = face.verts[0][2];
            for v in &face.verts {
                assert!((v[2] - z).abs() < 1e-8, "pad not planar in Z");
            }
            let xs: Vec<_> = face.verts.iter().map(|v| v[0]).collect();
            let ys: Vec<_> = face.verts.iter().map(|v| v[1]).collect();
            let dx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let dy = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - ys.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!((dx - 11.0).abs() < 1e-6, "dx={dx}");
            assert!((dy - 11.0).abs() < 1e-6, "dy={dy}");
        }
    }

    #[test]
    fn inclined_faces_are_planar() {
        for face in planar().evaluate().inclined() {
            let n = face_normal(&face.verts);
            let p0 = face.verts[0];
            for v in &face.verts[1..] {
                let d = dot(sub(*v, p0), n);
                assert!(d.abs() < 1e-8, "inclined face not planar: {d}");
            }
        }
    }

    #[test]
    fn build_emits_indexed_triangles() {
        let geom = planar().build();
        let tris = geom.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
        // 3*4 pads + 3*3 inclined = 21 quads → 42 tris
        assert_eq!(tris, 42);
        assert!(geom.attributes.contains_key("position"));
        assert!(geom.attributes.contains_key("normal"));
    }

    #[test]
    fn thickness_emits_prisms() {
        let thin = planar().build();
        let thick = planar().thickness(1.2).build();
        let n = |g: &BufferGeometry| g.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
        assert_eq!(n(&thick), n(&thin) * 6);
    }

    #[test]
    fn curved_tops_lie_on_their_planes() {
        let mesh = KirigamiExpandedMiura::new(4, 6)
            .cell(28.5, 29.5, 68.0)
            .height(40.0)
            .base(11.0, 11.0)
            .inclination(70.0_f64.to_radians())
            .surface(|_x, y| 30.0 + 0.08 * y)
            .evaluate();
        for face in mesh.tops() {
            let n = face_normal(&face.verts);
            let p0 = face.verts[0];
            for v in &face.verts[1..] {
                let d = dot(sub(*v, p0), n);
                assert!(d.abs() < 1e-6, "curved top not planar: {d}");
            }
            assert!(
                n[2].abs() > 0.5,
                "top pad should face generally ±Z, got {n:?}"
            );
        }
    }

    #[test]
    fn develop_preserves_area() {
        let mesh = planar().evaluate();
        let a3: f64 = mesh.faces.iter().map(|f| quad_area(&f.verts)).sum();
        let net = mesh.develop();
        let a2: f64 = net.panels.iter().map(|p| quad_area2(&p.verts)).sum();
        assert!((a3 - a2).abs() / a3.max(1.0) < 1e-4, "3D {a3} vs net {a2}");
        assert_eq!(net.panels.len(), mesh.faces.len());
        let svg = net.to_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<polygon"));
        assert!(svg.contains("stroke-dasharray"));
    }

    #[test]
    fn discrete_cells_cover_each_bay() {
        let mesh = planar().evaluate();
        let cells = mesh.discrete_cells();
        // 3 columns × 2 complete bays (j=0 and j=2; ny=4)
        assert_eq!(cells.len(), 6);
        assert!(cells.iter().all(|c| c.faces.len() == 3));
        let holes = cells[0].faces[0].rivet_holes(2);
        assert_eq!(holes.len(), 4);
    }

    #[test]
    fn exploded_keeps_cell_faces() {
        let mesh = planar().evaluate();
        let boom = mesh.exploded(20.0);
        assert_eq!(boom.discrete_cells().len(), mesh.discrete_cells().len());
        let (a, b) = mesh.aabb();
        let (c, d) = boom.aabb();
        let span = |lo: [f64; 3], hi: [f64; 3]| dist3(lo, hi);
        assert!(span(c, d) > span(a, b));
        assert!(boom.to_geometry_cells().attributes.contains_key("color"));
    }

    #[test]
    fn net_has_3d_geometry() {
        let net = planar().evaluate().develop();
        let geom = net.to_geometry(0.0);
        let tris = geom.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
        assert_eq!(tris, net.panels.len() * 2);
        assert!(!net.crease_geometry().attributes.is_empty());
    }

    #[test]
    fn presets_build() {
        for p in KirigamiPreset::all() {
            let m = p.evaluate(3, 4, 1.2);
            assert!(!m.faces.is_empty(), "{}", p.name());
            assert!(m.to_geometry().index.is_some());
        }
    }

    #[test]
    fn variant_index_maps_around_net_slot() {
        assert_eq!(KirigamiPreset::from_variant(4), Some(KirigamiPreset::Exploded));
        assert_eq!(KirigamiPreset::from_variant(KIRIGAMI_NET_VARIANT), None);
        assert_eq!(KirigamiPreset::from_variant(6), Some(KirigamiPreset::Shallow));
        assert_eq!(KirigamiPreset::from_variant(20), Some(KirigamiPreset::Gradient));
    }

    #[test]
    fn array_and_stack_assembly() {
        let unit = planar().evaluate();
        let tiled = unit.array(2, 2, 80.0, 90.0, 5.0);
        assert!(tiled.faces.len() > unit.faces.len());
        let stacked = unit.stack(2, 6.0, [20.0, 0.0]);
        let (a, b) = unit.aabb();
        let (c, d) = stacked.aabb();
        assert!(d[2] - c[2] > b[2] - a[2]);
    }

    #[test]
    fn develop_joined_is_wider_than_stacked() {
        let mesh = planar().evaluate();
        let joined = mesh.develop_joined();
        let stacked = mesh.develop();
        let (jmin, jmax, _, _) = joined.bounds();
        let (smin, smax, _, _) = stacked.bounds();
        assert!(jmax - jmin > smax - smin);
    }

    #[test]
    fn lattice_core_merges_with_plates() {
        let mesh = planar().evaluate();
        let geom = mesh.to_geometry_with_core(KirigamiCoreLattice::Gyroid, 0.18);
        let tris = geom.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
        let plates = mesh.to_geometry();
        let plate_tris = plates.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
        assert!(tris > plate_tris);
    }
}
