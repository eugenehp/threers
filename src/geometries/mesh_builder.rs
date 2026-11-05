//! Building a mesh in code, without assembling attribute arrays by hand.

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::{Color, Matrix4, Vector3};
use std::collections::HashMap;

/// Vertices and triangles, turned into a [`BufferGeometry`] by [`build`](Self::build).
///
/// Set what the next vertices carry with [`normal`](Self::normal),
/// [`uv`](Self::uv) and [`color`](Self::color), then add them. Whatever was never
/// set is filled in: normals are worked out from the faces, uvs are zero, and a
/// mesh with no colours has no colour attribute.
///
/// A triangle shows from the side its vertices run anticlockwise, and is invisible
/// from the other. That is easy to get wrong and silent when it is, so
/// [`facing`](Self::facing), [`ribbon`](Self::ribbon) and [`grid`](Self::grid) take
/// the side you mean rather than the order you happened to write.
///
/// ```
/// use threers::geometries::MeshBuilder;
/// use threers::math::Vector3;
///
/// let mut mesh = MeshBuilder::new();
/// let a = mesh.vertex(Vector3::new(0.0, 0.0, 0.0));
/// let b = mesh.vertex(Vector3::new(1.0, 0.0, 0.0));
/// let c = mesh.vertex(Vector3::new(1.0, 0.0, 1.0));
/// let d = mesh.vertex(Vector3::new(0.0, 0.0, 1.0));
/// mesh.facing(Vector3::UP).quad(a, b, c, d);
/// let geometry = mesh.build();
/// assert_eq!(geometry.triangle_count(), 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MeshBuilder {
    positions: Vec<Vector3>,
    normals: Vec<Option<Vector3>>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<Option<[f32; 3]>>,
    triangles: Vec<[u32; 3]>,
    next_normal: Option<Vector3>,
    next_uv: [f32; 2],
    next_color: Option<[f32; 3]>,
    side: Option<Vector3>,
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The normal of the vertices to come, instead of one from the faces.
    pub fn normal(&mut self, normal: Vector3) -> &mut Self {
        self.next_normal = Some(normal.normalize());
        self
    }

    /// Go back to working out normals from the faces.
    pub fn auto_normal(&mut self) -> &mut Self {
        self.next_normal = None;
        self
    }

    /// The texture coordinate of the vertices to come.
    pub fn uv(&mut self, u: f32, v: f32) -> &mut Self {
        self.next_uv = [u, v];
        self
    }

    /// The colour of the vertices to come. Needs a material with vertex colours on.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.next_color = Some([color.r, color.g, color.b]);
        self
    }

    /// Turn the triangles to come towards `side`, whatever order their vertices
    /// are given in. Their normal points within 90° of it.
    pub fn facing(&mut self, side: Vector3) -> &mut Self {
        self.side = Some(side);
        self
    }

    /// Take triangles as given again, anticlockwise from the front.
    pub fn as_wound(&mut self) -> &mut Self {
        self.side = None;
        self
    }

    /// Add a vertex, returning its index for [`triangle`](Self::triangle).
    pub fn vertex(&mut self, position: Vector3) -> u32 {
        self.positions.push(position);
        self.normals.push(self.next_normal);
        self.uvs.push(self.next_uv);
        self.colors.push(self.next_color);
        self.positions.len() as u32 - 1
    }

    /// How many vertices so far.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// How many triangles so far.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// # Panics
    ///
    /// If an index is not a vertex: here, where the mistake is, and not as a
    /// missing triangle later.
    pub fn triangle(&mut self, a: u32, b: u32, c: u32) -> &mut Self {
        let count = self.positions.len() as u32;
        for index in [a, b, c] {
            assert!(
                index < count,
                "vertex {index} does not exist: there are {count}"
            );
        }
        let (p, q, r) = (
            self.positions[a as usize],
            self.positions[b as usize],
            self.positions[c as usize],
        );
        let backwards = self
            .side
            .is_some_and(|side| (q - p).cross(r - p).dot(side) < 0.0);
        self.triangles
            .push(if backwards { [a, c, b] } else { [a, b, c] });
        self
    }

    /// Two triangles over four vertices in order round the edge.
    pub fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) -> &mut Self {
        self.triangle(a, b, c).triangle(a, c, d)
    }

    /// A strip between two edges, facing `side`. The edges pair up point by point.
    /// `closed` joins the last pair back to the first, for a loop of road.
    ///
    /// # Panics
    ///
    /// If the edges differ in length.
    pub fn ribbon(
        &mut self,
        left: &[Vector3],
        right: &[Vector3],
        side: Vector3,
        closed: bool,
    ) -> &mut Self {
        assert_eq!(left.len(), right.len(), "a ribbon's edges must pair up");
        let keep = self.side;
        self.facing(side);
        let mut along = 0.0;
        let first = self.positions.len() as u32;
        for (i, (l, r)) in left.iter().zip(right).enumerate() {
            if i > 0 {
                along += l.distance_to(left[i - 1]);
            }
            self.uv(0.0, along);
            self.vertex(*l);
            self.uv(1.0, along);
            self.vertex(*r);
        }
        let count = left.len() as u32;
        for i in 0..count.saturating_sub(1) {
            let at = first + i * 2;
            self.quad(at, at + 1, at + 3, at + 2);
        }
        if closed && count > 2 {
            let last = first + (count - 1) * 2;
            self.quad(last, last + 1, first + 1, first);
        }
        self.side = keep;
        self
    }

    /// A height field over x and z, centred on the origin and facing up: `height`
    /// gives y at each (x, z). `columns` and `rows` count cells.
    pub fn grid(
        &mut self,
        width: f32,
        depth: f32,
        columns: u32,
        rows: u32,
        mut height: impl FnMut(f32, f32) -> f32,
    ) -> &mut Self {
        let keep = self.side;
        self.facing(Vector3::UP);
        let (columns, rows) = (columns.max(1), rows.max(1));
        let first = self.positions.len() as u32;
        for row in 0..=rows {
            for column in 0..=columns {
                let (u, v) = (column as f32 / columns as f32, row as f32 / rows as f32);
                let (x, z) = ((u - 0.5) * width, (v - 0.5) * depth);
                self.uv(u, v);
                self.vertex(Vector3::new(x, height(x, z), z));
            }
        }
        let stride = columns + 1;
        for row in 0..rows {
            for column in 0..columns {
                let at = first + row * stride + column;
                self.quad(at, at + 1, at + stride + 1, at + stride);
            }
        }
        self.side = keep;
        self
    }

    /// Copy a whole geometry in, moved by `transform`: how many meshes become one
    /// mesh and one draw. It keeps its normals, uvs and colours. A mirroring
    /// transform turns a mesh inside out, so its triangles are turned back.
    pub fn append(&mut self, geometry: &BufferGeometry, transform: &Matrix4) -> &mut Self {
        let Some(positions) = geometry.get_attribute("position") else {
            return self;
        };
        let normals = geometry.get_attribute("normal");
        let uvs = geometry.get_attribute("uv");
        let colors = geometry.get_attribute("color");
        let e = &transform.elements;
        let cofactor = |x: Vector3| {
            let (c0, c1, c2) = (
                Vector3::new(e[0], e[1], e[2]),
                Vector3::new(e[4], e[5], e[6]),
                Vector3::new(e[8], e[9], e[10]),
            );
            // The inverse transpose, up to scale: right under any scaling.
            (c1.cross(c2) * x.x + c2.cross(c0) * x.y + c0.cross(c1) * x.z).normalize()
        };
        let mirrored = {
            let (c0, c1, c2) = (
                Vector3::new(e[0], e[1], e[2]),
                Vector3::new(e[4], e[5], e[6]),
                Vector3::new(e[8], e[9], e[10]),
            );
            c0.cross(c1).dot(c2) < 0.0
        };
        let sign = if mirrored { -1.0 } else { 1.0 };
        let first = self.positions.len() as u32;
        let read = |attribute: &BufferAttribute, i: usize, size: usize| -> [f32; 3] {
            let at = i * attribute.item_size;
            let mut out = [0.0; 3];
            out[..size].copy_from_slice(&attribute.array[at..at + size]);
            out
        };
        for i in 0..positions.count() {
            let [x, y, z] = read(positions, i, 3);
            self.positions
                .push(Vector3::new(x, y, z).apply_matrix4(transform));
            self.normals.push(normals.map(|n| {
                let [x, y, z] = read(n, i, 3);
                cofactor(Vector3::new(x, y, z)) * sign
            }));
            self.uvs.push(uvs.map_or([0.0; 2], |uv| {
                let [u, v, _] = read(uv, i, 2);
                [u, v]
            }));
            self.colors
                .push(colors.map(|c| read(c, i, 3)).or(self.next_color));
        }
        for t in 0..geometry.triangle_count() {
            let [a, b, c] = geometry.triangle(t).map(|index| index + first);
            self.triangles
                .push(if mirrored { [a, c, b] } else { [a, b, c] });
        }
        self
    }

    /// The finished geometry, sharing vertices, so a smooth surface shades smooth.
    pub fn build(&self) -> BufferGeometry {
        let mut geometry = self.assemble(&self.smooth_normals());
        geometry.compute_bounding_box();
        geometry.compute_bounding_sphere();
        geometry
    }

    /// The finished geometry with each triangle shaded flat, as cut facets are.
    pub fn build_flat(&self) -> BufferGeometry {
        let mut flat = MeshBuilder::new();
        for [a, b, c] in &self.triangles {
            let [a, b, c] = [*a as usize, *b as usize, *c as usize];
            let (p, q, r) = (self.positions[a], self.positions[b], self.positions[c]);
            let normal = (q - p).cross(r - p).normalize();
            for i in [a, b, c] {
                flat.positions.push(self.positions[i]);
                flat.normals.push(Some(normal));
                flat.uvs.push(self.uvs[i]);
                flat.colors.push(self.colors[i]);
            }
            let at = flat.positions.len() as u32;
            flat.triangles.push([at - 3, at - 2, at - 1]);
        }
        flat.build()
    }

    fn smooth_normals(&self) -> Vec<Vector3> {
        let mut sums = vec![Vector3::ZERO; self.positions.len()];
        for &[a, b, c] in &self.triangles {
            let [a, b, c] = [a as usize, b as usize, c as usize];
            // Unnormalised, so a bigger face counts for more.
            let face = (self.positions[b] - self.positions[a])
                .cross(self.positions[c] - self.positions[a]);
            sums[a] = sums[a] + face;
            sums[b] = sums[b] + face;
            sums[c] = sums[c] + face;
        }
        sums.iter()
            .zip(&self.normals)
            .map(|(sum, set)| match set {
                Some(normal) => *normal,
                None if sum.length() > 0.0 => sum.normalize(),
                None => Vector3::UP,
            })
            .collect()
    }

    fn assemble(&self, normals: &[Vector3]) -> BufferGeometry {
        let flatten = |vectors: &[Vector3]| {
            vectors
                .iter()
                .flat_map(|v| [v.x, v.y, v.z])
                .collect::<Vec<f32>>()
        };
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute(
            "position",
            BufferAttribute::new(flatten(&self.positions), 3),
        );
        geometry.set_attribute("normal", BufferAttribute::new(flatten(normals), 3));
        let uvs = self.uvs.iter().flatten().copied().collect();
        geometry.set_attribute("uv", BufferAttribute::new(uvs, 2));
        if self.colors.iter().any(Option::is_some) {
            let colors = self
                .colors
                .iter()
                .flat_map(|c| c.unwrap_or([1.0; 3]))
                .collect();
            geometry.set_attribute("color", BufferAttribute::new(colors, 3));
        }
        geometry.set_index(self.triangles.iter().flatten().copied().collect());
        geometry
    }
}

impl From<MeshBuilder> for BufferGeometry {
    fn from(builder: MeshBuilder) -> Self {
        builder.build()
    }
}

impl From<&MeshBuilder> for BufferGeometry {
    fn from(builder: &MeshBuilder) -> Self {
        builder.build()
    }
}

/// What is wrong with a mesh, if anything. Wrong meshes mostly look like nothing
/// at all, so this says why.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MeshReport {
    pub triangles: usize,
    /// Triangles with no area: they draw nothing and have no normal.
    pub degenerate: usize,
    /// Edges on only one triangle: the rim of a sheet, or a hole in a solid.
    pub open_edges: usize,
    /// Edges two triangles run along the same way, so that one of them is
    /// facing the wrong way.
    pub flipped_edges: usize,
    /// Edges shared by more than two triangles.
    pub crowded_edges: usize,
}

impl MeshReport {
    /// Vertices at one position count as one, so a box with hard edges, which
    /// repeats its corners, is still closed.
    pub fn of(geometry: &BufferGeometry) -> Self {
        let mut report = Self {
            triangles: geometry.triangle_count(),
            ..Self::default()
        };
        let Some(positions) = geometry.positions() else {
            return report;
        };
        let mut ids = HashMap::new();
        let mut welded = Vec::new();
        let mut points = Vec::new();
        for p in positions {
            let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
            let next = ids.len() as u32;
            welded.push(*ids.entry(key).or_insert(next));
            points.push(p);
        }
        // Each undirected edge, and how often it runs each way.
        let mut edges: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for t in 0..geometry.triangle_count() {
            let [a, b, c] = geometry.triangle(t);
            let area = (points[b as usize] - points[a as usize])
                .cross(points[c as usize] - points[a as usize])
                .length();
            if area <= f32::EPSILON {
                report.degenerate += 1;
                continue;
            }
            let [a, b, c] = [a, b, c].map(|i| welded[i as usize]);
            for (from, to) in [(a, b), (b, c), (c, a)] {
                let counts = edges.entry((from.min(to), from.max(to))).or_default();
                if from < to {
                    counts.0 += 1;
                } else {
                    counts.1 += 1;
                }
            }
        }
        for &(forward, back) in edges.values() {
            match forward + back {
                1 => report.open_edges += 1,
                2 if forward == 2 || back == 2 => report.flipped_edges += 1,
                2 => {}
                _ => report.crowded_edges += 1,
            }
        }
        report
    }

    /// Whether it is a closed, consistently facing solid.
    pub fn is_solid(&self) -> bool {
        self.triangles > 0
            && self.degenerate == 0
            && self.open_edges == 0
            && self.flipped_edges == 0
            && self.crowded_edges == 0
    }
}
