use super::BufferAttribute;
use crate::math::{Box3, Sphere, Vector3};
use std::collections::HashMap;

#[cfg(any(feature = "mesh-bvh", feature = "brep"))]
use std::sync::Arc;

/// A collection of named vertex attributes plus an optional index buffer.
/// Mirrors three.js's `BufferGeometry`: `geometry.setAttribute("position", ...)`,
/// `geometry.setIndex(...)`.
#[derive(Debug, Clone, Default)]
pub struct BufferGeometry {
    pub attributes: HashMap<String, BufferAttribute>,
    pub index: Option<Vec<u32>>,
    pub bounding_box: Option<Box3>,
    pub bounding_sphere: Option<Sphere>,
    /// Bumped whenever attributes/index change so the renderer re-uploads GPU
    /// buffers and the [derived-data cache](super::DerivedCache) rebuilds.
    /// After writing the public fields directly, call
    /// [`bump_version`](Self::bump_version).
    pub geometry_version: u32,
    /// Give this geometry's GPU vertex buffer `STORAGE` usage, so a compute pass
    /// can write it directly.
    ///
    /// The point is vertex animation that never round-trips through the CPU:
    /// upload the rest pose once, then let a compute shader rewrite positions and
    /// normals in place each frame. Nothing bumps
    /// [`geometry_version`](Self::geometry_version), so the renderer never
    /// re-uploads and the CPU-side attribute arrays simply stop being the truth.
    ///
    /// Pair it with [`Renderer::vertex_buffer`](crate::Renderer::vertex_buffer).
    /// Off by default: `STORAGE` is not free on every backend, and only a
    /// geometry that is actually driven this way should ask for it.
    pub gpu_writable: bool,
    /// Optional BVH acceleration structure (three-mesh-bvh `boundsTree`).
    #[cfg(feature = "mesh-bvh")]
    pub bounds_tree: Option<Arc<crate::mesh_bvh::MeshBvh>>,
    /// Optional analytic surface provenance: which surface each triangle was
    /// sampled from. See [`crate::brep`].
    ///
    /// Always optional and always droppable. It is invalidated by any write to
    /// the positions or the index, for the same reason `bounds_tree` is: a
    /// sidecar describing geometry that has since changed is worse than no
    /// sidecar, because consumers trust it.
    #[cfg(feature = "brep")]
    pub surfaces: Option<Arc<crate::brep::SurfaceTable>>,
    /// Optional polygon grouping: how many consecutive triangles, in index
    /// order, each source polygon became. `[2, 1, 3]` says triangles 0–1 were
    /// a quad, triangle 2 a triangle, and triangles 3–5 a pentagon.
    ///
    /// Counted in triangles rather than in corners because the triangle count
    /// is the one thing this geometry can check it against: the entries must
    /// sum to [`triangle_count`](Self::triangle_count), whatever triangulation
    /// produced them. A polygon with `n` corners and no interior points is
    /// `n - 2` triangles under any triangulation, so the corner count is
    /// recoverable where it means anything — see
    /// [`polygon_loop`](Self::polygon_loop).
    ///
    /// Always optional and always droppable. [`set_index`](Self::set_index)
    /// clears it, and so does a `position` write that changes the vertex
    /// count — which for unindexed geometry is the topology. Writes that keep
    /// the vertex count (a deformation, a bind transform, recomputed normals,
    /// new UVs) keep it, because the grouping describes connectivity and none
    /// of those change connectivity. Read it through
    /// [`polygon_counts`](Self::polygon_counts), which also refuses a stale
    /// table left behind by a direct write to the public `index` field.
    pub polygons: Option<Vec<u32>>,
    /// three.js `geometry.groups`: ranges of the index (or of the vertices,
    /// unindexed) drawn with `material[material_index]`. Generators that
    /// three.js gives groups — `ExtrudeGeometry`'s caps and sides,
    /// `ShapeGeometry`'s shapes, `BoxGeometry`'s faces — fill this; nothing
    /// clears it, as in three.js.
    pub groups: Vec<GeometryGroup>,
    /// three.js `geometry.drawRange`: the part of the index (or of the
    /// vertices, unindexed) every renderer draws and the raycaster tests. The
    /// default draws everything. Read each frame, like three.js's, so changing
    /// it needs no version bump. See [`DrawRange`].
    pub draw_range: DrawRange,
    /// three.js `geometry.morphAttributes`: per attribute name (`"position"`,
    /// `"normal"`, `"color"`), one attribute per morph target, each with the
    /// base attribute's vertex count. Absolute target values unless
    /// [`morph_targets_relative`](Self::morph_targets_relative). A mesh blends
    /// them with its `morph_influences` — on the GPU in the raster renderers,
    /// on the CPU (see [`Morph`](super::Morph)) everywhere else.
    ///
    /// After writing this directly call [`bump_version`](Self::bump_version)
    /// so the renderers rebuild their morph textures, or use
    /// [`set_morph_attribute`](Self::set_morph_attribute).
    pub morph_attributes: HashMap<String, Vec<BufferAttribute>>,
    /// three.js `geometry.morphTargetsRelative`: the targets are offsets added
    /// to the base rather than absolute values blended towards.
    pub morph_targets_relative: bool,
    /// The morph targets' names, by target index — three.js keeps these on
    /// each morph attribute's `name`. `Mesh::update_morph_targets` builds the
    /// mesh's `morph_target_dictionary` from them (a missing name is the
    /// index, as in three.js).
    pub morph_target_names: Vec<String>,
    /// Data derived on first use — bounding volumes when none are stored, the
    /// polygon prefix sums behind
    /// [`polygon_of_triangle`](Self::polygon_of_triangle) — and rebuilt when
    /// [`geometry_version`](Self::geometry_version) changes. Filled through
    /// `&self`, so a geometry shared behind an `Arc` caches too. See
    /// [`DerivedCache`](super::DerivedCache).
    pub derived: super::DerivedCache,
}

/// One entry of [`BufferGeometry::groups`] — three.js's
/// `{ start, count, materialIndex }`, counted in index entries (vertices when
/// unindexed), not triangles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryGroup {
    pub start: u32,
    pub count: u32,
    pub material_index: u32,
}

/// three.js `BufferGeometry.drawRange`: `{ start, count }` in index entries
/// (vertices when unindexed). `count` [`u32::MAX`] is three.js's `Infinity`,
/// the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawRange {
    pub start: u32,
    pub count: u32,
}

impl DrawRange {
    /// Everything: three.js's `{ start: 0, count: Infinity }`.
    pub const ALL: Self = Self {
        start: 0,
        count: u32::MAX,
    };

    /// The range's end, `start + count`, saturating — so `count: u32::MAX`
    /// is unbounded however large `start` is.
    pub fn end(&self) -> u32 {
        self.start.saturating_add(self.count)
    }
}

impl Default for DrawRange {
    fn default() -> Self {
        Self::ALL
    }
}

/// A process-wide monotonic version stamp. The renderer's GPU-buffer cache is
/// keyed by a geometry's heap *pointer* and re-uploads only when `geometry_version`
/// changes. When scenes are rebuilt every frame (e.g. an animation) a freed
/// geometry's address is often *recycled* for a new one; if versions could
/// coincide, the cache would false-hit and draw the freed geometry's stale
/// buffers ("ghost" meshes). Drawing every version from one global counter makes
/// every stamp unique, so a recycled address can never false-match.
fn next_geometry_version() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static VERSION: AtomicU32 = AtomicU32::new(1);
    VERSION.fetch_add(1, Ordering::Relaxed)
}

impl BufferGeometry {
    pub fn new() -> Self {
        Self {
            geometry_version: next_geometry_version(),
            ..Self::default()
        }
    }

    /// Record a change made by writing the public fields directly — an edit
    /// of an attribute's values in place, a new `index`, a new `polygons`
    /// table. three.js's `needsUpdate = true`, for the whole geometry: the
    /// renderers re-upload and the [derived-data cache](super::DerivedCache)
    /// rebuilds. Without it neither sees a change that keeps the arrays'
    /// lengths.
    ///
    /// Bumps the version and nothing else: a stored
    /// [`bounding_sphere`](Self::bounding_sphere) or
    /// [`bounding_box`](Self::bounding_box) stays as it is, as three.js's
    /// does, and so does a BVH.
    pub fn bump_version(&mut self) -> &mut Self {
        self.geometry_version = next_geometry_version();
        self
    }

    pub fn set_attribute(&mut self, name: impl Into<String>, attr: BufferAttribute) -> &mut Self {
        let name = name.into();
        if name == "position"
            && self.attributes.get("position").map(|a| a.count()) != Some(attr.count())
        {
            self.polygons = None;
        }
        self.attributes.insert(name, attr);
        // Invalidate cached bounds — they depend on positions.
        self.bounding_box = None;
        self.bounding_sphere = None;
        self.geometry_version = next_geometry_version();
        #[cfg(feature = "mesh-bvh")]
        {
            self.bounds_tree = None;
        }
        #[cfg(feature = "brep")]
        {
            self.surfaces = None;
        }
        self
    }

    pub fn get_attribute(&self, name: &str) -> Option<&BufferAttribute> {
        self.attributes.get(name)
    }

    /// three.js `addGroup(start, count, materialIndex)`.
    pub fn add_group(&mut self, start: u32, count: u32, material_index: u32) -> &mut Self {
        self.groups.push(GeometryGroup {
            start,
            count,
            material_index,
        });
        self
    }

    /// three.js `clearGroups()`.
    pub fn clear_groups(&mut self) -> &mut Self {
        self.groups.clear();
        self
    }

    /// three.js `setDrawRange(start, count)`. `count` [`u32::MAX`] draws to
    /// the end.
    pub fn set_draw_range(&mut self, start: u32, count: u32) -> &mut Self {
        self.draw_range = DrawRange { start, count };
        self
    }

    /// The part of the index (vertices, unindexed) a draw covers, as three.js's
    /// `WebGLRenderer.renderBufferDirect` computes it for triangles: the draw
    /// range, cut to `group` when there is one, then clamped to the index (or
    /// position) count. `start..end`, `None` when that is empty.
    ///
    /// The raycaster walks the same span — three.js's `Mesh.raycast` derives
    /// it identically — so what is hit is what is drawn.
    pub fn draw_span(&self, group: Option<&GeometryGroup>) -> Option<(u32, u32)> {
        let mut start = self.draw_range.start;
        let mut end = self.draw_range.end();
        if let Some(g) = group {
            start = start.max(g.start);
            end = end.min(g.start.saturating_add(g.count));
        }
        end = end.min(self.draw_count() as u32);
        (start < end).then_some((start, end))
    }

    /// Replace one entry of [`morph_attributes`](Self::morph_attributes) — the
    /// targets for `name` (`"position"`, `"normal"` or `"color"`) — and bump
    /// the version so the renderers rebuild their morph data.
    pub fn set_morph_attribute(
        &mut self,
        name: impl Into<String>,
        targets: Vec<BufferAttribute>,
    ) -> &mut Self {
        self.morph_attributes.insert(name.into(), targets);
        self.geometry_version = next_geometry_version();
        self
    }

    /// How many morph targets the geometry carries: the length of its
    /// longest morph attribute list.
    pub fn morph_target_count(&self) -> usize {
        self.morph_attributes
            .values()
            .map(|v| v.len())
            .max()
            .unwrap_or(0)
    }

    /// three.js `toNonIndexed`: a copy whose every attribute (and morph
    /// attribute) is expanded so each index becomes its own vertex — a
    /// triangle soup. Groups carry over; they count indices, which are now
    /// vertices. Already non-indexed, it is a plain copy.
    pub fn to_non_indexed(&self) -> BufferGeometry {
        let mut out = self.clone();
        out.expand_index();
        out
    }

    /// [`to_non_indexed`](Self::to_non_indexed) in place.
    pub fn expand_index(&mut self) {
        let Some(indices) = self.index.take() else {
            return;
        };
        let expand = |attr: &mut BufferAttribute| {
            let size = attr.item_size;
            let mut data = Vec::with_capacity(indices.len() * size);
            for &vi in &indices {
                let src = vi as usize * size;
                data.extend_from_slice(&attr.array[src..src + size]);
            }
            attr.array = data;
        };
        self.attributes.values_mut().for_each(expand);
        for targets in self.morph_attributes.values_mut() {
            targets.iter_mut().for_each(expand);
        }
        self.bounding_box = None;
        self.bounding_sphere = None;
        #[cfg(feature = "mesh-bvh")]
        {
            self.bounds_tree = None;
        }
        self.bump_version();
    }

    pub fn set_index(&mut self, indices: Vec<u32>) -> &mut Self {
        self.index = Some(indices);
        self.geometry_version = next_geometry_version();
        self.polygons = None;
        #[cfg(feature = "mesh-bvh")]
        {
            self.bounds_tree = None;
        }
        #[cfg(feature = "brep")]
        {
            self.surfaces = None;
        }
        self
    }

    /// Attach analytic surface provenance.
    ///
    /// Rejected — leaving the geometry untagged — unless the table describes
    /// exactly this triangle count. Silently accepting a mismatched table would
    /// hand every consumer an out-of-range index.
    ///
    /// Call this *after* the last `set_attribute` / `set_index`, both of which
    /// clear it.
    #[cfg(feature = "brep")]
    pub fn set_surfaces(&mut self, table: crate::brep::SurfaceTable) -> &mut Self {
        if table.matches(self) {
            self.surfaces = Some(Arc::new(table));
        }
        self
    }

    /// The provenance table, if one is attached and still describes this
    /// geometry.
    #[cfg(feature = "brep")]
    pub fn surface_table(&self) -> Option<&crate::brep::SurfaceTable> {
        self.surfaces.as_deref().filter(|t| t.matches(self))
    }

    /// Total draw count: index count if indexed, otherwise position count.
    pub fn draw_count(&self) -> usize {
        if let Some(idx) = &self.index {
            idx.len()
        } else {
            self.attributes
                .get("position")
                .map(|a| a.count())
                .unwrap_or(0)
        }
    }

    /// How many whole triangles the geometry draws: `draw_count() / 3`.
    pub fn triangle_count(&self) -> usize {
        self.draw_count() / 3
    }

    /// The vertex indices of triangle `t`, indexed or not. `t` must be below
    /// [`triangle_count`](Self::triangle_count).
    pub fn triangle(&self, t: usize) -> [u32; 3] {
        match &self.index {
            Some(idx) => [idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]],
            None => {
                let at = (t * 3) as u32;
                [at, at + 1, at + 2]
            }
        }
    }

    /// Attach polygon grouping: how many consecutive triangles each source
    /// polygon became. See [`polygons`](Self::polygons).
    ///
    /// Rejected — leaving the geometry ungrouped — unless every entry is at
    /// least one and they sum to exactly this geometry's triangle count. A
    /// table that does not cover the triangles would put every later polygon
    /// boundary in the wrong place.
    ///
    /// Call this *after* the last `set_index` and the last `position` write,
    /// which clear it.
    ///
    /// Bumps [`geometry_version`](Self::geometry_version) when the grouping
    /// actually changes: the renderers build the wireframe line list from it,
    /// so a geometry that is already on the GPU has to re-upload to show the
    /// new outlines.
    pub fn set_polygons(&mut self, triangles_per_polygon: Vec<u32>) -> &mut Self {
        let valid = triangles_per_polygon.iter().all(|&n| n > 0)
            && triangles_per_polygon
                .iter()
                .map(|&n| n as usize)
                .sum::<usize>()
                == self.triangle_count();
        let polygons = valid.then_some(triangles_per_polygon);
        if polygons != self.polygons {
            self.polygons = polygons;
            self.geometry_version = next_geometry_version();
        }
        self
    }

    /// The polygon grouping, if one is attached and still covers exactly this
    /// geometry's triangles.
    pub fn polygon_counts(&self) -> Option<&[u32]> {
        self.polygons
            .as_deref()
            .filter(|p| p.iter().map(|&n| n as usize).sum::<usize>() == self.triangle_count())
    }

    /// Each polygon's triangles, as a range of triangle numbers, in order.
    ///
    /// Without a grouping every triangle is its own polygon, so a consumer can
    /// walk polygons unconditionally and get today's behaviour for free.
    pub fn polygon_ranges(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        let counts = self.polygon_counts();
        let singles = if counts.is_none() {
            self.triangle_count()
        } else {
            0
        };
        let grouped = counts.unwrap_or(&[]).iter().scan(0usize, |at, &n| {
            let start = *at;
            *at += n as usize;
            Some(start..*at)
        });
        grouped.chain((0..singles).map(|t| t..t + 1))
    }

    /// Which polygon each triangle belongs to. The identity when there is no
    /// grouping.
    pub fn triangle_polygon_ids(&self) -> Vec<u32> {
        let mut ids = Vec::with_capacity(self.triangle_count());
        for (p, range) in self.polygon_ranges().enumerate() {
            ids.extend(std::iter::repeat_n(p as u32, range.len()));
        }
        ids
    }

    /// What identifies each corner when matching edges between triangles: the
    /// position's bits where there is a usable position, else the index. `-0`
    /// and `0` are one corner, as they are to three.js, which keys edges on the
    /// coordinates' decimal strings.
    ///
    /// Position rather than index because unindexed geometry — and any loader
    /// that splits vertices per corner — gives the two triangles either side of
    /// a fan diagonal different indices for the same point.
    ///
    /// Returned as a lookup so the `position` attribute is found once per
    /// walk rather than once per corner: a wireframe build keys six corners a
    /// triangle.
    pub(crate) fn corner_keys(&self) -> impl Fn(u32) -> [u32; 3] + '_ {
        let canonical = |v: f32| if v == 0.0 { 0u32 } else { v.to_bits() };
        let positions = self
            .attributes
            .get("position")
            .filter(|p| p.item_size == 3)
            .map(|p| p.array.as_slice());
        move |vertex| match positions {
            Some(p) if (vertex as usize) * 3 + 3 <= p.len() => {
                let at = vertex as usize * 3;
                [canonical(p[at]), canonical(p[at + 1]), canonical(p[at + 2])]
            }
            _ => [vertex, u32::MAX, u32::MAX],
        }
    }

    /// The directed edges of a polygon's triangles that no other triangle of
    /// the same polygon shares — its outline, in triangle winding order — as
    /// `(from key, to key, from vertex, to vertex)`, written into `directed`.
    /// `uses` is scratch; both are cleared first, so a caller walking every
    /// polygon allocates them once.
    #[allow(clippy::type_complexity)]
    fn polygon_boundary_into(
        &self,
        key: &impl Fn(u32) -> [u32; 3],
        triangles: std::ops::Range<usize>,
        directed: &mut Vec<([u32; 3], [u32; 3], u32, u32)>,
        uses: &mut HashMap<([u32; 3], [u32; 3]), u32>,
    ) {
        directed.clear();
        uses.clear();
        for t in triangles {
            let tri = self.triangle(t);
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let (ka, kb) = (key(a), key(b));
                directed.push((ka, kb, a, b));
                *uses.entry(unordered(ka, kb)).or_insert(0) += 1;
            }
        }
        directed.retain(|&(ka, kb, _, _)| uses[&unordered(ka, kb)] == 1);
    }

    /// Calls `emit(a, b)` for every polygon outline edge, each edge once, in
    /// polygon order. See [`polygon_edges`](Self::polygon_edges).
    fn for_each_polygon_edge(&self, emit: impl FnMut(u32, u32)) {
        self.for_each_polygon_edge_in(0..usize::MAX, emit);
    }

    /// [`for_each_polygon_edge`](Self::for_each_polygon_edge) over the
    /// polygons lying wholly inside the triangle range `triangles`.
    fn for_each_polygon_edge_in(
        &self,
        triangles: std::ops::Range<usize>,
        mut emit: impl FnMut(u32, u32),
    ) {
        let key = self.corner_keys();
        let mut seen = std::collections::HashSet::new();
        let (mut directed, mut uses) = (Vec::new(), HashMap::new());
        for range in self
            .polygon_ranges()
            .filter(|r| r.start >= triangles.start && r.end <= triangles.end)
        {
            self.polygon_boundary_into(&key, range, &mut directed, &mut uses);
            for &(ka, kb, a, b) in &directed {
                if seen.insert(unordered(ka, kb)) {
                    emit(a, b);
                }
            }
        }
    }

    /// Every polygon's outline edges as vertex-index pairs, each edge once.
    ///
    /// An edge inside a polygon — a fan diagonal — is shared by two of that
    /// polygon's triangles and is left out; an edge two polygons share is
    /// reported once. Edges are matched by position (see the note on
    /// unindexed geometry at [`polygon_loop`](Self::polygon_loop)), so a seam
    /// that splits vertices still draws one line. Without a grouping this is
    /// every triangle edge, once.
    pub fn polygon_edges(&self) -> Vec<[u32; 2]> {
        let mut edges = Vec::new();
        self.for_each_polygon_edge(|a, b| edges.push([a, b]));
        edges
    }

    /// One polygon's corners, in order, recovered from its triangles.
    ///
    /// `triangles` is one item of [`polygon_ranges`](Self::polygon_ranges).
    /// The loop follows the triangles' winding and starts at the first corner
    /// of the first triangle, so a polygon fanned from corner 0 — what every
    /// loader in this crate does — comes back exactly as authored, corner 0
    /// first.
    ///
    /// The recovery is combinatorial, not geometric: it chains the edges that
    /// only one of the polygon's triangles uses. So it does not care whether
    /// the polygon was planar or convex, or how it was triangulated. It
    /// returns `None` — and the caller should fall back to the triangles —
    /// when those edges do not form one simple loop with `triangles + 2`
    /// corners: a polygon with a hole, with an interior point, with a corner
    /// repeated, or whose triangles do not share edges at all.
    ///
    /// Corners are matched by position, so in unindexed geometry — where one
    /// corner is a different vertex in each triangle that uses it — the index
    /// returned for a corner is the one in the triangle whose outline edge
    /// leaves it; all of them sit at the same point. Two distinct corners of
    /// one polygon at the same position (a degenerate polygon) read as one.
    pub fn polygon_loop(&self, triangles: std::ops::Range<usize>) -> Option<Vec<u32>> {
        if triangles.is_empty() || triangles.end > self.triangle_count() {
            return None;
        }
        let key = self.corner_keys();
        let first = key(self.triangle(triangles.start)[0]);
        let expected = triangles.len() + 2;
        let mut boundary = Vec::with_capacity(triangles.len() * 3);
        self.polygon_boundary_into(&key, triangles, &mut boundary, &mut HashMap::new());
        if boundary.len() != expected {
            return None;
        }
        let mut next: HashMap<[u32; 3], ([u32; 3], u32)> = HashMap::new();
        for (ka, kb, a, _) in &boundary {
            if next.insert(*ka, (*kb, *a)).is_some() {
                // Two outline edges leave the same corner: not a simple loop.
                return None;
            }
        }
        let start = if next.contains_key(&first) {
            first
        } else {
            boundary[0].0
        };
        let mut corners = Vec::with_capacity(expected);
        let mut at = start;
        loop {
            let (to, vertex) = *next.get(&at)?;
            corners.push(vertex);
            at = to;
            if at == start || corners.len() > expected {
                break;
            }
        }
        (at == start && corners.len() == expected).then_some(corners)
    }

    /// The faces a polygon-capable writer (USD, OBJ, OFF, PLY) should emit, in
    /// order: one corner loop per polygon whose outline
    /// [`polygon_loop`](Self::polygon_loop) recovers, and that polygon's
    /// triangles one by one where it does not — so a face is never lost, only
    /// split. Without a grouping every triangle is its own face.
    ///
    /// Corners are vertex indices (for unindexed geometry, the sequential
    /// ones), so a writer can emit them against the geometry's own vertices.
    pub fn faces(&self) -> impl Iterator<Item = Vec<u32>> + '_ {
        let grouped = self.polygon_counts().is_some();
        self.polygon_ranges().flat_map(move |range| {
            if grouped && range.len() > 1 {
                if let Some(corners) = self.polygon_loop(range.clone()) {
                    return vec![corners];
                }
            }
            range.map(|t| self.triangle(t).to_vec()).collect()
        })
    }

    /// Which polygon each of `triangles` belongs to, or `None` for all of them
    /// when there is no grouping (or a stale one — the
    /// [`polygon_counts`](Self::polygon_counts) rule). A triangle number past
    /// the end maps to `None`.
    ///
    /// One pass over the grouping however many triangles are asked about — the
    /// walk up to the last one asked for doubles as the first part of the
    /// coverage check — so a caller with a handful of hits, a raycast, pays
    /// O(polygons) once rather than per hit. Nothing is cached: the geometry is
    /// shared behind an `Arc` and immutable while it is read. On a 2M-triangle,
    /// 1M-quad grid this measured 0.07–0.2 ms on an M-series Mac — bound by
    /// reading the 4 MB table once — so it dominates a BVH raycast
    /// (sub-microsecond on its own) on such a mesh, but not a pick.
    pub fn polygons_of_triangles(&self, triangles: &[usize]) -> Vec<Option<usize>> {
        let mut out = vec![None; triangles.len()];
        let Some(counts) = self.polygons.as_deref() else {
            return out;
        };
        let mut order: Vec<usize> = (0..triangles.len()).collect();
        order.sort_unstable_by_key(|&i| triangles[i]);
        // `end` is where polygon `polygon` starts. Whole blocks are skipped on
        // their (vectorised) sums, then single polygons.
        const BLOCK: usize = 64;
        let (mut end, mut polygon) = (0usize, 0usize);
        for &i in &order {
            let t = triangles[i];
            while polygon + BLOCK <= counts.len() {
                let block: usize = counts[polygon..polygon + BLOCK]
                    .iter()
                    .map(|&n| n as usize)
                    .sum();
                if end + block > t {
                    break;
                }
                end += block;
                polygon += BLOCK;
            }
            while polygon < counts.len() && end + counts[polygon] as usize <= t {
                end += counts[polygon] as usize;
                polygon += 1;
            }
            if polygon < counts.len() {
                out[i] = Some(polygon);
            }
        }
        let total = end + counts[polygon..].iter().map(|&n| n as usize).sum::<usize>();
        if total != self.triangle_count() {
            out.iter_mut().for_each(|p| *p = None);
        }
        out
    }

    /// Line-list indices for drawing this geometry as a wireframe.
    ///
    /// Without a polygon grouping this is three.js's `WebGLGeometries`
    /// wireframe exactly: each triangle emits `(a,b, b,c, c,a)`, nothing
    /// deduplicated. With one, it is [`polygon_edges`](Self::polygon_edges):
    /// polygon outlines only, so a quad mesh draws as quads instead of showing
    /// every fan diagonal.
    pub fn wireframe_indices(&self) -> Vec<u32> {
        let mut flat = Vec::with_capacity(self.triangle_count() * 6);
        if self.polygon_counts().is_some() {
            self.for_each_polygon_edge(|a, b| flat.extend_from_slice(&[a, b]));
            return flat;
        }
        for t in 0..self.triangle_count() {
            let [a, b, c] = self.triangle(t);
            flat.extend_from_slice(&[a, b, b, c, c, a]);
        }
        flat
    }

    /// [`wireframe_indices`](Self::wireframe_indices) for one draw span,
    /// `start..end` of the index (vertices, unindexed) — what a group or a
    /// draw range of a `wireframe: true` mesh draws. Without a polygon table
    /// this is three.js's wireframe of exactly the triangles the span's
    /// triangle draw rasterises (`j = start, start + 3, …`, whole triangles
    /// inside the span); with one, the outlines of the polygons lying wholly
    /// inside those triangles.
    pub fn wireframe_indices_in(&self, start: u32, end: u32) -> Vec<u32> {
        let end = (end as usize).min(self.draw_count());
        let start = start as usize;
        let mut flat = Vec::new();
        if start >= end {
            return flat;
        }
        if self.polygon_counts().is_some() && start.is_multiple_of(3) {
            self.for_each_polygon_edge_in(start / 3..end.div_ceil(3), |a, b| {
                flat.extend_from_slice(&[a, b])
            });
            return flat;
        }
        let corner = |j: usize| -> u32 {
            match &self.index {
                Some(idx) => idx[j],
                None => j as u32,
            }
        };
        let mut j = start;
        while j + 3 <= end {
            let (a, b, c) = (corner(j), corner(j + 1), corner(j + 2));
            flat.extend_from_slice(&[a, b, b, c, c, a]);
            j += 3;
        }
        flat
    }

    /// Iterate position vertices as `Vector3` (item_size must be 3).
    pub fn positions(&self) -> Option<impl Iterator<Item = Vector3> + '_> {
        let pos = self.attributes.get("position")?;
        if pos.item_size != 3 {
            return None;
        }
        Some(
            pos.array
                .chunks_exact(3)
                .map(|c| Vector3::new(c[0], c[1], c[2])),
        )
    }

    /// Compute (or refresh) the bounding box from the "position" attribute.
    /// Matches three.js's `computeBoundingBox`.
    pub fn compute_bounding_box(&mut self) -> Box3 {
        let bb = match self.positions() {
            Some(iter) => {
                let mut b = Box3::empty();
                for p in iter {
                    b.expand_by_point(p);
                }
                if b.is_empty() {
                    Box3::new(Vector3::ZERO, Vector3::ZERO)
                } else {
                    b
                }
            }
            None => Box3::new(Vector3::ZERO, Vector3::ZERO),
        };
        self.bounding_box = Some(bb);
        bb
    }

    /// Compute (or refresh) the bounding sphere from positions.
    /// Matches three.js's `computeBoundingSphere`: center on the bounding box
    /// center, then radius = max distance to any vertex.
    pub fn compute_bounding_sphere(&mut self) -> Sphere {
        let bb = self
            .bounding_box
            .unwrap_or_else(|| self.compute_bounding_box());
        let center = bb.center();
        let mut max_r2 = 0.0f32;
        if let Some(iter) = self.positions() {
            for p in iter {
                let d2 = (p - center).length_sq();
                if d2 > max_r2 {
                    max_r2 = d2;
                }
            }
        }
        let s = Sphere::new(center, max_r2.sqrt());
        self.bounding_sphere = Some(s);
        s
    }

    /// Build and store a BVH bounds tree on this geometry.
    #[cfg(feature = "mesh-bvh")]
    pub fn compute_bounds_tree(
        &mut self,
        options: crate::mesh_bvh::BuildOptions,
    ) -> Option<Arc<crate::mesh_bvh::MeshBvh>> {
        let bvh = Arc::new(crate::mesh_bvh::MeshBvh::build(self, options)?);
        self.bounds_tree = Some(bvh.clone());
        Some(bvh)
    }

    /// Drop the stored BVH bounds tree.
    #[cfg(feature = "mesh-bvh")]
    pub fn dispose_bounds_tree(&mut self) {
        self.bounds_tree = None;
    }
}

/// An edge's corner keys in a fixed order, so `a→b` and `b→a` match.
fn unordered(a: [u32; 3], b: [u32; 3]) -> ([u32; 3], [u32; 3]) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_positions() -> BufferAttribute {
        BufferAttribute::new(
            vec![
                -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
                1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            3,
        )
    }

    #[test]
    fn bounding_box_of_unit_cube() {
        let mut g = BufferGeometry::new();
        g.set_attribute("position", unit_cube_positions());
        let b = g.compute_bounding_box();
        assert_eq!(b.min, Vector3::new(-1.0, -1.0, -1.0));
        assert_eq!(b.max, Vector3::new(1.0, 1.0, 1.0));
    }

    /// A unit quad fanned from corner 0, plus a separate triangle beside it.
    fn quad_and_triangle() -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0,
                ],
                3,
            ),
        );
        g.set_index(vec![0, 1, 2, 0, 2, 3, 1, 4, 2]);
        g.set_polygons(vec![2, 1]);
        g
    }

    #[test]
    fn polygons_group_triangles() {
        let g = quad_and_triangle();
        assert_eq!(g.polygon_counts(), Some(&[2u32, 1][..]));
        let ranges: Vec<_> = g.polygon_ranges().collect();
        assert_eq!(ranges, [0..2, 2..3]);
        assert_eq!(g.triangle_polygon_ids(), [0, 0, 1]);
        assert_eq!(g.polygon_loop(0..2), Some(vec![0, 1, 2, 3]));
        assert_eq!(g.polygon_loop(2..3), Some(vec![1, 4, 2]));
    }

    #[test]
    fn a_table_that_does_not_cover_the_triangles_is_refused() {
        let mut g = quad_and_triangle();
        g.set_polygons(vec![2, 2]);
        assert!(g.polygons.is_none(), "sums to 4, there are 3");
        g.set_polygons(vec![3, 0]);
        assert!(g.polygons.is_none(), "an empty polygon");
    }

    #[test]
    fn without_polygons_every_triangle_is_one() {
        let mut g = quad_and_triangle();
        g.polygons = None;
        let ranges: Vec<_> = g.polygon_ranges().collect();
        assert_eq!(ranges, [0..1, 1..2, 2..3]);
        // Seven distinct edges, the diagonal among them.
        assert_eq!(g.polygon_edges().len(), 7);
        // three.js's wireframe: six indices a triangle, nothing deduplicated.
        assert_eq!(g.wireframe_indices().len(), 18);
    }

    #[test]
    fn polygon_edges_skip_the_diagonal_and_share_edges_once() {
        let g = quad_and_triangle();
        let mut edges: Vec<[u32; 2]> = g
            .polygon_edges()
            .into_iter()
            .map(|[a, b]| [a.min(b), a.max(b)])
            .collect();
        edges.sort();
        // The quad's four sides and the triangle's two others; 1–2 is shared.
        assert_eq!(edges, [[0, 1], [0, 3], [1, 2], [1, 4], [2, 3], [2, 4]]);
        assert!(!edges.contains(&[0, 2]), "the fan diagonal");
        assert_eq!(g.wireframe_indices().len(), 12);
    }

    #[test]
    fn unindexed_corners_match_by_position() {
        // The same quad as six separate vertices: the diagonal's ends are
        // different indices in each triangle.
        let mut g = BufferGeometry::new();
        let corner = |i: usize| {
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ][i]
        };
        let positions: Vec<f32> = [0, 1, 2, 0, 2, 3].iter().flat_map(|&i| corner(i)).collect();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_polygons(vec![2]);
        assert_eq!(g.polygon_edges().len(), 4);
        assert_eq!(g.polygon_loop(0..2), Some(vec![0, 1, 4, 5]));
    }

    #[test]
    fn a_polygon_with_an_interior_point_has_no_loop() {
        // Four triangles fanned around a centre point: the outline is a quad,
        // but writing it as one would drop the centre.
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.3,
                ],
                3,
            ),
        );
        g.set_index(vec![4, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0]);
        g.set_polygons(vec![4]);
        assert_eq!(g.polygon_edges().len(), 4);
        assert_eq!(g.polygon_loop(0..4), None);
    }

    #[test]
    fn triangles_map_to_their_polygons_in_one_pass() {
        let mut g = quad_and_triangle();
        // Grouping [2, 1]: triangles 0-1 the quad, triangle 2 alone.
        assert_eq!(
            g.polygons_of_triangles(&[2, 0, 1, 3, 0]),
            vec![Some(1), Some(0), Some(0), None, Some(0)]
        );
        g.polygons = None;
        assert_eq!(g.polygons_of_triangles(&[0, 2]), vec![None, None]);
        // A stale table — it no longer covers the triangles — answers nothing.
        g.polygons = Some(vec![1, 1]);
        assert_eq!(g.polygons_of_triangles(&[0]), vec![None]);
    }

    #[test]
    fn faces_are_polygon_loops_or_triangles() {
        let g = quad_and_triangle();
        let faces: Vec<Vec<u32>> = g.faces().collect();
        assert_eq!(faces.len(), 2);
        assert_eq!(faces[0].len(), 4);
        assert_eq!(faces[1], g.triangle(2).to_vec());
        let mut plain = g.clone();
        plain.polygons = None;
        assert_eq!(plain.faces().count(), 3);
    }

    #[test]
    fn set_polygons_bumps_the_version_only_on_a_change() {
        let mut g = quad_and_triangle();
        let v = g.geometry_version;
        g.set_polygons(vec![2, 1]);
        assert_eq!(g.geometry_version, v, "same table, nothing to re-upload");
        g.set_polygons(vec![1, 1, 1]);
        assert_ne!(g.geometry_version, v, "new outlines must re-upload");
        let v = g.geometry_version;
        g.set_polygons(vec![9]);
        assert!(g.polygons.is_none());
        assert_ne!(g.geometry_version, v, "a refused table drops the old one");
    }

    #[test]
    fn set_index_drops_polygons() {
        let mut g = quad_and_triangle();
        g.set_index(vec![0, 1, 2, 0, 2, 3, 1, 4, 2]);
        assert!(g.polygons.is_none());
    }

    #[test]
    fn a_direct_index_write_makes_the_table_stale() {
        let mut g = quad_and_triangle();
        g.index = Some(vec![0, 1, 2, 0, 2, 3]);
        assert!(g.polygons.is_some());
        assert_eq!(g.polygon_counts(), None, "3 triangles described, 2 present");
        assert_eq!(g.polygon_ranges().count(), 2, "one per triangle");
    }

    #[test]
    fn position_writes_drop_polygons_only_when_the_vertex_count_changes() {
        let mut g = quad_and_triangle();
        // Deforming in place, or replacing with as many points, is not a
        // topology change; nor are normals.
        let moved: Vec<f32> = g.attributes["position"]
            .array
            .iter()
            .map(|v| v * 2.0)
            .collect();
        g.set_attribute("position", BufferAttribute::new(moved, 3));
        g.set_attribute("normal", BufferAttribute::new(vec![0.0; 15], 3));
        assert_eq!(g.polygon_counts(), Some(&[2u32, 1][..]));
        // A different number of points is.
        g.set_attribute("position", BufferAttribute::new(vec![0.0; 18], 3));
        assert!(g.polygons.is_none());
    }

    #[test]
    fn merging_carries_polygons_across() {
        let a = quad_and_triangle();
        let mut b = quad_and_triangle();
        b.polygons = None;
        let merged = crate::utils::merge_geometries(&[a, b]).unwrap();
        assert_eq!(merged.polygon_counts(), Some(&[2u32, 1, 1, 1, 1][..]));
        assert_eq!(merged.polygon_loop(3..4), Some(vec![5, 6, 7]));
    }

    #[test]
    fn bounding_sphere_of_unit_cube_radius() {
        let mut g = BufferGeometry::new();
        g.set_attribute("position", unit_cube_positions());
        let s = g.compute_bounding_sphere();
        assert_eq!(s.center, Vector3::ZERO);
        assert!((s.radius - 3.0f32.sqrt()).abs() < 1e-5);
    }
}
