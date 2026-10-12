//! Attribute seams, and the point ids they split vertices into.
//!
//! A corner-table vertex is one position on the surface. A *point* is one unique
//! combination of attribute values, and the two differ wherever an attribute is
//! discontinuous across an edge — a texture atlas seam, or a hard shading edge.
//! One vertex at the corner of a UV chart becomes several points, each with the
//! same position and different texture coordinates, and the index buffer a glTF
//! wants is in points.
//!
//! So each non-position attribute gets its own corner table, sharing the base
//! table's corners but with the seam edges cut. Vertices are then renumbered per
//! attribute, and a final pass walks each base vertex's ring adding a point
//! wherever any attribute's number changes.
//!
//! Note which table each step swings on: finding the first corner of a ring uses
//! the seam-aware `swing_left`, and the walk that follows uses the *base*
//! table's `swing_right`. That asymmetry is Draco's and it is load-bearing —
//! swinging the seam-aware way in both places silently stops at the seam and
//! numbers only part of the ring.

use super::corner_table::{CornerTable, INVALID_CORNER, INVALID_VERTEX};
use super::rans::BitDecoder;
use super::DracoError;

/// One attribute's connectivity: the base table with its seam edges cut.
#[derive(Debug, Clone, Default)]
pub struct AttributeCornerTable {
    /// Per corner: is the edge this corner faces a seam?
    is_edge_on_seam: Vec<bool>,
    /// Per *base* vertex: does any seam touch it?
    is_vertex_on_seam: Vec<bool>,
    corner_to_vertex: Vec<u32>,
    vertex_to_leftmost_corner: Vec<u32>,
    /// Whether any seam is interior rather than a mesh boundary. The predictors
    /// use this to skip seam handling entirely on the common case.
    pub no_interior_seams: bool,
}

impl AttributeCornerTable {
    /// An attribute table over `base` with no seams cut yet.
    pub fn new(base: &CornerTable) -> Self {
        AttributeCornerTable {
            is_edge_on_seam: vec![false; base.num_corners() as usize],
            is_vertex_on_seam: vec![false; base.num_vertices() as usize],
            corner_to_vertex: vec![INVALID_VERTEX; base.num_corners() as usize],
            vertex_to_leftmost_corner: Vec::new(),
            no_interior_seams: true,
        }
    }

    /// Cut the edge that `corner` faces, and the edge facing it.
    pub fn add_seam_edge(&mut self, base: &CornerTable, corner: u32) -> Result<(), DracoError> {
        self.mark_edge(corner)?;
        self.mark_ends(base, corner)?;
        let opp = base.opposite(corner);
        if opp != INVALID_CORNER {
            // A seam with a face on both sides is interior; a boundary edge is
            // a seam too, but it does not cut anything that was joined.
            self.no_interior_seams = false;
            self.mark_edge(opp)?;
            self.mark_ends(base, opp)?;
        }
        Ok(())
    }

    fn mark_edge(&mut self, corner: u32) -> Result<(), DracoError> {
        *self
            .is_edge_on_seam
            .get_mut(corner as usize)
            .ok_or(DracoError::Corrupt)? = true;
        Ok(())
    }

    /// The two vertices at the ends of the edge `corner` faces.
    fn mark_ends(&mut self, base: &CornerTable, corner: u32) -> Result<(), DracoError> {
        for c in [base.next(corner), base.previous(corner)] {
            let v = base.vertex(c);
            if v == INVALID_VERTEX {
                return Err(DracoError::Corrupt);
            }
            *self
                .is_vertex_on_seam
                .get_mut(v as usize)
                .ok_or(DracoError::Corrupt)? = true;
        }
        Ok(())
    }

    pub fn is_corner_opposite_to_seam_edge(&self, corner: u32) -> bool {
        if corner == INVALID_CORNER {
            return false;
        }
        self.is_edge_on_seam
            .get(corner as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Whether the vertex under `corner` has any seam on it.
    pub fn is_corner_on_seam(&self, base: &CornerTable, corner: u32) -> bool {
        let v = base.vertex(corner);
        if v == INVALID_VERTEX {
            return false;
        }
        self.is_vertex_on_seam
            .get(v as usize)
            .copied()
            .unwrap_or(false)
    }

    /// The opposite corner, except that a seam edge has nothing across it.
    pub fn opposite(&self, base: &CornerTable, corner: u32) -> u32 {
        if corner == INVALID_CORNER || self.is_corner_opposite_to_seam_edge(corner) {
            return INVALID_CORNER;
        }
        base.opposite(corner)
    }

    pub fn swing_left(&self, base: &CornerTable, corner: u32) -> u32 {
        base.next(self.opposite(base, base.next(corner)))
    }

    pub fn swing_right(&self, base: &CornerTable, corner: u32) -> u32 {
        base.previous(self.opposite(base, base.previous(corner)))
    }

    pub fn vertex(&self, corner: u32) -> u32 {
        if corner == INVALID_CORNER {
            return INVALID_VERTEX;
        }
        self.corner_to_vertex
            .get(corner as usize)
            .copied()
            .unwrap_or(INVALID_VERTEX)
    }

    pub fn num_vertices(&self) -> u32 {
        self.vertex_to_leftmost_corner.len() as u32
    }

    pub fn leftmost_corner(&self, vert: u32) -> u32 {
        self.vertex_to_leftmost_corner
            .get(vert as usize)
            .copied()
            .unwrap_or(INVALID_CORNER)
    }

    /// Renumber vertices so that each side of every seam gets its own.
    pub fn recompute_vertices(&mut self, base: &CornerTable) -> Result<(), DracoError> {
        self.vertex_to_leftmost_corner.clear();
        let mut num_new_vertices = 0u32;
        for v in 0..base.num_vertices() {
            let c = base.leftmost_corner(v);
            if c == INVALID_CORNER {
                continue; // Isolated.
            }
            let mut vert_id = num_new_vertices;
            num_new_vertices += 1;

            // On a seam, back up counter-clockwise as far as the cuts allow, so
            // the walk below starts on a seam rather than in the middle of one
            // side of it.
            let mut first_c = c;
            if self.is_vertex_on_seam.get(v as usize).copied().unwrap_or(false) {
                let mut act_c = self.swing_left(base, first_c);
                let mut steps = 0u32;
                while act_c != INVALID_CORNER {
                    first_c = act_c;
                    act_c = self.swing_left(base, act_c);
                    if act_c == c {
                        // Swinging left from the left-most corner cannot return
                        // to it; if it does the table is inconsistent.
                        return Err(DracoError::Corrupt);
                    }
                    steps += 1;
                    if steps > base.num_corners() {
                        return Err(DracoError::Corrupt);
                    }
                }
            }
            self.set_vertex(first_c, vert_id)?;
            self.vertex_to_leftmost_corner.push(first_c);

            // Then clockwise over the whole ring — on the *base* table, so it
            // crosses seams — starting a new vertex at each cut.
            let mut act_c = base.swing_right(first_c);
            let mut steps = 0u32;
            while act_c != INVALID_CORNER && act_c != first_c {
                if self.is_corner_opposite_to_seam_edge(base.next(act_c)) {
                    vert_id = num_new_vertices;
                    num_new_vertices += 1;
                    self.vertex_to_leftmost_corner.push(act_c);
                }
                self.set_vertex(act_c, vert_id)?;
                act_c = base.swing_right(act_c);
                steps += 1;
                if steps > base.num_corners() {
                    return Err(DracoError::Corrupt);
                }
            }
        }
        Ok(())
    }

    fn set_vertex(&mut self, corner: u32, vert: u32) -> Result<(), DracoError> {
        *self
            .corner_to_vertex
            .get_mut(corner as usize)
            .ok_or(DracoError::Corrupt)? = vert;
        Ok(())
    }
}

/// Read which edges are attribute seams, one pass over the faces.
///
/// A boundary edge is a seam for every attribute without a bit being spent on
/// it, and an interior edge is read once from the lower-numbered of its two
/// faces. Reading it from both, or spending a bit on the boundary, desynchronises
/// every later bit — and the bits are one per attribute per edge, so a
/// desynchronised stream still yields seams, just the wrong ones.
pub fn decode_attribute_seams(
    base: &CornerTable,
    seams: &mut [BitDecoder<'_>],
) -> Result<Vec<AttributeCornerTable>, DracoError> {
    let mut seam_corners: Vec<Vec<u32>> = vec![Vec::new(); seams.len()];
    let mut face_start = 0u32;
    while face_start < base.num_corners() {
        let src_face = base.face(face_start);
        for corner in [face_start, base.next(face_start), base.previous(face_start)] {
            let opp = base.opposite(corner);
            if opp == INVALID_CORNER {
                for list in seam_corners.iter_mut() {
                    list.push(corner);
                }
                continue;
            }
            if base.face(opp) < src_face {
                continue; // Already read from the other side.
            }
            for (list, decoder) in seam_corners.iter_mut().zip(seams.iter_mut()) {
                if decoder.bit() != 0 {
                    list.push(corner);
                }
            }
        }
        face_start += 3;
    }

    let mut tables = Vec::with_capacity(seams.len());
    for corners in seam_corners {
        let mut table = AttributeCornerTable::new(base);
        for c in corners {
            table.add_seam_edge(base, c)?;
        }
        table.recompute_vertices(base)?;
        tables.push(table);
    }
    Ok(tables)
}

/// Which point each corner belongs to, and how many points there are.
///
/// With no non-position attributes a point *is* a vertex. Otherwise each base
/// vertex's ring is walked clockwise and a new point begins wherever any
/// attribute's vertex number changes from the previous corner.
pub fn assign_points_to_corners(
    base: &CornerTable,
    tables: &[AttributeCornerTable],
    is_vert_hole: &[bool],
    num_connectivity_verts: u32,
) -> Result<(Vec<u32>, Vec<u32>, u32), DracoError> {
    if tables.is_empty() {
        let corner_to_point: Vec<u32> = (0..base.num_corners()).map(|c| base.vertex(c)).collect();
        return Ok((corner_to_point, Vec::new(), num_connectivity_verts));
    }

    let mut corner_to_point = vec![0u32; base.num_corners() as usize];
    let mut point_to_corner: Vec<u32> = Vec::new();

    for v in 0..base.num_vertices() {
        let c = base.leftmost_corner(v);
        if c == INVALID_CORNER {
            continue;
        }
        // Where to begin the walk. On a boundary the left-most corner already
        // lies on it. Otherwise the ring is a closed loop with no natural start,
        // so a seam has to be found first — dedup from anywhere else and the
        // first and last corners of the walk end up in different points when
        // they belong in the same one.
        let mut dedup_first = c;
        let on_boundary = is_vert_hole.get(v as usize).copied().unwrap_or(false);
        if !on_boundary {
            for table in tables {
                if !table.is_corner_on_seam(base, c) {
                    continue;
                }
                let vert_id = table.vertex(c);
                let mut act_c = base.swing_right(c);
                let mut seam_found = false;
                let mut steps = 0u32;
                while act_c != c {
                    if act_c == INVALID_CORNER {
                        return Err(DracoError::Corrupt);
                    }
                    if table.vertex(act_c) != vert_id {
                        dedup_first = act_c;
                        seam_found = true;
                        break;
                    }
                    act_c = base.swing_right(act_c);
                    steps += 1;
                    if steps > base.num_corners() {
                        return Err(DracoError::Corrupt);
                    }
                }
                if seam_found {
                    break;
                }
            }
        }

        let mut prev_c = dedup_first;
        corner_to_point[dedup_first as usize] = point_to_corner.len() as u32;
        point_to_corner.push(dedup_first);
        let mut act_c = base.swing_right(dedup_first);
        let mut steps = 0u32;
        while act_c != INVALID_CORNER && act_c != dedup_first {
            let changed = tables
                .iter()
                .any(|t| t.vertex(act_c) != t.vertex(prev_c));
            if changed {
                corner_to_point[act_c as usize] = point_to_corner.len() as u32;
                point_to_corner.push(act_c);
            } else {
                corner_to_point[act_c as usize] = corner_to_point[prev_c as usize];
            }
            prev_c = act_c;
            act_c = base.swing_right(act_c);
            steps += 1;
            if steps > base.num_corners() {
                return Err(DracoError::Corrupt);
            }
        }
    }

    let num_points = point_to_corner.len() as u32;
    Ok((corner_to_point, point_to_corner, num_points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loaders::draco::edgebreaker_connectivity;

    const SPHERE: &[u8] = include_bytes!("testdata/sphere.drc");
    const BUNNY: &[u8] = include_bytes!("testdata/bunny.drc");

    /// FNV-1a over an index buffer, little-endian per index.
    ///
    /// The reference value comes from running the same hash over `draco3d`'s
    /// decoded face list, which makes this a byte-exact comparison against
    /// Google's decoder over every index without carrying a megabyte of fixture.
    fn fnv1a(indices: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in indices {
            for byte in v.to_le_bytes() {
                h = (h ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// The sphere's index buffer is Google's, index for index.
    ///
    /// The point count alone is a real check — 423 is not 362, and it is reached
    /// only by cutting the texture seam in exactly the right places — but it is
    /// a single number. The hash covers all 2160 indices, and the first thirty
    /// are spelled out so a failure says *where* rather than only *that*.
    #[test]
    fn the_sphere_index_buffer_matches_the_reference_decoder() {
        let d = edgebreaker_connectivity(SPHERE).expect("connectivity");
        assert_eq!(d.num_points, 423, "points, as draco3d reports them");
        let idx = d.indices();
        assert_eq!(idx.len(), 720 * 3);
        assert_eq!(
            &idx[..30],
            &[
                0, 1, 2, 3, 2, 4, 2, 1, 4, 4, 1, 5, 6, 5, 7, 5, 1, 7, 7, 1, 8, 1, 0, 8, 8, 0, 9,
                10, 9, 11
            ]
        );
        assert_eq!(fnv1a(&idx), 11727646580840884174, "all 2160 indices");
        assert!(idx.iter().all(|i| *i < d.num_points), "an index past the points");
    }

    /// And the bunny's 208353 indices.
    #[test]
    fn the_bunny_index_buffer_matches_the_reference_decoder() {
        let d = edgebreaker_connectivity(BUNNY).expect("connectivity");
        assert_eq!(d.num_points, 34834);
        let idx = d.indices();
        assert_eq!(idx.len(), 69451 * 3);
        assert_eq!(
            &idx[..30],
            &[
                15922, 0, 1, 0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 6, 7, 8, 9, 10, 11, 11, 10, 12,
                10, 8, 12, 8, 7, 12
            ]
        );
        assert_eq!(fnv1a(&idx), 5786114087933875773, "all 208353 indices");
        assert!(idx.iter().all(|i| *i < d.num_points));
    }

    /// Every point is reachable, and the two maps are inverse where they should
    /// be.
    ///
    /// `point_to_corner` is what the attribute decoders will sample values
    /// through, so a point with no corner would read whatever was at index zero.
    #[test]
    fn the_point_and_corner_maps_agree() {
        for bytes in [SPHERE, BUNNY] {
            let d = edgebreaker_connectivity(bytes).expect("connectivity");
            assert_eq!(d.point_to_corner.len(), d.num_points as usize);
            for (point, corner) in d.point_to_corner.iter().enumerate() {
                assert_eq!(
                    d.corner_to_point[*corner as usize], point as u32,
                    "point {point} does not round-trip through its corner"
                );
            }
            // Every point is used by some corner: the count is the map's length,
            // so a gap would mean two points sharing a corner.
            let used: std::collections::HashSet<u32> = d.corner_to_point.iter().copied().collect();
            assert_eq!(used.len(), d.num_points as usize, "an unused point id");
        }
    }

    /// Seams split the attribute tables exactly as far as the points require.
    ///
    /// The sphere's normals are continuous over a smooth ball, so that table has
    /// one vertex per position and no interior seam. Its texture coordinates are
    /// not — the UV chart has to be cut somewhere — and that table has one vertex
    /// per *point*. The point count is the union over attributes, which here
    /// means the texture coordinates alone decide it.
    #[test]
    fn the_sphere_seams_fall_on_the_texture_coordinates() {
        let d = edgebreaker_connectivity(SPHERE).expect("connectivity");
        assert_eq!(d.attribute_tables.len(), 2, "normals and texture coordinates");
        let normals = &d.attribute_tables[0];
        let uvs = &d.attribute_tables[1];
        assert!(
            normals.no_interior_seams,
            "a smooth sphere's normals are continuous"
        );
        assert_eq!(normals.num_vertices(), 362, "one per position");
        assert!(!uvs.no_interior_seams, "a UV chart has to be cut");
        assert_eq!(uvs.num_vertices(), 423, "one per point");
        assert_eq!(d.num_points, uvs.num_vertices());
    }

    /// A seam-aware swing stops at a cut where the base table walks through it.
    ///
    /// This is the distinction the whole module rests on, so it is checked
    /// directly rather than only through its consequences: on the sphere's UV
    /// seam there are corners whose base opposite exists and whose attribute
    /// opposite does not, and every such cut is cut from both sides.
    ///
    /// Note what is *not* true: the two sides of a cut need not carry different
    /// attribute vertices. A ring cut in one place is still a single path, so it
    /// remains one attribute vertex and the cut's two sides share it. A vertex
    /// splits only where its ring is cut twice or more, which is why the count
    /// below is over vertices rather than over edges.
    #[test]
    fn a_cut_edge_has_no_opposite_in_the_attribute_table() {
        let d = edgebreaker_connectivity(SPHERE).expect("connectivity");
        let base = &d.replay.table;
        let uvs = &d.attribute_tables[1];
        let is_cut = |c: u32| {
            base.opposite(c) != INVALID_CORNER && uvs.opposite(base, c) == INVALID_CORNER
        };
        let cut: Vec<u32> = (0..base.num_corners()).filter(|c| is_cut(*c)).collect();
        assert!(!cut.is_empty(), "the UV seam cuts no interior edge");
        for c in &cut {
            assert!(
                is_cut(base.opposite(*c)),
                "corner {c} is cut but the corner facing it is not"
            );
        }

        // Every base vertex carries one attribute vertex per arc its ring was cut
        // into, and those sum to the point count. 362 positions becoming 423
        // points means 61 extra arcs, all of them on the seam.
        let mut per_vertex: std::collections::HashMap<u32, std::collections::HashSet<u32>> =
            std::collections::HashMap::new();
        for c in 0..base.num_corners() {
            per_vertex.entry(base.vertex(c)).or_default().insert(uvs.vertex(c));
        }
        let total: usize = per_vertex.values().map(|s| s.len()).sum();
        assert_eq!(total, 423, "attribute vertices, summed over positions");
        let split = per_vertex.values().filter(|s| s.len() > 1).count();
        assert!(split > 0, "no position was split by the seam");
        assert_eq!(
            total - per_vertex.len(),
            423 - 362,
            "the extra points are exactly the extra arcs"
        );
    }
}
