//! The corner table, and replaying the edgebreaker symbols into one.
//!
//! A corner table is the mesh as three parallel arrays indexed by *corner* — a
//! (face, slot) pair, corner `3f + k` being slot `k` of face `f`. Each corner
//! records the vertex it sits on and the corner *opposite* it across the shared
//! edge, which is what makes walking the neighbourhood of a vertex an array
//! lookup instead of a search.
//!
//! Edgebreaker's decode is the reverse of its encode, and that reversal is the
//! part which cannot be inferred from the symbol table: the decoder maintains a
//! stack of *active corners*, one per boundary it is still growing. `C`, `L` and
//! `R` extend the boundary on top of the stack, `E` pushes a new one, and `S`
//! pops one and merges it into the next — which also means merging two vertices
//! into one, and relabelling every corner that pointed at the loser.
//!
//! The `symbol_id` a split event names is the *encoder's*, counting the other
//! way; `num_symbols - symbol_id - 1` converts. Getting that backwards still
//! decodes, into a mesh whose triangles are individually plausible and whose
//! surface is stitched together wrongly.

use super::rans::BitDecoder;
use super::traversal::{Symbol, SymbolSource, TopologySplit};
use super::{Connectivity, DracoError};

/// No corner — Draco's `kInvalidCornerIndex`.
pub const INVALID_CORNER: u32 = u32::MAX;
/// No vertex.
pub const INVALID_VERTEX: u32 = u32::MAX;

/// Draco's `RIGHT_FACE_EDGE`; the other value, 0, is `LEFT_FACE_EDGE`.
const RIGHT_FACE_EDGE: u32 = 1;

/// The mesh as corners: which vertex each sits on, and what is across the edge.
#[derive(Debug, Clone, Default)]
pub struct CornerTable {
    corner_to_vertex: Vec<u32>,
    opposite: Vec<u32>,
    /// The left-most corner of each vertex — the one to start a neighbourhood
    /// walk from, chosen so that swinging left from it reaches a boundary.
    vertex_corners: Vec<u32>,
}

impl CornerTable {
    /// Room for `num_faces` faces, nothing connected yet.
    fn with_faces(num_faces: u32) -> Self {
        let corners = num_faces as usize * 3;
        CornerTable {
            corner_to_vertex: vec![INVALID_VERTEX; corners],
            opposite: vec![INVALID_CORNER; corners],
            vertex_corners: Vec::new(),
        }
    }

    pub fn num_corners(&self) -> u32 {
        self.corner_to_vertex.len() as u32
    }

    pub fn num_faces(&self) -> u32 {
        self.num_corners() / 3
    }

    pub fn num_vertices(&self) -> u32 {
        self.vertex_corners.len() as u32
    }

    /// The face a corner belongs to.
    pub fn face(&self, corner: u32) -> u32 {
        corner / 3
    }

    /// The next corner round the same face.
    pub fn next(&self, corner: u32) -> u32 {
        if corner == INVALID_CORNER {
            return corner;
        }
        // Draco writes this as `LocalIndex(++corner) ? corner : corner - 3`:
        // step forward, and wrap to the start of the face when the step lands
        // on a multiple of three.
        if (corner + 1).is_multiple_of(3) {
            corner - 2
        } else {
            corner + 1
        }
    }

    /// The previous corner round the same face.
    pub fn previous(&self, corner: u32) -> u32 {
        if corner == INVALID_CORNER {
            return corner;
        }
        if corner.is_multiple_of(3) {
            corner + 2
        } else {
            corner - 1
        }
    }

    pub fn opposite(&self, corner: u32) -> u32 {
        if corner == INVALID_CORNER {
            return corner;
        }
        self.opposite[corner as usize]
    }

    pub fn vertex(&self, corner: u32) -> u32 {
        if corner == INVALID_CORNER {
            return INVALID_VERTEX;
        }
        self.corner_to_vertex[corner as usize]
    }

    pub fn leftmost_corner(&self, vert: u32) -> u32 {
        if vert == INVALID_VERTEX {
            return INVALID_CORNER;
        }
        self.vertex_corners
            .get(vert as usize)
            .copied()
            .unwrap_or(INVALID_CORNER)
    }

    /// The corner on the left-hand neighbouring face that sits on the same
    /// vertex. Invalid at a boundary, which is how a walk knows to turn round.
    pub fn swing_left(&self, corner: u32) -> u32 {
        self.next(self.opposite(self.next(corner)))
    }

    /// The same, the other way.
    pub fn swing_right(&self, corner: u32) -> u32 {
        self.previous(self.opposite(self.previous(corner)))
    }

    pub fn is_vertex_isolated(&self, vert: u32) -> bool {
        self.leftmost_corner(vert) == INVALID_CORNER
    }

    fn add_new_vertex(&mut self) -> u32 {
        self.vertex_corners.push(INVALID_CORNER);
        (self.vertex_corners.len() - 1) as u32
    }

    fn map_corner_to_vertex(&mut self, corner: u32, vert: u32) -> Result<(), DracoError> {
        let slot = self
            .corner_to_vertex
            .get_mut(corner as usize)
            .ok_or(DracoError::Corrupt)?;
        *slot = vert;
        Ok(())
    }

    fn set_leftmost_corner(&mut self, vert: u32, corner: u32) {
        if vert == INVALID_VERTEX {
            return;
        }
        if let Some(slot) = self.vertex_corners.get_mut(vert as usize) {
            *slot = corner;
        }
    }

    fn make_vertex_isolated(&mut self, vert: u32) {
        self.set_leftmost_corner(vert, INVALID_CORNER);
    }

    /// Both directions at once, which is the only way it is ever used — a
    /// one-sided opposite is a corner table that lies.
    fn set_opposite_corners(&mut self, a: u32, b: u32) -> Result<(), DracoError> {
        for (from, to) in [(a, b), (b, a)] {
            if from == INVALID_CORNER {
                continue;
            }
            let slot = self
                .opposite
                .get_mut(from as usize)
                .ok_or(DracoError::Corrupt)?;
            *slot = to;
        }
        Ok(())
    }

    /// Every corner sitting on `vert`, starting from its left-most.
    ///
    /// This is Draco's `VertexCornersIterator`: swing left until a boundary,
    /// then resume from one step right of the start and swing right. On a closed
    /// vertex the left swing comes back to the start and that is the end. The
    /// step count is capped because a corrupt table can be cyclic in a way a
    /// closed vertex is not.
    pub fn vertex_corners(&self, vert: u32) -> Vec<u32> {
        let start = self.leftmost_corner(vert);
        let mut out = Vec::new();
        let mut corner = start;
        let mut going_left = true;
        while corner != INVALID_CORNER && out.len() <= self.num_corners() as usize {
            out.push(corner);
            if going_left {
                corner = self.swing_left(corner);
                if corner == INVALID_CORNER {
                    corner = self.swing_right(start);
                    going_left = false;
                } else if corner == start {
                    corner = INVALID_CORNER;
                }
            } else {
                corner = self.swing_right(corner);
            }
        }
        out
    }

    /// The faces as vertex triples, in corner order.
    ///
    /// These are corner-table *vertices*, not Draco point ids: a seam in a
    /// texture coordinate splits one vertex into several points, so the index
    /// buffer a glTF wants is this mapped through the point assignment.
    pub fn triangles(&self) -> Vec<[u32; 3]> {
        (0..self.num_faces())
            .map(|f| {
                let c = f * 3;
                [self.vertex(c), self.vertex(c + 1), self.vertex(c + 2)]
            })
            .collect()
    }
}

/// A replayed connectivity: the table, plus what the attribute decoders need.
#[derive(Debug, Clone)]
pub struct Replay {
    pub table: CornerTable,
    /// The corner each component's traversal started from.
    pub init_corners: Vec<u32>,
    /// Whether each of those start faces was interior (a new face was added) or
    /// sat on an open boundary (none was).
    pub init_face_configurations: Vec<bool>,
    /// Whether each vertex lies on a boundary. The attribute predictors need it,
    /// and it falls out of the replay for free.
    pub is_vert_hole: Vec<bool>,
    /// Vertices left after isolated ones were folded away.
    pub num_vertices: u32,
}

/// Replay the symbols into a corner table.
///
/// `splits` is consumed from the end: events are ordered by ascending encoder
/// symbol id, and the decoder walks encoder ids downwards, so the last event is
/// always the next one due.
pub fn replay(
    source: &mut dyn SymbolSource,
    splits: &mut Vec<TopologySplit>,
    start_faces: &mut BitDecoder<'_>,
    conn: &Connectivity,
    remove_invalid_vertices: bool,
) -> Result<Replay, DracoError> {
    let num_symbols = conn.num_encoded_symbols;
    let mut table = CornerTable::with_faces(conn.num_faces);
    // A split adds a vertex, so this is the most the replay can ever create;
    // Draco uses the same bound to reject a stream that creates more.
    let max_num_vertices = conn
        .num_encoded_vertices
        .checked_add(conn.num_encoded_split_symbols)
        .ok_or(DracoError::Corrupt)? as usize;
    let mut is_vert_hole = vec![true; max_num_vertices];

    let mut active: Vec<u32> = Vec::new();
    // Keyed by *decoder* symbol id, which is why the conversion below matters.
    let mut split_active_corners: std::collections::HashMap<i64, u32> =
        std::collections::HashMap::new();
    let mut invalid_vertices: Vec<u32> = Vec::new();
    let mut num_faces: u32 = 0;

    let mark_interior = |is_vert_hole: &mut Vec<bool>, vert: u32| -> Result<(), DracoError> {
        let slot = is_vert_hole
            .get_mut(vert as usize)
            .ok_or(DracoError::Corrupt)?;
        *slot = false;
        Ok(())
    };

    for symbol_id in 0..num_symbols as usize {
        if num_faces >= conn.num_faces {
            return Err(DracoError::Corrupt);
        }
        // Pulled one at a time: the valence traversal chooses the next symbol's
        // entropy context from the valences of the mesh built so far, so it
        // cannot be read before this point.
        let symbol = source.next_symbol(&table)?;
        let face = num_faces;
        num_faces += 1;
        let corner = face * 3;
        let mut check_topology_split = false;

        match symbol {
            Symbol::C => {
                // A new face between two edges of the open boundary: the active
                // edge, opposite corner `a`, and the edge reached by swinging
                // round the shared vertex `x` to corner `b`.
                let corner_a = *active.last().ok_or(DracoError::Corrupt)?;
                let vertex_x = table.vertex(table.next(corner_a));
                let corner_b = table.next(table.leftmost_corner(vertex_x));
                if corner_a == corner_b {
                    return Err(DracoError::Corrupt);
                }
                if table.opposite(corner_a) != INVALID_CORNER
                    || table.opposite(corner_b) != INVALID_CORNER
                {
                    return Err(DracoError::Corrupt);
                }
                table.set_opposite_corners(corner_a, corner + 1)?;
                table.set_opposite_corners(corner_b, corner + 2)?;

                let vert_a_prev = table.vertex(table.previous(corner_a));
                let vert_b_next = table.vertex(table.next(corner_b));
                if vertex_x == vert_a_prev || vertex_x == vert_b_next {
                    // The face would be degenerate.
                    return Err(DracoError::Corrupt);
                }
                table.map_corner_to_vertex(corner, vertex_x)?;
                table.map_corner_to_vertex(corner + 1, vert_b_next)?;
                table.map_corner_to_vertex(corner + 2, vert_a_prev)?;
                table.set_leftmost_corner(vert_a_prev, corner + 2);
                mark_interior(&mut is_vert_hole, vertex_x)?;
                *active.last_mut().ok_or(DracoError::Corrupt)? = corner;
            }
            Symbol::L | Symbol::R => {
                // Extend the boundary across the active edge, creating one new
                // vertex opposite it. Which of the two new edges stays active is
                // the whole difference between L and R.
                let corner_a = *active.last().ok_or(DracoError::Corrupt)?;
                if table.opposite(corner_a) != INVALID_CORNER {
                    return Err(DracoError::Corrupt);
                }
                let (opp_corner, corner_l, corner_r) = if symbol == Symbol::R {
                    (corner + 2, corner + 1, corner)
                } else {
                    (corner + 1, corner, corner + 2)
                };
                table.set_opposite_corners(opp_corner, corner_a)?;
                let new_vert = table.add_new_vertex();
                if table.num_vertices() as usize > max_num_vertices {
                    return Err(DracoError::Corrupt);
                }
                table.map_corner_to_vertex(opp_corner, new_vert)?;
                table.set_leftmost_corner(new_vert, opp_corner);

                let vertex_r = table.vertex(table.previous(corner_a));
                table.map_corner_to_vertex(corner_r, vertex_r)?;
                table.set_leftmost_corner(vertex_r, corner_r);
                let vertex_l = table.vertex(table.next(corner_a));
                table.map_corner_to_vertex(corner_l, vertex_l)?;
                *active.last_mut().ok_or(DracoError::Corrupt)? = corner;
                check_topology_split = true;
            }
            Symbol::S => {
                // Merge the top two boundaries. No vertex is created; two are
                // merged, and every corner on the loser has to be relabelled.
                let corner_b = active.pop().ok_or(DracoError::Corrupt)?;
                // `a` is either the next boundary on the stack, or one parked
                // earlier by a topology split naming this symbol.
                if let Some(parked) = split_active_corners.get(&(symbol_id as i64)) {
                    active.push(*parked);
                }
                let corner_a = *active.last().ok_or(DracoError::Corrupt)?;
                if corner_a == corner_b {
                    return Err(DracoError::Corrupt);
                }
                if table.opposite(corner_a) != INVALID_CORNER
                    || table.opposite(corner_b) != INVALID_CORNER
                {
                    return Err(DracoError::Corrupt);
                }
                table.set_opposite_corners(corner_a, corner + 2)?;
                table.set_opposite_corners(corner_b, corner + 1)?;

                let vertex_p = table.vertex(table.previous(corner_a));
                table.map_corner_to_vertex(corner, vertex_p)?;
                let vertex_a_next = table.vertex(table.next(corner_a));
                table.map_corner_to_vertex(corner + 1, vertex_a_next)?;
                let vert_b_prev = table.vertex(table.previous(corner_b));
                table.map_corner_to_vertex(corner + 2, vert_b_prev)?;
                table.set_leftmost_corner(vert_b_prev, corner + 2);

                let mut corner_n = table.next(corner_b);
                let vertex_n = table.vertex(corner_n);
                // The valence traversal tracks valences per vertex, so it has to
                // learn that these two became one.
                source.merge_vertices(vertex_p, vertex_n);
                table.set_leftmost_corner(vertex_p, table.leftmost_corner(vertex_n));
                // Relabel the whole ring of the vertex being merged away.
                let first_corner = corner_n;
                let mut steps = 0u32;
                while corner_n != INVALID_CORNER {
                    table.map_corner_to_vertex(corner_n, vertex_p)?;
                    corner_n = table.swing_left(corner_n);
                    if corner_n == first_corner {
                        // A closed ring here would mean the split did not split.
                        return Err(DracoError::Corrupt);
                    }
                    steps += 1;
                    if steps > table.num_corners() {
                        return Err(DracoError::Corrupt);
                    }
                }
                table.make_vertex_isolated(vertex_n);
                if remove_invalid_vertices {
                    invalid_vertices.push(vertex_n);
                }
                *active.last_mut().ok_or(DracoError::Corrupt)? = corner;
            }
            Symbol::E => {
                // A face standing on its own: three new vertices, and a new
                // boundary pushed for later symbols to grow.
                let v0 = table.add_new_vertex();
                table.map_corner_to_vertex(corner, v0)?;
                let v1 = table.add_new_vertex();
                table.map_corner_to_vertex(corner + 1, v1)?;
                let v2 = table.add_new_vertex();
                table.map_corner_to_vertex(corner + 2, v2)?;
                if table.num_vertices() as usize > max_num_vertices {
                    return Err(DracoError::Corrupt);
                }
                table.set_leftmost_corner(v0, corner);
                table.set_leftmost_corner(v1, corner + 1);
                table.set_leftmost_corner(v2, corner + 2);
                active.push(corner);
                check_topology_split = true;
            }
        }

        // Report the corner the traversal now stands on, which is what decides
        // the next symbol's context.
        let active_top = *active.last().ok_or(DracoError::Corrupt)?;
        source.new_active_corner_reached(&table, active_top);

        if check_topology_split {
            // Only L, R and E can be the far side of a split: they are the
            // symbols whose face has edges a later S can attach to.
            let encoder_symbol_id = num_symbols as i64 - symbol_id as i64 - 1;
            while let Some(last) = splits.last() {
                if last.source_symbol_id as i64 > encoder_symbol_id {
                    // The event names a symbol already past; Draco reports this
                    // as a split id of -1 and gives up, and so do we.
                    return Err(DracoError::Corrupt);
                }
                if (last.source_symbol_id as i64) != encoder_symbol_id {
                    break;
                }
                let split = splits.pop().ok_or(DracoError::Corrupt)?;
                let act_top = *active.last().ok_or(DracoError::Corrupt)?;
                // The active edge has two idle edges beside it; the recorded
                // edge says which one the S face will arrive on.
                let new_active = if split.source_edge == RIGHT_FACE_EDGE {
                    table.next(act_top)
                } else {
                    table.previous(act_top)
                };
                let decoder_split_symbol_id =
                    num_symbols as i64 - split.split_symbol_id as i64 - 1;
                split_active_corners.insert(decoder_split_symbol_id, new_active);
            }
        }
    }

    if table.num_vertices() as usize > max_num_vertices {
        return Err(DracoError::Corrupt);
    }

    // Each boundary left on the stack is one component's start face, which was
    // never given a symbol of its own.
    let mut init_corners = Vec::new();
    let mut init_face_configurations = Vec::new();
    while let Some(corner) = active.pop() {
        let interior_face = start_faces.bit() != 0;
        if interior_face {
            if num_faces >= conn.num_faces {
                return Err(DracoError::Corrupt);
            }
            let corner_a = corner;
            let vert_n = table.vertex(table.next(corner_a));
            let corner_b = table.next(table.leftmost_corner(vert_n));
            let vert_x = table.vertex(table.next(corner_b));
            let corner_c = table.next(table.leftmost_corner(vert_x));
            if corner == corner_b || corner == corner_c || corner_b == corner_c {
                return Err(DracoError::Corrupt);
            }
            if table.opposite(corner) != INVALID_CORNER
                || table.opposite(corner_b) != INVALID_CORNER
                || table.opposite(corner_c) != INVALID_CORNER
            {
                return Err(DracoError::Corrupt);
            }
            let vert_p = table.vertex(table.next(corner_c));
            let face = num_faces;
            num_faces += 1;
            let new_corner = face * 3;
            table.set_opposite_corners(new_corner, corner)?;
            table.set_opposite_corners(new_corner + 1, corner_b)?;
            table.set_opposite_corners(new_corner + 2, corner_c)?;
            table.map_corner_to_vertex(new_corner, vert_x)?;
            table.map_corner_to_vertex(new_corner + 1, vert_p)?;
            table.map_corner_to_vertex(new_corner + 2, vert_n)?;
            for ci in 0..3 {
                mark_interior(&mut is_vert_hole, table.vertex(new_corner + ci))?;
            }
            init_face_configurations.push(true);
            init_corners.push(new_corner);
        } else {
            // The traversal started on an open boundary, so there is no face to
            // add — only the corner to remember.
            init_face_configurations.push(false);
            init_corners.push(corner);
        }
    }

    if num_faces != conn.num_faces {
        return Err(DracoError::Corrupt);
    }

    // Vertices merged away by `S` leave holes in the numbering. When there is no
    // other attribute connectivity, Draco closes the gaps by swapping the last
    // live vertex down, so that every id below the count is real.
    let mut num_vertices = table.num_vertices();
    for invalid_vert in invalid_vertices {
        if num_vertices == 0 {
            return Err(DracoError::Corrupt);
        }
        let mut src_vert = num_vertices - 1;
        while table.leftmost_corner(src_vert) == INVALID_CORNER {
            if num_vertices <= 1 {
                return Err(DracoError::Corrupt);
            }
            num_vertices -= 1;
            src_vert = num_vertices - 1;
        }
        if src_vert < invalid_vert {
            continue;
        }
        for cid in table.vertex_corners(src_vert) {
            if table.vertex(cid) != src_vert {
                return Err(DracoError::Corrupt);
            }
            table.map_corner_to_vertex(cid, invalid_vert)?;
        }
        table.set_leftmost_corner(invalid_vert, table.leftmost_corner(src_vert));
        table.make_vertex_isolated(src_vert);
        let moved = *is_vert_hole
            .get(src_vert as usize)
            .ok_or(DracoError::Corrupt)?;
        *is_vert_hole
            .get_mut(invalid_vert as usize)
            .ok_or(DracoError::Corrupt)? = moved;
        *is_vert_hole
            .get_mut(src_vert as usize)
            .ok_or(DracoError::Corrupt)? = false;
        num_vertices -= 1;
    }

    Ok(Replay {
        table,
        init_corners,
        init_face_configurations,
        is_vert_hole,
        num_vertices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loaders::draco::edgebreaker_connectivity;
    use std::collections::{BTreeMap, HashMap, HashSet};

    const SPHERE: &[u8] = include_bytes!("testdata/sphere.drc");
    const BUNNY: &[u8] = include_bytes!("testdata/bunny.drc");

    /// Neighbour sets of the vertices that appear in any face.
    fn adjacency(tris: &[[u32; 3]]) -> HashMap<u32, HashSet<u32>> {
        let mut nbrs: HashMap<u32, HashSet<u32>> = HashMap::new();
        for t in tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                nbrs.entry(a).or_default().insert(b);
                nbrs.entry(b).or_default().insert(a);
            }
        }
        nbrs
    }

    /// How many vertices have each valence, which is what the reference decoder
    /// is compared against.
    fn valence_histogram(tris: &[[u32; 3]]) -> BTreeMap<usize, usize> {
        let mut hist = BTreeMap::new();
        for (_, set) in adjacency(tris) {
            *hist.entry(set.len()).or_insert(0) += 1;
        }
        hist
    }

    fn edge_count(tris: &[[u32; 3]]) -> usize {
        adjacency(tris).values().map(|s| s.len()).sum::<usize>() / 2
    }

    /// Next and previous walk the face and nothing else.
    #[test]
    fn next_and_previous_cycle_within_a_face() {
        let table = CornerTable::with_faces(2);
        for face in 0..2u32 {
            let c = face * 3;
            assert_eq!(table.next(c), c + 1);
            assert_eq!(table.next(c + 1), c + 2);
            assert_eq!(table.next(c + 2), c, "next wraps to the face's first corner");
            assert_eq!(table.previous(c), c + 2, "previous wraps the other way");
            assert_eq!(table.previous(c + 1), c);
            assert_eq!(table.previous(c + 2), c + 1);
            // Three steps either way is the identity.
            assert_eq!(table.next(table.next(table.next(c + 1))), c + 1);
            assert_eq!(table.previous(table.next(c + 2)), c + 2);
        }
        assert_eq!(table.next(INVALID_CORNER), INVALID_CORNER);
        assert_eq!(table.previous(INVALID_CORNER), INVALID_CORNER);
    }

    /// The sphere replays into a closed manifold with Euler characteristic 2.
    ///
    /// This is the first check that the replay produced a *surface* rather than
    /// 720 unrelated triangles. A sphere has no boundary, so every one of the
    /// 2160 corners must have a real corner opposite it, and the relation must be
    /// an involution — a one-sided opposite is a corner table that lies about
    /// which faces touch. V - E + F = 2 then pins the whole thing: get the
    /// stitching wrong and the count of distinct edges moves.
    #[test]
    fn the_sphere_replays_into_a_closed_manifold() {
        let r = edgebreaker_connectivity(SPHERE).expect("connectivity").replay;
        let table = &r.table;
        assert_eq!(table.num_faces(), 720);

        for c in 0..table.num_corners() {
            let o = table.opposite(c);
            assert_ne!(o, INVALID_CORNER, "corner {c} has no opposite on a closed mesh");
            assert_eq!(table.opposite(o), c, "opposite is not an involution at {c}");
            assert_ne!(o, c, "a corner cannot be opposite itself");
            assert_ne!(table.vertex(c), INVALID_VERTEX, "corner {c} has no vertex");
        }

        let tris = table.triangles();
        let live = adjacency(&tris).len();
        let edges = edge_count(&tris);
        assert_eq!(live, 362, "the 362 vertices the header declared");
        assert_eq!(edges, 1080, "3F/2 for a closed triangle mesh");
        assert_eq!(
            live as i64 - edges as i64 + tris.len() as i64,
            2,
            "V - E + F for a sphere"
        );
        // The 19 split symbols each created a vertex that `S` then merged away.
        assert_eq!(table.num_vertices(), 362 + 19);
        assert_eq!(
            (0..table.num_vertices())
                .filter(|v| table.is_vertex_isolated(*v))
                .count(),
            19,
            "one isolated vertex per split"
        );
        // A closed mesh has no boundary, so every vertex still in use was
        // closed off by some `C` or by the interior start face. The merged-away
        // ones keep whatever flag they had, so they are skipped — with attribute
        // connectivity present they are not compacted out of the numbering.
        for v in 0..table.num_vertices() {
            if !table.is_vertex_isolated(v) {
                assert!(!r.is_vert_hole[v as usize], "vertex {v} is interior on a sphere");
            }
        }
    }

    /// The sphere's valences are the ones Google's decoder produces.
    ///
    /// Euler characteristic and manifoldness are topological: a differently
    /// stitched sphere can still satisfy both. The valence histogram is not — it
    /// counts how many faces meet at each vertex, and it is computed here from
    /// the replayed table and compared against the same histogram taken from
    /// `draco3d`'s decoded index buffer over unique positions. Two poles of
    /// valence 24, a ring of 48 at valence 5, and 312 ordinary vertices at 6.
    #[test]
    fn the_sphere_valences_match_the_reference_decoder() {
        let r = edgebreaker_connectivity(SPHERE).expect("connectivity").replay;
        let hist = valence_histogram(&r.table.triangles());
        let expected: BTreeMap<usize, usize> = [(5, 48), (6, 312), (24, 2)].into_iter().collect();
        assert_eq!(hist, expected, "valences disagree with draco3d");
    }

    /// The bunny replays too, holes and all.
    ///
    /// A second file, and a harder one: 69451 faces, topology splits, and five
    /// boundary loops, which is the case where the traversal starts on an open
    /// boundary and adds no face for it. Its Euler characteristic is -3 — that is
    /// 2 - 2g - b with genus 0 and five holes — so this exercises the branch the
    /// closed sphere never reaches.
    #[test]
    fn the_bunny_replays_with_its_boundaries() {
        let r = edgebreaker_connectivity(BUNNY).expect("connectivity").replay;
        let table = &r.table;
        assert_eq!(table.num_faces(), 69451);

        let boundary_corners = (0..table.num_corners())
            .filter(|c| table.opposite(*c) == INVALID_CORNER)
            .count();
        assert!(boundary_corners > 0, "the bunny has open boundaries");
        for c in 0..table.num_corners() {
            let o = table.opposite(c);
            if o != INVALID_CORNER {
                assert_eq!(table.opposite(o), c, "opposite is not an involution at {c}");
            }
            assert_ne!(table.vertex(c), INVALID_VERTEX, "corner {c} has no vertex");
        }

        let tris = table.triangles();
        let live = adjacency(&tris).len();
        let edges = edge_count(&tris);
        assert_eq!(live, 34834, "the point count draco3d reports");
        assert_eq!(edges, 104288);
        assert_eq!(
            live as i64 - edges as i64 + tris.len() as i64,
            -3,
            "V - E + F for a genus-0 surface with five holes"
        );
        // An interior edge is opposite two corners and a boundary edge only one,
        // so 3F = 2E - B: the corners with no opposite count the boundary twice
        // over from the face side and once from the edge side.
        assert_eq!(
            boundary_corners,
            2 * edges - 3 * tris.len(),
            "boundary edges, counted two ways"
        );
        assert!(
            r.is_vert_hole.iter().any(|h| *h),
            "vertices on the five holes are boundary vertices"
        );
    }

    /// And the bunny's valences, against the reference decoder.
    ///
    /// A 69451-face mesh with ten distinct valences is a far stronger fit than
    /// the sphere: the histogram is effectively a fingerprint, and a replay that
    /// stitched one split wrongly would move at least two of these counts.
    #[test]
    fn the_bunny_valences_match_the_reference_decoder() {
        let r = edgebreaker_connectivity(BUNNY).expect("connectivity").replay;
        let hist = valence_histogram(&r.table.triangles());
        let expected: BTreeMap<usize, usize> = [
            (2, 2),
            (3, 40),
            (4, 461),
            (5, 3948),
            (6, 26169),
            (7, 3889),
            (8, 299),
            (9, 22),
            (10, 3),
            (11, 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(hist, expected, "valences disagree with draco3d");
        assert_eq!(hist.values().sum::<usize>(), 34834);
    }

    /// Every vertex's corner ring is consistent with the faces.
    ///
    /// `vertex_corners` is the walk the attribute predictors will run millions of
    /// times, and it is easy to write one that terminates and visits the wrong
    /// set. Each ring must be exactly the corners that the face list puts on that
    /// vertex — no more, no fewer, no repeats.
    #[test]
    fn corner_rings_agree_with_the_face_list() {
        let r = edgebreaker_connectivity(SPHERE).expect("connectivity").replay;
        let table = &r.table;
        let mut from_faces: HashMap<u32, HashSet<u32>> = HashMap::new();
        for c in 0..table.num_corners() {
            from_faces.entry(table.vertex(c)).or_default().insert(c);
        }
        for v in 0..table.num_vertices() {
            let ring = table.vertex_corners(v);
            let unique: HashSet<u32> = ring.iter().copied().collect();
            assert_eq!(unique.len(), ring.len(), "vertex {v} ring repeats a corner");
            let expected = from_faces.remove(&v).unwrap_or_default();
            assert_eq!(unique, expected, "vertex {v} ring disagrees with the faces");
        }
    }

    /// Truncation and corruption are errors, not panics.
    ///
    /// The replay indexes three arrays by values it read out of the file, so this
    /// is the code where a corrupt stream would otherwise become a panic in a
    /// browser. Every prefix, and every single-byte flip in the first 512, has to
    /// come back as an error or a mesh.
    #[test]
    fn corrupt_streams_do_not_panic() {
        for cut in 0..SPHERE.len().min(600) {
            let _ = edgebreaker_connectivity(&SPHERE[..cut]);
        }
        let mut broken = SPHERE.to_vec();
        for i in 0..512.min(broken.len()) {
            let old = broken[i];
            for flip in [0x01u8, 0x80, 0xff] {
                broken[i] = old ^ flip;
                let _ = edgebreaker_connectivity(&broken);
            }
            broken[i] = old;
        }
    }
}
