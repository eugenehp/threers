//! Attribute values: the order they were written in, and the predictions.
//!
//! Draco does not store attribute values in point order. It walks the surface
//! depth-first and writes each value the first time its vertex is reached, so
//! the order is a function of the connectivity — which is why the corner table
//! has to be exactly right before a single position can be read. Each value is
//! then stored as a *correction* to a prediction made from neighbours already
//! decoded, and the prediction is recomputed here from the same walk.
//!
//! Three things in this file are ordering that cannot be inferred from the data,
//! only from the encoder:
//!
//! - Within one attribute decoder, every attribute's corrections are read first,
//!   then every attribute's quantization parameters, then the transforms are
//!   applied. Not attribute by attribute.
//! - A prediction scheme reads its own data *before* its transform reads the
//!   min/max bounds. Constrained multi-parallelogram has crease flags, and they
//!   come first.
//! - Corrections are zig-zag coded unless the scheme says its corrections are
//!   already positive, and the wrap transform says they are not.
//!
//! Each of those decodes to plausible numbers if it is wrong, so each is checked
//! against Google's decoder on real files rather than reasoned about.

use super::attribute_table::AttributeCornerTable;
use super::buffer::Buffer;
use super::corner_table::{CornerTable, INVALID_CORNER, INVALID_VERTEX};
use super::octahedron::{OctahedronPredictionTransform, OctahedronToolBox};
use super::rans::{self, BitDecoder};
use super::{DecodedConnectivity, DracoError};

/// Draco's `GeometryAttribute::Type`.
pub mod attribute_type {
    pub const POSITION: u8 = 0;
    pub const NORMAL: u8 = 1;
    pub const COLOR: u8 = 2;
    pub const TEX_COORD: u8 = 3;
    pub const GENERIC: u8 = 4;
    pub const NAMED_COUNT: u8 = 5;
}

/// Draco's `DataType`.
pub mod data_type {
    pub const INVALID: u8 = 0;
    pub const INT8: u8 = 1;
    pub const UINT8: u8 = 2;
    pub const INT16: u8 = 3;
    pub const UINT16: u8 = 4;
    pub const INT32: u8 = 5;
    pub const UINT32: u8 = 6;
    pub const INT64: u8 = 7;
    pub const UINT64: u8 = 8;
    pub const FLOAT32: u8 = 9;
    pub const FLOAT64: u8 = 10;
    pub const BOOL: u8 = 11;
    pub const COUNT: u8 = 12;

    /// Bytes one value of this type occupies.
    pub fn length(dt: u8) -> usize {
        match dt {
            INT8 | UINT8 | BOOL => 1,
            INT16 | UINT16 => 2,
            INT32 | UINT32 | FLOAT32 => 4,
            INT64 | UINT64 | FLOAT64 => 8,
            _ => 0,
        }
    }
}

/// Which sequential decoder reads an attribute's values.
mod sequential {
    pub const GENERIC: u8 = 0;
    pub const INTEGER: u8 = 1;
    pub const QUANTIZATION: u8 = 2;
    pub const NORMALS: u8 = 3;
}

/// Draco's `PredictionSchemeMethod`, as read from the stream.
mod prediction {
    pub const NONE: i8 = -2;
    pub const DIFFERENCE: i8 = 0;
    pub const PARALLELOGRAM: i8 = 1;
    pub const MULTI_PARALLELOGRAM: i8 = 2;
    pub const TEX_COORDS_DEPRECATED: i8 = 3;
    pub const CONSTRAINED_MULTI_PARALLELOGRAM: i8 = 4;
    pub const TEX_COORDS_PORTABLE: i8 = 5;
    pub const GEOMETRIC_NORMAL: i8 = 6;
    pub const COUNT: i8 = 7;
}

/// Draco's `PredictionSchemeTransformType`.
mod transform {
    pub const NONE: i8 = -1;
    pub const DELTA: i8 = 0;
    pub const WRAP: i8 = 1;
    pub const NORMAL_OCTAHEDRON: i8 = 2;
    pub const NORMAL_OCTAHEDRON_CANONICALIZED: i8 = 3;
    pub const COUNT: i8 = 4;
}

/// `MeshAttributeElementType` — whether values sit on vertices or on corners.
mod element {
    pub const VERTEX: u8 = 0;
    pub const CORNER: u8 = 1;
}

/// The most parallelograms the constrained scheme will average.
const MAX_NUM_PARALLELOGRAMS: usize = 4;

/// A connectivity a traversal can walk, either the base table or one attribute's.
///
/// Per-vertex attributes walk the base corner table; per-corner attributes walk
/// the attribute table, where a seam makes two vertices out of one. The
/// traversal, the prediction and the sequencing are identical either way, which
/// is the only reason this abstraction exists.
pub trait Topology {
    fn num_faces(&self) -> u32;
    fn num_corners(&self) -> u32;
    fn num_vertices(&self) -> u32;
    fn vertex(&self, corner: u32) -> u32;
    fn opposite(&self, corner: u32) -> u32;
    fn next(&self, corner: u32) -> u32;
    fn previous(&self, corner: u32) -> u32;
    fn leftmost_corner(&self, vert: u32) -> u32;
    fn swing_left(&self, corner: u32) -> u32;
    fn swing_right(&self, corner: u32) -> u32;

    fn left_corner(&self, corner: u32) -> u32 {
        if corner == INVALID_CORNER {
            return INVALID_CORNER;
        }
        self.opposite(self.previous(corner))
    }

    fn right_corner(&self, corner: u32) -> u32 {
        if corner == INVALID_CORNER {
            return INVALID_CORNER;
        }
        self.opposite(self.next(corner))
    }

    /// A vertex whose ring does not close.
    fn is_on_boundary(&self, vert: u32) -> bool {
        let corner = self.leftmost_corner(vert);
        if corner == INVALID_CORNER {
            return true;
        }
        self.swing_left(corner) == INVALID_CORNER
    }
}

impl Topology for CornerTable {
    fn num_faces(&self) -> u32 {
        CornerTable::num_faces(self)
    }
    fn num_corners(&self) -> u32 {
        CornerTable::num_corners(self)
    }
    fn num_vertices(&self) -> u32 {
        CornerTable::num_vertices(self)
    }
    fn vertex(&self, corner: u32) -> u32 {
        CornerTable::vertex(self, corner)
    }
    fn opposite(&self, corner: u32) -> u32 {
        CornerTable::opposite(self, corner)
    }
    fn next(&self, corner: u32) -> u32 {
        CornerTable::next(self, corner)
    }
    fn previous(&self, corner: u32) -> u32 {
        CornerTable::previous(self, corner)
    }
    fn leftmost_corner(&self, vert: u32) -> u32 {
        CornerTable::leftmost_corner(self, vert)
    }
    fn swing_left(&self, corner: u32) -> u32 {
        CornerTable::swing_left(self, corner)
    }
    fn swing_right(&self, corner: u32) -> u32 {
        CornerTable::swing_right(self, corner)
    }
}

/// One attribute's connectivity, seams cut, over the base table's corners.
pub struct SeamTopology<'a> {
    pub base: &'a CornerTable,
    pub attr: &'a AttributeCornerTable,
}

impl Topology for SeamTopology<'_> {
    fn num_faces(&self) -> u32 {
        self.base.num_faces()
    }
    fn num_corners(&self) -> u32 {
        self.base.num_corners()
    }
    fn num_vertices(&self) -> u32 {
        self.attr.num_vertices()
    }
    fn vertex(&self, corner: u32) -> u32 {
        self.attr.vertex(corner)
    }
    fn opposite(&self, corner: u32) -> u32 {
        self.attr.opposite(self.base, corner)
    }
    fn next(&self, corner: u32) -> u32 {
        self.base.next(corner)
    }
    fn previous(&self, corner: u32) -> u32 {
        self.base.previous(corner)
    }
    fn leftmost_corner(&self, vert: u32) -> u32 {
        self.attr.leftmost_corner(vert)
    }
    fn swing_left(&self, corner: u32) -> u32 {
        self.attr.swing_left(self.base, corner)
    }
    fn swing_right(&self, corner: u32) -> u32 {
        self.attr.swing_right(self.base, corner)
    }
}

/// The order values were written in, and the maps the predictors need.
#[derive(Debug, Clone, Default)]
pub struct Sequence {
    /// The point each value belongs to, in decode order.
    pub point_ids: Vec<u32>,
    /// One corner per value — where the predictor looks for neighbours.
    pub value_to_corner: Vec<u32>,
    /// Which value each vertex got. Zero for a vertex never reached, matching
    /// Draco, whose `resize` leaves zeros here rather than the -1 its comment
    /// claims.
    pub vertex_to_value: Vec<i32>,
    pub num_values: u32,
}

/// Walk the surface depth-first, recording each vertex the first time it is seen.
///
/// This is Draco's `DepthFirstTraverser` driven by `MeshTraversalSequencer`: one
/// attempt to start from corner `3f` of every face, and each attempt runs until
/// it can reach nothing new. The order is the whole point — get it wrong and
/// every value lands on the wrong vertex, which still produces a mesh.
pub fn traverse_depth_first(
    topo: &dyn Topology,
    corner_to_point: &[u32],
    map_size: u32,
) -> Result<Sequence, DracoError> {
    let mut seq = Sequence {
        point_ids: Vec::new(),
        value_to_corner: Vec::new(),
        vertex_to_value: vec![0i32; map_size as usize],
        num_values: 0,
    };
    let mut face_visited = vec![false; topo.num_faces() as usize];
    let mut vertex_visited = vec![false; topo.num_vertices() as usize];

    for face in 0..topo.num_faces() {
        traverse_from_corner(
            topo,
            face * 3,
            corner_to_point,
            &mut seq,
            &mut face_visited,
            &mut vertex_visited,
        )?;
    }
    Ok(seq)
}

/// Record a newly reached vertex.
fn visit_vertex(
    vert: u32,
    corner: u32,
    corner_to_point: &[u32],
    seq: &mut Sequence,
    vertex_visited: &mut [bool],
) -> Result<(), DracoError> {
    let slot = vertex_visited
        .get_mut(vert as usize)
        .ok_or(DracoError::Corrupt)?;
    *slot = true;
    let point = *corner_to_point
        .get(corner as usize)
        .ok_or(DracoError::Corrupt)?;
    seq.point_ids.push(point);
    seq.value_to_corner.push(corner);
    *seq.vertex_to_value
        .get_mut(vert as usize)
        .ok_or(DracoError::Corrupt)? = seq.num_values as i32;
    seq.num_values += 1;
    Ok(())
}

fn traverse_from_corner(
    topo: &dyn Topology,
    start: u32,
    corner_to_point: &[u32],
    seq: &mut Sequence,
    face_visited: &mut [bool],
    vertex_visited: &mut [bool],
) -> Result<(), DracoError> {
    let is_face_visited = |face_visited: &[bool], corner: u32| -> bool {
        if corner == INVALID_CORNER {
            return true; // A face that does not exist counts as done.
        }
        face_visited
            .get((corner / 3) as usize)
            .copied()
            .unwrap_or(true)
    };
    if is_face_visited(face_visited, start) {
        return Ok(());
    }

    // The seed face's other two corners may not be reached by the walk below, so
    // they are taken first.
    let next_vert = topo.vertex(topo.next(start));
    let prev_vert = topo.vertex(topo.previous(start));
    if next_vert == INVALID_VERTEX || prev_vert == INVALID_VERTEX {
        return Err(DracoError::Corrupt);
    }
    for (vert, corner) in [
        (next_vert, topo.next(start)),
        (prev_vert, topo.previous(start)),
    ] {
        if !vertex_visited
            .get(vert as usize)
            .copied()
            .ok_or(DracoError::Corrupt)?
        {
            visit_vertex(vert, corner, corner_to_point, seq, vertex_visited)?;
        }
    }

    let mut stack: Vec<u32> = vec![start];
    let mut guard = 0u64;
    let limit = 4 * topo.num_corners() as u64 + 16;
    while let Some(top) = stack.last().copied() {
        let mut corner = top;
        if is_face_visited(face_visited, corner) {
            stack.pop();
            continue;
        }
        loop {
            guard += 1;
            if guard > limit {
                return Err(DracoError::Corrupt);
            }
            face_visited[(corner / 3) as usize] = true;
            let vert = topo.vertex(corner);
            if vert == INVALID_VERTEX {
                return Err(DracoError::Corrupt);
            }
            if !vertex_visited
                .get(vert as usize)
                .copied()
                .ok_or(DracoError::Corrupt)?
            {
                let on_boundary = topo.is_on_boundary(vert);
                visit_vertex(vert, corner, corner_to_point, seq, vertex_visited)?;
                if !on_boundary {
                    // An interior vertex can be circled, so keep going right.
                    corner = topo.right_corner(corner);
                    continue;
                }
            }
            // Otherwise look at the two neighbouring faces.
            let right = topo.right_corner(corner);
            let left = topo.left_corner(corner);
            let right_done = is_face_visited(face_visited, right);
            let left_done = is_face_visited(face_visited, left);
            if right_done && left_done {
                stack.pop();
                break;
            } else if right_done {
                corner = left;
            } else if left_done {
                corner = right;
            } else {
                // Both open: park the left for later and take the right now.
                *stack.last_mut().ok_or(DracoError::Corrupt)? = left;
                stack.push(right);
                break;
            }
        }
    }
    Ok(())
}

/// What an attribute declares about itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttributeDescriptor {
    pub att_type: u8,
    pub data_type: u8,
    pub num_components: u8,
    pub normalized: bool,
    pub unique_id: u32,
}

/// One attribute's values, one entry per point.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValues {
    /// Integers as stored.
    Integer(Vec<i32>),
    /// Dequantized, or read as floats.
    Float(Vec<f32>),
}

/// A decoded attribute.
#[derive(Debug, Clone)]
pub struct DecodedAttribute {
    pub descriptor: AttributeDescriptor,
    /// `num_components` entries per point, indexed by point id.
    pub values: AttributeValues,
}

impl DecodedAttribute {
    pub fn num_components(&self) -> usize {
        self.descriptor.num_components as usize
    }

    /// The values as floats whatever they were stored as.
    pub fn as_f32(&self) -> Vec<f32> {
        match &self.values {
            AttributeValues::Float(v) => v.clone(),
            AttributeValues::Integer(v) => v.iter().map(|x| *x as f32).collect(),
        }
    }
}

/// The wrap transform: corrections stored modulo the value range.
///
/// The encoder subtracts the prediction and wraps the result into
/// `[-N/2, N/2]`, which keeps the corrections small and their entropy low. The
/// decoder adds and unwraps. Adding in `u32` is deliberate — a malformed file can
/// otherwise overflow a signed add, which is undefined in C++ and a panic here.
#[derive(Debug, Clone, Copy, Default)]
struct WrapTransform {
    num_components: usize,
    min_value: i32,
    max_value: i32,
    max_dif: i32,
}

impl WrapTransform {
    fn decode(buffer: &mut Buffer<'_>, num_components: usize) -> Result<Self, DracoError> {
        let min_value = buffer.u32()? as i32;
        let max_value = buffer.u32()? as i32;
        if min_value > max_value {
            return Err(DracoError::Corrupt);
        }
        let dif = max_value as i64 - min_value as i64;
        if dif < 0 || dif >= i32::MAX as i64 {
            return Err(DracoError::Corrupt);
        }
        Ok(WrapTransform {
            num_components,
            min_value,
            max_value,
            max_dif: 1 + dif as i32,
        })
    }

    fn compute_original(&self, predicted: &[i32], corr: &[i32], out: &mut [i32]) {
        for i in 0..self.num_components {
            let clamped = predicted[i].clamp(self.min_value, self.max_value);
            let mut v = (clamped as u32).wrapping_add(corr[i] as u32) as i32;
            if v > self.max_value {
                v = v.wrapping_sub(self.max_dif);
            } else if v < self.min_value {
                v = v.wrapping_add(self.max_dif);
            }
            out[i] = v;
        }
    }
}

/// The parallelogram prediction at one corner.
///
/// The opposite face gives three already-decoded values; the fourth corner of
/// the parallelogram they form is `next + previous - opposite`. It only applies
/// when all three were decoded *before* the value being predicted, which is what
/// the `< entry` comparisons are.
fn parallelogram_prediction(
    entry: usize,
    corner: u32,
    topo: &dyn Topology,
    vertex_to_value: &[i32],
    data: &[i32],
    num_components: usize,
    out: &mut [i32],
) -> bool {
    let opp = topo.opposite(corner);
    if opp == INVALID_CORNER {
        return false;
    }
    let entry_of = |c: u32| -> Option<usize> {
        let v = topo.vertex(c);
        if v == INVALID_VERTEX {
            return None;
        }
        let e = *vertex_to_value.get(v as usize)?;
        if e < 0 {
            return None;
        }
        Some(e as usize)
    };
    let (Some(v_opp), Some(v_next), Some(v_prev)) = (
        entry_of(opp),
        entry_of(topo.next(opp)),
        entry_of(topo.previous(opp)),
    ) else {
        return false;
    };
    if v_opp >= entry || v_next >= entry || v_prev >= entry {
        return false;
    }
    for c in 0..num_components {
        let next = data[v_next * num_components + c] as i64;
        let prev = data[v_prev * num_components + c] as i64;
        let opp = data[v_opp * num_components + c] as i64;
        out[c] = (next + prev - opp) as i32;
    }
    true
}

/// A prediction transform: how a correction becomes a value.
enum PredTransform {
    Wrap(WrapTransform),
    /// Normals, where the grid wraps and has to be rotated into one quadrant.
    Octahedron(OctahedronPredictionTransform),
}

impl PredTransform {
    /// Whether corrections were stored already-positive.
    ///
    /// The wrap transform says no and the octahedral one says yes, and this is
    /// what decides whether the symbols get un-zig-zagged. Reading it from the
    /// transform rather than assuming is the point: treating positive
    /// corrections as zig-zagged halves every one of them.
    fn corrections_are_positive(&self) -> bool {
        match self {
            PredTransform::Wrap(_) => false,
            PredTransform::Octahedron(t) => t.corrections_are_positive(),
        }
    }

    fn compute_original(&self, predicted: &[i32], corr: &[i32], out: &mut [i32]) {
        match self {
            PredTransform::Wrap(t) => t.compute_original(predicted, corr, out),
            PredTransform::Octahedron(t) => t.compute_original(predicted, corr, out),
        }
    }
}

/// The prediction scheme, with whatever data it read for itself.
enum Scheme<'a> {
    Difference,
    Parallelogram,
    /// Per-context flags saying which of the available parallelograms to skip.
    ConstrainedMultiParallelogram(Vec<Vec<bool>>),
    /// Normals predicted from the surrounding faces, plus one bit per value
    /// saying whether the predicted normal points the wrong way.
    GeometricNormal(BitDecoder<'a>),
    /// Texture coordinates predicted by carrying the triangle's shape from
    /// position space into UV space, plus one bit per value choosing which side
    /// of the edge the third corner falls on.
    TexCoordsPortable(Vec<bool>),
}

/// Read a scheme's own data and its transform's, in the order the encoder wrote
/// them.
///
/// These two orders are opposite and that is not a mistake: constrained
/// multi-parallelogram writes its crease flags *before* the transform bounds,
/// and geometric normal writes the transform data *before* its flip bits.
fn decode_prediction_data<'a>(
    method: i8,
    transform_type: i8,
    buffer: &mut Buffer<'a>,
    num_components: usize,
    num_corners: u32,
) -> Result<(Scheme<'a>, PredTransform), DracoError> {
    let read_transform = |buffer: &mut Buffer<'a>| -> Result<PredTransform, DracoError> {
        match transform_type {
            transform::WRAP => Ok(PredTransform::Wrap(WrapTransform::decode(
                buffer,
                num_components,
            )?)),
            transform::NORMAL_OCTAHEDRON_CANONICALIZED => {
                let max_quantized_value = buffer.u32()? as i32;
                let center_value = buffer.u32()? as i32;
                Ok(PredTransform::Octahedron(
                    OctahedronPredictionTransform::decode(max_quantized_value, center_value)?,
                ))
            }
            transform::NORMAL_OCTAHEDRON => {
                Err(DracoError::Unsupported("uncanonicalized octahedral normals"))
            }
            transform::DELTA => Err(DracoError::Unsupported("delta prediction transform")),
            _ => Err(DracoError::Unsupported("prediction transform")),
        }
    };

    match method {
        prediction::DIFFERENCE => Ok((Scheme::Difference, read_transform(buffer)?)),
        prediction::PARALLELOGRAM => Ok((Scheme::Parallelogram, read_transform(buffer)?)),
        prediction::CONSTRAINED_MULTI_PARALLELOGRAM => {
            // Crease flags first, then the transform.
            let mut contexts = Vec::with_capacity(MAX_NUM_PARALLELOGRAMS);
            for _ in 0..MAX_NUM_PARALLELOGRAMS {
                let num_flags = buffer.varint32()?;
                if num_flags > num_corners {
                    return Err(DracoError::Corrupt);
                }
                let mut flags = Vec::with_capacity(num_flags as usize);
                if num_flags > 0 {
                    let mut decoder = BitDecoder::start(buffer)?;
                    for _ in 0..num_flags {
                        flags.push(decoder.bit() != 0);
                    }
                }
                contexts.push(flags);
            }
            let transform = read_transform(buffer)?;
            Ok((Scheme::ConstrainedMultiParallelogram(contexts), transform))
        }
        prediction::GEOMETRIC_NORMAL => {
            // Transform first, then the flip bits.
            let transform = read_transform(buffer)?;
            let flips = BitDecoder::start(buffer)?;
            Ok((Scheme::GeometricNormal(flips), transform))
        }
        prediction::MULTI_PARALLELOGRAM => {
            Err(DracoError::Unsupported("multi-parallelogram prediction"))
        }
        prediction::TEX_COORDS_PORTABLE => {
            // Orientations first, then the transform — the same way round as
            // constrained multi-parallelogram, and opposite to geometric normal.
            let num_orientations = buffer.u32()? as i32;
            if num_orientations < 0 || num_orientations as u32 > num_corners {
                return Err(DracoError::Corrupt);
            }
            let mut decoder = BitDecoder::start(buffer)?;
            let mut orientations = Vec::with_capacity(num_orientations as usize);
            // Delta coded: a zero bit flips the running orientation, a one keeps
            // it. Runs of the same orientation are common, which is the point.
            let mut last = true;
            for _ in 0..num_orientations {
                if decoder.bit() == 0 {
                    last = !last;
                }
                orientations.push(last);
            }
            let transform = read_transform(buffer)?;
            Ok((Scheme::TexCoordsPortable(orientations), transform))
        }
        prediction::TEX_COORDS_DEPRECATED => Err(DracoError::Unsupported(
            "deprecated texture coordinate prediction",
        )),
        _ => Err(DracoError::Corrupt),
    }
}

/// An attribute already decoded, in the integer form predictors consume.
///
/// Geometric normal prediction needs the *quantized* positions, not the
/// dequantized ones — it works on the integer grid so that encoder and decoder
/// agree exactly.
pub struct PortableAttribute {
    pub att_type: u8,
    pub num_components: usize,
    pub values: Vec<i32>,
    pub point_to_value: Vec<usize>,
}

impl PortableAttribute {
    fn value(&self, point: usize) -> Option<&[i32]> {
        let v = *self.point_to_value.get(point)?;
        let at = v.checked_mul(self.num_components)?;
        self.values.get(at..at + self.num_components)
    }
}

/// Every corner on the same vertex as `start`, beginning at `start` itself.
///
/// Draco's `VertexCornersIterator` built from a corner rather than a vertex:
/// swing left until the ring ends, then resume one step right of the start.
fn corner_ring_from(topo: &dyn Topology, start: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut corner = start;
    let mut going_left = true;
    while corner != INVALID_CORNER && out.len() <= topo.num_corners() as usize {
        out.push(corner);
        if going_left {
            corner = topo.swing_left(corner);
            if corner == INVALID_CORNER {
                corner = topo.swing_right(start);
                going_left = false;
            } else if corner == start {
                corner = INVALID_CORNER;
            }
        } else {
            corner = topo.swing_right(corner);
        }
    }
    out
}

/// Exact floor of a square root, by Newton's method on integers.
///
/// Draco's `IntSqrt`. It has to be this and not `(x as f64).sqrt()`: the value
/// feeds straight into a predicted coordinate, so encoder and decoder must agree
/// on the last bit, and a double cannot represent every `u64` exactly.
fn int_sqrt(number: u64) -> u64 {
    if number == 0 {
        return 0;
    }
    // Start from a power of two near the answer, found by halving the exponent.
    let mut act_number = number;
    let mut square_root = 1u64;
    while act_number >= 2 {
        square_root *= 2;
        act_number /= 4;
    }
    loop {
        square_root = (square_root + number / square_root) / 2;
        // After the first step the estimate never undershoots, so squaring it and
        // comparing is a sufficient convergence test.
        if square_root.saturating_mul(square_root) <= number {
            return square_root;
        }
    }
}

/// The quantized position belonging to one decoded attribute entry.
fn portable_position(
    entry: usize,
    seq: &Sequence,
    positions: &PortableAttribute,
) -> Result<[i64; 3], DracoError> {
    let point = *seq.point_ids.get(entry).ok_or(DracoError::Corrupt)? as usize;
    let v = positions.value(point).ok_or(DracoError::Corrupt)?;
    Ok([v[0] as i64, v[1] as i64, v[2] as i64])
}

/// Texture coordinates predicted from the triangle's shape in position space.
///
/// The two UVs already decoded on this triangle fix an edge in UV space, and the
/// third position's projection onto the corresponding edge in *position* space
/// says where along it the third UV sits and how far off it stands. Carrying that
/// across predicts the UV without any per-vertex data at all — all the file has
/// to store is one bit per value for which side of the edge it lands on.
///
/// Every step is integer arithmetic scaled by `|PN|²` so that no division is
/// needed until the end, and the wrapping is in the unsigned domain because
/// Draco's is: a malformed file can make these intermediates exceed `i64`, and
/// both implementations must wrap identically rather than one of them panicking.
#[allow(clippy::too_many_arguments)]
fn tex_coords_portable_prediction(
    data_id: usize,
    corner: u32,
    topo: &dyn Topology,
    seq: &Sequence,
    positions: &PortableAttribute,
    data: &[i32],
    orientations: &mut Vec<bool>,
) -> Result<[i32; 2], DracoError> {
    let next_data = *seq
        .vertex_to_value
        .get(topo.vertex(topo.next(corner)) as usize)
        .ok_or(DracoError::Corrupt)?;
    let prev_data = *seq
        .vertex_to_value
        .get(topo.vertex(topo.previous(corner)) as usize)
        .ok_or(DracoError::Corrupt)?;
    let uv_at = |entry: i32| -> Result<[i64; 2], DracoError> {
        if entry < 0 {
            return Err(DracoError::Corrupt);
        }
        let at = (entry as usize).checked_mul(2).ok_or(DracoError::Corrupt)?;
        let pair = data.get(at..at + 2).ok_or(DracoError::Corrupt)?;
        Ok([pair[0] as i64, pair[1] as i64])
    };
    let here = data_id as i32;

    if prev_data < here && next_data < here {
        let n_uv = uv_at(next_data)?;
        let p_uv = uv_at(prev_data)?;
        if p_uv == n_uv {
            // A degenerate UV edge carries no direction, so there is nothing to
            // rotate and the prediction is just that point.
            return Ok([p_uv[0] as i32, p_uv[1] as i32]);
        }
        let tip_pos = portable_position(data_id, seq, positions)?;
        let next_pos = portable_position(next_data as usize, seq, positions)?;
        let prev_pos = portable_position(prev_data as usize, seq, positions)?;
        let pn = [
            prev_pos[0] - next_pos[0],
            prev_pos[1] - next_pos[1],
            prev_pos[2] - next_pos[2],
        ];
        let pn_norm2_squared = (pn[0] * pn[0] + pn[1] * pn[1] + pn[2] * pn[2]) as u64;
        if pn_norm2_squared != 0 {
            let cn = [
                tip_pos[0] - next_pos[0],
                tip_pos[1] - next_pos[1],
                tip_pos[2] - next_pos[2],
            ];
            let cn_dot_pn = pn[0] * cn[0] + pn[1] * cn[1] + pn[2] * cn[2];
            let pn_uv = [p_uv[0] - n_uv[0], p_uv[1] - n_uv[1]];

            // Draco refuses rather than wrapping where a product would leave
            // `i64`, and the refusals are part of the format: a file that would
            // overflow is malformed, so matching them keeps the two decoders
            // agreeing on which files are valid.
            let n_uv_absmax = n_uv[0].abs().max(n_uv[1].abs()) as u64;
            let limit = (i64::MAX as u64)
                .checked_div(pn_norm2_squared)
                .ok_or(DracoError::Corrupt)?;
            if n_uv_absmax > limit {
                return Err(DracoError::Corrupt);
            }
            let pn_uv_absmax = pn_uv[0].abs().max(pn_uv[1].abs());
            let limit = i64::MAX
                .checked_div(pn_uv_absmax)
                .ok_or(DracoError::Corrupt)?;
            if cn_dot_pn.abs() > limit {
                return Err(DracoError::Corrupt);
            }
            let pn_absmax = pn[0].abs().max(pn[1].abs()).max(pn[2].abs());
            let limit = i64::MAX.checked_div(pn_absmax).ok_or(DracoError::Corrupt)?;
            if cn_dot_pn.abs() > limit {
                return Err(DracoError::Corrupt);
            }

            // X is where the tip projects onto the edge, in UV space, scaled.
            let mut x_uv = [0i64; 2];
            for i in 0..2 {
                let a = (n_uv[i] as u64).wrapping_mul(pn_norm2_squared);
                let b = (cn_dot_pn as u64).wrapping_mul(pn_uv[i] as u64);
                x_uv[i] = a.wrapping_add(b) as i64;
            }
            // And the same projection in position space, to measure how far the
            // tip stands off the edge.
            // Signed, because Draco divides by this as the vector's own scalar
            // type; a value above `i64::MAX` becomes negative in both decoders.
            let divisor = pn_norm2_squared as i64;
            let mut x_pos = [0i64; 3];
            for i in 0..3 {
                let step = cn_dot_pn
                    .wrapping_mul(pn[i])
                    .checked_div(divisor)
                    .ok_or(DracoError::Corrupt)?;
                x_pos[i] = next_pos[i] + step;
            }
            let cx = [
                tip_pos[0] - x_pos[0],
                tip_pos[1] - x_pos[1],
                tip_pos[2] - x_pos[2],
            ];
            let cx_norm2_squared = (cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]) as u64;

            // The offset direction is the UV edge turned ninety degrees, scaled
            // by |CX| * |PN| so it stays in the same scaled space as x_uv.
            let mut cx_uv = [pn_uv[1], -pn_uv[0]];
            let norm = int_sqrt(cx_norm2_squared.wrapping_mul(pn_norm2_squared));
            for v in cx_uv.iter_mut() {
                *v = (*v as u64).wrapping_mul(norm) as i64;
            }

            // Which side of the edge — the one thing the file has to say.
            let orientation = orientations.pop().ok_or(DracoError::Corrupt)?;
            let mut predicted = [0i32; 2];
            for i in 0..2 {
                let sum = if orientation {
                    (x_uv[i] as u64).wrapping_add(cx_uv[i] as u64)
                } else {
                    (x_uv[i] as u64).wrapping_sub(cx_uv[i] as u64)
                };
                predicted[i] = (sum as i64)
                    .checked_div(divisor)
                    .ok_or(DracoError::Corrupt)? as i32;
            }
            return Ok(predicted);
        }
    }

    // Without two decoded UVs on the triangle there is no edge to carry, so this
    // falls back to delta coding.
    //
    // Draco assigns the previous corner's offset first and then overwrites it in
    // the branch below. That store is dead in every path that can reach here: the
    // branch above already handled the case where both corners have values, so at
    // most one of them does, and whichever `if` fires the value is replaced. The
    // dead store is reproduced by *omitting* it — writing it would change
    // nothing, and "tidying" it into the `else if` it looks like it wants to be
    // would change the result on exactly the meshes where only the previous
    // corner has a value.
    let data_offset = if next_data < here {
        next_data as usize * 2
    } else if data_id > 0 {
        (data_id - 1) * 2
    } else {
        // The very first value has nothing before it.
        return Ok([0, 0]);
    };
    let pair = data
        .get(data_offset..data_offset + 2)
        .ok_or(DracoError::Corrupt)?;
    Ok([pair[0], pair[1]])
}

/// The area-weighted normal of the faces around a corner, on the integer grid.
///
/// Every face around the vertex contributes the cross product of its two edges,
/// which is twice its area times its unit normal — so summing them weights each
/// face by its area without a single division. The sum is then scaled down to fit
/// in 29 bits, because the encoder did the same and the two must agree bit for
/// bit.
fn geometric_normal_prediction(
    corner: u32,
    topo: &dyn Topology,
    seq: &Sequence,
    positions: &PortableAttribute,
) -> Result<[i32; 3], DracoError> {
    let position_at = |ci: u32| -> Result<[i64; 3], DracoError> {
        let vert = topo.vertex(ci);
        if vert == INVALID_VERTEX {
            return Err(DracoError::Corrupt);
        }
        let data_id = *seq
            .vertex_to_value
            .get(vert as usize)
            .ok_or(DracoError::Corrupt)?;
        if data_id < 0 {
            return Err(DracoError::Corrupt);
        }
        portable_position(data_id as usize, seq, positions)
    };

    let centre = position_at(corner)?;
    let mut normal = [0i64; 3];
    for c in corner_ring_from(topo, corner) {
        let next = position_at(topo.next(c))?;
        let prev = position_at(topo.previous(c))?;
        let dn = [
            next[0] - centre[0],
            next[1] - centre[1],
            next[2] - centre[2],
        ];
        let dp = [
            prev[0] - centre[0],
            prev[1] - centre[1],
            prev[2] - centre[2],
        ];
        let cross = [
            dn[1] * dp[2] - dn[2] * dp[1],
            dn[2] * dp[0] - dn[0] * dp[2],
            dn[0] * dp[1] - dn[1] * dp[0],
        ];
        // Summed as unsigned, as Draco does, so a degenerate mesh wraps rather
        // than overflowing.
        for i in 0..3 {
            normal[i] = (normal[i] as u64).wrapping_add(cross[i] as u64) as i64;
        }
    }

    const UPPER_BOUND: i64 = 1 << 29;
    let abs_sum = normal[0].abs() + normal[1].abs() + normal[2].abs();
    if abs_sum > UPPER_BOUND {
        let quotient = abs_sum / UPPER_BOUND;
        if quotient != 0 {
            for n in normal.iter_mut() {
                *n /= quotient;
            }
        }
    }
    Ok([normal[0] as i32, normal[1] as i32, normal[2] as i32])
}

/// Turn corrections back into values.
fn compute_original_values(
    scheme: &mut Scheme<'_>,
    transform: &PredTransform,
    corr: &[i32],
    out: &mut [i32],
    num_components: usize,
    seq: &Sequence,
    topo: &dyn Topology,
    portables: &[PortableAttribute],
) -> Result<(), DracoError> {
    let nc = num_components;
    if out.len() < nc || corr.len() < out.len() {
        return Err(DracoError::Corrupt);
    }
    let entries = seq.value_to_corner.len();
    if entries * nc > out.len() {
        return Err(DracoError::Corrupt);
    }

    // Both of these predict from the positions rather than from neighbouring
    // values of their own attribute, so every entry is predicted — including the
    // first, which the parallelogram schemes have to restore separately.
    if matches!(
        scheme,
        Scheme::GeometricNormal(_) | Scheme::TexCoordsPortable(_)
    ) {
        let positions = portables
            .iter()
            .find(|p| p.att_type == attribute_type::POSITION && p.num_components == 3)
            .ok_or(DracoError::Corrupt)?;
        match scheme {
            Scheme::GeometricNormal(flips) => {
                let bits = match transform {
                    PredTransform::Octahedron(t) => t.quantization_bits(),
                    PredTransform::Wrap(_) => return Err(DracoError::Corrupt),
                };
                let tools = OctahedronToolBox::new(bits)?;
                for p in 0..entries {
                    let corner = seq.value_to_corner[p];
                    let mut pred = geometric_normal_prediction(corner, topo, seq, positions)?;
                    tools.canonicalize_integer_vector(&mut pred);
                    if flips.bit() != 0 {
                        for v in pred.iter_mut() {
                            *v = -*v;
                        }
                    }
                    let (s, t) = tools.integer_vector_to_quantized_octahedral_coords(&pred);
                    let at = p * nc;
                    transform.compute_original(&[s, t], &corr[at..at + nc], &mut out[at..at + nc]);
                }
            }
            Scheme::TexCoordsPortable(orientations) => {
                if nc != 2 {
                    return Err(DracoError::Corrupt);
                }
                for p in 0..entries {
                    let corner = seq.value_to_corner[p];
                    let at = p * nc;
                    let pred = tex_coords_portable_prediction(
                        p,
                        corner,
                        topo,
                        seq,
                        positions,
                        out,
                        orientations,
                    )?;
                    transform.compute_original(&pred, &corr[at..at + nc], &mut out[at..at + nc]);
                }
            }
            _ => unreachable!(),
        }
        return Ok(());
    }

    // The first value has nothing to predict from.
    let zeros = vec![0i32; nc];
    transform.compute_original(&zeros, &corr[..nc], &mut out[..nc]);

    match scheme {
        Scheme::GeometricNormal(_) | Scheme::TexCoordsPortable(_) => {
            Err(DracoError::Corrupt)
        }
        Scheme::Difference => {
            let mut at = nc;
            while at + nc <= out.len() {
                let (head, tail) = out.split_at_mut(at);
                transform.compute_original(&head[at - nc..], &corr[at..at + nc], &mut tail[..nc]);
                at += nc;
            }
            Ok(())
        }
        Scheme::Parallelogram | Scheme::ConstrainedMultiParallelogram(_) => {
            let mut pred = vec![vec![0i32; nc]; MAX_NUM_PARALLELOGRAMS];
            let mut multi = vec![0i32; nc];
            let mut crease_pos = [0usize; MAX_NUM_PARALLELOGRAMS];
            for p in 1..entries {
                let start = seq.value_to_corner[p];
                let at = p * nc;
                let used = match &*scheme {
                    Scheme::Parallelogram => {
                        if parallelogram_prediction(
                            p,
                            start,
                            topo,
                            &seq.vertex_to_value,
                            out,
                            nc,
                            &mut pred[0],
                        ) {
                            multi[..nc].copy_from_slice(&pred[0][..nc]);
                            1
                        } else {
                            0
                        }
                    }
                    Scheme::ConstrainedMultiParallelogram(contexts) => {
                        // Gather every parallelogram available at this vertex:
                        // swing left until the ring ends, then right from the
                        // start. At most four, which is the encoder's cap.
                        let mut found = 0usize;
                        let mut corner = start;
                        let mut first_pass = true;
                        let mut guard = 0u32;
                        while corner != INVALID_CORNER {
                            if parallelogram_prediction(
                                p,
                                corner,
                                topo,
                                &seq.vertex_to_value,
                                out,
                                nc,
                                &mut pred[found],
                            ) {
                                found += 1;
                                if found == MAX_NUM_PARALLELOGRAMS {
                                    break;
                                }
                            }
                            corner = if first_pass {
                                topo.swing_left(corner)
                            } else {
                                topo.swing_right(corner)
                            };
                            if corner == start {
                                break;
                            }
                            if corner == INVALID_CORNER && first_pass {
                                first_pass = false;
                                corner = topo.swing_right(start);
                            }
                            guard += 1;
                            if guard > topo.num_corners() {
                                return Err(DracoError::Corrupt);
                            }
                        }
                        // Then the flags say which of them the encoder kept. The
                        // context is how many were available, so the flag streams
                        // stay separable.
                        let mut kept = 0usize;
                        if found > 0 {
                            multi[..nc].fill(0);
                            let context = found - 1;
                            for candidate in pred.iter().take(found) {
                                let pos = crease_pos[context];
                                crease_pos[context] += 1;
                                let is_crease = *contexts
                                    .get(context)
                                    .and_then(|c| c.get(pos))
                                    .ok_or(DracoError::Corrupt)?;
                                if !is_crease {
                                    kept += 1;
                                    for (slot, add) in
                                        multi[..nc].iter_mut().zip(&candidate[..nc])
                                    {
                                        *slot = (*slot as u32).wrapping_add(*add as u32) as i32;
                                    }
                                }
                            }
                        }
                        kept
                    }
                    _ => unreachable!(),
                };
                if used == 0 {
                    // Nothing to predict from: fall back to the value before.
                    let (head, tail) = out.split_at_mut(at);
                    transform.compute_original(
                        &head[at - nc..],
                        &corr[at..at + nc],
                        &mut tail[..nc],
                    );
                } else {
                    if used > 1 {
                        for slot in multi[..nc].iter_mut() {
                            *slot /= used as i32;
                        }
                    }
                    transform.compute_original(&multi, &corr[at..at + nc], &mut out[at..at + nc]);
                }
            }
            Ok(())
        }
    }
}

/// One attribute decoder's framing, read before any value.
struct DecoderSpec {
    att_data_id: i8,
    element_type: u8,
    descriptors: Vec<AttributeDescriptor>,
    sequential_types: Vec<u8>,
}

/// How an attribute's integers become its final values.
enum OutputTransform {
    /// Already final.
    None,
    Quantization(Quantization),
    /// Two octahedral coordinates into three components of a unit normal.
    Octahedron(OctahedronToolBox),
}

/// Read the attribute section: every decoder, every attribute, every value.
///
/// The buffer must be positioned immediately after the connectivity — which is
/// after the last attribute-seam bit decoder, since those consume their own
/// lengths.
pub fn decode_attributes(
    buffer: &mut Buffer<'_>,
    conn: &DecodedConnectivity,
) -> Result<Vec<DecodedAttribute>, DracoError> {
    let num_decoders = buffer.u8()? as usize;
    if num_decoders == 0 {
        return Ok(Vec::new());
    }
    // Pass one: what each decoder is attached to. Draco reads every decoder's
    // identifier before any descriptor, so these are three separate sweeps.
    let mut specs: Vec<DecoderSpec> = Vec::with_capacity(num_decoders);
    for _ in 0..num_decoders {
        let att_data_id = buffer.u8()? as i8;
        let element_type = buffer.u8()?;
        let traversal_method = buffer.u8()?;
        if traversal_method != 0 {
            // MESH_TRAVERSAL_PREDICTION_DEGREE needs a different traverser.
            return Err(DracoError::Unsupported("prediction-degree traversal"));
        }
        if att_data_id >= 0 && att_data_id as usize >= conn.attribute_tables.len() {
            return Err(DracoError::Corrupt);
        }
        specs.push(DecoderSpec {
            att_data_id,
            element_type,
            descriptors: Vec::new(),
            sequential_types: Vec::new(),
        });
    }

    // Pass two: the attribute descriptors and which decoder reads each.
    for spec in specs.iter_mut() {
        let num_attributes = buffer.varint32()?;
        if num_attributes == 0 || num_attributes as usize > 5 * buffer.remaining() {
            return Err(DracoError::Corrupt);
        }
        for _ in 0..num_attributes {
            let att_type = buffer.u8()?;
            let dt = buffer.u8()?;
            let num_components = buffer.u8()?;
            let normalized = buffer.u8()?;
            if att_type >= attribute_type::NAMED_COUNT {
                return Err(DracoError::Corrupt);
            }
            if dt == data_type::INVALID || dt >= data_type::COUNT {
                return Err(DracoError::Corrupt);
            }
            if num_components == 0 {
                return Err(DracoError::Corrupt);
            }
            let unique_id = buffer.varint32()?;
            spec.descriptors.push(AttributeDescriptor {
                att_type,
                data_type: dt,
                num_components,
                normalized: normalized > 0,
                unique_id,
            });
        }
        for _ in 0..num_attributes {
            spec.sequential_types.push(buffer.u8()?);
        }
    }

    // Pass three: the values. Decoders run in order because a later one can
    // predict from an earlier one's integers — geometric normals need positions.
    let mut out = Vec::new();
    let mut portables: Vec<PortableAttribute> = Vec::new();
    for spec in &specs {
        decode_one_decoder(buffer, conn, spec, &mut out, &mut portables)?;
    }
    Ok(out)
}

/// How many components an attribute's *integer* form has.
///
/// Normals are the exception: three components go in and two come out, because
/// a unit vector has only two degrees of freedom.
fn portable_components(desc: &AttributeDescriptor, seq_type: u8) -> usize {
    if seq_type == sequential::NORMALS {
        2
    } else {
        desc.num_components as usize
    }
}

fn decode_one_decoder(
    buffer: &mut Buffer<'_>,
    conn: &DecodedConnectivity,
    spec: &DecoderSpec,
    out: &mut Vec<DecodedAttribute>,
    portables: &mut Vec<PortableAttribute>,
) -> Result<(), DracoError> {
    let base = &conn.replay.table;
    // A per-vertex attribute is sequenced on the base table; a per-corner one on
    // its own, where a seam has split the vertex in two.
    let seam;
    let (topo, map_size): (&dyn Topology, u32) = if spec.element_type == element::VERTEX {
        (base, base.num_vertices())
    } else if spec.element_type == element::CORNER {
        if spec.att_data_id < 0 {
            return Err(DracoError::Corrupt);
        }
        let attr = &conn.attribute_tables[spec.att_data_id as usize];
        seam = SeamTopology { base, attr };
        (&seam, attr.num_vertices().max(base.num_vertices()))
    } else {
        return Err(DracoError::Unsupported("per-face attributes"));
    };

    let seq = traverse_depth_first(topo, &conn.corner_to_point, map_size)?;
    let num_entries = seq.num_values as usize;

    // Each attribute's corrections, then each attribute's transform data, then
    // the transforms. Interleaving these is the single easiest way to decode a
    // file into plausible nonsense.
    let mut portable: Vec<Vec<i32>> = Vec::with_capacity(spec.descriptors.len());
    for (desc, seq_type) in spec.descriptors.iter().zip(&spec.sequential_types) {
        portable.push(decode_portable(
            buffer,
            desc,
            *seq_type,
            &seq,
            topo,
            num_entries,
            portables,
        )?);
    }

    let mut transforms: Vec<OutputTransform> = Vec::with_capacity(portable.len());
    for (desc, seq_type) in spec.descriptors.iter().zip(&spec.sequential_types) {
        transforms.push(match *seq_type {
            sequential::QUANTIZATION => OutputTransform::Quantization(Quantization::decode(buffer, desc)?),
            sequential::NORMALS => {
                let bits = buffer.u8()? as i32;
                OutputTransform::Octahedron(OctahedronToolBox::new(bits)?)
            }
            _ => OutputTransform::None,
        });
    }

    // Points, not values: several points can share one value, and the index
    // buffer is in points.
    let mut point_to_value = vec![0usize; conn.num_points as usize];
    for corner in 0..base.num_corners() {
        let point = *conn
            .corner_to_point
            .get(corner as usize)
            .ok_or(DracoError::Corrupt)? as usize;
        let vert = topo.vertex(corner);
        if vert == INVALID_VERTEX {
            return Err(DracoError::Corrupt);
        }
        let value = *seq
            .vertex_to_value
            .get(vert as usize)
            .ok_or(DracoError::Corrupt)?;
        if value < 0 || value as usize >= num_entries || point >= point_to_value.len() {
            return Err(DracoError::Corrupt);
        }
        point_to_value[point] = value as usize;
    }

    for (((desc, seq_type), ints), output) in spec
        .descriptors
        .iter()
        .zip(&spec.sequential_types)
        .zip(portable)
        .zip(transforms)
    {
        let pc = portable_components(desc, *seq_type);
        let nc = desc.num_components as usize;
        let values = &ints;
        let final_values = match output {
            OutputTransform::Quantization(q) => {
                let mut floats = vec![0f32; point_to_value.len() * nc];
                for (point, value) in point_to_value.iter().enumerate() {
                    for c in 0..nc {
                        let v = *values
                            .get(value * pc + c)
                            .ok_or(DracoError::Corrupt)?;
                        floats[point * nc + c] = q.dequantize(v, c);
                    }
                }
                AttributeValues::Float(floats)
            }
            OutputTransform::Octahedron(tools) => {
                if nc != 3 {
                    return Err(DracoError::Corrupt);
                }
                let mut floats = vec![0f32; point_to_value.len() * 3];
                for (point, value) in point_to_value.iter().enumerate() {
                    let s = *values.get(value * 2).ok_or(DracoError::Corrupt)?;
                    let t = *values.get(value * 2 + 1).ok_or(DracoError::Corrupt)?;
                    let n = tools.to_unit_vector(s, t);
                    floats[point * 3..point * 3 + 3].copy_from_slice(&n);
                }
                AttributeValues::Float(floats)
            }
            OutputTransform::None => {
                let mut ints = vec![0i32; point_to_value.len() * nc];
                for (point, value) in point_to_value.iter().enumerate() {
                    for c in 0..nc {
                        ints[point * nc + c] =
                            *values.get(value * pc + c).ok_or(DracoError::Corrupt)?;
                    }
                }
                AttributeValues::Integer(ints)
            }
        };
        // Keep the integer form: a later attribute may predict from it, and
        // geometric normals predict from quantized positions rather than the
        // dequantized ones, so that encoder and decoder agree exactly.
        portables.push(PortableAttribute {
            att_type: desc.att_type,
            num_components: pc,
            values: ints,
            point_to_value: point_to_value.clone(),
        });
        out.push(DecodedAttribute {
            descriptor: *desc,
            values: final_values,
        });
    }
    Ok(())
}

/// One attribute's integer values, predictions already undone.
fn decode_portable(
    buffer: &mut Buffer<'_>,
    desc: &AttributeDescriptor,
    seq_type: u8,
    seq: &Sequence,
    topo: &dyn Topology,
    num_entries: usize,
    portables: &[PortableAttribute],
) -> Result<Vec<i32>, DracoError> {
    match seq_type {
        sequential::GENERIC => {
            // Values in their original format, one entry after another.
            let nc = desc.num_components as usize;
            let size = data_type::length(desc.data_type);
            if size == 0 {
                return Err(DracoError::Corrupt);
            }
            let mut out = vec![0i32; num_entries * nc];
            for slot in out.iter_mut() {
                *slot = match size {
                    1 => buffer.u8()? as i32,
                    2 => buffer.u16()? as i32,
                    4 => buffer.u32()? as i32,
                    _ => return Err(DracoError::Unsupported("64-bit attribute values")),
                };
            }
            Ok(out)
        }
        sequential::INTEGER | sequential::QUANTIZATION | sequential::NORMALS => {
            let pc = portable_components(desc, seq_type);
            decode_integer_values(buffer, pc, num_entries, seq, topo, portables)
        }
        _ => Err(DracoError::Corrupt),
    }
}

fn decode_integer_values(
    buffer: &mut Buffer<'_>,
    num_components: usize,
    num_entries: usize,
    seq: &Sequence,
    topo: &dyn Topology,
    portables: &[PortableAttribute],
) -> Result<Vec<i32>, DracoError> {
    let method = buffer.u8()? as i8;
    if !(prediction::NONE..prediction::COUNT).contains(&method) {
        return Err(DracoError::Corrupt);
    }
    let mut transform_type = transform::NONE;
    if method != prediction::NONE {
        transform_type = buffer.u8()? as i8;
        if !(transform::NONE..transform::COUNT).contains(&transform_type) {
            return Err(DracoError::Corrupt);
        }
    }

    let num_values = num_entries
        .checked_mul(num_components)
        .ok_or(DracoError::Corrupt)?;
    let compressed = buffer.u8()?;
    let mut raw: Vec<u32> = if compressed > 0 {
        rans::decode_symbols(num_values, num_components, buffer)?
    } else {
        let num_bytes = buffer.u8()? as usize;
        let mut v = vec![0u32; num_values];
        for slot in v.iter_mut() {
            *slot = match num_bytes {
                1 => buffer.u8()? as u32,
                2 => buffer.u16()? as u32,
                4 => buffer.u32()?,
                _ => return Err(DracoError::Corrupt),
            };
        }
        v
    };

    if method == prediction::NONE {
        // No scheme, so the values are stored directly — still zig-zagged.
        return Ok(raw.iter().map(|v| unzigzag(*v)).collect());
    }

    let (mut scheme, transform) = decode_prediction_data(
        method,
        transform_type,
        buffer,
        num_components,
        topo.num_corners(),
    )?;
    let corr: Vec<i32> = if transform.corrections_are_positive() {
        raw.drain(..).map(|v| v as i32).collect()
    } else {
        raw.drain(..).map(unzigzag).collect()
    };
    let mut out = vec![0i32; num_values];
    compute_original_values(
        &mut scheme,
        &transform,
        &corr,
        &mut out,
        num_components,
        seq,
        topo,
        portables,
    )?;
    Ok(out)
}

/// Undo Draco's zig-zag: even is positive, odd is the negative below it.
fn unzigzag(v: u32) -> i32 {
    if v & 1 == 0 {
        (v >> 1) as i32
    } else {
        -((v >> 1) as i32) - 1
    }
}

/// A quantized attribute's grid: where it starts and how big a step is.
#[derive(Debug, Clone)]
struct Quantization {
    min_values: Vec<f32>,
    delta: f32,
}

impl Quantization {
    fn decode(buffer: &mut Buffer<'_>, desc: &AttributeDescriptor) -> Result<Self, DracoError> {
        let mut min_values = Vec::with_capacity(desc.num_components as usize);
        for _ in 0..desc.num_components {
            min_values.push(buffer.f32()?);
        }
        let range = buffer.f32()?;
        let bits = buffer.u8()?;
        if !(1..=31).contains(&bits) {
            return Err(DracoError::Corrupt);
        }
        let max_quantized = ((1u32 << bits) - 1) as f32;
        Ok(Quantization {
            min_values,
            delta: range / max_quantized,
        })
    }

    fn dequantize(&self, value: i32, component: usize) -> f32 {
        value as f32 * self.delta + self.min_values.get(component).copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::attribute_type;
    use crate::loaders::draco::decode_mesh;

    const SPHERE: &[u8] = include_bytes!("testdata/sphere.drc");
    const BUNNY: &[u8] = include_bytes!("testdata/bunny.drc");

    /// FNV-1a over the raw bit patterns of a float array.
    ///
    /// The reference values come from running the same hash over `draco3d`'s
    /// `GetAttributeFloatForAllPoints` output. Comparing bits rather than
    /// magnitudes is the point: both implementations do the same arithmetic in
    /// the same order on `f32`, so agreement should be exact, and anything
    /// approximate would mean an operation was reordered or widened.
    fn fnv1a_f32(values: &[f32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in values {
            for byte in v.to_le_bytes() {
                h = (h ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// Every one of the sphere's attribute values is bit-identical to Google's.
    ///
    /// Three attributes exercise three different paths: positions are quantized
    /// integers under a parallelogram prediction and a wrap transform; normals
    /// are octahedral, predicted from the surrounding faces' areas on the integer
    /// position grid, with corrections that are *not* zig-zagged; texture
    /// coordinates are quantized on the seam-cut attribute table rather than the
    /// base one. A single wrong step in any of those still yields finite numbers.
    #[test]
    fn the_sphere_attributes_match_the_reference_decoder() {
        let m = decode_mesh(SPHERE).expect("a decoded mesh");
        assert_eq!(m.num_points(), 423);
        assert_eq!(m.attributes.len(), 3);

        let pos = m.attribute(attribute_type::POSITION).expect("positions");
        assert_eq!(pos.num_components(), 3);
        let pos = pos.as_f32();
        assert_eq!(pos.len(), 423 * 3);
        assert_eq!(fnv1a_f32(&pos), 4012496520929325221, "positions");

        let normal = m.attribute(attribute_type::NORMAL).expect("normals");
        assert_eq!(normal.num_components(), 3);
        let normal = normal.as_f32();
        assert_eq!(normal.len(), 423 * 3);
        assert_eq!(fnv1a_f32(&normal), 3815382446277921992, "normals");

        let uv = m.attribute(attribute_type::TEX_COORD).expect("texture coordinates");
        assert_eq!(uv.num_components(), 2);
        let uv = uv.as_f32();
        assert_eq!(uv.len(), 423 * 2);
        assert_eq!(fnv1a_f32(&uv), 14363382847896556601, "texture coordinates");
    }

    /// And the bunny's, which is 34834 points of each.
    #[test]
    fn the_bunny_attributes_match_the_reference_decoder() {
        let m = decode_mesh(BUNNY).expect("a decoded mesh");
        assert_eq!(m.num_points(), 34834);
        assert_eq!(m.attributes.len(), 2);

        let pos = m.attribute(attribute_type::POSITION).expect("positions").as_f32();
        assert_eq!(pos.len(), 34834 * 3);
        assert_eq!(fnv1a_f32(&pos), 17836075622030403104, "positions");

        let normal = m.attribute(attribute_type::NORMAL).expect("normals").as_f32();
        assert_eq!(normal.len(), 34834 * 3);
        assert_eq!(fnv1a_f32(&normal), 12948339483123031117, "normals");
    }

    /// Decoded normals are unit vectors.
    ///
    /// This is independent of the reference: the octahedral round trip has to
    /// land back on the sphere whatever the encoder did, so a wrong
    /// `to_unit_vector` shows up here even if the integers were right.
    #[test]
    fn normals_come_back_as_unit_vectors() {
        for bytes in [SPHERE, BUNNY] {
            let m = decode_mesh(bytes).expect("a decoded mesh");
            let n = m.attribute(attribute_type::NORMAL).expect("normals").as_f32();
            for (i, c) in n.chunks(3).enumerate() {
                let length = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                assert!(
                    (length - 1.0).abs() < 1e-4,
                    "normal {i} has length {length}: {c:?}"
                );
            }
        }
    }

    /// Positions land inside the bounding box the quantization declared.
    ///
    /// Dequantization is `value * delta + min`, so every component must sit
    /// within one step of the range the file stated. A prediction that was undone
    /// wrongly escapes the box, which is why this catches more than it looks like
    /// it should.
    #[test]
    fn positions_stay_inside_the_declared_range() {
        // The ranges are the ones draco3d reports for these files.
        for (bytes, lo, hi) in [
            (SPHERE, [-1.0f32, -1.0, -1.0], [1.0f32, 1.0, 1.0]),
            (
                BUNNY,
                [-0.0947, 0.0329, -0.0619],
                [0.0611, 0.1874, 0.0589],
            ),
        ] {
            let m = decode_mesh(bytes).expect("a decoded mesh");
            let pos = m.attribute(attribute_type::POSITION).expect("positions").as_f32();
            for (i, p) in pos.chunks(3).enumerate() {
                for c in 0..3 {
                    assert!(
                        p[c] >= lo[c] - 1e-3 && p[c] <= hi[c] + 1e-3,
                        "position {i} component {c} is {} outside [{}, {}]",
                        p[c],
                        lo[c],
                        hi[c]
                    );
                }
            }
        }
    }

    /// Indices address the points, and the attributes cover them.
    #[test]
    fn the_index_buffer_and_the_attributes_agree() {
        for bytes in [SPHERE, BUNNY] {
            let m = decode_mesh(bytes).expect("a decoded mesh");
            let indices = m.indices();
            let points = m.num_points();
            assert!(indices.iter().all(|i| *i < points));
            for a in &m.attributes {
                assert_eq!(
                    a.as_f32().len(),
                    points as usize * a.num_components(),
                    "attribute {} does not cover every point",
                    a.descriptor.att_type
                );
            }
        }
    }

    /// Corruption anywhere is an error, not a panic.
    ///
    /// The attribute path indexes several arrays by values read from the file and
    /// runs prediction loops whose termination depends on the connectivity, so
    /// this is where a malformed stream would otherwise hang or panic in a
    /// browser.
    #[test]
    fn corrupt_attribute_data_does_not_panic() {
        // Every truncation, since a short read is the common corruption.
        for cut in 0..SPHERE.len() {
            let _ = decode_mesh(&SPHERE[..cut]);
        }
        // Then bit flips: every byte of the header and connectivity, where a
        // wrong count decides how much gets allocated and how long a loop runs,
        // and a stride through the attribute payload, which is most of the file
        // and where flips are interchangeable.
        let mut broken = SPHERE.to_vec();
        let dense = 400.min(broken.len());
        let positions = (0..dense).chain((dense..broken.len()).step_by(7));
        for i in positions {
            let old = broken[i];
            for flip in [0x01u8, 0x40, 0xff] {
                broken[i] = old ^ flip;
                let _ = decode_mesh(&broken);
            }
            broken[i] = old;
        }
    }

    /// A third file, from a different encoder run and a different topology.
    ///
    /// The sphere is closed and the bunny has five holes; this is an open disc
    /// with one boundary loop, so its Euler characteristic is 1. It was encoded
    /// separately with all three attributes quantized differently, which is what
    /// makes it a check on the decoder rather than on one encoder's habits.
    #[test]
    fn a_third_file_matches_the_reference_decoder() {
        const GRID: &[u8] = include_bytes!("testdata/grid.drc");
        let m = decode_mesh(GRID).expect("a decoded mesh");
        assert_eq!(m.num_points(), 625);
        assert_eq!(m.connectivity.replay.table.num_faces(), 1152);

        let indices = m.indices();
        assert_eq!(indices.len(), 1152 * 3);
        assert_eq!(
            &indices[..30],
            &[
                0, 1, 2, 2, 1, 3, 4, 3, 6, 3, 1, 6, 6, 1, 7, 7, 1, 8, 1, 0, 8, 8, 0, 9, 10, 9, 12,
                9, 0, 12
            ]
        );
        assert_eq!(fnv1a_f32(&[]), 0xcbf2_9ce4_8422_2325, "the hash's empty case");
        let pos = m.attribute(attribute_type::POSITION).expect("positions").as_f32();
        assert_eq!(fnv1a_f32(&pos), 8366847262056489837, "positions");
        let normal = m.attribute(attribute_type::NORMAL).expect("normals").as_f32();
        assert_eq!(fnv1a_f32(&normal), 4395065954670703389, "normals");
        let uv = m.attribute(attribute_type::TEX_COORD).expect("uvs").as_f32();
        assert_eq!(fnv1a_f32(&uv), 8550600021161492673, "texture coordinates");

        // An open disc: one boundary loop, so V - E + F is 1 rather than 2.
        let table = &m.connectivity.replay.table;
        let mut nbrs: std::collections::HashMap<u32, std::collections::HashSet<u32>> =
            std::collections::HashMap::new();
        for t in table.triangles() {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                nbrs.entry(a).or_default().insert(b);
                nbrs.entry(b).or_default().insert(a);
            }
        }
        let edges = nbrs.values().map(|s| s.len()).sum::<usize>() / 2;
        assert_eq!(edges, 1776);
        assert_eq!(nbrs.len() as i64 - edges as i64 + 1152, 1, "a disc");
    }

    /// Valence-coded connectivity decodes, and matches the reference decoder.
    ///
    /// This is the traversal Draco's own encoder selects by default: `-cl 7` maps
    /// to a decode speed of 3, and anything at 4 or below codes the symbols by
    /// valence instead of writing them out. So a file straight out of
    /// `draco_encoder` lands on this path.
    ///
    /// The symbols here cannot be read ahead of the replay at all. Each one comes
    /// from a stream chosen by the valence of the vertex the traversal is standing
    /// on *in the partly-built mesh*, so the decoder has to interleave building
    /// the corner table with reading symbols out of six separate streams, each
    /// consumed from its back. Every index and every attribute value below is
    /// compared against `draco3d` on the same file, because a valence tracked one
    /// step out of step still yields symbols and still builds a mesh.
    #[test]
    fn a_valence_coded_file_matches_the_reference_decoder() {
        const VALENCE: &[u8] = include_bytes!("testdata/valence.drc");
        // The traversal byte really is the valence one, so this fixture cannot
        // silently become a standard-traversal file.
        assert_eq!(VALENCE[crate::loaders::draco::HEADER_BYTES], 2);

        let m = decode_mesh(VALENCE).expect("a decoded mesh");
        assert_eq!(m.num_points(), 961);
        assert_eq!(m.connectivity.replay.table.num_faces(), 1800);

        let indices = m.indices();
        assert_eq!(indices.len(), 1800 * 3);
        assert_eq!(
            &indices[..30],
            &[
                0, 1, 2, 2, 1, 3, 4, 3, 5, 3, 1, 5, 5, 1, 6, 6, 1, 7, 1, 0, 7, 7, 0, 8, 9, 8, 10,
                8, 0, 10
            ]
        );
        assert_eq!(fnv1a_u32(&indices), 17713357916668641341, "all 5400 indices");

        let pos = m.attribute(attribute_type::POSITION).expect("positions").as_f32();
        assert_eq!(pos.len(), 961 * 3);
        assert_eq!(fnv1a_f32(&pos), 14293316393217882539, "positions");
        let normal = m.attribute(attribute_type::NORMAL).expect("normals").as_f32();
        assert_eq!(fnv1a_f32(&normal), 6763978815183453622, "normals");
    }

    /// A file from Draco's own encoder at its default setting decodes exactly.
    ///
    /// This is the case that matters most for reading files in the wild, and it
    /// is the one that needs the most machinery: `draco_encoder` defaults to
    /// `-cl 7`, which selects *both* the valence traversal for connectivity and
    /// the portable predictor for texture coordinates. Neither can be skipped
    /// and neither announces itself if decoded wrongly — the valence symbols
    /// come out of six streams chosen by a running valence, and the UV predictor
    /// reconstructs each coordinate from the triangle's shape in position space
    /// with one stored bit choosing the side.
    ///
    /// Every index and every value is compared against `draco3d` on the same
    /// file.
    #[test]
    fn a_default_encoded_file_matches_the_reference_decoder() {
        const DEFAULT: &[u8] = include_bytes!("testdata/default_cl7.drc");
        assert_eq!(
            DEFAULT[crate::loaders::draco::HEADER_BYTES],
            2,
            "the default setting really does use the valence traversal"
        );
        let m = decode_mesh(DEFAULT).expect("a decoded mesh");
        assert_eq!(m.num_points(), 625);
        assert_eq!(m.connectivity.replay.table.num_faces(), 1152);
        assert_eq!(m.attributes.len(), 3);

        let indices = m.indices();
        assert_eq!(
            &indices[..30],
            &[
                0, 1, 2, 2, 1, 3, 4, 3, 5, 3, 1, 5, 5, 1, 6, 6, 1, 7, 1, 0, 7, 7, 0, 8, 9, 8, 10,
                8, 0, 10
            ]
        );
        assert_eq!(fnv1a_u32(&indices), 16724102879085218717, "all 3456 indices");

        let pos = m.attribute(attribute_type::POSITION).expect("positions").as_f32();
        assert_eq!(fnv1a_f32(&pos), 2246799073516984197, "positions");
        let normal = m.attribute(attribute_type::NORMAL).expect("normals").as_f32();
        assert_eq!(fnv1a_f32(&normal), 6119629274032286637, "normals");
        let uv = m.attribute(attribute_type::TEX_COORD).expect("uvs").as_f32();
        assert_eq!(uv.len(), 625 * 2);
        assert_eq!(fnv1a_f32(&uv), 4556528882262129973, "texture coordinates");
        // The source grid's coordinates are a unit square, so anything outside it
        // is a prediction that went wrong in a way the hash would also catch but
        // this names.
        for (i, c) in uv.chunks(2).enumerate() {
            assert!(
                (-0.01..=1.01).contains(&c[0]) && (-0.01..=1.01).contains(&c[1]),
                "uv {i} is {c:?}, outside the unit square"
            );
        }
    }

    /// FNV-1a over an index buffer, matching the hash the oracle prints.
    fn fnv1a_u32(values: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in values {
            for byte in v.to_le_bytes() {
                h = (h ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }
}
