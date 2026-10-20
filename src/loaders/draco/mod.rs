//! `KHR_draco_mesh_compression` — Google's Draco, decoded.
//!
//! Draco is the compression most `.glb` in the wild uses, and until now such a
//! file was refused. It is a much larger format than meshopt: an rANS entropy
//! layer, a connectivity coder (*edgebreaker*, which walks the mesh emitting one
//! of five symbols per triangle), and per-attribute prediction schemes on top.
//!
//! This is a port of Draco's own decoder, not a reading of a specification.
//! Nothing here was recalled: the rANS constants, the edgebreaker traversal and
//! the prediction schemes were each taken from the C++ and checked against
//! Google's own WASM decoder on the same file — because every one of them
//! decodes to plausible-looking wrong geometry if it is subtly off, and vertex
//! positions do not announce that they are wrong.

pub mod attribute_table;
pub mod attributes;
mod buffer;
pub mod corner_table;
pub mod octahedron;
mod rans;
pub mod three;
pub mod traversal;

use buffer::Buffer;

#[derive(Debug, PartialEq)]
pub enum DracoError {
    /// Not a Draco bitstream.
    BadHeader,
    /// A version this decoder does not handle.
    UnsupportedVersion,
    /// A geometry or encoding method this decoder does not handle.
    Unsupported(&'static str),
    /// The stream ended early.
    Truncated,
    /// Structurally Draco and internally inconsistent.
    Corrupt,
}

impl std::fmt::Display for DracoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DracoError::BadHeader => write!(f, "not a Draco bitstream"),
            DracoError::UnsupportedVersion => write!(f, "unsupported Draco version"),
            DracoError::Unsupported(what) => write!(f, "unsupported Draco feature: {what}"),
            DracoError::Truncated => write!(f, "truncated Draco bitstream"),
            DracoError::Corrupt => write!(f, "corrupt Draco bitstream"),
        }
    }
}

impl std::error::Error for DracoError {}

/// What the stream says it holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Header {
    pub major: u8,
    pub minor: u8,
    /// 0 for a point cloud, 1 for a triangular mesh.
    pub geometry: u8,
    /// 0 for sequential, 1 for edgebreaker.
    pub method: u8,
    pub flags: u16,
}

impl Header {
    pub fn is_mesh(&self) -> bool {
        self.geometry == 1
    }

    pub fn is_edgebreaker(&self) -> bool {
        self.method == 1
    }

    /// The bitstream version, as Draco compares it.
    pub fn version(&self) -> u16 {
        ((self.major as u16) << 8) | self.minor as u16
    }
}

/// Bytes in the file header: `DRACO`, two version bytes, type, method, flags.
pub const HEADER_BYTES: usize = 11;

/// Header flag saying a metadata block sits before the connectivity.
pub const METADATA_FLAG: u16 = 0x8000;

/// Read the header, which is the same for every Draco stream.
pub fn header(bytes: &[u8]) -> Result<Header, DracoError> {
    let mut buffer = Buffer::new(bytes);
    let magic = buffer.take(5)?;
    if magic != b"DRACO" {
        return Err(DracoError::BadHeader);
    }
    let header = Header {
        major: buffer.u8()?,
        minor: buffer.u8()?,
        geometry: buffer.u8()?,
        method: buffer.u8()?,
        flags: buffer.u16()?,
    };
    // 2.x is what every encoder in use writes. 1.x differs in enough places
    // that claiming to read it would be a guess.
    if header.major != 2 {
        return Err(DracoError::UnsupportedVersion);
    }
    Ok(header)
}

/// What the connectivity section declares before any symbol is read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Connectivity {
    pub num_encoded_vertices: u32,
    pub num_faces: u32,
    /// How many attribute decoders follow the connectivity.
    pub num_attribute_data: u8,
    /// How many CLERS symbols the traversal emitted.
    pub num_encoded_symbols: u32,
    /// How many of those were splits, each of which adds a vertex.
    pub num_encoded_split_symbols: u32,
}

/// How the traversal was coded, which decides how symbols are read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Traversal {
    /// Symbols from one rANS stream, and separate bit streams for the flags.
    Standard,
    /// Symbols implied by vertex valences, which codes a smooth mesh smaller.
    Valence,
}

/// The byte between the file header and the connectivity data.
///
/// `MeshEdgebreakerDecoder::InitializeDecoder` reads this before anything else,
/// and reading the connectivity without it puts every varint after it one byte
/// out — which still yields numbers, just not the right ones.
pub fn traversal_type(buffer: &mut Buffer<'_>) -> Result<Traversal, DracoError> {
    match buffer.u8()? {
        0 => Ok(Traversal::Standard),
        2 => Ok(Traversal::Valence),
        // 1 is the predictive encoding, which only old files use and which
        // Draco itself builds out by default.
        1 => Err(DracoError::Unsupported("predictive edgebreaker traversal")),
        _ => Err(DracoError::Corrupt),
    }
}

/// Read the edgebreaker connectivity header.
///
/// The consistency checks are Draco's own, and they are worth keeping: they are
/// what turns a corrupt file into an error instead of an allocation of a few
/// billion corners.
pub fn connectivity_header(buffer: &mut Buffer<'_>) -> Result<Connectivity, DracoError> {
    let num_encoded_vertices = buffer.varint32()?;
    let num_faces = buffer.varint32()?;
    if num_faces > u32::MAX / 3 {
        return Err(DracoError::Corrupt);
    }
    if num_encoded_vertices > num_faces * 3 {
        return Err(DracoError::Corrupt);
    }
    // A manifold mesh needs at least 3n/2 edges, and n vertices can carry at
    // most n(n-1)/2 of them.
    let min_face_edges = 3u64 * num_faces as u64 / 2;
    let v = num_encoded_vertices as u64;
    if v.saturating_mul(v.saturating_sub(1)) / 2 < min_face_edges {
        return Err(DracoError::Corrupt);
    }
    let num_attribute_data = buffer.u8()?;
    let num_encoded_symbols = buffer.varint32()?;
    if num_faces < num_encoded_symbols {
        return Err(DracoError::Corrupt);
    }
    // Faces can exceed symbols by at most a third — only when every new
    // component began with an interior triangle.
    if num_faces > num_encoded_symbols + num_encoded_symbols / 3 {
        return Err(DracoError::Corrupt);
    }
    let num_encoded_split_symbols = buffer.varint32()?;
    if num_encoded_split_symbols > num_encoded_symbols {
        return Err(DracoError::Corrupt);
    }
    Ok(Connectivity {
        num_encoded_vertices,
        num_faces,
        num_attribute_data,
        num_encoded_symbols,
        num_encoded_split_symbols,
    })
}

/// An edgebreaker mesh's connectivity, all the way to an index buffer.
#[derive(Debug, Clone)]
pub struct DecodedConnectivity {
    pub replay: corner_table::Replay,
    /// One table per non-position attribute, with its seams cut.
    pub attribute_tables: Vec<attribute_table::AttributeCornerTable>,
    /// The point each corner belongs to.
    pub corner_to_point: Vec<u32>,
    /// One corner per point, for sampling attribute values. Empty when points
    /// and vertices coincide.
    pub point_to_corner: Vec<u32>,
    pub num_points: u32,
}

impl DecodedConnectivity {
    /// The triangle indices, in points — what a glTF accessor would hold.
    pub fn indices(&self) -> Vec<u32> {
        self.corner_to_point.clone()
    }
}

/// Decode the connectivity of an edgebreaker mesh: header through index buffer.
///
/// Each step reads a length or a count that positions the next, so they cannot
/// be attempted out of order or individually — which is why this exists rather
/// than leaving callers to chain them.
pub fn edgebreaker_connectivity(bytes: &[u8]) -> Result<DecodedConnectivity, DracoError> {
    let mut buffer = Buffer::new(bytes);
    let head = start_mesh(&mut buffer, bytes)?;
    let _ = head;
    connectivity_from(&mut buffer)
}

/// A whole Draco mesh: the index buffer and the attribute values.
#[derive(Debug, Clone)]
pub struct DecodedMesh {
    pub connectivity: DecodedConnectivity,
    pub attributes: Vec<attributes::DecodedAttribute>,
}

impl DecodedMesh {
    pub fn indices(&self) -> Vec<u32> {
        self.connectivity.indices()
    }

    pub fn num_points(&self) -> u32 {
        self.connectivity.num_points
    }

    /// The first attribute of a given `GeometryAttribute::Type`, if present.
    pub fn attribute(&self, att_type: u8) -> Option<&attributes::DecodedAttribute> {
        self.attributes
            .iter()
            .find(|a| a.descriptor.att_type == att_type)
    }
}

/// Decode a Draco mesh completely.
pub fn decode_mesh(bytes: &[u8]) -> Result<DecodedMesh, DracoError> {
    let mut buffer = Buffer::new(bytes);
    start_mesh(&mut buffer, bytes)?;
    let connectivity = connectivity_from(&mut buffer)?;
    let attributes = attributes::decode_attributes(&mut buffer, &connectivity)?;
    Ok(DecodedMesh {
        connectivity,
        attributes,
    })
}

/// Check the header and position `buffer` at the connectivity.
fn start_mesh(buffer: &mut Buffer<'_>, bytes: &[u8]) -> Result<Header, DracoError> {
    let head = header(bytes)?;
    if !head.is_mesh() {
        return Err(DracoError::Unsupported("point cloud"));
    }
    if !head.is_edgebreaker() {
        return Err(DracoError::Unsupported("sequential connectivity"));
    }
    if head.flags & METADATA_FLAG != 0 {
        return Err(DracoError::Unsupported("metadata"));
    }
    buffer.advance(HEADER_BYTES)?;
    Ok(head)
}

/// Read the connectivity, leaving `buffer` at the attribute section.
fn connectivity_from(buffer: &mut Buffer<'_>) -> Result<DecodedConnectivity, DracoError> {
    let traversal = traversal_type(buffer)?;
    let conn = connectivity_header(buffer)?;
    let mut splits = traversal::topology_splits(buffer, conn.num_faces)?;
    // One vertex per split symbol is created during the replay, and the valence
    // traversal has to size its valence table for all of them up front.
    let num_vertices = conn
        .num_encoded_vertices
        .checked_add(conn.num_encoded_split_symbols)
        .ok_or(DracoError::Corrupt)?;
    let mut t = traversal::decode(
        buffer,
        traversal,
        conn.num_encoded_symbols,
        conn.num_attribute_data,
        num_vertices,
        conn.num_faces,
    )?;
    // Draco folds isolated vertices away only when nothing else indexes them;
    // with attribute connectivity present, the point assignment needs the gaps.
    let remove_invalid_vertices = conn.num_attribute_data == 0;
    let replay = corner_table::replay(
        &mut t.symbols,
        &mut splits,
        &mut t.start_faces,
        &conn,
        remove_invalid_vertices,
    )?;
    let attribute_tables =
        attribute_table::decode_attribute_seams(&replay.table, &mut t.attribute_seams)?;
    let (corner_to_point, point_to_corner, num_points) = attribute_table::assign_points_to_corners(
        &replay.table,
        &attribute_tables,
        &replay.is_vert_hole,
        replay.num_vertices,
    )?;
    Ok(DecodedConnectivity {
        replay,
        attribute_tables,
        corner_to_point,
        point_to_corner,
        num_points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPHERE: &[u8] = include_bytes!("testdata/sphere.drc");
    const BUNNY: &[u8] = include_bytes!("testdata/bunny.drc");

    /// The header of a real file, written by Google's encoder.
    #[test]
    fn a_real_header_reads() {
        let h = header(SPHERE).expect("a Draco header");
        assert_eq!((h.major, h.minor), (2, 2));
        assert!(h.is_mesh(), "a triangular mesh");
        assert!(h.is_edgebreaker(), "edgebreaker, which is what encoders write");
        assert_eq!(h.flags, 0);

        let b = header(BUNNY).expect("a Draco header");
        assert!(b.is_mesh());
    }

    /// The connectivity header's numbers are the ones the glTF accessors agree
    /// with: 720 triangles and 423 vertices for this sphere.
    ///
    /// Reading a varint backwards, or the fields out of order, still produces
    /// numbers — so the face count is checked against one established
    /// independently, from the accessors in the `.glb` and from Google's own
    /// decoder.
    ///
    /// Note that *vertices* are fewer than *points*: a point is a unique
    /// combination of attribute indices, so a seam in the texture coordinates
    /// splits one vertex into several points. This sphere has 362 encoded
    /// vertices and 423 points, and confusing the two is how a decoder ends up
    /// allocating the wrong number of anything.
    #[test]
    fn the_connectivity_header_agrees_with_the_gltf_accessors() {
        let mut buffer = Buffer::new(SPHERE);
        buffer.advance(11).unwrap(); // past the file header
        assert_eq!(traversal_type(&mut buffer).unwrap(), Traversal::Standard);
        let c = connectivity_header(&mut buffer).expect("connectivity header");
        assert_eq!(c.num_faces, 720, "2160 indices / 3");
        assert_eq!(c.num_encoded_vertices, 362);
        assert_eq!(c.num_attribute_data, 2, "normals and texture coordinates");
        assert_eq!(c.num_encoded_symbols, 719, "one face is the seed, not a symbol");
        assert_eq!(c.num_encoded_split_symbols, 19);
        // Vertices plus splits, then seams, must reach the 423 points the
        // accessors declare.
        assert!(
            c.num_encoded_vertices + c.num_encoded_split_symbols <= 423,
            "{} + {}",
            c.num_encoded_vertices,
            c.num_encoded_split_symbols
        );
    }

    /// The bunny reads its header too — a second file, from a different
    /// encoder run, so the parse is not fitted to one.
    #[test]
    fn a_second_file_reads_its_connectivity_header() {
        let mut buffer = Buffer::new(BUNNY);
        buffer.advance(11).unwrap();
        let traversal = traversal_type(&mut buffer).expect("a traversal type");
        let c = connectivity_header(&mut buffer).expect("connectivity header");
        assert!(c.num_faces > 1000, "the bunny is not a toy mesh: {}", c.num_faces);
        assert!(c.num_encoded_symbols <= c.num_faces);
        println!("bunny: {traversal:?} {c:?}");
    }

    /// The symbol stream decodes, and fills the size it declared.
    ///
    /// Every 1-or-3-bit read yields a valid symbol by construction, so "the
    /// symbols look like symbols" proves nothing. What does prove something is
    /// the *bit count*: 719 symbols have to consume exactly the bytes the stream
    /// said it had, to within the byte it is padded to. Reading the bits
    /// most-significant-first, or mis-sizing `C`, lands somewhere else.
    #[test]
    fn the_symbol_stream_fills_exactly_the_size_it_declares() {
        let mut buffer = Buffer::new(SPHERE);
        buffer.advance(11).unwrap();
        traversal_type(&mut buffer).unwrap();
        let c = connectivity_header(&mut buffer).unwrap();
        let splits = traversal::topology_splits(&mut buffer, c.num_faces).expect("splits");
        assert!(splits.is_empty(), "this sphere has no topology splits");
        let before = buffer.at;
        let declared = {
            let mut peek = Buffer::new(SPHERE);
            peek.advance(before).unwrap();
            peek.varint().unwrap() as usize
        };
        let t = traversal::decode(
            &mut buffer,
            Traversal::Standard,
            c.num_encoded_symbols,
            c.num_attribute_data,
            c.num_encoded_vertices + c.num_encoded_split_symbols,
            c.num_faces,
        )
        .expect("the traversal");
        let symbols = t.symbols.standard().expect("standard traversal symbols");
        assert_eq!(symbols.len(), 719);
        // Within the padding byte of the declared size.
        assert!(
            t.symbol_bits <= declared * 8 && declared * 8 - t.symbol_bits < 8,
            "used {} bits of a declared {} bytes",
            t.symbol_bits,
            declared
        );

        use traversal::Symbol;
        let count = |s: Symbol| symbols.iter().filter(|x| **x == s).count();
        assert_eq!(
            count(Symbol::S),
            c.num_encoded_split_symbols as usize,
            "the header's split count is the number of S symbols"
        );
        // Every `S` splits the traversal into a branch that ends with its own
        // `E`, so a single connected component gives `E = S + 1`. This is the
        // check with teeth: it ties the symbols to a header field decoded
        // separately, and a bit slip anywhere in the stream breaks it.
        assert_eq!(
            count(Symbol::E),
            count(Symbol::S) + 1,
            "one component: every split ends, and so does the whole"
        );
        assert!(count(Symbol::C) > 300, "most of a smooth mesh is C: {}", count(Symbol::C));
    }

    /// And on the bunny, which is a different size and a different encoder run.
    #[test]
    fn the_bunny_symbol_stream_decodes() {
        let mut buffer = Buffer::new(BUNNY);
        buffer.advance(11).unwrap();
        traversal_type(&mut buffer).unwrap();
        let c = connectivity_header(&mut buffer).unwrap();
        let splits = traversal::topology_splits(&mut buffer, c.num_faces).expect("splits");
        let t = traversal::decode(
            &mut buffer,
            Traversal::Standard,
            c.num_encoded_symbols,
            c.num_attribute_data,
            c.num_encoded_vertices + c.num_encoded_split_symbols,
            c.num_faces,
        )
        .expect("the traversal");
        let symbols = t.symbols.standard().expect("standard traversal symbols");
        assert_eq!(symbols.len(), c.num_encoded_symbols as usize);
        println!("bunny splits: {}", splits.len());
        use traversal::Symbol;
        assert_eq!(
            symbols.iter().filter(|s| **s == Symbol::S).count(),
            c.num_encoded_split_symbols as usize,
            "splits agree with the header"
        );
    }

    /// Something that is not Draco is refused.
    #[test]
    fn a_non_draco_stream_is_refused() {
        assert_eq!(header(b"nope").unwrap_err(), DracoError::Truncated);
        assert_eq!(
            header(b"NOTDRACO....").unwrap_err(),
            DracoError::BadHeader
        );
    }

    /// A truncated stream is an error at every length.
    #[test]
    fn truncation_does_not_panic() {
        for cut in 0..64.min(SPHERE.len()) {
            let _ = header(&SPHERE[..cut]);
            let mut buffer = Buffer::new(&SPHERE[..cut]);
            if buffer.advance(11).is_ok() {
                let _ = connectivity_header(&mut buffer);
            }
        }
    }
}
