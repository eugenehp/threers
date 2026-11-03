//! The edgebreaker symbol stream.
//!
//! Edgebreaker walks the mesh one triangle at a time, emitting a symbol that
//! says how the new triangle attaches to the one before it:
//!
//! | symbol | code | meaning |
//! |---|---|---|
//! | `C` | `0` | a new vertex |
//! | `S` | `1 0 0` | the traversal splits |
//! | `L` | `1 1 0` | closes on the left |
//! | `R` | `1 0 1` | closes on the right |
//! | `E` | `1 1 1` | closes both — the component ends |
//!
//! `C` is one bit because it is most of them; the rest are three. The standard
//! traversal writes those bits plainly, and only the *flags* beside them — which
//! face a component starts on, where attribute seams fall — go through rANS.

use super::buffer::Buffer;
use super::corner_table::CornerTable;
use super::rans::BitDecoder;
use super::DracoError;

/// One of the five edgebreaker symbols, in Draco's own numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symbol {
    C = 0,
    S = 1,
    L = 3,
    R = 5,
    E = 7,
}

impl Symbol {
    fn from_code(code: u32) -> Option<Symbol> {
        Some(match code {
            0 => Symbol::C,
            1 => Symbol::S,
            3 => Symbol::L,
            5 => Symbol::R,
            7 => Symbol::E,
            _ => return None,
        })
    }
}

/// Where the replay gets its symbols from.
///
/// The standard traversal writes the symbols out and they can all be read before
/// the replay starts. The valence traversal does not: it picks an entropy context
/// from the valences of the vertices decoded so far, so the next symbol cannot be
/// known until the replay has built enough of the mesh to know those valences.
/// The replay therefore pulls one symbol at a time and reports back what it did,
/// which is the only shape that serves both.
pub trait SymbolSource {
    fn next_symbol(&mut self, table: &CornerTable) -> Result<Symbol, DracoError>;

    /// The replay has finished a face and this is the new active corner.
    fn new_active_corner_reached(&mut self, _table: &CornerTable, _corner: u32) {}

    /// An `S` symbol merged two vertices into one.
    fn merge_vertices(&mut self, _dest: u32, _source: u32) {}
}

/// Pre-decoded symbols, handed out in order.
#[derive(Debug, Clone, Default)]
pub struct StandardSymbols {
    pub symbols: Vec<Symbol>,
    at: usize,
}

impl SymbolSource for StandardSymbols {
    fn next_symbol(&mut self, _table: &CornerTable) -> Result<Symbol, DracoError> {
        let symbol = *self.symbols.get(self.at).ok_or(DracoError::Corrupt)?;
        self.at += 1;
        Ok(symbol)
    }
}

/// Draco's `edge_breaker_symbol_to_topology_id`.
const SYMBOL_ORDER: [Symbol; 5] = [Symbol::C, Symbol::S, Symbol::L, Symbol::R, Symbol::E];

/// Symbols chosen by the valence of the vertex the traversal is standing on.
///
/// A smooth mesh is mostly vertices of valence six, and which symbol comes next
/// is strongly correlated with that valence — so Draco keeps one symbol stream
/// per valence from 2 to 7 and entropy-codes each separately. Each stream is read
/// *from the back*, because the encoder filled it walking the other way.
///
/// The valences are of the partly-built mesh, which is why this has to be driven
/// by the replay rather than read ahead of it.
#[derive(Debug, Clone, Default)]
pub struct ValenceSymbols {
    vertex_valences: Vec<i32>,
    last_symbol: Option<Symbol>,
    /// Which stream the next symbol comes from, or -1 before any face exists.
    active_context: i32,
    min_valence: i32,
    max_valence: i32,
    context_symbols: Vec<Vec<u32>>,
    context_counters: Vec<i32>,
}

impl ValenceSymbols {
    /// Read the per-valence symbol streams.
    ///
    /// These sit after the start-face and attribute-seam decoders, not before
    /// them as the standard traversal's single stream does.
    pub fn start(
        buffer: &mut Buffer<'_>,
        num_vertices: u32,
        num_faces: u32,
    ) -> Result<Self, DracoError> {
        // Bitstream 2.2 fixed the range; older files wrote a mode byte for it.
        let (min_valence, max_valence) = (2i32, 7i32);
        let num_contexts = (max_valence - min_valence + 1) as usize;
        let mut context_symbols = Vec::with_capacity(num_contexts);
        let mut context_counters = vec![0i32; num_contexts];
        for counter in context_counters.iter_mut() {
            let num_symbols = buffer.varint32()?;
            if num_symbols > num_faces {
                return Err(DracoError::Corrupt);
            }
            if num_symbols > 0 {
                context_symbols.push(super::rans::decode_symbols(
                    num_symbols as usize,
                    1,
                    buffer,
                )?);
                *counter = num_symbols as i32;
            } else {
                // Not merely empty: Draco reads nothing at all for an empty
                // context, so consuming a scheme byte here would desynchronise
                // every stream after it.
                context_symbols.push(Vec::new());
            }
        }
        Ok(ValenceSymbols {
            vertex_valences: vec![0i32; num_vertices as usize],
            last_symbol: None,
            active_context: -1,
            min_valence,
            max_valence,
            context_symbols,
            context_counters,
        })
    }
}

impl SymbolSource for ValenceSymbols {
    fn next_symbol(&mut self, _table: &CornerTable) -> Result<Symbol, DracoError> {
        let symbol = if self.active_context != -1 {
            let context = self.active_context as usize;
            let counter = self
                .context_counters
                .get_mut(context)
                .ok_or(DracoError::Corrupt)?;
            *counter -= 1;
            if *counter < 0 {
                return Err(DracoError::Corrupt);
            }
            let at = *counter as usize;
            let symbol_id = *self
                .context_symbols
                .get(context)
                .and_then(|c| c.get(at))
                .ok_or(DracoError::Corrupt)?;
            if symbol_id > 4 {
                return Err(DracoError::Corrupt);
            }
            SYMBOL_ORDER[symbol_id as usize]
        } else {
            // With no context there is no vertex to have a valence, which only
            // happens at the very start of a component — and a component always
            // opens with `E`.
            Symbol::E
        };
        self.last_symbol = Some(symbol);
        Ok(symbol)
    }

    fn new_active_corner_reached(&mut self, table: &CornerTable, corner: u32) {
        let next = table.next(corner);
        let prev = table.previous(corner);
        // How much each of the face's three vertices gained depends on which
        // symbol built it — an `E` face contributes two edges at every corner, a
        // `C` face only closes the two either side of its tip.
        let deltas: [(u32, i32); 3] = match self.last_symbol {
            Some(Symbol::C) | Some(Symbol::S) => [(corner, 0), (next, 1), (prev, 1)],
            Some(Symbol::R) => [(corner, 1), (next, 1), (prev, 2)],
            Some(Symbol::L) => [(corner, 1), (next, 2), (prev, 1)],
            Some(Symbol::E) => [(corner, 2), (next, 2), (prev, 2)],
            None => [(corner, 0), (next, 0), (prev, 0)],
        };
        for (at, by) in deltas {
            if by == 0 {
                continue;
            }
            let vert = table.vertex(at);
            if let Some(slot) = self.vertex_valences.get_mut(vert as usize) {
                *slot += by;
            }
        }
        // The next symbol is read from the stream belonging to the valence of the
        // vertex the traversal now stands on, clamped into the coded range.
        let vert = table.vertex(next);
        let valence = self
            .vertex_valences
            .get(vert as usize)
            .copied()
            .unwrap_or(0)
            .clamp(self.min_valence, self.max_valence);
        self.active_context = valence - self.min_valence;
    }

    fn merge_vertices(&mut self, dest: u32, source: u32) {
        let gained = self
            .vertex_valences
            .get(source as usize)
            .copied()
            .unwrap_or(0);
        if let Some(slot) = self.vertex_valences.get_mut(dest as usize) {
            *slot += gained;
        }
    }
}

/// Either kind of symbol source.
#[derive(Debug, Clone)]
pub enum Symbols {
    Standard(StandardSymbols),
    Valence(ValenceSymbols),
}

impl Symbols {
    /// The pre-decoded symbols, for the standard traversal only.
    pub fn standard(&self) -> Option<&[Symbol]> {
        match self {
            Symbols::Standard(s) => Some(&s.symbols),
            Symbols::Valence(_) => None,
        }
    }
}

impl SymbolSource for Symbols {
    fn next_symbol(&mut self, table: &CornerTable) -> Result<Symbol, DracoError> {
        match self {
            Symbols::Standard(s) => s.next_symbol(table),
            Symbols::Valence(s) => s.next_symbol(table),
        }
    }
    fn new_active_corner_reached(&mut self, table: &CornerTable, corner: u32) {
        match self {
            Symbols::Standard(s) => s.new_active_corner_reached(table, corner),
            Symbols::Valence(s) => s.new_active_corner_reached(table, corner),
        }
    }
    fn merge_vertices(&mut self, dest: u32, source: u32) {
        match self {
            Symbols::Standard(s) => s.merge_vertices(dest, source),
            Symbols::Valence(s) => s.merge_vertices(dest, source),
        }
    }
}

/// The symbols of a traversal, and the rANS-coded flags that go with them.
pub struct TraversalData<'a> {
    pub symbols: Symbols,
    /// Whether each new component's first face is oriented one way or the other.
    pub start_faces: BitDecoder<'a>,
    /// One decoder per attribute data, saying which edges are seams.
    pub attribute_seams: Vec<BitDecoder<'a>>,
    /// How many bits the symbol stream used, for checking against what it
    /// claimed. Zero for the valence traversal, which has no single stream.
    pub symbol_bits: usize,
}

/// A split in the traversal: where it resumed, and where it had paused.
///
/// Edgebreaker's `S` symbol splits the traversal in two, and the second half is
/// picked up later. These say which symbol resumed which, so the corner table
/// can be stitched back together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologySplit {
    pub source_symbol_id: u32,
    pub split_symbol_id: u32,
    /// Which edge of the source face the split came from.
    pub source_edge: u32,
}

/// Read the topology split events, which sit between the connectivity header
/// and the traversal.
///
/// Missing this entirely is easy — for a mesh with none, it is a single zero
/// byte — and then every offset afterwards is one byte out. That decodes to a
/// plausible symbol stream of the wrong length, which is how it announces
/// itself.
///
/// Hole events are not read: from bitstream 2.1 the encoder stopped writing
/// them, and a 2.2 file that had them would be malformed rather than old.
pub fn topology_splits(buffer: &mut Buffer<'_>, num_faces: u32) -> Result<Vec<TopologySplit>, DracoError> {
    let count = buffer.varint32()?;
    if count == 0 {
        return Ok(Vec::new());
    }
    if count > num_faces {
        return Err(DracoError::Corrupt);
    }
    // Symbol ids are delta coded against the last source, and the split is a
    // *backwards* delta from its own source.
    let mut splits = Vec::with_capacity(count as usize);
    let mut last_source = 0u32;
    for _ in 0..count {
        let source_symbol_id = buffer.varint32()?.wrapping_add(last_source);
        let delta = buffer.varint32()?;
        if delta > source_symbol_id {
            return Err(DracoError::Corrupt);
        }
        splits.push(TopologySplit {
            source_symbol_id,
            split_symbol_id: source_symbol_id - delta,
            source_edge: 0,
        });
        last_source = source_symbol_id;
    }
    // The edges follow as one bit each, from a plain bit reader.
    buffer.start_bits();
    for split in splits.iter_mut() {
        split.source_edge = buffer.bits(1) & 1;
    }
    buffer.end_bits();
    Ok(splits)
}

/// Read the symbol source and the flag decoders, in the order written.
///
/// The two traversals interleave these differently and it is not negotiable:
/// the standard one writes its symbol stream *first*, then the flags; the valence
/// one writes the flags first and its per-valence streams after them. Each of
/// those reads a length out of the stream, so getting the order wrong puts every
/// later one somewhere arbitrary.
pub fn decode<'a>(
    buffer: &mut Buffer<'a>,
    traversal: super::Traversal,
    num_symbols: u32,
    num_attribute_data: u8,
    num_vertices: u32,
    num_faces: u32,
) -> Result<TraversalData<'a>, DracoError> {
    let mut symbol_bits = 0usize;
    let standard = match traversal {
        super::Traversal::Standard => {
            let (symbols, bits) = decode_symbol_stream(buffer, num_symbols)?;
            symbol_bits = bits;
            Some(symbols)
        }
        super::Traversal::Valence => None,
    };

    let start_faces = BitDecoder::start(buffer)?;
    let mut attribute_seams = Vec::with_capacity(num_attribute_data as usize);
    for _ in 0..num_attribute_data {
        attribute_seams.push(BitDecoder::start(buffer)?);
    }

    let symbols = match standard {
        Some(symbols) => Symbols::Standard(StandardSymbols { symbols, at: 0 }),
        None => Symbols::Valence(ValenceSymbols::start(buffer, num_vertices, num_faces)?),
    };

    Ok(TraversalData {
        symbols,
        start_faces,
        attribute_seams,
        symbol_bits,
    })
}

/// The standard traversal's symbols: a byte count, then that many bytes of bits.
fn decode_symbol_stream(
    buffer: &mut Buffer<'_>,
    num_symbols: u32,
) -> Result<(Vec<Symbol>, usize), DracoError> {
    let size = buffer.varint()? as usize;
    if size > buffer.remaining() {
        return Err(DracoError::Corrupt);
    }
    let start = buffer.at;
    buffer.start_bits();
    let mut symbols = Vec::with_capacity(num_symbols as usize);
    for _ in 0..num_symbols {
        // One bit for `C`, three for everything else.
        let mut code = buffer.bit();
        if code != 0 {
            code |= buffer.bits(2) << 1;
        }
        symbols.push(Symbol::from_code(code).ok_or(DracoError::Corrupt)?);
    }
    buffer.end_bits();
    let symbol_bits = (buffer.at - start) * 8;
    // The stream is padded to a byte, so the symbols must fit in the declared
    // size and leave less than a byte spare. More than that means the bits were
    // read in the wrong order or the wrong number of them.
    if buffer.at > start + size {
        return Err(DracoError::Corrupt);
    }
    buffer.at = start + size;
    Ok((symbols, symbol_bits))
}
