//! Draco's entropy layer: rANS, in the two shapes the format uses.
//!
//! Ported from Draco's own `ans.h`, `rans_symbol_decoder.h` and
//! `symbol_decoding.cc`. The constants are not guessable and not the ones a
//! reading of the rANS papers would suggest — the bit decoder renormalises
//! against `4096` while the symbol decoder uses `precision * 4`, and the
//! precision itself is `clamp(3 * bits / 2, 12, 20)`. Every one of those decodes
//! to a stream of plausible symbols if it is wrong.
//!
//! Both decoders read their buffer **backwards** from the end, which is how rANS
//! works: the encoder's final state is what the decoder starts from.

// The symbol decoder below is the half of the entropy layer the *attribute*
// stage needs, and the valence traversal with it. Connectivity for the standard
// traversal uses only the bit decoder, so until attributes are wired up these
// are written and unreached. They are kept here rather than added later because
// they were ported alongside the bit decoder from the same source, and
// splitting a port across two sittings is how the two halves come to disagree.
#![allow(dead_code)]

use super::buffer::Buffer;
use super::DracoError;

const L_BASE: u32 = 4096;
const IO_BASE: u32 = 256;
const P8_PRECISION: u32 = 256;

/// The state shared by both decoders: a buffer consumed from the back, and the
/// rANS state.
struct State<'a> {
    buf: &'a [u8],
    offset: usize,
    state: u32,
}

impl<'a> State<'a> {
    /// The encoder wrote the final state into the last one to four bytes, with
    /// the top two bits saying how many.
    fn init(buf: &'a [u8], len: usize, base: u32) -> Result<Self, DracoError> {
        if len < 1 || len > buf.len() {
            return Err(DracoError::Corrupt);
        }
        let last = buf[len - 1];
        let tag = last >> 6;
        let (offset, mut state) = match tag {
            0 => (len - 1, (last & 0x3f) as u32),
            1 => {
                if len < 2 {
                    return Err(DracoError::Corrupt);
                }
                (
                    len - 2,
                    u16::from_le_bytes([buf[len - 2], buf[len - 1]]) as u32 & 0x3fff,
                )
            }
            2 => {
                if len < 3 {
                    return Err(DracoError::Corrupt);
                }
                let v = buf[len - 3] as u32
                    | ((buf[len - 2] as u32) << 8)
                    | ((buf[len - 1] as u32) << 16);
                (len - 3, v & 0x3f_ffff)
            }
            _ => {
                if len < 4 {
                    return Err(DracoError::Corrupt);
                }
                let v = u32::from_le_bytes([buf[len - 4], buf[len - 3], buf[len - 2], buf[len - 1]]);
                (len - 4, v & 0x3fff_ffff)
            }
        };
        state += base;
        if state >= base.saturating_mul(IO_BASE) {
            return Err(DracoError::Corrupt);
        }
        Ok(State {
            buf,
            offset,
            state,
        })
    }
}

/// A sequence of bits, each coded against one fixed probability.
///
/// Used for the flags the format is littered with: whether a face is on a hole,
/// whether a prediction was taken, which corner a split came from.
pub struct BitDecoder<'a> {
    prob_zero: u8,
    state: State<'a>,
}

impl<'a> BitDecoder<'a> {
    pub fn start(buffer: &mut Buffer<'a>) -> Result<Self, DracoError> {
        let prob_zero = buffer.u8()?;
        let size = buffer.varint32()? as usize;
        if size > buffer.remaining() {
            return Err(DracoError::Corrupt);
        }
        let head = buffer.head();
        let state = State::init(head, size, L_BASE)?;
        buffer.advance(size)?;
        Ok(BitDecoder { prob_zero, state })
    }

    /// rABS with descending spread — the exact arithmetic from `rabs_desc_read`.
    pub fn bit(&mut self) -> u32 {
        let p = P8_PRECISION - self.prob_zero as u32;
        if self.state.state < L_BASE && self.state.offset > 0 {
            self.state.offset -= 1;
            self.state.state = self.state.state * IO_BASE + self.state.buf[self.state.offset] as u32;
        }
        let x = self.state.state;
        let quotient = x / P8_PRECISION;
        let remainder = x % P8_PRECISION;
        let scaled = quotient * p;
        if remainder < p {
            self.state.state = scaled + remainder;
            1
        } else {
            self.state.state = x - scaled - p;
            0
        }
    }

    /// `n` bits, most significant first — note the order differs from the
    /// buffer's bit mode, which is least significant first.
    pub fn bits(&mut self, n: u32) -> u32 {
        let mut value = 0u32;
        for _ in 0..n.min(32) {
            value = (value << 1) + self.bit();
        }
        value
    }
}

/// A symbol stream with a decoded probability table.
struct SymbolDecoder<'a> {
    /// `precision` entries mapping a remainder to the symbol that owns it.
    lut: Vec<u32>,
    /// `(probability, cumulative)` per symbol.
    table: Vec<(u32, u32)>,
    precision: u32,
    num_symbols: u32,
    state: Option<State<'a>>,
}

/// `clamp(3 * bits / 2, 12, 20)` — the rANS library only works in that range.
fn precision_bits(unique_symbols_bit_length: u32) -> u32 {
    ((3 * unique_symbols_bit_length) / 2).clamp(12, 20)
}

impl<'a> SymbolDecoder<'a> {
    /// Read the probability table.
    fn create(buffer: &mut Buffer<'a>, symbol_bits: u32) -> Result<Self, DracoError> {
        let precision = 1u32 << precision_bits(symbol_bits);
        let num_symbols = buffer.varint32()?;
        // The table needs at least one byte per 64 symbols, so a count far
        // beyond the buffer is corruption rather than a huge alphabet.
        if (num_symbols as usize) / 64 > buffer.remaining() {
            return Err(DracoError::Corrupt);
        }
        let mut probabilities = vec![0u32; num_symbols as usize];
        let mut i = 0usize;
        while i < num_symbols as usize {
            let first = buffer.u8()?;
            // The low two bits are a token: 0-2 say how many extra bytes carry
            // the probability, 3 is a run of zero-probability symbols.
            let token = first & 3;
            if token == 3 {
                let run = (first >> 2) as usize;
                if i + run >= num_symbols as usize {
                    return Err(DracoError::Corrupt);
                }
                for j in 0..=run {
                    probabilities[i + j] = 0;
                }
                i += run;
            } else {
                let mut probability = (first >> 2) as u32;
                for b in 0..token {
                    let extra = buffer.u8()? as u32;
                    probability |= extra << (8 * (b + 1) - 2);
                }
                probabilities[i] = probability;
            }
            i += 1;
        }

        // The look-up table: every remainder in [0, precision) maps to a symbol.
        let mut lut = vec![0u32; precision as usize];
        let mut table = Vec::with_capacity(num_symbols as usize);
        let mut cumulative = 0u32;
        for (symbol, &probability) in probabilities.iter().enumerate() {
            table.push((probability, cumulative));
            let next = cumulative + probability;
            if next > precision {
                return Err(DracoError::Corrupt);
            }
            for slot in cumulative..next {
                lut[slot as usize] = symbol as u32;
            }
            cumulative = next;
        }
        // The probabilities have to fill the precision exactly, or the decoder
        // has remainders that belong to no symbol.
        if cumulative != precision {
            return Err(DracoError::Corrupt);
        }
        Ok(SymbolDecoder {
            lut,
            table,
            precision,
            num_symbols,
            state: None,
        })
    }

    fn start(&mut self, buffer: &mut Buffer<'a>) -> Result<(), DracoError> {
        let bytes = buffer.varint()? as usize;
        if bytes > buffer.remaining() {
            return Err(DracoError::Corrupt);
        }
        let head = buffer.head();
        self.state = Some(State::init(head, bytes, self.precision * 4)?);
        buffer.advance(bytes)?;
        Ok(())
    }

    fn symbol(&mut self) -> u32 {
        let base = self.precision * 4;
        let Some(state) = &mut self.state else {
            return 0;
        };
        while state.state < base && state.offset > 0 {
            state.offset -= 1;
            state.state = state.state * IO_BASE + state.buf[state.offset] as u32;
        }
        let quotient = state.state / self.precision;
        let remainder = state.state % self.precision;
        let symbol = self.lut[remainder as usize];
        let (probability, cumulative) = self.table[symbol as usize];
        state.state = quotient * probability + remainder - cumulative;
        symbol
    }
}

/// Decode `count` symbols, in whichever of the two schemes the stream uses.
///
/// `components` matters only for the tagged scheme, where one tag gives the bit
/// length for a whole element — three position components share one tag,
/// because they tend to need the same number of bits.
pub fn decode_symbols(
    count: usize,
    components: usize,
    buffer: &mut Buffer<'_>,
) -> Result<Vec<u32>, DracoError> {
    let mut out = vec![0u32; count];
    if count == 0 {
        return Ok(out);
    }
    let scheme = buffer.u8()?;
    match scheme {
        0 => decode_tagged(&mut out, components.max(1), buffer)?,
        1 => decode_raw(&mut out, buffer)?,
        _ => return Err(DracoError::Corrupt),
    }
    Ok(out)
}

/// One rANS-coded tag per element giving a bit length, then the values as raw
/// bits of that length.
fn decode_tagged(
    out: &mut [u32],
    components: usize,
    buffer: &mut Buffer<'_>,
) -> Result<(), DracoError> {
    let mut tags = SymbolDecoder::create(buffer, 5)?;
    if out.is_empty() && tags.num_symbols == 0 {
        return Err(DracoError::Corrupt);
    }
    tags.start(buffer)?;
    // The tag stream and the value bits are interleaved in *position*, not in
    // the stream: the tags were read from the back, and the values start where
    // the byte reader now is.
    buffer.start_bits();
    let mut at = 0usize;
    while at < out.len() {
        let bit_length = tags.symbol();
        for _ in 0..components {
            if at >= out.len() {
                break;
            }
            out[at] = buffer.bits(bit_length);
            at += 1;
        }
    }
    buffer.end_bits();
    Ok(())
}

/// Every value straight out of one rANS stream.
fn decode_raw(out: &mut [u32], buffer: &mut Buffer<'_>) -> Result<(), DracoError> {
    let max_bit_length = buffer.u8()? as u32;
    if max_bit_length == 0 || max_bit_length > 18 {
        return Err(DracoError::Corrupt);
    }
    let mut decoder = SymbolDecoder::create(buffer, max_bit_length)?;
    if !out.is_empty() && decoder.num_symbols == 0 {
        return Err(DracoError::Corrupt);
    }
    decoder.start(buffer)?;
    for slot in out.iter_mut() {
        *slot = decoder.symbol();
    }
    Ok(())
}
