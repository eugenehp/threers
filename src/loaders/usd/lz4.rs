//! LZ4 block decoding, and the chunked wrapper USD puts around it.
//!
//! A crate file's sections are compressed with LZ4's *block* format — no frame
//! header, no checksum, just tokens — behind one byte saying how many chunks
//! follow. That byte is zero for anything under the chunk limit, which in
//! practice is every section of every file that is not enormous.
//!
//! # The block format
//!
//! A block is a run of sequences. Each begins with a token byte: the high
//! nibble is how many literal bytes follow, the low nibble is how many bytes to
//! copy from earlier in the output. A nibble of 15 means "and keep reading
//! bytes, adding each, until one is not 255". After the literals comes a
//! two-byte little-endian offset back into the output, and the match length is
//! the low nibble plus four — a match is never shorter than that, which is what
//! makes four a useful bias.
//!
//! Matches may overlap their own output, and that is not a corner case: a run
//! of one repeated byte is encoded as a one-byte literal and a match at offset
//! one. The copy therefore has to be byte at a time rather than a block move.

/// Decode one LZ4 block.
///
/// `limit` is a hard ceiling — USD stores the decompressed size, so an overrun
/// is corruption rather than a reason to grow. Whether falling *short* of the
/// limit is also an error depends on the caller: a section knows its exact
/// size, a packed integer array only knows a bound.
fn decode(src: &[u8], limit: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(limit.min(1 << 20));
    let mut i = 0usize;

    while i < src.len() {
        let token = src[i];
        i += 1;

        // Literals.
        let mut literal = (token >> 4) as usize;
        if literal == 15 {
            loop {
                let b = *src.get(i)?;
                i += 1;
                literal += b as usize;
                if b != 255 {
                    break;
                }
            }
        }
        if literal > 0 {
            let end = i.checked_add(literal)?;
            out.extend_from_slice(src.get(i..end)?);
            i = end;
        }
        // The last sequence is literals only and stops here.
        if i >= src.len() {
            break;
        }

        // Match: a distance back into what has been written already.
        let offset = u16::from_le_bytes([*src.get(i)?, *src.get(i + 1)?]) as usize;
        i += 2;
        if offset == 0 || offset > out.len() {
            return None;
        }
        let mut length = (token & 0x0F) as usize;
        if length == 15 {
            loop {
                let b = *src.get(i)?;
                i += 1;
                length += b as usize;
                if b != 255 {
                    break;
                }
            }
        }
        length += 4;

        // Byte at a time: the match is allowed to overlap what it is writing,
        // which is how a repeated byte is encoded.
        let start = out.len() - offset;
        for k in 0..length {
            let byte = out[start + k];
            out.push(byte);
        }
        if out.len() > limit {
            return None;
        }
    }
    Some(out)
}

/// Decompress a section body: one byte of chunk count, then the chunks.
///
/// Zero chunks means the whole remainder is a single block — the common case.
/// Otherwise each chunk is preceded by its own compressed length, because a
/// buffer larger than LZ4's input limit has to be cut up to be compressed at
/// all.
pub fn decompress(src: &[u8], expected: usize) -> Option<Vec<u8>> {
    let out = chunked(src, expected)?;
    (out.len() == expected).then_some(out)
}

/// Decompress when the exact output size is not known, only a ceiling.
///
/// The integer arrays are stored with a bound rather than a length — the packed
/// form is "at most this big", since how many of the deltas needed four bytes
/// is not known until they are read — so the block is decoded until it runs
/// out and whatever it produced is the answer.
pub fn decompress_upto(src: &[u8], ceiling: usize) -> Option<Vec<u8>> {
    chunked(src, ceiling)
}

fn chunked(src: &[u8], limit: usize) -> Option<Vec<u8>> {
    let chunks = *src.first()?;
    if chunks == 0 {
        return decode(&src[1..], limit);
    }
    // Each chunk decodes to the maximum block size except the last.
    const CHUNK: usize = 0x7E00_0000;
    let mut out = Vec::new();
    let mut at = 1usize;
    for chunk in 0..chunks as usize {
        let size = u64::from_le_bytes(src.get(at..at + 8)?.try_into().ok()?) as usize;
        at += 8;
        let body = src.get(at..at + size)?;
        at += size;
        let room = limit.checked_sub(out.len())?;
        let want = if chunk + 1 == chunks as usize {
            room
        } else {
            CHUNK.min(room)
        };
        out.extend_from_slice(&decode(body, want)?);
    }
    Some(out)
}

/// Compress into one LZ4 block, behind USD's chunk-count byte.
///
/// A greedy matcher with a small hash table: for each position, look up where
/// the last four bytes were seen before, and if they are still within reach
/// emit a match rather than the literals. That is the whole of LZ4 — there is
/// no entropy coding underneath — so a simple matcher gives up compression
/// ratio against `liblz4` but produces a stream any decoder reads.
pub fn compress(src: &[u8]) -> Vec<u8> {
    // A match is never shorter than this, which is what makes it worth two
    // bytes of offset.
    const MIN_MATCH: usize = 4;
    // The format requires the block to end in at least five literal bytes, and
    // no match may start within twelve of the end.
    const LAST_LITERALS: usize = 5;
    const MATCH_LIMIT: usize = 12;
    const HASH_BITS: u32 = 14;

    // The chunk count, zero for a single block.
    let mut out = vec![0u8];
    if src.len() < MATCH_LIMIT + 1 {
        emit_literals(&mut out, src);
        return out;
    }

    // Positions are stored one higher so that zero can mean "never seen".
    let mut table = vec![0u32; 1 << HASH_BITS];
    let mut anchor = 0usize;
    let mut i = 0usize;
    let limit = src.len() - MATCH_LIMIT;

    while i < limit {
        let key = hash(&src[i..i + MIN_MATCH], HASH_BITS);
        let candidate = table[key] as usize;
        table[key] = (i + 1) as u32;

        // No previous sighting, out of range, or a hash collision that is not
        // a real match: this byte is a literal.
        if candidate == 0
            || i + 1 - candidate > 0xFFFF
            || src[candidate - 1..candidate - 1 + MIN_MATCH] != src[i..i + MIN_MATCH]
        {
            i += 1;
            continue;
        }
        let at = candidate - 1;

        let mut length = MIN_MATCH;
        let end = src.len() - LAST_LITERALS;
        while i + length < end && src[at + length] == src[i + length] {
            length += 1;
        }
        emit_sequence(&mut out, &src[anchor..i], i - at, length);
        i += length;
        anchor = i;
    }
    emit_literals(&mut out, &src[anchor..]);
    out
}

fn hash(four: &[u8], bits: u32) -> usize {
    let v = u32::from_le_bytes(four.try_into().unwrap());
    // Knuth's multiplicative hash, which is what LZ4's own matcher uses.
    (v.wrapping_mul(2654435761) >> (32 - bits)) as usize
}

/// A length above what fits in a nibble, as a chain of bytes ending in one
/// that is not 255.
fn emit_extension(out: &mut Vec<u8>, mut remaining: usize) {
    while remaining >= 255 {
        out.push(255);
        remaining -= 255;
    }
    out.push(remaining as u8);
}

fn emit_sequence(out: &mut Vec<u8>, literals: &[u8], offset: usize, length: usize) {
    let match_length = length - 4;
    let token = ((literals.len().min(15) as u8) << 4) | match_length.min(15) as u8;
    out.push(token);
    if literals.len() >= 15 {
        emit_extension(out, literals.len() - 15);
    }
    out.extend_from_slice(literals);
    out.extend_from_slice(&(offset as u16).to_le_bytes());
    if match_length >= 15 {
        emit_extension(out, match_length - 15);
    }
}

/// The final sequence, which has literals and no match.
fn emit_literals(out: &mut Vec<u8>, literals: &[u8]) {
    if literals.is_empty() {
        return;
    }
    out.push((literals.len().min(15) as u8) << 4);
    if literals.len() >= 15 {
        emit_extension(out, literals.len() - 15);
    }
    out.extend_from_slice(literals);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The TOKENS section of a file `usdcat` wrote, decoded against the size it
    /// recorded — the only check that matters, since the encoder is not ours.
    #[test]
    fn decodes_a_real_token_section() {
        let file = include_bytes!("testdata/triangle.usdc");
        // From the table of contents: TOKENS at 260, and its own header gives
        // the counts.
        let at = 260usize;
        let count = u64::from_le_bytes(file[at..at + 8].try_into().unwrap()) as usize;
        let uncompressed = u64::from_le_bytes(file[at + 8..at + 16].try_into().unwrap()) as usize;
        let compressed = u64::from_le_bytes(file[at + 16..at + 24].try_into().unwrap()) as usize;
        assert_eq!(count, 27);

        let body = &file[at + 24..at + 24 + compressed];
        let text = decompress(body, uncompressed).expect("token section decodes");
        assert_eq!(text.len(), uncompressed);

        // The table is null-*terminated*, so splitting leaves a trailing empty
        // piece; and one of the tokens is itself the empty string, so the
        // count is taken from the header rather than by filtering.
        let names: Vec<&str> = text
            .split(|b| *b == 0)
            .take(count)
            .map(|s| std::str::from_utf8(s).unwrap())
            .collect();
        assert_eq!(names.len(), count);
        // The tokens a layer of this shape has to contain.
        for expected in ["defaultPrim", "Root", "Tri", "points", "faceVertexIndices", "Mesh"] {
            assert!(names.contains(&expected), "missing {expected} in {names:?}");
        }
    }

    /// Whatever the compressor produces, this crate's own decoder reads back
    /// — over inputs chosen to exercise matches, literals and the awkward
    /// lengths around the nibble boundaries.
    #[test]
    fn compression_round_trips() {
        let cases: Vec<Vec<u8>> = vec![
            Vec::new(),
            b"a".to_vec(),
            b"hello".to_vec(),
            vec![0u8; 1000],
            b"abcabcabcabcabcabcabcabcabcabcabc".to_vec(),
            // A run long enough to need a length extension.
            vec![7u8; 300],
            // Every length around the 12-byte match limit and the 15-byte
            // nibble boundary, where an off-by-one shows up.
            (0..40u8).collect(),
            b"the quick brown fox jumps over the quick brown fox".to_vec(),
        ];
        for case in cases {
            let packed = compress(&case);
            let back = decompress(&packed, case.len())
                .unwrap_or_else(|| panic!("failed to read back {} bytes", case.len()));
            assert_eq!(back, case, "round trip of {} bytes", case.len());
        }
    }

    #[test]
    fn every_length_up_to_a_few_hundred_round_trips() {
        // Repetitive enough to produce matches, varied enough that the matcher
        // has to stop and start.
        let source: Vec<u8> = (0..400u32).map(|i| (i % 17) as u8).collect();
        for n in 0..source.len() {
            let case = &source[..n];
            let packed = compress(case);
            assert_eq!(
                decompress(&packed, n).as_deref(),
                Some(case),
                "round trip at length {n}"
            );
        }
    }

    /// Repetitive data has to actually get smaller, or the matcher is not
    /// matching and the tests above would pass on a literal-only encoder.
    #[test]
    fn repetition_compresses() {
        let packed = compress(&vec![0u8; 10_000]);
        assert!(packed.len() < 200, "10k zeros became {} bytes", packed.len());

        let text = b"the quick brown fox ".repeat(500);
        let packed = compress(&text);
        assert!(
            packed.len() < text.len() / 10,
            "{} bytes became {}",
            text.len(),
            packed.len()
        );
    }

    #[test]
    fn an_overlapping_match_repeats_a_byte() {
        // One literal "a", then a match at offset 1 whose low nibble is 0, so
        // length is the minimum 4 — the encoding of "aaaaa" that a block copy
        // would get wrong.
        let block = [0x10u8, b'a', 0x01, 0x00];
        assert_eq!(decode(&block, 5).unwrap(), b"aaaaa");
    }

    #[test]
    fn extended_lengths_chain_through_255() {
        // 19 literals: high nibble 15, then one more byte of 4.
        let mut block = vec![0xF0u8, 4];
        block.extend_from_slice(&[b'z'; 19]);
        assert_eq!(decode(&block, 19).unwrap(), vec![b'z'; 19]);
    }

    #[test]
    fn corruption_is_none_rather_than_a_panic() {
        // An offset pointing before the start of the output.
        assert!(decode(&[0x04, 0x09, 0x00], 8).is_none());
        // More literals promised than the block holds.
        assert!(decode(&[0x50, b'a'], 99).is_none());
        // A promised length extension that never arrives.
        assert!(decode(&[0xF0], 99).is_none());

        // Falling short of the expected size is not the block decoder's
        // business: `[0x10, b'a']` is one perfectly good literal. It is the
        // caller who was told to expect 99 bytes that has to object.
        assert_eq!(decode(&[0x10, b'a'], 99).unwrap(), b"a");
        assert_eq!(decode(&[], 4).unwrap(), Vec::<u8>::new());
        assert!(decompress(&[0x00, 0x10, b'a'], 99).is_none());
        assert!(decompress(&[0x00], 4).is_none());
        assert!(decompress(&[], 4).is_none());
    }
}
