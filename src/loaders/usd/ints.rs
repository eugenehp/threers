//! USD's integer packing, which sits under the LZ4 in a crate file.
//!
//! The arrays a crate stores — token indices, path indices, spec types — are
//! mostly small numbers that mostly step by the same amount. USD exploits that
//! twice over: it stores *deltas* rather than values, and it spends two bits per
//! delta saying how wide that delta is.
//!
//! ```text
//! [ common value : 4 or 8 bytes ]
//! [ two bits per element, four to a byte, least significant first ]
//! [ the deltas that were not the common value, each 1, 2 or 4 bytes ]
//! ```
//!
//! A code of 0 means "this delta is the common value" and costs nothing beyond
//! its two bits; 1, 2 and 3 mean a signed delta of one, two or four bytes. The
//! values are then a running sum. An array of consecutive integers — which is
//! what a path table largely is — comes out at two bits an element.

/// Decode a 32-bit integer array from its packed form.
///
/// `count` comes from the section header, so the buffer's length is known
/// before it is read and a short one is corruption rather than a surprise.
pub fn decode_u32(src: &[u8], count: usize) -> Option<Vec<i32>> {
    if count == 0 {
        return Some(Vec::new());
    }
    let common = i32::from_le_bytes(src.get(0..4)?.try_into().ok()?);
    let codes_len = count.div_ceil(4);
    let codes = src.get(4..4 + codes_len)?;
    let mut at = 4 + codes_len;

    let mut out = Vec::with_capacity(count);
    let mut previous = 0i32;
    for i in 0..count {
        let code = (codes[i / 4] >> ((i % 4) * 2)) & 0b11;
        let delta = match code {
            0 => common,
            1 => {
                let v = *src.get(at)? as i8 as i32;
                at += 1;
                v
            }
            2 => {
                let v = i16::from_le_bytes(src.get(at..at + 2)?.try_into().ok()?) as i32;
                at += 2;
                v
            }
            _ => {
                let v = i32::from_le_bytes(src.get(at..at + 4)?.try_into().ok()?);
                at += 4;
                v
            }
        };
        previous = previous.wrapping_add(delta);
        out.push(previous);
    }
    Some(out)
}

/// The same for 64-bit arrays, which differ only in the width of the common
/// value and of the widest delta.
pub fn decode_u64(src: &[u8], count: usize) -> Option<Vec<i64>> {
    if count == 0 {
        return Some(Vec::new());
    }
    let common = i64::from_le_bytes(src.get(0..8)?.try_into().ok()?);
    let codes_len = count.div_ceil(4);
    let codes = src.get(8..8 + codes_len)?;
    let mut at = 8 + codes_len;

    let mut out = Vec::with_capacity(count);
    let mut previous = 0i64;
    for i in 0..count {
        let code = (codes[i / 4] >> ((i % 4) * 2)) & 0b11;
        let delta = match code {
            0 => common,
            1 => {
                let v = *src.get(at)? as i8 as i64;
                at += 1;
                v
            }
            2 => {
                let v = i16::from_le_bytes(src.get(at..at + 2)?.try_into().ok()?) as i64;
                at += 2;
                v
            }
            _ => {
                let v = i32::from_le_bytes(src.get(at..at + 4)?.try_into().ok()?) as i64;
                at += 4;
                v
            }
        };
        previous = previous.wrapping_add(delta);
        out.push(previous);
    }
    Some(out)
}

/// Read `u64 compressedSize` and the LZ4 body that follows it, then unpack
/// `count` 32-bit integers — the shape every compressed array in a crate has.
pub fn read_u32_array(file: &[u8], at: &mut usize, count: usize) -> Option<Vec<i32>> {
    let size = u64::from_le_bytes(file.get(*at..*at + 8)?.try_into().ok()?) as usize;
    *at += 8;
    let body = file.get(*at..*at + size)?;
    *at += size;
    // The packed form is at most this big: a common value, the codes, and a
    // four-byte delta each. Checked, because `count` came out of the file and
    // a corrupt one overflows the multiply rather than merely being wrong.
    let ceiling = count
        .checked_mul(4)?
        .checked_add(count.div_ceil(4))?
        .checked_add(8)?;
    let packed = super::lz4::decompress_upto(body, ceiling)?;
    decode_u32(&packed, count)
}

/// Pack a 32-bit integer array the way [`decode_u32`] reads it.
///
/// The common value is whichever delta occurs most often, because every
/// occurrence of it then costs two bits and nothing else. For the arrays this
/// is used on — path indices, spec types, runs of consecutive numbers — that
/// is usually `1` or `0` and most of the array disappears.
pub fn encode_u32(values: &[i32]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut deltas = Vec::with_capacity(values.len());
    let mut previous = 0i32;
    for v in values {
        deltas.push(v.wrapping_sub(previous));
        previous = *v;
    }
    let common = most_common(&deltas);

    let mut out = common.to_le_bytes().to_vec();
    let codes_at = out.len();
    out.resize(codes_at + values.len().div_ceil(4), 0);
    for (i, delta) in deltas.iter().enumerate() {
        let code = if *delta == common {
            0u8
        } else if i8::try_from(*delta).is_ok() {
            out.push(*delta as i8 as u8);
            1
        } else if let Ok(narrow) = i16::try_from(*delta) {
            out.extend_from_slice(&narrow.to_le_bytes());
            2
        } else {
            out.extend_from_slice(&delta.to_le_bytes());
            3
        };
        out[codes_at + i / 4] |= code << ((i % 4) * 2);
    }
    out
}

/// The same for 64-bit arrays. A delta too wide for `i32` cannot be encoded at
/// all — the format's widest code is four bytes — so those arrays fall back to
/// a common value of zero and a full-width code, which is still correct
/// because the deltas themselves are what is stored.
pub fn encode_u64(values: &[i64]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut deltas = Vec::with_capacity(values.len());
    let mut previous = 0i64;
    for v in values {
        deltas.push(v.wrapping_sub(previous));
        previous = *v;
    }
    let common = most_common(&deltas);

    let mut out = common.to_le_bytes().to_vec();
    let codes_at = out.len();
    out.resize(codes_at + values.len().div_ceil(4), 0);
    for (i, delta) in deltas.iter().enumerate() {
        let code = if *delta == common {
            0u8
        } else if i8::try_from(*delta).is_ok() {
            out.push(*delta as i8 as u8);
            1
        } else if let Ok(narrow) = i16::try_from(*delta) {
            out.extend_from_slice(&narrow.to_le_bytes());
            2
        } else {
            // Wider than four bytes cannot be expressed; this is why
            // `encodable_u64` exists and is checked before choosing this path.
            out.extend_from_slice(&(*delta as i32).to_le_bytes());
            3
        };
        out[codes_at + i / 4] |= code << ((i % 4) * 2);
    }
    out
}

/// Whether a 64-bit array's deltas all fit the format's four-byte widest code.
///
/// They usually do — the values may be enormous while the *steps* between them
/// are small — but an array that jumps around above 2^31 has to be written
/// uncompressed instead.
pub fn encodable_u64(values: &[i64]) -> bool {
    let mut previous = 0i64;
    for v in values {
        let delta = v.wrapping_sub(previous);
        if i32::try_from(delta).is_err() {
            return false;
        }
        previous = *v;
    }
    true
}

fn most_common<T: Copy + Eq + std::hash::Hash + Default>(deltas: &[T]) -> T {
    let mut counts: std::collections::HashMap<T, usize> = std::collections::HashMap::new();
    for d in deltas {
        *counts.entry(*d).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(v, _)| v)
        .unwrap_or_default()
}

/// Write a 32-bit array as `u64 compressedSize` and the packed, compressed
/// body — the shape [`read_u32_array`] expects.
pub fn write_u32_array(out: &mut Vec<u8>, values: &[i32]) {
    let body = super::lz4::compress(&encode_u32(values));
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(&body);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The field-name indices of a file `usdcat` wrote. Every one has to be a
    /// real token, which is a check the encoder cannot accidentally pass.
    #[test]
    fn decodes_the_field_names_of_a_real_crate() {
        let file = include_bytes!("testdata/triangle.usdc");
        let u64at = |at: usize| u64::from_le_bytes(file[at..at + 8].try_into().unwrap()) as usize;
        // Section offsets come from the file's own table of contents, so
        // regenerating the fixture does not silently point this at the wrong
        // bytes.
        let info = super::super::usdc::info(file).unwrap();
        let section = |name: &str| {
            info.sections
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, at, _)| *at as usize)
                .unwrap_or_else(|| panic!("no {name} section"))
        };

        // The token table, for checking the indices land on something.
        let tokens_at = section("TOKENS");
        let token_count = u64at(tokens_at);
        let text = super::super::lz4::decompress(
            &file[tokens_at + 24..tokens_at + 24 + u64at(tokens_at + 16)],
            u64at(tokens_at + 8),
        )
        .unwrap();
        let tokens: Vec<String> = text
            .split(|b| *b == 0)
            .take(token_count)
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();

        // FIELDS: a count, then the compressed array of token indices.
        let fields_at = section("FIELDS");
        let count = u64at(fields_at);
        let mut at = fields_at + 8;
        let indices = read_u32_array(file, &mut at, count).expect("field names decode");

        assert_eq!(indices.len(), count);
        let names: Vec<&str> = indices
            .iter()
            .map(|i| tokens.get(*i as usize).map(String::as_str).unwrap_or("<oob>"))
            .collect();
        assert!(
            !names.contains(&"<oob>"),
            "an index pointed outside the token table: {names:?}"
        );
        // FIELDS holds the *keys* a spec authors — `typeName`, `default`,
        // `specifier` — not the names of the properties, which live in the
        // path table. Getting real ones out is the check that the deltas were
        // summed in the right order with the right widths.
        for expected in [
            "defaultPrim",
            "metersPerUnit",
            "upAxis",
            "specifier",
            "typeName",
            "properties",
            "variability",
            "default",
        ] {
            assert!(names.contains(&expected), "missing {expected} in {names:?}");
        }
    }

    #[test]
    fn what_is_packed_unpacks() {
        let cases: Vec<Vec<i32>> = vec![
            vec![],
            vec![0],
            vec![7],
            (0..100).collect(),
            vec![0, 1, 2, 3, 4, 5],
            vec![5, 5, 5, 5, 5],
            vec![0, 1000, -1000, 70000, -70000, 0],
            vec![i32::MIN, i32::MAX, 0, i32::MAX],
            (0..50).map(|i| i * i).collect(),
        ];
        for case in cases {
            let packed = encode_u32(&case);
            assert_eq!(
                decode_u32(&packed, case.len()).as_deref(),
                Some(&case[..]),
                "round trip of {case:?}"
            );
        }
    }

    #[test]
    fn wide_values_round_trip_when_their_steps_are_narrow() {
        let case: Vec<i64> = vec![2_000_000_000, 2_000_000_001, 2_000_000_002, 1_999_999_999];
        assert!(encodable_u64(&case));
        let packed = encode_u64(&case);
        assert_eq!(decode_u64(&packed, case.len()).as_deref(), Some(&case[..]));
    }

    /// A step wider than four bytes cannot be written, and saying so is what
    /// keeps the writer from producing a file it could not read back.
    ///
    /// The first element counts as a step from zero, so an array that *starts*
    /// beyond `i32` is unpackable however gentle the rest of it is. That is a
    /// property of the format, not of this encoder, and it is why a fallback
    /// to an uncompressed array has to exist.
    #[test]
    fn a_step_too_wide_is_refused_rather_than_truncated() {
        assert!(!encodable_u64(&[0, 1 << 40]));
        assert!(!encodable_u64(&[1 << 40, (1 << 40) + 1]), "the first step counts");
        assert!(encodable_u64(&[0, 1, 2]));
    }

    /// A run of consecutive integers is what a path table mostly is, and it
    /// has to cost two bits an entry or the format is pointless.
    #[test]
    fn a_consecutive_run_packs_to_two_bits_each() {
        let packed = encode_u32(&(0..1000).collect::<Vec<_>>());
        // Four bytes of common value, a quarter of a byte each, and one byte
        // for the odd one out: the step from zero to the first element is 0
        // where every later step is 1, so it cannot use the common value.
        assert_eq!(packed.len(), 4 + 250 + 1);
    }

    #[test]
    fn a_run_of_consecutive_values_costs_two_bits_each() {
        // Common value 1, four codes of 0, and no payload at all.
        let mut packed = 1i32.to_le_bytes().to_vec();
        packed.push(0b0000_0000);
        assert_eq!(decode_u32(&packed, 4).unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn mixed_widths_are_read_in_order() {
        // Codes 1, 2, 3, 0 — an i8, an i16, an i32, then the common value.
        let mut packed = 10i32.to_le_bytes().to_vec();
        packed.push(0b00_11_10_01);
        packed.push(5u8); // +5
        packed.extend_from_slice(&(-3i16).to_le_bytes()); // -3
        packed.extend_from_slice(&1000i32.to_le_bytes()); // +1000
        assert_eq!(decode_u32(&packed, 4).unwrap(), vec![5, 2, 1002, 1012]);
    }

    #[test]
    fn a_truncated_buffer_is_none() {
        let packed = 1i32.to_le_bytes().to_vec();
        assert!(decode_u32(&packed, 4).is_none());
        assert!(decode_u32(&[], 1).is_none());
        assert_eq!(decode_u32(&[], 0).unwrap(), Vec::<i32>::new());
    }
}
