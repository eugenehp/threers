//! The binary form: a node tree written with byte offsets.
//!
//! Every read here is bounds-checked and every failure is an error rather than
//! a panic. A loader is a trust boundary — the bytes come from a file someone
//! else wrote — and the previous implementation sliced on lengths taken
//! straight out of those bytes, so a truncated file took the process down.

use super::tree::{Node, Property};
use super::FbxError;

/// A cursor over the file that refuses to read past the end.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], FbxError> {
        let end = self.at.checked_add(n).ok_or(FbxError::Corrupt)?;
        let slice = self.bytes.get(self.at..end).ok_or(FbxError::Corrupt)?;
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, FbxError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, FbxError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, FbxError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// A node header's three counts, which are 32-bit before version 7500 and
    /// 64-bit from it. Getting this wrong reads a plausible-looking tree from
    /// the wrong offsets, which is why it is decided by the version and not by
    /// guessing.
    fn offsets(&mut self, wide: bool) -> Result<(usize, usize, usize), FbxError> {
        if wide {
            Ok((
                self.u64()? as usize,
                self.u64()? as usize,
                self.u64()? as usize,
            ))
        } else {
            Ok((
                self.u32()? as usize,
                self.u32()? as usize,
                self.u32()? as usize,
            ))
        }
    }
}

/// Parse the whole document into one root node holding the top-level nodes.
pub fn parse(bytes: &[u8]) -> Result<Node, FbxError> {
    // 21 bytes of magic, two more of `[0x1A, 0x00]`, then the version.
    if bytes.len() < 27 || !bytes.starts_with(b"Kaydara FBX Binary") {
        return Err(FbxError::BadHeader);
    }
    let version = u32::from_le_bytes([bytes[23], bytes[24], bytes[25], bytes[26]]);
    let wide = version >= 7500;
    let mut reader = Reader { bytes, at: 27 };
    let children = nodes(&mut reader, wide, bytes.len(), 0)?;
    Ok(Node {
        name: String::new(),
        properties: vec![Property::Int(version as i32)],
        children,
    })
}

/// How deep a tree this will follow. Real files nest a dozen or so; anything
/// beyond this is a cycle or a corrupt offset, and recursing on it overflows
/// the stack.
const MAX_DEPTH: usize = 64;

fn nodes(
    reader: &mut Reader<'_>,
    wide: bool,
    limit: usize,
    depth: usize,
) -> Result<Vec<Node>, FbxError> {
    if depth > MAX_DEPTH {
        return Err(FbxError::Corrupt);
    }
    let mut out = Vec::new();
    loop {
        let start = reader.at;
        let Ok((end_offset, property_count, _property_bytes)) = reader.offsets(wide) else {
            return Ok(out);
        };
        let name_len = reader.u8().unwrap_or(0) as usize;
        // A header of all zeroes is the sentinel that ends a sibling list.
        if end_offset == 0 {
            return Ok(out);
        }
        // An offset that goes backwards or past the end is corruption; without
        // this the loop below never advances and the parse hangs.
        if end_offset <= start || end_offset > limit {
            return Err(FbxError::Corrupt);
        }
        let name = String::from_utf8_lossy(reader.take(name_len)?).into_owned();

        let mut properties = Vec::with_capacity(property_count.min(64));
        for _ in 0..property_count {
            let kind = reader.u8()?;
            properties.push(property(reader, kind)?);
        }

        // Whatever is left before `end_offset` is this node's children.
        let children = if reader.at < end_offset {
            nodes(reader, wide, end_offset, depth + 1)?
        } else {
            Vec::new()
        };
        reader.at = end_offset;
        out.push(Node {
            name,
            properties,
            children,
        });
        if reader.at >= limit {
            return Ok(out);
        }
    }
}

fn property(reader: &mut Reader<'_>, kind: u8) -> Result<Property, FbxError> {
    Ok(match kind {
        b'C' => Property::Bool(reader.u8()? != 0),
        b'Y' => {
            let b = reader.take(2)?;
            Property::Short(i16::from_le_bytes([b[0], b[1]]))
        }
        b'I' => Property::Int(reader.u32()? as i32),
        b'L' => Property::Long(reader.u64()? as i64),
        b'F' => Property::Float(f32::from_bits(reader.u32()?)),
        b'D' => Property::Double(f64::from_bits(reader.u64()?)),
        b'S' | b'R' => {
            let len = reader.u32()? as usize;
            let data = reader.take(len)?;
            if kind == b'S' {
                Property::String(String::from_utf8_lossy(data).into_owned())
            } else {
                Property::Raw(data.to_vec())
            }
        }
        b'd' | b'f' | b'l' | b'i' | b'b' => array(reader, kind)?,
        // An unknown type carries an unknown length, so there is no way to
        // skip it and stay in step with the stream.
        _ => return Err(FbxError::Corrupt),
    })
}

fn array(reader: &mut Reader<'_>, kind: u8) -> Result<Property, FbxError> {
    let count = reader.u32()? as usize;
    let encoding = reader.u32()?;
    let compressed_len = reader.u32()? as usize;
    let width: usize = match kind {
        b'd' | b'l' => 8,
        b'f' | b'i' => 4,
        _ => 1,
    };
    let data: Vec<u8> = if encoding == 0 {
        let n = count.checked_mul(width).ok_or(FbxError::Corrupt)?;
        reader.take(n)?.to_vec()
    } else {
        let raw = reader.take(compressed_len)?;
        super::super::deflate::inflate_zlib(raw).map_err(|_| FbxError::Corrupt)?
    };
    Ok(match kind {
        b'd' => Property::Doubles(
            data.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        b'f' => Property::Floats(
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        b'l' => Property::Longs(
            data.chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        b'i' => Property::Ints(
            data.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        _ => Property::Bools(data.iter().map(|&b| b != 0).collect()),
    })
}
