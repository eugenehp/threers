//! Ogawa — the container Alembic writes.
//!
//! A tree of two node kinds and nothing else:
//!
//! - a **group** is a 64-bit child count followed by that many 64-bit
//!   references;
//! - a **data** block is a 64-bit size followed by that many bytes.
//!
//! A reference's top bit says which it points at, and the remaining 63 bits are
//! an absolute file offset. Zero means empty. That is the whole format — the
//! structure above it (objects, properties, samples) is convention about which
//! child means what, and lives in the module above this one.

use super::AlembicError;

pub struct Ogawa<'a> {
    pub data: &'a [u8],
    pub root: usize,
    pub version: u16,
}

/// What a group's child points at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Child {
    Group(usize),
    Data(usize),
    Empty,
}

impl<'a> Ogawa<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self, AlembicError> {
        // "Ogawa", a byte that is 0xff once the file is complete, a version,
        // then the root group's offset.
        if data.len() < 16 || &data[..5] != b"Ogawa" {
            return Err(AlembicError::NotAlembic);
        }
        // A writer that died mid-stream leaves this byte clear, and the offsets
        // it wrote cannot be trusted.
        if data[5] != 0xff {
            return Err(AlembicError::Incomplete);
        }
        let version = u16::from_be_bytes([data[6], data[7]]);
        let root = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        if root >= data.len() {
            return Err(AlembicError::Corrupt);
        }
        Ok(Ogawa {
            data,
            root,
            version,
        })
    }

    fn u64_at(&self, at: usize) -> Result<u64, AlembicError> {
        let b = self
            .data
            .get(at..at + 8)
            .ok_or(AlembicError::Truncated)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    /// A group's children.
    pub fn group(&self, at: usize) -> Result<Vec<Child>, AlembicError> {
        let count = self.u64_at(at)? as usize;
        // Each child is eight bytes, so a count that could not fit in the file
        // is corruption rather than a very wide group.
        if count.saturating_mul(8) > self.data.len() {
            return Err(AlembicError::Corrupt);
        }
        let mut out = Vec::with_capacity(count.min(4096));
        for i in 0..count {
            let reference = self.u64_at(at + 8 + i * 8)?;
            let offset = (reference & !(1u64 << 63)) as usize;
            out.push(if offset == 0 {
                Child::Empty
            } else if reference >> 63 != 0 {
                Child::Data(offset)
            } else {
                Child::Group(offset)
            });
        }
        Ok(out)
    }

    /// A data block's bytes.
    pub fn data_at(&self, at: usize) -> Result<&'a [u8], AlembicError> {
        let size = self.u64_at(at)? as usize;
        self.data
            .get(at + 8..at + 8 + size)
            .ok_or(AlembicError::Truncated)
    }

    /// The bytes of a group's child, when it is data.
    pub fn child_data(&self, children: &[Child], index: usize) -> Option<&'a [u8]> {
        match children.get(index) {
            Some(Child::Data(at)) => self.data_at(*at).ok(),
            _ => None,
        }
    }
}
