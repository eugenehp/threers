//! Bounds-checked big-endian reads, the same shape `swiftui_font::read` has.
//!
//! Duplicated rather than imported because that module is this crate's dependency's
//! internals; four functions is cheaper than widening someone else's public surface.

/// A big-endian `u16` at `off`.
#[must_use]
pub fn u16_at(d: &[u8], off: usize) -> Option<u16> {
    let b = d.get(off..off.checked_add(2)?)?;
    Some(u16::from_be_bytes([b[0], b[1]]))
}

/// A big-endian `u32` at `off`.
#[must_use]
pub fn u32_at(d: &[u8], off: usize) -> Option<u32> {
    let b = d.get(off..off.checked_add(4)?)?;
    Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// A four-character tag at `off`.
#[must_use]
pub fn tag_at(d: &[u8], off: usize) -> Option<[u8; 4]> {
    let b = d.get(off..off.checked_add(4)?)?;
    Some([b[0], b[1], b[2], b[3]])
}
