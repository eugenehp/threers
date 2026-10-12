//! Reading a Draco bitstream: bytes, varints, and a bit mode.
//!
//! Draco's `DecoderBuffer` alternates between byte-aligned reads and a
//! bit-packed mode, and the switch between them is part of the format — the
//! bit reader starts where the byte reader had got to, and when it ends the
//! byte position advances by *whole bytes*, rounding up. Getting that rounding
//! wrong leaves everything after it one byte out, which decodes to plausible
//! garbage rather than to an error.

use super::DracoError;

pub struct Buffer<'a> {
    data: &'a [u8],
    pub at: usize,
    /// Bit position within the bit-mode window, which begins at `at`.
    bit_offset: usize,
    bit_start: usize,
    in_bit_mode: bool,
}

impl<'a> Buffer<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Buffer {
            data,
            at: 0,
            bit_offset: 0,
            bit_start: 0,
            in_bit_mode: false,
        }
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.at)
    }

    pub fn head(&self) -> &'a [u8] {
        self.data.get(self.at..).unwrap_or(&[])
    }

    pub fn u8(&mut self) -> Result<u8, DracoError> {
        let v = *self.data.get(self.at).ok_or(DracoError::Truncated)?;
        self.at += 1;
        Ok(v)
    }

    pub fn i8(&mut self) -> Result<i8, DracoError> {
        Ok(self.u8()? as i8)
    }

    pub fn u16(&mut self) -> Result<u16, DracoError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn u32(&mut self) -> Result<u32, DracoError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn f32(&mut self) -> Result<f32, DracoError> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], DracoError> {
        let end = self.at.checked_add(n).ok_or(DracoError::Truncated)?;
        let slice = self.data.get(self.at..end).ok_or(DracoError::Truncated)?;
        self.at = end;
        Ok(slice)
    }

    pub fn advance(&mut self, n: usize) -> Result<(), DracoError> {
        if self.at + n > self.data.len() {
            return Err(DracoError::Truncated);
        }
        self.at += n;
        Ok(())
    }

    /// LEB128, low 7 bits first.
    pub fn varint(&mut self) -> Result<u64, DracoError> {
        let mut value: u64 = 0;
        let mut shift = 0;
        loop {
            let b = self.u8()?;
            value |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                return Err(DracoError::Corrupt);
            }
        }
    }

    pub fn varint32(&mut self) -> Result<u32, DracoError> {
        let v = self.varint()?;
        u32::try_from(v).map_err(|_| DracoError::Corrupt)
    }

    /// Begin reading individual bits from where the bytes had reached.
    pub fn start_bits(&mut self) {
        self.in_bit_mode = true;
        self.bit_start = self.at;
        self.bit_offset = 0;
    }

    /// One bit. Bits run least-significant-first within each byte.
    pub fn bit(&mut self) -> u32 {
        let byte = self.bit_start + (self.bit_offset >> 3);
        let shift = self.bit_offset & 7;
        self.bit_offset += 1;
        match self.data.get(byte) {
            Some(b) => ((b >> shift) & 1) as u32,
            None => 0,
        }
    }

    /// `n` bits, assembled least-significant-first.
    pub fn bits(&mut self, n: u32) -> u32 {
        let mut value = 0u32;
        for i in 0..n.min(32) {
            value |= self.bit() << i;
        }
        value
    }

    /// Leave bit mode, advancing the byte position by the whole bytes used.
    pub fn end_bits(&mut self) {
        self.in_bit_mode = false;
        self.at = self.bit_start + self.bit_offset.div_ceil(8);
    }

    pub fn in_bit_mode(&self) -> bool {
        self.in_bit_mode
    }
}
