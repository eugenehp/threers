//! Bit-level writing for sequence and frame headers.

#[derive(Default)]
pub(super) struct BitWriter {
    out: Vec<u8>,
    bits: u32,
}

impl BitWriter {
    pub(super) fn put(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            if self.bits.is_multiple_of(8) {
                self.out.push(0);
            }
            let bit = ((value >> i) & 1) as u8;
            *self.out.last_mut().unwrap() |= bit << (7 - self.bits % 8);
            self.bits += 1;
        }
    }

    pub(super) fn flag(&mut self, value: bool) {
        self.put(value as u32, 1);
    }

    /// `trailing_bits`: a one, then zeros to the byte boundary.
    pub(super) fn trailing(mut self) -> Vec<u8> {
        self.put(1, 1);
        self.out
    }

    /// `byte_alignment`: zeros to the byte boundary.
    pub(super) fn aligned(self) -> Vec<u8> {
        self.out
    }
}

/// An OBU with its size field.
pub(super) fn obu(kind: u8, payload: &[u8], out: &mut Vec<u8>) {
    out.push(kind << 3 | 2);
    let mut size = payload.len();
    loop {
        let byte = (size & 0x7f) as u8;
        size >>= 7;
        out.push(if size > 0 { byte | 0x80 } else { byte });
        if size == 0 {
            break;
        }
    }
    out.extend_from_slice(payload);
}
