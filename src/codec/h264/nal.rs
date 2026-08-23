//! H.264 / AVC NAL units and Annex-B byte-stream packaging (ISO/IEC 14496-10 §7.3.1).
//!
//! An AVC NAL unit is a 1-byte header plus an EBSP payload. Annex-B streams
//! prefix each NAL with a `00 00 00 01` start code.

use crate::codec::bitstream::emulation_prevention;

/// NAL unit type (`nal_unit_type`, Table 7-1). Only values this encoder emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NalUnitType {
    /// Coded slice of an IDR picture.
    Idr = 5,
    /// Sequence parameter set.
    Sps = 7,
    /// Picture parameter set.
    Pps = 8,
}

impl NalUnitType {
    pub fn code(self) -> u8 {
        self as u8
    }
}

/// Annex-B start code prefix.
pub const START_CODE: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

/// Build a NAL unit: 1-byte header + emulation-prevented RBSP.
///
/// `nal_ref_idc` is typically `3` for parameter sets and IDR slices.
pub fn nal_unit(nal_type: NalUnitType, nal_ref_idc: u8, rbsp: &[u8]) -> Vec<u8> {
    let t = nal_type.code() & 0x1F;
    let ref_idc = nal_ref_idc & 0x03;
    let header = (ref_idc << 5) | t;
    let ebsp = emulation_prevention(rbsp);
    let mut out = Vec::with_capacity(1 + ebsp.len());
    out.push(header);
    out.extend_from_slice(&ebsp);
    out
}

/// Build a NAL with `nal_ref_idc = 3` (highest priority).
pub fn nal_unit_base(nal_type: NalUnitType, rbsp: &[u8]) -> Vec<u8> {
    nal_unit(nal_type, 3, rbsp)
}

/// Append `nal` to an Annex-B byte stream.
pub fn push_annexb(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&START_CODE);
    out.extend_from_slice(nal);
}

/// Split an Annex-B stream into individual NAL units (headers included).
pub fn split_annexb(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            starts.push((i, 3));
            i += 3;
        } else if i + 4 <= stream.len()
            && stream[i] == 0
            && stream[i + 1] == 0
            && stream[i + 2] == 0
            && stream[i + 3] == 1
        {
            starts.push((i, 4));
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::new();
    for (n, &(off, len)) in starts.iter().enumerate() {
        let end = starts.get(n + 1).map(|&(o, _)| o).unwrap_or(stream.len());
        nals.push(stream[off + len..end].to_vec());
    }
    nals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_bytes() {
        assert_eq!(nal_unit_base(NalUnitType::Sps, &[])[0], 0x67);
        assert_eq!(nal_unit_base(NalUnitType::Pps, &[])[0], 0x68);
        assert_eq!(nal_unit_base(NalUnitType::Idr, &[])[0], 0x65);
    }

    #[test]
    fn payload_gets_emulation_prevention() {
        let nal = nal_unit_base(NalUnitType::Pps, &[0x00, 0x00, 0x01, 0xFF]);
        assert_eq!(&nal[1..], &[0x00, 0x00, 0x03, 0x01, 0xFF]);
    }

    #[test]
    fn annexb_and_split_roundtrip() {
        let nals = [
            nal_unit_base(NalUnitType::Sps, &[0xAA]),
            nal_unit_base(NalUnitType::Pps, &[0xBB]),
        ];
        let mut stream = Vec::new();
        for nal in &nals {
            push_annexb(&mut stream, nal);
        }
        assert_eq!(split_annexb(&stream), nals);
    }
}
