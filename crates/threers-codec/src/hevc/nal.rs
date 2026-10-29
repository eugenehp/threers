//! HEVC NAL units and Annex-B byte-stream packaging (H.265 §7.3.1).
//!
//! A NAL unit is a 2-byte header plus an EBSP payload (the RBSP with
//! emulation-prevention bytes spliced in). The Annex-B byte stream every raw
//! `.265`/`.hevc` file uses just concatenates NAL units, each preceded by a
//! `00 00 00 01` start code.

use crate::bitstream::emulation_prevention;

/// NAL unit type (`nal_unit_type`, H.265 Table 7-1). Only the values the encoder
/// emits are enumerated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NalUnitType {
    /// Coded slice of a trailing, non-reference picture.
    TrailN = 0,
    /// Coded slice of a trailing, reference picture.
    TrailR = 1,
    /// Coded slice of an IDR picture that may have associated RADL pictures.
    IdrWRadl = 19,
    /// Coded slice of an IDR picture with no leading pictures.
    IdrNLp = 20,
    /// Coded slice of a CRA (clean random access) picture.
    CraNut = 21,
    /// Video parameter set.
    Vps = 32,
    /// Sequence parameter set.
    Sps = 33,
    /// Picture parameter set.
    Pps = 34,
    /// Access-unit delimiter.
    Aud = 35,
    /// Prefix SEI.
    PrefixSei = 39,
}

impl NalUnitType {
    /// The raw 6-bit `nal_unit_type` code.
    pub fn code(self) -> u8 {
        self as u8
    }

    /// Whether this is an IRAP (intra random-access point) picture type — the
    /// key-frame class a decoder can start from.
    pub fn is_irap(self) -> bool {
        matches!(
            self,
            NalUnitType::IdrWRadl | NalUnitType::IdrNLp | NalUnitType::CraNut
        )
    }
}

/// The 4-byte Annex-B start code prefixing a NAL unit at the start of an AU.
pub const START_CODE: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

/// Build a NAL unit: 2-byte header + emulation-prevented `rbsp`. `rbsp` must be
/// byte-aligned (end it with `rbsp_trailing_bits`).
///
/// The header packs `forbidden_zero_bit(0) | nal_unit_type(6) | nuh_layer_id(6)
/// | nuh_temporal_id_plus1(3)` into 16 bits.
pub fn nal_unit(nal_type: NalUnitType, layer_id: u8, temporal_id: u8, rbsp: &[u8]) -> Vec<u8> {
    let t = nal_type.code() & 0x3F;
    let layer = layer_id & 0x3F;
    let tid1 = (temporal_id & 0x07) + 1;
    let b0 = (t << 1) | (layer >> 5);
    let b1 = ((layer & 0x1F) << 3) | tid1;

    let ebsp = emulation_prevention(rbsp);
    let mut out = Vec::with_capacity(2 + ebsp.len());
    out.push(b0);
    out.push(b1);
    out.extend_from_slice(&ebsp);
    out
}

/// Build a NAL unit for layer 0 / temporal-id 0 (the common case).
pub fn nal_unit_base(nal_type: NalUnitType, rbsp: &[u8]) -> Vec<u8> {
    nal_unit(nal_type, 0, 0, rbsp)
}

/// Append `nal` to an Annex-B byte stream `out`, prefixed by a start code.
pub fn push_annexb(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&START_CODE);
    out.extend_from_slice(nal);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_bytes_base_layer() {
        // VPS (32), layer 0, temporal_id 0 → header 0x40 0x01.
        let nal = nal_unit_base(NalUnitType::Vps, &[0xAA]);
        assert_eq!(nal[0], 0x40, "nal_unit_type 32 in bits 1..7");
        assert_eq!(nal[1], 0x01, "layer 0, temporal_id_plus1 = 1");
        assert_eq!(&nal[2..], &[0xAA]);

        // SPS (33) → 0x42 0x01.
        assert_eq!(nal_unit_base(NalUnitType::Sps, &[])[0], 0x42);
        // PPS (34) → 0x44 0x01.
        assert_eq!(nal_unit_base(NalUnitType::Pps, &[])[0], 0x44);
    }

    #[test]
    fn payload_gets_emulation_prevention() {
        // RBSP containing a start-code-like run must be stuffed.
        let nal = nal_unit_base(NalUnitType::Pps, &[0x00, 0x00, 0x01, 0xFF]);
        assert_eq!(&nal[2..], &[0x00, 0x00, 0x03, 0x01, 0xFF]);
    }

    #[test]
    fn irap_classification() {
        assert!(NalUnitType::IdrWRadl.is_irap());
        assert!(NalUnitType::CraNut.is_irap());
        assert!(!NalUnitType::TrailR.is_irap());
        assert!(!NalUnitType::Vps.is_irap());
    }

    #[test]
    fn annexb_prefixes_start_code() {
        let mut stream = Vec::new();
        push_annexb(&mut stream, &[0x42, 0x01, 0x99]);
        assert_eq!(stream, vec![0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x99]);
    }
}
