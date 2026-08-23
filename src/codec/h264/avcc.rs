//! `AVCDecoderConfigurationRecord` (`avcC`) builder — ISO/IEC 14496-15 §5.2.4.1.
//!
//! Packs SPS/PPS NAL units (with their 1-byte headers) into the config record
//! an MP4 `avc1` sample entry carries.

/// Build the `avcC` box payload (without the 8-byte box header).
pub fn build_avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let profile = sps.get(1).copied().unwrap_or(100);
    let compat = sps.get(2).copied().unwrap_or(0);
    let level = sps.get(3).copied().unwrap_or(31);
    let mut v = vec![
        1,        // configurationVersion
        profile,  // AVCProfileIndication
        compat,   // profile_compatibility
        level,    // AVCLevelIndication
        0xFC | 3, // reserved(6) | lengthSizeMinusOne(2) = 3 → 4-byte lengths
        0xE1,     // reserved(3) | numOfSequenceParameterSets(5) = 1
    ];
    v.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    v.extend_from_slice(sps);
    v.push(1); // numOfPictureParameterSets
    v.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    v.extend_from_slice(pps);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::nal::nal_unit_base;
    use crate::codec::h264::params::{write_sps, H264Config};
    use crate::codec::h264::pps::write_pps;

    #[test]
    fn avcc_carries_parameter_sets() {
        let cfg = H264Config::new(64, 64);
        let sps = nal_unit_base(crate::codec::h264::nal::NalUnitType::Sps, &write_sps(&cfg));
        let pps = nal_unit_base(crate::codec::h264::nal::NalUnitType::Pps, &write_pps());
        let avcc = build_avcc(&sps, &pps);
        assert_eq!(avcc[0], 1);
        assert_eq!(avcc[1], 66); // Baseline profile
        assert!(avcc.len() > 8 + sps.len() + pps.len());
    }
}
