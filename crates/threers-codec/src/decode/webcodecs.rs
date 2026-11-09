//! What a platform decoder needs from a [`Demuxed`] track that the container
//! does not spell out: the browser's WebCodecs `VideoDecoder` takes a codec
//! string (`avc1.640028`, `hvc1.1.6.L120.90`, `av01.0.08M.08`,
//! `vp09.00.41.08`) and the configuration record as its `description`, and
//! each chunk marked key or delta.
//!
//! All of it is read from the configuration record and the coded samples, so
//! it is plain Rust, tested natively, and used by threers's wasm player to
//! hand the browser's hardware decoder the same files the native decoders
//! read.
//!
//! ```no_run
//! use threers_codec::decode::{demux, webcodecs};
//!
//! let track = demux(std::fs::read("clip.mp4").unwrap()).unwrap();
//! let codec = webcodecs::codec_string(&track).unwrap();
//! let first_is_key = webcodecs::is_keyframe(&track, 0);
//! # let _ = (codec, first_is_key);
//! ```

use super::reader::{vp9_record, Demuxed};
use crate::mp4_demux::TrackCodec;

/// The RFC 6381 / WebCodecs codec string for `track`, from its configuration
/// record (or, for VP9 without one, its first frame). `None` when there is
/// not enough to say: AV1 in IVF, a damaged record.
pub fn codec_string(track: &Demuxed) -> Option<String> {
    let c = &track.config;
    match track.codec {
        TrackCodec::H264 => {
            // avcC: version, profile, compatibility, level.
            let b = c.get(1..4)?;
            Some(format!("avc1.{:02x}{:02x}{:02x}", b[0], b[1], b[2]))
        }
        TrackCodec::Hevc => hevc_string(c),
        TrackCodec::Av1 if c.len() < 4 => {
            // No av1C (IVF, some Matroska): the sequence header the first
            // temporal unit carries says the same.
            let seq = av1_sequence_header(track.samples.first().map(|_| track.sample(0))?)?;
            let op = seq.operating_points.first()?;
            let level = (op.major_level.saturating_sub(2)) * 4 + op.minor_level;
            let tier = if op.tier != 0 { 'H' } else { 'M' };
            let depth = [8, 10, 12][seq.hbd.min(2) as usize];
            Some(format!("av01.{}.{level:02}{tier}.{depth:02}", seq.profile))
        }
        TrackCodec::Av1 => {
            // av1C: marker and version, then seq_profile (3) and
            // seq_level_idx_0 (5), then tier, high_bitdepth, twelve_bit.
            let (a, b) = (*c.get(1)?, *c.get(2)?);
            let (profile, level) = (a >> 5, a & 0x1f);
            let tier = if b & 0x80 != 0 { 'H' } else { 'M' };
            let depth = if b & 0x40 == 0 {
                8
            } else if b & 0x20 != 0 {
                12
            } else {
                10
            };
            Some(format!("av01.{profile}.{level:02}{tier}.{depth:02}"))
        }
        TrackCodec::Vp9 => {
            let record = if c.len() >= 7 {
                c.clone()
            } else {
                vp9_record(
                    track.samples.first().map(|_| track.sample(0))?,
                    track.width,
                    track.height,
                )?
            };
            // vpcC: version and flags, profile, level, bit depth (4 bits).
            Some(format!(
                "vp09.{:02}.{:02}.{:02}",
                record[4],
                record[5],
                record[6] >> 4
            ))
        }
    }
}

/// `hvc1.<space><profile>.<compatibility>.<tier><level>.<constraints>`, as
/// ISO/IEC 14496-15 Annex E writes it.
fn hevc_string(c: &[u8]) -> Option<String> {
    let b = c.get(1..13)?;
    let space = ["", "A", "B", "C"][(b[0] >> 6) as usize];
    let tier = if b[0] & 0x20 != 0 { 'H' } else { 'L' };
    let profile = b[0] & 0x1f;
    // The 32 compatibility flags, in reverse bit order, as hex.
    let compat = u32::from_be_bytes([b[1], b[2], b[3], b[4]]).reverse_bits();
    let level = b[11];
    let mut s = format!("hvc1.{space}{profile}.{compat:x}.{tier}{level}");
    // The six constraint bytes, trailing zero bytes left off.
    let constraints = &b[5..11];
    let used = constraints
        .iter()
        .rposition(|&x| x != 0)
        .map_or(0, |i| i + 1);
    for x in &constraints[..used] {
        s.push_str(&format!(".{x:02x}"));
    }
    Some(s)
}

/// The `description` a WebCodecs `VideoDecoder` is configured with: the
/// `avcC` or `hvcC` record for H.264 and HEVC, which arrive length-prefixed
/// and need it; the `av1C` for AV1 where there is one; nothing for VP9.
pub fn description(track: &Demuxed) -> Option<&[u8]> {
    match track.codec {
        TrackCodec::H264 | TrackCodec::Hevc | TrackCodec::Av1 if !track.config.is_empty() => {
            Some(&track.config)
        }
        _ => None,
    }
}

/// Whether sample `i` can be decoded on its own: an IDR picture (H.264), an
/// IRAP one (HEVC), a key frame (VP9), or a temporal unit opening with a
/// sequence header or a key frame (AV1). WebCodecs wants the first chunk after
/// a configure or reset to be one, and a decoder can start at any.
pub fn is_keyframe(track: &Demuxed, i: usize) -> bool {
    let Some(sample) = track.samples.get(i).map(|_| track.sample(i)) else {
        return false;
    };
    match track.codec {
        TrackCodec::H264 => {
            let len = track.config.get(4).map_or(4, |b| (b & 3) as usize + 1);
            nal_types(sample, len).any(|h| h & 0x1f == 5)
        }
        TrackCodec::Hevc => {
            let len = track.config.get(21).map_or(4, |b| (b & 3) as usize + 1);
            nal_types(sample, len).any(|h| (16..=23).contains(&((h >> 1) & 0x3f)))
        }
        TrackCodec::Vp9 => vp9_key(sample),
        TrackCodec::Av1 => av1_key(sample),
    }
}

/// The first byte of each length-prefixed NAL unit in `sample`.
fn nal_types(sample: &[u8], len: usize) -> impl Iterator<Item = u8> + '_ {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        let size = sample
            .get(at..at + len)?
            .iter()
            .fold(0usize, |a, &b| a << 8 | b as usize);
        let header = *sample.get(at + len)?;
        at += len + size;
        Some(header)
    })
}

/// VP9's uncompressed header: marker, profile, then `show_existing_frame`
/// and `frame_type` (0 is a key frame).
fn vp9_key(frame: &[u8]) -> bool {
    let Some(&b) = frame.first() else {
        return false;
    };
    if b >> 6 != 0b10 {
        return false;
    }
    let profile = ((b >> 5) & 1) | ((b >> 4) & 1) << 1;
    // Profile 3 has a reserved bit before the rest.
    let at = if profile == 3 { 3 } else { 2 };
    let bit = |n: u32| (b >> (5 - at as u32 - n)) & 1;
    bit(0) == 0 && bit(1) == 0
}

/// The OBUs of an AV1 temporal unit: `(type, payload)`.
fn obus(unit: &[u8]) -> impl Iterator<Item = (u8, &[u8])> + '_ {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        let header = *unit.get(at)?;
        let kind = (header >> 3) & 0xf;
        at += 1 + (header & 0x04 != 0) as usize;
        let size = if header & 0x02 != 0 {
            let mut value = 0usize;
            let mut shift = 0;
            loop {
                let byte = *unit.get(at)?;
                at += 1;
                value |= ((byte & 0x7f) as usize) << shift;
                shift += 7;
                if byte & 0x80 == 0 || shift > 56 {
                    break;
                }
            }
            value
        } else {
            unit.len().saturating_sub(at)
        };
        let payload = unit.get(at..at.checked_add(size)?)?;
        at += size;
        Some((kind, payload))
    })
}

/// The sequence header in an AV1 temporal unit, parsed by the decoder's own
/// parser.
fn av1_sequence_header(unit: &[u8]) -> Option<super::av1::obu::SeqHeader> {
    let (_, payload) = obus(unit).find(|(kind, _)| *kind == 1)?;
    let mut bits = super::av1::obu::GetBits::new(payload);
    super::av1::obu::parse_seq_hdr(&mut bits).ok()
}

/// An AV1 temporal unit starts a sequence when it carries a sequence header,
/// which encoders put before every key frame, or a frame header saying
/// `KEY_FRAME`.
fn av1_key(unit: &[u8]) -> bool {
    for (kind, payload) in obus(unit) {
        match kind {
            // OBU_SEQUENCE_HEADER.
            1 => return true,
            // OBU_FRAME_HEADER, OBU_FRAME: show_existing_frame, frame_type.
            3 | 6 => {
                let Some(&b) = payload.first() else {
                    return false;
                };
                return b & 0x80 == 0 && (b >> 5) & 0b11 == 0;
            }
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(codec: TrackCodec, config: Vec<u8>, samples: &[&[u8]]) -> Demuxed {
        let mut bytes = Vec::new();
        let mut list = Vec::new();
        for (i, s) in samples.iter().enumerate() {
            let start = bytes.len();
            bytes.extend_from_slice(s);
            list.push(super::super::reader::CodedSample {
                start,
                end: bytes.len(),
                pts: i as i64,
                dts: i as i64,
                duration: 1,
                seconds: i as f64 / 30.0,
            });
        }
        Demuxed {
            codec,
            config,
            width: 1920,
            height: 1080,
            bytes: bytes.into(),
            samples: list,
        }
    }

    #[test]
    fn codec_strings_are_read_from_the_records() {
        // High profile, level 4.0.
        let avc = track(TrackCodec::H264, vec![1, 0x64, 0x00, 0x28, 0xff], &[]);
        assert_eq!(codec_string(&avc).unwrap(), "avc1.640028");
        // Main, compatible with Main and Main10, main tier, level 4 (120),
        // the progressive-source constraint (0x90 = progressive + frame only).
        let mut hvcc = vec![1, 0x01, 0x60, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 120];
        hvcc.resize(23, 0);
        let hevc = track(TrackCodec::Hevc, hvcc, &[]);
        assert_eq!(codec_string(&hevc).unwrap(), "hvc1.1.6.L120.90");
        // Main profile, level 3.0 (8), main tier, 8-bit; and 10-bit high tier.
        let av1 = track(TrackCodec::Av1, vec![0x81, 0x08, 0x0c, 0], &[]);
        assert_eq!(codec_string(&av1).unwrap(), "av01.0.08M.08");
        let av1 = track(TrackCodec::Av1, vec![0x81, 0x2d, 0xc0, 0], &[]);
        assert_eq!(codec_string(&av1).unwrap(), "av01.1.13H.10");
        // VP9 from a vpcC, and from a profile 0 key frame without one.
        let vp9 = track(
            TrackCodec::Vp9,
            vec![1, 0, 0, 0, 2, 31, 0xa2, 2, 2, 2, 0, 0],
            &[],
        );
        assert_eq!(codec_string(&vp9).unwrap(), "vp09.02.31.10");
        let vp9 = track(TrackCodec::Vp9, Vec::new(), &[&[0x82, 0x49, 0x83, 0x42]]);
        assert_eq!(codec_string(&vp9).unwrap(), "vp09.00.41.08");
        // AV1 with no record has nothing to go on.
        assert!(codec_string(&track(TrackCodec::Av1, Vec::new(), &[])).is_none());
    }

    #[test]
    fn key_frames_are_found_in_each_codec() {
        let avcc = vec![1, 0x64, 0, 0x28, 0xff];
        // An SEI then an IDR slice; then a non-IDR slice.
        let h264 = track(
            TrackCodec::H264,
            avcc,
            &[
                &[0, 0, 0, 2, 0x06, 0x05, 0, 0, 0, 2, 0x65, 0x88],
                &[0, 0, 0, 2, 0x41, 0x9a],
            ],
        );
        assert!(is_keyframe(&h264, 0));
        assert!(!is_keyframe(&h264, 1));

        let mut hvcc = vec![0u8; 23];
        hvcc[21] = 0xff;
        // IDR_W_RADL (19), then TRAIL_R (1).
        let hevc = track(
            TrackCodec::Hevc,
            hvcc,
            &[&[0, 0, 0, 2, 19 << 1, 1], &[0, 0, 0, 2, 1 << 1, 1]],
        );
        assert!(is_keyframe(&hevc, 0));
        assert!(!is_keyframe(&hevc, 1));

        // Profile 0: marker 10, profile 00, show_existing 0, frame_type 0/1.
        let vp9 = track(
            TrackCodec::Vp9,
            Vec::new(),
            &[&[0b1000_0000], &[0b1000_0100], &[0b1000_1000]],
        );
        assert!(is_keyframe(&vp9, 0));
        assert!(!is_keyframe(&vp9, 1), "inter frame");
        assert!(!is_keyframe(&vp9, 2), "show existing frame");

        // A temporal delimiter then a sequence header; then a temporal
        // delimiter and an inter OBU_FRAME (frame_type 1).
        let av1 = track(
            TrackCodec::Av1,
            Vec::new(),
            &[
                &[0x12, 0x00, 0x0a, 0x01, 0x00],
                &[0x12, 0x00, 0x32, 0x01, 0b0010_0000],
            ],
        );
        assert!(is_keyframe(&av1, 0));
        assert!(!is_keyframe(&av1, 1));
    }
}
