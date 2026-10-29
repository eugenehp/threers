//! Transparent HEVC via an auxiliary alpha layer (H.265 Annex F) — the
//! Apple/QuickTime/Safari format.
//!
//! # What a transparent HEVC access unit looks like
//!
//! Reverse-engineered byte-for-byte from an AVFoundation `hevcWithAlpha` file
//! (see `tests`/scratch tooling). One frame is a base **color** layer plus an
//! independent **alpha** layer, both self-contained:
//!
//! | NAL | `nuh_layer_id` | role |
//! |-----|---------------|------|
//! | VPS | 0 | 2 layers; `vps_extension` marks layer 1 as `AUX_ALPHA` |
//! | SPS | 0 | base color (Main, 4:2:0) |
//! | SPS | 1 | alpha layer — **self-contained** (`MultiLayerExtSpsFlag = 0`) |
//! | PPS | 0 | base |
//! | PPS | 1 | alpha |
//! | SEI (prefix) | 0 | [`alpha_channel_info`](alpha_channel_info_sei) (payloadType 165) |
//! | slice | 0 | base color picture |
//! | slice | 1 | alpha picture |
//!
//! Key facts established from the reference:
//! - `vps_extension`: `vps_max_layers_minus1 = 1`, `scalability_mask[3] = 1`
//!   (AUXILIARY), layer 1 `dimension_id = 1` ⇒ `AuxId = 1` (`AUX_ALPHA`), and
//!   `direct_dependency_flag[1][0] = 0` — the alpha layer is coded independently
//!   of color (no inter-layer prediction).
//! - The alpha SPS begins `01 01 60 …`: `sps_ext_or_max_sub_layers_minus1 = 0`,
//!   so it carries its own profile/chroma/resolution rather than inheriting a
//!   VPS rep_format. That means each layer can be a normal self-contained SPS.
//! - The PPSs and the alpha SEI are resolution-independent; the VPS is not (its
//!   `rep_format` + `dpb_size` vary), so a general encoder must *generate* the
//!   `vps_extension`, not template it.
//!
//! # Status
//!
//! Done + verified here: the [`alpha_channel_info`](alpha_channel_info_sei) SEI
//! (byte-exact to Apple) and the per-layer coding — the base color layer and the
//! monochrome alpha layer each decode losslessly standalone (see
//! `HevcEncoder::new_monochrome` and `tests/hevc_ffmpeg.rs`).
//!
//! Remaining for a QuickTime-playable file: a `vps_extension` + `dpb_size`
//! generator (multi-layer, `AUX_ALPHA`) and an ISOBMFF/MP4 muxer with the alpha
//! `hvcC`. Those are tracked in the crate roadmap.

use crate::bitstream::{rbsp_trailing_bits, BitWriter};

/// Reference `pic_parameter_set_rbsp` bytes captured verbatim from Apple's
/// encoder (resolution-independent). Kept for cross-checking a generated PPS.
pub const APPLE_REF_PPS_BASE: [u8; 5] = [0xc0, 0x2c, 0xbd, 0x14, 0xd9];
/// Reference alpha-layer PPS bytes from Apple's encoder (resolution-independent).
pub const APPLE_REF_PPS_ALPHA: [u8; 6] = [0x48, 0x02, 0x8b, 0x1a, 0x29, 0xb2];
/// Reference `alpha_channel_info` SEI RBSP (payloadType 165) from Apple: use_idc
/// 1, 8-bit, transparent 0, opaque 255. [`alpha_channel_info_sei`] reproduces it.
pub const APPLE_REF_ALPHA_SEI: [u8; 7] = [0xa5, 0x04, 0x10, 0x00, 0x7f, 0x90, 0x80];

/// How the auxiliary layer's samples map to opacity (`alpha_channel_info`, F.14.2).
#[derive(Clone, Copy, Debug)]
pub struct AlphaChannelInfo {
    /// `alpha_channel_use_idc` (1 = alpha, per Apple).
    pub use_idc: u8,
    /// Sample value that is fully transparent.
    pub transparent_value: u16,
    /// Sample value that is fully opaque.
    pub opaque_value: u16,
}

impl Default for AlphaChannelInfo {
    fn default() -> Self {
        // Matches Apple's reference: 8-bit straight alpha, 0..255.
        Self {
            use_idc: 1,
            transparent_value: 0,
            opaque_value: 255,
        }
    }
}

/// Build the `alpha_channel_info` SEI RBSP (H.265 F.14.2), the prefix SEI that
/// tells the decoder to treat the auxiliary layer as alpha. 8-bit alpha only.
///
/// With [`AlphaChannelInfo::default`] this is byte-identical to
/// [`APPLE_REF_ALPHA_SEI`].
pub fn alpha_channel_info_sei(info: &AlphaChannelInfo) -> Vec<u8> {
    // --- sei_payload: alpha_channel_info(), 8-bit (bit_depth_minus8 = 0) ---
    let mut p = BitWriter::new();
    p.flag(false); // alpha_channel_cancel_flag
    p.write_bits(info.use_idc as u32 & 0x7, 3); // alpha_channel_use_idc
    p.write_bits(0, 3); // alpha_channel_bit_depth_minus8 (8-bit)
    p.write_bits(info.transparent_value as u32, 9); // alpha_transparent_value u(v), v = 9
    p.write_bits(info.opaque_value as u32, 9); // alpha_opaque_value u(v)
    p.flag(false); // alpha_channel_incr_flag
    p.flag(false); // alpha_channel_clip_flag (⇒ no clip_type_flag)
                   // sei_payload trailing: a 1 bit then zero-fill to a byte (payload_bit_equal_to_one).
    p.write_bit(1);
    let payload = p.finish();

    // --- sei_message wrapper + sei_rbsp trailing ---
    let mut w = BitWriter::new();
    w.write_byte(165); // last_payload_type_byte = 165 (alpha_channel_info)
    debug_assert!(payload.len() < 255);
    w.write_byte(payload.len() as u8); // last_payload_size_byte
    for &b in &payload {
        w.write_byte(b);
    }
    rbsp_trailing_bits(&mut w);
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_sei_matches_apple_reference() {
        let sei = alpha_channel_info_sei(&AlphaChannelInfo::default());
        assert_eq!(
            sei,
            APPLE_REF_ALPHA_SEI.to_vec(),
            "generated alpha SEI must match Apple's"
        );
    }

    #[test]
    fn alpha_sei_fields_roundtrip_size() {
        // A different (still 8-bit) config stays a valid 4-byte payload SEI.
        let sei = alpha_channel_info_sei(&AlphaChannelInfo {
            use_idc: 1,
            transparent_value: 0,
            opaque_value: 255,
        });
        assert_eq!(sei[0], 0xa5); // payloadType 165
        assert_eq!(sei[1], 0x04); // payloadSize 4
        assert_eq!(*sei.last().unwrap(), 0x80); // rbsp stop
    }
}
