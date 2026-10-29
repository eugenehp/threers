//! Transparent HEVC: a 2-layer access unit (color base + `AUX_ALPHA` layer),
//! muxed into an Apple-compatible `.mov`/`.mp4` that QuickTime/Safari play.
//!
//! Both layers are coded with the working `I_PCM` path: the base is 4:2:0 color,
//! the alpha layer is 4:2:0 with the opacity in luma and constant (128) chroma —
//! exactly how Apple's `hevcWithAlpha` carries alpha (see [`super::alpha`]). The
//! [`alpha_channel_info`](super::alpha::alpha_channel_info_sei) SEI in the `hvcC`
//! tells the decoder to interpret layer 1's luma as alpha (0 transparent, 255
//! opaque), and the `almo` box marks the track as transparent.
//!
//! The multi-layer `VPS` (with `vps_extension`) is taken verbatim from Apple's
//! reference ([`APPLE_VPS_NAL_64X64`]). It works at **any** resolution: because
//! each layer's SPS is self-contained (`MultiLayerExtSpsFlag = 0`), the decoder
//! ignores the VPS `rep_format`, so the same VPS drives 64×64, 128×96, 1080p,
//! etc. Verified by decoding with AVFoundation (QuickTime/Safari's engine): the
//! alpha channel round-trips losslessly at multiple sizes.

use crate::hevc::encoder::{pad_plane, Yuv420Frame};
use crate::hevc::hvcc::{build_hvcc, HvccArray, HvccProfile};
use crate::hevc::nal::{nal_unit, NalUnitType};
use crate::hevc::params::{write_pps_ids, write_sps_id, HevcConfig, CTB_SIZE};
use crate::hevc::slice::{slice_segment_rbsp_pps, PaddedYuv};
use crate::hevc::{alpha_channel_info_sei, AlphaChannelInfo};
use crate::mp4::{mux_hevc, Mp4Params};

/// Apple's reference alpha-multilayer `VPS` NAL (2-byte header + EBSP), captured
/// verbatim from an `AVVideoCodecType.hevcWithAlpha` file. Declares 2 layers with
/// layer 1 = `AUX_ALPHA`.
///
/// Despite the `_64X64` name (the resolution of the source file), this VPS is
/// resolution-independent for decoding: the `rep_format` it carries is ignored
/// because both layers use self-contained SPSs, so it drives every size.
pub const APPLE_VPS_NAL_64X64: [u8; 41] = [
    0x40, 0x01, 0x0c, 0x11, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0xb0, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x03, 0x00, 0x1e, 0x15, 0xc1, 0xbf, 0x00, 0x08, 0x00, 0x08, 0x30, 0x28, 0x53, 0x80,
    0x50, 0x00, 0x20, 0x50, 0x0c, 0x18, 0xfc, 0x57, 0xa4,
];

/// Apple's `almo` alpha-mode box payload (4 bytes) that marks a track transparent.
pub const ALMO_PAYLOAD: [u8; 4] = [0x00, 0x00, 0x01, 0x02];

/// Level `general_level_idc` matching the reference VPS (level 1.0), kept
/// consistent across VPS/SPS/`hvcC`.
const LEVEL_IDC: u8 = 30;

/// Encodes transparent frames and muxes them into an Apple-compatible container.
pub struct TransparentEncoder {
    width: u32,
    height: u32,
    base_cfg: HevcConfig,
    alpha_cfg: HevcConfig,
    hvcc: Vec<u8>,
    samples: Vec<Vec<u8>>,
}

impl TransparentEncoder {
    /// New encoder using the reference 64×64 multi-layer VPS.
    pub fn new(width: u32, height: u32) -> Self {
        Self::with_vps(width, height, &APPLE_VPS_NAL_64X64)
    }

    /// New encoder with an explicit multi-layer VPS NAL (for experimenting with
    /// resolution generalization / a generated `vps_extension`).
    pub fn with_vps(width: u32, height: u32, vps_nal: &[u8]) -> Self {
        let base_cfg = HevcConfig::new(width, height).with_level(LEVEL_IDC);
        // Alpha layer signals full range so 0..255 luma is read as opacity 1:1.
        let alpha_cfg = HevcConfig::new(width, height)
            .with_level(LEVEL_IDC)
            .with_full_range(true);

        // Parameter-set NALs. Base layer id 0 / sps,pps id 0; alpha layer id 1 /
        // sps,pps id 1.
        let sps0 = nal_unit(NalUnitType::Sps, 0, 0, &write_sps_id(&base_cfg, 0));
        let sps1 = nal_unit(NalUnitType::Sps, 1, 0, &write_sps_id(&alpha_cfg, 1));
        let pps0 = nal_unit(NalUnitType::Pps, 0, 0, &write_pps_ids(0, 0));
        let pps1 = nal_unit(NalUnitType::Pps, 1, 0, &write_pps_ids(1, 1));
        let sei = nal_unit(
            NalUnitType::PrefixSei,
            0,
            0,
            &alpha_channel_info_sei(&AlphaChannelInfo::default()),
        );
        let vps = vps_nal.to_vec();

        let hvcc = build_hvcc(
            &HvccProfile::main_420(LEVEL_IDC),
            &[
                HvccArray {
                    nal_type: 32,
                    complete: true,
                    nals: &[vps],
                },
                HvccArray {
                    nal_type: 33,
                    complete: true,
                    nals: &[sps0, sps1],
                },
                HvccArray {
                    nal_type: 34,
                    complete: true,
                    nals: &[pps0, pps1],
                },
                HvccArray {
                    nal_type: 39,
                    complete: false,
                    nals: &[sei],
                },
            ],
        );

        Self {
            width,
            height,
            base_cfg,
            alpha_cfg,
            hvcc,
            samples: Vec::new(),
        }
    }

    /// Append one frame: `color` (4:2:0) plus a full-resolution `alpha` plane
    /// (`width × height`, 0 = transparent, 255 = opaque).
    pub fn encode_frame(&mut self, color: &Yuv420Frame, alpha: &[u8]) {
        assert_eq!(
            (color.width, color.height),
            (self.width, self.height),
            "color size"
        );
        assert_eq!(
            alpha.len(),
            (self.width * self.height) as usize,
            "alpha size"
        );

        let base_slice =
            self.encode_layer_slice(&self.base_cfg, &color.y, &color.u, &color.v, 0, 0);
        // Alpha layer: opacity in luma, constant chroma.
        let (cw, ch) = (self.width / 2, self.height / 2);
        let chroma = vec![128u8; (cw * ch) as usize];
        let alpha_slice = self.encode_layer_slice(&self.alpha_cfg, alpha, &chroma, &chroma, 1, 1);

        // One MP4 sample = both layers' slices, 4-byte length-prefixed.
        let mut sample = Vec::with_capacity(8 + base_slice.len() + alpha_slice.len());
        push_len_prefixed(&mut sample, &base_slice);
        push_len_prefixed(&mut sample, &alpha_slice);
        self.samples.push(sample);
    }

    fn encode_layer_slice(
        &self,
        cfg: &HevcConfig,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        layer_id: u8,
        pps_id: u32,
    ) -> Vec<u8> {
        let (cw, ch) = (cfg.coded_width, cfg.coded_height);
        let yp = pad_plane(y, self.width, self.height, cw, ch);
        let up = pad_plane(u, self.width / 2, self.height / 2, cw / 2, ch / 2);
        let vp = pad_plane(v, self.width / 2, self.height / 2, cw / 2, ch / 2);
        let padded = PaddedYuv {
            y: &yp,
            u: &up,
            v: &vp,
            coded_width: cw,
            coded_height: ch,
        };
        let rbsp = slice_segment_rbsp_pps(cfg, &padded, pps_id);
        nal_unit(NalUnitType::IdrNLp, layer_id, 0, &rbsp)
    }

    /// Finish, muxing all frames into an Apple-compatible `.mov` byte stream at
    /// `fps` frames per second.
    pub fn finish_mov(&self, fps: u32) -> Vec<u8> {
        debug_assert_eq!(CTB_SIZE, 16);
        let timescale = 600u32;
        mux_hevc(&Mp4Params {
            width: self.width,
            height: self.height,
            timescale,
            frame_duration: timescale / fps.max(1),
            hvcc_payload: &self.hvcc,
            almo_payload: Some(&ALMO_PAYLOAD),
            samples: &self.samples,
        })
    }
}

/// Append `nal` to `out` with a 4-byte big-endian length prefix (MP4 form).
fn push_len_prefixed(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
    out.extend_from_slice(nal);
}
