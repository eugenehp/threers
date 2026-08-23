//! Frame-level HEVC encoder: pixels in, Annex-B byte stream out.
//!
//! [`HevcEncoder`] ties the layers together — it emits VPS/SPS/PPS once, then an
//! IDR I-slice per frame (milestone 3: `I_PCM`, so lossless in the YUV domain,
//! no compression yet). Everything is pure Rust returning `Vec<u8>`, so it works
//! on native and `wasm32` alike; the caller decides where the bytes go.

use crate::codec::hevc::nal::{nal_unit_base, push_annexb, NalUnitType};
use crate::codec::hevc::params::{write_pps, write_sps, write_vps, HevcConfig, CTB_SIZE};
use crate::codec::hevc::slice::{slice_segment_rbsp, PaddedYuv};

/// A planar 4:2:0 (`yuv420p`) picture at display resolution. `width`/`height`
/// must be even (chroma is subsampled 2×2). `y` is `width × height`; `u`/`v` are
/// `(width/2) × (height/2)`.
///
/// `alpha` is an optional full-resolution (`width × height`) opacity plane. When
/// present it is carried through to a transparent encode (an HEVC auxiliary
/// alpha layer); `None` means fully opaque. It is populated automatically by
/// [`from_rgba`](Self::from_rgba) when the source RGBA is not fully opaque.
#[derive(Clone, Debug)]
pub struct Yuv420Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub alpha: Option<Vec<u8>>,
}

impl Yuv420Frame {
    /// Allocate an all-black (well, all-zero-plane), fully-opaque frame.
    pub fn new(width: u32, height: u32) -> Self {
        assert!(
            width.is_multiple_of(2) && height.is_multiple_of(2),
            "4:2:0 needs even dimensions"
        );
        let (cw, ch) = (width / 2, height / 2);
        Self {
            width,
            height,
            y: vec![0; (width * height) as usize],
            u: vec![128; (cw * ch) as usize],
            v: vec![128; (cw * ch) as usize],
            alpha: None,
        }
    }

    /// Whether this frame carries a non-trivial alpha plane.
    pub fn is_transparent(&self) -> bool {
        self.alpha.is_some()
    }

    /// Convert tightly-packed RGBA8 (`width × height × 4`, top-left origin) to
    /// 4:2:0 using BT.601 limited-range coefficients with 2×2 box chroma
    /// subsampling. The alpha channel is preserved full-resolution in
    /// [`alpha`](Self::alpha) when any pixel is non-opaque, otherwise dropped.
    pub fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Self {
        assert!(
            width.is_multiple_of(2) && height.is_multiple_of(2),
            "4:2:0 needs even dimensions"
        );
        assert_eq!(
            rgba.len(),
            (width * height * 4) as usize,
            "rgba size mismatch"
        );
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = (w / 2, h / 2);
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        let mut a = vec![0u8; w * h];
        let mut any_transparent = false;

        // Luma + alpha per pixel.
        for i in 0..(w * h) {
            let (r, g, b) = (
                rgba[i * 4] as i32,
                rgba[i * 4 + 1] as i32,
                rgba[i * 4 + 2] as i32,
            );
            y[i] = clamp8(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
            let alpha = rgba[i * 4 + 3];
            a[i] = alpha;
            any_transparent |= alpha != 255;
        }
        // Chroma from 2×2 RGB averages.
        for cy in 0..ch {
            for cx in 0..cw {
                let mut r = 0i32;
                let mut g = 0i32;
                let mut b = 0i32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let p = ((cy * 2 + dy) * w + (cx * 2 + dx)) * 4;
                        r += rgba[p] as i32;
                        g += rgba[p + 1] as i32;
                        b += rgba[p + 2] as i32;
                    }
                }
                let (r, g, b) = (r / 4, g / 4, b / 4);
                u[cy * cw + cx] = clamp8(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128);
                v[cy * cw + cx] = clamp8(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128);
            }
        }
        Self {
            width,
            height,
            y,
            u,
            v,
            alpha: any_transparent.then_some(a),
        }
    }
}

#[inline]
fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// A from-scratch HEVC encoder (milestone 3: IDR-only, `I_PCM`).
pub struct HevcEncoder {
    cfg: HevcConfig,
    headers_emitted: bool,
}

impl HevcEncoder {
    /// New color (4:2:0) encoder for a `width × height` display picture.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            cfg: HevcConfig::new(width, height),
            headers_emitted: false,
        }
    }

    /// New monochrome (4:0:0) encoder — the format of a transparent video's
    /// alpha auxiliary layer. Feed single-plane frames via
    /// [`encode_gray_frame`](Self::encode_gray_frame).
    pub fn new_monochrome(width: u32, height: u32) -> Self {
        Self {
            cfg: HevcConfig::new_monochrome(width, height),
            headers_emitted: false,
        }
    }

    /// The active configuration (coded size, CTB grid, …).
    pub fn config(&self) -> &HevcConfig {
        &self.cfg
    }

    /// Encode one 4:2:0 frame, returning its Annex-B access unit. The first call
    /// prepends VPS/SPS/PPS; subsequent calls emit the IDR slice only.
    pub fn encode_frame(&mut self, frame: &Yuv420Frame) -> Vec<u8> {
        assert!(
            self.cfg.has_chroma(),
            "encode_frame on a monochrome encoder"
        );
        assert_eq!(
            (frame.width, frame.height),
            (self.cfg.width, self.cfg.height),
            "frame size differs from encoder size"
        );

        // Pad each plane up to the coded (CTB-aligned) size by edge replication.
        let (cw, ch) = (self.cfg.coded_width, self.cfg.coded_height);
        let y = pad_plane(&frame.y, frame.width, frame.height, cw, ch);
        let u = pad_plane(&frame.u, frame.width / 2, frame.height / 2, cw / 2, ch / 2);
        let v = pad_plane(&frame.v, frame.width / 2, frame.height / 2, cw / 2, ch / 2);

        let padded = PaddedYuv {
            y: &y,
            u: &u,
            v: &v,
            coded_width: cw,
            coded_height: ch,
        };
        self.assemble_au(&padded)
    }

    /// Encode one monochrome frame — a tightly-packed `width × height` 8-bit
    /// plane (e.g. an alpha channel). Returns the Annex-B access unit.
    pub fn encode_gray_frame(&mut self, gray: &[u8]) -> Vec<u8> {
        assert!(
            !self.cfg.has_chroma(),
            "encode_gray_frame on a color encoder"
        );
        assert_eq!(
            gray.len(),
            (self.cfg.width * self.cfg.height) as usize,
            "gray plane size differs from encoder size"
        );
        let (cw, ch) = (self.cfg.coded_width, self.cfg.coded_height);
        let y = pad_plane(gray, self.cfg.width, self.cfg.height, cw, ch);
        let padded = PaddedYuv {
            y: &y,
            u: &[],
            v: &[],
            coded_width: cw,
            coded_height: ch,
        };
        self.assemble_au(&padded)
    }

    /// Emit VPS/SPS/PPS (first AU only) + the IDR slice for `padded`.
    fn assemble_au(&mut self, padded: &PaddedYuv) -> Vec<u8> {
        let slice = slice_segment_rbsp(&self.cfg, padded);
        let mut au = Vec::new();
        if !self.headers_emitted {
            let mono = !self.cfg.has_chroma();
            push_annexb(&mut au, &nal_unit_base(NalUnitType::Vps, &write_vps(mono)));
            push_annexb(
                &mut au,
                &nal_unit_base(NalUnitType::Sps, &write_sps(&self.cfg)),
            );
            push_annexb(&mut au, &nal_unit_base(NalUnitType::Pps, &write_pps()));
            self.headers_emitted = true;
        }
        push_annexb(&mut au, &nal_unit_base(NalUnitType::IdrNLp, &slice));
        au
    }
}

/// Copy a `sw × sh` plane into a `dw × dh` buffer, replicating the right/bottom
/// edges into the padding. `dw >= sw`, `dh >= sh`.
pub(crate) fn pad_plane(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let (sw, sh, dw, dh) = (sw as usize, sh as usize, dw as usize, dh as usize);
    let mut out = vec![0u8; dw * dh];
    for y in 0..dh {
        let sy = y.min(sh - 1);
        for x in 0..dw {
            let sx = x.min(sw - 1);
            out[y * dw + x] = src[sy * sw + sx];
        }
    }
    out
}

// CTB_SIZE is re-exported for callers that want to align sizes themselves.
pub use crate::codec::hevc::params::CTB_SIZE as CODING_TREE_BLOCK_SIZE;
const _: () = assert!(CTB_SIZE == 16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::hevc::nal::START_CODE;

    fn find(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    /// A deterministic, non-flat test frame so a decoder mismatch can't hide in
    /// uniform data.
    fn gradient(width: u32, height: u32) -> Yuv420Frame {
        let mut f = Yuv420Frame::new(width, height);
        for j in 0..height {
            for i in 0..width {
                f.y[(j * width + i) as usize] = ((i * 3 + j * 5) & 0xFF) as u8;
            }
        }
        let (cw, ch) = (width / 2, height / 2);
        for j in 0..ch {
            for i in 0..cw {
                f.u[(j * cw + i) as usize] = ((i * 7 + 30) & 0xFF) as u8;
                f.v[(j * cw + i) as usize] = ((j * 11 + 60) & 0xFF) as u8;
            }
        }
        f
    }

    #[test]
    fn access_unit_has_expected_nals() {
        let mut enc = HevcEncoder::new(64, 64);
        let au = enc.encode_frame(&gradient(64, 64));
        assert!(au.starts_with(&START_CODE));
        // VPS(0x40 01), SPS(0x42 01), PPS(0x44 01), IDR_N_LP slice (20<<1 = 0x28 01).
        assert!(find(&au, &[0x40, 0x01]), "VPS NAL header");
        assert!(find(&au, &[0x42, 0x01]), "SPS NAL header");
        assert!(find(&au, &[0x44, 0x01]), "PPS NAL header");
        assert!(find(&au, &[0x28, 0x01]), "IDR slice NAL header");

        // Second frame is slice-only.
        let au2 = enc.encode_frame(&gradient(64, 64));
        assert!(!find(&au2, &[0x42, 0x01]), "SPS must not repeat");
        assert!(find(&au2, &[0x28, 0x01]), "IDR slice present");
    }
}
