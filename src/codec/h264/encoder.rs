//! Frame-level H.264 encoder: pixels in, Annex-B byte stream or MP4 out.
//!
//! [`H264Encoder`] emits SPS/PPS once, then an IDR slice per frame. Baseline
//! profile with CAVLC `I_PCM` macroblocks (lossless in the YUV domain) — the
//! same proving strategy as the HEVC encoder.

use crate::captions::CaptionTrack;
use crate::codec::h264::avcc::build_avcc;
use crate::codec::h264::nal::{nal_unit_base, push_annexb, split_annexb, NalUnitType};
use crate::codec::h264::params::{write_sps, H264Config, MB_SIZE};
use crate::codec::h264::pps::write_pps;
use crate::codec::h264::slice::{idr_slice_rbsp, PaddedYuv};
use crate::codec::hevc::encoder::{pad_plane, Yuv420Frame};
use crate::codec::mp4::{mux_h264, mux_h264_with_captions, H264Mp4Params};

/// A from-scratch H.264 encoder (IDR-only, `I_PCM`).
pub struct H264Encoder {
    cfg: H264Config,
    headers_emitted: bool,
    sps: Vec<u8>,
    pps: Vec<u8>,
}

impl H264Encoder {
    pub fn new(width: u32, height: u32) -> Self {
        let cfg = H264Config::new(width, height);
        let sps = nal_unit_base(NalUnitType::Sps, &write_sps(&cfg));
        let pps = nal_unit_base(NalUnitType::Pps, &write_pps());
        Self {
            cfg,
            headers_emitted: false,
            sps,
            pps,
        }
    }

    pub fn config(&self) -> &H264Config {
        &self.cfg
    }

    /// SPS NAL (1-byte header + EBSP) for muxing.
    pub fn sps(&self) -> &[u8] {
        &self.sps
    }

    /// PPS NAL for muxing.
    pub fn pps(&self) -> &[u8] {
        &self.pps
    }

    /// `avcC` payload built from this encoder's parameter sets.
    pub fn avcc(&self) -> Vec<u8> {
        build_avcc(&self.sps, &self.pps)
    }

    /// Encode one 4:2:0 frame, returning its Annex-B access unit.
    pub fn encode_frame(&mut self, frame: &Yuv420Frame) -> Vec<u8> {
        assert_eq!(
            (frame.width, frame.height),
            (self.cfg.width, self.cfg.height),
            "frame size differs from encoder size"
        );
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

    fn assemble_au(&mut self, padded: &PaddedYuv) -> Vec<u8> {
        let slice = idr_slice_rbsp(&self.cfg, padded);
        let mut au = Vec::new();
        if !self.headers_emitted {
            push_annexb(&mut au, &self.sps);
            push_annexb(&mut au, &self.pps);
            self.headers_emitted = true;
        }
        push_annexb(&mut au, &nal_unit_base(NalUnitType::Idr, &slice));
        au
    }

    /// Turn one Annex-B access unit into an MP4 sample (length-prefixed NALs).
    pub fn sample_from_au(au: &[u8]) -> Vec<u8> {
        let mut sample = Vec::new();
        for nal in split_annexb(au) {
            let t = nal[0] & 0x1F;
            if t == NalUnitType::Sps.code() || t == NalUnitType::Pps.code() {
                continue;
            }
            sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            sample.extend_from_slice(&nal);
        }
        sample
    }
}

/// Encode `frames` of YUV420 pictures into an MP4 byte stream.
pub fn encode_mp4(width: u32, height: u32, fps: u32, frames: &[Yuv420Frame]) -> Vec<u8> {
    encode_mp4_with_captions(width, height, fps, frames, None)
}

/// Like [`encode_mp4`], optionally muxing a soft `tx3g` caption track.
pub fn encode_mp4_with_captions(
    width: u32,
    height: u32,
    fps: u32,
    frames: &[Yuv420Frame],
    captions: Option<&CaptionTrack>,
) -> Vec<u8> {
    encode_mp4_from_iter(width, height, fps, captions, frames.iter().cloned())
}

/// [`encode_mp4`] over an iterator of frames, so the caller can generate them
/// lazily and never hold more than one.
pub fn encode_mp4_from_iter<I>(
    width: u32,
    height: u32,
    fps: u32,
    captions: Option<&CaptionTrack>,
    frames: I,
) -> Vec<u8>
where
    I: IntoIterator<Item = Yuv420Frame>,
{
    encode_mp4_inner(width, height, fps, captions, &mut frames.into_iter())
}

/// [`encode_mp4`] pulling one frame at a time instead of taking them all up
/// front.
///
/// A 4:2:0 frame is `width * height * 1.5` bytes, so at 4K or 8K the source
/// frames dwarf everything else the encoder touches — ten seconds of 8K at 24fps
/// is roughly 12 GB of them. Nothing needs them to coexist: the muxer wants the
/// *encoded* samples, and those are orders of magnitude smaller. `next` is
/// called once per frame index, in order.
pub fn encode_mp4_streaming<F>(
    width: u32,
    height: u32,
    fps: u32,
    frame_count: usize,
    captions: Option<&CaptionTrack>,
    next: F,
) -> Vec<u8>
where
    F: FnMut(usize) -> Yuv420Frame,
{
    encode_mp4_inner(width, height, fps, captions, &mut (0..frame_count).map(next))
}

/// [`encode_mp4_from_iter`] written straight to a sink.
///
/// Combined with a lazy `frames`, nothing ever holds the whole clip *or* the
/// whole file: peak memory is one frame plus the encoded samples.
pub fn write_mp4_from_iter<W, I>(
    out: &mut W,
    width: u32,
    height: u32,
    fps: u32,
    captions: Option<&CaptionTrack>,
    frames: I,
) -> std::io::Result<()>
where
    W: std::io::Write,
    I: IntoIterator<Item = Yuv420Frame>,
{
    let fps = fps.max(1);
    let timescale = 600u32;
    let mut enc = H264Encoder::new(width, height);
    let avcc = enc.avcc();
    let mut samples = Vec::new();
    for f in frames {
        samples.push(H264Encoder::sample_from_au(&enc.encode_frame(&f)));
    }
    let params = H264Mp4Params {
        width,
        height,
        timescale,
        frame_duration: timescale / fps,
        avcc_payload: &avcc,
        samples: &samples,
    };
    crate::codec::mp4::write_h264(out, &params, captions.filter(|t| !t.is_empty()))
}

fn encode_mp4_inner(
    width: u32,
    height: u32,
    fps: u32,
    captions: Option<&CaptionTrack>,
    frames: &mut dyn Iterator<Item = Yuv420Frame>,
) -> Vec<u8> {
    let fps = fps.max(1);
    let timescale = 600u32;
    let frame_duration = timescale / fps;
    let mut enc = H264Encoder::new(width, height);
    let avcc = enc.avcc();
    let mut samples = Vec::new();
    for f in frames {
        samples.push(H264Encoder::sample_from_au(&enc.encode_frame(&f)));
    }
    let params = H264Mp4Params {
        width,
        height,
        timescale,
        frame_duration,
        avcc_payload: &avcc,
        samples: &samples,
    };
    match captions {
        Some(track) if !track.is_empty() => mux_h264_with_captions(&params, track),
        _ => mux_h264(&params),
    }
}

pub use crate::codec::h264::params::MB_SIZE as MACROBLOCK_SIZE;
const _: () = assert!(MB_SIZE == 16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::nal::{split_annexb, START_CODE};

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

    fn nal_starts_with(au: &[u8], header: u8) -> bool {
        split_annexb(au)
            .iter()
            .any(|nal| !nal.is_empty() && nal[0] == header)
    }

    #[test]
    fn access_unit_has_expected_nals() {
        let mut enc = H264Encoder::new(64, 64);
        let au = enc.encode_frame(&gradient(64, 64));
        assert!(au.starts_with(&START_CODE));
        assert!(nal_starts_with(&au, 0x67), "SPS NAL header");
        assert!(nal_starts_with(&au, 0x68), "PPS NAL header");
        assert!(nal_starts_with(&au, 0x65), "IDR slice NAL header");

        let au2 = enc.encode_frame(&gradient(64, 64));
        assert!(!nal_starts_with(&au2, 0x67), "SPS must not repeat");
        assert!(nal_starts_with(&au2, 0x65), "IDR slice present");
    }

    #[test]
    fn mp4_mux_produces_ftyp() {
        let frames = [gradient(64, 64)];
        let mp4 = encode_mp4(64, 64, 30, &frames);
        assert_eq!(&mp4[4..8], b"ftyp");
    }
}
