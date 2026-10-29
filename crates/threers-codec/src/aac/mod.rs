//! Pure-Rust AAC-LC and HE-AAC v1/v2 decoding that reproduces ffmpeg n7.1 (`-c:a aac`, the
//! native float decoder, aarch64 build): the decoder ([`AacDecoder`]), the
//! MP4 track and edit-list handling of its mov demuxer ([`demux_aac`]), and
//! [`decode_mp4`], which yields the planar float samples
//! `ffmpeg -i in.mp4 -f f32le -c:a pcm_f32le` would.
//!
//! Agreement: bit for bit. Every float operation follows the reference
//! build: the C decoder and av_tx's MDCT wrapper unfused, libavutil's NEON
//! `float_dsp` kernels and the NEON split-radix FFT ([`tx_neon`]) with
//! their fused multiply-adds and lane orders.
//!
//! HE-AAC: SBR (`sbr`, explicit, backward-compatible or implicit
//! signalling, coupled stereo, single-rate) and parametric stereo (`ps`,
//! 20/34 bands, IPD/OPD), with the NEON `sbrdsp`/`psdsp` kernels and
//! `ps_tableinit`'s tables as the optimised reference build computes them.
//!
//! Anything outside one mono SCE / stereo CPE (indexed or a one-element
//! PCE) with 1024-sample frames is refused with [`AacError::Unsupported`].

pub mod bits;
pub mod decoder;
pub mod dsp;
pub mod mp4;
pub mod ps;
pub mod sbr;
#[allow(clippy::all)]
/// SBR and parametric-stereo tables, generated from ffmpeg n7.1.
pub mod sbr_tables;
#[allow(clippy::all)]
/// Huffman codebooks, scalefactor bands and windows, generated from ffmpeg
/// n7.1 `aactab.c`.
pub mod tables;
pub mod tx_neon;
pub mod vlc;

pub use decoder::{AacDecoder, AacError, AudioSpecificConfig, ToolStats};
pub use mp4::{demux_aac, AacTrack, AudioSample};

/// Decoded audio: one `Vec` per channel.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Samples per second.
    pub sample_rate: u32,
    /// One `Vec` per channel.
    pub planes: Vec<Vec<f32>>,
    /// Samples in each frame the decoder delivered after the skip, in order
    /// (what ffmpeg's filters receive per call).
    pub frame_lens: Vec<usize>,
    /// Which coding tools the stream used.
    pub stats: ToolStats,
}

/// Decode the first AAC track of an MP4/MOV the way ffmpeg's demuxer and
/// decoder together deliver it (priming dropped, edit list applied).
pub fn decode_mp4(data: &[u8]) -> Result<DecodedAudio, AacError> {
    let track = demux_aac(data).map_err(AacError::Unsupported)?;
    let mut dec = AacDecoder::new(&track.asc)?;
    let mut planes = vec![Vec::new(); dec.channels()];
    for s in &track.samples[track.keep.0..track.keep.1] {
        let pkt = data
            .get(s.offset as usize..s.offset as usize + s.size as usize)
            .ok_or_else(|| AacError::Invalid("sample outside the file".into()))?;
        dec.decode_frame(pkt, &mut planes)?;
    }
    let mut skip = track.skip_samples as usize;
    if dec.libfaac_skip {
        skip = 1024;
    }
    let total = planes[0].len();
    skip = skip.min(total);
    for p in &mut planes {
        p.drain(..skip);
    }
    let mut frame_lens = Vec::new();
    let mut at = 0;
    let flen = dec.frame_len();
    while at < total {
        let end = at + flen;
        if end > skip {
            frame_lens.push(end - at.max(skip));
        }
        at = end;
    }
    Ok(DecodedAudio { sample_rate: dec.sample_rate(), planes, frame_lens, stats: dec.stats })
}
