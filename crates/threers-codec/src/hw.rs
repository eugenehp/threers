//! Hardware acceleration with a software fallback, behind one interface each
//! way.
//!
//! | Direction | Type | Hardware | Fallback |
//! |-----------|------|----------|----------|
//! | encode | [`Mp4Encoder`] | VideoToolbox HEVC (macOS, `videotoolbox`); NVENC HEVC (Linux/Windows, `nvidia`); VAAPI HEVC (Linux, `vaapi`) | [`hevc::CompressedEncoder`](crate::hevc::CompressedEncoder), frame-parallel with `parallel` |
//! | decode | [`Decoder`] (feature `decode`) | VideoToolbox H.264/HEVC (macOS, `videotoolbox`); NVDEC H.264/HEVC/VP9/AV1 (Linux/Windows, `nvidia`); VAAPI H.264/HEVC/VP9/AV1 (Linux, `vaapi`) | [`NativeDecoder`](crate::decode::NativeDecoder): H.264, HEVC, VP9, AV1 |
//!
//! Code written against these runs on every target, `wasm32` included, and
//! picks up the hardware block where the build and the machine have one. When
//! the hardware refuses a stream — a profile it does not do, a session it will
//! not open — the constructor falls back to the software codec rather than
//! failing; [`Mp4Encoder::backend`] and [`Decoder::backend`] say which one
//! you got. Pin one with `with_backend` to measure or test it on its own.
//!
//! ```no_run
//! use threers_codec::hw::{EncodeOptions, Mp4Encoder};
//!
//! let (w, h) = (1280, 720);
//! let mut enc = Mp4Encoder::new(w, h, EncodeOptions::default()).unwrap();
//! println!("encoding with {:?}", enc.backend());
//! for i in 0..60u32 {
//!     let rgba = vec![(i * 4) as u8; (w * h * 4) as usize];
//!     enc.push_rgba(&rgba).unwrap();
//! }
//! let mp4: Vec<u8> = enc.finish().unwrap();
//! std::fs::write("clip.mp4", mp4).unwrap();
//! ```

#[cfg(feature = "decode")]
pub use crate::decode::{Backend, Decoder};

use crate::hevc::{CompressedEncoder, HvccArray, HvccProfile, Yuv420Frame};
use crate::mp4::{mux_hevc, Mp4Params};
use crate::rate::{RateControl, PROBE_QP};

#[cfg(all(feature = "videotoolbox", target_os = "macos"))]
use crate::videotoolbox::{VideoToolboxEncoder, VtTuning};
#[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
use crate::nvidia::{NvencEncoder, NvencRate};
#[cfg(all(feature = "vaapi", target_os = "linux"))]
use crate::vaapi::{VaapiEncoder, VaapiRate};

/// Which encoder an [`Mp4Encoder`] is using.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeBackend {
    /// Apple's hardware HEVC encoder. macOS with the `videotoolbox` feature.
    VideoToolbox,
    /// An NVIDIA GPU's NVENC engine. Linux and Windows with the `nvidia`
    /// feature, on a machine with the driver.
    Nvenc,
    /// A VA driver's HEVC encoder (AMD, Intel). Linux with the `vaapi`
    /// feature, on a machine with libva and a driver that encodes.
    Vaapi,
    /// threers' own all-intra HEVC encoder: pure Rust, every platform, and
    /// spread across cores with the `parallel` feature.
    Native,
}

impl EncodeBackend {
    /// Whether this build can encode with `self`.
    pub fn available(self) -> bool {
        match self {
            EncodeBackend::VideoToolbox => {
                cfg!(all(feature = "videotoolbox", target_os = "macos"))
            }
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            EncodeBackend::Nvenc => crate::nvidia::available(),
            #[cfg(not(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64")))]
            EncodeBackend::Nvenc => false,
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            EncodeBackend::Vaapi => crate::vaapi::encode_available(),
            #[cfg(not(all(feature = "vaapi", target_os = "linux")))]
            EncodeBackend::Vaapi => false,
            EncodeBackend::Native => true,
        }
    }

    /// The hardware backends, in the order [`Mp4Encoder::new`] tries them.
    pub const HARDWARE: [EncodeBackend; 3] = [EncodeBackend::VideoToolbox, EncodeBackend::Nvenc, EncodeBackend::Vaapi];

    /// The backend [`Mp4Encoder::new`] tries first.
    pub fn preferred() -> EncodeBackend {
        Self::HARDWARE.into_iter().find(|b| b.available()).unwrap_or(EncodeBackend::Native)
    }
}

/// What an [`Mp4Encoder`] is asked for. Each backend reads it in its own
/// terms; the fields say how.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    /// Frames per second.
    pub fps: u32,
    /// 0 (smallest) to 1 (best). VideoToolbox takes it as its own quality
    /// property; the native encoder, NVENC and VAAPI map it linearly onto
    /// quantizers 40 to 18, the range where the native encoder's output is
    /// usable, so 0.5 is QP 29.
    pub quality: f32,
    /// A target average bitrate in bits per second, which overrides `quality`
    /// on the native encoder, NVENC and VAAPI (variable bitrate) and caps it on
    /// VideoToolbox.
    pub bitrate: Option<u32>,
    /// Hint that encoding must keep pace with capture: VideoToolbox's realtime
    /// mode, NVENC's low-latency tuning and a faster preset. The native
    /// encoder ignores it.
    pub realtime: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            fps: 30,
            quality: 0.7,
            bitrate: None,
            realtime: false,
        }
    }
}

impl EncodeOptions {
    /// The quantizer the native encoder uses for [`quality`](Self::quality).
    pub fn native_qp(&self) -> i32 {
        (40.0 - self.quality.clamp(0.0, 1.0) * 22.0).round() as i32
    }
}

enum Inner {
    #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
    VideoToolbox(VideoToolboxEncoder),
    #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
    Nvenc(Box<NvencEncoder>),
    #[cfg(all(feature = "vaapi", target_os = "linux"))]
    Vaapi(Box<VaapiEncoder>),
    Native(Box<NativeMp4>),
}

/// An HEVC-in-MP4 encoder on whichever [`EncodeBackend`] suits the platform.
///
/// Push RGBA frames, then [`finish`](Self::finish) for the file's bytes. The
/// output is a standard `hvc1` MP4 either way; what differs is speed and that
/// the native encoder is all-intra, so its files are larger at the same
/// quality.
pub struct Mp4Encoder {
    inner: Inner,
}

impl Mp4Encoder {
    /// The first [hardware backend](EncodeBackend::HARDWARE) available that
    /// opens a session, or the native encoder when none does.
    pub fn new(width: u32, height: u32, options: EncodeOptions) -> Result<Self, String> {
        for b in EncodeBackend::HARDWARE.into_iter().filter(|b| b.available()) {
            if let Ok(e) = Self::with_backend(width, height, options, b) {
                return Ok(e);
            }
        }
        Self::with_backend(width, height, options, EncodeBackend::Native)
    }

    /// Exactly `backend`, failing if this build or this machine cannot.
    pub fn with_backend(
        width: u32,
        height: u32,
        options: EncodeOptions,
        backend: EncodeBackend,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err(format!("cannot encode a {width}x{height} picture"));
        }
        match backend {
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            EncodeBackend::VideoToolbox => {
                let mut tuning = if options.realtime {
                    VtTuning::realtime()
                } else {
                    VtTuning::default()
                };
                tuning.average_bitrate = options.bitrate;
                let enc = VideoToolboxEncoder::new_for_mp4_tuned(
                    width,
                    height,
                    options.fps.max(1) as i32,
                    options.quality,
                    tuning,
                )?;
                Ok(Self {
                    inner: Inner::VideoToolbox(enc),
                })
            }
            #[cfg(not(all(feature = "videotoolbox", target_os = "macos")))]
            EncodeBackend::VideoToolbox => {
                Err("VideoToolbox needs macOS and the `videotoolbox` feature".into())
            }
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            EncodeBackend::Nvenc => {
                let rate = match options.bitrate {
                    Some(bps) => NvencRate::Bitrate(bps),
                    None => NvencRate::ConstQp(options.native_qp() as u32),
                };
                let keyframe_secs = if options.realtime { 2 } else { 5 };
                let enc = NvencEncoder::new(width, height, options.fps, rate, keyframe_secs, options.realtime)?;
                Ok(Self { inner: Inner::Nvenc(Box::new(enc)) })
            }
            #[cfg(not(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64")))]
            EncodeBackend::Nvenc => Err("NVENC needs Linux or Windows and the `nvidia` feature".into()),
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            EncodeBackend::Vaapi => {
                let rate = match options.bitrate {
                    Some(bps) => VaapiRate::Bitrate(bps),
                    None => VaapiRate::ConstQp(options.native_qp() as u32),
                };
                let keyframe_secs = if options.realtime { 2 } else { 5 };
                let enc = VaapiEncoder::new(width, height, options.fps, rate, keyframe_secs)?;
                Ok(Self { inner: Inner::Vaapi(Box::new(enc)) })
            }
            #[cfg(not(all(feature = "vaapi", target_os = "linux")))]
            EncodeBackend::Vaapi => Err("VAAPI needs Linux and the `vaapi` feature".into()),
            EncodeBackend::Native => Ok(Self {
                inner: Inner::Native(Box::new(NativeMp4::new(width, height, options))),
            }),
        }
    }

    /// The backend in use.
    pub fn backend(&self) -> EncodeBackend {
        match &self.inner {
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Inner::VideoToolbox(_) => EncodeBackend::VideoToolbox,
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Inner::Nvenc(_) => EncodeBackend::Nvenc,
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Inner::Vaapi(_) => EncodeBackend::Vaapi,
            Inner::Native(_) => EncodeBackend::Native,
        }
    }

    /// Encode one frame of tightly packed RGBA8, `width * height * 4` bytes.
    pub fn push_rgba(&mut self, rgba: &[u8]) -> Result<(), String> {
        match &mut self.inner {
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Inner::VideoToolbox(e) => e.push_rgba(rgba),
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Inner::Nvenc(e) => e.push_rgba(rgba),
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Inner::Vaapi(e) => e.push_rgba(rgba),
            Inner::Native(e) => e.push_rgba(rgba),
        }
    }

    /// Flush the encoder and return the MP4 file.
    pub fn finish(self) -> Result<Vec<u8>, String> {
        match self.inner {
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Inner::VideoToolbox(e) => e.finish_mp4_bytes().map(|(_, mp4)| mp4),
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Inner::Nvenc(e) => e.finish(),
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Inner::Vaapi(e) => e.finish(),
            Inner::Native(e) => e.finish(),
        }
    }
}

/// The native side of [`Mp4Encoder`]: frames are buffered just long enough to
/// encode a batch of them at once, which is free because every frame is an
/// IDR and nothing carries from one to the next.
struct NativeMp4 {
    enc: CompressedEncoder,
    width: u32,
    height: u32,
    fps: u32,
    qp: i32,
    rc: Option<RateControl>,
    batch: usize,
    pending: Vec<Yuv420Frame>,
    samples: Vec<Vec<u8>>,
}

impl NativeMp4 {
    fn new(width: u32, height: u32, options: EncodeOptions) -> Self {
        let fps = options.fps.max(1);
        let rc = options
            .bitrate
            .map(|bps| RateControl::new(bps as u64, fps, None));
        let batch = match &rc {
            Some(r) => r.group_size(crate::frames_by_memory(width, height)),
            None => crate::frames_in_flight(width, height),
        };
        Self {
            enc: CompressedEncoder::new(width, height, PROBE_QP),
            width,
            height,
            fps,
            qp: options.native_qp(),
            rc,
            batch: batch.max(1),
            pending: Vec::new(),
            samples: Vec::new(),
        }
    }

    fn push_rgba(&mut self, rgba: &[u8]) -> Result<(), String> {
        let need = self.width as usize * self.height as usize * 4;
        if rgba.len() < need {
            return Err(format!("frame is {} bytes, want {need}", rgba.len()));
        }
        let frame = Yuv420Frame::from_rgba(self.width, self.height, &rgba[..need]);
        // The first frame tells a bitrate target what this content costs.
        if let Some(r) = self.rc.as_mut() {
            if self.samples.is_empty() && self.pending.is_empty() {
                r.seed(PROBE_QP, self.enc.encode_slice_au_at(&frame, PROBE_QP).len());
                if let Some(q2) = r.probe_again() {
                    r.seed_pair(q2, self.enc.encode_slice_au_at(&frame, q2).len());
                }
            }
        }
        self.pending.push(frame);
        if self.pending.len() >= self.batch {
            self.flush();
        }
        Ok(())
    }

    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let qp = self.rc.as_ref().map_or(self.qp, |r| r.next_qp());
        let enc = &self.enc;
        let aus = crate::map_frames(&self.pending, |f| enc.encode_slice_au_at(f, qp));
        self.pending.clear();
        for au in aus {
            if let Some(r) = self.rc.as_mut() {
                r.observe(qp, au.len());
            }
            let mut sample = Vec::with_capacity(au.len());
            for nal in crate::h264::nal::split_annexb(&au) {
                sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                sample.extend_from_slice(&nal);
            }
            self.samples.push(sample);
        }
    }

    fn finish(mut self) -> Result<Vec<u8>, String> {
        self.flush();
        if self.samples.is_empty() {
            return Err("no frames were pushed".into());
        }
        let (vps, sps, pps) = self.enc.parameter_sets();
        let (vps, sps, pps) = ([vps], [sps], [pps]);
        let hvcc = crate::hevc::build_hvcc(
            &HvccProfile::main_420(self.enc.config().level_idc),
            &[
                HvccArray { nal_type: 32, complete: true, nals: &vps },
                HvccArray { nal_type: 33, complete: true, nals: &sps },
                HvccArray { nal_type: 34, complete: true, nals: &pps },
            ],
        );
        let timescale = 600u32;
        Ok(mux_hevc(&Mp4Params {
            width: self.width,
            height: self.height,
            timescale,
            frame_duration: timescale / self.fps,
            hvcc_payload: &hvcc,
            almo_payload: None,
            samples: &self.samples,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_maps_onto_the_usable_quantizers() {
        let at = |quality| EncodeOptions { quality, ..Default::default() }.native_qp();
        assert_eq!(at(0.0), 40);
        assert_eq!(at(0.5), 29);
        assert_eq!(at(1.0), 18);
        assert_eq!(at(7.0), 18);
    }

    #[test]
    fn native_backend_writes_an_mp4() {
        let (w, h) = (64, 48);
        let opts = EncodeOptions::default();
        let mut enc = Mp4Encoder::with_backend(w, h, opts, EncodeBackend::Native).unwrap();
        assert_eq!(enc.backend(), EncodeBackend::Native);
        for i in 0..5u8 {
            let rgba: Vec<u8> = (0..w * h * 4).map(|p| (p as u8).wrapping_mul(i)).collect();
            enc.push_rgba(&rgba).unwrap();
        }
        let mp4 = enc.finish().unwrap();
        assert_eq!(&mp4[4..8], b"ftyp");
        let track = crate::mp4_demux::demux_video(&mp4).unwrap();
        assert_eq!(track.samples.len(), 5);
        assert_eq!((track.width, track.height), (w, h));
    }

    #[test]
    fn a_short_frame_is_refused() {
        let mut enc =
            Mp4Encoder::with_backend(16, 16, EncodeOptions::default(), EncodeBackend::Native)
                .unwrap();
        assert!(enc.push_rgba(&[0; 16]).is_err());
    }

    #[test]
    fn unavailable_backends_say_so() {
        if !EncodeBackend::VideoToolbox.available() {
            let r = Mp4Encoder::with_backend(16, 16, Default::default(), EncodeBackend::VideoToolbox);
            assert!(r.is_err());
        }
        assert!(EncodeBackend::Native.available());
    }
}
