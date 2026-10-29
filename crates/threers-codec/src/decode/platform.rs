//! The best decoder this build has: VideoToolbox on macOS (with the
//! `videotoolbox` feature), [`NativeDecoder`] everywhere else.
//!
//! [`Decoder`] has the interface both of them share — push samples in decode
//! order, take pictures back by presentation time — so code written against
//! it runs on every platform and gets hardware decoding where there is some.
//! When VideoToolbox will not open a stream, [`Decoder::new`] falls back to
//! the native decoder rather than failing.
//!
//! ```no_run
//! use threers_codec::decode::Decoder;
//! use threers_codec::mp4_demux::demux_video;
//!
//! let file = std::fs::read("clip.mp4").unwrap();
//! let track = demux_video(&file).unwrap();
//! let mut dec = Decoder::new(&track).unwrap();
//! println!("decoding with {:?}", dec.backend());
//! for s in &track.samples {
//!     dec.push(&file[s.offset as usize..][..s.size as usize], s.pts, s.dts, s.duration as i64).unwrap();
//!     let _bgra = dec.take(s.pts);
//! }
//! ```

use super::NativeDecoder;
use crate::mp4_demux::{TrackCodec, VideoTrack};

#[cfg(all(feature = "videotoolbox", target_os = "macos"))]
use crate::videotoolbox_decode::VideoToolboxDecoder;
#[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
use crate::nvidia::NvdecDecoder;
#[cfg(all(feature = "vaapi", target_os = "linux"))]
use crate::vaapi::VaapiDecoder;

/// Which decoder a [`Decoder`] is using.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Backend {
    /// Apple's, in hardware where the machine has it. macOS with the
    /// `videotoolbox` feature.
    VideoToolbox,
    /// An NVIDIA GPU's NVDEC engine. Linux and Windows with the `nvidia`
    /// feature, on a machine with the driver.
    Nvdec,
    /// A VA driver (AMD, Intel, ...). Linux with the `vaapi` feature, on a
    /// machine with libva and a driver for the GPU.
    Vaapi,
    /// [`NativeDecoder`]: pure Rust, every platform.
    Native,
}

impl Backend {
    /// Whether this build, on this machine, can decode with `self`. For
    /// NVDEC that means loading the driver, which happens once.
    pub fn available(self) -> bool {
        match self {
            Backend::VideoToolbox => cfg!(all(feature = "videotoolbox", target_os = "macos")),
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Backend::Nvdec => crate::nvidia::available(),
            #[cfg(not(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64")))]
            Backend::Nvdec => false,
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Backend::Vaapi => crate::vaapi::available(),
            #[cfg(not(all(feature = "vaapi", target_os = "linux")))]
            Backend::Vaapi => false,
            Backend::Native => true,
        }
    }

    /// The hardware backends, in the order [`Decoder::new`] tries them.
    pub const HARDWARE: [Backend; 3] = [Backend::VideoToolbox, Backend::Nvdec, Backend::Vaapi];

    /// The backend [`Decoder::new`] tries first.
    pub fn preferred() -> Backend {
        Self::HARDWARE.into_iter().find(|b| b.available()).unwrap_or(Backend::Native)
    }
}

enum Inner {
    #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
    VideoToolbox(VideoToolboxDecoder),
    #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
    Nvdec(Box<NvdecDecoder>),
    #[cfg(all(feature = "vaapi", target_os = "linux"))]
    Vaapi(Box<VaapiDecoder>),
    Native(Box<NativeDecoder>),
}

/// A video decoder on whichever [`Backend`] suits the platform.
pub struct Decoder {
    inner: Inner,
}

impl Decoder {
    /// The first [hardware backend](Backend::HARDWARE) available that takes
    /// `track`, or the native decoder when none does.
    pub fn new(track: &VideoTrack) -> Result<Self, String> {
        for b in Backend::HARDWARE.into_iter().filter(|b| b.available()) {
            if let Ok(d) = Self::with_backend(track, b) {
                return Ok(d);
            }
        }
        // On failure, the native decoder says why.
        Self::with_backend(track, Backend::Native)
    }

    /// [`new`](Self::new) from a decoder configuration alone — an `avcC` or
    /// `hvcC` payload — for streams that do not come from a demuxed file.
    /// `timescale` is the unit of the timestamps given to [`push`](Self::push).
    pub fn from_config(codec: TrackCodec, config: &[u8], width: u32, height: u32, timescale: u32) -> Result<Self, String> {
        Self::new(&config_track(codec, config, width, height, timescale))
    }

    /// Exactly `backend`, without falling back: an error when this build does
    /// not have it or it will not take the stream.
    pub fn with_backend(track: &VideoTrack, backend: Backend) -> Result<Self, String> {
        let inner = match backend {
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Backend::VideoToolbox => Inner::VideoToolbox(VideoToolboxDecoder::new(track)?),
            #[cfg(not(all(feature = "videotoolbox", target_os = "macos")))]
            Backend::VideoToolbox => {
                return Err("VideoToolbox needs macOS and threers' `videotoolbox` feature".into())
            }
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Backend::Nvdec => Inner::Nvdec(Box::new(NvdecDecoder::new(track)?)),
            #[cfg(not(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64")))]
            Backend::Nvdec => {
                return Err("NVDEC needs Linux or Windows and the `nvidia` feature".into())
            }
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Backend::Vaapi => Inner::Vaapi(Box::new(VaapiDecoder::new(track)?)),
            #[cfg(not(all(feature = "vaapi", target_os = "linux")))]
            Backend::Vaapi => return Err("VAAPI needs Linux and the `vaapi` feature".into()),
            Backend::Native => Inner::Native(Box::new(NativeDecoder::new(track)?)),
        };
        Ok(Decoder { inner })
    }

    /// Which decoder is doing the work.
    pub fn backend(&self) -> Backend {
        match self.inner {
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Inner::VideoToolbox(_) => Backend::VideoToolbox,
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Inner::Nvdec(_) => Backend::Nvdec,
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Inner::Vaapi(_) => Backend::Vaapi,
            Inner::Native(_) => Backend::Native,
        }
    }

    /// Size of the pictures, as the stream's configuration states it.
    pub fn size(&self) -> (u32, u32) {
        match &self.inner {
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Inner::VideoToolbox(d) => d.size(),
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Inner::Nvdec(d) => d.size(),
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Inner::Vaapi(d) => d.size(),
            Inner::Native(d) => d.size(),
        }
    }

    /// Hand one coded frame to the decoder, in **decode order**:
    /// length-prefixed NAL units, as they sit in an MP4 sample. Its picture
    /// can be [taken](Self::take) as soon as this returns.
    pub fn push(&mut self, sample: &[u8], pts: i64, dts: i64, duration: i64) -> Result<(), String> {
        match &mut self.inner {
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Inner::VideoToolbox(d) => d.push(sample, pts, dts, duration),
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Inner::Nvdec(d) => d.push(sample, pts, dts, duration),
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Inner::Vaapi(d) => d.push(sample, pts, dts, duration),
            Inner::Native(d) => d.push(sample, pts, dts, duration),
        }
    }

    /// The picture for presentation time `pts` as tightly packed BGRA,
    /// `width * height * 4`, if it has been decoded.
    pub fn take(&mut self, pts: i64) -> Option<Vec<u8>> {
        match &mut self.inner {
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Inner::VideoToolbox(d) => d.take(pts),
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Inner::Nvdec(d) => d.take(pts),
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Inner::Vaapi(d) => d.take(pts),
            Inner::Native(d) => d.take(pts),
        }
    }
}

fn config_track(codec: TrackCodec, config: &[u8], width: u32, height: u32, timescale: u32) -> VideoTrack {
    VideoTrack { codec, width, height, config: config.to_vec(), timescale, samples: Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h264::{CompressedEncoder, Yuv420Frame};

    #[test]
    fn decodes_on_every_backend_this_build_has() {
        let (w, h) = (64u32, 32u32);
        let rgba: Vec<u8> = (0..w * h).flat_map(|i| [(i % 64 * 4) as u8, 120, 200, 255]).collect();
        let enc = CompressedEncoder::new(w, h, 20);
        let sample = enc.encode_sample(&Yuv420Frame::from_rgba(w, h, &rgba));
        let track = config_track(TrackCodec::H264, &enc.avcc(), w, h, 90_000);
        for backend in [Backend::VideoToolbox, Backend::Nvdec, Backend::Vaapi, Backend::Native] {
            let mut dec = match Decoder::with_backend(&track, backend) {
                Ok(d) => d,
                // Hardware may refuse a stream (this one is tiny, below some
                // GPUs' minimum size); `Decoder::new` then decodes natively.
                Err(e) if backend != Backend::Native => {
                    eprintln!("{backend:?}: {e}");
                    continue;
                }
                Err(e) => panic!("the native decoder would not open: {e}"),
            };
            assert_eq!(dec.backend(), backend);
            dec.push(&sample, 0, 0, 3600).unwrap();
            let bgra = dec.take(0).expect("a picture");
            assert_eq!(bgra.len(), (w * h * 4) as usize);
            // Mid-frame, the blue channel is near the 200 put in.
            let px = &bgra[((16 * w + 32) * 4) as usize..][..4];
            assert!((px[0] as i32 - 200).abs() < 12, "{backend:?}: {px:?}");
        }
        let auto = Decoder::from_config(TrackCodec::H264, &enc.avcc(), w, h, 90_000).unwrap();
        // The preferred backend, unless it refused this stream.
        assert!([Backend::preferred(), Backend::Native].contains(&auto.backend()));
    }
}
