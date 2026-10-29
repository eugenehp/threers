//! The types most programs need, in one import.
//!
//! ```
//! use threers_codec::prelude::*;
//! ```
//!
//! brings in the hardware-first [`Mp4Encoder`] and [`Decoder`], each codec's
//! encoder and decoder type, the demuxers' track types, the frame and sample
//! containers they trade in, and the codec modules themselves (`h264::`,
//! `aac::` and so on) for everything one level down. Anything not here is a
//! path away under the crate root.

pub use crate::captions::{CaptionError, CaptionFormat, CaptionTrack, Cue};
pub use crate::hw::{EncodeBackend, EncodeOptions, Mp4Encoder};
pub use crate::rate::Quality;

// Video encoders and the frame they take.
pub use crate::h264::{H264Encoder, InterEncoder};
pub use crate::hevc::{CompressedEncoder as HevcCompressedEncoder, HevcEncoder, TransparentEncoder, Yuv420Frame};
pub use crate::apng::ApngEncoder;
#[cfg(feature = "gif")]
pub use crate::gif::{decode_gif, encode_gif, GifDecoder, GifEncoder, GifFrame, GifOptions};
pub use crate::animation::{encode_animation_rgba, AnimationEncodeOptions, BrowserCodec};

// Video decoders and what they return.
#[cfg(feature = "decode")]
pub use crate::decode::{
    Backend, ChromaFormat, ColorInfo, DecodedFrame, Decoder, H264Decoder, HevcDecoder, Matrix,
    NativeDecoder, Plane,
};

// Audio.
pub use crate::aac::{AacDecoder, AacTrack, DecodedAudio};
pub use crate::audio_file::{self, Kind as AudioKind};
pub use crate::pcm::Pcm;

// Containers.
pub use crate::mkv_demux::{Mkv, MkvFrame, MkvTrack, TrackKind};
pub use crate::mp4::Mp4Params;
pub use crate::mp4_demux::{demux_video, Sample, TrackCodec, VideoTrack};
pub use crate::mp4_stream::Mp4StreamWriter;
pub use crate::webm::{mux_webm, WebmCodec, WebmFrame, WebmParams};

// The codec modules, for the next level down.
pub use crate::{aac, decode, flac, h264, hevc, mp3, mp4, opus, vorbis, vp9, webm};

#[cfg(all(feature = "videotoolbox", target_os = "macos"))]
pub use crate::videotoolbox::VideoToolboxEncoder;
#[cfg(all(feature = "videotoolbox", target_os = "macos"))]
pub use crate::videotoolbox_decode::VideoToolboxDecoder;
#[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
pub use crate::nvidia::{NvdecDecoder, NvencEncoder};
#[cfg(all(feature = "vaapi", target_os = "linux"))]
pub use crate::vaapi::VaapiDecoder;
