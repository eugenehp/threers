//! NVDEC through the driver's own parser (`cuvidCreateVideoParser`): it reads
//! the bitstream and hands back filled-in picture parameters, so nothing
//! here parses H.264, HEVC, VP9 or AV1 syntax.
//!
//! Pictures are copied to the host as they are decoded, in decode order, and
//! queued under the presentation time of the sample that carried them — the
//! contract [`NativeDecoder`](crate::decode::NativeDecoder) and the
//! VideoToolbox session keep, so a picture can be taken as soon as `push`
//! returns. VP9 and AV1 can code frames that are never shown; for those the
//! display callback, which only fires for shown frames, does the copy
//! instead.

use std::ffi::{c_int, c_short, c_uint, c_ulong, c_void};
use std::sync::OnceLock;

use super::{check, Context, CuResult, Lib, Memcpy2d, MEMORYTYPE_DEVICE, MEMORYTYPE_HOST};
use crate::decode::{ChromaFormat, ColorInfo, DecodedFrame};
use crate::mp4_demux::{TrackCodec, VideoTrack};

type Parser = *mut c_void;
type CuvidDecoder = *mut c_void;

const CODEC_H264: c_int = 4;
const CODEC_HEVC: c_int = 8;
const CODEC_VP9: c_int = 10;
const CODEC_AV1: c_int = 11;
const CHROMA_420: c_int = 1;
const SURFACE_NV12: c_int = 0;
const SURFACE_P016: c_int = 1;
const CREATE_PREFER_CUVID: c_ulong = 4;
const PKT_TIMESTAMP: c_ulong = 2;
const PKT_ENDOFPICTURE: c_ulong = 8;

/// `CUVIDEOFORMAT`.
#[repr(C)]
#[derive(Clone, Copy)]
struct VideoFormat {
    codec: c_int,
    frame_rate: [c_uint; 2],
    progressive_sequence: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    min_num_decode_surfaces: u8,
    coded_width: c_uint,
    coded_height: c_uint,
    /// left, top, right, bottom
    display_area: [c_int; 4],
    chroma_format: c_int,
    bitrate: c_uint,
    display_aspect_ratio: [c_int; 2],
    /// video_format:3 | full_range:1 | reserved:4, then primaries, transfer,
    /// matrix.
    video_signal_description: [u8; 4],
    seqhdr_data_length: c_uint,
}
const _: () = assert!(std::mem::size_of::<VideoFormat>() == 64);

/// `CUVIDPARSERPARAMS`.
#[repr(C)]
struct ParserParams {
    codec_type: c_int,
    max_num_decode_surfaces: c_uint,
    clock_rate: c_uint,
    error_threshold: c_uint,
    max_display_delay: c_uint,
    /// `bAnnexb:1`, reserved:31.
    flags: c_uint,
    reserved1: [c_uint; 4],
    user_data: *mut c_void,
    sequence: Option<unsafe extern "C" fn(*mut c_void, *mut VideoFormat) -> c_int>,
    decode: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int>,
    display: Option<unsafe extern "C" fn(*mut c_void, *mut DispInfo) -> c_int>,
    operating_point: *mut c_void,
    sei: *mut c_void,
    reserved2: [*mut c_void; 5],
    ext_video_info: *mut c_void,
}
const _: () = assert!(std::mem::size_of::<ParserParams>() == 136);

/// `CUVIDSOURCEDATAPACKET`.
#[repr(C)]
struct Packet {
    flags: c_ulong,
    payload_size: c_ulong,
    payload: *const u8,
    timestamp: i64,
}

/// `CUVIDPARSERDISPINFO`.
#[repr(C)]
struct DispInfo {
    picture_index: c_int,
    progressive_frame: c_int,
    top_field_first: c_int,
    repeat_first_field: c_int,
    timestamp: i64,
}
const _: () = assert!(std::mem::size_of::<DispInfo>() == 24);

/// `CUVIDDECODECREATEINFO`.
#[repr(C)]
struct CreateInfo {
    width: c_ulong,
    height: c_ulong,
    num_decode_surfaces: c_ulong,
    codec_type: c_int,
    chroma_format: c_int,
    creation_flags: c_ulong,
    bit_depth_minus8: c_ulong,
    intra_decode_only: c_ulong,
    max_width: c_ulong,
    max_height: c_ulong,
    reserved1: c_ulong,
    display_area: [c_short; 4],
    output_format: c_int,
    deinterlace_mode: c_int,
    target_width: c_ulong,
    target_height: c_ulong,
    num_output_surfaces: c_ulong,
    vid_lock: *mut c_void,
    target_rect: [c_short; 4],
    enable_histogram: c_ulong,
    reserved2: [c_ulong; 4],
}
#[cfg(unix)]
const _: () = assert!(std::mem::size_of::<CreateInfo>() == 176);

/// `CUVIDPROCPARAMS`.
#[repr(C)]
struct ProcParams {
    progressive_frame: c_int,
    second_field: c_int,
    top_field_first: c_int,
    unpaired_field: c_int,
    reserved_flags: c_uint,
    reserved_zero: c_uint,
    raw_input_dptr: u64,
    raw_input_pitch: c_uint,
    raw_input_format: c_uint,
    raw_output_dptr: u64,
    raw_output_pitch: c_uint,
    reserved1: c_uint,
    output_stream: *mut c_void,
    reserved: [c_uint; 46],
    histogram_dptr: *mut u64,
    reserved2: [*mut c_void; 1],
}
const _: () = assert!(std::mem::size_of::<ProcParams>() == 264);

/// Offset of `CurrPicIdx` in `CUVIDPICPARAMS`, which is otherwise passed
/// through untouched.
const PICPARAMS_CURR_PIC_IDX: usize = 8;

/// The NVDEC calls, from `libnvcuvid`.
pub(crate) struct Cuvid {
    _lib: Lib,
    create_parser: unsafe extern "C" fn(*mut Parser, *mut ParserParams) -> CuResult,
    parse: unsafe extern "C" fn(Parser, *mut Packet) -> CuResult,
    destroy_parser: unsafe extern "C" fn(Parser) -> CuResult,
    create_decoder: unsafe extern "C" fn(*mut CuvidDecoder, *mut CreateInfo) -> CuResult,
    destroy_decoder: unsafe extern "C" fn(CuvidDecoder) -> CuResult,
    decode_picture: unsafe extern "C" fn(CuvidDecoder, *mut c_void) -> CuResult,
    map: unsafe extern "C" fn(CuvidDecoder, c_int, *mut u64, *mut c_uint, *mut ProcParams) -> CuResult,
    unmap: unsafe extern "C" fn(CuvidDecoder, u64) -> CuResult,
}

impl Cuvid {
    fn load() -> Option<Cuvid> {
        super::cuda()?;
        let lib = Lib::open(&["libnvcuvid.so.1", "libnvcuvid.so", "nvcuvid.dll"])?;
        unsafe {
            Some(Cuvid {
                create_parser: lib.sym("cuvidCreateVideoParser")?,
                parse: lib.sym("cuvidParseVideoData")?,
                destroy_parser: lib.sym("cuvidDestroyVideoParser")?,
                create_decoder: lib.sym("cuvidCreateDecoder")?,
                destroy_decoder: lib.sym("cuvidDestroyDecoder")?,
                decode_picture: lib.sym("cuvidDecodePicture")?,
                map: lib.sym("cuvidMapVideoFrame64")?,
                unmap: lib.sym("cuvidUnmapVideoFrame64")?,
                _lib: lib,
            })
        }
    }
}

/// NVDEC, loaded once, or `None` without the driver.
pub(crate) fn cuvid() -> Option<&'static Cuvid> {
    static CUVID: OnceLock<Option<Cuvid>> = OnceLock::new();
    CUVID.get_or_init(Cuvid::load).as_ref()
}

/// What the callbacks share with the decoder. Boxed, so its address — the
/// parser's user data — does not move.
struct State {
    cuvid: &'static Cuvid,
    decoder: CuvidDecoder,
    format: Option<VideoFormat>,
    /// Copy out from the display callback (VP9, AV1) rather than the decode one.
    on_display: bool,
    /// Presentation time of the sample being parsed.
    pts: i64,
    /// Pictures copied out during the current `push`.
    copied: i64,
    ready: Vec<(i64, DecodedFrame)>,
    error: Option<String>,
}

impl State {
    fn fail(&mut self, e: String) -> c_int {
        self.error.get_or_insert(e);
        0
    }

    /// Map surface `index` and copy the display area to the host.
    unsafe fn copy_out(&mut self, index: c_int) -> Result<DecodedFrame, String> {
        let fmt = self.format.ok_or("a picture before the sequence header")?;
        let cuda = super::cuda().ok_or("CUDA went away")?;
        let mut proc: ProcParams = std::mem::zeroed();
        proc.progressive_frame = 1;
        let (mut dptr, mut pitch) = (0u64, 0u32);
        check("cuvidMapVideoFrame", (self.cuvid.map)(self.decoder, index, &mut dptr, &mut pitch, &mut proc))?;
        let [left, top, right, bottom] = fmt.display_area;
        let (w, h) = ((right - left) as usize, (bottom - top) as usize);
        let deep = fmt.bit_depth_luma_minus8 > 0;
        let bps = if deep { 2 } else { 1 };
        let surface_h = fmt.coded_height as usize;
        let pitch = pitch as usize;
        let mut luma = vec![0u8; w * h * bps];
        let mut chroma = vec![0u8; w.div_ceil(2) * 2 * h.div_ceil(2) * bps];
        let copy = |src: u64, x: usize, y: usize, dst: &mut [u8], width: usize, rows: usize| {
            let mut m = Memcpy2d::zeroed();
            m.src_memory_type = MEMORYTYPE_DEVICE;
            m.src_device = src;
            m.src_pitch = pitch;
            m.src_x_in_bytes = x;
            m.src_y = y;
            m.dst_memory_type = MEMORYTYPE_HOST;
            m.dst_host = dst.as_mut_ptr().cast();
            m.dst_pitch = width;
            m.width_in_bytes = width;
            m.height = rows;
            check("cuMemcpy2D", (cuda.memcpy_2d)(&m))
        };
        let r = copy(dptr, left as usize * bps, top as usize, &mut luma, w * bps, h).and_then(|_| {
            let uv = dptr + (pitch * ((surface_h + 1) & !1)) as u64;
            copy(uv, (left as usize & !1) * bps, top as usize / 2, &mut chroma, w.div_ceil(2) * 2 * bps, h.div_ceil(2))
        });
        (self.cuvid.unmap)(self.decoder, dptr);
        r?;

        let sample = |b: &[u8], i: usize| -> u16 {
            if deep {
                // P016: the sample sits in the high bits.
                u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) >> (8 - fmt.bit_depth_luma_minus8)
            } else {
                b[i] as u16
            }
        };
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut u = Vec::with_capacity(cw * ch);
        let mut v = Vec::with_capacity(cw * ch);
        for i in 0..cw * ch {
            u.push(sample(&chroma, 2 * i));
            v.push(sample(&chroma, 2 * i + 1));
        }
        let vsd = fmt.video_signal_description;
        let full_range = vsd[0] & 0x08 != 0;
        Ok(DecodedFrame {
            width: w as u32,
            height: h as u32,
            bit_depth: 8 + fmt.bit_depth_luma_minus8,
            chroma: ChromaFormat::Yuv420,
            y: (0..w * h).map(|i| sample(&luma, i)).collect(),
            u,
            v,
            color: ColorInfo::from_vui(Some(vsd[3] as u32), full_range, h as u32),
        })
    }

    fn picture(&mut self, index: c_int) -> c_int {
        match unsafe { self.copy_out(index) } {
            Ok(f) => {
                self.ready.push((self.pts + self.copied, f));
                self.copied += 1;
                1
            }
            Err(e) => self.fail(e),
        }
    }
}

unsafe extern "C" fn on_sequence(user: *mut c_void, fmt: *mut VideoFormat) -> c_int {
    let s = &mut *(user as *mut State);
    let f = *fmt;
    if f.chroma_format != CHROMA_420 {
        return s.fail(format!("NVDEC path handles 4:2:0 only (chroma format {})", f.chroma_format));
    }
    let surfaces = (f.min_num_decode_surfaces as c_ulong).max(1) + 4;
    if !s.decoder.is_null() {
        (s.cuvid.destroy_decoder)(s.decoder);
        s.decoder = std::ptr::null_mut();
    }
    let mut ci: CreateInfo = std::mem::zeroed();
    ci.width = f.coded_width as c_ulong;
    ci.height = f.coded_height as c_ulong;
    ci.max_width = ci.width;
    ci.max_height = ci.height;
    ci.num_decode_surfaces = surfaces;
    ci.codec_type = f.codec;
    ci.chroma_format = f.chroma_format;
    ci.creation_flags = CREATE_PREFER_CUVID;
    ci.bit_depth_minus8 = f.bit_depth_luma_minus8 as c_ulong;
    ci.output_format = if f.bit_depth_luma_minus8 > 0 { SURFACE_P016 } else { SURFACE_NV12 };
    // Copies cover the display area themselves; the output surface is the
    // whole coded picture so the offsets mean the same on every driver.
    ci.display_area = [0, 0, f.coded_width as c_short, f.coded_height as c_short];
    ci.target_width = ci.width;
    ci.target_height = ci.height;
    ci.num_output_surfaces = 2;
    let r = (s.cuvid.create_decoder)(&mut s.decoder, &mut ci);
    if r != 0 {
        return s.fail(format!("cuvidCreateDecoder failed (CUDA error {r}): {}x{} codec {}", f.coded_width, f.coded_height, f.codec));
    }
    s.format = Some(f);
    surfaces as c_int
}

unsafe extern "C" fn on_decode(user: *mut c_void, pic: *mut c_void) -> c_int {
    let s = &mut *(user as *mut State);
    if s.decoder.is_null() {
        return s.fail("a picture before the decoder was created".into());
    }
    let r = (s.cuvid.decode_picture)(s.decoder, pic);
    if r != 0 {
        return s.fail(format!("cuvidDecodePicture failed (CUDA error {r})"));
    }
    if s.on_display {
        return 1;
    }
    let index = *(pic as *const u8).add(PICPARAMS_CURR_PIC_IDX).cast::<c_int>();
    s.picture(index)
}

unsafe extern "C" fn on_display(user: *mut c_void, info: *mut DispInfo) -> c_int {
    let s = &mut *(user as *mut State);
    if info.is_null() || !s.on_display {
        return 1;
    }
    s.picture((*info).picture_index)
}

/// A hardware decoder on an NVIDIA GPU's NVDEC engine, with the interface of
/// [`NativeDecoder`](crate::decode::NativeDecoder): open it from a demuxed
/// track, push samples in decode order, take pictures by presentation time.
///
/// H.264, HEVC (8- and 10-bit), VP9 and AV1, in 4:2:0, as far as the GPU's
/// generation decodes them (AV1 from Ampere on).
pub struct NvdecDecoder {
    // Field order is drop order: parser and decoder before the context.
    parser: Parser,
    state: Box<State>,
    ctx: Context,
    codec: TrackCodec,
    nal_len: usize,
    /// Parameter sets (or AV1 config OBUs) to put in front of the first sample.
    prefix: Option<Vec<u8>>,
    width: u32,
    height: u32,
    buf: Vec<u8>,
}

impl NvdecDecoder {
    /// Open a session for `track`'s codec and configuration.
    pub fn new(track: &VideoTrack) -> Result<Self, String> {
        let cuvid = cuvid().ok_or("no NVIDIA GPU or driver with NVDEC")?;
        let ctx = Context::new()?;
        let (codec_type, nal_len, prefix) = match track.codec {
            TrackCodec::H264 => {
                let (sets, n) = crate::decode::split_avcc(&track.config)?;
                (CODEC_H264, n, Some(annexb(&sets)))
            }
            TrackCodec::Hevc => {
                let (sets, n) = crate::decode::split_hvcc(&track.config)?;
                (CODEC_HEVC, n, Some(annexb(&sets)))
            }
            TrackCodec::Vp9 => (CODEC_VP9, 0, None),
            // `av1C`: four bytes of header, then the configuration OBUs
            // (normally the sequence header).
            TrackCodec::Av1 => (CODEC_AV1, 0, track.config.get(4..).filter(|o| !o.is_empty()).map(<[u8]>::to_vec)),
        };
        let mut state = Box::new(State {
            cuvid,
            decoder: std::ptr::null_mut(),
            format: None,
            on_display: matches!(track.codec, TrackCodec::Vp9 | TrackCodec::Av1),
            pts: 0,
            copied: 0,
            ready: Vec::new(),
            error: None,
        });
        let mut params: ParserParams = unsafe { std::mem::zeroed() };
        params.codec_type = codec_type;
        params.max_num_decode_surfaces = 1;
        params.clock_rate = 0;
        params.max_display_delay = 0;
        params.user_data = (&mut *state as *mut State).cast();
        params.sequence = Some(on_sequence);
        params.decode = Some(on_decode);
        params.display = Some(on_display);
        let mut parser = std::ptr::null_mut();
        ctx.with(|| check("cuvidCreateVideoParser", unsafe { (cuvid.create_parser)(&mut parser, &mut params) }))?;
        Ok(NvdecDecoder {
            parser,
            state,
            ctx,
            codec: track.codec,
            nal_len,
            prefix,
            width: track.width,
            height: track.height,
            buf: Vec::new(),
        })
    }

    /// Size of the track, as the container states it.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Hand one coded frame to the decoder, in decode order: an MP4 sample
    /// (length-prefixed NAL units), a VP9 frame or superframe, or an AV1
    /// temporal unit. Its picture is copied to the host before this returns.
    pub fn push(&mut self, sample: &[u8], pts: i64, _dts: i64, _duration: i64) -> Result<(), String> {
        self.buf.clear();
        if let Some(p) = self.prefix.take() {
            self.buf.extend_from_slice(&p);
        }
        match self.codec {
            TrackCodec::H264 | TrackCodec::Hevc => {
                for nal in crate::decode::bits::split_length_prefixed(sample, self.nal_len)? {
                    self.buf.extend_from_slice(&[0, 0, 0, 1]);
                    self.buf.extend_from_slice(nal);
                }
            }
            TrackCodec::Vp9 | TrackCodec::Av1 => self.buf.extend_from_slice(sample),
        }
        self.state.pts = pts;
        self.state.copied = 0;
        let mut pkt = Packet {
            flags: PKT_TIMESTAMP | PKT_ENDOFPICTURE,
            payload_size: self.buf.len() as c_ulong,
            payload: self.buf.as_ptr(),
            timestamp: pts,
        };
        let cuvid = self.state.cuvid;
        let parser = self.parser;
        let r = self.ctx.with(|| unsafe { (cuvid.parse)(parser, &mut pkt) });
        if let Some(e) = self.state.error.take() {
            return Err(e);
        }
        check("cuvidParseVideoData", r)
    }

    /// The picture for presentation time `pts` as tightly packed BGRA, if it
    /// has been decoded. Frames before `pts` are dropped on the way past.
    pub fn take(&mut self, pts: i64) -> Option<Vec<u8>> {
        self.take_frame(pts).map(|f| f.to_bgra())
    }

    /// [`take`](Self::take), but the planes at the stream's bit depth.
    pub fn take_frame(&mut self, pts: i64) -> Option<DecodedFrame> {
        let ready = &mut self.state.ready;
        let hit = ready.iter().position(|(p, _)| *p == pts)?;
        let (_, frame) = ready.remove(hit);
        ready.retain(|(p, _)| *p > pts);
        Some(frame)
    }
}

impl Drop for NvdecDecoder {
    fn drop(&mut self) {
        let cuvid = self.state.cuvid;
        let (parser, decoder) = (self.parser, self.state.decoder);
        self.ctx.with(|| unsafe {
            if !parser.is_null() {
                (cuvid.destroy_parser)(parser);
            }
            if !decoder.is_null() {
                (cuvid.destroy_decoder)(decoder);
            }
        });
    }
}

/// Parameter sets as an Annex B prefix.
fn annexb(sets: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for s in sets {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(s);
    }
    out
}
