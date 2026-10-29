//! Hardware video *decoding* in-process, through VideoToolbox.
//!
//! The mirror of [`videotoolbox`](crate::videotoolbox), and there for the same
//! reason. Getting a clip *into* a render meant spawning ffmpeg and reading raw
//! frames off a pipe: 8 MB a frame at 1080p, 99 MB at 8K, written by one
//! process and read by another, plus a process boundary in the middle of the
//! frame loop. The system decoder is the same hardware block ffmpeg's
//! `h264_videotoolbox` reaches, and it will hand the frame over directly.
//!
//! Pair it with [`mp4_demux`](crate::mp4_demux), which says where each
//! coded frame is and carries the `avcC`/`hvcC` a session needs to start:
//!
//! ```no_run
//! # #[cfg(all(feature = "videotoolbox", target_os = "macos"))] {
//! use threers_codec::mp4_demux::demux_video;
//! use threers_codec::videotoolbox_decode::VideoToolboxDecoder;
//!
//! let file = std::fs::read("clip.mp4").unwrap();
//! let track = demux_video(&file).unwrap();
//! let mut dec = VideoToolboxDecoder::new(&track).unwrap();
//! for s in &track.samples {
//!     let bytes = &file[s.offset as usize..][..s.size as usize];
//!     dec.push(bytes, s.pts, s.dts, s.duration as i64).unwrap();
//!     // Decode order is not display order, so ask for a frame by its own
//!     // presentation time rather than assuming one came back per push.
//!     if let Some(bgra) = dec.take(s.pts) {
//!         let _ = bgra; // width * height * 4
//!     }
//! }
//! # }
//! ```
//!
//! Frames come back as BGRA because VideoToolbox will do the YUV conversion on
//! the way out if asked, and the hardware does it for nothing. Ask for the
//! planes instead and it is the caller writing a colour conversion.

#![cfg(all(
    feature = "videotoolbox",
    target_os = "macos"
))]

use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;

use crate::mp4_demux::{TrackCodec, VideoTrack};

type OSStatus = i32;
type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFStringRef = *const c_void;
type CFNumberRef = *const c_void;
type CVPixelBufferRef = *mut c_void;
type CMSampleBufferRef = *mut c_void;
type CMBlockBufferRef = *mut c_void;
type CMFormatDescriptionRef = *const c_void;
type VTDecompressionSessionRef = *mut c_void;

const K_CF_NUMBER_SINT32: i32 = 3;
/// `kCVPixelFormatType_32BGRA`.
const PIXEL_FORMAT_BGRA: i32 = 0x4247_5241;

#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

impl CMTime {
    fn new(value: i64, timescale: u32) -> Self {
        CMTime {
            value,
            timescale: timescale as i32,
            flags: 1, // kCMTimeFlags_Valid
            epoch: 0,
        }
    }
}

/// `kVTDecodeFrame_EnableTemporalProcessing`.
const DECODE_ENABLE_TEMPORAL: u32 = 1 << 3;

/// `CMSampleTimingInfo`. Without one of these a sample buffer carries no
/// timestamps, the callback gets `kCMTimeInvalid` back, and there is no way to
/// tell which picture is which — which matters the moment a stream has
/// B-frames, because then the order they decode in is not the order they show
/// in.
#[repr(C)]
#[derive(Clone, Copy)]
struct CMSampleTimingInfo {
    duration: CMTime,
    presentation: CMTime,
    decode: CMTime,
}

#[repr(C)]
struct VTDecompressionOutputCallbackRecord {
    callback: Option<
        extern "C" fn(*mut c_void, *mut c_void, OSStatus, u32, CVPixelBufferRef, CMTime, CMTime),
    >,
    ref_con: *mut c_void,
}

// One block per framework rather than four `#[link]` attributes on one. Both
// link the same libraries, but the split records which framework each symbol
// actually comes from — and a repeated attribute on a single item reads like a
// mistake.
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const c_void);
    fn CFNumberCreate(alloc: CFAllocatorRef, ty: i32, value: *const c_void) -> CFNumberRef;
    fn CFDictionaryCreate(
        alloc: CFAllocatorRef,
        keys: *const *const c_void,
        values: *const *const c_void,
        count: isize,
        key_cb: *const c_void,
        value_cb: *const c_void,
    ) -> CFDictionaryRef;
    static kCFAllocatorNull: CFAllocatorRef;
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
}

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        alloc: CFAllocatorRef,
        count: usize,
        pointers: *const *const u8,
        sizes: *const usize,
        nal_header_len: i32,
        out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;
    fn CMVideoFormatDescriptionCreateFromHEVCParameterSets(
        alloc: CFAllocatorRef,
        count: usize,
        pointers: *const *const u8,
        sizes: *const usize,
        nal_header_len: i32,
        extensions: CFDictionaryRef,
        out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;

    fn CMBlockBufferCreateWithMemoryBlock(
        alloc: CFAllocatorRef,
        memory: *mut c_void,
        length: usize,
        block_alloc: CFAllocatorRef,
        custom: *const c_void,
        offset: usize,
        data_len: usize,
        flags: u32,
        out: *mut CMBlockBufferRef,
    ) -> OSStatus;
    fn CMSampleBufferCreateReady(
        alloc: CFAllocatorRef,
        block: CMBlockBufferRef,
        format: CMFormatDescriptionRef,
        num_samples: isize,
        num_timings: isize,
        timings: *const CMSampleTimingInfo,
        num_sizes: isize,
        sizes: *const usize,
        out: *mut CMSampleBufferRef,
    ) -> OSStatus;
}

#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    fn VTDecompressionSessionCreate(
        alloc: CFAllocatorRef,
        format: CMFormatDescriptionRef,
        spec: CFDictionaryRef,
        dest_attrs: CFDictionaryRef,
        output: *const VTDecompressionOutputCallbackRecord,
        out: *mut VTDecompressionSessionRef,
    ) -> OSStatus;
    fn VTDecompressionSessionDecodeFrame(
        session: VTDecompressionSessionRef,
        sample: CMSampleBufferRef,
        flags: u32,
        source_ref_con: *mut c_void,
        info_out: *mut u32,
    ) -> OSStatus;
    fn VTDecompressionSessionWaitForAsynchronousFrames(
        session: VTDecompressionSessionRef,
    ) -> OSStatus;
    fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: CFStringRef;

    fn CVPixelBufferLockBaseAddress(pb: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pb: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(pb: CVPixelBufferRef) -> *mut c_void;
    fn CVPixelBufferGetBytesPerRow(pb: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetWidth(pb: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetHeight(pb: CVPixelBufferRef) -> usize;
}

fn cfnum(v: i32) -> CFNumberRef {
    unsafe { CFNumberCreate(ptr::null(), K_CF_NUMBER_SINT32, &v as *const i32 as *const c_void) }
}

/// What the decode callback leaves behind.
///
/// A queue and not a single frame, because the decoder hands pictures back in
/// the order it decoded them and a stream with B-frames does not decode in the
/// order it displays. The caller asks for a presentation time and takes it out
/// when it arrives.
#[derive(Default)]
struct Sink {
    /// `(presentation time, tightly packed BGRA)`, in whatever order they came.
    ready: Vec<(i64, Vec<u8>)>,
    err: Option<String>,
}

extern "C" fn on_decoded(
    ref_con: *mut c_void,
    _source: *mut c_void,
    status: OSStatus,
    _flags: u32,
    image: CVPixelBufferRef,
    pts: CMTime,
    _dur: CMTime,
) {
    if ref_con.is_null() {
        return;
    }
    let shared = unsafe { &*(ref_con as *const Mutex<Sink>) };
    let Ok(mut sink) = shared.lock() else { return };
    if status != 0 {
        sink.err = Some(format!("decode callback status {status}"));
        return;
    }
    if image.is_null() {
        // A dropped frame, not an error — the session simply had nothing to
        // hand back for this one.
        return;
    }
    unsafe {
        if CVPixelBufferLockBaseAddress(image, 1) != 0 {
            sink.err = Some("could not lock the decoded pixel buffer".into());
            return;
        }
        let w = CVPixelBufferGetWidth(image);
        let h = CVPixelBufferGetHeight(image);
        let stride = CVPixelBufferGetBytesPerRow(image);
        let base = CVPixelBufferGetBaseAddress(image) as *const u8;
        if !base.is_null() && w > 0 && h > 0 {
            let mut frame = vec![0u8; w * h * 4];
            // Row by row: the buffer is padded to the hardware's stride, and a
            // straight copy of `h * stride` would carry the padding into the
            // picture and shear it.
            for y in 0..h {
                let src = base.add(y * stride);
                let dst = frame.as_mut_ptr().add(y * w * 4);
                ptr::copy_nonoverlapping(src, dst, w * 4);
            }
            sink.ready.push((pts.value, frame));
        }
        CVPixelBufferUnlockBaseAddress(image, 1);
    }
}

/// Split an `avcC` into its SPS and PPS, and report the NAL length prefix.
///
/// Layout is fixed: a version byte, three of profile/compatibility/level, then
/// `0xFC | (lengthSizeMinusOne)`, then `0xE0 | numSPS` and each set behind a
/// 16-bit length, then a count byte and the PPSs the same way.
fn split_avcc(cfg: &[u8]) -> Result<(Vec<(usize, usize)>, i32), String> {
    if cfg.len() < 7 {
        return Err("avcC too short".into());
    }
    let nal_len = (cfg[4] & 0x03) as i32 + 1;
    let mut sets = Vec::new();
    let mut at = 5usize;
    for round in 0..2 {
        let count = if round == 0 {
            let n = (cfg[at] & 0x1F) as usize;
            at += 1;
            n
        } else {
            if at >= cfg.len() {
                break;
            }
            let n = cfg[at] as usize;
            at += 1;
            n
        };
        for _ in 0..count {
            if at + 2 > cfg.len() {
                return Err("avcC ended inside a parameter set".into());
            }
            let len = u16::from_be_bytes([cfg[at], cfg[at + 1]]) as usize;
            at += 2;
            if at + len > cfg.len() {
                return Err("avcC parameter set runs past the box".into());
            }
            sets.push((at, len));
            at += len;
        }
    }
    if sets.is_empty() {
        return Err("avcC carried no parameter sets".into());
    }
    Ok((sets, nal_len))
}

/// Split an `hvcC` into its VPS/SPS/PPS arrays.
///
/// 22 bytes of fixed fields, then `lengthSizeMinusOne` in the low two bits of
/// byte 21, a count of arrays, and each array a type byte, a 16-bit NAL count
/// and then the NALs behind 16-bit lengths.
fn split_hvcc(cfg: &[u8]) -> Result<(Vec<(usize, usize)>, i32), String> {
    if cfg.len() < 23 {
        return Err("hvcC too short".into());
    }
    let nal_len = (cfg[21] & 0x03) as i32 + 1;
    let arrays = cfg[22] as usize;
    let mut sets = Vec::new();
    let mut at = 23usize;
    for _ in 0..arrays {
        if at + 3 > cfg.len() {
            return Err("hvcC ended inside an array header".into());
        }
        let count = u16::from_be_bytes([cfg[at + 1], cfg[at + 2]]) as usize;
        at += 3;
        for _ in 0..count {
            if at + 2 > cfg.len() {
                return Err("hvcC ended inside a parameter set".into());
            }
            let len = u16::from_be_bytes([cfg[at], cfg[at + 1]]) as usize;
            at += 2;
            if at + len > cfg.len() {
                return Err("hvcC parameter set runs past the box".into());
            }
            sets.push((at, len));
            at += len;
        }
    }
    if sets.is_empty() {
        return Err("hvcC carried no parameter sets".into());
    }
    Ok((sets, nal_len))
}

/// A VideoToolbox decompression session, fed one coded frame at a time.
pub struct VideoToolboxDecoder {
    session: VTDecompressionSessionRef,
    format: CMFormatDescriptionRef,
    sink: Box<Mutex<Sink>>,
    width: u32,
    height: u32,
    /// Kept alive for as long as the session: the sample buffers point into it.
    scratch: Vec<u8>,
    /// The track's, so timestamps mean the same thing here as in the demuxer.
    timescale: u32,
}

impl VideoToolboxDecoder {
    /// Open a session for `track`'s codec and configuration.
    pub fn new(track: &VideoTrack) -> Result<Self, String> {
        let (sets, nal_len) = match track.codec {
            TrackCodec::H264 => split_avcc(&track.config)?,
            TrackCodec::Hevc => split_hvcc(&track.config)?,
            TrackCodec::Vp9 | TrackCodec::Av1 => {
                return Err(format!("VideoToolbox decoding of {:?} is not wired up", track.codec))
            }
        };
        let pointers: Vec<*const u8> =
            sets.iter().map(|(o, _)| track.config[*o..].as_ptr()).collect();
        let sizes: Vec<usize> = sets.iter().map(|(_, l)| *l).collect();

        let mut format: CMFormatDescriptionRef = ptr::null();
        let st = unsafe {
            match track.codec {
                TrackCodec::H264 => CMVideoFormatDescriptionCreateFromH264ParameterSets(
                    ptr::null(),
                    pointers.len(),
                    pointers.as_ptr(),
                    sizes.as_ptr(),
                    nal_len,
                    &mut format,
                ),
                TrackCodec::Vp9 | TrackCodec::Av1 => unreachable!("refused above"),
                TrackCodec::Hevc => CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                    ptr::null(),
                    pointers.len(),
                    pointers.as_ptr(),
                    sizes.as_ptr(),
                    nal_len,
                    ptr::null(),
                    &mut format,
                ),
            }
        };
        if st != 0 || format.is_null() {
            return Err(format!("could not build a format description: status {st}"));
        }

        // Ask for BGRA out. The hardware does the YUV conversion on the way
        // past for nothing; not asking means writing one here.
        let sink = Box::new(Mutex::new(Sink::default()));
        let mut session: VTDecompressionSessionRef = ptr::null_mut();
        let st = unsafe {
            let key = kCVPixelBufferPixelFormatTypeKey;
            let val = cfnum(PIXEL_FORMAT_BGRA);
            let attrs = CFDictionaryCreate(
                ptr::null(),
                &key as *const CFStringRef,
                &val as *const CFNumberRef,
                1,
                &kCFTypeDictionaryKeyCallBacks as *const c_void,
                &kCFTypeDictionaryValueCallBacks as *const c_void,
            );
            let record = VTDecompressionOutputCallbackRecord {
                callback: Some(on_decoded),
                ref_con: &*sink as *const Mutex<Sink> as *mut c_void,
            };
            let st = VTDecompressionSessionCreate(
                ptr::null(),
                format,
                ptr::null(),
                attrs,
                &record,
                &mut session,
            );
            CFRelease(val);
            CFRelease(attrs);
            st
        };
        if st != 0 || session.is_null() {
            unsafe { CFRelease(format) };
            return Err(format!("VTDecompressionSessionCreate failed: status {st}"));
        }

        Ok(Self {
            session,
            format,
            sink,
            width: track.width,
            height: track.height,
            scratch: Vec::new(),
            timescale: track.timescale,
        })
    }

    /// Size of the frames this session returns.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Hand one coded frame to the decoder, in **decode order**.
    ///
    /// `sample` is the bytes the demuxer pointed at — length-prefixed NAL
    /// units, exactly as they sit in the file. The timestamps come from the
    /// demuxer too and are not optional: without them the decoder cannot say
    /// which picture it just produced, and a stream with B-frames comes back
    /// shuffled.
    ///
    /// Pictures go into an internal queue; take them with [`take`](Self::take).
    pub fn push(&mut self, sample: &[u8], pts: i64, dts: i64, duration: i64) -> Result<(), String> {
        // The block buffer borrows this rather than copying, so it has to
        // outlive the decode call — hence a field and not a local.
        self.scratch.clear();
        self.scratch.extend_from_slice(sample);
        {
            let mut s = self.sink.lock().map_err(|_| "decoder sink poisoned")?;
            s.err = None;
        }

        let mut block: CMBlockBufferRef = ptr::null_mut();
        let st = unsafe {
            CMBlockBufferCreateWithMemoryBlock(
                ptr::null(),
                self.scratch.as_mut_ptr() as *mut c_void,
                self.scratch.len(),
                // `kCFAllocatorNull`, and it has to be spelled out. The
                // parameter is the allocator CoreMedia will *deallocate* the
                // block with, and passing null there means the default one —
                // which then tries to free a Rust `Vec`'s buffer and aborts
                // the process. The null *allocator* is the way to say "this
                // memory is borrowed"; `scratch` outlives the call.
                kCFAllocatorNull,
                ptr::null(),
                0,
                self.scratch.len(),
                0,
                &mut block,
            )
        };
        if st != 0 || block.is_null() {
            return Err(format!("CMBlockBufferCreateWithMemoryBlock: status {st}"));
        }

        let timing = CMSampleTimingInfo {
            duration: CMTime::new(duration, self.timescale),
            presentation: CMTime::new(pts, self.timescale),
            decode: CMTime::new(dts, self.timescale),
        };
        let mut sbuf: CMSampleBufferRef = ptr::null_mut();
        let sizes = [self.scratch.len()];
        let st = unsafe {
            CMSampleBufferCreateReady(
                ptr::null(),
                block,
                self.format,
                1,
                1,
                &timing,
                1,
                sizes.as_ptr(),
                &mut sbuf,
            )
        };
        unsafe { CFRelease(block as *const c_void) };
        if st != 0 || sbuf.is_null() {
            return Err(format!("CMSampleBufferCreateReady: status {st}"));
        }

        let st = unsafe {
            let mut info = 0u32;
            // Temporal processing, so the decoder is allowed to hold a frame
            // back until the ones it depends on have arrived.
            let st = VTDecompressionSessionDecodeFrame(
                self.session,
                sbuf,
                DECODE_ENABLE_TEMPORAL,
                ptr::null_mut(),
                &mut info,
            );
            if st == 0 {
                VTDecompressionSessionWaitForAsynchronousFrames(self.session);
            }
            CFRelease(sbuf as *const c_void);
            st
        };
        if st != 0 {
            return Err(format!("VTDecompressionSessionDecodeFrame: status {st}"));
        }
        let s = self.sink.lock().map_err(|_| "decoder sink poisoned")?;
        if let Some(e) = &s.err {
            return Err(e.clone());
        }
        Ok(())
    }

    /// Take the picture for presentation time `pts`, if the decoder has
    /// produced it yet. Tightly packed BGRA, `width * height * 4`.
    ///
    /// Frames before `pts` are dropped on the way past: they have been shown.
    pub fn take(&mut self, pts: i64) -> Option<Vec<u8>> {
        let mut sink = self.sink.lock().ok()?;
        let hit = sink.ready.iter().position(|(p, _)| *p == pts)?;
        let (_, frame) = sink.ready.remove(hit);
        sink.ready.retain(|(p, _)| *p > pts);
        Some(frame)
    }

    /// How many decoded pictures are waiting to be taken.
    pub fn queued(&self) -> usize {
        self.sink.lock().map(|s| s.ready.len()).unwrap_or(0)
    }
}

impl Drop for VideoToolboxDecoder {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                VTDecompressionSessionInvalidate(self.session);
                CFRelease(self.session as *const c_void);
            }
            if !self.format.is_null() {
                CFRelease(self.format);
            }
        }
    }
}

// The session is only touched through `&mut self`, and the sink behind a mutex.
unsafe impl Send for VideoToolboxDecoder {}
