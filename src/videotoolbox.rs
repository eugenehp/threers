//! Hardware video encoding in-process, through VideoToolbox.
//!
//! The alternative is piping raw frames to ffmpeg, and at 8K that is the single
//! most expensive thing in a render: 99.5 MB a frame copied into a pipe and
//! copied out again, about 60 ms of pure memory traffic before any encoding
//! happens. The encoder itself needs roughly 44 ms. So more time goes on moving
//! the frame to the encoder than on encoding it.
//!
//! This removes the pipe. Frames go into a `CVPixelBuffer` and straight into a
//! `VTCompressionSession` — the same hardware block ffmpeg's `hevc_videotoolbox`
//! reaches, minus the process boundary.
//!
//! # What this is not
//!
//! It is not a muxer. VideoToolbox produces encoded frames, not files, and
//! writing a conformant MP4 is a separate job with its own edge cases. This
//! writes an Annex-B elementary stream, which any container can adopt without
//! re-encoding — `ffmpeg -i out.h265 -c copy out.mp4` is a remux, not a second
//! compression, and costs seconds for a whole film.
//!
//! # Why the C API and not AVFoundation
//!
//! `AVAssetWriter` would mux for us and is Objective-C, which would mean the
//! runtime, message sends and a class-lookup dance. VideoToolbox, CoreVideo and
//! CoreMedia are plain C: `extern "C"` declarations and nothing else in the
//! build. The same reasoning as the `metal` module — see its header.

#![cfg(all(feature = "videotoolbox", target_os = "macos"))]

use std::ffi::c_void;
use std::io::Write;
use std::ptr;

type OSStatus = i32;
type CFTypeRef = *const c_void;
type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFStringRef = *const c_void;
type CFNumberRef = *const c_void;
type CVPixelBufferRef = *mut c_void;
type CMSampleBufferRef = *mut c_void;
type CMBlockBufferRef = *mut c_void;
type CMFormatDescriptionRef = *const c_void;
type VTCompressionSessionRef = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

impl CMTime {
    fn frame(n: i64, fps: i32) -> Self {
        CMTime {
            value: n,
            timescale: fps,
            flags: 1, // kCMTimeFlags_Valid
            epoch: 0,
        }
    }
    fn invalid() -> Self {
        CMTime {
            value: 0,
            timescale: 0,
            flags: 0,
            epoch: 0,
        }
    }
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: CFTypeRef);
    fn CFStringCreateWithBytes(
        alloc: CFAllocatorRef,
        bytes: *const u8,
        len: isize,
        encoding: u32,
        external: u8,
    ) -> CFStringRef;
    fn CFNumberCreate(alloc: CFAllocatorRef, ty: i32, value: *const c_void) -> CFNumberRef;
    fn CFDictionaryCreate(
        alloc: CFAllocatorRef,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        count: isize,
        key_cb: *const c_void,
        value_cb: *const c_void,
    ) -> CFDictionaryRef;
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
    static kCFBooleanTrue: CFTypeRef;
    static kCFBooleanFalse: CFTypeRef;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    fn CVPixelBufferCreate(
        alloc: CFAllocatorRef,
        width: usize,
        height: usize,
        pixel_format: u32,
        attrs: CFDictionaryRef,
        out: *mut CVPixelBufferRef,
    ) -> i32;
    fn CVPixelBufferLockBaseAddress(pb: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pb: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(pb: CVPixelBufferRef) -> *mut c_void;
    fn CVPixelBufferGetBytesPerRow(pb: CVPixelBufferRef) -> usize;
}

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    fn CMSampleBufferGetDataBuffer(sbuf: CMSampleBufferRef) -> CMBlockBufferRef;
    fn CMBlockBufferGetDataLength(bbuf: CMBlockBufferRef) -> usize;
    fn CMBlockBufferCopyDataBytes(
        src: CMBlockBufferRef,
        offset: usize,
        len: usize,
        dst: *mut c_void,
    ) -> OSStatus;
    fn CMSampleBufferGetFormatDescription(sbuf: CMSampleBufferRef) -> CMFormatDescriptionRef;
    fn CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
        desc: CMFormatDescriptionRef,
        index: usize,
        out_ptr: *mut *const u8,
        out_size: *mut usize,
        out_count: *mut usize,
        out_nal_hdr_len: *mut i32,
    ) -> OSStatus;
}

type VTCompressionOutputCallback = extern "C" fn(
    output_ref_con: *mut c_void,
    source_frame_ref_con: *mut c_void,
    status: OSStatus,
    flags: u32,
    sample: CMSampleBufferRef,
);

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    fn VTCompressionSessionCreate(
        alloc: CFAllocatorRef,
        width: i32,
        height: i32,
        codec: u32,
        encoder_spec: CFDictionaryRef,
        source_attrs: CFDictionaryRef,
        compressed_alloc: CFAllocatorRef,
        cb: Option<VTCompressionOutputCallback>,
        cb_ref_con: *mut c_void,
        out: *mut VTCompressionSessionRef,
    ) -> OSStatus;
    fn VTCompressionSessionEncodeFrame(
        session: VTCompressionSessionRef,
        image: CVPixelBufferRef,
        pts: CMTime,
        duration: CMTime,
        frame_props: CFDictionaryRef,
        source_frame_ref_con: *mut c_void,
        info_flags_out: *mut u32,
    ) -> OSStatus;
    fn VTCompressionSessionCompleteFrames(
        session: VTCompressionSessionRef,
        complete_until: CMTime,
    ) -> OSStatus;
    fn VTCompressionSessionInvalidate(session: VTCompressionSessionRef);
    fn VTSessionSetProperty(
        session: VTCompressionSessionRef,
        key: CFStringRef,
        value: CFTypeRef,
    ) -> OSStatus;
}

const K_CV_PIXEL_FORMAT_TYPE_32BGRA: u32 = 0x42475241; // 'BGRA'
const K_CM_VIDEO_CODEC_TYPE_HEVC: u32 = 0x68766331; // 'hvc1'
const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
const K_CF_NUMBER_SINT32: i32 = 3;

fn cfstr(s: &str) -> CFStringRef {
    unsafe {
        CFStringCreateWithBytes(
            ptr::null(),
            s.as_ptr(),
            s.len() as isize,
            K_CF_STRING_ENCODING_UTF8,
            0,
        )
    }
}

fn cfnum(v: i32) -> CFNumberRef {
    unsafe {
        CFNumberCreate(
            ptr::null(),
            K_CF_NUMBER_SINT32,
            &v as *const i32 as *const c_void,
        )
    }
}

/// Encoded frames, collected by the VideoToolbox callback.
///
/// Behind a mutex because the callback does NOT run on the thread that called
/// `push`: VideoToolbox compresses asynchronously and calls back from its own
/// worker. Reading `err` from the encoding thread while the worker writes it is
/// a data race, and the kind that shows up as a corrupt file once a fortnight
/// rather than as a crash.
struct Sink {
    out: std::fs::File,
    frames: usize,
    bytes: usize,
    err: Option<String>,
    /// Whether VPS/SPS/PPS have been written yet.
    params_written: bool,
    /// Ring slots the encoder has not finished reading, one bit each.
    ///
    /// `VTCompressionSessionEncodeFrame` is ASYNCHRONOUS. It returns as soon as
    /// the frame is queued and the encoder reads the pixel buffer afterwards,
    /// on its own thread. Writing the next frame into the same buffer therefore
    /// overwrites a frame that is still being compressed -- row by row, so what
    /// comes out is bands of one frame displaced into the other. That is what
    /// the horizontal slabs across the Moon were.
    busy: u64,
}

extern "C" fn on_frame(
    output_ref_con: *mut c_void,
    _source: *mut c_void,
    status: OSStatus,
    _flags: u32,
    sample: CMSampleBufferRef,
) {
    if output_ref_con.is_null() {
        return;
    }
    let shared = unsafe { &*(output_ref_con as *const std::sync::Mutex<Sink>) };
    let Ok(mut sink) = shared.lock() else { return };
    // The encoder is done with this frame's pixel buffer, whatever else
    // happened: release the slot before any early return below.
    sink.busy &= !(1u64 << (_source as usize & 63));
    if status != 0 {
        sink.err = Some(format!("encode callback status {status}"));
        return;
    }
    if sample.is_null() {
        return;
    }
    unsafe {
        // PARAMETER SETS FIRST, and they do not arrive with the frames.
        //
        // VideoToolbox keeps VPS/SPS/PPS in the sample's format description,
        // not in its data, because the containers it is usually feeding carry
        // them out of band. An elementary stream has no out of band: without
        // these written ahead of the first slice a decoder has no dimensions,
        // no profile and no picture parameters, and says so — "PPS id out of
        // range" — which is what a stream that encodes fine and plays nowhere
        // looks like.
        if !sink.params_written {
            let desc = CMSampleBufferGetFormatDescription(sample);
            if !desc.is_null() {
                let mut count = 0usize;
                let mut ptr0: *const u8 = ptr::null();
                let mut size0 = 0usize;
                let mut hdr = 0i32;
                if CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                    desc, 0, &mut ptr0, &mut size0, &mut count, &mut hdr,
                ) == 0
                {
                    for i in 0..count {
                        let mut p: *const u8 = ptr::null();
                        let mut n = 0usize;
                        if CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                            desc,
                            i,
                            &mut p,
                            &mut n,
                            ptr::null_mut(),
                            ptr::null_mut(),
                        ) != 0
                            || p.is_null()
                        {
                            continue;
                        }
                        let set = std::slice::from_raw_parts(p, n);
                        if sink.out.write_all(&[0, 0, 0, 1]).is_err()
                            || sink.out.write_all(set).is_err()
                        {
                            sink.err = Some("write failed".into());
                            return;
                        }
                        sink.bytes += n + 4;
                    }
                    sink.params_written = true;
                }
            }
        }
        let bb = CMSampleBufferGetDataBuffer(sample);
        if bb.is_null() {
            return;
        }
        let len = CMBlockBufferGetDataLength(bb);
        let mut buf = vec![0u8; len];
        if CMBlockBufferCopyDataBytes(bb, 0, len, buf.as_mut_ptr() as *mut c_void) != 0 {
            sink.err = Some("CMBlockBufferCopyDataBytes failed".into());
            return;
        }
        // VideoToolbox emits length-prefixed NAL units (AVCC/HVCC style). An
        // elementary stream wants start codes, so rewrite the 4-byte lengths as
        // 00 00 00 01. Done here rather than later because the length prefix is
        // the only thing standing between this and a file ffmpeg can remux.
        let mut i = 0usize;
        while i + 4 <= len {
            let n = u32::from_be_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) as usize;
            if n == 0 || i + 4 + n > len {
                break;
            }
            if sink.out.write_all(&[0, 0, 0, 1]).is_err()
                || sink.out.write_all(&buf[i + 4..i + 4 + n]).is_err()
            {
                sink.err = Some("write failed".into());
                return;
            }
            sink.bytes += n + 4;
            i += 4 + n;
        }
        sink.frames += 1;
    }
}

/// An in-process hardware HEVC encoder.
pub struct VideoToolboxEncoder {
    session: VTCompressionSessionRef,
    /// A RING, not one buffer -- see `Sink::busy` for why one is wrong.
    pixbufs: Vec<CVPixelBufferRef>,
    next: usize,
    sink: Box<std::sync::Mutex<Sink>>,
    width: u32,
    height: u32,
    fps: i32,
    n: i64,
}

impl VideoToolboxEncoder {
    /// Open a session writing an Annex-B HEVC elementary stream to `path`.
    ///
    /// `quality` is 0..1, matching VideoToolbox's own scale rather than a
    /// bitrate: this is a quality-targeted encode, as the ffmpeg path is.
    pub fn new(
        path: &str,
        width: u32,
        height: u32,
        fps: i32,
        quality: f32,
    ) -> Result<Self, String> {
        let out = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
        let sink = Box::new(std::sync::Mutex::new(Sink {
            out,
            frames: 0,
            bytes: 0,
            err: None,
            params_written: false,
            busy: 0,
        }));

        let mut session: VTCompressionSessionRef = ptr::null_mut();
        // Ask for hardware explicitly. Without it VideoToolbox may silently pick
        // a software encoder, which is the one outcome that would make this
        // whole exercise pointless while still appearing to work.
        let spec = unsafe {
            let k = cfstr("EnableHardwareAcceleratedVideoEncoder");
            let keys = [k as CFTypeRef];
            let vals = [kCFBooleanTrue];
            let d = CFDictionaryCreate(
                ptr::null(),
                keys.as_ptr(),
                vals.as_ptr(),
                1,
                &kCFTypeDictionaryKeyCallBacks as *const _,
                &kCFTypeDictionaryValueCallBacks as *const _,
            );
            CFRelease(k);
            d
        };
        let status = unsafe {
            VTCompressionSessionCreate(
                ptr::null(),
                width as i32,
                height as i32,
                K_CM_VIDEO_CODEC_TYPE_HEVC,
                spec,
                ptr::null(),
                ptr::null(),
                Some(on_frame),
                &*sink as *const std::sync::Mutex<Sink> as *mut c_void,
                &mut session,
            )
        };
        if !spec.is_null() {
            unsafe { CFRelease(spec) };
        }
        if status != 0 || session.is_null() {
            return Err(format!("VTCompressionSessionCreate failed: {status}"));
        }

        unsafe {
            let set_num = |name: &str, v: i32| {
                let k = cfstr(name);
                let n = cfnum(v);
                VTSessionSetProperty(session, k, n);
                CFRelease(k);
                CFRelease(n);
            };
            let set_flag = |name: &str, on: bool| {
                let k = cfstr(name);
                VTSessionSetProperty(
                    session,
                    k,
                    if on { kCFBooleanTrue } else { kCFBooleanFalse },
                );
                CFRelease(k);
            };
            let set_bool = |name: &str| set_flag(name, true);
            // MAIN10, to match what the film is graded and delivered in. The
            // input stays 8-bit BGRA and VideoToolbox promotes it: the point of
            // 10 bits here is headroom in the ENCODE — the sky gradient bands
            // at 8-bit quantisation, not at 8-bit source.
            let k = cfstr("ProfileLevel");
            let v = cfstr("HEVC_Main10_AutoLevel");
            let st = VTSessionSetProperty(session, k, v as CFTypeRef);
            CFRelease(k);
            CFRelease(v);
            if st != 0 {
                // Not fatal — an older OS may not offer it — but say so rather
                // than silently deliver 8-bit where 10 was asked for.
                eprintln!("videotoolbox: Main10 unavailable ({st}), encoding 8-bit");
            }
            // COLOUR, STATED. Untagged, a player has to guess, and the guess
            // for 8K is not always BT.709 — the same reasoning as the explicit
            // colour flags on the ffmpeg path this replaces. Set on the session
            // so the tags travel in the bitstream rather than only in a
            // container that a remux might drop.
            for key in ["ColorPrimaries", "TransferFunction", "YCbCrMatrix"] {
                let k = cfstr(key);
                let v = cfstr("ITU_R_709_2");
                VTSessionSetProperty(session, k, v as CFTypeRef);
                CFRelease(k);
                CFRelease(v);
            }
            set_bool("RealTime");
            // NO B-FRAMES. The elementary stream this writes carries no
            // timestamps -- it is raw Annex-B -- so the remux reconstructs them
            // from a constant rate in the order the access units appear. With
            // reordering on, that order is DECODE order and the presentation
            // order differs, which tangles at the tail: the last frames arrive
            // with duplicate DTS and the muxer drops them. Measured before this:
            // a 120 frame render muxed to 118, a 240 to 238, always exactly the
            // last two. Reordering costs a little compression efficiency and
            // buys nothing here.
            set_flag("AllowFrameReordering", false);
            set_bool("PrioritizeEncodingSpeedOverQuality");
            set_num("MaxKeyFrameInterval", fps * 2);
            set_num("ExpectedFrameRate", fps);
            let k = cfstr("Quality");
            let q = quality.clamp(0.0, 1.0) as f64;
            // Quality wants a float; CFNumber type 6 is kCFNumberFloat64Type.
            let n = CFNumberCreate(ptr::null(), 6, &q as *const f64 as *const c_void);
            VTSessionSetProperty(session, k, n);
            CFRelease(k);
            CFRelease(n);
        }

        // A RING of pixel buffers, not one.
        //
        // One was the original design, on the reasoning that allocating per
        // frame would put a 132 MB allocation on the hot path and that the
        // encoder would be finished with the contents by the time the next
        // frame arrived. The first half is true; the second is not. Encoding is
        // asynchronous, so the next frame's write lands in a buffer the encoder
        // is still reading -- see `Sink::busy`.
        //
        // Four is enough to keep the writer ahead of a real-time encoder while
        // costing 531 MB at 8K, and `push_rgb` waits on the slot rather than
        // assuming, so the count is a throughput choice and not a correctness
        // one.
        const RING: usize = 4;
        let mut pixbufs: Vec<CVPixelBufferRef> = Vec::with_capacity(RING);
        for _ in 0..RING {
            let mut pb: CVPixelBufferRef = ptr::null_mut();
            let r = unsafe {
                CVPixelBufferCreate(
                    ptr::null(),
                    width as usize,
                    height as usize,
                    K_CV_PIXEL_FORMAT_TYPE_32BGRA,
                    ptr::null(),
                    &mut pb,
                )
            };
            if r != 0 || pb.is_null() {
                for old in pixbufs {
                    unsafe { CFRelease(old as CFTypeRef) };
                }
                return Err(format!("CVPixelBufferCreate failed: {r}"));
            }
            pixbufs.push(pb);
        }

        Ok(Self {
            session,
            pixbufs,
            next: 0,
            sink,
            width,
            height,
            fps,
            n: 0,
        })
    }

    /// Encode one frame of tightly-packed RGB (3 bytes per pixel).
    pub fn push_rgb(&mut self, rgb: &[u8]) -> Result<(), String> {
        let (w, h) = (self.width as usize, self.height as usize);
        if rgb.len() < w * h * 3 {
            return Err(format!("frame is {} bytes, want {}", rgb.len(), w * h * 3));
        }
        // Take the next ring slot and WAIT for the encoder to be finished with
        // it. Without this the write below lands in a buffer still being read.
        let slot = self.next % self.pixbufs.len();
        self.next = self.next.wrapping_add(1);
        let waited = std::time::Instant::now();
        loop {
            let g = self
                .sink
                .lock()
                .map_err(|_| "encoder state poisoned".to_string())?;
            if let Some(e) = &g.err {
                return Err(e.clone());
            }
            if g.busy & (1u64 << slot) == 0 {
                break;
            }
            drop(g);
            if waited.elapsed().as_secs_f32() > 30.0 {
                return Err(format!("encoder did not release slot {slot} in 30 s"));
            }
            std::thread::yield_now();
        }
        {
            let mut g = self
                .sink
                .lock()
                .map_err(|_| "encoder state poisoned".to_string())?;
            g.busy |= 1u64 << slot;
        }
        let pixbuf = self.pixbufs[slot];
        unsafe {
            CVPixelBufferLockBaseAddress(pixbuf, 0);
            let base = CVPixelBufferGetBaseAddress(pixbuf) as *mut u8;
            let stride = CVPixelBufferGetBytesPerRow(pixbuf);
            for y in 0..h {
                let src = &rgb[y * w * 3..y * w * 3 + w * 3];
                let dst = base.add(y * stride);
                for x in 0..w {
                    // BGRA from RGB.
                    *dst.add(x * 4) = src[x * 3 + 2];
                    *dst.add(x * 4 + 1) = src[x * 3 + 1];
                    *dst.add(x * 4 + 2) = src[x * 3];
                    *dst.add(x * 4 + 3) = 255;
                }
            }
            CVPixelBufferUnlockBaseAddress(pixbuf, 0);

            // The slot travels with the frame as its ref-con, so the callback
            // knows which buffer it has finished with.
            let st = VTCompressionSessionEncodeFrame(
                self.session,
                pixbuf,
                CMTime::frame(self.n, self.fps),
                CMTime::frame(1, self.fps),
                ptr::null(),
                slot as *mut c_void,
                ptr::null_mut(),
            );
            if st != 0 {
                if let Ok(mut g) = self.sink.lock() {
                    g.busy &= !(1u64 << slot);
                }
                return Err(format!("VTCompressionSessionEncodeFrame: {st}"));
            }
        }
        self.n += 1;
        if let Ok(g) = self.sink.lock() {
            if let Some(e) = &g.err {
                return Err(e.clone());
            }
        }
        Ok(())
    }

    /// Flush and close. Returns (frames, bytes written).
    pub fn finish(mut self) -> Result<(usize, usize), String> {
        unsafe {
            VTCompressionSessionCompleteFrames(self.session, CMTime::invalid());
            VTCompressionSessionInvalidate(self.session);
        }
        let r = {
            let mut g = self
                .sink
                .lock()
                .map_err(|_| "encoder state poisoned".to_string())?;
            g.out.flush().ok();
            if let Some(e) = &g.err {
                return Err(e.clone());
            }
            (g.frames, g.bytes)
        };
        self.session = ptr::null_mut();
        Ok(r)
    }
}

/// Safe to move between threads: everything it owns is a Core Foundation or
/// VideoToolbox object, none of which are thread-affine the way UI objects are,
/// and the shared state the async callback touches is behind a mutex. What is
/// NOT safe is using one from two threads at once, which `&mut self` on every
/// method already prevents — hence `Send` and not `Sync`.
unsafe impl Send for VideoToolboxEncoder {}

impl Drop for VideoToolboxEncoder {
    fn drop(&mut self) {
        unsafe {
            for pb in self.pixbufs.drain(..) {
                if !pb.is_null() {
                    CFRelease(pb as CFTypeRef);
                }
            }
            if !self.session.is_null() {
                VTCompressionSessionInvalidate(self.session);
            }
        }
    }
}
