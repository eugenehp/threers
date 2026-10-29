//! NVENC: HEVC from RGBA frames on an NVIDIA GPU's encode engine, muxed to
//! MP4 in process.
//!
//! The session runs synchronously with B-frames off, so every frame pushed
//! comes back as one access unit before `push_rgba` returns — the same
//! one-in, one-out shape as the native encoder, and the MP4 it writes needs
//! no composition offsets.

use std::ffi::{c_char, c_void, CStr};

use super::nvenc_layout::*;
use super::{Context, Lib};
use crate::hevc::hvcc::{build_hvcc, HvccArray, HvccProfile};

type Status = i32;
type Handle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct Guid(u32, u16, u16, [u8; 8]);

const CODEC_HEVC: Guid = Guid(0x790cdc88, 0x4522, 0x4d7b, [0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x03]);
const PRESET_P4: Guid = Guid(0x90a7b826, 0xdf06, 0x4862, [0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81]);
const PRESET_P2: Guid = Guid(0xf581cfb8, 0x88d6, 0x4381, [0x93, 0xf0, 0xdf, 0x13, 0xf9, 0xc2, 0x7d, 0xab]);

/// A zeroed C struct of a known size, with fields written at known offsets.
struct Blob(Vec<u64>);

impl Blob {
    fn new(size: usize, version: u32) -> Blob {
        let mut b = Blob(vec![0; size.div_ceil(8)]);
        b.u32(0, version);
        b
    }
    fn ptr(&mut self) -> *mut c_void {
        self.0.as_mut_ptr().cast()
    }
    fn bytes(&mut self) -> &mut [u8] {
        let n = self.0.len() * 8;
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr().cast(), n) }
    }
    fn u32(&mut self, at: usize, v: u32) {
        self.bytes()[at..at + 4].copy_from_slice(&v.to_ne_bytes());
    }
    fn u64(&mut self, at: usize, v: u64) {
        self.bytes()[at..at + 8].copy_from_slice(&v.to_ne_bytes());
    }
    fn set_ptr(&mut self, at: usize, p: *mut c_void) {
        self.u64(at, p as usize as u64);
    }
    fn guid(&mut self, at: usize, g: Guid) {
        let mut raw = [0u8; 16];
        raw[0..4].copy_from_slice(&g.0.to_ne_bytes());
        raw[4..6].copy_from_slice(&g.1.to_ne_bytes());
        raw[6..8].copy_from_slice(&g.2.to_ne_bytes());
        raw[8..].copy_from_slice(&g.3);
        self.bytes()[at..at + 16].copy_from_slice(&raw);
    }
    fn get_u32(&mut self, at: usize) -> u32 {
        u32::from_ne_bytes(self.bytes()[at..at + 4].try_into().unwrap())
    }
    fn get_ptr(&mut self, at: usize) -> *mut c_void {
        u64::from_ne_bytes(self.bytes()[at..at + 8].try_into().unwrap()) as usize as *mut c_void
    }
}

/// The function table `NvEncodeAPICreateInstance` fills.
struct Api {
    _lib: Lib,
    table: Blob,
}

impl Api {
    fn load() -> Result<Api, String> {
        let lib = Lib::open(&["libnvidia-encode.so.1", "libnvidia-encode.so", "nvEncodeAPI64.dll"])
            .ok_or("no NVENC library (libnvidia-encode) in this driver")?;
        unsafe {
            let max: unsafe extern "C" fn(*mut u32) -> Status =
                lib.sym("NvEncodeAPIGetMaxSupportedVersion").ok_or("NVENC library without a version query")?;
            let mut v = 0u32;
            max(&mut v);
            let have = ((v >> 4) & 0xfff, v & 0xf);
            if have < (API_VERSION & 0xff, API_VERSION >> 24) {
                return Err(format!(
                    "the driver's NVENC is API {}.{}; this needs 12.0 (driver 520 or newer)",
                    have.0, have.1
                ));
            }
            let create: unsafe extern "C" fn(*mut c_void) -> Status =
                lib.sym("NvEncodeAPICreateInstance").ok_or("NVENC library without NvEncodeAPICreateInstance")?;
            let mut table = Blob::new(FUNCTION_LIST_SIZE, FUNCTION_LIST_VER);
            let s = create(table.ptr());
            if s != 0 {
                return Err(format!("NvEncodeAPICreateInstance failed (NVENC status {s})"));
            }
            Ok(Api { _lib: lib, table })
        }
    }

    /// The entry at `offset` in the table, as a function of type `F`.
    unsafe fn f<F: Copy>(&mut self, offset: usize) -> F {
        let p = self.table.get_ptr(offset);
        assert!(!p.is_null(), "NVENC table entry {offset} is empty");
        std::mem::transmute_copy(&p)
    }
}

/// A hardware HEVC encoder on an NVIDIA GPU's NVENC engine: RGBA frames in,
/// an `hvc1` MP4 out.
pub struct NvencEncoder {
    api: Api,
    enc: Handle,
    input: Handle,
    output: Handle,
    // After the session, which uses it, in drop order.
    ctx: Context,
    width: u32,
    height: u32,
    fps: u32,
    frame: u32,
    vps: Vec<u8>,
    sps: Vec<u8>,
    pps: Vec<u8>,
    samples: Vec<Vec<u8>>,
}

/// How an [`NvencEncoder`] spends bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NvencRate {
    /// A fixed quantizer for every frame (0–51, lower is better).
    ConstQp(u32),
    /// Variable bitrate around this many bits per second.
    Bitrate(u32),
}

impl NvencEncoder {
    /// Open an encoder for `width × height` frames at `fps`, rate-controlled
    /// by `rate`, a keyframe every `keyframe_secs` seconds. `realtime` picks
    /// the low-latency tuning and a faster preset; otherwise high quality.
    pub fn new(width: u32, height: u32, fps: u32, rate: NvencRate, keyframe_secs: u32, realtime: bool) -> Result<Self, String> {
        let ctx = Context::new()?;
        let mut api = Api::load()?;
        let fps = fps.max(1);
        let mut enc: Handle = std::ptr::null_mut();
        unsafe {
            let open: unsafe extern "C" fn(*mut c_void, *mut Handle) -> Status = api.f(FN_OPEN_ENCODE_SESSION_EX);
            let mut p = Blob::new(OPEN_SIZE, OPEN_VER);
            p.u32(OPEN_DEVICE_TYPE, DEVICE_TYPE_CUDA);
            p.set_ptr(OPEN_DEVICE, ctx.raw());
            p.u32(OPEN_API_VERSION, API_VERSION);
            let s = open(p.ptr(), &mut enc);
            if s != 0 {
                return Err(format!("nvEncOpenEncodeSessionEx failed (NVENC status {s})"));
            }
        }
        let mut me = NvencEncoder {
            api,
            enc,
            input: std::ptr::null_mut(),
            output: std::ptr::null_mut(),
            ctx,
            width,
            height,
            fps,
            frame: 0,
            vps: Vec::new(),
            sps: Vec::new(),
            pps: Vec::new(),
            samples: Vec::new(),
        };
        me.initialize(rate, keyframe_secs, realtime)?;
        Ok(me)
    }

    fn status(&mut self, what: &str, s: Status) -> Result<(), String> {
        if s == 0 {
            return Ok(());
        }
        let detail = unsafe {
            let last: unsafe extern "C" fn(Handle) -> *const c_char = self.api.f(FN_GET_LAST_ERROR_STRING);
            let p = last(self.enc);
            if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
        };
        Err(format!("{what} failed (NVENC status {s}): {detail}"))
    }

    fn initialize(&mut self, rate: NvencRate, keyframe_secs: u32, realtime: bool) -> Result<(), String> {
        let (preset, tuning) = if realtime { (PRESET_P2, TUNING_LOW_LATENCY) } else { (PRESET_P4, TUNING_HIGH_QUALITY) };
        let gop = (self.fps * keyframe_secs.max(1)).max(1);
        unsafe {
            // The preset's configuration, then the few fields this changes.
            let get: unsafe extern "C" fn(Handle, Guid, Guid, u32, *mut c_void) -> Status = self.api.f(FN_GET_ENCODE_PRESET_CONFIG_EX);
            let mut preset_cfg = Blob::new(PRESET_SIZE, PRESET_VER);
            preset_cfg.u32(PRESET_CFG, CONFIG_VER);
            let s = get(self.enc, CODEC_HEVC, preset, tuning, preset_cfg.ptr());
            self.status("nvEncGetEncodePresetConfigEx", s)?;
            let mut cfg = Blob::new(CONFIG_SIZE, CONFIG_VER);
            cfg.bytes()[..CONFIG_SIZE].copy_from_slice(&preset_cfg.bytes()[PRESET_CFG..PRESET_CFG + CONFIG_SIZE]);
            cfg.u32(0, CONFIG_VER);
            cfg.u32(CONFIG_GOP_LENGTH, gop);
            cfg.u32(CONFIG_FRAME_INTERVAL_P, 1);
            cfg.u32(CONFIG_HEVC_IDR_PERIOD, gop);
            match rate {
                NvencRate::ConstQp(qp) => {
                    let qp = qp.min(51);
                    cfg.u32(CONFIG_RC_MODE, RC_CONSTQP);
                    for i in 0..3 {
                        cfg.u32(CONFIG_RC_CONST_QP + 4 * i, qp);
                    }
                }
                NvencRate::Bitrate(bps) => {
                    cfg.u32(CONFIG_RC_MODE, RC_VBR);
                    cfg.u32(CONFIG_RC_AVG_BITRATE, bps);
                    cfg.u32(CONFIG_RC_MAX_BITRATE, bps.saturating_mul(3) / 2);
                }
            }

            let init: unsafe extern "C" fn(Handle, *mut c_void) -> Status = self.api.f(FN_INITIALIZE_ENCODER);
            let mut p = Blob::new(INIT_SIZE, INIT_VER);
            p.guid(INIT_ENCODE_GUID, CODEC_HEVC);
            p.guid(INIT_PRESET_GUID, preset);
            p.u32(INIT_WIDTH, self.width);
            p.u32(INIT_HEIGHT, self.height);
            p.u32(INIT_DAR_WIDTH, self.width);
            p.u32(INIT_DAR_HEIGHT, self.height);
            p.u32(INIT_FRAME_RATE_NUM, self.fps);
            p.u32(INIT_FRAME_RATE_DEN, 1);
            p.u32(INIT_ENABLE_PTD, 1);
            p.set_ptr(INIT_ENCODE_CONFIG, cfg.ptr());
            p.u32(INIT_MAX_WIDTH, self.width);
            p.u32(INIT_MAX_HEIGHT, self.height);
            p.u32(INIT_TUNING_INFO, tuning);
            let s = init(self.enc, p.ptr());
            self.status("nvEncInitializeEncoder", s)?;

            let create_in: unsafe extern "C" fn(Handle, *mut c_void) -> Status = self.api.f(FN_CREATE_INPUT_BUFFER);
            let mut b = Blob::new(INPUT_SIZE, INPUT_VER);
            b.u32(INPUT_WIDTH, self.width);
            b.u32(INPUT_HEIGHT, self.height);
            b.u32(INPUT_FORMAT, BUFFER_FORMAT_ABGR);
            let s = create_in(self.enc, b.ptr());
            self.status("nvEncCreateInputBuffer", s)?;
            self.input = b.get_ptr(INPUT_BUFFER);

            let create_out: unsafe extern "C" fn(Handle, *mut c_void) -> Status = self.api.f(FN_CREATE_BITSTREAM_BUFFER);
            let mut b = Blob::new(BITSTREAM_SIZE, BITSTREAM_VER);
            let s = create_out(self.enc, b.ptr());
            self.status("nvEncCreateBitstreamBuffer", s)?;
            self.output = b.get_ptr(BITSTREAM_BUFFER);
        }
        Ok(())
    }

    /// Encode one frame of tightly packed RGBA8, `width * height * 4` bytes.
    pub fn push_rgba(&mut self, rgba: &[u8]) -> Result<(), String> {
        let (w, h) = (self.width as usize, self.height as usize);
        if rgba.len() < w * h * 4 {
            return Err(format!("frame is {} bytes, want {}", rgba.len(), w * h * 4));
        }
        unsafe {
            let lock: unsafe extern "C" fn(Handle, *mut c_void) -> Status = self.api.f(FN_LOCK_INPUT_BUFFER);
            let mut l = Blob::new(LOCK_INPUT_SIZE, LOCK_INPUT_VER);
            l.set_ptr(LOCK_INPUT_BUFFER, self.input);
            let s = lock(self.enc, l.ptr());
            self.status("nvEncLockInputBuffer", s)?;
            let base = l.get_ptr(LOCK_INPUT_DATA) as *mut u8;
            let pitch = l.get_u32(LOCK_INPUT_PITCH) as usize;
            // `ABGR` is a 32-bit word with R in the low byte: RGBA in memory.
            for y in 0..h {
                std::ptr::copy_nonoverlapping(rgba[y * w * 4..].as_ptr(), base.add(y * pitch), w * 4);
            }
            let unlock: unsafe extern "C" fn(Handle, Handle) -> Status = self.api.f(FN_UNLOCK_INPUT_BUFFER);
            let s = unlock(self.enc, self.input);
            self.status("nvEncUnlockInputBuffer", s)?;

            let encode: unsafe extern "C" fn(Handle, *mut c_void) -> Status = self.api.f(FN_ENCODE_PICTURE);
            let mut p = Blob::new(PIC_SIZE, PIC_VER);
            p.u32(PIC_INPUT_WIDTH, self.width);
            p.u32(PIC_INPUT_HEIGHT, self.height);
            p.u32(PIC_INPUT_PITCH, (pitch / 4) as u32);
            p.u32(PIC_FRAME_IDX, self.frame);
            p.u64(PIC_TIMESTAMP, self.frame as u64);
            p.set_ptr(PIC_INPUT_BUFFER, self.input);
            p.set_ptr(PIC_OUTPUT, self.output);
            p.u32(PIC_BUFFER_FORMAT, BUFFER_FORMAT_ABGR);
            p.u32(PIC_STRUCT, PIC_STRUCT_FRAME);
            let s = encode(self.enc, p.ptr());
            self.status("nvEncEncodePicture", s)?;
        }
        self.frame += 1;
        self.collect()
    }

    /// Read the access unit the last frame produced into the sample list.
    fn collect(&mut self) -> Result<(), String> {
        let au = unsafe {
            let lock: unsafe extern "C" fn(Handle, *mut c_void) -> Status = self.api.f(FN_LOCK_BITSTREAM);
            let mut l = Blob::new(LOCK_BS_SIZE, LOCK_BS_VER);
            l.set_ptr(LOCK_BS_OUTPUT, self.output);
            let s = lock(self.enc, l.ptr());
            self.status("nvEncLockBitstream", s)?;
            let n = l.get_u32(LOCK_BS_SIZE_BYTES) as usize;
            let p = l.get_ptr(LOCK_BS_PTR) as *const u8;
            let au = std::slice::from_raw_parts(p, n).to_vec();
            let unlock: unsafe extern "C" fn(Handle, Handle) -> Status = self.api.f(FN_UNLOCK_BITSTREAM);
            unlock(self.enc, self.output);
            au
        };
        let mut sample = Vec::with_capacity(au.len());
        for nal in crate::h264::nal::split_annexb(&au) {
            match nal.first().map(|b| (b >> 1) & 0x3f) {
                Some(32) if self.vps.is_empty() => self.vps = nal,
                Some(33) if self.sps.is_empty() => self.sps = nal,
                Some(34) if self.pps.is_empty() => self.pps = nal,
                // Repeated parameter sets live in `hvcC`, not the samples.
                Some(32..=34) => {}
                _ => {
                    sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                    sample.extend_from_slice(&nal);
                }
            }
        }
        if !sample.is_empty() {
            self.samples.push(sample);
        }
        Ok(())
    }

    /// Flush the encoder and return the MP4 file.
    pub fn finish(mut self) -> Result<Vec<u8>, String> {
        unsafe {
            let encode: unsafe extern "C" fn(Handle, *mut c_void) -> Status = self.api.f(FN_ENCODE_PICTURE);
            let mut p = Blob::new(PIC_SIZE, PIC_VER);
            p.u32(PIC_FLAGS, PIC_FLAG_EOS);
            let s = encode(self.enc, p.ptr());
            self.status("end of stream", s)?;
        }
        if self.samples.is_empty() {
            return Err("no frames were pushed".into());
        }
        if self.vps.is_empty() || self.sps.is_empty() || self.pps.is_empty() {
            return Err("NVENC did not emit HEVC parameter sets".into());
        }
        let profile = HvccProfile::from_sps(&self.sps).ok_or("NVENC's SPS did not parse")?;
        let (vps, sps, pps) = ([self.vps.clone()], [self.sps.clone()], [self.pps.clone()]);
        let hvcc = build_hvcc(
            &profile,
            &[
                HvccArray { nal_type: 32, complete: true, nals: &vps },
                HvccArray { nal_type: 33, complete: true, nals: &sps },
                HvccArray { nal_type: 34, complete: true, nals: &pps },
            ],
        );
        Ok(crate::mp4::mux_hevc(&crate::mp4::Mp4Params {
            width: self.width,
            height: self.height,
            timescale: self.fps * 1000,
            frame_duration: 1000,
            hvcc_payload: &hvcc,
            almo_payload: None,
            samples: &self.samples,
        }))
    }
}

impl Drop for NvencEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.input.is_null() {
                let f: unsafe extern "C" fn(Handle, Handle) -> Status = self.api.f(FN_DESTROY_INPUT_BUFFER);
                f(self.enc, self.input);
            }
            if !self.output.is_null() {
                let f: unsafe extern "C" fn(Handle, Handle) -> Status = self.api.f(FN_DESTROY_BITSTREAM_BUFFER);
                f(self.enc, self.output);
            }
            if !self.enc.is_null() {
                let f: unsafe extern "C" fn(Handle) -> Status = self.api.f(FN_DESTROY_ENCODER);
                f(self.enc);
            }
        }
        let _ = &self.ctx;
    }
}
