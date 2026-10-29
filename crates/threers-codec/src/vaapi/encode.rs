//! HEVC encoding on a VA driver: RGBA frames in, an `hvc1` MP4 out.
//!
//! IDR and P frames, each P frame predicted from the one before, so every
//! frame pushed comes back as one access unit before `push_rgba` returns.
//! The driver writes the parameter sets and slice headers itself; they are
//! lifted out of the stream into `hvcC`, as the NVENC path does.
//!
//! The struct layouts are libva's `va_enc_hevc.h`, written as zeroed blocks
//! of the C size with fields at the offsets the C compiler reported
//! ([`super::enc_layout`]).

use std::ffi::{c_int, c_uint, c_void};

use super::enc_layout::*;
use super::{ConfigAttrib, Device, Id, Image, ImageFormat, ATTRIB_RT_FORMAT, ENTRYPOINT_ENC_SLICE, ENTRYPOINT_ENC_SLICE_LP, INVALID_ID, PROFILE_HEVC_MAIN, PROGRESSIVE, RT_FORMAT_YUV420};
use crate::hevc::hvcc::{build_hvcc, HvccArray, HvccProfile};

const ATTRIB_RATE_CONTROL: c_int = 5;
const ATTRIB_ENC_PACKED_HEADERS: c_int = 10;
const ATTRIB_PREDICTION_DIRECTION: c_int = 39;
const ATTRIB_ENC_HEVC_FEATURES: c_int = 50;
const ATTRIB_ENC_HEVC_BLOCK_SIZES: c_int = 51;
const ATTRIB_NOT_SUPPORTED: c_uint = 0x8000_0000;
/// `VA_PREDICTION_DIRECTION_BI_NOT_EMPTY`: B slices must have a list 1.
const PREDICTION_BI_NOT_EMPTY: c_uint = 4;
const RC_CQP: c_uint = 0x10;
const RC_VBR: c_uint = 0x4;
const BUF_ENC_CODED: c_int = 21;
const BUF_ENC_SEQUENCE: c_int = 22;
const BUF_ENC_PICTURE: c_int = 23;
const BUF_ENC_SLICE: c_int = 24;
const BUF_ENC_MISC: c_int = 27;
const BUF_PACKED_HEADER_PARAMS: c_int = 25;
const BUF_PACKED_HEADER_DATA: c_int = 26;
const PACKED_SEQUENCE: u32 = 1;
const PACKED_SLICE: u32 = 3;
/// `VA_ENC_PACKED_HEADER_SEQUENCE | VA_ENC_PACKED_HEADER_SLICE`.
const PACKED_HEADERS: c_uint = 0x1 | 0x4;
/// `sizeof(VAEncPackedHeaderParameterBuffer)`.
const PACKED_PARAM_SIZE: usize = 28;
const PIC_INVALID: u32 = 0x01;
/// The largest CTB AMD's and Intel's encoders take, and the one they want.
const LOG2_CTB: u32 = 6;

/// How a [`VaapiEncoder`] spends bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VaapiRate {
    /// A fixed quantizer for every frame (0–51, lower is better).
    ConstQp(u32),
    /// Variable bitrate around this many bits per second.
    Bitrate(u32),
}

/// A zeroed C struct of a known size with fields written at known offsets.
struct Blob(Vec<u64>);

impl Blob {
    fn new(size: usize) -> Blob {
        Blob(vec![0; size.div_ceil(8)])
    }
    fn bytes(&mut self) -> &mut [u8] {
        let n = self.0.len() * 8;
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr().cast(), n) }
    }
    fn u8(&mut self, at: usize, v: u8) {
        self.bytes()[at] = v;
    }
    fn u16(&mut self, at: usize, v: u16) {
        self.bytes()[at..at + 2].copy_from_slice(&v.to_ne_bytes());
    }
    fn u32(&mut self, at: usize, v: u32) {
        self.bytes()[at..at + 4].copy_from_slice(&v.to_ne_bytes());
    }
    fn i32(&mut self, at: usize, v: i32) {
        self.u32(at, v as u32);
    }
    /// A `VAPictureHEVC`: surface, POC, flags.
    fn picture(&mut self, at: usize, surface: Id, poc: i32, flags: u32) {
        self.u32(at, surface);
        self.i32(at + 4, poc);
        self.u32(at + 8, flags);
    }
    fn invalid_pictures(&mut self, at: usize, n: usize) {
        for i in 0..n {
            self.picture(at + 28 * i, INVALID_ID, 0, PIC_INVALID);
        }
    }
    fn ptr(&self) -> *const c_void {
        self.0.as_ptr().cast()
    }
}

/// A device with an HEVC encode entry point, and which one: the full
/// encoder first, Intel's low-power one (the only one on recent Intel GPUs)
/// after.
pub(crate) fn open_device() -> Result<(Device, c_int), String> {
    Device::open_for(PROFILE_HEVC_MAIN, ENTRYPOINT_ENC_SLICE)
        .map(|d| (d, ENTRYPOINT_ENC_SLICE))
        .or_else(|e| Device::open_for(PROFILE_HEVC_MAIN, ENTRYPOINT_ENC_SLICE_LP).map(|d| (d, ENTRYPOINT_ENC_SLICE_LP)).map_err(|_| e))
}

/// What the driver's encoder requires of the stream, which our headers and
/// the VA parameters must both state.
#[derive(Clone, Copy, Debug)]
struct Requirements {
    /// Asymmetric motion partitions are always on (Intel's VDEnc).
    amp: bool,
    /// `cu_qp_delta_enabled_flag` must be set (Intel's VDEnc).
    cu_qp_delta: bool,
    /// P frames go as B slices whose list 1 repeats list 0 ("low-delay B"),
    /// for encoders without P slices.
    low_delay_b: bool,
    /// `max_transform_hierarchy_depth_inter` / `_intra`: the largest the
    /// encoder allows, which is what it uses.
    depth_inter: u32,
    depth_intra: u32,
}

impl Requirements {
    fn query(dev: &Device, entrypoint: c_int) -> Requirements {
        let mut a = [
            ConfigAttrib { kind: ATTRIB_ENC_HEVC_FEATURES, value: 0 },
            ConfigAttrib { kind: ATTRIB_PREDICTION_DIRECTION, value: 0 },
            ConfigAttrib { kind: ATTRIB_ENC_HEVC_BLOCK_SIZES, value: 0 },
        ];
        let ok = unsafe { (dev.va.get_config_attributes)(dev.dpy, PROFILE_HEVC_MAIN, entrypoint, a.as_mut_ptr(), 3) } == 0;
        let known = |v: c_uint| ok && v != ATTRIB_NOT_SUPPORTED;
        // VAConfigAttribValEncHEVCFeatures: two bits a feature, 2 = required.
        let feature = |i: u32| if known(a[0].value) { (a[0].value >> (2 * i)) & 3 } else { 0 };
        Requirements {
            amp: feature(2) == 2,
            cu_qp_delta: feature(11) == 2,
            low_delay_b: known(a[1].value) && a[1].value & PREDICTION_BI_NOT_EMPTY != 0,
            // VAConfigAttribValEncHEVCBlockSizes, two bits a field:
            // max_max_transform_hierarchy_depth_inter is field 5, _intra 7.
            depth_inter: if known(a[2].value) { (a[2].value >> 10) & 3 } else { 0 },
            depth_intra: if known(a[2].value) { (a[2].value >> 14) & 3 } else { 0 },
        }
    }
}

/// A hardware HEVC encoder on a VA driver (AMD, Intel): RGBA frames in, an
/// `hvc1` MP4 out.
pub struct VaapiEncoder {
    dev: Device,
    req: Requirements,
    config: Id,
    context: Id,
    input: Id,
    /// Reconstructed pictures: the one being coded and its reference.
    recon: [Id; 2],
    coded: Id,
    image: Image,
    width: u32,
    height: u32,
    fps: u32,
    gop: u32,
    rate: VaapiRate,
    frame: u32,
    vps: Vec<u8>,
    sps: Vec<u8>,
    pps: Vec<u8>,
    samples: Vec<Vec<u8>>,
}

impl VaapiEncoder {
    /// Open an encoder for `width × height` frames (even sizes) at `fps`,
    /// rate-controlled by `rate`, an IDR frame every `keyframe_secs` seconds.
    pub fn new(width: u32, height: u32, fps: u32, rate: VaapiRate, keyframe_secs: u32) -> Result<Self, String> {
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(format!("VAAPI encodes even sizes; {width}x{height} is not"));
        }
        let (dev, entrypoint) = open_device()?;
        let req = Requirements::query(&dev, entrypoint);
        let (va, dpy) = (dev.va, dev.dpy);
        let fps = fps.max(1);
        unsafe {
            let mut attribs = [
                ConfigAttrib { kind: ATTRIB_RT_FORMAT, value: RT_FORMAT_YUV420 },
                ConfigAttrib {
                    kind: ATTRIB_RATE_CONTROL,
                    value: match rate {
                        VaapiRate::ConstQp(_) => RC_CQP,
                        VaapiRate::Bitrate(_) => RC_VBR,
                    },
                },
                // Parameter sets and slice headers come from us ([`headers`]):
                // drivers differ on whether they write their own (Mesa writes
                // none), and ours carry the colour description.
                ConfigAttrib { kind: ATTRIB_ENC_PACKED_HEADERS, value: PACKED_HEADERS },
            ];
            let mut config = 0;
            va.check("vaCreateConfig", (va.create_config)(dpy, PROFILE_HEVC_MAIN, entrypoint, attribs.as_mut_ptr(), attribs.len() as c_int, &mut config))?;
            // Surfaces padded to 16, as other VA clients allocate them; the
            // SPS crops to the picture.
            let (sw, sh) = (width.next_multiple_of(16), height.next_multiple_of(16));
            let mut surfaces = [INVALID_ID; 3];
            let r = (va.create_surfaces)(dpy, RT_FORMAT_YUV420, sw, sh, surfaces.as_mut_ptr(), 3, std::ptr::null_mut(), 0);
            if let Err(e) = va.check("vaCreateSurfaces", r) {
                (va.destroy_config)(dpy, config);
                return Err(e);
            }
            let mut recon = [surfaces[1], surfaces[2]];
            let mut context = 0;
            let r = (va.create_context)(dpy, config, sw as c_int, sh as c_int, PROGRESSIVE, recon.as_mut_ptr(), 2, &mut context);
            if let Err(e) = va.check("vaCreateContext", r) {
                (va.destroy_surfaces)(dpy, surfaces.as_mut_ptr(), 3);
                (va.destroy_config)(dpy, config);
                return Err(e);
            }
            let mut coded = INVALID_ID;
            let size = (sw * sh * 3 / 2 + (1 << 16)) as c_uint;
            let r = (va.create_buffer)(dpy, context, BUF_ENC_CODED, size, 1, std::ptr::null_mut(), &mut coded);
            let mut fmt: ImageFormat = std::mem::zeroed();
            fmt.fourcc = super::FOURCC_NV12;
            fmt.byte_order = 1;
            fmt.bits_per_pixel = 12;
            let mut image: Image = std::mem::zeroed();
            let r2 = if r == 0 { (va.create_image)(dpy, &mut fmt, width as c_int, height as c_int, &mut image) } else { r };
            if let Err(e) = va.check("vaCreateBuffer/vaCreateImage", r2) {
                if coded != INVALID_ID {
                    (va.destroy_buffer)(dpy, coded);
                }
                (va.destroy_context)(dpy, context);
                (va.destroy_surfaces)(dpy, surfaces.as_mut_ptr(), 3);
                (va.destroy_config)(dpy, config);
                return Err(e);
            }
            Ok(VaapiEncoder {
                dev,
                req,
                config,
                context,
                input: surfaces[0],
                recon,
                coded,
                image,
                width,
                height,
                fps,
                gop: (fps * keyframe_secs.max(1)).max(1),
                rate,
                frame: 0,
                vps: Vec::new(),
                sps: Vec::new(),
                pps: Vec::new(),
                samples: Vec::new(),
            })
        }
    }

    /// RGBA into the input surface as NV12, BT.709 limited range for HD and
    /// BT.601 below it: what decoders assume of a stream that does not say,
    /// and the driver's SPS does not.
    fn upload(&mut self, rgba: &[u8]) -> Result<(), String> {
        let (w, h) = (self.width as usize, self.height as usize);
        // Y = (kr·R + kg·G + kb·B) in 8.8 fixed point, limited range.
        let (yr, yg, yb, ur, ug, ub, vr, vg, vb) = if h >= 720 {
            (47, 157, 16, -26, -87, 112, 112, -102, -10)
        } else {
            (66, 129, 25, -38, -74, 112, 112, -94, -18)
        };
        let (va, dpy) = (self.dev.va, self.dev.dpy);
        unsafe {
            let mut p: *mut c_void = std::ptr::null_mut();
            va.check("vaMapBuffer", (va.map_buffer)(dpy, self.image.buf, &mut p))?;
            let data = std::slice::from_raw_parts_mut(p as *mut u8, self.image.data_size as usize);
            let (yp, up) = (self.image.pitches[0] as usize, self.image.pitches[1] as usize);
            let (yo, uo) = (self.image.offsets[0] as usize, self.image.offsets[1] as usize);
            for y in 0..h {
                let src = &rgba[y * w * 4..][..w * 4];
                let dst = &mut data[yo + y * yp..][..w];
                for (d, px) in dst.iter_mut().zip(src.chunks_exact(4)) {
                    let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                    *d = (((yr * r + yg * g + yb * b + 128) >> 8) + 16).clamp(0, 255) as u8;
                }
            }
            for cy in 0..h / 2 {
                let (r0, r1) = (&rgba[2 * cy * w * 4..][..w * 4], &rgba[(2 * cy + 1) * w * 4..][..w * 4]);
                let dst = &mut data[uo + cy * up..][..w];
                for (cx, d) in dst.chunks_exact_mut(2).enumerate() {
                    let mut s = [0i32; 3];
                    for row in [r0, r1] {
                        for px in row[cx * 8..cx * 8 + 8].chunks_exact(4) {
                            s[0] += px[0] as i32;
                            s[1] += px[1] as i32;
                            s[2] += px[2] as i32;
                        }
                    }
                    let (r, g, b) = ((s[0] + 2) >> 2, (s[1] + 2) >> 2, (s[2] + 2) >> 2);
                    d[0] = (((ur * r + ug * g + ub * b + 128) >> 8) + 128).clamp(0, 255) as u8;
                    d[1] = (((vr * r + vg * g + vb * b + 128) >> 8) + 128).clamp(0, 255) as u8;
                }
            }
            (va.unmap_buffer)(dpy, self.image.buf);
            let r = (va.put_image)(dpy, self.input, self.image.image_id, 0, 0, w as c_uint, h as c_uint, 0, 0, w as c_uint, h as c_uint);
            va.check("vaPutImage", r)
        }
    }

    fn buffer(&self, kind: c_int, blob: &Blob, size: usize) -> Result<Id, String> {
        let mut id = INVALID_ID;
        let r = unsafe { (self.dev.va.create_buffer)(self.dev.dpy, self.context, kind, size as c_uint, 1, blob.ptr() as *mut c_void, &mut id) };
        self.dev.va.check("vaCreateBuffer", r)?;
        Ok(id)
    }

    /// A packed header: its parameter buffer and its data buffer.
    fn packed(&self, kind: u32, nal: &[u8], ids: &mut Vec<Id>) -> Result<(), String> {
        let mut p = Blob::new(PACKED_PARAM_SIZE);
        p.u32(0, kind);
        p.u32(4, (nal.len() * 8) as u32);
        p.u8(8, 1); // has_emulation_bytes
        ids.push(self.buffer(BUF_PACKED_HEADER_PARAMS, &p, PACKED_PARAM_SIZE)?);
        let mut id = INVALID_ID;
        let r = unsafe { (self.dev.va.create_buffer)(self.dev.dpy, self.context, BUF_PACKED_HEADER_DATA, nal.len() as c_uint, 1, nal.as_ptr() as *mut c_void, &mut id) };
        self.dev.va.check("vaCreateBuffer", r)?;
        ids.push(id);
        Ok(())
    }

    fn qp(&self) -> u32 {
        match self.rate {
            VaapiRate::ConstQp(qp) => qp.min(51),
            VaapiRate::Bitrate(_) => 26,
        }
    }

    /// Encode one frame of tightly packed RGBA8, `width * height * 4` bytes.
    pub fn push_rgba(&mut self, rgba: &[u8]) -> Result<(), String> {
        let (w, h) = (self.width as usize, self.height as usize);
        if rgba.len() < w * h * 4 {
            return Err(format!("frame is {} bytes, want {}", rgba.len(), w * h * 4));
        }
        self.upload(&rgba[..w * h * 4])?;
        let poc_in_gop = self.frame % self.gop;
        let idr = poc_in_gop == 0;
        let cur = self.recon[(self.frame % 2) as usize];
        let reference = self.recon[((self.frame + 1) % 2) as usize];
        let qp = self.qp();
        let ctb = 1u32 << LOG2_CTB;
        let ctus = self.width.div_ceil(ctb) * self.height.div_ceil(ctb);
        let bitrate = match self.rate {
            VaapiRate::Bitrate(b) => b,
            VaapiRate::ConstQp(_) => 0,
        };

        let mut ids = Vec::new();
        if idr {
            let mut seq = Blob::new(SEQ_SIZE);
            seq.u8(SEQ_PROFILE, 1);
            seq.u8(SEQ_LEVEL, level_idc(self.width, self.height, self.fps));
            seq.u32(SEQ_INTRA_PERIOD, self.gop);
            seq.u32(SEQ_IDR_PERIOD, self.gop);
            seq.u32(SEQ_IP_PERIOD, 1);
            seq.u32(SEQ_BPS, bitrate);
            seq.u16(SEQ_WIDTH, self.width.next_multiple_of(16) as u16);
            seq.u16(SEQ_HEIGHT, self.height.next_multiple_of(16) as u16);
            seq.u32(SEQ_FIELDS, 1 << SEQ_BIT_CHROMA_FORMAT | 1 << SEQ_BIT_LOW_DELAY | (self.req.amp as u32) << SEQ_BIT_AMP);
            seq.u8(SEQ_LOG2_MIN_CB, 0);
            seq.u8(SEQ_LOG2_DIFF_CB, (LOG2_CTB - 3) as u8);
            seq.u8(SEQ_LOG2_MIN_TB, 0);
            seq.u8(SEQ_LOG2_DIFF_TB, 3);
            seq.u8(SEQ_DEPTH_INTER, self.req.depth_inter as u8);
            seq.u8(SEQ_DEPTH_INTRA, self.req.depth_intra as u8);
            ids.push(self.buffer(BUF_ENC_SEQUENCE, &seq, SEQ_SIZE)?);
            if let VaapiRate::Bitrate(bps) = self.rate {
                let mut rc = Blob::new(MISC_HEADER_SIZE + MISC_RC_SIZE);
                rc.u32(0, MISC_TYPE_RATE_CONTROL);
                rc.u32(MISC_HEADER_SIZE + RC_BPS, bps);
                rc.u32(MISC_HEADER_SIZE + RC_TARGET_PCT, 90);
                rc.u32(MISC_HEADER_SIZE + RC_WINDOW, 1000);
                ids.push(self.buffer(BUF_ENC_MISC, &rc, MISC_HEADER_SIZE + MISC_RC_SIZE)?);
                let mut fr = Blob::new(MISC_HEADER_SIZE + MISC_FRAMERATE_SIZE);
                fr.u32(0, MISC_TYPE_FRAME_RATE);
                fr.u32(MISC_HEADER_SIZE, self.fps | (1 << 16));
                ids.push(self.buffer(BUF_ENC_MISC, &fr, MISC_HEADER_SIZE + MISC_FRAMERATE_SIZE)?);
            }
        }

        let hdr = headers::Params {
            width: self.width,
            height: self.height,
            level_idc: level_idc(self.width, self.height, self.fps),
            init_qp: qp as i32,
            cu_qp_delta: self.req.cu_qp_delta || matches!(self.rate, VaapiRate::Bitrate(_)),
            amp: self.req.amp,
            low_delay_b: self.req.low_delay_b,
            depth_inter: self.req.depth_inter,
            depth_intra: self.req.depth_intra,
            matrix: if self.height >= 720 { 1 } else { 6 },
        };
        let mut pic = Blob::new(PIC_SIZE);
        pic.picture(PIC_CURR, cur, poc_in_gop as i32, 0);
        pic.invalid_pictures(PIC_REFS, 15);
        if !idr {
            pic.picture(PIC_REFS, reference, poc_in_gop as i32 - 1, 0);
        }
        pic.u32(PIC_CODED_BUF, self.coded);
        pic.u8(PIC_COLLOCATED, 0xff);
        pic.u8(PIC_INIT_QP, qp as u8);
        // QP can change in every 8×8 quantization group, as rate control
        // needs (and as other clients set it).
        if hdr.cu_qp_delta {
            pic.u8(PIC_DIFF_CU_QP, (LOG2_CTB - 3) as u8);
        }
        pic.u8(PIC_NAL_TYPE, if idr { 19 } else { 1 });
        let coding_type = if idr { 1 } else { 2 };
        let cu_qp = hdr.cu_qp_delta as u32;
        pic.u32(
            PIC_FIELDS,
            (idr as u32) << PIC_BIT_IDR | coding_type << PIC_BIT_CODING_TYPE | 1 << PIC_BIT_REFERENCE | cu_qp << PIC_BIT_CU_QP_DELTA | 1 << PIC_BIT_LF_ACROSS_SLICES,
        );
        ids.push(self.buffer(BUF_ENC_PICTURE, &pic, PIC_SIZE)?);

        // The order other VA clients use, which drivers expect: sequence,
        // picture, then the packed sequence headers, then each slice's packed
        // header before its parameters.
        if idr {
            let mut seq = headers::vps(&hdr);
            seq.extend(headers::sps(&hdr));
            seq.extend(headers::pps(&hdr));
            self.packed(PACKED_SEQUENCE, &seq, &mut ids)?;
        }
        self.packed(PACKED_SLICE, &headers::slice(&hdr, idr, poc_in_gop), &mut ids)?;
        let mut slice = Blob::new(SLICE_SIZE);
        slice.u32(SLICE_ADDRESS, 0);
        slice.u32(SLICE_NUM_CTU, ctus);
        // I, P, or for encoders without P slices B with both lists the
        // previous frame.
        slice.u8(SLICE_TYPE, if idr { 2 } else if self.req.low_delay_b { 0 } else { 1 });
        slice.invalid_pictures(SLICE_REF_L0, 15);
        slice.invalid_pictures(SLICE_REF_L1, 15);
        if !idr {
            slice.picture(SLICE_REF_L0, reference, poc_in_gop as i32 - 1, 0);
            if self.req.low_delay_b {
                slice.picture(SLICE_REF_L1, reference, poc_in_gop as i32 - 1, 0);
            }
        }
        slice.u8(SLICE_MAX_MERGE, 5);
        slice.u32(SLICE_FIELDS, 1 << SLICE_BIT_LAST | 1 << SLICE_BIT_LF_ACROSS_SLICES);
        ids.push(self.buffer(BUF_ENC_SLICE, &slice, SLICE_SIZE)?);

        let (va, dpy) = (self.dev.va, self.dev.dpy);
        let r = unsafe {
            va.check("vaBeginPicture", (va.begin_picture)(dpy, self.context, self.input))
                .and_then(|_| va.check("vaRenderPicture", (va.render_picture)(dpy, self.context, ids.as_mut_ptr(), ids.len() as c_int)))
                .and_then(|_| va.check("vaEndPicture", (va.end_picture)(dpy, self.context)))
                .and_then(|_| va.check("vaSyncSurface", (va.sync_surface)(dpy, self.input)))
        };
        for id in ids {
            unsafe { (va.destroy_buffer)(dpy, id) };
        }
        r?;
        let au = self.read_coded()?;
        self.collect(&au);
        self.frame += 1;
        Ok(())
    }

    /// The bytes in the coded buffer: a list of segments.
    fn read_coded(&self) -> Result<Vec<u8>, String> {
        let (va, dpy) = (self.dev.va, self.dev.dpy);
        let mut out = Vec::new();
        unsafe {
            let mut p: *mut c_void = std::ptr::null_mut();
            va.check("vaMapBuffer", (va.map_buffer)(dpy, self.coded, &mut p))?;
            let mut seg = p as *const u8;
            while !seg.is_null() {
                let size = *(seg.add(SEG_SIZE) as *const u32) as usize;
                let buf = *(seg.add(SEG_BUF) as *const *const u8);
                if !buf.is_null() && size > 0 {
                    out.extend_from_slice(std::slice::from_raw_parts(buf, size));
                }
                seg = *(seg.add(SEG_NEXT) as *const *const u8);
            }
            (va.unmap_buffer)(dpy, self.coded);
        }
        Ok(out)
    }

    /// Parameter sets kept for `hvcC`; everything else becomes the sample.
    fn collect(&mut self, au: &[u8]) {
        let mut sample = Vec::with_capacity(au.len());
        for nal in crate::h264::nal::split_annexb(au) {
            match nal.first().map(|b| (b >> 1) & 0x3f) {
                Some(32) if self.vps.is_empty() => self.vps = nal,
                Some(33) if self.sps.is_empty() => self.sps = nal,
                Some(34) if self.pps.is_empty() => self.pps = nal,
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
    }

    /// Finish and return the MP4 file.
    pub fn finish(self) -> Result<Vec<u8>, String> {
        if self.samples.is_empty() {
            return Err("no frames were pushed".into());
        }
        if self.vps.is_empty() || self.sps.is_empty() || self.pps.is_empty() {
            return Err("the VA driver did not emit HEVC parameter sets".into());
        }
        let profile = HvccProfile::from_sps(&self.sps).ok_or("the driver's SPS did not parse")?;
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

impl Drop for VaapiEncoder {
    fn drop(&mut self) {
        let (va, dpy) = (self.dev.va, self.dev.dpy);
        unsafe {
            (va.destroy_image)(dpy, self.image.image_id);
            (va.destroy_buffer)(dpy, self.coded);
            (va.destroy_context)(dpy, self.context);
            let mut s = [self.input, self.recon[0], self.recon[1]];
            (va.destroy_surfaces)(dpy, s.as_mut_ptr(), 3);
            (va.destroy_config)(dpy, self.config);
        }
    }
}

/// `general_level_idc` (30 × level) for a picture size and rate (Table A.8,
/// Main tier).
fn level_idc(w: u32, h: u32, fps: u32) -> u8 {
    let samples = w as u64 * h as u64;
    let rate = samples * fps as u64;
    let levels: [(u64, u64, u8); 8] = [
        (552_960, 16_588_800, 93),       // 3.1
        (2_228_224, 66_846_720, 120),    // 4
        (2_228_224, 133_693_440, 123),   // 4.1
        (8_912_896, 267_386_880, 150),   // 5
        (8_912_896, 534_773_760, 153),   // 5.1
        (8_912_896, 1_069_547_520, 156), // 5.2
        (35_651_584, 1_069_547_520, 180),
        (35_651_584, 2_139_095_040, 183),
    ];
    levels.iter().find(|(s, r, _)| samples <= *s && rate <= *r).map_or(186, |l| l.2)
}

/// The VPS, SPS, PPS and slice headers, written to match exactly what the VA
/// parameters set up (a 64×64 CTB, 4–32 transforms of depth 0, no SAO, AMP
/// or temporal MV prediction; one reference), with a conformance window to
/// the picture's size and the colour description in the VUI.
mod headers {
    use crate::bitstream::{emulation_prevention, rbsp_trailing_bits, BitWriter};

    pub(super) struct Params {
        pub width: u32,
        pub height: u32,
        pub level_idc: u8,
        pub init_qp: i32,
        pub cu_qp_delta: bool,
        pub amp: bool,
        /// P frames coded as B slices with list 1 = list 0.
        pub low_delay_b: bool,
        pub depth_inter: u32,
        pub depth_intra: u32,
        /// `matrix_coefficients` (and primaries / transfer): 1 BT.709, 6 BT.601.
        pub matrix: u32,
    }

    /// `00 00 00 01`, the two-byte header, the payload with emulation
    /// prevention.
    fn nal(nal_type: u8, w: BitWriter) -> Vec<u8> {
        let mut out = vec![0, 0, 0, 1, nal_type << 1, 1];
        out.extend(emulation_prevention(&w.finish()));
        out
    }

    fn profile_tier_level(w: &mut BitWriter, level_idc: u8) {
        w.write_bits(0, 2); // general_profile_space
        w.flag(false); // general_tier_flag
        w.write_bits(1, 5); // general_profile_idc: Main
        w.write_bits(0x6000_0000, 32); // compatible with Main and Main 10
        w.flag(true); // progressive_source
        w.flag(false); // interlaced_source
        w.flag(false); // non_packed_constraint
        w.flag(true); // frame_only_constraint
        w.write_bits64(0, 44);
        w.write_bits(level_idc as u32, 8);
    }

    pub(super) fn vps(p: &Params) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(0, 4); // vps_video_parameter_set_id
        w.flag(true); // vps_base_layer_internal_flag
        w.flag(true); // vps_base_layer_available_flag
        w.write_bits(0, 6); // vps_max_layers_minus1
        w.write_bits(0, 3); // vps_max_sub_layers_minus1
        w.flag(true); // vps_temporal_id_nesting_flag
        w.write_bits(0xffff, 16);
        profile_tier_level(&mut w, p.level_idc);
        w.flag(true); // vps_sub_layer_ordering_info_present_flag
        w.write_ue(1); // vps_max_dec_pic_buffering_minus1
        w.write_ue(0); // vps_max_num_reorder_pics
        w.write_ue(0); // vps_max_latency_increase_plus1
        w.write_bits(0, 6); // vps_max_layer_id
        w.write_ue(0); // vps_num_layer_sets_minus1
        w.flag(false); // vps_timing_info_present_flag
        w.flag(false); // vps_extension_flag
        rbsp_trailing_bits(&mut w);
        nal(32, w)
    }

    pub(super) fn sps(p: &Params) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(0, 4); // sps_video_parameter_set_id
        w.write_bits(0, 3); // sps_max_sub_layers_minus1
        w.flag(true); // sps_temporal_id_nesting_flag
        profile_tier_level(&mut w, p.level_idc);
        w.write_ue(0); // sps_seq_parameter_set_id
        w.write_ue(1); // chroma_format_idc: 4:2:0
        // The coded size is the surface's (16-aligned); the window crops to
        // the picture. Mesa copies the window from here into the SPS it
        // writes, and always codes the surface's size.
        let (cw, ch) = (p.width.next_multiple_of(16), p.height.next_multiple_of(16));
        w.write_ue(cw);
        w.write_ue(ch);
        let crop = cw != p.width || ch != p.height;
        w.flag(crop); // conformance_window_flag
        if crop {
            // In chroma samples for 4:2:0.
            w.write_ue(0);
            w.write_ue((cw - p.width) / 2);
            w.write_ue(0);
            w.write_ue((ch - p.height) / 2);
        }
        w.write_ue(0); // bit_depth_luma_minus8
        w.write_ue(0); // bit_depth_chroma_minus8
        w.write_ue(4); // log2_max_pic_order_cnt_lsb_minus4: 8-bit POC LSB
        w.flag(true); // sps_sub_layer_ordering_info_present_flag
        w.write_ue(1); // sps_max_dec_pic_buffering_minus1
        w.write_ue(0); // sps_max_num_reorder_pics
        w.write_ue(0); // sps_max_latency_increase_plus1
        w.write_ue(0); // log2_min_luma_coding_block_size_minus3: 8×8
        w.write_ue(3); // log2_diff_max_min_luma_coding_block_size: 64×64 CTB
        w.write_ue(0); // log2_min_luma_transform_block_size_minus2: 4×4
        w.write_ue(3); // log2_diff_max_min_luma_transform_block_size: 32×32
        w.write_ue(p.depth_inter); // max_transform_hierarchy_depth_inter
        w.write_ue(p.depth_intra); // max_transform_hierarchy_depth_intra
        w.flag(false); // scaling_list_enabled_flag
        w.flag(p.amp); // amp_enabled_flag
        w.flag(false); // sample_adaptive_offset_enabled_flag
        w.flag(false); // pcm_enabled_flag
        w.write_ue(0); // num_short_term_ref_pic_sets: each slice codes its own
        w.flag(false); // long_term_ref_pics_present_flag
        w.flag(false); // sps_temporal_mvp_enabled_flag
        w.flag(false); // strong_intra_smoothing_enabled_flag
        w.flag(true); // vui_parameters_present_flag
        w.flag(false); // aspect_ratio_info_present_flag
        w.flag(false); // overscan_info_present_flag
        w.flag(true); // video_signal_type_present_flag
        w.write_bits(5, 3); // video_format: unspecified
        w.flag(false); // video_full_range_flag
        w.flag(true); // colour_description_present_flag
        w.write_bits(p.matrix, 8); // colour_primaries
        w.write_bits(p.matrix, 8); // transfer_characteristics
        w.write_bits(p.matrix, 8); // matrix_coeffs
        w.flag(false); // chroma_loc_info_present_flag
        w.flag(false); // neutral_chroma_indication_flag
        w.flag(false); // field_seq_flag
        w.flag(false); // frame_field_info_present_flag
        w.flag(false); // default_display_window_flag
        w.flag(false); // vui_timing_info_present_flag
        w.flag(false); // bitstream_restriction_flag
        w.flag(false); // sps_extension_present_flag
        rbsp_trailing_bits(&mut w);
        nal(33, w)
    }

    pub(super) fn pps(p: &Params) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_ue(0); // pps_pic_parameter_set_id
        w.write_ue(0); // pps_seq_parameter_set_id
        w.flag(false); // dependent_slice_segments_enabled_flag
        w.flag(false); // output_flag_present_flag
        w.write_bits(0, 3); // num_extra_slice_header_bits
        w.flag(false); // sign_data_hiding_enabled_flag
        w.flag(false); // cabac_init_present_flag
        w.write_ue(0); // num_ref_idx_l0_default_active_minus1
        w.write_ue(0); // num_ref_idx_l1_default_active_minus1
        w.write_se(p.init_qp - 26);
        w.flag(false); // constrained_intra_pred_flag
        w.flag(false); // transform_skip_enabled_flag
        w.flag(p.cu_qp_delta);
        if p.cu_qp_delta {
            w.write_ue(super::LOG2_CTB - 3); // diff_cu_qp_delta_depth: 8×8 groups
        }
        w.write_se(0); // pps_cb_qp_offset
        w.write_se(0); // pps_cr_qp_offset
        w.flag(false); // pps_slice_chroma_qp_offsets_present_flag
        w.flag(false); // weighted_pred_flag
        w.flag(false); // weighted_bipred_flag
        w.flag(false); // transquant_bypass_enabled_flag
        w.flag(false); // tiles_enabled_flag
        w.flag(false); // entropy_coding_sync_enabled_flag
        w.flag(true); // pps_loop_filter_across_slices_enabled_flag
        w.flag(false); // deblocking_filter_control_present_flag
        w.flag(false); // pps_scaling_list_data_present_flag
        w.flag(false); // lists_modification_present_flag
        w.write_ue(0); // log2_parallel_merge_level_minus2
        w.flag(false); // slice_segment_header_extension_present_flag
        w.flag(false); // pps_extension_present_flag
        rbsp_trailing_bits(&mut w);
        nal(34, w)
    }

    /// The slice segment header of a one-slice picture, through
    /// `byte_alignment()`; the encoder appends the slice data.
    pub(super) fn slice(p: &Params, idr: bool, poc: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.flag(true); // first_slice_segment_in_pic_flag
        if idr {
            w.flag(false); // no_output_of_prior_pics_flag
        }
        w.write_ue(0); // slice_pic_parameter_set_id
        w.write_ue(if idr { 2 } else if p.low_delay_b { 0 } else { 1 }); // slice_type: I, P or B
        if !idr {
            w.write_bits(poc & 0xff, 8); // slice_pic_order_cnt_lsb
            w.flag(false); // short_term_ref_pic_set_sps_flag
            // st_ref_pic_set(0): the previous picture, used.
            w.write_ue(1); // num_negative_pics
            w.write_ue(0); // num_positive_pics
            w.write_ue(0); // delta_poc_s0_minus1
            w.flag(true); // used_by_curr_pic_s0_flag
            w.flag(false); // num_ref_idx_active_override_flag: one in each list
            if p.low_delay_b {
                w.flag(false); // mvd_l1_zero_flag
            }
            w.write_ue(0); // five_minus_max_num_merge_cand
        }
        w.write_se(0); // slice_qp_delta
        // pps_loop_filter_across_slices_enabled_flag, deblocking on:
        w.flag(true); // slice_loop_filter_across_slices_enabled_flag
        // byte_alignment()
        w.write_bit(1);
        w.align_zero();
        nal(if idr { 19 } else { 1 }, w)
    }
}
