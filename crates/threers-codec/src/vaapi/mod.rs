//! VAAPI: hardware decoding and encoding on Linux GPUs with a VA driver —
//! AMD (Mesa `radeonsi`), Intel (`iHD`, `i965`) and others.
//!
//! [`VaapiDecoder`] decodes H.264, HEVC, VP9 and AV1; [`VaapiEncoder`]
//! encodes HEVC to MP4.
//!
//! `libva` and `libva-drm` are opened at run time, not linked, so a binary
//! built with the `vaapi` feature runs unchanged where there is no driver,
//! and [`available`] says so. Struct layouts are those of libva 2.x (VA-API
//! 1.x), checked against the C compiler's.
//!
//! VAAPI does not parse the bitstream: the caller supplies every picture's
//! parameters, reference frames and slice headers. So the native decoders do
//! that part — parameter sets, reference marking and lists, output order —
//! and hand each slice to the GPU through their accelerator hooks
//! ([`H264Accel`](crate::decode::h264::H264Accel) and its HEVC, VP9 and AV1
//! counterparts). The pictures that come back are the native decoder's,
//! sample for sample; only who computes them changes.
//!
//! [`decode::Decoder`](crate::decode::Decoder) uses this when it is available;
//! [`VaapiDecoder`] is the backend on its own.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::sync::OnceLock;

mod av1;
mod av1_layout;
mod encode;
mod enc_layout;
mod h264;
mod hevc;
mod vp9;

pub use encode::{VaapiEncoder, VaapiRate};

use crate::decode::{ChromaFormat, ColorInfo, DecodedFrame};
use crate::mp4_demux::{TrackCodec, VideoTrack};

pub(crate) type Display = *mut c_void;
pub(crate) type Status = c_int;
pub(crate) type Id = c_uint;

pub(crate) const INVALID_ID: Id = 0xffff_ffff;
pub(crate) const PROFILE_H264_HIGH: c_int = 7;
pub(crate) const PROFILE_HEVC_MAIN: c_int = 17;
pub(crate) const PROFILE_HEVC_MAIN10: c_int = 18;
pub(crate) const ENTRYPOINT_VLD: c_int = 1;
pub(crate) const ENTRYPOINT_ENC_SLICE: c_int = 6;
/// Intel's low-power (VDEnc) encoder.
pub(crate) const ENTRYPOINT_ENC_SLICE_LP: c_int = 8;
pub(crate) const ATTRIB_RT_FORMAT: c_int = 0;
pub(crate) const RT_FORMAT_YUV420: c_uint = 1;
pub(crate) const RT_FORMAT_YUV420_10: c_uint = 0x100;
pub(crate) const BUF_PICTURE_PARAMETER: c_int = 0;
pub(crate) const BUF_IQ_MATRIX: c_int = 1;
pub(crate) const BUF_SLICE_PARAMETER: c_int = 4;
pub(crate) const BUF_SLICE_DATA: c_int = 5;
pub(crate) const PROGRESSIVE: c_int = 1;
pub(crate) const FOURCC_NV12: u32 = 0x3231_564e;
const FOURCC_P010: u32 = 0x3031_3050;

/// `VAConfigAttrib`.
#[repr(C)]
pub(crate) struct ConfigAttrib {
    pub kind: c_int,
    pub value: c_uint,
}

/// `VAImageFormat`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ImageFormat {
    pub(crate) fourcc: u32,
    pub(crate) byte_order: u32,
    pub(crate) bits_per_pixel: u32,
    pub(crate) depth: u32,
    pub(crate) masks: [u32; 4],
    pub(crate) reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<ImageFormat>() == 48);

/// `VAImage`.
#[repr(C)]
pub(crate) struct Image {
    pub(crate) image_id: Id,
    pub(crate) format: ImageFormat,
    pub(crate) buf: Id,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) data_size: u32,
    pub(crate) num_planes: u32,
    pub(crate) pitches: [u32; 3],
    pub(crate) offsets: [u32; 3],
    pub(crate) num_palette_entries: i32,
    pub(crate) entry_bytes: i32,
    pub(crate) component_order: [i8; 4],
    pub(crate) reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<Image>() == 120);

/// The libva entry points used here.
pub(crate) struct Va {
    _lib: Lib,
    _drm: Lib,
    get_display_drm: unsafe extern "C" fn(c_int) -> Display,
    initialize: unsafe extern "C" fn(Display, *mut c_int, *mut c_int) -> Status,
    terminate: unsafe extern "C" fn(Display) -> Status,
    error_str: unsafe extern "C" fn(Status) -> *const c_char,
    set_info_callback: Option<unsafe extern "C" fn(Display, *mut c_void, *mut c_void) -> *mut c_void>,
    pub create_config: unsafe extern "C" fn(Display, c_int, c_int, *mut ConfigAttrib, c_int, *mut Id) -> Status,
    pub get_config_attributes: unsafe extern "C" fn(Display, c_int, c_int, *mut ConfigAttrib, c_int) -> Status,
    pub destroy_config: unsafe extern "C" fn(Display, Id) -> Status,
    pub create_surfaces: unsafe extern "C" fn(Display, c_uint, c_uint, c_uint, *mut Id, c_uint, *mut c_void, c_uint) -> Status,
    pub destroy_surfaces: unsafe extern "C" fn(Display, *mut Id, c_int) -> Status,
    pub create_context: unsafe extern "C" fn(Display, Id, c_int, c_int, c_int, *mut Id, c_int, *mut Id) -> Status,
    pub destroy_context: unsafe extern "C" fn(Display, Id) -> Status,
    pub create_buffer: unsafe extern "C" fn(Display, Id, c_int, c_uint, c_uint, *mut c_void, *mut Id) -> Status,
    pub destroy_buffer: unsafe extern "C" fn(Display, Id) -> Status,
    pub begin_picture: unsafe extern "C" fn(Display, Id, Id) -> Status,
    pub render_picture: unsafe extern "C" fn(Display, Id, *mut Id, c_int) -> Status,
    pub end_picture: unsafe extern "C" fn(Display, Id) -> Status,
    pub sync_surface: unsafe extern "C" fn(Display, Id) -> Status,
    derive_image: unsafe extern "C" fn(Display, Id, *mut Image) -> Status,
    pub create_image: unsafe extern "C" fn(Display, *mut ImageFormat, c_int, c_int, *mut Image) -> Status,
    get_image: unsafe extern "C" fn(Display, Id, c_int, c_int, c_uint, c_uint, Id) -> Status,
    pub put_image: unsafe extern "C" fn(Display, Id, Id, c_int, c_int, c_uint, c_uint, c_int, c_int, c_uint, c_uint) -> Status,
    pub destroy_image: unsafe extern "C" fn(Display, Id) -> Status,
    pub map_buffer: unsafe extern "C" fn(Display, Id, *mut *mut c_void) -> Status,
    pub unmap_buffer: unsafe extern "C" fn(Display, Id) -> Status,
}

/// A shared library opened at run time.
pub(crate) struct Lib(*mut c_void);
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

impl Lib {
    fn open(names: &[&str]) -> Option<Lib> {
        names.iter().find_map(|n| {
            let c = CString::new(*n).ok()?;
            // RTLD_NOW
            let h = unsafe { dlopen(c.as_ptr(), 2) };
            (!h.is_null()).then_some(Lib(h))
        })
    }

    /// # Safety
    /// `T` must be the symbol's true signature.
    unsafe fn sym<T: Copy>(&self, name: &str) -> Option<T> {
        let c = CString::new(name).ok()?;
        let p = dlsym(self.0, c.as_ptr());
        (!p.is_null()).then(|| std::mem::transmute_copy(&p))
    }
}

impl Va {
    fn load() -> Option<Va> {
        let lib = Lib::open(&["libva.so.2", "libva.so"])?;
        let drm = Lib::open(&["libva-drm.so.2", "libva-drm.so"])?;
        unsafe {
            Some(Va {
                get_display_drm: drm.sym("vaGetDisplayDRM")?,
                initialize: lib.sym("vaInitialize")?,
                terminate: lib.sym("vaTerminate")?,
                error_str: lib.sym("vaErrorStr")?,
                set_info_callback: lib.sym("vaSetInfoCallback"),
                create_config: lib.sym("vaCreateConfig")?,
                get_config_attributes: lib.sym("vaGetConfigAttributes")?,
                destroy_config: lib.sym("vaDestroyConfig")?,
                create_surfaces: lib.sym("vaCreateSurfaces")?,
                destroy_surfaces: lib.sym("vaDestroySurfaces")?,
                create_context: lib.sym("vaCreateContext")?,
                destroy_context: lib.sym("vaDestroyContext")?,
                create_buffer: lib.sym("vaCreateBuffer")?,
                destroy_buffer: lib.sym("vaDestroyBuffer")?,
                begin_picture: lib.sym("vaBeginPicture")?,
                render_picture: lib.sym("vaRenderPicture")?,
                end_picture: lib.sym("vaEndPicture")?,
                sync_surface: lib.sym("vaSyncSurface")?,
                derive_image: lib.sym("vaDeriveImage")?,
                create_image: lib.sym("vaCreateImage")?,
                get_image: lib.sym("vaGetImage")?,
                put_image: lib.sym("vaPutImage")?,
                destroy_image: lib.sym("vaDestroyImage")?,
                map_buffer: lib.sym("vaMapBuffer")?,
                unmap_buffer: lib.sym("vaUnmapBuffer")?,
                _lib: lib,
                _drm: drm,
            })
        }
    }

    pub(crate) fn check(&self, what: &str, s: Status) -> Result<(), String> {
        if s == 0 {
            return Ok(());
        }
        let msg = unsafe { CStr::from_ptr((self.error_str)(s)).to_string_lossy().into_owned() };
        Err(format!("{what} failed: {msg} (VA status {s})"))
    }
}

/// libva, loaded once, or `None` where it is not installed.
fn va() -> Option<&'static Va> {
    static VA: OnceLock<Option<Va>> = OnceLock::new();
    VA.get_or_init(Va::load).as_ref()
}

/// An initialised VA display on a DRM render node, terminated on drop.
pub(crate) struct Device {
    pub va: &'static Va,
    pub dpy: Display,
    _node: std::fs::File,
    /// The image frames are copied into for reading, with its size and
    /// format: created once, not per frame (each is megabytes of GPU memory
    /// and a round trip into the kernel).
    readback: std::cell::Cell<Option<(Image, usize, usize, u32)>>,
}

impl Device {
    /// The first render node whose driver initialises and decodes `profile`.
    pub(crate) fn open(profile: c_int) -> Result<Device, String> {
        Self::open_for(profile, ENTRYPOINT_VLD)
    }

    /// The first render node whose driver has `entrypoint` for `profile`.
    pub(crate) fn open_for(profile: c_int, entrypoint: c_int) -> Result<Device, String> {
        let va = va().ok_or("libva is not installed")?;
        let mut last = String::from("no DRM render node (/dev/dri/renderD*)");
        for n in 128..136 {
            let path = format!("/dev/dri/renderD{n}");
            let Ok(node) = std::fs::OpenOptions::new().read(true).write(true).open(&path) else { continue };
            use std::os::fd::AsRawFd;
            let dpy = unsafe { (va.get_display_drm)(node.as_raw_fd()) };
            if dpy.is_null() {
                continue;
            }
            // libva prints which driver it loads to stderr unless told not to;
            // a library has no business writing there.
            if let Some(f) = va.set_info_callback {
                unsafe { f(dpy, std::ptr::null_mut(), std::ptr::null_mut()) };
            }
            let (mut major, mut minor) = (0, 0);
            if let Err(e) = va.check(&format!("vaInitialize on {path}"), unsafe { (va.initialize)(dpy, &mut major, &mut minor) }) {
                last = e;
                continue;
            }
            let dev = Device { va, dpy, _node: node, readback: std::cell::Cell::new(None) };
            // Does this driver decode the profile at all?
            let mut attrib = ConfigAttrib { kind: ATTRIB_RT_FORMAT, value: RT_FORMAT_YUV420 };
            let mut cfg = 0;
            match va.check("vaCreateConfig", unsafe { (va.create_config)(dpy, profile, entrypoint, &mut attrib, 1, &mut cfg) }) {
                Ok(()) => {
                    unsafe { (va.destroy_config)(dpy, cfg) };
                    return Ok(dev);
                }
                Err(e) => last = format!("{path}: {e}"),
            }
        }
        Err(last)
    }

    /// `vaCreateImage` + `vaGetImage` into `img`; false (and nothing held)
    /// when the driver cannot.
    unsafe fn get_image(&self, surface: Id, w: usize, h: usize, fourcc: u32, deep: bool, img: &mut Image) -> bool {
        let va = self.va;
        match self.readback.take() {
            Some((cached, cw, ch, cf)) if (cw, ch, cf) == (w, h, fourcc) => {
                if (va.get_image)(self.dpy, surface, 0, 0, w as c_uint, h as c_uint, cached.image_id) == 0 {
                    *img = cached;
                    return true;
                }
                (va.destroy_image)(self.dpy, cached.image_id);
            }
            Some((stale, ..)) => {
                (va.destroy_image)(self.dpy, stale.image_id);
            }
            None => {}
        }
        let mut fmt: ImageFormat = std::mem::zeroed();
        fmt.fourcc = fourcc;
        fmt.byte_order = 1;
        fmt.bits_per_pixel = if deep { 24 } else { 12 };
        *img = std::mem::zeroed();
        if (va.create_image)(self.dpy, &mut fmt, w as c_int, h as c_int, img) != 0 {
            return false;
        }
        if (va.get_image)(self.dpy, surface, 0, 0, w as c_uint, h as c_uint, img.image_id) != 0 {
            (va.destroy_image)(self.dpy, img.image_id);
            *img = std::mem::zeroed();
            return false;
        }
        true
    }

    /// The decoded picture in `surface` (`w × h` coded, NV12, or P010 when
    /// `deep`), cropped by `crop` (`[left, right, top, bottom]`, luma
    /// samples) straight out of the mapped image, after waiting for it.
    pub(crate) fn read_frame(&self, surface: Id, w: usize, h: usize, deep: bool, crop: [usize; 4], color: ColorInfo) -> Result<DecodedFrame, String> {
        let fourcc = if deep { FOURCC_P010 } else { FOURCC_NV12 };
        let va = self.va;
        unsafe {
            va.check("vaSyncSurface", (va.sync_surface)(self.dpy, surface))?;
            // Copy the surface into an image first: mapping the surface itself
            // (vaDeriveImage) reads uncached video memory, which on Mesa costs
            // more than decoding the frame did. Derive only when the copy fails.
            let mut img: Image = std::mem::zeroed();
            let copied = self.get_image(surface, w, h, fourcc, deep, &mut img);
            if !copied {
                let derived = (va.derive_image)(self.dpy, surface, &mut img) == 0 && img.format.fourcc == fourcc;
                if !derived {
                    if img.image_id != 0 && img.image_id != INVALID_ID {
                        (va.destroy_image)(self.dpy, img.image_id);
                    }
                    return Err("the driver neither copies nor exposes the decoded surface".into());
                }
            }
            let mut p: *mut c_void = std::ptr::null_mut();
            let r = va.check("vaMapBuffer", (va.map_buffer)(self.dpy, img.buf, &mut p));
            if let Err(e) = r {
                (va.destroy_image)(self.dpy, img.image_id);
                return Err(e);
            }
            let data = std::slice::from_raw_parts(p as *const u8, img.data_size as usize);
            let (yp, up) = (img.pitches[0] as usize, img.pitches[1] as usize);
            let (yo, uo) = (img.offsets[0] as usize, img.offsets[1] as usize);
            let [left, right, top, bottom] = crop;
            let (fw, fh) = (w.saturating_sub(left + right), h.saturating_sub(top + bottom));
            let (cw, ch) = (fw.div_ceil(2), fh.div_ceil(2));
            let (cx, cy) = (left / 2, top / 2);
            // Straight loops per format, so they vectorise: this copy runs
            // once per frame over the whole picture.
            let bps = if deep { 2 } else { 1 };
            let mut y = vec![0u16; fw * fh];
            let mut u = vec![0u16; cw * ch];
            let mut v = vec![0u16; cw * ch];
            for r in 0..fh {
                let (src, dst) = (&data[yo + (top + r) * yp + left * bps..][..fw * bps], &mut y[r * fw..][..fw]);
                if deep {
                    // P010: 10-bit samples in the high bits of little-endian u16s.
                    for (d, s) in dst.iter_mut().zip(src.chunks_exact(2)) {
                        *d = u16::from_le_bytes([s[0], s[1]]) >> 6;
                    }
                } else {
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d = s as u16;
                    }
                }
            }
            for r in 0..ch {
                let src = &data[uo + (cy + r) * up + cx * 2 * bps..][..cw * 2 * bps];
                let (du, dv) = (&mut u[r * cw..][..cw], &mut v[r * cw..][..cw]);
                if deep {
                    for ((a, b), s) in du.iter_mut().zip(dv.iter_mut()).zip(src.chunks_exact(4)) {
                        *a = u16::from_le_bytes([s[0], s[1]]) >> 6;
                        *b = u16::from_le_bytes([s[2], s[3]]) >> 6;
                    }
                } else {
                    for ((a, b), s) in du.iter_mut().zip(dv.iter_mut()).zip(src.chunks_exact(2)) {
                        *a = s[0] as u16;
                        *b = s[1] as u16;
                    }
                }
            }
            let frame = DecodedFrame {
                width: fw as u32,
                height: fh as u32,
                bit_depth: if deep { 10 } else { 8 },
                chroma: ChromaFormat::Yuv420,
                y,
                u,
                v,
                color,
            };
            (va.unmap_buffer)(self.dpy, img.buf);
            if copied {
                self.readback.set(Some((img, w, h, fourcc)));
            } else {
                (va.destroy_image)(self.dpy, img.image_id);
            }
            Ok(frame)
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if let Some((img, ..)) = self.readback.take() {
            unsafe { (self.va.destroy_image)(self.dpy, img.image_id) };
        }
        unsafe { (self.va.terminate)(self.dpy) };
    }
}

/// Whether this machine has a VA driver that encodes HEVC.
pub fn encode_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| encode::open_device().is_ok())
}

/// Whether this machine has a VA driver that decodes H.264.
pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| Device::open(PROFILE_H264_HIGH).is_ok())
}

/// A VA decoding context: config, surfaces and which picture each surface
/// holds (`owner`, a key the codec chooses: a frame store id, a POC).
pub(crate) struct Session {
    pub config: Id,
    pub context: Id,
    surfaces: Vec<Id>,
    owner: Vec<Option<i64>>,
    /// Coded size in samples, profile and render-target format.
    pub key: (u32, u32, c_int, c_uint),
    /// Buffers of the picture being decoded, destroyed after `vaEndPicture`.
    buffers: Vec<Id>,
}

impl Session {
    pub(crate) fn new(dev: &Device, profile: c_int, rt: c_uint, w: u32, h: u32, n: usize) -> Result<Session, String> {
        let (va, dpy) = (dev.va, dev.dpy);
        unsafe {
            let mut attrib = ConfigAttrib { kind: ATTRIB_RT_FORMAT, value: rt };
            let mut config = 0;
            va.check("vaCreateConfig", (va.create_config)(dpy, profile, ENTRYPOINT_VLD, &mut attrib, 1, &mut config))?;
            let mut surfaces = vec![INVALID_ID; n];
            let r = (va.create_surfaces)(dpy, rt, w, h, surfaces.as_mut_ptr(), n as c_uint, std::ptr::null_mut(), 0);
            if let Err(e) = va.check("vaCreateSurfaces", r) {
                (va.destroy_config)(dpy, config);
                return Err(e);
            }
            let mut context = 0;
            let r = (va.create_context)(dpy, config, w as c_int, h as c_int, PROGRESSIVE, surfaces.as_mut_ptr(), n as c_int, &mut context);
            if let Err(e) = va.check("vaCreateContext", r) {
                (va.destroy_surfaces)(dpy, surfaces.as_mut_ptr(), n as c_int);
                (va.destroy_config)(dpy, config);
                return Err(e);
            }
            Ok(Session { config, context, surfaces, owner: vec![None; n], key: (w, h, profile, rt), buffers: Vec::new() })
        }
    }

    pub(crate) fn destroy(&mut self, dev: &Device) {
        let (va, dpy) = (dev.va, dev.dpy);
        unsafe {
            for id in self.buffers.drain(..) {
                (va.destroy_buffer)(dpy, id);
            }
            (va.destroy_context)(dpy, self.context);
            (va.destroy_surfaces)(dpy, self.surfaces.as_mut_ptr(), self.surfaces.len() as c_int);
            (va.destroy_config)(dpy, self.config);
        }
    }

    /// The surface holding picture `key`.
    pub(crate) fn surface_of(&self, key: i64) -> Option<Id> {
        self.owner.iter().position(|o| *o == Some(key)).map(|i| self.surfaces[i])
    }

    /// The surface for picture `key`: its own if it has one, else one whose
    /// picture is no longer `live`.
    pub(crate) fn claim(&mut self, key: i64, live: impl Fn(i64) -> bool) -> Result<Id, String> {
        if let Some(s) = self.surface_of(key) {
            return Ok(s);
        }
        let i = self.owner.iter().position(|o| o.is_none_or(|k| !live(k))).ok_or("every VA surface holds a reference")?;
        self.owner[i] = Some(key);
        Ok(self.surfaces[i])
    }

    pub(crate) fn buffer<T>(&mut self, dev: &Device, kind: c_int, data: &T) -> Result<Id, String> {
        self.raw_buffer(dev, kind, std::mem::size_of::<T>(), (data as *const T).cast())
    }

    pub(crate) fn raw_buffer(&mut self, dev: &Device, kind: c_int, size: usize, data: *const c_void) -> Result<Id, String> {
        self.raw_buffer_n(dev, kind, size, 1, data)
    }

    /// A buffer of `n` elements of `size` bytes each.
    pub(crate) fn raw_buffer_n(&mut self, dev: &Device, kind: c_int, size: usize, n: usize, data: *const c_void) -> Result<Id, String> {
        let mut id = INVALID_ID;
        let r = unsafe { (dev.va.create_buffer)(dev.dpy, self.context, kind, size as c_uint, n as c_uint, data as *mut c_void, &mut id) };
        dev.va.check("vaCreateBuffer", r)?;
        self.buffers.push(id);
        Ok(id)
    }

    pub(crate) fn begin(&mut self, dev: &Device, target: Id) -> Result<(), String> {
        dev.va.check("vaBeginPicture", unsafe { (dev.va.begin_picture)(dev.dpy, self.context, target) })
    }

    pub(crate) fn render(&mut self, dev: &Device, ids: &mut [Id]) -> Result<(), String> {
        dev.va.check("vaRenderPicture", unsafe { (dev.va.render_picture)(dev.dpy, self.context, ids.as_mut_ptr(), ids.len() as c_int) })
    }

    /// `vaEndPicture`, then free the picture's buffers.
    pub(crate) fn end(&mut self, dev: &Device) -> Result<(), String> {
        let r = dev.va.check("vaEndPicture", unsafe { (dev.va.end_picture)(dev.dpy, self.context) });
        for id in self.buffers.drain(..) {
            unsafe { (dev.va.destroy_buffer)(dev.dpy, id) };
        }
        r
    }
}

/// Bytes of the coded NAL unit (header included) up to RBSP byte `rbsp`
/// counted from `header` bytes in, and the emulation prevention bytes among
/// them.
pub(crate) fn coded_offset(nal: &[u8], header: usize, rbsp: usize) -> (usize, usize) {
    let (mut seen, mut zeros, mut i, mut epb) = (0usize, 0usize, header, 0usize);
    while i < nal.len() && seen < rbsp {
        let b = nal[i];
        if zeros >= 2 && b == 3 {
            zeros = 0;
            epb += 1;
            i += 1;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        seen += 1;
        i += 1;
    }
    // An emulation prevention byte sitting exactly at the target.
    if zeros >= 2 && i < nal.len() && nal[i] == 3 {
        epb += 1;
        i += 1;
    }
    (i, epb)
}

/// The native decoder doing the parsing for a [`VaapiDecoder`].
enum Inner {
    H264(Box<crate::decode::h264::H264Decoder>),
    Hevc(Box<crate::decode::hevc::HevcDecoder>),
    Vp9(Box<crate::decode::vp9::Vp9Decoder>),
    Av1(Box<crate::decode::av1::Av1Decoder>),
}

/// A hardware decoder on a VA driver, with the interface of
/// [`NativeDecoder`](crate::decode::NativeDecoder): open it from a demuxed
/// track, push samples in decode order, take pictures by presentation time.
///
/// H.264 (progressive, MBAFF and field-coded, 8-bit 4:2:0), HEVC (Main
/// and Main 10), VP9 (profiles 0 and 2) and AV1 (Main, 8- and 10-bit 4:2:0). Other streams are refused at [`new`](Self::new), and
/// [`Decoder`](crate::decode::Decoder) then decodes them natively.
pub struct VaapiDecoder {
    dec: Inner,
    nal_len: usize,
    width: u32,
    height: u32,
    ready: Vec<(i64, DecodedFrame)>,
}

impl VaapiDecoder {
    /// Open a session for `track`.
    pub fn new(track: &VideoTrack) -> Result<Self, String> {
        let (dec, sets, nal_len) = match track.codec {
            TrackCodec::H264 => {
                let device = Device::open(PROFILE_H264_HIGH)?;
                let (sets, nal_len) = crate::decode::split_avcc(&track.config)?;
                let mut d = crate::decode::h264::H264Decoder::new();
                d.set_accel(Box::new(h264::VaH264::new(device)));
                (Inner::H264(Box::new(d)), sets, nal_len)
            }
            TrackCodec::Hevc => {
                let device = Device::open(PROFILE_HEVC_MAIN)?;
                let (sets, nal_len) = crate::decode::split_hvcc(&track.config)?;
                let mut d = crate::decode::hevc::HevcDecoder::new();
                d.set_accel(Box::new(hevc::VaHevc::new(device)));
                (Inner::Hevc(Box::new(d)), sets, nal_len)
            }
            TrackCodec::Vp9 => {
                let device = Device::open(vp9::PROFILE_VP9_0)?;
                let mut d = crate::decode::vp9::Vp9Decoder::new();
                d.set_accel(Box::new(vp9::VaVp9::new(device)));
                (Inner::Vp9(Box::new(d)), Vec::new(), 0)
            }
            TrackCodec::Av1 => {
                let device = Device::open(av1_layout::PROFILE_AV1_0)?;
                let mut d = crate::decode::av1::Av1Decoder::new();
                d.set_accel(Box::new(av1::VaAv1::new(device)));
                (Inner::Av1(Box::new(d)), Vec::new(), 0)
            }
        };
        // Drivers have size limits (AMD's VCN will not go below 64 lines,
        // for one) that only show when a context is created, which would
        // otherwise be at the first frame. Try one now, so a stream the GPU
        // will not take is refused here and `Decoder` decodes it natively.
        {
            let (profile, align) = match track.codec {
                TrackCodec::H264 => (PROFILE_H264_HIGH, 16),
                TrackCodec::Vp9 => (vp9::PROFILE_VP9_0, 8),
                TrackCodec::Av1 => (av1_layout::PROFILE_AV1_0, 8),
                TrackCodec::Hevc => (PROFILE_HEVC_MAIN, 8),
            };
            let probe = Device::open(profile)?;
            let (w, h) = (track.width.next_multiple_of(align), track.height.next_multiple_of(align));
            Session::new(&probe, profile, RT_FORMAT_YUV420, w.max(1), h.max(1), 1)
                .map_err(|e| format!("{}x{}: {e}", track.width, track.height))?
                .destroy(&probe);
        }
        let mut me = VaapiDecoder { dec, nal_len, width: track.width, height: track.height, ready: Vec::new() };
        for s in sets {
            me.nal(s)?;
        }
        Ok(me)
    }

    fn nal(&mut self, nal: &[u8]) -> Result<(), String> {
        match &mut self.dec {
            Inner::H264(d) => d.decode_nal(nal),
            Inner::Hevc(d) => d.decode_nal(nal),
            Inner::Vp9(_) | Inner::Av1(_) => Err("VP9 and AV1 have no NAL units".into()),
        }
    }

    /// Size of the track, as the container states it.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Hand one coded frame (length-prefixed NAL units) to the decoder, in
    /// decode order. Its picture is read back before this returns.
    pub fn push(&mut self, sample: &[u8], pts: i64, _dts: i64, _duration: i64) -> Result<(), String> {
        if let Inner::Av1(d) = &mut self.dec {
            // A temporal unit shows at most one picture; the decoder has
            // already read it back from the GPU.
            for (i, f) in d.decode(sample)?.into_iter().enumerate() {
                self.ready.push((pts + i as i64, f.to_decoded()));
            }
            return Ok(());
        }
        if let Inner::Vp9(d) = &mut self.dec {
            // A packet (superframe) shows at most one picture.
            for (i, f) in d.decode(sample)?.into_iter().enumerate() {
                let frame = d.accel_read(&f)?;
                self.ready.push((pts + i as i64, frame));
            }
            return Ok(());
        }
        for nal in crate::decode::bits::split_length_prefixed(sample, self.nal_len)? {
            self.nal(nal)?;
        }
        let (frames, error): (Vec<DecodedFrame>, _) = match &mut self.dec {
            Inner::H264(d) => {
                d.finish_picture();
                (d.take_pictures().into_iter().map(|p| p.frame).collect(), d.take_accel_error())
            }
            Inner::Hevc(d) => {
                d.finish_picture();
                (d.take_pictures().into_iter().map(|p| p.frame).collect(), d.take_accel_error())
            }
            Inner::Vp9(_) | Inner::Av1(_) => unreachable!("VP9 and AV1 returned above"),
        };
        if let Some(e) = error {
            return Err(e);
        }
        for (i, f) in frames.into_iter().enumerate() {
            self.ready.push((pts + i as i64, f));
        }
        Ok(())
    }

    /// The picture for presentation time `pts` as tightly packed BGRA.
    pub fn take(&mut self, pts: i64) -> Option<Vec<u8>> {
        self.take_frame(pts).map(|f| f.to_bgra())
    }

    /// [`take`](Self::take), but the planes.
    pub fn take_frame(&mut self, pts: i64) -> Option<DecodedFrame> {
        let hit = self.ready.iter().position(|(p, _)| *p == pts)?;
        let (_, frame) = self.ready.remove(hit);
        self.ready.retain(|(p, _)| *p > pts);
        Some(frame)
    }
}
