//! NVIDIA's hardware video engines — NVDEC for decoding, NVENC for encoding —
//! reached through the driver's own libraries, loaded when first asked for.
//!
//! Nothing links against NVIDIA at build time: `libcuda`, `libnvcuvid` and
//! `libnvidia-encode` (`nvcuda.dll`, `nvcuvid.dll`, `nvEncodeAPI64.dll` on
//! Windows) are opened at run time, so a binary built with the `nvidia`
//! feature runs unchanged on a machine without the driver, and
//! [`available`] says so instead of failing to load. The struct layouts are
//! those of the Video Codec SDK 12.0 headers (driver 520 or newer on Linux,
//! 522 on Windows), checked field by field against the C compiler's.
//!
//! [`decode::Decoder`](crate::decode::Decoder) and
//! [`hw::Mp4Encoder`](crate::hw::Mp4Encoder) use these when they are
//! available; [`NvdecDecoder`] and [`NvencEncoder`] are the backends on
//! their own.

use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::sync::OnceLock;

mod nvdec;
mod nvenc;
mod nvenc_layout;

pub use nvdec::NvdecDecoder;
pub use nvenc::{NvencEncoder, NvencRate};

/// A shared library opened at run time.
pub(crate) struct Lib(*mut c_void);

// The handle is only ever read after opening, and the loader's own state is
// thread-safe on every platform this builds for.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

#[cfg(unix)]
mod sys {
    use std::ffi::{c_char, c_int, c_void};
    pub const RTLD_NOW: c_int = 2;
    #[link(name = "dl")]
    extern "C" {
        pub fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::{c_char, c_void};
    #[link(name = "kernel32")]
    extern "system" {
        pub fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        pub fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }
}

impl Lib {
    /// The first of `names` that opens.
    pub(crate) fn open(names: &[&str]) -> Option<Lib> {
        names.iter().find_map(|n| {
            let c = CString::new(*n).ok()?;
            #[cfg(unix)]
            let h = unsafe { sys::dlopen(c.as_ptr(), sys::RTLD_NOW) };
            #[cfg(windows)]
            let h = unsafe { sys::LoadLibraryA(c.as_ptr()) };
            (!h.is_null()).then_some(Lib(h))
        })
    }

    /// The address of `name`, as a function pointer of type `T`.
    ///
    /// # Safety
    /// `T` must be the symbol's true signature.
    pub(crate) unsafe fn sym<T: Copy>(&self, name: &str) -> Option<T> {
        assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
        let c = CString::new(name).ok()?;
        #[cfg(unix)]
        let p = sys::dlsym(self.0, c.as_ptr());
        #[cfg(windows)]
        let p = sys::GetProcAddress(self.0, c.as_ptr());
        (!p.is_null()).then(|| std::mem::transmute_copy(&p))
    }
}

pub(crate) type CuResult = c_int;
pub(crate) type CuContext = *mut c_void;
pub(crate) type CuDevicePtr = u64;

/// `CUDA_MEMCPY2D`.
#[repr(C)]
pub(crate) struct Memcpy2d {
    pub src_x_in_bytes: usize,
    pub src_y: usize,
    pub src_memory_type: c_uint,
    pub src_host: *const c_void,
    pub src_device: CuDevicePtr,
    pub src_array: *mut c_void,
    pub src_pitch: usize,
    pub dst_x_in_bytes: usize,
    pub dst_y: usize,
    pub dst_memory_type: c_uint,
    pub dst_host: *mut c_void,
    pub dst_device: CuDevicePtr,
    pub dst_array: *mut c_void,
    pub dst_pitch: usize,
    pub width_in_bytes: usize,
    pub height: usize,
}
const _: () = assert!(std::mem::size_of::<Memcpy2d>() == 128);

impl Memcpy2d {
    /// All fields zero (null pointers), as the C API expects unused ones.
    pub(crate) fn zeroed() -> Self {
        // SAFETY: integers and raw pointers, for which all-zero is valid.
        unsafe { std::mem::zeroed() }
    }
}

pub(crate) const MEMORYTYPE_HOST: c_uint = 1;
pub(crate) const MEMORYTYPE_DEVICE: c_uint = 2;

/// The CUDA driver API calls the codecs need.
pub(crate) struct Cuda {
    _lib: Lib,
    pub init: unsafe extern "C" fn(c_uint) -> CuResult,
    pub device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    pub device_get: unsafe extern "C" fn(*mut c_int, c_int) -> CuResult,
    pub device_get_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> CuResult,
    pub ctx_create: unsafe extern "C" fn(*mut CuContext, c_uint, c_int) -> CuResult,
    pub ctx_destroy: unsafe extern "C" fn(CuContext) -> CuResult,
    pub ctx_push: unsafe extern "C" fn(CuContext) -> CuResult,
    pub ctx_pop: unsafe extern "C" fn(*mut CuContext) -> CuResult,
    pub memcpy_2d: unsafe extern "C" fn(*const Memcpy2d) -> CuResult,
}

impl Cuda {
    fn load() -> Option<Cuda> {
        let lib = Lib::open(&["libcuda.so.1", "libcuda.so", "nvcuda.dll"])?;
        unsafe {
            let c = Cuda {
                init: lib.sym("cuInit")?,
                device_get_count: lib.sym("cuDeviceGetCount")?,
                device_get: lib.sym("cuDeviceGet")?,
                device_get_name: lib.sym("cuDeviceGetName")?,
                ctx_create: lib.sym("cuCtxCreate_v2")?,
                ctx_destroy: lib.sym("cuCtxDestroy_v2")?,
                ctx_push: lib.sym("cuCtxPushCurrent_v2")?,
                ctx_pop: lib.sym("cuCtxPopCurrent_v2")?,
                memcpy_2d: lib.sym("cuMemcpy2D_v2")?,
                _lib: lib,
            };
            let mut n = 0;
            if (c.init)(0) != 0 || (c.device_get_count)(&mut n) != 0 || n < 1 {
                return None;
            }
            Some(c)
        }
    }
}

/// The CUDA driver, loaded and initialised once, or `None` when this machine
/// has no NVIDIA GPU or driver.
pub(crate) fn cuda() -> Option<&'static Cuda> {
    static CUDA: OnceLock<Option<Cuda>> = OnceLock::new();
    CUDA.get_or_init(Cuda::load).as_ref()
}

pub(crate) fn check(what: &str, r: CuResult) -> Result<(), String> {
    if r == 0 {
        Ok(())
    } else {
        Err(format!("{what} failed (CUDA error {r})"))
    }
}

/// A CUDA context on the first GPU, destroyed on drop.
pub(crate) struct Context {
    cuda: &'static Cuda,
    ctx: CuContext,
}

impl Context {
    pub(crate) fn new() -> Result<Context, String> {
        let cuda = cuda().ok_or("no NVIDIA GPU or driver")?;
        let mut dev = 0;
        let mut ctx = std::ptr::null_mut();
        unsafe {
            check("cuDeviceGet", (cuda.device_get)(&mut dev, 0))?;
            check("cuCtxCreate", (cuda.ctx_create)(&mut ctx, 0, dev))?;
            // Created current; the codecs push it around each call instead.
            let mut popped = std::ptr::null_mut();
            (cuda.ctx_pop)(&mut popped);
        }
        Ok(Context { cuda, ctx })
    }

    pub(crate) fn raw(&self) -> CuContext {
        self.ctx
    }

    /// Run `f` with this context current on the calling thread.
    pub(crate) fn with<R>(&self, f: impl FnOnce() -> R) -> R {
        unsafe { (self.cuda.ctx_push)(self.ctx) };
        let r = f();
        let mut popped = std::ptr::null_mut();
        unsafe { (self.cuda.ctx_pop)(&mut popped) };
        r
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { (self.cuda.ctx_destroy)(self.ctx) };
    }
}

/// Whether this machine has an NVIDIA GPU and a driver new enough for the
/// codecs (NVDEC for decoding; NVENC is checked when an encoder opens).
pub fn available() -> bool {
    nvdec::cuvid().is_some()
}

/// The first GPU's name, as the driver reports it.
pub fn device_name() -> Option<String> {
    let cuda = cuda()?;
    let mut buf = [0 as c_char; 256];
    unsafe {
        let mut dev = 0;
        if (cuda.device_get)(&mut dev, 0) != 0 || (cuda.device_get_name)(buf.as_mut_ptr(), 256, dev) != 0 {
            return None;
        }
        Some(std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned())
    }
}
