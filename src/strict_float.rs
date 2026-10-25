//! IEEE float arithmetic in GPU shaders, for bit-level comparisons.
//!
//! wgpu hands Metal its shaders as source, compiled with Metal's defaults:
//! fast math. Every multiply, add and divide may then be reassociated,
//! `a * b + c` is fused into one rounding, division and the transcendental
//! functions are approximated — so a shader written to reproduce another
//! GPU's float32 arithmetic step by step (as the marks emulating deck.gl on
//! SwiftShader are) does not compute what it says. Even Metal's `safe` math
//! mode still fuses multiply-adds within an expression.
//!
//! [`enable`] makes every Metal library compiled from source in this process
//! from then on strict: it puts `#pragma METAL fp math_mode(safe)` and
//! `#pragma METAL fp contract(off)` in front of the source and asks for the
//! precise float functions. An explicit `fma` (WGSL's `fma`) stays fused.
//! wgpu offers no option for this, so the method that compiles source,
//! `-[MTLDevice newLibraryWithSource:options:error:]`, is replaced on the
//! system device's class through the Objective-C runtime. Process-wide and
//! permanent; meant for tests that compare images with another renderer's.
//! A no-op elsewhere than macOS.

/// Compile Metal shaders with strict IEEE float arithmetic from now on (see
/// the [module docs](self)). Returns whether it is in effect.
pub fn enable() -> bool {
    imp::enable()
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub fn enable() -> bool {
        false
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::{c_char, c_void, CStr};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;

    type Id = *mut c_void;
    type Sel = *const c_void;
    type Imp = unsafe extern "C" fn();
    type NewLibrary = unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id;

    #[link(name = "objc", kind = "dylib")]
    extern "C" {
        fn sel_registerName(name: *const c_char) -> Sel;
        fn object_getClass(obj: Id) -> Id;
        fn class_getInstanceMethod(cls: Id, sel: Sel) -> *mut c_void;
        fn method_getImplementation(m: *mut c_void) -> Option<Imp>;
        fn method_setImplementation(m: *mut c_void, imp: Imp) -> Option<Imp>;
        fn objc_getClass(name: *const c_char) -> Id;
        fn objc_msgSend();
    }

    #[link(name = "Metal", kind = "framework")]
    extern "C" {
        fn MTLCreateSystemDefaultDevice() -> Id;
    }

    /// The pragmas, before anything else in the source.
    const PRAGMAS: &str = "#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n";

    /// The replaced implementation, stored before ours is installed.
    static ORIGINAL: AtomicUsize = AtomicUsize::new(0);
    static INSTALLED: OnceLock<bool> = OnceLock::new();

    fn sel(name: &CStr) -> Sel {
        unsafe { sel_registerName(name.as_ptr()) }
    }

    unsafe extern "C" fn new_library(this: Id, cmd: Sel, source: Id, options: Id, error: *mut Id) -> Id {
        let original: NewLibrary = std::mem::transmute(ORIGINAL.load(Ordering::Acquire));
        let send_id: unsafe extern "C" fn(Id, Sel) -> Id = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let utf8: unsafe extern "C" fn(Id, Sel) -> *const c_char = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let from_utf8: unsafe extern "C" fn(Id, Sel, *const c_char) -> Id = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let set_u64: unsafe extern "C" fn(Id, Sel, u64) = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let responds: unsafe extern "C" fn(Id, Sel, Sel) -> i8 = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

        let text = utf8(source, sel(c"UTF8String"));
        let strict = if text.is_null() {
            source
        } else {
            let mut s = String::from(PRAGMAS);
            s.push_str(&CStr::from_ptr(text).to_string_lossy());
            s.push('\0');
            // Autoreleased; the caller's pool holds it for the call.
            from_utf8(objc_getClass(c"NSString".as_ptr()), sel(c"stringWithUTF8String:"), s.as_ptr().cast())
        };
        let options = if options.is_null() {
            send_id(send_id(objc_getClass(c"MTLCompileOptions".as_ptr()), sel(c"new")), sel(c"autorelease"))
        } else {
            options
        };
        // macOS 15: MTLMathModeSafe (0) and MTLMathFloatingPointFunctionsPrecise (1).
        for (setter, value) in [(c"setMathMode:", 0u64), (c"setMathFloatingPointFunctions:", 1)] {
            if responds(options, sel(c"respondsToSelector:"), sel(setter)) != 0 {
                set_u64(options, sel(setter), value);
            }
        }
        if responds(options, sel(c"respondsToSelector:"), sel(c"setFastMathEnabled:")) != 0 {
            let set_bool: unsafe extern "C" fn(Id, Sel, i8) = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            // Deprecated in favour of mathMode, which it would override when
            // set after it: only where mathMode is missing.
            if responds(options, sel(c"respondsToSelector:"), sel(c"setMathMode:")) == 0 {
                set_bool(options, sel(c"setFastMathEnabled:"), 0);
            }
        }
        original(this, cmd, strict, options, error)
    }

    pub fn enable() -> bool {
        *INSTALLED.get_or_init(|| unsafe {
            let device = MTLCreateSystemDefaultDevice();
            if device.is_null() {
                return false;
            }
            let class = object_getClass(device);
            let method = class_getInstanceMethod(class, sel(c"newLibraryWithSource:options:error:"));
            let release: unsafe extern "C" fn(Id, Sel) = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            // wgpu opens its own device, of the same class.
            release(device, sel(c"release"));
            let Some(previous) = (!method.is_null()).then(|| method_getImplementation(method)).flatten() else {
                return false;
            };
            ORIGINAL.store(previous as usize, Ordering::Release);
            let ours: NewLibrary = new_library;
            method_setImplementation(method, std::mem::transmute::<NewLibrary, Imp>(ours));
            true
        })
    }
}
