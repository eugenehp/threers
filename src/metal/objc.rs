//! The Objective-C runtime, by hand.
//!
//! Metal is an Objective-C API, so a Rust backend for it needs a way to send
//! messages. The usual answer is the `objc` crate (and `metal-rs` on top of
//! it); this module is the alternative — `objc_msgSend` declared directly,
//! transmuted per call site to the exact signature the method expects, plus
//! the three things needed to keep the memory honest: class lookup, selector
//! interning, and retain/release.
//!
//! ## Why transmute
//!
//! `objc_msgSend` has no single signature. It is dispatched to a method whose
//! arguments and return value are whatever that method declares, and on arm64
//! there is no variadic fallback — the caller *must* set up registers as if it
//! were calling the implementation directly. So the only correct way to call it
//! is to transmute the symbol to the concrete signature at each call site, which
//! is what [`msg0`]–[`msg9`] do. Every argument type crossing this boundary is
//! `#[repr(C)]`, so rustc lays it out with the platform C ABI and the callee
//! finds what it expects — including by-value structs like `MTLClearColor`.
//!
//! ## Ownership
//!
//! Cocoa's naming rule is the contract: a method whose name begins with `new`,
//! `alloc`, `copy` or `mutableCopy` — plus C functions with `Create` in the name
//! — returns an object the caller owns (+1) and must release. Everything else
//! returns an autoreleased object, valid until the enclosing pool drains.
//!
//! [`Owned`] models the first case and [`AutoreleasePool`] the second. Any frame
//! that touches autoreleased objects (command buffers, encoders, render pass
//! descriptors, drawables — all of them) must hold a pool, or the objects
//! accumulate until the process exits.

use std::ffi::{c_char, c_void, CStr, CString};

/// An opaque Objective-C object. Only ever handled behind a pointer.
#[repr(C)]
pub struct Object {
    _private: [u8; 0],
}

/// `id` — a pointer to an Objective-C object.
pub type Id = *mut Object;

/// `nil`.
pub const NIL: Id = std::ptr::null_mut();

/// `SEL` — an interned selector (method name).
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Sel(pub *const c_void);

/// Objective-C `BOOL`. `signed char` on every platform Metal runs on.
pub type Bool = i8;
pub const YES: Bool = 1;
pub const NO: Bool = 0;

#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> Id;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
    fn objc_retain(obj: Id) -> Id;
    fn objc_release(obj: Id);
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

/// Look up a class by its NUL-terminated name. Prefer the `class!` macro,
/// which caches the lookup.
///
/// Returns `nil` if the class is not registered — which for a framework class
/// means the framework was not linked into the binary.
pub fn get_class(name: &str) -> Id {
    debug_assert!(name.ends_with('\0'), "class name must be NUL-terminated");
    unsafe { objc_getClass(name.as_ptr().cast()) }
}

/// Intern a NUL-terminated selector name. Prefer the `sel!` macro, which
/// caches the result — interning is a lock + hash lookup, and a frame sends
/// thousands of messages.
pub fn get_sel(name: &str) -> Sel {
    debug_assert!(name.ends_with('\0'), "selector must be NUL-terminated");
    unsafe { sel_registerName(name.as_ptr().cast()) }
}

macro_rules! msg_fn {
    ($name:ident; $($p:ident : $t:ident),*) => {
        /// Send a message, transmuting `objc_msgSend` to this arity.
        ///
        /// # Safety
        /// The caller guarantees `recv` responds to `sel`, and that the type
        /// parameters match the method's actual signature exactly.
        #[inline]
        #[allow(clippy::too_many_arguments)] // a method's arity is the method's
        pub unsafe fn $name<R, $($t),*>(recv: Id, sel: Sel, $($p: $t),*) -> R {
            let send: unsafe extern "C" fn(Id, Sel, $($t),*) -> R =
                std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            send(recv, sel, $($p),*)
        }
    };
}

msg_fn!(msg0;);
msg_fn!(msg1; a: A);
msg_fn!(msg2; a: A, b: B);
msg_fn!(msg3; a: A, b: B, c: C);
msg_fn!(msg4; a: A, b: B, c: C, d: D);
msg_fn!(msg5; a: A, b: B, c: C, d: D, e: E);
msg_fn!(msg6; a: A, b: B, c: C, d: D, e: E, f: F);
msg_fn!(msg7; a: A, b: B, c: C, d: D, e: E, f: F, g: G);
msg_fn!(msg8; a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H);
msg_fn!(msg9; a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H, i: I);

/// `sel!("setLabel:")` — an interned selector, looked up once per call site.
macro_rules! sel {
    ($name:literal) => {{
        static CACHE: ::std::sync::OnceLock<usize> = ::std::sync::OnceLock::new();
        $crate::metal::objc::Sel(
            *CACHE.get_or_init(|| $crate::metal::objc::get_sel(concat!($name, "\0")).0 as usize)
                as *const ::std::ffi::c_void,
        )
    }};
}

/// `class!("MTLTextureDescriptor")` — a class pointer, looked up once per call site.
macro_rules! class {
    ($name:literal) => {{
        static CACHE: ::std::sync::OnceLock<usize> = ::std::sync::OnceLock::new();
        *CACHE.get_or_init(|| $crate::metal::objc::get_class(concat!($name, "\0")) as usize)
            as $crate::metal::objc::Id
    }};
}

pub(crate) use class;
pub(crate) use sel;

/// `[[Class alloc] init]`.
///
/// # Safety
/// `cls` must be a class whose designated initialiser is `-init`.
pub unsafe fn alloc_init(cls: Id) -> Id {
    let obj: Id = msg0(cls, sel!("alloc"));
    msg0(obj, sel!("init"))
}

/// An owned (+1) reference. Releases on drop, retains on clone.
///
/// Construct with [`Owned::from_retained`] for the result of a `new*`/`alloc`/
/// `copy` method or a `*Create*` function, and [`Owned::retain`] for anything
/// else you want to outlive the current autorelease pool.
pub struct Owned(Id);

impl Owned {
    /// Take ownership of an already-retained (+1) object. `None` for `nil`.
    ///
    /// # Safety
    /// `id` must be `nil` or an object the caller owns a reference to; that
    /// reference is transferred here.
    pub unsafe fn from_retained(id: Id) -> Option<Self> {
        if id.is_null() {
            None
        } else {
            Some(Self(id))
        }
    }

    /// Retain an object owned by someone else (typically autoreleased).
    ///
    /// # Safety
    /// `id` must be `nil` or a live object.
    pub unsafe fn retain(id: Id) -> Option<Self> {
        if id.is_null() {
            None
        } else {
            Some(Self(objc_retain(id)))
        }
    }

    /// The raw pointer. Borrowed — do not release it.
    #[inline]
    pub fn id(&self) -> Id {
        self.0
    }
}

impl Clone for Owned {
    fn clone(&self) -> Self {
        Self(unsafe { objc_retain(self.0) })
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { objc_release(self.0) }
    }
}

impl std::fmt::Debug for Owned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Owned({:p})", self.0)
    }
}

/// An autorelease pool, popped on drop.
///
/// Metal hands back autoreleased objects for everything transient — command
/// buffers, encoders, render pass descriptors, drawables. Without a live pool
/// they are never freed, so a render loop leaks a command buffer per frame.
pub struct AutoreleasePool(*mut c_void);

impl AutoreleasePool {
    pub fn new() -> Self {
        Self(unsafe { objc_autoreleasePoolPush() })
    }
}

impl Default for AutoreleasePool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        unsafe { objc_autoreleasePoolPop(self.0) }
    }
}

/// An autoreleased `NSString` from a Rust string.
///
/// # Safety
/// An [`AutoreleasePool`] must be live; the result is valid until it drains.
/// Returns `nil` if `s` contains an interior NUL.
pub unsafe fn nsstring(s: &str) -> Id {
    let Ok(c) = CString::new(s) else {
        return NIL;
    };
    let cls = class!("NSString");
    msg1(cls, sel!("stringWithUTF8String:"), c.as_ptr())
}

/// `[[nserror localizedDescription] UTF8String]`, or a placeholder.
///
/// # Safety
/// `err` must be `nil` or an `NSError`.
pub unsafe fn error_message(err: Id) -> String {
    if err.is_null() {
        return "unknown error (nil NSError)".into();
    }
    let desc: Id = msg0(err, sel!("localizedDescription"));
    if desc.is_null() {
        return "unknown error (no description)".into();
    }
    let utf8: *const c_char = msg0(desc, sel!("UTF8String"));
    if utf8.is_null() {
        return "unknown error (no UTF-8 description)".into();
    }
    CStr::from_ptr(utf8).to_string_lossy().into_owned()
}

/// The UTF-8 contents of an `NSString`, or `None` for `nil`.
///
/// # Safety
/// `s` must be `nil` or an `NSString`.
pub unsafe fn nsstring_to_string(s: Id) -> Option<String> {
    if s.is_null() {
        return None;
    }
    let utf8: *const c_char = msg0(s, sel!("UTF8String"));
    if utf8.is_null() {
        return None;
    }
    Some(CStr::from_ptr(utf8).to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selectors_and_classes_resolve() {
        assert!(!get_class("NSObject\0").is_null());
        assert!(!get_sel("init\0").0.is_null());
        // Interning is idempotent: the same name is the same pointer.
        assert_eq!(get_sel("description\0").0, get_sel("description\0").0);
    }

    #[test]
    fn unknown_class_is_nil() {
        assert!(get_class("ThreersNoSuchClass\0").is_null());
    }

    #[test]
    fn nsstring_round_trips() {
        let _pool = AutoreleasePool::new();
        unsafe {
            let s = nsstring("metal backend");
            assert_eq!(nsstring_to_string(s).as_deref(), Some("metal backend"));
        }
    }

    #[test]
    fn owned_retains_and_releases() {
        let _pool = AutoreleasePool::new();
        unsafe {
            let obj = alloc_init(class!("NSObject"));
            let owned = Owned::from_retained(obj).expect("NSObject alloc/init");
            let count: usize = msg0(owned.id(), sel!("retainCount"));
            let clone = owned.clone();
            let after: usize = msg0(owned.id(), sel!("retainCount"));
            assert_eq!(after, count + 1);
            drop(clone);
            let back: usize = msg0(owned.id(), sel!("retainCount"));
            assert_eq!(back, count);
        }
    }
}
