//! Device, command queue and shader library.

use std::ffi::c_void;

use super::enums::*;
use super::objc::{
    alloc_init, class, error_message, msg0, msg1, msg2, msg3, nsstring, nsstring_to_string, sel,
    AutoreleasePool, Bool, Id, Owned, NIL, YES,
};

#[link(name = "Metal", kind = "framework")]
extern "C" {
    /// Returns the system's preferred device, +1 retained (a `Create` function).
    fn MTLCreateSystemDefaultDevice() -> Id;
}

/// Anything that can go wrong on the way to a drawn frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalError {
    /// No Metal device — a Mac too old for Metal, or a process with no GPU
    /// access at all (some sandboxes, some SSH sessions).
    NoDevice,
    /// `newCommandQueue` returned nil.
    NoCommandQueue,
    /// The MSL failed to compile. Carries the compiler's diagnostics verbatim.
    ShaderCompilation(String),
    /// A shader function named in the library was not found.
    MissingFunction(String),
    /// Pipeline state creation failed (usually an attachment format mismatch).
    PipelineCreation(String),
    /// A texture, buffer or sampler could not be allocated.
    Allocation(String),
    /// An unsupported input reached the backend — see the message.
    Unsupported(String),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => write!(f, "no Metal device available"),
            Self::NoCommandQueue => write!(f, "could not create a Metal command queue"),
            Self::ShaderCompilation(m) => write!(f, "Metal shader compilation failed: {m}"),
            Self::MissingFunction(n) => write!(f, "shader function `{n}` not found in library"),
            Self::PipelineCreation(m) => write!(f, "Metal pipeline creation failed: {m}"),
            Self::Allocation(m) => write!(f, "Metal allocation failed: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported by the Metal backend: {m}"),
        }
    }
}

impl std::error::Error for MetalError {}

/// An `MTLDevice` plus the one command queue and one shader library the
/// backend needs.
///
/// Cloning is cheap and shares the same GPU objects, so a surface renderer and
/// an offscreen renderer can be driven from one device.
#[derive(Clone)]
pub struct MetalDevice {
    device: Owned,
    queue: Owned,
    library: Owned,
}

impl MetalDevice {
    /// Acquire the system default device and compile the built-in shader
    /// library. Compilation happens once, at construction — roughly 100 ms.
    pub fn new() -> Result<Self, MetalError> {
        Self::with_shader_source(super::SHADER_SOURCE)
    }

    /// As [`new`](Self::new), but compiling `source` instead of the built-in
    /// library. The source must define the same entry points; see
    /// [`crate::metal::SHADER_SOURCE`].
    pub fn with_shader_source(source: &str) -> Result<Self, MetalError> {
        let _pool = AutoreleasePool::new();
        // MTLCreateSystemDefaultDevice returns +1 — take it, do not retain again.
        let device = unsafe { Owned::from_retained(MTLCreateSystemDefaultDevice()) }
            .ok_or(MetalError::NoDevice)?;
        unsafe { Self::build(device, source) }
    }

    /// Build on an `MTLDevice` someone else owns.
    ///
    /// For the case where the device is not ours to choose: the visionOS
    /// compositor names the device its drawables belong to, and textures cannot
    /// cross from one device to another.
    ///
    /// # Safety
    /// `device` must be a live `MTLDevice`. It is retained for as long as this
    /// value lives.
    pub unsafe fn adopt(device: Id) -> Result<Self, MetalError> {
        let _pool = AutoreleasePool::new();
        let device = Owned::retain(device).ok_or(MetalError::NoDevice)?;
        Self::build(device, super::SHADER_SOURCE)
    }

    /// Command queue + shader library on an owned device.
    unsafe fn build(device: Owned, source: &str) -> Result<Self, MetalError> {
        let queue = Owned::from_retained(msg0(device.id(), sel!("newCommandQueue")))
            .ok_or(MetalError::NoCommandQueue)?;
        let library = compile_library(device.id(), source)?;
        Ok(Self {
            device,
            queue,
            library,
        })
    }

    /// The `MTLDevice`. Borrowed — do not release.
    #[inline]
    pub fn device_id(&self) -> Id {
        self.device.id()
    }

    /// The `MTLCommandQueue`. Borrowed — do not release.
    #[inline]
    pub fn queue_id(&self) -> Id {
        self.queue.id()
    }

    /// The compiled `MTLLibrary`. Borrowed — do not release.
    #[inline]
    pub fn library_id(&self) -> Id {
        self.library.id()
    }

    /// The GPU's name, e.g. `"Apple M2"`.
    pub fn name(&self) -> String {
        let _pool = AutoreleasePool::new();
        unsafe { nsstring_to_string(msg0(self.device.id(), sel!("name"))) }
            .unwrap_or_else(|| "unknown".into())
    }

    /// Whether the device shares memory with the CPU — true on Apple silicon
    /// and on Intel integrated GPUs, false on discrete AMD.
    ///
    /// Only an optimisation hint here: readback always goes through a shared
    /// buffer, which is correct on both.
    pub fn has_unified_memory(&self) -> bool {
        let has: Bool = unsafe { msg0(self.device.id(), sel!("hasUnifiedMemory")) };
        has != 0
    }

    /// Whether `sample_count` is a legal MSAA count for this device.
    pub fn supports_sample_count(&self, sample_count: u32) -> bool {
        let ok: Bool = unsafe {
            msg1(
                self.device.id(),
                sel!("supportsTextureSampleCount:"),
                sample_count as NSUInteger,
            )
        };
        ok != 0
    }

    /// Look up a function in the library. `+1`, so the caller owns it.
    pub fn function(&self, name: &str) -> Result<Owned, MetalError> {
        let _pool = AutoreleasePool::new();
        unsafe {
            let ns = nsstring(name);
            let f: Id = msg1(self.library.id(), sel!("newFunctionWithName:"), ns);
            Owned::from_retained(f).ok_or_else(|| MetalError::MissingFunction(name.into()))
        }
    }

    /// A GPU buffer holding a copy of `bytes`, in shared (CPU-visible) storage.
    ///
    /// Shared storage rather than private: it is coherent on discrete GPUs too,
    /// and this backend uploads geometry once and then only reads it.
    pub fn new_buffer(&self, bytes: &[u8]) -> Result<Owned, MetalError> {
        let len = bytes.len().max(1) as NSUInteger;
        unsafe {
            let buf: Id = if bytes.is_empty() {
                msg2(
                    self.device.id(),
                    sel!("newBufferWithLength:options:"),
                    len,
                    resource_options::STORAGE_MODE_SHARED,
                )
            } else {
                msg3(
                    self.device.id(),
                    sel!("newBufferWithBytes:length:options:"),
                    bytes.as_ptr() as *const c_void,
                    len,
                    resource_options::STORAGE_MODE_SHARED,
                )
            };
            Owned::from_retained(buf)
                .ok_or_else(|| MetalError::Allocation(format!("buffer of {len} bytes")))
        }
    }

    /// A command buffer from the queue. Autoreleased — a pool must be live.
    ///
    /// # Safety
    /// The returned pointer is valid until the enclosing [`AutoreleasePool`]
    /// drains.
    pub unsafe fn command_buffer(&self) -> Id {
        msg0(self.queue.id(), sel!("commandBuffer"))
    }

    /// Set an object's `label`, which is what Instruments and the GPU frame
    /// debugger show. Cheap, and the difference between a readable capture and
    /// a wall of `<MTLTexture: 0x…>`.
    ///
    /// # Safety
    /// `obj` must respond to `setLabel:` (every Metal object does).
    pub unsafe fn set_label(obj: Id, label: &str) {
        let _pool = AutoreleasePool::new();
        let _: () = msg1(obj, sel!("setLabel:"), nsstring(label));
    }
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice")
            .field("name", &self.name())
            .field("unified_memory", &self.has_unified_memory())
            .finish()
    }
}

/// `[device newLibraryWithSource:options:error:]`, with the compiler's
/// diagnostics preserved on failure — an MSL error message names the line, and
/// throwing it away in favour of "compilation failed" would make every shader
/// edit a guessing game.
unsafe fn compile_library(device: Id, source: &str) -> Result<Owned, MetalError> {
    let src = nsstring(source);
    if src.is_null() {
        return Err(MetalError::ShaderCompilation(
            "shader source contains an interior NUL byte".into(),
        ));
    }
    let mut err: Id = NIL;
    let lib: Id = msg3(
        device,
        sel!("newLibraryWithSource:options:error:"),
        src,
        NIL, // default MTLCompileOptions
        &mut err as *mut Id,
    );
    match Owned::from_retained(lib) {
        Some(lib) => Ok(lib),
        None => Err(MetalError::ShaderCompilation(error_message(err))),
    }
}

/// A `MTLDepthStencilState`: `depthCompareFunction` plus write enable.
pub fn depth_stencil_state(
    device: &MetalDevice,
    compare: NSUInteger,
    write: bool,
) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = alloc_init(class!("MTLDepthStencilDescriptor"));
        let desc = Owned::from_retained(desc)
            .ok_or_else(|| MetalError::Allocation("MTLDepthStencilDescriptor".into()))?;
        let _: () = msg1(desc.id(), sel!("setDepthCompareFunction:"), compare);
        let _: () = msg1(
            desc.id(),
            sel!("setDepthWriteEnabled:"),
            if write { YES } else { 0 as Bool },
        );
        let state: Id = msg1(
            device.device_id(),
            sel!("newDepthStencilStateWithDescriptor:"),
            desc.id(),
        );
        Owned::from_retained(state)
            .ok_or_else(|| MetalError::Allocation("MTLDepthStencilState".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_name_the_stage() {
        assert!(MetalError::NoDevice.to_string().contains("no Metal device"));
        assert!(MetalError::ShaderCompilation("line 4: oops".into())
            .to_string()
            .contains("line 4: oops"));
        assert!(MetalError::MissingFunction("vs_mesh".into())
            .to_string()
            .contains("vs_mesh"));
    }
}
