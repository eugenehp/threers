//! The Metal side of [`super::gpu`]: the same calls, on Metal objects.
//!
//! What wgpu decides for a shader — where each binding of each group lives —
//! is decided here from the pipeline layout: a group's buffers, textures and
//! samplers take the next free slot of their kind, in group then binding
//! order, in both stages alike, and vertex buffers count down from the top
//! buffer slot. The WGSL goes through naga's MSL writer with that map. Bind
//! groups are applied at draw time, against whatever pipeline is set, so a
//! group may be set before its pipeline as wgpu allows.
//!
//! A submit commits and returns, as wgpu's does; a map or a poll waits for
//! the last commit, and a buffer write the GPU may still be reading goes
//! through the queue (see `IN_FLIGHT`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::metal::enums::*;
use crate::metal::objc::{alloc_init, class, msg0, msg1, msg2, msg3, msg4, msg5, msg8, msg9, sel, AutoreleasePool, Id, Owned, NIL, YES};
use crate::metal::MetalDevice;

/// The highest buffer slot; vertex buffer `i` binds at `TOP - i`.
const TOP_BUFFER: u64 = 30;

/// A Metal device marks draw with.
pub struct MGpu {
    pub(crate) device: MetalDevice,
    /// Compiled libraries by their MSL — the compiler's whole input. Not
    /// by module: a module dropped after its pipeline is built frees an
    /// address the next module can be given, which then found the first
    /// one's library.
    libraries: Mutex<HashMap<String, Owned>>,
}

// SAFETY: Metal devices, queues and the objects made from them may be used
// from any thread; the cache is behind a mutex.
unsafe impl Send for MGpu {}
unsafe impl Sync for MGpu {}

impl MGpu {
    pub fn new(device: MetalDevice) -> Self {
        Self { device, libraries: Mutex::new(HashMap::new()) }
    }
}

// SAFETY: Metal buffers, textures, samplers and pipeline states are
// thread-safe objects (Apple: "Metal Best Practices — Resources are
// thread-safe"); the wrappers only retain and read them.
unsafe impl Send for MBuffer {}
unsafe impl Sync for MBuffer {}
unsafe impl Send for MTexture {}
unsafe impl Sync for MTexture {}
unsafe impl Send for MSampler {}
unsafe impl Sync for MSampler {}
unsafe impl Send for MRes {}
unsafe impl Sync for MRes {}
unsafe impl Send for PipelineInner {}
unsafe impl Sync for PipelineInner {}
unsafe impl Send for ComputeInner {}
unsafe impl Sync for ComputeInner {}

#[derive(Clone, Debug)]
pub struct MBuffer {
    pub(crate) obj: Owned,
    pub(crate) size: u64,
}

#[derive(Clone, Debug)]
pub struct MTexture {
    pub(crate) obj: Owned,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mips: u32,
    /// Above 1, a 3D texture this deep.
    pub(crate) depth: u32,
    /// Array layers: 1 for a plain 2D texture (or a one-layer view of an
    /// array); more for a 2D array.
    pub(crate) layers: u32,
}

#[derive(Clone, Debug)]
pub struct MSampler(pub(crate) Owned);

#[derive(Clone, Debug)]
pub struct MLayout(pub(crate) Arc<Vec<wgpu::BindGroupLayoutEntry>>);

#[derive(Clone, Debug)]
pub enum MRes {
    Buffer(Owned, u64),
    Texture(Owned),
    Sampler(Owned),
}

#[derive(Clone, Debug)]
pub struct MBindGroup {
    pub(crate) layout: MLayout,
    pub(crate) entries: Arc<Vec<(u32, MRes)>>,
}

#[derive(Clone, Debug)]
pub struct MPipelineLayout(pub(crate) Arc<Vec<MLayout>>);

pub struct ModuleSrc {
    source: String,
    parsed: OnceLock<Result<(wgpu::naga::Module, wgpu::naga::valid::ModuleInfo), String>>,
}

#[derive(Clone)]
pub struct MModule(pub(crate) Arc<ModuleSrc>);

impl std::fmt::Debug for MModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MModule")
    }
}

/// naga's sizes buffer, when the module has a storage buffer holding a
/// runtime-sized array: its slot, and the `(group, binding)` of each such
/// buffer, in declaration order — the order naga reads the sizes in.
type Sizes = Option<(u8, Vec<(u32, u32)>)>;

fn runtime_sized(module: &wgpu::naga::Module, slots: &Slots) -> Sizes {
    use wgpu::naga::{ArraySize, TypeInner};
    let dynamic = |ty: wgpu::naga::Handle<wgpu::naga::Type>| match &module.types[ty].inner {
        TypeInner::Array { size: ArraySize::Dynamic, .. } => true,
        TypeInner::Struct { members, .. } => members.last().is_some_and(|m| {
            matches!(module.types[m.ty].inner, TypeInner::Array { size: ArraySize::Dynamic, .. })
        }),
        _ => false,
    };
    let list: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .filter_map(|(_, g)| g.binding.as_ref().filter(|_| dynamic(g.ty)).map(|b| (b.group, b.binding)))
        .collect();
    if list.is_empty() {
        return None;
    }
    let next = slots.values().filter(|(k, _)| *k == 0).map(|(_, s)| s + 1).max().unwrap_or(0);
    Some((next, list))
}

/// The sizes buffer's contents: each listed buffer's bound length in bytes.
fn sizes_bytes(sizes: &[(u32, u32)], groups: &[Option<(MBindGroup, Vec<u32>)>]) -> Vec<u32> {
    sizes
        .iter()
        .map(|&(g, b)| {
            groups
                .get(g as usize)
                .and_then(|x| x.as_ref())
                .and_then(|(group, _)| group.entries.iter().find(|(binding, _)| *binding == b))
                .map_or(0, |(_, res)| match res {
                    MRes::Buffer(buf, off) => unsafe {
                        let len: NSUInteger = msg0(buf.id(), sel!("length"));
                        (len as u64).saturating_sub(*off) as u32
                    },
                    _ => 0,
                })
        })
        .collect()
}

/// Each entry point's name in MSL, or why naga could not write it.
type Names = HashMap<String, Result<String, String>>;

/// `entry`'s MSL name.
fn entry_name(names: &Names, entry: &str) -> Result<String, String> {
    match names.get(entry) {
        Some(Ok(n)) => Ok(n.clone()),
        Some(Err(e)) => Err(format!("`{entry}` did not translate: {e}")),
        None => Err(format!("no entry point `{entry}`")),
    }
}

/// Where a binding lives: `(kind, slot)`, kind 0 buffer, 1 texture, 2 sampler.
type Slots = HashMap<(u32, u32), (u8, u8)>;

pub struct PipelineInner {
    state: Owned,
    depth: Option<Owned>,
    primitive: NSUInteger,
    cull: NSUInteger,
    winding: NSUInteger,
    bias: Option<(f32, f32)>,
    slots: Slots,
    sizes: Sizes,
}

#[derive(Clone)]
pub struct MPipeline(pub(crate) Arc<PipelineInner>);

impl std::fmt::Debug for MPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MPipeline")
    }
}

pub struct ComputeInner {
    state: Owned,
    slots: Slots,
    threads: [u32; 3],
    sizes: Sizes,
}

#[derive(Clone)]
pub struct MComputePipeline(pub(crate) Arc<ComputeInner>);

impl std::fmt::Debug for MComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MComputePipeline")
    }
}

pub struct MComputePass {
    enc: Owned,
    pipeline: Option<MComputePipeline>,
    groups: Vec<Option<(MBindGroup, Vec<u32>)>>,
}

impl Drop for MComputePass {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg0(self.enc.id(), sel!("endEncoding"));
        }
    }
}

pub struct MEncoder {
    pub(crate) cmd: Owned,
}

pub struct MPass {
    enc: Owned,
    pipeline: Option<MPipeline>,
    groups: Vec<Option<(MBindGroup, Vec<u32>)>>,
    index: Option<(Owned, u64, NSUInteger)>,
}

impl Drop for MPass {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg0(self.enc.id(), sel!("endEncoding"));
        }
    }
}

// ------------------------------------------------------------- conversions

/// A private 2D array texture, `depth_or_array_layers` slices.
fn new_texture_2d_array(device: &MetalDevice, desc: &wgpu::TextureDescriptor, format: NSUInteger, usage: NSUInteger) -> Result<Owned, crate::metal::MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let d = Owned::from_retained(alloc_init(class!("MTLTextureDescriptor")))
            .ok_or_else(|| crate::metal::MetalError::Allocation("MTLTextureDescriptor".into()))?;
        let _: () = msg1(d.id(), sel!("setTextureType:"), texture_type::TYPE_2D_ARRAY);
        let _: () = msg1(d.id(), sel!("setPixelFormat:"), format);
        let _: () = msg1(d.id(), sel!("setWidth:"), desc.size.width.max(1) as NSUInteger);
        let _: () = msg1(d.id(), sel!("setHeight:"), desc.size.height.max(1) as NSUInteger);
        let _: () = msg1(d.id(), sel!("setArrayLength:"), desc.size.depth_or_array_layers as NSUInteger);
        let _: () = msg1(d.id(), sel!("setMipmapLevelCount:"), desc.mip_level_count.max(1) as NSUInteger);
        let _: () = msg1(d.id(), sel!("setUsage:"), usage);
        let _: () = msg1(d.id(), sel!("setStorageMode:"), storage_mode::PRIVATE);
        let tex: Id = msg1(device.device_id(), sel!("newTextureWithDescriptor:"), d.id());
        Owned::from_retained(tex).ok_or_else(|| crate::metal::MetalError::Allocation("2D array texture".into()))
    }
}

impl MTexture {
    /// A view, as wgpu's `create_view`: the texture itself when the view
    /// asks for all of it as it is, and otherwise a Metal texture view — one
    /// layer of an array as a 2D texture, or a 2D texture as a one-layer
    /// array, which is how a shader declaring `texture_2d_array` binds it.
    pub(crate) fn view(&self, desc: &wgpu::TextureViewDescriptor) -> MTexture {
        use wgpu::TextureViewDimension as D;
        let base = desc.base_array_layer.min(self.layers.saturating_sub(1));
        let count = desc.array_layer_count.unwrap_or(self.layers - base).clamp(1, self.layers - base);
        let natural = if self.layers > 1 { texture_type::TYPE_2D_ARRAY } else { texture_type::TYPE_2D };
        let ty = match desc.dimension {
            Some(D::D2) => texture_type::TYPE_2D,
            Some(D::D2Array) => texture_type::TYPE_2D_ARRAY,
            _ => natural,
        };
        if self.depth > 1 || (ty == natural && count == self.layers) {
            return self.clone();
        }
        let first_mip = desc.base_mip_level.min(self.mips - 1);
        let mips = desc.mip_level_count.unwrap_or(self.mips - first_mip);
        let _pool = AutoreleasePool::new();
        let obj = unsafe {
            let v: Id = msg4(
                self.obj.id(),
                sel!("newTextureViewWithPixelFormat:textureType:levels:slices:"),
                pixel_format(self.format),
                ty,
                NSRange { location: first_mip as NSUInteger, length: mips as NSUInteger },
                NSRange { location: base as NSUInteger, length: count as NSUInteger },
            );
            Owned::from_retained(v).expect("a Metal texture view")
        };
        MTexture { obj, format: self.format, width: self.width, height: self.height, mips, depth: 1, layers: count }
    }
}

pub(crate) fn pixel_format(f: wgpu::TextureFormat) -> NSUInteger {
    use wgpu::TextureFormat as F;
    match f {
        F::R8Unorm => 10,
        F::R8Uint => 13,
        F::R16Uint => 23,
        F::Rg8Unorm => 30,
        F::Rgba8Unorm => 70,
        F::Rgba8UnormSrgb => 71,
        F::Bgra8Unorm => 80,
        F::Bgra8UnormSrgb => 81,
        F::R16Float => 25,
        F::Rg16Float => 65,
        F::Rgba16Float => 115,
        F::R32Uint => 53,
        F::R32Float => 55,
        F::Rg32Uint => 103,
        F::Rg32Float => 105,
        F::Rgba32Uint => 123,
        F::Rgba32Float => 125,
        F::Depth32Float => 252,
        other => panic!("marks on Metal: no pixel format for {other:?}"),
    }
}

fn vertex_format(f: wgpu::VertexFormat) -> NSUInteger {
    use wgpu::VertexFormat as V;
    match f {
        V::Unorm8x4 => 9,
        V::Float32 => 28,
        V::Float32x2 => 29,
        V::Float32x3 => 30,
        V::Float32x4 => 31,
        V::Sint32 => 32,
        V::Sint32x2 => 33,
        V::Sint32x3 => 34,
        V::Sint32x4 => 35,
        V::Uint32 => 36,
        V::Uint32x2 => 37,
        V::Uint32x3 => 38,
        V::Uint32x4 => 39,
        other => panic!("marks on Metal: no vertex format for {other:?}"),
    }
}

fn blend_factor(f: wgpu::BlendFactor) -> NSUInteger {
    use wgpu::BlendFactor as B;
    match f {
        B::Zero => 0,
        B::One => 1,
        B::Src => 2,
        B::OneMinusSrc => 3,
        B::SrcAlpha => 4,
        B::OneMinusSrcAlpha => 5,
        B::Dst => 6,
        B::OneMinusDst => 7,
        B::DstAlpha => 8,
        B::OneMinusDstAlpha => 9,
        other => panic!("marks on Metal: no blend factor for {other:?}"),
    }
}

fn blend_operation(o: wgpu::BlendOperation) -> NSUInteger {
    match o {
        wgpu::BlendOperation::Add => 0,
        wgpu::BlendOperation::Subtract => 1,
        wgpu::BlendOperation::ReverseSubtract => 2,
        wgpu::BlendOperation::Min => 3,
        wgpu::BlendOperation::Max => 4,
    }
}

fn compare_function(c: wgpu::CompareFunction) -> NSUInteger {
    use wgpu::CompareFunction as C;
    match c {
        C::Never => 0,
        C::Less => 1,
        C::Equal => 2,
        C::LessEqual => 3,
        C::Greater => 4,
        C::NotEqual => 5,
        C::GreaterEqual => 6,
        C::Always => 7,
    }
}

fn load_action<T>(load: &wgpu::LoadOp<T>) -> NSUInteger {
    match load {
        wgpu::LoadOp::Clear(_) => load_action::CLEAR,
        _ => load_action::LOAD,
    }
}

// --------------------------------------------------------------- resources

/// A shared buffer's bytes, as the CPU sees them.
/// The last command buffer this module committed, until something needs its
/// results. A submit returns at once, as wgpu's does; what reads back (a map,
/// `poll_wait`) waits here first, and a `write_buffer` while it runs goes
/// through the queue rather than into memory the GPU may still be reading.
/// Command buffers on one queue run in the order they were committed, so the
/// last one finishing means all of them have.
static IN_FLIGHT: Mutex<Option<InFlight>> = Mutex::new(None);

struct InFlight(Owned);
// SAFETY: a committed `MTLCommandBuffer` may be waited on from any thread.
unsafe impl Send for InFlight {}

/// Remember `cmd`, just committed, as the one to wait for.
fn committed(cmd: &Owned) {
    let mut slot = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
    *slot = Some(InFlight(cmd.clone()));
}

/// Whether the last commit is still running.
fn busy() -> bool {
    let slot = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
    slot.as_ref().is_some_and(|c| {
        // Below MTLCommandBufferStatusCompleted (4).
        let status: NSUInteger = unsafe { msg0(c.0.id(), sel!("status")) };
        status < 4
    })
}

/// Wait for everything committed so far, and say if it failed.
pub fn wait_in_flight() {
    let cmd = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner()).take();
    let Some(InFlight(cmd)) = cmd else { return };
    unsafe {
        let _: () = msg0(cmd.id(), sel!("waitUntilCompleted"));
        let status: NSUInteger = msg0(cmd.id(), sel!("status"));
        // MTLCommandBufferStatusError.
        if status == 5 {
            let err: Id = msg0(cmd.id(), sel!("error"));
            log::warn!("marks on Metal: a command buffer failed: {}", crate::metal::objc::error_message(err));
        }
    }
}

/// `map_async` callbacks, each with the command buffer it waits for. They run
/// from a poll or a submit once that has finished, as wgpu's do — never
/// inside `map_async`, where a caller may hold the lock its callback takes.
static MAPS: Mutex<Vec<(Option<InFlight>, MapCallback)>> = Mutex::new(Vec::new());

type MapCallback = Box<dyn FnOnce() + Send>;

/// Run `callback` once the work committed before now has finished.
pub fn queue_map(callback: MapCallback) {
    let after = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|c| InFlight(c.0.clone()));
    MAPS.lock().unwrap_or_else(|e| e.into_inner()).push((after, callback));
}

/// Run the `map_async` callbacks whose work has finished: all of them after
/// waiting for the GPU when `wait`, otherwise only those already done.
pub fn run_maps(wait: bool) {
    if wait {
        wait_in_flight();
    }
    let ready: Vec<_> = {
        let mut maps = MAPS.lock().unwrap_or_else(|e| e.into_inner());
        let (ready, pending) = std::mem::take(&mut *maps).into_iter().partition(|(after, _)| {
            wait || after.as_ref().is_none_or(|c| {
                // MTLCommandBufferStatusCompleted (4) or Error (5).
                let status: NSUInteger = unsafe { msg0(c.0.id(), sel!("status")) };
                status >= 4
            })
        });
        *maps = pending;
        ready
    };
    // Outside the lock: a callback may queue another map.
    for (_, callback) in ready {
        callback();
    }
}

/// Copy `bytes` into `buffer`'s memory at `offset`, which no pass may be
/// reading.
fn copy_into(buffer: &MBuffer, offset: u64, bytes: &[u8]) {
    unsafe {
        let p: *mut u8 = msg0(buffer.obj.id(), sel!("contents"));
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(offset as usize), bytes.len());
    }
}

/// A buffer's bytes, once the work committed before now has finished.
pub fn mapped_bytes(buffer: &MBuffer) -> &[u8] {
    wait_in_flight();
    unsafe {
        let p: *const u8 = msg0(buffer.obj.id(), sel!("contents"));
        std::slice::from_raw_parts(p, buffer.size as usize)
    }
}

impl MGpu {
    pub fn limits(&self) -> wgpu::Limits {
        wgpu::Limits {
            max_texture_dimension_2d: if cfg!(target_os = "macos") { 16384 } else { 8192 },
            ..wgpu::Limits::default()
        }
    }

    pub fn create_buffer(&self, size: u64) -> MBuffer {
        let len = size.max(4) as NSUInteger;
        let obj = unsafe {
            let b: Id = msg2(
                self.device.device_id(),
                sel!("newBufferWithLength:options:"),
                len,
                resource_options::STORAGE_MODE_SHARED,
            );
            let b = Owned::from_retained(b).expect("a Metal buffer");
            // Zeroed, as wgpu's are: a binning pass counts into fresh ones.
            let p: *mut u8 = msg0(b.id(), sel!("contents"));
            std::ptr::write_bytes(p, 0, len as usize);
            b
        };
        MBuffer { obj, size }
    }

    pub fn create_buffer_init(&self, bytes: &[u8]) -> MBuffer {
        let b = self.create_buffer(bytes.len() as u64);
        // New, so no pass reads it: straight into memory. Through
        // `write_buffer` a busy GPU staged it in another new buffer, and that
        // one in another, until the stack ran out.
        copy_into(&b, 0, bytes);
        b
    }

    pub fn write_buffer(&self, buffer: &MBuffer, offset: u64, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let end = offset as usize + bytes.len();
        assert!(end as u64 <= buffer.size.max(4), "write past the end of a marks buffer");
        if !busy() {
            // Nothing committed is running: the memory is the CPU's.
            copy_into(buffer, offset, bytes);
            return;
        }
        // A pass may still read the old contents: copy on the queue, after
        // it and before the next submit, as wgpu's `write_buffer` orders.
        let staging = self.create_buffer_init(bytes);
        let _pool = AutoreleasePool::new();
        unsafe {
            let cmd = Owned::retain(self.device.command_buffer()).expect("a Metal command buffer");
            let blit: Id = msg0(cmd.id(), sel!("blitCommandEncoder"));
            let _: () = msg5(
                blit,
                sel!("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
                staging.obj.id(),
                0usize,
                buffer.obj.id(),
                offset as NSUInteger,
                bytes.len() as NSUInteger,
            );
            let _: () = msg0(blit, sel!("endEncoding"));
            let _: () = msg0(cmd.id(), sel!("commit"));
            committed(&cmd);
        }
    }


    pub fn create_texture(&self, desc: &wgpu::TextureDescriptor) -> MTexture {
        let mut usage = 0;
        if desc.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
            usage |= texture_usage::SHADER_READ;
        }
        if desc.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
            usage |= texture_usage::RENDER_TARGET;
        }
        if desc.usage.contains(wgpu::TextureUsages::STORAGE_BINDING) {
            usage |= texture_usage::SHADER_WRITE;
        }
        let volume = desc.dimension == wgpu::TextureDimension::D3;
        let layers = if volume { 1 } else { desc.size.depth_or_array_layers.max(1) };
        let obj = if layers > 1 {
            new_texture_2d_array(&self.device, desc, pixel_format(desc.format), usage)
        } else if volume {
            let s = desc.size;
            crate::metal::resources::new_texture_3d(
                &self.device,
                [s.width, s.height, s.depth_or_array_layers],
                pixel_format(desc.format),
                usage,
                storage_mode::PRIVATE,
            )
        } else {
            crate::metal::resources::new_texture_2d(
                &self.device,
                desc.size.width,
                desc.size.height,
                pixel_format(desc.format),
                usage,
                storage_mode::PRIVATE,
                desc.sample_count,
                desc.mip_level_count,
            )
        }
        .expect("a Metal texture");
        MTexture {
            obj,
            format: desc.format,
            width: desc.size.width.max(1),
            height: desc.size.height.max(1),
            mips: desc.mip_level_count.max(1),
            depth: if volume { desc.size.depth_or_array_layers.max(1) } else { 1 },
            layers,
        }
    }

    pub fn create_sampler(&self, desc: &wgpu::SamplerDescriptor) -> MSampler {
        let wrap = |m: wgpu::AddressMode| match m {
            wgpu::AddressMode::Repeat => 1,
            wgpu::AddressMode::MirrorRepeat => 2,
            _ => 0,
        };
        let key = crate::metal::resources::SamplerKey {
            mag_linear: desc.mag_filter == wgpu::FilterMode::Linear,
            min_linear: desc.min_filter == wgpu::FilterMode::Linear,
            mipmapped: desc.mipmap_filter == wgpu::MipmapFilterMode::Linear,
            mip_linear: desc.mipmap_filter == wgpu::MipmapFilterMode::Linear,
            wrap_s: wrap(desc.address_mode_u),
            wrap_t: wrap(desc.address_mode_v),
        };
        MSampler(crate::metal::resources::new_sampler(&self.device, key).expect("a Metal sampler"))
    }

    pub fn create_module(&self, source: String) -> MModule {
        MModule(Arc::new(ModuleSrc { source, parsed: OnceLock::new() }))
    }

    /// `bytes` into `texture` at `origin`, `bytes_per_row` apart, through a
    /// staging buffer and a blit (the texture is private to the GPU).
    pub fn write_texture(
        &self,
        texture: &MTexture,
        origin: wgpu::Origin3d,
        bytes: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        let bpp = texture.format.block_copy_size(None).unwrap_or(4) as u64;
        let row = layout.bytes_per_row.map_or(size.width as u64 * bpp, u64::from);
        let rows = layout.rows_per_image.unwrap_or(size.height) as u64;
        // A 3D texture's region is `depth` images deep, and an array's is
        // that many slices, copied one at a time; a 2D one's is one image.
        let images = if texture.depth > 1 || texture.layers > 1 { size.depth_or_array_layers.max(1) as u64 } else { 1 };
        let need = layout.offset + row * rows * (images - 1) + row * size.height as u64;
        if (bytes.len() as u64) < need {
            return;
        }
        let staging = self.create_buffer_init(bytes);
        let _pool = AutoreleasePool::new();
        // (source offset, slice, images in this copy, z)
        let copies: Vec<(u64, u64, u64, u64)> = if texture.layers > 1 {
            (0..images).map(|i| (layout.offset + row * rows * i, origin.z as u64 + i, 1, 0)).collect()
        } else if texture.depth > 1 {
            vec![(layout.offset, 0, images, origin.z as u64)]
        } else {
            vec![(layout.offset, 0, 1, 0)]
        };
        unsafe {
            let cmd = self.device.command_buffer();
            let blit: Id = msg0(cmd, sel!("blitCommandEncoder"));
            for (offset, slice, depth, z) in copies {
                let _: () = msg9(
                    blit,
                    sel!("copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
                    staging.obj.id(),
                    offset as NSUInteger,
                    row as NSUInteger,
                    (row * rows) as NSUInteger,
                    MTLSize { width: size.width as NSUInteger, height: size.height as NSUInteger, depth: depth as NSUInteger },
                    texture.obj.id(),
                    slice as NSUInteger,
                    0usize,
                    MTLOrigin { x: origin.x as NSUInteger, y: origin.y as NSUInteger, z: z as NSUInteger },
                );
            }
            let _: () = msg0(blit, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
            if let Some(cmd) = Owned::retain(cmd) {
                committed(&cmd);
            }
        }
    }

    pub fn create_encoder(&self) -> MEncoder {
        let _pool = AutoreleasePool::new();
        let cmd = unsafe { Owned::retain(self.device.command_buffer()) }.expect("a Metal command buffer");
        MEncoder { cmd }
    }

    /// Commit, and return without waiting: see `IN_FLIGHT`.
    pub fn submit(&self, encoder: MEncoder) {
        unsafe {
            let _: () = msg0(encoder.cmd.id(), sel!("commit"));
        }
        committed(&encoder.cmd);
    }
}

// ----------------------------------------------------------------- shaders

impl ModuleSrc {
    fn parsed(&self) -> Result<&(wgpu::naga::Module, wgpu::naga::valid::ModuleInfo), String> {
        self.parsed
            .get_or_init(|| {
                use wgpu::naga::valid;
                let module = wgpu::naga::front::wgsl::parse_str(&self.source)
                    .map_err(|e| e.emit_to_string(&self.source))?;
                let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
                    .validate(&module)
                    .map_err(|e| format!("{e:?}"))?;
                Ok((module, info))
            })
            .as_ref()
            .map_err(|e| e.clone())
    }
}

/// Slots for a layout: each group's buffers, textures and samplers in turn.
fn slots_for(layouts: &[MLayout]) -> Slots {
    let mut slots = Slots::new();
    let mut next = [0u8; 3];
    for (g, layout) in layouts.iter().enumerate() {
        let mut entries: Vec<&wgpu::BindGroupLayoutEntry> = layout.0.iter().collect();
        entries.sort_by_key(|e| e.binding);
        for e in entries {
            let kind = match e.ty {
                wgpu::BindingType::Buffer { .. } => 0,
                wgpu::BindingType::Texture { .. } | wgpu::BindingType::StorageTexture { .. } => 1,
                wgpu::BindingType::Sampler(_) => 2,
                _ => continue,
            };
            slots.insert((g as u32, e.binding), (kind, next[kind as usize]));
            next[kind as usize] += 1;
        }
    }
    slots
}

impl MGpu {
    /// The module translated for `slots` and compiled, and its entry points'
    /// names in MSL.
    fn library(&self, module: &MModule, slots: &Slots, layouts: &[MLayout]) -> Result<(Owned, Names, Sizes), String> {
        use wgpu::naga::back::msl;
        let (naga_module, info) = module.0.parsed()?;
        let sizes = runtime_sized(naga_module, slots);
        let mut resources = msl::BindingMap::new();
        for (g, layout) in layouts.iter().enumerate() {
            for e in layout.0.iter() {
                let Some(&(kind, slot)) = slots.get(&(g as u32, e.binding)) else { continue };
                let mut target = msl::BindTarget::default();
                match kind {
                    0 => {
                        target.buffer = Some(slot);
                        target.mutable = matches!(
                            e.ty,
                            wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, .. }
                        );
                    }
                    1 => target.texture = Some(slot),
                    _ => target.sampler = Some(msl::BindSamplerTarget::Resource(slot)),
                }
                resources.insert(wgpu::naga::ResourceBinding { group: g as u32, binding: e.binding }, target);
            }
        }
        let entry = msl::EntryPointResources {
            resources,
            sizes_buffer: sizes.as_ref().map(|(slot, _)| *slot),
            ..Default::default()
        };
        let options = msl::Options {
            lang_version: (2, 1),
            per_entry_point_map: naga_module.entry_points.iter().map(|ep| (ep.name.clone(), entry.clone())).collect(),
            fake_missing_bindings: false,
            ..Default::default()
        };
        let (text, tinfo) = msl::write_string(naga_module, info, &options, &msl::PipelineOptions::default())
            .map_err(|e| format!("{e:?}"))?;
        let names: Names = naga_module
            .entry_points
            .iter()
            .zip(&tinfo.entry_point_names)
            .map(|(ep, n)| (ep.name.clone(), n.clone().map_err(|e| format!("{e:?}"))))
            .collect();
        let mut libs = self.libraries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(lib) = libs.get(&text) {
            return Ok((lib.clone(), names, sizes));
        }
        let _pool = AutoreleasePool::new();
        let lib = unsafe { crate::metal::compile_library(self.device.device_id(), &text) }.map_err(|e| e.to_string())?;
        libs.insert(text, lib.clone());
        Ok((lib, names, sizes))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_render_pipeline(
        &self,
        layouts: &[MLayout],
        vertex_module: &MModule,
        vertex_entry: &str,
        buffers: &[Option<wgpu::VertexBufferLayout>],
        fragment: Option<(&MModule, &str, &[Option<wgpu::ColorTargetState>])>,
        primitive: &wgpu::PrimitiveState,
        depth_stencil: Option<&wgpu::DepthStencilState>,
        samples: u32,
    ) -> Result<MPipeline, String> {
        let slots = slots_for(layouts);
        let (vlib, vnames, sizes) = self.library(vertex_module, &slots, layouts)?;
        let fragment_lib = match fragment {
            Some((m, entry, targets)) => {
                let (lib, names, _) = self.library(m, &slots, layouts)?;
                Some((lib, entry_name(&names, entry)?, targets))
            }
            None => None,
        };
        let _pool = AutoreleasePool::new();
        let vs = crate::metal::resources::library_function(vlib.id(), &entry_name(&vnames, vertex_entry)?)
            .map_err(|e| e.to_string())?;
        unsafe {
            let desc = Owned::from_retained(alloc_init(class!("MTLRenderPipelineDescriptor"))).ok_or("descriptor")?;
            let _: () = msg1(desc.id(), sel!("setVertexFunction:"), vs.id());
            let mut _fs_keep = None;
            if let Some((lib, name, targets)) = &fragment_lib {
                let fs = crate::metal::resources::library_function(lib.id(), name).map_err(|e| e.to_string())?;
                let _: () = msg1(desc.id(), sel!("setFragmentFunction:"), fs.id());
                _fs_keep = Some(fs);
                let attachments: Id = msg0(desc.id(), sel!("colorAttachments"));
                for (i, t) in targets.iter().enumerate() {
                    let Some(t) = t else { continue };
                    let a: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), i);
                    let _: () = msg1(a, sel!("setPixelFormat:"), pixel_format(t.format));
                    // wgpu's bits run R G B A from 1; Metal's from A.
                    let w = t.write_mask;
                    let mask = (u64::from(w.contains(wgpu::ColorWrites::RED)) << 3)
                        | (u64::from(w.contains(wgpu::ColorWrites::GREEN)) << 2)
                        | (u64::from(w.contains(wgpu::ColorWrites::BLUE)) << 1)
                        | u64::from(w.contains(wgpu::ColorWrites::ALPHA));
                    let _: () = msg1(a, sel!("setWriteMask:"), mask as NSUInteger);
                    if let Some(b) = t.blend {
                        let _: () = msg1(a, sel!("setBlendingEnabled:"), YES);
                        let _: () = msg1(a, sel!("setRgbBlendOperation:"), blend_operation(b.color.operation));
                        let _: () = msg1(a, sel!("setAlphaBlendOperation:"), blend_operation(b.alpha.operation));
                        let _: () = msg1(a, sel!("setSourceRGBBlendFactor:"), blend_factor(b.color.src_factor));
                        let _: () = msg1(a, sel!("setDestinationRGBBlendFactor:"), blend_factor(b.color.dst_factor));
                        let _: () = msg1(a, sel!("setSourceAlphaBlendFactor:"), blend_factor(b.alpha.src_factor));
                        let _: () = msg1(a, sel!("setDestinationAlphaBlendFactor:"), blend_factor(b.alpha.dst_factor));
                    }
                }
            }
            if let Some(d) = depth_stencil {
                let _: () = msg1(desc.id(), sel!("setDepthAttachmentPixelFormat:"), pixel_format(d.format));
            }
            let _: () = msg1(desc.id(), sel!("setRasterSampleCount:"), samples.max(1) as NSUInteger);
            if !buffers.is_empty() {
                let vd: Id = msg0(class!("MTLVertexDescriptor"), sel!("vertexDescriptor"));
                let attrs: Id = msg0(vd, sel!("attributes"));
                let lays: Id = msg0(vd, sel!("layouts"));
                for (i, b) in buffers.iter().enumerate() {
                    let Some(b) = b else { continue };
                    let index = TOP_BUFFER - i as u64;
                    let l: Id = msg1(lays, sel!("objectAtIndexedSubscript:"), index as NSUInteger);
                    let _: () = msg1(l, sel!("setStride:"), b.array_stride.max(4) as NSUInteger);
                    let step: NSUInteger = match b.step_mode {
                        wgpu::VertexStepMode::Vertex => 1,
                        wgpu::VertexStepMode::Instance => 2,
                    };
                    let _: () = msg1(l, sel!("setStepFunction:"), step);
                    let _: () = msg1(l, sel!("setStepRate:"), 1usize);
                    for a in b.attributes {
                        let at: Id = msg1(attrs, sel!("objectAtIndexedSubscript:"), a.shader_location as NSUInteger);
                        let _: () = msg1(at, sel!("setFormat:"), vertex_format(a.format));
                        let _: () = msg1(at, sel!("setOffset:"), a.offset as NSUInteger);
                        let _: () = msg1(at, sel!("setBufferIndex:"), index as NSUInteger);
                    }
                }
                let _: () = msg1(desc.id(), sel!("setVertexDescriptor:"), vd);
            }
            let mut err: Id = NIL;
            let state: Id = msg2(
                self.device.device_id(),
                sel!("newRenderPipelineStateWithDescriptor:error:"),
                desc.id(),
                &mut err as *mut Id,
            );
            let state = Owned::from_retained(state).ok_or_else(|| crate::metal::objc::error_message(err))?;
            let depth = match depth_stencil {
                Some(d) => {
                    let dd = Owned::from_retained(alloc_init(class!("MTLDepthStencilDescriptor"))).ok_or("depth descriptor")?;
                    let compare = d.depth_compare.unwrap_or(wgpu::CompareFunction::Always);
                    let _: () = msg1(dd.id(), sel!("setDepthCompareFunction:"), compare_function(compare));
                    let _: () = msg1(dd.id(), sel!("setDepthWriteEnabled:"), d.depth_write_enabled.unwrap_or(false));
                    let s: Id = msg1(self.device.device_id(), sel!("newDepthStencilStateWithDescriptor:"), dd.id());
                    Owned::from_retained(s)
                }
                None => None,
            };
            let bias = depth_stencil
                .filter(|d| d.bias.constant != 0 || d.bias.slope_scale != 0.0)
                .map(|d| (d.bias.constant as f32, d.bias.slope_scale));
            Ok(MPipeline(Arc::new(PipelineInner {
                state,
                depth,
                primitive: match primitive.topology {
                    wgpu::PrimitiveTopology::LineList => primitive::LINE,
                    wgpu::PrimitiveTopology::LineStrip => primitive::LINE_STRIP,
                    wgpu::PrimitiveTopology::PointList => primitive::POINT,
                    wgpu::PrimitiveTopology::TriangleStrip => primitive::TRIANGLE_STRIP,
                    wgpu::PrimitiveTopology::TriangleList => primitive::TRIANGLE,
                },
                cull: match primitive.cull_mode {
                    Some(wgpu::Face::Back) => cull::BACK,
                    Some(wgpu::Face::Front) => cull::FRONT,
                    None => cull::NONE,
                },
                winding: match primitive.front_face {
                    wgpu::FrontFace::Ccw => winding::COUNTER_CLOCKWISE,
                    wgpu::FrontFace::Cw => winding::CLOCKWISE,
                },
                bias,
                slots,
                sizes,
            })))
        }
    }
}

impl MGpu {
    pub fn create_compute_pipeline(&self, layouts: &[MLayout], module: &MModule, entry: &str) -> Result<MComputePipeline, String> {
        let slots = slots_for(layouts);
        let (lib, names, sizes) = self.library(module, &slots, layouts)?;
        let (naga_module, _) = module.0.parsed()?;
        let threads = naga_module
            .entry_points
            .iter()
            .find(|ep| ep.name == entry)
            .map_or([64, 1, 1], |ep| ep.workgroup_size);
        let _pool = AutoreleasePool::new();
        let f = crate::metal::resources::library_function(lib.id(), &entry_name(&names, entry)?)
            .map_err(|e| e.to_string())?;
        unsafe {
            let mut err: Id = NIL;
            let state: Id = msg2(
                self.device.device_id(),
                sel!("newComputePipelineStateWithFunction:error:"),
                f.id(),
                &mut err as *mut Id,
            );
            let state = Owned::from_retained(state).ok_or_else(|| crate::metal::objc::error_message(err))?;
            Ok(MComputePipeline(Arc::new(ComputeInner { state, slots, threads, sizes })))
        }
    }
}

/// Bind the set groups' resources at `slots`: in both render stages, or the
/// compute stage.
fn bind_groups(e: Id, slots: &Slots, groups: &[Option<(MBindGroup, Vec<u32>)>], compute: bool) {
    for (g, group) in groups.iter().enumerate() {
        let Some((group, offsets)) = group else { continue };
        // Dynamic offsets apply in binding order to the entries that take one.
        let mut dynamic: Vec<u32> = group
            .layout
            .0
            .iter()
            .filter(|l| matches!(l.ty, wgpu::BindingType::Buffer { has_dynamic_offset: true, .. }))
            .map(|l| l.binding)
            .collect();
        dynamic.sort_unstable();
        for (binding, res) in group.entries.iter() {
            let Some(&(_, slot)) = slots.get(&(g as u32, *binding)) else { continue };
            let slot = slot as NSUInteger;
            unsafe {
                match res {
                    MRes::Buffer(b, off) => {
                        let extra = dynamic
                            .iter()
                            .position(|d| d == binding)
                            .and_then(|k| offsets.get(k))
                            .copied()
                            .unwrap_or(0) as u64;
                        let o = (off + extra) as NSUInteger;
                        if compute {
                            let _: () = msg3(e, sel!("setBuffer:offset:atIndex:"), b.id(), o, slot);
                        } else {
                            let _: () = msg3(e, sel!("setVertexBuffer:offset:atIndex:"), b.id(), o, slot);
                            let _: () = msg3(e, sel!("setFragmentBuffer:offset:atIndex:"), b.id(), o, slot);
                        }
                    }
                    MRes::Texture(t) => {
                        if compute {
                            let _: () = msg2(e, sel!("setTexture:atIndex:"), t.id(), slot);
                        } else {
                            let _: () = msg2(e, sel!("setVertexTexture:atIndex:"), t.id(), slot);
                            let _: () = msg2(e, sel!("setFragmentTexture:atIndex:"), t.id(), slot);
                        }
                    }
                    MRes::Sampler(s) => {
                        if compute {
                            let _: () = msg2(e, sel!("setSamplerState:atIndex:"), s.id(), slot);
                        } else {
                            let _: () = msg2(e, sel!("setVertexSamplerState:atIndex:"), s.id(), slot);
                            let _: () = msg2(e, sel!("setFragmentSamplerState:atIndex:"), s.id(), slot);
                        }
                    }
                }
            }
        }
    }
}

impl MComputePass {
    pub fn set_pipeline(&mut self, pipeline: &MComputePipeline) {
        unsafe {
            let _: () = msg1(self.enc.id(), sel!("setComputePipelineState:"), pipeline.0.state.id());
        }
        self.pipeline = Some(pipeline.clone());
    }

    pub fn set_bind_group(&mut self, index: u32, group: &MBindGroup, offsets: &[u32]) {
        let i = index as usize;
        if self.groups.len() <= i {
            self.groups.resize(i + 1, None);
        }
        self.groups[i] = Some((group.clone(), offsets.to_vec()));
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        let Some(p) = &self.pipeline else { return };
        bind_groups(self.enc.id(), &p.0.slots, &self.groups, true);
        if let Some((slot, list)) = &p.0.sizes {
            let bytes = sizes_bytes(list, &self.groups);
            let len = std::mem::size_of_val(bytes.as_slice()) as NSUInteger;
            unsafe {
                let _: () = msg3(self.enc.id(), sel!("setBytes:length:atIndex:"), bytes.as_ptr(), len, *slot as NSUInteger);
            }
        }
        let size = |v: [u32; 3]| MTLSize { width: v[0] as NSUInteger, height: v[1] as NSUInteger, depth: v[2] as NSUInteger };
        unsafe {
            let _: () = msg2(
                self.enc.id(),
                sel!("dispatchThreadgroups:threadsPerThreadgroup:"),
                size([x, y, z]),
                size(p.0.threads),
            );
        }
    }
}

// ------------------------------------------------------------------ passes

/// One colour attachment: `(texture, resolve, load, clear, store)`.
pub struct ColorAttachment {
    pub texture: Id,
    pub resolve: Option<Id>,
    pub load: wgpu::LoadOp<wgpu::Color>,
    pub store: wgpu::StoreOp,
}

pub struct DepthAttachment {
    pub texture: Id,
    pub load: wgpu::LoadOp<f32>,
    pub store: wgpu::StoreOp,
}

impl MEncoder {
    pub fn begin_render_pass(&mut self, color: &[ColorAttachment], depth: Option<DepthAttachment>) -> MPass {
        let _pool = AutoreleasePool::new();
        unsafe {
            let desc: Id = msg0(class!("MTLRenderPassDescriptor"), sel!("renderPassDescriptor"));
            let attachments: Id = msg0(desc, sel!("colorAttachments"));
            for (i, c) in color.iter().enumerate() {
                let a: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), i);
                let _: () = msg1(a, sel!("setTexture:"), c.texture);
                let _: () = msg1(a, sel!("setLoadAction:"), load_action(&c.load));
                if let wgpu::LoadOp::Clear(v) = c.load {
                    let _: () = msg1(a, sel!("setClearColor:"), MTLClearColor { red: v.r, green: v.g, blue: v.b, alpha: v.a });
                }
                let store = matches!(c.store, wgpu::StoreOp::Store);
                let action = match (c.resolve, store) {
                    (Some(r), s) => {
                        let _: () = msg1(a, sel!("setResolveTexture:"), r);
                        if s { store_action::STORE_AND_MULTISAMPLE_RESOLVE } else { store_action::MULTISAMPLE_RESOLVE }
                    }
                    (None, true) => store_action::STORE,
                    (None, false) => store_action::DONT_CARE,
                };
                let _: () = msg1(a, sel!("setStoreAction:"), action);
            }
            if let Some(d) = depth {
                let a: Id = msg0(desc, sel!("depthAttachment"));
                let _: () = msg1(a, sel!("setTexture:"), d.texture);
                let _: () = msg1(a, sel!("setLoadAction:"), load_action(&d.load));
                if let wgpu::LoadOp::Clear(v) = d.load {
                    let _: () = msg1(a, sel!("setClearDepth:"), v as f64);
                }
                let store = if matches!(d.store, wgpu::StoreOp::Store) { store_action::STORE } else { store_action::DONT_CARE };
                let _: () = msg1(a, sel!("setStoreAction:"), store);
            }
            let enc: Id = msg1(self.cmd.id(), sel!("renderCommandEncoderWithDescriptor:"), desc);
            MPass { enc: Owned::retain(enc).expect("a render encoder"), pipeline: None, groups: Vec::new(), index: None }
        }
    }

    pub fn begin_compute_pass(&mut self) -> MComputePass {
        let _pool = AutoreleasePool::new();
        unsafe {
            let enc: Id = msg0(self.cmd.id(), sel!("computeCommandEncoder"));
            MComputePass { enc: Owned::retain(enc).expect("a compute encoder"), pipeline: None, groups: Vec::new() }
        }
    }

    pub fn copy_buffer_to_buffer(&mut self, src: &MBuffer, src_offset: u64, dst: &MBuffer, dst_offset: u64, size: u64) {
        unsafe {
            let blit: Id = msg0(self.cmd.id(), sel!("blitCommandEncoder"));
            let _: () = msg5(
                blit,
                sel!("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
                src.obj.id(),
                src_offset as NSUInteger,
                dst.obj.id(),
                dst_offset as NSUInteger,
                size as NSUInteger,
            );
            let _: () = msg0(blit, sel!("endEncoding"));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn copy_texture_to_buffer(
        &mut self,
        texture: &MTexture,
        origin: wgpu::Origin3d,
        buffer: &MBuffer,
        offset: u64,
        bytes_per_row: u64,
        rows: u64,
        size: wgpu::Extent3d,
    ) {
        unsafe {
            let blit: Id = msg0(self.cmd.id(), sel!("blitCommandEncoder"));
            let _: () = msg9(
                blit,
                sel!("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:"),
                texture.obj.id(),
                // An array's layer is its slice, as wgpu's `origin.z`.
                if texture.layers > 1 { origin.z as NSUInteger } else { 0 },
                0usize,
                MTLOrigin { x: origin.x as NSUInteger, y: origin.y as NSUInteger, z: 0 },
                MTLSize { width: size.width as NSUInteger, height: size.height as NSUInteger, depth: 1 },
                buffer.obj.id(),
                offset as NSUInteger,
                bytes_per_row as NSUInteger,
                (bytes_per_row * rows) as NSUInteger,
            );
            let _: () = msg0(blit, sel!("endEncoding"));
        }
    }
}

impl MPass {
    pub fn set_pipeline(&mut self, pipeline: &MPipeline) {
        let p = &pipeline.0;
        unsafe {
            let e = self.enc.id();
            let _: () = msg1(e, sel!("setRenderPipelineState:"), p.state.id());
            if let Some(d) = &p.depth {
                let _: () = msg1(e, sel!("setDepthStencilState:"), d.id());
            }
            let _: () = msg1(e, sel!("setCullMode:"), p.cull);
            let _: () = msg1(e, sel!("setFrontFacingWinding:"), p.winding);
            let (c, s) = p.bias.unwrap_or((0.0, 0.0));
            let _: () = msg3(e, sel!("setDepthBias:slopeScale:clamp:"), c, s, 0.0f32);
        }
        self.pipeline = Some(pipeline.clone());
    }

    pub fn set_bind_group(&mut self, index: u32, group: &MBindGroup, offsets: &[u32]) {
        let i = index as usize;
        if self.groups.len() <= i {
            self.groups.resize(i + 1, None);
        }
        self.groups[i] = Some((group.clone(), offsets.to_vec()));
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &MBuffer, offset: u64) {
        unsafe {
            let _: () = msg3(
                self.enc.id(),
                sel!("setVertexBuffer:offset:atIndex:"),
                buffer.obj.id(),
                offset as NSUInteger,
                (TOP_BUFFER - slot as u64) as NSUInteger,
            );
        }
    }

    pub fn set_index_buffer(&mut self, buffer: &MBuffer, offset: u64, format: wgpu::IndexFormat) {
        let t = match format {
            wgpu::IndexFormat::Uint16 => index_type::UINT16,
            wgpu::IndexFormat::Uint32 => index_type::UINT32,
        };
        self.index = Some((buffer.obj.clone(), offset, t));
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        let vp = MTLViewport {
            origin_x: x as f64,
            origin_y: y as f64,
            width: w as f64,
            height: h as f64,
            znear: min_depth as f64,
            zfar: max_depth as f64,
        };
        unsafe {
            let _: () = msg1(self.enc.id(), sel!("setViewport:"), vp);
        }
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        let r = MTLScissorRect { x: x as NSUInteger, y: y as NSUInteger, width: w as NSUInteger, height: h as NSUInteger };
        unsafe {
            let _: () = msg1(self.enc.id(), sel!("setScissorRect:"), r);
        }
    }

    /// The set groups' resources at the current pipeline's slots, both stages.
    fn bind(&self) {
        if let Some(p) = &self.pipeline {
            bind_groups(self.enc.id(), &p.0.slots, &self.groups, false);
            if let Some((slot, list)) = &p.0.sizes {
                let bytes = sizes_bytes(list, &self.groups);
                let len = std::mem::size_of_val(bytes.as_slice()) as NSUInteger;
                unsafe {
                    let e = self.enc.id();
                    let _: () = msg3(e, sel!("setVertexBytes:length:atIndex:"), bytes.as_ptr(), len, *slot as NSUInteger);
                    let _: () = msg3(e, sel!("setFragmentBytes:length:atIndex:"), bytes.as_ptr(), len, *slot as NSUInteger);
                }
            }
        }
    }

    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        let Some(p) = &self.pipeline else { return };
        if vertices.is_empty() || instances.is_empty() {
            return;
        }
        self.bind();
        unsafe {
            let _: () = msg5(
                self.enc.id(),
                sel!("drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"),
                p.0.primitive,
                vertices.start as NSUInteger,
                vertices.len() as NSUInteger,
                instances.len() as NSUInteger,
                instances.start as NSUInteger,
            );
        }
    }

    pub fn draw_indexed(&mut self, indices: std::ops::Range<u32>, base_vertex: i32, instances: std::ops::Range<u32>) {
        let (Some(p), Some((buffer, offset, t))) = (&self.pipeline, &self.index) else { return };
        if indices.is_empty() || instances.is_empty() {
            return;
        }
        self.bind();
        let size = if *t == index_type::UINT16 { 2 } else { 4 };
        unsafe {
            let _: () = msg8(
                self.enc.id(),
                sel!("drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                p.0.primitive,
                indices.len() as NSUInteger,
                *t,
                buffer.id(),
                (offset + indices.start as u64 * size) as NSUInteger,
                instances.len() as NSUInteger,
                base_vertex as isize,
                instances.start as NSUInteger,
            );
        }
    }
}
