//! The GPU marks draw with: wgpu, or the native Metal backend.
//!
//! The same calls the marks make of wgpu — buffers, textures, bind groups,
//! pipelines from WGSL, render passes, copies — on either. Descriptors that
//! are plain data are wgpu's own; those that name GPU objects are copies of
//! wgpu's with these objects in them. On wgpu each object *is* the wgpu one,
//! so nothing about a wgpu frame changes. On Metal the WGSL goes through
//! naga to MSL; see `gpu_metal` for how bindings find their slots.

// Without the Metal backend every enum here has the one wgpu variant, and
// the matches that pick it apart are irrefutable.
#![cfg_attr(not(metal_backend), allow(irrefutable_let_patterns))]

use std::ops::{Bound, Deref, Range, RangeBounds};
#[cfg(metal_backend)]
use std::sync::Arc;

#[cfg(metal_backend)]
use super::gpu_metal as m;

/// A device and its queue.
#[derive(Clone)]
pub enum Gpu {
    Wgpu {
        device: wgpu::Device,
        queue: Option<wgpu::Queue>,
    },
    #[cfg(metal_backend)]
    Metal(Arc<m::MGpu>),
}

/// What [`PrepareContext`](super::PrepareContext) calls the queue: the same
/// handle.
pub type Queue = Gpu;
pub type Device = Gpu;

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.is_metal() {
            "Gpu::Metal"
        } else {
            "Gpu::Wgpu"
        })
    }
}

impl From<(&wgpu::Device, &wgpu::Queue)> for Gpu {
    fn from((device, queue): (&wgpu::Device, &wgpu::Queue)) -> Self {
        Gpu::Wgpu {
            device: device.clone(),
            queue: Some(queue.clone()),
        }
    }
}

impl From<&Gpu> for Gpu {
    fn from(gpu: &Gpu) -> Self {
        gpu.clone()
    }
}

impl From<&wgpu::Device> for Gpu {
    fn from(device: &wgpu::Device) -> Self {
        Gpu::Wgpu {
            device: device.clone(),
            queue: None,
        }
    }
}

impl From<&std::sync::Arc<wgpu::Device>> for Gpu {
    fn from(device: &std::sync::Arc<wgpu::Device>) -> Self {
        Gpu::from(&**device)
    }
}

#[cfg(metal_backend)]
impl From<&crate::metal::MetalDevice> for Gpu {
    fn from(device: &crate::metal::MetalDevice) -> Self {
        Gpu::Metal(Arc::new(m::MGpu::new(device.clone())))
    }
}

macro_rules! wrapper {
    ($(#[$doc:meta])* $name:ident, $wgpu:ty, $metal:ident) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        pub enum $name {
            Wgpu($wgpu),
            #[cfg(metal_backend)]
            Metal(m::$metal),
        }
        impl From<$wgpu> for $name {
            fn from(v: $wgpu) -> Self {
                Self::Wgpu(v)
            }
        }
        impl From<&$wgpu> for $name {
            fn from(v: &$wgpu) -> Self {
                Self::Wgpu(v.clone())
            }
        }
    };
}

wrapper!(Buffer, wgpu::Buffer, MBuffer);
wrapper!(Texture, wgpu::Texture, MTexture);
wrapper!(
    /// A texture as bound or drawn into. On Metal, the texture itself.
    TextureView,
    wgpu::TextureView,
    MTexture
);
wrapper!(Sampler, wgpu::Sampler, MSampler);

impl From<std::sync::Arc<wgpu::TextureView>> for TextureView {
    fn from(v: std::sync::Arc<wgpu::TextureView>) -> Self {
        Self::Wgpu((*v).clone())
    }
}
wrapper!(BindGroupLayout, wgpu::BindGroupLayout, MLayout);
wrapper!(BindGroup, wgpu::BindGroup, MBindGroup);
wrapper!(PipelineLayout, wgpu::PipelineLayout, MPipelineLayout);
wrapper!(ShaderModule, wgpu::ShaderModule, MModule);
wrapper!(RenderPipeline, wgpu::RenderPipeline, MPipeline);
wrapper!(ComputePipeline, wgpu::ComputePipeline, MComputePipeline);

#[cfg(metal_backend)]
impl Buffer {
    /// A Metal buffer, shared (retained while this lives).
    ///
    /// # Safety
    /// `buffer` is a live `MTLBuffer` at least `size` bytes long.
    pub unsafe fn from_metal(buffer: crate::metal::objc::Id, size: u64) -> Self {
        let obj = crate::metal::objc::Owned::retain(buffer).expect("a Metal buffer");
        Self::Metal(m::MBuffer { obj, size })
    }
}

#[cfg(metal_backend)]
impl TextureView {
    /// A Metal texture, borrowed for a pass (retained while the view lives).
    ///
    /// # Safety
    /// `texture` is a live `MTLTexture` of `format` and `width × height`.
    pub unsafe fn from_metal(
        texture: crate::metal::objc::Id,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let obj = crate::metal::objc::Owned::retain(texture).expect("a Metal texture");
        Self::Metal(m::MTexture {
            obj,
            format,
            width,
            height,
            mips: 1,
            depth: 1,
            layers: 1,
        })
    }
}

// ------------------------------------------------------------- descriptors

pub struct BindGroupDescriptor<'a> {
    pub label: wgpu::Label<'a>,
    pub layout: &'a BindGroupLayout,
    pub entries: &'a [BindGroupEntry<'a>],
}

pub struct BindGroupEntry<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

pub enum BindingResource<'a> {
    Buffer(BufferBinding<'a>),
    TextureView(&'a TextureView),
    Sampler(&'a Sampler),
}

pub struct BufferBinding<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: Option<wgpu::BufferSize>,
}

pub struct PipelineLayoutDescriptor<'a> {
    pub label: wgpu::Label<'a>,
    pub bind_group_layouts: &'a [Option<&'a BindGroupLayout>],
    pub immediate_size: u32,
}

pub struct ShaderModuleDescriptor<'a> {
    pub label: wgpu::Label<'a>,
    pub source: ShaderSource<'a>,
}

pub enum ShaderSource<'a> {
    Wgsl(std::borrow::Cow<'a, str>),
}

pub struct VertexState<'a> {
    pub module: &'a ShaderModule,
    pub entry_point: Option<&'a str>,
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    pub compilation_options: wgpu::PipelineCompilationOptions<'a>,
}

pub struct FragmentState<'a> {
    pub module: &'a ShaderModule,
    pub entry_point: Option<&'a str>,
    pub targets: &'a [Option<wgpu::ColorTargetState>],
    pub compilation_options: wgpu::PipelineCompilationOptions<'a>,
}

pub struct RenderPipelineDescriptor<'a> {
    pub label: wgpu::Label<'a>,
    pub layout: Option<&'a PipelineLayout>,
    pub vertex: VertexState<'a>,
    pub fragment: Option<FragmentState<'a>>,
    pub primitive: wgpu::PrimitiveState,
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub multisample: wgpu::MultisampleState,
    pub multiview_mask: Option<std::num::NonZeroU32>,
    pub cache: Option<()>,
}

pub struct ComputePipelineDescriptor<'a> {
    pub label: wgpu::Label<'a>,
    pub layout: Option<&'a PipelineLayout>,
    pub module: &'a ShaderModule,
    pub entry_point: Option<&'a str>,
    pub compilation_options: wgpu::PipelineCompilationOptions<'a>,
    pub cache: Option<()>,
}

pub struct RenderPassColorAttachment<'a> {
    pub view: &'a TextureView,
    pub depth_slice: Option<u32>,
    pub resolve_target: Option<&'a TextureView>,
    pub ops: wgpu::Operations<wgpu::Color>,
}

pub struct RenderPassDepthStencilAttachment<'a> {
    pub view: &'a TextureView,
    pub depth_ops: Option<wgpu::Operations<f32>>,
    pub stencil_ops: Option<wgpu::Operations<u32>>,
}

pub struct RenderPassDescriptor<'a> {
    pub label: wgpu::Label<'a>,
    pub color_attachments: &'a [Option<RenderPassColorAttachment<'a>>],
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'a>>,
    pub timestamp_writes: Option<()>,
    pub occlusion_query_set: Option<()>,
    pub multiview_mask: Option<std::num::NonZeroU32>,
}

pub type TexelCopyTextureInfo<'a> = wgpu::TexelCopyTextureInfoBase<&'a Texture>;
pub type TexelCopyBufferInfo<'a> = wgpu::TexelCopyBufferInfoBase<&'a Buffer>;

// ------------------------------------------------------------------ device

fn wq(queue: &Option<wgpu::Queue>) -> &wgpu::Queue {
    queue
        .as_ref()
        .expect("a marks GPU made from a device alone has no queue; give it the queue too")
}

impl Gpu {
    pub fn is_metal(&self) -> bool {
        !matches!(self, Gpu::Wgpu { .. })
    }

    /// The wgpu device, on wgpu.
    pub fn wgpu_device(&self) -> Option<&wgpu::Device> {
        match self {
            Gpu::Wgpu { device, .. } => Some(device),
            #[cfg(metal_backend)]
            Gpu::Metal(_) => None,
        }
    }

    pub fn limits(&self) -> wgpu::Limits {
        match self {
            Gpu::Wgpu { device, .. } => device.limits(),
            #[cfg(metal_backend)]
            Gpu::Metal(g) => g.limits(),
        }
    }

    pub fn create_buffer(&self, desc: &wgpu::BufferDescriptor) -> Buffer {
        match self {
            Gpu::Wgpu { device, .. } => Buffer::Wgpu(device.create_buffer(desc)),
            #[cfg(metal_backend)]
            Gpu::Metal(g) => Buffer::Metal(g.create_buffer(desc.size)),
        }
    }

    pub fn create_buffer_init(&self, desc: &wgpu::util::BufferInitDescriptor) -> Buffer {
        match self {
            Gpu::Wgpu { device, .. } => {
                Buffer::Wgpu(wgpu::util::DeviceExt::create_buffer_init(device, desc))
            }
            #[cfg(metal_backend)]
            Gpu::Metal(g) => Buffer::Metal(g.create_buffer_init(desc.contents)),
        }
    }

    pub fn create_texture(&self, desc: &wgpu::TextureDescriptor) -> Texture {
        match self {
            Gpu::Wgpu { device, .. } => Texture::Wgpu(device.create_texture(desc)),
            #[cfg(metal_backend)]
            Gpu::Metal(g) => Texture::Metal(g.create_texture(desc)),
        }
    }

    pub fn create_sampler(&self, desc: &wgpu::SamplerDescriptor) -> Sampler {
        match self {
            Gpu::Wgpu { device, .. } => Sampler::Wgpu(device.create_sampler(desc)),
            #[cfg(metal_backend)]
            Gpu::Metal(g) => Sampler::Metal(g.create_sampler(desc)),
        }
    }

    pub fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor,
    ) -> BindGroupLayout {
        match self {
            Gpu::Wgpu { device, .. } => {
                BindGroupLayout::Wgpu(device.create_bind_group_layout(desc))
            }
            #[cfg(metal_backend)]
            Gpu::Metal(_) => BindGroupLayout::Metal(m::MLayout(Arc::new(desc.entries.to_vec()))),
        }
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor) -> BindGroup {
        match self {
            Gpu::Wgpu { device, .. } => {
                let BindGroupLayout::Wgpu(layout) = desc.layout else {
                    mixed("Gpu::create_bind_group", "bind group layout", false)
                };
                let entries: Vec<wgpu::BindGroupEntry> = desc
                    .entries
                    .iter()
                    .map(|e| wgpu::BindGroupEntry {
                        binding: e.binding,
                        resource: match &e.resource {
                            BindingResource::Buffer(b) => {
                                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: b.buffer.wgpu(),
                                    offset: b.offset,
                                    size: b.size,
                                })
                            }
                            BindingResource::TextureView(v) => {
                                wgpu::BindingResource::TextureView(v.wgpu())
                            }
                            BindingResource::Sampler(s) => wgpu::BindingResource::Sampler(s.wgpu()),
                        },
                    })
                    .collect();
                BindGroup::Wgpu(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: desc.label,
                    layout,
                    entries: &entries,
                }))
            }
            #[cfg(metal_backend)]
            Gpu::Metal(_) => {
                let BindGroupLayout::Metal(layout) = desc.layout else {
                    mixed("Gpu::create_bind_group", "bind group layout", true)
                };
                let entries = desc
                    .entries
                    .iter()
                    .map(|e| {
                        let res = match &e.resource {
                            BindingResource::Buffer(b) => {
                                m::MRes::Buffer(b.buffer.metal().obj.clone(), b.offset)
                            }
                            BindingResource::TextureView(v) => {
                                m::MRes::Texture(v.metal().obj.clone())
                            }
                            BindingResource::Sampler(s) => m::MRes::Sampler(s.metal().0.clone()),
                        };
                        (e.binding, res)
                    })
                    .collect();
                BindGroup::Metal(m::MBindGroup {
                    layout: layout.clone(),
                    entries: Arc::new(entries),
                })
            }
        }
    }

    pub fn create_pipeline_layout(&self, desc: &PipelineLayoutDescriptor) -> PipelineLayout {
        match self {
            Gpu::Wgpu { device, .. } => {
                let layouts: Vec<Option<&wgpu::BindGroupLayout>> = desc
                    .bind_group_layouts
                    .iter()
                    .map(|l| l.map(|l| l.wgpu()))
                    .collect();
                PipelineLayout::Wgpu(device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: desc.label,
                        bind_group_layouts: &layouts,
                        immediate_size: desc.immediate_size,
                    },
                ))
            }
            #[cfg(metal_backend)]
            Gpu::Metal(_) => PipelineLayout::Metal(m::MPipelineLayout(Arc::new(
                desc.bind_group_layouts
                    .iter()
                    .map(|l| {
                        l.map_or_else(|| m::MLayout(Arc::new(Vec::new())), |l| l.metal().clone())
                    })
                    .collect(),
            ))),
        }
    }

    pub fn create_shader_module(&self, desc: ShaderModuleDescriptor) -> ShaderModule {
        let ShaderSource::Wgsl(source) = desc.source;
        match self {
            Gpu::Wgpu { device, .. } => {
                ShaderModule::Wgpu(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: desc.label,
                    source: wgpu::ShaderSource::Wgsl(source),
                }))
            }
            #[cfg(metal_backend)]
            Gpu::Metal(g) => ShaderModule::Metal(g.create_module(source.into_owned())),
        }
    }

    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor) -> RenderPipeline {
        match self {
            Gpu::Wgpu { device, .. } => RenderPipeline::Wgpu(device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: desc.label,
                    layout: desc.layout.map(|l| l.wgpu()),
                    vertex: wgpu::VertexState {
                        module: desc.vertex.module.wgpu(),
                        entry_point: desc.vertex.entry_point,
                        buffers: desc.vertex.buffers,
                        compilation_options: desc.vertex.compilation_options.clone(),
                    },
                    fragment: desc.fragment.as_ref().map(|f| wgpu::FragmentState {
                        module: f.module.wgpu(),
                        entry_point: f.entry_point,
                        targets: f.targets,
                        compilation_options: f.compilation_options.clone(),
                    }),
                    primitive: desc.primitive,
                    depth_stencil: desc.depth_stencil.clone(),
                    multisample: desc.multisample,
                    multiview_mask: desc.multiview_mask,
                    cache: None,
                },
            )),
            #[cfg(metal_backend)]
            Gpu::Metal(g) => {
                let layouts: Vec<m::MLayout> =
                    desc.layout.map_or_else(Vec::new, |l| l.metal().0.to_vec());
                let fragment = desc.fragment.as_ref().map(|f| {
                    (
                        f.module.metal(),
                        f.entry_point.unwrap_or("fs_main"),
                        f.targets,
                    )
                });
                match g.create_render_pipeline(
                    &layouts,
                    desc.vertex.module.metal(),
                    desc.vertex.entry_point.unwrap_or("vs_main"),
                    desc.vertex.buffers,
                    fragment,
                    &desc.primitive,
                    desc.depth_stencil.as_ref(),
                    desc.multisample.count,
                ) {
                    Ok(p) => RenderPipeline::Metal(p),
                    Err(e) => panic!(
                        "marks on Metal: {} did not build: {e}",
                        desc.label.unwrap_or("a pipeline")
                    ),
                }
            }
        }
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor) -> ComputePipeline {
        match self {
            Gpu::Wgpu { device, .. } => ComputePipeline::Wgpu(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: desc.label,
                    layout: desc.layout.map(|l| l.wgpu()),
                    module: desc.module.wgpu(),
                    entry_point: desc.entry_point,
                    compilation_options: desc.compilation_options.clone(),
                    cache: None,
                },
            )),
            #[cfg(metal_backend)]
            Gpu::Metal(g) => {
                let layouts: Vec<m::MLayout> =
                    desc.layout.map_or_else(Vec::new, |l| l.metal().0.to_vec());
                match g.create_compute_pipeline(
                    &layouts,
                    desc.module.metal(),
                    desc.entry_point.unwrap_or("main"),
                ) {
                    Ok(p) => ComputePipeline::Metal(p),
                    Err(e) => panic!(
                        "marks on Metal: {} did not build: {e}",
                        desc.label.unwrap_or("a compute pipeline")
                    ),
                }
            }
        }
    }

    pub fn create_command_encoder(&self, desc: &wgpu::CommandEncoderDescriptor) -> CommandEncoder {
        match self {
            Gpu::Wgpu { device, .. } => CommandEncoder::Wgpu(device.create_command_encoder(desc)),
            #[cfg(metal_backend)]
            Gpu::Metal(g) => CommandEncoder::Metal(g.create_encoder()),
        }
    }

    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        match self {
            Gpu::Wgpu { queue, .. } => wq(queue).write_buffer(buffer.wgpu(), offset, data),
            #[cfg(metal_backend)]
            Gpu::Metal(g) => g.write_buffer(buffer.metal(), offset, data),
        }
    }

    pub fn write_texture(
        &self,
        texture: TexelCopyTextureInfo,
        data: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        match self {
            Gpu::Wgpu { queue, .. } => wq(queue).write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: texture.texture.wgpu(),
                    mip_level: texture.mip_level,
                    origin: texture.origin,
                    aspect: texture.aspect,
                },
                data,
                layout,
                size,
            ),
            #[cfg(metal_backend)]
            Gpu::Metal(g) => {
                g.write_texture(texture.texture.metal(), texture.origin, data, layout, size)
            }
        }
    }

    pub fn submit<I: IntoIterator<Item = CommandBuffer>>(&self, buffers: I) {
        match self {
            Gpu::Wgpu { queue, .. } => {
                wq(queue).submit(buffers.into_iter().map(|b| match b {
                    CommandBuffer::Wgpu(b) => b,
                    #[cfg(metal_backend)]
                    CommandBuffer::Metal(_) => mixed("Gpu::submit", "command buffer", false),
                }));
            }
            #[cfg(metal_backend)]
            Gpu::Metal(g) => {
                for b in buffers {
                    match b {
                        CommandBuffer::Metal(e) => g.submit(e),
                        CommandBuffer::Wgpu(_) => mixed("Gpu::submit", "command buffer", true),
                    }
                }
                m::run_maps(false);
            }
        }
    }

    /// Wait for the GPU — wgpu's `poll(wait_indefinitely)`.
    pub fn poll_wait(&self) {
        match self {
            Gpu::Wgpu { device, .. } => {
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
            }
            #[cfg(metal_backend)]
            Gpu::Metal(_) => m::run_maps(true),
        }
    }

    /// Let finished work report back without waiting — wgpu's
    /// `poll(PollType::Poll)`: `map_async` callbacks whose copies are done run.
    pub fn poll(&self) {
        match self {
            Gpu::Wgpu { device, .. } => {
                let _ = device.poll(wgpu::PollType::Poll);
            }
            #[cfg(metal_backend)]
            Gpu::Metal(_) => m::run_maps(false),
        }
    }
}

/// An object of one backend handed to the other: a caller mixing what two
/// [`Gpu`]s made.
#[cfg_attr(not(metal_backend), allow(dead_code))]
#[cold]
#[track_caller]
fn mixed(call: &str, what: &str, on_metal: bool) -> ! {
    let (here, there) = if on_metal {
        ("Metal", "wgpu")
    } else {
        ("wgpu", "Metal")
    };
    panic!("{call}: a {there} {what} used on {here}; make every object a frame uses from the same marks::gpu::Gpu")
}

// ----------------------------------------------------------------- objects

macro_rules! unwrap_as {
    ($ty:ident, $wgpu:ty, $metal:ty) => {
        impl $ty {
            /// The wgpu object, or `None` for a Metal one.
            #[allow(dead_code)]
            pub fn as_wgpu(&self) -> Option<&$wgpu> {
                match self {
                    Self::Wgpu(v) => Some(v),
                    #[cfg(metal_backend)]
                    Self::Metal(_) => None,
                }
            }
            #[track_caller]
            pub fn wgpu(&self) -> &$wgpu {
                match self {
                    Self::Wgpu(v) => v,
                    #[cfg(metal_backend)]
                    Self::Metal(_) => panic!(concat!(
                        "a Metal ",
                        stringify!($ty),
                        " where wgpu's was needed"
                    )),
                }
            }
            /// The Metal object, or `None` for a wgpu one.
            #[cfg(metal_backend)]
            #[allow(dead_code)]
            pub(crate) fn as_metal(&self) -> Option<&$metal> {
                match self {
                    Self::Metal(v) => Some(v),
                    Self::Wgpu(_) => None,
                }
            }
            #[cfg(metal_backend)]
            #[track_caller]
            #[allow(dead_code)]
            pub(crate) fn metal(&self) -> &$metal {
                match self {
                    Self::Metal(v) => v,
                    Self::Wgpu(_) => panic!(concat!(
                        "a wgpu ",
                        stringify!($ty),
                        " where Metal's was needed"
                    )),
                }
            }
        }
    };
}

unwrap_as!(Buffer, wgpu::Buffer, m::MBuffer);
unwrap_as!(Texture, wgpu::Texture, m::MTexture);
unwrap_as!(TextureView, wgpu::TextureView, m::MTexture);
unwrap_as!(Sampler, wgpu::Sampler, m::MSampler);
unwrap_as!(BindGroupLayout, wgpu::BindGroupLayout, m::MLayout);
unwrap_as!(PipelineLayout, wgpu::PipelineLayout, m::MPipelineLayout);
unwrap_as!(ShaderModule, wgpu::ShaderModule, m::MModule);
unwrap_as!(BindGroup, wgpu::BindGroup, m::MBindGroup);
unwrap_as!(RenderPipeline, wgpu::RenderPipeline, m::MPipeline);

impl Buffer {
    pub fn size(&self) -> u64 {
        match self {
            Self::Wgpu(b) => b.size(),
            #[cfg(metal_backend)]
            Self::Metal(b) => b.size,
        }
    }

    pub fn slice<S: RangeBounds<u64>>(&self, bounds: S) -> BufferSlice<'_> {
        let start = match bounds.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&e) => e + 1,
            Bound::Excluded(&e) => e,
            Bound::Unbounded => self.size(),
        };
        BufferSlice {
            buffer: self,
            range: start..end,
        }
    }

    pub fn as_entire_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: self,
            offset: 0,
            size: None,
        })
    }

    pub fn unmap(&self) {
        if let Self::Wgpu(b) = self {
            b.unmap();
        }
    }
}

#[derive(Clone)]
pub struct BufferSlice<'a> {
    buffer: &'a Buffer,
    range: Range<u64>,
}

/// A mapped range, readable as bytes.
pub enum BufferView<'a> {
    Wgpu(wgpu::BufferView),
    Metal(&'a [u8]),
}

impl Deref for BufferView<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            BufferView::Wgpu(v) => v,
            BufferView::Metal(b) => b,
        }
    }
}

impl<'a> BufferSlice<'a> {
    fn wgpu(&self) -> wgpu::BufferSlice<'a> {
        self.buffer.wgpu().slice(self.range.clone())
    }

    /// wgpu's `map_async`. On either backend the callback runs from a later
    /// [`Gpu::poll`], [`Gpu::poll_wait`] or [`Gpu::submit`], once the work
    /// committed so far has finished — never inside this call.
    pub fn map_async(
        &self,
        mode: wgpu::MapMode,
        callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>) + wgpu::WasmNotSend + 'static,
    ) {
        match self.buffer {
            Buffer::Wgpu(_) => self.wgpu().map_async(mode, callback),
            #[cfg(metal_backend)]
            Buffer::Metal(_) => m::queue_map(Box::new(move || callback(Ok(())))),
        }
    }

    pub fn get_mapped_range(&self) -> Result<BufferView<'a>, MapRangeError> {
        match self.buffer {
            Buffer::Wgpu(_) => self
                .wgpu()
                .get_mapped_range()
                .map(BufferView::Wgpu)
                .map_err(|e| MapRangeError(e.to_string())),
            #[cfg(metal_backend)]
            Buffer::Metal(b) => {
                let all = m::mapped_bytes(b);
                let (start, end) = (self.range.start as usize, self.range.end as usize);
                all.get(start..end).map(BufferView::Metal).ok_or_else(|| {
                    MapRangeError(format!(
                        "range {start}..{end} of a {}-byte buffer",
                        all.len()
                    ))
                })
            }
        }
    }
}

/// Why [`BufferSlice::get_mapped_range`] could not read a range.
#[derive(Debug, Clone)]
pub struct MapRangeError(String);

impl std::fmt::Display for MapRangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mapped range: {}", self.0)
    }
}

impl std::error::Error for MapRangeError {}

impl Texture {
    pub fn create_view(&self, desc: &wgpu::TextureViewDescriptor) -> TextureView {
        match self {
            Self::Wgpu(t) => TextureView::Wgpu(t.create_view(desc)),
            #[cfg(metal_backend)]
            Self::Metal(t) => TextureView::Metal(t.view(desc)),
        }
    }

    pub fn size(&self) -> wgpu::Extent3d {
        match self {
            Self::Wgpu(t) => t.size(),
            #[cfg(metal_backend)]
            Self::Metal(t) => wgpu::Extent3d {
                width: t.width,
                height: t.height,
                depth_or_array_layers: t.depth.max(t.layers),
            },
        }
    }

    pub fn width(&self) -> u32 {
        self.size().width
    }

    pub fn mip_level_count(&self) -> u32 {
        match self {
            Self::Wgpu(t) => t.mip_level_count(),
            #[cfg(metal_backend)]
            Self::Metal(t) => t.mips,
        }
    }

    pub fn height(&self) -> u32 {
        self.size().height
    }
}

// ---------------------------------------------------------------- encoding

pub enum CommandEncoder {
    Wgpu(wgpu::CommandEncoder),
    #[cfg(metal_backend)]
    Metal(m::MEncoder),
}

pub enum CommandBuffer {
    Wgpu(wgpu::CommandBuffer),
    #[cfg(metal_backend)]
    Metal(m::MEncoder),
}

// wgpu's pass is the large variant; a pass lives on the stack for one
// encode, and boxing it would cost every wgpu frame an allocation.
#[allow(clippy::large_enum_variant)]
pub enum ComputePass<'e> {
    Wgpu(wgpu::ComputePass<'e>),
    #[cfg(metal_backend)]
    Metal(m::MComputePass, std::marker::PhantomData<&'e ()>),
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        match (self, pipeline) {
            (Self::Wgpu(p), ComputePipeline::Wgpu(pl)) => p.set_pipeline(pl),
            #[cfg(metal_backend)]
            (Self::Metal(p, _), ComputePipeline::Metal(pl)) => p.set_pipeline(pl),
            #[cfg(metal_backend)]
            (pass, _) => mixed(
                "ComputePass::set_pipeline",
                "pipeline",
                matches!(pass, Self::Metal(..)),
            ),
        }
    }

    pub fn set_bind_group(&mut self, index: u32, group: &BindGroup, offsets: &[u32]) {
        match (self, group) {
            (Self::Wgpu(p), BindGroup::Wgpu(g)) => p.set_bind_group(index, g, offsets),
            #[cfg(metal_backend)]
            (Self::Metal(p, _), BindGroup::Metal(g)) => p.set_bind_group(index, g, offsets),
            #[cfg(metal_backend)]
            (pass, _) => mixed(
                "ComputePass::set_bind_group",
                "bind group",
                matches!(pass, Self::Metal(..)),
            ),
        }
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        match self {
            Self::Wgpu(p) => p.dispatch_workgroups(x, y, z),
            #[cfg(metal_backend)]
            Self::Metal(p, _) => p.dispatch_workgroups(x, y, z),
        }
    }
}

// wgpu's pass is the large variant; a pass lives on the stack for one
// encode, and boxing it would cost every wgpu frame an allocation.
#[allow(clippy::large_enum_variant)]
pub enum RenderPass<'e> {
    Wgpu(wgpu::RenderPass<'e>),
    #[cfg(metal_backend)]
    Metal(m::MPass, std::marker::PhantomData<&'e ()>),
}

impl CommandEncoder {
    pub fn finish(self) -> CommandBuffer {
        match self {
            Self::Wgpu(e) => CommandBuffer::Wgpu(e.finish()),
            #[cfg(metal_backend)]
            Self::Metal(e) => CommandBuffer::Metal(e),
        }
    }

    pub fn begin_render_pass<'e>(&'e mut self, desc: &RenderPassDescriptor<'_>) -> RenderPass<'e> {
        match self {
            Self::Wgpu(e) => {
                let color: Vec<Option<wgpu::RenderPassColorAttachment>> = desc
                    .color_attachments
                    .iter()
                    .map(|c| {
                        c.as_ref().map(|c| wgpu::RenderPassColorAttachment {
                            view: c.view.wgpu(),
                            depth_slice: c.depth_slice,
                            resolve_target: c.resolve_target.map(|r| r.wgpu()),
                            ops: c.ops,
                        })
                    })
                    .collect();
                let pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: desc.label,
                    color_attachments: &color,
                    depth_stencil_attachment: desc.depth_stencil_attachment.as_ref().map(|d| {
                        wgpu::RenderPassDepthStencilAttachment {
                            view: d.view.wgpu(),
                            depth_ops: d.depth_ops,
                            stencil_ops: d.stencil_ops,
                        }
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: desc.multiview_mask,
                });
                RenderPass::Wgpu(pass.forget_lifetime())
            }
            #[cfg(metal_backend)]
            Self::Metal(e) => {
                let color: Vec<m::ColorAttachment> = desc
                    .color_attachments
                    .iter()
                    .flatten()
                    .map(|c| m::ColorAttachment {
                        texture: c.view.metal().obj.id(),
                        resolve: c.resolve_target.map(|r| r.metal().obj.id()),
                        load: c.ops.load,
                        store: c.ops.store,
                    })
                    .collect();
                let depth = desc.depth_stencil_attachment.as_ref().map(|d| {
                    let ops = d.depth_ops.unwrap_or(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    });
                    m::DepthAttachment {
                        texture: d.view.metal().obj.id(),
                        load: ops.load,
                        store: ops.store,
                    }
                });
                RenderPass::Metal(e.begin_render_pass(&color, depth), std::marker::PhantomData)
            }
        }
    }

    pub fn begin_compute_pass<'e>(
        &'e mut self,
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> ComputePass<'e> {
        match self {
            Self::Wgpu(e) => ComputePass::Wgpu(e.begin_compute_pass(desc).forget_lifetime()),
            #[cfg(metal_backend)]
            Self::Metal(e) => ComputePass::Metal(e.begin_compute_pass(), std::marker::PhantomData),
        }
    }

    pub fn copy_buffer_to_buffer(
        &mut self,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        size: u64,
    ) {
        match self {
            Self::Wgpu(e) => {
                e.copy_buffer_to_buffer(src.wgpu(), src_offset, dst.wgpu(), dst_offset, size)
            }
            #[cfg(metal_backend)]
            Self::Metal(e) => {
                e.copy_buffer_to_buffer(src.metal(), src_offset, dst.metal(), dst_offset, size)
            }
        }
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        source: TexelCopyTextureInfo,
        dest: TexelCopyBufferInfo,
        size: wgpu::Extent3d,
    ) {
        match self {
            Self::Wgpu(e) => e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: source.texture.wgpu(),
                    mip_level: source.mip_level,
                    origin: source.origin,
                    aspect: source.aspect,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: dest.buffer.wgpu(),
                    layout: dest.layout,
                },
                size,
            ),
            #[cfg(metal_backend)]
            Self::Metal(e) => {
                let row = dest.layout.bytes_per_row.unwrap_or(0) as u64;
                let rows = dest.layout.rows_per_image.unwrap_or(size.height) as u64;
                e.copy_texture_to_buffer(
                    source.texture.metal(),
                    source.origin,
                    dest.buffer.metal(),
                    dest.layout.offset,
                    row,
                    rows,
                    size,
                )
            }
        }
    }
}

impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        match (self, pipeline) {
            (Self::Wgpu(p), RenderPipeline::Wgpu(pl)) => p.set_pipeline(pl),
            #[cfg(metal_backend)]
            (Self::Metal(p, _), RenderPipeline::Metal(pl)) => p.set_pipeline(pl),
            #[cfg(metal_backend)]
            (pass, _) => mixed(
                "RenderPass::set_pipeline",
                "pipeline",
                matches!(pass, Self::Metal(..)),
            ),
        }
    }

    pub fn set_bind_group(&mut self, index: u32, group: &BindGroup, offsets: &[u32]) {
        match (self, group) {
            (Self::Wgpu(p), BindGroup::Wgpu(g)) => p.set_bind_group(index, g, offsets),
            #[cfg(metal_backend)]
            (Self::Metal(p, _), BindGroup::Metal(g)) => p.set_bind_group(index, g, offsets),
            #[cfg(metal_backend)]
            (pass, _) => mixed(
                "RenderPass::set_bind_group",
                "bind group",
                matches!(pass, Self::Metal(..)),
            ),
        }
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, slice: BufferSlice<'_>) {
        match self {
            Self::Wgpu(p) => p.set_vertex_buffer(slot, slice.wgpu()),
            #[cfg(metal_backend)]
            Self::Metal(p, _) => p.set_vertex_buffer(slot, slice.buffer.metal(), slice.range.start),
        }
    }

    pub fn set_index_buffer(&mut self, slice: BufferSlice<'_>, format: wgpu::IndexFormat) {
        match self {
            Self::Wgpu(p) => p.set_index_buffer(slice.wgpu(), format),
            #[cfg(metal_backend)]
            Self::Metal(p, _) => {
                p.set_index_buffer(slice.buffer.metal(), slice.range.start, format)
            }
        }
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        match self {
            Self::Wgpu(p) => p.set_viewport(x, y, w, h, min_depth, max_depth),
            #[cfg(metal_backend)]
            Self::Metal(p, _) => p.set_viewport(x, y, w, h, min_depth, max_depth),
        }
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        match self {
            Self::Wgpu(p) => p.set_scissor_rect(x, y, w, h),
            #[cfg(metal_backend)]
            Self::Metal(p, _) => p.set_scissor_rect(x, y, w, h),
        }
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        match self {
            Self::Wgpu(p) => p.draw(vertices, instances),
            #[cfg(metal_backend)]
            Self::Metal(p, _) => p.draw(vertices, instances),
        }
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        match self {
            Self::Wgpu(p) => p.draw_indexed(indices, base_vertex, instances),
            #[cfg(metal_backend)]
            Self::Metal(p, _) => p.draw_indexed(indices, base_vertex, instances),
        }
    }
}

#[cfg(metal_backend)]
impl super::MarkTarget<'static> {
    /// Marks over a Metal pass the scene was just drawn into: its colour
    /// (multisampled, and what that resolves to) and its depth, which the
    /// pass must have kept (`MetalRenderer::set_keep_attachments`).
    ///
    /// # Safety
    /// The pass's textures are live until the target is dropped.
    pub unsafe fn from_metal_pass(pass: &crate::metal::PassAttachments) -> Self {
        use crate::metal::enums::pixel_format as pf;
        let format = match pass.color_format {
            pf::BGRA8_UNORM_SRGB => wgpu::TextureFormat::Bgra8UnormSrgb,
            pf::BGRA8_UNORM => wgpu::TextureFormat::Bgra8Unorm,
            pf::RGBA8_UNORM => wgpu::TextureFormat::Rgba8Unorm,
            pf::RGBA16_FLOAT => wgpu::TextureFormat::Rgba16Float,
            _ => wgpu::TextureFormat::Rgba8UnormSrgb,
        };
        let (w, h) = (pass.width, pass.height);
        Self {
            color: TextureView::from_metal(pass.color, format, w, h),
            resolve: (!pass.resolve.is_null())
                .then(|| TextureView::from_metal(pass.resolve, format, w, h)),
            depth: (!pass.depth.is_null()).then(|| {
                TextureView::from_metal(pass.depth, wgpu::TextureFormat::Depth32Float, w, h)
            }),
            format,
            samples: pass.sample_count.max(1),
            rect: None,
            _borrow: std::marker::PhantomData,
        }
    }
}
