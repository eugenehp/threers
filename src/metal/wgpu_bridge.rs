//! wgpu on the native backend's own GPU: a `wgpu::Device` made from a
//! [`MetalDevice`]'s `MTLDevice` and its very `MTLCommandQueue`, and its
//! textures (a frame's drawable) as `wgpu::Texture`s.
//!
//! What it is for: what is written for wgpu — the GPU UI
//! ([`UiPass`](crate::renderer::UiPass)), views drawn on other threads
//! ([`ViewProducer`](crate::renderer::ViewProducer)) — on a window the
//! native backend draws. One queue, so what the scene submits and what
//! wgpu submits run in the order they were committed, with no fence
//! between them; one device, so a texture made by either is the other's.
//!
//! ```ignore
//! let (device, queue) = threers::metal::wgpu_on(&metal_device)?;
//! let frame = surface.next_frame().unwrap();
//! let target = unsafe { threers::metal::wgpu_texture(&device, frame.attachments().target_id(), wgpu::TextureUsages::RENDER_ATTACHMENT) };
//! ```

use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice, MTLPixelFormat, MTLTexture};

use super::{MetalDevice, MetalError};

/// The wgpu device and queue made for each `MTLDevice` and queue: one pair
/// each, kept for the process (a device made twice would not share
/// pipelines, and wgpu-core keeps devices it hands out alive anyway).
#[allow(clippy::type_complexity)]
static MADE: Mutex<Vec<(usize, usize, Arc<wgpu::Device>, Arc<wgpu::Queue>)>> = Mutex::new(Vec::new());

/// wgpu on `metal`'s device and command queue (made the first time, the
/// same pair after): what the scene and the UI's GPU pass submit runs in
/// order, on one GPU. Asks for what the renderer needs, as a window's wgpu
/// device does.
pub fn wgpu_on(metal: &MetalDevice) -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>), MetalError> {
    let (dev, q) = (metal.device_id() as usize, metal.queue_id() as usize);
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = made.iter().find(|m| m.0 == dev && m.1 == q) {
        return Ok((m.2.clone(), m.3.clone()));
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    // the adapter for this very MTLDevice
    let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::METAL))
        .into_iter()
        .find(|a| unsafe { a.as_hal::<wgpu::hal::api::Metal>() }.is_some_and(|h| {
            std::ptr::eq(h.raw_device() as *const _ as *const u8, dev as *const u8)
        }))
        .ok_or_else(|| MetalError::Unsupported("no wgpu adapter for this MTLDevice".into()))?;
    let mut limits = crate::gpu::renderer_limits(&adapter);
    limits.max_buffer_size = limits.max_buffer_size.max(adapter.limits().max_buffer_size);
    let features = adapter.features() & crate::gpu::TEXTURE_FEATURES;
    // SAFETY: the ids are the live device and queue `metal` holds; retained
    // here for as long as wgpu keeps them.
    let (raw_device, raw_queue) = unsafe {
        (
            Retained::<ProtocolObject<dyn MTLDevice>>::retain(dev as *mut _),
            Retained::<ProtocolObject<dyn MTLCommandQueue>>::retain(q as *mut _),
        )
    };
    let (raw_device, raw_queue) = raw_device
        .zip(raw_queue)
        .ok_or_else(|| MetalError::Unsupported("no device or queue".into()))?;
    let open = wgpu::hal::OpenDevice::<wgpu::hal::api::Metal> {
        device: unsafe { wgpu::hal::metal::Device::device_from_raw(raw_device, features, &limits) },
        queue: unsafe { wgpu::hal::metal::Queue::queue_from_raw(raw_queue, 1.0) },
    };
    // SAFETY: the hal device is this adapter's MTLDevice, asking for a
    // subset of its features.
    let (device, queue) = unsafe {
        adapter.create_device_from_hal(
            open,
            &wgpu::DeviceDescriptor {
                label: Some("threers on Metal"),
                required_features: features,
                required_limits: limits,
                ..Default::default()
            },
        )
    }
    .map_err(|e| MetalError::Unsupported(format!("wgpu on Metal: {e}")))?;
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    made.push((dev, q, device.clone(), queue.clone()));
    Ok((device, queue))
}

/// The wgpu format of a Metal pixel format the backend draws to.
fn format_of(f: MTLPixelFormat) -> Option<wgpu::TextureFormat> {
    Some(match f {
        MTLPixelFormat::BGRA8Unorm_sRGB => wgpu::TextureFormat::Bgra8UnormSrgb,
        MTLPixelFormat::BGRA8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        MTLPixelFormat::RGBA8Unorm_sRGB => wgpu::TextureFormat::Rgba8UnormSrgb,
        MTLPixelFormat::RGBA8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        MTLPixelFormat::RGBA16Float => wgpu::TextureFormat::Rgba16Float,
        MTLPixelFormat::BGR10A2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
        _ => return None,
    })
}

/// `texture` (an `MTLTexture` on `device`'s GPU: a frame's drawable, a
/// target) as a `wgpu::Texture`, for `usage` (a layer's drawable, which is
/// `framebufferOnly`, only `RENDER_ATTACHMENT`); `None` for a format wgpu
/// has no name for here.
///
/// # Safety
/// `texture` must be a live 2D `MTLTexture` made on the device `device` was
/// made from ([`wgpu_on`]), with every use `usage` names. It is retained for as
/// long as the returned texture lives.
pub unsafe fn wgpu_texture(device: &wgpu::Device, texture: super::objc::Id, usage: wgpu::TextureUsages) -> Option<wgpu::Texture> {
    let raw: Retained<ProtocolObject<dyn MTLTexture>> = unsafe { Retained::retain(texture as *mut _) }?;
    let format = format_of(raw.pixelFormat())?;
    let (w, h) = (raw.width() as u32, raw.height() as u32);
    let kind = raw.textureType();
    let hal = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(raw, format, kind, 1, 1, wgpu::hal::CopyExtent { width: w, height: h, depth: 1 }, None)
    };
    let desc = wgpu::TextureDescriptor {
        label: Some("threers Metal texture"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    };
    Some(unsafe { device.create_texture_from_hal::<wgpu::hal::api::Metal>(hal, &desc, wgpu::TextureUses::UNINITIALIZED) })
}
