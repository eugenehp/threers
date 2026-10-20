use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

/// 6-face cube render target. Holds a single cube color texture with one
/// 2D view per face (for rendering one face at a time) and one cube view
/// (for sampling as an env map).
pub struct CubeRenderTarget {
    pub side: u32,
    pub color_texture: wgpu::Texture,
    pub face_views: Vec<wgpu::TextureView>,
    pub cube_view: wgpu::TextureView,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub format: wgpu::TextureFormat,
}

/// Offscreen render target — a color texture plus matching depth. Pass to
/// `Renderer::render_to` to render a scene into it.
pub struct RenderTarget {
    pub width: u32,
    pub height: u32,
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    /// Whether this target stands in for the canvas — it *is* the picture,
    /// as a headless renderer's own target is — and so gets the renderer's
    /// tone mapping. three.js tone-maps only the final output: a
    /// `WebGLRenderTarget` (a composer's, a reflection's, a user's) stores
    /// untone-mapped light. Off by default.
    presents: AtomicBool,
    /// three.js `renderTarget.texture.colorSpace` is `SRGBColorSpace`: what is
    /// read back is sRGB-encoded. The default (`NoColorSpace`) reads back
    /// linear. Storage is the same either way — an 8-bit target holds encoded
    /// values and is sampled through an sRGB view, so a shader reads linear —
    /// which keeps one pipeline per format on every backend (WebGL2 cannot
    /// reinterpret a texture's format) and dark tones out of 8-bit banding.
    srgb_color_space: AtomicBool,
    /// three.js `renderTarget.samples`: above 1, `Renderer::render_to` draws
    /// into it multisampled and resolves colour and depth into it. 0 (the
    /// default) draws it single-sampled whatever the renderer's own MSAA, as
    /// three.js's `antialias` covers the canvas only.
    samples: AtomicU32,
    /// The device it was made on, for [`resized`](Self::resized).
    device: Arc<wgpu::Device>,
}

impl RenderTarget {
    /// A new target like this one — format, colour space, presenting — at
    /// `width × height`, its contents cleared: three.js's `setSize`.
    pub fn resized(&self, width: u32, height: u32) -> Self {
        let rt = Self::new(&self.device, width, height, self.format);
        rt.set_srgb_color_space(self.srgb_color_space());
        rt.set_presents(self.presents());
        rt.set_samples(self.samples());
        rt
    }

    /// three.js `renderTarget.samples` (see the field's docs). Any count
    /// above 1 draws at the renderer's one MSAA sample count, 4, in a
    /// target of the renderer's colour format; a target of another format (a
    /// half-float one) is drawn single-sampled.
    pub fn set_samples(&self, samples: u32) {
        self.samples.store(samples, Ordering::Relaxed);
    }

    pub fn samples(&self) -> u32 {
        self.samples.load(Ordering::Relaxed)
    }

    /// Mark this target as the picture itself (see the field's docs).
    pub fn set_presents(&self, on: bool) {
        self.presents.store(on, Ordering::Relaxed);
    }

    pub fn presents(&self) -> bool {
        self.presents.load(Ordering::Relaxed)
    }

    /// three.js `renderTarget.texture.colorSpace = SRGBColorSpace` (`true`)
    /// or the default `NoColorSpace` (`false`).
    pub fn set_srgb_color_space(&self, on: bool) {
        self.srgb_color_space.store(on, Ordering::Relaxed);
    }

    pub fn srgb_color_space(&self) -> bool {
        self.srgb_color_space.load(Ordering::Relaxed)
    }

    /// Turn bytes read back from this target into what three.js's
    /// `readRenderTargetPixels` returns for its colour space: an 8-bit target
    /// holds sRGB-encoded colour, decoded here unless the target is
    /// `SRGBColorSpace`. Alpha is untouched; float targets are left alone.
    pub fn bytes_as_color_space(&self, rgba: &mut [u8]) {
        let encoded = matches!(
            self.format,
            wgpu::TextureFormat::Rgba8Unorm
                | wgpu::TextureFormat::Rgba8UnormSrgb
                | wgpu::TextureFormat::Bgra8Unorm
                | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        if !encoded || self.srgb_color_space() {
            return;
        }
        let lut = srgb_to_linear_bytes();
        for px in rgba.chunks_exact_mut(4) {
            for c in &mut px[..3] {
                *c = lut[*c as usize];
            }
        }
    }
    /// Reusable view of the depth texture (recreated each call since wgpu
    /// `TextureView` is not `Clone`). Used by `Renderer::render_to` to swap
    /// its internal depth view without losing the surface depth.
    pub(crate) fn depth_view_clone(&self) -> wgpu::TextureView {
        self.depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Construct a cube render target — a 6-face cube texture sized side×side.
    /// Each face has its own 2D view for rendering; a single cube view is
    /// exposed for sampling as `texture_cube<f32>`.
    pub fn new_cube(
        device: &Arc<wgpu::Device>,
        side: u32,
        format: wgpu::TextureFormat,
    ) -> CubeRenderTarget {
        let size = wgpu::Extent3d {
            width: side.max(1),
            height: side.max(1),
            depth_or_array_layers: 6,
        };
        let srgb_variant = format.add_srgb_suffix();
        // As in `RenderTarget::new`: a backend without `VIEW_FORMATS` refuses
        // the texture outright if one is declared.
        let extra_formats =
            if srgb_variant != format && crate::renderer::caps::supports_view_formats(device) {
                vec![srgb_variant]
            } else {
                vec![]
            };
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers cube rt color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // COPY_SRC as well: a cube target is usually a reflection probe,
            // and a probe is only usable as image-based lighting once it has
            // been prefiltered per roughness — which means reading it back.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &extra_formats,
        });
        let mut face_views = Vec::with_capacity(6);
        for face in 0..6_u32 {
            let v = color_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers cube face view"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: face,
                array_layer_count: Some(1),
                ..Default::default()
            });
            face_views.push(v);
        }
        let cube_view = color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers cube combined view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            format: Some(srgb_variant),
            ..Default::default()
        });
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers cube rt depth"),
            size: wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        CubeRenderTarget {
            side,
            color_texture,
            face_views,
            cube_view,
            depth_texture,
            depth_view,
            format,
        }
    }

    pub fn new(
        device: &Arc<wgpu::Device>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };
        // Allow creating both linear and sRGB views of the color texture so
        // that downstream sampling can use the sRGB variant (which gives the
        // automatic sRGB→linear decode on read).
        // Both variants, in both directions. Adding only the sRGB one covered
        // the linear-target case and left the reverse impossible: a target that
        // IS sRGB could not be viewed as linear, so a pass wanting the stored
        // bytes untouched — the GPU alpha strip on the way to a video encoder —
        // was refused at view creation.
        let mut extra_formats = Vec::new();
        // Not every backend allows a texture to be viewed as another format:
        // WebGL2 does not, and declaring one there fails texture creation
        // outright. Where that is the case the target is created without them
        // and a reinterpreting view cannot be taken later.
        if crate::renderer::caps::supports_view_formats(device) {
            for v in [format.add_srgb_suffix(), format.remove_srgb_suffix()] {
                if v != format && !extra_formats.contains(&v) {
                    extra_formats.push(v);
                }
            }
        }
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers render target color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // COPY_DST so a target can also hold data a pass samples — a LUT,
            // a lookup table — uploaded rather than rendered.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &extra_formats,
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers render target depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            width,
            height,
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            format,
            presents: AtomicBool::new(false),
            srgb_color_space: AtomicBool::new(false),
            samples: AtomicU32::new(0),
            device: device.clone(),
        }
    }
}

/// Each sRGB-encoded byte as the 8-bit linear value it stands for.
fn srgb_to_linear_bytes() -> &'static [u8; 256] {
    static LUT: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|i| {
            let c = i as f32 / 255.0;
            let l = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
            (l * 255.0).round() as u8
        })
    })
}
