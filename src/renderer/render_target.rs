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
}

impl RenderTarget {
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
        let extra_formats = if srgb_variant != format {
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
        for v in [format.add_srgb_suffix(), format.remove_srgb_suffix()] {
            if v != format && !extra_formats.contains(&v) {
                extra_formats.push(v);
            }
        }
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers render target color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
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
        }
    }
}
