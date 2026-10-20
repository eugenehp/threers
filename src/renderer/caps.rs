//! What the device can actually do, for the passes that need more than WebGL2.
//!
//! WebGL2 has no compute stage and no storage buffers. wgpu does not refuse such
//! a device — it reports the shortfall by *zeroing* the corresponding limits, so
//! a pass that assumes compute is available gets a pipeline that was never
//! created and a frame that silently misses a step. On a canvas that looks like
//! a black or unchanged image at a full frame rate, which is far harder to place
//! than an error would be.
//!
//! So the passes that need compute ask here first and return `None` when the
//! answer is no, which their callers already handle as "this effect is
//! unavailable" rather than as a failure.
//!
//! The thresholds are read off `wgpu::Limits::downlevel_webgl2_defaults()`
//! rather than written down, and the tests compare against that same constant —
//! so if wgpu's idea of WebGL2 changes, the tests say so instead of the browser.

/// Whether shaders on a device with these limits can run compute passes.
pub fn limits_support_compute(limits: &wgpu::Limits) -> bool {
    limits.max_compute_workgroup_size_x > 0
        && limits.max_compute_workgroup_size_y > 0
        && limits.max_compute_invocations_per_workgroup > 0
        && limits.max_compute_workgroups_per_dimension > 0
}

/// Whether shaders on a device with these limits can bind storage buffers.
///
/// Skinning puts its bone matrices in one, and a [`ShaderMaterial`] may bind up
/// to three of its own.
///
/// [`ShaderMaterial`]: crate::materials::ShaderMaterial
pub fn limits_support_storage_buffers(limits: &wgpu::Limits) -> bool {
    limits.max_storage_buffers_per_shader_stage > 0
}

/// Whether a texture may declare alternative view formats.
///
/// wgpu reports this as `DownlevelFlags::VIEW_FORMATS`, an adapter property, so
/// the same inference applies — WebGL2 is the only backend here without a
/// compute stage. Render targets
/// normally declare both the sRGB and the linear spelling of their format so a
/// later pass can choose how the bytes are interpreted on read; where that is
/// unavailable the texture has to be created without them, and such a
/// reinterpreting view simply cannot be made.
pub fn supports_view_formats(device: &wgpu::Device) -> bool {
    supports_compute(device)
}

/// Whether each colour attachment may carry its own blend state.
///
/// wgpu reports this as `DownlevelFlags::INDEPENDENT_BLEND`, which is an
/// *adapter* property and so not reachable from a device, so it is inferred from
/// the absence of a compute stage: WebGL2 is the only backend this crate targets
/// without one. WebGL2 blends every attachment
/// alike, which the order-independent transparency pass cannot work with: it
/// accumulates weighted colour and coverage into two attachments that need
/// different blend equations.
pub fn supports_independent_blend(device: &wgpu::Device) -> bool {
    supports_compute(device)
}

/// Whether a shader can read a multisampled texture.
///
/// WebGL2 is OpenGL ES 3.0, which has no `sampler2DMS`; every other backend
/// this crate targets has one. Inferred, as
/// [`supports_independent_blend`] is, from the absence of compute there. The
/// screen-space refraction pass needs it to multisample: the glass marches a
/// single-sampled depth, resolved from the multisampled one.
pub fn supports_multisampled_textures(device: &wgpu::Device) -> bool {
    supports_compute(device)
}

/// Whether this device can run compute passes.
pub fn supports_compute(device: &wgpu::Device) -> bool {
    limits_support_compute(&device.limits())
}

/// Whether this device can bind storage buffers in a shader.
pub fn supports_storage_buffers(device: &wgpu::Device) -> bool {
    limits_support_storage_buffers(&device.limits())
}

/// Whether a `ShaderMaterial`'s two volume slots fit beside everything else
/// it binds.
///
/// WebGPU, Metal and Vulkan count sampled textures per shader stage, and the
/// custom layout's fourteen are inside the floor of sixteen. wgpu's GL backend
/// instead gives every texture entry in the whole pipeline layout a unit of
/// its own, and WebGL2 has sixteen: the custom layout comes to eighteen with
/// the volumes and exactly sixteen without, and going over panics inside
/// wgpu-hal rather than failing validation. WebGL2 is also the only backend
/// here without storage buffers, which is the test used — the same one that
/// leaves the storage slots out.
pub fn supports_shader_material_volumes(device: &wgpu::Device) -> bool {
    supports_storage_buffers(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every inference here matches what the adapter actually reports.
    ///
    /// Three of these functions answer from the absence of a compute stage
    /// because the property they describe is an *adapter* downlevel flag and so
    /// not reachable from a `Device`. The reasoning is that WebGL2 is the only
    /// backend this crate targets that lacks compute, and also the only one that
    /// lacks the rest — sound over that set rather than in general.
    ///
    /// This holds the reasoning to the authority on whatever device the suite
    /// runs on: the adapter's own `DownlevelCapabilities`. If a backend ever
    /// arrives that has compute but not independent blend, or view formats but
    /// not compute, this fails here rather than somewhere in a frame.
    #[test]
    fn the_inferences_match_the_adapter() {
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(d);
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            }))
        else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let flags = adapter.get_downlevel_capabilities().flags;
        let Ok((device, _queue)) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("caps test device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            },
        )) else {
            eprintln!("skipping: no device");
            return;
        };

        for (name, inferred, reported) in [
            (
                "compute",
                supports_compute(&device),
                flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS),
            ),
            (
                "independent blend",
                supports_independent_blend(&device),
                flags.contains(wgpu::DownlevelFlags::INDEPENDENT_BLEND),
            ),
            (
                "view formats",
                supports_view_formats(&device),
                flags.contains(wgpu::DownlevelFlags::VIEW_FORMATS),
            ),
        ] {
            assert_eq!(
                inferred, reported,
                "{name}: inferred {inferred} from the limits, but the adapter \
                 reports {reported} — the inference in this module no longer \
                 holds on this backend ({:?})",
                adapter.get_info().backend
            );
        }
    }

    /// A WebGL2 device reports no compute and no storage buffers.
    ///
    /// This is the case the guards exist for, and it is checked against wgpu's
    /// own `downlevel_webgl2_defaults` rather than against numbers copied out of
    /// it — the point is to track what wgpu considers WebGL2, not to restate it.
    #[test]
    fn webgl2_limits_offer_neither_compute_nor_storage() {
        let webgl2 = wgpu::Limits::downlevel_webgl2_defaults();
        assert!(
            !limits_support_compute(&webgl2),
            "WebGL2 has no compute stage, but the limits say it does"
        );
        assert!(
            !limits_support_storage_buffers(&webgl2),
            "WebGL2 has no storage buffers, but the limits say it does"
        );
    }

    /// A full WebGPU device offers both.
    ///
    /// Without this the guards could be satisfied by a function that always says
    /// no, which would turn bloom and ambient occlusion off everywhere.
    #[test]
    fn webgpu_defaults_offer_both() {
        let full = wgpu::Limits::default();
        assert!(limits_support_compute(&full));
        assert!(limits_support_storage_buffers(&full));
    }

    /// Neither shader module declares a storage buffer.
    ///
    /// This is the invariant that lets the GL backend compile these modules at
    /// all, and it is stronger than "nothing skinned is drawn": a single
    /// `var<storage>` declaration stops the whole module compiling on a GLSL
    /// backend whether or not any entry point reads it, and whether or not the
    /// layout offers it. Skinning and the light arrays were both moved to
    /// fixed-size uniform blocks for exactly this reason, and a future
    /// convenience that quietly reintroduces one would take WebGL2 out without
    /// touching anything named after it.
    ///
    /// `ShaderMaterial`'s three storage bindings are deliberately not here —
    /// they are appended to a *copy* of the source by
    /// `custom_shader_module_source`, and only where the device has them.
    #[test]
    fn the_shader_modules_declare_no_storage_buffers() {
        for (name, source) in [
            ("SHADER_SOURCE", crate::renderer::shader::legacy_shader_source()),
            ("POSTFX_SHADER", crate::renderer::shader::POSTFX_SHADER),
        ] {
            assert!(
                !source.contains("var<storage"),
                "{name} declares a storage buffer, which a GLSL backend cannot \
                 compile — see the note on this test"
            );
        }
    }

    /// The non-WebGL downlevel floor still has compute.
    ///
    /// `downlevel_defaults` is the portability floor the headless renderer asks
    /// for on native, and it is *not* the WebGL2 one. Conflating the two would
    /// disable compute on every native downlevel device for no reason.
    #[test]
    fn the_native_downlevel_floor_keeps_compute() {
        let downlevel = wgpu::Limits::downlevel_defaults();
        assert!(
            limits_support_compute(&downlevel),
            "downlevel_defaults is a native floor and does have compute"
        );
        assert!(limits_support_storage_buffers(&downlevel));
    }
}
