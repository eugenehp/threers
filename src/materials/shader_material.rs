//! User-defined custom-shader material — threers' extensibility escape hatch.
//!
//! A [`ShaderMaterial`] lets you plug arbitrary WGSL fragment code (with your own
//! uniform + storage data) into the renderer without editing the built-in shader
//! or the `Material` enum. It is compiled against threers' standard preamble, so
//! the fragment has the full built-in environment available:
//!
//! - the vertex output `VsOut` (`world_pos`, `world_normal`, `uv`, `vertex_color`,
//!   `view_z`, `clip_pos`, …),
//! - the `frame` uniform (camera, lights, ambient, fog, tone-mapping),
//! - the `mesh` uniform (model/normal matrices, color, params…),
//! - every built-in helper (`pbr_brdf`, `apply_fog`, `framebuffer_encode`, …),
//! - and a **user data group** at `@group(1)`:
//!
//! ```wgsl
//! struct ThreersUserData { data: array<vec4<f32>, 16>, };
//! @group(1) @binding(1) var<uniform> u_data: ThreersUserData;
//! @group(1) @binding(2) var<storage, read> u_s0: array<f32>;
//! @group(1) @binding(3) var<storage, read> u_s1: array<f32>;
//! @group(1) @binding(4) var<storage, read> u_s2: array<f32>;
//! // Four optional user textures plus a linear-repeat sampler. Unset slots
//! // read as opaque white.
//! @group(1) @binding(5) var u_tex0: texture_2d<f32>;
//! @group(1) @binding(6) var u_tex1: texture_2d<f32>;
//! @group(1) @binding(7) var u_tex2: texture_2d<f32>;
//! @group(1) @binding(8) var u_tex3: texture_2d<f32>;
//! @group(1) @binding(9) var u_samp: sampler;
//! // Two optional 3D textures (three.js `Data3DTexture` / `sampler3D`) and a
//! // linear clamp-to-edge sampler. Unset slots are a 1x1x1 black R8Unorm.
//! @group(1) @binding(13) var u_vol0: texture_3d<f32>;
//! @group(1) @binding(14) var u_vol1: texture_3d<f32>;
//! @group(1) @binding(15) var u_vol_samp: sampler;
//! ```
//!
//! You supply the fragment entry point:
//! ```wgsl
//! @fragment
//! fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
//!     let v = u_data.data[0].x;               // a scalar uniform
//!     let x = u_s0[0];                          // a storage element
//!     return vec4<f32>(in.vertex_color.rgb, 1.0);
//! }
//! ```

/// A custom-shader material. See the module docs for the WGSL contract.
#[derive(Debug, Clone)]
pub struct ShaderMaterial {
    /// WGSL defining `@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32>`.
    /// The standard preamble (structs, `vs_main`, helpers, user bindings) is
    /// prepended automatically.
    pub fragment: String,
    /// Optional WGSL defining `@vertex fn vs_custom(in: VsIn, ...) -> VsOut`.
    ///
    /// `None` uses the built-in `vs_main`, which is what almost everything
    /// wants. Supply one when the *position* has to be computed rather than
    /// read: a camera-facing ribbon, a strand expanded from control points, a
    /// billboard, anything that turns to face something.
    ///
    /// The whole preamble is in scope, including `standard_vertex` — the
    /// built-in transform, factored out of `vs_main` so it can be called — so
    /// the usual shape is to
    /// take the standard output and move it, rather than re-derive a transform
    /// that already handles displacement maps and the sky-depth case:
    ///
    /// ```wgsl
    /// @vertex
    /// fn vs_custom(in: VsIn, @builtin(vertex_index) vi: u32) -> VsOut {
    ///     var out = standard_vertex(in);
    ///     let side = f32(vi & 1u) * 2.0 - 1.0;          // which edge of the ribbon
    ///     let to_eye = normalize(frame.camera_position.xyz - out.world_pos);
    ///     out.world_pos += normalize(cross(out.world_normal, to_eye)) * side * s0(vi);
    ///     out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);
    ///     return out;
    /// }
    /// ```
    ///
    /// Morph targets apply as three.js's `#include <morphtarget_vertex>`
    /// does: the built-in `vs_main` morphs (it is `standard_vertex_morphed(in,
    /// vertex_index)`), and a custom stage calls the preamble's
    /// `morph_position(p, vi)`, `morph_normal(n, vi)` and `morph_color(c, vi)`
    /// — `vi` the `@builtin(vertex_index)` — or `standard_vertex_morphed`
    /// itself. (`standard_vertex` alone does not morph.)
    ///
    /// The user bindings are visible to the vertex stage as well as the
    /// fragment stage, so per-vertex data can come from `u_data`, `s0`/`s1`/`s2`
    /// or the user textures, indexed by `@builtin(vertex_index)` or
    /// `@builtin(instance_index)`.
    pub vertex: Option<String>,
    /// Up to 16 `vec4<f32>` uniform slots, exposed as `u_data.data[i]`.
    pub data: Vec<[f32; 4]>,
    /// Read-only `array<f32>` storage buffers, exposed as `u_s0`/`u_s1`/`u_s2`.
    pub storage0: Vec<f32>,
    pub storage1: Vec<f32>,
    pub storage2: Vec<f32>,
    /// Up to four user-owned texture views, exposed as `u_tex0`..`u_tex3` and
    /// sampled with `u_samp` (linear, repeat).
    ///
    /// GPU handles on purpose: the point is to sample something your own
    /// compute pass wrote this frame — a displacement cascade, an accumulated
    /// foam field, a wake map — without a CPU round trip. Made on the
    /// renderer's own GPU (for a `HeadlessRenderer` or `WindowRenderer`, its
    /// `marks_gpu`), they bind on wgpu and the native Metal backend alike, as
    /// [`volumes`](Self::volumes) do; a view from the other backend reads as
    /// an empty slot. Slots you leave empty read as opaque white.
    #[cfg(feature = "render")]
    pub textures: Vec<crate::marks::gpu::TextureView>,
    /// The same four slots from [`Texture`](crate::Texture)s the renderer
    /// uploads and caches itself, as it does a standard material's maps:
    /// `maps[i]` fills `u_tex{i}` wherever [`textures`](Self::textures) has
    /// no view at `i`. Unlike a wgpu view these bind on every backend,
    /// native Metal included. A texture's colour space is its format's —
    /// `Rgba8UnormSrgb` decodes on sample, `Rgba8Unorm` does not — and the
    /// mips it carries (or generates, with a `mipmap_filter`) are what
    /// `textureSampleLevel` reads; `u_samp` stays the one linear-repeat
    /// sampler either way.
    pub maps: Vec<std::sync::Arc<crate::textures::Texture>>,
    /// Up to two 3D texture views — a [`GpuVolume`](crate::volume::GpuVolume)'s
    /// `view`, made on the renderer's own GPU (for a `HeadlessRenderer`,
    /// [`marks_gpu`](crate::HeadlessRenderer::marks_gpu)), so on wgpu and the
    /// native Metal backend alike — exposed as `u_vol0`/`u_vol1` and sampled
    /// with `u_vol_samp` (linear, clamp-to-edge), or read exactly with
    /// `textureLoad`. three.js's
    /// `Data3DTexture` in a `ShaderMaterial`: a slice plane through a scan,
    /// a volume ray-march, a 3D colour lookup. Filtering needs a filterable
    /// format (`R8Unorm`, `Rgba8Unorm`, `R16Float`); the others are for
    /// `textureLoad` only.
    ///
    /// Not on WebGL2: its sixteen texture units are all taken by the rest of
    /// the layout, so the slots are left out there and a shader that reads
    /// `u_vol0` does not compile. See
    /// [`supports_shader_material_volumes`](crate::renderer::caps::supports_shader_material_volumes).
    #[cfg(feature = "render")]
    pub volumes: Vec<crate::marks::gpu::TextureView>,
    /// Draw this material in the screen-space pass, with the opaque scene
    /// available as `ss_color_tex` / `ss_depth_tex` at `@group(3)`.
    ///
    /// That is what makes screen-space reflection and refraction possible from a
    /// custom shader: without it those bindings are 1x1 placeholders. Implies
    /// alpha blending against the already-drawn scene, and the surface still
    /// writes depth so later geometry sorts against it.
    ///
    /// `ss_depth_tex` is a `texture_2d<f32>` read with `textureLoad`, the
    /// depth buffer as it is in `.r`: `1 - d` when the renderer uses reversed
    /// depth (`frame.camera_position.w > 0.5`). `ss_color_tex` is sampled
    /// through `ss_color_samp`.
    pub screen_space: bool,
    pub opacity: f32,
    /// Alpha-blend against what is already there, and do not write depth.
    ///
    /// Transparent custom draws are sorted back-to-front *against each other
    /// and against every other transparent object*, so overlapping surfaces
    /// composite correctly. What is not sorted is the triangles **inside** one
    /// mesh: a mesh that folds over itself — a ribbon that curls, a shell seen
    /// from both sides — blends its own triangles in buffer order.
    ///
    /// For something thin and faint (hair, fuzz, a thread) that is invisible
    /// and is the right trade. For something denser, either split it into
    /// meshes that can be sorted, or set
    /// `render_state.depth_write` and accept hard edges instead of
    /// wrong ones. The renderer's weighted-blended order-independent path is
    /// not reachable from here; it needs a second fragment entry point writing
    /// two targets, which this contract has no way to express.
    pub transparent: bool,
    /// `0` = FrontSide, `1` = BackSide, `2` = DoubleSide.
    pub side: u32,
    /// three.js `toneMapped` (default `true`): whether the renderer's tone
    /// mapping applies to this material's output.
    pub tone_mapped: bool,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    ///
    /// `depth_write: None` follows [`transparent`](Self::transparent): opaque
    /// writes, blended does not. `Some(true)` on a transparent material makes
    /// it self-occlude — its own nearer triangles reject its farther ones —
    /// which is usually what a dense translucent shell wants and never what
    /// hair wants.
    pub render_state: super::RenderState,
    /// What the native Metal renderer draws when this material's shader
    /// cannot run there: the mesh, unlit, in this colour. White by default.
    ///
    /// Metal translates the WGSL — fragment, custom vertex stage, and the
    /// preamble functions it calls — to MSL itself, its textures, volumes and
    /// the screen-space capture included; what it cannot run is a shader that
    /// writes a sample mask, or whose WGSL it cannot translate. It says why on
    /// stderr, once, and draws this.
    pub fallback_color: crate::math::Color,
    /// The fragment for the native Metal backend: MSL appended to the
    /// built-in library, defining `fragment float4 fs_custom(...)` (frame
    /// and draw at buffers 0 and 1, [`data`](Self::data) at 2, the storage
    /// arrays at 3–5 and their lengths at 6). [`from_shared`](Self::from_shared)
    /// fills it with the translation of its WGSL, so it is checked up front.
    /// `None`, the usual case: Metal translates the WGSL when it first draws
    /// it (see [`fallback_color`](Self::fallback_color)).
    pub msl: Option<String>,
    /// three.js `alphaToCoverage`: under MSAA, the fragment's alpha is the
    /// share of the pixel's samples it covers, and what is stored is opaque —
    /// so a cut-out edge comes out smooth. Return
    /// `vec4(colour, coverage_alpha(coverage))`: the preamble's
    /// `coverage_alpha` gives the coverage under MSAA, and in a frame drawn
    /// single-sampled cuts at half (discarding below). Opaque materials only.
    ///
    /// A shader can do better than one alpha for the pixel: test each sample
    /// itself and say which it covers. The preamble has what that needs —
    /// `THREERS_SAMPLES` (1, or 4 under MSAA), `threers_sample_offset(i)`
    /// (where sample `i` lies from the pixel's centre, in pixels: step the
    /// shader's coordinates there by their `dpdx`/`dpdy`), and
    /// `frag_out(colour, mask)` to return, declaring
    /// `fn fs_main(..) -> ThreersFragOut`. The mask is written as
    /// `@builtin(sample_mask)` where the backend has it
    /// (`THREERS_SAMPLE_MASK`); on WebGL2 it is not, and the shader returns
    /// the share of samples as its alpha for alpha-to-coverage to place. Fur
    /// and hair draw their strands this way.
    pub alpha_to_coverage: bool,
}

impl Default for ShaderMaterial {
    fn default() -> Self {
        Self {
            fragment: String::new(),
            vertex: None,
            data: Vec::new(),
            storage0: Vec::new(),
            storage1: Vec::new(),
            storage2: Vec::new(),
            #[cfg(feature = "render")]
            textures: Vec::new(),
            maps: Vec::new(),
            #[cfg(feature = "render")]
            volumes: Vec::new(),
            screen_space: false,
            opacity: 1.0,
            transparent: false,
            side: 0,
            tone_mapped: true,
            render_state: super::RenderState::DEFAULT,
            fallback_color: crate::math::Color::WHITE,
            msl: None,
            alpha_to_coverage: false,
        }
    }
}

impl ShaderMaterial {
    /// Create a material whose fragment stage is the given WGSL source
    /// (must define `@fragment fn fs_main`).
    pub fn new(fragment: impl Into<String>) -> Self {
        Self {
            fragment: fragment.into(),
            ..Default::default()
        }
    }

    /// A material from a shader written once for both backends — see
    /// [`shader!`](crate::shader). Its WGSL becomes
    /// [`fragment`](Self::fragment) and its MSL [`msl`](Self::msl).
    ///
    /// # Panics
    /// When the shared source does not translate (no `shade` entry point,
    /// unbalanced brackets, …); [`try_from_shared`](Self::try_from_shared)
    /// returns the message instead.
    pub fn from_shared(shader: &super::SharedShader) -> Self {
        Self::try_from_shared(shader).unwrap_or_else(|e| panic!("{e}"))
    }

    /// As [`from_shared`](Self::from_shared), returning a translation error.
    pub fn try_from_shared(shader: &super::SharedShader) -> Result<Self, String> {
        // The Metal fragment only matters to the Metal backend, which needs
        // `render`; without it there is nothing to translate for.
        #[cfg(feature = "render")]
        let msl = Some(shader.msl()?);
        #[cfg(not(feature = "render"))]
        let msl = None;
        Ok(Self {
            fragment: shader.wgsl()?,
            msl,
            ..Default::default()
        })
    }

    /// Give the Metal backend a fragment of its own. See [`msl`](Self::msl).
    pub fn with_msl(mut self, msl: impl Into<String>) -> Self {
        self.msl = Some(msl.into());
        self
    }

    /// Replace the vertex stage too. See [`vertex`](Self::vertex).
    pub fn with_vertex(mut self, vertex: impl Into<String>) -> Self {
        self.vertex = Some(vertex.into());
        self
    }

    /// Override depth writing. See [`render_state`](Self::render_state).
    pub fn with_depth_write(mut self, write: bool) -> Self {
        self.render_state.depth_write = Some(write);
        self
    }

    /// Whether [`validate`](Self::validate) can actually check anything.
    ///
    /// `false` on `wasm32`: naga is not in wgpu's browser build, so there is no
    /// WGSL front end to check against and the browser's own compiler does the
    /// job instead, reporting to the console. `validate` returns `Ok` there
    /// rather than pretending — check this if the distinction matters.
    pub const VALIDATION_AVAILABLE: bool = !cfg!(target_arch = "wasm32");

    /// Compile-check this material without a GPU, or even a renderer.
    ///
    /// The shader is assembled exactly as the renderer assembles it, then
    /// parsed and validated. That makes a broken `ShaderMaterial` a unit test
    /// rather than a runtime failure in a draw loop — which matters more here
    /// than for most shaders, because the module the error refers to is mostly
    /// built-in preamble the caller never sees.
    ///
    /// `storage_buffers` says which backend to check against: `true` for
    /// WebGPU, Metal, Vulkan and the rest, `false` for WebGL2, where there are
    /// none and the storage arrays are read from a texture instead. Check both
    /// if the material has to run in a browser.
    ///
    /// ```no_run
    /// # use threers::ShaderMaterial;
    /// let m = ShaderMaterial::new(
    ///     "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
    ///      return vec4<f32>(1.0); }",
    /// );
    /// assert!(m.validate(true).is_ok());
    /// ```
    /// The module the renderer builds from this material, single-sampled —
    /// the text [`validate`](Self::validate) checks, with the preamble, the
    /// bindings and the user's code in one string. `storage_buffers` as for
    /// `validate`: `false` is the WebGL2 shape.
    #[cfg(feature = "render")]
    pub fn module_source(&self, storage_buffers: bool) -> String {
        crate::renderer::custom_shader_module_source(
            &self.fragment,
            self.vertex.as_deref(),
            storage_buffers,
            false,
            1,
        )
    }

    #[cfg(feature = "render")]
    pub fn validate(&self, storage_buffers: bool) -> Result<(), String> {
        // As drawn single-sampled, and multisampled — where the module gains
        // `@builtin(sample_mask)` on a backend with storage buffers.
        for (coverage, samples) in [(false, 1), (self.alpha_to_coverage, 4)] {
            let source = crate::renderer::custom_shader_module_source(
                &self.fragment,
                self.vertex.as_deref(),
                storage_buffers,
                coverage,
                samples,
            );
            crate::renderer::validate_custom_shader(
                &source,
                crate::renderer::custom_preamble_lines(),
            )?;
        }
        Ok(())
    }

    /// Set the `@group(4)` uniform `vec4` slots (`u_data.data`).
    pub fn with_data(mut self, data: Vec<[f32; 4]>) -> Self {
        self.data = data;
        self
    }
    /// Set storage buffer `u_s0`.
    pub fn with_storage0(mut self, s: Vec<f32>) -> Self {
        self.storage0 = s;
        self
    }
    /// Set storage buffer `u_s1`.
    pub fn with_storage1(mut self, s: Vec<f32>) -> Self {
        self.storage1 = s;
        self
    }
    /// Set storage buffer `u_s2`.
    pub fn with_storage2(mut self, s: Vec<f32>) -> Self {
        self.storage2 = s;
        self
    }
    /// Attach user textures (`u_tex0`..`u_tex3`). At most four are used.
    #[cfg(feature = "render")]
    pub fn with_textures<T: Into<crate::marks::gpu::TextureView>>(
        mut self,
        t: impl IntoIterator<Item = T>,
    ) -> Self {
        self.textures = t.into_iter().map(Into::into).collect();
        self
    }

    /// Attach CPU-side textures for `u_tex0`..`u_tex3` — see
    /// [`maps`](Self::maps). At most four are used.
    pub fn with_maps(mut self, maps: Vec<std::sync::Arc<crate::textures::Texture>>) -> Self {
        self.maps = maps;
        self
    }
    /// Draw in the screen-space pass so `ss_color_tex` / `ss_depth_tex` carry the
    /// opaque scene. See [`Self::screen_space`].
    pub fn with_screen_space(mut self, on: bool) -> Self {
        self.screen_space = on;
        self
    }
    /// Material opacity (`0.0..=1.0`).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }
    /// Whether the material participates in alpha blending.
    pub fn with_transparent(mut self, transparent: bool) -> Self {
        self.transparent = transparent;
        self
    }
    /// Face culling: `0` front, `1` back, `2` double-sided.
    pub fn with_side(mut self, side: u32) -> Self {
        self.side = side;
        self
    }

    /// The 16 uniform slots as a flat `[[f32; 4]; 16]`, zero-padded/truncated.
    pub fn data_slots(&self) -> [[f32; 4]; 16] {
        let mut out = [[0.0f32; 4]; 16];
        for (i, v) in self.data.iter().take(16).enumerate() {
            out[i] = *v;
        }
        out
    }
}
