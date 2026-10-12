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
    /// These are raw wgpu handles on purpose: the point is to sample something
    /// your own compute pass wrote this frame — a displacement cascade, an
    /// accumulated foam field, a wake map — without a CPU round trip. Slots you
    /// leave empty read as opaque white.
    pub textures: Vec<std::sync::Arc<wgpu::TextureView>>,
    /// Draw this material in the screen-space pass, with the opaque scene
    /// available as `ss_color_tex` / `ss_depth_tex` at `@group(3)`.
    ///
    /// That is what makes screen-space reflection and refraction possible from a
    /// custom shader: without it those bindings are 1x1 placeholders. Implies
    /// alpha blending against the already-drawn scene, and the surface still
    /// writes depth so later geometry sorts against it.
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
    /// [`depth_write`](Self::depth_write) and accept hard edges instead of
    /// wrong ones. The renderer's weighted-blended order-independent path is
    /// not reachable from here; it needs a second fragment entry point writing
    /// two targets, which this contract has no way to express.
    pub transparent: bool,
    /// Override whether this material writes depth.
    ///
    /// `None` follows [`transparent`](Self::transparent): opaque writes, blended
    /// does not. `Some(true)` on a transparent material makes it self-occlude —
    /// its own nearer triangles reject its farther ones — which is usually what
    /// a dense translucent shell wants and never what hair wants.
    pub depth_write: Option<bool>,
    /// `0` = FrontSide, `1` = BackSide, `2` = DoubleSide.
    pub side: u32,
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
            textures: Vec::new(),
            screen_space: false,
            opacity: 1.0,
            transparent: false,
            depth_write: None,
            side: 0,
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

    /// Replace the vertex stage too. See [`vertex`](Self::vertex).
    pub fn with_vertex(mut self, vertex: impl Into<String>) -> Self {
        self.vertex = Some(vertex.into());
        self
    }

    /// Override depth writing. See [`depth_write`](Self::depth_write).
    pub fn with_depth_write(mut self, write: bool) -> Self {
        self.depth_write = Some(write);
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
    pub fn validate(&self, storage_buffers: bool) -> Result<(), String> {
        let source = crate::renderer::custom_shader_module_source(
            &self.fragment,
            self.vertex.as_deref(),
            storage_buffers,
        );
        crate::renderer::validate_custom_shader(
            &source,
            crate::renderer::custom_preamble_lines(),
        )
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
    pub fn with_textures(mut self, t: Vec<std::sync::Arc<wgpu::TextureView>>) -> Self {
        self.textures = t;
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
