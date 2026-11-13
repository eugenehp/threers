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
//!
//! # Changing it every frame
//!
//! The renderer keeps each material's uniform buffer, storage buffers and bind
//! groups from one frame to the next, so a material that is drawn unchanged
//! costs nothing to re-submit, and one whose uniform changed costs a 256-byte
//! write. Two ways to change one:
//!
//! - Through the fields, given `&mut` access (`Arc::make_mut` on the mesh's
//!   material). Uniform changes are seen by comparing the block; a storage field
//!   is seen when it is *replaced* — a new `Vec` is a new allocation. Editing
//!   one in place keeps the allocation, so follow that with
//!   [`mark_storage_dirty`](ShaderMaterial::mark_storage_dirty). Note that
//!   `make_mut` on an `Arc` other meshes also hold *copies* the material,
//!   storage and all, and the copy is uploaded afresh — use the setters there.
//! - Through the `&self` setters — [`set_uniform`](ShaderMaterial::set_uniform),
//!   [`set_data`](ShaderMaterial::set_data),
//!   [`set_storage`](ShaderMaterial::set_storage),
//!   [`set_storage_range`](ShaderMaterial::set_storage_range). These write to
//!   state that every clone of the material shares, so one call reaches every
//!   mesh it was handed to, however many `Arc`s that is, and the next frame
//!   draws with it. A range update uploads that range and nothing else. Once a
//!   value has been set this way it takes precedence over the field.
//!
//! Share one material between many meshes by sharing its `Arc<Material>`: the
//! storage is then uploaded once, and only the per-mesh transform differs from
//! draw to draw. Clones that were handed live storage share that upload too.

use std::sync::{Arc, Mutex, MutexGuard};

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
    /// The same two slots from [`Data3DTexture`](crate::volume::Data3DTexture)s
    /// the renderer uploads and caches itself, as [`maps`](Self::maps) are
    /// to `volumes`: `volume_maps[i]` fills `u_vol{i}` wherever `volumes`
    /// has no view at `i`, on wgpu and the native Metal backend alike, with
    /// no GPU handle of the app's. A new `Arc` is new contents (a 4D series'
    /// next frame): the same one is uploaded once.
    #[cfg(feature = "render")]
    pub volume_maps: Vec<std::sync::Arc<crate::volume::Data3DTexture>>,
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
    /// Cast shadows through this material's own vertex stage (`vs_custom`)
    /// instead of its mesh's triangles, for a stage that moves, grows or hides
    /// geometry: then the shadow is of what is drawn. Off by default, which is
    /// three.js's depth-material behaviour and what the Metal backend does.
    ///
    /// Depth only: the fragment stage does not run, so a cut-out (fur, a
    /// discard) casts solid. And the stage sees the light as its camera —
    /// `frame.view_proj` is the light's, `frame.camera_position` its position
    /// (none for a directional light) — so camera-facing geometry such as
    /// billboards or ribbons turns toward the light and should leave this off.
    /// Directional and spot lights; a point light casts the mesh. wgpu backend.
    pub shadow_vertex: bool,
    /// Identity for the renderer's GPU cache, and the uniform and storage
    /// values set through the `&self` setters, which clones share.
    ///
    /// Leave it at its default; it is public only so that struct-literal
    /// construction with `..Default::default()` keeps working.
    pub live: ShaderLive,
}

/// The shared, in-place-updatable half of a [`ShaderMaterial`].
///
/// Cloning gives the copy its own identity — so a copy whose *fields* are then
/// changed gets its own GPU resources — but the same live values, so a
/// [`set_uniform`](ShaderMaterial::set_uniform) on one is seen by all of them.
pub struct ShaderLive {
    /// This value's identity. Fresh for every clone.
    id: u64,
    /// Bumped by `mark_storage_dirty`, per slot.
    field_rev: [u64; 3],
    shared: Arc<LiveShared>,
}

struct LiveShared {
    /// Identity of the shared state, which keys live storage on the GPU.
    id: u64,
    state: Mutex<LiveState>,
}

#[derive(Default)]
pub(crate) struct LiveState {
    pub(crate) data: Option<[[f32; 4]; 16]>,
    pub(crate) storage: [Option<LiveSlot>; 3],
}

/// One storage slot set through the setters.
pub(crate) struct LiveSlot {
    pub(crate) data: Vec<f32>,
    /// Bumped on every change.
    pub(crate) version: u64,
    /// The version of the last change that was not a range: a replacement or a
    /// resize. A copy older than this has to be uploaded whole.
    pub(crate) base: u64,
    /// `(version, start, end)` for each range written since `base`, oldest
    /// first, capped at [`RANGE_LOG`]. A renderer whose copy is too old for the
    /// log to cover uploads the whole slot instead.
    pub(crate) ranges: std::collections::VecDeque<(u64, usize, usize)>,
}

/// How many range updates a slot remembers. Enough for a frame's worth of
/// scattered edits; past that a whole upload is about as cheap anyway.
const RANGE_LOG: usize = 256;

fn next_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl Default for ShaderLive {
    fn default() -> Self {
        Self {
            id: next_id(),
            field_rev: [0; 3],
            shared: Arc::new(LiveShared {
                id: next_id(),
                state: Mutex::new(LiveState::default()),
            }),
        }
    }
}

impl Clone for ShaderLive {
    fn clone(&self) -> Self {
        Self {
            id: next_id(),
            field_rev: self.field_rev,
            shared: self.shared.clone(),
        }
    }
}

impl std::fmt::Debug for ShaderLive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderLive")
            .field("id", &self.id)
            .field("shared", &self.shared.id)
            .finish()
    }
}

impl ShaderLive {
    /// This material value's identity.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }
    /// Identity of the state shared between clones.
    pub(crate) fn shared_id(&self) -> u64 {
        self.shared.id
    }
    pub(crate) fn field_rev(&self, slot: usize) -> u64 {
        self.field_rev[slot]
    }
    pub(crate) fn lock(&self) -> MutexGuard<'_, LiveState> {
        // A panic while holding this cannot leave the floats half-valid in any
        // way that matters, so a poisoned lock is just taken.
        self.shared.state.lock().unwrap_or_else(|e| e.into_inner())
    }
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
            #[cfg(feature = "render")]
            volume_maps: Vec::new(),
            screen_space: false,
            opacity: 1.0,
            transparent: false,
            side: 0,
            tone_mapped: true,
            render_state: super::RenderState::DEFAULT,
            fallback_color: crate::math::Color::WHITE,
            msl: None,
            alpha_to_coverage: false,
            shadow_vertex: false,
            live: ShaderLive::default(),
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
    /// A `precompiled-shaders` build has no WGSL compiler either.
    pub const VALIDATION_AVAILABLE: bool =
        !cfg!(any(target_arch = "wasm32", feature = "precompiled-shaders"));

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
    /// Attach CPU-side volumes for `u_vol0`/`u_vol1` — see
    /// [`volume_maps`](Self::volume_maps). At most two are used.
    #[cfg(feature = "render")]
    pub fn with_volume_maps(mut self, volumes: Vec<std::sync::Arc<crate::volume::Data3DTexture>>) -> Self {
        self.volume_maps = volumes;
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
    ///
    /// The live block if one was set through [`set_data`](Self::set_data) or
    /// [`set_uniform`](Self::set_uniform), otherwise the `data` field.
    pub fn data_slots(&self) -> [[f32; 4]; 16] {
        if let Some(d) = self.live.lock().data {
            return d;
        }
        self.field_slots()
    }

    pub(crate) fn field_slots(&self) -> [[f32; 4]; 16] {
        let mut out = [[0.0f32; 4]; 16];
        for (i, v) in self.data.iter().take(16).enumerate() {
            out[i] = *v;
        }
        out
    }

    /// Replace the uniform block in place, for this material and every clone
    /// of it. Seen by the next frame; costs a 256-byte write.
    pub fn set_data(&self, data: &[[f32; 4]]) {
        let mut out = [[0.0f32; 4]; 16];
        for (i, v) in data.iter().take(16).enumerate() {
            out[i] = *v;
        }
        self.live.lock().data = Some(out);
    }

    /// Set one `vec4` uniform slot (`u_data.data[slot]`) in place. Slots past
    /// 15 are ignored.
    pub fn set_uniform(&self, slot: usize, value: [f32; 4]) {
        if slot >= 16 {
            return;
        }
        let base = self.field_slots();
        let mut st = self.live.lock();
        st.data.get_or_insert(base)[slot] = value;
    }

    /// Write `values` into the uniform block as flat floats starting at float
    /// `offset` (so `offset` 5 is `u_data.data[1].y`). Anything past the
    /// sixty-fourth float is ignored.
    pub fn set_data_floats(&self, offset: usize, values: &[f32]) {
        let base = self.field_slots();
        let mut st = self.live.lock();
        let d = st.data.get_or_insert(base);
        for (k, v) in values.iter().enumerate() {
            let i = offset + k;
            if i >= 64 {
                break;
            }
            d[i / 4][i % 4] = *v;
        }
    }

    /// Replace storage slot `slot` (0, 1 or 2) in place, for this material and
    /// every clone of it. The whole slot is uploaded on the next frame.
    pub fn set_storage(&self, slot: usize, data: Vec<f32>) {
        let Some(slot) = Self::slot(slot) else { return };
        let mut st = self.live.lock();
        let version = st.storage[slot].as_ref().map_or(0, |s| s.version) + 1;
        st.storage[slot] = Some(LiveSlot {
            data,
            version,
            base: version,
            ranges: Default::default(),
        });
    }

    /// Overwrite `values.len()` floats of storage slot `slot` starting at
    /// `offset`, and upload only those on the next frame.
    ///
    /// The slot grows, zero-filled, if the range runs past its end — which
    /// makes that one update a whole upload. A slot not yet set through
    /// [`set_storage`](Self::set_storage) starts from a copy of its field.
    pub fn set_storage_range(&self, slot: usize, offset: usize, values: &[f32]) {
        let Some(slot) = Self::slot(slot) else { return };
        if values.is_empty() {
            return;
        }
        let mut st = self.live.lock();
        let s = st.storage[slot].get_or_insert_with(|| LiveSlot {
            data: self.storage_field(slot).to_vec(),
            version: 1,
            base: 1,
            ranges: Default::default(),
        });
        let end = offset + values.len();
        s.version += 1;
        if end > s.data.len() {
            s.data.resize(end, 0.0);
            s.base = s.version;
            s.ranges.clear();
        } else {
            if s.ranges.len() == RANGE_LOG {
                s.ranges.pop_front();
            }
            s.ranges.push_back((s.version, offset, end));
        }
        s.data[offset..end].copy_from_slice(values);
    }

    /// Say that storage field `slot` was edited in place, so the renderer
    /// re-uploads it. Only needed for in-place edits of `storage0`..`storage2`
    /// themselves — assigning a new `Vec` is noticed without it, and the
    /// setters never need it.
    pub fn mark_storage_dirty(&mut self, slot: usize) {
        if let Some(slot) = Self::slot(slot) {
            self.live.field_rev[slot] += 1;
        }
    }

    fn slot(slot: usize) -> Option<usize> {
        (slot < 3).then_some(slot)
    }

    /// One number per storage slot that changes whenever what the renderer
    /// would upload for it changes: the live version for a slot set through
    /// the setters, else the field's allocation, length and edit count — the
    /// same tests the upload cache makes, without reading the data.
    pub(crate) fn storage_stamps(&self) -> [u64; 3] {
        let live = self.live.lock();
        std::array::from_fn(|slot| match &live.storage[slot] {
            Some(ls) => ls.version ^ (self.live.shared_id() << 20) ^ (1 << 63),
            None => {
                let f = self.storage_field(slot);
                (f.as_ptr() as u64)
                    ^ (f.len() as u64).rotate_left(40)
                    ^ self.live.field_rev(slot).rotate_left(20)
            }
        })
    }

    /// The storage slot as the renderer sees it: the live data where it was
    /// set through the setters, else the field. For backends and readers that
    /// take the floats themselves rather than through the GPU cache.
    pub fn storage(&self, slot: usize) -> std::borrow::Cow<'_, [f32]> {
        let live = self.live.lock();
        match live.storage.get(slot).and_then(|s| s.as_ref()) {
            Some(ls) => std::borrow::Cow::Owned(ls.data.clone()),
            None => std::borrow::Cow::Borrowed(self.storage_field(slot)),
        }
    }

    pub(crate) fn storage_field(&self, slot: usize) -> &[f32] {
        match slot {
            0 => &self.storage0,
            1 => &self.storage1,
            _ => &self.storage2,
        }
    }
}
