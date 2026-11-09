// `Arc` is not decorative on wasm32, it is the same `Arc` the native build
// needs. The handles inside it — wgpu devices, queues, surfaces — are `Send +
// Sync` natively and are not on wasm, where the API is single-threaded by
// construction. One shared type has to satisfy both targets, so on wasm the
// lint fires on code that has no other form.
#![cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]
// On docs.rs (nightly, `--cfg docsrs`), every feature-gated item is badged
// with the features it needs.
#![cfg_attr(docsrs, feature(doc_cfg))]
//! threers — a drop-in three.js replacement for Rust and WebAssembly.
//!
//! Implements the three.js architecture and API surface: a scene graph of
//! [`Object3D`]s, [`BufferGeometry`] with named attributes, [`Material`]s,
//! cameras, and a [`Renderer`] that walks the graph and draws. Backed by
//! wgpu on native and WebGPU in the browser.
//!
//! # Crates and targets
//!
//! - **`rlib`**: native desktop / tooling (`cargo run --example cube`)
//! - **`cdylib`**: wasm32 WebAssembly (`wasm-pack build`)
//!
//! # Web usage
//!
//! Import `web/threejs-shim.js` for a drop-in `THREE.*` API over the wasm
//! bindings. Existing three.js r165-style apps can swap in with minimal changes.
//!
//! # Feature flags
//!
//! The core — math, the scene graph, geometry, materials, lights, cameras,
//! controls, loaders and exporters — is unconditional, and depends on `log`,
//! `slotmap` and `web-time` and nothing else: no GPU stack, and none of the
//! other threers crates. Everything below is a self-contained subsystem
//! behind a feature, the renderer included: turning one off removes its
//! module and its re-exports and touches nothing else.
//!
//! ```toml
//! # The core alone: scene graph, geometry and loaders, no GPU
//! threers = { version = "0.0.12", default-features = false }
//! # The renderer and nothing else
//! threers = { version = "0.0.12", default-features = false, features = ["render"] }
//! ```
//!
//! ## Leaf features
//!
//! | Feature | Default | Role | Extra deps |
//! |---------|---------|------|------------|
//! | `render` | **on** | The wgpu renderer and everything that draws: `Renderer`, headless and windowed rendering, post-processing, marks, volumes, GPU textures | wgpu, bytemuck, pollster |
//! | `captions` | **on** | Subtitles — SRT + WebVTT, on-screen and burned in | threers-captions |
//! | `sketch` | **on** | `sketch()` and the winit-window `WindowRenderer` (implies `render`); off, winit is not built | winit |
//! | `mesh-bvh` | | Accelerated raycast / shapecast | — |
//! | `raytrace` | | Path tracer: global illumination, area lights, refraction, DoF; CPU + wgpu-compute backends (implies `render`) | — |
//! | `bvh-csg` | | Constructive solid geometry (implies `mesh-bvh`) | — |
//! | `openscad` | | `.scad` front end + `Solid` DSL + exact-CSG kernel (implies `bvh-csg`) | — |
//! | `manifold` | | Optional Manifold kernel behind OpenSCAD booleans (implies `openscad`) | manifold-rust |
//! | `planet` | | Planets, moons, starfields, map generation | — |
//! | `fluid` | | Interactive 2D and 3D fluids on the GPU, as render passes (runs on WebGL2; implies `render`) | — |
//! | `softbody` | | Soft bodies that tear and regrow, with a WebGPU compute solver (implies `render`) | threers-softbody |
//! | `usd` | | Universal Scene Description — `.usda`, `.usdc`, `.usdz`, read and written | — |
//! | `native-codec` | | Pure-Rust video encoders, audio decoders and containers (wasm-safe) | threers-codec |
//! | `native-decode` | | Pure-Rust H.264, HEVC, VP9 and AV1 decoders (implies `native-codec`). Decodes on every core, and in hardware where there is some (VideoToolbox, NVDEC, VAAPI, loaded at run time); video on meshes and video export are the `threers-video` crate | threers-codec |
//! | `metal` | | A second renderer on Metal directly, via the Objective-C runtime (macOS/iOS; implies `render`) | — |
//! | `visionos` | | CompositorServices frame loop + ARKit world tracking, stereo and reverse-Z (implies `metal`) | — |
//! | `videotoolbox` | | In-process VideoToolbox encode and decode (macOS) | — |
//! | `nvidia` | | NVENC encode and NVDEC decode on NVIDIA GPUs (Linux/Windows, driver loaded at run time) | — |
//! | `vaapi` | | VAAPI HEVC encode and H.264/HEVC/VP9/AV1 decode on Linux (AMD, Intel; libva loaded at run time) | — |
//! | `parallel` | | rayon-backed CSG/BVH work and frame-parallel encoding (native; a no-op on wasm32) | rayon |
//! | `async` | | tokio runtime for callers driving loaders concurrently (native) | tokio |
//! | `rlx` | | Tensors ↔ geometry and pixels, graph runner, convolution, mesh smoothing, fitted colour grades, palettes | rlx |
//! | `rlx-geo` | | Exact Delaunay + adaptive refinement + Voronoi cell textures | rlx-geo |
//! | `nurbs` | | NURBS curves/surfaces: derivatives, knots, tessellation | — |
//! | `brep` | | Analytic surface provenance on geometry (implies `nurbs`) | — |
//! | `brep-csg` | | Closed-form SSI fast paths in the CSG kernel (implies `brep` + `openscad`) | — |
//! | `brep-kernel` | | True B-rep topology, boolean, fillets (implies `brep-csg`) | — |
//! | `step` | | STEP AP203/214 import + export (implies `brep-kernel`) | — |
//! | `assembly-check` | | Assembly motion checks (body correspondence, axis recovery, swept volume) | — |
//! | `learned-denoise` | | Trained U-Net denoiser beside À-Trous (implies `raytrace`) | — |
//! | `learned-denoise-metal`, `learned-denoise-gpu` | | The same network on Metal or wgpu (imply `learned-denoise`) | — |
//! | `webgl` | | WebGL2 in the browser through wgpu's GL backend, where WebGPU is missing (implies `render`) | — |
//!
//! ## Meta bundles
//!
//! | Bundle | Enables |
//! |--------|---------|
//! | `cad` | `openscad` + `nurbs` + `brep` + `brep-csg` + `parallel` |
//! | `media` | `native-decode` + `native-codec` + `gif` + `captions` |
//! | `gi` | `raytrace` + `learned-denoise` |
//! | `apple` | `metal` + `videotoolbox` |
//! | `full` | `cad` + `media` + `gi` + `planet` + `fluid` + `usd` + `rlx` + `rlx-geo` + `assembly-check` + `async` + `step` + `brep-kernel` |
//! | `wasm-full` | Browser kitchen sink: `captions` + `openscad` + `native-codec` + `nurbs` + `planet` + `raytrace` + `fluid` + `usd` + `softbody` |
//!
//! ```toml
//! # CAD stack in one flag
//! threers = { version = "0.0.10", features = ["cad"] }
//! # everything
//! threers = { version = "0.0.10", features = ["full"] }
//! # browser wasm kitchen sink
//! threers = { version = "0.0.10", features = ["wasm-full"] }
//! # nothing but the renderer
//! threers = { version = "0.0.10", default-features = false }
//! ```
//!
//! Rigid-body physics lives in the companion crate
//! [`threers-physics`](https://docs.rs/threers-physics). Language bindings:
//! see [`docs/bindings.md`](https://github.com/eugenehp/threers/blob/main/docs/bindings.md)
//! (wasm ESM, napi-rs Node native, PyO3).
//!
//! # Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`prelude`] | One glob import for the types nearly every scene uses; nested modules for controls / animation / … |
//! | [`core`] | Scene graph, geometry, raycaster |
//! | [`math`] | Vectors, matrices, colours, bounds, Delaunay, geographic coordinates — the `threers-math` crate |
//! | `audio` | Audio sources, listeners and positional audio as scene nodes — the `threers-audio` crate (`audio`) |
//! | [`origami`] | Rigid origami: crease patterns, degree-4 kinematics, folded meshes |
//! | [`renderer`] | wgpu draw + post-processing; headless offscreen (native) |
//! | `raytrace` | Path-traced rendering — global illumination, on CPU or GPU (`raytrace`) |
//! | [`postprocessing`] | three.js EffectComposer and passes (native and browser) |
//! | [`loaders`] | glTF, OBJ, HDR, … |
//! | [`extras`] | PMREM, noise, marching cubes, … |
//! | [`geometries::lattice`] | 35 lattice generators, and what they are worth: stiffness, conductivity, strength, porosity, pore size |
//! | [`materials`] | PBR materials and [`ShaderMaterial`] |
//! | `captions` | subtitles — SRT + WebVTT, on-screen and burned in (`captions`) |
//! | `planet` | Planets, moons, starfields (`planet` feature) |
//! | `mesh_bvh` | Opt-in BVH (`mesh-bvh` feature) |
//! | `csg` | Opt-in CSG (`bvh-csg` feature) |
//! | `nurbs` / `brep` / `step` | CAD surface stack (feature-gated) |
//! | `codec` | Opt-in media codecs (`native-codec` feature) |
//! | `rlx` | Opt-in RLX bridge (`rlx`, `rlx-geo` features) |

// `PlaneGeometry::new` returns a `BufferGeometry`, and `AxesHelper::new` an
// `Object3D`, because `new THREE.PlaneGeometry(...)` does. Mirroring three.js is
// the point of this crate, so the constructors mirror what three.js constructs
// rather than what Rust convention would have them construct.
#![allow(clippy::new_ret_no_self)]
// Graphics signatures are wide because the quantities are independent, not
// because they want grouping: a ray through a pixel takes an x, a y, a width, a
// height, a sub-pixel jitter, a lens sample and a shutter instant, and no two of
// those belong in a struct together. Bundling them to get under a threshold of
// seven would hide which of them a caller is actually varying.
#![allow(clippy::too_many_arguments)]

// One gate, written once. Rust has no way to name a `cfg` predicate without a
// build script, which this crate does not have; a macro that stamps the
// attribute on what it wraps is the next best thing. Defined before every
// module so all of them see it.

/// Items for where the native Metal backend is: the `metal` feature on macOS,
/// iOS, tvOS or visionOS.
#[allow(unused_macros)] // not every target has a user
macro_rules! cfg_metal {
    ($($item:item)*) => {
        $(
            #[cfg(metal_backend)]
            $item
        )*
    };
}

/// Items for Apple's Metal platforms, feature or not — what wgpu reaches
/// through a `CAMetalLayer`.
#[allow(unused_macros)] // not every target has a user
macro_rules! cfg_apple {
    ($($item:item)*) => {
        $(
            #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "visionos"))]
            $item
        )*
    };
}

/// Items for everywhere else — the same names with nothing behind them, so
/// code that holds an `Option` of a Metal type needs no gate of its own.
#[allow(unused_macros)] // not every target has a user
macro_rules! cfg_no_metal {
    ($($item:item)*) => {
        $(
            #[cfg(not(metal_backend))]
            $item
        )*
    };
}

/// Statements for where the native Metal backend is (see [`cfg_metal`]): a
/// block that is there or is not, for code inside a function.
#[allow(unused_macros)] // not every target has a user
macro_rules! on_metal {
    ($($body:tt)*) => {
        #[cfg(metal_backend)]
        {
            $($body)*
        }
    };
}

pub mod animation;
#[cfg(feature = "assembly-check")]
pub mod assembly;
/// Audio sources, listeners and positional audio: the
/// [`threers-audio`](https://docs.rs/threers-audio) crate (`audio` feature).
#[cfg(feature = "audio")]
pub use threers_audio as audio;
pub mod cameras;
#[cfg(feature = "captions")]
pub mod captions;
pub mod controls;
pub mod core;
pub mod curves;
pub mod exporters;
pub mod extras;
pub mod fur;
pub mod fx;
pub use fur::{FurGroom, FurMaterial};
pub mod geometries;
pub mod helpers;
pub mod kinematics;
pub mod lights;
pub mod loaders;
#[cfg(feature = "render")]
pub mod marks;
pub mod materials;
pub mod math;
pub mod origami;
pub mod particles;
#[cfg(feature = "render")]
pub mod postprocessing;
pub mod prelude;
pub mod renderers;
pub mod scene;
pub mod sdf;
#[cfg(feature = "softbody")]
pub mod softbody;
pub mod stats;
pub mod textures;
pub mod utils;
#[cfg(feature = "render")]
pub mod volume;
pub mod wgsl;

pub mod easing;
pub mod overlay;
// A retained UI: CSS-subset styles, flexbox layout, painting, hit-testing.
// (Its docs are the module's own `//!`; an outer `///` here would make their
// intra-doc links resolve in this scope, where `Ui` is not.)
#[cfg(feature = "render")]
pub mod renderer;
#[cfg(winit_window)]
pub mod sketch;
#[cfg(feature = "ui")]
pub mod ui;

// No outer doc comment: `gpu.rs` carries `//!` docs, and an outer `///` here
// would make the intra-doc links inside them resolve in this scope.
#[cfg(feature = "render")]
pub mod gpu;
pub mod strict_float;
/// The `THREE.*` JS shim's bindings (WebRenderer, WebMesh, loaders, ...).
/// Built with `--cfg threers_no_js_bindings` they are left out: a consumer
/// driving the Rust API from its own wasm-bindgen crate otherwise ships
/// ~1,200 exports it never calls (exports are roots, so nothing prunes them).
#[cfg(all(
    target_arch = "wasm32",
    feature = "render",
    not(threers_no_js_bindings)
))]
pub mod wasm;
#[cfg(all(target_arch = "wasm32", feature = "bvh-csg"))]
pub mod wasm_csg;
#[cfg(all(target_arch = "wasm32", feature = "raytrace"))]
pub mod wasm_denoise;
/// Progressive path tracing for the browser.
#[cfg(all(
    target_arch = "wasm32",
    feature = "raytrace",
    not(threers_no_js_bindings)
))]
pub mod wasm_pathtrace;

cfg_metal! {
    pub mod metal;
}

/// NVENC and NVDEC on NVIDIA GPUs; see [`codec::nvidia`](threers_codec::nvidia).
#[cfg(all(
    feature = "nvidia",
    any(target_os = "linux", target_os = "windows"),
    target_pointer_width = "64"
))]
pub use threers_codec::nvidia;
/// VAAPI decoding on Linux; see [`codec::vaapi`](threers_codec::vaapi).
#[cfg(all(feature = "vaapi", target_os = "linux"))]
pub use threers_codec::vaapi;
/// In-process hardware video encoding and decoding through VideoToolbox, so
/// frames reach the codec without crossing a process boundary. macOS only;
/// they live in `threers-codec`, and [`codec::hw`] puts both behind one
/// interface with a software fallback.
#[cfg(all(feature = "videotoolbox", target_os = "macos"))]
pub use threers_codec::videotoolbox;
#[cfg(all(
    feature = "videotoolbox",
    feature = "native-decode",
    target_os = "macos"
))]
pub use threers_codec::videotoolbox_decode;

#[cfg(feature = "mesh-bvh")]
pub mod mesh_bvh;

#[cfg(feature = "raytrace")]
pub mod raytrace;

#[cfg(feature = "bvh-csg")]
pub mod csg;

#[cfg(feature = "openscad")]
pub mod openscad;

#[cfg(feature = "planet")]
pub mod planet;

// No outer doc comment on purpose: `fluid/mod.rs` carries `//!` docs, and an
// outer `///` here would make the intra-doc links inside them resolve in this
// scope instead of the module's.
#[cfg(feature = "fluid")]
pub mod fluid;

#[cfg(feature = "nurbs")]
pub mod nurbs;

#[cfg(feature = "brep")]
pub mod brep;

#[cfg(feature = "step")]
pub mod step;

#[cfg(feature = "openscad")]
pub mod exact_csg;

#[cfg(any(feature = "rlx", feature = "rlx-geo"))]
pub mod rlx;

#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! log {
    ($($t:tt)*) => {{
        let s = format!($($t)*);
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&s));
    }};
}

#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! log {
    ($($t:tt)*) => {{ eprintln!($($t)*); }};
}

pub use animation::{
    AnimationAction, AnimationClip, AnimationMixer, Interpolation, KeyframeTrack, TrackTarget,
};
#[cfg(feature = "audio")]
pub use audio::{
    Audio, AudioAnalyser, AudioBackend, AudioListener, DistanceModel, NoopBackend, PositionalAudio,
    SpatialMix,
};
#[cfg(feature = "brep")]
pub use brep::{retessellate, Surface, SurfaceTable};
pub use cameras::{Camera, OrthographicCamera, PerspectiveCamera, ScreenProjector};
#[cfg(feature = "captions")]
pub use captions::{
    CaptionAlign, CaptionAnchor, CaptionError, CaptionFont, CaptionFormat, CaptionOverlay,
    CaptionPainter, CaptionStyle, CaptionTrack, Cue,
};
pub use controls::{
    ArcballControls, DragControls, FirstPersonControls, GlobeControls, MapControls, OrbitControls,
    PointerEvent, PointerLockControls, TrackballControls,
};
pub use core::{
    Bone, BufferAttribute, BufferGeometry, Clock, InstancedMesh, Intersection, Layers,
    LineSegments, Mesh, MorphAttributes, MorphTarget, Object3D, ObjectArena, ObjectId, ObjectKind,
    Points, Raycaster, Skeleton, SkinnedMesh, Sprite, DEFAULT_INFER_QUADS_ANGLE,
};
pub use curves::{
    parse_svg_subpaths, segments_to_path_data, CatmullRomCurve3, CubicBezierCurve,
    CubicBezierCurve3, Curve2, Curve3, CurvePath, EllipseCurve, LineCurve, LineCurve3, NURBSCurve,
    NURBSSurface, Path, PathSegment, QuadraticBezierCurve, QuadraticBezierCurve3, Shape,
    SplineCurve, SvgPathError,
};
pub use extras::{
    CcdIkSolver, IkBone, MarchingCubes, Octree, PmremGenerator, SimplexNoise, PMREM_MIP_LEVELS,
};
#[cfg(feature = "nurbs")]
pub use geometries::NurbsGeometry;
pub use geometries::{
    BoxGeometry, BoxLineGeometry, CapsuleGeometry, ChiralRule, CircleGeometry, Conductivity,
    ConeGeometry, Conform, ConvexGeometry, Cuboct, CuboctAssembly, CuboctAssemblyPlan,
    CuboctAssemblyStep, CuboctFrame, CuboctJoint, CuboctJointKind, CylinderGeometry, DecalGeometry,
    DodecahedronGeometry, EdgesGeometry, ExtrudeGeometry, Field, FrameMaterial, FrameResponse,
    Glyph, Hand, IcosahedronGeometry, Infill, KirigamiAssembly, KirigamiCell, KirigamiCoreLattice,
    KirigamiCrease, KirigamiCreaseKind, KirigamiExpandedMiura, KirigamiFace, KirigamiFaceKind,
    KirigamiMesh, KirigamiNet, KirigamiNetPanel, KirigamiPreset, LatheGeometry, Lattice,
    LatticeGeometry, LatticeKind, LatticeMetrics, LatticeStyle, OctahedronGeometry,
    ParametricGeometry, PlaneGeometry, PolygonGeometry, PolyhedronGeometry, QuadGeometry, Region,
    RingGeometry, SeedLayout, SolidMaterial, Solver, SphereGeometry, Stiffness, Stochastic,
    Strength, Strut, TetrahedronGeometry, TextGeometry, TorusGeometry, TorusKnotGeometry, Tpms,
    TubeGeometry, WireframeGeometry, KIRIGAMI_NET_VARIANT,
};
pub use helpers::{
    ArrowHelper, AxesHelper, BoxHelper, CameraHelper, DirectionalLightHelper, GridHelper,
    HemisphereLightHelper, PointLightHelper, PolarGridHelper, SkeletonHelper, SpotLightHelper,
    VertexNormalsHelper, VertexTangentsHelper,
};
pub use lights::{
    AmbientLight, DirectionalLight, HemisphereLight, Light, PointLight, RectAreaLight,
    ShadowSettings, ShadowType, SpotLight,
};
pub use loaders::{
    AlembicArchive, AlembicError, ColladaError, ColladaLoader, DracoError, DracoHeader, ExrError,
    ExrLoader, FbxError, FbxLoader, FbxScene, GltfError, GltfImages, GltfLoader, GltfScene,
    HdrError, HdrLoader, Ktx2, Ktx2Error, MeshoptError, ObjLoader, PlyLoader, PolyMesh, StlLoader,
    TtfError, TtfFont, TtfGlyph,
};
pub use origami::{
    book_cardinal_cp, book_cardinal_stages, classic_bird_cp, classic_bird_stages, eagle_cp,
    eagle_stages, frog_cp, frog_stages, giang_cardinal_cp, giang_cardinal_stages, AssembledBird,
    Assignment, BirdKind, BookCardinalStage, ClassicBirdStage, ClosureReport, CreasePattern,
    Degree4, EagleStage, Edge, EdgeKind, FoldedBirdPart, FoldedState, FrogStage,
    GiangCardinalStage, PartPose, VertexMode, TWIST_ALPHA,
};
// Writing, which the loaders module also provides.
pub use loaders::{scene_to_glb, scene_to_gltf, GltfExportOptions};
pub use materials::MirrorMaterial;
pub use materials::{
    AtmosphereMaterial, BasicMaterial, DepthMaterial, LambertMaterial, LineBasicMaterial,
    MatcapMaterial, Material, MaterialKind, MaterialTextureSlots, NormalMaterial, PhongMaterial,
    PhysicalMaterial, PointsMaterial, ShaderMaterial, SpriteMaterial, StandardMaterial,
    ToonMaterial, TransparencyMode,
};
pub use materials::{Blending, DepthFunc, RenderState};
pub use math::{
    Box2, Box3, Color, Cylindrical, Euler, EulerOrder, Frustum, Line3, Matrix3, Matrix4, Plane,
    Quaternion, Ray, Sphere, Spherical, Triangle, Vector2, Vector3, Vector4,
};
// Colour beyond `Color`: spaces and CSS parsing, ramps and schemes, scales.
pub use math::{
    ColorParseError, ColorRamp, ColorScale, ColorSpace, Palette, RampSpace, ScaleKind, SchemeKind,
};
#[cfg(feature = "nurbs")]
pub use nurbs::{NurbsCurve, NurbsSurface, TessellationOptions};
#[cfg(feature = "render")]
pub use postprocessing::{
    BloomPass, CopyPass, EffectComposer, FilmPass, FxaaPass, GlitchPass, OutlinePass, OutputPass,
    Pass, RenderPass, ShaderPass, SsaoPass, SsrPass, UnrealBloomPass,
};
#[cfg(all(feature = "render", not(target_arch = "wasm32")))]
pub use renderer::headless::{
    GpuVendor, HeadlessBuilder, HeadlessConfig, HeadlessRenderer, MappedFrame,
};
/// The `wgpu` this crate was built against, re-exported.
///
/// [`Renderer::new`] takes a `wgpu::Device`, a `wgpu::Queue` and a
/// `wgpu::TextureFormat`, so a caller on the native path necessarily has its own
/// `wgpu` — and a Rust type is scoped to the exact crate version it came from.
/// `wgpu::Device` from 0.20 and `wgpu::Device` from 30 are two unrelated types
/// that happen to print the same, and cargo will happily build both into one
/// graph, so the mismatch surfaces as `expected `wgpu::Device`, found
/// `wgpu::Device`` and no obvious cause.
///
/// Going through `threers::wgpu` makes that impossible to get wrong: it is by
/// construction the same crate this renderer was compiled against.
///
/// ```no_run
/// use threers::wgpu;
///
/// fn take(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>) {
///     let _ = threers::Renderer::new(device, queue, wgpu::TextureFormat::Bgra8UnormSrgb, 800, 600);
/// }
/// ```
#[cfg(feature = "render")]
pub use wgpu;

pub use math::ToneMapping;
#[cfg(feature = "render")]
pub use renderer::{halton, stratified_jitter, RenderBackend, RenderTarget, Renderer};
pub use renderers::{
    svg_from_rgba, Css2dRenderer, Css3dRenderer, SvgOptions, SvgRenderer, SvgShading,
};
pub use scene::Scene;
pub use stats::Stats;
pub use textures::{
    pack_rgba16f, CubeTexture, DataTexture, DepthTexture, Texture, TextureFilter, TextureFormat,
    TextureWrap,
};
pub use utils::png::{decode_png, encode_png, PngImage};
#[cfg(feature = "render")]
pub use volume::Data3DTexture;

pub mod time_smear;
pub use time_smear::TimeSmear;
pub use utils::{
    center as center_geometry, compute_tangents, compute_vertex_normals, merge_geometries,
    scale as scale_geometry,
};

#[cfg(feature = "native-codec")]
pub use codec::animation::{
    encode_animation_rgba, encode_animation_rgba_with_progress, format_animation_progress,
    AnimationEncodeError, AnimationEncodeOptions, AnimationExportPhase, AnimationExportProgress,
    BrowserCodec,
};
#[cfg(feature = "native-codec")]
pub use codec::apng::ApngEncoder;
#[cfg(feature = "gif")]
#[allow(deprecated)]
pub use codec::gif::DecodedFrame;
#[cfg(feature = "gif")]
pub use codec::gif::{
    decode_gif, encode_gif, DisposalMethod, DisposalMode, GifDecoder, GifEncoder, GifError,
    GifFrame, GifFrameMeta, GifInfo, GifOptions, GifVersion, GifWriter, LzwClearMode, PaletteMode,
    QuantizerKind,
};
#[cfg(feature = "native-codec")]
pub use codec::h264::{encode_mp4, encode_mp4_with_captions, H264Encoder};
#[cfg(feature = "native-codec")]
pub use codec::hevc::{HevcEncoder, TransparentEncoder, Yuv420Frame};
#[cfg(feature = "native-codec")]
pub use codec::vp9::{
    encode_inter_frame, encode_inter_frame_altref, encode_inter_frame_compound,
    encode_inter_frame_golden, encode_inter_frame_refresh, encode_inter_newmv_residual,
    encode_inter_newmv_skip, encode_inter_residual, encode_inter_zeromv_skip, encode_intra_frame,
    encode_intra_gray, Reconstruction,
};
#[cfg(feature = "native-codec")]
pub use codec::webm::{encode_gray_webm, encode_webm, mux_webm, WebmCodec, WebmFrame, WebmParams};
/// From-scratch, pure-Rust media codecs — the
/// [`threers-codec`](https://docs.rs/threers-codec) crate: H.264, HEVC and VP9
/// encoders, H.264/HEVC/VP9/AV1 decoders (`native-decode`), AAC, MP3, Opus,
/// Vorbis and FLAC, MP4/WebM/Matroska/Ogg, APNG and GIF — no ffmpeg, no C
/// bindings, and wasm-compatible. Enable the `native-codec` feature.
#[cfg(feature = "native-codec")]
pub use threers_codec as codec;

#[cfg(feature = "mesh-bvh")]
pub use mesh_bvh::{
    BuildOptions as MeshBvhBuildOptions, BvhHit, BvhNodeView, MeshBvh, SerializedMeshBvh, AVERAGE,
    CENTER, CONTAINED, INTERSECTED, NOT_INTERSECTED, SAH,
};

#[cfg(feature = "raytrace")]
pub use raytrace::{
    intersect_box, Aov, BackgroundMode, CpuBackend, RaytraceBackend, RaytraceError,
    RaytraceRenderer, RaytraceScene, RaytraceSettings,
};

#[cfg(feature = "bvh-csg")]
pub use csg::{
    evaluate_hierarchy, CsgBrush, CsgEvaluator, CsgNode, CsgOperation, CsgOperationGroup,
    TriangleIntersection, ADDITION, DIFFERENCE, HOLLOW_INTERSECTION, HOLLOW_SUBTRACTION,
    INTERSECTION, REVERSE_SUBTRACTION, SUBTRACTION,
};

#[cfg(feature = "openscad")]
pub use exact_csg::report::{mesh_report, DefectCluster, MeshReport};
#[cfg(feature = "openscad")]
pub use openscad::export::{
    geometry_to_3mf, geometry_to_glb, geometry_to_obj, geometry_to_off, geometry_to_ply,
    parts_to_glb,
};
#[cfg(feature = "openscad")]
pub use openscad::freecad::{
    fcstd_meshes, fcstd_to_geometry, fcstd_to_solid, geometry_to_fcstd, geometry_to_fcstd_with,
    parse_mesh_kernel, parse_mesh_kernel_full, parts_to_fcstd, parts_to_fcstd_with, FcstdDocument,
    FcstdDocumentMeta, FcstdError, FcstdLoader, FcstdMesh, FcstdMeshSpec, FcstdPlacement,
    FcstdReadOptions, FcstdSkipped, FcstdWriteOptions, FcstdWriteReport, FcstdWriter,
};
#[cfg(feature = "openscad")]
pub use openscad::mechanism::{
    DriveSpec, MateSpec, MateSpecKind, MechanismSpec, PartFit, PartSpec,
};
#[cfg(feature = "openscad")]
pub use openscad::scad::{
    clear_files, parse_scad, parse_scad_at, parse_scad_file, parse_scad_file_at,
    parse_scad_file_for_parts, parse_scad_file_with, parse_scad_mechanism, parse_scad_mechanism_at,
    parse_scad_mechanism_file, parse_scad_mechanism_file_at, parse_scad_with, parse_scad_with_base,
    register_file, scad_file_values, scad_value, scad_values, scad_values_in,
};
#[cfg(feature = "openscad")]
pub use openscad::schematic::{
    schematic_project, schematic_project_view, schematic_to_rgba, schematic_to_svg, Schematic,
    SchematicOptions, View,
};
#[cfg(feature = "openscad")]
pub use openscad::{build_nema17, build_printer};
#[cfg(feature = "openscad")]
pub use openscad::{
    cone, cube, cylinder, difference_all, frustum, geometry_bounds, geometry_to_stl, hull,
    intersection_all, linear_extrude, linear_extrude_holes, polyhedron, rotate_extrude,
    rotate_extrude_fn, set_origin, solid, sphere, sphere_fn, union_all, Origin, Solid,
};

#[cfg(feature = "planet")]
pub use planet::{
    generate_earth_maps, generate_starfield, Atmosphere, HeightField, MapBuffer, Planet,
    PlanetHandles, PlanetMaps, Starfield,
};
#[cfg(all(feature = "planet", not(target_arch = "wasm32")))]
pub use planet::{EarthTextures, MapSources};
