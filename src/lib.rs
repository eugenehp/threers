// `Arc` is not decorative on wasm32, it is the same `Arc` the native build
// needs. The handles inside it — wgpu devices, queues, surfaces — are `Send +
// Sync` natively and are not on wasm, where the API is single-threaded by
// construction. One shared type has to satisfy both targets, so on wasm the
// lint fires on code that has no other form.
#![cfg_attr(
    target_arch = "wasm32",
    allow(clippy::arc_with_non_send_sync)
)]
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
//! The three.js API surface — scene graph, geometry, materials, lights,
//! cameras, controls, loaders, post-processing, renderer — is unconditional.
//! Everything below is a self-contained subsystem: turning one off removes its
//! module and its re-exports and touches nothing else.
//!
//! ## Leaf features
//!
//! | Feature | Default | Role | Extra deps |
//! |---------|---------|------|------------|
//! | `captions` | **on** | Subtitles — SRT + WebVTT, on-screen and burned in | — |
//! | `mesh-bvh` | | Accelerated raycast / shapecast | — |
//! | `raytrace` | | Path tracer: global illumination, area lights, refraction, DoF; CPU + wgpu-compute backends | — |
//! | `bvh-csg` | | Constructive solid geometry (implies `mesh-bvh`) | — |
//! | `openscad` | | `.scad` front end + `Solid` DSL + exact-CSG kernel (implies `bvh-csg`) | — |
//! | `manifold` | | Optional Manifold kernel behind OpenSCAD booleans (implies `openscad`) | manifold-rust |
//! | `planet` | | Planets, moons, starfields, map generation | — |
//! | `video` | | Native frame-sequence export via ffmpeg (implies `captions`) | — |
//! | `native-codec` | | Pure-Rust GIF, APNG, VP9/WebM, H.264/MP4, HEVC (wasm-safe) | — |
//! | `metal` | | A second renderer on Metal directly, via the Objective-C runtime (macOS/iOS) | — |
//! | `visionos` | | CompositorServices frame loop + ARKit world tracking, stereo and reverse-Z (implies `metal`) | — |
//! | `videotoolbox` | | In-process VideoToolbox encode (macOS) | — |
//! | `parallel` | | rayon-backed CSG/BVH work (native; a no-op on wasm32) | rayon |
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
//!
//! ## Meta bundles
//!
//! | Bundle | Enables |
//! |--------|---------|
//! | `cad` | `openscad` + `nurbs` + `brep` + `brep-csg` + `parallel` |
//! | `media` | `video` + `native-codec` + `captions` |
//! | `gi` | `raytrace` + `learned-denoise` |
//! | `apple` | `metal` + `videotoolbox` |
//! | `full` | `cad` + `media` + `gi` + `planet` + `rlx` + `rlx-geo` + `assembly-check` + `async` + `step` + `brep-kernel` |
//! | `wasm-full` | Browser kitchen sink: `captions` + `openscad` + `native-codec` + `nurbs` + `planet` + `raytrace` |
//!
//! ```toml
//! # CAD stack in one flag
//! threers = { version = "0.0.5", features = ["cad"] }
//! # everything
//! threers = { version = "0.0.5", features = ["full"] }
//! # browser wasm kitchen sink
//! threers = { version = "0.0.5", features = ["wasm-full"] }
//! # nothing but the renderer
//! threers = { version = "0.0.5", default-features = false }
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
//! | [`origami`] | Rigid origami: crease patterns, degree-4 kinematics, folded meshes |
//! | [`renderer`] | wgpu draw + post-processing; headless offscreen (native) |
//! | [`raytrace`] | Path-traced rendering — global illumination, on CPU or GPU |
//! | [`postprocessing`] | EffectComposer-style pass chain (native) |
//! | [`loaders`] | glTF, OBJ, HDR, … |
//! | [`extras`] | PMREM, noise, marching cubes, … |
//! | [`geometries::lattice`] | 35 lattice generators, and what they are worth: stiffness, conductivity, strength, porosity, pore size |
//! | [`materials`] | PBR materials and [`ShaderMaterial`] |
//! | `captions` | subtitles — SRT + WebVTT, on-screen and burned in (`captions`) |
//! | `planet` | Planets, moons, starfields (`planet` feature) |
//! | `mesh_bvh` | Opt-in BVH (`mesh-bvh` feature) |
//! | `csg` | Opt-in CSG (`bvh-csg` feature) |
//! | `nurbs` / `brep` / `step` | CAD surface stack (feature-gated) |
//! | `video` | Opt-in video export (`video` feature, native) |
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

pub mod animation;
/// Generic checks for assemblies that move: body correspondence across poses,
/// axis recovery, and swept free volume. Knows nothing about what it inspects.
#[cfg(feature = "assembly-check")]
pub mod assembly;
pub mod audio;
pub mod cameras;
#[cfg(feature = "captions")]
pub mod captions;
pub mod controls;
pub mod core;
pub mod curves;
pub mod extras;
pub mod geometries;
pub mod helpers;
/// Inverse kinematics: damped least squares joint solving for serial arms,
/// with position and tool direction as separate residuals rather than a
/// weighted sum. See [`kinematics`].
pub mod kinematics;
/// Rigid origami: degree-4 vertex kinematics, crease-pattern closure, and a
/// dual walk that folds a consistent assignment off the plane.
pub mod origami;
pub mod lights;
pub mod loaders;
pub mod materials;
pub mod math;
pub mod postprocessing;
/// One import for the things nearly every scene uses — see [`prelude`].
pub mod prelude;
pub mod renderers;
pub mod scene;
pub mod stats;
pub mod textures;
pub mod utils;

pub mod renderer;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
/// The trained denoiser in a browser, split so the work can go to web workers.
#[cfg(all(target_arch = "wasm32", feature = "raytrace"))]
pub mod wasm_denoise;
/// Progressive path tracing for the browser.
#[cfg(all(target_arch = "wasm32", feature = "raytrace"))]
pub mod wasm_pathtrace;

/// A second renderer that talks to Metal directly, with no `metal-rs`, no
/// `objc` crate and no C in the build — see [`metal`](crate::metal) for what it
/// covers. macOS and iOS only; the `metal` feature is inert elsewhere.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal;

/// In-process hardware video encoding through VideoToolbox, so frames reach the
/// encoder without crossing a process boundary. macOS only; see
/// [`videotoolbox`](crate::videotoolbox).
#[cfg(all(feature = "videotoolbox", target_os = "macos"))]
pub mod videotoolbox;

#[cfg(feature = "mesh-bvh")]
pub mod mesh_bvh;

/// A physically-based path tracer beside the rasteriser — global illumination,
/// area lights, refraction, depth of field. CPU and wgpu-compute backends over
/// one scene form. Enable the `raytrace` feature.
#[cfg(feature = "raytrace")]
pub mod raytrace;

#[cfg(feature = "bvh-csg")]
pub mod csg;

#[cfg(feature = "openscad")]
pub mod openscad;

#[cfg(feature = "planet")]
pub mod planet;

/// NURBS curves and surfaces in f64 — derivatives, knot operations, exact conic
/// and quadric constructors, curvature-adaptive tessellation. Stage 0 of the
/// B-rep roadmap; see `docs/brep-nurbs-plan.md`.
///
/// Distinct from [`curves::NURBSCurve`] / [`curves::NURBSSurface`], which are
/// the unconditional f32 three.js parity types and convert into these.
#[cfg(feature = "nurbs")]
pub mod nurbs;

/// Analytic surface provenance: every triangle knows the surface it was sampled
/// from, so normals, UVs and re-tessellation derive from the surface rather than
/// the mesh. Stage 1 of the B-rep roadmap; see `docs/brep-nurbs-plan.md`.
#[cfg(feature = "brep")]
pub mod brep;

/// STEP (ISO 10303) AP203/AP214 advanced B-rep import and export. Stage 5 of
/// the B-rep roadmap; see `docs/brep-nurbs-plan.md`.
#[cfg(feature = "step")]
pub mod step;

/// Robust boolean kernel (M1 scaffold) — winding-number classification +
/// straddle detection. See `docs/openscad-plan.md`.
#[cfg(feature = "openscad")]
pub mod exact_csg;

/// The RLX bridge: tensors ↔ geometry and pixels, a compiled-graph runner, and
/// exact Delaunay/Voronoi. One module, two independent features (`rlx`,
/// `rlx-geo`) — the renderer itself is unchanged either way.
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
pub use audio::{Audio, AudioAnalyser, AudioBackend, AudioListener, NoopBackend, PositionalAudio};
#[cfg(feature = "brep")]
pub use brep::{retessellate, Surface, SurfaceTable};
pub use cameras::{Camera, OrthographicCamera, PerspectiveCamera};
#[cfg(feature = "captions")]
pub use captions::{
    CaptionAlign, CaptionAnchor, CaptionError, CaptionFont, CaptionFormat, CaptionOverlay,
    CaptionPainter, CaptionStyle, CaptionTrack, Cue,
};
pub use controls::{
    ArcballControls, DragControls, FirstPersonControls, OrbitControls, PointerEvent,
    PointerLockControls, TrackballControls,
};
pub use core::{
    Bone, BufferAttribute, BufferGeometry, Clock, InstancedMesh, Intersection, Layers,
    LineSegments, Mesh, MorphAttributes, MorphTarget, Object3D, ObjectArena, ObjectId, ObjectKind,
    Points, Raycaster, Skeleton, SkinnedMesh, Sprite,
};
pub use curves::{
    CatmullRomCurve3, CubicBezierCurve, CubicBezierCurve3, Curve2, Curve3, CurvePath, EllipseCurve,
    LineCurve, LineCurve3, NURBSCurve, NURBSSurface, Path, QuadraticBezierCurve,
    QuadraticBezierCurve3, Shape, SplineCurve,
};
pub use extras::{
    CcdIkSolver, IkBone, MarchingCubes, Octree, PmremGenerator, SimplexNoise, PMREM_MIP_LEVELS,
};
#[cfg(feature = "nurbs")]
pub use geometries::NurbsGeometry;
pub use geometries::{
    BoxGeometry, BoxLineGeometry, CapsuleGeometry, CircleGeometry, ConeGeometry, ConvexGeometry,
    CylinderGeometry, DecalGeometry, DodecahedronGeometry, EdgesGeometry, ExtrudeGeometry, Glyph,
    IcosahedronGeometry, Infill, KirigamiAssembly, KirigamiCell, KirigamiCoreLattice,
    KirigamiCrease, KirigamiCreaseKind,
    KirigamiExpandedMiura, KirigamiFace, KirigamiFaceKind, KirigamiMesh, KirigamiNet,
    KirigamiNetPanel, KirigamiPreset, KIRIGAMI_NET_VARIANT, LatheGeometry, Lattice, LatticeGeometry, LatticeKind, LatticeStyle,
    OctahedronGeometry, ParametricGeometry, PlaneGeometry, PolyhedronGeometry, Region,
    RingGeometry, SphereGeometry, Strut, TetrahedronGeometry, TextGeometry, TorusGeometry,
    TorusKnotGeometry, Tpms, TubeGeometry, WireframeGeometry, ChiralRule, Cuboct, CuboctAssembly,
    CuboctAssemblyPlan, CuboctAssemblyStep, CuboctFrame, CuboctJoint, CuboctJointKind, FrameMaterial,
    FrameResponse, Hand, Conductivity, Conform, Field, LatticeMetrics, SolidMaterial, Stiffness,
    Solver, Stochastic, Strength,
};
pub use helpers::{
    ArrowHelper, AxesHelper, BoxHelper, CameraHelper, DirectionalLightHelper, GridHelper,
    HemisphereLightHelper, PointLightHelper, PolarGridHelper, SkeletonHelper, SpotLightHelper,
    VertexNormalsHelper, VertexTangentsHelper,
};
pub use origami::{
    book_cardinal_cp, book_cardinal_stages, classic_bird_cp, classic_bird_stages, eagle_cp,
    eagle_stages, frog_cp, frog_stages, giang_cardinal_cp, giang_cardinal_stages, AssembledBird,
    Assignment, BirdKind, BookCardinalStage, ClassicBirdStage, ClosureReport, CreasePattern,
    Degree4, EagleStage, Edge, EdgeKind, FoldedBirdPart, FoldedState, FrogStage, GiangCardinalStage,
    PartPose, TWIST_ALPHA, VertexMode,
};
pub use lights::{
    AmbientLight, DirectionalLight, HemisphereLight, Light, PointLight, RectAreaLight,
    ShadowSettings, SpotLight,
};
pub use loaders::{
    ColladaError, ColladaLoader, ExrError, ExrLoader, FbxError, FbxLoader, GltfError, GltfImages,
    GltfLoader, GltfScene, HdrError, HdrLoader, ObjLoader, PlyLoader, StlLoader, TtfError, TtfFont,
    TtfGlyph,
};
pub use materials::MirrorMaterial;
pub use materials::{
    AtmosphereMaterial, BasicMaterial, DepthMaterial, LambertMaterial, LineBasicMaterial,
    MatcapMaterial, Material, MaterialKind, MaterialTextureSlots, NormalMaterial, PhongMaterial,
    PhysicalMaterial, PointsMaterial, ShaderMaterial, SpriteMaterial, StandardMaterial,
    ToonMaterial, TransparencyMode,
};
pub use math::{
    Box2, Box3, Color, Cylindrical, Euler, Frustum, Line3, Matrix3, Matrix4, Plane, Quaternion,
    Ray, Sphere, Spherical, Triangle, Vector2, Vector3, Vector4,
};
#[cfg(feature = "nurbs")]
pub use nurbs::{NurbsCurve, NurbsSurface, TessellationOptions};
pub use postprocessing::{
    BloomPass, CopyPass, EffectComposer, FilmPass, FxaaPass, GlitchPass, OutlinePass, Pass,
    RenderPass, SsaoPass, SsrPass, ToneMappingPass,
};
#[cfg(not(target_arch = "wasm32"))]
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
pub use wgpu;

pub use renderer::{RenderTarget, Renderer, ToneMapping};
pub use renderers::{Css2dRenderer, Css3dRenderer, SvgRenderer};
pub use scene::Scene;
pub use stats::Stats;
pub use textures::{
    pack_rgba16f, CubeTexture, DataTexture, DepthTexture, Texture, TextureFilter, TextureFormat,
    TextureWrap,
};
pub use utils::png::{decode_png, encode_png, PngImage};

/// Rendering a sequence: stills, a video, or both from one pass.
pub mod sequence;
pub use sequence::{
    render_sequence, video_available, SequenceError, SequenceOptions, SequenceReport,
};
pub use utils::{
    center as center_geometry, compute_tangents, compute_vertex_normals, merge_geometries,
    scale as scale_geometry,
};

/// Native video export (frame sequence → ffmpeg). Enable the `video` feature.
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
pub mod video;
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
pub use video::{
    export_video, export_video_with_progress, format_video_progress, CaptionExport, CaptionMode,
    VideoCodec, VideoError, VideoExportEvent, VideoExportPhase, VideoExportProgress, VideoExporter,
    VideoOptions, VideoQuality,
};

/// From-scratch, pure-Rust media codecs (H.264/MP4, HEVC, VP9/WebM, APNG, GIF,
/// bitstream) — no ffmpeg, no C bindings, and wasm-compatible. Enable the
/// `native-codec` feature.
#[cfg(feature = "native-codec")]
pub mod codec;
#[cfg(feature = "native-codec")]
pub use codec::animation::{
    encode_animation_rgba, encode_animation_rgba_with_progress, format_animation_progress,
    AnimationEncodeError, AnimationEncodeOptions, AnimationExportPhase, AnimationExportProgress,
    BrowserCodec,
};
#[cfg(feature = "native-codec")]
pub use codec::apng::ApngEncoder;
#[cfg(feature = "native-codec")]
pub use codec::gif::{
    decode_gif, encode_gif, DecodedFrame, DisposalMethod, DisposalMode, GifDecoder, GifEncoder,
    GifError, GifFrameMeta, GifInfo, GifOptions, GifVersion, GifWriter, LzwClearMode, PaletteMode,
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

#[cfg(feature = "mesh-bvh")]
pub use mesh_bvh::{
    BuildOptions as MeshBvhBuildOptions, BvhHit, MeshBvh, SerializedMeshBvh, AVERAGE, CENTER,
    CONTAINED, INTERSECTED, NOT_INTERSECTED, SAH,
};

#[cfg(feature = "raytrace")]
pub use raytrace::{
    intersect_box, Aov, BackgroundMode, CpuBackend, RaytraceBackend, RaytraceError, RaytraceRenderer,
    RaytraceScene, RaytraceSettings,
};

#[cfg(feature = "bvh-csg")]
pub use csg::{
    assert_step_verts, build_bvh_csg_hierarchy_geometry, evaluate_hierarchy, evaluate_live_through,
    evaluate_through, js_target_verts, load_positions_geometry_bin, step1_shell_cut,
    step2_add_sphere, step3_win_cut, step4_win_frame, CsgBrush, CsgEvaluator, CsgNode,
    CsgOperation, CsgOperationGroup, ADDITION, DIFFERENCE, HOLLOW_INTERSECTION, HOLLOW_SUBTRACTION,
    INTERSECTION, JS_STEP1_VERTS, JS_STEP2_VERTS, JS_STEP3_VERTS, JS_STEP4_VERTS,
    REVERSE_SUBTRACTION, SUBTRACTION,
};

#[cfg(feature = "openscad")]
pub use exact_csg::report::{mesh_report, DefectCluster, MeshReport};
#[cfg(feature = "openscad")]
pub use openscad::export::{geometry_to_3mf, geometry_to_glb, geometry_to_obj, geometry_to_off, parts_to_glb};
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
    parse_scad_file_for_parts, parse_scad_file_with, parse_scad_mechanism, parse_scad_mechanism_at, parse_scad_mechanism_file,
    parse_scad_mechanism_file_at, parse_scad_with, parse_scad_with_base, register_file,
    scad_file_values, scad_value, scad_values, scad_values_in,
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
