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
//! | Feature | Role |
//! |---------|------|
//! | `mesh-bvh` | Accelerated raycast / shapecast |
//! | `bvh-csg` | Constructive solid geometry (implies `mesh-bvh`) |
//! | `video` | Native frame-sequence export via ffmpeg / native GIF |
//! | `native-codec` | Pure-Rust GIF, APNG, VP9/WebM, HEVC (wasm-safe) |
//!
//! # Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`core`] | Scene graph, geometry, raycaster |
//! | [`renderer`] | wgpu draw + post-processing; headless offscreen (native) |
//! | [`postprocessing`] | EffectComposer-style pass chain (native) |
//! | [`loaders`] | glTF, OBJ, HDR, … |
//! | [`extras`] | PMREM, noise, marching cubes, … |
//! | [`materials`] | PBR materials and [`ShaderMaterial`] |
//! | `mesh_bvh` | Opt-in BVH (`mesh-bvh` feature) |
//! | `csg` | Opt-in CSG (`bvh-csg` feature) |
//! | `video` | Opt-in video export (`video` feature, native) |
//! | `codec` | Opt-in media codecs (`native-codec` feature) |

pub mod math;
pub mod core;
pub mod geometries;
pub mod materials;
pub mod lights;
pub mod textures;
pub mod curves;
pub mod cameras;
pub mod controls;
pub mod animation;
pub mod loaders;
pub mod audio;
pub mod postprocessing;
pub mod renderers;
pub mod utils;
pub mod helpers;
pub mod extras;
pub mod stats;
pub mod scene;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
pub mod renderer;

#[cfg(feature = "mesh-bvh")]
pub mod mesh_bvh;

#[cfg(feature = "bvh-csg")]
pub mod csg;

#[cfg(feature = "openscad")]
pub mod openscad;

/// Robust boolean kernel (M1 scaffold) — winding-number classification +
/// straddle detection. See `docs/openscad-plan.md`.
#[cfg(feature = "openscad")]
pub mod exact_csg;

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

pub use math::{
    Vector2, Vector3, Vector4,
    Matrix3, Matrix4,
    Quaternion, Euler, Color,
    Box2, Box3, Sphere, Ray, Plane, Triangle, Frustum,
    Spherical, Cylindrical, Line3,
};
pub use core::{
    Object3D, ObjectId, ObjectKind, ObjectArena,
    BufferGeometry, BufferAttribute, Mesh,
    Layers, Clock, Raycaster, Intersection,
    LineSegments, Points, Sprite, InstancedMesh,
    Bone, Skeleton, SkinnedMesh, MorphTarget, MorphAttributes,
};
pub use geometries::{
    BoxGeometry, PlaneGeometry, SphereGeometry,
    CircleGeometry, RingGeometry,
    CylinderGeometry, ConeGeometry,
    TorusGeometry, TorusKnotGeometry, CapsuleGeometry,
    PolyhedronGeometry, TetrahedronGeometry, OctahedronGeometry,
    IcosahedronGeometry, DodecahedronGeometry,
    EdgesGeometry, WireframeGeometry,
    LatheGeometry, TubeGeometry, ExtrudeGeometry,
    ParametricGeometry, ConvexGeometry, DecalGeometry,
    TextGeometry, Glyph, BoxLineGeometry,
};
pub use materials::{
    Material, MaterialKind, MaterialTextureSlots, TransparencyMode,
    BasicMaterial, LambertMaterial, PhongMaterial,
    StandardMaterial, PhysicalMaterial,
    NormalMaterial, DepthMaterial, ToonMaterial, MatcapMaterial,
    LineBasicMaterial, PointsMaterial, SpriteMaterial,
    ShaderMaterial,
};
pub use materials::MirrorMaterial;
pub use lights::{
    Light, AmbientLight, DirectionalLight, PointLight,
    SpotLight, HemisphereLight, RectAreaLight, ShadowSettings,
};
pub use textures::{
    Texture, TextureFormat, TextureFilter, TextureWrap,
    CubeTexture, DataTexture, DepthTexture,
};
pub use helpers::{
    AxesHelper, GridHelper, BoxHelper, CameraHelper, ArrowHelper, PolarGridHelper,
    DirectionalLightHelper, PointLightHelper, SpotLightHelper,
    HemisphereLightHelper, SkeletonHelper,
    VertexNormalsHelper, VertexTangentsHelper,
};
pub use curves::{
    Curve2, Curve3,
    LineCurve, LineCurve3,
    QuadraticBezierCurve, QuadraticBezierCurve3,
    CubicBezierCurve, CubicBezierCurve3,
    EllipseCurve,
    CatmullRomCurve3, SplineCurve,
    CurvePath, Path, Shape,
    NURBSCurve, NURBSSurface,
};
pub use cameras::{Camera, PerspectiveCamera, OrthographicCamera};
pub use controls::{
    OrbitControls, TrackballControls, FirstPersonControls,
    DragControls, ArcballControls, PointerLockControls, PointerEvent,
};
pub use animation::{
    Interpolation, KeyframeTrack, TrackTarget,
    AnimationClip, AnimationMixer, AnimationAction,
};
pub use loaders::{
    GltfLoader, GltfScene, GltfError, GltfImages,
    ObjLoader, StlLoader, PlyLoader,
    HdrLoader, HdrError, ExrLoader, ExrError,
    FbxLoader, FbxError, ColladaLoader, ColladaError,
    TtfFont, TtfError, TtfGlyph,
};
pub use audio::{Audio, AudioListener, PositionalAudio, AudioAnalyser, AudioBackend, NoopBackend};
pub use postprocessing::{
    EffectComposer, Pass,
    RenderPass, BloomPass, FxaaPass, OutlinePass, ToneMappingPass,
    FilmPass, GlitchPass, SsaoPass, SsrPass, CopyPass,
};
pub use utils::{compute_vertex_normals, compute_tangents, merge_geometries, center as center_geometry, scale as scale_geometry};
pub use renderers::{Css2dRenderer, Css3dRenderer, SvgRenderer};
pub use extras::{MarchingCubes, CcdIkSolver, IkBone, Octree, SimplexNoise, PmremGenerator, PMREM_MIP_LEVELS};
pub use stats::Stats;
pub use scene::Scene;
pub use renderer::{Renderer, RenderTarget};
#[cfg(not(target_arch = "wasm32"))]
pub use renderer::headless::{HeadlessBuilder, HeadlessConfig, HeadlessRenderer};

/// Native video export (frame sequence → ffmpeg). Enable the `video` feature.
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
pub mod video;
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
pub use video::{
    export_video, export_video_with_progress, format_video_progress, VideoCodec, VideoError,
    VideoExportEvent, VideoExportPhase, VideoExportProgress, VideoExporter, VideoOptions,
    VideoQuality,
};

/// From-scratch, pure-Rust media codecs (HEVC, VP9/WebM, APNG, GIF, bitstream) —
/// no ffmpeg, no C bindings, and wasm-compatible. Enable the `native-codec`
/// feature.
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
pub use codec::hevc::{HevcEncoder, TransparentEncoder, Yuv420Frame};
#[cfg(feature = "native-codec")]
pub use codec::webm::{encode_gray_webm, encode_webm, mux_webm, WebmCodec, WebmFrame, WebmParams};
#[cfg(feature = "native-codec")]
pub use codec::vp9::{
    encode_inter_frame, encode_inter_frame_altref, encode_inter_frame_compound,
    encode_inter_frame_golden, encode_inter_frame_refresh, encode_inter_newmv_residual,
    encode_inter_newmv_skip, encode_inter_residual, encode_inter_zeromv_skip, encode_intra_frame,
    encode_intra_gray, Reconstruction,
};

#[cfg(feature = "mesh-bvh")]
pub use mesh_bvh::{MeshBvh, BvhHit, BuildOptions as MeshBvhBuildOptions, SerializedMeshBvh, AVERAGE, CENTER, SAH, NOT_INTERSECTED, INTERSECTED, CONTAINED};

#[cfg(feature = "bvh-csg")]
pub use csg::{
    CsgBrush, CsgEvaluator, CsgOperation, CsgOperationGroup, CsgNode, evaluate_hierarchy,
    build_bvh_csg_hierarchy_geometry, load_positions_geometry_bin,
    assert_step_verts, evaluate_live_through, evaluate_through, js_target_verts,
    step1_shell_cut, step2_add_sphere, step3_win_cut, step4_win_frame,
    JS_STEP1_VERTS, JS_STEP2_VERTS, JS_STEP3_VERTS, JS_STEP4_VERTS,
    ADDITION, SUBTRACTION, REVERSE_SUBTRACTION, INTERSECTION, DIFFERENCE,
    HOLLOW_SUBTRACTION, HOLLOW_INTERSECTION,
};

#[cfg(feature = "openscad")]
pub use openscad::export::{geometry_to_3mf, geometry_to_glb, geometry_to_obj, geometry_to_off};
#[cfg(feature = "openscad")]
pub use openscad::{
    cone, cube, cylinder, difference_all, frustum, geometry_to_stl, hull, intersection_all,
    linear_extrude, linear_extrude_holes, polyhedron, rotate_extrude, rotate_extrude_fn, solid,
    sphere, sphere_fn, union_all, Solid,
};
#[cfg(feature = "openscad")]
pub use openscad::{build_nema17, build_printer};
#[cfg(feature = "openscad")]
pub use openscad::scad::{clear_files, parse_scad, parse_scad_file, register_file};
