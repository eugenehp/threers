//! One import for the things nearly every scene uses.
//!
//! ```
//! use threers::prelude::*;
//!
//! let mut scene = Scene::new();
//! scene.add(Object3D::mesh(Mesh::new(
//!     BoxGeometry::new(1.0, 1.0, 1.0),
//!     Material::Standard(StandardMaterial::new(Color::from_hex(0xff8844))),
//! )));
//! scene.add_light(DirectionalLight::new(Color::WHITE, 3.0));
//!
//! let mut camera = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 100.0);
//! camera.position = Vector3::new(3.0, 2.0, 4.0);
//! camera.look_at(Vector3::ZERO);
//! ```
//!
//! # What is in the root prelude
//!
//! The scene graph, the geometry and material types, lights, cameras, textures,
//! the renderers, the vector maths, and the [`Camera`] trait — which has to be
//! in scope to call `view_matrix` on a camera, and is the sort of thing a
//! prelude exists for.
//!
//! # Nested modules (opt-in)
//!
//! A prelude is glob-imported, so every name in the root is a name your own
//! code cannot use. Subsystems you reach for deliberately live in nested
//! modules — import them explicitly:
//!
//! ```ignore
//! use threers::prelude::*;
//! use threers::prelude::controls::*;
//! use threers::prelude::animation::*;
//! ```
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`controls`] | Orbit / trackball / first-person / … |
//! | [`animation`] | Mixer, clips, keyframe tracks |
//! | [`loaders`] | glTF, OBJ, HDR, … |
//! | [`helpers`] | Axes, grid, light helpers, … |
//! | `sketch` | A window, a scene and a loop in one call (native) |
//! | `csg` | CSG evaluator / brush (`bvh-csg`) |
//! | `openscad` | Solid DSL + `.scad` parse (`openscad`) |
//! | `nurbs` | NURBS curve/surface + geometry (`nurbs`) |
//! | `raytrace` | Path tracer entry points (`raytrace`) |
//! | [`captions`] | Cue tracks / overlay (`captions`) |
//! | `video` | GIF/APNG/WebM/H.264/HEVC encoders, MP4, the hardware-first `Mp4Encoder`, and the decoders (`native-codec`; decoders also `video` or `native-decode`) |
//!
//! # What is deliberately not in any prelude module
//!
//! - **Maths shapes** — `Plane`, `Sphere`, `Triangle`, `Ray`, `Box2`, `Box3`,
//!   `Frustum`, … Common words; a scene rarely needs them by name.
//! - **Curves and paths** — `Path`, `Shape`, `Curve2`, `Curve3` and friends.
//! - **Companion crates** — physics and cinematic animation stay in
//!   `threers_physics::prelude` / `threers_animation::prelude`.
//!
//! Everything the crate exports is still available from the root
//! (`threers::Frustum`, `threers::Path`, …); the prelude is a curated subset,
//! not a second API.

// Traits first: these have to be in scope for their methods to be callable.
pub use crate::cameras::Camera;

pub use crate::cameras::{OrthographicCamera, PerspectiveCamera};
pub use crate::core::{
    BufferAttribute, BufferGeometry, Clock, InstancedMesh, Layers, LineSegments, Mesh, Object3D,
    ObjectId, ObjectKind, Points, Raycaster, Sprite,
};
pub use crate::geometries::{
    BoxGeometry, CapsuleGeometry, CircleGeometry, ConeGeometry, CylinderGeometry, ExtrudeGeometry,
    IcosahedronGeometry, LatheGeometry, PlaneGeometry, QuadGeometry, RingGeometry, SphereGeometry,
    TorusGeometry, TorusKnotGeometry, TubeGeometry,
};
pub use crate::lights::{
    AmbientLight, DirectionalLight, HemisphereLight, Light, PointLight, RectAreaLight,
    ShadowSettings, ShadowType, SpotLight,
};
pub use crate::materials::{
    BasicMaterial, DepthMaterial, LambertMaterial, LineBasicMaterial, MatcapMaterial, Material,
    MaterialKind, NormalMaterial, PhongMaterial, PhysicalMaterial, PointsMaterial, ShaderMaterial,
    SpriteMaterial, StandardMaterial, ToonMaterial, TransparencyMode,
};
pub use crate::math::{Color, Euler, Matrix3, Matrix4, Quaternion, Vector2, Vector3, Vector4};
// The small things that get reached for constantly and were not here: a
// mapping from one range to another, a seeded generator, a gradient, an easing
// curve, noise. Every one of them was being hand-rolled at the call site —
// twenty-six separate copies of the generator alone.
pub use crate::easing::Easing;
pub use crate::extras::SimplexNoise;
pub use crate::math::{
    convex_hull, delaunay, inverse_lerp, lerp, map_range, map_range_clamped, smootherstep,
    smoothstep, ticks, voronoi_cells, ColorRamp, RampSpace, Rng, ScaleLinear, ScaleLog, ScalePow,
};
pub use crate::math::{ColorScale, ColorSpace, Palette, ScaleKind};
pub use crate::overlay::Overlay;
pub use crate::renderer::{RenderTarget, Renderer, ToneMapping};
pub use crate::scene::{FogParams, Scene};
pub use crate::textures::{
    CubeTexture, DataTexture, Texture, TextureFilter, TextureFormat, TextureWrap,
};

/// Offscreen rendering, and the PNG round trip that usually follows it.
#[cfg(not(target_arch = "wasm32"))]
pub use crate::renderer::headless::{HeadlessBuilder, HeadlessConfig, HeadlessRenderer};
pub use crate::utils::png::{decode_png, encode_png, PngImage};

/// Vector SVG export, and the raster-in-SVG passthrough beside it.
pub use crate::renderers::{svg_from_rgba, SvgOptions, SvgRenderer, SvgShading};

/// The Metal backend's entry points, when the `metal` feature is on.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub use crate::metal::{
    MetalDevice, MetalError, MetalHeadlessRenderer, MetalRenderTarget, MetalRenderer, RenderView,
};

/// The visionOS frame loop, when the `visionos` feature is on and that is what
/// is being built.
#[cfg(all(feature = "visionos", target_os = "visionos"))]
pub use crate::metal::visionos::{ImmersiveRenderer, WorldTracking};

/// Orbit / trackball / FPS / drag controls.
pub mod controls {
    pub use crate::controls::{
        ArcballControls, DragControls, FirstPersonControls, OrbitControls, PointerEvent,
        PointerLockControls, TrackballControls,
    };
}

/// Scene-graph animation mixer and keyframe clips (crate-root animation, not
/// `threers-animation`).
pub mod animation {
    pub use crate::animation::{
        AnimationAction, AnimationClip, AnimationMixer, Interpolation, KeyframeTrack, TrackTarget,
    };
}

/// Common asset loaders.
pub mod loaders {
    pub use crate::loaders::{
        AlembicArchive, AlembicError, ColladaError, ColladaLoader, DracoError, ExrError, ExrLoader,
        FbxError, FbxLoader, FbxScene, GltfError, GltfImages, GltfLoader, GltfScene, HdrError,
        HdrLoader, Ktx2, Ktx2Error, MeshoptError, ObjLoader, PlyLoader, PolyMesh, StlLoader,
        TtfError, TtfFont, TtfGlyph,
    };
    // Writing, not only reading.
    pub use crate::loaders::{scene_to_glb, scene_to_gltf, GltfExportOptions};
}

/// Debug / authoring helpers.
pub mod helpers {
    pub use crate::helpers::{
        ArrowHelper, AxesHelper, BoxHelper, CameraHelper, DirectionalLightHelper, GridHelper,
        HemisphereLightHelper, PointLightHelper, PolarGridHelper, SkeletonHelper, SpotLightHelper,
        VertexNormalsHelper, VertexTangentsHelper,
    };
}

/// Constructive solid geometry (`bvh-csg`).
#[cfg(feature = "bvh-csg")]
pub mod csg {
    pub use crate::csg::{
        CsgBrush, CsgEvaluator, CsgNode, CsgOperation, CsgOperationGroup, ADDITION, DIFFERENCE,
        INTERSECTION, SUBTRACTION,
    };
}

/// OpenSCAD solid DSL and `.scad` parse (`openscad`).
#[cfg(feature = "openscad")]
pub mod openscad {
    pub use crate::openscad::scad::parse_scad;
    pub use crate::openscad::{cone, cube, cylinder, sphere, Solid};
}

/// NURBS curves / surfaces (`nurbs`).
#[cfg(feature = "nurbs")]
pub mod nurbs {
    pub use crate::geometries::NurbsGeometry;
    pub use crate::nurbs::{NurbsCurve, NurbsSurface, TessellationOptions};
}

/// Path tracer (`raytrace`).
#[cfg(feature = "raytrace")]
pub mod raytrace {
    pub use crate::raytrace::{
        Aov, BackgroundMode, CpuBackend, RaytraceBackend, RaytraceError, RaytraceRenderer,
        RaytraceScene, RaytraceSettings,
    };
}

/// Video and animation codecs (`native-codec`), and the decoders (`video` or
/// `native-decode`). The codec crate's own prelude, `threers::codec::prelude`,
/// goes further: audio, containers and every codec module.
///
/// `h264` and `hevc` are imported as modules because both have a
/// `CompressedEncoder`: write `h264::CompressedEncoder`.
#[cfg(feature = "native-codec")]
pub mod video {
    #[cfg(any(feature = "video", feature = "native-decode"))]
    pub use crate::codec::decode::{Backend, DecodedFrame, Decoder, NativeDecoder};
    pub use crate::codec::hevc::Yuv420Frame;
    pub use crate::codec::hw::{EncodeBackend, EncodeOptions, Mp4Encoder};
    pub use crate::codec::{
        encode_animation_rgba, encode_webm, h264, hevc, mp4, mp4_demux, AnimationEncodeOptions,
        ApngEncoder, BrowserCodec,
    };
    #[cfg(feature = "gif")]
    pub use crate::codec::GifEncoder;
}

/// Subtitle cues and on-screen overlay (`captions`).
#[cfg(feature = "captions")]
pub mod captions {
    pub use crate::captions::{
        CaptionAlign, CaptionAnchor, CaptionError, CaptionFont, CaptionFormat, CaptionOverlay,
        CaptionPainter, CaptionStyle, CaptionTrack, Cue,
    };
}

#[cfg(test)]
mod tests {
    //! The prelude's job is to compile. These build a scene using nothing but
    //! it, which is the only way to catch a name that quietly stopped being
    //! exported.
    use super::*;

    #[test]
    fn a_scene_can_be_built_from_the_prelude_alone() {
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x101010);
        let id = scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(1.0, 16, 12),
            Material::Standard(StandardMaterial::new(Color::WHITE).with_roughness(0.4)),
        )));
        scene.add_light(AmbientLight::new(Color::WHITE, 0.2));
        scene.add(Object3D::light(DirectionalLight::new(Color::WHITE, 2.0)));

        if let Some(object) = scene.get_mut(id) {
            object.position = Vector3::new(0.0, 1.0, 0.0);
            object.quaternion = Quaternion::from_axis_angle(Vector3::UP, 0.5);
            object.scale = Vector3::ONE * 2.0;
        }
        scene.update_world();

        let world = scene.get(id).unwrap().matrix_world;
        assert_eq!(world.elements[13], 1.0);
    }

    #[test]
    fn the_camera_trait_is_in_scope() {
        let mut camera = PerspectiveCamera::new(50.0, 1.5, 0.1, 100.0);
        camera.position = Vector3::new(0.0, 0.0, 5.0);
        camera.look_at(Vector3::ZERO);
        // `view_matrix` comes from the trait: this line is the test.
        let view = camera.view_matrix();
        assert_eq!(view.elements[14], -5.0);

        let ortho = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
        assert_eq!(ortho.near_far(), (0.1, 10.0));
    }

    #[test]
    fn glass_shadows_and_colour_from_the_prelude_alone() {
        let glass = PhysicalMaterial::new(Color::WHITE)
            .with_transmission(1.0, 1.5, 0.5)
            .with_transparency(TransparencyMode::Refract);
        assert_eq!(glass.transparency, TransparencyMode::Refract);
        let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
        sun.shadow.filter = Some(ShadowType::Pcss);
        assert_eq!(sun.shadow.filter, Some(ShadowType::Pcss));

        let (red, alpha) = Color::parse_css("oklch(62.8% 0.2577 29.23 / 0.5)").unwrap();
        assert!(red.r > 0.9 && red.g < 0.05 && alpha == 0.5);
        let scale = ColorScale::sequential(ColorRamp::viridis(), 0.0, 1.0).log(10.0);
        let _ = scale.map(0.5);
        let palette = Palette::from_hex(&[0xff0000, 0x00ff00]);
        let _ = (palette, ScaleKind::Linear, ColorSpace::Oklch);
    }

    #[test]
    fn geometry_and_texture_types_are_reachable() {
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        assert_eq!(geometry.draw_count(), 3);

        let texture = Texture::new(1, 1, TextureFormat::Rgba8Unorm, vec![255, 0, 0, 255]);
        assert_eq!(texture.mag_filter, TextureFilter::Linear);
        assert_eq!(texture.wrap_s, TextureWrap::ClampToEdge);
    }

    #[test]
    fn png_round_trips_through_the_prelude() {
        let rgba = vec![255u8, 0, 0, 255, 0, 255, 0, 255];
        let png = encode_png(2, 1, &rgba);
        let image: PngImage = decode_png(&png).unwrap();
        assert_eq!((image.width, image.height), (2, 1));
        assert_eq!(image.rgba, rgba);
    }

    #[test]
    fn nested_controls_module_exports_orbit() {
        use super::controls::OrbitControls;
        let _ = std::mem::size_of::<OrbitControls>();
    }
}
