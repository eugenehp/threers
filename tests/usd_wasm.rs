//! USD in a browser.
//!
//! Every one of these runs twice: natively as an ordinary `#[test]`, and in a
//! browser as a `#[wasm_bindgen_test]`. The same assertions, so the two cannot
//! drift — a wasm test that has quietly stopped being run is worse than none,
//! and a wasm-only test tends to test less than its native twin.
//!
//! What is actually being checked is that the parts of this crate a browser
//! would use work *there*: the parser, the crate reader and writer, LZ4, the
//! zip container, the PNG codec, and composition. None of them may touch the
//! filesystem, the clock, or a thread.
//!
//! ```sh
//! SAFARIDRIVER=$(command -v safaridriver) \
//!   CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
//!   cargo test --target wasm32-unknown-unknown --features usd --test usd_wasm
//! ```
#![cfg(feature = "usd")]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

/// Run natively and in the browser, from one body.
macro_rules! both {
    ($(#[doc = $doc:expr])* fn $name:ident() $body:block) => {
        $(#[doc = $doc])*
        #[cfg_attr(not(target_arch = "wasm32"), test)]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn $name() $body
    };
}

use threers::core::{Mesh, Object3D, ObjectArena};
use threers::loaders::usd::{self, UsdExport, UsdLoader};
use threers::materials::{Material, StandardMaterial};
use threers::math::Color;

const SOURCE: &str = r#"#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "World"
{
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Tri"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
            interpolation = "vertex"
        )
    }
}
"#;

fn textured_scene() -> (ObjectArena, Vec<threers::core::ObjectId>) {
    use std::sync::Arc;
    use threers::textures::{Texture, TextureFormat};
    let texture = Texture::new(
        2,
        2,
        TextureFormat::Rgba8UnormSrgb,
        vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255],
    );
    let material = StandardMaterial {
        map: Some(Arc::new(texture)),
        ..StandardMaterial::new(Color::new(1.0, 1.0, 1.0))
    };
    let mut geometry = threers::core::BufferGeometry::new();
    geometry.set_attribute(
        "position",
        threers::core::BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
    );
    geometry.set_attribute(
        "uv",
        threers::core::BufferAttribute::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 2),
    );
    geometry.set_index(vec![0, 1, 2]);
    let mut arena = ObjectArena::new();
    let mut object = Object3D::mesh(Mesh::new(geometry, material.into()));
    object.name = "Quad".into();
    let id = arena.insert(object);
    (arena, vec![id])
}

both! {
    /// The text parser, and the scene it builds.
    fn a_usda_parses_and_becomes_a_scene() {
        let layer = UsdLoader::parse_layer(SOURCE.as_bytes()).expect("parses");
        assert_eq!(layer.prim_at("/World/Tri").map(|p| p.type_name.clone()), Some("Mesh".into()));

        let scene = UsdLoader::parse(SOURCE.as_bytes()).expect("a scene");
        assert!((scene.meters_per_unit - 0.01).abs() < 1e-6);
        assert_eq!(scene.up_axis, 'Y');
        let root = scene.arena.get(scene.roots[0]).expect("a root");
        assert!((root.position.y - 2.0).abs() < 1e-5, "the transform came through");
    }
}

both! {
    /// The crate writer and reader — LZ4, the integer packing, the path table.
    ///
    /// This is the part most likely to break on a different target: it is all
    /// byte layout and shifts, and wasm is 32-bit.
    fn a_usdc_round_trips() {
        let layer = UsdLoader::parse_layer(SOURCE.as_bytes()).expect("parses");
        let bytes = usd::layer_to_usdc(&layer);
        assert!(bytes.starts_with(b"PXR-USDC"), "not a crate");

        let back = UsdLoader::parse_layer(&bytes).expect("reads back");
        let tri = back.prim_at("/World/Tri").expect("the mesh");
        assert_eq!(tri.value("faceVertexIndices").unwrap().flat_u32(), vec![0, 1, 2]);
        assert_eq!(
            tri.value("points").unwrap().flat_f32(),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            back.prim_at("/World").unwrap().value("xformOp:translate").unwrap().flat_f32(),
            vec![1.0, 2.0, 3.0]
        );
    }
}

both! {
    /// The package: zip container, PNG codec, and the textures inside it.
    ///
    /// A browser is exactly where this matters — a `.usdz` fetched over the
    /// network has to yield a drawable scene without touching a disk.
    fn a_usdz_round_trips_with_its_textures() {
        let (arena, roots) = textured_scene();
        let bytes = UsdExport::scene(&arena, &roots).usdz();
        assert_eq!(&bytes[..2], b"PK", "not a zip");

        let scene = UsdLoader::parse(&bytes).expect("reads back");
        let quad = scene
            .arena
            .get_objects_by_name(scene.roots[0], "Quad")
            .into_iter()
            .next()
            .expect("the quad");
        let object = scene.arena.get(quad).unwrap();
        let threers::core::ObjectKind::Mesh(mesh) = &object.kind else {
            panic!("not a mesh");
        };
        let Material::Standard(material) = &*mesh.material else {
            panic!("not a standard material");
        };
        let map = material.map.as_ref().expect("the image came out of the package");
        assert_eq!((map.width, map.height), (2, 2));
        assert_eq!(&map.data[..4], &[255, 0, 0, 255], "the pixels changed");
    }
}

both! {
    /// Composition, with the layers held in memory rather than on a disk.
    fn layers_compose_from_memory() {
        let base = r#"#usda 1.0

def Xform "World"
{
    def Sphere "Ball"
    {
        double radius = 1
    }
}
"#;
        let shot = r#"#usda 1.0
(
    subLayers = [@base.usda@]
)

over "World"
{
    over "Ball"
    {
        double radius = 7
    }
}
"#;
        let mut resolver = usd::MemoryResolver::new();
        resolver.insert("base.usda", base.as_bytes());
        let scene = UsdLoader::open(
            shot.as_bytes(),
            "shot.usda",
            &resolver,
            &usd::ComposeOptions::default(),
        )
        .expect("composes");
        assert!(!scene.roots.is_empty(), "the composed stage has a root");

        let layer = UsdLoader::open_layer(
            shot.as_bytes(),
            "shot.usda",
            &resolver,
            &usd::ComposeOptions::default(),
        )
        .expect("composes");
        assert_eq!(
            layer.prim_at("/World/Ball").unwrap().value("radius").unwrap().flat_f32(),
            vec![7.0],
            "the stronger layer wins"
        );
    }
}

both! {
    /// All three forms describe the same document.
    fn the_three_forms_agree() {
        let (arena, roots) = textured_scene();
        let out = UsdExport::scene(&arena, &roots);
        let from_text = UsdLoader::parse_layer(out.usda().as_bytes()).expect("text");
        let from_crate = UsdLoader::parse_layer(&out.usdc()).expect("crate");
        let from_package = UsdLoader::parse_layer(&out.usdz()).expect("package");
        for layer in [&from_text, &from_crate, &from_package] {
            assert_eq!(
                layer
                    .prim_at("/Root/Material_0/DiffuseTex")
                    .and_then(|p| p.value("inputs:file"))
                    .and_then(|v| v.as_str().map(str::to_owned)),
                Some("textures/Material_0_DiffuseTex.png".into())
            );
        }
    }
}
