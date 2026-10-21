//! A USD stage, composed: one chess-like piece defined once as a `class`,
//! placed three times by internal reference, with a `finish` variant set
//! that swaps its `UsdPreviewSurface` material. Three panels in
//! `out/usd_stage.png`:
//!
//! 1. every piece in the class's default finish, brass;
//! 2. the default changed to enamel — one word in the layer;
//! 3. each piece selecting its own finish, and that stage written out by
//!    threers as a `.usdz` (a binary crate layer inside) and read back.
//!
//! ```bash
//! cargo run --release --features usd --example usd_stage
//! ```

use threers::core::ObjectArena;
use threers::lights::Light;
use threers::loaders::usd::{ComposeOptions, MemoryResolver};
use threers::loaders::{scene_to_usdz, UsdLoader};
use threers::prelude::*;

const W: u32 = 400;
const H: u32 = 300;

/// The stage. `$DEFAULT` is the class's own variant selection and
/// `$A`/`$B`/`$C` each placement's, if it has one.
const STAGE: &str = r#"#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
)

class Xform "Piece" (
    prepend variantSets = "finish"
    variants = {
        string finish = "$DEFAULT"
    }
)
{
    def Cylinder "Base"
    {
        double radius = 0.42
        double height = 0.16
        uniform token axis = "Y"
        double3 xformOp:translate = (0, 0.08, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cone "Body"
    {
        double radius = 0.3
        double height = 0.9
        uniform token axis = "Y"
        double3 xformOp:translate = (0, 0.6, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Sphere "Head"
    {
        double radius = 0.2
        double3 xformOp:translate = (0, 1.12, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    variantSet "finish" = {
        "brass" {
            over "Base" { rel material:binding = </Materials/Brass> }
            over "Body" { rel material:binding = </Materials/Brass> }
            over "Head" { rel material:binding = </Materials/Brass> }
        }
        "enamel" {
            over "Base" { rel material:binding = </Materials/Ink> }
            over "Body" { rel material:binding = </Materials/Enamel> }
            over "Head" { rel material:binding = </Materials/Enamel> }
        }
        "jade" {
            over "Base" { rel material:binding = </Materials/Ink> }
            over "Body" { rel material:binding = </Materials/Jade> }
            over "Head" { rel material:binding = </Materials/Brass> }
        }
    }
}

def Scope "Materials"
{
    def Material "Brass"
    {
        token outputs:surface.connect = </Materials/Brass/Surface.outputs:surface>
        def Shader "Surface"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.86, 0.62, 0.26)
            float inputs:metallic = 1
            float inputs:roughness = 0.28
            token outputs:surface
        }
    }
    def Material "Enamel"
    {
        token outputs:surface.connect = </Materials/Enamel/Surface.outputs:surface>
        def Shader "Surface"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.75, 0.12, 0.1)
            float inputs:metallic = 0
            float inputs:roughness = 0.18
            float inputs:clearcoat = 1
            token outputs:surface
        }
    }
    def Material "Jade"
    {
        token outputs:surface.connect = </Materials/Jade/Surface.outputs:surface>
        def Shader "Surface"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.2, 0.62, 0.42)
            float inputs:metallic = 0
            float inputs:roughness = 0.35
            token outputs:surface
        }
    }
    def Material "Ink"
    {
        token outputs:surface.connect = </Materials/Ink/Surface.outputs:surface>
        def Shader "Surface"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.06, 0.06, 0.07)
            float inputs:metallic = 0
            float inputs:roughness = 0.5
            token outputs:surface
        }
    }
    def Material "Board"
    {
        token outputs:surface.connect = </Materials/Board/Surface.outputs:surface>
        def Shader "Surface"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.78, 0.74, 0.66)
            float inputs:metallic = 0
            float inputs:roughness = 0.7
            token outputs:surface
        }
    }
}

def Xform "World"
{
    def Cube "Board"
    {
        double size = 1
        rel material:binding = </Materials/Board>
        double3 xformOp:translate = (0, -0.05, 0)
        float3 xformOp:scale = (4.2, 0.1, 2.4)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Xform "A" (
        prepend references = </Piece>$A
    )
    {
        double3 xformOp:translate = (-1.3, 0, 0.2)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Xform "B" (
        prepend references = </Piece>$B
    )
    {
        double3 xformOp:translate = (0, 0, -0.3)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Xform "C" (
        prepend references = </Piece>$C
    )
    {
        double3 xformOp:translate = (1.3, 0, 0.2)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"#;

/// A placement's own variant selection, or nothing: an empty selection is
/// still an opinion, and would hide the class's default rather than defer to
/// it.
fn select(finish: &str) -> String {
    if finish.is_empty() {
        String::new()
    } else {
        format!("\n        variants = {{ string finish = \"{finish}\" }}")
    }
}

/// The stage with these selections, as a scene ready to draw.
fn stage(default: &str, a: &str, b: &str, c: &str) -> (ObjectArena, Vec<ObjectId>) {
    let text = STAGE
        .replace("$DEFAULT", default)
        .replace("$A", &select(a))
        .replace("$B", &select(b))
        .replace("$C", &select(c));
    // `open` composes — references, variants, the class — where `parse`
    // would read the one layer as written.
    let usd = UsdLoader::open(
        text.as_bytes(),
        "stage.usda",
        &MemoryResolver::new(),
        &ComposeOptions::default(),
    )
    .expect("the stage composes");
    if std::env::var("USD_TREE").is_ok() {
        for &r in &usd.roots {
            usd.arena
                .traverse(r, &mut |_, o| println!("  {} {:?}", o.name, o.position));
        }
    }
    (usd.arena, usd.roots)
}

/// Lights and a backdrop around a loaded graph.
fn scene(arena: ObjectArena, roots: &[ObjectId]) -> Scene {
    let mut s = Scene::new();
    s.arena = arena;
    s.root = s.arena.insert(Object3D::group());
    for &r in roots {
        s.arena.add_child(s.root, r);
    }
    s.background = Color::from_hex(0x1c2028);
    // Something for the brass to reflect.
    s.environment = Some(threers::extras::RoomEnvironment::pmrem(128));
    s.environment_intensity = 0.35;
    s.add_light(HemisphereLight::new(
        Color::from_hex(0xc8d8ff),
        Color::from_hex(0x302820),
        0.8,
    ));
    let mut sun = DirectionalLight::new(Color::from_hex(0xfff0e0), 2.0);
    sun.cast_shadow = true;
    sun.shadow.camera_size = 6.0;
    sun.shadow.normal_bias = 0.02;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(-3.0, 5.0, 3.5);
    s.add(l);
    let mut ids = Vec::new();
    s.arena.traverse(s.root, &mut |id, _| ids.push(id));
    for id in ids {
        if let Some(o) = s.arena.get_mut(id) {
            o.cast_shadow = true;
            o.receive_shadow = true;
        }
    }
    s
}

fn main() {
    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    let mut camera = PerspectiveCamera::new(38.0, W as f32 / H as f32, 0.1, 50.0);
    camera.position = Vector3::new(0.6, 2.6, 5.2);
    camera.look_at(Vector3::new(0.0, 0.5, 0.0));

    let all_brass = stage("brass", "", "", "");
    let all_enamel = stage("enamel", "", "", "");
    // Each its own, then out through threers' writer and back in.
    let (arena, roots) = stage("brass", "brass", "enamel", "jade");
    let usdz = scene_to_usdz(&arena, &roots, &[]);
    let back = UsdLoader::parse(&usdz).expect("threers reads its own usdz");
    println!(
        "round trip: {} bytes of usdz, {} root prims back",
        usdz.len(),
        back.roots.len()
    );
    let panels = [all_brass, all_enamel, (back.arena, back.roots)];

    let mut sheet = vec![0u8; (W * 3 * H * 4) as usize];
    for (slot, (arena, roots)) in panels.into_iter().enumerate() {
        let mut s = scene(arena, roots.as_slice());
        let px = hr.render_to_rgba(&mut s, &camera);
        for y in 0..H {
            let row = &px[(y * W * 4) as usize..((y + 1) * W * 4) as usize];
            let at = ((y * W * 3 + slot as u32 * W) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/usd_stage.png", encode_png(W * 3, H, &sheet)).unwrap();
    println!("wrote out/usd_stage.png (brass, enamel, each its own after a usdz round trip)");
}
