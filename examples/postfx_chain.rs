//! Post-processing through `EffectComposer`, one scene six ways: as rendered,
//! then with ambient occlusion (SSAO), bloom (UnrealBloom), depth of field
//! (Bokeh), a selection outline, and a pixel-art pass. Each chain is a
//! `RenderPass`, the effect, and an `OutputPass` for tone mapping and sRGB —
//! three.js's composer, natively. Written to `out/postfx_chain.png`, three
//! across and two down.
//!
//! ```bash
//! cargo run --release --example postfx_chain
//! ```

use threers::lights::Light;
use threers::postprocessing::{
    BokehPass, EffectComposer, OutlinePass, OutputPass, Pass, RenderPass, RenderPixelatedPass,
    SsaoPass, UnrealBloomPass,
};
use threers::prelude::*;
use threers::renderer::{readback, RenderTarget};

const W: u32 = 420;
const H: u32 = 280;
const FORMAT: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba8Unorm;

/// A chain's effect, given the object to outline.
type Effect = Box<dyn Fn(ObjectId) -> Box<dyn Pass>>;

fn standard(hex: u32, roughness: f32, metalness: f32) -> StandardMaterial {
    let mut m = StandardMaterial::new(Color::from_hex(hex));
    m.roughness = roughness;
    m.metalness = metalness;
    m
}

/// A row of shapes receding from the camera on a floor, two of them lamps
/// bright enough to bloom. Returns the scene and the shape to outline.
fn scene() -> (Scene, ObjectId) {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x0c0f14);
    s.add_light(HemisphereLight::new(
        Color::from_hex(0xa8c4ff),
        Color::from_hex(0x2c241c),
        0.7,
    ));
    let mut sun = DirectionalLight::new(Color::from_hex(0xfff2e0), 2.4);
    sun.cast_shadow = true;
    sun.shadow.camera_size = 12.0;
    sun.shadow.normal_bias = 0.02;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(-4.0, 7.0, 3.0);
    s.add(l);

    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(40.0, 40.0),
        standard(0x8a8478, 0.85, 0.0).into(),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);

    let mut add = |geometry: BufferGeometry, material: StandardMaterial, at: Vector3| {
        let mut o = Object3D::mesh(Mesh::new(geometry, material.into()));
        o.position = at;
        o.cast_shadow = true;
        o.receive_shadow = true;
        s.add(o)
    };
    add(
        BoxGeometry::new(1.2, 1.2, 1.2),
        standard(0xc8553d, 0.5, 0.0),
        Vector3::new(-2.2, 0.6, 1.6),
    );
    let knot = add(
        TorusKnotGeometry::new(0.55, 0.17, 160, 24, 2, 3),
        standard(0xd8b050, 0.25, 1.0),
        Vector3::new(0.0, 1.0, 0.0),
    );
    add(
        SphereGeometry::new(0.7, 64, 32),
        standard(0x3d7fc8, 0.15, 0.0),
        Vector3::new(2.0, 0.7, -2.2),
    );
    add(
        BoxGeometry::new(0.9, 2.4, 0.9),
        standard(0x6a9a58, 0.6, 0.0),
        Vector3::new(-0.6, 1.2, -5.0),
    );
    for (x, z, hex) in [(1.2f32, 1.8f32, 0xff7a3c), (-2.6, -2.6, 0x52e0ff)] {
        let mut lamp = standard(0x111111, 0.4, 0.0);
        lamp.emissive = Color::from_hex(hex);
        lamp.emissive_intensity = 3.0;
        let mut o = Object3D::mesh(Mesh::new(SphereGeometry::new(0.28, 32, 16), lamp.into()));
        o.position = Vector3::new(x, 0.28, z);
        s.add(o);
    }
    (s, knot)
}

fn main() {
    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .color_format(FORMAT)
        .build()
        .expect("a GPU adapter");
    let mut camera = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.1, 60.0);
    camera.position = Vector3::new(2.4, 2.6, 6.8);
    camera.look_at(Vector3::new(-0.2, 0.7, -0.6));
    let focus = (camera.position - Vector3::new(0.0, 1.0, 0.0)).length();

    let chains: Vec<(&str, Effect)> = vec![
        ("as rendered", Box::new(|_| unreachable!())),
        (
            "ssao",
            Box::new(|_| {
                let mut p = SsaoPass::new(0, 0, W, H, 32);
                p.kernel_radius = 2.0;
                p.min_distance = 0.002;
                p.max_distance = 0.2;
                Box::new(p)
            }),
        ),
        (
            "bloom",
            Box::new(|_| Box::new(UnrealBloomPass::new(W, H, 0.45, 0.2, 1.0))),
        ),
        (
            "depth of field",
            Box::new(move |_| Box::new(BokehPass::new(0, 0, focus, 0.03, 0.02))),
        ),
        (
            "outline",
            Box::new(|knot| {
                let mut p = OutlinePass::new(W, H, 0, 0, vec![knot]);
                p.visible_edge_color = Color::from_hex(0xffe08a);
                p.edge_strength = 6.0;
                p.edge_thickness = 2.0;
                Box::new(p)
            }),
        ),
        (
            "pixelated",
            Box::new(|_| Box::new(RenderPixelatedPass::new(4, 0, 0))),
        ),
    ];

    let cols = 3u32;
    let rows = (chains.len() as u32).div_ceil(cols);
    let mut sheet = vec![0u8; (W * cols * H * rows * 4) as usize];
    for (i, (name, effect)) in chains.iter().enumerate() {
        let (mut s, knot) = scene();
        let mut composer =
            EffectComposer::new(hr.device().clone(), hr.queue().clone(), W, H, FORMAT);
        // The pixel-art pass draws the scene itself, in place of RenderPass.
        if *name != "pixelated" {
            composer.add_pass(RenderPass::new());
        }
        if *name != "as rendered" {
            composer.add_boxed(effect(knot));
        }
        composer.add_pass(OutputPass::new());
        let out = RenderTarget::new(hr.device(), W, H, FORMAT);
        composer.render(hr.renderer(), &mut s, &camera, &out.color_view);
        let px = readback::texture_bytes(hr.device(), hr.queue(), &out.color_texture);
        let (cx, cy) = (i as u32 % cols, i as u32 / cols);
        for y in 0..H {
            let row = &px[(y * W * 4) as usize..((y + 1) * W * 4) as usize];
            let at = (((cy * H + y) * W * cols + cx * W) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
        println!("  {name}");
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write(
        "out/postfx_chain.png",
        encode_png(W * cols, H * rows, &sheet),
    )
    .unwrap();
    println!("wrote out/postfx_chain.png");
}
