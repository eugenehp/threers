//! Minimal repro: how far does a flat ground plane actually render?
use threers::{
    BufferAttribute, BufferGeometry, Color, DirectionalLight, HeadlessRenderer, Light, Material,
    Mesh, Object3D, PerspectiveCamera, Scene, StandardMaterial, Vector3,
};

fn slab(x0: f32, z0: f32, x1: f32, z1: f32, y: f32, c: Color) -> BufferGeometry {
    let mut g = BufferGeometry::new();
    g.set_attribute(
        "position",
        BufferAttribute::new(
            vec![x0, y, z0, x0, y, z1, x1, y, z1, x1, y, z0],
            3,
        ),
    );
    g.set_attribute(
        "normal",
        BufferAttribute::new([0.0, 1.0, 0.0].repeat(4), 3),
    );
    g.set_attribute(
        "uv",
        BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0], 2),
    );
    g.set_attribute(
        "color",
        BufferAttribute::new([c.r, c.g, c.b].repeat(4), 3),
    );
    g.set_index(vec![0, 1, 2, 0, 2, 3]);
    g
}

fn main() {
    let far: f32 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20000.0);
    let reach: f32 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20000.0);
    let (w, h) = (600u32, 400u32);
    let mut scene = Scene::new();
    scene.background = Color::new(0.1, 0.3, 0.9);
    // Stripes 200 m apart along z, so the cut distance can be read straight
    // off the image instead of inferred from a projection I keep getting wrong.
    let mut z = 0.0f32;
    let mut i = 0;
    while z > -reach {
        let c = if i % 2 == 0 {
            Color::new(0.2, 0.7, 0.2)
        } else {
            Color::new(0.9, 0.9, 0.2)
        };
        scene.add(Object3D::mesh(Mesh::new(
            slab(-reach, z - 200.0, reach, z, 0.0, c),
            Material::Standard(StandardMaterial::new(Color::WHITE).with_roughness(1.0)),
        )));
        z -= 200.0;
        i += 1;
    }
    let mut sun = Object3D::light(Light::Directional(DirectionalLight::new(
        Color::WHITE,
        2.0,
    )));
    sun.position = Vector3::new(0.0, 500.0, 200.0);
    scene.add(sun);
    let mut camera = PerspectiveCamera::new(42.0, w as f32 / h as f32, 1.0, far);
    camera.position = Vector3::new(0.0, 180.0, 0.0);
    camera.look_at(Vector3::new(0.0, 180.0, -1000.0));
    let mut r = HeadlessRenderer::builder().size(w, h).build().unwrap();
    let px = r.render_to_rgba_resolved(&mut scene, &camera);
    // Walk down the middle column from the horizon and report the first row
    // that is ground rather than background.
    let mut first = None;
    for y in 0..h {
        let o = ((y * w + w / 2) * 4) as usize;
        let (rr, gg, bb) = (px[o], px[o + 1], px[o + 2]);
        if !(bb > rr + 40 && bb > gg + 40) {
            first = Some(y);
            break;
        }
    }
    println!("far={far} reach={reach} first ground row={first:?} of {h}");
}
