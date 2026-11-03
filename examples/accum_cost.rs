use std::time::Instant;
use threers::{
    AmbientLight, BoxGeometry, Color, HeadlessRenderer, Mesh, Object3D, PerspectiveCamera,
    PhysicalMaterial, Scene, Vector3,
};

fn bench(meshes: usize, ss: u32, msaa: u32, n: u32) {
    let mut r = HeadlessRenderer::builder()
        .size(1080, 1920).supersample(ss).taa(true)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build().unwrap();
    r.set_msaa(msaa);
    r.renderer().set_taa_converge(true);
    let mut s = Scene::new();
    s.background = Color::from_hex(0x101820);
    s.add_light(AmbientLight::new(Color::WHITE, 0.6));
    for i in 0..meshes {
        let a = i as f32 * 0.41;
        let mut m = PhysicalMaterial::new(Color::from_hex(0xd8dee9));
        m.metalness = 0.9; m.roughness = 0.25; m.clearcoat = 0.5;
        let mut o = Object3D::mesh(Mesh::new(BoxGeometry::new(0.4,0.4,0.4), m.into()));
        o.position = Vector3::new(a.cos()*1.5, (a*1.7).sin()*1.2, a.sin()*1.5);
        s.add(o);
    }
    let mut c = PerspectiveCamera::new(45.0, 1080.0/1920.0, 0.1, 100.0);
    c.position = Vector3::new(0.0,0.0,5.0); c.look_at(Vector3::ZERO);
    s.update_world();
    r.render(&mut s, &c);
    let t = Instant::now();
    for _ in 0..n { r.render(&mut s, &c); }
    let _ = r.read_rgba();
    println!("  {meshes:>3} meshes, ss {ss}, msaa {msaa}: {:6.2} ms per render",
             t.elapsed().as_secs_f32()*1000.0/n as f32);
}

fn main() {
    println!("per-render cost against scene size (1080x1920, converging):");
    for m in [0usize, 8, 32, 64] { bench(m, 1, 1, 50); }
    println!("and against pixels, at 32 meshes:");
    bench(32, 1, 1, 50);
    bench(32, 1, 4, 50);
    bench(32, 2, 4, 50);
}
