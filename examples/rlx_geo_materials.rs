//! Voronoi cells as material maps (`--features rlx-geo`).
//!
//! ```text
//! cargo run --release --features rlx-geo --example rlx_geo_materials
//! ```
//!
//! Writes the three maps it builds — `out/rlx_geo_mat_{albedo,roughness,normal}.png`
//! — and `out/rlx_geo_materials.png`, a sphere and a slab wearing them.
//!
//! Cell noise is one of the few procedural textures that is genuinely
//! *geometric*: cracked mud, dried paint, leather, stained glass, reptile skin
//! are all "which cell am I in, and how far into it". `rlx-geo` answers the
//! first with [`geo::voronoi_labels`], and the second twice over — distance to
//! the site ([`geo::voronoi_distance_field`], which peaks at the seams) and
//! distance to the wall ([`geo::voronoi_wall_distance`], which peaks in the
//! middle). The maps fall out of those three, and picking the wrong one of the
//! two distances is the difference between a plate and a pit.
//!
//! No rlx graph runtime is involved: this is the `rlx-geo` feature on its own.

use std::sync::Arc;

use threers::rlx::geo;
use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, PlaneGeometry, Quaternion, Scene, SphereGeometry, StandardMaterial, Texture,
    TextureFormat, TextureWrap, Vector2, Vector3,
};

const MAP: u32 = 512;
const CELLS: usize = 260;

fn main() {
    std::fs::create_dir_all("out").ok();

    // --- The cells --------------------------------------------------------
    let sites = poisson_ish(CELLS, MAP, MAP);
    let labels = geo::voronoi_labels(&sites, MAP, MAP);
    let distance = geo::voronoi_distance_field(&sites, MAP, MAP);
    let to_wall = geo::voronoi_wall_distance(&sites, MAP, MAP);
    let walls = geo::voronoi_edges(&sites, MAP, MAP);

    // Albedo: each cell a slightly different clay, darkened towards its wall.
    // The distance field is what makes the shading *inside* a cell possible —
    // a label map alone gives flat plates.
    let far = distance.iter().copied().fold(0.0f32, f32::max).max(1.0);
    let mut albedo = Vec::with_capacity((MAP * MAP * 4) as usize);
    for i in 0..(MAP * MAP) as usize {
        let cell = labels[i] as usize;
        let tint = 0.75 + 0.25 * hash01(cell as u64);
        let depth = (distance[i] / far * 3.0).min(1.0);
        let base = [0.62, 0.42, 0.30];
        for channel in base {
            let v = channel * tint * (0.55 + 0.45 * depth);
            albedo.push(quantize(v));
        }
        albedo.push(255);
    }
    write("albedo", &albedo);

    // Roughness: walls are rough, cell interiors are polished. Linear, not
    // sRGB — roughness is a material parameter, not a colour.
    let roughness: Vec<u8> = (0..(MAP * MAP) as usize)
        .flat_map(|i| {
            let r = 0.35 + 0.55 * walls[i] + 0.1 * (1.0 - (distance[i] / far * 4.0).min(1.0));
            let v = quantize_linear(r);
            [v, v, v, 255]
        })
        .collect();
    write("roughness", &roughness);

    // Height from the distance to the *wall*, not to the site: mud plates are
    // domed in the middle and sink at the seams, which is this quantity and
    // the inverse of the other one.
    let plate = to_wall.iter().copied().fold(0.0f32, f32::max).max(1.0);
    let height: Vec<f32> = to_wall
        .iter()
        .map(|d| (d / plate * 3.5).min(1.0).powf(0.5))
        .collect();
    let normal = geo::normal_map_from_height(&height, MAP, MAP, 26.0);
    write("normal", &normal.data);

    // --- Wear them --------------------------------------------------------
    let mut headless = match HeadlessRenderer::builder().size(900, 600).build() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — the maps were still written.");
            std::process::exit(2);
        }
    };
    let (width, height_px) = headless.render_size();

    let mut material = StandardMaterial::new(Color::WHITE);
    material.map = Some(Arc::new(tiled(Texture::new(
        MAP,
        MAP,
        TextureFormat::Rgba8UnormSrgb,
        albedo,
    ))));
    material.roughness_map = Some(Arc::new(tiled(Texture::new(
        MAP,
        MAP,
        TextureFormat::Rgba8Unorm,
        roughness,
    ))));
    material.normal_map = Some(Arc::new(tiled(normal)));
    material.normal_scale = Vector2::new(1.0, 1.0);
    material.metalness = 0.0;
    material.roughness = 1.0;

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x10141c);
    scene.add_light(AmbientLight::new(Color::from_hex(0x5a6a80), 0.5));
    scene.add_light(
        DirectionalLight::new(Color::from_hex(0xfff0dd), 3.0)
            .with_direction(Vector3::new(-0.55, -0.7, -0.45).normalize()),
    );

    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 96, 48),
        material.clone().into(),
    ));
    ball.position = Vector3::new(-1.35, 0.0, 0.0);
    scene.add(ball);

    let mut slab = Object3D::mesh(Mesh::new(PlaneGeometry::new(2.6, 2.6), material.into()));
    slab.position = Vector3::new(1.25, 0.0, 0.0);
    slab.quaternion = Quaternion::from_axis_angle(Vector3::UP, -0.45);
    scene.add(slab);

    let mut camera = PerspectiveCamera::new(42.0, width as f32 / height_px as f32, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.7, 5.2);
    camera.look_at(Vector3::new(0.0, -0.05, 0.0));

    let rgba = headless.render_to_rgba(&mut scene, &camera);
    std::fs::write(
        "out/rlx_geo_materials.png",
        encode_png(width, height_px, &rgba),
    )
    .expect("write png");
    println!("wrote out/rlx_geo_materials.png ({width}x{height_px})");
}

/// Sites spread by dart-throwing with a minimum separation, so the cells come
/// out roughly even. Pure random sites give a few enormous cells and a lot of
/// slivers, which reads as noise rather than as a surface.
fn poisson_ish(count: usize, width: u32, height: u32) -> Vec<Vector2> {
    let min_gap = 0.7 * (width as f32 * height as f32 / count as f32).sqrt();
    let mut sites: Vec<Vector2> = Vec::with_capacity(count);
    let mut rng = Lcg(0x5eed_1234);
    let mut attempts = 0;
    while sites.len() < count && attempts < count * 200 {
        attempts += 1;
        let candidate = Vector2::new(rng.unit() * width as f32, rng.unit() * height as f32);
        if sites
            .iter()
            .all(|s| (s.x - candidate.x).hypot(s.y - candidate.y) >= min_gap)
        {
            sites.push(candidate);
        }
    }
    sites
}

/// Repeat rather than clamp: these maps go onto a sphere and a slab, and a
/// clamped edge would smear the border cell across half the object.
fn tiled(mut texture: Texture) -> Texture {
    texture.wrap_s = TextureWrap::Repeat;
    texture.wrap_t = TextureWrap::Repeat;
    texture
}

fn write(name: &str, rgba: &[u8]) {
    let path = format!("out/rlx_geo_mat_{name}.png");
    std::fs::write(&path, encode_png(MAP, MAP, rgba)).expect("write png");
    println!("wrote {path}");
}

fn quantize(linear: f32) -> u8 {
    let encoded = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn quantize_linear(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn hash01(v: u64) -> f32 {
    let x = v.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    ((x >> 40) as f32) / (1u32 << 24) as f32
}

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as f32 / u32::MAX as f32
    }
}
