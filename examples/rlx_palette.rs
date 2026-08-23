//! Palette extraction, and a mosaic built from it (`--features rlx,rlx-geo`).
//!
//! ```text
//! cargo run --release --features "rlx,rlx-geo" --example rlx_palette
//! ```
//!
//! Writes `out/rlx_palette_{source,swatches,posterized,mosaic}.png`.
//!
//! k-means over the frame's pixels, with rlx's `vector_quantize` doing the
//! assignment step — the quarter-million nearest-code lookups — and the host
//! doing the twelve-accumulator update. Then the palette is put to work twice:
//! flat, by snapping every pixel to its nearest entry, and as stained glass,
//! by averaging each Voronoi cell of the image and snapping *that*.
//!
//! The mosaic is where the two features meet. `rlx-geo` says which pixels
//! belong to which cell; rlx says which colour each cell should be.

use threers::rlx::{geo, preferred_device, Palette, PaletteOptions};
use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, Scene, SphereGeometry, StandardMaterial, Vector2, Vector3,
};

const SIZE: u32 = 600;
const COLORS: usize = 10;
const CELLS: usize = 1_400;

fn main() {
    std::fs::create_dir_all("out").ok();
    let device = preferred_device();

    let mut headless = match HeadlessRenderer::builder().size(SIZE, SIZE).build() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter.");
            std::process::exit(2);
        }
    };
    let (width, height) = headless.render_size();
    let frame = render(&mut headless, width, height);
    write("source", width, height, &frame);

    // --- The palette ------------------------------------------------------
    let palette =
        Palette::extract(&frame, COLORS, &PaletteOptions::default(), device).expect("extract");
    println!("{} colours from {width}×{height} pixels:", palette.len());
    for c in &palette.colors {
        let hex = to_hex(*c);
        println!("  #{hex}");
    }
    let strip_height = 48;
    write(
        "swatches",
        width,
        strip_height,
        &swatch_strip(&palette, width, strip_height),
    );

    // --- Flat ------------------------------------------------------------
    let posterized = palette.posterize(&frame, device).expect("posterize");
    write("posterized", width, height, &posterized);

    // --- Stained glass ----------------------------------------------------
    let sites = jittered_sites(CELLS, width, height);
    let labels = geo::voronoi_labels(&sites, width, height);
    let mosaic = mosaic(&frame, &labels, sites.len(), &palette);
    write("mosaic", width, height, &mosaic);
}

/// Average each Voronoi cell's colour, then snap the average to the palette.
///
/// Averaging happens in linear light — the cells are a *mixture* of the light
/// under them, and mixing sRGB-encoded numbers is mixing the wrong quantity.
/// Snapping happens in the palette's own space, which is where its entries
/// were spaced out.
fn mosaic(frame: &[u8], labels: &[u32], cells: usize, palette: &Palette) -> Vec<u8> {
    let mut sums = vec![[0.0f64; 3]; cells];
    let mut counts = vec![0u32; cells];
    for (px, label) in frame.chunks_exact(4).zip(labels) {
        let cell = *label as usize;
        if cell >= cells {
            continue;
        }
        for c in 0..3 {
            let v = px[c] as f32 / 255.0;
            sums[cell][c] += srgb_to_linear(v) as f64;
        }
        counts[cell] += 1;
    }

    let cell_color: Vec<Color> = (0..cells)
        .map(|c| {
            if counts[c] == 0 {
                return Color::BLACK;
            }
            let n = counts[c] as f64;
            let average = Color::new(
                (sums[c][0] / n) as f32,
                (sums[c][1] / n) as f32,
                (sums[c][2] / n) as f32,
            );
            match palette.nearest(average) {
                Some(i) => palette.colors[i],
                None => average,
            }
        })
        .collect();

    let mut out = Vec::with_capacity(frame.len());
    for (px, label) in frame.chunks_exact(4).zip(labels) {
        let c = cell_color
            .get(*label as usize)
            .copied()
            .unwrap_or(Color::BLACK);
        for v in [c.r, c.g, c.b] {
            out.push((linear_to_srgb(v).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        out.push(px[3]);
    }
    out
}

/// Cell centres on a jittered grid: a plain grid gives square cells and no
/// reason to have used a Voronoi diagram at all.
fn jittered_sites(count: usize, width: u32, height: u32) -> Vec<Vector2> {
    let columns = (count as f32).sqrt().ceil() as usize;
    let rows = count.div_ceil(columns);
    let (dx, dy) = (width as f32 / columns as f32, height as f32 / rows as f32);
    let mut rng = Lcg(0xa11ce);
    let mut sites = Vec::with_capacity(columns * rows);
    for r in 0..rows {
        for c in 0..columns {
            sites.push(Vector2::new(
                (c as f32 + 0.15 + 0.7 * rng.unit()) * dx,
                (r as f32 + 0.15 + 0.7 * rng.unit()) * dy,
            ));
        }
    }
    sites
}

/// The palette as a strip of equal bands.
fn swatch_strip(palette: &Palette, width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..height {
        for x in 0..width {
            let slot = (x as usize * palette.len().max(1) / width as usize)
                .min(palette.len().saturating_sub(1));
            let c = palette.colors.get(slot).copied().unwrap_or(Color::BLACK);
            for v in [c.r, c.g, c.b] {
                out.push((linear_to_srgb(v).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
            out.push(255);
        }
    }
    out
}

fn to_hex(c: Color) -> String {
    let b = |v: f32| (linear_to_srgb(v).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    format!("{:02x}{:02x}{:02x}", b(c.r), b(c.g), b(c.b))
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Several differently coloured balls, so the palette has something to find.
fn render(headless: &mut HeadlessRenderer, width: u32, height: u32) -> Vec<u8> {
    let mut scene = Scene::new();
    scene.background = Color::new(0.05, 0.06, 0.09);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.22));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 1.8)
            .with_direction(Vector3::new(-0.4, -0.7, -0.55).normalize()),
    );

    // `from_hex` and not `new`: a `Color` is linear, so writing 0.85 for red
    // asks for the *light*, which displays as a pale salmon once it is
    // encoded. Hex goes through the sRGB decode, so these are the colours they
    // look like — which matters here, because a washed-out scene yields a
    // washed-out palette and the example would be demonstrating nothing.
    let balls = [
        (Vector3::new(-1.5, 0.6, 0.0), Color::from_hex(0xd93a2b)),
        (Vector3::new(0.0, 0.9, -0.6), Color::from_hex(0x2f7fd9)),
        (Vector3::new(1.5, 0.5, 0.1), Color::from_hex(0xe8b32a)),
        (Vector3::new(-0.8, -0.8, 0.5), Color::from_hex(0x2fa34e)),
        (Vector3::new(0.9, -0.9, 0.4), Color::from_hex(0x8a4fc4)),
    ];
    for (position, color) in balls {
        let mut material = StandardMaterial::new(color);
        material.roughness = 0.4;
        material.metalness = 0.1;
        let mut ball = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.65, 48, 24),
            material.into(),
        ));
        ball.position = position;
        scene.add(ball);
    }

    let mut camera = PerspectiveCamera::new(45.0, width as f32 / height as f32, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.2, 5.4);
    camera.look_at(Vector3::ZERO);
    headless.render_to_rgba(&mut scene, &camera)
}

fn write(name: &str, width: u32, height: u32, rgba: &[u8]) {
    let path = format!("out/rlx_palette_{name}.png");
    std::fs::write(&path, encode_png(width, height, rgba)).expect("write png");
    println!("wrote {path}");
}

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as f32 / u32::MAX as f32
    }
}
