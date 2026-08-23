//! What the sky sphere does to point sources.
//!
//! ```text
//! cargo run --release --features planet --example starfield_probe
//! ```
//!
//! A star is a point source: its flux is fixed, and a camera concentrates that
//! flux into about one pixel however wide the field of view is. So zooming out
//! should not dim a star — it should pack more stars into the frame while each
//! stays a bright point. Sampling a minified equirectangular map does the
//! opposite, and does it differently depending on whether the map has a mip
//! chain, which is why this measures both paths:
//!
//! - **peak** — the brightest pixel. Falls with field of view if the filter is
//!   spreading each star's flux over its neighbours.
//! - **stars** — pixels above a visibility threshold. The number in frame
//!   should *rise* as the field widens, not collapse.
//! - **shimmer** — RMS change between two renders a fifth of a pixel apart.
//!   Real stars slide smoothly; undersampled ones pop in and out, and that
//!   reads as twinkling, which is wrong in vacuum.

use threers::planet::{EarthTextures, Starfield};
use threers::{
    encode_png, Color, HeadlessRenderer, PerspectiveCamera, Scene, ToneMapping, Vector3,
};

const W: u32 = 800;
const H: u32 = 800;

fn arg(flag: &str) -> Option<String> {
    std::env::args()
        .position(|a| a == flag)
        .and_then(|i| std::env::args().nth(i + 1))
}

fn stats(rgba: &[u8]) -> (f32, f32, usize) {
    let mut peak = 0.0f32;
    let mut sum = 0.0f64;
    let mut lit = 0usize;
    for px in rgba.chunks_exact(4) {
        let l = 0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32;
        peak = peak.max(l);
        sum += l as f64;
        // 16/255 is about where a star stops reading as a star on a dark sky.
        if l >= 16.0 {
            lit += 1;
        }
    }
    (peak, (sum / (rgba.len() / 4) as f64) as f32, lit)
}

/// Elongation of the bright blobs in a pole-centred view.
///
/// For each local maximum, the intensity-weighted second moments of its
/// neighbourhood give an ellipse; the ratio of its axes is how stretched the
/// star is, and the angle of the major axis against the direction away from the
/// pole says which way. Radial elongation (0 degrees) is the pinwheel; a
/// tangential one (90) is a swirl.
fn elongation(img: &[u8], r0: f32, r1: f32) -> (f32, f32, usize) {
    let (cx, cy) = (W as f32 / 2.0, H as f32 / 2.0);
    let at = |x: i32, y: i32| -> f32 {
        if x < 0 || y < 0 || x >= W as i32 || y >= H as i32 {
            return 0.0;
        }
        let o = ((y as u32 * W + x as u32) * 4) as usize;
        0.2126 * img[o] as f32 + 0.7152 * img[o + 1] as f32 + 0.0722 * img[o + 2] as f32
    };
    let (mut ratio, mut angle, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in 3..H as i32 - 3 {
        for x in 3..W as i32 - 3 {
            let v = at(x, y);
            if v < 40.0 {
                continue;
            }
            let dx0 = x as f32 - cx;
            let dy0 = y as f32 - cy;
            let rad = (dx0 * dx0 + dy0 * dy0).sqrt();
            if rad < r0 || rad > r1 {
                continue;
            }
            // Local maximum only, so each star counts once.
            let mut peak = true;
            for j in -2..=2i32 {
                for i in -2..=2i32 {
                    if (i != 0 || j != 0) && at(x + i, y + j) > v {
                        peak = false
                    }
                }
            }
            if !peak {
                continue;
            }
            let (mut m00, mut m20, mut m02, mut m11) = (0.0f32, 0.0, 0.0, 0.0);
            for j in -3..=3i32 {
                for i in -3..=3i32 {
                    let w = at(x + i, y + j);
                    m00 += w;
                    m20 += w * i as f32 * i as f32;
                    m02 += w * j as f32 * j as f32;
                    m11 += w * i as f32 * j as f32;
                }
            }
            if m00 <= 0.0 {
                continue;
            }
            let (a, b, c) = (m20 / m00, m02 / m00, m11 / m00);
            let tr = a + b;
            let det = ((a - b) * (a - b) + 4.0 * c * c).sqrt();
            let (l1, l2) = ((tr + det) * 0.5, (tr - det) * 0.5);
            if l2 <= 1e-6 {
                continue;
            }
            // Major-axis direction against the outward radial direction.
            let theta = 0.5 * (2.0 * c).atan2(a - b);
            let radial = dy0.atan2(dx0);
            let mut d = (theta - radial).abs() % std::f32::consts::PI;
            if d > std::f32::consts::FRAC_PI_2 {
                d = std::f32::consts::PI - d
            }
            ratio += (l1 / l2).sqrt() as f64;
            angle += d.to_degrees() as f64;
            n += 1;
        }
    }
    if n == 0 {
        return (1.0, 0.0, 0);
    }
    ((ratio / n as f64) as f32, (angle / n as f64) as f32, n)
}

fn rms(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0f64;
    for i in (0..n).step_by(4) {
        let d = a[i] as f64 - b[i] as f64;
        acc += d * d;
    }
    (acc / (n / 4) as f64).sqrt() as f32
}

fn main() {
    let assets = arg("--assets").unwrap_or_else(|| "web/assets/earth".into());
    let mut r = HeadlessRenderer::builder()
        .size(W, H)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .expect("no GPU adapter");
    r.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    std::fs::create_dir_all("out").ok();

    let t0 = std::time::Instant::now();
    let (nasa, report) = EarthTextures::from_dir(&assets).load_starfield_reporting();
    eprintln!(
        "sky: {:?} {}x{} decoded in {:.1}s",
        report.loaded,
        nasa.width,
        nasa.height,
        t0.elapsed().as_secs_f64()
    );
    let t1 = std::time::Instant::now();
    let chain = threers::renderer::gpu_texture::mip_chain_for(&nasa);
    eprintln!(
        "     {} mip levels built in {:.1}s",
        chain,
        t1.elapsed().as_secs_f64()
    );
    let pw = arg("--procedural")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let t2 = std::time::Instant::now();
    let cube = Starfield::procedural(pw).faceted(0);
    eprintln!(
        "     cube faces built in {:.1}s",
        t2.elapsed().as_secs_f64()
    );
    let skies: Vec<(&str, Starfield)> = vec![
        ("nasa", Starfield::new(nasa)),
        ("procedural", Starfield::procedural(pw)),
        ("cube", cube),
    ];

    for (name, sky) in &skies {
        eprintln!(
            "--- {name} ({}x{}) ---",
            sky.texture.width, sky.texture.height
        );
        eprintln!(
            "{:>6}  {:>7}  {:>7}  {:>7}  {:>8}",
            "fov", "peak", "mean", "stars", "shimmer"
        );
        for fov in [4.0f32, 10.0, 25.0, 60.0, 100.0] {
            let mut render = |yaw: f32| {
                let mut scene = Scene::new();
                // Space is black; without this an alpha bug in the sky map shows
                // up as the default background rather than as itself.
                scene.background = Color::BLACK;
                sky.add_to(&mut scene);
                let mut c = PerspectiveCamera::new(fov, 1.0, 0.1, 1000.0);
                c.position = Vector3::ZERO;
                c.look_at(Vector3::new(yaw.sin(), 0.12, yaw.cos()));
                r.render_to_rgba(&mut scene, &c)
            };
            let a = render(0.0);
            // A fifth of a pixel of yaw: far too small to change what is in
            // frame, so anything it changes is the filter, not the scene.
            let b = render(fov.to_radians() / W as f32 * 0.2);
            let (peak, mean, lit) = stats(&a);
            eprintln!(
                "{fov:>6.0}  {peak:>7.1}  {mean:>7.3}  {lit:>7}  {:>8.3}",
                rms(&a, &b)
            );
            if fov == 25.0 {
                std::fs::write(format!("out/sky_{name}.png"), encode_png(W, H, &a)).unwrap();
            }
            if fov == 60.0 {
                // Straight up the map's pole, where the equirectangular grid
                // converges — a star smeared along longitude there comes out as
                // an arc around the pole, and a skyful of them as a vortex.
                let mut scene = Scene::new();
                scene.background = Color::BLACK;
                sky.add_to(&mut scene);
                let mut c = PerspectiveCamera::new(fov, 1.0, 0.1, 1000.0);
                c.position = Vector3::ZERO;
                c.up = Vector3::new(0.0, 0.0, -1.0);
                c.look_at(Vector3::new(0.0, 1.0, 0.0));
                let pole = r.render_to_rgba(&mut scene, &c);
                for (lo, hi) in [(20.0f32, 90.0), (90.0, 200.0), (200.0, 380.0)] {
                    let (ratio, deg, n) = elongation(&pole, lo, hi);
                    eprintln!(
                        "  {name} pole r={lo:.0}..{hi:.0}: axis ratio {ratio:.2}, \
                         {deg:.0} deg off radial, {n} stars"
                    );
                }
                std::fs::write(format!("out/sky_{name}_pole.png"), encode_png(W, H, &pole))
                    .unwrap();
            }
            if fov == 60.0 {
                // Toward the galactic centre, where the bulge and the dust
                // lanes are — the rest of the sky says nothing about those.
                let mut scene = Scene::new();
                scene.background = Color::BLACK;
                sky.add_to(&mut scene);
                let mut c = PerspectiveCamera::new(fov, 1.0, 0.1, 1000.0);
                c.position = Vector3::ZERO;
                c.look_at(Vector3::new(-0.055, -0.874, -0.483));
                let g = r.render_to_rgba(&mut scene, &c);
                std::fs::write(format!("out/sky_{name}_galaxy.png"), encode_png(W, H, &g)).unwrap();
            }
        }
    }
    // Sphere against cube, same camera: a wrong face basis or a flipped `v`
    // would scramble the sky without moving any of the statistics above, so
    // this compares them directly.
    {
        let sphere = Starfield::procedural(pw);
        let cube = Starfield::procedural(pw).faceted(0);
        let mut shot = |sky: &Starfield, yaw: f32, pitch: f32| {
            let mut scene = Scene::new();
            scene.background = Color::BLACK;
            sky.add_to(&mut scene);
            let mut c = PerspectiveCamera::new(50.0, 1.0, 0.1, 1000.0);
            c.position = Vector3::ZERO;
            c.look_at(Vector3::new(
                yaw.sin() * pitch.cos(),
                pitch.sin(),
                yaw.cos() * pitch.cos(),
            ));
            r.render_to_rgba(&mut scene, &c)
        };
        let mut worst = 0.0f64;
        use std::f32::consts::{FRAC_PI_2, PI};
        for (yaw, pitch, name) in [
            (0.0f32, 0.0f32, "front"),
            (FRAC_PI_2, 0.0, "right"),
            (PI, 0.0, "back"),
            (0.0, 1.35, "up"),
            (0.0, -1.35, "down"),
        ] {
            let a = shot(&sphere, yaw, pitch);
            let b = shot(&cube, yaw, pitch);
            // Mean absolute difference, and the correlation of the two — a
            // rotated or mirrored sky would still have a similar histogram but
            // would not correlate.
            let n = (a.len() / 4) as f64;
            let (mut sa, mut sb) = (0.0f64, 0.0f64);
            for i in (0..a.len()).step_by(4) {
                sa += a[i] as f64;
                sb += b[i] as f64;
            }
            let (ma, mb) = (sa / n, sb / n);
            let (mut cov, mut va, mut vb) = (0.0f64, 0.0f64, 0.0f64);
            for i in (0..a.len()).step_by(4) {
                let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
                cov += da * db;
                va += da * da;
                vb += db * db;
            }
            let corr = cov / (va.sqrt() * vb.sqrt()).max(1e-9);
            eprintln!("  {name:>5}: sphere mean {ma:.2}, cube mean {mb:.2}, correlation {corr:.3}");
            worst = worst.max(1.0 - corr);
        }
        eprintln!("  worst decorrelation: {worst:.3}");
        // Straight at a cube corner, where three faces meet: if the face
        // textures disagree at their edges it shows as a line, and a corner is
        // the worst case because two of them cross there.
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        cube.add_to(&mut scene);
        let mut c = PerspectiveCamera::new(70.0, 1.0, 0.1, 1000.0);
        c.position = Vector3::ZERO;
        c.look_at(Vector3::new(0.577, 0.577, 0.577));
        let img = r.render_to_rgba(&mut scene, &c);
        std::fs::write("out/sky_cube_corner.png", encode_png(W, H, &img)).unwrap();
        // A seam is a *column or row* of pixels that differs from both its
        // neighbours, so compare each column mean against the average of the
        // two beside it and take the worst.
        let col_mean = |x: u32| -> f64 {
            (0..H)
                .map(|y| img[((y * W + x) * 4) as usize] as f64)
                .sum::<f64>()
                / H as f64
        };
        let means: Vec<f64> = (0..W).map(col_mean).collect();
        let mut worst_col = 0.0f64;
        for x in 1..W as usize - 1 {
            worst_col = worst_col.max((means[x] - (means[x - 1] + means[x + 1]) * 0.5).abs());
        }
        eprintln!("  worst column step at a corner: {worst_col:.3} / 255");
    }
    eprintln!("wrote out/sky_*.png");
}
