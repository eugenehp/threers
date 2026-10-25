// Shared by `venom_button.rs` and `venom_gallery.rs`: a ghost hand for
// autoplay, scripted hands, headless output and argument parsing.
// `include!`d, so it sees the including file's `use`s.

/// The ghost hand for autoplay: a closed Catmull-Rom loop round the stage, in
/// fractions of it, with seconds per leg.
const GHOST: [[f32; 3]; 12] = [
    [0.5, 0.8, 1.1], [0.36, 0.74, 1.0], [0.6, 0.82, 0.9], [0.8, 0.64, 0.55], [0.98, 0.97, 1.4], [0.86, 0.28, 1.1],
    [0.7, 0.18, 1.0], [0.5, 0.12, 1.0], [0.3, 0.2, 0.55], [0.03, 0.04, 1.4], [0.16, 0.58, 1.1], [0.32, 0.86, 1.0],
];

fn ghost_at(t: f32, w: f32, h: f32) -> Vector2 {
    let total: f32 = GHOST.iter().map(|p| p[2]).sum();
    let mut u = t.rem_euclid(total);
    let mut i = 0;
    while u > GHOST[i][2] {
        u -= GHOST[i][2];
        i += 1;
    }
    let f = u / GHOST[i][2];
    let n = GHOST.len();
    let p = |k: usize| GHOST[k % n];
    let (p0, p1, p2, p3) = (p(i + n - 1), p(i), p(i + 1), p(i + 2));
    let e = f * f * (3.0 - 2.0 * f);
    let cr = |a: f32, b: f32, c: f32, d: f32| {
        0.5 * (2.0 * b + (-a + c) * e + (2.0 * a - 5.0 * b + 4.0 * c - d) * e * e + (-a + 3.0 * b - 3.0 * c + d) * e * e * e)
    };
    Vector2::new(cr(p0[0], p1[0], p2[0], p3[0]) * w, cr(p0[1], p1[1], p2[1], p3[1]) * h)
}

/// A hand moving through `keys` — (seconds, position) — eased between them.
fn scripted(keys: &[(f32, Vector2)], t: f32) -> Vector2 {
    let i = keys.iter().rposition(|k| k.0 <= t).unwrap_or(0).min(keys.len().saturating_sub(2));
    let (t0, a) = keys[i];
    let (t1, b) = keys[(i + 1).min(keys.len() - 1)];
    let u = if t1 > t0 { ((t - t0) / (t1 - t0)).clamp(0.0, 1.0) } else { 1.0 };
    a + (b - a) * (u * u * (3.0 - 2.0 * u))
}

/// Straight-alpha sRGB overlay over an opaque sRGB frame, mixed in linear
/// light as the windowed path's blend does.
fn composite(frame: &mut [u8], overlay: &[u8]) {
    let dec = |b: u8| {
        let s = b as f32 / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    let enc = |l: f32| {
        let l = l.clamp(0.0, 1.0);
        let s = if l <= 0.0031308 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0 + 0.5) as u8
    };
    for (d, s) in frame.chunks_exact_mut(4).zip(overlay.chunks_exact(4)) {
        let a = s[3] as f32 / 255.0;
        if a <= 0.0 {
            continue;
        }
        for c in 0..3 {
            d[c] = enc(dec(s[c]) * a + dec(d[c]) * (1.0 - a));
        }
        d[3] = 255;
    }
}

fn write_png(path: &str, width: u32, height: u32, rgba: &[u8]) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(path, threers::utils::png::encode_png(width, height, rgba)).expect("write png");
    println!("{path}");
}

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}
