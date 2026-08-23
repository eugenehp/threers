//! What BC1 actually costs on real imagery, measured rather than assumed.
//!
//! BC1 is eight to one, which is the difference between a whole planet at
//! native resolution fitting in memory and not. The doubt is where it fails:
//! four colours along one line per 4x4 block is generous for photographic
//! ground and mean for wide smooth gradients, and an equirectangular Earth is
//! half ocean — exactly the smooth gradient case. So this reports error over
//! the image AND over its smoothest and busiest regions separately, because a
//! single average hides precisely the failure being looked for.
//!
//!     cargo run --release --example bc1_quality -- web/assets/earth/leo.500m.png

use threers::textures::bc1;

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mut se = 0.0f64;
    let mut n = 0.0f64;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            let d = x[c] as f64 - y[c] as f64;
            se += d * d;
            n += 1.0;
        }
    }
    let mse = se / n.max(1.0);
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Mean absolute error and local variance for one tile, so smooth and busy
/// regions can be reported apart.
fn tile_stats(orig: &[u8], dec: &[u8], w: usize, x0: usize, y0: usize, n: usize) -> (f64, f64) {
    let (mut err, mut cnt) = (0.0f64, 0.0f64);
    let (mut sum, mut sum2) = (0.0f64, 0.0f64);
    for y in y0..y0 + n {
        for x in x0..x0 + n {
            let i = (y * w + x) * 4;
            let lum =
                0.2126 * orig[i] as f64 + 0.7152 * orig[i + 1] as f64 + 0.0722 * orig[i + 2] as f64;
            sum += lum;
            sum2 += lum * lum;
            for c in 0..3 {
                err += (orig[i + c] as f64 - dec[i + c] as f64).abs();
                cnt += 1.0;
            }
        }
    }
    let m = sum / (n * n) as f64;
    let var = (sum2 / (n * n) as f64 - m * m).max(0.0);
    (err / cnt, var.sqrt())
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "web/assets/earth/leo.500m.png".into());
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let img = threers::decode_png(&bytes).expect("decode png");
    let (w, h) = (img.width, img.height);
    println!("{path}: {w}x{h}");

    // A middle crop, big enough to be representative and small enough to be
    // quick. The whole image would take minutes and say the same thing.
    let n = 2048u32.min(w).min(h);
    let (x0, y0) = ((w - n) / 2, (h - n) / 2);
    let mut crop = Vec::with_capacity((n * n * 4) as usize);
    for y in 0..n {
        let src = (((y0 + y) * w + x0) * 4) as usize;
        crop.extend_from_slice(&img.rgba[src..src + (n * 4) as usize]);
    }

    let t = std::time::Instant::now();
    let enc = bc1::encode(&crop, n, n);
    let secs = t.elapsed().as_secs_f64();
    let dec = bc1::decode(&enc, n, n);

    let raw = (n as usize) * (n as usize) * 4;
    println!(
        "\n{n}x{n} crop: {:.1} MB -> {:.1} MB ({:.1}:1) in {secs:.2} s ({:.0} Mpx/s)",
        raw as f64 / 1e6,
        enc.len() as f64 / 1e6,
        raw as f64 / enc.len() as f64,
        (n as f64 * n as f64) / 1e6 / secs
    );
    println!("PSNR {:.2} dB overall", psnr(&crop, &dec));

    // Rank 64x64 tiles by local contrast, then report the extremes. The smooth
    // end is ocean and the busy end is coast; BC1 fails differently on each.
    let ts = 64usize;
    let mut tiles: Vec<(f64, f64)> = Vec::new();
    for ty in 0..(n as usize / ts) {
        for tx in 0..(n as usize / ts) {
            tiles.push(tile_stats(&crop, &dec, n as usize, tx * ts, ty * ts, ts));
        }
    }
    tiles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let q = tiles.len() / 10;
    let mean = |v: &[(f64, f64)]| v.iter().map(|t| t.0).sum::<f64>() / v.len().max(1) as f64;
    println!(
        "mean abs error: {:.2} over the smoothest tenth of tiles (ocean, sd {:.1}),\n\
         \x20               {:.2} over the busiest tenth (coast and cloud, sd {:.1})",
        mean(&tiles[..q]),
        tiles[q / 2].1,
        mean(&tiles[tiles.len() - q..]),
        tiles[tiles.len() - q / 2 - 1].1,
    );
    let worst = tiles.iter().map(|t| t.0).fold(0.0f64, f64::max);
    println!("worst tile: {worst:.2} levels mean abs error");
}
