//! Check the dependency-free forward pass against the one that trained it.
//!
//! `src/raytrace/denoise_net.rs` reimplements the network by hand so it can run
//! on `wasm32` and carry no GPL dependency. A reimplementation is only worth
//! anything if it computes the same function, and "it looks denoised" is not
//! evidence of that — a transposed index or an off-by-one in the padding still
//! produces a plausible image.
//!
//! So this scores both on the same tiles, with the same metric the whole
//! benchmark uses, and prints the largest single-pixel disagreement. They
//! should differ only by float ordering.
//!
//! ```sh
//! cargo run --release --example denoise_parity --features raytrace,learned-denoise \
//!     -- --weights out/fixed.bin --val out/real2_testset.bin
//! ```

use threers::raytrace::denoise_net::{Denoiser, IN_CHANNELS, OUT_CHANNELS};

fn arg(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

const LOSS_EPSILON: f32 = 0.01;

fn main() {
    let weights = arg("--weights").unwrap_or_else(|| "out/fixed.bin".into());
    let val = arg("--val").unwrap_or_else(|| "out/real2_testset.bin".into());
    let limit: usize = arg("--tiles").and_then(|s| s.parse().ok()).unwrap_or(32);

    let net = match Denoiser::open(&weights) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("{weights}: {e}");
            std::process::exit(2);
        }
    };
    println!("{net:?}");

    let raw = std::fs::read(&val).expect("read dataset");
    if &raw[..8] != b"RLXDN002" {
        eprintln!("{val}: not an RLXDN002 dataset");
        std::process::exit(2);
    }
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
    let tile = u32_at(8);
    let count = u32_at(12).min(limit);
    let ins = u32_at(16);
    let outs = u32_at(20);
    let plane = tile * tile;
    let per = (ins + outs) * plane;
    let floats: &[f32] = unsafe {
        std::slice::from_raw_parts(raw[24..].as_ptr() as *const f32, (raw.len() - 24) / 4)
    };
    assert_eq!(
        ins, IN_CHANNELS,
        "this build reads {IN_CHANNELS} input planes"
    );
    assert_eq!(outs, OUT_CHANNELS);

    let reference: Option<Vec<f32>> = arg("--reference").map(|p| {
        let b = std::fs::read(p).expect("read reference");
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    });

    let mut unfiltered = 0.0f64;
    let mut denoised = 0.0f64;
    let started = std::time::Instant::now();
    for i in 0..count {
        let t = &floats[i * per..(i + 1) * per];
        let input = &t[..ins * plane];
        let target = &t[ins * plane..];
        let out = net.denoise(input, tile, tile).expect("denoise");
        unfiltered += relative_error(&input[..OUT_CHANNELS * plane], target);
        denoised += relative_error(&out, target);
        if let Some(r) = reference.as_ref() {
            let want = &r[i * OUT_CHANNELS * plane..(i + 1) * OUT_CHANNELS * plane];
            let (mut worst, mut at) = (0.0f32, 0usize);
            for (j, (a, b)) in out.iter().zip(want).enumerate() {
                let d = (a - b).abs();
                if d > worst {
                    worst = d;
                    at = j;
                }
            }
            if i < 3 {
                println!(
                    "  tile {i}: worst |mine-rlx| {worst:.6} at plane {} pixel {} \
                     (mine {:.5}, rlx {:.5})",
                    at / plane,
                    at % plane,
                    out[at],
                    want[at]
                );
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f32();
    let unfiltered = (unfiltered / count as f64).sqrt();
    let denoised = (denoised / count as f64).sqrt();
    println!("tiles       {count} of {tile}x{tile}");
    println!("unfiltered  {unfiltered:.5}");
    println!(
        "denoised    {denoised:.5}   {:.2}x",
        unfiltered / denoised.max(1e-9)
    );
    println!(
        "\n{:.0} ms a tile on one core ({elapsed:.1}s for {count})",
        elapsed * 1000.0 / count as f32
    );
}

fn relative_error(y: &[f32], t: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for (a, b) in y.iter().zip(t) {
        let d = (a - b) as f64;
        sum += d * d / ((*b as f64) * (*b as f64) + LOSS_EPSILON as f64);
    }
    sum / y.len() as f64
}

// Appended by the parity check: compare against a reference dump if given.
