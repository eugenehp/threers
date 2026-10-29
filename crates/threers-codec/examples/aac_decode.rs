//! Decode the AAC track of an MP4/M4A and write interleaved f32le (as
//! `ffmpeg -i IN -f f32le -c:a pcm_f32le OUT`), or compare against such a file.
//!
//!   cargo run --release --example aac_decode --features native-codec -- IN OUT.f32
//!   cargo run --release --example aac_decode --features native-codec -- IN --compare REF.f32

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).expect("read input");
    let t = std::time::Instant::now();
    let mut a = threers_codec::aac::decode_mp4(&data).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    // Diagnostics: replace our float samples with ffmpeg's (interleaved f32le)
    // to test the post-processing on its own.
    if let Ok(f) = std::env::var("AAC_PLANES_FROM") {
        let r = std::fs::read(f).expect("read planes");
        let r: Vec<f32> = r.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let ch = a.planes.len();
        for (c, p) in a.planes.iter_mut().enumerate() {
            *p = r.iter().skip(c).step_by(ch).copied().collect();
        }
    }
    let ch = a.planes.len();
    let n = a.planes[0].len();
    eprintln!("{} Hz, {ch} ch, {n} samples, decoded in {:.3}s", a.sample_rate, t.elapsed().as_secs_f64());
    eprintln!("{:?}", a.stats);
    let mut inter = Vec::with_capacity(n * ch);
    for i in 0..n {
        for p in &a.planes {
            inter.push(p[i]);
        }
    }
    if args.get(2).map(String::as_str) == Some("--s16") {
        // Compare with `ffmpeg -i IN -vn -f s16le -c:a pcm_s16le -ar RATE -ac CH REF`
        // (`--s16 RATE REF` or `--s16 RATE REF CH`, CH defaulting to 2).
        let rate: u32 = args[3].parse().unwrap();
        let ch: usize = args.get(5).map_or(2, |c| c.parse().unwrap());
        let ours = threers_codec::swr::ffmpeg_cli_s16(
            &threers_codec::swr::DecodedFrames { planes: &a.planes, sample_rate: a.sample_rate, frame_lens: &a.frame_lens },
            rate,
            ch,
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        });
        let r = std::fs::read(&args[4]).expect("read reference");
        let r: Vec<i16> = r.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
        let m = ours.len().min(r.len());
        let diff: Vec<usize> = (0..m).filter(|&i| ours[i] != r[i]).collect();
        let maxd = diff.iter().map(|&i| (ours[i] as i32 - r[i] as i32).abs()).max().unwrap_or(0);
        println!(
            "s16: ours {} ref {} values, {} differ (max {maxd}), first {:?}",
            ours.len(),
            r.len(),
            diff.len(),
            &diff[..diff.len().min(5)]
        );
        return;
    }
    if args.get(2).map(String::as_str) == Some("--compare") {
        let r = std::fs::read(&args[3]).expect("read reference");
        let r: Vec<f32> = r.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        println!("samples: ours {} ref {}", inter.len() / ch, r.len() / ch);
        let m = inter.len().min(r.len());
        let (mut max_abs, mut max_ulp, mut exact) = (0f64, 0i64, 0usize);
        let mut first_bad = None;
        for i in 0..m {
            let d = (inter[i] as f64 - r[i] as f64).abs();
            if d == 0.0 {
                exact += 1;
            } else if first_bad.is_none() && d > 1e-4 {
                first_bad = Some(i);
            }
            max_abs = max_abs.max(d);
            let u = (inter[i].to_bits() as i32 as i64 - r[i].to_bits() as i32 as i64).abs();
            if inter[i].signum() == r[i].signum() && r[i].abs() > 1e-3 {
                max_ulp = max_ulp.max(u);
            }
        }
        let peak = r.iter().fold(0f32, |a, &b| a.max(b.abs()));
        println!("exact {exact}/{m}, max |d| {max_abs:.3e} (peak {peak:.3}), max ulp (|x|>1e-3) {max_ulp}, first |d|>1e-4 at {first_bad:?}");
    } else {
        let bytes: Vec<u8> = inter.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&args[2], bytes).expect("write output");
    }
}
