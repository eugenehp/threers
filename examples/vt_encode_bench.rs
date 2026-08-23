//! What in-process hardware encoding costs, against the pipe it replaces.
//!
//! The comparison that matters is not "is VideoToolbox fast" — ffmpeg was
//! already using it — but "what does the process boundary cost". Both paths
//! compress with the same hardware block, so the difference between them is
//! the pipe: a frame copied into it and copied out again.
//!
//!     cargo run --release --features videotoolbox \
//!         --example vt_encode_bench -- frames.raw 7680 4320 [count]
//!
//! `frames.raw` is tightly-packed RGB, which is what the renderer's GPU pack
//! produces. Encoding real frames matters: a constant field compresses in a
//! fraction of the time real content does, so a synthetic benchmark here would
//! flatter both paths and rank them wrong.

#[cfg(all(feature = "videotoolbox", target_os = "macos"))]
fn main() {
    use threers::videotoolbox::VideoToolboxEncoder;

    let a: Vec<String> = std::env::args().skip(1).collect();
    let path = a.first().cloned().unwrap_or_else(|| {
        eprintln!("usage: vt_encode_bench <frames.rgb> <w> <h> [count]");
        std::process::exit(2);
    });
    let w: u32 = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(7680);
    let h: u32 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(4320);
    let frame = (w as usize) * (h as usize) * 3;

    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let have = data.len() / frame;
    let n: usize = a
        .get(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(have)
        .min(have.max(1));
    if have == 0 {
        eprintln!(
            "{path} holds no whole {w}x{h} frames ({} bytes)",
            data.len()
        );
        std::process::exit(2);
    }
    println!("{path}: {have} frames of {w}x{h}, encoding {n}");

    let out = std::env::temp_dir().join("threers_vt_bench.h265");
    let mut enc = VideoToolboxEncoder::new(out.to_str().unwrap(), w, h, 30, 0.62)
        .unwrap_or_else(|e| panic!("open: {e}"));

    let t0 = std::time::Instant::now();
    for i in 0..n {
        let f = &data[(i % have) * frame..(i % have) * frame + frame];
        enc.push_rgb(f).unwrap_or_else(|e| panic!("frame {i}: {e}"));
    }
    let (frames, bytes) = enc.finish().unwrap_or_else(|e| panic!("finish: {e}"));
    let secs = t0.elapsed().as_secs_f64();

    println!(
        "in-process VideoToolbox: {:.2} s for {n} frames = {:.1} ms/frame",
        secs,
        secs * 1000.0 / n as f64
    );
    println!(
        "  emitted {frames} encoded frames, {:.1} MB ({:.2} MB/frame)",
        bytes as f64 / 1e6,
        bytes as f64 / 1e6 / frames.max(1) as f64
    );
    println!("  stream: {}", out.display());
    println!("  remux with:  ffmpeg -i {} -c copy out.mp4", out.display());
}

#[cfg(not(all(feature = "videotoolbox", target_os = "macos")))]
fn main() {
    eprintln!("build with --features videotoolbox, on macOS");
}
