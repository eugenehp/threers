//! Where a SCAD animation's time goes.
//!
//! ```text
//! cargo run --release --features openscad --example scad_profile
//! cargo run --release --features openscad --example scad_profile -- --frames 24 --concurrency 8
//! ```
//!
//! The answer is evaluation, by a distance: on the reference model the CSG is
//! about 99% of it and the GPU render under two milliseconds a frame. Which is
//! worth knowing before optimising the renderer, and worth re-checking after
//! any change to the CSG kernel.
//!
//! Two traps this exists to avoid. `render_frames` evaluates for itself, so
//! timing an `evaluate()` before it measures the CSG twice and calls half of it
//! rendering — subtract it, as below. And a contended machine makes any single
//! reading useless; take the best of several, since noise only ever adds.
use std::time::Instant;
use threers::openscad::animate::{ScadAnimation, ScadKernel, ScadQuality, ScadRender};

fn main() {
    let src = std::fs::read_to_string("examples/scad_animate.scad").unwrap();
    let frames: usize = std::env::args()
        .position(|a| a == "--frames")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);

    // `render_frames` evaluates for itself, so timing an `evaluate()` before it
    // measures the CSG twice and calls half of it rendering. It is not.
    let t = Instant::now();
    let conc: Option<usize> = std::env::args()
        .position(|a| a == "--concurrency")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok());
    let mut anim = ScadAnimation::from_source(&src).frames(frames);
    if let Some(c) = conc {
        anim = anim.concurrency(c);
    }
    if std::env::args().any(|a| a == "--float") {
        anim = anim.kernel(ScadKernel::Float);
    }
    let evaluated = anim.evaluate().unwrap();
    let eval = t.elapsed().as_secs_f64();
    let tris = evaluated.len();

    // How much of that CSG is redundant? Hash every part's output geometry and
    // count how many are repeats of one already produced by another frame. A
    // part whose subtree does not move produces identical vertices every frame,
    // and computing it once would be the whole saving.
    {
        use std::collections::HashMap;
        let mut seen: HashMap<u64, usize> = HashMap::new();
        let mut total = 0usize;
        for f in &evaluated {
            for part in f.parts.iter() {
                let pos = part.geometry.get_attribute("position").map(|a| &a.array);
                let mut h: u64 = 0xcbf29ce484222325;
                if let Some(a) = pos {
                    for v in a.iter() {
                        h ^= v.to_bits() as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                }
                *seen.entry(h).or_insert(0) += 1;
                total += 1;
            }
        }
        let repeats = total - seen.len();
        println!(
            "  parts: {total} produced, {} distinct, {repeats} repeats ({:.0}% redundant)",
            seen.len(),
            100.0 * repeats as f64 / total.max(1) as f64
        );
    }

    // Default to what the example actually renders at. Supersampling is the
    // one setting that changes where the render time goes, so measuring at the
    // library default (1×) would miss it entirely.
    let quality = match std::env::args()
        .position(|a| a == "--quality")
        .and_then(|i| std::env::args().nth(i + 1))
        .as_deref()
    {
        Some("draft") => ScadQuality::Draft,
        Some("fine") => ScadQuality::Fine,
        _ => ScadQuality::Balanced,
    };
    let size: u32 = std::env::args()
        .position(|a| a == "--width")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(960);
    let render = ScadRender::new(size, size * 9 / 16).quality(quality);
    // Render the frames already evaluated above. Subtracting a separately
    // measured `eval` from `render_frames` used to work, but the CSG now varies
    // by more between runs than the whole render costs — the subtraction was
    // returning negative render times.
    // `render_evaluated` acquires its own GPU device, and that costs about as
    // much as rendering twenty frames — charging it to every frame made the
    // per-frame figure a function of how many frames you asked for (57 ms at
    // 2 frames, 14 ms at 24). Two runs separate them: the difference between
    // them is per-frame, and what is left over is the setup.
    let t = Instant::now();
    let _ = render.render_evaluated(&evaluated[..1]).unwrap();
    let one = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let rgba = render.render_evaluated(&evaluated).unwrap();
    let draw = t.elapsed().as_secs_f64();
    let per_frame = if frames > 1 {
        ((draw - one) / (frames - 1) as f64).max(0.0)
    } else {
        draw
    };
    let setup = (one - per_frame).max(0.0);

    let t = Instant::now();
    let (w, h) = render.size();
    let mut bytes = 0usize;
    let mut last = Vec::new();
    for f in &rgba {
        last = threers::encode_png(w, h, f);
        bytes += last.len();
    }
    let png = t.elapsed().as_secs_f64();

    // The frames are encoded to time the encoder; keeping the last one costs
    // nothing and gives the profile something to look at.
    if !last.is_empty() {
        let out = "out/scad_profile.png";
        if let Some(dir) = std::path::Path::new(out).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if std::fs::write(out, &last).is_ok() {
            println!("  wrote {out} ({w}x{h})");
        }
    }

    println!(
        "{frames} frames at {w}x{h} {quality:?} ({}x supersample), {tris} evaluated",
        quality.supersample()
    );
    println!(
        "  evaluate (CSG)  {eval:6.2}s   {:5.0} ms/frame",
        eval * 1000.0 / frames as f64
    );
    println!(
        "  render (GPU)    {draw:6.2}s   {:5.1} ms/frame  (+{:.0} ms once, acquiring the GPU)",
        per_frame * 1000.0,
        setup * 1000.0
    );
    println!(
        "  encode PNG      {png:6.2}s   {:5.0} ms/frame  ({:.1} MB)",
        png * 1000.0 / frames as f64,
        bytes as f64 / 1e6
    );
}
