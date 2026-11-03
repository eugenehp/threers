//! Dump a `.tex` blob to PNG, to see what is actually in it.
//!
//! A compressed tile is opaque to every ordinary tool: the wrong content and a
//! correct pipeline look identical from the outside, which is exactly the
//! confusion this resolves.
//!
//!     cargo run --release --example tex_dump -- tile.tex out.png

// The `if n > 0` guards below each cover three divisions in one `println!`, which
// reads better than three `checked_div`s and an unwrap apiece.
#[allow(clippy::manual_checked_ops)]
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (src, dst) = (a[0].clone(), a.get(1).cloned().unwrap_or("out/tex_dump.png".into()));
    let t = threers::textures::blob::load(&src).unwrap_or_else(|e| panic!("{e}"));
    println!(
        "{src}: {}x{} {:?}, {} mips",
        t.width,
        t.height,
        t.format,
        t.mips.len()
    );
    // MIP=n dumps a level down the chain instead of level 0. At 14400x14400 the
    // full level is an 800 MB PNG that no viewer will open; mip 4 is 900x900 and
    // says everything about whether the decode was right.
    let mip: usize = std::env::var("MIP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mip = mip.min(t.mips.len());
    let (w, h) = ((t.width >> mip).max(1), (t.height >> mip).max(1));
    let data: &[u8] = if mip == 0 { &t.data } else { &t.mips[mip - 1] };
    let rgba = match t.format {
        threers::TextureFormat::Bc1RgbaUnormSrgb => threers::textures::bc1::decode(data, w, h),
        _ => data.to_vec(),
    };
    let (t_width, t_height) = (t.width, t.height);
    let _ = (t_width, t_height);
    let mut peak = 0u8;
    let mut lit = 0usize;
    for p in rgba.chunks_exact(4) {
        let m = p[0].max(p[1]).max(p[2]);
        peak = peak.max(m);
        if m > 40 {
            lit += 1;
        }
    }
    // A coarse histogram of the brightest channel. For a night map this is the
    // question that matters: how much of it is city and how much is the
    // composite's background, which an emissive map turns into a wash.
    let mut bins = [0usize; 8];
    for p in rgba.chunks_exact(4) {
        let m = p[0].max(p[1]).max(p[2]) as usize;
        bins[(m / 32).min(7)] += 1;
    }
    // What colour the background band actually is, which is what decides how to
    // separate it from the lights: the composite's water is a flat blue-grey and
    // city light is warm, so they part on hue, not only on brightness.
    let (mut br, mut bg, mut bb, mut bn) = (0u64, 0u64, 0u64, 0u64);
    let (mut lr, mut lg, mut lb, mut ln) = (0u64, 0u64, 0u64, 0u64);
    for p in rgba.chunks_exact(4) {
        let m = p[0].max(p[1]).max(p[2]);
        if (64..96).contains(&m) {
            br += p[0] as u64;
            bg += p[1] as u64;
            bb += p[2] as u64;
            bn += 1;
        } else if m >= 128 {
            lr += p[0] as u64;
            lg += p[1] as u64;
            lb += p[2] as u64;
            ln += 1;
        }
    }
    if bn > 0 {
        println!(
            "  band 64..95 mean rgb = ({}, {}, {}) over {bn} px",
            br / bn,
            bg / bn,
            bb / bn
        );
    }
    if ln > 0 {
        println!(
            "  band >=128  mean rgb = ({}, {}, {}) over {ln} px",
            lr / ln,
            lg / ln,
            lb / ln
        );
    }
    // BC1 has a punch-through mode: when endpoint0 <= endpoint1 the block gets
    // three colours plus TRANSPARENT BLACK at index 3. A near-black image is
    // exactly the input that pushes blocks into it, and a decoder that ignores
    // the mode reads back colours the GPU will not show.
    let zero_a = rgba.chunks_exact(4).filter(|p| p[3] == 0).count();
    println!(
        "  alpha 0 on {zero_a} px ({:.3}%)",
        100.0 * zero_a as f64 / (w as f64 * h as f64).max(1.0)
    );
    let total = (w as f64 * h as f64).max(1.0);
    for (i, n) in bins.iter().enumerate() {
        println!(
            "  {:>3}..{:<3} {:>7.3}%",
            i * 32,
            i * 32 + 31,
            100.0 * *n as f64 / total
        );
    }
    println!(
        "  peak channel {peak}, {} px above 40 ({:.3}% of the tile)",
        lit,
        100.0 * lit as f64 / (w as f64 * h as f64)
    );
    if let Some(dir) = std::path::Path::new(&dst).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&dst, threers::encode_png(w, h, &rgba)).unwrap();
    println!("  wrote {dst} at mip {mip} ({w}x{h})");
}
