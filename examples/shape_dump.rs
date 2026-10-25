//! One line of shaped glyph ids, in HarfBuzz's `--no-positions --no-glyph-names` format.
//!
//!     cargo run --release --example shape_dump -- <font.ttf> "text" [script]
//!
//! For `tests/../shapecmp.py`, which compares this against `hb-shape` over every
//! TrueType font on the machine. HarfBuzz is a third implementation — neither this code
//! nor the one it was ported from — which is the only kind of agreement worth much.

use threers::captions::font::CaptionFont;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: shape_dump <font> <text> [script]");
        std::process::exit(2);
    };
    let text = args.get(2).map_or("", String::as_str);
    let script: [u8; 4] = args
        .get(3)
        .and_then(|s| s.as_bytes().try_into().ok())
        .unwrap_or(*b"latn");
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    };
    let font = match CaptionFont::from_ttf_bytes(&bytes) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{path}: {e:?}");
            std::process::exit(1);
        }
    };
    // Logical order and no bidi, which is what `hb-shape` reports: this compares the
    // shaper, and the reordering is measured against ICU elsewhere. Clusters are byte
    // offsets, as HarfBuzz gives them.
    let chars: Vec<char> = text.chars().collect();
    let mut byte_of = Vec::with_capacity(chars.len());
    let mut at = 0usize;
    for c in &chars {
        byte_of.push(at);
        at += c.len_utf8();
    }
    let out: Vec<String> = font
        .shape_run(&chars, &script, 24.0)
        .into_iter()
        .map(|(gid, cluster)| {
            format!("{gid}={}", byte_of.get(cluster).copied().unwrap_or(0))
        })
        .collect();
    println!("{}", out.join("|"));
}
