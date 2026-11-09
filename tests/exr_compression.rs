//! Every EXR compressor against OpenEXR itself.
//!
//! The files in `src/loaders/testdata/exr` were written by OpenEXR 3.4 from
//! one image of ramps past 1 and below 0, noise and a flat region, at odd sizes
//! so chunks and blocks end part way; the `_decreasing` files are stored bottom
//! to top and the `_uint` ones carry a UINT `id` channel. The hashes are FNV-1a of what OpenEXR's
//! own reader decodes from each, as RGBA half with alpha 1 where there is none.
//! Lossy formats match too: decoding them is deterministic. DWA is the one
//! exception, since OpenEXR's own scalar, SSE and NEON DCTs round differently:
//! its `.ref` files hold OpenEXR's decode for a comparison with a tolerance.

use threers::ExrLoader;

const REFERENCE: &[(&str, u64)] = &[
    ("none_half.exr", 0xf42a1dc8928cc347),
    ("rle_half.exr", 0xf42a1dc8928cc347),
    ("rle_float.exr", 0x294cda5778093bfc),
    ("zip_half.exr", 0xf42a1dc8928cc347),
    ("piz_half.exr", 0xf42a1dc8928cc347),
    ("piz_float.exr", 0x294cda5778093bfc),
    ("piz_big.exr", 0x192195e83f7d79aa),
    ("pxr24_half.exr", 0xf42a1dc8928cc347),
    ("pxr24_float.exr", 0x494c9feb2825d67b),
    ("b44_half.exr", 0x787d7bd4d61b9e80),
    ("b44a_half.exr", 0x787d7bd4d61b9e80),
    ("b44_float.exr", 0x294cda5778093bfc),
    ("zip_decreasing.exr", 0xa9df51531ad1eca6),
    ("piz_decreasing.exr", 0xa9df51531ad1eca6),
    ("zip_uint.exr", 0xa9df51531ad1eca6),
    ("piz_uint.exr", 0xa9df51531ad1eca6),
    ("pxr24_uint.exr", 0xa9df51531ad1eca6),
    ("b44_uint.exr", 0x2c02ce70191106d5),
    ("b44_linear.exr", 0xbb256118ee9af549),
    ("b44a_linear.exr", 0xbb256118ee9af549),
];

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/src/loaders/testdata/exr/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(path).unwrap()
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

#[test]
fn every_compressor_decodes_what_openexr_decodes() {
    let mut wrong = Vec::new();
    for (name, want) in REFERENCE {
        let tex = ExrLoader::parse_hdr(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        if fnv(&tex.data) != *want {
            wrong.push(*name);
        }
    }
    assert!(wrong.is_empty(), "differ from OpenEXR: {wrong:?}");
}

/// `dwab_tall` passes 256 lines, so it holds two DWAB chunks; `dwaa_uint`
/// carries alpha through the run-length section and `id` through the unknown
/// one; `dwaa_linear` skips the transfer curve. Measured against OpenEXR on
/// this Mac: 1 sample in 10,000 differs, by no more than 8 half steps. A wrong
/// table or a misplaced block moves thousands by far more.
#[test]
fn dwa_decodes_what_openexr_decodes_to_rounding() {
    for name in [
        "dwaa_half",
        "dwab_half",
        "dwaa_float",
        "dwab_tall",
        "dwaa_uint",
        "dwaa_linear",
    ] {
        let tex = ExrLoader::parse_hdr(&fixture(&format!("{name}.exr"))).unwrap();
        let want = fixture(&format!("{name}.exr.ref"));
        assert_eq!(tex.data.len(), want.len(), "{name}");
        let half = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]) as i32;
        let steps: Vec<i32> = tex
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .zip(want.as_chunks::<2>().0)
            .map(|(a, b)| (half(a) - half(b)).abs())
            .collect();
        let differ = steps.iter().filter(|&&d| d != 0).count();
        assert!(
            differ * 2000 < steps.len(),
            "{name}: {differ} of {} differ",
            steps.len()
        );
        assert!(
            steps.iter().all(|&d| d <= 8),
            "{name}: {} half steps off",
            steps.iter().max().unwrap()
        );
    }
}

/// Damage in the compressed data has to come back as an image or an error,
/// never as a panic: a wasm build aborts on one.
#[test]
fn damaged_files_do_not_panic() {
    let mut seed = 0x2545f4914f6cdd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as usize
    };
    let dwa = ["dwaa_half.exr", "dwab_half.exr", "dwaa_uint.exr"];
    for name in REFERENCE.iter().map(|r| r.0).chain(dwa) {
        let good = fixture(name);
        for round in 0..200 {
            let mut bad = good.clone();
            for _ in 0..1 + round % 16 {
                let at = 300 + next() % (bad.len() - 300);
                bad[at] = next() as u8;
            }
            let _ = ExrLoader::parse_hdr(&bad);
            let cut = 300 + next() % (good.len() - 300);
            let _ = ExrLoader::parse_hdr(&good[..cut]);
        }
    }
}
