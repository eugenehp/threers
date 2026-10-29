//! Opus (WebM) against ffmpeg n7.1's native decoder: the FNV-1a hash of
//! `ffmpeg -i X.webm -f f32le -` for one-second clips that cover CELT
//! (mono, stereo, 60 ms code-3 packets, intensity with phase inversion,
//! dual stereo), SILK (NB, WB with LBRR, stereo), hybrid (10 ms, mixed-mode
//! streams with redundancy) and a 5.1 multistream file.
//!
//! The decoder's float arithmetic is pure Rust; only `exp2f`/`cosf`/`sinf`
//! come from the platform libm, so the hashes are asserted on the reference
//! platform (macOS aarch64) and the lengths everywhere.


use std::path::Path;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[test]
fn opus_webm_matches_ffmpeg() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/support/opus");
    let manifest = std::fs::read_to_string(dir.join("expected.txt")).unwrap();
    let mut checked = 0;
    for line in manifest.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let mut it = line.split_whitespace();
        let (name, len, hash) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        let len: usize = len.parse().unwrap();
        let hash = u64::from_str_radix(hash, 16).unwrap();
        let data = std::fs::read(dir.join(name)).unwrap();
        let d = threers_codec::opus::decode_webm(&data).unwrap_or_else(|e| panic!("{name}: {e}"));
        let n = d.planes[0].len();
        let mut out = Vec::with_capacity(n * d.planes.len() * 4);
        for i in 0..n {
            for p in &d.planes {
                out.extend_from_slice(&p[i].to_le_bytes());
            }
        }
        assert_eq!(out.len(), len, "{name}: output length");
        assert_eq!(d.frame_lens.iter().sum::<usize>(), n, "{name}: frame lengths");
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert_eq!(fnv1a(&out), hash, "{name}: samples differ from ffmpeg");
        }
        checked += 1;
    }
    assert_eq!(checked, 10);
}
