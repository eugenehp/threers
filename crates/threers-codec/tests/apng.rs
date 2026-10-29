//! Emit an APNG with a known alpha pattern for external verification.

use threers_codec::apng::ApngEncoder;

#[test]
fn write_apng() {
    let (w, h) = (32u32, 32u32);
    let mut enc = ApngEncoder::new(w, h, 0);
    // 3 frames: alpha ramps left→right; RGB shifts per frame.
    for f in 0..3u32 {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                rgba[i] = (x * 8 + f * 40) as u8;
                rgba[i + 1] = (y * 8) as u8;
                rgba[i + 2] = 200;
                rgba[i + 3] = (x * 255 / (w - 1)) as u8; // alpha ramp
            }
        }
        enc.add_frame(&rgba, 1, 10);
    }
    let png = enc.finish();
    assert!(png.windows(8).any(|w| w == b"acTL") || png.starts_with(b"\x89PNG"));
    assert!(png.len() > 64);
    if let Ok(path) = std::env::var("THREERS_APNG_OUT") {
        std::fs::write(&path, &png).unwrap();
    }
}
