//! Emit a transparent HEVC `.mov` for the AVFoundation/QuickTime oracle.
//!
//! Writes a 64×64 transparent clip (left→right alpha ramp) to
//! `$THREERS_ALPHA_OUT` (or a temp file) and sanity-checks the container. The
//! *composite* is verified out-of-band by decoding it with AVFoundation
//! (`decode_check.swift`) / opening it in QuickTime.
#![cfg(feature = "native-codec")]

use threers::codec::hevc::{TransparentEncoder, Yuv420Frame};

fn out_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("THREERS_ALPHA_OUT") {
        std::path::PathBuf::from(p)
    } else {
        let mut p = std::env::temp_dir();
        p.push("threers_alpha.mov");
        p
    }
}

#[test]
fn write_transparent_mov() {
    let (w, h) = (64u32, 64u32);

    // Color: a smooth RGB gradient.
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            rgba[i] = (x * 4) as u8; // R
            rgba[i + 1] = (y * 4) as u8; // G
            rgba[i + 2] = 128; // B
            rgba[i + 3] = 255;
        }
    }
    let color = Yuv420Frame::from_rgba(w, h, &rgba);

    // Alpha ramp: fully transparent at the left edge → opaque at the right.
    let alpha: Vec<u8> = (0..w * h)
        .map(|k| ((k % w) * 255 / (w - 1)) as u8)
        .collect();

    let mut enc = TransparentEncoder::new(w, h);
    enc.encode_frame(&color, &alpha);
    let mov = enc.finish_mov(30);

    // Container sanity.
    assert_eq!(&mov[4..8], b"ftyp", "starts with ftyp");
    assert!(find(&mov, b"hvcC"), "has hvcC config");
    assert!(find(&mov, b"almo"), "has almo alpha-mode box");
    assert!(find(&mov, b"mdat"), "has mdat");

    let path = out_path();
    std::fs::write(&path, &mov).unwrap();
    eprintln!(
        "wrote transparent .mov: {} ({} bytes)",
        path.display(),
        mov.len()
    );
}

fn find(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// End-to-end oracle: decode our transparent `.mov` with AVFoundation (== the
/// QuickTime/Safari decoder) via a tiny Swift program and assert the alpha
/// channel is lossless. Skips if `swiftc` is unavailable (non-macOS / no
/// toolchain). Runs at two sizes to prove the baked VPS is resolution-independent.
#[test]
fn avfoundation_decodes_alpha_losslessly() {
    if std::process::Command::new("swiftc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| !s.success())
        .unwrap_or(true)
    {
        eprintln!("skipping AVFoundation alpha check: swiftc not available");
        return;
    }
    for (w, h) in [(64u32, 64u32), (128, 96)] {
        check_size(w, h);
    }
}

#[cfg(test)]
fn check_size(w: u32, h: u32) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir();
    let mov = dir.join(format!("threers_a_{nanos}_{w}x{h}.mov"));
    let swift = dir.join(format!("threers_dec_{nanos}.swift"));
    let bin = dir.join(format!("threers_dec_{nanos}"));
    let bgra = dir.join(format!("threers_dec_{nanos}.bgra"));

    // Encode a transparent clip: left→right alpha ramp.
    let rgba: Vec<u8> = (0..w * h)
        .flat_map(|k| {
            let (x, y) = (k % w, k / w);
            [(x * 2) as u8, (y * 2) as u8, 100, 255]
        })
        .collect();
    let color = Yuv420Frame::from_rgba(w, h, &rgba);
    let alpha: Vec<u8> = (0..w * h)
        .map(|k| ((k % w) * 255 / (w - 1)) as u8)
        .collect();
    let mut enc = TransparentEncoder::new(w, h);
    enc.encode_frame(&color, &alpha);
    std::fs::write(&mov, enc.finish_mov(30)).unwrap();

    std::fs::write(&swift, SWIFT_DECODER).unwrap();
    let ok = std::process::Command::new("swiftc")
        .args(["-O"])
        .arg(&swift)
        .arg("-o")
        .arg(&bin)
        .status()
        .expect("swiftc")
        .success();
    assert!(ok, "swiftc failed to build the decoder");
    let ok = std::process::Command::new(&bin)
        .arg(&mov)
        .arg(&bgra)
        .status()
        .expect("run decoder")
        .success();
    assert!(
        ok,
        "AVFoundation failed to decode our transparent {w}x{h} mov"
    );

    let out = std::fs::read(&bgra).unwrap();
    assert_eq!(out.len(), (w * h * 4) as usize, "decoded size {w}x{h}");
    let mut max_err = 0i32;
    for k in 0..(w * h) as usize {
        let (x, _) = ((k as u32) % w, (k as u32) / w);
        let got = out[k * 4 + 3] as i32;
        let want = (x * 255 / (w - 1)) as i32;
        max_err = max_err.max((got - want).abs());
    }
    let _ = std::fs::remove_file(&mov);
    let _ = std::fs::remove_file(&swift);
    let _ = std::fs::remove_file(&bin);
    let _ = std::fs::remove_file(&bgra);
    assert!(
        max_err <= 1,
        "alpha not lossless at {w}x{h}: max_err={max_err}"
    );
    eprintln!("AVFoundation alpha OK: {w}x{h} lossless (max_err {max_err})");
}

/// Minimal Swift/AVFoundation decoder: writes the first frame's BGRA to argv[2].
#[cfg(test)]
const SWIFT_DECODER: &str = r#"
import AVFoundation
import CoreVideo
import Foundation
let a = CommandLine.arguments
let asset = AVURLAsset(url: URL(fileURLWithPath: a[1]))
let sem = DispatchSemaphore(value: 0)
Task {
    guard let t = try? await asset.loadTracks(withMediaType: .video).first else { exit(3) }
    let r = try! AVAssetReader(asset: asset)
    let o = AVAssetReaderTrackOutput(track: t, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)])
    r.add(o); r.startReading()
    guard let s = o.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(s) else { exit(4) }
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    let rb = CVPixelBufferGetBytesPerRow(pb)
    let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
    var out = Data(capacity: w*h*4)
    for y in 0..<h { for x in 0..<w { let p = y*rb + x*4; out.append(base[p]); out.append(base[p+1]); out.append(base[p+2]); out.append(base[p+3]) } }
    CVPixelBufferUnlockBaseAddress(pb, .readOnly)
    try? out.write(to: URL(fileURLWithPath: a[2]))
    sem.signal()
}
sem.wait()
"#;
