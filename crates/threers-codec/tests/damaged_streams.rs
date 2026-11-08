//! Damaged input must end in frames or an error: never a panic, never a hang.
//!
//! It matters more than a crash usually does. On wasm a panic aborts the
//! whole instance, and a size that merely fails to allocate does the same, so
//! nothing a stream says may reach an index, a loop count or an allocation
//! unchecked. Each case here is a way that has happened.

use std::process::Command;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use threers_codec::decode::av1::decode::Fd;
use threers_codec::decode::av1::Av1Decoder;
use threers_codec::decode::vp9::Vp9Decoder;
use threers_codec::decode::{decode_annexb, decode_video};
use threers_codec::mkv_demux::parse_ivf;
use threers_codec::mp4_demux::TrackCodec;

const FIXTURES: [&[u8]; 3] = [
    include_bytes!("support/hevc/chroma_qp_offset_list_422_10.bin"),
    include_bytes!("support/hevc/rext_lossless_rdpcm_444_8.bin"),
    include_bytes!("support/hevc/rext_high_precision_wp_444_10.bin"),
];

fn ffmpeg_with(lib: &str) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(lib))
        .unwrap_or(false)
}

/// Run `f` with a time limit, so that a hang fails rather than stalls.
fn within(secs: u64, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    match rx.recv_timeout(Duration::from_secs(secs)) {
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("still decoding after {secs} s"),
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("the decoder panicked"),
        Ok(()) => {}
    }
}

/// Parameter sets and slice headers are where a stream states sizes, counts
/// and QPs, and they all sit in the first few hundred bytes. The stream is cut
/// short too, so that many rounds stay quick. An unoptimised build is about
/// fifteen times slower, so it runs a tenth of the rounds; the release runs
/// in `docs/releasing.md` cover the rest.
#[test]
fn hevc_header_damage_ends_cleanly() {
    let rounds = if cfg!(debug_assertions) { 600 } else { 6000 };
    within(240, move || {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as usize
        };
        for f in FIXTURES {
            let clean = &f[..f.len().min(6000)];
            for _ in 0..rounds {
                let mut d = clean.to_vec();
                for _ in 0..1 + rnd() % 3 {
                    let at = rnd() % 300;
                    d[at] ^= 1 << (rnd() % 8);
                }
                let _ = decode_annexb(TrackCodec::Hevc, &d);
            }
        }
    });
}

/// A header claiming the largest size its fields can hold, which no level
/// allows. It has to be refused before anything is allocated from it.
#[test]
fn av1_refuses_a_picture_no_level_allows() {
    if !ffmpeg_with("libsvtav1") {
        eprintln!("SKIP: needs ffmpeg with libsvtav1");
        return;
    }
    let path = std::env::temp_dir().join(format!("threers_damaged_{}.ivf", std::process::id()));
    let ok = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10",
            "-frames:v",
            "1",
        ])
        .args(["-pix_fmt", "yuv420p", "-c:v", "libsvtav1", "-preset", "10"])
        .arg(&path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "SVT-AV1 failed");
    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let units = parse_ivf(&data).unwrap().4;
    let frame = Av1Decoder::new()
        .decode(&units[0].1)
        .unwrap()
        .pop()
        .unwrap();
    let mut hdr = (*frame.hdr).clone();
    hdr.width = [65535; 2];
    hdr.height = 65535;
    let e = Fd::new(frame.seq.clone(), Arc::new(hdr), &Default::default())
        .err()
        .expect("a 65535x65535 frame was accepted");
    assert!(e.contains("larger than any level"), "{e}");
}

/// A VP9 key frame's size sits at a fixed place in its header.
#[test]
fn vp9_refuses_a_picture_no_level_allows() {
    if !ffmpeg_with("libvpx-vp9") {
        eprintln!("SKIP: needs ffmpeg with libvpx-vp9");
        return;
    }
    let path = std::env::temp_dir().join(format!("threers_damaged_{}.vp9.ivf", std::process::id()));
    let ok = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10",
            "-frames:v",
            "1",
        ])
        .args([
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libvpx-vp9",
            "-cpu-used",
            "8",
        ])
        .arg(&path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "libvpx failed");
    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let mut key = parse_ivf(&data).unwrap().4.remove(0).1;
    assert!(Vp9Decoder::new().decode(&key).is_ok());
    // Profile 0: width - 1 starts 36 bits in, height - 1 follows.
    for bit in 36..68 {
        key[bit / 8] |= 0x80 >> (bit % 8);
    }
    let e = Vp9Decoder::new()
        .decode(&key)
        .err()
        .expect("a 65536x65536 frame was accepted");
    assert!(e.contains("larger than any level"), "{e}");
}

/// Flipped bits anywhere in the headers, and frames cut off at every short
/// length: one that ends before its compressed header starts, for one.
#[test]
fn vp9_header_damage_ends_cleanly() {
    if !ffmpeg_with("libvpx-vp9") {
        eprintln!("SKIP: needs ffmpeg with libvpx-vp9");
        return;
    }
    let path =
        std::env::temp_dir().join(format!("threers_damaged_{}.vp9b.ivf", std::process::id()));
    let ok = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10",
            "-frames:v",
            "3",
        ])
        .args([
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libvpx-vp9",
            "-cpu-used",
            "8",
        ])
        .arg(&path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "libvpx failed");
    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let units: Vec<Vec<u8>> = parse_ivf(&data)
        .unwrap()
        .4
        .into_iter()
        .map(|(_, d)| d)
        .collect();
    within(120, move || {
        for at in 0..24 {
            for bit in 0..8 {
                for u in 0..units.len() {
                    let mut d = units.clone();
                    if at < d[u].len() {
                        d[u][at] ^= 1 << bit;
                    }
                    let mut dec = Vp9Decoder::new();
                    for p in &d {
                        let _ = dec.decode(p);
                    }
                }
            }
        }
        for u in &units {
            for n in 0..u.len().min(64) {
                let _ = Vp9Decoder::new().decode(&u[..n]);
            }
        }
    });
}

/// One bad packet costs that frame, not the file. Found on a real clip: two
/// damaged frames in ten seconds, which dav1d skipped and this gave up on.
#[test]
fn a_damaged_frame_is_skipped_and_counted() {
    if !ffmpeg_with("libsvtav1") {
        eprintln!("SKIP: needs ffmpeg with libsvtav1");
        return;
    }
    let path =
        std::env::temp_dir().join(format!("threers_damaged_{}.skip.ivf", std::process::id()));
    let ok = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x96:rate=10",
            "-frames:v",
            "40",
        ])
        .args(["-pix_fmt", "yuv420p", "-c:v", "libsvtav1", "-preset", "10"])
        .arg(&path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "SVT-AV1 failed");
    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let count = |d: &[u8]| {
        let mut n = 0;
        decode_video(d, |_, _| {
            n += 1;
            Ok(())
        })
        .map(|skipped| (n, skipped))
    };
    assert_eq!(count(&data), Ok((40, 0)));
    // Cut the last unit off partway through an OBU.
    let mut at = 32;
    let mut last = (0, 0);
    while at + 12 <= data.len() {
        let n = u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        last = (at, n);
        at += 12 + n;
    }
    let keep = last.1 / 2;
    let mut d = data[..last.0 + 12 + keep].to_vec();
    d[last.0..last.0 + 4].copy_from_slice(&(keep as u32).to_le_bytes());
    assert_eq!(count(&d), Ok((39, 1)));
    // Nothing decodable is an error, not an empty success.
    let mut d = data[..32].to_vec();
    d.extend_from_slice(&[4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff]);
    assert!(count(&d).is_err());
}
