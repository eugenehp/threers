//! The in-process compressed HEVC MP4 encoder: it must produce a file a real
//! decoder plays, and it must be dramatically smaller than the `I_PCM` path it
//! exists to replace.

use std::process::Command;

use threers::codec::hevc::{
    encode_compressed_mp4, encode_compressed_mp4_streaming, Yuv420Frame,
};

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// A detailed moving frame — smooth content would not exercise the coefficient
/// coder, which is the whole point of this path.
fn frame(w: u32, h: u32, t: u32) -> Yuv420Frame {
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let checker = (((x / 7) + (y / 5) + t) % 2) as u8;
            rgba[i] = (x * 3 + t * 11) as u8 ^ (checker * 200);
            rgba[i + 1] = (y * 5) as u8;
            rgba[i + 2] = (x ^ y) as u8;
            rgba[i + 3] = 255;
        }
    }
    Yuv420Frame::from_rgba(w, h, &rgba)
}

/// Sizes that are not a multiple of the 16-sample CTB exercise the conformance
/// window: the picture is coded padded and cropped back on output.
#[test]
fn non_ctb_aligned_sizes_are_playable() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    for &(w, h) in &[(100u32, 60u32), (66, 34), (128, 90)] {
        let frames: Vec<Yuv420Frame> = (0..3).map(|t| frame(w, h, t)).collect();
        let bytes = encode_compressed_mp4(w, h, 24, 27, &frames);
        let path = std::env::temp_dir()
            .join(format!("threers-native-{}-{w}x{h}.mp4", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let out = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-count_frames",
                   "-show_entries", "stream=codec_name,width,height,nb_read_frames",
                   "-of", "csv=p=0"])
            .arg(&path)
            .output()
            .expect("run ffprobe");
        let probe = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let _ = std::fs::remove_file(&path);
        assert_eq!(probe, format!("hevc,{w},{h},3"), "at {w}x{h}");
    }
}

#[test]
fn compressed_mp4_is_playable_and_far_smaller_than_pcm() {
    let (w, h, n) = (128u32, 96u32, 6u32);
    let frames: Vec<Yuv420Frame> = (0..n).map(|t| frame(w, h, t)).collect();

    let compressed = encode_compressed_mp4(w, h, 24, 27, &frames);
    let pcm = threers::codec::h264::encode_mp4(w, h, 24, &frames);

    let raw = (w * h * 3 / 2 * n) as usize;
    assert!(
        compressed.len() * 4 < pcm.len(),
        "compressed ({}) should be far under I_PCM ({pcm_len}) / raw ({raw})",
        compressed.len(),
        pcm_len = pcm.len()
    );

    if !have_ffmpeg() {
        eprintln!("skipping playback check: ffmpeg not found");
        return;
    }
    let path = std::env::temp_dir().join(format!("threers-native-{}.mp4", std::process::id()));
    std::fs::write(&path, &compressed).unwrap();

    // ffprobe must see an hvc1 track of the right size and frame count.
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-count_frames",
               "-show_entries", "stream=codec_name,width,height,nb_read_frames",
               "-of", "csv=p=0"])
        .arg(&path)
        .output()
        .expect("run ffprobe");
    let probe = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let _ = std::fs::remove_file(&path);
    assert!(
        probe.starts_with("hevc,128,96,") && probe.ends_with(&n.to_string()),
        "expected an hevc 128x96 track with {n} frames, ffprobe said {probe:?}"
    );
}


/// The streaming form must be a drop-in for the slice form.
///
/// It exists for high resolutions, where holding every source frame is the
/// dominant cost — a 4:2:0 8K frame is ~50 MB, so ten seconds at 24fps is about
/// 12 GB of sources against a few MB of encoded samples. Measured on a 12-frame
/// 8K clip it cut peak RSS from 1242 MB to 177 MB, and stays roughly flat as
/// frames are added where the slice form grows ~50 MB each. The bytes it
/// produces are identical, which is what this pins.
#[test]
fn streaming_matches_the_slice_form_byte_for_byte() {
    let (w, h, n) = (96u32, 64u32, 5u32);
    let frames: Vec<Yuv420Frame> = (0..n).map(|t| frame(w, h, t)).collect();

    let eager = encode_compressed_mp4(w, h, 24, 29, &frames);
    let streamed =
        encode_compressed_mp4_streaming(w, h, 24, 29, n as usize, None, |i| frame(w, h, i as u32));

    assert_eq!(
        eager, streamed,
        "streaming produced {} bytes against the slice form's {}",
        streamed.len(),
        eager.len()
    );
    assert!(!eager.is_empty());
}

/// The callback must see every index once, in order — an encoder that pulled out
/// of order would still produce a playable file, just the wrong one.
#[test]
fn streaming_pulls_frames_in_order() {
    let (w, h, n) = (64u32, 64u32, 6usize);
    let mut seen = Vec::new();
    let _ = encode_compressed_mp4_streaming(w, h, 24, 30, n, None, |i| {
        seen.push(i);
        frame(w, h, i as u32)
    });
    assert_eq!(seen, (0..n).collect::<Vec<_>>());
}
