//! Validate the WebM muxer end-to-end against ffmpeg: take real VP9 frames
//! (encoded by ffmpeg into an IVF, used purely as a dev oracle), re-container them
//! with our muxer, and require ffmpeg to decode the result to *exactly* the same
//! pixels as the original. A malformed container would fail to open or decode
//! differently.
#![cfg(feature = "native-codec")]

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use threers::codec::hevc::Yuv420Frame;
use threers::codec::vp9::{
    encode_inter_frame, encode_inter_frame_altref, encode_inter_frame_compound,
    encode_inter_frame_golden, encode_inter_frame_refresh, encode_inter_newmv_residual,
    encode_inter_newmv_skip, encode_inter_residual, encode_inter_zeromv_skip, encode_intra_frame,
    encode_intra_gray,
};
use threers::codec::webm::{
    encode_gray_webm, encode_webm, mux_webm, WebmCodec, WebmFrame, WebmParams,
};

fn ffmpeg_ok() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn has_vp9_encoder() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-h", "encoder=libvpx-vp9"])
        .output()
        .map(|o| o.status.success() && !String::from_utf8_lossy(&o.stdout).contains("Unknown"))
        .unwrap_or(false)
}

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "threers_webm_{}_{n}_{seq}_{name}",
        std::process::id()
    ))
}

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Parse an IVF file into `(width, height, [frame bitstreams])`.
fn parse_ivf(buf: &[u8]) -> (u32, u32, Vec<Vec<u8>>) {
    assert_eq!(&buf[0..4], b"DKIF", "not an IVF file");
    let hdr_len = le16(buf, 6) as usize;
    let (w, h) = (le16(buf, 12) as u32, le16(buf, 14) as u32);
    let mut frames = Vec::new();
    let mut o = hdr_len;
    while o + 12 <= buf.len() {
        let size = le32(buf, o) as usize;
        o += 12; // 4-byte size + 8-byte timestamp
        assert!(o + size <= buf.len(), "truncated IVF frame");
        frames.push(buf[o..o + size].to_vec());
        o += size;
    }
    (w, h, frames)
}

fn decode_to_yuv(input: &std::path::Path) -> Option<Vec<u8>> {
    let out = tmp("dec.yuv");
    let ok = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&out)
        .status()
        .ok()?
        .success();
    let data = if ok { std::fs::read(&out).ok() } else { None };
    let _ = std::fs::remove_file(&out);
    data
}

/// Like [`decode_to_yuv`], but `-fps_mode passthrough` so WebM streams that lack
/// `DefaultDuration` (and thus report 1k tbr) are not CFR-duplicated.
fn decode_to_yuv_passthrough(input: &std::path::Path) -> Option<Vec<u8>> {
    let out = tmp("dec_pass.yuv");
    let ok = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args([
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&out)
        .status()
        .ok()?
        .success();
    let data = if ok { std::fs::read(&out).ok() } else { None };
    let _ = std::fs::remove_file(&out);
    data
}

#[test]
fn webm_muxes_vp9_that_ffmpeg_decodes() {
    if !ffmpeg_ok() || !has_vp9_encoder() {
        eprintln!("skipping webm ffmpeg test: ffmpeg or libvpx-vp9 not available");
        return;
    }

    // 1. Encode a few deterministic VP9 keyframes into an IVF (dev oracle only).
    let ivf = tmp("src.ivf");
    let ok = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=3:duration=1",
            "-c:v",
            "libvpx-vp9",
            "-g",
            "1",
            "-f",
            "ivf",
        ])
        .arg(&ivf)
        .status()
        .expect("run ffmpeg")
        .success();
    assert!(ok, "ffmpeg failed to produce VP9 IVF");

    let ivf_bytes = std::fs::read(&ivf).unwrap();
    let (w, h, frames) = parse_ivf(&ivf_bytes);
    assert!(
        frames.len() >= 2,
        "expected multiple frames, got {}",
        frames.len()
    );
    eprintln!("re-muxing {} VP9 frames at {w}x{h}", frames.len());

    // 2. Re-container the identical VP9 frames with our muxer.
    let webm_frames: Vec<WebmFrame> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| WebmFrame {
            data: f,
            alpha: None,
            timecode: (i as u64) * 333,
            keyframe: true,
        })
        .collect();
    let webm = mux_webm(&WebmParams {
        width: w,
        height: h,
        codec: WebmCodec::Vp9,
        timecode_scale_ns: 1_000_000,
        frames: &webm_frames,
    });
    let webm_path = tmp("mine.webm");
    std::fs::write(&webm_path, &webm).unwrap();

    // 3. ffmpeg must open our .webm and decode it to the same pixels as the IVF.
    let ours = decode_to_yuv(&webm_path).expect("ffmpeg could not decode our .webm");
    let reference = decode_to_yuv(&ivf).expect("ffmpeg could not decode reference IVF");

    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&webm_path);

    assert!(!ours.is_empty(), "our .webm decoded to nothing");
    assert_eq!(
        ours.len(),
        reference.len(),
        "decoded byte count differs (container dropped/duplicated frames?)"
    );
    assert!(
        ours == reference,
        "our .webm decoded to different pixels than the source VP9"
    );
    eprintln!("conformant: our WebM container decodes identically to the source VP9");
}

/// Native VP9 gray keyframe → WebM → ffmpeg must decode to solid mid-gray
/// (Y=U=V=128). This is the VP9 analog of the HEVC `I_PCM` conformance test:
/// no interesting compression yet, but it proves the full header / bool-coder /
/// partition / skip / mux pipeline against an independent decoder.
#[test]
fn native_vp9_gray_webm_ffmpeg_decodes_mid_gray() {
    if !ffmpeg_ok() {
        eprintln!("skipping native VP9→WebM ffmpeg test: ffmpeg not available");
        return;
    }

    let (w, h, nframes) = (64u32, 64u32, 3u32);
    let webm = encode_gray_webm(w, h, nframes, 10);
    let webm_path = tmp("native_gray.webm");
    std::fs::write(&webm_path, &webm).unwrap();

    let decoded = decode_to_yuv(&webm_path).expect("ffmpeg could not decode native VP9 .webm");
    let _ = std::fs::remove_file(&webm_path);

    let frame_bytes = (w * h + 2 * (w / 2) * (h / 2)) as usize; // yuv420p
    assert_eq!(
        decoded.len(),
        frame_bytes * nframes as usize,
        "decoded byte count (got {}, want {}×{})",
        decoded.len(),
        frame_bytes,
        nframes
    );
    if let Some(pos) = decoded.iter().position(|&b| b != 128) {
        panic!(
            "expected solid mid-gray (128), got {} at byte {pos} (frame {})",
            decoded[pos],
            pos / frame_bytes
        );
    }
    eprintln!(
        "conformant: native VP9 gray → WebM decodes to mid-gray ({} frames, {} bytes .webm)",
        nframes,
        webm.len()
    );
}

/// Raw VP9 bitstream (IVF-wrapped for ffmpeg) must also decode to mid-gray —
/// isolates encoder bugs from container bugs.
#[test]
fn native_vp9_keyframe_ivf_ffmpeg_decodes_mid_gray() {
    if !ffmpeg_ok() {
        eprintln!("skipping native VP9 IVF ffmpeg test: ffmpeg not available");
        return;
    }

    let (w, h) = (64u32, 64u32);
    let frame = encode_intra_gray(w, h);

    // Minimal IVF: 32-byte header + one frame.
    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes()); // version
    ivf.extend_from_slice(&32u16.to_le_bytes()); // header length
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes()); // timebase denom
    ivf.extend_from_slice(&1u32.to_le_bytes()); // timebase num
    ivf.extend_from_slice(&1u32.to_le_bytes()); // frame count
    ivf.extend_from_slice(&0u32.to_le_bytes()); // unused
    ivf.extend_from_slice(&(frame.len() as u32).to_le_bytes());
    ivf.extend_from_slice(&0u64.to_le_bytes()); // timestamp
    ivf.extend_from_slice(&frame);

    let ivf_path = tmp("native.ivf");
    std::fs::write(&ivf_path, &ivf).unwrap();
    let decoded = decode_to_yuv(&ivf_path).expect("ffmpeg could not decode native VP9 IVF");
    let _ = std::fs::remove_file(&ivf_path);

    let frame_bytes = (w * h + 2 * (w / 2) * (h / 2)) as usize;
    assert_eq!(decoded.len(), frame_bytes);
    if let Some(pos) = decoded.iter().position(|&b| b != 128) {
        panic!(
            "expected solid mid-gray (128), got {} at byte {pos}",
            decoded[pos]
        );
    }
    eprintln!(
        "conformant: native VP9 keyframe IVF decodes to mid-gray ({} bytes)",
        frame.len()
    );
}

/// Intra with 4×4 DCT residual: ffmpeg must decode to exactly our
/// reconstruction — the VP9 analog of `tests/hevc_compress.rs`.
#[test]
fn native_vp9_residual_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 residual ffmpeg test: ffmpeg not available");
        return;
    }
    // 64×64 full SB, 80×48 partial SB edge, 8×8 minimum.
    for &(w, h) in &[(64u32, 64), (80, 48), (8, 8)] {
        assert_residual_matches_recon(w, h);
    }
}

fn assert_residual_matches_recon(w: u32, h: u32) {
    let mut frame = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            frame.y[(j * w + i) as usize] = ((i.wrapping_mul(3) + j.wrapping_mul(5)) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..ch {
        for i in 0..cw {
            frame.u[(j * cw + i) as usize] = (128 + (i as i32 - 16).clamp(-40, 40)) as u8;
            frame.v[(j * cw + i) as usize] = (128 + (j as i32 - 16).clamp(-40, 40)) as u8;
        }
    }

    let (bitstream, recon) = encode_intra_frame(&frame);

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    ivf.extend_from_slice(&(bitstream.len() as u32).to_le_bytes());
    ivf.extend_from_slice(&0u64.to_le_bytes());
    ivf.extend_from_slice(&bitstream);

    let ivf_path = tmp(&format!("residual_{w}x{h}.ivf"));
    std::fs::write(&ivf_path, &ivf).unwrap();
    let decoded = decode_to_yuv(&ivf_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode residual VP9 {w}x{h}"));
    let _ = std::fs::remove_file(&ivf_path);

    let mut expected = Vec::with_capacity(recon.y.len() + recon.u.len() + recon.v.len());
    expected.extend_from_slice(&recon.y);
    expected.extend_from_slice(&recon.u);
    expected.extend_from_slice(&recon.v);
    assert_eq!(decoded.len(), expected.len(), "{w}x{h} plane size mismatch");
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        let n = recon.y.len().min(16);
        panic!(
            "{w}x{h} residual mismatch at byte {pos}: decoded {} != recon {} (recon y[0..{n}]={:?}, decoded y[0..{n}]={:?})",
            decoded[pos],
            expected[pos],
            &recon.y[..n],
            &decoded[..n]
        );
    }

    let webm = encode_webm(&[&frame], 30);
    let webm_path = tmp(&format!("residual_{w}x{h}.webm"));
    std::fs::write(&webm_path, &webm).unwrap();
    let from_webm = decode_to_yuv(&webm_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode residual .webm {w}x{h}"));
    let _ = std::fs::remove_file(&webm_path);
    assert_eq!(
        from_webm, expected,
        "{w}x{h} WebM-muxed residual differs from IVF"
    );

    eprintln!(
        "conformant: VP9 residual {w}x{h} matches recon ({} bytes bitstream, {} bytes .webm)",
        bitstream.len(),
        webm.len()
    );
}

/// Decode first frame as `yuva420p` via libvpx-vp9 (needed to surface BlockAdditional alpha).
fn decode_to_yuva(input: &std::path::Path, w: u32, h: u32) -> Option<(Vec<u8>, Vec<u8>)> {
    let out = tmp("dec_yuva.yuv");
    let ok = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-c:v",
            "libvpx-vp9",
            "-i",
        ])
        .arg(input)
        .args(["-frames:v", "1", "-pix_fmt", "yuva420p", "-f", "rawvideo"])
        .arg(&out)
        .status()
        .ok()?
        .success();
    let data = if ok { std::fs::read(&out).ok() } else { None };
    let _ = std::fs::remove_file(&out);
    let data = data?;
    let y_sz = (w * h) as usize;
    let c_sz = (w * h / 4) as usize;
    let need = y_sz + c_sz + c_sz + y_sz;
    if data.len() < need {
        return None;
    }
    let yuv = data[..y_sz + c_sz + c_sz].to_vec();
    let alpha = data[y_sz + c_sz + c_sz..need].to_vec();
    Some((yuv, alpha))
}

/// Native VP9 + WebM alpha (`BlockAdditional`): ffmpeg/libvpx must decode colour
/// and alpha to our reconstructions.
#[test]
fn native_vp9_alpha_webm_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 alpha ffmpeg test: ffmpeg not available");
        return;
    }

    let (w, h) = (64u32, 48u32);
    let mut frame = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            frame.y[(j * w + i) as usize] = ((i.wrapping_mul(3) + j.wrapping_mul(5)) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..ch {
        for i in 0..cw {
            frame.u[(j * cw + i) as usize] = (128 + (i as i32 - 8).clamp(-40, 40)) as u8;
            frame.v[(j * cw + i) as usize] = (128 + (j as i32 - 8).clamp(-40, 40)) as u8;
        }
    }
    let mut alpha = vec![0u8; (w * h) as usize];
    for j in 0..h {
        for i in 0..w {
            alpha[(j * w + i) as usize] = ((i * 255) / (w - 1)) as u8;
        }
    }
    frame.alpha = Some(alpha.clone());

    // Expected recons: colour frame + alpha-as-luma frame.
    let (_, color_recon) = encode_intra_frame(&frame);
    let mut alpha_src = Yuv420Frame::new(w, h);
    alpha_src.y.copy_from_slice(&alpha);
    alpha_src.u.fill(128);
    alpha_src.v.fill(128);
    let (_, alpha_recon) = encode_intra_frame(&alpha_src);

    let webm = encode_webm(&[&frame], 30);
    assert!(
        webm.windows(2).any(|x| x == [0x53, 0xC0]),
        "AlphaMode missing from .webm"
    );
    assert!(
        webm.windows(2).any(|x| x == [0x75, 0xA1]),
        "BlockAdditions missing from .webm"
    );

    let webm_path = tmp("alpha.webm");
    std::fs::write(&webm_path, &webm).unwrap();
    let (decoded_yuv, decoded_a) =
        decode_to_yuva(&webm_path, w, h).expect("ffmpeg/libvpx could not decode alpha .webm");
    let _ = std::fs::remove_file(&webm_path);

    let mut expected_yuv = Vec::new();
    expected_yuv.extend_from_slice(&color_recon.y);
    expected_yuv.extend_from_slice(&color_recon.u);
    expected_yuv.extend_from_slice(&color_recon.v);
    assert_eq!(decoded_yuv, expected_yuv, "colour planes mismatch");
    assert_eq!(decoded_a, alpha_recon.y, "alpha plane mismatch");

    eprintln!(
        "conformant: VP9 alpha WebM matches recon ({} bytes .webm, {}x{})",
        webm.len(),
        w,
        h
    );
}

/// Keyframe + ZEROMV/skip P-frame: ffmpeg must decode both frames to the keyframe recon.
#[test]
fn native_vp9_inter_zeromv_skip_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 inter ffmpeg test: ffmpeg not available");
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48), (8, 8)] {
        assert_inter_zeromv_matches_recon(w, h);
    }
}

fn assert_inter_zeromv_matches_recon(w: u32, h: u32) {
    let mut frame = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            frame.y[(j * w + i) as usize] = ((i.wrapping_mul(7) + j.wrapping_mul(11)) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..ch {
        for i in 0..cw {
            frame.u[(j * cw + i) as usize] = (128 + (i as i32 - 8).clamp(-30, 30)) as u8;
            frame.v[(j * cw + i) as usize] = (128 + (j as i32 - 8).clamp(-30, 30)) as u8;
        }
    }

    let (key_bits, key_recon) = encode_intra_frame(&frame);
    let (p_bits, p_recon) = encode_inter_zeromv_skip(&key_recon);
    // Loop filter runs on the P-frame; recon is post-LF (not a raw copy of the key).

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&2u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (ts, bits) in [(0u64, &key_bits), (1u64, &p_bits)] {
        ivf.extend_from_slice(&(bits.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&ts.to_le_bytes());
        ivf.extend_from_slice(bits);
    }

    let ivf_path = tmp(&format!("inter_{w}x{h}.ivf"));
    std::fs::write(&ivf_path, &ivf).unwrap();
    let decoded = decode_to_yuv(&ivf_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode inter VP9 {w}x{h}"));
    let _ = std::fs::remove_file(&ivf_path);

    let mut expected = Vec::new();
    expected.extend_from_slice(&key_recon.y);
    expected.extend_from_slice(&key_recon.u);
    expected.extend_from_slice(&key_recon.v);
    let frame_bytes = expected.len();
    expected.extend_from_slice(&p_recon.y);
    expected.extend_from_slice(&p_recon.u);
    expected.extend_from_slice(&p_recon.v);
    assert_eq!(decoded.len(), frame_bytes * 2, "{w}x{h} expected 2 frames");
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{w}x{h} ZEROMV mismatch at byte {pos}: decoded {} != recon {} (frame {})",
            decoded[pos],
            expected[pos],
            pos / frame_bytes
        );
    }
    let webm = mux_webm(&WebmParams {
        width: w,
        height: h,
        codec: WebmCodec::Vp9,
        timecode_scale_ns: 1_000_000,
        frames: &[
            WebmFrame {
                data: &key_bits,
                alpha: None,
                timecode: 0,
                keyframe: true,
            },
            WebmFrame {
                data: &p_bits,
                alpha: None,
                timecode: 100,
                keyframe: false,
            },
        ],
    });
    let webm_path = tmp(&format!("inter_{w}x{h}.webm"));
    std::fs::write(&webm_path, &webm).unwrap();
    // Non-keyframe WebM has 1k tbn and no DefaultDuration; without passthrough
    // ffmpeg CFR-duplicates to one frame per ms of Duration.
    let from_webm = decode_to_yuv_passthrough(&webm_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode inter .webm {w}x{h}"));
    let _ = std::fs::remove_file(&webm_path);
    assert_eq!(from_webm, decoded, "{w}x{h} WebM inter differs from IVF");

    eprintln!(
        "conformant: VP9 inter ZEROMV/skip {w}x{h} (key {} + P {} bytes)",
        key_bits.len(),
        p_bits.len()
    );
}

/// Keyframe + NEWMV/skip P-frame (constant +2 luma pel = 16 in Q3): ffmpeg must
/// match our motion-compensated recon on the P-frame.
#[test]
fn native_vp9_inter_newmv_skip_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 NEWMV ffmpeg test: ffmpeg not available");
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48), (8, 8)] {
        assert_inter_newmv_matches_recon(w, h, 0, 16);
    }
}

/// Half-pel horizontal NEWMV/skip (MV col = 4 in 1/8-pel): ffmpeg EIGHTTAP match.
#[test]
fn native_vp9_inter_newmv_halfpel_skip_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 NEWMV half-pel ffmpeg test: ffmpeg not available");
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48), (8, 8), (16, 16)] {
        assert_inter_newmv_matches_recon(w, h, 0, 4);
    }
}

fn assert_inter_newmv_matches_recon(w: u32, h: u32, mv_row: i16, mv_col: i16) {
    let mut frame = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            frame.y[(j * w + i) as usize] = ((i.wrapping_mul(5) + j.wrapping_mul(9)) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..ch {
        for i in 0..cw {
            frame.u[(j * cw + i) as usize] = (128 + (i as i32 - 10).clamp(-40, 40)) as u8;
            frame.v[(j * cw + i) as usize] = (128 + (j as i32 - 10).clamp(-40, 40)) as u8;
        }
    }

    let (key_bits, key_recon) = encode_intra_frame(&frame);
    let (p_bits, p_recon) = encode_inter_newmv_skip(&key_recon, mv_row, mv_col);

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&2u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (ts, bits) in [(0u64, &key_bits), (1u64, &p_bits)] {
        ivf.extend_from_slice(&(bits.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&ts.to_le_bytes());
        ivf.extend_from_slice(bits);
    }

    let ivf_path = tmp(&format!("newmv_{w}x{h}.ivf"));
    std::fs::write(&ivf_path, &ivf).unwrap();
    let decoded = decode_to_yuv(&ivf_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode NEWMV VP9 {w}x{h}"));
    let _ = std::fs::remove_file(&ivf_path);

    let mut expected = Vec::new();
    expected.extend_from_slice(&key_recon.y);
    expected.extend_from_slice(&key_recon.u);
    expected.extend_from_slice(&key_recon.v);
    let frame_bytes = expected.len();
    expected.extend_from_slice(&p_recon.y);
    expected.extend_from_slice(&p_recon.u);
    expected.extend_from_slice(&p_recon.v);

    assert_eq!(decoded.len(), frame_bytes * 2, "{w}x{h} expected 2 frames");
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{w}x{h} NEWMV mismatch at byte {pos}: decoded {} != recon {} (frame {})",
            decoded[pos],
            expected[pos],
            pos / frame_bytes
        );
    }

    let webm = mux_webm(&WebmParams {
        width: w,
        height: h,
        codec: WebmCodec::Vp9,
        timecode_scale_ns: 1_000_000,
        frames: &[
            WebmFrame {
                data: &key_bits,
                alpha: None,
                timecode: 0,
                keyframe: true,
            },
            WebmFrame {
                data: &p_bits,
                alpha: None,
                timecode: 100,
                keyframe: false,
            },
        ],
    });
    let webm_path = tmp(&format!("newmv_{w}x{h}.webm"));
    std::fs::write(&webm_path, &webm).unwrap();
    let from_webm = decode_to_yuv_passthrough(&webm_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode NEWMV .webm {w}x{h}"));
    let _ = std::fs::remove_file(&webm_path);
    assert_eq!(from_webm, decoded, "{w}x{h} WebM NEWMV differs from IVF");

    eprintln!(
        "conformant: VP9 inter NEWMV/skip {w}x{h} mv=({mv_row},{mv_col}) (key {} + P {} bytes)",
        key_bits.len(),
        p_bits.len()
    );
}

/// Keyframe + ZEROMV residual P-frame: ffmpeg must match our recon on both frames.
#[test]
fn native_vp9_inter_residual_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 inter residual ffmpeg test: ffmpeg not available");
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48), (8, 8)] {
        assert_inter_residual_matches_recon(w, h);
    }
}

fn assert_inter_residual_matches_recon(w: u32, h: u32) {
    let mut key_src = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            key_src.y[(j * w + i) as usize] = ((i.wrapping_mul(2) + j) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..ch {
        for i in 0..cw {
            key_src.u[(j * cw + i) as usize] = 128;
            key_src.v[(j * cw + i) as usize] = 128;
        }
    }

    let mut p_src = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            p_src.y[(j * w + i) as usize] =
                ((i.wrapping_mul(7) + j.wrapping_mul(13) + 40) & 0xFF) as u8;
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            p_src.u[(j * cw + i) as usize] = (100 + (i % 40) as u8).min(200);
            p_src.v[(j * cw + i) as usize] = (140 + (j % 30) as u8).min(220);
        }
    }

    let (key_bits, key_recon) = encode_intra_frame(&key_src);
    let (p_bits, p_recon) = encode_inter_residual(&p_src, &key_recon);

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&2u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (ts, bits) in [(0u64, &key_bits), (1u64, &p_bits)] {
        ivf.extend_from_slice(&(bits.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&ts.to_le_bytes());
        ivf.extend_from_slice(bits);
    }

    let ivf_path = tmp(&format!("inter_res_{w}x{h}.ivf"));
    std::fs::write(&ivf_path, &ivf).unwrap();
    let decoded = decode_to_yuv(&ivf_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode inter residual VP9 {w}x{h}"));
    let _ = std::fs::remove_file(&ivf_path);

    let mut expected = Vec::new();
    expected.extend_from_slice(&key_recon.y);
    expected.extend_from_slice(&key_recon.u);
    expected.extend_from_slice(&key_recon.v);
    let frame_bytes = expected.len();
    expected.extend_from_slice(&p_recon.y);
    expected.extend_from_slice(&p_recon.u);
    expected.extend_from_slice(&p_recon.v);

    assert_eq!(decoded.len(), frame_bytes * 2, "{w}x{h} expected 2 frames");
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{w}x{h} inter residual mismatch at byte {pos}: decoded {} != recon {} (frame {})",
            decoded[pos],
            expected[pos],
            pos / frame_bytes
        );
    }

    let webm = mux_webm(&WebmParams {
        width: w,
        height: h,
        codec: WebmCodec::Vp9,
        timecode_scale_ns: 1_000_000,
        frames: &[
            WebmFrame {
                data: &key_bits,
                alpha: None,
                timecode: 0,
                keyframe: true,
            },
            WebmFrame {
                data: &p_bits,
                alpha: None,
                timecode: 100,
                keyframe: false,
            },
        ],
    });
    let webm_path = tmp(&format!("inter_res_{w}x{h}.webm"));
    std::fs::write(&webm_path, &webm).unwrap();
    let from_webm = decode_to_yuv_passthrough(&webm_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode inter residual .webm {w}x{h}"));
    let _ = std::fs::remove_file(&webm_path);
    assert_eq!(
        from_webm, decoded,
        "{w}x{h} WebM inter residual differs from IVF"
    );

    eprintln!(
        "conformant: VP9 inter residual {w}x{h} (key {} + P {} bytes)",
        key_bits.len(),
        p_bits.len()
    );
}

/// Keyframe + NEWMV residual P-frame (shifted content): ffmpeg must match recon.
#[test]
fn native_vp9_inter_newmv_residual_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 NEWMV residual ffmpeg test: ffmpeg not available");
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48), (8, 8)] {
        assert_inter_newmv_residual_matches_recon(w, h, 0, 16);
    }
}

/// Half-pel NEWMV residual: source ≈ EIGHTTAP MC + small bump.
#[test]
fn native_vp9_inter_newmv_halfpel_residual_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 NEWMV half-pel residual ffmpeg test: ffmpeg not available");
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48), (8, 8), (16, 16)] {
        assert_inter_newmv_residual_matches_recon(w, h, 0, 4);
    }
}

fn assert_inter_newmv_residual_matches_recon(w: u32, h: u32, mv_row: i16, mv_col: i16) {
    let mut key_src = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            key_src.y[(j * w + i) as usize] =
                ((i.wrapping_mul(3) + j.wrapping_mul(5)) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..ch {
        for i in 0..cw {
            key_src.u[(j * cw + i) as usize] = (110 + (i % 50) as u8).min(200);
            key_src.v[(j * cw + i) as usize] = (130 + (j % 40) as u8).min(210);
        }
    }
    let (key_bits, key_recon) = encode_intra_frame(&key_src);

    // Ideal prediction for this MV, then a small residual bump on luma.
    let (_, mc) = encode_inter_newmv_skip(&key_recon, mv_row, mv_col);
    let mut p_src = Yuv420Frame::new(w, h);
    for (i, &y) in mc.y.iter().enumerate() {
        p_src.y[i] = y.saturating_add((i as u8) & 3);
    }
    p_src.u.copy_from_slice(&mc.u);
    p_src.v.copy_from_slice(&mc.v);

    let (p_bits, p_recon) = encode_inter_newmv_residual(&p_src, &key_recon, mv_row, mv_col);

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&2u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (ts, bits) in [(0u64, &key_bits), (1u64, &p_bits)] {
        ivf.extend_from_slice(&(bits.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&ts.to_le_bytes());
        ivf.extend_from_slice(bits);
    }

    let ivf_path = tmp(&format!("newmv_res_{w}x{h}.ivf"));
    std::fs::write(&ivf_path, &ivf).unwrap();
    let decoded = decode_to_yuv(&ivf_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode NEWMV residual VP9 {w}x{h}"));
    let _ = std::fs::remove_file(&ivf_path);

    let mut expected = Vec::new();
    expected.extend_from_slice(&key_recon.y);
    expected.extend_from_slice(&key_recon.u);
    expected.extend_from_slice(&key_recon.v);
    let frame_bytes = expected.len();
    expected.extend_from_slice(&p_recon.y);
    expected.extend_from_slice(&p_recon.u);
    expected.extend_from_slice(&p_recon.v);

    assert_eq!(decoded.len(), frame_bytes * 2, "{w}x{h} expected 2 frames");
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{w}x{h} NEWMV residual mismatch at byte {pos}: decoded {} != recon {} (frame {})",
            decoded[pos],
            expected[pos],
            pos / frame_bytes
        );
    }

    let webm = mux_webm(&WebmParams {
        width: w,
        height: h,
        codec: WebmCodec::Vp9,
        timecode_scale_ns: 1_000_000,
        frames: &[
            WebmFrame {
                data: &key_bits,
                alpha: None,
                timecode: 0,
                keyframe: true,
            },
            WebmFrame {
                data: &p_bits,
                alpha: None,
                timecode: 100,
                keyframe: false,
            },
        ],
    });
    let webm_path = tmp(&format!("newmv_res_{w}x{h}.webm"));
    std::fs::write(&webm_path, &webm).unwrap();
    let from_webm = decode_to_yuv_passthrough(&webm_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode NEWMV residual .webm {w}x{h}"));
    let _ = std::fs::remove_file(&webm_path);
    assert_eq!(
        from_webm, decoded,
        "{w}x{h} WebM NEWMV residual differs from IVF"
    );

    eprintln!(
        "conformant: VP9 inter NEWMV residual {w}x{h} mv=({mv_row},{mv_col}) (key {} + P {} bytes)",
        key_bits.len(),
        p_bits.len()
    );
}

/// Keyframe + per-block ME residual P-frame (mixed shifts): ffmpeg must match recon.
#[test]
fn native_vp9_inter_me_frame_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 ME-frame ffmpeg test: ffmpeg not available");
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48), (16, 16)] {
        assert_inter_me_matches_recon(w, h);
    }
}

fn assert_inter_me_matches_recon(w: u32, h: u32) {
    // Exercises ALLOW_32X32: large PARTITION_NONE → luma TX_32X32 (and UV
    // TX_16/TX_32 per block size), plane-major U-then-V tokens. UV residual is
    // intentional — p_src chroma matches the key recon while luma is MC'd.
    let mut key_src = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            key_src.y[(j * w + i) as usize] =
                ((i.wrapping_mul(3) + j.wrapping_mul(5)) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..ch {
        for i in 0..cw {
            key_src.u[(j * cw + i) as usize] = (110 + (i % 50) as u8).min(200);
            key_src.v[(j * cw + i) as usize] = (130 + (j % 40) as u8).min(210);
        }
    }
    let (key_bits, key_recon) = encode_intra_frame(&key_src);

    // Left half: +2 pel horizontal; right half: +1 pel vertical (full-pel Q3).
    let mut p_src = Yuv420Frame::new(w, h);
    p_src.u.copy_from_slice(&key_recon.u);
    p_src.v.copy_from_slice(&key_recon.v);
    let mid = (w / 2) as usize;
    for j in 0..h as usize {
        for i in 0..w as usize {
            let (sx, sy) = if i < mid {
                ((i + 2).min(w as usize - 1), j)
            } else {
                (i, (j + 1).min(h as usize - 1))
            };
            p_src.y[j * w as usize + i] = key_recon.y[sy * w as usize + sx];
        }
    }

    let (p_bits, p_recon) = encode_inter_frame(&p_src, &key_recon);

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&2u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (ts, bits) in [(0u64, &key_bits), (1u64, &p_bits)] {
        ivf.extend_from_slice(&(bits.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&ts.to_le_bytes());
        ivf.extend_from_slice(bits);
    }

    let ivf_path = tmp(&format!("me_{w}x{h}.ivf"));
    std::fs::write(&ivf_path, &ivf).unwrap();
    let decoded = decode_to_yuv(&ivf_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode ME VP9 {w}x{h}"));
    let _ = std::fs::remove_file(&ivf_path);

    let mut expected = Vec::new();
    expected.extend_from_slice(&key_recon.y);
    expected.extend_from_slice(&key_recon.u);
    expected.extend_from_slice(&key_recon.v);
    let frame_bytes = expected.len();
    expected.extend_from_slice(&p_recon.y);
    expected.extend_from_slice(&p_recon.u);
    expected.extend_from_slice(&p_recon.v);

    assert_eq!(decoded.len(), frame_bytes * 2, "{w}x{h} expected 2 frames");
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{w}x{h} ME mismatch at byte {pos}: decoded {} != recon {} (frame {})",
            decoded[pos],
            expected[pos],
            pos / frame_bytes
        );
    }

    let webm = mux_webm(&WebmParams {
        width: w,
        height: h,
        codec: WebmCodec::Vp9,
        timecode_scale_ns: 1_000_000,
        frames: &[
            WebmFrame {
                data: &key_bits,
                alpha: None,
                timecode: 0,
                keyframe: true,
            },
            WebmFrame {
                data: &p_bits,
                alpha: None,
                timecode: 100,
                keyframe: false,
            },
        ],
    });
    let webm_path = tmp(&format!("me_{w}x{h}.webm"));
    std::fs::write(&webm_path, &webm).unwrap();
    let from_webm = decode_to_yuv_passthrough(&webm_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode ME .webm {w}x{h}"));
    let _ = std::fs::remove_file(&webm_path);
    assert_eq!(from_webm, decoded, "{w}x{h} WebM ME differs from IVF");

    eprintln!(
        "conformant: VP9 inter ME frame {w}x{h} (key {} + P {} bytes)",
        key_bits.len(),
        p_bits.len()
    );
}

/// Key + P1 (LAST shift) + P2 (GOLDEN ≈ key): ffmpeg must match 3-frame recon.
#[test]
fn native_vp9_inter_golden_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 GOLDEN ffmpeg test: ffmpeg not available");
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48)] {
        assert_inter_golden_matches_recon(w, h);
    }
}

fn assert_inter_golden_matches_recon(w: u32, h: u32) {
    let mut key_src = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            key_src.y[(j * w + i) as usize] =
                ((i.wrapping_mul(3) + j.wrapping_mul(5)) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..ch {
        for i in 0..cw {
            key_src.u[(j * cw + i) as usize] = (110 + (i % 50) as u8).min(200);
            key_src.v[(j * cw + i) as usize] = (130 + (j % 40) as u8).min(210);
        }
    }
    let (key_bits, key_recon) = encode_intra_frame(&key_src);

    // P1: +2 pel horizontal → LAST drifts; GOLDEN (slot 1) stays at key.
    let mut p1_src = Yuv420Frame::new(w, h);
    p1_src.u.copy_from_slice(&key_recon.u);
    p1_src.v.copy_from_slice(&key_recon.v);
    for j in 0..h as usize {
        for i in 0..w as usize {
            let sx = (i + 2).min(w as usize - 1);
            p1_src.y[j * w as usize + i] = key_recon.y[j * w as usize + sx];
        }
    }
    let (p1_bits, last) = encode_inter_frame(&p1_src, &key_recon);

    // P2: source ≈ key again → encoder should prefer GOLDEN.
    let mut p2_src = Yuv420Frame::new(w, h);
    p2_src.y.copy_from_slice(&key_recon.y);
    p2_src.u.copy_from_slice(&key_recon.u);
    p2_src.v.copy_from_slice(&key_recon.v);
    let (p2_bits, p2_recon) = encode_inter_frame_golden(&p2_src, &last, &key_recon);

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&3u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (ts, bits) in [(0u64, &key_bits), (1u64, &p1_bits), (2u64, &p2_bits)] {
        ivf.extend_from_slice(&(bits.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&ts.to_le_bytes());
        ivf.extend_from_slice(bits);
    }

    let ivf_path = tmp(&format!("golden_{w}x{h}.ivf"));
    std::fs::write(&ivf_path, &ivf).unwrap();
    let decoded = decode_to_yuv(&ivf_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode GOLDEN VP9 {w}x{h}"));
    let _ = std::fs::remove_file(&ivf_path);

    let mut expected = Vec::new();
    expected.extend_from_slice(&key_recon.y);
    expected.extend_from_slice(&key_recon.u);
    expected.extend_from_slice(&key_recon.v);
    let frame_bytes = expected.len();
    expected.extend_from_slice(&last.y);
    expected.extend_from_slice(&last.u);
    expected.extend_from_slice(&last.v);
    expected.extend_from_slice(&p2_recon.y);
    expected.extend_from_slice(&p2_recon.u);
    expected.extend_from_slice(&p2_recon.v);

    assert_eq!(decoded.len(), frame_bytes * 3, "{w}x{h} expected 3 frames");
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{w}x{h} GOLDEN mismatch at byte {pos}: decoded {} != recon {} (frame {})",
            decoded[pos],
            expected[pos],
            pos / frame_bytes
        );
    }

    let webm = mux_webm(&WebmParams {
        width: w,
        height: h,
        codec: WebmCodec::Vp9,
        timecode_scale_ns: 1_000_000,
        frames: &[
            WebmFrame {
                data: &key_bits,
                alpha: None,
                timecode: 0,
                keyframe: true,
            },
            WebmFrame {
                data: &p1_bits,
                alpha: None,
                timecode: 100,
                keyframe: false,
            },
            WebmFrame {
                data: &p2_bits,
                alpha: None,
                timecode: 200,
                keyframe: false,
            },
        ],
    });
    let webm_path = tmp(&format!("golden_{w}x{h}.webm"));
    std::fs::write(&webm_path, &webm).unwrap();
    let from_webm = decode_to_yuv_passthrough(&webm_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode GOLDEN .webm {w}x{h}"));
    let _ = std::fs::remove_file(&webm_path);
    assert_eq!(from_webm, decoded, "{w}x{h} WebM GOLDEN differs from IVF");

    eprintln!(
        "conformant: VP9 GOLDEN {w}x{h} (key {} + P1 {} + P2 {} bytes)",
        key_bits.len(),
        p1_bits.len(),
        p2_bits.len()
    );
}

/// Key + P1 (refresh ALTREF only) + P2 (prefer ALTREF): ffmpeg must match recon.
#[test]
fn native_vp9_inter_altref_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 ALTREF ffmpeg test: ffmpeg not available");
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48)] {
        assert_inter_altref_matches_recon(w, h);
    }
}

fn assert_inter_altref_matches_recon(w: u32, h: u32) {
    let mut key_src = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            key_src.y[(j * w + i) as usize] =
                ((i.wrapping_mul(3) + j.wrapping_mul(5)) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..ch {
        for i in 0..cw {
            key_src.u[(j * cw + i) as usize] = (110 + (i % 50) as u8).min(200);
            key_src.v[(j * cw + i) as usize] = (130 + (j % 40) as u8).min(210);
        }
    }
    let (key_bits, key_recon) = encode_intra_frame(&key_src);

    // P1: +2 pel shift into ALTREF slot only (0x04). LAST/GOLDEN stay at key.
    let mut p1_src = Yuv420Frame::new(w, h);
    p1_src.u.copy_from_slice(&key_recon.u);
    p1_src.v.copy_from_slice(&key_recon.v);
    for j in 0..h as usize {
        for i in 0..w as usize {
            let sx = (i + 2).min(w as usize - 1);
            p1_src.y[j * w as usize + i] = key_recon.y[j * w as usize + sx];
        }
    }
    let (p1_bits, alt) = encode_inter_frame_refresh(&p1_src, &key_recon, 0x04);

    // P2: source ≈ alt → encoder should prefer ALTREF over LAST/GOLDEN (= key).
    let mut p2_src = Yuv420Frame::new(w, h);
    p2_src.y.copy_from_slice(&alt.y);
    p2_src.u.copy_from_slice(&alt.u);
    p2_src.v.copy_from_slice(&alt.v);
    let (p2_bits, p2_recon) = encode_inter_frame_altref(&p2_src, &key_recon, &key_recon, &alt);

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&3u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (ts, bits) in [(0u64, &key_bits), (1u64, &p1_bits), (2u64, &p2_bits)] {
        ivf.extend_from_slice(&(bits.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&ts.to_le_bytes());
        ivf.extend_from_slice(bits);
    }

    let ivf_path = tmp(&format!("altref_{w}x{h}.ivf"));
    std::fs::write(&ivf_path, &ivf).unwrap();
    let decoded = decode_to_yuv(&ivf_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode ALTREF VP9 {w}x{h}"));
    let _ = std::fs::remove_file(&ivf_path);

    let mut expected = Vec::new();
    expected.extend_from_slice(&key_recon.y);
    expected.extend_from_slice(&key_recon.u);
    expected.extend_from_slice(&key_recon.v);
    let frame_bytes = expected.len();
    expected.extend_from_slice(&alt.y);
    expected.extend_from_slice(&alt.u);
    expected.extend_from_slice(&alt.v);
    expected.extend_from_slice(&p2_recon.y);
    expected.extend_from_slice(&p2_recon.u);
    expected.extend_from_slice(&p2_recon.v);

    assert_eq!(decoded.len(), frame_bytes * 3, "{w}x{h} expected 3 frames");
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{w}x{h} ALTREF mismatch at byte {pos}: decoded {} != recon {} (frame {})",
            decoded[pos],
            expected[pos],
            pos / frame_bytes
        );
    }

    let webm = mux_webm(&WebmParams {
        width: w,
        height: h,
        codec: WebmCodec::Vp9,
        timecode_scale_ns: 1_000_000,
        frames: &[
            WebmFrame {
                data: &key_bits,
                alpha: None,
                timecode: 0,
                keyframe: true,
            },
            WebmFrame {
                data: &p1_bits,
                alpha: None,
                timecode: 100,
                keyframe: false,
            },
            WebmFrame {
                data: &p2_bits,
                alpha: None,
                timecode: 200,
                keyframe: false,
            },
        ],
    });
    let webm_path = tmp(&format!("altref_{w}x{h}.webm"));
    std::fs::write(&webm_path, &webm).unwrap();
    let from_webm = decode_to_yuv_passthrough(&webm_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode ALTREF .webm {w}x{h}"));
    let _ = std::fs::remove_file(&webm_path);
    assert_eq!(from_webm, decoded, "{w}x{h} WebM ALTREF differs from IVF");

    eprintln!(
        "conformant: VP9 ALTREF {w}x{h} (key {} + P1 {} + P2 {} bytes)",
        key_bits.len(),
        p1_bits.len(),
        p2_bits.len()
    );
}

#[test]
fn native_vp9_inter_compound_matches_recon() {
    if !ffmpeg_ok() {
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48)] {
        assert_inter_compound_matches_recon(w, h);
    }
}

fn assert_inter_compound_matches_recon(w: u32, h: u32) {
    let mut key_src = Yuv420Frame::new(w, h);
    for j in 0..h {
        for i in 0..w {
            key_src.y[(j * w + i) as usize] =
                ((i.wrapping_mul(7) + j.wrapping_mul(11)) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..ch {
        for i in 0..cw {
            key_src.u[(j * cw + i) as usize] = (100 + (i % 40) as u8).min(200);
            key_src.v[(j * cw + i) as usize] = (140 + (j % 30) as u8).min(210);
        }
    }
    let (key_bits, key_recon) = encode_intra_frame(&key_src);

    // P1: +4 pel shift into ALTREF only.
    let mut p1_src = Yuv420Frame::new(w, h);
    p1_src.u.copy_from_slice(&key_recon.u);
    p1_src.v.copy_from_slice(&key_recon.v);
    for j in 0..h as usize {
        for i in 0..w as usize {
            let sx = (i + 4).min(w as usize - 1);
            p1_src.y[j * w as usize + i] = key_recon.y[j * w as usize + sx];
        }
    }
    let (p1_bits, alt) = encode_inter_frame_refresh(&p1_src, &key_recon, 0x04);

    // P2: source = avg(LAST=key, ALTREF=alt) → compound should beat either single ref.
    let mut p2_src = Yuv420Frame::new(w, h);
    for i in 0..key_recon.y.len() {
        p2_src.y[i] = ((u16::from(key_recon.y[i]) + u16::from(alt.y[i]) + 1) >> 1) as u8;
    }
    for i in 0..key_recon.u.len() {
        p2_src.u[i] = ((u16::from(key_recon.u[i]) + u16::from(alt.u[i]) + 1) >> 1) as u8;
        p2_src.v[i] = ((u16::from(key_recon.v[i]) + u16::from(alt.v[i]) + 1) >> 1) as u8;
    }
    let (p2_bits, p2_recon) = encode_inter_frame_compound(&p2_src, &key_recon, &key_recon, &alt);

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&3u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (ts, bits) in [(0u64, &key_bits), (1u64, &p1_bits), (2u64, &p2_bits)] {
        ivf.extend_from_slice(&(bits.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&ts.to_le_bytes());
        ivf.extend_from_slice(bits);
    }

    let ivf_path = tmp(&format!("compound_{w}x{h}.ivf"));
    std::fs::write(&ivf_path, &ivf).unwrap();
    let decoded = decode_to_yuv(&ivf_path)
        .unwrap_or_else(|| panic!("ffmpeg could not decode compound VP9 {w}x{h}"));
    let _ = std::fs::remove_file(&ivf_path);

    let mut expected = Vec::new();
    expected.extend_from_slice(&key_recon.y);
    expected.extend_from_slice(&key_recon.u);
    expected.extend_from_slice(&key_recon.v);
    let frame_bytes = expected.len();
    expected.extend_from_slice(&alt.y);
    expected.extend_from_slice(&alt.u);
    expected.extend_from_slice(&alt.v);
    expected.extend_from_slice(&p2_recon.y);
    expected.extend_from_slice(&p2_recon.u);
    expected.extend_from_slice(&p2_recon.v);

    assert_eq!(decoded.len(), frame_bytes * 3, "{w}x{h} expected 3 frames");
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{w}x{h} compound mismatch at byte {pos}: decoded {} != recon {} (frame {})",
            decoded[pos],
            expected[pos],
            pos / frame_bytes
        );
    }
    eprintln!(
        "conformant: VP9 compound {w}x{h} (key {} + P1 {} + P2 {} bytes)",
        key_bits.len(),
        p1_bits.len(),
        p2_bits.len()
    );
}

#[test]
fn native_vp9_inter_newmv_neg_col_matches_recon() {
    if !ffmpeg_ok() {
        return;
    }
    for &(w, h) in &[(64u32, 64), (80, 48)] {
        assert_inter_newmv_matches_recon(w, h, 0, -16);
    }
}

/// 512×64 forces `log2_tile_cols = 1` (two columns). ffmpeg must match recon.
#[test]
fn native_vp9_inter_two_tile_cols_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 two-tile-cols ffmpeg test: ffmpeg not available");
        return;
    }
    assert_inter_me_matches_recon(512, 64);
}

/// Segmentation map + `SEG_LVL_ALT_Q` on the right half: ffmpeg must match recon.
#[test]
fn native_vp9_inter_segmentation_alt_q_matches_recon() {
    if !ffmpeg_ok() {
        eprintln!("skipping VP9 segmentation ffmpeg test: ffmpeg not available");
        return;
    }
    // Residual ME path so ALT_Q actually changes decoded samples on the right.
    assert_inter_me_matches_recon(64, 64);
    assert_inter_me_matches_recon(80, 48);
}
