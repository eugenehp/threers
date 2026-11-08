//! The MP3 encoder against the decoders: this crate's own (a port of
//! mpg123) and, where installed, ffmpeg, LAME and Apple's.

use std::process::{Command, Stdio};

use threers_codec::mp3::encoder::{encode_mp3, Mp3Config, Mp3Encoder, DELAY, FRAME};

/// A signal with no repeats, so the lag between two copies is unambiguous:
/// low-passed noise under a sweeping tone.
fn aperiodic(sample_rate: u32, seconds: f64, seed: u32) -> Vec<f32> {
    let mut state = seed.max(1);
    let n = (seconds * f64::from(sample_rate)) as usize;
    let mut low = 0.0f32;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state >> 8) as f32 / 8_388_608.0 - 1.0;
            low += 0.15 * (noise - low);
            let t = i as f64 / f64::from(sample_rate);
            0.5 * low + 0.2 * (2.0 * std::f64::consts::PI * (200.0 * t + 300.0 * t * t)).sin() as f32
        })
        .collect()
}

/// Raw frames only: no Info frame, so a decoder cannot trim anything.
fn encode_raw(planar: &[Vec<f32>], cfg: Mp3Config) -> Vec<u8> {
    let mut enc = Mp3Encoder::new(cfg).unwrap();
    let views: Vec<&[f32]> = planar.iter().map(Vec::as_slice).collect();
    let mut out = enc.push(&views).unwrap();
    out.extend(enc.finish());
    out
}

fn decode(bytes: &[u8]) -> Vec<Vec<f32>> {
    let m = threers_codec::mp3::decode(bytes).unwrap();
    (0..m.channels).map(|c| m.samples.iter().skip(c).step_by(m.channels).copied().collect()).collect()
}

fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut ab, mut aa, mut noise) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        ab += f64::from(*x) * f64::from(*y);
        aa += f64::from(*x) * f64::from(*x);
        noise += (f64::from(*y) - f64::from(*x)).powi(2);
    }
    (ab / aa, 10.0 * (aa / noise.max(1e-30)).log10())
}

/// The shift of `decoded` against `original` (positive: decoded starts late).
fn find_lag(original: &[f32], decoded: &[f32]) -> i64 {
    let (a, len) = (6_000usize, 8_000usize);
    (-2_000i64..3_000)
        .max_by_key(|&d| {
            let start = (a as i64 + d) as usize;
            let dot: f64 = original[a..a + len].iter().zip(&decoded[start..start + len]).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
            (dot * 1e6) as i64
        })
        .unwrap()
}

fn stereo(sample_rate: u32, seconds: f64) -> Vec<Vec<f32>> {
    let a = aperiodic(sample_rate, seconds, 3);
    let b = aperiodic(sample_rate, seconds, 4);
    // Mostly common to both channels, as in a mix.
    vec![a.iter().zip(&b).map(|(x, y)| x + 0.15 * y).collect(), a.iter().zip(&b).map(|(x, y)| x - 0.15 * y).collect()]
}

/// Gain and SNR of a decoded raw stream against its input, after the delay.
fn raw_quality(input: &[f32], decoded: &[f32]) -> (i64, f64, f64) {
    let lag = find_lag(input, decoded);
    let n = input.len().min(decoded.len() - lag as usize) - 4 * FRAME;
    let (gain, snr) = compare(&input[..n], &decoded[lag as usize..lag as usize + n]);
    (lag, gain, snr)
}

#[test]
fn a_mono_stream_decodes_to_the_input_after_the_known_delay() {
    let x = aperiodic(44_100, 3.0, 7);
    let bytes = encode_raw(std::slice::from_ref(&x), Mp3Config::new(44_100, 1, 128));
    let (lag, gain, snr) = raw_quality(&x, &decode(&bytes)[0]);
    eprintln!("mono 128k: lag {lag}, gain {gain:.4}, SNR {snr:.1} dB");
    assert_eq!(lag as usize, DELAY);
    assert!((gain - 1.0).abs() < 0.02 && snr > 20.0, "gain {gain}, SNR {snr}");
}

#[test]
fn stereo_with_mid_side_round_trips() {
    let pcm = stereo(44_100, 3.0);
    let bytes = encode_raw(&pcm, Mp3Config::new(44_100, 2, 128));
    let decoded = decode(&bytes);
    for ch in 0..2 {
        let (lag, gain, snr) = raw_quality(&pcm[ch], &decoded[ch]);
        eprintln!("stereo ch{ch}: lag {lag}, gain {gain:.4}, SNR {snr:.1} dB");
        assert_eq!(lag as usize, DELAY);
        assert!((gain - 1.0).abs() < 0.02 && snr > 15.0, "ch{ch}: gain {gain}, SNR {snr}");
    }
}

#[test]
fn every_rate_and_bitrate_works_and_the_size_is_constant() {
    for rate in [32_000u32, 44_100, 48_000] {
        let x = aperiodic(rate, 2.0, 11);
        for kbps in [32u32, 64, 128, 320] {
            let bytes = encode_raw(std::slice::from_ref(&x), Mp3Config::new(rate, 1, kbps));
            let frames = (x.len() + DELAY).div_ceil(FRAME);
            let expected = frames as f64 * 144_000.0 * f64::from(kbps) / f64::from(rate);
            assert!((bytes.len() as f64 - expected).abs() <= frames as f64, "{rate}/{kbps}: {} bytes, expected {expected}", bytes.len());
            let (lag, gain, snr) = raw_quality(&x, &decode(&bytes)[0]);
            assert_eq!(lag as usize, DELAY, "{rate}/{kbps}");
            // Low rates drop the top of the spectrum, which costs a little level.
            assert!((gain - 1.0).abs() < 0.06, "{rate}/{kbps}: gain {gain}");
            assert!(snr > if kbps >= 64 { 14.0 } else { 6.0 }, "{rate}/{kbps}: SNR {snr}");
        }
    }
}

#[test]
fn more_bits_sound_closer() {
    let x = aperiodic(44_100, 2.0, 5);
    let mut last = f64::MIN;
    for kbps in [32u32, 64, 128, 256] {
        let (_, _, snr) = raw_quality(&x, &decode(&encode_raw(std::slice::from_ref(&x), Mp3Config::new(44_100, 1, kbps)))[0]);
        eprintln!("{kbps}k: SNR {snr:.1} dB");
        assert!(snr > last, "{kbps}k should beat the one below ({snr} vs {last})");
        last = snr;
    }
}

#[test]
fn loud_low_tones_use_the_linbits_tables() {
    // A full-scale low tone makes large coefficients: the escape tables.
    let x: Vec<f32> = (0..44_100 * 2).map(|i| (0.95 * (2.0 * std::f64::consts::PI * 80.0 * i as f64 / 44_100.0).sin()) as f32).collect();
    let bytes = encode_raw(std::slice::from_ref(&x), Mp3Config::new(44_100, 1, 256));
    let decoded = &decode(&bytes)[0];
    let n = x.len() - 4 * FRAME;
    let (gain, snr) = compare(&x[..n], &decoded[DELAY..DELAY + n]);
    eprintln!("full-scale 80 Hz: gain {gain:.4}, SNR {snr:.1} dB");
    assert!((gain - 1.0).abs() < 0.02 && snr > 30.0, "gain {gain}, SNR {snr}");
}

#[test]
fn silence_decodes_to_silence_and_costs_nothing_extra() {
    let bytes = encode_raw(&[vec![0.0; 44_100], vec![0.0; 44_100]], Mp3Config::new(44_100, 2, 128));
    assert!(decode(&bytes).iter().all(|c| c.iter().all(|&s| s.abs() < 1e-6)));
}

#[test]
fn input_can_arrive_in_any_size_pieces() {
    let x = aperiodic(44_100, 2.0, 9);
    let cfg = Mp3Config::new(44_100, 1, 96);
    let whole = encode_raw(std::slice::from_ref(&x), cfg);
    let mut enc = Mp3Encoder::new(cfg).unwrap();
    let mut out = Vec::new();
    for piece in x.chunks(777) {
        out.extend(enc.push(&[piece]).unwrap());
    }
    out.extend(enc.finish());
    assert_eq!(out, whole, "the same stream however the input is cut");
}

#[test]
fn short_and_empty_inputs_still_make_valid_streams() {
    for n in [0usize, 1, 500, 1152, 1153, 3000] {
        let x = vec![0.2f32; n];
        let bytes = encode_raw(&[x], Mp3Config::new(44_100, 1, 64));
        if bytes.is_empty() {
            continue;
        }
        let d = decode(&bytes);
        assert!(d[0].len() >= n, "{n}: {} samples out", d[0].len());
    }
}

/// Walks the frame headers: every frame starts where the last ended, and the
/// file ends on a frame boundary.
fn frame_sizes(bytes: &[u8]) -> Vec<usize> {
    const KBPS: [u32; 15] = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
    const RATE: [u32; 3] = [44_100, 48_000, 32_000];
    let (mut at, mut sizes) = (0, Vec::new());
    while at < bytes.len() {
        assert_eq!((bytes[at], bytes[at + 1] & 0xFE), (0xFF, 0xFA), "frame {} has no sync", sizes.len());
        let (br, sr, pad) = (KBPS[(bytes[at + 2] >> 4) as usize], RATE[((bytes[at + 2] >> 2) & 3) as usize], (bytes[at + 2] >> 1) & 1);
        let size = (144_000 * br / sr + u32::from(pad)) as usize;
        sizes.push(size);
        at += size;
    }
    assert_eq!(at, bytes.len(), "the file ends on a frame boundary");
    sizes
}

#[test]
fn the_file_is_a_clean_run_of_frames_with_an_info_frame_first() {
    let x = aperiodic(44_100, 2.0, 2);
    let file = encode_mp3(std::slice::from_ref(&x), &Mp3Config::new(44_100, 1, 128)).unwrap();
    let sizes = frame_sizes(&file);
    assert_eq!(&file[21..25], b"Info");
    assert_eq!(sizes.len(), (x.len() + DELAY).div_ceil(FRAME) + 1);
}

#[test]
fn the_info_frame_makes_decoders_trim_to_the_input() {
    let x = aperiodic(44_100, 3.0, 6);
    let file = encode_mp3(std::slice::from_ref(&x), &Mp3Config::new(44_100, 1, 128)).unwrap();
    let decoded = &decode(&file)[0];
    eprintln!("with Info frame: {} samples out for {} in", decoded.len(), x.len());
    assert_eq!(decoded.len(), x.len(), "gapless: exactly the input's length");
    let n = x.len() - 4 * FRAME;
    let (gain, snr) = compare(&x[..n], &decoded[..n]);
    eprintln!("gapless alignment: gain {gain:.4}, SNR {snr:.1} dB");
    assert!((gain - 1.0).abs() < 0.02 && snr > 20.0, "gain {gain}, SNR {snr}");
}

fn have(tool: &str, flag: &str) -> bool {
    Command::new(tool).arg(flag).stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok()
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("threers-mp3-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn ffmpeg_reads_the_file_with_the_right_length_and_alignment() {
    if !have("ffmpeg", "-version") {
        eprintln!("skipped: no ffmpeg");
        return;
    }
    let pcm = stereo(44_100, 3.0);
    let file = encode_mp3(&pcm, &Mp3Config::new(44_100, 2, 128)).unwrap();
    let dir = scratch("ffmpeg");
    let path = dir.join("t.mp3");
    std::fs::write(&path, &file).unwrap();
    let out = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&path).args(["-f", "f32le", "-ac", "2", "pipe:1"]).output().unwrap();
    assert!(out.status.success() && out.stderr.is_empty(), "ffmpeg: {}", String::from_utf8_lossy(&out.stderr));
    let inter: Vec<f32> = out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let left: Vec<f32> = inter.iter().step_by(2).copied().collect();
    eprintln!("ffmpeg: {} samples for {} in", left.len(), pcm[0].len());
    assert_eq!(left.len(), pcm[0].len(), "ffmpeg trims by the Info frame to exactly the input");
    let n = left.len() - 4 * FRAME;
    let (gain, snr) = compare(&pcm[0][..n], &left[..n]);
    eprintln!("ffmpeg: gain {gain:.4}, SNR {snr:.1} dB");
    assert!((gain - 1.0).abs() < 0.02 && snr > 15.0, "gain {gain}, SNR {snr}");
    let probe = Command::new("ffprobe").args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]).arg(&path).output().unwrap();
    let secs: f64 = String::from_utf8_lossy(&probe.stdout).trim().parse().unwrap();
    eprintln!("ffprobe duration {secs}");
    assert!((secs - 3.0).abs() < 0.03, "duration {secs}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lame_decodes_it() {
    if !have("lame", "--version") {
        eprintln!("skipped: no lame");
        return;
    }
    let pcm = stereo(44_100, 3.0);
    let file = encode_mp3(&pcm, &Mp3Config::new(44_100, 2, 128)).unwrap();
    let dir = scratch("lame");
    let (src, dst) = (dir.join("t.mp3"), dir.join("t.wav"));
    std::fs::write(&src, &file).unwrap();
    let out = Command::new("lame").args(["--quiet", "--decode"]).arg(&src).arg(&dst).output().unwrap();
    assert!(out.status.success(), "lame: {}", String::from_utf8_lossy(&out.stderr));
    let wav = threers_codec::pcm::read_wav(&std::fs::read(&dst).unwrap()).unwrap();
    let left = &wav.channels[0];
    // LAME's decoder reads the Info frame too, so the audio starts at 0.
    let lag = find_lag(&pcm[0], left);
    eprintln!("lame: {} samples out, lag {lag}", left.len());
    let n = pcm[0].len() - 4 * FRAME;
    let (gain, snr) = compare(&pcm[0][..n], &left[lag.max(0) as usize..lag.max(0) as usize + n]);
    eprintln!("lame: gain {gain:.4}, SNR {snr:.1} dB");
    assert!((gain - 1.0).abs() < 0.02 && snr > 15.0, "gain {gain}, SNR {snr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn apples_decoder_reads_it() {
    if !have("afconvert", "-h") {
        eprintln!("skipped: no afconvert");
        return;
    }
    let pcm = stereo(44_100, 3.0);
    let file = encode_mp3(&pcm, &Mp3Config::new(44_100, 2, 128)).unwrap();
    let dir = scratch("apple");
    let (src, dst) = (dir.join("t.mp3"), dir.join("t.wav"));
    std::fs::write(&src, &file).unwrap();
    let out = Command::new("afconvert").args(["-f", "WAVE", "-d", "F32@44100"]).arg(&src).arg(&dst).output().unwrap();
    assert!(out.status.success(), "afconvert: {}", String::from_utf8_lossy(&out.stderr));
    let wav = threers_codec::pcm::read_wav(&std::fs::read(&dst).unwrap()).unwrap();
    let left = &wav.channels[0];
    let lag = find_lag(&pcm[0], left);
    eprintln!("Apple: {} samples out, lag {lag}", left.len());
    let n = pcm[0].len() - 4 * FRAME;
    let start = lag.max(0) as usize;
    let (gain, snr) = compare(&pcm[0][..n], &left[start..start + n]);
    eprintln!("Apple: gain {gain:.4}, SNR {snr:.1} dB");
    assert!((gain - 1.0).abs() < 0.02 && snr > 15.0, "gain {gain}, SNR {snr}");
    let _ = std::fs::remove_dir_all(&dir);
}
