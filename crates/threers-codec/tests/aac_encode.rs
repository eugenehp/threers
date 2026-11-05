//! The AAC-LC encoder against the decoders: this crate's own (bit-exact with
//! ffmpeg's) and, where it is installed, ffmpeg itself.

use std::io::Write;
use std::process::{Command, Stdio};

use threers_codec::aac::encoder::{encode_adts, AacEncoder, EncoderConfig, DELAY, FRAME};
use threers_codec::aac::AacDecoder;

fn tone(sample_rate: u32, hz: f64, amp: f64, n: usize) -> Vec<f32> {
    (0..n).map(|i| (amp * (2.0 * std::f64::consts::PI * hz * i as f64 / f64::from(sample_rate)).sin()) as f32).collect()
}

/// A little music: a few partials with a slow envelope and a click track, so
/// there is both tone and transient.
fn music(sample_rate: u32, seconds: f64, seed: u32) -> Vec<f32> {
    let n = (seconds * f64::from(sample_rate)) as usize;
    let mut state = seed.max(1);
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 8) as f32 / 8_388_608.0 - 1.0
    };
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            let env = 0.5 + 0.5 * (2.0 * std::f64::consts::PI * 0.7 * t).sin();
            let notes: f64 = [220.0, 330.0, 440.0, 1_320.0, 3_000.0].iter().enumerate().map(|(k, f)| (2.0 * std::f64::consts::PI * f * t).sin() / (k + 1) as f64).sum();
            let click = if (i % (sample_rate as usize / 4)) < 200 { noise() * 0.5 } else { 0.0 };
            (0.25 * env * notes) as f32 + click * 0.3
        })
        .collect()
}

/// Encodes to raw access units and decodes them with this crate's decoder.
fn round_trip(planar: &[Vec<f32>], cfg: EncoderConfig) -> (Vec<Vec<f32>>, usize) {
    let mut enc = AacEncoder::new(cfg).unwrap();
    let views: Vec<&[f32]> = planar.iter().map(Vec::as_slice).collect();
    let mut units = enc.push(&views).unwrap();
    units.extend(enc.finish());
    let bytes: usize = units.iter().map(Vec::len).sum();
    let mut dec = AacDecoder::new(&enc.audio_specific_config()).unwrap();
    let mut out = vec![Vec::new(); cfg.channels];
    for u in &units {
        dec.decode_frame(u, &mut out).unwrap();
    }
    (out, bytes)
}

/// Gain and SNR (dB) of `decoded` against `original`, after the encoder delay.
fn compare(original: &[f32], decoded: &[f32]) -> (f64, f64) {
    let n = original.len().min(decoded.len().saturating_sub(DELAY)).saturating_sub(FRAME);
    let (a, b) = (&original[..n], &decoded[DELAY..DELAY + n]);
    let (mut ab, mut aa) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        ab += f64::from(*x) * f64::from(*y);
        aa += f64::from(*x) * f64::from(*x);
    }
    let gain = ab / aa;
    let (mut signal, mut noise) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        signal += f64::from(*x) * f64::from(*x);
        noise += (f64::from(*y) - f64::from(*x)).powi(2);
    }
    (gain, 10.0 * (signal / noise.max(1e-30)).log10())
}

#[test]
fn a_mono_tone_comes_back_at_the_same_level() {
    let x = tone(44_100, 1_000.0, 0.5, 44_100 * 2);
    let (decoded, bytes) = round_trip(&[x.clone()], EncoderConfig::new(44_100, 1, 64_000));
    let (gain, snr) = compare(&x, &decoded[0]);
    eprintln!("mono tone: gain {gain:.4}, SNR {snr:.1} dB, {bytes} bytes");
    assert!((gain - 1.0).abs() < 0.03, "gain {gain}");
    assert!(snr > 20.0, "SNR {snr}");
}

fn stereo_music(sample_rate: u32, seconds: f64) -> Vec<Vec<f32>> {
    let a = music(sample_rate, seconds, 1);
    let b = music(sample_rate, seconds, 2);
    // Mostly the same in both channels, a little different: typical of a mix.
    let left: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + 0.15 * y).collect();
    let right: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - 0.15 * y).collect();
    vec![left, right]
}

#[test]
fn stereo_music_round_trips_with_mid_side() {
    let pcm = stereo_music(44_100, 3.0);
    let (decoded, bytes) = round_trip(&pcm, EncoderConfig::new(44_100, 2, 128_000));
    for ch in 0..2 {
        let (gain, snr) = compare(&pcm[ch], &decoded[ch]);
        eprintln!("stereo ch{ch}: gain {gain:.4}, SNR {snr:.1} dB");
        assert!((gain - 1.0).abs() < 0.03, "ch{ch} gain {gain}");
        assert!(snr > 14.0, "ch{ch} SNR {snr}");
    }
    let kbps = bytes as f64 * 8.0 / 3.0 / 1000.0;
    eprintln!("stereo at 128k: {kbps:.1} kbit/s");
    assert!((kbps - 128.0).abs() < 128.0 * 0.12, "{kbps} kbit/s");
}

#[test]
fn the_bitrate_is_met_across_the_range() {
    for (rate, channels, kbps) in [(44_100u32, 2usize, 64u32), (44_100, 2, 192), (48_000, 1, 32), (32_000, 2, 96)] {
        let pcm: Vec<Vec<f32>> = (0..channels).map(|c| music(rate, 4.0, 10 + c as u32)).collect();
        let (_, bytes) = round_trip(&pcm, EncoderConfig::new(rate, channels, kbps * 1000));
        let got = bytes as f64 * 8.0 / 4.0 / 1000.0;
        eprintln!("{rate} Hz x{channels} at {kbps}k: {got:.1} kbit/s");
        assert!((got - f64::from(kbps)).abs() < f64::from(kbps) * 0.15, "{rate}/{channels}/{kbps}: {got}");
    }
}

#[test]
fn higher_bitrates_sound_closer() {
    let pcm = stereo_music(44_100, 3.0);
    let mut last = f64::MIN;
    for kbps in [48u32, 96, 192] {
        let (decoded, _) = round_trip(&pcm, EncoderConfig::new(44_100, 2, kbps * 1000));
        let (_, snr) = compare(&pcm[0], &decoded[0]);
        eprintln!("{kbps}k: SNR {snr:.1} dB");
        assert!(snr > last, "{kbps}k should beat the one below ({snr} vs {last})");
        last = snr;
    }
}

#[test]
fn every_sample_rate_works() {
    for rate in [96_000u32, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000] {
        let hz = (f64::from(rate) / 8.0).min(3_000.0);
        let x = tone(rate, hz, 0.4, rate as usize);
        let kbps = (rate / 1000 * 2).clamp(16, 128);
        let (decoded, _) = round_trip(&[x.clone()], EncoderConfig::new(rate, 1, kbps * 1000));
        let (gain, snr) = compare(&x, &decoded[0]);
        assert!((gain - 1.0).abs() < 0.05 && snr > 15.0, "{rate} Hz: gain {gain}, SNR {snr}");
    }
}

#[test]
fn loud_and_full_scale_audio_survives_escape_codes() {
    // Full scale low tones make large coefficients: codebook 11 with escapes.
    let x: Vec<f32> = (0..44_100 * 2).map(|i| (0.99 * (2.0 * std::f64::consts::PI * 60.0 * i as f64 / 44_100.0).sin()) as f32).collect();
    let (decoded, _) = round_trip(&[x.clone()], EncoderConfig::new(44_100, 1, 192_000));
    let (gain, snr) = compare(&x, &decoded[0]);
    eprintln!("full-scale 60 Hz: gain {gain:.4}, SNR {snr:.1} dB");
    assert!((gain - 1.0).abs() < 0.02 && snr > 30.0, "gain {gain}, SNR {snr}");
}

#[test]
fn silence_costs_almost_nothing_and_decodes_to_silence() {
    let (decoded, bytes) = round_trip(&[vec![0.0; 44_100], vec![0.0; 44_100]], EncoderConfig::new(44_100, 2, 128_000));
    assert!(bytes < 44_100 / 1024 * 40 + 200, "{bytes} bytes for a second of silence");
    assert!(decoded.iter().all(|c| c.iter().all(|&s| s.abs() < 1e-6)));
}

#[test]
fn input_can_arrive_in_any_size_pieces() {
    let x = music(44_100, 2.0, 5);
    let cfg = EncoderConfig::new(44_100, 1, 96_000);
    let (whole, _) = round_trip(&[x.clone()], cfg);
    let mut enc = AacEncoder::new(cfg).unwrap();
    let mut units = Vec::new();
    for piece in x.chunks(777) {
        units.extend(enc.push(&[piece]).unwrap());
    }
    units.extend(enc.finish());
    let mut dec = AacDecoder::new(&enc.audio_specific_config()).unwrap();
    let mut out = vec![Vec::new()];
    for u in &units {
        dec.decode_frame(u, &mut out).unwrap();
    }
    assert_eq!(out[0], whole[0], "the same stream however the input is cut");
}

#[test]
fn a_short_input_still_makes_a_valid_stream() {
    for n in [0usize, 1, 500, 1024, 1025] {
        let x = tone(44_100, 440.0, 0.3, n);
        let (decoded, _) = round_trip(&[x], EncoderConfig::new(44_100, 1, 64_000));
        assert!(decoded[0].len() >= n + DELAY - FRAME.min(n + DELAY), "{n}: {} samples out", decoded[0].len());
    }
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

/// Decodes ADTS with ffmpeg to f32 PCM, interleaved.
fn ffmpeg_decode(adts: &[u8], channels: usize) -> Vec<f32> {
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-f", "aac", "-i", "pipe:0", "-f", "f32le", "-ac", &channels.to_string(), "pipe:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let data = adts.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&data);
    });
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(out.status.success(), "ffmpeg: {}", String::from_utf8_lossy(&out.stderr));
    assert!(out.stderr.is_empty(), "ffmpeg complained: {}", String::from_utf8_lossy(&out.stderr));
    out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

#[test]
fn ffmpeg_decodes_the_adts_stream() {
    if !have_ffmpeg() {
        eprintln!("skipped: no ffmpeg");
        return;
    }
    let pcm = stereo_music(44_100, 3.0);
    let adts = encode_adts(&pcm, &EncoderConfig::new(44_100, 2, 128_000)).unwrap();
    let inter = ffmpeg_decode(&adts, 2);
    let left: Vec<f32> = inter.iter().step_by(2).copied().collect();
    let right: Vec<f32> = inter.iter().skip(1).step_by(2).copied().collect();
    // ffmpeg skips the encoder delay for ADTS? Find the alignment instead.
    let best = (0..2_200).max_by_key(|&d| {
        let n = 20_000.min(left.len().saturating_sub(d));
        (pcm[0][..n].iter().zip(&left[d..d + n]).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() * 1_000.0) as i64
    }).unwrap();
    let n = pcm[0].len().min(left.len() - best) - FRAME;
    let (gain, snr) = compare_aligned(&pcm[0][..n], &left[best..best + n]);
    let (_, snr_r) = compare_aligned(&pcm[1][..n], &right[best..best + n]);
    eprintln!("ffmpeg: delay {best}, gain {gain:.4}, SNR {snr:.1}/{snr_r:.1} dB");
    assert!((gain - 1.0).abs() < 0.03 && snr > 14.0 && snr_r > 14.0, "gain {gain}, SNR {snr}/{snr_r}");
    assert!(left.len() >= pcm[0].len(), "all the audio comes out");
}

fn compare_aligned(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut ab, mut aa, mut noise) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        ab += f64::from(*x) * f64::from(*y);
        aa += f64::from(*x) * f64::from(*x);
        noise += (f64::from(*y) - f64::from(*x)).powi(2);
    }
    (ab / aa, 10.0 * (aa / noise.max(1e-30)).log10())
}

fn m4a_file(pcm: &[Vec<f32>], cfg: EncoderConfig) -> Vec<u8> {
    threers_codec::aac::m4a::encode_m4a(pcm, &cfg).unwrap()
}

#[test]
fn the_m4a_decodes_aligned_with_the_delay_skipped() {
    let pcm = stereo_music(44_100, 3.0);
    let file = m4a_file(&pcm, EncoderConfig::new(44_100, 2, 128_000));
    assert_eq!(&file[4..8], b"ftyp");
    let decoded = threers_codec::aac::decode_mp4(&file).unwrap();
    assert_eq!(decoded.sample_rate, 44_100);
    assert_eq!(decoded.planes.len(), 2);
    // The edit list took out the priming: no lag now.
    let n = pcm[0].len() - FRAME;
    let (gain, snr) = compare_aligned(&pcm[0][..n], &decoded.planes[0][..n]);
    eprintln!("m4a via threers: {} samples out for {} in, gain {gain:.4}, SNR {snr:.1} dB", decoded.planes[0].len(), pcm[0].len());
    assert!((gain - 1.0).abs() < 0.03 && snr > 14.0, "gain {gain}, SNR {snr}");
    assert!(decoded.planes[0].len() >= pcm[0].len());
}

#[test]
fn ffmpeg_reads_the_m4a_with_the_right_length_and_alignment() {
    if !have_ffmpeg() {
        eprintln!("skipped: no ffmpeg");
        return;
    }
    let pcm = stereo_music(44_100, 3.0);
    let file = m4a_file(&pcm, EncoderConfig::new(44_100, 2, 128_000));
    let dir = std::env::temp_dir().join(format!("threers-m4a-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.m4a");
    std::fs::write(&path, &file).unwrap();
    let out = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&path).args(["-f", "f32le", "-ac", "2", "pipe:1"]).output().unwrap();
    assert!(out.status.success() && out.stderr.is_empty(), "ffmpeg: {}", String::from_utf8_lossy(&out.stderr));
    let inter: Vec<f32> = out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let left: Vec<f32> = inter.iter().step_by(2).copied().collect();
    eprintln!("ffmpeg m4a: {} samples for {} in", left.len(), pcm[0].len());
    assert!((left.len() as i64 - pcm[0].len() as i64).abs() <= 2_048, "length {} vs {}", left.len(), pcm[0].len());
    let n = pcm[0].len() - FRAME;
    let (gain, snr) = compare_aligned(&pcm[0][..n], &left[..n]);
    assert!((gain - 1.0).abs() < 0.03 && snr > 14.0, "gain {gain}, SNR {snr}");
    // And the container itself passes ffprobe's eye: the duration is the input's.
    let probe = Command::new("ffprobe").args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]).arg(&path).output();
    if let Ok(p) = probe {
        let secs: f64 = String::from_utf8_lossy(&p.stdout).trim().parse().unwrap_or(0.0);
        eprintln!("ffprobe duration {secs}");
        assert!((secs - 3.0).abs() < 0.05, "duration {secs}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A signal with no repeats, so the lag between two copies is unambiguous.
fn aperiodic(sample_rate: u32, seconds: f64) -> Vec<Vec<f32>> {
    let mut state = 12_345u32;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 8) as f32 / 8_388_608.0 - 1.0
    };
    let n = (seconds * f64::from(sample_rate)) as usize;
    let mut low = 0.0f32;
    let ch = |noise: &mut dyn FnMut() -> f32, low: &mut f32| -> Vec<f32> {
        // Low-passed noise (so it codes well) under a sweeping tone.
        (0..n)
            .map(|i| {
                *low += 0.15 * (noise() - *low);
                let t = i as f64 / f64::from(sample_rate);
                0.5 * *low + 0.2 * (2.0 * std::f64::consts::PI * (200.0 * t + 300.0 * t * t)).sin() as f32
            })
            .collect()
    };
    let left = ch(&mut noise, &mut low);
    let right: Vec<f32> = left.iter().map(|x| x * 0.8).collect();
    vec![left, right]
}

/// The shift of `decoded` against `original` (positive: decoded starts late).
fn find_lag(original: &[f32], decoded: &[f32]) -> i64 {
    let (a, len) = (6_000usize, 8_000usize);
    (-3_000i64..4_400)
        .max_by_key(|&d| {
            let start = (a as i64 + d) as usize;
            let dot: f64 = original[a..a + len].iter().zip(&decoded[start..start + len]).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
            (dot * 1e6) as i64
        })
        .unwrap()
}

#[test]
fn apples_decoder_reads_both_the_m4a_and_the_adts() {
    let have = Command::new("afconvert").arg("-h").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok();
    if !have {
        eprintln!("skipped: no afconvert");
        return;
    }
    let pcm = aperiodic(44_100, 3.0);
    let cfg = EncoderConfig::new(44_100, 2, 128_000);
    let dir = std::env::temp_dir().join(format!("threers-af-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, data) in [("t.m4a", m4a_file(&pcm, cfg)), ("t.aac", encode_adts(&pcm, &cfg).unwrap())] {
        let (src, dst) = (dir.join(name), dir.join(format!("{name}.wav")));
        std::fs::write(&src, data).unwrap();
        let out = Command::new("afconvert").args(["-f", "WAVE", "-d", "F32@44100"]).arg(&src).arg(&dst).output().unwrap();
        assert!(out.status.success(), "afconvert {name}: {}", String::from_utf8_lossy(&out.stderr));
        let decoded = threers_codec::pcm::read_wav(&std::fs::read(&dst).unwrap()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let left = &decoded.channels[0];
        let lag = find_lag(&pcm[0], left);
        let n = (pcm[0].len() - 3 * FRAME).min(left.len() - lag.unsigned_abs() as usize - 1);
        let (from_orig, from_dec) = if lag >= 0 { (0, lag as usize) } else { (lag.unsigned_abs() as usize, 0) };
        let (gain, snr) = compare_aligned(&pcm[0][from_orig..from_orig + n], &left[from_dec..from_dec + n]);
        eprintln!("Apple on {name}: lag {lag}, {} samples out, gain {gain:.4}, SNR {snr:.1} dB", left.len());
        assert!((gain - 1.0).abs() < 0.03 && snr > 14.0, "{name}: lag {lag}, gain {gain}, SNR {snr}");
        // The m4a starts where the audio does; raw ADTS leaves the delay in.
        assert_eq!(lag, if name.ends_with("m4a") { 0 } else { DELAY as i64 }, "{name}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_bitrate_beyond_what_the_rate_can_carry_is_capped_not_refused() {
    // 320 kbit/s at 22.05 kHz is 14,860 bits a frame; a frame holds 6,144 a channel.
    let x = aperiodic(22_050, 2.0);
    let (decoded, bytes) = round_trip(&[x[0].clone()], EncoderConfig::new(22_050, 1, 320_000));
    let kbps = bytes as f64 * 8.0 / 2.0 / 1000.0;
    eprintln!("320k at 22.05 kHz mono: {kbps:.1} kbit/s");
    assert!(kbps < 140.0, "capped near the frame limit, got {kbps}");
    let (gain, snr) = compare(&x[0], &decoded[0]);
    assert!((gain - 1.0).abs() < 0.02 && snr > 25.0, "gain {gain}, SNR {snr}");
}
