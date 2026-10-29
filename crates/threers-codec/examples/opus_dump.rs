//! Decode a WebM/Matroska Opus track to interleaved f32le, as
//! `ffmpeg -i IN -f f32le OUT` writes it. usage: opus_dump IN OUT
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("read");
    let t = std::time::Instant::now();
    let d = threers_codec::opus::decode_webm(&data).expect("decode");
    let el = t.elapsed();
    let n = d.planes[0].len();
    let mut out = Vec::with_capacity(n * d.planes.len() * 4);
    for i in 0..n {
        for p in &d.planes {
            out.extend_from_slice(&p[i].to_le_bytes());
        }
    }
    std::fs::write(&a[2], out).expect("write");
    if std::env::var_os("OPUS_STATS").is_some() {
        use threers_codec::opus::celt::{STATS, STAT_NAMES};
        let mkv = threers_codec::mkv_demux::Mkv::parse(&data).unwrap();
        let mut cfg = std::collections::BTreeMap::new();
        for f in &mkv.frames {
            if let Ok(p) = threers_codec::opus::parse_packet(&f.data, false) {
                *cfg.entry((p.config, p.stereo, p.code)).or_insert(0) += 1;
            }
        }
        let s: Vec<String> = STAT_NAMES.iter().zip(&STATS).map(|(n, v)| format!("{n}={}", v.load(std::sync::atomic::Ordering::Relaxed))).collect();
        println!("cfg(config,stereo,code)={cfg:?}\n  {}", s.join(" "));
    }
    eprintln!("{} ch, {} samples, {} frames, {:.1} ms", d.planes.len(), n, d.frame_lens.len(), el.as_secs_f64() * 1e3);
}
