//! Decode a WebM Opus file N times (profiling aid). usage: opus_bench IN N
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("read");
    let n: usize = a.get(2).map_or(10, |s| s.parse().unwrap());
    let t = std::time::Instant::now();
    let mut tot = 0;
    for _ in 0..n {
        tot += threers_codec::opus::decode_webm(&data).expect("decode").planes[0].len();
    }
    eprintln!("{:.1} ms per decode ({tot})", t.elapsed().as_secs_f64() * 1e3 / n as f64);
}
