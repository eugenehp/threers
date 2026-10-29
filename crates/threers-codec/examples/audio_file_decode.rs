//! Decode any supported audio file (threers_codec::audio_file) and write
//! interleaved f32, or compare with a reference dump (e.g. soundfile's).
//!   audio_file_decode IN OUT.f32
//!   audio_file_decode IN --compare REF.f32
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("read");
    let t = std::time::Instant::now();
    let p = threers_codec::audio_file::read(&data).unwrap_or_else(|e| {
        println!("ERROR {e}");
        std::process::exit(2)
    });
    let (ch, n) = (p.channels.len(), p.frames());
    let mut inter = Vec::with_capacity(n * ch);
    for i in 0..n {
        for c in &p.channels {
            inter.push(c[i]);
        }
    }
    eprintln!("{} Hz, {ch} ch, {n} frames in {:.3}s", p.sample_rate, t.elapsed().as_secs_f64());
    if a.get(2).map(String::as_str) == Some("--compare") {
        let r = std::fs::read(&a[3]).expect("reference");
        let r: Vec<f32> = r.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let m = inter.len().min(r.len());
        let diff = (0..m).filter(|&i| inter[i].to_bits() != r[i].to_bits()).count();
        let maxd = (0..m).map(|i| (inter[i] as f64 - r[i] as f64).abs()).fold(0.0, f64::max);
        let first = (0..m).find(|&i| inter[i].to_bits() != r[i].to_bits());
        println!("values ours {} ref {}: {diff} differ, max |d| {maxd:.3e}, first {first:?}", inter.len(), r.len());
    } else {
        std::fs::write(&a[2], inter.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
}
