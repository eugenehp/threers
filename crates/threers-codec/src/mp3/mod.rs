//! MPEG-1/2/2.5 Layer III decoding, ported from mpg123 1.32.3 as libsndfile
//! 1.2.2 (`soundfile`) runs it on arm64: float output, gapless, the NEON64
//! decoder. mpg123 is built with `-ffast-math`, so the float kernels are
//! generated from the reference binary's machine code ([`built`], run on
//! [`cpu::Cpu`]).

pub mod cpu;
/// The reference build's float kernels, generated from its arm64 machine code
/// and run on [`cpu::Cpu`] so their rounding matches it bit for bit.
#[allow(clippy::all)]
pub mod built;
pub mod decoder;
/// Dequantisation and synthesis tables, generated from mpg123 1.32.3.
#[allow(clippy::all)]
pub mod tables;

pub use decoder::{decode, probe, Mp3};

#[cfg(test)]
mod leaf_parity {
    use super::cpu::Cpu;
    const Z: usize = 4096;

    fn rd(p: &str) -> Option<Vec<f32>> {
        Some(std::fs::read(p).ok()?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }

    /// The generated kernels against the reference's own functions
    /// (inputs/outputs from `/tmp/leaf_ref.py`); skipped without the dumps.
    #[test]
    fn generated_kernels_match_reference() {
        let cases: [(&str, usize, Option<usize>, fn(&mut Cpu)); 5] = [
            ("dct12", 5, None, super::built::dct12),
            ("dct36", 5, None, super::built::INT123_dct36_neon64),
            ("dct64", 3, None, super::built::INT123_dct64_real_neon64),
            ("synth", 4, Some(3), super::built::INT123_synth_1to1_real_neon64_asm),
            ("synth_s", 5, Some(4), super::built::INT123_synth_1to1_real_s_neon64_asm),
        ];
        for (name, nargs, intarg, f) in cases {
            for trial in 0..3 {
                let (Some(inp), Some(want)) = (rd(&format!("/tmp/leaf_{name}_{trial}_in.f32")), rd(&format!("/tmp/leaf_{name}_{trial}_out.f32"))) else {
                    eprintln!("SKIP {name}");
                    continue;
                };
                let mut c = Cpu::default();
                for i in 0..nargs {
                    let base = (i * Z * 4 + 0x1000) as u64;
                    c.put(base, &inp[i * Z..(i + 1) * Z]);
                    c.x[i] = if Some(i) == intarg { (trial * 5 + 1) as u64 } else { base + 4 * (Z / 2) as u64 };
                }
                f(&mut c);
                let ret: u64 = std::fs::read_to_string(format!("/tmp/leaf_{name}_{trial}_ret.txt")).unwrap().trim().parse::<i64>().map(|v| v as u64).unwrap_or(0);
                let mut bad = 0;
                for i in 0..nargs {
                    if Some(i) == intarg {
                        continue;
                    }
                    let got = c.get((i * Z * 4 + 0x1000) as u64, Z);
                    let d: Vec<(usize, f32, f32)> = (0..Z).filter(|&k| got[k].to_bits() != want[i * Z + k].to_bits()).map(|k| (k as isize - (Z / 2) as isize, got[k], want[i * Z + k])).map(|(k, a, b)| (k as usize, a, b)).take(6).collect();
                    if !d.is_empty() { eprintln!("  zone {i}: {:?}", d); }
                    bad += got.iter().zip(&want[i * Z..(i + 1) * Z]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                }
                eprintln!("{name} trial {trial}: {bad} floats differ; ret ours {} ref {ret}", c.x[0] as u32);
                assert_eq!(bad, 0, "{name}");
            }
        }
    }
}
