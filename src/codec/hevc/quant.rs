//! HEVC uniform-reconstruction quantization (H.265 §8.6.3), flat scaling list.
//!
//! [`dequant`] is bit-exact to the spec (the decoder runs it; the encoder mirrors
//! it to reconstruct). [`quant`] is the encoder's forward step. Higher QP → more
//! coefficients driven to zero → smaller bitstream, more loss.
//!
//! 8-bit samples, no scaling list (`m = 16`). Pure integer math (wasm-safe).

/// `levelScale[qP % 6]` — the dequant multiplier (§8.6.3).
const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];
/// Forward-quant multiplier (reciprocal-ish of [`LEVEL_SCALE`]).
const QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];

#[inline]
fn log2(n: usize) -> i32 {
    n.trailing_zeros() as i32
}

/// Dequantize coefficient levels for an `n×n` block at `qp` (H.265 §8.6.3),
/// producing transform coefficients ready for the inverse transform.
pub fn dequant(levels: &[i32], n: usize, qp: i32) -> Vec<i32> {
    let bd_shift = 8 + log2(n) - 5; // BitDepth + Log2(nTbS) - 5, 8-bit
    let m = 16i64; // flat scaling list
    let per = (qp / 6) as i64;
    let rem = (qp % 6) as usize;
    let add = if bd_shift > 0 {
        1i64 << (bd_shift - 1)
    } else {
        0
    };
    levels
        .iter()
        .map(|&l| {
            let d = (((l as i64 * m * LEVEL_SCALE[rem]) << per) + add) >> bd_shift;
            d.clamp(-32768, 32767) as i32
        })
        .collect()
}

/// Forward-quantize transform coefficients for an `n×n` block at `qp`. `intra`
/// selects the rounding offset (deadzone). Returns signed levels.
pub fn quant(coeff: &[i32], n: usize, qp: i32, intra: bool) -> Vec<i32> {
    let transform_shift = 15 - 8 - log2(n); // MAX_TR_DYNAMIC_RANGE - BitDepth - Log2(nTbS)
    let q_bits = 14 + qp / 6 + transform_shift;
    let rem = (qp % 6) as usize;
    let offset = (1i64 << q_bits) / if intra { 3 } else { 6 };
    coeff
        .iter()
        .map(|&c| {
            let sign = if c < 0 { -1 } else { 1 };
            let level = (c.unsigned_abs() as i64 * QUANT_SCALE[rem] + offset) >> q_bits;
            sign * level as i32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_dequant_preserves_dc_lowqp() {
        // At low QP a DC coefficient survives near-losslessly.
        let coeff = vec![512i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let levels = quant(&coeff, 4, 4, true);
        let back = dequant(&levels, 4, 4);
        assert!(
            (back[0] - coeff[0]).abs() < 40,
            "DC ~preserved, got {}",
            back[0]
        );
    }

    #[test]
    fn higher_qp_zeros_more() {
        let coeff: Vec<i32> = (0..16).map(|i| (i - 8) * 20).collect();
        let low = quant(&coeff, 4, 8, true);
        let high = quant(&coeff, 4, 40, true);
        let nz_low = low.iter().filter(|&&x| x != 0).count();
        let nz_high = high.iter().filter(|&&x| x != 0).count();
        assert!(
            nz_high <= nz_low,
            "higher QP should zero more: {nz_high} vs {nz_low}"
        );
    }

    #[test]
    fn zero_stays_zero() {
        let z = vec![0i32; 64];
        assert!(quant(&z, 8, 26, true).iter().all(|&x| x == 0));
        assert!(dequant(&z, 8, 26,).iter().all(|&x| x == 0));
    }

    #[test]
    fn sign_preserved() {
        let coeff = vec![-300i32, 300, -50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let levels = quant(&coeff, 4, 20, true);
        assert!(levels[0] < 0 && levels[1] > 0, "signs kept: {levels:?}");
    }
}
