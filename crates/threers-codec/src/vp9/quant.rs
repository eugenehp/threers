//! VP9 8-bit quantizer tables and 4×4 coefficient quantization.
//!
//! `dc_qlookup` / `ac_qlookup` are the libvpx `vp9_quant_common.c` tables
//! (QINDEX_RANGE = 256). Quantization uses a round-half-up deadzone that the
//! decoder inverts with `dqcoeff = qcoeff * quant` (`dq_shift = 0` for 4×4).

/// libvpx `dc_qlookup` (8-bit), indexed by `qindex` in `0..256`.
const DC_QLOOKUP: [i16; 256] = [
    4, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26, 26, 27,
    28, 29, 30, 31, 32, 32, 33, 34, 35, 36, 37, 38, 38, 39, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48,
    48, 49, 50, 51, 52, 53, 53, 54, 55, 56, 57, 57, 58, 59, 60, 61, 62, 62, 63, 64, 65, 66, 66, 67,
    68, 69, 70, 70, 71, 72, 73, 74, 74, 75, 76, 77, 78, 78, 79, 80, 81, 81, 82, 83, 84, 85, 85, 87,
    88, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114, 116, 117,
    118, 120, 121, 123, 125, 127, 129, 131, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154,
    156, 158, 161, 164, 166, 169, 172, 174, 177, 180, 182, 185, 187, 190, 192, 195, 199, 202, 205,
    208, 211, 214, 217, 220, 223, 226, 230, 233, 237, 240, 243, 247, 250, 253, 257, 261, 265, 269,
    272, 276, 280, 284, 288, 292, 296, 300, 304, 309, 313, 317, 322, 326, 330, 335, 340, 344, 349,
    354, 359, 364, 369, 374, 379, 384, 389, 395, 400, 406, 411, 417, 423, 429, 435, 441, 447, 454,
    461, 467, 475, 482, 489, 497, 505, 513, 522, 530, 539, 549, 559, 569, 579, 590, 602, 614, 626,
    640, 654, 668, 684, 700, 717, 736, 755, 775, 796, 819, 843, 869, 896, 925, 955, 988, 1022,
    1058, 1098, 1139, 1184, 1232, 1282, 1336,
];

/// libvpx `ac_qlookup` (8-bit), indexed by `qindex` in `0..256`.
const AC_QLOOKUP: [i16; 256] = [
    4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138,
    140, 142, 144, 146, 148, 150, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188,
    191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 260,
    265, 270, 275, 280, 285, 290, 295, 300, 305, 311, 317, 323, 329, 335, 341, 347, 353, 359, 366,
    373, 380, 387, 394, 401, 408, 416, 424, 432, 440, 448, 456, 465, 474, 483, 492, 501, 510, 520,
    530, 540, 550, 560, 571, 582, 593, 604, 615, 627, 639, 651, 663, 676, 689, 702, 715, 729, 743,
    757, 771, 786, 801, 816, 832, 848, 864, 881, 898, 915, 933, 951, 969, 988, 1007, 1026, 1046,
    1066, 1087, 1108, 1129, 1151, 1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343, 1369, 1396, 1423,
    1451, 1479, 1508, 1537, 1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,
];

/// DC quantizer for 8-bit (`vp9_dc_quant` with delta 0).
pub fn dc_quant(qindex: usize) -> i32 {
    DC_QLOOKUP[qindex.min(255)] as i32
}

/// AC quantizer for 8-bit (`vp9_ac_quant` with delta 0).
pub fn ac_quant(qindex: usize) -> i32 {
    AC_QLOOKUP[qindex.min(255)] as i32
}

/// Quantize one coefficient: `q = round(|c| / quant)` with sign, `dq = q * quant`.
#[inline]
fn quantize_coeff(coeff: i32, quant: i32) -> (i32, i32) {
    debug_assert!(quant > 0);
    let abs_c = coeff.unsigned_abs() as i32;
    let abs_q = (abs_c + (quant >> 1)) / quant;
    let qcoeff = if coeff < 0 { -abs_q } else { abs_q };
    let dqcoeff = qcoeff * quant;
    (qcoeff, dqcoeff)
}

/// Quantize a 4×4 coefficient block in raster order.
///
/// Position 0 (DC) uses `dc_quant(qindex)`; the rest use `ac_quant(qindex)`.
/// Returns `(qcoeff, dqcoeff)` where `dqcoeff[i] = qcoeff[i] * quant_value` —
/// the form the IDCT consumes after token decode.
pub fn quantize(coeff: &[i32; 16], qindex: usize) -> ([i32; 16], [i32; 16]) {
    let dc = dc_quant(qindex);
    let ac = ac_quant(qindex);
    let mut qcoeff = [0i32; 16];
    let mut dqcoeff = [0i32; 16];
    for i in 0..16 {
        let q = if i == 0 { dc } else { ac };
        let (qc, dqc) = quantize_coeff(coeff[i], q);
        qcoeff[i] = qc;
        dqcoeff[i] = dqc;
    }
    (qcoeff, dqcoeff)
}

/// Quantize an 8×8 coefficient block (same deadzone as 4×4; `dq_shift = 0`).
pub fn quantize_8x8(coeff: &[i32; 64], qindex: usize) -> ([i32; 64], [i32; 64]) {
    let dc = dc_quant(qindex);
    let ac = ac_quant(qindex);
    let mut qcoeff = [0i32; 64];
    let mut dqcoeff = [0i32; 64];
    for i in 0..64 {
        let q = if i == 0 { dc } else { ac };
        let (qc, dqc) = quantize_coeff(coeff[i], q);
        qcoeff[i] = qc;
        dqcoeff[i] = dqc;
    }
    (qcoeff, dqcoeff)
}

/// Quantize a 16×16 coefficient block (same deadzone; `dq_shift = 0`).
pub fn quantize_16x16(coeff: &[i32; 256], qindex: usize) -> ([i32; 256], [i32; 256]) {
    let dc = dc_quant(qindex);
    let ac = ac_quant(qindex);
    let mut qcoeff = [0i32; 256];
    let mut dqcoeff = [0i32; 256];
    for i in 0..256 {
        let q = if i == 0 { dc } else { ac };
        let (qc, dqc) = quantize_coeff(coeff[i], q);
        qcoeff[i] = qc;
        dqcoeff[i] = dqc;
    }
    (qcoeff, dqcoeff)
}

/// Quantize a 32×32 coefficient block.
///
/// libvpx uses `dq_shift = 1` for TX_32X32 (`dqcoeff = (qcoeff * quant) >> 1`).
pub fn quantize_32x32(coeff: &[i32; 1024], qindex: usize) -> ([i32; 1024], [i32; 1024]) {
    let dc = dc_quant(qindex);
    let ac = ac_quant(qindex);
    let mut qcoeff = [0i32; 1024];
    let mut dqcoeff = [0i32; 1024];
    for i in 0..1024 {
        let q = if i == 0 { dc } else { ac };
        let (qc, _) = quantize_coeff(coeff[i], q);
        qcoeff[i] = qc;
        dqcoeff[i] = (qc * q) >> 1;
    }
    (qcoeff, dqcoeff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qlookup_endpoints() {
        assert_eq!(dc_quant(0), 4);
        assert_eq!(ac_quant(0), 4);
        assert_eq!(dc_quant(255), 1336);
        assert_eq!(ac_quant(255), 1828);
        assert_eq!(dc_quant(999), 1336);
    }

    #[test]
    fn quantize_zero_stays_zero() {
        let (q, dq) = quantize(&[0; 16], 100);
        assert_eq!(q, [0; 16]);
        assert_eq!(dq, [0; 16]);
    }

    #[test]
    fn quantize_dequant_roundtrip_identity_on_multiples() {
        let qindex = 40;
        let dc = dc_quant(qindex);
        let ac = ac_quant(qindex);
        let mut coeff = [0i32; 16];
        coeff[0] = dc * 3;
        coeff[1] = -ac * 2;
        coeff[5] = ac * 7;
        let (q, dq) = quantize(&coeff, qindex);
        assert_eq!(q[0], 3);
        assert_eq!(q[1], -2);
        assert_eq!(q[5], 7);
        assert_eq!(dq[0], coeff[0]);
        assert_eq!(dq[1], coeff[1]);
        assert_eq!(dq[5], coeff[5]);
    }

    #[test]
    fn quantize_rounds_half_up() {
        // quant=10: 5 → 1, 4 → 0
        let (q, dq) = quantize_coeff(5, 10);
        assert_eq!((q, dq), (1, 10));
        let (q, dq) = quantize_coeff(4, 10);
        assert_eq!((q, dq), (0, 0));
        let (q, dq) = quantize_coeff(-5, 10);
        assert_eq!((q, dq), (-1, -10));
    }

    #[test]
    fn dc_uses_dc_table_ac_uses_ac() {
        let qindex = 120;
        assert_ne!(dc_quant(qindex), ac_quant(qindex));
        let mut coeff = [0i32; 16];
        // Mid-way value that survives both quants differently.
        let mid = (dc_quant(qindex) + ac_quant(qindex)) * 2;
        coeff[0] = mid;
        coeff[1] = mid;
        let (q, _) = quantize(&coeff, qindex);
        assert_eq!(q[0], (mid + dc_quant(qindex) / 2) / dc_quant(qindex));
        assert_eq!(q[1], (mid + ac_quant(qindex) / 2) / ac_quant(qindex));
    }
}
