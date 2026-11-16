//! Fixed-point arithmetic with the rounding FreeType's results carry:
//! 16.16 products and quotients rounded half away from zero, 26.6 pixel
//! coordinates.

/// `a * b / 0x10000`, rounded half away from zero.
pub fn mul_fix(a: i64, b: i64) -> i64 {
    let s = (a < 0) != (b < 0);
    let (a, b) = (a.unsigned_abs(), b.unsigned_abs());
    let r = ((a as u128 * b as u128 + 0x8000) >> 16) as i64;
    if s { -r } else { r }
}

/// `a * 0x10000 / b`, rounded half away from zero; `0x7FFFFFFF` for a
/// zero `b`.
pub fn div_fix(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0x7FFF_FFFF;
    }

    let s = (a < 0) != (b < 0);
    let (a, b) = (a.unsigned_abs() as u128, b.unsigned_abs() as u128);
    let r = (((a << 16) + (b >> 1)) / b) as i64;
    if s { -r } else { r }
}

/// `a * b / c`, rounded half away from zero; the largest value for a zero
/// `c`.
pub fn mul_div(a: i64, b: i64, c: i64) -> i64 {
    let s = ((a < 0) != (b < 0)) != (c < 0);
    let (a, b, c) = (a.unsigned_abs() as u128, b.unsigned_abs() as u128, c.unsigned_abs() as u128);
    let r = (a * b + (c >> 1)).checked_div(c).map_or(0x7FFF_FFFF, |q| q as i64);
    if s { -r } else { r }
}

/// `a * b / c`, truncated toward zero.
pub fn mul_div_no_round(a: i64, b: i64, c: i64) -> i64 {
    let s = ((a < 0) != (b < 0)) != (c < 0);
    let (a, b, c) = (a.unsigned_abs() as u128, b.unsigned_abs() as u128, c.unsigned_abs() as u128);
    let r = (a * b).checked_div(c).map_or(0x7FFF_FFFF, |q| q as i64);
    if s { -r } else { r }
}

/// A 26.6 value rounded to the nearest whole pixel, halves up.
pub fn pix_round(v: i64) -> i64 {
    (v + 32) & !63
}

/// A 26.6 value down to a whole pixel.
pub fn pix_floor(v: i64) -> i64 {
    v & !63
}

/// A 26.6 value up to a whole pixel.
pub fn pix_ceil(v: i64) -> i64 {
    (v + 63) & !63
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding() {
        assert_eq!(div_fix(22928, 2048), 733696);
        assert_eq!(mul_fix(-3, 0x8000), -2);
        assert_eq!(mul_div(5, 3, 2), 8);
        assert_eq!(mul_div(-5, 3, 2), -8);
    }
}
