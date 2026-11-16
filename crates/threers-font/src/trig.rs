//! FreeType's fixed-point trigonometry (CORDIC): what `FT_Vector_Length`
//! returns, used for the length of a component's scale.

/// The CORDIC gain correction, 0.858785336480436 in 32-bit fixed point.
const TRIG_SCALE: u64 = 0xDBD9_5B16;

/// The arctangents of 2^-i in 16.16 degrees.
const ARCTAN: [i64; 22] = [
    1740967, 919879, 466945, 234379, 117304, 58666, 29335, 14668, 7334, 3667, 1833, 917, 458, 229, 115, 57, 29, 14, 7, 4, 2,
    1,
];

/// The most significant bit kept in the pseudo-rotations.
const SAFE_MSB: i32 = 29;

fn msb(v: u64) -> i32 {
    63 - v.leading_zeros() as i32
}

/// The vector scaled so its larger coordinate has `SAFE_MSB` bits, and the
/// shift that did it.
fn prenorm(x: i64, y: i64) -> (i64, i64, i32) {
    let shift = msb((x.unsigned_abs()) | (y.unsigned_abs()));

    if shift <= SAFE_MSB {
        let s = SAFE_MSB - shift;
        (x << s, y << s, s)
    } else {
        let s = shift - SAFE_MSB;
        (x >> s, y >> s, -s)
    }
}

/// The length the pseudo-rotations leave in `x`, before the gain is taken off.
fn pseudo_polarize(mut x: i64, mut y: i64) -> i64 {
    if y > x {
        if y > -x {
            let t = y;
            y = -x;
            x = t;
        } else {
            x = -x;
            y = -y;
        }
    } else if y < -x {
        let t = -y;
        y = x;
        x = t;
    }

    let mut b = 1i64;

    for i in 1..23 {
        let t = if y > 0 { x + ((y + b) >> i) } else { x - ((y + b) >> i) };
        y = if y > 0 { y - ((x + b) >> i) } else { y + ((x + b) >> i) };
        x = t;
        b <<= 1;
        let _ = ARCTAN[(i - 1) as usize];
    }

    x
}

/// The gain taken off a pseudo-rotated value.
fn downscale(v: i64) -> i64 {
    let s = v < 0;
    let r = ((v.unsigned_abs() as u128 * u128::from(TRIG_SCALE) + 0x1_0000_0000) >> 32) as i64;
    if s { -r } else { r }
}

/// `FT_Vector_Length`
pub fn vector_length(x: i64, y: i64) -> i64 {
    if x == 0 {
        return y.abs();
    }

    if y == 0 {
        return x.abs();
    }

    let (px, py, shift) = prenorm(x, y);
    let len = downscale(pseudo_polarize(px, py));

    if shift > 0 { (len + (1 << (shift - 1))) >> shift } else { len << -shift }
}
