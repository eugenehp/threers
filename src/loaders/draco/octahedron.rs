//! Normals on an octahedron.
//!
//! A unit normal has two degrees of freedom, so Draco stores two numbers rather
//! than three. It projects the normal onto the surface of an octahedron and
//! unwraps that surface into a square: the central diamond of the square is the
//! `+x` hemisphere and the four corner triangles are the `-x` one, folded out.
//! Decoding is the reverse, with one wrinkle — a point outside the diamond has to
//! be mirrored back across the nearest diagonal, which is what the `x_offset`
//! term does.
//!
//! The integer arithmetic here is deliberately done in `u32` where Draco does.
//! It is not an optimisation: on malformed input the doubling in `invert_diamond`
//! overflows, which is undefined in C++ and a panic in Rust, and both
//! implementations have to produce the same wrapped answer for a file that
//! happens to contain such a value.

use super::DracoError;

/// The octahedral grid: how many steps across, and where its centre sits.
#[derive(Debug, Clone, Copy)]
pub struct OctahedronToolBox {
    quantization_bits: i32,
    max_quantized_value: i32,
    max_value: i32,
    dequantization_scale: f32,
    center_value: i32,
}

impl OctahedronToolBox {
    /// Draco's `SetQuantizationBits`. Fewer than two bits leaves no grid at all.
    pub fn new(bits: i32) -> Result<Self, DracoError> {
        if !(2..=30).contains(&bits) {
            return Err(DracoError::Corrupt);
        }
        let max_quantized_value = ((1u32 << bits) - 1) as i32;
        let max_value = max_quantized_value - 1;
        Ok(OctahedronToolBox {
            quantization_bits: bits,
            max_quantized_value,
            max_value,
            dequantization_scale: 2.0 / max_value as f32,
            center_value: max_value / 2,
        })
    }

    pub fn quantization_bits(&self) -> i32 {
        self.quantization_bits
    }

    pub fn center_value(&self) -> i32 {
        self.center_value
    }

    pub fn max_quantized_value(&self) -> i32 {
        self.max_quantized_value
    }

    /// Whether `(s, t)`, measured from the centre, is inside the diamond — that
    /// is, on the near hemisphere.
    pub fn is_in_diamond(&self, s: i32, t: i32) -> bool {
        s.unsigned_abs() + t.unsigned_abs() <= self.center_value as u32
    }

    /// Fold a point across the nearest diagonal, between the two hemispheres.
    pub fn invert_diamond(&self, s: &mut i32, t: &mut i32) {
        let (sign_s, sign_t) = if *s >= 0 && *t >= 0 {
            (1i32, 1i32)
        } else if *s <= 0 && *t <= 0 {
            (-1, -1)
        } else {
            (if *s > 0 { 1 } else { -1 }, if *t > 0 { 1 } else { -1 })
        };
        let corner_s = (sign_s * self.center_value) as u32;
        let corner_t = (sign_t * self.center_value) as u32;
        let mut us = *s as u32;
        let mut ut = *t as u32;
        us = us.wrapping_add(us).wrapping_sub(corner_s);
        ut = ut.wrapping_add(ut).wrapping_sub(corner_t);
        if sign_s * sign_t >= 0 {
            let temp = us;
            us = ut.wrapping_neg();
            ut = temp.wrapping_neg();
        } else {
            std::mem::swap(&mut us, &mut ut);
        }
        us = us.wrapping_add(corner_s);
        ut = ut.wrapping_add(corner_t);
        // Reinterpreted as signed and then halved, exactly as Draco does.
        *s = (us as i32) / 2;
        *t = (ut as i32) / 2;
    }

    /// Bring a value back inside the grid, wrapping at the far edge.
    pub fn mod_max(&self, x: i32) -> i32 {
        if x > self.center_value {
            x - self.max_quantized_value
        } else if x < -self.center_value {
            x + self.max_quantized_value
        } else {
            x
        }
    }

    /// Move edge and corner points to the one representative Draco chose.
    ///
    /// The unwrapped square has edges that are the *same* edge of the octahedron
    /// twice over, so a point on one is ambiguous. Every such point is mapped to
    /// a single canonical position; without it the predictor and the encoder pick
    /// different sides and the correction is off by the width of the grid.
    pub fn canonicalize_octahedral_coords(&self, s: i32, t: i32) -> (i32, i32) {
        let (mut s, mut t) = (s, t);
        let max = self.max_value;
        let center = self.center_value;
        if (s == 0 && t == 0) || (s == 0 && t == max) || (s == max && t == 0) {
            s = max;
            t = max;
        } else if s == 0 && t > center {
            t = center - (t - center);
        } else if s == max && t < center {
            t = center + (center - t);
        } else if t == max && s < center {
            s = center + (center - s);
        } else if t == 0 && s > center {
            s = center - (s - center);
        }
        (s, t)
    }

    /// An integer direction whose components sum to the centre value, as
    /// octahedral coordinates.
    pub fn integer_vector_to_quantized_octahedral_coords(&self, v: &[i32; 3]) -> (i32, i32) {
        let (s, t) = if v[0] >= 0 {
            (v[1] + self.center_value, v[2] + self.center_value)
        } else {
            let s = if v[1] < 0 {
                v[2].abs()
            } else {
                self.max_value - v[2].abs()
            };
            let t = if v[2] < 0 {
                v[1].abs()
            } else {
                self.max_value - v[1].abs()
            };
            (s, t)
        };
        self.canonicalize_octahedral_coords(s, t)
    }

    /// Scale an integer direction so its components sum to the centre value.
    pub fn canonicalize_integer_vector(&self, v: &mut [i32; 3]) {
        let abs_sum = v[0].abs() as i64 + v[1].abs() as i64 + v[2].abs() as i64;
        if abs_sum == 0 {
            v[0] = self.center_value; // The other two are already zero.
            return;
        }
        v[0] = ((v[0] as i64 * self.center_value as i64) / abs_sum) as i32;
        v[1] = ((v[1] as i64 * self.center_value as i64) / abs_sum) as i32;
        let remainder = self.center_value - v[0].abs() - v[1].abs();
        v[2] = if v[2] >= 0 { remainder } else { -remainder };
    }

    /// Octahedral coordinates back to a unit normal.
    pub fn to_unit_vector(&self, s: i32, t: i32) -> [f32; 3] {
        let scaled_s = s as f32 * self.dequantization_scale - 1.0;
        let scaled_t = t as f32 * self.dequantization_scale - 1.0;
        let mut y = scaled_s;
        let mut z = scaled_t;
        // `x` is a signed distance from the diamond's diagonals: positive inside
        // (near hemisphere), negative outside.
        let x = 1.0 - y.abs() - z.abs();
        let offset = (-x).max(0.0);
        // Nothing happens inside the diamond; outside, this mirrors the point
        // back across the nearest diagonal edge.
        y += if y < 0.0 { offset } else { -offset };
        z += if z < 0.0 { offset } else { -offset };
        let norm_squared = x * x + y * y + z * z;
        if norm_squared < 1e-6 {
            [0.0, 0.0, 0.0]
        } else {
            let d = 1.0 / norm_squared.sqrt();
            [x * d, y * d, z * d]
        }
    }
}

/// The prediction transform for canonicalized octahedral normals.
///
/// A correction cannot simply be added: the grid wraps, and a prediction outside
/// the diamond has to be folded in and rotated into one quadrant first, so that
/// corrections are small in the same way everywhere. The rotation is then undone
/// on the result.
#[derive(Debug, Clone, Copy)]
pub struct OctahedronPredictionTransform {
    tools: OctahedronToolBox,
}

impl OctahedronPredictionTransform {
    pub fn decode(max_quantized_value: i32, _center_value: i32) -> Result<Self, DracoError> {
        // The grid width is always 2^b - 1, so an even value is a corrupt file
        // rather than a grid this code does not handle.
        if max_quantized_value <= 0 || max_quantized_value % 2 == 0 {
            return Err(DracoError::Corrupt);
        }
        let bits = 32 - max_quantized_value.leading_zeros() as i32; // MSB + 1
        Ok(OctahedronPredictionTransform {
            tools: OctahedronToolBox::new(bits)?,
        })
    }

    pub fn quantization_bits(&self) -> i32 {
        self.tools.quantization_bits()
    }

    /// Corrections are kept positive by this transform, so they are *not*
    /// zig-zag coded — unlike every other scheme here.
    pub fn corrections_are_positive(&self) -> bool {
        true
    }

    fn is_in_bottom_left(p: [i32; 2]) -> bool {
        if p[0] == 0 && p[1] == 0 {
            return true;
        }
        p[0] < 0 && p[1] <= 0
    }

    fn rotation_count(p: [i32; 2]) -> i32 {
        let (x, y) = (p[0], p[1]);
        if x == 0 {
            if y == 0 {
                0
            } else if y > 0 {
                3
            } else {
                1
            }
        } else if x > 0 {
            if y >= 0 {
                2
            } else {
                1
            }
        } else if y <= 0 {
            0
        } else {
            3
        }
    }

    fn rotate(p: [i32; 2], count: i32) -> [i32; 2] {
        match count {
            1 => [p[1], -p[0]],
            2 => [-p[0], -p[1]],
            3 => [-p[1], p[0]],
            _ => p,
        }
    }

    pub fn compute_original(&self, predicted: &[i32], corr: &[i32], out: &mut [i32]) {
        let center = self.tools.center_value();
        let mut pred = [predicted[0] - center, predicted[1] - center];
        let in_diamond = self.tools.is_in_diamond(pred[0], pred[1]);
        if !in_diamond {
            let (mut s, mut t) = (pred[0], pred[1]);
            self.tools.invert_diamond(&mut s, &mut t);
            pred = [s, t];
        }
        let in_bottom_left = Self::is_in_bottom_left(pred);
        let rotation = Self::rotation_count(pred);
        if !in_bottom_left {
            pred = Self::rotate(pred, rotation);
        }
        let mut orig = [
            self.tools
                .mod_max((pred[0] as u32).wrapping_add(corr[0] as u32) as i32),
            self.tools
                .mod_max((pred[1] as u32).wrapping_add(corr[1] as u32) as i32),
        ];
        if !in_bottom_left {
            orig = Self::rotate(orig, (4 - rotation) % 4);
        }
        if !in_diamond {
            let (mut s, mut t) = (orig[0], orig[1]);
            self.tools.invert_diamond(&mut s, &mut t);
            orig = [s, t];
        }
        out[0] = orig[0] + center;
        out[1] = orig[1] + center;
    }
}
