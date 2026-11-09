/// A typed vertex attribute, equivalent to three.js's `BufferAttribute`.
///
/// `item_size` is the number of components per vertex (e.g. 3 for position).
/// Components are f32, what the vertex stage reads; the `from_*`
/// constructors bring f64 and f16 data in, each value rounded once.
#[derive(Debug, Clone)]
pub struct BufferAttribute {
    pub array: Vec<f32>,
    pub item_size: usize,
}

impl BufferAttribute {
    pub fn new(array: Vec<f32>, item_size: usize) -> Self {
        Self { array, item_size }
    }

    /// f64 components narrowed to the nearest f32. Far from zero this loses
    /// what f32 cannot hold (6 cm steps a million metres out); use
    /// [`from_f64_relative`](Self::from_f64_relative) for such data.
    pub fn from_f64(array: &[f64], item_size: usize) -> Self {
        Self::new(array.iter().map(|v| *v as f32).collect(), item_size)
    }

    /// f64 components less `origin` (one value per component), subtracted in
    /// f64 and then narrowed — what keeps far-away geometry precise. Place
    /// the mesh at `origin` to put it back.
    pub fn from_f64_relative(array: &[f64], item_size: usize, origin: &[f64]) -> Self {
        Self::new(
            crate::math::float::f64_relative_to_f32(array, item_size, origin),
            item_size,
        )
    }

    /// Binary16 components (raw bits, as a `Float16Array` or a half-float
    /// buffer holds them), widened exactly.
    pub fn from_f16_bits(bits: &[u16], item_size: usize) -> Self {
        Self::new(
            bits.iter()
                .map(|b| crate::math::float::f16_to_f32(*b))
                .collect(),
            item_size,
        )
    }

    pub fn count(&self) -> usize {
        self.array.len() / self.item_size
    }

    /// The components as f64s.
    pub fn to_f64(&self) -> Vec<f64> {
        self.array.iter().map(|v| *v as f64).collect()
    }

    /// The components rounded to binary16 bits (to nearest, ties to even),
    /// for a half-float vertex buffer or export.
    pub fn to_f16_bits(&self) -> Vec<u16> {
        self.array
            .iter()
            .map(|v| crate::math::float::f32_to_f16(*v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_and_f16_in_and_out() {
        let far = [1_000_000.123_456, 2_000_000.654_321, 3.0];
        let plain = BufferAttribute::from_f64(&far, 3);
        let rel = BufferAttribute::from_f64_relative(&far, 3, &[1_000_000.0, 2_000_000.0, 0.0]);
        assert!(
            (plain.array[0] as f64 - far[0]).abs() > 1e-3,
            "f32 cannot hold it directly"
        );
        assert!((rel.array[0] as f64 + 1_000_000.0 - far[0]).abs() < 1e-6);
        assert!((rel.array[1] as f64 + 2_000_000.0 - far[1]).abs() < 1e-6);
        let half = BufferAttribute::from_f16_bits(&[0x3c00, 0xc000, 0x7bff], 1);
        assert_eq!(half.array, [1.0, -2.0, 65504.0]);
        assert_eq!(half.to_f16_bits(), [0x3c00, 0xc000, 0x7bff]);
        assert_eq!(half.to_f64(), [1.0, -2.0, 65504.0]);
    }
}
