use super::Matrix4;

/// A 3x3 column-major matrix (same layout as three.js's `Matrix3`).
///
/// `elements[0..3]` is column 0, `elements[3..6]` is column 1, `elements[6..9]` is column 2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3 {
    pub elements: [f32; 9],
}

impl Default for Matrix3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Matrix3 {
    pub const fn identity() -> Self {
        Self {
            elements: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let a = &self.elements;
        let b = &other.elements;
        let mut r = [0.0f32; 9];
        for col in 0..3 {
            for row in 0..3 {
                r[col * 3 + row] =
                    a[row] * b[col * 3] + a[row + 3] * b[col * 3 + 1] + a[row + 6] * b[col * 3 + 2];
            }
        }
        Self { elements: r }
    }

    pub fn transpose(&self) -> Self {
        let m = &self.elements;
        Self {
            elements: [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]],
        }
    }

    /// Inverse via cofactor expansion. Returns identity if singular.
    pub fn invert(&self) -> Self {
        let m = &self.elements;
        let n11 = m[0];
        let n21 = m[1];
        let n31 = m[2];
        let n12 = m[3];
        let n22 = m[4];
        let n32 = m[5];
        let n13 = m[6];
        let n23 = m[7];
        let n33 = m[8];

        let t11 = n33 * n22 - n32 * n23;
        let t12 = n32 * n13 - n33 * n12;
        let t13 = n23 * n12 - n22 * n13;

        let det = n11 * t11 + n21 * t12 + n31 * t13;
        if det == 0.0 {
            return Self::identity();
        }
        let inv = 1.0 / det;
        Self {
            elements: [
                t11 * inv,
                (n31 * n23 - n33 * n21) * inv,
                (n32 * n21 - n31 * n22) * inv,
                t12 * inv,
                (n33 * n11 - n31 * n13) * inv,
                (n31 * n12 - n32 * n11) * inv,
                t13 * inv,
                (n21 * n13 - n23 * n11) * inv,
                (n22 * n11 - n21 * n12) * inv,
            ],
        }
    }

    /// Extract the upper-left 3x3 of a Matrix4.
    pub fn from_matrix4(m4: &Matrix4) -> Self {
        let m = &m4.elements;
        Self {
            elements: [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]],
        }
    }

    /// Normal matrix: transpose of the inverse of the upper-left 3x3.
    /// Matches three.js's `Matrix3.getNormalMatrix(Matrix4)`.
    pub fn normal_matrix(m4: &Matrix4) -> Self {
        Self::from_matrix4(m4).invert().transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector3;

    #[test]
    fn identity_times_identity() {
        let i = Matrix3::identity();
        assert_eq!(i.multiply(&i), i);
    }

    #[test]
    fn transpose_of_transpose_is_original() {
        let m = Matrix3 {
            elements: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        assert_eq!(m.transpose().transpose(), m);
    }

    #[test]
    fn normal_matrix_of_translation_is_identity() {
        let t = Matrix4::translation(Vector3::new(7.0, -3.0, 2.0));
        let n = Matrix3::normal_matrix(&t);
        let id = Matrix3::identity();
        for i in 0..9 {
            assert!(
                (n.elements[i] - id.elements[i]).abs() < 1e-5,
                "i={} got {} want {}",
                i,
                n.elements[i],
                id.elements[i]
            );
        }
    }
}
