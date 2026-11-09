use super::{Matrix4, Quaternion};

/// The axis order of an [`Euler`] — three.js's `'XYZ'`, `'YXZ'`, … — naming
/// the *intrinsic* rotation order: `XYZ` rotates about X, then the new Y,
/// then the new Z, i.e. the matrix `Rx · Ry · Rz`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum EulerOrder {
    /// three.js's default.
    #[default]
    XYZ,
    YXZ,
    ZXY,
    ZYX,
    YZX,
    XZY,
}

impl EulerOrder {
    /// All six, in three.js's `Euler` switch order.
    pub const ALL: [Self; 6] = [
        Self::XYZ,
        Self::YXZ,
        Self::ZXY,
        Self::ZYX,
        Self::YZX,
        Self::XZY,
    ];

    /// Parse three.js's order string (`"XYZ"`, …).
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "XYZ" => Self::XYZ,
            "YXZ" => Self::YXZ,
            "ZXY" => Self::ZXY,
            "ZYX" => Self::ZYX,
            "YZX" => Self::YZX,
            "XZY" => Self::XZY,
            _ => return None,
        })
    }

    /// three.js's order string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::XYZ => "XYZ",
            Self::YXZ => "YXZ",
            Self::ZXY => "ZXY",
            Self::ZYX => "ZYX",
            Self::YZX => "YZX",
            Self::XZY => "XZY",
        }
    }
}

/// Euler angles in radians plus their axis order — three.js's `Euler`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Euler {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub order: EulerOrder,
}

impl Euler {
    /// `XYZ` order, three.js's default.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            x,
            y,
            z,
            order: EulerOrder::XYZ,
        }
    }

    pub const fn with_order(x: f32, y: f32, z: f32, order: EulerOrder) -> Self {
        Self { x, y, z, order }
    }

    /// three.js `Quaternion.setFromEuler`, every order.
    pub fn to_quaternion(self) -> Quaternion {
        Quaternion::from_euler(self)
    }

    /// three.js `Euler.setFromRotationMatrix(m, order)`: the upper 3x3 of `m`
    /// must be a pure rotation.
    pub fn from_rotation_matrix(m: &Matrix4, order: EulerOrder) -> Self {
        let te = &m.elements;
        let (m11, m12, m13) = (te[0], te[4], te[8]);
        let (m21, m22, m23) = (te[1], te[5], te[9]);
        let (m31, m32, m33) = (te[2], te[6], te[10]);
        let clamp = |v: f32| v.clamp(-1.0, 1.0);
        const EDGE: f32 = 0.999_999_9;
        let (x, y, z) = match order {
            EulerOrder::XYZ => {
                let y = clamp(m13).asin();
                if m13.abs() < EDGE {
                    ((-m23).atan2(m33), y, (-m12).atan2(m11))
                } else {
                    (m32.atan2(m22), y, 0.0)
                }
            }
            EulerOrder::YXZ => {
                let x = (-clamp(m23)).asin();
                if m23.abs() < EDGE {
                    (x, m13.atan2(m33), m21.atan2(m22))
                } else {
                    (x, (-m31).atan2(m11), 0.0)
                }
            }
            EulerOrder::ZXY => {
                let x = clamp(m32).asin();
                if m32.abs() < EDGE {
                    (x, (-m31).atan2(m33), (-m12).atan2(m22))
                } else {
                    (x, 0.0, m21.atan2(m11))
                }
            }
            EulerOrder::ZYX => {
                let y = (-clamp(m31)).asin();
                if m31.abs() < EDGE {
                    (m32.atan2(m33), y, m21.atan2(m11))
                } else {
                    (0.0, y, (-m12).atan2(m22))
                }
            }
            EulerOrder::YZX => {
                let z = clamp(m21).asin();
                if m21.abs() < EDGE {
                    ((-m23).atan2(m22), (-m31).atan2(m11), z)
                } else {
                    (0.0, m13.atan2(m33), z)
                }
            }
            EulerOrder::XZY => {
                let z = (-clamp(m12)).asin();
                if m12.abs() < EDGE {
                    (m32.atan2(m22), m13.atan2(m11), z)
                } else {
                    ((-m23).atan2(m33), 0.0, z)
                }
            }
        };
        Self { x, y, z, order }
    }

    /// three.js `Euler.setFromQuaternion(q, order)`.
    pub fn from_quaternion(q: Quaternion, order: EulerOrder) -> Self {
        Self::from_rotation_matrix(&Matrix4::from_quaternion(q), order)
    }

    /// three.js `Euler.reorder(order)`: the same rotation in another order.
    pub fn reorder(self, order: EulerOrder) -> Self {
        Self::from_quaternion(self.to_quaternion(), order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Matrix4::makeRotationFromEuler` from three.js r165, every order.
    fn rotation_from_euler(e: Euler) -> [f32; 9] {
        let (a, b) = (e.x.cos(), e.x.sin());
        let (c, d) = (e.y.cos(), e.y.sin());
        let (ee, f) = (e.z.cos(), e.z.sin());
        // Row-major m11..m33.
        match e.order {
            EulerOrder::XYZ => {
                let (ae, af, be, bf) = (a * ee, a * f, b * ee, b * f);
                [
                    c * ee,
                    -c * f,
                    d,
                    af + be * d,
                    ae - bf * d,
                    -b * c,
                    bf - ae * d,
                    be + af * d,
                    a * c,
                ]
            }
            EulerOrder::YXZ => {
                let (ce, cf, de, df) = (c * ee, c * f, d * ee, d * f);
                [
                    ce + df * b,
                    de * b - cf,
                    a * d,
                    a * f,
                    a * ee,
                    -b,
                    cf * b - de,
                    df + ce * b,
                    a * c,
                ]
            }
            EulerOrder::ZXY => {
                let (ce, cf, de, df) = (c * ee, c * f, d * ee, d * f);
                [
                    ce - df * b,
                    -a * f,
                    de + cf * b,
                    cf + de * b,
                    a * ee,
                    df - ce * b,
                    -a * d,
                    b,
                    a * c,
                ]
            }
            EulerOrder::ZYX => {
                let (ae, af, be, bf) = (a * ee, a * f, b * ee, b * f);
                [
                    c * ee,
                    be * d - af,
                    ae * d + bf,
                    c * f,
                    bf * d + ae,
                    af * d - be,
                    -d,
                    b * c,
                    a * c,
                ]
            }
            EulerOrder::YZX => {
                let (ac, ad, bc, bd) = (a * c, a * d, b * c, b * d);
                [
                    c * ee,
                    bd - ac * f,
                    bc * f + ad,
                    f,
                    a * ee,
                    -b * ee,
                    -d * ee,
                    ad * f + bc,
                    ac - bd * f,
                ]
            }
            EulerOrder::XZY => {
                let (ac, ad, bc, bd) = (a * c, a * d, b * c, b * d);
                [
                    c * ee,
                    -f,
                    d * ee,
                    ac * f + bd,
                    a * ee,
                    ad * f - bc,
                    bc * f - ad,
                    b * ee,
                    bd * f + ac,
                ]
            }
        }
    }

    #[test]
    fn quaternion_from_euler_matches_three_rotation_matrix_every_order() {
        for order in EulerOrder::ALL {
            let e = Euler::with_order(0.3, -0.7, 1.1, order);
            let m = Matrix4::from_quaternion(e.to_quaternion()).elements;
            let r = rotation_from_euler(e);
            // Column-major m: m[col*4+row].
            for row in 0..3 {
                for col in 0..3 {
                    let got = m[col * 4 + row];
                    let want = r[row * 3 + col];
                    assert!(
                        (got - want).abs() < 1e-5,
                        "{order:?} [{row}][{col}] {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn euler_round_trips_through_the_quaternion_every_order() {
        for order in EulerOrder::ALL {
            let e = Euler::with_order(0.3, -0.7, 1.1, order);
            let back = Euler::from_quaternion(e.to_quaternion(), order);
            assert!((back.x - e.x).abs() < 1e-5, "{order:?} x");
            assert!((back.y - e.y).abs() < 1e-5, "{order:?} y");
            assert!((back.z - e.z).abs() < 1e-5, "{order:?} z");
            assert_eq!(back.order, order);
        }
    }

    /// Values printed by three.js r165:
    /// `new Euler().setFromQuaternion(new Quaternion().setFromEuler(new Euler(0.3,-0.7,1.1,'XYZ')), o)`.
    #[test]
    // three.js's own reference values, kept digit for digit.
    #[allow(clippy::excessive_precision)]
    fn reorder_matches_three() {
        let want = [
            (EulerOrder::XYZ, [0.3, -0.7, 1.1]),
            (EulerOrder::YXZ, [0.227_996_49, -0.722_593_38, 0.903_296_96]),
            (
                EulerOrder::ZXY,
                [-0.427_329_58, -0.638_685_54, 0.846_528_74],
            ),
            (
                EulerOrder::ZYX,
                [-0.515_951_96, -0.573_449_65, 1.145_056_89],
            ),
            (EulerOrder::YZX, [0.358_624_79, -1.001_864_7, 0.871_115_21]),
            (
                EulerOrder::XZY,
                [-0.602_148_23, -1.076_803_21, 0.749_992_11],
            ),
        ];
        let e = Euler::new(0.3, -0.7, 1.1);
        for (order, w) in want {
            let r = e.reorder(order);
            for (got, want) in [r.x, r.y, r.z].iter().zip(w) {
                assert!((got - want).abs() < 2e-5, "{order:?}: {got} vs {want}");
            }
        }
    }
}
