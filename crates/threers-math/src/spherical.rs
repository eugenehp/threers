use super::Vector3;

/// Spherical coordinates with `phi` from +Y (polar) and `theta` around Y (azimuth).
/// Matches three.js's `Spherical`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spherical {
    pub radius: f32,
    pub phi: f32,
    pub theta: f32,
}

impl Default for Spherical {
    fn default() -> Self {
        Self {
            radius: 1.0,
            phi: 0.0,
            theta: 0.0,
        }
    }
}

impl Spherical {
    pub const fn new(radius: f32, phi: f32, theta: f32) -> Self {
        Self { radius, phi, theta }
    }

    pub fn from_vector3(v: Vector3) -> Self {
        Self::from_cartesian_coords(v.x, v.y, v.z)
    }

    pub fn from_cartesian_coords(x: f32, y: f32, z: f32) -> Self {
        let radius = (x * x + y * y + z * z).sqrt();
        if radius == 0.0 {
            return Self {
                radius: 0.0,
                phi: 0.0,
                theta: 0.0,
            };
        }
        Self {
            radius,
            phi: (y / radius).clamp(-1.0, 1.0).acos(),
            theta: x.atan2(z),
        }
    }

    /// Clamp phi to avoid the poles (matches three.js's `makeSafe`).
    pub fn make_safe(&self) -> Self {
        const EPS: f32 = 1e-6;
        Self {
            radius: self.radius,
            phi: self.phi.clamp(EPS, std::f32::consts::PI - EPS),
            theta: self.theta,
        }
    }

    pub fn to_vector3(&self) -> Vector3 {
        let sin_phi_r = self.phi.sin() * self.radius;
        Vector3::new(
            sin_phi_r * self.theta.sin(),
            self.phi.cos() * self.radius,
            sin_phi_r * self.theta.cos(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_cartesian() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let s = Spherical::from_vector3(v);
        let back = s.to_vector3();
        assert!((back.x - v.x).abs() < 1e-4, "x: {} vs {}", back.x, v.x);
        assert!((back.y - v.y).abs() < 1e-4, "y: {} vs {}", back.y, v.y);
        assert!((back.z - v.z).abs() < 1e-4, "z: {} vs {}", back.z, v.z);
    }
}
