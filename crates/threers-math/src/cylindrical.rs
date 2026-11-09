use super::Vector3;

/// Cylindrical coordinates (radius around Y, theta around Y, height y).
/// Matches three.js's `Cylindrical`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylindrical {
    pub radius: f32,
    pub theta: f32,
    pub y: f32,
}

impl Default for Cylindrical {
    fn default() -> Self {
        Self {
            radius: 1.0,
            theta: 0.0,
            y: 0.0,
        }
    }
}

impl Cylindrical {
    pub const fn new(radius: f32, theta: f32, y: f32) -> Self {
        Self { radius, theta, y }
    }

    pub fn from_vector3(v: Vector3) -> Self {
        Self::from_cartesian_coords(v.x, v.y, v.z)
    }

    pub fn from_cartesian_coords(x: f32, y: f32, z: f32) -> Self {
        Self {
            radius: (x * x + z * z).sqrt(),
            theta: x.atan2(z),
            y,
        }
    }

    pub fn to_vector3(&self) -> Vector3 {
        Vector3::new(
            self.radius * self.theta.sin(),
            self.y,
            self.radius * self.theta.cos(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_cartesian() {
        let v = Vector3::new(2.0, -1.0, 3.0);
        let c = Cylindrical::from_vector3(v);
        let back = c.to_vector3();
        assert!((back.x - v.x).abs() < 1e-5);
        assert!((back.y - v.y).abs() < 1e-5);
        assert!((back.z - v.z).abs() < 1e-5);
    }
}
