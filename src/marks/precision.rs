//! Positions kept in `f64` on the CPU and drawn relative to a movable origin.
//!
//! An `f32` has 24 bits of mantissa, so at ten million units from the origin
//! the spacing between representable values is a whole unit. Web Mercator in
//! metres puts every city millions of metres out; a scene of street furniture
//! there snaps to a one-metre grid and visibly shakes as the camera moves.
//!
//! The fix is the one deck.gl and most globe renderers use. Each coordinate is
//! split into a high part — the nearest `f32` — and a low part, the `f32` of
//! what the high part missed. Both go to the GPU once. The *origin* is split
//! the same way and sent as a uniform, and the shader subtracts high from high
//! and low from low before adding. Two nearby `f32`s subtract exactly (the
//! Sterbenz lemma), so the difference keeps the precision the `f64` had, and
//! moving the origin is a uniform write rather than a re-upload of every point.
//!
//! The camera and any ordinary meshes then live in *origin-relative*
//! coordinates, which are small. Re-centre ([`Origin::rebase`]) when the camera
//! has travelled far enough that those stop being small.

/// `v` as an `f32` pair whose sum is `v` to about 48 bits.
///
/// ```
/// use threers::marks::split_f64;
/// let (hi, lo) = split_f64(12_345_678.901_234);
/// assert!((hi as f64 + lo as f64 - 12_345_678.901_234).abs() < 1e-6);
/// ```
pub fn split_f64(v: f64) -> (f32, f32) {
    let hi = v as f32;
    let lo = (v - hi as f64) as f32;
    (hi, lo)
}

/// [`split_f64`] for each component: `([hi; 3], [lo; 3])`.
pub fn split_f64x3(v: [f64; 3]) -> ([f32; 3], [f32; 3]) {
    let (x, xl) = split_f64(v[0]);
    let (y, yl) = split_f64(v[1]);
    let (z, zl) = split_f64(v[2]);
    ([x, y, z], [xl, yl, zl])
}

/// The point that origin-relative coordinates are measured from.
///
/// Marks store absolute `f64` positions and subtract this on the GPU. A camera
/// and meshes that sit beside marks must be placed with [`to_local`](Self::to_local).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Origin {
    pub position: [f64; 3],
}

impl Origin {
    pub const ZERO: Origin = Origin { position: [0.0; 3] };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            position: [x, y, z],
        }
    }

    /// An absolute position, relative to this origin, as the `f32` the camera
    /// and meshes use.
    pub fn to_local(&self, p: [f64; 3]) -> crate::math::Vector3 {
        crate::math::Vector3::new(
            (p[0] - self.position[0]) as f32,
            (p[1] - self.position[1]) as f32,
            (p[2] - self.position[2]) as f32,
        )
    }

    /// The absolute position of an origin-relative point.
    pub fn to_world(&self, p: crate::math::Vector3) -> [f64; 3] {
        [
            self.position[0] + p.x as f64,
            self.position[1] + p.y as f64,
            self.position[2] + p.z as f64,
        ]
    }

    /// Move the origin to `new_origin` and return the offset every
    /// origin-relative thing (camera, controls target, meshes) must be moved
    /// by — add it to them to keep them where they were in absolute terms.
    pub fn rebase(&mut self, new_origin: [f64; 3]) -> crate::math::Vector3 {
        let delta = crate::math::Vector3::new(
            (self.position[0] - new_origin[0]) as f32,
            (self.position[1] - new_origin[1]) as f32,
            (self.position[2] - new_origin[2]) as f32,
        );
        self.position = new_origin;
        delta
    }

    /// `(hi, lo)` as the shader's two `vec4`s.
    pub(crate) fn split(&self) -> ([f32; 4], [f32; 4]) {
        let (hi, lo) = split_f64x3(self.position);
        ([hi[0], hi[1], hi[2], 0.0], [lo[0], lo[1], lo[2], 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_recovers_what_a_plain_cast_loses() {
        // 1e7 + 0.25: the f32 grid is a whole unit wide here, so a plain cast
        // lands on 1e7 and the quarter is gone.
        let v = 10_000_000.25;
        assert_eq!(v as f32 as f64, 10_000_000.0);
        let (hi, lo) = split_f64(v);
        assert_eq!(hi as f64 + lo as f64, v);
    }

    #[test]
    fn difference_of_splits_keeps_sub_unit_offsets() {
        // What the shader does: (hi - ohi) + (lo - olo), all in f32.
        let origin = Origin::new(6_378_137.0, -3_000_000.5, 0.0);
        let (ohi, olo) = origin.split();
        for k in 0..16 {
            let p = 6_378_137.0 + k as f64 * 0.125;
            let (hi, lo) = split_f64(p);
            let rel = (hi - ohi[0]) + (lo - olo[0]);
            assert!(
                (rel as f64 - k as f64 * 0.125).abs() < 1e-6,
                "{k}: got {rel}"
            );
        }
    }

    #[test]
    fn rebase_returns_the_shift_for_local_content() {
        let mut o = Origin::new(100.0, 0.0, 0.0);
        let cam = o.to_local([105.0, 2.0, 0.0]);
        let shift = o.rebase([104.0, 0.0, 0.0]);
        let moved = cam + shift;
        assert_eq!(o.to_world(moved), [105.0, 2.0, 0.0]);
    }
}
