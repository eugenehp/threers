//! Head-gaze dwell: where the head points (the shell's ARKit device
//! anchor, [`crate::threers_ios_head`]) on the room's pieces that ask for
//! it ([`Piece::dwell`](crate::Piece::dwell)), and the control there
//! pressed once it has been pointed at long enough. visionOS keeps where
//! the eyes look to itself; the head's direction is the app's to read.

use threers::math::{Quaternion, Vector3};
use threers::ui::native::Control;

use crate::{Piece, PieceIn};

/// Where a ray from the head meets a piece: which, the UI pixel, the world
/// point and the piece's facing, and how far.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Hit {
    pub piece: u32,
    pub x: f32,
    pub y: f32,
    pub at: Vector3,
    pub normal: Vector3,
    pub distance: f32,
}

/// The nearest of `pieces` (in the room, dwelt on) that a ray from `head`
/// along its forward (`-z`) meets, from the front. A piece that faces you
/// is taken as facing the head.
pub(crate) fn hit(head: (Vector3, Quaternion), pieces: &[Piece]) -> Option<Hit> {
    let (from, turn) = head;
    let ahead = Vector3::new(0.0, 0.0, -1.0).apply_quaternion(turn).normalize();
    let mut best: Option<Hit> = None;
    for p in pieces.iter().filter(|p| p.at == PieceIn::Room && p.dwell > 0.0) {
        let rotation = if p.faces_you { facing(p.position, from) } else { p.rotation };
        let normal = Vector3::new(0.0, 0.0, 1.0).apply_quaternion(rotation);
        let towards = ahead.dot(normal);
        // (from behind, or along it)
        if towards > -1e-4 {
            continue;
        }
        let t = (p.position - from).dot(normal) / towards;
        if t <= 0.0 || !t.is_finite() {
            continue;
        }
        let at = from + ahead * t;
        let local = (at - p.position).apply_quaternion(rotation.conjugate());
        let (w, h) = (p.width, p.height);
        let x = local.x / p.metres_per_pixel + w / 2.0;
        let y = h / 2.0 - local.y / p.metres_per_pixel;
        if !(0.0..=w).contains(&x) || !(0.0..=h).contains(&y) {
            continue;
        }
        if best.is_none_or(|b| t < b.distance) {
            best = Some(Hit { piece: p.id, x, y, at, normal, distance: t });
        }
    }
    best
}

/// A turn about `y` that faces a piece at `at` towards `head`.
fn facing(at: Vector3, head: Vector3) -> Quaternion {
    let d = head - at;
    Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), d.x.atan2(d.z))
}

/// The control at `(x, y)` that dwelling presses: pressed ones (not slid,
/// not typed into, not disabled), the smallest that holds the point.
pub(crate) fn control_at(controls: &[Control], x: f32, y: f32) -> Option<&Control> {
    controls
        .iter()
        .filter(|c| !c.slider && !c.editable && !c.disabled)
        .filter(|c| {
            let r = c.rect;
            x >= r.x && x <= r.x + r.width && y >= r.y && y <= r.y + r.height
        })
        .min_by(|a, b| (a.rect.width * a.rect.height).total_cmp(&(b.rect.width * b.rect.height)))
}

/// What is dwelt on, since when, and whether it has been pressed.
#[derive(Debug, Default)]
pub(crate) struct Dwell {
    target: Option<(u32, u64)>,
    since: f64,
    pressed: bool,
}

impl Dwell {
    /// Pointed at `target` (a piece and one of its controls) at `now`, for
    /// `wait` seconds to press: how far along (0 to 1), and whether to
    /// press now. Once pressed, it waits till something else is pointed at.
    pub fn step(&mut self, target: Option<(u32, u64)>, now: f64, wait: f32) -> (f32, bool) {
        if target != self.target {
            *self = Dwell { target, since: now, pressed: false };
        }
        if target.is_none() || self.pressed {
            return (if self.pressed { 1.0 } else { 0.0 }, false);
        }
        let k = ((now - self.since) / f64::from(wait.max(0.05))) as f32;
        if k >= 1.0 {
            self.pressed = true;
            return (1.0, true);
        }
        (k.max(0.0), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(id: u32, at: Vector3) -> Piece {
        let mut p = Piece::new(id, PieceIn::Room, 200.0, 100.0, at);
        p.metres_per_pixel = 0.001;
        p.dwell = 1.0;
        p
    }

    #[test]
    fn the_head_points_at_a_pixel() {
        let head = (Vector3::new(0.0, 1.5, 0.0), Quaternion::IDENTITY);
        let p = piece(3, Vector3::new(0.0, 1.5, -1.0));
        let h = hit(head, &[p]).expect("straight ahead");
        assert_eq!(h.piece, 3);
        assert!((h.x - 100.0).abs() < 1e-3 && (h.y - 50.0).abs() < 1e-3, "{h:?}");
        assert!((h.distance - 1.0).abs() < 1e-4);
        // a little up and to the left: up the piece, to its left
        let turn = Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 0.02)
            * Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), 0.02);
        let h = hit((head.0, turn), &[p]).unwrap();
        assert!(h.x < 100.0 - 15.0 && h.y < 50.0 - 15.0, "{h:?}");
        // turned away, behind it, or one not dwelt on: none
        let away = Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), std::f32::consts::PI);
        assert!(hit((head.0, away), &[p]).is_none());
        assert!(hit((Vector3::new(0.0, 1.5, -2.0), away), &[p]).is_none(), "from behind");
        let mut off = p;
        off.dwell = 0.0;
        assert!(hit(head, &[off]).is_none());
        // the nearer of two in line
        let near = piece(4, Vector3::new(0.0, 1.5, -0.5));
        assert_eq!(hit(head, &[p, near]).unwrap().piece, 4);
    }

    #[test]
    fn a_turned_piece_is_hit_in_its_own_pixels() {
        let mut p = piece(1, Vector3::new(1.0, 1.5, -1.0));
        p.rotation = Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), -std::f32::consts::FRAC_PI_4);
        // looked at square on, along its normal back: its middle
        let normal = Vector3::new(0.0, 0.0, 1.0).apply_quaternion(p.rotation);
        let from = p.position + normal * 2.0;
        let look = Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), normal.x.atan2(normal.z));
        let h = hit((from, look), &[p]).unwrap();
        assert!((h.x - 100.0).abs() < 0.01 && (h.y - 50.0).abs() < 0.01, "{h:?}");
        // one that faces you, from anywhere in front of it
        let mut f = piece(2, Vector3::new(0.0, 1.5, -1.0));
        f.faces_you = true;
        let from = Vector3::new(1.0, 1.5, 0.0);
        let d = f.position - from;
        let look = Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), (-d.x).atan2(-d.z));
        let h = hit((from, look), &[f]).unwrap();
        assert!((h.x - 100.0).abs() < 0.5 && (h.y - 50.0).abs() < 0.5, "{h:?}");
    }

    /// Shown twice its size (a volume's scale), a metre of it is half the
    /// pixels.
    #[test]
    fn a_scaled_piece_is_hit_in_its_own_pixels() {
        let mut p = piece(5, Vector3::new(0.0, 1.5, -1.0));
        p.metres_per_pixel *= 2.0;
        let turn = Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), -(0.1f32).atan());
        let h = hit((Vector3::new(0.0, 1.5, 0.0), turn), &[p]).unwrap();
        assert!((h.x - 150.0).abs() < 0.5, "0.1 m right of the middle at 2 mm a pixel: {h:?}");
    }

    #[test]
    fn dwelling_presses_once() {
        let mut d = Dwell::default();
        let t = Some((1, 7));
        assert_eq!(d.step(t, 10.0, 1.0), (0.0, false));
        let (k, press) = d.step(t, 10.5, 1.0);
        assert!((k - 0.5).abs() < 1e-6 && !press);
        assert_eq!(d.step(t, 11.0, 1.0), (1.0, true));
        assert_eq!(d.step(t, 12.0, 1.0), (1.0, false), "once");
        // away and back: again
        assert_eq!(d.step(None, 12.1, 1.0), (0.0, false));
        assert_eq!(d.step(t, 12.2, 1.0), (0.0, false));
        assert_eq!(d.step(t, 13.2, 1.0), (1.0, true));
        // moving to another restarts it
        d.step(Some((1, 8)), 13.3, 1.0);
        assert!(!d.step(Some((1, 8)), 13.9, 1.0).1);
    }
}
