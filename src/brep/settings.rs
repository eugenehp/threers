//! Kernel settings: the numbers the kernel chooses for itself.
//!
//! Operations take their tolerance as an argument. What is left are the few
//! choices no argument carries, and they are stated here rather than as
//! constants where they are used, so a caller can change them:
//!
//! ```ignore
//! use threers::brep::{with_settings, Body, Settings};
//!
//! let fine = Settings { rim_segments: 64, ..Settings::default() };
//! let body = with_settings(fine, || Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0));
//! ```
//!
//! Settings are per thread and scoped: [`with_settings`] applies them for
//! the duration of a closure and puts the previous ones back afterwards,
//! also when the closure panics. The sampling settings change how finely the
//! kernel works, not what a body *is*; [`ChamferCorner`] is the one that
//! chooses between two different results.

use std::cell::Cell;

/// What a chamfer leaves where three bevelled edges meet at a corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChamferCorner {
    /// The three bevels run on until they meet in a point.
    #[default]
    Point,
    /// The corner is cut off by a triangular facet through the three points
    /// where the bevels' edges meet on the original faces, which is what
    /// OpenCASCADE-based CAD systems produce. It removes a further
    /// `d³ / 12` at a right-angled corner bevelled by `d`.
    Facet,
}

/// See the [module documentation](self).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settings {
    /// Vertices the built-in primitives ([`super::Body::cylinder`], `cone`,
    /// `frustum`, `plate_with_holes`) put on a circular rim before any
    /// refinement. At least 3.
    pub rim_segments: usize,
    /// How many times an edge segment may be halved to bring it within
    /// tolerance of the two surfaces it lies between
    /// ([`super::Body::refine_edges`]). Each level doubles the vertices a
    /// segment can gain, so 10 allows 1024.
    pub max_refine_depth: usize,
    /// The shape of a corner where three chamfers meet.
    pub chamfer_corner: ChamferCorner,
    /// Whether a boolean goes through the general fuse ([`super::fuse`])
    /// before the face-splitting boolean, which then takes only what the fuse
    /// declines. Off, every boolean takes the older path.
    pub general_fuse: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            rim_segments: 16,
            max_refine_depth: 10,
            chamfer_corner: ChamferCorner::Point,
            general_fuse: true,
        }
    }
}

thread_local! {
    static CURRENT: Cell<Settings> = Cell::new(Settings::default());
}

/// The settings in force on this thread.
pub fn settings() -> Settings {
    CURRENT.with(Cell::get)
}

/// Run `f` with `settings` in force on this thread, then restore the
/// settings that were in force before.
pub fn with_settings<R>(settings: Settings, f: impl FnOnce() -> R) -> R {
    struct Restore(Settings);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(CURRENT.with(|c| c.replace(settings)));
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_and_restored() {
        assert_eq!(settings(), Settings::default());
        let fine = Settings {
            rim_segments: 64,
            ..Settings::default()
        };
        let inner = with_settings(fine, || {
            assert_eq!(settings().rim_segments, 64);
            // Nested scopes restore to the enclosing one.
            with_settings(Settings::default(), || {
                assert_eq!(settings().rim_segments, 16)
            });
            settings().rim_segments
        });
        assert_eq!(inner, 64);
        assert_eq!(settings(), Settings::default());
        // Restored after a panic too.
        let caught = std::panic::catch_unwind(|| with_settings(fine, || panic!("inside")));
        assert!(caught.is_err());
        assert_eq!(settings(), Settings::default());
    }
}
