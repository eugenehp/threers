//! Animation for [`threers`](https://docs.rs/threers): easing, tweens, springs,
//! timelines, cinematic cameras — and blending an animated pose into a simulated one.
//!
//! Pure Rust, no dependencies beyond `threers`, and wasm-ready.
//!
//! ```
//! use threers_animation::prelude::*;
//!
//! let mut fade = Tween::new(0.0f32, 1.0, 0.5).easing(Easing::CubicOut);
//! fade.update(0.25);
//! assert!(fade.value() > 0.5);
//! ```
//!
//! # Which tool for which job
//!
//! | Need | Use |
//! |---|---|
//! | A→B over a known duration | [`Tween`] |
//! | Chase a target that keeps moving | [`Spring`] |
//! | Several things on one clock | [`Timeline`] |
//! | Drive scene nodes directly | [`Animator`] |
//! | Cinematic camera moves / paths / shots | [`camera`] |
//! | Hand a character to the physics solver | [`PhysicsBlend`] (`physics` feature) |
//!
//! The distinction that matters most is the first two. A tween is told how long
//! to take, which is right for scripted motion and wrong for anything reacting
//! to input — retarget a tween mid-flight and it either restarts or jumps. A
//! spring has no duration at all: move its target and it adjusts from wherever
//! it is, at whatever speed it was already going.
//!
//! # Feature flags
//!
//! | Flag | Effect |
//! |---|---|
//! | `physics` | [`PhysicsBlend`] and [`RagdollBlend`], which need `threers-physics` |

pub mod animatable;
pub mod camera;
pub mod easing;
pub mod scene;
pub mod spring;
pub mod timeline;
pub mod tween;

#[cfg(feature = "physics")]
pub mod blend;

/// Everything you normally need, in one import.
pub mod prelude {
    pub use crate::animatable::Animatable;
    pub use crate::camera::{
        anamorphic_horizontal_fov, bake_eye_target, bake_poses, fov_for_sensor_fit, frame_bounds,
        screen_offset_for, triangular_shutter, CamTake, CameraAction, CameraAnimator, CameraClip,
        CameraCollision, CameraConstraint, CameraCurves, CameraFollow, CameraGuides, CameraMixer,
        CameraPath, CameraPose, CameraShake, CollisionBox, CollisionVolume, CompositionGuide,
        ConstraintStack, CurveKey, Driver, DriverChannel, DriverOp, DriverStack, Extrapolation,
        FCurve, FCurveModifier, FilmBack, FocusTracker, FollowTrackData, FramingBounds,
        FramingStyle, GateMask, GuideLine, KeyframeKind, Letterbox, MarkerCameraBind,
        ModifiedFCurve, MultiCam, NlaStrip, NlaTrack, OrbitRecorder, OrbitSample, Passepartout,
        PathKind, PathSpeedMode, SafeAreas, ScreenSpaceParam, ShakeLayer, ShakeStack, Shot,
        ShotSource, ShotTimeline, ShotTransition, ShutterCurve, SolvedCameraClip, SolvedCameraKey,
        SpeedRamp, StereoPair, StereoPivot, StereoRig, TangentMode, TiltedPath, TiltedPoint,
        TimeRemap, TimelineMarker, TrackAxis, TrackMarker2d, WalkFlyInput, WalkFlyNav,
    };
    pub use crate::easing::Easing;
    pub use crate::scene::{Animator, ClipHandle, ClipId, Property};
    pub use crate::spring::{Spring, SpringValue};
    pub use crate::timeline::{Timeline, TrackId};
    pub use crate::tween::{Repeat, Tween};
    pub use threers::math::{Color, Quaternion, Vector2, Vector3, Vector4};

    #[cfg(feature = "physics")]
    pub use crate::blend::{BlendState, PhysicsBlend, RagdollBlend};
}
