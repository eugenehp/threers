//! Cinematic camera animation — Blender-parity feature set.
//!
//! Interactive orbit damping / auto-rotate live on
//! [`threers::controls::OrbitControls`](threers::controls::OrbitControls).

mod animator;
mod bake;
mod collision;
mod constraint;
mod curve;
mod curve_follow;
mod driver;
mod fcurve_mod;
mod focus;
mod follow;
mod framing;
mod guides;
mod markers;
mod mixer;
mod motion_blur;
mod multicam;
mod nla;
mod path;
mod physical;
mod pose;
mod ramp;
mod shake;
mod shake_layer;
mod shot;
mod shutter;
mod stereo;
mod time_remap;
mod tracking;
mod walk_fly;

pub use animator::CameraAnimator;
pub use bake::{bake_eye_target, bake_poses, OrbitRecorder, OrbitSample};
pub use collision::{CameraCollision, CollisionBox, CollisionVolume};
pub use constraint::{CameraConstraint, ConstraintStack, TrackAxis};
pub use curve::{CameraCurves, CurveKey, Extrapolation, FCurve, TangentMode};
pub use curve_follow::{TiltedPath, TiltedPoint};
pub use driver::{Driver, DriverChannel, DriverOp, DriverStack};
pub use fcurve_mod::{FCurveModifier, KeyframeKind, ModifiedFCurve};
pub use focus::FocusTracker;
pub use follow::CameraFollow;
pub use framing::{frame_bounds, screen_offset_for, FramingBounds, FramingStyle};
pub use guides::{CameraGuides, CompositionGuide, GuideLine, Passepartout, SafeAreas};
pub use markers::{MarkerCameraBind, TimelineMarker};
pub use mixer::{CameraAction, CameraClip, CameraMixer};
pub use motion_blur::{PathSpeedMode, ScreenSpaceParam};
pub use multicam::{CamTake, GateMask, Letterbox, MultiCam};
pub use nla::{NlaStrip, NlaTrack};
pub use path::{CameraPath, PathKind};
pub use physical::{anamorphic_horizontal_fov, fov_for_sensor_fit, FilmBack};
pub use pose::CameraPose;
pub use ramp::SpeedRamp;
pub use shake::CameraShake;
pub use shake_layer::{ShakeLayer, ShakeStack};
pub use shot::{Shot, ShotSource, ShotTimeline, ShotTransition};
pub use shutter::{triangular_shutter, ShutterCurve};
pub use stereo::{StereoPair, StereoPivot, StereoRig};
pub use time_remap::TimeRemap;
pub use tracking::{FollowTrackData, SolvedCameraClip, SolvedCameraKey, TrackMarker2d};
pub use walk_fly::{WalkFlyInput, WalkFlyNav};
