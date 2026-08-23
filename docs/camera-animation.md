# Cinematic camera animation

Scripted / cinematic work lives in [`threers-animation`](../crates/threers-animation) (Rust) and in the **additive** browser helpers `THREE.CameraAnimator`, `THREE.CameraPath`, and `THREE.ShotTimeline`. Classic **three.js camera APIs stay the entry point** — nothing requires migrating off `PerspectiveCamera`, `OrbitControls`, or `AnimationMixer`.

## three.js backwards compatibility

These keep working unchanged:

```js
const camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 100);
camera.position.set(2, 2, 4);
camera.lookAt(0, 0, 0);
camera.updateProjectionMatrix();

const controls = new THREE.OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0, 0, 0);
controls.minDistance = 1;
controls.update(); // every frame when damping / autoRotate

// Keyframe the camera like any Object3D
const mixer = new THREE.AnimationMixer(camera);
mixer.clipAction(clip).play();
mixer.update(dt);

// Stereo
const stereo = new THREE.StereoCamera();
stereo.eyeSep = 0.064;
stereo.update(camera);
```

## Browser cinematic helpers (additive)

```js
const anim = new THREE.CameraAnimator(camera);
anim.flyTo({
  position: new THREE.Vector3(5, 2, 5),
  target: new THREE.Vector3(0, 0, 0),
  duration: 2,
  ramp: 'inOut',   // ease-in → cruise → ease-out
  easeIn: 0.2,
  easeOut: 0.3,
});
anim.orbitBy({ azimuth: Math.PI / 2, duration: 1.5, ramp: 'smoother' });
anim.vertigo({ targetFov: 25, duration: 2 }); // dolly-zoom
anim.followPath({
  path: THREE.CameraPath.catmullRom(
    [[4, 2, 4], [0, 3, 5], [-3, 1, 2]],
    { fixedTarget: [0, 0, 0], ramp: 'smoother' },
  ),
  duration: 4,
});
// in the loop:
anim.update(dt);
```

Editorial cuts:

```js
const timeline = new THREE.ShotTimeline();
timeline.hold('wide', { position: [4, 2, 5], target: [0, 0, 0], duration: 2 });
timeline.fly('push', {
  from: { position: [4, 2, 5], target: [0, 0, 0], fov: 55 },
  to:   { position: [2, 1, 3], target: [0, 0.5, 0], fov: 40 },
  duration: 3, ramp: 'inOut',
  transition: { type: 'blend', duration: 0.5 },
});
timeline.update(dt);
timeline.apply(camera); // uses wasm setView when available
```

Live demo: [`web/examples/camera-shots.html`](../web/examples/camera-shots.html).

| three.js surface | Status |
|------------------|--------|
| `PerspectiveCamera(fov, aspect, near, far)` | Preserved (+ `fov` setter, `filmOffset`, `focus`, `up`, …) |
| `camera.lookAt` / `position` / `updateProjectionMatrix` | Preserved |
| `OrbitControls` + damping / autoRotate / `target` / distances | Preserved |
| `AnimationMixer` on camera `.position` / `.quaternion` / `.fov` | Preserved + camera-root binding |
| `StereoCamera.update(camera)` | Implemented |
| `CameraAnimator` fly / orbit / path / Vertigo / ramps | Additive helper |
| `CameraPath` + `ShotTimeline` | Additive helpers |
| `WebCamera.setView` (wasm) | Atomic eye+target+fov for cinematic frames |
| Full Blender-parity stack | Rust `threers-animation` (native / tooling) |

## Pro DCC features (Rust)

See the [feature map](#blender-parity-map) below for Track To, NLA, F-curve modifiers, markers, lens shift, panoramics, etc. Those APIs live on `threers_animation::prelude::*` and compose on top of `PerspectiveCamera` without changing the three.js-shaped surface.

## Blender-parity map

| Blender | threers |
|---------|---------|
| Track To / Damped / Locked Track | `CameraConstraint::{TrackTo,DampedTrack,LockedTrack}` |
| Child Of / Parent | `ChildOf`, `Parent` |
| Follow Path (tilt + radius) | `FollowPath` + `TiltedPath` |
| Clamp To / Floor / Limit Loc/Rot | `ClampTo`, `Floor`, `LimitLocation`, `LimitRotation` |
| Copy Transforms / Action | `CopyTransforms`, `Action` |
| Camera Solver / Follow Track | `CameraSolver`, `FollowTrack` (+ `SolvedCameraClip`) |
| F-curve modifiers | `FCurveModifier` / `ModifiedFCurve` |
| Keyframe kinds | `KeyframeKind` |
| Drivers | `DriverStack` |
| NLA strips | `NlaTrack` / `NlaStrip` |
| Bind Camera to Markers | `MarkerCameraBind` |
| Lens shift | `CameraPose::{shift_x,shift_y}` → `PerspectiveCamera` |
| Panoramic | `ProjectionKind::{Equirectangular,FisheyeEquidistant,MirrorBall}` |
| Sensor fit | `SensorFit` + `fov_for_sensor_fit` |
| DoF focus object / f-stop / blades | `FocusTracker` |
| Stereo pivot / spherical | `StereoPivot`, `StereoRig::spherical` |
| Composition guides / safe / passepartout | `CameraGuides`, … |
| Walk/Fly navigate → record | `WalkFlyNav` |
| Time remapping | `TimeRemap` |
| Rolling shutter | `ShutterCurve` |

Demo (Rust): `cargo run -p threers-animation --example camera_cinematic`
