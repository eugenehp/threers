# Animation

threers keeps the **three.js animation API** as the browser entry point
(`AnimationClip` / `AnimationMixer` / `AnimationAction` / `KeyframeTrack`) and
adds cinematic cameras plus general motion helpers on top — without requiring
apps to migrate off the classic surface.

## three.js surface (shim)

```js
const mixer = new THREE.AnimationMixer(scene);
const a = mixer.clipAction(idleClip).play();
const b = mixer.clipAction(runClip);
b.crossFadeFrom(a, 0.4);          // real fade, not a stub
b.setEffectiveWeight(1);
runClip.blendMode = THREE.AdditiveAnimationBlendMode;

// PropertyBinding resolves node.property, material.color, morphTargetInfluences[i]
mixer.update(dt);
```

Tracks may carry procedural modifiers:

```js
track.modifiers = [
  THREE.TrackModifier.noise({ amplitude: 0.02, frequency: 3 }),
  THREE.TrackModifier.cycles(),
];
```

## Motion primitives (additive)

| Need | Use |
|------|-----|
| A→B over a duration | `THREE.Tween` |
| Chase a moving target | `THREE.Spring` / `THREE.ObjectSpring` |
| Shared clock + markers | `THREE.Timeline` |
| Animated ↔ limp pose | `THREE.PoseBlend` |
| Cinematic camera | `THREE.CameraAnimator` / `ShotTimeline` |

```js
const tween = new THREE.Tween(from, to, 1.2).setEasing('cubicOut');
const lag = new THREE.ObjectSpring(follower, { frequency: 6 });
lag.setTarget(leader).update(dt);
```

Demo: [`web/examples/animation-demo.html`](../web/examples/animation-demo.html) ·
camera shots: [`web/examples/camera-shots.html`](../web/examples/camera-shots.html).

## Rust

- Core mixer: `threers::animation` — weighted apply, fades, `cross_fade`,
  `TrackTarget::MorphWeight`, glTF `weights` channels.
- Companion crate: [`threers-animation`](../crates/threers-animation) — Tween,
  Spring, Timeline (markers + time remap), cinematic camera stack, optional
  `PhysicsBlend` (`--features physics`).

### Animation ↔ physics

`PhysicsBlend` / `RagdollBlend` crossfade kinematic animation into the solver
(and back) without a visible snap. Important rules:

1. Freeze the animated pose at limp time (a continuing clip must not yank the fade).
2. Inherit stride velocity when the kinematic body had none yet.
3. During a handover, **do not** call bare `world.sync_to_scene` — it overwrites
   the blended draw pose. Use `blend.apply_to_scene` + `sync_to_scene_where`.

```rust
let drawn = blend.update(dt, animated, &mut world);
blend.apply_to_scene(drawn, &world, &mut arena);
world.step(dt);
world.sync_to_scene_where(&mut arena, |id, _| !(id == body && blend.owns_scene_write()));
```

Demo: `cargo run -p threers-animation --example anim_physics_blend --features physics`

See also [`docs/camera-animation.md`](camera-animation.md).
