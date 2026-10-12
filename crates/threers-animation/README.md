# threers-animation

Tweens, springs, timelines and cinematic camera animation for
[threers](https://crates.io/crates/threers).

```toml
[dependencies]
threers-animation = "0.0.1"
```

```rust
use threers_animation::prelude::*;

let mut fade = Tween::new(0.0f32, 1.0, 0.5).easing(Easing::CubicOut);
fade.update(0.25);
assert!(fade.value() > 0.5);
```

## Which tool for which job

| Need | Use |
|---|---|
| A→B over a known duration | `Tween` |
| Chase a target that keeps moving | `Spring` |
| Several things on one clock | `Timeline` |
| Drive scene nodes directly | `Animator` |
| Cinematic camera moves, paths, shots | `camera` |
| Hand a character to the physics solver | `blend::PhysicsBlend` (`physics` feature) |

The distinction that matters most is the first two. A tween is told how long to
take, which is right for scripted motion and wrong for anything reacting to
input — retarget a tween mid-flight and it either restarts or jumps. A spring
has no duration at all: move its target and it adjusts from wherever it is, at
whatever speed it was already going.

## The camera

The largest part of the crate, and the reason it is a crate rather than a
module: a Blender-parity feature set for moving a camera on purpose rather than
by hand. Fly and orbit rigs, paths with speed control, follow and look-at
constraints, handheld shake as a layer, framing helpers, an NLA-style mixer, and
a shot list you can cut between.

## Features

| Flag | Effect |
|---|---|
| `physics` | `blend::PhysicsBlend` and `blend::RagdollBlend`, which need `threers-physics` |

Easing curves themselves live in `threers` — a curve from 0 to 1 is not
animation, and keeping it here meant `cargo add threers` got you none.
`threers_animation::easing` re-exports it, so either path works.

## Licence

MIT.
