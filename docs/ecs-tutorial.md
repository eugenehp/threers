# Your first game

Part of the [threers](../README.md) documentation. The reference is [ECS](ecs.md).

A ball rolls round an arena collecting eight orbs before a minute is up. The finished game is [`examples/tutorial.rs`](../crates/threers-ecs/examples/tutorial.rs), about 170 lines, and a test plays it to a win by its keys, so what follows is code that runs.

```sh
cargo run -p threers-ecs --release --features physics,window,audio --example tutorial
```

Or start your own from the template, which is the smallest game: `templates/game`.

## 1. A world, and a function that builds it

Everything goes in a `build(app)` function, and `main` adds the window:

```rust
use threers_ecs::prelude::*;   // the whole vocabulary: one import

pub fn build(app: &mut App) { /* the game */ }

fn main() {
    let mut app = App::new();
    build(&mut app);
    app.plugin(AudioPlugin::Device).plugin(threers_ecs::dev::DevTools);
    run(app, WindowOptions::new("orbs"));
}
```

Keeping the window out of `build` is what lets a test, a recording or a rewind build the same game with no window.

## 2. Things are nodes, with components

Every scene node is an entity. Spawn one and set it up in one expression:

```rust
w.mesh(SphereGeometry::new(0.6, 24, 16), StandardMaterial::new(Color::from_hex(0xffcc33)))
    .at(Vector3::new(0.0, 1.0, 0.0))
    .body(RigidBody::dynamic().shape(Shape::ball(0.6)).mass(1.0))
    .with((Player, CameraTarget));
```

A component is any type with `impl Component for Player {}`. Forget that line and the compiler says so, and how to fix it. A mesh of your own comes from `MeshBuilder`: the arena is a `grid` with its rim raised, made solid with `.solid()`.

## 3. Input is named, and systems are functions

Bind names to keys, pads and touch once, and read the names:

```rust
input.bind_axis("x", Key::KeyA, Key::KeyD).bind_analog("x", PadAxis::LeftX);

fn roll(w: &mut World) {
    let push = w.resource::<Input>().axis2("x", "z");
    let dt = w.resource::<Time>().delta;
    let Some((id, _)) = w.single::<(Entity, With<Player>)>() else { return };
    if let Some(ball) = w.body_mut(id) {
        ball.apply_impulse(Vector3::new(push.x, 0.0, -push.y) * 15.0 * dt);
    }
}
```

A system can also take what it uses as parameters, `fn roll(input: Res<Input>, mut balls: Query<&mut Ball>)`, which says at a glance what it touches; see [ECS](ecs.md#systems-that-say-what-they-use).

Gameplay goes in `app.fixed_system(roll)`, which runs at a steady 60 a second whatever the frame rate; drawing-only work in `app.system(..)`. Order them with `.before(physics_step)` and `.after(..)`.

## 4. Physics tells you what touched

`PhysicsPlugin` sends a `Collision` event for every contact that starts or ends. Read them by reference, and check who touched what: the orbs sit on the floor, and that is a contact too.

```rust
w.read_events::<Collision>(|hit| { /* hit.a, hit.b, hit.started */ });
```

## 5. What the player sees

`ChaseCamera` as a plugin follows whatever has `CameraTarget`. Text and shapes go on the HUD each frame:

```rust
with_hud(w, |hud| {
    hud.text_size(30.0);
    hud.text(24.0, 44.0, &format!("{} / {ORBS}", round.collected));
});
```

`w.sun(direction, intensity, reach)` gives the level shadows that follow the camera.

## 6. States, timers and tuning

`app.state(Phase::Playing)` and `System::in_state(Phase::Playing)` stop the ball once the round is over. `Timer::once(60.0)` is the clock. `app.tune::<Tuning>("tuning/ball.json")` writes a file you can edit while the game runs: change the push, save, and feel the difference at once.

## 7. Sound

`tone(990.0, 0.12, 0.3)` makes a beep with no file, `waveform` a loop for an engine or a hum. `out.play(&sound)` plays it; a `Speaker` component plays from where its entity is.

## 8. Look inside while it runs

With `DevTools`: F3 shows each system's cost, any that failed and the selected entity's fields; click something to select it; F8 to F10 change a field; F2 flies a free camera; F4 draws colliders and, for a car, its tyres; F5 pauses and F6 steps; the backquote key opens a console (`help`); F12 records what you do. A system that panics is switched off and named, and the game goes on.

## 9. Test it by playing it

```rust
let mut game = TestGame::new(build);
game.hold(Key::KeyW, 60);
assert!(game.world().single::<(NodeRef, With<Player>)>().unwrap().0.position.z < 0.0);
```

`tests/devx2.rs` has the one that steers to every orb and wins. A recording made with F12 plays back exactly with `Recording::from_text(..).play(&mut app)`, which turns a bug someone found into a test.

## 10. Keep going while you change it, and ship it

`python3 scripts/watch-game.py -p threers-ecs --example tutorial --features physics,window,audio` rebuilds when you save and, with the `HotReload` plugin, starts the new build where the old one was.

For the browser, the game needs a `#[wasm_bindgen]` function taking a canvas (see `crates/threers-hop`), and then:

```sh
python3 scripts/web-game.py -p my-game --start start --title Orbs --serve 8080 --watch
```

It builds an optimised, compressed package and a page with a loading bar, a click to start (which is what lets sound play) and fullscreen, and reloads the open page when you save.
