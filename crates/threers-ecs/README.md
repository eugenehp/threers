# threers-ecs

Entities, components and systems over the [threers](https://crates.io/crates/threers)
scene graph. Every scene node is an entity, so the renderer and loaders work
unchanged and nothing has to be kept in sync.

```rust
use threers_ecs::prelude::*;

struct Velocity(Vector3);
impl Component for Velocity {}

fn movement(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    w.each::<(Node, &Velocity)>(|(node, velocity)| {
        node.position = node.position + velocity.0 * dt;
    });
}

let mut app = App::new();
app.fixed_system(movement);

let ball = app.world.spawn(Object3D::group());
app.world.insert(ball, Velocity(Vector3::new(1.0, 0.0, 0.0)));

app.frame(1.0 / 60.0);   // each frame, with the time that passed
// then draw app.world.scene with any threers renderer
```

```bash
cargo run -p threers-ecs --example bouncing
```

## What is in it

Always there:

- **Components** in packed tables keyed by `ObjectId`, so a query costs what its
  rarest component has rows.
- **Queries**: `World::each`, and `World::each_ref` when reading is enough. `Entity`,
  `Node`, `NodeRef`, `&C`, `&mut C`, `Option<..>`, `With`, `Without`, `Added`,
  `Changed`, and tuples of them.
- **Commands** for spawning and despawning from inside a query, and **resources**,
  with `World::with_resource` to use one alongside a query.
- **Events**: `World::send` and `World::events`. Each system keeps its own place, so
  the frame rate cannot lose or repeat one.
- **Systems** ordered with `.before(f)` and `.after(f)`, naming the function itself,
  and gated with `.run_if` and `.in_state`.
- **`App`**: a fixed-timestep loop that opens no window, with game states, pause,
  slow motion and single-stepping.
- **`Input`**: named actions and axes over keys, mouse and gamepads, with presses
  that reach fixed systems exactly once at any frame rate.
- **`registry`**: `reflect!` a component and it can be saved, loaded, stamped out as a
  prefab and shown in the inspector.
- **`assets`**: background loading, hot reload, and `spawn_scene` for `.glb` files,
  whose `extras` become components.
- **`animator`**: a state machine over clips, with crossfades.
- **`hud`**: a canvas over the game, and a retained UI with the `ui` feature.
- **`testing`**: `TestGame` plays a game with no window, and compares rendered frames.
- **`dev`**: F3 to F7 for frame time, colliders, pause, step and an inspector.

Features:

| Feature | Adds |
|---|---|
| `physics` | Rigid bodies, characters and collisions as entities |
| `audio` | A mixer, spatial speakers, WAV, Ogg and MP3 |
| `window` | A native window |
| `metal` | That window drawn by threers' native Metal renderer instead of wgpu (Apple platforms) |
| `gamepad` | Real pads through gilrs |
| `web` | A browser canvas, with DOM input, the Gamepad API, Web Audio and `fetch` |
| `ui` | The retained UI layer |

## Physics

With the `physics` feature, entities take rigid bodies from
[threers-physics](https://crates.io/crates/threers-physics):

```rust

app.plugin(PhysicsPlugin::default());
let ball = app.world.scene.add_mesh(geometry, material);
app.world.attach_body(ball, RigidBody::dynamic().shape(Shape::ball(0.5)));
```

The solver steps at the app's fixed rate and dynamic nodes are drawn between the
last two steps. Write a kinematic node's `position` and its body follows; move a
dynamic body through `Physics`, not its node. Removing the `Body` component in any
way removes the body. `Collision` events name entities, and any system can read them.

Nodes under moved or rotated parents work. A scaled ancestor is refused, since a
rigid body cannot scale.

```bash
cargo run -p threers-ecs --features physics --example falling_crate
```

## Characters

`attach_character` adds a walking, jumping `Character`. Set `walk` and call `jump`
from a fixed system ordered `.before(character_step)`.

## Input and windows

```rust
let input = app.world.resource_mut::<Input>();
input.bind("jump", Key::Space).bind("jump", PadButton::South);
input.bind_axis("move_x", Key::KeyA, Key::KeyD);
input.bind_analog("move_x", PadAxis::LeftX);

// In a system
let input = w.resource::<Input>();
if input.just_pressed("jump") { /* ... */ }
let direction = input.axis2("move_x", "move_y");
```

The `window` feature adds `window::run(app, WindowOptions::new("title"))`, which
feeds `Input`, calls `App::frame` and draws through the `MainCamera` resource.
The `gamepad` feature reads real pads through gilrs, in the window or from
`Gamepads::poll`. Any number of pads feed one `Input`. A test can feed it too,
which is how `tests/hop.rs` plays the `hop` example from start to finish:

```bash
cargo run -p threers-ecs --features physics,window,gamepad,audio --example hop
```

With the `metal` feature, `WindowOptions::new("title").metal()` (or
`THREERS_BACKEND=metal`) draws through threers' own Metal renderer. The race
takes it as a flag:

```bash
cargo run -p threers-ecs --release --features physics,metal,gamepad,audio --example race -- --metal
```

## In a browser

`web::run(app, canvas).await` runs the same `App` on a canvas, on WebGPU or WebGL2.
`crates/threers-hop` is `hop` as a browser package, and this plays it in a real
Chrome on both, with real key events and real screenshots:

```bash
python3 scripts/hop-browser-test.py
```

## In a Tauri app

[`examples/tauri-race`](examples/tauri-race/src/main.rs) is the race's browser
build inside a Tauri window: its own small crate, outside the workspace.

```bash
python3 scripts/web-game.py -p threers-hop --start start_race --title Race \
    --out crates/threers-ecs/examples/tauri-race/dist
cargo run --release --manifest-path crates/threers-ecs/examples/tauri-race/Cargo.toml
```

## Removing things

Remove entities with `World::despawn`. `ObjectArena::delete` works too, but the
side tables cannot see it, so its rows stay until `World::prune`.
