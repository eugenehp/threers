# ECS

Part of the [threers](../README.md) documentation.

[`threers-ecs`](../crates/threers-ecs) is a game layer over the scene graph: every scene node is an entity, so the renderer and loaders work unchanged. The [crate README](../crates/threers-ecs/README.md) has the tour, and the table at the top of its crate docs says which tool fits which job. This page covers the rules that are easy to get wrong. New to it? Start with [Your first game](ecs-tutorial.md).

## Systems that say what they use

A system can take the whole `World`, or the things it uses as parameters, and the two mix in one schedule:

```rust
fn movement(mut moving: Query<(Node, &Velocity)>, time: Res<Time>) {
    moving.each(|(node, velocity)| node.position = node.position + velocity.0 * time.delta);
}

fn scoring(mut score: ResMut<Score>, mut out: EventWriter<Scored>, healthy: Query<&Health>) {
    score.0 += healthy.iter().count() as u32;
    out.send(Scored(score.0));
}

app.fixed_system(movement).fixed_system(System::new(scoring).after(movement));
```

`Query`, `Res`, `ResMut`, `EventReader`, `EventWriter` and `Commands`, up to eight. Two parameters that would alias, such as `Query<&mut Health>` beside `Query<&Health>`, are refused when the system is added, by name. Systems run one at a time unless asked otherwise.

### Parallel systems

`System::parallel(f)` lets a parameter system run on the thread pool at the same time as others:

```rust
app.fixed_system(System::parallel(movement))
    .fixed_system(System::parallel(spinning))
    .fixed_system(System::parallel(animate));
```

The schedule runs a set of them together when they are next to each other in its order, none is ordered after another (`after`, `before`, directly or through others), and no two touch the same thing where one changes it: two `Query<&mut Spin>`s, or `ResMut<Score>` beside `Res<Score>`, take turns. `Node` counts as one thing for the whole scene, so two systems moving nodes take turns too. A plain `&mut World` system between them splits the set. Their run conditions are checked before any starts.

Parameters are fetched on the main thread and only the function body runs elsewhere, so the world is never touched by two threads at once outside what the parameters lend. Every parameter has to be able to cross threads: `Commands` cannot (it holds an `Rc`), nor can a component or resource that is not `Send`/`Sync`, and the compiler says which. In a browser with `web-threads`, `Node` and `NodeRef` cannot either: a node holds WebGPU handles, which stay on the thread that made them. Since none uses what another changes, the world afterwards is the one running them in turn would give, and replays stay bit-exact.

Without the `parallel` feature, and in a browser without `web-threads`, parallel systems simply run in order. A panic inside one is caught like any other system's when `catch_panics` is on, and `Profile` times each on its own thread. Worth it when systems are heavy: each batch costs a pool hand-off of tens of microseconds.

## What a query can name

```rust
w.each::<(Entity, &mut Health, With<Enemy>)>(|(id, health, _)| health.0 -= 1.0);
w.each::<(&Velocity, Changed<Velocity>)>(|(velocity, _)| { /* … */ });
```

A query panics if it borrows one value twice: `(&A, &mut A)`, or `(Node, NodeRef)`, which are two views of one scene node. Filters (`With`, `Without`, `Added`, `Changed`) read only a row's bookkeeping, so they can name a component the query also borrows.

A query asks every term whether a row matches before it fetches from any of them. Fetching `&mut C` marks the row changed, so the order matters: in `(&mut A, Without<B>)`, only rows actually handed to the closure count as changed.

## Systems and identity

A system is known by its Rust type, which is how `.after(read_input)` keeps compiling or stops compiling with the function it names.

| Added twice | Result |
|---|---|
| The same function | Panics: it would run the same code twice |
| A capturing closure from one function | Allowed: `for i in 0..3 { app.system(make(i)); }` |
| `fn` pointers | Allowed: all share one type but need not be the same function |

Each repeat has its own place in event queues and its own `Added`/`Changed` window. Ordering against a repeated type waits for all of them. One function in both the fixed and the frame schedule counts as one system and shares a place.

## Development tools

`DevTools` as a plugin, left out of a release build:

| Key | Does |
|---|---|
| F1 | rewind five seconds, with the `Rewind` plugin |
| F2 | fly a free camera, the game paused |
| F3 | the panel: frame time, each system's cost, any that failed, the selected entity's fields |
| click | select what is under the pointer |
| F4 | colliders, and each car's tyre grip, suspension and speed |
| F5, F6 | pause, step |
| F7, F8, F9, F10 | select the next entity, its next field, lower or raise it |
| F12 | record your input to `recordings/` |
| backquote | the console: `help`, `select`, `set`, `tp`, `speed`, `systems`, and a game's own (`World::console_command`) |

The console takes the keys while open (`Input::captured`), and the Escape or backquote that closes it is consumed, so it does not also reach the game's pause menu.

It also turns on `App::catch_panics`: a system that panics is switched off and named on the panel, and the game goes on. A failing picture test (`TestGame::assert_frame_matches`) writes `<name>.actual.png` and `<name>.diff.png` beside the reference.

`scripts/watch-game.py` rebuilds and restarts the game on every save, and with `HotReload` it starts where the old build was. `scripts/web-game.py` builds the browser package and page, serves it, and with `--watch` reloads the open page.

### Build times

The template builds dependencies optimised in a debug build (`[profile.dev.package."*"] opt-level = 3`), so the game runs at speed while its own code compiles quickly.

Measured on an Apple Silicon Mac under heavy load (load average ~150–190), with the compile cache off, building the `race` example with `physics,audio,window,gamepad`:

| Build | Wall | CPU (user + sys) |
|---|---|---|
| First build, debug | 150 s | 174 s |
| After editing the game's code | 2.7–3.7 s | 2.0–2.3 s |
| After editing `threers-ecs` | 2.5–5.8 s | 2.5–2.8 s |

Editing and rebuilding is already quick. `debug = "line-tables-only"` made no measurable difference to it (the two ranges overlap). The first build is where the time goes, and most of it is `threers` itself (75 s, on the critical path), then naga and wgpu-core (about 15 s each), and the sound backend's generated bindings (`coreaudio-sys` and `bindgen`, about 17 s, only with `audio`). Leave out features you don't use, and share one compile cache or `CARGO_TARGET_DIR` between games so the engine is built once rather than per checkout.

## Building a game

The [`race`](../crates/threers-ecs/examples/race/main.rs) example is the worked one, a module per concern:

| Module | Has |
|---|---|
| `track` | the circuit's centre line, hills and banked corners, and the scenery: ground, road, kerbs, barriers, trees, built with `MeshBuilder` and made solid |
| `rules` | laps, lap and best-lap times (`LapDone` events), places, the wrong-way check, recovery, starting again |
| `ai` | rivals that follow the road at the speed its bends allow, and go round a car in their lane or follow it when there is no room |
| `cars` | a body built from blocks, engine and tyre-squeal speakers |
| `effects` | skid marks, one mesh rebuilt as they change |
| `ghost` | the player's best lap, sampled every step and driven again by a see-through car |
| `hud` | times, the start lights, a map of the circuit, the wrong-way warning |
| `menu` | Escape pauses; resume, start again, music and effects volume, quit |

`tests/race.rs` plays it: a race by the keys, rivals passing a stopped car without touching it, a lap timed and kept as the ghost, the menu, skid marks and squeal, and a recorded race played back exactly.

```rust
let w = &mut app.world;
w.sun(Vector3::new(-1.0, -2.0, -1.5), 3.0, 150.0);            // shadows that follow the camera
let car = w
    .mesh(BoxGeometry::new(1.8, 0.7, 4.0), StandardMaterial::new(Color::WHITE))
    .at(Vector3::new(0.0, 1.0, 0.0))
    .named("player")
    .car(CarSpec::default())                                  // a body, four drawn wheels
    .with((Player, CameraTarget))
    .id();
```

One import covers a game:

```rust
use threers_ecs::prelude::*;
```

It carries `threers`'s prelude (maths, geometry, materials, lights, `MeshBuilder`), the camera, HUD, touch controls and `TestGame`, and with their features physics and cars, sound, and the window's `run`. The physics solver's own `World` is `Solver`, so `World` is always the game's. Rarer things stay in their modules: `assets`, `saves`, `tasks`, `dev`, `gamepad`. Code that still globs `threers::prelude`, `threers_ecs::physics` and this together compiles as before.

| Need | Use |
|---|---|
| Spawn and set up in one go | `World::mesh`, `World::object` |
| A mesh of your own | `MeshBuilder`: `ribbon`, `grid`, `quad`, `append`; `MeshReport` says what is wrong with one |
| Many objects as one draw | `MeshBuilder::append` each into one mesh, `Shape::compound` for its body |
| Level geometry that is solid | `Spawned::solid`: a fixed body shaped like the mesh |
| Change a mesh while running | `World::edit_mesh` |
| Read the world | `iter`, `read(id)`, `single`, `entities`, `count`: all `&self` |
| Change one entity | `update(id, ...)` |
| Where something really is | `world_transform`, through its parents |
| The one player, or boss | `World::single`, read-only |
| Shadows at the size of a level | `World::sun` |
| A car | `Spawned::car`; set `Car::throttle`, `brake` and `steer` before `car_step` |
| A camera behind something | `ChaseCamera` as a plugin, `CameraTarget` on the thing or `.follow(id)`; `avoid_walls`, `looking_back`, `shake` |
| An engine, a hum | `waveform` and `Speaker::looping`; set `Speaker::pitch` each frame |
| Music | `AudioOut::play_music` crossfades; `Bus` volumes for a settings screen |
| Put something back | `teleport_body`: moving only the node leaves the solver to pull it back |
| Run in every state but one | `System::not_in_state` |
| Order against a system that may be absent | `System::after_if_added` |

A test can play the game by its keys and look at it: `tests/race.rs` drives a whole race through `Input`, and `TestGame::render` returns the frame with the `Hud` on it. A test that only reads numbers will not see a kerb facing the wrong way.

## Starting, tuning, looking, testing

| Need | Use |
|---|---|
| A new game | `templates/game`, which is `examples/minimal.rs` with a `Cargo.toml` |
| Numbers to tune while it runs | `app.tune::<T>("tuning/t.json")`: edit and save, and `T` changes; mistakes are reported, the rest applies |
| See what code is thinking | `w.resource_mut::<Gizmos>()`: `line`, `arrow`, `ray`, `point`, `cube`, `sphere`, `circle`, `path` |
| Change a component in the running game | `DevTools`: F7 picks, F8 moves to a field, F9 and F10 change it |
| Countdowns and cooldowns | `Timer::once`, `Timer::repeating` |
| Do something later | `w.after(seconds, \|w\| ...)` |
| Glide something somewhere | `Tween::to_position`, `to_scale`, `to_rotation`, then `TweenFinished` |
| Turn a bug someone found into a test | F12 with `DevTools` records to `recordings/`; `Recording::from_text(..).play(&mut app)` plays it back exactly |
| A level from Blender | `spawn_scene("level.glb")`; custom properties become components, and `solid`, `sensor` and `body` give collision |
| A loading screen | `AssetsPlugin::new("assets")` and `LoadingScreen { during, then }`: the state moves on when nothing is loading |
| Save slots | `reflect!` what to keep, then `SaveSlots::new("game")` with `write`, `read`, `list`: files natively, `localStorage` in a browser |
| Stamp out a thing many times | `World::capture_prefab(root)`, then `spawn_prefab`: the entity, its children and their reflected components |
| Animation states | `Animator`: `state`, `once`, `transition` on `Condition::above`/`below`/`trigger`, crossfades |
| A walking character | `attach_character` and `Character::walk`/`jump`; `character_step` moves it |
| Thumbs on a phone | `TouchControls::new().stick("turn", "forward").button(..)`, read through the same actions |
| Work longer than a frame | `Tasks::spawn` natively, `Jobs` for the same job on threads or Web Workers |
| Hundreds of still meshes | mark them `Static` and call `World::bake_static`: one draw per material |
| A screen showing a video | feature `video`: `w.mesh(..).video(VideoPlayer::open(bytes)?)`. It plays on game time, so a paused game pauses it. With `audio` and an `AudioPlugin` its soundtrack plays in step, through seeks, loops and pauses; insert `Video::muted(player)` for none |
| An overlay that takes a key | `Input::consume(key)`: nothing later this frame sees it pressed, as the console does with Escape |
| Know which systems broke | `Failures` and `SystemFailed` events, with `App::catch_panics(true)` |

Gizmos drawn by a fixed system stay up until the next fixed step, since a frame may run none. Timers, delays and tweens run on game time, so a pause stops them.

Mistakes say what to do. A missing resource names the plugin that adds it and lists what the world has; one taken by `with_resource` says so; a misspelt action lists the bound ones; and each points at the line that asked, not into the crate. At compile time, a type used as a component without `impl Component`, a query term that is not one, and `&mut` in a read-only query each get a message with the fix.

## Where a game runs

| Host | How |
|---|---|
| A native window, wgpu | `window::run(app, WindowOptions::new("title"))` with the `window` feature |
| A native window, Metal | the `metal` feature and `WindowOptions::new("title").metal()`, or `THREERS_BACKEND=metal`; the race takes `--metal` |
| A browser | `scripts/web-game.py`, with a `cdylib` whose `#[wasm_bindgen]` start function takes a canvas, as `threers-hop`'s `start_race` does |
| A Tauri app | the browser build inside Tauri's webview: [`examples/tauri-race`](../crates/threers-ecs/examples/tauri-race/src/main.rs) |

The Metal window draws the same picture as wgpu: `the_race_on_metal_looks_like_the_race_on_wgpu` in `tests/race.rs` renders the race both ways and compares them region by region.

## Determinism

A game played again from the same start with the same input, frame by frame, comes out the same to the last bit: same positions, same contacts, same lap times. `tests/race.rs` holds it to that (`a_recorded_race_plays_back_exactly`, and `rewinding_puts_the_race_back_where_it_was`, which compares a rewound car's position bit for bit), and CI runs both on every change. [`Recording`](../crates/threers-ecs/src/replay.rs), `Rewind` and replay tests are built on it.

What it holds across:

- the same build of the game, on the same platform and CPU architecture;
- the same frame lengths, which a `Recording` keeps, so the fixed steps fall the same;
- any order a `HashMap` iterates in: nothing the simulation does depends on one.

What breaks it:

- a different CPU architecture or compiler: floating point can round differently (fused multiply-adds, for one), so a replay from an ARM Mac may part from one on x86;
- reading the real clock, a random seed from the system, or the network inside a fixed system;
- `par_query` writing to anything but its own rows: each row is its own, but the order rows run in is not fixed.

Networked lockstep or cross-platform replays need more than this: fixed-point maths or strict floating point throughout, which this does not promise.

## Custom meshes

`MeshBuilder` turns vertices and triangles into a `BufferGeometry` without attribute arrays. The mistake it is built around is winding: a triangle drawn the wrong way round is invisible from the side you look at, with no error. `facing(side)`, `ribbon` and `grid` take the side you mean, and `MeshReport::of(&geometry)` counts flipped, open and degenerate edges when something does not show.

Merging is where it pays in a game. The race's scenery went from 529 meshes to 31 by appending each barrier box and tree into one mesh, with the barriers' collision as one `Shape::compound`. On an M4 Pro the frame went from about 3.0 ms to 2.4–2.8 ms at the median; the gain is larger where draws cost more, as in a browser.

## Threads

`World::par_each` runs over one component's packed rows. `World::par_query` takes several terms, as long as they are `Parallel`: components and filters, not `Node`, because the arena is one value rather than a row each. Both run on one thread without the `parallel` feature, for short lists, and on a browser's main thread.

Threads pay only for real work per row. On 100,000 rows doing one addition, `par_query` measured about nine times slower than `each` (3.3 ms against 0.38 ms, on a loaded machine): dispatch costs more than the loop. Use it for pathfinding or a heavy update, and measure with `cargo bench`.

## Every module

| Module | What is in it |
|---|---|
| `world`, `query` (root) | `World`, `Entity`, queries and their terms, resources, events, commands |
| `params` | `Query`, `Res`, `ResMut`, `EventReader`, `EventWriter`, `Commands`; `IntoSystem`, `IntoParallel` |
| `schedule` (root) | `System` (`before`, `after`, `run_if`, `in_state`, `parallel`), `Schedule` |
| `app` (root) | `App`, `Plugin`, `Time`, `Exit`, `MainCamera`, game states |
| `input` | `Input`, `Key`, `MouseButton`, `PadButton`, `PadAxis`, `Button`, `Rumble` |
| `touch` | `TouchControls`, `TouchButton` |
| `gamepad` | `Gamepads` (feature `gamepad`) |
| `physics` | `Body`, `Collision`, `Character`, `AttachBody` (feature `physics`) |
| `car` | `Car`, `CarSpec`, `Driven`, `AttachCar` (feature `physics`) |
| `camera` | `ChaseCamera`, `CameraTarget` |
| `audio` | `AudioOut`, `Speaker`, `Speakers`, `Bus`, `AudioPlugin` (feature `audio`) |
| `animator` | `Animator`, `Condition`, `AnimationFinished` |
| `assets` | `AssetsPlugin`, `Handle`, `LoadingScreen`, `SceneInstance`, `AssetLoaded`, `AssetFailed` |
| `registry` | `reflect!`, `Reflect`, `Field`, `FieldValue`, `Prefab` |
| `saves` | `SaveSlots` |
| `batch` | `Static`, `Baked` |
| `timer` | `Timer`, `Tween`, `TweenFinished` |
| `video` | `Video`, `VideoPlayer`, `VideoTexture` (feature `video`) |
| `tasks` | `Tasks`, `Jobs` |
| `hud` | `Hud`, `UiLayer` (feature `ui`) |
| `gizmos` | `Gizmos` |
| `tuning` | `App::tune` |
| `profile` | `Profile`, `SystemTime`, `Failures`, `SystemFailed` |
| `replay` | `Recording`, `InputEvent`, `Rewind`, `RewindState` |
| `dev` | `DevTools`, `DevState`, `Flight` |
| `console` | `Console`, `ConsoleState`, `World::console_command` |
| `hot` | `HotReload`, `HotState` (native) |
| `testing` | `TestGame` |
| `window` | `run`, `WindowOptions` (feature `window`; `metal` for the Metal backend) |
| `web` | `run` on a canvas (feature `web`, wasm32) |

Every public item has rustdoc, and `#![warn(missing_docs)]` keeps it so: `cargo doc -p threers-ecs --all-features --open`.

## Testing it

The `unsafe` in `query.rs` is checked by `tests/query_rules.rs`, under Miri as well as normally:

```sh
MIRIFLAGS=-Zmiri-disable-isolation env -u CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER \
  cargo +nightly miri test -p threers-ecs --test query_rules
```

`tests/params.rs` and `tests/parallel_systems.rs` run under Miri the same way. With `--features parallel` the second needs `-Zmiri-tree-borrows` as well: rayon's queue (crossbeam-epoch) fails the stricter default model on its own, inside the crate rather than here.

Most targets need a feature, so a plain `cargo test -p threers-ecs` skips them without saying so. To run them all:

```sh
cargo test -p threers-ecs --release --features physics,window,gamepad,audio,parallel,ui
```

`cargo bench -p threers-ecs` times the query paths.
