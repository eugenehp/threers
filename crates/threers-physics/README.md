# threers-physics

Rigid-body physics for [threers](https://github.com/eugenehp/threers): gravity,
collisions, joints, scene queries and inverse kinematics.

Pure Rust, no dependencies beyond `threers` itself, and wasm-ready.

```toml
[dependencies]
threers = "0.0.4"
threers-physics = "0.0.1"
```

```rust
use threers_physics::prelude::*;

let mut world = World::new();                       // gravity is -9.81 Y
world.add_body(RigidBody::fixed().shape(Shape::ground()));

let ball = world.add_body(
    RigidBody::dynamic()
        .shape(Shape::ball(0.5))
        .translation(Vector3::new(0.0, 10.0, 0.0))
        .restitution(0.7),
);

world.step(1.0 / 60.0);                             // per frame
println!("{:?}", world.body(ball).unwrap().translation());
```

## What's in it

| Area | |
|---|---|
| **Shapes** | ball, cuboid, capsule, cylinder, cone, half-space, convex hull, triangle mesh, compound — each with analytic mass properties |
| **Collision** | sweep-and-prune broad phase; analytic + face-clipping + GJK/EPA narrow phase with persistent manifolds |
| **Continuous collision** | per-body opt-in (`RigidBody::ccd`): a step longer than the body is thick is swept instead of sampled, so a bullet cannot cross a wall between two frames |
| **Solver** | sequential impulses, warm starting, friction cones, split-impulse position correction for contacts *and* joints, sleeping |
| **Joints** | fixed, spherical, revolute, prismatic, distance, rope, spring — with limits, motors and breaking |
| **Machine elements** | position (servo) drives under a torque ceiling, joint state readout, gears, rack and pinion, lead screws — `mechanism` |
| **Assembly** | parts mated by geometry, an assembly-time pose solve, interference and range-of-motion checks, and motion driven through the solver — `assembly` |
| **SCAD mechanisms** | a `.scad` model declares its own parts and joints; each frame is a solver step, not a re-evaluation — `assembly` + `openscad` |
| **Queries** | `raycast`, `raycast_all`, `cast_shape`, `intersections_with_shape`, `project_point`, with filters |
| **IK** | FABRIK and cyclic-coordinate-descent chains with cone and hinge limits, readable from and writable to the scene graph. The CCD sweep detects its own stalls and escapes them, so a chain pointing straight at its target still folds to reach one closer in |
| **Characters** | kinematic `move_and_slide` controller with slope limits and ground snapping |
| **Vehicles** | raycast wheels with suspension, load transfer and a slip-based grip curve |
| **Tendons** | cables routed through guides and around obstacles, carrying tension along the path |
| **Decomposition** | approximate convex decomposition of a concave mesh into a `Shape::Compound` |
| **Terrain** | heightfield colliders, and `snapshot` for saving and restoring a whole world |
| **Diagnostics** | per-step counts, timings and warnings — including islands, and the NaN or infinity a caller wrote into a public field |

## Conventions worth knowing up front

**Sizes are half-extents**, because that is what the solver works in. Every such
constructor has a `*_from_size` sibling taking the full dimensions a `*Geometry`
would, so you never have to remember which one you are holding:

```rust
Shape::cuboid(0.5, 1.0, 0.5)          // half extents
Shape::cuboid_from_size(1.0, 2.0, 1.0) // same box, BoxGeometry(w, h, d) style
Shape::capsule_from_size(0.3, 1.2)     // CapsuleGeometry(radius, length)
```

**Capsules, cylinders and cones are Y-aligned**, matching three.js.

**Restitution combines with `Max`, friction with `Average`.** Friction really is
a property of the pair — rubber on ice is slippery. Bounciness is not: people set
it on the ball and leave the floor alone, and averaging would silently halve it.
Override per material with `with_restitution_combine`.

**Jointed bodies do not collide with each other** by default. A hinge lives
inside both the door and the frame; leaving that contact on makes the contact and
the constraint fight. Opt in with `.collide_connected(true)`.

## Driving a scene

Bind a body to a scene node and the transform follows:

```rust
let node = arena.insert(Object3D::mesh(mesh));
world.add_body(
    RigidBody::dynamic()
        .shape(Shape::ball(0.5))
        .scene_object(node),
);

// each frame
world.sync_from_scene(&arena);            // animation -> kinematic bodies
world.step(frame_dt);
world.sync_to_scene_interpolated(&mut arena);  // simulation -> meshes
```

For ragdoll / limp handovers use `threers-animation`'s `PhysicsBlend` and
`sync_to_scene_where` so the crossfade is not overwritten — see
[`docs/animation.md`](../../docs/animation.md).

`step` accumulates real frame time and consumes it in fixed internal steps, so
the simulation behaves identically at 30, 60 or 144 fps and a stalled frame
cannot destabilise it. `sync_to_scene_interpolated` blends between physics steps
using the leftover, which removes the stutter you would otherwise get whenever
the render rate does not match `world.timestep`.

Colliders can come straight from the geometry you are already drawing:

```rust
Shape::trimesh_from_geometry(&geometry)      // exact; for static level geometry
Shape::convex_hull_from_geometry(&geometry)  // for dynamic bodies
Shape::aabb_from_geometry(&geometry)         // cheapest
```

Triangle meshes are hollow surfaces, not solids — correct for static geometry,
but a dynamic body with one can be pushed out the wrong face. Use a convex hull
or a compound of primitives for anything that moves.

## Examples

```bash
cargo run -p threers-physics --example bouncing_balls        # gravity, restitution
cargo run -p threers-physics --example stacking              # friction, stacking, sleeping
cargo run -p threers-physics --example raycast_picking       # every scene query
cargo run -p threers-physics --example ragdoll_joints        # hinges, motors, ropes, breaking
cargo run -p threers-physics --example inverse_kinematics    # FABRIK and cyclic coordinate descent
cargo run -p threers-physics --example character_controller  # move-and-slide
cargo run -p threers-physics --example scene_sync            # threers scene graph binding
cargo run -p threers-physics --example vehicle               # suspension, weight transfer, grip
cargo run -p threers-physics --example zero_gravity_orbits   # orbital models
cargo run -p threers-physics --example robot_arm             # a jointed arm under servo drive
cargo run -p threers-physics --features gpu --example gpu_broadphase
cargo run -p threers-physics --features assembly --example hinged_assembly
                                                             # mates, checks, driven motion
cargo run -p threers-physics --features assembly,openscad --example scad_mechanism
                                                             # a .scad model that describes itself
cargo run -p threers-physics --features assembly,openscad --example mechanism_tour
                                                             # declare, drive, then check the geometry agrees

cargo run -p threers-robot-arm --example console             # a pick-and-place cell
cargo run -p threers-robot-arm --example performance         # motor and link sizing
```

In a browser, with the model editable and the joints plotted as they move:

```bash
crates/threers-mechanism-tour/build.sh
python3 -m http.server --directory web 8080   # then /mechanism-tour/
```

The whole stack builds for `wasm32` — the SCAD front end, the exact CSG kernel,
the solver, the checks and the verification — and one run of the bench above
takes about 16 ms of it.

## Feature flags

| Flag | Effect |
|---|---|
| `parallel` | Multi-threaded narrow phase via rayon. **Native only.** |
| `gpu` | Broad phase on a wgpu compute shader. Native **and** wasm32/WebGPU. |
| `async` | Pulls in a tokio runtime for callers driving the world from async code. Native only. |
| `openscad` | Colliders straight from OpenSCAD solids, with no STL round trip. |
| `mechanism` | Servo drives, joint state readout, and gear / rack / screw couplings. |
| `assembly` | Mates, the pose solve, interference and travel checks, driven motion. Implies `mechanism`. |

`parallel` keeps input order, so enabling it does not change the simulation.

### Going faster

`parallel` splits contact generation across threads — the largest single cost in
a busy scene. rayon needs OS threads, so on `wasm32` the feature is deliberately
a no-op rather than a build error.

`gpu` moves the broad phase onto a compute shader. That stage is the one that
parallelises cleanly: every AABB pair is independent, and the result is a short
list of indices. The narrow phase and the solver are not — sequential impulses
are sequential by construction, and shuttling bodies to the GPU per iteration
would cost more than it saves.

A dispatch plus readback costs roughly 0.2–1 ms whatever it does, and the CPU
sweep-and-prune handles a few thousand bodies well inside that. **Measure before
switching.** What it does unlock is parallel physics in the browser, where rayon
cannot run at all.

Share the renderer's device rather than creating a second one:

```rust
use threers_physics::gpu::GpuBroadPhase;

let mut gpu = GpuBroadPhase::from_device(renderer.device_arc(), renderer.queue_arc());

// each frame
let pairs = gpu.find_pairs(&world.broadphase_aabbs()).await;
world.set_broadphase_pairs(pairs);
world.step_fixed();
```

Pairs are sorted before being returned, so results do not depend on GPU
scheduling and the simulation stays deterministic.

## License

MIT
