# threers-continuum

Tendon-driven continuum robots for [`threers-physics`](../threers-physics).

A continuum robot has no joints. It is a flexible rod bent by cables run down
its length, and it goes wherever the balance between the two puts it. This
crate discretises the rod into rigid links joined by elastic stations whose
stiffness comes from beam theory rather than from tuning, routes cables through
them, and closes a task-space loop on the tip.

```rust
use threers_continuum::prelude::*;
use threers_physics::prelude::*;

let mut world = World::new();
world.substeps = 16;                                // a rod is a long chain —
world.solver_config.velocity_iterations = 64;       // see "Solver budget"

let rod = Rod::new(0.3, 10)
    .radius(0.002)
    .material(2.0e9, 0.35, 1200.0)
    .damped(0.05);

let mut arm = Continuum::build(&mut world, rod, None, Vector3::ZERO, Vector3::UP);
arm.add_tendon_ring(&mut world, 3, 0.0015, 0.0, 0, 0.0);

arm.set_pull(&mut world, 0, 0.0015, 20.0);          // reel cable 0 in by 1.5 mm
for _ in 0..900 {
    world.step_fixed();
}
assert!(arm.tip(&world).x > 0.01);                  // it curled toward the cable
```

## The model

```text
K_bend  = n·E·Iₓ / L      Iₓ = π/4 (rₒ⁴ − rᵢ⁴)
K_twist = n·G·I_z / L     I_z = 2Iₓ,  G = E / 2(1+ν)
```

The `n` in the numerator is the whole trick: a station standing in for a
shorter piece of the same beam is proportionally stiffer, so the rod bends the
same however finely it is chopped. `links` is a fidelity knob, not a physics
one.

Two radii, and they are usually not the same: stiffness comes from the
load-bearing core, mass from the rod as drawn. A 6 mm silicone finger over a
0.5 mm steel spine has almost all of one and almost none of the other, and
using the visible radius for stiffness overstates it by `(6/0.5)⁴`.

## Validation

Two levels. `tests/rod.rs` puts a horizontal cantilever under its own weight
and compares the settled tip droop to `wL⁴/8EI` — 2% at five links, 0.5% at
ten. That only covers small deflections of an unloaded rod, which is the easy
half.

`tests/sorosim.rs` compares against [SoRoSim](https://github.com/SoRoSim/SoRoSim)
Cosserat-rod solutions: 500 equilibrium shapes per material, each under its own
randomly oriented gravity plus a wrench at mid-span and another at the tip, bent
through tens of degrees. Mean shape error as a percentage of rod length, at 32
substeps of 128 iterations:

| links | TPU, 5 mm, `E` = 70 MPa | spring steel, 0.8 mm, `E` = 200 GPa |
|---|---|---|
| 5 | 1.56% | **1.48%** |
| 10 | 0.63% | 2.31% |
| 20 | 0.32% | 6.95% |
| 30 | **0.23%** | 12.23% |

The data is not vendored — see `src/sorosim.rs` for where to get it. Every test
skips when it is absent.

Two findings worth carrying away. **Torsion is the largest single term**: with
twist locked, the same comparison reads 5.7% for TPU and 42% for steel. And
**the discretisation converges but the solver is what limits the stiff case** —
the soft rod halves its error at every doubling of `links`, while steel is best
at *five* links and worse at thirty.

## Solver budget

**Raise it, and by how much depends on the rod.** `World`'s defaults are tuned
for scenes of loose bodies; on a chain they diverge past a handful of links.
Sixteen substeps of sixty-four iterations hold ten links of a soft rod; a stiff
thin one needs thirty-two of a hundred and twenty-eight and still degrades past
five links. A plain row of `Joint::fixed` welds diverges at the same lengths,
which is how you can tell it is not the springs.

## Layout

| Module | Role |
|---|---|
| `rod` | `Rod` — what the rod is, and the beam theory |
| `build` | `Continuum` — the rod as bodies, stations and cables |
| `clark` | `Clark` — three cables as one two-dimensional bend |
| `kinematics` | The constant-curvature model, and closed-loop tip control |
| `scad` | Building one from a `.scad` model's `continuum()` declaration |

## From a `.scad` model

With the `openscad` feature, a model declares its own rod:

```scad
part("base", fixed = true) cylinder(h = 10, r = 12);

continuum("backbone", on = "base", at = [0, 0, 10], axis = [0, 0, 1],
          length = 300, links = 10, radius = 6, segments = 1,
          youngs = 200e9, density = 1200, backbone_radius = 0.5);

tendon("t0", along = "backbone", offset = 4, phase = 0,   pretension = 1);
tendon("t1", along = "backbone", offset = 4, phase = 120, pretension = 1);
tendon("t2", along = "backbone", offset = 4, phase = 240, pretension = 1);
```

and `build_scad_continua` turns it into bodies, stations and cables.

## License

MIT.
