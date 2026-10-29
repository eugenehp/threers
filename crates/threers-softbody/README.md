# threers-softbody

Soft bodies that tear and grow back: XPBD on tetrahedral lattices. The core of
[`threers::softbody`](https://docs.rs/threers/latest/threers/softbody/) — the
lattice, the solver, skinning — on its own, with no rendering or windowing
dependencies.

![A gummy jellyfish, octopus, gecko and bear](https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/gummy_jelly.jpg)

A creature is a soft **body** with **parts** glued on (tentacles, arms, a
tail), **chains** of particles rooted in it (gills, fringes), and anything else
bound to its lattice to ride along. Pull a part hard enough and it tears off,
keeps twitching, and a bud grows back from the stump.

```rust
use threers_softbody::{Bounds, Part};

// A lattice over any field (negative inside), with skin points bound to it.
let ball = |x: f64, y: f64, z: f64| ((x * x + (y - 0.3) * (y - 0.3) + z * z).sqrt() - 0.3);
let bounds = Bounds { min: [-0.4, -0.03, -0.4], max: [0.4, 0.7, 0.4] };
let part = Part::new(&ball, &bounds, 0.08, &[0.0, 0.6, 0.0]);
assert!(part.lattice.tet_count() > 0);
```

Building creatures from signed distance fields, binding them for the browser,
and drawing them is threers' side — see
[docs/softbody.md](https://github.com/eugenehp/threers/blob/main/docs/softbody.md)
and the demo, `web/examples/jelly.html`.

| | |
|---|---|
| `SoftBody` | the solver: `step`, grabbing, tearing and regrowth, muscles, chains, goo, picking, readouts |
| `Part`, `Lattice`, `embed_points` | a field's tetrahedral lattice and its rest state; points bound to it |
| `carry` | where bound points (a skin) are now, and their normals |
| `gpu::GpuSolver` (feature `gpu`) | the substeps as WebGPU compute, on wgpu; `suspend`/`resume` to switch solvers on the fly |
| `js_math` | V8's `Math.hypot`, fdlibm, `ToInt32` and a seedable LCG — what parity with JavaScript needs |

**The CPU solver is the JavaScript original's, bit for bit.** It is a port of
[Axolotl Jelly](https://github.com/saulfloresjr/axolotl-jelly)'s solver by
Dream Meridian (MIT; see `LICENSE`): values computed in f64 and stored in f32
wherever the original stores into a `Float32Array`, fdlibm for `Math.sin`,
`exp` and `pow`, V8's compensated `Math.hypot`, and insertion-ordered maps.
threers' `tests/softbody_parity.rs` holds it to a fixture the JavaScript
records — lattice, glue, every particle, chain and drop through a grab and a
tear.

**The GPU solver** (`gpu`) runs the substeps as compute: the lattice's cubes in
eight parity classes (cubes alike in parity share no node), glue a part node to
a thread with its pull on the body summed in fixed point — fifteen dispatches a
substep, two to three storage buffers. Same physics in another order: the same
shapes and the same tears as the CPU solver, not its bits.

In a wasm build optimised for size, build this crate for speed:

```toml
[profile.release.package.threers-softbody]
opt-level = 3
```

At `opt-level = "z"` a step took 50–80 ms in a browser; at 3, about 8.

MIT.
