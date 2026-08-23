# threers-mechanism-tour

The [threers-physics](../threers-physics) assembly stack, in a browser, on six
mechanisms.

A `.scad` model declares its own parts, joints and drives. This reads those
declarations, puts the assembly together, checks it, sweeps its travel, runs it
through the solver, records the run as poses, reduces it to keyframes, and then
asks the geometry whether the declarations were true — the same seven acts
`threers-physics/examples/mechanism_tour.rs` prints to a terminal, with the
model editable and the run on screen.

```bash
crates/threers-mechanism-tour/build.sh
python3 -m http.server --directory web 8080
open http://localhost:8080/mechanism-tour/
```

There is a terminal version too, so the page and the console can be compared
line for line:

```bash
cargo run -p threers-mechanism-tour --example console              # the bench
cargo run -p threers-mechanism-tour --example console -- latch     # one of them
cargo run -p threers-mechanism-tour --example console -- --all     # all of them
cargo run -p threers-mechanism-tour --example console -- --trace ratchet
cargo run -p threers-mechanism-tour --example console -- my.scad   # your own
```

and four things that ask the models questions the readings cannot answer:

```bash
cargo run --release -p threers-mechanism-tour --example probe      # is every joint built?
cargo run --release -p threers-mechanism-tour --example audit      # does anything pass through anything?
cargo run --release -p threers-mechanism-tour --example buildtime  # what does the geometry cost?
cargo run --release -p threers-mechanism-tour --example look       # what does it look like?
```

## The mechanisms

| | |
|---|---|
| **bench** | one of everything: hinge, gear, slider, thread, and a lid asked for more travel than it has |
| **gear train** | three shafts, two meshes, 7.5:1 — a compound reduction, where the ratio is a consequence rather than a number anyone typed |
| **four-bar** | a closed loop. The crank goes round, the rocker cannot, and the geometry decides where it turns back |
| **cam** | nothing declares how the follower moves. It rests on the cam, and the contact *is* the mechanism |
| **ratchet** | free one way and locked the other, out of a tooth shape and a spring. Driven back with the same force and the same distance asked for, it does not move |
| **latch** | rack and pinion, a spring latch that catches, and a drive that then **fails** to open it until a plunger releases it |

The last two are the argument. Nothing in either model says "locked". The pawl
drops behind the catch because it is heavy and sprung, the catch runs into it
because it is in the way, and a drive with a force ceiling fails to reach the
number it was given. A drive that could always reach its target would teleport
the gate through the pawl, and the animation would look fine and be a lie.

## A mate is not a bearing

The models used to declare joints they had not built. A `hinge()` is a sentence
about two parts: it puts no pin in a bore, no rail under a carriage and no teeth
on a gear, and it draws none of them. The solver holds the two parts in a perfect
hinge relationship regardless — and then, because a real pin *does* live inside
both halves of a real joint, the pair is excluded from the collision solver and
from the interference check. The one omission that could have been caught is
hidden by the same sentence that caused it.

So every reading was right while the metal was missing. `--example probe` asks
the geometry instead, and found nine of the tour's thirty-two joints to be
declarations and nothing else:

| | |
|---|---|
| **gear train** | the output wheel threaded on nothing, 9.1 mm above the deck |
| **four-bar** | four hinges, four pairs of perfectly aligned eyes, no pins |
| **bench** | a "gear" pair that was two plain discs 6 mm apart |
| **ratchet** | a carriage floating 12 mm over its frame |
| **latch** | an arm hinged to a post 14 mm away, a plunger with no guide, a pinion 9 mm off its own shaft, and a rack and pinion 50 mm apart with no teeth on either |

All of them ran, verified and passed every test in `tests/models.rs` before they
were built. They are built now — shafts, pins, clevises, rails, gibs, teeth — and
`every_declared_joint_is_actually_built` keeps it that way.

The measurement is one thing: probe a cylinder of points around the joint's own
axis, ask at each how far it is *into* each part (nought if inside), and take the
smallest sum. That is nought where a pin fills a bore, a fit's clearance where a
stem runs in a guide, and the whole empty distance where nothing was ever drawn.
It has to be a cylinder rather than the axis line, because half of these joints
are holes: a slider's axis runs straight down the middle of its guide, where
there is deliberately nothing at all.

## What the page shows

| | |
|---|---|
| **The bench** | every part drawn where the solver put it; orbit and zoom, click a part in the legend to hide it |
| **The transcript** | the seven acts — declarations, checks, travel, readings, cost, verification, measurements |
| **The joints** | every mate's reading against time, in the units it was declared in, with the travel the model *claimed* drawn behind it |
| **The model** | editable, with a picker for the six above. Change one and run again, and every number changes with it |

## How it is put together

[`Tour::run`](src/lib.rs) is ordinary Rust with no browser in it — it is what
the console example calls. [`web`](src/web.rs) is the thin part: geometry
crosses into JavaScript once, poses cross once, and a frame after that is a
matrix multiply. A part is rigid, so 240 frames of the bench are 54 kB of poses
where baking every frame would be 5.3 MB.

The page is a few hundred lines of WebGL2 with no dependencies —
[`web/mechanism-tour/index.html`](../../web/mechanism-tour/index.html).

## Notes on wasm

`wasm32` has no threads, so the SCAD evaluator and the CSG kernel run on the
main stack rather than the 1 GB worker they get natively. `build.sh` asks the
linker for 32 MB, which costs nothing until it is used.

**Curved booleans are the thing to spend sparingly on the web.** An exact CSG
kernel is integer arithmetic rather than floating point, and wasm has no
hardware for the 64- and 128-bit work underneath: three wheels with lightening
holes drilled through them took a second natively and **two minutes** in the
browser. The same wheels with a raised key instead of drilled holes take half a
second. Nothing else in the pipeline — the assembly, the solver, the checks, the
verification — is remotely as sensitive.

Model for model, the whole tour runs in 4–29 ms of solver time and well under a
second of wall clock, in the browser, for every mechanism above.
