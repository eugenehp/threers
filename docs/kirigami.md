# Kirigami corrugations

Part of the [threers](../README.md) documentation.

# Kirigami corrugations

Kirigami Expanded Miura plate lattices — pads, inclined walls, custom curvature,
discrete origami cells, tiled assemblies, TPMS/strut lattice cores, and a
joined crease net.

```bash
cargo run --release --example kirigami         # gallery PNG + SVG nets → out/
cargo run --release --example kirigami_orbit   # windowed viewer (13 presets + net)
```

```rust
use threers::{KirigamiExpandedMiura, KirigamiPreset};

let geom = KirigamiExpandedMiura::new(5, 6)
    .cell(28.5, 29.5, 68.0)
    .height(50.0)
    .thickness(1.2)
    .build();

let saddle = KirigamiPreset::Saddle.evaluate_default(1.2);
let hybrid = saddle.to_geometry_with_core(KirigamiCoreLattice::Gyroid, 0.20);
let tiled = KirigamiAssembly::tile(&KirigamiPreset::Planar.evaluate_default(1.2), 2, 2, 8.0)
    .evaluate();
std::fs::write("net.svg", saddle.develop_joined().to_svg()).unwrap();
```

Browser: `web/examples/kirigami.html` (needs a wasm build with `web/build.sh`).
