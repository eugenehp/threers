# OpenSCAD front-end demos

Parse any of these with the pure-Rust front end and write an STL:

```sh
cargo run --example scad2stl --features openscad -- examples/scad-demos/part.scad /tmp/part.stl
```

They exercise the features added on top of the core language. Each was checked
against real OpenSCAD/CGAL rendering the same source (`scripts/ci-openscad.sh`
plumbing) — "parity" below means watertight + matching volume + matching Euler
characteristic + Hausdorff within tolerance.

| File | Shows | vs OpenSCAD/CGAL |
|------|-------|------------------|
| `booleans_2d.scad` | 2D `union`/`difference` of a plus-with-hole (arrangement kernel: non-convex, holes, collinear edges) | **parity** (Hausdorff 2e-5) |
| `sheared.scad` | `multmatrix` (shear + scale) | **parity** (Hausdorff 0) |
| `default_cyl.scad` | default fragment count *and phase* from `$fa`/`$fs` (OpenSCAD-exact tessellation) | **parity** (Hausdorff 4e-6) |
| `terrain.scad` + `terrain.dat` | `surface()` heightmap → solid | **parity** (Hausdorff 0) |
| `rounded_box.scad` | 3D `minkowski` rounding a convex box with a sphere | vol + topology match (sphere facet phase) |
| `rounded_plate.scad` | `offset(r)` + 2D boolean holes + `$fa`/`$fs`, extruded | vol + topology match; → parity as `$fn` rises |
| `part.scad` + `hardware.scad` | `use <…>` library, 2D holes subtracted then extruded (avoids 3D CSG) | watertight; vol + topology match |
| `label.scad` | `text()` via the bundled DejaVu Sans font, extruded | smooth, watertight glyphs; won't match OpenSCAD's Liberation Sans (pass `font="…ttf"` to match) |

Also gated in `scripts/ci-openscad.sh`: `booleans_2d`, `sheared`, `default_cyl`,
`surf_dat` (surface), `mink2d` (non-convex 2D Minkowski). `offset`/`minkowski`-with-
sphere match geometry (volume, topology) at default resolution and tighten toward
Hausdorff parity as `$fn` rises. `text()` uses a bundled DejaVu Sans outline font
(smooth, watertight glyphs); it differs from OpenSCAD's Liberation Sans, so supply
`font="…ttf"` to control the exact typeface.
