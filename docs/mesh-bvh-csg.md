# Mesh BVH and CSG

Part of the [threers](../README.md) documentation.

# Optional: mesh-bvh / CSG

```bash
# BVH picking demo
cargo run --example mesh_bvh_picking --features mesh-bvh

# The hierarchy parity scene, evaluated live; `-- 2` stops after two operations
cargo run --example bvh_csg_hierarchy --features bvh-csg
cargo run --example bvh_csg_hierarchy --features bvh-csg -- 2
```

The CSG is three-bvh-csg@0.0.16's, result for result: every evaluation the
bvh-csg parity scenes make, recorded from the library itself
(`tests/parity/export-bvh-csg-upstream.mjs`), comes out with the same
triangles, area and volume (`tests/csg_upstream.rs`). That takes
three-mesh-bvh's tree and traversal order and three.js's f64 arithmetic down
to the operand order: a face is cut in the order the BVH meets its cutting
triangles, and a vertex lying on another face is classified by the last bit.

A BVH built for *querying* — raycast, closest-point — wants
`split_degenerate`, which keeps a split plane that separates nothing from
collapsing the whole subtree into a linear scan. It is off by default because
`bvhcast` and the CSG evaluator read the tree's shape rather than merely
querying it; see [Native (desktop)](#native-desktop), where baking the
procedural city is what turned this up, for why that matters.

```rust
// Cargo.toml: threers = { version = "0.0.8", features = ["mesh-bvh"] }
use threers::mesh_bvh::{BuildOptions, MeshBvh, SAH};

let bvh = MeshBvh::build(
    &geometry,
    BuildOptions {
        strategy: SAH,
        split_degenerate: true,   // a query tree: nothing reads its shape
        ..Default::default()
    },
)
.expect("geometry has positions");
```

```rust
// Cargo.toml: threers = { version = "0.0.8", features = ["bvh-csg"] }
use threers::{CsgBrush, CsgEvaluator, BoxGeometry, SUBTRACTION};

let mut ev = CsgEvaluator::new();
let mut a = CsgBrush::new(BoxGeometry::new(2.0, 2.0, 2.0));
let mut b = CsgBrush::new(BoxGeometry::new(1.0, 1.0, 1.0));
let _geom = ev.evaluate(&mut a, &mut b, SUBTRACTION);
```
