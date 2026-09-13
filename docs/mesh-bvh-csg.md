# Mesh BVH and CSG

Part of the [threers](../README.md) documentation.

# Optional: mesh-bvh / CSG

```bash
# BVH picking demo
cargo run --example mesh_bvh_picking --features mesh-bvh

# Hierarchy CSG (exact TriKey parity with JS three-bvh-csg@0.0.16)
cargo run --example bvh_csg_hierarchy --features bvh-csg
cargo run --example bvh_csg_steps --features bvh-csg -- 2 --live
```

A BVH built for *querying* — raycast, closest-point — wants
`split_degenerate`, which keeps a split plane that separates nothing from
collapsing the whole subtree into a linear scan. It is off by default because
`bvhcast` and the CSG evaluator read the tree's shape rather than merely
querying it; see [Native (desktop)](#native-desktop), where baking the
procedural city is what turned this up, for why that matters.

```rust
// Cargo.toml: threers = { version = "0.0.5", features = ["mesh-bvh"] }
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
// Cargo.toml: threers = { version = "0.0.5", features = ["bvh-csg"] }
use threers::{CsgBrush, CsgEvaluator, BoxGeometry, SUBTRACTION};

let mut ev = CsgEvaluator::new();
let mut a = CsgBrush::new(BoxGeometry::new(2.0, 2.0, 2.0));
let mut b = CsgBrush::new(BoxGeometry::new(1.0, 1.0, 1.0));
let _geom = ev.evaluate(&mut a, &mut b, SUBTRACTION);
```
