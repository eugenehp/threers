# threers-math

The math under [threers](https://github.com/eugenehp/threers): vectors,
matrices, quaternions, Euler angles, colours, colour scales and ramps, boxes,
spheres, rays, planes and frustums, Delaunay triangulation, convex hulls and
Voronoi cells, Web Mercator and globe coordinates, and tone mapping — with
three.js's semantics, and no dependencies.

`threers` re-exports all of it as `threers::math`; depend on this crate
directly when the math is all you need.

```rust
use threers_math::{Quaternion, Vector3};

let q = Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_2);
let v = Vector3::new(1.0, 0.0, 0.0).apply_quaternion(q);
assert!((v.z + 1.0).abs() < 1e-6);
```
