# threers-math

The math under [threers](https://github.com/eugenehp/threers): vectors,
matrices, quaternions, Euler angles, colours, colour scales and ramps, boxes,
spheres, rays, planes and frustums, Delaunay triangulation, convex hulls and
Voronoi cells, Web Mercator and globe coordinates, and tone mapping — with
three.js's semantics, and no dependencies.

Beyond three.js, `polygon_ops` does booleans (union, intersection,
difference, xor), offsets (round, square, bevel and miter joins; open paths
butt, square or round ended) and simplification of polygons in 64-bit
integer coordinates, results exact to the point and the order of points.
Polygons that nearly touch — T-junctions after rounding to the grid — can
come out wrong, as in the C++ library; grow each by a unit or two first
(see the module docs).
Its algorithms follow Angus Johnson's polygon clipping and offsetting library
and are used under the Boost Software License 1.0
(`src/polygon_ops/LICENSE-BSL-1.0`).

`threers` re-exports all of it as `threers::math`; depend on this crate
directly when the math is all you need.

```rust
use threers_math::{Quaternion, Vector3};

let q = Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_2);
let v = Vector3::new(1.0, 0.0, 0.0).apply_quaternion(q);
assert!((v.z + 1.0).abs() < 1e-6);
```
