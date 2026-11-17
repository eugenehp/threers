# threers-splat-sort

Back-to-front ordering for Gaussian splats: the sort behind
[`threers::splats`](https://docs.rs/threers/latest/threers/splats/), on its
own and with no dependencies.

```rust
use threers_splat_sort::DepthSort;

let centres = [[0.0, 0.0, -1.0], [0.0, 0.0, -5.0], [0.0, 0.0, -3.0]];
let mut sort = DepthSort::new(&centres);
// Ascending along the axis: with a camera's view axis (row 2 of its
// model-view matrix), farthest first.
sort.sort([0.0, 0.0, 1.0]);
assert_eq!(sort.order(), &[1, 2, 0]);
```

The depths are quantised to 24 bits across the splats' bounding box and
sorted by a two-pass radix sort. The keys come from a loop of fixed-width,
branch-free chunks that the compiler turns into SIMD (NEON, SSE/AVX, or wasm
`simd128` when it is enabled); the sort itself is two scatters.

It is a crate of its own so a wasm build sized with `opt-level = "z"` can
still compile it at 3:

```toml
[profile.release-wasm.package.threers-splat-sort]
opt-level = 3
```
