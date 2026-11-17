`minimal-v2.spz`, `minimal-v3.spz` and `minimal-v4.spz` come from PlayCanvas's
`splat-transform` (MIT, `test/fixtures/splat/`), where
`test/fixtures/create-fixtures.mjs` makes them. Version 2 is laid out by hand;
versions 3 and 4 are written by `@adobe/spz`, the reference library built for
the web, from four splats on a 2×2 grid converted from `y`-down to `y`-up.
`tests/splats.rs` checks the values that script put in.
