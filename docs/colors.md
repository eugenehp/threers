# Colour

`threers::math` holds colours, the spaces they convert between, the named
schemes data is drawn in, and the scales that map numbers onto them.

## `Color` and its spaces

A [`Color`] is linear sRGB, as three.js's is: `Color::from_hex(0xff8000)`
decodes the sRGB literal, and `to_hex` encodes it back. Beside that:

| | |
|---|---|
| `to_srgb` / `from_srgb` | The encoded components |
| `to_hsl` / `from_hsl`, `to_hsv` / `from_hsv`, `to_hwb` / `from_hwb` | Of the encoded form, each `0..1` |
| `to_xyz` / `from_xyz` | CIE XYZ, D65 |
| `to_lab` / `from_lab`, `to_lch` / `from_lch` | CIELAB and CIELCh, D50, as CSS and d3 define them |
| `to_oklab` / `from_oklab`, `to_oklch` / `from_oklch` | OkLab and OkLCh |
| `mix(other, t, space)` | Interpolation in a [`ColorSpace`]: linear light, sRGB, OkLab (the default), OkLCh, Lab, LCh or HSL — hues the short way round, a grey taking the other end's hue |
| `contrast_ratio(other)` | WCAG 2's, 1 to 21 |
| `from_css` / `parse_css` | Any CSS Color 4 string, below; `parse_css` returns the alpha too |

The conversion matrices are CSS Color 4's sample code, and its worked
example — one colour written as `rgb()`, `lab()`, `lch()`, `oklab()` and
`oklch()` — is a test.

### CSS strings

`Color::from_css` reads `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`; the 148 named
colours and `transparent`; `rgb()`, `rgba()`, `hsl()`, `hsla()`, `hwb()` —
comma or space separated, numbers or percentages, `/ alpha`, hues in `deg`,
`rad`, `grad` or `turn`; `lab()`, `lch()`, `oklab()`, `oklch()`; and
`color(srgb …)`, `color(srgb-linear …)`, `color(display-p3 …)`,
`color(xyz-d65 …)`. In the three.js shim, `Color.setStyle` takes all of it:
what three.js parses itself it parses as three.js does, and the rest goes to
this parser (`WebColor.fromCss` from JavaScript).

## Schemes

`ColorRamp` is a gradient, blended in any `ColorSpace`; `Palette` is a set of
distinct colours for categories. Named ones are d3-scale-chromatic's, from
ColorBrewer's and Tableau's own swatches:

| Kind | Names |
|---|---|
| Sequential | `viridis`, `magma`, `inferno`, `plasma`, `cividis`, `turbo`; `blues`, `greens`, `greys`, `oranges`, `purples`, `reds`, `bu_gn`, `bu_pu`, `gn_bu`, `or_rd`, `pu_bu`, `pu_bu_gn`, `pu_rd`, `rd_pu`, `yl_gn`, `yl_gn_bu`, `yl_or_br`, `yl_or_rd` |
| Diverging | `br_bg`, `pi_yg`, `pr_gn`, `pu_or`, `rd_bu`, `rd_gy`, `rd_yl_bu`, `rd_yl_gn`, `spectral`, `cool_warm` |
| Cyclic | `twilight` |
| Categorical (`Palette`) | `category10` / `tableau10`, `tableau20`, `set1`, `set2`, `set3`, `pastel1`, `pastel2`, `paired`, `dark2`, `accent` |

`ColorRamp::named` and `Palette::named` find them however they are spelled —
`"RdYlBu"`, `"rd_yl_bu"` and `"interpolateRdYlBu"` are one scheme.
`reversed()`, `slice(a, b)` and `in_space(space)` adjust a ramp;
`Palette::from_ramp(&ramp, n)` makes an `n`-class one.

## Scales

`ColorScale` maps numbers to colours, as d3's scales do:

```rust
use threers::math::{ColorRamp, ColorScale};

// Orders of magnitude: log-scaled.
let density = ColorScale::sequential(ColorRamp::named("yl_or_rd").unwrap(), 1.0, 1e4).log(10.0);
// Around a meaningful zero, the halves stretched independently.
let anomaly = ColorScale::diverging(ColorRamp::named("rd_bu").unwrap().reversed(), -3.0, 0.0, 5.0);
// A choropleth: five classes of equal count.
let classes = ColorScale::quantile(ColorRamp::named("blues").unwrap(), &[3.0, 9.0, 1.0, 7.0, 4.0], 5);
let rgba: Vec<[u8; 4]> = classes.map_rgba8(&[3.0, 9.0]);
```

Continuous kinds are linear, `log`, `pow`, `sqrt` and `symlog`; classed ones
`quantize` (equal widths), `quantile` (equal counts) and `threshold` (your
breaks). `map_rgba8` gives the bytes marks layers and data textures take, and
`legend(n)` the rows of a legend. From JavaScript, `WebColorScale` is the
same, with `mapRgba8` returning a `Uint8Array`.
