//! Colour spaces, interpolation between them, contrast, and CSS colour
//! strings.
//!
//! [`Color`] holds linear sRGB, as three.js's does. Everything here converts
//! to and from that: sRGB's encoded form, HSL, HSV and HWB on top of it; CIE
//! XYZ, CIELAB and CIELCh (D50, as CSS Color 4 and d3 define them, through
//! the Bradford adaptation from sRGB's D65); OkLab and OkLCh. The matrices
//! are CSS Color 4's sample code, and its worked examples are the tests.
//!
//! ```
//! use threers::math::{Color, ColorSpace};
//!
//! let red = Color::from_css("rebeccapurple").unwrap();
//! let lab = Color::from_css("lab(29.2345% 39.3825 20.0664)").unwrap();
//! assert_eq!(lab.to_hex(), 0x7d2329);
//! let mid = red.mix(lab, 0.5, ColorSpace::Oklch);
//! assert!(Color::WHITE.contrast_ratio(Color::BLACK) > 20.9);
//! ```

use super::color_schemes::CSS_NAMES;
use super::Color;

/// A space to interpolate colours in — see [`Color::mix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// Linear light: right for adding up light, grey in the middle of two
    /// saturated colours.
    LinearRgb,
    /// The encoded sRGB values: what CSS gradients and d3's default
    /// `interpolateRgb` blend through.
    Srgb,
    /// OkLab: perceptually even. The default.
    #[default]
    Oklab,
    /// OkLCh: OkLab's lightness, chroma and hue, the hue taken the short way
    /// round — keeps a gradient between two saturated colours saturated.
    Oklch,
    /// CIELAB (D50).
    Lab,
    /// CIELCh (D50), hue the short way round.
    Lch,
    /// HSL of the encoded values, hue the short way round.
    Hsl,
}

/// What went wrong parsing a CSS colour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorParseError(pub String);

impl std::fmt::Display for ColorParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "not a CSS colour: {}", self.0)
    }
}

impl std::error::Error for ColorParseError {}

/// sRGB's transfer function, one component.
pub(crate) fn srgb_encode(v: f32) -> f32 {
    let a = v.abs();
    let e = if a <= 0.003_130_8 {
        a * 12.92
    } else {
        1.055 * a.powf(1.0 / 2.4) - 0.055
    };
    e.copysign(v)
}

/// Its inverse.
pub(crate) fn srgb_decode(v: f32) -> f32 {
    let a = v.abs();
    let d = if a <= 0.040_45 {
        a / 12.92
    } else {
        ((a + 0.055) / 1.055).powf(2.4)
    };
    d.copysign(v)
}

fn mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// CSS Color 4, §17 sample code.
const LIN_SRGB_TO_XYZ_D65: [[f64; 3]; 3] = [
    [
        0.412_390_799_265_959_34,
        0.357_584_339_383_878,
        0.180_480_788_401_834_3,
    ],
    [
        0.212_639_005_871_510_27,
        0.715_168_678_767_756,
        0.072_192_315_360_733_71,
    ],
    [
        0.019_330_818_715_591_82,
        0.119_194_779_794_625_98,
        0.950_532_152_249_660_7,
    ],
];
const XYZ_D65_TO_LIN_SRGB: [[f64; 3]; 3] = [
    [
        3.240_969_941_904_522_6,
        -1.537_383_177_570_094,
        -0.498_610_760_293_003_4,
    ],
    [
        -0.969_243_636_280_879_6,
        1.875_967_501_507_720_2,
        0.041_555_057_407_175_59,
    ],
    [
        0.055_630_079_696_993_66,
        -0.203_976_958_888_976_52,
        1.056_971_514_242_878_6,
    ],
];
const D65_TO_D50: [[f64; 3]; 3] = [
    [
        1.047_929_820_840_548_8,
        0.022_946_793_341_019_088,
        -0.050_192_229_543_135_57,
    ],
    [
        0.029_627_815_688_159_344,
        0.990_434_484_573_249,
        -0.017_073_825_029_385_14,
    ],
    [
        -0.009_243_058_152_591_178,
        0.015_055_144_896_577_895,
        0.751_874_289_958_000_8,
    ],
];
const D50_TO_D65: [[f64; 3]; 3] = [
    [
        0.955_473_452_704_218_2,
        -0.023_098_536_874_261_423,
        0.063_259_308_661_021_7,
    ],
    [
        -0.028_369_706_963_208_136,
        1.009_995_458_005_822_6,
        0.021_041_398_966_943_008,
    ],
    [
        0.012_314_001_688_319_899,
        -0.020_507_696_433_477_912,
        1.330_365_936_608_075_3,
    ],
];
const LIN_P3_TO_XYZ_D65: [[f64; 3]; 3] = [
    [
        0.486_570_948_648_216_2,
        0.265_667_693_169_093_06,
        0.198_217_285_234_362_5,
    ],
    [
        0.228_974_564_069_748_8,
        0.691_738_521_836_506_4,
        0.079_286_914_093_745,
    ],
    [0.0, 0.045_113_381_858_902_64, 1.043_944_368_900_976],
];
/// D50's white, in XYZ.
const D50: [f64; 3] = [0.345_7 / 0.358_5, 1.0, (1.0 - 0.345_7 - 0.358_5) / 0.358_5];

/// Hue difference the short way round, in degrees.
fn hue_lerp(a: f32, b: f32, t: f32) -> f32 {
    let mut d = (b - a).rem_euclid(360.0);
    if d > 180.0 {
        d -= 360.0;
    }
    (a + d * t).rem_euclid(360.0)
}

impl Color {
    /// The encoded sRGB components, `0..1` for colours a display shows.
    pub fn to_srgb(&self) -> [f32; 3] {
        [
            srgb_encode(self.r),
            srgb_encode(self.g),
            srgb_encode(self.b),
        ]
    }

    /// From encoded sRGB components.
    pub fn from_srgb(c: [f32; 3]) -> Self {
        Self::new(srgb_decode(c[0]), srgb_decode(c[1]), srgb_decode(c[2]))
    }

    /// CIE XYZ, D65 white (sRGB's own).
    pub fn to_xyz(&self) -> [f32; 3] {
        let v = mul(
            &LIN_SRGB_TO_XYZ_D65,
            [self.r as f64, self.g as f64, self.b as f64],
        );
        [v[0] as f32, v[1] as f32, v[2] as f32]
    }

    /// From CIE XYZ, D65 white.
    pub fn from_xyz(xyz: [f32; 3]) -> Self {
        let v = mul(
            &XYZ_D65_TO_LIN_SRGB,
            [xyz[0] as f64, xyz[1] as f64, xyz[2] as f64],
        );
        Self::new(v[0] as f32, v[1] as f32, v[2] as f32)
    }

    /// CIELAB `[L, a, b]` with D50 white, as CSS `lab()` and d3 use: `L`
    /// 0..100, `a` and `b` roughly ±125.
    pub fn to_lab(&self) -> [f32; 3] {
        let xyz65 = mul(
            &LIN_SRGB_TO_XYZ_D65,
            [self.r as f64, self.g as f64, self.b as f64],
        );
        let xyz = mul(&D65_TO_D50, xyz65);
        const E: f64 = 216.0 / 24389.0;
        const K: f64 = 24389.0 / 27.0;
        let f = |v: f64| {
            if v > E {
                v.cbrt()
            } else {
                (K * v + 16.0) / 116.0
            }
        };
        let (fx, fy, fz) = (f(xyz[0] / D50[0]), f(xyz[1] / D50[1]), f(xyz[2] / D50[2]));
        [
            (116.0 * fy - 16.0) as f32,
            (500.0 * (fx - fy)) as f32,
            (200.0 * (fy - fz)) as f32,
        ]
    }

    /// From CIELAB with D50 white.
    pub fn from_lab(lab: [f32; 3]) -> Self {
        const E: f64 = 216.0 / 24389.0;
        const K: f64 = 24389.0 / 27.0;
        let (l, a, b) = (lab[0] as f64, lab[1] as f64, lab[2] as f64);
        let fy = (l + 16.0) / 116.0;
        let fx = a / 500.0 + fy;
        let fz = fy - b / 200.0;
        let x = if fx.powi(3) > E {
            fx.powi(3)
        } else {
            (116.0 * fx - 16.0) / K
        };
        let y = if l > K * E { fy.powi(3) } else { l / K };
        let z = if fz.powi(3) > E {
            fz.powi(3)
        } else {
            (116.0 * fz - 16.0) / K
        };
        let xyz65 = mul(&D50_TO_D65, [x * D50[0], y * D50[1], z * D50[2]]);
        let v = mul(&XYZ_D65_TO_LIN_SRGB, xyz65);
        Self::new(v[0] as f32, v[1] as f32, v[2] as f32)
    }

    /// CIELCh `[L, C, h]` (D50): lightness, chroma, hue in degrees.
    pub fn to_lch(&self) -> [f32; 3] {
        let [l, a, b] = self.to_lab();
        [l, a.hypot(b), b.atan2(a).to_degrees().rem_euclid(360.0)]
    }

    /// From CIELCh (D50), hue in degrees.
    pub fn from_lch(lch: [f32; 3]) -> Self {
        let h = lch[2].to_radians();
        Self::from_lab([lch[0], lch[1] * h.cos(), lch[1] * h.sin()])
    }

    /// OkLCh `[L, C, h]`: OkLab's lightness, chroma and hue in degrees.
    pub fn to_oklch(&self) -> [f32; 3] {
        let [l, a, b] = self.to_oklab();
        [l, a.hypot(b), b.atan2(a).to_degrees().rem_euclid(360.0)]
    }

    /// From OkLCh, hue in degrees.
    pub fn from_oklch(lch: [f32; 3]) -> Self {
        let h = lch[2].to_radians();
        Self::from_oklab([lch[0], lch[1] * h.cos(), lch[1] * h.sin()])
    }

    /// Hue, saturation and value of the encoded sRGB form, each `0..1`.
    pub fn to_hsv(&self) -> [f32; 3] {
        let [r, g, b] = self.to_srgb().map(|v| v.clamp(0.0, 1.0));
        let max = r.max(g).max(b);
        let d = max - r.min(g).min(b);
        let h = if d < 1e-6 {
            0.0
        } else if max == r {
            ((g - b) / d).rem_euclid(6.0)
        } else if max == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
        [
            (h / 6.0).rem_euclid(1.0),
            if max > 0.0 { d / max } else { 0.0 },
            max,
        ]
    }

    /// From hue, saturation and value of the encoded form, each `0..1`.
    pub fn from_hsv(h: f32, s: f32, v: f32) -> Self {
        let l = v * (1.0 - s * 0.5);
        let sl = if l <= 0.0 || l >= 1.0 {
            0.0
        } else {
            (v - l) / l.min(1.0 - l)
        };
        Self::from_hsl(h, sl, l)
    }

    /// Hue, whiteness and blackness (CSS `hwb()`), each `0..1`.
    pub fn to_hwb(&self) -> [f32; 3] {
        let [h, s, v] = self.to_hsv();
        [h, (1.0 - s) * v, 1.0 - v]
    }

    /// From hue, whiteness and blackness, each `0..1`.
    pub fn from_hwb(h: f32, w: f32, b: f32) -> Self {
        if w + b >= 1.0 {
            let grey = w / (w + b);
            return Self::from_srgb([grey; 3]);
        }
        let v = 1.0 - b;
        Self::from_hsv(h, 1.0 - w / v, v)
    }

    /// The colour `t` of the way to `other`, through `space`. Hue spaces take
    /// the short way round the wheel, and a grey (no hue) takes the other
    /// colour's.
    pub fn mix(self, other: Self, t: f32, space: ColorSpace) -> Self {
        let l3 = |a: [f32; 3], b: [f32; 3]| [0, 1, 2].map(|k| a[k] + (b[k] - a[k]) * t);
        // Polar spaces: the hue of a colour with no chroma is meaningless, so
        // it borrows the other's rather than swinging round from red.
        let polar = |a: [f32; 3], b: [f32; 3], grey: f32| {
            let (ha, hb) = match (a[1] < grey, b[1] < grey) {
                (true, false) => (b[2], b[2]),
                (false, true) => (a[2], a[2]),
                _ => (a[2], b[2]),
            };
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                hue_lerp(ha, hb, t),
            ]
        };
        match space {
            ColorSpace::LinearRgb => self.lerp(other, t),
            ColorSpace::Srgb => Self::from_srgb(l3(self.to_srgb(), other.to_srgb())),
            ColorSpace::Oklab => self.lerp_oklab(other, t),
            ColorSpace::Oklch => Self::from_oklch(polar(self.to_oklch(), other.to_oklch(), 1e-4)),
            ColorSpace::Lab => Self::from_lab(l3(self.to_lab(), other.to_lab())),
            ColorSpace::Lch => Self::from_lch(polar(self.to_lch(), other.to_lch(), 1e-2)),
            ColorSpace::Hsl => {
                let (a, b) = (self.to_hsl(), other.to_hsl());
                // HSL as `[l, s, h]`, so `polar` reads saturation as chroma.
                let [l, s, h] = polar([a[2], a[1], a[0] * 360.0], [b[2], b[1], b[0] * 360.0], 1e-4);
                Self::from_hsl(h / 360.0, s, l)
            }
        }
    }

    /// WCAG 2's contrast ratio, `1..21`: 4.5 is the floor for body text, 3
    /// for large text and interface parts.
    pub fn contrast_ratio(&self, other: Self) -> f32 {
        let (a, b) = (
            self.luminance().max(0.0) + 0.05,
            other.luminance().max(0.0) + 0.05,
        );
        a.max(b) / a.min(b)
    }

    /// `#rrggbb`.
    pub fn to_css_hex(&self) -> String {
        format!("#{:06x}", self.to_hex())
    }

    /// A CSS colour, opaque or not: [`parse_css`](Self::parse_css) without the
    /// alpha.
    pub fn from_css(s: &str) -> Result<Self, ColorParseError> {
        Self::parse_css(s).map(|(c, _)| c)
    }

    /// A CSS colour string and its alpha — three.js's `Color.setStyle`, and
    /// the rest of CSS Color 4:
    ///
    /// - `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`;
    /// - the 148 named colours and `transparent`;
    /// - `rgb()`, `rgba()`, `hsl()`, `hsla()`, `hwb()` — comma or space
    ///   separated, numbers or percentages, an optional `/ alpha`, hues in
    ///   `deg`, `rad`, `grad` or `turn`;
    /// - `lab()`, `lch()`, `oklab()`, `oklch()`;
    /// - `color(srgb …)`, `color(srgb-linear …)`, `color(display-p3 …)`,
    ///   `color(xyz …)` / `xyz-d65` / `xyz-d50`.
    ///
    /// Out-of-gamut colours (a saturated `oklch()`, wide `display-p3`) keep
    /// components outside `0..1`; clamp if that matters.
    pub fn parse_css(s: &str) -> Result<(Self, f32), ColorParseError> {
        let err = || ColorParseError(s.to_string());
        let t = s.trim().to_ascii_lowercase();
        if let Some(hex) = t.strip_prefix('#') {
            return parse_hex(hex).ok_or_else(err);
        }
        if t == "transparent" {
            return Ok((Color::BLACK, 0.0));
        }
        if let Some(&(_, v)) = CSS_NAMES.iter().find(|(n, _)| *n == t) {
            return Ok((Color::from_hex(v), 1.0));
        }
        let open = t.find('(').ok_or_else(err)?;
        let name = t[..open].trim();
        let inner = t[open + 1..].strip_suffix(')').ok_or_else(err)?;
        // Arguments, with any `/ alpha`.
        let (body, alpha) = match inner.split_once('/') {
            Some((b, a)) => (b, Some(a.trim())),
            None => (inner, None),
        };
        let mut args: Vec<&str> = body
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|a| !a.is_empty())
            .collect();
        let mut alpha = alpha.map(str::to_string);
        // The legacy comma form puts alpha fourth.
        if alpha.is_none() && args.len() == 4 && name != "color" {
            alpha = Some(args.pop().unwrap().to_string());
        }
        let alpha = match alpha {
            Some(a) => number(&a, 1.0).ok_or_else(err)?.clamp(0.0, 1.0),
            None => 1.0,
        };
        let n = |i: usize, full: f32| args.get(i).and_then(|a| number(a, full)).ok_or_else(err);
        let hue = |i: usize| args.get(i).and_then(|a| angle(a)).ok_or_else(err);
        let color = match name {
            "rgb" | "rgba" if args.len() == 3 => Color::from_srgb([
                n(0, 255.0)? / 255.0,
                n(1, 255.0)? / 255.0,
                n(2, 255.0)? / 255.0,
            ]),
            "hsl" | "hsla" if args.len() == 3 => {
                Color::from_hsl(hue(0)? / 360.0, n(1, 100.0)? / 100.0, n(2, 100.0)? / 100.0)
            }
            "hwb" if args.len() == 3 => {
                Color::from_hwb(hue(0)? / 360.0, n(1, 100.0)? / 100.0, n(2, 100.0)? / 100.0)
            }
            "lab" if args.len() == 3 => Color::from_lab([n(0, 100.0)?, n(1, 125.0)?, n(2, 125.0)?]),
            "lch" if args.len() == 3 => Color::from_lch([n(0, 100.0)?, n(1, 150.0)?, hue(2)?]),
            "oklab" if args.len() == 3 => Color::from_oklab([n(0, 1.0)?, n(1, 0.4)?, n(2, 0.4)?]),
            "oklch" if args.len() == 3 => Color::from_oklch([n(0, 1.0)?, n(1, 0.4)?, hue(2)?]),
            "color" if args.len() == 4 => {
                let v = [n(1, 1.0)?, n(2, 1.0)?, n(3, 1.0)?];
                match args[0] {
                    "srgb" => Color::from_srgb(v),
                    "srgb-linear" => Color::new(v[0], v[1], v[2]),
                    "display-p3" => {
                        let lin = v.map(|c| srgb_decode(c) as f64);
                        let xyz = mul(&LIN_P3_TO_XYZ_D65, lin);
                        Color::from_xyz([xyz[0] as f32, xyz[1] as f32, xyz[2] as f32])
                    }
                    "xyz" | "xyz-d65" => Color::from_xyz(v),
                    "xyz-d50" => {
                        let xyz = mul(&D50_TO_D65, [v[0] as f64, v[1] as f64, v[2] as f64]);
                        Color::from_xyz([xyz[0] as f32, xyz[1] as f32, xyz[2] as f32])
                    }
                    _ => return Err(err()),
                }
            }
            _ => return Err(err()),
        };
        Ok((color, alpha))
    }
}

fn parse_hex(hex: &str) -> Option<(Color, f32)> {
    let digit = |c: char| c.to_digit(16);
    let chars: Vec<u32> = hex.chars().map(digit).collect::<Option<_>>()?;
    let (rgb, a) = match chars.len() {
        3 | 4 => (
            [chars[0] * 17, chars[1] * 17, chars[2] * 17],
            chars.get(3).map_or(255, |a| a * 17),
        ),
        6 | 8 => (
            [
                chars[0] * 16 + chars[1],
                chars[2] * 16 + chars[3],
                chars[4] * 16 + chars[5],
            ],
            if chars.len() == 8 {
                chars[6] * 16 + chars[7]
            } else {
                255
            },
        ),
        _ => return None,
    };
    Some((
        Color::from_hex((rgb[0] << 16) | (rgb[1] << 8) | rgb[2]),
        a as f32 / 255.0,
    ))
}

/// A CSS number, a percentage of `full`, or `none` (0).
fn number(s: &str, full: f32) -> Option<f32> {
    if s == "none" {
        return Some(0.0);
    }
    match s.strip_suffix('%') {
        Some(p) => p.parse::<f32>().ok().map(|v| v / 100.0 * full),
        None => s.parse::<f32>().ok(),
    }
}

/// A CSS hue, in degrees.
fn angle(s: &str) -> Option<f32> {
    if s == "none" {
        return Some(0.0);
    }
    for (unit, k) in [
        ("deg", 1.0),
        ("grad", 0.9),
        ("rad", 180.0 / std::f32::consts::PI),
        ("turn", 360.0),
    ] {
        if let Some(v) = s.strip_suffix(unit) {
            return v.parse::<f32>().ok().map(|v| v * k);
        }
    }
    s.parse::<f32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> u32 {
        Color::from_css(s)
            .unwrap_or_else(|e| panic!("{e}"))
            .to_hex()
    }

    fn near(a: u32, b: u32) -> bool {
        (0..3).all(|k| (((a >> (8 * k)) & 0xff) as i32 - ((b >> (8 * k)) & 0xff) as i32).abs() <= 1)
    }

    #[test]
    fn css_strings_parse_to_their_colours() {
        assert_eq!(hex("#f00"), 0xff0000);
        assert_eq!(hex("#00ff0080"), 0x00ff00);
        assert_eq!(Color::parse_css("#00ff0080").unwrap().1, 128.0 / 255.0);
        assert_eq!(hex("RebeccaPurple"), 0x663399);
        assert_eq!(hex("rgb(255, 128, 0)"), 0xff8000);
        assert_eq!(hex("rgb(100% 50% 0%)"), 0xff8000);
        assert_eq!(Color::parse_css("rgba(0,0,0,0.25)").unwrap().1, 0.25);
        assert_eq!(Color::parse_css("rgb(0 0 0 / 40%)").unwrap().1, 0.4);
        assert_eq!(hex("hsl(120, 100%, 50%)"), 0x00ff00);
        assert_eq!(hex("hsl(0.5turn 100% 50%)"), 0x00ffff);
        assert_eq!(hex("hwb(0 0% 0%)"), 0xff0000);
        assert_eq!(hex("hwb(0 50% 50%)"), 0x808080);
        assert_eq!(hex("color(srgb 1 0 0)"), 0xff0000);
        assert_eq!(hex("color(srgb-linear 0.21586 0.21586 0.21586)"), 0x808080);
        assert_eq!(Color::parse_css("transparent").unwrap().1, 0.0);
        assert!(Color::from_css("rgb(1,2)").is_err());
        assert!(Color::from_css("#12345").is_err());
        assert!(Color::from_css("notacolour").is_err());
    }

    /// CSS Color 4's own worked example: one colour, written in each space.
    #[test]
    fn the_css_color_4_example_reads_alike_in_every_space() {
        for s in [
            "rgb(125 35 41)",
            "lab(29.2345% 39.3825 20.0664)",
            "lch(29.2345% 44.2 27)",
            "oklab(40.101% 0.1147 0.0453)",
            "oklch(40.101% 0.12332 21.555)",
        ] {
            assert!(near(hex(s), 0x7d2329), "{s} reads #{:06x}", hex(s));
        }
    }

    #[test]
    fn spaces_round_trip() {
        for h in [
            0x000000u32,
            0xffffff,
            0xff0000,
            0x3498db,
            0x7d2329,
            0x12ab34,
            0xfde725,
        ] {
            let c = Color::from_hex(h);
            assert_eq!(Color::from_lab(c.to_lab()).to_hex(), h, "lab");
            assert_eq!(Color::from_lch(c.to_lch()).to_hex(), h, "lch");
            assert_eq!(Color::from_oklch(c.to_oklch()).to_hex(), h, "oklch");
            assert_eq!(Color::from_xyz(c.to_xyz()).to_hex(), h, "xyz");
            let [hh, s, v] = c.to_hsv();
            assert_eq!(Color::from_hsv(hh, s, v).to_hex(), h, "hsv");
            let [hh, w, b] = c.to_hwb();
            assert_eq!(Color::from_hwb(hh, w, b).to_hex(), h, "hwb");
        }
        let white = Color::WHITE.to_lab();
        assert!(
            (white[0] - 100.0).abs() < 1e-3 && white[1].abs() < 1e-3 && white[2].abs() < 1e-3,
            "{white:?}"
        );
    }

    #[test]
    fn mixing_spaces_differ_where_they_should() {
        let (blue, yellow) = (Color::from_hex(0x0000ff), Color::from_hex(0xffff00));
        // Through linear light, grey; through OkLCh, still saturated.
        let chroma = |c: Color| c.to_oklch()[1];
        assert!(
            chroma(blue.mix(yellow, 0.5, ColorSpace::Oklch))
                > 2.0 * chroma(blue.mix(yellow, 0.5, ColorSpace::LinearRgb))
        );
        // The ends are the ends in every space.
        for space in [
            ColorSpace::LinearRgb,
            ColorSpace::Srgb,
            ColorSpace::Oklab,
            ColorSpace::Oklch,
            ColorSpace::Lab,
            ColorSpace::Lch,
            ColorSpace::Hsl,
        ] {
            assert_eq!(blue.mix(yellow, 0.0, space).to_hex(), 0x0000ff, "{space:?}");
            assert_eq!(blue.mix(yellow, 1.0, space).to_hex(), 0xffff00, "{space:?}");
        }
        // A grey takes the other end's hue rather than swinging through red.
        let to_green =
            Color::from_hex(0x808080).mix(Color::from_hex(0x00ff00), 0.5, ColorSpace::Hsl);
        let [r, g, b] = to_green.to_srgb();
        assert!(g > r && g > b, "{:06x}", to_green.to_hex());
    }

    #[test]
    fn contrast_follows_wcag() {
        assert!((Color::WHITE.contrast_ratio(Color::BLACK) - 21.0).abs() < 0.01);
        // #767676 on white is WCAG's best-known 4.5:1 grey.
        let r = Color::from_hex(0x767676).contrast_ratio(Color::WHITE);
        assert!((r - 4.54).abs() < 0.02, "{r}");
    }
}
