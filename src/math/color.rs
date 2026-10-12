#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };
    pub const RED: Self = Self {
        r: 1.0,
        g: 0.0,
        b: 0.0,
    };
    pub const GREEN: Self = Self {
        r: 0.0,
        g: 1.0,
        b: 0.0,
    };
    pub const BLUE: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 1.0,
    };

    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Build a color from a packed 0xRRGGBB hex value, three.js style.
    /// three.js treats hex/CSS colors as sRGB and decodes them to linear for
    /// shader work, so we do the same — otherwise PBR output diverges by the
    /// linear→sRGB curve on every color literal.
    pub fn from_hex(hex: u32) -> Self {
        fn srgb_to_linear(s: f32) -> f32 {
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        }
        Self {
            r: srgb_to_linear(((hex >> 16) & 0xff) as f32 / 255.0),
            g: srgb_to_linear(((hex >> 8) & 0xff) as f32 / 255.0),
            b: srgb_to_linear((hex & 0xff) as f32 / 255.0),
        }
    }

    pub fn to_array(&self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// Back to a packed `0xRRGGBB`, re-encoding to sRGB on the way out so it
    /// round-trips with [`from_hex`](Self::from_hex).
    pub fn to_hex(&self) -> u32 {
        let enc = |v: f32| {
            let v = v.clamp(0.0, 1.0);
            let s = if v <= 0.003_130_8 {
                v * 12.92
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0 + 0.5) as u32
        };
        (enc(self.r) << 16) | (enc(self.g) << 8) | enc(self.b)
    }

    /// Hue, saturation and lightness, each `0..1`, as three.js's `setHSL`
    /// takes them — which means they describe the *sRGB* colour, and the result
    /// is decoded to linear like any other literal.
    pub fn from_hsl(h: f32, s: f32, l: f32) -> Self {
        let h = h.rem_euclid(1.0);
        let s = s.clamp(0.0, 1.0);
        let l = l.clamp(0.0, 1.0);
        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
        let m = l - c * 0.5;
        let (r, g, b) = match (h * 6.0) as u32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        let to_linear = |v: f32| {
            if v <= 0.040_45 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        Self {
            r: to_linear(r + m),
            g: to_linear(g + m),
            b: to_linear(b + m),
        }
    }

    /// The inverse of [`from_hsl`](Self::from_hsl): hue, saturation and
    /// lightness of the sRGB form, each `0..1`.
    pub fn to_hsl(&self) -> [f32; 3] {
        let enc = |v: f32| {
            let v = v.clamp(0.0, 1.0);
            if v <= 0.003_130_8 {
                v * 12.92
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            }
        };
        let (r, g, b) = (enc(self.r), enc(self.g), enc(self.b));
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let l = (max + min) * 0.5;
        let d = max - min;
        if d < 1e-6 {
            return [0.0, 0.0, l];
        }
        let s = d / (1.0 - (2.0 * l - 1.0).abs()).max(1e-6);
        let h = if max == r {
            ((g - b) / d).rem_euclid(6.0)
        } else if max == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
        [(h / 6.0).rem_euclid(1.0), s.clamp(0.0, 1.0), l]
    }

    /// Mix two colours in **linear** light.
    ///
    /// This is the one that is physically right for *light* — two lamps, two
    /// samples of an image, anything being added up. It is the wrong one for
    /// mixing two *paints*: linear light interpolation between saturated blue
    /// and saturated yellow passes through a desaturated grey, because the
    /// midpoint of the two spectra genuinely is grey and that is not what
    /// anyone picking a gradient meant. Use [`lerp_oklab`](Self::lerp_oklab)
    /// for that.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }

    /// Mix two colours perceptually, through OkLab.
    ///
    /// What a gradient wants. OkLab is built so that equal steps in it look
    /// like equal steps, so the midpoint of blue and yellow is the green a
    /// person would name — not the grey that linear light gives, and not the
    /// muddy purple that interpolating sRGB bytes gives.
    pub fn lerp_oklab(self, other: Self, t: f32) -> Self {
        let a = self.to_oklab();
        let b = other.to_oklab();
        Self::from_oklab([
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ])
    }

    /// `[L, a, b]` in OkLab. `L` is 0..1 lightness; `a` and `b` are the
    /// green–red and blue–yellow axes, roughly ±0.4.
    ///
    /// Defined from *linear* sRGB, which is what this type already stores — so
    /// there is no encode step here, and putting one in is the usual way this
    /// conversion comes out wrong.
    pub fn to_oklab(&self) -> [f32; 3] {
        let l = 0.412_221_47 * self.r + 0.536_332_55 * self.g + 0.051_445_995 * self.b;
        let m = 0.211_903_5 * self.r + 0.680_699_5 * self.g + 0.107_396_96 * self.b;
        let s = 0.088_302_46 * self.r + 0.281_718_85 * self.g + 0.629_978_7 * self.b;
        let (l, m, s) = (l.cbrt(), m.cbrt(), s.cbrt());
        [
            0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
            1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
            0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
        ]
    }

    /// The inverse of [`to_oklab`](Self::to_oklab). Can return components
    /// outside `0..1` for colours no display can show; clamp if that matters.
    pub fn from_oklab(lab: [f32; 3]) -> Self {
        let l_ = lab[0] + 0.396_337_78 * lab[1] + 0.215_803_76 * lab[2];
        let m_ = lab[0] - 0.105_561_346 * lab[1] - 0.063_854_17 * lab[2];
        let s_ = lab[0] - 0.089_484_18 * lab[1] - 1.291_485_5 * lab[2];
        let (l, m, s) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
        Self {
            r: 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
            g: -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
            b: -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
        }
    }

    /// Relative luminance, Rec. 709, from the linear components.
    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Scale every component. Brightening *light*, not lightening a paint —
    /// for the latter, move `L` in [`to_oklab`](Self::to_oklab).
    pub fn scaled(self, k: f32) -> Self {
        Self {
            r: self.r * k,
            g: self.g * k,
            b: self.b * k,
        }
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}
