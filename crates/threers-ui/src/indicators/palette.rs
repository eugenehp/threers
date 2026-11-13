//! Palettes: three colours an indicator runs through, and ways to make them.

use super::*;

/// Three colours an indicator runs through from its start to its end
/// (sRGB, 0 to 1, as in CSS).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Palette(pub [Rgba; 3]);

pub(super) const fn hex(c: u32) -> Rgba {
    [((c >> 16) & 255) as f32 / 255.0, ((c >> 8) & 255) as f32 / 255.0, (c & 255) as f32 / 255.0, 1.0]
}

/// `a` mixed `t` of the way to `b`.
pub(super) fn mix(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let t = t.clamp(0.0, 1.0);
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// A colour from its hue (degrees), saturation and lightness (0 to 1).
pub(super) fn hsl(h: f32, s: f32, l: f32) -> Rgba {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s.clamp(0.0, 1.0);
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    [r + m, g + m, b + m, 1.0]
}

impl Palette {
    /// Green through cyan to violet.
    pub const AURORA: Palette = Palette([hex(0x22ff9a), hex(0x22d3ee), hex(0xa855f7)]);
    /// Deep red through orange to yellow.
    pub const EMBER: Palette = Palette([hex(0xff2d55), hex(0xff8a00), hex(0xffe14d)]);
    /// Navy blue through azure to aqua.
    pub const OCEAN: Palette = Palette([hex(0x2b5cff), hex(0x00a8ff), hex(0x5ef2ff)]);
    /// Hot pink through violet to electric blue.
    pub const NEON: Palette = Palette([hex(0xff2bd6), hex(0x8b5cf6), hex(0x00e5ff)]);
    /// Lime through gold to coral.
    pub const CITRUS: Palette = Palette([hex(0x9dff3a), hex(0xffd21f), hex(0xff6b6b)]);

    /// Rose through magenta to amber.
    pub const SUNSET: Palette = Palette([hex(0xff4d8d), hex(0xd946ef), hex(0xffb020)]);
    /// Sea green through mint to pale lime.
    pub const MINT: Palette = Palette([hex(0x10b981), hex(0x5eead4), hex(0xd9f99d)]);
    /// Steel blue through ice to white.
    pub const ICE: Palette = Palette([hex(0x60a5fa), hex(0xa5f3fc), hex(0xf0f9ff)]);
    /// Bronze through gold to cream.
    pub const GOLD: Palette = Palette([hex(0xb45309), hex(0xfbbf24), hex(0xfff1b8)]);
    /// Fire: deep red embers, orange, yellow flame.
    pub const FIRE: Palette = Palette([hex(0x7f1d1d), hex(0xf97316), hex(0xfde047)]);
    /// Red, amber, green: a battery's charge, low to full.
    pub const BATTERY: Palette = Palette([hex(0xef4444), hex(0xf59e0b), hex(0x22c55e)]);
    /// Grey to white.
    pub const MONO: Palette = Palette([hex(0x6b7280), hex(0xc7ccd6), hex(0xffffff)]);

    /// Every named palette, with its name.
    pub const NAMED: [(&'static str, Palette); 11] = [
        ("Aurora", Palette::AURORA),
        ("Ember", Palette::EMBER),
        ("Ocean", Palette::OCEAN),
        ("Neon", Palette::NEON),
        ("Citrus", Palette::CITRUS),
        ("Sunset", Palette::SUNSET),
        ("Mint", Palette::MINT),
        ("Ice", Palette::ICE),
        ("Gold", Palette::GOLD),
        ("Mono", Palette::MONO),
        ("Rainbow", Palette::RAINBOW),
    ];

    /// Red through green to blue: the hues, as far as three stops go.
    pub const RAINBOW: Palette = Palette([hex(0xff3b3b), hex(0x3bff7a), hex(0x3b6bff)]);

    /// The palette called `name` (any case), of [`NAMED`](Self::NAMED).
    pub fn named(name: &str) -> Option<Palette> {
        Palette::NAMED.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|&(_, p)| p)
    }

    /// One colour, end to end.
    pub fn solid(c: Rgba) -> Palette {
        Palette([c, c, c])
    }

    /// From one colour to another, through their mix.
    pub fn two(from: Rgba, to: Rgba) -> Palette {
        Palette([from, std::array::from_fn(|i| (from[i] + to[i]) * 0.5), to])
    }

    /// From three `0xRRGGBB` colours.
    pub const fn hex(from: u32, mid: u32, to: u32) -> Palette {
        Palette([hex(from), hex(mid), hex(to)])
    }

    /// From CSS colours (`#22d3ee`, `rgb(…)`, `hsl(…)`, a name): one for
    /// a single colour, two for a gradient between them, three for one
    /// through all three. `None` if one does not parse, or there are none
    /// or more than three.
    pub fn css(colours: &[&str]) -> Option<Palette> {
        let c: Vec<Rgba> = colours.iter().map(|c| crate::css::color(c)).collect::<Option<_>>()?;
        match c.as_slice() {
            [a] => Some(Palette::solid(*a)),
            [a, b] => Some(Palette::two(*a, *b)),
            [a, b, c] => Some(Palette([*a, *b, *c])),
            _ => None,
        }
    }

    /// Any number of gradient stops `(position 0 to 1, colour)`, sampled
    /// at the start, the middle and the end (the shaders run through three).
    pub fn stops(stops: &[(f32, Rgba)]) -> Option<Palette> {
        if stops.is_empty() {
            return None;
        }
        let mut s = stops.to_vec();
        s.sort_by(|a, b| a.0.total_cmp(&b.0));
        Some(Palette([0.0, 0.5, 1.0].map(|t| crate::gradient_at(&s, t))))
    }

    /// Round the colour wheel from `from` to `to` degrees, at `saturation`
    /// and `lightness` (0 to 1).
    pub fn hues(from: f32, to: f32, saturation: f32, lightness: f32) -> Palette {
        Palette([0.0, 0.5, 1.0].map(|t| hsl(from + (to - from) * t, saturation, lightness)))
    }

    /// Shades of one colour: darker at the start, lighter at the end.
    pub fn shades(c: Rgba) -> Palette {
        Palette([mix(c, [0.0, 0.0, 0.0, c[3]], 0.45), c, mix(c, [1.0, 1.0, 1.0, c[3]], 0.55)])
    }

    /// End to start.
    pub fn reversed(self) -> Palette {
        let [a, b, c] = self.0;
        Palette([c, b, a])
    }

    /// Each colour mixed `amount` (0 to 1) of the way to white.
    pub fn lighter(self, amount: f32) -> Palette {
        Palette(self.0.map(|c| mix(c, [1.0, 1.0, 1.0, c[3]], amount)))
    }

    /// Each colour mixed `amount` (0 to 1) of the way to black.
    pub fn darker(self, amount: f32) -> Palette {
        Palette(self.0.map(|c| mix(c, [0.0, 0.0, 0.0, c[3]], amount)))
    }

    /// The middle colour replaced.
    pub fn through(self, mid: Rgba) -> Palette {
        Palette([self.0[0], mid, self.0[2]])
    }

    /// The colour `t` (0 to 1) of the way along, mixed as encoded (as CSS
    /// gradients mix).
    pub fn at(&self, t: f32) -> Rgba {
        let t = t.clamp(0.0, 1.0);
        let [a, b, c] = self.0;
        let (from, to, f) = if t < 0.5 { (a, b, t * 2.0) } else { (b, c, t * 2.0 - 1.0) };
        std::array::from_fn(|i| from[i] + (to[i] - from[i]) * f)
    }
}

impl Default for Palette {
    fn default() -> Self {
        Palette::AURORA
    }
}
