//! The scales classes are read with: [`Tailwind`], its defaults and its
//! builder.

use std::collections::HashMap;

use crate::style::Rgba;

/// A colour, and, if it differs there, its value in the dark appearance:
/// a theme's `primary` that is near black in the light and near white in
/// the dark. A class using one takes the dark value with the appearance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Paint {
    pub light: Rgba,
    pub dark: Option<Rgba>,
}

impl Paint {
    pub fn fixed(c: Rgba) -> Self {
        Self {
            light: c,
            dark: None,
        }
    }

    pub fn adaptive(light: Rgba, dark: Rgba) -> Self {
        Self {
            light,
            dark: Some(dark),
        }
    }

    /// In `dark` or the light appearance.
    pub fn in_dark(&self, dark: bool) -> Rgba {
        if dark {
            self.dark.unwrap_or(self.light)
        } else {
            self.light
        }
    }

    /// Whether it changes with the appearance.
    pub fn is_adaptive(&self) -> bool {
        self.dark.is_some_and(|d| d != self.light)
    }

    /// With its opacity times `a` (`bg-primary/50`).
    pub fn faded(self, a: f32) -> Self {
        let f = |mut c: Rgba| {
            c[3] *= a;
            c
        };
        Self {
            light: f(self.light),
            dark: self.dark.map(f),
        }
    }

    /// With no opacity: where a gradient fades to.
    pub fn transparent(self) -> Self {
        self.faded(0.0)
    }
}

/// The scales classes are read with: Tailwind's defaults
/// ([`Tailwind::default`]), shadcn/ui's look ([`Tailwind::shadcn`]), or
/// your own, built with the methods below or read from JSON
/// ([`extend_json`](Self::extend_json)). Set one as the UI's global for
/// [`Ui::tw`](crate::Ui::tw) to use:
///
/// ```
/// use threers_ui::tailwind::Tailwind;
/// use threers_ui::Ui;
///
/// let mut ui = Ui::new();
/// ui.set_global(
///     Tailwind::default()
///         .color("brand", "#ff6600")
///         .adaptive_color("surface", "#ffffff", "#18181b")
///         .spacing("18", 72.0)
///         .screen("tablet", 900.0)
///         .component("btn", "px-4 py-2 rounded-md bg-brand hover:bg-brand/90 text-white"),
/// );
/// let root = ui.root();
/// let b = ui.add_tw(root, "btn w-18");
/// assert_eq!(ui.style(b).background, Some([1.0, 0.4, 0.0, 1.0]));
/// assert_eq!(ui.style(b).width, threers_ui::Length::Px(72.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Tailwind {
    /// Pixels in a rem, for values given in rems.
    pub(super) rem: f32,
    /// Pixels per spacing step: `p-1` (4).
    pub(super) spacing_unit: f32,
    pub(super) spacing: HashMap<String, f32>,
    /// `red-500`, `black`, `primary`.
    pub(super) colors: HashMap<String, Paint>,
    /// Ascending by width.
    pub(super) screens: Vec<(String, f32)>,
    /// Size and line height (CSS text: `20px`, or a bare multiple).
    pub(super) font_sizes: HashMap<String, (f32, Option<String>)>,
    pub(super) font_weights: HashMap<String, u16>,
    pub(super) font_families: HashMap<String, String>,
    pub(super) line_heights: HashMap<String, String>,
    /// `rounded` is the key `""`, as `DEFAULT` is in Tailwind; so too for
    /// shadows and blurs.
    pub(super) radii: HashMap<String, f32>,
    pub(super) shadows: HashMap<String, String>,
    pub(super) blurs: HashMap<String, f32>,
    pub(super) max_widths: HashMap<String, f32>,
    /// `perspective-<name>`: the viewer's distance.
    pub(super) perspectives: HashMap<String, f32>,
    /// Classes made of others.
    pub(super) components: HashMap<String, String>,
    /// What a border width with no colour of its own is drawn in.
    pub(super) border_color: Paint,
    /// `ring`'s width and what a ring with no colour of its own is drawn in.
    pub(super) ring_width: f32,
    pub(super) ring_color: Paint,
    /// What `transition` takes with no `duration-*`, seconds.
    pub(super) duration: f32,
    /// Boxes measure with `border-box`, as Tailwind's preflight has them.
    pub(super) preflight: bool,
}

/// Tailwind 3.4's palette, `50` to `950`.
const SHADES: [&str; 11] = [
    "50", "100", "200", "300", "400", "500", "600", "700", "800", "900", "950",
];
#[rustfmt::skip]
const PALETTE: [(&str, [u32; 11]); 22] = [
    ("slate", [0xf8fafc, 0xf1f5f9, 0xe2e8f0, 0xcbd5e1, 0x94a3b8, 0x64748b, 0x475569, 0x334155, 0x1e293b, 0x0f172a, 0x020617]),
    ("gray", [0xf9fafb, 0xf3f4f6, 0xe5e7eb, 0xd1d5db, 0x9ca3af, 0x6b7280, 0x4b5563, 0x374151, 0x1f2937, 0x111827, 0x030712]),
    ("zinc", [0xfafafa, 0xf4f4f5, 0xe4e4e7, 0xd4d4d8, 0xa1a1aa, 0x71717a, 0x52525b, 0x3f3f46, 0x27272a, 0x18181b, 0x09090b]),
    ("neutral", [0xfafafa, 0xf5f5f5, 0xe5e5e5, 0xd4d4d4, 0xa3a3a3, 0x737373, 0x525252, 0x404040, 0x262626, 0x171717, 0x0a0a0a]),
    ("stone", [0xfafaf9, 0xf5f5f4, 0xe7e5e4, 0xd6d3d1, 0xa8a29e, 0x78716c, 0x57534e, 0x44403c, 0x292524, 0x1c1917, 0x0c0a09]),
    ("red", [0xfef2f2, 0xfee2e2, 0xfecaca, 0xfca5a5, 0xf87171, 0xef4444, 0xdc2626, 0xb91c1c, 0x991b1b, 0x7f1d1d, 0x450a0a]),
    ("orange", [0xfff7ed, 0xffedd5, 0xfed7aa, 0xfdba74, 0xfb923c, 0xf97316, 0xea580c, 0xc2410c, 0x9a3412, 0x7c2d12, 0x431407]),
    ("amber", [0xfffbeb, 0xfef3c7, 0xfde68a, 0xfcd34d, 0xfbbf24, 0xf59e0b, 0xd97706, 0xb45309, 0x92400e, 0x78350f, 0x451a03]),
    ("yellow", [0xfefce8, 0xfef9c3, 0xfef08a, 0xfde047, 0xfacc15, 0xeab308, 0xca8a04, 0xa16207, 0x854d0e, 0x713f12, 0x422006]),
    ("lime", [0xf7fee7, 0xecfccb, 0xd9f99d, 0xbef264, 0xa3e635, 0x84cc16, 0x65a30d, 0x4d7c0f, 0x3f6212, 0x365314, 0x1a2e05]),
    ("green", [0xf0fdf4, 0xdcfce7, 0xbbf7d0, 0x86efac, 0x4ade80, 0x22c55e, 0x16a34a, 0x15803d, 0x166534, 0x14532d, 0x052e16]),
    ("emerald", [0xecfdf5, 0xd1fae5, 0xa7f3d0, 0x6ee7b7, 0x34d399, 0x10b981, 0x059669, 0x047857, 0x065f46, 0x064e3b, 0x022c22]),
    ("teal", [0xf0fdfa, 0xccfbf1, 0x99f6e4, 0x5eead4, 0x2dd4bf, 0x14b8a6, 0x0d9488, 0x0f766e, 0x115e59, 0x134e4a, 0x042f2e]),
    ("cyan", [0xecfeff, 0xcffafe, 0xa5f3fc, 0x67e8f9, 0x22d3ee, 0x06b6d4, 0x0891b2, 0x0e7490, 0x155e75, 0x164e63, 0x083344]),
    ("sky", [0xf0f9ff, 0xe0f2fe, 0xbae6fd, 0x7dd3fc, 0x38bdf8, 0x0ea5e9, 0x0284c7, 0x0369a1, 0x075985, 0x0c4a6e, 0x082f49]),
    ("blue", [0xeff6ff, 0xdbeafe, 0xbfdbfe, 0x93c5fd, 0x60a5fa, 0x3b82f6, 0x2563eb, 0x1d4ed8, 0x1e40af, 0x1e3a8a, 0x172554]),
    ("indigo", [0xeef2ff, 0xe0e7ff, 0xc7d2fe, 0xa5b4fc, 0x818cf8, 0x6366f1, 0x4f46e5, 0x4338ca, 0x3730a3, 0x312e81, 0x1e1b4b]),
    ("violet", [0xf5f3ff, 0xede9fe, 0xddd6fe, 0xc4b5fd, 0xa78bfa, 0x8b5cf6, 0x7c3aed, 0x6d28d9, 0x5b21b6, 0x4c1d95, 0x2e1065]),
    ("purple", [0xfaf5ff, 0xf3e8ff, 0xe9d5ff, 0xd8b4fe, 0xc084fc, 0xa855f7, 0x9333ea, 0x7e22ce, 0x6b21a8, 0x581c87, 0x3b0764]),
    ("fuchsia", [0xfdf4ff, 0xfae8ff, 0xf5d0fe, 0xf0abfc, 0xe879f9, 0xd946ef, 0xc026d3, 0xa21caf, 0x86198f, 0x701a75, 0x4a044e]),
    ("pink", [0xfdf2f8, 0xfce7f3, 0xfbcfe8, 0xf9a8d4, 0xf472b6, 0xec4899, 0xdb2777, 0xbe185d, 0x9d174d, 0x831843, 0x500724]),
    ("rose", [0xfff1f2, 0xffe4e6, 0xfecdd3, 0xfda4af, 0xfb7185, 0xf43f5e, 0xe11d48, 0xbe123c, 0x9f1239, 0x881337, 0x4c0519]),
];

pub(super) fn rgb(hex: u32) -> Rgba {
    let c = |shift: u32| ((hex >> shift) & 0xff) as f32 / 255.0;
    [c(16), c(8), c(0), 1.0]
}

fn table<V: Clone>(entries: &[(&str, V)]) -> HashMap<String, V> {
    entries
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect()
}

/// A line height in config text: rems to pixels, a bare multiple kept.
pub(super) fn rem_text(v: &str, rem: f32) -> String {
    match v.strip_suffix("rem") {
        Some(r) => format!("{}px", r.parse::<f32>().unwrap_or(0.0) * rem),
        None => v.to_owned(),
    }
}

#[track_caller]
pub(super) fn parse_color(css: &str) -> Rgba {
    crate::css::color(css).unwrap_or_else(|| panic!("tailwind: `{css}` is not a colour"))
}

impl Default for Tailwind {
    /// Tailwind 3.4's default theme, with Tailwind 4's `shadow-xs`,
    /// `shadow-2xs` and `perspective-*`.
    fn default() -> Self {
        let mut colors = HashMap::new();
        for (name, shades) in PALETTE {
            for (shade, hex) in SHADES.iter().zip(shades) {
                colors.insert(format!("{name}-{shade}"), Paint::fixed(rgb(hex)));
            }
        }
        colors.insert("black".into(), Paint::fixed([0.0, 0.0, 0.0, 1.0]));
        colors.insert("white".into(), Paint::fixed([1.0; 4]));
        colors.insert("transparent".into(), Paint::fixed([0.0; 4]));
        let rem = 16.0;
        #[rustfmt::skip]
        let font_sizes = [
            ("xs", 0.75, "1rem"), ("sm", 0.875, "1.25rem"), ("base", 1.0, "1.5rem"), ("lg", 1.125, "1.75rem"),
            ("xl", 1.25, "1.75rem"), ("2xl", 1.5, "2rem"), ("3xl", 1.875, "2.25rem"), ("4xl", 2.25, "2.5rem"),
            ("5xl", 3.0, "1"), ("6xl", 3.75, "1"), ("7xl", 4.5, "1"), ("8xl", 6.0, "1"), ("9xl", 8.0, "1"),
        ]
        .into_iter()
        .map(|(k, size, lh)| (k.to_owned(), (size * rem, Some(rem_text(lh, rem)))))
        .collect();
        #[rustfmt::skip]
        let line_heights = [
            ("3", ".75rem"), ("4", "1rem"), ("5", "1.25rem"), ("6", "1.5rem"), ("7", "1.75rem"), ("8", "2rem"),
            ("9", "2.25rem"), ("10", "2.5rem"), ("none", "1"), ("tight", "1.25"), ("snug", "1.375"),
            ("normal", "1.5"), ("relaxed", "1.625"), ("loose", "2"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), rem_text(v, rem)))
        .collect();
        #[rustfmt::skip]
        let max_widths = [
            ("0", 0.0), ("xs", 20.0), ("sm", 24.0), ("md", 28.0), ("lg", 32.0), ("xl", 36.0), ("2xl", 42.0),
            ("3xl", 48.0), ("4xl", 56.0), ("5xl", 64.0), ("6xl", 72.0), ("7xl", 80.0),
        ]
        .into_iter()
        .map(|(k, r)| (k.to_owned(), r * rem))
        .collect();
        let s = |v: &str| v.to_owned();
        Self {
            rem,
            spacing_unit: 4.0,
            spacing: table(&[("px", 1.0)]),
            colors,
            screens: [
                ("sm", 640.0),
                ("md", 768.0),
                ("lg", 1024.0),
                ("xl", 1280.0),
                ("2xl", 1536.0),
            ]
            .into_iter()
            .map(|(k, w)| (k.to_owned(), w))
            .collect(),
            font_sizes,
            #[rustfmt::skip]
            font_weights: table(&[
                ("thin", 100), ("extralight", 200), ("light", 300), ("normal", 400), ("medium", 500),
                ("semibold", 600), ("bold", 700), ("extrabold", 800), ("black", 900),
            ]),
            font_families: table(&[
                ("sans", s("sans-serif")),
                ("serif", s("serif")),
                ("mono", s("monospace")),
            ]),
            line_heights,
            #[rustfmt::skip]
            radii: table(&[
                ("none", 0.0), ("sm", 2.0), ("", 4.0), ("md", 6.0), ("lg", 8.0), ("xl", 12.0), ("2xl", 16.0),
                ("3xl", 24.0), ("full", 9999.0),
            ]),
            shadows: table(&[
                ("2xs", s("0 1px rgba(0, 0, 0, 0.05)")),
                ("xs", s("0 1px 2px 0 rgba(0, 0, 0, 0.05)")),
                ("sm", s("0 1px 2px 0 rgba(0, 0, 0, 0.05)")),
                (
                    "",
                    s("0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)"),
                ),
                (
                    "md",
                    s("0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)"),
                ),
                (
                    "lg",
                    s("0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)"),
                ),
                (
                    "xl",
                    s("0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)"),
                ),
                ("2xl", s("0 25px 50px -12px rgba(0, 0, 0, 0.25)")),
                ("none", s("none")),
            ]),
            #[rustfmt::skip]
            blurs: table(&[
                ("0", 0.0), ("none", 0.0), ("sm", 4.0), ("", 8.0), ("md", 12.0), ("lg", 16.0), ("xl", 24.0),
                ("2xl", 40.0), ("3xl", 64.0),
            ]),
            max_widths,
            perspectives: table(&[
                ("dramatic", 100.0),
                ("near", 300.0),
                ("normal", 500.0),
                ("midrange", 800.0),
                ("distant", 1200.0),
            ]),
            components: HashMap::new(),
            border_color: Paint::fixed(rgb(0xe5e7eb)),
            ring_width: 3.0,
            ring_color: Paint::fixed([59.0 / 255.0, 130.0 / 255.0, 246.0 / 255.0, 0.5]),
            duration: 0.15,
            preflight: true,
        }
    }
}

impl Tailwind {
    /// Adds or replaces a colour, by name as classes use it after
    /// `bg-`, `text-`, …: `"brand"`, `"brand-500"`. The value is CSS.
    #[track_caller]
    pub fn color(mut self, name: &str, css: &str) -> Self {
        self.colors
            .insert(name.to_owned(), Paint::fixed(parse_color(css)));
        self
    }

    /// A colour with a value of its own in the dark appearance (a
    /// theme's `background`, `primary`, …): a node using it changes as the
    /// appearance does.
    #[track_caller]
    pub fn adaptive_color(mut self, name: &str, light: &str, dark: &str) -> Self {
        self.colors.insert(
            name.to_owned(),
            Paint::adaptive(parse_color(light), parse_color(dark)),
        );
        self
    }

    /// A colour's shades: `("brand", &[("50", "#fff7ed"), …])` names
    /// `brand-50`, …; the shade `"DEFAULT"` names `brand` itself.
    #[track_caller]
    pub fn colors(mut self, name: &str, shades: &[(&str, &str)]) -> Self {
        for (shade, css) in shades {
            let key = if *shade == "DEFAULT" {
                name.to_owned()
            } else {
                format!("{name}-{shade}")
            };
            self.colors.insert(key, Paint::fixed(parse_color(css)));
        }
        self
    }

    /// Takes every colour out, the palette included, for a palette
    /// entirely of your own.
    pub fn clear_colors(mut self) -> Self {
        self.colors.clear();
        self
    }

    /// Pixels per spacing step (4: `p-4` is 16px).
    pub fn spacing_unit(mut self, px: f32) -> Self {
        self.spacing_unit = px;
        self
    }

    /// A named spacing value: `p-<name>`, `w-<name>`, `gap-<name>`, …
    pub fn spacing(mut self, name: &str, px: f32) -> Self {
        self.spacing.insert(name.to_owned(), px);
        self
    }

    /// A breakpoint: `<name>:` for windows `min_width` wide and wider,
    /// `max-<name>:` for narrower ones.
    pub fn screen(mut self, name: &str, min_width: f32) -> Self {
        self.screens.retain(|(n, _)| n != name);
        self.screens.push((name.to_owned(), min_width));
        self.screens.sort_by(|a, b| a.1.total_cmp(&b.1));
        self
    }

    /// Takes the breakpoints out, for a set entirely of your own.
    pub fn clear_screens(mut self) -> Self {
        self.screens.clear();
        self
    }

    /// `text-<name>`: a font size and, if given, its line height in pixels.
    pub fn font_size(mut self, name: &str, px: f32, line_height: Option<f32>) -> Self {
        self.font_sizes
            .insert(name.to_owned(), (px, line_height.map(|l| format!("{l}px"))));
        self
    }

    /// `font-<name>`: a family.
    pub fn font_family(mut self, name: &str, family: &str) -> Self {
        self.font_families
            .insert(name.to_owned(), family.to_owned());
        self
    }

    /// `rounded-<name>` (`""` for `rounded`).
    pub fn radius(mut self, name: &str, px: f32) -> Self {
        self.radii.insert(name.to_owned(), px);
        self
    }

    /// `shadow-<name>` (`""` for `shadow`): a CSS `box-shadow`.
    pub fn shadow(mut self, name: &str, css: &str) -> Self {
        self.shadows.insert(name.to_owned(), css.to_owned());
        self
    }

    /// `blur-<name>` and `backdrop-blur-<name>` (`""` for `blur`).
    pub fn blur(mut self, name: &str, px: f32) -> Self {
        self.blurs.insert(name.to_owned(), px);
        self
    }

    /// `perspective-<name>`: the viewer's distance in pixels.
    pub fn perspective(mut self, name: &str, px: f32) -> Self {
        self.perspectives.insert(name.to_owned(), px);
        self
    }

    /// A class standing for others, as CSS's `@apply` makes one:
    /// `component("btn", "px-4 py-2 rounded hover:bg-blue-600")`. A variant
    /// on it (`md:btn`) goes onto each of them. Components may use others.
    pub fn component(mut self, name: &str, classes: &str) -> Self {
        self.components.insert(name.to_owned(), classes.to_owned());
        self
    }

    /// The colour a border width with no colour class is drawn in
    /// (`gray-200`): a colour name of the theme or CSS.
    #[track_caller]
    pub fn default_border_color(mut self, color: &str) -> Self {
        self.border_color = self
            .colors
            .get(color)
            .copied()
            .unwrap_or_else(|| Paint::fixed(parse_color(color)));
        self
    }

    /// `ring`'s width (3) and the colour of a ring with no colour class
    /// (`blue-500` at half opacity): a colour name of the theme or CSS.
    #[track_caller]
    pub fn default_ring(mut self, width: f32, color: &str) -> Self {
        self.ring_width = width.max(0.0);
        self.ring_color = self
            .colors
            .get(color)
            .copied()
            .unwrap_or_else(|| Paint::fixed(parse_color(color)));
        self
    }

    /// What `transition` takes without a `duration-*`, in seconds (0.15).
    pub fn default_duration(mut self, seconds: f32) -> Self {
        self.duration = seconds.max(0.0);
        self
    }

    /// Whether boxes given classes measure with `border-box` (yes, as
    /// Tailwind's preflight makes every element: `h-9 py-2` is 36 pixels
    /// tall in all).
    pub fn preflight(mut self, on: bool) -> Self {
        self.preflight = on;
        self
    }

    /// The colour names, for suggestions.
    pub(super) fn color_names(&self) -> impl Iterator<Item = &str> {
        self.colors.keys().map(String::as_str)
    }
}
