//! The Shaders page: cards drawn through custom shaders, each a few lines
//! of WGSL written inline with `ui_shader!` — a ripple, holographic foil, a
//! lens, a peeling page, a curved screen, frosted glass, a colour split,
//! pixels, a burn, a swirl, halftone dots, a kaleidoscope, neon edges, a
//! flag and a glitch — some following the pointer, some the time, some a
//! slider.

use threers_ui::environment::Token;
use threers_ui::input::Input;
use threers_ui::pickers::slider;
use threers_ui::shader::Effect;
use threers_ui::{style, ui_shader, NodeId, Role, Semantics, Ui};

use crate::app::State;
use crate::showcase::group;

/// A card for an effect: a gradient, a title and a line under it.
fn card(ui: &mut Ui, parent: NodeId, title: &str, about: &str, k: usize) -> NodeId {
    let c = ui.add_box(parent, style! { display: flex; flex-direction: column; justify-content: flex-end; gap: 2px; width: 180px; height: 120px; padding: 12px; border-radius: 12px; flex-shrink: 0; box-sizing: border-box; color: #fcfcfc; });
    let stops: [([f32; 3], [f32; 3]); 4] = [
        ([0.11, 0.27, 0.75], [0.4, 0.13, 0.66]),
        ([0.03, 0.37, 0.31], [0.11, 0.27, 0.75]),
        ([0.66, 0.2, 0.05], [0.6, 0.09, 0.3]),
        ([0.2, 0.22, 0.3], [0.45, 0.48, 0.6]),
    ];
    let (a, b) = stops[k % stops.len()];
    ui.update_style(c, |s| {
        s.background = Some([a[0], a[1], a[2], 1.0]);
        s.background_gradient = vec![(0.0, [a[0], a[1], a[2], 1.0]), (1.0, [b[0], b[1], b[2], 1.0])];
    });
    let big = ui.add_box(c, style! { font-size: 30px; font-weight: bold; });
    ui.add_text(big, "threers");
    let t = ui.add_box(c, style! { font-size: 14px; font-weight: bold; });
    ui.add_text(t, title);
    let a = ui.add_box(c, style! { font-size: 11px; });
    ui.add_text(a, about);
    ui.set_semantics(c, Semantics::new(Role::Group).label(title));
    c
}

/// What the page's sliders turn, to set on the cards when they move.
pub struct Shaders {
    chromatic: NodeId,
    pixels: NodeId,
    burn: NodeId,
    last: (f32, f32, f32),
}

impl Shaders {
    /// The sliders' values onto the cards' effects, when they changed.
    pub fn update(&mut self, ui: &mut Ui, s: &State) {
        let now = (s.shader_split, s.shader_cell, s.shader_burn);
        if now == self.last {
            return;
        }
        self.last = now;
        ui.set_effect_arg(self.chromatic, "amount", now.0);
        ui.set_effect_arg(self.pixels, "cell", now.1);
        ui.set_effect_arg(self.burn, "progress", now.2);
    }
}

/// The source the page shows, as it is written in this file.
const RIPPLE_SOURCE: &str = "let ripple = ui_shader! {
    fn ripple(p: vec2<f32>, time: f32, centre: vec2<f32>, strength: f32) -> vec4<f32> {
        let d = p - centre;
        let r = length(d);
        let wave = sin(r * 0.15 - time * 6.0) * strength * exp(-r * 0.012);
        let c = layer(p + d / max(r, 0.001) * wave);
        return vec4<f32>(c.rgb * (1.0 + wave * 0.05), c.a);
    }
};
ui.set_effect(card, Some(Effect::new(&ripple).arg(\"strength\", 5.0)));
input.effect_follows_pointer(card, \"centre\");";

pub fn build(ui: &mut Ui, input: &mut Input<State>, state: &State, page: NodeId) -> Shaders {
    let g = group(ui, page, "Effects", "Each card is drawn into a layer, then through its shader. Move the pointer over them; the sliders turn Split, Pixels and Burn.");
    let note = ui.add_box(g, style! { font-size: 12px; white-space: normal; });
    ui.token_style(note, &[("color", Token::SecondaryText)]);
    ui.add_text(note, "Shaders run on the GPU renderer (WebGPU or WebGL2); the 2D canvas draws the cards as they are.");
    let grid = ui.add_box(g, style! { display: flex; flex-wrap: wrap; gap: 14px; });

    // Rings running out from the pointer.
    let ripple = ui_shader! {
        fn ripple(p: vec2<f32>, time: f32, centre: vec2<f32>, strength: f32) -> vec4<f32> {
            let d = p - centre;
            let r = length(d);
            let wave = sin(r * 0.15 - time * 6.0) * strength * exp(-r * 0.012);
            let c = layer(p + d / max(r, 0.001) * wave);
            return vec4<f32>(c.rgb * (1.0 + wave * 0.05), c.a);
        }
    };
    let c = card(ui, grid, "Ripple", "Rings from the pointer", 0);
    ui.set_effect(c, Some(Effect::new(&ripple).arg("strength", 5.0).arg("centre", [90.0, 60.0])));
    input.effect_follows_pointer(c, "centre");

    // Holographic foil: a rainbow sheen where the light catches the tilted
    // card.
    let foil = ui_shader! {
        fn foil(p: vec2<f32>, color: vec4<f32>, size: vec2<f32>, time: f32, light: vec2<f32>) -> vec4<f32> {
            let uv = p / size;
            let l = light / size;
            let band = dot(uv - l, vec2<f32>(0.7, 0.7));
            let hue = fract(band * 1.5 + time * 0.05);
            let rainbow = vec3<f32>(0.5) + 0.5 * cos(6.2831 * (vec3<f32>(hue) + vec3<f32>(0.0, 0.33, 0.67)));
            let shine = exp(-pow(length(uv - l) * 2.2, 2.0));
            let rgb = mix(color.rgb, rainbow * color.a, 0.3 + 0.3 * shine) + vec3<f32>(shine * 0.3 * color.a);
            return vec4<f32>(rgb, color.a);
        }
    };
    let c = card(ui, grid, "Foil", "Tilts, and catches the light", 3);
    ui.set_effect(c, Some(Effect::new(&foil).arg("light", [40.0, 30.0])));
    input.effect_follows_pointer(c, "light");
    input.tilt_on_hover(ui, c, 14.0);

    // A lens over the card, at the pointer: what is under it bulges.
    let lens = ui_shader! {
        fn lens(p: vec2<f32>, centre: vec2<f32>, radius: f32) -> vec4<f32> {
            let d = p - centre;
            let r = length(d) / radius;
            if (r >= 1.0) {
                return layer(p);
            }
            let c = layer(centre + d * (0.5 + 0.5 * r * r));
            let rim = smoothstep(0.82, 1.0, r);
            let lit = vec3<f32>(0.18 * (1.0 - r) * c.a);
            return vec4<f32>(c.rgb * (1.0 - 0.35 * rim) + lit, c.a);
        }
    };
    let c = card(ui, grid, "Lens", "Magnifies under the pointer", 1);
    ui.set_effect(c, Some(Effect::new(&lens).arg("radius", 38.0).arg("centre", [60.0, 50.0])));
    input.effect_follows_pointer(c, "centre");

    // A page peeled back from its corner to the pointer: the fold is
    // halfway between them, what is past it is lifted over, its back shows
    // as paper, and it shades the page under it.
    let peel = ui_shader! {
        fn peel(p: vec2<f32>, size: vec2<f32>, grab: vec2<f32>) -> vec4<f32> {
            let g = clamp(grab, vec2<f32>(1.0), size - vec2<f32>(1.0));
            let n = normalize(size - g);
            let mid = (size + g) * 0.5;
            let d = dot(p - mid, n);
            var base = vec4<f32>(0.0);
            if (d <= 0.0) {
                let c = layer(p);
                base = vec4<f32>(c.rgb * (1.0 - 0.4 * exp(d / 8.0)), c.a);
            }
            let q = p - 2.0 * d * n;
            if (d < 0.0 && all(q >= vec2<f32>(0.0)) && all(q <= size)) {
                let back = layer(q);
                let light = 0.7 + 0.3 * clamp(-d / 30.0, 0.0, 1.0);
                let paper = mix(vec3<f32>(0.94, 0.93, 0.9), back.rgb / max(back.a, 0.001), 0.15) * light;
                let flap = vec4<f32>(paper * back.a, back.a);
                return flap + base * (1.0 - flap.a);
            }
            return base;
        }
    };
    let c = card(ui, grid, "Peel", "Lifts from the corner to the pointer", 1);
    ui.set_effect(c, Some(Effect::new(&peel).arg("grab", [150.0, 96.0])));
    input.effect_follows_pointer(c, "grab");

    // A curved screen: barrel-bent, scanlines rolling, a green glow.
    let screen = ui_shader! {
        fn screen(p: vec2<f32>, size: vec2<f32>, time: f32) -> vec4<f32> {
            var uv = p / size * 2.0 - vec2<f32>(1.0);
            uv = uv * (1.0 + 0.1 * dot(uv, uv));
            if (any(abs(uv) > vec2<f32>(1.0))) {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            let q = (uv * 0.5 + vec2<f32>(0.5)) * size;
            let c = layer(q);
            let scan = 0.7 + 0.3 * sin(q.y * 2.2 - time * 10.0);
            let edge = 1.0 - 0.4 * dot(uv * uv, uv * uv);
            let mono = vec3<f32>(dot(c.rgb, vec3<f32>(0.3, 0.55, 0.15))) * vec3<f32>(0.35, 1.0, 0.55) * 1.5;
            return vec4<f32>((mono * scan + vec3<f32>(0.02, 0.05, 0.03)) * edge, 1.0);
        }
    };
    let c = card(ui, grid, "Screen", "A curved, scanning tube", 0);
    ui.set_effect(c, Some(Effect::new(&screen)));

    // Frosted glass: the layer averaged over a disc.
    let frost = ui_shader! {
        fn frost(p: vec2<f32>, size: vec2<f32>, radius: f32) -> vec4<f32> {
            var sum = vec4<f32>(0.0);
            for (var i = -3; i <= 3; i = i + 1) {
                for (var j = -3; j <= 3; j = j + 1) {
                    sum = sum + layer(p + vec2<f32>(f32(i), f32(j)) * radius / 3.0);
                }
            }
            let band = smoothstep(0.42, 0.45, p.y / size.y);
            return mix(layer(p), sum / 49.0, band);
        }
    };
    let c = card(ui, grid, "Frost", "Frosted below a line", 2);
    ui.set_effect(c, Some(Effect::new(&frost).arg("radius", 6.0)));

    // The colour channels pulled apart, from the middle out.
    let split = ui_shader! {
        fn split(p: vec2<f32>, size: vec2<f32>, amount: f32) -> vec4<f32> {
            let o = (p - size * 0.5) / max(size.x, size.y) * amount;
            // Kept on the card: past its edge there is nothing to pull in.
            let lo = vec2<f32>(1.0);
            let hi = size - vec2<f32>(1.0);
            let r = layer(clamp(p + o, lo, hi));
            let g = layer(p);
            let b = layer(clamp(p - o, lo, hi));
            // Within the card's own outline (its channels apart inside it,
            // not a frame of them round it).
            // Each channel as a colour (where a sample falls in a rounded
            // corner, on nothing, the card's own there).
            let shape = g.a;
            let own = g.rgb / max(g.a, 0.001);
            let red = select(own.r, r.r / max(r.a, 0.001), r.a > 0.01);
            let blue = select(own.b, b.b / max(b.a, 0.001), b.a > 0.01);
            return vec4<f32>(vec3<f32>(red, own.g, blue) * shape, shape);
        }
    };
    let chromatic = card(ui, grid, "Split", "Colour channels apart", 2);
    ui.set_effect(chromatic, Some(Effect::new(&split).arg("amount", state.shader_split)));

    // Big pixels.
    let pixels = ui_shader! {
        fn pixels(p: vec2<f32>, cell: f32) -> vec4<f32> {
            let c = max(cell, 1.0);
            return layer((floor(p / c) + vec2<f32>(0.5)) * c);
        }
    };
    let pix = card(ui, grid, "Pixels", "Blocks of a size", 1);
    ui.set_effect(pix, Some(Effect::new(&pixels).arg("cell", state.shader_cell)));

    // Burnt away through noise, embers at the edge.
    let burn = ui_shader! {
        fn hash(p: vec2<f32>) -> f32 {
            return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.547);
        }
        fn noise(p: vec2<f32>) -> f32 {
            let i = floor(p);
            let f = fract(p);
            let u = f * f * (vec2<f32>(3.0) - 2.0 * f);
            let a = mix(hash(i), hash(i + vec2<f32>(1.0, 0.0)), u.x);
            let b = mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x);
            return mix(a, b, u.y);
        }
        fn burn(p: vec2<f32>, color: vec4<f32>, progress: f32) -> vec4<f32> {
            let n = noise(p * 0.07) * 0.7 + noise(p * 0.25) * 0.3;
            let t = progress * 1.1;
            let keep = smoothstep(t - 0.04, t, n);
            let ember = keep * (1.0 - smoothstep(t, t + 0.08, n));
            let rgb = color.rgb * keep + vec3<f32>(1.0, 0.45, 0.1) * ember * color.a;
            return vec4<f32>(rgb, color.a * keep);
        }
    };
    let burnt = card(ui, grid, "Burn", "Through noise, with embers", 2);
    ui.set_effect(burnt, Some(Effect::new(&burn).arg("progress", state.shader_burn)));

    // Twisted round the pointer, most at the middle.
    let swirl = ui_shader! {
        fn swirl(p: vec2<f32>, centre: vec2<f32>, radius: f32, strength: f32) -> vec4<f32> {
            let d = p - centre;
            let r = length(d) / radius;
            if (r >= 1.0) {
                return layer(p);
            }
            let turn = strength * (1.0 - r) * (1.0 - r);
            let cs = cos(turn);
            let sn = sin(turn);
            return layer(centre + vec2<f32>(d.x * cs - d.y * sn, d.x * sn + d.y * cs));
        }
    };
    let c = card(ui, grid, "Swirl", "Twists round the pointer", 0);
    ui.set_effect(c, Some(Effect::new(&swirl).arg("radius", 70.0).arg("strength", 2.6).arg("centre", [90.0, 60.0])));
    input.effect_follows_pointer(c, "centre");

    // Printed in dots on a turned grid, each as large as it is bright.
    let halftone = ui_shader! {
        fn halftone(p: vec2<f32>, cell: f32) -> vec4<f32> {
            let k = 0.7071;
            let q = vec2<f32>(p.x * k - p.y * k, p.x * k + p.y * k);
            let mid = (floor(q / cell) + vec2<f32>(0.5)) * cell;
            let at = vec2<f32>(mid.x * k + mid.y * k, -mid.x * k + mid.y * k);
            let c = layer(at);
            let rgb = c.rgb / max(c.a, 0.001);
            let bright = max(max(rgb.r, rgb.g), rgb.b);
            let r = cell * 0.62 * sqrt(bright);
            let dot_ = 1.0 - smoothstep(r - 0.8, r + 0.8, length(q - mid));
            let ink = rgb * (0.25 + 0.75 * dot_);
            // The card's own outline, not the grid's.
            let shape = layer(p).a;
            return vec4<f32>(ink * shape, shape);
        }
    };
    let c = card(ui, grid, "Halftone", "Dots as large as bright", 3);
    ui.set_effect(c, Some(Effect::new(&halftone).arg("cell", 6.0)));

    // Mirrored wedges round the middle, turning.
    let kaleido = ui_shader! {
        fn kaleidoscope(p: vec2<f32>, size: vec2<f32>, time: f32, wedges: f32) -> vec4<f32> {
            let mid = size * 0.5;
            let d = p - mid;
            let r = length(d);
            let wedge = 6.2831853 / wedges;
            var a = atan2(d.y, d.x) + time * 0.3;
            a = a - wedge * floor(a / wedge);
            a = min(a, wedge - a);
            let q = mid + vec2<f32>(cos(a), sin(a)) * r;
            let c = layer(clamp(q, vec2<f32>(1.0), size - vec2<f32>(1.0)));
            // In the card's own outline.
            let shape = layer(p).a;
            return vec4<f32>(c.rgb / max(c.a, 0.001) * shape, shape);
        }
    };
    let c = card(ui, grid, "Kaleidoscope", "Mirrored, and turning", 0);
    ui.set_effect(c, Some(Effect::new(&kaleido).arg("wedges", 6.0)));

    // Only the edges, glowing, the hue running round.
    let neon = ui_shader! {
        fn luma(q: vec2<f32>) -> f32 {
            let c = layer(q);
            return dot(c.rgb, vec3<f32>(0.3, 0.59, 0.11)) + c.a * 0.2;
        }
        fn neon(p: vec2<f32>, time: f32) -> vec4<f32> {
            let gx = luma(p + vec2<f32>(1.0, -1.0)) + 2.0 * luma(p + vec2<f32>(1.0, 0.0)) + luma(p + vec2<f32>(1.0, 1.0))
                - luma(p - vec2<f32>(1.0, 1.0)) - 2.0 * luma(p - vec2<f32>(1.0, 0.0)) - luma(p + vec2<f32>(-1.0, 1.0));
            let gy = luma(p + vec2<f32>(-1.0, 1.0)) + 2.0 * luma(p + vec2<f32>(0.0, 1.0)) + luma(p + vec2<f32>(1.0, 1.0))
                - luma(p - vec2<f32>(1.0, 1.0)) - 2.0 * luma(p - vec2<f32>(0.0, 1.0)) - luma(p + vec2<f32>(1.0, -1.0));
            // Above a gradient's steps from one byte to the next.
            let edge = smoothstep(0.12, 0.45, length(vec2<f32>(gx, gy)));
            let hue = fract(p.x * 0.004 + p.y * 0.003 + time * 0.15);
            let glow = vec3<f32>(0.5) + 0.5 * cos(6.2831 * (vec3<f32>(hue) + vec3<f32>(0.0, 0.33, 0.67)));
            let c = layer(p);
            return vec4<f32>(vec3<f32>(0.03, 0.03, 0.06) * c.a + glow * edge, max(c.a, edge));
        }
    };
    let c = card(ui, grid, "Neon", "Edges, glowing", 3);
    ui.set_effect(c, Some(Effect::new(&neon)));

    // A flag in the wind: waves running along it, shaded as they turn.
    let flag = ui_shader! {
        fn flag(p: vec2<f32>, size: vec2<f32>, time: f32) -> vec4<f32> {
            let reach = p.x / size.x;
            let phase = p.x * 0.05 - time * 4.0;
            let lift = sin(phase) * 7.0 * reach;
            let c = layer(vec2<f32>(p.x, p.y - lift));
            let shade = 1.0 + cos(phase) * 0.22 * reach;
            return vec4<f32>(c.rgb * shade, c.a);
        }
    };
    let c = card(ui, grid, "Flag", "Waves in the wind", 1);
    ui.set_effect(c, Some(Effect::new(&flag).overflow(8.0)));

    // Bands that jump sideways, now and then, their colours apart.
    let glitch = ui_shader! {
        fn rand(v: vec2<f32>) -> f32 {
            return fract(sin(dot(v, vec2<f32>(12.9898, 78.233))) * 43758.547);
        }
        fn glitch(p: vec2<f32>, time: f32) -> vec4<f32> {
            let tick = floor(time * 9.0);
            let band = floor(p.y / 7.0);
            let roll = rand(vec2<f32>(band, tick));
            var shift = 0.0;
            if (roll > 0.82) {
                shift = (rand(vec2<f32>(tick, band)) - 0.5) * 30.0;
            }
            let q = vec2<f32>(p.x + shift, p.y);
            let r = layer(q + vec2<f32>(abs(shift) * 0.2, 0.0));
            let g = layer(q);
            let b = layer(q - vec2<f32>(abs(shift) * 0.2, 0.0));
            return vec4<f32>(r.r, g.g, b.b, (r.a + g.a + b.a) / 3.0);
        }
    };
    let c = card(ui, grid, "Glitch", "A bad signal", 2);
    ui.set_effect(c, Some(Effect::new(&glitch).overflow(8.0)));

    let sliders = ui.add_box(g, style! { display: grid; grid-template-columns: 90px 1fr; column-gap: 12px; row-gap: 6px; align-items: center; max-width: 520px; });
    for (label, value, range) in [
        ("Split", threers_ui::binding!(|s: State| s.shader_split), 0.0..=40.0),
        ("Pixels", threers_ui::binding!(|s: State| s.shader_cell), 1.0..=16.0),
        ("Burn", threers_ui::binding!(|s: State| s.shader_burn), 0.0..=1.0),
    ] {
        let l = ui.add_box(sliders, style! { font-size: 12px; });
        ui.add_text(l, label);
        let step = if label == "Burn" { 0.01 } else { 1.0 };
        slider(ui, input, state, sliders, label, value, range, step);
    }

    // How one of them is written.
    let src = group(ui, page, "Writing one", "A shader is a WGSL function: the position first, then what it wants filled in (color, time, size) and its own arguments, set by name. layer(q) reads the box anywhere.");
    let code = ui.add_box(src, style! { font-family: monospace; font-size: 12px; white-space: pre; padding: 12px; border-radius: 8px; overflow: auto; line-height: 18px; });
    ui.token_style(code, &[("background", Token::Raised)]);
    ui.add_text(code, RIPPLE_SOURCE);

    Shaders { chromatic, pixels: pix, burn: burnt, last: (state.shader_split, state.shader_cell, state.shader_burn) }
}
