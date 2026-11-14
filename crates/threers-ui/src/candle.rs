//! A candle for a UI: a wax pillar shaded as a cylinder, a melt pool, a
//! wick, a flame, and the warm glow it throws on whatever it stands in
//! front of. Built from ordinary nodes and animated from Rust: call
//! [`Candle::update`] each frame with the time.
//!
//! - [`FlameStyle::Simple`]: the flame is four nested teardrop paths
//!   (orange, yellow, a white core, a blue base), stretched, narrowed and
//!   leaned each frame by the flicker. Draws everywhere, the 2D canvas
//!   included.
//! - [`FlameStyle::Realistic`]: on the GPU the flame is drawn through a
//!   shader effect, a volume of hot gas seen through: a teardrop pushed
//!   about by rising noise, coloured as a black body by its temperature,
//!   with the dark zone over the wick and the blue sheet at its base (the
//!   3D candles' flame, `threers::extras::candle`, looked at side on).
//!   Where shaders do not run, the simple flame shows instead.
//!
//! The flicker is [`threers_math::flicker`], the 3D candles' too: the
//! flame, the glow and the light in the wax rise and fall together.
//!
//! ```
//! use threers_ui::candle::{candle, FlameStyle};
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let c = candle(&mut ui, root, FlameStyle::Simple, 140.0, 1.0);
//! c.update(&mut ui, 0.5);
//! ```

use threers_math::flicker::{flame, sway};

use crate::shader::{Effect, Shader};
use crate::style::Length;
use crate::transform::Transformation;
use crate::{style, NodeId, Role, Semantics, Ui};

/// How the flame is drawn. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlameStyle {
    #[default]
    Simple,
    Realistic,
}

/// A candle's nodes, for [`update`](Candle::update) to move.
#[derive(Debug, Clone)]
pub struct Candle {
    /// The candle's box (`width` × twice that, about).
    pub root: NodeId,
    glow: NodeId,
    lit_wax: NodeId,
    flame: NodeId,
    /// The flame's layers, outside in, and how much of the flame each is.
    layers: Vec<(NodeId, f32)>,
    style: FlameStyle,
    seed: f32,
}

/// The view box the flame's paths are drawn in: base at the bottom middle.
const VIEW: [f32; 4] = [0.0, 0.0, 100.0, 200.0];

/// A teardrop `w` of the view box wide and `h` tall, standing on the view
/// box's bottom middle: round below, drawn out to a point.
#[cfg(test)]
fn teardrop(w: f32, h: f32) -> String {
    teardrop_lifted(w, h, 0.0)
}

/// [`teardrop`] standing `lift` view-box units above the floor (the
/// flame's inner zones start above the dark zone over the wick).
fn teardrop_lifted(w: f32, h: f32, lift: f32) -> String {
    let (cx, base) = (50.0, 200.0 - lift);
    let (hw, top) = (50.0 * w, 200.0 - 200.0 * h);
    let top = top.min(base - 1.0);
    let shoulder = base - (base - top) * 0.42;
    format!(
        "M{cx} {base} C{l1} {base} {l0} {s1} {l0} {shoulder} C{l0} {s2} {cx} {t1} {cx} {top} C{cx} {t1} {r0} {s2} {r0} {shoulder} C{r0} {s1} {r1} {base} {cx} {base} Z",
        l1 = cx - hw * 0.62,
        l0 = cx - hw,
        r0 = cx + hw,
        r1 = cx + hw * 0.62,
        s1 = base - (base - shoulder) * 0.15,
        s2 = shoulder - (shoulder - top) * 0.45,
        t1 = top + (shoulder - top) * 0.35,
    )
}

/// The realistic flame's effect: for each pixel, the gas
/// ([`FLAME_WGSL`](threers_math::flicker::FLAME_WGSL)) gathered through the
/// flame's depth, side on. `t`, `flicker` and `lean` are set from Rust each
/// frame (no `time`: the page decides when to animate).
fn flame_source() -> String {
    format!("{}{}", threers_math::flicker::FLAME_WGSL, FLAME_ENTRY)
}

const FLAME_ENTRY: &str = r#"
fn candle_flame(p: vec2<f32>, size: vec2<f32>, t: f32, flicker: f32, lean: vec2<f32>, seed: f32) -> vec4<f32> {
    // Across the flame -1..1 (the box is the flame's width with room to
    // lean), up it 0..1 from the base at the box's bottom.
    let x = (p.x / size.x * 2.0 - 1.0) * 1.3;
    // The box runs from just under the gas's base to past its tip, so
    // both round off inside it rather than being cut by its edges.
    let y = (1.0 - p.y / size.y) * 1.12 + 0.03;
    var c = vec3<f32>(0.0);
    let steps = 32;
    // A different start in each pixel, so the steps show as grain rather
    // than rings round the thin blue cup.
    let jitter = flame_hash(vec3<f32>(p, 0.0));
    for (var i = 0; i < steps; i++) {
        let z = (f32(i) + jitter) / f32(steps) * 2.0 - 1.0;
        c = c + flame_gas(vec3<f32>(x, y, z), t, flicker, lean, seed) * (2.0 / f32(steps));
    }
    // Tone mapped as a camera would: the core, many times brighter than
    // white, reads white-hot; the rim keeps its orange; faint light is
    // see-through.
    let mapped = vec3<f32>(1.0) - exp(-c * flicker * 1.1);
    let a = clamp(max(mapped.r, max(mapped.g, mapped.b)), 0.0, 1.0);
    let straight = min(mapped / max(a, 1e-4), vec3<f32>(1.0));
    // Premultiplied, as an effect returns it.
    return vec4<f32>(pow(straight, vec3<f32>(1.0 / 2.2)) * a, a);
}
"#;

/// A candle `width` UI pixels wide (and twice as tall) in `parent`.
/// `seed` sets its flicker apart from another candle's.
pub fn candle(ui: &mut Ui, parent: NodeId, style: FlameStyle, width: f32, seed: f32) -> Candle {
    let (w, h) = (width, width * 2.0);
    let root = ui.add_box(parent, style! { position: relative; flex-shrink: 0; });
    ui.update_style(root, |s| {
        s.width = Length::Px(w);
        s.height = Length::Px(h);
    });
    ui.set_semantics(
        root,
        Semantics::new(Role::Image).label(match style {
            FlameStyle::Simple => "A candle, its flame flickering",
            FlameStyle::Realistic => "A candle with a realistic flame, flickering",
        }),
    );
    let px = |v: f32| Length::Px(v);
    let place = |ui: &mut Ui, n: NodeId, x: f32, y: f32, bw: f32, bh: f32| {
        ui.update_style(n, |s| {
            s.left = px(x);
            s.top = px(y);
            s.width = px(bw);
            s.height = px(bh);
        });
    };
    // Proportions: the wax a third of the width, its top just under half way.
    let (wax_w, wax_top) = (w * 0.34, h * 0.48);
    let cx = w / 2.0;
    let (fw, fh) = (wax_w * 0.62, h * 0.32);
    let flame_base = wax_top - h * 0.018;

    // The glow on what stands behind, round the flame.
    let glow = ui.add_box(root, style! { position: absolute; border-radius: 50%; background: radial-gradient(rgba(255, 168, 82, 0.5), rgba(255, 140, 60, 0.18) 45%, rgba(255, 120, 40, 0) 70%); pointer-events: none; });
    place(
        ui,
        glow,
        cx - w * 0.5,
        flame_base - fh * 0.5 - w * 0.5,
        w,
        w,
    );

    // The wax: shaded as a cylinder lit from the front left, a lighter
    // band where the flame's light comes through near the top.
    let wax = ui.add_box(root, style! { position: absolute; border-radius: 5px; background: linear-gradient(90deg, #8f8475, #efe6d4 32%, #f7f0e2 45%, #c9bda9 78%, #7a705f); });
    place(ui, wax, cx - wax_w / 2.0, wax_top, wax_w, h - wax_top - 2.0);
    let lit_wax = ui.add_box(wax, style! { position: absolute; left: 0px; top: 0px; width: 100%; border-radius: 5px; background: linear-gradient(180deg, rgba(255, 196, 120, 0.85), rgba(255, 170, 90, 0.35) 40%, rgba(255, 160, 80, 0) 100%); });
    ui.update_style(lit_wax, |s| s.height = px((h - wax_top) * 0.45));
    // The melt pool, a lit ellipse.
    let pool = ui.add_box(root, style! { position: absolute; border-radius: 50%; background: radial-gradient(#fff4d6, #f6d9a4 55%, #d9b98a); });
    place(
        ui,
        pool,
        cx - wax_w * 0.46,
        wax_top - wax_w * 0.06,
        wax_w * 0.92,
        wax_w * 0.2,
    );
    // The wick: dark, bent a little, its tip an ember.
    let wick = ui.add_box(root, style! { position: absolute; border-radius: 2px; background: linear-gradient(180deg, #ff6a2a, #2a1c12 30%, #1a120c); });
    place(
        ui,
        wick,
        cx - 1.5,
        flame_base + 2.0,
        3.0,
        wax_top - flame_base + 2.0,
    );

    // The flame: nested teardrops (the simple flame, and the realistic
    // one's stand-in where shaders do not run).
    let flame_box = ui.add_box(root, style! { position: absolute; pointer-events: none; });
    let (bw, bh) = (fw * 1.6, fh * 1.15);
    place(
        ui,
        flame_box,
        cx - bw / 2.0,
        flame_base - bh + bh * 0.03 / 1.12,
        bw,
        bh,
    );
    let mut layers = vec![];
    // Outer orange, the yellow body and the white-hot core lifted off the
    // wick (the dark zone shows under them), then the thin blue cup round
    // the base, see-through.
    for (color, scale, wide, lift) in [
        ("rgba(255, 112, 30, 0.8)", 1.0, 0.55, 0.0),
        ("rgba(255, 190, 64, 0.95)", 0.82, 0.44, 14.0),
        ("rgba(255, 246, 220, 1)", 0.56, 0.28, 34.0),
    ] {
        let path = ui.add_path(
            flame_box,
            style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; },
            vec![teardrop_lifted(wide, scale, lift)],
            VIEW,
            None,
            true,
        );
        ui.update_style(path, |s| {
            crate::css::apply(s, "color", color);
        });
        layers.push((path, scale));
    }
    let cup = ui.add_path(
        flame_box,
        style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; },
        vec!["M28 178 A22 22 0 0 0 72 178".to_owned()],
        VIEW,
        Some(5.0),
        false,
    );
    ui.update_style(cup, |s| {
        crate::css::apply(s, "color", "rgba(90, 150, 255, 0.55)")
    });
    layers.push((cup, 1.0));
    if style == FlameStyle::Realistic {
        let shader = Shader::new(flame_source());
        debug_assert!(shader.error().is_none(), "{:?}", shader.error());
        ui.set_effect(
            flame_box,
            Some(
                Effect::new(&shader)
                    .arg("t", 0.0)
                    .arg("flicker", 1.0)
                    .arg("lean", [0.0, 0.0])
                    .arg("seed", seed),
            ),
        );
    }
    Candle {
        root,
        glow,
        lit_wax,
        flame: flame_box,
        layers,
        style,
        seed,
    }
}

impl Candle {
    /// The flame, its glow and the light in the wax at `time` seconds.
    /// (Under reduced motion, call it once: the candle then stays as it is.)
    pub fn update(&self, ui: &mut Ui, time: f32) {
        let f = flame(time, self.seed);
        let (sx, sz) = sway(time, self.seed);
        // Each layer stretched as it flares and narrowed to keep its fire,
        // leaned with the draught, its base kept on the wick.
        for &(node, _) in &self.layers {
            let mut t = Transformation {
                rotate: sx * 0.12,
                scale: (1.0 / f.sqrt(), f),
                translate: (0.0, 0.0),
            };
            let base = (50.0, 200.0);
            let moved = t.apply(base, (50.0, 100.0));
            t.translate = (base.0 - moved.0, base.1 - moved.1);
            ui.set_path_transform(node, t);
        }
        let k = (f - 0.8) / 0.4;
        ui.update_style(self.glow, |s| s.opacity = 0.65 + 0.35 * k);
        ui.update_style(self.lit_wax, |s| s.opacity = 0.55 + 0.45 * k);
        if self.style == FlameStyle::Realistic {
            ui.set_effect_arg(self.flame, "t", time);
            ui.set_effect_arg(self.flame, "flicker", f);
            ui.set_effect_arg(self.flame, "lean", [sx, sz]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flame_shader_compiles_as_an_effect() {
        let s = Shader::new(flame_source());
        assert_eq!(s.error(), None);
        assert_eq!(s.function(), "candle_flame");
    }

    #[test]
    fn a_candle_flickers_its_flame_and_glow() {
        let mut ui = Ui::new();
        let root = ui.root();
        let c = candle(&mut ui, root, FlameStyle::Realistic, 140.0, 2.0);
        let mut seen = vec![];
        for i in 0..40 {
            c.update(&mut ui, i as f32 * 0.09);
            seen.push((
                ui.style(c.glow).opacity,
                ui.path_transform(c.layers[0].0).scale.1,
            ));
        }
        let spread = |v: Vec<f32>| {
            v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min)
        };
        assert!(
            spread(seen.iter().map(|s| s.0).collect()) > 0.05,
            "the glow breathes"
        );
        assert!(
            spread(seen.iter().map(|s| s.1).collect()) > 0.05,
            "the flame stretches"
        );
        // The base stays on the wick as the flame stretches and leans.
        let t = ui.path_transform(c.layers[0].0);
        let base = t.apply((50.0, 200.0), (50.0, 100.0));
        assert!(
            (base.0 - 50.0).abs() < 1e-3 && (base.1 - 200.0).abs() < 1e-3,
            "{base:?}"
        );
        assert!(
            ui.effect(c.flame).is_some(),
            "the realistic flame is an effect"
        );
    }

    #[test]
    fn a_teardrop_stands_on_the_view_box_floor() {
        let d = teardrop(0.5, 1.0);
        assert!(d.starts_with("M50 200 "), "{d}");
        assert!(d.contains(" 50 0 "), "its tip at the top: {d}");
    }
}
