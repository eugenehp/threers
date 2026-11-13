//! The indicators' shaders: WGSL in `shaders/`, put together for each kind.

use super::*;

/// The shaders' common parts: colour, distance and output.
const LIB: &str = include_str!("shaders/lib.wgsl");

/// A flat bar. `look`: glow, shimmer, busy, unused.
const BAR_FLAT: &str = include_str!("shaders/bar_flat.wgsl");

/// A flat ring. `look`: glow, shimmer, busy, the stroke's width in pixels.
const RING_FLAT: &str = include_str!("shaders/ring_flat.wgsl");

/// What the solid indicators share: their pose, a studio to reflect, ray
/// and capsule geometry, and glass.
const SOLID_LIB: &str = concat!(include_str!("../studio.wgsl"), include_str!("shaders/solid.wgsl"));

/// A solid bar: a glass capsule, liquid inside, ray-traced exactly in a
/// mild perspective (each intersection is a quadratic).
const BAR_SOLID: &str = include_str!("shaders/bar_solid.wgsl");

/// A solid ring: a glass torus tipped back, liquid round part of it,
/// ray-marched in a mild perspective; the liquid through the glass along
/// the refracted ray.
const RING_SOLID: &str = include_str!("shaders/ring_solid.wgsl");

/// Moves every kind: the pixel taken back through the motion before the
/// indicator is drawn. Solid ones turn in 3D (through `pfx_pose`); flat
/// ones are turned in the plane and narrowed as a card turning would be.
/// `m`: the motion (see [`Motion::code`]), its speed, its amount, and how
/// far a hop goes in pixels.
const MOTION: &str = include_str!("shaders/motion.wgsl");

/// A battery: a rounded body and a terminal, the charge inside it. Flat:
/// a light outline. Solid: the body is glass (`pfx_glass`), bevelled at
/// its rim. The fill takes the palette's colour at the charge.
const BATTERY: &str = include_str!("shaders/battery.wgsl");

/// Electricity along a bar or round a ring: a few bolts, each a line
/// pushed about by noise, struck anew many times a second, with a white
/// core and coloured light round it.
const ELECTRIC: &str = include_str!("shaders/electric.wgsl");

/// Fire, water, plasma and frost (`PFX_KIND` 1 to 4, set before this):
/// one function for the look, called along a bar or round a ring.
const EFFECTS: &str = include_str!("shaders/effects.wgsl");
/// A fill drawn along a bar and round a ring, calling `PFX_FILL`.
const FILL: &str = include_str!("shaders/fill.wgsl");

/// The effect's WGSL for `shape` and `depth`: taking the time when
/// `animated`, else a `clock` set from Rust (and left still).
pub fn source(shape: Shape, depth: Depth, animated: bool) -> String {
    source_filled(shape, depth, Fill::Neon, animated)
}

/// [`source`] with what fills it.
pub fn source_filled(shape: Shape, depth: Depth, fill: Fill, animated: bool) -> String {
    let ring = shape == Shape::Ring;
    let pick = |r: &'static str, b: &'static str| if ring { r } else { b };
    // What it needs past the common parts (the 3D library, the effect's
    // kind), its body and the body's function, and whether it turns in 3D
    // (else motion turns it as a flat card).
    let fill_body = |call: &str| format!("{}\n{}", if call == "pfx_fx" { EFFECTS } else { "" }, FILL.replace("PFX_FILL", call));
    let (needs, body, name, turns): (String, String, &str, bool) = match (fill, shape, depth) {
        (Fill::Electric, ..) => (String::new(), ELECTRIC.to_owned(), pick("pfx_ring_electric", "pfx_bar_electric"), false),
        (Fill::Custom(c), ..) => {
            let (source, function) = custom_source(c);
            (source, fill_body(&function), pick("pfx_ring_fx", "pfx_bar_fx"), false)
        }
        (f, ..) if f.kind() > 0 => (format!("const PFX_KIND: i32 = {};\n", f.kind()), fill_body("pfx_fx"), pick("pfx_ring_fx", "pfx_bar_fx"), false),
        // Both batteries are one function, the glass in it.
        (_, Shape::Battery, d) => (SOLID_LIB.to_owned(), BATTERY.to_owned(), if d == Depth::Solid { "pfx_battery_solid" } else { "pfx_battery_flat" }, false),
        (_, _, Depth::Flat) => (String::new(), pick(RING_FLAT, BAR_FLAT).to_owned(), pick("pfx_ring_flat", "pfx_bar_flat"), false),
        (_, _, Depth::Solid) => (SOLID_LIB.to_owned(), pick(RING_SOLID, BAR_SOLID).to_owned(), pick("pfx_ring_solid", "pfx_bar_solid"), true),
    };
    let clock = if animated { "time" } else { "clock" };
    let flat = !turns;
    // The shader library's colour and shapes under everything; its noise
    // and fills under the fills (a custom one may use any of it).
    use crate::shader_lib::{with, Library};
    let libs = match fill {
        Fill::Custom(_) => with(&Library::ALL, ""),
        f if f.kind() > 0 => with(&[Library::Color, Library::Sdf, Library::Fx], ""),
        _ => with(&[Library::Color, Library::Sdf], ""),
    };
    format!(
        "{libs}{LIB}{needs}{MOTION}{body}
fn progress_indicator(p: vec2<f32>, size: vec2<f32>, {clock}: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32, motion: vec4<f32>) -> vec4<f32> {{
    pfx_on_light = reach < 0.0;
    let q = pfx_motion(p, size, {clock}, motion, {flat});
    return {name}(q, size, {clock}, value, c0, c1, c2, look, abs(reach));
}}
"
    )
}

/// The custom fills: each one's source, its function, and what is wrong
/// with it.
struct Custom {
    source: String,
    function: String,
    error: Option<String>,
}

thread_local! {
    static CUSTOM: std::cell::RefCell<Vec<Custom>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Registers a custom fill's source (the same source, the same fill).
pub(super) fn register_custom(source: &str) -> CustomFill {
    CUSTOM.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(i) = c.iter().position(|x| x.source == source) {
            return CustomFill::new(i as u32);
        }
        let (function, error) = match fill_signature(source) {
            Ok(f) => (f, None),
            Err(e) => (String::new(), Some(e)),
        };
        c.push(Custom { source: source.to_owned(), function, error });
        CustomFill::new(c.len() as u32 - 1)
    })
}

pub(super) fn custom_error(c: CustomFill) -> Option<String> {
    CUSTOM.with(|x| x.borrow().get(c.id() as usize).and_then(|x| x.error.clone()))
}

/// A custom fill's source and its function's name (a stand-in that draws
/// nothing, when the source is not a fill).
fn custom_source(c: CustomFill) -> (String, String) {
    CUSTOM.with(|x| match x.borrow().get(c.id() as usize) {
        Some(x) if x.error.is_none() => (x.source.clone(), x.function.clone()),
        _ => (
            "fn pfx_no_fill(along: f32, up: f32, half: f32, env: f32, cover: f32, u: f32, now: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>) -> vec4<f32> { return vec4<f32>(0.0); }".to_owned(),
            "pfx_no_fill".to_owned(),
        ),
    })
}

/// The name of `source`'s last function, if it is a fill: ten parameters,
/// returning a `vec4<f32>`.
fn fill_signature(source: &str) -> Result<String, String> {
    let at = source.rfind("fn ").ok_or("no function: the last `fn` in the source is the fill")?;
    let rest = &source[at + 3..];
    let open = rest.find('(').ok_or("no parameter list")?;
    let name = rest[..open].trim().to_owned();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Err(format!("`{name}` is not a function name"));
    }
    let mut depth = 0;
    let mut close = None;
    for (i, ch) in rest[open..].char_indices() {
        match ch {
            '(' | '<' => depth += 1,
            ')' | '>' if depth > 0 => {
                depth -= 1;
                if depth == 0 && ch == ')' {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close.ok_or("unclosed parameter list")?;
    let params = rest[open + 1..close].split(',').filter(|p| p.contains(':')).count();
    let returns: String = rest[close + 1..].split('{').next().unwrap_or("").chars().filter(|c| !c.is_whitespace()).collect();
    if params != 10 || (returns != "->vec4<f32>" && returns != "->vec4f") {
        return Err(format!(
            "`{name}` must take (along, up, half, env, cover, u, now: f32, c0, c1, c2: vec4<f32>) and return vec4<f32>; it takes {params} and returns `{}`",
            returns.trim_start_matches("->")
        ));
    }
    Ok(name)
}

thread_local! {
    /// Each kind's shader, made once.
    static SHADERS: std::cell::RefCell<std::collections::HashMap<(Shape, Depth, Fill, bool), Shader>> = Default::default();
}

pub(super) fn shader(shape: Shape, depth: Depth, fill: Fill, animated: bool) -> Shader {
    SHADERS.with(|m| m.borrow_mut().entry((shape, depth, fill, animated)).or_insert_with(|| Shader::new(source_filled(shape, depth, fill, animated))).clone())
}

/// `c` decoded: effects mix and blend in linear light.
pub(super) fn linear(c: Rgba) -> [f32; 4] {
    use threers_math::srgb::srgb_to_linear_f32 as d;
    [d(c[0]), d(c[1]), d(c[2]), c[3]]
}
