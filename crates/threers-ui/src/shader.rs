//! Custom shaders on boxes: a few lines of WGSL that recolour, warp or
//! otherwise remake what a box and everything inside it draws.
//!
//! An [`Effect`] on a node has the GPU painter draw the node's subtree into
//! a layer the size of its box, then put the layer on screen through the
//! effect's function, run for every pixel of the box. The function's first
//! parameter is where the pixel is, in UI pixels from the box's top left;
//! it returns that pixel's colour, premultiplied RGBA from 0 to 1. Other
//! parameters are filled in by name:
//!
//! - `color: vec4<f32>` — the layer at that point: what the box drew there;
//! - `time: f32` — seconds on the UI's clock; a shader that takes it is
//!   animated, and keeps frames coming (see [`Ui::needs_frame`]);
//! - `size: vec2<f32>` — the box's size in UI pixels;
//! - anything else, an `f32` or a `vec2`/`vec3`/`vec4` of `f32`s, is an
//!   argument: set by name with [`Effect::arg`] (up to eight).
//!
//! And `layer(q)` reads the layer anywhere: at a moved point to warp it, at
//! several to blur or glow. Write the WGSL inline with [`ui_shader!`](crate::ui_shader), helpers
//! first, the effect last:
//!
//! ```
//! use threers_ui::ui_shader;
//! use threers_ui::shader::Effect;
//!
//! // A wave running across the box.
//! let wave = ui_shader! {
//!     fn wave(p: vec2<f32>, time: f32, size: vec2<f32>, amplitude: f32) -> vec4<f32> {
//!         let dy = sin(p.x / size.x * 12.0 + time * 4.0) * amplitude;
//!         return layer(p + vec2<f32>(0.0, dy));
//!     }
//! };
//! assert!(wave.error().is_none());
//! let effect = Effect::new(&wave).arg("amplitude", 6.0).overflow(6.0);
//! assert_eq!(effect.values()[0][0], 6.0);
//! ```
//!
//! Painters that cannot run shaders (the 2D canvas, the CPU overlay) draw
//! the box as it is; so does the GPU painter when the shader fails to
//! compile, saying why in its stats.

use std::sync::Arc;

use crate::input::{Event, Input, Phase};
use crate::{NodeId, Ui};

/// How much the animated shader effects (those taking the `time`) may do,
/// for a phone to stay cool: see [`Ui::set_animation_budget`]. The default
/// is [`UNLIMITED`](Self::UNLIMITED): every one, every frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimationBudget {
    /// How many run at full rate, the largest on screen first; `None`, all.
    pub max_full_rate: Option<usize>,
    /// How fast the rest go, frames a second; 0 holds them still.
    pub overflow_fps: f32,
    /// Effects whose shorter side is under this (UI pixels) are small…
    pub small: f32,
    /// …and go this fast.
    pub small_fps: f32,
    /// None faster than this; 0 for no cap.
    pub max_fps: f32,
}

impl AnimationBudget {
    /// Every animated effect, every frame.
    pub const UNLIMITED: AnimationBudget = AnimationBudget { max_full_rate: None, overflow_fps: 0.0, small: 0.0, small_fps: 0.0, max_fps: 0.0 };

    /// For a phone: four at up to 30 frames a second, the rest at 8, those
    /// under 48 pixels at 15.
    pub const PHONE: AnimationBudget = AnimationBudget { max_full_rate: Some(4), overflow_fps: 8.0, small: 48.0, small_fps: 15.0, max_fps: 30.0 };

    /// At most `n` at full rate, the rest held still.
    pub fn at_most(n: usize) -> Self {
        Self { max_full_rate: Some(n), ..Self::UNLIMITED }
    }

    pub fn overflow_fps(self, fps: f32) -> Self {
        Self { overflow_fps: fps.max(0.0), ..self }
    }

    /// Effects smaller than `side` UI pixels go `fps` frames a second.
    pub fn small(self, side: f32, fps: f32) -> Self {
        Self { small: side.max(0.0), small_fps: fps.max(0.0), ..self }
    }

    pub fn max_fps(self, fps: f32) -> Self {
        Self { max_fps: fps.max(0.0), ..self }
    }
}

impl Default for AnimationBudget {
    fn default() -> Self {
        Self::UNLIMITED
    }
}

/// How many arguments a shader can take besides the ones filled in.
pub const MAX_ARGS: usize = 8;

/// WGSL for an [`Effect`]: made with [`ui_shader!`](crate::ui_shader) or [`Shader::new`], and
/// cheap to clone. See the [module docs](self).
#[derive(Clone)]
pub struct Shader(Arc<Inner>);

struct Inner {
    id: u64,
    source: String,
    function: String,
    params: Vec<Param>,
    error: Option<String>,
}

/// One of a shader function's parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
}

/// What fills a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// The pixel's place in the box, in UI pixels (the first parameter).
    Position,
    /// The layer at the pixel.
    Color,
    /// Seconds on the UI's clock.
    Time,
    /// The box's size in UI pixels.
    Size,
    /// An argument of this many `f32`s (1 to 4), in this argument slot.
    Arg { components: u8, slot: u8 },
}

impl Shader {
    /// A shader from WGSL whose last function is the effect (see the
    /// [module docs](self)). A source that does not fit is still a shader:
    /// [`error`](Self::error) says what is wrong, and painters draw boxes
    /// with it as they are.
    pub fn new(source: impl Into<String>) -> Shader {
        let source = source.into();
        let id = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            source.hash(&mut h);
            h.finish()
        };
        let (function, params, error) = match signature(&source) {
            Ok((f, p)) => (f, p, None),
            Err(e) => (String::new(), vec![], Some(e)),
        };
        Shader(Arc::new(Inner { id, source, function, params, error }))
    }

    /// Identifies the source: equal sources, equal ids.
    pub fn id(&self) -> u64 {
        self.0.id
    }

    pub fn source(&self) -> &str {
        &self.0.source
    }

    /// The effect function's name.
    pub fn function(&self) -> &str {
        &self.0.function
    }

    pub fn params(&self) -> &[Param] {
        &self.0.params
    }

    /// Why the source cannot be an effect, when it cannot. Only its shape is
    /// checked here; the GPU painter reports what its compiler finds.
    pub fn error(&self) -> Option<&str> {
        self.0.error.as_deref()
    }

    /// Whether it takes the time, and so changes from frame to frame.
    pub fn animated(&self) -> bool {
        self.0.params.iter().any(|p| p.kind == ParamKind::Time)
    }

    /// The argument slot of the parameter `name`.
    fn slot(&self, name: &str) -> Option<usize> {
        self.0.params.iter().find_map(|p| match p.kind {
            ParamKind::Arg { slot, .. } if p.name == name => Some(slot as usize),
            _ => None,
        })
    }
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shader").field("function", &self.0.function).field("id", &self.0.id).finish()
    }
}

impl PartialEq for Shader {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

/// The effect function: the source's last `fn`, its name and parameters.
fn signature(source: &str) -> Result<(String, Vec<Param>), String> {
    let at = last_fn(source).ok_or("no function: the last `fn` in the source is the effect")?;
    let rest = &source[at + 2..];
    let open = rest.find('(').ok_or("no parameter list")?;
    let name = rest[..open].trim().to_string();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Err(format!("`{name}` is not a function name"));
    }
    // To the matching parenthesis.
    let mut depth = 0;
    let mut close = None;
    for (i, c) in rest[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close.ok_or("unclosed parameter list")?;
    let returns: String = rest[close + 1..].split('{').next().unwrap_or("").chars().filter(|c| !c.is_whitespace()).collect();
    if returns != "->vec4<f32>" && returns != "->vec4f" {
        return Err(format!("`{name}` must return vec4<f32>, the pixel's premultiplied colour"));
    }
    let mut params = vec![];
    let mut slot = 0u8;
    for (i, part) in split_top(&rest[open + 1..close]).into_iter().enumerate() {
        let (pname, ty) = part.split_once(':').ok_or_else(|| format!("parameter `{}` has no type", part.trim()))?;
        // Attributes, if any, come before the name.
        let pname = pname.split_whitespace().last().unwrap_or("").to_string();
        let ty: String = ty.chars().filter(|c| !c.is_whitespace()).collect();
        let components = match ty.as_str() {
            "f32" => 1,
            "vec2<f32>" | "vec2f" => 2,
            "vec3<f32>" | "vec3f" => 3,
            "vec4<f32>" | "vec4f" => 4,
            _ => return Err(format!("`{pname}: {ty}`: parameters are f32 or vec2/vec3/vec4 of f32")),
        };
        let kind = if i == 0 {
            if components != 2 {
                return Err(format!("the first parameter, `{pname}`, is the position: a vec2<f32>"));
            }
            ParamKind::Position
        } else {
            match (pname.as_str(), components) {
                ("color", 4) => ParamKind::Color,
                ("time", 1) => ParamKind::Time,
                ("size", 2) => ParamKind::Size,
                ("color" | "time" | "size", _) => {
                    return Err(format!("`{pname}` is filled in: color is a vec4<f32>, time an f32, size a vec2<f32>"))
                }
                _ => {
                    if slot as usize == MAX_ARGS {
                        return Err(format!("more than {MAX_ARGS} arguments"));
                    }
                    slot += 1;
                    ParamKind::Arg { components, slot: slot - 1 }
                }
            }
        };
        params.push(Param { name: pname, kind });
    }
    if params.is_empty() {
        return Err(format!("`{name}` needs the position, a vec2<f32>, first"));
    }
    Ok((name, params))
}

/// Where the last `fn` keyword starts.
fn last_fn(source: &str) -> Option<usize> {
    let b = source.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    source.match_indices("fn").map(|(i, _)| i).filter(|&i| (i == 0 || !ident(b[i - 1])) && b.get(i + 2).is_some_and(|c| c.is_ascii_whitespace())).last()
}

/// `s` split at commas outside angle brackets and parentheses.
fn split_top(s: &str) -> Vec<&str> {
    let mut out = vec![];
    let (mut depth, mut start) = (0i32, 0);
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' => depth += 1,
            '>' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out.into_iter().filter(|p| !p.trim().is_empty()).collect()
}

/// A shader on a box, with its arguments. See the [module docs](self).
#[derive(Debug, Clone, PartialEq)]
pub struct Effect {
    pub shader: Shader,
    values: [[f32; 4]; MAX_ARGS],
    /// How far past its box the effect may draw, in UI pixels: the layer is
    /// this much bigger on every side (a wave or a glow reaches out).
    pub overflow: f32,
    /// How what it makes is put over what is behind the box (from the box's
    /// `mix-blend-mode`).
    pub blend: crate::BlendMode,
}

impl Effect {
    /// `shader` with its arguments at zero.
    pub fn new(shader: &Shader) -> Effect {
        Effect { shader: shader.clone(), values: [[0.0; 4]; MAX_ARGS], overflow: 0.0, blend: crate::BlendMode::Normal }
    }

    /// With the argument `name` at `value`.
    pub fn arg(mut self, name: &str, value: impl Into<ArgValue>) -> Effect {
        self.set(name, value);
        self
    }

    /// Sets the argument `name`; `false` when the shader has none by that name.
    pub fn set(&mut self, name: &str, value: impl Into<ArgValue>) -> bool {
        match self.shader.slot(name) {
            Some(slot) => {
                self.values[slot] = value.into().0;
                true
            }
            None => false,
        }
    }

    /// With room to draw `px` UI pixels past the box on every side.
    pub fn overflow(mut self, px: f32) -> Effect {
        self.overflow = px.max(0.0);
        self
    }

    /// The arguments by slot, each padded to four.
    pub fn values(&self) -> &[[f32; 4]; MAX_ARGS] {
        &self.values
    }
}

/// An argument's value: an `f32`, or two to four of them (a colour is four).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArgValue(pub [f32; 4]);

impl From<f32> for ArgValue {
    fn from(v: f32) -> Self {
        ArgValue([v, 0.0, 0.0, 0.0])
    }
}

impl From<[f32; 2]> for ArgValue {
    fn from(v: [f32; 2]) -> Self {
        ArgValue([v[0], v[1], 0.0, 0.0])
    }
}

impl From<[f32; 3]> for ArgValue {
    fn from(v: [f32; 3]) -> Self {
        ArgValue([v[0], v[1], v[2], 0.0])
    }
}

impl From<[f32; 4]> for ArgValue {
    fn from(v: [f32; 4]) -> Self {
        ArgValue(v)
    }
}

/// A shader from WGSL written inline, its last function the effect. See the
/// [module docs](crate::shader).
///
/// The WGSL goes through Rust's tokenizer, so `//` comments vanish and
/// spacing changes, which WGSL does not mind; [`Shader::new`] takes a string
/// for what Rust cannot tokenize.
///
/// `use` parts of the [`shader_lib`](crate::shader_lib) first to put them
/// in front: `ui_shader! { use Noise, Color; fn … }`.
#[macro_export]
macro_rules! ui_shader {
    (use $($lib:ident),+ ; $($wgsl:tt)*) => {
        $crate::shader::Shader::with_libraries(&[$($crate::shader_lib::Library::$lib),+], ::core::stringify!($($wgsl)*))
    };
    ($($wgsl:tt)*) => {
        $crate::shader::Shader::new(::core::stringify!($($wgsl)*))
    };
}

impl Ui {
    /// Draws `node` and what is inside it through `effect` (on painters that
    /// run shaders), or as they are with `None`.
    pub fn set_effect(&mut self, node: NodeId, effect: Option<Effect>) {
        match effect {
            Some(e) => self.env.effects.insert(node, e),
            None => self.env.effects.remove(&node),
        };
        self.repaint();
    }

    pub fn effect(&self, node: NodeId) -> Option<&Effect> {
        self.env.effects.get(&node)
    }

    /// Sets an argument of `node`'s effect (say, from a slider or a drag);
    /// `false` when it has no effect or the effect no such argument.
    pub fn set_effect_arg(&mut self, node: NodeId, name: &str, value: impl Into<ArgValue>) -> bool {
        let set = self.env.effects.get_mut(&node).is_some_and(|e| e.set(name, value));
        if set {
            self.repaint();
        }
        set
    }

    /// Draws again: nothing moved, but what the boxes show did.
    fn repaint(&mut self) {
        self.env.repaint.set(true);
    }

    /// Whether a box on screen has an animated effect, so frames must keep
    /// coming.
    pub(crate) fn effects_animate(&self) -> bool {
        let drawn = self.env.effect_drawn.borrow();
        self.env.effects.iter().any(|(&n, e)| e.shader.animated() && self.shows(n) && drawn.get(&n) != Some(&self.effect_time(n)))
    }

    /// Sets how much the shader effects may animate (see
    /// [`AnimationBudget`]): how many at full rate, how fast the small and
    /// the rest go. Unlimited by default.
    pub fn set_animation_budget(&mut self, budget: AnimationBudget) {
        self.env.budget = budget;
        self.plan_effects();
        self.repaint();
    }

    pub fn animation_budget(&self) -> AnimationBudget {
        self.env.budget
    }

    /// Seconds until an animated effect on screen next changes (0: it
    /// changes every frame), `None` when none will: for an app that sleeps
    /// between frames, when to wake.
    pub fn next_effect_frame_in(&self) -> Option<f64> {
        let clock = self.clock;
        self.env
            .effects
            .iter()
            .filter(|(n, e)| e.shader.animated() && self.shows(**n))
            .filter_map(|(n, _)| {
                let rate = self.env.effect_rates.get(n).copied().unwrap_or(f32::INFINITY) as f64;
                if rate.is_infinite() {
                    Some(0.0)
                } else if rate <= 0.0 {
                    None
                } else {
                    Some(((clock * rate).floor() + 1.0) / rate - clock)
                }
            })
            .reduce(f64::min)
    }

    /// The time `node`'s effect is drawn at now: the clock, stepped to its
    /// rate, or held where it stopped.
    fn effect_time(&self, node: NodeId) -> f32 {
        let rate = self.env.effect_rates.get(&node).copied().unwrap_or(f32::INFINITY);
        let clock = self.clock as f32;
        if rate.is_infinite() {
            clock
        } else if rate <= 0.0 {
            self.env.effect_drawn.borrow().get(&node).copied().unwrap_or(clock)
        } else {
            (self.clock * rate as f64).floor() as f32 / rate
        }
    }

    /// Each animated effect on screen's rate under the budget: the largest
    /// at full rate up to the limit, the rest slowed (or held), the small
    /// ones slowed, none past the cap.
    pub(crate) fn plan_effects(&mut self) {
        let b = self.env.budget;
        let mut shown: Vec<(NodeId, f32, f32)> = self
            .env
            .effects
            .iter()
            .filter(|(n, e)| e.shader.animated() && self.shows(**n))
            .map(|(&n, _)| {
                let r = self.rect(n);
                (n, r.width * r.height, r.width.min(r.height))
            })
            .collect();
        // Largest first; ties by node, so the plan is the same each frame.
        shown.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0 .0.cmp(&b.0 .0)));
        self.env.effect_rates.clear();
        for (i, (n, _, side)) in shown.into_iter().enumerate() {
            let mut rate = if side < b.small { b.small_fps } else { f32::INFINITY };
            if b.max_full_rate.is_some_and(|max| i >= max) {
                rate = rate.min(b.overflow_fps);
            }
            if b.max_fps > 0.0 {
                rate = rate.min(b.max_fps);
            }
            if rate.is_finite() {
                self.env.effect_rates.insert(n, rate.max(0.0));
            }
        }
    }

    /// Whether `node` is drawn, somewhere in the window: it and the boxes
    /// around it displayed and not see-through.
    fn shows(&self, node: NodeId) -> bool {
        self.is_on_screen(node)
    }

    /// Whether any of `node` shows, as last laid out: it and the boxes
    /// round it displayed and not see-through, and some of it inside the
    /// window and every box that clips it (a scrolled-away list row does
    /// not). What an animation asks before it asks for frames.
    pub fn is_on_screen(&self, node: NodeId) -> bool {
        if !self.contains(node) {
            return false;
        }
        let hidden = |n: NodeId| {
            let s = self.style(n);
            s.display == crate::Display::None || s.opacity <= 0.0
        };
        if hidden(node) || self.ancestors(node).any(hidden) {
            return false;
        }
        let overlap = |r: crate::Rect, w: crate::Rect| r.x < w.x + w.width && r.y < w.y + w.height && r.x + r.width > w.x && r.y + r.height > w.y;
        let r = self.rect(node);
        overlap(r, self.rect(self.root())) && self.ancestors(node).skip(1).filter(|&a| self.style(a).clip).all(|a| overlap(r, self.rect(a)))
    }

    /// Opens `node`'s effect on `painter`, if it has one; whether it did.
    pub(crate) fn push_effect(&self, node: NodeId, painter: &mut dyn crate::Painter) -> bool {
        let s = &self.nodes[node.0].style;
        let rect = self.nodes[node.0].rect;
        let custom = self.env.effects.get(&node).filter(|e| e.shader.error().is_none());
        // Its own shader; else a mask's; else, to blend, one that changes
        // nothing.
        let mut effect = match (custom, &s.mask) {
            (Some(e), _) => e.clone(),
            (None, Some((shape, stops))) => mask_effect(rect, *shape, stops),
            (None, None) if s.mix_blend_mode != crate::BlendMode::Normal => PLAIN.with(Effect::new),
            _ => return false,
        };
        effect.blend = s.mix_blend_mode;
        let time = self.effect_time(node);
        if effect.shader.animated() {
            self.env.effect_drawn.borrow_mut().insert(node, time);
        }
        painter.push_effect(rect, &effect, time);
        true
    }
}

thread_local! {
    /// Draws the layer as it is: for a blend mode on a box with no shader.
    static PLAIN: Shader = Shader::new("fn plain(p: vec2<f32>) -> vec4<f32> { return layer(p); }");
    /// Draws the layer as opaque as a gradient is (see `mask_effect`).
    static MASK: Shader = Shader::new(MASK_SOURCE);
}

/// `shape`: x the kind (0 along a line, 1 rings, 2 round), then for a line
/// its direction (yz) over its length (w), for rings the centre (yz, in the
/// box) out to the farthest corner (w), for round the centre (yz) from an
/// angle (w). `a` and `b`: four stops, `(t, alpha)` each.
const MASK_SOURCE: &str = "
fn mask(p: vec2<f32>, size: vec2<f32>, shape: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    var t = 0.0;
    if (shape.x < 0.5) {
        t = dot(p - size * 0.5, shape.yz) / max(shape.w, 0.000001) + 0.5;
    } else if (shape.x < 1.5) {
        t = length(p - shape.yz) / max(shape.w, 0.000001);
    } else {
        let d = p - shape.yz;
        t = fract((atan2(d.x, -d.y) - shape.w) / 6.2831853);
    }
    let ts = vec4<f32>(a.x, a.z, b.x, b.z);
    let vs = vec4<f32>(a.y, a.w, b.y, b.w);
    var alpha = vs.x;
    for (var i = 0; i < 3; i = i + 1) {
        if (t > ts[i]) {
            let f = clamp((t - ts[i]) / max(ts[i + 1] - ts[i], 0.000001), 0.0, 1.0);
            alpha = mix(vs[i], vs[i + 1], f);
        }
    }
    return layer(p) * alpha;
}";

/// The built-in mask's effect for a box `rect` masked by a gradient of
/// `shape` through `stops` (their alpha; the first four).
fn mask_effect(rect: crate::Rect, shape: crate::GradientShape, stops: &[(f32, crate::Rgba)]) -> Effect {
    use crate::GradientShape;
    let (w, h) = (rect.width, rect.height);
    let shape = match shape {
        GradientShape::Linear(deg) => {
            let r = deg.to_radians();
            let (dx, dy) = (r.sin(), -r.cos());
            [0.0, dx, dy, (w * dx).abs() + (h * dy).abs()]
        }
        GradientShape::Radial { cx, cy } => {
            let (px, py) = (w * cx, h * cy);
            let far = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)].iter().map(|&(x, y): &(f32, f32)| ((x - px).powi(2) + (y - py).powi(2)).sqrt()).fold(0.0f32, f32::max);
            [1.0, px, py, far]
        }
        GradientShape::Conic { cx, cy, from } => [2.0, w * cx, h * cy, from.to_radians()],
    };
    // Four stops: the given (first four), the last repeated past 1.
    let mut four: Vec<(f32, f32)> = stops.iter().take(4).map(|&(t, c)| (t, c[3])).collect();
    let last = four.last().copied().unwrap_or((1.0, 1.0));
    while four.len() < 4 {
        four.push((last.0.max(1.0) + four.len() as f32, last.1));
    }
    MASK.with(|m| {
        Effect::new(m)
            .arg("shape", shape)
            .arg("a", [four[0].0, four[0].1, four[1].0, four[1].1])
            .arg("b", [four[2].0, four[2].1, four[3].0, four[3].1])
    })
}

impl<S: 'static> Input<S> {
    /// Keeps the argument `arg` (a `vec2<f32>`) of `node`'s effect at the
    /// pointer, in UI pixels from the box's top left, as it moves over the
    /// box: a lens, a light or a ripple that follows it.
    pub fn effect_follows_pointer(&mut self, node: NodeId, arg: &'static str) {
        self.on(node, Phase::Bubble, move |_, ui, e, _| {
            if let Event::PointerMove { x, y } = *e {
                let r = ui.rect(node);
                ui.set_effect_arg(node, arg, [x - r.x, y - r.y]);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_last_function_is_the_effect() {
        let s = Shader::new(
            "fn helper(x: f32) -> f32 { return x; }
             fn tint(p: vec2<f32>, color: vec4<f32>, time: f32, strength: f32, tint: vec4<f32>, size: vec2f) -> vec4<f32> { return color; }",
        );
        assert_eq!(s.error(), None);
        assert_eq!(s.function(), "tint");
        let kinds: Vec<_> = s.params().iter().map(|p| p.kind).collect();
        assert_eq!(
            kinds,
            [
                ParamKind::Position,
                ParamKind::Color,
                ParamKind::Time,
                ParamKind::Arg { components: 1, slot: 0 },
                ParamKind::Arg { components: 4, slot: 1 },
                ParamKind::Size
            ]
        );
        assert!(s.animated());
        let e = Effect::new(&s).arg("tint", [1.0, 0.5, 0.0, 1.0]).arg("strength", 0.25);
        assert_eq!(e.values()[0], [0.25, 0.0, 0.0, 0.0]);
        assert_eq!(e.values()[1], [1.0, 0.5, 0.0, 1.0]);
        assert!(!Effect::new(&s).set("nothing", 1.0));
    }

    #[test]
    fn misshapen_sources_say_why() {
        assert!(Shader::new("let x = 1;").error().unwrap().contains("no function"));
        assert!(Shader::new("fn f(p: vec2<f32>) -> f32 { return 1.0; }").error().unwrap().contains("vec4"));
        assert!(Shader::new("fn f(p: f32) -> vec4<f32> { return vec4<f32>(); }").error().unwrap().contains("position"));
        assert!(Shader::new("fn f(p: vec2<f32>, m: mat2x2<f32>) -> vec4<f32> { return vec4<f32>(); }").error().is_some());
        assert!(Shader::new("fn f(p: vec2<f32>, time: vec2<f32>) -> vec4<f32> { return vec4<f32>(); }").error().unwrap().contains("filled in"));
    }

    #[test]
    fn a_box_with_an_effect_is_wrapped_in_it_and_animates_while_it_shows() {
        use crate::scene::{Item, Scene};
        use crate::{style, FixedWidthText};
        let mut ui = Ui::new();
        let root = ui.root();
        let page = ui.add_box(root, style! { width: 200px; height: 100px; });
        let card = ui.add_box(page, style! { width: 100px; height: 50px; background: #f00; });
        ui.add_box(card, style! { width: 10px; height: 10px; background: #0f0; });
        let still = Shader::new("fn f(p: vec2<f32>, color: vec4<f32>) -> vec4<f32> { return color; }");
        ui.set_effect(card, Some(Effect::new(&still)));
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        let mut scene = Scene::default();
        ui.paint(&mut scene.recorder());
        let kinds: Vec<&str> = scene
            .items
            .iter()
            .map(|i| match i {
                Item::PushEffect { .. } => "push",
                Item::PopEffect => "pop",
                _ => "draw",
            })
            .collect();
        assert_eq!(kinds, ["push", "draw", "draw", "pop"], "the box and its child inside");
        assert!(!ui.needs_frame(), "a still effect asks for nothing");
        // Its arguments: a frame to show them.
        let tint = Shader::new("fn f(p: vec2<f32>, color: vec4<f32>, k: f32) -> vec4<f32> { return color * k; }");
        ui.set_effect(card, Some(Effect::new(&tint)));
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        assert!(ui.set_effect_arg(card, "k", 0.5));
        assert!(ui.needs_frame());
        ui.paint(&mut Scene::default().recorder());
        assert!(!ui.needs_frame());
        // One that takes the time keeps frames coming, but not once hidden.
        let pulse = Shader::new("fn f(p: vec2<f32>, color: vec4<f32>, time: f32) -> vec4<f32> { return color * sin(time); }");
        ui.set_effect(card, Some(Effect::new(&pulse)));
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        assert!(ui.needs_frame());
        ui.update_style(page, |s| s.display = crate::Display::None);
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        assert!(!ui.needs_frame());
    }

    #[test]
    fn the_macro_keeps_the_wgsl() {
        let s = crate::ui_shader! {
            fn ripple(p: vec2<f32>, size: vec2<f32>, centre: vec2<f32>, amount: f32) -> vec4<f32> {
                let d = p - centre * size;
                let k = 1.0 + amount * sin(length(d) * 0.2);
                return layer(centre * size + d * k);
            }
        };
        assert_eq!(s.error(), None);
        assert_eq!(s.function(), "ripple");
        assert_eq!(s.params().len(), 4);
        assert!(!s.animated());
    }

    #[test]
    fn the_budget_slows_and_holds_effects() {
        use crate::scene::Scene;
        use crate::{style, FixedWidthText};
        let mut ui = Ui::new();
        let root = ui.root();
        let pulse = Shader::new("fn f(p: vec2<f32>, color: vec4<f32>, time: f32) -> vec4<f32> { return color * sin(time); }");
        let big = ui.add_box(root, style! { width: 200px; height: 100px; });
        let mid = ui.add_box(root, style! { width: 100px; height: 100px; });
        let small = ui.add_box(root, style! { width: 20px; height: 20px; });
        for n in [big, mid, small] {
            ui.set_effect(n, Some(Effect::new(&pulse)));
        }
        ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
        let time_of = |ui: &Ui, n: NodeId| {
            ui.paint(&mut Scene::default().recorder());
            ui.effect_time(n)
        };
        // Small ones slowed, the rest at full rate.
        ui.set_animation_budget(AnimationBudget::UNLIMITED.small(48.0, 10.0));
        ui.step(0.37);
        assert_eq!(time_of(&ui, big), 0.37, "full rate");
        assert_eq!(time_of(&ui, small), 0.3, "the small one: stepped at 10 a second");
        // At most one: the largest runs, the rest (the small one too) held.
        ui.set_animation_budget(AnimationBudget::at_most(1));
        let held = time_of(&ui, mid);
        ui.step(0.5);
        assert_eq!(time_of(&ui, big), 0.87);
        assert_eq!(time_of(&ui, mid), held, "past the limit: held");
        assert_eq!(ui.next_effect_frame_in(), Some(0.0), "the big one runs every frame");
        // The big one gone, the next largest takes its place.
        ui.set_effect(big, None);
        ui.step(0.1);
        assert_eq!(time_of(&ui, mid), ui.clock as f32, "promoted to full rate");
        // None at full rate: all held, and no frames asked for.
        ui.set_animation_budget(AnimationBudget::at_most(0));
        ui.step(0.1);
        ui.paint(&mut Scene::default().recorder());
        assert!(!ui.needs_frame());
        assert_eq!(ui.next_effect_frame_in(), None);
        // Unlimited again: all at full rate.
        ui.set_animation_budget(AnimationBudget::UNLIMITED);
        assert_eq!(time_of(&ui, mid), ui.clock as f32);
    }
}
