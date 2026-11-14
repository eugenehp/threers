//! Flat or solid, for any box: [`Depth::Solid`] draws a box and what is in
//! it as a slab of glass, its edge rounded over (shaped by what it draws:
//! a rounded rectangle, a ring, a glyph), lit from above left, reflecting a studio at its rim
//! (more at a glancing angle, as glass does), its own content bent where
//! the edge curves away, and a soft shadow under it. [`Depth::Flat`] takes
//! that off again. Buttons, toggles, sliders, cards: anything.
//!
//! - [`Ui::set_depth`] on one box;
//! - [`Ui::set_subtree_depth`] on every control in a part of the UI (a
//!   button, a field, a slider, a tab, a progress bar…), leaving alone
//!   boxes drawn through shaders of their own;
//! - the [indicators](crate::indicators) take the same [`Depth`] in their
//!   style, where solid ones are glass tubes in true 3D.
//!
//! It is a shader effect, so it shows on the GPU painter; the 2D canvas and
//! the CPU overlay draw the box flat.
//!
//! It is true 3D: the box's drawing extruded into a slab, ray-traced in
//! perspective. Turn it — with a [motion](crate::motion), a tilt on hover,
//! or its own [3D transform](crate::depth) — and its walls show and its
//! front foreshortens as a solid's do: the turn is taken from the painter
//! and done in the shader. Unturned, its face is exactly where the box was
//! laid out.
//!
//! ```
//! use threers_ui::solid::Depth;
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let card = ui.add_box(root, style! { width: 160px; height: 90px; border-radius: 14px; background: #1d4ed8; });
//! ui.set_depth(card, Depth::Solid);
//! assert!(ui.effect(card).is_some());
//! ui.set_depth(card, Depth::Flat);
//! assert!(ui.effect(card).is_none());
//! ```

use crate::shader::{Effect, Shader};
use crate::tree::{NodeId, Ui};

/// How deep something looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Depth {
    /// Flat, as drawn.
    #[default]
    Flat,
    /// Solid: lit, bevelled glass (see the [module docs](self)).
    Solid,
}

/// Which face of a solid box lies on the page it was laid out on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DepthAlign {
    /// Its front: drawn exactly as laid out, its depth behind (set into
    /// the page).
    #[default]
    Front,
    /// Its middle: half in front of the page, half behind.
    Center,
    /// Its back: standing out of the page toward the viewer (a little
    /// larger, in perspective, its walls showing more as it turns).
    Back,
}

/// How a solid box is made: its depth, which face is on the page, how far
/// it is lifted off it, its bevel and its gloss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidStyle {
    /// Its depth in UI pixels; `None` by its size (a quarter of its
    /// shorter side, 4 to 18).
    pub thickness: Option<f32>,
    pub align: DepthAlign,
    /// Lifted toward the viewer, in UI pixels (its shadow falls further).
    pub lift: f32,
    /// How wide its rounded edge is, in UI pixels; `None` by its size.
    pub bevel: Option<f32>,
    /// How much it reflects, 0 (matte) to 1 (glass).
    pub gloss: f32,
    /// How dark its shadow is, 0 for none.
    pub shadow: f32,
}

impl Default for SolidStyle {
    fn default() -> Self {
        Self {
            thickness: None,
            align: DepthAlign::Front,
            lift: 0.0,
            bevel: None,
            gloss: 1.0,
            shadow: 1.0,
        }
    }
}

impl SolidStyle {
    pub fn thickness(self, px: f32) -> Self {
        Self {
            thickness: Some(px.max(0.5)),
            ..self
        }
    }
    pub fn align(self, align: DepthAlign) -> Self {
        Self { align, ..self }
    }
    pub fn lift(self, px: f32) -> Self {
        Self { lift: px, ..self }
    }
    pub fn bevel(self, px: f32) -> Self {
        Self {
            bevel: Some(px.max(0.0)),
            ..self
        }
    }
    pub fn gloss(self, gloss: f32) -> Self {
        Self {
            gloss: gloss.clamp(0.0, 1.0),
            ..self
        }
    }
    pub fn shadow(self, shadow: f32) -> Self {
        Self {
            shadow: shadow.max(0.0),
            ..self
        }
    }
}

/// How far a solid box's shadow reaches past it, in UI pixels.
const SHADOW: f32 = 12.0;

const SOURCE: &str = concat!(
    include_str!("studio.wgsl"),
    r#"

// What the box drew at `q` (the slab's own coordinates: x right, y up,
// from its middle `c`), and how much of it.
fn solid_at(q: vec2<f32>, c: vec2<f32>) -> vec4<f32> {
    return layer(vec2<f32>(q.x + c.x, c.y - q.y));
}

// Which way the shape's edge faces at `q`, in the slab's coordinates (out
// of the shape), as steep as the edge is near: the coverage, blurred over
// `bevel`, sampled on two rings round it.
fn solid_slope(q: vec2<f32>, c: vec2<f32>, bevel: f32) -> vec2<f32> {
    var g = vec2<f32>(0.0);
    for (var i = 0; i < 8; i = i + 1) {
        let ang = f32(i) * 0.7853982;
        let d = vec2<f32>(cos(ang), sin(ang));
        g = g + d * (solid_at(q + d * bevel * 0.5, c).a + solid_at(q + d * bevel, c).a);
    }
    return -g * 0.32;
}

// Light on a surface of colour `col` with normal `n`, seen along `rd`
// (both in the view), glossy by `gloss`.
fn solid_light(col: vec3<f32>, n: vec3<f32>, rd: vec3<f32>, gloss: f32) -> vec3<f32> {
    let ndv = clamp(-dot(n, rd), 0.0, 1.0);
    let fres = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    let sun = normalize(vec3<f32>(-0.4, 0.8, 0.45));
    let nh = max(dot(n, normalize(sun - rd)), 0.0);
    let spec = pow(nh, 160.0) * 1.6 + pow(nh, 24.0) * 0.06;
    let diffuse = 0.68 + 0.3 * dot(n, sun);
    return col * diffuse * (1.0 - fres * gloss) + (studio_env(reflect(rd, n)) * fres + vec3<f32>(spec)) * gloss;
}

// Where a ray from `ro` along `rd` (the slab's own space) first meets the
// slab `hz` thick: 0 nothing, 1 its front, 2 a wall, 3 its back; where; and
// what the box drew there (its coverage the front's or back's softness).
struct SolidHit {
    kind: i32,
    q: vec3<f32>,
    entry: vec4<f32>,
}

fn solid_trace(ro: vec3<f32>, rd: vec3<f32>, hz: f32, c: vec2<f32>, steps: i32) -> SolidHit {
    var h: SolidHit;
    let dz = select(rd.z, -0.000001, abs(rd.z) < 0.000001);
    let tf = (hz - ro.z) / dz;
    let tb = (-hz - ro.z) / dz;
    let tn = min(tf, tb);
    let tx = max(tf, tb);
    h.q = ro + rd * tn;
    h.entry = solid_at(h.q.xy, c);
    h.kind = 0;
    if (h.entry.a > 0.02) {
        h.kind = select(3, 1, tf <= tb);
        return h;
    }
    // The first point inside the shape between the faces: a wall.
    var prev = tn;
    for (var i = 1; i <= steps; i = i + 1) {
        let ti = mix(tn, tx, f32(i) / f32(steps));
        if (solid_at((ro + rd * ti).xy, c).a >= 0.5) {
            var lo = prev;
            var hi = ti;
            for (var j = 0; j < 5; j = j + 1) {
                let mid = (lo + hi) * 0.5;
                if (solid_at((ro + rd * mid).xy, c).a >= 0.5) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            h.q = ro + rd * hi;
            h.kind = 2;
            return h;
        }
        prev = ti;
    }
    return h;
}

// The box as a slab: what it drew, extruded `pose.z` thick (0: by its
// size), turned by `pose.x` about its vertical (the right side away) and
// `pose.y` about its horizontal (the top away), seen from `pose.w` away
// in perspective, and ray-traced: the front face, the walls round the
// shape, or the back. Unturned, the front face lies exactly where the box
// was laid out. A pixel on the outline of a wall is traced four times
// over, for a smooth edge; `room` is how far past the box it may draw.
fn solid_box(p: vec2<f32>, size: vec2<f32>, look: vec4<f32>, pose: vec4<f32>, room: f32, place: vec2<f32>) -> vec4<f32> {
    let c = size * 0.5;
    let th = select(clamp(min(size.x, size.y) * 0.25, 4.0, 18.0), pose.z, pose.z > 0.0);
    let hz = th * 0.5;
    let persp = select(800.0, pose.w, pose.w > 1.0);
    let bevel = select(clamp(min(size.x, size.y) * 0.18, 2.0, 12.0), look.x, look.x > 0.0);
    let a = pose.y;
    let b = pose.x;
    let rx = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, cos(a), -sin(a)), vec3<f32>(0.0, sin(a), cos(a)));
    let ry = mat3x3<f32>(vec3<f32>(cos(b), 0.0, -sin(b)), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(sin(b), 0.0, cos(b)));
    let m = ry * rx;
    let mt = transpose(m);
    // Which of its faces lies on the page (`place.x`: 1 its front, 0 its
    // middle, -1 its back) and how far it is lifted off it (`place.y`):
    // where the screen's plane cuts the slab, which is where it is drawn
    // as laid out.
    let screen_z = place.x * hz - place.y;
    let eye = vec3<f32>(0.0, 0.0, screen_z + persp);
    let ro = mt * eye;
    // Steps through the walls: as many as the slab is pixels thick, seen
    // edge on at most (a thin slab needs few).
    let steps = clamp(i32(th * 1.2), 6, 24);
    let rdv = normalize(vec3<f32>(p.x - c.x, c.y - p.y, screen_z) - eye);
    var h = solid_trace(ro, mt * rdv, hz, c, steps);
    var rd_shade = rdv;
    // On a wall's outline (it hits and a neighbour does not, or the other
    // way): four traces across the pixel, for how much of it is covered,
    // and one of them to shade if the middle missed.
    let edge = fwidth(f32(min(h.kind, 2) > 0)) > 0.0;
    var cover = select(0.0, 1.0, h.kind == 2);
    if (h.kind == 1 || h.kind == 3) {
        cover = h.entry.a;
    }
    if (edge && h.kind != 1) {
        var n = 0.0;
        for (var i = 0; i < 4; i = i + 1) {
            let o = vec2<f32>(select(-0.25, 0.25, (i & 1) == 1), select(-0.25, 0.25, i >= 2));
            let rds = normalize(vec3<f32>(p.x + o.x - c.x, c.y - p.y - o.y, screen_z) - eye);
            let hs = solid_trace(ro, mt * rds, hz, c, steps);
            if (hs.kind > 0) {
                n = n + select(1.0, hs.entry.a, hs.kind != 2);
                if (h.kind == 0) {
                    h = hs;
                    rd_shade = rds;
                }
            }
        }
        cover = n / 4.0;
    }
    let q = h.q;
    let gloss = look.z;
    var body = vec4<f32>(0.0);
    if (h.kind == 1) {
        // The front: its content (bent where the edge curves away), the
        // edge rounded over.
        let slope = solid_slope(q.xy, c, bevel);
        let e = clamp(length(slope), 0.0, 1.0);
        let n = normalize(m * normalize(vec3<f32>(slope, 1.0)));
        let src = solid_at(q.xy - slope * look.y, c);
        let col = src.rgb / max(src.a, 0.0001);
        body = vec4<f32>(min(solid_light(col, n, rd_shade, gloss * (0.4 + 0.6 * e)), vec3<f32>(1.0)) * cover, cover);
    } else if (h.kind == 2) {
        // A wall: the edge's colour, a little darker, lit by which way it
        // faces.
        let slope = solid_slope(q.xy, c, 1.5);
        let out = normalize(slope + vec2<f32>(0.000001, 0.0));
        let n = normalize(m * vec3<f32>(out, 0.0));
        let src = solid_at(q.xy - out * 1.5, c);
        let col = src.rgb / max(src.a, 0.0001) * 0.78;
        body = vec4<f32>(min(solid_light(col, n, rd_shade, gloss * 0.6), vec3<f32>(1.0)) * cover, cover);
    } else if (h.kind == 3) {
        // The back: the same shape, plain and darker.
        let col = h.entry.rgb / max(h.entry.a, 0.0001) * 0.5;
        let n = normalize(m * vec3<f32>(0.0, 0.0, -1.0));
        body = vec4<f32>(min(solid_light(col, n, rd_shade, gloss * 0.5), vec3<f32>(1.0)) * cover, cover);
    }
    // Its shadow on the page just behind it: from where this pixel's ray
    // meets the page, toward a light above and in front, through the
    // turned slab's middle, softened; gone by the edge of its room.
    // The page: the plane it was laid out on, where the screen plane is
    // unlifted; a slab whose front is on it is set into it, its shadow just
    // behind.
    let page = min(place.x * hz, -hz) - select(0.0, 4.0, place.x > 0.5) - place.y;
    let sv = eye + rdv * ((page - eye.z) / rdv.z);
    let ld = normalize(vec3<f32>(0.0, 0.35, 1.0));
    let so = mt * sv;
    let sd = mt * ld;
    let st = -so.z / select(sd.z, 0.000001, abs(sd.z) < 0.000001);
    let sq = (so + sd * st).xy;
    let soft = 3.0 + hz * 0.3;
    var shade = solid_at(sq, c).a;
    for (var i = 0; i < 8; i = i + 1) {
        let ang = f32(i) * 0.7853982;
        shade = shade + solid_at(sq + vec2<f32>(cos(ang), sin(ang)) * soft, c).a;
    }
    let outside = max(max(-p.x, p.x - size.x), max(-p.y, p.y - size.y));
    let keep = 1.0 - smoothstep(room * 0.5, room, outside);
    let shadow = 0.3 * look.w * shade / 9.0 * keep;
    return body + vec4<f32>(0.0, 0.0, 0.0, shadow * (1.0 - body.a));
}
"#
);

thread_local! {
    static SHADER: Shader = Shader::new(SOURCE);
}

/// The solid box's shader (for tests and to translate).
pub fn shader() -> Shader {
    SHADER.with(Shader::clone)
}

/// A solid box's style, turn and room, as its shader was last given them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidState {
    pub style: SolidStyle,
    /// About its vertical, about its horizontal, its thickness (0: by its
    /// size), the viewer's distance.
    pub pose: [f32; 4],
    /// How far past its box it draws.
    pub overflow: f32,
}

fn solid_effect(state: SolidState) -> Effect {
    let st = state.style;
    let place = match st.align {
        DepthAlign::Front => 1.0,
        DepthAlign::Center => 0.0,
        DepthAlign::Back => -1.0,
    };
    Effect::new(&shader())
        .arg("look", [st.bevel.unwrap_or(0.0), 3.0, st.gloss, st.shadow])
        .arg("pose", state.pose)
        .arg("room", state.overflow)
        .arg("place", [place, st.lift])
        .overflow(state.overflow)
}

impl Ui {
    /// `node`'s turn (from its 3D transform: its own, a motion's, a tilt
    /// on hover) and the room it needs turned, walls and all.
    fn solid_state(&self, node: NodeId, style: SolidStyle) -> SolidState {
        let r = self.rect(node);
        let t = self.transform3d(node).unwrap_or_default();
        let th = style
            .thickness
            .unwrap_or((r.width.min(r.height) * 0.25).clamp(4.0, 18.0));
        // Turned, its near edge grows by its perspective: room for that,
        // its walls and its shadow.
        let turned = t.rotate_x.sin().abs().max(t.rotate_y.sin().abs());
        let grow = r.width.max(r.height)
            * 0.5
            * turned
            * (r.width.max(r.height) / t.perspective.max(1.0)).min(1.0);
        // Standing out of the page, it is nearer and so larger.
        let out = match style.align {
            DepthAlign::Front => 0.0,
            DepthAlign::Center => th * 0.5,
            DepthAlign::Back => th,
        } + style.lift.max(0.0);
        let near = r.width.max(r.height) * 0.5 * out / t.perspective.max(1.0);
        SolidState {
            style,
            pose: [
                t.rotate_y,
                t.rotate_x,
                style.thickness.unwrap_or(0.0),
                t.perspective,
            ],
            overflow: (SHADOW + th + grow + near + style.lift.max(0.0) * 0.5).ceil(),
        }
    }

    /// Draws `node` flat or solid (see the [module docs](self)). Solid
    /// takes the box's effect slot: a box with a shader of its own keeps
    /// it, and this does nothing.
    pub fn set_depth(&mut self, node: NodeId, depth: Depth) {
        match depth {
            Depth::Flat => {
                if self.env.solids.remove(&node).is_some() {
                    self.set_effect(node, None);
                }
            }
            Depth::Solid => {
                let style = self
                    .env
                    .solids
                    .get(&node)
                    .map_or_else(SolidStyle::default, |s| s.style);
                self.set_solid(node, style);
            }
        }
    }

    /// Draws `node` solid, made as `style` says (see [`SolidStyle`]): its
    /// depth, which face is on the page, how far it is lifted. A box with a
    /// shader of its own keeps it, and this does nothing.
    pub fn set_solid(&mut self, node: NodeId, style: SolidStyle) {
        if self.effect(node).is_some() && !self.env.solids.contains_key(&node) {
            return;
        }
        let state = self.solid_state(node, style);
        self.env.solids.insert(node, state);
        self.set_effect(node, Some(solid_effect(state)));
    }

    /// How `node` is made solid, if it is.
    pub fn solid_style(&self, node: NodeId) -> Option<SolidStyle> {
        self.env.solids.get(&node).map(|s| s.style)
    }

    /// How deep `node` is drawn.
    pub fn depth_of(&self, node: NodeId) -> Depth {
        if self.env.solids.contains_key(&node) {
            Depth::Solid
        } else {
            Depth::Flat
        }
    }

    /// Draws every control under `root` (and `root`, if it is one) flat or
    /// solid: the nodes whose role can be pressed, edited, chosen or
    /// moved, and progress bars. A box drawn through a shader of its own,
    /// or with one inside it, is left as it is.
    pub fn set_subtree_depth(&mut self, root: NodeId, depth: Depth) {
        let mut stack = vec![root];
        let mut found = vec![];
        while let Some(n) = stack.pop() {
            let role = self.semantics(n).map(|s| s.role);
            if role.is_some_and(|r| crate::spatial::is_control(r) || r == crate::Role::ProgressBar)
            {
                found.push(n);
            } else {
                stack.extend(self.children(n).iter().copied());
            }
        }
        for n in found {
            let own = self.effect(n).is_some() && !self.env.solids.contains_key(&n);
            if !own && !self.has_effect_inside(n) {
                self.set_depth(n, depth);
            }
        }
    }

    /// The solid boxes kept up with their turns and sizes.
    pub(crate) fn step_solids(&mut self) {
        if self.env.solids.is_empty() {
            return;
        }
        let nodes: Vec<(NodeId, SolidState)> =
            self.env.solids.iter().map(|(&n, &s)| (n, s)).collect();
        for (n, was) in nodes {
            if !self.contains(n) {
                self.env.solids.remove(&n);
                continue;
            }
            let now = self.solid_state(n, was.style);
            if now == was {
                continue;
            }
            self.env.solids.insert(n, now);
            if now.overflow != was.overflow {
                self.set_effect(n, Some(solid_effect(now)));
            } else {
                self.set_effect_arg(n, "pose", now.pose);
            }
        }
    }

    /// Whether a box under `node` has an effect.
    fn has_effect_inside(&self, node: NodeId) -> bool {
        let mut stack: Vec<NodeId> = self.children(node).to_vec();
        while let Some(n) = stack.pop() {
            if self.effect(n).is_some() {
                return true;
            }
            stack.extend(self.children(n).iter().copied());
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{style, Role, Semantics};

    #[test]
    fn the_shader_is_an_effect() {
        let s = shader();
        assert_eq!(s.error(), None);
        assert!(!s.animated(), "a still box asks for nothing");
    }

    #[test]
    fn a_subtree_turns_solid_but_its_own_shaders_stay() {
        let mut ui = Ui::new();
        let root = ui.root();
        let button = ui.add_box(
            root,
            style! { width: 80px; height: 30px; border-radius: 8px; },
        );
        ui.set_semantics(button, Semantics::new(Role::Button).label("Go"));
        let label = ui.add_box(root, style! { width: 80px; height: 30px; });
        let fancy = ui.add_box(root, style! { width: 80px; height: 30px; });
        ui.set_semantics(fancy, Semantics::new(Role::Button).label("Fancy"));
        let own =
            Shader::new("fn f(p: vec2<f32>, color: vec4<f32>) -> vec4<f32> { return color; }");
        ui.set_effect(fancy, Some(Effect::new(&own)));
        ui.set_subtree_depth(root, Depth::Solid);
        assert_eq!(ui.depth_of(button), Depth::Solid);
        assert_eq!(ui.depth_of(label), Depth::Flat, "not a control");
        assert_eq!(ui.effect(fancy).unwrap().shader, own, "its own shader kept");
        ui.set_subtree_depth(root, Depth::Flat);
        assert_eq!(ui.depth_of(button), Depth::Flat);
        assert!(ui.effect(button).is_none());
        assert_eq!(ui.effect(fancy).unwrap().shader, own);
    }
}
