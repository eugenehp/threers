//! What flows down a UI to everything in it:
//!
//! - **Values** of any type set on a node and read by anything inside it
//!   ([`Ui::set_environment`], [`Ui::environment`]), the nearest setting
//!   winning, and an app-wide [global](crate::globals) under them all.
//! - **Colours by purpose** ([`Token`]): a node styled with
//!   [`Ui::token_style`] takes the colour its purpose has in the appearance
//!   showing, from the [`Tokens`] the UI is given — so components carry no
//!   colours of their own, and a new palette restyles everything.
//! - **Text size** ([`Ui::set_text_scale`]): every text node's font and
//!   line height scaled, as browsers' text zoom does.
//! - **Reduced motion** ([`Ui::set_reduce_motion`]): animations of place
//!   and size jump to their end; fades still run.
//!
//! ```
//! use threers_ui::environment::Token;
//! use threers_ui::states::Appearance;
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let card = ui.add_box(root, style! { padding: 8px; });
//! ui.token_style(card, &[("background", Token::Surface), ("color", Token::Text)]);
//! ui.set_system_appearance(Appearance::Light);
//! assert_eq!(ui.style(card).background, Some(ui.tokens().light.surface));
//! ui.set_system_appearance(Appearance::Dark);
//! assert_eq!(ui.style(card).background, Some(ui.tokens().dark.surface));
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::rc::Rc;

use crate::animation::Property;
use crate::states::Appearance;
use crate::tree::{NodeId, Ui};
use crate::{Edges, Length, NodeKind, Rgba};

/// A colour's purpose. See [`TokenSet`] for what each is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Token {
    Background,
    Surface,
    Raised,
    Text,
    SecondaryText,
    Border,
    Separator,
    Accent,
    OnAccent,
    Hover,
    Destructive,
    /// Behind white text: a filled destructive button.
    DestructiveFill,
    Success,
    Focus,
}

/// The colours of one appearance, by purpose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenSet {
    /// Behind everything.
    pub background: Rgba,
    /// Panels and cards on it.
    pub surface: Rgba,
    /// What floats over the rest: menus, popovers, dialogs, notices.
    pub raised: Rgba,
    pub text: Rgba,
    /// Captions, hints, shortcut labels.
    pub secondary_text: Rgba,
    pub border: Rgba,
    pub separator: Rgba,
    /// The highlight: selection, the default button, a switch that is on.
    pub accent: Rgba,
    /// Text on the accent.
    pub on_accent: Rgba,
    /// Under the pointer.
    pub hover: Rgba,
    pub destructive: Rgba,
    /// The destructive colour as a fill under [`on_accent`](Self::on_accent)
    /// text: darker where `destructive` is a light red for text on dark.
    pub destructive_fill: Rgba,
    pub success: Rgba,
    /// The focus ring.
    pub focus: Rgba,
}

impl TokenSet {
    pub fn get(&self, token: Token) -> Rgba {
        match token {
            Token::Background => self.background,
            Token::Surface => self.surface,
            Token::Raised => self.raised,
            Token::Text => self.text,
            Token::SecondaryText => self.secondary_text,
            Token::Border => self.border,
            Token::Separator => self.separator,
            Token::Accent => self.accent,
            Token::OnAccent => self.on_accent,
            Token::Hover => self.hover,
            Token::Destructive => self.destructive,
            Token::DestructiveFill => self.destructive_fill,
            Token::Success => self.success,
            Token::Focus => self.focus,
        }
    }
}

fn hex(v: u32) -> Rgba {
    let c = |s: u32| ((v >> s) & 0xff) as f32 / 255.0;
    [c(16), c(8), c(0), 1.0]
}

/// The colours of both appearances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tokens {
    pub light: TokenSet,
    pub dark: TokenSet,
}

impl Default for Tokens {
    fn default() -> Self {
        Self {
            light: TokenSet {
                background: hex(0xf2f4f7),
                surface: hex(0xffffff),
                raised: hex(0xffffff),
                text: hex(0x1f2329),
                secondary_text: hex(0x5b6270),
                border: hex(0xd5d9e0),
                separator: hex(0xe2e5ea),
                accent: hex(0x2563eb),
                on_accent: [0.99, 0.99, 0.99, 1.0],
                hover: [0.5, 0.54, 0.63, 0.18],
                // 5.9:1 on the page's grey as text; under white text, the
                // fill's 4.7:1.
                destructive: hex(0xb91c1c),
                destructive_fill: hex(0xdc2626),
                success: hex(0x10b981),
                focus: hex(0x93c5fd),
            },
            dark: TokenSet {
                background: hex(0x12151b),
                surface: hex(0x1b1f27),
                raised: hex(0x1f2430),
                text: hex(0xe5e7eb),
                secondary_text: hex(0x979eaf),
                border: hex(0x3a4150),
                separator: hex(0x2a303c),
                accent: hex(0x2563eb),
                on_accent: [0.99, 0.99, 0.99, 1.0],
                hover: [0.5, 0.54, 0.63, 0.28],
                // 5.6:1 or more as text on every dark surface; the fill
                // darker, 4.7:1 under white text (#ef4444 was 3.7:1).
                destructive: hex(0xf87171),
                destructive_fill: hex(0xdc2626),
                success: hex(0x10b981),
                focus: hex(0x93c5fd),
            },
        }
    }
}

/// The environment's state in a [`Ui`].
#[derive(Clone, Default)]
pub(crate) struct Env {
    pub tokens: Tokens,
    /// Values set on nodes, by type.
    pub values: HashMap<NodeId, HashMap<TypeId, Rc<dyn Any>>>,
    /// 1 when unset (`0.0` reads as 1).
    pub text_scale: f32,
    /// Text nodes' own font size and line height, before the scale.
    pub unscaled: HashMap<NodeId, (f32, Option<f32>)>,
    pub reduce_motion: bool,
    /// The smallest target the audit allows, in UI pixels; 0 when unset
    /// (WCAG's 24).
    pub min_target: f32,
    /// Bold Text and Increase Contrast (see [`legibility`](crate::legibility)),
    /// and each text node's weight before Bold Text.
    pub bold_text: bool,
    pub increase_contrast: bool,
    pub unbolded: HashMap<NodeId, crate::FontWeight>,
    /// Parts of the UI in a theme of their own (see [`theme`](crate::theme)).
    pub subtree_themes: HashMap<NodeId, crate::states::Theme>,
    /// Names nodes are linked to by (see [`anchors`](crate::anchors)).
    pub anchors: HashMap<NodeId, String>,
    /// Gauges gliding to a new value.
    pub glides: Vec<crate::charts::Glide>,
    /// Boxes whose spacing and fixed sizes follow the text size, with
    /// their own (see [`Ui::scale_with_text`]).
    pub scaled: HashMap<NodeId, Metrics>,
    /// The window's edges the system covers (see [`Ui::set_safe_area`]).
    pub safe_area: Edges,
    /// Boxes padded clear of them: their own padding and which edges.
    pub safe_nodes: HashMap<NodeId, (Edges, [bool; 4])>,
    /// Subtrees drawn as placeholders (see [`Ui::set_redacted`]).
    pub redacted: std::collections::HashSet<NodeId>,
    pub direction: LayoutDirection,
    /// Boxes turned and lifted in 3D (see [`depth`](crate::depth)).
    pub transforms3d: HashMap<NodeId, crate::depth::Transform3d>,
    /// Boxes moving on their own, and their transforms without the motion
    /// (see [`motion`](crate::motion)).
    pub motions: HashMap<NodeId, (crate::motion::MotionSpec, Option<crate::depth::Transform3d>)>,
    /// Named styles and the nodes that have them (see [`patch`](crate::patch)).
    pub classes: crate::patch::Classes,
    /// Boxes drawn solid (see [`solid`](crate::solid)), and the turn and
    /// room their shader was last given.
    pub solids: HashMap<NodeId, crate::solid::SolidState>,
    /// Layers moving by their depth when the box around them tilts.
    pub parallax: HashMap<NodeId, f32>,
    /// Strips whose items turn away from their middle.
    pub carousels: Vec<NodeId>,
    /// Where light falls from, for glare and shading.
    pub light: crate::depth::Light,
    /// How the device is held: its tilt forward and sideways, in degrees.
    pub device_tilt: (f32, f32),
    /// Lists whose rows roll over a drum.
    pub cylinders: Vec<NodeId>,
    /// Boxes mirrored below themselves: strength and height shown.
    pub reflections: HashMap<NodeId, (f32, f32)>,
    /// Boxes drawn through custom shaders.
    pub effects: HashMap<NodeId, crate::shader::Effect>,
    /// How much the effects may animate (see [`Ui::set_animation_budget`](crate::Ui::set_animation_budget)).
    pub budget: crate::shader::AnimationBudget,
    /// The animated effects' rates under it, frames a second (infinite:
    /// every frame; 0: held), as the last step planned them.
    pub effect_rates: HashMap<NodeId, f32>,
    /// The time each animated effect was last drawn at.
    pub effect_drawn: std::cell::RefCell<HashMap<NodeId, f32>>,
    /// What boxes show changed, not where they are: paint again.
    pub repaint: std::cell::Cell<bool>,
    /// Heights set by `aspect-ratio`, by box (so that a height the app set
    /// is told from one that follows the width).
    pub aspect_heights: HashMap<NodeId, f32>,
    /// Scrolling boxes whose items change by where they are in view.
    pub scroll_transitions: Vec<(NodeId, crate::scroll_transition::ScrollTransition)>,
    /// Keyframes playing (see [`keyframes`](crate::keyframes)).
    pub keyframes: Vec<crate::keyframes::Playing>,
    /// Work off the UI thread and what to do when it is done.
    pub jobs: Rc<std::cell::RefCell<crate::jobs::Jobs>>,
    /// Painting a reflection now (it does not reflect again).
    pub reflecting: std::cell::Cell<bool>,
}

/// A box's spacing and fixed sizes as given, before the text scale.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Metrics {
    padding: Edges,
    gap: f32,
    row_gap: Option<f32>,
    width: Length,
    height: Length,
    min_width: Length,
    min_height: Length,
    radius: Length,
}

/// Which way a UI reads (see [`Ui::set_layout_direction`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutDirection {
    #[default]
    LeftToRight,
    /// Arabic, Hebrew, Persian: rows start at the right.
    RightToLeft,
}

impl std::fmt::Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env").field("text_scale", &self.text_scale).field("reduce_motion", &self.reduce_motion).finish()
    }
}

/// A colour as CSS, for a declaration.
pub(crate) fn css_color(c: Rgba) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("rgba({},{},{},{})", b(c[0]), b(c[1]), b(c[2]), c[3])
}

impl Ui {
    /// Sets `value` for `node` and everything inside it (until something
    /// inside sets its own).
    pub fn set_environment<T: 'static>(&mut self, node: NodeId, value: T) {
        self.env.values.retain(|n, _| self.nodes.get(n.0).is_some_and(|x| !x.removed && x.generation == n.1));
        self.env.values.entry(node).or_default().insert(TypeId::of::<T>(), Rc::new(value));
    }

    /// The `T` set on `node` itself, if one is (not inherited).
    pub fn own_environment<T: Clone + 'static>(&self, node: NodeId) -> Option<T> {
        self.env.values.get(&node).and_then(|m| m.get(&TypeId::of::<T>())).and_then(|v| v.downcast_ref::<T>().cloned())
    }

    /// The `T` in effect at `node`: set on it or its nearest ancestor that
    /// has one, else the app-wide [global](Self::global)'s value.
    pub fn environment<T: Clone + 'static>(&self, node: NodeId) -> Option<T> {
        let mut at = Some(node);
        while let Some(n) = at.filter(|&n| self.contains(n)) {
            if let Some(v) = self.env.values.get(&n).and_then(|m| m.get(&TypeId::of::<T>())) {
                return v.downcast_ref::<T>().cloned();
            }
            at = self.parent(n);
        }
        Some(self.global::<T>()?.read().clone())
    }

    /// The colours by purpose.
    pub fn tokens(&self) -> &Tokens {
        &self.env.tokens
    }

    /// New colours by purpose: everything styled by token takes them.
    pub fn set_tokens(&mut self, tokens: Tokens) {
        if self.env.tokens != tokens {
            self.env.tokens = tokens;
            self.restyle_all();
        }
    }

    /// `token`'s colour in the appearance showing.
    pub fn token(&self, token: Token) -> Rgba {
        self.token_in(self.appearance(), token)
    }

    /// `token`'s colour in `appearance` (with the contrast asked for).
    pub fn token_in(&self, appearance: Appearance, token: Token) -> Rgba {
        let set = match appearance {
            Appearance::Light => &self.env.tokens.light,
            Appearance::Dark => &self.env.tokens.dark,
        };
        let c = set.get(token);
        if self.env.increase_contrast {
            crate::legibility::stronger(token, c, set.get(Token::Text))
        } else {
            c
        }
    }

    /// Gives `node` colours by purpose: each CSS property (`background`,
    /// `color`, `border-color`, …) the token's colour in the appearance
    /// showing, kept as the appearance and the tokens change. They go over
    /// its own style and its appearance styles, under its state styles. A
    /// property given again takes its new token.
    pub fn token_style(&mut self, node: NodeId, props: &[(&str, Token)]) {
        let tokens = &mut self.states_mut(node).tokens;
        for (p, t) in props {
            // A property given again takes the new token.
            tokens.retain(|(q, _)| q != p);
            tokens.push((p.to_string(), *t));
        }
        self.restyle(node);
    }

    /// Takes back the token `node` had for `property`: its own style's value
    /// shows again.
    pub fn clear_token_style(&mut self, node: NodeId, property: &str) {
        if let Some(s) = self.nodes[node.0].states.as_mut() {
            s.tokens.retain(|(p, _)| p != property);
        }
        self.restyle(node);
    }

    /// The text size, as a multiple of each text node's own (1 by default).
    pub fn text_scale(&self) -> f32 {
        if self.env.text_scale > 0.0 { self.env.text_scale } else { 1.0 }
    }

    /// Scales every text node's font size and line height by `scale`
    /// (a user's larger text). Text added later is scaled as it is added;
    /// a text node's own style given after should be given unscaled
    /// through [`set_style`](Self::set_style).
    pub fn set_text_scale(&mut self, scale: f32) {
        let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        if scale == self.text_scale() {
            return;
        }
        self.env.text_scale = scale;
        let texts: Vec<NodeId> =
            (0..self.nodes.len()).filter(|&i| !self.nodes[i].removed && matches!(self.nodes[i].kind, NodeKind::Text(_))).map(|i| self.id_at(i)).collect();
        for id in texts {
            self.scale_text(id);
        }
        self.apply_scaled();
    }

    /// Sets a text node's font size and line height from its own and the
    /// scale.
    pub(crate) fn scale_text(&mut self, id: NodeId) {
        let mine = (self.style(id).font_size, self.style(id).line_height);
        let own = *self.env.unscaled.entry(id).or_insert(mine);
        let k = self.text_scale();
        let (size, line) = (own.0 * k, own.1.map(|l| l * k));
        if self.style(id).font_size != size || self.style(id).line_height != line {
            let st = self.style_mut(id);
            st.font_size = size;
            st.line_height = line;
        }
    }

    /// A text node's style was given anew: that is its own size now.
    pub(crate) fn rescale_text_after_set(&mut self, id: NodeId) {
        if self.text_scale() != 1.0 {
            let s = self.style(id);
            self.env.unscaled.insert(id, (s.font_size, s.line_height));
            self.scale_text(id);
        }
    }

    pub fn reduce_motion(&self) -> bool {
        self.env.reduce_motion
    }

    /// Animations of place, size, turn and scrolling jump to their end, for
    /// users who ask for less motion; fades and colour changes still run.
    pub fn set_reduce_motion(&mut self, on: bool) {
        self.env.reduce_motion = on;
    }

    /// `value` (a padding, a size) at the text scale: for spacing set in
    /// code that should grow with the text.
    pub fn scaled(&self, value: f32) -> f32 {
        value * self.text_scale()
    }

    /// Makes `node`'s padding, gaps, fixed sizes and corner radius follow
    /// the text size, as its text does: at twice the text, twice the
    /// padding and a row twice as tall, so larger text still fits. Its
    /// sizes now are its own; one given later through
    /// [`set_style`](Self::set_style) is taken as scaled.
    pub fn scale_with_text(&mut self, node: NodeId) {
        let s = self.style(node);
        let k = self.text_scale();
        let own = |l: Length| if let Length::Px(v) = l { Length::Px(v / k) } else { l };
        let m = Metrics {
            padding: Edges { top: s.padding.top / k, right: s.padding.right / k, bottom: s.padding.bottom / k, left: s.padding.left / k },
            gap: s.gap / k,
            row_gap: s.row_gap.map(|g| g / k),
            width: own(s.width),
            height: own(s.height),
            min_width: own(s.min_width),
            min_height: own(s.min_height),
            radius: own(s.border_radius),
        };
        self.env.scaled.insert(node, m);
    }

    /// [`scale_with_text`](Self::scale_with_text) for `node` and every box
    /// inside it now.
    pub fn scale_tree_with_text(&mut self, node: NodeId) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if matches!(self.kind(n), NodeKind::Box) {
                self.scale_with_text(n);
            }
            stack.extend(self.children(n).iter().copied());
        }
    }

    fn apply_scaled(&mut self) {
        let k = self.text_scale();
        let nodes: Vec<(NodeId, Metrics)> = self.env.scaled.iter().map(|(n, m)| (*n, *m)).collect();
        for (n, m) in nodes {
            if !self.contains(n) {
                self.env.scaled.remove(&n);
                continue;
            }
            let px = |l: Length| if let Length::Px(v) = l { Length::Px(v * k) } else { l };
            self.update_style(n, |s| {
                s.padding = Edges { top: m.padding.top * k, right: m.padding.right * k, bottom: m.padding.bottom * k, left: m.padding.left * k };
                s.gap = m.gap * k;
                s.row_gap = m.row_gap.map(|g| g * k);
                s.width = px(m.width);
                s.height = px(m.height);
                s.min_width = px(m.min_width);
                s.min_height = px(m.min_height);
                s.border_radius = px(m.radius);
            });
        }
    }

    /// How far in from each edge of the window the system draws over it (a
    /// phone's notch, its home indicator, rounded corners): the page's
    /// `env(safe-area-inset-*)`. Boxes given [`pad_safe_area`](Self::pad_safe_area)
    /// keep their content clear of it.
    pub fn set_safe_area(&mut self, insets: Edges) {
        if self.env.safe_area == insets {
            return;
        }
        self.env.safe_area = insets;
        let nodes: Vec<NodeId> = self.env.safe_nodes.keys().copied().collect();
        for n in nodes {
            self.apply_safe(n);
        }
    }

    pub fn safe_area(&self) -> Edges {
        self.env.safe_area
    }

    /// Adds the safe area's insets to `node`'s padding on the edges asked
    /// for (`[top, right, bottom, left]`): its own padding plus the inset.
    pub fn pad_safe_area(&mut self, node: NodeId, edges: [bool; 4]) {
        let own = self.style(node).padding;
        self.env.safe_nodes.insert(node, (own, edges));
        self.apply_safe(node);
    }

    fn apply_safe(&mut self, node: NodeId) {
        let Some(&(own, [t, r, b, l])) = self.env.safe_nodes.get(&node) else { return };
        if !self.contains(node) {
            self.env.safe_nodes.remove(&node);
            return;
        }
        let i = self.env.safe_area;
        let on = |yes: bool, v: f32| if yes { v } else { 0.0 };
        let want = Edges { top: own.top + on(t, i.top), right: own.right + on(r, i.right), bottom: own.bottom + on(b, i.bottom), left: own.left + on(l, i.left) };
        if self.style(node).padding != want {
            self.update_style(node, |s| s.padding = want);
        }
    }

    /// Draws `node`'s text and images as plain bars (a placeholder while
    /// what it shows loads), or as themselves again.
    pub fn set_redacted(&mut self, node: NodeId, on: bool) {
        let changed = if on { self.env.redacted.insert(node) } else { self.env.redacted.remove(&node) };
        if changed {
            self.touch(node);
        }
    }

    /// Whether `node` or a box around it is drawn as a placeholder.
    pub fn is_redacted(&self, node: NodeId) -> bool {
        !self.env.redacted.is_empty() && self.ancestors(node).any(|a| self.env.redacted.contains(&a))
    }

    /// Lays rows out from the right, as Arabic and Hebrew read: everything
    /// is mirrored across the window, so what starts a row is at its right
    /// end, and a sideways strip starts at the right.
    pub fn set_layout_direction(&mut self, direction: LayoutDirection) {
        if self.env.direction != direction {
            self.env.direction = direction;
            self.invalidate_layout();
        }
    }

    pub fn layout_direction(&self) -> LayoutDirection {
        self.env.direction
    }

    /// After layout, right to left: every box mirrored across the root.
    pub(crate) fn mirror_layout(&mut self) {
        if self.env.direction != LayoutDirection::RightToLeft {
            return;
        }
        let width = self.nodes[0].rect.width;
        for n in self.nodes.iter_mut().skip(1).filter(|n| !n.removed) {
            n.rect.x = width - n.rect.x - n.rect.width;
        }
    }

    /// Whether reduced motion skips animating `property`.
    pub(crate) fn motion_skipped(&self, property: Property) -> bool {
        self.env.reduce_motion
            && matches!(property, Property::Left | Property::Top | Property::Right | Property::Bottom | Property::Width | Property::Height | Property::ScrollTop | Property::ScrollLeft | Property::Rotate | Property::RotateX | Property::RotateY | Property::Depth | Property::Scale | Property::RotateZ | Property::TranslateX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{style, FixedWidthText};

    #[test]
    fn values_flow_down_and_the_nearest_wins() {
        #[derive(Clone, Debug, PartialEq)]
        struct Density(u8);
        let mut ui = Ui::new();
        let root = ui.root();
        let a = ui.add_box(root, style! {});
        let b = ui.add_box(a, style! {});
        let c = ui.add_box(b, style! {});
        assert_eq!(ui.environment::<Density>(c), None);
        ui.set_global(Density(1));
        assert_eq!(ui.environment::<Density>(c), Some(Density(1)));
        ui.set_environment(a, Density(2));
        assert_eq!(ui.environment::<Density>(c), Some(Density(2)));
        ui.set_environment(b, Density(3));
        assert_eq!(ui.environment::<Density>(c), Some(Density(3)));
        assert_eq!(ui.environment::<Density>(a), Some(Density(2)));
    }

    #[test]
    fn tokens_follow_the_appearance_and_new_tokens_and_states_go_over_them() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! {});
        ui.token_style(b, &[("background", Token::Surface)]);
        ui.hover_style(b, "background: #ff0000");
        let surface = ui.token(Token::Surface);
        assert_eq!(ui.style(b).background, Some(surface));
        let mut t = Tokens::default();
        t.dark.surface = [0.0, 0.5, 0.0, 1.0];
        t.light.surface = [0.0, 0.5, 0.0, 1.0];
        ui.set_tokens(t);
        assert_eq!(ui.style(b).background.map(|c| (c[1] * 255.0).round()), Some(128.0));
        ui.set_interaction(crate::states::Interaction { hovered: vec![b], ..Default::default() });
        assert_eq!(ui.style(b).background, Some([1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn text_scales_and_motion_reduces() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { font-size: 10px; });
        let t = ui.add_text(b, "abc");
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        let h = ui.rect(t).height;
        ui.set_text_scale(2.0);
        assert_eq!(ui.style(t).font_size, 20.0);
        let late = ui.add_text(b, "later");
        assert_eq!(ui.style(late).font_size, 20.0, "scaled as it is added");
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(t).height, 2.0 * h);
        ui.set_text_scale(1.0);
        assert_eq!(ui.style(t).font_size, 10.0);
        ui.set_reduce_motion(true);
        ui.animate(b, crate::animation::Animation::new(Property::Left, crate::animation::Value::Number(0.0), crate::animation::Value::Number(50.0), 1.0));
        assert!(!ui.is_animating(b));
        assert_eq!(ui.style(b).left, crate::Length::Px(50.0));
    }
}

#[cfg(test)]
mod animated_state_tests {
    use crate::animation::{Animation, Property, Value};
    use crate::{style, Length, Ui};

    /// A node with state styles keeps where an animation put it when the
    /// pointer moves and it is restyled.
    #[test]
    fn a_restyle_keeps_what_an_animation_moved() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { position: absolute; left: 0px; });
        ui.token_style(b, &[("background", super::Token::Accent)]);
        ui.hover_style(b, "opacity: 0.5");
        ui.animate(b, Animation::new(Property::Left, Value::Number(0.0), Value::Number(100.0), 0.2));
        ui.step(0.3);
        assert_eq!(ui.style(b).left, Length::Px(100.0));
        ui.set_interaction(crate::states::Interaction { hovered: vec![b], ..Default::default() });
        assert_eq!(ui.style(b).left, Length::Px(100.0), "not back at 0");
        assert_eq!(ui.style(b).opacity, 0.5, "and its hover style on top");
    }
}
