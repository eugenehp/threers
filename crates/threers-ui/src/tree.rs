//! The box tree.

use super::layout::TextMeasure;
use super::style::Style;

/// A node of a [`Ui`]: a slot and the generation of what lives in it.
///
/// A removed node's slot is reused by the next node added, under a new
/// generation, so an id kept past [`Ui::remove`] never reaches the node that
/// took its place: [`Ui::contains`] says it is gone, and the accessors panic
/// on it rather than answer for a stranger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub(crate) usize, pub(crate) u32);

/// What a node draws besides its box (background and borders).
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    /// Only its box and children.
    Box,
    /// One line of text, in the node's font and colour (`white-space: nowrap`).
    Text(String),
    /// SVG path data in a `view_box` (x, y, width, height), scaled to the
    /// content box: stroked with `stroke` px lines and/or filled, in the
    /// node's colour (`currentColor`).
    Path {
        d: Vec<String>,
        view_box: [f32; 4],
        stroke: Option<f32>,
        fill: bool,
    },
    /// An image, fitted to the content box by `object-fit` (see
    /// [`image`](super::image)).
    Image { image: super::image::ImageRef },
    /// A texture the renderer has under `id` (a video frame, a camera, a
    /// 3D view), `width × height` texels, fitted like an image. Painters
    /// that have no such textures draw nothing for it.
    Surface { id: u64, width: u32, height: u32 },
}

/// A laid-out box: the border box, in pixels from the top-left of the UI.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height
    }

    /// Whether a pointer at `(x, y)` is over the box, as Chrome hit-tests:
    /// the pixel `[x, x+1) × [y, y+1)` overlapping it, so a box starting at
    /// 60.95 is under a pointer at 60.
    pub fn hit(&self, x: f32, y: f32) -> bool {
        x + 1.0 > self.x && y + 1.0 > self.y && x < self.x + self.width && y < self.y + self.height
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub style: Style,
    pub kind: NodeKind,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub rect: Rect,
    pub removed: bool,
    /// For a scrolling box: the height of its content and how far it is
    /// scrolled, as last laid out.
    pub scroll_height: f32,
    pub scrolled: f32,
    /// Sideways, likewise: the content's width and how far it is scrolled.
    pub scroll_width: f32,
    pub scrolled_x: f32,
    /// A wrapped text node's lines, as last laid out.
    pub lines: Vec<String>,
    /// The width of each line as drawn (the text alone when it has no
    /// `lines`), for underlines and styled runs.
    pub line_widths: Vec<f32>,
    /// Styled runs over the text (see [`Ui::set_text_runs`]).
    pub runs: Option<Vec<super::rich_text::TextRun>>,
    /// Where each run's piece of each line goes, as last laid out.
    pub pieces: Vec<super::rich_text::Piece>,
    /// What it means to assistive technology (see [`super::a11y`]).
    pub semantics: Option<super::a11y::Semantics>,
    /// Which tenant of the slot this is (see [`NodeId`]).
    pub generation: u32,
    /// Changed, or something inside changed, since it was last laid out.
    pub dirty: bool,
    /// Content sizes measured for it, by `(width, height base)` bits: good
    /// for the pass in `measured_pass`, or for as long as the node stays
    /// clean inside a [`Ui::cache_layout`] boundary.
    pub measured: Vec<Measured>,
    pub measured_pass: u64,
    /// The layout pass that last placed it itself (not from a cache).
    pub placed_pass: u64,
    /// For a [`Ui::cache_layout`] boundary: the subtree as its last layout
    /// left it, and the box it was given.
    pub cache: Option<Box<LayoutCache>>,
    pub cache_boundary: bool,
    /// Painted after the rest, by this priority (see [`Ui::defer`]).
    pub deferred: Option<i32>,
    /// Placed against an anchor after layout (see [`Ui::anchor`]).
    pub anchor: Option<super::layers::Anchor>,
    /// Where in the source the node was added, for the inspector.
    pub origin: Option<&'static std::panic::Location<'static>>,
    /// Styles by state (see [`states`](super::states)).
    pub states: Option<Box<super::states::StateStyles>>,
    /// A group name, for group-hover styles.
    pub group: Option<String>,
    /// A text node's colour is its parent's, as CSS inherits `color`, until
    /// it is given its own (see [`Ui::color_of`]).
    pub inherit_color: bool,
    /// A text node's font (family, size, weight, line height, style,
    /// wrapping) follows its parent's when that is restyled, as CSS
    /// inherits them, until it is given a style of its own.
    pub inherit_font: bool,
    /// A path node's shape turned, scaled or moved (see
    /// [`Ui::set_path_transform`]).
    pub transform: Option<super::transform::Transformation>,
    /// Styles by a container's width (see [`Ui::container_query`]).
    pub query: Option<Box<super::responsive::ContainerQuery>>,
}

/// A content size, under the bits of the width and height base it was
/// measured for.
pub(crate) type Measured = (Option<u32>, Option<u32>, (f32, f32));

/// What laying out a cached subtree produced, to be put back when it is
/// given the same box again with nothing inside it changed.
#[derive(Debug, Clone)]
pub(crate) struct LayoutCache {
    pub rect: Rect,
    pub layout_units: bool,
    pub nodes: Vec<(usize, Rect, Vec<String>, f32, f32)>,
}

impl Node {
    fn new(style: Style, kind: NodeKind, parent: Option<NodeId>, generation: u32) -> Self {
        Node {
            style,
            kind,
            parent,
            children: vec![],
            rect: Rect::default(),
            removed: false,
            scroll_height: 0.0,
            scroll_width: 0.0,
            scrolled_x: 0.0,
            scrolled: 0.0,
            lines: vec![],
            line_widths: vec![],
            runs: None,
            pieces: vec![],
            semantics: None,
            generation,
            dirty: true,
            measured: vec![],
            measured_pass: 0,
            placed_pass: 0,
            cache: None,
            cache_boundary: false,
            deferred: None,
            anchor: None,
            origin: None,
            query: None,
            states: None,
            group: None,
            inherit_color: false,
            inherit_font: false,
            transform: None,
        }
    }
}

/// A tree of styled boxes. Node 0 is the root.
#[derive(Debug, Clone)]
pub struct Ui {
    pub(crate) nodes: Vec<Node>,
    /// Removed slots, for the next nodes added.
    pub(crate) free: Vec<usize>,
    /// Whether layout puts boxes on Blink's 1/64 px grid (see
    /// [`Ui::set_layout_units`]).
    pub(crate) layout_units: bool,
    /// Counts layouts, so a measurement knows which one it was taken in.
    pub(crate) pass: u64,
    /// The viewport of the last layout.
    pub(crate) viewport: (f32, f32),
    /// Built on the first hit test after the boxes change.
    pub(crate) hits: std::sync::OnceLock<super::layers::HitIndex>,
    /// Animations running (see [`Ui::animate`]).
    pub(crate) animations: Vec<super::animation::Running>,
    /// Device pixels per UI pixel, which scroll offsets are snapped to.
    pub(crate) pixel_ratio: f32,
    /// What the pointer and keyboard are doing (see [`Ui::set_interaction`]).
    pub(crate) interaction: super::states::Interaction,
    /// Whether text can be selected at all (see [`Ui::set_text_selection`]).
    pub(crate) text_selection: bool,
    /// The theme asked for, and the system's appearance (see
    /// [`Ui::appearance`]).
    pub(crate) theme: super::states::Theme,
    pub(crate) system_appearance: super::states::Appearance,
    /// App-wide values, one per type (see [`Ui::set_global`]).
    pub(crate) globals: super::globals::Globals,
    /// Values, colours, text size and motion flowing down (see
    /// [`environment`](super::environment)).
    pub(crate) env: super::environment::Env,
    /// Nodes leaving with a transition (see [`transitions`](super::transitions)).
    pub(crate) leaving: Vec<(NodeId, f32)>,
    /// Containers showing the first child that fits (see [`fits`](super::fits)).
    pub(crate) fits: Vec<(NodeId, Vec<super::Display>)>,
    /// Boxes following another's place (see [`matched`](super::matched)).
    pub(crate) matched: Vec<super::matched::Matched>,
    /// Properties pulled by springs (see [`springs`](super::springs)).
    pub(crate) springs: Vec<super::springs::Sprung>,
    /// Values stated for ancestors (see [`preferences`](super::preferences)).
    /// The modal dialog open, if one is: the accessibility tree is it
    /// alone (see [`Ui::set_accessibility_modal`]).
    pub(crate) a11y_modal: Option<NodeId>,
    /// Messages for assistive technology to read, with an id each, and
    /// whether they interrupt (see [`Ui::announce`]).
    pub(crate) announcements: Vec<(u64, String, bool)>,
    pub(crate) announced: u64,
    /// Draw a focus ring where a focused node has no focus style of its own.
    pub(crate) focus_ring: bool,
    /// Boxes that settle on snap points (see [`scroll_snap`](super::scroll_snap)).
    pub(crate) snaps: Vec<(NodeId, super::scroll_snap::Snap)>,
    pub(crate) prefs: Vec<(NodeId, std::any::TypeId, std::rc::Rc<dyn std::any::Any>)>,
    /// Seconds stepped so far (see [`Ui::step`]), the clock `auto`
    /// scrollbars fade by.
    pub(crate) clock: f64,
    /// For each scrolling box with an `auto` scrollbar: the scroll offset it
    /// was last seen at, and when its bar was last used.
    pub(crate) scrollbar_use: std::collections::HashMap<NodeId, (f32, f64)>,
}

impl Default for Ui {
    fn default() -> Self {
        Self::new()
    }
}

impl Ui {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::new(Style::default(), NodeKind::Box, None, 0)],
            free: vec![],
            layout_units: false,
            pass: 0,
            viewport: (0.0, 0.0),
            hits: std::sync::OnceLock::new(),
            animations: vec![],
            pixel_ratio: 1.0,
            interaction: Default::default(),
            text_selection: true,
            theme: Default::default(),
            system_appearance: Default::default(),
            globals: Default::default(),
            env: Default::default(),
            leaving: vec![],
            fits: vec![],
            matched: vec![],
            springs: vec![],
            prefs: vec![],
            snaps: vec![],
            a11y_modal: None,
            announcements: vec![],
            announced: 0,
            focus_ring: true,
            clock: 0.0,
            scrollbar_use: Default::default(),
        }
    }

    /// Lays boxes out on Blink's `LayoutUnit` grid (1/64 px), as a page's
    /// are, where float error is all that keeps one off it: a position or
    /// size within 1/512 px of the grid is put on it. Off by default.
    ///
    /// Where text lands decides how it rasterizes: Skia puts glyphs on
    /// quarter pixels, so a run laid out at 976.6199 px is drawn a quarter
    /// pixel left of the same run at Blink's 976.625. A UI meant to match a
    /// page pixel for pixel turns this on (and puts its own geometry on the
    /// grid with [`super::layout_unit`]).
    /// Device pixels per UI pixel (a display's scale factor): scroll
    /// offsets are laid out on whole device pixels, as a browser scrolls.
    pub fn set_pixel_ratio(&mut self, ratio: f32) {
        let ratio = if ratio.is_finite() && ratio > 0.0 { ratio } else { 1.0 };
        if self.pixel_ratio != ratio {
            self.pixel_ratio = ratio;
            self.invalidate_layout();
        }
    }

    pub fn set_layout_units(&mut self, on: bool) {
        if self.layout_units != on {
            self.layout_units = on;
            self.invalidate_layout();
        }
    }

    pub fn root(&self) -> NodeId {
        NodeId(0, self.nodes[0].generation)
    }

    /// Whether `id` is a node of this tree: added and not removed since.
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes
            .get(id.0)
            .is_some_and(|n| n.generation == id.1 && !n.removed)
    }

    /// The node `id` names, or a panic naming the mistake: an id from
    /// another tree, or one whose node was removed (its slot may since hold
    /// another).
    #[track_caller]
    pub(crate) fn node(&self, id: NodeId) -> &Node {
        assert!(self.contains(id), "{id:?} is not in this Ui: it was removed, or is another tree's");
        &self.nodes[id.0]
    }

    #[track_caller]
    pub(crate) fn node_mut(&mut self, id: NodeId) -> &mut Node {
        assert!(self.contains(id), "{id:?} is not in this Ui: it was removed, or is another tree's");
        &mut self.nodes[id.0]
    }

    /// How many nodes the tree has (the root included).
    pub fn live_nodes(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    /// How many node slots the tree holds, live or free: its memory, which
    /// removing and adding nodes reuses rather than grows.
    pub fn slots(&self) -> usize {
        self.nodes.len()
    }

    /// The id of what lives in slot `index` now.
    pub(crate) fn id_at(&self, index: usize) -> NodeId {
        NodeId(index, self.nodes[index].generation)
    }

    /// Marks `id` and every ancestor as needing layout. Always to the root:
    /// an ancestor can be dirty already (a `display: none` box is never laid
    /// out, so never cleaned) and still have clean ancestors above it.
    pub(crate) fn touch(&mut self, id: NodeId) {
        self.invalidate_hits();
        let mut at = Some(id.0);
        while let Some(i) = at {
            let n = &mut self.nodes[i];
            n.dirty = true;
            n.measured.clear();
            at = n.parent.map(|p| p.0);
        }
    }

    /// How many layout passes have run: what a test reads to see that a
    /// change was only drawn.
    pub fn layout_passes(&self) -> u64 {
        self.pass
    }

    /// Asks for a repaint without a layout: what changed is how boxes look,
    /// not where they go.
    pub(crate) fn touch_paint(&mut self) {
        self.invalidate_hits();
        self.env.repaint.set(true);
    }

    /// Forgets every measurement and cached layout. Call it when the text
    /// measurer starts measuring differently (a font loaded, the scale
    /// changed): what [`Ui::cache_layout`] kept was measured by the old one.
    pub fn invalidate_layout(&mut self) {
        for n in &mut self.nodes {
            n.dirty = true;
            n.measured.clear();
            n.cache = None;
        }
    }

    /// Makes `id` a layout cache boundary: while
    /// nothing inside it changes and it is given the same box, a layout puts
    /// back what the last one computed for its subtree instead of measuring
    /// and placing it again, and the measurements inside it are kept from
    /// one layout to the next.
    ///
    /// Changes go through the tree's own methods ([`Ui::style_mut`],
    /// [`Ui::set_text`], adding and removing), which is how it knows. What it
    /// cannot know is that the [`TextMeasure`] passed to [`Ui::layout`]
    /// measures as it did last time: when that changes, call
    /// [`Ui::invalidate_layout`]. The result is the same layout either way —
    /// the cache only skips work whose answer is already known.
    pub fn cache_layout(&mut self, id: NodeId, on: bool) {
        let n = self.node_mut(id);
        n.cache_boundary = on;
        if !on {
            n.cache = None;
        }
    }

    /// How many nodes the last [`Ui::layout`] laid out itself, as against
    /// putting back from a [`Ui::cache_layout`] boundary.
    pub fn laid_out_last(&self) -> usize {
        self.nodes.iter().filter(|n| n.placed_pass == self.pass && !n.removed).count()
    }

    #[track_caller]
    fn add(&mut self, parent: NodeId, style: Style, kind: NodeKind) -> NodeId {
        self.node(parent);
        let origin = Some(std::panic::Location::caller());
        let id = match self.free.pop() {
            Some(i) => {
                let generation = self.nodes[i].generation;
                self.nodes[i] = Node {
                    origin,
                    ..Node::new(style, kind, Some(parent), generation)
                };
                NodeId(i, generation)
            }
            None => {
                self.nodes.push(Node {
                    origin,
                    ..Node::new(style, kind, Some(parent), 0)
                });
                NodeId(self.nodes.len() - 1, 0)
            }
        };
        self.nodes[parent.0].children.push(id);
        self.touch(parent);
        id
    }

    /// A box child of `parent`.
    #[track_caller]
    pub fn add_box(&mut self, parent: NodeId, style: Style) -> NodeId {
        self.add(parent, style, NodeKind::Box)
    }

    /// A line of text under `parent`, with `parent`'s font and colour (CSS
    /// inheritance of `color` and `font-*`).
    #[track_caller]
    pub fn add_text(&mut self, parent: NodeId, text: impl Into<String>) -> NodeId {
        let p = &self.node(parent).style;
        let style = Style {
            color: p.color,
            font_family: p.font_family.clone(),
            font_size: p.font_size,
            font_weight: p.font_weight,
            line_height: p.line_height,
            line_through: p.line_through,
            wrap: p.wrap,
            text_align: p.text_align,
            line_clamp: p.line_clamp,
            italic: p.italic,
            vertical: p.vertical,
            underline: p.underline,
            decoration_wavy: p.decoration_wavy,
            decoration_color: p.decoration_color,
            ..Style::default()
        };
        let id = self.add(parent, style, NodeKind::Text(text.into()));
        self.nodes[id.0].inherit_color = true;
        self.nodes[id.0].inherit_font = true;
        if self.text_scale() != 1.0 {
            self.scale_text(id);
        }
        if self.bold_text() {
            self.embolden_text(id);
        }
        id
    }

    /// The colour `id` paints text and shapes in, as CSS inherits `color`: a
    /// text node's own once it has been given one (by [`set_style`](Self::set_style)),
    /// else the nearest box's above it that has a colour of its own (given
    /// in CSS, white too, or other than the default white); a box's or a
    /// shape's own, likewise. So a
    /// panel recoloured for the dark appearance, or a button's hover
    /// colour, reaches the text inside.
    pub fn color_of(&self, id: NodeId) -> super::style::Rgba {
        const DEFAULT: super::style::Rgba = [1.0, 1.0, 1.0, 1.0];
        let n = &self.nodes[id.0];
        if matches!(n.kind, NodeKind::Text(_)) && !n.inherit_color {
            return n.style.color;
        }
        let skip = usize::from(matches!(n.kind, NodeKind::Text(_)));
        self.ancestors(id)
            .skip(skip)
            .map(|a| &self.nodes[a.0].style)
            .find(|s| s.color_set || s.color != DEFAULT)
            .map_or(DEFAULT, |s| s.color)
    }

    /// An icon or other vector shape (see [`NodeKind::Path`]).
    #[track_caller]
    pub fn add_path(
        &mut self,
        parent: NodeId,
        style: Style,
        d: Vec<String>,
        view_box: [f32; 4],
        stroke: Option<f32>,
        fill: bool,
    ) -> NodeId {
        self.add(
            parent,
            style,
            NodeKind::Path {
                d,
                view_box,
                stroke,
                fill,
            },
        )
    }

    /// Replaces a path node's data (`d`), keeping its view box, stroke and
    /// fill. Does nothing to a node that is not a path.
    pub fn set_path_data(&mut self, id: NodeId, d: Vec<String>) {
        if let NodeKind::Path { view_box, stroke, fill, d: old } = self.kind(id).clone()
            && old != d
        {
            self.set_kind(id, NodeKind::Path { d, view_box, stroke, fill });
        }
    }

    /// Removes `id` and its subtree, freeing their slots for new nodes.
    /// Removing a node already gone does nothing; removing the root removes
    /// everything inside it, and keeps the root.
    pub fn remove(&mut self, id: NodeId) {
        if !self.contains(id) {
            return;
        }
        if id.0 == 0 {
            self.clear_children(id);
            return;
        }
        if let Some(p) = self.nodes[id.0].parent {
            self.nodes[p.0].children.retain(|&c| c != id);
            self.touch(p);
        }
        let mut stack = vec![id.0];
        while let Some(i) = stack.pop() {
            let n = &mut self.nodes[i];
            stack.extend(n.children.iter().map(|c| c.0));
            // Emptied, so a removed subtree holds no memory, and marked under
            // the next generation, so its old ids stop naming it.
            *n = Node {
                removed: true,
                ..Node::new(Style::default(), NodeKind::Box, None, n.generation.wrapping_add(1))
            };
            self.free.push(i);
        }
    }

    /// Moves `id` and everything inside it by `(dx, dy)` after layout (for
    /// an app placing something as a renderer would, by its own rounding).
    pub fn offset(&mut self, id: NodeId, dx: f32, dy: f32) {
        self.node(id);
        // The next layout puts it back where layout would, so it must run.
        self.touch(id);
        self.offset_laid_out(id, dx, dy);
    }

    /// Moves `id`'s subtree as part of layout (anchoring): the result is
    /// the layout's, so nothing is marked for another.
    pub(crate) fn offset_laid_out(&mut self, id: NodeId, dx: f32, dy: f32) {
        self.invalidate_hits();
        let mut stack = vec![id];
        while let Some(n) = stack.pop() {
            self.nodes[n.0].rect.x += dx;
            self.nodes[n.0].rect.y += dy;
            stack.extend(self.nodes[n.0].children.iter().copied());
        }
    }

    /// Moves `child` (a child of `parent`) to position `index` among its
    /// siblings (CSS `order`).
    pub fn move_child(&mut self, parent: NodeId, child: NodeId, index: usize) {
        let children = &mut self.node_mut(parent).children;
        if let Some(i) = children.iter().position(|&c| c == child) {
            children.remove(i);
            children.insert(index.min(children.len()), child);
            self.touch(parent);
        }
    }

    /// Moves `child` from wherever it is to the end of `parent`'s children
    /// (the DOM's `appendChild` of an attached node).
    pub fn reparent(&mut self, child: NodeId, parent: NodeId) {
        self.node(parent);
        if let Some(p) = self.node(child).parent {
            self.nodes[p.0].children.retain(|&c| c != child);
            self.touch(p);
        }
        self.nodes[child.0].parent = Some(parent);
        self.nodes[parent.0].children.push(child);
        self.touch(child);
    }

    /// Removes every child of `id`.
    pub fn clear_children(&mut self, id: NodeId) {
        for c in self.node(id).children.clone() {
            self.remove(c);
        }
    }

    #[track_caller]
    pub fn style(&self, id: NodeId) -> &Style {
        &self.node(id).style
    }

    /// The node's style, to change in place. Marks it for layout, whether or
    /// not anything is then changed.
    #[track_caller]
    pub fn style_mut(&mut self, id: NodeId) -> &mut Style {
        self.node(id);
        self.touch(id);
        &mut self.nodes[id.0].style
    }

    #[track_caller]
    /// Changes `id`'s style with `f`. On a node with state or appearance
    /// styles, it changes the style they go over (which [`style_mut`](Self::style_mut)
    /// would not: the next change of state puts that back), and they go
    /// over it again.
    pub fn update_style(&mut self, id: NodeId, f: impl FnOnce(&mut Style)) {
        if let Some(base) = self.node_mut(id).states.as_mut().and_then(|s| s.base.as_mut()) {
            f(base);
            self.restyle(id);
        } else {
            let was = self.node(id).style.clone();
            f(&mut self.nodes[id.0].style);
            // Only how it looks changed: drawn again, not laid out.
            if was.same_layout(&self.nodes[id.0].style) {
                if was != self.nodes[id.0].style {
                    self.touch_paint();
                }
            } else {
                self.touch(id);
            }
        }
    }

    pub fn set_style(&mut self, id: NodeId, style: Style) {
        // A node with state styles takes this as its style out of every
        // state, and the state it is in goes on top.
        if let Some(states) = &mut self.node_mut(id).states {
            states.base = Some(style);
            self.restyle(id);
            return;
        }
        if self.node(id).style != style {
            let text = matches!(self.nodes[id.0].kind, NodeKind::Text(_));
            // A text node given a colour of its own stops inheriting one,
            // and given a style of its own, its font too.
            if text && style.color != self.nodes[id.0].style.color {
                self.nodes[id.0].inherit_color = false;
            }
            if text {
                self.nodes[id.0].inherit_font = false;
            }
            if self.nodes[id.0].style.same_layout(&style) {
                self.touch_paint();
            } else {
                self.touch(id);
            }
            let weight = style.font_weight;
            self.nodes[id.0].style = style;
            if text {
                self.rescale_text_after_set(id);
                self.reembolden_text(id, weight);
            } else {
                self.inherit_text_style(id);
            }
        }
    }

    /// The text right inside `id` takes its font again (a button restyled
    /// bold keeps no lighter text), as CSS inherits it.
    pub(crate) fn inherit_text_style(&mut self, id: NodeId) {
        let p = self.nodes[id.0].style.clone();
        let texts: Vec<NodeId> = self
            .children(id)
            .iter()
            .copied()
            .filter(|&c| matches!(self.nodes[c.0].kind, NodeKind::Text(_)) && self.nodes[c.0].inherit_font)
            .collect();
        for c in texts {
            let s = &self.nodes[c.0].style;
            let same = s.font_family == p.font_family
                && s.text_align == p.text_align
                && self.own_weight(c) == p.font_weight
                && s.italic == p.italic
                && s.wrap == p.wrap
                && self.env.unscaled.get(&c).map_or((s.font_size, s.line_height), |u| *u) == (p.font_size, p.line_height);
            if same {
                continue;
            }
            self.touch(c);
            let s = &mut self.nodes[c.0].style;
            s.font_family = p.font_family.clone();
            s.font_weight = p.font_weight;
            s.italic = p.italic;
            s.wrap = p.wrap;
            s.text_align = p.text_align;
            s.font_size = p.font_size;
            s.line_height = p.line_height;
            if self.text_scale() != 1.0 {
                self.env.unscaled.insert(c, (p.font_size, p.line_height));
                self.scale_text(c);
            }
            self.reembolden_text(c, p.font_weight);
        }
    }

    #[track_caller]
    pub fn kind(&self, id: NodeId) -> &NodeKind {
        &self.node(id).kind
    }

    #[track_caller]
    pub(crate) fn add_kind(&mut self, parent: NodeId, style: Style, kind: NodeKind) -> NodeId {
        self.add(parent, style, kind)
    }

    pub(crate) fn set_kind(&mut self, id: NodeId, kind: NodeKind) {
        self.node(id);
        self.touch(id);
        self.nodes[id.0].kind = kind;
    }

    /// Replaces a text node's text.
    #[track_caller]
    /// Text node `id`'s lines as last laid out: as wrapped, clamped or
    /// ellipsised; its text whole when it was none of those.
    pub fn text_lines_of(&self, id: NodeId) -> Vec<String> {
        let n = self.node(id);
        match &n.kind {
            NodeKind::Text(t) if n.lines.is_empty() => vec![t.clone()],
            NodeKind::Text(_) => n.lines.clone(),
            _ => vec![],
        }
    }

    pub fn set_text(&mut self, id: NodeId, text: impl Into<String>) {
        let kind = NodeKind::Text(text.into());
        if self.node(id).kind != kind {
            self.touch(id);
            self.nodes[id.0].kind = kind;
        }
    }

    #[track_caller]
    pub fn children(&self, id: NodeId) -> &[NodeId] {
        &self.node(id).children
    }

    #[track_caller]
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.node(id).parent
    }

    /// The border box from the last [`layout`](Self::layout).
    #[track_caller]
    pub fn rect(&self, id: NodeId) -> Rect {
        self.node(id).rect
    }

    /// Lays the tree out in a `width × height` viewport. Nothing is done
    /// when nothing has changed since the last layout at this size, so
    /// calling it every frame, or
    /// several times a frame, costs nothing while nothing happens. Changes
    /// through the tree's methods count; a [`TextMeasure`] that starts
    /// measuring differently does not — call [`Ui::invalidate_layout`].
    pub fn layout(&mut self, width: f32, height: f32, text: &mut dyn TextMeasure) {
        // A frame is coming: what asked for a repaint gets it.
        self.env.repaint.set(false);
        if self.pass > 0 && !self.nodes[0].dirty && self.viewport == (width, height) {
            return;
        }
        self.viewport = (width, height);
        super::layout::layout(self, width, height, text);
        // Container queries restyle by what layout found, and a restyle can
        // move what they read: lay out again until they settle (in two
        // more at most, given containment).
        for _ in 0..2 {
            let queries = self.apply_container_queries();
            let ratios = self.apply_aspect_ratios();
            if !(self.apply_fits(text) | queries | ratios) {
                break;
            }
            super::layout::layout(self, width, height, text);
        }
        self.mirror_layout();
        self.place_carousels();
        self.place_cylinders();
        self.place_scroll_transitions();
        // Indicators placed for the first time (or at once) need their
        // boxes laid out; animated ones move from the next frame.
        if self.place_matched() {
            super::layout::layout(self, width, height, text);
            self.mirror_layout();
        }
        self.place_anchored();
        self.invalidate_hits();
    }

    /// The deepest visible node containing `(x, y)`, last painted first —
    /// deferred nodes before the rest. Looked up in an index of the boxes
    /// as last laid out (see [`layers`](super::layers)).
    pub fn hit_test(&self, x: f32, y: f32) -> Option<NodeId> {
        // Boxes turned in 3D are hit where they are drawn (see `depth`).
        if !self.env.transforms3d.is_empty() || !self.env.parallax.is_empty() {
            return self.hit_through_transforms(x, y);
        }
        self.hits
            .get_or_init(|| super::layers::HitIndex::build(self))
            .hit(self, x, y)
    }

    /// Boxes with an `aspect-ratio` and no height of their own get the
    /// height their width gives. Returns whether any changed (lay out
    /// again).
    fn apply_aspect_ratios(&mut self) -> bool {
        let mut changed = false;
        for i in 0..self.nodes.len() {
            let ratio = self.nodes[i].style.aspect_ratio;
            if self.nodes[i].removed || ratio <= 0.0 {
                continue;
            }
            let id = self.id_at(i);
            let ours = self.env.aspect_heights.get(&id).copied();
            let free = match self.nodes[i].style.height {
                super::Length::Auto => true,
                super::Length::Px(v) => ours == Some(v),
                _ => false,
            };
            let width = self.nodes[i].rect.width;
            if !free || width <= 0.0 {
                continue;
            }
            let want = width / ratio;
            if self.nodes[i].style.height != super::Length::Px(want) {
                self.nodes[i].style.height = super::Length::Px(want);
                self.env.aspect_heights.insert(id, want);
                self.touch(id);
                changed = true;
            }
        }
        changed
    }

    /// The pointer's shape at `(x, y)`: the [`Style::cursor`] of the node
    /// there or its nearest ancestor that sets one, as CSS inherits it;
    /// [`Cursor::Default`](super::Cursor::Default) over nothing that does.
    pub fn cursor_at(&self, x: f32, y: f32) -> super::Cursor {
        self.hit_test(x, y)
            .map_or(super::Cursor::Default, |n| self.cursor_of(n))
    }

    /// The cursor `id` shows: its own, or its nearest ancestor's.
    pub fn cursor_of(&self, id: NodeId) -> super::Cursor {
        self.ancestors(id)
            .map(|n| self.nodes[n.0].style.cursor)
            .find(|c| *c != super::Cursor::Auto)
            .unwrap_or(super::Cursor::Default)
    }

    pub(crate) fn is_positioned(&self, id: NodeId) -> bool {
        self.nodes[id.0].style.position != super::style::Position::Static
    }

    /// `id`'s children in painting order: in-flow boxes first, then
    /// positioned ones (`relative`, `absolute`), each in tree order — CSS's
    /// order without `z-index`.
    pub fn paint_order(&self, id: NodeId) -> Vec<NodeId> {
        self.paint_order_iter(id).collect()
    }

    /// [`Ui::paint_order`] without collecting it.
    pub(crate) fn paint_order_iter(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        let children = &self.node(id).children;
        let in_flow = children.iter().copied().filter(|&c| !self.is_positioned(c));
        in_flow.chain(children.iter().copied().filter(|&c| self.is_positioned(c)))
    }

    /// A scrolling box's content height and the height it shows (its
    /// padding box), and how far it is scrolled, as last laid out.
    pub fn scroll_extent(&self, id: NodeId) -> (f32, f32, f32) {
        let n = self.node(id);
        let s = &n.style;
        (
            n.scroll_height,
            (n.rect.height - s.border.vertical()).max(0.0),
            n.scrolled,
        )
    }

    /// [`scroll_extent`](Self::scroll_extent) sideways: the content's width,
    /// the view's, and how far it is scrolled, as last laid out.
    pub fn scroll_extent_x(&self, id: NodeId) -> (f32, f32, f32) {
        let n = self.node(id);
        (n.scroll_width, (n.rect.width - n.style.border.horizontal()).max(0.0), n.scrolled_x)
    }

    /// Scrolls `container` sideways to `offset`, inside its extent.
    pub fn scroll_to_x(&mut self, container: NodeId, offset: f32) {
        let (content, view, _) = self.scroll_extent_x(container);
        let to = offset.clamp(0.0, (content - view).max(0.0));
        if self.style(container).scroll_left != to {
            self.style_mut(container).scroll_left = to;
        }
    }

    /// The innermost scrolling box among `id` and its ancestors.
    pub fn scroll_container(&self, id: NodeId) -> Option<NodeId> {
        self.ancestors(id).find(|n| self.nodes[n.0].style.scroll)
    }

    /// `id` and its ancestors, innermost first.
    pub fn ancestors(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.node(id);
        std::iter::successors(Some(id), move |n| self.nodes[n.0].parent)
    }
}
