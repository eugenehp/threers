//! The supported subset of CSS, with CSS's initial values as defaults.

use crate::tree::Rect;

/// Straight (not premultiplied) RGBA, each in [0, 1].
pub type Rgba = [f32; 4];

/// A length: pixels, a percentage of the containing block, or `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Length {
    #[default]
    Auto,
    Px(f32),
    Percent(f32),
}

impl Length {
    /// The length against `base` (the containing block's size); `None` for
    /// `auto`, or a percentage of an unknown base.
    pub fn resolve(self, base: Option<f32>) -> Option<f32> {
        match self {
            Length::Auto => None,
            Length::Px(v) => Some(v),
            Length::Percent(p) => base.map(|b| b * p / 100.0),
        }
    }
}

/// How `background_gradient`'s stops are laid over the box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientShape {
    /// Along a line at this angle, in degrees, as CSS measures it (0 bottom
    /// to top, 90 left to right, 180 top to bottom), corner to corner.
    Linear(f32),
    /// In rings round `(cx, cy)` (fractions of the box), the last stop at
    /// the farthest corner.
    Radial { cx: f32, cy: f32 },
    /// Round `(cx, cy)`, starting `from` degrees clockwise from straight up.
    Conic { cx: f32, cy: f32, from: f32 },
}

impl Default for GradientShape {
    /// Left to right.
    fn default() -> Self {
        GradientShape::Linear(90.0)
    }
}

impl GradientShape {
    /// Whether it is the left-to-right gradient every painter draws.
    pub fn is_across(self) -> bool {
        matches!(self, GradientShape::Linear(a) if (a - 90.0).abs() < 1e-4)
    }

    /// How far along the stops `(x, y)` is in a box `r`: 0 at the first
    /// stop's place, 1 at the last's (outside that range past the ends).
    pub fn t(self, r: Rect, x: f32, y: f32) -> f32 {
        match self {
            GradientShape::Linear(deg) => {
                let a = deg.to_radians();
                let (dx, dy) = (a.sin(), -a.cos());
                // The line through the middle, as long as the box's corners
                // reach along it.
                let len = (r.width * dx).abs() + (r.height * dy).abs();
                let (mx, my) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
                ((x - mx) * dx + (y - my) * dy) / len.max(1e-6) + 0.5
            }
            GradientShape::Radial { cx, cy } => {
                let (px, py) = (r.x + r.width * cx, r.y + r.height * cy);
                let far = [
                    (r.x, r.y),
                    (r.x + r.width, r.y),
                    (r.x, r.y + r.height),
                    (r.x + r.width, r.y + r.height),
                ]
                .iter()
                .map(|&(qx, qy)| ((qx - px).powi(2) + (qy - py).powi(2)).sqrt())
                .fold(0.0f32, f32::max);
                ((x - px).powi(2) + (y - py).powi(2)).sqrt() / far.max(1e-6)
            }
            GradientShape::Conic { cx, cy, from } => {
                let (px, py) = (r.x + r.width * cx, r.y + r.height * cy);
                let turn = (x - px).atan2(-(y - py)) - from.to_radians();
                (turn / std::f32::consts::TAU).rem_euclid(1.0)
            }
        }
    }
}

/// The colour of `stops` (positions ascending) at `t`, mixed as encoded,
/// the ends carried out.
pub fn gradient_at(stops: &[(f32, Rgba)], t: f32) -> Rgba {
    let (Some(first), Some(last)) = (stops.first(), stops.last()) else {
        return [0.0; 4];
    };
    if t <= first.0 {
        return first.1;
    }
    if t >= last.0 {
        return last.1;
    }
    let i = stops
        .windows(2)
        .position(|w| t >= w[0].0 && t < w[1].0)
        .unwrap_or(0);
    let ((t0, c0), (t1, c1)) = (stops[i], stops[i + 1]);
    let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
    std::array::from_fn(|c| c0[c] + (c1[c] - c0[c]) * f)
}

/// How a box's drawing is put over what is behind it (`mix-blend-mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendMode {
    /// Over it, as everything is.
    #[default]
    Normal,
    /// Darkened by it: the colours multiplied (ink on paper).
    Multiply,
    /// Lightened by it: one minus the product of their inverses (light on
    /// a screen).
    Screen,
    /// Added (`plus-lighter`): glows.
    Plus,
    /// The darker of the two.
    Darken,
    /// The lighter of the two.
    Lighten,
}

/// A shape to cut a box to (`clip-path`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClipShape {
    /// The largest circle in the box, centred (`circle()`).
    Circle,
    /// The ellipse the box holds (`ellipse()`).
    Ellipse,
    /// Its ends round (`capsule`): a pill.
    Capsule,
    /// The box inset by these edges, with corners of this radius
    /// (`inset(4px round 8px)`).
    Inset(Edges, f32),
}

impl ClipShape {
    /// Where it cuts in a box `r`: a rectangle and its corners' radii.
    pub fn place(self, r: Rect) -> (Rect, f32, f32) {
        match self {
            ClipShape::Circle => {
                let d = r.width.min(r.height);
                (
                    Rect {
                        x: r.x + (r.width - d) / 2.0,
                        y: r.y + (r.height - d) / 2.0,
                        width: d,
                        height: d,
                    },
                    d / 2.0,
                    d / 2.0,
                )
            }
            ClipShape::Ellipse => (r, r.width / 2.0, r.height / 2.0),
            ClipShape::Capsule => {
                let h = r.width.min(r.height) / 2.0;
                (r, h, h)
            }
            ClipShape::Inset(e, radius) => {
                let inner = Rect {
                    x: r.x + e.left,
                    y: r.y + e.top,
                    width: (r.width - e.horizontal()).max(0.0),
                    height: (r.height - e.vertical()).max(0.0),
                };
                let rr = radius.min(inner.width.min(inner.height) / 2.0);
                (inner, rr, rr)
            }
        }
    }
}

/// Four sides, CSS order: top, right, bottom, left.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Edges {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl Edges {
    pub const ZERO: Edges = Edges {
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
        left: 0.0,
    };

    pub fn all(v: f32) -> Self {
        Self {
            top: v,
            right: v,
            bottom: v,
            left: v,
        }
    }

    /// `vertical horizontal`, as in `padding: 0px 2px`.
    pub fn symmetric(vertical: f32, horizontal: f32) -> Self {
        Self {
            top: vertical,
            right: horizontal,
            bottom: vertical,
            left: horizontal,
        }
    }

    pub fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Display {
    /// Children stack vertically and fill the width.
    #[default]
    Block,
    Flex,
    /// CSS Grid (see [`Track`] and [`GridPlacement`]).
    Grid,
    None,
}

/// One end of a grid track's size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrackSize {
    Px(f32),
    /// Of the grid's content box.
    Percent(f32),
    /// A share of the space left (a maximum only).
    Fr(f32),
    /// As large as the items in it need (their max-content size).
    Auto,
}

/// A grid track: `minmax(min, max)`. `100px` is both ends 100px; `auto`
/// both auto; `1fr` is `minmax(0, 1fr)` (CSS's minimum would be the
/// content's min-content size, which this layout does not measure).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Track {
    pub min: TrackSize,
    pub max: TrackSize,
}

impl Track {
    pub fn px(v: f32) -> Self {
        Track {
            min: TrackSize::Px(v),
            max: TrackSize::Px(v),
        }
    }
    pub fn fr(v: f32) -> Self {
        Track {
            min: TrackSize::Px(0.0),
            max: TrackSize::Fr(v),
        }
    }
    pub fn auto() -> Self {
        Track {
            min: TrackSize::Auto,
            max: TrackSize::Auto,
        }
    }
}

/// Where a grid item goes on one axis: CSS `grid-column` / `grid-row`.
/// Lines count from 1, or from the end when negative; with no start the
/// item is placed automatically.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GridPlacement {
    pub start: Option<i32>,
    pub end: Option<i32>,
    /// Tracks covered when `end` is not given (0 and 1 are one).
    pub span: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlexDirection {
    #[default]
    Row,
    Column,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlignItems {
    #[default]
    Stretch,
    FlexStart,
    Center,
    FlexEnd,
    /// In a row, the first lines of text on one line (`baseline`); in a
    /// column or grid, `flex-start`, as browsers have it.
    Baseline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JustifyContent {
    #[default]
    FlexStart,
    Center,
    FlexEnd,
    SpaceBetween,
    /// Equal space around each item (half of it at the ends).
    SpaceAround,
    /// Equal space between the items and at the ends.
    SpaceEvenly,
}

/// Whether a flex container's items wrap onto more lines (`flex-wrap`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlexWrap {
    #[default]
    NoWrap,
    Wrap,
    /// Wrapping, the lines stacked from the far side.
    WrapReverse,
}

/// How a wrapping flex container's lines share its cross axis
/// (`align-content`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlignContent {
    /// Each line grows to share what is left.
    #[default]
    Stretch,
    FlexStart,
    Center,
    FlexEnd,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Position {
    #[default]
    Static,
    Relative,
    /// Out of flow, placed by `left`/`top`/`right`/`bottom` in the nearest
    /// positioned ancestor's padding box.
    Absolute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoxSizing {
    /// `width`/`height` give the content box (CSS's initial value).
    #[default]
    ContentBox,
    /// `width`/`height` include padding and border.
    BorderBox,
}

/// CSS `cursor`: the pointer's shape over a box. `Auto` takes the nearest
/// ancestor's (the platform's arrow at the root).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Cursor {
    #[default]
    Auto,
    Default,
    Pointer,
    Text,
    Move,
    Grab,
    Grabbing,
    Crosshair,
    NotAllowed,
    Wait,
    Help,
    /// Resizing left–right (`ew-resize`, `col-resize`).
    ResizeHorizontal,
    /// Resizing up–down (`ns-resize`, `row-resize`).
    ResizeVertical,
    ResizeDiagonalDown,
    ResizeDiagonalUp,
    /// No pointer at all.
    None,
}

impl Cursor {
    /// The CSS keyword, for a browser canvas's `style.cursor`.
    pub fn css(self) -> &'static str {
        match self {
            Cursor::Auto => "auto",
            Cursor::Default => "default",
            Cursor::Pointer => "pointer",
            Cursor::Text => "text",
            Cursor::Move => "move",
            Cursor::Grab => "grab",
            Cursor::Grabbing => "grabbing",
            Cursor::Crosshair => "crosshair",
            Cursor::NotAllowed => "not-allowed",
            Cursor::Wait => "wait",
            Cursor::Help => "help",
            Cursor::ResizeHorizontal => "ew-resize",
            Cursor::ResizeVertical => "ns-resize",
            Cursor::ResizeDiagonalDown => "nwse-resize",
            Cursor::ResizeDiagonalUp => "nesw-resize",
            Cursor::None => "none",
        }
    }

    /// Reads a CSS `cursor` keyword.
    pub fn parse(css: &str) -> Option<Self> {
        Some(match css.trim() {
            "auto" => Cursor::Auto,
            "default" => Cursor::Default,
            "pointer" => Cursor::Pointer,
            "text" | "vertical-text" => Cursor::Text,
            "move" | "all-scroll" => Cursor::Move,
            "grab" => Cursor::Grab,
            "grabbing" => Cursor::Grabbing,
            "crosshair" | "cell" => Cursor::Crosshair,
            "not-allowed" | "no-drop" => Cursor::NotAllowed,
            "wait" | "progress" => Cursor::Wait,
            "help" => Cursor::Help,
            "ew-resize" | "col-resize" | "e-resize" | "w-resize" => Cursor::ResizeHorizontal,
            "ns-resize" | "row-resize" | "n-resize" | "s-resize" => Cursor::ResizeVertical,
            "nwse-resize" | "nw-resize" | "se-resize" => Cursor::ResizeDiagonalDown,
            "nesw-resize" | "ne-resize" | "sw-resize" => Cursor::ResizeDiagonalUp,
            "none" => Cursor::None,
            _ => return None,
        })
    }
}

/// One CSS `box-shadow`: the box, moved by `(x, y)`, grown by `spread`,
/// blurred by `blur` (CSS's blur radius: a Gaussian of half that standard
/// deviation), in `color`, under the box.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BoxShadow {
    pub x: f32,
    pub y: f32,
    pub blur: f32,
    pub spread: f32,
    pub color: Rgba,
}

/// CSS `scrollbar-width`: whether a scrolling box draws a scrollbar, and
/// how wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Scrollbar {
    /// None (the default here, unlike CSS: overlay scrollbars, as on a Mac).
    #[default]
    None,
    Thin,
    Auto,
}

/// Whether a scrolling box's scrollbar always shows, or only while it is
/// used, as overlay scrollbars do on a Mac or a phone. Not CSS: browsers
/// leave this to the platform. Here it is `scrollbar-visibility`, with
/// `scrollbar-fade` for its timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ScrollbarVisibility {
    /// Shown whenever there is more content than fits.
    #[default]
    Always,
    /// Shown while the box scrolls and while the pointer is over the bar or
    /// drags its thumb; it fades out after `scrollbar-fade`.
    Auto,
}

/// CSS `user-select`: whether a node's text can be selected (and copied)
/// with the pointer or a finger. `Auto` takes the nearest ancestor's
/// choice; with none, text is selectable, as in a browser, except inside
/// controls (buttons, tabs, menu items, checkboxes, radio buttons, sliders,
/// options), as a browser's own styles make them. `None` turns it off for a
/// node and what is inside it; [`Ui::set_text_selection`] for the whole UI.
///
/// [`Ui::set_text_selection`]: crate::Ui::set_text_selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum UserSelect {
    #[default]
    Auto,
    Text,
    None,
}

/// CSS `object-fit`: how an image fills its box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ObjectFit {
    /// Stretched to the box.
    #[default]
    Fill,
    /// As large as fits whole, centred.
    Contain,
    /// As small as covers the box, centred and cut.
    Cover,
    /// At its own size, centred.
    None,
    /// `Contain`, but never larger than its own size.
    ScaleDown,
}

/// Where lines of text sit across their box (`text-align`, inherited).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Start,
    Center,
    End,
}

impl TextAlign {
    /// The share of a line's spare width before it: 0, ½ or 1.
    pub fn offset(self, spare: f32) -> f32 {
        match self {
            TextAlign::Start => 0.0,
            TextAlign::Center => spare.max(0.0) / 2.0,
            TextAlign::End => spare.max(0.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontWeight {
    #[default]
    Normal,
    Bold,
    /// A numeric weight (100–900).
    Weight(u16),
}

/// One box's style.
#[derive(Debug, Clone, PartialEq)]
pub struct Style {
    pub display: Display,
    pub position: Position,
    pub box_sizing: BoxSizing,
    pub flex_direction: FlexDirection,
    pub flex_grow: f32,
    pub flex_shrink: f32,
    pub flex_basis: Length,
    pub align_items: AlignItems,
    /// `align-self`; `None` follows the parent's `align-items`.
    pub align_self: Option<AlignItems>,
    pub justify_content: JustifyContent,
    /// `column-gap` / `row-gap` along the main axis.
    pub gap: f32,
    /// `flex-wrap`.
    pub flex_wrap: FlexWrap,
    /// `flex-direction: row-reverse` or `column-reverse`: items laid from
    /// the end of the main axis.
    pub flex_reverse: bool,
    /// `order`: items placed by it (then by their place in the tree).
    pub order: i32,
    /// `align-content`.
    pub align_content: AlignContent,
    /// `column-gap`, where it differs from `gap`.
    pub column_gap: Option<f32>,
    pub width: Length,
    pub height: Length,
    pub min_width: Length,
    pub min_height: Length,
    pub max_width: Length,
    pub max_height: Length,
    pub margin: Edges,
    pub padding: Edges,
    pub border: Edges,
    pub border_color: Rgba,
    /// `border-style: dashed` (else solid): dashes along the outline,
    /// round its corners too, as a browser draws them.
    pub border_dashed: bool,
    pub background: Option<Rgba>,
    pub color: Rgba,
    /// `color` was given, so the text inside takes it even when it is the
    /// default white (see [`Ui::color_of`](crate::Ui::color_of)).
    pub color_set: bool,
    pub font_family: String,
    /// `font-family` was given; a box without one takes its parent's, as
    /// CSS inherits it (see [`Ui::set_style`](crate::Ui::set_style)).
    pub font_family_set: bool,
    /// These were given; a box without them takes its parent's, as CSS
    /// inherits them (line height only with the font size: threers keeps
    /// it in pixels, and a size of its own wants its own).
    pub font_size_set: bool,
    pub font_weight_set: bool,
    pub line_height_set: bool,
    pub text_align_set: bool,
    pub font_size: f32,
    pub font_weight: FontWeight,
    /// `line-height` in pixels; `None` is `normal` (the font's own).
    pub line_height: Option<f32>,
    pub left: Length,
    pub top: Length,
    pub right: Length,
    pub bottom: Length,
    /// `overflow: hidden`: children are clipped to the padding box.
    pub clip: bool,
    pub opacity: f32,
    /// `border-radius`, one value for every corner; a percentage is of the
    /// box's width horizontally and its height vertically, as in CSS.
    pub border_radius: Length,
    /// `text-decoration-line: line-through` (inherited by text, as the
    /// decoration propagates).
    pub line_through: bool,
    /// `background: linear-gradient(to right, …)`: colour stops at
    /// fractions of the width, painted over `background`. Two stops at one
    /// position make a hard edge.
    pub background_gradient: Vec<(f32, Rgba)>,
    /// `filter: blur(…)`, the standard deviation in pixels, applied to this
    /// box's background (not its children).
    pub blur: f32,
    /// `backdrop-filter: blur(…)`: what is behind the box, blurred by this
    /// standard deviation in pixels, shows through its (translucent)
    /// background: a material. 0 for none.
    pub backdrop_blur: f32,
    /// `backdrop-filter: … saturate(…)`: the backdrop's colourfulness, 1 as
    /// it is.
    pub backdrop_saturate: f32,
    /// `aspect-ratio`: width over height, for a box whose height is `auto`
    /// (its height follows its width). 0 for none.
    pub aspect_ratio: f32,
    /// `clip-path`: the box and all it draws cut to this shape.
    pub clip_path: Option<ClipShape>,
    /// How `background_gradient` lies over the box (left to right unless
    /// `linear-gradient` turns it, or it is `radial-` or `conic-`).
    pub gradient_shape: GradientShape,
    /// `mix-blend-mode`.
    pub mix_blend_mode: BlendMode,
    /// `mask-image`: the box shown as opaque as the gradient is (its stops'
    /// alpha), laid as a background gradient is; up to four stops count.
    pub mask: Option<(GradientShape, Vec<(f32, Rgba)>)>,
    /// `overflow: auto`: the content scrolls vertically (implies `clip`),
    /// `scroll_top` pixels down (clamped to the content when laid out).
    pub scroll: bool,
    pub scroll_top: f32,
    /// `overflow-x: auto`: the box scrolls sideways too, its content moved
    /// left by `scroll_left`.
    pub scroll_x: bool,
    pub scroll_left: f32,
    /// `background-clip: content-box`.
    pub background_clip_content: bool,
    /// Painted at fractional positions, not snapped to pixels: a subtree
    /// placed by a CSS `transform` (`translate(-50%, -50%)`), which browsers
    /// antialias. Holds for the whole subtree.
    pub subpixel: bool,
    /// `white-space: normal; word-wrap: break-word`: text wraps at spaces
    /// and zero-width spaces (U+200B, `<wbr>`), or anywhere in a word too
    /// long for the line (inherited).
    pub wrap: bool,
    /// `white-space: pre`, `pre-wrap`, `pre-line` or `break-spaces`: a
    /// line break in the text starts a new line, and the box is as tall as
    /// its lines and as wide as the widest (inherited). Without it a text is
    /// laid out as one run, as `normal` and `nowrap` treat a break as space.
    pub keep_newlines: bool,
    /// `text-align`: each line of a text across its box (inherited).
    pub text_align: TextAlign,
    /// `font-style: italic` (inherited).
    pub italic: bool,
    /// `text-overflow: ellipsis` (not inherited): text in this box (which
    /// does not wrap) wider than its content box ends in `…` where it is
    /// cut.
    pub ellipsis: bool,
    /// `writing-mode: vertical-lr`: text runs top to bottom, turned a
    /// quarter clockwise (inherited).
    pub vertical: bool,
    /// The pointer over this box (see [`Ui::cursor_at`](super::Ui::cursor_at)).
    pub cursor: Cursor,
    /// How an image node fills its box.
    pub object_fit: ObjectFit,
    /// `grid-template-columns`.
    pub grid_columns: Vec<Track>,
    /// `grid-template-rows`; rows past these are `auto`.
    pub grid_rows: Vec<Track>,
    /// `grid-column`, for an item of a grid.
    pub grid_column: GridPlacement,
    /// `grid-row`.
    pub grid_row: GridPlacement,
    /// `row-gap` in a grid; `gap` when `None`.
    pub row_gap: Option<f32>,
    /// `scrollbar-width`, for a scrolling box.
    pub scrollbar: Scrollbar,
    /// `scrollbar-color`: thumb and track.
    pub scrollbar_color: Option<(Rgba, Rgba)>,
    /// `scrollbar-visibility` (not CSS; see [`ScrollbarVisibility`]).
    pub scrollbar_visibility: ScrollbarVisibility,
    /// `scrollbar-fade: <delay> <duration>`: how long an `auto` scrollbar
    /// stays after it was last used, and how long it takes to fade, in
    /// seconds.
    pub scrollbar_fade: (f32, f32),
    /// `pointer-events`: `false` (`none`) and the pointer passes through the
    /// box to what is under it, though not through its children (not
    /// inherited, unlike CSS: set it on each node that should let go).
    /// `border-beam`: light running round the border (see
    /// [`border_beam`](crate::border_beam)).
    pub border_beam: Option<crate::border_beam::BeamStyle>,
    pub pointer_events: bool,
    /// `user-select` (see [`UserSelect`]).
    pub user_select: UserSelect,
    /// `line-clamp` (or `-webkit-line-clamp`): wrapping text shows at most
    /// this many lines, the last ending in `…` if more were cut (inherited
    /// by text added under it).
    pub line_clamp: Option<u32>,
    /// `box-shadow`, the first painted on top (not inherited).
    pub box_shadow: Vec<BoxShadow>,
    /// `text-decoration: underline` (inherited by text added under it).
    pub underline: bool,
    /// `text-decoration-style: wavy`: the underline as a wave.
    pub decoration_wavy: bool,
    /// `text-decoration-color`; the text's colour when `None`.
    pub decoration_color: Option<Rgba>,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            display: Display::Block,
            position: Position::Static,
            box_sizing: BoxSizing::ContentBox,
            flex_direction: FlexDirection::Row,
            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: Length::Auto,
            align_items: AlignItems::Stretch,
            align_self: None,
            justify_content: JustifyContent::FlexStart,
            gap: 0.0,
            flex_wrap: FlexWrap::NoWrap,
            flex_reverse: false,
            order: 0,
            align_content: AlignContent::Stretch,
            column_gap: None,
            width: Length::Auto,
            height: Length::Auto,
            min_width: Length::Auto,
            min_height: Length::Auto,
            max_width: Length::Auto,
            max_height: Length::Auto,
            margin: Edges::ZERO,
            padding: Edges::ZERO,
            border: Edges::ZERO,
            border_color: [0.0, 0.0, 0.0, 1.0],
            border_dashed: false,
            background: None,
            color: [1.0, 1.0, 1.0, 1.0],
            color_set: false,
            font_family: "sans-serif".into(),
            font_family_set: false,
            font_size_set: false,
            font_weight_set: false,
            line_height_set: false,
            text_align_set: false,
            font_size: 16.0,
            font_weight: FontWeight::Normal,
            line_height: None,
            left: Length::Auto,
            top: Length::Auto,
            right: Length::Auto,
            bottom: Length::Auto,
            clip: false,
            opacity: 1.0,
            border_radius: Length::Auto,
            line_through: false,
            background_gradient: Vec::new(),
            blur: 0.0,
            backdrop_blur: 0.0,
            backdrop_saturate: 1.0,
            aspect_ratio: 0.0,
            clip_path: None,
            gradient_shape: GradientShape::Linear(90.0),
            mix_blend_mode: BlendMode::Normal,
            mask: None,
            scroll: false,
            scroll_top: 0.0,
            scroll_x: false,
            scroll_left: 0.0,
            background_clip_content: false,
            subpixel: false,
            wrap: false,
            keep_newlines: false,
            text_align: TextAlign::Start,
            italic: false,
            ellipsis: false,
            vertical: false,
            cursor: Cursor::Auto,
            object_fit: ObjectFit::Fill,
            pointer_events: true,
            border_beam: None,
            user_select: UserSelect::Auto,
            line_clamp: None,
            scrollbar: Scrollbar::None,
            grid_columns: vec![],
            grid_rows: vec![],
            grid_column: GridPlacement::default(),
            grid_row: GridPlacement::default(),
            row_gap: None,
            scrollbar_color: None,
            scrollbar_visibility: ScrollbarVisibility::Always,
            scrollbar_fade: (1.0, 0.3),
            box_shadow: vec![],
            underline: false,
            decoration_wavy: false,
            decoration_color: None,
        }
    }
}

impl Style {
    /// `display: flex` with `flex-direction`.
    pub fn flex(direction: FlexDirection) -> Self {
        Self {
            display: Display::Flex,
            flex_direction: direction,
            ..Self::default()
        }
    }

    /// Whether `self` and `other` lay out the same: they differ, if at all,
    /// only in how boxes look (colours, shadows, corners, filters, the
    /// cursor), so a change from one to the other needs a repaint and no
    /// layout.
    pub fn same_layout(&self, other: &Style) -> bool {
        let mut o = other.clone();
        o.border_color = self.border_color;
        o.background = self.background;
        o.border_beam = self.border_beam;
        o.color = self.color;
        o.color_set = self.color_set;
        o.opacity = self.opacity;
        o.border_radius = self.border_radius;
        o.line_through = self.line_through;
        o.background_gradient.clone_from(&self.background_gradient);
        o.blur = self.blur;
        o.backdrop_blur = self.backdrop_blur;
        o.backdrop_saturate = self.backdrop_saturate;
        o.clip_path = self.clip_path;
        o.gradient_shape = self.gradient_shape;
        o.mix_blend_mode = self.mix_blend_mode;
        o.mask.clone_from(&self.mask);
        o.background_clip_content = self.background_clip_content;
        o.cursor = self.cursor;
        o.scrollbar_color = self.scrollbar_color;
        o.box_shadow.clone_from(&self.box_shadow);
        o.underline = self.underline;
        o.decoration_wavy = self.decoration_wavy;
        o.decoration_color = self.decoration_color;
        *self == o
    }
}
