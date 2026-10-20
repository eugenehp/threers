//! The supported subset of CSS, with CSS's initial values as defaults.

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

/// Four sides, CSS order: top, right, bottom, left.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Edges {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl Edges {
    pub const ZERO: Edges = Edges { top: 0.0, right: 0.0, bottom: 0.0, left: 0.0 };

    pub fn all(v: f32) -> Self {
        Self { top: v, right: v, bottom: v, left: v }
    }

    /// `vertical horizontal`, as in `padding: 0px 2px`.
    pub fn symmetric(vertical: f32, horizontal: f32) -> Self {
        Self { top: vertical, right: horizontal, bottom: vertical, left: horizontal }
    }

    pub fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self { top, right, bottom, left }
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
    None,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JustifyContent {
    #[default]
    FlexStart,
    Center,
    FlexEnd,
    SpaceBetween,
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
    pub background: Option<Rgba>,
    pub color: Rgba,
    pub font_family: String,
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
    /// `overflow: auto`: the content scrolls vertically (implies `clip`),
    /// `scroll_top` pixels down (clamped to the content when laid out).
    pub scroll: bool,
    pub scroll_top: f32,
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
    /// `font-style: italic` (inherited).
    pub italic: bool,
    /// `writing-mode: vertical-lr`: text runs top to bottom, turned a
    /// quarter clockwise (inherited).
    pub vertical: bool,
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
            background: None,
            color: [1.0, 1.0, 1.0, 1.0],
            font_family: "sans-serif".into(),
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
            scroll: false,
            scroll_top: 0.0,
            background_clip_content: false,
            subpixel: false,
            wrap: false,
            italic: false,
            vertical: false,
        }
    }
}

impl Style {
    /// `display: flex` with `flex-direction`.
    pub fn flex(direction: FlexDirection) -> Self {
        Self { display: Display::Flex, flex_direction: direction, ..Self::default() }
    }
}
