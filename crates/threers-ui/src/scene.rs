//! A painted frame as data: the [`Painter`] calls [`Ui::paint`] makes, in
//! order, kept rather than carried out.
//!
//! A frame goes to the renderer as a list of primitives, and the renderer
//! decides how to draw them; a [`Scene`] is that hand-off. threers' GPU pass (`threers::renderer::UiPass`) draws one
//! with a single instanced draw call. [`Scene::replay`] gives the same calls
//! to any other painter — the CPU overlay, a browser canvas — so a scene
//! drawn on the GPU can be checked against one drawn anywhere else, and two
//! scenes compare with `==` to say whether a frame looks any different.
//!
//! ```
//! use threers_ui::scene::Item;
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! ui.set_style(root, style! { background: #222; padding: 4px; });
//! ui.add_text(root, "hi");
//! ui.layout(100.0, 40.0, &mut FixedWidthText(7.0));
//! let scene = ui.scene();
//! assert!(matches!(scene.items[0], Item::Rect { .. }));
//! assert!(matches!(&scene.items[1], Item::Text { text, .. } if text == "hi"));
//! assert_eq!(scene, ui.scene());
//! ```

use super::layout::TextMeasure;
use super::paint::{Painter, TextStyle};
use super::style::{Rgba, Style};
use super::tree::{Rect, Ui};

/// One step of painting.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    /// A filled rectangle; with corners of radii `rx` × `ry` when they are
    /// above zero.
    Rect {
        rect: Rect,
        rx: f32,
        ry: f32,
        color: Rgba,
    },
    /// A left-to-right gradient (see [`Painter::fill_linear_gradient`]).
    Gradient {
        rect: Rect,
        stops: Vec<(f32, Rgba)>,
        blur: f32,
    },
    /// A gradient of another shape (see [`Painter::fill_gradient`]).
    ShapedGradient {
        rect: Rect,
        shape: crate::GradientShape,
        stops: Vec<(f32, Rgba)>,
    },
    /// One line of text in its line box.
    Text {
        line: Rect,
        text: String,
        style: TextStyle,
    },
    /// SVG path data (see [`Painter::path`]).
    Path {
        rect: Rect,
        d: Vec<String>,
        view_box: [f32; 4],
        stroke: Option<f32>,
        fill: bool,
        color: Rgba,
    },
    /// A circular arc, stroked (see [`Painter::stroke_arc`]).
    Arc {
        arc: crate::paint::ArcShape,
        width: f32,
        color: Rgba,
    },
    /// Painting is restricted to `rect`, with corner radii, until the
    /// matching [`Item::PopClip`].
    PushClip {
        rect: Rect,
        rx: f32,
        ry: f32,
    },
    PopClip,
    /// What follows is mapped through an affine transform (see
    /// [`Painter::push_transform`]) until the matching
    /// [`Item::PopTransform`].
    PushTransform([f32; 6]),
    /// Like [`Item::PushTransform`], in true perspective (see
    /// [`Painter::push_projective`]).
    PushProjective {
        h: [f32; 9],
        affine: [f32; 6],
    },
    PopTransform,
    /// What follows is blurred by this many pixels until [`Item::PopBlur`].
    PushBlur(f32),
    PopBlur,
    /// What follows is drawn through a custom shader until
    /// [`Item::PopEffect`] (see [`Painter::push_effect`]).
    PushEffect {
        rect: Rect,
        effect: crate::shader::Effect,
        time: f32,
    },
    PopEffect,
    /// Into or out of a [`Style::subpixel`] subtree.
    Subpixel(bool),
    /// A box shadow (see [`Painter::shadow`]).
    Shadow {
        rect: Rect,
        radius: f32,
        blur: f32,
        color: Rgba,
    },
    /// A line under text (see [`Painter::underline`]).
    Underline {
        x: f32,
        y: f32,
        width: f32,
        thickness: f32,
        color: Rgba,
        wavy: bool,
    },
    /// The renderer's texture `id` drawn to `rect` at `opacity`.
    Surface {
        rect: Rect,
        id: u64,
        opacity: f32,
    },
    /// What is under `rect`, blurred (see [`Painter::backdrop`]).
    Backdrop {
        rect: Rect,
        radius: f32,
        blur: f32,
        saturate: f32,
    },
    /// An image drawn to `rect` at `opacity`.
    Image {
        rect: Rect,
        image: crate::image::ImageRef,
        opacity: f32,
    },
}

/// A frame's painting, in order. See the [module docs](self).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scene {
    pub items: Vec<Item>,
}

impl Scene {
    /// Makes the calls this scene records on `painter`, in order.
    pub fn replay(&self, painter: &mut dyn Painter) {
        for item in &self.items {
            match item {
                Item::Rect {
                    rect,
                    rx,
                    ry,
                    color,
                } => {
                    if *rx > 0.0 && *ry > 0.0 {
                        painter.fill_rounded_rect(*rect, *rx, *ry, *color);
                    } else {
                        painter.fill_rect(*rect, *color);
                    }
                }
                Item::Gradient { rect, stops, blur } => {
                    painter.fill_linear_gradient(*rect, stops, *blur)
                }
                Item::ShapedGradient { rect, shape, stops } => {
                    painter.fill_gradient(*rect, *shape, stops, 0.0)
                }
                Item::Text { line, text, style } => painter.text(*line, text, style),
                Item::Path {
                    rect,
                    d,
                    view_box,
                    stroke,
                    fill,
                    color,
                } => painter.path(*rect, d, *view_box, *stroke, *fill, *color),
                Item::Arc { arc, width, color } => painter.stroke_arc(*arc, *width, *color),
                Item::PushClip { rect, rx, ry } => {
                    if *rx > 0.0 && *ry > 0.0 {
                        painter.push_clip_rounded(*rect, *rx, *ry);
                    } else {
                        painter.push_clip(*rect);
                    }
                }
                Item::PopClip => painter.pop_clip(),
                Item::PushTransform(m) => painter.push_transform(*m),
                Item::PushProjective { h, affine } => painter.push_projective(*h, *affine),
                Item::PushBlur(r) => painter.push_blur(*r),
                Item::PopBlur => painter.pop_blur(),
                Item::PopTransform => painter.pop_transform(),
                Item::Subpixel(on) => painter.set_subpixel(*on),
                Item::PushEffect { rect, effect, time } => {
                    painter.push_effect(*rect, effect, *time)
                }
                Item::PopEffect => painter.pop_effect(),
                Item::Image {
                    rect,
                    image,
                    opacity,
                } => painter.image(*rect, image, *opacity),
                Item::Surface { rect, id, opacity } => painter.surface(*rect, *id, *opacity),
                Item::Backdrop {
                    rect,
                    radius,
                    blur,
                    saturate,
                } => painter.backdrop(*rect, *radius, *blur, *saturate),
                Item::Shadow {
                    rect,
                    radius,
                    blur,
                    color,
                } => painter.shadow(*rect, *radius, *blur, *color),
                Item::Underline {
                    x,
                    y,
                    width,
                    thickness,
                    color,
                    wavy,
                } => painter.underline(*x, *y, *width, *thickness, *color, *wavy),
            }
        }
    }
}

impl Scene {
    /// A painter that appends to this scene: for painting more over a
    /// recorded frame (an inspector highlight, the debug overlay).
    pub fn recorder(&mut self) -> SceneRecorder<'_> {
        SceneRecorder(self)
    }
}

/// Records instead of painting. Painting never measures text — layout did
/// that — so measuring here is a mistake in this crate, not the caller's.
pub struct SceneRecorder<'a>(&'a mut Scene);

impl TextMeasure for SceneRecorder<'_> {
    fn measure(&mut self, _: &str, _: &Style) -> (f32, f32) {
        unreachable!("painting does not measure text")
    }
}

impl Painter for SceneRecorder<'_> {
    fn fill_rect(&mut self, rect: Rect, color: Rgba) {
        self.0.items.push(Item::Rect {
            rect,
            rx: 0.0,
            ry: 0.0,
            color,
        });
    }

    fn fill_rounded_rect(&mut self, rect: Rect, rx: f32, ry: f32, color: Rgba) {
        self.0.items.push(Item::Rect {
            rect,
            rx,
            ry,
            color,
        });
    }

    fn fill_gradient(
        &mut self,
        rect: Rect,
        shape: crate::GradientShape,
        stops: &[(f32, Rgba)],
        blur: f32,
    ) {
        if shape.is_across() {
            return self.fill_linear_gradient(rect, stops, blur);
        }
        self.0.items.push(Item::ShapedGradient {
            rect,
            shape,
            stops: stops.to_vec(),
        });
    }

    fn fill_linear_gradient(&mut self, rect: Rect, stops: &[(f32, Rgba)], blur: f32) {
        self.0.items.push(Item::Gradient {
            rect,
            stops: stops.to_vec(),
            blur,
        });
    }

    fn text(&mut self, line: Rect, text: &str, style: &TextStyle) {
        self.0.items.push(Item::Text {
            line,
            text: text.to_owned(),
            style: style.clone(),
        });
    }

    fn path(
        &mut self,
        rect: Rect,
        d: &[String],
        view_box: [f32; 4],
        stroke: Option<f32>,
        fill: bool,
        color: Rgba,
    ) {
        self.0.items.push(Item::Path {
            rect,
            d: d.to_vec(),
            view_box,
            stroke,
            fill,
            color,
        });
    }

    fn stroke_arc(&mut self, arc: crate::paint::ArcShape, width: f32, color: Rgba) {
        self.0.items.push(Item::Arc { arc, width, color });
    }

    fn push_clip(&mut self, rect: Rect) {
        self.0.items.push(Item::PushClip {
            rect,
            rx: 0.0,
            ry: 0.0,
        });
    }

    fn push_clip_rounded(&mut self, rect: Rect, rx: f32, ry: f32) {
        self.0.items.push(Item::PushClip { rect, rx, ry });
    }

    fn pop_clip(&mut self) {
        self.0.items.push(Item::PopClip);
    }

    fn push_transform(&mut self, m: [f32; 6]) {
        self.0.items.push(Item::PushTransform(m));
    }

    fn pop_transform(&mut self) {
        self.0.items.push(Item::PopTransform);
    }

    fn push_projective(&mut self, h: [f32; 9], affine: [f32; 6]) {
        self.0.items.push(Item::PushProjective { h, affine });
    }

    fn push_blur(&mut self, radius: f32) {
        self.0.items.push(Item::PushBlur(radius));
    }

    fn pop_blur(&mut self) {
        self.0.items.push(Item::PopBlur);
    }

    fn push_effect(&mut self, rect: Rect, effect: &crate::shader::Effect, time: f32) {
        self.0.items.push(Item::PushEffect {
            rect,
            effect: effect.clone(),
            time,
        });
    }

    fn pop_effect(&mut self) {
        self.0.items.push(Item::PopEffect);
    }

    fn set_subpixel(&mut self, on: bool) {
        self.0.items.push(Item::Subpixel(on));
    }

    fn shadow(&mut self, rect: Rect, radius: f32, blur: f32, color: Rgba) {
        self.0.items.push(Item::Shadow {
            rect,
            radius,
            blur,
            color,
        });
    }

    fn underline(&mut self, x: f32, y: f32, width: f32, thickness: f32, color: Rgba, wavy: bool) {
        self.0.items.push(Item::Underline {
            x,
            y,
            width,
            thickness,
            color,
            wavy,
        });
    }

    fn surface(&mut self, rect: Rect, id: u64, opacity: f32) {
        self.0.items.push(Item::Surface { rect, id, opacity });
    }

    fn backdrop(&mut self, rect: Rect, radius: f32, blur: f32, saturate: f32) {
        self.0.items.push(Item::Backdrop {
            rect,
            radius,
            blur,
            saturate,
        });
    }

    fn image(&mut self, rect: Rect, image: &crate::image::ImageRef, opacity: f32) {
        self.0.items.push(Item::Image {
            rect,
            image: image.clone(),
            opacity,
        });
    }
}

impl Ui {
    /// The tree as last laid out, painted into a [`Scene`].
    pub fn scene(&self) -> Scene {
        let mut scene = Scene::default();
        self.paint(&mut scene.recorder());
        scene
    }
}

/// What differs between two frames' scenes, for a painter that keeps the
/// last frame and paints only what changed (see [`Scene::damage`]).
#[derive(Debug, Clone, PartialEq)]
pub enum SceneDamage {
    /// They draw the same.
    None,
    /// These rectangles (UI pixels, whole, overlapping none of the others).
    Regions(Vec<Rect>),
    /// Paint it all.
    Full,
}

/// Where an item draws, as far as a scene can tell: its own box, padded
/// for what reaches past it, through the transforms and clips around it.
struct Walk {
    /// The transform in force, a 3 × 3 by rows.
    xform: Vec<[f32; 9]>,
    /// The clip in force, already in window space.
    clip: Vec<Option<Rect>>,
    /// Boxes drawn through a shader: anything inside may land anywhere in
    /// them, so an item there stands for the whole of it.
    effect: Vec<Rect>,
    /// Blur in force: what is drawn spreads by three times it.
    blur: Vec<f32>,
    /// The text mode last set.
    subpixel: Option<bool>,
    view: Rect,
}

/// What an item is drawn under, besides itself: two equal items under
/// the same draw the same.
#[derive(PartialEq)]
struct Under {
    xform: [f32; 9],
    clip: Option<Rect>,
    effect: Option<Rect>,
    blur: f32,
    subpixel: Option<bool>,
}

impl Walk {
    fn new(view: Rect) -> Self {
        Walk {
            xform: vec![IDENTITY],
            clip: vec![None],
            effect: vec![],
            blur: vec![0.0],
            subpixel: None,
            view,
        }
    }

    fn under(&self) -> Under {
        Under {
            xform: *self.xform.last().expect("a transform"),
            clip: self.clip.last().copied().flatten(),
            effect: self.effect.last().copied(),
            blur: self.blur.last().copied().unwrap_or(0.0),
            subpixel: self.subpixel,
        }
    }

    /// `r` in window space, its bounding box; the whole window when part
    /// of it is behind the eye.
    fn place(&self, r: Rect) -> Rect {
        let h = *self.xform.last().expect("a transform");
        let mut lo = (f32::MAX, f32::MAX);
        let mut hi = (f32::MIN, f32::MIN);
        for (x, y) in [
            (r.x, r.y),
            (r.x + r.width, r.y),
            (r.x, r.y + r.height),
            (r.x + r.width, r.y + r.height),
        ] {
            let w = h[6] * x + h[7] * y + h[8];
            if w <= 1e-6 {
                return self.view;
            }
            let (px, py) = (
                (h[0] * x + h[1] * y + h[2]) / w,
                (h[3] * x + h[4] * y + h[5]) / w,
            );
            lo = (lo.0.min(px), lo.1.min(py));
            hi = (hi.0.max(px), hi.1.max(py));
        }
        Rect {
            x: lo.0,
            y: lo.1,
            width: hi.0 - lo.0,
            height: hi.1 - lo.1,
        }
    }

    /// Where an item whose own box is `r` (padded by `pad`) shows: inside
    /// an effect, all of the effect's box.
    fn bounds(&self, r: Rect, pad: f32) -> Option<Rect> {
        if let Some(e) = self.effect.last() {
            return Some(*e);
        }
        let spread = pad + 3.0 * self.blur.last().copied().unwrap_or(0.0);
        let r = self.place(Rect {
            x: r.x - spread,
            y: r.y - spread,
            width: r.width + 2.0 * spread,
            height: r.height + 2.0 * spread,
        });
        let r = match self.clip.last().copied().flatten() {
            Some(c) => intersect(r, c)?,
            None => r,
        };
        intersect(r, self.view)
    }

    /// One item's mark on the window (`None`: it draws nothing itself),
    /// and its effect on what follows.
    fn step(&mut self, item: &Item) -> Option<Rect> {
        let top = |v: &Vec<[f32; 9]>| *v.last().expect("a transform");
        match item {
            Item::Rect { rect, .. }
            | Item::Gradient { rect, .. }
            | Item::ShapedGradient { rect, .. }
            | Item::Image { rect, .. }
            | Item::Surface { rect, .. }
            | Item::Backdrop { rect, .. } => self.bounds(*rect, 1.0),
            Item::Text { line, style, .. } => {
                // Glyphs reach past the line box (ascenders, an italic's
                // lean), and a run is drawn into a sprite wider than it.
                let fs = style.font_size.max(1.0);
                self.bounds(
                    Rect {
                        x: line.x - fs,
                        y: line.y - fs * 0.5,
                        width: line.width * 1.25 + fs * 2.0,
                        height: line.height + fs,
                    },
                    1.0,
                )
            }
            Item::Path { rect, stroke, .. } => {
                self.bounds(*rect, stroke.unwrap_or(0.0) * 2.0 + 2.0)
            }
            Item::Arc { arc, width, .. } => self.bounds(arc.bounds(width / 2.0), 2.0),
            Item::Shadow { rect, blur, .. } => self.bounds(*rect, blur * 1.5 + 2.0),
            Item::Underline {
                x,
                y,
                width,
                thickness,
                ..
            } => {
                let reach = thickness * 3.0 + 2.0;
                self.bounds(
                    Rect {
                        x: *x,
                        y: y - reach,
                        width: *width,
                        height: reach * 2.0,
                    },
                    1.0,
                )
            }
            Item::PushClip { rect, .. } => {
                let placed = self.place(*rect);
                let c = match self.clip.last().copied().flatten() {
                    Some(c) => intersect(placed, c).unwrap_or(Rect {
                        x: c.x,
                        y: c.y,
                        width: 0.0,
                        height: 0.0,
                    }),
                    None => placed,
                };
                self.clip.push(Some(c));
                None
            }
            Item::PopClip => {
                if self.clip.len() > 1 {
                    self.clip.pop();
                }
                None
            }
            Item::PushTransform(m) => {
                let h = [m[0], m[2], m[4], m[1], m[3], m[5], 0.0, 0.0, 1.0];
                let t = top(&self.xform);
                self.xform.push(compose(t, h));
                None
            }
            Item::PushProjective { h, .. } => {
                let t = top(&self.xform);
                self.xform.push(compose(t, *h));
                None
            }
            Item::PopTransform => {
                if self.xform.len() > 1 {
                    self.xform.pop();
                }
                None
            }
            Item::PushBlur(b) => {
                let now = self.blur.last().copied().unwrap_or(0.0);
                self.blur.push(now + b.max(0.0));
                None
            }
            Item::PopBlur => {
                if self.blur.len() > 1 {
                    self.blur.pop();
                }
                None
            }
            Item::PushEffect { rect, effect, .. } => {
                let o = effect.overflow;
                // The effect's whole box, before the effect is in force.
                let whole = self.bounds(
                    Rect {
                        x: rect.x - o,
                        y: rect.y - o,
                        width: rect.width + 2.0 * o,
                        height: rect.height + 2.0 * o,
                    },
                    1.0,
                );
                let whole = whole.unwrap_or(Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 0.0,
                });
                self.effect.push(whole);
                Some(whole)
            }
            Item::PopEffect => {
                self.effect.pop();
                None
            }
            Item::Subpixel(on) => {
                self.subpixel = Some(*on);
                None
            }
        }
    }
}

const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

fn compose(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    std::array::from_fn(|i| {
        let (r, c) = (i / 3, i % 3);
        (0..3).map(|k| a[r * 3 + k] * b[k * 3 + c]).sum()
    })
}

fn intersect(a: Rect, b: Rect) -> Option<Rect> {
    let (x0, y0) = (a.x.max(b.x), a.y.max(b.y));
    let (x1, y1) = (
        (a.x + a.width).min(b.x + b.width),
        (a.y + a.height).min(b.y + b.height),
    );
    (x1 > x0 && y1 > y0).then_some(Rect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

impl Scene {
    /// Where this scene draws differently from `prev`, in a window
    /// `width × height` UI pixels: the places of items that are not the
    /// same (or not under the same transforms and clips) in both, joined
    /// into a few rectangles. Surfaces always count as changed (their
    /// pictures change outside the scene). Much of the window changed is
    /// [`SceneDamage::Full`].
    pub fn damage(&self, prev: &Scene, width: f32, height: f32) -> SceneDamage {
        let view = Rect {
            x: 0.0,
            y: 0.0,
            width,
            height,
        };
        // Each item's place and what it is drawn under, both frames.
        let marks = |s: &Scene| -> (Vec<Option<Rect>>, Vec<Under>) {
            let mut w = Walk::new(view);
            s.items
                .iter()
                .map(|i| {
                    let under = w.under();
                    (w.step(i), under)
                })
                .unzip()
        };
        let (a, b) = (&prev.items, &self.items);
        let ((ma, ua), (mb, ub)) = (marks(prev), marks(self));
        let mut rects: Vec<Rect> = vec![];
        if a.len() == b.len() {
            // The same items: those that differ, or are drawn under
            // something that does (a transform, a clip), changed.
            for i in 0..a.len() {
                if a[i] != b[i] || ua[i] != ub[i] || ma[i] != mb[i] {
                    rects.extend(ma[i]);
                    rects.extend(mb[i]);
                }
            }
        } else {
            // The same from the start and from the end; what lies between
            // changed. An item matches only where everything before it in
            // the frame does too (so its transforms and clips are the same).
            let head = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
            let tail = a[head..]
                .iter()
                .rev()
                .zip(b[head..].iter().rev())
                .take_while(|(x, y)| x == y)
                .count();
            // A tail matched by its items alone may sit under a changed
            // transform or clip: its marks have to agree too.
            let tail = (0..tail)
                .take_while(|&k| {
                    ma[a.len() - 1 - k] == mb[b.len() - 1 - k]
                        && ua[a.len() - 1 - k] == ub[b.len() - 1 - k]
                })
                .count();
            rects.extend(ma[head..a.len() - tail].iter().flatten().copied());
            rects.extend(mb[head..b.len() - tail].iter().flatten().copied());
        }
        // Surfaces: drawn anew each frame, wherever they are.
        for (i, item) in b.iter().enumerate() {
            if matches!(item, Item::Surface { .. })
                && let Some(r) = mb[i]
            {
                rects.push(r);
            }
        }
        // A material shows what is under it, blurred: anything changed
        // within its reach changes all of it.
        for (i, item) in b.iter().enumerate() {
            if let (Item::Backdrop { blur, .. }, Some(r)) = (item, mb[i]) {
                let reach = blur * 3.0;
                let near = Rect {
                    x: r.x - reach,
                    y: r.y - reach,
                    width: r.width + 2.0 * reach,
                    height: r.height + 2.0 * reach,
                };
                if rects.iter().any(|&q| intersect(q, near).is_some()) {
                    rects.push(r);
                }
            }
        }
        if rects.is_empty() {
            return SceneDamage::None;
        }
        // Whole pixels, joined where they touch, and few.
        let mut rects: Vec<Rect> = rects
            .into_iter()
            .filter_map(|r| {
                let (x0, y0) = (r.x.floor().max(0.0), r.y.floor().max(0.0));
                let (x1, y1) = (
                    (r.x + r.width).ceil().min(width),
                    (r.y + r.height).ceil().min(height),
                );
                (x1 > x0 && y1 > y0).then_some(Rect {
                    x: x0,
                    y: y0,
                    width: x1 - x0,
                    height: y1 - y0,
                })
            })
            .collect();
        let join = |a: Rect, b: Rect| {
            let (x0, y0) = (a.x.min(b.x), a.y.min(b.y));
            Rect {
                x: x0,
                y: y0,
                width: (a.x + a.width).max(b.x + b.width) - x0,
                height: (a.y + a.height).max(b.y + b.height) - y0,
            }
        };
        let area = |r: Rect| r.width * r.height;
        // Very many: one box round them all (joining pair by pair would
        // cost more than drawing).
        if rects.len() > 64 {
            let all = rects.iter().copied().reduce(join).expect("some");
            rects = vec![all];
        }
        loop {
            // Join any two that overlap or touch; past eight, the pair whose
            // join wastes least.
            let mut best: Option<(usize, usize, f32)> = None;
            for i in 0..rects.len() {
                for j in i + 1..rects.len() {
                    let (p, q) = (rects[i], rects[j]);
                    let touch = p.x <= q.x + q.width
                        && q.x <= p.x + p.width
                        && p.y <= q.y + q.height
                        && q.y <= p.y + p.height;
                    let waste = area(join(p, q)) - area(p) - area(q);
                    let cost = if touch { f32::MIN } else { waste };
                    if best.is_none_or(|(_, _, c)| cost < c) {
                        best = Some((i, j, cost));
                    }
                }
            }
            match best {
                Some((i, j, cost)) if cost == f32::MIN || rects.len() > 8 => {
                    let q = rects.swap_remove(j);
                    rects[i] = join(rects[i], q);
                }
                _ => break,
            }
        }
        let changed: f32 = rects.iter().map(|&r| area(r)).sum();
        if changed > 0.6 * width * height {
            return SceneDamage::Full;
        }
        SceneDamage::Regions(rects)
    }
}

#[cfg(test)]
mod damage_tests {
    use super::*;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect {
            x,
            y,
            width: w,
            height: h,
        }
    }

    fn fill(r: Rect, c: f32) -> Item {
        Item::Rect {
            rect: r,
            rx: 0.0,
            ry: 0.0,
            color: [c, c, c, 1.0],
        }
    }

    fn inside(outer: &Rect, inner: Rect) -> bool {
        outer.x <= inner.x
            && outer.y <= inner.y
            && outer.x + outer.width >= inner.x + inner.width
            && outer.y + outer.height >= inner.y + inner.height
    }

    fn regions(d: SceneDamage) -> Vec<Rect> {
        match d {
            SceneDamage::Regions(r) => r,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn two_changes_far_apart_damage_their_places_not_all_between() {
        // A backdrop, many boxes in a row, the first and the last changing.
        let mut items = vec![fill(rect(0.0, 0.0, 1000.0, 800.0), 0.9)];
        items.extend((0..300).map(|i| {
            fill(
                rect(
                    (i % 30) as f32 * 30.0,
                    (i / 30) as f32 * 30.0 + 100.0,
                    20.0,
                    20.0,
                ),
                0.3,
            )
        }));
        let base = Scene { items };
        let mut b = base.clone();
        b.items[1] = fill(rect(0.0, 100.0, 20.0, 20.0), 0.6);
        b.items[300] = fill(rect(870.0, 370.0, 20.0, 20.0), 0.6);
        let r = regions(b.damage(&base, 1000.0, 800.0));
        let area: f32 = r.iter().map(|r| r.width * r.height).sum();
        assert!(area <= 2.0 * 22.0 * 22.0, "{r:?}");
        // A transform changed over some: those, and not the rest.
        let mut c = base.clone();
        c.items
            .insert(150, Item::PushTransform([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
        c.items.insert(153, Item::PopTransform);
        let mut d = c.clone();
        d.items[150] = Item::PushTransform([1.0, 0.0, 0.0, 1.0, 5.0, 0.0]);
        let r = regions(d.damage(&c, 1000.0, 800.0));
        let area: f32 = r.iter().map(|r| r.width * r.height).sum();
        assert!(area > 0.0 && area < 6.0 * 30.0 * 30.0, "{r:?}");
    }

    #[test]
    fn only_what_changed_is_damaged() {
        let base = Scene {
            items: vec![
                fill(rect(0.0, 0.0, 800.0, 600.0), 0.9),
                fill(rect(10.0, 10.0, 50.0, 30.0), 0.2),
                fill(rect(400.0, 300.0, 60.0, 20.0), 0.2),
            ],
        };
        assert_eq!(base.damage(&base, 800.0, 600.0), SceneDamage::None);
        // One box recoloured: its place alone.
        let mut b = base.clone();
        b.items[2] = fill(rect(400.0, 300.0, 60.0, 20.0), 0.5);
        let r = regions(b.damage(&base, 800.0, 600.0));
        assert_eq!(r.len(), 1);
        assert!(
            inside(&r[0], rect(400.0, 300.0, 60.0, 20.0)) && r[0].width < 70.0,
            "{r:?}"
        );
        // One added in the middle: its place.
        let mut c = base.clone();
        c.items.insert(2, fill(rect(200.0, 100.0, 10.0, 10.0), 0.1));
        let r = regions(c.damage(&base, 800.0, 600.0));
        assert!(
            r.iter().any(|q| inside(q, rect(200.0, 100.0, 10.0, 10.0)))
                && r.iter().all(|q| q.width < 30.0),
            "{r:?}"
        );
    }

    #[test]
    fn an_effect_that_moves_on_damages_all_of_its_box_and_a_scroll_everything() {
        let shader = crate::shader::Shader::new(
            "fn f(p: vec2<f32>, time: f32) -> vec4<f32> { return layer(p); }",
        );
        let effect = crate::shader::Effect::new(&shader).overflow(4.0);
        let frame = |time: f32| Scene {
            items: vec![
                fill(rect(0.0, 0.0, 800.0, 600.0), 0.9),
                Item::PushEffect {
                    rect: rect(100.0, 100.0, 180.0, 120.0),
                    effect: effect.clone(),
                    time,
                },
                fill(rect(100.0, 100.0, 180.0, 120.0), 0.3),
                Item::PopEffect,
                fill(rect(500.0, 400.0, 50.0, 50.0), 0.2),
            ],
        };
        let r = regions(frame(1.0).damage(&frame(0.0), 800.0, 600.0));
        assert_eq!(r.len(), 1);
        assert!(
            inside(&r[0], rect(96.0, 96.0, 188.0, 128.0)) && r[0].width < 200.0,
            "{r:?}"
        );
        // Everything moved up: all of it.
        let scrolled = |dy: f32| Scene {
            items: (0..40)
                .map(|i| fill(rect(0.0, i as f32 * 20.0 - dy, 800.0, 18.0), 0.5))
                .collect(),
        };
        assert_eq!(
            scrolled(5.0).damage(&scrolled(0.0), 800.0, 600.0),
            SceneDamage::Full
        );
    }

    #[test]
    fn surfaces_count_as_changed_and_a_changed_transform_moves_what_it_holds() {
        let s = Scene {
            items: vec![
                fill(rect(0.0, 0.0, 800.0, 600.0), 0.9),
                Item::Surface {
                    rect: rect(10.0, 10.0, 100.0, 100.0),
                    id: 1,
                    opacity: 1.0,
                },
            ],
        };
        let r = regions(s.damage(&s, 800.0, 600.0));
        assert!(inside(&r[0], rect(10.0, 10.0, 100.0, 100.0)));
        // The same box under another transform: both places.
        let t = |dx: f32| Scene {
            items: vec![
                fill(rect(0.0, 0.0, 800.0, 600.0), 0.9),
                Item::PushTransform([1.0, 0.0, 0.0, 1.0, dx, 0.0]),
                fill(rect(100.0, 100.0, 40.0, 40.0), 0.2),
                Item::PopTransform,
            ],
        };
        let r = regions(t(200.0).damage(&t(0.0), 800.0, 600.0));
        assert!(
            r.iter().any(|q| inside(q, rect(100.0, 100.0, 40.0, 40.0))),
            "{r:?}"
        );
        assert!(
            r.iter().any(|q| inside(q, rect(300.0, 100.0, 40.0, 40.0))),
            "{r:?}"
        );
    }
}
