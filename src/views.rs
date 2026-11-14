//! Many cameras at once: named cameras, viewed side by side or one at a
//! time, switched or blended between, and seen as textures — on a screen in
//! the scene, inside another view, or inside themselves.
//!
//! - [`CameraSet`] keeps named cameras and which is live. [`switch_to`] cuts
//!   to another; [`transition_to`] goes there over time: a [`Morph`] flies
//!   the camera from one pose and lens to the other, a [`Crossfade`] blends
//!   the two pictures, a [`Wipe`] slides the new one across.
//! - [`Layout`] puts several cameras' views on one target: split in two, a
//!   grid, a picture in a picture, or any rectangles. [`CameraSet::render`]
//!   draws the live camera (or the layout) into a renderer's own target.
//! - [`CameraTexture`] renders a camera into a texture of its own, for any
//!   material: a monitor, a minimap, a portal, a rear-view mirror. It keeps
//!   two pictures, so a camera may see its own (it shows the one before).
//! - [`Feedback`] is a camera that sees its own screen, `levels` deep each
//!   frame — a corridor of screens, a hall of mirrors.
//!
//! ```no_run
//! use threers::views::{CameraSet, Layout, TransitionKind};
//! use threers::prelude::*;
//!
//! let mut cams = CameraSet::new();
//! cams.add("front", PerspectiveCamera::new(50.0, 1.6, 0.1, 100.0));
//! cams.add("top", PerspectiveCamera::new(35.0, 1.6, 0.1, 100.0));
//! // Each frame, with `hr` a renderer and `scene` the scene:
//! # let mut hr = HeadlessRenderer::builder().size(640, 400).build().unwrap();
//! # let mut scene = Scene::new();
//! cams.render(&mut hr, &mut scene, &Layout::split_vertical("front", "top"));
//! cams.transition_to("top", TransitionKind::Morph, 1.2);
//! cams.update(1.0 / 60.0);
//! cams.render(&mut hr, &mut scene, &Layout::Live);
//! ```
//!
//! [`switch_to`]: CameraSet::switch_to
//! [`transition_to`]: CameraSet::transition_to
//! [`Morph`]: TransitionKind::Morph
//! [`Crossfade`]: TransitionKind::Crossfade
//! [`Wipe`]: TransitionKind::Wipe

use std::sync::Arc;

use crate::cameras::{Camera, OrthographicCamera, PerspectiveCamera};
use crate::core::{Mesh, Object3D};
use crate::easing::Easing;
use crate::geometries::PlaneGeometry;
use crate::materials::BasicMaterial;
use crate::math::{Color, Vector3};
#[cfg(not(target_arch = "wasm32"))]
use crate::renderer::HeadlessRenderer;
use crate::renderer::{RenderTarget, Renderer};
use crate::scene::Scene;
use crate::textures::{Texture, TextureFormat};

/// Something to draw cameras into: a renderer with a target of its own.
pub trait ViewTarget {
    /// The renderer, for its viewport, scissor and render targets.
    fn renderer(&mut self) -> Option<&mut Renderer>;
    /// Draws `scene` from `camera` into its own target.
    fn draw(&mut self, scene: &mut Scene, camera: &dyn Camera);
}

#[cfg(not(target_arch = "wasm32"))]
impl ViewTarget for HeadlessRenderer {
    fn renderer(&mut self) -> Option<&mut Renderer> {
        Some(HeadlessRenderer::renderer(self))
    }
    fn draw(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        self.render(scene, camera);
    }
}

#[cfg(winit_window)]
impl ViewTarget for crate::renderer::WindowRenderer {
    fn renderer(&mut self) -> Option<&mut Renderer> {
        crate::renderer::WindowRenderer::renderer(self)
    }
    fn draw(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        self.render(scene, camera);
    }
}

/// A rectangle as shares of the target (0 to 1), from its top left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frac {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Frac {
    pub const WHOLE: Frac = Frac {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    };

    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// Where a picture in a picture sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Which cameras show where on the target.
#[derive(Debug, Clone, PartialEq)]
pub enum Layout {
    /// The live camera (mid-transition, as the transition has it), whole.
    Live,
    /// One camera, whole.
    One(String),
    /// Side by side (`vertical`: a line down the middle) or one over the
    /// other, the first taking `ratio` of the room.
    Split {
        first: String,
        second: String,
        vertical: bool,
        ratio: f32,
        gap: f32,
    },
    /// In a grid `columns` across, row by row.
    Grid {
        cameras: Vec<String>,
        columns: usize,
        gap: f32,
    },
    /// `main` whole, `inset` small in a corner (`size`: its share of the
    /// width), `margin` in from the edges, in pixels.
    PictureInPicture {
        main: String,
        inset: String,
        corner: Corner,
        size: f32,
        margin: f32,
    },
    /// Any cameras, anywhere, drawn in order (later over earlier).
    Custom(Vec<(String, Frac)>),
}

impl Layout {
    /// Two side by side, halves, a 2 px gap.
    pub fn split_vertical(first: &str, second: &str) -> Self {
        Layout::Split {
            first: first.into(),
            second: second.into(),
            vertical: true,
            ratio: 0.5,
            gap: 2.0,
        }
    }

    /// One over the other, halves, a 2 px gap.
    pub fn split_horizontal(first: &str, second: &str) -> Self {
        Layout::Split {
            first: first.into(),
            second: second.into(),
            vertical: false,
            ratio: 0.5,
            gap: 2.0,
        }
    }

    /// A grid as square as the cameras allow.
    pub fn grid(cameras: &[&str]) -> Self {
        let columns = (cameras.len() as f32).sqrt().ceil().max(1.0) as usize;
        Layout::Grid {
            cameras: cameras.iter().map(|c| c.to_string()).collect(),
            columns,
            gap: 2.0,
        }
    }

    /// `inset` a quarter as wide as `main`, top right.
    pub fn picture_in_picture(main: &str, inset: &str) -> Self {
        Layout::PictureInPicture {
            main: main.into(),
            inset: inset.into(),
            corner: Corner::TopRight,
            size: 0.28,
            margin: 12.0,
        }
    }

    /// Each camera's rectangle in a `width` × `height` target (pixels, top
    /// left), in drawing order; `live` names the live camera.
    pub fn rects(&self, width: f32, height: f32, live: &str) -> Vec<(String, [f32; 4])> {
        match self {
            Layout::Live => vec![(live.to_owned(), [0.0, 0.0, width, height])],
            Layout::One(c) => vec![(c.clone(), [0.0, 0.0, width, height])],
            Layout::Split {
                first,
                second,
                vertical,
                ratio,
                gap,
            } => {
                let r = ratio.clamp(0.0, 1.0);
                if *vertical {
                    let w1 = ((width - gap) * r).max(0.0);
                    vec![
                        (first.clone(), [0.0, 0.0, w1, height]),
                        (
                            second.clone(),
                            [w1 + gap, 0.0, (width - w1 - gap).max(0.0), height],
                        ),
                    ]
                } else {
                    let h1 = ((height - gap) * r).max(0.0);
                    vec![
                        (first.clone(), [0.0, 0.0, width, h1]),
                        (
                            second.clone(),
                            [0.0, h1 + gap, width, (height - h1 - gap).max(0.0)],
                        ),
                    ]
                }
            }
            Layout::Grid {
                cameras,
                columns,
                gap,
            } => {
                let cols = (*columns).max(1);
                let rows = cameras.len().div_ceil(cols).max(1);
                let cw = (width - gap * (cols - 1) as f32) / cols as f32;
                let ch = (height - gap * (rows - 1) as f32) / rows as f32;
                cameras
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        (
                            c.clone(),
                            [
                                (i % cols) as f32 * (cw + gap),
                                (i / cols) as f32 * (ch + gap),
                                cw,
                                ch,
                            ],
                        )
                    })
                    .collect()
            }
            Layout::PictureInPicture {
                main,
                inset,
                corner,
                size,
                margin,
            } => {
                let w = width * size.clamp(0.05, 1.0);
                let h = w * height / width.max(1.0);
                let x = match corner {
                    Corner::TopLeft | Corner::BottomLeft => *margin,
                    _ => width - w - margin,
                };
                let y = match corner {
                    Corner::TopLeft | Corner::TopRight => *margin,
                    _ => height - h - margin,
                };
                vec![
                    (main.clone(), [0.0, 0.0, width, height]),
                    (inset.clone(), [x, y, w, h]),
                ]
            }
            Layout::Custom(list) => list
                .iter()
                .map(|(c, f)| {
                    (
                        c.clone(),
                        [
                            f.x * width,
                            f.y * height,
                            f.width * width,
                            f.height * height,
                        ],
                    )
                })
                .collect(),
        }
    }
}

/// How one camera gives way to another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionKind {
    /// At once.
    Cut,
    /// The camera flies from one to the other: its place, where it looks,
    /// its lens.
    Morph,
    /// The two pictures blended, the new one coming up through the old.
    Crossfade,
    /// The new picture sliding across the old, from the left (`0`), the
    /// top (`90`), the right (`180`) or the bottom (`270`) degrees.
    Wipe(f32),
}

/// A transition under way.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub from: String,
    pub to: String,
    pub kind: TransitionKind,
    pub duration: f32,
    pub easing: Easing,
    /// Seconds into it.
    pub elapsed: f32,
}

impl Transition {
    /// How far along, 0 to 1, eased.
    pub fn progress(&self) -> f32 {
        self.easing
            .apply((self.elapsed / self.duration.max(1e-6)).clamp(0.0, 1.0))
    }
}

/// Named cameras, one live. See the [module docs](self).
#[derive(Default)]
pub struct CameraSet {
    cameras: Vec<(String, PerspectiveCamera)>,
    live: usize,
    transition: Option<Transition>,
    /// Targets for a crossfade, made when one first runs.
    fade: Option<Fade>,
}

/// Two pictures and the quads that blend them.
struct Fade {
    a: CameraTexture,
    b: CameraTexture,
    scene: Scene,
    camera: OrthographicCamera,
    /// The quad showing the new picture, over the old.
    top: crate::core::ObjectId,
}

impl CameraSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds (or replaces) the camera `name`; the first added is live.
    pub fn add(&mut self, name: &str, camera: PerspectiveCamera) -> &mut Self {
        match self.cameras.iter_mut().find(|(n, _)| n == name) {
            Some((_, c)) => *c = camera,
            None => self.cameras.push((name.to_owned(), camera)),
        }
        self
    }

    pub fn get(&self, name: &str) -> Option<&PerspectiveCamera> {
        self.cameras.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut PerspectiveCamera> {
        self.cameras
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.cameras.iter().map(|(n, _)| n.as_str())
    }

    /// The live camera's name (the one being gone to, mid-transition).
    pub fn live(&self) -> &str {
        self.cameras.get(self.live).map_or("", |(n, _)| n.as_str())
    }

    /// The transition under way, if one is.
    pub fn transition(&self) -> Option<&Transition> {
        self.transition.as_ref()
    }

    /// Cuts to `name`; `false` if there is no such camera.
    pub fn switch_to(&mut self, name: &str) -> bool {
        match self.cameras.iter().position(|(n, _)| n == name) {
            Some(i) => {
                self.live = i;
                self.transition = None;
                true
            }
            None => false,
        }
    }

    /// Goes to `name` over `seconds`, eased in and out; `false` if there is
    /// no such camera. Call [`update`](Self::update) each frame.
    pub fn transition_to(&mut self, name: &str, kind: TransitionKind, seconds: f32) -> bool {
        self.transition_eased(name, kind, seconds, Easing::CubicInOut)
    }

    /// [`transition_to`](Self::transition_to) with an easing of its own.
    pub fn transition_eased(
        &mut self,
        name: &str,
        kind: TransitionKind,
        seconds: f32,
        easing: Easing,
    ) -> bool {
        let Some(to) = self.cameras.iter().position(|(n, _)| n == name) else {
            return false;
        };
        if matches!(kind, TransitionKind::Cut) || seconds <= 0.0 || to == self.live {
            return self.switch_to(name);
        }
        let from = self.live().to_owned();
        self.live = to;
        self.transition = Some(Transition {
            from,
            to: name.to_owned(),
            kind,
            duration: seconds,
            easing,
            elapsed: 0.0,
        });
        true
    }

    /// Moves a transition on by `dt` seconds, ending it when it is done.
    pub fn update(&mut self, dt: f32) {
        if let Some(t) = &mut self.transition {
            t.elapsed += dt.max(0.0);
            if t.elapsed >= t.duration {
                self.transition = None;
            }
        }
    }

    /// The camera to draw the live view with: the live one, or mid-morph,
    /// the one between.
    pub fn current(&self) -> PerspectiveCamera {
        let live = self
            .cameras
            .get(self.live)
            .map(|(_, c)| c.clone())
            .unwrap_or_else(|| PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0));
        match &self.transition {
            Some(t) if t.kind == TransitionKind::Morph => match self.get(&t.from) {
                Some(from) => between(from, &live, t.progress()),
                None => live,
            },
            _ => live,
        }
    }

    /// Draws `layout` into `target`'s own picture: each camera into its
    /// rectangle, at that rectangle's aspect (the cameras' own aspects are
    /// left as they are). The live view plays its transition.
    pub fn render(&mut self, target: &mut dyn ViewTarget, scene: &mut Scene, layout: &Layout) {
        let Some((w, h)) = target.renderer().map(|r| r.size()) else {
            // No renderer to place views with (the native Metal backend):
            // the live camera, whole.
            target.draw(scene, &self.current());
            return;
        };
        let live = self.live().to_owned();
        let rects = layout.rects(w, h, &live);
        let single = rects.len() == 1 && rects[0].1 == [0.0, 0.0, w, h];
        for (name, [x, y, rw, rh]) in rects {
            if rw < 1.0 || rh < 1.0 {
                continue;
            }
            if !single {
                let r = target.renderer().expect("checked above");
                // Bottom-left origin, as three.js's viewport has it.
                r.set_viewport(x, h - y - rh, rw, rh);
                r.set_scissor(x, h - y - rh, rw, rh);
                r.set_scissor_test(true);
            }
            if name == live && self.transition.is_some() {
                self.render_transition(target, scene, [x, y, rw, rh], h);
            } else {
                let mut cam = if name == live {
                    self.current()
                } else {
                    self.get(&name).cloned().unwrap_or_else(|| self.current())
                };
                cam.aspect = rw / rh;
                target.draw(scene, &cam);
            }
        }
        if let Some(r) = target.renderer() {
            r.reset_viewport();
            r.reset_scissor();
            r.set_scissor_test(false);
        }
    }

    /// The live view mid-transition, into the rectangle `rect` (top left)
    /// of a target `h` tall.
    fn render_transition(
        &mut self,
        target: &mut dyn ViewTarget,
        scene: &mut Scene,
        rect: [f32; 4],
        h: f32,
    ) {
        let Some(t) = self.transition.clone() else {
            return;
        };
        let [x, y, rw, rh] = rect;
        let aspect = rw / rh;
        let mut to = self.current();
        to.aspect = aspect;
        let mut from = self.get(&t.from).cloned().unwrap_or_else(|| to.clone());
        from.aspect = aspect;
        let p = t.progress();
        match t.kind {
            TransitionKind::Cut | TransitionKind::Morph => target.draw(scene, &to),
            TransitionKind::Wipe(degrees) => {
                // The old picture whole, then the new one over the part of
                // the rectangle it has reached.
                target.draw(scene, &from);
                let a = degrees.to_radians();
                let (dx, dy) = (a.cos(), a.sin());
                let (nx, ny, nw, nh) = if dx.abs() >= dy.abs() {
                    let nw = rw * p;
                    (if dx >= 0.0 { x } else { x + rw - nw }, y, nw, rh)
                } else {
                    let nh = rh * p;
                    (x, if dy >= 0.0 { y } else { y + rh - nh }, rw, nh)
                };
                if nw >= 1.0
                    && nh >= 1.0
                    && let Some(r) = target.renderer()
                {
                    r.set_viewport(x, h - y - rh, rw, rh);
                    r.set_scissor(nx, h - ny - nh, nw, nh);
                    r.set_scissor_test(true);
                    target.draw(scene, &to);
                    // Back to the whole rectangle for whatever comes next.
                    if let Some(r) = target.renderer() {
                        r.set_scissor(x, h - y - rh, rw, rh);
                    }
                }
            }
            TransitionKind::Crossfade => {
                let Some(r) = target.renderer() else {
                    target.draw(scene, &to);
                    return;
                };
                // Both pictures into textures, then the two blended over the
                // rectangle (the targets keep light untone-mapped, so the
                // blend is tone-mapped once, as a picture is).
                let (tw, th) = (
                    (rw * r.pixel_ratio()).round().max(1.0) as u32,
                    (rh * r.pixel_ratio()).round().max(1.0) as u32,
                );
                let viewport = r.viewport();
                let scissor = (r.scissor(), r.scissor_test());
                r.reset_viewport();
                r.set_scissor_test(false);
                let fade = self.fade.get_or_insert_with(|| Fade::new(r, tw, th));
                fade.fit(r, tw, th);
                fade.a.render(r, scene, &from);
                fade.b.render(r, scene, &to);
                fade.set(p);
                if let Some(v) = viewport {
                    r.set_viewport(v.x, v.y, v.width, v.height);
                }
                if let Some(s) = scissor.0 {
                    r.set_scissor(s.x, s.y, s.width, s.height);
                }
                r.set_scissor_test(scissor.1);
                let Fade {
                    scene: fs, camera, ..
                } = fade;
                target.draw(fs, camera);
            }
        }
    }
}

/// `a` turned toward `b`, `t` of the way: place, aim, up and lens.
pub fn between(a: &PerspectiveCamera, b: &PerspectiveCamera, t: f32) -> PerspectiveCamera {
    let mut c = b.clone();
    c.position = a.position.lerp(b.position, t);
    c.target = a.target.lerp(b.target, t);
    let up = a.up.lerp(b.up, t);
    c.up = if up.length() > 1e-6 {
        up.normalize()
    } else {
        b.up
    };
    c.fov = a.fov + (b.fov - a.fov) * t;
    c.near = a.near + (b.near - a.near) * t;
    c.far = a.far + (b.far - a.far) * t;
    c.aspect = a.aspect + (b.aspect - a.aspect) * t;
    c
}

/// Ids handed to the renderer for the textures made here, counting down
/// from the top of the range so they do not meet the ids apps choose.
fn next_id() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(u32::MAX - 1);
    NEXT.fetch_sub(1, Ordering::Relaxed)
}

/// A camera's picture as a texture, for any material: render it each frame
/// with [`render`](Self::render), use [`texture`](Self::texture) as a map.
/// It keeps two pictures and draws into one while materials show the
/// other, so the camera may see its own texture (a monitor in the room it
/// films): it shows the picture before.
pub struct CameraTexture {
    id: u32,
    targets: [Arc<RenderTarget>; 2],
    texture: Arc<Texture>,
    /// Which target holds the newest picture.
    newest: usize,
}

impl CameraTexture {
    /// A `width` × `height` picture, made on `renderer`'s device.
    pub fn new(renderer: &mut Renderer, width: u32, height: u32) -> Self {
        Self::with_format(renderer, width, height, wgpu::TextureFormat::Rgba16Float)
    }

    /// In a format of its own (`Rgba16Float`, the default, keeps light past
    /// white for the tone mapping where it is shown; `Rgba8Unorm` is
    /// smaller).
    pub fn with_format(
        renderer: &mut Renderer,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let id = next_id();
        let device = renderer.device_arc();
        let targets = [
            Arc::new(RenderTarget::new(&device, width, height, format)),
            Arc::new(RenderTarget::new(&device, width, height, format)),
        ];
        renderer.register_render_target(id, &targets[0]);
        let tf = match format {
            wgpu::TextureFormat::Rgba16Float => TextureFormat::Rgba16Float,
            _ => TextureFormat::Rgba8Unorm,
        };
        let texture = Arc::new(Texture::from_render_target(id, width, height, tf));
        Self {
            id,
            targets,
            texture,
            newest: 0,
        }
    }

    /// The texture to put on a material (it shows the newest picture).
    pub fn texture(&self) -> Arc<Texture> {
        self.texture.clone()
    }

    /// The id it is registered with on the renderer (for a
    /// [`Texture::from_render_target`] of your own).
    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn size(&self) -> (u32, u32) {
        (self.targets[0].width, self.targets[0].height)
    }

    /// The target holding the newest picture (to read back, or to hand on).
    pub fn picture(&self) -> &Arc<RenderTarget> {
        &self.targets[self.newest]
    }

    /// Draws `scene` from `camera` into the picture not shown; then shows
    /// it.
    pub fn render(&mut self, renderer: &mut Renderer, scene: &mut Scene, camera: &dyn Camera) {
        let shown = self.newest;
        let drawn = 1 - shown;
        renderer.register_render_target(self.id, &self.targets[shown]);
        renderer.render_to(scene, camera, &self.targets[drawn]);
        self.newest = drawn;
        renderer.register_render_target(self.id, &self.targets[drawn]);
    }
}

/// A camera that sees its own picture, `levels` deep each frame: shown on
/// a screen (or a mirror, a portal) in the scene it looks at, which shows a
/// picture of the screen, which shows … Put [`texture`](Self::texture) on
/// the screen's material.
pub struct Feedback {
    picture: CameraTexture,
    /// How many times over a render draws it.
    pub levels: u32,
}

impl Feedback {
    pub fn new(renderer: &mut Renderer, width: u32, height: u32, levels: u32) -> Self {
        Self {
            picture: CameraTexture::new(renderer, width, height),
            levels: levels.max(1),
        }
    }

    /// The texture showing the newest picture.
    pub fn texture(&self) -> Arc<Texture> {
        self.picture.texture()
    }

    /// The target holding the newest picture.
    pub fn picture(&self) -> &Arc<RenderTarget> {
        self.picture.picture()
    }

    /// Draws `scene` from `camera` `levels` times, each with the picture
    /// before on the screen.
    pub fn render(&mut self, renderer: &mut Renderer, scene: &mut Scene, camera: &dyn Camera) {
        for _ in 0..self.levels {
            self.picture.render(renderer, scene, camera);
        }
    }
}

impl Fade {
    fn new(r: &mut Renderer, w: u32, h: u32) -> Self {
        let a = CameraTexture::new(r, w, h);
        let b = CameraTexture::new(r, w, h);
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        let mut top = None;
        for (i, t) in [&a, &b].into_iter().enumerate() {
            let mut m = BasicMaterial::new(Color::WHITE);
            m.map = Some(t.texture());
            m.transparent = i == 1;
            let mut quad = Object3D::mesh(Mesh::new(PlaneGeometry::new(2.0, 2.0), m.into()));
            quad.position = Vector3::new(0.0, 0.0, i as f32 * 0.01);
            top = Some(scene.add(quad));
        }
        let mut camera = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.0, 10.0);
        camera.position = Vector3::new(0.0, 0.0, 1.0);
        Self {
            a,
            b,
            scene,
            camera,
            top: top.expect("two quads"),
        }
    }

    /// New textures if the rectangle changed size.
    fn fit(&mut self, r: &mut Renderer, w: u32, h: u32) {
        if self.a.size() != (w, h) {
            *self = Fade::new(r, w, h);
        }
    }

    /// The new picture `p` of the way up through the old.
    fn set(&mut self, p: f32) {
        if let Some(o) = self.scene.get_mut(self.top)
            && let crate::core::ObjectKind::Mesh(m) = &mut o.kind
            && let crate::materials::Material::Basic(b) = Arc::make_mut(&mut m.material)
        {
            b.opacity = p;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_place_their_cameras() {
        let r = Layout::split_vertical("a", "b").rects(202.0, 100.0, "a");
        assert_eq!(r[0].1, [0.0, 0.0, 100.0, 100.0]);
        assert_eq!(r[1].1, [102.0, 0.0, 100.0, 100.0]);
        let g = Layout::grid(&["a", "b", "c"]).rects(102.0, 102.0, "a");
        assert_eq!(g.len(), 3);
        assert_eq!(g[2].1, [0.0, 52.0, 50.0, 50.0]);
        let p = Layout::picture_in_picture("a", "b").rects(400.0, 200.0, "a");
        assert_eq!(p[1].0, "b");
        assert!(
            (p[1].1[0] + p[1].1[2] - 388.0).abs() < 1e-3,
            "in from the right: {:?}",
            p[1].1
        );
        assert_eq!(Layout::Live.rects(10.0, 10.0, "x")[0].0, "x");
    }

    #[test]
    fn transitions_morph_and_end() {
        let mut set = CameraSet::new();
        let mut a = PerspectiveCamera::new(40.0, 1.0, 0.1, 10.0);
        a.position = Vector3::new(0.0, 0.0, 5.0);
        let mut b = PerspectiveCamera::new(80.0, 1.0, 0.1, 10.0);
        b.position = Vector3::new(4.0, 0.0, 0.0);
        set.add("a", a).add("b", b);
        assert_eq!(set.live(), "a");
        assert!(set.transition_eased("b", TransitionKind::Morph, 2.0, Easing::Linear));
        set.update(1.0);
        let mid = set.current();
        assert!((mid.position.x - 2.0).abs() < 1e-4 && (mid.fov.to_degrees() - 60.0).abs() < 1e-3);
        set.update(1.5);
        assert!(set.transition().is_none());
        assert_eq!(set.current().position.x, 4.0);
        assert!(!set.switch_to("nope"));
    }
}
