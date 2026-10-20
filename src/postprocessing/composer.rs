//! three.js's `EffectComposer`: a chain of passes over two ping-pong buffers.

use super::context::{kinds, Dest, Fx, PassContext, Screen};
use super::pass::{MaskRole, Pass};
use super::target::{ensure, Target, INTERMEDIATE_FORMAT};
use crate::cameras::Camera;
use crate::renderer::{RenderTarget, Renderer};
use crate::scene::Scene;

/// three.js `EffectComposer`.
///
/// Two half-float buffers, `writeBuffer` and `readBuffer`; each enabled pass
/// reads one and writes the other (or draws into `readBuffer` in place), the
/// two swap after any pass that [`needs_swap`](super::PassState::needs_swap),
/// and the last enabled pass draws to the screen when
/// [`render_to_screen`](Self::render_to_screen) is on. `MaskPass` /
/// `ClearMaskPass` confine the passes between them to the mask, as three.js's
/// stencil test does.
///
/// Scenes and cameras are not stored: passes name them by index, and each
/// frame's [`render_frame`](Self::render_frame) is handed the lists those
/// indices point into. [`render`](Self::render) is the one-scene case.
///
/// ```no_run
/// # use threers::postprocessing::{EffectComposer, RenderPass, UnrealBloomPass, OutputPass};
/// # fn demo(r: &mut threers::Renderer, scene: &mut threers::Scene,
/// #         camera: &threers::PerspectiveCamera, view: &threers::wgpu::TextureView) {
/// let mut composer = EffectComposer::with_size(800, 600);
/// composer.add_pass(RenderPass::new());
/// composer.add_pass(UnrealBloomPass::new(800, 600, 1.5, 0.4, 0.85));
/// composer.add_pass(OutputPass::new());
/// composer.set_screen_format(threers::wgpu::TextureFormat::Bgra8Unorm);
/// composer.render(r, scene, camera, view);
/// # }
/// ```
pub struct EffectComposer {
    passes: Vec<Box<dyn Pass>>,
    /// three.js `renderToScreen`: the last enabled pass draws to the screen.
    pub render_to_screen: bool,
    width: u32,
    height: u32,
    pixel_ratio: f32,
    format: wgpu::TextureFormat,
    rt1: Option<Target>,
    rt2: Option<Target>,
    /// A target the caller supplied as `renderTarget1`; `rt2` follows it.
    user_target: bool,
    /// three.js starts with `writeBuffer = renderTarget1`.
    write_is_rt1: bool,
    clock: crate::core::Clock,
    stand_in: Option<Target>,
    scratch: Option<Target>,
    mask_before: Option<Target>,
    mask_scratch: Option<Target>,
    /// The format of the views [`render`](Self::render) is handed.
    screen_format: wgpu::TextureFormat,
}

impl std::fmt::Debug for EffectComposer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EffectComposer")
            .field("passes", &self.pass_names())
            .field("size", &(self.width, self.height))
            .field("pixel_ratio", &self.pixel_ratio)
            .field("render_to_screen", &self.render_to_screen)
            .finish()
    }
}

impl EffectComposer {
    /// A composer for a `width × height` drawing buffer (three.js takes the
    /// renderer's size). Its buffers are allocated on the first render.
    pub fn with_size(width: u32, height: u32) -> Self {
        Self {
            passes: Vec::new(),
            render_to_screen: true,
            width: width.max(1),
            height: height.max(1),
            pixel_ratio: 1.0,
            format: INTERMEDIATE_FORMAT,
            rt1: None,
            rt2: None,
            user_target: false,
            write_is_rt1: true,
            clock: crate::core::Clock::new(true),
            stand_in: None,
            scratch: None,
            mask_before: None,
            mask_scratch: None,
            screen_format: wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    /// [`with_size`](Self::with_size), with `format` the format of the views
    /// [`render`](Self::render) will draw the last pass into. The device and
    /// queue are the renderer's, which allocates the buffers.
    pub fn new(
        _device: std::sync::Arc<wgpu::Device>,
        _queue: std::sync::Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let mut c = Self::with_size(width, height);
        c.screen_format = format;
        c
    }

    /// three.js `new EffectComposer(renderer, renderTarget)`: chain through
    /// `target` (and a second target like it) rather than half-float buffers
    /// of the composer's own.
    pub fn with_target(target: Target) -> Self {
        let mut c = Self::with_size(target.width(), target.height());
        c.format = target.format();
        c.rt1 = Some(target);
        c.user_target = true;
        c
    }

    /// The format of the views [`render`](Self::render) draws into.
    pub fn set_screen_format(&mut self, format: wgpu::TextureFormat) {
        self.screen_format = format;
    }

    fn effective(&self) -> (u32, u32) {
        (
            ((self.width as f32) * self.pixel_ratio).round().max(1.0) as u32,
            ((self.height as f32) * self.pixel_ratio).round().max(1.0) as u32,
        )
    }

    /// three.js `addPass`: append, and size the pass to the composer.
    pub fn add_pass(&mut self, pass: impl Pass + 'static) -> &mut Self {
        self.add_boxed(Box::new(pass))
    }

    /// [`add_pass`](Self::add_pass) for a pass already boxed.
    pub fn add_boxed(&mut self, mut pass: Box<dyn Pass>) -> &mut Self {
        let (w, h) = self.effective();
        pass.set_size(w, h);
        self.passes.push(pass);
        self
    }

    /// three.js `insertPass`.
    pub fn insert_boxed(&mut self, mut pass: Box<dyn Pass>, index: usize) {
        let (w, h) = self.effective();
        pass.set_size(w, h);
        let index = index.min(self.passes.len());
        self.passes.insert(index, pass);
    }

    /// three.js `removePass`, by position.
    pub fn remove_pass(&mut self, index: usize) -> Option<Box<dyn Pass>> {
        (index < self.passes.len()).then(|| self.passes.remove(index))
    }

    pub fn passes(&self) -> &[Box<dyn Pass>] {
        &self.passes
    }

    pub fn passes_mut(&mut self) -> &mut [Box<dyn Pass>] {
        &mut self.passes
    }

    pub fn pass_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|p| p.name()).collect()
    }

    /// three.js `isLastEnabledPass`.
    pub fn is_last_enabled_pass(&self, index: usize) -> bool {
        !self.passes[index + 1..].iter().any(|p| p.state().enabled)
    }

    /// three.js `setSize`: the drawing buffer in logical pixels.
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        let (w, h) = self.effective();
        for p in &mut self.passes {
            p.set_size(w, h);
        }
    }

    /// three.js `setPixelRatio`.
    pub fn set_pixel_ratio(&mut self, ratio: f32) {
        self.pixel_ratio = if ratio > 0.0 { ratio } else { 1.0 };
        self.set_size(self.width, self.height);
    }

    pub fn pixel_ratio(&self) -> f32 {
        self.pixel_ratio
    }

    /// Logical size, as [`set_size`](Self::set_size) set it.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The buffers' size in device pixels.
    pub fn drawing_size(&self) -> (u32, u32) {
        self.effective()
    }

    /// three.js `reset`: new buffers (`target` as the first, if given) and
    /// `writeBuffer` back to the first.
    pub fn reset(&mut self, target: Option<Target>) {
        self.user_target = target.is_some();
        if let Some(t) = &target {
            self.width = t.width();
            self.height = t.height();
            self.format = t.format();
        }
        self.rt1 = target;
        self.rt2 = None;
        self.write_is_rt1 = true;
    }

    /// three.js `swapBuffers`.
    pub fn swap_buffers(&mut self) {
        self.write_is_rt1 = !self.write_is_rt1;
    }

    /// `renderTarget1`, once allocated.
    pub fn render_target1(&self) -> Option<&Target> {
        self.rt1.as_ref()
    }

    pub fn render_target2(&self) -> Option<&Target> {
        self.rt2.as_ref()
    }

    /// three.js `writeBuffer`, once allocated.
    pub fn write_buffer(&self) -> Option<&Target> {
        if self.write_is_rt1 {
            self.rt1.as_ref()
        } else {
            self.rt2.as_ref()
        }
    }

    /// three.js `readBuffer`, once allocated.
    pub fn read_buffer(&self) -> Option<&Target> {
        if self.write_is_rt1 {
            self.rt2.as_ref()
        } else {
            self.rt1.as_ref()
        }
    }

    /// Allocate (or resize) the two buffers now rather than on the first render.
    pub fn ensure_buffers(&mut self, renderer: &mut Renderer) {
        let (w, h) = self.effective();
        if self.user_target {
            if let Some(t) = &self.rt1 {
                let (tw, th, tf) = (t.width(), t.height(), t.format());
                ensure(&mut self.rt2, renderer, tw, th, tf);
                return;
            }
        }
        ensure(&mut self.rt1, renderer, w, h, self.format);
        ensure(&mut self.rt2, renderer, w, h, self.format);
    }

    /// three.js `render(deltaTime)` for one scene and camera, drawing the last
    /// pass into `view` (of the format given to [`new`](Self::new) or
    /// [`set_screen_format`](Self::set_screen_format), the composer's drawing
    /// size). Passes' scene and camera indices are all 0 here.
    pub fn render(
        &mut self,
        renderer: &mut Renderer,
        scene: &mut Scene,
        camera: &dyn Camera,
        view: &wgpu::TextureView,
    ) {
        let (width, height) = self.effective();
        let screen = Screen::View {
            view,
            format: self.screen_format,
            width,
            height,
        };
        self.render_frame(renderer, &mut [scene], &[camera], &screen, None);
    }

    /// [`render`](Self::render) into a render target instead of a view.
    pub fn render_to_target(
        &mut self,
        renderer: &mut Renderer,
        scene: &mut Scene,
        camera: &dyn Camera,
        target: Target,
    ) {
        let screen = Screen::Target(target);
        self.render_frame(renderer, &mut [scene], &[camera], &screen, None);
    }

    /// three.js `render(deltaTime)`. `scenes` and `cameras` are what the
    /// passes' indices point into; `screen` is where the last pass draws;
    /// `delta` is seconds since the last frame, taken from the composer's
    /// clock when `None`.
    pub fn render_frame(
        &mut self,
        renderer: &mut Renderer,
        scenes: &mut [&mut Scene],
        cameras: &[&dyn Camera],
        screen: &Screen<'_>,
        delta: Option<f32>,
    ) {
        let delta = delta.unwrap_or_else(|| self.clock.get_delta() as f32);
        self.ensure_buffers(renderer);
        let (width, height) = self.effective();
        let mut mask: Option<(Target, bool)> = None;
        let count = self.passes.len();
        for i in 0..count {
            if !self.passes[i].state().enabled {
                continue;
            }
            let to_screen = self.render_to_screen && self.is_last_enabled_pass(i);
            let mut state = self.passes[i].state();
            state.render_to_screen = to_screen;
            self.passes[i].set_state(state);
            let (Some(write), Some(read)) = (
                self.write_buffer().cloned(),
                self.read_buffer().cloned(),
            ) else {
                return;
            };
            // MaskPass / ClearMaskPass are not themselves masked; what they
            // set takes effect after they render (MaskPass's target exists
            // only then).
            let is_mask_pass = !matches!(self.passes[i].mask_role(), MaskRole::None)
                || matches!(self.passes[i].name(), "MaskPass" | "ClearMaskPass");
            let masked = mask.is_some() && !to_screen && !is_mask_pass;
            let needs_swap;
            {
                let Self {
                    passes,
                    stand_in,
                    scratch,
                    mask_before,
                    mask_scratch,
                    ..
                } = self;
                let mut ctx = PassContext {
                    renderer: &mut *renderer,
                    scenes: &mut *scenes,
                    cameras,
                    screen,
                    width,
                    height,
                    stand_in,
                    scratch,
                };
                // The stencil test, after the fact: keep what the buffer the
                // pass draws into held outside the mask.
                let before = masked.then(|| {
                    let t = ctx.target(mask_before, read.width(), read.height());
                    ctx.fx(Fx::copy(&read, 1.0), Dest::Target(&t));
                    t
                });
                let pass = &mut passes[i];
                pass.render(&mut ctx, &write, &read, delta, mask.is_some());
                needs_swap = pass.state().needs_swap;
                if let (Some(before), Some((mask_target, inverse))) = (before, &mask) {
                    let dest = if needs_swap { &write } else { &read };
                    let tmp = ctx.target(mask_scratch, dest.width(), dest.height());
                    ctx.fx(Fx::copy(dest, 1.0), Dest::Target(&tmp));
                    ctx.fx(
                        Fx::new(kinds::MASK_MIX)
                            .input(&tmp)
                            .depth(&before)
                            .hw_depth(mask_target)
                            .p2([if *inverse { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0]),
                        Dest::Target(dest),
                    );
                }
            }
            if needs_swap {
                self.swap_buffers();
            }
            match self.passes[i].mask_role() {
                MaskRole::Set { target, inverse } => mask = Some((target, inverse)),
                MaskRole::Clear => mask = None,
                MaskRole::None => {}
            }
        }
    }

    /// The render target behind `readBuffer`, for a caller that wants the
    /// chain's result as a texture.
    pub fn read_render_target(&self) -> Option<&std::sync::Arc<RenderTarget>> {
        self.read_buffer().map(|t| t.render_target())
    }
}
