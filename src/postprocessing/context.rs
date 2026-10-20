//! What a pass is handed each frame: the renderer, the scenes and cameras,
//! where the screen is, and the operations it is built from.

use std::collections::HashSet;
use std::sync::Arc;

use super::target::{ensure, Target, INTERMEDIATE_FORMAT};
use crate::cameras::Camera;
use crate::core::{ObjectId, ObjectKind};
use crate::materials::Material;
use crate::math::{Color, Matrix4};
use crate::renderer::{
    PostFxBlend, PostFxCamera, PostFxNoise, PostFxOp, PostFxOutput, Renderer,
};
use crate::scene::Scene;

/// Where the composer's last pass draws — three.js's `renderer.setRenderTarget(null)`.
pub enum Screen<'a> {
    /// A registered target standing in for the screen: the WebGL2 back
    /// buffer, an offscreen render, a test's target.
    Target(Target),
    /// A view the caller owns, a swapchain image.
    View {
        view: &'a wgpu::TextureView,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    },
    /// Nowhere: `renderToScreen` is off and nothing should reach a display.
    None,
}

/// Where one operation writes.
#[derive(Clone, Copy)]
pub enum Dest<'t> {
    /// A target, as three.js renders into a `WebGLRenderTarget`: what is
    /// written is the value itself (linear, above 1 if it is).
    Target(&'t Target),
    /// The screen, as three.js renders with `setRenderTarget(null)`: clamped,
    /// and converted where three.js converts (a `MeshBasicMaterial` copy, a
    /// scene render).
    Screen,
    /// A half-float target standing in for the screen: written as the screen
    /// would be, so several draws can be blended into it the way three.js
    /// blends them onto an 8-bit canvas, and copied there once. See
    /// [`PassContext::screen_stand_in`].
    StandIn(&'t Target),
}

/// One post-fx draw, built up: `Fx::new(kind).input(&read).p2([..])`.
#[derive(Clone, Debug)]
pub struct Fx {
    pub op: PostFxOp,
}

impl Fx {
    pub fn new(kind: u32) -> Self {
        Self {
            op: PostFxOp::new(kind, 0),
        }
    }

    /// `CopyShader`: `opacity * texel`.
    pub fn copy(src: &Target, opacity: f32) -> Self {
        Self::new(kinds::COPY).input(src).p2([opacity, 0.0, 0.0, 0.0])
    }

    /// A `MeshBasicMaterial` copy: converted to sRGB onto the screen.
    pub fn basic_copy(src: &Target) -> Self {
        Self::new(kinds::BASIC_COPY).input(src).p2([1.0, 0.0, 0.0, 0.0])
    }

    pub fn input(mut self, t: &Target) -> Self {
        self.op.input = t.id();
        self
    }

    pub fn depth(mut self, t: &Target) -> Self {
        self.op.depth = t.id();
        self
    }

    pub fn normal(mut self, t: &Target) -> Self {
        self.op.normal = t.id();
        self
    }

    /// Bind `t`'s depth attachment as `hw_depth_tex`.
    pub fn hw_depth(mut self, t: &Target) -> Self {
        self.op.hw_depth = t.id();
        self
    }

    pub fn noise(mut self, n: PostFxNoise) -> Self {
        self.op.noise = n;
        self
    }

    pub fn time(mut self, t: f32) -> Self {
        self.op.time = t;
        self
    }

    pub fn p0z(mut self, v: f32) -> Self {
        self.op.p0z = v;
        self
    }

    pub fn p2(mut self, v: [f32; 4]) -> Self {
        self.op.p2 = v;
        self
    }

    pub fn p3(mut self, v: [f32; 4]) -> Self {
        self.op.p3 = v;
        self
    }

    pub fn blend(mut self, b: PostFxBlend) -> Self {
        self.op.blend = b;
        self
    }

    pub fn camera(mut self, c: PostFxCamera) -> Self {
        self.op.camera = c;
        self
    }

    pub fn linear_aux(mut self) -> Self {
        self.op.linear_aux = true;
        self
    }

    pub fn clear_color(mut self, c: [f64; 4]) -> Self {
        self.op.clear = c;
        self
    }
}

/// The `POSTFX_SHADER` kinds the passes use. 0–34 are the original set;
/// 40 up are the three.js passes' own shaders (`postfx_kinds.wgsl`).
pub mod kinds {
    pub const FXAA: u32 = 1;
    pub const FILM: u32 = 2;
    pub const DOT_SCREEN: u32 = 3;
    pub const HALFTONE: u32 = 4;
    pub const GLITCH: u32 = 5;
    pub const SSAO: u32 = 11;
    pub const SSAO_BLUR: u32 = 13;
    pub const HALVE: u32 = 20;
    pub const TONE_MAP: u32 = 25;
    pub const BOKEH: u32 = 26;
    pub const GLITCH_NO_SNOW: u32 = 31;
    pub const LUMINOSITY: u32 = 33;
    pub const ADAPT_LUMINANCE: u32 = 34;
    pub const COPY: u32 = 40;
    pub const BASIC_COPY: u32 = 41;
    pub const LUMINOSITY_HIGH_PASS: u32 = 42;
    pub const BLOOM_BLUR: u32 = 43;
    pub const BLOOM_COMPOSITE: u32 = 44;
    pub const CONVOLUTION: u32 = 45;
    pub const COMBINE: u32 = 46;
    pub const AFTERIMAGE: u32 = 47;
    pub const OUTLINE_MASK: u32 = 48;
    pub const OUTLINE_EDGE: u32 = 49;
    pub const OUTLINE_BLUR: u32 = 50;
    pub const OUTLINE_OVERLAY: u32 = 51;
    pub const DEPTH_VIEW: u32 = 52;
    pub const DEPTH_LIMITED_BLUR: u32 = 53;
    pub const SAO: u32 = 54;
    pub const SSR: u32 = 55;
    pub const SSR_BLUR: u32 = 56;
    pub const SMAA_EDGES: u32 = 58;
    pub const SMAA_WEIGHTS: u32 = 59;
    pub const SMAA_BLEND: u32 = 60;
    pub const PIXELATED: u32 = 61;
    pub const LUT: u32 = 62;
    pub const TRANSITION: u32 = 63;
    pub const OUTPUT: u32 = 64;
    pub const MASK_MIX: u32 = 65;
    pub const CUBE_BACKGROUND: u32 = 66;
}

/// Which objects a scene render draws.
#[derive(Clone, Debug, Default)]
pub enum Visibility {
    /// As the scene says.
    #[default]
    All,
    /// Lines and points hidden — SSAOPass's `overrideVisibility`, "honor only
    /// meshes".
    MeshesOnly,
    /// Only these meshes and sprites (with their descendants); every other
    /// mesh, sprite, line and point hidden — OutlinePass's selected pass.
    Only(Vec<ObjectId>),
    /// These meshes (with their descendants) hidden — OutlinePass's depth
    /// pass of everything else.
    Except(Vec<ObjectId>),
}

/// One scene render, three.js `renderer.render(scene, camera)` with the state
/// a pass sets around it.
#[derive(Clone, Debug, Default)]
pub struct SceneDraw {
    /// Index into the scenes the composer was given.
    pub scene: usize,
    /// Index into the cameras.
    pub camera: usize,
    /// three.js `scene.overrideMaterial`, for every mesh.
    pub override_material: Option<Arc<Material>>,
    /// A material for the listed meshes (and their descendants) and another
    /// for every other mesh — SSRPass's metalness pass.
    pub override_by_selection: Option<(Vec<ObjectId>, Arc<Material>, Arc<Material>)>,
    pub visibility: Visibility,
    /// Clear to this colour and alpha first, in place of the scene's
    /// background — `renderer.setClearColor(c, a); renderer.clear()` with
    /// `scene.background` null.
    pub clear: Option<(Color, f32)>,
    /// Keep what the target holds (`autoClear` off, no clear colour).
    pub keep: bool,
    /// Sub-pixel projection offset in pixels, three.js `camera.setViewOffset`
    /// by `(x, y)` (y down, as three.js has it).
    pub view_offset: Option<(f32, f32)>,
}

/// What [`Pass::render`](super::Pass::render) works with.
pub struct PassContext<'a, 's> {
    pub renderer: &'a mut Renderer,
    pub(crate) scenes: &'a mut [&'s mut Scene],
    pub(crate) cameras: &'a [&'a dyn Camera],
    pub(crate) screen: &'a Screen<'a>,
    /// The composer's size in device pixels: `setSize × pixelRatio`.
    pub width: u32,
    pub height: u32,
    pub(crate) stand_in: &'a mut Option<Target>,
    pub(crate) scratch: &'a mut Option<Target>,
}

impl<'a, 's> PassContext<'a, 's> {
    /// Run `fx` into `dest`.
    pub fn fx(&mut self, fx: Fx, dest: Dest<'_>) {
        let mut op = fx.op;
        match dest {
            Dest::Target(t) => {
                op.display = false;
                self.renderer.run_postfx(&op, PostFxOutput::Target(t.id()));
            }
            Dest::StandIn(t) => {
                op.display = true;
                self.renderer.run_postfx(&op, PostFxOutput::Target(t.id()));
            }
            Dest::Screen => {
                op.display = true;
                match self.screen {
                    Screen::Target(t) => {
                        self.renderer.run_postfx(&op, PostFxOutput::Target(t.id()))
                    }
                    Screen::View {
                        view,
                        format,
                        width,
                        height,
                    } => self.renderer.run_postfx(
                        &op,
                        PostFxOutput::View {
                            view,
                            format: *format,
                            width: *width,
                            height: *height,
                        },
                    ),
                    Screen::None => {}
                }
            }
        }
    }

    /// Clear `dest` — three.js `renderer.clear()` with the clear colour set to
    /// `color` (linear, as three.js `Color` is) and `alpha`.
    pub fn clear(&mut self, dest: Dest<'_>, color: Color, alpha: f32) {
        let c = [color.r as f64, color.g as f64, color.b as f64, alpha as f64];
        match dest {
            Dest::Target(t) => {
                self.renderer
                    .clear_postfx_output(PostFxOutput::Target(t.id()), c, false)
            }
            Dest::StandIn(t) => {
                self.renderer
                    .clear_postfx_output(PostFxOutput::Target(t.id()), c, true)
            }
            Dest::Screen => match self.screen {
                Screen::Target(t) => {
                    self.renderer
                        .clear_postfx_output(PostFxOutput::Target(t.id()), c, true)
                }
                Screen::View {
                    view,
                    format,
                    width,
                    height,
                } => self.renderer.clear_postfx_output(
                    PostFxOutput::View {
                        view,
                        format: *format,
                        width: *width,
                        height: *height,
                    },
                    c,
                    true,
                ),
                Screen::None => {}
            },
        }
    }

    /// A half-float target the size of the screen, for a pass that blends
    /// several draws onto the screen: drawn with [`Dest::StandIn`] and copied
    /// over with [`present_stand_in`](Self::present_stand_in). three.js blends
    /// those draws in the canvas's own 8-bit values, which a WebGL2 sRGB
    /// canvas would instead blend in linear light; blending them here, as
    /// display values, and copying the result once keeps both backends — and
    /// three.js — to the same bytes.
    pub fn screen_stand_in(&mut self) -> Option<Target> {
        let (w, h) = self.screen_size()?;
        Some(ensure(self.stand_in, self.renderer, w, h, INTERMEDIATE_FORMAT))
    }

    /// Copy a stand-in onto the screen as it stands.
    pub fn present_stand_in(&mut self, t: &Target) {
        self.fx(Fx::copy(t, 1.0), Dest::Screen);
    }

    /// The screen's size, if there is one.
    pub fn screen_size(&self) -> Option<(u32, u32)> {
        match self.screen {
            Screen::Target(t) => Some(t.size()),
            Screen::View { width, height, .. } => Some((*width, *height)),
            Screen::None => None,
        }
    }

    /// Whether there is a screen to draw to.
    pub fn has_screen(&self) -> bool {
        !matches!(self.screen, Screen::None)
    }

    /// The target in `slot`, made `width × height` half-float.
    pub fn target(&mut self, slot: &mut Option<Target>, width: u32, height: u32) -> Target {
        ensure(slot, self.renderer, width, height, INTERMEDIATE_FORMAT)
    }

    /// Camera `index`, or `None` if the composer was given fewer.
    pub fn camera(&self, index: usize) -> Option<&dyn Camera> {
        self.cameras.get(index).copied()
    }

    /// Scene `index`, mutably.
    pub fn scene_mut(&mut self, index: usize) -> Option<&mut Scene> {
        self.scenes.get_mut(index).map(|s| &mut **s)
    }

    /// The uniforms a depth-reading kind needs from camera `index`: near, far,
    /// the projection and its inverse.
    pub fn postfx_camera(&self, index: usize) -> PostFxCamera {
        let Some(camera) = self.camera(index) else {
            return PostFxCamera::default();
        };
        let proj = camera.projection_matrix();
        let (near, far) = camera.near_far();
        PostFxCamera {
            near,
            far,
            kernel_radius: 8.0,
            kernel_size: 32,
            proj: proj.elements,
            inv_proj: proj.invert().elements,
        }
    }

    /// Whether camera `index` is a perspective one — its projection divides
    /// by depth (`elements[11]` is -1), three.js `isPerspectiveCamera`.
    pub fn camera_is_perspective(&self, index: usize) -> bool {
        self.camera(index)
            .map(|c| c.projection_matrix().elements[11].abs() > 0.5)
            .unwrap_or(true)
    }

    /// Camera `index`'s world matrix.
    pub fn camera_world(&self, index: usize) -> Matrix4 {
        self.camera(index)
            .map(|c| c.view_matrix().invert())
            .unwrap_or_else(Matrix4::identity)
    }

    /// Render a scene into `dest`, with the state `draw` describes set around
    /// it and put back after.
    pub fn render_scene(&mut self, dest: Dest<'_>, draw: &SceneDraw) {
        let Some(&camera) = self.cameras.get(draw.camera) else {
            return;
        };
        let Some(scene) = self.scenes.get_mut(draw.scene) else {
            return;
        };
        let scene: &mut Scene = scene;

        // Visibility and materials, saved to be put back.
        let mut saved: Vec<(ObjectId, bool, Option<Arc<Material>>)> = Vec::new();
        let expand = |scene: &Scene, ids: &[ObjectId]| -> HashSet<ObjectId> {
            let mut set = HashSet::new();
            for &id in ids {
                if scene.arena.get(id).is_some() {
                    scene.arena.traverse(id, &mut |child, _| {
                        set.insert(child);
                    });
                }
            }
            set
        };
        let only = match &draw.visibility {
            Visibility::Only(ids) => Some(expand(scene, ids)),
            _ => None,
        };
        let except = match &draw.visibility {
            Visibility::Except(ids) => Some(expand(scene, ids)),
            _ => None,
        };
        let selection = draw
            .override_by_selection
            .as_ref()
            .map(|(ids, on, off)| (expand(scene, ids), on.clone(), off.clone()));
        for (id, obj) in scene.arena.nodes.iter_mut() {
            let is_mesh = matches!(
                obj.kind,
                ObjectKind::Mesh(_) | ObjectKind::SkinnedMesh(_) | ObjectKind::InstancedMesh(_)
            );
            let is_sprite = matches!(obj.kind, ObjectKind::Sprite(_));
            let is_line_or_points =
                matches!(obj.kind, ObjectKind::LineSegments(_) | ObjectKind::Points(_));
            let mut visible = obj.visible;
            match &draw.visibility {
                Visibility::All => {}
                Visibility::MeshesOnly => {
                    if is_line_or_points {
                        visible = false;
                    }
                }
                Visibility::Only(_) => {
                    if is_mesh || is_sprite {
                        visible = visible && only.as_ref().is_some_and(|s| s.contains(&id));
                    } else if is_line_or_points {
                        visible = false;
                    }
                }
                Visibility::Except(_) => {
                    if is_mesh && except.as_ref().is_some_and(|s| s.contains(&id)) {
                        visible = false;
                    }
                }
            }
            let chosen = match &selection {
                Some((set, on, off)) => Some(if set.contains(&id) { on.clone() } else { off.clone() }),
                None => draw.override_material.clone(),
            };
            let material = match (&chosen, &mut obj.kind) {
                (Some(m), ObjectKind::Mesh(mesh)) => Some(std::mem::replace(&mut mesh.material, m.clone())),
                (Some(m), ObjectKind::SkinnedMesh(mesh)) => {
                    Some(std::mem::replace(&mut mesh.material, m.clone()))
                }
                (Some(m), ObjectKind::InstancedMesh(mesh)) => {
                    Some(std::mem::replace(&mut mesh.material, m.clone()))
                }
                _ => None,
            };
            if visible != obj.visible || material.is_some() {
                saved.push((id, obj.visible, material));
                obj.visible = visible;
            }
        }

        let background = (scene.background, scene.background_alpha);
        if let Some((c, a)) = draw.clear {
            scene.background = c;
            scene.background_alpha = a;
        }
        let auto_clear = self.renderer.auto_clear();
        self.renderer.set_auto_clear(!draw.keep);
        if let Some((x, y)) = draw.view_offset {
            // three.js `setViewOffset` moves the window right and down; the
            // renderer's jitter is in projection terms, y up.
            self.renderer.set_jitter_override(Some((x, -y)));
        }

        let mut stand_in_copy = None;
        match dest {
            Dest::Target(t) => {
                self.renderer.render_to(scene, camera, t.render_target());
            }
            Dest::StandIn(t) => {
                // A stand-in holds what the screen would: the render goes
                // through a target of its own and is converted on the copy.
                let (w, h) = t.size();
                let tmp = ensure(self.scratch, self.renderer, w, h, INTERMEDIATE_FORMAT);
                self.renderer.render_to(scene, camera, tmp.render_target());
                stand_in_copy = Some((tmp, t.clone()));
            }
            Dest::Screen => match self.screen {
                Screen::Target(t) => self.renderer.render_to(scene, camera, t.render_target()),
                Screen::View { view, .. } => self.renderer.render(scene, camera, view, false),
                Screen::None => {}
            },
        }

        if draw.view_offset.is_some() {
            self.renderer.set_jitter_override(None);
        }
        self.renderer.set_auto_clear(auto_clear);
        if let Some((tmp, t)) = stand_in_copy {
            let blend = if draw.keep {
                PostFxBlend::Normal
            } else {
                PostFxBlend::Replace
            };
            self.fx(Fx::basic_copy(&tmp).blend(blend), Dest::StandIn(&t));
        }
        let Some(scene) = self.scenes.get_mut(draw.scene) else {
            return;
        };
        let scene: &mut Scene = scene;
        scene.background = background.0;
        scene.background_alpha = background.1;
        for (id, visible, material) in saved {
            if let Some(obj) = scene.arena.get_mut(id) {
                obj.visible = visible;
                if let Some(m) = material {
                    match &mut obj.kind {
                        ObjectKind::Mesh(mesh) => mesh.material = m,
                        ObjectKind::SkinnedMesh(mesh) => mesh.material = m,
                        ObjectKind::InstancedMesh(mesh) => mesh.material = m,
                        _ => {}
                    }
                }
            }
        }
    }
}
