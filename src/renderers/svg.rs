//! Vector SVG export — see [`SvgRenderer`] for the user-facing story.
//!
//! Three things in here decide whether the output is right, and each exists
//! because the obvious version of it produces a visibly wrong picture. Their
//! reasoning lives with the code:
//!
//! - [`Item::depth`] — faces sort by how far back they *reach*, not by their
//!   centroid. The centroid is the textbook answer and it slices objects off at
//!   the floor line.
//! - [`subdivide_by_depth`] — one depth cannot order a polygon that spans other
//!   geometry, so those get cut up until it can.
//! - [`curved_outline`] — edges bend onto the surface the tessellation stands
//!   for, so a sphere's outline is a circle rather than a 24-gon.
//!
//! None of it is asserted. `tests/svg_render.rs` renders a set of scenes, casts
//! a ray per sampled pixel to find what is actually in front of the camera, and
//! compares that against what the document paints.

use crate::cameras::Camera;
use crate::core::{BufferGeometry, ObjectKind};
use crate::curves::{segments_to_path_data, PathSegment};
use crate::lights::Light;
use crate::materials::Material;
use crate::math::{Color, Matrix4, Vector2, Vector3};
use crate::renderer::ToneMapping;
use crate::scene::Scene;
use crate::utils::format_svg_number as fmt;
use std::sync::Arc;

/// The GPU renderer's per-frame uniform holds four lights of each kind. The SVG
/// path has no such constraint, but honouring it keeps the two outputs of the
/// same scene in agreement — a fifth light that lit the SVG and not the PNG
/// would be a worse bug than a fifth light that lit neither.
const MAX_LIGHTS_PER_KIND: usize = 4;

/// Vertices on or behind the eye make the perspective divide meaningless, so
/// faces are clipped to `w >= NEAR_W`. Under an orthographic camera `w` is
/// always 1 and this never fires.
const NEAR_W: f32 = 1e-4;

/// How faces are coloured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvgShading {
    /// One Lambert/Phong evaluation per face, using the scene's lights — the
    /// closest match to what [`Renderer`](crate::Renderer) draws.
    #[default]
    Lit,
    /// The material's base colour, unlit. Gives the flat-vector look, and is
    /// what a scene with no lights in it gets regardless of this setting.
    Flat,
    /// Triangle edges only, no fills.
    Wireframe,
}

/// Knobs for [`SvgRenderer`].
#[derive(Debug, Clone, Copy)]
pub struct SvgOptions {
    /// How faces are coloured — lit, flat, or edges only.
    pub shading: SvgShading,
    /// Tone-mapping curve applied to the shaded linear colour. Match this to
    /// [`Renderer::set_tone_mapping`](crate::Renderer::set_tone_mapping) to get
    /// the same colours out of both renderers.
    pub tone_mapping: ToneMapping,
    /// Exposure multiplier, applied before the tone-mapping curve. Ignored by
    /// [`ToneMapping::None`], which is what that mode means.
    pub exposure: f32,
    /// Drop faces pointing away from the camera, honouring the material's
    /// `side`. Worth turning off only for a mesh that is not closed and whose
    /// back faces are meant to show.
    pub cull_backfaces: bool,
    /// Emit a full-canvas `<rect>` behind the scene. With this off the document
    /// is transparent, which is usually what you want when the SVG is going to
    /// sit on a page that has its own background.
    pub background: bool,
    /// Colour for that rect, overriding `scene.background`. `None` takes the
    /// scene's own.
    ///
    /// An override rather than "just set `scene.background`" because the same
    /// scene is often drawn to more than one place — a dark canvas and a white
    /// page — and the background is a property of where it is going, not of
    /// what is in it.
    pub clear_color: Option<Color>,
    /// Opacity for that rect, overriding `scene.background_alpha`. `None` takes
    /// the scene's own.
    pub clear_alpha: Option<f32>,
    /// Hairline stroke, in pixels, drawn on each face in its own fill colour.
    ///
    /// Neighbouring polygons that share an edge do not composite to full
    /// coverage along it — each covers about half the boundary pixel, so the
    /// background shows through as a light seam over every shared edge. This is
    /// the standard fix: grow each face by half a hairline so the pairs
    /// overlap. 0 disables it.
    pub seam_stroke: f32,
    /// Decimal places kept on coordinates. Two is below what a screen can
    /// resolve and keeps documents roughly a third the size of full `f32`.
    pub precision: usize,
    /// Depth-sort faces back to front. Off is only useful when the caller has
    /// already ordered the scene and wants the traversal order preserved.
    pub sort: bool,
    /// Draw a face's edges as Bézier curves when a straight one would miss the
    /// real surface by more than this many pixels. `None` keeps every edge
    /// straight.
    ///
    /// A tessellated sphere has a polygonal outline, and no amount of shading
    /// hides that its silhouette is a 24-gon — which is a strange thing to ship
    /// in a format whose whole point is that it has curves in it. Where the
    /// geometry carries vertex normals, the surface between two vertices is
    /// known, so each edge can be bent back onto it.
    ///
    /// Only edges that need it are curved. A cube's vertex normals are constant
    /// across each face, so its edges stay straight and its document stays
    /// small; a sphere's are not, so its silhouette comes out round.
    ///
    /// Measured against an analytic sphere, this puts the outline *exactly* on
    /// the silhouette from about sixteen segments upwards, where straight edges
    /// still visibly under-fill it. Below that it closes most of the gap and
    /// not all: the outline's corners are mesh vertices, and on a mesh that
    /// coarse those sit inside the true silhouette to begin with. Bending the
    /// edges between them cannot move the corners, so an eight-segment sphere
    /// comes out round but a couple of pixels small. Tessellate it rather than
    /// tightening this — the limit is where the outline is allowed to be, not
    /// how accurately it is drawn.
    pub curve_tolerance: Option<f32>,
    /// Split faces that span more than this fraction of their own distance from
    /// the camera, so that one depth per face means something. `None` disables
    /// splitting.
    ///
    /// One number cannot order a polygon that spans other geometry, and this is
    /// the way out: cut it up until each piece is shallow enough that it does
    /// not span anything. What is left over after the depth key does its job is
    /// a large polygon that occludes something while reaching past it — a
    /// receding wall in front of a small object — and splitting is what fixes
    /// those.
    ///
    /// Lower is more correct and bigger. The default is where the curve bends:
    /// across ten test scenes it is exact on nine, and it leaves tessellated
    /// geometry completely alone — a torus knot emits the same number of paths
    /// as with splitting off, because none of its faces is deep enough to trip
    /// the test. Only the big flat things get cut, which is the point.
    /// Tightening to `0.02` fixes the last scene and roughly doubles the
    /// document, because at that threshold ordinary tessellation starts
    /// splitting too.
    pub depth_split: Option<f32>,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            shading: SvgShading::default(),
            tone_mapping: ToneMapping::None,
            exposure: 1.0,
            cull_backfaces: true,
            background: true,
            clear_color: None,
            clear_alpha: None,
            seam_stroke: 0.8,
            precision: 2,
            sort: true,
            curve_tolerance: Some(0.15),
            depth_split: Some(0.05),
        }
    }
}

/// Renders a [`Scene`] to an SVG document: every triangle becomes a `<path>`,
/// depth-sorted back to front and filled with the colour the scene's lights
/// give it. Mirrors three.js's `SVGRenderer`, which is the same idea — a
/// painter's-algorithm rasteriser whose "pixels" are polygons.
///
/// It needs no GPU, no window and no adapter, which is what makes it work in
/// CI, over SSH, and in a wasm build with no WebGPU.
///
/// ```no_run
/// use threers::{PerspectiveCamera, Scene, SvgRenderer};
///
/// # fn demo(scene: &mut Scene, camera: &PerspectiveCamera) -> std::io::Result<()> {
/// SvgRenderer::new(800, 600).render_to_file(scene, camera, "out/frame.svg")
/// # }
/// ```
///
/// # Why vectors
///
/// The output stays resolution-independent and editable: it drops into a paper,
/// a slide or Illustrator, and scales to a plotter without going soft. Curved
/// surfaces come out as real curves rather than as the polygons they are stored
/// as — see [`SvgOptions::curve_tolerance`].
///
/// # Colour
///
/// Shading tracks the wgpu [`Renderer`](crate::Renderer) rather than inventing
/// its own look: the same Lambert/Phong accumulation, the same punctual
/// attenuation, the same tone-mapping curves, and the same linear→sRGB encode
/// on the way out. Set [`SvgOptions::tone_mapping`] to whatever the wgpu
/// renderer is using and the two agree.
///
/// # What it cannot draw
///
/// Every difference from the wgpu renderer comes from the same place — one
/// value per face instead of one per pixel. Faces are flat-shaded, so there are
/// no textures, no shadow maps and no post-processing, and a [`Sprite`] is
/// skipped for the same reason: a textured billboard is the one thing this has
/// no way to express. `Points` become `<circle>`, `LineSegments` become
/// `<line>`, and an `InstancedMesh` expands to one sorted draw per instance.
///
/// Sorting whole faces also cannot resolve geometry that *interpenetrates* —
/// two cubes pushed through each other show the seam where they cross. Only a
/// z-buffer fixes that, and a z-buffer is a raster image: use [`svg_from_rgba`]
/// when the scene needs one.
///
/// [`Sprite`]: crate::Sprite
pub struct SvgRenderer {
    /// Canvas width in pixels. Becomes the document's `width` and `viewBox`.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Everything else, with reasonable defaults — see [`SvgOptions`].
    pub options: SvgOptions,
}

impl SvgRenderer {
    /// A renderer for a canvas of this size, with [`SvgOptions::default`].
    ///
    /// The camera's aspect ratio is the caller's: nothing here changes it, so a
    /// camera built for a different one draws a stretched picture.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            options: SvgOptions::default(),
        }
    }

    /// Builder form: `SvgRenderer::new(w, h).with_options(opts)`.
    pub fn with_options(mut self, options: SvgOptions) -> Self {
        self.options = options;
        self
    }

    /// Resize the canvas, keeping the options. Mirrors three.js's
    /// `SVGRenderer.setSize`.
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Background colour for the document, overriding `scene.background`.
    /// Mirrors three.js's `setClearColor`; pass `None` to go back to the
    /// scene's own.
    pub fn set_clear_color(&mut self, color: Option<Color>, alpha: Option<f32>) {
        self.options.clear_color = color;
        self.options.clear_alpha = alpha;
    }

    /// Match the tone mapping of a wgpu [`Renderer`](crate::Renderer) so the
    /// two agree on colour. Pair with
    /// [`Renderer::tone_mapping`](crate::Renderer::tone_mapping).
    pub fn set_tone_mapping(&mut self, mode: ToneMapping, exposure: f32) {
        self.options.tone_mapping = mode;
        self.options.exposure = exposure;
    }

    /// The SVG document, as markup.
    pub fn render_to_string(&self, scene: &mut Scene, camera: &dyn Camera) -> String {
        scene.update_world();
        let opt = &self.options;
        let view = camera.view_matrix();
        let vp = camera.projection_matrix().multiply(&view);
        let eye = camera.position();
        let cam_layers = camera.layers();
        let half_w = self.width as f32 * 0.5;
        let half_h = self.height as f32 * 0.5;

        let lighting = Lighting::collect(scene);
        let draws = collect_draws(scene, cam_layers);

        let mut items: Vec<Item> = Vec::new();
        for d in &draws {
            match d.topology {
                Topology::Triangles => {
                    self.emit_faces(d, &vp, &view, eye, &lighting, half_w, half_h, &mut items)
                }
                Topology::Lines => self.emit_lines(d, &vp, &view, half_w, half_h, &mut items),
                Topology::Points => self.emit_points(d, &vp, &view, half_w, half_h, &mut items),
            }
        }

        if opt.sort {
            // Stable, so equal keys keep scene-graph order. `render_order`
            // outranks depth for the same reason it does on the GPU: it is the
            // caller saying "this goes on top" about an overlay whose depth
            // would otherwise bury it.
            items.sort_by(|a, b| {
                a.order.cmp(&b.order).then(
                    b.depth
                        .partial_cmp(&a.depth)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
            });
        }

        self.document(scene, &items)
    }

    /// Render straight to a `.svg` file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_to_file(
        &self,
        scene: &mut Scene,
        camera: &dyn Camera,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        std::fs::write(path, self.render_to_string(scene, camera))
    }

    /// Wrap an already-rendered RGBA frame — from
    /// [`HeadlessRenderer`](crate::renderer::HeadlessRenderer) or the path
    /// tracer — in an SVG document at this renderer's size. See
    /// [`svg_from_rgba`].
    pub fn embed_rgba(&self, rgba: &[u8]) -> String {
        svg_from_rgba(self.width, self.height, rgba)
    }

    // ---- document assembly ----

    fn document(&self, scene: &Scene, items: &[Item]) -> String {
        let opt = &self.options;
        let p = opt.precision;
        let (w, h) = (self.width, self.height);
        let mut out = String::with_capacity(64 * items.len() + 512);
        out.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
             viewBox=\"0 0 {w} {h}\" shape-rendering=\"geometricPrecision\">\n"
        ));
        let clear = opt.clear_color.unwrap_or(scene.background);
        let clear_alpha = opt.clear_alpha.unwrap_or(scene.background_alpha);
        if opt.background && clear_alpha > 0.0 {
            let bg = encode([clear.r, clear.g, clear.b], opt.tone_mapping, opt.exposure);
            out.push_str(&format!(
                "<rect width=\"{w}\" height=\"{h}\" fill=\"{}\"{}/>\n",
                hex(bg),
                opacity_attr("fill-opacity", clear_alpha, p),
            ));
        }
        for it in items {
            match &it.prim {
                Prim::Face {
                    outline,
                    fill,
                    opacity,
                    stroke,
                } => {
                    let f = hex(*fill);
                    out.push_str(&format!(
                        "<path d=\"{}\" fill=\"{f}\"",
                        outline.to_path_data(p)
                    ));
                    if *stroke > 0.0 {
                        out.push_str(&format!(
                            " stroke=\"{f}\" stroke-width=\"{}\" stroke-linejoin=\"round\"",
                            fmt(*stroke, p)
                        ));
                    }
                    out.push_str(&opacity_attr("opacity", *opacity, p));
                    out.push_str("/>\n");
                }
                Prim::Wire {
                    outline,
                    stroke,
                    width,
                    opacity,
                } => {
                    out.push_str(&format!(
                        "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\"{}/>\n",
                        outline.to_path_data(p),
                        hex(*stroke),
                        fmt(*width, p),
                        opacity_attr("stroke-opacity", *opacity, p),
                    ));
                }
                Prim::Line {
                    a,
                    b,
                    stroke,
                    width,
                    opacity,
                    dash,
                } => {
                    out.push_str(&format!(
                        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" \
                         stroke-width=\"{}\" stroke-linecap=\"round\"",
                        fmt(a[0], p),
                        fmt(a[1], p),
                        fmt(b[0], p),
                        fmt(b[1], p),
                        hex(*stroke),
                        fmt(*width, p),
                    ));
                    if let Some((dash, gap)) = dash {
                        out.push_str(&format!(
                            " stroke-dasharray=\"{},{}\"",
                            fmt(*dash, p),
                            fmt(*gap, p)
                        ));
                    }
                    out.push_str(&opacity_attr("stroke-opacity", *opacity, p));
                    out.push_str("/>\n");
                }
                Prim::Dot {
                    c,
                    r,
                    fill,
                    opacity,
                } => {
                    out.push_str(&format!(
                        "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\"{}/>\n",
                        fmt(c[0], p),
                        fmt(c[1], p),
                        fmt(*r, p),
                        hex(*fill),
                        opacity_attr("fill-opacity", *opacity, p),
                    ));
                }
            }
        }
        out.push_str("</svg>\n");
        out
    }
}

// ======================================================================
//                          SCENE → DRAW LIST
// ======================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Topology {
    Triangles,
    Lines,
    Points,
}

struct Draw {
    geometry: Arc<BufferGeometry>,
    material: Arc<Material>,
    model: Matrix4,
    /// Per-instance tint from an `InstancedMesh`, multiplied into the base
    /// colour. `None` for everything else.
    tint: Option<Color>,
    order: i32,
    topology: Topology,
}

/// Flatten the visible scene graph into one draw per primitive batch.
///
/// `InstancedMesh` expands to one draw per instance: the SVG has to sort
/// instances against each other and against the rest of the scene, so there is
/// nothing to gain by keeping them together the way a GPU draw call does.
///
/// `Sprite` is skipped — a sprite is a textured billboard, and a texture is the
/// one thing this renderer has no way to express.
fn collect_draws(scene: &Scene, cam_layers: crate::core::Layers) -> Vec<Draw> {
    let mut draws = Vec::new();
    let root = scene.root;
    scene.arena.traverse_visible(root, &mut |_, obj| {
        if !cam_layers.test(&obj.layers) {
            return;
        }
        let order = obj.render_order;
        match &obj.kind {
            ObjectKind::Mesh(m) => draws.push(Draw {
                geometry: m.geometry.clone(),
                material: m.material.clone(),
                model: obj.matrix_world,
                tint: None,
                order,
                topology: Topology::Triangles,
            }),
            ObjectKind::SkinnedMesh(m) => draws.push(Draw {
                geometry: m.geometry.clone(),
                material: m.material.clone(),
                model: obj.matrix_world,
                tint: None,
                order,
                topology: Topology::Triangles,
            }),
            ObjectKind::InstancedMesh(m) => {
                for (i, t) in m.transforms.iter().enumerate() {
                    draws.push(Draw {
                        geometry: m.geometry.clone(),
                        material: m.material.clone(),
                        model: obj.matrix_world.multiply(t),
                        tint: m.colors.get(i).copied(),
                        order,
                        topology: Topology::Triangles,
                    });
                }
            }
            ObjectKind::LineSegments(l) => draws.push(Draw {
                geometry: l.geometry.clone(),
                material: l.material.clone(),
                model: obj.matrix_world,
                tint: None,
                order,
                topology: Topology::Lines,
            }),
            ObjectKind::Points(p) => draws.push(Draw {
                geometry: p.geometry.clone(),
                material: p.material.clone(),
                model: obj.matrix_world,
                tint: None,
                order,
                topology: Topology::Points,
            }),
            ObjectKind::Group | ObjectKind::Light(_) | ObjectKind::Sprite(_) => {}
        }
    });
    draws
}

// ======================================================================
//                               LIGHTING
// ======================================================================

#[derive(Debug, Clone, Copy)]
struct DirTerm {
    /// Direction the light travels, normalised.
    dir: Vector3,
    color: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
struct PointTerm {
    pos: Vector3,
    color: [f32; 3],
    distance: f32,
    decay: f32,
}

#[derive(Debug, Clone, Copy)]
struct SpotTerm {
    pos: Vector3,
    dir: Vector3,
    color: [f32; 3],
    distance: f32,
    decay: f32,
    cos_outer: f32,
    cos_inner: f32,
}

#[derive(Debug, Clone, Copy)]
struct HemiTerm {
    up: Vector3,
    sky: [f32; 3],
    ground: [f32; 3],
}

/// The scene's lights, resolved to world space, in the same shape the GPU
/// renderer packs into its frame uniform.
#[derive(Debug, Default, Clone)]
struct Lighting {
    ambient: [f32; 3],
    dir: Vec<DirTerm>,
    point: Vec<PointTerm>,
    spot: Vec<SpotTerm>,
    hemi: Vec<HemiTerm>,
}

impl Lighting {
    fn collect(scene: &Scene) -> Self {
        let mut out = Lighting::default();
        let root = scene.root;
        scene.arena.traverse_visible(root, &mut |_, obj| {
            let ObjectKind::Light(light) = &obj.kind else {
                return;
            };
            match light {
                Light::Ambient(l) => {
                    out.ambient[0] += l.color.r * l.intensity;
                    out.ambient[1] += l.color.g * l.intensity;
                    out.ambient[2] += l.color.b * l.intensity;
                }
                Light::Directional(l) => {
                    if out.dir.len() >= MAX_LIGHTS_PER_KIND {
                        return;
                    }
                    // three.js: direction = normalize(target - position), with
                    // the default target at the origin. A light sitting exactly
                    // at the origin has no such vector, so fall back to the
                    // explicit `direction` field.
                    let pos = obj.world_position();
                    let d = -pos;
                    let dir = if d.length_sq() > 1e-8 {
                        d.normalize()
                    } else {
                        transform_direction(&obj.matrix_world, l.direction).normalize()
                    };
                    out.dir.push(DirTerm {
                        dir,
                        color: scaled(l.color, l.intensity),
                    });
                }
                Light::Point(l) => {
                    if out.point.len() >= MAX_LIGHTS_PER_KIND {
                        return;
                    }
                    out.point.push(PointTerm {
                        pos: obj.world_position(),
                        color: scaled(l.color, l.intensity),
                        distance: l.distance,
                        decay: l.decay,
                    });
                }
                Light::Spot(l) => {
                    if out.spot.len() >= MAX_LIGHTS_PER_KIND {
                        return;
                    }
                    out.spot.push(SpotTerm {
                        pos: obj.world_position(),
                        dir: transform_direction(&obj.matrix_world, l.direction).normalize(),
                        color: scaled(l.color, l.intensity),
                        distance: l.distance,
                        decay: l.decay,
                        cos_outer: l.angle.cos(),
                        cos_inner: (l.angle * (1.0 - l.penumbra)).cos(),
                    });
                }
                Light::Hemisphere(l) => {
                    if out.hemi.len() >= MAX_LIGHTS_PER_KIND {
                        return;
                    }
                    out.hemi.push(HemiTerm {
                        up: transform_direction(&obj.matrix_world, Vector3::UP).normalize(),
                        sky: scaled(l.sky_color, l.intensity),
                        ground: scaled(l.ground_color, l.intensity),
                    });
                }
                // A rect-area light's contribution is an integral over its
                // surface that only pays off per-pixel; at one sample per face
                // it would read as a slightly odd point light. Left out rather
                // than approximated badly.
                Light::RectArea(_) => {}
            }
        });
        out
    }

    fn is_empty(&self) -> bool {
        self.ambient == [0.0; 3]
            && self.dir.is_empty()
            && self.point.is_empty()
            && self.spot.is_empty()
            && self.hemi.is_empty()
    }

    /// Lambert (+ Blinn-Phong specular for `MeshPhongMaterial`) at one point.
    /// Mirrors the `MAT_LAMBERT`/`MAT_PHONG` branch of the wgpu shader, down to
    /// the `1/PI` that separates direct irradiance from the indirect terms.
    fn shade(
        &self,
        n: Vector3,
        world: Vector3,
        view_dir: Vector3,
        specular: [f32; 3],
        shininess: f32,
    ) -> [f32; 3] {
        // three.js applies `BRDF_Lambert` (diffuse / PI) to direct irradiance
        // but not to the indirect terms, which arrive already premultiplied.
        // The wgpu shader splits the sum the same way.
        use std::f32::consts::FRAC_1_PI;
        let mut indirect = self.ambient;
        let mut direct = [0.0f32; 3];

        for l in &self.dir {
            let to_light = -l.dir;
            add_punctual(
                &mut direct,
                n,
                to_light,
                view_dir,
                l.color,
                1.0,
                specular,
                shininess,
            );
        }
        for l in &self.point {
            let to = l.pos - world;
            let d = to.length();
            let to_light = to * (1.0 / d.max(1e-4));
            let att = punctual_attenuation(d, l.distance, l.decay);
            add_punctual(
                &mut direct,
                n,
                to_light,
                view_dir,
                l.color,
                att,
                specular,
                shininess,
            );
        }
        for l in &self.spot {
            let to = l.pos - world;
            let d = to.length();
            let to_light = to * (1.0 / d.max(1e-4));
            let cos_angle = (-to_light).dot(l.dir);
            let cone = if cos_angle > l.cos_outer {
                smoothstep(l.cos_outer, l.cos_inner, cos_angle)
            } else {
                0.0
            };
            if cone <= 0.0 {
                continue;
            }
            let att = punctual_attenuation(d, l.distance, l.decay) * cone;
            add_punctual(
                &mut direct,
                n,
                to_light,
                view_dir,
                l.color,
                att,
                specular,
                shininess,
            );
        }
        for l in &self.hemi {
            let t = n.dot(l.up) * 0.5 + 0.5;
            for (c, (sky, ground)) in indirect.iter_mut().zip(l.sky.iter().zip(&l.ground)) {
                *c += ground + (sky - ground) * t;
            }
        }

        [
            indirect[0] + direct[0] * FRAC_1_PI,
            indirect[1] + direct[1] * FRAC_1_PI,
            indirect[2] + direct[2] * FRAC_1_PI,
        ]
    }
}

#[allow(clippy::too_many_arguments)]
fn add_punctual(
    acc: &mut [f32; 3],
    n: Vector3,
    to_light: Vector3,
    view_dir: Vector3,
    color: [f32; 3],
    att: f32,
    specular: [f32; 3],
    shininess: f32,
) {
    let lambert = n.dot(to_light).max(0.0);
    if lambert <= 0.0 || att <= 0.0 {
        return;
    }
    let spec = if specular != [0.0; 3] {
        let half_v = (to_light + view_dir).normalize();
        n.dot(half_v).max(0.0).powf(shininess.max(1.0))
    } else {
        0.0
    };
    for i in 0..3 {
        acc[i] += color[i] * (lambert + specular[i] * spec) * att;
    }
}

/// three.js `getDistanceAttenuation`: `1 / max(d^decay, 0.01)`, with a quartic
/// window rolling it to zero at `max_d` when that cutoff is set.
fn punctual_attenuation(d: f32, max_d: f32, decay: f32) -> f32 {
    let mut att = 1.0 / d.powf(decay).max(0.01);
    if max_d > 0.0 {
        let t = (1.0 - (d / max_d).powi(4)).clamp(0.0, 1.0);
        att *= t * t;
    }
    att
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-8 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn scaled(c: Color, k: f32) -> [f32; 3] {
    [c.r * k, c.g * k, c.b * k]
}

// ======================================================================
//                         PRIMITIVE EMISSION
// ======================================================================

/// A closed outline: where the pen starts, then one segment per edge.
///
/// Edges are [`PathSegment`]s rather than points because a face's boundary is
/// not always straight — see [`SvgOptions::curve_tolerance`] — and because it
/// puts the renderer's path writing through the same code as
/// [`Path::to_svg_path_data`](crate::Path::to_svg_path_data).
struct Outline {
    start: Vector2,
    edges: Vec<PathSegment>,
}

impl Outline {
    fn straight(pts: &[[f32; 2]]) -> Self {
        Self {
            start: Vector2::new(pts[0][0], pts[0][1]),
            edges: pts[1..]
                .iter()
                .map(|p| PathSegment::Line {
                    to: Vector2::new(p[0], p[1]),
                })
                .collect(),
        }
    }

    fn to_path_data(&self, precision: usize) -> String {
        segments_to_path_data(self.start, &self.edges, precision, true)
    }
}

enum Prim {
    Face {
        outline: Outline,
        fill: [u8; 3],
        opacity: f32,
        stroke: f32,
    },
    Wire {
        outline: Outline,
        stroke: [u8; 3],
        width: f32,
        opacity: f32,
    },
    Line {
        a: [f32; 2],
        b: [f32; 2],
        stroke: [u8; 3],
        width: f32,
        opacity: f32,
        dash: Option<(f32, f32)>,
    },
    Dot {
        c: [f32; 2],
        r: f32,
        fill: [u8; 3],
        opacity: f32,
    },
}

struct Item {
    order: i32,
    /// How far back this primitive reaches: the largest view-space distance
    /// over its vertices. Sorted descending, so the thing that reaches farthest
    /// is painted first.
    ///
    /// Two choices here, both of which cost pictures to get wrong.
    ///
    /// **Linear distance, not NDC z.** NDC z is hyperbolic — it spends most of
    /// its range in the first few units past the near plane — so a face left
    /// with a vertex on the near plane by the clipper reads as far nearer than
    /// it is. A ground plane running off behind the camera sorted to the front
    /// of the scene and painted over everything.
    ///
    /// **Farthest vertex, not the centroid.** The centroid is the textbook
    /// key and it is worse, because the case that actually turns up is a large
    /// polygon and small objects sitting on it. A floor's centroid lands out in
    /// the middle of the scene, among the objects, and whichever ones it sorts
    /// in front of get their bases painted over — the "everything is sliced off
    /// at the floor line" artefact. Where the polygon *reaches* is the more
    /// useful question: a floor reaches to the horizon, so it is always painted
    /// first, and anything standing on it survives. Measured over ten scenes,
    /// this key was exactly right where the centroid was wrong on three of them
    /// and no worse on any.
    ///
    /// It is still one number for a whole face, so it can still be wrong: a
    /// large polygon that *occludes* something while reaching past it sorts
    /// behind and lets the hidden thing show through. That is what
    /// [`SvgOptions::depth_split`] is for.
    depth: f32,
    prim: Prim,
}

/// A vertex in flight: clip space for the divide, world space for shading, and
/// the linear depth the painter's sort runs on.
///
/// All three are affine in the world position, so a midpoint or a clip-plane
/// crossing interpolates all of them with the same parameter.
#[derive(Debug, Clone, Copy)]
struct Vtx {
    clip: [f32; 4],
    world: Vector3,
    depth: f32,
    /// Surface normal here, when the geometry carries one. Interpolated along
    /// with the rest so that a vertex introduced by a split or a clip still
    /// knows which way the surface faces — which is what
    /// [`SvgOptions::curve_tolerance`] needs to bend an edge back onto it.
    normal: Option<Vector3>,
}

impl Vtx {
    fn new(vp: &Matrix4, view: &Matrix4, world: Vector3, normal: Option<Vector3>) -> Self {
        Self {
            clip: to_clip(vp, world),
            world,
            depth: -transform_point(view, world).z,
            normal,
        }
    }

    fn lerp(a: Self, b: Self, t: f32) -> Self {
        Self {
            clip: lerp4(a.clip, b.clip, t),
            world: a.world + (b.world - a.world) * t,
            depth: a.depth + (b.depth - a.depth) * t,
            normal: match (a.normal, b.normal) {
                (Some(na), Some(nb)) => {
                    let n = na + (nb - na) * t;
                    Some(if n.length_sq() > 1e-12 {
                        n.normalize()
                    } else {
                        na
                    })
                }
                _ => None,
            },
        }
    }
}

/// Recursion cap on [`subdivide_by_depth`]. The tolerance normally stops the
/// recursion long before this; the cap is only there so a polygon stretching to
/// the horizon cannot run away, and it binds on exactly that case — a ground
/// plane, where the span-to-distance ratio near the camera is unbounded.
const MAX_DEPTH_SPLITS: u8 = 10;

/// Split a triangle until no piece spans much depth relative to its own
/// distance from the camera.
///
/// The painter's algorithm gives each face one depth, and one number cannot
/// order a polygon that straddles another. The stock symptom is a ground plane
/// drawing over the bottom of everything standing on it: the floor is two
/// triangles spanning the whole scene, and whichever side of the sphere their
/// single depth lands on, half the picture is wrong.
///
/// Splitting the edge with the largest depth difference halves the span each
/// time, so a few levels bring the pieces down to where a per-face depth is
/// meaningful. Geometry that is already tessellated — anything from a sphere to
/// a CAD import — passes the test on the first call and costs nothing.
fn subdivide_by_depth(tri: [Vtx; 3], tolerance: f32, budget: u8, out: &mut Vec<[Vtx; 3]>) {
    let d = [tri[0].depth, tri[1].depth, tri[2].depth];
    let near = d.iter().copied().fold(f32::INFINITY, f32::min);
    let span = d.iter().copied().fold(f32::NEG_INFINITY, f32::max) - near;
    // A non-finite span means a degenerate triangle; splitting it forever would
    // not make it any more sortable.
    if budget == 0 || !span.is_finite() || span <= tolerance * near.max(1e-3) {
        out.push(tri);
        return;
    }
    // Rotations of (0, 1, 2), so both halves keep the parent's winding — the
    // backface cull reads it downstream.
    let (a, b, c) = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        .into_iter()
        .max_by(|x, y| {
            let dx = (d[x.0] - d[x.1]).abs();
            let dy = (d[y.0] - d[y.1]).abs();
            dx.partial_cmp(&dy).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    let m = Vtx::lerp(tri[a], tri[b], 0.5);
    subdivide_by_depth([tri[a], m, tri[c]], tolerance, budget - 1, out);
    subdivide_by_depth([m, tri[b], tri[c]], tolerance, budget - 1, out);
}

impl SvgRenderer {
    #[allow(clippy::too_many_arguments)]
    fn emit_faces(
        &self,
        d: &Draw,
        vp: &Matrix4,
        view: &Matrix4,
        eye: Vector3,
        lighting: &Lighting,
        half_w: f32,
        half_h: f32,
        items: &mut Vec<Item>,
    ) {
        let opt = &self.options;
        let Some(iter) = d.geometry.positions() else {
            return;
        };
        let local: Vec<Vector3> = iter.collect();
        if local.len() < 3 {
            return;
        }
        let normals = world_normals(&d.geometry, &d.model);
        let verts: Vec<Vtx> = local
            .iter()
            .enumerate()
            .map(|(i, p)| {
                Vtx::new(
                    vp,
                    view,
                    transform_point(&d.model, *p),
                    normals.as_ref().and_then(|ns| ns.get(i).copied()),
                )
            })
            .collect();

        let mat = &*d.material;
        let wireframe = opt.shading == SvgShading::Wireframe || mat.wireframe();
        // With no lights at all, `Lit` would render the whole scene black. That
        // is technically correct and never what anyone wants from a line
        // drawing, so an unlit scene falls back to flat material colour.
        let lit = opt.shading == SvgShading::Lit && !lighting.is_empty();
        let side = mat.side();
        let cull = opt.cull_backfaces && !wireframe;
        let opacity = mat.opacity().clamp(0.0, 1.0);
        if opacity <= 0.0 {
            return;
        }
        let base = tinted(mat.color(), d.tint);
        let emissive = mat.emissive();
        let (specular, shininess) = match mat {
            Material::Phong(m) => ([m.specular.r, m.specular.g, m.specular.b], m.shininess),
            _ => ([0.0; 3], 1.0),
        };

        let mut pieces: Vec<[Vtx; 3]> = Vec::new();
        for t in triangles(&d.geometry, local.len()) {
            // Shade from the source triangle, not the piece: subdividing is a
            // sorting fix, and letting it change the shading normal would make
            // the tessellation visible.
            let n_face = face_normal(&verts, &t, normals.as_deref());

            pieces.clear();
            // Splitting exists to make the depth sort work. Wireframe has no
            // fills to sort, so it would buy nothing and cost plenty: every cut
            // is another stroked edge, and the tessellation would be drawn on
            // top of the model as if it were part of it.
            match opt.depth_split.filter(|_| !wireframe) {
                Some(tolerance) => subdivide_by_depth(
                    [verts[t[0]], verts[t[1]], verts[t[2]]],
                    tolerance.max(1e-3),
                    MAX_DEPTH_SPLITS,
                    &mut pieces,
                ),
                None => pieces.push([verts[t[0]], verts[t[1]], verts[t[2]]]),
            }

            for piece in &pieces {
                let clipped = clip_polygon(piece);
                if clipped.len() < 3 {
                    continue;
                }
                let ndc: Vec<[f32; 3]> = clipped.iter().map(|v| ndc_of(v.clip)).collect();
                if outside_frustum(&ndc) {
                    continue;
                }

                // Front-facing is CCW in NDC, matching `FrontFace::Ccw` on the
                // wgpu pipelines.
                let front = signed_area(&ndc) > 0.0;
                if cull {
                    let keep = match side {
                        1 => !front, // BackSide
                        2 => true,   // DoubleSide
                        _ => front,  // FrontSide
                    };
                    if !keep {
                        continue;
                    }
                }
                // BackSide and DoubleSide light whichever face the camera can
                // actually see, so the normal follows the winding.
                let n = if side != 0 && !front { -n_face } else { n_face };

                let pts: Vec<[f32; 2]> = ndc
                    .iter()
                    .map(|p| [(p[0] + 1.0) * half_w, (1.0 - p[1]) * half_h])
                    .collect();
                let outline = match opt.curve_tolerance {
                    Some(tolerance) => {
                        curved_outline(&clipped, &pts, vp, half_w, half_h, tolerance)
                    }
                    None => Outline::straight(&pts),
                };
                let depth = clipped
                    .iter()
                    .map(|v| v.depth)
                    .fold(f32::NEG_INFINITY, f32::max);
                let centroid = clipped.iter().fold(Vector3::ZERO, |acc, v| acc + v.world)
                    * (1.0 / clipped.len() as f32);
                let ndc_z = ndc.iter().map(|p| p[2]).sum::<f32>() / ndc.len() as f32;

                let (rgb, display_referred) = shade_face(
                    mat, base, emissive, specular, shininess, n, centroid, eye, view, ndc_z, lit,
                    lighting,
                );
                let fill = if display_referred {
                    rgb.map(|c| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
                } else {
                    encode(rgb, opt.tone_mapping, opt.exposure)
                };

                items.push(Item {
                    order: d.order,
                    depth,
                    prim: if wireframe {
                        Prim::Wire {
                            outline,
                            stroke: fill,
                            width: 1.0,
                            opacity,
                        }
                    } else {
                        Prim::Face {
                            outline,
                            fill,
                            opacity,
                            stroke: opt.seam_stroke,
                        }
                    },
                });
            }
        }
    }

    fn emit_lines(
        &self,
        d: &Draw,
        vp: &Matrix4,
        view: &Matrix4,
        half_w: f32,
        half_h: f32,
        items: &mut Vec<Item>,
    ) {
        let opt = &self.options;
        let Some(iter) = d.geometry.positions() else {
            return;
        };
        let local: Vec<Vector3> = iter.collect();
        let verts: Vec<Vtx> = local
            .iter()
            .map(|p| Vtx::new(vp, view, transform_point(&d.model, *p), None))
            .collect();

        let mat = &*d.material;
        let opacity = mat.opacity().clamp(0.0, 1.0);
        let c = mat.color();
        let stroke = encode([c.r, c.g, c.b], opt.tone_mapping, opt.exposure);
        let (width, dash) = match mat {
            Material::Line(m) => (
                m.line_width.max(0.1),
                if m.dashed {
                    Some((m.dash_size * m.dash_scale, m.gap_size * m.dash_scale))
                } else {
                    None
                },
            ),
            _ => (1.0, None),
        };

        let pairs: Vec<[usize; 2]> = match &d.geometry.index {
            Some(idx) => idx
                .chunks_exact(2)
                .map(|c| [c[0] as usize, c[1] as usize])
                .collect(),
            None => (0..local.len() / 2).map(|i| [i * 2, i * 2 + 1]).collect(),
        };

        for s in pairs {
            if s[0] >= verts.len() || s[1] >= verts.len() {
                continue;
            }
            let Some((a, b)) = clip_segment(verts[s[0]], verts[s[1]]) else {
                continue;
            };
            let (na, nb) = (ndc_of(a.clip), ndc_of(b.clip));
            if outside_frustum(&[na, nb]) {
                continue;
            }
            items.push(Item {
                order: d.order,
                depth: (a.depth + b.depth) * 0.5,
                prim: Prim::Line {
                    a: [(na[0] + 1.0) * half_w, (1.0 - na[1]) * half_h],
                    b: [(nb[0] + 1.0) * half_w, (1.0 - nb[1]) * half_h],
                    stroke,
                    width,
                    opacity,
                    dash,
                },
            });
        }
    }

    fn emit_points(
        &self,
        d: &Draw,
        vp: &Matrix4,
        view: &Matrix4,
        half_w: f32,
        half_h: f32,
        items: &mut Vec<Item>,
    ) {
        let opt = &self.options;
        let Some(iter) = d.geometry.positions() else {
            return;
        };
        let mat = &*d.material;
        let opacity = mat.opacity().clamp(0.0, 1.0);
        let c = mat.color();
        let fill = encode([c.r, c.g, c.b], opt.tone_mapping, opt.exposure);
        let (size, attenuate) = match mat {
            Material::Points(m) => (m.size, m.size_attenuation),
            _ => (1.0, true),
        };

        for p in iter {
            let v = Vtx::new(vp, view, transform_point(&d.model, p), None);
            if v.clip[3] < NEAR_W || v.clip[2] < 0.0 {
                continue;
            }
            let n = ndc_of(v.clip);
            if outside_frustum(&[n]) {
                continue;
            }
            // Size attenuation on the GPU is `size / -viewZ` scaled by half the
            // viewport height. Under an orthographic camera `w` is 1, which is
            // exactly the "ignored under orthographic" the material documents.
            let r = if attenuate {
                (size * half_h / v.clip[3]).max(0.05)
            } else {
                size * 0.5
            };
            items.push(Item {
                order: d.order,
                depth: v.depth,
                prim: Prim::Dot {
                    c: [(n[0] + 1.0) * half_w, (1.0 - n[1]) * half_h],
                    r,
                    fill,
                    opacity,
                },
            });
        }
    }
}

/// The face's colour, plus whether it is already display-referred and must skip
/// the tone-map/sRGB encode — which is how the wgpu shader treats
/// `MeshNormalMaterial` and `MeshDepthMaterial`, both of which return raw
/// debug values rather than light.
#[allow(clippy::too_many_arguments)]
fn shade_face(
    mat: &Material,
    base: [f32; 3],
    emissive: Color,
    specular: [f32; 3],
    shininess: f32,
    n: Vector3,
    world: Vector3,
    eye: Vector3,
    view: &Matrix4,
    ndc_z: f32,
    lit: bool,
    lighting: &Lighting,
) -> ([f32; 3], bool) {
    match mat {
        Material::Normal(_) => {
            let vn = transform_direction(view, n).normalize();
            ([vn.x, vn.y, vn.z].map(|c| c * 0.5 + 0.5), true)
        }
        Material::Depth(_) => {
            let vis = 1.0 - (0.5 * ndc_z + 0.5);
            ([vis; 3], true)
        }
        // MeshBasicMaterial is unlit by definition.
        Material::Basic(_) | Material::Line(_) | Material::Points(_) | Material::Sprite(_) => {
            (base, false)
        }
        _ if !lit => (base, false),
        _ => {
            let view_dir = (eye - world).normalize();
            let light = lighting.shade(n, world, view_dir, specular, shininess);
            (
                [
                    base[0] * light[0] + emissive.r,
                    base[1] * light[1] + emissive.g,
                    base[2] * light[2] + emissive.b,
                ],
                false,
            )
        }
    }
}

fn tinted(c: Color, tint: Option<Color>) -> [f32; 3] {
    match tint {
        Some(t) => [c.r * t.r, c.g * t.g, c.b * t.b],
        None => [c.r, c.g, c.b],
    }
}

/// Averaged vertex normal when the geometry carries one, else the triangle's
/// own plane normal. Averaging reads better across a tessellated curve: the
/// facets still show, but their shading follows the surface rather than the
/// mesh.
fn face_normal(verts: &[Vtx], t: &[usize; 3], normals: Option<&[Vector3]>) -> Vector3 {
    if let Some(ns) = normals {
        if t[0] < ns.len() && t[1] < ns.len() && t[2] < ns.len() {
            let sum = ns[t[0]] + ns[t[1]] + ns[t[2]];
            if sum.length_sq() > 1e-12 {
                return sum.normalize();
            }
        }
    }
    let e1 = verts[t[1]].world - verts[t[0]].world;
    let e2 = verts[t[2]].world - verts[t[0]].world;
    let n = e1.cross(e2);
    if n.length_sq() > 1e-20 {
        n.normalize()
    } else {
        Vector3::UP
    }
}

/// A face's boundary with each edge bent back onto the surface it came from.
///
/// The construction is the edge half of a PN triangle (Vlachos et al. 2001):
/// take an edge's two endpoints and the surface normals there, and place the
/// two cubic control points at the thirds of the chord, pushed off it by
/// however much the normal says the surface leans away. The result meets the
/// surface's tangent plane at both ends, which is the curve the tessellation
/// was approximating with a chord in the first place.
///
/// The reason neighbouring faces still tile with no crack: the curve depends
/// only on the edge's two endpoints and their normals, so the two faces sharing
/// an edge compute the *same* curve, one of them traversing it backwards.
/// Nothing here is per-face.
fn curved_outline(
    poly: &[Vtx],
    pts: &[[f32; 2]],
    vp: &Matrix4,
    half_w: f32,
    half_h: f32,
    tolerance: f32,
) -> Outline {
    let screen = |p: [f32; 2]| Vector2::new(p[0], p[1]);
    let n = poly.len();
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b) = (i, (i + 1) % n);
        let to = screen(pts[b]);
        let line = PathSegment::Line { to };
        let (Some(na), Some(nb)) = (poly[a].normal, poly[b].normal) else {
            edges.push(line);
            continue;
        };
        let (pa, pb) = (poly[a].world, poly[b].world);
        let chord = pb - pa;
        let b1 = (pa * 2.0 + pb - na * chord.dot(na)) * (1.0 / 3.0);
        let b2 = (pb * 2.0 + pa - nb * (-chord).dot(nb)) * (1.0 / 3.0);

        let (Some(c1), Some(c2)) = (
            project(vp, b1, half_w, half_h),
            project(vp, b2, half_w, half_h),
        ) else {
            edges.push(line);
            continue;
        };
        // How far the curve strays from the chord it replaces. A cubic stays
        // within 3/4 of its control points' offset from the chord, so this
        // bounds the error without evaluating the curve. Below the tolerance
        // the shorter command wins: a flat face has one normal across it, both
        // control points land on the chord, and every edge stays an `L`.
        let from = screen(pts[a]);
        let deviation =
            0.75 * distance_to_segment(c1, from, to).max(distance_to_segment(c2, from, to));
        // A sane surface arc bulges by a fraction of its chord — a half circle,
        // the most an edge should ever span, reaches half of it. Anything past
        // that is a normal that disagrees with the geometry it is attached to,
        // and bending to it throws a spike out through the silhouette. Those
        // show up on coarse meshes and on normals that were authored rather
        // than computed.
        let over = deviation > (to - from).length() * 0.5;
        edges.push(if deviation > tolerance && !over {
            PathSegment::Cubic { c1, c2, to }
        } else {
            line
        });
    }
    Outline {
        start: screen(pts[0]),
        edges,
    }
}

/// World point to pixels, or `None` if it is not in front of the camera.
fn project(vp: &Matrix4, p: Vector3, half_w: f32, half_h: f32) -> Option<Vector2> {
    let c = to_clip(vp, p);
    if c[3] < NEAR_W {
        return None;
    }
    let n = ndc_of(c);
    Some(Vector2::new((n[0] + 1.0) * half_w, (1.0 - n[1]) * half_h))
}

fn distance_to_segment(p: Vector2, a: Vector2, b: Vector2) -> f32 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    if len2 < 1e-12 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

fn world_normals(geometry: &BufferGeometry, model: &Matrix4) -> Option<Vec<Vector3>> {
    let attr = geometry.get_attribute("normal")?;
    if attr.item_size != 3 {
        return None;
    }
    // Normals transform by the inverse transpose, or non-uniform scale shears
    // them off the surface.
    let inv = model.invert();
    Some(
        attr.array
            .chunks_exact(3)
            .map(|c| transform_normal(&inv, Vector3::new(c[0], c[1], c[2])))
            .collect(),
    )
}

fn triangles(geometry: &BufferGeometry, vertex_count: usize) -> Vec<[usize; 3]> {
    match &geometry.index {
        Some(idx) => idx
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .filter(|t| t.iter().all(|&i| i < vertex_count))
            .collect(),
        None => (0..vertex_count / 3)
            .map(|i| [i * 3, i * 3 + 1, i * 3 + 2])
            .collect(),
    }
}

// ======================================================================
//                      PROJECTION / CLIP GEOMETRY
// ======================================================================

fn transform_point(m: &Matrix4, p: Vector3) -> Vector3 {
    let e = &m.elements;
    Vector3::new(
        e[0] * p.x + e[4] * p.y + e[8] * p.z + e[12],
        e[1] * p.x + e[5] * p.y + e[9] * p.z + e[13],
        e[2] * p.x + e[6] * p.y + e[10] * p.z + e[14],
    )
}

fn transform_direction(m: &Matrix4, v: Vector3) -> Vector3 {
    let e = &m.elements;
    Vector3::new(
        e[0] * v.x + e[4] * v.y + e[8] * v.z,
        e[1] * v.x + e[5] * v.y + e[9] * v.z,
        e[2] * v.x + e[6] * v.y + e[10] * v.z,
    )
}

/// `(M^-1)^T * n` for the upper 3x3 — the same product as
/// [`transform_direction`] but reading `inv` down its columns instead of across
/// its rows, which is what the transpose amounts to.
fn transform_normal(inv: &Matrix4, n: Vector3) -> Vector3 {
    let e = &inv.elements;
    let out = Vector3::new(
        e[0] * n.x + e[1] * n.y + e[2] * n.z,
        e[4] * n.x + e[5] * n.y + e[6] * n.z,
        e[8] * n.x + e[9] * n.y + e[10] * n.z,
    );
    if out.length_sq() > 1e-20 {
        out.normalize()
    } else {
        n
    }
}

fn to_clip(m: &Matrix4, p: Vector3) -> [f32; 4] {
    let e = &m.elements;
    [
        e[0] * p.x + e[4] * p.y + e[8] * p.z + e[12],
        e[1] * p.x + e[5] * p.y + e[9] * p.z + e[13],
        e[2] * p.x + e[6] * p.y + e[10] * p.z + e[14],
        e[3] * p.x + e[7] * p.y + e[11] * p.z + e[15],
    ]
}

fn ndc_of(c: [f32; 4]) -> [f32; 3] {
    let inv_w = 1.0 / c[3];
    [c[0] * inv_w, c[1] * inv_w, c[2] * inv_w]
}

/// The clip-space planes a primitive is cut against before the perspective
/// divide, as signed distances that are positive inside.
///
/// **Near, `z >= 0`.** The crate's projection matrices put NDC z in `[0, 1]`
/// (the wgpu/D3D convention — see
/// [`Matrix4::perspective`](crate::math::Matrix4::perspective)). Clipping at
/// `w > 0` alone would stop the *sign* flip that flings geometry behind the eye
/// to the far side of the screen, but not the magnitude: a vertex just in front
/// of the eye has a vanishing `w` and `x / w` still runs to millions. The near
/// plane pins `w` to the camera's `near`.
///
/// **`w >= NEAR_W`** guards a degenerate or overridden projection against
/// dividing by zero. Under a well-formed matrix it never binds, because
/// `z >= 0` already implies `w >= near`.
///
/// **The four sides.** A viewer clips to the `viewBox` anyway, so these buy
/// nothing on screen — they matter because a document is not only ever looked
/// at in a browser. A floor plane running to the horizon projects to
/// coordinates thousands of units outside a 640-pixel frame, and an editor that
/// fits the *content* bounding box rather than the `viewBox` — Illustrator,
/// Inkscape, Figma, a thumbnailer — then shows the artwork as a small offset
/// speck. Cutting the geometry to the frame keeps the bounding box and the
/// canvas the same thing, and makes the file smaller besides.
const CLIP_PLANES: [fn(&Vtx) -> f32; 6] = [
    |v| v.clip[2],
    |v| v.clip[3] - NEAR_W,
    |v| v.clip[0] + GUARD * v.clip[3],
    |v| GUARD * v.clip[3] - v.clip[0],
    |v| v.clip[1] + GUARD * v.clip[3],
    |v| GUARD * v.clip[3] - v.clip[1],
];

/// How far past the viewport edge the side planes cut, in NDC.
///
/// Not exactly `1.0`: a cut edge gets the seam stroke drawn along it like any
/// other, so clipping exactly on the frame would draw a hairline border around
/// the whole document. A couple of percent puts the cut edges, and their
/// strokes, just off what anyone sees — a dozen pixels on a 640-wide frame.
const GUARD: f32 = 1.02;

/// Sutherland-Hodgman against [`CLIP_PLANES`], one exact pass per plane.
///
/// Dropping a straddling triangle instead would punch a hole in anything the
/// camera is inside of, so crossings get interpolated and a cut face comes back
/// as a polygon with more vertices than it started with.
fn clip_polygon(poly: &[Vtx]) -> Vec<Vtx> {
    let mut cur: Vec<Vtx> = poly.to_vec();
    let mut next: Vec<Vtx> = Vec::with_capacity(poly.len() + 2);
    for plane in CLIP_PLANES {
        if cur.len() < 3 {
            return Vec::new();
        }
        next.clear();
        for i in 0..cur.len() {
            let a = cur[i];
            let b = cur[(i + 1) % cur.len()];
            let (da, db) = (plane(&a), plane(&b));
            if da >= 0.0 {
                next.push(a);
            }
            if (da >= 0.0) != (db >= 0.0) {
                next.push(Vtx::lerp(a, b, da / (da - db)));
            }
        }
        std::mem::swap(&mut cur, &mut next);
    }
    if cur.len() < 3 {
        Vec::new()
    } else {
        cur
    }
}

/// [`clip_polygon`] for an open two-point segment.
fn clip_segment(mut a: Vtx, mut b: Vtx) -> Option<(Vtx, Vtx)> {
    for plane in CLIP_PLANES {
        let (da, db) = (plane(&a), plane(&b));
        match (da >= 0.0, db >= 0.0) {
            (true, true) => {}
            (false, false) => return None,
            (true, false) => b = Vtx::lerp(a, b, da / (da - db)),
            (false, true) => a = Vtx::lerp(b, a, db / (db - da)),
        }
    }
    Some((a, b))
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Conservative reject: true only when every vertex sits outside the *same*
/// frustum plane, which is the one case where the primitive cannot possibly
/// touch the canvas.
fn outside_frustum(ndc: &[[f32; 3]]) -> bool {
    ndc.iter().all(|p| p[0] < -1.0)
        || ndc.iter().all(|p| p[0] > 1.0)
        || ndc.iter().all(|p| p[1] < -1.0)
        || ndc.iter().all(|p| p[1] > 1.0)
        || ndc.iter().all(|p| p[2] < 0.0)
        || ndc.iter().all(|p| p[2] > 1.0)
}

/// Twice the signed area of the polygon in NDC. Positive is counter-clockwise.
fn signed_area(ndc: &[[f32; 3]]) -> f32 {
    let mut a = 0.0;
    for i in 0..ndc.len() {
        let p = ndc[i];
        let q = ndc[(i + 1) % ndc.len()];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a
}

// ======================================================================
//                         COLOUR AND FORMATTING
// ======================================================================

/// ACES filmic approximation — the same curve and constants as the shader's
/// `aces_tonemap`.
fn aces(x: f32) -> f32 {
    let (a, b, c, d, e) = (2.51, 0.03, 2.43, 0.59, 0.14);
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

/// IEC 61966-2-1 sRGB OETF — three.js's `sRGBTransferOETF`.
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        c.max(0.0).powf(1.0 / 2.4) * 1.055 - 0.055
    }
}

/// Linear light to an sRGB byte triple, applying the same exposure and curve
/// the wgpu renderer applies.
///
/// Skipping this is the single easiest way to get an SVG that does not match
/// its PNG: [`Color::from_hex`] decodes to linear, so writing those floats
/// straight into a `fill` attribute hands the browser linear values labelled
/// as sRGB and renders mid-tones far too dark.
fn encode(c: [f32; 3], tm: ToneMapping, exposure: f32) -> [u8; 3] {
    let mapped = match tm {
        ToneMapping::None => c,
        ToneMapping::Linear => [c[0] * exposure, c[1] * exposure, c[2] * exposure],
        ToneMapping::AcesFilmic => [
            aces(c[0] * exposure),
            aces(c[1] * exposure),
            aces(c[2] * exposure),
        ],
    };
    mapped.map(|v| (linear_to_srgb(v.clamp(0.0, 1.0)).clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// Emit `name="v"` only when `v` is not 1 — SVG's default for every opacity
/// attribute, so writing it is pure noise.
fn opacity_attr(name: &str, v: f32, precision: usize) -> String {
    if v >= 1.0 {
        String::new()
    } else {
        format!(" {name}=\"{}\"", fmt(v.max(0.0), precision.max(3)))
    }
}

// ======================================================================
//                          RASTER PASSTHROUGH
// ======================================================================

/// Wrap an already-rendered RGBA frame in an SVG document, as a losslessly
/// compressed PNG in a `data:` URI.
///
/// This is the other half of "export the render as SVG": [`SvgRenderer`] gives
/// you real vectors but only what a per-face painter's algorithm can express,
/// while this gives you exactly what the GPU drew — textures, shadows, post-fx,
/// path-traced global illumination — in a container that drops into the same
/// documents and pipelines. The pixels do not scale, but nothing is lost or
/// reordered either.
///
/// ```no_run
/// # fn demo(headless: &mut threers::renderer::HeadlessRenderer,
/// #         scene: &mut threers::Scene, camera: &threers::PerspectiveCamera)
/// # -> std::io::Result<()> {
/// let rgba = headless.render_to_rgba(scene, camera);
/// let (w, h) = headless.render_size();
/// std::fs::write("out/frame.svg", threers::svg_from_rgba(w, h, &rgba))
/// # }
/// ```
///
/// Panics if `rgba` is not `width * height * 4` bytes.
pub fn svg_from_rgba(width: u32, height: u32, rgba: &[u8]) -> String {
    let png = crate::utils::png::encode_png(width, height, rgba);
    let b64 = base64_encode(&png);
    let mut out = String::with_capacity(b64.len() + 256);
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" \
         width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n\
         <image width=\"{width}\" height=\"{height}\" image-rendering=\"pixelated\" \
         xlink:href=\"data:image/png;base64,"
    ));
    out.push_str(&b64);
    out.push_str("\"/>\n</svg>\n");
    out
}

/// RFC 4648 base64, padded. The crate ships a decoder for glTF data URIs but
/// no encoder, and one loop is cheaper than a dependency.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cameras::PerspectiveCamera;
    use crate::core::{LineSegments, Mesh, Object3D, Points};
    use crate::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
    use crate::lights::{AmbientLight, DirectionalLight};
    use crate::materials::{BasicMaterial, LambertMaterial, LineBasicMaterial, PointsMaterial};

    fn camera_at(x: f32, y: f32, z: f32) -> PerspectiveCamera {
        let mut c = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        c.position = Vector3::new(x, y, z);
        c.target = Vector3::ZERO;
        c
    }

    fn count(svg: &str, tag: &str) -> usize {
        svg.matches(tag).count()
    }

    /// A `PlaneGeometry` faces +Z. Seen from +Z it must draw; seen from -Z the
    /// backface cull must drop it. This is what pins the winding convention to
    /// the `FrontFace::Ccw` on the wgpu pipelines — get the sign backwards and
    /// every closed mesh renders inside-out.
    fn plane_scene() -> Scene {
        let mut scene = Scene::new();
        let mesh = Mesh::new(
            PlaneGeometry::new(2.0, 2.0),
            Material::Basic(BasicMaterial::new(Color::from_hex(0xff8800))),
        );
        scene.add(Object3D::mesh(mesh));
        scene
    }

    #[test]
    fn front_facing_plane_draws_and_back_facing_is_culled() {
        let r = SvgRenderer::new(200, 200);
        let front = r.render_to_string(&mut plane_scene(), &camera_at(0.0, 0.0, 5.0));
        let back = r.render_to_string(&mut plane_scene(), &camera_at(0.0, 0.0, -5.0));
        assert_eq!(count(&front, "<path"), 2, "two triangles from the front");
        assert_eq!(count(&back, "<path"), 0, "nothing from behind");
    }

    /// The cull is the only thing standing between a closed mesh and double the
    /// polygons it needs. From a corner exactly three of a cube's six faces face
    /// the camera.
    #[test]
    fn cube_shows_three_of_six_faces() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Basic(BasicMaterial::new(Color::WHITE)),
        )));
        // Depth splitting off, so the count reflects the cull and nothing else.
        let opts = SvgOptions {
            depth_split: None,
            ..SvgOptions::default()
        };
        let svg = SvgRenderer::new(200, 200)
            .with_options(opts)
            .render_to_string(&mut scene, &camera_at(3.0, 3.0, 3.0));
        assert_eq!(count(&svg, "<path"), 6, "3 faces x 2 triangles");

        let all = SvgRenderer::new(200, 200)
            .with_options(SvgOptions {
                cull_backfaces: false,
                ..opts
            })
            .render_to_string(&mut scene, &camera_at(3.0, 3.0, 3.0));
        assert_eq!(count(&all, "<path"), 12, "the whole cube when not culling");
    }

    /// `Color::from_hex` decodes sRGB to linear, so the value in the material is
    /// not the value that belongs in a `fill` attribute. An unlit basic material
    /// has to come back out exactly as it went in.
    #[test]
    fn base_colour_round_trips_through_srgb() {
        let svg = SvgRenderer::new(64, 64)
            .render_to_string(&mut plane_scene(), &camera_at(0.0, 0.0, 5.0));
        assert!(
            svg.contains("fill=\"#ff8800\""),
            "expected #ff8800, got: {}",
            &svg[..svg.len().min(400)]
        );
    }

    /// Painter's algorithm: the far quad has to be written first so the near one
    /// paints over it.
    #[test]
    fn faces_sort_back_to_front() {
        let mut scene = Scene::new();
        let mut far = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(2.0, 2.0),
            Material::Basic(BasicMaterial::new(Color::from_hex(0x0000ff))),
        ));
        far.position = Vector3::new(0.0, 0.0, -2.0);
        let mut near = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(2.0, 2.0),
            Material::Basic(BasicMaterial::new(Color::from_hex(0xff0000))),
        ));
        near.position = Vector3::new(0.0, 0.0, 2.0);
        // Added near-first, so traversal order alone would get this wrong.
        scene.add(near);
        scene.add(far);

        let svg =
            SvgRenderer::new(200, 200).render_to_string(&mut scene, &camera_at(0.0, 0.0, 6.0));
        let blue = svg.find("#0000ff").expect("far quad missing");
        let red = svg.find("#ff0000").expect("near quad missing");
        assert!(blue < red, "far quad must be emitted before the near one");
    }

    /// `render_order` is the caller overriding depth, and it has to win.
    #[test]
    fn render_order_beats_depth() {
        let mut scene = Scene::new();
        let mut far = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(2.0, 2.0),
            Material::Basic(BasicMaterial::new(Color::from_hex(0x0000ff))),
        ));
        far.position = Vector3::new(0.0, 0.0, -2.0);
        far.render_order = 5;
        let mut near = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(2.0, 2.0),
            Material::Basic(BasicMaterial::new(Color::from_hex(0xff0000))),
        ));
        near.position = Vector3::new(0.0, 0.0, 2.0);
        scene.add(far);
        scene.add(near);

        let svg =
            SvgRenderer::new(200, 200).render_to_string(&mut scene, &camera_at(0.0, 0.0, 6.0));
        let blue = svg.find("#0000ff").unwrap();
        let red = svg.find("#ff0000").unwrap();
        assert!(red < blue, "the higher render_order draws last");
    }

    /// A face straddling the camera plane is the classic way to get coordinates
    /// in the millions smeared across the viewport. Clipping has to keep every
    /// emitted number bounded.
    #[test]
    fn geometry_through_the_camera_plane_stays_bounded() {
        let mut scene = Scene::new();
        // A big plane the camera sits in the middle of, rotated to cut the
        // near plane at an angle.
        let mut obj = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(40.0, 40.0),
            Material::Basic(BasicMaterial::new(Color::WHITE)),
        ));
        obj.quaternion = crate::math::Quaternion::from_axis_angle(
            Vector3::new(1.0, 0.0, 0.0),
            std::f32::consts::FRAC_PI_3,
        );
        scene.add(obj);

        let svg =
            SvgRenderer::new(200, 200).render_to_string(&mut scene, &camera_at(0.0, 0.0, 0.5));
        let mut seen = 0;
        // Only the path data — a colour literal like `#111111` would otherwise
        // read as a coordinate.
        for coords in svg
            .split("d=\"")
            .skip(1)
            .filter_map(|s| s.split('"').next())
        {
            for tok in coords.split(['M', 'L', 'Z', ' ']) {
                if tok.is_empty() {
                    continue;
                }
                let v: f32 = tok
                    .parse()
                    .unwrap_or_else(|_| panic!("bad coordinate {tok:?}"));
                assert!(v.is_finite(), "non-finite coordinate {v}");
                assert!(
                    v.abs() < 100_000.0,
                    "runaway coordinate {v} — near clip failed"
                );
                seen += 1;
            }
        }
        assert!(seen > 0, "nothing was drawn");
    }

    /// The whole reason the shading code exists: a lit scene must not come out
    /// as one flat colour.
    #[test]
    fn lighting_varies_across_a_cube() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Lambert(LambertMaterial::new(Color::WHITE)),
        )));
        scene.add(Object3D::light(AmbientLight::new(Color::WHITE, 0.1)));
        let mut key = Object3D::light(DirectionalLight::new(Color::WHITE, 3.0));
        key.position = Vector3::new(4.0, 5.0, 3.0);
        scene.add(key);

        let svg =
            SvgRenderer::new(200, 200).render_to_string(&mut scene, &camera_at(3.0, 3.0, 3.0));
        let fills: std::collections::HashSet<&str> = svg
            .match_indices("fill=\"#")
            .map(|(i, _)| &svg[i + 7..i + 13])
            .collect();
        assert!(
            fills.len() >= 3,
            "three visible faces at three angles to the key light should give three tones, got {fills:?}"
        );
    }

    /// No lights is a legitimate scene, and rendering it black would be useless.
    #[test]
    fn unlit_scene_falls_back_to_flat_colour() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Lambert(LambertMaterial::new(Color::from_hex(0x3366cc))),
        )));
        let svg =
            SvgRenderer::new(200, 200).render_to_string(&mut scene, &camera_at(3.0, 3.0, 3.0));
        assert!(svg.contains("#3366cc"), "expected the flat base colour");
    }

    #[test]
    fn lines_and_points_render() {
        let mut scene = Scene::new();
        let mut seg = BufferGeometry::new();
        seg.set_attribute(
            "position",
            crate::core::BufferAttribute::new(vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 3),
        );
        scene.add(Object3D::line_segments(LineSegments::new(
            seg,
            Material::Line(LineBasicMaterial::new(Color::from_hex(0x00ff00))),
        )));

        let mut dots = BufferGeometry::new();
        dots.set_attribute(
            "position",
            crate::core::BufferAttribute::new(vec![0.0, 1.0, 0.0, 0.0, -1.0, 0.0], 3),
        );
        scene.add(Object3D::points(Points::new(
            dots,
            Material::Points(PointsMaterial::new(Color::from_hex(0xff00ff), 4.0)),
        )));

        let svg =
            SvgRenderer::new(200, 200).render_to_string(&mut scene, &camera_at(0.0, 0.0, 5.0));
        assert_eq!(count(&svg, "<line"), 1);
        assert_eq!(count(&svg, "<circle"), 2);
        assert!(svg.contains("#00ff00") && svg.contains("#ff00ff"));
    }

    #[test]
    fn background_is_emitted_and_suppressible() {
        let mut scene = plane_scene();
        scene.background = Color::from_hex(0x102030);
        let cam = camera_at(0.0, 0.0, 5.0);
        let on = SvgRenderer::new(64, 64).render_to_string(&mut scene, &cam);
        assert!(on.contains("<rect") && on.contains("#102030"));

        let off = SvgRenderer::new(64, 64)
            .with_options(SvgOptions {
                background: false,
                ..SvgOptions::default()
            })
            .render_to_string(&mut scene, &cam);
        assert!(!off.contains("<rect"));
    }

    #[test]
    fn wireframe_emits_strokes_and_no_fills() {
        let mut scene = plane_scene();
        let svg = SvgRenderer::new(64, 64)
            .with_options(SvgOptions {
                shading: SvgShading::Wireframe,
                ..SvgOptions::default()
            })
            .render_to_string(&mut scene, &camera_at(0.0, 0.0, 5.0));
        assert!(svg.contains("fill=\"none\""));
        assert_eq!(count(&svg, "<path"), 2);
    }

    #[test]
    fn empty_scene_is_a_valid_document() {
        let mut scene = Scene::new();
        let svg = SvgRenderer::new(10, 20).render_to_string(&mut scene, &camera_at(0.0, 0.0, 5.0));
        assert!(svg.starts_with("<svg "));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert!(svg.contains("width=\"10\"") && svg.contains("height=\"20\""));
    }

    /// Walk the document the way a viewer paints it and report the colour that
    /// ends up on top at `(x, y)` — the fill of the last `<path>` covering it.
    fn top_colour_at(svg: &str, x: f32, y: f32) -> Option<String> {
        let mut top = None;
        for chunk in svg.split("<path ").skip(1) {
            let Some(d) = chunk.split("d=\"").nth(1).and_then(|t| t.split('"').next()) else {
                continue;
            };
            let Some(fill) = chunk
                .split("fill=\"")
                .nth(1)
                .and_then(|t| t.split('"').next())
            else {
                continue;
            };
            if fill == "none" {
                continue;
            }
            let n: Vec<f32> = d
                .split(['M', 'L', 'Z', ' '])
                .filter(|t| !t.is_empty())
                .filter_map(|t| t.parse().ok())
                .collect();
            let poly: Vec<[f32; 2]> = n.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
            if poly.len() >= 3 && contains(&poly, x, y) {
                top = Some(fill.to_string());
            }
        }
        top
    }

    /// Ray casting along +x.
    fn contains(poly: &[[f32; 2]], x: f32, y: f32) -> bool {
        let mut inside = false;
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            if (a[1] > y) != (b[1] > y) {
                let t = (y - a[1]) / (b[1] - a[1]);
                if x < a[0] + (b[0] - a[0]) * t {
                    inside = !inside;
                }
            }
        }
        inside
    }

    /// The bug this renderer exists to not have: a two-triangle ground plane
    /// painting over the bottom of everything standing on it, slicing the
    /// objects off at the floor line.
    ///
    /// Two things caused it. The floor extends behind the camera, so the near
    /// clip leaves it with a vertex on the near plane — and under a hyperbolic
    /// NDC z that dragged its average depth to the front of the scene. Even
    /// with a linear depth key, a polygon spanning the whole scene has no
    /// single correct depth, so it has to be split.
    #[test]
    fn a_ground_plane_does_not_paint_over_what_stands_on_it() {
        const FLOOR: &str = "#6699cc";
        const BOX: &str = "#cc3311";

        let mut scene = Scene::new();
        // Big enough to run off behind the camera, and only two triangles.
        let mut floor = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(60.0, 60.0),
            Material::Basic(BasicMaterial::new(Color::from_hex(0x6699cc))),
        ));
        floor.quaternion = crate::math::Quaternion::from_axis_angle(
            Vector3::new(1.0, 0.0, 0.0),
            -std::f32::consts::FRAC_PI_2,
        );
        scene.add(floor);

        let mut cube = Object3D::mesh(Mesh::new(
            BoxGeometry::new(2.0, 2.0, 2.0),
            Material::Basic(BasicMaterial::new(Color::from_hex(0xcc3311))),
        ));
        cube.position = Vector3::new(0.0, 1.0, 0.0);
        scene.add(cube);

        let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 200.0);
        camera.position = Vector3::new(0.0, 3.0, 8.0);
        camera.target = Vector3::new(0.0, 1.0, 0.0);

        // A point on the cube's front face, a hair above the floor — dead in
        // the middle of the band the floor used to swallow.
        let vp = camera.projection_matrix().multiply(&camera.view_matrix());
        let c = to_clip(&vp, Vector3::new(0.0, 0.05, 1.0));
        let (sx, sy) = ((c[0] / c[3] + 1.0) * 200.0, (1.0 - c[1] / c[3]) * 200.0);

        let svg = SvgRenderer::new(400, 400).render_to_string(&mut scene, &camera);
        assert_eq!(
            top_colour_at(&svg, sx, sy).as_deref(),
            Some(BOX),
            "the floor is covering the base of the cube at ({sx:.0}, {sy:.0})"
        );

        // And the floor still wins where it should: open ground beside the cube.
        let c = to_clip(&vp, Vector3::new(3.0, 0.0, 0.0));
        let (fx, fy) = ((c[0] / c[3] + 1.0) * 200.0, (1.0 - c[1] / c[3]) * 200.0);
        assert_eq!(
            top_colour_at(&svg, fx, fy).as_deref(),
            Some(FLOOR),
            "floor in front of the cube should be floor"
        );
    }

    /// Curving is driven by the vertex normals, so a flat face — where they are
    /// constant — must stay straight. A cube that came out full of Béziers
    /// would be paying for curves it does not have.
    #[test]
    fn flat_faces_keep_straight_edges() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Basic(BasicMaterial::new(Color::WHITE)),
        )));
        let svg =
            SvgRenderer::new(200, 200).render_to_string(&mut scene, &camera_at(3.0, 3.0, 3.0));
        assert_eq!(count(&svg, "C"), 0, "a cube needs no curves");
    }

    /// A curved surface is the case curving exists for.
    #[test]
    fn curved_surfaces_get_curved_edges() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(1.0, 16, 10),
            Material::Basic(BasicMaterial::new(Color::WHITE)),
        )));
        let cam = camera_at(0.0, 0.0, 4.0);
        let svg = SvgRenderer::new(400, 400).render_to_string(&mut scene, &cam);
        assert!(
            count(&svg, "C") > 20,
            "expected curves, got {}",
            count(&svg, "C")
        );

        let off = SvgRenderer::new(400, 400)
            .with_options(SvgOptions {
                curve_tolerance: None,
                ..SvgOptions::default()
            })
            .render_to_string(&mut scene, &cam);
        assert_eq!(count(&off, "C"), 0, "curve_tolerance None must disable it");
    }

    /// No normals, nothing to curve to — and guessing from the face plane would
    /// just reproduce the straight edge anyway.
    #[test]
    fn geometry_without_normals_stays_straight() {
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute(
            "position",
            crate::core::BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(Mesh::new(
            geometry,
            Material::Basic(BasicMaterial::new(Color::WHITE)),
        )));
        let svg =
            SvgRenderer::new(200, 200).render_to_string(&mut scene, &camera_at(0.0, 0.0, 3.0));
        assert!(svg.contains("<path"), "the triangle should still draw");
        assert_eq!(count(&svg, "C"), 0);
    }

    /// The background belongs to where the drawing is going, not to what is in
    /// it, so the renderer can override the scene's without touching it.
    #[test]
    fn clear_colour_overrides_the_scene_background() {
        let mut scene = plane_scene();
        scene.background = Color::from_hex(0x102030);
        let cam = camera_at(0.0, 0.0, 5.0);

        let plain = SvgRenderer::new(64, 64).render_to_string(&mut scene, &cam);
        assert!(plain.contains("#102030"));

        let mut r = SvgRenderer::new(64, 64);
        r.set_clear_color(Some(Color::from_hex(0xfafafa)), None);
        let overridden = r.render_to_string(&mut scene, &cam);
        assert!(overridden.contains("#fafafa"), "override ignored");
        assert!(!overridden.contains("#102030"), "scene colour still there");
        assert_eq!(
            scene.background,
            Color::from_hex(0x102030),
            "the scene must come out unchanged"
        );

        // And back again.
        r.set_clear_color(None, None);
        assert!(r.render_to_string(&mut scene, &cam).contains("#102030"));
    }

    #[test]
    fn clear_alpha_overrides_the_scene_alpha() {
        let mut scene = plane_scene();
        scene.background_alpha = 1.0;
        let cam = camera_at(0.0, 0.0, 5.0);

        let mut r = SvgRenderer::new(64, 64);
        r.set_clear_color(None, Some(0.25));
        let svg = r.render_to_string(&mut scene, &cam);
        assert!(svg.contains("fill-opacity=\"0.25\""), "{svg}");

        // Zero means no rect at all rather than an invisible one.
        r.set_clear_color(None, Some(0.0));
        assert!(!r.render_to_string(&mut scene, &cam).contains("<rect"));
    }

    /// Resizing must not be a re-construction: the options set before it have
    /// to survive, which is what the JS shim's `setSize` depends on.
    #[test]
    fn set_size_keeps_the_options() {
        let mut r = SvgRenderer::new(64, 64).with_options(SvgOptions {
            shading: SvgShading::Flat,
            background: false,
            ..SvgOptions::default()
        });
        r.set_size(200, 120);
        assert_eq!((r.width, r.height), (200, 120));
        assert_eq!(r.options.shading, SvgShading::Flat);
        assert!(!r.options.background);

        let svg = r.render_to_string(&mut plane_scene(), &camera_at(0.0, 0.0, 5.0));
        assert!(svg.contains("width=\"200\"") && svg.contains("height=\"120\""));
        assert!(!svg.contains("<rect"), "background option was lost");
    }

    #[test]
    fn coordinates_are_trimmed() {
        assert_eq!(fmt(1.5, 2), "1.5");
        assert_eq!(fmt(3.0, 2), "3");
        assert_eq!(fmt(-0.001, 2), "0");
        assert_eq!(fmt(f32::NAN, 2), "0");
        assert_eq!(fmt(12.3456, 2), "12.35");
    }

    /// RFC 4648 section 10 test vectors.
    #[test]
    fn base64_matches_rfc4648() {
        for (input, want) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_encode(input.as_bytes()), want, "input {input:?}");
        }
    }

    #[test]
    fn rgba_passthrough_embeds_a_decodable_png() {
        let rgba: Vec<u8> = (0..4 * 4 * 4).map(|i| (i * 7 % 256) as u8).collect();
        let svg = svg_from_rgba(4, 4, &rgba);
        assert!(svg.contains("data:image/png;base64,"));
        assert!(svg.contains("width=\"4\""));

        // The embedded payload has to be a PNG that decodes back to the frame.
        let start = svg.find("base64,").unwrap() + "base64,".len();
        let end = svg[start..].find('"').unwrap() + start;
        let png = decode_base64_for_test(&svg[start..end]);
        let img = crate::utils::png::decode_png(&png).expect("embedded PNG did not decode");
        assert_eq!((img.width, img.height), (4, 4));
        assert_eq!(img.rgba, rgba);
    }

    fn decode_base64_for_test(s: &str) -> Vec<u8> {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut bits = 0u32;
        let mut n = 0;
        let mut out = Vec::new();
        for ch in s.bytes().filter(|b| *b != b'=') {
            let v = ALPHABET.iter().position(|a| *a == ch).expect("bad base64") as u32;
            bits = (bits << 6) | v;
            n += 6;
            if n >= 8 {
                n -= 8;
                out.push((bits >> n) as u8);
            }
        }
        out
    }
}
