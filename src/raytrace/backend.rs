//! The seam between "what to trace" and "what traces it".
//!
//! [`RaytraceScene`] is already device-agnostic — a flat triangle soup, a
//! material table, a light table. A backend takes that plus a camera and adds
//! samples to a [`Film`]. The CPU implementation lives here; the GPU one in
//! [`super::gpu`] implements the same trait over the same scene, which is what
//! lets [`super::RaytraceRenderer`] switch between them without the calling
//! code changing.
//!
//! Backends accumulate rather than replace. `render` is told which sample index
//! to start at and how many to add, so a renderer can drive one backend in
//! small batches for interactivity and another in large ones for throughput.

use crate::core::Raycaster;
use crate::math::Vector2;
use crate::math::{Box3, Vector3};

use super::camera::RtCamera;
use super::film::{Film, Pixel};
use super::integrator::{Integrator, Scratch};
use super::sampler::Rng;
use super::scene::RaytraceScene;
use super::settings::RaytraceSettings;

/// Pixel rectangle for partial-frame tracing (CPU backend).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl RenderRect {
    pub fn full(film: &Film) -> Self {
        Self {
            x: 0,
            y: 0,
            width: film.width(),
            height: film.height(),
        }
    }

    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x
            && y >= self.y
            && x < self.x + self.width
            && y < self.y + self.height
    }

    /// Split a full frame into fixed-size tiles (last row/column may be smaller).
    pub fn tile_grid(film_w: u32, film_h: u32, tile_size: u32) -> impl Iterator<Item = Self> {
        let tile = tile_size.max(1);
        let mut rects = Vec::new();
        let mut y = 0u32;
        while y < film_h {
            let mut x = 0u32;
            let th = tile.min(film_h - y);
            while x < film_w {
                let tw = tile.min(film_w - x);
                rects.push(Self {
                    x,
                    y,
                    width: tw,
                    height: th,
                });
                x += tile;
            }
            y += tile;
        }
        rects.into_iter()
    }

    /// Tiles that still contain at least one pixel below the adaptive threshold.
    pub fn unconverged_tiles(
        film: &Film,
        threshold: f32,
        min_samples: u32,
        tile_size: u32,
    ) -> impl Iterator<Item = Self> + '_ {
        Self::tile_grid(film.width(), film.height(), tile_size).filter(move |tile| {
            tile.any_unconverged(film, threshold, min_samples)
        })
    }

    /// Whether any pixel in this rectangle has not converged.
    pub fn any_unconverged(&self, film: &Film, threshold: f32, min_samples: u32) -> bool {
        if threshold <= 0.0 {
            return true;
        }
        let fw = film.width();
        let x_end = self.x.saturating_add(self.width).min(fw);
        let y_end = self.y.saturating_add(self.height).min(film.height());
        for y in self.y..y_end {
            let row = y as usize * fw as usize;
            for x in self.x..x_end {
                if !film.pixels()[row + x as usize].is_converged(threshold, min_samples) {
                    return true;
                }
            }
        }
        false
    }

    /// Peak relative error among pixels in this rectangle.
    pub fn max_relative_error(&self, film: &Film) -> f32 {
        let fw = film.width();
        let x_end = self.x.saturating_add(self.width).min(fw);
        let y_end = self.y.saturating_add(self.height).min(film.height());
        let mut worst = 0.0f32;
        for y in self.y..y_end {
            let row = y as usize * fw as usize;
            for x in self.x..x_end {
                worst = worst.max(film.pixels()[row + x as usize].relative_error());
            }
        }
        worst
    }

    /// Tile size that yields roughly 32–128 tiles for a frame of this resolution.
    /// Rounded up to a multiple of 8 for GPU workgroups.
    pub fn recommended_tile_size(film_w: u32, film_h: u32) -> u32 {
        let pixels = (film_w as u64).saturating_mul(film_h as u64).max(1);
        let target_tiles = 64u64;
        let side = ((pixels / target_tiles) as f64).sqrt().ceil() as u32;
        side.clamp(8, 256).div_ceil(8) * 8
    }
}

fn trace_one(
    integrator: &Integrator<'_>,
    camera: &RtCamera,
    settings: &RaytraceSettings,
    film_w: u32,
    film_h: u32,
    x: u32,
    y: u32,
    pixel: &mut Pixel,
    scratch: &mut Scratch,
) {
    let index = pixel.samples;
    let mut rng = Rng::for_sample(x, y, index, settings.seed);
    rng.set_bounce(0);
    let jitter = rng.next_2d();
    let lens = rng.next_2d();
    let shutter = rng.next_f32();
    let (origin, dir) = camera.pixel_ray(x, y, film_w, film_h, jitter, lens, shutter);
    let r = integrator.trace(origin, dir, &mut rng, scratch);
    pixel.add(r.radiance, r.alpha);
    pixel.add_aux(r.albedo, r.normal, r.depth);
}

fn render_pixels_standard(
    integrator: &Integrator<'_>,
    camera: &RtCamera,
    settings: &RaytraceSettings,
    film_w: u32,
    film_h: u32,
    x0: u32,
    y0: u32,
    row_width: u32,
    pixels: &mut [Pixel],
    samples: u32,
) {
    let mut scratch = Scratch::default();
    let rw = row_width.max(1);
    for (i, pixel) in pixels.iter_mut().enumerate() {
        let x = x0 + (i as u32 % rw);
        let y = y0 + (i as u32 / rw);
        if x >= film_w || y >= film_h {
            break;
        }
        for _s in 0..samples {
            if pixel.is_converged(settings.adaptive_threshold, settings.adaptive_min_samples) {
                break;
            }
            trace_one(integrator, camera, settings, film_w, film_h, x, y, pixel, &mut scratch);
        }
    }
}

fn render_pixels_redistributed(
    integrator: &Integrator<'_>,
    camera: &RtCamera,
    settings: &RaytraceSettings,
    film_w: u32,
    film_h: u32,
    x0: u32,
    y0: u32,
    row_width: u32,
    pixels: &mut [Pixel],
    samples: u32,
    chunk_index: usize,
    batch_seed: u32,
) {
    let mut scratch = Scratch::default();
    let rw = row_width.max(1);
    let mut coords: Vec<(u32, u32)> = Vec::new();
    for (i, _) in pixels.iter().enumerate() {
        let x = x0 + (i as u32 % rw);
        let y = y0 + (i as u32 / rw);
        if x >= film_w || y >= film_h {
            break;
        }
        coords.push((x, y));
    }
    if coords.is_empty() {
        return;
    }
    let mut active: Vec<usize> = (0..coords.len()).collect();
    let mut per_pixel = vec![samples; coords.len()];
    let mut budget = coords.len() as u32 * samples;
    let mut pick = Rng::new(
        (chunk_index as u64).wrapping_add(batch_seed as u64),
        settings.seed.wrapping_add(0xA11_0C51),
    );
    while budget > 0 && !active.is_empty() {
        let slot = pick.next_index(active.len());
        let pi = active[slot];
        if per_pixel[pi] == 0 {
            active.swap_remove(slot);
            continue;
        }
        let (x, y) = coords[pi];
        let pixel = &mut pixels[pi];
        if pixel.is_converged(settings.adaptive_threshold, settings.adaptive_min_samples) {
            budget -= per_pixel[pi];
            per_pixel[pi] = 0;
            active.swap_remove(slot);
            continue;
        }
        trace_one(
            integrator,
            camera,
            settings,
            film_w,
            film_h,
            x,
            y,
            pixel,
            &mut scratch,
        );
        budget -= 1;
        per_pixel[pi] -= 1;
        if per_pixel[pi] == 0
            || pixel.is_converged(settings.adaptive_threshold, settings.adaptive_min_samples)
        {
            active.swap_remove(slot);
        }
    }
}

/// What can go wrong outside the integrator itself.
#[derive(Debug, Clone)]
pub enum RaytraceError {
    /// No adapter, device, or required limit for the GPU backend.
    NoDevice(String),
    /// The scene exceeded a device limit — buffer size, texture size, binding
    /// count. The message says which.
    TooLarge(String),
    /// The GPU produced no result within the timeout, or the device was lost.
    DeviceLost(String),
    /// [`super::RaytraceRenderer::accumulate`] was called before [`super::RaytraceRenderer::prepare`].
    NotPrepared,
}

impl std::fmt::Display for RaytraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice(m) => write!(f, "no usable device for the path tracer: {m}"),
            Self::TooLarge(m) => write!(f, "scene exceeds a device limit: {m}"),
            Self::DeviceLost(m) => write!(f, "device lost during tracing: {m}"),
            Self::NotPrepared => write!(f, "call prepare() before accumulate()"),
        }
    }
}

impl std::error::Error for RaytraceError {}

/// Something that can turn a [`RaytraceScene`] into samples on a [`Film`].
#[cfg(not(target_arch = "wasm32"))]
pub trait RaytraceBackend: Send {
    /// Short name, for logs and for [`super::RaytraceRenderer::backend_name`].
    fn name(&self) -> &'static str;

    /// Add `samples` samples per pixel to `film`.
    ///
    /// `first_sample` is the index the batch starts at; seeding from it rather
    /// than from a counter is what makes a render reproducible whatever batch
    /// sizes it was run in.
    fn render(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
    ) -> Result<(), RaytraceError>;

    /// Drop anything cached for a previous scene. Called when the geometry,
    /// materials or lights change.
    fn invalidate(&mut self) {}

    /// Device limits when this backend runs on the GPU.
    fn gpu_caps(&self) -> Option<super::gpu::GpuCaps> {
        None
    }

    /// Downcast hook for backend-specific APIs (e.g. region rendering on CPU).
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any
    where
        Self: 'static,
    {
        panic!("as_any_mut not implemented for {}", self.name())
    }
}

/// Something that can turn a [`RaytraceScene`] into samples on a [`Film`].
///
/// On `wasm32` the trait is not `Send` because browser WebGPU handles are
/// main-thread bound.
#[cfg(target_arch = "wasm32")]
pub trait RaytraceBackend {
    /// Short name, for logs and for [`super::RaytraceRenderer::backend_name`].
    fn name(&self) -> &'static str;

    /// Add `samples` samples per pixel to `film`.
    fn render(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
    ) -> Result<(), RaytraceError>;

    /// Drop anything cached for a previous scene.
    fn invalidate(&mut self) {}

    /// Device limits when this backend runs on the GPU.
    fn gpu_caps(&self) -> Option<super::gpu::GpuCaps> {
        None
    }

    /// Downcast hook for backend-specific APIs.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any
    where
        Self: 'static,
    {
        panic!("as_any_mut not implemented for {}", self.name())
    }
}

/// The reference backend: plain Rust, one path at a time.
///
/// It is the definition of correct for this module — the GPU kernel is checked
/// against it — and it is the only backend that works everywhere, including
/// `wasm32` and machines with no usable adapter.
#[derive(Debug, Default, Clone)]
pub struct CpuBackend {
    /// Rows per work unit. Larger units mean less scheduling overhead and worse
    /// load balance; a few rows is the usual compromise, because the cost of a
    /// row varies enormously with what is in front of it.
    rows_per_task: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self { rows_per_task: 4 }
    }

    /// Override the scheduling granularity.
    pub fn with_rows_per_task(mut self, rows: usize) -> Self {
        self.rows_per_task = rows.max(1);
        self
    }

    /// Trace only pixels inside `rect`. Other pixels are untouched.
    pub fn render_rect(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
        rect: RenderRect,
    ) -> Result<(), RaytraceError> {
        self.render_rect_impl(
            scene, camera, settings, film, first_sample, samples, rect,
        )?;
        film.advance(samples);
        Ok(())
    }

    /// Trace several disjoint regions, advancing the film budget once.
    pub fn render_regions(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
        regions: &[RenderRect],
    ) -> Result<(), RaytraceError> {
        if samples == 0 || regions.is_empty() {
            return Ok(());
        }
        for rect in regions {
            self.render_rect_impl(
                scene,
                camera,
                settings,
                film,
                first_sample,
                samples,
                *rect,
            )?;
        }
        film.advance(samples);
        Ok(())
    }

    /// Like [`Self::render_rect`], but does not bump the film's global sample counter.
    pub(crate) fn render_rect_impl(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
        rect: RenderRect,
    ) -> Result<(), RaytraceError> {
        if samples == 0 {
            return Ok(());
        }
        let integrator = Integrator::new(scene, settings);
        let (fw, fh) = (film.width(), film.height());
        let y_end = rect.y.saturating_add(rect.height).min(fh);
        let x_end = rect.x.saturating_add(rect.width).min(fw);
        if rect.x >= fw || rect.y >= fh || x_end <= rect.x || y_end <= rect.y {
            return Ok(());
        }
        let use_redist = settings.sample_redistribution && settings.adaptive_threshold > 0.0;
        let x0 = rect.x;
        let row_w = x_end - rect.x;
        let y0 = rect.y;
        let start = y0 as usize * fw as usize;
        let end = y_end as usize * fw as usize;
        let rows = &mut film.pixels_mut()[start..end];

        let paint_row = |row_i: usize, full_row: &mut [Pixel]| {
            let y = y0 + row_i as u32;
            let slice = &mut full_row[x0 as usize..x_end as usize];
            if use_redist {
                render_pixels_redistributed(
                    &integrator,
                    camera,
                    settings,
                    fw,
                    fh,
                    x0,
                    y,
                    row_w,
                    slice,
                    samples,
                    y as usize,
                    first_sample,
                );
            } else {
                render_pixels_standard(
                    &integrator,
                    camera,
                    settings,
                    fw,
                    fh,
                    x0,
                    y,
                    row_w,
                    slice,
                    samples,
                );
            }
        };

        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use rayon::prelude::*;
            rows.par_chunks_mut(fw as usize)
                .enumerate()
                .for_each(|(i, row)| paint_row(i, row));
        }
        #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
        {
            for (i, row) in rows.chunks_mut(fw as usize).enumerate() {
                paint_row(i, row);
            }
        }
        Ok(())
    }
}

impl RaytraceBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn render(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
    ) -> Result<(), RaytraceError> {
        if samples == 0 {
            return Ok(());
        }
        let integrator = Integrator::new(scene, settings);
        let (width, height) = (film.width(), film.height());
        let rows = self.rows_per_task.max(1);
        let chunk = width as usize * rows;
        let use_redist = settings.sample_redistribution && settings.adaptive_threshold > 0.0;

        let render_chunk = |chunk_index: usize, pixels: &mut [Pixel]| {
            let y0 = chunk_index * rows;
            if use_redist {
                render_pixels_redistributed(
                    &integrator,
                    camera,
                    settings,
                    width,
                    height,
                    0,
                    y0 as u32,
                    width,
                    pixels,
                    samples,
                    chunk_index,
                    first_sample,
                );
            } else {
                render_pixels_standard(
                    &integrator,
                    camera,
                    settings,
                    width,
                    height,
                    0,
                    y0 as u32,
                    width,
                    pixels,
                    samples,
                );
            }
        };

        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use rayon::prelude::*;
            film.pixels_mut()
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(ci, px)| render_chunk(ci, px));
        }
        #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
        {
            for (ci, px) in film.pixels_mut().chunks_mut(chunk).enumerate() {
                render_chunk(ci, px);
            }
        }

        film.advance(samples);
        Ok(())
    }
}

/// Intersect a world-space axis-aligned box along a ray.
///
/// Delegates to [`Raycaster::intersect_box`] so picking, cutaways, and the path
/// tracer share the same slab intersection as the raster [`Raycaster`].
pub fn intersect_box(
    origin: Vector3,
    direction: Vector3,
    near: f32,
    far: f32,
    bounds: &Box3,
) -> Option<f32> {
    let dir = if direction.length_sq() > 1e-12 {
        direction.normalize()
    } else {
        return None;
    };
    Raycaster::new(origin, dir, near, far).intersect_box(bounds)
}

/// Probe the scene along the centre of frame, for autofocus. Returns the
/// distance along the view axis to whatever is there, or `None` if nothing is.
pub fn probe_focus_distance(scene: &RaytraceScene, camera: &RtCamera) -> Option<f32> {
    let (origin, dir) = camera.ray(Vector2::new(0.0, 0.0), (0.5, 0.5));
    let hit = scene
        .bvh
        .intersect(&scene.tris, origin, dir, 1e-4, f32::INFINITY)?;
    // Project onto the view axis: the focal *plane* is flat, so a hit off to
    // one side is not at its own ray distance.
    let p = origin + dir * hit.t;
    let d = (p - camera.position()).dot(camera.forward());
    (d > 0.0).then_some(d)
}

/// Sum of a slice of radiance, for tests that need a quick "is it lit" check.
#[cfg(test)]
pub(crate) fn mean_luminance(hdr: &[f32]) -> f32 {
    use crate::math::Vector3;
    if hdr.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for px in hdr.chunks_exact(4) {
        sum += super::scene::luminance(Vector3::new(px[0], px[1], px[2])) as f64;
        n += 1;
    }
    (sum / n.max(1) as f64) as f32
}

/// Test scaffolding shared by both backends: a sky that is black except for a
/// small bright block, which is what separates a renderer that importance-samples
/// its environment from one that does not.
#[cfg(test)]
pub(crate) mod sun {
    use super::*;
    use crate::core::Object3D;
    use crate::geometries::PlaneGeometry;
    use crate::materials::{Material, StandardMaterial};
    use crate::math::{Color, Vector3};
    use crate::scene::Scene;

    /// A cube map that is black except for a small bright block on its +Z
    /// face — a stand-in for the sun in an HDRI, and the case cosine-weighted
    /// world sampling cannot resolve.
    pub(crate) fn sun_environment(
        face_size: u32,
        block: u32,
        power: u8,
    ) -> std::sync::Arc<crate::textures::CubeTexture> {
        let mut faces: [Vec<u8>; 6] = Default::default();
        for f in faces.iter_mut() {
            *f = vec![0u8; (face_size * face_size * 4) as usize];
            for p in f.chunks_exact_mut(4) {
                p[3] = 255;
            }
        }
        let lo = face_size / 2 - block / 2;
        for y in lo..lo + block {
            for x in lo..lo + block {
                let i = ((y * face_size + x) * 4) as usize;
                // Face 4 is +Z in CubeTexture order.
                faces[4][i] = power;
                faces[4][i + 1] = power;
                faces[4][i + 2] = power;
            }
        }
        let mut cube = crate::textures::CubeTexture::new(
            face_size,
            crate::textures::TextureFormat::Rgba8Unorm,
            faces,
        );
        // Nearest, so the block stays a block and the reference and the test
        // are looking at the same sky.
        cube.mag_filter = crate::textures::TextureFilter::Nearest;
        cube.min_filter = crate::textures::TextureFilter::Nearest;
        std::sync::Arc::new(cube)
    }

    pub(crate) fn sun_scene() -> Scene {
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        scene.environment = Some(sun_environment(64, 2, 255));
        let mut m = StandardMaterial::new(Color::new(0.8, 0.8, 0.8));
        m.roughness = 1.0;
        m.metalness = 0.0;
        // A plane at the origin facing +Z, lit by the block behind the camera.
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            PlaneGeometry::new(40.0, 40.0),
            Material::Standard(m),
        )));
        scene
    }

    /// A small area light over a glossy floor. Most of the variance is in the
    /// light and BSDF dimensions, which is where stratified pairs earn their
    /// keep.
    pub(crate) fn area_light_scene() -> Scene {
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        let mut floor_mat = StandardMaterial::new(Color::new(0.6, 0.55, 0.5));
        floor_mat.roughness = 0.45;
        let mut floor = Object3D::mesh(crate::core::Mesh::new(
            PlaneGeometry::new(30.0, 30.0),
            Material::Standard(floor_mat),
        ));
        floor.rotate_x(-std::f32::consts::FRAC_PI_2);
        scene.add(floor);

        let mut em = StandardMaterial::new(Color::BLACK);
        em.emissive = Color::WHITE;
        em.emissive_intensity = 30.0;
        let mut panel = Object3D::mesh(crate::core::Mesh::new(
            PlaneGeometry::new(1.2, 1.2),
            Material::Standard(em),
        ));
        panel.position = Vector3::new(0.0, 3.0, 0.0);
        panel.rotate_x(std::f32::consts::FRAC_PI_2);
        scene.add(panel);
        scene
    }

    /// Settings for [`area_light_scene`] at a given sample count.
    pub(crate) fn area_light_settings(samples: u32) -> RaytraceSettings {
        RaytraceSettings {
            samples_per_pixel: samples,
            max_bounces: 3,
            min_bounces: 3,
            clamp_indirect: 0.0,
            adaptive_threshold: 0.0,
            denoise: false,
            ..Default::default()
        }
    }

    /// The camera [`area_light_scene`] is framed for.
    pub(crate) fn area_light_camera() -> crate::cameras::PerspectiveCamera {
        let mut cam = crate::cameras::PerspectiveCamera::new(55.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 1.6, 4.0);
        cam.target = Vector3::new(0.0, 0.4, 0.0);
        cam
    }

    pub(crate) fn sun_settings(samples: u32) -> RaytraceSettings {
        RaytraceSettings {
            samples_per_pixel: samples,
            max_bounces: 1,
            min_bounces: 1,
            clamp_indirect: 0.0,
            environment_intensity: 60.0,
            denoise: false,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sun::*;
    use super::*;
    use crate::cameras::PerspectiveCamera;
    use crate::core::Object3D;
    use crate::geometries::{BoxGeometry, PlaneGeometry};
    use crate::lights::AmbientLight;
    use crate::materials::{Material, StandardMaterial};
    use crate::math::{Color, Vector3};
    use crate::scene::Scene;

    fn camera() -> PerspectiveCamera {
        let mut c = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        c.position = Vector3::new(0.0, 0.0, 4.0);
        c.target = Vector3::ZERO;
        c
    }

    #[test]
    fn an_empty_scene_renders_the_background() {
        let mut scene = Scene::new();
        scene.background = Color::new(0.25, 0.5, 0.75);
        let settings = RaytraceSettings::default().with_samples(4);
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);
        let mut film = Film::new(8, 8);
        CpuBackend::new()
            .render(&rt, &cam, &settings, &mut film, 0, 4)
            .unwrap();
        let hdr = film.resolve_hdr();
        assert!((hdr[0] - 0.25).abs() < 1e-5, "{:?}", &hdr[..4]);
        assert!((hdr[1] - 0.5).abs() < 1e-5);
        assert!((hdr[2] - 0.75).abs() < 1e-5);
        assert!((hdr[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_transparent_background_leaves_alpha_at_zero() {
        let mut scene = Scene::new();
        let settings = RaytraceSettings::default()
            .with_samples(2)
            .with_background(super::super::settings::BackgroundMode::Transparent);
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);
        let mut film = Film::new(4, 4);
        CpuBackend::new()
            .render(&rt, &cam, &settings, &mut film, 0, 2)
            .unwrap();
        let hdr = film.resolve_hdr();
        assert_eq!(hdr[3], 0.0);
    }

    /// A white Lambertian plane under uniform ambient radiance `L` must render
    /// at `albedo · L`, exactly. This is the furnace test: it catches a missing
    /// `1/π`, a missing cosine, and a wrong pdf all at once.
    #[test]
    fn furnace_test() {
        let mut scene = Scene::new();
        let mut m = StandardMaterial::new(Color::new(0.6, 0.6, 0.6));
        m.roughness = 1.0;
        m.metalness = 0.0;
        let mut plane = Object3D::mesh(crate::core::Mesh::new(
            PlaneGeometry::new(50.0, 50.0),
            Material::Standard(m),
        ));
        plane.position = Vector3::new(0.0, 0.0, 0.0);
        scene.add(plane);
        scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
        scene.background = Color::BLACK;

        let settings = RaytraceSettings {
            samples_per_pixel: 256,
            max_bounces: 1,
            min_bounces: 1,
            clamp_indirect: 0.0,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);
        let mut film = Film::new(16, 16);
        CpuBackend::new()
            .render(
                &rt,
                &cam,
                &settings,
                &mut film,
                0,
                settings.samples_per_pixel,
            )
            .unwrap();
        let hdr = film.resolve_hdr();
        // The centre pixel looks straight at the plane.
        let i = (8 * 16 + 8) * 4;
        assert!(
            (hdr[i] - 0.6).abs() < 0.02,
            "expected 0.6, got {} (albedo x ambient)",
            hdr[i]
        );
    }

    /// Reproducibility: the seed alone must determine the image, whatever the
    /// batch schedule was.
    #[test]
    fn batching_does_not_change_the_result() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.5, 1.5, 1.5),
            Material::Standard(StandardMaterial::new(Color::new(0.8, 0.3, 0.2))),
        )));
        scene.add_light(AmbientLight::new(Color::WHITE, 0.8));
        let settings = RaytraceSettings {
            samples_per_pixel: 8,
            max_bounces: 3,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);

        let mut one = Film::new(12, 12);
        CpuBackend::new()
            .render(&rt, &cam, &settings, &mut one, 0, 8)
            .unwrap();

        let mut split = Film::new(12, 12);
        let mut backend = CpuBackend::new().with_rows_per_task(1);
        backend
            .render(&rt, &cam, &settings, &mut split, 0, 3)
            .unwrap();
        backend
            .render(&rt, &cam, &settings, &mut split, 3, 5)
            .unwrap();

        let a = one.resolve_hdr();
        let b = split.resolve_hdr();
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-5, "pixel value {i}: {x} vs {y}");
        }
    }

    #[test]
    fn tile_grid_covers_the_film_once() {
        let tiles: Vec<_> = RenderRect::tile_grid(10, 10, 4).collect();
        assert_eq!(tiles.len(), 9);
        let mut covered = [false; 100];
        for t in tiles {
            for y in t.y..t.y + t.height {
                for x in t.x..t.x + t.width {
                    covered[(y * 10 + x) as usize] = true;
                }
            }
        }
        assert!(covered.iter().all(|&c| c));
    }

    #[test]
    fn unconverged_tiles_skip_fully_converged_regions() {
        let mut film = Film::new(4, 4);
        film.advance(16);
        // Top-left 2×2 tile: zero-variance white → converged.
        for y in 0..2 {
            for x in 0..2 {
                let p = &mut film.pixels_mut()[y as usize * 4 + x as usize];
                p.samples = 16;
                p.color = [16.0, 16.0, 16.0];
                p.lum_sq = 0.0; // Welford M₂ — zero spread
            }
        }
        let tiles: Vec<_> =
            RenderRect::unconverged_tiles(&film, 0.02, 2, 2).collect();
        assert_eq!(tiles.len(), 3);
        assert!(tiles.iter().all(|t| t.x >= 2 || t.y >= 2));
    }

    #[test]
    fn recommended_tile_size_scales_with_resolution() {
        assert_eq!(RenderRect::recommended_tile_size(64, 64), 8);
        let mid = RenderRect::recommended_tile_size(640, 480);
        assert!((8..=256).contains(&mid));
        assert_eq!(mid % 8, 0);
        let hi = RenderRect::recommended_tile_size(1920, 1080);
        assert!(hi >= mid);
        assert_eq!(hi % 8, 0);
    }

    #[test]
    fn converged_fraction_counts_quiet_pixels() {
        let mut film = Film::new(2, 2);
        film.advance(8);
        for p in film.pixels_mut().iter_mut().take(3) {
            p.samples = 8;
            p.color = [8.0, 8.0, 8.0];
            p.lum_sq = 0.0; // converged
        }
        film.pixels_mut()[3].samples = 8;
        film.pixels_mut()[3].color = [0.0, 16.0, 0.0];
        film.pixels_mut()[3].lum_sq = 112.0; // high M₂ → noisy
        assert!((film.converged_fraction(0.02, 2) - 0.75).abs() < 1e-5);
    }

    #[test]
    fn noisier_tiles_sort_ahead_of_quiet_ones() {
        let mut film = Film::new(4, 4);
        film.advance(8);
        for y in 0..2 {
            for x in 0..2 {
                let p = &mut film.pixels_mut()[y as usize * 4 + x as usize];
                p.samples = 8;
                p.color = [8.0, 8.0, 8.0];
                p.lum_sq = 0.0;
            }
        }
        film.pixels_mut()[7].samples = 8;
        film.pixels_mut()[7].color = [0.0, 16.0, 0.0];
        film.pixels_mut()[7].lum_sq = 112.0;
        let mut tiles: Vec<_> =
            RenderRect::unconverged_tiles(&film, 0.02, 2, 2).collect();
        tiles.sort_by(|a, b| {
            b.max_relative_error(&film)
                .partial_cmp(&a.max_relative_error(&film))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assert_eq!(tiles[0].x, 2);
        assert_eq!(tiles[0].y, 0);
    }

    #[test]
    fn render_rect_only_updates_the_requested_region() {
        let mut scene = Scene::new();
        scene.background = Color::new(0.2, 0.3, 0.4);
        let settings = RaytraceSettings {
            samples_per_pixel: 4,
            adaptive_threshold: 0.0,
            sample_redistribution: false,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);
        let mut film = Film::new(16, 16);
        let mut backend = CpuBackend::new();
        backend
            .render_rect(
                &rt,
                &cam,
                &settings,
                &mut film,
                0,
                4,
                RenderRect {
                    x: 2,
                    y: 3,
                    width: 4,
                    height: 2,
                },
            )
            .unwrap();
        let at = |row: usize, column: usize| row * 16 + column;
        let outside = at(0, 0);
        let inside = at(3, 2);
        assert_eq!(film.pixels()[outside].samples, 0);
        assert_eq!(film.pixels()[inside].samples, 4);
    }

    #[test]
    fn sample_redistribution_improves_convergence_on_mixed_scenes() {
        let build = || {
            let mut scene = Scene::new();
            scene.background = Color::new(0.05, 0.05, 0.06);
            scene.add(Object3D::mesh(crate::core::Mesh::new(
                PlaneGeometry::new(20.0, 20.0),
                Material::Standard(StandardMaterial::new(Color::new(0.75, 0.75, 0.75))),
            )));
            scene.add(Object3D::mesh(crate::core::Mesh::new(
                BoxGeometry::new(0.4, 0.4, 0.4),
                Material::Standard(StandardMaterial::new(Color::new(0.9, 0.1, 0.1))),
            )));
            scene.add_light(AmbientLight::new(Color::WHITE, 0.4));
            scene
        };
        let render = |redistribute: bool| {
            let settings = RaytraceSettings {
                samples_per_pixel: 64,
                max_bounces: 3,
                adaptive_threshold: 0.05,
                adaptive_min_samples: 4,
                sample_redistribution: redistribute,
                denoise: false,
                ..Default::default()
            };
            let mut scene = build();
            let rt = RaytraceScene::build(&mut scene, &settings);
            let rtc = RtCamera::new(&camera(), &settings);
            let mut film = Film::new(32, 32);
            CpuBackend::new()
                .render(&rt, &rtc, &settings, &mut film, 0, 64)
                .unwrap();
            let converged = film
                .pixels()
                .iter()
                .filter(|p| {
                    p.is_converged(settings.adaptive_threshold, settings.adaptive_min_samples)
                })
                .count();
            let counts = film.resolve_sample_counts();
            let taken: u64 = counts.iter().map(|&n| n as u64).sum();
            let budget = counts.len() as u64 * film.samples() as u64;
            let efficiency = taken as f32 / budget.max(1) as f32;
            (converged as f32 / film.pixels().len() as f32, efficiency)
        };
        let (conv_off, _) = render(false);
        let (conv_on, eff_on) = render(true);
        assert!(
            conv_on >= conv_off,
            "redistribution should converge at least as many pixels: {conv_on} vs {conv_off}"
        );
        assert!(
            eff_on <= 1.0 && eff_on > 0.0,
            "efficiency should stay in (0, 1]: {eff_on}"
        );
    }

    /// Adaptive sampling has to save work *and* land on the same image. A flat
    /// well-lit surface converges long before a budget of 256 is spent; the
    /// point is that it stops, and that stopping does not move the answer.
    #[test]
    fn adaptive_sampling_saves_samples_without_moving_the_image() {
        let build = || {
            let mut scene = Scene::new();
            scene.background = Color::new(0.1, 0.1, 0.12);
            scene.add(Object3D::mesh(crate::core::Mesh::new(
                PlaneGeometry::new(20.0, 20.0),
                Material::Standard(StandardMaterial::new(Color::new(0.7, 0.5, 0.3))),
            )));
            scene.add_light(AmbientLight::new(Color::WHITE, 0.9));
            scene
        };
        let render = |threshold: f32| {
            let settings = RaytraceSettings {
                samples_per_pixel: 256,
                max_bounces: 2,
                adaptive_threshold: threshold,
                adaptive_min_samples: 16,
                denoise: false,
                ..Default::default()
            };
            let mut scene = build();
            let rt = RaytraceScene::build(&mut scene, &settings);
            let rtc = RtCamera::new(&camera(), &settings);
            let mut film = Film::new(24, 24);
            CpuBackend::new()
                .render(&rt, &rtc, &settings, &mut film, 0, 256)
                .unwrap();
            film
        };

        let full = render(0.0);
        let adaptive = render(0.02);

        assert_eq!(
            full.sample_range(),
            (256, 256),
            "every pixel should take the budget"
        );
        let (lo, hi) = adaptive.sample_range();
        assert!(
            lo < 256,
            "adaptive sampling stopped nothing (range {lo}..{hi})"
        );
        let total: u64 = adaptive
            .resolve_sample_counts()
            .iter()
            .map(|&n| n as u64)
            .sum();
        let budget = 256u64 * 24 * 24;
        assert!(
            total < budget / 2,
            "adaptive took {total} of {budget} samples — barely a saving"
        );

        // And the picture is the same to within the noise it stopped at.
        let (a, b) = (full.resolve_hdr(), adaptive.resolve_hdr());
        let mean =
            |v: &[f32]| v.chunks_exact(4).map(|p| p[0] as f64).sum::<f64>() / (v.len() / 4) as f64;
        let (ma, mb) = (mean(&a), mean(&b));
        assert!(
            (ma - mb).abs() < 0.02 * ma,
            "adaptive changed the mean: {ma} -> {mb}"
        );
    }

    /// Adaptive or not, the result must not depend on how the samples were
    /// batched — the stopping rule reads only what the pixel has accumulated,
    /// which is the same at any batch boundary.
    #[test]
    fn adaptive_sampling_is_still_batch_independent() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(2.0, 2.0, 2.0),
            Material::Standard(StandardMaterial::new(Color::new(0.8, 0.4, 0.3))),
        )));
        scene.add_light(AmbientLight::new(Color::WHITE, 0.8));
        let settings = RaytraceSettings {
            samples_per_pixel: 64,
            max_bounces: 3,
            adaptive_threshold: 0.03,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let rtc = RtCamera::new(&camera(), &settings);

        let mut one = Film::new(16, 16);
        CpuBackend::new()
            .render(&rt, &rtc, &settings, &mut one, 0, 64)
            .unwrap();

        let mut split = Film::new(16, 16);
        let mut backend = CpuBackend::new();
        for start in [0u32, 7, 20, 41] {
            let end = match start {
                0 => 7,
                7 => 20,
                20 => 41,
                _ => 64,
            };
            backend
                .render(&rt, &rtc, &settings, &mut split, start, end - start)
                .unwrap();
        }

        assert_eq!(one.resolve_sample_counts(), split.resolve_sample_counts());
        for (i, (x, y)) in one
            .resolve_hdr()
            .iter()
            .zip(split.resolve_hdr().iter())
            .enumerate()
        {
            assert!((x - y).abs() < 1e-5, "channel {i}: {x} vs {y}");
        }
    }

    #[test]
    fn a_lit_box_is_brighter_than_a_black_scene() {
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(2.0, 2.0, 2.0),
            Material::Standard(StandardMaterial::new(Color::WHITE)),
        )));
        let mut light = Object3D::light(crate::lights::DirectionalLight::new(Color::WHITE, 3.0));
        light.position = Vector3::new(2.0, 3.0, 4.0);
        scene.add(light);

        let settings = RaytraceSettings {
            samples_per_pixel: 16,
            max_bounces: 2,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        assert!(rt.has_light());
        let cam = RtCamera::new(&camera(), &settings);
        let mut film = Film::new(16, 16);
        CpuBackend::new()
            .render(&rt, &cam, &settings, &mut film, 0, 16)
            .unwrap();
        assert!(
            mean_luminance(&film.resolve_hdr()) > 0.05,
            "the box should be lit"
        );
    }

    /// Nothing may produce a NaN or a negative radiance, however odd the scene.
    #[test]
    fn output_is_always_finite() {
        let mut scene = Scene::new();
        let mut glass = crate::materials::PhysicalMaterial::new(Color::WHITE);
        glass.transmission = 1.0;
        glass.roughness = 0.0;
        glass.ior = 1.5;
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            crate::geometries::SphereGeometry::new(1.0, 16, 12),
            Material::Physical(glass),
        )));
        let mut emitter = StandardMaterial::new(Color::BLACK);
        emitter.emissive = Color::WHITE;
        emitter.emissive_intensity = 50.0;
        let mut panel = Object3D::mesh(crate::core::Mesh::new(
            PlaneGeometry::new(1.0, 1.0),
            Material::Standard(emitter),
        ));
        panel.position = Vector3::new(0.0, 0.0, -3.0);
        scene.add(panel);

        let settings = RaytraceSettings {
            samples_per_pixel: 8,
            max_bounces: 8,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);
        let mut film = Film::new(12, 12);
        CpuBackend::new()
            .render(&rt, &cam, &settings, &mut film, 0, 8)
            .unwrap();
        for (i, v) in film.resolve_hdr().iter().enumerate() {
            assert!(v.is_finite() && *v >= 0.0, "channel {i} = {v}");
        }
    }

    /// A clear glass ball in front of a uniform emitter must let nearly all of
    /// it through: two Fresnel reflections cost a few percent and nothing else
    /// should.
    ///
    /// This is the regression test for the bug where the shading frame was
    /// flipped toward the ray, so the BSDF could not tell entering glass from
    /// leaving it and applied the `1/eta^2` radiance factor twice in the same
    /// direction — which cost a factor of `eta^4`, about 5x.
    #[test]
    fn clear_glass_passes_light_through() {
        let build = |with_ball: bool| {
            let mut scene = Scene::new();
            scene.background = Color::BLACK;
            let mut em = StandardMaterial::new(Color::BLACK);
            em.emissive = Color::WHITE;
            em.emissive_intensity = 4.0;
            let mut wall = Object3D::mesh(crate::core::Mesh::new(
                PlaneGeometry::new(20.0, 20.0),
                Material::Standard(em),
            ));
            wall.position = Vector3::new(0.0, 0.0, -4.0);
            scene.add(wall);
            if with_ball {
                let mut glass = crate::materials::PhysicalMaterial::new(Color::WHITE);
                glass.transmission = 1.0;
                glass.roughness = 0.0;
                glass.ior = 1.52;
                scene.add(Object3D::mesh(crate::core::Mesh::new(
                    crate::geometries::SphereGeometry::new(1.0, 48, 32),
                    Material::Physical(glass),
                )));
            }
            scene
        };

        let settings = RaytraceSettings {
            samples_per_pixel: 64,
            max_bounces: 12,
            min_bounces: 8,
            clamp_indirect: 0.0,
            denoise: false,
            ..Default::default()
        };
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 5.0);
        cam.target = Vector3::ZERO;

        let centre = |with_ball: bool| {
            let mut scene = build(with_ball);
            let rt = RaytraceScene::build(&mut scene, &settings);
            let rtc = RtCamera::new(&cam, &settings);
            let mut film = Film::new(32, 32);
            CpuBackend::new()
                .render(&rt, &rtc, &settings, &mut film, 0, 64)
                .unwrap();
            film.resolve_hdr()[(16 * 32 + 16) * 4]
        };

        let clear = centre(false);
        let through_glass = centre(true);
        assert!(
            (clear - 4.0).abs() < 0.01,
            "backdrop should read 4.0, got {clear}"
        );
        assert!(
            through_glass > 0.85 * clear,
            "glass passed only {through_glass} of {clear}"
        );
        assert!(
            through_glass <= clear * 1.02,
            "glass amplified {clear} to {through_glass}"
        );
    }

    /// The distribution has to actually concentrate on the bright block —
    /// otherwise everything below it is a no-op that happens to still be
    /// unbiased.
    #[test]
    fn the_environment_distribution_finds_the_sun() {
        let mut scene = sun_scene();
        let rt = RaytraceScene::build(&mut scene, &sun_settings(1));
        let dist = rt
            .world
            .env_distribution()
            .expect("a lit environment should produce a distribution");

        let sun = Vector3::new(0.0, 0.0, 1.0);
        let away = Vector3::new(0.0, -1.0, 0.0);
        let uniform = 1.0 / (4.0 * std::f32::consts::PI);
        let at_sun = dist.pdf(sun);
        assert!(
            at_sun > uniform * 100.0,
            "density at the sun is {at_sun}, barely above uniform {uniform}"
        );
        assert!(
            dist.pdf(away) < uniform,
            "the dark sky should be sampled less than uniformly"
        );
    }

    /// Adaptive must still stop quiet pixels under a bright area light — HDR
    /// means alone must not keep the whole frame open.
    #[test]
    fn adaptive_stops_on_hdr_area_light() {
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        let mut floor = Object3D::mesh(crate::core::Mesh::new(
            PlaneGeometry::new(20.0, 20.0),
            Material::Standard({
                let mut m = StandardMaterial::new(Color::new(0.55, 0.5, 0.45));
                m.roughness = 0.7;
                m
            }),
        ));
        floor.rotate_x(-std::f32::consts::FRAC_PI_2);
        scene.add(floor);
        let mut em = StandardMaterial::new(Color::BLACK);
        em.emissive = Color::WHITE;
        em.emissive_intensity = 40.0;
        let mut panel = Object3D::mesh(crate::core::Mesh::new(
            PlaneGeometry::new(1.5, 1.5),
            Material::Standard(em),
        ));
        panel.position = Vector3::new(0.0, 4.0, 0.0);
        panel.rotate_x(std::f32::consts::FRAC_PI_2);
        scene.add(panel);

        let settings = RaytraceSettings {
            samples_per_pixel: 128,
            max_bounces: 3,
            clamp_indirect: 0.0,
            adaptive_threshold: 0.02,
            adaptive_min_samples: 16,
            denoise: false,
            ..Default::default()
        };
        let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 1.5, 5.0);
        cam.target = Vector3::new(0.0, 0.5, 0.0);
        let rt = RaytraceScene::build(&mut scene, &settings);
        let rtc = RtCamera::new(&cam, &settings);
        let mut film = Film::new(32, 32);
        CpuBackend::new()
            .render(&rt, &rtc, &settings, &mut film, 0, 128)
            .unwrap();
        let (lo, hi) = film.sample_range();
        assert!(
            lo < 128,
            "HDR adaptive stopped nothing (range {lo}..{hi})"
        );
        let total: u64 = film
            .resolve_sample_counts()
            .iter()
            .map(|&n| n as u64)
            .sum();
        let budget = 128u64 * 32 * 32;
        assert!(
            total < (budget * 3) / 4,
            "HDR adaptive took {total}/{budget} — barely a saving"
        );
        // Mean must stay finite and lit.
        let hdr = film.resolve_hdr();
        let mean = hdr.chunks_exact(4).map(|p| p[0] as f64).sum::<f64>() / (hdr.len() / 4) as f64;
        assert!(mean.is_finite() && mean > 0.01, "mean {mean}");
    }

    /// End to end: a render with a handful of samples has to land close to a
    /// converged one. Cosine-weighted sampling finds this sun about once in
    /// eight hundred samples, so without importance sampling a 24-sample render
    /// is mostly black with the occasional white pixel — nowhere near the mean.
    #[test]
    fn a_sun_lit_scene_converges_in_a_few_samples() {
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 5.0);
        cam.target = Vector3::ZERO;

        let render = |samples: u32| {
            let settings = sun_settings(samples);
            let mut scene = sun_scene();
            let rt = RaytraceScene::build(&mut scene, &settings);
            let rtc = RtCamera::new(&cam, &settings);
            let mut film = Film::new(16, 16);
            CpuBackend::new()
                .render(&rt, &rtc, &settings, &mut film, 0, samples)
                .unwrap();
            film.resolve_hdr()
        };

        let reference = render(2048);
        let quick = render(24);
        let mean =
            |v: &[f32]| v.chunks_exact(4).map(|p| p[0] as f64).sum::<f64>() / (v.len() / 4) as f64;
        let (r, q) = (mean(&reference), mean(&quick));
        assert!(r > 0.02, "the reference render is not lit ({r})");
        assert!(
            (q - r).abs() < 0.1 * r,
            "24 samples gave {q}, converged is {r} — the sun is not being sampled"
        );

        // And the noise has to be low, not merely centred: a handful of huge
        // spikes averaging out to the right number is the failure this is for.
        let peak = quick
            .chunks_exact(4)
            .map(|p| p[0] as f64)
            .fold(0.0f64, f64::max);
        assert!(
            peak < 4.0 * r,
            "brightest pixel is {peak} against a mean of {r} — that is speckle"
        );
    }

    /// Stratified sampling has to show up as a *render* that converges faster,
    /// not just as a nicer point set. A small area light plus a glossy floor
    /// puts most of the variance in the light and BSDF dimensions, which is
    /// exactly what the Sobol pairs are for.
    #[test]
    fn a_few_samples_land_close_to_converged() {
        let render = |samples: u32| {
            let settings = area_light_settings(samples);
            let mut scene = area_light_scene();
            let rt = RaytraceScene::build(&mut scene, &settings);
            let rtc = RtCamera::new(&area_light_camera(), &settings);
            let mut film = Film::new(24, 24);
            CpuBackend::new()
                .render(&rt, &rtc, &settings, &mut film, 0, samples)
                .unwrap();
            film.resolve_hdr()
        };

        let reference = render(1024);
        let quick = render(16);
        let mut sse = 0.0f64;
        let mut n = 0usize;
        let mut mean = 0.0f64;
        for (a, b) in quick.chunks_exact(4).zip(reference.chunks_exact(4)) {
            for k in 0..3 {
                let d = (a[k] - b[k]) as f64;
                sse += d * d;
                mean += b[k] as f64;
                n += 1;
            }
        }
        let rms = (sse / n as f64).sqrt();
        let mean = mean / n as f64;
        assert!(mean > 0.05, "the scene is not lit ({mean})");
        // Independent draws land around 12% of the mean here; the stratified
        // pairs get to about 4%.
        // Independent draws land at 5.9% of the mean on this scene; the
        // stratified pairs get to 3.9%. The gap is wider on a scene with more
        // going on — three times, on one with an environment, a sun and an
        // emitter — but this one is cheap enough to run in a unit test.
        assert!(
            rms < 0.048 * mean,
            "16 samples are {:.1}% off a converged render — stratification is not working",
            100.0 * rms / mean
        );
    }

    /// Denoising a scene whose subject is a *mirror*.
    ///
    /// This is the case that separates a guided denoiser from a blur. The
    /// mirror's own albedo and normal describe the mirror, not the image in it,
    /// so guides taken at the first hit tell the filter that a whole sphere of
    /// wildly varying reflected content is one smooth surface. Following the
    /// path through the specular bounce puts the *reflected* albedo and normal
    /// in the guides instead, which is what lets the noise come out without the
    /// reflection going with it.
    #[test]
    fn denoising_a_mirror_beats_the_raw_estimate() {
        let build = || {
            let mut scene = Scene::new();
            scene.background = Color::BLACK;
            // A chequered floor, so there is real detail for the filter to lose.
            let n = 16u32;
            let mut px = vec![255u8; (n * n * 4) as usize];
            for y in 0..n {
                for x in 0..n {
                    let on = (x + y) % 2 == 0;
                    let i = ((y * n + x) * 4) as usize;
                    px[i] = if on { 235 } else { 30 };
                    px[i + 1] = if on { 220 } else { 40 };
                    px[i + 2] = if on { 200 } else { 60 };
                }
            }
            let mut tex = crate::textures::Texture::new(
                n,
                n,
                crate::textures::TextureFormat::Rgba8UnormSrgb,
                px,
            );
            tex.wrap_s = crate::textures::TextureWrap::Repeat;
            tex.wrap_t = crate::textures::TextureWrap::Repeat;
            tex.repeat = crate::math::Vector2::new(6.0, 6.0);
            tex.flip_y = false;
            let mut floor_mat = StandardMaterial::new(Color::WHITE);
            floor_mat.map = Some(std::sync::Arc::new(tex));
            floor_mat.roughness = 0.6;
            let mut floor = Object3D::mesh(crate::core::Mesh::new(
                PlaneGeometry::new(24.0, 24.0),
                Material::Standard(floor_mat),
            ));
            floor.rotate_x(-std::f32::consts::FRAC_PI_2);
            scene.add(floor);

            let mut mirror = StandardMaterial::new(Color::new(0.95, 0.95, 0.95));
            mirror.roughness = 0.02;
            mirror.metalness = 1.0;
            let mut ball = Object3D::mesh(crate::core::Mesh::new(
                crate::geometries::SphereGeometry::new(1.0, 48, 32),
                Material::Standard(mirror),
            ));
            ball.position = Vector3::new(0.0, 1.0, 0.0);
            scene.add(ball);

            let mut em = StandardMaterial::new(Color::BLACK);
            em.emissive = Color::WHITE;
            em.emissive_intensity = 25.0;
            let mut panel = Object3D::mesh(crate::core::Mesh::new(
                PlaneGeometry::new(2.0, 2.0),
                Material::Standard(em),
            ));
            panel.position = Vector3::new(0.0, 5.0, 0.0);
            panel.rotate_x(std::f32::consts::FRAC_PI_2);
            scene.add(panel);
            scene
        };
        let camera = || {
            let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
            cam.position = Vector3::new(0.0, 2.2, 6.0);
            cam.target = Vector3::new(0.0, 1.0, 0.0);
            cam
        };
        let render = |samples: u32, denoise: bool| {
            let settings = RaytraceSettings {
                samples_per_pixel: samples,
                max_bounces: 4,
                min_bounces: 4,
                clamp_indirect: 0.0,
                adaptive_threshold: 0.0,
                denoise,
                ..Default::default()
            };
            let mut scene = build();
            let mut r = super::super::RaytraceRenderer::new(48, 48);
            r.set_settings(settings);
            let cam = camera();
            r.render(&mut scene, &cam).unwrap();
            r.resolve_hdr()
        };

        let reference = render(1024, false);
        let raw = render(32, false);
        let denoised = render(32, true);

        // Measured over the ball alone. Averaged over the whole frame the floor
        // dominates, and the floor is a case any filter handles.
        let rms = |v: &[f32], ball_only: bool| {
            let mut sse = 0.0f64;
            let mut n = 0usize;
            for (i, (a, b)) in v.chunks_exact(4).zip(reference.chunks_exact(4)).enumerate() {
                let (x, y) = ((i % 48) as i32, (i / 48) as i32);
                let inside = (x - 24).pow(2) + (y - 22).pow(2) < 13 * 13;
                if ball_only != inside {
                    continue;
                }
                for k in 0..3 {
                    let d = (a[k] - b[k]) as f64;
                    sse += d * d;
                    n += 1;
                }
            }
            (sse / n.max(1) as f64).sqrt()
        };
        let (raw_err, denoised_err) = (rms(&raw, true), rms(&denoised, true));
        // Log-luminance colour weights trade a little mirror improvement for
        // stable HDR filtering. Retuned widths (see DenoiseParams::default)
        // keep this under ~0.88× of the raw RMS.
        assert!(
            denoised_err < 0.88 * raw_err,
            "denoising took RMS from {raw_err:.4} to {denoised_err:.4} — it is not helping"
        );
    }

    /// What the denoiser costs on content it cannot help with.
    ///
    /// This scene is a smooth falloff under a large area light: at 32 samples it
    /// is already nearly converged, and what difference remains between
    /// neighbouring pixels is *signal* of the same magnitude as the noise. No
    /// filter can tell those apart, so the honest guarantee is not "never
    /// worse" — it is "worse by a bounded, small amount", and the bound is what
    /// this pins down. The filter standing down on converged pixels is what
    /// keeps it small.
    ///
    /// `denoising_a_mirror_beats_the_raw_estimate` is the other half: on
    /// content that *is* noisy, it has to actually help.
    #[test]
    fn denoising_costs_little_where_it_cannot_help() {
        let render = |denoise: bool| {
            let mut settings = area_light_settings(32);
            settings.denoise = denoise;
            let mut scene = area_light_scene();
            let mut r = super::super::RaytraceRenderer::new(48, 48);
            r.set_settings(settings);
            r.render(&mut scene, &area_light_camera()).unwrap();
            r.resolve_hdr()
        };
        let reference = {
            let mut settings = area_light_settings(1024);
            settings.denoise = false;
            let mut scene = area_light_scene();
            let mut r = super::super::RaytraceRenderer::new(48, 48);
            r.set_settings(settings);
            r.render(&mut scene, &area_light_camera()).unwrap();
            r.resolve_hdr()
        };

        // The same relative metric the parameters were fitted against —
        // `mean((a - b)^2 / (b^2 + 0.01))`. Plain RMS in linear radiance is
        // dominated by the brightest pixels, and holding the filter to a
        // measure it was explicitly not tuned for would only test the
        // disagreement between the two.
        let rel_rms = |v: &[f32]| {
            let mut sum = 0.0f64;
            let mut n = 0usize;
            for (a, b) in v.chunks_exact(4).zip(reference.chunks_exact(4)) {
                for k in 0..3 {
                    let d = (a[k] - b[k]) as f64;
                    let r = b[k] as f64;
                    sum += d * d / (r * r + 0.01);
                    n += 1;
                }
            }
            (sum / n as f64).sqrt()
        };
        let raw = rel_rms(&render(false));
        let denoised = rel_rms(&render(true));
        assert!(
            denoised < 1.12 * raw,
            "denoising took relative RMS from {raw:.4} to {denoised:.4} on a nearly \
             converged scene — it should be standing down, not smearing"
        );
    }

    #[test]
    fn intersect_box_matches_raycaster_on_aabb() {
        let b = Box3::new(Vector3::new(-1.0, -1.0, -5.0), Vector3::new(1.0, 1.0, -3.0));
        let origin = Vector3::new(0.0, 0.0, 0.0);
        let dir = Vector3::new(0.0, 0.0, -1.0);
        let t = super::intersect_box(origin, dir, 0.0, 100.0, &b).expect("hit");
        assert!((t - 3.0).abs() < 1e-4, "entry t={t}");
        assert!(super::super::RaytraceRenderer::intersect_box(origin, dir, 0.0, 100.0, &b) == Some(t));
    }

    #[test]
    fn autofocus_finds_the_object_in_front_of_the_camera() {
        let mut scene = Scene::new();
        let mut obj = Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Standard(StandardMaterial::new(Color::WHITE)),
        ));
        obj.position = Vector3::new(0.0, 0.0, -6.0);
        scene.add(obj);
        let settings = RaytraceSettings::default();
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);
        let d = probe_focus_distance(&rt, &cam).expect("should have found the box");
        // Camera at z=4, box front face at z=-5.5.
        assert!((d - 9.5).abs() < 0.1, "focus distance {d}");
    }
}
