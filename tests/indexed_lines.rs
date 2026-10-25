//! Indexed `LineSegments` must draw the segments the index names.
//!
//! A three.js `LineSegments` with an index draws `index[2k] → index[2k+1]`.
//! The atlas (a brain viewer on threers) draws neuron reconstructions that
//! way — axon and dendrites as two objects over one position buffer, each
//! with its own index — and saw a tangle of straight lines where three.js drew
//! the arbor: segments joined to the wrong vertices. These render indexed
//! lines and check the pixels: what the index joins is drawn, what it does
//! not join is not.
//!
//! Skipped when no GPU adapter is available.

use threers::{
    BufferAttribute, BufferGeometry, Color, HeadlessRenderer, LineBasicMaterial, LineSegments, Material, Object3D,
    OrthographicCamera, Scene, Vector3,
};

const W: u32 = 128;
const H: u32 = 128;

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder().size(W, H).color_format(wgpu::TextureFormat::Rgba8Unorm).build().ok()
}

fn camera() -> OrthographicCamera {
    // world x, y in [-1, 1] fill the frame
    let mut c = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    c.position = Vector3::new(0.0, 0.0, 5.0);
    c.look_at(Vector3::new(0.0, 0.0, 0.0));
    c
}

fn lines(pos: &[f32], index: Vec<u32>, opacity: f32) -> Object3D {
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos.to_vec(), 3));
    g.set_index(index);
    let mut m = LineBasicMaterial::default();
    m.color = Color::from_hex(0xffffff);
    m.opacity = opacity;
    let o = Object3D::line_segments(LineSegments::new(g, Material::Line(m)));
    o
}

/// is anything drawn within `r` pixels of world point (x, y)
fn lit(px: &[u8], x: f32, y: f32, r: i32) -> bool {
    let cx = ((x + 1.0) * 0.5 * W as f32) as i32;
    let cy = ((1.0 - (y + 1.0) * 0.5) * H as f32) as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            let (u, v) = (cx + dx, cy + dy);
            if u < 0 || v < 0 || u >= W as i32 || v >= H as i32 {
                continue;
            }
            let i = ((v as u32 * W + u as u32) * 4) as usize;
            if px[i] > 100 {
                return true;
            }
        }
    }
    false
}

fn scene_with(objs: Vec<Object3D>) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x000000);
    for o in objs {
        s.add(o);
    }
    s
}

#[test]
fn an_index_joins_the_vertices_it_names() {
    let Some(mut r) = renderer() else { return };
    // the corners of a square; the index joins the diagonals, not the sides
    let pos = [-0.8, -0.8, 0.0, 0.8, -0.8, 0.0, 0.8, 0.8, 0.0, -0.8, 0.8, 0.0];
    for opacity in [1.0, 0.95] {
        let px = r.render_to_rgba(&mut scene_with(vec![lines(&pos, vec![0, 2, 1, 3], opacity)]), &camera());
        assert!(lit(&px, 0.0, 0.0, 2), "opacity {opacity}: the diagonals cross at the centre");
        assert!(!lit(&px, 0.0, -0.8, 2), "opacity {opacity}: the bottom side is not in the index");
        assert!(!lit(&px, 0.8, 0.0, 2), "opacity {opacity}: the right side is not in the index");
    }
}

#[test]
fn a_shared_position_buffer_with_two_indices_and_many_vertices() {
    let Some(mut r) = renderer() else { return };
    // 100,000 points round a circle, each joined to the next: the first half
    // of the circle in one object, the second in another, over the same
    // positions (a neuron's axon and dendrites). Wrongly paired, segments
    // would cut across the circle's empty middle.
    let n = 100_000u32;
    let mut pos = Vec::with_capacity(n as usize * 3);
    for k in 0..n {
        let a = k as f32 / n as f32 * std::f32::consts::TAU;
        pos.extend_from_slice(&[0.7 * a.cos(), 0.7 * a.sin(), 0.0]);
    }
    let half = n / 2;
    let a: Vec<u32> = (0..half).flat_map(|k| [k, k + 1]).collect();
    let b: Vec<u32> = (half..n).flat_map(|k| [k, (k + 1) % n]).collect();
    for opacity in [1.0, 0.95] {
        let px = r.render_to_rgba(&mut scene_with(vec![lines(&pos, a.clone(), opacity), lines(&pos, b.clone(), opacity)]), &camera());
        assert!(lit(&px, 0.7, 0.0, 2) && lit(&px, -0.7, 0.0, 2), "opacity {opacity}: the circle is drawn");
        let inside = (0..20)
            .flat_map(|i| (0..20).map(move |j| (-0.4 + i as f32 * 0.04, -0.4 + j as f32 * 0.04)))
            .filter(|&(x, y)| lit(&px, x, y, 0))
            .count();
        assert_eq!(inside, 0, "opacity {opacity}: {inside} points inside the circle are drawn");
    }
}

#[test]
fn lines_filled_in_after_their_first_frame() {
    // as an embedder that mirrors another scene graph does it (the atlas's
    // presenter): the object is added with an empty geometry and drawn, then
    // its positions and index are set in place, a frame or more later, and a
    // second object over the same positions arrives meanwhile
    let Some(mut r) = renderer() else { return };
    let n = 60_000u32;
    let mut pos = Vec::with_capacity(n as usize * 3);
    for k in 0..n {
        let a = k as f32 / n as f32 * std::f32::consts::TAU;
        pos.extend_from_slice(&[0.7 * a.cos(), 0.7 * a.sin(), 0.0]);
    }
    let mut scene = scene_with(vec![]);
    let mut empty = BufferGeometry::new();
    empty.set_attribute("position", BufferAttribute::new(Vec::new(), 3));
    let mut m = LineBasicMaterial::default();
    m.color = Color::from_hex(0xffffff);
    m.opacity = 0.95;
    let mat = Material::Line(m);
    let ida = scene.add(Object3D::line_segments(LineSegments::new(empty.clone(), mat.clone())));
    let _ = r.render_to_rgba(&mut scene, &camera());
    let idb = scene.add(Object3D::line_segments(LineSegments::new(empty, mat)));
    let fill = |scene: &mut Scene, id, index: Vec<u32>| {
        if let Some(o) = scene.arena.get_mut(id) {
            if let threers::ObjectKind::LineSegments(l) = &mut o.kind {
                let g = std::sync::Arc::make_mut(&mut l.geometry);
                g.set_attribute("position", BufferAttribute::new(pos.clone(), 3));
                g.bump_version();
                g.set_index(index);
                g.bump_version();
            }
        }
    };
    let half = n / 2;
    fill(&mut scene, ida, (0..half).flat_map(|k| [k, k + 1]).collect());
    let _ = r.render_to_rgba(&mut scene, &camera());
    fill(&mut scene, idb, (half..n).flat_map(|k| [k, (k + 1) % n]).collect());
    for frame in 0..3 {
        let px = r.render_to_rgba(&mut scene, &camera());
        assert!(lit(&px, 0.7, 0.0, 2) && lit(&px, -0.7, 0.0, 2), "frame {frame}: the circle is drawn");
        let inside = (0..20)
            .flat_map(|i| (0..20).map(move |j| (-0.4 + i as f32 * 0.04, -0.4 + j as f32 * 0.04)))
            .filter(|&(x, y)| lit(&px, x, y, 0))
            .count();
        assert_eq!(inside, 0, "frame {frame}: {inside} points inside the circle are drawn");
    }
}

#[test]
fn as_an_embedder_draws_them_on_its_own_target() {
    // the atlas's render core: `Renderer::new` on an sRGB target, ACES tone
    // mapping, `render` into a view of its own texture, lines with
    // `LineBasicMaterial::new`, 0.95 opacity, depth tested
    let Some(gpu) = threers::gpu::testing::context("indexed lines (embedder)") else { return };
    let fmt = wgpu::TextureFormat::Bgra8UnormSrgb;
    let mut r = threers::renderer::Renderer::new(gpu.device.clone(), gpu.queue.clone(), fmt, W, H);
    r.set_tone_mapping(threers::ToneMapping::AcesFilmic, 1.0);
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let n = 60_000u32;
    let mut pos = Vec::with_capacity(n as usize * 3);
    for k in 0..n {
        let a = k as f32 / n as f32 * std::f32::consts::TAU;
        pos.extend_from_slice(&[0.7 * a.cos(), 0.7 * a.sin(), 0.0]);
    }
    let mut scene = scene_with(vec![]);
    let mut m = LineBasicMaterial::new(Color::new(1.0, 1.0, 1.0)).with_depth_test(true);
    m.opacity = 0.95;
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos.clone(), 3));
    g.set_index((0..n).flat_map(|k| [k, (k + 1) % n]).collect());
    scene.add(Object3D::line_segments(LineSegments::new(g, Material::Line(m))));
    for frame in 0..2 {
        r.render(&mut scene, &camera(), &view, false);
        // BGRA: the red channel is byte 2; `lit` reads byte 0 — the same for white
        let px = threers::renderer::readback::texture_bytes(&gpu.device, &gpu.queue, &tex);
        assert!(lit(&px, 0.7, 0.0, 2) && lit(&px, -0.7, 0.0, 2), "frame {frame}: the circle is drawn");
        let inside = (0..20)
            .flat_map(|i| (0..20).map(move |j| (-0.4 + i as f32 * 0.04, -0.4 + j as f32 * 0.04)))
            .filter(|&(x, y)| lit(&px, x, y, 0))
            .count();
        assert_eq!(inside, 0, "frame {frame}: {inside} points inside the circle are drawn");
    }
}
