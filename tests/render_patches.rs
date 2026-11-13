//! Rendering part of a picture instead of all of it: a camera's view
//! offset (3D), and the UI pass's patches and regions. Each patch must be
//! exactly the pixels of the full picture it covers.
#![cfg(not(target_arch = "wasm32"))]

use threers::prelude::*;

/// The largest difference of any channel.
fn max_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0)
}

/// Rows `y0..y0+h`, columns `x0..x0+w` of a `width`-wide RGBA image.
fn crop(img: &[u8], width: usize, x0: usize, y0: usize, w: usize, h: usize) -> Vec<u8> {
    (y0..y0 + h).flat_map(|y| img[(y * width + x0) * 4..(y * width + x0 + w) * 4].to_vec()).collect()
}

fn scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x202830);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.4));
    let mut l = Object3D::light(DirectionalLight::new(Color::WHITE, 2.0).with_direction(Vector3::new(-0.4, -1.0, -0.5)));
    l.position = Vector3::new(2.0, 3.0, 2.0);
    scene.add(l);
    let mut cube = Object3D::mesh(Mesh::new(BoxGeometry::new(1.2, 1.2, 1.2), StandardMaterial::new(Color::from_hex(0xd9622b)).into()));
    cube.quaternion = Euler::new(0.5, 0.7, 0.0).to_quaternion();
    scene.add(cube);
    scene
}

/// Four tiles of a camera's view, each rendered on its own through a view
/// offset, are the full render.
#[test]
fn tiles_of_a_view_are_the_view() {
    let (w, h) = (96u32, 64u32);
    let Ok(mut full) = HeadlessRenderer::builder().size(w, h).build() else { return };
    let mut s = scene();
    let mut camera = PerspectiveCamera::new(50.0, w as f32 / h as f32, 0.1, 50.0);
    camera.position = Vector3::new(1.5, 1.2, 3.0);
    camera.look_at(Vector3::ZERO);
    let whole = full.render_to_rgba(&mut s, &camera);
    let (tw, th) = (w / 2, h / 2);
    let mut tile_r = HeadlessRenderer::builder().size(tw, th).build().unwrap();
    for (tx, ty) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        let mut c = camera.clone();
        c.set_view_offset(w as f32, h as f32, (tx * tw) as f32, (ty * th) as f32, tw as f32, th as f32);
        let tile = tile_r.render_to_rgba(&mut s, &c);
        let want = crop(&whole, w as usize, (tx * tw) as usize, (ty * th) as usize, tw as usize, th as usize);
        assert!(max_diff(&tile, &want) <= 2, "tile ({tx}, {ty}): {}", max_diff(&tile, &want));
    }
}

/// A camera's picture on a quad that fills the view is the picture, the
/// right way up: red above, blue below, as rendered directly.
#[test]
fn a_camera_texture_shows_the_right_way_up() {
    use threers::views::CameraTexture;
    let (w, h) = (64u32, 64u32);
    let Ok(mut hr) = HeadlessRenderer::builder().size(w, h).build() else { return };
    let mut s = Scene::new();
    s.background = Color::BLACK;
    for (c, y) in [(0xff0000u32, 0.5f32), (0x0000ff, -0.5)] {
        let mut m = BasicMaterial::new(Color::from_hex(c));
        m.tone_mapped = false;
        let mut p = Object3D::mesh(Mesh::new(PlaneGeometry::new(4.0, 1.0), m.into()));
        p.position = Vector3::new(0.0, y, 0.0);
        s.add(p);
    }
    let mut cam = PerspectiveCamera::new(60.0, 1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 1.5);
    cam.look_at(Vector3::ZERO);
    let direct = hr.render_to_rgba(&mut s, &cam);
    let mut tex = CameraTexture::new(hr.renderer(), w, h);
    tex.render(hr.renderer(), &mut s, &cam);
    let mut shown = Scene::new();
    let mut m = BasicMaterial::new(Color::WHITE);
    m.map = Some(tex.texture());
    shown.add(Object3D::mesh(Mesh::new(PlaneGeometry::new(2.0, 2.0), m.into())));
    let mut ortho = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.0, 10.0);
    ortho.position = Vector3::new(0.0, 0.0, 1.0);
    let seen = hr.render_to_rgba(&mut shown, &ortho);
    let px = |img: &[u8], y: usize| [img[(y * w as usize + 32) * 4], img[(y * w as usize + 32) * 4 + 1], img[(y * w as usize + 32) * 4 + 2]];
    assert!(px(&direct, 16)[0] > 200 && px(&direct, 48)[2] > 200, "direct: red above, blue below");
    assert!(px(&seen, 16)[0] > 150, "through the texture, red above: {:?} / {:?}", px(&seen, 16), px(&seen, 48));
    assert!(px(&seen, 48)[2] > 150, "and blue below: {:?}", px(&seen, 48));
}

#[cfg(feature = "ui")]
mod ui {
    use super::*;
    use threers::renderer::UiPass;
    use threers::style;
    use threers::ui::indicators::{indicator, IndicatorStyle, Shape};
    use threers::ui::{Rect, Ui};

    fn build() -> Ui {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, style! { display: flex; flex-wrap: wrap; gap: 12px; padding: 16px; background: #f1f3f6; });
        for i in 0..6 {
            let b = ui.add_box(root, style! { width: 90px; height: 50px; border-radius: 10px; color: #fff; font-size: 14px; display: flex; align-items: center; justify-content: center; });
            ui.update_style(b, |s| s.background = Some([0.1 + 0.12 * i as f32, 0.3, 0.8 - 0.1 * i as f32, 1.0]));
            ui.add_text(b, format!("Card {i}"));
        }
        // A box drawn through a shader, so a patch carries effects too.
        indicator(&mut ui, root, "Load", IndicatorStyle::new(Shape::Ring).size(70.0), Some(0.6));
        ui
    }

    /// A patch of the UI on its own is that patch of the whole, at 1x and
    /// at 2x.
    #[test]
    fn a_patch_is_that_part_of_the_whole() {
        let Some(gpu) = threers::gpu::testing::context("the UI pass") else { return };
        let (device, queue) = (&gpu.device, &gpu.queue);
        let mut pass = UiPass::new(device);
        let mut ui = build();
        let (w, h) = (330u32, 220u32);
        for scale in [1.0f32, 2.0] {
            ui.layout(w as f32, h as f32, &mut pass.text_measure(scale));
            let (dw, dh) = ((w as f32 * scale) as u32, (h as f32 * scale) as u32);
            let whole = pass.render_to_rgba(device, queue, &ui.scene(), scale, (dw, dh), [1.0; 4]);
            for region in [Rect { x: 110.0, y: 16.0, width: 100.0, height: 60.0 }, Rect { x: 0.0, y: 70.0, width: 330.0, height: 150.0 }] {
                let (size, patch) = pass.render_patch_to_rgba(device, queue, &ui.scene(), region, scale, [1.0; 4]);
                let want = crop(&whole, dw as usize, (region.x * scale) as usize, (region.y * scale) as usize, size.0 as usize, size.1 as usize);
                let d = max_diff(&patch, &want);
                assert!(d <= 3, "{region:?} at {scale}x: {d}");
            }
        }
    }

    /// Drawing only regions changes those, and nothing else.
    #[test]
    fn regions_change_only_themselves() {
        let Some(gpu) = threers::gpu::testing::context("the UI pass") else { return };
        let (device, queue) = (&gpu.device, &gpu.queue);
        let mut pass = UiPass::new(device);
        let (w, h) = (200u32, 120u32);
        let format = threers::wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture = device.create_texture(&threers::wgpu::TextureDescriptor {
            label: None,
            size: threers::wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: threers::wgpu::TextureDimension::D2,
            format,
            usage: threers::wgpu::TextureUsages::RENDER_ATTACHMENT | threers::wgpu::TextureUsages::COPY_SRC | threers::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let page = |css: &str| {
            let mut ui = Ui::new();
            let root = ui.root();
            ui.set_style(root, threers::ui::patch::Patch::css(css).into());
            ui.layout(w as f32, h as f32, &mut threers::ui::FixedWidthText(7.0));
            ui
        };
        let red = page("background: #ff0000");
        let blue = page("background: #0000ff");
        pass.draw(device, queue, &red.scene(), 1.0, &view, (w, h), format);
        let region = Rect { x: 40.0, y: 30.0, width: 50.0, height: 20.0 };
        pass.draw_regions(device, queue, &blue.scene(), 1.0, &view, (w, h), format, &[region]);
        let img = threers::renderer::readback::texture_bytes(device, queue, &texture);
        let at = |x: usize, y: usize| [img[(y * w as usize + x) * 4], img[(y * w as usize + x) * 4 + 1], img[(y * w as usize + x) * 4 + 2]];
        assert_eq!(at(60, 40), [0, 0, 255], "inside the region: the new scene");
        assert_eq!(at(10, 10), [255, 0, 0], "outside: as it was");
        assert_eq!(at(150, 100), [255, 0, 0]);
    }
}
