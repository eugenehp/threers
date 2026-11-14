//! The headset drawing on a real GPU, against the projection worked by
//! hand: a scripted [`XrSource`] (two eyes side by side in one
//! framebuffer, as every platform lays them out), a scene drawn per eye by
//! `Renderer::render_xr`, and a UI panel by `PanelPass`, alone and in the
//! scene.

#![cfg(not(target_arch = "wasm32"))]

use threers::renderer::readback::texture_bytes;
use threers::xr::{Panel, PanelPass, XrEvent, XrEye, XrInput, XrSource};
use threers::{
    BasicMaterial, Color, Matrix4, Mesh, Object3D, PerspectiveCamera, PlaneGeometry, Scene, Vector3,
};

const EYE: u32 = 200;

/// Two eyes, `apart` metres apart at head height, each a 90° square view in
/// its half of a 400 × 200 framebuffer.
struct Headset {
    apart: f32,
    see_through: bool,
}

fn gl_projection(n: f32, f: f32) -> [f32; 16] {
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -(f + n) / (f - n),
        -1.0,
        0.0,
        0.0,
        -2.0 * f * n / (f - n),
        0.0,
    ]
}

impl XrSource for Headset {
    fn head(&self) -> Option<[f32; 16]> {
        Some(Matrix4::translation(Vector3::new(0.0, 1.6, 0.0)).elements)
    }
    fn eyes(&self, origin: &Matrix4) -> Vec<XrEye> {
        [(-self.apart / 2.0, 0), (self.apart / 2.0, EYE as i32)]
            .into_iter()
            .map(|(x, at)| {
                XrEye::new(
                    gl_projection(0.05, 100.0),
                    origin,
                    &Matrix4::translation(Vector3::new(x, 1.6, 0.0)),
                    [at, 0, EYE as i32, EYE as i32],
                )
            })
            .collect()
    }
    fn framebuffer_size(&self) -> (u32, u32) {
        (2 * EYE, EYE)
    }
    fn inputs(&self) -> Vec<XrInput> {
        vec![]
    }
    fn take_events(&mut self) -> Vec<XrEvent> {
        vec![]
    }
    fn see_through(&self) -> bool {
        self.see_through
    }
}

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn target(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("xr test target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// `view` filled with `color` (the UI's texture, standing in for a UI).
fn fill(device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView, color: wgpu::Color) {
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    queue.submit(Some(encoder.finish()));
}

fn px(bytes: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = 4 * (y * 2 * EYE + x) as usize;
    [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
}

fn green(p: [u8; 4]) -> bool {
    p[1] > 200 && p[0] < 40 && p[2] < 40
}

fn red(p: [u8; 4]) -> bool {
    p[0] > 200 && p[1] < 40 && p[2] < 40
}

/// The panel's edges in an eye at the head's middle: it stands 1.3 m ahead,
/// 1.6 m by 1.0 m with its middle 0.15 m below the eyes. Under a 90° view
/// x = ±0.8 / 1.3 lands at 38.5 and 161.5 pixels; its top, 0.35 m above
/// the eyes, at 73.1 from the top, and its bottom, 0.65 m below, at 150.
#[test]
fn the_panel_lands_where_the_projection_puts_it() {
    let Some(gpu) = threers::gpu::testing::context("the XR panel") else {
        return;
    };
    let (device, queue) = (&*gpu.device, &*gpu.queue);
    let source = Headset {
        apart: 0.0,
        see_through: false,
    };
    let mut panel = Panel::new(1280.0, 800.0);
    panel.place_facing(&source.head().unwrap());
    let mut pass = PanelPass::new(device, FORMAT);
    let ui = pass.ui_target(device, (320, 200)).clone();
    fill(device, queue, &ui, wgpu::Color::GREEN);
    let out = target(device, 2 * EYE, EYE);
    pass.draw(
        device,
        queue,
        &out.create_view(&Default::default()),
        &source,
        &panel,
        &[],
    );
    let bytes = texture_bytes(device, queue, &out);
    for eye in [0, EYE] {
        // Inside, just inside each edge, and just outside.
        for (x, y) in [(100, 110), (41, 110), (159, 110), (100, 76), (100, 147)] {
            assert!(
                green(px(&bytes, eye + x, y)),
                "panel at ({x}, {y}) in the eye at {eye}: {:?}",
                px(&bytes, eye + x, y)
            );
        }
        for (x, y) in [(35, 110), (165, 110), (100, 70), (100, 153)] {
            assert!(
                !green(px(&bytes, eye + x, y)),
                "room at ({x}, {y}) in the eye at {eye}: {:?}",
                px(&bytes, eye + x, y)
            );
        }
        // The room: sky above, opaque.
        assert_eq!(px(&bytes, eye + 100, 5)[3], 255);
    }
    // See-through: nothing but the panel is drawn.
    let source = Headset {
        apart: 0.0,
        see_through: true,
    };
    pass.draw(
        device,
        queue,
        &out.create_view(&Default::default()),
        &source,
        &panel,
        &[],
    );
    let bytes = texture_bytes(device, queue, &out);
    assert_eq!(px(&bytes, 100, 5), [0, 0, 0, 0]);
    assert!(green(px(&bytes, 100, 110)));
}

/// A scene drawn per eye by `Renderer::render_xr`: a red square 0.4 m wide,
/// 3 m ahead at eye height, seen by eyes 0.6 m apart. Under a 90° view its
/// middle is at x = ±0.3 / 3 = ±0.1, so 110 pixels into the left eye and 90
/// into the right. Then a panel over it, the square showing beside it.
#[test]
fn a_scene_is_drawn_from_each_eye() {
    let Some(gpu) = threers::gpu::testing::context("XR stereo") else {
        return;
    };
    let (device, queue) = (&*gpu.device, &*gpu.queue);
    let mut renderer =
        threers::Renderer::new(gpu.device.clone(), gpu.queue.clone(), FORMAT, 2 * EYE, EYE);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let mut square = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(0.4, 0.4),
        BasicMaterial::new(Color::from_hex(0xff0000)).into(),
    ));
    square.position = Vector3::new(0.0, 1.6, -3.0);
    scene.add(square);
    let source = Headset {
        apart: 0.6,
        see_through: false,
    };
    let out = target(device, 2 * EYE, EYE);
    let view = out.create_view(&Default::default());
    let lens = PerspectiveCamera::new(50.0, 1.0, 0.05, 100.0);
    assert!(renderer.render_xr(&mut scene, &lens, &source, &Matrix4::identity(), &view));
    let bytes = texture_bytes(device, queue, &out);
    let centroid = |eye: u32| {
        let (mut sum, mut n) = (0.0, 0);
        for y in 0..EYE {
            for x in 0..EYE {
                if red(px(&bytes, eye + x, y)) {
                    sum += x as f32 + 0.5;
                    n += 1;
                }
            }
        }
        (sum / n.max(1) as f32, n)
    };
    let ((left, nl), (right, nr)) = (centroid(0), centroid(EYE));
    assert!(
        (left - 110.0).abs() < 0.6 && (right - 90.0).abs() < 0.6,
        "left {left} right {right}"
    );
    // 0.4 / 3 of the half-width each way: 13.3 pixels square.
    assert!((150..=220).contains(&nl) && nl == nr, "{nl} {nr} pixels");
    // A rig moved 1 m to the right moves the square 1 / 3 of a half-width left.
    renderer.render_xr(
        &mut scene,
        &lens,
        &source,
        &Matrix4::translation(Vector3::new(1.0, 0.0, 0.0)),
        &view,
    );
    let bytes = texture_bytes(device, queue, &out);
    let left = {
        let xs: Vec<f32> = (0..EYE)
            .flat_map(|y| (0..EYE).map(move |x| (x, y)))
            .filter(|&(x, y)| red(px(&bytes, x, y)))
            .map(|(x, _)| x as f32 + 0.5)
            .collect();
        xs.iter().sum::<f32>() / xs.len() as f32
    };
    assert!(
        (left - 76.67).abs() < 0.6,
        "left eye after the rig moved: {left}"
    );

    // A panel standing in the scene, the scene kept around it.
    let mut panel = Panel::new(400.0, 240.0);
    panel.place_facing(&source.head().unwrap());
    let mut pass = PanelPass::new(device, FORMAT);
    let ui = pass.ui_target(device, (100, 60)).clone();
    fill(device, queue, &ui, wgpu::Color::GREEN);
    pass.draw_in_scene(
        &mut renderer,
        &mut scene,
        &lens,
        &source,
        &Matrix4::identity(),
        &view,
        &panel,
        &[],
    );
    let bytes = texture_bytes(device, queue, &out);
    // The panel (0.5 m by 0.3 m, 1.3 m ahead) covers the square's place...
    assert!(green(px(&bytes, 110, 120)), "{:?}", px(&bytes, 110, 120));
    // ...and the scene's black stays where the panel is not, with no room drawn.
    assert_eq!(&px(&bytes, 100, 5)[..3], &[0, 0, 0]);
}

/// The panel inside a scene, depth-sorted: a red square 0.3 m in front of
/// the panel hides it (its middle, 1 m ahead and 0.15 m below the eyes, at
/// pixel (100, 115)); a blue one behind the panel is hidden by it (its
/// middle at x = 0.3 / 2 → pixel 115). With the renderer's MSAA, and with
/// its reversed depth, the same.
#[test]
fn the_panel_is_depth_sorted_with_the_scene() {
    let Some(gpu) = threers::gpu::testing::context("the XR panel in a scene") else {
        return;
    };
    let (device, queue) = (&*gpu.device, &*gpu.queue);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    for (hex, size, at) in [
        (0xff0000, 0.2, Vector3::new(0.0, 1.45, -1.0)),
        (0x0000ff, 0.4, Vector3::new(0.3, 1.45, -2.0)),
    ] {
        let mut o = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(size, size),
            BasicMaterial::new(Color::from_hex(hex)).into(),
        ));
        o.position = at;
        scene.add(o);
    }
    let source = Headset {
        apart: 0.0,
        see_through: false,
    };
    let mut panel = Panel::new(1280.0, 800.0);
    panel.place_facing(&source.head().unwrap());
    let lens = PerspectiveCamera::new(50.0, 1.0, 0.05, 100.0);
    let out = target(device, 2 * EYE, EYE);
    let view = out.create_view(&Default::default());
    let blue = |p: [u8; 4]| p[2] > 200 && p[0] < 40 && p[1] < 40;
    for (msaa, reversed) in [(1, false), (4, false), (1, true)] {
        let mut renderer =
            threers::Renderer::new(gpu.device.clone(), gpu.queue.clone(), FORMAT, 2 * EYE, EYE);
        renderer.set_msaa(msaa);
        renderer.set_reversed_z(reversed);
        let mut pass = PanelPass::new(device, FORMAT);
        let ui = pass.ui_target(device, (320, 200)).clone();
        fill(device, queue, &ui, wgpu::Color::GREEN);
        assert!(pass.draw_in_scene(
            &mut renderer,
            &mut scene,
            &lens,
            &source,
            &Matrix4::identity(),
            &view,
            &panel,
            &[]
        ));
        let bytes = texture_bytes(device, queue, &out);
        let case = format!("msaa {msaa}, reversed {reversed}");
        for eye in [0, EYE] {
            assert!(
                red(px(&bytes, eye + 100, 115)),
                "{case}: the square in front shows: {:?}",
                px(&bytes, eye + 100, 115)
            );
            assert!(
                green(px(&bytes, eye + 80, 115)),
                "{case}: the panel beside it: {:?}",
                px(&bytes, eye + 80, 115)
            );
            let behind = px(&bytes, eye + 115, 112);
            assert!(
                green(behind) && !blue(behind),
                "{case}: the square behind is hidden: {behind:?}"
            );
        }
    }
}

/// On an sRGB target (visionOS's compositor uses one) the UI's bytes come
/// out as the UI drew them: a mid grey of 128 stays 128, not brightened by
/// a second encoding.
#[test]
fn an_srgb_target_keeps_the_uis_bytes() {
    let Some(gpu) = threers::gpu::testing::context("the XR panel on sRGB") else {
        return;
    };
    let (device, queue) = (&*gpu.device, &*gpu.queue);
    let srgb = wgpu::TextureFormat::Rgba8UnormSrgb;
    let source = Headset {
        apart: 0.0,
        see_through: true,
    };
    let mut panel = Panel::new(1280.0, 800.0);
    panel.place_facing(&source.head().unwrap());
    let mut pass = PanelPass::new(device, srgb);
    assert_eq!(pass.ui_format(), wgpu::TextureFormat::Rgba8Unorm);
    let ui = pass.ui_target(device, (320, 200)).clone();
    let grey = 128.0 / 255.0;
    fill(
        device,
        queue,
        &ui,
        wgpu::Color {
            r: grey,
            g: grey,
            b: grey,
            a: 1.0,
        },
    );
    let out = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 2 * EYE,
            height: EYE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: srgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    pass.draw(
        device,
        queue,
        &out.create_view(&Default::default()),
        &source,
        &panel,
        &[],
    );
    let p = px(&texture_bytes(device, queue, &out), 100, 110);
    assert!(p[..3].iter().all(|&c| (127..=129).contains(&c)), "{p:?}");
}

/// A headset whose system highlights what is looked at (as Apple Vision
/// Pro's tracking areas): an integer texture the size of the framebuffer,
/// and a value for each id declared, up to one it refuses.
struct HoverHeadset {
    eyes: Headset,
    target: wgpu::Texture,
    declared: std::cell::RefCell<Vec<u64>>,
}

impl XrSource for HoverHeadset {
    fn head(&self) -> Option<[f32; 16]> {
        self.eyes.head()
    }
    fn eyes(&self, origin: &Matrix4) -> Vec<XrEye> {
        self.eyes.eyes(origin)
    }
    fn framebuffer_size(&self) -> (u32, u32) {
        self.eyes.framebuffer_size()
    }
    fn inputs(&self) -> Vec<XrInput> {
        vec![]
    }
    fn take_events(&mut self) -> Vec<XrEvent> {
        vec![]
    }
    fn see_through(&self) -> bool {
        false
    }
    fn hover_target(&self) -> Option<threers::xr::HoverTarget> {
        Some(threers::xr::HoverTarget {
            view: self.target.create_view(&Default::default()),
            format: self.target.format(),
            size: (self.target.width(), self.target.height()),
        })
    }
    fn declare_hover(&self, id: u64) -> Option<u32> {
        if id == 999 {
            return None;
        }
        let mut d = self.declared.borrow_mut();
        d.push(id);
        Some(d.len() as u32 * 100)
    }
}

/// Areas on the panel land in each eye where the panel's projection puts
/// them (see `the_panel_lands_where_the_projection_puts_it`: a UI pixel is
/// 123 / 1280 of an eye pixel across, 76.9 / 800 down, from 38.5 and 73.1),
/// the later over the earlier, each as its value; one the system refuses,
/// and everything off the areas, stays 0.
#[test]
fn hover_areas_are_drawn_as_their_values_where_they_show() {
    let Some(gpu) = threers::gpu::testing::context("the XR hover areas") else {
        return;
    };
    let (device, queue) = (&*gpu.device, &*gpu.queue);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hover areas"),
        size: wgpu::Extent3d {
            width: 2 * EYE,
            height: EYE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Uint,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let source = HoverHeadset {
        eyes: Headset {
            apart: 0.0,
            see_through: false,
        },
        target,
        declared: Default::default(),
    };
    let mut panel = Panel::new(1280.0, 800.0);
    panel.place_facing(&source.head().unwrap());
    let mut pass = PanelPass::new(device, FORMAT);
    use threers::xr::HoverArea;
    let areas = [
        HoverArea {
            id: 7,
            rect: [0.0, 0.0, 640.0, 400.0],
            radius: 0.0,
        },
        HoverArea {
            id: 999,
            rect: [700.0, 500.0, 400.0, 200.0],
            radius: 0.0,
        },
        HoverArea {
            id: 8,
            rect: [600.0, 380.0, 200.0, 100.0],
            radius: 50.0,
        },
    ];
    assert_eq!(
        pass.draw_hover_regions(device, queue, &source, &panel, &areas),
        2
    );
    assert_eq!(*source.declared.borrow(), [7, 8]);
    let bytes = texture_bytes(device, queue, &source.target);
    let at = |x: u32, y: u32| {
        u16::from_le_bytes([
            bytes[2 * (y * 2 * EYE + x) as usize],
            bytes[2 * (y * 2 * EYE + x) as usize + 1],
        ])
    };
    for eye in [0, EYE] {
        assert_eq!(at(eye + 60, 90), 100, "the first area, in the eye at {eye}");
        assert_eq!(at(eye + 105, 114), 200, "the second, over the first");
        // Its rounded corner (50 UI pixels: about 5 eye pixels) cut off.
        assert_eq!(at(eye + 97, 110), 100);
        assert_eq!(at(eye + 140, 140), 0, "refused");
        assert_eq!(at(eye + 20, 20), 0, "off the panel");
    }
}

/// A UI of black and white squares 4 UI pixels wide, seen small (the panel
/// 123 eye pixels across for 1280 UI pixels: a square is 0.4 of an eye
/// pixel): through its smaller copies every eye pixel is their mean grey,
/// where reading the whole texture picks black or white by where each
/// pixel happens to land.
#[test]
fn a_panel_seen_small_reads_its_smaller_copies() {
    let Some(gpu) = threers::gpu::testing::context("the XR panel's copies") else {
        return;
    };
    let (device, queue) = (&*gpu.device, &*gpu.queue);
    let source = Headset {
        apart: 0.0,
        see_through: false,
    };
    let mut panel = Panel::new(1280.0, 800.0);
    panel.place_facing(&source.head().unwrap());
    let mut pass = PanelPass::new(device, FORMAT);
    let (w, h) = (1280u32, 800u32);
    pass.ui_target(device, (w, h));
    let checks: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            if (i % w / 4 + i / w / 4) % 2 == 0 {
                [255u8; 4]
            } else {
                [0, 0, 0, 255]
            }
        })
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: pass.ui_texture().unwrap(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &checks,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    let out = target(device, 2 * EYE, EYE);
    pass.draw(
        device,
        queue,
        &out.create_view(&Default::default()),
        &source,
        &panel,
        &[],
    );
    let bytes = texture_bytes(device, queue, &out);
    let mut grey = 0;
    for y in 80..140 {
        for x in 50..150 {
            let p = px(&bytes, x, y);
            grey += usize::from(p[..3].iter().all(|&c| (96..=160).contains(&c)));
        }
    }
    assert!(grey > 6000 * 95 / 100, "{grey} of 6000 grey");
    // White drawn over it but said to be unchanged: the copies (grey)
    // stand; drawn again for real, they are made again (white).
    let white = vec![255u8; (w * h * 4) as usize];
    let upload = |pass: &PanelPass| {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: pass.ui_texture().unwrap(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &white,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        )
    };
    pass.ui_target(device, (w, h));
    upload(&pass);
    pass.ui_unchanged();
    pass.draw(
        device,
        queue,
        &out.create_view(&Default::default()),
        &source,
        &panel,
        &[],
    );
    assert!(
        px(&texture_bytes(device, queue, &out), 100, 110)[0] < 170,
        "the old copies kept"
    );
    pass.ui_target(device, (w, h));
    upload(&pass);
    pass.draw(
        device,
        queue,
        &out.create_view(&Default::default()),
        &source,
        &panel,
        &[],
    );
    assert!(
        px(&texture_bytes(device, queue, &out), 100, 110)[0] > 240,
        "made again"
    );
    // The scale the eyes need: 100 eye pixels per unit of tangent, the
    // panel 1.6 m wide at 1.3086 m, over 1280 UI pixels.
    let scale = PanelPass::ui_scale_for(&source, &panel).unwrap();
    assert!(
        (scale - 100.0 * 1.6 / (1.3f32.hypot(0.15)) / 1280.0).abs() < 1e-4,
        "{scale}"
    );
}
