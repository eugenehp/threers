//! Eager/lazy image parity and material-state behavior through the public API.
#![cfg(not(target_arch = "wasm32"))]
#[path = "support/material_state.rs"]
mod state;
use std::{sync::Arc, time::Instant};
use threers::{
    AmbientLight, BasicMaterial, BoxGeometry, Color, DirectionalLight, Mesh, Object3D,
    OrthographicCamera, PerspectiveCamera, RenderTarget, Renderer, Scene, StandardMaterial,
    Vector3,
};

#[test]
fn lazy_initialization_preserves_rendered_pixels_and_material_states() {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(desc);
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    eprintln!("[lazy-probe] adapter={:?}", adapter.get_info());
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let mut eager_images: Option<Vec<Vec<u8>>> = None;
    for mode in ["eager", "lazy"] {
        // Cover validation during both construction and first-use draws.
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let total = Instant::now();
        let mut images = Vec::new();
        let constructor = Instant::now();
        let make = if mode == "lazy" {
            Renderer::new_lazy
        } else {
            Renderer::new
        };
        let mut renderer = make(
            device.clone(),
            queue.clone(),
            wgpu::TextureFormat::Rgba8Unorm,
            64,
            64,
        );
        let constructor_ms = constructor.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[lazy-probe] mode={mode} constructor_ms={constructor_ms:.2}");
        renderer.set_reversed_z(false);
        renderer.set_reflective_transparency(false);
        let mut perspective = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        perspective.position = Vector3::new(2.5, 2.0, 3.5);
        perspective.look_at(Vector3::ZERO);
        let ortho = OrthographicCamera::new(-1.5, 1.5, 1.5, -1.5, 0.1, 100.0);
        let mut fixture_count = 0;
        for samples in [1, 4] {
            for lit in [false, true] {
                let mut scene = Scene::new();
                scene.background = Color::from_hex(0x182030);
                let material = if lit {
                    StandardMaterial::new(Color::from_hex(0xc08040))
                        .with_roughness(0.6)
                        .into()
                } else {
                    BasicMaterial::new(Color::from_hex(0xc08040)).into()
                };
                scene.add(Object3D::mesh(Mesh::new(
                    BoxGeometry::new(1.0, 1.0, 1.0),
                    material,
                )));
                scene.add_light(AmbientLight::new(Color::WHITE, 0.5));
                scene.add_light(
                    DirectionalLight::new(Color::WHITE, 1.0)
                        .with_direction(Vector3::new(-0.5, -1.0, -0.4)),
                );
                // A small warm target followed by a larger target catches size-dependent
                // first-use work instead of reporting constructor time alone.
                for size in [64, 256] {
                    let target =
                        RenderTarget::new(&device, size, size, wgpu::TextureFormat::Rgba8Unorm);
                    target.set_samples(samples);
                    target.set_presents(true);
                    for (camera_name, camera) in [
                        ("front", &ortho as &dyn threers::cameras::Camera),
                        ("orbit", &perspective as &dyn threers::cameras::Camera),
                    ] {
                        let name = format!("{samples}-{lit}-{size}-{camera_name}");
                        let frame = Instant::now();
                        renderer.render_to(&mut scene, camera, &target);
                        let rgba = threers::renderer::readback::texture_bytes(
                            &device,
                            &queue,
                            &target.color_texture,
                        );
                        eprintln!(
                            "[lazy-probe] mode={mode} fixture={name} complete_ms={:.2}",
                            frame.elapsed().as_secs_f64() * 1000.0
                        );
                        assert_eq!(rgba.len(), (size * size * 4) as usize);
                        let first = &rgba[..4];
                        assert!(
                            rgba.as_chunks::<4>().0.iter().any(|pixel| pixel != first),
                            "blank fixture {name}"
                        );
                        images.push(rgba.clone());
                        renderer.render_to(&mut scene, camera, &target);
                        let repeat = threers::renderer::readback::texture_bytes(
                            &device,
                            &queue,
                            &target.color_texture,
                        );
                        assert_eq!(rgba, repeat, "unstable repeated image {name}");
                        fixture_count += 1;
                    }
                }
            }
        }
        assert!(
            pollster::block_on(validation.pop()).is_none(),
            "wgpu validation error"
        );
        eprintln!(
        "[lazy-probe] mode={mode} fixtures={fixture_count} constructor_plus_all_frames_ms={:.2}",
        total.elapsed().as_secs_f64() * 1000.0
    );
        drop(renderer);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(fixture_count, 16);
        if let Some(previous) = eager_images.take() {
            for (index, (eager, lazy)) in previous.iter().zip(&images).enumerate() {
                assert!(
                    eager == lazy,
                    "eager and lazy pixels differ in fixture {index}"
                );
            }
        } else {
            eager_images = Some(images);
        }
    }

    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut state_cases = 0;
    for format in [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    ] {
        for samples in [1, 4] {
            let mut renderer =
                Renderer::new_lazy(device.clone(), queue.clone(), format, state::N, state::N);
            renderer.set_reversed_z(false);
            let target = RenderTarget::new(&device, state::N, state::N, format);
            target.set_samples(samples);
            target.set_presents(true);
            let failures = state::check(
                &format!("lazy {format:?} {samples}x"),
                &state::KINDS,
                &mut |scene, camera| {
                    renderer.render_to(scene, camera, &target);
                    state_cases += 1;
                    threers::renderer::readback::texture_bytes(
                        &device,
                        &queue,
                        &target.color_texture,
                    )
                },
            );
            assert!(failures.is_empty(), "{}", failures.join("\n"));
        }
    }
    assert_eq!(state_cases, 208);
    assert!(
        pollster::block_on(validation.pop()).is_none(),
        "wgpu validation error"
    );
}
