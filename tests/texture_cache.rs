//! The wgpu renderer lets go of textures nothing draws any more.
//!
//! Each texture's GPU copy was cached by its id and kept for good, so a scene
//! that kept making textures — a video thumbnail strip, a generated map, a
//! texture per frame — grew without end, about a texture's worth a frame.

use std::sync::Arc;
use threers::prelude::*;
use threers::textures::{Texture, TextureFormat};
use threers::RenderBackend;

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(32, 32)
        .backend(RenderBackend::Wgpu)
        .build()
        .ok()
}

/// A scene with one textured cube, and the camera on it.
fn scene_with(texture: Arc<Texture>) -> (Scene, PerspectiveCamera, ObjectId) {
    let mut scene = Scene::new();
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let material = StandardMaterial::new(Color::WHITE).with_map(texture);
    let cube = scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        material.into(),
    )));
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 10.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    (scene, camera, cube)
}

fn texture(shade: u8) -> Arc<Texture> {
    Arc::new(Texture::new(
        4,
        4,
        TextureFormat::Rgba8UnormSrgb,
        vec![shade; 64],
    ))
}

/// Draw `frames` frames with a new texture on the cube each one.
fn churn(hr: &mut HeadlessRenderer, frames: usize) -> usize {
    let (mut scene, camera, mut cube) = scene_with(texture(0));
    for frame in 0..frames {
        scene.arena.delete(cube);
        let material = StandardMaterial::new(Color::WHITE).with_map(texture(frame as u8));
        cube = scene.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            material.into(),
        )));
        hr.render(&mut scene, &camera);
    }
    hr.renderer().cached_texture_count()
}

#[test]
fn textures_nothing_draws_are_let_go() {
    let Some(mut hr) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let held = churn(&mut hr, 300);
    // The ones drawn within the retention window, and no more.
    assert!(
        held <= threers::renderer::DEFAULT_CACHE_RETENTION as usize + 1,
        "{held} textures held after 300 frames of new ones"
    );
}

#[test]
fn retention_off_keeps_every_texture() {
    let Some(mut hr) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    hr.renderer().set_cache_retention(u64::MAX);
    assert!(churn(&mut hr, 200) >= 200, "the setting is honoured");
}

#[test]
fn a_texture_still_drawn_is_kept() {
    let Some(mut hr) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut scene, camera, _) = scene_with(texture(128));
    for _ in 0..400 {
        hr.render(&mut scene, &camera);
    }
    assert_eq!(hr.renderer().cached_texture_count(), 1);
}

#[test]
fn a_preloaded_texture_outlives_the_retention() {
    let Some(mut hr) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Preload, then draw only the data-less handle, after a long spell off
    // screen: there are no pixels to upload it again from.
    let full = texture(200);
    hr.renderer().preload_texture(&full);
    let handle = Arc::new(full.handle());
    drop(full);
    let (mut scene, camera, cube) = scene_with(texture(1));
    for _ in 0..300 {
        hr.render(&mut scene, &camera);
    }
    scene.arena.delete(cube);
    let material = StandardMaterial::new(Color::WHITE).with_map(handle.clone());
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        material.into(),
    )));
    let rgba = hr.render_to_rgba(&mut scene, &camera);
    let centre = (16 * 32 + 16) * 4;
    assert!(
        rgba[centre] > 100,
        "the handle still samples its pixels: {:?}",
        &rgba[centre..centre + 4]
    );
    hr.renderer().clear_texture_cache();
    assert_eq!(hr.renderer().cached_texture_count(), 0);
}
