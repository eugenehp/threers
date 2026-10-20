//! three.js `Points` with `PointsMaterial`, against three.js r165's
//! `points_vert` / `points_frag` rather than another backend.
//!
//! An orthographic camera looks down −z over a 64-pixel view of [−1, 1]²,
//! one pixel to 1/32. A point at the origin, black background:
//!
//! - **size**: `size` 16 covers 16 pixels — 7 off centre is lit, 9 is not;
//! - **attenuation**: a perspective camera 4 away, `size` 2 attenuated to
//!   `2 × (64 / 2) / 4 = 16` pixels (7 off centre lit); unattenuated it stays 2;
//! - **vertex colours** tint it: a red `color` attribute on a white material;
//! - **map** at three.js's `(gl_PointCoord.x, 1 − gl_PointCoord.y)`: a 2×2
//!   map whose top row (v = 1) is red | green and bottom row blue | yellow
//!   shows red top-left, green top-right, blue bottom-left, yellow
//!   bottom-right;
//! - **alphaMap** green 0 | 255 under `alphaTest` 0.5 cuts the left half;
//! - **transparent** at opacity 0.5 over black blends to 128, as three.js
//!   blends the encoded colours.
//!
//! Shared by `tests/points.rs` (wgpu) and `tests/metal_backend.rs`.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera, PerspectiveCamera};
use threers::core::{BufferAttribute, BufferGeometry, Object3D, Points};
use threers::materials::{Material, PointsMaterial};
use threers::math::{Color, Vector3};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFilter, TextureFormat};

pub const N: u32 = 64;

fn ortho() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    cam
}

fn scene(material: PointsMaterial, colour: Option<[f32; 3]>) -> Scene {
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(vec![0.0, 0.0, 0.0], 3));
    if let Some(c) = colour {
        g.set_attribute("color", BufferAttribute::new(c.to_vec(), 3));
    }
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::points(Points::new(g, Material::Points(material))));
    scene
}

fn tex(w: u32, h: u32, px: &[[u8; 4]]) -> Arc<Texture> {
    let mut t = Texture::new(w, h, TextureFormat::Rgba8Unorm, px.concat());
    t.mag_filter = TextureFilter::Nearest;
    t.min_filter = TextureFilter::Nearest;
    t.mipmap_filter = None;
    // A `DataTexture`: rows bottom first (`flipY` false).
    t.flip_y = false;
    Arc::new(t)
}

fn at(rgba: &[u8], x: i32, y: i32) -> [u8; 3] {
    let (x, y) = ((N as i32 / 2 + x) as u32, (N as i32 / 2 + y) as u32);
    let i = ((y * N + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}

/// Every failure, as a line; empty when each case draws as three.js.
pub fn check(label: &str, render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>) -> Vec<String> {
    let mut failures = Vec::new();
    let mut expect = |case: &str, got: [u8; 3], want: [u8; 3]| {
        if (0..3).any(|k| (got[k] as i32 - want[k] as i32).abs() > 2) {
            failures.push(format!("{label} {case}: got {got:?}, three.js gives {want:?}"));
        }
    };
    let white = [255, 255, 255];
    let black = [0, 0, 0];
    let point = |size: f32, attenuate: bool| {
        let mut m = PointsMaterial::new(Color::WHITE, size);
        m.size_attenuation = attenuate;
        m
    };

    let img = render(&mut scene(point(16.0, false), None), &ortho());
    expect("size 16, 7 px off centre", at(&img, 7, 0), white);
    expect("size 16, 9 px off centre", at(&img, 9, 0), black);

    let mut persp = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
    persp.position = Vector3::new(0.0, 0.0, 4.0);
    persp.look_at(Vector3::ZERO);
    let img = render(&mut scene(point(2.0, true), None), &persp);
    expect("attenuated size 2 at 4, 7 px off centre", at(&img, 7, 0), white);
    let img = render(&mut scene(point(2.0, false), None), &persp);
    expect("unattenuated size 2, 7 px off centre", at(&img, 7, 0), black);

    let img = render(&mut scene(point(16.0, false), Some([1.0, 0.0, 0.0])), &ortho());
    expect("vertex colour", at(&img, 0, 0), [255, 0, 0]);

    let mut mapped = point(16.0, false);
    // Row 0 is v = 0, the bottom.
    mapped.map = Some(tex(2, 2, &[[0, 0, 255, 255], [255, 255, 0, 255], [255, 0, 0, 255], [0, 255, 0, 255]]));
    let img = render(&mut scene(mapped, None), &ortho());
    expect("map, top-left", at(&img, -4, -4), [255, 0, 0]);
    expect("map, top-right", at(&img, 3, -4), [0, 255, 0]);
    expect("map, bottom-left", at(&img, -4, 3), [0, 0, 255]);
    expect("map, bottom-right", at(&img, 3, 3), [255, 255, 0]);

    let mut cut = point(16.0, false);
    cut.alpha_map = Some(tex(2, 1, &[[255, 0, 255, 255], [0, 255, 0, 255]]));
    cut.alpha_test = 0.5;
    let img = render(&mut scene(cut, None), &ortho());
    expect("alphaMap, left half cut", at(&img, -4, 0), black);
    expect("alphaMap, right half kept", at(&img, 3, 0), white);

    let mut glass = point(16.0, false);
    glass.opacity = 0.5;
    glass.transparent = Some(true);
    let img = render(&mut scene(glass, None), &ortho());
    expect("transparent at 0.5", at(&img, 0, 0), [128, 128, 128]);
    failures
}
