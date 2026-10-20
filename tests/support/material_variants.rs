//! Every texture keeps its own sampler state and uv transform, as three.js
//! gives each one its own `magFilter`, `minFilter`, `wrapS`, `wrapT` and
//! `matrix`. Shared by `tests/material_variants.rs` (wgpu, where a material
//! whose maps disagree gets a shader variant from `renderer::material_maps`)
//! and `tests/metal_backend.rs`.
//!
//! A white Lambert quad fills an orthographic view under ambient light at
//! intensity π, which shades it to exactly its map (`BRDF_Lambert` is
//! albedo / π); a pixel's x is its u.
//!
//! - **Transforms**: the colour map is red | blue, left to right; the emissive
//!   map is black | white, mirrored by its own transform (`repeat.x = −1`,
//!   `offset.x = 1`, three.js's `setUvTransform`), so it reads white on the
//!   left and black on the right. Each map keeping its own transform gives
//!   white on the left (red + white, clamped) and blue on the right; one
//!   transform shared between them — the colour map's — would give red, then
//!   white.
//! - **Wrapping**: a red | green map with `repeat.x = 2` reads red, green,
//!   green, red across the quad under `MirroredRepeatWrapping`; red, green,
//!   red, green under `RepeatWrapping`; and red then green held under
//!   `ClampToEdgeWrapping`.
//! - **Mipmaps**: a one-texel checkerboard minified 8× reads as its average
//!   (188) mipmapped, and as single texels (0 or 255) under `NearestFilter`,
//!   which takes no mipmaps;
//! - **Rotation**: a red | green map turned half a turn about its `center`
//!   (½, ½) reads green, then red.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::lights::AmbientLight;
use threers::materials::LambertMaterial;
use threers::math::{Color, Vector2, Vector3};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFilter, TextureFormat, TextureWrap};

pub const N: u32 = 64;

fn strip(left: [u8; 4], right: [u8; 4]) -> Texture {
    let mut t = Texture::new(2, 1, TextureFormat::Rgba8Unorm, [left, right].concat());
    t.mag_filter = TextureFilter::Nearest;
    t.min_filter = TextureFilter::Nearest;
    t
}

fn scene(mat: LambertMaterial) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add_light(AmbientLight::new(Color::WHITE, std::f32::consts::PI));
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0, 2.0),
        mat.into(),
    )));
    scene
}

/// Every failure, as a line; empty when each texture keeps its own state.
pub fn check(
    label: &str,
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    let mut failures = Vec::new();
    // The middle row at u = k/8.
    let at = |rgba: &[u8], k: u32| {
        let i = ((N / 2 * N + N * k / 8) * 4) as usize;
        [rgba[i], rgba[i + 1], rgba[i + 2]]
    };

    let mut emissive = strip([0, 0, 0, 255], [255, 255, 255, 255]);
    emissive.repeat = Vector2::new(-1.0, 1.0);
    emissive.offset = Vector2::new(1.0, 0.0);
    let mut mat = LambertMaterial::new(Color::WHITE);
    mat.map = Some(Arc::new(strip([255, 0, 0, 255], [0, 0, 255, 255])));
    mat.emissive = Color::WHITE;
    mat.emissive_map = Some(Arc::new(emissive));
    let rgba = render(&mut scene(mat), &cam);
    let got = [at(&rgba, 2), at(&rgba, 6)];
    let want = [[255, 255, 255], [0, 0, 255]];
    if got != want {
        failures.push(format!(
            "{label} per-map transforms: got {got:?}, three.js gives {want:?}"
        ));
    }

    for (wrap, want) in [
        (TextureWrap::MirroredRepeat, [0, 255, 255, 0]),
        (TextureWrap::Repeat, [0, 255, 0, 255]),
        (TextureWrap::ClampToEdge, [0, 255, 255, 255]),
    ] {
        let mut map = strip([255, 0, 0, 255], [0, 255, 0, 255]);
        map.wrap_s = wrap;
        map.wrap_t = wrap;
        map.repeat = Vector2::new(2.0, 1.0);
        let mut mat = LambertMaterial::new(Color::WHITE);
        mat.map = Some(Arc::new(map));
        let rgba = render(&mut scene(mat), &cam);
        let greens = [1, 3, 5, 7].map(|k| at(&rgba, k)[1]);
        if greens != want {
            failures.push(format!("{label} {wrap:?}: green at u = 1/8, 3/8, 5/8, 7/8 is {greens:?}, three.js gives {want:?}"));
        }
    }
    // Minification 8×: a 64×64 one-texel checkerboard on an 8-pixel-wide
    // quad. Mipmapped (three.js's default `LinearMipmapLinearFilter`) every
    // pixel is the average, linear ½ — 188 encoded; `NearestFilter` takes no
    // mipmaps and every pixel is one texel, black or white.
    for (mip, want_grey) in [(Some(TextureFilter::Linear), true), (None, false)] {
        let mut board = Texture::new(
            64,
            64,
            TextureFormat::Rgba8Unorm,
            (0..64 * 64).flat_map(|i| {
                let v = if (i % 64 + i / 64) % 2 == 0 { 255 } else { 0 };
                [v, v, v, 255]
            }).collect(),
        );
        board.mag_filter = TextureFilter::Nearest;
        board.min_filter = if mip.is_some() { TextureFilter::Linear } else { TextureFilter::Nearest };
        board.mipmap_filter = mip;
        let mut mat = threers::materials::BasicMaterial::new(Color::WHITE);
        mat.map = Some(Arc::new(board));
        let mut sc = Scene::new();
        sc.background = Color::from_hex(0x000000);
        sc.add(Object3D::mesh(Mesh::new(PlaneGeometry::new(0.25, 0.25), mat.into())));
        let rgba = render(&mut sc, &cam);
        let px: Vec<u8> = (N / 2 - 2..N / 2 + 2).map(|x| rgba[((N / 2 * N + x) * 4) as usize]).collect();
        let ok = if want_grey {
            px.iter().all(|v| (*v as i32 - 188).abs() <= 6)
        } else {
            px.iter().all(|v| *v == 0 || *v == 255)
        };
        if !ok {
            failures.push(format!(
                "{label} mipmap_filter {mip:?}: centre row {px:?}; three.js gives {}",
                if want_grey { "the average, 188" } else { "single texels, 0 or 255" }
            ));
        }
    }

    let mut turned = strip([255, 0, 0, 255], [0, 255, 0, 255]);
    turned.rotation = std::f32::consts::PI;
    turned.center = Vector2::new(0.5, 0.5);
    let mut mat = LambertMaterial::new(Color::WHITE);
    mat.map = Some(Arc::new(turned));
    let rgba = render(&mut scene(mat), &cam);
    let greens = [at(&rgba, 2)[1], at(&rgba, 6)[1]];
    if greens != [255, 0] {
        failures.push(format!("{label} rotation about centre: green at u = ¼, ¾ is {greens:?}, three.js gives [255, 0]"));
    }

    // `Texture.channel`: the quad's `uv1`, `uv2` and `uv3` are its `uv`
    // mirrored left to right, so a red | green map read through any of them
    // reads green, then red — and through `uv` red, then green. A geometry
    // without the set reads zero there, as three.js's missing attribute does:
    // the map's left texel everywhere.
    for channel in 0..4u32 {
        for with_sets in [true, false] {
            let mut map = strip([255, 0, 0, 255], [0, 255, 0, 255]);
            map.channel = channel;
            let mut mat = LambertMaterial::new(Color::WHITE);
            mat.map = Some(Arc::new(map));
            let mut geometry = PlaneGeometry::new(2.0, 2.0);
            if with_sets {
                let uv = geometry.get_attribute("uv").expect("a plane has uvs").clone();
                let mirrored: Vec<f32> = uv.array.chunks(2).flat_map(|c| [1.0 - c[0], c[1]]).collect();
                for name in ["uv1", "uv2", "uv3"] {
                    geometry.set_attribute(name, threers::core::BufferAttribute::new(mirrored.clone(), 2));
                }
            }
            let mut sc = Scene::new();
            sc.background = Color::from_hex(0x000000);
            sc.add_light(AmbientLight::new(Color::WHITE, std::f32::consts::PI));
            sc.add(Object3D::mesh(Mesh::new(geometry, mat.into())));
            let rgba = render(&mut sc, &cam);
            let greens = [at(&rgba, 2)[1], at(&rgba, 6)[1]];
            let want = match (channel, with_sets) {
                (0, _) => [0, 255],
                (_, true) => [255, 0],
                (_, false) => [0, 0],
            };
            if greens != want {
                failures.push(format!(
                    "{label} channel {channel}{}: green at u = ¼, ¾ is {greens:?}, three.js gives {want:?}",
                    if with_sets { "" } else { " (no such uv set)" }
                ));
            }
        }
    }
    failures
}
