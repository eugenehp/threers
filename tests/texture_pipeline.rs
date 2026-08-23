//! Two properties of the 2D texture path that only show up once you point a
//! camera at them: a bound `map` contributes its alpha to lit materials, and
//! minified textures are filtered through a mip chain instead of aliasing.
//!
//! Skipped when no GPU adapter is available.

use threers::{
    AmbientLight, Color, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera,
    PlaneGeometry, Scene, StandardMaterial, Texture, TextureFormat, TextureWrap, Vector3,
};

const W: u32 = 200;
const H: u32 = 200;

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(W, H)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
}

/// A quad filling the view, facing the camera.
fn quad_scene(material: StandardMaterial, background: Color) -> Scene {
    let mut scene = Scene::new();
    scene.background = background;
    // Flat, full lighting so the test measures the texture, not the shading.
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0, 2.0),
        Material::Standard(material),
    )));
    scene
}

fn camera(distance: f32) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(45.0, 1.0, 0.01, 100.0);
    c.position = Vector3::new(0.0, 0.0, distance);
    c.look_at(Vector3::ZERO);
    c
}

fn pixel(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

#[test]
fn a_maps_alpha_makes_a_lit_material_see_through() {
    let Some(mut renderer) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Left half opaque white, right half fully transparent.
    let (tw, th) = (64u32, 64u32);
    let mut data = vec![0u8; (tw * th * 4) as usize];
    for y in 0..th {
        for x in 0..tw {
            let i = ((y * tw + x) * 4) as usize;
            let opaque = x < tw / 2;
            data[i..i + 4].copy_from_slice(&[255, 255, 255, if opaque { 255 } else { 0 }]);
        }
    }
    let texture = Texture::new(tw, th, TextureFormat::Rgba8UnormSrgb, data);

    let mut material = StandardMaterial::new(Color::WHITE);
    material.map = Some(std::sync::Arc::new(texture));
    material.roughness = 1.0;
    material.metalness = 0.0;
    // Below 1 so the draw is routed to the alpha pipeline at all; the map's own
    // alpha does the shaping from there.
    material.opacity = 0.99;

    let background = Color::new(0.0, 0.0, 1.0);
    let mut scene = quad_scene(material, background);
    let rgba = renderer.render_to_rgba(&mut scene, &camera(2.6));

    // The plane covers the middle of the frame; sample inside each half.
    let left = pixel(&rgba, W / 2 - 30, H / 2);
    let right = pixel(&rgba, W / 2 + 30, H / 2);
    // Ambient-lit white lands around 150, not 255 — what matters is that it is
    // bright and neutral rather than the blue behind it.
    assert!(
        left[0] > 110 && (left[2] as i32 - left[0] as i32).abs() < 25,
        "the opaque half should be a lit white quad, got {left:?}"
    );
    assert!(
        right[2] as i32 - right[0] as i32 > 60,
        "the alpha-0 half should show the blue background, got {right:?}"
    );
}

#[test]
fn a_minified_checkerboard_averages_out_instead_of_aliasing() {
    let Some(mut renderer) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Checkerboard in 4×4 texel blocks. A *one*-texel checker is the wrong
    // probe: bilinear filtering averages its four neighbours and lands on grey
    // whether or not mips exist. Blocks wider than the bilinear footprint do
    // not average out, so a sampler stuck on level 0 returns whichever block
    // each pixel happens to hit — and the result speckles.
    const BLOCK: u32 = 4;
    let (tw, th) = (512u32, 512u32);
    let mut data = vec![0u8; (tw * th * 4) as usize];
    for y in 0..th {
        for x in 0..tw {
            let i = ((y * tw + x) * 4) as usize;
            let v = if ((x / BLOCK) + (y / BLOCK)).is_multiple_of(2) {
                255
            } else {
                0
            };
            data[i..i + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    let mut texture = Texture::new(tw, th, TextureFormat::Rgba8Unorm, data);
    texture.wrap_s = TextureWrap::Repeat;
    texture.wrap_t = TextureWrap::Repeat;

    let mut material = StandardMaterial::new(Color::WHITE);
    material.map = Some(std::sync::Arc::new(texture));
    material.roughness = 1.0;
    material.metalness = 0.0;

    let mut scene = quad_scene(material, Color::BLACK);
    // Far enough that 512 texels cross ~60 pixels: a ~8× minification.
    let rgba = renderer.render_to_rgba(&mut scene, &camera(14.0));

    // Sample a block in the middle of the quad.
    let mut values = Vec::new();
    for y in (H / 2 - 6)..(H / 2 + 6) {
        for x in (W / 2 - 6)..(W / 2 + 6) {
            values.push(pixel(&rgba, x, y)[0] as f32);
        }
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let deviation = variance.sqrt();

    // The average of black and white in linear light is mid-grey (~188 in sRGB).
    assert!(
        (60.0..240.0).contains(&mean),
        "minified checkerboard should average to grey, got {mean}"
    );
    // And it should be *flat*: an unfiltered sampler leaves this speckled with
    // pure black and white, pushing the deviation past 100.
    assert!(
        deviation < 25.0,
        "minified texture is aliasing — deviation {deviation:.1} over {} samples",
        values.len()
    );
}

/// A texture whose top half is `top` and bottom half is `bottom`.
fn split_texture(top: [u8; 3], bottom: [u8; 3], flip_y: bool) -> Texture {
    let (w, h) = (64u32, 32u32);
    let mut data = vec![255u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let o = ((y * w + x) * 4) as usize;
            data[o..o + 3].copy_from_slice(if y < h / 2 { &top } else { &bottom });
        }
    }
    let mut t = Texture::new(w, h, TextureFormat::Rgba8UnormSrgb, data);
    t.flip_y = flip_y;
    t
}

/// Render a unit sphere from +Z and return the colour above and below centre.
fn sphere_poles(r: &mut HeadlessRenderer, texture: &std::sync::Arc<Texture>) -> ([u8; 4], [u8; 4]) {
    use threers::{SphereGeometry, Vector3};
    let mut m = StandardMaterial::new(Color::WHITE);
    m.map = Some(texture.clone());
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 3.0));
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 64, 32),
        Material::Standard(m),
    )));
    let mut c = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 3.0);
    c.look_at(Vector3::ZERO);
    let img = r.render_to_rgba(&mut scene, &c);
    (pixel(&img, W / 2, H / 4), pixel(&img, W / 2, H * 3 / 4))
}

#[test]
fn flip_y_puts_an_images_top_row_at_the_north_pole() {
    let Some(mut r) = renderer() else { return };
    const RED: [u8; 3] = [220, 20, 20];
    const BLUE: [u8; 3] = [20, 20, 220];

    // An equirectangular image stores the north pole in its first row, and
    // `SphereGeometry` puts the north pole at `uv.y = 1` — the *last* row of an
    // unflipped upload. `flip_y` is what reconciles the two, and it is on by
    // default exactly as in three.js. Ignoring it renders every photographic
    // planet, sky and globe upside down.
    // Both textures are built up front and held for the whole test: the
    // renderer's texture cache is keyed by the `Arc`'s address, so dropping the
    // first would let the second land on the same address and be served the
    // stale upload.
    let flipped = std::sync::Arc::new(split_texture(RED, BLUE, true));
    let verbatim = std::sync::Arc::new(split_texture(RED, BLUE, false));

    let (top, bottom) = sphere_poles(&mut r, &flipped);
    assert!(
        top[0] > top[2] && bottom[2] > bottom[0],
        "north should show the image's first row: top {top:?} bottom {bottom:?}"
    );

    // …and clearing the flag is still honoured, for data authored bottom-up
    // (DataTexture, render targets).
    let (top, bottom) = sphere_poles(&mut r, &verbatim);
    assert!(
        top[2] > top[0] && bottom[0] > bottom[2],
        "flip_y = false should upload verbatim: top {top:?} bottom {bottom:?}"
    );
}

#[test]
fn a_maps_offset_and_repeat_select_a_sub_rectangle() {
    let Some(mut r) = renderer() else { return };
    // Four quadrants, each a different colour. `repeat = 0.5` with an `offset`
    // picks exactly one of them — which is what lets a tiled globe share one
    // atlas between patches instead of slicing a texture per tile.
    let (w, h) = (64u32, 64u32);
    let mut data = vec![255u8; (w * h * 4) as usize];
    const TL: [u8; 3] = [220, 20, 20];
    const TR: [u8; 3] = [20, 220, 20];
    const BL: [u8; 3] = [20, 20, 220];
    const BR: [u8; 3] = [220, 220, 20];
    for y in 0..h {
        for x in 0..w {
            let o = ((y * w + x) * 4) as usize;
            let c = match (x < w / 2, y < h / 2) {
                (true, true) => TL,
                (false, true) => TR,
                (true, false) => BL,
                (false, false) => BR,
            };
            data[o..o + 3].copy_from_slice(&c);
        }
    }

    // Hold every texture for the whole test — the cache is address-keyed.
    let quads: Vec<std::sync::Arc<Texture>> = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
        .iter()
        .map(|&(ox, oy)| {
            let mut t = Texture::new(w, h, TextureFormat::Rgba8UnormSrgb, data.clone());
            t.offset = threers::Vector2::new(ox, oy);
            t.repeat = threers::Vector2::new(0.5, 0.5);
            std::sync::Arc::new(t)
        })
        .collect();

    let sample = |r: &mut HeadlessRenderer, t: &std::sync::Arc<Texture>| {
        let mut m = StandardMaterial::new(Color::WHITE);
        m.map = Some(t.clone());
        let img = {
            let mut scene = quad_scene(m, Color::BLACK);
            scene.add_light(AmbientLight::new(Color::WHITE, 3.0));
            r.render_to_rgba(&mut scene, &camera(2.0))
        };
        pixel(&img, W / 2, H / 2)
    };

    // flip_y is on, so texture row 0 (TL/TR) lands at v = 1 — the *upper* half
    // of the quad. Offset 0 therefore selects the bottom-left source quadrant.
    let dominant = |p: [u8; 4]| {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        if r > 120 && g > 120 {
            "yellow"
        } else if r > 120 {
            "red"
        } else if g > 120 {
            "green"
        } else if b > 120 {
            "blue"
        } else {
            "none"
        }
    };
    let seen: Vec<&str> = quads.iter().map(|t| dominant(sample(&mut r, t))).collect();
    assert_eq!(
        seen.len(),
        seen.iter().collect::<std::collections::HashSet<_>>().len(),
        "each offset should select a different quadrant, got {seen:?}"
    );
    assert!(
        !seen.contains(&"none"),
        "some offset sampled nothing: {seen:?}"
    );
}

/// Several textures over one buffer should cost one GPU texture between them.
///
/// A body split into patches for tile streaming wants each patch to address its
/// own quarter of a shared map through `offset`/`repeat`. Keying the upload
/// cache on the `Texture` object made that eight uploads of the same bytes; the
/// Moon was slicing and uploading twenty-four copies of three maps.
#[test]
fn textures_over_one_buffer_share_a_single_upload() {
    let Some(mut renderer) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (w, h) = (64u32, 64u32);
    let mut data = vec![255u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            data[i] = (x * 4) as u8;
            data[i + 1] = (y * 4) as u8;
        }
    }
    let base = Texture::new(w, h, TextureFormat::Rgba8Unorm, data);

    // Four quadrant views of the same bytes.
    let views: Vec<std::sync::Arc<Texture>> = (0..4)
        .map(|i| {
            let mut t = base.clone();
            t.offset = threers::Vector2::new(if i % 2 == 0 { 0.0 } else { 0.5 }, 0.0);
            t.repeat = threers::Vector2::new(0.5, 0.5);
            std::sync::Arc::new(t)
        })
        .collect();
    // Same buffer behind all of them — that is the premise of the test.
    for v in &views {
        assert!(
            std::sync::Arc::ptr_eq(&v.data, &base.data),
            "cloning a Texture must not clone its pixels"
        );
    }

    renderer.renderer().clear_texture_cache();
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    for (i, v) in views.iter().enumerate() {
        let mut m = StandardMaterial::new(Color::WHITE);
        m.map = Some(v.clone());
        let mut obj = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(0.8, 0.8),
            Material::Standard(m),
        ));
        obj.position = Vector3::new(if i % 2 == 0 { -0.5 } else { 0.5 }, 0.0, 0.0);
        scene.add(obj);
    }
    let _ = renderer.render_to_rgba(&mut scene, &camera(2.5));
    let shared = renderer.renderer().cached_texture_count();

    // The same four, each over its own copy of the bytes.
    renderer.renderer().clear_texture_cache();
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    for i in 0..4 {
        let mut t = Texture::new(w, h, TextureFormat::Rgba8Unorm, base.data.as_ref().clone());
        t.offset = threers::Vector2::new(if i % 2 == 0 { 0.0 } else { 0.5 }, 0.0);
        t.repeat = threers::Vector2::new(0.5, 0.5);
        let mut m = StandardMaterial::new(Color::WHITE);
        m.map = Some(std::sync::Arc::new(t));
        let mut obj = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(0.8, 0.8),
            Material::Standard(m),
        ));
        obj.position = Vector3::new(if i % 2 == 0 { -0.5 } else { 0.5 }, 0.0, 0.0);
        scene.add(obj);
    }
    let _ = renderer.render_to_rgba(&mut scene, &camera(2.5));
    let copied = renderer.renderer().cached_texture_count();

    eprintln!("uploads — four views of one buffer: {shared}, four copies: {copied}");
    assert!(
        copied >= shared + 3,
        "copies should upload separately: {copied} vs {shared}"
    );
    assert!(
        shared <= copied - 3,
        "views over one buffer should share an upload: {shared} against {copied}"
    );
}

/// Mips filtered on the GPU must match the ones computed on the CPU.
///
/// Both average a 2x2 block in linear light; the GPU does it with one bilinear
/// tap and lets the render target's own sRGB conversion handle the encode,
/// which is the part most likely to be wrong. Rendering a checkerboard at a
/// size that forces a high mip is what makes a difference visible: a wrong
/// transfer function shows as the wrong grey.
#[test]
fn gpu_filtered_mips_match_the_cpu_ones() {
    let Some(mut renderer) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Black-and-white checkerboard: its average is a specific grey, and in
    // linear light that grey is 188 in sRGB, not 128.
    let (w, h) = (256u32, 256u32);
    let mut data = vec![255u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let v = if (x + y) % 2 == 0 { 255 } else { 0 };
            data[i] = v;
            data[i + 1] = v;
            data[i + 2] = v;
        }
    }
    let tex = std::sync::Arc::new(Texture::new(w, h, TextureFormat::Rgba8UnormSrgb, data));

    // Far enough away that the plane samples deep into the chain.
    let mut m = StandardMaterial::new(Color::WHITE);
    m.map = Some(tex);
    let mut scene = quad_scene(m, Color::BLACK);
    let img = renderer.render_to_rgba(&mut scene, &camera(24.0));

    // The quad is tiny at this distance; measure the middle of it.
    let mid = pixel(&img, W / 2, H / 2);

    // Compared against a solid texture of the colour the checker *should*
    // average to, rendered through the same material, lighting and tone
    // mapping — an absolute expectation here would be measuring those instead.
    // A 50/50 black-and-white checker averages to 0.5 in linear light, which is
    // 188 in sRGB; averaging the bytes would give 128.
    let solid = std::sync::Arc::new(Texture::new(
        4,
        4,
        TextureFormat::Rgba8UnormSrgb,
        vec![188; 4 * 4 * 4],
    ));
    let mut m = StandardMaterial::new(Color::WHITE);
    m.map = Some(solid);
    let mut scene = quad_scene(m, Color::BLACK);
    let reference = renderer.render_to_rgba(&mut scene, &camera(24.0));
    let want = pixel(&reference, W / 2, H / 2);

    eprintln!(
        "minified checker {:?} against solid 188 {:?}",
        &mid[..3],
        &want[..3]
    );
    for c in 0..3 {
        assert!(
            (mid[c] as i32 - want[c] as i32).abs() < 12,
            "channel {c}: minified checker gave {} where a linear-light average \
             renders {} — a chain that skipped the transfer function would come \
             out far darker",
            mid[c],
            want[c]
        );
    }
}
