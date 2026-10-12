//! The same scene through every backend, compared to each other.
//!
//! `tests/interpolation.rs` asks each renderer whether it draws an edge well.
//! This one asks whether they all draw the *same picture*, which is a different
//! question and catches a different class of bug: not "is this wrong" but "is
//! this wrong only over here". A backend that is self-consistently mirrored,
//! or that quietly drops a texture, passes every test that only ever looks at
//! one renderer.
//!
//! What is *not* compared is as deliberate as what is. The Metal backend is a
//! documented subset — its own docs hand environment/IBL, shadow maps,
//! post-processing, skinning, and the normal, roughness, metalness, AO and
//! emissive map slots to [`threers::Renderer`] instead — so holding it to wgpu
//! on a scene that uses any of them would be testing a promise nobody made.
//! Every case here stays inside the set both renderers implement, and the two
//! agree there to 0.0002 rms or better.
//!
//! That boundary is worth knowing precisely, because the difference between a
//! limitation and a bug is entirely in whether it was chosen. The phone render
//! this crate exists for draws a blank white screen under `--backend metal`,
//! and for a long time that was written down as the Metal backend being
//! broken. It is not: the screen is an emissive-mapped material, because a
//! phone screen emits its picture rather than reflecting it, and the emissive
//! map slot is one of the ones Metal does not implement. The backend is exactly
//! right about everything it claims to do.
//!
//! The reference is the wgpu renderer, because it is the one the project ships
//! on and the one the other tests hold to an absolute standard. What is
//! compared is coarse on purpose — a downsampled luma grid, and where the
//! bright parts of the frame are — because two backends will never agree to the
//! last bit, and a test that demanded they did would be turned off within a
//! week. What they must agree on is *what is where*.

#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::BasicMaterial;
use threers::math::{Color, Vector3};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFormat};

const N: u32 = 96;

/// A texture with an unmistakable top, bottom and hue.
///
/// A bright band across the top eighth and a square left of centre pin both
/// axes. The three channels differ everywhere — the band is warm, the field is
/// cool — because a greyscale test texture cannot tell a backend that drops
/// colour from one that keeps it, and the first version of this was greyscale
/// and did exactly that: it passed while the phone's screen rendered in black
/// and white.
fn marked_texture() -> Texture {
    const W: u32 = 64;
    let mut data = vec![0u8; (W * W * 4) as usize];
    for y in 0..W {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            let top_band = y < W / 8;
            let left_mark = (16..24).contains(&x) && (28..36).contains(&y);
            let rgb: [u8; 3] = if top_band {
                [255, 180, 40]
            } else if left_mark {
                [40, 255, 120]
            } else {
                [25, 40, 90]
            };
            data[i] = rgb[0];
            data[i + 1] = rgb[1];
            data[i + 2] = rgb[2];
            data[i + 3] = 255;
        }
    }
    Texture::new(W, W, TextureFormat::Rgba8UnormSrgb, data)
}

/// How far the tiles are from grey, on average — a backend that drops colour
/// scores near zero however well it places the pattern.
fn saturation(px: &[u8]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for p in px.chunks(4) {
        let (r, g, b) = (p[0] as f64, p[1] as f64, p[2] as f64);
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        if mx > 10.0 {
            sum += (mx - mn) / mx;
            n += 1.0;
        }
    }
    if n > 0.0 {
        sum / n
    } else {
        0.0
    }
}

/// A single textured quad facing the camera, filling most of the frame.
fn scene() -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x000000);
    let mut mat = BasicMaterial::new(Color::WHITE);
    mat.map = Some(std::sync::Arc::new(marked_texture()));
    let q = Object3D::mesh(Mesh::new(PlaneGeometry::new(40.0, 40.0), mat.into()));
    s.add(q);
    s
}

fn camera() -> PerspectiveCamera {
    let fov: f32 = 30.0;
    let d = N as f32 / (2.0 * (fov.to_radians() / 2.0).tan());
    let mut c = PerspectiveCamera::new(fov, 1.0, 0.1, 1000.0);
    c.position = Vector3::new(0.0, 0.0, d);
    c.look_at(Vector3::ZERO);
    c
}

fn to_linear(b: u8) -> f64 {
    let v = b as f64 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Mean linear luma over a `G × G` grid of tiles — coarse enough that two
/// backends can agree, fine enough that a mirrored frame cannot.
const G: usize = 12;
fn tiles(px: &[u8]) -> Vec<f64> {
    let step = N as usize / G;
    let mut out = vec![0.0; G * G];
    for ty in 0..G {
        for tx in 0..G {
            let mut sum = 0.0;
            for y in ty * step..(ty + 1) * step {
                for x in tx * step..(tx + 1) * step {
                    let i = (y * N as usize + x) * 4;
                    sum += 0.2126 * to_linear(px[i])
                        + 0.7152 * to_linear(px[i + 1])
                        + 0.0722 * to_linear(px[i + 2]);
                }
            }
            out[ty * G + tx] = sum / (step * step) as f64;
        }
    }
    out
}

fn rms(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

fn flip_v(t: &[f64]) -> Vec<f64> {
    (0..G).flat_map(|y| (0..G).map(move |x| (G - 1 - y) * G + x)).map(|i| t[i]).collect()
}

fn flip_h(t: &[f64]) -> Vec<f64> {
    (0..G).flat_map(|y| (0..G).map(move |x| y * G + (G - 1 - x))).map(|i| t[i]).collect()
}

fn wgpu_reference() -> Option<Vec<f64>> {
    let mut r = threers::HeadlessRenderer::builder()
        .size(N, N)
        .build()
        .ok()?;
    Some(tiles(&r.render_to_rgba(&mut scene(), &camera())))
}

/// The Metal backend must draw the same picture as wgpu, the same way up.
///
/// The orientation half of this is not hypothetical. Metal's texture origin is
/// the top-left corner and WebGPU's sampling convention puts `v = 0` there too,
/// but the two disagree once a renderer has made its own choice about which way
/// `v` runs, and nothing in a symmetric test scene can tell. The phone render
/// this crate exists for came out with its screen upside down — the status bar
/// along the bottom edge — while every existing test passed, because every one
/// of them used an untextured material or a texture with no top.
#[test]
fn metal_draws_the_same_picture_as_wgpu() {
    use threers::metal::{MetalError, MetalHeadlessRenderer};
    let Some(reference) = wgpu_reference() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Every combination the app can ask for, because the supersample and MSAA
    // paths take different code to the framebuffer and a flip can live in one
    // of them alone.
    for (ss, msaa) in [(1u32, 1u32), (2, 1), (1, 4), (2, 4)] {
    let mut m = match MetalHeadlessRenderer::builder()
        .size(N, N)
        .supersample(ss)
        .msaa(msaa)
        .build()
    {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let got = tiles(&m.render_to_rgba(&mut scene(), &camera()).expect("metal render"));

    let direct = rms(&reference, &got);
    let mirrored = rms(&reference, &flip_v(&got));
    let flopped = rms(&reference, &flip_h(&got));
    eprintln!(
        "metal ss{ss} msaa{msaa} vs wgpu: {direct:.4} rms as drawn, {mirrored:.4} \
         mirrored vertically, {flopped:.4} horizontally"
    );
    assert!(
        direct < mirrored,
        "the Metal backend draws this texture upside down: it matches wgpu \
         {mirrored:.4} rms when flipped and only {direct:.4} as drawn"
    );
    assert!(
        direct < flopped,
        "the Metal backend draws this texture mirrored left to right"
    );
    // Absolute agreement, once orientation is out of the way. Loose, because
    // two rasterisers will not agree to the bit -- but a dropped texture, a
    // missing material, or a transfer curve apart shows up well inside it.
    assert!(
        direct < 0.05,
        "the Metal backend and wgpu disagree by {direct:.4} rms over a 12x12 \
         luma grid at supersample {ss}, msaa {msaa} -- more than filtering \
         differences account for"
    );
    }
}

/// A child mesh must inherit its parent's transform, in every backend.
///
/// The scene graph computes `matrix_world` once, before any backend sees it,
/// which is the whole point of doing it there — so this should be free. It is
/// worth a test anyway because the failure is invisible in any scene whose
/// objects sit at the origin, and that is most test scenes. The phone render
/// this crate exists for hangs its whole model off one rig node and poses that
/// node; a backend that reads local transforms instead draws a phone that never
/// moves, which is exactly what it did.
#[test]
fn metal_inherits_parent_transforms() {
    use threers::math::Euler;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    // The quad is a child of a rig rotated well off axis, so an ignored parent
    // transform cannot look like a near miss.
    let build = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        let mut mat = BasicMaterial::new(Color::WHITE);
        mat.map = Some(std::sync::Arc::new(marked_texture()));
        let quad = Object3D::mesh(Mesh::new(PlaneGeometry::new(40.0, 40.0), mat.into()));
        let mut rig = Object3D::group();
        rig.quaternion =
            Euler::new(0.35f32, 0.55f32, 0.20f32).to_quaternion();
        rig.position = Vector3::new(4.0, -3.0, 0.0);
        let rig_id = s.add(rig);
        s.add_to(rig_id, quad);
        s
    };
    let Some(mut w) = threers::HeadlessRenderer::builder().size(N, N).build().ok() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let reference = tiles(&w.render_to_rgba(&mut build(), &camera()));
    // What the frame looks like with the rig left at identity — if a backend
    // scores closer to *this* than to the reference, it is ignoring the parent.
    let mut flat = Scene::new();
    flat.background = Color::from_hex(0x000000);
    let mut mat = BasicMaterial::new(Color::WHITE);
    mat.map = Some(std::sync::Arc::new(marked_texture()));
    flat.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(40.0, 40.0),
        mat.into(),
    )));
    let unposed = tiles(&w.render_to_rgba(&mut flat, &camera()));

    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let got = tiles(&m.render_to_rgba(&mut build(), &camera()).expect("metal render"));
    let posed = rms(&reference, &got);
    let ignored = rms(&unposed, &got);
    eprintln!("metal with a posed rig: {posed:.4} rms from posed, {ignored:.4} from unposed");
    assert!(
        posed < ignored,
        "the Metal backend is drawing the child where the parent is not: it \
         matches an unposed rig to {ignored:.4} rms and the posed one only to \
         {posed:.4}"
    );
    assert!(
        posed < 0.05,
        "the Metal backend and wgpu disagree by {posed:.4} rms on a scene whose \
         only difficulty is one parent transform"
    );
}

/// The emissive map slot, now that Metal implements it.
///
/// This is the one the phone render needs: its screen is an emissive-mapped
/// material, because a screen emits its picture rather than reflecting it, and
/// for a long time this backend drew that panel blank white.
#[test]
fn metal_draws_emissive_maps() {
    use threers::materials::StandardMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    // Unlit scene, black base: the emissive map is the only thing that can put
    // a pattern on this quad, so a backend that ignores the slot draws it flat.
    let build = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        let mut mat = StandardMaterial::new(Color::BLACK);
        mat.metalness = 0.0;
        mat.roughness = 1.0;
        mat.emissive = Color::WHITE;
        mat.emissive_map = Some(std::sync::Arc::new(marked_texture()));
        s.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(40.0, 40.0),
            mat.into(),
        )));
        s
    };

    let Some(mut w) = threers::HeadlessRenderer::builder().size(N, N).build().ok() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let w_px = w.render_to_rgba(&mut build(), &camera());
    let m_px = m.render_to_rgba(&mut build(), &camera()).expect("metal render");
    let reference = tiles(&w_px);
    let got = tiles(&m_px);
    let d = rms(&reference, &got);
    let mirrored = rms(&reference, &flip_v(&got));
    eprintln!(
        "emissive map: {d:.4} rms from wgpu, {mirrored:.4} mirrored, saturation \
         {:.3} against {:.3}",
        saturation(&m_px),
        saturation(&w_px)
    );
    assert!(
        saturation(&m_px) > 0.5 * saturation(&w_px),
        "the emissive map is drawn without its colour: saturation {:.3} against \
         wgpu's {:.3}",
        saturation(&m_px),
        saturation(&w_px)
    );
    assert!(
        d < mirrored,
        "the emissive map is drawn upside down: {mirrored:.4} rms flipped \
         against {d:.4} as drawn"
    );
    assert!(
        d < 0.05,
        "the Metal backend and wgpu disagree by {d:.4} rms on an emissive map"
    );
}

/// The rest of the `StandardMaterial` map slots, one at a time.
///
/// One test per slot rather than one scene using all of them, so a failure
/// names the channel instead of saying the picture is different. Each scene is
/// built so that slot is the *only* thing that can vary the surface: a flat
/// quad under one directional light, with the map carrying the variation.
#[test]
fn metal_draws_the_remaining_material_maps() {
    use threers::lights::DirectionalLight;
    use threers::materials::StandardMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    #[derive(Clone, Copy, PartialEq)]
    enum Slot {
        Normal,
        MetallicRoughness,
        AmbientOcclusion,
    }

    let build = |slot: Slot| {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        // Off-axis, so a perturbed normal changes the shading rather than
        // pointing straight back at the light where it would not.
        let mut light = DirectionalLight::new(Color::WHITE, 2.0);
        light.direction = Vector3::new(3.0, -2.0, -4.0);
        s.add_light(light);
        s.add_light(threers::lights::AmbientLight::new(Color::WHITE, 0.3));
        // Linear, because these are quantities and not colours — glTF §3.9.2 —
        // and a test that authored them as sRGB would be asking both renderers
        // to be wrong in the same way.
        let mut data = marked_texture();
        data.format = threers::textures::TextureFormat::Rgba8Unorm;
        let tex = std::sync::Arc::new(data);
        let mut mat = StandardMaterial::new(Color::from_hex(0x808080));
        mat.metalness = 0.0;
        mat.roughness = 0.5;
        match slot {
            Slot::Normal => mat.normal_map = Some(tex),
            Slot::MetallicRoughness => mat.roughness_map = Some(tex),
            Slot::AmbientOcclusion => mat.ao_map = Some(tex),
        }
        s.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(40.0, 40.0),
            mat.into(),
        )));
        s
    };
    let spread = |t: &[f64]| {
        let mean = t.iter().sum::<f64>() / t.len() as f64;
        (t.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / t.len() as f64).sqrt()
    };

    let Some(mut w) = threers::HeadlessRenderer::builder().size(N, N).build().ok() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    for (name, slot) in [
        ("normal_map", Slot::Normal),
        ("roughness_map", Slot::MetallicRoughness),
        ("ao_map", Slot::AmbientOcclusion),
    ] {
        let reference = tiles(&w.render_to_rgba(&mut build(slot), &camera()));
        let got = tiles(&m.render_to_rgba(&mut build(slot), &camera()).expect("metal render"));
        let d = rms(&reference, &got);
        eprintln!(
            "{name:14}: {d:.4} rms from wgpu, contrast {:.4} against {:.4}",
            spread(&got),
            spread(&reference)
        );
        assert!(
            spread(&reference) > 0.005,
            "wgpu draws {name} flat too — the scene does not exercise the slot"
        );
        assert!(
            spread(&got) > 0.5 * spread(&reference),
            "the Metal backend draws a flat quad where wgpu varies with \
             `{name}` — the slot is not reaching it"
        );
        // 0.02, and it used to be 0.11. A roughness map is what makes a
        // direct-lighting difference visible, and there were two: Metal's
        // diffuse lobe was missing its 1/pi, and its geometry term was the
        // separable Schlick-GGX approximation where the wgpu path uses
        // height-correlated Smith. Matching both took `roughness_map` from
        // 0.0945 rms to 0.0160 and `ao_map` — once occlusion stopped
        // attenuating the direct lights too — from 0.0331 to 0.0046.
        //
        // What is left is the roughness floor and geometric specular
        // antialiasing the wgpu path applies and this one does not.
        assert!(
            d < 0.02,
            "the Metal backend and wgpu disagree by {d:.4} rms on `{name}`"
        );
    }
}

/// Image-based lighting: a metal sphere must reflect the room, the same room.
///
/// The environment is where the two renderers have the most room to drift —
/// there is a mip chain to build, a roughness-to-level mapping to agree on, and
/// a split-sum approximation with a page of constants in it. They share the mip
/// chain by construction (`renderer::gpu_texture::mip_chain`, a box downsample
/// in linear light) and the rest is mirrored line for line, so this test is
/// what says the mirroring is faithful.
#[test]
fn metal_lights_from_the_environment() {
    use threers::geometries::SphereGeometry;
    use threers::materials::StandardMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};
    use threers::textures::CubeTexture;

    // Six faces, each a flat but different colour, so a reflection that lands
    // on the wrong face is a different colour rather than a subtler error.
    let environment = {
        let size = 16u32;
        let colours: [[u8; 3]; 6] = [
            [220, 40, 40],
            [40, 220, 40],
            [40, 40, 220],
            [220, 220, 40],
            [220, 40, 220],
            [40, 220, 220],
        ];
        let faces = std::array::from_fn(|i| {
            let c = colours[i];
            (0..size * size)
                .flat_map(|_| [c[0], c[1], c[2], 255])
                .collect::<Vec<u8>>()
        });
        std::sync::Arc::new(CubeTexture::new(
            size,
            threers::textures::TextureFormat::Rgba8UnormSrgb,
            faces,
        ))
    };

    let build = |roughness: f32| {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        s.environment = Some(environment.clone());
        let mut mat = StandardMaterial::new(Color::WHITE);
        mat.metalness = 1.0;
        mat.roughness = roughness;
        s.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(14.0, 48, 32),
            mat.into(),
        )));
        s
    };
    let spread = |t: &[f64]| {
        let mean = t.iter().sum::<f64>() / t.len() as f64;
        (t.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / t.len() as f64).sqrt()
    };

    let Some(mut w) = threers::HeadlessRenderer::builder().size(N, N).build().ok() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    // Mirror and rough: the first says the reflection lands in the right place,
    // the second says the two agree about how roughness walks the mip chain.
    for roughness in [0.05f32, 0.6] {
        let reference = tiles(&w.render_to_rgba(&mut build(roughness), &camera()));
        let got = tiles(&m.render_to_rgba(&mut build(roughness), &camera()).expect("metal"));
        let d = rms(&reference, &got);
        eprintln!(
            "ibl roughness {roughness:.2}: {d:.4} rms from wgpu, contrast {:.4} against {:.4}",
            spread(&got),
            spread(&reference)
        );
        assert!(
            spread(&got) > 0.5 * spread(&reference),
            "the Metal backend draws a flat sphere where wgpu reflects a room \
             at roughness {roughness} — the environment is not reaching it"
        );
        assert!(
            d < 0.08,
            "the Metal backend and wgpu disagree by {d:.4} rms on an \
             environment reflection at roughness {roughness}"
        );
    }
}

/// The panel material the phone render actually uses: one texture in *both*
/// the base-colour and emissive slots, lit by an environment.
///
/// Each half of this passes on its own — `metal_draws_emissive_maps` and
/// `metal_lights_from_the_environment` — which is the reason to test the
/// combination as well. The app's screen came out in black and white long after
/// both of those were green.
#[test]
fn metal_draws_a_screen_the_way_the_app_builds_one() {
    use threers::materials::PhysicalMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    let build = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        let tex = std::sync::Arc::new(marked_texture());
        let mut mat = PhysicalMaterial::new(Color::WHITE);
        mat.metalness = 0.0;
        mat.roughness = 0.4;
        mat.emissive = Color::WHITE;
        // Exactly what `set_panel` does in the app: the same image in both.
        mat.map = Some(tex.clone());
        mat.emissive_map = Some(tex);
        s.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(40.0, 40.0),
            mat.into(),
        )));
        s
    };

    let Some(mut w) = threers::HeadlessRenderer::builder().size(N, N).build().ok() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let w_px = w.render_to_rgba(&mut build(), &camera());
    let m_px = m.render_to_rgba(&mut build(), &camera()).expect("metal render");
    let d = rms(&tiles(&w_px), &tiles(&m_px));
    eprintln!(
        "app-style panel: {d:.4} rms from wgpu, saturation {:.3} against {:.3}",
        saturation(&m_px),
        saturation(&w_px)
    );
    assert!(
        saturation(&m_px) > 0.5 * saturation(&w_px),
        "the panel is drawn in black and white: saturation {:.3} against \
         wgpu's {:.3}",
        saturation(&m_px),
        saturation(&w_px)
    );
    assert!(d < 0.11, "the Metal backend and wgpu disagree by {d:.4} rms");
}

/// The same panel at the size and shape the app actually uses.
///
/// The 64×64 case above passes; the phone's screen is 1179×2556 — large, and
/// not a power of two in either axis. Texture upload has more paths than
/// material binding does (mip generation, row padding, format fallbacks), and
/// they are chosen by size.
#[test]
fn metal_uploads_a_large_non_power_of_two_texture() {
    use threers::materials::PhysicalMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    const PW: u32 = 1179;
    const PH: u32 = 2556;
    let panel = {
        let mut data = vec![0u8; (PW * PH * 4) as usize];
        for y in 0..PH {
            for x in 0..PW {
                let i = ((y * PW + x) * 4) as usize;
                // A strong hue gradient, and a bright band along the top so the
                // orientation is readable too.
                let top = y < PH / 8;
                data[i] = if top { 255 } else { (x * 255 / PW) as u8 };
                data[i + 1] = if top { 180 } else { 40 };
                data[i + 2] = if top { 40 } else { (y * 255 / PH) as u8 };
                data[i + 3] = 255;
            }
        }
        std::sync::Arc::new(Texture::new(PW, PH, TextureFormat::Rgba8UnormSrgb, data))
    };

    let build = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        let mut mat = PhysicalMaterial::new(Color::WHITE);
        mat.metalness = 0.0;
        mat.roughness = 0.4;
        mat.emissive = Color::WHITE;
        mat.map = Some(panel.clone());
        mat.emissive_map = Some(panel.clone());
        s.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(20.0, 40.0),
            mat.into(),
        )));
        s
    };

    let Some(mut w) = threers::HeadlessRenderer::builder().size(N, N).build().ok() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let w_px = w.render_to_rgba(&mut build(), &camera());
    let m_px = m.render_to_rgba(&mut build(), &camera()).expect("metal render");
    let got = tiles(&m_px);
    let reference = tiles(&w_px);
    let d = rms(&reference, &got);
    let mirrored = rms(&reference, &flip_v(&got));
    eprintln!(
        "{PW}x{PH} panel: {d:.4} rms from wgpu, {mirrored:.4} mirrored, \
         saturation {:.3} against {:.3}",
        saturation(&m_px),
        saturation(&w_px)
    );
    assert!(
        saturation(&m_px) > 0.5 * saturation(&w_px),
        "a {PW}x{PH} texture comes out in black and white: saturation {:.3} \
         against wgpu's {:.3}",
        saturation(&m_px),
        saturation(&w_px)
    );
    assert!(
        d < mirrored,
        "a {PW}x{PH} texture is drawn upside down: {mirrored:.4} rms flipped \
         against {d:.4} as drawn"
    );
    assert!(d < 0.11, "the Metal backend and wgpu disagree by {d:.4} rms");
}

/// `set_tonemap` must not take the colour out of the picture.
///
/// Everything above builds its Metal renderer with the defaults; the app that
/// drives this crate calls `set_tonemap(exposure, white)` first, and its screen
/// came out in black and white while every test here was green. The difference
/// between a test and the thing it is meant to cover is usually a line like
/// that one.
#[test]
fn metal_tonemapping_keeps_the_colour() {
    use threers::materials::PhysicalMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    let build = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        let tex = std::sync::Arc::new(marked_texture());
        let mut mat = PhysicalMaterial::new(Color::WHITE);
        mat.metalness = 0.0;
        mat.roughness = 0.4;
        mat.emissive = Color::WHITE;
        mat.map = Some(tex.clone());
        mat.emissive_map = Some(tex);
        s.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(40.0, 40.0),
            mat.into(),
        )));
        s
    };

    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let plain = m.render_to_rgba(&mut build(), &camera()).expect("metal render");
    // What the app asks for.
    m.set_tonemap(1.0, 3.4);
    let graded = m.render_to_rgba(&mut build(), &camera()).expect("metal render");
    eprintln!(
        "tonemap: saturation {:.3} before, {:.3} after set_tonemap(1.0, 3.4)",
        saturation(&plain),
        saturation(&graded)
    );
    assert!(
        saturation(&graded) > 0.5 * saturation(&plain),
        "set_tonemap drops the colour: saturation {:.3} against {:.3} without \
         it",
        saturation(&graded),
        saturation(&plain)
    );
}

/// `set_tonemap`'s exposure must reach the pixels — at every supersample
/// factor.
///
/// The app that drives this crate renders through Metal at `supersample(2)` and
/// found that `--exposure` changed nothing at all: 0.15, 0.4 and 1.0 produced
/// byte-identical frames. A grade that silently does nothing is worse than one
/// that is wrong, because the number in the command line says otherwise.
#[test]
fn metal_exposure_reaches_the_pixels() {
    use threers::materials::BasicMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    let build = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        let mut mat = BasicMaterial::new(Color::WHITE);
        mat.map = Some(std::sync::Arc::new(marked_texture()));
        s.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(40.0, 40.0),
            mat.into(),
        )));
        s
    };
    let mean = |px: &[u8]| {
        px.chunks(4).map(|p| p[0] as f64 + p[1] as f64 + p[2] as f64).sum::<f64>()
            / (px.len() / 4) as f64
            / 3.0
    };

    for (ss, msaa) in [(1u32, 1u32), (2, 1), (1, 4), (2, 4)] {
        let mut m = match MetalHeadlessRenderer::builder()
            .size(N, N)
            .supersample(ss)
            .msaa(msaa)
            .build()
        {
            Ok(m) => m,
            Err(MetalError::NoDevice) => {
                eprintln!("skipping: no Metal device on this machine");
                return;
            }
            Err(e) => panic!("metal renderer: {e:?}"),
        };
        m.set_tonemap(1.0, 3.4);
        let px = m.render_to_rgba(&mut build(), &camera()).expect("metal");
        let bright = mean(&px);
        let sat = saturation(&px);
        m.set_tonemap(0.2, 3.4);
        let dim = mean(&m.render_to_rgba(&mut build(), &camera()).expect("metal"));
        eprintln!(
            "ss{ss} msaa{msaa}: exposure 1.0 -> {bright:.1}, 0.2 -> {dim:.1}, \
             saturation {sat:.3}"
        );
        assert!(
            dim < bright * 0.9,
            "at supersample {ss}, msaa {msaa}, exposure 0.2 renders {dim:.1} \
             against {bright:.1} at 1.0 — the grade is not reaching the frame"
        );
        // The graded frame must still be in colour. This is the combination the
        // app runs and the one that came back in black and white.
        assert!(
            sat > 0.3,
            "at supersample {ss}, msaa {msaa}, the graded frame has saturation \
             {sat:.3} — the colour is being lost somewhere in the chain"
        );
    }
}

/// The film grade — vignette and grain — must exist on the Metal backend and
/// do the same thing it does on the wgpu one.
///
/// It is the last of the post-processing gap: the chain here had bloom, SSAO,
/// a tone map and the resamplers, and no vignette or grain at all, so a caller
/// asking for a film look got one on one backend and not the other.
#[test]
fn metal_applies_the_film_grade() {
    use threers::materials::BasicMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    let build = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x202020);
        // A flat mid-grey quad filling the frame, so a vignette shows as a
        // corner-to-centre ramp and nothing else can be mistaken for one.
        let quad = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(400.0, 400.0),
            BasicMaterial::new(Color::from_hex(0x909090)).into(),
        ));
        s.add(quad);
        s
    };
    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let plain = m.render_to_rgba(&mut build(), &camera()).expect("metal");

    // Vignette: the corners must lose light and the centre must not.
    m.set_film_grade(1.0, 0.0);
    let vig = m.render_to_rgba(&mut build(), &camera()).expect("metal");
    let at = |px: &[u8], x: usize, y: usize| px[(y * N as usize + x) * 4] as f64;
    let (cx, cy) = (N as usize / 2, N as usize / 2);
    let centre_drop = at(&plain, cx, cy) - at(&vig, cx, cy);
    let corner_drop = at(&plain, 2, 2) - at(&vig, 2, 2);
    eprintln!("vignette: centre loses {centre_drop:.1}, corner loses {corner_drop:.1}");
    assert!(
        corner_drop > 8.0,
        "the vignette does not darken the corners ({corner_drop:.1} levels)"
    );
    assert!(
        centre_drop.abs() < corner_drop * 0.35,
        "the vignette darkens the centre by {centre_drop:.1} against \
         {corner_drop:.1} at the corner — it is a global dim, not a vignette"
    );

    // Grain: must vary pixel to pixel, and must move with the seed.
    m.set_film_grade(0.0, 0.6);
    m.set_grain_seed(1.0);
    let g1 = m.render_to_rgba(&mut build(), &camera()).expect("metal");
    m.set_grain_seed(2.0);
    let g2 = m.render_to_rgba(&mut build(), &camera()).expect("metal");
    let spatial = {
        let row: Vec<f64> = (0..N as usize).map(|x| at(&g1, x, cy)).collect();
        let mean = row.iter().sum::<f64>() / row.len() as f64;
        (row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / row.len() as f64).sqrt()
    };
    let moved = g1
        .iter()
        .zip(&g2)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap();
    eprintln!("grain: rms {spatial:.2} levels across a row, {moved} levels between seeds");
    assert!(spatial > 1.0, "the grain is flat ({spatial:.2} levels rms)");
    assert!(
        moved > 0,
        "the grain does not move with the seed — that is dirt on the lens"
    );
}

/// Skinning: a bone that moves must move the vertices bound to it.
///
/// The Metal backend used to draw every skinned mesh in its bind pose — the
/// geometry was right, the animation simply did nothing, which is the kind of
/// failure that looks like a rigging problem in the asset. The test rotates one
/// bone well off axis and asks two things: that the frame changes at all, and
/// that it changes into the same picture wgpu draws.
#[test]
fn metal_skins_a_mesh() {
    use threers::core::{BufferAttribute, Bone, Skeleton, SkinnedMesh};
    use threers::materials::BasicMaterial;
    use threers::math::{Euler, Matrix4};
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    // A tall quad split top/bottom: the lower half rides bone 0 (held still),
    // the upper half bone 1 (rotated), so a working blend bends it visibly.
    let build = |angle: f32| {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        let mut geom = PlaneGeometry::with_segments(12.0, 40.0, 1, 8);
        let count = geom.get_attribute("position").map(|a| a.count()).unwrap_or(0);
        let ys: Vec<f32> = geom
            .get_attribute("position")
            .map(|a| (0..count).map(|i| a.array[i * 3 + 1]).collect())
            .unwrap_or_default();
        let mut joints = vec![0.0f32; count * 4];
        let mut weights = vec![0.0f32; count * 4];
        for i in 0..count {
            // One bone per vertex, chosen by height — a hard split, so a
            // backend that ignores weights and a backend that ignores joints
            // both fail rather than averaging into something plausible.
            let upper = ys.get(i).copied().unwrap_or(0.0) > 0.0;
            joints[i * 4] = if upper { 1.0 } else { 0.0 };
            weights[i * 4] = 1.0;
        }
        geom.set_attribute("joint", BufferAttribute::new(joints, 4));
        geom.set_attribute("weight", BufferAttribute::new(weights, 4));

        let mut skeleton = Skeleton::new(vec![
            Bone { node: threers::core::ObjectId::default(), inverse_bind: Matrix4::identity() },
            Bone { node: threers::core::ObjectId::default(), inverse_bind: Matrix4::identity() },
        ]);
        skeleton.bone_matrices = vec![
            Matrix4::identity(),
            Matrix4::from_quaternion(Euler::new(0.0, 0.0, angle).to_quaternion()),
        ];
        s.add(Object3D::skinned_mesh(SkinnedMesh::new(
            geom,
            BasicMaterial::new(Color::WHITE).into(),
            skeleton,
        )));
        s
    };

    let Some(mut w) = threers::HeadlessRenderer::builder().size(N, N).build().ok() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };

    let rest = tiles(&m.render_to_rgba(&mut build(0.0), &camera()).expect("metal"));
    let bent = tiles(&m.render_to_rgba(&mut build(0.9), &camera()).expect("metal"));
    let moved = rms(&rest, &bent);
    eprintln!("skinning: the pose moves the mesh by {moved:.4} rms");
    assert!(
        moved > 0.02,
        "rotating a bone changed the frame by {moved:.4} rms — the mesh is \
         being drawn in its bind pose and the skeleton ignored"
    );

    let reference = tiles(&w.render_to_rgba(&mut build(0.9), &camera()));
    let d = rms(&reference, &bent);
    eprintln!("skinning: {d:.4} rms from wgpu in the same pose");
    assert!(
        d < 0.03,
        "the Metal backend and wgpu disagree by {d:.4} rms on a skinned pose"
    );
}

/// Shadow maps: an occluder between the light and the floor must darken it,
/// and darken the same place wgpu does.
///
/// Two assertions and they check different failures. That the floor darkens at
/// all says the depth pass ran and the fragment stage read it — the backend
/// used to have neither, and a scene with `cast_shadow` set simply came out
/// evenly lit. That the *unshadowed* part stays bright says the bias is not
/// swallowing the whole floor, which is what a shadow map gets wrong second.
#[test]
fn metal_casts_a_directional_shadow() {
    use threers::geometries::BoxGeometry;
    use threers::lights::DirectionalLight;
    use threers::materials::StandardMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    // A floor with a box hovering over it, lit from almost straight above, so
    // the shadow lands in the middle of frame and nothing else can be mistaken
    // for it.
    let build = |occluder: bool| {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        let mut light = DirectionalLight::new(Color::WHITE, 2.5);
        light.direction = Vector3::new(0.0, -0.35, -1.0);
        light.cast_shadow = true;
        light.shadow.map_size = 1024;
        light.shadow.camera_size = 30.0;
        light.shadow.normal_bias = 0.05;
        s.add_light(light);

        let mut floor_mat = StandardMaterial::new(Color::WHITE);
        floor_mat.metalness = 0.0;
        floor_mat.roughness = 1.0;
        let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(60.0, 60.0), floor_mat.into()));
        floor.position = Vector3::new(0.0, 0.0, -6.0);
        // Receivers opt in too, on both backends.
        floor.receive_shadow = true;
        s.add(floor);

        if occluder {
            let mut box_mat = StandardMaterial::new(Color::WHITE);
            box_mat.metalness = 0.0;
            box_mat.roughness = 1.0;
            let mut b = Object3D::mesh(Mesh::new(BoxGeometry::new(14.0, 14.0, 1.0), box_mat.into()));
            b.position = Vector3::new(0.0, 0.0, 6.0);
            // Casters opt in, on both backends.
            b.cast_shadow = true;
            s.add(b);
        }
        s
    };

    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let lit = tiles(&m.render_to_rgba(&mut build(false), &camera()).expect("metal"));
    let shadowed = tiles(&m.render_to_rgba(&mut build(true), &camera()).expect("metal"));

    // The occluder covers the middle of the frame, so compare the floor where
    // it is visible: the bottom row of tiles, below the box.
    // The occluder sits above frame centre and the light rakes from +z, so the
    // shadow lands just below it. Sample the floor there, and well away from it
    // for the control.
    let centre = (lit[(G / 2 + 1) * G + G / 2] + lit[(G / 2 + 1) * G + G / 2 - 1]) / 2.0;
    let centre_s =
        (shadowed[(G / 2 + 1) * G + G / 2] + shadowed[(G / 2 + 1) * G + G / 2 - 1]) / 2.0;
    let edge = lit[3 * G + 3];
    let edge_s = shadowed[3 * G + 3];
    eprintln!(
        "shadow: under the occluder {centre:.4} -> {centre_s:.4}, \
         away from it {edge:.4} -> {edge_s:.4}"
    );
    assert!(
        centre_s < centre * 0.9,
        "the floor under the occluder went {centre:.4} -> {centre_s:.4} — \
         nothing is casting"
    );
    assert!(
        edge_s > edge * 0.7,
        "the floor away from the occluder went {edge:.4} -> {edge_s:.4} — the \
         whole scene is in shadow, which is a bias or a matrix problem rather \
         than a shadow"
    );

    // And it must fall where wgpu puts it. The two do not have to agree on the
    // penumbra to the last level — the filters are the same shape but the
    // rasterisers are not — so this compares where the shadow *is* rather than
    // exactly how dark: the difference each backend's occluder makes, tile by
    // tile, correlated.
    let Some(mut w) = threers::HeadlessRenderer::builder().size(N, N).build().ok() else {
        return;
    };
    let w_lit = tiles(&w.render_to_rgba(&mut build(false), &camera()));
    let w_shadowed = tiles(&w.render_to_rgba(&mut build(true), &camera()));
    let delta = |a: &[f64], b: &[f64]| -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    };
    let dm = delta(&lit, &shadowed);
    let dw = delta(&w_lit, &w_shadowed);
    let dot: f64 = dm.iter().zip(&dw).map(|(a, b)| a * b).sum();
    let na = dm.iter().map(|a| a * a).sum::<f64>().sqrt();
    let nb = dw.iter().map(|b| b * b).sum::<f64>().sqrt();
    let correlation = if na > 1e-9 && nb > 1e-9 { dot / (na * nb) } else { 0.0 };
    eprintln!("shadow: metal and wgpu agree on its shape to {correlation:.4}");
    assert!(
        correlation > 0.9,
        "the shadow is in a different place than wgpu puts it (correlation \
         {correlation:.4} between where each darkens)"
    );
}

/// Spot shadows: an occluder must darken the floor, where wgpu darkens it.
///
/// Same shape as the directional test and the same two failures it guards: a
/// shadow that never appears, and one that swallows the whole scene. The
/// correlation is what says it is in the right *place* — the two backends need
/// not agree on the penumbra to the last level, but a shadow cast from the
/// wrong position lands somewhere else entirely.
#[test]
fn metal_casts_a_spot_shadow() {
    let (darkest, darkened, correlation, _) = shadow_scene(ShadowKind::Spot);
    eprintln!(
        "spot shadow: darkest tile loses {darkest:.4}, {darkened} of {} \
         darkened, agrees with wgpu to {correlation:.4}",
        G * G
    );
    assert!(
        darkest > 0.02,
        "the spot light casts nothing: the most any tile darkened was {darkest:.4}"
    );
    assert!(
        darkened < (G * G) / 2,
        "the spot light shadows the whole scene: {darkened} of {} tiles \
         darkened, so this is a bias or a matrix problem rather than a shadow",
        G * G
    );
    assert!(
        correlation > 0.85,
        "the spot shadow is in a different place than wgpu puts it \
         (correlation {correlation:.4})"
    );
}

/// Point shadows: an occluder must darken the floor, where a spot light in the
/// same place darkens it.
///
/// Checked three ways, because this one needed all three. Against the geometry in
/// `tests/shadow_fixtures.rs`, which is the reference that actually settled it;
/// against wgpu, which is only useful now that *its* point shadows work too;
/// and against this backend's own spot light, since a point and a spot light at
/// one position throw one shadow.
///
/// Two things had to be right to get here, and each was invisible on its own.
/// The cube stores *radial* distance, written from a fragment stage — the only
/// shadow pass here that has one — because the lookup compares
/// `length(world - light) / radius`, and projected depth is per-face and
/// discontinuous at the seams. Both renderers stored projected depth and so
/// never matched at all. And the face matrices negate clip-space x, because
/// cube maps are left-handed in every graphics API while `look_at` is
/// right-handed; without it each face is individually reasonable and the
/// shadow lands on the wrong side.
#[test]
fn metal_casts_a_point_shadow() {
    let (darkest, darkened, correlation, _) = shadow_scene(ShadowKind::Point);
    let (spot_darkest, spot_darkened, _, _) = shadow_scene(ShadowKind::Spot);
    eprintln!(
        "point shadow: darkest tile loses {darkest:.4} over {darkened} tiles, \
         against {spot_darkest:.4} over {spot_darkened} for a spot light in \
         the same place"
    );
    assert!(
        darkest > 0.02,
        "the point light casts nothing: the most any tile darkened was {darkest:.4}"
    );
    assert!(
        darkened < (G * G) / 2,
        "the point light shadows the whole scene: {darkened} of {} tiles \
         darkened",
        G * G
    );
    // Same position, same occluder, same floor — so the same shadow.
    assert!(
        (darkest - spot_darkest).abs() < spot_darkest * 0.35,
        "the point light darkens by {darkest:.4} where a spot light in the \
         same place darkens by {spot_darkest:.4} — the cube is being sampled \
         somewhere other than where the occluder is"
    );
    assert!(
        darkened.abs_diff(spot_darkened) <= 3,
        "the point shadow covers {darkened} tiles and the spot shadow \
         {spot_darkened} — they are not the same shadow"
    );
    assert!(
        correlation > 0.85,
        "the point shadow is in a different place than wgpu puts it \
         (correlation {correlation:.4})"
    );
}

#[derive(Clone, Copy, PartialEq)]
enum ShadowKind {
    Spot,
    Point,
}

/// Render an occluder over a floor, lit by one shadow-casting light, through
/// both backends. Returns the darkest tile's loss, how many tiles lost more
/// than a quarter of that, how well the two backends agree on *where* the loss
/// is, and wgpu's own darkest loss.
fn shadow_scene(kind: ShadowKind) -> (f64, usize, f64, f64) {
    use threers::geometries::BoxGeometry;
    use threers::lights::{Light, PointLight, SpotLight};
    use threers::materials::StandardMaterial;
    use threers::metal::MetalHeadlessRenderer;

    let build = |occluder: bool| {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        // Off-axis, so the shadow falls clear of the occluder rather than
        // directly behind it where the box would hide its own shadow.
        let at = Vector3::new(14.0, 14.0, 26.0);
        let light = match kind {
            ShadowKind::Spot => {
                let mut l = SpotLight::new(Color::WHITE, 900.0);
                l.angle = 0.7;
                l.penumbra = 0.2;
                l.distance = 120.0;
                l.cast_shadow = true;
                l.shadow.map_size = 1024;
                l.shadow.camera_near = 1.0;
                l.shadow.camera_far = 120.0;
                l.direction = Vector3::new(-14.0, -14.0, -32.0).normalize();
                Light::Spot(l)
            }
            ShadowKind::Point => {
                let mut l = PointLight::new(Color::WHITE, 900.0);
                l.distance = 120.0;
                l.cast_shadow = true;
                Light::Point(l)
            }
        };
        let mut obj = Object3D::light(light);
        obj.position = at;
        s.add(obj);

        let mut floor_mat = StandardMaterial::new(Color::WHITE);
        floor_mat.metalness = 0.0;
        floor_mat.roughness = 1.0;
        let mut floor =
            Object3D::mesh(Mesh::new(PlaneGeometry::new(60.0, 60.0), floor_mat.into()));
        floor.position = Vector3::new(0.0, 0.0, -6.0);
        floor.receive_shadow = true;
        s.add(floor);

        if occluder {
            let mut box_mat = StandardMaterial::new(Color::WHITE);
            box_mat.metalness = 0.0;
            box_mat.roughness = 1.0;
            let mut b =
                Object3D::mesh(Mesh::new(BoxGeometry::new(12.0, 12.0, 1.0), box_mat.into()));
            b.position = Vector3::new(0.0, 0.0, 10.0);
            b.cast_shadow = true;
            s.add(b);
        }
        s
    };

    let mut m = MetalHeadlessRenderer::builder().size(N, N).build().expect("metal");
    let mut w = threers::HeadlessRenderer::builder().size(N, N).build().expect("wgpu");
    let lit = tiles(&m.render_to_rgba(&mut build(false), &camera()).expect("metal"));
    let shadowed = tiles(&m.render_to_rgba(&mut build(true), &camera()).expect("metal"));
    let w_lit = tiles(&w.render_to_rgba(&mut build(false), &camera()));
    let w_shadowed = tiles(&w.render_to_rgba(&mut build(true), &camera()));

    // Found rather than hard-coded: where a shadow lands depends on the light,
    // the occluder and the floor, and a test that assumes a tile breaks the
    // moment any of the three moves.
    let dm: Vec<f64> = lit.iter().zip(&shadowed).map(|(a, b)| a - b).collect();
    let dw: Vec<f64> = w_lit.iter().zip(&w_shadowed).map(|(a, b)| a - b).collect();
    let darkest = dm.iter().cloned().fold(f64::MIN, f64::max);
    let darkened = dm.iter().filter(|d| **d > darkest * 0.25).count();
    let dot: f64 = dm.iter().zip(&dw).map(|(a, b)| a * b).sum();
    let na = dm.iter().map(|a| a * a).sum::<f64>().sqrt();
    let nb = dw.iter().map(|b| b * b).sum::<f64>().sqrt();
    let correlation = if na > 1e-9 && nb > 1e-9 { dot / (na * nb) } else { 0.0 };
    (darkest, darkened, correlation, dw.iter().cloned().fold(f64::MIN, f64::max))
}

/// Bloom must not take the colour out of the picture.
///
/// The one post-processing path no test here had switched on, and the reason
/// the phone render came out in black and white under `--backend metal` while
/// every material, texture, tone-map and antialiasing test passed: the app
/// enables bloom by default and none of these did.
///
/// The cause is a struct that is not the same struct on both sides of the
/// language boundary. Metal aligns `float2` to eight bytes and Rust aligns
/// `[f32; 2]` to four, so `CompositeParams` disagreed about where everything
/// after `bloom_size` lived — the shader read `chroma_sat` out of the bytes
/// Rust had written `chroma_lift` into. Zero saturation scale means
/// `rgb = maxc + (rgb - maxc) * 0`, which is the max channel in all three
/// components: a perfectly grey frame, from a shader whose arithmetic is a
/// no-op when its parameters are right.
///
/// Worth stating as a rule rather than a fix: any struct shared with a shader
/// needs its layout asserted, not assumed. `bytemuck` will tell you the Rust
/// side is a plain old data type and nothing at all about whether the other
/// language agrees.
#[test]
fn metal_bloom_keeps_the_colour() {
    use threers::materials::BasicMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    let build = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        let mut mat = BasicMaterial::new(Color::WHITE);
        mat.map = Some(std::sync::Arc::new(marked_texture()));
        s.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(40.0, 40.0),
            mat.into(),
        )));
        s
    };

    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let plain = m.render_to_rgba(&mut build(), &camera()).expect("metal");
    // What the app asks for.
    m.set_bloom(0.15, 0.80, 0.9);
    let bloomed = m.render_to_rgba(&mut build(), &camera()).expect("metal");
    eprintln!(
        "bloom: saturation {:.3} before, {:.3} after set_bloom(0.15, …)",
        saturation(&plain),
        saturation(&bloomed)
    );
    assert!(
        saturation(&bloomed) > 0.5 * saturation(&plain),
        "bloom drops the colour: saturation {:.3} against {:.3} without it",
        saturation(&bloomed),
        saturation(&plain)
    );
}

/// Which post-processing stage turns the picture upside down.
///
/// Orientation is checked elsewhere here, but only through the plain and
/// supersampled paths — and the app's frames go through bloom and a tone map
/// as well. That is not a hypothetical gap: the phone render comes out
/// inverted under `--backend metal`, with its status bar along the bottom
/// edge, and every orientation test passed.
///
/// The suspect is named in `metal/headless.rs`: the upscale pass samples by
/// `uv`, whose origin is the bottom, while the resolve samples by `position`,
/// whose origin is the top. A chain that mixes the two flips whatever passes
/// through it. This walks the stages one at a time so the answer is a stage
/// rather than a suspicion.
#[test]
fn metal_post_processing_keeps_the_picture_upright() {
    use threers::materials::BasicMaterial;
    use threers::metal::{MetalError, MetalHeadlessRenderer};

    let build = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        let mut mat = BasicMaterial::new(Color::WHITE);
        mat.map = Some(std::sync::Arc::new(marked_texture()));
        s.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(40.0, 40.0),
            mat.into(),
        )));
        s
    };
    let reference = tiles(&threers::HeadlessRenderer::builder()
        .size(N, N)
        .build()
        .expect("wgpu")
        .render_to_rgba(&mut build(), &camera()));

    // Each stage added to the one before it, so the first row that flips names
    // the stage that flips it.
    let stages: [(&str, u32, bool, bool); 5] = [
        ("plain", 1, false, false),
        ("supersample 2", 2, false, false),
        ("+ tonemap", 2, true, false),
        ("+ bloom", 2, true, true),
        ("bloom only", 1, false, true),
    ];
    let mut flipped: Vec<&str> = Vec::new();
    for (label, ss, tonemap, bloom) in stages {
        let mut m = match MetalHeadlessRenderer::builder().size(N, N).supersample(ss).build() {
            Ok(m) => m,
            Err(MetalError::NoDevice) => {
                eprintln!("skipping: no Metal device on this machine");
                return;
            }
            Err(e) => panic!("metal renderer: {e:?}"),
        };
        if tonemap {
            m.set_tonemap(1.0, 3.4);
        }
        if bloom {
            m.set_bloom(0.15, 0.80, 0.9);
        }
        let got = tiles(&m.render_to_rgba(&mut build(), &camera()).expect("metal"));
        let direct = rms(&reference, &got);
        let mirrored = rms(&reference, &flip_v(&got));
        eprintln!("{label:16} {direct:.4} as drawn, {mirrored:.4} mirrored");
        if mirrored < direct {
            flipped.push(label);
        }
    }
    assert!(
        flipped.is_empty(),
        "these stages draw the picture upside down: {flipped:?}"
    );
}
