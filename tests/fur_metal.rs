//! Shell fur on the Metal renderer — its own port of the fur shader — drawn
//! against the same geometry as `tests/fur.rs`, and against the wgpu backend.
//!
//! Skips without a Metal device (or a wgpu adapter, for the comparison).

#![cfg(feature = "metal")]

use threers::fur::{fur_object, FurGroom, FurMaterial, FurPlacement};
use threers::metal::MetalHeadlessRenderer;
use threers::prelude::*;

const N: u32 = 160;

fn scene(fur: &FurMaterial) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.direction = Vector3::new(-0.3, -0.5, -1.0).normalize();
    scene.add(Object3D::light(threers::lights::Light::Directional(sun)));
    scene.add(fur_object(SphereGeometry::new(1.0, 64, 32), fur));
    scene
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(40.0, 1.0, 0.1, 50.0);
    c.position = Vector3::new(0.0, 0.0, 5.0);
    c.look_at(Vector3::ZERO);
    c
}

fn ray_meets_sphere(x: u32, y: u32, r: f32) -> bool {
    let cam = camera();
    let ndc = Vector3::new(
        (x as f32 + 0.5) / N as f32 * 2.0 - 1.0,
        1.0 - (y as f32 + 0.5) / N as f32 * 2.0,
        0.5,
    );
    let dir =
        (ndc.unproject(&cam.view_matrix(), &cam.projection_matrix()) - cam.position).normalize();
    let to_centre = Vector3::ZERO - cam.position;
    (to_centre - dir * to_centre.dot(dir)).length() < r
}

fn lit(p: &[u8]) -> bool {
    p[0] as u32 + p[1] as u32 + p[2] as u32 > 30
}

/// (skin covered, band covered, lit beyond the tips)
fn coverage(px: &[u8]) -> (f32, f32, u32) {
    let (mut skin, mut skin_lit, mut band, mut band_lit, mut beyond) = (0, 0, 0, 0, 0);
    for y in 0..N {
        for x in 0..N {
            let l = lit(&px[((y * N + x) * 4) as usize..][..4]);
            if ray_meets_sphere(x, y, 0.95) {
                skin += 1;
                skin_lit += l as u32;
            } else if ray_meets_sphere(x, y, 1.2) && !ray_meets_sphere(x, y, 1.05) {
                band += 1;
                band_lit += l as u32;
            } else if !ray_meets_sphere(x, y, 1.45) {
                beyond += l as u32;
            }
        }
    }
    (
        skin_lit as f32 / skin as f32,
        band_lit as f32 / band as f32,
        beyond,
    )
}

fn fur(placement: FurPlacement) -> FurMaterial {
    FurMaterial {
        groom: FurGroom {
            length: 0.35,
            density: 60.0,
            placement,
            gravity: Vector3::ZERO,
            ..Default::default()
        },
        root_color: Color::new(0.2, 0.1, 0.05),
        tip_color: Color::new(0.9, 0.75, 0.5),
        ..Default::default()
    }
}

#[test]
fn metal_draws_fur_as_wgpu_does() {
    let Ok(mut metal) = MetalHeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    for placement in [FurPlacement::Uv, FurPlacement::Surface { cell: 0.03 }] {
        let f = fur(placement);
        let m = metal.render_to_rgba(&mut scene(&f), &camera()).unwrap();
        let (skin, band, beyond) = coverage(&m);
        eprintln!("metal {placement:?}: skin {skin:.3}, band {band:.2}, beyond {beyond}");
        assert!(
            skin > 0.97,
            "{placement:?}: the skin is not covered ({skin:.3})"
        );
        assert!(
            band > 0.08 && band < 0.995,
            "{placement:?}: band cover {band:.2}"
        );
        assert_eq!(beyond, 0, "{placement:?}: drawn beyond the tips");

        // The same scene on wgpu: the same strands, so near the same cover,
        // and the same light on them.
        let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
            continue;
        };
        let w = hr.render_to_rgba(&mut scene(&f), &camera());
        let (_, band_w, _) = coverage(&w);
        let mean = |px: &[u8]| {
            let lit_px: Vec<&[u8]> = px.chunks(4).filter(|p| lit(p)).collect();
            lit_px.iter().map(|p| p[0] as f64).sum::<f64>() / lit_px.len() as f64
        };
        let (mm, mw) = (mean(&m), mean(&w));
        eprintln!("  wgpu band {band_w:.2}; mean red of lit pixels: metal {mm:.1}, wgpu {mw:.1}");
        assert!(
            (band - band_w).abs() < 0.04,
            "{placement:?}: band cover metal {band:.2}, wgpu {band_w:.2}"
        );
        assert!(
            (mm - mw).abs() < 6.0,
            "{placement:?}: brightness metal {mm:.1}, wgpu {mw:.1}"
        );
    }
}

#[test]
fn metal_draws_hair_as_wgpu_does() {
    use threers::fur::strands::{hair_object, StrandOptions};
    let Ok(mut metal) = MetalHeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let mut hair = fur(FurPlacement::Uv);
    hair.groom.gravity = Vector3::new(0.0, -1.5, 0.0);
    hair.wind = Vector3::new(0.3, 0.0, 0.0);
    hair.flutter = 0.1;
    let options = StrandOptions {
        count: 6000,
        segments: 8,
        width: 0.012,
        ..Default::default()
    };
    let scene = || {
        let mut s = Scene::new();
        s.background = Color::BLACK;
        s.add_light(AmbientLight::new(Color::WHITE, 1.0));
        let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
        sun.direction = Vector3::new(-0.3, -0.5, -1.0).normalize();
        s.add(Object3D::light(threers::lights::Light::Directional(sun)));
        let mut h = hair_object(&SphereGeometry::new(1.0, 48, 24), &hair, &options);
        threers::fur::set_fur_time(&mut h, 0.4);
        s.add(h);
        s
    };
    let m = metal.render_to_rgba(&mut scene(), &camera()).unwrap();
    let w = hr.render_to_rgba(&mut scene(), &camera());
    let (cover_m, cover_w) = (
        m.chunks(4).filter(|p| lit(p)).count(),
        w.chunks(4).filter(|p| lit(p)).count(),
    );
    let same = m
        .chunks(4)
        .zip(w.chunks(4))
        .filter(|(a, b)| lit(a) == lit(b))
        .count();
    let mean = |px: &[u8]| {
        let lit_px: Vec<&[u8]> = px.chunks(4).filter(|p| lit(p)).collect();
        lit_px.iter().map(|p| p[0] as f64).sum::<f64>() / lit_px.len() as f64
    };
    let (mm, mw) = (mean(&m), mean(&w));
    eprintln!(
        "hair: lit metal {cover_m}, wgpu {cover_w}; agree on {same} of {}; red metal {mm:.1}, wgpu {mw:.1}",
        N * N
    );
    assert!(cover_m > 2000, "Metal drew almost no hair ({cover_m} px)");
    assert!(
        (cover_m as f32 - cover_w as f32).abs() < 0.05 * cover_w as f32,
        "hair cover metal {cover_m}, wgpu {cover_w}"
    );
    assert!(
        same as f32 > 0.95 * (N * N) as f32,
        "only {same} pixels agree"
    );
    assert!(
        (mm - mw).abs() < 6.0,
        "brightness metal {mm:.1}, wgpu {mw:.1}"
    );
}

/// Hair on a skinned mesh, caught swinging: Metal reads each strand's lag
/// from the uv sets as wgpu does, and draws the same pixels.
#[test]
fn metal_draws_skinned_hair_as_wgpu_does() {
    use threers::core::{BufferAttribute, Skeleton, SkinnedMesh};
    use threers::fur::strands::StrandOptions;
    use threers::fur::FurAnimator;
    let Ok(mut metal) = MetalHeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 1.5));
    let bone = scene.add(Object3D::group());
    let mut g = SphereGeometry::new(0.6, 32, 16);
    let count = g.get_attribute("position").unwrap().count();
    g.set_attribute("joint", BufferAttribute::new(vec![0.0; count * 4], 4));
    g.set_attribute(
        "weight",
        BufferAttribute::new((0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(), 4),
    );
    let body = scene.add(Object3D::skinned_mesh(SkinnedMesh::new(
        g,
        Material::Basic(BasicMaterial::new(Color::BLACK)),
        Skeleton::new(vec![threers::core::Bone {
            node: bone,
            inverse_bind: Matrix4::identity(),
        }]),
    )));
    let mut hair = fur(FurPlacement::Uv);
    hair.groom.length = 0.5;
    hair.dynamics.stiffness = 30.0;
    let mut animator = FurAnimator::new();
    animator
        .grow_hair_on_skinned(
            &mut scene,
            body,
            &hair,
            &StrandOptions {
                count: 3000,
                segments: 6,
                width: 0.015,
                ..Default::default()
            },
        )
        .unwrap();
    let still =
        |scene: &mut Scene, metal: &mut MetalHeadlessRenderer, hr: &mut HeadlessRenderer| {
            (
                metal.render_to_rgba(scene, &camera()).unwrap(),
                hr.render_to_rgba(scene, &camera()),
            )
        };
    let (mut v, mut x) = (0.0f32, -0.5f32);
    for f in 0..80 {
        let accel = if f < 60 { 0.0 } else { 15.0 };
        v += accel / 60.0;
        x += v / 60.0;
        scene.get_mut(bone).unwrap().position.x = x;
        animator.update(&mut scene, f as f32 / 60.0, 1.0 / 60.0);
    }
    let (m, w) = still(&mut scene, &mut metal, &mut hr);
    let lit_m = m.chunks(4).filter(|p| lit(p)).count();
    let differ = m
        .chunks(4)
        .zip(w.chunks(4))
        .filter(|(a, b)| lit(a) != lit(b))
        .count();
    eprintln!("skinned hair, swinging: metal lit {lit_m}, pixels differing in cover {differ}");
    assert!(lit_m > 1000, "Metal drew almost no hair ({lit_m})");
    assert!(differ < 50, "{differ} pixels differ between Metal and wgpu");
}

/// Hair's shadow on Metal: the caster is an ordinary mesh drawn into the
/// shadow map only, so Metal casts the same shadow wgpu does.
#[test]
fn metal_casts_the_hair_shadow_wgpu_does() {
    use threers::fur::strands::{hair_object, StrandOptions};
    shadow_parity("hair", &|casts| {
        let mut f = fur(FurPlacement::Uv);
        f.groom.length = 0.5;
        f.groom.gravity = Vector3::new(1.0, 0.0, 0.0);
        hair_object(
            &SphereGeometry::new(0.4, 32, 16),
            &f,
            &StrandOptions {
                count: 4000,
                segments: 6,
                width: 0.012,
                shadow_strands: if casts { 0.25 } else { 0.0 },
            },
        )
    });
}

/// Shell fur's shadow on Metal: its caster group, as wgpu draws it.
#[test]
fn metal_casts_the_fur_shadow_wgpu_does() {
    shadow_parity("fur", &|casts| {
        let mut f = fur(FurPlacement::Uv);
        f.groom.length = 0.4;
        f.groom.gravity = Vector3::new(1.0, 0.0, 0.0);
        f.shadow_height = if casts { 0.6 } else { 0.0 };
        fur_object(SphereGeometry::new(0.5, 48, 24), &f)
    });
}

/// `make(casts)` over a floor under an overhead sun, on Metal and wgpu: the
/// floor pixels its shadow darkens are the same on both.
fn shadow_parity(what: &str, make: &dyn Fn(bool) -> Object3D) {
    const S: u32 = 160;
    let Ok(mut metal) = MetalHeadlessRenderer::builder().size(S, S).build() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let Ok(mut hr) = HeadlessRenderer::builder().size(S, S).build() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let scene = |shadow_strands: f32| {
        let mut s = Scene::new();
        s.background = Color::BLACK;
        s.add_light(AmbientLight::new(Color::WHITE, 0.3));
        let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
        sun.direction = Vector3::new(0.0, -1.0, 0.0);
        sun.cast_shadow = true;
        sun.shadow.map_size = 1024;
        sun.shadow.camera_size = 4.0;
        sun.shadow.camera_near = 0.5;
        sun.shadow.camera_far = 30.0;
        let mut l = Object3D::light(threers::lights::Light::Directional(sun));
        l.position = Vector3::new(0.0, 10.0, 0.0);
        s.add(l);
        let mut m = StandardMaterial::new(Color::WHITE);
        m.metalness = 0.0;
        m.roughness = 1.0;
        let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(40.0, 40.0), m.into()));
        floor.rotate_x(-std::f32::consts::FRAC_PI_2);
        floor.receive_shadow = true;
        s.add(floor);
        let mut hair = make(shadow_strands > 0.0);
        hair.position = Vector3::new(0.0, 2.0, 0.0);
        s.add(hair);
        s
    };
    // From above and to the side, so the floor under the hair shows.
    let mut cam = PerspectiveCamera::new(40.0, 1.0, 0.1, 50.0);
    cam.position = Vector3::new(0.0, 8.0, 6.0);
    cam.look_at(Vector3::ZERO);
    // Each pixel the shadow darkens: lit without a caster, darker with one.
    let shadowed = |with: &[u8], without: &[u8]| -> Vec<bool> {
        with.chunks(4)
            .zip(without.chunks(4))
            .map(|(a, b)| (a[0] as i32) < (b[0] as i32) - 20)
            .collect()
    };
    let m = shadowed(
        &metal.render_to_rgba(&mut scene(0.25), &cam).unwrap(),
        &metal.render_to_rgba(&mut scene(0.0), &cam).unwrap(),
    );
    let w = shadowed(
        &hr.render_to_rgba(&mut scene(0.25), &cam),
        &hr.render_to_rgba(&mut scene(0.0), &cam),
    );
    let (nm, nw) = (
        m.iter().filter(|&&b| b).count(),
        w.iter().filter(|&&b| b).count(),
    );
    let both = m.iter().zip(&w).filter(|(a, b)| **a && **b).count();
    eprintln!("{what} shadow pixels: metal {nm}, wgpu {nw}, both {both}");
    assert!(nw > 300, "wgpu cast almost no {what} shadow ({nw})");
    assert!(
        both as f32 > 0.9 * nm.max(nw) as f32,
        "the shadows differ: metal {nm}, wgpu {nw}, both {both}"
    );
}

/// Under 4× MSAA, Metal tests each sample against the strands as wgpu does
/// (`[[sample_mask]]`, the standard sample positions): the two agree, and
/// both differ from a single-sampled frame at the strands' edges.
#[test]
fn metal_multisamples_fur_as_wgpu_does() {
    let Ok(mut metal) = MetalHeadlessRenderer::builder().size(N, N).msaa(4).build() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    hr.set_msaa(4);
    let f = fur(FurPlacement::Uv);
    let m = metal.render_to_rgba(&mut scene(&f), &camera()).unwrap();
    let w = hr.render_to_rgba(&mut scene(&f), &camera());
    let mut single = HeadlessRenderer::builder().size(N, N).build().unwrap();
    let s = single.render_to_rgba(&mut scene(&f), &camera());
    let diff = |a: &[u8], b: &[u8]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .sum::<f64>()
            / a.len() as f64
    };
    let (mw, ws) = (diff(&m, &w), diff(&w, &s));
    eprintln!("MSAA fur: metal vs wgpu {mw:.2}, wgpu MSAA vs one sample {ws:.2}");
    assert!(ws > 1.0, "MSAA changed nothing ({ws:.2})");
    assert!(
        mw < 0.25 * ws,
        "Metal's MSAA ({mw:.2} off wgpu's) is not wgpu's"
    );
}
