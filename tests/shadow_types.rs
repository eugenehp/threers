//! Shadow filters (`ShadowType`) and cascaded shadow maps.
//!
//! Each frame is measured against the same frame with nothing casting, so
//! the floor's own shading cancels: a pixel's ratio is the light it keeps.
//! A *partial* pixel keeps between a tenth and nine tenths — an edge.
//!
//! - The filters order by edge width: `Basic` hardest, then PCF, then soft
//!   PCF.
//! - PCSS hardens toward contact: a pole's shadow is sharp at its foot and
//!   soft at its far end, where PCF is equally wide along it.
//! - Cascades put fine texels near the camera: a small box near a camera
//!   whose shadows must reach far is drawn with a much sharper edge through
//!   four cascades than through one map stretched over the whole reach, and
//!   in the same place as a map fitted tightly around it.
//!
//! On both renderers: wgpu, and Metal with the `metal` feature.
//!
//! Skipped when no GPU adapter is available.

use threers::lights::{Light, ShadowType};
use threers::prelude::*;

const N: u32 = 256;

fn floor() -> Object3D {
    let mut m = StandardMaterial::new(Color::WHITE);
    m.metalness = 0.0;
    m.roughness = 1.0;
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(400.0, 400.0), m.into()));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    floor
}

fn caster(size: Vector3, at: Vector3, casts: bool) -> Object3D {
    let mut b = Object3D::mesh(Mesh::new(
        BoxGeometry::new(size.x, size.y, size.z),
        Material::Standard(StandardMaterial::new(Color::WHITE)),
    ));
    b.position = at;
    b.cast_shadow = casts;
    b
}

/// A renderer under test: a scene and camera in, RGBA out.
type Render<'a> = &'a mut dyn FnMut(&mut Scene, &PerspectiveCamera) -> Vec<u8>;

/// Each pixel's kept light, against the same frame with nothing casting.
fn kept(render: Render, build: &dyn Fn(bool) -> (Scene, PerspectiveCamera)) -> Vec<f64> {
    let (mut lit, cam) = build(false);
    let (mut shadowed, _) = build(true);
    let a = render(&mut lit, &cam);
    let b = render(&mut shadowed, &cam);
    let lin = |v: u8| {
        let c = v as f64 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    a.chunks(4)
        .zip(b.chunks(4))
        .map(|(p, q)| {
            let (x, y) = (lin(p[1]), lin(q[1]));
            if x < 1e-3 {
                1.0
            } else {
                (y / x).min(1.0)
            }
        })
        .collect()
}

fn partial(k: &[f64], region: impl Fn(u32, u32) -> bool) -> usize {
    (0..N)
        .flat_map(|y| (0..N).map(move |x| (x, y)))
        .filter(|&(x, y)| region(x, y))
        .filter(|&(x, y)| {
            let v = k[(y * N + x) as usize];
            v > 0.1 && v < 0.9
        })
        .count()
}

fn shadowed(k: &[f64]) -> usize {
    k.iter().filter(|&&v| v < 0.5).count()
}

/// A box on the floor under a steep sun, seen from above.
fn box_scene(kind: ShadowType, casts: bool) -> (Scene, PerspectiveCamera) {
    let mut s = Scene::new();
    s.background = Color::BLACK;
    s.add_light(AmbientLight::new(Color::WHITE, 0.2));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.cast_shadow = true;
    sun.shadow.map_size = 512;
    sun.shadow.camera_size = 6.0;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 40.0;
    sun.shadow.normal_bias = 0.02;
    sun.shadow.radius = 2.0;
    sun.shadow.filter = Some(kind);
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(4.0, 10.0, 3.0);
    s.add(l);
    s.add(floor());
    s.add(caster(
        Vector3::new(2.0, 2.0, 2.0),
        Vector3::new(0.0, 1.0, 0.0),
        casts,
    ));
    let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 12.0, 0.01);
    cam.look_at(Vector3::ZERO);
    (s, cam)
}

fn filters_order_by_edge_width(render: Render) {
    let mut edge = |kind| {
        let k = kept(&mut *render, &|casts| box_scene(kind, casts));
        (partial(&k, |_, _| true), shadowed(&k))
    };
    let (basic, b_area) = edge(ShadowType::Basic);
    let (pcf, p_area) = edge(ShadowType::Pcf);
    let (soft, s_area) = edge(ShadowType::PcfSoft);
    let (pcss, c_area) = edge(ShadowType::Pcss);
    eprintln!("edge pixels: basic {basic}, pcf {pcf}, soft {soft}, pcss {pcss}");
    eprintln!("shadowed: basic {b_area}, pcf {p_area}, soft {s_area}, pcss {c_area}");
    assert!(b_area > 300, "no shadow to measure ({b_area} px)");
    assert!(basic < pcf, "basic {basic} is no harder than PCF {pcf}");
    assert!(pcf < soft, "soft PCF {soft} is no softer than PCF {pcf}");
    // The same shadow, only its edge differs.
    for (name, area) in [("pcf", p_area), ("soft", s_area), ("pcss", c_area)] {
        let ratio = area as f64 / b_area as f64;
        assert!(
            (0.8..1.25).contains(&ratio),
            "{name}: {area} shadowed px against basic's {b_area}"
        );
    }
}

/// A tall pole under a low sun: its shadow runs from its foot (contact) out
/// to the shadow of its tip, far from the pole.
fn pole_scene(kind: ShadowType, casts: bool) -> (Scene, PerspectiveCamera) {
    let mut s = Scene::new();
    s.background = Color::BLACK;
    s.add_light(AmbientLight::new(Color::WHITE, 0.2));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.camera_size = 8.0;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 40.0;
    sun.shadow.normal_bias = 0.02;
    sun.shadow.filter = Some(kind);
    sun.shadow.light_size = 0.08;
    let mut l = Object3D::light(Light::Directional(sun));
    // Shadow falls along +x.
    l.position = Vector3::new(-6.0, 6.0, 0.0);
    s.add(l);
    s.add(floor());
    s.add(caster(
        Vector3::new(0.3, 5.0, 0.3),
        Vector3::new(-3.0, 2.5, 0.0),
        casts,
    ));
    let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 12.0, 0.01);
    cam.look_at(Vector3::ZERO);
    (s, cam)
}

fn pcss_is_sharp_at_contact_and_soft_far_off(render: Render) {
    // The pole stands at x = −3; its shadow runs to x = +2. Columns near the
    // foot and near the far end, a band around the shadow's centre line.
    let px = |x: f32| ((x / (2.0 * 12.0 * (25f32).to_radians().tan()) + 0.5) * N as f32) as u32;
    let (foot, tip) = ((px(-2.7), px(-1.7)), (px(0.8), px(1.8)));
    let band = |cols: (u32, u32)| {
        move |x: u32, y: u32| x >= cols.0 && x < cols.1 && y > N / 2 - 30 && y < N / 2 + 30
    };
    let mut widths = |kind| {
        let k = kept(&mut *render, &|casts| pole_scene(kind, casts));
        (partial(&k, band(foot)), partial(&k, band(tip)))
    };
    let (pcf_foot, pcf_tip) = widths(ShadowType::Pcf);
    let (pcss_foot, pcss_tip) = widths(ShadowType::Pcss);
    eprintln!("edge px near the foot / far end: pcf {pcf_foot} / {pcf_tip}, pcss {pcss_foot} / {pcss_tip}");
    assert!(
        pcss_tip > pcf_tip,
        "PCSS is no softer far from the caster ({pcss_tip} against PCF's {pcf_tip})"
    );
    assert!(
        pcss_tip as f64 > 1.5 * pcss_foot.max(1) as f64,
        "PCSS does not harden toward contact: {pcss_foot} edge px at the foot, {pcss_tip} far off"
    );
}

/// A long view over the floor, shadows wanted out to 120 units, and a small
/// box just in front of the camera.
fn reach_scene(cascades: u32, tight: bool, casts: bool) -> (Scene, PerspectiveCamera) {
    let mut s = Scene::new();
    s.background = Color::BLACK;
    s.add_light(AmbientLight::new(Color::WHITE, 0.2));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.cast_shadow = true;
    sun.shadow.map_size = 1024;
    sun.shadow.normal_bias = 0.01;
    sun.shadow.radius = 0.0;
    if tight {
        // The reference: one map fitted closely around the box.
        sun.shadow.camera_size = 2.0;
        sun.shadow.camera_near = 0.5;
        sun.shadow.camera_far = 60.0;
    } else {
        // One map over the whole reach, or cascades.
        sun.shadow.camera_size = 120.0;
        sun.shadow.camera_near = 0.5;
        sun.shadow.camera_far = 400.0;
        sun.shadow.cascades = cascades;
        sun.shadow.cascade_distance = 120.0;
    }
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(3.0, 10.0, 2.0);
    s.add(l);
    s.add(floor());
    s.add(caster(
        Vector3::new(0.4, 0.4, 0.4),
        Vector3::new(0.0, 0.2, 0.0),
        casts,
    ));
    let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 200.0);
    cam.position = Vector3::new(0.0, 1.5, 2.5);
    cam.look_at(Vector3::new(0.0, 0.0, -2.0));
    (s, cam)
}

fn cascades_sharpen_the_near_shadow_and_keep_it_in_place(render: Render) {
    let reference = kept(&mut *render, &|casts| reach_scene(1, true, casts));
    let single = kept(&mut *render, &|casts| reach_scene(1, false, casts));
    let cascaded = kept(&mut *render, &|casts| reach_scene(4, false, casts));
    let mask = |k: &[f64]| k.iter().map(|&v| v < 0.5).collect::<Vec<bool>>();
    let iou = |a: &[bool], b: &[bool]| {
        let inter = a.iter().zip(b).filter(|(x, y)| **x && **y).count();
        let union = a.iter().zip(b).filter(|(x, y)| **x || **y).count();
        inter as f64 / union.max(1) as f64
    };
    let (r, s1, c4) = (mask(&reference), mask(&single), mask(&cascaded));
    let (iou_single, iou_cascaded) = (iou(&r, &s1), iou(&r, &c4));
    eprintln!(
        "shadowed px: reference {}, one map {}, four cascades {}; overlap with the reference: {iou_single:.3}, {iou_cascaded:.3}",
        r.iter().filter(|&&v| v).count(),
        s1.iter().filter(|&&v| v).count(),
        c4.iter().filter(|&&v| v).count()
    );
    assert!(
        r.iter().filter(|&&v| v).count() > 100,
        "no reference shadow to compare with"
    );
    assert!(
        iou_cascaded > 0.8,
        "the cascaded shadow is not where the fitted map puts it ({iou_cascaded:.3})"
    );
    assert!(
        iou_cascaded > iou_single + 0.1,
        "four cascades ({iou_cascaded:.3}) are no closer to the fitted map than one stretched map ({iou_single:.3})"
    );
}

/// Every check, through `render`.
fn all(render: Render) {
    filters_order_by_edge_width(&mut *render);
    pcss_is_sharp_at_contact_and_soft_far_off(&mut *render);
    cascades_sharpen_the_near_shadow_and_keep_it_in_place(&mut *render);
}

#[test]
fn shadow_filters_and_cascades() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    all(&mut |scene, cam| hr.render_to_rgba(scene, cam));
}

#[cfg(feature = "metal")]
#[test]
fn shadow_filters_and_cascades_on_metal() {
    use threers::metal::{MetalError, MetalHeadlessRenderer};
    let mut r = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(r) => r,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    all(&mut |scene, cam| r.render_to_rgba(scene, cam).expect("metal"));
}
