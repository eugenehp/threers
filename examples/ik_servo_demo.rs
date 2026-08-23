//! Visual demo: three identical arms track the same IK path under different
//! transmissions — direct drive, QDD (15:1), and a high-ratio servo (288:1).
//!
//! Ideal IK is drawn as a translucent ghost; the solid arm is what the servo
//! physics actually reached. An error bar grows with tip miss. Writes a contact
//! sheet and a mid-run hero frame.
//!
//! ```text
//! cargo run --release --example ik_servo_demo [-- out/ik-servo-demo.png]
//! ```

use threers::kinematics::{
    sim::{IkServoSim, Waypoint},
    GearTrain, Goal, Ik, PlanarArm, SimReport,
};
use threers::{
    decode_png, encode_png, AmbientLight, BoxGeometry, BufferAttribute, BufferGeometry, Color,
    CylinderGeometry, DirectionalLight, Euler, HeadlessRenderer, LineBasicMaterial, LineSegments,
    Mesh, Object3D, ObjectId, PerspectiveCamera, Quaternion, Scene, SphereGeometry,
    StandardMaterial, Vector3,
};

const DOWN: [f64; 3] = [0.0, 0.0, -1.0];
/// mm → scene units
const SCALE: f32 = 1.0 / 120.0;

fn out_arg(default: &str) -> String {
    std::env::args()
        .skip_while(|a| a != "--")
        .nth(1)
        .unwrap_or_else(|| default.into())
}

fn waypoints() -> Vec<Waypoint> {
    // Fast transit then settle — acceleration (not steady ramp lag) shows N².
    let mut wps = Vec::new();
    for i in 0..=8 {
        let u = i as f64 / 8.0;
        wps.push(Waypoint {
            t: u * 0.15,
            goal: Goal::PointAlong {
                at: [280.0 + 140.0 * u, 0.0, 220.0],
                along: DOWN,
            },
        });
    }
    wps.push(Waypoint {
        t: 0.9,
        goal: Goal::PointAlong {
            at: [420.0, 0.0, 220.0],
            along: DOWN,
        },
    });
    wps
}

fn ik() -> Ik {
    let chain = threers::kinematics::SerialChain::planar_3r();
    Ik {
        limits: (-179.0, 179.0),
        max_iters: 120,
        joint_limits: chain.joints.iter().map(|j| j.limits).collect(),
        ..Default::default()
    }
}

fn seed() -> [f64; 3] {
    [-2.8, 88.9, 93.9]
}

fn v(p: [f64; 3]) -> Vector3 {
    Vector3::new(
        (p[0] as f32) * SCALE,
        (p[1] as f32) * SCALE,
        (p[2] as f32) * SCALE,
    )
}

fn lights(scene: &mut Scene) {
    scene.background = Color::new(0.07, 0.08, 0.10);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.45));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.0)
            .with_direction(Vector3::new(-0.4, -0.85, -0.35).normalize()),
    );
    scene.add_light(
        DirectionalLight::new(Color::new(0.45, 0.55, 0.9), 0.55)
            .with_direction(Vector3::new(0.7, 0.1, 0.6).normalize()),
    );
}

fn mat(color: Color, rough: f32, metal: f32) -> StandardMaterial {
    let mut m = StandardMaterial::new(color);
    m.roughness = rough;
    m.metalness = metal;
    m
}

/// Cylinder along +Y of length `len`, then rotate so +Y maps onto `dir`.
fn link_between(
    scene: &mut Scene,
    parent: ObjectId,
    a: Vector3,
    b: Vector3,
    radius: f32,
    color: Color,
) {
    let mid = Vector3::new(
        0.5 * (a.x + b.x),
        0.5 * (a.y + b.y),
        0.5 * (a.z + b.z),
    );
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let dz = b.z - a.z;
    let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-4);
    let geom = CylinderGeometry::new(radius, radius, len, 16, 1, false, 0.0, std::f32::consts::TAU);
    let mut obj = Object3D::mesh(Mesh::new(geom, mat(color, 0.45, 0.15).into()));
    obj.position = mid;
    // Cylinder default is +Y; map +Y → unit direction of the link.
    let ux = dx / len;
    let uy = dy / len;
    let uz = dz / len;
    // Quaternion taking (0,1,0) to (ux,uy,uz).
    let dot = uy; // (0,1,0)·u
    if (dot - 1.0).abs() < 1e-5 {
        obj.quaternion = Euler::new(0.0, 0.0, 0.0).to_quaternion();
    } else if (dot + 1.0).abs() < 1e-5 {
        obj.quaternion = Euler::new(std::f32::consts::PI, 0.0, 0.0).to_quaternion();
    } else {
        // axis = (0,1,0) × u = (uz, 0, -ux)
        let ax = uz;
        let ay = 0.0;
        let az = -ux;
        let alen = (ax * ax + ay * ay + az * az).sqrt();
        let angle = dot.clamp(-1.0, 1.0).acos();
        obj.quaternion = Quaternion::from_axis_angle(
            Vector3::new(ax / alen, ay / alen, az / alen),
            angle,
        );
    }
    scene.add_to(parent, obj);
}

fn ball(scene: &mut Scene, parent: ObjectId, at: Vector3, r: f32, color: Color) {
    let geom = SphereGeometry::new(r, 20, 12);
    let mut obj = Object3D::mesh(Mesh::new(geom, mat(color, 0.35, 0.2).into()));
    obj.position = at;
    scene.add_to(parent, obj);
}

fn error_line(scene: &mut Scene, parent: ObjectId, a: Vector3, b: Vector3, color: Color) {
    let mut positions = Vec::with_capacity(6);
    positions.extend_from_slice(&[a.x, a.y, a.z, b.x, b.y, b.z]);
    let mut geom = BufferGeometry::new();
    geom.set_attribute("position", BufferAttribute::new(positions, 3));
    let mut lm = LineBasicMaterial::new(color);
    lm.line_width = 2.5;
    scene.add_to(
        parent,
        Object3D::line_segments(LineSegments::new(geom, lm.into())),
    );
}

fn pedestal(scene: &mut Scene, parent: ObjectId, color: Color) {
    let geom = BoxGeometry::new(2.4, 0.1, 2.4);
    let mut obj = Object3D::mesh(Mesh::new(geom, mat(color, 0.7, 0.05).into()));
    // In the arm's local frame (before the -90° X tip), floor is −Z.
    obj.position = Vector3::new(0.0, 0.0, -0.05);
    scene.add_to(parent, obj);
}

struct ArmView {
    label: &'static str,
    color: Color,
    ghost: Color,
    q_act: Vec<f64>,
    q_cmd: Vec<f64>,
    target: [f64; 3],
    tip_err: f64,
}

fn hinge_pin(scene: &mut Scene, parent: ObjectId, origin: [f64; 3], axis: [f64; 3]) {
    // Joint *module*: housing cylinder along the revolute axis (industrial /
    // URDF style), plus flanges and a thin axis cue. Link beams are separate.
    let o = v(origin);
    let ax = Vector3::new(axis[0] as f32, axis[1] as f32, axis[2] as f32).normalize();
    let along = |s: f32| Vector3::new(o.x + ax.x * s, o.y + ax.y * s, o.z + ax.z * s);
    let house = 0.11_f32;
    let housing = Color::new(0.42, 0.46, 0.52);
    let flange = Color::new(0.62, 0.66, 0.72);
    let cue = Color::new(1.0, 0.8, 0.35);
    link_between(scene, parent, along(-house), along(house), 0.13, housing);
    link_between(scene, parent, along(-(house + 0.036)), along(-(house + 0.004)), 0.155, flange);
    link_between(scene, parent, along(house + 0.004), along(house + 0.036), 0.155, flange);
    link_between(scene, parent, along(-(house + 0.1)), along(house + 0.1), 0.014, cue);
}

fn add_arm(scene: &mut Scene, arm: &PlanarArm, view: &ArmView, at: Vector3) {
    let mut root = Object3D::group();
    root.position = at;
    root.quaternion = Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
    let root_id = scene.add(root);
    pedestal(scene, root_id, Color::new(0.12, 0.13, 0.15));

    let cmd = arm.chain.poses(&view.q_cmd);
    let act = arm.chain.poses(&view.q_act);

    let ghost = Color::new(
        (view.ghost.r * 0.5 + 0.55).min(1.0),
        (view.ghost.g * 0.5 + 0.55).min(1.0),
        (view.ghost.b * 0.5 + 0.6).min(1.0),
    );
    for p in &cmd {
        link_between(scene, root_id, v(p.origin), v(p.distal), 0.032, ghost);
    }

    let pin = Color::new(0.72, 0.74, 0.78);
    let _ = pin;
    let standoff = 0.18_f32;
    for p in &act {
        let o = v(p.origin);
        let d = v(p.distal);
        let dx = d.x - o.x;
        let dy = d.y - o.y;
        let dz = d.z - o.z;
        let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-4);
        let ux = dx / len;
        let uy = dy / len;
        let uz = dz / len;
        let a = Vector3::new(
            o.x + ux * standoff,
            o.y + uy * standoff,
            o.z + uz * standoff,
        );
        let b = Vector3::new(
            d.x - ux * standoff * 0.65,
            d.y - uy * standoff * 0.65,
            d.z - uz * standoff * 0.65,
        );
        link_between(scene, root_id, a, b, 0.065, view.color);
        hinge_pin(scene, root_id, p.origin, p.axis);
    }
    if let Some(tip) = act.last() {
        ball(scene, root_id, v(tip.distal), 0.11, view.color);
    }

    let tgt = v(view.target);
    ball(scene, root_id, tgt, 0.12, Color::new(0.95, 0.35, 0.2));
    if let Some(tip) = act.last() {
        error_line(scene, root_id, v(tip.distal), tgt, Color::new(1.0, 0.55, 0.15));
    }

    let bar_h = (view.tip_err as f32 * SCALE * 25.0).clamp(0.08, 3.0);
    let bar = BoxGeometry::new(0.12, 0.12, bar_h);
    let mut bar_obj = Object3D::mesh(Mesh::new(
        bar,
        mat(Color::new(0.95, 0.4, 0.2), 0.5, 0.1).into(),
    ));
    bar_obj.position = Vector3::new(-1.4, 0.0, bar_h * 0.5);
    scene.add_to(root_id, bar_obj);
    let _ = view.label;
}

fn render_panel(
    out: &str,
    arm: &PlanarArm,
    views: &[ArmView],
    cam_pos: Vector3,
    look: Vector3,
    w: u32,
    h: u32,
) {
    let mut renderer = match HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e})");
            std::process::exit(2);
        }
    };
    let (rw, rh) = renderer.render_size();
    let mut scene = Scene::new();
    lights(&mut scene);

    let span = 5.2_f32;
    let origin = -0.5 * (views.len() as f32 - 1.0) * span;
    for (i, view) in views.iter().enumerate() {
        add_arm(
            &mut scene,
            arm,
            view,
            Vector3::new(origin + i as f32 * span, 0.0, 0.0),
        );
    }

    let mut camera = PerspectiveCamera::new(42.0, rw as f32 / rh as f32, 0.1, 100.0);
    camera.position = cam_pos;
    camera.look_at(look);

    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    if let Err(e) = std::fs::write(out, encode_png(rw, rh, &rgba)) {
        eprintln!("write {out}: {e}");
    } else {
        println!("wrote {out} ({rw}×{rh})");
    }
}

fn views_at(arm: &PlanarArm, reports: &[SimReport; 3], t: f64) -> Vec<ArmView> {
    let colors = [
        (Color::new(0.35, 0.75, 0.95), Color::new(0.25, 0.4, 0.5)), // direct — cyan
        (Color::new(0.45, 0.85, 0.55), Color::new(0.3, 0.45, 0.32)), // qdd — green
        (Color::new(0.95, 0.55, 0.35), Color::new(0.5, 0.32, 0.25)), // high — orange
    ];
    reports
        .iter()
        .zip(colors)
        .map(|(r, (color, ghost))| {
            let f = r.frame_at(t).expect("frame");
            let (tip, _) = arm.tool(&f.q_act);
            // Recover target from tip + position error direction is messy; store
            // goal from cmd tool instead.
            let (cmd_tip, _) = arm.tool(&f.q_cmd);
            let _ = tip;
            ArmView {
                label: r.transmission,
                color,
                ghost,
                q_act: f.q_act.clone(),
                q_cmd: f.q_cmd.clone(),
                target: cmd_tip,
                tip_err: f.position_err,
            }
        })
        .collect()
}

fn contact_strip(paths: &[&str], out: &str, cell_w: u32, cell_h: u32) {
    let imgs: Vec<_> = paths
        .iter()
        .map(|p| {
            let bytes = std::fs::read(p).expect(p);
            decode_png(&bytes).expect(p)
        })
        .collect();
    let n = imgs.len() as u32;
    let w = cell_w * n;
    let h = cell_h;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for (i, img) in imgs.iter().enumerate() {
        let ox = i as u32 * cell_w;
        for y in 0..cell_h.min(img.height) {
            for x in 0..cell_w.min(img.width) {
                let si = ((y * img.width + x) * 4) as usize;
                let di = ((y * w + ox + x) * 4) as usize;
                rgba[di..di + 4].copy_from_slice(&img.rgba[si..si + 4]);
            }
        }
    }
    std::fs::write(out, encode_png(w, h, &rgba)).expect("strip");
    println!("wrote {out} ({w}×{h})");
}

fn main() {
    let out = out_arg("out/ik-servo-demo.png");
    let _ = std::fs::create_dir_all(
        std::path::Path::new(&out)
            .parent()
            .unwrap_or(std::path::Path::new(".")),
    );

    let arm = PlanarArm::three_link();
    let wps = waypoints();
    let s = seed();

    println!("simulating coupled plant — direct / QDD 15:1 / servo 288:1 …");
    let reports = IkServoSim::compare_plant(ik(), &s, &wps, arm.chain.clone());
    for r in &reports {
        r.print_summary();
    }

    // Hero frame mid-settle, where high-ratio lag is still visible.
    let hero_t = 0.35;
    let views = views_at(&arm, &reports, hero_t);
    render_panel(
        &out,
        &arm,
        &views,
        Vector3::new(0.0, 2.8, 11.0),
        Vector3::new(0.0, 2.2, 1.0),
        1920,
        900,
    );

    // Sequence strip at four times.
    let times = [0.08, 0.2, 0.4, 0.7];
    let mut frame_paths = Vec::new();
    for (i, &t) in times.iter().enumerate() {
        let path = out.replace(".png", &format!("-t{i}.png"));
        let views = views_at(&arm, &reports, t);
        render_panel(
            &path,
            &arm,
            &views,
            Vector3::new(0.0, 2.6, 10.5),
            Vector3::new(0.0, 2.0, 1.0),
            1280,
            640,
        );
        frame_paths.push(path);
    }
    let strip = out.replace(".png", "-strip.png");
    let refs: Vec<&str> = frame_paths.iter().map(|s| s.as_str()).collect();
    contact_strip(&refs, &strip, 1280, 640);

    // Reflected-inertia callout (no render — console).
    println!("\nreflected inertia J = N² · J_rotor:");
    for (label, train) in [
        ("direct", GearTrain::direct_drive()),
        ("qdd-15:1", GearTrain::qdd()),
        ("servo-288:1", GearTrain::high_ratio_servo()),
    ] {
        println!(
            "  {label:<12} N={:<5.0}  J_ref={:.4e} kg·m²  SNR@10mNm={:.2}",
            train.ratio,
            train.reflected_inertia(),
            train.force_snr(0.01)
        );
    }

    let high = &reports[2];
    let direct = &reports[0];
    println!(
        "\nhigh-ratio worst tool error {:.1}× direct ({:.2} mm vs {:.2} mm); mean joint {:.1}× ({:.2}° vs {:.2}°)",
        high.worst_position_err / direct.worst_position_err.max(1e-6),
        high.worst_position_err,
        direct.worst_position_err,
        high.mean_joint_err / direct.mean_joint_err.max(1e-6),
        high.mean_joint_err,
        direct.mean_joint_err
    );
}
