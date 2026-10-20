use super::*;

fn camera_at(pos: V3) -> CameraState {
    // A perspective camera at `pos` looking at the origin, fov 50, aspect 1,
    // near 0.1, far 100.
    let mut m = math::IDENTITY_M4;
    math::look_at(&mut m, pos, [0.0; 3], [0.0, 1.0, 0.0]);
    let q = math::quat_from_rotation_matrix(&m);
    let world = math::compose(pos, q, [1.0; 3]);
    let (fov, aspect, near, far) = (50.0_f64, 1.0_f64, 0.1_f64, 100.0_f64);
    let top = near * (std::f64::consts::PI / 180.0 * 0.5 * fov).tan();
    let height = 2.0 * top;
    let width = aspect * height;
    let left = -0.5 * width;
    let (right, bottom) = (left + width, top - height);
    let x = 2.0 * near / (right - left);
    let y = 2.0 * near / (top - bottom);
    let a = (right + left) / (right - left);
    let b = (top + bottom) / (top - bottom);
    let c = -(far + near) / (far - near);
    let d = -2.0 * far * near / (far - near);
    let proj = [
        x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, a, b, c, -1.0, 0.0, 0.0, d, 0.0,
    ];
    CameraState {
        matrix_world: world,
        projection_matrix_inverse: math::invert(&proj),
        fov,
        near,
        far,
        ..CameraState::default()
    }
}

fn object_at(p: V3) -> ObjectState {
    ObjectState {
        position: p,
        quaternion: math::IDENTITY_Q,
        scale: [1.0; 3],
        matrix_world: math::compose(p, math::IDENTITY_Q, [1.0; 3]),
        parent_matrix_world: Some(math::IDENTITY_M4),
    }
}

/// Project a world point to NDC through `camera`.
fn ndc(camera: &CameraState, p: V3) -> (f64, f64) {
    let view = math::invert(&camera.matrix_world);
    let proj = math::invert(&camera.projection_matrix_inverse);
    let v = math::apply_m4(math::apply_m4(p, &view), &proj);
    (v[0], v[1])
}

#[test]
fn handle_counts_match_three() {
    let h = gizmo::handles();
    let count = |g: Group, m: Mode| h.iter().filter(|x| x.group == g && x.mode == m).count();
    assert_eq!(count(Group::Gizmo, Mode::Translate), 13);
    assert_eq!(count(Group::Picker, Mode::Translate), 10);
    assert_eq!(count(Group::Helper, Mode::Translate), 6);
    assert_eq!(count(Group::Gizmo, Mode::Rotate), 5);
    assert_eq!(count(Group::Picker, Mode::Rotate), 5);
    assert_eq!(count(Group::Helper, Mode::Rotate), 1);
    assert_eq!(count(Group::Gizmo, Mode::Scale), 13);
    assert_eq!(count(Group::Picker, Mode::Scale), 10);
    assert_eq!(count(Group::Helper, Mode::Scale), 3);
    // CylinderGeometry(0, 0.04, 0.1, 12): torso 2×13, bottom cap 12+13.
    // Entries go in last first: X is [line, arrow(-0.5), arrow(0.5)].
    let arrow = &h[1].geometry;
    assert_eq!(arrow.position.len() / 3, 26 + 25);
    // OctahedronGeometry(r, 0): 8 faces, non-indexed.
    let octa = gizmo::octahedron(0.1, 0);
    assert_eq!(octa.position.len(), 8 * 9);
    assert!(octa.index.is_none());
    assert_eq!(gizmo::octahedron(0.01, 2).position.len(), 8 * 9 * 9);
}

#[test]
fn translate_x_follows_the_pointer_along_x() {
    let cam = camera_at([3.0, 4.0, 5.0]);
    let obj = object_at([0.0; 3]);
    let mut tc = TransformControls::new();
    tc.set_has_object(true);
    tc.update_matrix_world(Some(&obj), &cam, &math::IDENTITY_M4);
    tc.pointer_hover(Some("X"));
    tc.update_matrix_world(Some(&obj), &cam, &math::IDENTITY_M4);
    let (x0, y0) = ndc(&cam, [0.3, 0.0, 0.0]);
    let ev = tc.pointer_down(
        Some(Pointer {
            x: x0,
            y: y0,
            button: 0,
        }),
        Some(&obj),
        &cam,
    );
    assert!(ev.contains(&TcEvent::MouseDown("translate".into())));
    assert!(tc.dragging());
    let (x1, y1) = ndc(&cam, [1.3, 0.0, 0.0]);
    let ev = tc.pointer_move(
        Some(Pointer {
            x: x1,
            y: y1,
            button: -1,
        }),
        Some(&obj),
        &cam,
    );
    let p = ev
        .iter()
        .find_map(|e| match e {
            TcEvent::SetPosition(p) => Some(*p),
            _ => None,
        })
        .expect("moved");
    assert!((p[0] - 1.0).abs() < 1e-9, "{p:?}");
    assert_eq!(p[1], 0.0);
    assert_eq!(p[2], 0.0);
    let ev = tc.pointer_up(Some(Pointer {
        x: x1,
        y: y1,
        button: 0,
    }));
    assert!(ev.contains(&TcEvent::MouseUp("translate".into())));
    assert!(!tc.dragging());
    assert_eq!(tc.axis(), None);
}

#[test]
fn translation_snap_rounds_in_world_space() {
    let cam = camera_at([3.0, 4.0, 5.0]);
    let obj = object_at([0.0; 3]);
    let mut tc = TransformControls::new();
    tc.set_has_object(true);
    tc.set_property("translationSnap", PropValue::Num(0.5));
    tc.pointer_hover(Some("XZ"));
    tc.update_matrix_world(Some(&obj), &cam, &math::IDENTITY_M4);
    let (x0, y0) = ndc(&cam, [0.15, 0.0, 0.15]);
    tc.pointer_down(
        Some(Pointer {
            x: x0,
            y: y0,
            button: 0,
        }),
        Some(&obj),
        &cam,
    );
    let (x1, y1) = ndc(&cam, [0.9, 0.0, -0.6]);
    let ev = tc.pointer_move(
        Some(Pointer {
            x: x1,
            y: y1,
            button: -1,
        }),
        Some(&obj),
        &cam,
    );
    let p = ev
        .iter()
        .find_map(|e| match e {
            TcEvent::SetPosition(p) => Some(*p),
            _ => None,
        })
        .unwrap();
    assert_eq!(p, [0.5, 0.0, -0.5]);
}

#[test]
fn property_events_fire_only_on_change() {
    let mut tc = TransformControls::new();
    assert!(tc
        .set_property("mode", PropValue::Str("translate".into()))
        .is_empty());
    let ev = tc.set_property("mode", PropValue::Str("rotate".into()));
    assert_eq!(
        ev,
        vec![TcEvent::Prop("mode", PropValue::Str("rotate".into()))]
    );
    assert_eq!(tc.mode(), "rotate");
}

#[test]
fn js_round_matches_math_round() {
    assert_eq!(js_round(2.5), 3.0);
    assert_eq!(js_round(-2.5), -2.0);
    assert_eq!(js_round(0.49999999999999994), 0.0);
    assert_eq!(js_round(-0.6), -1.0);
}
