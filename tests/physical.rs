//! MeshPhysicalMaterial's dielectric reflectance on wgpu, against three.js
//! r165's formulas — see `tests/support/physical_specular.rs`. Skipped without
//! a GPU adapter.

#[path = "support/physical_specular.rs"]
mod support;

#[test]
fn ior_and_specular_follow_three_js() {
    let Ok(mut r) = threers::HeadlessRenderer::builder()
        .size(support::N, support::N)
        .build()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = support::check("wgpu", &mut |s, c| r.render_to_rgba(s, c));
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[path = "support/physical_maps.rs"]
mod maps;

/// Each MeshPhysicalMaterial map draws what its scalar does with the texel
/// folded in — see `tests/support/physical_maps.rs`.
#[test]
fn physical_maps_scale_their_scalars_as_three_js_does() {
    let Ok(mut r) = threers::HeadlessRenderer::builder()
        .size(maps::N, maps::N)
        .build()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = maps::check("wgpu", &mut |s, c| r.render_to_rgba(s, c));
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// An InstancedMesh of one identity instance draws a transmissive material as
/// the Mesh it stands for does — the transmission capture is taken for it, so
/// the scene behind refracts through rather than reading the empty stand-in.
#[test]
fn an_instanced_transmissive_mesh_draws_as_the_mesh_does() {
    use threers::cameras::PerspectiveCamera;
    use threers::core::{InstancedMesh, Mesh, Object3D};
    use threers::geometries::{BoxGeometry, SphereGeometry};
    use threers::lights::DirectionalLight;
    use threers::materials::{PhysicalMaterial, StandardMaterial};
    use threers::math::{Color, Vector3};
    use threers::scene::Scene;

    const S: u32 = 96;
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(S, S).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut cam = PerspectiveCamera::new(35.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    let build = |instanced: bool| {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        s.add_light(DirectionalLight::new(Color::WHITE, 2.0).with_direction(Vector3::new(-1.0, -1.0, -2.0)));
        let mut back = StandardMaterial::new(Color::from_hex(0xd04020));
        back.emissive = Color::from_hex(0xd04020);
        let mut b = Object3D::mesh(Mesh::new(BoxGeometry::new(1.2, 6.0, 0.2), back.into()));
        b.position = Vector3::new(0.0, 0.0, -2.0);
        s.add(b);
        let mut glass = PhysicalMaterial::new(Color::WHITE);
        glass.roughness = 0.1;
        glass.transmission = 1.0;
        glass.thickness = 1.0;
        let geometry = SphereGeometry::new(1.1, 48, 24);
        s.add(if instanced {
            Object3D::instanced_mesh(InstancedMesh::new(geometry, glass.into(), 1))
        } else {
            Object3D::mesh(Mesh::new(geometry, glass.into()))
        });
        s
    };
    let mesh = r.render_to_rgba(&mut build(false), &cam);
    let inst = r.render_to_rgba(&mut build(true), &cam);
    // The bar behind shows through the glass, refracted: red at the centre.
    let c = ((S / 2 * S + S / 2) * 4) as usize;
    assert!(mesh[c] > 120 && mesh[c] > 2 * mesh[c + 2], "mesh centre {:?}", &mesh[c..c + 3]);
    let worst = mesh
        .chunks(4)
        .zip(inst.chunks(4))
        .map(|(a, b)| (0..3).map(|k| (a[k] as i32 - b[k] as i32).abs()).max().unwrap())
        .max()
        .unwrap();
    assert!(worst <= 1, "the instanced glass is {worst} levels from the mesh");
}
