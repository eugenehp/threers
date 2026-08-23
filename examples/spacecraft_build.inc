// Shared spacecraft assembly, included by the interactive example.

fn mesh_at(geom: threers::BufferGeometry, mat: PhysicalMaterial, pos: Vector3) -> Object3D {
    let mut o = Object3D::mesh(Mesh::new(geom, Material::Physical(mat)));
    o.position = pos;
    // Self-shadowing does a lot of work on a craft like this: without it the
    // solar wings float free of the bus and the whole assembly reads flat.
    o.cast_shadow = true;
    o.receive_shadow = true;
    o
}

/// Assemble the spacecraft. Split out from `run` so the interesting part — which
/// preset goes on which part — reads on its own.
fn build_spacecraft(scene: &mut Scene) {
    let crinkle = crinkle_normal_map(512, 11.0, 2.4);
    let fine_crinkle = crinkle_normal_map(512, 20.0, 1.6);
    let cells = solar_cell_map(512, 5);

    // --- Bus: gold MLI blanket ------------------------------------------
    // The crinkle normal map is what makes this read as foil rather than as a
    // gold-painted box. Without it the panels look like solid metal plate.
    let mut gold = presets::gold_foil();
    gold.normal_map = Some(crinkle.clone());
    gold.normal_scale = threers::Vector2::new(1.6, 1.6);
    gold.roughness = 0.30;
    scene.add(mesh_at(BoxGeometry::new(1.6, 1.3, 1.6), gold, Vector3::ZERO));

    // --- Thruster module: silver / aluminized kapton ---------------------
    let mut silver = presets::silver_foil();
    silver.normal_map = Some(fine_crinkle.clone());
    silver.roughness = 0.22;
    scene.add(mesh_at(
        CylinderGeometry::new(0.42, 0.42, 0.7, 32, 1, false, 0.0, TAU),
        silver,
        Vector3::new(0.0, -0.95, 0.0),
    ));

    // --- Engine bell: heat-tinted titanium -------------------------------
    // Thin-film iridescence on the oxide layer — the hue comes from
    // `iridescence_thickness`, not from the base color.
    let bell = presets::anodized_titanium(320.0).with_roughness(0.22);
    scene.add(mesh_at(
        CylinderGeometry::new(0.34, 0.15, 0.5, 32, 1, true, 0.0, TAU),
        bell,
        Vector3::new(0.0, -1.55, 0.0),
    ));

    // --- Solar arrays: two wings of cells on a backing -------------------
    for side in [-1.0f32, 1.0] {
        // Boom out to the wing.
        let boom = presets::brushed_aluminum(PI / 2.0).with_roughness(0.28);
        let mut boom_o = mesh_at(
            CylinderGeometry::new(0.06, 0.06, 1.1, 16, 1, false, 0.0, TAU),
            boom,
            Vector3::new(side * 1.35, 0.0, 0.0),
        );
        boom_o.rotate_z(PI / 2.0);
        scene.add(boom_o);

        // The cell face. Anisotropy runs along the busbar grid; the streak
        // direction is `anisotropy_rotation`, which needs the UV-derived
        // tangent frame to be meaningful — PlaneGeometry has UVs, so it is.
        let mut panel = presets::solar_cell();
        panel.map = Some(cells.clone());
        panel.anisotropy = 0.7;
        panel.anisotropy_rotation = PI / 2.0;
        let mut panel_o = mesh_at(
            BoxGeometry::new(2.6, 0.04, 1.5),
            panel,
            Vector3::new(side * 3.2, 0.0, 0.0),
        );
        panel_o.rotate_x(0.10); // slight sun-tracking tilt
        scene.add(panel_o);

        // Backing structure behind the cells.
        let backing = presets::array_backing();
        let mut back_o = mesh_at(
            BoxGeometry::new(2.62, 0.03, 1.52),
            backing,
            Vector3::new(side * 3.2, -0.05, 0.0),
        );
        back_o.rotate_x(0.10);
        scene.add(back_o);
    }

    // --- High-gain dish: white thermal paint over an aluminium rim -------
    let dish = presets::white_thermal_paint().with_roughness(0.55);
    let mut dish_o = mesh_at(
        SphereGeometry::new(0.62, 48, 24),
        dish,
        Vector3::new(0.0, 0.95, -0.35),
    );
    dish_o.scale = Vector3::new(1.0, 0.28, 1.0);
    scene.add(dish_o);

    let rim = presets::aluminum().with_roughness(0.3);
    let mut rim_o = mesh_at(TorusGeometry::new(0.62, 0.035, 12, 48, TAU), rim, Vector3::new(0.0, 0.95, -0.35));
    rim_o.rotate_x(PI / 2.0);
    scene.add(rim_o);

    // --- Sensor porthole: glass ------------------------------------------
    // TransparencyMode::Refract does screen-space refraction and needs the
    // environment for its reflection fallback — already set on the scene.
    //
    // Sits clear of the bus face rather than half-buried in it: screen-space
    // refraction marches the captured depth buffer, so a lens sunk into opaque
    // geometry mostly refracts the inside of the box and reads as a solid ball.
    let glass = presets::optical_glass()
        .with_roughness(0.02)
        .with_attenuation(Color::from_hex(0x9fd4e8), 0.9);
    scene.add(mesh_at(
        SphereGeometry::new(0.26, 48, 24),
        glass,
        Vector3::new(0.0, 0.15, 1.30),
    ));

    // Black anti-stray-light baffle ringing the porthole.
    let baffle = presets::black_kapton();
    let baffle_o = mesh_at(TorusGeometry::new(0.34, 0.06, 12, 32, TAU), baffle, Vector3::new(0.0, 0.15, 0.95));
    scene.add(baffle_o);
}

