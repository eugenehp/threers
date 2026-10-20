/**
 * Convert threers parity scene setup (JS) into a native Rust example.
 * Used by generate-rust-scenes.js — output is illustrative; review before copying.
 */

/** Full-file overrides for scenes that need hand-written Rust. */
export const RUST_OVERRIDES = {
    reflector: `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x1a1a22);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.8));

    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 0.5));
    key.position = Vector3::new(1.0, 2.0, 3.0);
    scene.add(key);

    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.6, 0.6, 0.6),
        BasicMaterial::new(Color::from_hex(0xff6644)),
    ));
    cube.position = Vector3::new(0.4, 0.4, 1.0);
    scene.add(cube);

    // \`THREE.Reflector\` in the browser shim renders an offscreen pass each frame
    // and writes the projective texture matrix into \`MirrorMaterial\`. Native Rust:
    // allocate a \`RenderTarget\`, call \`renderer.render_to\`, then assign
    // \`mirror_mat.map\` + \`mirror_mat.texture_matrix\` before the main pass.
    let mirror_mat = MirrorMaterial {
        color: Color::from_hex(0x7f7f7f),
        ..Default::default()
    };
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(4.0, 3.0),
        mirror_mat,
    )));

    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::ZERO);`,
    lathe: `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x1a1a22);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.4));
    let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    dl.position = Vector3::new(2.0, 3.0, 4.0);
    scene.add(dl);
    let points = [
        Vector2::new(0.4, -1.0),
        Vector2::new(0.6, -0.5),
        Vector2::new(0.5, 0.0),
        Vector2::new(0.3, 0.5),
        Vector2::new(0.5, 1.0),
    ];
    let mut lathe = Object3D::mesh(Mesh::new(
        LatheGeometry::new(&points, 16, 0.0, std::f32::consts::TAU),
        StandardMaterial::new(Color::from_hex(0xddaa66)).with_roughness(0.45).with_metalness(0.0).into(),
    ));
    lathe.quaternion = Euler::new(0.0, 0.4, 0.0).to_quaternion();
    scene.add(lathe);
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.3, 4.0);
    camera.look_at(Vector3::ZERO);`,
    'normal-map': `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0a0a0a);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.3));
    let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    dl.position = Vector3::new(2.0, 3.0, 4.0);
    scene.add(dl);
    // Procedural 16×16 tangent-space normal map (matches threers-normal-map.html).
    let tex_size: u32 = 16;
    let mut data = vec![0u8; (tex_size * tex_size * 4) as usize];
    for y in 0..tex_size {
        for x in 0..tex_size {
            let u = (x % 4) as f32 / 3.0 - 0.5;
            let v = (y % 4) as f32 / 3.0 - 0.5;
            let z = (1.0 - u * u - v * v).max(0.0).sqrt();
            let i = ((y * tex_size + x) * 4) as usize;
            data[i] = ((u * 0.5 + 0.5) * 255.0).round() as u8;
            data[i + 1] = ((v * 0.5 + 0.5) * 255.0).round() as u8;
            data[i + 2] = ((z * 0.5 + 0.5) * 255.0).round() as u8;
            data[i + 3] = 255;
        }
    }
    let normal_tex = Arc::new(DataTexture::new(tex_size, tex_size, TextureFormat::Rgba8Unorm, data));
    let mat = StandardMaterial::new(Color::from_hex(0xaaaaaa))
        .with_roughness(0.4)
        .with_metalness(0.0)
        .with_normal_map(normal_tex)
        .into();
    scene.add(Object3D::mesh(Mesh::new(SphereGeometry::new(1.1, 48, 24), mat)));
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.5);
    camera.look_at(Vector3::ZERO);`,
    'many-meshes': `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101418);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.4));
    let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    dl.position = Vector3::new(2.0, 3.0, 4.0);
    scene.add(dl);
    let geom = BoxGeometry::new(0.35, 0.35, 0.35);
    for i in 0..4 {
        for j in 0..4 {
            let hue = (i * 4 + j) as f32 / 16.0;
            let mat = StandardMaterial::new(parity_hsl(hue, 0.6, 0.55))
                .with_roughness(0.5)
                .with_metalness(0.0)
                .into();
            let mut m = Object3D::mesh(Mesh::new(geom.clone(), mat));
            m.position = Vector3::new(i as f32 * 0.6 - 0.9, j as f32 * 0.6 - 0.9, 0.0);
            m.quaternion = Euler::new(i as f32 * 0.3, j as f32 * 0.3, 0.0).to_quaternion();
            scene.add(m);
        }
    }
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 4.5);
    camera.look_at(Vector3::ZERO);`,
    'depth-mat': `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.1, 32, 16),
        DepthMaterial::new().into(),
    )));
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.7, 15.0);
    camera.position = Vector3::new(0.0, 0.0, 3.2);
    camera.look_at(Vector3::ZERO);`,
    shadow: `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101820);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.4));
    let mut dl = Object3D::light({
        let mut l = DirectionalLight::new(Color::from_hex(0xffffff), 1.0);
        l.cast_shadow = true;
        l.shadow.camera_size = 3.0;
        l.shadow.camera_near = 0.1;
        l.shadow.camera_far = 20.0;
        l
    });
    dl.position = Vector3::new(3.0, 5.0, 2.0);
    scene.add(dl);
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xcc4444)).with_roughness(0.5).with_metalness(0.0).into(),
    ));
    cube.cast_shadow = true;
    cube.position = Vector3::new(0.0, 0.5, 0.0);
    scene.add(cube);
    let mut ground = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(6.0, 6.0),
        StandardMaterial::new(Color::from_hex(0xaaaaaa)).with_roughness(0.7).with_metalness(0.0).into(),
    ));
    ground.receive_shadow = true;
    ground.quaternion = Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
    scene.add(ground);
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(2.5, 3.0, 4.0);
    camera.look_at(Vector3::ZERO);`,
    'spot-light': `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x080812);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.05));
    let mut sl = Object3D::light({
        let mut l = SpotLight::new(Color::from_hex(0xffffff), 4.0);
        l.distance = 0.0;
        l.angle = std::f32::consts::PI / 6.0;
        l.penumbra = 0.2;
        l.decay = 1.5;
        l.direction = Vector3::new(-0.42640142, -0.56853527, -0.71066904);
        l
    });
    sl.position = Vector3::new(1.5, 2.0, 2.5);
    scene.add(sl);
    let mut plane = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(3.0, 2.2),
        {
            let mut m = StandardMaterial::new(Color::from_hex(0xddccaa)).with_roughness(0.8).with_metalness(0.0);
            m.side = 2.into();
            m.into()
        },
    ));
    plane.quaternion = Euler::new(-0.2, 0.1, 0.0).to_quaternion();
    scene.add(plane);
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.2, 3.5);
    camera.look_at(Vector3::ZERO);`,
    'spot-shadow': `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0a0a0e);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.15));
    let mut sl = Object3D::light({
        let mut l = SpotLight::new(Color::from_hex(0xffffff), 12.0);
        l.distance = 0.0;
        l.angle = std::f32::consts::PI / 5.0;
        l.penumbra = 0.25;
        l.decay = 1.5;
        l.cast_shadow = true;
        l.direction = Vector3::new(-0.42640142, -0.8528028, -0.42640142);
        l
    });
    sl.position = Vector3::new(2.5, 5.0, 2.5);
    scene.add(sl);
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xeebb44)).with_roughness(0.55).with_metalness(0.0).into(),
    ));
    cube.cast_shadow = true;
    cube.position = Vector3::new(0.0, 0.5, 0.0);
    scene.add(cube);
    let mut ground = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(6.0, 6.0),
        StandardMaterial::new(Color::from_hex(0xbbbbbb)).with_roughness(0.65).with_metalness(0.0).into(),
    ));
    ground.receive_shadow = true;
    ground.quaternion = Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
    scene.add(ground);
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(2.5, 3.0, 4.0);
    camera.look_at(Vector3::new(0.0, 0.4, 0.0));`,
    'multi-shadow': `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x1a1a1f);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.3));
    let mut dl = Object3D::light({
        let mut l = DirectionalLight::new(Color::from_hex(0xffffff), 1.0);
        l.cast_shadow = true;
        l.shadow.camera_size = 4.0;
        l
    });
    dl.position = Vector3::new(3.0, 5.0, 3.0);
    scene.add(dl);
    let mut left = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xcc5555)).with_roughness(0.5).with_metalness(0.0).into(),
    ));
    left.cast_shadow = true;
    left.position = Vector3::new(-1.2, 0.5, 0.0);
    scene.add(left);
    let mut right = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.6, 24, 12),
        StandardMaterial::new(Color::from_hex(0x55cc88)).with_roughness(0.45).with_metalness(0.0).into(),
    ));
    right.cast_shadow = true;
    right.position = Vector3::new(1.2, 0.6, 0.0);
    scene.add(right);
    let mut ground = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(8.0, 8.0),
        StandardMaterial::new(Color::from_hex(0xcccccc)).with_roughness(0.7).with_metalness(0.0).into(),
    ));
    ground.receive_shadow = true;
    ground.quaternion = Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
    scene.add(ground);
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 3.0, 5.0);
    camera.look_at(Vector3::new(0.0, 0.4, 0.0));`,
    physical: `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0c0c10);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.1));
    let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.5));
    dl.position = Vector3::new(2.0, 3.0, 4.0);
    scene.add(dl);
    let mat = {
        let mut m = PhysicalMaterial::new(Color::from_hex(0xe05050));
        m.roughness = 0.25;
        m.metalness = 0.3;
        m.clearcoat = 0.7;
        m.clearcoat_roughness = 0.15;
        Material::Physical(m)
    };
    scene.add(Object3D::mesh(Mesh::new(SphereGeometry::new(1.0, 48, 24), mat)));
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.2, 3.5);
    camera.look_at(Vector3::ZERO);`,
    'shape-geom': `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x181820);
    // ShapeGeometry parity — triangle after winding fix + Earcut (matches three.js).
    let mut geom = BufferGeometry::new();
    geom.set_attribute(
        "position",
        BufferAttribute::new(vec![-0.6, -0.5, 0.0, 0.0, 0.7, 0.0, 0.6, -0.5, 0.0], 3),
    );
    geom.set_attribute(
        "normal",
        BufferAttribute::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3),
    );
    geom.set_index(vec![1, 0, 2]);
    let mut mat = BasicMaterial::new(Color::from_hex(0x77ddaa));
    mat.side = 2; // DoubleSide
    scene.add(Object3D::mesh(Mesh::new(geom, mat)));
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 2.0);
    camera.look_at(Vector3::ZERO);`,
    glitch: `    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101010);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        BasicMaterial::new(Color::from_hex(0x00ff88)),
    )));
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::ZERO);

    // DigitalGlitch parity — same LCG + frame-0 uniforms as three.js GlitchPass (seed 42).
    let mut rng = ParityRng::new(42);
    let dt_size = 64u32;
    let mut height_data = vec![0f32; (dt_size * dt_size) as usize];
    for v in &mut height_data {
        *v = rng.rand_float(0.0, 1.0);
    }
    for _ in 0..16 {
        rng.next_f32();
    }
    let _rand_x = rng.rand_int(120, 240);
    let glitch_seed = rng.next_f32();
    let glitch_amount = rng.next_f32() / 30.0;
    let glitch_angle = rng.rand_float(-std::f32::consts::PI, std::f32::consts::PI);
    let glitch_seed_x = rng.rand_float(-1.0, 1.0);
    let glitch_seed_y = rng.rand_float(-1.0, 1.0);
    let glitch_dist_x = rng.next_f32();
    let glitch_dist_y = rng.next_f32();
    let glitch_byp = 0.0f32;

    renderer.set_glitch_disp(&height_data, dt_size);
    let disp_bake = bake_glitch_disp(config.width, config.height, glitch_seed, &height_data, dt_size);
    renderer.set_glitch_snow(&disp_bake, config.width, config.height);
    let scene_rt = RenderTarget::new(
        &device,
        config.width,
        config.height,
        wgpu::TextureFormat::Rgba16Float,
    );`,
};

/** Extra module items for specific scenes (structs/helpers). */
export const RUST_MODULE_OVERRIDES = {
    'many-meshes': `fn parity_hsl(h: f32, s: f32, l: f32) -> Color {
    fn srgb_to_linear(c: f32) -> f32 {
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    }
    let h = h - h.floor();
    let (r, g, b) = if s == 0.0 {
        (l, l, l)
    } else {
        let q = if l <= 0.5 { l * (1.0 + s) } else { l + s - l * s };
        let p = 2.0 * l - q;
        let hue2rgb = |mut t: f32| {
            if t < 0.0 { t += 1.0; }
            if t > 1.0 { t -= 1.0; }
            if t < 1.0 / 6.0 { p + (q - p) * 6.0 * t }
            else if t < 0.5 { q }
            else if t < 2.0 / 3.0 { p + (q - p) * (2.0 / 3.0 - t) * 6.0 }
            else { p }
        };
        (hue2rgb(h + 1.0 / 3.0), hue2rgb(h), hue2rgb(h - 1.0 / 3.0))
    };
    Color::new(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))
}
`,
    glitch: `struct ParityRng { state: u32 }

impl ParityRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.state as f32 / 4_294_967_296.0
    }
    fn rand_float(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
    fn rand_int(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_f32() * (hi - lo + 1) as f32) as i32
    }
}

fn sample_height_linear(data: &[f32], size: u32, u: f32, v: f32) -> f32 {
    let stx = u * size as f32 - 0.5;
    let sty = v * size as f32 - 0.5;
    let i0x = stx.floor() as i32;
    let i0y = sty.floor() as i32;
    let fx = stx - i0x as f32;
    let fy = sty - i0y as f32;
    let max = size as i32 - 1;
    let load = |ix: i32, iy: i32| {
        let cx = ix.clamp(0, max) as u32;
        let cy = iy.clamp(0, max) as u32;
        data[(cy * size + cx) as usize]
    };
    let c0 = load(i0x, i0y) * (1.0 - fx) + load(i0x + 1, i0y) * fx;
    let c1 = load(i0x, i0y + 1) * (1.0 - fx) + load(i0x + 1, i0y + 1) * fx;
    c0 * (1.0 - fy) + c1 * fy
}

/// Full-screen disp bake — same as \`_glslGlitchDispBake\` / wasm kind 32.
fn bake_glitch_disp(w: u32, h: u32, seed: f32, height: &[f32], dt: u32) -> Vec<f32> {
    let mut out = vec![0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            out[(y * w + x) as usize] =
                sample_height_linear(height, dt, u * seed * seed, v * seed * seed);
        }
    }
    out
}
`,
};

/** Custom draw path inside \`RedrawRequested\` (replaces \`renderer.render(...)\`). */
export const RUST_DRAW_OVERRIDES = {
    glitch: `                                    renderer.render_to(&mut scene, &camera, &scene_rt);
                                    renderer.apply_postfx(
                                        &scene_rt.color_view,
                                        &view,
                                        32,
                                        glitch_seed,
                                        [
                                            glitch_amount,
                                            glitch_angle,
                                            glitch_dist_x,
                                            glitch_dist_y,
                                        ],
                                        [glitch_seed_x, glitch_seed_y, 0.05, glitch_byp],
                                        config.width,
                                        config.height,
                                    );`,
};

const GEOMETRY = {
    BoxGeometry: 'BoxGeometry',
    SphereGeometry: 'SphereGeometry',
    PlaneGeometry: 'PlaneGeometry',
    CircleGeometry: 'CircleGeometry',
    RingGeometry: 'RingGeometry',
    CylinderGeometry: 'CylinderGeometry',
    ConeGeometry: 'ConeGeometry',
    TorusGeometry: 'TorusGeometry',
    TorusKnotGeometry: 'TorusKnotGeometry',
    CapsuleGeometry: 'CapsuleGeometry',
    IcosahedronGeometry: 'IcosahedronGeometry',
    DodecahedronGeometry: 'DodecahedronGeometry',
    TetrahedronGeometry: 'TetrahedronGeometry',
    OctahedronGeometry: 'OctahedronGeometry',
    ParametricGeometry: 'ParametricGeometry',
    LatheGeometry: 'LatheGeometry',
    TubeGeometry: 'TubeGeometry',
    ExtrudeGeometry: 'ExtrudeGeometry',
};

function toRustF32(v) {
    const s = String(v).trim();
    if (/^Math\.PI/.test(s)) {
        return s.replace(/Math\.PI/g, 'std::f32::consts::PI');
    }
    if (/^-?\d+$/.test(s)) return `${s}.0`;
    if (/^-?[\d.]+$/.test(s) && !s.includes('.')) return `${s}.0`;
    return s;
}

function toRustUsize(v) {
    const s = String(v).trim();
    if (/^-?\d+$/.test(s)) return s;
    if (/^-?\d+\.0+$/.test(s)) return s.replace(/\.0+$/, '');
    return s;
}

function hexColor(n) {
    const s = String(n).replace(/^0x/i, '');
    return `Color::from_hex(0x${s})`;
}

function sideFromThree(sideExpr) {
    if (/DoubleSide/.test(sideExpr)) return '2';
    if (/BackSide/.test(sideExpr)) return '1';
    return '0';
}

function wrapMaterialForMesh(s) {
    if (!s) return s;
    if (s.startsWith('{ let mut m =')) {
        return s.replace(/;\s*m\s\}$/, '.into() }');
    }
    return `${s}.into()`;
}

function parseMaterial(expr) {
    const cleaned = expr.replace(/^new THREE\./, '');
    const basic = cleaned.match(/MeshBasicMaterial\(\s*\{([^}]*)\}\s*\)/);
    if (basic) {
        const opts = basic[1];
        const color = opts.match(/color:\s*(0x[0-9a-fA-F]+|\d+)/);
        const wire = /wireframe:\s*true/.test(opts);
        let s = `BasicMaterial::new(${hexColor(color ? color[1] : '0xffffff')})`;
        if (wire) s += '.wireframe(true)';
        const opacity = opts.match(/opacity:\s*([\d.]+)/);
        if (opacity) s += `.with_opacity(${toRustF32(opacity[1])})`;
        const transparent = /transparent:\s*true/.test(opts);
        if (transparent) s += '.with_transparent(true)';
        const side = opts.match(/side:\s*(THREE\.\w+)/);
        if (side && sideFromThree(side[1]) !== '0') {
            s = `{ let mut m = ${s}; m.side = ${sideFromThree(side[1])}; m }`;
        }
        return wrapMaterialForMesh(s);
    }

    const std = cleaned.match(/MeshStandardMaterial\(\s*\{([^}]*)\}\s*\)/);
    if (std) {
        const opts = std[1];
        const color = opts.match(/color:\s*(0x[0-9a-fA-F]+|\d+)/);
        let s = `StandardMaterial::new(${hexColor(color ? color[1] : '0xffffff')})`;
        const rough = opts.match(/roughness:\s*([\d.]+)/);
        if (rough) s += `.with_roughness(${toRustF32(rough[1])})`;
        const metal = opts.match(/metalness:\s*([\d.]+)/);
        if (metal) s += `.with_metalness(${toRustF32(metal[1])})`;
        const side = opts.match(/side:\s*(THREE\.\w+)/);
        if (side && sideFromThree(side[1]) !== '0') {
            s = `{ let mut m = ${s}; m.side = ${sideFromThree(side[1])}; m }`;
        }
        return wrapMaterialForMesh(s);
    }

    const phys = cleaned.match(/MeshPhysicalMaterial\(\s*\{([^}]*)\}\s*\)/);
    if (phys) {
        const opts = phys[1];
        const color = opts.match(/color:\s*(0x[0-9a-fA-F]+|\d+)/);
        let s = `{ let mut m = PhysicalMaterial::new(${hexColor(color ? color[1] : '0xffffff')});`;
        const rough = opts.match(/roughness:\s*([\d.]+)/);
        if (rough) s += ` m.roughness = ${toRustF32(rough[1])};`;
        const metal = opts.match(/metalness:\s*([\d.]+)/);
        if (metal) s += ` m.metalness = ${toRustF32(metal[1])};`;
        const cc = opts.match(/clearcoat:\s*([\d.]+)/);
        if (cc) s += ` m.clearcoat = ${toRustF32(cc[1])};`;
        const ccr = opts.match(/clearcoatRoughness:\s*([\d.]+)/);
        if (ccr) s += ` m.clearcoat_roughness = ${toRustF32(ccr[1])};`;
        const side = opts.match(/side:\s*(THREE\.\w+)/);
        if (side && sideFromThree(side[1]) !== '0') {
            s += ` m.side = ${sideFromThree(side[1])};`;
        }
        s += ' Material::Physical(m) }';
        return s;
    }

    const lambert = cleaned.match(/MeshLambertMaterial\(\s*\{([^}]*)\}\s*\)/);
    if (lambert) {
        const color = lambert[1].match(/color:\s*(0x[0-9a-fA-F]+|\d+)/);
        return wrapMaterialForMesh(`LambertMaterial::new(${hexColor(color ? color[1] : '0xffffff')})`);
    }

    const phong = cleaned.match(/MeshPhongMaterial\(\s*\{([^}]*)\}\s*\)/);
    if (phong) {
        const color = phong[1].match(/color:\s*(0x[0-9a-fA-F]+|\d+)/);
        return wrapMaterialForMesh(`PhongMaterial::new(${hexColor(color ? color[1] : '0xffffff')})`);
    }

    if (/MeshDepthMaterial\(\s*\)/.test(cleaned)) {
        return wrapMaterialForMesh('DepthMaterial::new()');
    }

    if (/MeshNormalMaterial\(\s*\)/.test(cleaned)) {
        return wrapMaterialForMesh('NormalMaterial::new()');
    }

    return null;
}

function parseVector2ArrayBody(body) {
    const pts = [];
    for (const m of body.matchAll(/Vector2\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)/g)) {
        pts.push(`Vector2::new(${m[1]}, ${m[2]})`);
    }
    return pts;
}

function rustLatheGeometryArg(name, ctx) {
    if (ctx?.vector2Arrays?.[name]) return `&${name}`;
    return `&[/* unresolved ${name} */]`;
}

function parseGeometry(expr, ctx = null) {
    const cleaned = expr.replace(/^new THREE\./, '');
    const lathe = cleaned.match(/LatheGeometry\(\s*([^,]+)\s*,\s*([^,)]+)(?:,\s*([^,)]+))?(?:,\s*([^)]+))?\s*\)/);
    if (lathe) {
        const ptsRef = lathe[1].trim();
        const segs = toRustUsize(lathe[2].trim());
        const phiStart = toRustF32(lathe[3]?.trim() || '0');
        const phiLen = lathe[4]?.trim() || 'std::f32::consts::TAU';
        const phiLenRust = phiLen.includes('Math.PI')
            ? phiLen.replace(/\*\s*2/g, ' * 2.0').replace(/Math\.PI/g, 'std::f32::consts::PI')
            : phiLen;
        return `LatheGeometry::new(${rustLatheGeometryArg(ptsRef, ctx)}, ${segs}, ${phiStart}, ${phiLenRust})`;
    }
    const sphere = cleaned.match(/SphereGeometry\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)/);
    if (sphere) {
        return `SphereGeometry::new(${toRustF32(sphere[1])}, ${toRustUsize(sphere[2])}, ${toRustUsize(sphere[3])})`;
    }
    for (const [jsName, rustName] of Object.entries(GEOMETRY)) {
        const re = new RegExp(`${jsName}\\(([^)]*)\\)`);
        const m = cleaned.match(re);
        if (m) {
            const args = m[1]
                .split(',')
                .map((a) => a.trim())
                .filter(Boolean)
                .map((a) => {
                    if (/^0x[0-9a-fA-F]+$/.test(a)) return hexColor(a);
                    if (/^\d+$/.test(a)) return `${a}.0`;
                    if (/^[\d.]+$/.test(a)) return a.includes('.') ? a : `${a}.0`;
                    if (a === 'THREE.DoubleSide') return '2';
                    if (a === 'false') return 'false';
                    if (a === 'true') return 'true';
                    return a;
                });
            if (rustName === 'ParametricGeometry') {
                return `// ParametricGeometry: pass a Rust closure to ParametricGeometry::new(...)`;
            }
            return `${rustName}::new(${args.join(', ')})`;
        }
    }
    if (/PlaneGeometry/.test(expr)) return `PlaneGeometry::new(/* … */)`;
    return null;
}

function parseMeshConstructor(expr, ctx = null) {
    const start = expr.indexOf('new THREE.Mesh(');
    if (start < 0) return null;
    let i = start + 'new THREE.Mesh('.length;
    let depth = 1;
    let comma = -1;
    while (i < expr.length && depth > 0) {
        const ch = expr[i];
        if (ch === '(') depth += 1;
        else if (ch === ')') depth -= 1;
        else if (ch === ',' && depth === 1 && comma < 0) comma = i;
        i += 1;
    }
    if (comma < 0 || depth !== 0) return null;
    const geomExpr = expr.slice(start + 'new THREE.Mesh('.length, comma).trim();
    const matExpr = expr.slice(comma + 1, i - 1).trim();
    const geom = parseGeometry(geomExpr, ctx);
    let mat = parseMaterial(matExpr);
    if (!mat && ctx?.materials && /^[\w$]+$/.test(matExpr)) {
        mat = `${ctx.materials[matExpr]}.clone()`;
    }
    if (!geom || !mat) return null;
    return `Object3D::mesh(Mesh::new(${geom}, ${mat}))`;
}

function convertLine(line, ctx) {
    let s = line.trim();
    if (!s || s.startsWith('//') || s.startsWith('/*')) return s ? `    // ${s.replace(/^\/\/\s?/, '')}` : null;

    // strip trailing semicolon for pattern matching
    const semi = s.endsWith(';');
    if (semi) s = s.slice(0, -1);

    if (/^const scene = new THREE\.Scene\(\)/.test(s)) {
        ctx.hasScene = true;
        return '    let mut scene = Scene::new();';
    }

    const bg = s.match(/^scene\.background = new THREE\.Color\((0x[0-9a-fA-F]+|\d+)\)/);
    if (bg) return `    scene.background = ${hexColor(bg[1])};`;

    const fogExp = s.match(/^scene\.fog = new THREE\.FogExp2\((0x[0-9a-fA-F]+|\d+),\s*([\d.]+)\)/);
    if (fogExp) {
        return `    scene.fog = FogParams { color: ${hexColor(fogExp[1])}, density: ${fogExp[2]}, mode: 2, ..Default::default() };`;
    }

    const fogLin = s.match(/^scene\.fog = new THREE\.Fog\((0x[0-9a-fA-F]+|\d+),\s*([\d.]+),\s*([\d.]+)\)/);
    if (fogLin) {
        return `    scene.fog = FogParams { color: ${hexColor(fogLin[1])}, near: ${fogLin[2]}, far: ${fogLin[3]}, mode: 1, ..Default::default() };`;
    }

    const amb = s.match(/^scene\.add\(new THREE\.AmbientLight\((0x[0-9a-fA-F]+|\d+),\s*([\d.]+)\)\)/);
    if (amb) {
        return `    scene.add_light(AmbientLight::new(${hexColor(amb[1])}, ${amb[2]}));`;
    }

    const dlInline = s.match(/^scene\.add\(new THREE\.DirectionalLight\((0x[0-9a-fA-F]+|\d+),\s*([\d.]+)\)\)/);
    if (dlInline) {
        return `    scene.add(Object3D::light(DirectionalLight::new(${hexColor(dlInline[1])}, ${dlInline[2]})));`;
    }

    const dl = s.match(/^const (dl|\w+) = new THREE\.DirectionalLight\((0x[0-9a-fA-F]+|\d+),\s*([\d.]+)\)/);
    if (dl) {
        ctx.vars[dl[1]] = { kind: 'directional_light', color: dl[2], intensity: dl[3] };
        return `    let mut ${dl[1]} = Object3D::light(DirectionalLight::new(${hexColor(dl[2])}, ${dl[3]}));`;
    }

    const plInline = s.match(/^scene\.add\(new THREE\.PointLight\((0x[0-9a-fA-F]+|\d+),\s*([\d.]+)(?:,\s*([\d.]+))?(?:,\s*([\d.]+))?\)\)/);
    if (plInline) {
        const dist = plInline[3] ?? '0';
        const decay = plInline[4] ?? '2';
        return `    scene.add(Object3D::light(PointLight::new(${hexColor(plInline[1])}, ${plInline[2]}).with_distance(${dist}).with_decay(${decay})));`;
    }

    const pl = s.match(/^const (\w+) = new THREE\.PointLight\((0x[0-9a-fA-F]+|\d+),\s*([\d.]+)(?:,\s*([\d.]+))?(?:,\s*([\d.]+))?\)/);
    if (pl) {
        const dist = pl[4] ?? '0';
        const decay = pl[5] ?? '2';
        ctx.vars[pl[1]] = { kind: 'point_light' };
        return `    let mut ${pl[1]} = Object3D::light(PointLight::new(${hexColor(pl[2])}, ${pl[3]}).with_distance(${dist}).with_decay(${decay}));`;
    }

    const sl = s.match(/^const (\w+) = new THREE\.SpotLight\((0x[0-9a-fA-F]+|\d+),\s*([\d.]+)(?:,\s*([\d.]+))?(?:,\s*([\d.]+))?(?:,\s*([\d.]+))?(?:,\s*([\d.]+))?\)/);
    if (sl) {
        const dist = sl[4] ?? '0';
        const angle = sl[5] ?? 'std::f32::consts::FRAC_PI_4';
        const penumbra = sl[6] ?? '0';
        const decay = sl[7] ?? '2';
        ctx.vars[sl[1]] = { kind: 'spot_light' };
        const angleRust = angle.includes('Math.PI')
            ? angle.replace(/Math\.PI\s*\/\s*4/, 'std::f32::consts::FRAC_PI_4').replace(/Math\.PI/g, 'std::f32::consts::PI')
            : angle;
        return `    let mut ${sl[1]} = Object3D::light({
        let mut l = SpotLight::new(${hexColor(sl[2])}, ${sl[3]});
        l.distance = ${dist}; l.decay = ${decay}; l.angle = ${angleRust}; l.penumbra = ${penumbra};
        l
    });`;
    }

    const posComp = s.match(/^(\w+)\.position\.(x|y|z)\s*=\s*([-\d.]+)/);
    if (posComp) {
        const name = posComp[1] === 'cam' ? 'camera' : posComp[1];
        return `    ${name}.position.${posComp[2]} = ${toRustF32(posComp[3])};`;
    }

    const posSet = s.match(/^(\w+)\.position\.set\(([-\d.]+),\s*([-.\d]+),\s*([-.\d]+)\)/);
    if (posSet) {
        const name = posSet[1] === 'cam' ? 'camera' : posSet[1];
        return `    ${name}.position = Vector3::new(${toRustF32(posSet[2])}, ${toRustF32(posSet[3])}, ${toRustF32(posSet[4])});`;
    }

    const rotSet = s.match(/^(\w+)\.rotation\.set\(([^,]+),\s*([^,]+),\s*([^)]+)\)/);
    if (rotSet) {
        return `    ${rotSet[1]}.quaternion = Euler::new(${toRustF32(rotSet[2])}, ${toRustF32(rotSet[3])}, ${toRustF32(rotSet[4])}).to_quaternion();`;
    }

    const meshCastShadow = s.match(/^(\w+)\.castShadow\s*=\s*true/);
    if (meshCastShadow) return `    ${meshCastShadow[1]}.cast_shadow = true;`;

    const meshReceiveShadow = s.match(/^(\w+)\.receiveShadow\s*=\s*true/);
    if (meshReceiveShadow) return `    ${meshReceiveShadow[1]}.receive_shadow = true;`;

    const scaleScalar = s.match(/^(\w+)\.scale\.setScalar\(([-\d.]+)\)/);
    if (scaleScalar) {
        return `    ${scaleScalar[1]}.scale = Vector3::splat(${scaleScalar[2]});`;
    }

    const scaleSet = s.match(/^(\w+)\.scale\.set\(([-\d.]+),\s*([-.\d]+),\s*([-.\d]+)\)/);
    if (scaleSet) {
        return `    ${scaleSet[1]}.scale = Vector3::new(${scaleSet[2]}, ${scaleSet[3]}, ${scaleSet[4]});`;
    }

    const addVar = s.match(/^scene\.add\((\w+)\)/);
    if (addVar) return `    scene.add(${addVar[1]});`;

    const matDecl = s.match(/^const (\w+) = new THREE\.MeshStandardMaterial\(\s*\{([^}]*)\}\s*\)/);
    if (matDecl) {
        const parsed = parseMaterial(`MeshStandardMaterial({${matDecl[2]}})`);
        if (parsed) {
            if (!ctx.materials) ctx.materials = {};
            ctx.materials[matDecl[1]] = parsed;
            return `    let ${matDecl[1]} = ${parsed};`;
        }
    }

    const inlineMesh = s.match(/^scene\.add\(new THREE\.Mesh\((.+)\)\)/);
    if (inlineMesh) {
        const mesh = parseMeshConstructor(`new THREE.Mesh(${inlineMesh[1]})`, ctx);
        if (mesh) return `    scene.add(${mesh});`;
    }

    const vec2Array = s.match(/^const (\w+) = \[(.*)\]$/);
    if (vec2Array && /Vector2/.test(vec2Array[2])) {
        const pts = parseVector2ArrayBody(vec2Array[2]);
        if (pts.length) {
            ctx.vector2Arrays = ctx.vector2Arrays || {};
            ctx.vector2Arrays[vec2Array[1]] = true;
            return `    let ${vec2Array[1]} = [${pts.join(', ')}];`;
        }
    }

    const meshDecl = s.match(/^const (\w+) = new THREE\.Mesh\((.+)\)/);
    if (meshDecl) {
        const mesh = parseMeshConstructor(`new THREE.Mesh(${meshDecl[2]})`, ctx);
        if (mesh) {
            ctx.vars[meshDecl[1]] = { kind: 'mesh' };
            return `    let mut ${meshDecl[1]} = ${mesh};`;
        }
    }

    const cam = s.match(/^const cam = new THREE\.PerspectiveCamera\(([\d.]+),\s*800\s*\/\s*600,\s*([\d.]+),\s*([\d.]+)\)/);
    if (cam) {
        ctx.hasCamera = true;
        return `    let mut camera = PerspectiveCamera::new(${toRustF32(cam[1])}, 800.0 / 600.0, ${toRustF32(cam[2])}, ${toRustF32(cam[3])});`;
    }

    const camPos = s.match(/^cam\.position\.set\(([-\d.]+),\s*([-.\d]+),\s*([-.\d]+)\)/);
    if (camPos) {
        return `    camera.position = Vector3::new(${toRustF32(camPos[1])}, ${toRustF32(camPos[2])}, ${toRustF32(camPos[3])});`;
    }

    const camLook = s.match(/^cam\.lookAt\(([-\d.]+),\s*([-.\d]+),\s*([-.\d]+)\)/);
    if (camLook) {
        return `    camera.look_at(Vector3::new(${toRustF32(camLook[1])}, ${toRustF32(camLook[2])}, ${toRustF32(camLook[3])}));`;
    }

    const camLookZero = /^cam\.lookAt\(0,\s*0,\s*0\)/.test(s);
    if (camLookZero) return '    camera.look_at(Vector3::ZERO);';

    // stray cam.* after camera rename
    if (/^cam\./.test(s)) {
        return `    // ${s};  // (use \`camera\` in Rust)`;
    }

    if (/composer|EffectComposer|ShaderPass|RenderPass|GlitchPass|FilmPass|DotScreenPass|HalftonePass|FXAA|SSAO|SSR|BloomPass|OutlinePass|Reflector|Sky|Water|Refractor/.test(s)) {
        return `    // ${s};  // browser-shim / JS-only — see threers::postprocessing or web/threejs-shim.js`;
    }

    if (/^await |^const r =|^r\.|^document\.|^errEl|^THREE\.|^import |^async |^\(\(\)|^try |^catch |^for \(let /.test(s)) {
        return null;
    }

    return `    // ${s};`;
}

export function extractSetupFromThreersHtml(html) {
    const mod = html.match(/<script[^>]*type=["']module["'][^>]*>([\s\S]*?)<\/script>/i);
    if (!mod) return '';
    let body = mod[1];
    body = body.replace(/^import[\s\S]*?;\s*/m, '');
    body = body.replace(/^const errEl[\s\S]*?;\s*/m, '');
    body = body.replace(/\(async \(\) => \{[\s\S]*?try \{[\s\S]*?await initThreers\([^)]*\);\s*/m, '');
    body = body.replace(/const r = await THREE\.WebGLRenderer\.create[\s\S]*$/m, '');
    return body.trim();
}

export function convertJsSetupToRust(setup, slug) {
    if (RUST_OVERRIDES[slug]) return RUST_OVERRIDES[slug];

    const ctx = { vars: {}, materials: {}, hasScene: false, hasCamera: false };
    const lines = normalizeSetup(setup).split('\n');
    const out = [];

    for (const raw of lines) {
        const converted = convertLine(raw, ctx);
        if (converted == null) continue;
        out.push(converted);
    }

    if (!ctx.hasCamera) {
        out.push('');
        out.push('    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);');
        out.push('    camera.position = Vector3::new(0.0, 0.0, 3.0);');
        out.push('    camera.look_at(Vector3::ZERO);');
    }

    return out.join('\n');
}

function collectImports(rustBody) {
    const names = new Set(['Camera', 'Color', 'Mesh', 'Object3D', 'PerspectiveCamera', 'Renderer', 'Scene', 'Vector3']);
    const needsFog = /\bFogParams\b/.test(rustBody);
    const needsMirrorMod = /\bMirrorMaterial\b/.test(rustBody);
    const patterns = [
        ['AmbientLight', /\bAmbientLight\b/],
        ['RenderTarget', /\bRenderTarget\b/],
        ['DirectionalLight', /\bDirectionalLight\b/],
        ['PointLight', /\bPointLight\b/],
        ['SpotLight', /\bSpotLight\b/],
        ['BasicMaterial', /\bBasicMaterial\b/],
        ['StandardMaterial', /\bStandardMaterial\b/],
        ['PhysicalMaterial', /\bPhysicalMaterial\b/],
        ['DataTexture', /\bDataTexture\b/],
        ['TextureFormat', /\bTextureFormat\b/],
        ['LambertMaterial', /\bLambertMaterial\b/],
        ['PhongMaterial', /\bPhongMaterial\b/],
        ['DepthMaterial', /\bDepthMaterial\b/],
        ['NormalMaterial', /\bNormalMaterial\b/],
        ['MirrorMaterial', /\bMirrorMaterial\b/],
        ['Material', /\bMaterial::/],
        ['BufferAttribute', /\bBufferAttribute\b/],
        ['BufferGeometry', /\bBufferGeometry\b/],
        ['BoxGeometry', /\bBoxGeometry\b/],
        ['SphereGeometry', /\bSphereGeometry\b/],
        ['PlaneGeometry', /\bPlaneGeometry\b/],
        ['CircleGeometry', /\bCircleGeometry\b/],
        ['RingGeometry', /\bRingGeometry\b/],
        ['CylinderGeometry', /\bCylinderGeometry\b/],
        ['ConeGeometry', /\bConeGeometry\b/],
        ['TorusGeometry', /\bTorusGeometry\b/],
        ['TorusKnotGeometry', /\bTorusKnotGeometry\b/],
        ['CapsuleGeometry', /\bCapsuleGeometry\b/],
        ['LatheGeometry', /\bLatheGeometry\b/],
        ['TubeGeometry', /\bTubeGeometry\b/],
        ['ExtrudeGeometry', /\bExtrudeGeometry\b/],
        ['Vector2', /\bVector2\b/],
        ['IcosahedronGeometry', /\bIcosahedronGeometry\b/],
        ['DodecahedronGeometry', /\bDodecahedronGeometry\b/],
        ['TetrahedronGeometry', /\bTetrahedronGeometry\b/],
        ['OctahedronGeometry', /\bOctahedronGeometry\b/],
        ['Euler', /\bEuler\b/],
    ];
    for (const [name, re] of patterns) {
        if (re.test(rustBody)) names.add(name);
    }
    const threers = [...names]
        .filter((n) => n !== 'Camera' && n !== 'FogParams' && !(needsMirrorMod && n === 'MirrorMaterial'))
        .sort();
    return { threers, needsFog, needsMirrorMod };
}

function normalizeSetup(setup) {
    return setup
        .replace(/\r\n/g, '\n')
        .replace(/\n\s+/g, ' ')
        .replace(/;\s*/g, ';\n')
        .split('\n')
        .map((l) => l.trim())
        .filter(Boolean)
        .join('\n');
}

export function wrapRustScene(slug, sceneBody) {
    const { threers, needsFog, needsMirrorMod } = collectImports(sceneBody);
    const extraUse = [
        needsMirrorMod ? 'use threers::materials::MirrorMaterial;' : '',
        needsFog ? 'use threers::scene::FogParams;' : '',
    ].filter(Boolean).join('\n');
    const moduleExtra = RUST_MODULE_OVERRIDES[slug] || '';
    const drawBody = RUST_DRAW_OVERRIDES[slug]
        || '                                    renderer.render(&mut scene, &camera, &view, false);';

    return `//! Parity scene \`${slug}\` — native Rust (winit + wgpu).
//!
//! Generated from \`tests/parity/scenes/threers-${slug}.html\`.
//! Compare with the JavaScript tab above; adjust imports before copying to \`examples/\`.
//!
//! \`\`\`text
//! cargo run --example ${slug.replace(/-/g, '_')}
//! \`\`\`

use std::sync::Arc;

use threers::cameras::Camera;
${extraUse ? `${extraUse}\n` : ''}use threers::{
    ${threers.join(',\n    ')},
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

${moduleExtra}
fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — ${slug}")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)
            .expect("window"),
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let surface = instance.create_surface(window.clone()).expect("surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .expect("device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let mut renderer = Renderer::new(device.clone(), queue.clone(), format, config.width, config.height);

    // --- scene (matches threers parity iframe) ---
${sceneBody}

    scene.update_world();
    let window_for_loop = window.clone();

    event_loop
        .run(move |event, target| {
            match event {
                Event::WindowEvent { event, window_id } if window_id == window_for_loop.id() => {
                    match event {
                        WindowEvent::CloseRequested => target.exit(),
                        WindowEvent::Resized(new_size) => {
                            config.width = new_size.width.max(1);
                            config.height = new_size.height.max(1);
                            surface.configure(&device, &config);
                            renderer.resize(config.width, config.height);
                            camera.set_aspect(config.width as f32 / config.height as f32);
                        }
                        WindowEvent::RedrawRequested => {
                            scene.update_world();
                            match surface.get_current_texture() {
                                Ok(frame) => {
                                    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
${drawBody}
                                    frame.present();
                                }
                                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                    surface.configure(&device, &config);
                                }
                                Err(e) => log::error!("surface: {e:?}"),
                            }
                        }
                        _ => {}
                    }
                }
                Event::AboutToWait => window_for_loop.request_redraw(),
                _ => {}
            }
        })
        .expect("event loop");
}
`;
}

export function threersHtmlToRust(html, slug) {
    const setup = extractSetupFromThreersHtml(html);
    const body = convertJsSetupToRust(setup, slug);
    return wrapRustScene(slug, body);
}
