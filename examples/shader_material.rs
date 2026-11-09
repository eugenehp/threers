//! Custom-shader material demo — the extensibility escape hatch.
//!
//! A cube drawn with a [`ShaderMaterial`]: the fragment is user WGSL that reads
//! a user uniform (base color) and a user storage array (stripe frequency),
//! shaded against the built-in `frame` lights. Compiled against threers'
//! standard preamble, so `VsOut`, `frame`, and `framebuffer_encode` are free.
//!
//! ```text
//! cargo run --example shader_material
//! ```

use std::time::Instant;

use threers::cameras::Camera;
use threers::renderer::WindowRenderer;
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Euler, Mesh, Object3D, PerspectiveCamera,
    Scene, ShaderMaterial, Vector3,
};
use winit::event::{Event, WindowEvent};

const FRAGMENT: &str = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let base = u_data.data[0].rgb;         // user uniform: base color
    let freq = u_s0[0];                     // user storage: stripe frequency
    let stripe = 0.5 + 0.5 * sin(in.world_pos.y * freq);
    let tint = mix(base, in.world_normal * 0.5 + 0.5, stripe);

    var light = frame.ambient.rgb;
    for (var i: u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
        if (i >= 4u) { break; }
        let dir = normalize(frame.dir_lights[i].direction.xyz);
        light = light + frame.dir_lights[i].color.rgb * max(dot(normalize(in.world_normal), -dir), 0.0);
    }
    return vec4<f32>(framebuffer_encode(tint * light), 1.0);
}
"#;

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — shader material")
        .with_inner_size(winit::dpi::LogicalSize::new(900, 700));
    threers::sketch::run_window(attributes, move |window| {
        // wgpu by default; the native Metal backend with `--features metal` and
        // `THREERS_BACKEND=metal` — where the custom WGSL does not run yet, and
        // the cube draws unlit in its base colour.
        let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x101018);

        // A cube with a custom-shader material.
        let mut data = vec![[0.0f32; 4]; 16];
        data[0] = [1.0, 0.45, 0.2, 0.0]; // base color
        let material = ShaderMaterial::new(FRAGMENT)
            .with_data(data)
            .with_storage0(vec![8.0]);
        let cube_id = scene.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(1.2, 1.2, 1.2),
            material.into(),
        )));

        scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.3));
        let mut key = Object3D::light(
            DirectionalLight::new(Color::from_hex(0xffffff), 1.0)
                .with_direction(Vector3::new(-0.5, -1.0, -0.4)),
        );
        key.position = Vector3::new(3.0, 5.0, 2.0);
        scene.add(key);

        let mut camera = PerspectiveCamera::new(60.0, view.aspect(), 0.1, 100.0);
        camera.position = Vector3::new(2.5, 2.0, 3.5);
        camera.look_at(Vector3::ZERO);

        let start = Instant::now();
        let win = window.clone();
        move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == win.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(s) => {
                    view.resize(s.width, s.height);
                    camera.set_aspect(view.aspect());
                }
                WindowEvent::RedrawRequested => {
                    let t = start.elapsed().as_secs_f32();
                    if let Some(obj) = scene.get_mut(cube_id) {
                        obj.quaternion = Euler::new(t * 0.6, t * 0.9, 0.0).to_quaternion();
                    }
                    view.render(&mut scene, &camera);
                    win.request_redraw();
                }
                _ => {}
            },
            _ => {}
        }
    });
}
