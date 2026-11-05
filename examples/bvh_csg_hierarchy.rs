//! The bvh-csg hierarchy parity scene, evaluated live and drawn in a window: a
//! shell (box minus a smaller box), a sphere added on, and a window cut and
//! framed through an operation group — three-bvh-csg's `evaluateHierarchy`.
//!
//! An optional step, 1 to 4, applies only that many of the root's
//! operations, to watch the hierarchy build up. Each prints its triangle
//! count beside three-bvh-csg 0.0.16's (`tests/csg_upstream.rs` holds them
//! equal).
//!
//! ```text
//! cargo run --example bvh_csg_hierarchy --features bvh-csg [-- 1|2|3|4]
//! THREERS_BACKEND=metal cargo run --example bvh_csg_hierarchy --features bvh-csg,metal
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::renderer::WindowRenderer;
use threers::{
    evaluate_hierarchy, AmbientLight, BoxGeometry, BufferGeometry, Color, CsgEvaluator, CsgNode,
    CsgOperation, CsgOperationGroup, DirectionalLight, Mesh, Object3D, PerspectiveCamera, Scene,
    SphereGeometry, StandardMaterial, Vector3, ADDITION, SUBTRACTION,
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

/// three-bvh-csg's triangle count after each step.
const UPSTREAM_TRIANGLES: [usize; 4] = [24, 774, 804, 816];

/// The root with its first `steps` operations: the cut, the sphere, and the
/// window group's cut and frame.
fn build_hierarchy_geometry(steps: usize) -> BufferGeometry {
    let soup = |g: BufferGeometry| g.to_non_indexed();
    let mut root = CsgOperation::new(soup(BoxGeometry::new(4.0, 2.5, 2.5)), ADDITION);
    root.add_child(CsgNode::Operation(CsgOperation::new(
        soup(BoxGeometry::new(3.6, 2.1, 2.1)),
        SUBTRACTION,
    )));
    if steps >= 2 {
        let mut sphere = CsgOperation::new(soup(SphereGeometry::new(0.55, 24, 12)), ADDITION);
        sphere.set_position(-1.1, 0.2, 1.35);
        root.add_child(CsgNode::Operation(sphere));
    }
    if steps >= 3 {
        let mut window = CsgOperationGroup::new();
        window.add_child(CsgNode::Operation(CsgOperation::new(
            soup(BoxGeometry::new(1.2, 1.0, 0.5)),
            SUBTRACTION,
        )));
        if steps >= 4 {
            window.add_child(CsgNode::Operation(CsgOperation::new(
                soup(BoxGeometry::new(1.2, 1.0, 0.12)),
                ADDITION,
            )));
        }
        window.set_position(0.8, 0.15, 1.35);
        root.add_child(CsgNode::Group(window));
    }
    let mut evaluator = CsgEvaluator::new();
    evaluator.use_groups = false;
    let mut geometry = evaluate_hierarchy(&mut root, &mut evaluator);
    threers::utils::compute_vertex_normals(&mut geometry);
    geometry
}

fn main() {
    env_logger::init();
    let steps = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .filter(|n| (1..=4).contains(n))
        .unwrap_or(4);
    pollster::block_on(run(steps));
}

async fn run(steps: usize) {
    let geometry = build_hierarchy_geometry(steps);
    let triangles = geometry
        .get_attribute("position")
        .map_or(0, |a| a.count() / 3);
    println!(
        "step {steps}: {triangles} triangles (three-bvh-csg {})",
        UPSTREAM_TRIANGLES[steps - 1]
    );
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(format!(
                "threers — bvh-csg hierarchy, step {steps}: {triangles} triangles"
            ))
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)
            .expect("window"),
    );

    // wgpu by default; the native Metal backend with `--features metal` and
    // `THREERS_BACKEND=metal`.
    let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101418);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.45));
    let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    dl.position = Vector3::new(3.0, 5.0, 2.0);
    scene.add(dl);

    let mat = StandardMaterial::new(Color::from_hex(0x4488cc))
        .with_roughness(0.45)
        .with_metalness(0.1);
    let mesh = Object3D::mesh(Mesh::new(geometry, mat.into()));
    scene.add(mesh);

    let mut camera = PerspectiveCamera::new(50.0, view.aspect(), 0.1, 100.0);
    camera.position = Vector3::new(-3.8, 2.9, 1.25);
    camera.look_at(Vector3::ZERO);

    scene.update_world();
    let window_for_loop = window.clone();

    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == window_for_loop.id() => {
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(new_size) => {
                        view.resize(new_size.width, new_size.height);
                        camera.set_aspect(view.aspect());
                    }
                    WindowEvent::RedrawRequested => {
                        scene.update_world();
                        // `false` while the window is hidden: nothing to do.
                        view.render(&mut scene, &camera);
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => window_for_loop.request_redraw(),
            _ => {}
        })
        .expect("event loop");
}
