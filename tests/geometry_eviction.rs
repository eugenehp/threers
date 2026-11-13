//! Removing a mesh frees it: the arena drops the object, and the renderer drops
//! the GPU buffers of a geometry nothing draws any more.
//!
//! Before this, `WebScene::remove` only detached the node, so the object and its
//! `Arc<BufferGeometry>` lived for the life of the page, and the renderer's
//! geometry cache — keyed by the `Arc`'s address — never let go of a vertex
//! buffer. An app streaming map tiles in and out grew GPU memory without bound.

use std::sync::Arc;

use threers::{BoxGeometry, Material, Mesh, Object3D, PerspectiveCamera, Renderer, Scene, Vector3};

const W: u32 = 32;

struct Rig {
    renderer: Renderer,
    view: wgpu::TextureView,
    _target: wgpu::Texture,
}

impl Rig {
    fn new() -> Option<Rig> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("eviction test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        let (device, queue) = (Arc::new(device), Arc::new(queue));
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut renderer = Renderer::new(device.clone(), queue, format, W, W);
        // Retention 0: what the last render did not draw is dropped at once, so
        // these can check the very next frame. The default keeps an undrawn
        // geometry for `DEFAULT_CACHE_RETENTION` renders first.
        renderer.set_cache_retention(0);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width: W,
                height: W,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        Some(Rig {
            renderer,
            view,
            _target: target,
        })
    }

    fn render(&mut self, scene: &mut Scene) {
        let mut cam = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 8.0);
        cam.look_at(Vector3::ZERO);
        self.renderer.render(scene, &cam, &self.view, false);
    }
}

fn cube(x: f32) -> Object3D {
    let mut o = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Basic(Default::default()),
    ));
    o.position = Vector3::new(x, 0.0, 0.0);
    o
}

#[test]
fn removed_meshes_release_their_gpu_geometry() {
    let Some(mut rig) = Rig::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut scene = Scene::new();
    let a = scene.add(cube(-1.5));
    let _b = scene.add(cube(1.5));
    rig.render(&mut scene);
    assert_eq!(rig.renderer.cached_geometry_count(), 2);
    let objects = scene.arena.nodes.len();

    assert_eq!(scene.arena.remove_subtree(a), 1);
    assert_eq!(
        scene.arena.nodes.len(),
        objects - 1,
        "the object stayed in the arena"
    );
    assert!(scene.arena.get(a).is_none());
    rig.render(&mut scene);
    assert_eq!(
        rig.renderer.cached_geometry_count(),
        1,
        "the removed mesh's GPU geometry was kept"
    );

    // Streaming: add and remove many times; the cache stays at what is live.
    for i in 0..50 {
        let id = scene.add(cube(i as f32 * 0.01));
        rig.render(&mut scene);
        scene.arena.remove_subtree(id);
    }
    rig.render(&mut scene);
    assert_eq!(rig.renderer.cached_geometry_count(), 1);
}

#[test]
fn geometry_still_held_elsewhere_is_kept() {
    let Some(mut rig) = Rig::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut scene = Scene::new();
    let geom = Arc::new(BoxGeometry::new(1.0, 1.0, 1.0));
    let mut a = Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Basic(Default::default()),
    );
    a.geometry = geom.clone();
    let mut b = Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Basic(Default::default()),
    );
    b.geometry = geom.clone();
    let ia = scene.add(Object3D::mesh(a));
    let _ib = scene.add(Object3D::mesh(b));
    rig.render(&mut scene);
    assert_eq!(
        rig.renderer.cached_geometry_count(),
        1,
        "one shared geometry, one upload"
    );
    scene.arena.remove_subtree(ia);
    rig.render(&mut scene);
    assert_eq!(
        rig.renderer.cached_geometry_count(),
        1,
        "a geometry another mesh draws was dropped"
    );
    drop(geom);
}

#[test]
fn remove_subtree_takes_the_descendants() {
    let mut scene = Scene::new();
    let parent = scene.add(Object3D::group());
    let child = scene.add_to(parent, cube(0.0));
    let grandchild = scene.add_to(child, cube(1.0));
    let before = scene.arena.nodes.len();
    assert_eq!(scene.arena.remove_subtree(parent), 3);
    assert_eq!(scene.arena.nodes.len(), before - 3);
    assert!(scene.arena.get(child).is_none() && scene.arena.get(grandchild).is_none());
    // the root lost the child id
    let root = scene.root;
    assert!(!scene.arena.get(root).unwrap().children.contains(&parent));
}
