//! A `ShaderMaterial`'s GPU resources outlive the frame.
//!
//! The renderer used to build a uniform buffer, three storage buffers (or data
//! textures) and a bind group for every custom draw, every frame — megabytes a
//! frame for a material carrying per-object data, times every mesh it was
//! handed to. These pin the replacement down: an unchanged material creates
//! nothing, a uniform change is one small write, a range update uploads the
//! range, a material shared by many meshes is uploaded once — and the pixels
//! still follow every one of those changes.
//!
//! Each test runs twice: on the adapter as it comes, and on a device asked for
//! with no storage buffers, which is what WebGL2 reports and takes the
//! data-texture path instead.

use std::sync::Arc;

use threers::renderer::CustomShaderStats;
use threers::{
    BoxGeometry, Color, Material, Mesh, Object3D, ObjectId, ObjectKind, PerspectiveCamera,
    Renderer, Scene, ShaderMaterial, ToneMapping, Vector3,
};

const W: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Colour from the uniform block.
const UNIFORM_TINT: &str = "@fragment\n\
    fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
    return vec4<f32>(u_data.data[0].rgb, 1.0);\n\
    }\n";

/// Red from storage slot 0 at the index in `u_data.data[0].x`.
const STORAGE_AT: &str = "@fragment\n\
    fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
    return vec4<f32>(s0(u32(u_data.data[0].x)), 0.0, 0.0, 1.0);\n\
    }\n";

/// A renderer drawing into a small target it can read back.
struct Rig {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: Renderer,
    target: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Rig {
    /// `storage: false` asks for a device with no storage buffers.
    fn new(storage: bool) -> Option<Rig> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let mut limits = adapter.limits();
        if !storage {
            limits.max_storage_buffers_per_shader_stage = 0;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("persist test device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .ok()?;
        let (device, queue) = (Arc::new(device), Arc::new(queue));
        assert_eq!(
            threers::renderer::caps::supports_storage_buffers(&device),
            storage,
            "the device did not come back with the storage support asked for"
        );
        let mut renderer = Renderer::new(device.clone(), queue.clone(), FORMAT, W, W);
        renderer.set_tone_mapping(ToneMapping::None, 1.0);
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
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        Some(Rig {
            device,
            queue,
            renderer,
            target,
            view,
        })
    }

    fn stats(&self) -> CustomShaderStats {
        self.renderer.custom_shader_stats()
    }

    /// Render and read the whole target back.
    fn render(&mut self, scene: &mut Scene) -> Vec<u8> {
        let mut cam = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 6.0);
        cam.look_at(Vector3::ZERO);
        scene.update_world();
        self.renderer.render(scene, &cam, &self.view, false);

        let bpr = W * 4;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bpr * W) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(W),
                },
            },
            wgpu::Extent3d {
                width: W,
                height: W,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(enc.finish()));
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        buf.slice(..)
            .get_mapped_range()
            .expect("buffer range is mapped")
            .to_vec()
    }
}

fn at(px: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

fn centre(px: &[u8]) -> [u8; 3] {
    at(px, W / 2, W / 2)
}

fn near(got: [u8; 3], want: [f32; 3]) -> bool {
    got.iter()
        .zip(want)
        .all(|(&g, w)| (g as i32 - (w * 255.0).round() as i32).abs() <= 2)
}

/// A box covering the view, drawn with `material`, sharing its `Arc`.
fn backdrop(scene: &mut Scene, material: &Arc<Material>) -> ObjectId {
    let mut mesh = Mesh::new(
        BoxGeometry::new(20.0, 20.0, 0.1),
        Material::Basic(Default::default()),
    );
    mesh.material = material.clone();
    scene.add(Object3D::mesh(mesh))
}

fn shader(m: &Arc<Material>) -> &ShaderMaterial {
    match &**m {
        Material::Shader(sm) => sm,
        _ => unreachable!(),
    }
}

/// Run `test` on both backends, skipping where there is no adapter.
fn both(test: impl Fn(Rig, bool)) {
    for storage in [true, false] {
        match Rig::new(storage) {
            Some(rig) => test(rig, storage),
            None => eprintln!("skipping (storage buffers: {storage}): no suitable adapter"),
        }
    }
}

/// Changing the uniform block every frame writes 256 bytes a frame and builds
/// nothing — and the picture follows it.
#[test]
fn a_uniform_change_reuses_the_buffers_and_bind_group() {
    both(|mut rig, storage| {
        let m = Arc::new(Material::Shader(
            ShaderMaterial::new(UNIFORM_TINT)
                .with_data(vec![[0.2, 0.4, 0.6, 1.0]])
                .with_storage0(vec![1.0; 4096]),
        ));
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        backdrop(&mut scene, &m);

        let px = rig.render(&mut scene);
        assert!(
            near(centre(&px), [0.2, 0.4, 0.6]),
            "first frame drew {:?}",
            centre(&px)
        );
        let first = rig.stats();
        assert_eq!(first.bind_groups_created, 1);

        // Unchanged: nothing created, nothing uploaded.
        rig.render(&mut scene);
        let idle = rig.stats();
        assert_eq!(
            idle, first,
            "an unchanged material re-uploaded or rebuilt something"
        );

        for k in 0..5 {
            let v = 0.1 * k as f32;
            shader(&m).set_uniform(0, [v, 0.5, 1.0 - v, 1.0]);
            let px = rig.render(&mut scene);
            assert!(
                near(centre(&px), [v, 0.5, 1.0 - v]),
                "frame {k} (storage {storage}) drew {:?}, wanted the new uniform",
                centre(&px)
            );
        }
        let after = rig.stats();
        assert_eq!(after.buffers_created, first.buffers_created);
        assert_eq!(after.textures_created, first.textures_created);
        assert_eq!(after.bind_groups_created, first.bind_groups_created);
        assert_eq!(after.bytes_uploaded - first.bytes_uploaded, 5 * 256);
    });
}

/// The pattern code already uses — `Arc::make_mut` and write the field — is
/// noticed too, without the setters.
#[test]
fn writing_the_data_field_in_place_is_seen() {
    both(|mut rig, _| {
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        let m = Arc::new(Material::Shader(
            ShaderMaterial::new(UNIFORM_TINT).with_data(vec![[1.0, 0.0, 0.0, 1.0]]),
        ));
        let id = backdrop(&mut scene, &m);
        drop(m);
        rig.render(&mut scene);
        let first = rig.stats();

        let Some(ObjectKind::Mesh(mesh)) = scene.get_mut(id).map(|o| &mut o.kind) else {
            unreachable!()
        };
        let Material::Shader(sm) = Arc::make_mut(&mut mesh.material) else {
            unreachable!()
        };
        sm.data[0] = [0.0, 1.0, 0.0, 1.0];
        let px = rig.render(&mut scene);
        assert!(near(centre(&px), [0.0, 1.0, 0.0]), "drew {:?}", centre(&px));
        assert_eq!(rig.stats().bind_groups_created, first.bind_groups_created);
        assert_eq!(rig.stats().buffers_created, first.buffers_created);
    });
}

/// A range update uploads the range, not the slot, and lands where it was
/// written.
#[test]
fn a_storage_range_update_uploads_only_the_range() {
    both(|mut rig, storage| {
        // 40,000 floats: five texture rows of 2048 texels on the texture path,
        // so a one-float write is one row there rather than the whole slot.
        const N: usize = 40_000;
        let m = Arc::new(Material::Shader(
            ShaderMaterial::new(STORAGE_AT).with_data(vec![[30_000.0, 0.0, 0.0, 0.0]]),
        ));
        shader(&m).set_storage(0, vec![0.25; N]);
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        backdrop(&mut scene, &m);

        let px = rig.render(&mut scene);
        assert!(
            near(centre(&px), [0.25, 0.0, 0.0]),
            "drew {:?}",
            centre(&px)
        );
        let first = rig.stats();

        shader(&m).set_storage_range(0, 30_000, &[0.75]);
        let px = rig.render(&mut scene);
        assert!(
            near(centre(&px), [0.75, 0.0, 0.0]),
            "storage {storage}: the range update did not reach the shader, drew {:?}",
            centre(&px)
        );
        let after = rig.stats();
        let uploaded = after.bytes_uploaded - first.bytes_uploaded;
        if storage {
            assert_eq!(uploaded, 4, "a one-float range uploaded {uploaded} bytes");
        } else {
            assert_eq!(
                uploaded,
                2048 * 16,
                "a one-float range uploaded {uploaded} bytes"
            );
        }
        assert_eq!(after.buffers_created, first.buffers_created);
        assert_eq!(after.textures_created, first.textures_created);
        assert_eq!(after.bind_groups_created, first.bind_groups_created);

        // Several ranges between two frames all land.
        shader(&m).set_storage_range(0, 10, &[0.5]);
        shader(&m).set_storage_range(0, 30_000, &[1.0]);
        let px = rig.render(&mut scene);
        assert!(near(centre(&px), [1.0, 0.0, 0.0]), "drew {:?}", centre(&px));
        shader(&m).set_data(&[[10.0, 0.0, 0.0, 0.0]]);
        let px = rig.render(&mut scene);
        assert!(near(centre(&px), [0.5, 0.0, 0.0]), "drew {:?}", centre(&px));
    });
}

/// Storage that outgrows its buffer gets a new one, and is drawn from it.
#[test]
fn storage_that_grows_is_reallocated() {
    both(|mut rig, _| {
        let m = Arc::new(Material::Shader(
            ShaderMaterial::new(STORAGE_AT).with_data(vec![[3.0, 0.0, 0.0, 0.0]]),
        ));
        shader(&m).set_storage(0, vec![0.25; 4]);
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        backdrop(&mut scene, &m);
        rig.render(&mut scene);
        let first = rig.stats();

        shader(&m).set_storage_range(0, 5000, &[0.75]);
        shader(&m).set_data(&[[5000.0, 0.0, 0.0, 0.0]]);
        let px = rig.render(&mut scene);
        assert!(
            near(centre(&px), [0.75, 0.0, 0.0]),
            "drew {:?}",
            centre(&px)
        );
        let after = rig.stats();
        assert_eq!(
            after.buffers_created + after.textures_created,
            first.buffers_created + first.textures_created + 1
        );
        assert_eq!(after.bind_groups_created, first.bind_groups_created + 1);
    });
}

/// One material handed to several meshes is uploaded once, and one in-place
/// change reaches all of them. Only the mesh transform differs per draw.
#[test]
fn meshes_sharing_a_material_share_its_buffers() {
    both(|mut rig, storage| {
        let m = Arc::new(Material::Shader(
            ShaderMaterial::new(UNIFORM_TINT)
                .with_data(vec![[1.0, 0.0, 0.0, 1.0]])
                .with_storage0(vec![0.0; 100_000]),
        ));
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        for x in [-2.0f32, 0.0, 2.0] {
            let mut mesh = Mesh::new(
                BoxGeometry::new(1.5, 1.5, 0.1),
                Material::Basic(Default::default()),
            );
            mesh.material = m.clone();
            let mut obj = Object3D::mesh(mesh);
            obj.position = Vector3::new(x, 0.0, 0.0);
            scene.add(obj);
        }
        // Box centres in pixels: the half-width of the view at the boxes is
        // 6 tan 30° ≈ 3.46.
        let xs = [-2.0f32, 0.0, 2.0].map(|x| (W as f32 * 0.5 * (1.0 + x / 3.464)) as u32);

        let px = rig.render(&mut scene);
        for x in xs {
            assert!(
                near(at(&px, x, W / 2), [1.0, 0.0, 0.0]),
                "box at {x} drew {:?}",
                at(&px, x, W / 2)
            );
        }
        let first = rig.stats();
        // One uniform buffer and one copy of each storage slot, not three.
        if storage {
            assert_eq!(first.buffers_created, 4);
        } else {
            assert_eq!((first.buffers_created, first.textures_created), (1, 3));
        }
        assert_eq!(first.bind_groups_created, 3, "one bind group per draw");
        assert_eq!(first.materials_cached, 1);
        let full = (100_000 * 4 + 2 * 4 + 256) as u64;
        if storage {
            assert_eq!(
                first.bytes_uploaded, full,
                "the shared storage was uploaded more than once"
            );
        }

        shader(&m).set_uniform(0, [0.0, 0.0, 1.0, 1.0]);
        let px = rig.render(&mut scene);
        for x in xs {
            assert!(
                near(at(&px, x, W / 2), [0.0, 0.0, 1.0]),
                "box at {x} drew {:?}",
                at(&px, x, W / 2)
            );
        }
        let after = rig.stats();
        assert_eq!(after.bytes_uploaded - first.bytes_uploaded, 256);
        assert_eq!(after.bind_groups_created, first.bind_groups_created);
        assert_eq!(after.buffers_created, first.buffers_created);
    });
}

/// Clones each get their own identity, so a copy whose *fields* were changed
/// draws its own data in the same frame as the original rather than both
/// drawing whichever was uploaded last. Live values stay shared.
#[test]
fn clones_with_different_fields_draw_their_own_data() {
    both(|mut rig, _| {
        let mut a = ShaderMaterial::new(STORAGE_AT).with_data(vec![[1.0, 0.0, 0.0, 0.0]]);
        a.storage0 = vec![0.0, 0.25];
        let mut b = a.clone();
        b.storage0 = vec![0.0, 0.75];
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        for (x, m) in [(-1.5f32, a.clone()), (1.5, b)] {
            let mut obj = Object3D::mesh(Mesh::new(
                BoxGeometry::new(2.0, 2.0, 0.1),
                Material::Shader(m),
            ));
            obj.position = Vector3::new(x, 0.0, 0.0);
            scene.add(obj);
        }
        let (l, r) = (W / 2 - 14, W / 2 + 14);
        let px = rig.render(&mut scene);
        assert!(
            near(at(&px, l, W / 2), [0.25, 0.0, 0.0]),
            "left drew {:?}",
            at(&px, l, W / 2)
        );
        assert!(
            near(at(&px, r, W / 2), [0.75, 0.0, 0.0]),
            "right drew {:?}",
            at(&px, r, W / 2)
        );

        // A live value set on the original reaches both copies.
        a.set_storage(0, vec![0.0, 0.5]);
        let px = rig.render(&mut scene);
        assert!(
            near(at(&px, l, W / 2), [0.5, 0.0, 0.0]),
            "left drew {:?}",
            at(&px, l, W / 2)
        );
        assert!(
            near(at(&px, r, W / 2), [0.5, 0.0, 0.0]),
            "right drew {:?}",
            at(&px, r, W / 2)
        );
    });
}

/// A storage field edited in place keeps its allocation, so it is invisible to
/// the cache until `mark_storage_dirty` says otherwise; replacing it is seen.
#[test]
fn storage_fields_are_seen_when_replaced_or_marked() {
    both(|mut rig, _| {
        let mut sm = ShaderMaterial::new(STORAGE_AT).with_data(vec![[1.0, 0.0, 0.0, 0.0]]);
        sm.storage0 = vec![0.0, 0.25];
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        let id = scene.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(20.0, 20.0, 0.1),
            Material::Shader(sm),
        )));
        rig.render(&mut scene);

        let edit = |scene: &mut Scene, f: &dyn Fn(&mut ShaderMaterial)| {
            let Some(ObjectKind::Mesh(mesh)) = scene.get_mut(id).map(|o| &mut o.kind) else {
                unreachable!()
            };
            let Material::Shader(sm) = Arc::make_mut(&mut mesh.material) else {
                unreachable!()
            };
            f(sm);
        };
        edit(&mut scene, &|sm| sm.storage0 = vec![0.0, 0.5]);
        let px = rig.render(&mut scene);
        assert!(
            near(centre(&px), [0.5, 0.0, 0.0]),
            "replaced: drew {:?}",
            centre(&px)
        );

        edit(&mut scene, &|sm| {
            sm.storage0[1] = 1.0;
            sm.mark_storage_dirty(0);
        });
        let px = rig.render(&mut scene);
        assert!(
            near(centre(&px), [1.0, 0.0, 0.0]),
            "marked: drew {:?}",
            centre(&px)
        );
    });
}

/// A material that stops being drawn has its resources dropped, eventually.
#[test]
fn materials_no_longer_drawn_are_evicted() {
    let Some(mut rig) = Rig::new(true) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let m = Arc::new(Material::Shader(ShaderMaterial::new(UNIFORM_TINT)));
    let mut scene = Scene::new();
    let id = backdrop(&mut scene, &m);
    rig.render(&mut scene);
    assert_eq!(rig.stats().materials_cached, 1);

    if let Some(ObjectKind::Mesh(mesh)) = scene.get_mut(id).map(|o| &mut o.kind) {
        mesh.material = Arc::new(Material::Basic(Default::default()));
    }
    for _ in 0..300 {
        rig.renderer.render(
            &mut scene,
            &PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0),
            &rig.view,
            false,
        );
    }
    assert_eq!(
        rig.stats().materials_cached,
        0,
        "an undrawn material was kept"
    );
}
