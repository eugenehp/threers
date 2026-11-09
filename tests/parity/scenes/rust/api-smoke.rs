//! Parity scene `api-smoke` — native Rust (winit + wgpu).
//!
//! Generated from `tests/parity/scenes/threers-api-smoke.html`.
//! Compare with the JavaScript tab above; adjust imports before copying to `examples/`.
//!
//! ```text
//! cargo run --example api_smoke
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::{
    BasicMaterial,
    BoxGeometry,
    Color,
    Euler,
    ExtrudeGeometry,
    LatheGeometry,
    Mesh,
    Object3D,
    PerspectiveCamera,
    Renderer,
    Scene,
    TubeGeometry,
    Vector2,
    Vector3,
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::ActiveEventLoop,
};


fn main() {
    env_logger::init();
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — api-smoke")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
    threers::sketch::run_window(attributes, |window| pollster::block_on(setup(window)));
}

/// The scene for a window that has just opened, and the handler that draws it.
async fn setup(window: Arc<winit::window::Window>) -> impl FnMut(Event<()>, &ActiveEventLoop) {

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
    // ---- Smoke: construct/use a swath of the new shim API surface. // If any of these throw, the parity harness reports an error. // Verify Sprite / InstancedMesh / SkinnedMesh / Skeleton / Bone exist and instantiate cleanly. new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xff0000 }));
    // new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1), new THREE.MeshBasicMaterial({ color: 0x00ff00 }), 4);
    // const bones = [new THREE.Bone(), new THREE.Bone()];
    // const skel = new THREE.Skeleton(bones);
    // if (skel.bones.length !== 2) throw new Error('Skeleton.bones broken: ' + skel.bones.length);
    // const sm = new THREE.SkinnedMesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());
    // sm.bind(skel);
    // if (!sm.skeleton) throw new Error('SkinnedMesh.bind broken');
    // EdgesGeometry / WireframeGeometry should be real, with extracted edges. const eg = new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1));
    // if (!eg.attributes?.position) throw new Error('EdgesGeometry missing position');
    // const wg = new THREE.WireframeGeometry(new THREE.BoxGeometry(1,1,1));
    // if (!wg.attributes?.position) throw new Error('WireframeGeometry missing position');
    // Math const v = new THREE.Vector3(1, 2, 3).add(new THREE.Vector3(4, 5, 6)) .sub(new THREE.Vector3(1, 1, 1)).normalize();
    // if (Math.abs(v.length() - 1) > 1e-5) throw new Error('Vector3.normalize broken');
    // const v2 = new THREE.Vector3(0, 0, 0).lerpVectors(new THREE.Vector3(0,0,0), new THREE.Vector3(2,4,6), 0.5);
    // if (v2.x !== 1 || v2.y !== 2 || v2.z !== 3) throw new Error('lerpVectors broken: ' + JSON.stringify(v2));
    // const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0), Math.PI / 2);
    // const m4 = new THREE.Matrix4().makeTranslation(1, 2, 3).multiply(new THREE.Matrix4().makeRotationY(Math.PI/4));
    // const cInv = new THREE.Matrix4().copy(m4).invert();
    // const test = new THREE.Vector3(0, 0, 0).applyMatrix4(m4).applyMatrix4(cInv);
    // if (Math.abs(test.x) > 1e-4 || Math.abs(test.y) > 1e-4 || Math.abs(test.z) > 1e-4) throw new Error('Matrix4.invert broken: ' + JSON.stringify(test));
    // Color const col = new THREE.Color(0xff8800);
    // if (Math.abs(col.getHex() - 0xff8800) > 0xff) throw new Error('Color.getHex broken: ' + col.getHex().toString(16));
    // new THREE.Color().setHSL(0.5, 1, 0.5);
    // new THREE.Color().setStyle('#aabbcc');
    // Geometries (including new fallback stubs) new THREE.BoxGeometry();
    // new THREE.LatheGeometry();
    // new THREE.TubeGeometry();
    // new THREE.ExtrudeGeometry();
    // new THREE.EdgesGeometry();
    // new THREE.WireframeGeometry();
    // new THREE.ShapeGeometry();
    // new THREE.ConvexGeometry();
    // new THREE.DecalGeometry();
    // new THREE.TextGeometry();
    // new THREE.ParametricGeometry();
    // Assert real triangulators produce position data (not just BoxGeometry fallback). const lath = new THREE.LatheGeometry();
    // if (!lath.attributes?.position?.array?.length) throw new Error('LatheGeometry has no positions');
    // const par = new THREE.ParametricGeometry();
    // if (!par.attributes?.position?.array?.length) throw new Error('ParametricGeometry has no positions');
    // const shg = new THREE.ShapeGeometry();
    // if (!shg.attributes?.position?.array?.length) throw new Error('ShapeGeometry has no positions');
    // Helpers (no-op stubs) new THREE.DirectionalLightHelper();
    // new THREE.PointLightHelper();
    // new THREE.SpotLightHelper();
    // new THREE.CameraHelper();
    // new THREE.ArrowHelper();
    // new THREE.PlaneHelper();
    // new THREE.SkeletonHelper();
    // EventDispatcher const ed = new THREE.EventDispatcher();
    // let fired = 0;
    // const listener = () => { fired++;
    // };
    // ed.addEventListener('foo', listener);
    // ed.dispatchEvent({ type: 'foo' });
    // ed.removeEventListener('foo', listener);
    // ed.dispatchEvent({ type: 'foo' });
    // if (fired !== 1) throw new Error('EventDispatcher fired ' + fired + ', expected 1');
    // Loaders (instantiation only — actual fetch in network smoke tests) const lm = new THREE.LoadingManager();
    // new THREE.FileLoader(lm);
    // new THREE.ImageLoader(lm);
    // new THREE.TextureLoader(lm);
    // new THREE.FontLoader(lm);
    // new THREE.GLTFLoader(lm);
    // new THREE.FBXLoader(lm);
    // new THREE.AudioLoader(lm);
    // new THREE.CubeTextureLoader(lm);
    // new THREE.EXRLoader(lm);
    // Controls const cam = new THREE.PerspectiveCamera(45, 800/600, 0.1, 100);
    // new THREE.DragControls();
    // new THREE.ArcballControls();
    // new THREE.MapControls();
    // new THREE.FlyControls();
    // new THREE.TransformControls();
    // Render targets const rt = new THREE.WebGLRenderTarget(256, 256);
    // rt.setSize(512, 512);
    // rt.dispose();
    // Animation tracks new THREE.NumberKeyframeTrack('.opacity', [0, 1], [0, 1]);
    // new THREE.VectorKeyframeTrack('.position', [0, 1], [0,0,0, 1,1,1]);
    // new THREE.QuaternionKeyframeTrack('.quaternion', [0, 1], [0,0,0,1, 0,1,0,0]);
    // new THREE.ColorKeyframeTrack('.color', [0, 1], [1,0,0, 0,1,0]);
    // const ag = new THREE.AnimationObjectGroup({ position: new THREE.Vector3() });
    // Curves const qbc = new THREE.QuadraticBezierCurve(new THREE.Vector2(0, 0), new THREE.Vector2(0.5, 1), new THREE.Vector2(1, 0));
    // const pts = qbc.getPoints(8);
    // if (pts.length !== 9) throw new Error('QuadraticBezierCurve.getPoints broken: ' + pts.length);
    // new THREE.CubicBezierCurve3(new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3());
    // new THREE.SplineCurve([new THREE.Vector2(0,0), new THREE.Vector2(1,1)]);
    // new THREE.ArcCurve(0, 0, 1, 0, Math.PI, false).getPoints(8);
    // MathUtils if (Math.abs(THREE.MathUtils.degToRad(180) - Math.PI) > 1e-9) throw new Error('MathUtils.degToRad broken');
    // if (THREE.MathUtils.clamp(5, 0, 3) !== 3) throw new Error('MathUtils.clamp broken');
    // if (THREE.MathUtils.lerp(0, 10, 0.5) !== 5) throw new Error('MathUtils.lerp broken');
    // const uuid = THREE.MathUtils.generateUUID();
    // if (!/^[0-9A-F-]{36}$/.test(uuid)) throw new Error('UUID looks wrong: ' + uuid);
    // const s = THREE.MathUtils.smoothstep(0.5, 0, 1);
    // if (Math.abs(s - 0.5) > 1e-9) throw new Error('smoothstep(0.5,0,1) != 0.5: ' + s);
    // Constants const expected = { RGBAFormat: 1023, UnsignedByteType: 1009, NearestFilter: 1003, FrontSide: 0, BackSide: 1, DoubleSide: 2, NoBlending: 0, NormalBlending: 1, SRGBColorSpace: 'srgb', NoToneMapping: 0, ACESFilmicToneMapping: 4, LoopOnce: 2200, LoopRepeat: 2201, BasicDepthPacking: 3200, RGBADepthPacking: 3201 };
    // for (const [k, v] of Object.entries(expected)) { if (THREE[k] !== v) throw new Error(`THREE.${k} expected ${v}, got ${THREE[k]}`);
    // } if (THREE.REVISION !== '165') throw new Error('REVISION expected 165, got ' + THREE.REVISION);
    // ---- Now render a tiny scene so the parity test has a baseline image. ---- const canvas = document.getElementById('c');
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x202028);
    let mut cube = Object3D::mesh(Mesh::new(BoxGeometry::new(1.0, 1.0, 1.0), BasicMaterial::new(Color::from_hex(0x44cc88)).into()));
    cube.quaternion = Euler::new(0.5, 0.5, 0.0).to_quaternion();
    cube.scale = Vector3::splat(0.99985);
    scene.add(cube);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::new(0.0, 0.0, 0.0));

    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::ZERO);

    scene.update_world();
    let window_for_loop = window.clone();

    move |event, target| {
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
                                    renderer.render(&mut scene, &camera, &view, false);
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
        }
}
