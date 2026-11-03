//! Warming a lazy renderer's pipelines ahead of the first draw, and keeping
//! them in a pipeline cache between runs: everything gets built, from any
//! number of threads, and the pixels stay the eager renderer's.
#![cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Mutex};
use std::time::Instant;
use threers::gpu::GpuContext;
use threers::renderer::{CallbackStore, FileStore, MemoryStore, PipelineCache, PipelineCacheStore, RendererOptions};
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Mesh, Object3D, PerspectiveCamera,
    RenderTarget, Renderer, Scene, StandardMaterial, Vector3,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const N: u32 = 64;

fn gpu() -> Option<GpuContext> {
    match GpuContext::headless_blocking() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

fn scene() -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x182030);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xc08040)).with_roughness(0.6).into(),
    )));
    scene.add_light(AmbientLight::new(Color::WHITE, 0.5));
    scene.add_light(DirectionalLight::new(Color::WHITE, 1.0).with_direction(Vector3::new(-0.5, -1.0, -0.4)));
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(2.5, 2.0, 3.5);
    camera.look_at(Vector3::ZERO);
    (scene, camera)
}

fn image(gpu: &GpuContext, renderer: &mut Renderer) -> Vec<u8> {
    renderer.set_reversed_z(false);
    let (mut scene, camera) = scene();
    let target = RenderTarget::new(&gpu.device, N, N, FORMAT);
    target.set_samples(4);
    target.set_presents(true);
    renderer.render_to(&mut scene, &camera, &target);
    threers::renderer::readback::texture_bytes(&gpu.device, &gpu.queue, &target.color_texture)
}

fn eager_image(gpu: &GpuContext) -> Vec<u8> {
    let mut renderer = Renderer::new(gpu.device.clone(), gpu.queue.clone(), FORMAT, N, N);
    image(gpu, &mut renderer)
}

fn lazy(gpu: &GpuContext) -> Renderer {
    Renderer::new_lazy(gpu.device.clone(), gpu.queue.clone(), FORMAT, N, N)
}

#[test]
fn warm_up_builds_every_pending_pipeline_across_threads() {
    let Some(gpu) = gpu() else { return };
    let validation = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut renderer = lazy(&gpu);
    let pending = renderer.pending_pipelines();
    assert!(pending > 0, "a lazy renderer starts with pipelines to build");
    assert!(!renderer.pending_pipeline_labels().is_empty());

    let warmup = renderer.warm_up();
    assert_eq!(warmup.len(), pending);
    assert!(!warmup.is_ready());
    let start = Instant::now();
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let w = warmup.clone();
            std::thread::spawn(move || w.run())
        })
        .collect();
    warmup.wait();
    let built: usize = workers.into_iter().map(|w| w.join().unwrap()).sum();
    eprintln!("[warmup] {pending} pipelines on 4 threads in {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    assert_eq!(built, pending, "each pipeline built exactly once");
    assert!(warmup.is_ready());
    assert_eq!(warmup.progress(), (pending, pending));
    assert!(warmup.errors().is_empty(), "{:?}", warmup.errors());
    assert_eq!(renderer.pending_pipelines(), 0);
    assert!(renderer.warm_up().is_empty(), "nothing left for a second warm-up");

    assert!(image(&gpu, &mut renderer) == eager_image(&gpu), "warmed lazy pixels differ from eager");
    assert!(pollster::block_on(validation.pop()).is_none(), "wgpu validation error");
}

#[test]
fn warm_up_matching_builds_only_what_it_keeps() {
    let Some(gpu) = gpu() else { return };
    let renderer = lazy(&gpu);
    let before = renderer.pending_pipelines();
    let label = renderer.pending_pipeline_labels().into_iter().next().unwrap();
    let warmup = renderer.warm_up_matching(|l| l == label);
    assert!(!warmup.is_empty());
    assert!(warmup.len() < before, "the filter kept everything");
    assert_eq!(warmup.run(), warmup.len());
    assert_eq!(renderer.pending_pipelines(), before - warmup.len());
    assert!(!renderer.pending_pipeline_labels().contains(&label));
}

#[test]
fn warm_up_steps_one_pipeline_at_a_time() {
    let Some(gpu) = gpu() else { return };
    let renderer = lazy(&gpu);
    let warmup = renderer.warm_up();
    let mut steps = 0;
    while warmup.step() {
        steps += 1;
        assert_eq!(warmup.progress(), (steps, warmup.len()));
    }
    assert_eq!(steps, warmup.len());
    assert!(warmup.is_ready());
    assert!(!warmup.step(), "a finished job stays finished");
}

#[test]
fn an_eager_renderer_has_nothing_to_warm() {
    let Some(gpu) = gpu() else { return };
    let renderer = Renderer::new(gpu.device.clone(), gpu.queue.clone(), FORMAT, N, N);
    assert_eq!(renderer.pending_pipelines(), 0);
    let warmup = renderer.warm_up();
    assert!(warmup.is_empty() && warmup.is_ready());
    warmup.wait(); // returns at once
    assert_eq!(warmup.run(), 0);
}

#[test]
fn warm_scene_builds_what_the_scene_draws_with() {
    let Some(gpu) = gpu() else { return };
    let validation = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut renderer = lazy(&gpu);
    renderer.set_reversed_z(false);
    let before = renderer.pending_pipelines();
    let (mut scene, camera) = scene();
    renderer.warm_scene(&mut scene, &camera);
    let after = renderer.pending_pipelines();
    assert!(after < before, "drawing the scene built none of its pipelines ({before} → {after})");
    assert!(image(&gpu, &mut renderer) == eager_image(&gpu), "warmed-by-scene pixels differ from eager");
    assert!(pollster::block_on(validation.pop()).is_none(), "wgpu validation error");
}

/// The cache through each kind of store, cold then from what was saved:
/// pixels stay the eager renderer's, and the second run reads the bytes
/// the first wrote.
#[test]
fn pipeline_cache_round_trips_through_every_store() {
    let Some(gpu) = gpu() else { return };
    let info = gpu.adapter.get_info();
    if PipelineCache::in_memory(&gpu.device, &info).is_none() {
        assert!(
            info.backend != wgpu::Backend::Vulkan
                || !gpu.adapter.features().contains(wgpu::Features::PIPELINE_CACHE),
            "Vulkan with PIPELINE_CACHE, and no cache: GpuContext did not ask for the feature"
        );
        eprintln!("skipping: no pipeline cache on {:?}", info.backend);
        return;
    }
    let validation = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let eager = eager_image(&gpu);
    let tmp = std::env::temp_dir().join(format!("threers-pipeline-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);

    let memory = MemoryStore::new();
    let callback_bytes = Arc::new(Mutex::new(None::<Vec<u8>>));
    let stores: Vec<(&str, Arc<dyn PipelineCacheStore>)> = vec![
        ("directory", Arc::new(tmp.join("dir"))),
        ("file", Arc::new(FileStore(tmp.join("exact/cache.bin")))),
        ("memory", Arc::new(memory.clone())),
        ("callback", {
            let (load, save) = (callback_bytes.clone(), callback_bytes.clone());
            Arc::new(CallbackStore::new(
                move |_: &str| load.lock().unwrap().clone(),
                move |_: &str, data: &[u8]| {
                    *save.lock().unwrap() = Some(data.to_vec());
                    Ok(())
                },
            ))
        }),
    ];
    for (name, store) in stores {
        for run in ["cold", "warm"] {
            let cache = PipelineCache::open(&gpu.device, &info, store.clone()).unwrap();
            assert_eq!(store.load(cache.key()).is_some(), run == "warm", "{name} {run}: what the store held");
            let options = RendererOptions::default().lazy(true).pipeline_cache(&cache);
            let mut renderer = Renderer::with_options(gpu.device.clone(), gpu.queue.clone(), FORMAT, N, N, options);
            assert!(renderer.pipeline_cache().is_some());
            let start = Instant::now();
            let warmup = renderer.warm_up();
            warmup.run();
            eprintln!(
                "[cache] {name} {run}: {} pipelines in {:.1} ms",
                warmup.len(),
                start.elapsed().as_secs_f64() * 1e3
            );
            assert!(warmup.errors().is_empty(), "{:?}", warmup.errors());
            assert!(image(&gpu, &mut renderer) == eager, "pixels through the cache differ ({name} {run})");
            assert!(cache.save().unwrap(), "the driver gave no cache data");
            let saved = store.load(cache.key()).expect("saved bytes");
            assert!(!saved.is_empty());
            assert_eq!(Some(saved.len()), cache.data().map(|d| d.len()));
        }
    }
    // Bytes handed back by the app: what `data` gave, through `from_bytes`.
    let bytes = memory.get(&wgpu::util::pipeline_cache_key(&info).unwrap()).unwrap();
    let cache = PipelineCache::from_bytes(&gpu.device, &info, Some(&bytes)).unwrap();
    let mut renderer = Renderer::with_options(
        gpu.device.clone(),
        gpu.queue.clone(),
        FORMAT,
        N,
        N,
        RendererOptions::default().pipeline_cache(&cache),
    );
    assert!(image(&gpu, &mut renderer) == eager, "pixels from app-held bytes differ");
    // Garbage is dropped, not trusted.
    let junk = PipelineCache::from_bytes(&gpu.device, &info, Some(b"not a pipeline cache")).unwrap();
    let mut renderer = Renderer::with_options(
        gpu.device.clone(),
        gpu.queue.clone(),
        FORMAT,
        N,
        N,
        RendererOptions::default().pipeline_cache(&junk),
    );
    assert!(image(&gpu, &mut renderer) == eager, "pixels after junk cache bytes differ");
    assert!(pollster::block_on(validation.pop()).is_none(), "wgpu validation error");
    let _ = std::fs::remove_dir_all(&tmp);
}
