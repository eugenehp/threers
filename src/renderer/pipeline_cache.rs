//! A `wgpu::PipelineCache` that outlives the process — on disk, in memory,
//! or wherever the application keeps its state.
//!
//! Lazy pipelines move compilation; a pipeline cache removes it. The driver's
//! compiled pipelines are saved after a run and handed back on the next, so
//! building a pipeline it has seen before skips the shader compiler. That is
//! the difference between a viewport ready in milliseconds on every launch and
//! only on the launches where the driver's own cache happened to keep things.
//!
//! Where the bytes live is a [`PipelineCacheStore`]: a directory (any
//! `PathBuf`), one exact file ([`FileStore`]), memory ([`MemoryStore`] —
//! keep the bytes yourself, hand them to another device, or test with it), or
//! two closures ([`CallbackStore`] — a settings database, a Tauri store, a
//! blob in your own save format). Implement the trait for anything else.
//!
//! wgpu supports this on Vulkan (`Features::PIPELINE_CACHE`, which
//! [`GpuContext`](crate::gpu::GpuContext) asks for wherever the adapter has
//! it). Everywhere else [`PipelineCache::open`] returns `None`, and a renderer
//! given `None` behaves exactly as without a cache — so the same code runs on
//! every platform. Entries are keyed by adapter and driver
//! ([`wgpu::util::pipeline_cache_key`]), and data another GPU or an older
//! driver wrote is checked and dropped by wgpu, never trusted.
//!
//! ```ignore
//! use threers::renderer::{PipelineCache, RendererOptions};
//! let gpu = GpuContext::headless_blocking()?;
//! // Anywhere: a directory…
//! let cache = PipelineCache::in_dir(&gpu.device, &gpu.adapter.get_info(), app_cache_dir);
//! // …or memory, persisted by the app however it likes:
//! // let cache = PipelineCache::from_bytes(&gpu.device, &info, saved.as_deref());
//! let options = RendererOptions::default().lazy(true).pipeline_cache(&cache);
//! let renderer = Renderer::with_options(gpu.device.clone(), gpu.queue.clone(), format, w, h, options);
//! renderer.warm_up().run();
//! if let Some(cache) = &cache { cache.save()?; }      // or `cache.data()` for the bytes
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Where a [`PipelineCache`] keeps its bytes. `key` names the adapter and
/// driver; one store can hold several (a laptop's two GPUs, say).
pub trait PipelineCacheStore: Send + Sync {
    /// The bytes last saved under `key`, if any. A store that cannot read
    /// should say `None`: the cache then starts empty, which is only slower.
    fn load(&self, key: &str) -> Option<Vec<u8>>;
    /// Keep `data` under `key`, replacing what was there.
    fn save(&self, key: &str, data: &[u8]) -> std::io::Result<()>;
}

/// A directory: one file per adapter, named by its key, created on first
/// save. Written through a temporary file and a rename, so a crash mid-write
/// leaves the previous file rather than half of one.
impl PipelineCacheStore for PathBuf {
    fn load(&self, key: &str) -> Option<Vec<u8>> {
        std::fs::read(self.join(key)).ok()
    }

    fn save(&self, key: &str, data: &[u8]) -> std::io::Result<()> {
        write_atomically(&self.join(key), data)
    }
}

/// One exact file, whatever the key — for an app that already decided where
/// its cache goes. Holds one adapter's data; another adapter's replaces it.
#[derive(Clone, Debug)]
pub struct FileStore(pub PathBuf);

impl PipelineCacheStore for FileStore {
    fn load(&self, _key: &str) -> Option<Vec<u8>> {
        std::fs::read(&self.0).ok()
    }

    fn save(&self, _key: &str, data: &[u8]) -> std::io::Result<()> {
        write_atomically(&self.0, data)
    }
}

fn write_atomically(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)
}

/// In memory. Clones share the same entries, so keep one to read the bytes
/// back ([`get`](Self::get)) and persist them however the application likes,
/// or to seed a cache with bytes from elsewhere ([`set`](Self::set)).
#[derive(Clone, Debug, Default)]
pub struct MemoryStore(Arc<Mutex<HashMap<String, Vec<u8>>>>);

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// The bytes saved under `key`.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.lock().get(key).cloned()
    }

    /// Put `data` under `key` — before opening a cache on this store, so it
    /// starts from them.
    pub fn set(&self, key: impl Into<String>, data: Vec<u8>) {
        self.lock().insert(key.into(), data);
    }

    /// Every key held.
    pub fn keys(&self) -> Vec<String> {
        self.lock().keys().cloned().collect()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Vec<u8>>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl PipelineCacheStore for MemoryStore {
    fn load(&self, key: &str) -> Option<Vec<u8>> {
        self.get(key)
    }

    fn save(&self, key: &str, data: &[u8]) -> std::io::Result<()> {
        self.set(key, data.to_vec());
        Ok(())
    }
}

/// Two closures — for a store that is neither a file nor memory: a settings
/// database, a key-value plugin, a section of the app's own save file.
pub struct CallbackStore<L, S> {
    load: L,
    save: S,
}

impl<L, S> CallbackStore<L, S>
where
    L: Fn(&str) -> Option<Vec<u8>> + Send + Sync,
    S: Fn(&str, &[u8]) -> std::io::Result<()> + Send + Sync,
{
    pub fn new(load: L, save: S) -> Self {
        Self { load, save }
    }
}

impl<L, S> PipelineCacheStore for CallbackStore<L, S>
where
    L: Fn(&str) -> Option<Vec<u8>> + Send + Sync,
    S: Fn(&str, &[u8]) -> std::io::Result<()> + Send + Sync,
{
    fn load(&self, key: &str) -> Option<Vec<u8>> {
        (self.load)(key)
    }

    fn save(&self, key: &str, data: &[u8]) -> std::io::Result<()> {
        (self.save)(key, data)
    }
}

impl<T: PipelineCacheStore + ?Sized> PipelineCacheStore for Arc<T> {
    fn load(&self, key: &str) -> Option<Vec<u8>> {
        (**self).load(key)
    }

    fn save(&self, key: &str, data: &[u8]) -> std::io::Result<()> {
        (**self).save(key, data)
    }
}

/// A `wgpu::PipelineCache` and the [`PipelineCacheStore`] it loads from and
/// saves to. Cheap to clone; clones are the same cache.
///
/// Give it to renderers with [`RendererOptions::pipeline_cache`](super::RendererOptions::pipeline_cache)
/// (`.pipeline_cache(&cache)`), or take the wgpu object with
/// [`wgpu`](Self::wgpu) for pipelines of your own.
#[derive(Clone)]
pub struct PipelineCache {
    cache: wgpu::PipelineCache,
    key: String,
    store: Arc<dyn PipelineCacheStore>,
}

impl PipelineCache {
    /// The cache `store` holds for this adapter, empty if it holds none.
    /// `None` when the device was opened without `Features::PIPELINE_CACHE`
    /// or the backend has no pipeline cache (everything but Vulkan).
    pub fn open(device: &wgpu::Device, adapter: &wgpu::AdapterInfo, store: impl PipelineCacheStore + 'static) -> Option<Self> {
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return None;
        }
        let key = wgpu::util::pipeline_cache_key(adapter)?;
        let data = store.load(&key);
        // SAFETY: wgpu validates the data's header against this adapter and
        // driver, and with `fallback` drops data that does not match rather
        // than failing. What it cannot prove is that bytes it wrote were not
        // tampered with on the way back; a store is the application's own.
        let cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("threers pipeline cache"),
                data: data.as_deref(),
                fallback: true,
            })
        };
        Some(Self { cache, key, store: Arc::new(store) })
    }

    /// [`open`](Self::open) on a directory: one file per adapter in `dir`.
    pub fn in_dir(device: &wgpu::Device, adapter: &wgpu::AdapterInfo, dir: impl Into<PathBuf>) -> Option<Self> {
        Self::open(device, adapter, dir.into())
    }

    /// [`open`](Self::open) in memory, starting empty. Read the bytes with
    /// [`data`](Self::data) when you want them.
    pub fn in_memory(device: &wgpu::Device, adapter: &wgpu::AdapterInfo) -> Option<Self> {
        Self::open(device, adapter, MemoryStore::new())
    }

    /// [`open`](Self::open) in memory, starting from `bytes` — what an
    /// earlier [`data`](Self::data) returned, kept wherever the app keeps
    /// things. Stale or foreign bytes are dropped, not trusted.
    pub fn from_bytes(device: &wgpu::Device, adapter: &wgpu::AdapterInfo, bytes: Option<&[u8]>) -> Option<Self> {
        let store = MemoryStore::new();
        if let (Some(bytes), Some(key)) = (bytes, wgpu::util::pipeline_cache_key(adapter)) {
            store.set(key, bytes.to_vec());
        }
        Self::open(device, adapter, store)
    }

    /// The wgpu cache, for pipelines of your own.
    pub fn wgpu(&self) -> &wgpu::PipelineCache {
        &self.cache
    }

    /// What the store files it under: the adapter and driver.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Everything the driver has compiled into it so far.
    pub fn data(&self) -> Option<Vec<u8>> {
        self.cache.get_data()
    }

    /// Write [`data`](Self::data) to the store — after a warm-up, on exit,
    /// or now and then. `Ok(false)`: the backend had nothing to give.
    pub fn save(&self) -> std::io::Result<bool> {
        let Some(data) = self.cache.get_data() else {
            return Ok(false);
        };
        self.store.save(&self.key, &data)?;
        Ok(true)
    }
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache").field("key", &self.key).finish_non_exhaustive()
    }
}

/// What [`RendererOptions::pipeline_cache`](super::RendererOptions::pipeline_cache)
/// takes: a threers [`PipelineCache`] — by reference, or the `Option` that
/// [`PipelineCache::open`] returns, as it comes — or a `wgpu::PipelineCache`.
pub trait IntoPipelineCache {
    fn into_pipeline_cache(self) -> Option<wgpu::PipelineCache>;
}

impl IntoPipelineCache for wgpu::PipelineCache {
    fn into_pipeline_cache(self) -> Option<wgpu::PipelineCache> {
        Some(self)
    }
}

impl IntoPipelineCache for Option<wgpu::PipelineCache> {
    fn into_pipeline_cache(self) -> Option<wgpu::PipelineCache> {
        self
    }
}

impl IntoPipelineCache for &PipelineCache {
    fn into_pipeline_cache(self) -> Option<wgpu::PipelineCache> {
        Some(self.cache.clone())
    }
}

impl IntoPipelineCache for PipelineCache {
    fn into_pipeline_cache(self) -> Option<wgpu::PipelineCache> {
        Some(self.cache)
    }
}

impl IntoPipelineCache for Option<&PipelineCache> {
    fn into_pipeline_cache(self) -> Option<wgpu::PipelineCache> {
        self.map(|c| c.cache.clone())
    }
}

impl IntoPipelineCache for &Option<PipelineCache> {
    fn into_pipeline_cache(self) -> Option<wgpu::PipelineCache> {
        self.as_ref().map(|c| c.cache.clone())
    }
}

impl IntoPipelineCache for Option<PipelineCache> {
    fn into_pipeline_cache(self) -> Option<wgpu::PipelineCache> {
        self.map(|c| c.cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_store_round_trips_and_leaves_no_temporary() {
        let dir = std::env::temp_dir().join(format!("threers-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(dir.load("k"), None);
        dir.save("k", b"one").unwrap();
        dir.save("k", b"two").unwrap();
        assert_eq!(dir.load("k").as_deref(), Some(&b"two"[..]));
        let names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names, ["k"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_store_ignores_the_key() {
        let path = std::env::temp_dir().join(format!("threers-store-{}/nested/cache.bin", std::process::id()));
        let store = FileStore(path.clone());
        store.save("a", b"bytes").unwrap();
        assert_eq!(store.load("b").as_deref(), Some(&b"bytes"[..]));
        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn memory_store_clones_share_entries() {
        let a = MemoryStore::new();
        let b = a.clone();
        a.save("k", b"x").unwrap();
        assert_eq!(b.get("k").as_deref(), Some(&b"x"[..]));
        assert_eq!(b.keys(), ["k"]);
    }

    #[test]
    fn a_callback_store_calls_through() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        let store = CallbackStore::new(
            |key: &str| (key == "k").then(|| b"hit".to_vec()),
            move |key: &str, data: &[u8]| {
                log.lock().unwrap().push((key.to_owned(), data.to_vec()));
                Ok(())
            },
        );
        assert_eq!(store.load("k").as_deref(), Some(&b"hit"[..]));
        assert_eq!(store.load("other"), None);
        store.save("k", b"d").unwrap();
        assert_eq!(*seen.lock().unwrap(), [("k".to_owned(), b"d".to_vec())]);
    }
}
