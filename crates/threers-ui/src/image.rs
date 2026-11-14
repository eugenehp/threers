//! Images in the tree: decoded pixels in a node, sized by their own
//! dimensions and fitted to their box with `object-fit`; and a cache that
//! loads them off the main thread.
//!
//! An [`Image`] is decoded, straight-alpha sRGB RGBA. Decoding is the app's
//! (threers has PNG and JPEG decoders; a browser has `createImageBitmap`),
//! so this crate stays free of codecs. A node shows one with
//! [`Ui::add_image`]: with no `width` or `height` it is as large as the
//! image in pixels, with one of them it keeps its aspect ratio, and
//! [`ObjectFit`] decides how it fills a box of another shape.
//!
//! ```
//! use threers_ui::image::Image;
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let photo = Image::new(400, 300, vec![200; 400 * 300 * 4]).unwrap();
//! let node = ui.add_image(root, style! { width: 200px; }, photo);
//! ui.layout(800.0, 600.0, &mut FixedWidthText(7.0));
//! assert_eq!(ui.rect(node).height, 150.0);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::style::{ObjectFit, Style};
use super::tree::{NodeId, NodeKind, Rect, Ui};

/// Decoded pixels: `width × height`, RGBA8, sRGB-encoded, straight alpha,
/// rows from the top. Compared by identity: two images are equal when they
/// are the same image, so a scene or a damage check never compares pixels.
pub struct Image {
    id: u64,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// An image shared between the tree, scenes and painters.
pub type ImageRef = Arc<Image>;

impl Image {
    /// Fails when `rgba` is not `width × height × 4` bytes.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Result<ImageRef, String> {
        let want = width as usize * height as usize * 4;
        if rgba.len() != want || width == 0 || height == 0 {
            return Err(format!(
                "{width}×{height} RGBA is {want} bytes, not {}",
                rgba.len()
            ));
        }
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Ok(Arc::new(Image {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            width,
            height,
            rgba,
        }))
    }

    /// Unique to this image, for caches keyed by image.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The pixel at `(x, y)`.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ]
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Image#{} {}×{}", self.id, self.width, self.height)
    }
}

/// Where an image of `iw × ih` goes in `content` under `fit`: the rectangle
/// it is drawn to, which for [`ObjectFit::Cover`] and [`ObjectFit::None`]
/// can be larger than the box (the box clips it). Centred, as
/// `object-position: 50% 50%` places it.
pub fn fit(fit: ObjectFit, content: Rect, iw: f32, ih: f32) -> Rect {
    let (bw, bh) = (content.width, content.height);
    let (w, h) = match fit {
        ObjectFit::Fill => (bw, bh),
        ObjectFit::None => (iw, ih),
        ObjectFit::Contain | ObjectFit::Cover | ObjectFit::ScaleDown => {
            let sx = bw / iw.max(1e-6);
            let sy = bh / ih.max(1e-6);
            let mut s = if fit == ObjectFit::Cover {
                sx.max(sy)
            } else {
                sx.min(sy)
            };
            if fit == ObjectFit::ScaleDown {
                s = s.min(1.0);
            }
            (iw * s, ih * s)
        }
    };
    Rect {
        x: content.x + (bw - w) / 2.0,
        y: content.y + (bh - h) / 2.0,
        width: w,
        height: h,
    }
}

impl Ui {
    /// An image child of `parent`, styled `style` (see the
    /// [module docs](self) for how it is sized).
    #[track_caller]
    pub fn add_image(&mut self, parent: NodeId, style: Style, image: ImageRef) -> NodeId {
        self.add_kind(parent, style, NodeKind::Image { image })
    }

    /// A node showing the renderer's texture `id` (see
    /// [`NodeKind::Surface`]), `size` texels, sized and fitted like an image.
    #[track_caller]
    pub fn add_surface(
        &mut self,
        parent: NodeId,
        style: Style,
        id: u64,
        size: (u32, u32),
    ) -> NodeId {
        self.add_kind(
            parent,
            style,
            NodeKind::Surface {
                id,
                width: size.0.max(1),
                height: size.1.max(1),
            },
        )
    }

    /// Replaces an image node's image.
    #[track_caller]
    pub fn set_image(&mut self, id: NodeId, image: ImageRef) {
        // Another picture of the same size takes the same room: only drawn.
        if let NodeKind::Image { image: old } = self.kind(id)
            && (old.width, old.height) == (image.width, image.height)
        {
            if *old != image {
                self.node_mut(id).kind = NodeKind::Image { image };
                self.touch_paint();
            }
            return;
        }
        let kind = NodeKind::Image { image };
        if *self.kind(id) != kind {
            self.set_kind(id, kind);
        }
    }
}

/// Where a cached image is.
#[derive(Debug, Clone)]
pub enum ImageState {
    Loading,
    Ready(ImageRef),
    Failed(String),
}

struct Entry {
    state: ImageState,
    used: u64,
}

#[cfg(not(target_arch = "wasm32"))]
type Done = (String, Result<ImageRef, String>);

/// Images by key (a path, a URL), loaded off the main thread and dropped
/// when no frame has asked for them in a while. See the [module docs](self).
pub struct ImageCache {
    entries: HashMap<String, Entry>,
    frame: u64,
    /// Frames an image is kept after the last one that asked for it.
    pub keep_frames: u64,
    #[cfg(not(target_arch = "wasm32"))]
    done: (
        std::sync::mpsc::Sender<Done>,
        std::sync::mpsc::Receiver<Done>,
    ),
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageCache {
    pub fn new() -> Self {
        ImageCache {
            entries: HashMap::new(),
            frame: 0,
            keep_frames: 120,
            #[cfg(not(target_arch = "wasm32"))]
            done: std::sync::mpsc::channel(),
        }
    }

    /// The image under `key`, if it has been asked for, marking it used
    /// this frame.
    pub fn get(&mut self, key: &str) -> Option<&ImageState> {
        let frame = self.frame;
        self.entries.get_mut(key).map(|e| {
            e.used = frame;
            &e.state
        })
    }

    /// The image under `key`; the first time, `load` starts producing it
    /// on another thread (in a browser, where there are none, it runs now:
    /// fetch with [`ImageCache::insert`] instead). Until it is ready the
    /// state is [`ImageState::Loading`] and [`poll`](Self::poll) says when
    /// that changes.
    pub fn get_or_load(
        &mut self,
        key: &str,
        load: impl FnOnce() -> Result<ImageRef, String> + Send + 'static,
    ) -> &ImageState {
        if !self.entries.contains_key(key) {
            self.entries.insert(
                key.to_owned(),
                Entry {
                    state: ImageState::Loading,
                    used: self.frame,
                },
            );
            #[cfg(not(target_arch = "wasm32"))]
            {
                let tx = self.done.0.clone();
                let key = key.to_owned();
                std::thread::spawn(move || {
                    let _ = tx.send((key, load()));
                });
            }
            #[cfg(target_arch = "wasm32")]
            self.insert(key, load());
        }
        self.get(key).expect("just made")
    }

    /// Puts a loaded image (or the reason it failed) under `key`: for
    /// loads the app runs itself.
    pub fn insert(&mut self, key: &str, result: Result<ImageRef, String>) {
        let state = match result {
            Ok(image) => ImageState::Ready(image),
            Err(e) => ImageState::Failed(e),
        };
        let used = self.entries.get(key).map_or(self.frame, |e| e.used);
        self.entries.insert(key.to_owned(), Entry { state, used });
    }

    /// Takes in loads that have finished. Returns whether any did: lay out
    /// and draw again to show them.
    pub fn poll(&mut self) -> bool {
        // In a browser, loads finish where they start; nothing arrives later.
        #[cfg(target_arch = "wasm32")]
        return false;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut any = false;
            while let Ok((key, result)) = self.done.1.try_recv() {
                self.insert(&key, result);
                any = true;
            }
            any
        }
    }

    /// Whether a load is still running.
    pub fn loading(&self) -> bool {
        self.entries
            .values()
            .any(|e| matches!(e.state, ImageState::Loading))
    }

    /// Ends a frame: images no frame has asked for in
    /// [`keep_frames`](Self::keep_frames) are let go (loads in progress
    /// are kept).
    pub fn finish_frame(&mut self) {
        self.frame += 1;
        let (frame, keep) = (self.frame, self.keep_frames);
        self.entries.retain(|_, e| {
            matches!(e.state, ImageState::Loading) || frame.saturating_sub(e.used) <= keep
        });
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: f32, h: f32) -> Rect {
        Rect {
            x: 0.0,
            y: 0.0,
            width: w,
            height: h,
        }
    }

    #[test]
    fn object_fit_places_as_css_does() {
        let b = rect(200.0, 100.0);
        assert_eq!(fit(ObjectFit::Fill, b, 50.0, 50.0), b);
        let c = fit(ObjectFit::Contain, b, 50.0, 50.0);
        assert_eq!((c.x, c.y, c.width, c.height), (50.0, 0.0, 100.0, 100.0));
        let c = fit(ObjectFit::Cover, b, 50.0, 50.0);
        assert_eq!((c.x, c.y, c.width, c.height), (0.0, -50.0, 200.0, 200.0));
        let c = fit(ObjectFit::None, b, 50.0, 50.0);
        assert_eq!((c.x, c.y, c.width, c.height), (75.0, 25.0, 50.0, 50.0));
        let c = fit(ObjectFit::ScaleDown, b, 50.0, 50.0);
        assert_eq!((c.width, c.height), (50.0, 50.0));
        let c = fit(ObjectFit::ScaleDown, b, 400.0, 400.0);
        assert_eq!((c.width, c.height), (100.0, 100.0));
    }

    #[test]
    fn images_are_equal_only_to_themselves() {
        let a = Image::new(1, 1, vec![0; 4]).unwrap();
        let b = Image::new(1, 1, vec![0; 4]).unwrap();
        assert_eq!(a, a.clone());
        assert_ne!(a, b);
        assert!(Image::new(2, 2, vec![0; 4]).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_cache_loads_off_thread_and_forgets_what_is_not_shown() {
        let mut cache = ImageCache::new();
        cache.keep_frames = 2;
        let state = cache.get_or_load("a", || Image::new(1, 1, vec![9; 4]));
        assert!(matches!(state, ImageState::Loading));
        let start = std::time::Instant::now();
        while !cache.poll() {
            assert!(start.elapsed().as_secs() < 5, "never loaded");
            std::thread::yield_now();
        }
        assert!(matches!(cache.get("a"), Some(ImageState::Ready(_))));
        cache.insert("bad", Err("404".into()));
        for _ in 0..3 {
            cache.finish_frame();
        }
        assert!(cache.is_empty(), "unused for three frames");
    }
}
