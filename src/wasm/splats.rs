//! Gaussian splats (`threers::splats`) from JavaScript: a captured scene's file
//! added to a `WebScene`, kept sorted for the camera by every render.

use wasm_bindgen::prelude::*;

use super::{CameraInner, WebCamera, WebObjectHandle, WebScene};
use crate::splats::{splat_object, DepthSort, GaussianSplats, SplatOptions, SplatSorter};

#[wasm_bindgen]
impl WebScene {
    /// Add the splats in a file's bytes: the reference trainer's PLY (plain
    /// or chunk-compressed), Niantic's `.spz`, or the 32-byte-a-splat
    /// `.splat` format, told apart by their first bytes. `.spz` is `y` up;
    /// the others are usually `y` down, as trainers write them.
    /// `maxShDegree` (default 3) caps the view-dependent colour uploaded.
    /// The renderer sorts them for its camera on every `render`.
    #[wasm_bindgen(js_name = addSplats)]
    pub fn add_splats(
        &mut self,
        bytes: &[u8],
        max_sh_degree: Option<u32>,
    ) -> Result<WebObjectHandle, JsError> {
        let splats = GaussianSplats::from_bytes(bytes)?;
        let options = SplatOptions {
            max_sh_degree: max_sh_degree.unwrap_or(3),
            ..Default::default()
        };
        let (object, sorter) = splat_object(&splats, &options)?;
        let id = self.inner.add(object);
        self.splat_sorters.push((id, sorter, frame(&splats)));
        Ok(WebObjectHandle { id })
    }

    /// Where to aim a camera at the splats of `handle`, in their own (object)
    /// space: `[x, y, z, distance]` — the per-axis median of their centres
    /// (trained scenes have far-flung outliers that would drag a box's centre
    /// off), and how far from it to look from. For a scanned object, far
    /// enough to see all of it; for a captured scene, whose background
    /// stretches far past its subject, close, among where its cameras were —
    /// from outside, a scene shows the floaters its training never had to
    /// clear. Empty for a handle that is not a splat object.
    #[wasm_bindgen(js_name = splatFrame)]
    pub fn splat_frame(&self, handle: &WebObjectHandle) -> Vec<f32> {
        self.splat_sorters
            .iter()
            .find(|(id, ..)| *id == handle.id)
            .map_or_else(Vec::new, |(.., f)| f.to_vec())
    }

    /// Leave sorting to the page: `render` stops sorting the splats, and the
    /// order comes from [`setSplatOrder`](Self::set_splat_order) — from a
    /// `WebSplatSort` in a worker, given [`splatCentres`](Self::splat_centres)
    /// once and [`splatViewAxis`](Self::splat_view_axis) each frame, so the
    /// thread that renders never waits on a sort. `false` (the default) sorts
    /// on every `render`.
    #[wasm_bindgen(js_name = setSplatsSortedElsewhere)]
    pub fn set_splats_sorted_elsewhere(&mut self, on: bool) {
        self.splats_sorted_elsewhere = on;
    }

    /// The centres of the splats of `handle`, for a `WebSplatSort`: every
    /// `x`, then every `y`, then every `z`. Empty for a handle that is not a
    /// splat object.
    #[wasm_bindgen(js_name = splatCentres)]
    pub fn splat_centres(&self, handle: &WebObjectHandle) -> Vec<f32> {
        self.sorter(handle).map_or_else(Vec::new, |s| s.centres().concat())
    }

    /// The axis to sort the splats of `handle` along for `camera`: the unit
    /// view direction in their own space, `[x, y, z]`. Empty for a handle that
    /// is not a splat object, or a degenerate view.
    #[wasm_bindgen(js_name = splatViewAxis)]
    pub fn splat_view_axis(&mut self, handle: &WebObjectHandle, camera: &WebCamera) -> Vec<f32> {
        if self.sorter(handle).is_none() {
            return Vec::new();
        }
        let axis = match &camera.inner {
            CameraInner::Perspective(c) => SplatSorter::view_axis(&mut self.inner, handle.id, c),
            CameraInner::Orthographic(c) => SplatSorter::view_axis(&mut self.inner, handle.id, c),
        };
        axis.map_or_else(Vec::new, |a| a.to_vec())
    }

    /// Draw the splats of `handle` in `order` — what a `WebSplatSort` made,
    /// one index a splat, back to front. Throws for an order of the wrong
    /// length or a handle that is not a splat object.
    #[wasm_bindgen(js_name = setSplatOrder)]
    pub fn set_splat_order(&mut self, handle: &WebObjectHandle, order: Vec<f32>) -> Result<(), JsError> {
        let sorter = self
            .splat_sorters
            .iter_mut()
            .find(|(id, ..)| *id == handle.id)
            .map(|(_, s, _)| s)
            .ok_or_else(|| JsError::new("not a splat object"))?;
        if order.len() != sorter.order().len() {
            return Err(JsError::new(&format!(
                "an order of {} for {} splats",
                order.len(),
                sorter.order().len()
            )));
        }
        sorter.set_order(order);
        Ok(())
    }

    /// How many splats the scene's splat objects hold between them.
    #[wasm_bindgen(js_name = splatCount)]
    pub fn splat_count(&self) -> u32 {
        self.splat_sorters
            .iter()
            .map(|(_, s, _)| s.order().len() as u32)
            .sum()
    }
}

impl WebScene {
    fn sorter(&self, handle: &WebObjectHandle) -> Option<&SplatSorter> {
        self.splat_sorters
            .iter()
            .find(|(id, ..)| *id == handle.id)
            .map(|(_, s, _)| s)
    }

    /// Order every splat object back to front for `camera`; a no-op for one
    /// whose view axis has not turned. Objects removed from the scene are
    /// forgotten.
    pub(crate) fn sort_splats<C: crate::cameras::Camera + ?Sized>(&mut self, camera: &C) {
        if self.splat_sorters.is_empty() || self.splats_sorted_elsewhere {
            return;
        }
        let scene = &mut self.inner;
        self.splat_sorters.retain(|(id, ..)| scene.get(*id).is_some());
        for (id, sorter, _) in &mut self.splat_sorters {
            sorter.update(scene, *id, camera);
        }
    }
}

/// `[x, y, z, distance]` for `splatFrame`.
fn frame(splats: &GaussianSplats) -> [f32; 4] {
    let (c, distance) = splats.framing();
    [c.x, c.y, c.z, distance]
}

/// The splat sort on its own, for a Worker: made once from
/// `WebScene.splatCentres`, then asked for an order per view axis
/// (`WebScene.splatViewAxis`), which goes back to the renderer's thread for
/// `WebScene.setSplatOrder`. The radix sort of `threers-splat-sort`, its
/// keys in SIMD where the package was built with it.
#[wasm_bindgen]
pub struct WebSplatSort {
    sort: DepthSort,
    order: Vec<f32>,
}

#[wasm_bindgen]
impl WebSplatSort {
    /// From `centres` as `splatCentres` gives them: every `x`, `y`, `z`.
    #[wasm_bindgen(constructor)]
    pub fn new(centres: &[f32]) -> Result<WebSplatSort, JsError> {
        if centres.len() % 3 != 0 {
            return Err(JsError::new("centres come as three equal columns"));
        }
        let n = centres.len() / 3;
        let (x, rest) = centres.split_at(n);
        let (y, z) = rest.split_at(n);
        Ok(Self {
            sort: DepthSort::from_columns(x.to_vec(), y.to_vec(), z.to_vec()),
            order: Vec::new(),
        })
    }

    /// The order for view axis `(x, y, z)`: splat indices as floats, back
    /// to front.
    pub fn sort(&mut self, x: f32, y: f32, z: f32) -> Vec<f32> {
        self.sort.sort([x, y, z]);
        self.sort.order_f32(&mut self.order);
        self.order.clone()
    }

    /// [`sort`](Self::sort) into `out` (one float a splat), allocating
    /// nothing on the JavaScript side: hand the same buffers back and forth
    /// between threads, and a frame's sort makes no garbage to collect.
    /// Throws for an `out` of the wrong length.
    #[wasm_bindgen(js_name = sortInto)]
    pub fn sort_into(&mut self, x: f32, y: f32, z: f32, out: &mut [f32]) -> Result<(), JsError> {
        if out.len() != self.sort.len() {
            return Err(JsError::new(&format!("an order buffer of {} for {} splats", out.len(), self.sort.len())));
        }
        self.sort.sort([x, y, z]);
        for (o, &i) in out.iter_mut().zip(self.sort.order()) {
            *o = i as f32;
        }
        Ok(())
    }

    /// How many splats it sorts.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> u32 {
        self.sort.len() as u32
    }
}
