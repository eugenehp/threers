//! Browser bindings: one run of the tour, handed to JavaScript as flat buffers.
//!
//! Geometry crosses once and poses cross once. A frame is seven floats per part
//! and the player asks for a different one sixty times a second; building JS
//! objects at that rate would cost more than the physics did.

use crate::Tour;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    /// `performance.now()` — the clock the page itself is timed against.
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn perf_now() -> f64;
}

pub(crate) fn now_ms() -> f64 {
    perf_now()
}

/// The model the page opens on.
#[wasm_bindgen(js_name = defaultModel)]
pub fn default_model() -> String {
    crate::MODEL.to_string()
}

/// One run: read the model, assemble it, check it, simulate it, verify it.
#[wasm_bindgen]
pub struct Run {
    tour: Tour,
}

#[wasm_bindgen]
impl Run {
    /// Everything, from source. Returns the parse or assembly error as it is
    /// written, because that error is the useful half of editing a model.
    #[wasm_bindgen(constructor)]
    pub fn new(source: &str, frames: u32, fps: u32, units_per_metre: f32) -> Result<Run, JsError> {
        console_error_panic_hook::set_once();
        match Tour::run(source, frames as usize, fps, units_per_metre) {
            Ok(tour) => Ok(Run { tour }),
            Err(e) => Err(JsError::new(&e)),
        }
    }

    // ---- the transcript ----------------------------------------------------

    /// The whole thing, as the terminal example prints it.
    pub fn transcript(&self) -> String {
        self.tour.transcript()
    }

    #[wasm_bindgen(js_name = actCount)]
    pub fn act_count(&self) -> u32 {
        self.tour.acts.len() as u32
    }

    #[wasm_bindgen(js_name = actTitle)]
    pub fn act_title(&self, i: u32) -> String {
        self.tour
            .acts
            .get(i as usize)
            .map(|a| a.title.clone())
            .unwrap_or_default()
    }

    #[wasm_bindgen(js_name = actBody)]
    pub fn act_body(&self, i: u32) -> String {
        self.tour
            .acts
            .get(i as usize)
            .map(|a| a.body.clone())
            .unwrap_or_default()
    }

    // ---- what to draw ------------------------------------------------------

    /// Part names, newline separated, in pose order.
    #[wasm_bindgen(js_name = partNames)]
    pub fn part_names(&self) -> String {
        self.tour.parts.join("\n")
    }

    #[wasm_bindgen(js_name = pieceCount)]
    pub fn piece_count(&self) -> u32 {
        self.tour.pieces.len() as u32
    }

    /// Which part's pose moves this piece.
    #[wasm_bindgen(js_name = pieceOwner)]
    pub fn piece_owner(&self, i: u32) -> u32 {
        self.tour
            .pieces
            .get(i as usize)
            .map_or(0, |p| p.owner as u32)
    }

    /// Linear RGBA, from the model's own `color()`.
    #[wasm_bindgen(js_name = pieceColor)]
    pub fn piece_color(&self, i: u32) -> Vec<f32> {
        self.tour
            .pieces
            .get(i as usize)
            .map_or_else(Vec::new, |p| p.color.to_vec())
    }

    /// Three vertices per triangle, in the part's own frame.
    #[wasm_bindgen(js_name = piecePositions)]
    pub fn piece_positions(&self, i: u32) -> Vec<f32> {
        self.tour
            .pieces
            .get(i as usize)
            .map_or_else(Vec::new, |p| p.positions.clone())
    }

    #[wasm_bindgen(js_name = pieceNormals)]
    pub fn piece_normals(&self, i: u32) -> Vec<f32> {
        self.tour
            .pieces
            .get(i as usize)
            .map_or_else(Vec::new, |p| p.normals.clone())
    }

    // ---- how it moved ------------------------------------------------------

    /// Every frame's poses, back to back: `frame * partCount * 7`, position
    /// then quaternion.
    ///
    /// One view over the whole run rather than a copy per frame — the player
    /// slices it, which costs nothing.
    pub fn poses(&self) -> Vec<f32> {
        self.tour.poses.clone()
    }

    #[wasm_bindgen(js_name = poseStride)]
    pub fn pose_stride(&self) -> u32 {
        self.tour.stride as u32
    }

    #[wasm_bindgen(js_name = frameCount)]
    pub fn frame_count(&self) -> u32 {
        self.tour.frames as u32
    }

    pub fn fps(&self) -> u32 {
        self.tour.fps
    }

    /// `[minX, minY, minZ, maxX, maxY, maxZ]` over the whole run, so a camera
    /// framed on it never has to re-frame.
    pub fn bounds(&self) -> Vec<f32> {
        self.tour.bounds.to_vec()
    }

    // ---- what the joints did ----------------------------------------------

    #[wasm_bindgen(js_name = jointCount)]
    pub fn joint_count(&self) -> u32 {
        self.tour.joints.len() as u32
    }

    #[wasm_bindgen(js_name = jointName)]
    pub fn joint_name(&self, i: u32) -> String {
        self.tour
            .joints
            .get(i as usize)
            .map(|j| j.name.clone())
            .unwrap_or_default()
    }

    #[wasm_bindgen(js_name = jointKind)]
    pub fn joint_kind(&self, i: u32) -> String {
        self.tour
            .joints
            .get(i as usize)
            .map(|j| j.kind.clone())
            .unwrap_or_default()
    }

    #[wasm_bindgen(js_name = jointUnit)]
    pub fn joint_unit(&self, i: u32) -> String {
        self.tour
            .joints
            .get(i as usize)
            .map(|j| j.unit.clone())
            .unwrap_or_default()
    }

    /// The reading at every frame, in the units it was declared in.
    #[wasm_bindgen(js_name = jointValues)]
    pub fn joint_values(&self, i: u32) -> Vec<f32> {
        self.tour
            .joints
            .get(i as usize)
            .map_or_else(Vec::new, |j| j.values.clone())
    }

    /// The declared travel, or empty when the model declared none.
    #[wasm_bindgen(js_name = jointLimit)]
    pub fn joint_limit(&self, i: u32) -> Vec<f32> {
        self.tour
            .joints
            .get(i as usize)
            .and_then(|j| j.limit)
            .map_or_else(Vec::new, |l| l.to_vec())
    }

    // ---- the numbers the header shows -------------------------------------

    #[wasm_bindgen(js_name = partCount)]
    pub fn part_count(&self) -> u32 {
        self.tour.facts.parts as u32
    }

    #[wasm_bindgen(js_name = mateCount)]
    pub fn mate_count(&self) -> u32 {
        self.tour.facts.mates as u32
    }

    #[wasm_bindgen(js_name = driveCount)]
    pub fn drive_count(&self) -> u32 {
        self.tour.facts.drives as u32
    }

    pub fn triangles(&self) -> u32 {
        self.tour.facts.triangles as u32
    }

    pub fn mobility(&self) -> i32 {
        self.tour.facts.mobility
    }

    pub fn interferences(&self) -> u32 {
        self.tour.facts.interferences as u32
    }

    pub fn untestable(&self) -> u32 {
        self.tour.facts.unchecked as u32
    }

    pub fn ungrounded(&self) -> bool {
        self.tour.facts.ungrounded
    }

    #[wasm_bindgen(js_name = poseBytes)]
    pub fn pose_bytes(&self) -> f64 {
        self.tour.facts.pose_bytes as f64
    }

    #[wasm_bindgen(js_name = bakedBytes)]
    pub fn baked_bytes(&self) -> f64 {
        self.tour.facts.baked_bytes as f64
    }

    pub fn keys(&self) -> u32 {
        self.tour.facts.keys as u32
    }

    #[wasm_bindgen(js_name = reducedKeys)]
    pub fn reduced_keys(&self) -> u32 {
        self.tour.facts.reduced_keys as u32
    }

    pub fn confirmed(&self) -> u32 {
        self.tour.facts.confirmed as u32
    }

    pub fn agrees(&self) -> bool {
        self.tour.facts.agrees
    }

    pub fn replayed(&self) -> bool {
        self.tour.facts.replayed
    }

    /// Time in the solver for the run that was recorded, in milliseconds.
    #[wasm_bindgen(js_name = simulatedMs)]
    pub fn simulated_ms(&self) -> f64 {
        self.tour.facts.simulated_ms
    }
}
