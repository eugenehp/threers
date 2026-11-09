//! Bindings split out of `wasm.rs` by area (see the `mod` list there): helpers.

use super::*;

// ---- helpers ----
#[wasm_bindgen]
pub struct WebAxesHelper {
    pub(super) obj: crate::core::Object3D,
}
#[wasm_bindgen]
impl WebAxesHelper {
    #[wasm_bindgen(constructor)]
    pub fn new(size: f32) -> WebAxesHelper {
        WebAxesHelper {
            obj: crate::AxesHelper::new(size),
        }
    }
}

#[wasm_bindgen]
pub struct WebGridHelper {
    pub(super) obj: crate::core::Object3D,
}
#[wasm_bindgen]
impl WebGridHelper {
    #[wasm_bindgen(constructor)]
    pub fn new(size: f32, divisions: usize, color1: u32, color2: u32) -> WebGridHelper {
        WebGridHelper {
            obj: crate::GridHelper::new_from_hex(size, divisions, color1, color2),
        }
    }
}

#[wasm_bindgen]
pub struct WebBoxHelper {
    pub(super) obj: crate::core::Object3D,
}
#[wasm_bindgen]
impl WebBoxHelper {
    #[wasm_bindgen(constructor)]
    pub fn new(bb: &WebBox3) -> WebBoxHelper {
        WebBoxHelper {
            obj: crate::BoxHelper::new(&bb.inner, crate::Color::WHITE),
        }
    }
}

#[wasm_bindgen]
pub struct WebPolarGridHelper {
    pub(super) obj: crate::core::Object3D,
}
#[wasm_bindgen]
impl WebPolarGridHelper {
    #[wasm_bindgen(constructor)]
    pub fn new(radius: f32, segments: usize, circles: usize) -> WebPolarGridHelper {
        WebPolarGridHelper {
            obj: crate::PolarGridHelper::default_(radius, segments, circles),
        }
    }
}

#[wasm_bindgen]
impl WebScene {
    /// Add a helper as a scene-graph child (returns its handle for later removal).
    #[wasm_bindgen(js_name = addHelper)]
    pub fn add_helper(
        &mut self,
        axes: Option<WebAxesHelper>,
        grid: Option<WebGridHelper>,
    ) -> WebObjectHandle {
        let obj = if let Some(a) = axes {
            a.obj
        } else if let Some(g) = grid {
            g.obj
        } else {
            crate::core::Object3D::group()
        };
        let id = self.inner.add(obj);
        WebObjectHandle { id }
    }
}
