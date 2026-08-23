use super::BufferGeometry;
use crate::materials::Material;
use std::sync::Arc;

/// A drawable: geometry + material. The renderer caches per-mesh GPU buffers
/// using identity of these Arcs.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub geometry: Arc<BufferGeometry>,
    pub material: Arc<Material>,
    /// Per-morph-target blend weights (glTF / three.js `morphTargetInfluences`).
    pub morph_influences: Vec<f32>,
}

impl Mesh {
    pub fn new(geometry: BufferGeometry, material: Material) -> Self {
        Self {
            geometry: Arc::new(geometry),
            material: Arc::new(material),
            morph_influences: Vec::new(),
        }
    }

    pub fn from_arc(geometry: Arc<BufferGeometry>, material: Arc<Material>) -> Self {
        Self {
            geometry,
            material,
            morph_influences: Vec::new(),
        }
    }
}
