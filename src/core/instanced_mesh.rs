use crate::core::BufferGeometry;
use crate::materials::Material;
use crate::math::Matrix4;
use std::sync::Arc;

/// Mesh drawn many times with per-instance transforms. Mirrors three.js's
/// `InstancedMesh`. The renderer uploads `transforms` into a per-instance
/// vertex buffer (mat4 = 4 vec4 attrs).
#[derive(Debug, Clone)]
pub struct InstancedMesh {
    pub geometry: Arc<BufferGeometry>,
    pub material: Arc<Material>,
    pub transforms: Vec<Matrix4>,
    /// Per-instance tint, multiplied into vertex colour. Empty means white.
    ///
    /// Without this the only way to draw a hundred cars in six colours is six
    /// instanced meshes, which is six draw calls — and the same again for
    /// every pose, every skin tone and every livery. One buffer with a colour
    /// in it collapses all of that to one draw per mesh.
    ///
    /// Shorter than `transforms` is fine: instances past its end are white.
    pub colors: Vec<crate::math::Color>,
}

impl InstancedMesh {
    pub fn new(geometry: BufferGeometry, material: Material, count: usize) -> Self {
        Self {
            geometry: Arc::new(geometry),
            material: Arc::new(material),
            transforms: vec![Matrix4::identity(); count],
            colors: Vec::new(),
        }
    }

    pub fn set_matrix_at(&mut self, i: usize, m: Matrix4) {
        if i < self.transforms.len() {
            self.transforms[i] = m;
        }
    }

    pub fn count(&self) -> usize {
        self.transforms.len()
    }
}
