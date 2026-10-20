use crate::core::BufferGeometry;
use crate::materials::Material;
use crate::math::Matrix4;
use std::collections::HashMap;
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
    /// three.js's material *array* (`InstancedMesh` extends `Mesh`): empty
    /// draws the whole draw range with [`material`](Self::material);
    /// otherwise every geometry group is drawn, instanced, with
    /// `materials[group.material_index]` — see [`Mesh`](super::Mesh).
    pub materials: Vec<Arc<Material>>,
    /// `morphTargetInfluences`, shared by every instance that has no row in
    /// [`morph_texture`](Self::morph_texture) — three.js's uniform path.
    pub morph_influences: Vec<f32>,
    /// three.js `morphTargetDictionary`.
    pub morph_target_dictionary: HashMap<String, usize>,
    /// three.js's `InstancedMesh.morphTexture` (r159+): one row per instance,
    /// `morph_influences.len() + 1` floats each — the instance's
    /// `morphTargetBaseInfluence`, then its influences. Written by
    /// [`set_morph_at`](Self::set_morph_at); empty until then, and while it
    /// is empty every instance blends with
    /// [`morph_influences`](Self::morph_influences).
    pub morph_texture: Vec<f32>,
    /// Two per-instance custom columns of one to four floats each — deck.gl's
    /// custom attributes — for a [`ShaderMaterial`](crate::ShaderMaterial).
    /// Its vertex stage reads them as `in.icustom0` and `in.icustom1`, each a
    /// `vec4<f32>` zero-padded from [`custom_components`](Self::custom_components)
    /// floats; see [`set_custom`](Self::set_custom). Instances past a column's
    /// end read zero. They are vertex attributes, not storage, so they work on
    /// WebGL2 too. Built-in materials ignore them.
    pub custom: [Vec<f32>; 2],
    /// Floats an instance in each of [`custom`](Self::custom), `1..=4`.
    pub custom_components: [u32; 2],
    /// three.js's `instanceMatrix.needsUpdate`, opt-in. `None` (the default)
    /// re-reads `transforms`, `colors` and the custom columns every frame.
    /// `Some(v)` re-reads them only when `v` changes (or the instance count
    /// does): bump it after editing them. For a hundred thousand instances
    /// that are mostly still, that skips building and comparing megabytes of
    /// instance data every frame.
    pub instance_version: Option<u64>,
}

impl InstancedMesh {
    pub fn new(geometry: BufferGeometry, material: Material, count: usize) -> Self {
        Self::from_arc(Arc::new(geometry), Arc::new(material), count)
    }

    pub fn from_arc(geometry: Arc<BufferGeometry>, material: Arc<Material>, count: usize) -> Self {
        let (influences, dictionary) = super::mesh::morph_target_defaults(&geometry);
        Self {
            geometry,
            material,
            transforms: vec![Matrix4::identity(); count],
            colors: Vec::new(),
            materials: Vec::new(),
            morph_influences: influences.unwrap_or_default(),
            morph_target_dictionary: dictionary,
            morph_texture: Vec::new(),
            custom: [Vec::new(), Vec::new()],
            custom_components: [4, 4],
            instance_version: None,
        }
    }

    /// Replace custom column `slot` (0 or 1): `components` floats an instance,
    /// row-major.
    ///
    /// ```
    /// use threers::{BoxGeometry, InstancedMesh, Material, ShaderMaterial};
    /// let mut im = InstancedMesh::new(
    ///     BoxGeometry::new(1.0, 1.0, 1.0),
    ///     Material::Shader(ShaderMaterial::new("@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> { return in.vertex_color; }")),
    ///     3,
    /// );
    /// // A height per instance, which a custom vertex stage reads as `in.icustom0.x`.
    /// im.set_custom(0, 1, &[1.0, 2.5, 4.0]);
    /// assert_eq!(im.custom_row(0, 1), [2.5, 0.0, 0.0, 0.0]);
    /// ```
    pub fn set_custom(&mut self, slot: usize, components: u32, values: &[f32]) {
        assert!(slot < 2, "custom slot {slot} is not 0 or 1");
        assert!((1..=4).contains(&components), "{components} components is not 1..=4");
        assert!(
            values.len().is_multiple_of(components as usize),
            "{} floats is not a whole number of {components}-float rows",
            values.len()
        );
        self.custom[slot] = values.to_vec();
        self.custom_components[slot] = components;
    }

    /// Instance `index`'s value in custom column `slot`, zero-padded to four
    /// floats — what the shader reads.
    pub fn custom_row(&self, slot: usize, index: usize) -> [f32; 4] {
        let n = self.custom_components[slot].clamp(1, 4) as usize;
        let mut out = [0.0; 4];
        if let Some(row) = self.custom[slot].get(index * n..index * n + n) {
            out[..n].copy_from_slice(row);
        }
        out
    }

    /// Whether either custom column has data.
    pub fn has_custom(&self) -> bool {
        self.custom.iter().any(|c| !c.is_empty())
    }

    /// three.js `new InstancedMesh(geometry, [a, b, …], count)`;
    /// [`material`](Self::material) is the first entry.
    pub fn with_materials(
        geometry: Arc<BufferGeometry>,
        materials: Vec<Arc<Material>>,
        count: usize,
    ) -> Self {
        let first = materials
            .first()
            .cloned()
            .unwrap_or_else(|| Arc::new(Material::Basic(Default::default())));
        let mut im = Self::from_arc(geometry, first, count);
        im.materials = materials;
        im
    }

    /// Every draw this mesh makes — per group with a material array — each
    /// drawn once per instance. See [`mesh_draws`](super::mesh_draws).
    pub fn draws(&self) -> Vec<super::MeshDraw<'_>> {
        super::mesh_draws(&self.geometry, &self.material, &self.materials)
    }

    /// Number of floats in one [`morph_texture`](Self::morph_texture) row.
    pub fn morph_row_len(&self) -> usize {
        self.morph_influences.len() + 1
    }

    /// three.js `InstancedMesh.setMorphAt(index, mesh)`: instance `index`
    /// blends its morph targets by `influences` (a mesh's
    /// `morphTargetInfluences`). The first call allocates a row for every
    /// instance, zeroed; the base influence is computed as three.js does
    /// (`1` for relative targets, else `1 − Σ influences`).
    pub fn set_morph_at(&mut self, index: usize, influences: &[f32]) {
        if index >= self.transforms.len() {
            return;
        }
        let len = self.morph_row_len();
        if self.morph_texture.len() != len * self.transforms.len() {
            self.morph_texture = vec![0.0; len * self.transforms.len()];
        }
        let row = &mut self.morph_texture[index * len..(index + 1) * len];
        row[0] = super::skinning::Morph::base_influence(&self.geometry, influences);
        for (k, w) in row[1..].iter_mut().enumerate() {
            *w = influences.get(k).copied().unwrap_or(0.0);
        }
    }

    /// three.js `InstancedMesh.getMorphAt`: instance `index`'s influences, or
    /// `None` without a morph texture.
    pub fn get_morph_at(&self, index: usize) -> Option<Vec<f32>> {
        let len = self.morph_row_len();
        self.morph_texture
            .get(index * len..(index + 1) * len)
            .map(|row| row[1..].to_vec())
    }

    /// The influences instance `index` is drawn with: its morph-texture row,
    /// or the shared [`morph_influences`](Self::morph_influences).
    pub fn morph_influences_at(&self, index: usize) -> Vec<f32> {
        self.get_morph_at(index)
            .unwrap_or_else(|| self.morph_influences.clone())
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
