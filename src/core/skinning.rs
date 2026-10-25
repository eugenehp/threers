//! Skeletal skinning, and the one definition of a posed vertex that every
//! consumer shares.
//!
//! The convention is documented on [`SkinnedMesh`].

use super::{BufferAttribute, BufferGeometry, ObjectId};
use crate::materials::Material;
use crate::math::{Matrix4, Vector3};
use std::sync::Arc;

/// How many bone matrices the renderers' skinning reads; joint indices are
/// clamped to `MAX_BONES - 1`. The wgpu shader's bone array is a fixed-size
/// uniform (WebGL2 has no storage buffers), and every other consumer matches it.
pub(crate) const MAX_BONES: usize = 128;

/// A bone identified by its node id + the inverse-bind matrix that maps mesh
/// vertices into bone-local space. Mirrors three.js's `Bone` plus
/// `Skeleton.boneInverses[i]`.
#[derive(Debug, Clone, Copy)]
pub struct Bone {
    pub node: ObjectId,
    pub inverse_bind: Matrix4,
}

/// Collection of bones plus their world matrices (updated each frame).
/// The renderer reads `bone_matrices` via a storage buffer.
#[derive(Debug, Clone, Default)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
    /// Final per-bone matrix = node.matrix_world * inverse_bind. Caller
    /// (typically the user's animation loop or `Skeleton::update`) is
    /// responsible for writing this.
    pub bone_matrices: Vec<Matrix4>,
}

impl Skeleton {
    pub fn new(bones: Vec<Bone>) -> Self {
        let count = bones.len();
        Self {
            bones,
            bone_matrices: vec![Matrix4::identity(); count],
        }
    }

    /// Recompute every `bone_matrices[i]` from the corresponding node's
    /// current `matrix_world` and the bone's `inverse_bind`. Mirrors
    /// three.js's `Skeleton.update()`.
    pub fn update(&mut self, arena: &super::ObjectArena) {
        if self.bone_matrices.len() != self.bones.len() {
            self.bone_matrices
                .resize(self.bones.len(), Matrix4::identity());
        }
        for (i, bone) in self.bones.iter().enumerate() {
            let world = arena
                .get(bone.node)
                .map(|o| o.matrix_world)
                .unwrap_or_else(Matrix4::identity);
            self.bone_matrices[i] = world.multiply(&bone.inverse_bind);
        }
    }
}

/// Refresh every skinned mesh's palette from its bones' current world
/// matrices — three.js's `skeleton.update()`, which its renderer runs for
/// every skinned mesh it draws. Run after the world matrices; see
/// [`SkinnedMesh::skinning_palette`]. Skeletons whose bones are not nodes of
/// `arena` keep the matrices they were given.
pub fn update_skeletons(arena: &mut super::ObjectArena) {
    let mut palettes = Vec::new();
    for (id, node) in arena.nodes.iter() {
        if let super::ObjectKind::SkinnedMesh(sm) = &node.kind {
            if sm.skeleton.bones.is_empty() {
                continue;
            }
            if let Some(p) = sm.skinning_palette(arena, &node.matrix_world) {
                palettes.push((id, p));
            }
        }
    }
    for (id, palette) in palettes {
        if let Some(super::ObjectKind::SkinnedMesh(sm)) =
            arena.nodes.get_mut(id).map(|n| &mut n.kind)
        {
            if sm.skeleton.bone_matrices != palette {
                Arc::make_mut(&mut sm.skeleton).bone_matrices = palette;
            }
        }
    }
}

/// Mesh that follows a skeleton. The geometry must carry `joint` (vec4 of bone
/// indices) and `weight` (vec4 of bone weights) attributes.
///
/// # Where the vertices go
///
/// three.js r165 skins in the vertex shader (`skinning_vertex.glsl`):
///
/// ```text
/// skinned     = Σ wᵢ · boneMatrixᵢ · bindMatrix · vec4(position, 1)
/// transformed = (bindMatrixInverse · skinned).xyz
/// world       = modelMatrix · vec4(transformed, 1)
/// ```
///
/// with `boneMatrixᵢ = bone.matrixWorld · boneInverse` — this crate's
/// [`Skeleton::bone_matrices`]. A [`SkinnedMesh`] here has no `bindMatrix`:
/// it is three.js's default `AttachedBindMode` with an identity
/// `bindMatrix` — what `GLTFLoader` binds (`mesh.bind(skeleton,
/// _identityMatrix)`) — so
/// `bindMatrixInverse` is the inverse of the mesh's current `matrix_world`.
/// With weights summing to one that makes the world position
/// `Σ wᵢ · boneMatrixᵢ · p` — the bones place the mesh, its own transform
/// cancels out. With weights that do not sum to one, the `w` of `skinned` is
/// their sum and the shader's arithmetic leaves
/// `Σ wᵢ · boneMatrixᵢ · p + (1 − Σ wᵢ) · translation(matrix_world)`. That is
/// what three.js *draws*, so it is what every renderer here draws and what the
/// raycaster hits. (three.js's own CPU path, `applyBoneTransform`, drops the
/// `w` and so disagrees with its own render for such weights; its loaders
/// normalise weights on load with `normalizeSkinWeights` so the case does not
/// arise for loaded assets, and this crate's loaders do the same.)
///
/// Normals follow `skinnormal_vertex.glsl`:
/// `objectNormal = (bindMatrixInverse · Σ wᵢ · boneMatrixᵢ · bindMatrix) · n`,
/// then the model's normal matrix.
///
/// Joint indices read as the shaders read them: clamped to 127 (the
/// renderers read at most 128 bones), and a joint past the end of the
/// skeleton reads the identity (the bone array is padded with identities).
///
/// Morph targets apply first, as three.js's `morphtarget_vertex.glsl` runs
/// before `skinning_vertex.glsl`: see `posed_geometry` and [`Morph`].
#[derive(Debug, Clone)]
pub struct SkinnedMesh {
    pub geometry: Arc<BufferGeometry>,
    pub material: Arc<Material>,
    pub skeleton: Arc<Skeleton>,
    /// three.js's material array, as on [`Mesh`](super::Mesh::materials):
    /// empty for one material, otherwise drawn per geometry group.
    pub materials: Vec<Arc<Material>>,
    /// three.js `morphTargetInfluences`, as on [`Mesh`](super::Mesh).
    pub morph_influences: Vec<f32>,
    /// three.js `morphTargetDictionary`.
    pub morph_target_dictionary: std::collections::HashMap<String, usize>,
    /// three.js `bindMatrix`: where the mesh was when bound to its skeleton.
    /// Identity by default — what `GLTFLoader` binds, and the convention the
    /// type docs describe. Folded into the skinning palette by
    /// [`skinning_palette`](Self::skinning_palette), so the renderers and the
    /// raycaster need nothing else.
    pub bind_matrix: Matrix4,
    /// three.js `bindMode`: whether `bindMatrixInverse` follows the mesh
    /// (`Attached`, three.js's default) or stays the inverse of
    /// [`bind_matrix`](Self::bind_matrix) (`Detached`).
    pub bind_mode: BindMode,
}

/// three.js `SkinnedMesh.bindMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BindMode {
    /// `AttachedBindMode`: `bindMatrixInverse` is the inverse of the mesh's
    /// current `matrixWorld`, so the bones place the mesh.
    #[default]
    Attached,
    /// `DetachedBindMode`: `bindMatrixInverse` is the inverse of
    /// `bindMatrix`, so the mesh's own transform applies on top.
    Detached,
}

impl SkinnedMesh {
    pub fn new(geometry: BufferGeometry, material: Material, skeleton: Skeleton) -> Self {
        Self::from_arc(Arc::new(geometry), Arc::new(material), Arc::new(skeleton))
    }

    /// Build from shared parts. Sizes the morph influences to the geometry's
    /// targets, as three.js's constructor does.
    pub fn from_arc(
        geometry: Arc<BufferGeometry>,
        material: Arc<Material>,
        skeleton: Arc<Skeleton>,
    ) -> Self {
        let (influences, dictionary) = super::mesh::morph_target_defaults(&geometry);
        Self {
            geometry,
            material,
            skeleton,
            materials: Vec::new(),
            morph_influences: influences.unwrap_or_default(),
            morph_target_dictionary: dictionary,
            bind_matrix: Matrix4::identity(),
            bind_mode: BindMode::Attached,
        }
    }

    /// The skinning palette — what [`Skeleton::bone_matrices`] holds for the
    /// renderers — from the bones' world matrices in `arena`, for a mesh
    /// whose world matrix is `matrix_world`: three.js's `boneMatrices`
    /// (`bone.matrixWorld · boneInverse`) with the bind matrices folded in,
    ///
    /// ```text
    /// attached:  paletteᵢ = boneMatrixᵢ · bindMatrix
    /// detached:  paletteᵢ = matrixWorld · bindMatrix⁻¹ · boneMatrixᵢ · bindMatrix
    /// ```
    ///
    /// so `Σ wᵢ · paletteᵢ · p` is three.js's world-space vertex
    /// (`modelMatrix · bindMatrixInverse · Σ wᵢ · boneMatrixᵢ · bindMatrix · p`,
    /// with `bindMatrixInverse` the inverse of `matrixWorld` when attached).
    /// `None` when a bone's node is not in `arena` — a skeleton whose
    /// matrices are set by hand (the bones are not scene nodes) is left alone.
    pub fn skinning_palette(
        &self,
        arena: &super::ObjectArena,
        matrix_world: &Matrix4,
    ) -> Option<Vec<Matrix4>> {
        let identity = Matrix4::identity();
        let bind = self.bind_matrix;
        let pre = match self.bind_mode {
            BindMode::Attached => identity,
            BindMode::Detached => matrix_world.multiply(&bind.invert()),
        };
        let fold_bind = bind != identity;
        let fold_pre = pre != identity;
        self.skeleton
            .bones
            .iter()
            .map(|b| {
                let world = arena.get(b.node)?.matrix_world;
                let mut m = world.multiply(&b.inverse_bind);
                if fold_bind {
                    m = m.multiply(&bind);
                }
                if fold_pre {
                    m = pre.multiply(&m);
                }
                Some(m)
            })
            .collect()
    }

    /// three.js `updateMorphTargets()`. See
    /// [`Mesh::update_morph_targets`](super::Mesh::update_morph_targets).
    pub fn update_morph_targets(&mut self) {
        let (influences, dictionary) = super::mesh::morph_target_defaults(&self.geometry);
        if let Some(influences) = influences {
            self.morph_influences = influences;
            self.morph_target_dictionary = dictionary;
        }
    }

    /// Every draw this mesh makes. See [`super::mesh_draws`].
    pub fn draws(&self) -> Vec<super::MeshDraw<'_>> {
        super::mesh_draws(&self.geometry, &self.material, &self.materials)
    }

    /// Convenience: build a skinned mesh from a non-skinned `BufferGeometry`
    /// by adding zeroed joint/weight attributes (every vertex bound to bone 0
    /// with weight 1).
    pub fn from_unweighted(
        mut geometry: BufferGeometry,
        material: Material,
        skeleton: Skeleton,
    ) -> Self {
        let count = geometry
            .get_attribute("position")
            .map(|a| a.count())
            .unwrap_or(0);
        if geometry.get_attribute("joint").is_none() {
            geometry.set_attribute("joint", BufferAttribute::new(vec![0.0; count * 4], 4));
        }
        if geometry.get_attribute("weight").is_none() {
            let mut weights = vec![0.0; count * 4];
            for chunk in weights.chunks_exact_mut(4) {
                chunk[0] = 1.0;
            }
            geometry.set_attribute("weight", BufferAttribute::new(weights, 4));
        }
        Self::new(geometry, material, skeleton)
    }

    /// The vertices in their current pose, xyz per vertex, in the mesh's
    /// **local** space — three.js's `transformed` after
    /// `skinning_vertex.glsl` — for a mesh whose world matrix is
    /// `matrix_world`. Place them with `matrix_world` to get what the
    /// renderers draw. `None` without `position`, `joint` and `weight`
    /// attributes (the renderers then draw the rest pose under
    /// `matrix_world`).
    pub fn posed_positions(&self, matrix_world: &Matrix4) -> Option<Vec<f32>> {
        Skinning::new(&self.geometry, &self.skeleton, matrix_world)?;
        posed_positions(
            &self.geometry,
            &self.morph_influences,
            Some((&self.skeleton, matrix_world)),
        )
    }

    /// The geometry in its current pose — morphed, then skinned — in the
    /// mesh's local space, for a CPU renderer to draw under `matrix_world`.
    /// See [`posed_geometry`].
    pub fn posed_geometry(&self, matrix_world: &Matrix4) -> Arc<BufferGeometry> {
        posed_geometry(
            &self.geometry,
            &self.morph_influences,
            Some((&self.skeleton, matrix_world)),
        )
    }
}

/// The most morph targets any renderer blends. The wgpu shader carries the
/// influences in a fixed uniform array (WebGL2 has no storage buffers), and
/// every other consumer — Metal, the CPU pose, the raycaster — reads the same
/// number, so all of them draw and hit the same surface. Targets past it are
/// ignored everywhere.
pub const MAX_MORPH_TARGETS: usize = 64;

/// three.js's morph-target blend for one geometry under one set of
/// influences: `morphtarget_vertex.glsl`, `morphnormal_vertex.glsl` and
/// `morphcolor_vertex.glsl`, which are also `Mesh.getVertexPosition` on the
/// CPU.
///
/// ```text
/// base        = morphTargetsRelative ? 1 : 1 - Σ influences
/// transformed = base · attribute + Σ wᵢ · targetᵢ
/// ```
///
/// With absolute targets that is `attribute + Σ wᵢ (targetᵢ − attribute)` —
/// three.js's `getVertexPosition` — and with relative ones
/// `attribute + Σ wᵢ targetᵢ`. The sum in `base` runs over every influence the
/// mesh carries, as `WebGLMorphtargets` sums `morphTargetInfluences`. Normals
/// blend the same way (when the geometry has normal targets), and colours
/// (when it has colour targets) — rgb for a 3-component colour, all four
/// channels for a 4-component one.
#[derive(Debug, Clone)]
pub struct Morph<'a> {
    positions: Option<&'a [BufferAttribute]>,
    normals: Option<&'a [BufferAttribute]>,
    colors: Option<&'a [BufferAttribute]>,
    /// `(target, influence)`, non-zero influences only, at most
    /// [`MAX_MORPH_TARGETS`] targets.
    weights: Vec<(usize, f32)>,
    base: f32,
}

impl<'a> Morph<'a> {
    /// `None` when there is nothing to blend: no morph attributes, or every
    /// influence zero (the blend is then the identity, either way round).
    pub fn new(geometry: &'a BufferGeometry, influences: &[f32]) -> Option<Self> {
        let get = |k: &str| {
            geometry
                .morph_attributes
                .get(k)
                .map(|v| v.as_slice())
                .filter(|v| !v.is_empty())
        };
        let (positions, normals, colors) = (get("position"), get("normal"), get("color"));
        if positions.is_none() && normals.is_none() && colors.is_none() {
            return None;
        }
        let targets = geometry.morph_target_count().min(MAX_MORPH_TARGETS);
        let weights: Vec<(usize, f32)> = influences
            .iter()
            .take(targets)
            .copied()
            .enumerate()
            .filter(|&(_, w)| w != 0.0)
            .collect();
        if weights.is_empty() {
            return None;
        }
        Some(Self {
            positions,
            normals,
            colors,
            weights,
            base: Self::base_influence(geometry, influences),
        })
    }

    /// three.js's `morphTargetBaseInfluence`: 1 for relative targets, else
    /// `1 − Σ influences`.
    pub fn base_influence(geometry: &BufferGeometry, influences: &[f32]) -> f32 {
        if geometry.morph_targets_relative {
            1.0
        } else {
            1.0 - influences.iter().sum::<f32>()
        }
    }

    /// Component `k` of vertex `v` in `attr`, `fill` past its item size.
    fn read(attr: &BufferAttribute, v: usize, k: usize, fill: f32) -> f32 {
        if k >= attr.item_size {
            return fill;
        }
        attr.array
            .get(v * attr.item_size + k)
            .copied()
            .unwrap_or(0.0)
    }

    fn blend3(&self, targets: Option<&[BufferAttribute]>, v: usize, x: Vector3) -> Vector3 {
        let Some(targets) = targets else {
            return x;
        };
        let mut out = x * self.base;
        for &(i, w) in &self.weights {
            if let Some(t) = targets.get(i) {
                out = out
                    + Vector3::new(
                        Self::read(t, v, 0, 0.0),
                        Self::read(t, v, 1, 0.0),
                        Self::read(t, v, 2, 0.0),
                    ) * w;
            }
        }
        out
    }

    /// Vertex `v`'s position `p`, morphed.
    pub fn position(&self, v: usize, p: Vector3) -> Vector3 {
        self.blend3(self.positions, v, p)
    }

    /// Vertex `v`'s normal `n`, morphed (not normalised). Unchanged without
    /// normal targets.
    pub fn normal(&self, v: usize, n: Vector3) -> Vector3 {
        self.blend3(self.normals, v, n)
    }

    /// Whether colours blend.
    pub fn has_colors(&self) -> bool {
        self.colors.is_some()
    }

    /// Vertex `v`'s colour, `item_size` components of it in `c`, morphed.
    /// Unchanged without colour targets.
    pub fn color(&self, v: usize, c: &mut [f32]) {
        let Some(targets) = self.colors else {
            return;
        };
        let n = c.len().min(4);
        for x in c.iter_mut().take(n) {
            *x *= self.base;
        }
        for &(i, w) in &self.weights {
            if let Some(t) = targets.get(i) {
                for (k, x) in c.iter_mut().take(n).enumerate() {
                    // A 3-component target pads alpha with 1, as three.js's
                    // morph texture does.
                    *x += Self::read(t, v, k, 1.0) * w;
                }
            }
        }
    }
}

/// A geometry's morph targets packed as three.js's `WebGLMorphtargets` packs
/// its morph texture, for the GPU renderers: target `t`, vertex `v`, slot `k`
/// (0 position, 1 normal, 2 colour) at texel `(t · vertex_count + v) · stride
/// + k`, four floats a texel. `stride` is 3 with colour targets, 2 with
/// normal targets, else 1 — colour keeps slot 2 even without normals, as in
/// three.js. A 3-component colour target pads alpha with 1; missing data is
/// zero. At most [`MAX_MORPH_TARGETS`] targets.
#[derive(Debug, Clone)]
pub struct MorphTexels {
    pub data: Vec<f32>,
    pub stride: u32,
    pub targets: u32,
    pub vertex_count: u32,
    /// Bit 0: normals blend. Bit 1: colours blend. Bit 2: the colour
    /// attribute has alpha (blend all four channels). Bit 3: positions
    /// blend (a geometry with only normal or colour targets leaves its
    /// positions alone, as [`Morph`] does).
    pub flags: u32,
}

impl MorphTexels {
    /// `None` without morph attributes or positions.
    pub fn build(geometry: &BufferGeometry) -> Option<Self> {
        let vertex_count = geometry.get_attribute("position")?.count();
        let get = |k: &str| {
            geometry
                .morph_attributes
                .get(k)
                .filter(|v| !v.is_empty())
                .map(|v| v.as_slice())
        };
        let (pos, nrm, col) = (get("position"), get("normal"), get("color"));
        if pos.is_none() && nrm.is_none() && col.is_none() {
            return None;
        }
        let targets = geometry.morph_target_count().min(MAX_MORPH_TARGETS);
        let stride = if col.is_some() {
            3
        } else if nrm.is_some() {
            2
        } else {
            1
        };
        let mut data = vec![0.0f32; targets * vertex_count * stride * 4];
        for t in 0..targets {
            for (slot, attrs, fill_w) in [(0, pos, 0.0), (1, nrm, 0.0), (2, col, 1.0)] {
                let Some(attr) = attrs.and_then(|a| a.get(t)) else {
                    continue;
                };
                let n = attr.item_size.max(1);
                for v in 0..vertex_count.min(attr.array.len() / n) {
                    let at = ((t * vertex_count + v) * stride + slot) * 4;
                    for k in 0..4 {
                        data[at + k] = if k < n {
                            attr.array[v * n + k]
                        } else if k == 3 {
                            fill_w
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        let mut flags = 0;
        if pos.is_some() {
            flags |= 8;
        }
        if nrm.is_some() {
            flags |= 1;
        }
        if col.is_some() {
            flags |= 2;
            if geometry
                .get_attribute("color")
                .is_some_and(|c| c.item_size == 4)
            {
                flags |= 4;
            }
        }
        Some(Self {
            data,
            stride: stride as u32,
            targets: targets as u32,
            vertex_count: vertex_count as u32,
            flags,
        })
    }
}

/// The per-draw morph uniforms the GPU renderers upload: `header` = (target
/// count, three.js's `morphTargetBaseInfluence`, texels per vertex, flags as
/// in [`MorphTexels::flags`]) and the influences, [`MAX_MORPH_TARGETS`] of
/// them, zero-padded. `None` when nothing blends (see [`Morph::new`]) — the
/// draw then uploads a zero count.
pub fn morph_uniforms(
    geometry: &BufferGeometry,
    influences: &[f32],
) -> Option<([f32; 4], [f32; MAX_MORPH_TARGETS])> {
    Morph::new(geometry, influences)?;
    let mut header = morph_header(geometry)?;
    header[1] = Morph::base_influence(geometry, influences);
    let targets = header[0] as usize;
    let mut weights = [0.0f32; MAX_MORPH_TARGETS];
    for (w, i) in weights.iter_mut().zip(influences.iter().take(targets)) {
        *w = *i;
    }
    Some((header, weights))
}

/// [`morph_uniforms`]' header without the influences: (target count, `0`
/// for the base influence, texels per vertex, flags), or `None` without
/// morph attributes. For draws whose influences live elsewhere — an
/// `InstancedMesh`'s per-instance morph texture.
pub fn morph_header(geometry: &BufferGeometry) -> Option<[f32; 4]> {
    let has = |k: &str| {
        geometry
            .morph_attributes
            .get(k)
            .is_some_and(|v| !v.is_empty())
    };
    let targets = geometry.morph_target_count().min(MAX_MORPH_TARGETS);
    let stride = if has("color") {
        3
    } else if has("normal") {
        2
    } else {
        1
    };
    let mut flags = 0u32;
    if has("position") {
        flags |= 8;
    }
    if has("normal") {
        flags |= 1;
    }
    if has("color") {
        flags |= 2;
        if geometry
            .get_attribute("color")
            .is_some_and(|c| c.item_size == 4)
        {
            flags |= 4;
        }
    }
    if targets == 0 || flags == 0 {
        return None;
    }
    Some([targets as f32, 0.0, stride as f32, flags as f32])
}

/// Positions as every renderer draws them, xyz per vertex in the mesh's local
/// space: morph targets blended, then skinned when `skin` gives a skeleton
/// and the world matrix (see [`SkinnedMesh`] for that convention). `None`
/// when neither applies — the rest positions are the drawn ones — or there
/// are no positions.
pub(crate) fn posed_positions(
    geometry: &BufferGeometry,
    influences: &[f32],
    skin: Option<(&Skeleton, &Matrix4)>,
) -> Option<Vec<f32>> {
    let rest = geometry.get_attribute("position")?;
    let morph = Morph::new(geometry, influences);
    let skin = skin.and_then(|(s, m)| Skinning::new(geometry, s, m));
    if morph.is_none() && skin.is_none() {
        return None;
    }
    let n = rest.item_size.max(1);
    let count = rest.array.len() / n;
    let mut out = Vec::with_capacity(count * 3);
    for v in 0..count {
        let r = &rest.array[v * n..];
        let mut p = Vector3::new(
            r[0],
            if n > 1 { r[1] } else { 0.0 },
            if n > 2 { r[2] } else { 0.0 },
        );
        if let Some(m) = &morph {
            p = m.position(v, p);
        }
        if let Some(s) = &skin {
            p = s.position(v, p);
        }
        out.extend_from_slice(&[p.x, p.y, p.z]);
    }
    Some(out)
}

/// The geometry as every renderer draws it, for the CPU consumers (the path
/// tracer, the SVG renderer): morph targets blended into `position`, `normal`
/// and `color` (three.js's morph chunks), then skinned when `skin` gives a
/// skeleton and world matrix — positions and (unnormalised) normals in the
/// mesh's local space, to be placed with that world matrix. Everything else —
/// index, groups, draw range, other attributes — is shared. The geometry
/// itself when neither applies, which is also what the GPU renderers draw
/// then. The copy carries no morph attributes, so it is never blended twice.
pub(crate) fn posed_geometry(
    geometry: &Arc<BufferGeometry>,
    influences: &[f32],
    skin: Option<(&Skeleton, &Matrix4)>,
) -> Arc<BufferGeometry> {
    let morph = Morph::new(geometry, influences);
    let skinning = skin.and_then(|(s, m)| Skinning::new(geometry, s, m));
    if morph.is_none() && skinning.is_none() {
        return geometry.clone();
    }
    let Some(positions) = posed_positions(geometry, influences, skin) else {
        return geometry.clone();
    };
    let mut g = (**geometry).clone();
    g.morph_attributes.clear();
    g.set_attribute("position", BufferAttribute::new(positions, 3));
    if let Some(n) = geometry
        .get_attribute("normal")
        .filter(|a| a.item_size >= 3)
    {
        let mut out = Vec::with_capacity(n.count() * 3);
        for v in 0..n.count() {
            let a = &n.array[v * n.item_size..];
            let mut q = Vector3::new(a[0], a[1], a[2]);
            if let Some(m) = &morph {
                q = m.normal(v, q);
            }
            if let Some(s) = &skinning {
                q = s.normal(v, q);
            }
            out.extend_from_slice(&[q.x, q.y, q.z]);
        }
        g.set_attribute("normal", BufferAttribute::new(out, 3));
    }
    if let (Some(m), Some(c)) = (
        morph.as_ref().filter(|m| m.has_colors()),
        geometry.get_attribute("color"),
    ) {
        let mut out = c.array.clone();
        let k = c.item_size.max(1);
        for (v, chunk) in out.chunks_exact_mut(k).enumerate() {
            m.color(v, chunk);
        }
        g.set_attribute("color", BufferAttribute::new(out, c.item_size));
    }
    // The copy is a pose, not new topology: keep the polygon table the
    // position write would have kept anyway, and let the version be new so
    // nothing caches it as the source geometry.
    Arc::new(g)
}

/// three.js's skinning of one geometry by one skeleton, under one world
/// matrix. See [`SkinnedMesh`].
pub(crate) struct Skinning<'a> {
    joints: &'a BufferAttribute,
    weights: &'a BufferAttribute,
    bones: &'a [Matrix4],
    /// `bindMatrixInverse`: the inverse of the mesh's world matrix.
    bind_inverse: Matrix4,
}

impl<'a> Skinning<'a> {
    /// `None` without `joint` and `weight` attributes.
    pub fn new(
        geometry: &'a BufferGeometry,
        skeleton: &'a Skeleton,
        matrix_world: &Matrix4,
    ) -> Option<Self> {
        Some(Self {
            joints: geometry.get_attribute("joint")?,
            weights: geometry.get_attribute("weight")?,
            bones: &skeleton.bone_matrices,
            bind_inverse: matrix_world.invert(),
        })
    }

    /// The four influences of vertex `v`, missing components zero.
    fn read4(attr: &BufferAttribute, v: usize) -> [f32; 4] {
        let n = attr.item_size;
        let mut out = [0.0; 4];
        if let Some(src) = attr.array.get(v * n..v * n + n.min(4)) {
            out[..src.len()].copy_from_slice(src);
        }
        out
    }

    /// The bone matrix the shaders read for a joint index.
    fn bone(&self, j: f32) -> Matrix4 {
        let j = (j.max(0.0) as usize).min(MAX_BONES - 1);
        self.bones.get(j).copied().unwrap_or_else(Matrix4::identity)
    }

    /// `Σ wᵢ · boneMatrixᵢ` for vertex `v`, as the shader sums it.
    pub fn blend(&self, v: usize) -> Matrix4 {
        let (j, w) = (Self::read4(self.joints, v), Self::read4(self.weights, v));
        let mut m = [0.0f32; 16];
        for k in 0..4 {
            if w[k] != 0.0 {
                let b = self.bone(j[k]);
                for (acc, e) in m.iter_mut().zip(b.elements) {
                    *acc += e * w[k];
                }
            }
        }
        Matrix4 { elements: m }
    }

    /// Vertex `v` at rest position `p`, posed: three.js's `transformed`, in
    /// the mesh's local space.
    pub fn position(&self, v: usize, p: Vector3) -> Vector3 {
        let e = self.blend(v).elements;
        // `skinned`, all four components: its w is the weight sum.
        let s = [
            e[0] * p.x + e[4] * p.y + e[8] * p.z + e[12],
            e[1] * p.x + e[5] * p.y + e[9] * p.z + e[13],
            e[2] * p.x + e[6] * p.y + e[10] * p.z + e[14],
            e[3] * p.x + e[7] * p.y + e[11] * p.z + e[15],
        ];
        // `(bindMatrixInverse * skinned).xyz` — no divide by w.
        let b = &self.bind_inverse.elements;
        Vector3::new(
            b[0] * s[0] + b[4] * s[1] + b[8] * s[2] + b[12] * s[3],
            b[1] * s[0] + b[5] * s[1] + b[9] * s[2] + b[13] * s[3],
            b[2] * s[0] + b[6] * s[1] + b[10] * s[2] + b[14] * s[3],
        )
    }

    /// Vertex `v`'s normal `n`, skinned as `skinnormal_vertex.glsl` does: in
    /// the mesh's local space, not normalised.
    pub fn normal(&self, v: usize, n: Vector3) -> Vector3 {
        let e = self.blend(v).elements;
        let s = Vector3::new(
            e[0] * n.x + e[4] * n.y + e[8] * n.z,
            e[1] * n.x + e[5] * n.y + e[9] * n.z,
            e[2] * n.x + e[6] * n.y + e[10] * n.z,
        );
        let b = &self.bind_inverse.elements;
        Vector3::new(
            b[0] * s.x + b[4] * s.y + b[8] * s.z,
            b[1] * s.x + b[5] * s.y + b[9] * s.z,
            b[2] * s.x + b[6] * s.y + b[10] * s.z,
        )
    }
}

/// One animatable per-vertex offset target (a "morph"). Mirrors three.js's
/// morph-target data. Stored as deltas relative to the base position.
#[derive(Debug, Clone, Default)]
pub struct MorphTarget {
    pub name: String,
    /// Per-vertex position deltas (length = vertex_count * 3).
    pub position_delta: Vec<f32>,
    /// Optional per-vertex normal deltas.
    pub normal_delta: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default)]
pub struct MorphAttributes {
    pub targets: Vec<MorphTarget>,
    pub influences: Vec<f32>,
}

impl MorphAttributes {
    /// Hang these targets off `geometry` as three.js morph attributes:
    /// relative (`morph_targets_relative`), `position` from each target's
    /// deltas and `normal` when every target carries normal deltas, names
    /// in [`morph_target_names`](BufferGeometry::morph_target_names). Bumps
    /// the version. Returns the influences, for the mesh.
    pub fn apply_to(&self, geometry: &mut BufferGeometry) -> Vec<f32> {
        let positions: Vec<BufferAttribute> = self
            .targets
            .iter()
            .map(|t| BufferAttribute::new(t.position_delta.clone(), 3))
            .collect();
        let normals: Option<Vec<BufferAttribute>> = self
            .targets
            .iter()
            .map(|t| t.normal_delta.clone().map(|n| BufferAttribute::new(n, 3)))
            .collect();
        geometry.morph_targets_relative = true;
        geometry.morph_target_names = self.targets.iter().map(|t| t.name.clone()).collect();
        if let Some(n) = normals.filter(|n| !n.is_empty()) {
            geometry.morph_attributes.insert("normal".into(), n);
        }
        geometry.set_morph_attribute("position", positions);
        let mut influences = self.influences.clone();
        influences.resize(self.targets.len(), 0.0);
        influences
    }
}

#[cfg(test)]
mod palette_tests {
    use super::*;
    use crate::core::{Object3D, ObjectArena, ObjectKind};
    use crate::materials::BasicMaterial;
    use crate::math::{Color, Quaternion};

    fn close(a: &Matrix4, b: &Matrix4) -> bool {
        a.elements
            .iter()
            .zip(b.elements)
            .all(|(x, y)| (x - y).abs() < 1e-5)
    }

    /// The palette is three.js's `boneMatrix · bindMatrix` (attached) or
    /// `matrixWorld · bindMatrix⁻¹ · boneMatrix · bindMatrix` (detached),
    /// from the bones' nodes; `update_skeletons` writes it for the renderers.
    #[test]
    fn palette_folds_the_bind_matrix() {
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let mut b = Object3D::group();
        b.position = Vector3::new(0.0, 1.0, 0.0);
        b.quaternion = Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), 0.5);
        let bone = arena.insert(b);
        arena.add_child(root, bone);
        let inverse_bind = Matrix4::translation(Vector3::new(0.0, -0.5, 0.0));
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute("position", BufferAttribute::new(vec![0.0; 3], 3));
        let mut sm = SkinnedMesh::new(
            geometry,
            Material::Basic(BasicMaterial::new(Color::WHITE)),
            Skeleton::new(vec![Bone {
                node: bone,
                inverse_bind,
            }]),
        );
        sm.bind_matrix = Matrix4::translation(Vector3::new(0.3, 0.0, 0.0));
        let mut mesh = Object3D::skinned_mesh(sm);
        mesh.position = Vector3::new(2.0, 0.0, 0.0);
        let mesh = arena.insert(mesh);
        arena.add_child(root, mesh);
        arena.update_world_matrices(root, Matrix4::identity());
        let bone_world = arena.get(bone).unwrap().matrix_world;
        let mesh_world = arena.get(mesh).unwrap().matrix_world;
        let bind = Matrix4::translation(Vector3::new(0.3, 0.0, 0.0));
        let bone_matrix = bone_world.multiply(&inverse_bind);

        update_skeletons(&mut arena);
        let ObjectKind::SkinnedMesh(sm) = &arena.get(mesh).unwrap().kind else {
            unreachable!()
        };
        assert!(
            close(&sm.skeleton.bone_matrices[0], &bone_matrix.multiply(&bind)),
            "attached"
        );

        if let ObjectKind::SkinnedMesh(sm) = &mut arena.get_mut(mesh).unwrap().kind {
            sm.bind_mode = BindMode::Detached;
        }
        update_skeletons(&mut arena);
        let ObjectKind::SkinnedMesh(sm) = &arena.get(mesh).unwrap().kind else {
            unreachable!()
        };
        let want = mesh_world
            .multiply(&bind.invert())
            .multiply(&bone_matrix)
            .multiply(&bind);
        assert!(close(&sm.skeleton.bone_matrices[0], &want), "detached");
    }

    /// Bones that are not nodes of the arena (matrices set by hand) are left
    /// alone.
    #[test]
    fn hand_set_palettes_are_kept() {
        let mut arena = ObjectArena::new();
        let mut skeleton = Skeleton::new(vec![Bone {
            node: crate::core::ObjectId::default(),
            inverse_bind: Matrix4::identity(),
        }]);
        let set = Matrix4::translation(Vector3::new(1.0, 2.0, 3.0));
        skeleton.bone_matrices = vec![set];
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute("position", BufferAttribute::new(vec![0.0; 3], 3));
        let sm = SkinnedMesh::new(
            geometry,
            Material::Basic(BasicMaterial::new(Color::WHITE)),
            skeleton,
        );
        let mesh = arena.insert(Object3D::skinned_mesh(sm));
        update_skeletons(&mut arena);
        let ObjectKind::SkinnedMesh(sm) = &arena.get(mesh).unwrap().kind else {
            unreachable!()
        };
        assert_eq!(sm.skeleton.bone_matrices, vec![set]);
    }
}
