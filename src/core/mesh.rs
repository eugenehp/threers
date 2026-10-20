use super::{BufferGeometry, GeometryGroup};
use crate::materials::Material;
use std::collections::HashMap;
use std::sync::Arc;

/// A drawable: geometry + material. The renderer caches per-mesh GPU buffers
/// using identity of these Arcs.
///
/// # One material or an array
///
/// three.js's `Mesh.material` is either a material or an array of them. Here
/// it is two fields: [`material`](Self::material), always present, and
/// [`materials`](Self::materials), empty for an ordinary mesh. A non-empty
/// `materials` *is* three.js's array: every renderer then draws each of the
/// geometry's [`groups`](BufferGeometry::groups) with
/// `materials[group.material_index]` (a group whose index has no material
/// draws nothing, and a geometry with no groups draws nothing at all — both
/// as three.js), and the raycaster reports the group's
/// [`material_index`](super::Face::material_index). `material` is then only
/// the fallback for the consumers that take one material per object (an
/// exporter, a bounding helper); [`Mesh::with_materials`] sets it to the
/// first entry.
///
/// Two fields rather than an enum because some seventy sites destructure a
/// mesh and read `mesh.material`; every one of them keeps compiling and stays
/// right for every single-material mesh, and the multi-material case is
/// handled where it matters — through [`draws`](Self::draws) — instead of at
/// each of them.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub geometry: Arc<BufferGeometry>,
    pub material: Arc<Material>,
    /// three.js's material *array*. Empty: the mesh draws its whole draw range
    /// with [`material`](Self::material), ignoring groups. See the type docs.
    pub materials: Vec<Arc<Material>>,
    /// Per-morph-target blend weights (glTF / three.js `morphTargetInfluences`),
    /// applied to the geometry's
    /// [`morph_attributes`](BufferGeometry::morph_attributes).
    pub morph_influences: Vec<f32>,
    /// three.js `morphTargetDictionary`: target name → index into
    /// [`morph_influences`](Self::morph_influences). Built by
    /// [`update_morph_targets`](Self::update_morph_targets).
    pub morph_target_dictionary: HashMap<String, usize>,
}

/// One draw of a mesh: a span of its index (vertices, unindexed) and the
/// material it is drawn with — three.js's render-list item, `{ object,
/// geometry, material, group }`.
#[derive(Debug, Clone, Copy)]
pub struct MeshDraw<'a> {
    /// First index entry (vertex, unindexed) drawn.
    pub start: u32,
    /// One past the last.
    pub end: u32,
    pub material: &'a Arc<Material>,
    /// The group's `material_index` for a multi-material mesh, `0` otherwise
    /// (three.js's `face.materialIndex` default).
    pub material_index: u32,
}

impl MeshDraw<'_> {
    /// `end - start`.
    pub fn count(&self) -> u32 {
        self.end - self.start
    }
}

impl Mesh {
    pub fn new(geometry: BufferGeometry, material: Material) -> Self {
        Self::from_arc(Arc::new(geometry), Arc::new(material))
    }

    pub fn from_arc(geometry: Arc<BufferGeometry>, material: Arc<Material>) -> Self {
        let mut mesh = Self {
            geometry,
            material,
            materials: Vec::new(),
            morph_influences: Vec::new(),
            morph_target_dictionary: HashMap::new(),
        };
        mesh.update_morph_targets();
        mesh
    }

    /// A multi-material mesh: three.js's `new Mesh(geometry, [a, b, …])`.
    /// [`material`](Self::material) is set to the first entry (a default
    /// material when the list is empty).
    pub fn with_materials(geometry: Arc<BufferGeometry>, materials: Vec<Arc<Material>>) -> Self {
        let first = materials
            .first()
            .cloned()
            .unwrap_or_else(|| Arc::new(Material::Basic(Default::default())));
        let mut mesh = Self::from_arc(geometry, first);
        mesh.materials = materials;
        mesh
    }

    /// Whether this mesh draws per group from a material array.
    pub fn is_multi_material(&self) -> bool {
        !self.materials.is_empty()
    }

    /// three.js `updateMorphTargets()`: size
    /// [`morph_influences`](Self::morph_influences) to the geometry's morph
    /// targets (all zero) and rebuild the dictionary from their names. A
    /// geometry without morph attributes leaves both as they are, as in
    /// three.js.
    pub fn update_morph_targets(&mut self) {
        let (influences, dictionary) = morph_target_defaults(&self.geometry);
        if let Some(influences) = influences {
            self.morph_influences = influences;
            self.morph_target_dictionary = dictionary;
        }
    }

    /// Every draw this mesh makes, in order — what the renderers draw and the
    /// raycaster tests. See [`mesh_draws`].
    pub fn draws(&self) -> Vec<MeshDraw<'_>> {
        mesh_draws(&self.geometry, &self.material, &self.materials)
    }

    /// The geometry with its morph targets blended by
    /// [`morph_influences`](Self::morph_influences) — what the GPU renderers
    /// draw — for the CPU consumers. The geometry itself when nothing blends.
    /// See `skinning::posed_geometry`.
    pub fn posed_geometry(&self) -> Arc<BufferGeometry> {
        super::skinning::posed_geometry(&self.geometry, &self.morph_influences, None)
    }
}

/// three.js `Mesh.updateMorphTargets` for one geometry: zeroed influences and
/// the name dictionary, or `None` without morph attributes.
pub(crate) fn morph_target_defaults(
    geometry: &BufferGeometry,
) -> (Option<Vec<f32>>, HashMap<String, usize>) {
    // three.js reads `morphAttributes[Object.keys(morphAttributes)[0]]` —
    // the first attribute inserted, in practice `position`. A map has no
    // insertion order, so take position, then normal, then colour.
    let first = ["position", "normal", "color"]
        .iter()
        .find_map(|k| geometry.morph_attributes.get(*k))
        .or_else(|| geometry.morph_attributes.values().next());
    let Some(targets) = first else {
        return (None, HashMap::new());
    };
    let mut dictionary = HashMap::new();
    for m in 0..targets.len() {
        let name = geometry
            .morph_target_names
            .get(m)
            .filter(|n| !n.is_empty())
            .cloned()
            .unwrap_or_else(|| m.to_string());
        dictionary.insert(name, m);
    }
    (Some(vec![0.0; targets.len()]), dictionary)
}

/// The draws of a mesh, as three.js's `WebGLRenderer` builds its render list
/// (`projectObject`) and `renderBufferDirect` clips each one:
///
/// - one material (`materials` empty): one draw over the
///   [draw range](BufferGeometry::draw_range), groups ignored;
/// - a material array: one draw per group with
///   `materials[group.material_index]`, each the group cut to the draw range.
///   A group with no material at its index is skipped, and no groups means no
///   draws.
///
/// Empty spans are left out. The raycaster walks exactly these spans.
pub fn mesh_draws<'a>(
    geometry: &BufferGeometry,
    material: &'a Arc<Material>,
    materials: &'a [Arc<Material>],
) -> Vec<MeshDraw<'a>> {
    if materials.is_empty() {
        return geometry
            .draw_span(None)
            .map(|(start, end)| MeshDraw {
                start,
                end,
                material,
                material_index: 0,
            })
            .into_iter()
            .collect();
    }
    geometry
        .groups
        .iter()
        .filter_map(|g: &GeometryGroup| {
            let material = materials.get(g.material_index as usize)?;
            let (start, end) = geometry.draw_span(Some(g))?;
            Some(MeshDraw {
                start,
                end,
                material,
                material_index: g.material_index,
            })
        })
        .collect()
}
