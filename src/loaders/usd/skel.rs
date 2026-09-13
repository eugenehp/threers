//! `UsdSkel` — skeletons, skinning and the animation that drives them.
//!
//! A skinned character is the one thing in USD that is not described where it
//! is used. The mesh says which skeleton binds it and which joints touch each
//! vertex; the skeleton says where the joints rest and what the mesh was bound
//! at; and a third prim, the animation, says where the joints are at each
//! frame. None of the three is much use without the other two.
//!
//! # Joints are paths, not names
//!
//! `joints = ["Root", "Root/Hip", "Root/Hip/Knee"]` is a *hierarchy* written
//! flat. The order of that list is the order every other array in the schema is
//! indexed by — bind transforms, rest transforms, the animation's channels, and
//! the per-vertex indices — so it is the one thing that must not be reordered.
//! The slashes are what says `Knee` hangs off `Hip`.
//!
//! # What is skinned where
//!
//! USD stores a bind transform per joint in *world* space and expects the
//! renderer to skin by `world * inverse(bind)`. That is the same convention
//! this crate's [`Bone`] holds, so the inverse is taken once here rather than
//! every frame.

use crate::animation::{AnimationClip, KeyframeTrack, TrackTarget};
use crate::core::{
    BufferAttribute, BufferGeometry, Bone, MorphTarget, Object3D, ObjectArena, ObjectId, Skeleton,
};
use crate::math::{Matrix4, Quaternion, Vector3};

use super::parse::{UsdLayer, UsdPrim};
use super::value::UsdValue;

/// How many joints may touch one vertex.
///
/// Four is what the shader reads and what every real-time renderer settles on.
/// A mesh authored with more keeps the four heaviest, which is what every
/// engine does with an eight-influence rig.
const INFLUENCES: usize = 4;

/// A skeleton read off a stage, with the nodes its joints became.
pub struct UsdSkeleton {
    /// The joint paths, in the order everything else indexes by.
    pub joints: Vec<String>,
    pub skeleton: Skeleton,
    /// The scene node for each joint, in the same order.
    pub nodes: Vec<ObjectId>,
}

/// Read a `Skeleton` prim, building a node per joint under `parent`.
///
/// The joints come back as real scene nodes because that is what a bone points
/// at — moving one has to move what hangs off it, which is the scene graph's
/// job rather than the skeleton's.
pub fn skeleton(prim: &UsdPrim, arena: &mut ObjectArena, parent: ObjectId) -> Option<UsdSkeleton> {
    let joints: Vec<String> = prim
        .value("joints")?
        .flat_tokens()
        .into_iter()
        .map(str::to_string)
        .collect();
    if joints.is_empty() {
        return None;
    }

    let bind = matrices(prim.value("bindTransforms"));
    let rest = matrices(prim.value("restTransforms"));

    // One node per joint, parented by what its path says.
    let mut nodes: Vec<ObjectId> = Vec::with_capacity(joints.len());
    for (i, path) in joints.iter().enumerate() {
        let mut node = Object3D::group();
        node.name = path.rsplit('/').next().unwrap_or(path).to_string();
        // The rest transform is the joint's *local* pose, which is exactly
        // what a scene node holds.
        if let Some(local) = rest.get(i) {
            let (position, quaternion, scale) = local.decompose();
            node.position = position;
            node.quaternion = quaternion;
            node.scale = scale;
        }
        let id = arena.insert(node);
        nodes.push(id);

        // A joint's parent is the joint whose path is this one's prefix.
        let owner = match path.rfind('/') {
            Some(at) => joints
                .iter()
                .position(|j| j == &path[..at])
                .map(|i| nodes[i])
                .unwrap_or(parent),
            None => parent,
        };
        arena.add_child(owner, id);
    }

    // USD's bind transforms are world space, and skinning wants their inverse.
    let bones: Vec<Bone> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| Bone {
            node: *node,
            inverse_bind: bind
                .get(i)
                .map(|m| m.invert())
                .unwrap_or_else(Matrix4::identity),
        })
        .collect();

    Some(UsdSkeleton {
        joints,
        skeleton: Skeleton::new(bones),
        nodes,
    })
}

/// The `joint` and `weight` attributes a skinned mesh needs, from the primvars
/// USD stores them in.
///
/// `elementSize` says how many influences each vertex has, and it is not
/// optional to read: the same array with an element size of two and of four
/// describes two entirely different rigs.
pub fn skin_attributes(prim: &UsdPrim, vertices: usize, geometry: &mut BufferGeometry) -> bool {
    let Some(indices) = prim
        .property("primvars:skel:jointIndices")
        .or_else(|| prim.property("skel:jointIndices"))
    else {
        return false;
    };
    let Some(weights) = prim
        .property("primvars:skel:jointWeights")
        .or_else(|| prim.property("skel:jointWeights"))
    else {
        return false;
    };

    let per_vertex = indices
        .metadata
        .iter()
        .find(|(k, _)| k == "elementSize")
        .and_then(|(_, v)| v.as_f64())
        .map(|v| v as usize)
        .filter(|v| *v > 0)
        .unwrap_or(INFLUENCES);

    let indices = indices.value.flat_f32();
    let weights = weights.value.flat_f32();
    if indices.is_empty() || indices.len() != weights.len() {
        return false;
    }

    let mut joint = vec![0.0f32; vertices * INFLUENCES];
    let mut weight = vec![0.0f32; vertices * INFLUENCES];
    for v in 0..vertices {
        let from = v * per_vertex;
        if from + per_vertex > indices.len() {
            break;
        }
        // Keep the heaviest influences when there are more than the shader
        // reads, rather than the first — dropping a 0.9 to keep a 0.05 is how
        // a limb ends up attached to the wrong bone.
        let mut pairs: Vec<(f32, f32)> = (0..per_vertex)
            .map(|k| (weights[from + k], indices[from + k]))
            .collect();
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        pairs.truncate(INFLUENCES);

        // Renormalise, since dropping influences loses their share.
        let total: f32 = pairs.iter().map(|(w, _)| *w).sum();
        for (k, (w, index)) in pairs.iter().enumerate() {
            joint[v * INFLUENCES + k] = *index;
            weight[v * INFLUENCES + k] = if total > 0.0 { w / total } else { 0.0 };
        }
    }

    geometry.set_attribute("joint", BufferAttribute::new(joint, INFLUENCES));
    geometry.set_attribute("weight", BufferAttribute::new(weight, INFLUENCES));
    true
}

/// Put a mesh into the space its weights were authored in.
///
/// `geomBindTransform` is where the mesh sat when it was bound, which is not
/// always where it sits now. Skipping it is invisible on a rig authored at the
/// origin and bends the character everywhere else.
pub fn apply_bind_transform(prim: &UsdPrim, geometry: &mut BufferGeometry) {
    let Some(value) = prim
        .value("primvars:skel:geomBindTransform")
        .or_else(|| prim.value("skel:geomBindTransform"))
    else {
        return;
    };
    let Some(bind) = matrices(Some(value)).into_iter().next() else {
        return;
    };
    if bind == Matrix4::identity() {
        return;
    }
    let Some(position) = geometry.get_attribute("position") else {
        return;
    };
    let moved: Vec<f32> = position
        .array
        .chunks_exact(3)
        .flat_map(|p| {
            // Column-major, as this crate stores matrices — which is the same
            // memory layout USD's row-vector convention produces, so the
            // sixteen numbers went in without rearranging.
            let m = &bind.elements;
            [
                m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
                m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
                m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
            ]
        })
        .collect();
    geometry.set_attribute("position", BufferAttribute::new(moved, 3));
}

/// The blend shapes bound to a mesh, as morph targets.
///
/// A `BlendShape` stores *offsets* from the base mesh, and usually only for the
/// vertices it touches — a smile moves the mouth and leaves the skull alone, so
/// `pointIndices` names the handful of vertices it has anything to say about.
/// A renderer wants a delta for every vertex, so the sparse form is expanded
/// here rather than left for every consumer to expand again.
pub fn blend_shapes(prim: &UsdPrim, layer: &UsdLayer, vertices: usize) -> Vec<MorphTarget> {
    // The names, which are what the animation keys its weights by, and the
    // prims that hold the offsets. The two lists run in parallel.
    let names: Vec<String> = prim
        .value("skel:blendShapes")
        .map(|v| v.flat_tokens().into_iter().map(str::to_string).collect())
        .unwrap_or_default();
    let targets: Vec<String> = prim
        .value("skel:blendShapeTargets")
        .map(|v| v.flat_tokens().into_iter().map(str::to_string).collect())
        .unwrap_or_default();

    let mut out = Vec::new();
    for (i, path) in targets.iter().enumerate() {
        // A target may be named by path or sit as a child of the mesh.
        let Some(shape) = layer
            .prim_at(path)
            .or_else(|| prim.children.iter().find(|c| Some(c.name.as_str()) == path.rsplit('/').next()))
        else {
            continue;
        };
        let offsets = shape.value("offsets").map(|v| v.flat_f32()).unwrap_or_default();
        if offsets.is_empty() {
            continue;
        }
        let indices = shape
            .value("pointIndices")
            .map(|v| v.flat_u32())
            .unwrap_or_default();

        out.push(MorphTarget {
            name: names.get(i).cloned().unwrap_or_else(|| shape.name.clone()),
            position_delta: dense(&offsets, &indices, vertices),
            normal_delta: shape
                .value("normalOffsets")
                .map(|v| v.flat_f32())
                .filter(|n| !n.is_empty())
                .map(|n| dense(&n, &indices, vertices)),
        });
    }
    out
}

/// Sparse offsets as a delta for every vertex.
fn dense(offsets: &[f32], indices: &[u32], vertices: usize) -> Vec<f32> {
    let mut out = vec![0.0; vertices * 3];
    if indices.is_empty() {
        // No index list means the offsets are for every vertex in order.
        let n = offsets.len().min(out.len());
        out[..n].copy_from_slice(&offsets[..n]);
        return out;
    }
    for (slot, vertex) in indices.iter().enumerate() {
        let (from, to) = (slot * 3, *vertex as usize * 3);
        if from + 3 > offsets.len() || to + 3 > out.len() {
            continue;
        }
        out[to..to + 3].copy_from_slice(&offsets[from..from + 3]);
    }
    out
}

/// The blend-shape weight tracks a `SkelAnimation` drives.
///
/// The animation keys *all* weights at once, in the order of its own
/// `blendShapes` list — which need not be the mesh's order, or even the same
/// set — so each weight is matched to the mesh's target by name.
pub fn blend_shape_tracks(
    anim: &UsdPrim,
    order: &[String],
    object: ObjectId,
    rate: f64,
) -> Vec<KeyframeTrack> {
    let Some(names) = anim.value("blendShapes").map(|v| {
        v.flat_tokens()
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    }) else {
        return Vec::new();
    };
    let Some(samples) = anim.value("blendShapeWeights").and_then(|v| v.samples().map(<[_]>::to_vec))
    else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for (slot, name) in names.iter().enumerate() {
        let Some(index) = order.iter().position(|n| n == name) else {
            continue;
        };
        let mut times = Vec::with_capacity(samples.len());
        let mut values = Vec::with_capacity(samples.len());
        for (time, frame) in &samples {
            let weights = frame.flat_f32();
            let Some(weight) = weights.get(slot) else {
                continue;
            };
            times.push((time / rate) as f32);
            values.push(*weight);
        }
        if times.is_empty() {
            continue;
        }
        out.push(KeyframeTrack::scalar(
            object,
            TrackTarget::MorphWeight { index },
            times,
            values,
        ));
    }
    out
}

/// A `SkelAnimation` as tracks on the joint nodes it names.
///
/// The animation names its own joints, and they need not be the skeleton's in
/// the same order — or even all of them — so each channel is matched by name
/// rather than by position.
pub fn animation(
    prim: &UsdPrim,
    skeleton: &UsdSkeleton,
    rate: f64,
    name: &str,
) -> Option<AnimationClip> {
    let joints: Vec<String> = prim
        .value("joints")?
        .flat_tokens()
        .into_iter()
        .map(str::to_string)
        .collect();

    let mut tracks = Vec::new();
    let mut duration = 0.0f32;
    for (slot, joint) in joints.iter().enumerate() {
        let Some(node) = skeleton
            .joints
            .iter()
            .position(|j| j == joint)
            .map(|i| skeleton.nodes[i])
        else {
            continue;
        };

        if let Some((times, values)) = channel(prim, "translations", slot, 3) {
            duration = duration.max(times.last().copied().unwrap_or(0.0) as f32);
            tracks.push(KeyframeTrack::vector(
                node,
                TrackTarget::Position,
                times.iter().map(|t| (t / rate) as f32).collect(),
                values
                    .chunks_exact(3)
                    .map(|v| Vector3::new(v[0], v[1], v[2]))
                    .collect(),
            ));
        }
        if let Some((times, values)) = channel(prim, "rotations", slot, 4) {
            duration = duration.max(times.last().copied().unwrap_or(0.0) as f32);
            tracks.push(KeyframeTrack::quaternion(
                node,
                TrackTarget::Quaternion,
                times.iter().map(|t| (t / rate) as f32).collect(),
                values
                    // USD writes the real part first; this one takes it last.
                    .chunks_exact(4)
                    .map(|q| Quaternion::new(q[1], q[2], q[3], q[0]))
                    .collect(),
            ));
        }
        if let Some((times, values)) = channel(prim, "scales", slot, 3) {
            tracks.push(KeyframeTrack::vector(
                node,
                TrackTarget::Scale,
                times.iter().map(|t| (t / rate) as f32).collect(),
                values
                    .chunks_exact(3)
                    .map(|v| Vector3::new(v[0], v[1], v[2]))
                    .collect(),
            ));
        }
    }

    if tracks.is_empty() {
        return None;
    }
    Some(AnimationClip::new(
        name,
        (duration as f64 / rate) as f32,
        tracks,
    ))
}

/// One joint's channel out of an array-per-frame.
///
/// A `SkelAnimation` keys *all* joints at once — each sample is an array with
/// one entry per joint — so a single joint's curve is a column out of that,
/// not a row.
fn channel(
    prim: &UsdPrim,
    name: &str,
    slot: usize,
    lanes: usize,
) -> Option<(Vec<f64>, Vec<f32>)> {
    let value = prim.value(name)?;
    let samples = value.samples()?;
    if samples.is_empty() {
        return None;
    }

    let mut times = Vec::with_capacity(samples.len());
    let mut values = Vec::with_capacity(samples.len() * lanes);
    for (time, frame) in samples {
        let flat = frame.flat_f32();
        let from = slot * lanes;
        if from + lanes > flat.len() {
            return None;
        }
        times.push(*time);
        values.extend_from_slice(&flat[from..from + lanes]);
    }
    Some((times, values))
}

/// Whether a mesh binds to a skeleton at all.
///
/// A mesh is skinned because it *says* so, not because it happens to sit under
/// a `SkelRoot`: a rig often keeps static props in the same subtree, and they
/// must not pick up the skeleton by proximity.
pub fn binds_a_skeleton(prim: &UsdPrim) -> bool {
    prim.property("skel:skeleton").is_some()
}

/// The animation a skeleton is driven by.
pub fn animation_source<'a>(prim: &UsdPrim, layer: &'a UsdLayer) -> Option<&'a UsdPrim> {
    layer.prim_at(prim.value("skel:animationSource")?.as_str()?)
}

/// Matrices out of a value, four rows at a time.
fn matrices(value: Option<&UsdValue>) -> Vec<Matrix4> {
    let Some(value) = value else {
        return Vec::new();
    };
    let numbers = value.flat_f32();
    numbers
        .chunks_exact(16)
        .map(|m| {
            let mut out = Matrix4::identity();
            out.elements.copy_from_slice(m);
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{parse::parse, scene::to_scene, scene::UsdScene};
    use crate::animation::TrackValues;
    use crate::core::ObjectKind;

    fn stage() -> UsdScene {
        to_scene(&parse(include_str!("testdata/skel.usda")).expect("parses"))
    }

    fn find_id(scene: &UsdScene, name: &str) -> Option<ObjectId> {
        fn walk(scene: &UsdScene, id: ObjectId, name: &str) -> Option<ObjectId> {
            let node = scene.arena.get(id)?;
            if node.name == name {
                return Some(id);
            }
            node.children.iter().find_map(|c| walk(scene, *c, name))
        }
        scene.roots.iter().find_map(|r| walk(scene, *r, name))
    }

    fn find<'a>(scene: &'a UsdScene, name: &str) -> &'a Object3D {
        fn walk<'a>(scene: &'a UsdScene, id: ObjectId, name: &str) -> Option<&'a Object3D> {
            let node = scene.arena.get(id)?;
            if node.name == name {
                return Some(node);
            }
            node.children.iter().find_map(|c| walk(scene, *c, name))
        }
        scene
            .roots
            .iter()
            .find_map(|r| walk(scene, *r, name))
            .unwrap_or_else(|| panic!("no node called {name}"))
    }

    /// A mesh that names a skeleton and carries weights is skinned, not a
    /// static mesh with some unused primvars on it.
    #[test]
    fn a_bound_mesh_becomes_skinned() {
        let scene = stage();
        let ObjectKind::SkinnedMesh(mesh) = &find(&scene, "Body").kind else {
            panic!("expected a skinned mesh, got {:?}", find(&scene, "Body").kind);
        };
        assert_eq!(mesh.skeleton.bones.len(), 3, "three joints");

        // The shader reads four influences per vertex whatever the rig has.
        let joint = mesh.geometry.get_attribute("joint").expect("joint indices");
        let weight = mesh.geometry.get_attribute("weight").expect("weights");
        assert_eq!(joint.item_size, 4);
        assert_eq!(weight.item_size, 4);
        assert_eq!(joint.count(), 4, "one per vertex");
    }

    /// The rig has two influences per vertex; the two the shader does not get
    /// are zero-weighted rather than absent, and the weights still sum to one.
    #[test]
    fn influences_are_padded_and_normalised() {
        let scene = stage();
        let ObjectKind::SkinnedMesh(mesh) = &find(&scene, "Body").kind else {
            panic!("skinned");
        };
        let joint = mesh.geometry.get_attribute("joint").unwrap();
        let weight = mesh.geometry.get_attribute("weight").unwrap();

        // Vertex 0 is 0.8 of joint 0 and 0.2 of joint 1.
        assert_eq!(&joint.array[0..2], &[0.0, 1.0]);
        assert!((weight.array[0] - 0.8).abs() < 1e-5, "{}", weight.array[0]);
        assert!((weight.array[1] - 0.2).abs() < 1e-5);
        assert_eq!(&weight.array[2..4], &[0.0, 0.0], "unused influences");

        for v in 0..joint.count() {
            let total: f32 = weight.array[v * 4..v * 4 + 4].iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "vertex {v} sums to {total}");
        }
    }

    /// `joints = ["Root", "Root/Mid", "Root/Mid/Tip"]` is a hierarchy written
    /// flat: the slashes say what hangs off what.
    #[test]
    fn joint_paths_build_a_hierarchy() {
        let scene = stage();
        let root = find(&scene, "Root");
        assert_eq!(root.children.len(), 1, "Root has one child");
        let mid = scene.arena.get(root.children[0]).unwrap();
        assert_eq!(mid.name, "Mid");
        assert_eq!(mid.children.len(), 1);
        assert_eq!(scene.arena.get(mid.children[0]).unwrap().name, "Tip");

        // The rest transform is the joint's local pose.
        assert_eq!(mid.position, Vector3::new(0.0, 2.0, 0.0));
    }

    /// A bind transform is world space; skinning wants its inverse, so the
    /// inversion happens once here rather than every frame.
    #[test]
    fn bind_transforms_are_inverted_once() {
        let scene = stage();
        let ObjectKind::SkinnedMesh(mesh) = &find(&scene, "Body").kind else {
            panic!("skinned");
        };
        // The second joint binds at y = 2, so its inverse translates by -2.
        let inverse = mesh.skeleton.bones[1].inverse_bind.elements;
        assert!((inverse[13] + 2.0).abs() < 1e-5, "got {}", inverse[13]);
        // And the third at y = 4.
        assert!((mesh.skeleton.bones[2].inverse_bind.elements[13] + 4.0).abs() < 1e-5);
    }

    /// A `SkelAnimation` keys every joint at once — each sample is an array
    /// with one entry per joint — so one joint's curve is a column out of it.
    #[test]
    fn the_animation_drives_the_joints() {
        let scene = stage();
        let clip = scene
            .animations
            .iter()
            .find(|c| !c.tracks.is_empty())
            .expect("a clip");

        // Walk to `Mid` rather than scanning the arena, so the id is the one
        // the hierarchy actually placed.
        let root_id = find_id(&scene, "Root").expect("Root");
        let mid_id = scene.arena.get(root_id).unwrap().children[0];
        assert_eq!(scene.arena.get(mid_id).unwrap().name, "Mid");

        let rotation = clip
            .tracks
            .iter()
            .find(|t| t.object == mid_id && matches!(t.target, TrackTarget::Quaternion))
            .expect("Mid should have a rotation track");
        assert_eq!(rotation.times, vec![0.0, 1.0], "24 codes at 24 a second");

        let TrackValues::Quaternion(values) = &rotation.values else {
            panic!("quaternions");
        };
        // USD writes the real part first; this crate keeps it last.
        assert!((values[1].w - 0.9239).abs() < 1e-3, "w was {}", values[1].w);
        assert!((values[1].z - 0.3827).abs() < 1e-3, "z was {}", values[1].z);
    }

    /// And the same rig read from the crate form.
    #[test]
    fn the_binary_form_skins_the_same_way() {
        let binary = to_scene(
            &super::super::UsdLoader::parse_layer(include_bytes!("testdata/skel.usdc"))
                .expect("the crate parses"),
        );
        let ObjectKind::SkinnedMesh(mesh) = &find(&binary, "Body").kind else {
            panic!("expected a skinned mesh from the crate form");
        };
        assert_eq!(mesh.skeleton.bones.len(), 3);
        assert!(!binary.animations.is_empty(), "the animation survived");
    }
}

#[cfg(test)]
mod blend_tests {
    use super::super::{parse::parse, scene::to_scene};
    use crate::animation::{TrackTarget, TrackValues};

    /// A blend shape stores offsets from the base mesh, usually for only the
    /// vertices it touches. A renderer wants a delta per vertex, so the sparse
    /// form is expanded — and expanding it wrongly moves the wrong face.
    #[test]
    fn sparse_offsets_land_on_the_vertices_they_name() {
        let scene = to_scene(&parse(include_str!("testdata/blend.usda")).expect("parses"));
        let (_, morphs) = scene
            .morph_targets
            .iter()
            .find(|(name, _)| name == "Face")
            .expect("the face has blend shapes");
        assert_eq!(morphs.targets.len(), 2);

        // `Smile` names points 2 and 3 of a four-point quad.
        let smile = morphs.targets.iter().find(|t| t.name == "smile").expect("smile");
        assert_eq!(smile.position_delta.len(), 4 * 3, "one delta per vertex");
        assert_eq!(&smile.position_delta[0..6], &[0.0; 6], "points 0 and 1 unmoved");
        assert_eq!(smile.position_delta[7], 0.5, "point 2 moved up");
        assert_eq!(smile.position_delta[10], 0.5, "point 3 moved up");

        // `Frown` names no indices, so its offsets are for every vertex in
        // order.
        let frown = morphs.targets.iter().find(|t| t.name == "frown").expect("frown");
        assert_eq!(frown.position_delta.len(), 4 * 3);
        assert_eq!(frown.position_delta[1], -0.5);
        assert_eq!(frown.position_delta[10], -0.5);
    }

    /// The weights are animated, and the animation keys all of them at once —
    /// so one shape's curve is a column out of the samples.
    #[test]
    fn blend_shape_weights_become_tracks() {
        let scene = to_scene(&parse(include_str!("testdata/blend.usda")).unwrap());
        let tracks: Vec<&crate::animation::KeyframeTrack> = scene
            .animations
            .iter()
            .flat_map(|c| &c.tracks)
            .filter(|t| matches!(t.target, TrackTarget::MorphWeight { .. }))
            .collect();
        assert_eq!(tracks.len(), 2, "one track per shape");

        let smile = tracks
            .iter()
            .find(|t| matches!(t.target, TrackTarget::MorphWeight { index: 0 }))
            .expect("a track for the first shape");
        assert_eq!(smile.times, vec![0.0, 1.0], "24 codes at 24 a second");
        let TrackValues::Scalar(values) = &smile.values else {
            panic!("scalars");
        };
        // Smile goes 0 -> 1 while frown goes 1 -> 0.
        assert_eq!(values, &vec![0.0, 1.0]);

        let frown = tracks
            .iter()
            .find(|t| matches!(t.target, TrackTarget::MorphWeight { index: 1 }))
            .unwrap();
        let TrackValues::Scalar(values) = &frown.values else {
            panic!("scalars");
        };
        assert_eq!(values, &vec![1.0, 0.0]);
    }

    /// And the same rig out of the crate form.
    #[test]
    fn blend_shapes_survive_the_binary_form() {
        let binary = to_scene(
            &super::super::UsdLoader::parse_layer(include_bytes!("testdata/blend.usdc"))
                .expect("the crate parses"),
        );
        let (_, morphs) = binary
            .morph_targets
            .iter()
            .find(|(name, _)| name == "Face")
            .expect("blend shapes from the crate");
        assert_eq!(morphs.targets.len(), 2);
        assert_eq!(morphs.targets[0].position_delta[7], 0.5);
    }
}
