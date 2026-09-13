//! Turning a layer into a scene, and a scene back into a layer.
//!
//! USD's geometry model and [`BufferGeometry`]'s differ in two ways that matter,
//! and both are resolved here rather than being pushed onto the caller:
//!
//! - **Faces are arbitrary polygons.** `faceVertexCounts` says how many corners
//!   each face has; a `BufferGeometry` holds triangles. Faces are fanned.
//! - **Primvars can be per-face-corner.** A `faceVarying` normal or UV gives a
//!   different value to the same point depending on which face is using it,
//!   which an indexed buffer cannot express. When any primvar is face-varying
//!   the mesh is expanded to unindexed triangles, which is the only lossless
//!   reading of it.

use crate::animation::{AnimationClip, KeyframeTrack, TrackTarget};
use crate::core::{
    BufferAttribute, BufferGeometry, Mesh, Object3D, ObjectArena, ObjectId, ObjectKind,
};
use crate::materials::{Material, StandardMaterial};
use crate::math::{Color, Matrix4, Quaternion, Vector3};

use super::parse::{UsdLayer, UsdPrim, UsdProperty};
use super::schemas::{self, UsdCamera};
use super::skel::{self, UsdSkeleton};
use super::shade;
use super::subdiv;
use crate::core::Skeleton;
use std::rc::Rc;
use super::usdz::UsdzArchive;
use super::value::UsdValue;
use crate::textures::{Texture, TextureFormat, TextureWrap};

/// A USD layer as a scene graph.
pub struct UsdScene {
    pub roots: Vec<ObjectId>,
    pub arena: ObjectArena,
    /// `metersPerUnit` from the layer, for callers that care what a unit was.
    pub meters_per_unit: f32,
    /// `upAxis` — `Y` unless the layer says `Z`.
    pub up_axis: char,
    /// One clip holding every animated transform in the layer, or nothing if
    /// the layer is static. USD has no notion of separate named animations
    /// within a layer the way glTF does — the layer *is* the animation — so
    /// there is at most one.
    pub animations: Vec<AnimationClip>,
    /// The cameras on the stage. Nothing in `ObjectKind` is a camera, so they
    /// come back beside the graph, each naming the node that places it.
    pub cameras: Vec<UsdCamera>,
    /// The skeletons, in the order they were found. A skinned mesh holds its
    /// own copy; these are here for a caller that wants to pose one.
    pub skeletons: Vec<Skeleton>,
    /// The blend shapes each mesh carries, by prim name.
    ///
    /// `Mesh` holds the influences but has nowhere for the targets, so they
    /// come back beside the graph the way the cameras and textures do.
    pub morph_targets: Vec<(String, crate::core::MorphAttributes)>,
    /// The textures each mesh's material asked for, by prim name.
    ///
    /// Decoding an image needs a decoder and the files beside the layer,
    /// neither of which belongs in a scene-description reader — so what the
    /// material wants is reported and the caller loads it.
    pub textures: Vec<(String, Vec<shade::TextureRequest>)>,
}

/// What to do while turning a layer into a scene.
#[derive(Debug, Clone, Default)]
pub struct SceneOptions {
    /// How many times to refine meshes that are subdivision surfaces.
    ///
    /// Zero — the default — draws the control cage, which is what every
    /// real-time loader does and what this crate did before the option
    /// existed. It is *not* what the file says: `subdivisionScheme` defaults
    /// to `catmullClark`, so most meshes are surfaces whose points are only a
    /// cage. One level is usually enough to look right; each one multiplies
    /// the face count by four, which is why nothing is refined unasked.
    pub subdivision_level: usize,
    /// Draw `purpose = "guide"` geometry, which is normally for the rigger.
    pub show_guides: bool,
    /// Draw `purpose = "proxy"` stand-ins alongside the real geometry.
    pub show_proxies: bool,
}

/// Build a scene from a parsed layer.
///
/// An animated layer is built at the first frame of its time range, and the
/// animation comes back alongside as a clip.
pub fn to_scene(layer: &UsdLayer) -> UsdScene {
    let start = layer
        .time_range()
        .or_else(|| layer.sampled_time_range())
        .map(|(start, _)| start)
        .unwrap_or(0.0);
    to_scene_at(layer, start)
}

/// Build a scene showing one instant of the layer.
pub fn to_scene_at(layer: &UsdLayer, time: f64) -> UsdScene {
    to_scene_with(layer, time, &SceneOptions::default())
}

/// Build a scene at an instant, with the options spelled out.
pub fn to_scene_with(layer: &UsdLayer, time: f64, options: &SceneOptions) -> UsdScene {
    // Resolve once up front so nothing below has to know about time.
    let resolved = layer.at_time(time);
    let mut arena = ObjectArena::new();
    let mut roots = Vec::new();
    let mut placed = Vec::new();
    let mut cameras = Vec::new();
    let mut skeletons = Vec::new();
    let mut skeleton_clips = Vec::new();
    let mut texture_requests: Vec<(String, Vec<shade::TextureRequest>)> = Vec::new();
    let mut morph_targets: Vec<(String, crate::core::MorphAttributes)> = Vec::new();
    for (prim, source) in resolved.prims.iter().zip(&layer.prims) {
        let mut context = Build {
            layer: &resolved,
            source_layer: layer,
            placed: &mut placed,
            cameras: &mut cameras,
            skeletons: &mut skeletons,
            clips: &mut skeleton_clips,
            rate: layer.time_codes_per_second(),
            options,
            skeleton_animation: None,
            textures: &mut texture_requests,
            morphs: &mut morph_targets,
            skeleton: None,
        };
        if let Some(id) = build(prim, &mut arena, source, "", &mut context) {
            roots.push(id);
        }
    }
    let mut animations: Vec<AnimationClip> = clip(layer, &placed).into_iter().collect();
    // A skeleton is driven by a `SkelAnimation` of its own, which keys every
    // joint at once rather than one prim at a time.
    animations.extend(skeleton_clips);
    UsdScene {
        roots,
        arena,
        meters_per_unit: layer.meters_per_unit(),
        up_axis: layer.up_axis(),
        animations,
        cameras,
        skeletons,
        morph_targets,
        textures: texture_requests
            .into_iter()
            .filter(|(_, t)| !t.is_empty())
            .collect(),
    }
}

/// Turn every animated transform in the layer into one clip.
///
/// Each prim's whole `xformOpOrder` stack is evaluated at every time any of
/// its ops is keyed at, then decomposed. Going through the stack rather than
/// translating op by op means a rotation expressed as `rotateXYZ` and one
/// expressed as `orient` produce the same track, and a prim whose translate
/// and rotate are keyed on different frames still animates correctly.
fn clip(layer: &UsdLayer, placed: &[(ObjectId, UsdPrim)]) -> Option<AnimationClip> {
    let rate = layer.time_codes_per_second();
    let mut tracks = Vec::new();
    let mut span: Option<(f64, f64)> = None;

    for (id, prim) in placed {
        let mut times = keyed_times(prim);
        if times.is_empty() {
            continue;
        }
        times.sort_by(|a, b| a.total_cmp(b));
        times.dedup();

        let sampled: Vec<_> = times
            .iter()
            .map(|t| transform_of(&resolve_prim(prim, *t)))
            .collect();
        let seconds: Vec<f32> = times.iter().map(|t| (t / rate) as f32).collect();
        if let (Some(first), Some(last)) = (times.first(), times.last()) {
            span = Some(match span {
                None => (*first, *last),
                Some((lo, hi)) => (lo.min(*first), hi.max(*last)),
            });
        }

        // A channel that never changes is not worth a track; a static scale on
        // an animated prim would otherwise cost as much as the motion does.
        let positions: Vec<_> = sampled.iter().map(|t| t.0).collect();
        if varies(&positions, |a, b| a.distance_to(*b) > 1e-6) {
            tracks.push(KeyframeTrack::vector(
                *id,
                TrackTarget::Position,
                seconds.clone(),
                positions,
            ));
        }
        let rotations: Vec<_> = sampled.iter().map(|t| t.1).collect();
        if varies(&rotations, |a, b| {
            (a.x - b.x).abs() > 1e-6
                || (a.y - b.y).abs() > 1e-6
                || (a.z - b.z).abs() > 1e-6
                || (a.w - b.w).abs() > 1e-6
        }) {
            tracks.push(KeyframeTrack::quaternion(
                *id,
                TrackTarget::Quaternion,
                seconds.clone(),
                rotations,
            ));
        }
        let scales: Vec<_> = sampled.iter().map(|t| t.2).collect();
        if varies(&scales, |a, b| a.distance_to(*b) > 1e-6) {
            tracks.push(KeyframeTrack::vector(
                *id,
                TrackTarget::Scale,
                seconds,
                scales,
            ));
        }
    }

    if tracks.is_empty() {
        return None;
    }
    // The layer's declared range wins over what the samples happen to cover,
    // because a layer may hold a pose past its last key.
    let (start, end) = layer.time_range().or(span).unwrap_or((0.0, 0.0));
    let duration = ((end - start) / rate).max(0.0) as f32;
    Some(AnimationClip::new("usd", duration, tracks))
}

fn varies<T: Copy>(values: &[T], differs: impl Fn(&T, &T) -> bool) -> bool {
    values.windows(2).any(|w| differs(&w[0], &w[1]))
}

/// A refined cage as a geometry: positions and topology only.
///
/// The primvars are deliberately dropped. A normal or a UV authored per corner
/// of the control cage does not index the refined mesh, and carrying it across
/// would attach the wrong value to the wrong vertex — worse than recomputing
/// the normals, which is what happens instead.
fn refined_geometry(
    points: Vec<f32>,
    counts: Vec<u32>,
    indices: Vec<u32>,
    prim: &UsdPrim,
) -> Option<BufferGeometry> {
    let vertices = points.len() / 3;
    if indices.iter().any(|i| *i as usize >= vertices) {
        return None;
    }
    let flip = prim.value("orientation").and_then(|v| v.as_str()) == Some("leftHanded");
    let mut triangles: Vec<u32> = Vec::new();
    let mut at = 0usize;
    for count in &counts {
        let count = *count as usize;
        if count < 3 || at + count > indices.len() {
            at += count;
            continue;
        }
        for i in 1..count - 1 {
            let (a, b, c) = (indices[at], indices[at + i], indices[at + i + 1]);
            if flip {
                triangles.extend_from_slice(&[a, c, b]);
            } else {
                triangles.extend_from_slice(&[a, b, c]);
            }
        }
        at += count;
    }
    if triangles.is_empty() {
        return None;
    }

    let mut geometry = BufferGeometry::new();
    geometry.set_attribute("position", BufferAttribute::new(points, 3));
    geometry.set_index(triangles);
    crate::compute_vertex_normals(&mut geometry);
    Some(geometry)
}

/// A mesh split into one child per material subset.
///
/// The faces no subset claims keep the mesh's own binding, which is what a
/// `nonOverlapping` family means by leaving them out — a `partition` leaves
/// none, and then there is no remainder to build.
fn split_by_subset(
    prim: &UsdPrim,
    arena: &mut ObjectArena,
    source: &UsdPrim,
    path: &str,
    context: &mut Build,
) -> ObjectId {
    let layer = context.layer;
    let level = context.options.subdivision_level;
    let subsets = shade::material_subsets(prim);

    let mut group = Object3D::group();
    group.name = prim.name.clone();
    let id = arena.insert(group);
    place_transform(arena, id, prim);
    context.placed.push((id, source.clone()));

    let total = prim
        .value("faceVertexCounts")
        .map(|v| v.flat_u32().len())
        .unwrap_or(0);

    // Each subset, then whatever it left behind.
    let remainder = shade::unclaimed_faces(total, &subsets);
    let parts = subsets
        .iter()
        .map(|s| (s.name.clone(), s.faces.clone(), s.material.clone()))
        .chain((!remainder.is_empty()).then(|| (prim.name.clone(), remainder, None)));

    for (name, faces, material_path) in parts {
        let Some(geometry) = mesh_geometry_of_faces(prim, level, &faces) else {
            continue;
        };
        // A subset's own binding, or the mesh's where it has none.
        let (material, textures) = match material_path {
            Some(found) => match layer.prim_at(&found) {
                Some(material) => {
                    let resolved = shade::resolve(material, layer);
                    (resolved.material, resolved.textures)
                }
                None => bound_material_at(path, layer),
            },
            None => bound_material_at(path, layer),
        };
        context.textures.push((name.clone(), textures));

        let mut node = Object3D::mesh(Mesh::new(geometry, material));
        node.name = name;
        let child = arena.insert(node);
        arena.add_child(id, child);
    }
    id
}

/// Put a prim's own transform onto a node already in the arena.
fn place_transform(arena: &mut ObjectArena, id: ObjectId, prim: &UsdPrim) {
    let (position, quaternion, scale) = transform_of(prim);
    if let Some(node) = arena.get_mut(id) {
        node.position = position;
        node.quaternion = quaternion;
        node.scale = scale;
    }
}

/// Animated vertices, as morph targets and the weights that switch between
/// them.
///
/// USD animates a mesh's points by keying the whole array. A renderer has no
/// per-frame vertex buffer to hand, but it has morph targets — so each frame
/// after the first becomes a target holding its difference from the first, and
/// the weights ramp one into the next. That is exactly how a cached
/// simulation is played back in three.js, and it costs one target per frame
/// rather than one mesh per frame.
///
/// The base mesh is the first sample, so a stage opened at its start time looks
/// right before anything is played.
fn vertex_animation(
    source: &UsdPrim,
    object: ObjectId,
    rate: f64,
) -> Option<(crate::core::MorphAttributes, Vec<KeyframeTrack>)> {
    let samples = source.value("points")?.samples()?.to_vec();
    if samples.len() < 2 {
        return None;
    }
    let base = samples[0].1.flat_f32();
    if base.is_empty() {
        return None;
    }

    let mut targets = Vec::new();
    let mut tracks = Vec::new();
    for (index, (_, frame)) in samples.iter().enumerate().skip(1) {
        let points = frame.flat_f32();
        if points.len() != base.len() {
            // A frame with a different vertex count is a different mesh, not a
            // deformation of this one.
            return None;
        }
        targets.push(crate::core::MorphTarget {
            name: format!("frame{index}"),
            position_delta: points.iter().zip(&base).map(|(p, b)| p - b).collect(),
            normal_delta: None,
        });

        // This frame's weight rises from the frame before and falls to the one
        // after, so the blend between any two neighbours is exactly the linear
        // interpolation USD specifies.
        let mut times = Vec::new();
        let mut values = Vec::new();
        for (slot, (time, _)) in samples.iter().enumerate() {
            times.push((time / rate) as f32);
            values.push(if slot == index { 1.0 } else { 0.0 });
        }
        tracks.push(KeyframeTrack::scalar(
            object,
            TrackTarget::MorphWeight {
                index: index - 1,
            },
            times,
            values,
        ));
    }

    Some((
        crate::core::MorphAttributes {
            influences: vec![0.0; targets.len()],
            targets,
        },
        tracks,
    ))
}

/// Tracks for everything else a prim animates: a light's colour and
/// brightness, a material's colour, whether the prim is drawn at all.
///
/// These are not transform ops, so the op-stack pass does not see them, and a
/// lamp that switches on halfway through a shot stays on — or off — for all of
/// it.
fn property_tracks(
    prim: &UsdPrim,
    layer: &UsdLayer,
    path: &str,
    object: ObjectId,
    rate: f64,
) -> Vec<KeyframeTrack> {
    let mut out = Vec::new();

    let scalar = |out: &mut Vec<KeyframeTrack>, value: &UsdValue, target: TrackTarget| {
        let Some(samples) = value.samples() else { return };
        let times: Vec<f32> = samples.iter().map(|(t, _)| (t / rate) as f32).collect();
        let values: Vec<f32> = samples
            .iter()
            .map(|(_, v)| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        out.push(KeyframeTrack::scalar(object, target, times, values));
    };
    let colour = |out: &mut Vec<KeyframeTrack>, value: &UsdValue| {
        let Some(samples) = value.samples() else { return };
        let times: Vec<f32> = samples.iter().map(|(t, _)| (t / rate) as f32).collect();
        let values: Vec<Color> = samples
            .iter()
            .map(|(_, v)| {
                let n = v.flat_f32();
                if n.len() >= 3 {
                    Color::new(n[0], n[1], n[2])
                } else {
                    Color::WHITE
                }
            })
            .collect();
        out.push(KeyframeTrack::color(object, TrackTarget::Color, times, values));
    };

    // `visibility` is a token — `inherited` or `invisible` — so what is
    // animated is which of the two, not a number.
    if let Some(value) = prim.value("visibility") {
        if let Some(samples) = value.samples() {
            let times: Vec<f32> = samples.iter().map(|(t, _)| (t / rate) as f32).collect();
            let values: Vec<f32> = samples
                .iter()
                .map(|(_, v)| {
                    if v.as_str() == Some("invisible") {
                        0.0
                    } else {
                        1.0
                    }
                })
                .collect();
            out.push(KeyframeTrack::scalar(
                object,
                TrackTarget::Visibility,
                times,
                values,
            ));
        }
    }

    if schemas::is_light(&prim.type_name) {
        if let Some(value) = prim.value("inputs:intensity").or_else(|| prim.value("intensity")) {
            scalar(&mut out, value, TrackTarget::Intensity);
        }
        if let Some(value) = prim.value("inputs:color").or_else(|| prim.value("color")) {
            colour(&mut out, value);
        }
    }

    // A mesh's colour is animated on the shader that its material binds, which
    // is a different prim in a different part of the tree.
    if prim.type_name == "Mesh" || schemas::is_gprim(&prim.type_name) {
        if let Some(shader) = shade::bound_material_path(layer, path)
            .and_then(|p| layer.prim_at(&p))
            .and_then(|material| shade::surface_shader(material, layer))
        {
            if let Some(value) = shader.value("inputs:diffuseColor") {
                colour(&mut out, value);
            }
        }
    }
    out
}

/// Every time code any of this prim's transform ops is keyed at.
fn keyed_times(prim: &UsdPrim) -> Vec<f64> {
    let mut times = Vec::new();
    for property in &prim.properties {
        if !property.name.starts_with("xformOp:") {
            continue;
        }
        if let Some(samples) = property.value.samples() {
            times.extend(samples.iter().map(|(t, _)| *t));
        }
    }
    times
}

fn resolve_prim(prim: &UsdPrim, time: f64) -> UsdPrim {
    let mut out = prim.clone();
    for property in &mut out.properties {
        if property.value.samples().is_some() {
            property.value = property.value.at_time(time);
        }
    }
    out
}

/// What a build pass carries down the tree.
///
/// The skeleton is here because skinning is not local: a `Mesh` says which
/// skeleton binds it, and that skeleton is a sibling — so it has to be read
/// before the mesh that uses it, and carried down.
struct Build<'a> {
    layer: &'a UsdLayer,
    /// The layer *before* it was resolved to an instant. A `SkelAnimation` is
    /// nothing but time samples, and the resolved layer has already replaced
    /// them with the pose at one frame — so reading the animation from it
    /// finds a single unchanging value.
    source_layer: &'a UsdLayer,
    placed: &'a mut Vec<(ObjectId, UsdPrim)>,
    cameras: &'a mut Vec<UsdCamera>,
    skeletons: &'a mut Vec<Skeleton>,
    /// Clips a `SkelAnimation` produced, which are per-skeleton rather than
    /// per-prim and so cannot come from the transform pass.
    clips: &'a mut Vec<AnimationClip>,
    /// Time codes per second, for turning the animation's keys into seconds.
    rate: f64,
    options: &'a SceneOptions,
    /// Textures each mesh asked for, by prim name. Decoding an image is not
    /// this module's business, so what a material wants is reported rather
    /// than loaded.
    textures: &'a mut Vec<(String, Vec<shade::TextureRequest>)>,
    /// Blend shapes, which a `Mesh` has no field for.
    morphs: &'a mut Vec<(String, crate::core::MorphAttributes)>,
    /// The skeleton in scope, and the joints it was built from.
    skeleton: Option<Rc<UsdSkeleton>>,
    /// Where the skeleton's animation lives, for the meshes bound to it: a
    /// blend shape's weights are keyed on the animation and applied to the
    /// mesh, which are different prims.
    skeleton_animation: Option<String>,
}

fn build(
    prim: &UsdPrim,
    arena: &mut ObjectArena,
    source: &UsdPrim,
    parent: &str,
    context: &mut Build,
) -> Option<ObjectId> {
    let layer = context.layer;
    // The absolute path, which material binding needs: a binding applies to
    // everything beneath the prim it sits on, so resolving one means walking
    // back up the namespace.
    let path = format!("{parent}/{}", prim.name);
    // A class is a template for other prims, never itself part of the scene.
    if prim.specifier == super::parse::Specifier::Class {
        return None;
    }
    // `guide` geometry is never drawn, and `proxy` is a stand-in shown only
    // when the real thing is absent — which here it never is.
    if !schemas::is_drawable(prim, context.options.show_guides, context.options.show_proxies) {
        return None;
    }
    // Filled in if this prim turns out to carry blend shapes, so the weight
    // tracks can index them in the order they came out.
    let mut morph_order: Vec<String> = Vec::new();
    let mut object = match prim.type_name.as_str() {
        // A mesh with material subsets is several meshes: this crate's `Mesh`
        // carries one material, so the faces are split rather than the
        // material list extended.
        "Mesh" if !shade::material_subsets(prim).is_empty() => {
            return Some(split_by_subset(prim, arena, source, &path, context));
        }
        "Mesh" => {
            let mut geometry = mesh_geometry_at(prim, context.options.subdivision_level)?;
            let (material, textures) = bound_material_at(&path, layer);
            context.textures.push((prim.name.clone(), textures));
            // A mesh that names a skeleton and carries weights is skinned.
            match context.skeleton.clone() {
                Some(skeleton)
                    // It must *say* it binds one: a rig keeps static props in
                    // the same subtree, and they must not be skinned by
                    // proximity.
                    if skel::binds_a_skeleton(prim)
                        && skel::skin_attributes(
                            prim,
                            geometry.get_attribute("position").map(|a| a.count()).unwrap_or(0),
                            &mut geometry,
                        ) =>
                {
                    skel::apply_bind_transform(prim, &mut geometry);
                    // Blend shapes, whose weights the skeleton's animation
                    // drives. The order here is the order the weight tracks
                    // index by.
                    let vertices = geometry
                        .get_attribute("position")
                        .map(|a| a.count())
                        .unwrap_or(0);
                    let targets = skel::blend_shapes(prim, layer, vertices);
                    if !targets.is_empty() {
                        morph_order = targets.iter().map(|t| t.name.clone()).collect();
                        context.morphs.push((
                            prim.name.clone(),
                            crate::core::MorphAttributes {
                                influences: vec![0.0; targets.len()],
                                targets,
                            },
                        ));
                    }
                    Object3D::skinned_mesh(crate::core::SkinnedMesh::new(
                        geometry,
                        material,
                        skeleton.skeleton.clone(),
                    ))
                }
                _ => Object3D::mesh(Mesh::new(geometry, material)),
            }
        }
        // The quadrics and the cube, which carry their shape in a handful of
        // numbers rather than in a vertex buffer.
        type_name if schemas::is_gprim(type_name) => match schemas::gprim(prim) {
            Some(geometry) => {
                let (material, textures) = bound_material_at(&path, layer);
                context.textures.push((prim.name.clone(), textures));
                Object3D::mesh(Mesh::new(geometry, material))
            }
            None => Object3D::group(),
        },
        type_name if schemas::is_light(type_name) => match schemas::light(prim) {
            Some(light) => Object3D::light(light),
            None => Object3D::group(),
        },
        "Points" => match schemas::points(prim) {
            Some(points) => Object3D::points(points),
            None => Object3D::group(),
        },
        "BasisCurves" | "NurbsCurves" => match schemas::curves(prim) {
            Some(curves) => Object3D::line_segments(curves),
            None => Object3D::group(),
        },
        // A point instancer becomes a group with one instanced mesh under it
        // per prototype: an instanced draw carries one geometry, so a scatter
        // of trees and rocks is two of them rather than one.
        "PointInstancer" => {
            let level = context.options.subdivision_level;
            let prototype_root = path.clone();
            let meshes = schemas::point_instancer(prim, layer, |prototype| {
                let path = format!("{prototype_root}/{}", prototype.name);
                mesh_geometry_at(prototype, level)
                    .map(|g| (g, bound_material_at(&path, layer).0))
            });
            let mut group = Object3D::group();
            group.name = prim.name.clone();
            let id = arena.insert(group);
            for (name, mesh) in meshes {
                let mut node = Object3D::instanced_mesh(mesh);
                node.name = name;
                let child = arena.insert(node);
                arena.add_child(id, child);
            }
            // The prototypes themselves are not part of the scene: they are
            // the thing being instanced, not a copy standing beside it.
            place_transform(arena, id, prim);
            context.placed.push((id, source.clone()));
            return Some(id);
        }
        // Everything else becomes a group: an Xform carries a transform, a
        // Scope carries nothing, and an unknown type is still a place in the
        // hierarchy that its children hang from.
        _ => Object3D::group(),
    };
    object.name = prim.name.clone();
    let (position, quaternion, scale) = transform_of(prim);
    object.position = position;
    object.quaternion = quaternion;
    object.scale = scale;
    if prim.value("visibility").and_then(|v| v.as_str()) == Some("invisible") {
        object.visible = false;
    }

    let id = arena.insert(object);
    // Blend-shape weights are driven by the skeleton's animation, and the
    // tracks point at the mesh rather than at a joint.
    if !morph_order.is_empty() {
        if let Some(anim) = source
            .children
            .iter()
            .find(|c| c.type_name == "SkelAnimation")
            .or_else(|| {
                context
                    .skeleton_animation
                    .as_ref()
                    .and_then(|path| context.source_layer.prim_at(path))
            })
        {
            let tracks = skel::blend_shape_tracks(anim, &morph_order, id, context.rate);
            if !tracks.is_empty() {
                context
                    .clips
                    .push(AnimationClip::new(format!("{}-blend", prim.name), 0.0, tracks));
            }
        }
    }
    // A camera is not a node kind, so it is recorded against the node that
    // carries its transform.
    if prim.type_name == "Camera" {
        context.cameras.push(schemas::camera(prim, id));
    }
    // Remember the *unresolved* prim, so the clip can re-evaluate it at each
    // of its own key times.
    context.placed.push((id, source.clone()));

    // Anything animated that is not a transform op: the colours, the
    // brightness, the visibility. Read from the unresolved layer, since the
    // resolved one has already been reduced to a single frame.
    let mut tracks = property_tracks(source, context.source_layer, &path, id, context.rate);

    // Animated vertices, which become morph targets rather than a mesh a
    // frame. Only where the prim has no blend shapes of its own: the two would
    // be indexing the same weights.
    if morph_order.is_empty() {
        if let Some((morphs, weights)) = vertex_animation(source, id, context.rate) {
            context.morphs.push((prim.name.clone(), morphs));
            tracks.extend(weights);
        }
    }

    if !tracks.is_empty() {
        context
            .clips
            .push(AnimationClip::new(format!("{}-properties", prim.name), 0.0, tracks));
    }

    // A `SkelRoot` reads its skeleton before its meshes, because the meshes
    // are what bind to it.
    let outer = context.skeleton.take();
    let skeleton_prim = prim.children.iter().find(|c| c.type_name == "Skeleton");
    match skeleton_prim.and_then(|c| skel::skeleton(c, arena, id)) {
        Some(found) => {
            // The animation the skeleton names, keyed on its own joints, read
            // from the unresolved layer so its samples are still samples.
            let source_skeleton = source.children.iter().find(|c| c.type_name == "Skeleton");
            if let Some(clip) = source_skeleton
                .and_then(|c| skel::animation_source(c, context.source_layer))
                .or_else(|| source.children.iter().find(|c| c.type_name == "SkelAnimation"))
                .and_then(|anim| skel::animation(anim, &found, context.rate, &prim.name))
            {
                context.clips.push(clip);
            }
            context.skeleton_animation = source_skeleton
                .and_then(|c| c.value("skel:animationSource"))
                .and_then(|v| v.as_str())
                .map(str::to_string);
            context.skeletons.push(found.skeleton.clone());
            context.skeleton = Some(Rc::new(found));
        }
        None => context.skeleton = outer.clone(),
    }

    for (child, child_source) in prim.children.iter().zip(&source.children) {
        // The skeleton's own joints were built above; the prim itself is not
        // part of the scene.
        if child.type_name == "Skeleton" {
            continue;
        }
        if let Some(child_id) = build(child, arena, child_source, &path, context) {
            arena.add_child(id, child_id);
        }
    }
    context.skeleton = outer;
    Some(id)
}

/// The transform a prim's `xformOp`s add up to.
///
/// `xformOpOrder` is authoritative: the ops are applied in the order it lists,
/// and an op present but unlisted is deliberately inactive. Without it, USD
/// applies nothing — which is not the same as identity by accident, so the
/// absence is honoured rather than guessed around.
fn transform_of(prim: &UsdPrim) -> (Vector3, Quaternion, Vector3) {
    let identity = (Vector3::ZERO, Quaternion::identity(), Vector3::new(1.0, 1.0, 1.0));
    let Some(order) = prim.value("xformOpOrder") else {
        return identity;
    };
    let mut matrix = Matrix4::identity();
    let mut any = false;
    for op in order.items() {
        let Some(name) = op.as_str() else { continue };
        // `!invert!xformOp:translate` inverts the op it names.
        let (invert, name) = match name.strip_prefix("!invert!") {
            Some(rest) => (true, rest),
            None => (false, name),
        };
        let Some(value) = prim.value(name) else { continue };
        let step = op_matrix(name, value);
        let Some(step) = step else { continue };
        any = true;
        let step = if invert { step.invert() } else { step };
        matrix = matrix.multiply(&step);
    }
    if !any {
        return identity;
    }
    matrix.decompose()
}

fn op_matrix(name: &str, value: &UsdValue) -> Option<Matrix4> {
    let n = value.flat_f32();
    // The op's kind is in its name after the namespace; a suffixed op like
    // `xformOp:translate:pivot` is still a translate.
    let kind = name.strip_prefix("xformOp:")?;
    let kind = kind.split(':').next().unwrap_or(kind);
    match kind {
        "translate" if n.len() >= 3 => Some(Matrix4::compose(
            Vector3::new(n[0], n[1], n[2]),
            Quaternion::identity(),
            Vector3::new(1.0, 1.0, 1.0),
        )),
        "scale" if n.len() >= 3 => Some(Matrix4::compose(
            Vector3::ZERO,
            Quaternion::identity(),
            Vector3::new(n[0], n[1], n[2]),
        )),
        // A single-axis rotation, in degrees as USD writes them.
        "rotateX" | "rotateY" | "rotateZ" if !n.is_empty() => {
            let axis = match kind {
                "rotateX" => Vector3::new(1.0, 0.0, 0.0),
                "rotateY" => Vector3::new(0.0, 1.0, 0.0),
                _ => Vector3::new(0.0, 0.0, 1.0),
            };
            Some(Matrix4::compose(
                Vector3::ZERO,
                Quaternion::from_axis_angle(axis, n[0].to_radians()),
                Vector3::new(1.0, 1.0, 1.0),
            ))
        }
        // Euler triples. The name gives the order the axes are applied in.
        "rotateXYZ" | "rotateXZY" | "rotateYXZ" | "rotateYZX" | "rotateZXY" | "rotateZYX"
            if n.len() >= 3 =>
        {
            let axes = &kind[6..];
            let mut q = Quaternion::identity();
            for (i, axis) in axes.chars().enumerate() {
                let v = match axis {
                    'X' => Vector3::new(1.0, 0.0, 0.0),
                    'Y' => Vector3::new(0.0, 1.0, 0.0),
                    _ => Vector3::new(0.0, 0.0, 1.0),
                };
                let step = Quaternion::from_axis_angle(v, n[i].to_radians());
                q = q.multiply(step);
            }
            Some(Matrix4::compose(
                Vector3::ZERO,
                q,
                Vector3::new(1.0, 1.0, 1.0),
            ))
        }
        // USD writes a quaternion real-part first; this crate stores it last.
        "orient" if n.len() >= 4 => Some(Matrix4::compose(
            Vector3::ZERO,
            Quaternion::new(n[1], n[2], n[3], n[0]),
            Vector3::new(1.0, 1.0, 1.0),
        )),
        "transform" if n.len() >= 16 => {
            // USD matrices are row-major and pre-multiplied; this crate's are
            // column-major, so the sixteen numbers transpose on the way in.
            let mut m = [0.0f32; 16];
            for row in 0..4 {
                for col in 0..4 {
                    m[col * 4 + row] = n[row * 4 + col];
                }
            }
            Some(Matrix4 { elements: m })
        }
        _ => None,
    }
}

/// Read a `UsdGeomMesh` as triangles.
pub fn mesh_geometry(prim: &UsdPrim) -> Option<BufferGeometry> {
    mesh_geometry_at(prim, 0)
}

/// A mesh's geometry restricted to a set of faces.
///
/// The vertices are not re-indexed: dropping the faces a subset does not claim
/// leaves unreferenced points behind, which cost memory and nothing else — and
/// keeping the indices as authored means a primvar or a skin weight still lines
/// up with the vertex it was written for.
pub fn mesh_geometry_of_faces(
    prim: &UsdPrim,
    level: usize,
    faces: &[u32],
) -> Option<BufferGeometry> {
    let counts = prim.value("faceVertexCounts")?.flat_u32();
    let indices = prim.value("faceVertexIndices")?.flat_u32();

    // Where each face starts in the index buffer.
    let mut starts = Vec::with_capacity(counts.len());
    let mut at = 0usize;
    for count in &counts {
        starts.push(at);
        at += *count as usize;
    }

    let mut kept_counts = Vec::with_capacity(faces.len());
    let mut kept_indices = Vec::new();
    for face in faces {
        let (Some(start), Some(count)) = (
            starts.get(*face as usize).copied(),
            counts.get(*face as usize).copied(),
        ) else {
            continue;
        };
        if start + count as usize > indices.len() {
            continue;
        }
        kept_counts.push(count);
        kept_indices.extend_from_slice(&indices[start..start + count as usize]);
    }
    if kept_counts.is_empty() {
        return None;
    }

    // Build a prim standing for just those faces, so the usual path — winding,
    // primvars, subdivision — applies unchanged.
    let mut only = prim.clone();
    only.children.clear();
    for (name, value) in [
        ("faceVertexCounts", UsdValue::Array(
            kept_counts.iter().map(|c| UsdValue::Int(*c as i128)).collect(),
        )),
        ("faceVertexIndices", UsdValue::Array(
            kept_indices.iter().map(|i| UsdValue::Int(*i as i128)).collect(),
        )),
    ] {
        if let Some(property) = only.properties.iter_mut().find(|p| p.name == name) {
            property.value = value;
        }
    }
    mesh_geometry_at(&only, level)
}

/// The same, refined `level` times if the mesh says it is a subdivision
/// surface.
///
/// Worth knowing: `subdivisionScheme` defaults to `catmullClark`, so a mesh
/// that says nothing is one — and its points are a control cage rather than
/// the surface. Refining is not free and the level is the renderer's call, not
/// the file's, which is why it is a parameter here and a complexity slider in
/// usdview.
pub fn mesh_geometry_at(prim: &UsdPrim, level: usize) -> Option<BufferGeometry> {
    let mut points = prim.value("points")?.flat_f32();
    if points.len() < 9 {
        return None;
    }
    let mut counts = prim
        .value("faceVertexCounts")
        .map(|v| v.flat_u32())
        .unwrap_or_default();
    let mut indices = prim
        .value("faceVertexIndices")
        .map(|v| v.flat_u32())
        .unwrap_or_default();
    // A mesh with no topology is a point cloud as far as this is concerned, and
    // there is nothing to draw.
    if counts.is_empty() || indices.is_empty() {
        return None;
    }

    // Refinement happens on the polygons, before anything is triangulated:
    // Catmull–Clark is defined on the cage, and a fan of triangles is a
    // different cage with a different limit surface.
    if level > 0 && subdiv::wanted(prim) {
        let refined = subdiv::subdivide(
            subdiv::Cage {
                positions: points.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect(),
                counts: counts.clone(),
                indices: indices.clone(),
            },
            level,
            &subdiv::options_of(prim),
        );
        points = refined.positions.iter().flat_map(|p| *p).collect();
        counts = refined.counts;
        indices = refined.indices;
        // Refining rebuilds the topology, so anything authored per corner of
        // the old cage no longer lines up with the new one.
        return refined_geometry(points, counts, indices, prim);
    }

    // `leftHanded` reverses what USD considers the front of every face.
    let flip = prim.value("orientation").and_then(|v| v.as_str()) == Some("leftHanded");

    // An index past the end of the points is a file saying something about a
    // vertex that is not there. It has to be refused here: everything
    // downstream — normals, tangents, the renderer — trusts an index buffer to
    // be in range, and a corrupt asset would otherwise take the process with
    // it.
    let vertices = points.len() / 3;
    if indices.iter().any(|i| *i as usize >= vertices) {
        return None;
    }

    // Corner `c` of the mesh, as a position in `indices`, fanned into triangles.
    let mut corners: Vec<usize> = Vec::new();
    let mut at = 0usize;
    for count in &counts {
        let n = *count as usize;
        if n >= 3 && at + n <= indices.len() {
            for k in 1..n - 1 {
                if flip {
                    corners.extend_from_slice(&[at, at + k + 1, at + k]);
                } else {
                    corners.extend_from_slice(&[at, at + k, at + k + 1]);
                }
            }
        }
        at += n;
    }
    if corners.is_empty() {
        return None;
    }

    let normals = primvar(prim, "normals").or_else(|| primvar(prim, "primvars:normals"));
    let uvs = primvar(prim, "primvars:st")
        .or_else(|| primvar(prim, "primvars:st0"))
        .or_else(|| primvar(prim, "primvars:UVMap"));
    // `displayColor` is USD's fallback shading and every tool that does not
    // evaluate a shader graph uses it. Per vertex it is a colour attribute;
    // one value for the whole mesh is the material's colour and is left to the
    // material, which already reads it.
    let colours = primvar(prim, "primvars:displayColor").filter(|pv| !pv.constant);
    let face_varying = [&normals, &uvs, &colours]
        .iter()
        .any(|p| matches!(p, Some(pv) if pv.face_varying));

    let mut geometry = BufferGeometry::new();
    if face_varying {
        // One vertex per corner: the only way a per-corner normal survives.
        let mut pos = Vec::with_capacity(corners.len() * 3);
        let mut nrm = Vec::new();
        let mut uv = Vec::new();
        let mut col = Vec::new();
        for &c in &corners {
            let point = indices[c] as usize;
            pos.extend_from_slice(slice3(&points, point));
            if let Some(pv) = &normals {
                nrm.extend_from_slice(&pv.at3(c, indices[c] as usize));
            }
            if let Some(pv) = &uvs {
                uv.extend_from_slice(&pv.at2(c, indices[c] as usize));
            }
            if let Some(pv) = &colours {
                col.extend_from_slice(&pv.at3(c, indices[c] as usize));
            }
        }
        geometry.set_attribute("position", BufferAttribute::new(pos, 3));
        if !nrm.is_empty() {
            geometry.set_attribute("normal", BufferAttribute::new(nrm, 3));
        }
        if !uv.is_empty() {
            geometry.set_attribute("uv", BufferAttribute::new(uv, 2));
        }
        if !col.is_empty() {
            geometry.set_attribute("color", BufferAttribute::new(col, 3));
        }
    } else {
        geometry.set_attribute("position", BufferAttribute::new(points.clone(), 3));
        let vertices = points.len() / 3;
        if let Some(pv) = &normals {
            let mut nrm = Vec::with_capacity(vertices * 3);
            for v in 0..vertices {
                nrm.extend_from_slice(&pv.at3(v, v));
            }
            geometry.set_attribute("normal", BufferAttribute::new(nrm, 3));
        }
        if let Some(pv) = &uvs {
            let mut uv = Vec::with_capacity(vertices * 2);
            for v in 0..vertices {
                uv.extend_from_slice(&pv.at2(v, v));
            }
            geometry.set_attribute("uv", BufferAttribute::new(uv, 2));
        }
        if let Some(pv) = &colours {
            let mut col = Vec::with_capacity(vertices * 3);
            for v in 0..vertices {
                col.extend_from_slice(&pv.at3(v, v));
            }
            geometry.set_attribute("color", BufferAttribute::new(col, 3));
        }
        geometry.set_index(corners.iter().map(|&c| indices[c]).collect());
    }
    if geometry.get_attribute("normal").is_none() {
        crate::compute_vertex_normals(&mut geometry);
    }
    Some(geometry)
}

fn slice3(v: &[f32], i: usize) -> &[f32] {
    let at = i * 3;
    if at + 3 <= v.len() {
        &v[at..at + 3]
    } else {
        &[0.0, 0.0, 0.0]
    }
}

/// A primvar's values, plus how they are addressed.
struct Primvar {
    data: Vec<f32>,
    /// `primvars:st:indices` — an indirection USD uses to share values.
    indices: Vec<u32>,
    face_varying: bool,
    /// `interpolation = "constant"` — one value for the whole surface, which
    /// is a material colour rather than a vertex attribute. Reading it as one
    /// gives the first vertex the colour and the rest whatever
    /// out-of-range reads back, which for `displayColor` is black.
    constant: bool,
    stride: usize,
}

impl Primvar {
    fn get(&self, corner: usize, vertex: usize) -> usize {
        let i = if self.face_varying { corner } else { vertex };
        if self.indices.is_empty() {
            i
        } else {
            *self.indices.get(i).unwrap_or(&0) as usize
        }
    }

    fn at3(&self, corner: usize, vertex: usize) -> [f32; 3] {
        let at = self.get(corner, vertex) * self.stride;
        [
            *self.data.get(at).unwrap_or(&0.0),
            *self.data.get(at + 1).unwrap_or(&0.0),
            *self.data.get(at + 2).unwrap_or(&0.0),
        ]
    }

    fn at2(&self, corner: usize, vertex: usize) -> [f32; 2] {
        let at = self.get(corner, vertex) * self.stride;
        [
            *self.data.get(at).unwrap_or(&0.0),
            *self.data.get(at + 1).unwrap_or(&0.0),
        ]
    }
}

fn primvar(prim: &UsdPrim, name: &str) -> Option<Primvar> {
    let property: &UsdProperty = prim.property(name)?;
    let data = property.value.flat_f32();
    if data.is_empty() {
        return None;
    }
    let interpolation = property.meta("interpolation").and_then(|v| v.as_str());
    let face_varying = interpolation == Some("faceVarying");
    let constant = interpolation == Some("constant");
    let indices = prim
        .value(&format!("{name}:indices"))
        .map(|v| v.flat_u32())
        .unwrap_or_default();
    let stride = if name.contains("st") || name.contains("UV") {
        2
    } else {
        3
    };
    // One value and nothing said about interpolation is constant too: USD's
    // default is `constant`, and a single colour is not a vertex attribute.
    let constant = constant || data.len() == stride;
    Some(Primvar {
        data,
        indices,
        face_varying,
        constant,
        stride,
    })
}

/// The material a mesh is bound to, read as a `UsdPreviewSurface`.
/// The material bound to a prim, resolved through its shader graph.
///
/// `path` is the prim's absolute path, which is needed because binding is
/// inherited: a mesh with none of its own wears the nearest ancestor's.
fn bound_material_at(path: &str, layer: &UsdLayer) -> (Material, Vec<shade::TextureRequest>) {
    let (mut material, textures) =
        match shade::bound_material_path(layer, path).and_then(|p| layer.prim_at(&p)) {
            Some(prim) => {
                let resolved = shade::resolve(prim, layer);
                (resolved.material, resolved.textures)
            }
            None => (
                StandardMaterial::new(Color::new(0.8, 0.8, 0.8)).into(),
                Vec::new(),
            ),
        };
    // USD keeps sidedness on the surface and this crate keeps it on the
    // material, so it has to move across. It is a gprim attribute and does not
    // inherit, so it is read from the mesh itself and from nowhere else.
    if matches!(
        layer.prim_at(path).and_then(|p| p.value("doubleSided")),
        Some(UsdValue::Bool(true))
    ) {
        material.set_side(2);
    }
    (material, textures)
}

/// Load the images a `.usdz`'s materials asked for, out of the archive itself.
///
/// [`UsdScene::textures`] exists because a scene-description reader has no
/// business decoding images or reaching for files beside the layer. A `.usdz`
/// is the case where neither objection holds: the package is required to be
/// self-contained, so the bytes are already in hand, and this crate has a PNG
/// decoder. So a packaged asset arrives ready to draw instead of arriving with
/// a list of homework.
///
/// Requests this cannot satisfy — a JPEG, which needs a decoder that is not
/// here, or a file the archive does not carry — are left in
/// [`UsdScene::textures`] for the caller, so nothing is lost by asking.
///
/// Returns how many images were attached.
pub fn attach_textures(scene: &mut UsdScene, archive: &UsdzArchive) -> usize {
    attach_with(scene, |request| decode_from(archive, request))
}

/// Walk every request, attaching whatever `load` can resolve and keeping the
/// rest for the caller.
fn attach_with(
    scene: &mut UsdScene,
    load: impl Fn(&shade::TextureRequest) -> Option<Texture>,
) -> usize {
    let mut attached = 0;
    let mut pending: Vec<(String, Vec<shade::TextureRequest>)> = Vec::new();

    for (name, requests) in std::mem::take(&mut scene.textures) {
        let mut unresolved = Vec::new();
        let targets: Vec<ObjectId> = scene
            .roots
            .iter()
            .flat_map(|root| scene.arena.get_objects_by_name(*root, &name))
            .collect();
        for request in requests {
            match load(&request) {
                Some(texture) => {
                    let texture = std::sync::Arc::new(texture);
                    let mut used = false;
                    for id in &targets {
                        if let Some(object) = scene.arena.get_mut(*id) {
                            used |= put_map(object, request.slot, &texture);
                        }
                    }
                    if used {
                        attached += 1;
                    } else {
                        unresolved.push(request);
                    }
                }
                None => unresolved.push(request),
            }
        }
        if !unresolved.is_empty() {
            pending.push((name, unresolved));
        }
    }
    scene.textures = pending;
    attached
}

/// Load the images a `.usda` or `.usdc` asked for, from beside the layer.
///
/// The counterpart to [`attach_textures`] for the two forms that are not
/// packages. A layer refers to its maps by path relative to itself, so `base`
/// is the directory the layer was read from. Nothing is guessed at: a path
/// that escapes `base` or names a file that is not there is left in
/// [`UsdScene::textures`] for the caller, exactly as before.
///
/// Returns how many images were attached.
#[cfg(not(target_arch = "wasm32"))]
pub fn attach_textures_from_dir(scene: &mut UsdScene, base: &std::path::Path) -> usize {
    let read = |request: &shade::TextureRequest| -> Option<Vec<u8>> {
        let relative = std::path::Path::new(request.file.trim_start_matches("./"));
        // An absolute path, or one climbing out with `..`, is not this
        // function's to follow — it was asked for a directory, not the disk.
        if relative.is_absolute()
            || relative
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return None;
        }
        std::fs::read(base.join(relative)).ok()
    };
    attach_with(scene, |request| read(request).and_then(|bytes| decode(&bytes, request)))
}

/// One request as a texture, out of the archive that carries it.
fn decode_from(archive: &UsdzArchive, request: &shade::TextureRequest) -> Option<Texture> {
    // An asset path may be written `./foo.png` or with the package prefix a
    // flattened layer picks up; the archive keys on the plain name.
    let wanted = request.file.trim_start_matches("./");
    let entry = archive
        .entries
        .iter()
        .find(|e| e.name == wanted || e.name.ends_with(&format!("/{wanted}")))?;
    decode(&entry.data, request)
}

/// Image bytes as the texture the request describes.
fn decode(bytes: &[u8], request: &shade::TextureRequest) -> Option<Texture> {
    let image = crate::utils::png::decode_png(bytes).ok()?;
    // A data map must not be read through an sRGB curve, and a colour map
    // must. The request carries which, because USD authors it.
    let format = if request.color_space == "raw" {
        TextureFormat::Rgba8Unorm
    } else {
        TextureFormat::Rgba8UnormSrgb
    };
    Some(Texture {
        wrap_s: wrap_from(&request.wrap_s),
        wrap_t: wrap_from(&request.wrap_t),
        ..Texture::new(image.width, image.height, format, image.rgba)
    })
}

/// USD's wrap mode as this crate's.
///
/// `black` has no equivalent here — it is a border colour, not a wrap — and
/// clamping is the nearest thing that does not tile.
fn wrap_from(name: &str) -> TextureWrap {
    match name {
        "repeat" => TextureWrap::Repeat,
        "mirror" => TextureWrap::MirroredRepeat,
        _ => TextureWrap::ClampToEdge,
    }
}

/// Hang a texture on the slot it belongs to, reporting whether it fit.
fn put_map(object: &mut Object3D, slot: shade::TextureSlot, texture: &std::sync::Arc<Texture>) -> bool {
    use shade::TextureSlot as S;
    let ObjectKind::Mesh(mesh) = &mut object.kind else {
        return false;
    };
    let material = std::sync::Arc::make_mut(&mut mesh.material);
    macro_rules! set {
        ($m:expr, $($slot:ident => $field:ident),* $(,)?) => {
            match slot {
                $(S::$slot => { $m.$field = Some(texture.clone()); true })*
                _ => false,
            }
        };
    }
    match material {
        Material::Standard(m) => set!(m,
            BaseColor => map,
            Normal => normal_map,
            Roughness => roughness_map,
            Metalness => metalness_map,
            Emissive => emissive_map,
            Occlusion => ao_map,
            Displacement => displacement_map,
        ),
        Material::Physical(m) => set!(m,
            BaseColor => map,
            Normal => normal_map,
            Roughness => roughness_map,
            Metalness => metalness_map,
            Emissive => emissive_map,
            Occlusion => ao_map,
            Displacement => displacement_map,
        ),
        Material::Basic(m) => set!(m, BaseColor => map),
        Material::Sprite(m) => set!(m, BaseColor => map),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse;
    use super::*;
    use crate::animation::TrackValues;

    const QUAD: &str = r#"#usda 1.0
def Xform "World"
{
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Quad"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </World/Red>
    }

    def Material "Red"
    {
        def Shader "S"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1, 0, 0)
            float inputs:roughness = 0.25
            float inputs:metallic = 1
        }
    }
}
"#;

    #[test]
    fn an_animated_layer_comes_back_with_a_clip() {
        let layer = parse(include_str!("testdata/anim.usda")).unwrap();
        let scene = to_scene(&layer);

        let clip = scene.animations.first().expect("a clip");
        // 48 time codes at 24 a second.
        assert_eq!(clip.duration, 2.0);

        // The prim has a moving translate and a moving rotate, so it gets a
        // position and a quaternion track — and no scale track, because
        // nothing scales it.
        let spin = scene.roots[0];
        let targets: Vec<_> = clip
            .tracks
            .iter()
            .filter(|t| t.object == spin)
            .map(|t| format!("{:?}", t.target))
            .collect();
        assert!(targets.contains(&"Position".to_string()), "{targets:?}");
        assert!(targets.contains(&"Quaternion".to_string()), "{targets:?}");
        assert!(!targets.contains(&"Scale".to_string()), "{targets:?}");
    }

    #[test]
    fn the_position_track_holds_what_the_samples_said() {
        let layer = parse(include_str!("testdata/anim.usda")).unwrap();
        let scene = to_scene(&layer);
        let clip = &scene.animations[0];
        let track = clip
            .tracks
            .iter()
            .find(|t| matches!(t.target, TrackTarget::Position))
            .expect("position track");

        // Times are seconds, not time codes.
        assert_eq!(track.times, vec![0.0, 1.0, 2.0]);
        let TrackValues::Vector(values) = &track.values else {
            panic!("expected vectors");
        };
        assert_eq!(values[0], Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(values[1], Vector3::new(10.0, 0.0, 0.0));
        assert_eq!(values[2], Vector3::new(10.0, 10.0, 0.0));
    }

    /// A prim keyed on two different frame sets gets one track covering both,
    /// because the transform is one thing however it was authored.
    #[test]
    fn ops_keyed_on_different_frames_are_merged() {
        let layer = parse(
            r#"#usda 1.0
def Xform "A"
{
    double3 xformOp:translate.timeSamples = { 0: (0, 0, 0), 10: (1, 0, 0) }
    float3 xformOp:scale.timeSamples = { 5: (1, 1, 1), 20: (2, 2, 2) }
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
}
"#,
        )
        .unwrap();
        let scene = to_scene(&layer);
        let clip = &scene.animations[0];
        let position = clip
            .tracks
            .iter()
            .find(|t| matches!(t.target, TrackTarget::Position))
            .unwrap();
        // The union of both key sets: 0, 5, 10, 20.
        assert_eq!(position.times.len(), 4);
        assert_eq!(
            clip.tracks
                .iter()
                .filter(|t| matches!(t.target, TrackTarget::Scale))
                .count(),
            1
        );
    }

    /// The scene is posed at the first frame, not left at the identity.
    #[test]
    fn a_scene_is_built_at_the_start_of_its_range() {
        let layer = parse(
            r#"#usda 1.0
(
    startTimeCode = 10
    endTimeCode = 20
)
def Xform "A"
{
    double3 xformOp:translate.timeSamples = { 0: (0, 0, 0), 10: (7, 0, 0), 20: (9, 0, 0) }
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
"#,
        )
        .unwrap();
        let scene = to_scene(&layer);
        let a = scene.arena.get(scene.roots[0]).unwrap();
        assert_eq!(a.position, Vector3::new(7.0, 0.0, 0.0), "posed at frame 10");

        // And any other instant can be asked for directly.
        let midway = to_scene_at(&layer, 15.0);
        assert_eq!(
            midway.arena.get(midway.roots[0]).unwrap().position,
            Vector3::new(8.0, 0.0, 0.0)
        );
    }

    /// An animated mesh still has geometry — the points resolve to the frame
    /// the scene was built at rather than flattening to nothing.
    #[test]
    fn an_animated_mesh_still_has_its_points() {
        let layer = parse(include_str!("testdata/anim.usda")).unwrap();
        let scene = to_scene(&layer);
        let body = scene
            .arena
            .get(scene.arena.get(scene.roots[0]).unwrap().children[0])
            .unwrap();
        let crate::core::ObjectKind::Mesh(mesh) = &body.kind else {
            panic!("expected a mesh");
        };
        let position = mesh.geometry.get_attribute("position").expect("positions");
        assert_eq!(position.count(), 3);
        // Frame 0's triangle, not frame 48's.
        assert_eq!(&position.array[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&position.array[3..6], &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_static_layer_has_no_clip() {
        let scene = to_scene(&parse(QUAD).unwrap());
        assert!(scene.animations.is_empty());
    }

    #[test]
    fn a_quad_becomes_two_triangles() {
        let layer = parse(QUAD).unwrap();
        let quad = layer.prim_at("/World/Quad").unwrap();
        let g = mesh_geometry(quad).unwrap();
        assert_eq!(g.get_attribute("position").unwrap().array.len(), 12);
        // Fanned: 4 corners give 2 triangles.
        assert_eq!(g.index.as_ref().unwrap().len(), 6);
        assert_eq!(g.get_attribute("uv").unwrap().array.len(), 8);
    }

    #[test]
    fn left_handed_orientation_flips_the_winding() {
        let right = parse(QUAD).unwrap();
        let a = mesh_geometry(right.prim_at("/World/Quad").unwrap()).unwrap();
        let flipped = QUAD.replace(
            "int[] faceVertexCounts = [4]",
            "uniform token orientation = \"leftHanded\"\n        int[] faceVertexCounts = [4]",
        );
        let left = parse(&flipped).unwrap();
        let b = mesh_geometry(left.prim_at("/World/Quad").unwrap()).unwrap();
        let (ia, ib) = (a.index.clone().unwrap(), b.index.clone().unwrap());
        assert_eq!(ia.len(), ib.len());
        // Same first corner, opposite order for the other two.
        assert_eq!((ia[0], ia[1], ia[2]), (ib[0], ib[2], ib[1]));
    }

    #[test]
    fn face_varying_primvars_split_the_vertices() {
        let src = QUAD.replace("interpolation = \"vertex\"", "interpolation = \"faceVarying\"");
        let layer = parse(&src).unwrap();
        let g = mesh_geometry(layer.prim_at("/World/Quad").unwrap()).unwrap();
        // Unindexed: six corners, because a per-corner UV cannot be shared.
        assert!(g.index.is_none());
        assert_eq!(g.get_attribute("position").unwrap().array.len(), 18);
        assert_eq!(g.get_attribute("uv").unwrap().array.len(), 12);
    }

    #[test]
    fn transforms_come_from_the_op_order() {
        let layer = parse(QUAD).unwrap();
        let scene = to_scene(&layer);
        let world = scene.arena.get(scene.roots[0]).unwrap();
        assert_eq!(world.name, "World");
        assert!((world.position.x - 1.0).abs() < 1e-6);
        assert!((world.position.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn an_op_that_is_not_in_the_order_is_inactive() {
        // USD applies exactly what xformOpOrder lists; an authored op that is
        // left out is deliberately off.
        let src = QUAD.replace(
            "uniform token[] xformOpOrder = [\"xformOp:translate\"]",
            "uniform token[] xformOpOrder = []",
        );
        let scene = to_scene(&parse(&src).unwrap());
        let world = scene.arena.get(scene.roots[0]).unwrap();
        assert!(world.position.length() < 1e-9);
    }

    #[test]
    fn the_bound_material_is_read() {
        let layer = parse(QUAD).unwrap();
        let scene = to_scene(&layer);
        let world = scene.roots[0];
        let quad = scene.arena.get(world).unwrap().children[0];
        let mesh = scene.arena.get(quad).unwrap();
        let material = match &mesh.kind {
            crate::core::ObjectKind::Mesh(m) => m.material.clone(),
            other => panic!("expected a mesh, got {other:?}"),
        };
        match &*material {
            Material::Standard(s) => {
                assert!((s.color.r - 1.0).abs() < 1e-6);
                assert!((s.roughness - 0.25).abs() < 1e-6);
                assert!((s.metalness - 1.0).abs() < 1e-6);
            }
            other => panic!("expected a standard material, got {other:?}"),
        }
    }

    #[test]
    fn the_hierarchy_is_preserved() {
        let scene = to_scene(&parse(QUAD).unwrap());
        assert_eq!(scene.roots.len(), 1);
        // World has the mesh and the material prim beneath it; the material is
        // a group with nothing to draw, which is still a node.
        assert_eq!(scene.arena.get(scene.roots[0]).unwrap().children.len(), 2);
    }
}

#[cfg(test)]
mod animation_kinds {
    use super::*;
    use crate::animation::{TrackTarget, TrackValues};
    use crate::core::ObjectKind;

    fn stage() -> UsdScene {
        to_scene(&super::super::parse::parse(include_str!("testdata/allanim.usda")).expect("parses"))
    }

    fn id_of(scene: &UsdScene, name: &str) -> ObjectId {
        fn walk(scene: &UsdScene, id: ObjectId, name: &str) -> Option<ObjectId> {
            let node = scene.arena.get(id)?;
            if node.name == name {
                return Some(id);
            }
            node.children.iter().find_map(|c| walk(scene, *c, name))
        }
        scene
            .roots
            .iter()
            .find_map(|r| walk(scene, *r, name))
            .unwrap_or_else(|| panic!("no prim called {name}"))
    }

    fn tracks_for(
        scene: &UsdScene,
        object: ObjectId,
    ) -> Vec<&crate::animation::KeyframeTrack> {
        scene
            .animations
            .iter()
            .flat_map(|c| &c.tracks)
            .filter(|t| t.object == object)
            .collect()
    }

    /// A lamp that brightens and changes colour. Neither is a transform op, so
    /// the op-stack pass never saw them and the lamp stayed as it started.
    #[test]
    fn a_lights_brightness_and_colour_animate() {
        let scene = stage();
        let lamp = id_of(&scene, "Lamp");
        let tracks = tracks_for(&scene, lamp);

        let intensity = tracks
            .iter()
            .find(|t| matches!(t.target, TrackTarget::Intensity))
            .expect("an intensity track");
        assert_eq!(intensity.times, vec![0.0, 2.0], "48 codes at 24 a second");
        let TrackValues::Scalar(values) = &intensity.values else {
            panic!("scalars");
        };
        assert_eq!(values, &vec![0.0, 100.0]);

        let colour = tracks
            .iter()
            .find(|t| matches!(t.target, TrackTarget::Color))
            .expect("a colour track");
        let TrackValues::Color(values) = &colour.values else {
            panic!("colours");
        };
        assert_eq!((values[0].r, values[0].b), (1.0, 0.0));
        assert_eq!((values[1].r, values[1].b), (0.0, 1.0));
    }

    /// `visibility` is a token rather than a number, so what animates is which
    /// of two words it holds.
    #[test]
    fn visibility_animates() {
        let scene = stage();
        let track = tracks_for(&scene, id_of(&scene, "Blinker"))
            .into_iter()
            .find(|t| matches!(t.target, TrackTarget::Visibility))
            .expect("a visibility track");
        let TrackValues::Scalar(values) = &track.values else {
            panic!("scalars");
        };
        assert_eq!(values, &vec![1.0, 0.0, 1.0], "on, off, on");
    }

    /// A material's colour is animated on the shader its binding names, which
    /// is a different prim in a different part of the tree.
    #[test]
    fn a_bound_materials_colour_animates() {
        let scene = stage();
        let track = tracks_for(&scene, id_of(&scene, "Painted"))
            .into_iter()
            .find(|t| matches!(t.target, TrackTarget::Color))
            .expect("the bound material's colour");
        let TrackValues::Color(values) = &track.values else {
            panic!("colours");
        };
        assert_eq!((values[0].r, values[0].g, values[0].b), (1.0, 1.0, 0.0));
        assert_eq!((values[1].r, values[1].g, values[1].b), (0.0, 1.0, 1.0));
    }

    /// Animated vertices become morph targets: one per frame after the first,
    /// each holding its difference from the base, with weights that ramp one
    /// into the next.
    #[test]
    fn animated_vertices_become_morph_targets() {
        let scene = stage();
        let (_, morphs) = scene
            .morph_targets
            .iter()
            .find(|(name, _)| name == "Wobble")
            .expect("the wobbling mesh");
        assert_eq!(morphs.targets.len(), 2, "three frames, two targets");

        // Frame 1 moves the second point from x=1 to x=2.
        assert_eq!(morphs.targets[0].position_delta[3], 1.0);
        // Frame 2 moves the third point from y=1 to y=3.
        assert_eq!(morphs.targets[1].position_delta[7], 2.0);

        let tracks = tracks_for(&scene, id_of(&scene, "Wobble"));
        let weights: Vec<&crate::animation::KeyframeTrack> = tracks
            .iter()
            .filter(|t| matches!(t.target, TrackTarget::MorphWeight { .. }))
            .copied()
            .collect();
        assert_eq!(weights.len(), 2);

        // Each weight is one at its own frame and zero at the others, so the
        // blend between two neighbours is the linear interpolation USD means.
        let TrackValues::Scalar(values) = &weights[0].values else {
            panic!("scalars");
        };
        assert_eq!(values, &vec![0.0, 1.0, 0.0]);
        let TrackValues::Scalar(values) = &weights[1].values else {
            panic!("scalars");
        };
        assert_eq!(values, &vec![0.0, 0.0, 1.0]);

        // And the mesh itself is the first frame, so a stage opened at its
        // start looks right before anything is played.
        let ObjectKind::Mesh(mesh) = &scene.arena.get(id_of(&scene, "Wobble")).unwrap().kind else {
            panic!("a mesh");
        };
        assert_eq!(&mesh.geometry.get_attribute("position").unwrap().array[3..5], &[1.0, 0.0]);
    }

    /// And every one of them again out of the crate form.
    #[test]
    fn every_kind_survives_the_binary_form() {
        let binary = to_scene(
            &super::super::UsdLoader::parse_layer(include_bytes!("testdata/allanim.usdc"))
                .expect("the crate parses"),
        );
        let kinds: std::collections::BTreeSet<String> = binary
            .animations
            .iter()
            .flat_map(|c| &c.tracks)
            .map(|t| format!("{:?}", t.target))
            .collect();
        for expected in ["Intensity", "Visibility", "Color"] {
            assert!(
                kinds.iter().any(|k| k.contains(expected)),
                "{expected} missing from {kinds:?}"
            );
        }
        assert!(binary.morph_targets.iter().any(|(n, _)| n == "Wobble"));
    }
}
