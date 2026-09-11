//! A cheap hash of everything that changes what [`RaytraceScene::build`] produces.
//!
//! Used to skip BVH rebuilds when only the camera moves, or when `prepare` is
//! called repeatedly on an unchanged scene during viewport interaction.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::core::{ObjectId, ObjectKind};
use crate::materials::Material;
use crate::scene::Scene;

use super::settings::RaytraceSettings;

/// Fingerprint of a scene + settings slice that affects flattening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneFingerprint(u64);

impl SceneFingerprint {
    /// Hash visible geometry, transforms, materials, lights, fog, and settings
    /// that are baked into the flattened scene.
    pub fn of(scene: &Scene, settings: &RaytraceSettings) -> Self {
        let mut h = DefaultHasher::new();
        settings.environment_intensity.to_bits().hash(&mut h);
        (settings.background as u8).hash(&mut h);
        settings.sun_angular_radius.to_bits().hash(&mut h);
        settings.light_radius.to_bits().hash(&mut h);
        settings.light_layers.mask.hash(&mut h);
        scene.background.r.to_bits().hash(&mut h);
        scene.background.g.to_bits().hash(&mut h);
        scene.background.b.to_bits().hash(&mut h);
        scene.background_alpha.to_bits().hash(&mut h);
        scene.fog.color.r.to_bits().hash(&mut h);
        scene.fog.color.g.to_bits().hash(&mut h);
        scene.fog.color.b.to_bits().hash(&mut h);
        scene.fog.near.to_bits().hash(&mut h);
        scene.fog.far.to_bits().hash(&mut h);
        scene.fog.density.to_bits().hash(&mut h);
        scene.fog.mode.hash(&mut h);
        hash_opt_cube(&mut h, &scene.environment);
        hash_object(scene, scene.root, &mut h);
        Self(h.finish())
    }
}

fn hash_opt_cube(h: &mut DefaultHasher, env: &Option<Arc<crate::textures::CubeTexture>>) {
    match env {
        Some(c) => {
            Arc::as_ptr(c).hash(h);
            c.size.hash(h);
        }
        None => 0u8.hash(h),
    }
}

fn hash_object(scene: &Scene, id: ObjectId, h: &mut DefaultHasher) {
    let Some(obj) = scene.arena.get(id) else {
        return;
    };
    if !obj.visible {
        return;
    }
    obj.name.hash(h);
    obj.matrix_world.elements.iter().for_each(|v| v.to_bits().hash(h));
    obj.layers.mask.hash(h);
    match &obj.kind {
        ObjectKind::Mesh(m) => {
            hash_geometry(&m.geometry, h);
            hash_material(&m.material, h);
        }
        ObjectKind::SkinnedMesh(sm) => {
            hash_geometry(&sm.geometry, h);
            hash_material(&sm.material, h);
            sm.skeleton.bones.len().hash(h);
        }
        ObjectKind::InstancedMesh(im) => {
            hash_geometry(&im.geometry, h);
            hash_material(&im.material, h);
            im.transforms.len().hash(h);
            for t in &im.transforms {
                t.elements.iter().for_each(|v| v.to_bits().hash(h));
            }
        }
        ObjectKind::LineSegments(ls) => {
            hash_geometry(&ls.geometry, h);
            hash_material(&ls.material, h);
        }
        ObjectKind::Points(p) => {
            hash_geometry(&p.geometry, h);
            hash_material(&p.material, h);
        }
        ObjectKind::Sprite(s) => {
            hash_material(&s.material, h);
        }
        ObjectKind::Light(l) => {
            std::mem::discriminant(l).hash(h);
            match l {
                crate::lights::Light::Ambient(a) => {
                    a.intensity.to_bits().hash(h);
                    a.color.r.to_bits().hash(h);
                }
                crate::lights::Light::Hemisphere(hl) => {
                    hl.intensity.to_bits().hash(h);
                }
                crate::lights::Light::Directional(d) => {
                    d.intensity.to_bits().hash(h);
                    d.color.r.to_bits().hash(h);
                }
                crate::lights::Light::Point(p) => {
                    p.intensity.to_bits().hash(h);
                    p.distance.to_bits().hash(h);
                }
                crate::lights::Light::Spot(s) => {
                    s.intensity.to_bits().hash(h);
                    s.angle.to_bits().hash(h);
                }
                crate::lights::Light::RectArea(r) => {
                    r.intensity.to_bits().hash(h);
                    r.width.to_bits().hash(h);
                    r.height.to_bits().hash(h);
                }
            }
            obj.layers.mask.hash(h);
        }
        ObjectKind::Group => {}
    }
    for child in scene.arena.get(id).into_iter().flat_map(|o| o.children.clone()) {
        hash_object(scene, child, h);
    }
}

fn hash_geometry(g: &Arc<crate::core::BufferGeometry>, h: &mut DefaultHasher) {
    Arc::as_ptr(g).hash(h);
    if let Some(pos) = g.get_attribute("position") {
        pos.count().hash(h);
        pos.array.len().hash(h);
    }
    if let Some(idx) = &g.index {
        idx.len().hash(h);
    }
}

fn hash_material(m: &Arc<Material>, h: &mut DefaultHasher) {
    Arc::as_ptr(m).hash(h);
}
