//! Objects that flicker at a given frequency (see [`Object3D::flicker`]).
//!
//! [`Scene::apply_flicker`](crate::Scene::apply_flicker) writes each
//! flickering object's level into what the renderers already read: hidden
//! at 0, its materials' opacity (made transparent) and its light's
//! intensity scaled in between. What it overwrites is kept here, and put
//! back once the object is fully shown again, so a square wave between 0
//! and 1 only ever touches `visible`.

use super::{Object3D, ObjectKind};
use crate::materials::Material;
use std::sync::Arc;

/// What a flicker took over on an object, to give back. The materials are
/// kept beside it, in [`Object3D::flicker_materials`]: nested in here they
/// put `Material` deep enough in `Object3D` to overflow the `Sync` check of
/// the types that hold a scene.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Saved {
    visible: bool,
    intensity: Option<f32>,
}

/// Every material `kind` draws with.
fn materials_mut(kind: &mut ObjectKind) -> Vec<&mut Arc<Material>> {
    match kind {
        ObjectKind::Mesh(m) => std::iter::once(&mut m.material)
            .chain(&mut m.materials)
            .collect(),
        ObjectKind::SkinnedMesh(m) => std::iter::once(&mut m.material)
            .chain(&mut m.materials)
            .collect(),
        ObjectKind::InstancedMesh(m) => std::iter::once(&mut m.material)
            .chain(&mut m.materials)
            .collect(),
        ObjectKind::LineSegments(l) => vec![&mut l.material],
        ObjectKind::Points(p) => vec![&mut p.material],
        ObjectKind::Sprite(s) => vec![&mut s.material],
        _ => vec![],
    }
}

impl Object3D {
    /// Shows this object at `level` (0..=1) of itself, as a flicker does.
    pub(crate) fn show_at(&mut self, level: f32) {
        if level >= 1.0 {
            if let Some(saved) = self.flicker_saved.take() {
                self.visible = saved.visible;
                let originals = std::mem::take(&mut self.flicker_materials);
                for (m, original) in materials_mut(&mut self.kind).into_iter().zip(originals) {
                    *m = original;
                }
                if let (Some(i), ObjectKind::Light(l)) = (saved.intensity, &mut self.kind) {
                    l.set_intensity(i);
                }
            }
            return;
        }
        let visible = self.visible;
        let saved = self.flicker_saved.get_or_insert(Saved {
            visible,
            intensity: None,
        });
        self.visible = saved.visible && level > 0.0;
        if level <= 0.0 {
            return;
        }
        let mut current = materials_mut(&mut self.kind);
        if self.flicker_materials.is_empty() {
            self.flicker_materials = current.iter().map(|m| Arc::clone(m)).collect();
        }
        for (m, original) in current.iter_mut().zip(&self.flicker_materials) {
            // A copy the first time, then the same copy written again.
            let copy = Arc::make_mut(m);
            copy.set_opacity(original.opacity() * level);
            copy.set_transparent(true);
        }
        if let ObjectKind::Light(l) = &mut self.kind {
            let full = *saved.intensity.get_or_insert(l.intensity());
            l.set_intensity(full * level);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{BufferGeometry, Mesh, Object3D, ObjectKind};
    use crate::lights::PointLight;
    use crate::materials::{BasicMaterial, Material};
    use crate::math::flicker::Flicker;
    use crate::math::Color;
    use crate::Scene;
    use std::sync::Arc;

    fn material(scene: &Scene, id: crate::ObjectId) -> Arc<Material> {
        match &scene.get(id).unwrap().kind {
            ObjectKind::Mesh(m) => m.material.clone(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn a_square_flicker_toggles_visibility_and_nothing_else() {
        let mut scene = Scene::new();
        let mat = Arc::new(Material::Basic(BasicMaterial::default()));
        let id = scene.add(Object3D::mesh(Mesh::from_arc(
            Arc::new(BufferGeometry::default()),
            mat.clone(),
        )));
        scene.get_mut(id).unwrap().flicker = Some(Flicker::square(10.0));
        scene.apply_flicker(0.01);
        assert!(scene.get(id).unwrap().visible);
        scene.apply_flicker(0.06);
        assert!(!scene.get(id).unwrap().visible);
        assert!(
            Arc::ptr_eq(&material(&scene, id), &mat),
            "the material is untouched"
        );
        scene.apply_flicker(0.11);
        assert!(scene.get(id).unwrap().visible);
        // Stopped while off: shown again at the next apply.
        scene.apply_flicker(0.06);
        scene.get_mut(id).unwrap().flicker = None;
        scene.apply_flicker(0.06);
        assert!(scene.get(id).unwrap().visible);
    }

    #[test]
    fn a_sine_flicker_scales_opacity_down_the_tree_and_gives_it_back() {
        let mut scene = Scene::new();
        let group = scene.add(Object3D::group());
        let mat = Arc::new(Material::Basic(BasicMaterial {
            opacity: 0.8,
            ..Default::default()
        }));
        let mesh = scene.add_to(
            group,
            Object3D::mesh(Mesh::from_arc(
                Arc::new(BufferGeometry::default()),
                mat.clone(),
            )),
        );
        let light = scene.add_to(group, Object3D::light(PointLight::new(Color::WHITE, 2.0)));
        scene.get_mut(group).unwrap().flicker = Some(Flicker::sine(1.0));
        scene.apply_flicker(0.25); // half way down
        let m = material(&scene, mesh);
        assert!(
            !Arc::ptr_eq(&m, &mat),
            "a copy, so other users of it stay as they are"
        );
        assert!((m.opacity() - 0.4).abs() < 1e-5 && m.transparent());
        assert_eq!(mat.opacity(), 0.8);
        let ObjectKind::Light(l) = &scene.get(light).unwrap().kind else {
            unreachable!()
        };
        assert!((l.intensity() - 1.0).abs() < 1e-5);
        // From the original each frame, not compounding.
        scene.apply_flicker(0.25);
        assert!((material(&scene, mesh).opacity() - 0.4).abs() < 1e-5);
        // Fully shown: the originals back.
        scene.apply_flicker(1.0);
        assert!(Arc::ptr_eq(&material(&scene, mesh), &mat));
        let ObjectKind::Light(l) = &scene.get(light).unwrap().kind else {
            unreachable!()
        };
        assert_eq!(l.intensity(), 2.0);
    }

    #[test]
    fn a_hidden_object_stays_hidden() {
        let mut scene = Scene::new();
        let id = scene.add(Object3D::group());
        let o = scene.get_mut(id).unwrap();
        o.visible = false;
        o.flicker = Some(Flicker::square(10.0));
        scene.apply_flicker(0.01);
        assert!(!scene.get(id).unwrap().visible);
        scene.apply_flicker(0.06);
        scene.apply_flicker(0.1);
        assert!(!scene.get(id).unwrap().visible);
    }
}
