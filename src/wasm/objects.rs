//! Bindings split out of `wasm.rs` by area (see the `mod` list there): geometries (extras), materials (extras), lights (extras), object3d + nested objects, buffer geometry + buffer attribute.

use super::*;

// ---- geometries (extras) ----
#[wasm_bindgen]
impl WebGeometry {
    #[wasm_bindgen(js_name = circle)]
    pub fn circle(radius: f32, segments: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::CircleGeometry::new(
                radius,
                segments.max(3),
                0.0,
                std::f32::consts::PI * 2.0,
            )),
        }
    }
    #[wasm_bindgen(js_name = ring)]
    pub fn ring(inner: f32, outer: f32, theta_segments: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::RingGeometry::new(
                inner,
                outer,
                theta_segments.max(3),
                1,
                0.0,
                std::f32::consts::PI * 2.0,
            )),
        }
    }
    #[wasm_bindgen(js_name = cone)]
    pub fn cone(radius: f32, height: f32, radial_segments: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::ConeGeometry::new(
                radius,
                height,
                radial_segments.max(3),
                1,
                false,
                0.0,
                std::f32::consts::PI * 2.0,
            )),
        }
    }
    #[wasm_bindgen(js_name = torusKnot)]
    pub fn torus_knot(
        radius: f32,
        tube: f32,
        tubular_segments: usize,
        radial_segments: usize,
        p: u32,
        q: u32,
    ) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::TorusKnotGeometry::new(
                radius,
                tube,
                tubular_segments.max(3),
                radial_segments.max(3),
                p,
                q,
            )),
        }
    }
    #[wasm_bindgen(js_name = capsule)]
    pub fn capsule(
        radius: f64,
        length: f64,
        cap_segments: usize,
        radial_segments: usize,
    ) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::CapsuleGeometry::with_f64(
                radius,
                length,
                cap_segments.max(1),
                radial_segments.max(3),
            )),
        }
    }
    #[wasm_bindgen(js_name = tetrahedron)]
    pub fn tetrahedron(radius: f32, detail: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::TetrahedronGeometry::new(radius, detail)),
        }
    }
    #[wasm_bindgen(js_name = octahedron)]
    pub fn octahedron(radius: f32, detail: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::OctahedronGeometry::new(radius, detail)),
        }
    }
    #[wasm_bindgen(js_name = icosahedron)]
    pub fn icosahedron(radius: f32, detail: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::IcosahedronGeometry::new(radius, detail)),
        }
    }
    #[wasm_bindgen(js_name = dodecahedron)]
    pub fn dodecahedron(radius: f32, detail: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::DodecahedronGeometry::new(radius, detail)),
        }
    }
    #[wasm_bindgen(js_name = boxLine)]
    pub fn box_line(w: f32, h: f32, d: f32) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::BoxLineGeometry::new(w, h, d)),
        }
    }

    /// Clone into a `WebBufferGeometry` handle (same underlying data).
    #[wasm_bindgen(js_name = toBufferGeometry)]
    pub fn to_buffer_geometry(&self) -> WebBufferGeometry {
        WebBufferGeometry {
            inner: self.inner.clone(),
        }
    }
}

// ---- materials (extras) ----
#[wasm_bindgen]
impl WebMaterial {
    #[wasm_bindgen(js_name = phong)]
    pub fn phong(color: &WebColor) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Phong(crate::PhongMaterial::new(
                color.inner,
            ))),
        }
    }
    #[wasm_bindgen(js_name = physical)]
    pub fn physical(
        color: &WebColor,
        roughness: f32,
        metalness: f32,
        clearcoat: f32,
        clearcoat_roughness: f32,
    ) -> WebMaterial {
        let mut m = crate::PhysicalMaterial::new(color.inner);
        m.roughness = roughness;
        m.metalness = metalness;
        m.clearcoat = clearcoat;
        m.clearcoat_roughness = clearcoat_roughness;
        WebMaterial {
            inner: Arc::new(crate::Material::Physical(m)),
        }
    }
    #[wasm_bindgen(js_name = normalMat)]
    pub fn normal_mat() -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Normal(crate::NormalMaterial::new())),
        }
    }
    #[wasm_bindgen(js_name = depth)]
    pub fn depth() -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Depth(crate::DepthMaterial::new())),
        }
    }
    #[wasm_bindgen(js_name = toon)]
    pub fn toon(color: &WebColor) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Toon(crate::ToonMaterial::new(color.inner))),
        }
    }
    #[wasm_bindgen(js_name = lineDashed)]
    pub fn line_dashed(color: &WebColor, scale: f32, dash_size: f32, gap_size: f32) -> WebMaterial {
        let mut m = crate::LineBasicMaterial::new(color.inner);
        m.dashed = true;
        m.dash_scale = scale;
        m.dash_size = dash_size;
        m.gap_size = gap_size;
        WebMaterial {
            inner: Arc::new(crate::Material::Line(m)),
        }
    }
    #[wasm_bindgen(js_name = line)]
    pub fn line(color: &WebColor) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Line(crate::LineBasicMaterial::new(
                color.inner,
            ))),
        }
    }
    #[wasm_bindgen(js_name = points)]
    pub fn points(color: &WebColor, size: f32) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Points(crate::PointsMaterial::new(
                color.inner,
                size,
            ))),
        }
    }
    #[wasm_bindgen(js_name = sprite)]
    pub fn sprite(color: &WebColor) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Sprite(crate::SpriteMaterial::new(
                color.inner,
            ))),
        }
    }
}

// ---- lights (extras) ----
#[wasm_bindgen]
impl WebLight {
    #[wasm_bindgen(js_name = point)]
    pub fn point(color: &WebColor, intensity: f32, distance: f32, decay: f32) -> WebLight {
        let mut p = crate::PointLight::new(color.inner, intensity);
        p.distance = distance;
        p.decay = decay;
        WebLight {
            inner: LightInner::Point(p),
        }
    }

    // Adjust intensity in-place (three.js .intensity = ...).
    #[wasm_bindgen(js_name = setIntensity)]
    pub fn set_intensity(&mut self, intensity: f32) {
        match &mut self.inner {
            LightInner::Ambient(l) => l.intensity = intensity,
            LightInner::Directional(l) => l.intensity = intensity,
            LightInner::Point(l) => l.intensity = intensity,
            LightInner::Spot(l) => l.intensity = intensity,
            LightInner::Hemisphere(l) => l.intensity = intensity,
            LightInner::RectArea(l) => l.intensity = intensity,
        }
    }

    /// Toggle whether this light casts shadows. Only Directional / Spot /
    /// Point variants honor it; every one that does gets a shadow map of its
    /// own.
    #[wasm_bindgen(js_name = setCastShadow)]
    pub fn set_cast_shadow(&mut self, cast: bool) {
        match &mut self.inner {
            LightInner::Directional(l) => l.cast_shadow = cast,
            LightInner::Spot(l) => l.cast_shadow = cast,
            LightInner::Point(l) => l.cast_shadow = cast,
            _ => {}
        }
    }

    /// Orthographic shadow frustum (three.js `light.shadow.camera.*`).
    #[wasm_bindgen(js_name = setShadowCamera)]
    pub fn set_shadow_camera(
        &mut self,
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
        near: f32,
        far: f32,
    ) {
        let size = ((right - left).abs().max((top - bottom).abs())) * 0.5;
        let settings = crate::ShadowSettings {
            camera_size: size.max(0.1),
            camera_near: near,
            camera_far: far,
            ..match &self.inner {
                LightInner::Directional(l) => l.shadow,
                LightInner::Spot(l) => l.shadow,
                LightInner::Point(l) => l.shadow,
                _ => crate::ShadowSettings::default(),
            }
        };
        match &mut self.inner {
            LightInner::Directional(l) => l.shadow = settings,
            LightInner::Spot(l) => l.shadow = settings,
            LightInner::Point(l) => l.shadow = settings,
            _ => {}
        }
    }
    /// The light's shadow settings as this renderer holds them:
    /// `[bias, normalBias, radius, mapSize]` — bias positive, the depth this
    /// renderer subtracts (see `setShadowParams`). Empty for a light that
    /// casts none.
    #[wasm_bindgen(getter, js_name = shadowSettings)]
    pub fn shadow_settings(&self) -> Vec<f32> {
        let shadow = match &self.inner {
            LightInner::Directional(l) => &l.shadow,
            LightInner::Spot(l) => &l.shadow,
            LightInner::Point(l) => &l.shadow,
            _ => return Vec::new(),
        };
        vec![
            shadow.bias,
            shadow.normal_bias,
            shadow.radius,
            shadow.map_size as f32,
        ]
    }

    /// Cascaded shadow maps for a directional light
    /// (`ShadowSettings::cascades`): `count` (1 to 4) maps over the camera's
    /// view out to `distance` (0: its far plane), split between even and
    /// logarithmic by `split` (0 to 1).
    #[wasm_bindgen(js_name = setShadowCascades)]
    pub fn set_shadow_cascades(&mut self, count: u32, distance: f32, split: f32) {
        if let LightInner::Directional(l) = &mut self.inner {
            l.shadow.cascades = count.clamp(1, 4);
            l.shadow.cascade_distance = distance.max(0.0);
            l.shadow.cascade_split = split.clamp(0.0, 1.0);
        }
    }

    /// This light's size for soft shadows (`ShadowSettings::light_size`) and
    /// its own filter, overriding the renderer's (three.js shadow-map type
    /// numbers; anything else to follow the renderer).
    #[wasm_bindgen(js_name = setShadowSoftness)]
    pub fn set_shadow_softness(&mut self, light_size: f32, filter: i32) {
        let shadow = match &mut self.inner {
            LightInner::Directional(l) => &mut l.shadow,
            LightInner::Spot(l) => &mut l.shadow,
            LightInner::Point(l) => &mut l.shadow,
            _ => return,
        };
        shadow.light_size = light_size.max(0.0);
        shadow.filter = (0..=3)
            .contains(&filter)
            .then(|| shadow_type_of(filter as u32));
    }

    /// three.js `light.shadow.mapSize`: the shadow map's side in texels (the
    /// larger of width and height; the maps are square), for directional and
    /// spot lights. A point light's cube faces are fixed at 512.
    #[wasm_bindgen(js_name = setShadowMapSize)]
    pub fn set_shadow_map_size(&mut self, width: u32, height: u32) {
        let size = width.max(height).max(1);
        match &mut self.inner {
            LightInner::Directional(l) => l.shadow.map_size = size,
            LightInner::Spot(l) => l.shadow.map_size = size,
            LightInner::Point(l) => l.shadow.map_size = size,
            _ => {}
        }
    }

    /// three.js `light.shadow.bias`, `.normalBias` and `.radius`: the depth
    /// offset, the offset along the surface normal, and how soft the edge is
    /// (the radius, in shadow-map texels, it is filtered over; 1 by default).
    #[wasm_bindgen(js_name = setShadowParams)]
    pub fn set_shadow_params(&mut self, bias: f32, normal_bias: f32, radius: f32) {
        let shadow = match &mut self.inner {
            LightInner::Directional(l) => &mut l.shadow,
            LightInner::Spot(l) => &mut l.shadow,
            LightInner::Point(l) => &mut l.shadow,
            _ => return,
        };
        // three.js adds its bias to the shadow coordinate, so the usual
        // value is negative; this renderer subtracts a positive one.
        shadow.bias = (-bias).max(0.0);
        shadow.normal_bias = normal_bias.max(0.0);
        shadow.radius = radius.max(0.0);
    }

    #[wasm_bindgen(js_name = spot)]
    pub fn spot(
        color: &WebColor,
        intensity: f32,
        distance: f32,
        angle: f32,
        penumbra: f32,
        decay: f32,
    ) -> WebLight {
        let mut s = crate::SpotLight::new(color.inner, intensity);
        s.distance = distance;
        s.angle = angle;
        s.penumbra = penumbra;
        s.decay = decay;
        WebLight {
            inner: LightInner::Spot(s),
        }
    }
    #[wasm_bindgen(js_name = hemisphere)]
    pub fn hemisphere(sky: &WebColor, ground: &WebColor, intensity: f32) -> WebLight {
        let h = crate::HemisphereLight::new(sky.inner, ground.inner, intensity);
        WebLight {
            inner: LightInner::Hemisphere(h),
        }
    }
    #[wasm_bindgen(js_name = rectArea)]
    pub fn rect_area(color: &WebColor, intensity: f32, width: f32, height: f32) -> WebLight {
        let r = crate::RectAreaLight::new(color.inner, intensity, width, height);
        WebLight {
            inner: LightInner::RectArea(r),
        }
    }
}

// ---- object3d + nested objects ----
/// Generic Object3D handle (Group or any scene-graph node).
#[wasm_bindgen]
pub struct WebObject3D {
    pub(crate) inner: Option<crate::core::Object3D>,
}

#[wasm_bindgen]
impl WebObject3D {
    /// Empty group — three.js `new Group()` / `new Object3D()`.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebObject3D {
        WebObject3D {
            inner: Some(crate::core::Object3D::group()),
        }
    }
}

#[wasm_bindgen]
impl WebScene {
    /// Add a generic Object3D (group, helper, etc.) to the scene.
    #[wasm_bindgen(js_name = addObject)]
    pub fn add_object(&mut self, obj: &mut WebObject3D) -> WebObjectHandle {
        let o = obj
            .inner
            .take()
            .unwrap_or_else(crate::core::Object3D::group);
        WebObjectHandle {
            id: self.inner.add(o),
        }
    }

    /// Remove an object from the scene (by handle).
    pub fn remove(&mut self, handle: &WebObjectHandle) {
        self.inner.arena.remove_child(handle.id);
    }

    /// Replace a mesh's geometry positions in place. Used for morph-target
    /// vertex blending and any dynamic vertex animation.
    #[wasm_bindgen(js_name = setMeshPositions)]
    pub fn set_mesh_positions(&mut self, handle: &WebObjectHandle, positions: Vec<f32>) {
        if let Some(obj) = self.inner.arena.get_mut(handle.id) {
            match &mut obj.kind {
                crate::core::ObjectKind::Mesh(mesh) => {
                    let mut new_geom = (*mesh.geometry).clone();
                    new_geom
                        .set_attribute("position", crate::core::BufferAttribute::new(positions, 3));
                    mesh.geometry = std::sync::Arc::new(new_geom);
                }
                crate::core::ObjectKind::SkinnedMesh(sm) => {
                    let mut new_geom = (*sm.geometry).clone();
                    new_geom
                        .set_attribute("position", crate::core::BufferAttribute::new(positions, 3));
                    sm.geometry = std::sync::Arc::new(new_geom);
                }
                _ => {}
            }
        }
    }

    /// Replace a mesh's geometry normals in place.
    #[wasm_bindgen(js_name = setMeshNormals)]
    pub fn set_mesh_normals(&mut self, handle: &WebObjectHandle, normals: Vec<f32>) {
        if let Some(obj) = self.inner.arena.get_mut(handle.id)
            && let crate::core::ObjectKind::Mesh(mesh) = &mut obj.kind
        {
            let mut new_geom = (*mesh.geometry).clone();
            new_geom.set_attribute("normal", crate::core::BufferAttribute::new(normals, 3));
            mesh.geometry = std::sync::Arc::new(new_geom);
        }
    }

    /// Update a skinned mesh's bone matrices. Each matrix is 16 floats
    /// (column-major); the caller passes the matrices concatenated.
    /// Re-uploaded to the storage buffer on the next render.
    #[wasm_bindgen(js_name = setBoneMatrices)]
    pub fn set_bone_matrices(&mut self, handle: &WebObjectHandle, matrices: Vec<f32>) {
        if let Some(obj) = self.inner.arena.get_mut(handle.id)
            && let crate::core::ObjectKind::SkinnedMesh(sm) = &mut obj.kind
        {
            let mut new_skel = (*sm.skeleton).clone();
            let mat_count = matrices.len() / 16;
            new_skel.bone_matrices.clear();
            for i in 0..mat_count {
                let mut e = [0.0_f32; 16];
                e.copy_from_slice(&matrices[i * 16..(i + 1) * 16]);
                new_skel
                    .bone_matrices
                    .push(crate::math::Matrix4 { elements: e });
            }
            sm.skeleton = std::sync::Arc::new(new_skel);
        }
    }

    /// Update a skinned mesh's per-vertex joint indices.
    #[wasm_bindgen(js_name = setSkinJoints)]
    pub fn set_skin_joints(&mut self, handle: &WebObjectHandle, joints: Vec<f32>) {
        if let Some(obj) = self.inner.arena.get_mut(handle.id)
            && let crate::core::ObjectKind::SkinnedMesh(sm) = &mut obj.kind
        {
            let mut new_geom = (*sm.geometry).clone();
            new_geom.set_attribute("joint", crate::core::BufferAttribute::new(joints, 4));
            sm.geometry = std::sync::Arc::new(new_geom);
        }
    }

    /// Update a skinned mesh's per-vertex bone weights.
    #[wasm_bindgen(js_name = setSkinWeights)]
    pub fn set_skin_weights(&mut self, handle: &WebObjectHandle, weights: Vec<f32>) {
        if let Some(obj) = self.inner.arena.get_mut(handle.id)
            && let crate::core::ObjectKind::SkinnedMesh(sm) = &mut obj.kind
        {
            let mut new_geom = (*sm.geometry).clone();
            new_geom.set_attribute("weight", crate::core::BufferAttribute::new(weights, 4));
            sm.geometry = std::sync::Arc::new(new_geom);
        }
    }

    /// The scene root's object id: `raycaster.intersectObject(scene,
    /// scene.rootId, true)` tests everything in the scene.
    #[wasm_bindgen(getter, js_name = rootId)]
    pub fn root_id(&self) -> f64 {
        object_id_to_f64(self.inner.root)
    }

    /// The handle for an object id (`WebObjectHandle.id`, a hit's
    /// `objectId`), or undefined when no such object is in the scene.
    #[wasm_bindgen(js_name = handleOf)]
    pub fn handle_of(&self, id: f64) -> Option<WebObjectHandle> {
        let id = object_id_from_f64(id)?;
        self.inner.arena.get(id).map(|_| WebObjectHandle { id })
    }

    /// Find the first object whose name matches.
    #[wasm_bindgen(js_name = getObjectByName)]
    pub fn get_object_by_name(&self, name: &str) -> Option<WebObjectHandle> {
        let root = self.inner.root;
        self.inner
            .arena
            .get_object_by_name(root, name)
            .map(|id| WebObjectHandle { id })
    }

    /// Set a name on the object identified by `handle`.
    #[wasm_bindgen(js_name = setName)]
    pub fn set_name(&mut self, handle: &WebObjectHandle, name: &str) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.name = name.into();
        }
    }

    /// Apply a translation to the object's local position.
    pub fn translate(&mut self, handle: &WebObjectHandle, dx: f32, dy: f32, dz: f32) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.position = obj.position + crate::Vector3::new(dx, dy, dz);
        }
    }

    /// Set scale.
    #[wasm_bindgen(js_name = setScale)]
    pub fn set_scale(&mut self, handle: &WebObjectHandle, sx: f32, sy: f32, sz: f32) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.scale = crate::Vector3::new(sx, sy, sz);
        }
    }

    /// Set visibility.
    #[wasm_bindgen(js_name = setVisible)]
    pub fn set_visible(&mut self, handle: &WebObjectHandle, visible: bool) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.visible = visible;
        }
    }

    /// three.js Object3D.renderOrder — lower draws first within transparency class.
    #[wasm_bindgen(js_name = setRenderOrder)]
    pub fn set_render_order(&mut self, handle: &WebObjectHandle, order: i32) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.render_order = order;
        }
    }

    /// three.js `lookAt`: a light turns its -Z to the world-space target,
    /// anything else its +Z, with the parent's rotation divided out.
    #[wasm_bindgen(js_name = lookAt)]
    pub fn look_at(&mut self, handle: &WebObjectHandle, x: f32, y: f32, z: f32) {
        self.inner
            .arena
            .look_at(handle.id, crate::Vector3::new(x, y, z));
    }

    /// Get the world-space position (after parent transforms).
    #[wasm_bindgen(js_name = getWorldPosition)]
    pub fn get_world_position(&mut self, handle: &WebObjectHandle) -> WebVector3 {
        self.inner.update_world();
        let p = self
            .inner
            .get(handle.id)
            .map(|o| o.world_position())
            .unwrap_or(crate::Vector3::ZERO);
        WebVector3 { inner: p }
    }
}

// ---- buffer geometry + buffer attribute ----
#[wasm_bindgen]
pub struct WebBufferAttribute {
    pub(crate) inner: crate::BufferAttribute,
}

#[wasm_bindgen]
impl WebBufferAttribute {
    #[wasm_bindgen(constructor)]
    pub fn new(array: Vec<f32>, item_size: usize) -> WebBufferAttribute {
        WebBufferAttribute {
            inner: crate::BufferAttribute::new(array, item_size),
        }
    }
    pub fn count(&self) -> usize {
        self.inner.count()
    }
    #[wasm_bindgen(js_name = itemSize, getter)]
    pub fn item_size(&self) -> usize {
        self.inner.item_size
    }
    pub fn array(&self) -> Vec<f32> {
        self.inner.array.clone()
    }
}

#[wasm_bindgen]
pub struct WebBufferGeometry {
    pub(crate) inner: Arc<crate::BufferGeometry>,
}

#[wasm_bindgen]
impl WebBufferGeometry {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebBufferGeometry {
        WebBufferGeometry {
            inner: Arc::new(crate::BufferGeometry::new()),
        }
    }
    #[wasm_bindgen(js_name = setAttribute)]
    pub fn set_attribute(&mut self, name: &str, attr: WebBufferAttribute) {
        Arc::make_mut(&mut self.inner).set_attribute(name, attr.inner);
    }
    #[wasm_bindgen(js_name = setIndex)]
    pub fn set_index(&mut self, indices: Vec<u32>) {
        Arc::make_mut(&mut self.inner).set_index(indices);
    }
    #[wasm_bindgen(js_name = computeBoundingBox)]
    pub fn compute_bounding_box(&mut self) -> WebBox3 {
        WebBox3 {
            inner: Arc::make_mut(&mut self.inner).compute_bounding_box(),
        }
    }
    #[wasm_bindgen(js_name = computeBoundingSphere)]
    pub fn compute_bounding_sphere(&mut self) -> WebSphere {
        WebSphere {
            inner: Arc::make_mut(&mut self.inner).compute_bounding_sphere(),
        }
    }
    #[wasm_bindgen(js_name = computeVertexNormals)]
    pub fn compute_vertex_normals(&mut self) {
        crate::compute_vertex_normals(Arc::make_mut(&mut self.inner));
    }
    #[wasm_bindgen(js_name = hasAttribute)]
    pub fn has_attribute(&self, name: &str) -> bool {
        self.inner.get_attribute(name).is_some()
    }
    #[wasm_bindgen(js_name = drawCount)]
    pub fn draw_count(&self) -> usize {
        self.inner.draw_count()
    }

    /// Group triangles into polygons: `counts[i]` consecutive triangles, in
    /// index order, came from source polygon `i` (a quad is `2`). Wireframes
    /// then outline the polygons instead of drawing the fan diagonals. See
    /// `BufferGeometry::polygons`.
    ///
    /// Call after the last `setIndex` / `position` write — both drop the
    /// grouping. Throws unless every count is at least one and they sum to the
    /// geometry's triangle count; the geometry is left ungrouped then.
    #[wasm_bindgen(js_name = setPolygons)]
    pub fn set_polygons(&mut self, counts: Vec<u32>) -> Result<(), JsValue> {
        let triangles = self.inner.triangle_count();
        let sum: u64 = counts.iter().map(|&n| n as u64).sum();
        let geometry = Arc::make_mut(&mut self.inner);
        if let Some(i) = counts.iter().position(|&n| n == 0) {
            geometry.polygons = None;
            return Err(JsValue::from_str(&format!(
                "setPolygons: polygon {i} has zero triangles"
            )));
        }
        if sum != triangles as u64 {
            geometry.polygons = None;
            return Err(JsValue::from_str(&format!(
                "setPolygons: counts sum to {sum} triangles but the geometry draws {triangles}"
            )));
        }
        geometry.set_polygons(counts);
        Ok(())
    }

    /// Triangles per polygon, or `undefined` when there is no grouping (or
    /// the one set no longer covers the triangles).
    #[wasm_bindgen(js_name = polygonCounts)]
    pub fn polygon_counts(&self) -> Option<Vec<u32>> {
        self.inner.polygon_counts().map(|p| p.to_vec())
    }

    /// threers extension: pair adjacent triangles back into quads (Blender's
    /// "Tris to Quads") — for glTF/STL imports, which carry triangles only.
    /// Reorders the index so each quad's triangles are consecutive and sets
    /// the polygon grouping; no vertex changes. Angles in radians, both
    /// defaulting to 40°. Returns how many quads were formed. See
    /// `BufferGeometry::infer_quads`.
    #[wasm_bindgen(js_name = inferQuads)]
    pub fn infer_quads(
        &mut self,
        max_face_angle: Option<f32>,
        max_shape_angle: Option<f32>,
    ) -> u32 {
        let d = crate::core::DEFAULT_INFER_QUADS_ANGLE;
        Arc::make_mut(&mut self.inner)
            .infer_quads(max_face_angle.unwrap_or(d), max_shape_angle.unwrap_or(d)) as u32
    }

    /// threers extension: Catmull–Clark subdivision, `levels` rounds (default
    /// 1, at most 6 — each multiplies the face count by about 4). Returns a
    /// new geometry: indexed, every face a quad grouped as a polygon, with
    /// `position` and smooth `normal` only (UVs and other attributes are
    /// dropped). The cage is this geometry's polygons — its triangles where
    /// there is no grouping — welded by position. `sharpBoundary` (default
    /// true) holds open borders in place. See `BufferGeometry::subdivide_with`.
    #[wasm_bindgen(js_name = subdivide)]
    pub fn subdivide(
        &self,
        levels: Option<u32>,
        sharp_boundary: Option<bool>,
    ) -> Result<WebBufferGeometry, JsValue> {
        const MAX_LEVELS: u32 = 6;
        let levels = levels.unwrap_or(1);
        if levels > MAX_LEVELS {
            return Err(JsValue::from_str(&format!(
                "subdivide: levels must be 0..={MAX_LEVELS}, got {levels}"
            )));
        }
        let options = crate::geometries::SubdivisionOptions {
            sharp_boundary: sharp_boundary.unwrap_or(true),
            ..Default::default()
        };
        Ok(WebBufferGeometry {
            inner: Arc::new(self.inner.subdivide_with(levels as usize, &options)),
        })
    }

    /// The faces a polygon-capable writer emits (`BufferGeometry::faces`),
    /// flattened as `[n, v0 … v(n-1), n, …]`: one corner loop per polygon whose
    /// outline is recoverable, its triangles one by one otherwise, and every
    /// triangle on its own without a grouping. Corners are vertex indices.
    #[wasm_bindgen(js_name = faces)]
    pub fn faces(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.inner.triangle_count() * 4);
        for face in self.inner.faces() {
            out.push(face.len() as u32);
            out.extend_from_slice(&face);
        }
        out
    }
}

// Readback accessors. Not feature-gated: the shim's `QuadGeometry` and
// `computeVertexNormals` read generated buffers back in every build, not only
// in builds with `mesh-bvh`.
#[wasm_bindgen]
impl WebBufferGeometry {
    #[wasm_bindgen(js_name = getAttributeArray)]
    pub fn get_attribute_array(&self, name: &str) -> Option<Vec<f32>> {
        self.inner.get_attribute(name).map(|a| a.array.clone())
    }

    #[wasm_bindgen(js_name = getAttributeItemSize)]
    pub fn get_attribute_item_size(&self, name: &str) -> Option<usize> {
        self.inner.get_attribute(name).map(|a| a.item_size)
    }

    #[wasm_bindgen(js_name = getIndexArray)]
    pub fn get_index_array(&self) -> Option<Vec<u32>> {
        self.inner.index.clone()
    }

    /// Every attribute's name, sorted, so a reader can copy all of them.
    #[wasm_bindgen(js_name = attributeNames)]
    pub fn attribute_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.inner.attributes.keys().cloned().collect();
        names.sort();
        names
    }

    /// three.js `groups`, flattened as `[start, count, materialIndex, …]`.
    #[wasm_bindgen(js_name = groupsArray)]
    pub fn groups_array(&self) -> Vec<u32> {
        self.inner
            .groups
            .iter()
            .flat_map(|g| [g.start, g.count, g.material_index])
            .collect()
    }
}

// Convert a user-built BufferGeometry into a WebGeometry handle.
#[wasm_bindgen]
impl WebGeometry {
    #[wasm_bindgen(js_name = fromBufferGeometry)]
    pub fn from_buffer_geometry(g: WebBufferGeometry) -> WebGeometry {
        WebGeometry { inner: g.inner }
    }
}
