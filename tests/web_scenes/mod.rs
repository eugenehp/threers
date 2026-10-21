//! Whole scenes in a real browser, shared by the WebGL2 suite
//! (`tests/webgl.rs`) and the WebGPU one (`tests/webgpu.rs`): lit, skinned and
//! refracting draws, a textured, shadowed, transparent, post-processed scene, a
//! `ShaderMaterial` with a storage array, and morph targets with geometry
//! groups. Each was written for a WebGL2 limit — sixteen texture units, no
//! storage buffers — and runs on WebGPU too, where Tint is the compiler that
//! has to accept the same WGSL.
//!
//! Each including file supplies `super::views_renderer()`.

use wasm_bindgen_test::wasm_bindgen_test;

/// Put the shared renderer back to its defaults: other tests leave viewports,
/// scissors and targets behind.
fn reset(renderer: &mut threers::wasm::WebRenderer) {
    renderer.set_render_target(0);
    renderer.reset_viewport();
    renderer.set_scissor_test(false);
    renderer.set_auto_clear(true);
    renderer.set_pixel_ratio(1.0);
}

/// Pull a render target back to the CPU, awaiting the promise it returns.
async fn read_target(
    renderer: &threers::wasm::WebRenderer,
    target_id: u32,
    w: u32,
    h: u32,
) -> Vec<u8> {
    let promise = renderer.read_render_target_pixels(target_id, 0, 0, w, h);
    let value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("read-back rejected");
    js_sys::Uint8Array::new(&value).to_vec()
}

/// The renderer draws — lit, skinned, and refracting — and the pixels prove
/// each one.
///
/// Every test here shares the suite's one renderer deliberately. A browser
/// caps how many live WebGL2 contexts a page may hold and does not release one
/// when its canvas goes out of scope, so a renderer per test exhausted the
/// pool and whichever ran last hung waiting for a context that was never
/// coming. Each draw still gets its own render target, which costs nothing.
///
/// The assertions are on read-back pixels rather than on the absence of a panic:
/// a draw that binds nothing and writes nothing raises no error at all.
#[wasm_bindgen_test]
async fn the_renderer_draws_lit_skinned_and_refracting() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);

    let white = threers::wasm::WebColor::new(1.0, 1.0, 1.0);
    let sun = || {
        let mut l = threers::wasm::WebLight::directional(&white, 2.5);
        l.set_direction(-0.4, -0.7, -0.6);
        l
    };
    let camera = || {
        let mut c = threers::wasm::WebCamera::perspective(60.0, 1.0, 0.1, 100.0);
        c.set_position(0.0, 0.0, 4.0);
        c
    };

    // --- A lit standard material. ---
    //
    // Getting here needed the texture budget brought under WebGL2's sixteen
    // units. The symptom was not a validation error but a panic inside the
    // backend, `index out of bounds: the len is 16 but the index is 16`, on a
    // draw that sampled none of the textures that overflowed it.
    {
        let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
        let mut scene = threers::wasm::WebScene::new();
        let colour = threers::wasm::WebColor::new(0.9, 0.3, 0.1);
        let material = threers::wasm::WebMaterial::standard(&colour, 0.4, 0.0);
        let geometry = threers::wasm::WebGeometry::box_(1.5, 1.5, 1.5);
        scene.add(&threers::wasm::WebMesh::new(&geometry, &material));
        scene.add_light(&sun());
        scene.add_light(&threers::wasm::WebLight::ambient(&white, 0.4));

        renderer.set_render_target(target.id);
        renderer.render(&mut scene, &camera());
        renderer.render(&mut scene, &camera());
        let pixels = read_target(&renderer, target.id, 64, 64).await;
        assert_eq!(pixels.len(), 64 * 64 * 4, "short read-back");
        let lit = pixels
            .chunks(4)
            .filter(|p| p[0] > 40 && p[0] > p[2])
            .count();
        assert!(
            lit > 50,
            "only {lit} of {} pixels show the lit cube",
            pixels.len() / 4
        );
    }

    // --- A skinned material. ---
    //
    // Skinning used to put its bone matrices in a storage buffer, which WebGL2
    // does not have — and one `var<storage>` declaration stops the whole module
    // compiling on a GLSL backend whether or not anything skinned is drawn. The
    // bones are identities here; the point is that the skinned pipeline binds
    // and runs, not that it deforms.
    {
        let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
        let mut scene = threers::wasm::WebScene::new();
        let colour = threers::wasm::WebColor::new(0.2, 0.8, 0.3);
        let material = threers::wasm::WebMaterial::standard(&colour, 0.4, 0.0);
        let geometry = threers::wasm::WebGeometry::box_(1.5, 1.5, 1.5);
        scene.add_skinned_mesh(&geometry, &material, 4);
        scene.add_light(&sun());
        scene.add_light(&threers::wasm::WebLight::ambient(&white, 0.4));

        renderer.set_render_target(target.id);
        renderer.render(&mut scene, &camera());
        renderer.render(&mut scene, &camera());
        let pixels = read_target(&renderer, target.id, 64, 64).await;
        let lit = pixels
            .chunks(4)
            .filter(|p| p[1] > 40 && p[1] > p[0])
            .count();
        assert!(
            lit > 50,
            "only {lit} of {} pixels show the skinned cube",
            pixels.len() / 4
        );
    }

    // --- Screen-space refraction. ---
    //
    // The pass that could not be built here until the bind groups were split.
    // wgpu-hal's GL backend assigns a texture unit per *layout* entry rather
    // than per use, so one shared group 3 carrying both the shadow maps and the
    // screen-space capture came to seventeen against sixteen. The refraction
    // pass samples no shadow map, so it now has a group 3 with the capture where
    // those were, and both layouts come to fourteen.
    //
    // An opaque plane stands behind the glass so there is something to refract.
    {
        let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
        let mut scene = threers::wasm::WebScene::new();
        let backdrop_colour = threers::wasm::WebColor::new(0.1, 0.2, 0.9);
        let backdrop = threers::wasm::WebMaterial::standard(&backdrop_colour, 0.5, 0.0);
        let plane = threers::wasm::WebGeometry::box_(6.0, 6.0, 0.2);
        let back = scene.add(&threers::wasm::WebMesh::new(&plane, &backdrop));
        scene.set_transform(
            &back,
            &threers::wasm::WebVector3::new(0.0, 0.0, -2.0),
            &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
        );

        let glass_colour = threers::wasm::WebColor::new(1.0, 1.0, 1.0);
        let mut glass = threers::wasm::WebMaterial::physical(&glass_colour, 0.05, 0.0, 0.0, 0.0);
        glass.set_transmission(1.0, 1.5, 0.5, 0.0);
        let sphere = threers::wasm::WebGeometry::sphere(1.0, 24, 16);
        scene.add(&threers::wasm::WebMesh::new(&sphere, &glass));
        scene.add_light(&sun());
        scene.add_light(&threers::wasm::WebLight::ambient(&white, 0.5));

        renderer.set_render_target(target.id);
        renderer.render(&mut scene, &camera());
        renderer.render(&mut scene, &camera());
        let pixels = read_target(&renderer, target.id, 64, 64).await;
        let blue = pixels
            .chunks(4)
            .filter(|p| p[2] > 60 && p[2] > p[0])
            .count();
        assert!(
            blue > 500,
            "only {blue} of {} pixels are backdrop-blue — nothing drew",
            pixels.len() / 4
        );
        // A pass that failed to run would leave a flat backdrop behind an unlit
        // sphere; what refraction adds is variation *through* the glass.
        let distinct: std::collections::BTreeSet<(u8, u8, u8)> =
            pixels.chunks(4).map(|p| (p[0], p[1], p[2])).collect();
        assert!(
            distinct.len() > 20,
            "only {} distinct colours — the frame is a flat fill, so the glass \
             refracted nothing",
            distinct.len()
        );
    }
}

/// A scene with a texture, a shadow and post-processing draws.
///
/// The draws above are deliberately plain, which means they exercise one
/// pipeline each. This one is the opposite: a textured material, a shadow-casting
/// light (so the shadow pipelines and the shadow map are built and sampled), a
/// transparent surface (so the sorted path runs), and a post-processing pass
/// over the result. It is here to find what the narrow tests cannot — anything
/// still assuming a capability the backend lacks (WebGL2 lacks many) shows up as a validation error or a
/// blank frame rather than as a wrong colour.
#[wasm_bindgen_test]
async fn a_full_scene_draws() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let scene_rt = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
    let post_rt = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);

    let mut scene = threers::wasm::WebScene::new();

    // A checked texture, so a sampler that reads the wrong thing is visible.
    let mut data = vec![0u8; 8 * 8 * 4];
    for y in 0..8usize {
        for x in 0..8usize {
            let on = (x / 2 + y / 2) % 2 == 0;
            let i = (y * 8 + x) * 4;
            data[i] = if on { 230 } else { 30 };
            data[i + 1] = if on { 120 } else { 20 };
            data[i + 2] = 40;
            data[i + 3] = 255;
        }
    }
    let texture = threers::wasm::WebDataTexture::new(8, 8, data);
    let white = threers::wasm::WebColor::new(1.0, 1.0, 1.0);
    let mut textured = threers::wasm::WebMaterial::standard(&white, 0.5, 0.0);
    textured.set_map_data(&texture);

    // A floor to receive the shadow, and a box above it to cast one.
    let floor = scene.add(&threers::wasm::WebMesh::new(
        &threers::wasm::WebGeometry::box_(6.0, 0.2, 6.0),
        &textured,
    ));
    scene.set_transform(
        &floor,
        &threers::wasm::WebVector3::new(0.0, -1.5, 0.0),
        &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
    );
    let caster_colour = threers::wasm::WebColor::new(0.3, 0.6, 0.9);
    let caster = scene.add(&threers::wasm::WebMesh::new(
        &threers::wasm::WebGeometry::box_(1.2, 1.2, 1.2),
        &threers::wasm::WebMaterial::standard(&caster_colour, 0.3, 0.0),
    ));
    scene.set_object_cast_shadow(&caster, true);

    // And a transparent pane in front, which takes the sorted path.
    let glass_colour = threers::wasm::WebColor::new(0.9, 0.9, 1.0);
    let mut pane = threers::wasm::WebMaterial::standard(&glass_colour, 0.1, 0.0);
    pane.set_opacity(0.4);
    let front = scene.add(&threers::wasm::WebMesh::new(
        &threers::wasm::WebGeometry::box_(2.0, 2.0, 0.05),
        &pane,
    ));
    scene.set_transform(
        &front,
        &threers::wasm::WebVector3::new(0.0, 0.0, 1.2),
        &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
    );

    let mut sun = threers::wasm::WebLight::directional(&white, 2.2);
    sun.set_direction(-0.3, -0.9, -0.4);
    scene.add_light(&sun);
    scene.add_light(&threers::wasm::WebLight::ambient(&white, 0.35));

    let mut camera = threers::wasm::WebCamera::perspective(55.0, 1.0, 0.1, 100.0);
    camera.set_position(0.0, 1.5, 5.0);

    renderer.set_render_target(scene_rt.id);
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);

    let before = read_target(&renderer, scene_rt.id, 64, 64).await;
    let distinct_before: std::collections::BTreeSet<(u8, u8, u8)> =
        before.chunks(4).map(|p| (p[0], p[1], p[2])).collect();
    assert!(
        distinct_before.len() > 30,
        "the scene came back with only {} distinct colours — a textured, lit, \
         shadowed frame cannot be that flat",
        distinct_before.len()
    );

    // Post-processing over the rendered frame. Kind 0 is the plain copy, which
    // is enough to prove the pass compiles and runs on this backend; the depth
    // and normal inputs reuse the scene target because this is about whether the
    // pipeline runs, not about what a particular effect looks like.
    let identity: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    renderer.apply_post_fx_to_rt(
        scene_rt.id,
        post_rt.id,
        scene_rt.id,
        scene_rt.id,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        100.0,
        0.5,
        16,
        identity.clone(),
        identity,
    );
    let after = read_target(&renderer, post_rt.id, 64, 64).await;
    assert_eq!(after.len(), before.len());
    let lit_after = after
        .chunks(4)
        .filter(|p| p[0] > 8 || p[1] > 8 || p[2] > 8)
        .count();
    assert!(
        lit_after > 500,
        "only {lit_after} of {} pixels survived post-processing — the pass ran \
         but produced nothing",
        after.len() / 4
    );
}

/// A custom shader, through the JavaScript API, in a browser.
///
/// `ShaderMaterial` had no counterpart in the wasm bindings at all, so every
/// custom shader in the crate was reachable from Rust and nowhere else — and
/// the shim's `THREE.ShaderMaterial` was a stub that quietly built a *basic*
/// material and rendered a flat colour, which is worse than the class being
/// absent.
///
/// The storage array is the part worth checking here rather than on a desktop:
/// WebGL2 has no storage buffers, so `s0(i)` has to resolve to a texture read.
/// A shader written with `u_s0[i]` does not compile on this backend at all,
/// which is exactly the failure this is meant to catch before a user does.
#[wasm_bindgen_test]
async fn a_shader_material_draws() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);

    // Colour comes from the uniform block, brightness from the storage slot —
    // so a frame that renders but binds either wrongly is the wrong colour
    // rather than no pixels.
    let mut material = threers::wasm::WebShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         let tint = u_data.data[0].rgb;\n\
         let gain = s0(3u);\n\
         return vec4<f32>(framebuffer_encode(tint * gain), 1.0);\n\
         }\n",
    );
    material.set_data(vec![0.0, 0.8, 0.2, 1.0]);
    material.set_storage(0, vec![0.0, 0.0, 0.0, 1.0]);

    let mut scene = threers::wasm::WebScene::new();
    scene.add(&threers::wasm::WebMesh::new(
        &threers::wasm::WebGeometry::box_(2.0, 2.0, 2.0),
        &material.material(),
    ));
    let mut camera = threers::wasm::WebCamera::perspective(55.0, 1.0, 0.1, 100.0);
    camera.set_position(0.0, 0.0, 4.0);

    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);
    let px = read_target(&renderer, target.id, 64, 64).await;

    let centre = ((32 * 64 + 32) * 4) as usize;
    let (r, g, b) = (px[centre], px[centre + 1], px[centre + 2]);
    assert!(
        g > 120 && g > r + 40 && g > b + 40,
        "expected the green the uniform asked for, got ({r}, {g}, {b}) — the \
         shader material did not reach the backend"
    );
}

/// A `ShaderMaterial` on an `InstancedMesh`, reading per-instance custom
/// columns — vertex attributes, so WebGL2 has them too, unlike the storage
/// arrays.
#[wasm_bindgen_test]
async fn an_instanced_shader_material_reads_custom_columns() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
    let mut material = threers::wasm::WebShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(framebuffer_encode(in.vertex_color.rgb), 1.0);\n\
         }\n",
    );
    material.set_vertex(
        "@vertex\n\
         fn vs_custom(in: InstancedCustomIn, @builtin(vertex_index) vid: u32,\n\
                      @builtin(instance_index) iid: u32) -> VsOut {\n\
         var out = standard_vertex_instanced(instanced_in(in), vid, iid);\n\
         out.vertex_color = vec4<f32>(in.icustom1.rgb, 1.0);\n\
         return out;\n\
         }\n",
    );
    let mut scene = threers::wasm::WebScene::new();
    let cubes = scene.create_instanced_mesh(
        &threers::wasm::WebGeometry::box_(0.8, 0.8, 0.8),
        &material.material(),
        2,
    );
    // Left and right of centre; red on the left, blue on the right.
    let mut matrices = Vec::new();
    for x in [-0.8f32, 0.8] {
        matrices.extend_from_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, 0.0, 0.0, 1.0,
        ]);
    }
    scene.set_instance_matrices(&cubes, matrices);
    scene.set_instance_custom(&cubes, 1, 3, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    scene.set_children(&scene.root_handle(), vec![cubes.id()]);
    let mut camera = threers::wasm::WebCamera::perspective(55.0, 1.0, 0.1, 100.0);
    camera.set_position(0.0, 0.0, 4.0);

    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);
    let px = read_target(&renderer, target.id, 64, 64).await;
    let at = |x: usize| {
        let i = (32 * 64 + x) * 4;
        (px[i], px[i + 1], px[i + 2])
    };
    let (left, middle, right) = (at(22), at(32), at(42));
    assert!(left.0 > 200 && left.2 < 40, "left instance red: {left:?}");
    assert!(
        right.2 > 200 && right.0 < 40,
        "right instance blue: {right:?}"
    );
    assert!(
        middle.0 < 40 && middle.2 < 40,
        "nothing between them: {middle:?}"
    );
}

/// Reversed depth (`setReversedZ`) in a browser: the same frame as standard
/// depth — boxes hiding each other, a blended pane — on this backend's own
/// depth compares, clears and shaders.
#[wasm_bindgen_test]
async fn reversed_depth_draws_the_same_scene() {
    use threers::wasm::{
        WebCamera, WebColor, WebEuler, WebGeometry, WebLight, WebMaterial, WebMesh,
        WebRenderTarget, WebScene, WebVector3,
    };
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let rt = WebRenderTarget::new(&renderer, 64, 64);
    let mut scene = WebScene::new();
    let put =
        |scene: &mut WebScene, size: f32, colour: (f32, f32, f32), opacity: f32, z: f32, x: f32| {
            let mut m =
                WebMaterial::standard(&WebColor::new(colour.0, colour.1, colour.2), 0.5, 0.0);
            if opacity < 1.0 {
                m.set_opacity(opacity);
            }
            let h = scene.add(&WebMesh::new(&WebGeometry::box_(size, size, size), &m));
            scene.set_transform(
                &h,
                &WebVector3::new(x, 0.0, z),
                &WebEuler::new(0.3, 0.5, 0.0),
            );
        };
    put(&mut scene, 2.0, (0.2, 0.3, 1.0), 1.0, -1.0, 0.4);
    put(&mut scene, 1.4, (1.0, 0.2, 0.1), 1.0, 0.3, -0.3);
    put(&mut scene, 1.0, (0.9, 0.9, 0.2), 0.5, 1.4, 0.2);
    let white = WebColor::new(1.0, 1.0, 1.0);
    let mut sun = WebLight::directional(&white, 2.0);
    sun.set_direction(-0.3, -0.9, -0.4);
    scene.add_light(&sun);
    scene.add_light(&WebLight::ambient(&white, 0.35));
    let mut camera = WebCamera::perspective(50.0, 1.0, 0.1, 100.0);
    camera.set_position(0.0, 0.5, 6.0);

    let mut frames = Vec::new();
    for reversed in [false, true] {
        renderer.set_reversed_z(reversed);
        renderer.set_render_target(rt.id);
        renderer.render(&mut scene, &camera);
        renderer.render(&mut scene, &camera);
        frames.push(read_target(&renderer, rt.id, 64, 64).await);
    }
    renderer.set_reversed_z(false);
    let lit = frames[0]
        .chunks(4)
        .filter(|p| p[0] > 60 || p[2] > 60)
        .count();
    assert!(lit > 400, "the boxes drew: {lit} px");
    let differ = frames[0]
        .chunks(4)
        .zip(frames[1].chunks(4))
        .filter(|(a, b)| (0..3).any(|c| a[c].abs_diff(b[c]) > 24))
        .count();
    assert!(
        differ <= 12,
        "{differ} px differ between standard and reversed depth"
    );
}

/// Morph targets — the morph texture read in the vertex stage with
/// `texelFetch`, the path that exists because WebGL2 has no storage buffers —
/// and a material array over geometry groups. A plane morphed 0.6 to the right
/// by an absolute target draws exactly what the same plane moved there by its
/// transform draws; and a box drawn per group shows both of its materials.
#[wasm_bindgen_test]
async fn morph_targets_and_groups_draw() {
    use threers::wasm::{
        WebCamera, WebColor, WebEuler, WebGeometry, WebMaterial, WebMesh, WebRenderTarget,
        WebScene, WebVector3,
    };
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let rt = WebRenderTarget::new(&renderer, 64, 64);
    let mut camera = WebCamera::perspective(50.0, 1.0, 0.1, 100.0);
    camera.set_position(0.0, 0.0, 4.0);
    let white = WebColor::new(1.0, 1.0, 1.0);

    // Morphed: a 1x1 plane with an absolute target 0.6 to the right.
    let mut morphed = WebScene::new();
    let h = morphed.add(&WebMesh::new(
        &WebGeometry::plane(1.0, 1.0),
        &WebMaterial::basic(&white),
    ));
    let shifted: Vec<f32> = vec![
        -0.5 + 0.6,
        0.5,
        0.0,
        0.5 + 0.6,
        0.5,
        0.0,
        -0.5 + 0.6,
        -0.5,
        0.0,
        0.5 + 0.6,
        -0.5,
        0.0,
    ];
    morphed.set_morph_attribute(&h, "position", 0, shifted, 3);
    morphed.update_morph_targets(&h);
    morphed.set_morph_target_influences(&h, vec![1.0]);
    renderer.set_render_target(rt.id);
    renderer.render(&mut morphed, &camera);
    let a = read_target(&renderer, rt.id, 64, 64).await;

    // Reference: the same plane moved by its transform.
    let mut moved = WebScene::new();
    let h2 = moved.add(&WebMesh::new(
        &WebGeometry::plane(1.0, 1.0),
        &WebMaterial::basic(&white),
    ));
    moved.set_transform(
        &h2,
        &WebVector3::new(0.6, 0.0, 0.0),
        &WebEuler::new(0.0, 0.0, 0.0),
    );
    renderer.render(&mut moved, &camera);
    let b = read_target(&renderer, rt.id, 64, 64).await;
    let lit = |img: &[u8]| img.chunks(4).filter(|p| p[0] > 128).count();
    let differ = a
        .chunks(4)
        .zip(b.chunks(4))
        .filter(|(p, q)| (p[0] > 128) != (q[0] > 128))
        .count();
    assert!(lit(&b) > 100, "the reference drew {} pixels", lit(&b));
    assert!(
        differ <= lit(&b) / 50,
        "the morph drew {} pixels, the moved plane {}, {differ} differ",
        lit(&a),
        lit(&b)
    );

    // Groups: a box drawn per face group with two alternating materials.
    let mut grouped = WebScene::new();
    let red = WebMaterial::basic(&WebColor::new(1.0, 0.0, 0.0));
    let green = WebMaterial::basic(&WebColor::new(0.0, 1.0, 0.0));
    let hb = grouped.add(&WebMesh::new(&WebGeometry::box_(1.2, 1.2, 1.2), &red));
    grouped.clear_mesh_materials(&hb);
    grouped.push_mesh_material(&hb, &red);
    grouped.push_mesh_material(&hb, &green);
    // BoxGeometry's six groups use materials 0..5; map them onto two.
    grouped.set_mesh_groups(
        &hb,
        vec![0, 6, 0, 6, 6, 1, 12, 6, 0, 18, 6, 1, 24, 6, 0, 30, 6, 1],
    );
    grouped.set_transform(
        &hb,
        &WebVector3::new(0.0, 0.0, 0.0),
        &WebEuler::new(0.5, 0.6, 0.0),
    );
    renderer.render(&mut grouped, &camera);
    let c = read_target(&renderer, rt.id, 64, 64).await;
    let reds = c.chunks(4).filter(|p| p[0] > 128 && p[1] < 64).count();
    let greens = c.chunks(4).filter(|p| p[1] > 128 && p[0] < 64).count();
    assert!(
        reds > 50 && greens > 50,
        "per-group materials: {reds} red, {greens} green"
    );
}

/// `readRenderTargetPixels` returns RGBA, as three.js's does, whatever the
/// target's format: a WebGPU target takes the canvas's `bgra8unorm`, and its
/// bytes came back with red and blue swapped.
#[wasm_bindgen_test]
async fn render_target_read_back_is_rgba() {
    use threers::wasm::{
        WebCamera, WebColor, WebGeometry, WebMaterial, WebMesh, WebRenderTarget, WebScene,
    };
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 1.0));
    scene.add(&WebMesh::new(
        &WebGeometry::plane(1.0, 2.0),
        &WebMaterial::basic(&WebColor::new(1.0, 0.0, 0.0)),
    ));
    let mut camera = WebCamera::orthographic(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    camera.set_position(0.0, 0.0, 5.0);
    camera.look_at(0.0, 0.0, 0.0);
    let target = WebRenderTarget::new(&renderer, 16, 16);
    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    let px = read_target(&renderer, target.id, 16, 16).await;
    renderer.set_render_target(0);
    let at = |x: usize| px[(8 * 16 + x) * 4..(8 * 16 + x) * 4 + 4].to_vec();
    assert_eq!(at(8), vec![255, 0, 0, 255], "the red quad");
    assert_eq!(at(1), vec![0, 0, 255, 255], "the blue background");
}

/// Render targets as three.js treats them: no tone mapping inside one (only
/// the final output is tone-mapped), read-back in the target's colour space
/// (linear for `NoColorSpace`, encoded for `SRGBColorSpace`), and `setSize`
/// reallocating in place under the same id.
#[wasm_bindgen_test]
async fn render_targets_follow_three_js() {
    use threers::wasm::{
        WebCamera, WebColor, WebGeometry, WebMaterial, WebMesh, WebRenderTarget, WebScene,
    };
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let quad = |grey: f32| {
        let mut scene = WebScene::new();
        scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
        scene.add(&WebMesh::new(
            &WebGeometry::plane(2.0, 2.0),
            &WebMaterial::basic(&WebColor::new(grey, grey, grey)),
        ));
        scene
    };
    let mut camera = WebCamera::orthographic(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    camera.set_position(0.0, 0.0, 5.0);
    camera.look_at(0.0, 0.0, 0.0);
    let mut target = WebRenderTarget::new(&renderer, 16, 16);

    // Linear tone mapping at exposure 0.5 would halve white; not in a target.
    renderer.set_tone_mapping(1, 0.5);
    renderer.set_render_target(target.id);
    renderer.render(&mut quad(1.0), &camera);
    let px = read_target(&renderer, target.id, 16, 16).await;
    renderer.set_tone_mapping(0, 1.0);
    assert_eq!(
        px[(8 * 16 + 8) * 4],
        255,
        "a render target stores untone-mapped light"
    );

    renderer.render(&mut quad(0.5), &camera);
    let encoded = read_target(&renderer, target.id, 16, 16).await;
    assert_eq!(
        encoded[(8 * 16 + 8) * 4],
        188,
        "this API's default reads back the stored, encoded bytes"
    );
    target.set_srgb_color_space(false);
    let linear = read_target(&renderer, target.id, 16, 16).await;
    assert_eq!(
        linear[(8 * 16 + 8) * 4],
        128,
        "NoColorSpace — three.js's default — reads back linear"
    );

    target.set_size(32, 8);
    renderer.set_render_target(target.id);
    renderer.render(&mut quad(1.0), &camera);
    let resized = read_target(&renderer, target.id, 32, 8).await;
    renderer.set_render_target(0);
    assert_eq!(
        resized.len(),
        32 * 8 * 4,
        "the resized target reads back at its new size"
    );
    assert_eq!(resized[(4 * 32 + 16) * 4], 255, "and holds the new frame");
}

/// three.js `Points`: one instanced draw, each point a `size`-pixel quad
/// (`vs_points`) — 16 pixels: lit 7 off centre, dark 9 off — tinted by its
/// vertex colour, and with `sizeAttenuation` under a perspective camera
/// `size × (height / 2) / −z`.
#[wasm_bindgen_test]
async fn points_draw_sized_quads() {
    use threers::wasm::{
        WebBufferAttribute, WebBufferGeometry, WebCamera, WebColor, WebMaterial, WebRenderTarget,
        WebScene,
    };
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let n = 64u32;
    let target = WebRenderTarget::new(&renderer, n, n);
    let draw = |renderer: &mut threers::wasm::WebRenderer,
                size: f32,
                attenuate: bool,
                camera: &WebCamera| {
        let mut g = WebBufferGeometry::new();
        g.set_attribute("position", WebBufferAttribute::new(vec![0.0, 0.0, 0.0], 3));
        g.set_attribute("color", WebBufferAttribute::new(vec![0.0, 1.0, 0.0], 3));
        let mut m = WebMaterial::points(&WebColor::new(1.0, 1.0, 1.0), size);
        m.set_size_attenuation(attenuate);
        let mut scene = WebScene::new();
        scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
        scene.add_points(&g, &m);
        renderer.render(&mut scene, camera);
    };
    let mut ortho = WebCamera::orthographic(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    ortho.set_position(0.0, 0.0, 5.0);
    ortho.look_at(0.0, 0.0, 0.0);
    renderer.set_render_target(target.id);
    draw(&mut renderer, 16.0, false, &ortho);
    let px = read_target(&renderer, target.id, n, n).await;
    let at = |px: &[u8], dx: u32| {
        let i = ((n / 2 * n + n / 2 + dx) * 4) as usize;
        [px[i], px[i + 1], px[i + 2]]
    };
    assert_eq!(
        at(&px, 7),
        [0, 255, 0],
        "16 px, 7 off centre, in its vertex colour"
    );
    assert_eq!(at(&px, 9), [0, 0, 0], "16 px, 9 off centre");

    let mut persp = WebCamera::perspective(60.0, 1.0, 0.1, 100.0);
    persp.set_position(0.0, 0.0, 4.0);
    persp.look_at(0.0, 0.0, 0.0);
    renderer.set_render_target(target.id);
    draw(&mut renderer, 2.0, true, &persp);
    let px = read_target(&renderer, target.id, n, n).await;
    renderer.set_render_target(0);
    assert_eq!(
        at(&px, 7),
        [0, 255, 0],
        "size 2 attenuated at distance 4: 16 px"
    );
}

/// Every glass mode draws under several shadow-casting lights, with MSAA on.
///
/// Each mode reached a WebGL2 limit of its own: the refraction pass's group
/// 3 has room for one texture beside its capture, so its glass gets its
/// shadows packed in an integer texture (`fs_glass_shadow`); `Glass` draws in
/// two halves selected by a pipeline-overridable constant; `Oit` cannot blend
/// two attachments differently on WebGL2 and falls back to `Glass`; and the
/// shadow maps are depth arrays, a layer per light. On WebGPU, Tint has to
/// accept the same WGSL and the multisampled OIT and refraction paths run.
#[wasm_bindgen_test]
async fn glass_modes_draw_under_many_shadowing_lights() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    renderer.set_msaa(4);

    let white = threers::wasm::WebColor::new(1.0, 1.0, 1.0);
    for mode in 0..4u32 {
        let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
        let mut scene = threers::wasm::WebScene::new();
        let floor_colour = threers::wasm::WebColor::new(0.8, 0.8, 0.8);
        let floor = scene.add(&threers::wasm::WebMesh::new(
            &threers::wasm::WebGeometry::box_(8.0, 0.2, 8.0),
            &threers::wasm::WebMaterial::standard(&floor_colour, 0.8, 0.0),
        ));
        scene.set_transform(
            &floor,
            &threers::wasm::WebVector3::new(0.0, -1.2, 0.0),
            &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
        );
        scene.set_object_receive_shadow(&floor, true);
        let caster_colour = threers::wasm::WebColor::new(0.9, 0.4, 0.2);
        let caster = scene.add(&threers::wasm::WebMesh::new(
            &threers::wasm::WebGeometry::box_(0.8, 0.8, 0.8),
            &threers::wasm::WebMaterial::standard(&caster_colour, 0.4, 0.0),
        ));
        scene.set_transform(
            &caster,
            &threers::wasm::WebVector3::new(-1.0, -0.6, -0.5),
            &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
        );
        scene.set_object_cast_shadow(&caster, true);

        let tint = threers::wasm::WebColor::new(0.3, 0.9, 0.4);
        let mut glass = threers::wasm::WebMaterial::physical(&tint, 0.1, 0.0, 0.0, 0.0);
        glass.set_transmission(1.0, 1.5, 0.4, 0.0);
        glass.set_transparency_mode(mode);
        let ball = scene.add(&threers::wasm::WebMesh::new(
            &threers::wasm::WebGeometry::sphere(0.8, 24, 16),
            &glass,
        ));
        scene.set_object_receive_shadow(&ball, true);

        // Two of each shadow-casting kind, so every map array has two layers.
        for (dx, dz) in [(-0.4f32, -0.6f32), (0.5, -0.3)] {
            let mut sun = threers::wasm::WebLight::directional(&white, 1.2);
            sun.set_direction(dx, -0.8, dz);
            sun.set_cast_shadow(true);
            // Small maps: WebGL2 runs here on SwiftShader, in software.
            sun.set_shadow_map_size(1024, 1024);
            scene.add_light(&sun);
        }
        for x in [-2.0f32, 2.0] {
            let mut lamp = threers::wasm::WebLight::point(&white, 6.0, 0.0, 2.0);
            lamp.set_cast_shadow(true);
            let h = scene.add_light(&lamp);
            scene.set_transform(
                &h,
                &threers::wasm::WebVector3::new(x, 2.0, 1.0),
                &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
            );
        }
        scene.add_light(&threers::wasm::WebLight::ambient(&white, 0.3));

        let mut camera = threers::wasm::WebCamera::perspective(55.0, 1.0, 0.1, 100.0);
        camera.set_position(0.0, 1.0, 4.5);
        renderer.set_render_target(target.id);
        renderer.render(&mut scene, &camera);
        renderer.render(&mut scene, &camera);
        let pixels = read_target(&renderer, target.id, 64, 64).await;
        let distinct: std::collections::BTreeSet<(u8, u8, u8)> =
            pixels.chunks(4).map(|p| (p[0], p[1], p[2])).collect();
        assert!(
            distinct.len() > 30,
            "glass mode {mode}: only {} distinct colours — the frame did not draw",
            distinct.len()
        );
        // The ball is tinted green, in every mode.
        let green = pixels
            .chunks(4)
            .filter(|p| p[1] as i32 > p[0] as i32 + 20 && p[1] as i32 > p[2] as i32 + 10)
            .count();
        assert!(
            green > 40,
            "glass mode {mode}: {green} green pixels — the glass did not draw"
        );
        // And onto the canvas, the multisampled surface path.
        renderer.set_render_target(0);
        renderer.render(&mut scene, &camera);
    }
    renderer.set_msaa(1);
}

/// One pixel of the canvas, through a 2D canvas: no read-back API reaches
/// the canvas, and MSAA is only drawn there — a render target is drawn with
/// its own live depth, which keeps it single-sampled. Call it in the same
/// task as the render, before the browser presents and clears the frame.
fn canvas_pixel(canvas: &web_sys::HtmlCanvasElement, x: u32, y: u32) -> [u8; 3] {
    let read = js_sys::Function::new_with_args(
        "c, x, y",
        "const o = document.createElement('canvas'); o.width = c.width; o.height = c.height;
         const g = o.getContext('2d'); g.drawImage(c, 0, 0);
         return Array.from(g.getImageData(x, y, 1, 1).data);",
    );
    let v = read
        .call3(&wasm_bindgen::JsValue::NULL, canvas, &x.into(), &y.into())
        .expect("canvas read");
    let a = js_sys::Array::from(&v);
    [0, 1, 2].map(|k| a.get(k).as_f64().unwrap_or(0.0) as u8)
}

/// What `glass_occlusion.rs`, `glass_layers.rs` and `shadow_types.rs` check
/// natively, on the browser's own backend. On WebGL2 the multisampled depth
/// cannot be read by a shader, so the refraction pass's depth is the opaque
/// scene drawn again — the path the native tests only reach by forcing it.
#[wasm_bindgen_test]
async fn glass_layers_msaa_occlusion_and_shadow_types() {
    use threers::wasm::{
        WebCamera, WebColor, WebEuler, WebGeometry, WebLight, WebMaterial, WebMesh,
        WebRenderTarget, WebScene, WebVector3,
    };
    const N: u32 = 64;
    let (shared, canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let (cw, ch) = (canvas.width(), canvas.height());
    let glass = |r: f32, g: f32, b: f32| {
        let mut m = WebMaterial::physical(&WebColor::new(r, g, b), 0.0, 0.0, 0.0, 0.0);
        m.set_transmission(1.0, 1.5, 0.0, 0.0);
        m.set_transparency_mode(3); // Refract
        m
    };
    let place = |scene: &mut WebScene, mesh: &WebMesh, x: f32, y: f32, z: f32| {
        let h = scene.add(mesh);
        scene.set_transform(&h, &WebVector3::new(x, y, z), &WebEuler::new(0.0, 0.0, 0.0));
        h
    };
    let front_camera = || {
        let mut c = WebCamera::perspective(40.0, cw as f32 / ch as f32, 0.1, 100.0);
        c.set_position(0.0, 0.0, 5.0);
        c.look_at(0.0, 0.0, 0.0);
        c
    };
    // Pixels per world unit at z = 0: 40 degrees across the height, 5 away.
    let ppu = ch as f32 / (2.0 * 5.0 * 20f32.to_radians().tan());
    let to_px = |x: f32| (cw as f32 / 2.0 + x * ppu) as u32;

    // --- An opaque box in front of refraction glass, under MSAA. ---
    {
        let mut scene = WebScene::new();
        scene.set_background(&WebColor::new(0.1, 0.15, 0.25));
        scene.add_light(&WebLight::ambient(&WebColor::new(1.0, 1.0, 1.0), 1.0));
        let wall = WebMesh::new(
            &WebGeometry::plane(8.0, 8.0),
            &WebMaterial::basic(&WebColor::new(0.1, 0.2, 1.0)),
        );
        place(&mut scene, &wall, 0.0, 0.0, -3.0);
        // Strongly green: glass drawn over the red box would darken it.
        let ball = WebMesh::new(&WebGeometry::sphere(1.0, 24, 16), &glass(0.05, 0.9, 0.05));
        place(&mut scene, &ball, 0.0, 0.0, 0.0);
        let red = WebMesh::new(
            &WebGeometry::box_(0.8, 0.8, 0.8),
            &WebMaterial::basic(&WebColor::new(1.0, 0.0, 0.0)),
        );
        place(&mut scene, &red, 0.0, 0.0, 1.8);
        renderer.set_msaa(4);
        renderer.render(&mut scene, &front_camera());
        let multisampled = renderer.last_frame_multisampled();
        let c = canvas_pixel(&canvas, cw / 2, ch / 2);
        renderer.set_msaa(1);
        assert!(
            multisampled,
            "MSAA asked for over refraction glass, a single-sampled frame drawn"
        );
        assert!(
            c[0] > 240 && c[1] < 20 && c[2] < 20,
            "the box in front of the glass reads {c:?} — the glass drew over it"
        );

        // The same into a render target with three.js's `samples`, the
        // renderer's own MSAA off: multisampled, and read back directly.
        let target = WebRenderTarget::new(&renderer, N, N);
        target.set_samples(4);
        renderer.set_render_target(target.id);
        let mut square = WebCamera::perspective(40.0, 1.0, 0.1, 100.0);
        square.set_position(0.0, 0.0, 5.0);
        square.look_at(0.0, 0.0, 0.0);
        renderer.render(&mut scene, &square);
        let multisampled = renderer.last_frame_multisampled();
        renderer.set_render_target(0);
        let px = read_target(&renderer, target.id, N, N).await;
        assert!(
            multisampled,
            "a render target with samples 4 drew single-sampled"
        );
        let i = ((N / 2 * N + N / 2) * 4) as usize;
        let c = [px[i], px[i + 1], px[i + 2]];
        assert!(
            c[0] > 240 && c[1] < 20 && c[2] < 20,
            "in a multisampled target the box in front reads {c:?}"
        );
    }

    // --- Glass behind glass: a second layer shows it through the first. ---
    {
        let mut frame = |layers: u32| {
            let mut scene = WebScene::new();
            scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
            scene.add_light(&WebLight::ambient(&WebColor::new(1.0, 1.0, 1.0), 1.0));
            let wall = WebMesh::new(
                &WebGeometry::plane(10.0, 10.0),
                &WebMaterial::basic(&WebColor::new(1.0, 1.0, 1.0)),
            );
            place(&mut scene, &wall, 0.0, 0.0, -2.0);
            let back = WebMesh::new(&WebGeometry::plane(2.0, 2.0), &glass(0.95, 0.05, 0.05));
            place(&mut scene, &back, 0.0, 0.0, -1.0);
            let front = WebMesh::new(&WebGeometry::plane(1.4, 1.4), &glass(0.95, 0.95, 0.95));
            place(&mut scene, &front, -0.6, 0.0, 0.0);
            renderer.set_refraction_layers(layers);
            renderer.set_msaa(4);
            renderer.render(&mut scene, &front_camera());
            let multisampled = renderer.last_frame_multisampled();
            // Where both plates cover, left of centre.
            let c = canvas_pixel(&canvas, to_px(-0.3), ch / 2);
            renderer.set_msaa(1);
            assert!(multisampled, "{layers} layers: not multisampled");
            c
        };
        let (o, t) = (frame(1), frame(2));
        assert!(
            o[1] > 150 && o[2] > 150,
            "one layer: the wall through the clear plate, read {o:?}"
        );
        assert!(
            t[0] > 150 && t[1] < 80 && t[2] < 80,
            "two layers: the red plate behind the clear one reads {t:?}"
        );
    }

    // --- Each of three.js's shadow map types draws, and they differ. ---
    {
        let mut frames = Vec::new();
        for kind in 0..4u32 {
            renderer.set_shadow_type(kind);
            assert_eq!(renderer.shadow_type(), kind, "shadowType did not stick");
            let target = WebRenderTarget::new(&renderer, N, N);
            let mut scene = WebScene::new();
            let white = WebColor::new(1.0, 1.0, 1.0);
            let floor = WebMesh::new(
                &WebGeometry::box_(8.0, 0.2, 8.0),
                &WebMaterial::standard(&WebColor::new(0.8, 0.8, 0.8), 0.9, 0.0),
            );
            let floor = place(&mut scene, &floor, 0.0, -0.1, 0.0);
            scene.set_object_receive_shadow(&floor, true);
            let caster = WebMesh::new(
                &WebGeometry::box_(1.0, 1.0, 1.0),
                &WebMaterial::standard(&WebColor::new(0.9, 0.4, 0.2), 0.5, 0.0),
            );
            let caster = place(&mut scene, &caster, 0.0, 1.5, 0.0);
            scene.set_object_cast_shadow(&caster, true);
            let mut sun = WebLight::directional(&white, 2.0);
            sun.set_direction(-0.3, -1.0, -0.2);
            sun.set_cast_shadow(true);
            sun.set_shadow_map_size(1024, 1024);
            scene.add_light(&sun);
            scene.add_light(&WebLight::ambient(&white, 0.2));
            let mut camera = WebCamera::perspective(50.0, 1.0, 0.1, 100.0);
            camera.set_position(0.0, 5.0, 4.0);
            camera.look_at(0.0, 0.0, 0.0);
            renderer.set_render_target(target.id);
            renderer.render(&mut scene, &camera);
            frames.push(read_target(&renderer, target.id, N, N).await);
        }
        renderer.set_shadow_type(1);
        for (kind, px) in frames.iter().enumerate() {
            // Floor in shadow: grey, and well below the lit floor.
            let dark = px
                .chunks(4)
                .filter(|p| p[0] < 90 && (p[0] as i32 - p[2] as i32).abs() < 12)
                .count();
            assert!(
                dark > 20,
                "shadow type {kind}: {dark} shadowed floor pixels"
            );
        }
        let differ = |a: &[u8], b: &[u8]| {
            a.chunks(4)
                .zip(b.chunks(4))
                .filter(|(p, q)| (p[0] as i32 - q[0] as i32).abs() > 4)
                .count()
        };
        let (hard_soft, hard_pcss) = (
            differ(&frames[0], &frames[2]),
            differ(&frames[0], &frames[3]),
        );
        assert!(
            hard_soft > 3,
            "basic and PCF soft shadows draw the same ({hard_soft} pixels differ)"
        );
        assert!(
            hard_pcss > 3,
            "basic and PCSS shadows draw the same ({hard_pcss} pixels differ)"
        );
    }
    renderer.set_refraction_layers(2);
    renderer.set_render_target(0);
}

/// Shell fur through the JS API (`WebScene.addFur`): a shader material on an
/// instanced mesh with a custom vertex stage — Tint, and on WebGL2 ANGLE,
/// have to take the shader and draw shells past the skin's silhouette.
#[wasm_bindgen_test]
async fn fur_grows_past_the_skin() {
    use threers::wasm::{WebCamera, WebColor, WebFurMaterial, WebGeometry, WebLight, WebScene};
    const N: u32 = 96;
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let target = threers::wasm::WebRenderTarget::new(&renderer, N, N);

    let mut fur = WebFurMaterial::new();
    fur.set_length(0.4);
    fur.set_density(40.0);
    fur.set_gravity(0.0, 0.0, 0.0);
    fur.set_shells(24);
    fur.set_colors(&WebColor::new(0.6, 0.5, 0.4), &WebColor::new(1.0, 0.9, 0.7));
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    scene.add_light(&WebLight::ambient(&WebColor::new(1.0, 1.0, 1.0), 2.0));
    scene.add_fur(&WebGeometry::sphere(1.0, 48, 24), &fur);
    let mut camera = WebCamera::perspective(40.0, 1.0, 0.1, 50.0);
    camera.set_position(0.0, 0.0, 5.0);
    camera.look_at(0.0, 0.0, 0.0);

    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    renderer.set_render_target(0);
    let px = read_target(&renderer, target.id, N, N).await;

    // The bare sphere's silhouette is ~26 px across from the centre; the fur
    // reaches ~35. A band between should be part-covered, and nothing far out.
    let per_unit = N as f32 / 2.0 / (5.0 * 20f32.to_radians().tan());
    let (mut band, mut band_lit, mut far_lit) = (0, 0, 0);
    for y in 0..N {
        for x in 0..N {
            let r = ((x as f32 + 0.5 - N as f32 / 2.0).powi(2)
                + (y as f32 + 0.5 - N as f32 / 2.0).powi(2))
            .sqrt()
                / per_unit;
            let i = ((y * N + x) * 4) as usize;
            let lit = px[i] as u32 + px[i + 1] as u32 + px[i + 2] as u32 > 30;
            if r > 1.08 && r < 1.25 {
                band += 1;
                band_lit += lit as u32;
            } else if r > 1.6 {
                far_lit += lit as u32;
            }
        }
    }
    assert!(
        band_lit as f32 > 0.08 * band as f32,
        "fur covers {band_lit} of {band} pixels past the skin — the shells did not draw"
    );
    assert_eq!(far_lit, 0, "{far_lit} pixels lit far beyond the fur");
}

/// Strand hair through the JS API (`WebScene.addHair`): ribbons expanded in
/// a custom vertex stage, with the fur fragment's hair path — they have to
/// draw past the sphere they grow on and hang down under gravity.
#[wasm_bindgen_test]
async fn hair_hangs_past_the_scalp() {
    use threers::wasm::{WebCamera, WebColor, WebFurMaterial, WebGeometry, WebLight, WebScene};
    const N: u32 = 96;
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let target = threers::wasm::WebRenderTarget::new(&renderer, N, N);

    let mut hair = WebFurMaterial::new();
    hair.set_length(0.4);
    hair.set_gravity(0.0, -2.0, 0.0);
    hair.set_colors(&WebColor::new(0.6, 0.5, 0.4), &WebColor::new(1.0, 0.9, 0.7));
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    scene.add_light(&WebLight::ambient(&WebColor::new(1.0, 1.0, 1.0), 2.0));
    scene.add_hair(
        &WebGeometry::sphere(1.0, 48, 24),
        &hair,
        4000,
        6,
        0.03,
        None,
    );
    let mut camera = WebCamera::perspective(40.0, 1.0, 0.1, 50.0);
    camera.set_position(0.0, 0.0, 5.0);
    camera.look_at(0.0, 0.0, 0.0);

    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    renderer.set_render_target(0);
    let px = read_target(&renderer, target.id, N, N).await;

    let per_unit = N as f32 / 2.0 / (5.0 * 20f32.to_radians().tan());
    let (mut above, mut below, mut far_lit) = (0, 0, 0);
    for y in 0..N {
        for x in 0..N {
            let (dx, dy) = (
                x as f32 + 0.5 - N as f32 / 2.0,
                y as f32 + 0.5 - N as f32 / 2.0,
            );
            let r = (dx * dx + dy * dy).sqrt() / per_unit;
            let i = ((y * N + x) * 4) as usize;
            let lit = px[i] as u32 + px[i + 1] as u32 + px[i + 2] as u32 > 30;
            if r > 1.06 && r < 1.5 && lit {
                if dy < 0.0 {
                    above += 1;
                } else {
                    below += 1;
                }
            } else if r > 1.6 {
                far_lit += lit as u32;
            }
        }
    }
    assert!(
        below > 60,
        "only {below} hair pixels below the scalp — the ribbons did not draw"
    );
    assert!(
        below > above * 3 / 2,
        "gravity did not pull the hair down ({above} above, {below} below)"
    );
    assert_eq!(far_lit, 0, "{far_lit} pixels lit far beyond the hair");
}

/// Hair on a skinned mesh through the JS scene graph, as the three.js shim
/// builds it (`createSkinnedMesh`, `setSkeleton`, `createSkinnedHair`,
/// `setChildren`): it draws, and moves with the bone — the posed strands and
/// their per-strand lag (in uv1/uv2, read by the custom vertex stage) on the
/// browser's backend.
#[wasm_bindgen_test]
async fn skinned_hair_follows_its_bone() {
    use threers::prelude::*;
    use threers::wasm::{
        WebBufferAttribute, WebBufferGeometry, WebCamera, WebColor, WebFurMaterial, WebGeometry,
        WebLight, WebMaterial, WebScene,
    };
    const N: u32 = 96;
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let target = threers::wasm::WebRenderTarget::new(&renderer, N, N);

    let sphere = SphereGeometry::new(0.5, 32, 16);
    let mut bg = WebBufferGeometry::new();
    for (name, size) in [("position", 3), ("normal", 3), ("uv", 2)] {
        let a = sphere.get_attribute(name).unwrap();
        bg.set_attribute(name, WebBufferAttribute::new(a.array.clone(), size));
    }
    let count = sphere.get_attribute("position").unwrap().count();
    bg.set_attribute("joint", WebBufferAttribute::new(vec![0.0; count * 4], 4));
    bg.set_attribute(
        "weight",
        WebBufferAttribute::new((0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(), 4),
    );
    bg.set_index(sphere.index.clone().unwrap());
    let geometry = WebGeometry::share_buffer_geometry(&bg);

    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    let bone = scene.create_group(false);
    let mesh = scene.create_skinned_mesh(
        &geometry,
        &WebMaterial::basic(&WebColor::new(0.0, 0.0, 0.0)),
    );
    scene.set_children(&scene.root_handle(), vec![bone.id(), mesh.id()]);
    // After the root's children are set: they replace whatever was there.
    scene.add_light(&WebLight::ambient(&WebColor::new(1.0, 1.0, 1.0), 2.0));
    let identity: Vec<f32> = Matrix4::identity().elements.to_vec();
    scene.set_skeleton(&mesh, vec![bone.id()], identity.clone());
    let mut hair = WebFurMaterial::new();
    hair.set_length(0.35);
    hair.set_gravity(0.0, 0.0, 0.0);
    hair.set_colors(&WebColor::new(0.6, 0.5, 0.4), &WebColor::new(1.0, 0.9, 0.7));
    let strands = scene.create_skinned_hair(&mesh, &hair, 3000, 6, 0.03, None);
    scene.set_children(&mesh, vec![strands.id()]);
    let mut camera = WebCamera::perspective(40.0, 1.0, 0.1, 50.0);
    camera.set_position(0.0, 0.0, 5.0);
    camera.look_at(0.0, 0.0, 0.0);

    let mut centres = Vec::new();
    for x in [-0.8f32, 0.8] {
        let mut m = identity.clone();
        m[12] = x;
        scene.set_local_matrix(&bone, m);
        for f in 0..60 {
            scene.step_fur(f as f32 / 60.0, 1.0 / 60.0);
        }
        renderer.set_render_target(target.id);
        renderer.render(&mut scene, &camera);
        renderer.set_render_target(0);
        let px = read_target(&renderer, target.id, N, N).await;
        let (mut sx, mut n) = (0.0f32, 0u32);
        for y in 0..N {
            for x in 0..N {
                let i = ((y * N + x) * 4) as usize;
                if px[i] as u32 + px[i + 1] as u32 + px[i + 2] as u32 > 30 {
                    sx += x as f32;
                    n += 1;
                }
            }
        }
        assert!(
            n > 300,
            "only {n} hair pixels: the skinned hair did not draw"
        );
        centres.push(sx / n as f32);
    }
    let per_unit = N as f32 / 2.0 / (5.0 * 20f32.to_radians().tan());
    let moved = (centres[1] - centres[0]) / per_unit;
    assert!(
        (moved - 1.6).abs() < 0.1,
        "the hair moved {moved:.2}, the bone 1.6"
    );
}

/// Fur's and hair's shadows in the browser: each object's invisible caster
/// group (the coat's pushed-out skin, the hair's tubes) drawn into the
/// shadow map, darkening the floor under it — and nothing with the caster
/// turned off. The floor pixels are picked by ray-casting the camera.
#[wasm_bindgen_test]
async fn fur_and_hair_cast_shadows() {
    use threers::prelude::{Camera, PerspectiveCamera, Vector3};
    use threers::wasm::{
        WebCamera, WebColor, WebEuler, WebFurMaterial, WebGeometry, WebLight, WebMaterial, WebMesh,
        WebScene, WebVector3,
    };
    const N: u32 = 128;
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let target = threers::wasm::WebRenderTarget::new(&renderer, N, N);

    // The camera, in Rust too, for the rays that find the floor.
    let (eye, at) = (Vector3::new(0.0, 3.0, 5.0), Vector3::ZERO);
    let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 50.0);
    cam.position = eye;
    cam.look_at(at);
    // Floor pixels (the floor's top is y = −1) within 0.6 of under the
    // object at y = 1, whose rays pass clear of it.
    let under: Vec<bool> = (0..N * N)
        .map(|i| {
            let (x, y) = (i % N, i / N);
            let ndc = Vector3::new(
                (x as f32 + 0.5) / N as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / N as f32 * 2.0,
                0.5,
            );
            let dir =
                (ndc.unproject(&cam.view_matrix(), &cam.projection_matrix()) - eye).normalize();
            let to_obj = Vector3::new(0.0, 1.0, 0.0) - eye;
            if (to_obj - dir * to_obj.dot(dir)).length() < 1.3 || dir.y >= 0.0 {
                return false;
            }
            let p = eye + dir * ((-1.0 - eye.y) / dir.y);
            (p.x * p.x + p.z * p.z).sqrt() < 0.6
        })
        .collect();
    let floor_px = under.iter().filter(|&&u| u).count();
    assert!(
        floor_px > 100,
        "only {floor_px} floor pixels in view under the object"
    );

    let white = WebColor::new(1.0, 1.0, 1.0);
    let mut camera = WebCamera::perspective(45.0, 1.0, 0.1, 50.0);
    camera.set_position(eye.x, eye.y, eye.z);
    camera.look_at(at.x, at.y, at.z);
    for (what, hair) in [("fur", false), ("hair", true)] {
        let mut light = [0.0f64; 2];
        for (k, casts) in [true, false].into_iter().enumerate() {
            let mut scene = WebScene::new();
            scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
            let floor = scene.add(&WebMesh::new(
                &WebGeometry::box_(8.0, 0.2, 8.0),
                &WebMaterial::standard(&white, 1.0, 0.0),
            ));
            scene.set_transform(
                &floor,
                &WebVector3::new(0.0, -1.1, 0.0),
                &WebEuler::new(0.0, 0.0, 0.0),
            );
            scene.set_object_receive_shadow(&floor, true);
            let mut fur = WebFurMaterial::new();
            fur.set_length(0.4);
            fur.set_density(40.0);
            fur.set_gravity(0.0, 0.0, 0.0);
            fur.set_colors(&WebColor::new(0.6, 0.5, 0.4), &WebColor::new(1.0, 0.9, 0.7));
            if !casts {
                fur.set_shadow_height(0.0);
            }
            let obj = if hair {
                let share = if casts { 0.25 } else { 0.0 };
                scene.add_hair(
                    &WebGeometry::sphere(0.4, 32, 16),
                    &fur,
                    4000,
                    6,
                    0.015,
                    Some(share),
                )
            } else {
                scene.add_fur(&WebGeometry::sphere(0.5, 48, 24), &fur)
            };
            scene.set_transform(
                &obj,
                &WebVector3::new(0.0, 1.0, 0.0),
                &WebEuler::new(0.0, 0.0, 0.0),
            );
            let mut sun = WebLight::directional(&white, 2.0);
            sun.set_direction(0.0, -1.0, 0.0);
            sun.set_cast_shadow(true);
            sun.set_shadow_camera(-3.0, 3.0, 3.0, -3.0, 0.1, 30.0);
            sun.set_shadow_map_size(1024, 1024);
            scene.add_light(&sun);
            scene.add_light(&WebLight::ambient(&white, 0.3));

            renderer.set_render_target(target.id);
            renderer.render(&mut scene, &camera);
            renderer.set_render_target(0);
            let px = read_target(&renderer, target.id, N, N).await;
            let sum: f64 = under
                .iter()
                .enumerate()
                .filter(|(_, &u)| u)
                .map(|(i, _)| px[i * 4] as f64)
                .sum();
            light[k] = sum / floor_px as f64;
        }
        let [with, without] = light;
        assert!(
            without > 60.0,
            "{what}: the floor under it is dark even without a caster ({without:.0})"
        );
        assert!(
            with < 0.75 * without,
            "{what}: no shadow on the floor (red {with:.0} with the caster, {without:.0} without)"
        );
    }
}

/// Fur and hair under MSAA in the browser: the frame stays multisampled, and
/// the same strands draw the same light as at one sample.
#[wasm_bindgen_test]
async fn fur_and_hair_multisample() {
    use threers::wasm::{WebCamera, WebColor, WebFurMaterial, WebGeometry, WebLight, WebScene};
    const N: u32 = 96;
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let mut camera = WebCamera::perspective(40.0, 1.0, 0.1, 50.0);
    camera.set_position(0.0, 0.0, 5.0);
    camera.look_at(0.0, 0.0, 0.0);
    // Summed differences between neighbouring pixels: what aliasing adds.
    let roughness = |px: &[u8]| -> f64 {
        let mut tv = 0.0;
        for y in 0..N - 1 {
            for x in 0..N - 1 {
                let i = ((y * N + x) * 4) as usize;
                for c in 0..3 {
                    tv += (px[i + c] as f64 - px[i + 4 + c] as f64).abs();
                    tv += (px[i + c] as f64 - px[i + (N * 4) as usize + c] as f64).abs();
                }
            }
        }
        tv
    };
    // The light drawn: the same strands drawn the same, multisampled or not —
    // thin ones as more, fainter pixels rather than fewer, solid ones.
    let light = |px: &[u8]| -> f64 {
        px.chunks(4)
            .map(|p| p[0] as f64 + p[1] as f64 + p[2] as f64)
            .sum()
    };
    for (what, hair) in [("fur", false), ("hair", true)] {
        let mut out = Vec::new();
        for samples in [1u32, 4] {
            let mut fur = WebFurMaterial::new();
            fur.set_length(0.35);
            fur.set_density(70.0);
            fur.set_lod(false);
            fur.set_colors(
                &WebColor::new(0.2, 0.1, 0.05),
                &WebColor::new(1.0, 0.9, 0.7),
            );
            let mut scene = WebScene::new();
            scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
            scene.add_light(&WebLight::ambient(&WebColor::new(1.0, 1.0, 1.0), 2.0));
            if hair {
                fur.set_gravity(0.0, -1.0, 0.0);
                scene.add_hair(
                    &WebGeometry::sphere(0.9, 32, 16),
                    &fur,
                    5000,
                    6,
                    0.006,
                    Some(0.0),
                );
            } else {
                scene.add_fur(&WebGeometry::sphere(1.0, 48, 24), &fur);
            }
            let target = threers::wasm::WebRenderTarget::new(&renderer, N, N);
            target.set_samples(samples);
            renderer.set_render_target(target.id);
            renderer.render(&mut scene, &camera);
            let multisampled = renderer.last_frame_multisampled();
            renderer.set_render_target(0);
            let px = read_target(&renderer, target.id, N, N).await;
            out.push((multisampled, roughness(&px), light(&px)));
        }
        let [(m1, tv1, e1), (m4, tv4, e4)] = [out[0], out[1]];
        assert!(
            !m1 && m4,
            "{what}: multisampled {m1} at one sample, {m4} at four"
        );
        // In a browser fur and hair cut their strands at the pixel's centre
        // even under MSAA (per-sample coverage would cost a per-sample
        // shading there — `sample_mask_is_cheap`): the edges of geometry are
        // smoothed, the strands drawn as at one sample. So no smoother here;
        // natively, `tests/fur.rs` measures that.
        let _ = (tv1, tv4);
        assert!(
            (e4 - e1).abs() < 0.2 * e1,
            "{what}: light drawn {e1:.0} at one sample, {e4:.0} at four"
        );
    }
}

/// Fur and hair drawn straight to the canvas (not a render target), with the
/// renderer's MSAA on and off: the frame — floor, background and coat —
/// reaches the canvas, and the coat is where it should be.
#[wasm_bindgen_test]
async fn fur_draws_to_the_canvas() {
    use threers::wasm::{
        WebCamera, WebColor, WebEuler, WebFurMaterial, WebGeometry, WebLight, WebMaterial, WebMesh,
        WebScene, WebVector3,
    };
    let (shared, canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let (cw, ch) = (canvas.width(), canvas.height());
    for (what, hair) in [("fur", false), ("hair", true)] {
        for samples in [1u32, 4] {
            let mut scene = WebScene::new();
            scene.set_background(&WebColor::new(0.0, 0.0, 1.0));
            scene.add_light(&WebLight::ambient(&WebColor::new(1.0, 1.0, 1.0), 2.0));
            let floor = scene.add(&WebMesh::new(
                &WebGeometry::box_(20.0, 0.1, 20.0),
                &WebMaterial::basic(&WebColor::new(0.0, 1.0, 0.0)),
            ));
            scene.set_transform(
                &floor,
                &WebVector3::new(0.0, -2.0, 0.0),
                &WebEuler::new(0.0, 0.0, 0.0),
            );
            let mut fur = WebFurMaterial::new();
            fur.set_length(0.4);
            fur.set_colors(&WebColor::new(1.0, 0.2, 0.2), &WebColor::new(1.0, 0.3, 0.3));
            if hair {
                scene.add_hair(&WebGeometry::sphere(1.0, 24, 12), &fur, 3000, 4, 0.03, None);
            } else {
                scene.add_fur(&WebGeometry::sphere(1.0, 32, 16), &fur);
            }
            scene.step_fur(0.0, 1.0 / 60.0);
            let mut camera = WebCamera::perspective(40.0, cw as f32 / ch as f32, 0.1, 50.0);
            camera.set_position(0.0, 0.0, 5.0);
            camera.look_at(0.0, 0.0, 0.0);
            renderer.set_msaa(samples);
            renderer.render(&mut scene, &camera);
            let centre = canvas_pixel(&canvas, cw / 2, ch / 2);
            let sky = canvas_pixel(&canvas, cw / 2, ch / 20);
            let ground = canvas_pixel(&canvas, cw / 2, ch - ch / 20);
            renderer.set_msaa(1);
            assert!(
                sky[2] > 200 && sky[0] < 40,
                "{what}, {samples} sample(s): the background reads {sky:?}"
            );
            assert!(
                ground[1] > 150,
                "{what}, {samples} sample(s): the floor reads {ground:?}"
            );
            // Red, the coat's colour (lit brighter or darker, not blue) —
            // across the middle row for hair, whose strands at the very
            // centre point at the camera, edge-on.
            let red = |p: [u8; 3]| p[0] > 120 && p[0] as i32 > p[2] as i32 + 60;
            let coat = if hair {
                (0..20)
                    .map(|k| canvas_pixel(&canvas, cw / 4 + k * cw / 40, ch / 2))
                    .filter(|p| red(*p))
                    .count()
                    >= 3
            } else {
                red(centre)
            };
            assert!(
                coat,
                "{what}, {samples} sample(s): no coat (centre {centre:?})"
            );
        }
    }
}

/// Fur and hair through the scene graph, as the three.js shim builds them
/// (`createFur`, `createHair`, `setChildren`), drawn to the canvas: the frame
/// — background and coat — reaches it. (Their shadow casters write neither
/// colour nor depth; on WebGL2 drawing one in the colour pass blacked out
/// the whole frame.)
#[wasm_bindgen_test]
async fn graph_fur_draws_to_the_canvas() {
    use threers::wasm::{WebCamera, WebColor, WebFurMaterial, WebGeometry, WebLight, WebScene};
    let (shared, canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let (cw, ch) = (canvas.width(), canvas.height());
    for hair in [false, true] {
        let mut scene = WebScene::new();
        scene.set_background(&WebColor::new(0.0, 0.0, 1.0));
        let mut fur = WebFurMaterial::new();
        fur.set_length(0.4);
        fur.set_colors(&WebColor::new(1.0, 0.2, 0.2), &WebColor::new(1.0, 0.3, 0.3));
        let coat = if hair {
            scene.create_hair(&WebGeometry::sphere(1.0, 24, 12), &fur, 4000, 4, 0.04, None)
        } else {
            scene.create_fur(&WebGeometry::sphere(1.0, 32, 16), &fur)
        };
        scene.set_children(&scene.root_handle(), vec![coat.id()]);
        scene.add_light(&WebLight::ambient(&WebColor::new(1.0, 1.0, 1.0), 2.0));
        scene.step_fur(0.0, 1.0 / 60.0);
        let mut camera = WebCamera::perspective(40.0, cw as f32 / ch as f32, 0.1, 50.0);
        camera.set_position(0.0, 0.0, 5.0);
        camera.look_at(0.0, 0.0, 0.0);
        renderer.render(&mut scene, &camera);
        let what = if hair { "hair" } else { "fur" };
        let sky = canvas_pixel(&canvas, cw / 2, ch / 20);
        assert!(
            sky[2] > 200 && sky[0] < 40,
            "{what}: the background reads {sky:?}"
        );
        let red = |p: [u8; 3]| p[0] > 120 && p[0] as i32 > p[2] as i32 + 60;
        let coat_px = (0..20)
            .map(|k| canvas_pixel(&canvas, cw / 4 + k * cw / 40, ch / 2))
            .filter(|p| red(*p))
            .count();
        assert!(coat_px >= 3, "{what}: no coat across the middle");
    }
}
