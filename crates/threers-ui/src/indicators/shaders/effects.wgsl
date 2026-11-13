// The built-in fills, by `PFX_KIND` (set before this), on the shader
// library's `ui_fire` and the rest.
fn pfx_fx(along: f32, up: f32, half: f32, env: f32, cover: f32, u: f32, now: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>) -> vec4<f32> {
    if (PFX_KIND == 1) {
        return ui_fire(along, up, half, env, cover, u, now, c0, c1, c2);
    }
    if (PFX_KIND == 2) {
        return ui_water(along, up, half, env, cover, u, now, c0, c1, c2);
    }
    if (PFX_KIND == 3) {
        return ui_plasma(along, up, half, env, cover, u, now, c0, c1, c2);
    }
    return ui_frost(along, up, half, env, cover, u, now, c0, c1, c2);
}
