//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Image-based lighting.
// ---------------------------------------------------------------------------

/// Direction of the texel at `(u, v)` on cube face `face`, `u`/`v` in `-1..1`.
pub(crate) fn cube_dir(face: usize, u: f32, v: f32) -> Vector3 {
    match face {
        0 => Vector3::new(1.0, -v, -u),
        1 => Vector3::new(-1.0, -v, u),
        2 => Vector3::new(u, 1.0, v),
        3 => Vector3::new(u, -1.0, -v),
        4 => Vector3::new(u, -v, 1.0),
        _ => Vector3::new(-u, -v, -1.0),
    }
}

/// A sky-and-ground cube for image-based lighting, prefiltered for roughness.
///
/// The renderer reads this for both indirect diffuse and the specular
/// reflection, and a curtain wall is mostly the second one: with no
/// environment, `metalness` has nothing to return and a glass tower renders as
/// a dark slab. f32 on purpose — the sun is orders of magnitude brighter than
/// the sky it sits in, and that ratio is the whole reason an HDR environment
/// exists. Read it with a tone-mapped renderer or it clips to white.
pub(crate) fn sky_environment(sky: &Sky, size: u32) -> Arc<CubeTexture> {
    let n = size as usize;
    let sun = sky.sun_dir;
    // Radiance of the solar disc relative to diffuse white. Falls to nothing as
    // the sun sets, so dusk reflections come from the sky rather than a disc
    // burning through the horizon.
    let disc = 220.0 * sky.elevation.max(0.0).powf(0.7);
    let zenith = scale_color(sky.background, 0.82);
    let horizon = mix_color(scale_color(sky.background, 1.35), Color::WHITE, 0.10 * sky.day);
    let ground = scale_color(Color::from_hex(C_TERRAIN), 0.10 + 0.55 * sky.day);
    // After dark the ground half of the sphere is the city lighting itself,
    // which is what a night-time window actually reflects.
    let city_glow = scale_color(Color::from_hex(0xff9a44), 0.16 * sky.lights);

    let mut faces: Vec<Vec<f32>> = Vec::with_capacity(6);
    for face in 0..6 {
        let mut data = vec![0.0f32; n * n * 4];
        for y in 0..n {
            for x in 0..n {
                let u = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let v = (y as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                // NOTE the X flip: the PMREM atlas builder samples the source
                // cube at (-x, y, z), so a sun authored at direction `d` is
                // later seen at (-d.x, d.y, d.z) — mirrored away from the
                // DirectionalLight it is supposed to agree with.
                let dc = cube_dir(face, u, v).normalize();
                let d = Vector3::new(-dc.x, dc.y, dc.z);

                let up = smoothstep(-0.03, 0.03, d.y);
                let sky_col = mix_color(horizon, zenith, smoothstep(0.0, 0.62, d.y));
                let below = mix_color(ground, city_glow, smoothstep(-0.35, 0.0, d.y));
                let mut c = mix_color(below, sky_col, up);

                let cos_sun = d.dot(sun);
                if cos_sun > 0.9998 {
                    c = scale_color(sky.sun_color, disc);
                } else if cos_sun > 0.0 {
                    // Two lobes: a tight one for the aureole, a broad one for
                    // the general brightening of the sky around the sun.
                    let halo = cos_sun.powf(2200.0) * 12.0 + cos_sun.powf(18.0) * 0.55;
                    c = Color::new(
                        c.r + sky.sun_color.r * halo * sky.day,
                        c.g + sky.sun_color.g * halo * sky.day,
                        c.b + sky.sun_color.b * halo * sky.day,
                    );
                }

                let i = (y * n + x) * 4;
                data[i] = c.r;
                data[i + 1] = c.g;
                data[i + 2] = c.b;
                data[i + 3] = 1.0;
            }
        }
        faces.push(data);
    }
    let faces: [Vec<f32>; 6] = faces.try_into().expect("six cube faces");
    let cube = CubeTexture::new_f32(size, faces);
    Arc::new(PmremGenerator::generate_pmrem(&cube, size))
}
