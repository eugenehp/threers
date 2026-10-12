// The dried styles — the wiry tails left where each drupelet's stigma dried.
//
// Drawn as camera-facing ribbons rather than as tubes, for the same reason the
// fuzz is: a style tapers to a point, so somewhere along its length it falls
// under a pixel wide. A solid tube has no way to say "a third of this pixel" —
// the rasteriser either covers the centre or it does not — so the tip breaks
// into dashes and the taper reads as a hard stop. Expanded here instead, never
// below a pixel, fading by exactly the factor it was widened by.
//
// `vertex_color` arrives as (t along the strand, dryness, random, radius) and
// leaves as (t, dryness, random, coverage). `uv` is (side, t).

@vertex
fn vs_custom(in: VsIn) -> VsOut {
    var out = standard_vertex(in);
    // The strand's tangent travels in the normal slot; a fibre has no normal of
    // its own, only a tangent and a whole circle of normals around it.
    let tangent = normalize(out.world_normal);
    let to_eye = normalize(frame.camera_position.xyz - out.world_pos);
    var side = cross(tangent, to_eye);
    if (length(side) < 1e-5) {
        side = normalize(cross(tangent, vec3<f32>(0.0, 1.0, 0.0)));
    } else {
        side = normalize(side);
    }

    let radius = in.color.a;
    let ppu = pixels_per_unit(out.world_pos, side);
    let w_px = 2.0 * radius * ppu;
    let w = max(w_px, 1.2);
    let coverage = clamp(w_px / w, 0.0, 1.0);

    out.world_pos = out.world_pos + side * in.uv.x * (0.5 * w / ppu);
    out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);
    // The shading normal for a fibre: the one in its circle of normals that
    // faces the camera.
    out.world_normal = normalize(to_eye - tangent * dot(to_eye, tangent));
    out.vertex_color = vec4<f32>(in.color.rgb, coverage);
    return out;
}
