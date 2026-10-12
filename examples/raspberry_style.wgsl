/// Pale base, golden through the middle, then the dark dried stigma. Four
/// stops: with two it reads as a plastic bristle, because what makes a dried
/// thread look dried is that the darkening is not linear in its length.
fn ramp(t: f32) -> vec3<f32> {
    var c = mix(
        vec3<f32>(0.62, 0.36, 0.13),
        vec3<f32>(0.72, 0.50, 0.17),
        smoothstep(0.0, 0.12, t),
    );
    c = mix(c, vec3<f32>(0.52, 0.27, 0.06), smoothstep(0.12, 0.62, t));
    c = mix(c, vec3<f32>(0.20, 0.07, 0.02), smoothstep(0.62, 0.90, t));
    c = mix(c, vec3<f32>(0.030, 0.012, 0.006), smoothstep(0.90, 1.0, t));
    return c;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let exposure = u_data.data[0].x;
    let brightness = u_data.data[0].y;
    let specular = u_data.data[0].z;
    let berry_center = u_data.data[1].xyz;

    let t = in.vertex_color.r;
    let dryness = in.vertex_color.g;
    let coverage = in.vertex_color.a;

    // Round profile: a style is a cylinder, not a strip.
    let across = in.uv.x;
    let edge = 1.0 - across * across * 0.55;
    let alpha = coverage * edge;
    if (alpha < 0.02) { discard; }

    let n = normalize(in.world_normal);
    let v = normalize(frame.camera_position.xyz - in.world_pos);
    let base = mix(ramp(t), vec3<f32>(0.045, 0.02, 0.012), smoothstep(0.55, 1.0, dryness) * 0.85);

    // The berry is in the way of roughly the hemisphere behind the thread, and
    // the root is down in a crevice. Without these two the styles are lit from
    // every side and read as bright hooks stuck onto the fruit rather than as
    // threads growing out of it — which is the single biggest thing separating
    // a convincing one from an unconvincing one.
    let outward = normalize(in.local_pos - berry_center);
    let occ = mix(0.35, 1.0, sat(dot(n, outward) * 0.5 + 0.5))
        * mix(0.55, 1.0, smoothstep(0.0, 0.25, t));

    // A fibre this thin is lit most of the way round it, so the irradiance is
    // blended towards the direction the thread points away from the berry.
    let irr = mix(sh_irradiance(n), sh_irradiance(outward), 0.45);
    var col = base * irr * (1.0 / 3.14159265) * occ * brightness;

    // Fresnel, and a strong one: on a cylinder almost every visible point is
    // near grazing, so this is most of what you actually see.
    let n_o_v = sat(dot(n, v));
    let f = 0.04 + 0.96 * pow(1.0 - n_o_v, 5.0);
    col = col + sample_env_cube(reflect(-v, n), 0.5) * f * occ * specular * 0.6;

    return vec4<f32>(framebuffer_encode(pbr_neutral(col * exposure)), alpha);
}
