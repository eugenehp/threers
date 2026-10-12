// The skin of the berry: one drupelet per `vertex_color.a`, shaded as a
// translucent bead rather than as a surface.
//
// Compiled against threers' standard preamble, so `VsOut`, `frame`, `mesh`,
// `sample_env_cube`, `linear_to_srgb` and `framebuffer_encode` are all in
// scope, as are the user bindings `u_data`, `u_s0` and `u_s1`.
//
//   u_s0[i*8 ..]  centre.xyz, radius, outward.xyz, random        (per drupelet)
//   u_s1[0..27]   incident radiance, SH9, model space
//   u_s1[27..36]  three soft-box directions (towards the light)
//   u_s1[36..45]  three soft-box irradiances
//   u_s1[45..48]  extinction per cm, per channel
//   u_s1[48..63]  albedo ripe / bright / dark / cavity, then the sheen colour
//   u_s1[63..66]  the inner lamp's colour

// The skin's own slice of `u_s1`. The bands the other materials share — the
// SH9 and the three soft boxes — are read through `raspberry_common.wgsl`.
fn sigma() -> vec3<f32> { return s1v(45u); }
fn albedo_ripe() -> vec3<f32> { return s1v(48u); }
fn albedo_bright() -> vec3<f32> { return s1v(51u); }
fn albedo_dark() -> vec3<f32> { return s1v(54u); }
fn albedo_inner() -> vec3<f32> { return s1v(57u); }
fn sheen_color() -> vec3<f32> { return s1v(60u); }
fn lamp_color() -> vec3<f32> { return s1v(63u); }

// ---- procedural detail ----------------------------------------------------
// The pores and epidermal cells. A baked detail map would be better and is
// what a production asset uses; value noise costs nothing and this example
// downloads nothing.

fn hash3(p: vec3<f32>) -> f32 {
    let q = fract(p * 0.3183099 + vec3<f32>(0.1, 0.2, 0.3));
    let r = q * 17.0;
    return fract(r.x * r.y * r.z * (r.x + r.y + r.z));
}

fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = hash3(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

/// Karis' analytic fit of the split-sum environment BRDF.
fn env_brdf(roughness: f32, n_o_v: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_o_v)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let sss_w = u_data.data[0].x;
    let bead_w = u_data.data[0].y;
    let see_w = u_data.data[0].z;
    let seed_w = u_data.data[0].w;
    let spec_w = u_data.data[1].x;
    let f0 = u_data.data[1].y;
    let rough_scale = u_data.data[1].z;
    let rough_bias = u_data.data[1].w;
    let detail_w = u_data.data[2].x;
    let detail_tile = u_data.data[2].y;
    let sheen_w = u_data.data[2].z;
    let ao_power = u_data.data[2].w;
    let env_scale = u_data.data[3].x;
    let saturation = u_data.data[3].y;
    let exposure = u_data.data[3].z;
    let reflection = u_data.data[3].w;
    let berry_center = u_data.data[4].xyz;
    let inner_wall = u_data.data[4].w;
    let inner_pos = u_data.data[5].xyz;
    let tint = u_data.data[6].xyz;
    let trans_depth = u_data.data[6].w;
    let backdrop = u_data.data[7].xyz;
    let rough_lo = u_data.data[8].x;
    let rough_hi = u_data.data[8].y;

    // ---- which drupelet, and where on it -----------------------------------
    let mask = in.vertex_color;
    let free = mask.r;
    let depth = mask.g;
    let rnd = mask.b;
    let di = u32(max(mask.a, 0.0) + 0.5) * 8u;
    let center = vec3<f32>(u_s0[di], u_s0[di + 1u], u_s0[di + 2u]);
    let bead_r = u_s0[di + 3u];
    let outward = vec3<f32>(u_s0[di + 4u], u_s0[di + 5u], u_s0[di + 6u]);

    let p_model = in.local_pos;
    let to_c = center - p_model;
    let r_local = max(length(to_c), 1e-4);
    // Bead normal: outward from the sac's own centre.
    let n_bead = -to_c / r_local;
    // 1 on the dome's crown, 0 where it dives into a crevice.
    let dome = sat(dot(n_bead, outward));

    let model_rot = mat3x3<f32>(
        mesh.model[0].xyz,
        mesh.model[1].xyz,
        mesh.model[2].xyz,
    );
    let n_base = normalize(in.world_normal);
    // The geometric normal in model space, for the parts that work in the
    // berry's own frame.
    let n_base_m = normalize(transpose(model_rot) * n_base);

    // ---- detail ------------------------------------------------------------
    let q = p_model / max(detail_tile, 1e-4);
    let e = 0.35;
    let h = vnoise(q * 2.2);
    let hx = vnoise(q * 2.2 + vec3<f32>(e, 0.0, 0.0));
    let hy = vnoise(q * 2.2 + vec3<f32>(0.0, e, 0.0));
    let hz = vnoise(q * 2.2 + vec3<f32>(0.0, 0.0, e));
    let grad = vec3<f32>(hx - h, hy - h, hz - h) / e;
    // Only the part of the gradient along the surface bends the normal.
    let tang_grad = grad - n_base_m * dot(grad, n_base_m);
    let n_model = normalize(n_base_m - tang_grad * detail_w * 0.30);
    let dh = 0.5 + h;

    let n = normalize(model_rot * n_model);
    let n_g = n_base;
    let v = normalize(frame.camera_position.xyz - in.world_pos);
    let n_o_v = sat(dot(n, v));

    // ---- the maps, computed instead of sampled -----------------------------
    // `free` is the baked ambient occlusion of a real asset: how much of the
    // sky this point can see, given the neighbours it is wedged against.
    let ao = pow(sat(mix(0.06, 1.0, free)), ao_power);
    let thickness = sat(mix(0.35, 1.0, free) * mix(0.8, 1.0, dome));
    var roughness = mix(rough_lo, rough_hi, sat(0.35 + 0.5 * (1.0 - dome) + 0.3 * (rnd - 0.5)));
    roughness = clamp(roughness * rough_scale + rough_bias + (dh - 1.0) * 0.06, 0.045, 1.0);

    var albedo = mix(albedo_dark(), albedo_ripe(), smoothstep(0.08, 0.70, dome));
    albedo = mix(albedo, albedo_bright(), smoothstep(0.55, 1.0, dome) * (0.30 + 0.70 * rnd));
    // Per-drupelet ripeness: a few are nearly black cherry, a few nearly
    // orange, and the spread is what keeps 117 of them from looking moulded.
    let ripeness = rnd * rnd * (3.0 - 2.0 * rnd);
    albedo = albedo * mix(0.42, 1.45, ripeness);
    albedo = albedo * vec3<f32>(1.0, mix(0.55, 2.1, ripeness), mix(0.75, 1.5, ripeness));
    // Torn flesh on the cavity lining is pale and almost opaque.
    albedo = mix(albedo, albedo_inner(), smoothstep(0.45, 0.95, depth));

    // ---- diffuse + subsurface ---------------------------------------------
    // Albedo × wrapped irradiance, with *coloured* ambient occlusion: the
    // channel the flesh does not absorb is the one that survives in a crease,
    // so creases go deep red rather than grey. A single grey occlusion factor
    // is the usual reason a berry reads as plastic.
    var irr = sh_irradiance(n_base_m);
    let amb = PI * 0.282095 * sh_coeff(0u);
    irr = irr + max(amb - irr, vec3<f32>(0.0)) * vec3<f32>(0.55, 0.12, 0.10);
    let ao_c = pow(vec3<f32>(ao), vec3<f32>(1.0) - 0.75 * albedo / max(albedo.r, 1e-3));
    let sss = pow(albedo, vec3<f32>(0.85, 1.25, 1.25)) * irr * (0.75 / PI) * ao_c;

    // ---- translucency ------------------------------------------------------
    // A drupelet is a sac of juice a couple of millimetres across. Treating it
    // as one gives three things a surface model cannot: light that enters on
    // the lit side and leaves past the terminator, the soft boxes seen through
    // the sacs on the limb, and the pale seed inside.
    let v_m = normalize(transpose(model_rot) * v);
    let out_dir = normalize(center - berry_center);
    let sig = sigma();
    let open = 0.3 + 0.7 * ao;

    // (a) light in on the lit side, out past the terminator
    var bead = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        let l_m = normalize(transpose(model_rot) * light_dir(i));
        let nl = dot(n_bead, l_m);
        // Light always crosses some flesh, even where the bead is edge on.
        let chord = 2.0 * bead_r * max(-nl, 0.0) + 0.10;
        let past = smoothstep(0.45, -0.30, nl);
        // ...and it has to be able to get at this drupelet at all.
        let reach = smoothstep(-0.45, 0.20, dot(out_dir, l_m));
        bead = bead + light_irr(i) * exp(-sig * chord) * past * reach;
    }
    bead = bead * (0.07 * bead_w * open);

    // (b) looking through a sac at whatever is behind it
    let tr = refract(-v_m, n_base_m, 1.0 / 1.33);
    let chord_v = max(2.0 * dot(to_c, tr), 0.0);
    let exit_p = p_model + tr * chord_v;
    var leave = smoothstep(-0.15, 0.45, dot(normalize(exit_p - berry_center), tr));
    leave = leave * smoothstep(0.30, 0.85, 1.0 - sat(dot(out_dir, v_m)));
    leave = leave * smoothstep(0.12, 0.60, ao);
    var through = vec3<f32>(0.0);
    if (leave * see_w > 0.002) {
        let behind = sample_env_cube(normalize(model_rot * tr), 0.6) * env_scale + backdrop;
        through = behind * exp(-sig * (chord_v + 0.06)) * leave * (0.25 + 0.75 * free) * see_w * 0.55;
    }

    // (c) the seed, a pale blur one bead deep
    let seed_r = 0.085;
    let sb = dot(to_c, tr);
    let sh2 = sb * sb - (r_local * r_local - seed_r * seed_r);
    let seed_hit = smoothstep(0.0, seed_r * seed_r * 0.9, sh2) * step(0.0, sb);
    let sd = max(sb - sqrt(max(sh2, 0.0)), 0.0);
    let seed_light = sh_irradiance(n_bead) * (0.5 / PI) * open + bead * 2.0;
    let seed = vec3<f32>(0.80, 0.56, 0.30) * seed_light * exp(-sig * (sd + 0.03)) * seed_hit * seed_w;

    var glow = bead + through + seed;
    let see_through = leave * sat(see_w);

    // A small lamp in the hollow core. On the cavity lining it is ordinary
    // direct light; everywhere else it has to cross the flesh to get here, so
    // it arrives filtered and brightest in the creases, where the wall is
    // thinnest. A uniform branch, so the lamp can be switched without a
    // recompile.
    let lamp = lamp_color();
    if (lamp.r + lamp.g + lamp.b > 0.0) {
        let to_l = inner_pos - p_model;
        let d2 = dot(to_l, to_l);
        let facing = dot(n_base_m, to_l * inverseSqrt(max(d2, 1e-6)));
        let flesh = max(inner_wall - depth * 0.7, 0.0) * (0.6 + 0.8 * thickness);
        let seed_shadow = 1.0 - 0.4 * smoothstep(0.80, 1.0, dome);
        let crossed = exp(-sig * flesh) * (0.30 + 0.20 * max(-facing, 0.0)) * seed_shadow;
        let lit = albedo * max(facing, 0.0) * smoothstep(0.3, 0.6, depth) * 0.14;
        glow = glow + lamp * (crossed + lit) / (PI * (d2 + 0.03));
    }

    var diffuse = (sss * sss_w * (1.0 - 0.35 * see_through) + glow) * tint;
    let lum = dot(diffuse, vec3<f32>(0.2126, 0.7152, 0.0722));
    diffuse = max(mix(vec3<f32>(lum), diffuse, saturation), vec3<f32>(0.0));

    // ---- specular: the prefiltered studio ----------------------------------
    var refl = reflect(-v, n);
    refl = normalize(mix(refl, n, roughness * roughness));
    let env = sample_env_cube(refl, roughness) * env_scale;
    let dfg = env_brdf(roughness, n_o_v);
    // Specular occlusion (Lagarde) and a horizon fade against the smooth
    // normal, so the detail normal cannot reflect the floor through the berry.
    var so = sat(pow(n_o_v + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);
    let horizon = sat(1.0 + dot(refl, n_g));
    so = so * horizon * horizon;
    // The berry itself is in the way of anything reflected from behind it.
    so = so * smoothstep(-0.35, 0.15, dot(normalize(transpose(model_rot) * refl), out_dir));
    let cavity = mix(1.0, 0.55 + 0.45 * dh, sat(detail_w));
    let specular = env * (f0 * dfg.x + dfg.y) * so * cavity * spec_w * reflection;

    // ---- velvet from the hairs too fine to draw -----------------------------
    // Lit by what actually reaches this spot rather than by a plain irradiance
    // lookup, so the sheen goes dark in a crease like the hairs do.
    let fres = pow(1.0 - n_o_v, 3.0);
    let local_light = sss.r * 2.2;
    let sheen = sheen_color() * local_light * fres * sheen_w * (0.4 + 0.6 * ao);

    let hdr = (diffuse + specular + sheen) * exposure;

    let dbg = u_data.data[9].x;
    if (dbg > 0.5) {
        var t = vec3<f32>(0.0);
        if (dbg < 1.5) { t = diffuse * exposure; }
        else if (dbg < 2.5) { t = specular * exposure; }
        else if (dbg < 3.5) { t = sheen * exposure; }
        else if (dbg < 4.5) { t = glow * exposure; }
        else if (dbg < 5.5) { t = vec3<f32>(ao); }
        else if (dbg < 6.5) { t = vec3<f32>(dome); }
        else if (dbg < 7.5) { t = vec3<f32>(free); }
        else if (dbg < 8.5) { t = env * exposure; }
        else { t = n * 0.5 + vec3<f32>(0.5); }
        return vec4<f32>(framebuffer_encode(t), 1.0);
    }
    return vec4<f32>(framebuffer_encode(pbr_neutral(hdr)), 1.0);
}
