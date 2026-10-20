
// ---------------------------------------------------------------------------
// Post-fx kinds 40 and up: the shaders of three.js r165's post-processing
// passes, ported for the composer in `crate::postprocessing` and run through
// `Renderer::run_postfx`. Appended to POSTFX_SHADER, so every binding and
// helper above is in scope; `postfx_fragment` hands kinds >= 40 to
// `postfx_fragment_ext` at the bottom.
//
// Conventions, the same for every kind here:
//
// * `uv` is texture space (v = 0 at the top row). A three.js shader's `vUv`
//   is GL's (v = 0 at the bottom); a port that does arithmetic in `vUv` — an
//   edge walk, a ring of samples, `gl_FragCoord` — works in `gl_uv(uv)` and
//   converts back with `gl_uv` for each fetch of a rendered target. Data
//   textures (SMAA's tables, a LUT) are uploaded row 0 first, which is where
//   GL puts row 0 too, so their coordinates are used as they are.
// * Inputs are linear: `run_postfx` binds every target through a view that
//   decodes, so a half-float and an 8-bit target read alike.
// * What a kind returns is three.js's `gl_FragColor`. Into a float target
//   that is stored as is. Onto the display a `ShaderMaterial` writes it
//   clamped and unconverted (`fx_raw`), and a `MeshBasicMaterial` — or a
//   shader that includes `colorspace_fragment` — converts to sRGB first
//   (`fx_display`).
// ---------------------------------------------------------------------------

const FX_PI: f32 = 3.141592653589793;
const FX_PI2: f32 = 6.283185307179586;
const FX_EPSILON: f32 = 1e-6;

fn fx_raw(c: vec4<f32>) -> vec4<f32> {
    if (postfx_linear_out()) {
        return c;
    }
    return clamp(c, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn fx_display(c: vec4<f32>) -> vec4<f32> {
    if (postfx_linear_out()) {
        return c;
    }
    return vec4<f32>(postfx_srgb_encode(c.rgb), clamp(c.a, 0.0, 1.0));
}

// `input_tex` at a texture-space position, filtered.
fn fx_tex(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(input_tex, input_sampler, uv, 0.0);
}

// `input_tex` at a GL `vUv` position, filtered.
fn fx_tex_gl(p: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(input_tex, input_sampler, gl_uv(p), 0.0);
}

// `input_tex` with nearest filtering, for a three.js target whose texture is
// `NearestFilter`: the texel under `uv`.
fn fx_load(uv: vec2<f32>) -> vec4<f32> {
    let dim = textureDimensions(input_tex, 0);
    let c = clamp(
        vec2<i32>(floor(uv * vec2<f32>(dim))),
        vec2<i32>(0),
        vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1),
    );
    return textureLoad(input_tex, c, 0);
}

// `depth_tex` / `normal_tex` read through the aux sampler.
fn fx_depth_tex(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(depth_tex, depth_sampler, uv, 0.0);
}

fn fx_normal_tex(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(normal_tex, depth_sampler, uv, 0.0);
}

// The hardware depth bound as `hw_depth_tex`, at a texture-space position.
fn fx_hw_depth(uv: vec2<f32>) -> f32 {
    return sample_postfx_depth(uv, true);
}

// A second depth buffer, bound in the `noise_tex` slot: the texel under
// `uv`, sampled nearest at its centre (see `sample_postfx_depth` for why a
// sample rather than a fetch).
fn fx_noise_depth(uv: vec2<f32>) -> f32 {
    let dim = textureDimensions(noise_tex, 0);
    let c = clamp(
        vec2<i32>(uv * vec2<f32>(dim)),
        vec2<i32>(0),
        vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1),
    );
    let centre = (vec2<f32>(c) + vec2<f32>(0.5)) / vec2<f32>(dim);
    let d = textureSampleLevel(noise_tex, noise_sampler, centre, 0.0).r;
    // The renderer drew it too: reversed with bit 16, like `hw_depth_tex`.
    return select(d, 1.0 - d, postfx_flag(16u));
}

// One float of the uniform weight array, four to a `vec4`.
fn fx_weight(i: u32) -> f32 {
    let v = ssao_kernel_buf.samples[i / 4u];
    let j = i % 4u;
    if (j == 0u) { return v.x; }
    if (j == 1u) { return v.y; }
    if (j == 2u) { return v.z; }
    return v.w;
}

// three.js `common` `rand`: the `mod(dt, PI)` variant.
fn fx_three_rand(uv: vec2<f32>) -> f32 {
    let dt = dot(uv, vec2<f32>(12.9898, 78.233));
    let sn = dt - FX_PI * floor(dt / FX_PI);
    return fract(sin(sn) * 43758.5453);
}

fn fx_view_z(depth: f32, near: f32, far: f32, perspective: bool) -> f32 {
    if (perspective) {
        return perspective_depth_to_view_z(depth, near, far);
    }
    // three.js `orthographicDepthToViewZ`.
    return depth * (near - far) - near;
}

// An 8-bit `UnsignedByteType` target's storage: three.js's SSR targets are
// those, and their reflections are quantized in them.
fn fx_q8(c: vec4<f32>) -> vec4<f32> {
    return floor(clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0 + vec4<f32>(0.5)) / 255.0;
}

// ---- 40, 41: copies ------------------------------------------------------

// 40: CopyShader, `opacity * texel` (params2.x = opacity).
fn fx_copy(uv: vec2<f32>) -> vec4<f32> {
    return fx_raw(fx_tex(uv) * u.params2.x);
}

// 41: a MeshBasicMaterial copy — the map, alpha times params2.x, into the
// output colour space.
fn fx_basic_copy(uv: vec2<f32>) -> vec4<f32> {
    let t = fx_tex(uv);
    return fx_display(vec4<f32>(t.rgb, t.a * u.params2.x));
}

// ---- 42-44: UnrealBloomPass ----------------------------------------------

// 42: LuminosityHighPassShader. params2: threshold, smoothWidth,
// defaultOpacity; params3.rgb: defaultColor.
fn fx_luminosity_high_pass(uv: vec2<f32>) -> vec4<f32> {
    let texel = fx_tex(uv);
    let v = dot(texel.xyz, vec3<f32>(0.299, 0.587, 0.114));
    let output_color = vec4<f32>(u.params3.xyz, u.params2.z);
    let alpha = smoothstep(u.params2.x, u.params2.x + u.params2.y, v);
    return fx_raw(mix(output_color, texel, alpha));
}

// 43: UnrealBloomPass's separable blur. params2: KERNEL_RADIUS, direction;
// `invSize` is the output's (each mip blurs at its own size).
fn fx_bloom_blur(uv: vec2<f32>, res: vec2<f32>) -> vec4<f32> {
    let kernel = i32(u.params2.x + 0.5);
    let kf = f32(kernel);
    let direction = u.params2.yz;
    let inv_size = 1.0 / res;
    var weight_sum = 0.39894 / kf;
    var diffuse_sum = fx_tex(uv).rgb * weight_sum;
    for (var i: i32 = 1; i < kernel; i = i + 1) {
        let x = f32(i);
        let w = 0.39894 * exp(-0.5 * x * x / (kf * kf)) / kf;
        // Symmetric, so the GL +y needs no flip.
        let uv_offset = direction * inv_size * x;
        let sample1 = fx_tex(uv + uv_offset).rgb;
        let sample2 = fx_tex(uv - uv_offset).rgb;
        diffuse_sum = diffuse_sum + (sample1 + sample2) * w;
        weight_sum = weight_sum + 2.0 * w;
    }
    return fx_raw(vec4<f32>(diffuse_sum / weight_sum, 1.0));
}

// 44: UnrealBloomPass's composite, a few mips at a time: params2.z is the
// first mip, params2.w how many (at most three: `input_tex`, `depth_tex`,
// `normal_tex`). The weight array holds (tint, bloomFactor) per mip;
// params2.xy = bloomStrength, bloomRadius.
fn fx_bloom_composite(uv: vec2<f32>) -> vec4<f32> {
    let first = u32(u.params2.z + 0.5);
    let count = u32(u.params2.w + 0.5);
    var acc = vec4<f32>(0.0);
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
        if (j >= count) {
            break;
        }
        let k = ssao_kernel_buf.samples[first + j];
        let factor = mix(k.w, 1.2 - k.w, u.params2.y);
        var t = fx_tex(uv);
        if (j == 1u) {
            t = fx_depth_tex(uv);
        } else if (j == 2u) {
            t = fx_normal_tex(uv);
        }
        acc = acc + factor * vec4<f32>(k.xyz, 1.0) * t;
    }
    return fx_raw(u.params2.x * acc);
}

// ---- 45, 46: BloomPass ---------------------------------------------------

// 45: ConvolutionShader. params2: uImageIncrement (GL units), KERNEL_SIZE;
// the kernel in the weight array.
fn fx_convolution(uv: vec2<f32>) -> vec4<f32> {
    let increment = u.params2.xy;
    let n = i32(u.params2.z + 0.5);
    var coord = gl_uv(uv) - ((f32(n) - 1.0) / 2.0) * increment;
    var sum = vec4<f32>(0.0);
    for (var i: i32 = 0; i < n; i = i + 1) {
        sum = sum + fx_tex_gl(coord) * fx_weight(u32(i));
        coord = coord + increment;
    }
    return fx_raw(sum);
}

// 46: BloomPass's CombineShader, `strength * texel`.
fn fx_combine(uv: vec2<f32>) -> vec4<f32> {
    return fx_raw(u.params2.x * fx_tex(uv));
}

// ---- 47: AfterimagePass --------------------------------------------------

// 47: AfterimageShader. `input_tex` is tNew, `normal_tex` tOld; params2.x =
// damp.
fn fx_afterimage(uv: vec2<f32>) -> vec4<f32> {
    var texel_old = fx_normal_tex(uv);
    let texel_new = fx_tex(uv);
    texel_old = texel_old * (u.params2.x * max(sign(texel_old - vec4<f32>(0.1)), vec4<f32>(0.0)));
    return fx_raw(max(texel_new, texel_old));
}

// ---- 48-51: OutlinePass --------------------------------------------------

// 48: the selection mask. three.js renders the selected objects with a
// material comparing each fragment's view depth against a depth pass of
// everything else; here the two depth buffers are compared directly — the
// non-selected objects' as `hw_depth_tex`, the selected ones' in `noise_tex`.
// White where nothing is selected (the cleared mask), else
// `(0, hidden, 1, 1)`.
fn fx_outline_mask(uv: vec2<f32>) -> vec4<f32> {
    let selected = fx_noise_depth(uv);
    if (selected >= 1.0) {
        return vec4<f32>(1.0);
    }
    let others = fx_hw_depth(uv);
    let hidden = select(0.0, 1.0, selected > others);
    return vec4<f32>(0.0, hidden, 1.0, 1.0);
}

// 49: OutlinePass's edge detection. params2.rgb: visibleEdgeColor,
// params3.rgb: hiddenEdgeColor; `texSize` is the output's.
fn fx_outline_edge(uv: vec2<f32>, res: vec2<f32>) -> vec4<f32> {
    let inv_size = 1.0 / res;
    let c1 = fx_tex(uv + vec2<f32>(inv_size.x, 0.0));
    let c2 = fx_tex(uv - vec2<f32>(inv_size.x, 0.0));
    let c3 = fx_tex(uv + vec2<f32>(0.0, inv_size.y));
    let c4 = fx_tex(uv - vec2<f32>(0.0, inv_size.y));
    let diff1 = (c1.r - c2.r) * 0.5;
    let diff2 = (c3.r - c4.r) * 0.5;
    let d = length(vec2<f32>(diff1, diff2));
    let a1 = min(c1.g, c2.g);
    let a2 = min(c3.g, c4.g);
    let visibility = min(a1, a2);
    let edge_color = select(u.params3.rgb, u.params2.rgb, 1.0 - visibility > 0.001);
    return fx_raw(vec4<f32>(edge_color, 1.0) * vec4<f32>(d));
}

// 50: OutlinePass's separable blur. params2: kernelRadius, direction,
// MAX_RADIUS; `texSize` is the output's.
fn fx_outline_blur(uv: vec2<f32>, res: vec2<f32>) -> vec4<f32> {
    let kernel_radius = u.params2.x;
    let direction = u.params2.yz;
    let max_radius = i32(u.params2.w + 0.5);
    let inv_size = 1.0 / res;
    let sigma = kernel_radius / 2.0;
    var weight_sum = outline_gaussian_pdf(0.0, sigma);
    var diffuse_sum = fx_tex(uv) * weight_sum;
    let delta = direction * inv_size * kernel_radius / f32(max_radius);
    var uv_offset = delta;
    for (var i: i32 = 1; i <= max_radius; i = i + 1) {
        let x = kernel_radius * f32(i) / f32(max_radius);
        let w = outline_gaussian_pdf(x, sigma);
        diffuse_sum = diffuse_sum + (fx_tex(uv + uv_offset) + fx_tex(uv - uv_offset)) * w;
        weight_sum = weight_sum + 2.0 * w;
        uv_offset = uv_offset + delta;
    }
    return fx_raw(diffuse_sum / weight_sum);
}

// 51: OutlinePass's overlay. `input_tex`: edgeTexture1, `depth_tex`:
// edgeTexture2, `normal_tex`: maskTexture, `noise_tex`: patternTexture;
// params2: edgeStrength, edgeGlow, usePatternTexture.
fn fx_outline_overlay(uv: vec2<f32>) -> vec4<f32> {
    let edge_value1 = fx_tex(uv);
    let edge_value2 = fx_depth_tex(uv);
    let mask_color = fx_normal_tex(uv);
    let pattern_color = textureSampleLevel(noise_tex, noise_sampler, gl_uv(6.0 * gl_uv(uv)), 0.0);
    let visibility = select(0.5, 1.0, 1.0 - mask_color.g > 0.0);
    let edge_value = edge_value1 + edge_value2 * u.params2.y;
    var final_color = u.params2.x * mask_color.r * edge_value;
    if (u.params2.z > 0.5) {
        final_color = final_color + vec4<f32>(visibility * (1.0 - mask_color.r) * (1.0 - pattern_color.r));
    }
    return fx_raw(final_color);
}

// ---- 52: linear depth (SSAODepthShader / SSRDepthShader) ------------------

// 52: `1 - linear depth` of `hw_depth_tex`. params2.x > 0.5 first rounds the
// depth to 16 bits — SSRPass's depth texture is `UnsignedShortType`.
fn fx_depth_view(uv: vec2<f32>) -> vec4<f32> {
    var depth = fx_hw_depth(uv);
    if (u.params2.x > 0.5) {
        depth = floor(depth * 65535.0 + 0.5) / 65535.0;
    }
    let view_z = perspective_depth_to_view_z(depth, u.ssao0.x, u.ssao0.y);
    let lin = view_z_to_ortho_depth(view_z, u.ssao0.x, u.ssao0.y);
    return fx_raw(vec4<f32>(vec3<f32>(1.0 - lin), 1.0));
}

// ---- 53, 54: SAOPass -----------------------------------------------------

// 53: DepthLimitedBlurShader (DEPTH_PACKING 0). params2: KERNEL_RADIUS,
// direction (pixels), depthCutoff; params3.x: perspective camera; the
// weights in the weight array. A discarded fragment keeps the white the pass
// clears to.
fn fx_depth_limited_blur(uv: vec2<f32>, res: vec2<f32>) -> vec4<f32> {
    let near = u.ssao0.x;
    let far = u.ssao0.y;
    let perspective = u.params3.x > 0.5;
    let p = gl_uv(uv);
    let depth = fx_hw_depth(uv);
    if (depth >= 1.0 - FX_EPSILON) {
        return fx_raw(vec4<f32>(1.0));
    }
    let center_view_z = -fx_view_z(depth, near, far, perspective);
    var r_break = false;
    var l_break = false;
    let kernel = i32(u.params2.x + 0.5);
    let direction = u.params2.yz;
    let cutoff = u.params2.w;
    let inv_size = 1.0 / res;
    var weight_sum = fx_weight(0u);
    var diffuse_sum = fx_tex(uv) * weight_sum;
    for (var i: i32 = 1; i <= kernel; i = i + 1) {
        let sample_weight = fx_weight(u32(i));
        let sample_offset = direction * f32(i) * inv_size;
        var sample_uv = p + sample_offset;
        var view_z = -fx_view_z(fx_hw_depth(gl_uv(sample_uv)), near, far, perspective);
        if (abs(view_z - center_view_z) > cutoff) {
            r_break = true;
        }
        if (!r_break) {
            diffuse_sum = diffuse_sum + fx_tex_gl(sample_uv) * sample_weight;
            weight_sum = weight_sum + sample_weight;
        }
        sample_uv = p - sample_offset;
        view_z = -fx_view_z(fx_hw_depth(gl_uv(sample_uv)), near, far, perspective);
        if (abs(view_z - center_view_z) > cutoff) {
            l_break = true;
        }
        if (!l_break) {
            diffuse_sum = diffuse_sum + fx_tex_gl(sample_uv) * sample_weight;
            weight_sum = weight_sum + sample_weight;
        }
    }
    return fx_raw(diffuse_sum / weight_sum);
}

// 54: SAOShader (NUM_SAMPLES 7, NUM_RINGS 4, DIFFUSE_TEXTURE 0). Depth is
// `hw_depth_tex`, normals `normal_tex`. params2: bias, intensity, scale,
// kernelRadius; params3: minResolution, randomSeed, perspective camera.
fn fx_sao_view_position(p: vec2<f32>, depth: f32, view_z: f32) -> vec3<f32> {
    return view_pos_from_depth(p, depth, view_z, u.proj, u.inv_proj);
}

fn fx_sao(uv: vec2<f32>, res: vec2<f32>) -> vec4<f32> {
    let near = u.ssao0.x;
    let far = u.ssao0.y;
    let perspective = u.params3.z > 0.5;
    let p = gl_uv(uv);
    let center_depth = fx_hw_depth(uv);
    if (center_depth >= 1.0 - FX_EPSILON) {
        return fx_raw(vec4<f32>(1.0));
    }
    let center_view_z = fx_view_z(center_depth, near, far, perspective);
    let view_position = fx_sao_view_position(p, center_depth, center_view_z);

    let bias = u.params2.x;
    let intensity = u.params2.y;
    let scale = u.params2.z;
    let kernel_radius = u.params2.w;
    let scale_divided_by_far = scale / far;
    let min_resolution_by_far = u.params3.x * far;
    let center_normal = 2.0 * fx_normal_tex(uv).xyz - 1.0;

    let num_samples = 7;
    let angle_step = FX_PI2 * 4.0 / 7.0;
    var angle = fx_three_rand(p + vec2<f32>(u.params3.y)) * FX_PI2;
    var radius = vec2<f32>(kernel_radius * (1.0 / 7.0)) / res;
    let radius_step = radius;
    var occlusion_sum = 0.0;
    var weight_sum = 0.0;
    for (var i: i32 = 0; i < num_samples; i = i + 1) {
        let sample_uv = p + vec2<f32>(cos(angle), sin(angle)) * radius;
        radius = radius + radius_step;
        angle = angle + angle_step;
        let sample_depth = fx_hw_depth(gl_uv(sample_uv));
        if (sample_depth >= 1.0 - FX_EPSILON) {
            continue;
        }
        let sample_view_z = fx_view_z(sample_depth, near, far, perspective);
        let sample_position = fx_sao_view_position(sample_uv, sample_depth, sample_view_z);
        let view_delta = sample_position - view_position;
        let view_distance = length(view_delta);
        let scaled_distance = scale_divided_by_far * view_distance;
        occlusion_sum = occlusion_sum
            + max(0.0, (dot(center_normal, view_delta) - min_resolution_by_far) / scaled_distance - bias)
            / (1.0 + scaled_distance * scaled_distance);
        weight_sum = weight_sum + 1.0;
    }
    if (weight_sum == 0.0) {
        return fx_raw(vec4<f32>(1.0));
    }
    let ao = occlusion_sum * (intensity / weight_sum);
    return fx_raw(vec4<f32>(vec3<f32>(1.0 - ao), 1.0));
}

// ---- 55, 56: SSRPass -----------------------------------------------------

// 55: SSRShader. `input_tex`: the beauty (nearest), `hw_depth_tex`: its
// depth (16-bit, as SSRPass's `UnsignedShortType` depth texture), `normal_tex`:
// normals, `depth_tex`: metalness. params2: opacity, maxDistance, thickness,
// flags (1 DISTANCE_ATTENUATION, 2 FRESNEL, 4 INFINITE_THICK, 8 SELECTIVE);
// params3: MAX_STEP, perspective camera. Written quantized, as into
// SSRPass's 8-bit target; a fragment three.js leaves unwritten is zero.
fn fx_ssr_depth(uv_gl: vec2<f32>) -> f32 {
    let d = fx_hw_depth(gl_uv(uv_gl));
    return floor(d * 65535.0 + 0.5) / 65535.0;
}

fn fx_ssr_view_position(uv_gl: vec2<f32>, depth: f32, clip_w: f32) -> vec3<f32> {
    var clip_position = vec4<f32>((vec3<f32>(uv_gl, depth) - 0.5) * 2.0, 1.0);
    clip_position = clip_position * clip_w;
    return (u.inv_proj * clip_position).xyz;
}

fn fx_ssr_xy(view_position: vec3<f32>, res: vec2<f32>) -> vec2<f32> {
    let clip = u.proj * vec4<f32>(view_position, 1.0);
    var xy = clip.xy / clip.w;
    xy = (xy + 1.0) / 2.0;
    return xy * res;
}

fn fx_point_line_distance(x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>) -> f32 {
    return length(cross(x0 - x1, x0 - x2)) / length(x2 - x1);
}

fn fx_point_plane_distance(point: vec3<f32>, plane_point: vec3<f32>, n: vec3<f32>) -> f32 {
    let d = -(n.x * plane_point.x + n.y * plane_point.y + n.z * plane_point.z);
    return (n.x * point.x + n.y * point.y + n.z * point.z + d) / sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
}

fn fx_ssr(uv: vec2<f32>, res: vec2<f32>, frag: vec2<f32>) -> vec4<f32> {
    let near = u.ssao0.x;
    let far = u.ssao0.y;
    let opacity = u.params2.x;
    let max_distance = u.params2.y;
    let thickness = u.params2.z;
    let flags = u32(u.params2.w + 0.5);
    let max_step = u.params3.x;
    let perspective = u.params3.y > 0.5;
    let none = vec4<f32>(0.0);
    let v_uv = gl_uv(uv);

    if ((flags & 8u) != 0u) {
        let metalness = fx_depth_tex(uv).r;
        if (metalness == 0.0) {
            return none;
        }
    }
    let depth = fx_ssr_depth(v_uv);
    let view_z = fx_view_z(depth, near, far, perspective);
    if (-view_z >= far) {
        return none;
    }
    let clip_w = u.proj[2][3] * view_z + u.proj[3][3];
    let view_position = fx_ssr_view_position(v_uv, depth, clip_w);
    // gl_FragCoord: pixel centres counted up from the bottom row.
    let d0 = vec2<f32>(frag.x, res.y - frag.y);
    let view_normal = 2.0 * fx_normal_tex(uv).xyz - 1.0;
    var view_incident = vec3<f32>(0.0, 0.0, -1.0);
    if (perspective) {
        view_incident = normalize(view_position);
    }
    let view_reflect = reflect(view_incident, view_normal);
    let max_reflect_len = max_distance / dot(-view_incident, view_normal);
    var d1_view_position = view_position + view_reflect * max_reflect_len;
    if (perspective && d1_view_position.z > -near) {
        let t = (-near - view_position.z) / view_reflect.z;
        d1_view_position = view_position + view_reflect * t;
    }
    let d1 = fx_ssr_xy(d1_view_position, res);
    let total_len = length(d1 - d0);
    let x_len = d1.x - d0.x;
    let y_len = d1.y - d0.y;
    let total_step = max(abs(x_len), abs(y_len));
    let x_span = x_len / total_step;
    let y_span = y_len / total_step;
    var i = 0.0;
    loop {
        if (i >= max_step || i >= total_step) {
            break;
        }
        let xy = vec2<f32>(d0.x + i * x_span, d0.y + i * y_span);
        i = i + 1.0;
        if (xy.x < 0.0 || xy.x > res.x || xy.y < 0.0 || xy.y > res.y) {
            break;
        }
        let s = length(xy - d0) / total_len;
        let suv = xy / res;
        let d = fx_ssr_depth(suv);
        let vz = fx_view_z(d, near, far, perspective);
        if (-vz >= far) {
            continue;
        }
        let cw = u.proj[2][3] * vz + u.proj[3][3];
        let vp = fx_ssr_view_position(suv, d, cw);
        var view_reflect_ray_z = view_position.z + s * (d1_view_position.z - view_position.z);
        if (perspective) {
            let recip = 1.0 / view_position.z;
            view_reflect_ray_z = 1.0 / (recip + s * (1.0 / d1_view_position.z - recip));
        }
        if (view_reflect_ray_z <= vz) {
            var hit = true;
            if ((flags & 4u) == 0u) {
                let away = fx_point_line_distance(vp, view_position, d1_view_position);
                let uv_neighbor = (xy + vec2<f32>(1.0, 0.0)) / res;
                let vp_neighbor = fx_ssr_view_position(uv_neighbor, d, cw);
                let min_thickness = (vp_neighbor.x - vp.x) * 3.0;
                let tk = max(min_thickness, thickness);
                hit = away <= tk;
            }
            if (hit) {
                let vn = 2.0 * fx_normal_tex(gl_uv(suv)).xyz - 1.0;
                if (dot(view_reflect, vn) >= 0.0) {
                    continue;
                }
                let distance = fx_point_plane_distance(vp, view_position, view_normal);
                if (distance > max_distance) {
                    break;
                }
                var op = opacity;
                if ((flags & 1u) != 0u) {
                    let ratio = 1.0 - (distance / max_distance);
                    op = opacity * ratio * ratio;
                }
                if ((flags & 2u) != 0u) {
                    op = op * (dot(view_incident, view_reflect) + 1.0) / 2.0;
                }
                let reflect_color = fx_load(gl_uv(suv));
                return fx_q8(vec4<f32>(reflect_color.xyz, op));
            }
        }
    }
    return none;
}

// 56: SSRBlurShader, into an 8-bit target like SSRPass's.
fn fx_ssr_blur(uv: vec2<f32>, res: vec2<f32>) -> vec4<f32> {
    let texel = 1.0 / res;
    let c = fx_tex(uv);
    let cl = fx_tex(uv + vec2<f32>(-texel.x, 0.0));
    let cr = fx_tex(uv + vec2<f32>(texel.x, 0.0));
    let cb = fx_tex(uv + vec2<f32>(0.0, texel.y));
    let ct = fx_tex(uv + vec2<f32>(0.0, -texel.y));
    let a = (c.a + cl.a + cr.a + cb.a + ct.a) * 0.2;
    if (a == 0.0) {
        // 0 / 0 in three.js; an 8-bit target stores NaN as zero.
        return vec4<f32>(0.0);
    }
    let rgb = (c.rgb * c.a + cl.rgb * cl.a + cr.rgb * cr.a + cb.rgb * cb.a + ct.rgb * ct.a) * 0.2 / a;
    return fx_q8(vec4<f32>(rgb, a));
}

// ---- 58-60: SMAAPass -----------------------------------------------------

// `resolution` in SMAA is the texel size of the pass's targets.
fn fx_smaa_tex(p: vec2<f32>) -> vec4<f32> {
    return fx_tex_gl(p);
}

// 58: SMAAEdgesShader (colour edge detection, SMAA_THRESHOLD 0.1). A
// discarded fragment is the zero an untouched edges target holds.
fn fx_smaa_edges(uv: vec2<f32>, res: vec2<f32>) -> vec4<f32> {
    let r = 1.0 / res;
    let p = gl_uv(uv);
    let off0 = p.xyxy + r.xyxy * vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    let off1 = p.xyxy + r.xyxy * vec4<f32>(1.0, 0.0, 0.0, -1.0);
    let off2 = p.xyxy + r.xyxy * vec4<f32>(-2.0, 0.0, 0.0, 2.0);
    let threshold = vec2<f32>(0.1, 0.1);
    var delta = vec4<f32>(0.0);
    let c = fx_smaa_tex(p).rgb;
    var t = abs(c - fx_smaa_tex(off0.xy).rgb);
    delta.x = max(max(t.r, t.g), t.b);
    t = abs(c - fx_smaa_tex(off0.zw).rgb);
    delta.y = max(max(t.r, t.g), t.b);
    var edges = step(threshold, delta.xy);
    if (dot(edges, vec2<f32>(1.0, 1.0)) == 0.0) {
        return vec4<f32>(0.0);
    }
    t = abs(c - fx_smaa_tex(off1.xy).rgb);
    delta.z = max(max(t.r, t.g), t.b);
    t = abs(c - fx_smaa_tex(off1.zw).rgb);
    delta.w = max(max(t.r, t.g), t.b);
    var max_delta = max(max(max(delta.x, delta.y), delta.z), delta.w);
    t = abs(c - fx_smaa_tex(off2.xy).rgb);
    delta.z = max(max(t.r, t.g), t.b);
    t = abs(c - fx_smaa_tex(off2.zw).rgb);
    delta.w = max(max(t.r, t.g), t.b);
    max_delta = max(max(max_delta, delta.z), delta.w);
    edges = edges * step(vec2<f32>(0.5 * max_delta), delta.xy);
    return vec4<f32>(edges, 0.0, 0.0);
}

// The edges target (`input_tex`) at a GL position.
fn fx_smaa_edges_at(p: vec2<f32>) -> vec2<f32> {
    return fx_tex_gl(p).rg;
}

// The search table (`noise_tex`, nearest): data coordinates.
fn fx_smaa_search_length(e_in: vec2<f32>, bias: f32, scale: f32) -> f32 {
    var e = e_in;
    e.x = bias + e.x * scale;
    return 255.0 * textureSampleLevel(noise_tex, noise_sampler, e, 0.0).r;
}

fn fx_smaa_search_x_left(tc: vec2<f32>, end: f32, r: vec2<f32>) -> f32 {
    var texcoord = tc;
    var e = vec2<f32>(0.0, 1.0);
    for (var i: i32 = 0; i < 8; i = i + 1) {
        e = fx_smaa_edges_at(texcoord);
        texcoord = texcoord - vec2<f32>(2.0, 0.0) * r;
        if (!(texcoord.x > end && e.y > 0.8281 && e.x == 0.0)) {
            break;
        }
    }
    texcoord.x = texcoord.x + 0.25 * r.x;
    texcoord.x = texcoord.x + r.x;
    texcoord.x = texcoord.x + 2.0 * r.x;
    texcoord.x = texcoord.x - r.x * fx_smaa_search_length(e, 0.0, 0.5);
    return texcoord.x;
}

fn fx_smaa_search_x_right(tc: vec2<f32>, end: f32, r: vec2<f32>) -> f32 {
    var texcoord = tc;
    var e = vec2<f32>(0.0, 1.0);
    for (var i: i32 = 0; i < 8; i = i + 1) {
        e = fx_smaa_edges_at(texcoord);
        texcoord = texcoord + vec2<f32>(2.0, 0.0) * r;
        if (!(texcoord.x < end && e.y > 0.8281 && e.x == 0.0)) {
            break;
        }
    }
    texcoord.x = texcoord.x - 0.25 * r.x;
    texcoord.x = texcoord.x - r.x;
    texcoord.x = texcoord.x - 2.0 * r.x;
    texcoord.x = texcoord.x + r.x * fx_smaa_search_length(e, 0.5, 0.5);
    return texcoord.x;
}

fn fx_smaa_search_y_up(tc: vec2<f32>, end: f32, r: vec2<f32>) -> f32 {
    var texcoord = tc;
    var e = vec2<f32>(1.0, 0.0);
    for (var i: i32 = 0; i < 8; i = i + 1) {
        e = fx_smaa_edges_at(texcoord);
        texcoord = texcoord + vec2<f32>(0.0, 2.0) * r;
        if (!(texcoord.y > end && e.x > 0.8281 && e.y == 0.0)) {
            break;
        }
    }
    texcoord.y = texcoord.y - 0.25 * r.y;
    texcoord.y = texcoord.y - r.y;
    texcoord.y = texcoord.y - 2.0 * r.y;
    texcoord.y = texcoord.y + r.y * fx_smaa_search_length(e.yx, 0.0, 0.5);
    return texcoord.y;
}

fn fx_smaa_search_y_down(tc: vec2<f32>, end: f32, r: vec2<f32>) -> f32 {
    var texcoord = tc;
    var e = vec2<f32>(1.0, 0.0);
    for (var i: i32 = 0; i < 8; i = i + 1) {
        e = fx_smaa_edges_at(texcoord);
        texcoord = texcoord - vec2<f32>(0.0, 2.0) * r;
        if (!(texcoord.y < end && e.x > 0.8281 && e.y == 0.0)) {
            break;
        }
    }
    texcoord.y = texcoord.y + 0.25 * r.y;
    texcoord.y = texcoord.y + r.y;
    texcoord.y = texcoord.y + 2.0 * r.y;
    texcoord.y = texcoord.y - r.y * fx_smaa_search_length(e.yx, 0.5, 0.5);
    return texcoord.y;
}

// The area table (`depth_tex`, bilinear): data coordinates.
fn fx_smaa_area(dist: vec2<f32>, e1: f32, e2: f32, offset: f32) -> vec2<f32> {
    let pixel_size = 1.0 / vec2<f32>(160.0, 560.0);
    var texcoord = 16.0 * round(4.0 * vec2<f32>(e1, e2)) + dist;
    texcoord = pixel_size * texcoord + 0.5 * pixel_size;
    texcoord.y = texcoord.y + (1.0 / 7.0) * offset;
    return fx_depth_tex(texcoord).rg;
}

// 59: SMAAWeightsShader. `input_tex`: edges, `depth_tex`: the area table,
// `noise_tex`: the search table.
fn fx_smaa_weights(uv: vec2<f32>, res: vec2<f32>) -> vec4<f32> {
    let r = 1.0 / res;
    let texcoord = gl_uv(uv);
    let pixcoord = texcoord / r;
    let off0 = texcoord.xyxy + r.xyxy * vec4<f32>(-0.25, 0.125, 1.25, 0.125);
    let off1 = texcoord.xyxy + r.xyxy * vec4<f32>(-0.125, 0.25, -0.125, -1.25);
    let off2 = vec4<f32>(off0.xz, off1.yw) + vec4<f32>(-2.0, 2.0, -2.0, 2.0) * r.xxyy * 8.0;
    var weights = vec4<f32>(0.0);
    let e = fx_smaa_edges_at(texcoord);
    if (e.y > 0.0) {
        var d: vec2<f32>;
        var coords: vec2<f32>;
        coords.x = fx_smaa_search_x_left(off0.xy, off2.x, r);
        coords.y = off1.y;
        d.x = coords.x;
        let e1 = fx_smaa_edges_at(coords).x;
        coords.x = fx_smaa_search_x_right(off0.zw, off2.y, r);
        d.y = coords.x;
        d = d / r.x - pixcoord.x;
        let sqrt_d = sqrt(abs(d));
        coords.y = coords.y - r.y;
        let e2 = fx_smaa_edges_at(coords + vec2<f32>(1.0, 0.0) * r).x;
        let area = fx_smaa_area(sqrt_d, e1, e2, 0.0);
        weights.x = area.x;
        weights.y = area.y;
    }
    if (e.x > 0.0) {
        var d: vec2<f32>;
        var coords: vec2<f32>;
        coords.y = fx_smaa_search_y_up(off1.xy, off2.z, r);
        coords.x = off0.x;
        d.x = coords.y;
        let e1 = fx_smaa_edges_at(coords).y;
        coords.y = fx_smaa_search_y_down(off1.zw, off2.w, r);
        d.y = coords.y;
        d = d / r.y - pixcoord.y;
        let sqrt_d = sqrt(abs(d));
        coords.y = coords.y - r.y;
        let e2 = fx_smaa_edges_at(coords + vec2<f32>(0.0, 1.0) * r).y;
        let area = fx_smaa_area(sqrt_d, e1, e2, 0.0);
        weights.z = area.x;
        weights.w = area.y;
    }
    return weights;
}

// 60: SMAABlendShader. `input_tex`: the colour, `normal_tex`: the weights.
fn fx_smaa_blend(uv: vec2<f32>, res: vec2<f32>) -> vec4<f32> {
    let r = 1.0 / res;
    var texcoord = gl_uv(uv);
    let off0 = texcoord.xyxy + r.xyxy * vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    let off1 = texcoord.xyxy + r.xyxy * vec4<f32>(1.0, 0.0, 0.0, -1.0);
    var a: vec4<f32>;
    let w = fx_normal_tex(gl_uv(texcoord));
    a.x = w.x;
    a.z = w.z;
    a.y = fx_normal_tex(gl_uv(off1.zw)).g;
    a.w = fx_normal_tex(gl_uv(off0.xy)).a;
    if (dot(a, vec4<f32>(1.0)) < 1e-5) {
        return fx_raw(fx_tex_gl(texcoord));
    }
    var offset: vec2<f32>;
    offset.x = select(-a.z, a.w, a.w > a.z);
    offset.y = select(a.x, -a.y, a.y > a.x);
    if (abs(offset.x) > abs(offset.y)) {
        offset.y = 0.0;
    } else {
        offset.x = 0.0;
    }
    var c = fx_tex_gl(texcoord);
    texcoord = texcoord + sign(offset) * r;
    var cop = fx_tex_gl(texcoord);
    let s = select(abs(offset.y), abs(offset.x), abs(offset.x) > abs(offset.y));
    c = vec4<f32>(pow(c.xyz, vec3<f32>(2.2)), c.w);
    cop = vec4<f32>(pow(cop.xyz, vec3<f32>(2.2)), cop.w);
    var mixed = mix(c, cop, s);
    mixed = vec4<f32>(pow(mixed.xyz, vec3<f32>(1.0 / 2.2)), mixed.w);
    return fx_raw(mixed);
}

// ---- 61: RenderPixelatedPass ---------------------------------------------

// 61: the pixelated material. `input_tex`: the low-resolution beauty
// (nearest), `hw_depth_tex`: its depth, `normal_tex`: normals (nearest).
// params2: normalEdgeStrength, depthEdgeStrength; params3: the beauty's size
// and texel size (`resolution`).
fn fx_pixel_depth(p: vec2<f32>, x: i32, y: i32) -> f32 {
    return fx_hw_depth(gl_uv(p + vec2<f32>(f32(x), f32(y)) * u.params3.zw));
}

fn fx_pixel_normal(p: vec2<f32>, x: i32, y: i32) -> vec3<f32> {
    return fx_normal_tex(gl_uv(p + vec2<f32>(f32(x), f32(y)) * u.params3.zw)).rgb * 2.0 - 1.0;
}

fn fx_pixel_neighbor_normal_edge(p: vec2<f32>, x: i32, y: i32, depth: f32, normal: vec3<f32>) -> f32 {
    let depth_diff = fx_pixel_depth(p, x, y) - depth;
    let neighbor_normal = fx_pixel_normal(p, x, y);
    let normal_diff = dot(normal - neighbor_normal, vec3<f32>(1.0, 1.0, 1.0));
    let normal_indicator = clamp(smoothstep(-0.01, 0.01, normal_diff), 0.0, 1.0);
    let depth_indicator = clamp(sign(depth_diff * 0.25 + 0.0025), 0.0, 1.0);
    return (1.0 - dot(normal, neighbor_normal)) * depth_indicator * normal_indicator;
}

fn fx_pixelated(uv: vec2<f32>) -> vec4<f32> {
    let p = gl_uv(uv);
    let texel = fx_load(uv);
    let normal_strength = u.params2.x;
    let depth_strength = u.params2.y;
    var depth = 0.0;
    var normal = vec3<f32>(0.0);
    if (depth_strength > 0.0 || normal_strength > 0.0) {
        depth = fx_pixel_depth(p, 0, 0);
        normal = fx_pixel_normal(p, 0, 0);
    }
    var dei = 0.0;
    if (depth_strength > 0.0) {
        var diff = 0.0;
        diff = diff + clamp(fx_pixel_depth(p, 1, 0) - depth, 0.0, 1.0);
        diff = diff + clamp(fx_pixel_depth(p, -1, 0) - depth, 0.0, 1.0);
        diff = diff + clamp(fx_pixel_depth(p, 0, 1) - depth, 0.0, 1.0);
        diff = diff + clamp(fx_pixel_depth(p, 0, -1) - depth, 0.0, 1.0);
        dei = floor(smoothstep(0.01, 0.02, diff) * 2.0) / 2.0;
    }
    var nei = 0.0;
    if (normal_strength > 0.0) {
        var indicator = 0.0;
        indicator = indicator + fx_pixel_neighbor_normal_edge(p, 0, -1, depth, normal);
        indicator = indicator + fx_pixel_neighbor_normal_edge(p, 0, 1, depth, normal);
        indicator = indicator + fx_pixel_neighbor_normal_edge(p, -1, 0, depth, normal);
        indicator = indicator + fx_pixel_neighbor_normal_edge(p, 1, 0, depth, normal);
        nei = step(0.1, indicator);
    }
    let strength = select(1.0 + normal_strength * nei, 1.0 - depth_strength * dei, dei > 0.0);
    return fx_raw(texel * strength);
}

// ---- 62: LUTPass ---------------------------------------------------------

// 62: LUTShader. `normal_tex` holds the cube's slices side by side
// (`lutSize² × lutSize`), read bilinearly; the third axis is interpolated
// here, which is what a 3D texture's trilinear fetch does. params2: lutSize,
// intensity.
fn fx_lut(uv: vec2<f32>) -> vec4<f32> {
    let val = fx_tex(uv);
    let size = u.params2.x;
    let pixel_width = 1.0 / size;
    let half_pixel_width = 0.5 / size;
    let uvw = vec3<f32>(half_pixel_width) + val.rgb * (1.0 - pixel_width);
    let z = uvw.z * size - 0.5;
    let z0 = clamp(floor(z), 0.0, size - 1.0);
    let z1 = min(z0 + 1.0, size - 1.0);
    let t = clamp(z - floor(z), 0.0, 1.0);
    let a = fx_normal_tex(vec2<f32>((z0 + uvw.x) / size, uvw.y));
    let b = fx_normal_tex(vec2<f32>((z1 + uvw.x) / size, uvw.y));
    var w = t;
    if (z < 0.0) {
        w = 0.0;
    }
    let lut_val = vec4<f32>(mix(a, b, w).rgb, val.a);
    return fx_raw(mix(val, lut_val, u.params2.y));
}

// ---- 63: RenderTransitionPass --------------------------------------------

// 63: the transition material. `input_tex`: tDiffuse1, `normal_tex`:
// tDiffuse2, `depth_tex`: tMixTexture; params2: mixRatio, threshold,
// useTexture. It ends in `colorspace_fragment`.
fn fx_transition(uv: vec2<f32>) -> vec4<f32> {
    let texel1 = fx_tex(uv);
    let texel2 = fx_normal_tex(uv);
    let mix_ratio = u.params2.x;
    let threshold = u.params2.y;
    if (u.params2.z > 0.5) {
        let transition = fx_depth_tex(uv);
        let r = mix_ratio * (1.0 + threshold * 2.0) - threshold;
        let mixf = clamp((transition.r - r) * (1.0 / threshold), 0.0, 1.0);
        return fx_display(mix(texel1, texel2, mixf));
    }
    return fx_display(mix(texel2, texel1, mix_ratio));
}

// ---- 64: OutputPass ------------------------------------------------------

fn fx_rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

fn fx_agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

// three.js `tonemapping_pars_fragment`, keyed by its constants: 1 Linear,
// 2 Reinhard, 3 Cineon, 4 ACESFilmic, 6 AgX, 7 Neutral; anything else none.
fn fx_tone_map(color_in: vec3<f32>, mode: u32, exposure: f32) -> vec3<f32> {
    var color = color_in;
    if (mode == 1u) {
        return clamp(exposure * color, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (mode == 2u) {
        color = color * exposure;
        return clamp(color / (vec3<f32>(1.0) + color), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (mode == 3u) {
        color = color * exposure;
        color = max(vec3<f32>(0.0), color - 0.004);
        return pow((color * (6.2 * color + 0.5)) / (color * (6.2 * color + 1.7) + 0.06), vec3<f32>(2.2));
    }
    if (mode == 4u) {
        let aces_input = mat3x3<f32>(
            vec3<f32>(0.59719, 0.07600, 0.02840),
            vec3<f32>(0.35458, 0.90834, 0.13383),
            vec3<f32>(0.04823, 0.01566, 0.83777),
        );
        let aces_output = mat3x3<f32>(
            vec3<f32>(1.60475, -0.10208, -0.00327),
            vec3<f32>(-0.53108, 1.10813, -0.07276),
            vec3<f32>(-0.07367, -0.00605, 1.07602),
        );
        color = color * (exposure / 0.6);
        color = aces_input * color;
        color = fx_rrt_and_odt_fit(color);
        color = aces_output * color;
        return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (mode == 6u) {
        let rec2020_to_srgb = mat3x3<f32>(
            vec3<f32>(1.6605, -0.1246, -0.0182),
            vec3<f32>(-0.5876, 1.1329, -0.1006),
            vec3<f32>(-0.0728, -0.0083, 1.1187),
        );
        let srgb_to_rec2020 = mat3x3<f32>(
            vec3<f32>(0.6274, 0.0691, 0.0164),
            vec3<f32>(0.3293, 0.9195, 0.0880),
            vec3<f32>(0.0433, 0.0113, 0.8956),
        );
        let inset = mat3x3<f32>(
            vec3<f32>(0.856627153315983, 0.137318972929847, 0.11189821299995),
            vec3<f32>(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
            vec3<f32>(0.0482516061458583, 0.101439036467562, 0.811302368396859),
        );
        let outset = mat3x3<f32>(
            vec3<f32>(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
            vec3<f32>(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
            vec3<f32>(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405),
        );
        let min_ev = -12.47393;
        let max_ev = 4.026069;
        color = color * exposure;
        color = srgb_to_rec2020 * color;
        color = inset * color;
        color = max(color, vec3<f32>(1e-10));
        color = log2(color);
        color = (color - min_ev) / (max_ev - min_ev);
        color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
        color = fx_agx_contrast(color);
        color = outset * color;
        color = pow(max(vec3<f32>(0.0), color), vec3<f32>(2.2));
        color = rec2020_to_srgb * color;
        return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (mode == 7u) {
        let start_compression = 0.8 - 0.04;
        let desaturation = 0.15;
        color = color * exposure;
        let x = min(color.r, min(color.g, color.b));
        let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
        color = color - offset;
        let peak = max(color.r, max(color.g, color.b));
        if (peak < start_compression) {
            return color;
        }
        let d = 1.0 - start_compression;
        let new_peak = 1.0 - d * d / (peak + d - start_compression);
        color = color * (new_peak / peak);
        let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
        return mix(color, vec3<f32>(new_peak), g);
    }
    return color;
}

// three.js `sRGBTransferOETF`, its 0.41666 exponent and all.
fn fx_srgb_oetf(c: vec3<f32>) -> vec3<f32> {
    return select(
        pow(c, vec3<f32>(0.41666)) * 1.055 - vec3<f32>(0.055),
        c * 12.92,
        c <= vec3<f32>(0.0031308),
    );
}

// 64: OutputShader. params2: toneMapping, toneMappingExposure, SRGB_TRANSFER.
fn fx_output(uv: vec2<f32>) -> vec4<f32> {
    var c = fx_tex(uv);
    let rgb = fx_tone_map(c.rgb, u32(u.params2.x + 0.5), u.params2.y);
    c = vec4<f32>(rgb, c.a);
    if (u.params2.z > 0.5) {
        c = vec4<f32>(fx_srgb_oetf(c.rgb), c.a);
    }
    return fx_raw(c);
}

// ---- 65: MaskPass --------------------------------------------------------

// 65: what a stencil test does, after the fact. `input_tex`: the masked
// pass's output, `depth_tex`: what the destination held before it,
// `hw_depth_tex`: the mask scene's depth — covered where something was drawn.
// params2.x: MaskPass.inverse.
fn fx_mask_mix(uv: vec2<f32>) -> vec4<f32> {
    let covered = fx_hw_depth(uv) < 1.0;
    let inside = covered != (u.params2.x > 0.5);
    if (inside) {
        return fx_raw(fx_tex(uv));
    }
    return fx_raw(fx_depth_tex(uv));
}

// ---- 66: CubeTexturePass -------------------------------------------------

// 66: ShaderLib.cube. `normal_tex`: the six faces side by side (+x, -x, +y,
// -y, +z, -z), each in GL's cube-face orientation; `inv_proj`: the camera's
// inverse projection, and `proj` standing in for its world matrix, of which
// the rotation is used. params2: opacity, tFlip, face size.
fn fx_cube_background(uv: vec2<f32>) -> vec4<f32> {
    let ndc = gl_uv(uv) * 2.0 - 1.0;
    let v = u.inv_proj * vec4<f32>(ndc, 1.0, 1.0);
    let view_dir = v.xyz / v.w;
    let world = (u.proj * vec4<f32>(view_dir, 0.0)).xyz;
    let d = vec3<f32>(u.params2.y * world.x, world.y, world.z);
    let a = abs(d);
    var face = 0.0;
    var sc = 0.0;
    var tc = 0.0;
    var ma = 1.0;
    if (a.x >= a.y && a.x >= a.z) {
        ma = a.x;
        if (d.x > 0.0) { face = 0.0; sc = -d.z; tc = -d.y; } else { face = 1.0; sc = d.z; tc = -d.y; }
    } else if (a.y >= a.z) {
        ma = a.y;
        if (d.y > 0.0) { face = 2.0; sc = d.x; tc = d.z; } else { face = 3.0; sc = d.x; tc = -d.z; }
    } else {
        ma = a.z;
        if (d.z > 0.0) { face = 4.0; sc = d.x; tc = -d.y; } else { face = 5.0; sc = -d.x; tc = -d.y; }
    }
    let half_texel = 0.5 / max(u.params2.z, 1.0);
    let s = clamp((sc / ma + 1.0) * 0.5, half_texel, 1.0 - half_texel);
    let t = clamp((tc / ma + 1.0) * 0.5, half_texel, 1.0 - half_texel);
    var c = fx_normal_tex(vec2<f32>((face + s) / 6.0, t));
    c.a = c.a * u.params2.x;
    return fx_display(c);
}

fn postfx_fragment_ext(
    frag: Out,
    kind: u32,
    uv: vec2<f32>,
    res: vec2<f32>,
    texel: vec2<f32>,
    base: vec4<f32>,
) -> vec4<f32> {
    switch kind {
        case 40u: { return fx_copy(uv); }
        case 41u: { return fx_basic_copy(uv); }
        case 42u: { return fx_luminosity_high_pass(uv); }
        case 43u: { return fx_bloom_blur(uv, res); }
        case 44u: { return fx_bloom_composite(uv); }
        case 45u: { return fx_convolution(uv); }
        case 46u: { return fx_combine(uv); }
        case 47u: { return fx_afterimage(uv); }
        case 48u: { return fx_outline_mask(uv); }
        case 49u: { return fx_outline_edge(uv, res); }
        case 50u: { return fx_outline_blur(uv, res); }
        case 51u: { return fx_outline_overlay(uv); }
        case 52u: { return fx_depth_view(uv); }
        case 53u: { return fx_depth_limited_blur(uv, res); }
        case 54u: { return fx_sao(uv, res); }
        case 55u: { return fx_ssr(uv, res, frag.pos.xy); }
        case 56u: { return fx_ssr_blur(uv, res); }
        case 58u: { return fx_smaa_edges(uv, res); }
        case 59u: { return fx_smaa_weights(uv, res); }
        case 60u: { return fx_smaa_blend(uv, res); }
        case 61u: { return fx_pixelated(uv); }
        case 62u: { return fx_lut(uv); }
        case 63u: { return fx_transition(uv); }
        case 64u: { return fx_output(uv); }
        case 65u: { return fx_mask_mix(uv); }
        case 66u: { return fx_cube_background(uv); }
        default: { return base; }
    }
}
