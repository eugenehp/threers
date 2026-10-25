// The quad: a 2x2 million unit square no frustum culls; only the signs of its
// corners are read, and it spans the box worth shading.
@vertex
fn vs_custom(in: VsIn) -> VsOut {
    var out = standard_vertex(in);
    let corner = sign(in.position.xy) * 0.5 + vec2<f32>(0.5);
    let box = u_data.data[4];
    let view = u_data.data[0].xy;
    let css = mix(box.xy, box.zw, vec2<f32>(corner.x, 1.0 - corner.y));
    out.clip_pos = vec4<f32>(css.x / view.x * 2.0 - 1.0, 1.0 - css.y / view.y * 2.0, 0.0, 1.0);
    return out;
}
