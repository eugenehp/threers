// One hair. Everything interesting happened on the CPU: `vertex_color.rgb` is
// the radiance leaving this point and `.a` is the coverage the width rule gave
// back when the ribbon was widened to a pixel.
//
// What is left is the round profile. A hair is a cylinder, not a flat strip, so
// its edges fall away — without this the fuzz is a mat of tiny rectangles and
// the silhouette goes hard again, which is the thing the fuzz was for.

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let exposure = u_data.data[0].x;
    let across = in.uv.x;
    let edge = 1.0 - across * across * 0.6;
    let alpha = in.vertex_color.a * edge;
    if (alpha < 0.02) { discard; }
    let hdr = in.vertex_color.rgb * exposure;
    return vec4<f32>(framebuffer_encode(pbr_neutral(hdr)), alpha);
}
