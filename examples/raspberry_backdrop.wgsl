// A soft pool of light behind the subject, and black everywhere else.
//
// Deliberately not the studio seen through the camera: the panels are three
// orders of magnitude brighter than the berry's diffuse, so showing them would
// leave the subject a silhouette. What a photographer does here is flag the
// panels off the background and light it separately, which is this.

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let exposure = u_data.data[0].x;
    // Distance from the point the berry stands in front of, in world units.
    let d = length(in.world_pos.xy - vec2<f32>(0.0, 1.15));
    let pool = exp(-d * d * 0.055);
    // Warm, and a long way down: it reads as a wall catching spill rather than
    // as a light of its own.
    let col = vec3<f32>(0.95, 0.78, 0.78) * pool * 1.1 + vec3<f32>(0.008, 0.006, 0.010);
    return vec4<f32>(framebuffer_encode(pbr_neutral(col * exposure)), 1.0);
}
