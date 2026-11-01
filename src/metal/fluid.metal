// The fluid's frame-rate read-out, in Metal Shading Language.
//
// Every other pass of the fluid is generated, with the WGSL the wgpu module
// runs, from one `passes!` source (`src/fluid/shader.rs`), and comes before
// this in `fluid_shader_source()`: `FluidParams` and `fluid_PassVertex` are
// its. The read-out is written per language because its glyphs are shared
// with the volume solver's WGSL, in `src/fluid/read_out.rs`.

// ------------------------------------------------------- the frame-rate read-out
//
// A transcription of `src/fluid/read_out.rs`, which the wgpu module shares
// between its two solvers. The two constants below are that module's
// `Frames::LEN` and `Frames::SCALE_MS`; they are literals here rather than
// substituted in, because `FLUID_SHADER_SOURCE` is a public
// constant, and `the_msl_read_out_agrees_about_the_row` is what keeps
// them in step with the Rust.

constant int   FLUID_HISTORY_COLUMNS  = 64;
constant float FLUID_HISTORY_SCALE_MS = 250.0;

// Rows top-to-bottom, three bits each, leftmost pixel in the low bit.
static uint fluid_glyph_bits(int d) {
    switch (d) {
        case 0: return 31599u;
        case 1: return 29850u;
        case 2: return 29671u;
        case 3: return 31207u;
        case 4: return 18925u;
        case 5: return 31183u;
        case 6: return 31695u;
        case 7: return 18727u;
        case 8: return 31727u;
        default: return 31215u;
    }
}

static bool fluid_glyph_on(int d, int x, int y) {
    if (x < 0 || x > 2 || y < 0 || y > 4) {
        return false;
    }
    return ((fluid_glyph_bits(d) >> uint(y * 3 + x)) & 1u) == 1u;
}

// Up to three digits at a fixed pixel size in the top left. Fixed, because a
// read-out that grew with the window would be a banner on a phone and a
// watermark on a desktop.
static float fluid_read_out(float2 clip, float value) {
    const int scale = 3;
    const int pad = 10;
    const int2 cell = (int2(clip) - int2(pad, pad)) / scale;
    if (cell.x < 0 || cell.x > 11 || cell.y < 0 || cell.y > 4) {
        return 0.0;
    }
    const int shown = int(clamp(round(value), 0.0, 999.0));
    const int column = cell.x / 4;
    // Not `pow`: ten squared comes back as 99.99999 in float and truncates
    // to 99, so 999 divides to ten and the first digit reads zero.
    int tens = 1;
    if (column == 0) {
        tens = 100;
    } else if (column == 1) {
        tens = 10;
    }
    const int digit = (shown / tens) % 10;
    // Leading zeros are not drawn: "7" rather than "007".
    const bool leading = (column == 0 && shown < 100) || (column == 1 && shown < 10);
    if (leading || !fluid_glyph_on(digit, cell.x % 4, cell.y)) {
        return 0.0;
    }
    return 0.85;
}

// The last sixty-four frames, as a bar a frame, under the number. See the
// WGSL for why the height is twice the *current* frame time rather than
// twice the refresh target.
static float4 fluid_chart(
    float2 clip,
    texture2d<float> hist,
    float scale_ms,
    float target_ms
) {
    const int pad = 10;
    const int top = 29;
    const int height = 18;
    const int columns = FLUID_HISTORY_COLUMNS;
    const int2 px = int2(clip) - int2(pad, top);
    if (px.x < 0 || px.x >= columns * 2 || px.y < 0 || px.y >= height) {
        return float4(0.0);
    }
    // Fetched by index, not sampled: a bar is one whole texel and there is
    // nothing to interpolate.
    const float ms = hist.read(uint2(uint(px.x / 2), 0u)).x * FLUID_HISTORY_SCALE_MS;
    const int up = height - 1 - px.y;
    const float full = max(scale_ms, 0.001);
    // The target, where it falls — off the top of the chart when the fluid
    // is running at less than half the rate it is aiming for, which is
    // itself worth seeing.
    const int line = int(target_ms / full * float(height));
    if (up == line && line < height) {
        return float4(1.0, 1.0, 1.0, 0.25);
    }
    if (ms <= 0.0) {
        return float4(0.0);
    }
    const int bar = int(clamp(ms / full, 0.0, 1.0) * float(height));
    if (up >= bar) {
        return float4(0.0);
    }
    if (ms > target_ms) {
        return float4(1.0, 0.45, 0.3, 0.75);
    }
    return float4(0.65, 0.85, 1.0, 0.5);
}

// `value` is the number, `radius` the milliseconds the chart is full at and
// `curl` the refresh target — the same three fields the WGSL reads, so the
// uniform block stays shared.
//
// The one line that is not a transcription is the return: the wgpu overlay
// pipeline blends straight alpha and every blended pipeline here is
// premultiplied, so the colour is premultiplied in the shader instead.
// `src * srcAlpha + dst * (1 - srcAlpha)` and `src' * 1 + dst * (1 -
// srcAlpha)` with `src' = src * srcAlpha` are the same arithmetic, which is
// what `the_two_read_outs_agree` checks rather than assumes.
fragment float4 fs_fluid_overlay(
    fluid_PassVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]],
    texture2d<float> tex_a [[texture(0)]]
) {
    const float a = fluid_read_out(in.pos.xy, u.value);
    if (a > 0.0) {
        return float4(float3(1.0) * a, a);
    }
    const float4 bar = fluid_chart(in.pos.xy, tex_a, u.radius, u.curl);
    if (bar.a <= 0.0) {
        discard_fragment();
    }
    return float4(bar.rgb * bar.a, bar.a);
}
