//! The frame-rate read-out's glyphs, shared by both solvers' shaders.
//!
//! A 3x5 bitmap font packed into fifteen bits a digit. The alternative is
//! the CPU overlay canvas the main renderer composites, which wants a
//! texture upload a frame and a path to get it here; this is three digits,
//! and it costs one branchless lookup per pixel and works the same in a
//! browser, on a phone and in a headless capture.
//!
//! Both modules concatenate this and then declare their own entry point,
//! because the plane and the volume name their uniform block differently and
//! the shared part must not mention it. `the_read_out_draws_only_where_it_says`
//! covers the volume's; `the_plane_draws_its_read_out_too` the plane's.

/// The ring of frame times, and the one row of bytes it uploads as.
///
/// Split out of [`History`] because the Metal backend keeps the same ring
/// behind an `MTLTexture` of its own. What the two backends have to agree
/// about is not the texture, it is what a byte in that row *means* — so that
/// is decided once, here, and both the WGSL and the MSL get the range from
/// these constants rather than carrying their own copy.
#[derive(Clone)]
pub(crate) struct Frames {
    samples: [f32; Frames::LEN],
    next: usize,
}

impl Default for Frames {
    fn default() -> Self {
        Self {
            samples: [0.0; Frames::LEN],
            next: 0,
        }
    }
}

impl Frames {
    /// About a second at sixty, which is long enough to see a hitch and
    /// short enough to fit beside three digits.
    pub(crate) const LEN: usize = 64;

    /// A frame time of this many milliseconds is a full byte. Anything
    /// slower than four frames a second is off the end of the chart, and
    /// anything that slow has a bigger problem than its chart.
    pub(crate) const SCALE_MS: f32 = 250.0;

    /// Record one frame, in milliseconds.
    pub(crate) fn push(&mut self, ms: f32) {
        self.samples[self.next] = ms;
        self.next = (self.next + 1) % Self::LEN;
    }

    /// One row of RGBA8, oldest sample first so the chart reads left to
    /// right and the newest frame is at the right-hand edge.
    ///
    /// Eight-bit unsigned and scaled, rather than a half float: `r16float`
    /// is not renderable everywhere the rest of the fluid runs, and the
    /// range costs nothing to give up — [`SCALE_MS`](Self::SCALE_MS) over
    /// 255 steps is a fortieth of a millisecond, and the chart is eighteen
    /// pixels tall.
    pub(crate) fn row(&self) -> Vec<u8> {
        (0..Self::LEN)
            .flat_map(|i| {
                let v = self.samples[(self.next + i) % Self::LEN];
                let byte = ((v / Self::SCALE_MS).clamp(0.0, 1.0) * 255.0).round() as u8;
                [byte, 0, 0, 255]
            })
            .collect()
    }
}

/// The smoothed frame rate, the ring behind the chart, and the one rule
/// that decides what counts as a frame.
///
/// All three backends keep this rather than a clock and a float each: the
/// rule below is subtle enough that three copies of it would drift, and the
/// read-out is the one place where all three are supposed to read the same.
#[derive(Default)]
pub(crate) struct Rate {
    frames: Frames,
    last: Option<web_time::Instant>,
    fps: f32,
    /// Set by the tests, so a chart can be checked against frame times
    /// chosen rather than against whatever the clock said.
    #[cfg(test)]
    pinned: bool,
}

impl Rate {
    /// Fold one render into the rate.
    pub(crate) fn tick(&mut self) {
        #[cfg(test)]
        if self.pinned {
            return;
        }
        let now = web_time::Instant::now();
        let Some(last) = self.last.replace(now) else {
            return;
        };
        self.fold(now.duration_since(last).as_secs_f32());
    }

    /// [`tick`](Self::tick) with the gap handed in rather than measured, in
    /// seconds. Separate so the rule below can be tested without a clock.
    fn fold(&mut self, dt: f32) {
        if dt <= self.floor() {
            return;
        }
        self.frames.push(dt * 1000.0);
        let rate = 1.0 / dt;
        self.fps = if self.fps > 0.0 {
            self.fps + (rate - self.fps) * 0.1
        } else {
            rate
        };
    }

    /// The shortest gap that is a frame and not an artefact.
    ///
    /// Two renders inside one frame — a capture beside a present, a resize
    /// that redraws — are not a thousand frames a second, and letting one
    /// in sends the smoothed rate somewhere it takes a hundred frames to
    /// come back from. It showed up as a read-out pinned at its 999 clamp
    /// while the page was plainly running at sixty.
    ///
    /// The threshold is **relative**, because a fixed one has no right
    /// value: a flat millisecond is correct at sixty frames a second and
    /// wrong at eight hundred, which is what the Metal backend reaches at
    /// 720p on an M4 — a real frame there is 1.2 ms, so a 1 ms floor threw
    /// away half of them and the chart had bars for 38 of 90 renders. A
    /// duplicate render is a small fraction of a frame time and a genuine
    /// speed-up is at most a doubling, so a fifth of the measured frame
    /// time separates them at any rate.
    ///
    /// Capped at 4 ms so the floor cannot deadlock: seeded from one slow
    /// first frame, an uncapped relative floor would reject every frame
    /// after it and so never update the rate that set it.
    fn floor(&self) -> f32 {
        if self.fps > 0.0 {
            (0.2 / self.fps).min(0.004)
        } else {
            0.0
        }
    }

    pub(crate) fn fps(&self) -> f32 {
        self.fps
    }

    pub(crate) fn frames(&self) -> &Frames {
        &self.frames
    }

    /// Milliseconds at the top of the chart: twice the current frame time,
    /// so a steady run sits halfway up however fast it is going.
    pub(crate) fn scale_ms(&self) -> f32 {
        2000.0 / self.fps.max(1.0)
    }

    /// The refresh the chart draws a line at. Rounded to a real one, so the
    /// line means "keeping up" rather than an arbitrary sixty.
    pub(crate) fn target_ms(&self) -> f32 {
        if self.fps > 90.0 {
            1000.0 / 120.0
        } else {
            1000.0 / 60.0
        }
    }

    /// Force the history and the number the read-out shows.
    #[cfg(test)]
    pub(crate) fn pin_frame_times(&mut self, ms: &[f32]) {
        for v in ms {
            self.frames.push(*v);
        }
        self.fps = 1000.0 / ms.last().copied().unwrap_or(16.0).max(0.01);
        self.pinned = true;
    }

    /// Force the number alone, so a test can check the glyphs at a value it
    /// chose rather than at whatever the clock said.
    #[cfg(test)]
    pub(crate) fn pin_fps(&mut self, fps: f32) {
        self.fps = fps;
        self.pinned = true;
    }
}

/// [`Frames`], plus the wgpu texture it is drawn from.
///
/// A texture rather than more uniform: the plane's uniform block is shared
/// with the Metal backend byte for byte, and sixty-four floats is not a
/// thing to put in it for a read-out. One `write_texture` a frame.
#[derive(Default)]
pub(crate) struct History {
    texture: Option<wgpu::Texture>,
    view: Option<wgpu::TextureView>,
}

impl History {
    /// The texture to bind, uploaded from `frames` as it is handed over.
    pub(crate) fn view(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frames: &Frames,
    ) -> &wgpu::TextureView {
        if self.texture.is_none() {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fluid frame times"),
                size: wgpu::Extent3d {
                    width: Frames::LEN as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.view = Some(texture.create_view(&Default::default()));
            self.texture = Some(texture);
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.texture.as_ref().expect("just made"),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frames.row(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(Frames::LEN as u32 * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: Frames::LEN as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.view.as_ref().expect("just made")
    }
}

/// WGSL: `read_out(clip, value) -> f32` for the digits and
/// `chart(clip, history, scale_ms, target_ms) -> vec4<f32>` for the
/// sparkline under them. Zero alpha means nothing is there.
///
/// [`Frames::LEN`] and [`Frames::SCALE_MS`] are substituted in, so the
/// shader cannot disagree with the upload about how wide the row is or what
/// a byte means. The MSL transcription in `src/metal/fluid.metal` carries
/// the two as literals instead, because `FLUID_SHADER_SOURCE` is public and
/// has to compile as it stands; `the_msl_read_out_agrees_about_the_row`
/// holds them to these.
pub(crate) fn glyphs() -> String {
    SOURCE
        .replace("COLUMNS", &Frames::LEN.to_string())
        .replace("SCALE_MS", &format!("{:?}", Frames::SCALE_MS))
}

const SOURCE: &str = r#"
// Rows top-to-bottom, three bits each, leftmost pixel in the low bit.
fn glyph_bits(d : i32) -> u32 {
    switch (d) {
        case 0: { return 31599u; }
        case 1: { return 29850u; }
        case 2: { return 29671u; }
        case 3: { return 31207u; }
        case 4: { return 18925u; }
        case 5: { return 31183u; }
        case 6: { return 31695u; }
        case 7: { return 18727u; }
        case 8: { return 31727u; }
        default: { return 31215u; }
    }
}

fn glyph_on(d : i32, x : i32, y : i32) -> bool {
    if (x < 0 || x > 2 || y < 0 || y > 4) {
        return false;
    }
    // Parenthesised: `==` binds tighter than `&` here, as it does in C and
    // does not in Rust.
    return ((glyph_bits(d) >> u32(y * 3 + x)) & 1u) == 1u;
}

// Up to three digits at a fixed pixel size in the top left. Fixed, because a
// read-out that grew with the window would be a banner on a phone and a
// watermark on a desktop.
fn read_out(clip : vec2<f32>, value : f32) -> f32 {
    let scale = 3;
    let pad = 10;
    let cell = (vec2<i32>(clip) - vec2<i32>(pad, pad)) / scale;
    if (cell.x < 0 || cell.x > 11 || cell.y < 0 || cell.y > 4) {
        return 0.0;
    }
    let shown = i32(clamp(round(value), 0.0, 999.0));
    let column = cell.x / 4;
    // Not `pow`: ten squared comes back as 99.99999 in f32 and truncates to
    // 99, so 999 divides to ten and the first digit reads zero. Everything
    // below a hundred is unaffected, which is why it survived being looked
    // at.
    var tens = 1;
    if (column == 0) {
        tens = 100;
    } else if (column == 1) {
        tens = 10;
    }
    let digit = (shown / tens) % 10;
    // Leading zeros are not drawn: "7" rather than "007".
    //
    // Parenthesised, and it has to be: WGSL forbids mixing `&&` and `||`
    // without grouping, naga accepts it anyway, and Tint — which is what
    // Chrome and Safari actually compile with — rejects the *module*. One
    // unparenthesised line here took every pipeline in the fluid down and
    // left a black canvas running at sixty frames a second, with nothing in
    // wgpu's log, because the error is the browser's and not wgpu's.
    // `no_wgsl_mixes_and_with_or` is the guard.
    let leading = (column == 0 && shown < 100) || (column == 1 && shown < 10);
    if (leading || !glyph_on(digit, cell.x % 4, cell.y)) {
        return 0.0;
    }
    return 0.85;
}

// The last sixty-four frames, as a bar a frame, under the number.
//
// A number tells you the rate now; a chart tells you whether it is steady,
// which is the question anyone looking at a frame counter is actually
// asking. A hitch that lasts two frames never shows in a smoothed figure
// and is obvious as a spike here.
//
// `hist` is one row of frame times in milliseconds, oldest first.
//
// The height is twice the *current* frame time, not twice the refresh
// target. Scaling to the target looks right until the fluid is slower than
// it, and then every bar clips and the chart is a solid block that says
// only "slow" — which the number already said. Scaling to what is actually
// happening means a steady run sits at half height whatever the rate, and
// the thing the chart is for, the variation, is always visible.
//
// The target still gets a line, where it falls, so the absolute reference
// is not lost.
fn chart(clip : vec2<f32>, hist : texture_2d<f32>, scale_ms : f32, target_ms : f32) -> vec4<f32> {
    let pad = 10;
    let top = 29;
    let height = 18;
    let columns = COLUMNS;
    let px = vec2<i32>(clip) - vec2<i32>(pad, top);
    if (px.x < 0 || px.x >= columns * 2 || px.y < 0 || px.y >= height) {
        return vec4<f32>(0.0);
    }
    // Fetched by index, not sampled: a bar is one whole texel and there is
    // nothing to interpolate. Stored as a fraction of `Frames::SCALE_MS`;
    // see there for why it is a byte rather than a half float.
    let ms = textureLoad(hist, vec2<i32>(px.x / 2, 0), 0).x * SCALE_MS;
    let up = height - 1 - px.y;
    let full = max(scale_ms, 0.001);
    // The target, where it falls — off the top of the chart when the fluid
    // is running at less than half the rate it is aiming for, which is
    // itself worth seeing.
    let line = i32(target_ms / full * f32(height));
    if (up == line && line < height) {
        return vec4<f32>(1.0, 1.0, 1.0, 0.25);
    }
    if (ms <= 0.0) {
        return vec4<f32>(0.0);
    }
    let bar = i32(clamp(ms / full, 0.0, 1.0) * f32(height));
    if (up >= bar) {
        return vec4<f32>(0.0);
    }
    if (ms > target_ms) {
        return vec4<f32>(1.0, 0.45, 0.3, 0.75);
    }
    return vec4<f32>(0.65, 0.85, 1.0, 0.5);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// How many of the sixty-four samples are non-zero, read the way the
    /// shader reads them.
    fn counted(r: &Rate) -> usize {
        r.frames()
            .row()
            .chunks_exact(4)
            .filter(|p| p[0] > 0)
            .count()
    }

    /// The rule that decides what counts as a frame, at both ends of the
    /// range it has to work over.
    ///
    /// It replaced a flat one-millisecond floor, which was right at sixty
    /// and wrong at eight hundred — the rate the Metal backend reaches at
    /// 720p on an M4, where a frame is 1.2 ms. Half of them fell under the
    /// floor and the chart had bars for 38 of 90 renders.
    #[test]
    fn the_rate_counts_frames_and_not_duplicate_renders() {
        // Sixty, with a second render beside every present — the shape that
        // pinned the read-out at its 999 clamp.
        let mut r = Rate::default();
        for _ in 0..200 {
            r.fold(1.0 / 60.0);
            r.fold(0.001);
        }
        assert!(
            (r.fps() - 60.0).abs() < 1.0,
            "a capture beside every present read as {:.0} fps",
            r.fps()
        );

        // Eight hundred, where every frame is real and under the old floor.
        let mut r = Rate::default();
        for _ in 0..200 {
            r.fold(1.0 / 820.0);
        }
        assert!(
            (r.fps() - 820.0).abs() < 20.0,
            "eight hundred real frames read as {:.0} fps",
            r.fps()
        );
        assert_eq!(counted(&r), Frames::LEN, "the chart would have gaps in it");

        // And one slow first frame must not latch the floor shut: an
        // uncapped relative floor would reject every frame after it, and so
        // never update the rate that set it.
        let mut r = Rate::default();
        r.fold(0.5);
        for _ in 0..200 {
            r.fold(1.0 / 60.0);
        }
        assert!(
            (r.fps() - 60.0).abs() < 1.0,
            "a half-second first frame left the rate stuck at {:.0} fps",
            r.fps()
        );
    }
}
