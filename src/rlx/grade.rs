//! Fitting a colour grade to a reference, by gradient descent.
//!
//! Everything else in this bridge *runs* a computation someone wrote. This one
//! writes it: give it a render and a reference image, and it solves for the
//! 3×3 matrix and offset that carry one to the other. rlx differentiates the
//! loss with respect to those twelve numbers ([`grad_with_loss`]), and the
//! host does the descent.
//!
//! ```no_run
//! use threers::rlx::{preferred_device, ColorGrade, FitOptions};
//!
//! # let (render, reference) = (vec![0u8; 4], vec![0u8; 4]);
//! let report = ColorGrade::fit(&render, &reference, &FitOptions::default(), preferred_device())
//!     .unwrap();
//! println!("loss {:.5} → {:.5}", report.initial_loss, report.final_loss);
//! let graded = report.grade.apply(&render);
//! ```
//!
//! # What this can and cannot match
//!
//! A 3×3 matrix plus an offset is an *affine* map on colour: it can rotate the
//! primaries, change the white point, cross-talk the channels, lift the blacks.
//! It cannot bend a curve, so it will not learn a filmic toe or a gamma
//! change, and it has no idea where anything is, so it will not learn a
//! vignette or a gradient. Fitting one against a reference that differs in
//! those ways gives the best affine approximation and a `final_loss` that says
//! how far short it fell — which is the useful answer, not a failure.
//!
//! The fit is in linear light by default, where "twice the value" means twice
//! the light and a matrix is the natural object. Fitting sRGB-encoded values
//! instead is what an image editor does; [`FitOptions::color_space`] switches.
//!
//! One more limit worth knowing before reading a `final_loss`: an 8-bit
//! reference has already clamped everything its grade pushed past white, and
//! no affine map reproduces a clamp. Fitting against a reference with blown
//! highlights therefore converges to a residual and stays there — the fit is
//! finished, not stuck.

use ::rlx::{grad_with_loss, DType, Device, Graph, GraphExt, Session, Shape};

use crate::textures::{Texture, TextureFormat};

use super::session::GraphRunner;
use super::tensor::{ColorSpace, Layout, Tensor, TensorError};
use super::FrameFilter;

/// An affine colour transform: `out = matrix · in + bias`, in whichever space
/// it was fitted.
///
/// `matrix[o][i]` is the contribution of input channel `i` to output channel
/// `o` — the way it reads on paper. (The graph wants the transpose, which is
/// this module's business and not the caller's.)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorGrade {
    pub matrix: [[f32; 3]; 3],
    pub bias: [f32; 3],
}

impl Default for ColorGrade {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ColorGrade {
    /// Leaves colour alone.
    pub const IDENTITY: Self = Self {
        matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        bias: [0.0; 3],
    };

    /// The transform applied to one RGB triple, unclamped.
    pub fn apply_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let mut out = self.bias;
        for (o, row) in self.matrix.iter().enumerate() {
            for (i, m) in row.iter().enumerate() {
                out[o] += m * rgb[i];
            }
        }
        out
    }

    /// The transform applied to an RGBA8 frame on the host, in the space the
    /// grade was fitted in. Alpha is copied.
    ///
    /// For a single frame this is the cheaper path — twelve multiplies per
    /// pixel does not need a graph. [`Self::filter`] is for the case where the
    /// frames keep coming.
    pub fn apply(&self, rgba: &[u8]) -> Vec<u8> {
        self.apply_in(rgba, ColorSpace::Linear)
    }

    /// [`Self::apply`] in an explicit space — pass the same one the fit used.
    pub fn apply_in(&self, rgba: &[u8], space: ColorSpace) -> Vec<u8> {
        let linear = matches!(space, ColorSpace::Linear);
        let mut out = Vec::with_capacity(rgba.len());
        for px in rgba.chunks_exact(4) {
            let mut rgb = [0.0f32; 3];
            for (c, v) in rgb.iter_mut().enumerate() {
                let raw = px[c] as f32 / 255.0;
                *v = if linear {
                    super::srgb_to_linear(raw)
                } else {
                    raw
                };
            }
            let graded = self.apply_rgb(rgb);
            for v in graded {
                let v = if linear { super::linear_to_srgb(v) } else { v };
                out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
            out.push(px[3]);
        }
        out
    }

    /// The same transform as a [`Texture`], for a render target or a material
    /// slot rather than a file.
    pub fn apply_to_texture(&self, rgba: &[u8], width: u32, height: u32) -> Texture {
        Texture::new(
            width,
            height,
            TextureFormat::Rgba8UnormSrgb,
            self.apply(rgba),
        )
    }

    /// A compiled per-frame filter, for a sequence.
    ///
    /// The grade is a 4×4 with an identity row for alpha, so coverage passes
    /// through the same matrix multiply as the colour and needs no separate
    /// handling.
    pub fn filter(&self, width: u32, height: u32, device: Device) -> FrameFilter {
        let mut filter = FrameFilter::new(apply_graph(width, height), device, "frame")
            .layout(Layout::Nhwc)
            .color_space(ColorSpace::Linear);
        self.bind(filter.runner_mut());
        filter
    }

    /// Bind this grade's numbers into a runner built from [`apply_graph`].
    fn bind(&self, runner: &mut GraphRunner) {
        // Row-major `[in][out]`, which is what `mm` reads, with alpha as an
        // identity fourth channel.
        let mut m = [0.0f32; 16];
        for o in 0..3 {
            for i in 0..3 {
                m[i * 4 + o] = self.matrix[o][i];
            }
        }
        m[15] = 1.0;
        let b = [self.bias[0], self.bias[1], self.bias[2], 0.0];
        runner.set_param("grade", &m);
        runner.set_param("offset", &b);
    }

    /// Fit a grade carrying `source` onto `target`, both RGBA8 frames of the
    /// same size.
    pub fn fit(
        source: &[u8],
        target: &[u8],
        options: &FitOptions,
        device: Device,
    ) -> Result<FitReport, TensorError> {
        if source.len() != target.len() {
            return Err(TensorError::ByteCount {
                expected: source.len(),
                got: target.len(),
            });
        }
        let (x, t) = sample_pairs(source, target, options);
        let n = x.dims()[0];
        if n == 0 {
            return Err(TensorError::Empty);
        }

        // Forward: mean squared error, as output 0 — `grad_with_loss` reads
        // the loss from there.
        let mut forward = Graph::new("grade_loss");
        let xs = forward.input("source", Shape::new(&[n, 3], DType::F32));
        let ts = forward.input("target", Shape::new(&[n, 3], DType::F32));
        let m = forward.param("grade", Shape::new(&[3, 3], DType::F32));
        let b = forward.param("offset", Shape::new(&[1, 3], DType::F32));
        let mapped = forward.mm(xs, m);
        let mapped = forward.add(mapped, b);
        let residual = forward.sub(mapped, ts);
        let squared = forward.mul(residual, residual);
        let loss = forward.mean(squared, vec![0, 1], false);
        forward.set_outputs(vec![loss]);

        // Backward: [loss, ∂loss/∂grade, ∂loss/∂offset].
        let backward = grad_with_loss(&forward, &[m, b]);
        let mut compiled = Session::new(device).compile(backward);

        let mut matrix = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut offset = [0.0f32; 3];
        // `d_output` is the seed of the chain rule: the gradient of the loss
        // with respect to itself. rlx makes it an *input* rather than a
        // baked-in 1.0, which is what lets a caller weight one loss among
        // several — and what makes leaving it unbound a silent zero, since an
        // unbound input is zeros and zero times anything is no gradient at all.
        let seed = [1.0f32];
        let inputs: [(&str, &[f32]); 3] = [
            ("source", x.data()),
            ("target", t.data()),
            ("d_output", &seed),
        ];

        // Velocities for the momentum term. Colour channels are strongly
        // correlated — a render's red and green move together — which makes
        // the loss surface a long narrow valley that plain descent crawls
        // along. Momentum is what gets down it in hundreds of steps instead of
        // tens of thousands.
        let mut v_matrix = [0.0f32; 9];
        let mut v_offset = [0.0f32; 3];

        let mut initial_loss = f32::NAN;
        let mut final_loss = f32::NAN;
        let mut diverged = false;
        // The last weights that produced a finite loss. A learning rate too
        // large for the data does not fail — it overshoots, doubles, and is
        // NaN within a dozen steps — and every number after that is NaN too.
        // Returning that as a "grade" would paint the whole frame black and
        // report success, so the last good step is what comes back, with
        // `diverged` set to say so.
        let mut last_good = (matrix, offset);
        for step in 0..options.iterations {
            compiled.set_param("grade", &matrix);
            compiled.set_param("offset", &offset);
            let out = compiled.run(&inputs);
            let loss = out[0].first().copied().unwrap_or(f32::NAN);
            if step == 0 {
                initial_loss = loss;
            }
            if !loss.is_finite() {
                diverged = true;
                let (m, o) = last_good;
                matrix = m;
                offset = o;
                break;
            }
            final_loss = loss;
            last_good = (matrix, offset);

            descend(
                &mut matrix,
                &mut v_matrix,
                &out[1],
                options.learning_rate,
                options.momentum,
            );
            descend(
                &mut offset,
                &mut v_offset,
                &out[2],
                options.learning_rate,
                options.momentum,
            );
        }

        // The graph's `[in][out]` back to the readable `[out][in]`.
        let mut grade = ColorGrade {
            matrix: [[0.0; 3]; 3],
            bias: offset,
        };
        for o in 0..3 {
            for i in 0..3 {
                grade.matrix[o][i] = matrix[i * 3 + o];
            }
        }
        Ok(FitReport {
            grade,
            initial_loss,
            final_loss,
            samples: n,
            diverged,
        })
    }
}

/// One momentum step: `v ← βv + g`, `w ← w − ηv`.
fn descend(weights: &mut [f32], velocity: &mut [f32], gradient: &[f32], rate: f32, beta: f32) {
    for ((w, v), g) in weights.iter_mut().zip(velocity.iter_mut()).zip(gradient) {
        *v = beta * *v + g;
        *w -= rate * *v;
    }
}

/// How to run the descent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitOptions {
    pub iterations: usize,
    pub learning_rate: f32,
    /// Momentum β. 0 is plain gradient descent; 0.9 is the usual choice and
    /// roughly a tenfold effective step along a consistent direction.
    pub momentum: f32,
    /// How many pixels to fit against. The full frame is not needed and is not
    /// free: every iteration ships the samples to the device, so a 1080p frame
    /// would move 25 MB per step to learn twelve numbers. Sampling every
    /// `len/samples`-th pixel covers the same colour distribution for a
    /// thousandth of the traffic.
    pub samples: usize,
    /// Which space to fit in. See the module note.
    pub color_space: ColorSpace,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            iterations: 400,
            learning_rate: 0.5,
            momentum: 0.9,
            samples: 20_000,
            color_space: ColorSpace::Linear,
        }
    }
}

/// What the fit found, and how well it did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitReport {
    pub grade: ColorGrade,
    /// Mean squared error before the first step — the distance to beat.
    pub initial_loss: f32,
    /// Mean squared error after the last one.
    pub final_loss: f32,
    /// How many pixels were actually fitted against.
    pub samples: usize,
    /// The descent produced a non-finite loss and was stopped early. The grade
    /// is the last one that did not, and `final_loss` is its loss — so the
    /// result is usable, just not converged. Lower `learning_rate`.
    pub diverged: bool,
}

/// `frame · grade + offset`, over an NHWC RGBA frame.
///
/// The reshape is free — an NHWC frame is already `[pixels, 4]` in memory —
/// and it is what lets one matrix multiply do the whole image.
fn apply_graph(width: u32, height: u32) -> Graph {
    let pixels = width as usize * height as usize;
    let mut g = Graph::new("grade_apply");
    let x = g.input(
        "frame",
        Shape::new(&[1, height as usize, width as usize, 4], DType::F32),
    );
    let m = g.param("grade", Shape::new(&[4, 4], DType::F32));
    let b = g.param("offset", Shape::new(&[1, 4], DType::F32));
    let flat = g.reshape_(x, vec![pixels as i64, 4]);
    let mapped = g.mm(flat, m);
    let mapped = g.add(mapped, b);
    let out = g.reshape_(mapped, vec![1, height as i64, width as i64, 4]);
    g.set_outputs(vec![out]);
    g
}

/// Stride-sample matching pixels from both frames into two `[n, 3]` tensors.
fn sample_pairs(source: &[u8], target: &[u8], options: &FitOptions) -> (Tensor, Tensor) {
    let pixels = source.len() / 4;
    let wanted = options.samples.clamp(1, pixels.max(1));
    let stride = (pixels / wanted).max(1);
    let linear = matches!(options.color_space, ColorSpace::Linear);
    let mut xs = Vec::with_capacity(wanted * 3);
    let mut ts = Vec::with_capacity(wanted * 3);
    let mut taken = 0;
    for p in (0..pixels).step_by(stride) {
        for c in 0..3 {
            let s = source[p * 4 + c] as f32 / 255.0;
            let t = target[p * 4 + c] as f32 / 255.0;
            if linear {
                xs.push(super::srgb_to_linear(s));
                ts.push(super::srgb_to_linear(t));
            } else {
                xs.push(s);
                ts.push(t);
            }
        }
        taken += 1;
    }
    (
        Tensor::new(xs, &[taken, 3]).expect("three per sample"),
        Tensor::new(ts, &[taken, 3]).expect("three per sample"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rlx::preferred_device;

    /// A frame of assorted colours — enough spread that a fit is determined.
    fn swatches() -> Vec<u8> {
        let mut out = Vec::new();
        let mut state = 0x1234_5678u64;
        for _ in 0..1024 {
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) % 256) as u8
            };
            out.extend_from_slice(&[next(), next(), next(), 255]);
        }
        out
    }

    #[test]
    fn a_known_grade_is_recovered_from_the_images_it_produced() {
        // Chosen to keep every channel inside `0..=1`. A grade that pushes
        // values past white is not recoverable from the image it produced,
        // because the image clamped them — the information is gone, and the
        // best affine fit to a clamped target is not the grade that made it.
        // That is a property of 8-bit images, not of the fitter, and
        // `final_loss` is where it shows up.
        let known = ColorGrade {
            matrix: [[0.9, 0.05, 0.0], [0.0, 0.85, 0.05], [0.02, 0.0, 0.88]],
            bias: [0.01, 0.01, 0.01],
        };
        let source = swatches();
        let target = known.apply(&source);

        let report = ColorGrade::fit(
            &source,
            &target,
            &FitOptions {
                iterations: 600,
                ..Default::default()
            },
            preferred_device(),
        )
        .unwrap();

        assert!(
            report.final_loss < report.initial_loss * 0.01,
            "loss barely moved: {} → {}",
            report.initial_loss,
            report.final_loss
        );
        for o in 0..3 {
            for i in 0..3 {
                assert!(
                    (report.grade.matrix[o][i] - known.matrix[o][i]).abs() < 0.05,
                    "matrix[{o}][{i}] = {}, wanted {}",
                    report.grade.matrix[o][i],
                    known.matrix[o][i]
                );
            }
            assert!(
                (report.grade.bias[o] - known.bias[o]).abs() < 0.05,
                "bias[{o}] = {}, wanted {}",
                report.grade.bias[o],
                known.bias[o]
            );
        }
    }

    #[test]
    fn the_identity_grade_leaves_a_frame_alone() {
        let frame = swatches();
        assert_eq!(ColorGrade::IDENTITY.apply(&frame), frame);
    }

    #[test]
    fn fitting_a_frame_to_itself_finds_the_identity() {
        let frame = swatches();
        let report =
            ColorGrade::fit(&frame, &frame, &FitOptions::default(), preferred_device()).unwrap();
        assert!(report.final_loss < 1e-6, "loss {}", report.final_loss);
        let graded = report.grade.apply(&frame);
        for (a, b) in graded.iter().zip(frame.iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 2, "{a} vs {b}");
        }
    }

    #[test]
    fn the_compiled_filter_agrees_with_the_host_path() {
        let known = ColorGrade {
            matrix: [[1.2, 0.0, 0.0], [0.0, 0.85, 0.1], [0.0, 0.0, 1.05]],
            bias: [0.02, 0.0, -0.01],
        };
        let (w, h) = (16u32, 16u32);
        let frame: Vec<u8> = (0..(w * h * 4)).map(|i| ((i * 37) % 251) as u8).collect();

        let host = known.apply(&frame);
        let mut filter = known.filter(w, h, preferred_device());
        let (device_side, _, _) = filter.apply(&frame, w, h).unwrap();

        for (i, (a, b)) in host.iter().zip(device_side.iter()).enumerate() {
            assert!(
                (*a as i32 - *b as i32).abs() <= 1,
                "byte {i}: host {a}, graph {b}"
            );
        }
    }

    #[test]
    fn an_empty_frame_is_refused_rather_than_compiled() {
        assert_eq!(
            ColorGrade::fit(&[], &[], &FitOptions::default(), preferred_device()),
            Err(TensorError::Empty)
        );
    }

    #[test]
    fn a_diverging_fit_reports_itself_and_returns_the_last_good_grade() {
        let source = swatches();
        let target: Vec<u8> = source.iter().map(|v| v.wrapping_add(40)).collect();
        let report = ColorGrade::fit(
            &source,
            &target,
            &FitOptions {
                learning_rate: 80.0,
                iterations: 200,
                ..Default::default()
            },
            preferred_device(),
        )
        .unwrap();

        assert!(report.diverged, "an 80.0 learning rate did not diverge");
        assert!(report.final_loss.is_finite(), "reported a NaN loss");
        for row in report.grade.matrix {
            for v in row {
                assert!(v.is_finite(), "returned a NaN grade: {:?}", report.grade);
            }
        }
        // …and the grade it returns is usable: applying it produces pixels.
        let out = report.grade.apply(&source);
        assert_eq!(out.len(), source.len());
    }

    #[test]
    fn mismatched_frames_are_refused() {
        assert!(
            ColorGrade::fit(&[0; 8], &[0; 4], &FitOptions::default(), preferred_device()).is_err()
        );
    }
}
