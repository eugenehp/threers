//! Average-bitrate control for the all-intra encoders.
//!
//! Both native encoders take a quantizer, not a bitrate: you say how hard to
//! quantize and find out afterwards how big the file is. That is the wrong way
//! round for the question people actually ask, which is "make this fit in ten
//! megabytes".
//!
//! Intra pictures follow a rate law simple enough to steer by:
//!
//! ```text
//! log2(bytes) ≈ k − slope · qp
//! ```
//!
//! `k` describes the *content* — how much there is to code — and `slope` how
//! fast the encoder gives it up. A controller that learns both from what it has
//! already coded can invert the law to pick the next quantizer.
//!
//! The obvious value for `slope` is 1/6: the quantizer step doubles every six
//! QP, so the rate should halve. It does not. Measured on this encoder the
//! slope is about **0.116**, nearer 1/8.6 — coefficients that quantize to zero
//! stop costing anything at all, and the residual's entropy does not fall as
//! fast as its magnitude. Assuming 1/6 makes the controller under-correct, and
//! it undershot a 20 Mbit/s target by half before this was measured rather than
//! assumed. So the slope is learned too, from pairs of frames coded at
//! different quantizers, and 1/6 appears nowhere.
//!
//! It is single-pass by design. A second pass would be more accurate, and for
//! all-intra content it would be easy, but it doubles the encode time to buy an
//! accuracy the measurements below suggest is not needed.

/// What a caller wants from the encoder.
#[derive(Clone, Copy, Debug)]
pub enum Quality {
    /// A fixed quantizer: predictable quality, unpredictable size.
    Qp(i32),
    /// A target average bitrate in bits per second: predictable size,
    /// quality that moves with the content.
    Bitrate(u64),
}

/// Why a bitrate target cannot be met.
#[derive(Clone, Copy, Debug)]
pub enum Reach {
    /// The content is too detailed: even the coarsest quantizer overshoots.
    TooComplex { times: f64 },
    /// The content is too simple: even the finest quantizer undershoots.
    TooSimple { times: f64 },
}

/// Tracks the rate law and picks a quantizer per frame.
#[derive(Debug)]
pub struct RateControl {
    /// Bytes per frame the clip is aiming at.
    target: f64,
    /// The content constant: `log2(bytes) + slope·qp`, learned as frames go by.
    k: f64,
    /// How much `log2(bytes)` falls per step of quantizer, also learned.
    slope: f64,
    /// The last observation, to estimate the slope from a pair.
    last: Option<(f64, f64)>,
    /// How many slope measurements have been made.
    pairs: u32,
    /// The quantizer last handed out, to limit how far the next one may move.
    prev_qp: Option<i32>,
    /// How many observations have gone into `k`.
    seen: u32,
    /// Bytes emitted so far.
    spent: f64,
    /// Frames emitted so far.
    done: u32,
    /// Total frames, when the caller knows. Without it the controller spreads
    /// its correction over a fixed horizon instead of over what is left.
    total: Option<u32>,
    qp_min: i32,
    qp_max: i32,
    /// Whether the controller ever ran out of quantizer in either direction.
    pinned_coarse: bool,
    pinned_fine: bool,
}

/// Quantizer the probe frame is coded at, and the fallback when nothing has
/// been observed yet. The middle of the useful range, so the first correction
/// is a small one whichever way it goes.
pub const PROBE_QP: i32 = 26;

/// How many frames to spread a rate correction over, at most.
const BLIND_HORIZON: f64 = 12.0;

/// How far the quantizer may move between batches once under way.
///
/// Deriving this from the measured slope — so that one correction always
/// changes the rate by the same ratio — is the elegant version, and it measured
/// worse than a constant. On steep content it collapses to a single step, and
/// the controller then cannot reach its target before the clip ends.
const MAX_QP_STEP: i32 = 8;

/// Fall in `log2(bytes)` per step of quantizer, before anything is measured.
///
/// The measured value for this encoder on photographic content, which is a far
/// better starting point than the 1/6 the quantizer step size suggests.
const DEFAULT_SLOPE: f64 = 0.116;

/// The slope cannot be trusted outside this range — a pair of frames whose
/// content changed between them can imply anything, including a negative slope.
///
/// Wide, because the slope is not one number. Measured on noise-like content it
/// is 0.075 between QP 20 and 30 and **0.32** between 40 and 51: once most
/// coefficients quantize to zero, each further step costs far more of the rate
/// than it did while there was still detail to lose. A narrow clamp pinned the
/// estimate below the truth and left the controller unable to reach high
/// quantizers, which is only survivable when there are enough feedback rounds to
/// crawl there one step at a time — so it worked single-threaded and failed with
/// large batches.
const SLOPE_RANGE: (f64, f64) = (0.04, 0.50);

impl RateControl {
    /// `bits_per_second` at `fps`, over `total` frames if that is known.
    pub fn new(bits_per_second: u64, fps: u32, total: Option<u32>) -> Self {
        let fps = fps.max(1) as f64;
        Self {
            target: bits_per_second as f64 / 8.0 / fps,
            k: 0.0,
            slope: DEFAULT_SLOPE,
            last: None,
            pairs: 0,
            prev_qp: None,
            seen: 0,
            spent: 0.0,
            done: 0,
            total,
            qp_min: 0,
            qp_max: 51,
            pinned_coarse: false,
            pinned_fine: false,
        }
    }

    /// Restrict the quantizer range, for callers who would rather miss the
    /// target than let quality out of a band.
    pub fn clamp_qp(mut self, min: i32, max: i32) -> Self {
        self.qp_min = min.clamp(0, 51);
        self.qp_max = max.clamp(self.qp_min, 51);
        self
    }

    /// The quantizer for the next frame.
    pub fn next_qp(&self) -> i32 {
        if self.seen == 0 {
            return PROBE_QP.clamp(self.qp_min, self.qp_max);
        }
        // Aim at the target, adjusted by a slice of whatever has been over- or
        // underspent so far.
        //
        // The obvious alternative — aim at the exact average the remaining
        // frames must hit — is far too sharp. It asks one frame to repay the
        // whole debt, which on steep content means a large quantizer step,
        // which overshoots the other way and repeats. Spreading the repayment
        // over a horizon damps that, and the horizon shrinks to whatever is
        // actually left so the clip still lands on target.
        let debt = self.spent - self.target * f64::from(self.done);
        let horizon = match self.total {
            Some(n) if n > self.done => f64::from(n - self.done).min(BLIND_HORIZON),
            _ => BLIND_HORIZON,
        }
        .max(1.0);
        let want = self.target - debt / horizon;
        // A frame that would have to be four times smaller than the average is
        // a sign the content changed, not that the quantizer should collapse.
        let want = want.clamp(self.target * 0.25, self.target * 4.0);
        let qp = ((self.k - want.max(1.0).log2()) / self.slope).round() as i32;
        // Damp the step once the controller is under way. The rate curve is far
        // steeper at high quantizers than at low ones, so a model fitted in one
        // region badly overshoots when aimed at another — and an overshoot in a
        // feedback loop is an oscillation. The first move is undamped because it
        // is the one that has to cross the distance from the probe.
        let qp = match self.prev_qp {
            // Wide enough to cross the useful range in two corrections. A
            // tighter limit sounds safer and measures worse: the controller
            // cannot reach the target inside a short clip, and spends the whole
            // clip travelling.
            Some(p) => qp.clamp(p - MAX_QP_STEP, p + MAX_QP_STEP),
            None => qp,
        };
        qp.clamp(self.qp_min, self.qp_max)
    }

    /// Record what a frame actually cost at the quantizer it was coded with.
    pub fn observe(&mut self, qp: i32, bytes: usize) {
        let lg = (bytes.max(1) as f64).log2();
        let q = f64::from(qp);
        // Two frames at different quantizers are a measurement of the slope.
        // Frames at the same quantizer are not, and neither is a pair a step
        // apart, where the content's own variation swamps the signal.
        if let Some((q0, lg0)) = self.last {
            if (q - q0).abs() >= 2.0 {
                let obs = ((lg0 - lg) / (q - q0)).clamp(SLOPE_RANGE.0, SLOPE_RANGE.1);
                // Trust the first real measurement outright. Blending it with
                // the default drags the estimate toward a slope that describes
                // some other content, and on steep material that error is worth
                // ten quantizer steps — which is what set the oscillation off.
                // Weighted hard toward the newest measurement. The slope is not
                // one number — it is 0.12 where there is still detail to lose
                // and 0.44 once most coefficients have gone — so an estimate
                // that averages across regions describes neither, and an
                // underestimate makes every correction overshoot. Swept against
                // real encodes: 0.4 left a 30% miss where 0.7 leaves 8%.
                let w = if self.pairs == 0 { 1.0 } else { 0.7 };
                self.slope = self.slope * (1.0 - w) + obs * w;
                self.pairs += 1;
            }
        }
        self.last = Some((q, lg));
        self.prev_qp = Some(qp);
        self.pinned_coarse |= qp >= self.qp_max;
        self.pinned_fine |= qp <= self.qp_min;
        let k_obs = lg + q * self.slope;
        // Trust the first observation completely and then settle down: the
        // content constant is genuinely constant for a static scene, and moves
        // slowly for anything else, so a long memory beats a responsive one.
        let weight = if self.seen == 0 {
            1.0
        } else {
            (1.0 / f64::from(self.seen + 1)).max(0.15)
        };
        self.k = self.k * (1.0 - weight) + k_obs * weight;
        self.seen += 1;
        self.spent += bytes as f64;
        self.done += 1;
    }

    /// Seed the law from a probe encode without counting it against the budget.
    pub fn seed(&mut self, qp: i32, bytes: usize) {
        let lg = (bytes.max(1) as f64).log2();
        self.k = lg + f64::from(qp) * self.slope;
        self.last = Some((f64::from(qp), lg));
        self.seen = 1;
    }

    /// A second quantizer worth probing before committing, if there is one.
    ///
    /// The probe and the answer can be twenty-five quantizer steps apart, and
    /// the rate curve bends enough over that distance that the chord between
    /// them is a poor guide — the slope is 0.075 per step at one end and 0.32 at
    /// the other. A second probe near where the answer is thought to be turns
    /// that chord into a local measurement.
    ///
    /// It costs one frame encode, against a whole clip coded at the wrong rate.
    pub fn probe_again(&self) -> Option<i32> {
        let (last, _) = self.last?;
        let want = self.next_qp();
        ((want - last as i32).abs() > 6).then_some(want)
    }

    /// Fold a second probe in: it refines the law but does not count against
    /// the budget, because its frame will be coded again for real.
    pub fn seed_pair(&mut self, qp: i32, bytes: usize) {
        let lg = (bytes.max(1) as f64).log2();
        let q = f64::from(qp);
        if let Some((q0, lg0)) = self.last {
            if (q - q0).abs() >= 2.0 {
                self.slope = ((lg0 - lg) / (q - q0)).clamp(SLOPE_RANGE.0, SLOPE_RANGE.1);
                self.pairs += 1;
            }
        }
        self.k = lg + q * self.slope;
        self.last = Some((q, lg));
    }

    /// Bytes emitted so far.
    pub fn spent(&self) -> f64 {
        self.spent
    }

    /// Whether the target turned out to be out of reach, and in which direction.
    ///
    /// A bitrate can be unreachable both ways. Past `qp_max` the encoder has
    /// nothing coarser to offer, so content too complex cannot be squeezed into
    /// the target. Past `qp_min` it has nothing finer, so content too *simple*
    /// cannot be made to fill it — a flat render at QP 0 is as large as it will
    /// ever get, and asking for ten times that gets the same file.
    ///
    /// Both are properties of the request rather than faults, and both are worth
    /// saying: a file at a fifth of the requested bitrate looks exactly like a
    /// bug from outside.
    ///
    /// Judged from what was actually coded, not from the model. Extrapolating
    /// the rate law from the quantizers in use out to 0 or 51 is exactly where
    /// it stops holding — doing that reported a rendered clip as "2.9x too
    /// simple" for a target it in fact came within 3% of.
    pub fn outcome(&self) -> Option<Reach> {
        if self.done == 0 {
            return None;
        }
        let ratio = self.spent / (self.target * f64::from(self.done));
        if self.pinned_coarse && ratio > 1.10 {
            Some(Reach::TooComplex { times: ratio })
        } else if self.pinned_fine && ratio < 0.90 {
            Some(Reach::TooSimple { times: 1.0 / ratio })
        } else {
            None
        }
    }

    /// How many frames to encode before consulting the controller again.
    ///
    /// Frames are encoded in parallel batches, and the controller only learns at
    /// a batch boundary — so a clip that fits in one batch gets no feedback at
    /// all, and lands wherever the first extrapolation put it. That was worth
    /// 10-24% on a ten-frame clip.
    ///
    /// Eight rounds, and the group size is derived only from the clip length and
    /// the memory budget — never from the core count.
    ///
    /// The controller learns once per group, so the group size *is* the
    /// accuracy. Sizing it by how many frames happen to fit on the machine's
    /// cores made the same clip encode to a different size on a different
    /// machine: one thread gave ninety-six single-frame rounds and fourteen gave
    /// twelve, and the two missed the target by 19% and 1% respectively. Which
    /// is the wrong kind of surprise.
    pub fn group_size(&self, memory_limit: usize) -> usize {
        const ROUNDS: usize = 8;
        match self.total {
            Some(n) if n > 0 => memory_limit.min(((n as usize) / ROUNDS).max(1)).max(1),
            _ => memory_limit.min(BLIND_HORIZON as usize).max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic encoder that obeys the rate law exactly, to check the
    /// controller inverts it correctly rather than to check the law.
    /// A synthetic encoder obeying the measured law, not the textbook one.
    fn coded_size(qp: i32, k: f64) -> usize {
        (2f64.powf(k - f64::from(qp) * DEFAULT_SLOPE)) as usize
    }

    /// A content constant whose ideal quantizer is `qp`.
    ///
    /// Derived rather than written down: a hard-coded constant silently becomes
    /// unreachable — or trivially reachable — whenever the model's slope
    /// changes, and then the test is measuring the clamp instead of the
    /// controller. It has happened twice.
    fn content_needing_qp(bits_per_second: u64, fps: u32, qp: f64) -> f64 {
        let target = bits_per_second as f64 / 8.0 / f64::from(fps);
        target.log2() + qp * DEFAULT_SLOPE
    }

    #[test]
    fn it_converges_on_a_target() {
        // 4 Mbit/s at 30fps is 16667 bytes a frame.
        let mut rc = RateControl::new(4_000_000, 30, Some(60));
        let k = content_needing_qp(4_000_000, 30, 38.0);
        for _ in 0..60 {
            let qp = rc.next_qp();
            rc.observe(qp, coded_size(qp, k));
        }
        let mean = rc.spent() / 60.0;
        let target = 4_000_000.0 / 8.0 / 30.0;
        assert!(
            (mean - target).abs() / target < 0.05,
            "mean {mean:.0} bytes vs target {target:.0}"
        );
    }

    /// The first frame is a probe, so it is the *rest* that has to absorb the
    /// error — a controller that only corrected going forward would overshoot
    /// by whatever the probe cost.
    #[test]
    fn it_pays_back_a_bad_first_frame() {
        let mut rc = RateControl::new(4_000_000, 30, Some(30));
        // Reachable, but only well below the probe quantizer, so the probe
        // frame overspends badly and the rest has to make it up.
        let k = content_needing_qp(4_000_000, 30, 44.0);
        let qp0 = rc.next_qp();
        assert_eq!(qp0, PROBE_QP);
        rc.observe(qp0, coded_size(qp0, k));
        for _ in 1..30 {
            let qp = rc.next_qp();
            rc.observe(qp, coded_size(qp, k));
        }
        let mean = rc.spent() / 30.0;
        let target = 4_000_000.0 / 8.0 / 30.0;
        assert!(
            (mean - target).abs() / target < 0.10,
            "mean {mean:.0} vs target {target:.0} — the overspend was not paid back"
        );
    }

    /// Content that changes must not make the quantizer oscillate.
    #[test]
    fn it_does_not_oscillate_when_content_changes() {
        let mut rc = RateControl::new(4_000_000, 30, Some(80));
        let mut qps = Vec::new();
        for i in 0..80 {
            // The scene gets busier a third of the way through.
            let k = content_needing_qp(4_000_000, 30, if i < 40 { 32.0 } else { 42.0 });
            let qp = rc.next_qp();
            rc.observe(qp, coded_size(qp, k));
            qps.push(qp);
        }
        // After the change has been absorbed, successive quantizers should be
        // within a step or two of each other.
        let tail = &qps[55..];
        let swing = tail.iter().max().unwrap() - tail.iter().min().unwrap();
        assert!(swing <= 2, "quantizer swings by {swing} after settling: {tail:?}");
    }

    /// Asking for less than the content can be coded in is a request the
    /// encoder cannot meet, and it should be detectable rather than silent.
    #[test]
    fn an_impossible_target_is_reported() {
        // Content whose ideal quantizer is past 51: it cannot be made to fit.
        let k = content_needing_qp(4_000_000, 30, 70.0);
        let mut rc = RateControl::new(4_000_000, 30, Some(20));
        for _ in 0..20 {
            let qp = rc.next_qp();
            rc.observe(qp, coded_size(qp, k));
        }
        match rc.outcome() {
            Some(Reach::TooComplex { times }) => {
                assert!(times > 1.3, "reported only {times:.2}x over")
            }
            other => panic!("expected TooComplex, got {other:?}"),
        }

        let k_ok = content_needing_qp(4_000_000, 30, 40.0);
        let mut ok = RateControl::new(4_000_000, 30, Some(20));
        for _ in 0..20 {
            let qp = ok.next_qp();
            ok.observe(qp, coded_size(qp, k_ok));
        }
        assert!(ok.outcome().is_none(), "this target is reachable");
    }

    /// Content too simple to fill the target is the other failure, and it looks
    /// identical from outside: a file far off the requested size.
    #[test]
    fn a_target_the_content_cannot_fill_is_reported() {
        // Content whose ideal quantizer is below 0: it cannot be made that big.
        let k = content_needing_qp(20_000_000, 30, -40.0);
        let mut rc = RateControl::new(20_000_000, 30, Some(20));
        for _ in 0..20 {
            let qp = rc.next_qp();
            rc.observe(qp, coded_size(qp, k));
        }
        match rc.outcome() {
            Some(Reach::TooSimple { times }) => assert!(times > 1.3, "only {times:.2}x under"),
            other => panic!("expected TooSimple, got {other:?}"),
        }
    }

    /// Without a frame count the controller still has to converge.
    #[test]
    fn it_works_without_knowing_the_clip_length() {
        let mut rc = RateControl::new(4_000_000, 30, None);
        let k = content_needing_qp(4_000_000, 30, 36.0);
        for _ in 0..120 {
            let qp = rc.next_qp();
            rc.observe(qp, coded_size(qp, k));
        }
        let mean = rc.spent() / 120.0;
        let target = 4_000_000.0 / 8.0 / 30.0;
        assert!(
            (mean - target).abs() / target < 0.10,
            "mean {mean:.0} vs target {target:.0}"
        );
    }
}
