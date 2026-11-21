//! Training: batches of made-up and recorded swipes, each on its own
//! moved keyboard, CTC loss, AdamW on a cosine schedule.
//!
//! The recipe follows FUTO's published one (AdamW, warmup then cosine
//! decay, gradient clipping at 1, geometric augmentation every step),
//! smaller. Each swipe of a batch draws its own randomness from the seed,
//! the step and its place in the batch, so a run is the same on one
//! thread or many.
//!
//! ```no_run
//! use threers_swipe::{KeyboardGeometry, Lexicon, Model, TrainConfig, Trainer};
//!
//! let words = Lexicon::load("words_en.txt").unwrap();
//! let mut trainer = Trainer::new(TrainConfig::default());
//! for keyboard in KeyboardGeometry::built_in().into_iter().filter(|k| k.language == "en") {
//!     trainer.add_synthetic(keyboard, words.word_list());
//! }
//! trainer.run(|step, stats| if step % 100 == 0 { println!("{step} {:.3}", stats.loss) });
//! trainer.model.save("swipe.safetensors").unwrap();
//! ```

use crate::augment::{augment, AugmentConfig};
use crate::ctc;
use crate::data::Recordings;
use crate::decode::{Decoder, Lexicon};
use crate::geometry::KeyboardGeometry;
use crate::model::{Model, ModelConfig};
use crate::synth::{self, Rng, SynthConfig};
use crate::trace::features;

/// What and how long to train.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub model: ModelConfig,
    pub steps: usize,
    pub batch: usize,
    /// Peak learning rate, reached after `warmup` of the steps, then down
    /// a cosine to `min_lr`.
    pub lr: f32,
    pub min_lr: f32,
    pub warmup: f32,
    pub weight_decay: f32,
    /// Gradients scaled down to this norm at most.
    pub clip: f32,
    /// Of each batch, the share drawn from recordings when there are any.
    pub recorded_share: f32,
    pub seed: u64,
    pub synth: SynthConfig,
    pub augment: AugmentConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            steps: 20_000,
            batch: 64,
            lr: 2e-3,
            min_lr: 2e-5,
            warmup: 0.05,
            weight_decay: 1e-4,
            clip: 1.0,
            recorded_share: 0.7,
            seed: 1,
            synth: SynthConfig::default(),
            augment: AugmentConfig::default(),
        }
    }
}

/// One step's numbers.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepStats {
    /// Mean CTC loss over the swipes that fit.
    pub loss: f32,
    /// Swipes left out (word longer than the swipe has points).
    pub skipped: usize,
    pub lr: f32,
    /// Before clipping.
    pub grad_norm: f32,
}

/// A training sample: key centres, features, target classes.
struct Sample {
    keys: Vec<(f32, f32)>,
    x: Vec<f32>,
    target: Vec<usize>,
}

/// A model, its optimiser state and what it learns from.
pub struct Trainer {
    pub config: TrainConfig,
    pub model: Model,
    pub step: usize,
    synthetic: Vec<(KeyboardGeometry, Vec<String>)>,
    recorded: Option<Recordings>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Trainer {
    pub fn new(config: TrainConfig) -> Self {
        let model = Model::new(config.model.clone(), config.seed);
        Self::resume(config, model, 0)
    }

    /// Carries on training `model` from `step` (optimiser state fresh).
    pub fn resume(config: TrainConfig, model: Model, step: usize) -> Self {
        let n = model.len();
        Self {
            config,
            model,
            step,
            synthetic: vec![],
            recorded: None,
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    /// Made-up swipes of `words` on `keyboard`; each layout added is drawn
    /// as often as the others.
    pub fn add_synthetic(&mut self, keyboard: KeyboardGeometry, words: Vec<String>) {
        let words: Vec<String> = words
            .into_iter()
            .filter(|w| keyboard.word_keys(w).is_some())
            .collect();
        if !words.is_empty() {
            self.synthetic.push((keyboard, words));
        }
    }

    /// Recorded swipes, drawn for [`TrainConfig::recorded_share`] of each
    /// batch.
    pub fn set_recorded(&mut self, recs: Recordings) {
        self.recorded = (!recs.items.is_empty()).then_some(recs);
    }

    /// The learning rate at `step`.
    pub fn lr_at(&self, step: usize) -> f32 {
        let c = &self.config;
        let total = c.steps.max(1) as f32;
        let warm = (c.warmup * total).max(1.0);
        let s = step as f32;
        if s < warm {
            return c.lr * (s + 1.0) / warm;
        }
        let f = ((s - warm) / (total - warm).max(1.0)).min(1.0);
        c.min_lr + 0.5 * (c.lr - c.min_lr) * (1.0 + (std::f32::consts::PI * f).cos())
    }

    fn sample(&self, rng: &mut Rng) -> Option<Sample> {
        let use_recorded = match (&self.recorded, self.synthetic.is_empty()) {
            (Some(_), true) => true,
            (Some(_), false) => rng.chance(self.config.recorded_share),
            (None, _) => false,
        };
        let (mut keys, mut points, mut target) = if use_recorded {
            let recs = self.recorded.as_ref()?;
            let r = &recs.items[rng.below(recs.items.len())];
            let g = &recs.keyboards[r.keyboard];
            (centres(g), r.points.clone(), g.word_path(&r.word)?)
        } else {
            if self.synthetic.is_empty() {
                return None;
            }
            let (g, words) = &self.synthetic[rng.below(self.synthetic.len())];
            let (w, pts) = synth::random_swipe(g, words, &self.config.synth, rng)?;
            (centres(g), pts, g.word_path(w)?)
        };
        augment(
            &mut keys,
            &mut points,
            &mut target,
            &self.config.augment,
            rng,
        );
        Some(Sample {
            keys,
            x: features(&points),
            target: target.into_iter().map(|k| k + 1).collect(),
        })
    }

    /// The loss and gradient of one sample, added to `grad`.
    fn accumulate(&self, s: &Sample, grad: &mut [f32]) -> Option<f32> {
        let act = self.model.forward(&s.x, &s.keys);
        let (loss, dz) = ctc::loss(&act.logits, s.keys.len() + 1, &s.target)?;
        if !loss.is_finite() {
            return None;
        }
        self.model.backward(&s.x, &s.keys, &act, &dz, grad);
        Some(loss)
    }

    /// The batch's summed gradient, summed loss and how many fit.
    fn batch_gradient(&self) -> (Vec<f32>, f32, usize) {
        let n = self.model.len();
        let batch = self.config.batch;
        let seed = self.config.seed ^ (self.step as u64).wrapping_mul(0x2545_f491_4f6c_dd1d);
        let one = |i: usize, acc: &mut (Vec<f32>, f32, usize)| {
            let mut rng = Rng::new(seed.wrapping_add(i as u64 * 0x9e37_79b9));
            if let Some(s) = self.sample(&mut rng)
                && let Some(l) = self.accumulate(&s, &mut acc.0)
            {
                acc.1 += l;
                acc.2 += 1;
            }
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..batch)
                .into_par_iter()
                .fold(
                    || (vec![0.0f32; n], 0.0f32, 0usize),
                    |mut acc, i| {
                        one(i, &mut acc);
                        acc
                    },
                )
                .reduce(
                    || (vec![0.0f32; n], 0.0, 0),
                    |mut a, b| {
                        for (x, y) in a.0.iter_mut().zip(&b.0) {
                            *x += y;
                        }
                        (a.0, a.1 + b.1, a.2 + b.2)
                    },
                )
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut acc = (vec![0.0f32; n], 0.0f32, 0usize);
            for i in 0..batch {
                one(i, &mut acc);
            }
            acc
        }
    }

    /// One optimiser step.
    pub fn train_step(&mut self) -> StepStats {
        let (mut grad, loss_sum, used) = self.batch_gradient();
        let lr = self.lr_at(self.step);
        self.step += 1;
        if used == 0 {
            return StepStats {
                skipped: self.config.batch,
                lr,
                ..StepStats::default()
            };
        }
        let inv = 1.0 / used as f32;
        let mut norm = 0.0f32;
        for g in &mut grad {
            *g *= inv;
            norm += *g * *g;
        }
        let norm = norm.sqrt();
        let scale = if norm > self.config.clip {
            self.config.clip / norm
        } else {
            1.0
        };
        // AdamW; the bias corrections go by optimiser step, not by weight.
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let t = self.step as i32;
        let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
        let wd = self.config.weight_decay;
        for (((p, g), m), v) in self
            .model
            .params
            .iter_mut()
            .zip(&grad)
            .zip(&mut self.m)
            .zip(&mut self.v)
        {
            let g = g * scale;
            *m = b1 * *m + (1.0 - b1) * g;
            *v = b2 * *v + (1.0 - b2) * g * g;
            *p -= lr * ((*m / c1) / ((*v / c2).sqrt() + eps) + wd * *p);
        }
        StepStats {
            loss: loss_sum * inv,
            skipped: self.config.batch - used,
            lr,
            grad_norm: norm,
        }
    }

    /// Steps up to [`TrainConfig::steps`], calling `report` after each.
    pub fn run(&mut self, mut report: impl FnMut(usize, &StepStats)) {
        while self.step < self.config.steps {
            let s = self.train_step();
            report(self.step, &s);
        }
    }
}

fn centres(g: &KeyboardGeometry) -> Vec<(f32, f32)> {
    g.keys.iter().map(|k| (k.x, k.y)).collect()
}

/// How often the right word comes first, and in the top three.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Accuracy {
    pub top1: f32,
    pub top3: f32,
    pub swipes: usize,
    /// Of those, words the lexicon lacks: counted wrong, as no decoder
    /// over it can type them.
    pub unknown: usize,
}

impl std::fmt::Display for Accuracy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "top-1 {:.1}%  top-3 {:.1}%  ({} swipes, {} words not in the lexicon)",
            self.top1 * 100.0,
            self.top3 * 100.0,
            self.swipes,
            self.unknown
        )
    }
}

/// Scores `cases` (decoder, swipe, word), on every core with `parallel`.
fn score(
    model: &Model,
    cases: Vec<(&Decoder, Vec<crate::TouchPoint>, String)>,
    unknown: usize,
) -> Accuracy {
    let one = |(dec, pts, word): &(&Decoder, Vec<crate::TouchPoint>, String)| {
        let got = dec.decode(model, pts, 3);
        let word = word.to_lowercase();
        (
            got.first().is_some_and(|c| c.word == word) as usize,
            got.iter().any(|c| c.word == word) as usize,
        )
    };
    #[cfg(feature = "parallel")]
    let (t1, t3) = {
        use rayon::prelude::*;
        cases
            .par_iter()
            .map(one)
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    };
    #[cfg(not(feature = "parallel"))]
    let (t1, t3) = cases
        .iter()
        .map(one)
        .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    let n = cases.len() + unknown;
    let d = n.max(1) as f32;
    Accuracy {
        top1: t1 as f32 / d,
        top3: t3 as f32 / d,
        swipes: n,
        unknown,
    }
}

/// Accuracy on `n` fresh made-up swipes of `lexicon`'s words on
/// `keyboard`, not moved.
pub fn evaluate_synthetic(
    model: &Model,
    keyboard: &KeyboardGeometry,
    lexicon: &Lexicon,
    cfg: &SynthConfig,
    n: usize,
    seed: u64,
) -> Accuracy {
    let dec = Decoder::new(keyboard, lexicon);
    let words = lexicon.word_list();
    let mut rng = Rng::new(seed);
    let cases: Vec<_> = (0..n)
        .filter_map(|_| synth::random_swipe(keyboard, &words, cfg, &mut rng))
        .map(|(w, p)| (&dec, p, w.to_owned()))
        .collect();
    score(model, cases, 0)
}

/// Accuracy on recorded swipes against `lexicon`; a word it lacks counts
/// as a miss.
pub fn evaluate_recorded(model: &Model, recs: &Recordings, lexicon: &Lexicon) -> Accuracy {
    let decoders: Vec<Decoder> = recs
        .keyboards
        .iter()
        .map(|g| Decoder::new(g, lexicon))
        .collect();
    let known: std::collections::HashSet<&str> =
        lexicon.words.iter().map(|(w, _)| w.as_str()).collect();
    let (mut cases, mut unknown) = (vec![], 0);
    for r in &recs.items {
        if known.contains(r.word.to_lowercase().as_str()) {
            cases.push((&decoders[r.keyboard], r.points.clone(), r.word.clone()));
        } else {
            unknown += 1;
        }
    }
    score(model, cases, unknown)
}

/// Recordings split in two by session (by line where there is none):
/// about `share` of the sessions held out, the same ones every time.
pub fn split_by_session(recs: Recordings, share: f32) -> (Recordings, Recordings) {
    let Recordings {
        keyboards,
        items,
        errors,
        skipped,
    } = recs;
    let mut train = Recordings {
        keyboards: keyboards.clone(),
        items: vec![],
        errors,
        skipped,
    };
    let mut held = Recordings {
        keyboards,
        ..Recordings::default()
    };
    for (i, r) in items.into_iter().enumerate() {
        let h = match &r.session {
            Some(s) => s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
            }),
            None => i as u64 * 0x9e37_79b9_7f4a_7c15,
        };
        if (h % 10_000) as f32 / 10_000.0 < share {
            held.items.push(r);
        } else {
            train.items.push(r);
        }
    }
    (train, held)
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout};

    #[test]
    fn schedule_warms_up_then_decays() {
        let t = Trainer::new(TrainConfig {
            steps: 100,
            model: ModelConfig::tiny(),
            ..TrainConfig::default()
        });
        assert!(t.lr_at(0) < t.lr_at(4));
        assert!((t.lr_at(5) - 2e-3).abs() < 1e-4);
        assert!(t.lr_at(99) < 1e-4);
    }

    #[test]
    fn loss_falls_on_a_few_words() {
        let g = KeyboardGeometry::from_config(
            &KeyboardConfig::iphone(),
            &KeyboardLayout::english(),
            390.0,
        );
        let cfg = TrainConfig {
            model: ModelConfig::tiny(),
            steps: 60,
            batch: 16,
            lr: 5e-3,
            ..TrainConfig::default()
        };
        let mut t = Trainer::new(cfg);
        t.add_synthetic(
            g,
            ["hello", "world", "keyboard", "swipe", "type"]
                .map(String::from)
                .to_vec(),
        );
        let first = t.train_step().loss;
        let mut last = 0.0;
        t.run(|_, s| last = s.loss);
        assert!(last < first * 0.7, "{first} → {last}");
    }
}
