//! Training: batches of made-up and recorded swipes, each on its own
//! moved keyboard, CTC loss, AdamW on a cosine schedule.
//!
//! The recipe follows FUTO's published one (AdamW, warmup then cosine
//! decay, gradient clipping at 1, geometric augmentation every step),
//! smaller. A step:
//!
//! 1. Draws [`TrainConfig::batch`] swipes: a recorded one with chance
//!    [`TrainConfig::recorded_share`] (when there are recordings), else a
//!    made-up one of a random word on a random keyboard added with
//!    [`Trainer::add_synthetic`].
//! 2. Moves each swipe and its keyboard together ([`augment`]), takes the
//!    word's key path (a key typed twice running once) as the CTC target,
//!    and resamples the swipe to the model's input ([`features`]).
//! 3. Sums the CTC gradients over the batch (on every core with
//!    `parallel`), leaving out swipes too short for their word.
//! 4. Averages, clips to [`TrainConfig::clip`], and takes an AdamW step at
//!    the schedule's rate ([`Trainer::lr_at`]).
//! 5. Updates the moving average of the weights ([`TrainConfig::ema`]),
//!    which is what evaluation and export use: it is smoother than the
//!    last step's weights and scores better on held-out swipes.
//!
//! Each swipe of a batch draws its own randomness from the seed, the step
//! and its place in the batch, so a run is the same on one thread or many,
//! and a run resumed from its [state](Trainer::save_state) — weights,
//! optimiser moments, moving average and step — carries on as if it had
//! never stopped.
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

use std::collections::HashMap;

use crate::augment::{augment, AugmentConfig};
use crate::ctc;
use crate::data::Recordings;
use crate::decode::{DecodeConfig, Decoder, Lexicon};
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
    /// Decay of the weights' moving average (0: none, the last step's
    /// weights are the model). Ramped up from 0 over the first steps, so
    /// the average does not remember the untrained start.
    pub ema: f32,
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
            ema: 0.999,
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
    /// The weights' moving average ([`TrainConfig::ema`]).
    ema: Option<Vec<f32>>,
}

/// Swipes summed together, in order, before groups are added up
/// ([`Trainer::batch_gradient`]).
const GROUP: usize = 8;

/// What [`Trainer::save_state`] writes, besides the tensors.
const STATE_FORMAT: &str = "threers-swipe-state/1";

impl Trainer {
    pub fn new(config: TrainConfig) -> Self {
        let model = Model::new(config.model.clone(), config.seed);
        Self::resume(config, model, 0)
    }

    /// Carries on training `model` from `step`, the optimiser's moments and
    /// the moving average starting afresh; [`load_state`](Self::load_state)
    /// keeps them.
    pub fn resume(config: TrainConfig, model: Model, step: usize) -> Self {
        let n = model.len();
        let ema = (config.ema > 0.0).then(|| model.params.clone());
        Self {
            config,
            model,
            step,
            synthetic: vec![],
            recorded: None,
            m: vec![0.0; n],
            v: vec![0.0; n],
            ema,
        }
    }

    /// The model to evaluate and export: the moving average of the
    /// weights, or the last step's without one.
    pub fn averaged(&self) -> Model {
        let mut m = self.model.clone();
        if let Some(e) = &self.ema {
            m.params.clone_from(e);
        }
        m
    }

    /// Everything a run needs to carry on exactly: weights, the
    /// optimiser's moments, the moving average and the step, as
    /// safetensors (`params`, `adam.m`, `adam.v`, `ema`), the model config
    /// and step in the metadata.
    pub fn state_bytes(&self) -> Result<Vec<u8>, String> {
        use safetensors::{tensor::TensorView, Dtype};
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut parts: Vec<(&str, Vec<u8>)> = vec![
            ("params", bytes(&self.model.params)),
            ("adam.m", bytes(&self.m)),
            ("adam.v", bytes(&self.v)),
        ];
        if let Some(e) = &self.ema {
            parts.push(("ema", bytes(e)));
        }
        let n = self.model.len();
        let views = parts
            .iter()
            .map(|(name, b)| {
                TensorView::new(Dtype::F32, vec![n], b)
                    .map(|v| (name.to_string(), v))
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut meta = HashMap::new();
        meta.insert("format".into(), STATE_FORMAT.into());
        meta.insert("step".into(), self.step.to_string());
        meta.insert(
            "config".into(),
            serde_json::to_string(&self.model.config).map_err(|e| e.to_string())?,
        );
        safetensors::serialize(views, Some(meta)).map_err(|e| e.to_string())
    }

    /// Writes [`state_bytes`](Self::state_bytes) to `path`, through a
    /// temporary file, so a run stopped mid-write keeps its last state.
    pub fn save_state(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let path = path.as_ref();
        let tmp = path.with_extension("partial");
        std::fs::write(&tmp, self.state_bytes()?).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// A trainer carrying on from a [saved state](Self::save_state); the
    /// model's size comes from the state, the rest of `config` as given.
    pub fn load_state(
        config: TrainConfig,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, String> {
        use safetensors::{Dtype, SafeTensors};
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (_, meta) = SafeTensors::read_metadata(&bytes).map_err(|e| e.to_string())?;
        let meta = meta.metadata().clone().unwrap_or_default();
        if meta.get("format").map(String::as_str) != Some(STATE_FORMAT) {
            return Err(format!(
                "{}: not a threers-swipe training state",
                path.display()
            ));
        }
        let model_config: ModelConfig = meta
            .get("config")
            .and_then(|c| serde_json::from_str(c).ok())
            .ok_or("training state without a model config")?;
        let step: usize = meta.get("step").and_then(|s| s.parse().ok()).unwrap_or(0);
        let st = SafeTensors::deserialize(&bytes).map_err(|e| e.to_string())?;
        let mut model = Model::new(model_config.clone(), 0);
        let n = model.len();
        let read = |name: &str| -> Result<Option<Vec<f32>>, String> {
            let Ok(t) = st.tensor(name) else {
                return Ok(None);
            };
            if t.dtype() != Dtype::F32 || t.shape() != [n] {
                return Err(format!("{name}: expected f32 [{n}]"));
            }
            Ok(Some(
                t.data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
            ))
        };
        model.params = read("params")?.ok_or("training state without params")?;
        let mut t = Self::resume(
            TrainConfig {
                model: model_config,
                ..config
            },
            model,
            step,
        );
        t.m = read("adam.m")?.unwrap_or(t.m);
        t.v = read("adam.v")?.unwrap_or(t.v);
        if t.ema.is_some() {
            t.ema = Some(read("ema")?.unwrap_or_else(|| t.model.params.clone()));
        }
        Ok(t)
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

    /// The batch's summed gradient, summed loss and how many fit. Summed
    /// in fixed groups of [`GROUP`] swipes, each in order, and the groups
    /// in order: the same sums however many threads there are.
    fn batch_gradient(&self) -> (Vec<f32>, f32, usize) {
        let n = self.model.len();
        let batch = self.config.batch;
        let seed = self.config.seed ^ (self.step as u64).wrapping_mul(0x2545_f491_4f6c_dd1d);
        let group = |g: usize| {
            let mut acc = (vec![0.0f32; n], 0.0f32, 0usize);
            for i in g * GROUP..((g + 1) * GROUP).min(batch) {
                let mut rng = Rng::new(seed.wrapping_add(i as u64 * 0x9e37_79b9));
                if let Some(s) = self.sample(&mut rng)
                    && let Some(l) = self.accumulate(&s, &mut acc.0)
                {
                    acc.1 += l;
                    acc.2 += 1;
                }
            }
            acc
        };
        let groups = batch.div_ceil(GROUP);
        #[cfg(feature = "parallel")]
        let parts: Vec<_> = {
            use rayon::prelude::*;
            (0..groups).into_par_iter().map(group).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let parts: Vec<_> = (0..groups).map(group).collect();
        let mut total = (vec![0.0f32; n], 0.0f32, 0usize);
        for (g, l, u) in parts {
            for (x, y) in total.0.iter_mut().zip(&g) {
                *x += y;
            }
            total.1 += l;
            total.2 += u;
        }
        total
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
        if let Some(e) = self.ema.as_mut() {
            // Ramped: (1 + t) / (10 + t) early on, then the set decay.
            let t = self.step as f32;
            let d = self.config.ema.min((1.0 + t) / (10.0 + t));
            for (a, p) in e.iter_mut().zip(&self.model.params) {
                *a = d * *a + (1.0 - d) * p;
            }
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

/// A swipe to score: its decoder, the keyboard it was on (when that is
/// not the decoder's own), its points and its word.
type Case<'a> = (
    &'a Decoder,
    Option<&'a KeyboardGeometry>,
    Vec<crate::TouchPoint>,
    String,
);

/// Scores `cases`, on every core with `parallel`.
fn score(model: &Model, cases: Vec<Case<'_>>, unknown: usize) -> Accuracy {
    let one = |(dec, kb, pts, word): &Case<'_>| {
        let got = match kb {
            Some(g) => dec.decode_on(model, g, pts, 3).unwrap_or_default(),
            None => dec.decode(model, pts, 3),
        };
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
        .map(|(w, p)| (&dec, None, p, w.to_owned()))
        .collect();
    score(model, cases, 0)
}

/// Accuracy on recorded swipes against `lexicon`; a word it lacks counts
/// as a miss.
pub fn evaluate_recorded(model: &Model, recs: &Recordings, lexicon: &Lexicon) -> Accuracy {
    evaluate_recorded_with(model, recs, lexicon, &DecodeConfig::default())
}

/// [`evaluate_recorded`] with these decoder settings. Keyboards with the
/// same keys share one trie of the lexicon.
pub fn evaluate_recorded_with(
    model: &Model,
    recs: &Recordings,
    lexicon: &Lexicon,
    config: &DecodeConfig,
) -> Accuracy {
    let mut tries: HashMap<String, usize> = HashMap::new();
    let mut decoders: Vec<Decoder> = vec![];
    let which: Vec<usize> = recs
        .keyboards
        .iter()
        .map(|g| {
            let id: String = g
                .keys
                .iter()
                .map(|k| k.text.as_str())
                .collect::<Vec<_>>()
                .join(",");
            *tries.entry(id).or_insert_with(|| {
                decoders.push(Decoder::new(g, lexicon).with_config(config.clone()));
                decoders.len() - 1
            })
        })
        .collect();
    let known: std::collections::HashSet<&str> =
        lexicon.words.iter().map(|(w, _)| w.as_str()).collect();
    let (mut cases, mut unknown) = (vec![], 0);
    for r in &recs.items {
        if known.contains(r.word.to_lowercase().as_str()) {
            cases.push((
                &decoders[which[r.keyboard]],
                Some(&recs.keyboards[r.keyboard]),
                r.points.clone(),
                r.word.clone(),
            ));
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
    fn a_saved_state_carries_on_exactly() {
        let g = KeyboardGeometry::from_config(
            &KeyboardConfig::iphone(),
            &KeyboardLayout::english(),
            390.0,
        );
        let words = ["hello", "world", "swipe"].map(String::from).to_vec();
        let cfg = TrainConfig {
            model: ModelConfig::tiny(),
            steps: 12,
            batch: 4,
            ..TrainConfig::default()
        };
        let make = |t: &mut Trainer| t.add_synthetic(g.clone(), words.clone());
        // Straight through…
        let mut a = Trainer::new(cfg.clone());
        make(&mut a);
        a.run(|_, _| {});
        // …and stopped at 5, saved, loaded, finished.
        let mut b = Trainer::new(cfg.clone());
        make(&mut b);
        for _ in 0..5 {
            b.train_step();
        }
        let dir = std::env::temp_dir().join(format!("threers-swipe-state-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.safetensors");
        b.save_state(&path).unwrap();
        let mut c = Trainer::load_state(cfg, &path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(c.step, 5);
        make(&mut c);
        c.run(|_, _| {});
        assert_eq!(c.model.params, a.model.params, "the same weights");
        assert_eq!(c.averaged().params, a.averaged().params, "the same average");
        assert_ne!(a.averaged().params, a.model.params);
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
