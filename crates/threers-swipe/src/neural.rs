//! A small neural next-word model, trained and run by RLX: the last few
//! words' embeddings, one hidden layer, a softmax over the vocabulary
//! (Bengio et al.'s feed-forward language model).
//!
//! ```text
//! words [K] ─ embed [V × E] ─► [K·E] ─ linear ─ tanh ─► [H] ─ linear ─► logits [V]
//! ```
//!
//! It is not the swipe model: that one reads the finger, this one the
//! sentence. One forward pass scores every word at once, so a swipe's
//! candidates and the bar's next words cost the same single pass, and the
//! last context's scores are kept for the next question about it.
//!
//! Weights save as safetensors (`embed`, `hidden_w`, `hidden_b`, `out_w`,
//! `out_b`), the [`NeuralConfig`] and the vocabulary as JSON in the
//! metadata; f32, or f16 for half the size ([`NeuralLm::to_safetensors_f16`]),
//! read back as f32 either way.

use std::cell::RefCell;
use std::collections::HashMap;

use rlx::{grad_with_loss, DType, Device, Graph, GraphExt, Session, Shape};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::{Deserialize, Serialize};

use crate::next_word::{normalize, sentences, WordModel, BOS, EOS, UNK};
use crate::synth::Rng;

const UNK_ID: u32 = 0;
const BOS_ID: u32 = 1;
const EOS_ID: u32 = 2;

/// The network's size.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Words read before the next one.
    pub context: usize,
    /// Width of a word's embedding.
    pub embed: usize,
    /// Width of the hidden layer.
    pub hidden: usize,
}

impl NeuralConfig {
    /// With 10 000 words, about 1.6 M parameters, 6.4 MB f32.
    pub fn small() -> Self {
        Self {
            context: 3,
            embed: 32,
            hidden: 128,
        }
    }

    /// For tests and quick tries.
    pub fn tiny() -> Self {
        Self {
            context: 2,
            embed: 8,
            hidden: 16,
        }
    }
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self::small()
    }
}

/// How it is trained.
#[derive(Debug, Clone)]
pub struct NeuralTrainConfig {
    pub model: NeuralConfig,
    /// Most words kept, the commonest; the rest are `<unk>`.
    pub vocab: usize,
    pub steps: usize,
    pub batch: usize,
    /// Adam's.
    pub lr: f32,
    pub seed: u64,
    /// Where RLX runs it.
    pub device: Device,
}

impl Default for NeuralTrainConfig {
    fn default() -> Self {
        Self {
            model: NeuralConfig::default(),
            vocab: 10_000,
            steps: 3_000,
            batch: 128,
            lr: 3e-3,
            seed: 1,
            device: Device::Cpu,
        }
    }
}

/// The parameters by name and shape, in the order the gradients come.
fn specs(c: &NeuralConfig, vocab: usize) -> [(&'static str, Vec<usize>); 5] {
    [
        ("embed", vec![vocab, c.embed]),
        ("hidden_w", vec![c.context * c.embed, c.hidden]),
        ("hidden_b", vec![c.hidden]),
        ("out_w", vec![c.hidden, vocab]),
        ("out_b", vec![vocab]),
    ]
}

/// The forward pass on `context [batch, K]` (word ids as f32) to logits
/// `[batch, V]`.
fn forward(graph: &mut Graph, c: &NeuralConfig, vocab: usize, batch: usize) -> (rlx::NodeId, rlx::NodeId) {
    let p: Vec<rlx::NodeId> = specs(c, vocab)
        .iter()
        .map(|(n, s)| graph.param(*n, Shape::new(s, DType::F32)))
        .collect();
    let ids = graph.input("context", Shape::new(&[batch, c.context], DType::F32));
    let e = graph.gather_(p[0], ids, 0);
    let x = graph.reshape_(e, vec![batch as i64, (c.context * c.embed) as i64]);
    let h = graph.mm(x, p[1]);
    let h = graph.add(h, p[2]);
    let h = graph.tanh(h);
    let z = graph.mm(h, p[3]);
    (ids, graph.add(z, p[4]))
}

fn log_softmax(z: &mut [f32]) {
    let m = z.iter().copied().fold(f32::MIN, f32::max);
    let lse = m + z.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
    for v in z {
        *v -= lse;
    }
}

/// A trained model, ready to score words.
pub struct NeuralLm {
    config: NeuralConfig,
    /// By id: `<unk>`, `<s>`, `</s>`, then the words, commonest first.
    words: Vec<String>,
    ids: HashMap<String, u32>,
    params: Vec<Vec<f32>>,
    device: Device,
    /// The one-context forward pass, compiled on first use.
    run: RefCell<Option<rlx::CompiledGraph>>,
    /// The last context asked about and its log-probabilities.
    last: RefCell<Option<(Vec<u32>, Vec<f32>)>>,
}

impl std::fmt::Debug for NeuralLm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralLm")
            .field("config", &self.config)
            .field("vocab", &self.vocab_len())
            .finish_non_exhaustive()
    }
}

impl NeuralLm {
    fn new(config: NeuralConfig, words: Vec<String>, params: Vec<Vec<f32>>, device: Device) -> Self {
        let ids = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32))
            .collect();
        Self {
            config,
            words,
            ids,
            params,
            device,
            run: RefCell::default(),
            last: RefCell::default(),
        }
    }

    /// Trains on `text`'s sentences, calling `progress(step, loss)`.
    pub fn train(text: &str, config: &NeuralTrainConfig, progress: impl FnMut(usize, f32)) -> Result<Self, String> {
        Self::train_sentences(&sentences(text), config, progress)
    }

    pub fn train_sentences(
        sentences: &[Vec<String>],
        config: &NeuralTrainConfig,
        mut progress: impl FnMut(usize, f32),
    ) -> Result<Self, String> {
        let c = &config.model;
        let mut freq: HashMap<&str, u64> = HashMap::new();
        for s in sentences {
            for w in s {
                *freq.entry(w.as_str()).or_default() += 1;
            }
        }
        let mut vocab: Vec<(&str, u64)> = freq.into_iter().collect();
        vocab.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        vocab.truncate(config.vocab.max(1));
        let mut words: Vec<String> = [UNK, BOS, EOS].map(String::from).to_vec();
        words.extend(vocab.iter().map(|(w, _)| w.to_string()));
        let v = words.len();
        let ids: HashMap<&str, u32> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        // Every word (and each sentence's end) after its K words.
        let mut examples: Vec<(Vec<u32>, u32)> = vec![];
        for s in sentences {
            let mut seq = vec![BOS_ID; c.context];
            seq.extend(s.iter().map(|w| ids.get(w.as_str()).copied().unwrap_or(UNK_ID)));
            seq.push(EOS_ID);
            for t in c.context..seq.len() {
                examples.push((seq[t - c.context..t].to_vec(), seq[t]));
            }
        }
        if examples.is_empty() {
            return Err("no words to train on".into());
        }

        let batch = config.batch.max(1);
        let mut g = Graph::new("next_word_loss");
        let (_, logits) = forward(&mut g, c, v, batch);
        let labels = g.input("labels", Shape::new(&[batch], DType::F32));
        let per_row = g.softmax_cross_entropy_with_logits(logits, labels);
        let loss = g.mean(per_row, vec![0], false);
        g.set_outputs(vec![loss]);
        let specs = specs(c, v);
        let wrt: Vec<_> = specs
            .iter()
            .map(|(n, s)| g.param(*n, Shape::new(s, DType::F32)))
            .collect();
        let mut step_graph = Session::new(config.device).compile(grad_with_loss(&g, &wrt));

        let mut rng = Rng::new(config.seed);
        let mut params: Vec<Vec<f32>> = specs
            .iter()
            .map(|(name, s)| {
                let n: usize = s.iter().product();
                match *name {
                    "embed" => (0..n).map(|_| 0.1 * rng.normal()).collect(),
                    "hidden_w" => {
                        let scale = (1.0 / s[0] as f32).sqrt();
                        (0..n).map(|_| scale * rng.normal()).collect()
                    }
                    "out_w" => {
                        let scale = (1.0 / s[0] as f32).sqrt();
                        (0..n).map(|_| scale * rng.normal()).collect()
                    }
                    _ => vec![0.0; n],
                }
            })
            .collect();
        let mut m1: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.len()]).collect();
        let mut m2 = m1.clone();
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let mut ctx = vec![0.0f32; batch * c.context];
        let mut lab = vec![0.0f32; batch];
        for step in 1..=config.steps {
            for b in 0..batch {
                let (x, y) = &examples[(rng.next_u64() % examples.len() as u64) as usize];
                for (k, &id) in x.iter().enumerate() {
                    ctx[b * c.context + k] = id as f32;
                }
                lab[b] = *y as f32;
            }
            for ((name, _), p) in specs.iter().zip(&params) {
                step_graph.set_param(name, p);
            }
            let out = step_graph.run(&[("context", &ctx), ("labels", &lab), ("d_output", &[1.0])]);
            let loss = out.first().and_then(|o| o.first()).copied().unwrap_or(f32::NAN);
            if !loss.is_finite() {
                return Err(format!("step {step}: loss {loss}"));
            }
            let (c1, c2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
            for i in 0..params.len() {
                let grad = &out[i + 1];
                if grad.len() != params[i].len() {
                    return Err(format!(
                        "{}: {} gradients for {} parameters",
                        specs[i].0,
                        grad.len(),
                        params[i].len()
                    ));
                }
                for j in 0..grad.len() {
                    let gr = grad[j];
                    m1[i][j] = b1 * m1[i][j] + (1.0 - b1) * gr;
                    m2[i][j] = b2 * m2[i][j] + (1.0 - b2) * gr * gr;
                    params[i][j] -= config.lr * (m1[i][j] / c1) / ((m2[i][j] / c2).sqrt() + eps);
                }
            }
            progress(step, loss);
        }
        Ok(Self::new(c.clone(), words, params, config.device))
    }

    pub fn config(&self) -> &NeuralConfig {
        &self.config
    }

    /// Words known, `<unk>`, `<s>` and `</s>` not counted.
    pub fn vocab_len(&self) -> usize {
        self.words.len() - 3
    }

    /// Runs it on `device` from now on.
    pub fn on_device(mut self, device: Device) -> Self {
        self.device = device;
        self.run = RefCell::default();
        self
    }

    /// The K ids before the next word: a sentence's start padded with
    /// `<s>`, an unknown or short context with `<unk>` before it.
    fn context_ids(&self, context: &[&str]) -> Vec<u32> {
        let k = self.config.context;
        let keep = context.len().min(k);
        let mut ids: Vec<u32> = context[context.len() - keep..]
            .iter()
            .map(|w| self.ids.get(*w).copied().unwrap_or(UNK_ID))
            .collect();
        let pad = if ids.first() == Some(&BOS_ID) { BOS_ID } else { UNK_ID };
        ids.splice(0..0, std::iter::repeat_n(pad, k - ids.len()));
        ids
    }

    /// log p(each word | `ids`), one forward pass, the last one kept.
    fn distribution(&self, ids: &[u32]) -> Vec<f32> {
        if let Some((at, lp)) = self.last.borrow().as_ref()
            && at == ids
        {
            return lp.clone();
        }
        let mut run = self.run.borrow_mut();
        let graph = run.get_or_insert_with(|| {
            let mut g = Graph::new("next_word");
            let (_, logits) = forward(&mut g, &self.config, self.words.len(), 1);
            g.set_outputs(vec![logits]);
            let mut compiled = Session::new(self.device).compile(g);
            for ((name, _), p) in specs(&self.config, self.words.len()).iter().zip(&self.params) {
                compiled.set_param(name, p);
            }
            compiled
        });
        let x: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        let mut z = graph.run(&[("context", &x)]).swap_remove(0);
        log_softmax(&mut z);
        *self.last.borrow_mut() = Some((ids.to_vec(), z.clone()));
        z
    }

    /// The weights as f32 safetensors, the config and vocabulary in the
    /// metadata.
    pub fn to_safetensors(&self) -> Result<Vec<u8>, String> {
        self.serialize(Dtype::F32)
    }

    /// As [`to_safetensors`](Self::to_safetensors), the weights f16: half
    /// the size, scores within about 1e-3 of the f32 ones.
    pub fn to_safetensors_f16(&self) -> Result<Vec<u8>, String> {
        self.serialize(Dtype::F16)
    }

    fn serialize(&self, dtype: Dtype) -> Result<Vec<u8>, String> {
        let specs = specs(&self.config, self.words.len());
        let bytes: Vec<Vec<u8>> = self
            .params
            .iter()
            .map(|p| match dtype {
                Dtype::F16 => p
                    .iter()
                    .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                    .collect(),
                _ => p.iter().flat_map(|v| v.to_le_bytes()).collect(),
            })
            .collect();
        let views = specs
            .iter()
            .zip(&bytes)
            .map(|((n, s), b)| {
                Ok((
                    n.to_string(),
                    TensorView::new(dtype, s.clone(), b).map_err(|e| e.to_string())?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut meta = HashMap::new();
        meta.insert(
            "config".into(),
            serde_json::to_string(&self.config).map_err(|e| e.to_string())?,
        );
        meta.insert(
            "vocab".into(),
            serde_json::to_string(&self.words).map_err(|e| e.to_string())?,
        );
        meta.insert("format".into(), "threers-next-word/1".into());
        safetensors::serialize(views, Some(meta)).map_err(|e| e.to_string())
    }

    /// Runs on the CPU; [`on_device`](Self::on_device) moves it.
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, String> {
        let (_, meta) = SafeTensors::read_metadata(bytes).map_err(|e| e.to_string())?;
        let meta = meta
            .metadata()
            .as_ref()
            .ok_or("no metadata: not a threers next-word model")?;
        let config: NeuralConfig = meta
            .get("config")
            .ok_or("no config in the metadata: not a threers next-word model")
            .and_then(|s| serde_json::from_str(s).map_err(|_| "unreadable config"))?;
        let words: Vec<String> = meta
            .get("vocab")
            .ok_or("no vocabulary in the metadata")
            .and_then(|s| serde_json::from_str(s).map_err(|_| "unreadable vocabulary"))?;
        if words.len() < 3 || words[..3] != [UNK, BOS, EOS] {
            return Err("the vocabulary does not start <unk> <s> </s>".into());
        }
        let st = SafeTensors::deserialize(bytes).map_err(|e| e.to_string())?;
        let params = specs(&config, words.len())
            .iter()
            .map(|(name, shape)| {
                let t = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
                if !matches!(t.dtype(), Dtype::F32 | Dtype::F16) || t.shape() != shape.as_slice() {
                    return Err(format!(
                        "{name}: expected f32 or f16 {shape:?}, found {:?} {:?}",
                        t.dtype(),
                        t.shape()
                    ));
                }
                Ok(match t.dtype() {
                    Dtype::F16 => t
                        .data()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| half::f16::from_le_bytes(*b).to_f32())
                        .collect(),
                    _ => t
                        .data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                })
            })
            .collect::<Result<Vec<Vec<f32>>, String>>()?;
        Ok(Self::new(config, words, params, Device::Cpu))
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        std::fs::write(path, self.to_safetensors()?).map_err(|e| e.to_string())
    }

    /// Saves the f16 form ([`to_safetensors_f16`](Self::to_safetensors_f16)).
    pub fn save_f16(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        std::fs::write(path, self.to_safetensors_f16()?).map_err(|e| e.to_string())
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        Self::from_safetensors(&std::fs::read(path).map_err(|e| e.to_string())?)
    }
}

impl WordModel for NeuralLm {
    fn context_len(&self) -> usize {
        self.config.context
    }

    fn unknown_log_prob(&self, context: &[&str]) -> Option<f32> {
        Some(self.distribution(&self.context_ids(context))[UNK_ID as usize])
    }

    fn log_probs(&self, context: &[&str], words: &[&str]) -> Vec<Option<f32>> {
        let lp = self.distribution(&self.context_ids(context));
        words
            .iter()
            .map(|w| {
                let id = *self.ids.get(*w)?;
                (id > EOS_ID).then(|| lp[id as usize])
            })
            .collect()
    }

    fn predict(&self, context: &[&str], prefix: &str, n: usize) -> Vec<(String, f32)> {
        let lp = self.distribution(&self.context_ids(context));
        let prefix = normalize(prefix);
        let mut out: Vec<(u32, f32)> = (EOS_ID + 1..self.words.len() as u32)
            .filter(|&w| prefix.is_empty() || self.words[w as usize].starts_with(&prefix))
            .map(|w| (w, lp[w as usize]))
            .collect();
        let best = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if out.len() > n {
            out.select_nth_unstable_by(n, best);
            out.truncate(n);
        }
        out.sort_by(best);
        out.into_iter()
            .map(|(w, p)| (self.words[w as usize].clone(), p))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "I am going to the shop. I am going to bed. \
        It is too late. Me too. I want to go to the park. \
        Going to the beach is fun. I am going to swim.";

    fn trained() -> NeuralLm {
        let config = NeuralTrainConfig {
            model: NeuralConfig::tiny(),
            steps: 400,
            batch: 16,
            lr: 2e-2,
            ..Default::default()
        };
        let mut first = None;
        let mut last = 0.0;
        let lm = NeuralLm::train(TEXT, &config, |_, l| {
            first.get_or_insert(l);
            last = l;
        })
        .unwrap();
        assert!(last < 0.5 * first.unwrap(), "loss {} → {last}", first.unwrap());
        lm
    }

    #[test]
    fn learns_what_follows() {
        let lm = trained();
        let p = lm.log_probs(&["am", "going"], &["to", "too"]);
        assert!(p[0].unwrap() > p[1].unwrap() + 1.0, "{p:?}");
        assert_eq!(lm.predict(&["going", "to"], "", 1)[0].0, "the");
        assert_eq!(lm.predict(&[BOS], "", 1)[0].0, "i");
        assert_eq!(lm.predict(&["the"], "pa", 1)[0].0, "park");
        assert_eq!(lm.log_probs(&[], &["zebra", BOS]), [None, None]);
        // A distribution over the vocabulary and the end.
        let all: Vec<&str> = lm.words[3..].iter().map(String::as_str).collect();
        let lp = lm.distribution(&lm.context_ids(&["going"]));
        let sum: f32 = lp.iter().map(|l| l.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert_eq!(lm.log_probs(&["going"], &all).len(), all.len());
    }

    #[test]
    fn saves_and_loads() {
        let lm = trained();
        let back = NeuralLm::from_safetensors(&lm.to_safetensors().unwrap()).unwrap();
        let ctx = ["am", "going"];
        assert_eq!(lm.log_probs(&ctx, &["to", "the"]), back.log_probs(&ctx, &["to", "the"]));
        assert!(NeuralLm::from_safetensors(b"nonsense").is_err());
        let small = lm.to_safetensors_f16().unwrap();
        assert!(small.len() < lm.to_safetensors().unwrap().len() * 6 / 10);
        let half = NeuralLm::from_safetensors(&small).unwrap();
        for (a, b) in lm.log_probs(&ctx, &["to", "the"]).iter().zip(half.log_probs(&ctx, &["to", "the"])) {
            assert!((a.unwrap() - b.unwrap()).abs() < 1e-2, "{a:?} {b:?}");
        }
    }
}
