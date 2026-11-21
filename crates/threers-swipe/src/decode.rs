//! From the model's scores to words: a beam search over the language's
//! words only, weighted by how common each is.
//!
//! The words become a trie of key paths (a key typed twice running once,
//! as a swipe shows it), and CTC prefix beam search walks it, so every
//! hypothesis is the start of a real word and the suggestions while the
//! finger is still down are too. A word's score is its CTC
//! log-probability plus [`DecodeConfig::lm_weight`] times its log
//! frequency: "of" over "if" when the swipe cannot tell. Given the words
//! before it and a [`WordModel`], the log-probability of the word after
//! them instead (for a word the model does not know, its share by
//! frequency of the model's probability of an unknown word): "to" over
//! "too" after "I'm going".

use std::collections::HashMap;
use std::path::Path;

use crate::ctc::log_softmax;
use crate::geometry::KeyboardGeometry;
use crate::model::Model;
use crate::next_word::WordModel;
use crate::trace::{features, TouchPoint};

/// Words and how common they are.
#[derive(Debug, Clone, Default)]
pub struct Lexicon {
    /// Each word and its log-probability among them.
    pub words: Vec<(String, f32)>,
}

impl Lexicon {
    /// From words and counts (any scale); words are lower-cased, repeats
    /// added up.
    pub fn from_counts<S: AsRef<str>>(counts: impl IntoIterator<Item = (S, f32)>) -> Self {
        let mut merged: HashMap<String, f32> = HashMap::new();
        let mut order = vec![];
        for (w, c) in counts {
            let w = w.as_ref().trim().to_lowercase();
            if w.is_empty() || c <= 0.0 {
                continue;
            }
            if !merged.contains_key(&w) {
                order.push(w.clone());
            }
            *merged.entry(w).or_default() += c;
        }
        let total: f32 = merged.values().sum();
        Self {
            words: order
                .into_iter()
                .map(|w| {
                    let c = merged[&w];
                    (w, (c / total).ln())
                })
                .collect(),
        }
    }

    /// One word a line, optionally followed by its count (`the 23135851`).
    /// Without counts, the order is taken as rank (Zipf: count ∝ 1/rank).
    pub fn parse(text: &str) -> Self {
        Self::from_counts(text.lines().enumerate().filter_map(|(rank, line)| {
            let mut it = line.split_whitespace();
            let w = it.next()?;
            let c = it
                .next()
                .and_then(|c| c.parse::<f32>().ok())
                .unwrap_or(1.0 / (rank + 1) as f32);
            Some((w.to_owned(), c))
        }))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let p = path.as_ref();
        std::fs::read_to_string(p)
            .map(|t| Self::parse(&t))
            .map_err(|e| format!("{}: {e}", p.display()))
    }

    /// The words alone, for [`synth`](crate::synth).
    pub fn word_list(&self) -> Vec<String> {
        self.words.iter().map(|(w, _)| w.clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
}

/// How wide and how fussy the search is.
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// Hypotheses kept per point.
    pub beam: usize,
    /// How much a word's frequency counts against the swipe.
    pub lm_weight: f32,
    /// A key scoring this far (log) under the point's best is not tried.
    pub prune: f32,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            beam: 48,
            lm_weight: 0.5,
            prune: 10.0,
        }
    }
}

/// A word a swipe may be.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub word: String,
    /// What it was ranked by: `ctc + lm_weight · log p(word)`, `p` the
    /// word's frequency or, with a [`WordModel`], its probability after
    /// the words before.
    pub score: f32,
    /// log p(swipe ↦ the word's keys).
    pub ctc: f32,
}

#[derive(Debug, Clone, Default)]
struct Node {
    /// Model class (key + 1) of the key leading here; 0 at the root.
    label: usize,
    children: Vec<(usize, u32)>,
    /// Lexicon indices of the words ending here.
    words: Vec<u32>,
}

/// A lexicon laid over one keyboard, ready to decode its swipes.
#[derive(Debug, Clone)]
pub struct Decoder {
    keys: Vec<(f32, f32)>,
    nodes: Vec<Node>,
    words: Vec<(String, f32)>,
    pub config: DecodeConfig,
    /// Lexicon words with a letter the keyboard has no key for.
    pub skipped: usize,
}

fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

impl Decoder {
    pub fn new(keyboard: &KeyboardGeometry, lexicon: &Lexicon) -> Self {
        let mut nodes = vec![Node::default()];
        let mut words = vec![];
        let mut skipped = 0;
        for (w, lp) in &lexicon.words {
            let Some(path) = keyboard.word_path(w) else {
                skipped += 1;
                continue;
            };
            let mut at = 0usize;
            for k in path {
                let label = k + 1;
                at = match nodes[at].children.iter().find(|c| c.0 == label) {
                    Some(&(_, n)) => n as usize,
                    None => {
                        nodes.push(Node {
                            label,
                            ..Node::default()
                        });
                        let n = nodes.len() - 1;
                        nodes[at].children.push((label, n as u32));
                        n
                    }
                };
            }
            nodes[at].words.push(words.len() as u32);
            words.push((w.clone(), *lp));
        }
        Self {
            keys: keyboard.keys.iter().map(|k| (k.x, k.y)).collect(),
            nodes,
            words,
            config: DecodeConfig::default(),
            skipped,
        }
    }

    pub fn with_config(mut self, config: DecodeConfig) -> Self {
        self.config = config;
        self
    }

    /// The keyboard's key centres, in the model's class order.
    pub fn keys(&self) -> &[(f32, f32)] {
        &self.keys
    }

    /// The `n` likeliest words for a swipe in the keyboard's key units
    /// ([`KeyboardGeometry::to_units`]), best first.
    pub fn decode(&self, model: &Model, points: &[TouchPoint], n: usize) -> Vec<Candidate> {
        let x = features(points);
        self.decode_logits(&model.logits(&x, &self.keys), n)
    }

    /// As [`decode`](Self::decode), on `keyboard` as it is now: the same
    /// keys in the same order as the decoder was made for, moved (a
    /// keyboard resized, split or floating). `None` when its keys differ.
    pub fn decode_on(
        &self,
        model: &Model,
        keyboard: &KeyboardGeometry,
        points: &[TouchPoint],
        n: usize,
    ) -> Option<Vec<Candidate>> {
        if keyboard.keys.len() != self.keys.len() {
            return None;
        }
        let keys: Vec<(f32, f32)> = keyboard.keys.iter().map(|k| (k.x, k.y)).collect();
        let x = features(points);
        Some(self.decode_logits(&model.logits(&x, &keys), n))
    }

    /// As [`decode_on`](Self::decode_on), the words ranked by how likely
    /// `words` thinks each after `context` (oldest first, as
    /// [`split_context`](crate::next_word::split_context) gives it).
    pub fn decode_on_after(
        &self,
        model: &Model,
        keyboard: &KeyboardGeometry,
        points: &[TouchPoint],
        words: &dyn WordModel,
        context: &[&str],
        n: usize,
    ) -> Option<Vec<Candidate>> {
        if keyboard.keys.len() != self.keys.len() {
            return None;
        }
        let keys: Vec<(f32, f32)> = keyboard.keys.iter().map(|k| (k.x, k.y)).collect();
        let x = features(points);
        Some(self.decode_logits_after(&model.logits(&x, &keys), words, context, n))
    }

    /// As [`decode`](Self::decode), from the model's logits.
    pub fn decode_logits(&self, logits: &[f32], n: usize) -> Vec<Candidate> {
        let hits = self.search(logits);
        let freq: Vec<f32> = hits.iter().map(|&(w, _)| self.words[w as usize].1).collect();
        self.rank(hits, &freq, n)
    }

    /// As [`decode_logits`](Self::decode_logits), each word's frequency
    /// replaced by its log-probability after `context` under `words`.
    pub fn decode_logits_after(
        &self,
        logits: &[f32],
        words: &dyn WordModel,
        context: &[&str],
        n: usize,
    ) -> Vec<Candidate> {
        let hits = self.search(logits);
        let spelled: Vec<&str> = hits
            .iter()
            .map(|&(w, _)| self.words[w as usize].0.as_str())
            .collect();
        let freq: HashMap<&str, f32> = hits
            .iter()
            .map(|&(w, _)| (self.words[w as usize].0.as_str(), self.words[w as usize].1))
            .collect();
        let lm = crate::next_word::log_probs_or(words, context, &spelled, |w| freq[w]);
        self.rank(hits, &lm, n)
    }

    fn rank(&self, hits: Vec<(u32, f32)>, lm: &[f32], n: usize) -> Vec<Candidate> {
        let mut out: Vec<Candidate> = hits
            .into_iter()
            .zip(lm)
            .map(|((w, ctc), lp)| Candidate {
                word: self.words[w as usize].0.clone(),
                score: ctc + self.config.lm_weight * lp,
                ctc,
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.truncate(n);
        out
    }

    /// The beam's words and their CTC log-probabilities.
    fn search(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        let classes = self.keys.len() + 1;
        let lp = log_softmax(logits, classes);
        let ninf = f64::NEG_INFINITY;
        // node → (ends in blank, ends in its key)
        let mut beams: Vec<(u32, f64, f64)> = vec![(0, 0.0, ninf)];
        let mut next: HashMap<u32, (f64, f64)> = HashMap::new();
        for row in lp.chunks(classes) {
            next.clear();
            let best = row[1..].iter().copied().fold(f32::MIN, f32::max);
            let floor = best - self.config.prune;
            for &(id, pb, pnb) in &beams {
                let node = &self.nodes[id as usize];
                let total = lse(pb, pnb);
                let e = next.entry(id).or_insert((ninf, ninf));
                e.0 = lse(e.0, total + row[0] as f64);
                if id != 0 {
                    e.1 = lse(e.1, pnb + row[node.label] as f64);
                }
                for &(label, child) in &node.children {
                    let l = row[label];
                    if l < floor {
                        continue;
                    }
                    let from = if label == node.label { pb } else { total };
                    let e = next.entry(child).or_insert((ninf, ninf));
                    e.1 = lse(e.1, from + l as f64);
                }
            }
            beams.clear();
            beams.extend(next.iter().map(|(&id, &(b, nb))| (id, b, nb)));
            beams.sort_by(|a, b| lse(b.1, b.2).total_cmp(&lse(a.1, a.2)));
            beams.truncate(self.config.beam);
        }
        beams
            .iter()
            .flat_map(|&(id, pb, pnb)| {
                let ctc = lse(pb, pnb) as f32;
                self.nodes[id as usize].words.iter().map(move |&w| (w, ctc))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout};

    fn keyboard() -> KeyboardGeometry {
        KeyboardGeometry::from_config(&KeyboardConfig::iphone(), &KeyboardLayout::english(), 390.0)
    }

    /// Logits that spell `keys` (classes) with blanks between.
    fn spelled(classes: usize, labels: &[usize]) -> Vec<f32> {
        let mut z = vec![];
        for &l in labels {
            for k in [l, l, 0] {
                let mut row = vec![-8.0f32; classes];
                row[k] = 4.0;
                z.extend(row);
            }
        }
        z
    }

    #[test]
    fn finds_the_word_its_keys_spell() {
        let g = keyboard();
        let lex = Lexicon::parse("the 100\nhello 10\nhelo 1\nworld 5\n");
        let d = Decoder::new(&g, &lex);
        let path: Vec<usize> = g
            .word_path("hello")
            .unwrap()
            .iter()
            .map(|k| k + 1)
            .collect();
        let got = d.decode_logits(&spelled(27, &path), 3);
        // "hello" and "helo" share a path; frequency settles it.
        assert_eq!(got[0].word, "hello");
        assert_eq!(got[1].word, "helo");
        assert!(got.iter().all(|c| c.word != "world"));
    }

    #[test]
    fn the_words_before_settle_it() {
        use crate::next_word::WordModel;
        // Knows "to" after "going", "too" after "me".
        struct After;
        impl WordModel for After {
            fn context_len(&self) -> usize {
                1
            }
            fn log_probs(&self, context: &[&str], words: &[&str]) -> Vec<Option<f32>> {
                let likely = if context.last() == Some(&"going") { "to" } else { "too" };
                words
                    .iter()
                    .map(|w| match *w {
                        "to" | "too" => Some(if *w == likely { -0.1 } else { -5.0 }),
                        _ => None,
                    })
                    .collect()
            }
            fn predict(&self, _: &[&str], _: &str, _: usize) -> Vec<(String, f32)> {
                vec![]
            }
        }
        let g = keyboard();
        let d = Decoder::new(&g, &Lexicon::parse("to 100\ntoo 50\n"));
        // A swipe that cannot tell them apart: "too" is "to" on the keys.
        let path: Vec<usize> = g.word_path("to").unwrap().iter().map(|k| k + 1).collect();
        assert_eq!(g.word_path("to"), g.word_path("too"));
        let z = spelled(27, &path);
        assert_eq!(d.decode_logits(&z, 1)[0].word, "to", "the commoner alone");
        assert_eq!(d.decode_logits_after(&z, &After, &["going"], 1)[0].word, "to");
        assert_eq!(d.decode_logits_after(&z, &After, &["me"], 1)[0].word, "too");
    }

    #[test]
    fn lexicon_counts_and_ranks() {
        let l = Lexicon::parse("a\nb\nc 5\n");
        assert_eq!(l.len(), 3);
        assert!(
            l.words[2].1 > l.words[0].1,
            "c's count beats a's rank weight"
        );
        let g = keyboard();
        let d = Decoder::new(&g, &Lexicon::parse("ok\n日本\n"));
        assert_eq!(d.skipped, 1);
    }
}
