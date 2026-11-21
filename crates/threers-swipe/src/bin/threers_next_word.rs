//! `threers-next-word`: train and score the next-word models, alone and
//! as context for swipes.
//!
//! ```text
//! threers-next-word ngram  --text sentences.txt --out en.ngram [--order 3] [--vocab 40000] [--min-count 1]
//! threers-next-word neural --text sentences.txt --out en.next.safetensors
//!                          [--size small|tiny] [--vocab 10000] [--steps 3000] [--batch 128] [--lr 3e-3] [--f16 1]
//! threers-next-word convert --neural en.next.safetensors --out en.next.f16.safetensors [--f16 1]
//! threers-next-word eval   --text held-out.txt [--ngram en.ngram] [--neural en.next.safetensors]
//!                          [--mix 0.5] [--n 20000] [--learn 0.2]
//! threers-next-word swipes --model swipe.safetensors --words words_en.txt --held-out dev.jsonl
//!                          [--futo-layout qwerty.json] [--ngram …] [--neural …] [--mix 0.5] [--n 3000]
//! threers-next-word bench  --text held-out.txt [--ngram …] [--neural …] [--n 300]
//! threers-next-word demo   --ngram … [--neural …] [--type "I am going to|…"]
//!                          [--model … --words … --held-out test.jsonl --futo-layout …]
//! ```
//!
//! Text: one sentence a line, or FUTO's JSON lines (their `sentence`s,
//! each once). `eval` scores the bar: the word after a space, and after
//! its first letter. `swipes` decodes FUTO's recorded swipes after the
//! words before them in their sentence, against the word list alone, for
//! a range of `lm_weight`s. `ngram` needs the `ngram` feature, `neural`
//! the `rlx` feature.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::BufRead;
use std::process::ExitCode;
use std::rc::Rc;
use std::time::Instant;

use serde::Deserialize;
use threers_swipe::data::{FutoKeyboard, Recordings};
use threers_swipe::decode::DecodeConfig;
use threers_swipe::next_word::{self, Interpolated, WordModel};
use threers_swipe::user::{UserModel, WithUser};
use threers_swipe::{Decoder, KeyboardGeometry, Lexicon, Model, TouchPoint};
use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout};

struct Args {
    flags: BTreeMap<String, String>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Self, String> {
        let mut flags = BTreeMap::new();
        let mut it = raw.iter();
        while let Some(a) = it.next() {
            let name = a
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected `{a}`"))?;
            let value = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
            flags.insert(name.to_owned(), value.clone());
        }
        Ok(Self { flags })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    fn num<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("--{name}: cannot read `{v}`")),
        }
    }

    fn need(&self, name: &str) -> Result<&str, String> {
        self.get(name).ok_or_else(|| format!("--{name} is needed"))
    }
}

/// A file's sentences: FUTO's `sentence`s, each once, or its lines.
fn read_sentences(path: &str) -> Result<Vec<String>, String> {
    #[derive(Deserialize)]
    struct S {
        sentence: Option<String>,
    }
    let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut seen = HashSet::new();
    let mut out = vec![];
    for line in std::io::BufReader::with_capacity(1 << 20, f).lines() {
        let line = line.map_err(|e| format!("{path}: {e}"))?;
        let s = if line.starts_with('{') {
            match serde_json::from_str::<S>(&line) {
                Ok(S { sentence: Some(s) }) => s,
                _ => continue,
            }
        } else {
            line
        };
        let s = s.trim().to_owned();
        if !s.is_empty() && seen.insert(s.clone()) {
            out.push(s);
        }
    }
    Ok(out)
}

/// Words of each sentence, one sentence a line however it is punctuated.
fn words_of(sentences: &[String]) -> Vec<Vec<String>> {
    sentences
        .iter()
        .map(|s| next_word::sentences(s).concat())
        .filter(|w| !w.is_empty())
        .collect()
}

#[cfg(feature = "ngram")]
fn cmd_ngram(args: &Args) -> Result<(), String> {
    use threers_swipe::ngram::{NgramConfig, NgramModel};
    let text = words_of(&read_sentences(args.need("text")?)?);
    let out = args.need("out")?;
    let d = NgramConfig::default();
    let config = NgramConfig {
        order: args.num("order", d.order)?,
        vocab: args.num("vocab", d.vocab)?,
        min_count: args.num("min-count", d.min_count)?,
        discount: args.num("discount", d.discount)?,
    };
    let start = Instant::now();
    let lm = NgramModel::train_sentences(&text, &config);
    lm.save(out)?;
    eprintln!(
        "wrote {out}: order {}, {} words, from {} sentences ({:.1}s)",
        lm.order(),
        lm.vocab_len(),
        text.len(),
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

#[cfg(not(feature = "ngram"))]
fn cmd_ngram(_: &Args) -> Result<(), String> {
    Err("built without the `ngram` feature".into())
}

#[cfg(feature = "rlx")]
fn cmd_neural(args: &Args) -> Result<(), String> {
    use threers_swipe::neural::{NeuralConfig, NeuralLm, NeuralTrainConfig};
    let text = words_of(&read_sentences(args.need("text")?)?);
    let out = args.need("out")?;
    let d = NeuralTrainConfig::default();
    let config = NeuralTrainConfig {
        model: match args.get("size").unwrap_or("small") {
            "small" => NeuralConfig::small(),
            "tiny" => NeuralConfig::tiny(),
            s => return Err(format!("--size {s}: small or tiny")),
        },
        vocab: args.num("vocab", d.vocab)?,
        steps: args.num("steps", d.steps)?,
        batch: args.num("batch", d.batch)?,
        lr: args.num("lr", d.lr)?,
        seed: args.num("seed", d.seed)?,
        device: d.device,
    };
    let start = Instant::now();
    let mut mean = 0.0f32;
    let every = (config.steps / 20).max(1);
    let lm = NeuralLm::train_sentences(&text, &config, |step, loss| {
        mean += loss;
        if step % every == 0 {
            eprintln!(
                "step {step:6}  loss {:.3}  {:.0}s",
                mean / every as f32,
                start.elapsed().as_secs_f32()
            );
            mean = 0.0;
        }
    })?;
    if args.get("f16").is_some_and(|v| v != "0") {
        lm.save_f16(out)?;
    } else {
        lm.save(out)?;
    }
    eprintln!("wrote {out}: {} words", lm.vocab_len());
    Ok(())
}

#[cfg(not(feature = "rlx"))]
fn cmd_neural(_: &Args) -> Result<(), String> {
    Err("built without the `rlx` feature".into())
}

/// The models `--ngram` and `--neural` name (each a comma-separated list
/// of files), and with more than one, all of them mixed: `--mix` a weight
/// each (`0.3,0.3,0.4`), or with two one number `w` for `w : 1 − w`, equal
/// shares without it.
fn models(args: &Args) -> Result<Vec<(String, Rc<dyn WordModel>)>, String> {
    let mut out: Vec<(String, Rc<dyn WordModel>)> = vec![];
    #[cfg_attr(not(any(feature = "ngram", feature = "rlx")), allow(unused_variables))]
    let stem = |p: &str| {
        std::path::Path::new(p)
            .file_stem()
            .map_or_else(|| p.to_owned(), |s| s.to_string_lossy().into_owned())
    };
    for p in args.get("ngram").into_iter().flat_map(|l| l.split(',')) {
        #[cfg(feature = "ngram")]
        {
            let m = threers_swipe::ngram::NgramModel::load(p)?;
            out.push((format!("ngram {}", stem(p)), Rc::new(m)));
        }
        #[cfg(not(feature = "ngram"))]
        return Err(format!("{p}: built without the `ngram` feature"));
    }
    for p in args.get("neural").into_iter().flat_map(|l| l.split(',')) {
        #[cfg(feature = "rlx")]
        out.push((format!("neural {}", stem(p)), Rc::new(threers_swipe::neural::NeuralLm::load(p)?)));
        #[cfg(not(feature = "rlx"))]
        return Err(format!("{p}: built without the `rlx` feature"));
    }
    if out.len() > 1 {
        let given: Vec<f32> = match args.get("mix") {
            None => vec![1.0; out.len()],
            Some(m) => m
                .split(',')
                .map(|w| w.trim().parse().map_err(|_| format!("--mix: `{w}`")))
                .collect::<Result<_, _>>()?,
        };
        let weights = match given.as_slice() {
            [w] if out.len() == 2 => vec![*w, 1.0 - w],
            w if w.len() == out.len() => w.to_vec(),
            _ => return Err(format!("--mix: {} weights for {} models", given.len(), out.len())),
        };
        let mix = out
            .iter()
            .zip(&weights)
            .fold(Interpolated::new(), |m, ((_, x), &w)| m.with(x.clone(), w));
        let label = weights.iter().map(|w| format!("{w}")).collect::<Vec<_>>().join(":");
        out.push((format!("mix {label}"), Rc::new(mix)));
    }
    if out.is_empty() {
        return Err("no model: --ngram and/or --neural".into());
    }
    Ok(out)
}

#[cfg(any(feature = "ngram", feature = "rlx"))]
/// Seconds of CPU this thread has used.
#[cfg(unix)]
fn cpu_now() -> f64 {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `t` is a valid timespec for the call to fill.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) };
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
}

#[cfg(any(feature = "ngram", feature = "rlx"))]
#[cfg(not(unix))]
fn cpu_now() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

#[cfg(any(feature = "ngram", feature = "rlx"))]
/// `f` run `n` times: its CPU time per run, p10 / median / p90.
fn timed(name: &str, n: usize, mut f: impl FnMut(usize)) {
    let mut t: Vec<f64> = (0..n)
        .map(|i| {
            let t0 = cpu_now();
            f(i);
            cpu_now() - t0
        })
        .collect();
    t.sort_by(f64::total_cmp);
    let at = |q: f64| t[((t.len() - 1) as f64 * q) as usize] * 1e3;
    println!("{name:<34} {:>9.3} {:>9.3} {:>9.3} ms  ({n} runs)", at(0.1), at(0.5), at(0.9));
}

/// Load and query times of the models, on this thread's CPU clock.
fn cmd_bench(args: &Args) -> Result<(), String> {
    let text = words_of(&read_sentences(args.need("text")?)?);
    let n: usize = args.num("n", 300)?;
    // Contexts and prefixes from real sentences.
    let mut cases: Vec<(Vec<String>, String)> = vec![];
    for s in &text {
        for i in 0..s.len() {
            let mut ctx = vec![next_word::BOS.to_owned()];
            ctx.extend(s[..i].iter().cloned());
            cases.push((ctx, s[i].chars().take(1).collect()));
        }
        if cases.len() >= n {
            break;
        }
    }
    println!("{:<34} {:>9} {:>9} {:>9}", "", "p10", "median", "p90");
    if let Some(p) = args.get("ngram") {
        #[cfg(feature = "ngram")]
        {
            let bytes = std::fs::read(p).map_err(|e| format!("{p}: {e}"))?;
            let mut lm = None;
            timed("ngram load", 3, |_| {
                lm = Some(threers_swipe::ngram::NgramModel::from_bytes(&bytes).expect("loads"))
            });
            bench_model("ngram", lm.as_ref().expect("loaded"), &cases);
        }
        #[cfg(not(feature = "ngram"))]
        return Err(format!("{p}: built without the `ngram` feature"));
    }
    if let Some(p) = args.get("neural") {
        #[cfg(feature = "rlx")]
        {
            let bytes = std::fs::read(p).map_err(|e| format!("{p}: {e}"))?;
            let mut lm = None;
            timed("neural load", 3, |_| {
                lm = Some(threers_swipe::neural::NeuralLm::from_safetensors(&bytes).expect("loads"))
            });
            bench_model("neural", lm.as_ref().expect("loaded"), &cases);
        }
        #[cfg(not(feature = "rlx"))]
        return Err(format!("{p}: built without the `rlx` feature"));
    }
    Ok(())
}

#[cfg(any(feature = "ngram", feature = "rlx"))]
fn bench_model(name: &str, m: &dyn WordModel, cases: &[(Vec<String>, String)]) {
    let ctx = |i: usize| -> Vec<&str> { cases[i].0.iter().map(String::as_str).collect() };
    // A first call each, as the page's first keystroke would be.
    timed(&format!("{name} next word (new context)"), cases.len(), |i| {
        std::hint::black_box(m.predict(&ctx(i), "", 3));
    });
    timed(&format!("{name} after one letter"), cases.len(), |i| {
        std::hint::black_box(m.predict(&ctx(i), &cases[i].1, 3));
    });
    let words = ["the", "to", "of", "and", "a", "in", "is", "it", "that", "for"];
    timed(&format!("{name} 10 words' log p"), cases.len(), |i| {
        std::hint::black_box(m.log_probs(&ctx(i), &words));
    });
}

/// A saved neural model again, as f16 (`--f16 1`) or f32.
#[cfg(feature = "rlx")]
fn cmd_convert(args: &Args) -> Result<(), String> {
    let lm = threers_swipe::neural::NeuralLm::load(args.need("neural")?)?;
    let out = args.need("out")?;
    if args.get("f16").is_some_and(|v| v != "0") {
        lm.save_f16(out)?;
    } else {
        lm.save(out)?;
    }
    let size = std::fs::metadata(out).map_err(|e| e.to_string())?.len();
    eprintln!("wrote {out}: {:.1} MB", size as f64 / 1e6);
    Ok(())
}

#[cfg(not(feature = "rlx"))]
fn cmd_convert(_: &Args) -> Result<(), String> {
    Err("built without the `rlx` feature".into())
}

fn cmd_eval(args: &Args) -> Result<(), String> {
    let text = words_of(&read_sentences(args.need("text")?)?);
    let n: usize = args.num("n", 20_000)?;
    let learn: f32 = args.num("learn", 0.0)?;
    // Each model, and with --learn each again with a user model learning
    // the text as it goes (a word scored, then learnt).
    let mut models: Vec<(String, Rc<dyn WordModel>, Option<Rc<Learning>>)> =
        models(args)?.into_iter().map(|(n, m)| (n, m, None)).collect();
    if learn > 0.0 {
        for (name, m, _) in models.clone() {
            let l = Rc::new(Learning { base: m, user: UserModel::new(), weight: learn });
            models.push((format!("{name} + you {learn}"), l.clone(), Some(l)));
        }
    }
    println!(
        "{:<12} {:>10} {:>7} {:>9} {:>9} {:>11}  {}",
        "model", "perplexity", "unknown", "next top1", "next top3", "1 letter t3", "words"
    );
    for (name, m, learner) in &models {
        let start = Instant::now();
        let k = m.context_len();
        let (mut nll, mut known, mut words) = (0.0f64, 0usize, 0usize);
        let (mut t1, mut t3, mut l3) = (0usize, 0usize, 0usize);
        'all: for s in &text {
            let mut ctx: Vec<&str> = vec![next_word::BOS];
            for w in s {
                if words == n {
                    break 'all;
                }
                words += 1;
                let tail = &ctx[ctx.len().saturating_sub(k)..];
                if let Some(lp) = m.log_probs(tail, &[w.as_str()])[0] {
                    nll -= lp as f64;
                    known += 1;
                }
                let next = m.predict(tail, "", 3);
                t1 += next.first().is_some_and(|p| &p.0 == w) as usize;
                t3 += next.iter().any(|p| &p.0 == w) as usize;
                let first: String = w.chars().take(1).collect();
                l3 += m.predict(tail, &first, 3).iter().any(|p| &p.0 == w) as usize;
                if let Some(l) = learner {
                    l.user.learn(tail, w);
                }
                ctx.push(w);
            }
        }
        let d = words.max(1) as f32 / 100.0;
        println!(
            "{name:<12} {:>10.1} {:>6.1}% {:>8.1}% {:>8.1}% {:>10.1}%  {words}  [{:.0}s]",
            (nll / known.max(1) as f64).exp(),
            (words - known) as f32 / d,
            t1 as f32 / d,
            t3 as f32 / d,
            l3 as f32 / d,
            start.elapsed().as_secs_f32()
        );
    }
    println!("perplexity over the words a model knows; top-k over all, an unknown word a miss");
    Ok(())
}

/// A model whose user model learns each word after it is scored.
struct Learning {
    base: Rc<dyn WordModel>,
    user: UserModel,
    weight: f32,
}

impl Learning {
    fn mixed(&self) -> WithUser<'_> {
        WithUser { base: self.base.as_ref(), user: &self.user, weight: self.weight }
    }
}

impl WordModel for Learning {
    fn context_len(&self) -> usize {
        self.mixed().context_len()
    }
    fn log_probs(&self, context: &[&str], words: &[&str]) -> Vec<Option<f32>> {
        self.mixed().log_probs(context, words)
    }
    fn predict(&self, context: &[&str], prefix: &str, n: usize) -> Vec<(String, f32)> {
        self.mixed().predict(context, prefix, n)
    }
    fn unknown_log_prob(&self, context: &[&str]) -> Option<f32> {
        self.mixed().unknown_log_prob(context)
    }
}

/// A recorded swipe with the words before it in its sentence.
struct InContext {
    keyboard: KeyboardGeometry,
    points: Vec<TouchPoint>,
    word: String,
    /// The sentence up to the swiped word.
    before: String,
}

/// FUTO's held-out swipes with `sentence` and `word_idx`, every
/// `stride`-th one, each line read as [`Recordings`] reads it.
fn in_context(path: &str, futo: &FutoKeyboard, n: usize) -> Result<Vec<InContext>, String> {
    #[derive(Deserialize)]
    struct L {
        sentence: Option<String>,
        word_idx: Option<usize>,
    }
    let lines = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?)
        .lines()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{path}: {e}"))?;
    let stride = (lines.len() / n.max(1)).max(1);
    let phone = KeyboardGeometry::from_config(&KeyboardConfig::iphone(), &KeyboardLayout::english(), 390.0);
    let mut out = vec![];
    for line in lines.iter().step_by(stride) {
        let Ok(L {
            sentence: Some(s),
            word_idx: Some(i),
        }) = serde_json::from_str::<L>(line)
        else {
            continue;
        };
        let recs = Recordings::parse(line, &phone, futo);
        let Some(r) = recs.items.into_iter().next() else {
            continue;
        };
        let before: Vec<&str> = s.split_whitespace().take(i).collect();
        out.push(InContext {
            keyboard: recs.keyboards[r.keyboard].clone(),
            points: r.points,
            word: r.word.to_lowercase(),
            before: before.join(" "),
        });
        if out.len() == n {
            break;
        }
    }
    Ok(out)
}

fn cmd_swipes(args: &Args) -> Result<(), String> {
    let model = Model::load(args.need("model")?)?;
    let lexicon = Lexicon::load(args.need("words")?)?;
    let mut futo = match args.get("futo-layout") {
        Some(p) => FutoKeyboard::from_layout_json(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?)?,
        None => FutoKeyboard::default(),
    };
    if args.get("futo-aspect").is_some() {
        futo.aspect = Some(args.num("futo-aspect", 4.0)?);
    }
    let lms = models(args)?;
    let start = Instant::now();
    let cases = in_context(args.need("held-out")?, &futo, args.num("n", 3000)?)?;
    let first = cases.first().ok_or("no swipes with a sentence")?;
    // Every beam word and its CTC score, once; ranking them is cheap.
    let decoder = Decoder::new(&first.keyboard, &lexicon).with_config(DecodeConfig {
        lm_weight: 0.0,
        ..DecodeConfig::default()
    });
    let search = |c: &InContext| {
        decoder
            .decode_on(&model, &c.keyboard, &c.points, usize::MAX)
            .unwrap_or_default()
            .into_iter()
            .map(|h| (h.word, h.ctc))
            .collect::<Vec<_>>()
    };
    #[cfg(feature = "parallel")]
    let hits: Vec<Vec<(String, f32)>> = {
        use rayon::prelude::*;
        cases.par_iter().map(search).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let hits: Vec<Vec<(String, f32)>> = cases.iter().map(search).collect();
    eprintln!("{} swipes searched ({:.0}s)", cases.len(), start.elapsed().as_secs_f32());
    let freq: HashMap<&str, f32> = lexicon.words.iter().map(|(w, f)| (w.as_str(), *f)).collect();
    let known = cases.iter().filter(|c| freq.contains_key(c.word.as_str())).count();
    // Each word's log p: the lexicon's, and each model's after the context.
    let mut scorers: Vec<(String, Vec<Vec<f32>>)> = vec![(
        "word list".into(),
        hits.iter()
            .map(|h| h.iter().map(|(w, _)| freq[w.as_str()]).collect())
            .collect(),
    )];
    for (name, lm) in &lms {
        let lp = cases
            .iter()
            .zip(&hits)
            .map(|(c, h)| {
                let (ctx, _) = next_word::split_context(&format!("{} ", c.before), lm.context_len());
                let ctx: Vec<&str> = ctx.iter().map(String::as_str).collect();
                let words: Vec<&str> = h.iter().map(|(w, _)| w.as_str()).collect();
                next_word::log_probs_or(lm.as_ref(), &ctx, &words, |w| freq[w])
            })
            .collect();
        scorers.push((name.clone(), lp));
    }
    let weights: Vec<f32> = args
        .get("lm-weights")
        .unwrap_or("0,0.25,0.5,0.75,1,1.5,2")
        .split(',')
        .map(|w| w.trim().parse().map_err(|_| format!("--lm-weights: `{w}`")))
        .collect::<Result<_, _>>()?;
    println!(
        "{} swipes in their sentences, {} of their words in the word list (the most top-1 can reach: {:.1}%)",
        cases.len(),
        known,
        known as f32 * 100.0 / cases.len() as f32
    );
    print!("{:<12}", "lm-weight");
    for w in &weights {
        print!(" {:>13}", format!("{w}"));
    }
    println!("\n{:<12}{}", "", "  top-1 top-3 ".repeat(weights.len()));
    for (name, lp) in &scorers {
        print!("{name:<12}");
        for &w in &weights {
            let (mut t1, mut t3) = (0, 0);
            for ((c, h), lp) in cases.iter().zip(&hits).zip(lp) {
                let mut ranked: Vec<(&str, f32)> = h
                    .iter()
                    .zip(lp)
                    .map(|((word, ctc), l)| (word.as_str(), ctc + w * l))
                    .collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                t1 += ranked.first().is_some_and(|r| r.0 == c.word) as usize;
                t3 += ranked.iter().take(3).any(|r| r.0 == c.word) as usize;
            }
            let d = cases.len() as f32 / 100.0;
            print!("  {:>5.1} {:>5.1}", t1 as f32 / d, t3 as f32 / d);
        }
        println!();
    }
    Ok(())
}

/// The bar as a sentence is typed, and recorded swipes the words before
/// them changed, through the decoder the keyboard uses.
fn cmd_demo(args: &Args) -> Result<(), String> {
    let lms = models(args)?;
    let (name, lm) = lms.last().expect("models() gives one at least");
    let bar = |before: &str| {
        let words = next_word::suggest(lm.as_ref(), before, 3);
        words.iter().map(|w| format!("[{w}]")).collect::<Vec<_>>().join(" ")
    };
    println!("── the bar, {name} ──");
    for text in args
        .get("type")
        .unwrap_or("I am going to the|Thank you so much for|The capital of France is|What time do you")
        .split('|')
    {
        let mut typed = String::new();
        for w in text.split_whitespace() {
            println!("  {:<36} {}", format!("{typed}▏"), bar(&typed));
            let first: String = w.chars().take(1).collect();
            println!("  {:<36} {}", format!("{typed}{first}▏"), bar(&format!("{typed}{first}")));
            typed.push_str(w);
            typed.push(' ');
        }
        println!("  {:<36} {}\n", format!("{typed}▏"), bar(&typed));
    }
    let Some(held) = args.get("held-out") else {
        return Ok(());
    };
    let model = Model::load(args.need("model")?)?;
    let lexicon = Lexicon::load(args.need("words")?)?;
    let futo = match args.get("futo-layout") {
        Some(p) => FutoKeyboard::from_layout_json(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?)?,
        None => FutoKeyboard::default(),
    };
    let cases = in_context(held, &futo, args.num("n", 3000)?)?;
    let first = cases.first().ok_or("no swipes with a sentence")?;
    let decoder = Decoder::new(&first.keyboard, &lexicon).with_config(DecodeConfig {
        lm_weight: args.num("lm-weight", 0.75)?,
        ..DecodeConfig::default()
    });
    let show: usize = args.num("show", 12)?;
    let (mut fixed, mut broke, mut shown) = (vec![], vec![], 0);
    for c in &cases {
        let alone = decoder.decode_on(&model, &c.keyboard, &c.points, 3).unwrap_or_default();
        let (ctx, _) = next_word::split_context(&format!("{} ", c.before), lm.context_len());
        let ctx: Vec<&str> = ctx.iter().map(String::as_str).collect();
        let after = decoder
            .decode_on_after(&model, &c.keyboard, &c.points, lm.as_ref(), &ctx, 3)
            .unwrap_or_default();
        let (a, b) = (
            alone.first().map_or("", |x| x.word.as_str()),
            after.first().map_or("", |x| x.word.as_str()),
        );
        if a == b {
            continue;
        }
        let line = format!(
            "  …{:>34} ⟨{}⟩   word list: {:<12} in context: {:<12}",
            c.before.chars().rev().take(34).collect::<String>().chars().rev().collect::<String>(),
            c.word,
            a,
            b
        );
        if b == c.word {
            fixed.push(line);
        } else if a == c.word {
            broke.push(line);
        }
        shown += 1;
    }
    println!("── {} recorded swipes (held out), the swiped word in ⟨⟩, lm-weight {} ──", cases.len(), decoder.config.lm_weight);
    println!("fixed by the words before ({}):", fixed.len());
    for l in fixed.iter().take(show) {
        println!("{l}");
    }
    println!("broken by them ({}):", broke.len());
    for l in broke.iter().take(show / 2) {
        println!("{l}");
    }
    println!("{} changed in all; the rest were wrong either way", shown);
    Ok(())
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first() else {
        eprintln!("usage: threers-next-word <ngram|neural|convert|eval|swipes|demo|bench> [--flag value …]");
        return ExitCode::FAILURE;
    };
    let result = Args::parse(&raw[1..]).and_then(|args| match cmd.as_str() {
        "ngram" => cmd_ngram(&args),
        "neural" => cmd_neural(&args),
        "eval" => cmd_eval(&args),
        "swipes" => cmd_swipes(&args),
        "demo" => cmd_demo(&args),
        "convert" => cmd_convert(&args),
        "bench" => cmd_bench(&args),
        other => Err(format!("no command `{other}`: ngram, neural, convert, eval, swipes, demo, bench")),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
