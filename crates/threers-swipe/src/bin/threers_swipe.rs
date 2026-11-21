//! `threers-swipe`: train, check and try gesture typing models.
//!
//! ```text
//! threers-swipe train  --words en=words_en.txt [--words de=words_de.txt …]
//!                      [--recorded swipes.jsonl] [--steps 20000] [--batch 64]
//!                      [--size small|tiny] [--lr 2e-3] [--seed 1] [--resume M]
//!                      [--held-out dev.jsonl] [--eval-every 1000] [--out swipe.safetensors]
//! threers-swipe eval   --model M --words en=… [--recorded swipes.jsonl] [--n 1000]
//! threers-swipe decode --model M --words en=… --word hello [--n 5]
//! threers-swipe check  --recorded swipes.jsonl [--futo-layout qwerty.json] [--futo-aspect A]
//! threers-swipe lexicon --from train.jsonl --out words_en.txt [--top 100000]
//! threers-swipe synth  --words en=… --n 1000 --out synthetic.jsonl
//! ```
//!
//! Word files: one word a line, optionally with a count. Languages are the
//! built-in layouts' tags: en es de fr ru uk el. Each trains on a phone's,
//! a tablet's and a large tablet's keyboard.

use std::collections::BTreeMap;
use std::process::ExitCode;
use std::time::Instant;

use threers_swipe::data::{self, FutoKeyboard, Recording, Recordings};
use threers_swipe::synth::{self, Rng, SynthConfig};
use threers_swipe::train::{self, Accuracy, TrainConfig, Trainer};
use threers_swipe::{Decoder, KeyboardGeometry, Lexicon, Model, ModelConfig};
use threers_ui::virtual_keyboard::{Device, KeyboardConfig, KeyboardLayout};

struct Args {
    flags: BTreeMap<String, Vec<String>>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Self, String> {
        let mut flags: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut it = raw.iter();
        while let Some(a) = it.next() {
            let name = a
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected `{a}`"))?;
            let value = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
            flags
                .entry(name.to_owned())
                .or_default()
                .push(value.clone());
        }
        Ok(Self { flags })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags
            .get(name)
            .and_then(|v| v.last())
            .map(String::as_str)
    }

    fn all(&self, name: &str) -> &[String] {
        self.flags.get(name).map_or(&[], Vec::as_slice)
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

/// `--words lang=path`, each language's keyboards and lexicon.
fn languages(args: &Args) -> Result<Vec<(String, Lexicon, Vec<KeyboardGeometry>)>, String> {
    let mut out = vec![];
    for spec in args.all("words") {
        let (lang, path) = spec
            .split_once('=')
            .ok_or_else(|| format!("--words {spec}: write it LANG=PATH"))?;
        let layout = KeyboardLayout::all()
            .into_iter()
            .find(|l| l.language == lang)
            .ok_or_else(|| format!("no built-in layout for `{lang}` (en es de fr ru uk el)"))?;
        let lexicon = Lexicon::load(path)?;
        let keyboards = [
            (Device::Phone, 390.0),
            (Device::Tablet, 820.0),
            (Device::LargeTablet, 1024.0),
        ]
        .iter()
        .map(|&(d, w)| KeyboardGeometry::from_config(&KeyboardConfig::for_device(d), &layout, w))
        .collect();
        out.push((lang.to_owned(), lexicon, keyboards));
    }
    if out.is_empty() {
        return Err("give at least one --words LANG=PATH".into());
    }
    Ok(out)
}

fn english_phone() -> KeyboardGeometry {
    KeyboardGeometry::from_config(&KeyboardConfig::iphone(), &KeyboardLayout::english(), 390.0)
}

fn futo(args: &Args) -> Result<FutoKeyboard, String> {
    let mut f = match args.get("futo-layout") {
        Some(p) => FutoKeyboard::from_layout_json(
            &std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?,
        )?,
        None => FutoKeyboard::default(),
    };
    if args.get("futo-aspect").is_some() {
        f.aspect = Some(args.num("futo-aspect", 4.0)?);
    }
    Ok(f)
}

fn load_recorded(args: &Args, flag: &str) -> Result<Option<Recordings>, String> {
    let Some(path) = args.get(flag) else {
        return Ok(None);
    };
    let start = Instant::now();
    let recs = Recordings::load(path, &english_phone(), &futo(args)?)?;
    eprintln!(
        "{path}: {} swipes on {} keyboard shapes, {} lines left out ({:.0}s)",
        recs.items.len(),
        recs.keyboards.len(),
        recs.skipped,
        start.elapsed().as_secs_f32()
    );
    for e in &recs.errors {
        eprintln!("  {e}");
    }
    Ok(Some(recs))
}

fn recorded(args: &Args) -> Result<Option<Recordings>, String> {
    load_recorded(args, "recorded")
}

fn subsample(recs: &Recordings, n: usize) -> Recordings {
    let mut out = recs.clone();
    if out.items.len() > n {
        let step = out.items.len() as f32 / n as f32;
        out.items = (0..n)
            .map(|i| recs.items[(i as f32 * step) as usize].clone())
            .collect();
    }
    out
}

fn report(
    langs: &[(String, Lexicon, Vec<KeyboardGeometry>)],
    model: &Model,
    held: Option<&Recordings>,
    n: usize,
) {
    for (lang, lex, kbs) in langs {
        let acc =
            train::evaluate_synthetic(model, &kbs[0], lex, &SynthConfig::default(), n, 0xe7a1);
        eprintln!("  {lang} synthetic: {acc}");
    }
    if let Some(h) = held {
        // Recorded swipes are English (FUTO); scored against the English words.
        if let Some((_, lex, _)) = langs.iter().find(|l| l.0 == "en") {
            let acc: Accuracy = train::evaluate_recorded(model, &subsample(h, n), lex);
            eprintln!("  recorded (held out): {acc}");
        }
    }
}

fn cmd_train(args: &Args) -> Result<(), String> {
    let langs = languages(args)?;
    let model_cfg = match args.get("size").unwrap_or("small") {
        "small" => ModelConfig::small(),
        "tiny" => ModelConfig::tiny(),
        s => return Err(format!("--size {s}: small or tiny")),
    };
    let cfg = TrainConfig {
        model: model_cfg,
        steps: args.num("steps", 20_000)?,
        batch: args.num("batch", 64)?,
        lr: args.num("lr", 2e-3)?,
        seed: args.num("seed", 1)?,
        recorded_share: args.num("recorded-share", 0.7)?,
        ..TrainConfig::default()
    };
    let out = args.get("out").unwrap_or("swipe.safetensors").to_owned();
    let eval_every: usize = args.num("eval-every", 1000)?;
    let mut trainer = match args.get("resume") {
        Some(p) => {
            let m = Model::load(p)?;
            let step = args.num("resume-step", 0)?;
            Trainer::resume(
                TrainConfig {
                    model: m.config.clone(),
                    ..cfg
                },
                m,
                step,
            )
        }
        None => Trainer::new(cfg),
    };
    for (_, lex, kbs) in &langs {
        for kb in kbs {
            trainer.add_synthetic(kb.clone(), lex.word_list());
        }
    }
    let mut held = load_recorded(args, "held-out")?;
    if let Some(recs) = recorded(args)? {
        let tr = if held.is_none() {
            let (tr, h) = train::split_by_session(recs, 0.05);
            held = Some(h);
            tr
        } else {
            recs
        };
        eprintln!(
            "recorded: {} to train on, {} held out",
            tr.items.len(),
            held.as_ref().map_or(0, |h| h.items.len())
        );
        trainer.set_recorded(tr);
    }
    eprintln!(
        "model: {} weights; {} steps of {}; {} keyboards",
        trainer.model.len(),
        trainer.config.steps,
        trainer.config.batch,
        langs.iter().map(|l| l.2.len()).sum::<usize>()
    );
    let start = Instant::now();
    let (mut avg, mut count) = (0.0f32, 0usize);
    while trainer.step < trainer.config.steps {
        let s = trainer.train_step();
        let step = trainer.step;
        avg += s.loss;
        count += 1;
        if step % 100 == 0 {
            eprintln!(
                "step {step:6}  loss {:.4}  lr {:.2e}  |g| {:.3}  skipped {}  {:.0}s",
                avg / count as f32,
                s.lr,
                s.grad_norm,
                s.skipped,
                start.elapsed().as_secs_f32()
            );
            (avg, count) = (0.0, 0);
        }
        if eval_every > 0 && step % eval_every == 0 && step < trainer.config.steps {
            trainer.model.save(&out)?;
            eprintln!("step {step}: checkpoint {out}");
            report(&langs, &trainer.model, held.as_ref(), 200);
        }
    }
    trainer.model.save(&out)?;
    eprintln!("wrote {out}");
    report(&langs, &trainer.model, held.as_ref(), 500);
    Ok(())
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first() else {
        eprintln!("usage: threers-swipe <train|eval|decode|check|synth|lexicon> [--flag value …]");
        return ExitCode::FAILURE;
    };
    let result = Args::parse(&raw[1..]).and_then(|args| match cmd.as_str() {
        "train" => cmd_train(&args),
        "eval" => cmd_eval(&args),
        "decode" => cmd_decode(&args),
        "check" => cmd_check(&args),
        "synth" => cmd_synth(&args),
        "lexicon" => cmd_lexicon(&args),
        other => Err(format!(
            "no command `{other}`: train, eval, decode, check, synth, lexicon"
        )),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_eval(args: &Args) -> Result<(), String> {
    let model = Model::load(args.need("model")?)?;
    let langs = languages(args)?;
    let n = args.num("n", 1000)?;
    let held = match load_recorded(args, "held-out")? {
        Some(h) => Some(h),
        None => recorded(args)?.map(|r| train::split_by_session(r, 0.05).1),
    };
    report(&langs, &model, held.as_ref(), n);
    Ok(())
}

fn cmd_decode(args: &Args) -> Result<(), String> {
    let model = Model::load(args.need("model")?)?;
    let langs = languages(args)?;
    let word = args.need("word")?;
    let n = args.num("n", 5)?;
    let mut rng = Rng::new(args.num("seed", 7)?);
    for (lang, lex, kbs) in &langs {
        let dec = Decoder::new(&kbs[0], lex);
        let Some(swipe) = synth::swipe(&kbs[0], word, &SynthConfig::default(), &mut rng) else {
            eprintln!("{lang}: `{word}` has letters this keyboard lacks");
            continue;
        };
        println!(
            "{lang}: a made-up swipe of `{word}`, {} points:",
            swipe.len()
        );
        for c in dec.decode(&model, &swipe, n) {
            println!("  {:<20} score {:8.3}  ctc {:8.3}", c.word, c.score, c.ctc);
        }
    }
    Ok(())
}

fn cmd_check(args: &Args) -> Result<(), String> {
    let recs = recorded(args)?.ok_or("--recorded is needed")?;
    let (s, e) = data::alignment(&recs);
    println!(
        "{} swipes; mean distance to the first letter {s:.2} keys, from the last {e:.2} keys",
        recs.items.len()
    );
    println!("about 0.5 or less: the keyboard is where it is assumed to be; a key or more: check --futo-layout / --futo-aspect");
    Ok(())
}

fn cmd_synth(args: &Args) -> Result<(), String> {
    let langs = languages(args)?;
    let n: usize = args.num("n", 1000)?;
    let path = args.need("out")?;
    let mut rng = Rng::new(args.num("seed", 1)?);
    let mut lines = String::new();
    for (lang, lex, kbs) in &langs {
        let words = lex.word_list();
        for i in 0..n {
            let kb = &kbs[i % kbs.len()];
            if let Some((w, points)) =
                synth::random_swipe(kb, &words, &SynthConfig::default(), &mut rng)
            {
                let r = Recording {
                    word: w.to_owned(),
                    points,
                    keyboard: 0,
                    session: Some(format!("synth-{lang}")),
                };
                lines.push_str(&r.to_json(kb));
                lines.push('\n');
            }
        }
    }
    std::fs::write(path, lines).map_err(|e| format!("{path}: {e}"))?;
    eprintln!("wrote {path}");
    Ok(())
}

fn cmd_lexicon(args: &Args) -> Result<(), String> {
    let from = args.need("from")?;
    let out = args.need("out")?;
    let top: usize = args.num("top", 100_000)?;
    let mut counts = data::sentence_counts(from)?;
    let total = counts.len();
    counts.truncate(top);
    let text: String = counts.iter().map(|(w, c)| format!("{w} {c}\n")).collect();
    std::fs::write(out, text).map_err(|e| format!("{out}: {e}"))?;
    eprintln!("wrote {out}: {} of {total} words", counts.len());
    Ok(())
}
