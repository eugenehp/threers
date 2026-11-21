//! Trains a small swipe model on made-up swipes of 200 common English
//! words on the iPhone and iPad keyboards, then scores it on fresh ones
//! and on a keyboard it never saw (the iPad Pro's, and AZERTY).
//!
//! ```text
//! cargo run -p threers-swipe --release --example swipe_train_synth [-- STEPS]
//! ```

use std::time::Instant;

use threers_swipe::synth::SynthConfig;
use threers_swipe::train::{self, TrainConfig, Trainer};
use threers_swipe::{synth, Decoder, KeyboardGeometry, Lexicon, ModelConfig};
use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout};

const WORDS: &str = "the of and to in is you that it he was for on are as with his they at be this have from \
or one had by word but not what all were we when your can said there use an each which she do how their if will \
up other about out many then them these so some her would make like him into time has look two more write go see \
number no way could people my than first water been call who oil its now find long down day did get come made may \
part over new sound take only little work know place year live me back give most very after thing our just name \
good sentence man think say great where help through much before line right too mean old any same tell boy follow \
came want show also around form three small set put end does another well large must big even such because turn \
here why ask went men read need land different home us move try kind hand picture again change off play spell air \
away animal house point page letter mother answer found study still learn should america world high every near add \
food between own below country plant last school father keep tree never start city earth eye light thought head \
under story saw left few while along might close something seem next hard open example begin life always those both";

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1500);
    let lexicon = Lexicon::parse(&WORDS.split_whitespace().collect::<Vec<_>>().join("\n"));
    let en = KeyboardLayout::english();
    let phone = KeyboardGeometry::from_config(&KeyboardConfig::iphone(), &en, 390.0);
    let tablet = KeyboardGeometry::from_config(&KeyboardConfig::ipad(), &en, 820.0);
    let unseen = KeyboardGeometry::from_config(&KeyboardConfig::ipad_pro(), &en, 1024.0);

    let mut trainer = Trainer::new(TrainConfig {
        model: ModelConfig {
            channels: 64,
            dilations: vec![1, 2, 4, 8, 1],
        },
        steps,
        batch: 32,
        lr: 3e-3,
        ..TrainConfig::default()
    });
    trainer.add_synthetic(phone.clone(), lexicon.word_list());
    trainer.add_synthetic(tablet, lexicon.word_list());
    println!(
        "{} weights, {steps} steps of {}",
        trainer.model.len(),
        trainer.config.batch
    );

    let start = Instant::now();
    let mut avg = 0.0;
    trainer.run(|step, s| {
        avg += s.loss;
        if step % 250 == 0 {
            println!(
                "step {step:5}  loss {:.3}  {:.1}s",
                avg / 250.0,
                start.elapsed().as_secs_f32()
            );
            avg = 0.0;
        }
    });

    let cfg = SynthConfig::default();
    let model = &trainer.model;
    println!(
        "iPhone, fresh swipes:        {}",
        train::evaluate_synthetic(model, &phone, &lexicon, &cfg, 400, 99)
    );
    println!(
        "iPad Pro (never trained on): {}",
        train::evaluate_synthetic(model, &unseen, &lexicon, &cfg, 400, 99)
    );
    // AZERTY moves a, q, z, w and m: the same words, other paths.
    let azerty =
        KeyboardGeometry::from_config(&KeyboardConfig::iphone(), &KeyboardLayout::french(), 390.0);
    println!(
        "AZERTY (never trained on):   {}",
        train::evaluate_synthetic(model, &azerty, &lexicon, &cfg, 400, 99)
    );

    let dec = Decoder::new(&phone, &lexicon);
    let mut rng = synth::Rng::new(5);
    for w in ["picture", "answer", "between"] {
        let s = synth::swipe(&phone, w, &cfg, &mut rng).unwrap();
        let got: Vec<String> = dec
            .decode(model, &s, 3)
            .into_iter()
            .map(|c| c.word)
            .collect();
        println!("{w:>10} → {got:?}");
    }
}
