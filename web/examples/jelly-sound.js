// Ported from Axolotl Jelly (https://github.com/saulfloresjr/axolotl-jelly),
// unchanged apart from the module export at the end.
//
// MIT License
//
// Copyright (c) 2026 Dream Meridian
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of the experiments in this folder (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies of the
// Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ─────────────────────────────────────────────────────────────────────────────
// sound.js — every noise is synthesized with Web Audio; there are no files.
//
// The set was chosen on the Squish Board (tools/sound-lab): water balloon
// grab, plunger release, gummy-groan pull, pop-and-bwoing tear, fish-flop twitching, boing
// startle, jelly-drum and plip drops, jello-plop slaps, jelly-jiggle twisting.
//
// One-shots are short graphs of oscillators and filtered noise built on demand
// and left to the garbage collector. Voices that run continuously are steered
// every frame:
//   groan    a low, unevenly drifting stretch with a soft wet hiss, from the
//            moment you hold a leg until its glue is about to give
//   grow     a pair of detuned sines that climb as a bud fills in, with
//            bubbles popping faster near the end
//   wibble   a jelly-on-a-plate wobble that swells with the body's bounce
//   jiggle   a low jiggling wub while you twist that wibbles out when you stop
// Kneading the body scatters short wet grains, faster the harder you work it.
// A sour coating adds a crunch of sugar clicks to grabs and kneading.
// ¼ speed goes in with a whoom and out with a boing, and everything in
// between plays lower and slower.
// ─────────────────────────────────────────────────────────────────────────────
const JellySound = (() => {
  let ctx = null, master = null, noise = null, enabled = true, rate = 1;
  let creak = null, grow = null, wub = null, jiggle = null, grainDebt = 0, grit = 0, twist = { wound: 0, nextClick: 0, lastTurn: 0 };
  const lastAt = {};

  try { enabled = localStorage.getItem('axolotl-sound') !== 'off'; } catch (_) {}

  function unlock() {
    if (!enabled) return;
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      ctx = new AC();
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -18; comp.ratio.value = 4;
      master = ctx.createGain(); master.gain.value = 0.7;
      master.connect(comp).connect(ctx.destination);
      noise = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
      const d = noise.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    }
    if (ctx.state === 'suspended') ctx.resume();
  }
  const ready = () => enabled && ctx && ctx.state === 'running';
  // don't let a burst of events stack into a roar
  const throttle = (key, ms) => { const t = performance.now(); if (lastAt[key] && t - lastAt[key] < ms) return false; lastAt[key] = t; return true; };
  const jitter = (x, amt = 0.12) => x * (1 + (Math.random() * 2 - 1) * amt);

  // ── building blocks ──
  function env(gainNode, t, peak, attack, decay) {
    const g = gainNode.gain;
    g.setValueAtTime(0.0001, t);
    g.exponentialRampToValueAtTime(Math.max(peak, 0.0002), t + attack);
    g.exponentialRampToValueAtTime(0.0001, t + attack + decay);
  }
  // vib: a pitch wobble that dies away (the boing in a boing);
  // am: a loudness wobble (the wub in jello); lp: a lowpass on the way out
  function tone({ type = 'sine', f0, f1 = f0, dur, peak, attack = 0.005, delay = 0, curve = 'exp', vib = 0, vibRate = 12, am, lp }) {
    const t = ctx.currentTime + delay / rate, D = dur / rate;
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = type;
    o.frequency.setValueAtTime(f0 * rate, t);
    if (curve === 'exp') o.frequency.exponentialRampToValueAtTime(Math.max(20, f1 * rate), t + D);
    else o.frequency.linearRampToValueAtTime(f1 * rate, t + D);
    const stop = t + D + attack / rate + 0.05;
    if (vib) {
      const l = ctx.createOscillator(), lg = ctx.createGain();
      l.frequency.value = vibRate * rate;
      lg.gain.setValueAtTime(vib * rate, t); lg.gain.exponentialRampToValueAtTime(0.5, t + D);
      l.connect(lg).connect(o.frequency); l.start(t); l.stop(stop);
    }
    env(g, t, peak, attack / rate, D);
    let out = o;
    if (lp) { const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = lp * rate; f.Q.value = 2; out.connect(f); out = f; }
    out.connect(g); out = g;
    if (am) {
      const a = ctx.createGain(), l = ctx.createOscillator(), lg = ctx.createGain();
      a.gain.value = 1 - am.depth / 2; l.frequency.setValueAtTime(am.rate * rate, t);
      if (am.rate1) l.frequency.linearRampToValueAtTime(am.rate1 * rate, t + D);
      lg.gain.value = am.depth / 2; l.connect(lg).connect(a.gain); l.start(t); l.stop(stop);
      out.connect(a); out = a;
    }
    out.connect(master);
    o.start(t); o.stop(stop);
  }
  function hiss({ filter = 'bandpass', f0, f1 = f0, q = 1, dur, peak, attack = 0.004, delay = 0 }) {
    const t = ctx.currentTime + delay / rate, D = dur / rate;
    const s = ctx.createBufferSource(), bq = ctx.createBiquadFilter(), g = ctx.createGain();
    s.buffer = noise; s.playbackRate.value = rate;
    bq.type = filter; bq.Q.value = q;
    bq.frequency.setValueAtTime(f0 * rate, t);
    bq.frequency.exponentialRampToValueAtTime(Math.max(30, f1 * rate), t + D);
    env(g, t, peak, attack / rate, D);
    s.connect(bq).connect(g).connect(master);
    s.start(t, Math.random() * 1.5); s.stop(t + D + 0.1);
  }

  // sugar crystals crunching: a scatter of tiny bright clicks
  function crunch(n, spread, peak) {
    for (let k = 0; k < n; k++) hiss({ filter: 'highpass', f0: jitter(3800, 0.3), f1: 2600, q: 0.9, dur: 0.008 + Math.random() * 0.012, peak: peak * (0.5 + Math.random() * 0.5), attack: 0.001, delay: Math.random() * spread });
  }

  // a cartoon spring
  const boing = (f, len, peak, delay = 0) => tone({ type: 'triangle', f0: f * 1.25, f1: f, dur: len, peak, vib: f * 0.35, vibRate: 17, delay });
  // a cork leaving a bottle
  function corkPop(peak = 1) {
    hiss({ filter: 'highpass', f0: 3200, f1: 3200, dur: 0.012, peak: 0.55 * peak, attack: 0.0005 });
    tone({ f0: 380, f1: 1500, dur: 0.05, peak: 0.3 * peak });
    hiss({ f0: 1200, f1: 3200, q: 1, dur: 0.12, peak: 0.07 * peak, delay: 0.02 });
  }

  // ── one-shots ──
  const sfx = {
    // grabbing jelly: a water balloon, a slosh that keeps wobbling
    squish(strength = 1) {
      if (!ready() || !throttle('squish', 60)) return;
      hiss({ filter: 'lowpass', f0: 900, f1: 200, q: 2, dur: 0.18, peak: 0.35 * strength });
      tone({ f0: jitter(170, 0.08), f1: 118, dur: 0.5, peak: 0.3 * strength, vib: 26, vibRate: 13 });
      if (grit) crunch(Math.round(10 * grit), 0.14, 0.12 * grit * strength);
    },
    // a leg giving way: a cork pop, then a big spring
    tear() {
      if (!ready()) return;
      corkPop();
      boing(150, 0.9, 0.4, 0.05);
    },
    // a torn leg flopping on the table like a fish
    wriggle(amount) {
      if (!ready() || amount < 0.08) return;
      hiss({ filter: 'lowpass', f0: 1800, f1: 400, q: 1, dur: 0.05, peak: 0.35 * amount, attack: 0.001 });
      tone({ f0: jitter(220, 0.1), f1: 150, dur: 0.05, peak: 0.15 * amount });
    },
    // a goo drop landing
    plip(radius, speed) {
      if (!ready() || !throttle('plip', 28)) return;
      const f = jitter(1700 - radius * 26000, 0.08), v = Math.min(1, speed / 4);
      tone({ f0: f, f1: f * 1.8, dur: 0.05, peak: 0.09 * v, attack: 0.002 });
    },
    // the body slapping the table: lands, then goes wub-wub-wub
    thud(speed) {
      if (!ready() || speed < 1.2 || !throttle('thud', 160)) return;
      const v = Math.min(1, (speed - 1.2) / 3.5);
      hiss({ filter: 'lowpass', f0: 900, f1: 200, dur: 0.1, peak: 0.3 * v });
      tone({ f0: jitter(120, 0.08), f1: 75, dur: 0.75, peak: 0.42 * v, am: { rate: 9, rate1: 5, depth: 0.95 } });
    },
    // Startle: one big boing
    startle() {
      if (!ready()) return;
      boing(130, 0.9, 0.42);
    },
    // a bud reaching full size: the hum ends in a soft boing
    regrown() {
      if (!ready()) return;
      boing(220, 0.7, 0.3);
    },
    // letting go: a plunger, a long suck then a pop
    release(strength = 1) {
      if (!ready() || !throttle('release', 80)) return;
      hiss({ f0: 380, f1: 2600, q: 5, dur: 0.2, peak: 0.26 * strength });
      tone({ f0: 180, f1: 900, dur: 0.16, peak: 0.14 * strength });
      tone({ f0: 900, f1: 560, dur: 0.06, peak: 0.22 * strength, delay: 0.17 });
    },
    // a drop landing on the axolotl: it wobbles like a drum skin
    plop(radius, speed) {
      if (!ready() || !throttle('plop', 45)) return;
      const f = jitter(340 - radius * 2000, 0.1), v = Math.min(1, speed / 3);
      tone({ f0: f, f1: f * 0.8, dur: 0.35, peak: 0.2 * v, vib: 40, vibRate: 15 });
    },
    // two drops bumping
    tick(speed) {
      if (!ready() || !throttle('tick', 45)) return;
      tone({ type: 'triangle', f0: jitter(1900, 0.2), f1: 1500, dur: 0.03, peak: 0.04 * Math.min(1, speed / 2), attack: 0.001 });
    },
    // a thrown drop leaving your hand
    fwip(speed) {
      if (!ready() || speed < 2 || !throttle('fwip', 120)) return;
      hiss({ f0: 700, f1: 3400, q: 2, dur: 0.12, peak: 0.12 * Math.min(1, speed / 6) });
    },
    // a bud stretching like a balloon as it swells
    stretchSqueak() {
      if (!ready() || !throttle('stretch', 300)) return;
      const f = jitter(700, 0.25);
      tone({ type: 'sawtooth', f0: f, f1: f * 1.35, dur: 0.14, peak: 0.03, attack: 0.01, curve: 'lin' });
    },
    // ¼ speed on: a swelling low boom
    slowIn() {
      if (!ready()) return;
      tone({ f0: 90, f1: 40, dur: 1.4, peak: 0.45, attack: 0.4 });
    },
    // back to full speed: snap back with a boing
    slowOut() {
      if (!ready()) return;
      boing(260, 0.4, 0.22);
    },
    // Reset: everything slurped back into place
    slurp() {
      if (!ready()) return;
      hiss({ f0: 260, f1: 2200, q: 4, dur: 0.35, peak: 0.25 });
      tone({ f0: 120, f1: 520, dur: 0.32, peak: 0.18 });
    },
  };

  // ── continuous voices ──
  // Gummy groan: a soft wet hiss under a low tone whose pitch drifts unevenly,
  // so it sounds like rubbery gel stretching rather than a machine. Both open
  // up and get louder as the pull grows.
  function makeGroan() {
    const n = ctx.createBufferSource(), nlp = ctx.createBiquadFilter(), gn = ctx.createGain();
    const o = ctx.createOscillator(), olp = ctx.createBiquadFilter(), go = ctx.createGain(), g = ctx.createGain();
    n.buffer = noise; n.loop = true; nlp.type = 'lowpass'; nlp.Q.value = 4;
    o.type = 'triangle'; olp.type = 'lowpass'; olp.Q.value = 1;
    n.connect(nlp).connect(gn).connect(g); o.connect(olp).connect(go).connect(g); g.connect(master);
    g.gain.value = 0; n.start(0, Math.random()); o.start();
    return { nlp, gn, o, olp, go, g, nextDrift: 0 };
  }
  // the regrowth hum: two rising voices with a tremolo
  function makeGrow() {
    const a = ctx.createOscillator(), b = ctx.createOscillator(), g = ctx.createGain(), trem = ctx.createOscillator(), tg = ctx.createGain(), am = ctx.createGain();
    a.type = 'sine'; b.type = 'triangle';
    trem.frequency.value = 7; tg.gain.value = 0.35; am.gain.value = 0.65;
    trem.connect(tg).connect(am.gain);
    a.connect(am); b.connect(am); am.connect(g).connect(master);
    g.gain.value = 0;
    a.start(); b.start(); trem.start();
    return { a, b, g, nextBubble: 0 };
  }
  // Wibble: jello on a plate. A sine with a fast pitch wobble, loudness and
  // pitch following the body's own bounce (signed vertical velocity of the
  // back relative to the body)
  function makeWibble() {
    const o = ctx.createOscillator(), g = ctx.createGain(), l = ctx.createOscillator(), lg = ctx.createGain();
    o.type = 'sine'; l.frequency.value = 7; lg.gain.value = 0;
    l.connect(lg).connect(o.frequency);
    o.connect(g).connect(master);
    g.gain.value = 0; o.start(); l.start();
    return { o, g, lg };
  }
  function setWobble(signal) {
    if (!ready()) return;
    const a = Math.min(1, Math.abs(signal) / 0.6);
    if (!wub) { if (a < 0.05) return; wub = makeWibble(); }
    const t = ctx.currentTime;
    wub.g.gain.setTargetAtTime(a > 0.06 ? 0.32 * a * a : 0, t, 0.03);
    wub.o.frequency.setTargetAtTime((150 + 30 * Math.max(-1, Math.min(1, signal / 0.6))) * rate, t, 0.03);
    wub.lg.gain.setTargetAtTime(40 * a * rate, t, 0.05);
  }
  // Kneading: wet grains scattered at a rate set by how fast the jelly is
  // being worked, the occasional deep gloop among them.
  function squelch(speed, dt) {
    if (!ready()) return;
    const level = Math.min(1, speed / 3);
    grainDebt += level * 38 * dt;
    while (grainDebt >= 1) {
      grainDebt -= 1;
      const f = jitter(1100, 0.45);
      hiss({ f0: f, f1: f * 0.45, q: 6 + Math.random() * 6, dur: 0.03 + Math.random() * 0.05, peak: 0.07 + 0.1 * level, delay: Math.random() * 0.03 });
      if (Math.random() < 0.18) tone({ f0: jitter(160, 0.25), f1: 70, dur: 0.12, peak: 0.12 * level, delay: Math.random() * 0.03 });
      if (grit && Math.random() < 0.7 * grit) crunch(2, 0.03, 0.09 * grit);
    }
    if (level === 0) grainDebt = 0;
  }
  // Twisting is a jelly jiggle: a low wub that jiggles faster the faster you
  // twist; pause after a good twist and it wibbles out like jello on a plate.
  function makeJiggle() {
    const o = ctx.createOscillator(), am = ctx.createGain(), lfo = ctx.createOscillator(), lg = ctx.createGain(), vib = ctx.createOscillator(), vg = ctx.createGain(), g = ctx.createGain();
    o.type = 'sine'; am.gain.value = 1 - 0.9 / 2; lg.gain.value = 0.9 / 2; lfo.frequency.value = 8;
    vib.frequency.value = 4; vg.gain.value = 6;
    lfo.connect(lg).connect(am.gain); vib.connect(vg).connect(o.frequency);
    o.connect(am).connect(g).connect(master);
    g.gain.value = 0; o.start(); lfo.start(); vib.start();
    return { o, lfo, g };
  }
  function setSqueak(spin) {
    if (!ready()) return;
    const S = Math.min(1, Math.abs(spin) / 4), t = ctx.currentTime;
    if (!jiggle) { if (S < 0.05) return; jiggle = makeJiggle(); }
    jiggle.g.gain.setTargetAtTime(S > 0.05 ? 0.2 + 0.2 * S : 0, t, 0.04);
    jiggle.o.frequency.setTargetAtTime((85 + 45 * S) * rate, t, 0.05);
    jiggle.lfo.frequency.setTargetAtTime((8 + 10 * S) * rate, t, 0.05);
    if (S > 0.05) { twist.wound = Math.min(1, twist.wound + S * 0.02); twist.lastTurn = t; }
    else if (twist.wound > 0 && t - twist.lastTurn > 0.25) {   // wheel twists come in bursts: wait for a real pause
      if (twist.wound > 0.1) tone({ f0: 160, f1: 140, dur: 1.0, peak: 0.26, attack: 0.04, vib: 40, vibRate: 7, am: { rate: 7, depth: 0.7 } });
      twist.wound = 0;
    }
  }

  // level 0 = not holding a leg; 0.1 = holding it; 1 = about to tear
  function setCreak(level) {
    if (!ready()) return;
    if (!creak) { if (level <= 0) return; creak = makeGroan(); }
    const t = ctx.currentTime, L = Math.min(1, Math.max(0, level));
    creak.g.gain.setTargetAtTime(L > 0.05 ? 1 : 0, t, 0.05);
    creak.nlp.frequency.setTargetAtTime((220 + 600 * L) * rate, t, 0.05);
    creak.gn.gain.setTargetAtTime(0.08 + 0.16 * L, t, 0.05);
    creak.olp.frequency.setTargetAtTime((260 + 500 * L) * rate, t, 0.05);
    creak.go.gain.setTargetAtTime(0.12 + 0.12 * L, t, 0.05);
    if (t > creak.nextDrift) {   // a slow, uneven drift in pitch
      creak.o.frequency.setTargetAtTime((65 + 55 * L) * (0.88 + Math.random() * 0.24) * rate, t, 0.15);
      creak.nextDrift = t + (0.15 + Math.random() * 0.2) / rate;
    }
  }
  // progress 0..1 of the furthest-along bud, or -1 when nothing is growing
  function setGrow(progress) {
    if (!ready()) return;
    if (!grow) { if (progress < 0) return; grow = makeGrow(); }
    const t = ctx.currentTime;
    if (progress < 0) { grow.g.gain.setTargetAtTime(0, t, 0.08); return; }
    const f = 196 * Math.pow(2, progress * 1.6) * rate;
    grow.a.frequency.setTargetAtTime(f, t, 0.1);
    grow.b.frequency.setTargetAtTime(f * 1.503, t, 0.1);    // a slightly sour fifth: it's alive
    grow.g.gain.setTargetAtTime(0.035 + 0.035 * progress, t, 0.15);
    if (t > grow.nextBubble) {
      tone({ f0: jitter(500 + 1100 * progress, 0.25), f1: jitter(900 + 1600 * progress, 0.2), dur: 0.05, peak: 0.05, attack: 0.002 });
      grow.nextBubble = t + (0.25 - 0.18 * progress) * (0.6 + Math.random() * 0.8) / rate;
      if (Math.random() < 0.12) sfx.stretchSqueak();
    }
  }

  function setEnabled(on) {
    enabled = on;
    try { localStorage.setItem('axolotl-sound', on ? 'on' : 'off'); } catch (_) {}
    if (on) unlock();
    else if (ctx) for (const v of [grow, creak, wub, jiggle]) if (v) v.g.gain.value = 0;
    if (ctx) master.gain.setTargetAtTime(on ? 0.7 : 0, ctx.currentTime, 0.02);
  }

  return {
    unlock, setCreak, setGrow, setWobble, setSqueak, squelch, setEnabled, ...sfx,
    get enabled() { return enabled; },
    setGrit(g) { grit = g; },
    get state() { return ctx ? ctx.state : 'not started'; },
    // ¼ speed in with a whoom (already pitched down), out with a boing
    setSlow(slow) {
      if (slow === rate < 1) return;
      rate = slow ? 0.6 : 1;
      if (slow) sfx.slowIn(); else sfx.slowOut();
    },
  };
})();

export { JellySound };
