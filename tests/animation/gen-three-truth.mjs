// Writes three-r165-truth.json: three.js r165 (web/node_modules/three) run on
// the scenarios in scenarios.mjs and on unit cases for the pieces under the
// mixer — track-name parsing, every interpolant, track and clip edits,
// AnimationUtils. src/animation/three/tests.rs replays the same inputs on the
// Rust port and compares.
//
//   node tests/animation/gen-three-truth.mjs
//
// Non-finite numbers are written as {"$num": "Infinity" | "-Infinity" | "NaN"}
// (JSON has none); uuids are fixed so the output is reproducible.

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { SCENARIOS, runScenario, gltfInterpolants } from './scenarios.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, '../..');
const LoopPingPongC = 2202;
const THREE = await import(pathToFileURL(join(repo, 'web/node_modules/three/build/three.module.js')).href);
if (THREE.REVISION !== '165') throw new Error(`three r${THREE.REVISION}, expected r165`);

// Reproducible uuids.
let uuidN = 0;
THREE.MathUtils.generateUUID = () => `00000000-0000-4000-8000-${String(++uuidN).padStart(12, '0')}`;

const enc = (v) => JSON.parse(JSON.stringify(v, (_k, x) => (typeof x === 'number' && !Number.isFinite(x)
    ? { $num: String(x) } : x)));

// ---- parseTrackName ----
const NAMES = [
    'Box.position', '.position', 'position.x', '.morphTargetInfluences[smile]', 'mesh.morphTargetInfluences[0]',
    'Box.material.color', 'Box.material[1].opacity', 'mesh.materials[1].opacity', '.bones[hip].position',
    'Armature.bones[Spine.001].quaternion', 'dir/sub:Arm.L.quaternion', 'a.b.c.d', 'node.map.offset',
    'Arm.L.R.scale[2]', 'node name.visible', 'ünïcödé.position', 'x.y[a[b]].z', 'x.y[a].z[b]', '.a[1]',
    'uuid-1234-5678.scale', 'Cube.userData[speed].x', 'a/b.c', 'a:b:c.d', '.material.uniforms[time].value',
    'n.bones[0].position', 'Head_end.position', 'node.prop[]', 'a..b', 'trailing.', '[x].y', 'a/.b', '',
];
const parse = NAMES.map((n) => {
    try {
        const r = THREE.PropertyBinding.parseTrackName(n);
        return { name: n, nodeName: r.nodeName ?? null, objectName: r.objectName ?? null, objectIndex: r.objectIndex ?? null,
            propertyName: r.propertyName, propertyIndex: r.propertyIndex ?? null };
    } catch (e) {
        return { name: n, error: true };
    }
});
const sanitize = ['a b', 'x.y[z]:w/v', '  tab\there', 'ok_name-1'].map((n) => [n, THREE.PropertyBinding.sanitizeNodeName(n)]);

// ---- interpolants ----
const makeGltf = gltfInterpolants(THREE);
const T = [0, 0.4, 1, 1.25, 2.5];
const sweep = [-0.5, 0, 0.1, 0.39, 0.4, 0.41, 0.9, 1, 1.1, 1.249, 1.25, 2, 2.4999, 2.5, 3, 1.7, 0.2, -1, 0.05, 2.45, 1.0001, 0.7];
const interpCases = [];
const quats = [[0, 0, 0, 1], [0, 0.6, 0, 0.8], [0.5, 0.5, 0.5, 0.5], [0, -0.6, 0, -0.8], [0.1, 0.2, 0.3, 0.927362]];
for (const [kind, size, values] of [
    ['linear', 2, [0, 1, 2, 3, -1, 4, 5, 5, 0.5, -2]],
    ['discrete', 2, [0, 1, 2, 3, -1, 4, 5, 5, 0.5, -2]],
    ['cubic', 1, [0, 2, -1, 3, 0.5]],
    ['quaternion', 4, quats.flat()],
]) {
    const settingsList = kind === 'cubic'
        ? [null, [2401, 2401], [2402, 2402], [2400, 2402], [2402, 2401]]
        : [null];
    for (const settings of settingsList) {
        const C = { linear: THREE.LinearInterpolant, discrete: THREE.DiscreteInterpolant, cubic: THREE.CubicInterpolant,
            quaternion: THREE.QuaternionLinearInterpolant }[kind];
        const interp = new C(new Float32Array(T), new Float32Array(values), size);
        if (settings) interp.settings = { endingStart: settings[0], endingEnd: settings[1] };
        const results = sweep.map((t) => Array.from(interp.evaluate(t)));
        interpCases.push({ kind, size, times: T, values, settings, ts: sweep, results });
    }
}
for (const quatTrack of [false, true]) {
    const size = quatTrack ? 4 : 2;
    const values = [];
    for (let k = 0; k < T.length; k++) {
        for (let part = 0; part < 3; part++) {
            for (let j = 0; j < size; j++) values.push(quatTrack && part === 1 ? quats[k][j] : Math.sin(k * 3 + part * 1.7 + j));
        }
    }
    const C = quatTrack ? THREE.QuaternionKeyframeTrack : THREE.VectorKeyframeTrack;
    const tr = new C('x.y', T, values);
    makeGltf(tr);
    const interp = tr.createInterpolant();
    const results = sweep.map((t) => Array.from(interp.evaluate(t)));
    interpCases.push({ kind: quatTrack ? 'gltfQuat' : 'gltf', size, times: T, values: Array.from(tr.values), settings: null, ts: sweep, results });
}

// ---- tracks ----
const trackJSON = (t) => enc(THREE.KeyframeTrack.toJSON(t));
const trackCases = [];
const mk = (type, name, times, values, interp) => {
    const C = { number: THREE.NumberKeyframeTrack, vector: THREE.VectorKeyframeTrack, quaternion: THREE.QuaternionKeyframeTrack,
        color: THREE.ColorKeyframeTrack, bool: THREE.BooleanKeyframeTrack, string: THREE.StringKeyframeTrack }[type];
    return new C(name, times, values, interp);
};
const trackInputs = [
    ['number', 'a.x', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 0, 0, 1, 1, 1, 0], undefined],
    ['vector', 'a.position', [0, 1, 1, 2, 3, 3.5], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 2, 0, 0], undefined],
    ['number', 'a.s', [0, 0.5, 1, 1.5], [1, 1, 1, 1], 2302],
    ['quaternion', 'a.quaternion', [0, 0.25, 0.75], [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0], undefined],
    ['bool', 'a.visible', [0, 1, 2, 3], [true, true, false, false], undefined],
    ['string', 'a.label', [0, 1, 2], ['x', 'x', 'y'], undefined],
    ['color', 'a.color', [0.1, 0.7, 1.3], [1, 0, 0, 0, 1, 0, 0, 0, 1], 2300],
    ['number', 'a.q', [0, 2, 1], [0, 1, 2], undefined],
    ['number', 'a.n', [0, 1], [0, NaN], undefined],
    ['vector', 'a.v', [0, 1], [0, 1, 2, 3, 4], undefined],
    ['quaternion', 'a.qs', [0, 1], [0, 0, 0, 1, 0, 0, 0, 1], 2302],
];
for (const [type, name, times, values, interp] of trackInputs) {
    const base = () => mk(type, name, times, values, interp);
    const input = { type, name, times, values, interpolation: interp ?? null };
    const out = { input, getInterpolation: base().getInterpolation() ?? null, valueSize: base().getValueSize(), validate: base().validate() };
    out.json = trackJSON(base());
    // three.js loops forever optimising a fractional value size.
    if (!Number.isInteger(base().getValueSize())) { trackCases.push(enc(out)); continue; }
    out.optimize = trackJSON(base().optimize());
    out.trim = [[0.2, 1.1], [-1, 0.5], [5, 6], [1, 1]].map(([s, e]) => ({ args: [s, e], out: trackJSON(base().trim(s, e)) }));
    out.shift = trackJSON(base().shift(0.123456789));
    out.scale = trackJSON(base().scale(1 / 30));
    trackCases.push(enc(out));
}

// ---- clips ----
const clipJSON = (c) => enc(THREE.AnimationClip.toJSON(c));
const clipCases = {};
{
    const tracks = () => [
        mk('vector', 'a.position', [0, 0.5, 2], [0, 0, 0, 1, 2, 3, 4, 5, 6]),
        mk('quaternion', 'a.quaternion', [0.25, 1.5], [0, 0, 0, 1, 0, 0.6, 0, 0.8]),
        mk('number', 'b.x', [0, 0.1, 0.2, 3], [1, 1, 1, 2], 2302),
    ];
    const c = new THREE.AnimationClip('walk', -1, tracks());
    clipCases.resetDuration = c.duration;
    clipCases.toJSON = clipJSON(c);
    clipCases.trim = clipJSON(new THREE.AnimationClip('t', 1, tracks()).trim());
    clipCases.optimize = clipJSON(new THREE.AnimationClip('o', -1, tracks()).optimize());
    clipCases.validate = new THREE.AnimationClip('v', -1, tracks()).validate();
    const parseIn = { name: 'p', fps: 25, uuid: 'FIXED-UUID', blendMode: 2501, tracks: [
        { name: 'a.position', type: 'vector', times: [0, 10, 25], values: [0, 0, 0, 1, 1, 1, 2, 2, 2] },
        { name: 'a.visible', type: 'boolean', keys: [{ time: 0, value: true }, { time: 5, value: false }, { time: 9 }] },
        { name: 'a.s', type: 'scalar', keys: [{ time: 0, value: [1] }, { time: 12, value: [3] }], interpolation: 2302 },
        { name: 'a.label', type: 'string', times: [0, 3], values: ['a', 'b'] },
    ] };
    clipCases.parseIn = parseIn;
    clipCases.parse = clipJSON(THREE.AnimationClip.parse(JSON.parse(JSON.stringify(parseIn))));
    clipCases.parseNoDuration = clipJSON(THREE.AnimationClip.parse({ name: 'nd', tracks: [{ name: '.x', type: 'number', times: [0, 4], values: [0, 1] }] }));
    const morphs = ['walk_001', 'walk_002', 'walk_003', 'run1_001', 'run1_002', '12', 'jump-a10', 'bad.name1', 'idle'];
    clipCases.morphSeq = [false, true].map((noLoop) =>
        clipJSON(THREE.AnimationClip.CreateFromMorphTargetSequence('seq', morphs.slice(0, 4).map((name) => ({ name })), 24, noLoop)));
    clipCases.morphClips = THREE.AnimationClip.CreateClipsFromMorphTargetSequences(morphs.map((name) => ({ name })), 30, false).map(clipJSON);
    clipCases.morphNames = morphs;
    const src = new THREE.AnimationClip('src', -1, [
        mk('vector', 'a.position', [0, 0.1, 0.2, 0.3, 0.4, 0.5], [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]),
        mk('number', 'b.x', [0.05, 0.45], [7, 8]),
        mk('number', 'c.x', [0.9, 1], [7, 8]),
    ]);
    clipCases.subclipSrc = clipJSON(src);
    clipCases.subclip = [[3, 12, 30], [0, 3, 30], [2, 14, 24]].map(([s, e, fps]) => ({ args: [s, e, fps],
        out: clipJSON(THREE.AnimationUtils.subclip(src, 'sub', s, e, fps)) }));
    // makeClipAdditive: reference before, after, between keys; quaternion; glTF.
    const addCases = [];
    for (const [frame, fps, useRef] of [[0, 30, false], [6, 30, false], [100, 30, false], [5, 24, true], [3, 0, false]]) {
        const target = new THREE.AnimationClip('tgt', -1, [
            mk('vector', 'a.position', [0, 0.25, 0.5], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            mk('quaternion', 'a.quaternion', [0, 0.5], [0, 0, 0, 1, 0, 0.6, 0, 0.8]),
            mk('bool', 'a.visible', [0], [true]),
            mk('number', 'a.s', [0, 0.1, 0.2], [1, 2, 4], 2302),
        ]);
        const ref = useRef ? new THREE.AnimationClip('ref', -1, [
            mk('vector', 'a.position', [0, 1], [10, 10, 10, 20, 20, 20]),
            mk('quaternion', 'a.quaternion', [0, 1], [0, 0, 0.6, 0.8, 0, 0, 0, 1]),
        ]) : undefined;
        THREE.AnimationUtils.makeClipAdditive(target, frame, ref, fps);
        addCases.push({ args: [frame, fps, useRef], out: clipJSON(target) });
    }
    // a glTF cubic target, reference at a key and between keys
    for (const frame of [0, 9]) {
        const vals = [];
        for (let k = 0; k < 3; k++) for (let p = 0; p < 3; p++) for (let j = 0; j < 3; j++) vals.push(k + p * 0.1 + j * 0.01);
        const tr = mk('vector', 'a.position', [0, 0.25, 0.5], vals);
        makeGltf(tr);
        const qv = [];
        for (let k = 0; k < 2; k++) for (let p = 0; p < 3; p++) qv.push(...(p === 1 ? quats[k + 1] : [0.01, 0.02, 0.03, 0.04]));
        const tq = mk('quaternion', 'a.quaternion', [0, 0.5], qv);
        makeGltf(tq);
        const target = new THREE.AnimationClip('gltf', -1, [tr, tq]);
        THREE.AnimationUtils.makeClipAdditive(target, frame, undefined, 30);
        addCases.push({ args: [frame, 30, false], gltf: true, out: clipJSON(target) });
    }
    clipCases.makeClipAdditive = addCases;
    clipCases.keyframeOrder = THREE.AnimationUtils.getKeyframeOrder([3, 1, 2, 1, 0.5, NaN, 2]);
    clipCases.sortedArray = Array.from(THREE.AnimationUtils.sortedArray([10, 11, 20, 21, 30, 31], 2, [2, 0, 1]));
}

// ---- a glTF file through GLTFLoader: every animated node, sampled ----
globalThis.self ??= globalThis;
globalThis.ProgressEvent ??= class { constructor(type, o) { Object.assign(this, o); this.type = type; } };
const { GLTFLoader } = await import(pathToFileURL(join(repo, 'web/node_modules/three/examples/jsm/loaders/GLTFLoader.js')).href);
const gltfText = readFileSync(join(repo, 'src/loaders/testdata/gltf/features.gltf'), 'utf8');
const gltf = await new Promise((res, rej) => {
    const [e, w] = [console.error, console.warn];
    console.error = () => {}; console.warn = () => {};
    new GLTFLoader().parse(gltfText, '', (g) => { console.error = e; console.warn = w; res(g); },
        (err) => { console.error = e; console.warn = w; rej(err); });
});
const gltfTruth = { clips: gltf.animations.map((c) => [c.name, c.duration, c.tracks.map((t) => t.name)]), frames: [] };
{
    const names = [...new Set(gltf.animations.flatMap((c) => c.tracks.map((t) => THREE.PropertyBinding.parseTrackName(t.name).nodeName)))];
    gltfTruth.nodes = names;
    const mixer = new THREE.AnimationMixer(gltf.scene);
    const [move, other] = gltf.animations;
    mixer.clipAction(move).play();
    const b = mixer.clipAction(other);
    b.setLoop(LoopPingPongC, Infinity);
    b.play();
    const sample = () => names.map((n) => {
        const o = THREE.PropertyBinding.findNode(gltf.scene, n);
        if (!o) return null;
        return [o.position.toArray(), o.quaternion.toArray(), o.scale.toArray(), o.morphTargetInfluences ? o.morphTargetInfluences.slice() : null];
    });
    for (let i = 0; i < 40; i++) {
        mixer.update(i === 0 ? 0 : 0.07 + (i % 3) * 0.01);
        gltfTruth.frames.push({ time: mixer.time, nodes: sample() });
    }
}

// ---- scenarios ----
const scenarios = SCENARIOS.map((sc) => {
    uuidN = 0;
    const r = runScenario(THREE, sc, { makeGltf });
    return enc({ scenario: sc, frames: r.frames, events: r.events, clips: r.clips ?? null });
});

const out = { three: THREE.REVISION, gltf: enc(gltfTruth), parse, sanitize, interpolants: interpCases, tracks: trackCases, clips: enc(clipCases), scenarios };
const file = join(here, 'three-r165-truth.json');
writeFileSync(file, JSON.stringify(out));
console.log(`wrote ${file}: ${scenarios.length} scenarios, ${scenarios.reduce((n, s) => n + s.frames.length, 0)} frames, `
    + `${interpCases.length} interpolant sweeps, ${trackCases.length} tracks, ${NAMES.length} names`);
