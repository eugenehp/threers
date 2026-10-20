// The shim's animation system (three.js r165's, run in Rust through the real
// wasm build) against three.js r165 itself, on the scenarios in
// tests/animation/scenarios.mjs: the same object trees, clips and calls on
// both sides, compared frame by frame — every recorded property of every
// object, every action's time / effective weight / effective time scale /
// flags, the mixer's time and stats, and the events in the order listeners
// saw them. Then the parts under the mixer (interpolants, track-name parsing,
// track and clip edits, AnimationUtils), that what the mixer writes reaches
// the Rust scene graph's nodes, and the cost of an update for a 100-bone
// skinned character.
//
// The shim runs as a module over a wasm build (`THREERS_PKG_DIR`, else
// web/pkg); three.js is web/node_modules/three (0.165). Values agree to 1e-6
// relative.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { SCENARIOS, runScenario, near } from '../../../tests/animation/scenarios.mjs';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const threeDir = join(repo, 'web/node_modules/three');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');

async function loadShimOver(dir) {
    const pkgUrl = pathToFileURL(join(dir, 'threers.js')).href;
    const m = await import(pkgUrl);
    if (typeof m.WebAnimationObjectGroup !== 'function') return null;
    m.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
    const web = join(repo, 'web');
    const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
        .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
        .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
    const file = join(mkdtempSync(join(tmpdir(), 'threers-anim-')), 'shim.mjs');
    writeFileSync(file, src);
    return import(pathToFileURL(file).href);
}

const haveThree = existsSync(join(threeDir, 'build/three.module.js'));
const S = existsSync(join(pkgDir, 'threers_bg.wasm')) ? await loadShimOver(pkgDir) : null;
const THREE = haveThree ? await import(pathToFileURL(join(threeDir, 'build/three.module.js')).href) : null;
const skip = (!S && `no wasm build with the animation bindings in ${pkgDir}`) || (!haveThree && 'three missing');

// three.js and the shim both log what three.js logs for unbindable tracks.
const quiet = (f) => {
    const [e, w] = [console.error, console.warn];
    console.error = () => {};
    console.warn = () => {};
    try { return f(); } finally { console.error = e; console.warn = w; }
};

/// The shim marks a glTF cubic-spline track as three.js's GLTFLoader does.
const shimGltf = (track) => {
    const f = function InterpolantFactoryMethodGLTFCubicSpline() { return null; };
    f.isInterpolantFactoryMethodGLTFCubicSpline = true;
    track.createInterpolant = f;
};

function sameValue(a, b) {
    if (typeof a === 'string' || typeof b === 'string' || a === null || b === null) return a === b;
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!near(a[i], b[i])) return false;
    return true;
}

function compareFrames(name, ours, theirs) {
    assert.equal(ours.length, theirs.length, `${name}: frame count`);
    for (let i = 0; i < theirs.length; i++) {
        const [o, t] = [ours[i], theirs[i]];
        const where = `${name} frame ${i} (${t.op})`;
        const show = () => `\n  shim:  ${JSON.stringify(o)}\n  three: ${JSON.stringify(t)}`;
        assert.equal(o.op, t.op, `${where}: op`);
        assert.ok(near(o.time, t.time), `${where}: mixer time${show()}`);
        o.props.forEach((p, k) => assert.ok(sameValue(p, t.props[k]), `${where}: property ${k}${show()}`));
        o.actions.forEach((a, k) => assert.ok(sameValue(a, t.actions[k]), `${where}: action ${k}${show()}`));
        assert.deepEqual(o.stats, t.stats, `${where}: stats${show()}`);
        assert.equal(o.events.length, t.events.length, `${where}: event count${show()}`);
        o.events.forEach((e, k) => {
            const f = t.events[k];
            assert.ok(e[0] === f[0] && e[1] === f[1] && near(e[2], f[2]), `${where}: event ${k}${show()}`);
        });
    }
}

describe('animation: the shim against three.js r165', { skip }, () => {
    for (const sc of SCENARIOS) {
        it(`scenario ${sc.name}`, () => {
            const theirs = quiet(() => runScenario(THREE, sc));
            const ours = quiet(() => runScenario(S, sc, { makeGltf: shimGltf }));
            compareFrames(sc.name, ours.frames, theirs.frames);
            if (theirs.clips) {
                for (let i = 0; i < theirs.clips.length; i++) {
                    const [a, b] = [ours.clips[i], theirs.clips[i]];
                    a.tracks.forEach((t, k) => {
                        assert.equal(t.name, b.tracks[k].name);
                        assert.ok(sameValue(t.values, b.tracks[k].values), `${sc.name}: clip ${i} track ${t.name} values`);
                    });
                }
            }
            // What the mixer wrote reached the Rust scene graph's nodes.
            for (const [path, obj] of Object.entries(ours.nodes)) {
                if (!(obj instanceof S.Object3D)) continue;
                obj._syncNow();
                const m = obj._m;
                const trs = m.graph.w.getTRS(m.handle);
                const js = [obj.position.x, obj.position.y, obj.position.z, obj.quaternion.x, obj.quaternion.y,
                    obj.quaternion.z, obj.quaternion.w, obj.scale.x, obj.scale.y, obj.scale.z];
                for (let k = 0; k < 10; k++) {
                    assert.ok(Math.abs(trs[k] - js[k]) <= 1e-6 * Math.max(1, Math.abs(js[k])),
                        `${sc.name}: ${path} node TRS[${k}] ${trs[k]} vs object ${js[k]}`);
                }
            }
        });
    }

    it('interpolants, standalone', () => {
        const times = [0, 0.4, 1, 1.25, 2.5];
        const sweep = [-0.5, 0, 0.1, 0.39, 0.4, 0.9, 1.1, 1.25, 2.4, 2.5, 3, 1.7, 0.2, -1, 2.45, 0.7];
        const quats = [0, 0, 0, 1, 0, 0.6, 0, 0.8, 0.5, 0.5, 0.5, 0.5, 0, -0.6, 0, -0.8, 0.1, 0.2, 0.3, 0.927362];
        for (const [name, size, values] of [['LinearInterpolant', 2, [0, 1, 2, 3, -1, 4, 5, 5, 0.5, -2]],
            ['DiscreteInterpolant', 2, [0, 1, 2, 3, -1, 4, 5, 5, 0.5, -2]], ['CubicInterpolant', 1, [0, 2, -1, 3, 0.5]],
            ['QuaternionLinearInterpolant', 4, quats]]) {
            for (const settings of [null, { endingStart: 2401, endingEnd: 2402 }]) {
                const a = new S[name](new Float32Array(times), new Float32Array(values), size);
                const b = new THREE[name](new Float32Array(times), new Float32Array(values), size);
                a.settings = b.settings = settings;
                for (const t of sweep) {
                    assert.ok(sameValue(Array.from(a.evaluate(t)), Array.from(b.evaluate(t))), `${name} t=${t}`);
                }
            }
        }
        // A JS subclass with its own interpolate_ gets the seek and its methods.
        class Half extends S.Interpolant {
            interpolate_(i1, t0, t, t1) { this.resultBuffer[0] = (i1 + (t - t0) / (t1 - t0)) / 2; return this.resultBuffer; }
        }
        class HalfT extends THREE.Interpolant {
            interpolate_(i1, t0, t, t1) { this.resultBuffer[0] = (i1 + (t - t0) / (t1 - t0)) / 2; return this.resultBuffer; }
        }
        const h = new Half(new Float32Array(times), new Float32Array([1, 2, 3, 4, 5]), 1);
        const ht = new HalfT(new Float32Array(times), new Float32Array([1, 2, 3, 4, 5]), 1);
        for (const t of sweep) assert.ok(near(h.evaluate(t)[0], ht.evaluate(t)[0]), `subclass t=${t}`);
    });

    it('track names, tracks and clips', () => {
        for (const n of ['Box.position', '.morphTargetInfluences[smile]', 'Box.material.color', '.bones[hip].position',
            'dir/sub:Arm.L.quaternion', 'x.y[a[b]].z', 'Cube.userData[speed].x', 'a.b.c.d']) {
            const a = S.PropertyBinding.parseTrackName(n), b = THREE.PropertyBinding.parseTrackName(n);
            for (const k of ['nodeName', 'objectName', 'objectIndex', 'propertyName', 'propertyIndex']) {
                assert.equal(a[k], b[k], `${n}: ${k}`);
            }
        }
        assert.throws(() => S.PropertyBinding.parseTrackName('position'));
        assert.equal(S.PropertyBinding.sanitizeNodeName('a b.c[d]'), THREE.PropertyBinding.sanitizeNodeName('a b.c[d]'));
        const make = (L) => [
            new L.VectorKeyframeTrack('a.position', [0, 0.5, 1, 1.5, 2], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
            new L.QuaternionKeyframeTrack('a.quaternion', [0.25, 1.5], [0, 0, 0, 1, 0, 0.6, 0, 0.8]),
            new L.NumberKeyframeTrack('b.x', [0, 0.1, 0.2, 3], [1, 1, 1, 2], L.InterpolateSmooth),
            new L.BooleanKeyframeTrack('b.visible', [0, 1, 2], [true, true, false]),
            new L.StringKeyframeTrack('b.label', [0, 1], ['a', 'b']),
        ];
        const json = (L, c) => JSON.parse(JSON.stringify(L.AnimationClip.toJSON(c)));
        const strip = (j) => { delete j.uuid; return j; };
        const sameJson = (a, b, what) => {
            const flat = (j) => JSON.stringify(j, (_k, v) => (typeof v === 'number' ? Math.round(v * 1e6) / 1e6 : v));
            assert.equal(flat(strip(a)), flat(strip(b)), what);
        };
        const ca = new S.AnimationClip('c', -1, make(S)), cb = new THREE.AnimationClip('c', -1, make(THREE));
        assert.equal(ca.duration, cb.duration);
        sameJson(json(S, ca), json(THREE, cb), 'toJSON');
        sameJson(json(S, ca.clone().optimize()), json(THREE, cb.clone().optimize()), 'optimize');
        sameJson(json(S, new S.AnimationClip('t', 1, make(S)).trim()), json(THREE, new THREE.AnimationClip('t', 1, make(THREE)).trim()), 'trim');
        sameJson(json(S, S.AnimationUtils.subclip(ca, 's', 3, 40, 30)), json(THREE, THREE.AnimationUtils.subclip(cb, 's', 3, 40, 30)), 'subclip');
        const pa = S.AnimationClip.parse(json(THREE, cb)), pb = THREE.AnimationClip.parse(json(THREE, cb));
        sameJson(json(S, pa), json(THREE, pb), 'parse');
        const morphs = ['walk_001', 'walk_002', 'run_001', 'run_002', 'idle'].map((name) => ({ name }));
        const ma = S.AnimationClip.CreateClipsFromMorphTargetSequences(morphs, 24, false);
        const mb = THREE.AnimationClip.CreateClipsFromMorphTargetSequences(morphs, 24, false);
        assert.equal(ma.length, mb.length);
        ma.forEach((c, i) => sameJson(json(S, c), json(THREE, mb[i]), `morph clip ${i}`));
        const ta = make(S), tb = make(THREE);
        S.AnimationUtils.makeClipAdditive(new S.AnimationClip('add', -1, ta), 6, undefined, 30);
        THREE.AnimationUtils.makeClipAdditive(new THREE.AnimationClip('add', -1, tb), 6, undefined, 30);
        ta.forEach((t, i) => assert.ok(sameValue(Array.from(t.values), Array.from(tb[i].values)) || t.ValueTypeName === 'bool'
            || t.ValueTypeName === 'string', `makeClipAdditive ${t.name}`));
        assert.deepEqual(S.AnimationUtils.getKeyframeOrder([3, 1, 2, 1, 0.5]), THREE.AnimationUtils.getKeyframeOrder([3, 1, 2, 1, 0.5]));
        const vt = new S.NumberKeyframeTrack('x.y', [0, 2, 1], [0, 1, 2]);
        assert.equal(quiet(() => vt.validate()), false, 'out-of-order keys');
    });

    it('a custom property on a plain object (no scene graph) and a hand-made PropertyMixer', () => {
        const run = (L) => {
            const target = { name: 'root', children: [{ name: 'gauge', reading: 0, needle: new L.Vector3() }] };
            const mixer = new L.AnimationMixer(target);
            const clip = new L.AnimationClip('g', -1, [
                new L.NumberKeyframeTrack('gauge.reading', [0, 1], [0, 100]),
                new L.VectorKeyframeTrack('gauge.needle', [0, 1], [0, 0, 0, 1, 2, 3]),
            ]);
            mixer.clipAction(clip).play();
            const out = [];
            for (let i = 0; i < 6; i++) {
                mixer.update(0.15);
                out.push(target.children[0].reading, ...target.children[0].needle.toArray());
            }
            const pm = new L.PropertyMixer(new L.PropertyBinding(target, 'gauge.reading'), 'number', 1);
            pm.saveOriginalState();
            pm.buffer[0] = 7; pm.accumulate(0, 0.5);
            pm.buffer[0] = 9; pm.accumulate(0, 0.5);
            pm.apply(0);
            out.push(target.children[0].reading);
            return out;
        };
        assert.ok(sameValue(run(S), run(THREE)), `${run(S)} vs ${run(THREE)}`);
    });

    it('a 100-bone skinned character at 60 fps: cost of an update', () => {
        const build = (L) => {
            const bones = [];
            let parent = null;
            for (let i = 0; i < 100; i++) {
                const b = new L.Bone();
                b.name = `bone${i}`;
                b.position.set(0, 0.1, 0);
                if (parent) parent.add(b);
                bones.push(b);
                parent = b;
            }
            const g = new L.BufferGeometry();
            g.setAttribute('position', new L.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3));
            g.setAttribute('skinIndex', new L.Uint16BufferAttribute([0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0], 4));
            g.setAttribute('skinWeight', new L.Float32BufferAttribute([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 4));
            const mesh = new L.SkinnedMesh(g, new L.MeshBasicMaterial());
            mesh.add(bones[0]);
            mesh.bind(new L.Skeleton(bones));
            const scene = new L.Scene();
            scene.add(mesh);
            const times = Array.from({ length: 31 }, (_, k) => k / 30);
            const tracks = [];
            for (const [j, b] of bones.entries()) {
                const q = [], p = [], s = [];
                for (let k = 0; k < times.length; k++) {
                    const a = Math.sin(k / 5 + j) * 0.3;
                    q.push(0, Math.sin(a / 2), 0, Math.cos(a / 2));
                    p.push(0, 0.1 + 0.01 * Math.sin(k), 0);
                    s.push(1, 1, 1);
                }
                tracks.push(new L.QuaternionKeyframeTrack(`${b.name}.quaternion`, times, q));
                tracks.push(new L.VectorKeyframeTrack(`${b.name}.position`, times, p));
                tracks.push(new L.VectorKeyframeTrack(`${b.name}.scale`, times, s));
            }
            const mixer = new L.AnimationMixer(mesh);
            mixer.clipAction(new L.AnimationClip('walk', -1, tracks)).play();
            return { mixer, scene, bones };
        };
        const time = (L, frames, sync) => {
            const { mixer, scene, bones } = build(L);
            for (let i = 0; i < 30; i++) { mixer.update(1 / 60); sync?.(scene); }
            const t0 = performance.now();
            for (let i = 0; i < frames; i++) { mixer.update(1 / 60); sync?.(scene); }
            const ms = (performance.now() - t0) / frames;
            return { ms, last: bones[99].quaternion.toArray() };
        };
        const ours = time(S, 600);
        const withSync = time(S, 600, (scene) => scene._syncNow());
        const theirs = time(THREE, 600);
        assert.ok(sameValue(ours.last, theirs.last), 'same pose');
        console.log(`  100 bones x 3 tracks: shim update ${ours.ms.toFixed(3)} ms (with the scene-graph sync ${withSync.ms.toFixed(3)} ms), three.js ${theirs.ms.toFixed(3)} ms`);
        assert.ok(withSync.ms < 16.7, `an update and its sync fit a 60 fps frame (${withSync.ms} ms)`);
    });
});
