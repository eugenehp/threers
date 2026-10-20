// Animation scenarios run three ways: on three.js r165 (the truth, written
// to three-r165-truth.json by gen-three-truth.mjs), on the shim over a real
// wasm build (crates/threers-js/test/animation-parity.test.mjs, frame by
// frame against three.js), and on the Rust port with a JS-like object model
// (src/animation/three/tests.rs, against the JSON).
//
// A scenario is data: a tree of objects (with three.js properties and custom
// ones), clips, and a list of operations on a mixer and its actions; the
// record is every recorded property, every action's state and the mixer's
// stats after each update, plus the events in the order they fired.
// `runScenario(THREE, scenario)` works with any three.js-shaped namespace.

const LoopOnce = 2200, LoopRepeat = 2201, LoopPingPong = 2202;
const Additive = 2501;
const Discrete = 2300, Linear = 2301, Smooth = 2302;

// ---- scenario building helpers ----
const node = (name, extra = {}, children = []) => ({ name, ...extra, children });
const track = (type, name, times, values, interpolation) =>
    interpolation === undefined ? { type, name, times, values } : { type, name, times, values, interpolation };
const quat = (ax, ay, az, angle) => {
    const l = Math.hypot(ax, ay, az), s = Math.sin(angle / 2) / l;
    return [ax * s, ay * s, az * s, Math.cos(angle / 2)];
};
const steps = (dt, n) => Array.from({ length: n }, () => ['update', dt]);

const basicTree = () => node('root', {}, [
    node('cube', {
        position: [0, 0, 0],
        custom: { custom: 0.25, flag: { bool: true }, label: { string: 'idle' }, weights: { array: [0, 0, 0] },
            tint: { color: [1, 1, 1] }, offset: { vector3: [0, 0, 0] } },
    }),
    node('arm', { position: [1, 0, 0], quaternion: [0, 0, 0, 1] }, [
        node('hand', { position: [0, 1, 0], scale: [1, 1, 1], custom: { custom: 1 } }),
    ]),
]);

const qA = quat(0, 1, 0, 0), qB = quat(0, 1, 0, 2.5), qC = quat(1, 1, 0, -1.2), qD = quat(0.2, 0.1, 1, 3.0);

export const SCENARIOS = [
    {
        name: 'linear-repeat-loops',
        tree: basicTree(),
        clips: [{ name: 'move', duration: -1, tracks: [
            track('vector', 'cube.position', [0, 0.5, 1.25, 2], [0, 0, 0, 1, 2, 3, -1, 0.5, 2, 0, 0, 0]),
            track('number', 'cube.custom', [0, 1, 2], [0.25, 3, -1]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['play', 'a'],
            ...steps(0.1, 5), ['update', 0.77], ['update', 1.9], ['update', 4.33], ['update', 0], ['update', 0.016],
            ['set', 'a', 'timeScale', -1.5], ...steps(0.3, 6), ['set', 'a', 'repetitions', 3], ...steps(0.9, 6)],
        record: [['cube', 'position'], ['cube', 'custom']],
    },
    {
        name: 'quaternion-pingpong-finish',
        tree: basicTree(),
        clips: [{ name: 'turn', duration: -1, tracks: [
            track('quaternion', 'arm.quaternion', [0, 0.4, 1, 1.5], [...qA, ...qB, ...qC, ...qD]),
            track('vector', 'hand.scale', [0, 1.5], [1, 1, 1, 2, 0.5, 3]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['setLoop', 'a', LoopPingPong, 3], ['play', 'a'],
            ...steps(0.13, 40)],
        record: [['arm', 'quaternion'], ['hand', 'scale']],
    },
    {
        name: 'smooth-endings',
        tree: basicTree(),
        clips: [{ name: 'wave', duration: -1, tracks: [
            track('number', 'cube.custom', [0, 0.3, 0.9, 1.4, 2], [0, 1, -0.5, 2, 0.5], Smooth),
            track('vector', 'cube.position', [0, 1, 2], [0, 0, 0, 1, 1, 1, 0, 2, 0], Smooth),
            track('color', 'cube.tint', [0, 1, 2], [1, 0, 0, 0, 1, 0, 0, 0, 1], Smooth),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['play', 'a'], ...steps(0.11, 25),
            ['set', 'a', 'zeroSlopeAtStart', false], ['set', 'a', 'zeroSlopeAtEnd', false], ['reset', 'a'],
            ['setLoop', 'a', LoopRepeat, 2], ...steps(0.17, 15),
            ['stop', 'a'], ['setLoop', 'a', LoopPingPong, Infinity], ['play', 'a'], ...steps(0.21, 20),
            ['stop', 'a'], ['setLoop', 'a', LoopOnce, 1], ['play', 'a'], ...steps(0.19, 14)],
        record: [['cube', 'custom'], ['cube', 'position'], ['cube', 'tint']],
    },
    {
        name: 'discrete-bool-string',
        tree: basicTree(),
        clips: [{ name: 'flags', duration: 2, tracks: [
            track('bool', 'cube.flag', [0, 0.5, 1.5], [true, false, true]),
            track('string', 'cube.label', [0, 0.7, 1.2], ['idle', 'walk', 'run']),
            track('number', 'cube.custom', [0, 0.6, 1.1], [1, 5, -2], Discrete),
            track('vector', 'cube.offset', [0, 1], [0, 0, 0, 3, 4, 5], Discrete),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['play', 'a'], ...steps(0.15, 20),
            ['clipAction', 'b', 0, null, Additive], ['set', 'b', 'weight', 0.6], ['play', 'b'], ...steps(0.15, 10)],
        record: [['cube', 'flag'], ['cube', 'label'], ['cube', 'custom'], ['cube', 'offset']],
    },
    {
        name: 'loop-once-clamp-reverse',
        tree: basicTree(),
        clips: [{ name: 'once', duration: -1, tracks: [
            track('vector', 'cube.position', [0, 1], [0, 0, 0, 4, 2, 1]),
        ] }, { name: 'once2', duration: 1.5, tracks: [
            track('number', 'cube.custom', [0, 1], [10, 20]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['setLoop', 'a', LoopOnce, 1], ['set', 'a', 'clampWhenFinished', true],
            ['play', 'a'], ...steps(0.3, 5), ['set', 'a', 'paused', false], ['set', 'a', 'timeScale', -1], ...steps(0.3, 5),
            ['clipAction', 'b', 1], ['setLoop', 'b', LoopOnce, 1], ['play', 'b'], ...steps(0.4, 5),
            ['reset', 'b'], ['play', 'b'], ['update', 2], ['update', 0.1]],
        record: [['cube', 'position'], ['cube', 'custom']],
    },
    {
        name: 'weights-blend-to-original',
        tree: basicTree(),
        clips: [{ name: 'a', duration: -1, tracks: [
            track('vector', 'cube.position', [0, 1], [0, 0, 0, 2, 0, 0]),
            track('quaternion', 'arm.quaternion', [0, 1], [...qA, ...qB]),
            track('number', 'cube.custom', [0, 1], [1, 2]),
        ] }, { name: 'b', duration: -1, tracks: [
            track('vector', 'cube.position', [0, 2], [0, 3, 0, 0, -3, 1]),
            track('quaternion', 'arm.quaternion', [0, 2], [...qC, ...qD]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['clipAction', 'b', 1],
            ['setEffectiveWeight', 'a', 0.3], ['setEffectiveWeight', 'b', 0.45], ['play', 'a'], ['play', 'b'], ...steps(0.1, 8),
            ['setEffectiveWeight', 'b', 0], ...steps(0.1, 3), ['set', 'a', 'enabled', false], ...steps(0.1, 3),
            ['set', 'a', 'enabled', true], ['setEffectiveWeight', 'a', 1.0], ['setEffectiveWeight', 'b', 2.0], ...steps(0.2, 4),
            ['stop', 'b'], ['update', 0.1], ['stop', 'a'], ['update', 0.1]],
        record: [['cube', 'position'], ['arm', 'quaternion'], ['cube', 'custom']],
    },
    {
        name: 'fades-crossfades-warps',
        tree: basicTree(),
        clips: [{ name: 'walk', duration: -1, tracks: [
            track('vector', 'cube.position', [0, 0.5, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
            track('quaternion', 'arm.quaternion', [0, 1], [...qA, ...qB]),
        ] }, { name: 'run', duration: -1, tracks: [
            track('vector', 'cube.position', [0, 0.3, 0.6], [1, 0, 0, 1, 2, 0, 1, 0, 0]),
            track('quaternion', 'arm.quaternion', [0, 0.6], [...qC, ...qD]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'w', 0], ['clipAction', 'r', 1], ['play', 'w'], ['fadeIn', 'w', 0.5], ...steps(0.1, 7),
            ['play', 'r'], ['crossFadeTo', 'w', 'r', 0.6, true], ...steps(0.1, 9),
            ['play', 'w'], ['crossFadeFrom', 'w', 'r', 0.4, false], ...steps(0.1, 6),
            ['halt', 'w', 0.5], ...steps(0.1, 7), ['set', 'w', 'paused', false], ['warp', 'w', 0.5, 2, 1], ...steps(0.15, 8),
            ['setEffectiveTimeScale', 'w', 0.7], ['mixerTimeScale', 1.5], ...steps(0.1, 4), ['fadeOut', 'w', 0.3], ...steps(0.1, 5)],
        record: [['cube', 'position'], ['arm', 'quaternion']],
    },
    {
        name: 'additive-layers',
        tree: basicTree(),
        clips: [{ name: 'base', duration: -1, tracks: [
            track('vector', 'cube.position', [0, 1], [0, 0, 0, 1, 1, 1]),
            track('quaternion', 'arm.quaternion', [0, 1], [...qA, ...qB]),
            track('number', 'cube.custom', [0, 1], [0, 1]),
        ] }, { name: 'layer', duration: -1, tracks: [
            track('vector', 'cube.position', [0, 0.5, 1], [0, 0, 0, 0, 2, 0, 0, 0, 0]),
            track('quaternion', 'arm.quaternion', [0, 0.5, 1], [...qA, ...qC, ...qA]),
            track('number', 'cube.custom', [0, 1], [0.5, 3]),
        ] }, { name: 'layer2', duration: -1, tracks: [
            track('quaternion', 'arm.quaternion', [0, 0.25, 1], [...qD, ...qB, ...qC]),
            track('vector', 'cube.position', [0, 1], [1, 2, 3, 0, 0, 0], Smooth),
        ] }],
        ops: [['makeClipAdditive', 1, 0, null, 30], ['makeClipAdditive', 2, 6, 1, 24],
            ['mixer', 'root'], ['clipAction', 'b', 0], ['clipAction', 'l', 1], ['clipAction', 'l2', 2],
            ['play', 'b'], ['play', 'l'], ['setEffectiveWeight', 'l', 0.7], ...steps(0.1, 6),
            ['play', 'l2'], ['setEffectiveWeight', 'l2', 0.5], ...steps(0.13, 8),
            ['stop', 'b'], ...steps(0.1, 3)],
        record: [['cube', 'position'], ['arm', 'quaternion'], ['cube', 'custom']],
        clipsOut: true,
    },
    {
        name: 'schedule-settime-sync',
        tree: basicTree(),
        clips: [{ name: 'a', duration: 2, tracks: [
            track('vector', 'cube.position', [0, 2], [0, 0, 0, 2, 4, 6]),
        ] }, { name: 'b', duration: 1, tracks: [
            track('number', 'cube.custom', [0, 1], [0, 10]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['clipAction', 'b', 1], ['startAt', 'a', 0.35], ['play', 'a'], ['play', 'b'],
            ...steps(0.1, 6), ['setTime', 1.3], ['update', 0.05], ['syncWith', 'b', 'a'], ['update', 0.1],
            ['setDuration', 'b', 4], ...steps(0.25, 4), ['stopAllAction'], ['update', 0.1], ['play', 'a'], ['update', 0.3]],
        record: [['cube', 'position'], ['cube', 'custom']],
    },
    {
        name: 'listeners-mid-update',
        tree: basicTree(),
        clips: [{ name: 'once', duration: 0.5, tracks: [
            track('vector', 'cube.position', [0, 0.5], [0, 0, 0, 1, 0, 0]),
        ] }, { name: 'loop', duration: 0.4, tracks: [
            track('number', 'cube.custom', [0, 0.4], [0, 1]),
            track('vector', 'cube.position', [0, 0.4], [0, 1, 0, 0, 2, 0]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'o', 0], ['clipAction', 'l', 1], ['setLoop', 'o', LoopOnce, 1],
            ['play', 'o'], ...steps(0.15, 12)],
        listeners: {
            finished: [['play', 'l'], ['crossFadeFrom', 'l', '$event', 0.3, false]],
            loop: [['setEffectiveWeight', '$event', 0.5]],
        },
        record: [['cube', 'position'], ['cube', 'custom']],
    },
    {
        name: 'morph-material-bones',
        tree: node('root', {}, [
            node('face', { kind: 'mesh', morphTargets: ['smile', 'frown', 'blink'], material: { color: [1, 0, 0], opacity: 1 } }),
            node('body', { kind: 'skinned', bones: ['hip', 'knee'], material: { color: [0, 1, 0], opacity: 0.5 } }, [
                node('hip', { kind: 'bone', position: [0, 1, 0] }, [node('knee', { kind: 'bone', position: [0, -0.5, 0] })]),
            ]),
            node('lamp', { uuid: 'ABCDEF01-2345-4678-89AB-CDEF01234567', position: [0, 3, 0] }),
        ]),
        clips: [{ name: 'all', duration: -1, tracks: [
            track('number', 'face.morphTargetInfluences[smile]', [0, 1], [0, 1]),
            track('number', 'face.morphTargetInfluences[2]', [0, 1], [1, 0.25]),
            track('number', 'face.material.opacity', [0, 1], [1, 0.2]),
            track('color', 'face.material.color', [0, 1], [1, 0, 0, 0, 0, 1]),
            track('vector', 'body.bones[hip].position', [0, 1], [0, 1, 0, 0, 2, 0]),
            track('quaternion', 'knee.quaternion', [0, 1], [...qA, ...qC]),
            track('vector', 'ABCDEF01-2345-4678-89AB-CDEF01234567.position', [0, 1], [0, 3, 0, 1, 3, 1]),
            track('number', 'body.material.opacity', [0, 1], [0.5, 1]),
        ] }, { name: 'array', duration: -1, tracks: [
            track('number', 'face.morphTargetInfluences', [0, 1], [0, 0, 0, 1, 0.5, 0.25]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['play', 'a'], ...steps(0.2, 4),
            ['clipAction', 'm', 1], ['setEffectiveWeight', 'm', 0.5], ['play', 'm'], ...steps(0.2, 4), ['stopAllAction'], ['update', 0.1]],
        record: [['face', 'morphTargetInfluences'], ['face', 'material.opacity'], ['face', 'material.color'],
            ['hip', 'position'], ['knee', 'quaternion'], ['lamp', 'position'], ['body', 'material.opacity']],
    },
    {
        name: 'object-group',
        tree: node('root', {}, [
            node('g1', { position: [0, 0, 0] }), node('g2', { position: [1, 0, 0] }), node('g3', { position: [2, 0, 0] }),
            node('g4', { position: [3, 0, 0] }),
        ]),
        clips: [{ name: 'bob', duration: -1, tracks: [
            track('vector', '.position', [0, 1], [0, 0, 0, 0, 2, 0]),
            track('quaternion', '.quaternion', [0, 1], [...qA, ...qD]),
        ] }],
        ops: [['group', ['g1', 'g2']], ['mixerGroup'], ['clipAction', 'a', 0], ['play', 'a'], ...steps(0.2, 2),
            ['groupAdd', ['g3']], ...steps(0.2, 2), ['groupRemove', ['g1']], ...steps(0.2, 2),
            ['groupAdd', ['g1', 'g4']], ...steps(0.2, 2), ['groupUncache', ['g2']], ...steps(0.2, 2)],
        record: [['g1', 'position'], ['g2', 'position'], ['g3', 'position'], ['g4', 'position'], ['g3', 'quaternion']],
    },
    {
        name: 'uncache-and-roots',
        tree: node('root', {}, [
            node('cube', { position: [0, 0, 0] }),
            node('other', { position: [5, 0, 0] }, [node('cube', { position: [6, 0, 0] })]),
        ]),
        clips: [{ name: 'slide', duration: -1, tracks: [
            track('vector', 'cube.position', [0, 1], [0, 0, 0, 1, 1, 0]),
        ] }, { name: 'lift', duration: -1, tracks: [
            track('vector', 'cube.position', [0, 1], [0, 0, 0, 0, 0, 3]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['clipAction', 'b', 0, 'other'], ['clipAction', 'c', 1],
            ['clipAction', 'd', 0, null, Additive], ['play', 'a'], ['play', 'b'], ['play', 'c'], ['play', 'd'], ...steps(0.2, 3),
            ['uncacheAction', 1], ['update', 0.2], ['uncacheRoot', 'other'], ['update', 0.2], ['uncacheClip', 0], ['update', 0.2]],
        record: [['cube', 'position'], ['other/cube', 'position']],
    },
    {
        name: 'gltf-cubic-spline',
        tree: basicTree(),
        clips: [{ name: 'cubic', duration: -1, tracks: [
            { ...track('vector', 'cube.position', [0, 0.6, 1.5],
                [0, 0, 0, 0, 0, 0, 1, 2, 0, -1, 1, 0, 2, 1, 1, 0, 3, 0, 0.5, 0.5, 0.5, 1, 0, 2, 0, 0, 0]), gltfCubic: true },
            { ...track('quaternion', 'arm.quaternion', [0, 1, 2],
                [0, 0, 0, 0, ...qA, 0.1, 0.2, 0, 0, 0, 0.3, 0.1, 0, ...qB, 0, -0.2, 0.1, 0.05, 0, 0, 0, 0, ...qC, 0, 0, 0, 0]),
            gltfCubic: true },
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['play', 'a'], ...steps(0.07, 34)],
        record: [['cube', 'position'], ['arm', 'quaternion']],
    },
    {
        name: 'custom-js-properties',
        tree: basicTree(),
        clips: [{ name: 'props', duration: -1, tracks: [
            track('number', 'cube.weights', [0, 1], [0, 0, 0, 1, 2, 3]),
            track('number', 'cube.weights[1]', [0, 0.5, 1], [5, -5, 5]),
            track('color', 'cube.tint', [0, 1], [0.2, 0.4, 0.6, 1, 0, 0.5]),
            track('vector', 'cube.offset', [0, 1], [1, 2, 3, -3, -2, -1]),
            track('number', 'hand.custom', [0, 1], [7, 9]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['play', 'a'], ...steps(0.12, 10),
            ['setProp', 'cube', 'custom', 42], ['stop', 'a'], ['update', 0.1]],
        record: [['cube', 'weights'], ['cube', 'tint'], ['cube', 'offset'], ['cube', 'custom']],
    },
    {
        name: 'existing-action-blend-modes',
        tree: basicTree(),
        clips: [{ name: 'x', duration: 1, tracks: [
            track('vector', 'cube.position', [0, 1], [0, 0, 0, 1, 0, 0]),
        ] }],
        ops: [['mixer', 'root'], ['clipAction', 'a', 0], ['clipAction', 'a2', 0], ['clipAction', 'add', 0, null, Additive],
            ['existingAction', 'e', 0], ['play', 'a'], ['play', 'add'], ['setEffectiveWeight', 'add', 0.5], ...steps(0.3, 5),
            ['clipActionByName', 'n', 'x'], ['update', 0.1]],
        record: [['cube', 'position']],
    },
];

// ---- runner ----

function readValue(v) {
    if (typeof v === 'number') return [v];
    if (typeof v === 'boolean') return [v ? 1 : 0];
    if (typeof v === 'string') return v;
    if (Array.isArray(v)) return v.slice();
    if (v == null) return null;
    if (v.isQuaternion) return [v.x, v.y, v.z, v.w];
    if (v.isVector3) return [v.x, v.y, v.z];
    if (v.isColor) return [v.r, v.g, v.b];
    if (v.isEuler) return [v.x, v.y, v.z];
    if (typeof v.toArray === 'function') return v.toArray();
    return null;
}

/// Build a scenario's object tree; returns `{ root, nodes }` keyed by path
/// (`name`, or `parent/name` for later duplicates).
export function buildTree(THREE, spec) {
    const nodes = {};
    const skinned = [];
    const make = (s, parentPath) => {
        let obj;
        if (s.kind === 'bone') obj = new THREE.Bone();
        else if (s.kind === 'mesh' || s.kind === 'skinned') {
            const g = new THREE.BufferGeometry();
            g.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3));
            if (s.morphTargets) {
                g.morphAttributes.position = s.morphTargets.map((n, i) => {
                    const a = new THREE.Float32BufferAttribute([0, 0, i, 0, 0, i, 0, 0, i], 3);
                    a.name = n;
                    return a;
                });
            }
            const m = s.material || { color: [1, 1, 1], opacity: 1 };
            const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(...m.color), opacity: m.opacity, transparent: true });
            obj = s.kind === 'skinned' ? new THREE.SkinnedMesh(g, mat) : new THREE.Mesh(g, mat);
            if (s.kind === 'skinned') skinned.push([obj, s.bones]);
        } else obj = new THREE.Object3D();
        obj.name = s.name;
        if (s.uuid) obj.uuid = s.uuid;
        if (s.position) obj.position.fromArray(s.position);
        if (s.quaternion) obj.quaternion.fromArray(s.quaternion);
        if (s.scale) obj.scale.fromArray(s.scale);
        for (const [k, v] of Object.entries(s.custom || {})) {
            if (typeof v === 'number') obj[k] = v;
            else if ('bool' in v) obj[k] = v.bool;
            else if ('string' in v) obj[k] = v.string;
            else if ('array' in v) obj[k] = v.array.slice();
            else if ('color' in v) obj[k] = new THREE.Color(...v.color);
            else if ('vector3' in v) obj[k] = new THREE.Vector3(...v.vector3);
        }
        const path = nodes[s.name] ? `${parentPath}/${s.name}` : s.name;
        nodes[path] = obj;
        for (const c of s.children || []) obj.add(make(c, s.name));
        return obj;
    };
    const root = make(spec, '');
    for (const [mesh, bones] of skinned) mesh.bind(new THREE.Skeleton(bones.map((b) => nodes[b])));
    root.updateMatrixWorld(true);
    return { root, nodes };
}

/// GLTFLoader's cubic-spline interpolants, for the three.js side (three
/// keeps them private to GLTFLoader.js; this is its code).
export function gltfInterpolants(THREE) {
    class GLTFCubicSplineInterpolant extends THREE.Interpolant {
        constructor(pp, sv, ss, rb) { super(pp, sv, ss, rb); }
        copySampleValue_(index) {
            const result = this.resultBuffer, values = this.sampleValues, valueSize = this.valueSize,
                offset = index * valueSize * 3 + valueSize;
            for (let i = 0; i !== valueSize; i++) result[i] = values[offset + i];
            return result;
        }
        interpolate_(i1, t0, t, t1) {
            const result = this.resultBuffer, values = this.sampleValues, stride = this.valueSize;
            const stride2 = stride * 2, stride3 = stride * 3, td = t1 - t0, p = (t - t0) / td, pp = p * p, ppp = pp * p;
            const offset1 = i1 * stride3, offset0 = offset1 - stride3;
            const s2 = -2 * ppp + 3 * pp, s3 = ppp - pp, s0 = 1 - s2, s1 = s3 - pp + p;
            for (let i = 0; i !== stride; i++) {
                const p0 = values[offset0 + i + stride], m0 = values[offset0 + i + stride2] * td;
                const p1 = values[offset1 + i + stride], m1 = values[offset1 + i] * td;
                result[i] = s0 * p0 + s1 * m0 + s2 * p1 + s3 * m1;
            }
            return result;
        }
    }
    const _q = new THREE.Quaternion();
    class GLTFCubicSplineQuaternionInterpolant extends GLTFCubicSplineInterpolant {
        interpolate_(i1, t0, t, t1) {
            const result = super.interpolate_(i1, t0, t, t1);
            _q.fromArray(result).normalize().toArray(result);
            return result;
        }
    }
    return (track) => {
        track.createInterpolant = function InterpolantFactoryMethodGLTFCubicSpline(result) {
            const C = this instanceof THREE.QuaternionKeyframeTrack ? GLTFCubicSplineQuaternionInterpolant : GLTFCubicSplineInterpolant;
            return new C(this.times, this.values, this.getValueSize() / 3, result);
        };
        track.createInterpolant.isInterpolantFactoryMethodGLTFCubicSpline = true;
    };
}

const TRACKS = {
    number: 'NumberKeyframeTrack', vector: 'VectorKeyframeTrack', quaternion: 'QuaternionKeyframeTrack',
    color: 'ColorKeyframeTrack', bool: 'BooleanKeyframeTrack', string: 'StringKeyframeTrack',
};

export function buildClip(THREE, spec, makeGltf) {
    const tracks = spec.tracks.map((t) => {
        const C = THREE[TRACKS[t.type]];
        const tr = new C(t.name, t.times, t.values, t.interpolation);
        if (t.gltfCubic) makeGltf(tr);
        return tr;
    });
    return new THREE.AnimationClip(spec.name, spec.duration, tracks, spec.blendMode);
}

/// Run a scenario; returns `{ frames, events, clips }`. `opts.makeGltf`
/// marks a track as a glTF cubic spline; `opts.onFrame(record)` sees each
/// frame as it is recorded.
export function runScenario(THREE, sc, opts = {}) {
    const makeGltf = opts.makeGltf || gltfInterpolants(THREE);
    const { root, nodes } = buildTree(THREE, sc.tree);
    const clips = sc.clips.map((c) => buildClip(THREE, c, makeGltf));
    root.animations = clips;
    const vars = {}, varOf = new Map(), order = [];
    let mixer = null, group = null;
    const events = [];
    const frames = [];
    let pendingEvents = [];
    const ref = (v, ev) => (v === '$event' ? ev : vars[v]);
    const nodeOf = (n) => (n == null ? undefined : nodes[n]);
    const bindVar = (name, action) => {
        vars[name] = action;
        if (action && !varOf.has(action)) varOf.set(action, name);
        if (!order.includes(name)) order.push(name);
    };
    const record = (label) => {
        const props = sc.record.map(([n, p]) => readValue(p.split('.').reduce((o, k) => o?.[k], nodes[n])));
        const actions = order.map((name) => {
            const a = vars[name];
            if (!a) return null;
            return [a.time, a.getEffectiveWeight(), a.getEffectiveTimeScale(), a.enabled ? 1 : 0, a.paused ? 1 : 0,
                a.isRunning() ? 1 : 0, a.isScheduled() ? 1 : 0, a.timeScale, a.weight];
        });
        const s = mixer.stats;
        const stats = [s.actions.total, s.actions.inUse, s.bindings.total, s.bindings.inUse,
            s.controlInterpolants.total, s.controlInterpolants.inUse];
        const frame = { op: label, time: mixer.time, props, actions, stats, events: pendingEvents };
        pendingEvents = [];
        frames.push(frame);
        opts.onFrame?.(frame);
    };
    const exec = (op, ev) => {
        const [kind, ...a] = op;
        const act = (v) => ref(v, ev);
        switch (kind) {
            case 'mixer': mixer = new THREE.AnimationMixer(nodes[a[0]]); hook(); break;
            case 'group': group = new THREE.AnimationObjectGroup(...a[0].map((n) => nodes[n])); break;
            case 'mixerGroup': mixer = new THREE.AnimationMixer(group); hook(); break;
            case 'groupAdd': group.add(...a[0].map((n) => nodes[n])); break;
            case 'groupRemove': group.remove(...a[0].map((n) => nodes[n])); break;
            case 'groupUncache': group.uncache(...a[0].map((n) => nodes[n])); break;
            case 'clipAction': bindVar(a[0], mixer.clipAction(clips[a[1]], nodeOf(a[2]), a[3] ?? undefined)); break;
            case 'clipActionByName': bindVar(a[0], mixer.clipAction(a[1])); break;
            case 'existingAction': bindVar(a[0], mixer.existingAction(clips[a[1]], nodeOf(a[2]))); break;
            case 'play': act(a[0]).play(); break;
            case 'stop': act(a[0]).stop(); record(kind); break;
            case 'reset': act(a[0]).reset(); break;
            case 'fadeIn': act(a[0]).fadeIn(a[1]); break;
            case 'fadeOut': act(a[0]).fadeOut(a[1]); break;
            case 'crossFadeTo': act(a[0]).crossFadeTo(act(a[1]), a[2], a[3]); break;
            case 'crossFadeFrom': act(a[0]).crossFadeFrom(act(a[1]), a[2], a[3]); break;
            case 'warp': act(a[0]).warp(a[1], a[2], a[3]); break;
            case 'halt': act(a[0]).halt(a[1]); break;
            case 'startAt': act(a[0]).startAt(a[1]); break;
            case 'setLoop': act(a[0]).setLoop(a[1], a[2]); break;
            case 'setEffectiveWeight': act(a[0]).setEffectiveWeight(a[1]); break;
            case 'setEffectiveTimeScale': act(a[0]).setEffectiveTimeScale(a[1]); break;
            case 'setDuration': act(a[0]).setDuration(a[1]); break;
            case 'syncWith': act(a[0]).syncWith(act(a[1])); break;
            case 'stopFading': act(a[0]).stopFading(); break;
            case 'stopWarping': act(a[0]).stopWarping(); break;
            case 'set': act(a[0])[a[1] === 'loop' ? 'loop' : a[1]] = a[2]; break;
            case 'mixerTimeScale': mixer.timeScale = a[0]; break;
            case 'setProp': {
                const o = nodes[a[0]];
                const cur = o[a[1]];
                if (cur && typeof cur.fromArray === 'function') cur.fromArray(a[2]); else o[a[1]] = a[2];
                break;
            }
            case 'update': mixer.update(a[0]); record(kind); break;
            case 'setTime': mixer.setTime(a[0]); record(kind); break;
            case 'stopAllAction': mixer.stopAllAction(); record(kind); break;
            case 'uncacheClip': mixer.uncacheClip(clips[a[0]]); break;
            case 'uncacheAction': mixer.uncacheAction(clips[a[0]], nodeOf(a[1])); break;
            case 'uncacheRoot': mixer.uncacheRoot(nodes[a[0]]); break;
            case 'makeClipAdditive':
                THREE.AnimationUtils.makeClipAdditive(clips[a[0]], a[1], a[2] == null ? undefined : clips[a[2]], a[3]);
                break;
            default: throw new Error(`unknown op ${kind}`);
        }
    };
    let depth = 0;
    const hook = () => {
        for (const type of ['finished', 'loop']) {
            mixer.addEventListener(type, (e) => {
                const ev = [type, varOf.get(e.action) ?? null, type === 'finished' ? e.direction : e.loopDelta];
                events.push(ev);
                pendingEvents.push(ev);
                if (depth > 4) return;
                depth++;
                for (const op of sc.listeners?.[type] || []) exec(op, e.action);
                depth--;
            });
        }
    };
    for (const op of sc.ops) exec(op);
    const clipsOut = sc.clipsOut ? clips.map((c) => JSON.parse(JSON.stringify(THREE.AnimationClip.toJSON(c)))) : undefined;
    return { frames, events, clips: clipsOut, mixer, nodes, vars };
}

/// Values agree to `rel` relative (absolute below `abs`); strings exactly.
export function near(a, b, rel = 1e-6, abs = 1e-9) {
    if (typeof a === 'string' || typeof b === 'string') return a === b;
    if (a === null || b === null) return a === b;
    if (Number.isNaN(a) && Number.isNaN(b)) return true;
    if (a === b) return true;
    return Math.abs(a - b) <= Math.max(abs, rel * Math.max(Math.abs(a), Math.abs(b)));
}
