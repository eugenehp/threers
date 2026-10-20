#!/usr/bin/env node
// The shim's GLTFLoader against three.js r165's, in Node, on the same files:
// scene structure, names, types, transforms, userData, geometry (every
// attribute value, index, morph targets, bounds), materials (class and
// parameters, which textures, texture parameters), skeletons and bind
// matrices, instancing, cameras, lights, animation tracks (names, types,
// times, values, interpolation), `gltf.cameras`, `asset`, `userData`, and
// `parser.associations` — and which objects share a geometry or material.
//
//   PKG_DIR=/path/to/wasm-bindgen/out node tests/gltf/gltf-parity.mjs [file.gltf|file.glb ...]
//
// With no files: the fixtures in src/loaders/testdata (tests/gltf/make-fixtures.mjs
// writes the gltf/ ones), plus — with SAMPLES=1 — a set of Khronos
// glTF-Sample-Assets models, downloaded once into SAMPLES_DIR
// (default $TMPDIR/gltf-samples). PKG_DIR defaults to web/pkg.
//
// Node has no DOM, so both loaders get the same minimal stand-ins: `fetch` of
// a local path reads the file, `createImageBitmap` reads a PNG/JPEG header
// for the size, and an `OffscreenCanvas` that "decodes" to grey. PNG and KTX2
// never reach them on the shim's side (Rust decodes those).
import fs from 'node:fs';
import path from 'node:path';
import * as nodeModule from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const pkgDir = path.resolve(process.env.PKG_DIR || path.join(root, 'web/pkg'));

// ---------------------------------------------------------------- environment
// The shim imports its wasm glue as `./pkg/threers.js`; point that at PKG_DIR.
{
    const from = pathToFileURL(path.join(root, 'web/pkg')).href + '/';
    const to = pathToFileURL(pkgDir).href + '/';
    nodeModule.registerHooks({
        resolve(spec, ctx, next) {
            const r = next(spec, ctx);
            return r.url.startsWith(from) ? { ...r, url: to + r.url.slice(from.length) } : r;
        },
    });
}

globalThis.self = globalThis;
globalThis.ProgressEvent ??= class ProgressEvent extends Event { constructor(t, o = {}) { super(t); Object.assign(this, o); } };
const nativeFetch = globalThis.fetch;
globalThis.fetch = async (url, init) => {
    const u = typeof url === 'string' ? url : (url?.url ?? String(url));
    if (/^(https?|data|blob):/i.test(u)) return nativeFetch(url, init);
    const file = u.startsWith('file:') ? fileURLToPath(u) : u;
    try {
        const bytes = fs.readFileSync(file);
        return new Response(bytes, { status: 200, statusText: 'OK' });
    } catch (_) {
        return new Response(null, { status: 404, statusText: 'Not Found' });
    }
};
/// Image size from a PNG or JPEG header.
function imageSize(b) {
    if (b[0] === 0x89 && b[1] === 0x50) {
        const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
        return { width: dv.getUint32(16), height: dv.getUint32(20) };
    }
    if (b[0] === 0xff && b[1] === 0xd8) {
        let i = 2;
        while (i + 9 < b.length) {
            if (b[i] !== 0xff) { i++; continue; }
            const marker = b[i + 1], len = (b[i + 2] << 8) | b[i + 3];
            if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
                return { height: (b[i + 5] << 8) | b[i + 6], width: (b[i + 7] << 8) | b[i + 8] };
            }
            i += 2 + len;
        }
    }
    return null;
}
globalThis.createImageBitmap = async (blob) => {
    const b = new Uint8Array(await blob.arrayBuffer());
    if (blob.type === 'image/webp' || blob.type === 'image/avif') throw new Error('no ' + blob.type + ' decoder');
    const size = imageSize(b);
    // A browser rejects what is not an image (a 404 page, say).
    if (!size) throw new Error('InvalidStateError: the source image could not be decoded');
    return { ...size, close() {} };
};
globalThis.OffscreenCanvas = class {
    constructor(w, h) { this.width = w; this.height = h; }
    getContext() {
        const { width, height } = this;
        return { drawImage() {}, getImageData: () => ({ data: new Uint8ClampedArray(width * height * 4).fill(128) }) };
    }
};

const THREE = await import(pathToFileURL(path.join(root, 'web/node_modules/three/build/three.module.js')).href);
const { GLTFLoader: ThreeGLTFLoader } = await import(pathToFileURL(path.join(root, 'web/node_modules/three/examples/jsm/loaders/GLTFLoader.js')).href);
const { MeshoptDecoder } = await import(pathToFileURL(path.join(root, 'web/node_modules/three/examples/jsm/libs/meshopt_decoder.module.js')).href);
const shim = await import(pathToFileURL(path.join(root, 'web/threejs-shim.js')).href);
await shim.initThreers({ module_or_path: fs.readFileSync(path.join(pkgDir, 'threers_bg.wasm')) });
const S = shim.default;

// ----------------------------------------------------------------------- files
const files = process.argv.slice(2);
if (!files.length) {
    const td = path.join(root, 'src/loaders/testdata');
    files.push(path.join(td, 'gltf/features.gltf'), path.join(td, 'gltf/features.glb'),
        path.join(td, 'gltf/external.gltf'), path.join(td, 'morph_sparse.gltf'),
        path.join(td, 'sphere_plain.glb'), path.join(td, 'sphere_meshopt.glb'));
    if (process.env.SAMPLES === '1') files.push(...await fetchSamples());
}

async function fetchSamples() {
    const dir = process.env.SAMPLES_DIR || path.join(process.env.TMPDIR || '/tmp', 'gltf-samples');
    const base = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/';
    const models = [
        'Fox/glTF-Binary/Fox.glb', 'BoxAnimated/glTF-Binary/BoxAnimated.glb', 'CesiumMan/glTF-Binary/CesiumMan.glb',
        'AnimatedMorphCube/glTF-Binary/AnimatedMorphCube.glb', 'MorphPrimitivesTest/glTF-Binary/MorphPrimitivesTest.glb',
        'SimpleSparseAccessor/glTF-Embedded/SimpleSparseAccessor.gltf', 'InterpolationTest/glTF-Binary/InterpolationTest.glb',
        'MeshPrimitiveModes/glTF-Embedded/MeshPrimitiveModes.gltf', 'SimpleInstancing/glTF-Binary/SimpleInstancing.glb',
        'TextureTransformTest/glTF/TextureTransformTest.gltf', 'LightsPunctualLamp/glTF-Binary/LightsPunctualLamp.glb',
        'Cameras/glTF-Embedded/Cameras.gltf', 'ClearCoatTest/glTF-Binary/ClearCoatTest.glb',
        'SheenChair/glTF-Binary/SheenChair.glb', 'TransmissionTest/glTF-Binary/TransmissionTest.glb',
        'IridescenceLamp/glTF-Binary/IridescenceLamp.glb', 'AnisotropyBarnLamp/glTF-Binary/AnisotropyBarnLamp.glb',
        'SpecularTest/glTF-Binary/SpecularTest.glb', 'AlphaBlendModeTest/glTF-Binary/AlphaBlendModeTest.glb',
        'VertexColorTest/glTF-Binary/VertexColorTest.glb', 'TextureCoordinateTest/glTF-Binary/TextureCoordinateTest.glb',
        'NormalTangentMirrorTest/glTF-Binary/NormalTangentMirrorTest.glb', 'UnlitTest/glTF-Binary/UnlitTest.glb',
        'EmissiveStrengthTest/glTF-Binary/EmissiveStrengthTest.glb', 'DispersionTest/glTF-Binary/DispersionTest.glb',
        'MultipleScenes/glTF-Embedded/MultipleScenes.gltf', 'RiggedFigure/glTF-Binary/RiggedFigure.glb',
        'BrainStem/glTF-Binary/BrainStem.glb', 'DamagedHelmet/glTF-Binary/DamagedHelmet.glb',
        'MetalRoughSpheres/glTF-Binary/MetalRoughSpheres.glb', 'Duck/glTF-Binary/Duck.glb',
        'Duck/glTF/Duck.gltf', 'Duck/glTF/Duck0.bin', 'Duck/glTF/DuckCM.png',
        'TextureTransformTest/glTF/TextureTransformTest.bin', 'TextureTransformTest/glTF/UV.png',
        'TextureTransformTest/glTF/Arrow.png', 'TextureTransformTest/glTF/Correct.png',
        'TextureTransformTest/glTF/NotSupported.png', 'TextureTransformTest/glTF/Error.png',
        'TextureSettingsTest/glTF-Binary/TextureSettingsTest.glb', 'OrientationTest/glTF-Binary/OrientationTest.glb',
        'RecursiveSkeletons/glTF-Binary/RecursiveSkeletons.glb', 'SimpleSkin/glTF-Embedded/SimpleSkin.gltf',
        'SimpleMorph/glTF-Embedded/SimpleMorph.gltf', 'BoxVertexColors/glTF-Binary/BoxVertexColors.glb',
        'MaterialsVariantsShoe/glTF-Binary/MaterialsVariantsShoe.glb',
    ];
    const out = [];
    for (const m of models) {
        const file = path.join(dir, m);
        if (!fs.existsSync(file)) {
            fs.mkdirSync(path.dirname(file), { recursive: true });
            const res = await nativeFetch(base + m);
            if (!res.ok) { console.warn(`skip ${m}: ${res.status}`); continue; }
            fs.writeFileSync(file, Buffer.from(await res.arrayBuffer()));
        }
        if (/\.(gltf|glb)$/.test(m)) out.push(file);
    }
    return out;
}

// ------------------------------------------------------------------------ dump
const MATERIAL_KEYS = [
    'opacity', 'transparent', 'side', 'depthWrite', 'depthTest', 'alphaTest', 'vertexColors', 'flatShading',
    'roughness', 'metalness', 'emissiveIntensity', 'aoMapIntensity', 'bumpScale', 'clearcoat', 'clearcoatRoughness',
    'ior', 'dispersion', 'sheen', 'sheenRoughness', 'transmission', 'thickness', 'attenuationDistance',
    'specularIntensity', 'iridescence', 'iridescenceIOR', 'anisotropy', 'anisotropyRotation', 'size', 'sizeAttenuation',
];
const MATERIAL_COLORS = ['color', 'emissive', 'sheenColor', 'attenuationColor', 'specularColor'];
const MATERIAL_VEC2 = ['normalScale', 'clearcoatNormalScale'];
const MATERIAL_MAPS = [
    'map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap', 'clearcoatMap', 'clearcoatRoughnessMap',
    'clearcoatNormalMap', 'sheenColorMap', 'sheenRoughnessMap', 'transmissionMap', 'thicknessMap',
    'specularIntensityMap', 'specularColorMap', 'iridescenceMap', 'iridescenceThicknessMap', 'anisotropyMap', 'bumpMap',
];

async function dump(gltf) {
    const ids = { geometry: new Map(), material: new Map(), texture: new Map() };
    const id = (kind, o) => {
        if (!o) return null;
        if (!ids[kind].has(o)) ids[kind].set(o, ids[kind].size);
        return ids[kind].get(o);
    };
    const vals = (attr) => {
        const out = [];
        const n = attr.count, s = attr.itemSize;
        // Matrices (inverse bind matrices): float, read as stored.
        if (s > 4) return Array.from(attr.array.subarray ? attr.array.subarray(0, n * s) : attr.array.slice(0, n * s));
        const get = [(i) => attr.getX(i), (i) => attr.getY(i), (i) => attr.getZ(i), (i) => attr.getW(i)];
        for (let i = 0; i < n; i++) for (let k = 0; k < s; k++) out.push(get[k](i));
        return out;
    };
    const texture = (t) => t && {
        id: id('texture', t), name: t.name, channel: t.channel ?? 0, offset: [t.offset.x, t.offset.y],
        repeat: [t.repeat.x, t.repeat.y], rotation: t.rotation, center: [t.center.x, t.center.y],
        wrapS: t.wrapS, wrapT: t.wrapT, magFilter: t.magFilter, minFilter: t.minFilter, flipY: t.flipY,
        colorSpace: t.colorSpace || '', userData: t.userData, image: t.image && [t.image.width, t.image.height],
    };
    const material = (m) => {
        if (!m) return null;
        const out = { id: id('material', m), type: m.type, name: m.name, userData: m.userData };
        for (const k of MATERIAL_KEYS) if (m[k] !== undefined) out[k] = m[k];
        for (const k of MATERIAL_COLORS) if (m[k]) out[k] = [m[k].r, m[k].g, m[k].b];
        for (const k of MATERIAL_VEC2) if (m[k]) out[k] = [m[k].x, m[k].y];
        if (m.iridescenceThicknessRange) out.iridescenceThicknessRange = [...m.iridescenceThicknessRange];
        for (const k of MATERIAL_MAPS) if (m[k] !== undefined) out[k] = texture(m[k]);
        return out;
    };
    const geometry = (g) => {
        if (!g) return null;
        const out = { id: id('geometry', g), attributes: {}, userData: g.userData, morphTargetsRelative: g.morphTargetsRelative };
        for (const [k, a] of Object.entries(g.attributes)) out.attributes[k] = { itemSize: a.itemSize, values: vals(a) };
        out.index = g.index ? Array.from(g.index.array) : null;
        out.morphAttributes = {};
        for (const [k, list] of Object.entries(g.morphAttributes || {})) out.morphAttributes[k] = list.map(vals);
        if (g.boundingBox) out.boundingBox = [g.boundingBox.min.toArray(), g.boundingBox.max.toArray()];
        if (g.boundingSphere) out.boundingSphere = [g.boundingSphere.center.toArray(), g.boundingSphere.radius];
        return out;
    };
    const object = (o) => {
        const out = {
            type: o.isInstancedMesh ? 'InstancedMesh' : o.type, name: o.name, userData: o.userData,
            position: o.position.toArray(), quaternion: o.quaternion.toArray(), scale: o.scale.toArray(),
        };
        if (o.geometry) out.geometry = geometry(o.geometry);
        if (o.material) out.material = Array.isArray(o.material) ? o.material.map(material) : material(o.material);
        if (o.morphTargetInfluences) out.morphTargetInfluences = [...o.morphTargetInfluences];
        if (o.morphTargetDictionary) out.morphTargetDictionary = o.morphTargetDictionary;
        if (o.isSkinnedMesh) {
            out.bindMode = o.bindMode;
            out.bindMatrix = [...o.bindMatrix.elements];
            out.skeleton = o.skeleton && { bones: o.skeleton.bones.map((b) => b.name), boneInverses: o.skeleton.boneInverses.map((m) => [...m.elements]) };
        }
        if (o.isInstancedMesh) {
            out.count = o.count;
            out.instanceMatrix = Array.from(o.instanceMatrix.array.subarray(0, o.count * 16));
            out.instanceColor = o.instanceColor ? vals(o.instanceColor) : null;
        }
        if (o.isPerspectiveCamera) Object.assign(out, { fov: o.fov, aspect: o.aspect, near: o.near, far: o.far });
        if (o.isOrthographicCamera) Object.assign(out, { left: o.left, right: o.right, top: o.top, bottom: o.bottom, near: o.near, far: o.far });
        if (o.isLight) {
            Object.assign(out, { color: [o.color.r, o.color.g, o.color.b], intensity: o.intensity });
            for (const k of ['distance', 'decay', 'angle', 'penumbra']) if (o[k] !== undefined) out[k] = o[k];
            if (o.target) out.target = { child: o.children.indexOf(o.target), position: o.target.position.toArray() };
        }
        out.children = o.children.map(object);
        return out;
    };
    const track = (t) => ({
        name: trackName(t.name), type: t.ValueTypeName, times: Array.from(t.times), values: Array.from(t.values),
        interpolation: t.interpolation ?? t.getInterpolation?.(),
        cubic: !!t.createInterpolant?.isInterpolantFactoryMethodGLTFCubicSpline,
    });
    // A track of an unnamed object is named by its uuid, which no two loads
    // share: name it by the object's place instead.
    const uuidPath = new Map();
    gltf.scenes.forEach((s, i) => {
        const walk = (o, p) => { uuidPath.set(o.uuid, p); o.children.forEach((c, k) => walk(c, p + '/' + k)); };
        walk(s, 'scene' + i);
    });
    const trackName = (n) => {
        const dot = n.lastIndexOf('.');
        const target = n.slice(0, dot);
        return uuidPath.has(target) ? `<${uuidPath.get(target)}>${n.slice(dot)}` : n;
    };
    const scenes = gltf.scenes.map(object);
    // With several scenes, three.js's `parser.associations` depends on which
    // scene's promise settles first (each `loadScene` filters the map down
    // to its own subtree); only a single-scene file has a defined answer.
    const assoc = [];
    if (gltf.scenes.length > 1) assoc.push(['several scenes', 'not compared']);
    else {
    const walk = (o, p) => {
        const a = gltf.parser.associations;
        assoc.push([p, a.has(o) ? (a.get(o) === undefined ? 'undefined' : { ...a.get(o) }) : 'absent']);
        o.children.forEach((c, i) => walk(c, p + '/' + i));
    };
    gltf.scenes.forEach((s, i) => walk(s, 'scene' + i));
    for (const [m] of ids.material) assoc.push(['material' + ids.material.get(m), gltf.parser.associations.has(m) ? (gltf.parser.associations.get(m) ?? 'undefined') : 'absent']);
    for (const [t] of ids.texture) assoc.push(['texture' + ids.texture.get(t), gltf.parser.associations.has(t) ? (gltf.parser.associations.get(t) ?? 'undefined') : 'absent']);
    }
    // `parser.getDependency` for every definition (what no scene used too):
    // three.js loads those lazily, and so does the shim.
    const deps = {};
    const json = gltf.parser.json;
    const skinWeightAccessors = new Set((json.meshes || []).flatMap((m) => m.primitives.map((p) => p.attributes?.WEIGHTS_0)));
    for (const [type, list] of [['material', 'materials'], ['texture', 'textures'], ['mesh', 'meshes'], ['node', 'nodes'],
        ['camera', 'cameras'], ['skin', 'skins'], ['animation', 'animations'], ['accessor', 'accessors']]) {
        deps[type] = [];
        for (let i = 0; i < (json[list] || []).length; i++) {
            let d;
            try { d = await gltf.parser.getDependency(type, i); } catch (e) { deps[type].push('error'); continue; }
            if (d == null) deps[type].push(null);
            else if (type === 'material') deps[type].push(material(d));
            else if (type === 'texture') deps[type].push(texture(d));
            else if (type === 'skin') deps[type].push({ bones: d.bones.map((b) => b.name), boneInverses: d.boneInverses.map((m) => [...m.elements]) });
            else if (type === 'animation') deps[type].push({ name: d.name, tracks: d.tracks.length });
            // three.js hands back the very attribute a SkinnedMesh normalized
            // in place; the shim reads the accessor as the file has it.
            else if (type === 'accessor') deps[type].push(skinWeightAccessors.has(i) ? 'skin weights' : { itemSize: d.itemSize, values: vals(d) });
            else deps[type].push({ name: d.name, type: d.isInstancedMesh ? 'InstancedMesh' : d.type, children: d.children.length });
        }
    }
    return {
        deps,
        scene: gltf.scenes.indexOf(gltf.scene),
        scenes,
        animations: gltf.animations.map((a) => ({ name: a.name, duration: a.duration, tracks: a.tracks.map(track) })),
        cameras: gltf.cameras.map((c) => c && { type: c.type, name: c.name, inScene: gltf.scenes.some((s) => { let f = false; s.traverse((o) => { f ||= o === c; }); return f; }) }),
        asset: gltf.asset,
        userData: gltf.userData,
        associations: assoc,
    };
}

// --------------------------------------------------------------------- compare
function compare(a, b, at, diffs) {
    if (diffs.length > 40) return;
    if (typeof a === 'number' && typeof b === 'number') {
        const tol = 1e-5 * Math.max(1, Math.abs(a), Math.abs(b));
        if (!(a === b || Math.abs(a - b) <= tol || (Number.isNaN(a) && Number.isNaN(b)))) diffs.push(`${at}: three ${a} vs shim ${b}`);
        return;
    }
    if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) { diffs.push(`${at}: length three ${a.length} vs shim ${b.length}`); return; }
        a.forEach((x, i) => compare(x, b[i], `${at}[${i}]`, diffs));
        return;
    }
    if (a && b && typeof a === 'object' && typeof b === 'object') {
        for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
            if (!(k in a)) { diffs.push(`${at}.${k}: missing in three (shim ${JSON.stringify(b[k])?.slice(0, 80)})`); continue; }
            if (!(k in b)) { diffs.push(`${at}.${k}: missing in shim (three ${JSON.stringify(a[k])?.slice(0, 80)})`); continue; }
            compare(a[k], b[k], `${at}.${k}`, diffs);
        }
        return;
    }
    if (a !== b) diffs.push(`${at}: three ${JSON.stringify(a)?.slice(0, 80)} vs shim ${JSON.stringify(b)?.slice(0, 80)}`);
}

let failed = 0;
for (const file of files) {
    const bytes = fs.readFileSync(file);
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    // A file: URL — three.js's FileLoader makes a `Request`, which needs one.
    const dir = pathToFileURL(path.dirname(file)).href + '/';
    const label = path.relative(root, file);
    let three, ours, threeGltf;
    try {
        const tl = new ThreeGLTFLoader().setMeshoptDecoder(MeshoptDecoder);
        threeGltf = await tl.parseAsync(buf, dir);
    } catch (e) {
        console.log(`SKIP ${label}: three.js cannot load it (${e.message})`);
        continue;
    }
    three = await dump(threeGltf);
    try {
        ours = await dump(await new S.GLTFLoader().parseAsync(buf, dir));
    } catch (e) {
        failed++;
        console.log(`FAIL ${label}: shim threw ${e?.stack ?? e}`);
        continue;
    }
    const diffs = [];
    compare(three, ours, '', diffs);
    if (diffs.length) {
        failed++;
        console.log(`FAIL ${label}: ${diffs.length} differences`);
        for (const d of diffs.slice(0, 40)) console.log('   ' + d);
    } else {
        console.log(`ok   ${label}`);
    }
}
console.log(failed ? `${failed} file(s) differ` : `all ${files.length} files match three.js r165`);
process.exit(failed ? 1 : 0);
