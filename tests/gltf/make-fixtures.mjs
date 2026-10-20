#!/usr/bin/env node
// Writes the glTF fixtures the loader is tested on:
//
//   src/loaders/testdata/gltf/features.gltf   every feature, one file (data URIs)
//   src/loaders/testdata/gltf/features.glb    the same as a GLB (images in buffer views)
//   src/loaders/testdata/gltf/external.gltf   external .bin and .png by relative URI
//   src/loaders/testdata/gltf/external.bin
//   src/loaders/testdata/gltf/checker.png
//
// Run from anywhere: `node tests/gltf/make-fixtures.mjs`. The output is
// deterministic, so rerunning it leaves the files unchanged.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const outDir = path.join(root, 'src/loaders/testdata/gltf');
fs.mkdirSync(outDir, { recursive: true });

// ---------------------------------------------------------------- PNG writer
const crcTable = new Uint32Array(256).map((_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
});
function crc32(buf) {
    let c = 0xffffffff;
    for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
}
function png(width, height, pixel) {
    const raw = Buffer.alloc((width * 4 + 1) * height);
    for (let y = 0; y < height; y++) {
        raw[y * (width * 4 + 1)] = 0;
        for (let x = 0; x < width; x++) {
            const [r, g, b, a] = pixel(x, y);
            const o = y * (width * 4 + 1) + 1 + x * 4;
            raw[o] = r; raw[o + 1] = g; raw[o + 2] = b; raw[o + 3] = a;
        }
    }
    const chunk = (type, data) => {
        const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
        const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
        const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
        return Buffer.concat([len, td, crc]);
    };
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
    ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    return Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0)),
    ]);
}
const checker = png(4, 4, (x, y) => ((x + y) & 1 ? [255, 64, 32, 255] : [16, 200, 96, 128]));
const gradient = png(2, 3, (x, y) => [x * 200, y * 100, 50, 255]);
const normalPng = png(2, 2, () => [128, 128, 255, 255]);

// ------------------------------------------------------------- buffer builder
class Bin {
    constructor() { this.parts = []; this.length = 0; this.bufferViews = []; this.accessors = []; }
    align(n) {
        const pad = (n - (this.length % n)) % n;
        if (pad) { this.parts.push(Buffer.alloc(pad)); this.length += pad; }
    }
    view(bytes, extra = {}) {
        this.align(4);
        const v = { buffer: 0, byteOffset: this.length, byteLength: bytes.byteLength, ...extra };
        this.parts.push(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength));
        this.length += bytes.byteLength;
        this.bufferViews.push(v);
        return this.bufferViews.length - 1;
    }
    accessor(def) { this.accessors.push(def); return this.accessors.length - 1; }
    /// A view holding `array` and an accessor over all of it.
    add(array, type, extra = {}, viewExtra = {}) {
        const ct = array instanceof Float32Array ? 5126 : array instanceof Uint32Array ? 5125
            : array instanceof Uint16Array ? 5123 : array instanceof Int16Array ? 5122
            : array instanceof Uint8Array ? 5121 : 5120;
        const size = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }[type];
        const view = this.view(array, viewExtra);
        const acc = { bufferView: view, componentType: ct, count: array.length / size, type, ...extra };
        if (extra.minmax) {
            delete acc.minmax;
            const min = new Array(size).fill(Infinity), max = new Array(size).fill(-Infinity);
            for (let i = 0; i < array.length; i++) {
                min[i % size] = Math.min(min[i % size], array[i]);
                max[i % size] = Math.max(max[i % size], array[i]);
            }
            acc.min = min; acc.max = max;
        }
        return this.accessor(acc);
    }
    bytes() { this.align(4); return Buffer.concat(this.parts); }
}

const f32 = (...v) => new Float32Array(v.flat());
const bin = new Bin();

// A unit quad (two triangles), with every attribute.
const quadPos = bin.add(f32([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]), 'VEC3', { minmax: true });
const quadNrm = bin.add(f32([0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]), 'VEC3');
const quadTan = bin.add(f32([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1], [1, 0, 0, 1]), 'VEC4');
const quadUv0 = bin.add(f32([0, 1], [1, 1], [1, 0], [0, 0]), 'VEC2');
const quadUv1 = bin.add(f32([0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]), 'VEC2');
const quadIdx = bin.add(new Uint16Array([0, 1, 2, 0, 2, 3]), 'SCALAR');
// A triangle, positions only, indexed with UNSIGNED_BYTE (padded view).
const triPos = bin.add(f32([0, 0, 0], [1, 0, 0], [0, 1, 0]), 'VEC3', { minmax: true });
const triIdx8 = bin.add(new Uint8Array([0, 1, 2, 0]), 'SCALAR', { count: 3 });
bin.accessors[triIdx8].count = 3;
// Interleaved position + normal (byteStride 24).
const inter = new Float32Array([0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 1, 0]);
const interView = bin.view(inter, { byteStride: 24, target: 34962 });
const interPos = bin.accessor({ bufferView: interView, byteOffset: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [2, 0, 2] });
const interNrm = bin.accessor({ bufferView: interView, byteOffset: 12, componentType: 5126, count: 3, type: 'VEC3' });
// Quantized: SHORT positions (not normalized), BYTE normals (normalized),
// UNSIGNED_SHORT uvs (normalized), UNSIGNED_BYTE colours (normalized VEC4).
const qPos = bin.add(new Int16Array([0, 0, 0, 0, 100, 0, 0, 0, 50, 0, 100, 0, 0, 0, 0, 0]), 'VEC3', {}, { byteStride: 8 });
bin.accessors[qPos].count = 4;
// (stride 8: three shorts + padding per vertex, as KHR_mesh_quantization lays them out)
Object.assign(bin.accessors[qPos], { min: [0, 0, 0], max: [100, 100, 0] });
const qNrm = bin.add(new Int8Array([0, 0, 127, 0, 0, 0, 127, 0, 0, 0, -128, 0, 0, 127, 0, 0]), 'VEC3', { normalized: true }, { byteStride: 4 });
bin.accessors[qNrm].count = 4;
const qUv = bin.add(new Uint16Array([0, 0, 65535, 0, 65535, 65535, 0, 32768]), 'VEC2', { normalized: true });
const qCol = bin.add(new Uint8Array([255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 255, 255, 255, 255, 0]), 'VEC4', { normalized: true });
const qIdx = bin.add(new Uint32Array([0, 1, 2, 2, 1, 3]), 'SCALAR');
// Morph targets: dense POSITION and NORMAL; a sparse POSITION target with no
// buffer view; a target with NORMAL only.
const morphPos0 = bin.add(f32([0, 0, 0.5], [0, 0, 0.5], [0, 0, 0], [0, 0, 0]), 'VEC3', { minmax: true });
const morphNrm0 = bin.add(f32([0, 0.1, 0], [0, 0.1, 0], [0, 0, 0], [0, 0, 0]), 'VEC3');
const sparseIdx = bin.view(new Uint16Array([2, 0]));
const sparseVal = bin.view(f32([0.25, 0.5, 0]));
const morphSparse = bin.accessor({
    componentType: 5126, count: 4, type: 'VEC3', min: [0, 0, 0], max: [0.25, 0.5, 0],
    sparse: { count: 1, indices: { bufferView: sparseIdx, componentType: 5123 }, values: { bufferView: sparseVal } },
});
const morphNrm2 = bin.add(f32([0, 0, -0.2], [0, 0, -0.2], [0, 0, -0.2], [0, 0, -0.2]), 'VEC3');
// A sparse accessor over a buffer view (values substituted into a copy).
const sparseBaseIdx = bin.view(new Uint8Array([1, 0, 0, 0]));
const sparseBaseVal = bin.view(f32([0.5, 0.5, 0.5]));
const sparseOver = bin.accessor({
    bufferView: bin.view(f32([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0])), componentType: 5126, count: 4, type: 'VEC3',
    min: [0, 0, 0], max: [1, 1, 0],
    sparse: { count: 1, indices: { bufferView: sparseBaseIdx, componentType: 5121 }, values: { bufferView: sparseBaseVal } },
});
// Skin: two joints, 4 vertices, UNSIGNED_BYTE joints, normalized UNSIGNED_BYTE
// weights that do not sum to one (normalizeSkinWeights fixes them).
const skinJoints = bin.add(new Uint8Array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), 'VEC4');
const skinWeights = bin.add(new Uint8Array([200, 55, 0, 0, 100, 100, 0, 0, 255, 0, 0, 0, 128, 0, 0, 0]), 'VEC4', { normalized: true });
const ibm = bin.add(f32([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.5, -1, 0, 1]), 'MAT4');
// Instancing: three instances.
const instT = bin.add(f32([0, 0, 0], [2, 0, 0], [4, 1, 0]), 'VEC3');
const instR = bin.add(new Int16Array([0, 0, 0, 32767, 0, 23170, 0, 23170, 0, 0, 32767, 0]), 'VEC4', { normalized: true });
const instS = bin.add(f32([1, 1, 1], [0.5, 0.5, 0.5], [1, 2, 1]), 'VEC3');
const instC = bin.add(new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]), 'VEC3', { normalized: true });
const instId = bin.add(f32([7, 8, 9]), 'SCALAR');
// Primitive modes: 6 points.
const linePos = bin.add(f32([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0]), 'VEC3', { minmax: true });
const stripIdx = bin.add(new Uint16Array([0, 1, 3, 2, 4, 5]), 'SCALAR');
// Animations.
const tIn = bin.add(f32([0, 0.5, 1]), 'SCALAR', { minmax: true });
const tTrans = bin.add(f32([0, 0, 0], [0, 1, 0], [1, 1, 0]), 'VEC3');
const tRot = bin.add(f32([0, 0, 0, 1], [0, 0.7071068, 0, 0.7071068], [0, 1, 0, 0]), 'VEC4');
const tScaleCubic = bin.add(f32(
    [0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5],
    [0.1, 0.1, 0.1], [2, 2, 2], [0.2, 0.2, 0.2],
    [0, 0, 0], [1, 1, 1], [0, 0, 0],
), 'VEC3');
const tWeights = bin.add(f32([0, 1, 0], [1, 0, 0.5], [0, 0, 1]), 'SCALAR');
const tWeightsU8 = bin.add(new Uint8Array([0, 255]), 'SCALAR', { normalized: true });
const tIn2 = bin.add(f32([0, 2]), 'SCALAR', { minmax: true });
// Images: one PNG in a buffer view (the GLB keeps all three there).
const pngView = bin.view(new Uint8Array(gradient));

const dataUri = (mime, bytes) => `data:${mime};base64,${Buffer.from(bytes).toString('base64')}`;

// --------------------------------------------------------------- the glTF
const quadAttrs = { POSITION: quadPos, NORMAL: quadNrm, TANGENT: quadTan, TEXCOORD_0: quadUv0, TEXCOORD_1: quadUv1 };
const gltf = {
    asset: { version: '2.0', generator: 'threers tests/gltf/make-fixtures.mjs', copyright: 'CC0' },
    extensionsUsed: [
        'KHR_texture_transform', 'KHR_materials_unlit', 'KHR_lights_punctual', 'KHR_mesh_quantization',
        'KHR_materials_clearcoat', 'KHR_materials_sheen', 'KHR_materials_transmission', 'KHR_materials_volume',
        'KHR_materials_ior', 'KHR_materials_specular', 'KHR_materials_iridescence', 'KHR_materials_anisotropy',
        'KHR_materials_dispersion', 'KHR_materials_emissive_strength', 'EXT_materials_bump', 'EXT_mesh_gpu_instancing',
        'EXT_custom_thing',
    ],
    extensionsRequired: ['KHR_mesh_quantization'],
    extras: { author: 'fixture', nested: { a: [1, 2] } },
    extensions: {
        KHR_lights_punctual: {
            lights: [
                { type: 'directional', color: [1, 0.9, 0.8], intensity: 3, name: 'Sun' },
                { type: 'point', color: [0.2, 0.4, 1], intensity: 20, range: 12 },
                { type: 'spot', intensity: 5, spot: { innerConeAngle: 0.2, outerConeAngle: 0.6 }, name: 'Spot Light', extras: { kind: 'spot' } },
            ],
        },
        EXT_custom_thing: { root: true },
    },
    scene: 0,
    scenes: [
        { name: 'Main Scene', nodes: [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25], extras: { s: 1 } },
        { name: 'Other', nodes: [18] },
    ],
    nodes: [
        /* 0 */ { name: 'Parent', translation: [1, 2, 3], rotation: [0, 0.3826834, 0, 0.9238795], scale: [2, 2, 2], children: [1, 2], extras: { tag: 'p' } },
        /* 1 */ { name: 'Child', matrix: [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 5, 6, 7, 1], mesh: 0 },
        /* 2 */ { name: 'Dup', mesh: 1, extensions: { EXT_custom_thing: { n: 2 } } },
        /* 3 */ { name: 'Dup', mesh: 2 },
        /* 4 */ { name: 'Shared A', mesh: 3, translation: [0, 1, 0] },
        /* 5 */ { name: 'Shared B', mesh: 3, translation: [0, -1, 0] },
        /* 6 */ { mesh: 4 },
        /* 7 */ { name: 'Cam.Persp', camera: 0, translation: [0, 0, 10] },
        /* 8 */ { name: 'Cam Ortho', camera: 1 },
        /* 9 */ { name: 'CamShared', camera: 0, children: [10] },
        /* 10 */ { name: 'Lit', extensions: { KHR_lights_punctual: { light: 0 } }, rotation: [-0.3826834, 0, 0, 0.9238795] },
        /* 11 */ { extensions: { KHR_lights_punctual: { light: 1 } }, translation: [3, 3, 3] },
        /* 12 */ { name: 'SpotA', extensions: { KHR_lights_punctual: { light: 2 } } },
        /* 13 */ { name: 'SpotB', extensions: { KHR_lights_punctual: { light: 2 } }, mesh: 2 },
        /* 14 */ { name: 'Skinned', mesh: 5, skin: 0 },
        /* 15 */ { name: 'Morph', mesh: 6, weights: [0.1, 0.2, 0.3] },
        /* 16 */ { name: 'Instanced', mesh: 7, extensions: { EXT_mesh_gpu_instancing: { attributes: { TRANSLATION: instT, ROTATION: instR, SCALE: instS, _COLOR_0: instC, _ID: instId } } } },
        /* 17 */ { name: 'Modes', mesh: 8 },
        /* 18 */ { name: 'Elsewhere', mesh: 9 },
        /* 19 */ { name: 'Bone Root', translation: [0, 0, 0], children: [20] },
        /* 20 */ { name: 'Bone Tip', translation: [0.5, 1, 0] },
        /* 21 */ { name: 'Animated Only', translation: [9, 9, 9] },
        /* 22 */ { name: 'Skeleton', children: [19] },
        /* 23 */ { name: 'Quantized', mesh: 10 },
        /* 24 */ { name: 'Multi Morph', mesh: 11 },
        /* 25 */ { name: 'Physical', mesh: 12 },
    ],
    meshes: [
        /* 0 */ { name: 'Quad', primitives: [{ attributes: quadAttrs, indices: quadIdx, material: 0 }], extras: { m: 0 } },
        /* 1 */ { name: 'Dup', primitives: [{ attributes: { POSITION: triPos }, indices: triIdx8, material: 1 }] },
        /* 2 */ { name: 'Multi', primitives: [
            { attributes: quadAttrs, indices: quadIdx, material: 2 },
            { attributes: { POSITION: interPos, NORMAL: interNrm } },
            { attributes: { POSITION: triPos }, material: 3, extensions: { EXT_custom_thing: { prim: true } } },
        ], extensions: { EXT_custom_thing: { mesh: true } } },
        /* 3 */ { name: 'Shared', primitives: [{ attributes: { POSITION: triPos, NORMAL: interNrm }, material: 4 }] },
        /* 4 */ { primitives: [{ attributes: { POSITION: sparseOver }, material: 1 }] },
        /* 5 */ { name: 'SkinMesh', primitives: [{ attributes: { POSITION: quadPos, NORMAL: quadNrm, JOINTS_0: skinJoints, WEIGHTS_0: skinWeights }, indices: quadIdx, material: 4 }] },
        /* 6 */ { name: 'MorphMesh', primitives: [{ attributes: { POSITION: quadPos, NORMAL: quadNrm }, indices: quadIdx, material: 4,
            targets: [{ POSITION: morphPos0, NORMAL: morphNrm0 }, { POSITION: morphSparse }, { NORMAL: morphNrm2 }] }],
            weights: [0.5, 0.25, 0], extras: { targetNames: ['bulge', 'sparse', 'tilt'] } },
        /* 7 */ { name: 'InstMesh', primitives: [{ attributes: { POSITION: triPos }, material: 4 }] },
        /* 8 */ { name: 'Modes', primitives: [
            { attributes: { POSITION: linePos }, mode: 0, material: 1 },
            { attributes: { POSITION: linePos }, mode: 1, material: 1 },
            { attributes: { POSITION: linePos }, mode: 2 },
            { attributes: { POSITION: linePos }, mode: 3, material: 5 },
            { attributes: { POSITION: linePos }, indices: stripIdx, mode: 5, material: 4 },
            { attributes: { POSITION: linePos }, mode: 6, material: 4 },
        ] },
        /* 9 */ { name: 'Elsewhere', primitives: [{ attributes: { POSITION: triPos } }] },
        /* 10 */ { name: 'Quantized', primitives: [{ attributes: { POSITION: qPos, NORMAL: qNrm, TEXCOORD_0: qUv, COLOR_0: qCol }, indices: qIdx, material: 0 }] },
        /* 11 */ { name: 'MultiMorph', weights: [1], primitives: [
            { attributes: { POSITION: quadPos }, indices: quadIdx, targets: [{ POSITION: morphPos0 }] },
            { attributes: { POSITION: quadPos }, targets: [{ POSITION: morphSparse }] },
        ] },
        /* 12 */ { name: 'PhysicalMesh', primitives: [
            { attributes: quadAttrs, indices: quadIdx, material: 6 },
            { attributes: { POSITION: quadPos, NORMAL: quadNrm, TEXCOORD_0: quadUv0 }, indices: quadIdx, material: 7 },
            { attributes: { POSITION: quadPos, NORMAL: quadNrm, TEXCOORD_0: quadUv0 }, indices: quadIdx, material: 8 },
        ] },
    ],
    materials: [
        /* 0 */ {
            name: 'Textured', doubleSided: true, alphaMode: 'MASK', alphaCutoff: 0.3,
            pbrMetallicRoughness: {
                baseColorFactor: [1, 0.5, 0.25, 0.75], metallicFactor: 0.25, roughnessFactor: 0.75,
                baseColorTexture: { index: 0, extensions: { KHR_texture_transform: { offset: [0.5, 0.25], rotation: 0.3, scale: [2, 3] } } },
                metallicRoughnessTexture: { index: 1 },
            },
            normalTexture: { index: 2, scale: 0.5 },
            occlusionTexture: { index: 1, texCoord: 1, strength: 0.6 },
            emissiveTexture: { index: 0 },
            emissiveFactor: [0.1, 0.2, 0.3],
            extras: { mat: 'textured' },
            extensions: { EXT_custom_thing: { mat: true } },
        },
        /* 1 */ { name: 'Blend', alphaMode: 'BLEND', pbrMetallicRoughness: { baseColorFactor: [0.2, 0.3, 0.4, 0.5] } },
        /* 2 */ { name: 'Unlit', extensions: { KHR_materials_unlit: {} }, pbrMetallicRoughness: { baseColorFactor: [0.9, 0.1, 0.1, 1], baseColorTexture: { index: 3 } }, normalTexture: { index: 2 } },
        /* 3 */ { pbrMetallicRoughness: { metallicFactor: 0 } },
        /* 4 */ { name: 'Plain', pbrMetallicRoughness: { baseColorFactor: [0.5, 0.5, 0.5, 1] } },
        /* 5 */ { name: 'Lines', emissiveFactor: [1, 1, 0], pbrMetallicRoughness: { baseColorTexture: { index: 4 } } },
        /* 6 */ {
            name: 'Clearcoat & co',
            pbrMetallicRoughness: { baseColorTexture: { index: 4, texCoord: 1 } },
            normalTexture: { index: 2 },
            extensions: {
                KHR_materials_clearcoat: { clearcoatFactor: 0.8, clearcoatRoughnessFactor: 0.2, clearcoatTexture: { index: 1 }, clearcoatNormalTexture: { index: 2, scale: 0.7 } },
                KHR_materials_sheen: { sheenColorFactor: [0.3, 0.2, 0.1], sheenRoughnessFactor: 0.4, sheenColorTexture: { index: 0 } },
                KHR_materials_emissive_strength: { emissiveStrength: 4 },
                KHR_materials_ior: { ior: 1.8 },
                KHR_materials_specular: { specularFactor: 0.6, specularColorFactor: [1, 0.5, 0.5], specularColorTexture: { index: 3 } },
            },
        },
        /* 7 */ {
            name: 'Glass',
            alphaMode: 'BLEND',
            extensions: {
                KHR_materials_transmission: { transmissionFactor: 0.9, transmissionTexture: { index: 1 } },
                KHR_materials_volume: { thicknessFactor: 0.5, attenuationDistance: 2, attenuationColor: [0.9, 0.8, 0.7] },
                KHR_materials_dispersion: { dispersion: 0.3 },
                KHR_materials_iridescence: { iridescenceFactor: 1, iridescenceIor: 1.4, iridescenceThicknessMaximum: 500, iridescenceThicknessTexture: { index: 1 } },
            },
        },
        /* 8 */ {
            name: 'Brushed',
            extensions: {
                KHR_materials_anisotropy: { anisotropyStrength: 0.6, anisotropyRotation: 1.2, anisotropyTexture: { index: 2 } },
                EXT_materials_bump: { bumpFactor: 0.05, bumpTexture: { index: 1 } },
                KHR_materials_volume: {},
            },
        },
    ],
    textures: [
        /* 0 */ { source: 0, sampler: 0, name: 'Checker' },
        /* 1 */ { source: 1, sampler: 1 },
        /* 2 */ { source: 2 },
        /* 3 */ { source: 0, sampler: 2 },
        /* 4 */ { source: 0, sampler: 0 },
    ],
    samplers: [
        { magFilter: 9728, minFilter: 9984, wrapS: 33648, wrapT: 33071 },
        { magFilter: 9729, minFilter: 9987, wrapS: 10497, wrapT: 10497 },
        { minFilter: 9729 },
    ],
    images: [
        { uri: dataUri('image/png', checker), name: 'checker' },
        { bufferView: pngView, mimeType: 'image/png', extras: { from: 'bufferView' } },
        { uri: dataUri('image/png', normalPng) },
    ],
    cameras: [
        { type: 'perspective', name: 'Lens', perspective: { yfov: 0.8, aspectRatio: 1.5, znear: 0.05, zfar: 500 }, extras: { c: 0 } },
        { type: 'orthographic', orthographic: { xmag: 2, ymag: 1.5, znear: 0.1, zfar: 50 } },
        { type: 'perspective', name: 'Unused', perspective: { yfov: 1.0, znear: 0.1 } },
    ],
    skins: [{ joints: [19, 20], inverseBindMatrices: ibm, skeleton: 19, name: 'Rig' }],
    animations: [
        {
            name: 'Move',
            samplers: [
                { input: tIn, output: tTrans },
                { input: tIn, output: tRot, interpolation: 'STEP' },
                { input: tIn, output: tScaleCubic, interpolation: 'CUBICSPLINE' },
                { input: tIn, output: tWeights },
                { input: tIn2, output: tTrans, interpolation: 'LINEAR' },
            ],
            channels: [
                { sampler: 0, target: { node: 0, path: 'translation' } },
                { sampler: 1, target: { node: 20, path: 'rotation' } },
                { sampler: 2, target: { node: 4, path: 'scale' } },
                { sampler: 3, target: { node: 15, path: 'weights' } },
                { sampler: 0, target: { node: 21, path: 'translation' } },
                { sampler: 0, target: { path: 'translation' } },
            ],
        },
        {
            samplers: [{ input: tIn2, output: tWeightsU8 }],
            channels: [{ sampler: 0, target: { node: 24, path: 'weights' } }],
        },
    ],
};

const body = bin.bytes();
fs.writeFileSync(path.join(outDir, 'features.gltf'),
    JSON.stringify({ ...gltf, buffers: [{ uri: dataUri('application/octet-stream', body), byteLength: body.length }], bufferViews: bin.bufferViews, accessors: bin.accessors }, null, 1) + '\n');

// The GLB: every image in a buffer view, the body as the BIN chunk.
{
    const b2 = new Bin();
    b2.parts = [...bin.parts]; b2.length = bin.length; b2.bufferViews = bin.bufferViews.map((v) => ({ ...v })); b2.accessors = bin.accessors;
    const v0 = b2.view(new Uint8Array(checker));
    const v2 = b2.view(new Uint8Array(normalPng));
    const images = [
        { bufferView: v0, mimeType: 'image/png', name: 'checker' },
        gltf.images[1],
        { bufferView: v2, mimeType: 'image/png' },
    ];
    const bodyBytes = b2.bytes();
    const json = Buffer.from(JSON.stringify({ ...gltf, images, buffers: [{ byteLength: bodyBytes.length }], bufferViews: b2.bufferViews, accessors: b2.accessors }));
    const jsonPadded = Buffer.concat([json, Buffer.alloc((4 - json.length % 4) % 4, 0x20)]);
    const header = Buffer.alloc(12);
    header.write('glTF', 0, 'ascii'); header.writeUInt32LE(2, 4);
    header.writeUInt32LE(12 + 8 + jsonPadded.length + 8 + bodyBytes.length, 8);
    const ch = (type, data) => { const h = Buffer.alloc(8); h.writeUInt32LE(data.length, 0); h.writeUInt32LE(type, 4); return Buffer.concat([h, data]); };
    fs.writeFileSync(path.join(outDir, 'features.glb'), Buffer.concat([header, ch(0x4e4f534a, jsonPadded), ch(0x004e4942, bodyBytes)]));
}

// External resources by relative URI (a .bin and a .png beside the .gltf).
{
    const b = new Bin();
    const pos = b.add(f32([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]), 'VEC3', { minmax: true });
    const uv = b.add(f32([0, 1], [1, 1], [1, 0], [0, 0]), 'VEC2');
    const idx = b.add(new Uint16Array([0, 1, 2, 0, 2, 3]), 'SCALAR');
    const bytes = b.bytes();
    fs.writeFileSync(path.join(outDir, 'external.bin'), bytes);
    fs.writeFileSync(path.join(outDir, 'checker.png'), checker);
    fs.writeFileSync(path.join(outDir, 'external.gltf'), JSON.stringify({
        asset: { version: '2.0' },
        scenes: [{ nodes: [0] }],
        nodes: [{ name: 'Ext', mesh: 0 }],
        meshes: [{ primitives: [{ attributes: { POSITION: pos, TEXCOORD_0: uv }, indices: idx, material: 0 }] }],
        materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
        textures: [{ source: 0 }],
        images: [{ uri: 'checker.png' }],
        buffers: [{ uri: 'external.bin', byteLength: bytes.length }],
        bufferViews: b.bufferViews,
        accessors: b.accessors,
    }, null, 1) + '\n');
}
console.log(`wrote fixtures to ${path.relative(root, outDir)}`);
