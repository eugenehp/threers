// Parity scene registry + API coverage inventory.
// CORE_SCENES: hand-authored baseline suite (91 scenes).
// GENERATED_SCENES: machine-generated edge-case pairs from generate-scenes.js.

import { GENERATED_SCENES } from './generate-scenes.js';

/** @type {string[]} */
export const CORE_SCENES = [
    'cube', 'sphere-point', 'basic-color', 'multi-mesh', 'plane-rot', 'ortho-cube',
    'spot-light', 'hemi-light', 'multi-light', 'metal-sphere', 'nested', 'phong',
    'textured-plane', 'vertex-colors', 'lines', 'points', 'torusknot', 'lambert',
    'normal-mat', 'depth-overlap', 'cylinder', 'cone', 'torus', 'icosahedron',
    'transparent', 'wireframe', 'depth-mat', 'tetra-octa', 'api-smoke', 'shadow',
    'spot-shadow', 'multi-shadow', 'shadow-cutout', 'lit-maps', 'envmap', 'pbr-env', 'physical-specular', 'physical-sheen', 'physical-iridescence', 'physical-anisotropy', 'physical-env', 'rect-area', 'transmission', 'toon', 'toon-gradient', 'physical', 'many-meshes', 'camera-tilt',
    'emissive', 'background-only', 'colored-dir', 'mesh-scale', 'normal-map',
    'three-points', 'rotation-axes', 'instanced', 'sprite', 'lathe', 'gltf-parse', 'gltf-features',
    'anim-mixer', 'edges-geom', 'shape-geom', 'canvas-tex', 'buffergeo-load',
    'object-load', 'group-xform', 'nested-group', 'quaternion', 'lineloop', 'extrude',
    'visible', 'tex-tint', 'geo-translate', 'dirlight-target', 'pointlight-decay',
    'mixed-mats', 'two-pass', 'matcap', 'layers', 'fog', 'dashed', 'merged-geom',
    'raycast', 'convex', 'morph', 'multi-mat', 'rt', 'skinned', 'postfx', 'cubecam',
    'sampler', 'env-cube', 'readback', 'postfx-copy', 'bloom', 'sky', 'skinned-morph',
    'batched', 'reflector', 'outline', 'ssao', 'ssr', 'pmrem', 'text-geom', 'shape-holes',
    'rounded-box', 'teapot', 'decal', 'interleaved',
];

/** Scene slugs from generated edge-case pairs. */
export const GENERATED_SCENE_SLUGS = GENERATED_SCENES.map(s => s.slug);

/** Full parity run list: core + generated (deduped). */
export const ALL_SCENES = [...new Set([...CORE_SCENES, ...GENERATED_SCENE_SLUGS])];

/**
 * Maps scene slug → primary three.js APIs exercised (for coverage reporting).
 * @type {Record<string, string[]>}
 */
export const SCENE_API_MAP = {
    parametric: ['ParametricGeometry'],
    circle: ['CircleGeometry'],
    ring: ['RingGeometry'],
    capsule: ['CapsuleGeometry'],
    'rounded-box': ['RoundedBoxGeometry'],
    teapot: ['TeapotGeometry'],
    decal: ['DecalGeometry'],
    interleaved: ['InterleavedBuffer', 'InterleavedBufferAttribute'],
    dodecahedron: ['DodecahedronGeometry'],
    'fog-exp2': ['FogExp2'],
    doubleside: ['DoubleSide'],
    backside: ['BackSide'],
    alphatest: ['alphaTest'],
    'wireframe-geom': ['WireframeGeometry'],
    'tube-curve': ['TubeGeometry', 'CatmullRomCurve3'],
    'extrude-bevel': ['ExtrudeGeometry', 'bevelEnabled'],
    'text-geom': ['TextGeometry', 'FontLoader', 'Font'],
    'shape-holes': ['Shape', 'Path', 'ShapeGeometry', 'ExtrudeGeometry', 'extrudePath'],
    'render-order': ['renderOrder'],
    'axes-grid': ['AxesHelper', 'GridHelper'],
    'quat-track': ['QuaternionKeyframeTrack', 'AnimationMixer'],
    'color-track': ['ColorKeyframeTrack', 'AnimationMixer'],
    'shadow-mat': ['ShadowMaterial', 'castShadow', 'receiveShadow'],
    'shadow-cutout': ['alphaTest', 'castShadow', 'DataTexture', 'transparent'],
    'lit-maps': ['MeshLambertMaterial.map', 'MeshPhongMaterial.aoMap', 'MeshToonMaterial.emissiveMap', 'Texture.colorSpace'],
    envmap: ['CubeTexture', 'MeshBasicMaterial.envMap', 'CubeRefractionMapping', 'envMapRotation'],
    'physical-specular': ['MeshPhysicalMaterial.ior', 'MeshPhysicalMaterial.specularIntensity', 'MeshPhysicalMaterial.specularColor'],
    'physical-sheen': ['MeshPhysicalMaterial.sheen', 'MeshPhysicalMaterial.sheenColor', 'MeshPhysicalMaterial.sheenRoughness'],
    'physical-iridescence': ['MeshPhysicalMaterial.iridescence', 'MeshPhysicalMaterial.iridescenceIOR', 'MeshPhysicalMaterial.iridescenceThicknessRange'],
    'physical-anisotropy': ['MeshPhysicalMaterial.anisotropy', 'MeshPhysicalMaterial.anisotropyRotation'],
    'physical-env': ['MeshPhysicalMaterial.sheen', 'MeshPhysicalMaterial.iridescence', 'MeshPhysicalMaterial.anisotropy', 'MeshPhysicalMaterial.clearcoat', 'Scene.environment'],
    'rect-area': ['RectAreaLight', 'RectAreaLightUniformsLib'],
    'toon-gradient': ['MeshToonMaterial.gradientMap', 'DataTexture', 'NearestFilter'],
    transmission: ['MeshPhysicalMaterial.transmission', 'MeshPhysicalMaterial.thickness', 'MeshPhysicalMaterial.attenuationColor', 'MeshPhysicalMaterial.dispersion'],
    'pbr-env': ['Scene.environment', 'Scene.environmentIntensity', 'Scene.environmentRotation', 'MeshStandardMaterial.envMap', 'envMapIntensity'],
    fxaa: ['FXAAShader', 'WebGLRenderTarget'],
    film: ['FilmPass'],
    dotscreen: ['DotScreenPass'],
    'halftone-postfx': ['HalftonePass'],
    glitch: ['GlitchPass'],
};

/** Post-fx scenes using simplified WGSL approximations (not pixel-perfect vs three.js passes). */
export const APPROXIMATE_SCENES = [];

/** Max allowed diff % for APPROXIMATE_SCENES (documented WGSL/GLSL drift). */
export const APPROXIMATE_THRESHOLD = 16;

/** Stub/no-op exports in threejs-shim.js (not pixel-tested). */
export const STUB_APIS = [];

/** APIs with parity scenes but known partial implementation. */
export const PARTIAL_APIS = [
    'RectAreaLight',
    // Core DigitalGlitch (RGB/displacement/bands) ~4% snow-free with GLSL disp bake; snow adds GLSL sin drift.
    'GlitchPass',
    'DRACOLoader', 'MeshoptDecoder',
];

export function coverageReport() {
    const tested = new Set(Object.values(SCENE_API_MAP).flat());
    return {
        coreScenes: CORE_SCENES.length,
        generatedScenes: GENERATED_SCENE_SLUGS.length,
        totalScenes: ALL_SCENES.length,
        apisWithScenes: tested.size,
        approximateScenes: APPROXIMATE_SCENES.length,
        stubApis: STUB_APIS.length,
        partialApis: PARTIAL_APIS.length,
    };
}
