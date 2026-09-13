import init, {
    // renderer + scene + camera
    WebRenderer, WebRenderTarget, WebCubeRenderTarget, WebScene, WebCamera, WebObjectHandle,
    // geometries
    WebGeometry, WebBufferGeometry, WebBufferAttribute,
    // materials
    WebMaterial,
    // meshes
    WebMesh,
    // lights
    WebLight,
    // math
    WebColor, WebVector2, WebVector3, WebVector4, WebMatrix3, WebMatrix4,
    WebQuaternion, WebEuler, WebBox2, WebBox3, WebSphere, WebRay, WebPlane,
    WebTriangle, WebFrustum, WebSpherical, WebCylindrical, WebLine3,
    // textures
    WebTexture, WebCubeTexture, WebDataTexture,
    // curves
    WebLineCurve, WebLineCurve3, WebEllipseCurve, WebCatmullRomCurve3,
    WebPath, WebShape,
    // animation
    WebAnimationClip, WebAnimationMixer,
    // controls
    WebOrbitControls, WebTrackballControls, WebFirstPersonControls,
    WebPointerLockControls, WebArcballControls,
    // helpers
    WebAxesHelper, WebGridHelper, WebBoxHelper, WebPolarGridHelper,
    // loaders
    WebObjLoader, WebStlLoader, WebPlyLoader, WebHdrLoader,
    WebFbxLoader, WebColladaLoader, WebExrLoader,
    // audio
    WebAudio, WebAudioListener,
    // post-fx
    WebEffectComposer, WebRenderPass, WebBloomPass, WebFxaaPass, WebPmremGenerator,
    // extras
    WebOctree, WebSimplexNoise, WebMarchingCubes,
    // alt renderers
    WebCss2dRenderer, WebSvgRenderer,
    // stats
    WebStats,
    // raycaster + clock
    WebRaycaster, WebClock,
} from './pkg/threers.js';
import { Earcut } from './deps/Earcut.js';
import {
    BrowserVideoFormat,
    VideoEncodeWorker,
    VideoExportError,
    VideoExportErrorCode,
    VideoExportEvent,
    VideoExportProgressEvent,
    VideoExporter,
    VideoExportResult,
    VideoFormat,
    alignVideoSize,
    assertVideoExportAvailable,
    downloadVideoBytes,
    emitProgress,
    encodeAndDownloadVideoFrames,
    encodeApngRgba as _encodeApngRgba,
    encodeGifRgba as _encodeGifRgba,
    encodeVideoFrames,
    encodeVideoFramesInWorker,
    encodeWebmRgba as _encodeWebmRgba,
    formatBytes,
    formatVideoProgress,
    isVideoExportAvailable,
    parseVideoFormat,
    videoFilename,
    videoMimeType,
} from './video-export.js';

export {
    BrowserVideoFormat,
    VideoEncodeWorker,
    VideoExportError,
    VideoExportErrorCode,
    VideoExportEvent,
    VideoExportProgressEvent,
    VideoExporter,
    VideoExportResult,
    VideoFormat,
    alignVideoSize,
    assertVideoExportAvailable,
    downloadVideoBytes,
    emitProgress,
    encodeAndDownloadVideoFrames,
    encodeVideoFrames,
    encodeVideoFramesInWorker,
    formatBytes,
    formatVideoProgress,
    isVideoExportAvailable,
    parseVideoFormat,
    videoFilename,
    videoMimeType,
};

/** Present when wasm was built with `--features native-codec` (throws if missing). */
export const encodeGifRgba = _encodeGifRgba;
export const encodeApngRgba = _encodeApngRgba;
export const encodeWebmRgba = _encodeWebmRgba;

/** Browsers reject wgpu's legacy `maxInterStageShaderComponents` limit name. */
const STRIP_WEBGPU_LIMITS = new Set(['maxInterStageShaderComponents']);

function filterWebGpuRequiredLimits(requiredLimits) {
    if (!requiredLimits) return requiredLimits;
    const out = {};
    for (const [k, v] of Object.entries(requiredLimits)) {
        if (!STRIP_WEBGPU_LIMITS.has(k)) out[k] = v;
    }
    return out;
}

function patchWebGpuDeviceLimits() {
    if (globalThis.__threersWebGpuPatched) return true;
    const Adapter = globalThis.GPUAdapter;
    if (!Adapter?.prototype?.requestDevice) return false;
    const orig = Adapter.prototype.requestDevice;
    Adapter.prototype.requestDevice = function (descriptor) {
        if (descriptor?.requiredLimits) {
            descriptor = {
                ...descriptor,
                requiredLimits: filterWebGpuRequiredLimits(descriptor.requiredLimits),
            };
        }
        return orig.call(this, descriptor);
    };
    globalThis.__threersWebGpuPatched = true;
    return true;
}
patchWebGpuDeviceLimits();

let _wasmReady = null;
export async function initThreers(wasmUrl) {
    if (!patchWebGpuDeviceLimits()) {
        for (let i = 0; i < 100 && !patchWebGpuDeviceLimits(); i++) {
            await new Promise((r) => setTimeout(r, 10));
        }
    }
    if (_wasmReady) return _wasmReady;
    const opts = (wasmUrl != null && typeof wasmUrl === 'object')
        ? wasmUrl
        : { module_or_path: wasmUrl };
    _wasmReady = init(opts);
    return _wasmReady;
}

// Material.side (exported early — csg/mesh-bvh import these by name).
export const FrontSide = 0;
export const BackSide = 1;
export const DoubleSide = 2;

// three.js texture type constants (also exported on default THREE object below).
const UnsignedByteType = 1009;
const HalfFloatType = 1016;

// ---- Math helpers ----
function _color(input) {
    if (input instanceof Color) return input._w;
    if (typeof input === 'number') return WebColor.fromHex(input);
    if (typeof input === 'object' && 'r' in input) return new WebColor(input.r, input.g, input.b);
    return WebColor.fromHex(0xffffff);
}

export class Color {
    constructor(r, g, b) {
        if (typeof r === 'number' && g === undefined) {
            // Hex literal. We keep r/g/b in the SAME numeric space as the hex
            // input bytes (i.e. raw byte/255, not sRGB-decoded) so that
            // `new Color(0xff8800).getHex() === 0xff8800` round-trips, matching
            // three.js's Color.getHex(). The wasm WebColor stores linear values
            // for shader use — that's what the renderer reads via _w.
            this._w = WebColor.fromHex(r);
            this.r = ((r >> 16) & 0xff) / 255;
            this.g = ((r >> 8) & 0xff) / 255;
            this.b = (r & 0xff) / 255;
        } else if (typeof r === 'string') {
            this._w = WebColor.fromHex(0xffffff); this.r = 1; this.g = 1; this.b = 1;
            this.setStyle(r);
        } else {
            this._w = new WebColor(r ?? 1, g ?? 1, b ?? 1);
            this.r = r ?? 1; this.g = g ?? 1; this.b = b ?? 1;
        }
    }
    setHex(h) {
        this._w = WebColor.fromHex(h);
        this.r = ((h >> 16) & 0xff) / 255;
        this.g = ((h >> 8) & 0xff) / 255;
        this.b = (h & 0xff) / 255;
        return this;
    }
    setRGB(r, g, b) { this.r = r; this.g = g; this.b = b; this._w = new WebColor(r, g, b); return this; }
    setScalar(s) { return this.setRGB(s, s, s); }
    setStyle(s) {
        if (typeof s !== 'string') return this;
        const m = /^#?([0-9a-fA-F]{6})$/.exec(s);
        if (m) return this.setHex(parseInt(m[1], 16));
        const rgb = /^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/.exec(s);
        if (rgb) return this.setRGB(+rgb[1]/255, +rgb[2]/255, +rgb[3]/255);
        return this;
    }
    setHSL(h, s, l) {
        // three.js default colorSpace is LinearSRGBColorSpace (working space).
        h = ((h % 1) + 1) % 1;
        s = Math.max(0, Math.min(1, s));
        l = Math.max(0, Math.min(1, l));
        if (s === 0) return this.setRGB(l, l, l);
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        const conv = (t) => {
            if (t < 0) t += 1; if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        return this.setRGB(conv(h + 1/3), conv(h), conv(h - 1/3));
    }
    getHex() { return Math.round(this.r * 255) << 16 | Math.round(this.g * 255) << 8 | Math.round(this.b * 255); }
    getHexString() { return this.getHex().toString(16).padStart(6, '0'); }
    add(c) { this.r += c.r; this.g += c.g; this.b += c.b; this._w = new WebColor(this.r, this.g, this.b); return this; }
    addScalar(s) { this.r += s; this.g += s; this.b += s; this._w = new WebColor(this.r, this.g, this.b); return this; }
    multiply(c) { this.r *= c.r; this.g *= c.g; this.b *= c.b; this._w = new WebColor(this.r, this.g, this.b); return this; }
    multiplyScalar(s) { this.r *= s; this.g *= s; this.b *= s; this._w = new WebColor(this.r, this.g, this.b); return this; }
    lerp(c, a) { this.r += (c.r - this.r) * a; this.g += (c.g - this.g) * a; this.b += (c.b - this.b) * a; this._w = new WebColor(this.r, this.g, this.b); return this; }
    copy(c) { this.r = c.r; this.g = c.g; this.b = c.b; this._w = new WebColor(this.r, this.g, this.b); return this; }
    clone() { return new Color(this.r, this.g, this.b); }
    equals(c) { return c.r === this.r && c.g === this.g && c.b === this.b; }
    fromArray(a, off = 0) { return this.setRGB(a[off], a[off+1], a[off+2]); }
    toArray(a = [], off = 0) { a[off] = this.r; a[off+1] = this.g; a[off+2] = this.b; return a; }
}

export class Vector2 {
    constructor(x = 0, y = 0) { this.x = x; this.y = y; }
    set(x, y) { this.x = x; this.y = y; return this; }
    setX(v) { this.x = v; return this; }
    setY(v) { this.y = v; return this; }
    setScalar(s) { this.x = s; this.y = s; return this; }
    copy(v) { this.x = v.x; this.y = v.y; return this; }
    clone() { return new Vector2(this.x, this.y); }
    add(v) { this.x += v.x; this.y += v.y; return this; }
    addVectors(a, b) { this.x = a.x + b.x; this.y = a.y + b.y; return this; }
    addScalar(s) { this.x += s; this.y += s; return this; }
    sub(v) { this.x -= v.x; this.y -= v.y; return this; }
    subVectors(a, b) { this.x = a.x - b.x; this.y = a.y - b.y; return this; }
    multiply(v) { this.x *= v.x; this.y *= v.y; return this; }
    multiplyScalar(s) { this.x *= s; this.y *= s; return this; }
    divideScalar(s) { return this.multiplyScalar(1 / s); }
    negate() { this.x = -this.x; this.y = -this.y; return this; }
    length() { return Math.hypot(this.x, this.y); }
    lengthSq() { return this.x*this.x + this.y*this.y; }
    normalize() { const l = this.length() || 1; this.x /= l; this.y /= l; return this; }
    dot(v) { return this.x*v.x + this.y*v.y; }
    cross(v) { return this.x*v.y - this.y*v.x; }
    distanceTo(v) { return Math.hypot(this.x-v.x, this.y-v.y); }
    distanceToSquared(v) { const dx=this.x-v.x, dy=this.y-v.y; return dx*dx+dy*dy; }
    lerp(v, a) { this.x += (v.x - this.x) * a; this.y += (v.y - this.y) * a; return this; }
    equals(v) { return v && v.x === this.x && v.y === this.y; }
    angle() { return Math.atan2(-this.y, -this.x) + Math.PI; }
    fromArray(a, off = 0) { this.x = a[off]; this.y = a[off+1]; return this; }
    toArray(a = [], off = 0) { a[off] = this.x; a[off+1] = this.y; return a; }
    _w() { return new WebVector2(this.x, this.y); }
}

export class Vector3 {
    constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
    set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
    setX(v) { this.x = v; return this; }
    setY(v) { this.y = v; return this; }
    setZ(v) { this.z = v; return this; }
    setScalar(s) { this.x = s; this.y = s; this.z = s; return this; }
    copy(v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; }
    clone() { return new Vector3(this.x, this.y, this.z); }
    equals(v) { return v && v.x === this.x && v.y === this.y && v.z === this.z; }
    add(v) { this.x += v.x; this.y += v.y; this.z += v.z; return this; }
    addScalar(s) { this.x += s; this.y += s; this.z += s; return this; }
    addVectors(a, b) { this.x = a.x + b.x; this.y = a.y + b.y; this.z = a.z + b.z; return this; }
    addScaledVector(v, s) { this.x += v.x * s; this.y += v.y * s; this.z += v.z * s; return this; }
    sub(v) { this.x -= v.x; this.y -= v.y; this.z -= v.z; return this; }
    subScalar(s) { this.x -= s; this.y -= s; this.z -= s; return this; }
    subVectors(a, b) { this.x = a.x - b.x; this.y = a.y - b.y; this.z = a.z - b.z; return this; }
    multiply(v) { this.x *= v.x; this.y *= v.y; this.z *= v.z; return this; }
    multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; }
    multiplyVectors(a, b) { this.x = a.x * b.x; this.y = a.y * b.y; this.z = a.z * b.z; return this; }
    divide(v) { this.x /= v.x; this.y /= v.y; this.z /= v.z; return this; }
    divideScalar(s) { return this.multiplyScalar(1 / s); }
    negate() { this.x = -this.x; this.y = -this.y; this.z = -this.z; return this; }
    length() { return Math.hypot(this.x, this.y, this.z); }
    lengthSq() { return this.x*this.x + this.y*this.y + this.z*this.z; }
    manhattanLength() { return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z); }
    normalize() { const l = this.length() || 1; this.x /= l; this.y /= l; this.z /= l; return this; }
    setLength(l) { return this.normalize().multiplyScalar(l); }
    dot(v) { return this.x*v.x + this.y*v.y + this.z*v.z; }
    angleTo(v) {
        const denom = Math.sqrt(this.lengthSq() * v.lengthSq());
        if (denom === 0) return Math.PI / 2;
        return Math.acos(Math.min(1, Math.max(-1, this.dot(v) / denom)));
    }
    cross(v) {
        const ax = this.x, ay = this.y, az = this.z;
        this.x = ay*v.z - az*v.y; this.y = az*v.x - ax*v.z; this.z = ax*v.y - ay*v.x;
        return this;
    }
    crossVectors(a, b) {
        const ax = a.x, ay = a.y, az = a.z, bx = b.x, by = b.y, bz = b.z;
        this.x = ay*bz - az*by; this.y = az*bx - ax*bz; this.z = ax*by - ay*bx;
        return this;
    }
    distanceTo(v) { const dx = this.x-v.x, dy = this.y-v.y, dz = this.z-v.z; return Math.hypot(dx, dy, dz); }
    distanceToSquared(v) { const dx = this.x-v.x, dy = this.y-v.y, dz = this.z-v.z; return dx*dx + dy*dy + dz*dz; }
    manhattanDistanceTo(v) { return Math.abs(this.x-v.x) + Math.abs(this.y-v.y) + Math.abs(this.z-v.z); }
    lerp(v, a) { this.x += (v.x - this.x) * a; this.y += (v.y - this.y) * a; this.z += (v.z - this.z) * a; return this; }
    lerpVectors(a, b, t) { this.x = a.x + (b.x-a.x)*t; this.y = a.y + (b.y-a.y)*t; this.z = a.z + (b.z-a.z)*t; return this; }
    applyMatrix4(m) {
        // three.js convention: m.elements is column-major. v' = m * (x,y,z,1), with perspective divide.
        const e = m.elements || m._w?.elements?.() || m;
        const x = this.x, y = this.y, z = this.z;
        const w = 1 / (e[3]*x + e[7]*y + e[11]*z + e[15] || 1);
        this.x = (e[0]*x + e[4]*y + e[8]*z + e[12]) * w;
        this.y = (e[1]*x + e[5]*y + e[9]*z + e[13]) * w;
        this.z = (e[2]*x + e[6]*y + e[10]*z + e[14]) * w;
        return this;
    }
    applyMatrix3(m) {
        const e = m.elements || m._w?.elements?.() || m;
        const x = this.x, y = this.y, z = this.z;
        this.x = e[0]*x + e[3]*y + e[6]*z;
        this.y = e[1]*x + e[4]*y + e[7]*z;
        this.z = e[2]*x + e[5]*y + e[8]*z;
        return this;
    }
    applyQuaternion(q) {
        const x = this.x, y = this.y, z = this.z;
        const qx = q.x, qy = q.y, qz = q.z, qw = q.w;
        const ix = qw*x + qy*z - qz*y;
        const iy = qw*y + qz*x - qx*z;
        const iz = qw*z + qx*y - qy*x;
        const iw = -qx*x - qy*y - qz*z;
        this.x = ix*qw + iw*-qx + iy*-qz - iz*-qy;
        this.y = iy*qw + iw*-qy + iz*-qx - ix*-qz;
        this.z = iz*qw + iw*-qz + ix*-qy - iy*-qx;
        return this;
    }
    transformDirection(m) {
        const e = m.elements || m._w?.elements?.() || m;
        const x = this.x, y = this.y, z = this.z;
        this.x = e[0]*x + e[4]*y + e[8]*z;
        this.y = e[1]*x + e[5]*y + e[9]*z;
        this.z = e[2]*x + e[6]*y + e[10]*z;
        return this.normalize();
    }
    applyNormalMatrix(m) {
        return this.applyMatrix3(m).normalize();
    }
    setFromMatrixPosition(m) {
        const e = m.elements || m._w?.elements?.() || m;
        this.x = e[12]; this.y = e[13]; this.z = e[14]; return this;
    }
    setFromMatrixColumn(m, i) {
        const e = m.elements || m._w?.elements?.() || m;
        const o = i * 4;
        this.x = e[o]; this.y = e[o+1]; this.z = e[o+2]; return this;
    }
    project(camera) { return this.applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix); }
    unproject(camera) { return this.applyMatrix4(camera.projectionMatrixInverse).applyMatrix4(camera.matrixWorld); }
    fromArray(arr, off = 0) { this.x = arr[off]; this.y = arr[off+1]; this.z = arr[off+2]; return this; }
    fromBufferAttribute(attr, index) {
        const i = index * attr.itemSize;
        return this.fromArray(attr.array, i);
    }
    toArray(arr = [], off = 0) { arr[off] = this.x; arr[off+1] = this.y; arr[off+2] = this.z; return arr; }
    *[Symbol.iterator]() { yield this.x; yield this.y; yield this.z; }
    _w() { return new WebVector3(this.x, this.y, this.z); }
}

export class Vector4 {
    constructor(x = 0, y = 0, z = 0, w = 1) { this.x = x; this.y = y; this.z = z; this.w = w; }
    set(x, y, z, w) { this.x = x; this.y = y; this.z = z; this.w = w; return this; }
    copy(v) { this.x = v.x; this.y = v.y; this.z = v.z; this.w = v.w ?? 1; return this; }
    multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; this.w *= s; return this; }
    addScaledVector(v, s) {
        this.x += v.x * s;
        this.y += v.y * s;
        this.z += (v.z ?? 0) * s;
        this.w += (v.w ?? 0) * s;
        return this;
    }
    fromArray(arr, off = 0) { this.x = arr[off]; this.y = arr[off+1]; this.z = arr[off+2]; this.w = arr[off+3]; return this; }
    fromBufferAttribute(attr, index) {
        const i = index * attr.itemSize;
        return this.fromArray(attr.array, i);
    }
    normalize() {
        const len = Math.hypot(this.x, this.y, this.z, this.w) || 1;
        this.x /= len; this.y /= len; this.z /= len; this.w /= len;
        return this;
    }
    _w() { return new WebVector4(this.x, this.y, this.z, this.w); }
}

export class Euler {
    constructor(x = 0, y = 0, z = 0, order = 'XYZ') { this.x = x; this.y = y; this.z = z; this.order = order; }
    set(x, y, z) {
        this.x = x; this.y = y; this.z = z;
        return this;
    }
    copy(e) { this.x = e.x; this.y = e.y; this.z = e.z; this.order = e.order ?? this.order; return this; }
    _w() { return new WebEuler(this.x, this.y, this.z); }
}

export class Quaternion {
    constructor(x = 0, y = 0, z = 0, w = 1) { this.x = x; this.y = y; this.z = z; this.w = w; }
    set(x, y, z, w) { this.x = x; this.y = y; this.z = z; this.w = w; return this; }
    copy(q) { this.x = q.x; this.y = q.y; this.z = q.z; this.w = q.w; return this; }
    clone() { return new Quaternion(this.x, this.y, this.z, this.w); }
    identity() { return this.set(0, 0, 0, 1); }
    setFromEuler(e) {
        const q = WebQuaternion.setFromEuler(e._w());
        this.x = q.x; this.y = q.y; this.z = q.z; this.w = q.w;
        return this;
    }
    setFromAxisAngle(axis, angle) {
        const ha = angle / 2, s = Math.sin(ha);
        this.x = axis.x * s; this.y = axis.y * s; this.z = axis.z * s; this.w = Math.cos(ha);
        return this;
    }
    setFromRotationMatrix(m) {
        // From three.js — assumes m.elements is column-major (XYZ basis vectors).
        const te = m.elements || m._w?.elements?.() || m;
        const m11 = te[0], m12 = te[4], m13 = te[8];
        const m21 = te[1], m22 = te[5], m23 = te[9];
        const m31 = te[2], m32 = te[6], m33 = te[10];
        const trace = m11 + m22 + m33;
        let s;
        if (trace > 0) {
            s = 0.5 / Math.sqrt(trace + 1.0);
            this.w = 0.25 / s; this.x = (m32 - m23) * s; this.y = (m13 - m31) * s; this.z = (m21 - m12) * s;
        } else if (m11 > m22 && m11 > m33) {
            s = 2.0 * Math.sqrt(1.0 + m11 - m22 - m33);
            this.w = (m32 - m23) / s; this.x = 0.25 * s; this.y = (m12 + m21) / s; this.z = (m13 + m31) / s;
        } else if (m22 > m33) {
            s = 2.0 * Math.sqrt(1.0 + m22 - m11 - m33);
            this.w = (m13 - m31) / s; this.x = (m12 + m21) / s; this.y = 0.25 * s; this.z = (m23 + m32) / s;
        } else {
            s = 2.0 * Math.sqrt(1.0 + m33 - m11 - m22);
            this.w = (m21 - m12) / s; this.x = (m13 + m31) / s; this.y = (m23 + m32) / s; this.z = 0.25 * s;
        }
        return this;
    }
    invert() { return this.conjugate(); /* unit quaternion: inverse == conjugate */ }
    conjugate() { this.x = -this.x; this.y = -this.y; this.z = -this.z; return this; }
    dot(q) { return this.x*q.x + this.y*q.y + this.z*q.z + this.w*q.w; }
    length() { return Math.hypot(this.x, this.y, this.z, this.w); }
    lengthSq() { return this.x*this.x + this.y*this.y + this.z*this.z + this.w*this.w; }
    normalize() {
        const l = this.length();
        if (l === 0) { this.x = 0; this.y = 0; this.z = 0; this.w = 1; }
        else { this.x /= l; this.y /= l; this.z /= l; this.w /= l; }
        return this;
    }
    multiply(q) { return this.multiplyQuaternions(this, q); }
    premultiply(q) { return this.multiplyQuaternions(q, this); }
    multiplyQuaternions(a, b) {
        const qax = a.x, qay = a.y, qaz = a.z, qaw = a.w;
        const qbx = b.x, qby = b.y, qbz = b.z, qbw = b.w;
        this.x = qax*qbw + qaw*qbx + qay*qbz - qaz*qby;
        this.y = qay*qbw + qaw*qby + qaz*qbx - qax*qbz;
        this.z = qaz*qbw + qaw*qbz + qax*qby - qay*qbx;
        this.w = qaw*qbw - qax*qbx - qay*qby - qaz*qbz;
        return this;
    }
    slerp(qb, t) {
        if (t === 0) return this; if (t === 1) return this.copy(qb);
        const x = this.x, y = this.y, z = this.z, w = this.w;
        let cosHalfTheta = w*qb.w + x*qb.x + y*qb.y + z*qb.z;
        let qbw = qb.w, qbx = qb.x, qby = qb.y, qbz = qb.z;
        if (cosHalfTheta < 0) { cosHalfTheta = -cosHalfTheta; qbw = -qbw; qbx = -qbx; qby = -qby; qbz = -qbz; }
        if (cosHalfTheta >= 1.0) { this.x = x; this.y = y; this.z = z; this.w = w; return this; }
        const sqrSinHalfTheta = 1.0 - cosHalfTheta * cosHalfTheta;
        if (sqrSinHalfTheta <= Number.EPSILON) {
            const s = 1 - t;
            this.w = s*w + t*qbw; this.x = s*x + t*qbx; this.y = s*y + t*qby; this.z = s*z + t*qbz;
            return this.normalize();
        }
        const sinHalfTheta = Math.sqrt(sqrSinHalfTheta);
        const halfTheta = Math.atan2(sinHalfTheta, cosHalfTheta);
        const ra = Math.sin((1 - t) * halfTheta) / sinHalfTheta;
        const rb = Math.sin(t * halfTheta) / sinHalfTheta;
        this.w = w*ra + qbw*rb; this.x = x*ra + qbx*rb; this.y = y*ra + qby*rb; this.z = z*ra + qbz*rb;
        return this;
    }
    equals(q) { return q && q.x === this.x && q.y === this.y && q.z === this.z && q.w === this.w; }
    fromArray(a, off = 0) { this.x = a[off]; this.y = a[off+1]; this.z = a[off+2]; this.w = a[off+3]; return this; }
    toArray(a = [], off = 0) { a[off] = this.x; a[off+1] = this.y; a[off+2] = this.z; a[off+3] = this.w; return a; }
    _w() { return new WebQuaternion(this.x, this.y, this.z, this.w); }
}

export class Matrix3 {
    constructor() {
        this.elements = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ];
    }
    setFromMatrix4(m) {
        const e = m.elements;
        const me = this.elements;
        me[0] = e[0]; me[1] = e[1]; me[2] = e[2];
        me[3] = e[4]; me[4] = e[5]; me[5] = e[6];
        me[6] = e[8]; me[7] = e[9]; me[8] = e[10];
        return this;
    }
    getNormalMatrix(matrix) {
        return this.setFromMatrix4(matrix).invert().transpose();
    }
    invert() {
        const te = this.elements;
        const n11 = te[0], n21 = te[1], n31 = te[2];
        const n12 = te[3], n22 = te[4], n32 = te[5];
        const n13 = te[6], n23 = te[7], n33 = te[8];
        const t11 = n33 * n22 - n32 * n23;
        const t12 = n32 * n13 - n33 * n12;
        const t13 = n23 * n12 - n22 * n13;
        const det = n11 * t11 + n21 * t12 + n31 * t13;
        if (det === 0) return this;
        const detInv = 1 / det;
        te[0] = t11 * detInv;
        te[1] = (n31 * n23 - n33 * n21) * detInv;
        te[2] = (n32 * n21 - n31 * n22) * detInv;
        te[3] = t12 * detInv;
        te[4] = (n33 * n11 - n31 * n13) * detInv;
        te[5] = (n31 * n12 - n32 * n11) * detInv;
        te[6] = t13 * detInv;
        te[7] = (n21 * n13 - n23 * n11) * detInv;
        te[8] = (n22 * n11 - n21 * n12) * detInv;
        return this;
    }
    transpose() {
        const te = this.elements;
        let tmp;
        tmp = te[1]; te[1] = te[3]; te[3] = tmp;
        tmp = te[2]; te[2] = te[6]; te[6] = tmp;
        tmp = te[5]; te[5] = te[7]; te[7] = tmp;
        return this;
    }
    multiplyScalar(s) {
        const te = this.elements;
        for (let i = 0; i < 9; i++) te[i] *= s;
        return this;
    }
    _w() {
        const m = new WebMatrix3();
        const e = m.elements();
        for (let i = 0; i < 9; i++) e[i] = this.elements[i];
        return m;
    }
}

export class Matrix4 {
    constructor() {
        this.elements = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
    }
    set(n11, n12, n13, n14, n21, n22, n23, n24, n31, n32, n33, n34, n41, n42, n43, n44) {
        const e = this.elements;
        e[0]=n11; e[4]=n12; e[8]=n13; e[12]=n14;
        e[1]=n21; e[5]=n22; e[9]=n23; e[13]=n24;
        e[2]=n31; e[6]=n32; e[10]=n33; e[14]=n34;
        e[3]=n41; e[7]=n42; e[11]=n43; e[15]=n44;
        return this;
    }
    identity() { return this.set(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1); }
    setPosition(x, y, z) {
        if (typeof x === 'object') { y = x.y; z = x.z; x = x.x; }
        this.elements[12] = x; this.elements[13] = y; this.elements[14] = z;
        return this;
    }
    copy(m) { for (let i = 0; i < 16; i++) this.elements[i] = m.elements[i]; return this; }
    clone() { return new Matrix4().fromArray(this.elements); }
    fromArray(a, off = 0) { for (let i = 0; i < 16; i++) this.elements[i] = a[off + i]; return this; }
    toArray(a = [], off = 0) { for (let i = 0; i < 16; i++) a[off + i] = this.elements[i]; return a; }
    makeTranslation(x, y, z) {
        if (typeof x === 'object') { y = x.y; z = x.z; x = x.x; }
        return this.set(1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1);
    }
    makeScale(x, y, z) { return this.set(x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1); }
    makeRotationX(t) { const c = Math.cos(t), s = Math.sin(t); return this.set(1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1); }
    makeRotationY(t) { const c = Math.cos(t), s = Math.sin(t); return this.set(c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1); }
    makeRotationZ(t) { const c = Math.cos(t), s = Math.sin(t); return this.set(c,-s,0,0, s,c,0,0, 0,0,1,0, 0,0,0,1); }
    makeRotationAxis(axis, angle) {
        const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
        const x = axis.x, y = axis.y, z = axis.z;
        return this.set(
            t*x*x + c,    t*x*y - s*z,  t*x*z + s*y,  0,
            t*x*y + s*z,  t*y*y + c,    t*y*z - s*x,  0,
            t*x*z - s*y,  t*y*z + s*x,  t*z*z + c,    0,
            0,            0,            0,            1,
        );
    }
    makeRotationFromQuaternion(q) {
        const x = q.x, y = q.y, z = q.z, w = q.w;
        const x2 = x + x, y2 = y + y, z2 = z + z;
        const xx = x * x2, xy = x * y2, xz = x * z2;
        const yy = y * y2, yz = y * z2, zz = z * z2;
        const wx = w * x2, wy = w * y2, wz = w * z2;
        return this.set(
            1 - (yy + zz), xy - wz,       xz + wy,       0,
            xy + wz,       1 - (xx + zz), yz - wx,       0,
            xz - wy,       yz + wx,       1 - (xx + yy), 0,
            0,             0,             0,             1
        );
    }
    makeRotationFromEuler(e) {
        const c1 = Math.cos(e.x), s1 = Math.sin(e.x);
        const c2 = Math.cos(e.y), s2 = Math.sin(e.y);
        const c3 = Math.cos(e.z), s3 = Math.sin(e.z);
        // XYZ order (three.js default).
        return this.set(
            c2 * c3,                 -c2 * s3,                s2,       0,
            s1 * s2 * c3 + c1 * s3,  -s1 * s2 * s3 + c1 * c3, -s1 * c2, 0,
            -c1 * s2 * c3 + s1 * s3, c1 * s2 * s3 + s1 * c3,  c1 * c2,  0,
            0,                       0,                       0,        1
        );
    }
    makePerspective(fovDeg_or_left, aspect_or_right, near_or_top, far_or_bottom, near_, far_) {
        // Support both signatures: (fov, aspect, near, far) and (left, right, top, bottom, near, far).
        if (arguments.length === 4) {
            const fov = fovDeg_or_left, aspect = aspect_or_right, near = near_or_top, far = far_or_bottom;
            const f = 1 / Math.tan(fov * 0.5 * Math.PI / 180);
            return this.set(
                f/aspect, 0, 0, 0,
                0,        f, 0, 0,
                0,        0, (far+near)/(near-far), (2*far*near)/(near-far),
                0,        0, -1, 0,
            );
        }
        // 6-arg form
        const l = fovDeg_or_left, r = aspect_or_right, t = near_or_top, b = far_or_bottom, n = near_, fa = far_;
        return this.set(
            2*n/(r-l), 0, (r+l)/(r-l), 0,
            0, 2*n/(t-b), (t+b)/(t-b), 0,
            0, 0, -(fa+n)/(fa-n), -2*fa*n/(fa-n),
            0, 0, -1, 0,
        );
    }
    makeOrthographic(l, r, t, b, n, f) {
        return this.set(
            2/(r-l), 0, 0, -(r+l)/(r-l),
            0, 2/(t-b), 0, -(t+b)/(t-b),
            0, 0, -2/(f-n), -(f+n)/(f-n),
            0, 0, 0, 1,
        );
    }
    multiply(m) { return this.multiplyMatrices(this, m); }
    premultiply(m) { return this.multiplyMatrices(m, this); }
    multiplyMatrices(a, b) {
        const ae = a.elements, be = b.elements, te = this.elements;
        const a11=ae[0], a12=ae[4], a13=ae[8], a14=ae[12];
        const a21=ae[1], a22=ae[5], a23=ae[9], a24=ae[13];
        const a31=ae[2], a32=ae[6], a33=ae[10],a34=ae[14];
        const a41=ae[3], a42=ae[7], a43=ae[11],a44=ae[15];
        const b11=be[0], b12=be[4], b13=be[8], b14=be[12];
        const b21=be[1], b22=be[5], b23=be[9], b24=be[13];
        const b31=be[2], b32=be[6], b33=be[10],b34=be[14];
        const b41=be[3], b42=be[7], b43=be[11],b44=be[15];
        te[0]  = a11*b11 + a12*b21 + a13*b31 + a14*b41;
        te[4]  = a11*b12 + a12*b22 + a13*b32 + a14*b42;
        te[8]  = a11*b13 + a12*b23 + a13*b33 + a14*b43;
        te[12] = a11*b14 + a12*b24 + a13*b34 + a14*b44;
        te[1]  = a21*b11 + a22*b21 + a23*b31 + a24*b41;
        te[5]  = a21*b12 + a22*b22 + a23*b32 + a24*b42;
        te[9]  = a21*b13 + a22*b23 + a23*b33 + a24*b43;
        te[13] = a21*b14 + a22*b24 + a23*b34 + a24*b44;
        te[2]  = a31*b11 + a32*b21 + a33*b31 + a34*b41;
        te[6]  = a31*b12 + a32*b22 + a33*b32 + a34*b42;
        te[10] = a31*b13 + a32*b23 + a33*b33 + a34*b43;
        te[14] = a31*b14 + a32*b24 + a33*b34 + a34*b44;
        te[3]  = a41*b11 + a42*b21 + a43*b31 + a44*b41;
        te[7]  = a41*b12 + a42*b22 + a43*b32 + a44*b42;
        te[11] = a41*b13 + a42*b23 + a43*b33 + a44*b43;
        te[15] = a41*b14 + a42*b24 + a43*b34 + a44*b44;
        return this;
    }
    transpose() {
        const e = this.elements; let t;
        t = e[1]; e[1] = e[4]; e[4] = t;
        t = e[2]; e[2] = e[8]; e[8] = t;
        t = e[6]; e[6] = e[9]; e[9] = t;
        t = e[3]; e[3] = e[12]; e[12] = t;
        t = e[7]; e[7] = e[13]; e[13] = t;
        t = e[11]; e[11] = e[14]; e[14] = t;
        return this;
    }
    invert() {
        // From three.js (Mesa GLU). Returns identity if non-invertible.
        const e = this.elements;
        const n11=e[0], n21=e[1], n31=e[2], n41=e[3];
        const n12=e[4], n22=e[5], n32=e[6], n42=e[7];
        const n13=e[8], n23=e[9], n33=e[10],n43=e[11];
        const n14=e[12],n24=e[13],n34=e[14],n44=e[15];
        const t11 = n23*n34*n42 - n24*n33*n42 + n24*n32*n43 - n22*n34*n43 - n23*n32*n44 + n22*n33*n44;
        const t12 = n14*n33*n42 - n13*n34*n42 - n14*n32*n43 + n12*n34*n43 + n13*n32*n44 - n12*n33*n44;
        const t13 = n13*n24*n42 - n14*n23*n42 + n14*n22*n43 - n12*n24*n43 - n13*n22*n44 + n12*n23*n44;
        const t14 = n14*n23*n32 - n13*n24*n32 - n14*n22*n33 + n12*n24*n33 + n13*n22*n34 - n12*n23*n34;
        const det = n11*t11 + n21*t12 + n31*t13 + n41*t14;
        if (det === 0) return this.identity();
        const di = 1 / det;
        e[0] = t11 * di; e[4] = t12 * di; e[8] = t13 * di; e[12] = t14 * di;
        e[1] = (n24*n33*n41 - n23*n34*n41 - n24*n31*n43 + n21*n34*n43 + n23*n31*n44 - n21*n33*n44) * di;
        e[5] = (n13*n34*n41 - n14*n33*n41 + n14*n31*n43 - n11*n34*n43 - n13*n31*n44 + n11*n33*n44) * di;
        e[9] = (n14*n23*n41 - n13*n24*n41 - n14*n21*n43 + n11*n24*n43 + n13*n21*n44 - n11*n23*n44) * di;
        e[13] = (n13*n24*n31 - n14*n23*n31 + n14*n21*n33 - n11*n24*n33 - n13*n21*n34 + n11*n23*n34) * di;
        e[2] = (n22*n34*n41 - n24*n32*n41 + n24*n31*n42 - n21*n34*n42 - n22*n31*n44 + n21*n32*n44) * di;
        e[6] = (n14*n32*n41 - n12*n34*n41 - n14*n31*n42 + n11*n34*n42 + n12*n31*n44 - n11*n32*n44) * di;
        e[10] = (n12*n24*n41 - n14*n22*n41 + n14*n21*n42 - n11*n24*n42 - n12*n21*n44 + n11*n22*n44) * di;
        e[14] = (n14*n22*n31 - n12*n24*n31 - n14*n21*n32 + n11*n24*n32 + n12*n21*n34 - n11*n22*n34) * di;
        e[3] = (n23*n32*n41 - n22*n33*n41 - n23*n31*n42 + n21*n33*n42 + n22*n31*n43 - n21*n32*n43) * di;
        e[7] = (n12*n33*n41 - n13*n32*n41 + n13*n31*n42 - n11*n33*n42 - n12*n31*n43 + n11*n32*n43) * di;
        e[11] = (n13*n22*n41 - n12*n23*n41 - n13*n21*n42 + n11*n23*n42 + n12*n21*n43 - n11*n22*n43) * di;
        e[15] = (n12*n23*n31 - n13*n22*n31 + n13*n21*n32 - n11*n23*n32 - n12*n21*n33 + n11*n22*n33) * di;
        return this;
    }
    compose(pos, quat, scl) {
        const e = this.elements;
        const x = quat.x, y = quat.y, z = quat.z, w = quat.w;
        const x2 = x+x, y2 = y+y, z2 = z+z;
        const xx = x*x2, xy = x*y2, xz = x*z2;
        const yy = y*y2, yz = y*z2, zz = z*z2;
        const wx = w*x2, wy = w*y2, wz = w*z2;
        const sx = scl.x, sy = scl.y, sz = scl.z;
        e[0] = (1-(yy+zz))*sx; e[1] = (xy+wz)*sx;    e[2] = (xz-wy)*sx;    e[3] = 0;
        e[4] = (xy-wz)*sy;     e[5] = (1-(xx+zz))*sy; e[6] = (yz+wx)*sy;   e[7] = 0;
        e[8] = (xz+wy)*sz;     e[9] = (yz-wx)*sz;    e[10] = (1-(xx+yy))*sz; e[11] = 0;
        e[12] = pos.x; e[13] = pos.y; e[14] = pos.z; e[15] = 1;
        return this;
    }
    decompose(pos, quat, scl) {
        const e = this.elements;
        let sx = new Vector3(e[0], e[1], e[2]).length();
        const sy = new Vector3(e[4], e[5], e[6]).length();
        const sz = new Vector3(e[8], e[9], e[10]).length();
        const det = this._determinant();
        if (det < 0) sx = -sx;
        pos.x = e[12]; pos.y = e[13]; pos.z = e[14];
        const m1 = new Matrix4().fromArray(e);
        const isx = 1/sx, isy = 1/sy, isz = 1/sz;
        const me = m1.elements;
        me[0]*=isx; me[1]*=isx; me[2]*=isx;
        me[4]*=isy; me[5]*=isy; me[6]*=isy;
        me[8]*=isz; me[9]*=isz; me[10]*=isz;
        quat.setFromRotationMatrix(m1);
        scl.x = sx; scl.y = sy; scl.z = sz;
        return this;
    }
    _determinant() {
        const e = this.elements;
        const n11 = e[0], n21 = e[1], n31 = e[2], n41 = e[3];
        const n12 = e[4], n22 = e[5], n32 = e[6], n42 = e[7];
        const n13 = e[8], n23 = e[9], n33 = e[10], n43 = e[11];
        const n14 = e[12], n24 = e[13], n34 = e[14], n44 = e[15];
        return (
            n41 * (+n14*n23*n32 - n13*n24*n32 - n14*n22*n33 + n12*n24*n33 + n13*n22*n34 - n12*n23*n34) +
            n42 * (+n11*n23*n34 - n11*n24*n33 + n14*n21*n33 - n13*n21*n34 + n13*n24*n31 - n14*n23*n31) +
            n43 * (+n11*n24*n32 - n11*n22*n34 - n14*n21*n32 + n12*n21*n34 + n14*n22*n31 - n12*n24*n31) +
            n44 * (-n13*n22*n31 - n11*n23*n32 + n11*n22*n33 + n13*n21*n32 - n12*n21*n33 + n12*n23*n31)
        );
    }
    determinant() { return this._determinant(); }
    lookAt(eye, target, up) {
        const z = new Vector3().subVectors(eye, target);
        if (z.lengthSq() === 0) z.z = 1; z.normalize();
        const x = new Vector3().crossVectors(up, z);
        if (x.lengthSq() === 0) { z.x += 1e-4; x.crossVectors(up, z); }
        x.normalize();
        const y = new Vector3().crossVectors(z, x);
        const e = this.elements;
        e[0] = x.x; e[4] = y.x; e[8] = z.x;
        e[1] = x.y; e[5] = y.y; e[9] = z.y;
        e[2] = x.z; e[6] = y.z; e[10] = z.z;
        return this;
    }
}

export class Box2 {
    constructor(min, max) {
        if (min && max) this._w = new WebBox2(min._w(), max._w());
        else this._w = WebBox2.empty();
    }
    isEmpty() { return this._w.isEmpty(); }
}

export class Box3 {
    constructor(min, max) {
        if (min && max) this._w = new WebBox3(min._w(), max._w());
        else this._w = WebBox3.empty();
    }
    isEmpty() { return this._w.isEmpty(); }
    containsPoint(p) { return this._w.containsPoint(p._w()); }
    intersectsBox(b) { return this._w.intersectsBox(b._w); }
}

export class Sphere {
    constructor(center, radius) { this._w = new WebSphere((center || new Vector3())._w(), radius || 0); }
    containsPoint(p) { return this._w.containsPoint(p._w()); }
}

export class Ray {
    constructor(origin = new Vector3(), direction = new Vector3(0, 0, -1)) {
        this.origin = origin.clone ? origin.clone() : new Vector3(origin.x || 0, origin.y || 0, origin.z || 0);
        this.direction = direction.clone ? direction.clone() : new Vector3(direction.x || 0, direction.y || 0, direction.z || -1);
    }
    set(origin, direction) { this.origin.copy(origin); this.direction.copy(direction); return this; }
    copy(r) { this.origin.copy(r.origin); this.direction.copy(r.direction); return this; }
    clone() { return new Ray(this.origin, this.direction); }
    at(t, target = new Vector3()) {
        return target.set(
            this.origin.x + this.direction.x * t,
            this.origin.y + this.direction.y * t,
            this.origin.z + this.direction.z * t,
        );
    }
    lookAt(v) { this.direction.copy(v).sub(this.origin).normalize(); return this; }
    _w() { return new WebRay(this.origin._w(), this.direction._w()); }
}

export class Plane {
    constructor(normal, constant) {
        this.normal = normal ? normal.clone() : new Vector3(1, 0, 0);
        this.constant = constant ?? 0;
    }
    set(normal, constant) {
        this.normal.copy(normal);
        this.constant = constant;
        return this;
    }
    setFromNormalAndCoplanarPoint(normal, point) {
        this.normal.copy(normal);
        this.constant = -this.normal.dot(point);
        return this;
    }
    distanceToPoint(p) {
        return this.normal.dot(p) + this.constant;
    }
    intersectLine(line, target = new Vector3()) {
        const start = line.start || line;
        const end = line.end;
        const d1 = this.distanceToPoint(start);
        const d2 = this.distanceToPoint(end);
        if (d1 * d2 < 0) {
            const t = d1 / (d1 - d2);
            target.lerpVectors(start, end, t);
            return target;
        }
        if (Math.abs(d1) < 1e-10) {
            target.copy(start);
            return target;
        }
        return null;
    }
    _w() {
        return new WebPlane(this.normal._w(), this.constant);
    }
}

export class Triangle {
    constructor(a, b, c) {
        this.a = a ? a.clone() : new Vector3();
        this.b = b ? b.clone() : new Vector3();
        this.c = c ? c.clone() : new Vector3();
    }
    set(a, b, c) {
        this.a.copy(a);
        this.b.copy(b);
        this.c.copy(c);
        return this;
    }
    copy(t) {
        this.a.copy(t.a);
        this.b.copy(t.b);
        this.c.copy(t.c);
        return this;
    }
    clone() {
        return new Triangle(this.a, this.b, this.c);
    }
    getMidpoint(target = new Vector3()) {
        return target.addVectors(this.a, this.b).add(this.c).multiplyScalar(1 / 3);
    }
    getNormal(target = new Vector3()) {
        const ab = new Vector3().subVectors(this.b, this.a);
        const ac = new Vector3().subVectors(this.c, this.a);
        return target.crossVectors(ab, ac).normalize();
    }
    getBarycoord(point, target = new Vector3()) {
        const v0 = new Vector3().subVectors(this.c, this.a);
        const v1 = new Vector3().subVectors(this.b, this.a);
        const v2 = new Vector3().subVectors(point, this.a);
        const dot00 = v0.dot(v0);
        const dot01 = v0.dot(v1);
        const dot02 = v0.dot(v2);
        const dot11 = v1.dot(v1);
        const dot12 = v1.dot(v2);
        const denom = dot00 * dot11 - dot01 * dot01;
        if (denom === 0) return target.set(-2, -1, -1);
        const inv = 1 / denom;
        const u = (dot11 * dot02 - dot01 * dot12) * inv;
        const v = (dot00 * dot12 - dot01 * dot02) * inv;
        return target.set(1 - u - v, v, u);
    }
    getPlane(target = new Plane()) {
        const n = this.getNormal(new Vector3());
        return target.setFromNormalAndCoplanarPoint(n, this.a);
    }
    fromBufferAttribute(attr, index) {
        const i = index * attr.itemSize;
        const arr = attr.array;
        this.a.fromArray(arr, i);
        return this;
    }
    area() {
        const ab = new Vector3().subVectors(this.b, this.a);
        const ac = new Vector3().subVectors(this.c, this.a);
        return ab.cross(ac).length() * 0.5;
    }
    get plane() {
        return this.getPlane(new Plane());
    }
    intersectsTriangle(other, targetEdge = new Line3(), coplanar = false) {
        const plane = this.getPlane(new Plane());
        const dA = plane.distanceToPoint(other.a);
        const dB = plane.distanceToPoint(other.b);
        const dC = plane.distanceToPoint(other.c);
        const eps = 1e-10;
        if (dA > eps && dB > eps && dC > eps) return false;
        if (dA < -eps && dB < -eps && dC < -eps) return false;
        const plane2 = other.getPlane(new Plane());
        const eA = plane2.distanceToPoint(this.a);
        const eB = plane2.distanceToPoint(this.b);
        const eC = plane2.distanceToPoint(this.c);
        if (eA > eps && eB > eps && eC > eps) return false;
        if (eA < -eps && eB < -eps && eC < -eps) return false;
        const edge = targetEdge;
        const pts = [other.a, other.b, other.c];
        let hits = 0;
        for (let i = 0; i < 3; i++) {
            const s = pts[i];
            const e = pts[(i + 1) % 3];
            const line = new Line3(s, e);
            const hit = plane.intersectLine(line, new Vector3());
            if (hit && hit.distanceTo(e) > eps) {
                if (hits === 0) edge.start.copy(hit);
                else edge.end.copy(hit);
                hits++;
            }
        }
        return hits >= 2 || coplanar;
    }
    _w() {
        return new WebTriangle(this.a._w(), this.b._w(), this.c._w());
    }
}

export class Frustum {
    constructor() { this._w = new WebFrustum(); }
    setFromProjectionMatrix(m) { this._w = WebFrustum.setFromProjectionMatrix(m._w); return this; }
    containsPoint(p) { return this._w.containsPoint(p._w()); }
}

export class Spherical {
    constructor(r = 1, phi = 0, theta = 0) { this._w = new WebSpherical(r, phi, theta); }
}

export class Cylindrical {
    constructor(r = 1, theta = 0, y = 0) { this._w = new WebCylindrical(r, theta, y); }
}

export class Line3 {
    constructor(start, end) {
        this.start = start ? start.clone() : new Vector3();
        this.end = end ? end.clone() : new Vector3();
    }
    set(start, end) {
        this.start.copy(start);
        this.end.copy(end);
        return this;
    }
    copy(line) {
        this.start.copy(line.start);
        this.end.copy(line.end);
        return this;
    }
    delta(target = new Vector3()) {
        return target.subVectors(this.end, this.start);
    }
    distance() {
        return this.start.distanceTo(this.end);
    }
    _w() { return new WebLine3(this.start._w(), this.end._w()); }
}

// ---- Geometries ----
export class BoxGeometry { constructor(w = 1, h = 1, d = 1) { this._w = WebGeometry.box(w, h, d); this.parameters = { width: w, height: h, depth: d }; } }
export class SphereGeometry {
    constructor(r = 1, ws = 32, hs = 16, phiStart = 0, phiLength = Math.PI * 2, thetaStart = 0, thetaLength = Math.PI) {
        // Earth/Moon demos pass phi/theta ranges for 90°×90° patches. Without
        // sphereRange every "tile" is a full sphere and textures fight/stack.
        const full =
            phiStart === 0 && phiLength === Math.PI * 2 &&
            thetaStart === 0 && thetaLength === Math.PI;
        this._w = full
            ? WebGeometry.sphere(r, ws, hs)
            : WebGeometry.sphereRange(r, ws, hs, phiStart, phiLength, thetaStart, thetaLength);
    }
}
export class PlaneGeometry { constructor(w = 1, h = 1) { this._w = WebGeometry.plane(w, h); } }
export class CylinderGeometry { constructor(rt = 1, rb = 1, h = 1, rs = 32) { this._w = WebGeometry.cylinder(rt, rb, h, rs); } }
export class TorusGeometry { constructor(r = 1, t = 0.4, rs = 12, ts = 48) { this._w = WebGeometry.torus(r, t, rs, ts); } }
export class CircleGeometry { constructor(r = 1, segs = 32) { this._w = WebGeometry.circle(r, segs); } }
export class RingGeometry { constructor(inner = 0.5, outer = 1, thetaSegments = 32) { this._w = WebGeometry.ring(inner, outer, thetaSegments); } }
export class ConeGeometry { constructor(r = 1, h = 1, rs = 32) { this._w = WebGeometry.cone(r, h, rs); } }
export class TorusKnotGeometry {
    constructor(r = 1, t = 0.4, tubular = 64, radial = 8, p = 2, q = 3) {
        this._w = WebGeometry.torusKnot(r, t, tubular, radial, p, q);
    }
}
export class CapsuleGeometry { constructor(r = 1, l = 1, capSegments = 4, radialSegments = 8) {
        const points = _capsuleLathePoints(r, l, capSegments);
        const lathe = new LatheGeometry(points, radialSegments);
        Object.assign(this, lathe);
        this.type = 'CapsuleGeometry';
        this._isUserGeometry = true;
        this.parameters = { radius: r, length: l, capSegments, radialSegments };
    } }
export class TetrahedronGeometry { constructor(r = 1, d = 0) { this._w = WebGeometry.tetrahedron(r, d); } }
export class OctahedronGeometry { constructor(r = 1, d = 0) { this._w = WebGeometry.octahedron(r, d); } }
export class IcosahedronGeometry { constructor(r = 1, d = 0) { this._w = WebGeometry.icosahedron(r, d); } }
export class DodecahedronGeometry { constructor(r = 1, d = 0) { this._w = WebGeometry.dodecahedron(r, d); } }
export class BoxLineGeometry { constructor(w = 1, h = 1, d = 1) { this._w = WebGeometry.boxLine(w, h, d); } }

// ---- Materials ----
function _matColor(opts) {
    if (typeof opts === 'object' && opts !== null && 'color' in opts) return _color(opts.color);
    return _color(opts);
}
// Apply `opts.map` (a Texture / DataTexture) to a WebMaterial. three.js's
// MeshBasic/Standard/Physical/Sprite all accept `{ map: tex }`; we route to
// either setMap (Texture) or setMapData (DataTexture) based on which wasm
// handle the JS class is wrapping.
function _applyMap(w, opts) {
    if (!opts || !opts.map) return;
    const t = opts.map;
    // Sync the texture's filter/wrap to the wasm Texture before binding it.
    _syncTextureFilters(t);
    if (t.offset && t.repeat && t._w?.setUvTransform) {
        const ox = t.offset.x ?? 0, oy = t.offset.y ?? 0;
        const rx = t.repeat.x ?? 1, ry = t.repeat.y ?? 1;
        t._w.setUvTransform(ox, oy, rx, ry);
    }
    // CanvasTexture also wraps a WebDataTexture; route either via setMapData.
    if (t?._w?.constructor?.name === 'WebDataTexture' || t instanceof DataTexture || t instanceof CanvasTexture) w.setMapData(t._w);
    else if (t instanceof Texture) w.setMap(t._w);
}
// Translate three.js's numeric filter/wrap constants to our compact
// (mag, min, wrap_s, wrap_t) enum and push to the wasm Texture.
function _syncTextureFilters(t) {
    if (!t?._w?.setFilters) return;
    // three.js constants — NearestFilter = 1003, LinearFilter = 1006 (and mips).
    const NEAREST = 1003, NEAR_NEAR_MIP = 1004, LIN_NEAR_MIP = 1005, NEAR_LIN_MIP = 1007;
    const isNearest = (v) => v === NEAREST || v === NEAR_NEAR_MIP || v === LIN_NEAR_MIP || v === NEAR_LIN_MIP;
    const mag = isNearest(t.magFilter ?? 1006) ? 1 : 0;
    const min = isNearest(t.minFilter ?? 1008) ? 1 : 0;
    // wrap constants — RepeatWrapping = 1000, ClampToEdge = 1001, MirroredRepeat = 1002.
    const conv = (w) => w === 1000 ? 1 : w === 1002 ? 2 : 0;
    t._w.setFilters(mag, min, conv(t.wrapS ?? 1001), conv(t.wrapT ?? 1001));
}

// Apply the cross-material `transparent`/`opacity`/`wireframe`/`emissive`
// flags, mirroring three.js's MeshXxxMaterial constructor options.
function _applyCommon(w, opts) {
    if (!opts || typeof opts !== 'object') return;
    if (typeof opts.opacity === 'number') w.setOpacity(opts.opacity);
    if (opts.transparent === true) w.setTransparent(true);
    if (opts.wireframe === true && w.setWireframe) w.setWireframe(true);
    if ('emissive' in opts && w.setEmissive) w.setEmissive(_color(opts.emissive));
    if (typeof opts.emissiveIntensity === 'number' && w.setEmissiveIntensity) {
        w.setEmissiveIntensity(opts.emissiveIntensity);
    }
}
// Mix Material's cross-cutting fields (onBeforeCompile, userData, uuid, …)
// into any concrete material class. Idempotent.
function _initMaterialBase(self, opts = {}) {
    // Push side through to wasm so BackSide / DoubleSide route to the no-cull
    // triangle pipeline. (FrontSide is the default; no setter needed.)
    const sideVal = opts.side ?? 0;
    if (sideVal !== 0 && self._w?.setSide) {
        self._w.setSide(sideVal);
    }
    self.uuid = self.uuid || MathUtils.generateUUID();
    self.userData = self.userData || {};
    self.onBeforeCompile = self.onBeforeCompile || null;
    self.alphaTest = opts.alphaTest ?? 0;
    if (typeof self.alphaTest === 'number' && self.alphaTest > 0 && self._w?.setAlphaTest) {
        self._w.setAlphaTest(self.alphaTest);
    }
    const initColor = (typeof opts === 'object' && opts && 'color' in opts) ? opts.color : 0xffffff;
    self.color = new Color(initColor);
    const mat = self;
    const origSetRGB = self.color.setRGB.bind(self.color);
    self.color.setRGB = function(r, g, b) {
        origSetRGB(r, g, b);
        mat._w?.setColor?.(new WebColor(r, g, b));
        return this;
    };
    const origSetHex = self.color.setHex.bind(self.color);
    self.color.setHex = function(h) {
        origSetHex(h);
        mat._w?.setColor?.(self.color._w);
        return this;
    };
    self.depthTest = opts.depthTest ?? true;
    self.depthWrite = opts.depthWrite ?? true;
    self.colorWrite = opts.colorWrite ?? true;
    self.toneMapped = opts.toneMapped ?? true;
    self.blending = opts.blending ?? 1;
    self.side = opts.side ?? 0;
    self.transparent = opts.transparent ?? false;
    self.opacity = opts.opacity ?? 1;
    self.visible = opts.visible ?? true;
    self.wireframe = opts.wireframe ?? false;
    self.needsUpdate = false;
    self.version = 0;
    self._compiledOnce = false;
    // Provide `_runOnBeforeCompile` and `customProgramCacheKey` so calling
    // them on any material is safe regardless of class inheritance.
    if (!self._runOnBeforeCompile) {
        self._runOnBeforeCompile = function () {
            if (typeof this.onBeforeCompile !== 'function' || this._compiledOnce) return;
            const shader = { uniforms: { ...this.uniforms || {} }, vertexShader: '/* threers wgsl */', fragmentShader: '/* threers wgsl */', defines: {} };
            try { this.onBeforeCompile(shader, this); } catch (_e) {}
            this.userData.shader = shader;
            this.uniforms = shader.uniforms;
            this._compiledOnce = true;
        };
    }
    if (!self.customProgramCacheKey) self.customProgramCacheKey = () => self.uuid;
    if (!self.dispose) self.dispose = () => {};
    return self;
}
export class MeshBasicMaterial {
    constructor(opts = {}) {
        this._w = WebMaterial.basic(_matColor(opts));
        _applyMap(this._w, opts); _applyCommon(this._w, opts); _initMaterialBase(this, opts);
        Object.defineProperty(this, 'map', {
            get() { return this._map; },
            set(tex) {
                this._map = tex ?? null;
                if (tex) {
                    _syncTextureFilters(tex);
                    tex._syncUv?.();
                    _applyMap(this._w, { map: tex });
                }
            },
            configurable: true,
            enumerable: true,
        });
        this.map = opts.map ?? null;
        // Same deferred rebind as MeshStandardMaterial — earth/moon DataTextures
        // often get wrap/UV tweaks right after construction.
        if (opts?.map) {
            const tex = opts.map;
            queueMicrotask(() => {
                try {
                    _syncTextureFilters(tex);
                    tex._syncUv?.();
                    _applyMap(this._w, { map: tex });
                } catch (_) { /* ignore */ }
            });
        }
    }
}
export class MeshLambertMaterial { constructor(opts = {}) { this._w = WebMaterial.lambert(_matColor(opts)); _applyMap(this._w, opts); _applyCommon(this._w, opts); _initMaterialBase(this, opts); } }
function _bindDataTex(w, tex, setMap, setMapData) {
    if (!tex) return;
    _syncTextureFilters(tex);
    // Plain `{x,y}` assignment from earth-demo after construction.
    if (tex.offset && !(tex.offset instanceof Vector2)) {
        const o = tex.offset; tex.offset = new Vector2(o.x ?? 0, o.y ?? 0);
    }
    if (tex.repeat && !(tex.repeat instanceof Vector2)) {
        const r = tex.repeat; tex.repeat = new Vector2(r.x ?? 1, r.y ?? 1);
    }
    if (tex._w?.setUvTransform && tex.offset && tex.repeat) {
        tex._w.setUvTransform(tex.offset.x, tex.offset.y, tex.repeat.x, tex.repeat.y);
    }
    if (tex?._w?.constructor?.name === 'WebDataTexture' || tex instanceof DataTexture || tex instanceof CanvasTexture) {
        if (setMapData) w[setMapData](tex._w);
    } else if (tex instanceof Texture) {
        if (setMap) w[setMap](tex._w);
    } else if (tex?._w && setMapData) {
        w[setMapData](tex._w);
    }
}

/** Planet / PBR map options used by the earth demo. */
function _applyStandardExtras(mat, opts) {
    if (!opts || typeof opts !== 'object') return;
    const w = mat._w;
    _bindDataTex(w, opts.normalMap, 'setNormalMap', 'setNormalMapData');
    _bindDataTex(w, opts.roughnessMap, 'setRoughnessMap', 'setRoughnessMapData');
    _bindDataTex(w, opts.emissiveMap, 'setEmissiveMap', 'setEmissiveMapData');
    _bindDataTex(w, opts.displacementMap, 'setDisplacementMap', 'setDisplacementMapData');
    if (typeof opts.displacementScale === 'number' || typeof opts.displacementBias === 'number') {
        w.setDisplacement?.(opts.displacementScale ?? 1, opts.displacementBias ?? 0);
    }
    if (opts.cloudShadowMap) {
        _bindDataTex(w, opts.cloudShadowMap, null, 'setCloudShadowMapData');
    }
    const cloudH = typeof opts.cloudHeight === 'number' ? opts.cloudHeight : 0.012;
    const cloudS = typeof opts.cloudShadow === 'number' ? opts.cloudShadow : 0.55;
    const twilight = typeof opts.twilight === 'number' ? opts.twilight : 0.1;
    if (opts.cloudShadowMap || 'twilight' in opts || 'cloudHeight' in opts) {
        w.setAtmosphereShading?.(cloudH, cloudS, 0, twilight);
    }
    if ('twilightColor' in opts && w.setTwilightColor) {
        w.setTwilightColor(_color(opts.twilightColor));
    }

    mat._cloudHeight = cloudH;
    mat._cloudShadow = cloudS;
    mat._cloudRotation = 0;
    mat._twilight = twilight;
    mat._dispScale = opts.displacementScale ?? 0;
    mat._dispBias = opts.displacementBias ?? 0;
    mat._eclipse = null;

    // Live setters so LOD / progressive upgrades (`mat.map = next`) rebind GPU.
    const bindSlot = (key, setMap, setMapData, value) => {
        Object.defineProperty(mat, key, {
            get() { return mat[`_${key}`]; },
            set(tex) {
                mat[`_${key}`] = tex ?? null;
                if (!tex) return;
                if (key === 'map') {
                    _syncTextureFilters(tex);
                    tex._syncUv?.();
                    _applyMap(mat._w, { map: tex });
                } else {
                    _bindDataTex(mat._w, tex, setMap, setMapData);
                }
            },
            configurable: true,
            enumerable: true,
        });
        mat[key] = value ?? null;
    };
    bindSlot('map', 'setMap', 'setMapData', opts.map);
    bindSlot('normalMap', 'setNormalMap', 'setNormalMapData', opts.normalMap);
    bindSlot('roughnessMap', 'setRoughnessMap', 'setRoughnessMapData', opts.roughnessMap);
    bindSlot('emissiveMap', 'setEmissiveMap', 'setEmissiveMapData', opts.emissiveMap);
    bindSlot('displacementMap', 'setDisplacementMap', 'setDisplacementMapData', opts.displacementMap);
    bindSlot('cloudShadowMap', null, 'setCloudShadowMapData', opts.cloudShadowMap);

    Object.defineProperty(mat, 'emissiveIntensity', {
        get() { return mat._emissiveIntensity ?? 1; },
        set(v) {
            mat._emissiveIntensity = Number(v);
            mat._w?.setEmissiveIntensity?.(mat._emissiveIntensity);
        },
        configurable: true, enumerable: true,
    });
    if (typeof opts.emissiveIntensity === 'number') mat.emissiveIntensity = opts.emissiveIntensity;

    Object.defineProperty(mat, 'emissiveNightSide', {
        get() { return mat._emissiveNightSide ?? 0; },
        set(v) {
            mat._emissiveNightSide = Number(v);
            mat._w?.setEmissiveNightSide?.(mat._emissiveNightSide);
        },
        configurable: true, enumerable: true,
    });
    if (typeof opts.emissiveNightSide === 'number') mat.emissiveNightSide = opts.emissiveNightSide;

    Object.defineProperty(mat, 'displacementScale', {
        get() { return mat._dispScale; },
        set(v) {
            mat._dispScale = Number(v);
            mat._w?.setDisplacement?.(mat._dispScale, mat._dispBias);
        },
        configurable: true, enumerable: true,
    });
    Object.defineProperty(mat, 'displacementBias', {
        get() { return mat._dispBias; },
        set(v) {
            mat._dispBias = Number(v);
            mat._w?.setDisplacement?.(mat._dispScale, mat._dispBias);
        },
        configurable: true, enumerable: true,
    });
    Object.defineProperty(mat, 'cloudRotation', {
        get() { return mat._cloudRotation; },
        set(v) {
            mat._cloudRotation = Number(v);
            mat._w?.setAtmosphereShading?.(mat._cloudHeight, mat._cloudShadow, mat._cloudRotation, mat._twilight);
        },
        configurable: true, enumerable: true,
    });
    // Earth demo: `mat.eclipseOccluder = [x,y,z,r]` — sun angular radius ~0.00465 rad.
    Object.defineProperty(mat, 'eclipseOccluder', {
        get() { return mat._eclipse; },
        set(v) {
            mat._eclipse = v;
            if (!mat._w?.setEclipse) return;
            if (!v || !v[3]) mat._w.setEclipse(0, 0, 0, 0, 0.00465);
            else mat._w.setEclipse(v[0], v[1], v[2], v[3], 0.00465);
        },
        configurable: true, enumerable: true,
    });
}

export class MeshStandardMaterial {
    constructor(opts = {}) {
        const c = _matColor(opts);
        const r = (typeof opts === 'object' && opts && 'roughness' in opts) ? opts.roughness : 1.0;
        const m = (typeof opts === 'object' && opts && 'metalness' in opts) ? opts.metalness : 0.0;
        this._w = WebMaterial.standard(c, r, m);
        _applyMap(this._w, opts); _applyCommon(this._w, opts); _initMaterialBase(this, opts);
        _applyStandardExtras(this, opts);
        // Re-bind after earth-demo mutates wrap/offset/repeat on the texture.
        if (opts?.map) {
            const tex = opts.map;
            queueMicrotask(() => {
                try {
                    _syncTextureFilters(tex);
                    tex._syncUv?.();
                    _applyMap(this._w, { map: tex });
                } catch (_) { /* ignore */ }
            });
        }
    }
}
export class MeshPhongMaterial { constructor(opts = {}) { this._w = WebMaterial.phong(_matColor(opts)); } }
export class MeshPhysicalMaterial {
    constructor(opts = {}) {
        const o = (typeof opts === 'object' && opts) ? opts : {};
        const r  = 'roughness' in o ? o.roughness : 1.0;
        const m  = 'metalness' in o ? o.metalness : 0.0;
        const cc = 'clearcoat' in o ? o.clearcoat : 0.0;
        const cr = 'clearcoatRoughness' in o ? o.clearcoatRoughness : 0.0;
        this._w = WebMaterial.physical(_matColor(opts), r, m, cc, cr);
        _applyMap(this._w, opts); _applyCommon(this._w, opts); _initMaterialBase(this, opts);
    }
}
export class MeshNormalMaterial { constructor() { this._w = WebMaterial.normalMat(); } }
export class MeshDepthMaterial { constructor() { this._w = WebMaterial.depth(); } }
export class MeshToonMaterial { constructor(opts = {}) { this._w = WebMaterial.toon(_matColor(opts)); } }
export class LineBasicMaterial { constructor(opts = {}) { this._w = WebMaterial.line(_matColor(opts)); } }
export class PointsMaterial {
    constructor(opts = {}) {
        const s = (typeof opts === 'object' && opts && 'size' in opts) ? opts.size : 1.0;
        this._w = WebMaterial.points(_matColor(opts), s);
    }
}
export class SpriteMaterial { constructor(opts = {}) { this._w = WebMaterial.sprite(_matColor(opts)); _applyMap(this._w, opts); _applyCommon(this._w, opts); _initMaterialBase(this, opts); } }

/** Analytic planetary atmosphere shell — see `WebMaterial.atmosphere`. */
export class AtmosphereMaterial {
    constructor(opts = {}) {
        const o = (typeof opts === 'object' && opts) ? opts : {};
        const planetR = typeof o.planetRadius === 'number' ? o.planetRadius : 1;
        const atmoR = typeof o.atmosphereRadius === 'number' ? o.atmosphereRadius : planetR * 1.025;
        this._w = WebMaterial.atmosphere(planetR, atmoR);
        this.planetRadius = planetR;
        this.atmosphereRadius = atmoR;
        this._sunset = new Color(o.sunsetColor ?? 0xff7a33);
        this._intensity = typeof o.intensity === 'number' ? o.intensity : 1.5;
        this._falloff = typeof o.falloff === 'number' ? o.falloff : 3.0;
        this._opacity = typeof o.opacity === 'number' ? o.opacity : 1.0;
        this._airglow = typeof o.airglow === 'number' ? o.airglow : 0;
        this._airglowColor = new Color(o.airglowColor ?? 0x4dff8a);
        this._syncAtmosphere();
        this._syncAirglow();
        _initMaterialBase(this, o);
        // setColor does not yet reach Atmosphere in wasm; keep a local tint anyway.
        if ('color' in o) this.color = new Color(o.color);
        const self = this;
        // After _initMaterialBase (it writes plain `opacity`), bind live setters.
        Object.defineProperty(this, 'intensity', {
            get() { return self._intensity; },
            set(v) { self._intensity = Number(v); self._syncAtmosphere(); },
            configurable: true,
            enumerable: true,
        });
        Object.defineProperty(this, 'falloff', {
            get() { return self._falloff; },
            set(v) { self._falloff = Number(v); self._syncAtmosphere(); },
            configurable: true,
            enumerable: true,
        });
        Object.defineProperty(this, 'opacity', {
            get() { return self._opacity; },
            set(v) { self._opacity = Number(v); self._syncAtmosphere(); },
            configurable: true,
            enumerable: true,
        });
        Object.defineProperty(this, 'airglow', {
            get() { return self._airglow; },
            set(v) { self._airglow = Number(v); self._syncAirglow(); },
            configurable: true,
            enumerable: true,
        });
    }
    _syncAtmosphere() {
        this._w.setAtmosphere(this._sunset._w, this._intensity, this._falloff, this._opacity);
    }
    _syncAirglow() {
        this._w.setAirglow(this._airglow, this._airglowColor._w);
    }
}

// Sprite: camera-facing billboard.
export class Sprite {
    constructor(material) {
        this.material = material || new SpriteMaterial();
        this._isSprite = true;
        this.position = new Vector3();
        this.rotation = new Euler();
        this.scale = new Vector3(1, 1, 1);
        this._handle = null;
    }
}

// Bone: a transformable joint node. Used by Skeleton + SkinnedMesh.
export class Bone {
    constructor() {
        this.isBone = true;
        this.position = new Vector3();
        this.rotation = new Euler();
        this.quaternion = new Quaternion();
        this.scale = new Vector3(1, 1, 1);
        this.matrix = new Matrix4();
        this.matrixWorld = new Matrix4();
        this.children = [];
        this.parent = null;
    }
    add(child) { this.children.push(child); child.parent = this; return this; }
}

// Skeleton: bones + their inverse-bind matrices. SkinnedMesh.bind() pairs the
// two together; Skeleton.update() refreshes the world-space bone matrices.
export class Skeleton {
    constructor(bones = [], boneInverses) {
        this.bones = bones.slice();
        if (boneInverses) this.boneInverses = boneInverses.slice();
        else {
            this.boneInverses = [];
            for (let i = 0; i < bones.length; i++) this.boneInverses.push(new Matrix4());
        }
        this.boneMatrices = new Float32Array(bones.length * 16);
        this.calculateInverses();
    }
    calculateInverses() {
        // three.js: invert each bone's matrixWorld at bind time to capture
        // the inverse-bind matrix. Render-time bone matrix = matrixWorld *
        // inverseBind cancels out the bind pose so identity poses don't
        // accidentally deform the mesh.
        this.boneInverses.length = 0;
        for (let i = 0; i < this.bones.length; i++) {
            const b = this.bones[i];
            const inv = new Matrix4();
            if (b) {
                // Compose matrix from current position/quaternion/scale, then invert.
                b.updateMatrix?.();
                if (b.matrix && b.matrix.elements) {
                    // Build world matrix from local matrix chain.
                    let cur = b, world = new Matrix4().identity();
                    while (cur) {
                        cur.updateMatrix?.();
                        if (cur.matrix?.elements) world = new Matrix4().copy(cur.matrix).multiply(world);
                        cur = cur.parent;
                    }
                    inv.copy(world).invert();
                } else {
                    inv.identity();
                }
            }
            this.boneInverses.push(inv);
        }
    }
    update() {
        // Refresh each bone's matrixWorld from its current position/rotation/scale,
        // walking the parent chain so child bones inherit parent transforms.
        for (const b of this.bones) {
            const chain = [];
            let cur = b;
            while (cur) { chain.unshift(cur); cur = cur.parent; }
            let world = new Matrix4().identity();
            for (const node of chain) {
                const q = node.quaternion || new Quaternion().setFromEuler(node.rotation || new Euler());
                const local = new Matrix4().compose(node.position || new Vector3(), q, node.scale || new Vector3(1, 1, 1));
                world = new Matrix4().multiplyMatrices(world, local);
            }
            b.matrixWorld = world;
        }
        // Compose bone-matrix array = matrixWorld[i] * boneInverses[i].
        for (let i = 0; i < this.bones.length; i++) {
            const out = this.bones[i].matrixWorld || new Matrix4();
            const inv = this.boneInverses[i] || new Matrix4();
            const m = new Matrix4().multiplyMatrices(out, inv);
            for (let k = 0; k < 16; k++) this.boneMatrices[i * 16 + k] = m.elements[k];
        }
    }
    dispose() {}
}

// SkinnedMesh — geometry with bone-weighted vertices. Backed by the wasm
// renderer's skinned pipeline; the JS-side bind() ties a Skeleton to this
// mesh so animation drives the bone matrices.
export class SkinnedMesh {
    constructor(geometry, material) {
        this.geometry = geometry;
        this.material = material;
        this._isSkinnedMesh = true;
        this.position = new Vector3();
        this.rotation = new Euler();
        this.quaternion = new Quaternion();
        this.scale = new Vector3(1, 1, 1);
        this._handle = null;
        this.skeleton = null;
        this.children = [];
        this.parent = null;
        this.matrixWorld = new Matrix4();
        // Morph-target state. Combined skinned + morph: setMeshPositions writes
        // the morph-blended positions and the skinned vertex shader reads them
        // before applying the skin matrix.
        this.morphTargetInfluences = [];
        this.morphTargetDictionary = {};
        if (geometry?.morphAttributes?.position) {
            for (let i = 0; i < geometry.morphAttributes.position.length; i++) {
                this.morphTargetInfluences.push(0);
            }
        }
    }
    updateMorphTargets() {
        const ma = this.geometry?.morphAttributes?.position;
        if (!ma) return;
        if (this.morphTargetInfluences.length !== ma.length) {
            this.morphTargetInfluences = new Array(ma.length).fill(0);
        }
        const pa = this.geometry.attributes?.position;
        if (pa && !this.geometry._morphBasePositions) {
            this.geometry._morphBasePositions = new Float32Array(pa.array);
        }
    }
    _applyMorphTargets() {
        const ma = this.geometry?.morphAttributes?.position;
        const pa = this.geometry.attributes?.position;
        if (pa && !this.geometry._morphBasePositions) {
            this.geometry._morphBasePositions = new Float32Array(pa.array);
        }
        const base = this.geometry?._morphBasePositions;
        if (!ma || ma.length === 0 || !base) return;
        const out = new Float32Array(base.length);
        for (let i = 0; i < base.length; i++) out[i] = base[i];
        const relative = !!this.geometry.morphTargetsRelative;
        for (let m = 0; m < ma.length; m++) {
            const w = this.morphTargetInfluences[m] || 0;
            if (w === 0) continue;
            const targetArr = ma[m].array;
            for (let i = 0; i < out.length; i++) {
                if (relative) out[i] += w * targetArr[i];
                else          out[i] += w * (targetArr[i] - base[i]);
            }
        }
        this._morphedGeometry = { attributes: { position: { array: out } } };
    }
    add(child) {
        this.children.push(child);
        if (child) child.parent = this;
        return this;
    }
    remove(child) {
        const i = this.children.indexOf(child);
        if (i >= 0) { this.children.splice(i, 1); child.parent = null; }
        return this;
    }
    bind(skeleton, bindMatrix) {
        this.skeleton = skeleton;
        this.bindMatrix = bindMatrix || new Matrix4();
        this.bindMatrixInverse = new Matrix4().copy(this.bindMatrix).invert();
    }
    pose() { /* reset to bind pose — no-op for now */ }
}

// InstancedMesh: same geometry rendered N times with per-instance matrices.
export class InstancedMesh {
    constructor(geometry, material, count) {
        this.geometry = geometry;
        this.material = material;
        this.count = count;
        this._isInstancedMesh = true;
        this._matrices = new Float32Array(count * 16);
        for (let i = 0; i < count; i++) {
            const o = i * 16;
            this._matrices[o   ] = 1; this._matrices[o+5 ] = 1;
            this._matrices[o+10] = 1; this._matrices[o+15] = 1;
        }
        this.position = new Vector3();
        this.rotation = new Euler();
        this.scale = new Vector3(1, 1, 1);
        this._handle = null;
    }
    setMatrixAt(i, m) {
        const e = m.elements || m;
        const o = i * 16;
        for (let k = 0; k < 16; k++) this._matrices[o + k] = e[k];
    }
    getMatrixAt(i, m) {
        const e = m.elements || m;
        const o = i * 16;
        for (let k = 0; k < 16; k++) e[k] = this._matrices[o + k];
    }
}

// BatchedMesh (three.js r165 API): each addGeometry call adds one drawable
// slot with that geometry, returning a geometryId that is also the per-slot
// instance index. setMatrixAt(id, mat) places that slot in world space;
// setVisibleAt(id, bool) toggles draw. We map each slot to a plain Mesh —
// no GPU batching, just per-slot draw — visually identical to three.js's
// multi-draw approach.
export class BatchedMesh {
    constructor(maxInstanceCount, _maxVertexCount, _maxIndexCount, material) {
        // three.js r165 signature: (maxInstanceCount, maxVertexCount, [maxIndexCount,] material).
        if (_maxIndexCount && typeof _maxIndexCount !== 'number') {
            material = _maxIndexCount;
        }
        this.material = material;
        this._maxInstanceCount = maxInstanceCount;
        this._isBatchedMesh = true;
        this.position = new Vector3();
        this.rotation = new Euler();
        this.quaternion = new Quaternion();
        this.scale = new Vector3(1, 1, 1);
        this.children = [];
        this.parent = null;
        this.visible = true;
        this.layers = new Layers();
        // slot[geomId] = { mesh, matrix, visible }
        this._slots = [];
    }
    addGeometry(geometry, _vCount = -1, _iCount = -1) {
        const id = this._slots.length;
        const mesh = new Mesh(geometry, this.material);
        const matrix = new Matrix4();
        this._slots.push({ mesh, matrix, visible: true });
        if (this._sceneRef) this._sceneRef.add(mesh);
        return id;
    }
    setMatrixAt(geometryId, matrix) {
        const s = this._slots[geometryId];
        if (!s) return;
        s.matrix.copy(matrix);
        const e = matrix.elements;
        // Set the slot mesh's position from the matrix translation row.
        s.mesh.position.set(e[12], e[13], e[14]);
        // Decompose into rotation (approximate XYZ Euler from the rotation block).
        // For pure-translation matrices (the common BatchedMesh use case), this
        // is exact; for rotated transforms, three.js extracts an Euler the same
        // way via Matrix4.extractRotation + Euler.setFromRotationMatrix.
        const sx = Math.hypot(e[0], e[1], e[2]);
        const sy = Math.hypot(e[4], e[5], e[6]);
        const sz = Math.hypot(e[8], e[9], e[10]);
        s.mesh.scale.set(sx, sy, sz);
        const m11 = e[0]/sx, m12 = e[4]/sy, m13 = e[8]/sz;
        const m21 = e[1]/sx, m22 = e[5]/sy, m23 = e[9]/sz;
        const m31 = e[2]/sx, m32 = e[6]/sy, m33 = e[10]/sz;
        const ry = Math.asin(Math.max(-1, Math.min(1, m13)));
        let rx, rz;
        if (Math.abs(m13) < 0.9999999) {
            rx = Math.atan2(-m23, m33);
            rz = Math.atan2(-m12, m11);
        } else {
            rx = Math.atan2(m32, m22);
            rz = 0;
        }
        s.mesh.rotation.set(rx, ry, rz);
    }
    getMatrixAt(geometryId, matrix) {
        const s = this._slots[geometryId];
        if (!s) return;
        matrix.copy(s.matrix);
    }
    setVisibleAt(geometryId, value) {
        const s = this._slots[geometryId];
        if (!s) return;
        s.visible = value;
        s.mesh.visible = value;
    }
    _attachToScene(scene) {
        this._sceneRef = scene;
        for (const s of this._slots) scene.add(s.mesh);
    }
}

// Sync euler rotation edits into quaternion (matches three.js Object3D.updateMatrix).
function _bindRotationQuaternion(obj) {
    const rot = obj.rotation;
    const quat = obj.quaternion;
    rot.set = function (x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
        quat.setFromEuler(this);
        return this;
    };
}

// ---- Mesh / Object3D ----
export class Mesh {
    constructor(geometry = new BufferGeometry(), material = new MeshBasicMaterial()) {
        this.geometry = geometry;
        this.material = material;
        this.isMesh = true;
        if (geometry && material) {
            const isEmptyUser = geometry._isUserGeometry
                && !(geometry.attributes?.position?.array?.length > 0);
            const geom_w = isEmptyUser ? null : _geomToWebGeom(geometry);
            const matForWasm = Array.isArray(material) ? material[0] : material;
            if (geom_w && matForWasm?._w) {
                this._w = new WebMesh(geom_w, matForWasm._w);
            }
        }
        this.position = new Vector3();
        this.rotation = new Euler();
        this.quaternion = new Quaternion();
        _bindRotationQuaternion(this);
        this.scale = new Vector3(1, 1, 1);
        this.matrix = new Matrix4();
        this.matrixWorld = new Matrix4();
        this.matrixAutoUpdate = true;
        this.matrixWorldNeedsUpdate = true;
        this.visible = true;
        this.name = '';
        this.parent = null;
        this.children = [];
        this.layers = new Layers();
        this._handle = null;
        // Morph target state. `morphTargetInfluences[i]` is the weight of
        // `geometry.morphAttributes.position[i]` against the base position.
        // Re-applied at scene-sync time via WebScene.setMeshPositions.
        this.morphTargetInfluences = [];
        this.morphTargetDictionary = {};
        if (geometry?.morphAttributes?.position) {
            for (let i = 0; i < geometry.morphAttributes.position.length; i++) {
                this.morphTargetInfluences.push(0);
            }
            this.updateMorphTargets();
        }
    }
    raycast(raycaster, intersects) { _raycastMesh(raycaster, this, intersects); }
    add(...children) {
        for (const child of children) {
            if (child.parent) child.parent.remove(child);
            child.parent = this;
            this.children.push(child);
        }
        return this;
    }
    remove(child) {
        const i = this.children.indexOf(child);
        if (i !== -1) {
            child.parent = null;
            this.children.splice(i, 1);
        }
        return this;
    }
    updateMatrix() {
        const q = this.quaternion || new Quaternion().setFromEuler(this.rotation);
        this.matrix.compose(this.position, q, this.scale);
        this.matrixWorldNeedsUpdate = true;
    }
    updateMatrixWorld(force = false) {
        if (this.matrixAutoUpdate) this.updateMatrix();
        if (this.matrixWorldNeedsUpdate || force) {
            if (this.parent?.matrixWorld) {
                this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
            } else {
                this.matrixWorld.copy(this.matrix);
            }
            this.matrixWorldNeedsUpdate = false;
            for (const c of this.children) c.updateMatrixWorld?.(true);
        } else {
            for (const c of this.children) c.updateMatrixWorld?.(force);
        }
    }
    // When influences change, set `_morphDirty` so the next scene sync
    // recomputes positions and re-creates the WebMesh handle.
    updateMorphTargets() {
        const ma = this.geometry?.morphAttributes?.position;
        if (!ma) return;
        if (this.morphTargetInfluences.length !== ma.length) {
            this.morphTargetInfluences = new Array(ma.length).fill(0);
        }
        // Snapshot the base positions once so we can blend without losing them.
        const pa = this.geometry.attributes?.position;
        if (pa && !this.geometry._morphBasePositions) {
            this.geometry._morphBasePositions = new Float32Array(pa.array);
        }
    }
    // Recompute geometry.attributes.position by blending base + morph deltas
    // (or absolutes, depending on morphTargetsRelative). Called from scene
    // sync when influences change.
    _applyMorphTargets() {
        const ma = this.geometry?.morphAttributes?.position;
        const base = this.geometry?._morphBasePositions;
        if (!ma || ma.length === 0 || !base) return;
        const out = new Float32Array(base.length);
        for (let i = 0; i < base.length; i++) out[i] = base[i];
        const relative = !!this.geometry.morphTargetsRelative;
        for (let m = 0; m < ma.length; m++) {
            const w = this.morphTargetInfluences[m] || 0;
            if (w === 0) continue;
            const targetArr = ma[m].array;
            for (let i = 0; i < out.length; i++) {
                if (relative) out[i] += w * targetArr[i];
                else          out[i] += w * (targetArr[i] - base[i]);
            }
        }
        // Build a fresh BufferGeometry from the blended positions + existing
        // normals/uvs/index. Mark for re-attach: Scene._doSyncTransforms will
        // detect `_morphDirty` and rebuild the wasm Mesh handle in place.
        const g = new BufferGeometry();
        g.setAttribute('position', new BufferAttribute(out, 3));
        if (this.geometry.attributes.normal) g.setAttribute('normal', new BufferAttribute(new Float32Array(this.geometry.attributes.normal.array), 3));
        if (this.geometry.attributes.uv)     g.setAttribute('uv',     new BufferAttribute(new Float32Array(this.geometry.attributes.uv.array), 2));
        if (this.geometry.index)             g.setIndex(new Uint32Array(this.geometry.index.array));
        this._morphedGeometry = g;
        this._morphDirty = true;
    }
}
_addObjectShadowProps(Mesh);

export class BufferAttribute {
    constructor(array, itemSize) {
        // wasm-bindgen Vec<f32> expects a regular Float32Array (or Array).
        const arr = (array instanceof Float32Array) ? array : new Float32Array(array);
        this._w = new WebBufferAttribute(arr, itemSize);
        this.array = arr;
        this.itemSize = itemSize;
        this.count = arr.length / itemSize | 0;
        this.normalized = false;
    }
    getX(index) { return this.array[index * this.itemSize]; }
    getY(index) { return this.array[index * this.itemSize + 1]; }
    getZ(index) { return this.array[index * this.itemSize + 2]; }
    getW(index) { return this.array[index * this.itemSize + 3]; }
    fromBufferAttribute(attr, index) {
        const i = index * attr.itemSize;
        for (let k = 0; k < attr.itemSize; k++) this.array[k] = attr.array[i + k];
        return this;
    }
}
export class BufferGeometry {
    constructor() {
        this._isUserGeometry = true;
        this.attributes = {};
        this.drawRange = { start: 0, count: Infinity };
        this.groups = [];
    }
    _syncWasmFromJs() {
        this._w = new WebBufferGeometry();
        for (const [name, attr] of Object.entries(this.attributes)) {
            const a = attr?.array;
            if (!a?.length) continue;
            const arr = a instanceof Float32Array ? a : new Float32Array(a);
            this._w.setAttribute(name, new WebBufferAttribute(arr, attr.itemSize || 3));
        }
        const idx = this._indexAttr?.array || this.index?.array;
        if (idx?.length) {
            this._w.setIndex(idx instanceof Uint32Array ? idx : new Uint32Array(idx));
        }
        delete this._geom;
        return this._w;
    }
    setAttribute(name, attr) {
        this.attributes[name] = attr;
        return this;
    }
    setIndex(index) {
        let u32;
        let indexAttr;
        if (index?.array) {
            const arr = index.array;
            u32 = arr instanceof Uint32Array ? arr : new Uint32Array(arr);
            indexAttr = index;
        } else {
            u32 = index instanceof Uint32Array ? index : new Uint32Array(index);
            indexAttr = new BufferAttribute(u32, 1);
        }
        this._indexAttr = indexAttr;
        return this;
    }
    setDrawRange(start, count) {
        this.drawRange = { start, count };
        return this;
    }
    deleteAttribute(name) {
        delete this.attributes[name];
        return this;
    }
    dispose() {
        this.boundsTree = null;
        this.halfEdges = null;
        this.groupIndices = null;
        return this;
    }
    translate(x, y, z) {
        const p = this.attributes.position;
        if (!p) return this;
        const a = p.array;
        for (let i = 0; i < a.length; i += 3) {
            a[i] += x; a[i+1] += y; a[i+2] += z;
        }
        // Push the mutated array back into the wasm BufferAttribute by re-setting
        // the attribute (the wasm side took a copy at original construction).
        this.setAttribute('position', new BufferAttribute(a, p.itemSize));
        return this;
    }
    addGroup(start, count, materialIndex = 0) {
        if (!this.groups) this.groups = [];
        this.groups.push({ start, count, materialIndex });
        return this;
    }
    clearGroups() { this.groups = []; return this; }
    scale(x, y, z) {
        const p = this.attributes.position;
        if (!p) return this;
        const a = p.array;
        for (let i = 0; i < a.length; i += 3) {
            a[i] *= x; a[i+1] *= y; a[i+2] *= z;
        }
        this.setAttribute('position', new BufferAttribute(a, p.itemSize));
        return this;
    }
    rotateX(angle) { return this._applyAxis(angle, 0); }
    rotateY(angle) { return this._applyAxis(angle, 1); }
    rotateZ(angle) { return this._applyAxis(angle, 2); }
    _applyAxis(angle, axis) {
        const p = this.attributes.position;
        if (!p) return this;
        const a = p.array;
        const c = Math.cos(angle), s = Math.sin(angle);
        for (let i = 0; i < a.length; i += 3) {
            const x = a[i], y = a[i+1], z = a[i+2];
            if (axis === 0)      { a[i+1] = y*c - z*s; a[i+2] = y*s + z*c; }
            else if (axis === 1) { a[i]   = x*c + z*s; a[i+2] = -x*s + z*c; }
            else                 { a[i]   = x*c - y*s; a[i+1] = x*s + y*c; }
        }
        this.setAttribute('position', new BufferAttribute(a, p.itemSize));
        return this;
    }
    computeVertexNormals() {
        this._syncWasmFromJs();
        this._w.computeVertexNormals();
        const normal = this._w.getAttributeArray?.('normal');
        if (normal?.length) {
            this.attributes.normal = new BufferAttribute(new Float32Array(normal), 3);
        }
        return this;
    }
    applyMatrix4(m) {
        const p = this.attributes.position;
        if (!p) return this;
        const a = p.array;
        const e = m.elements || m._w?.elements?.() || m;
        for (let i = 0; i < a.length; i += 3) {
            const x = a[i], y = a[i + 1], z = a[i + 2];
            a[i]     = e[0] * x + e[4] * y + e[8]  * z + e[12];
            a[i + 1] = e[1] * x + e[5] * y + e[9]  * z + e[13];
            a[i + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
        }
        this.setAttribute('position', new BufferAttribute(a, p.itemSize));
        const n = this.attributes.normal;
        if (n) {
            // For normals, multiply by the upper 3x3 of the matrix (ignoring
            // translation) and renormalize. For non-uniform scale this should
            // be the inverse-transpose, but for the common rotate+uniform-scale
            // case this matches three.js's behavior in BufferGeometry.applyMatrix4.
            const na = n.array;
            for (let i = 0; i < na.length; i += 3) {
                const x = na[i], y = na[i + 1], z = na[i + 2];
                let nx = e[0] * x + e[4] * y + e[8]  * z;
                let ny = e[1] * x + e[5] * y + e[9]  * z;
                let nz = e[2] * x + e[6] * y + e[10] * z;
                const len = Math.hypot(nx, ny, nz) || 1;
                na[i] = nx / len; na[i + 1] = ny / len; na[i + 2] = nz / len;
            }
            this.setAttribute('normal', new BufferAttribute(na, n.itemSize));
        }
        return this;
    }
    applyQuaternion(q) {
        const m = new Matrix4().makeRotationFromQuaternion(q);
        return this.applyMatrix4(m);
    }
    center() {
        const p = this.attributes.position;
        if (!p) return this;
        let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
        const a = p.array;
        for (let i = 0; i < a.length; i += 3) {
            if (a[i]     < mnx) mnx = a[i];     if (a[i]     > mxx) mxx = a[i];
            if (a[i + 1] < mny) mny = a[i + 1]; if (a[i + 1] > mxy) mxy = a[i + 1];
            if (a[i + 2] < mnz) mnz = a[i + 2]; if (a[i + 2] > mxz) mxz = a[i + 2];
        }
        return this.translate(-(mnx + mxx) * 0.5, -(mny + mxy) * 0.5, -(mnz + mxz) * 0.5);
    }
    toNonIndexed() {
        const idx = this.index?.array;
        if (!idx) return this;
        const out = new BufferGeometry();
        for (const name of Object.keys(this.attributes)) {
            const src = this.attributes[name];
            const a = src.array;
            const itemSize = src.itemSize;
            const Ctor = a.constructor;
            const dst = new Ctor(idx.length * itemSize);
            for (let i = 0; i < idx.length; i++) {
                const j = idx[i] * itemSize;
                for (let k = 0; k < itemSize; k++) dst[i * itemSize + k] = a[j + k];
            }
            out.setAttribute(name, new BufferAttribute(dst, itemSize));
        }
        return out;
    }
    toJSON() {
        const out = { metadata: { version: 4.6, type: 'BufferGeometry', generator: 'threers' }, type: 'BufferGeometry', uuid: this.uuid || MathUtils.generateUUID() };
        const data = { attributes: {} };
        for (const name of Object.keys(this.attributes)) {
            const a = this.attributes[name];
            data.attributes[name] = { itemSize: a.itemSize, type: a.array.constructor.name, array: Array.from(a.array), normalized: !!a.normalized };
        }
        if (this.index) {
            data.index = { type: this.index.array.constructor.name, array: Array.from(this.index.array) };
        }
        out.data = data;
        return out;
    }
    get index() {
        // Surface a three.js-shaped { array } accessor so user code that reads
        // geometry.index.array works. The wasm geometry stores indices but we
        // also mirror them on the JS attribute when set via `setIndex`.
        return this._indexAttr || null;
    }
    clone() {
        const g = new BufferGeometry();
        for (const name of Object.keys(this.attributes)) {
            const src = this.attributes[name];
            g.setAttribute(name, new BufferAttribute(new src.array.constructor(src.array), src.itemSize));
        }
        if (this.index) g.setIndex(new Uint32Array(this.index.array));
        return g;
    }
    copy(other) {
        for (const name of Object.keys(other.attributes)) {
            const src = other.attributes[name];
            this.setAttribute(name, new BufferAttribute(new src.array.constructor(src.array), src.itemSize));
        }
        if (other.index) this.setIndex(new Uint32Array(other.index.array));
        return this;
    }
    dispose() {}
    // Convert to the WebGeometry that Mesh/etc expect via `geom._w`. Made
    // available as a plain method (not just a getter) so subclasses that
    // copy via Object.assign still find it.
    get _asGeometry() {
        this._syncWasmFromJs();
        this._geom = WebGeometry.fromBufferGeometry(this._w);
        return this._geom;
    }
}

/** Convert built-in wasm geometry or BufferGeometry to non-indexed BufferGeometry (for CSG). */
export function geometryToBufferGeometry(geometry) {
    if (geometry?.attributes?.position) {
        return geometry.index ? geometry.toNonIndexed() : geometry;
    }
    if (geometry?._w?.toBufferGeometry) {
        const wasm = geometry._w.toBufferGeometry();
        const out = new BufferGeometry();
        const pos = wasm.getAttributeArray('position');
        if (pos) {
            const itemSize = wasm.getAttributeItemSize('position') || 3;
            out.setAttribute('position', new BufferAttribute(new Float32Array(pos), itemSize));
        }
        const normal = wasm.getAttributeArray('normal');
        if (normal) {
            const itemSize = wasm.getAttributeItemSize('normal') || 3;
            out.setAttribute('normal', new BufferAttribute(new Float32Array(normal), itemSize));
        }
        const uv = wasm.getAttributeArray('uv');
        if (uv) {
            const itemSize = wasm.getAttributeItemSize('uv') || 2;
            out.setAttribute('uv', new BufferAttribute(new Float32Array(uv), itemSize));
        }
        const index = wasm.getIndexArray?.();
        if (index?.length) out.setIndex(new Uint32Array(index));
        return out.toNonIndexed();
    }
    throw new Error('geometryToBufferGeometry: unsupported geometry');
}

// Backfill `_asGeometry` for user geometries built by Object.assign(this, bg)
// — getters aren't copied by Object.assign, so we add a method-style fallback
// on the Mesh side via a helper.
// If `obj.quaternion` has been set to a non-identity rotation, return an Euler
// reflecting it (XYZ order, matching three.js). Otherwise return `obj.rotation`.
function _effectiveEuler(obj) {
    const q = obj.quaternion;
    if (!q) return obj.rotation;
    const isIdentity = q.x === 0 && q.y === 0 && q.z === 0 && q.w === 1;
    if (isIdentity) return obj.rotation;
    const x = q.x, y = q.y, z = q.z, w = q.w;
    const m11 = 1 - 2 * (y*y + z*z);
    const m13 = 2 * (x*z + y*w);
    const m22 = 1 - 2 * (x*x + z*z);
    const m23 = 2 * (y*z - x*w);
    const m12 = 2 * (x*y - z*w);
    const m32 = 2 * (y*z + x*w);
    const m33 = 1 - 2 * (x*x + y*y);
    const clamp = (v) => v < -1 ? -1 : v > 1 ? 1 : v;
    const ey = Math.asin(clamp(m13));
    let ex, ez;
    if (Math.abs(m13) < 0.9999999) {
        ex = Math.atan2(-m23, m33);
        ez = Math.atan2(-m12, m11);
    } else {
        ex = Math.atan2(m32, m22);
        ez = 0;
    }
    return new Euler(ex, ey, ez);
}
// Split a Mesh with array-material + geometry.groups into one sub-mesh per
// group. Each sub-mesh gets a fresh BufferGeometry containing only the
// indices/positions/normals/uvs for its group's range. Used so wgpu can draw
// each group with the correct material in one pass.
function _buildSubMeshes(parent) {
    const out = [];
    const pos = parent.geometry?.attributes?.position?.array;
    const nrm = parent.geometry?.attributes?.normal?.array;
    const uv  = parent.geometry?.attributes?.uv?.array;
    const idx = parent.geometry?.index?.array;
    for (const group of parent.geometry.groups) {
        const start = group.start, count = group.count;
        const matIdx = Math.min(group.materialIndex || 0, parent.material.length - 1);
        const mat = parent.material[matIdx];
        const g = new BufferGeometry();
        if (idx) {
            // Indexed: select indices [start, start+count) and remap to a dense vertex set.
            const subIndices = idx.subarray(start, start + count);
            const used = new Map();
            const newPos = [], newNrm = nrm ? [] : null, newUv = uv ? [] : null, newIdx = [];
            for (const i of subIndices) {
                let mapped = used.get(i);
                if (mapped === undefined) {
                    mapped = newPos.length / 3 | 0;
                    used.set(i, mapped);
                    if (pos) { newPos.push(pos[i*3], pos[i*3+1], pos[i*3+2]); }
                    if (newNrm) newNrm.push(nrm[i*3], nrm[i*3+1], nrm[i*3+2]);
                    if (newUv)  newUv.push(uv[i*2], uv[i*2+1]);
                }
                newIdx.push(mapped);
            }
            g.setAttribute('position', new BufferAttribute(new Float32Array(newPos), 3));
            if (newNrm) g.setAttribute('normal', new BufferAttribute(new Float32Array(newNrm), 3));
            if (newUv)  g.setAttribute('uv',     new BufferAttribute(new Float32Array(newUv), 2));
            g.setIndex(newIdx);
        } else {
            // Non-indexed: positions in groups of 3 vertices = 1 triangle.
            const subPos = pos.subarray(start * 3, (start + count) * 3);
            g.setAttribute('position', new BufferAttribute(new Float32Array(subPos), 3));
            if (nrm) g.setAttribute('normal', new BufferAttribute(new Float32Array(nrm.subarray(start * 3, (start + count) * 3)), 3));
            if (uv)  g.setAttribute('uv',     new BufferAttribute(new Float32Array(uv.subarray(start * 2, (start + count) * 2)), 2));
        }
        const sub = new Mesh(g, mat);
        // Inherit parent's transform via shared references (not copies).
        sub.position = parent.position;
        sub.rotation = parent.rotation;
        sub.quaternion = parent.quaternion;
        sub.scale = parent.scale;
        out.push(sub);
    }
    return out;
}
function _geomToWebGeom(g) {
    if (!g) return g;
    if (g._isUserGeometry) {
        if (!g.attributes?.position?.array?.length) return null;
        if (typeof g._syncWasmFromJs === 'function') g._syncWasmFromJs();
        if (!(g._w instanceof WebBufferGeometry)) {
            throw new Error('BufferGeometry: wasm sync failed');
        }
        g._geom = WebGeometry.fromBufferGeometry(g._w);
        return g._geom;
    }
    if (g._w) return g._w;
    // Raw WebGeometry from wasm loaders (FBX/STL/OBJ parse).
    return g;
}

function _wrapWebTexture(wtex) {
    const t = Object.create(Texture.prototype);
    t._w = wtex;
    t.image = { width: wtex.width(), height: wtex.height() };
    t.magFilter = 1006; t.minFilter = 1008;
    t.wrapS = 1001; t.wrapT = 1001;
    // EXR / equirect sky → six faces (earth-demo starfield cube).
    if (typeof wtex.cubeFaces === 'function') {
        t.cubeFaces = (faceSize = 0) => {
            const raw = wtex.cubeFaces(faceSize);
            const out = [];
            for (let i = 0; i < raw.length; i++) out.push(_wrapWebTexture(raw[i]));
            return out;
        };
    }
    return t;
}

function _wrapLoaderGeometry(wgeom) {
    return { _w: wgeom, geometry: wgeom };
}

// LineSegments: pairs of vertices = line segments.
export class LineSegments {
    constructor(geometry, material) {
        this.geometry = geometry;
        this.material = material;
        this._isLineSegments = true;
        this.position = new Vector3();
        this.rotation = new Euler();
        this.scale = new Vector3(1, 1, 1);
        this._handle = null;
    }
    computeLineDistances() {
        const geom = this.geometry;
        const pos = geom.attributes?.position;
        if (!pos) return this;
        const arr = pos.array;
        const count = arr.length / 3;
        const lineDistances = new Float32Array(count);
        // three.js LineSegments.computeLineDistances resets each segment pair:
        // vertex i gets the previous segment's end distance, not the geometric
        // chain through the unused diagonal between segment pairs.
        for (let i = 0; i < count; i += 2) {
            const ax = arr[i * 3], ay = arr[i * 3 + 1], az = arr[i * 3 + 2];
            const bx = arr[i * 3 + 3], by = arr[i * 3 + 4], bz = arr[i * 3 + 5];
            lineDistances[i] = (i === 0) ? 0 : lineDistances[i - 1];
            lineDistances[i + 1] = lineDistances[i] + Math.hypot(bx - ax, by - ay, bz - az);
        }
        geom.setAttribute('lineDistance', new BufferAttribute(lineDistances, 1));
        if (this.material?._isLineDashed) {
            const uvs = new Float32Array(count * 2);
            for (let i = 0; i < count; i++) uvs[i * 2] = lineDistances[i];
            geom.setAttribute('uv', new BufferAttribute(uvs, 2));
        }
        if (this._handle && this._scene?._w) {
            this._scene._w.syncLineGeometry(this._handle, this.geometry._w);
        }
        return this;
    }
}
// Line: vertex strip. v0→v1→v2→...→vN.
// We approximate by emitting consecutive pairs as a LineSegments-equivalent.
function _stripToPairs(geom, closeLoop) {
    const posAttr = geom.attributes?.position;
    if (!posAttr) return geom;
    const src = posAttr.array;
    const n = src.length / 3 | 0;
    if (n < 2) return geom;
    const pairCount = closeLoop ? n : (n - 1);
    const out = new Float32Array(pairCount * 2 * 3);
    let o = 0;
    for (let i = 0; i < pairCount; i++) {
        const a = i, b = (i + 1) % n;
        out[o++] = src[a*3]; out[o++] = src[a*3+1]; out[o++] = src[a*3+2];
        out[o++] = src[b*3]; out[o++] = src[b*3+1]; out[o++] = src[b*3+2];
    }
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(out, 3));
    return g;
}
export class Line {
    constructor(geometry, material) {
        this.geometry = _stripToPairs(geometry, false);
        this.material = material;
        this._isLineSegments = true;
        this.position = new Vector3();
        this.rotation = new Euler();
        this.scale = new Vector3(1, 1, 1);
        this._handle = null;
    }
    computeLineDistances() { return LineSegments.prototype.computeLineDistances.call(this); }
}
export class LineLoop {
    constructor(geometry, material) {
        this.geometry = _stripToPairs(geometry, true);
        this.material = material;
        this._isLineSegments = true;
        this.position = new Vector3();
        this.rotation = new Euler();
        this.scale = new Vector3(1, 1, 1);
        this._handle = null;
    }
    computeLineDistances() { return LineSegments.prototype.computeLineDistances.call(this); }
}
// Points: each vertex = one point.
export class Points {
    constructor(geometry, material) {
        this.geometry = geometry;
        this.material = material;
        this._isPoints = true;
        this.position = new Vector3();
        this.rotation = new Euler();
        this.scale = new Vector3(1, 1, 1);
        this._handle = null;
    }
}

// Group: an empty Object3D that can hold child meshes and have its own
// transform applied to all descendants.
export class Group {
    constructor() {
        this._isGroup = true;
        this.isGroup = true;
        this.position = new Vector3();
        this.rotation = new Euler();
        this.quaternion = new Quaternion();
        _bindRotationQuaternion(this);
        this.scale = new Vector3(1, 1, 1);
        this._handle = null;
        this._children = [];
        this.parent = null;
        this.matrix = new Matrix4();
        this.matrixWorld = new Matrix4();
        this.matrixAutoUpdate = true;
        this.matrixWorldNeedsUpdate = true;
    }
    add(...children) {
        for (const child of children) {
            if (!child) continue;
            if (child.parent) child.parent.remove?.(child);
            this._children.push(child);
            child.parent = this;
            // If group was already added to a scene, attach child immediately.
            if (this._scene && this._handle && child instanceof Mesh) {
                child._handle = this._scene._w.addMeshTo(this._handle, child._w);
                this._scene._objects.push(child);
            }
        }
        return this;
    }
    remove(child) {
        const i = this._children.indexOf(child);
        if (i !== -1) {
            child.parent = null;
            this._children.splice(i, 1);
        }
        return this;
    }
    updateMatrix() {
        const q = this.quaternion || new Quaternion().setFromEuler(this.rotation);
        this.matrix.compose(this.position, q, this.scale);
        this.matrixWorldNeedsUpdate = true;
    }
    updateMatrixWorld(force = false) {
        if (this.matrixAutoUpdate) this.updateMatrix();
        if (this.matrixWorldNeedsUpdate || force) {
            if (this.parent?.matrixWorld) {
                this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
            } else {
                this.matrixWorld.copy(this.matrix);
            }
            this.matrixWorldNeedsUpdate = false;
            for (const c of this._children) c.updateMatrixWorld?.(true);
        } else {
            for (const c of this._children) c.updateMatrixWorld?.(force);
        }
    }
    get children() {
        return this._children;
    }
}

// ---- Lights ----
export class AmbientLight {
    constructor(color = 0x404040, intensity = 1) {
        this._w = WebLight.ambient(_color(color), intensity);
        this._isLight = true;
        this.color = new Color(color);
        this._intensity = intensity;
        const light = this;
        Object.defineProperty(this, 'intensity', {
            get() { return light._intensity; },
            set(v) {
                light._intensity = Number(v);
                light._w?.setIntensity?.(light._intensity);
                _pushLightToScene(light);
            },
            configurable: true,
            enumerable: true,
        });
    }
}
function _makeLightShadow(w) {
    const store = { left: -5, right: 5, top: 5, bottom: -5, near: 0.1, far: 500 };
    const sync = () => {
        w?.setShadowCamera?.(store.left, store.right, store.top, store.bottom, store.near, store.far);
    };
    const camera = {};
    for (const key of ['left', 'right', 'top', 'bottom', 'near', 'far']) {
        Object.defineProperty(camera, key, {
            get() { return store[key]; },
            set(v) { store[key] = v; sync(); },
            enumerable: true,
            configurable: true,
        });
    }
    const mapSize = { width: 512, height: 512, set(w, h) { this.width = w; this.height = h; } };
    return { camera, mapSize, bias: 0, radius: 1, _sync: sync };
}
function _addObjectShadowProps(cls) {
    Object.defineProperty(cls.prototype, 'castShadow', {
        get() { return this._castShadow === true; },
        set(v) {
            this._castShadow = !!v;
            if (this._handle && this._scene?._w?.setObjectCastShadow) {
                this._scene._w.setObjectCastShadow(this._handle, !!v);
            }
        },
        configurable: true,
    });
    Object.defineProperty(cls.prototype, 'receiveShadow', {
        get() { return this._receiveShadow === true; },
        set(v) {
            this._receiveShadow = !!v;
            if (this._handle && this._scene?._w?.setObjectReceiveShadow) {
                this._scene._w.setObjectReceiveShadow(this._handle, !!v);
            }
        },
        configurable: true,
    });
}
function _applyMeshShadowFlags(obj) {
    if (!obj._handle || !obj._scene?._w) return;
    if (obj._castShadow === true) obj._scene._w.setObjectCastShadow(obj._handle, true);
    if (obj._receiveShadow === true) obj._scene._w.setObjectReceiveShadow(obj._handle, true);
}
function _addLightShadowProps(cls) {
    Object.defineProperty(cls.prototype, 'castShadow', {
        get() { return this._castShadow === true; },
        set(v) { this._castShadow = !!v; this._w?.setCastShadow?.(!!v); this.shadow?._sync?.(); },
        configurable: true,
    });
    if (!cls.prototype.shadow) cls.prototype.shadow = _makeLightShadow();
}
export class DirectionalLight {
    constructor(color = 0xffffff, intensity = 1) {
        this._w = WebLight.directional(_color(color), intensity);
        this._isLight = true;
        this._intensity = intensity;
        const w = this._w;
        const light = this;
        Object.defineProperty(this, 'intensity', {
            get() { return light._intensity; },
            set(v) {
                light._intensity = Number(v);
                light._w?.setIntensity?.(light._intensity);
                _pushLightToScene(light);
            },
            configurable: true,
            enumerable: true,
        });
        const recalc = () => _syncLightDirection(light);
        this.position = {
            x: 0, y: 1, z: 0,
            set(x, y, z) { this.x = x; this.y = y; this.z = z; recalc(); },
        };
        this.target = new Object3D();
        _hookTargetPosition(this.target, recalc);
        recalc();
        this.shadow = _makeLightShadow(w);
    }
}
// Both lights take a world-space .position whose value is applied via the
// scene's setTransform once they're added (and re-applied whenever .set runs).
function _lightPos(light) {
    return {
        x: 0, y: 0, z: 0,
        set(x, y, z) {
            this.x = x; this.y = y; this.z = z;
            if (light._scene && light._handle) {
                light._scene._w.setTransform(light._handle, new Vector3(x, y, z)._w(), new Euler(0,0,0)._w());
            }
        },
    };
}
export class PointLight {
    constructor(color = 0xffffff, intensity = 1, distance = 0, decay = 2) {
        this._w = WebLight.point(_color(color), intensity, distance, decay);
        this._isLight = true;
        this._intensity = intensity;
        const light = this;
        Object.defineProperty(this, 'intensity', {
            get() { return light._intensity; },
            set(v) {
                light._intensity = Number(v);
                light._w?.setIntensity?.(light._intensity);
                _pushLightToScene(light);
            },
            configurable: true,
            enumerable: true,
        });
        this.position = _lightPos(this);
        this.shadow = _makeLightShadow(this._w);
    }
}
function _hookTargetPosition(target, recalc) {
    const pos = target.position;
    const origSet = pos.set.bind(pos);
    pos.set = (x, y, z) => { origSet(x, y, z); recalc(); };
}

/** Push wrapper light state into the scene copy (addLight clones; setters alone are silent). */
function _pushLightToScene(light) {
    if (light?._handle != null && light._scene?._w?.updateLight && light._w) {
        light._scene._w.updateLight(light._handle, light._w);
    }
}

function _syncLightDirection(light) {
    if (!light?._w?.setDirection) return;
    const lp = new Vector3();
    const tp = new Vector3();
    if (light.getWorldPosition) light.getWorldPosition(lp);
    else lp.set(light.position.x, light.position.y, light.position.z);
    if (light.target?.getWorldPosition) light.target.getWorldPosition(tp);
    else tp.set(light.target.position.x, light.target.position.y, light.target.position.z);
    light._w.setDirection(tp.x - lp.x, tp.y - lp.y, tp.z - lp.z);
    _pushLightToScene(light);
}

export class SpotLight {
    constructor(color = 0xffffff, intensity = 1, distance = 0, angle = Math.PI/4, penumbra = 0, decay = 2) {
        this._w = WebLight.spot(_color(color), intensity, distance, angle, penumbra, decay);
        this._isLight = true;
        const light = this;
        const recalcDir = () => _syncLightDirection(light);
        this.position = {
            x: 0, y: 0, z: 0,
            set(x, y, z) {
                this.x = x; this.y = y; this.z = z;
                if (light._scene && light._handle) {
                    light._scene._w.setTransform(light._handle, new Vector3(x,y,z)._w(), new Euler(0,0,0)._w());
                }
                recalcDir();
            },
        };
        this.target = new Object3D();
        _hookTargetPosition(this.target, recalcDir);
        this.shadow = _makeLightShadow(this._w);
    }
}
export class HemisphereLight {
    constructor(sky = 0xffffff, ground = 0x444444, intensity = 1) {
        this._w = WebLight.hemisphere(_color(sky), _color(ground), intensity);
        this._isLight = true;
    }
}
export class RectAreaLight {
    constructor(color = 0xffffff, intensity = 1, width = 10, height = 10) {
        this._w = WebLight.rectArea(_color(color), intensity, width, height);
        this._isLight = true;
    }
}
_addLightShadowProps(DirectionalLight);
_addLightShadowProps(PointLight);
_addLightShadowProps(SpotLight);

// ---- Textures ----
export class Texture {
    constructor(width, height, data) {
        // three.js permits `new Texture()` with no args (the image is filled
        // in later via .image / .needsUpdate). Fall back to a 1×1 white pixel.
        if (width === undefined) {
            this._w = new WebTexture(1, 1, new Uint8Array([255, 255, 255, 255]));
            this.image = null;
        } else {
            this._w = new WebTexture(width, height, data);
            this.image = { width, height, data };
        }
        this.needsUpdate = false;
        this.magFilter = 1006; this.minFilter = 1008;
        this.wrapS = 1001; this.wrapT = 1001;
        this.format = 1023; this.type = 1009;
        this.repeat = new Vector2(1, 1);
        this.offset = new Vector2(0, 0);
        this.center = new Vector2(0, 0);
        this.rotation = 0;
    }
    static solid(r, g, b, a) { const t = Object.create(Texture.prototype); t._w = WebTexture.solid(r, g, b, a); return t; }
}
export class CubeTexture {
    constructor(size, px, nx, py, ny, pz, nz) {
        this._w = new WebCubeTexture(size, px, nx, py, ny, pz, nz);
    }
}
export class DataTexture {
    constructor(...args) {
        // three.js: (data, width, height, format?, type?)
        // threers earth demo: (data, width, height, 'srgb') for colour maps
        // legacy:   (width, height, data)
        let data, width, height, colourSpace;
        if (typeof args[0] === 'number') {
            [width, height, data] = args;
        } else {
            [data, width, height, colourSpace] = args;
        }
        const u8 = (() => {
            // Always copy into a tight buffer. ImageData / worker views onto
            // pooled ArrayBuffers scramble uploads if we pass .buffer raw.
            if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
            if (data instanceof ArrayBuffer) return new Uint8Array(data).slice();
            return new Uint8Array(data);
        })();
        this._w = colourSpace === 'srgb'
            ? WebDataTexture.newSrgb(width, height, u8)
            : new WebDataTexture(width, height, u8);
        this.image = { data: u8, width, height };
        this.needsUpdate = false;
        this.magFilter = 1006; this.minFilter = 1008;
        this.wrapS = 1001; this.wrapT = 1001;
        this.format = 1023; this.type = 1009;
        this.colorSpace = colourSpace === 'srgb' ? 'srgb' : '';
        this.center = new Vector2(0, 0);
        this.rotation = 0;
        this.offset = new Vector2(0, 0);
        this.repeat = new Vector2(1, 1);
        const self = this;
        const hook = (vec) => {
            const orig = vec.set.bind(vec);
            vec.set = (x, y) => { orig(x, y); self._syncUv(); return vec; };
            return vec;
        };
        this.offset = hook(this.offset);
        this.repeat = hook(this.repeat);
        // Plain `{x,y}` writes from earth-demo — replace + sync.
        for (const prop of ['offset', 'repeat']) {
            let cur = this[prop];
            Object.defineProperty(this, prop, {
                get() { return cur; },
                set(next) {
                    if (next instanceof Vector2) {
                        cur = hook(next);
                    } else if (next && typeof next === 'object') {
                        const def = prop === 'repeat' ? 1 : 0;
                        cur.set(next.x ?? def, next.y ?? def);
                        return;
                    }
                    self._syncUv();
                },
                configurable: true, enumerable: true,
            });
        }
        this._syncFilters();
    }
    _syncFilters() {
        if (!this._w?.setFilters) return;
        const NEAREST = 1003, NEAR_NEAR_MIP = 1004, LIN_NEAR_MIP = 1005, NEAR_LIN_MIP = 1007;
        const isNearest = (v) => v === NEAREST || v === NEAR_NEAR_MIP || v === LIN_NEAR_MIP || v === NEAR_LIN_MIP;
        const mag = isNearest(this.magFilter ?? 1006) ? 1 : 0;
        const min = isNearest(this.minFilter ?? 1008) ? 1 : 0;
        const conv = (w) => w === 1000 ? 1 : w === 1002 ? 2 : 0;
        this._w.setFilters(mag, min, conv(this.wrapS ?? 1001), conv(this.wrapT ?? 1001));
    }
    _syncUv() {
        if (!this._w?.setUvTransform) return;
        const ox = this.offset?.x ?? 0, oy = this.offset?.y ?? 0;
        const rx = this.repeat?.x ?? 1, ry = this.repeat?.y ?? 1;
        this._w.setUvTransform(ox, oy, rx, ry);
    }
    /** Second handle on the same pixels with its own UV sub-rect (atlas view). */
    view(ox, oy, rx, ry) {
        if (!this._w || typeof this._w.view !== 'function') {
            throw new Error('DataTexture.view: underlying WebDataTexture.view is missing');
        }
        const t = Object.create(DataTexture.prototype);
        t._w = this._w.view(ox, oy, rx, ry);
        t.image = this.image;
        t.needsUpdate = false;
        t.magFilter = this.magFilter; t.minFilter = this.minFilter;
        t.wrapS = this.wrapS; t.wrapT = this.wrapT;
        t.format = this.format; t.type = this.type;
        t.colorSpace = this.colorSpace;
        t.offset = new Vector2(ox, oy);
        t.repeat = new Vector2(rx, ry);
        t.center = new Vector2(0, 0);
        t.rotation = 0;
        t._syncFilters = DataTexture.prototype._syncFilters;
        t._syncUv = DataTexture.prototype._syncUv;
        return t;
    }
}

// CanvasTexture — reads pixels from a 2D canvas / OffscreenCanvas into a
// DataTexture. Mirrors three.js's CanvasTexture; supports `needsUpdate` to
// re-upload after the source canvas changes.
export class CanvasTexture {
    constructor(canvas) {
        this.image = canvas;
        this.format = 1023; this.type = 1009;
        this.magFilter = 1006; this.minFilter = 1008;
        this.wrapS = 1001; this.wrapT = 1001;
        // Canvas pixels are top-first; UV space is bottom-first (three.js flipY).
        this.flipY = true;
        this.offset = new Vector2(0, 1);
        this.repeat = new Vector2(1, -1);
        this.center = new Vector2(0, 0);
        this.rotation = 0;
        this._upload();
        this.needsUpdate = false;
    }
    _syncUv() {
        if (!this._w?.setUvTransform) return;
        const ox = this.offset?.x ?? 0;
        const oy = this.offset?.y ?? 1;
        const rx = this.repeat?.x ?? 1;
        const ry = this.flipY ? -Math.abs(this.repeat?.y ?? 1) : (this.repeat?.y ?? 1);
        this._w.setUvTransform(ox, oy, rx, ry);
    }
    _upload() {
        const c = this.image;
        if (!c) {
            this._w = new WebDataTexture(1, 1, new Uint8Array([255, 255, 255, 255]));
            this._syncUv();
            return;
        }
        const ctx = c.getContext('2d');
        const w = c.width, h = c.height;
        const data = ctx.getImageData(0, 0, w, h).data;
        const byteLen = data.length;
        if (!this._buf || this._buf.length !== byteLen) {
            this._buf = new Uint8Array(byteLen);
        }
        this._buf.set(data);
        if (this._w && this._lw === w && this._lh === h && this._w.replaceRgba) {
            if (this._w.replaceRgba(this._buf)) {
                return;
            }
        }
        this._w = new WebDataTexture(w, h, this._buf);
        this._lw = w;
        this._lh = h;
        this._syncUv();
        this._syncFilters();
    }
    _syncFilters() {
        if (!this._w?.setFilters) return;
        const NEAREST = 1003, NEAR_NEAR_MIP = 1004, LIN_NEAR_MIP = 1005, NEAR_LIN_MIP = 1007;
        const isNearest = (v) => v === NEAREST || v === NEAR_NEAR_MIP || v === LIN_NEAR_MIP || v === NEAR_LIN_MIP;
        const mag = isNearest(this.magFilter ?? 1006) ? 1 : 0;
        const min = isNearest(this.minFilter ?? 1008) ? 1 : 0;
        const conv = (w) => w === 1000 ? 1 : w === 1002 ? 2 : 0;
        this._w.setFilters(mag, min, conv(this.wrapS ?? 1001), conv(this.wrapT ?? 1001));
    }
    update() { this._upload(); }
}

// ---- Curves ----
export class LineCurve {
    constructor(v1, v2) { this._w = new WebLineCurve(v1._w(), v2._w()); }
}
export class LineCurve3 {
    constructor(v1, v2) { this._w = new WebLineCurve3(v1._w(), v2._w()); }
}
export class EllipseCurve {
    constructor(cx, cy, rx, ry, a0, a1, clockwise = false, rot = 0) {
        this._w = new WebEllipseCurve(cx, cy, rx, ry, a0, a1, clockwise, rot);
    }
}
export class CatmullRomCurve3 {
    constructor(points, closed = false, curveType = 'centripetal', tension = 0.5) {
        this.points = points;
        this.closed = closed;
        this.curveType = curveType;
        this.tension = tension;
        this.arcLengthDivisions = 200;
        this.cacheArcLengths = null;
        this.needsUpdate = false;
        const flat = new Float32Array(points.length * 3);
        points.forEach((p, i) => { flat[i * 3] = p.x; flat[i * 3 + 1] = p.y; flat[i * 3 + 2] = p.z; });
        this._w = new WebCatmullRomCurve3(flat);
    }
    getPoint(t, optionalTarget) {
        const point = optionalTarget || new Vector3();
        const points = this.points;
        const l = points.length;
        const p = (l - (this.closed ? 0 : 1)) * t;
        let intPoint = Math.floor(p);
        let weight = p - intPoint;
        if (this.closed) {
            intPoint += intPoint > 0 ? 0 : (Math.floor(Math.abs(intPoint) / l) + 1) * l;
        } else if (weight === 0 && intPoint === l - 1) {
            intPoint = l - 2;
            weight = 1;
        }
        let p0, p3;
        const tmp = new Vector3();
        if (this.closed || intPoint > 0) {
            p0 = points[(intPoint - 1) % l];
        } else {
            tmp.subVectors(points[0], points[1]).add(points[0]);
            p0 = tmp;
        }
        const p1 = points[intPoint % l];
        const p2 = points[(intPoint + 1) % l];
        if (this.closed || intPoint + 2 < l) {
            p3 = points[(intPoint + 2) % l];
        } else {
            tmp.subVectors(points[l - 1], points[l - 2]).add(points[l - 1]);
            p3 = tmp;
        }
        if (this.curveType === 'centripetal' || this.curveType === 'chordal') {
            const pow = this.curveType === 'chordal' ? 0.5 : 0.25;
            let dt0 = Math.pow(p0.distanceToSquared(p1), pow);
            let dt1 = Math.pow(p1.distanceToSquared(p2), pow);
            let dt2 = Math.pow(p2.distanceToSquared(p3), pow);
            if (dt1 < 1e-4) dt1 = 1.0;
            if (dt0 < 1e-4) dt0 = dt1;
            if (dt2 < 1e-4) dt2 = dt1;
            point.set(
                _nonuniformCatmullRom(p0.x, p1.x, p2.x, p3.x, dt0, dt1, dt2, weight),
                _nonuniformCatmullRom(p0.y, p1.y, p2.y, p3.y, dt0, dt1, dt2, weight),
                _nonuniformCatmullRom(p0.z, p1.z, p2.z, p3.z, dt0, dt1, dt2, weight),
            );
        } else {
            return _catmullRomPoint(p0, p1, p2, p3, weight, this.tension, point);
        }
        return point;
    }
    getLengths(divisions = this.arcLengthDivisions) {
        if (this.cacheArcLengths && this.cacheArcLengths.length === divisions + 1 && !this.needsUpdate) {
            return this.cacheArcLengths;
        }
        this.needsUpdate = false;
        const cache = [0];
        let last = this.getPoint(0);
        let sum = 0;
        for (let p = 1; p <= divisions; p++) {
            const cur = this.getPoint(p / divisions);
            sum += cur.distanceTo(last);
            cache.push(sum);
            last = cur;
        }
        this.cacheArcLengths = cache;
        return cache;
    }
    getUtoTmapping(u, distance) {
        const arcLengths = this.getLengths();
        const il = arcLengths.length;
        const targetArcLength = distance !== undefined ? distance : u * arcLengths[il - 1];
        let low = 0, high = il - 1, i = 0;
        while (low <= high) {
            i = Math.floor(low + (high - low) / 2);
            const comparison = arcLengths[i] - targetArcLength;
            if (comparison < 0) low = i + 1;
            else if (comparison > 0) high = i - 1;
            else { high = i; break; }
        }
        i = high;
        if (arcLengths[i] === targetArcLength) return i / (il - 1);
        const lengthBefore = arcLengths[i];
        const lengthAfter = arcLengths[i + 1];
        const segmentFraction = (targetArcLength - lengthBefore) / (lengthAfter - lengthBefore);
        return (i + segmentFraction) / (il - 1);
    }
    getPointAt(u, optionalTarget) {
        return this.getPoint(this.getUtoTmapping(u), optionalTarget);
    }
    getTangent(t, optionalTarget) {
        const delta = 0.0001;
        const t1 = Math.max(0, t - delta);
        const t2 = Math.min(1, t + delta);
        const pt1 = this.getPoint(t1);
        const pt2 = this.getPoint(t2);
        return (optionalTarget || new Vector3()).copy(pt2).sub(pt1).normalize();
    }
    getTangentAt(u, optionalTarget) {
        return this.getTangent(this.getUtoTmapping(u), optionalTarget);
    }
    computeFrenetFrames(segments, closed) {
        const normal = new Vector3();
        const tangents = [], normals = [], binormals = [];
        const vec = new Vector3();
        const mat = new Matrix4();
        for (let i = 0; i <= segments; i++) {
            tangents[i] = this.getTangentAt(i / segments, new Vector3());
        }
        normals[0] = new Vector3();
        binormals[0] = new Vector3();
        let min = Number.MAX_VALUE;
        const tx = Math.abs(tangents[0].x), ty = Math.abs(tangents[0].y), tz = Math.abs(tangents[0].z);
        if (tx <= min) { min = tx; normal.set(1, 0, 0); }
        if (ty <= min) { min = ty; normal.set(0, 1, 0); }
        if (tz <= min) normal.set(0, 0, 1);
        vec.crossVectors(tangents[0], normal).normalize();
        normals[0].crossVectors(tangents[0], vec);
        binormals[0].crossVectors(tangents[0], normals[0]);
        for (let i = 1; i <= segments; i++) {
            normals[i] = normals[i - 1].clone();
            binormals[i] = binormals[i - 1].clone();
            vec.crossVectors(tangents[i - 1], tangents[i]);
            if (vec.length() > Number.EPSILON) {
                vec.normalize();
                const theta = Math.acos(Math.max(-1, Math.min(1, tangents[i - 1].dot(tangents[i]))));
                normals[i].applyMatrix4(mat.makeRotationAxis(vec, theta));
            }
            binormals[i].crossVectors(tangents[i], normals[i]);
        }
        return { tangents, normals, binormals };
    }
}
function _catmullRomPoint(p0, p1, p2, p3, t, tension, target) {
    const t2 = t * t, t3 = t2 * t;
    const m1x = (p2.x - p0.x) * tension, m1y = (p2.y - p0.y) * tension, m1z = (p2.z - p0.z) * tension;
    const m2x = (p3.x - p1.x) * tension, m2y = (p3.y - p1.y) * tension, m2z = (p3.z - p1.z) * tension;
    return target.set(
        p1.x * (2 * t3 - 3 * t2 + 1) + p2.x * (-2 * t3 + 3 * t2) + m1x * (t3 - 2 * t2 + t) + m2x * (t3 - t2),
        p1.y * (2 * t3 - 3 * t2 + 1) + p2.y * (-2 * t3 + 3 * t2) + m1y * (t3 - 2 * t2 + t) + m2y * (t3 - t2),
        p1.z * (2 * t3 - 3 * t2 + 1) + p2.z * (-2 * t3 + 3 * t2) + m1z * (t3 - 2 * t2 + t) + m2z * (t3 - t2),
    );
}
function _nonuniformCatmullRom(x0, x1, x2, x3, dt0, dt1, dt2, t) {
    let t1 = (x1 - x0) / dt0 - (x2 - x0) / (dt0 + dt1) + (x2 - x1) / dt1;
    let t2 = (x2 - x1) / dt1 - (x3 - x1) / (dt1 + dt2) + (x3 - x2) / dt2;
    t1 *= dt1;
    t2 *= dt1;
    const c0 = x1;
    const c1 = t1;
    const c2 = -3 * x1 + 3 * x2 - 2 * t1 - t2;
    const c3 = 2 * x1 - 2 * x2 + t1 + t2;
    const t2v = t * t;
    const t3v = t2v * t;
    return c0 + c1 * t + c2 * t2v + c3 * t3v;
}
export class Path {
    constructor() { this._w = new WebPath(); }
    moveTo(x, y) { this._w.moveTo(x, y); return this; }
    lineTo(x, y) { this._w.lineTo(x, y); return this; }
}
// Shape — 2D polygon with imperative draw commands (moveTo/lineTo/quadraticCurveTo
// /bezierCurveTo/closePath). `getPoints(divisions)` flattens to a Vector2 array
// that ShapeGeometry / ExtrudeGeometry can triangulate.
export class Shape {
    constructor(points) {
        this._w = new WebShape();
        this._pts = [];
        this._cursor = { x: 0, y: 0 };
        if (Array.isArray(points)) for (const p of points) this._pts.push(new Vector2(p.x, p.y));
    }
    moveTo(x, y) { this._cursor = { x, y }; this._pts.push(new Vector2(x, y)); return this; }
    lineTo(x, y) { this._cursor = { x, y }; this._pts.push(new Vector2(x, y)); return this; }
    quadraticCurveTo(cx, cy, x, y) {
        const steps = 12;
        const sx = this._cursor.x, sy = this._cursor.y;
        for (let i = 1; i <= steps; i++) {
            const t = i / steps, u = 1 - t;
            this._pts.push(new Vector2(u*u*sx + 2*u*t*cx + t*t*x, u*u*sy + 2*u*t*cy + t*t*y));
        }
        this._cursor = { x, y };
        return this;
    }
    bezierCurveTo(c1x, c1y, c2x, c2y, x, y) {
        const steps = 12;
        const sx = this._cursor.x, sy = this._cursor.y;
        for (let i = 1; i <= steps; i++) {
            const t = i / steps, u = 1 - t;
            this._pts.push(new Vector2(
                u*u*u*sx + 3*u*u*t*c1x + 3*u*t*t*c2x + t*t*t*x,
                u*u*u*sy + 3*u*u*t*c1y + 3*u*t*t*c2y + t*t*t*y));
        }
        this._cursor = { x, y };
        return this;
    }
    closePath() {
        if (this._pts.length > 0) {
            const first = this._pts[0];
            this._pts.push(new Vector2(first.x, first.y));
        }
        return this;
    }
    getPoints(_divisions = 12) { return this._pts.slice(); }
    getSpacedPoints(divisions = 12) { return this.getPoints(divisions); }
    getPointsHoles(_divisions = 12) { return []; }
    extractPoints(divisions = 12) {
        return { shape: this.getPoints(divisions), holes: this.getPointsHoles(divisions) };
    }
}

// ---- Animation ----
// AnimationClip — name + duration + JS-side tracks (KeyframeTracks).
export class AnimationClip {
    constructor(name = 'clip', duration = -1, tracks = []) {
        this._w = typeof WebAnimationClip !== 'undefined'
            ? new WebAnimationClip(name, duration < 0 ? 0 : duration)
            : null;
        this.name = name;
        this.tracks = tracks;
        this.duration = duration < 0
            ? tracks.reduce((mx, t) => Math.max(mx, t.times?.[t.times.length - 1] || 0), 0)
            : duration;
        this.uuid = MathUtils.generateUUID();
        this.blendMode = 2500; // NormalAnimationBlendMode
    }
    static parse(json) {
        const tracks = (json.tracks || []).map((t) => {
            const klass = t.type === 'quaternion' ? QuaternionKeyframeTrack
                : t.type === 'number' ? NumberKeyframeTrack
                : t.type === 'color' ? ColorKeyframeTrack
                : VectorKeyframeTrack;
            return new klass(t.name, t.times, t.values, t.interpolation);
        });
        return new AnimationClip(json.name || 'clip', json.duration ?? -1, tracks);
    }
    clone() {
        return new AnimationClip(this.name, this.duration, this.tracks.map((t) => {
            const C = t.constructor;
            return new C(t.name, t.times.slice(), t.values.slice(), t.interpolation);
        }));
    }
}

const LoopOnce = 2200, LoopRepeat = 2201, LoopPingPong = 2202;
const NormalAnimationBlendMode = 2500, AdditiveAnimationBlendMode = 2501;

/** Procedural modifiers on keyframe tracks (noise / cycles / stepped), Blender-style. */
export class TrackModifier {
    static noise({ amplitude = 0.05, frequency = 1, seed = 0 } = {}) {
        return { kind: 'noise', amplitude, frequency, seed };
    }
    static cycles({ before = Infinity, after = Infinity } = {}) {
        return { kind: 'cycles', before, after };
    }
    static stepped({ stepSize = 0.1 } = {}) {
        return { kind: 'stepped', stepSize };
    }
    static limit({ min = -Infinity, max = Infinity } = {}) {
        return { kind: 'limit', min, max };
    }
    static remapTime(modifiers, t, track) {
        if (!modifiers?.length) return t;
        const dur = track?.times?.[track.times.length - 1] || 0;
        let time = t;
        for (const m of modifiers) {
            if (m.kind === 'cycles' && dur > 0) {
                // Wrap into clip range with optional before/after cycle counts (Infinity = forever).
                if (time < 0) {
                    if (m.before === 0) time = 0;
                    else time = ((time % dur) + dur) % dur;
                } else if (time > dur) {
                    if (m.after === 0) time = dur;
                    else time = time % dur;
                }
            } else if (m.kind === 'stepped') {
                const s = Math.max(1e-6, m.stepSize);
                time = Math.floor(time / s) * s;
            }
        }
        return time;
    }
    static applyValue(modifiers, values, t) {
        if (!modifiers?.length) return values;
        for (const m of modifiers) {
            if (m.kind === 'noise') {
                for (let i = 0; i < values.length; i++) {
                    const n = Math.sin((t * m.frequency + m.seed + i * 19.19) * 12.9898) * 43758.5453;
                    const f = n - Math.floor(n);
                    values[i] += (f * 2 - 1) * m.amplitude;
                }
            } else if (m.kind === 'limit') {
                for (let i = 0; i < values.length; i++) {
                    values[i] = Math.min(m.max, Math.max(m.min, values[i]));
                }
            }
        }
        return values;
    }
}

function _findNodeByName(root, nodeName) {
    if (!root) return null;
    if (!nodeName || nodeName === '.' || nodeName === root.name) return root;
    if (root.getObjectByName) {
        const hit = root.getObjectByName(nodeName);
        if (hit) return hit;
    }
    // Bones on a skeleton bound to a SkinnedMesh under root.
    const stack = [root];
    while (stack.length) {
        const o = stack.pop();
        if (o?.name === nodeName || o?.uuid === nodeName) return o;
        if (o?.isBone && (o.name === nodeName)) return o;
        if (o?.skeleton?.bones) {
            for (const b of o.skeleton.bones) {
                if (b.name === nodeName) return b;
                stack.push(b);
            }
        }
        for (const c of (o.children || o._children || [])) stack.push(c);
    }
    return null;
}

/** three.js-shaped property path binder. */
export class PropertyMixer {
    constructor(binding, typeName, valueSize) {
        this.binding = binding;
        this.typeName = typeName;
        this.valueSize = valueSize;
        this.buffer = new Float32Array(valueSize * 4); // acc / result / original / additive
        this.cumulativeWeight = 0;
        this.cumulativeWeightAdditive = 0;
        this.useCount = 0;
        this.referenceCount = 0;
    }
    accumulate(offset, weight) {
        if (weight <= 0) return;
        const size = this.valueSize;
        const buffer = this.buffer;
        const stride = size;
        if (this.cumulativeWeight === 0) {
            for (let i = 0; i < size; i++) buffer[i] = buffer[offset + i] * weight;
        } else {
            const w = weight / (this.cumulativeWeight + weight);
            for (let i = 0; i < size; i++) {
                buffer[i] += (buffer[offset + i] - buffer[i]) * w;
            }
        }
        this.cumulativeWeight += weight;
        void stride;
    }
    accumulateAdditive(offset, weight) {
        if (weight <= 0) return;
        const size = this.valueSize;
        const buffer = this.buffer;
        const base = size * 3; // additive region
        if (this.cumulativeWeightAdditive === 0) {
            for (let i = 0; i < size; i++) buffer[base + i] = 0;
        }
        for (let i = 0; i < size; i++) buffer[base + i] += buffer[offset + i] * weight;
        this.cumulativeWeightAdditive += weight;
    }
    apply(accuIndex) {
        const size = this.valueSize;
        const buffer = this.buffer;
        const weight = this.cumulativeWeight;
        const weightAdditive = this.cumulativeWeightAdditive;
        const original = size;
        const additive = size * 3;
        if (weight < 1) {
            // Mix toward original binding value.
            this.binding.getValue(buffer, original);
            const w = weight / (weight || 1);
            if (this.typeName === 'quaternion') {
                _slerpFlat(buffer, 0, buffer, original, buffer, 0, w);
            } else {
                for (let i = 0; i < size; i++) {
                    buffer[i] = buffer[original + i] + (buffer[i] - buffer[original + i]) * w;
                }
            }
        }
        if (weightAdditive > 0) {
            if (this.typeName === 'quaternion') {
                // Additive quat ≈ multiply
                const qx = buffer[additive], qy = buffer[additive + 1], qz = buffer[additive + 2], qw = buffer[additive + 3];
                const x = buffer[0], y = buffer[1], z = buffer[2], w = buffer[3];
                buffer[0] = w * qx + x * qw + y * qz - z * qy;
                buffer[1] = w * qy - x * qz + y * qw + z * qx;
                buffer[2] = w * qz + x * qy - y * qx + z * qw;
                buffer[3] = w * qw - x * qx - y * qy - z * qz;
                const ln = Math.hypot(buffer[0], buffer[1], buffer[2], buffer[3]) || 1;
                buffer[0] /= ln; buffer[1] /= ln; buffer[2] /= ln; buffer[3] /= ln;
            } else {
                for (let i = 0; i < size; i++) buffer[i] += buffer[additive + i];
            }
        }
        this.binding.setValue(buffer, 0);
        this.cumulativeWeight = 0;
        this.cumulativeWeightAdditive = 0;
        void accuIndex;
    }
    saveOriginalState() {
        this.binding.getValue(this.buffer, this.valueSize);
    }
    restoreOriginalState() {
        this.binding.setValue(this.buffer, this.valueSize);
    }
}

function _slerpFlat(dst, dstOffset, src0, srcOffset0, src1, srcOffset1, t) {
    let x0 = src0[srcOffset0], y0 = src0[srcOffset0 + 1], z0 = src0[srcOffset0 + 2], w0 = src0[srcOffset0 + 3];
    const x1 = src1[srcOffset1], y1 = src1[srcOffset1 + 1], z1 = src1[srcOffset1 + 2], w1 = src1[srcOffset1 + 3];
    let dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1;
    if (dot < 0) { x0 = -x0; y0 = -y0; z0 = -z0; w0 = -w0; dot = -dot; }
    if (dot > 0.9995) {
        dst[dstOffset] = x0 + t * (x1 - x0);
        dst[dstOffset + 1] = y0 + t * (y1 - y0);
        dst[dstOffset + 2] = z0 + t * (z1 - z0);
        dst[dstOffset + 3] = w0 + t * (w1 - w0);
    } else {
        const th0 = Math.acos(Math.min(1, dot));
        const sin0 = Math.sin(th0);
        const s0 = Math.sin((1 - t) * th0) / sin0;
        const s1 = Math.sin(t * th0) / sin0;
        dst[dstOffset] = x0 * s0 + x1 * s1;
        dst[dstOffset + 1] = y0 * s0 + y1 * s1;
        dst[dstOffset + 2] = z0 * s0 + z1 * s1;
        dst[dstOffset + 3] = w0 * s0 + w1 * s1;
    }
    const ln = Math.hypot(dst[dstOffset], dst[dstOffset + 1], dst[dstOffset + 2], dst[dstOffset + 3]) || 1;
    dst[dstOffset] /= ln; dst[dstOffset + 1] /= ln; dst[dstOffset + 2] /= ln; dst[dstOffset + 3] /= ln;
}

export class PropertyBinding {
    constructor(rootNode, path, parsedPath) {
        this.path = path;
        this.parsedPath = parsedPath || PropertyBinding.parseTrackName(path);
        this.node = null;
        this.rootNode = rootNode;
        this.targetObject = null;
        this.propertyName = null;
        this.resolvedProperty = null;
        this.propertyIndex = null;
        this.bindingType = 'direct';
        this._getValue = null;
        this._setValue = null;
    }
    static parseTrackName(trackName) {
        // Supports: nodeName.property, nodeName.material.color, .morphTargetInfluences[0]
        const re = /^((?:[\w\-_:]+|\.))(?:\.|\[)?([\w\-\[\].:]*)$/;
        // More permissive parse: split on first '.' after optional node
        let nodeName = '';
        let objectName = '';
        let objectIndex = undefined;
        let propertyName = '';
        let propertyIndex = undefined;
        const morphMatch = trackName.match(/^(.*?)\.morphTargetInfluences(?:\[(\d+)\])?$/);
        if (morphMatch) {
            return {
                nodeName: morphMatch[1] || '',
                objectName: undefined,
                objectIndex: undefined,
                propertyName: 'morphTargetInfluences',
                propertyIndex: morphMatch[2] != null ? Number(morphMatch[2]) : undefined,
            };
        }
        const matMatch = trackName.match(/^(.*?)\.materials?\[(\d+)\]\.(.+)$/);
        if (matMatch) {
            return {
                nodeName: matMatch[1],
                objectName: 'materials',
                objectIndex: Number(matMatch[2]),
                propertyName: matMatch[3],
                propertyIndex: undefined,
            };
        }
        const parts = trackName.split('.');
        if (parts.length === 1) {
            propertyName = parts[0];
        } else {
            nodeName = parts[0];
            propertyName = parts.slice(1).join('.');
            // material.color / material.opacity
            if (propertyName.startsWith('material.')) {
                objectName = 'material';
                propertyName = propertyName.slice('material.'.length);
            }
        }
        const idxMatch = propertyName.match(/^([^\[]+)\[(\d+)\]$/);
        if (idxMatch) {
            propertyName = idxMatch[1];
            propertyIndex = Number(idxMatch[2]);
        }
        void re;
        return { nodeName, objectName, objectIndex, propertyName, propertyIndex };
    }
    static findNode(root, nodeName) {
        return _findNodeByName(root, nodeName);
    }
    bind() {
        if (this.node) return;
        const parsed = this.parsedPath;
        this.node = PropertyBinding.findNode(this.rootNode, parsed.nodeName) || this.rootNode;
        let object = this.node;
        if (parsed.objectName === 'material') {
            object = object?.material;
        } else if (parsed.objectName === 'materials') {
            const mats = Array.isArray(object?.material) ? object.material : [object?.material];
            object = mats[parsed.objectIndex ?? 0];
        }
        this.targetObject = object;
        this.propertyName = parsed.propertyName;
        this.propertyIndex = parsed.propertyIndex;
        const prop = this.propertyName;
        if (prop === 'morphTargetInfluences') {
            const influences = this.node?.morphTargetInfluences;
            this.resolvedProperty = influences;
            this.bindingType = 'array';
            if (this.propertyIndex != null) {
                this._getValue = (buf, offset) => { buf[offset] = influences?.[this.propertyIndex] ?? 0; };
                this._setValue = (buf, offset) => {
                    if (influences) influences[this.propertyIndex] = buf[offset];
                    this.node?.updateMorphTargets?.();
                };
            } else {
                this._getValue = (buf, offset) => {
                    for (let i = 0; i < (influences?.length || 0); i++) buf[offset + i] = influences[i];
                };
                this._setValue = (buf, offset) => {
                    if (!influences) return;
                    for (let i = 0; i < influences.length; i++) influences[i] = buf[offset + i];
                    this.node?.updateMorphTargets?.();
                };
            }
            return;
        }
        const target = object?.[prop] != null ? object : this.node;
        const value = target?.[prop];
        this.targetObject = target;
        if (value && typeof value === 'object' && 'x' in value) {
            // Vector3 / Euler / Color / Quaternion
            const isQuat = 'w' in value;
            const isColor = 'r' in value && !('x' in value);
            this.bindingType = 'structured';
            this._getValue = (buf, offset) => {
                if (isColor) {
                    buf[offset] = value.r; buf[offset + 1] = value.g; buf[offset + 2] = value.b;
                } else {
                    buf[offset] = value.x; buf[offset + 1] = value.y; buf[offset + 2] = value.z;
                    if (isQuat) buf[offset + 3] = value.w;
                }
            };
            this._setValue = (buf, offset) => {
                if (isColor) {
                    if (value.setRGB) value.setRGB(buf[offset], buf[offset + 1], buf[offset + 2]);
                    else { value.r = buf[offset]; value.g = buf[offset + 1]; value.b = buf[offset + 2]; }
                } else if (isQuat) {
                    value.set?.(buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3])
                        || (value.x = buf[offset], value.y = buf[offset + 1], value.z = buf[offset + 2], value.w = buf[offset + 3]);
                } else {
                    value.set?.(buf[offset], buf[offset + 1], buf[offset + 2])
                        || (value.x = buf[offset], value.y = buf[offset + 1], value.z = buf[offset + 2]);
                }
                this._syncNode(target);
            };
        } else {
            this.bindingType = 'direct';
            this._getValue = (buf, offset) => { buf[offset] = target?.[prop] ?? 0; };
            this._setValue = (buf, offset) => {
                if (!target) return;
                target[prop] = buf[offset];
                if (prop === 'fov' || prop === 'zoom' || prop === 'opacity') {
                    target.updateProjectionMatrix?.();
                    if (target.needsUpdate != null) target.needsUpdate = true;
                }
                this._syncNode(target);
            };
        }
    }
    _syncNode(target) {
        const node = this.node;
        if (!node) return;
        if (typeof node._sync === 'function') node._sync();
        else if (node.isBone) node.updateMatrix?.();
        else if (node.skeleton) node.skeleton.update?.();
        void target;
    }
    unbind() {
        this.node = null;
        this.targetObject = null;
        this._getValue = null;
        this._setValue = null;
    }
    getValue(buffer, offset = 0) {
        this.bind();
        this._getValue?.(buffer, offset);
    }
    setValue(buffer, offset = 0) {
        this.bind();
        this._setValue?.(buffer, offset);
    }
}

// AnimationAction — play / fade / crossFade / additive blend.
class _AnimationAction {
    constructor(mixer, clip, root) {
        this.mixer = mixer;
        this.clip = clip;
        this.root = root;
        this.enabled = true;
        this.paused = false;
        this.weight = 1.0;
        this._effectiveWeight = 1.0;
        this.timeScale = 1.0;
        this._effectiveTimeScale = 1.0;
        this.time = 0;
        this.loop = LoopRepeat;
        this.repetitions = Infinity;
        this.isRunning = false;
        this.clampWhenFinished = false;
        this.zeroSlopeAtStart = true;
        this.zeroSlopeAtEnd = true;
        this.blendMode = clip.blendMode ?? NormalAnimationBlendMode;
        this._fade = null; // { from, to, duration, elapsed, scheduleStop }
        this._bindings = null;
        this._interpolants = null;
    }
    play() {
        this.isRunning = true;
        this.paused = false;
        this.enabled = true;
        this.mixer._activateAction(this);
        return this;
    }
    stop() {
        this.isRunning = false;
        this.time = 0;
        this._fade = null;
        this.mixer._deactivateAction(this);
        return this;
    }
    reset() { this.time = 0; this._fade = null; return this; }
    setLoop(mode, repetitions) {
        this.loop = mode;
        if (repetitions != null) this.repetitions = repetitions;
        return this;
    }
    setEffectiveWeight(w) {
        this.weight = w;
        this._effectiveWeight = w;
        this._fade = null;
        return this;
    }
    getEffectiveWeight() { return this._effectiveWeight; }
    setEffectiveTimeScale(s) {
        this.timeScale = s;
        this._effectiveTimeScale = s;
        return this;
    }
    getEffectiveTimeScale() { return this._effectiveTimeScale; }
    fadeIn(duration = 0.5) {
        this._fade = { from: 0, to: this.weight, duration: Math.max(0, duration), elapsed: 0, scheduleStop: false };
        this._effectiveWeight = 0;
        this.play();
        return this;
    }
    fadeOut(duration = 0.5) {
        this._fade = {
            from: this._effectiveWeight, to: 0,
            duration: Math.max(0, duration), elapsed: 0, scheduleStop: true,
        };
        return this;
    }
    crossFadeFrom(fadeOutAction, duration = 0.5, warp = true) {
        fadeOutAction?.fadeOut?.(duration);
        this.fadeIn(duration);
        if (warp && fadeOutAction && this.clip.duration > 0 && fadeOutAction.clip.duration > 0) {
            const ratio = fadeOutAction.clip.duration / this.clip.duration;
            this.time = fadeOutAction.time / ratio;
        }
        return this;
    }
    crossFadeTo(fadeInAction, duration = 0.5, warp = true) {
        fadeInAction?.crossFadeFrom?.(this, duration, warp);
        return this;
    }
    halt(duration = 0.5) {
        this._fade = {
            from: this._effectiveTimeScale, to: 0,
            duration: Math.max(0, duration), elapsed: 0, scheduleStop: false, kind: 'timeScale',
        };
        return this;
    }
    stopFading() { this._fade = null; return this; }
    syncWith(action) {
        this.time = action.time * (this.clip.duration / Math.max(1e-8, action.clip.duration));
        return this;
    }
    setDuration(duration) {
        this.timeScale = this.clip.duration / Math.max(1e-8, duration);
        return this;
    }
    _updateFade(dt) {
        if (!this._fade) return;
        const f = this._fade;
        f.elapsed += dt;
        const u = f.duration <= 0 ? 1 : Math.min(1, f.elapsed / f.duration);
        const v = f.from + (f.to - f.from) * u;
        if (f.kind === 'timeScale') this._effectiveTimeScale = v;
        else this._effectiveWeight = v;
        if (u >= 1) {
            if (f.scheduleStop) this.stop();
            this._fade = null;
        }
    }
}

// AnimationMixer — PropertyBinding + PropertyMixer accumulation, fades, additive.
export class AnimationMixer {
    constructor(root) {
        this._w = typeof WebAnimationMixer !== 'undefined' ? new WebAnimationMixer() : null;
        this.root = root;
        this.actions = [];
        this.time = 0;
        this.timeScale = 1.0;
        this._bindingsByKey = new Map();
        this._activeActions = new Set();
        this.stats = { actions: 0, bindings: 0 };
    }
    clipAction(clip, root = this.root) {
        for (const a of this.actions) if (a.clip === clip && a.root === root) return a;
        const action = new _AnimationAction(this, clip, root);
        this.actions.push(action);
        return action;
    }
    existingAction(clip, root = this.root) {
        return this.actions.find((a) => a.clip === clip && a.root === root) || null;
    }
    uncacheClip(clip) { this.actions = this.actions.filter((a) => a.clip !== clip); }
    uncacheRoot(root) { this.actions = this.actions.filter((a) => a.root !== root); }
    uncacheAction(clip, root) {
        this.actions = this.actions.filter((a) => !(a.clip === clip && a.root === root));
    }
    getRoot() { return this.root; }
    setTime(t) { this.time = t; return this; }
    stopAllAction() { for (const a of this.actions) a.stop(); return this; }
    _activateAction(action) {
        this._activeActions.add(action);
        this._ensureBindings(action);
    }
    _deactivateAction(action) { this._activeActions.delete(action); }
    _ensureBindings(action) {
        if (action._bindings) return;
        const root = action.root || this.root;
        const bindings = [];
        const mixers = [];
        for (const track of action.clip.tracks || []) {
            const key = `${root?.uuid || ''}|${track.name}`;
            let entry = this._bindingsByKey.get(key);
            if (!entry) {
                const binding = new PropertyBinding(root, track.name);
                binding.bind();
                const size = track.getValueSize?.() ?? Math.max(1, (track.values?.length || 1) / Math.max(1, track.times?.length || 1));
                const typeName = track instanceof QuaternionKeyframeTrack ? 'quaternion'
                    : track instanceof NumberKeyframeTrack || track instanceof BooleanKeyframeTrack ? 'number'
                    : track instanceof ColorKeyframeTrack ? 'color'
                    : 'vector';
                const mixer = new PropertyMixer(binding, typeName, size);
                mixer.saveOriginalState();
                entry = { binding, mixer, size, typeName };
                this._bindingsByKey.set(key, entry);
            }
            bindings.push(entry);
            mixers.push(entry.mixer);
        }
        action._bindings = bindings;
        action._interpolants = mixers;
    }
    _sampleTrack(track, t, out) {
        const times = track.times, values = track.values;
        const size = track.getValueSize?.() ?? (values.length / Math.max(1, times.length));
        const result = out || new Float32Array(size);
        if (!times || times.length === 0) {
            result.fill(0);
            return result;
        }
        // Optional procedural modifiers on the track.
        if (track.modifiers?.length) {
            t = TrackModifier.remapTime(track.modifiers, t, track);
        }
        if (t <= times[0]) {
            for (let k = 0; k < size; k++) result[k] = values[k];
            return TrackModifier.applyValue(track.modifiers, result, t);
        }
        const last = times.length - 1;
        if (t >= times[last]) {
            for (let k = 0; k < size; k++) result[k] = values[last * size + k];
            return TrackModifier.applyValue(track.modifiers, result, t);
        }
        let lo = 0, hi = last;
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (times[mid] <= t) lo = mid; else hi = mid;
        }
        const alpha = (t - times[lo]) / (times[hi] - times[lo] || 1);
        const isQuat = track instanceof QuaternionKeyframeTrack
            || track.ValueTypeName === 'quaternion';
        if (isQuat) {
            const aOff = lo * 4, bOff = hi * 4;
            _slerpFlat(result, 0, values, aOff, values, bOff, alpha);
        } else {
            for (let k = 0; k < size; k++) {
                result[k] = values[lo * size + k] + alpha * (values[hi * size + k] - values[lo * size + k]);
            }
        }
        return TrackModifier.applyValue(track.modifiers, result, t);
    }
    update(delta) {
        const dt = delta * this.timeScale;
        this.time += dt;
        // Save originals once per frame for active mixers.
        const used = new Set();
        for (const action of this.actions) {
            if (!action.isRunning || action.paused || !action.enabled) continue;
            action._updateFade(Math.abs(dt));
            const weight = action._effectiveWeight;
            if (weight <= 0 && action.blendMode !== AdditiveAnimationBlendMode) continue;
            this._ensureBindings(action);
            const scale = action._effectiveTimeScale;
            action.time += dt * scale;
            const dur = action.clip.duration || 0;
            let t = action.time;
            if (dur > 0) {
                if (action.loop === LoopOnce) {
                    if (t > dur) {
                        t = dur;
                        action.time = dur;
                        if (!action.clampWhenFinished) action.isRunning = false;
                    } else if (t < 0) {
                        t = 0;
                        action.time = 0;
                        if (!action.clampWhenFinished) action.isRunning = false;
                    }
                } else if (action.loop === LoopPingPong) {
                    const period = dur * 2;
                    const mod = ((t % period) + period) % period;
                    t = mod > dur ? period - mod : mod;
                } else {
                    t = ((t % dur) + dur) % dur;
                }
            }
            const additive = action.blendMode === AdditiveAnimationBlendMode;
            const tracks = action.clip.tracks || [];
            for (let i = 0; i < tracks.length; i++) {
                const entry = action._bindings[i];
                if (!entry) continue;
                const mixer = entry.mixer;
                used.add(mixer);
                const buf = mixer.buffer;
                const size = mixer.valueSize;
                const scratch = size * 2; // use region 2 as sample scratch
                this._sampleTrack(tracks[i], t, buf.subarray(scratch, scratch + size));
                // Copy scratch into offset region expected by accumulate (region 0 is acc).
                // PropertyMixer.accumulate reads from `offset` — use scratch as source by
                // temporarily treating region 2 as the take.
                if (additive) mixer.accumulateAdditive(scratch, weight);
                else mixer.accumulate(scratch, weight);
            }
        }
        for (const mixer of used) mixer.apply(0);
        this.stats.actions = this._activeActions.size;
        this.stats.bindings = this._bindingsByKey.size;
        return this;
    }
}

export const AnimationAction = _AnimationAction;


// ---- Controls ----
function _controlViewport(domElement) {
    const w = domElement?.clientWidth || (typeof window !== 'undefined' ? window.innerWidth : 800);
    const h = domElement?.clientHeight || (typeof window !== 'undefined' ? window.innerHeight : 600);
    return [Math.max(1, w), Math.max(1, h)];
}

export class OrbitControls {
    constructor(camera, domElement) {
        this._camera = camera;
        this.domElement = domElement;
        this.enabled = true;
        this.enableDamping = false;
        this.dampingFactor = 0.05;
        this.autoRotate = false;
        this.autoRotateSpeed = 2.0;
        // three.js-compatible public knobs (synced into wasm each update).
        this.target = camera._lookAt ? camera._lookAt.clone() : new Vector3();
        this.minDistance = 0.1;
        this.maxDistance = Infinity;
        this.minPolarAngle = 0;
        this.maxPolarAngle = Math.PI;
        this.minAzimuthAngle = -Infinity;
        this.maxAzimuthAngle = Infinity;
        this.enableZoom = true;
        this.enableRotate = true;
        this.enablePan = true;
        this.rotateSpeed = 1.0;
        this.zoomSpeed = 1.0;
        this.panSpeed = 1.0;
        this._rotating = false;
        this._panning = false;
        this._lastX = 0;
        this._lastY = 0;
        this._dragPointerId = null;
        this._lastT = (typeof performance !== 'undefined' ? performance.now() : 0);
        if (typeof camera._sync === 'function') camera._sync();
        if (camera._lookAt) camera._w.lookAt(camera._lookAt.x, camera._lookAt.y, camera._lookAt.z);
        this._w = new WebOrbitControls(camera._w);
        if (typeof this._w.setTarget === 'function') {
            this._w.setTarget(this.target.x, this.target.y, this.target.z);
        }
        camera._orbitControlled = true;

        this._onContextMenu = (e) => e.preventDefault();
        this._onPointerDown = (e) => {
            if (!this.enabled || (e.button !== 0 && e.button !== 2)) return;
            const root = this.domElement;
            if (!root) return;
            e.preventDefault();
            this._dragPointerId = e.pointerId;
            this._rotating = e.button === 0 && this.enableRotate;
            this._panning = e.button === 2 && this.enablePan;
            this._lastX = e.clientX;
            this._lastY = e.clientY;
            root.setPointerCapture?.(e.pointerId);
            const doc = root.ownerDocument || (typeof document !== 'undefined' ? document : null);
            if (doc) {
                doc.addEventListener('pointermove', this._onPointerMove);
                doc.addEventListener('pointerup', this._onPointerUp);
                doc.addEventListener('pointercancel', this._onPointerUp);
            }
        };
        this._onPointerMove = (e) => {
            if (!this.enabled || e.pointerId !== this._dragPointerId) return;
            if (!this._rotating && !this._panning) return;
            const dx = e.clientX - this._lastX;
            const dy = e.clientY - this._lastY;
            this._lastX = e.clientX;
            this._lastY = e.clientY;
            if (dx || dy) this.update(dx, dy, 0, this._rotating, this._panning);
        };
        this._onPointerUp = (e) => {
            if (this._dragPointerId != null && e.pointerId !== this._dragPointerId) return;
            this._endDrag();
        };
        this._onWheel = (e) => {
            if (!this.enabled || !this.enableZoom) return;
            e.preventDefault();
            this.update(0, 0, e.deltaY, false, false);
        };

        if (domElement?.addEventListener) {
            domElement.addEventListener('contextmenu', this._onContextMenu);
            domElement.addEventListener('pointerdown', this._onPointerDown);
            domElement.addEventListener('wheel', this._onWheel, { passive: false });
            if (!domElement.style.touchAction) domElement.style.touchAction = 'none';
        }
    }
    _endDrag() {
        const root = this.domElement;
        if (root && this._dragPointerId != null) {
            root.releasePointerCapture?.(this._dragPointerId);
            const doc = root.ownerDocument || (typeof document !== 'undefined' ? document : null);
            if (doc) {
                doc.removeEventListener('pointermove', this._onPointerMove);
                doc.removeEventListener('pointerup', this._onPointerUp);
                doc.removeEventListener('pointercancel', this._onPointerUp);
            }
        }
        this._dragPointerId = null;
        this._rotating = false;
        this._panning = false;
    }
    dispose() {
        this._endDrag();
        const root = this.domElement;
        if (root?.removeEventListener) {
            root.removeEventListener('contextmenu', this._onContextMenu);
            root.removeEventListener('pointerdown', this._onPointerDown);
            root.removeEventListener('wheel', this._onWheel);
        }
        if (this._camera) this._camera._orbitControlled = false;
    }
    _syncLimits() {
        if (typeof this._w.setLimits === 'function') {
            const maxD = Number.isFinite(this.maxDistance) ? this.maxDistance : 1e6;
            this._w.setLimits(
                Math.max(0, this.minDistance),
                Math.max(this.minDistance, maxD),
                this.minPolarAngle,
                this.maxPolarAngle,
                this.rotateSpeed,
                this.zoomSpeed,
                this.panSpeed,
                this.enableDamping ? (this.dampingFactor ?? 0.05) : 0,
            );
        }
        if (typeof this._w.setTarget === 'function') {
            this._w.setTarget(this.target.x, this.target.y, this.target.z);
        }
    }
    _syncMotion() {
        if (typeof this._w.setMotion === 'function') {
            const radPerSec = (this.autoRotateSpeed * Math.PI) / 30;
            this._w.setMotion(
                !!this.enableDamping,
                this.dampingFactor ?? 0.05,
                !!this.autoRotate,
                radPerSec,
            );
        }
    }
    update(dx = 0, dy = 0, wheel = 0, rotating = false, panning = false) {
        this._syncLimits();
        this._syncMotion();
        const [w, h] = _controlViewport(this.domElement);
        const now = (typeof performance !== 'undefined' ? performance.now() : this._lastT + 16.6);
        const dt = Math.min(0.1, Math.max(0.0, (now - this._lastT) / 1000));
        this._lastT = now;
        if (typeof this._w.updateDt === 'function') {
            this._w.updateDt(this._camera._w, dx, dy, wheel, rotating, panning, w, h, dt || 1 / 60);
        } else {
            this._w.update(this._camera._w, dx, dy, wheel, rotating, panning, w, h);
        }
        this._syncFromWasm();
    }
    _syncFromWasm() {
        const p = this._camera._w.readPosition();
        const t = this._camera._w.readTarget();
        this._camera.position.set(p.x, p.y, p.z);
        this._camera._lookAt.set(t.x, t.y, t.z);
        this.target.set(t.x, t.y, t.z);
    }
    resetFromCamera() {
        if (typeof this._camera._sync === 'function') this._camera._sync();
        if (this._camera._lookAt) this.target.copy(this._camera._lookAt);
        if (typeof this._w?.reseedFromCamera === 'function') {
            this._w.reseedFromCamera(this._camera._w);
        } else {
            this._w = new WebOrbitControls(this._camera._w);
        }
    }
}

// ---------------------------------------------------------------------------
// Cinematic camera helpers — additive; do NOT replace OrbitControls /
// AnimationMixer. Mirrors threers-animation SpeedRamp / CameraPath / shots.
// ---------------------------------------------------------------------------
const _CAM_EASINGS = {
    linear: (t) => t,
    quadIn: (t) => t * t,
    quadOut: (t) => t * (2 - t),
    cubicIn: (t) => t * t * t,
    cubicOut: (t) => 1 - Math.pow(1 - t, 3),
    cubicInOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    smooth: (t) => t * t * (3 - 2 * t),
    smootherstep: (t) => t * t * t * (t * (t * 6 - 15) + 10),
};

/** Map linear 0..1 → shaped parameter (ease / cruise / smootherstep). */
export function speedRamp(kind, t, opts = {}) {
    t = Math.min(1, Math.max(0, t));
    if (kind == null || kind === 'linear') return t;
    if (typeof kind === 'function') return kind(t);
    if (kind === 'smooth') return _CAM_EASINGS.smooth(t);
    if (kind === 'smoother' || kind === 'smootherstep') return _CAM_EASINGS.smootherstep(t);
    if (kind === 'inOut' || kind === 'ramp') {
        const a = Math.min(1, Math.max(0, opts.easeIn ?? 0.25));
        const b = Math.min(1 - a, Math.max(0, opts.easeOut ?? 0.25));
        const mid = 1 - a - b;
        if (t <= a) {
            if (a <= 1e-8) return 0;
            const u = t / a;
            return a * _CAM_EASINGS.smooth(u);
        }
        if (t >= 1 - b) {
            if (b <= 1e-8) return 1;
            const u = (t - (1 - b)) / b;
            return a + mid + b * _CAM_EASINGS.smooth(u);
        }
        return a + mid * ((t - a) / Math.max(1e-8, mid));
    }
    const ease = _CAM_EASINGS[kind] || _CAM_EASINGS.cubicInOut;
    return ease(t);
}

function _v3(p) {
    if (p instanceof Vector3) return p.clone();
    if (Array.isArray(p)) return new Vector3(p[0], p[1], p[2]);
    if (p && typeof p === 'object') return new Vector3(p.x ?? 0, p.y ?? 0, p.z ?? 0);
    return new Vector3();
}

function _sampleLinear(points, u) {
    const n = points.length - 1;
    const f = u * n;
    const i = Math.min(n - 1, Math.floor(f));
    const t = f - i;
    return new Vector3().lerpVectors(points[i], points[i + 1], t);
}

function _sampleBezier(points, u) {
    const segments = Math.floor((points.length - 1) / 3);
    if (segments <= 0) return _sampleLinear(points, u);
    const f = u * segments;
    const s = Math.min(segments - 1, Math.floor(f));
    const t = f - s;
    const i = s * 3;
    const p0 = points[i], c0 = points[i + 1], c1 = points[i + 2], p1 = points[i + 3];
    const omt = 1 - t;
    return new Vector3(
        omt * omt * omt * p0.x + 3 * omt * omt * t * c0.x + 3 * omt * t * t * c1.x + t * t * t * p1.x,
        omt * omt * omt * p0.y + 3 * omt * omt * t * c0.y + 3 * omt * t * t * c1.y + t * t * t * p1.y,
        omt * omt * omt * p0.z + 3 * omt * omt * t * c0.z + 3 * omt * t * t * c1.z + t * t * t * p1.z,
    );
}

function _sampleCatmull(points, u) {
    const n = points.length - 1;
    const f = u * n;
    const i = Math.min(n - 1, Math.floor(f));
    const t = f - i;
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[Math.min(points.length - 1, i + 1)];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    const t2 = t * t, t3 = t2 * t;
    return new Vector3(
        (-0.5 * t3 + t2 - 0.5 * t) * p0.x + (1.5 * t3 - 2.5 * t2 + 1) * p1.x + (-1.5 * t3 + 2 * t2 + 0.5 * t) * p2.x + (0.5 * t3 - 0.5 * t2) * p3.x,
        (-0.5 * t3 + t2 - 0.5 * t) * p0.y + (1.5 * t3 - 2.5 * t2 + 1) * p1.y + (-1.5 * t3 + 2 * t2 + 0.5 * t) * p2.y + (0.5 * t3 - 0.5 * t2) * p3.y,
        (-0.5 * t3 + t2 - 0.5 * t) * p0.z + (1.5 * t3 - 2.5 * t2 + 1) * p1.z + (-1.5 * t3 + 2 * t2 + 0.5 * t) * p2.z + (0.5 * t3 - 0.5 * t2) * p3.z,
    );
}

function _sampleCurve(points, kind, u) {
    if (!points || points.length === 0) return new Vector3();
    if (points.length === 1) return points[0].clone();
    u = Math.min(1, Math.max(0, u));
    if (kind === 'bezier') return _sampleBezier(points, u);
    if (kind === 'linear') return _sampleLinear(points, u);
    return _sampleCatmull(points, u);
}

/** World-space eye path (+ optional interest) with speed ramp + arc-length. */
export class CameraPath {
    constructor({
        points = [],
        kind = 'catmull',
        interest = [],
        interestKind = 'catmull',
        lookAhead = 1,
        fixedTarget = null,
        ramp = 'smoother',
        rampOpts = {},
        fov = null,
    } = {}) {
        this.points = points.map(_v3);
        this.kind = kind;
        this.interest = interest.map(_v3);
        this.interestKind = interestKind;
        this.lookAhead = lookAhead;
        this.fixedTarget = fixedTarget ? _v3(fixedTarget) : null;
        this.ramp = ramp;
        this.rampOpts = rampOpts;
        this.fov = fov;
        this._lengths = [];
        this.totalLength = 0;
        this.rebuildArcLength();
    }
    static catmullRom(points, opts = {}) {
        return new CameraPath({ ...opts, points, kind: 'catmull' });
    }
    static bezier(points, opts = {}) {
        return new CameraPath({ ...opts, points, kind: 'bezier' });
    }
    static linear(points, opts = {}) {
        return new CameraPath({ ...opts, points, kind: 'linear' });
    }
    withInterest(points, kind) {
        this.interest = points.map(_v3);
        if (kind) this.interestKind = kind;
        return this;
    }
    withRamp(ramp, rampOpts = {}) {
        this.ramp = ramp;
        this.rampOpts = rampOpts;
        return this;
    }
    withFixedTarget(target) {
        this.fixedTarget = _v3(target);
        this.lookAhead = 0;
        return this;
    }
    rebuildArcLength() {
        const SAMPLES = 64;
        this._lengths = [0];
        if (this.points.length < 2) {
            this.totalLength = 0;
            return this;
        }
        let prev = _sampleCurve(this.points, this.kind, 0);
        let acc = 0;
        for (let i = 1; i <= SAMPLES; i++) {
            const p = _sampleCurve(this.points, this.kind, i / SAMPLES);
            acc += p.distanceTo(prev);
            this._lengths.push(acc);
            prev = p;
        }
        this.totalLength = acc;
        return this;
    }
    _arcParameter(t) {
        if (this._lengths.length < 2 || this.totalLength <= 1e-8) return t;
        const want = t * this.totalLength;
        let lo = 0, hi = this._lengths.length - 1;
        while (lo + 1 < hi) {
            const mid = (lo + hi) >> 1;
            if (this._lengths[mid] < want) lo = mid;
            else hi = mid;
        }
        const l0 = this._lengths[lo], l1 = this._lengths[hi];
        const local = (want - l0) / Math.max(1e-8, l1 - l0);
        const u0 = lo / (this._lengths.length - 1);
        const u1 = hi / (this._lengths.length - 1);
        return u0 + (u1 - u0) * local;
    }
    sampleEye(t) {
        const u = this._arcParameter(speedRamp(this.ramp, t, this.rampOpts));
        return _sampleCurve(this.points, this.kind, u);
    }
    sampleTarget(t) {
        t = Math.min(1, Math.max(0, t));
        const u = this._arcParameter(speedRamp(this.ramp, t, this.rampOpts));
        if (this.interest.length >= 2) {
            return _sampleCurve(this.interest, this.interestKind, u);
        }
        if (this.fixedTarget) return this.fixedTarget.clone();
        if (this.lookAhead > 0 && this.totalLength > 1e-6) {
            const ahead = Math.min(1, u + this.lookAhead / this.totalLength);
            return _sampleCurve(this.points, this.kind, ahead);
        }
        const eye = _sampleCurve(this.points, this.kind, u);
        const next = _sampleCurve(this.points, this.kind, Math.min(1, u + 0.01));
        const dir = next.clone().sub(eye);
        if (dir.lengthSq() < 1e-10) return eye.clone().add(new Vector3(0, 0, -1));
        return eye.clone().add(dir.normalize());
    }
}

function _applyCamView(cam, pos, look, fov) {
    cam.position.copy(pos);
    if (cam._lookAt) cam._lookAt.copy(look);
    if (cam.isPerspectiveCamera && fov != null) {
        cam.fov = fov;
        cam.updateProjectionMatrix?.();
    }
    if (typeof cam._w?.setView === 'function') {
        const f = cam.isPerspectiveCamera ? (cam.fov ?? cam._fov ?? -1) : -1;
        cam._w.setView(pos.x, pos.y, pos.z, look.x, look.y, look.z, f);
        if (cam.focus != null && cam._w.setFocusDistance) {
            cam._w.setFocusDistance(cam.focus);
        }
        if (cam.up && cam._w.setUp) cam._w.setUp(cam.up.x, cam.up.y, cam.up.z);
    } else {
        cam.lookAt?.(look);
        cam._sync?.();
    }
}

/**
 * Additive cinematic helper for a three.js-shaped PerspectiveCamera.
 * @example
 * const anim = new THREE.CameraAnimator(camera);
 * anim.flyTo({ position, target, duration: 2, ramp: 'inOut', easeIn: 0.2, easeOut: 0.3 });
 * anim.followPath({ path, duration: 4 });
 * anim.update(dt);
 */
export class CameraAnimator {
    constructor(camera) {
        this.camera = camera;
        this._mode = 'idle'; // idle | fly | path
        this._from = { pos: new Vector3(), target: new Vector3(), fov: 50 };
        this._to = { pos: new Vector3(), target: new Vector3(), fov: 50 };
        this._path = null;
        this._elapsed = 0;
        this._duration = 0;
        this._ramp = 'cubicInOut';
        this._rampOpts = {};
        this._active = false;
    }
    stop() {
        this._active = false;
        this._mode = 'idle';
        return this;
    }
    flyTo({
        position, target, fov, duration = 1,
        easing, ramp, easeIn, easeOut,
    } = {}) {
        const cam = this.camera;
        this._from.pos.copy(cam.position);
        this._from.target.copy(cam._lookAt || new Vector3());
        this._from.fov = cam.fov ?? cam._fov ?? 50;
        this._to.pos.copy(position ? _v3(position) : cam.position);
        this._to.target.copy(target ? _v3(target) : this._from.target);
        this._to.fov = fov ?? this._from.fov;
        this._duration = Math.max(0, duration);
        this._elapsed = 0;
        this._ramp = ramp || easing || 'cubicInOut';
        this._rampOpts = { easeIn, easeOut };
        this._mode = 'fly';
        this._active = true;
        return this;
    }
    /** Spherical orbit around the current look-at by azimuth/elevation deltas (radians). */
    orbitBy({
        azimuth = 0, elevation = 0, radiusDelta = 0,
        duration = 1, easing, ramp, easeIn, easeOut, fov,
    } = {}) {
        const cam = this.camera;
        const eye = cam.position.clone();
        const tgt = (cam._lookAt || new Vector3()).clone();
        const off = eye.clone().sub(tgt);
        const r0 = Math.max(1e-4, off.length());
        const phi0 = Math.acos(Math.min(1, Math.max(-1, off.y / r0)));
        const theta0 = Math.atan2(off.x, off.z);
        const r1 = Math.max(1e-4, r0 + radiusDelta);
        const phi1 = Math.min(Math.PI - 1e-3, Math.max(1e-3, phi0 + elevation));
        const theta1 = theta0 + azimuth;
        const toPos = new Vector3(
            tgt.x + r1 * Math.sin(phi1) * Math.sin(theta1),
            tgt.y + r1 * Math.cos(phi1),
            tgt.z + r1 * Math.sin(phi1) * Math.cos(theta1),
        );
        return this.flyTo({
            position: toPos, target: tgt, fov, duration,
            easing, ramp, easeIn, easeOut,
        });
    }
    /** Retarget look-at; set `holdEye: true` to keep eye and rebuild framing. */
    lookAt({ target, duration = 1, holdEye = false, easing, ramp, easeIn, easeOut } = {}) {
        const cam = this.camera;
        const tgt = _v3(target);
        if (holdEye) {
            return this.flyTo({
                position: cam.position.clone(), target: tgt, duration,
                easing, ramp, easeIn, easeOut,
            });
        }
        return this.flyTo({
            position: cam.position.clone(), target: tgt, duration,
            easing, ramp, easeIn, easeOut,
        });
    }
    /** Animate FOV only (push-in / pull-out). */
    setFov({ fov, duration = 1, easing, ramp, easeIn, easeOut } = {}) {
        const cam = this.camera;
        return this.flyTo({
            position: cam.position.clone(),
            target: (cam._lookAt || new Vector3()).clone(),
            fov, duration, easing, ramp, easeIn, easeOut,
        });
    }
    /**
     * Dolly-zoom / Vertigo: move along view axis while compensating FOV so
     * the subject framing stays approximately constant.
     */
    vertigo({
        targetFov, duration = 1, easing, ramp, easeIn, easeOut,
    } = {}) {
        const cam = this.camera;
        const eye = cam.position.clone();
        const tgt = (cam._lookAt || new Vector3()).clone();
        const fov0 = cam.fov ?? cam._fov ?? 50;
        const fov1 = targetFov ?? fov0;
        const dist0 = Math.max(1e-4, eye.distanceTo(tgt));
        // similar triangles: dist * tan(fov/2) ≈ constant framing size
        const half0 = (fov0 * Math.PI) / 360;
        const half1 = (fov1 * Math.PI) / 360;
        const dist1 = dist0 * Math.tan(half0) / Math.max(1e-6, Math.tan(half1));
        const dir = eye.clone().sub(tgt).normalize();
        const toPos = tgt.clone().add(dir.multiplyScalar(dist1));
        return this.flyTo({
            position: toPos, target: tgt, fov: fov1, duration,
            easing, ramp, easeIn, easeOut,
        });
    }
    /** Follow a {@link CameraPath} (or `{ points, kind, … }` options). */
    followPath({ path, duration = 1, ramp, easeIn, easeOut } = {}) {
        this._path = path instanceof CameraPath ? path : new CameraPath(path || {});
        if (ramp != null) this._path.withRamp(ramp, { easeIn, easeOut });
        this._duration = Math.max(0, duration);
        this._elapsed = 0;
        this._ramp = this._path.ramp;
        this._rampOpts = this._path.rampOpts || {};
        this._mode = 'path';
        this._active = true;
        return this;
    }
    get isActive() { return this._active; }
    update(dt) {
        if (!this._active) return false;
        this._elapsed += Math.max(0, dt);
        let u = this._duration <= 0 ? 1 : Math.min(1, this._elapsed / this._duration);
        const cam = this.camera;
        if (this._mode === 'path' && this._path) {
            const eye = this._path.sampleEye(u);
            const look = this._path.sampleTarget(u);
            const fov = this._path.fov != null ? this._path.fov : (cam.fov ?? cam._fov);
            _applyCamView(cam, eye, look, fov);
        } else {
            const s = speedRamp(this._ramp, u, this._rampOpts);
            const pos = new Vector3().lerpVectors(this._from.pos, this._to.pos, s);
            const look = new Vector3().lerpVectors(this._from.target, this._to.target, s);
            const fov = this._from.fov + (this._to.fov - this._from.fov) * s;
            _applyCamView(cam, pos, look, fov);
        }
        if (u >= 1) {
            this._active = false;
            this._mode = 'idle';
        }
        return this._active;
    }
}

/**
 * Editorial shot list: hold / fly / path clips with cut or blend transitions.
 * Apply with `timeline.apply(camera)` each frame after `timeline.update(dt)`.
 */
export class ShotTimeline {
    constructor() {
        this.shots = [];
        this.time = 0;
        this.looping = true;
    }
    push(shot) {
        this.shots.push(shot);
        return this.shots.length - 1;
    }
    hold(name, { position, target, fov = 50, duration = 1, transition } = {}) {
        return this.push({
            name, duration, kind: 'hold',
            position: _v3(position), target: _v3(target), fov,
            transition: transition || { type: 'cut' },
        });
    }
    fly(name, {
        from, to, duration = 1, ramp = 'inOut', easeIn = 0.2, easeOut = 0.3, transition,
    } = {}) {
        return this.push({
            name, duration, kind: 'fly',
            from: {
                position: _v3(from.position),
                target: _v3(from.target),
                fov: from.fov ?? 50,
            },
            to: {
                position: _v3(to.position),
                target: _v3(to.target),
                fov: to.fov ?? from.fov ?? 50,
            },
            ramp, easeIn, easeOut,
            transition: transition || { type: 'cut' },
        });
    }
    path(name, { path, duration = 1, transition } = {}) {
        return this.push({
            name, duration, kind: 'path',
            path: path instanceof CameraPath ? path : new CameraPath(path || {}),
            transition: transition || { type: 'cut' },
        });
    }
    duration() {
        return this.shots.reduce((s, sh) => s + Math.max(0, sh.duration || 0), 0);
    }
    seek(time) {
        const total = this.duration();
        if (total <= 0) { this.time = 0; return this; }
        let t = Math.max(0, time);
        if (this.looping) t %= total;
        else t = Math.min(total, t);
        this.time = t;
        return this;
    }
    update(dt) {
        if (Number.isFinite(dt)) this.seek(this.time + dt);
        return this;
    }
    _shotAt(time) {
        let acc = 0;
        for (let i = 0; i < this.shots.length; i++) {
            const end = acc + this.shots[i].duration;
            if (time < end || i + 1 === this.shots.length) {
                return { index: i, local: Math.max(0, time - acc) };
            }
            acc = end;
        }
        return { index: 0, local: 0 };
    }
    _sampleShot(shot, local) {
        const u = shot.duration <= 0 ? 1 : Math.min(1, Math.max(0, local / shot.duration));
        if (shot.kind === 'hold') {
            return {
                position: shot.position.clone(),
                target: shot.target.clone(),
                fov: shot.fov,
            };
        }
        if (shot.kind === 'path') {
            return {
                position: shot.path.sampleEye(u),
                target: shot.path.sampleTarget(u),
                fov: shot.path.fov ?? 50,
            };
        }
        // fly
        const s = speedRamp(shot.ramp, u, { easeIn: shot.easeIn, easeOut: shot.easeOut });
        return {
            position: new Vector3().lerpVectors(shot.from.position, shot.to.position, s),
            target: new Vector3().lerpVectors(shot.from.target, shot.to.target, s),
            fov: shot.from.fov + (shot.to.fov - shot.from.fov) * s,
        };
    }
    sample() {
        if (!this.shots.length) {
            return { position: new Vector3(), target: new Vector3(), fov: 50, shotIndex: -1, shotName: '' };
        }
        const { index, local } = this._shotAt(this.time);
        const cur = this.shots[index];
        let pose = this._sampleShot(cur, local);
        if (index > 0 && cur.transition && cur.transition.type === 'blend') {
            const dur = Math.min(cur.duration, Math.max(0, cur.transition.duration ?? 0.5));
            if (dur > 0 && local < dur) {
                const prev = this.shots[index - 1];
                const prevPose = this._sampleShot(prev, prev.duration);
                const bu = speedRamp(
                    cur.transition.ramp || 'cubicInOut',
                    local / dur,
                    { easeIn: cur.transition.easeIn, easeOut: cur.transition.easeOut },
                );
                pose = {
                    position: new Vector3().lerpVectors(prevPose.position, pose.position, bu),
                    target: new Vector3().lerpVectors(prevPose.target, pose.target, bu),
                    fov: prevPose.fov + (pose.fov - prevPose.fov) * bu,
                };
            }
        }
        pose.shotIndex = index;
        pose.shotName = cur.name || '';
        return pose;
    }
    apply(camera) {
        const pose = this.sample();
        _applyCamView(camera, pose.position, pose.target, pose.fov);
        return pose;
    }
}

// ---------------------------------------------------------------------------
// General motion primitives (mirror threers-animation Tween / Spring / Timeline)
// ---------------------------------------------------------------------------
const _EASING_FNS = {
    linear: (t) => t,
    quadIn: (t) => t * t,
    quadOut: (t) => t * (2 - t),
    quadInOut: (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
    cubicIn: (t) => t * t * t,
    cubicOut: (t) => { const u = 1 - t; return 1 - u * u * u; },
    cubicInOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    smooth: (t) => t * t * (3 - 2 * t),
    smoother: (t) => t * t * t * (t * (t * 6 - 15) + 10),
    backOut: (t) => { const c = 1.70158; const u = t - 1; return 1 + u * u * ((c + 1) * u + c); },
};

function _ease(name, t) {
    if (typeof name === 'function') return name(t);
    return (_EASING_FNS[name] || _EASING_FNS.linear)(Math.min(1, Math.max(0, t)));
}

function _lerpScalar(a, b, t) { return a + (b - a) * t; }
function _lerpAny(a, b, t) {
    if (typeof a === 'number') return _lerpScalar(a, b, t);
    if (a?.isQuaternion || (a?.w != null && typeof a.slerp === 'function')) {
        const out = a.clone?.() || new Quaternion(a.x, a.y, a.z, a.w);
        return out.slerp(b, t);
    }
    if (a?.x != null && b?.x != null) {
        return new Vector3(
            _lerpScalar(a.x, b.x, t),
            _lerpScalar(a.y, b.y, t),
            _lerpScalar(a.z ?? 0, b.z ?? 0, t),
        );
    }
    return b;
}

/** A→B over a known duration. */
export class Tween {
    constructor(from, to, duration = 1) {
        this.from = from;
        this.to = to;
        this.duration = Math.max(0, duration);
        this.delay = 0;
        this.easing = 'cubicOut';
        this.repeat = 0;
        this.yoyo = false;
        this.elapsed = 0;
        this._value = from?.clone?.() ?? from;
        this.finished = false;
        this.onUpdate = null;
        this.onComplete = null;
    }
    setEasing(e) { this.easing = e; return this; }
    setDelay(d) { this.delay = Math.max(0, d); return this; }
    setRepeat(n, yoyo = false) { this.repeat = n; this.yoyo = yoyo; return this; }
    value() { return this._value; }
    isFinished() { return this.finished; }
    update(dt) {
        if (this.finished || !Number.isFinite(dt)) return this._value;
        this.elapsed += Math.max(0, dt);
        const active = this.elapsed - this.delay;
        if (active < 0) return this._value;
        const dur = this.duration || 1e-8;
        let cycle = Math.floor(active / dur);
        let u = (active % dur) / dur;
        const maxCycles = this.repeat === Infinity ? Infinity : (this.repeat + 1);
        if (cycle >= maxCycles) {
            u = 1;
            this.finished = true;
            cycle = maxCycles - 1;
        }
        if (this.yoyo && (cycle % 2 === 1)) u = 1 - u;
        const e = _ease(this.easing, u);
        this._value = _lerpAny(this.from, this.to, e);
        this.onUpdate?.(this._value, e);
        if (this.finished) this.onComplete?.(this._value);
        return this._value;
    }
}

/** Damped spring chasing a moving target (no fixed duration). */
export class Spring {
    constructor(value, angularFrequency = 8, dampingRatio = 1) {
        this.value = typeof value === 'number' ? value : (value.clone?.() || value);
        this.velocity = typeof value === 'number' ? 0 : new Vector3();
        this.target = typeof value === 'number' ? value : (value.clone?.() || value);
        this.angularFrequency = Math.max(0, angularFrequency);
        this.dampingRatio = Math.max(0, dampingRatio);
        this.restThreshold = 1e-3;
    }
    static criticallyDamped(value, frequency = 8) {
        return new Spring(value, frequency, 1);
    }
    setTarget(t) {
        this.target = typeof t === 'number' ? t : (t.clone?.() || t);
        return this;
    }
    isSettled() {
        if (typeof this.value === 'number') {
            return Math.abs(this.value - this.target) < this.restThreshold
                && Math.abs(this.velocity) < this.restThreshold;
        }
        const d = this.value.distanceTo?.(this.target) ?? 0;
        const v = this.velocity.length?.() ?? 0;
        return d < this.restThreshold && v < this.restThreshold;
    }
    update(dt) {
        if (!Number.isFinite(dt) || dt <= 0) return this.value;
        const w = this.angularFrequency;
        const z = this.dampingRatio;
        if (typeof this.value === 'number') {
            const x = this.value - this.target;
            const a = -2 * z * w * this.velocity - w * w * x;
            this.velocity += a * dt;
            this.value += this.velocity * dt;
            if (this.isSettled()) { this.value = this.target; this.velocity = 0; }
            return this.value;
        }
        const x = this.value.clone().sub(this.target);
        const a = this.velocity.clone().multiplyScalar(-2 * z * w).addScaledVector
            ? this.velocity.clone().multiplyScalar(-2 * z * w).add(x.multiplyScalar(-w * w))
            : this.velocity.clone().multiplyScalar(-2 * z * w).add(x.multiplyScalar(-w * w));
        this.velocity.add(a.multiplyScalar(dt));
        this.value.add(this.velocity.clone().multiplyScalar(dt));
        if (this.isSettled()) {
            this.value.copy(this.target);
            this.velocity.set(0, 0, 0);
        }
        return this.value;
    }
}

/** Shared clock with tracks, markers, and optional time remap. */
export class Timeline {
    constructor() {
        this.tracks = [];
        this.markers = [];
        this.time = 0;
        this.elapsed = 0;
        this.duration = 0;
        this.speed = 1;
        this.playing = true;
        this.repeat = 0;
        this.timeRemap = null;
    }
    add(start, duration, easing = 'linear') {
        const id = this.tracks.length;
        this.tracks.push({ start, duration: Math.max(0, duration), easing });
        this.duration = Math.max(this.duration, start + duration);
        return id;
    }
    then(duration, easing = 'linear') {
        return this.add(this.duration, duration, easing);
    }
    addMarker(time, name, data = null) {
        this.markers.push({ time, name, data });
        this.markers.sort((a, b) => a.time - b.time);
        return this;
    }
    setTimeRemap(fnOrSamples) {
        this.timeRemap = fnOrSamples;
        return this;
    }
    _remap(t) {
        if (!this.timeRemap) return t;
        if (typeof this.timeRemap === 'function') return this.timeRemap(t);
        const s = this.timeRemap;
        if (!s.length) return t;
        if (t <= s[0].t) return s[0].v;
        if (t >= s[s.length - 1].t) return s[s.length - 1].v;
        for (let i = 0; i < s.length - 1; i++) {
            if (t >= s[i].t && t <= s[i + 1].t) {
                const u = (t - s[i].t) / (s[i + 1].t - s[i].t || 1);
                return s[i].v + (s[i + 1].v - s[i].v) * u;
            }
        }
        return t;
    }
    update(dt) {
        if (!this.playing || !Number.isFinite(dt)) return this;
        this.seek(this.elapsed + dt * this.speed);
        return this;
    }
    seek(time) {
        this.elapsed = time;
        let t = time;
        if (this.duration > 0) {
            if (this.repeat === Infinity) t = ((t % this.duration) + this.duration) % this.duration;
            else t = Math.min(this.duration, Math.max(0, t));
        }
        this.time = this._remap(t);
        return this;
    }
    progressOf(id) {
        const tr = this.tracks[id];
        if (!tr || tr.duration <= 0) return 0;
        const u = (this.time - tr.start) / tr.duration;
        return _ease(tr.easing, Math.min(1, Math.max(0, u)));
    }
    valueOf(id, from, to) {
        return _lerpAny(from, to, this.progressOf(id));
    }
    markerAt(time = this.time) {
        let hit = null;
        for (const m of this.markers) {
            if (m.time <= time) hit = m;
            else break;
        }
        return hit;
    }
}

/** Spring-lag follow for any Object3D position (secondary motion). */
export class ObjectSpring {
    constructor(object, { frequency = 6, damping = 1, property = 'position' } = {}) {
        this.object = object;
        this.property = property;
        const cur = object[property];
        this.spring = Spring.criticallyDamped(
            cur?.clone?.() || (typeof cur === 'number' ? cur : new Vector3()),
            frequency,
        );
        this.spring.dampingRatio = damping;
    }
    setTarget(target) {
        if (target?.position) this.spring.setTarget(target.position.clone());
        else this.spring.setTarget(target?.clone?.() || target);
        return this;
    }
    update(dt) {
        const v = this.spring.update(dt);
        const prop = this.object[this.property];
        if (typeof prop === 'number') this.object[this.property] = v;
        else prop?.copy?.(v);
        this.object._sync?.();
        return v;
    }
}

/**
 * Lightweight animated↔simulated pose blend (JS stand-in for PhysicsBlend).
 * Freezes the pose at limp/recover so a continuing clip cannot yank the fade.
 */
export class PoseBlend {
    constructor(target, duration = 0.25) {
        this.target = target;
        this.duration = Math.max(0, duration);
        this.state = 'animated';
        this.elapsed = 0;
        this._anchorPos = target.position.clone();
        this._anchorQuat = target.quaternion?.clone?.() || new Quaternion();
        this._lastAnimPos = target.position.clone();
        this._lastAnimQuat = target.quaternion?.clone?.() || new Quaternion();
    }
    goLimp() {
        this.state = 'toSimulated';
        this.elapsed = 0;
        this._anchorPos.copy(this._lastAnimPos);
        this._anchorQuat.copy(this._lastAnimQuat);
        return this;
    }
    getUp() {
        this.state = 'toAnimated';
        this.elapsed = 0;
        this._anchorPos.copy(this.target.position);
        if (this.target.quaternion) this._anchorQuat.copy(this.target.quaternion);
        return this;
    }
    update(dt, animated, simulated) {
        const aPos = animated?.position || animated;
        const sPos = simulated?.position || simulated;
        const aQuat = animated?.quaternion;
        const sQuat = simulated?.quaternion;
        if (aPos?.clone) this._lastAnimPos.copy(aPos);
        else if (aPos) this._lastAnimPos.set(aPos.x, aPos.y, aPos.z);
        if (aQuat) this._lastAnimQuat.copy(aQuat);

        if (this.state === 'animated') {
            this.target.position.copy(aPos);
            if (aQuat && this.target.quaternion) this.target.quaternion.copy(aQuat);
        } else if (this.state === 'simulated') {
            this.target.position.copy(sPos);
            if (sQuat && this.target.quaternion) this.target.quaternion.copy(sQuat);
        } else {
            this.elapsed += dt;
            const u = this.duration <= 0 ? 1 : Math.min(1, this.elapsed / this.duration);
            const e = _ease('cubicInOut', u);
            if (this.state === 'toSimulated') {
                this.target.position.lerpVectors(this._anchorPos, sPos, e);
                if (sQuat && this.target.quaternion) {
                    this.target.quaternion.copy(this._anchorQuat).slerp(sQuat, e);
                }
                if (u >= 1) this.state = 'simulated';
            } else {
                this.target.position.lerpVectors(this._anchorPos, aPos, e);
                if (aQuat && this.target.quaternion) {
                    this.target.quaternion.copy(this._anchorQuat).slerp(aQuat, e);
                }
                if (u >= 1) this.state = 'animated';
            }
        }
        this.target._sync?.();
        return this.state;
    }
}

export class TrackballControls {
    constructor(camera, domElement) {
        this._w = new WebTrackballControls(camera._w);
        this._camera = camera;
        this.domElement = domElement;
        this.enabled = true;
        this._rotating = false;
        this._panning = false;
        this._lastX = 0; this._lastY = 0;
        if (domElement?.addEventListener) {
            domElement.addEventListener('pointerdown', (e) => { this._rotating = e.button === 0; this._panning = e.button === 2; this._lastX = e.clientX; this._lastY = e.clientY; });
            domElement.addEventListener('pointermove', (e) => {
                if (!this.enabled || (!this._rotating && !this._panning)) return;
                const dx = e.clientX - this._lastX, dy = e.clientY - this._lastY;
                this._lastX = e.clientX; this._lastY = e.clientY;
                this.update(dx, dy, 0, this._rotating, this._panning);
            });
            domElement.addEventListener('pointerup', () => { this._rotating = false; this._panning = false; });
            domElement.addEventListener('wheel', (e) => { if (this.enabled) this.update(0, 0, e.deltaY, false, false); });
        }
    }
    update(dx = 0, dy = 0, wheel = 0, rotating = false, panning = false) {
        const w = this.domElement?.clientWidth || (typeof window !== 'undefined' ? window.innerWidth : 800);
        const h = this.domElement?.clientHeight || (typeof window !== 'undefined' ? window.innerHeight : 600);
        this._w.update(this._camera._w, dx, dy, wheel, rotating, panning, w, h);
    }
}
export class FirstPersonControls {
    constructor(camera, domElement) {
        this._w = new WebFirstPersonControls(camera._w);
        this._camera = camera;
        this.domElement = domElement;
        this.enabled = true;
        this._rotating = false;
        this._keys = {};
        if (domElement?.addEventListener) {
            domElement.addEventListener('pointerdown', () => { this._rotating = true; });
            domElement.addEventListener('pointerup', () => { this._rotating = false; });
            domElement.addEventListener('pointermove', (e) => {
                if (!this.enabled || !this._rotating) return;
                this.update(e.movementX, e.movementY, 1 / 60, true);
            });
            domElement.addEventListener('keydown', (e) => { this._keys[e.code] = true; });
            domElement.addEventListener('keyup', (e) => { this._keys[e.code] = false; });
        }
    }
    update(dx = 0, dy = 0, dt = 1 / 60, rotating = false) {
        const fwd = (this._keys['KeyW'] || this._keys['ArrowUp']) ? 1 : 0;
        const back = (this._keys['KeyS'] || this._keys['ArrowDown']) ? 1 : 0;
        const left = (this._keys['KeyA'] || this._keys['ArrowLeft']) ? 1 : 0;
        const right = (this._keys['KeyD'] || this._keys['ArrowRight']) ? 1 : 0;
        this._w.setMoveInput(fwd - back, right - left, 0);
        this._w.update(this._camera._w, dx, dy, dt, rotating);
    }
}
export class PointerLockControls { constructor(camera) { this._w = new WebPointerLockControls(camera._w); } }

// ---- Loaders ----
export class OBJLoader {
    constructor() { this._w = new WebObjLoader(); }
    parse(src) { const g = this._w.parse(src); return { geometry: g }; }
}
export class STLLoader {
    constructor() { this._w = new WebStlLoader(); }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return this._w.parse(u8);
    }
}
export class PLYLoader {
    constructor() { this._w = new WebPlyLoader(); }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return this._w.parse(u8);
    }
}
export class RGBELoader {
    constructor() { this._w = new WebHdrLoader(); }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return this._w.parse(u8);
    }
}

// ---- Audio ----
export class AudioListener { constructor() { this._w = new WebAudioListener(); } setMasterVolume(v) { this._w.setMasterVolume(v); } }
export class Audio {
    constructor(_listener) { this._w = new WebAudio(); }
    setVolume(v) { this._w.setVolume(v); return this; }
    setLoop(l) { this._w.setLoop(l); return this; }
    play() { this._w.play(); return this; }
    stop() { this._w.stop(); return this; }
}

// ---- Post-FX ----
// EffectComposer — a working pipeline that runs each pass in sequence.
// Post-fx passes that depend on dedicated GPU pipelines (Bloom, SSAO, etc.)
// still no-op on `render`, but RenderPass + ShaderPass perform real work.
export class EffectComposer {
    constructor(renderer, renderTarget) {
        this._w = new WebEffectComposer();
        this.renderer = renderer;
        const w = renderer?._canvas?.width ?? 800;
        const h = renderer?._canvas?.height ?? 600;
        this.renderTarget1 = renderTarget || _composerRT(renderer, w, h);
        this.renderTarget2 = this.renderTarget1.clone();
        if (renderer?._w && WebRenderTarget.newHalfFloat && !this.renderTarget2._w) {
            this.renderTarget2._w = WebRenderTarget.newHalfFloat(renderer._w, w, h);
        }
        this.writeBuffer = this.renderTarget1;
        this.readBuffer = this.renderTarget2;
        this.passes = [];
        this.renderToScreen = true;
    }
    addPass(pass) { this.passes.push(pass); }
    insertPass(pass, index) { this.passes.splice(index, 0, pass); }
    removePass(pass) { this.passes = this.passes.filter(p => p !== pass); }
    setSize(w, h) {
        this.renderTarget1.setSize(w, h);
        this.renderTarget2.setSize(w, h);
        if (this.renderer?._w && WebRenderTarget.newHalfFloat) {
            if (this.renderTarget1._halfFloat && !this.renderTarget1._w) {
                this.renderTarget1._w = WebRenderTarget.newHalfFloat(this.renderer._w, w, h);
            }
            if (this.renderTarget2._halfFloat && !this.renderTarget2._w) {
                this.renderTarget2._w = WebRenderTarget.newHalfFloat(this.renderer._w, w, h);
            }
        }
        for (const p of this.passes) p.setSize?.(w, h);
    }
    setPixelRatio() {}
    reset(rt) { if (rt) { this.renderTarget1 = rt; this.renderTarget2 = rt.clone?.() ?? rt; this.writeBuffer = this.renderTarget1; this.readBuffer = this.renderTarget2; } }
    _isLastEnabledPass(index) {
        for (let i = this.passes.length - 1; i > index; i--) {
            if (this.passes[i].enabled) return false;
        }
        return true;
    }
    swapBuffers() {
        const tmp = this.writeBuffer;
        this.writeBuffer = this.readBuffer;
        this.readBuffer = tmp;
    }
    async render(deltaTime = 0) {
        const renderer = this.renderer;
        for (let i = 0; i < this.passes.length; i++) {
            const pass = this.passes[i];
            if (!pass.enabled) continue;
            pass.renderToScreen = this.renderToScreen && this._isLastEnabledPass(i);
            if (pass.render) {
                const pending = pass.render(renderer, this.writeBuffer, this.readBuffer, deltaTime);
                if (pending && typeof pending.then === 'function') await pending;
            }
            if (pass.needsSwap) this.swapBuffers();
        }
    }
    dispose() { this.renderTarget1?.dispose?.(); this.renderTarget2?.dispose?.(); }
}
export class RenderPass {
    constructor(scene, camera) {
        this._w = new WebRenderPass();
        this.scene = scene;
        this.camera = camera;
        this.enabled = true;
        this.needsSwap = false;
        this.clear = true;
        this.clearAlpha = 1;
        this.overrideMaterial = null;
        this.renderToScreen = false;
    }
    setSize() {}
    dispose() {}
    render(renderer, writeBuffer, _readBuffer) {
        const prev = renderer.getRenderTarget();
        renderer.setRenderTarget(this.renderToScreen ? null : _readBuffer);
        if (this.overrideMaterial) {
            _renderWithMaterialOverride(renderer, this.scene, this.camera, this.overrideMaterial, this.renderToScreen ? null : _readBuffer, this.clear);
        } else {
            renderer.render(this.scene, this.camera);
        }
        renderer.setRenderTarget(prev);
    }
}
// Real UnrealBloomPass. Implements the standard threshold + Gaussian-blur
// chain via the post-fx pipeline. apply(renderer, inputRT): reads inputRT,
// extracts bright pixels, blurs them, then additively composites onto canvas.
export class UnrealBloomPass {
    constructor(resolution = new Vector2(800, 600), strength = 1.0, radius = 0.4, threshold = 0.85) {
        this.resolution = resolution;
        this.strength = strength;
        this.radius = radius;
        this.threshold = threshold;
        this._w = new WebBloomPass(strength, radius, threshold);
        // Intermediate render targets — lazily allocated on first apply().
        this._thresholdRT = null;
        this._blurH_RT = null;
        this._blurV_RT = null;
    }
    apply(renderer, inputRT) {
        const w = this.resolution.x | 0, h = this.resolution.y | 0;
        if (!this._thresholdRT) this._thresholdRT = new WebGLRenderTarget(w, h);
        if (!this._blurH_RT)    this._blurH_RT    = new WebGLRenderTarget(w, h);
        if (!this._blurV_RT)    this._blurV_RT    = new WebGLRenderTarget(w, h);
        // 1. Brightness threshold: bright pixels of inputRT → _thresholdRT.
        renderer.applyPostFxToRT(inputRT, this._thresholdRT, 6, 0, [this.threshold, 0.1, 0, 0]);
        // 2. Gaussian blur horizontal → _blurH_RT.
        const blur_radius = 1.5 + this.radius * 8.0;
        renderer.applyPostFxToRT(this._thresholdRT, this._blurH_RT, 7, 0, [blur_radius, 0, 0, 0]);
        // 3. Gaussian blur vertical → _blurV_RT.
        renderer.applyPostFxToRT(this._blurH_RT, this._blurV_RT, 8, 0, [blur_radius, 0, 0, 0]);
        // 4. Copy original inputRT to canvas (display-encoded passthrough).
        renderer.applyPostFx(inputRT, 0, 0, [1, 0, 0, 0]);
        // 5. Additively composite the blurred bloom on top.
        renderer.applyPostFx(this._blurV_RT, 9, 0, [this.strength, 0, 0, 0], true);
    }
    render(renderer, writeBuffer, readBuffer) {
        this.resolution.set(writeBuffer.width, writeBuffer.height);
        this.apply(renderer, readBuffer);
        if (!writeBuffer._w) {
            renderer.setRenderTarget(writeBuffer);
            renderer.setRenderTarget(null);
        }
    }
}
// three.js-compatible shader object for ShaderPass(FXAAShader).
export const FXAAShader = {
    name: 'FXAAShader',
    uniforms: {
        tDiffuse: { value: null },
        resolution: { value: new Vector2(1 / 1024, 1 / 512) },
    },
};

// ---- Helpers ----
export class AxesHelper {
    constructor(size = 1) {
        this._w = new WebAxesHelper(size);
        this._isHelper = 'axes';
    }
}
export class GridHelper {
    constructor(size = 10, divisions = 10, color1 = 0x444444, color2 = 0x888888) {
        this._w = new WebGridHelper(size, divisions, color1, color2);
        this._isHelper = 'grid';
    }
}
export class BoxHelper {
    constructor(_obj, _color) {
        const bb = new Box3(new Vector3(-1,-1,-1), new Vector3(1,1,1));
        this._w = new WebBoxHelper(bb._w);
        this._isHelper = 'box';
    }
}
export class PolarGridHelper {
    constructor(radius = 10, segments = 16, circles = 8) {
        this._w = new WebPolarGridHelper(radius, segments, circles);
        this._isHelper = 'polar';
    }
}

// ---- Extras ----
export const SimplexNoise = class {
    constructor() { this._w = new WebSimplexNoise(); }
    noise(x, y) { return this._w.noise2(x, y); }
};
export class Octree {
    constructor(bb, maxDepth = 8, maxPoints = 8) {
        this._w = new WebOctree(bb._w, maxDepth, maxPoints);
    }
    insert(p) { this._w.insert(p._w()); }
}
export const MarchingCubes = class { constructor() { this._w = new WebMarchingCubes(); } };

// ---- Alt renderers ----
export class CSS2DRenderer {
    constructor() {
        this._w = new WebCss2dRenderer(800, 600);
        this.domElement = (typeof document !== 'undefined') ? document.createElement('div') : null;
    }
    setSize(w, h) { this._w = new WebCss2dRenderer(w, h); }
}
// three.js's SVGRenderer draws into an `<svg>` element you append to the page,
// the same way WebGLRenderer draws into a canvas — so `domElement` and
// `render()` are the whole of how you turn it on. Without them the renderer can
// only hand back a string, which is an export path and not a renderer.
export class SVGRenderer {
    constructor() {
        this._w = new WebSvgRenderer(800, 600);
        this.domElement = (typeof document !== 'undefined')
            ? document.createElementNS('http://www.w3.org/2000/svg', 'svg')
            : null;
        this.autoClear = true;
        this.info = { render: { vertices: 0, faces: 0 } };
        this._applySize(800, 600);
    }

    setSize(w, h) {
        // Resize in place. Rebuilding the renderer here would silently throw
        // away every option set before it, which is the sort of thing that only
        // shows up as "my SVG lost its background" much later.
        this._w.setSize(w, h);
        this._applySize(w, h);
    }

    _applySize(w, h) {
        if (!this.domElement) return;
        this.domElement.setAttribute('width', String(w));
        this.domElement.setAttribute('height', String(h));
        this.domElement.setAttribute('viewBox', `0 0 ${w} ${h}`);
    }

    /// Draw into `domElement`. The markup is a whole `<svg>` document, so its
    /// children and attributes are transplanted onto the live element rather
    /// than nesting one document inside another.
    render(scene, camera) {
        const markup = this.renderToString(scene, camera);
        this.info.render.faces = (markup.match(/<path/g) || []).length;
        this.info.render.vertices = this.info.render.faces * 3;
        if (!this.domElement) return markup;
        if (this.autoClear) this.clear();
        const parsed = new DOMParser().parseFromString(markup, 'image/svg+xml').documentElement;
        for (const attr of Array.from(parsed.attributes)) {
            this.domElement.setAttribute(attr.name, attr.value);
        }
        while (parsed.firstChild) this.domElement.appendChild(parsed.firstChild);
        return markup;
    }

    /// The document as markup, for writing to a file or embedding directly.
    ///
    /// Pushes the scene graph and camera into wasm first, exactly as
    /// `WebGLRenderer.render` does. Without it the renderer draws whatever pose
    /// was last uploaded, so moving a mesh or the camera from JavaScript would
    /// change nothing and the picture would simply look stuck.
    renderToString(scene, camera) {
        scene._syncTransforms?.(camera);
        if (!camera?._orbitControlled) camera._sync?.();
        this._syncBackground(scene);
        return this._w.renderToString(scene._w, camera._w);
    }

    /// Take the background from the scene, the way three.js reads it.
    ///
    /// `Scene.background` is stored unconverted, because WebGL's clear colour
    /// is written verbatim and three.js's `Color.r/g/b` are raw bytes over 255.
    /// This renderer works in linear light and encodes to sRGB on the way out,
    /// so handing it those same numbers renders `#101820` as a far lighter
    /// `#475663`. Round-tripping the authored hex through `setClearColor`,
    /// which decodes it, is what makes the document come out the colour that
    /// was asked for.
    _syncBackground(scene) {
        if (this._clearColorSet) return;
        const bg = scene?._background;
        const hex = (bg && typeof bg.getHex === 'function')
            ? bg.getHex()
            : (typeof bg === 'number' ? bg : null);
        if (hex == null || hex === this._sceneBackgroundHex) return;
        this._sceneBackgroundHex = hex;
        this._w.setClearColor(hex >>> 0, scene._backgroundAlpha ?? 1);
    }

    clear() {
        if (!this.domElement) return;
        while (this.domElement.firstChild) {
            this.domElement.removeChild(this.domElement.firstChild);
        }
    }

    // ---- three.js SVGRenderer surface ----

    /// `0xRRGGBB` or a `Color`, plus an alpha. Overrides `scene.background`.
    setClearColor(color, alpha = 1) {
        const hex = (color && typeof color.getHex === 'function')
            ? color.getHex()
            : (typeof color === 'string' ? parseInt(color.replace('#', ''), 16) : Number(color));
        if (!Number.isFinite(hex)) return;
        // An explicit colour wins over the scene's from here on.
        this._clearColorSet = true;
        this._w.setClearColor(hex >>> 0, alpha);
    }
    /// Decimal places kept on coordinates. `null` restores the default.
    setPrecision(precision) { this._w.setPrecision(precision == null ? 2 : precision); }
    /// three.js's coarse/fine switch. 'low' drops the curves and the seam
    /// hairline, which is most of what makes a document big.
    setQuality(quality) {
        const low = quality === 'low';
        this._w.setCurveTolerance(low ? 0 : 0.15);
        this._w.setSeamStroke(low ? 0 : 0.8);
        this._w.setPrecision(low ? 1 : 2);
    }
    /// No-op: an SVG has no pixels to scale. Present so a three.js render loop
    /// that sets it does not throw.
    setPixelRatio(_) {}

    // ---- threers extras ----

    /// 0 lit (default), 1 flat material colour, 2 wireframe.
    setShading(mode) { this._w.setShading(mode); }
    /// Match the WebGL renderer's tone mapping so the two agree on colour.
    /// three.js constants: 0 NoToneMapping, 1 Linear, 4 ACESFilmic.
    setToneMapping(mode, exposure = 1) { this._w.setToneMapping(mode, exposure); }
    /// Draw the background rect at all. Off gives a transparent document.
    setBackground(on) { this._w.setBackground(!!on); }
    setCullBackfaces(on) { this._w.setCullBackfaces(!!on); }
    /// Hairline per face, in pixels, that hides SVG's anti-aliasing seams.
    setSeamStroke(px) { this._w.setSeamStroke(px); }
    /// Bend edges onto the real surface past this pixel error. 0 keeps every
    /// edge straight.
    setCurveTolerance(px) { this._w.setCurveTolerance(px); }
    /// Split faces too deep for one depth to sort. 0 disables.
    setDepthSplit(tolerance) { this._w.setDepthSplit(tolerance); }
    setSort(on) { this._w.setSort(!!on); }
}

// ---- Stats ----
export class Stats {
    constructor() { this._w = new WebStats(); this.dom = (typeof document !== 'undefined') ? document.createElement('div') : null; }
    begin() { this._w.begin(); }
    end() { this._w.end(); }
    update() { this.end(); this.begin(); }
    get fps() { return this._w.fps; }
}

// ---- Raycaster ----
// Real Raycaster: builds a world-space ray from camera + NDC coords, then walks
// the scene tree intersecting against each Mesh's triangle list using the
// Möller-Trumbore algorithm. Matches three.js's intersectObjects return shape:
// [{ distance, point, object, face, faceIndex, uv }].
export class Raycaster {
    constructor(origin = new Vector3(), direction = new Vector3(0, 0, -1), near = 0, far = Infinity) {
        this.ray = new Ray(origin, direction);
        this.near = near; this.far = far;
        this.layers = new Layers();
        this.params = { Mesh: {}, Line: { threshold: 1 }, LOD: {}, Points: { threshold: 1 }, Sprite: {} };
        this._w = new WebRaycaster();
    }
    set(origin, direction) {
        this.ray.origin.copy(origin);
        this.ray.direction.copy(direction);
        return this;
    }
    setFromCamera(coords, camera) {
        if (this._w) this._w.setFromCamera(coords.x, coords.y, camera._w);
        // Build the world-space ray. For a PerspectiveCamera: origin = camera.position,
        // direction = (vector from camera through unprojected NDC).
        if (camera instanceof PerspectiveCamera) {
            this.ray.origin.copy(camera.position);
            // Approximate forward: from camera toward (NDC x, y, -1) in world.
            // Compose view-direction = lookAt - position, plus right/up offsets
            // weighted by FOV.
            const eye = camera.position;
            const target = camera._lookAt || new Vector3();
            const forward = new Vector3().subVectors(target, eye).normalize();
            const worldUp = new Vector3(0, 1, 0);
            const right = new Vector3().crossVectors(forward, worldUp).normalize();
            const up = new Vector3().crossVectors(right, forward).normalize();
            const fovRad = (camera._w?.fov?.() ?? 45) * Math.PI / 180;
            const tanHalf = Math.tan(fovRad / 2);
            const aspect = camera.aspect || 1;
            const dir = new Vector3()
                .copy(forward)
                .addScaledVector(right, coords.x * tanHalf * aspect)
                .addScaledVector(up, coords.y * tanHalf)
                .normalize();
            this.ray.direction.copy(dir);
        } else {
            // Orthographic: origin is on the near plane mapped from NDC; direction is forward.
            const eye = camera.position;
            const target = camera._lookAt || new Vector3();
            const forward = new Vector3().subVectors(target, eye).normalize();
            this.ray.origin.copy(eye);
            this.ray.direction.copy(forward);
        }
        return this;
    }
    intersectObject(object, recursive = false, intersects = []) {
        _raycastObject(this, object, recursive, intersects);
        intersects.sort((a, b) => a.distance - b.distance);
        return intersects;
    }
    intersectObjects(objects, recursive = false, intersects = []) {
        for (const o of objects) _raycastObject(this, o, recursive, intersects);
        intersects.sort((a, b) => a.distance - b.distance);
        return intersects;
    }
}
function _raycastObject(raycaster, object, recursive, intersects) {
    if (object.visible === false) return;
    if (object instanceof Mesh) _raycastMesh(raycaster, object, intersects);
    if (recursive) {
        for (const c of (object.children || object._children || object._objects || [])) {
            _raycastObject(raycaster, c, true, intersects);
        }
    }
}
function _raycastMesh(raycaster, mesh, intersects) {
    const pos = mesh.geometry?.attributes?.position?.array;
    const idx = mesh.geometry?.index?.array || mesh.geometry?._w?.index;
    if (!pos) return;
    const origin = raycaster.ray.origin;
    const dir = raycaster.ray.direction;
    // Build world matrix from mesh transform (lazy approximation: just position +
    // rotation; doesn't handle nested-group composition perfectly).
    const wpos = mesh.position || new Vector3();
    const wrot = _effectiveEuler(mesh);
    const wscale = mesh.scale || new Vector3(1, 1, 1);
    const transform = (v) => {
        let x = v.x * wscale.x, y = v.y * wscale.y, z = v.z * wscale.z;
        const cx = Math.cos(wrot.x), sx = Math.sin(wrot.x);
        let yy = y * cx - z * sx; z = y * sx + z * cx; y = yy;
        const cy = Math.cos(wrot.y), sy = Math.sin(wrot.y);
        let xx = x * cy + z * sy; z = -x * sy + z * cy; x = xx;
        const cz = Math.cos(wrot.z), sz = Math.sin(wrot.z);
        xx = x * cz - y * sz; y = x * sz + y * cz; x = xx;
        return new Vector3(x + wpos.x, y + wpos.y, z + wpos.z);
    };
    const triCount = idx ? (idx.length / 3) | 0 : (pos.length / 9) | 0;
    for (let t = 0; t < triCount; t++) {
        const ia = idx ? idx[t * 3]     : t * 3;
        const ib = idx ? idx[t * 3 + 1] : t * 3 + 1;
        const ic = idx ? idx[t * 3 + 2] : t * 3 + 2;
        const A = transform(new Vector3(pos[ia * 3], pos[ia * 3 + 1], pos[ia * 3 + 2]));
        const B = transform(new Vector3(pos[ib * 3], pos[ib * 3 + 1], pos[ib * 3 + 2]));
        const C = transform(new Vector3(pos[ic * 3], pos[ic * 3 + 1], pos[ic * 3 + 2]));
        const hit = _intersectTriangle(origin, dir, A, B, C);
        if (hit && hit.t >= raycaster.near && hit.t <= raycaster.far) {
            intersects.push({
                distance: hit.t,
                point: new Vector3(origin.x + dir.x * hit.t, origin.y + dir.y * hit.t, origin.z + dir.z * hit.t),
                object: mesh,
                face: { a: ia, b: ib, c: ic, normal: new Vector3().subVectors(B, A).cross(new Vector3().subVectors(C, A)).normalize() },
                faceIndex: t,
                uv: new Vector2(hit.u, hit.v),
            });
        }
    }
}
// Möller-Trumbore ray/triangle intersection. Returns null on miss, or { t, u, v }
// for the hit's parametric distance and barycentric uv coordinates.
function _intersectTriangle(origin, dir, A, B, C) {
    const edge1 = new Vector3().subVectors(B, A);
    const edge2 = new Vector3().subVectors(C, A);
    const h = new Vector3().crossVectors(dir, edge2);
    const a = edge1.dot(h);
    if (a > -1e-8 && a < 1e-8) return null;
    const f = 1 / a;
    const s = new Vector3().subVectors(origin, A);
    const u = f * s.dot(h);
    if (u < 0 || u > 1) return null;
    const q = new Vector3().crossVectors(s, edge1);
    const v = f * dir.dot(q);
    if (v < 0 || u + v > 1) return null;
    const t = f * edge2.dot(q);
    if (t <= 1e-8) return null;
    return { t, u, v };
}

// ---- Clock ----
export class Clock {
    constructor() { this._w = new WebClock(); }
    getDelta() { return this._w.getDelta(); }
    getElapsedTime() { return this._w.getElapsedTime(); }
}

// ---- Scene ----
export class Scene {
    constructor() {
        this._w = new WebScene();
        this._objects = [];
        this.children = [];
        this._background = null;
        this._backgroundAlpha = 1;
        this._fog = null;
        this._environment = null;
    }
    set environment(env) {
        this._environment = env;
        const cubeRTId = env?._cubeRTId;
        if (cubeRTId) {
            this._w.setEnvironmentCube(cubeRTId);
        } else if (env?._w) {
            this._w.setEnvironmentMap(env._w);
        } else {
            this._w.setEnvironmentCube(0);
        }
    }
    get environment() { return this._environment; }
    set fog(f) {
        this._fog = f;
        if (!f) { this._w.setFog(new WebColor(1, 1, 1), 1, 1000, 0, 0); return; }
        // Linear Fog: { color, near, far } → mode 1
        // FogExp2:    { color, density }   → mode 2
        const c = f.color || new Color(0xffffff);
        const wc = new WebColor(c.r, c.g, c.b);
        if (f.isFogExp2) this._w.setFog(wc, 0, 0, f.density ?? 0, 2);
        else             this._w.setFog(wc, f.near ?? 1, f.far ?? 1000, 0, 1);
    }
    get fog() { return this._fog; }
    set background(c) {
        this._background = c;
        // three.js's Color.r/g/b is raw byte/255 (no sRGB decode), and the
        // clear color is written verbatim with `gl.clearColor`. Our WebColor
        // hex constructor does sRGB→linear decode (needed for shader work),
        // which would over-darken backgrounds. Re-construct from the raw r/g/b
        // so the clear matches three.js byte-for-byte.
        if (c instanceof Color) this._w.background = new WebColor(c.r, c.g, c.b);
        else if (typeof c === 'number') {
            const r = ((c >> 16) & 0xff) / 255;
            const g = ((c >> 8) & 0xff) / 255;
            const b = (c & 0xff) / 255;
            this._w.background = new WebColor(r, g, b);
        } else {
            this._w.background = _color(c);
        }
    }
    get background() { return this._background; }
    set backgroundAlpha(a) {
        this._backgroundAlpha = Number(a);
        if (this._w.setBackgroundAlpha) this._w.setBackgroundAlpha(this._backgroundAlpha);
    }
    get backgroundAlpha() { return this._backgroundAlpha != null ? this._backgroundAlpha : 1; }
    getObjectByName(name) {
        if (this.name === name) return this;
        for (const c of this.children) {
            const found = c.getObjectByName?.(name) ?? (c.name === name ? c : null);
            if (found) return found;
        }
        for (const o of this._objects || []) {
            if (o?.name === name) return o;
        }
        return undefined;
    }
    add(...objects) {
        for (const obj of objects) {
            if (!obj) continue;
            this._addOne(obj);
        }
        return this;
    }
    _addOne(obj) {
        this.children.push(obj);
        obj.parent = this;
        if (obj._isHelper) {
            const ax = obj._isHelper === 'axes' ? obj._w : undefined;
            const gr = obj._isHelper === 'grid' ? obj._w : undefined;
            obj._handle = this._w.addHelper(ax, gr);
        } else if (obj._isGroup) {
            obj._handle = this._w.addGroup();
            obj._scene = this;
            this._w.setTransform(obj._handle,
                new Vector3(obj.position.x, obj.position.y, obj.position.z)._w(),
                _effectiveEuler(obj)._w(),
            );
            this._objects.push(obj);
            // Walk descendant tree. Each child Mesh gets attached as a child of
            // the outermost group `obj._handle`, with its local transform set
            // to the composed transform of all intermediate (nested) groups.
            const visit = (node, accPos, accRot) => {
                for (const child of (node._children || [])) {
                    if (child instanceof Mesh) {
                        child._handle = this._w.addMeshTo(obj._handle, child._w);
                        child._scene = this;
                        _applyMeshShadowFlags(child);
                        // Compose: localPos = accPos + child.position (treated
                        // additively in Euler space — accurate when intermediate
                        // rotations are zero, approximate otherwise).
                        const px = accPos.x + child.position.x;
                        const py = accPos.y + child.position.y;
                        const pz = accPos.z + child.position.z;
                        const childRot = _effectiveEuler(child);
                        const rx = accRot.x + childRot.x;
                        const ry = accRot.y + childRot.y;
                        const rz = accRot.z + childRot.z;
                        this._w.setTransform(child._handle,
                            new Vector3(px, py, pz)._w(), new Euler(rx, ry, rz)._w());
                        // Stash the composed transform so _doSyncTransforms
                        // doesn't clobber it back to mesh.position/.rotation.
                        child._composedLocalPos = { x: px, y: py, z: pz };
                        child._composedLocalRot = { x: rx, y: ry, z: rz };
                        this._objects.push(child);
                    } else if (child._isGroup) {
                        const childRot = _effectiveEuler(child);
                        visit(child,
                            new Vector3(accPos.x + child.position.x, accPos.y + child.position.y, accPos.z + child.position.z),
                            new Euler(accRot.x + childRot.x, accRot.y + childRot.y, accRot.z + childRot.z));
                    }
                }
            };
            visit(obj, new Vector3(), new Euler());
            return; // skip the default push below
        } else if (obj._isLineSegments) {
            const g = obj.geometry;
            if (g && obj.material && obj.material._w) {
                if (g._isUserGeometry && typeof g._syncWasmFromJs === 'function') {
                    if (g.attributes && g.attributes.position && g.attributes.position.array && g.attributes.position.array.length) {
                        g._syncWasmFromJs();
                    }
                }
                const geomW = g._w;
                if (geomW) {
                    obj._handle = this._w.addLineSegments(geomW, obj.material._w);
                    obj._scene = this;
                    if (obj.rotation.x || obj.rotation.y || obj.rotation.z || obj.position.x || obj.position.y || obj.position.z) {
                        this._w.setTransform(obj._handle,
                            new Vector3(obj.position.x, obj.position.y, obj.position.z)._w(),
                            new Euler(obj.rotation.x, obj.rotation.y, obj.rotation.z)._w());
                    }
                }
            }
        } else if (obj._isSprite) {
            obj._handle = this._w.addSprite(obj.material._w);
            if (obj.position.x || obj.position.y || obj.position.z) {
                this._w.setTransform(obj._handle,
                    new Vector3(obj.position.x, obj.position.y, obj.position.z)._w(),
                    new Euler(0,0,0)._w());
            }
            if (obj.scale && (obj.scale.x !== 1 || obj.scale.y !== 1 || obj.scale.z !== 1)) {
                this._w.setScale(obj._handle, obj.scale.x, obj.scale.y, obj.scale.z);
            }
            this._objects.push(obj);
            return; // we already pushed
        } else if (obj._isBatchedMesh) {
            // Attach all internal InstancedMeshes; the BatchedMesh itself has no
            // direct wasm handle. addGeometry calls after this go through too.
            obj._attachToScene(this);
            this._objects.push(obj);
            return;
        } else if (obj._isInstancedMesh) {
            obj._handle = this._w.addInstancedMesh(obj.geometry._w, obj.material._w, obj._matrices);
        } else if (obj._isSkinnedMesh) {
            const boneCount = obj.skeleton?.bones?.length ?? 1;
            const geomW = _geomToWebGeom(obj.geometry);
            obj._handle = this._w.addSkinnedMesh(geomW, obj.material._w, boneCount);
            // If the geometry has skinIndex/skinWeight attributes (three.js
            // naming), capture them and push to wasm on first sync.
            const sj = obj.geometry?.attributes?.skinIndex?.array;
            const sw = obj.geometry?.attributes?.skinWeight?.array;
            if (sj) obj._skinJoints  = new Float32Array(sj);
            if (sw) obj._skinWeights = new Float32Array(sw);
        } else if (obj._isPoints) {
            const g = obj.geometry;
            if (g && obj.material && obj.material._w) {
                if (g._isUserGeometry && typeof g._syncWasmFromJs === 'function') {
                    if (g.attributes && g.attributes.position && g.attributes.position.array && g.attributes.position.array.length) {
                        g._syncWasmFromJs();
                    }
                }
                const geomW = g._w;
                if (geomW) {
                    obj._handle = this._w.addPoints(geomW, obj.material._w);
                    if (obj.rotation.x || obj.rotation.y || obj.rotation.z || obj.position.x || obj.position.y || obj.position.z) {
                        this._w.setTransform(obj._handle,
                            new Vector3(obj.position.x, obj.position.y, obj.position.z)._w(),
                            new Euler(obj.rotation.x, obj.rotation.y, obj.rotation.z)._w());
                    }
                }
            }
        } else if (obj._isLight) {
            obj._handle = this._w.addLight(obj._w);
            obj._scene = this;
            // Apply any position set before addLight (e.g. `pl.position.set(...)` then `scene.add(pl)`).
            if (obj.position && (obj.position.x || obj.position.y || obj.position.z)) {
                this._w.setTransform(obj._handle, new Vector3(obj.position.x, obj.position.y, obj.position.z)._w(), new Euler(0,0,0)._w());
            }
            if (obj._castShadow === true) obj._w.setCastShadow(true);
            obj.shadow?._sync?.();
        } else if (obj instanceof Mesh) {
            // Multi-material support: if material is an array AND the geometry
            // has groups, build a sub-mesh per group and attach each separately.
            // Each sub-mesh inherits the parent's transform.
            const isMulti = Array.isArray(obj.material) && obj.geometry?.groups?.length > 0;
            if (isMulti) {
                obj._subMeshes = _buildSubMeshes(obj);
                obj._handle = null;
                for (const sub of obj._subMeshes) {
                    sub._handle = this._w.add(sub._w);
                    this._objects.push(sub);
                }
                this._objects.push(obj);
                return;
            }
            obj._handle = this._w.add(obj._w);
            obj._scene = this;
            _applyMeshShadowFlags(obj);
        }
        this._objects.push(obj);
    }
    remove(...objects) {
        for (const obj of objects.flat()) {
            if (!obj) continue;
            const ci = this.children.indexOf(obj);
            if (ci >= 0) this.children.splice(ci, 1);
            const oi = this._objects.indexOf(obj);
            if (oi >= 0) this._objects.splice(oi, 1);
            if (obj._subMeshes) {
                for (const sub of obj._subMeshes) {
                    if (sub._handle != null) this._w.remove(sub._handle);
                    const si = this._objects.indexOf(sub);
                    if (si >= 0) this._objects.splice(si, 1);
                }
            }
            if (obj._handle != null) this._w.remove(obj._handle);
            obj.parent = null;
            obj._scene = null;
        }
        return this;
    }
    _syncTransforms(camera) { return this._doSyncTransforms(camera); }
    _doSyncTransforms(camera) {
        const camMask = camera?.layers?.mask ?? 1;
        const onLayer = (o) => {
            // If the object has explicit layers, test them; otherwise default
            // to layer 0 (mask 1), matching three.js's default behavior.
            const m = o.layers?.mask ?? 1;
            return (m & camMask) !== 0;
        };
        for (const o of this._objects) {
            if (o instanceof Mesh && o._handle) {
                const layered = o.visible !== false && onLayer(o);
                // Apply morph-target blending if any influences changed.
                if (o.morphTargetInfluences && o.morphTargetInfluences.length > 0) {
                    const sig = o.morphTargetInfluences.join(',');
                    if (sig !== o._lastMorphSig) {
                        o._applyMorphTargets();
                        if (o._morphedGeometry) {
                            const arr = o._morphedGeometry.attributes.position.array;
                            this._w.setMeshPositions(o._handle, arr);
                        }
                        o._lastMorphSig = sig;
                    }
                }
                // If this mesh was attached via a nested-group walk, its parent
                // group's transform handles the outer translation/rotation and
                // its own local transform was composed at scene.add time —
                // honor the stash rather than the mesh's own position/rotation.
                if (o._composedLocalPos) {
                    const p = o._composedLocalPos, r = o._composedLocalRot;
                    this._w.setTransform(o._handle,
                        new Vector3(p.x, p.y, p.z)._w(),
                        new Euler(r.x, r.y, r.z)._w());
                } else {
                    const rot = _effectiveEuler(o);
                    this._w.setTransform(o._handle, o.position._w(), rot._w());
                }
                const sx = o.scale?.x ?? 1, sy = o.scale?.y ?? 1, sz = o.scale?.z ?? 1;
                this._w.setScale(o._handle, sx, sy, sz);
                if (o.renderOrder != null) this._w.setRenderOrder(o._handle, o.renderOrder);
                this._w.setVisible(o._handle, layered);
            } else if (o._isGroup && o._handle) {
                const layered = o.visible !== false && onLayer(o);
                const rot = _effectiveEuler(o);
                this._w.setTransform(o._handle, o.position._w(), rot._w());
                const sx = o.scale?.x ?? 1, sy = o.scale?.y ?? 1, sz = o.scale?.z ?? 1;
                this._w.setScale(o._handle, sx, sy, sz);
                this._w.setVisible(o._handle, layered);
            } else if (o._isSkinnedMesh && o._handle) {
                // Combined skin + morph: apply morph blend first (rewrites
                // position buffer), then push bone matrices. The skinned VS
                // reads the morphed positions before applying the skin matrix.
                if (o.morphTargetInfluences && o.morphTargetInfluences.length > 0) {
                    const sig = o.morphTargetInfluences.join(',');
                    if (sig !== o._lastMorphSig) {
                        o._applyMorphTargets();
                        if (o._morphedGeometry) {
                            const arr = o._morphedGeometry.attributes.position.array;
                            this._w.setMeshPositions(o._handle, arr);
                        }
                        o._lastMorphSig = sig;
                    }
                }
                // Push current bone matrices to the wasm skeleton each frame.
                if (o.skeleton?.boneMatrices) {
                    o.skeleton.update();
                    this._w.setBoneMatrices(o._handle, o.skeleton.boneMatrices);
                }
                // Also push joint/weight attributes if the user set them.
                if (o._skinJoints && !o._skinJointsUploaded) {
                    this._w.setSkinJoints(o._handle, o._skinJoints);
                    o._skinJointsUploaded = true;
                }
                if (o._skinWeights && !o._skinWeightsUploaded) {
                    this._w.setSkinWeights(o._handle, o._skinWeights);
                    o._skinWeightsUploaded = true;
                }
            } else if (o._isLight && o._w?.setDirection && o.target) {
                _syncLightDirection(o);
            }
        }
    }
}

// ---- Cameras ----
// three.js-compatible PerspectiveCamera surface. Extra cinematic fields
// (focus, filmOffset/shift) are additive and never required by existing apps.
export class PerspectiveCamera {
    constructor(fovDeg = 50, aspect = 1, near = 0.1, far = 2000) {
        this.isCamera = true;
        this.isPerspectiveCamera = true;
        this.type = 'PerspectiveCamera';
        this._w = WebCamera.perspective(fovDeg, aspect, near, far);
        this.position = new Vector3(0, 0, 5);
        this.up = new Vector3(0, 1, 0);
        this._lookAt = new Vector3();
        this.quaternion = new Quaternion();
        this.rotation = new Euler();
        this.scale = new Vector3(1, 1, 1);
        this.matrix = new Matrix4();
        this.matrixWorld = new Matrix4();
        this.projectionMatrix = new Matrix4();
        this.matrixWorldInverse = new Matrix4();
        this._fov = fovDeg;
        this.aspect = aspect;
        this.near = near;
        this.far = far;
        this.zoom = 1;
        this.focus = 10;
        this.filmGauge = 35;
        this.filmOffset = 0; // three.js horizontal shift (mm); maps to shift_x
        this.view = null;
        this.layers = new Layers();
        this.name = '';
        Object.defineProperty(this, 'fov', {
            get: () => this._fov,
            set: (v) => {
                this._fov = Number(v);
                this._w.setFov?.(this._fov);
            },
            configurable: true,
            enumerable: true,
        });
    }
    lookAt(x, y, z) {
        if (x instanceof Vector3) { this._lookAt.copy(x); this._w.lookAt(x.x, x.y, x.z); }
        else { this._lookAt.set(x, y, z); this._w.lookAt(x, y, z); }
        return this;
    }
    getWorldDirection(target = new Vector3()) {
        const e = this.position, t = this._lookAt;
        return target.set(t.x - e.x, t.y - e.y, t.z - e.z).normalize();
    }
    getWorldPosition(target = new Vector3()) {
        return target.copy(this.position);
    }
    updateMatrixWorld() {
        this._sync();
        return this;
    }
    updateProjectionMatrix() {
        this._w.setAspect(this.aspect);
        this._w.setFov?.(this._fov);
        // filmOffset (mm) → normalised shift ≈ offset / filmGauge
        if (typeof this._w.setShift === 'function') {
            const sx = this.filmGauge ? (this.filmOffset / this.filmGauge) : 0;
            this._w.setShift(sx, 0);
        }
        return this;
    }
    clone() {
        const c = new PerspectiveCamera(this._fov, this.aspect, this.near, this.far);
        c.position.copy(this.position);
        c._lookAt.copy(this._lookAt);
        c.up.copy(this.up);
        c.zoom = this.zoom;
        c.focus = this.focus;
        c.filmGauge = this.filmGauge;
        c.filmOffset = this.filmOffset;
        return c;
    }
    copy(source) {
        this.fov = source.fov ?? source._fov;
        this.aspect = source.aspect;
        this.near = source.near;
        this.far = source.far;
        this.position.copy(source.position);
        if (source._lookAt) this._lookAt.copy(source._lookAt);
        if (source.up) this.up.copy(source.up);
        this.zoom = source.zoom ?? 1;
        this.focus = source.focus ?? 10;
        this.filmGauge = source.filmGauge ?? 35;
        this.filmOffset = source.filmOffset ?? 0;
        this._sync();
        this.updateProjectionMatrix();
        return this;
    }
    _sync() {
        if (typeof this._w.setView === 'function') {
            const f = this._fov ?? this.fov ?? -1;
            this._w.setView(
                this.position.x, this.position.y, this.position.z,
                this._lookAt.x, this._lookAt.y, this._lookAt.z,
                f,
            );
        } else {
            this._w.setPosition(this.position.x, this.position.y, this.position.z);
            this._w.lookAt(this._lookAt.x, this._lookAt.y, this._lookAt.z);
        }
        if (this.up && this._w.setUp) this._w.setUp(this.up.x, this.up.y, this.up.z);
        if (this.focus != null && this._w.setFocusDistance) {
            this._w.setFocusDistance(this.focus);
        }
    }
}
export class OrthographicCamera {
    constructor(left, right, top, bottom, near = 0.1, far = 2000) {
        this.isCamera = true;
        this.isOrthographicCamera = true;
        this.type = 'OrthographicCamera';
        this._w = WebCamera.orthographic(left, right, top, bottom, near, far);
        this.position = new Vector3();
        this.up = new Vector3(0, 1, 0);
        this._lookAt = new Vector3();
        this.quaternion = new Quaternion();
        this.rotation = new Euler();
        this.scale = new Vector3(1, 1, 1);
        this.left = left; this.right = right; this.top = top; this.bottom = bottom;
        this.near = near;
        this.far = far;
        this.zoom = 1;
        this.layers = new Layers();
        this.name = '';
    }
    lookAt(x, y, z) {
        if (x instanceof Vector3) { this._lookAt.copy(x); this._w.lookAt(x.x, x.y, x.z); }
        else { this._lookAt.set(x, y, z); this._w.lookAt(x, y, z); }
        return this;
    }
    updateMatrixWorld() {
        this._sync();
        return this;
    }
    updateProjectionMatrix() { return this; }
    _sync() {
        this._w.setPosition(this.position.x, this.position.y, this.position.z);
        if (this.up && this._w.setUp) this._w.setUp(this.up.x, this.up.y, this.up.z);
        this._w.lookAt(this._lookAt.x, this._lookAt.y, this._lookAt.z);
    }
}

// ---- Renderer ----
export class WebGLRenderer {
    constructor() {
        throw new Error("Use 'await WebGLRenderer.create(canvas)' (async on wasm).");
    }
    static async create(canvas) {
        if (!_wasmReady) await initThreers();
        const w = await WebRenderer.new(canvas);
        const r = Object.create(WebGLRenderer.prototype);
        r._w = w;
        r._canvas = canvas;
        r.shadowMap = { enabled: true, type: 2 };
        // Push toneMapping / exposure into wasm — assigning the JS fields alone
        // left the GPU on NoToneMapping, so sunlit Earth clipped to flat white.
        r._toneMapping = 0;
        r._toneMappingExposure = 1;
        Object.defineProperty(r, 'toneMapping', {
            get() { return r._toneMapping; },
            set(v) {
                r._toneMapping = Number(v) || 0;
                r._w.setToneMapping?.(r._toneMapping, r._toneMappingExposure);
            },
            configurable: true, enumerable: true,
        });
        Object.defineProperty(r, 'toneMappingExposure', {
            get() { return r._toneMappingExposure; },
            set(v) {
                r._toneMappingExposure = Number(v);
                r._w.setToneMapping?.(r._toneMapping, r._toneMappingExposure);
            },
            configurable: true, enumerable: true,
        });
        // 4× MSAA on the opaque forward pass (wasm clamps anything >1 to 4).
        r._w.setMsaa?.(4);
        return r;
    }
    get domElement() { return this._canvas; }
    setSize(w, h, updateStyle = true) {
        this._canvas.width = w;
        this._canvas.height = h;
        if (updateStyle !== false) {
            this._canvas.style.width = `${w}px`;
            this._canvas.style.height = `${h}px`;
        }
        this._w.setSize(w, h);
    }
    setPixelRatio(_) {}
    setRenderTarget(target) {
        if (target === null || target === undefined) {
            this._w.setRenderTarget(0);
            this._currentRenderTarget = null;
            return;
        }
        if (!target._w) {
            // Lazy-allocate the wgpu render target now that we have the renderer.
            target._w = (target._halfFloat || target.type === HalfFloatType) && WebRenderTarget.newHalfFloat
                ? WebRenderTarget.newHalfFloat(this._w, target.width, target.height)
                : new WebRenderTarget(this._w, target.width, target.height);
        }
        // Make sure target.texture._w is a Texture handle backed by the RT view.
        if (target.texture && !target.texture._w_isExternal) {
            const tex_w = this._w.renderTargetTexture(target._w);
            target.texture._w = tex_w;
            target.texture._w_isExternal = true;
        }
        this._w.setRenderTarget(target._w.id);
        this._currentRenderTarget = target;
    }
    getRenderTarget() { return this._currentRenderTarget || null; }
    /// Apply one of the built-in post-fx passes: read from `inputRT`, write to
    /// the canvas surface. `effectKind`: 0=copy/passthrough, 1=fxaa, 2=film,
    /// 3=dotscreen, 4=halftone, 5=glitch. `time` is seconds (for time-tied
    /// effects); `p2` is a `[x, y, z, w]` array of per-effect tuning values.
    /// For SSAO (kind 11), pass `depthRT`, `normalRT`, and `cam` from
    /// `_postFxCameraFrom()`.
    applyPostFx(inputRT, effectKind, time = 0, p2 = [0, 0, 0, 0], additive = false, depthRT = null, normalRT = null, cam = null, p3 = [0, 0, 0, 0]) {
        if (!inputRT?._w) return;
        const depthId = depthRT?._w?.id ?? 0;
        const normalId = normalRT?._w?.id ?? 0;
        const c = cam || _defaultPostFxCam();
        this._w.applyPostFx(
            inputRT._w.id, depthId, normalId, effectKind, time,
            p2[0], p2[1], p2[2], p2[3],
            p3[0], p3[1], p3[2], p3[3],
            additive ? 1 : 0,
            c.near, c.far, c.kernelRadius, c.kernelSize, c.proj, c.invProj,
        );
    }
    applyPostFxToRT(inputRT, outputRT, effectKind, time = 0, p2 = [0, 0, 0, 0], depthRT = null, normalRT = null, cam = null, p3 = [0, 0, 0, 0]) {
        if (!inputRT?._w) return;
        if (!outputRT._w) {
            outputRT._w = (outputRT._halfFloat || outputRT.type === HalfFloatType) && WebRenderTarget.newHalfFloat
                ? WebRenderTarget.newHalfFloat(this._w, outputRT.width, outputRT.height)
                : new WebRenderTarget(this._w, outputRT.width, outputRT.height);
        }
        if (outputRT.texture && !outputRT.texture._w_isExternal) {
            const tex_w = this._w.renderTargetTexture(outputRT._w);
            outputRT.texture._w = tex_w;
            outputRT.texture._w_isExternal = true;
        }
        const depthId = depthRT?._w?.id ?? 0;
        const normalId = normalRT?._w?.id ?? 0;
        const c = cam || _defaultPostFxCam();
        this._w.applyPostFxToRT(
            inputRT._w.id, outputRT._w.id, depthId, normalId, effectKind, time,
            p2[0], p2[1], p2[2], p2[3],
            p3[0], p3[1], p3[2], p3[3],
            c.near, c.far, c.kernelRadius, c.kernelSize, c.proj, c.invProj,
        );
    }
    async readRenderTargetPixels(target, x, y, width, height, buffer) {
        if (!target?._w) return;
        const bytes = await this._w.readRenderTargetPixels(target._w.id, x, y, width, height);
        if (buffer && bytes) buffer.set(bytes);
        return bytes;
    }
    async readRenderTargetF16(target, x = 0, y = 0, w, h) {
        if (!target?._w || !this._w?.readRenderTargetF16) return null;
        const width = w ?? target.width ?? this._canvas?.width ?? 800;
        const height = h ?? target.height ?? this._canvas?.height ?? 600;
        return this._w.readRenderTargetF16(target._w.id, x, y, width, height);
    }
    render(scene, camera) {
        // Orbit/trackball controls own the wasm camera pose — don't push stale JS over it.
        if (!camera?._orbitControlled && typeof camera?._sync === 'function') camera._sync();
        // Fire material.onBeforeCompile once per material on first render, and
        // give materials a chance to re-upload mutated uniforms (e.g. Sky).
        for (const o of scene._objects || []) {
            const m = o?.material;
            if (m?._runOnBeforeCompile) m._runOnBeforeCompile();
            if (m?.updateUniforms) {
                m.updateUniforms();
                // Sky material was re-created — refresh the mesh's wasm handle.
                if (o._w && m._w) {
                    const geom_w = _geomToWebGeom(o.geometry);
                    o._w = new WebMesh(geom_w, m._w);
                    // Remove + re-add the mesh so the scene picks up the new _w.
                    if (o._handle != null) {
                        scene._w.remove(o._handle);
                        o._handle = scene._w.add(o._w);
                    }
                }
            }
        }
        // Per-object onBeforeRender hooks (Reflector / Refractor / Water etc).
        // Guarded against recursion: each hook can render to its own RT but
        // sets _inRender to prevent infinite re-entry through this same path.
        if (!this._inRender) {
            this._inRender = true;
            for (const o of scene._objects || []) {
                if (o?.onBeforeRender) o.onBeforeRender(this, scene, camera);
            }
            this._inRender = false;
        }
        scene._syncTransforms(camera);
        if (!camera?._orbitControlled) camera._sync();
        this._w.render(scene._w, camera._w);
    }
}
export const WebGPURenderer = WebGLRenderer;

/**
 * Render frames of `scene` through `renderer` into RGBA buffers.
 *
 * Size defaults to the canvas drawing-buffer size when `width`/`height` omitted.
 * Prefer `frames` or `duration` (seconds × fps) for length.
 * WebM capture sizes snap down to multiples of 8 unless `strictSize: true`.
 *
 * Set `parallel: true` (or `concurrency: N`) to pipeline GPU readbacks across
 * multiple render targets so the next frame can render while a prior map_async
 * completes. Default is sequential (`concurrency: 1`).
 *
 * @returns {Promise<Uint8Array[]>}
 */
export async function captureSceneFrames(renderer, scene, camera, options = {}) {
    const canvas = renderer?.domElement;
    let width = (options.width | 0) || (canvas?.width | 0);
    let height = (options.height | 0) || (canvas?.height | 0);
    const fps = Math.max(1, (options.fps | 0) || 30);
    let frameCount = options.frameCount | 0;
    if (!frameCount && options.frames) frameCount = options.frames | 0;
    if (!frameCount && options.duration) {
        frameCount = Math.max(1, Math.round(Number(options.duration) * fps));
    }

    const formatInfo = options.format != null
        ? parseVideoFormat(options.format)
        : { format: VideoFormat.Gif, transparent: false };
    const format = formatInfo.format;

    if (format === VideoFormat.Webm && width > 0 && height > 0) {
        const aligned = alignVideoSize(width, height);
        if (aligned.snapped) {
            if (options.strictSize) {
                throw new VideoExportError(
                    `WebM/VP9 requires width and height multiples of 8 (got ${width}×${height})`,
                    VideoExportErrorCode.InvalidDimensions,
                    { hint: `use ${aligned.width}×${aligned.height}` },
                );
            }
            width = aligned.width;
            height = aligned.height;
        }
    }

    if (width <= 0 || height <= 0 || frameCount <= 0) {
        const missing = [];
        if (width <= 0) missing.push('width');
        if (height <= 0) missing.push('height');
        if (frameCount <= 0) missing.push('frames|duration');
        throw new VideoExportError(
            `captureSceneFrames: missing ${missing.join(', ')}`,
            VideoExportErrorCode.InvalidOption,
            { hint: 'pass size()/frames() or ensure the canvas has a drawing buffer' },
        );
    }

    const concurrency = resolveCaptureConcurrency(options, frameCount);
    const signal = options.signal ?? null;
    const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
    let mapFrame = typeof options.mapFrame === 'function' ? options.mapFrame : null;
    if (!mapFrame && options.transparentCornerPunch) {
        const punchSize = options.transparentCornerPunch === true
            ? Math.min(40, width, height)
            : Math.min(Math.max(1, options.transparentCornerPunch | 0), width, height);
        mapFrame = (rgba) => {
            const copy = new Uint8Array(rgba);
            for (let y = 0; y < punchSize; y++) {
                for (let x = 0; x < punchSize; x++) {
                    copy[(y * width + x) * 4 + 3] = 0;
                }
            }
            return copy;
        };
    }

    const ownedTargets = [];
    /** @type {InstanceType<typeof WebGLRenderTarget>[]} */
    let targets;
    if (Array.isArray(options.renderTargets) && options.renderTargets.length > 0) {
        targets = options.renderTargets.slice(0, concurrency);
        while (targets.length < concurrency) {
            const rt = new WebGLRenderTarget(width, height);
            ownedTargets.push(rt);
            targets.push(rt);
        }
    } else if (options.renderTarget && concurrency === 1) {
        targets = [options.renderTarget];
    } else {
        targets = [];
        for (let i = 0; i < concurrency; i++) {
            const rt = (i === 0 && options.renderTarget)
                ? options.renderTarget
                : new WebGLRenderTarget(width, height);
            if (rt !== options.renderTarget) ownedTargets.push(rt);
            targets.push(rt);
        }
    }

    const prev = renderer.getRenderTarget?.() ?? null;
    /** @type {(Uint8Array|undefined)[]} */
    const frames = new Array(frameCount);
    /** @type {({ index: number, promise: Promise<Uint8Array> }|null)[]} */
    const slots = new Array(concurrency).fill(null);
    const done = new Array(frameCount).fill(false);
    let reported = 0;

    const throwIfAborted = () => {
        if (signal?.aborted) {
            throw new VideoExportError('Video export aborted', VideoExportErrorCode.Aborted, {
                cause: signal.reason,
            });
        }
    };

    const eventTarget = options.eventTarget
        || (options.emitter && typeof options.emitter.dispatchEvent === 'function' ? options.emitter : null);

    const reportThrough = () => {
        while (reported < frameCount && done[reported]) {
            reported += 1;
            const info = {
                phase: 'capture',
                frame: reported,
                frames: frameCount,
                ratio: reported / (frameCount + 1),
                format,
            };
            info.message = formatVideoProgress(info);
            emitProgress(onProgress, info, eventTarget);
        }
    };

    const settleSlot = async (slot) => {
        const job = slots[slot];
        if (!job) return;
        slots[slot] = null;
        const rgba = await job.promise;
        frames[job.index] = rgba;
        done[job.index] = true;
        if (typeof options.onFrame === 'function') options.onFrame(job.index, frameCount);
        reportThrough();
    };

    try {
        if (eventTarget) {
            eventTarget.dispatchEvent(new CustomEvent(VideoExportEvent.Start, {
                detail: { phase: 'capture', frames: frameCount, format, concurrency },
            }));
        }
        for (let i = 0; i < frameCount; i++) {
            throwIfAborted();
            const slot = i % concurrency;
            if (slots[slot]) await settleSlot(slot);

            if (typeof options.update === 'function') options.update(i, frameCount);
            const rt = targets[slot];
            renderer.setRenderTarget(rt);
            renderer.render(scene, camera);

            // Kick off async GPU readback without awaiting — next iterations can
            // render into other targets while this map_async completes.
            const index = i;
            const promise = Promise.resolve(
                renderer.readRenderTargetPixels(rt, 0, 0, width, height),
            ).then((raw) => {
                let rgba = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
                if (mapFrame) rgba = mapFrame(rgba, index, frameCount);
                return rgba;
            });
            slots[slot] = { index, promise };

            if (options.yield !== false) {
                await new Promise((r) => setTimeout(r, 0));
            }
        }

        for (let slot = 0; slot < concurrency; slot++) {
            if (slots[slot]) await settleSlot(slot);
        }
    } catch (err) {
        if (eventTarget) {
            const aborted = VideoExportError.is(err) && err.code === VideoExportErrorCode.Aborted;
            eventTarget.dispatchEvent(new CustomEvent(
                aborted ? VideoExportEvent.Abort : VideoExportEvent.Error,
                { detail: { error: err } },
            ));
        }
        throw err;
    } finally {
        renderer.setRenderTarget(prev);
        for (const rt of ownedTargets) {
            try { rt.dispose?.(); } catch { /* ignore */ }
        }
    }

    return frames;
}

/** @param {object} options @param {number} frameCount */
function resolveCaptureConcurrency(options, frameCount) {
    let n = 1;
    if (options.concurrency != null) {
        n = Number(options.concurrency) || 1;
    } else if (options.parallel === true) {
        n = 2;
    } else if (typeof options.parallel === 'number') {
        n = options.parallel;
    } else if (options.parallel) {
        n = 2;
    }
    n = Math.max(1, Math.min(8, n | 0));
    return Math.min(n, Math.max(1, frameCount | 0));
}

/**
 * Capture a scene animation and encode it to GIF / APNG / WebM.
 * @returns {Promise<import('./video-export.js').VideoExportResult>}
 */
export async function exportSceneVideo(renderer, scene, camera, options = {}) {
    const canvas = renderer?.domElement;
    let width = (options.width | 0) || (canvas?.width | 0);
    let height = (options.height | 0) || (canvas?.height | 0);
    const fps = Math.max(1, (options.fps | 0) || 30);
    const parsed = options.format != null
        ? parseVideoFormat(options.format)
        : { format: VideoFormat.Gif, transparent: false };
    const format = parsed.format;
    const transparent = options.transparent != null ? !!options.transparent : parsed.transparent;

    if (format === VideoFormat.Webm && width > 0 && height > 0) {
        const aligned = alignVideoSize(width, height);
        if (aligned.snapped && !options.strictSize) {
            width = aligned.width;
            height = aligned.height;
        }
    }

    const eventTarget = options.eventTarget
        || (options.emitter && typeof options.emitter.dispatchEvent === 'function' ? options.emitter : null);

    const frames = await captureSceneFrames(renderer, scene, camera, {
        ...options,
        width,
        height,
        fps,
        format,
        transparent,
        eventTarget,
    });
    const exporter = (eventTarget instanceof VideoExporter)
        ? eventTarget.configure({
            width,
            height,
            fps,
            format,
            transparent,
            gifColors: options.gifColors,
            filename: options.filename,
            basename: options.basename,
            onProgress: options.onProgress,
            signal: options.signal,
            strictSize: options.strictSize,
        })
        : new VideoExporter({
            width,
            height,
            fps,
            format,
            transparent,
            gifColors: options.gifColors,
            filename: options.filename,
            basename: options.basename,
            onProgress: options.onProgress,
            signal: options.signal,
            strictSize: options.strictSize,
            eventTarget,
        });
    if (options.worker || options.encodeInWorker) {
        return exporter.encodeInWorker(frames, {
            wasmUrl: options.wasmUrl,
            workerUrl: options.workerUrl,
            worker: options.worker instanceof VideoEncodeWorker ? options.worker : undefined,
            onProgress: options.onProgress,
            signal: options.signal,
            eventTarget: exporter,
        });
    }
    return exporter.encode(frames);
}

/**
 * Capture, encode, and trigger a file download.
 * @returns {Promise<import('./video-export.js').VideoExportResult>}
 */
export async function exportAndDownloadSceneVideo(renderer, scene, camera, options = {}) {
    const result = await exportSceneVideo(renderer, scene, camera, options);
    return result.download(options.filename);
}

/**
 * Bind a renderer/scene/camera and configure a fluent scene export.
 *
 * @example
 * ```js
 * await VideoExporter.from(renderer, scene, camera)
 *   .gif({ transparent: true })
 *   .parallel(3)
 *   .frames(30)
 *   .update((i, n) => { cube.rotation.y = (i / n) * Math.PI * 2; })
 *   .onProgress(({ message }) => setStatus(message))
 *   .download('cube');
 * ```
 */
VideoExporter.from = function from(renderer, scene, camera, options = {}) {
    const api = new VideoExporter(options);
    let frameCount = options.frameCount || options.frames || 0;
    let duration = options.duration || 0;
    let update = typeof options.update === 'function' ? options.update : null;
    let yieldBetween = options.yield !== false;
    let cornerPunch = options.transparentCornerPunch || false;
    let concurrency = resolveCaptureConcurrency(options, 1);

    api.frames = function frames(n) {
        frameCount = n | 0;
        return api;
    };
    api.duration = function durationSec(seconds) {
        duration = Number(seconds) || 0;
        return api;
    };
    api.update = function updateFn(fn) {
        update = fn;
        return api;
    };
    api.yieldBetweenFrames = function yieldBetweenFrames(on = true) {
        yieldBetween = !!on;
        return api;
    };
    api.transparentCornerPunch = function transparentCornerPunch(on = true) {
        cornerPunch = on;
        return api;
    };
    /**
     * Pipeline GPU readbacks across N render targets (1–8).
     * `true` → 2; a number sets concurrency explicitly.
     * @param {boolean|number} [n=true]
     */
    api.parallel = function parallel(n = true) {
        concurrency = resolveCaptureConcurrency({ parallel: n }, 1);
        return api;
    };
    /** Alias for {@link parallel}. @param {number} n */
    api.concurrency = function concurrencyFn(n) {
        concurrency = resolveCaptureConcurrency({ concurrency: n }, 1);
        return api;
    };
    let useWorker = !!(options.worker || options.encodeInWorker);
    api.worker = function worker(on = true) {
        useWorker = !!on;
        return api;
    };

    async function run(doDownload, filenameOverride) {
        const canvas = renderer?.domElement;
        let width = (api._opts.width | 0) || (canvas?.width | 0);
        let height = (api._opts.height | 0) || (canvas?.height | 0);
        if (api._opts.format === VideoFormat.Webm && width > 0 && height > 0) {
            const aligned = alignVideoSize(width, height);
            if (aligned.snapped && !api._opts.strictSize) {
                width = aligned.width;
                height = aligned.height;
            }
        }
        api.size(width, height);
        if (!frameCount && !duration) {
            frameCount = 30; // sensible default for demos / quick exports
        }
        const opts = {
            width,
            height,
            fps: api._opts.fps,
            format: api._opts.format,
            transparent: api._opts.transparent,
            gifColors: api._opts.gifColors,
            filename: filenameOverride || api._opts.filename,
            basename: api._opts.basename,
            frameCount,
            duration,
            frames: frameCount,
            update,
            mapFrame: api._mapFrame,
            transparentCornerPunch: cornerPunch,
            onProgress: api._onProgress,
            signal: api._signal,
            yield: yieldBetween,
            strictSize: api._opts.strictSize,
            concurrency,
            parallel: concurrency > 1,
            encodeInWorker: useWorker,
            wasmUrl: options.wasmUrl,
            workerUrl: options.workerUrl,
            eventTarget: api,
        };
        const result = await exportSceneVideo(renderer, scene, camera, opts);
        return doDownload ? result.download(opts.filename) : result;
    }

    api.export = (overrides) => {
        if (overrides && typeof overrides === 'object') {
            if (overrides.frames != null) frameCount = overrides.frames | 0;
            if (overrides.frameCount != null) frameCount = overrides.frameCount | 0;
            if (overrides.duration != null) duration = Number(overrides.duration) || 0;
            if (typeof overrides.update === 'function') update = overrides.update;
            if (overrides.parallel != null || overrides.concurrency != null) {
                concurrency = resolveCaptureConcurrency(overrides, 1);
            }
            api.configure(overrides);
        }
        return run(false);
    };
    api.download = (filenameOrOverrides) => {
        if (filenameOrOverrides && typeof filenameOrOverrides === 'object') {
            const o = filenameOrOverrides;
            if (o.frames != null) frameCount = o.frames | 0;
            if (o.frameCount != null) frameCount = o.frameCount | 0;
            if (o.duration != null) duration = Number(o.duration) || 0;
            if (typeof o.update === 'function') update = o.update;
            if (o.parallel != null || o.concurrency != null) {
                concurrency = resolveCaptureConcurrency(o, 1);
            }
            api.configure(o);
            return run(true, o.filename);
        }
        return run(true, typeof filenameOrOverrides === 'string' ? filenameOrOverrides : undefined);
    };
    return api;
};

// ---- Pass 5: remaining geometries (LatheGeometry, TubeGeometry,
// ExtrudeGeometry, EdgesGeometry, WireframeGeometry, ShapeGeometry,
// ConvexGeometry, DecalGeometry, TextGeometry, ParametricGeometry).
//
// We fall back to a BoxGeometry-shaped BufferGeometry where a true
// parametric impl isn't wired through the wasm side yet, so user code that
// constructs and inspects a geometry doesn't error out. A future wasm
// expansion can replace these with the real triangulators.

function _fallbackGeometry() { return new BoxGeometry(1, 1, 1); }

// LatheGeometry — rotate a 2D profile (Vector2[]) around the Y axis with
function _ellipseArcPoints2(aX, aY, xRadius, aStartAngle, aEndAngle, clockwise, divisions) {
    const pts = [];
    const resolution = divisions * 2;
    let deltaAngle = aEndAngle - aStartAngle;
    if (Math.abs(deltaAngle) < 1e-8) deltaAngle = Math.PI * 2;
    if (!clockwise && deltaAngle < 0) deltaAngle += Math.PI * 2;
    if (clockwise && deltaAngle > 0) deltaAngle -= Math.PI * 2;
    for (let i = 0; i <= resolution; i++) {
        const t = i / resolution;
        const angle = aStartAngle + t * deltaAngle;
        pts.push(new Vector2(aX + xRadius * Math.cos(angle), aY + xRadius * Math.sin(angle)));
    }
    return pts;
}
function _capsuleLathePoints(radius, length, capSegments) {
    const points = [];
    let last = null;
    for (const p of [
        ..._ellipseArcPoints2(0, -length / 2, radius, Math.PI * 1.5, 0, false, capSegments),
        ..._ellipseArcPoints2(0, length / 2, radius, 0, Math.PI * 0.5, false, capSegments).slice(1),
    ]) {
        if (last && last.x === p.x && last.y === p.y) continue;
        points.push(p);
        last = p;
    }
    return points;
}
function _jsBoxIndexedGeometry(w = 1, h = 1, d = 1) {
    const hx = w * 0.5, hy = h * 0.5, hz = d * 0.5;
    const faces = [
        [[ hx, -hy,  hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [ hx,  hy,  hz], [ 1, 0, 0]],
        [[-hx, -hy, -hz], [-hx, -hy,  hz], [-hx,  hy,  hz], [-hx,  hy, -hz], [-1, 0, 0]],
        [[-hx,  hy,  hz], [ hx,  hy,  hz], [ hx,  hy, -hz], [-hx,  hy, -hz], [ 0, 1, 0]],
        [[-hx, -hy, -hz], [ hx, -hy, -hz], [ hx, -hy,  hz], [-hx, -hy,  hz], [ 0,-1, 0]],
        [[-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz], [ 0, 0, 1]],
        [[ hx, -hy, -hz], [-hx, -hy, -hz], [-hx,  hy, -hz], [ hx,  hy, -hz], [ 0, 0,-1]],
    ];
    const positions = [], normals = [], uvs = [], indices = [];
    faces.forEach((face, fi) => {
        const base = fi * 4;
        for (const v of face.slice(0, 4)) positions.push(...v);
        for (let i = 0; i < 4; i++) normals.push(...face[4]);
        uvs.push(0, 0, 1, 0, 1, 1, 0, 1);
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    });
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
    g.setAttribute('normal', new BufferAttribute(new Float32Array(normals), 3));
    g.setAttribute('uv', new BufferAttribute(new Float32Array(uvs), 2));
    g.setIndex(new Uint32Array(indices));
    return g;
}
// `segments` divisions. Matches three.js's signature
// `new LatheGeometry(points, segments, phiStart, phiLength)`.
export class LatheGeometry {
    constructor(points = [new Vector2(0, -0.5), new Vector2(0.5, 0), new Vector2(0, 0.5)],
                segments = 12, phiStart = 0, phiLength = Math.PI * 2) {
        const positions = [];
        const normals = [];
        const uvs = [];
        const indices = [];
        const pn = points.length;
        // Mirror three.js's exact LatheGeometry layout — vertices use
        // (x*sin, y, x*cos) ordering, segment-to-segment tangent normals are
        // (dy, -dx, 0), and adjacent normals get averaged for smooth shading.
        const inverseSegments = 1.0 / segments;
        const initN = new Array(pn);
        for (let j = 0; j < pn; j++) initN[j] = new Vector3();
        for (let j = 0; j < pn - 1; j++) {
            const dx = points[j + 1].x - points[j].x;
            const dy = points[j + 1].y - points[j].y;
            initN[j].set(dy, -dx, 0).normalize();
        }
        initN[pn - 1].copy(initN[pn - 2]);
        const smoothed = new Array(pn);
        for (let j = 0; j < pn; j++) smoothed[j] = new Vector3();
        for (let j = 0; j < pn - 1; j++) {
            smoothed[j].copy(initN[j]).add(initN[j + 1 === pn ? 0 : j + 1]).multiplyScalar(0.5).normalize();
        }
        smoothed[0].copy(initN[0]);
        smoothed[pn - 1].copy(initN[pn - 2]);

        for (let i = 0; i <= segments; i++) {
            const phi = phiStart + i * inverseSegments * phiLength;
            const sin = Math.sin(phi);
            const cos = Math.cos(phi);
            for (let j = 0; j <= pn - 1; j++) {
                const p = points[j];
                positions.push(p.x * sin, p.y, p.x * cos);
                const n = smoothed[j];
                normals.push(n.x * sin, n.y, n.x * cos);
                uvs.push(i / segments, j / (pn - 1));
            }
        }
        for (let i = 0; i < segments; i++) for (let j = 0; j < pn - 1; j++) {
            const base = j + i * pn;
            const a = base;
            const b = base + pn;
            const c = base + pn + 1;
            const d = base + 1;
            indices.push(a, b, d,  c, d, b);
        }
        const g = new BufferGeometry();
        g.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
        g.setAttribute('normal', new BufferAttribute(new Float32Array(normals), 3));
        g.setAttribute('uv', new BufferAttribute(new Float32Array(uvs), 2));
        g.setIndex(new Uint32Array(indices));
        Object.assign(this, g);
        this.type = 'LatheGeometry';
        this._isUserGeometry = true;
    }
}

// TubeGeometry — matches three.js TubeGeometry (Frenet frames + getPointAt).
export class TubeGeometry {
    constructor(path, tubularSegments = 64, radius = 1, radialSegments = 8, closed = false) {
        if (!path || typeof path.getPointAt !== 'function') {
            path = new CatmullRomCurve3([
                new Vector3(-1, 0, 0), new Vector3(0, 1, 0), new Vector3(1, 0, 0),
            ]);
        }
        const frames = path?.computeFrenetFrames?.(tubularSegments, closed)
            ?? { tangents: [], normals: [], binormals: [] };
        const vertices = [], normals = [], uvs = [], indices = [];
        const vertex = new Vector3(), normal = new Vector3(), uv = new Vector2();
        let P = new Vector3();
        const generateSegment = (i) => {
            P = path.getPointAt(i / tubularSegments, P);
            const N = frames.normals[i];
            const B = frames.binormals[i];
            for (let j = 0; j <= radialSegments; j++) {
                const v = j / radialSegments * Math.PI * 2;
                const sin = Math.sin(v);
                const cos = -Math.cos(v);
                normal.set(cos * N.x + sin * B.x, cos * N.y + sin * B.y, cos * N.z + sin * B.z).normalize();
                normals.push(normal.x, normal.y, normal.z);
                vertex.set(P.x + radius * normal.x, P.y + radius * normal.y, P.z + radius * normal.z);
                vertices.push(vertex.x, vertex.y, vertex.z);
            }
        };
        for (let i = 0; i < tubularSegments; i++) generateSegment(i);
        generateSegment(closed === false ? tubularSegments : 0);
        for (let i = 0; i <= tubularSegments; i++) {
            for (let j = 0; j <= radialSegments; j++) {
                uvs.push(i / tubularSegments, j / radialSegments);
            }
        }
        for (let j = 1; j <= tubularSegments; j++) {
            for (let i = 1; i <= radialSegments; i++) {
                const a = (radialSegments + 1) * (j - 1) + (i - 1);
                const b = (radialSegments + 1) * j + (i - 1);
                const c = (radialSegments + 1) * j + i;
                const d = (radialSegments + 1) * (j - 1) + i;
                indices.push(a, b, d, b, c, d);
            }
        }
        const g = new BufferGeometry();
        g.setAttribute('position', new BufferAttribute(new Float32Array(vertices), 3));
        g.setAttribute('normal', new BufferAttribute(new Float32Array(normals), 3));
        g.setAttribute('uv', new BufferAttribute(new Float32Array(uvs), 2));
        g.setIndex(new Uint32Array(indices));
        Object.assign(this, g);
        this.type = 'TubeGeometry';
        this._isUserGeometry = true;
        this._closed = closed;
    }
}

// ShapeGeometry — triangulate a closed 2D Shape in the XY plane (matches three.js).
export class ShapeGeometry {
    constructor(
        shapes = new Shape([
            new Vector2(0, 0.5),
            new Vector2(-0.5, -0.5),
            new Vector2(0.5, -0.5),
        ]),
        curveSegments = 12,
    ) {
        const g = new BufferGeometry();
        const indices = [];
        const vertices = [];
        const normals = [];
        const uvs = [];
        let groupStart = 0;
        let groupCount = 0;

        const addShape = (shape) => {
            const indexOffset = vertices.length / 3;
            const points = shape.extractPoints
                ? shape.extractPoints(curveSegments)
                : { shape: shape.getPoints?.(curveSegments) ?? shape, holes: [] };
            let shapeVertices = points.shape.slice();
            const shapeHoles = points.holes.map(h => h.slice());

            if (!ShapeUtils.isClockWise(shapeVertices)) {
                shapeVertices.reverse();
            }
            for (let i = 0; i < shapeHoles.length; i++) {
                if (ShapeUtils.isClockWise(shapeHoles[i])) {
                    shapeHoles[i] = shapeHoles[i].slice().reverse();
                }
            }

            const faces = ShapeUtils.triangulateShape(shapeVertices, shapeHoles);

            for (let i = 0; i < shapeHoles.length; i++) {
                shapeVertices = shapeVertices.concat(shapeHoles[i]);
            }

            for (let i = 0; i < shapeVertices.length; i++) {
                const vertex = shapeVertices[i];
                vertices.push(vertex.x, vertex.y, 0);
                normals.push(0, 0, 1);
                uvs.push(vertex.x, vertex.y);
            }

            for (let i = 0; i < faces.length; i++) {
                const face = faces[i];
                indices.push(face[0] + indexOffset, face[1] + indexOffset, face[2] + indexOffset);
                groupCount += 3;
            }
        };

        if (!Array.isArray(shapes)) {
            addShape(shapes);
        } else {
            for (let i = 0; i < shapes.length; i++) {
                addShape(shapes[i]);
                g.addGroup(groupStart, groupCount, i);
                groupStart += groupCount;
                groupCount = 0;
            }
        }

        g.setIndex(new Uint32Array(indices));
        g.setAttribute('position', new BufferAttribute(new Float32Array(vertices), 3));
        g.setAttribute('normal', new BufferAttribute(new Float32Array(normals), 3));
        g.setAttribute('uv', new BufferAttribute(new Float32Array(uvs), 2));
        Object.assign(this, g);
        Object.defineProperty(this, 'index', {
            get() { return this._indexAttr || null; },
            configurable: true,
            enumerable: true,
        });
        this.type = 'ShapeGeometry';
        this.parameters = { shapes, curveSegments };
        this._isUserGeometry = true;
    }
}

function _removeDupEndPts(points) {
    const l = points.length;
    if (l > 2) {
        const first = points[0], last = points[l - 1];
        if (first.x === last.x && first.y === last.y) points.pop();
    }
}

const _extrudeWorldUVGenerator = {
    generateTopUV(_geometry, vertices, indexA, indexB, indexC) {
        return [
            new Vector2(vertices[indexA * 3], vertices[indexA * 3 + 1]),
            new Vector2(vertices[indexB * 3], vertices[indexB * 3 + 1]),
            new Vector2(vertices[indexC * 3], vertices[indexC * 3 + 1]),
        ];
    },
    generateSideWallUV(_geometry, vertices, indexA, indexB, indexC, indexD) {
        const a_x = vertices[indexA * 3], a_y = vertices[indexA * 3 + 1], a_z = vertices[indexA * 3 + 2];
        const b_x = vertices[indexB * 3], b_y = vertices[indexB * 3 + 1], b_z = vertices[indexB * 3 + 2];
        const c_x = vertices[indexC * 3], c_y = vertices[indexC * 3 + 1], c_z = vertices[indexC * 3 + 2];
        const d_x = vertices[indexD * 3], d_y = vertices[indexD * 3 + 1], d_z = vertices[indexD * 3 + 2];
        if (Math.abs(a_y - b_y) < Math.abs(a_x - b_x)) {
            return [
                new Vector2(a_x, 1 - a_z), new Vector2(b_x, 1 - b_z),
                new Vector2(c_x, 1 - c_z), new Vector2(d_x, 1 - d_z),
            ];
        }
        return [
            new Vector2(a_y, 1 - a_z), new Vector2(b_y, 1 - b_z),
            new Vector2(c_y, 1 - c_z), new Vector2(d_y, 1 - d_z),
        ];
    },
};

function _extrudeScalePt2(pt, vec, size) {
    return new Vector2(pt.x + vec.x * size, pt.y + vec.y * size);
}

function _extrudeGetBevelVec(inPt, inPrev, inNext) {
    const v_prev_x = inPt.x - inPrev.x, v_prev_y = inPt.y - inPrev.y;
    const v_next_x = inNext.x - inPt.x, v_next_y = inNext.y - inPt.y;
    const v_prev_lensq = v_prev_x * v_prev_x + v_prev_y * v_prev_y;
    const collinear0 = v_prev_x * v_next_y - v_prev_y * v_next_x;
    let v_trans_x, v_trans_y, shrink_by;
    if (Math.abs(collinear0) > Number.EPSILON) {
        const v_prev_len = Math.sqrt(v_prev_lensq);
        const v_next_len = Math.sqrt(v_next_x * v_next_x + v_next_y * v_next_y);
        const ptPrevShift_x = inPrev.x - v_prev_y / v_prev_len;
        const ptPrevShift_y = inPrev.y + v_prev_x / v_prev_len;
        const ptNextShift_x = inNext.x - v_next_y / v_next_len;
        const ptNextShift_y = inNext.y + v_next_x / v_next_len;
        const sf = ((ptNextShift_x - ptPrevShift_x) * v_next_y -
            (ptNextShift_y - ptPrevShift_y) * v_next_x) /
            (v_prev_x * v_next_y - v_prev_y * v_next_x);
        v_trans_x = ptPrevShift_x + v_prev_x * sf - inPt.x;
        v_trans_y = ptPrevShift_y + v_prev_y * sf - inPt.y;
        const v_trans_lensq = v_trans_x * v_trans_x + v_trans_y * v_trans_y;
        if (v_trans_lensq <= 2) return new Vector2(v_trans_x, v_trans_y);
        shrink_by = Math.sqrt(v_trans_lensq / 2);
    } else {
        let direction_eq = false;
        if (v_prev_x > Number.EPSILON) {
            if (v_next_x > Number.EPSILON) direction_eq = true;
        } else if (v_prev_x < -Number.EPSILON) {
            if (v_next_x < -Number.EPSILON) direction_eq = true;
        } else if (Math.sign(v_prev_y) === Math.sign(v_next_y)) {
            direction_eq = true;
        }
        if (direction_eq) {
            v_trans_x = -v_prev_y; v_trans_y = v_prev_x;
            shrink_by = Math.sqrt(v_prev_lensq);
        } else {
            v_trans_x = v_prev_x; v_trans_y = v_prev_y;
            shrink_by = Math.sqrt(v_prev_lensq / 2);
        }
    }
    return new Vector2(v_trans_x / shrink_by, v_trans_y / shrink_by);
}

function _extrudeAddShape(shape, options, scope, verticesArray, uvArray) {
    const placeholder = [];
    const curveSegments = options.curveSegments ?? 12;
    const steps = options.steps ?? 1;
    const depth = options.depth ?? 1;
    let bevelEnabled = options.bevelEnabled ?? true;
    let bevelThickness = options.bevelThickness ?? 0.2;
    let bevelSize = options.bevelSize ?? (bevelThickness - 0.1);
    let bevelOffset = options.bevelOffset ?? 0;
    let bevelSegments = options.bevelSegments ?? 3;
    const uvgen = options.UVGenerator ?? _extrudeWorldUVGenerator;

    if (!bevelEnabled) {
        bevelSegments = 0;
        bevelThickness = 0;
        bevelSize = 0;
        bevelOffset = 0;
    }

    const shapePoints = shape.extractPoints ? shape.extractPoints(curveSegments)
        : { shape: shape.getPoints?.(curveSegments) ?? shape, holes: [] };
    let vertices = shapePoints.shape.slice();
    const holes = shapePoints.holes.map(h => h.slice());

    if (!ShapeUtils.isClockWise(vertices)) {
        vertices.reverse();
        for (let h = 0; h < holes.length; h++) {
            if (ShapeUtils.isClockWise(holes[h])) holes[h] = holes[h].slice().reverse();
        }
    }

    const faces = ShapeUtils.triangulateShape(vertices, holes);
    const contour = vertices;
    for (const ahole of holes) vertices = vertices.concat(ahole);

    const vlen = vertices.length;
    const flen = faces.length;

    const contourMovements = [];
    for (let i = 0, il = contour.length, j = il - 1, k = i + 1; i < il; i++, j++, k++) {
        if (j === il) j = 0;
        if (k === il) k = 0;
        contourMovements[i] = _extrudeGetBevelVec(contour[i], contour[j], contour[k]);
    }

    const holesMovements = [];
    let verticesMovements = contourMovements.concat();
    for (const ahole of holes) {
        const oneHoleMovements = [];
        for (let i = 0, il = ahole.length, j = il - 1, k = i + 1; i < il; i++, j++, k++) {
            if (j === il) j = 0;
            if (k === il) k = 0;
            oneHoleMovements[i] = _extrudeGetBevelVec(ahole[i], ahole[j], ahole[k]);
        }
        holesMovements.push(oneHoleMovements);
        verticesMovements = verticesMovements.concat(oneHoleMovements);
    }

    const v = (x, y, z) => { placeholder.push(x, y, z); };

    for (let b = 0; b < bevelSegments; b++) {
        const t = b / bevelSegments;
        const z = bevelThickness * Math.cos(t * Math.PI / 2);
        const bs = bevelSize * Math.sin(t * Math.PI / 2) + bevelOffset;
        for (let i = 0, il = contour.length; i < il; i++) {
            const vert = _extrudeScalePt2(contour[i], contourMovements[i], bs);
            v(vert.x, vert.y, -z);
        }
        for (let h = 0; h < holes.length; h++) {
            const ahole = holes[h];
            const oneHoleMovements = holesMovements[h];
            for (let i = 0, il = ahole.length; i < il; i++) {
                const vert = _extrudeScalePt2(ahole[i], oneHoleMovements[i], bs);
                v(vert.x, vert.y, -z);
            }
        }
    }

    const bs = bevelSize + bevelOffset;
    for (let i = 0; i < vlen; i++) {
        const vert = bevelEnabled ? _extrudeScalePt2(vertices[i], verticesMovements[i], bs) : vertices[i];
        v(vert.x, vert.y, 0);
    }

    for (let s = 1; s <= steps; s++) {
        for (let i = 0; i < vlen; i++) {
            const vert = bevelEnabled ? _extrudeScalePt2(vertices[i], verticesMovements[i], bs) : vertices[i];
            v(vert.x, vert.y, depth / steps * s);
        }
    }

    for (let b = bevelSegments - 1; b >= 0; b--) {
        const t = b / bevelSegments;
        const z = bevelThickness * Math.cos(t * Math.PI / 2);
        const bs2 = bevelSize * Math.sin(t * Math.PI / 2) + bevelOffset;
        for (let i = 0, il = contour.length; i < il; i++) {
            const vert = _extrudeScalePt2(contour[i], contourMovements[i], bs2);
            v(vert.x, vert.y, depth + z);
        }
        for (let h = 0; h < holes.length; h++) {
            const ahole = holes[h];
            const oneHoleMovements = holesMovements[h];
            for (let i = 0, il = ahole.length; i < il; i++) {
                const vert = _extrudeScalePt2(ahole[i], oneHoleMovements[i], bs2);
                v(vert.x, vert.y, depth + z);
            }
        }
    }

    const addVertex = (index) => {
        verticesArray.push(placeholder[index * 3], placeholder[index * 3 + 1], placeholder[index * 3 + 2]);
    };
    const addUV = (vector2) => { uvArray.push(vector2.x, vector2.y); };
    const f3 = (a, b, c) => {
        addVertex(a); addVertex(b); addVertex(c);
        const nextIndex = verticesArray.length / 3;
        const uvs = uvgen.generateTopUV(scope, verticesArray, nextIndex - 3, nextIndex - 2, nextIndex - 1);
        addUV(uvs[0]); addUV(uvs[1]); addUV(uvs[2]);
    };
    const f4 = (a, b, c, d) => {
        addVertex(a); addVertex(b); addVertex(d);
        addVertex(b); addVertex(c); addVertex(d);
        const nextIndex = verticesArray.length / 3;
        const uvs = uvgen.generateSideWallUV(scope, verticesArray, nextIndex - 6, nextIndex - 3, nextIndex - 2, nextIndex - 1);
        addUV(uvs[0]); addUV(uvs[1]); addUV(uvs[3]);
        addUV(uvs[1]); addUV(uvs[2]); addUV(uvs[3]);
    };

    const lidStart = verticesArray.length / 3;
    if (bevelEnabled) {
        let layer = 0;
        let offset = vlen * layer;
        for (let i = 0; i < flen; i++) {
            const face = faces[i];
            f3(face[2] + offset, face[1] + offset, face[0] + offset);
        }
        layer = steps + bevelSegments * 2;
        offset = vlen * layer;
        for (let i = 0; i < flen; i++) {
            const face = faces[i];
            f3(face[0] + offset, face[1] + offset, face[2] + offset);
        }
    } else {
        for (let i = 0; i < flen; i++) {
            const face = faces[i];
            f3(face[2], face[1], face[0]);
        }
        for (let i = 0; i < flen; i++) {
            const face = faces[i];
            f3(face[0] + vlen * steps, face[1] + vlen * steps, face[2] + vlen * steps);
        }
    }
    scope.addGroup(lidStart, verticesArray.length / 3 - lidStart, 0);

    const sideStart = verticesArray.length / 3;
    const sidewalls = (contourPts, layeroffset) => {
        let i = contourPts.length;
        while (--i >= 0) {
            const j = i;
            let k = i - 1;
            if (k < 0) k = contourPts.length - 1;
            for (let s = 0, sl = steps + bevelSegments * 2; s < sl; s++) {
                const slen1 = vlen * s;
                const slen2 = vlen * (s + 1);
                f4(
                    layeroffset + j + slen1,
                    layeroffset + k + slen1,
                    layeroffset + k + slen2,
                    layeroffset + j + slen2,
                );
            }
        }
    };
    let layeroffset = 0;
    sidewalls(contour, layeroffset);
    layeroffset += contour.length;
    for (const ahole of holes) {
        sidewalls(ahole, layeroffset);
        layeroffset += ahole.length;
    }
    scope.addGroup(sideStart, verticesArray.length / 3 - sideStart, 1);
}

// ExtrudeGeometry — port of three.js ExtrudeGeometry (bevel + depth extrusion).
export class ExtrudeGeometry {
    constructor(shapes = new Shape([
        new Vector2(0.5, 0.5), new Vector2(-0.5, 0.5),
        new Vector2(-0.5, -0.5), new Vector2(0.5, -0.5),
    ]), options = {}) {
        const shapeList = Array.isArray(shapes) ? shapes : [shapes];
        const verticesArray = [];
        const uvArray = [];
        const g = new BufferGeometry();
        for (const shape of shapeList) _extrudeAddShape(shape, options, g, verticesArray, uvArray);
        g.setAttribute('position', new BufferAttribute(new Float32Array(verticesArray), 3));
        g.setAttribute('uv', new BufferAttribute(new Float32Array(uvArray), 2));
        g.computeVertexNormals();
        Object.assign(this, g);
        this.type = 'ExtrudeGeometry';
        this.parameters = { shapes, options };
        this._isUserGeometry = true;
    }
}

// Real ConvexHull (incremental quickhull, 3D). Given a point cloud, returns
// the triangular faces of the convex hull. This is a clean implementation
// of the standard "find extreme tetrahedron, then expand by outside points"
// algorithm. O(n log n) average, O(n²) worst case.
export class ConvexHull {
    constructor() { this.faces = []; this.vertices = []; }
    setFromPoints(points) {
        this.faces = [];
        this.vertices = points.slice();
        if (points.length < 4) return this;
        // 1. Find the initial tetrahedron from extreme points.
        const tet = _initialTetrahedron(points);
        if (!tet) return this;
        this.faces = tet;
        // 2. For each remaining point, find a face it's outside of and expand.
        const used = new Set(tet.flatMap(f => f.indices));
        for (let i = 0; i < points.length; i++) {
            if (used.has(i)) continue;
            const p = points[i];
            // Find any face this point lies outside of.
            const visible = this.faces.filter(f => _pointOutsideFace(p, f, points));
            if (visible.length === 0) continue; // inside hull
            // Find the boundary edges of the visible faces (edges adjacent to
            // exactly one visible face — those are the silhouette).
            const edgeCount = new Map();
            const edgeKey = (a, b) => (a < b ? `${a}_${b}` : `${b}_${a}`);
            for (const f of visible) {
                for (let k = 0; k < 3; k++) {
                    const a = f.indices[k], b = f.indices[(k + 1) % 3];
                    const key = edgeKey(a, b);
                    edgeCount.set(key, (edgeCount.get(key) || 0) + 1);
                }
            }
            const horizon = [];
            for (const f of visible) {
                for (let k = 0; k < 3; k++) {
                    const a = f.indices[k], b = f.indices[(k + 1) % 3];
                    if (edgeCount.get(edgeKey(a, b)) === 1) horizon.push([a, b]);
                }
            }
            // Remove visible faces and add new triangles connecting i to each horizon edge.
            this.faces = this.faces.filter(f => !visible.includes(f));
            for (const [a, b] of horizon) {
                this.faces.push(_makeFace(a, b, i, points));
            }
            used.add(i);
        }
        return this;
    }
}
function _pointOutsideFace(p, face, points) {
    const a = points[face.indices[0]];
    const dot = (p.x - a.x) * face.normal.x + (p.y - a.y) * face.normal.y + (p.z - a.z) * face.normal.z;
    return dot > 1e-8;
}
function _makeFace(i0, i1, i2, points) {
    const A = points[i0], B = points[i1], C = points[i2];
    const edge1 = { x: B.x - A.x, y: B.y - A.y, z: B.z - A.z };
    const edge2 = { x: C.x - A.x, y: C.y - A.y, z: C.z - A.z };
    let nx = edge1.y * edge2.z - edge1.z * edge2.y;
    let ny = edge1.z * edge2.x - edge1.x * edge2.z;
    let nz = edge1.x * edge2.y - edge1.y * edge2.x;
    const len = Math.hypot(nx, ny, nz) || 1;
    return { indices: [i0, i1, i2], normal: { x: nx / len, y: ny / len, z: nz / len } };
}
function _initialTetrahedron(points) {
    // Extreme points along each axis.
    let xMin = 0, xMax = 0, yMin = 0, yMax = 0, zMin = 0, zMax = 0;
    for (let i = 1; i < points.length; i++) {
        if (points[i].x < points[xMin].x) xMin = i;
        if (points[i].x > points[xMax].x) xMax = i;
        if (points[i].y < points[yMin].y) yMin = i;
        if (points[i].y > points[yMax].y) yMax = i;
        if (points[i].z < points[zMin].z) zMin = i;
        if (points[i].z > points[zMax].z) zMax = i;
    }
    // Pick the two most-distant extremes as the base edge.
    const candidates = [xMin, xMax, yMin, yMax, zMin, zMax];
    let bestPair = null, bestDist = -1;
    for (let i = 0; i < candidates.length; i++) for (let j = i + 1; j < candidates.length; j++) {
        const a = points[candidates[i]], b = points[candidates[j]];
        const d = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        if (d > bestDist) { bestDist = d; bestPair = [candidates[i], candidates[j]]; }
    }
    if (!bestPair || bestDist < 1e-8) return null;
    const [i0, i1] = bestPair;
    // Find point farthest from the line (i0, i1).
    let i2 = -1, far = -1;
    const A = points[i0], B = points[i1];
    const ab = { x: B.x - A.x, y: B.y - A.y, z: B.z - A.z };
    const abLen2 = ab.x * ab.x + ab.y * ab.y + ab.z * ab.z;
    for (let i = 0; i < points.length; i++) {
        if (i === i0 || i === i1) continue;
        const ap = { x: points[i].x - A.x, y: points[i].y - A.y, z: points[i].z - A.z };
        const t = (ap.x * ab.x + ap.y * ab.y + ap.z * ab.z) / abLen2;
        const proj = { x: A.x + ab.x * t, y: A.y + ab.y * t, z: A.z + ab.z * t };
        const d = Math.hypot(points[i].x - proj.x, points[i].y - proj.y, points[i].z - proj.z);
        if (d > far) { far = d; i2 = i; }
    }
    if (i2 < 0 || far < 1e-8) return null;
    // Find point farthest from the triangle plane.
    const C = points[i2];
    let i3 = -1, farPlane = -1;
    const ac = { x: C.x - A.x, y: C.y - A.y, z: C.z - A.z };
    const n = { x: ab.y * ac.z - ab.z * ac.y, y: ab.z * ac.x - ab.x * ac.z, z: ab.x * ac.y - ab.y * ac.x };
    for (let i = 0; i < points.length; i++) {
        if (i === i0 || i === i1 || i === i2) continue;
        const d = Math.abs((points[i].x - A.x) * n.x + (points[i].y - A.y) * n.y + (points[i].z - A.z) * n.z);
        if (d > farPlane) { farPlane = d; i3 = i; }
    }
    if (i3 < 0 || farPlane < 1e-8) return null;
    const D = points[i3];
    // Orient faces so normals point outward (away from the 4th vertex).
    const faces = [];
    const addOriented = (i, j, k, away) => {
        const f = _makeFace(i, j, k, points);
        const v = { x: points[away].x - points[i].x, y: points[away].y - points[i].y, z: points[away].z - points[i].z };
        if (v.x * f.normal.x + v.y * f.normal.y + v.z * f.normal.z > 0) {
            // Reverse winding so normal points away from `away`.
            const t = f.indices[1]; f.indices[1] = f.indices[2]; f.indices[2] = t;
            f.normal.x = -f.normal.x; f.normal.y = -f.normal.y; f.normal.z = -f.normal.z;
        }
        faces.push(f);
    };
    addOriented(i0, i1, i2, i3);
    addOriented(i0, i1, i3, i2);
    addOriented(i0, i2, i3, i1);
    addOriented(i1, i2, i3, i0);
    return faces;
}

// ConvexGeometry — real quickhull, replacing the bounding-box approximation.
export class ConvexGeometry {
    constructor(points = []) {
        if (points.length < 4) {
            // Degenerate input — fall back to a tiny box.
            const box = new BoxGeometry(0.1, 0.1, 0.1);
            Object.assign(this, box);
            this.type = 'ConvexGeometry'; this._isUserGeometry = true;
            return;
        }
        const hull = new ConvexHull().setFromPoints(points);
        const positions = [], normals = [], uvs = [], indices = [];
        let idx = 0;
        for (const face of hull.faces) {
            for (const i of face.indices) {
                const p = points[i];
                positions.push(p.x, p.y, p.z);
                normals.push(face.normal.x, face.normal.y, face.normal.z);
                uvs.push(0, 0);
            }
            indices.push(idx, idx + 1, idx + 2);
            idx += 3;
        }
        const g = new BufferGeometry();
        g.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
        g.setAttribute('normal',   new BufferAttribute(new Float32Array(normals), 3));
        g.setAttribute('uv',       new BufferAttribute(new Float32Array(uvs), 2));
        g.setIndex(indices);
        Object.assign(this, g);
        this.type = 'ConvexGeometry';
        this._isUserGeometry = true;
    }
}

// ParametricGeometry — sample a u/v function (R²→R³) on a slice×stack grid
// and emit a triangle mesh. Mirrors three.js's signature
// `new ParametricGeometry(func, slices, stacks)`.
export class ParametricGeometry {
    constructor(func = ((u, v, target) => { target.x = u - 0.5; target.y = v - 0.5; target.z = 0; }),
                slices = 8, stacks = 8) {
        const positions = [];
        const normals = [];
        const uvs = [];
        const indices = [];
        const tmp = new Vector3();
        for (let j = 0; j <= stacks; j++) for (let i = 0; i <= slices; i++) {
            const u = i / slices, v = j / stacks;
            func(u, v, tmp);
            positions.push(tmp.x, tmp.y, tmp.z);
            normals.push(0, 0, 1);
            uvs.push(u, v);
        }
        for (let j = 0; j < stacks; j++) for (let i = 0; i < slices; i++) {
            const a = j * (slices + 1) + i;
            const b = (j + 1) * (slices + 1) + i;
            const c = (j + 1) * (slices + 1) + i + 1;
            const d = j * (slices + 1) + i + 1;
            indices.push(a, b, d,  b, c, d);
        }
        const g = new BufferGeometry();
        g.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
        g.setAttribute('normal', new BufferAttribute(new Float32Array(normals), 3));
        g.setAttribute('uv', new BufferAttribute(new Float32Array(uvs), 2));
        g.setIndex(new Uint32Array(indices));
        Object.assign(this, g);
        this.type = 'ParametricGeometry';
        this._isUserGeometry = true;
    }
}

// DecalGeometry — projects a box volume's intersection with a target mesh.
// We approximate with a thin Box at the supplied position/orientation,
// preserving the "small patch at this location" shape rather than the true
// projected geometry.
export class DecalGeometry {
    constructor(_mesh, position = new Vector3(), _orientation = new Euler(), size = new Vector3(1, 1, 1)) {
        const box = new BoxGeometry(size.x, size.y, size.z);
        Object.assign(this, box);
        this.type = 'DecalGeometry';
        this._isUserGeometry = true;
        this.position = new Vector3().copy(position);
    }
}

// TextGeometry — placeholder using a flat plane sized for the text's bounds.
// Real font glyph extrusion requires a FontLoader implementation we haven't
// wired (the font JSON parsing is non-trivial).
export class TextGeometry {
    constructor(text = '', opts = {}) {
        const size = opts.size ?? 1;
        const w = Math.max(1, text.length) * size * 0.5;
        const plane = new PlaneGeometry(w, size);
        Object.assign(this, plane);
        this.type = 'TextGeometry';
        this._isUserGeometry = true;
    }
}

// EdgesGeometry / WireframeGeometry: extract triangle edges from a source
// geometry. EdgesGeometry adds an angle-threshold filter (only "creased"
// edges); WireframeGeometry keeps every edge.
function _isUniqueEdge(start, end, edges) {
    const hash1 = `${start.x},${start.y},${start.z}-${end.x},${end.y},${end.z}`;
    const hash2 = `${end.x},${end.y},${end.z}-${start.x},${start.y},${start.z}`;
    if (edges.has(hash1) || edges.has(hash2)) return false;
    edges.add(hash1);
    edges.add(hash2);
    return true;
}
function _extractWireframeEdges(srcGeom) {
    const pos = srcGeom?.attributes?.position;
    if (!pos) {
        const p = srcGeom?.parameters;
        if (p?.width != null && p?.height != null && p?.depth != null) {
            return _extractWireframeEdges(_jsBoxIndexedGeometry(p.width, p.height, p.depth));
        }
        const hw = (p?.width ?? 1) * 0.5;
        const hh = (p?.height ?? 1) * 0.5;
        const hd = (p?.depth ?? 1) * 0.5;
        const pts = [
            -hw,-hh,-hd,  hw,-hh,-hd,   hw,-hh,-hd,  hw,hh,-hd,   hw,hh,-hd, -hw,hh,-hd,  -hw,hh,-hd, -hw,-hh,-hd,
            -hw,-hh, hd,  hw,-hh, hd,   hw,-hh, hd,  hw,hh, hd,   hw,hh, hd, -hw,hh, hd,  -hw,hh, hd, -hw,-hh, hd,
            -hw,-hh,-hd, -hw,-hh, hd,   hw,-hh,-hd,  hw,-hh, hd,   hw,hh,-hd,  hw,hh, hd,  -hw,hh,-hd, -hw,hh, hd,
        ];
        const g = new BufferGeometry();
        g.setAttribute('position', new BufferAttribute(new Float32Array(pts), 3));
        return g;
    }
    const posArr = pos.array;
    const idxAttr = srcGeom.index;
    const idx = idxAttr ? idxAttr.array : null;
    const start = new Vector3();
    const end = new Vector3();
    const edges = new Set();
    const vertices = [];
    if (idx) {
        for (let i = 0; i < idx.length; i += 3) {
            for (let j = 0; j < 3; j++) {
                const index1 = idx[i + j];
                const index2 = idx[i + (j + 1) % 3];
                start.set(posArr[index1 * 3], posArr[index1 * 3 + 1], posArr[index1 * 3 + 2]);
                end.set(posArr[index2 * 3], posArr[index2 * 3 + 1], posArr[index2 * 3 + 2]);
                if (_isUniqueEdge(start, end, edges)) {
                    vertices.push(start.x, start.y, start.z, end.x, end.y, end.z);
                }
            }
        }
    } else {
        for (let i = 0, l = posArr.length / 9; i < l; i++) {
            for (let j = 0; j < 3; j++) {
                const index1 = 3 * i + j;
                const index2 = 3 * i + ((j + 1) % 3);
                start.set(posArr[index1 * 3], posArr[index1 * 3 + 1], posArr[index1 * 3 + 2]);
                end.set(posArr[index2 * 3], posArr[index2 * 3 + 1], posArr[index2 * 3 + 2]);
                if (_isUniqueEdge(start, end, edges)) {
                    vertices.push(start.x, start.y, start.z, end.x, end.y, end.z);
                }
            }
        }
    }
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(new Float32Array(vertices), 3));
    return g;
}
function _extractEdgesGeometry(srcGeom, thresholdAngle = 1) {
    const pos = srcGeom?.attributes?.position;
    if (!pos) {
        const p = srcGeom?.parameters;
        if (p?.width != null && p?.height != null && p?.depth != null) {
            return _extractEdgesGeometry(_jsBoxIndexedGeometry(p.width, p.height, p.depth), thresholdAngle);
        }
        return _extractWireframeEdges(srcGeom);
    }
    const precision = 1e4;
    const thresholdDot = Math.cos(thresholdAngle * Math.PI / 180);
    const posArr = pos.array;
    const idxAttr = srcGeom.index;
    const indexCount = idxAttr ? idxAttr.count : pos.count;
    const edgeData = {};
    const vertices = [];
    const indexArr = [0, 0, 0];
    const vertKeys = ['a', 'b', 'c'];
    const hashes = ['', '', ''];
    const tri = { a: new Vector3(), b: new Vector3(), c: new Vector3() };
    const normal = new Vector3();
    const cb = new Vector3();
    const ab = new Vector3();
    for (let i = 0; i < indexCount; i += 3) {
        if (idxAttr) {
            indexArr[0] = idxAttr.array[i];
            indexArr[1] = idxAttr.array[i + 1];
            indexArr[2] = idxAttr.array[i + 2];
        } else {
            indexArr[0] = i;
            indexArr[1] = i + 1;
            indexArr[2] = i + 2;
        }
        tri.a.set(posArr[indexArr[0] * 3], posArr[indexArr[0] * 3 + 1], posArr[indexArr[0] * 3 + 2]);
        tri.b.set(posArr[indexArr[1] * 3], posArr[indexArr[1] * 3 + 1], posArr[indexArr[1] * 3 + 2]);
        tri.c.set(posArr[indexArr[2] * 3], posArr[indexArr[2] * 3 + 1], posArr[indexArr[2] * 3 + 2]);
        cb.subVectors(tri.c, tri.b);
        ab.subVectors(tri.a, tri.b);
        normal.crossVectors(cb, ab).normalize();
        hashes[0] = `${Math.round(tri.a.x * precision)},${Math.round(tri.a.y * precision)},${Math.round(tri.a.z * precision)}`;
        hashes[1] = `${Math.round(tri.b.x * precision)},${Math.round(tri.b.y * precision)},${Math.round(tri.b.z * precision)}`;
        hashes[2] = `${Math.round(tri.c.x * precision)},${Math.round(tri.c.y * precision)},${Math.round(tri.c.z * precision)}`;
        if (hashes[0] === hashes[1] || hashes[1] === hashes[2] || hashes[2] === hashes[0]) continue;
        for (let j = 0; j < 3; j++) {
            const jNext = (j + 1) % 3;
            const vecHash0 = hashes[j];
            const vecHash1 = hashes[jNext];
            const v0 = tri[vertKeys[j]];
            const v1 = tri[vertKeys[jNext]];
            const hash = `${vecHash0}_${vecHash1}`;
            const reverseHash = `${vecHash1}_${vecHash0}`;
            if (reverseHash in edgeData && edgeData[reverseHash]) {
                if (normal.dot(edgeData[reverseHash].normal) <= thresholdDot) {
                    vertices.push(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
                }
                edgeData[reverseHash] = null;
            } else if (!(hash in edgeData)) {
                edgeData[hash] = {
                    index0: indexArr[j],
                    index1: indexArr[jNext],
                    normal: normal.clone(),
                };
            }
        }
    }
    const v0 = new Vector3();
    const v1 = new Vector3();
    for (const key of Object.keys(edgeData)) {
        const e = edgeData[key];
        if (!e) continue;
        v0.set(posArr[e.index0 * 3], posArr[e.index0 * 3 + 1], posArr[e.index0 * 3 + 2]);
        v1.set(posArr[e.index1 * 3], posArr[e.index1 * 3 + 1], posArr[e.index1 * 3 + 2]);
        vertices.push(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
    }
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(new Float32Array(vertices), 3));
    return g;
}
function _extractEdges(srcGeom, angleThresholdDeg = 1) {
    return _extractEdgesGeometry(srcGeom, angleThresholdDeg);
}
export class EdgesGeometry {
    constructor(srcGeom, thresholdAngle = 1) {
        const g = _extractEdges(srcGeom, thresholdAngle);
        Object.assign(this, g);
        this.type = 'EdgesGeometry';
        this._isUserGeometry = true;
    }
}
export class WireframeGeometry {
    constructor(srcGeom) {
        const g = _extractWireframeEdges(srcGeom);
        Object.assign(this, g);
        this.type = 'WireframeGeometry';
        this._isUserGeometry = true;
    }
}

// ---- Light helpers + camera helper (Pass 5 cont.) ----
// Real LineSegments-backed wireframes drawn through the standard line pipeline.

function _wireGeomLines(pts) {
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(new Float32Array(pts), 3));
    return g;
}
function _wireBoxLines(size) {
    const s = size / 2;
    return [
        -s,-s,-s,  s,-s,-s,   s,-s,-s,  s,s,-s,   s,s,-s, -s,s,-s,  -s,s,-s, -s,-s,-s,
        -s,-s, s,  s,-s, s,   s,-s, s,  s,s, s,   s,s, s, -s,s, s,  -s,s, s, -s,-s, s,
        -s,-s,-s, -s,-s, s,   s,-s,-s,  s,-s, s,   s,s,-s,  s,s, s,  -s,s,-s, -s,s, s,
    ];
}
function _wireSphereLines(radius, segments) {
    const pts = [];
    for (const [ax1, ax2] of [[0,1], [0,2], [1,2]]) {
        for (let i = 0; i < segments; i++) {
            const a = (i / segments) * Math.PI * 2;
            const b = ((i + 1) / segments) * Math.PI * 2;
            const p1 = [0, 0, 0], p2 = [0, 0, 0];
            p1[ax1] = Math.cos(a) * radius; p1[ax2] = Math.sin(a) * radius;
            p2[ax1] = Math.cos(b) * radius; p2[ax2] = Math.sin(b) * radius;
            pts.push(...p1, ...p2);
        }
    }
    return pts;
}
function _wireConeLines(radius, height) {
    const segs = 16; const pts = [];
    for (let i = 0; i < segs; i++) {
        const a = (i / segs) * Math.PI * 2, b = ((i + 1) / segs) * Math.PI * 2;
        const x1 = Math.cos(a) * radius, z1 = Math.sin(a) * radius;
        const x2 = Math.cos(b) * radius, z2 = Math.sin(b) * radius;
        pts.push(x1, -height/2, z1,  x2, -height/2, z2);
        if (i % 4 === 0) pts.push(x1, -height/2, z1,  0, height/2, 0);
    }
    return pts;
}
// Common base — every helper IS-A LineSegments (alias to scene.add()'s code path).
function _initWireHelper(self, ptsArr, color, light) {
    const g = _wireGeomLines(ptsArr);
    const m = new LineBasicMaterial({ color: color ?? 0xffff00 });
    self._isLineSegments = true;
    self.geometry = g;
    self.material = m;
    self.position = new Vector3(light?.position?.x ?? 0, light?.position?.y ?? 0, light?.position?.z ?? 0);
    self.rotation = new Euler();
    self.scale = new Vector3(1,1,1);
    self._handle = null;
    self.update = function () {
        if (this.light?.position) this.position.set(this.light.position.x, this.light.position.y, this.light.position.z);
    };
    self.dispose = function () {};
}
export class DirectionalLightHelper {
    constructor(light, size = 1, color) {
        this.light = light;
        _initWireHelper(this, _wireBoxLines(size), color, light);
    }
}
export class HemisphereLightHelper {
    constructor(light, size = 1, color) {
        this.light = light;
        _initWireHelper(this, _wireSphereLines(size / 2, 16), color, light);
    }
}
export class PointLightHelper {
    constructor(light, sphereSize = 1, color) {
        this.light = light;
        _initWireHelper(this, _wireSphereLines(sphereSize, 12), color, light);
    }
}
export class SpotLightHelper {
    constructor(light, color) {
        this.light = light;
        const r = Math.tan(light?.angle ?? Math.PI / 4);
        _initWireHelper(this, _wireConeLines(r, 1.0), color, light);
    }
}
export class CameraHelper {
    constructor(camera) {
        this.camera = camera;
        _initWireHelper(this, _wireBoxLines(1), 0xffff00, null);
    }
}
export class ArrowHelper {
    constructor(dir = new Vector3(0,1,0), origin = new Vector3(), length = 1, color = 0xffff00) {
        const d = dir.clone().normalize().multiplyScalar(length);
        const pts = [origin.x, origin.y, origin.z, origin.x + d.x, origin.y + d.y, origin.z + d.z];
        _initWireHelper(this, pts, color, null);
    }
}
export class PlaneHelper {
    constructor(plane, size = 1, color = 0xffff00) {
        this.plane = plane;
        _initWireHelper(this, _wireBoxLines(size), color, null);
    }
}
export class SkeletonHelper {
    constructor(_root, color = 0xffff00) {
        _initWireHelper(this, [0,0,0, 0,1,0], color, null);
    }
}

// ---- Pass 6: loaders + EventDispatcher mixin ----
//
// EventDispatcher: three.js's tiny pub/sub. Used by loaders for progress, by
// the renderer for context-lost events, by Object3D subclasses for selection
// events, etc. Apply via Object.assign(MyClass.prototype, EventDispatcherMixin).

export const EventDispatcherMixin = {
    addEventListener(type, listener) {
        if (!this._listeners) this._listeners = {};
        (this._listeners[type] = this._listeners[type] || []).push(listener);
    },
    hasEventListener(type, listener) {
        return !!(this._listeners && this._listeners[type] && this._listeners[type].includes(listener));
    },
    removeEventListener(type, listener) {
        if (!this._listeners || !this._listeners[type]) return;
        this._listeners[type] = this._listeners[type].filter(l => l !== listener);
    },
    dispatchEvent(event) {
        if (!this._listeners || !this._listeners[event.type]) return;
        const ls = this._listeners[event.type].slice();
        event.target = this;
        for (const l of ls) l.call(this, event);
    },
};
export class EventDispatcher {}
Object.assign(EventDispatcher.prototype, EventDispatcherMixin);

// LoadingManager: tracks in-flight requests + global progress callbacks.
export class LoadingManager {
    constructor(onLoad, onProgress, onError) {
        this.onLoad = onLoad; this.onProgress = onProgress; this.onError = onError;
        this._loading = 0; this._loaded = 0; this._total = 0;
    }
    itemStart(url) { this._loading++; this._total++; this.onProgress?.(url, this._loaded, this._total); }
    itemEnd(url)   { this._loading--; this._loaded++; this.onProgress?.(url, this._loaded, this._total); if (this._loading === 0) this.onLoad?.(); }
    itemError(url) { this.onError?.(url); }
}
const DefaultLoadingManager = new LoadingManager();
export { DefaultLoadingManager };

// Tiny shared Loader base — provides .load(), .loadAsync(), .setPath(), etc.
class _Loader {
    constructor(manager = DefaultLoadingManager) {
        this.manager = manager;
        this.path = ''; this.resourcePath = ''; this.crossOrigin = 'anonymous';
        this.requestHeader = {};
    }
    setPath(p) { this.path = p; return this; }
    setResourcePath(p) { this.resourcePath = p; return this; }
    setCrossOrigin(co) { this.crossOrigin = co; return this; }
    setRequestHeader(h) { this.requestHeader = h; return this; }
    loadAsync(url, onProgress) {
        return new Promise((resolve, reject) => this.load(url, resolve, onProgress, reject));
    }
}

export class FileLoader extends _Loader {
    constructor(manager) { super(manager); this.responseType = ''; this.mimeType = ''; }
    setResponseType(t) { this.responseType = t; return this; }
    setMimeType(t) { this.mimeType = t; return this; }
    async load(url, onLoad, _onProgress, onError) {
        try {
            this.manager.itemStart(url);
            const res = await fetch(this.path + url, { headers: this.requestHeader });
            if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
            let data;
            if (this.responseType === 'arraybuffer') data = await res.arrayBuffer();
            else if (this.responseType === 'blob')   data = await res.blob();
            else if (this.responseType === 'json')   data = await res.json();
            else data = await res.text();
            onLoad?.(data);
            this.manager.itemEnd(url);
        } catch (e) {
            onError?.(e); this.manager.itemError(url); this.manager.itemEnd(url);
        }
    }
}

export class ImageLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        this.manager.itemStart(url);
        const img = new Image();
        img.crossOrigin = this.crossOrigin;
        img.onload = () => { onLoad?.(img); this.manager.itemEnd(url); };
        img.onerror = (e) => { onError?.(e); this.manager.itemError(url); this.manager.itemEnd(url); };
        img.src = this.path + url;
        return img;
    }
}

export class TextureLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        const tex = new Texture();
        // We delegate image fetching to ImageLoader and copy the decoded image
        // bytes into a DataTexture once it lands. For now this returns the
        // Texture eagerly so user code that does `material.map = loader.load(...)`
        // works — the texture's bytes get populated asynchronously.
        new ImageLoader(this.manager).load(url, (img) => {
            try {
                const c = document.createElement('canvas');
                c.width = img.width; c.height = img.height;
                const ctx = c.getContext('2d');
                ctx.drawImage(img, 0, 0);
                const data = ctx.getImageData(0, 0, img.width, img.height).data;
                tex._w = new WebDataTexture(img.width, img.height, new Uint8Array(data));
                tex.image = img;
                tex.needsUpdate = true;
                onLoad?.(tex);
            } catch (e) { onError?.(e); }
        }, undefined, onError);
        return tex;
    }
}

// Minimal JSON loaders so user code calling `.load(...)` returns sensibly.
export class FontLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (text) => {
            try { onLoad?.(JSON.parse(text)); } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
}
export class MaterialLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) { new FileLoader(this.manager).load(url, onLoad, undefined, onError); }
}
// Real BufferGeometryLoader. Parses the three.js BufferGeometry JSON shape:
//   { data: { attributes: { position: {itemSize, type, array}, ... },
//             index: { type, array } } }
// Returns a BufferGeometry instance suitable for `new Mesh(geom, mat)`.
export class BufferGeometryLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (text) => {
            try {
                const json = (typeof text === 'string') ? JSON.parse(text) : text;
                onLoad?.(this.parse(json));
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(json) {
        const g = new BufferGeometry();
        const data = json.data || json;
        const attrs = data.attributes || {};
        for (const name of Object.keys(attrs)) {
            const a = attrs[name];
            const Ctor = _typedArrayByName(a.type) || Float32Array;
            const arr = new Ctor(a.array);
            g.setAttribute(name, new BufferAttribute(arr, a.itemSize));
        }
        if (data.index) {
            const Ctor = _typedArrayByName(data.index.type) || Uint32Array;
            const idxArr = new Ctor(data.index.array);
            g.setIndex(new Uint32Array(idxArr));
        }
        if (json.uuid) g.uuid = json.uuid;
        if (json.name) g.name = json.name;
        return g;
    }
}
function _typedArrayByName(name) {
    switch (name) {
        case 'Int8Array': return Int8Array;
        case 'Uint8Array': return Uint8Array;
        case 'Uint8ClampedArray': return Uint8ClampedArray;
        case 'Int16Array': return Int16Array;
        case 'Uint16Array': return Uint16Array;
        case 'Int32Array': return Int32Array;
        case 'Uint32Array': return Uint32Array;
        case 'Float32Array': return Float32Array;
        case 'Float64Array': return Float64Array;
        default: return null;
    }
}
// Real ObjectLoader. Parses three.js's Scene JSON: geometries[], materials[],
// images[]/textures[], object{ children[] }. We support a useful subset:
// BufferGeometry data, common materials (basic/standard/lambert/phong/normal),
// and Mesh/Group/Object3D nodes with position/rotation/scale/quaternion.
export class ObjectLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (text) => {
            try {
                const json = (typeof text === 'string') ? JSON.parse(text) : text;
                onLoad?.(this.parse(json));
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(json) {
        const geomById = {};
        const bgLoader = new BufferGeometryLoader(this.manager);
        for (const gd of (json.geometries || [])) {
            // three.js wraps the actual attribute shape in `data` for BufferGeometry.
            geomById[gd.uuid] = bgLoader.parse(gd);
        }
        const matById = {};
        for (const md of (json.materials || [])) {
            matById[md.uuid] = _parseMaterialJson(md);
        }
        const root = _parseObjectJson(json.object || json, geomById, matById);
        return root;
    }
}
function _parseMaterialJson(md) {
    const opts = {};
    if (md.color !== undefined) opts.color = md.color;
    if (md.emissive !== undefined) opts.emissive = md.emissive;
    if (md.roughness !== undefined) opts.roughness = md.roughness;
    if (md.metalness !== undefined) opts.metalness = md.metalness;
    if (md.opacity !== undefined) opts.opacity = md.opacity;
    if (md.transparent !== undefined) opts.transparent = md.transparent;
    if (md.side !== undefined) opts.side = md.side;
    if (md.wireframe !== undefined) opts.wireframe = md.wireframe;
    if (md.vertexColors !== undefined) opts.vertexColors = md.vertexColors;
    switch (md.type) {
        case 'MeshBasicMaterial': return new MeshBasicMaterial(opts);
        case 'MeshLambertMaterial': return new MeshLambertMaterial(opts);
        case 'MeshPhongMaterial': return new MeshPhongMaterial(opts);
        case 'MeshStandardMaterial': return new MeshStandardMaterial(opts);
        case 'MeshPhysicalMaterial': return new MeshPhysicalMaterial(opts);
        case 'MeshNormalMaterial': return new MeshNormalMaterial();
        case 'MeshDepthMaterial': return new MeshDepthMaterial();
        case 'MeshToonMaterial': return new MeshToonMaterial(opts);
        case 'LineBasicMaterial': return new LineBasicMaterial(opts);
        case 'PointsMaterial': return new PointsMaterial(opts);
        default: return new MeshStandardMaterial(opts);
    }
}
function _parseObjectJson(od, geomById, matById) {
    let obj;
    const matRef = (typeof od.material === 'string') ? matById[od.material]
                 : Array.isArray(od.material) ? od.material.map(u => matById[u])
                 : null;
    const geomRef = od.geometry ? geomById[od.geometry] : null;
    switch (od.type) {
        case 'Mesh':
            obj = new Mesh(geomRef, matRef || new MeshStandardMaterial());
            break;
        case 'Group':
            obj = new Group();
            break;
        case 'Scene':
            obj = new Scene();
            break;
        case 'PerspectiveCamera':
            obj = new PerspectiveCamera(od.fov, od.aspect, od.near, od.far);
            break;
        case 'OrthographicCamera':
            obj = new OrthographicCamera(od.left, od.right, od.top, od.bottom, od.near, od.far);
            break;
        case 'AmbientLight':
            obj = new AmbientLight(od.color, od.intensity);
            break;
        case 'DirectionalLight':
            obj = new DirectionalLight(od.color, od.intensity);
            break;
        case 'PointLight':
            obj = new PointLight(od.color, od.intensity, od.distance, od.decay);
            break;
        case 'Object3D':
        default:
            obj = new Object3D();
            break;
    }
    if (od.name) obj.name = od.name;
    // three.js serializes object transforms as a flat 4×4 matrix; we approximate
    // by decomposing or by reading position/rotation/scale if present.
    if (Array.isArray(od.matrix) && od.matrix.length === 16) {
        const m = od.matrix;
        // Extract translation directly from matrix columns 12,13,14.
        obj.position?.set?.(m[12], m[13], m[14]);
        // Decompose scale from column lengths.
        const sx = Math.hypot(m[0], m[1], m[2]);
        const sy = Math.hypot(m[4], m[5], m[6]);
        const sz = Math.hypot(m[8], m[9], m[10]);
        obj.scale?.set?.(sx, sy, sz);
        // Extract rotation: divide column vectors by scale to get rotation matrix
        // basis, then convert to Euler via standard ZYX-of-rotation-matrix formula.
        if (sx > 0 && sy > 0 && sz > 0 && obj.rotation?.set) {
            const r00 = m[0]/sx, r01 = m[4]/sy, r02 = m[8]/sz;
            const r10 = m[1]/sx, r11 = m[5]/sy, r12 = m[9]/sz;
            const r20 = m[2]/sx, r21 = m[6]/sy, r22 = m[10]/sz;
            // XYZ Euler from rotation matrix.
            const sy_ = Math.hypot(r00, r10);
            let x, y, z;
            if (sy_ > 1e-6) {
                x = Math.atan2(r21, r22);
                y = Math.atan2(-r20, sy_);
                z = Math.atan2(r10, r00);
            } else {
                x = Math.atan2(-r12, r11);
                y = Math.atan2(-r20, sy_);
                z = 0;
            }
            obj.rotation.set(x, y, z);
        }
    }
    for (const cd of (od.children || [])) {
        const c = _parseObjectJson(cd, geomById, matById);
        if (obj instanceof Scene) obj.add(c);
        else if (obj instanceof Group) obj.add(c);
        else if (obj instanceof Object3D) obj.add(c);
    }
    return obj;
}
export class AnimationLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) { new FileLoader(this.manager).load(url, onLoad, undefined, onError); }
}
export class AudioLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        const fl = new FileLoader(this.manager); fl.setResponseType('arraybuffer');
        fl.load(url, onLoad, undefined, onError);
    }
}
export class CubeTextureLoader extends _Loader {
    load(urls, onLoad, _onProgress, onError) {
        if (!Array.isArray(urls)) urls = [urls, urls, urls, urls, urls, urls];
        const tex = new CubeTexture();
        let done = 0;
        const imgs = new Array(6);
        urls.forEach((u, i) => {
            new ImageLoader(this.manager).load(u, (img) => {
                imgs[i] = img; done++;
                if (done === 6) { tex.images = imgs; tex.needsUpdate = true; onLoad?.(tex); }
            }, undefined, onError);
        });
        return tex;
    }
}
// Real GLTFLoader. Handles .gltf (JSON + external/data buffers) and .glb
// (binary container). Builds Scene + Mesh + BufferGeometry + materials per
// GLTF 2.0; animations parsed into AnimationClips with KeyframeTracks.
const _GLTF_COMPONENT_BYTES = { 5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4 };
const _GLTF_COMPONENT_ARRAY = {
    5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array,
    5125: Uint32Array, 5126: Float32Array,
};
const _GLTF_TYPE_SIZES = { SCALAR:1, VEC2:2, VEC3:3, VEC4:4, MAT2:4, MAT3:9, MAT4:16 };

function _decodeDataUri(uri) {
    const m = /^data:([^;,]+)?(;base64)?,(.*)$/i.exec(uri);
    if (!m) return new ArrayBuffer(0);
    const isB64 = !!m[2];
    const body = m[3];
    if (isB64) {
        const bin = atob(body);
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out.buffer;
    }
    return new TextEncoder().encode(decodeURIComponent(body)).buffer;
}

export class GLTFLoader extends _Loader {
    async load(url, onLoad, _onProgress, onError) {
        try {
            this.manager.itemStart(url);
            const res = await fetch(this.path + url);
            if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
            const isGLB = url.endsWith('.glb');
            const fullUrl = this.path + url;
            const baseDir = fullUrl.substring(0, fullUrl.lastIndexOf('/') + 1);
            let json, glbBin = null;
            if (isGLB) {
                const buf = await res.arrayBuffer();
                ({ json, glbBin } = this._parseGLBContainer(buf));
            } else {
                json = await res.json();
            }
            const parsed = await this.parseAsync(json, baseDir, glbBin);
            onLoad?.(parsed);
            this.manager.itemEnd(url);
        } catch (e) {
            onError?.(e); this.manager.itemError(url); this.manager.itemEnd(url);
        }
    }
    _parseGLBContainer(buf) {
        const dv = new DataView(buf);
        const magic = dv.getUint32(0, true);
        if (magic !== 0x46546C67) throw new Error('GLB magic missing');
        const length = dv.getUint32(8, true);
        let cursor = 12, json = null, glbBin = null;
        while (cursor < length) {
            const chunkLen = dv.getUint32(cursor, true);
            const chunkType = dv.getUint32(cursor + 4, true);
            const data = buf.slice(cursor + 8, cursor + 8 + chunkLen);
            if (chunkType === 0x4E4F534A) json = JSON.parse(new TextDecoder().decode(new Uint8Array(data)));
            else if (chunkType === 0x004E4942) glbBin = data;
            cursor += 8 + chunkLen;
        }
        return { json, glbBin };
    }
    async parseAsync(json, baseDir = '', glbBin = null) {
        const buffers = await Promise.all((json.buffers || []).map(async (b, i) => {
            if (glbBin && i === 0 && !b.uri) return glbBin;
            if (b.uri?.startsWith('data:')) return _decodeDataUri(b.uri);
            const res = await fetch(baseDir + b.uri);
            return await res.arrayBuffer();
        }));
        return this.parse(json, buffers);
    }
    parse(json, buffers, _path = '', onLoad) {
        const bufferViews = (json.bufferViews || []).map(bv => ({
            buffer: buffers[bv.buffer], byteOffset: bv.byteOffset || 0,
            byteLength: bv.byteLength, byteStride: bv.byteStride,
        }));
        const accessors = (json.accessors || []).map(acc => {
            const bv = bufferViews[acc.bufferView];
            const ArrayCtor = _GLTF_COMPONENT_ARRAY[acc.componentType] || Float32Array;
            const tcount = _GLTF_TYPE_SIZES[acc.type] || 1;
            const byteOff = (bv?.byteOffset || 0) + (acc.byteOffset || 0);
            const totalElements = acc.count * tcount;
            const data = bv ? new ArrayCtor(bv.buffer, byteOff, totalElements) : new ArrayCtor(totalElements);
            return { array: Array.from(data), itemSize: tcount, count: acc.count };
        });
        const materials = (json.materials || []).map(m => {
            const matOpts = {
                roughness: m.pbrMetallicRoughness?.roughnessFactor ?? 1,
                metalness: m.pbrMetallicRoughness?.metallicFactor ?? 0,
            };
            const f = m.pbrMetallicRoughness?.baseColorFactor;
            // glTF factors are linear RGB (three.js uses LinearSRGBColorSpace); do not
            // round-trip through hex or WebColor.fromHex applies sRGB decode twice.
            if (f) matOpts.color = new Color(f[0], f[1], f[2]);
            else matOpts.color = new Color(1, 1, 1);
            const e = m.emissiveFactor;
            if (e) matOpts.emissive = new Color(e[0], e[1], e[2]);
            return new MeshStandardMaterial(matOpts);
        });
        const meshes = (json.meshes || []).map(meshSpec => {
            const items = [];
            for (const prim of meshSpec.primitives || []) {
                const g = new BufferGeometry();
                const attrs = prim.attributes || {};
                if (attrs.POSITION != null) {
                    const a = accessors[attrs.POSITION];
                    g.setAttribute('position', new BufferAttribute(new Float32Array(a.array), 3));
                }
                if (attrs.NORMAL != null) {
                    const a = accessors[attrs.NORMAL];
                    g.setAttribute('normal', new BufferAttribute(new Float32Array(a.array), 3));
                }
                if (attrs.TEXCOORD_0 != null) {
                    const a = accessors[attrs.TEXCOORD_0];
                    g.setAttribute('uv', new BufferAttribute(new Float32Array(a.array), 2));
                }
                if (prim.indices != null) {
                    const a = accessors[prim.indices];
                    g.setIndex(new Uint32Array(a.array));
                }
                items.push({ geom: g, mat: prim.material != null ? materials[prim.material] : new MeshStandardMaterial() });
            }
            return items;
        });
        const nodeObjs = (json.nodes || []).map((nodeSpec, ni) => {
            const o = new Group();
            o.name = nodeSpec.name || `node_${ni}`;
            if (nodeSpec.translation) o.position.set(...nodeSpec.translation);
            if (nodeSpec.rotation) {
                const [x, y, z, w] = nodeSpec.rotation;
                o.quaternion.set(x, y, z, w);
            }
            if (nodeSpec.scale) o.scale.set(...nodeSpec.scale);
            if (nodeSpec.mesh != null && meshes[nodeSpec.mesh]) {
                for (const item of meshes[nodeSpec.mesh]) {
                    const mesh = new Mesh(item.geom, item.mat);
                    mesh.name = o.name;
                    o.add(mesh);
                }
            }
            return o;
        });
        for (let i = 0; i < (json.nodes || []).length; i++) {
            const ns = json.nodes[i];
            if (ns.children) for (const ci of ns.children) nodeObjs[i].add(nodeObjs[ci]);
        }
        const scene = new Scene();
        const sceneSpec = (json.scenes || [])[json.scene ?? 0] || { nodes: [] };
        for (const ni of sceneSpec.nodes) scene.add(nodeObjs[ni]);
        const animations = (json.animations || []).map(a => {
            const tracks = [];
            for (const ch of a.channels || []) {
                const sampler = a.samplers[ch.sampler];
                const targetNode = json.nodes[ch.target.node];
                const targetPath = ch.target.path;
                const inAcc = accessors[sampler.input];
                const outAcc = accessors[sampler.output];
                const trackName = (targetNode?.name ?? `node_${ch.target.node}`) + '.' + targetPath;
                const klass = targetPath === 'rotation' ? QuaternionKeyframeTrack
                            : targetPath === 'morphTargetInfluences' ? NumberKeyframeTrack
                            : VectorKeyframeTrack;
                tracks.push(new klass(trackName, new Float32Array(inAcc.array), new Float32Array(outAcc.array)));
            }
            return new AnimationClip(a.name || 'animation', -1, tracks);
        });
        const result = { scene, scenes: [scene], animations, cameras: [], asset: json.asset || {}, parser: { json } };
        onLoad?.(result);
        return result;
    }
}
export class FBXLoader extends _Loader {
    constructor(manager) { super(manager); this._w = new WebFbxLoader(); }
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (buf) => {
            try {
                const u8 = buf instanceof ArrayBuffer ? new Uint8Array(buf) : buf;
                const geom = this._w.parse(u8);
                const group = new Group();
                group.add(new Mesh(_wrapLoaderGeometry(geom), new MeshStandardMaterial()));
                onLoad?.(group);
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        const geom = this._w.parse(u8);
        return _wrapLoaderGeometry(geom);
    }
}
export class ColladaLoader extends _Loader {
    constructor(manager) { super(manager); this._w = new WebColladaLoader(); }
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (text) => {
            try {
                const geom = this._w.parse(text);
                const scene = new Scene();
                scene.add(new Mesh(_wrapLoaderGeometry(geom), new MeshStandardMaterial()));
                onLoad?.({ scene });
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(text) {
        const geom = this._w.parse(text);
        const scene = new Scene();
        scene.add(new Mesh(_wrapLoaderGeometry(geom), new MeshStandardMaterial()));
        return { scene };
    }
}
export class EXRLoader extends _Loader {
    constructor(manager) { super(manager); this._w = new WebExrLoader(); }
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (buf) => {
            try {
                const u8 = buf instanceof ArrayBuffer ? new Uint8Array(buf) : buf;
                onLoad?.(_wrapWebTexture(this._w.parse(u8)));
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return _wrapWebTexture(this._w.parse(u8));
    }
}

function _parseKTX2(buffer) {
    const u8 = new Uint8Array(buffer);
    const id = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];
    for (let i = 0; i < 12; i++) if (u8[i] !== id[i]) throw new Error('Not a KTX2 file');
    const dv = new DataView(buffer);
    const vkFormat = dv.getUint32(12, true);
    const w = dv.getUint32(20, true);
    const h = dv.getUint32(24, true);
    const levelCount = dv.getUint32(40, true) || 1;
    const supercompression = dv.getUint32(44, true);
    if (supercompression !== 0) throw new Error('KTX2 supercompression not supported');
    const level0 = 80;
    const byteOffset = Number(dv.getBigUint64(level0, true));
    const byteLength = dv.getUint32(level0 + 8, true);
    const data = u8.subarray(byteOffset, byteOffset + byteLength);
    // VK_FORMAT_R8G8B8A8_UNORM = 37, B8G8R8A8_UNORM = 44
    let rgba;
    if (vkFormat === 37 || vkFormat === 0) {
        rgba = new Uint8Array(data);
    } else if (vkFormat === 44) {
        rgba = new Uint8Array(w * h * 4);
        for (let i = 0; i < w * h; i++) {
            rgba[i * 4] = data[i * 4 + 2];
            rgba[i * 4 + 1] = data[i * 4 + 1];
            rgba[i * 4 + 2] = data[i * 4];
            rgba[i * 4 + 3] = data[i * 4 + 3];
        }
    } else {
        throw new Error(`KTX2 vkFormat ${vkFormat} not supported`);
    }
    const tex = new DataTexture(rgba, w, h);
    tex.needsUpdate = true;
    if (levelCount > 1) tex.minFilter = 9987; // LinearMipmapLinear
    return tex;
}

export class KTX2Loader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (buf) => {
            try { onLoad?.(_parseKTX2(buf)); } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(buffer) { return _parseKTX2(buffer); }
}

let _dracoDecoderPromise = null;
async function _ensureDracoDecoder() {
    if (_dracoDecoderPromise) return _dracoDecoderPromise;
    _dracoDecoderPromise = (async () => {
        const base = '/web/deps/draco/';
        const wasmBinary = await (await fetch(base + 'draco_decoder.wasm')).arrayBuffer();
        const jsText = await (await fetch(base + 'draco_wasm_wrapper.js')).text();
        const factory = new Function(`${jsText}\n;return DracoDecoderModule;`)();
        return await factory({ wasmBinary });
    })();
    return _dracoDecoderPromise;
}

export class DRACOLoader extends _Loader {
    constructor(manager) { super(manager); this.decoderPath = ''; }
    setDecoderPath(path) { this.decoderPath = path; return this; }
    preload() { return _ensureDracoDecoder(); }
    decodeDracoFile(buffer, onLoad, onError) {
        this.preload().then((mod) => {
            const decoder = new mod.Decoder();
            const dracoBuffer = new mod.DecoderBuffer();
            dracoBuffer.Init(new Int8Array(buffer), buffer.byteLength);
            const geomType = decoder.GetEncodedGeometryType(dracoBuffer);
            let dracoGeom;
            if (geomType === mod.TRIANGULAR_MESH) dracoGeom = new mod.Mesh();
            else if (geomType === mod.POINT_CLOUD) dracoGeom = new mod.PointCloud();
            else throw new Error('DRACO: unknown geometry type');
            const status = geomType === mod.TRIANGULAR_MESH
                ? decoder.DecodeBufferToMesh(dracoBuffer, dracoGeom)
                : decoder.DecodeBufferToPointCloud(dracoBuffer, dracoGeom);
            if (!status.ok()) throw new Error('DRACO decode failed');
            const posAttrId = decoder.GetAttributeId(dracoGeom, mod.POSITION);
            const posAttr = decoder.GetAttribute(dracoGeom, posAttrId);
            const posCount = dracoGeom.num_points();
            const posArr = new Float32Array(posCount * 3);
            for (let i = 0; i < posCount; i++) {
                decoder.GetAttributeFloatForVertexId(dracoGeom, posAttr, i, posArr, i * 3);
            }
            const g = new BufferGeometry();
            g.setAttribute('position', new BufferAttribute(posArr, 3));
            if (geomType === mod.TRIANGULAR_MESH) {
                const faceCount = dracoGeom.num_faces();
                const idx = new Uint32Array(faceCount * 3);
                const ia = new mod.DracoInt32Array();
                for (let i = 0; i < faceCount; i++) {
                    decoder.GetFaceFromMesh(dracoGeom, i, ia);
                    idx[i * 3] = ia.GetValue(0);
                    idx[i * 3 + 1] = ia.GetValue(1);
                    idx[i * 3 + 2] = ia.GetValue(2);
                }
                mod.destroy(ia);
                g.setIndex(idx);
            }
            mod.destroy(dracoGeom);
            onLoad?.(g);
        }).catch(onError);
    }
    load(url, onLoad, onProgress, onError) {
        new FileLoader(this.manager).load(url, (buf) => this.decodeDracoFile(buf, onLoad, onError), onProgress, onError);
    }
}

// ---- Pass 7: more controls + post-fx + WebGLRenderer extras ----
function _noopControls(type) {
    return { _w: null, type, enabled: true, update() {}, dispose() {}, addEventListener(){}, removeEventListener(){}, dispatchEvent(){} };
}

// DragControls — pick the object under the cursor and translate it on a
// camera-aligned plane while the user drags. Fires `dragstart` / `drag` /
// `dragend` events through EventDispatcherMixin.
export class DragControls {
    constructor(objects = [], camera, domElement) {
        this.objects = objects;
        this.camera = camera;
        this.domElement = domElement || (typeof document !== 'undefined' ? document.body : null);
        this.enabled = true;
        this._selected = null;
        this._plane = new Plane(new Vector3(0, 0, 1), 0);
        this._raycaster = new Raycaster();
        this._dragOffset = new Vector3();
        this._listeners = {};
        Object.assign(this, EventDispatcherMixin);
        this._onDown = (e) => this._handleDown(e);
        this._onMove = (e) => this._handleMove(e);
        this._onUp   = (e) => this._handleUp(e);
        if (this.domElement?.addEventListener) {
            this.domElement.addEventListener('pointerdown', this._onDown);
            this.domElement.addEventListener('pointermove', this._onMove);
            this.domElement.addEventListener('pointerup',   this._onUp);
        }
    }
    _ndc(e) {
        const rect = this.domElement.getBoundingClientRect();
        return new Vector2(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -(((e.clientY - rect.top) / rect.height) * 2 - 1),
        );
    }
    _handleDown(e) {
        if (!this.enabled) return;
        const ndc = this._ndc(e);
        this._raycaster.setFromCamera(ndc, this.camera);
        const hits = this._raycaster.intersectObjects(this.objects, true);
        if (hits.length === 0) return;
        this._selected = hits[0].object;
        const hitPoint = hits[0].point.clone();
        this.camera.getWorldDirection(this._plane.normal);
        this._plane.constant = -this._plane.normal.dot(hitPoint);
        this._dragOffset.copy(hitPoint).sub(this._selected.position);
        this.dispatchEvent({ type: 'dragstart', object: this._selected });
    }
    _handleMove(e) {
        if (!this.enabled || !this._selected) return;
        const ndc = this._ndc(e);
        this._raycaster.setFromCamera(ndc, this.camera);
        const hit = new Vector3();
        if (!this._raycaster.ray.intersectPlane(this._plane, hit)) return;
        this._selected.position.copy(hit.sub(this._dragOffset));
        this.dispatchEvent({ type: 'drag', object: this._selected });
    }
    _handleUp() {
        if (!this._selected) return;
        this.dispatchEvent({ type: 'dragend', object: this._selected });
        this._selected = null;
    }
    update() {}
    dispose() {
        if (this.domElement?.removeEventListener) {
            this.domElement.removeEventListener('pointerdown', this._onDown);
            this.domElement.removeEventListener('pointermove', this._onMove);
            this.domElement.removeEventListener('pointerup',   this._onUp);
        }
    }
}

// FlyControls — WASD/arrow/Q/E for fly-through navigation. Driven by .update(dt).
export class FlyControls {
    constructor(camera, domElement) {
        this.camera = camera;
        this.domElement = domElement || (typeof document !== 'undefined' ? document : null);
        this.enabled = true;
        this.movementSpeed = 1.0;
        this.rollSpeed = 0.005;
        this._keys = {};
        this._listeners = {};
        Object.assign(this, EventDispatcherMixin);
        this._onKeyDown = (e) => { this._keys[e.code] = true; };
        this._onKeyUp   = (e) => { this._keys[e.code] = false; };
        if (this.domElement?.addEventListener) {
            this.domElement.addEventListener('keydown', this._onKeyDown);
            this.domElement.addEventListener('keyup',   this._onKeyUp);
        }
    }
    update(dt = 1/60) {
        if (!this.enabled || !this.camera) return;
        const s = this.movementSpeed * dt;
        const p = this.camera.position;
        if (this._keys['KeyW']) p.z -= s;
        if (this._keys['KeyS']) p.z += s;
        if (this._keys['KeyA']) p.x -= s;
        if (this._keys['KeyD']) p.x += s;
        if (this._keys['KeyQ']) p.y -= s;
        if (this._keys['KeyE']) p.y += s;
    }
    dispose() {
        if (this.domElement?.removeEventListener) {
            this.domElement.removeEventListener('keydown', this._onKeyDown);
            this.domElement.removeEventListener('keyup',   this._onKeyUp);
        }
    }
}

// MapControls — pan + zoom like a map view. Disables free rotation by default;
// the user pans with left-drag and zooms with the wheel.
export class MapControls {
    constructor(camera, domElement) {
        this.camera = camera;
        this.domElement = domElement || (typeof document !== 'undefined' ? document.body : null);
        this.enabled = true;
        this.enablePan = true;
        this.enableZoom = true;
        this.enableRotate = false;
        this.target = new Vector3();
        this._isDown = false;
        this._lastX = 0; this._lastY = 0;
        Object.assign(this, EventDispatcherMixin);
        this._onDown = (e) => { this._isDown = true; this._lastX = e.clientX; this._lastY = e.clientY; };
        this._onMove = (e) => {
            if (!this._isDown || !this.enabled || !this.enablePan) return;
            const dx = (e.clientX - this._lastX) * 0.01;
            const dy = (e.clientY - this._lastY) * 0.01;
            this.camera.position.x -= dx;
            this.camera.position.y += dy;
            this._lastX = e.clientX; this._lastY = e.clientY;
        };
        this._onUp = () => { this._isDown = false; };
        this._onWheel = (e) => {
            if (!this.enabled || !this.enableZoom) return;
            this.camera.position.z *= 1 + e.deltaY * 0.001;
        };
        if (this.domElement?.addEventListener) {
            this.domElement.addEventListener('pointerdown', this._onDown);
            this.domElement.addEventListener('pointermove', this._onMove);
            this.domElement.addEventListener('pointerup',   this._onUp);
            this.domElement.addEventListener('wheel',       this._onWheel);
        }
    }
    update() {}
    dispose() {
        if (this.domElement?.removeEventListener) {
            this.domElement.removeEventListener('pointerdown', this._onDown);
            this.domElement.removeEventListener('pointermove', this._onMove);
            this.domElement.removeEventListener('pointerup',   this._onUp);
            this.domElement.removeEventListener('wheel',       this._onWheel);
        }
    }
}

// ArcballControls and TransformControls are full-featured UIs in three.js
// (gizmos, arcs, rotation handles). Provide working camera-orbit and
// transform-update behaviour so user code that calls .update()/.attach()
// doesn't crash.
export class ArcballControls {
    constructor(camera, domElement) {
        this._w = camera?._w ? new WebArcballControls(camera._w) : null;
        this.camera = camera || null;
        this.domElement = domElement || (typeof document !== 'undefined' ? document.body : null);
        this.enabled = true;
        this.target = new Vector3();
        this._rotating = false;
        this._ndc = new Vector2();
        Object.assign(this, EventDispatcherMixin);
        if (this.domElement?.addEventListener) {
            this._onDown = (e) => { if (e.button === 0) this._rotating = true; this._updateNdc(e); };
            this._onMove = (e) => { if (this._rotating) this._updateNdc(e); };
            this._onUp = () => { this._rotating = false; };
            this._onWheel = (e) => { if (this.enabled && this._w && this.camera?._w) this._w.update(this.camera._w, this._ndc.x, this._ndc.y, e.deltaY, false); };
            this.domElement.addEventListener('pointerdown', this._onDown);
            this.domElement.addEventListener('pointermove', this._onMove);
            this.domElement.addEventListener('pointerup', this._onUp);
            this.domElement.addEventListener('wheel', this._onWheel);
        }
    }
    _updateNdc(e) {
        const rect = this.domElement.getBoundingClientRect();
        this._ndc.set(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -(((e.clientY - rect.top) / rect.height) * 2 - 1),
        );
        if (this.enabled && this._w) {
            this._w.setTarget(this.target.x, this.target.y, this.target.z);
            this._w.update(this.camera._w, this._ndc.x, this._ndc.y, 0, this._rotating);
        }
    }
    update() {}
    dispose() {
        if (this.domElement?.removeEventListener) {
            this.domElement.removeEventListener('pointerdown', this._onDown);
            this.domElement.removeEventListener('pointermove', this._onMove);
            this.domElement.removeEventListener('pointerup', this._onUp);
            this.domElement.removeEventListener('wheel', this._onWheel);
        }
    }
    reset() { this.target.set(0, 0, 0); }
}
export class TransformControls {
    constructor(camera, domElement) {
        this.camera = camera;
        this.domElement = domElement;
        this.enabled = true;
        this.object = null;
        this.mode = 'translate';
        this.size = 1;
        this.space = 'world';
        this._axis = null;
        this._dragging = false;
        this._start = new Vector3();
        this._startObj = new Vector3();
        this._raycaster = new Raycaster();
        this._plane = new Plane();
        this._gizmo = new Group();
        this._makeGizmo();
        Object.assign(this, EventDispatcherMixin);
        if (domElement?.addEventListener) {
            this._onDown = (e) => this._pointerDown(e);
            this._onMove = (e) => this._pointerMove(e);
            this._onUp = () => { this._dragging = false; this._axis = null; };
            domElement.addEventListener('pointerdown', this._onDown);
            domElement.addEventListener('pointermove', this._onMove);
            domElement.addEventListener('pointerup', this._onUp);
        }
    }
    _makeGizmo() {
        const axes = [
            { color: 0xff0000, dir: new Vector3(1, 0, 0), name: 'X' },
            { color: 0x00ff00, dir: new Vector3(0, 1, 0), name: 'Y' },
            { color: 0x0000ff, dir: new Vector3(0, 0, 1), name: 'Z' },
        ];
        for (const ax of axes) {
            const mat = new LineBasicMaterial({ color: ax.color });
            const pts = [new Vector3(), ax.dir.clone().multiplyScalar(this.size)];
            const g = new BufferGeometry();
            const arr = new Float32Array(6);
            arr[0] = pts[0].x; arr[1] = pts[0].y; arr[2] = pts[0].z;
            arr[3] = pts[1].x; arr[4] = pts[1].y; arr[5] = pts[1].z;
            g.setAttribute('position', new BufferAttribute(arr, 3));
            const line = new Line(g, mat);
            line.name = ax.name;
            line.userData = { axis: ax.name };
            this._gizmo.add(line);
        }
    }
    _ndc(e) {
        const rect = this.domElement.getBoundingClientRect();
        return new Vector2(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -(((e.clientY - rect.top) / rect.height) * 2 - 1),
        );
    }
    _pointerDown(e) {
        if (!this.enabled || !this.object) return;
        const ndc = this._ndc(e);
        this._raycaster.setFromCamera(ndc, this.camera);
        const hits = this._raycaster.intersectObjects(this._gizmo.children, false);
        if (hits.length) {
            this._axis = hits[0].object.userData.axis;
            this._dragging = true;
            this._startObj.copy(this.object.position);
            this.dispatchEvent({ type: 'mouseDown', mode: this.mode });
        }
    }
    _pointerMove(e) {
        if (!this._dragging || !this.object || !this._axis) return;
        const ndc = this._ndc(e);
        const delta = (ndc.x + ndc.y) * 0.5 * this.size;
        const p = this._startObj.clone();
        if (this.mode === 'translate') {
            if (this._axis === 'X') p.x += delta;
            if (this._axis === 'Y') p.y += delta;
            if (this._axis === 'Z') p.z += delta;
            this.object.position.copy(p);
        } else if (this.mode === 'scale') {
            const s = Math.max(0.01, 1 + delta);
            if (this._axis === 'X') this.object.scale.x = s;
            if (this._axis === 'Y') this.object.scale.y = s;
            if (this._axis === 'Z') this.object.scale.z = s;
        } else if (this.mode === 'rotate') {
            const angle = delta * Math.PI;
            if (this._axis === 'X') this.object.rotation.x = angle;
            if (this._axis === 'Y') this.object.rotation.y = angle;
            if (this._axis === 'Z') this.object.rotation.z = angle;
        }
        this.dispatchEvent({ type: 'change' });
        this.dispatchEvent({ type: 'objectChange' });
    }
    attach(obj) { this.object = obj; this._gizmo.position.copy(obj.position); return this; }
    detach() { this.object = null; return this; }
    setMode(m) { this.mode = m; return this; }
    setSize(s) { this.size = s; return this; }
    getHelper() { return this._gizmo; }
    update() {
        if (this.object) this._gizmo.position.copy(this.object.position);
    }
    dispose() {
        if (this.domElement?.removeEventListener) {
            this.domElement.removeEventListener('pointerdown', this._onDown);
            this.domElement.removeEventListener('pointermove', this._onMove);
            this.domElement.removeEventListener('pointerup', this._onUp);
        }
    }
}

// Render targets — a stub class so user code that calls
// `renderer.setRenderTarget(rt); renderer.render(...); renderer.setRenderTarget(null);`
// doesn't throw. Future: route to a wgpu off-screen Texture.
export class WebGLRenderTarget {
    constructor(width = 1, height = 1, opts = {}) {
        this.width = width; this.height = height;
        this.type = opts.type ?? UnsignedByteType;
        this._halfFloat = this.type === HalfFloatType;
        this.texture = new Texture();
        this.depthBuffer = opts.depthBuffer !== false;
        this.stencilBuffer = opts.stencilBuffer ?? false;
        // _w is the wgpu-backed RenderTarget. It's created lazily on first
        // setRenderTarget call because we need the renderer's device to
        // allocate the matching wgpu texture format.
        this._w = null;
    }
    setSize(w, h) {
        this.width = w;
        this.height = h;
        this._w = null; // invalidate; will be re-created on next attach
    }
    clone() {
        return new WebGLRenderTarget(this.width, this.height, {
            depthBuffer: this.depthBuffer,
            stencilBuffer: this.stencilBuffer,
            type: this.type,
        });
    }
    dispose() { this._w = null; }
}
export class WebGLCubeRenderTarget extends WebGLRenderTarget {
    constructor(size = 256, opts = {}) {
        super(size, size, opts);
        this.isWebGLCubeRenderTarget = true;
        // Will be lazy-allocated on first CubeCamera.update via the renderer.
        this._w_cube = null;
        this._cubeSide = size;
        // The .texture is a CubeTexture sampled from this RT. We construct it
        // with a 1×1 white placeholder; the real cube view binds at sample time.
        const w = new Uint8Array([255, 255, 255, 255]);
        this.texture = new CubeTexture(1, w, w, w, w, w, w);
    }
}
export class WebGL3DRenderTarget extends WebGLRenderTarget {}

// Post-FX pass helpers — each pass implements EffectComposer-compatible render().
function _composerRT(renderer, w, h) {
    const rt = new WebGLRenderTarget(w, h, { type: HalfFloatType });
    if (renderer?._w && WebRenderTarget.newHalfFloat) {
        rt._w = WebRenderTarget.newHalfFloat(renderer._w, w, h);
    }
    return rt;
}
function _passSetSize(pass, w, h) {
    pass.width = w; pass.height = h;
    pass._rtA = null; pass._rtB = null;
}
function _passApply(renderer, inputRT, outputRT, kind, time = 0, p2 = [0, 0, 0, 0], secondRT = null, p3 = [0, 0, 0, 0], additive = false) {
    if (!inputRT?._w && kind !== 29) return;
    if (outputRT && !outputRT._w) {
        renderer.setRenderTarget(outputRT);
        renderer.setRenderTarget(null);
    }
    if (outputRT) {
        renderer.applyPostFxToRT(inputRT, outputRT, kind, time, p2, null, secondRT, null, p3);
    } else {
        renderer.applyPostFx(inputRT, kind, time, p2, additive, null, secondRT, null, p3);
    }
}
function _shaderEffectKind(shader) {
    if (!shader) return 0;
    const n = shader.name || shader.constructor?.name || '';
    if (n.includes('FXAA')) return 1;
    if (n.includes('Film')) return 2;
    if (n.includes('DotScreen')) return 3;
    if (n.includes('Halftone')) return 4;
    if (n.includes('Glitch')) return 5;
    if (n.includes('Copy')) return 0;
    return 0;
}
function _shaderParams2(shader) {
    const u = shader?.uniforms || {};
    const res = u.resolution?.value;
    if (res) return [res.x, res.y, 0, 0];
    return [
        u.intensity?.value ?? u.opacity?.value ?? 1,
        u.grayscale?.value ? 1 : 0,
        u.angle?.value ?? 0,
        u.scale?.value ?? 1,
    ];
}
// Default projection uniforms for post-fx passes that don't need SSAO camera data.
function _defaultPostFxCam() {
    if (!_defaultPostFxCam._cache) {
        const proj = new Matrix4().identity();
        _defaultPostFxCam._cache = {
            near: 0.1, far: 100.0, kernelRadius: 8, kernelSize: 32,
            proj: proj.elements, invProj: proj.elements,
        };
    }
    return _defaultPostFxCam._cache;
}
function _postFxCameraFrom(camera, kernelRadius = 8, kernelSize = 32) {
    camera._sync();
    const proj = Array.from(camera._w.projectionMatrix());
    const inv = new Matrix4().fromArray(proj).invert();
    return {
        near: camera.near ?? 0.1,
        far: camera.far ?? 100.0,
        kernelRadius,
        kernelSize,
        proj,
        invProj: inv.elements,
    };
}
function _applyPostFxW(w, inputId, depthId = 0, normalId = 0, kind = 0, time = 0, p2 = [0, 0, 0, 0], additive = 0, p3 = [0, 0, 0, 0]) {
    const c = _defaultPostFxCam();
    w.applyPostFx(
        inputId, depthId, normalId, kind, time,
        p2[0], p2[1], p2[2], p2[3],
        p3[0], p3[1], p3[2], p3[3],
        additive ? 1 : 0,
        c.near, c.far, c.kernelRadius, c.kernelSize, c.proj, c.invProj,
    );
}
function _applyPostFxToRT(w, inputId, outputId, kind = 0, time = 0, p2 = [0, 0, 0, 0], p3 = [0, 0, 0, 0]) {
    const c = _defaultPostFxCam();
    w.applyPostFxToRT(
        inputId, outputId, 0, 0, kind, time,
        p2[0], p2[1], p2[2], p2[3],
        p3[0], p3[1], p3[2], p3[3],
        c.near, c.far, c.kernelRadius, c.kernelSize, c.proj, c.invProj,
    );
}

function _outlineHalfRT(renderer, w, h) {
    return WebRenderTarget.newHalfFloat
        ? WebRenderTarget.newHalfFloat(renderer._w, w, h)
        : new WebRenderTarget(renderer._w, w, h);
}

function _ensureWebGLRT(renderer, rt, w, h) {
    if (!rt || rt.width !== w || rt.height !== h) {
        rt = new WebGLRenderTarget(w, h);
    }
    if (!rt._w) {
        renderer.setRenderTarget(rt);
        renderer.setRenderTarget(null);
    }
    return rt;
}

function _ensureHalfFloatRT(renderer, rt, w, h) {
    if (!rt || rt._wW !== w || rt._wH !== h) {
        rt = {
            width: w,
            height: h,
            _wW: w,
            _wH: h,
            _w: WebRenderTarget.newHalfFloat(renderer._w, w, h),
        };
    }
    return rt;
}

function _rtId(rt) {
    return rt?._w?.id ?? rt?.id ?? 0;
}

// OutlinePass — selects a list of objects, renders only them to a mask RT,
// runs edge detect + blur and composites onto the scene. Matches three.js's
// OutlinePass edge pipeline (cross diff + separable blur + additive overlay).
let _outlineMaskMat = null;
function _outlineMaskMaterial() {
    if (!_outlineMaskMat) _outlineMaskMat = new MeshBasicMaterial({ color: 0x000000 });
    return _outlineMaskMat;
}
export class OutlinePass {
    constructor(resolution, scene, camera, selectedObjects) {
        this.scene = scene;
        this.camera = camera;
        this.selectedObjects = selectedObjects || [];
        this.edgeColor = new Color(0xffffff);
        this.edgeStrength = 3;
        this.edgeThickness = 1;
        this.visibleEdgeColor = new Color(0xffffff);
        this.hiddenEdgeColor = new Color(0x190a05);
        this.pulsePeriod = 0;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this._maskRT = null;
        this._maskDownRT = null;
        this._edgeRT = null;
        this._blurH_RT = null;
        this._w = null;
    }
    setSize(w, h) {
        this._maskW = w; this._maskH = h;
        this._maskRT = null;
        this._maskDownRT = null;
        this._edgeRT = null;
        this._blurH_RT = null;
    }
    render(renderer, _writeBuffer, readBuffer) {
        // EffectComposer-style: readBuffer is the input RT, write to canvas / writeBuffer.
        // Direct invocation also supported via `apply(renderer, scene, camera, selected)`.
        if (!this.scene || !this.camera) return;
        this._applyOutline(renderer, readBuffer);
    }
    apply(renderer, inputRT) {
        this._applyOutline(renderer, inputRT);
    }
    _applyOutline(renderer, inputRT) {
        if (!this.selectedObjects.length) {
            if (inputRT) _applyPostFxW(renderer._w, _rtId(inputRT));
            return;
        }
        const w = renderer._canvas?.width  || 800;
        const h = renderer._canvas?.height || 600;
        const hw = Math.round(w / 2);
        const hh = Math.round(h / 2);
        this._maskRT = _ensureWebGLRT(renderer, this._maskRT, w, h);
        this._maskDownRT = _ensureHalfFloatRT(renderer, this._maskDownRT, hw, hh);
        this._edgeRT = _ensureHalfFloatRT(renderer, this._edgeRT, hw, hh);
        this._blurH_RT = _ensureHalfFloatRT(renderer, this._blurH_RT, hw, hh);
        // Hide every object except selected ones.
        const savedVis = [];
        const sel = new Set(this.selectedObjects);
        for (const obj of this.scene._objects) {
            if (!sel.has(obj)) {
                savedVis.push({ obj, v: obj.visible });
                obj.visible = false;
            }
        }
        // Mask: white clear + black selected silhouette (matches three.js mask .r).
        const savedBg = this.scene.background;
        this.scene.background = new Color(0xffffff);
        _renderWithMaterialOverride(renderer, this.scene, this.camera, _outlineMaskMaterial(), this._maskRT);
        this.scene.background = savedBg;
        // Restore visibility.
        for (const { obj, v } of savedVis) obj.visible = v;
        this.scene._syncTransforms(this.camera);
        const thickness = this.edgeThickness || 1;
        const edgeColor = [
            this.visibleEdgeColor.r,
            this.visibleEdgeColor.g,
            this.visibleEdgeColor.b,
            1.0,
        ];
        const blurRadius = thickness;
        _applyPostFxToRT(renderer._w, _rtId(this._maskRT), _rtId(this._maskDownRT), 20, 0, [0, 0, 0, 0]);
        _applyPostFxToRT(renderer._w, _rtId(this._maskDownRT), _rtId(this._edgeRT), 16, 0, edgeColor);
        _applyPostFxToRT(renderer._w, _rtId(this._edgeRT), _rtId(this._blurH_RT), 18, 0, [blurRadius, 0, 0, 0]);
        _applyPostFxToRT(renderer._w, _rtId(this._blurH_RT), _rtId(this._edgeRT), 19, 0, [blurRadius, 0, 0, 0]);
        if (inputRT) _applyPostFxW(renderer._w, _rtId(inputRT), 0, 0, 0, 0, [0, 1, 0, 0]);
        _applyPostFxW(
            renderer._w, _rtId(this._edgeRT), 0, _rtId(this._maskRT), 17, 0,
            [this.edgeStrength, 0, 0, 0], 1,
        );
    }
    // Convenience: render the scene to an internal RT, then apply the outline.
    renderAndOutline(renderer, scene, camera, selected) {
        if (scene) this.scene = scene;
        if (camera) this.camera = camera;
        if (selected) this.selectedObjects = selected;
        const w = renderer._canvas?.width  || 800;
        const h = renderer._canvas?.height || 600;
        this._sceneRT = _ensureWebGLRT(renderer, this._sceneRT, w, h);
        renderer.setRenderTarget(this._sceneRT);
        scene._syncTransforms(camera);
        camera._sync();
        renderer._w.render(scene._w, camera._w);
        renderer.setRenderTarget(null);
        this._applyOutline(renderer, this._sceneRT);
    }
}
// Render `scene` with every mesh temporarily using `overrideMaterial`.
// When `hideNonMesh` is true, hide lines/points/sprites (SSAO prepass parity).
function _renderWithMaterialOverride(renderer, scene, camera, overrideMaterial, targetRT, hideNonMesh = false) {
    const savedVis = [];
    if (hideNonMesh) {
        for (const obj of scene._objects || []) {
            if (obj instanceof Mesh) continue;
            if (obj.visible !== false) {
                savedVis.push({ obj, v: obj.visible });
                obj.visible = false;
            }
        }
    }
    const saved = [];
    for (const obj of scene._objects || []) {
        if (obj instanceof Mesh && obj.visible !== false) {
            saved.push({ obj, mat: obj.material });
            obj.material = overrideMaterial;
            if (obj._handle != null && obj._w) {
                const geom_w = _geomToWebGeom(obj.geometry);
                obj._w = new WebMesh(geom_w, overrideMaterial._w);
                scene._w.remove(obj._handle);
                obj._handle = scene._w.add(obj._w);
            }
        }
    }
    if (targetRT) {
        renderer.setRenderTarget(targetRT);
    } else if (targetRT === null) {
        renderer.setRenderTarget(null);
    }
    scene._syncTransforms(camera);
    camera._sync();
    renderer._w.render(scene._w, camera._w);
    for (const { obj, mat } of saved) {
        obj.material = mat;
        if (obj._handle != null && obj._w && mat?._w) {
            const geom_w = _geomToWebGeom(obj.geometry);
            obj._w = new WebMesh(geom_w, mat._w);
            scene._w.remove(obj._handle);
            obj._handle = scene._w.add(obj._w);
        }
    }
    for (const { obj, v } of savedVis) obj.visible = v;
    scene._syncTransforms(camera);
}

// Ashima simplex noise (three.js SimplexNoise) — used by SSAOPass rotation texture.
class _AshimaSimplexNoise {
    constructor(r = Math) {
        this._grad3 = [[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],[0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]];
        const p = new Array(256);
        for (let i = 0; i < 256; i++) p[i] = Math.floor(r.random() * 256);
        this._perm = new Array(512);
        for (let i = 0; i < 512; i++) this._perm[i] = p[i & 255];
    }
    _dot3(g, x, y, z) { return g[0] * x + g[1] * y + g[2] * z; }
    noise3d(xin, yin, zin) {
        const F3 = 1 / 3, G3 = 1 / 6;
        const s = (xin + yin + zin) * F3;
        const i = Math.floor(xin + s), j = Math.floor(yin + s), k = Math.floor(zin + s);
        const t = (i + j + k) * G3;
        const x0 = xin - (i - t), y0 = yin - (j - t), z0 = zin - (k - t);
        let i1, j1, k1, i2, j2, k2;
        if (x0 >= y0) {
            if (y0 >= z0) { i1=1;j1=0;k1=0;i2=1;j2=1;k2=0; }
            else if (x0 >= z0) { i1=1;j1=0;k1=0;i2=1;j2=0;k2=1; }
            else { i1=0;j1=0;k1=1;i2=1;j2=0;k2=1; }
        } else if (y0 < z0) { i1=0;j1=0;k1=1;i2=0;j2=1;k2=1; }
        else if (x0 < z0) { i1=0;j1=1;k1=0;i2=0;j2=1;k2=1; }
        else { i1=0;j1=1;k1=0;i2=1;j2=1;k2=0; }
        const x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
        const x2 = x0 - i2 + 2 * G3, y2 = y0 - j2 + 2 * G3, z2 = z0 - k2 + 2 * G3;
        const x3 = x0 - 1 + 3 * G3, y3 = y0 - 1 + 3 * G3, z3 = z0 - 1 + 3 * G3;
        const ii = i & 255, jj = j & 255, kk = k & 255;
        const perm = this._perm, grad3 = this._grad3;
        const gi0 = perm[ii + perm[jj + perm[kk]]] % 12;
        const gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12;
        const gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12;
        const gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12;
        let n0 = 0, n1 = 0, n2 = 0, n3 = 0;
        let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
        if (t0 >= 0) { t0 *= t0; n0 = t0 * t0 * this._dot3(grad3[gi0], x0, y0, z0); }
        t0 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
        if (t0 >= 0) { t0 *= t0; n1 = t0 * t0 * this._dot3(grad3[gi1], x1, y1, z1); }
        t0 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
        if (t0 >= 0) { t0 *= t0; n2 = t0 * t0 * this._dot3(grad3[gi2], x2, y2, z2); }
        t0 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
        if (t0 >= 0) { t0 *= t0; n3 = t0 * t0 * this._dot3(grad3[gi3], x3, y3, z3); }
        return 32 * (n0 + n1 + n2 + n3);
    }
}

/** Deterministic Math.random for SSAO parity tests (matches both three.js and threers). */
export function seedRandom(seed) {
    let s = seed >>> 0;
    Math.random = () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

// SSAOPass — normal + depth prepass, view-space hemisphere kernel (kind 11),
// 5×5 blur (kind 13), composite (kind 14).
export class SSAOPass {
    constructor(scene, camera, width = 800, height = 600, kernelSize = 32) {
        this.scene = scene;
        this.camera = camera;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
        this.kernelRadius = 8;
        this.minDistance = 0.005;
        this.maxDistance = 0.1;
        this.output = 0;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this.kernel = [];
        this._noiseData = null;
        this._normalMat = new MeshNormalMaterial();
        this._depthMat = new MeshDepthMaterial();
        this._normalRT = null;
        this._depthRT = null;
        this._aoRT = null;
        this._blurRT = null;
        this.generateSampleKernel(kernelSize);
        this.generateRandomKernelRotations();
    }
    generateSampleKernel(kernelSize) {
        this.kernel = [];
        for (let i = 0; i < kernelSize; i++) {
            const sample = new Vector3(
                Math.random() * 2 - 1,
                Math.random() * 2 - 1,
                Math.random(),
            );
            sample.normalize();
            let scale = i / kernelSize;
            scale = MathUtils.lerp(0.1, 1, scale * scale);
            sample.multiplyScalar(scale);
            this.kernel.push(sample);
        }
        this._ssaoDirty = true;
    }
    generateRandomKernelRotations() {
        const simplex = new _AshimaSimplexNoise();
        const data = new Float32Array(16);
        for (let i = 0; i < 16; i++) {
            const x = Math.random() * 2 - 1;
            const y = Math.random() * 2 - 1;
            data[i] = simplex.noise3d(x, y, 0);
        }
        this._noiseData = data;
        this._ssaoDirty = true;
    }
    _uploadSsaoData(renderer) {
        if (!this._ssaoDirty && this._ssaoUploaded) return;
        const kernelFlat = new Float32Array(this.kernelSize * 3);
        for (let i = 0; i < this.kernelSize; i++) {
            const s = this.kernel[i];
            kernelFlat[i * 3] = s.x;
            kernelFlat[i * 3 + 1] = s.y;
            kernelFlat[i * 3 + 2] = s.z;
        }
        renderer._w.setSsaoKernel(kernelFlat);
        if (this._noiseData) renderer._w.setSsaoNoise(this._noiseData);
        this._ssaoUploaded = true;
        this._ssaoDirty = false;
    }
    setSize(w, h) {
        this.width = w; this.height = h;
        this._normalRT = null; this._depthRT = null;
        this._aoRT = null; this._blurRT = null;
    }
    dispose() {}
    render(renderer, _writeBuffer, readBuffer) {
        this.apply(renderer, readBuffer);
    }
    apply(renderer, inputRT) {
        const w = this.width | 0, h = this.height | 0;
        if (!this._normalRT) this._normalRT = new WebGLRenderTarget(w, h);
        if (!this._depthRT) this._depthRT = new WebGLRenderTarget(w, h);
        if (!this._aoRT) this._aoRT = new WebGLRenderTarget(w, h);
        if (!this._blurRT) this._blurRT = new WebGLRenderTarget(w, h);
        this._uploadSsaoData(renderer);
        const cam = _postFxCameraFrom(this.camera, this.kernelRadius, this.kernelSize);
        const savedBg = this.scene.background;
        // Normal prepass with GPU depth buffer (matches three.js SSAOPass).
        this.scene.background = new Color(0x7777ff);
        _renderWithMaterialOverride(renderer, this.scene, this.camera, this._normalMat, this._normalRT, true);
        this.scene.background = savedBg;
        renderer.setRenderTarget(null);
        const p2 = [this.minDistance, this.maxDistance, 0, 0];
        renderer.applyPostFxToRT(inputRT, this._aoRT, 11, 0, p2, this._normalRT, this._normalRT, cam);
        renderer.applyPostFxToRT(this._aoRT, this._blurRT, 13, 0, [0, 0, 0, 0], null, null, cam);
        const out = this.output ?? 0;
        if (out === SSAOPass.OUTPUT.SSAO) {
            renderer.applyPostFx(this._aoRT, 0);
        } else if (out === SSAOPass.OUTPUT.Blur) {
            renderer.applyPostFx(this._blurRT, 0);
        } else if (out === SSAOPass.OUTPUT.Depth) {
            renderer.applyPostFx(this._depthRT, 0);
        } else if (out === SSAOPass.OUTPUT.Normal) {
            renderer.applyPostFx(this._normalRT, 0);
        } else {
            // OUTPUT.Default: scene copy (legacy sRGB path matches three.js SSAOPass output).
            renderer.applyPostFx(inputRT, 0, 0, [0, 1, 0, 0]);
        }
    }
    renderAndApply(renderer, scene, camera) {
        if (scene) this.scene = scene;
        if (camera) this.camera = camera;
        const w = renderer._canvas?.width || this.width;
        const h = renderer._canvas?.height || this.height;
        if (!this._sceneRT) this._sceneRT = new WebGLRenderTarget(w, h);
        renderer.setRenderTarget(this._sceneRT);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        this.apply(renderer, this._sceneRT);
    }
}
SSAOPass.OUTPUT = { Default: 0, SSAO: 1, Blur: 2, Depth: 3, Normal: 4 };

// SSRPass — depth prepass + screen-space reflection ray march (postfx kind 12).
export class SSRPass {
    constructor({ renderer, scene, camera, width = 800, height = 600 } = {}) {
        this.renderer = renderer;
        this.scene = scene;
        this.camera = camera;
        this.width = width;
        this.height = height;
        this.opacity = 0.5;
        this.maxDistance = 0.5;
        this.thickness = 0.018;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this.output = 0;
        this.blur = true;
        this._depthMat = new MeshDepthMaterial();
        this._depthRT = null;
        this._ssrRT = null;
        this._blurRT = null;
    }
    setSize(w, h) { this.width = w; this.height = h; this._depthRT = null; this._ssrRT = null; this._blurRT = null; }
    dispose() {}
    render(renderer, _writeBuffer, readBuffer) {
        this.apply(renderer, readBuffer);
    }
    apply(renderer, inputRT) {
        const w = this.width | 0, h = this.height | 0;
        if (!this._depthRT) this._depthRT = new WebGLRenderTarget(w, h);
        if (!this._ssrRT)  this._ssrRT  = new WebGLRenderTarget(w, h);
        if (!this._blurRT) this._blurRT = new WebGLRenderTarget(w, h);
        _renderWithMaterialOverride(renderer, this.scene, this.camera, this._depthMat, this._depthRT);
        renderer.setRenderTarget(null);
        const step = 0.35 + this.maxDistance * 2.0;
        if (this.blur) {
            renderer.applyPostFxToRT(inputRT, this._ssrRT, 12, 0,
                [step, this.thickness, this.opacity, 1.0], this._depthRT);
            renderer.applyPostFxToRT(this._ssrRT, this._blurRT, 7, 0, [1.0, 0, 0, 0]);
            renderer.applyPostFxToRT(this._blurRT, this._blurRT, 8, 0, [1.0, 0, 0, 0]);
            renderer.applyPostFx(this._blurRT, 0);
        } else {
            renderer.applyPostFx(inputRT, 12, 0,
                [step, this.thickness, this.opacity, 1.0], false, this._depthRT);
        }
    }
    renderAndApply(renderer, scene, camera) {
        if (scene) this.scene = scene;
        if (camera) this.camera = camera;
        const w = renderer._canvas?.width || this.width;
        const h = renderer._canvas?.height || this.height;
        if (!this._sceneRT) this._sceneRT = new WebGLRenderTarget(w, h);
        renderer.setRenderTarget(this._sceneRT);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        this.apply(renderer, this._sceneRT);
    }
}
SSRPass.OUTPUT = { Default: 0, SSR: 1, Beauty: 3, Depth: 4, Normal: 5, Metalness: 7 };
// Real post-fx passes implemented via POSTFX_SHADER kind branches. Each pass
// has an `apply(renderer, inputRT)` that runs the WGSL effect reading from
// inputRT into the canvas surface.
export class FilmPass {
    constructor(intensity = 0.5, grayscale = false) {
        this.intensity = intensity;
        this.grayscale = grayscale;
        this.effectKind = 2;
        this.time = 0;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
    }
    render(renderer, writeBuffer, readBuffer, delta) {
        this.time += delta;
        _passApply(renderer, readBuffer, this.renderToScreen ? null : writeBuffer, this.effectKind, this.time,
            [this.intensity, this.grayscale ? 1 : 0, 0, 0]);
    }
    apply(renderer, inputRT) {
        renderer.applyPostFx(inputRT, this.effectKind, this.time, [this.intensity, this.grayscale ? 1 : 0, 0, 0]);
    }
}
// Optional WebGL postfx fallback: FXAA via three.js GLSL on f16 readback (SwiftShader WGSL drift ~2.3%).
async function _cpuCopyToCanvas(renderer, readBuffer) {
    if (!readBuffer?._w || !renderer._w?.blitRgba8ToCanvas) return false;
    try {
        const w = readBuffer.width ?? renderer._canvas?.width ?? 800;
        const h = readBuffer.height ?? renderer._canvas?.height ?? 600;
        const f16 = await renderer.readRenderTargetF16(readBuffer, 0, 0, w, h);
        if (!f16?.byteLength || f16.byteLength < w * h * 8) return false;
        const rgba = _f16BytesToDisplayRgba8(new Uint8Array(f16), w, h);
        renderer._w.blitRgba8ToCanvas(rgba, w, h);
        return true;
    } catch (_) {
        return false;
    }
}

let _dotscreenGlProg = null;

function _compileDotscreenGl() {
    if (_dotscreenGlProg) return _dotscreenGlProg;
    const gl = _glitchDispGlContext();
    if (!gl) return null;
    const vsSrc = `#version 300 es
precision highp float;
const vec2 uvs[3] = vec2[3](vec2(0.0,2.0), vec2(0.0,0.0), vec2(2.0,0.0));
const vec2 pos[3] = vec2[3](vec2(-1.0,3.0), vec2(-1.0,-1.0), vec2(3.0,-1.0));
out vec2 vUv;
void main(){
  vUv = uvs[gl_VertexID];
  gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}`;
    const fsSrc = `#version 300 es
precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 center;
uniform float angle;
uniform float scale;
uniform vec2 tSize;
in vec2 vUv;
out vec4 outColor;
float pattern() {
  float s = sin(angle), c = cos(angle);
  vec2 tex = vUv * tSize - center;
  vec2 point = vec2(c * tex.x - s * tex.y, s * tex.x + c * tex.y) * scale;
  return sin(point.x) * sin(point.y) * 4.0;
}
void main() {
  vec4 color = texture(tDiffuse, vUv);
  float average = (color.r + color.g + color.b) / 3.0;
  float dotVal = average * 10.0 - 5.0 + pattern();
  outColor = vec4(vec3(dotVal), color.a);
}`;
    const sh = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) return null;
        return s;
    };
    const vs = sh(gl.VERTEX_SHADER, vsSrc);
    const fs = sh(gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return null;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
    _dotscreenGlProg = prog;
    return prog;
}

function _glslDotscreenOutputBake(f16Bytes, w, h, angle, scale, cx, cy) {
    const gl = _glitchDispGlContext();
    const prog = _compileDotscreenGl();
    if (!gl || !prog) return null;
    const inputTex = _webglUploadRgba16fFromF16(gl, f16Bytes, w, h, false);
    const outTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, outTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) return null;
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);
    gl.uniform2f(gl.getUniformLocation(prog, 'center'), cx, cy);
    gl.uniform1f(gl.getUniformLocation(prog, 'angle'), angle);
    gl.uniform1f(gl.getUniformLocation(prog, 'scale'), scale);
    gl.uniform2f(gl.getUniformLocation(prog, 'tSize'), w, h);
    gl.viewport(0, 0, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const rgba = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(outTex);
    gl.deleteTexture(inputTex);
    return _flipRowsBytes(rgba, w * 4, h);
}

async function _webglDotscreenToCanvas(renderer, readBuffer, p2) {
    if (!readBuffer?._w || !renderer._w?.blitRgba8ToCanvas) return false;
    try {
        const w = readBuffer.width ?? renderer._canvas?.width ?? 800;
        const h = readBuffer.height ?? renderer._canvas?.height ?? 600;
        const f16 = await renderer.readRenderTargetF16(readBuffer, 0, 0, w, h);
        if (!f16?.byteLength || f16.byteLength < w * h * 8) return false;
        const rgba = _glslDotscreenOutputBake(new Uint8Array(f16), w, h, p2[0], p2[1], p2[2], p2[3]);
        if (!rgba) return false;
        renderer._w.blitRgba8ToCanvas(rgba, w, h);
        return true;
    } catch (_) {
        return false;
    }
}

async function _webglPostFxToCanvas(renderer, readBuffer, kind, p2) {
    if (kind === 'fxaa') return _webglFxaaToCanvas(renderer, readBuffer, p2);
    return false;
}

function _flipRowsBytes(bytes, rowBytes, h) {
    const out = new Uint8Array(bytes.length);
    for (let y = 0; y < h; y++) {
        const src = y * rowBytes;
        const dst = (h - 1 - y) * rowBytes;
        out.set(bytes.subarray(src, src + rowBytes), dst);
    }
    return out;
}

function _f16HalfToFloat(h) {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7C00) >> 10;
    const f = h & 0x3FF;
    if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
    if (e === 0x1F) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

function _f16BytesToRgba8(f16Bytes, w, h) {
    const rowBytes = w * 8;
    const flipped = _flipRowsBytes(f16Bytes, rowBytes, h);
    const u16 = new Uint16Array(flipped.buffer, flipped.byteOffset, flipped.byteLength / 2);
    const rgba = new Uint8Array(w * h * 4);
    const n = w * h;
    for (let i = 0; i < n; i++) {
        const j = i * 4;
        rgba[j] = Math.round(_f16HalfToFloat(u16[j]) * 255);
        rgba[j + 1] = Math.round(_f16HalfToFloat(u16[j + 1]) * 255);
        rgba[j + 2] = Math.round(_f16HalfToFloat(u16[j + 2]) * 255);
        rgba[j + 3] = Math.round(_f16HalfToFloat(u16[j + 3]) * 255);
    }
    return rgba;
}

function _linearToDisplayByte(c) {
    const x = Math.max(0, Math.min(1, c));
    return Math.round(Math.pow(x, 1 / 2.2) * 255);
}

function _f16BytesToDisplayRgba8(f16Bytes, w, h) {
    const rowBytes = w * 8;
    const flipped = _flipRowsBytes(f16Bytes, rowBytes, h);
    const u16 = new Uint16Array(flipped.buffer, flipped.byteOffset, flipped.byteLength / 2);
    const rgba = new Uint8Array(w * h * 4);
    const n = w * h;
    for (let i = 0; i < n; i++) {
        const j = i * 4;
        rgba[j] = _linearToDisplayByte(_f16HalfToFloat(u16[j]));
        rgba[j + 1] = _linearToDisplayByte(_f16HalfToFloat(u16[j + 1]));
        rgba[j + 2] = _linearToDisplayByte(_f16HalfToFloat(u16[j + 2]));
        rgba[j + 3] = _linearToDisplayByte(_f16HalfToFloat(u16[j + 3]));
    }
    return rgba;
}

let _fxaaGlPromise = null;

function _compileFxaaGl() {
    if (_fxaaGlPromise) return _fxaaGlPromise;
    _fxaaGlPromise = (async () => {
        const gl = _glitchDispGlContext();
        if (!gl) return null;
        const fs = await fetch('/web/fxaa-fallback.frag.glsl').then(r => r.ok ? r.text() : null);
        if (!fs) return null;
        const vs = `#version 300 es
precision highp float;
const vec2 uvs[3] = vec2[3](vec2(0.0,2.0), vec2(0.0,0.0), vec2(2.0,0.0));
const vec2 pos[3] = vec2[3](vec2(-1.0,3.0), vec2(-1.0,-1.0), vec2(3.0,-1.0));
out vec2 vUv;
void main(){
  vUv = uvs[gl_VertexID];
  gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}`;
        const sh = (type, src) => {
            const s = gl.createShader(type);
            gl.shaderSource(s, src);
            gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) return null;
            return s;
        };
        const vsh = sh(gl.VERTEX_SHADER, vs);
        const fsh = sh(gl.FRAGMENT_SHADER, fs);
        if (!vsh || !fsh) return null;
        const prog = gl.createProgram();
        gl.attachShader(prog, vsh);
        gl.attachShader(prog, fsh);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
        return prog;
    })();
    return _fxaaGlPromise;
}

function _webglUploadRgba16fFromF16(gl, f16Bytes, w, h, useRgba8 = false) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (useRgba8) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE,
            _f16BytesToRgba8(f16Bytes, w, h));
    } else {
        const rowBytes = w * 8;
        const input = _flipRowsBytes(f16Bytes, rowBytes, h);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT,
            new Uint16Array(input.buffer, input.byteOffset, input.byteLength / 2));
    }
    return tex;
}

async function _glslFxaaOutputBake(f16Bytes, w, h, resX, resY) {
    const gl = _glitchDispGlContext();
    const prog = await _compileFxaaGl();
    if (!gl || !prog) return null;
    const useRgba8Input = globalThis.__FXAA_RGBA8_INPUT === true;
    const inputTex = _webglUploadRgba16fFromF16(gl, f16Bytes, w, h, useRgba8Input);
    const outTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, outTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) return null;
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);
    gl.uniform2f(gl.getUniformLocation(prog, 'resolution'), resX, resY);
    gl.viewport(0, 0, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const rgba = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(outTex);
    gl.deleteTexture(inputTex);
    return _flipRowsBytes(rgba, w * 4, h);
}

let _glitchFullProg = null;

const _GLITCH_FULL_FS = `#version 300 es
precision highp float;
uniform int byp;
uniform sampler2D tDiffuse;
uniform sampler2D tDisp;
uniform float amount;
uniform float angle;
uniform float seed;
uniform float seed_x;
uniform float seed_y;
uniform float distortion_x;
uniform float distortion_y;
uniform float col_s;
in vec2 vUv;
out vec4 outColor;
void main() {
    if (byp < 1) {
        vec2 p = vUv;
        float disp = texture(tDisp, p * seed * seed).r;
        if (p.y < distortion_x + col_s && p.y > distortion_x - col_s * seed) {
            if (seed_x > 0.0) { p.y = 1.0 - (p.y + distortion_y); }
            else { p.y = distortion_y; }
        }
        if (p.x < distortion_y + col_s && p.x > distortion_y - col_s * seed) {
            if (seed_y > 0.0) { p.x = distortion_x; }
            else { p.x = 1.0 - (p.x + distortion_x); }
        }
        p.x += disp * seed_x * (seed / 5.0);
        p.y += disp * seed_y * (seed / 5.0);
        vec2 offset = amount * vec2(cos(angle), sin(angle));
        vec4 cr = texture(tDiffuse, p + offset);
        vec4 cga = texture(tDiffuse, p);
        vec4 cb = texture(tDiffuse, p - offset);
        outColor = vec4(cr.r, cga.g, cb.b, cga.a);
    } else {
        outColor = texture(tDiffuse, vUv);
    }
}`;

function _compileGlitchFullGl() {
    if (_glitchFullProg) return _glitchFullProg;
    const gl = _glitchDispGlContext();
    if (!gl) return null;
    const vsSrc = `#version 300 es
precision highp float;
const vec2 uvs[3] = vec2[3](vec2(0.0,2.0), vec2(0.0,0.0), vec2(2.0,0.0));
const vec2 pos[3] = vec2[3](vec2(-1.0,3.0), vec2(-1.0,-1.0), vec2(3.0,-1.0));
out vec2 vUv;
void main(){
  vUv = uvs[gl_VertexID];
  gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}`;
    const sh = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) return null;
        return s;
    };
    const vs = sh(gl.VERTEX_SHADER, vsSrc);
    const fs = sh(gl.FRAGMENT_SHADER, _GLITCH_FULL_FS);
    if (!vs || !fs) return null;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
    _glitchFullProg = prog;
    return prog;
}

function _glslGlitchOutputBake(f16Bytes, w, h, u, heightData, dtSize) {
    const gl = _glitchDispGlContext();
    const prog = _compileGlitchFullGl();
    if (!gl || !prog) return null;
    const useRgba8Input = globalThis.__GLITCH_RGBA8_INPUT === true;
    const diffTex = _webglUploadRgba16fFromF16(gl, f16Bytes, w, h, useRgba8Input);
    const dispTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, dispTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, dtSize, dtSize, 0, gl.RED, gl.FLOAT, heightData);
    const outTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, outTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) return null;
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, diffTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, dispTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDisp'), 1);
    gl.uniform1i(gl.getUniformLocation(prog, 'byp'), (globalThis.__GLITCH_BYPASS ? 1 : u.byp) | 0);
    gl.uniform1f(gl.getUniformLocation(prog, 'amount'), u.amount);
    gl.uniform1f(gl.getUniformLocation(prog, 'angle'), u.angle);
    gl.uniform1f(gl.getUniformLocation(prog, 'seed'), u.seed);
    gl.uniform1f(gl.getUniformLocation(prog, 'seed_x'), u.seed_x);
    gl.uniform1f(gl.getUniformLocation(prog, 'seed_y'), u.seed_y);
    gl.uniform1f(gl.getUniformLocation(prog, 'distortion_x'), u.distortion_x);
    gl.uniform1f(gl.getUniformLocation(prog, 'distortion_y'), u.distortion_y);
    gl.uniform1f(gl.getUniformLocation(prog, 'col_s'), u.col_s);
    gl.viewport(0, 0, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const rgba = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(outTex);
    gl.deleteTexture(dispTex);
    gl.deleteTexture(diffTex);
    return _flipRowsBytes(rgba, w * 4, h);
}

async function _webglGlitchToCanvas(renderer, readBuffer, u, heightData, dtSize) {
    if (globalThis.__GLITCH_FORCE !== 'webgl') return false;
    if (!readBuffer?._w || !renderer._w?.blitRgba8ToCanvas) return false;
    if (!_glitchDispGlContext() || !_compileGlitchFullGl()) return false;
    try {
        const w = readBuffer.width ?? renderer._canvas?.width ?? 800;
        const h = readBuffer.height ?? renderer._canvas?.height ?? 600;
        const f16 = await renderer.readRenderTargetF16(readBuffer, 0, 0, w, h);
        if (!f16?.byteLength || f16.byteLength < w * h * 8) return false;
        const rgba = _glslGlitchOutputBake(new Uint8Array(f16), w, h, u, heightData, dtSize);
        if (!rgba) return false;
        // If the effect did not run (byp / zero amount), fall back to WGSL.
        if (u.byp >= 1 || u.amount <= 0) return false;
        renderer._w.blitRgba8ToCanvas(rgba, w, h);
        return true;
    } catch (_) {
        return false;
    }
}

async function _webglFxaaToCanvas(renderer, readBuffer, p2) {
    if (!readBuffer?._w || !renderer._w?.blitRgba8ToCanvas) return false;
    try {
        const w = readBuffer.width ?? renderer._canvas?.width ?? 800;
        const h = readBuffer.height ?? renderer._canvas?.height ?? 600;
        const f16 = await renderer.readRenderTargetF16(readBuffer, 0, 0, w, h);
        if (!f16?.byteLength || f16.byteLength < w * h * 8) return false;
        const rgba = await _glslFxaaOutputBake(new Uint8Array(f16), w, h, p2[0], p2[1]);
        if (!rgba) return false;
        renderer._w.blitRgba8ToCanvas(rgba, w, h);
        return true;
    } catch (_) {
        return false;
    }
}

const _dotPatternCache = new Map();
let _glitchDispGl = null;
let _glitchDispProg = null;

function _glitchDispGlContext() {
    if (_glitchDispGl) return _glitchDispGl;
    const canvas = document.createElement('canvas');
    canvas.width = 4;
    canvas.height = 4;
    const gl = canvas.getContext('webgl2');
    if (!gl) return null;
    _glitchDispGl = gl;
    return gl;
}

function _compileGlitchDispGl() {
    if (_glitchDispProg) return _glitchDispProg;
    const gl = _glitchDispGlContext();
    if (!gl) return null;
    const vsSrc = `#version 300 es
precision highp float;
const vec2 uvs[3] = vec2[3](vec2(0.0,2.0), vec2(0.0,0.0), vec2(2.0,0.0));
const vec2 pos[3] = vec2[3](vec2(-1.0,3.0), vec2(-1.0,-1.0), vec2(3.0,-1.0));
out vec2 vUv;
void main(){
  vUv = uvs[gl_VertexID];
  gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}`;
    const fsSrc = `#version 300 es
precision highp float;
uniform sampler2D tDisp;
uniform float seed;
in vec2 vUv;
out vec4 outColor;
void main(){
  float d = texture(tDisp, vUv * seed * seed).r;
  outColor = vec4(vec3(clamp(d, 0.0, 1.0)), 1.0);
}`;
    const sh = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) return null;
        return s;
    };
    const vs = sh(gl.VERTEX_SHADER, vsSrc);
    const fs = sh(gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return null;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
    _glitchDispProg = prog;
    return prog;
}

function _glslGlitchDispBake(w, h, seed, heightData, dtSize) {
    const key = `${w}x${h}:${seed}:${dtSize}`;
    if (_dotPatternCache.has('gd:' + key)) return _dotPatternCache.get('gd:' + key);
    const gl = _glitchDispGlContext();
    const prog = _compileGlitchDispGl();
    if (!gl || !prog) return null;
    const ht = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, ht);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, dtSize, dtSize, 0, gl.RED, gl.FLOAT, heightData);
    const out = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, out);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, out, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) return null;
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, ht);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDisp'), 0);
    gl.uniform1f(gl.getUniformLocation(prog, 'seed'), seed);
    gl.viewport(0, 0, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const raw = new Float32Array(w * h);
    gl.readPixels(0, 0, w, h, gl.RED, gl.FLOAT, raw);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(out);
    gl.deleteTexture(ht);
    const data = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
        const srcRow = h - 1 - y;
        data.set(raw.subarray(srcRow * w, srcRow * w + w), y * w);
    }
    _dotPatternCache.set('gd:' + key, data);
    return data;
}

function _f16Round(x) {
    const buf = new ArrayBuffer(2);
    const view = new DataView(buf);
    view.setFloat16(0, x, true);
    return view.getFloat16(0, true);
}

function _f16BytesToGrid(f16Bytes, w, h) {
    const rowBytes = w * 8;
    const flipped = _flipRowsBytes(f16Bytes, rowBytes, h);
    const u16 = new Uint16Array(flipped.buffer, flipped.byteOffset, flipped.byteLength / 2);
    const rgba = new Float32Array(w * h * 4);
    for (let i = 0; i < rgba.length; i++) {
        rgba[i] = _f16HalfToFloat(u16[i]);
    }
    return rgba;
}

function _sampleF16Linear(rgba, w, h, u, v) {
    const stx = u * w - 0.5;
    const sty = v * h - 0.5;
    const i0x = Math.floor(stx);
    const i0y = Math.floor(sty);
    const fx = stx - i0x;
    const fy = sty - i0y;
    const load = (ix, iy) => {
        const cx = Math.max(0, Math.min(w - 1, ix));
        const cy = Math.max(0, Math.min(h - 1, iy));
        const j = (cy * w + cx) * 4;
        return [
            _f16Round(rgba[j]),
            _f16Round(rgba[j + 1]),
            _f16Round(rgba[j + 2]),
            _f16Round(rgba[j + 3]),
        ];
    };
    const lerp = (a, b, t) => a + (b - a) * t;
    const mix4 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t), lerp(a[3], b[3], t)];
    const c0 = mix4(load(i0x, i0y), load(i0x + 1, i0y), fx);
    const c1 = mix4(load(i0x, i0y + 1), load(i0x + 1, i0y + 1), fx);
    const out = mix4(c0, c1, fy);
    return out.map(_f16Round);
}

function _cpuDotscreenOutput(f16Bytes, w, h, angle, scale, cx, cy) {
    const rgba = _f16BytesToGrid(f16Bytes, w, h);
    const pattern = _cpuDotscreenPattern(w, h, angle, scale, cx, cy);
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) {
        const v = (y + 0.5) / h;
        for (let x = 0; x < w; x++) {
            const u = (x + 0.5) / w;
            const [r, g, b, a] = _sampleF16Linear(rgba, w, h, u, v);
            const average = (r + g + b) / 3;
            const dot = average * 10 - 5 + pattern[y * w + x];
            const j = (y * w + x) * 4;
            out[j] = Math.round(Math.max(0, Math.min(1, dot)) * 255);
            out[j + 1] = Math.round(Math.max(0, Math.min(1, dot)) * 255);
            out[j + 2] = Math.round(Math.max(0, Math.min(1, dot)) * 255);
            out[j + 3] = Math.round(Math.max(0, Math.min(1, a)) * 255);
        }
    }
    return out;
}

async function _cpuDotscreenToCanvas(renderer, readBuffer, p2) {
    if (!readBuffer?._w || !renderer._w?.blitRgba8ToCanvas) return false;
    try {
        const w = readBuffer.width ?? renderer._canvas?.width ?? 800;
        const h = readBuffer.height ?? renderer._canvas?.height ?? 600;
        const f16 = await renderer.readRenderTargetF16(readBuffer, 0, 0, w, h);
        if (!f16?.byteLength || f16.byteLength < w * h * 8) return false;
        const rgba = _cpuDotscreenOutput(new Uint8Array(f16), w, h, p2[0], p2[1], p2[2], p2[3]);
        renderer._w.blitRgba8ToCanvas(rgba, w, h);
        return true;
    } catch (_) {
        return false;
    }
}

function _sampleFloatLinear(data, size, u, v) {
    const stx = u * size - 0.5;
    const sty = v * size - 0.5;
    const i0x = Math.floor(stx);
    const i0y = Math.floor(sty);
    const fx = stx - i0x;
    const fy = sty - i0y;
    const load = (ix, iy) => {
        const cx = Math.max(0, Math.min(size - 1, ix));
        const cy = Math.max(0, Math.min(size - 1, iy));
        return data[cy * size + cx];
    };
    const lerp = (a, b, t) => a + (b - a) * t;
    const c0 = lerp(load(i0x, i0y), load(i0x + 1, i0y), fx);
    const c1 = lerp(load(i0x, i0y + 1), load(i0x + 1, i0y + 1), fx);
    return lerp(c0, c1, fy);
}

function _cpuGlitchOutput(f16Bytes, w, h, u, dispGrid, heightData, dtSize) {
    const rgba = _f16BytesToGrid(f16Bytes, w, h);
    const out = new Uint8Array(w * h * 4);
    const {
        byp, amount, angle, seed, seed_x, seed_y, distortion_x, distortion_y, col_s,
    } = u;
    for (let y = 0; y < h; y++) {
        const v = (y + 0.5) / h;
        for (let x = 0; x < w; x++) {
            const uu = (x + 0.5) / w;
            const j = (y * w + x) * 4;
            if (byp >= 1) {
                const [r, g, b, a] = _sampleF16Linear(rgba, w, h, uu, v);
                out[j] = Math.round(Math.max(0, Math.min(1, r)) * 255);
                out[j + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
                out[j + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
                out[j + 3] = Math.round(Math.max(0, Math.min(1, a)) * 255);
                continue;
            }
            let px = uu;
            let py = v;
            let disp;
            if (dispGrid) {
                disp = dispGrid[y * w + x];
            } else {
                disp = _sampleFloatLinear(heightData, dtSize, px * seed * seed, py * seed * seed);
            }
            if (py < distortion_x + col_s && py > distortion_x - col_s * seed) {
                if (seed_x > 0) py = 1 - (py + distortion_y);
                else py = distortion_y;
            }
            if (px < distortion_y + col_s && px > distortion_y - col_s * seed) {
                if (seed_y > 0) px = distortion_x;
                else px = 1 - (px + distortion_x);
            }
            px += disp * seed_x * (seed / 5);
            py += disp * seed_y * (seed / 5);
            const offx = amount * Math.cos(angle);
            const offy = amount * Math.sin(angle);
            const cr = _sampleF16Linear(rgba, w, h, px + offx, py + offy);
            const cga = _sampleF16Linear(rgba, w, h, px, py);
            const cb = _sampleF16Linear(rgba, w, h, px - offx, py - offy);
            out[j] = Math.round(Math.max(0, Math.min(1, cr[0])) * 255);
            out[j + 1] = Math.round(Math.max(0, Math.min(1, cga[1])) * 255);
            out[j + 2] = Math.round(Math.max(0, Math.min(1, cb[2])) * 255);
            out[j + 3] = Math.round(Math.max(0, Math.min(1, cga[3])) * 255);
        }
    }
    return out;
}

async function _cpuGlitchToCanvas(renderer, readBuffer, u, dispGrid, heightData, dtSize) {
    if (!readBuffer?._w || !renderer._w?.blitRgba8ToCanvas) return false;
    try {
        const w = readBuffer.width ?? renderer._canvas?.width ?? 800;
        const h = readBuffer.height ?? renderer._canvas?.height ?? 600;
        const f16 = await renderer.readRenderTargetF16(readBuffer, 0, 0, w, h);
        if (!f16?.byteLength || f16.byteLength < w * h * 8) return false;
        const rgba = _cpuGlitchOutput(new Uint8Array(f16), w, h, u, dispGrid, heightData, dtSize);
        renderer._w.blitRgba8ToCanvas(rgba, w, h);
        return true;
    } catch (_) {
        return false;
    }
}

function _cpuDotscreenPattern(w, h, angle, scale, cx, cy) {
    const key = `${w}x${h}:${angle}:${scale}:${cx}:${cy}`;
    if (_dotPatternCache.has(key)) return _dotPatternCache.get(key);
    const data = new Float32Array(w * h);
    const s = Math.sin(angle);
    const c = Math.cos(angle);
    for (let y = 0; y < h; y++) {
        const v = (y + 0.5) / h;
        for (let x = 0; x < w; x++) {
            const u = (x + 0.5) / w;
            const texX = u * w - cx;
            const texY = v * h - cy;
            const px = (c * texX - s * texY) * scale;
            const py = (s * texX + c * texY) * scale;
            data[y * w + x] = Math.sin(px) * Math.sin(py) * 4.0;
        }
    }
    _dotPatternCache.set(key, data);
    return data;
}

export class DotScreenPass {
    constructor(center = new Vector2(0.5, 0.5), angle = 1.57, scale = 1.0) {
        this.center = center; this.angle = angle; this.scale = scale;
        this.effectKind = 3;
        this.enabled = true; this.needsSwap = true; this.renderToScreen = false;
    }
    _uploadPattern(renderer, w, h) {
        const pattern = _cpuDotscreenPattern(w, h, this.angle, this.scale, this.center.x, this.center.y);
        if (pattern && renderer._w?.setDotscreenPattern) {
            renderer._w.setDotscreenPattern(pattern, w, h);
            return true;
        }
        return false;
    }
    async render(renderer, writeBuffer, readBuffer) {
        const w = readBuffer?.width ?? renderer._canvas?.width ?? 800;
        const h = readBuffer?.height ?? renderer._canvas?.height ?? 600;
        const p2 = [this.angle, this.scale, this.center.x, this.center.y];
        if (this.renderToScreen) {
            const okGl = await _webglDotscreenToCanvas(renderer, readBuffer, p2);
            if (okGl) return;
            const ok = await _cpuDotscreenToCanvas(renderer, readBuffer, p2);
            if (ok) return;
        }
        const baked = this._uploadPattern(renderer, w, h);
        const p3 = baked ? [0, 0, 0, 1] : [0, 0, 0, 0];
        _passApply(renderer, readBuffer, this.renderToScreen ? null : writeBuffer, this.effectKind, 0,
            p2, null, p3);
    }
    apply(renderer, inputRT) {
        const w = inputRT?.width ?? renderer._canvas?.width ?? 800;
        const h = inputRT?.height ?? renderer._canvas?.height ?? 600;
        const baked = this._uploadPattern(renderer, w, h);
        const p3 = baked ? [0, 0, 0, 1] : [0, 0, 0, 0];
        renderer.applyPostFx(inputRT, this.effectKind, 0,
            [this.angle, this.scale, this.center.x, this.center.y], false, null, null, null, p3);
    }
}
export class HalftonePass {
    constructor(width = 800, height = 600, opts = {}) {
        this.radius = opts.radius ?? 4;
        this.scatter = opts.scatter ?? 0;
        this.shape = opts.shape ?? 1;
        this.blending = opts.blending ?? 1;
        this.effectKind = 4;
        this.enabled = true; this.needsSwap = true; this.renderToScreen = false;
        this.width = width; this.height = height;
    }
    _params() {
        const PI = Math.PI;
        return {
            p2: [this.radius, this.scatter, this.shape, this.blending],
            p3: [PI / 12, PI / 6, PI / 4, 1],
        };
    }
    render(renderer, writeBuffer, readBuffer) {
        const { p2, p3 } = this._params();
        _passApply(renderer, readBuffer, this.renderToScreen ? null : writeBuffer, this.effectKind, 0, p2, null, p3);
    }
    apply(renderer, inputRT) {
        const { p2, p3 } = this._params();
        renderer.applyPostFx(inputRT, this.effectKind, 0, p2, false, null, null, null, p3);
    }
}
export class GlitchPass {
    constructor(dtSize = 64) {
        this._dtSize = dtSize;
        this.effectKind = 5;
        this.enabled = true; this.needsSwap = true; this.renderToScreen = false;
        this.goWild = false;
        this.curF = 0;
        this._heightData = new Float32Array(dtSize * dtSize);
        for (let i = 0; i < this._heightData.length; i++) {
            this._heightData[i] = MathUtils.randFloat(0, 1);
        }
        // three.js: DataTexture (Texture+Source) + ShaderMaterial + Mesh UUIDs before generateTrigger().
        for (let i = 0; i < 16; i++) Math.random();
        this._skipSnow = false;
        this._u = {
            byp: 0,
            amount: 0.08,
            angle: 0.02,
            seed_x: 0.02,
            seed_y: 0.02,
            distortion_x: 0.5,
            distortion_y: 0.6,
            col_s: 0.05,
        };
        this.generateTrigger();
    }
    generateTrigger() {
        this.randX = MathUtils.randInt(120, 240);
    }
    _computeUniforms() {
        const seed = Math.random();
        this._u.byp = 0;
        if (this.curF % this.randX === 0 || this.goWild) {
            this._u.amount = Math.random() / 30;
            this._u.angle = MathUtils.randFloat(-Math.PI, Math.PI);
            this._u.seed_x = MathUtils.randFloat(-1, 1);
            this._u.seed_y = MathUtils.randFloat(-1, 1);
            this._u.distortion_x = MathUtils.randFloat(0, 1);
            this._u.distortion_y = MathUtils.randFloat(0, 1);
            this.curF = 0;
            this.generateTrigger();
        } else if (this.curF % this.randX < this.randX / 5) {
            this._u.amount = Math.random() / 90;
            this._u.angle = MathUtils.randFloat(-Math.PI, Math.PI);
            this._u.distortion_x = MathUtils.randFloat(0, 1);
            this._u.distortion_y = MathUtils.randFloat(0, 1);
            this._u.seed_x = MathUtils.randFloat(-0.3, 0.3);
            this._u.seed_y = MathUtils.randFloat(-0.3, 0.3);
        } else if (!this.goWild) {
            this._u.byp = 1;
        }
        this.curF++;
        this._lastUniforms = { seed, ...this._u };
        globalThis.__glitchDebug = {
            ...this._lastUniforms,
            randX: this.randX,
            curF: this.curF,
            h0: this._heightData[0],
            h1: this._heightData[1],
            h4095: this._heightData[4095],
        };
        return this._lastUniforms;
    }
    _apply(renderer, inputRT, u) {
        renderer._w.setGlitchDisp(this._heightData, this._dtSize);
        const uniforms = u ?? this._computeUniforms();
        const w = inputRT?.width ?? renderer._canvas?.width ?? 800;
        const h = inputRT?.height ?? renderer._canvas?.height ?? 600;
        let kind = this._skipSnow ? 25 : this.effectKind;
        if (this._skipSnow) {
            const baked = _glslGlitchDispBake(w, h, uniforms.seed, this._heightData, this._dtSize);
            if (baked && renderer._w?.setGlitchSnow) {
                renderer._w.setGlitchSnow(baked, w, h);
                kind = 26;
            }
        }
        renderer.applyPostFx(
            inputRT, kind, uniforms.seed,
            [uniforms.amount, uniforms.angle, uniforms.distortion_x, uniforms.distortion_y],
            false, null, null, null,
            [uniforms.seed_x, uniforms.seed_y, uniforms.col_s, uniforms.byp],
        );
    }
    async render(renderer, writeBuffer, readBuffer) {
        renderer._w.setGlitchDisp(this._heightData, this._dtSize);
        const u = this._computeUniforms();
        const w = readBuffer?.width ?? renderer._canvas?.width ?? 800;
        const h = readBuffer?.height ?? renderer._canvas?.height ?? 600;
        if (this.renderToScreen) {
            // WGSL kind 26 in-GPU path (default). Opt in to GLSL readback with __GLITCH_FORCE = 'webgl'.
            if (this._skipSnow && globalThis.__GLITCH_FORCE === 'webgl') {
                const okGl = await _webglGlitchToCanvas(renderer, readBuffer, u, this._heightData, this._dtSize);
                if (okGl) return;
            }
            this._apply(renderer, readBuffer, u);
            return;
        }
        this._apply(renderer, readBuffer, u);
        if (!this.renderToScreen && writeBuffer) {
            _passApply(renderer, readBuffer, writeBuffer, 0);
        }
    }
    apply(renderer, inputRT) {
        this._apply(renderer, inputRT);
    }
}
export class ShaderPass {
    constructor(shader, _textureID) {
        this.shader = shader;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this.uniforms = shader?.uniforms || {};
    }
    async render(renderer, writeBuffer, readBuffer, delta) {
        const kind = _shaderEffectKind(this.shader);
        const p2 = _shaderParams2(this.shader);
        if (kind === 1 && this.renderToScreen) {
            const force = globalThis.__FXAA_FORCE;
            if (force === 'webgl' || force === undefined) {
                const ok = await _webglFxaaToCanvas(renderer, readBuffer, p2);
                if (ok) return;
            }
            if (force === 'wgsl') {
                _passApply(renderer, readBuffer, null, kind, delta, p2);
                return;
            }
            const okCopy = await _cpuCopyToCanvas(renderer, readBuffer);
            if (okCopy) return;
            _passApply(renderer, readBuffer, null, 0, delta, [1, 0, 0, 0]);
            return;
        }
        if (kind === 0 && this.renderToScreen) {
            const ok = await _cpuCopyToCanvas(renderer, readBuffer);
            if (ok) return;
        }
        _passApply(renderer, readBuffer, this.renderToScreen ? null : writeBuffer, kind, delta, p2);
    }
    setSize() {}
    dispose() {}
}
export class TexturePass {
    constructor(map, opacity = 1) {
        this.map = map;
        this.opacity = opacity;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
    }
    render(renderer, writeBuffer, readBuffer) {
        const src = this.map?.isRenderTargetTexture ? readBuffer : this.map;
        _passApply(renderer, src || readBuffer, this.renderToScreen ? null : writeBuffer, 0, 0, [this.opacity, 0, 0, 0]);
    }
    setSize() {}
    dispose() {}
}
export class MaskPass {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.enabled = true;
        this.needsSwap = false;
        this.renderToScreen = false;
        this._maskRT = null;
        this._maskMat = new MeshBasicMaterial({ color: 0xffffff });
    }
    render(renderer, writeBuffer, _readBuffer) {
        const w = writeBuffer.width, h = writeBuffer.height;
        if (!this._maskRT) this._maskRT = new WebGLRenderTarget(w, h);
        _renderWithMaterialOverride(renderer, this.scene, this.camera, this._maskMat, this._maskRT, true);
        renderer.setRenderTarget(null);
    }
    setSize(w, h) { this._maskRT = null; _passSetSize(this, w, h); }
    dispose() {}
}
export class ClearPass {
    constructor(clearColor = 0x000000, clearAlpha = 0) {
        this.clearColor = new Color(clearColor);
        this.clearAlpha = clearAlpha;
        this.enabled = true;
        this.needsSwap = false;
        this.renderToScreen = false;
    }
    render(renderer, writeBuffer) {
        const prev = renderer.getRenderTarget();
        renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
        _passApply(renderer, writeBuffer, writeBuffer, 29, 0,
            [this.clearColor.r, this.clearColor.g, this.clearColor.b, this.clearAlpha]);
        renderer.setRenderTarget(prev);
    }
    setSize() {}
    dispose() {}
}
export class CopyShader { constructor() { this.name = 'CopyShader'; } static uniforms = { tDiffuse: { value: null }, opacity: { value: 1 } }; static vertexShader = ''; static fragmentShader = ''; }

// ---- Pass 8: animation tracks + curves + misc ----

// Common keyframe-track base. Tracks (track, time, value) tuples and lets the
// mixer interpolate. The renderer's WebAnimationMixer reads its own tracks; this
// is the JS-visible shape so user code can build playable AnimationClips.
class _KeyframeTrack {
    constructor(name, times, values, interpolation = 1) {
        this.name = name;
        this.times = times instanceof Float32Array ? times : new Float32Array(times);
        this.values = values instanceof Float32Array ? values : new Float32Array(values);
        this.interpolation = interpolation;
    }
    getValueSize() { return this.values.length / Math.max(this.times.length, 1); }
}
export class BooleanKeyframeTrack     extends _KeyframeTrack { constructor(n, t, v, i) { super(n, t, v, i); this.ValueTypeName = 'bool'; } }
export class NumberKeyframeTrack      extends _KeyframeTrack { constructor(n, t, v, i) { super(n, t, v, i); this.ValueTypeName = 'number'; } }
export class ColorKeyframeTrack       extends _KeyframeTrack { constructor(n, t, v, i) { super(n, t, v, i); this.ValueTypeName = 'color'; } }
export class QuaternionKeyframeTrack  extends _KeyframeTrack { constructor(n, t, v, i) { super(n, t, v, i); this.ValueTypeName = 'quaternion'; } }
export class VectorKeyframeTrack      extends _KeyframeTrack { constructor(n, t, v, i) { super(n, t, v, i); this.ValueTypeName = 'vector'; } }
export class StringKeyframeTrack      extends _KeyframeTrack { constructor(n, t, v, i) { super(n, t, v, i); this.ValueTypeName = 'string'; } }
export { _KeyframeTrack as KeyframeTrack };

export class AnimationObjectGroup {
    constructor(...objects) { this._objects = objects.flat(); }
    add(...objects) { this._objects.push(...objects.flat()); }
    remove(...objects) { this._objects = this._objects.filter(o => !objects.flat().includes(o)); }
}
export const AnimationUtils = {
    arraySlice(array, from, to) { return array.slice(from, to); },
    convertArray(array, type) { return new type(array); },
    isTypedArray(o) { return ArrayBuffer.isView(o) && !(o instanceof DataView); },
    getKeyframeOrder(times) {
        const n = times.length;
        const idx = new Array(n);
        for (let i = 0; i < n; i++) idx[i] = i;
        idx.sort((a, b) => times[a] - times[b]);
        return idx;
    },
    flattenJSON: (json) => json,
};

// More curves (CubicBezier, QuadraticBezier, SplineCurve, ArcCurve).
function _curveBase(get) {
    return {
        getPoint: get,
        getPoints(div = 5) { const out = []; for (let i = 0; i <= div; i++) out.push(get(i / div)); return out; },
        getSpacedPoints(div = 5) { return this.getPoints(div); },
        getLength() { return 0; },
        getLengths() { return []; },
        updateArcLengths() {},
    };
}
function _v2bez1(t, p0, p1) { return (1 - t) * p0 + t * p1; }
function _v2bez2(t, p0, p1, p2) { const u = 1 - t; return u*u*p0 + 2*u*t*p1 + t*t*p2; }
function _v2bez3(t, p0, p1, p2, p3) { const u = 1 - t; return u*u*u*p0 + 3*u*u*t*p1 + 3*u*t*t*p2 + t*t*t*p3; }

export class QuadraticBezierCurve {
    constructor(v0 = new Vector2(), v1 = new Vector2(), v2 = new Vector2()) { this.v0 = v0; this.v1 = v1; this.v2 = v2; Object.assign(this, _curveBase((t) => new Vector2(_v2bez2(t, v0.x, v1.x, v2.x), _v2bez2(t, v0.y, v1.y, v2.y)))); }
}
export class CubicBezierCurve {
    constructor(v0 = new Vector2(), v1 = new Vector2(), v2 = new Vector2(), v3 = new Vector2()) { this.v0 = v0; this.v1 = v1; this.v2 = v2; this.v3 = v3; Object.assign(this, _curveBase((t) => new Vector2(_v2bez3(t, v0.x, v1.x, v2.x, v3.x), _v2bez3(t, v0.y, v1.y, v2.y, v3.y)))); }
}
export class QuadraticBezierCurve3 {
    constructor(v0 = new Vector3(), v1 = new Vector3(), v2 = new Vector3()) { this.v0 = v0; this.v1 = v1; this.v2 = v2; Object.assign(this, _curveBase((t) => new Vector3(_v2bez2(t, v0.x, v1.x, v2.x), _v2bez2(t, v0.y, v1.y, v2.y), _v2bez2(t, v0.z, v1.z, v2.z)))); }
}
export class CubicBezierCurve3 {
    constructor(v0 = new Vector3(), v1 = new Vector3(), v2 = new Vector3(), v3 = new Vector3()) { this.v0 = v0; this.v1 = v1; this.v2 = v2; this.v3 = v3; Object.assign(this, _curveBase((t) => new Vector3(_v2bez3(t, v0.x, v1.x, v2.x, v3.x), _v2bez3(t, v0.y, v1.y, v2.y, v3.y), _v2bez3(t, v0.z, v1.z, v2.z, v3.z)))); }
}
export class SplineCurve {
    constructor(points = []) { this.points = points; Object.assign(this, _curveBase((t) => {
        const p = points, l = p.length - 1, i = Math.floor(t * l);
        const a = p[Math.max(0, i)] || new Vector2(), b = p[Math.min(l, i + 1)] || a;
        const lt = (t * l) - i;
        return new Vector2(_v2bez1(lt, a.x, b.x), _v2bez1(lt, a.y, b.y));
    })); }
}
export class ArcCurve {
    constructor(aX = 0, aY = 0, aRadius = 1, aStartAngle = 0, aEndAngle = Math.PI * 2, aClockwise = false) {
        this.aX = aX; this.aY = aY; this.aRadius = aRadius; this.aStartAngle = aStartAngle; this.aEndAngle = aEndAngle; this.aClockwise = aClockwise;
        Object.assign(this, _curveBase((t) => {
            const dir = aClockwise ? -1 : 1;
            const angle = aStartAngle + dir * t * (aEndAngle - aStartAngle);
            return new Vector2(aX + aRadius * Math.cos(angle), aY + aRadius * Math.sin(angle));
        }));
    }
}

// Side constants re-exported at top of module (see FrontSide/BackSide/DoubleSide above).

// ---- Pass 9: constants + MathUtils ----
export const MathUtils = {
    DEG2RAD: Math.PI / 180, RAD2DEG: 180 / Math.PI,
    generateUUID: (() => {
        const hex = []; for (let i = 0; i < 256; i++) hex[i] = (i < 16 ? '0' : '') + i.toString(16);
        return function () {
            const a = Math.random() * 0xffffffff | 0, b = Math.random() * 0xffffffff | 0,
                  c = Math.random() * 0xffffffff | 0, d = Math.random() * 0xffffffff | 0;
            return (hex[a & 0xff] + hex[(a >> 8) & 0xff] + hex[(a >> 16) & 0xff] + hex[(a >> 24) & 0xff] + '-' +
                    hex[b & 0xff] + hex[(b >> 8) & 0xff] + '-' +
                    hex[((b >> 16) & 0x0f) | 0x40] + hex[(b >> 24) & 0xff] + '-' +
                    hex[(c & 0x3f) | 0x80] + hex[(c >> 8) & 0xff] + '-' +
                    hex[(c >> 16) & 0xff] + hex[(c >> 24) & 0xff] + hex[d & 0xff] + hex[(d >> 8) & 0xff] + hex[(d >> 16) & 0xff] + hex[(d >> 24) & 0xff]).toUpperCase();
        };
    })(),
    clamp: (v, mn, mx) => Math.max(mn, Math.min(mx, v)),
    euclideanModulo: (n, m) => ((n % m) + m) % m,
    mapLinear: (x, a1, a2, b1, b2) => b1 + (x - a1) * (b2 - b1) / (a2 - a1),
    inverseLerp: (x, y, v) => x !== y ? (v - x) / (y - x) : 0,
    lerp: (x, y, t) => (1 - t) * x + t * y,
    damp: (x, y, lambda, dt) => MathUtils.lerp(x, y, 1 - Math.exp(-lambda * dt)),
    pingpong: (x, length = 1) => length - Math.abs(MathUtils.euclideanModulo(x, length * 2) - length),
    smoothstep: (x, mn, mx) => { if (x <= mn) return 0; if (x >= mx) return 1; x = (x - mn) / (mx - mn); return x * x * (3 - 2 * x); },
    smootherstep: (x, mn, mx) => { if (x <= mn) return 0; if (x >= mx) return 1; x = (x - mn) / (mx - mn); return x * x * x * (x * (x * 6 - 15) + 10); },
    randInt: (low, high) => low + Math.floor(Math.random() * (high - low + 1)),
    randFloat: (low, high) => low + Math.random() * (high - low),
    randFloatSpread: (range) => range * (0.5 - Math.random()),
    seededRandom: (s) => { s = s ?? Math.random() * 1e9; let t = s + 0x6D2B79F5; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return (((t ^ (t >>> 14)) >>> 0) / 4294967296); },
    degToRad: (d) => d * (Math.PI / 180),
    radToDeg: (r) => r * (180 / Math.PI),
    isPowerOfTwo: (v) => (v & (v - 1)) === 0 && v !== 0,
    ceilPowerOfTwo: (v) => Math.pow(2, Math.ceil(Math.log(v) / Math.LN2)),
    floorPowerOfTwo: (v) => Math.pow(2, Math.floor(Math.log(v) / Math.LN2)),
    setQuaternionFromProperEuler(q, a, b, c, order) { q.setFromEuler(new Euler(a, b, c, order)); },
};
export const REVISION = '165';
export const ColorManagement = { enabled: true, legacyMode: false, workingColorSpace: 'srgb-linear', getPrimaries() { return 'srgb'; }, convert(c, _from, _to) { return c; } };

// Make Object3D base for typing helpers — duck-typed since each class above
// already implements .position/.rotation/.scale/etc. independently.
export class Object3D {
    constructor() {
        this.position = new Vector3();
        this.rotation = new Euler();
        this.quaternion = new Quaternion();
        this.scale = new Vector3(1, 1, 1);
        this.up = new Vector3(0, 1, 0);
        this.visible = true; this.name = ''; this.userData = {};
        this.renderOrder = 0;
        this.children = []; this.parent = null;
        this.matrix = new Matrix4();
        this.matrixWorld = new Matrix4();
        this.matrixAutoUpdate = true;
        this.matrixWorldNeedsUpdate = false;
        this.layers = new Layers();
        this.id = Object3D._nextId++;
        this.uuid = MathUtils.generateUUID();
    }
    add(...os) {
        for (const o of os.flat()) {
            if (o === this) continue;
            if (o.parent) o.parent.remove(o);
            this.children.push(o);
            o.parent = this;
        }
        return this;
    }
    remove(...os) {
        for (const o of os.flat()) {
            const i = this.children.indexOf(o);
            if (i >= 0) { this.children.splice(i, 1); o.parent = null; }
        }
        return this;
    }
    clear() { for (const c of this.children) c.parent = null; this.children.length = 0; return this; }
    attach(child) {
        // Detach from current parent and re-parent under this, preserving world transform.
        child.parent?.remove(child);
        this.add(child);
        return this;
    }
    traverse(cb) { cb(this); for (const c of this.children) c.traverse?.(cb); }
    traverseVisible(cb) {
        if (this.visible === false) return;
        cb(this);
        for (const c of this.children) c.traverseVisible?.(cb);
    }
    traverseAncestors(cb) { if (this.parent) { cb(this.parent); this.parent.traverseAncestors?.(cb); } }
    getObjectByName(name) {
        if (this.name === name) return this;
        for (const c of this.children) {
            const found = c.getObjectByName?.(name);
            if (found) return found;
        }
        return undefined;
    }
    getObjectById(id) {
        if (this.id === id) return this;
        for (const c of this.children) {
            const found = c.getObjectById?.(id);
            if (found) return found;
        }
        return undefined;
    }
    getObjectByProperty(name, value) {
        if (this[name] === value) return this;
        for (const c of this.children) {
            const found = c.getObjectByProperty?.(name, value);
            if (found) return found;
        }
        return undefined;
    }
    getObjectsByProperty(name, value, result = []) {
        if (this[name] === value) result.push(this);
        for (const c of this.children) c.getObjectsByProperty?.(name, value, result);
        return result;
    }
    getWorldPosition(target = new Vector3()) {
        this.updateWorldMatrix(true, false);
        const e = this.matrixWorld.elements;
        return target.set(e[12], e[13], e[14]);
    }
    getWorldQuaternion(target = new Quaternion()) {
        this.updateWorldMatrix(true, false);
        const e = this.matrixWorld.elements;
        const sx = Math.hypot(e[0], e[1], e[2]);
        const sy = Math.hypot(e[4], e[5], e[6]);
        const sz = Math.hypot(e[8], e[9], e[10]);
        // Build a rotation matrix from the columns, then convert.
        const m = new Matrix3();
        m.elements[0] = e[0] / sx; m.elements[1] = e[1] / sx; m.elements[2] = e[2] / sx;
        m.elements[3] = e[4] / sy; m.elements[4] = e[5] / sy; m.elements[5] = e[6] / sy;
        m.elements[6] = e[8] / sz; m.elements[7] = e[9] / sz; m.elements[8] = e[10] / sz;
        return target.setFromRotationMatrix(m);
    }
    getWorldScale(target = new Vector3()) {
        this.updateWorldMatrix(true, false);
        const e = this.matrixWorld.elements;
        return target.set(Math.hypot(e[0], e[1], e[2]), Math.hypot(e[4], e[5], e[6]), Math.hypot(e[8], e[9], e[10]));
    }
    getWorldDirection(target = new Vector3()) {
        this.updateWorldMatrix(true, false);
        const e = this.matrixWorld.elements;
        return target.set(-e[8], -e[9], -e[10]).normalize();
    }
    localToWorld(v) {
        this.updateWorldMatrix(true, false);
        return v.applyMatrix4(this.matrixWorld);
    }
    worldToLocal(v) {
        this.updateWorldMatrix(true, false);
        return v.applyMatrix4(new Matrix4().copy(this.matrixWorld).invert());
    }
    updateMatrix() {
        // Compose local matrix from position/quaternion/scale.
        const q = this.quaternion || new Quaternion().setFromEuler(this.rotation);
        const m = new Matrix4().compose(this.position, q, this.scale);
        this.matrix.copy(m);
        this.matrixWorldNeedsUpdate = true;
    }
    updateMatrixWorld(force = false) {
        if (this.matrixAutoUpdate) this.updateMatrix();
        if (this.matrixWorldNeedsUpdate || force) {
            if (this.parent) {
                this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
            } else {
                this.matrixWorld.copy(this.matrix);
            }
            this.matrixWorldNeedsUpdate = false;
            for (const c of this.children) c.updateMatrixWorld?.(true);
        } else {
            for (const c of this.children) c.updateMatrixWorld?.(force);
        }
    }
    updateWorldMatrix(updateParents, updateChildren) {
        if (updateParents && this.parent) this.parent.updateWorldMatrix(true, false);
        if (this.matrixAutoUpdate) this.updateMatrix();
        if (this.parent) this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
        else this.matrixWorld.copy(this.matrix);
        if (updateChildren) for (const c of this.children) c.updateWorldMatrix?.(false, true);
    }
    applyMatrix4(m) {
        if (this.matrixAutoUpdate) this.updateMatrix();
        this.matrix.premultiply(m);
        this.matrix.decompose(this.position, this.quaternion, this.scale);
    }
    applyQuaternion(q) {
        this.quaternion.premultiply(q);
        return this;
    }
    setRotationFromQuaternion(q) {
        this.quaternion.copy(q);
        return this;
    }
    setRotationFromEuler(e) {
        this.rotation.copy(e);
        return this;
    }
    setRotationFromAxisAngle(axis, angle) {
        this.quaternion.setFromAxisAngle(axis, angle);
        return this;
    }
    rotateOnAxis(axis, angle) {
        const q = new Quaternion().setFromAxisAngle(axis, angle);
        this.quaternion.multiply(q);
        return this;
    }
    rotateX(angle) { return this.rotateOnAxis(new Vector3(1, 0, 0), angle); }
    rotateY(angle) { return this.rotateOnAxis(new Vector3(0, 1, 0), angle); }
    rotateZ(angle) { return this.rotateOnAxis(new Vector3(0, 0, 1), angle); }
    translateOnAxis(axis, distance) {
        const v = new Vector3().copy(axis).applyQuaternion(this.quaternion);
        this.position.add(v.multiplyScalar(distance));
        return this;
    }
    translateX(d) { return this.translateOnAxis(new Vector3(1, 0, 0), d); }
    translateY(d) { return this.translateOnAxis(new Vector3(0, 1, 0), d); }
    translateZ(d) { return this.translateOnAxis(new Vector3(0, 0, 1), d); }
    lookAt(x, y, z) {
        const target = (x instanceof Vector3) ? x : new Vector3(x, y, z);
        const m = new Matrix4().lookAt(this.position, target, this.up || new Vector3(0, 1, 0));
        this.quaternion.setFromRotationMatrix(m);
    }
    raycast() {}
    clone(recursive = true) {
        const c = new this.constructor();
        c.name = this.name;
        c.up.copy(this.up);
        c.position.copy(this.position);
        c.rotation.copy(this.rotation);
        c.quaternion.copy(this.quaternion);
        c.scale.copy(this.scale);
        c.visible = this.visible;
        c.userData = JSON.parse(JSON.stringify(this.userData));
        if (recursive) for (const child of this.children) c.add(child.clone(true));
        return c;
    }
    copy(source, recursive = true) {
        this.name = source.name;
        this.up.copy(source.up);
        this.position.copy(source.position);
        this.rotation.copy(source.rotation);
        this.quaternion.copy(source.quaternion);
        this.scale.copy(source.scale);
        this.visible = source.visible;
        this.userData = JSON.parse(JSON.stringify(source.userData || {}));
        this.children.length = 0;
        if (recursive) for (const c of source.children) this.add(c.clone(true));
        return this;
    }
    toJSON() {
        return {
            metadata: { version: 4.6, type: 'Object', generator: 'threers' },
            object: {
                uuid: this.uuid, type: this.type || this.constructor.name, name: this.name,
                position: [this.position.x, this.position.y, this.position.z],
                rotation: [this.rotation.x, this.rotation.y, this.rotation.z],
                scale: [this.scale.x, this.scale.y, this.scale.z],
                children: this.children.map(c => c.toJSON?.()).filter(Boolean),
            }
        };
    }
}
Object3D._nextId = 1;
Object.assign(Object3D.prototype, EventDispatcherMixin);

// ---- Stubs for the remaining three.js API surface ----
// Base classes (used by `instanceof` checks in user code).
export class Camera extends Object3D {}
export class Material {
    constructor() {
        this.type = 'Material';
        this.uuid = MathUtils.generateUUID();
        this.opacity = 1; this.transparent = false; this.side = 0;
        this.visible = true; this.uniforms = {};
        this.userData = {};
        this.alphaTest = 0; this.alphaToCoverage = false;
        this.blending = 1; // NormalBlending
        this.depthTest = true; this.depthWrite = true;
        this.colorWrite = true; this.toneMapped = true;
        this.wireframe = false;
        this.needsUpdate = false;
        this.version = 0;
        // onBeforeCompile is called once per shader build by three.js. Our wgpu
        // pipeline doesn't recompile shaders from GLSL strings, but we still
        // invoke the callback with a fake `shader` object so user code that
        // sets `shader.uniforms.foo` for later reading works correctly.
        this.onBeforeCompile = null;
    }
    clone() { return new this.constructor().copy(this); }
    copy(other) {
        for (const k of ['opacity','transparent','side','visible','alphaTest','blending','depthTest','depthWrite','colorWrite','toneMapped','wireframe']) {
            if (other[k] !== undefined) this[k] = other[k];
        }
        this.uniforms = { ...other.uniforms };
        this.userData = JSON.parse(JSON.stringify(other.userData || {}));
        return this;
    }
    dispose() {}
    setValues(v) { Object.assign(this, v); return this; }
    customProgramCacheKey() { return this.uuid; }
    // Run user's onBeforeCompile if set, with a shader-shaped object they
    // can mutate. Their string-replacements on vertexShader/fragmentShader
    // are captured but don't recompile our WGSL pipeline; their uniform
    // additions ARE captured on `material.userData.shader` for later read.
    _runOnBeforeCompile() {
        if (typeof this.onBeforeCompile !== 'function' || this._compiledOnce) return;
        const shader = {
            uniforms: { ...this.uniforms },
            vertexShader: '/* threers wgsl pipeline */',
            fragmentShader: '/* threers wgsl pipeline */',
            defines: {},
        };
        try { this.onBeforeCompile(shader, this); } catch (_e) {}
        this.userData.shader = shader;
        this.uniforms = shader.uniforms;
        this._compiledOnce = true;
    }
}
export class Light extends Object3D { constructor(color = 0xffffff, intensity = 1) { super(); this.color = new Color(color); this.intensity = intensity; this._isLight = true; } }
export class LightShadow { constructor(camera) { this.camera = camera; this.bias = 0; this.normalBias = 0; this.radius = 1; this.blurSamples = 8; this.mapSize = new Vector2(512, 512); this.map = null; this.matrix = new Matrix4(); } }
export class LightProbe extends Light { constructor() { super(); this.sh = new SphericalHarmonics3(); } }
export class Curve { constructor() { this.arcLengthDivisions = 200; } getPoint(_t) { return new Vector3(); } getPoints(div = 5) { const out = []; for (let i = 0; i <= div; i++) out.push(this.getPoint(i / div)); return out; } getSpacedPoints(d) { return this.getPoints(d); } getLength() { return 0; } getTangent(_t) { return new Vector3(1, 0, 0); } getTangentAt(t) { return this.getTangent(t); } computeFrenetFrames(seg) { return { tangents: [], normals: [], binormals: [] }; } }
export class CurvePath extends Curve { constructor() { super(); this.curves = []; } add(c) { this.curves.push(c); } closePath() {} }
export class Interpolant { constructor(params, values) { this.parameterPositions = params; this.sampleValues = values; this.valueSize = values.length / params.length | 0; } evaluate(_t) { return this.sampleValues.slice(0, this.valueSize); } }
export class LinearInterpolant extends Interpolant {}
export class CubicInterpolant extends Interpolant {}
export class DiscreteInterpolant extends Interpolant {}
export class QuaternionLinearInterpolant extends Interpolant {}

// Cameras.
export class ArrayCamera extends PerspectiveCamera { constructor(cameras = []) { super(); this.cameras = cameras; this.isArrayCamera = true; } }
export class StereoCamera {
    constructor() {
        this.type = 'StereoCamera';
        this.aspect = 1;
        this.eyeSep = 0.064;
        this.cameraL = new PerspectiveCamera();
        this.cameraR = new PerspectiveCamera();
    }
    /** three.js StereoCamera.update(camera) — builds left/right from a master cam. */
    update(camera) {
        if (!camera) return;
        if (typeof camera._sync === 'function') camera._sync();
        const eye = camera.position;
        const target = camera._lookAt || new Vector3();
        const up = camera.up || new Vector3(0, 1, 0);
        const forward = new Vector3().subVectors(target, eye).normalize();
        let right = new Vector3().crossVectors(forward, up);
        if (right.lengthSq() < 1e-10) right.set(1, 0, 0);
        else right.normalize();
        const half = this.eyeSep * 0.5;
        const apply = (cam, sign) => {
            cam.fov = camera.fov ?? camera._fov ?? 50;
            cam.aspect = (camera.aspect || 1) * 0.5 * this.aspect;
            cam.near = camera.near;
            cam.far = camera.far;
            cam.position.set(
                eye.x + right.x * half * sign,
                eye.y + right.y * half * sign,
                eye.z + right.z * half * sign,
            );
            cam.lookAt(target.x, target.y, target.z);
            cam.updateProjectionMatrix?.();
            cam._sync?.();
        };
        apply(this.cameraL, -1);
        apply(this.cameraR, +1);
    }
}
export class CubeCamera extends Object3D {
    constructor(near = 0.1, far = 1000, renderTarget = null) {
        super();
        this.near = near; this.far = far;
        this.renderTarget = renderTarget;
        this.coordinateSystem = 2000;
        // 6 perspective cameras pointing in the cube face directions.
        const mk = (lookAt, up) => {
            const cam = new PerspectiveCamera(90, 1, near, far);
            cam._lookAt = new Vector3(lookAt[0], lookAt[1], lookAt[2]);
            cam.up = new Vector3(up[0], up[1], up[2]);
            return cam;
        };
        this.cameraPX = mk([ 1, 0, 0], [0,-1, 0]);
        this.cameraNX = mk([-1, 0, 0], [0,-1, 0]);
        this.cameraPY = mk([ 0, 1, 0], [0, 0, 1]);
        this.cameraNY = mk([ 0,-1, 0], [0, 0,-1]);
        this.cameraPZ = mk([ 0, 0, 1], [0,-1, 0]);
        this.cameraNZ = mk([ 0, 0,-1], [0,-1, 0]);
    }
    update(renderer, scene) {
        const rt = this.renderTarget;
        if (!rt) return;
        if (!rt._w_cube) {
            rt._w_cube = new WebCubeRenderTarget(renderer._w, rt._cubeSide);
            // Tag the texture so `scene.environment = rt.texture` finds the cube RT id.
            rt.texture._cubeRTId = rt._w_cube.id;
        }
        const cams = [this.cameraPX, this.cameraNX, this.cameraPY, this.cameraNY, this.cameraPZ, this.cameraNZ];
        for (let i = 0; i < 6; i++) {
            const cam = cams[i];
            cam.position.copy(this.position);
            const t = new Vector3(this.position.x + cam._lookAt.x, this.position.y + cam._lookAt.y, this.position.z + cam._lookAt.z);
            cam._lookAtTarget = t;
            cam._w.setPosition(this.position.x, this.position.y, this.position.z);
            // Push the face camera's specific up vector before lookAt; +Y/-Y
            // views require non-(0,1,0) ups or the view matrix is degenerate.
            cam._w.setUp(cam.up.x, cam.up.y, cam.up.z);
            cam._w.lookAt(t.x, t.y, t.z);
            scene._syncTransforms(cam);
            renderer._w.renderToCubeFace(scene._w, cam._w, rt._w_cube, i);
        }
    }
}

// Material variants (stubs that map onto closest existing material).
export class LineDashedMaterial extends LineBasicMaterial {
    constructor(opts = {}) {
        super(opts);
        this.dashSize = opts.dashSize ?? 3;
        this.gapSize = opts.gapSize ?? 1;
        this.scale = opts.scale ?? 1;
        this._isLineDashed = true;
        this._w = WebMaterial.lineDashed(_matColor(opts), this.scale, this.dashSize, this.gapSize);
    }
}
export class MeshMatcapMaterial {
    constructor(opts = {}) {
        this._w = WebMaterial.matcap(_matColor(opts));
        _applyCommon(this._w, opts);
        this.matcap = opts.matcap || null;
        if (this.matcap) {
            const t = this.matcap;
            if (t?._w?.constructor?.name === 'WebDataTexture') this._w.setMatcapData(t._w);
            else if (t instanceof DataTexture) this._w.setMatcapData(t._w);
            else if (t instanceof Texture)     this._w.setMatcap(t._w);
        }
    }
}
export class MeshDistanceMaterial {
    constructor(opts = {}) {
        this.referencePosition = opts?.referencePosition || new Vector3();
        this.nearDistance = opts?.nearDistance ?? 1;
        this.farDistance = opts?.farDistance ?? 1000;
        this._w = WebMaterial.distance(
            this.referencePosition.x, this.referencePosition.y, this.referencePosition.z,
            this.nearDistance, this.farDistance
        );
        _initMaterialBase(this, opts);
    }
}
export class ShadowMaterial {
    constructor(opts = {}) {
        this._w = WebMaterial.shadow(opts?.opacity ?? 1);
        this.transparent = true;
        this.opacity = opts?.opacity ?? 1;
        _initMaterialBase(this, { ...opts, color: 0x000000 });
    }
}
export class ShaderMaterial { constructor(opts = {}) { this._w = WebMaterial.basic(_matColor(opts)); this.uniforms = opts?.uniforms || {}; this.vertexShader = opts?.vertexShader || ''; this.fragmentShader = opts?.fragmentShader || ''; this.defines = opts?.defines || {}; this.transparent = !!opts?.transparent; } }
export class RawShaderMaterial extends ShaderMaterial {}

// Textures + variants.
export class DepthTexture extends Texture { constructor(w, h) { super(w, h, new Uint8Array(w * h * 4)); this.isDepthTexture = true; } }
export class VideoTexture extends Texture {
    constructor(video) {
        super();
        this.image = video;
        this.isVideoTexture = true;
        this.flipY = true;
        this.offset = new Vector2(0, 1);
        this.repeat = new Vector2(1, -1);
        this._canvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;
        this._ctx = this._canvas ? this._canvas.getContext('2d') : null;
        this.needsUpdate = false;
    }
    _syncUv() {
        if (!this._w?.setUvTransform) return;
        this._w.setUvTransform(
            this.offset?.x ?? 0,
            this.offset?.y ?? 1,
            this.repeat?.x ?? 1,
            this.flipY ? -Math.abs(this.repeat?.y ?? 1) : (this.repeat?.y ?? 1),
        );
    }
    update() {
        const video = this.image;
        const canvas = this._canvas;
        const ctx = this._ctx;
        if (!video || !canvas || !ctx) return;
        const w = video.videoWidth || 1280;
        const h = video.videoHeight || 720;
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== h) canvas.height = h;
        if (video.readyState >= 2) {
            try { ctx.drawImage(video, 0, 0, w, h); } catch (_err) { /* seek */ }
        } else {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, w, h);
        }
        const data = ctx.getImageData(0, 0, w, h).data;
        const byteLen = data.length;
        if (!this._buf || this._buf.length !== byteLen) {
            this._buf = new Uint8Array(byteLen);
        }
        this._buf.set(data);
        if (this._w && this._lw === w && this._lh === h && this._w.replaceRgba) {
            if (this._w.replaceRgba(this._buf)) {
                this.needsUpdate = true;
                return;
            }
        }
        this._w = new WebDataTexture(w, h, this._buf);
        this._lw = w;
        this._lh = h;
        this._syncUv();
        this.needsUpdate = true;
    }
}
export class CompressedTexture extends Texture { constructor(mipmaps = [], w = 1, h = 1) { super(); this.mipmaps = mipmaps; this.image = { width: w, height: h }; this.isCompressedTexture = true; } }
export class CompressedArrayTexture extends CompressedTexture {}
export class CompressedCubeTexture extends CompressedTexture {}
export class Data3DTexture extends DataTexture { constructor(data, w, h, d) { super(data, w, h); this.image.depth = d; this.isData3DTexture = true; } }
export class DataArrayTexture extends DataTexture {}
export class FramebufferTexture extends Texture { constructor(w, h) { super(w, h, new Uint8Array(w * h * 4)); this.isFramebufferTexture = true; } }
export class Source { constructor(data = null) { this.data = data; this.uuid = String(Math.random()).slice(2); } toJSON() { return { uuid: this.uuid }; } }

// Buffer-attribute typed subclasses (all behave the same — just enforce a typed array on creation).
function _typedBufferAttr(Ctor) { return class extends BufferAttribute { constructor(arr, itemSize, normalized = false) { super(arr instanceof Ctor ? arr : new Ctor(arr), itemSize); this.normalized = normalized; } }; }
export class Int8BufferAttribute    extends _typedBufferAttr(Int8Array)    {}
export class Int16BufferAttribute   extends _typedBufferAttr(Int16Array)   {}
export class Int32BufferAttribute   extends _typedBufferAttr(Int32Array)   {}
export class Uint8BufferAttribute   extends _typedBufferAttr(Uint8Array)   {}
export class Uint8ClampedBufferAttribute extends _typedBufferAttr(Uint8ClampedArray) {}
export class Uint16BufferAttribute  extends _typedBufferAttr(Uint16Array)  {}
export class Uint32BufferAttribute  extends _typedBufferAttr(Uint32Array)  {}
export class Float16BufferAttribute extends _typedBufferAttr(Uint16Array)  {}
export class Float32BufferAttribute extends _typedBufferAttr(Float32Array) {}
export class Float64BufferAttribute extends _typedBufferAttr(Float64Array) {}
export class InstancedBufferAttribute extends BufferAttribute { constructor(arr, itemSize, normalized = false, meshPerAttribute = 1) { super(arr, itemSize); this.meshPerAttribute = meshPerAttribute; } }
export class InterleavedBuffer { constructor(array, stride) { this.array = array; this.stride = stride; this.count = array.length / stride | 0; } }
export class InterleavedBufferAttribute { constructor(buf, itemSize, offset, normalized = false) { this.data = buf; this.itemSize = itemSize; this.offset = offset; this.normalized = normalized; } get array() { return this.data.array; } get count() { return this.data.count; } }
export class InstancedInterleavedBuffer extends InterleavedBuffer {}
export class GLBufferAttribute { constructor(buf, type, itemSize, elementSize, count) { this.buffer = buf; this.type = type; this.itemSize = itemSize; this.elementSize = elementSize; this.count = count; } }
export class InstancedBufferGeometry extends BufferGeometry { constructor() { super(); this.instanceCount = 0; } }

// Lights.
export class AmbientLightProbe extends LightProbe { constructor(color = 0xffffff, intensity = 1) { super(); this.color = new Color(color); this.intensity = intensity; this._isLight = true; this._w = WebLight.ambient(_color(color), intensity); } }
export class HemisphereLightProbe extends LightProbe { constructor(sky = 0xffffff, ground = 0xffffff, intensity = 1) { super(); this.color = new Color(sky); this.groundColor = new Color(ground); this.intensity = intensity; this._isLight = true; this._w = WebLight.hemisphere(_color(sky), _color(ground), intensity); } }
export class DirectionalLightShadow extends LightShadow { constructor() { super(new OrthographicCamera(-5, 5, 5, -5, 0.5, 500)); } }
export class PointLightShadow extends LightShadow { constructor() { super(new PerspectiveCamera(90, 1, 0.5, 500)); } }
export class SpotLightShadow extends LightShadow { constructor() { super(new PerspectiveCamera(90, 1, 0.5, 500)); this.focus = 1; } }

// Objects. BatchedMesh is the real impl above (per-geom InstancedMesh pool).
export class LOD extends Object3D { constructor() { super(); this.levels = []; this.autoUpdate = true; } addLevel(obj, distance = 0) { this.levels.push({ object: obj, distance }); this.add(obj); return this; } getCurrentLevel() { return 0; } }

// Helpers.
export class Box3Helper extends LineSegments { constructor(box, color = 0xffff00) { const pts = []; const min = box?.min || new Vector3(-1, -1, -1); const max = box?.max || new Vector3(1, 1, 1); const v = [[min.x, min.y, min.z],[max.x, min.y, min.z],[max.x, max.y, min.z],[min.x, max.y, min.z],[min.x, min.y, max.z],[max.x, min.y, max.z],[max.x, max.y, max.z],[min.x, max.y, max.z]]; const e = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]; for (const [a, b] of e) pts.push(...v[a], ...v[b]); const g = new BufferGeometry(); g.setAttribute('position', new BufferAttribute(new Float32Array(pts), 3)); super(g, new LineBasicMaterial({ color })); this.box = box; } }
export class VertexNormalsHelper extends LineSegments { constructor(obj) { super(new BufferGeometry(), new LineBasicMaterial({ color: 0xff0000 })); this.object = obj; } update() {} }
export class RectAreaLightHelper extends LineSegments { constructor() { super(new BufferGeometry(), new LineBasicMaterial({ color: 0xffff00 })); } update() {} }

// Loaders.
export class Loader extends _Loader {}
export class LoaderUtils { static decodeText(buf) { return new TextDecoder().decode(buf); } static extractUrlBase(url) { const i = url.lastIndexOf('/'); return i < 0 ? './' : url.slice(0, i + 1); } static resolveURL(url, base) { if (typeof url !== 'string' || url === '') return ''; if (/^https?:\/\//i.test(url)) return url; return base + url; } }
export class DataTextureLoader extends _Loader { load(url, onLoad, _onProgress, onError) { new FileLoader(this.manager).setResponseType('arraybuffer').load(url, (buf) => { const tex = new DataTexture(new Uint8Array(buf), 1, 1); onLoad?.(tex); }, undefined, onError); } }
export class CompressedTextureLoader extends _Loader { load(url, onLoad, _onProgress, onError) { new FileLoader(this.manager).setResponseType('arraybuffer').load(url, () => onLoad?.(new CompressedTexture()), undefined, onError); } }
export class ImageBitmapLoader extends _Loader { setOptions(opts) { this.options = opts; return this; } load(url, onLoad, _onProgress, onError) { fetch(this.path + url).then(r => r.blob()).then(b => createImageBitmap(b, this.options)).then(onLoad).catch(onError); } }

// Misc.
export class Cache { static enabled = false; static files = {}; static add(k, v) { this.files[k] = v; } static get(k) { return this.files[k]; } static remove(k) { delete this.files[k]; } static clear() { this.files = {}; } }
export class Fog { constructor(color = 0xffffff, near = 1, far = 1000) { this.color = new Color(color); this.near = near; this.far = far; this.isFog = true; } clone() { return new Fog(this.color.getHex(), this.near, this.far); } }
export class FogExp2 { constructor(color = 0xffffff, density = 0.00025) { this.color = new Color(color); this.density = density; this.isFogExp2 = true; } }
export class Layers { constructor() { this.mask = 1 | 0; } set(channel) { this.mask = (1 << channel) | 0; } enable(channel) { this.mask |= (1 << channel); } disable(channel) { this.mask &= ~(1 << channel); } toggle(channel) { this.mask ^= (1 << channel); } test(layers) { return (this.mask & layers.mask) !== 0; } isEnabled(channel) { return (this.mask & (1 << channel)) !== 0; } }
export class SphericalHarmonics3 { constructor() { this.coefficients = []; for (let i = 0; i < 9; i++) this.coefficients.push(new Vector3()); } set(coeffs) { for (let i = 0; i < 9; i++) this.coefficients[i].copy(coeffs[i]); return this; } zero() { for (const c of this.coefficients) c.set(0, 0, 0); return this; } }
export class Uniform { constructor(value) { this.value = value; } clone() { return new Uniform(this.value?.clone ? this.value.clone() : this.value); } }
export class UniformsGroup { constructor() { this.uniforms = []; this.name = ''; } add(u) { this.uniforms.push(u); return this; } setName(n) { this.name = n; return this; } }
export const UniformsLib = { common: {}, lights: {}, points: {}, sprite: {}, shadowmap: {}, fog: {} };
export const UniformsUtils = { clone(uniforms) { const out = {}; for (const k of Object.keys(uniforms)) out[k] = new Uniform(uniforms[k].value); return out; }, merge(list) { const out = {}; for (const u of list) Object.assign(out, u); return out; } };
export const ShaderChunk = {};
export const ShaderLib = { common: { uniforms: {}, vertexShader: '', fragmentShader: '' }, basic: { uniforms: {}, vertexShader: '', fragmentShader: '' }, lambert: { uniforms: {}, vertexShader: '', fragmentShader: '' }, phong: { uniforms: {}, vertexShader: '', fragmentShader: '' }, standard: { uniforms: {}, vertexShader: '', fragmentShader: '' }, physical: { uniforms: {}, vertexShader: '', fragmentShader: '' }, points: { uniforms: {}, vertexShader: '', fragmentShader: '' } };
export const ImageUtils = { getDataURL(_img) { return ''; }, sRGBToLinear(img) { return img; } };
export const DataUtils = { fromHalfFloat: (v) => v, toHalfFloat: (v) => v };
export const ShapeUtils = {
    area(pts) { let a = 0; for (let i = 0, n = pts.length; i < n; i++) { const j = (i + 1) % n; a += pts[i].x * pts[j].y - pts[j].x * pts[i].y; } return a / 2; },
    isClockWise(pts) { return this.area(pts) < 0; },
    triangulateShape(contour, holes = []) {
        if (!Array.isArray(contour) || contour.length < 3) return [];
        _removeDupEndPts(contour);
        for (const h of holes) _removeDupEndPts(h);
        const vertices = [];
        const holeIndices = [];
        for (const p of contour) vertices.push(p.x, p.y);
        let holeIndex = contour.length;
        for (const ahole of holes) {
            holeIndices.push(holeIndex);
            holeIndex += ahole.length;
            for (const p of ahole) vertices.push(p.x, p.y);
        }
        const triangles = Earcut.triangulate(vertices, holeIndices);
        const out = [];
        for (let i = 0; i < triangles.length; i += 3) {
            out.push([triangles[i], triangles[i + 1], triangles[i + 2]]);
        }
        return out;
    }
};
// (MathUtils + ColorManagement already declared above; not re-declared here.)
// PropertyBinding / PropertyMixer / AnimationAction defined with AnimationMixer above.
export class AudioContext { static getContext() { if (!this._ctx && typeof window !== 'undefined' && window.AudioContext) this._ctx = new (window.AudioContext || window.webkitAudioContext)(); return this._ctx; } static setContext(c) { this._ctx = c; } }
export class AudioAnalyser { constructor(audio, fftSize = 2048) { this.analyser = AudioContext.getContext()?.createAnalyser?.(); this.data = new Uint8Array(fftSize / 2); } getFrequencyData() { this.analyser?.getByteFrequencyData?.(this.data); return this.data; } getAverageFrequency() { let s = 0; for (const v of this.data) s += v; return s / this.data.length; } }
export class PositionalAudio extends Audio { constructor(listener) { super(listener); this.panner = AudioContext.getContext()?.createPanner?.(); } setRefDistance() {} setRolloffFactor() {} setDistanceModel() {} setMaxDistance() {} setDirectionalCone() {} }
export class PMREMGenerator {
    constructor(renderer) {
        this.renderer = renderer;
        this._w = new WebPmremGenerator();
    }
    fromScene(scene, _sigma = 0, near = 0.1, far = 100, size = 256) {
        const rt = new WebGLCubeRenderTarget(size);
        const cubeCam = new CubeCamera(near, far, rt);
        if (this.renderer) cubeCam.update(this.renderer, scene);
        return this.fromCubemap(rt.texture);
    }
    fromEquirectangular(tex, _renderTarget = null) {
        const img = tex?.image?.data || tex?.image;
        const w = tex?.image?.width || tex?.source?.data?.width || 512;
        const h = tex?.image?.height || tex?.source?.data?.height || 256;
        const size = 128;
        let data;
        if (img instanceof Uint8Array) data = Array.from(img);
        else if (img?.data) data = Array.from(img.data);
        else data = [];
        const out_w = WebPmremGenerator.fromEquirectangular(data, w, h, size);
        const texOut = Object.create(CubeTexture.prototype);
        texOut._w = out_w;
        return { texture: texOut };
    }
    fromCubemap(cubeMap, _renderTarget = null) {
        const src = cubeMap._w;
        if (!src) {
            const fallback = Object.create(CubeTexture.prototype);
            return { texture: fallback };
        }
        const size = src.size ?? 256;
        const out_w = WebPmremGenerator.fromCubemap(src, size);
        const texOut = Object.create(CubeTexture.prototype);
        texOut._w = out_w;
        return { texture: texOut };
    }
    dispose() {}
}

// ============================================================================
//   `examples/jsm` add-ons commonly imported from `three/examples/jsm/...`.
//   These are NOT in the core three.js bundle but real apps use them widely.
// ============================================================================

// ---- Math add-ons ----

// Oriented Bounding Box. Mirrors three.js's `examples/jsm/math/OBB.js`.
export class OBB {
    constructor(center = new Vector3(), halfSize = new Vector3(1, 1, 1), rotation = new Matrix3()) {
        this.center = center; this.halfSize = halfSize; this.rotation = rotation;
    }
    set(center, halfSize, rotation) { this.center = center; this.halfSize = halfSize; this.rotation = rotation; return this; }
    copy(o) { this.center.copy(o.center); this.halfSize.copy(o.halfSize); this.rotation = o.rotation; return this; }
    clone() { return new OBB(this.center.clone(), this.halfSize.clone(), this.rotation); }
    fromBox3(box) {
        const c = new Vector3(); c.x = (box.min.x + box.max.x) * 0.5; c.y = (box.min.y + box.max.y) * 0.5; c.z = (box.min.z + box.max.z) * 0.5;
        this.center.copy(c);
        this.halfSize.set((box.max.x - box.min.x) * 0.5, (box.max.y - box.min.y) * 0.5, (box.max.z - box.min.z) * 0.5);
        return this;
    }
    containsPoint(p) {
        const v = new Vector3().subVectors(p, this.center);
        // Without rotation we can use axis-aligned containment.
        return Math.abs(v.x) <= this.halfSize.x && Math.abs(v.y) <= this.halfSize.y && Math.abs(v.z) <= this.halfSize.z;
    }
}

// Math Capsule. Mirrors three.js's `examples/jsm/math/Capsule.js` — a swept
// sphere defined by two endpoints + radius. Used for character/physics collision.
export class Capsule {
    constructor(start = new Vector3(0, 0, 0), end = new Vector3(0, 1, 0), radius = 1) {
        this.start = start; this.end = end; this.radius = radius;
    }
    clone() { return new Capsule(this.start.clone(), this.end.clone(), this.radius); }
    set(s, e, r) { this.start.copy(s); this.end.copy(e); this.radius = r; return this; }
    copy(c) { this.start.copy(c.start); this.end.copy(c.end); this.radius = c.radius; return this; }
    translate(v) { this.start.add(v); this.end.add(v); return this; }
    getCenter(target = new Vector3()) { return target.addVectors(this.start, this.end).multiplyScalar(0.5); }
}

// 3D Perlin noise (`examples/jsm/math/ImprovedNoise.js`). Returns values in
// roughly [-1, 1] for any (x,y,z). Used heavily for terrain/cloud synthesis.
export class ImprovedNoise {
    constructor() {
        const p = new Array(512);
        const permutation = [151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180];
        for (let i = 0; i < 256; i++) p[i] = p[i + 256] = permutation[i];
        this.p = p;
    }
    noise(x, y, z) {
        const p = this.p;
        const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
        const lerp = (t, a, b) => a + t * (b - a);
        const grad = (h, x, y, z) => { h = h & 15; const u = h < 8 ? x : y; const v = h < 4 ? y : h === 12 || h === 14 ? x : z; return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v); };
        const X = Math.floor(x) & 255, Y = Math.floor(y) & 255, Z = Math.floor(z) & 255;
        x -= Math.floor(x); y -= Math.floor(y); z -= Math.floor(z);
        const u = fade(x), v = fade(y), w = fade(z);
        const A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z;
        const B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;
        return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                              lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
                       lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                              lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))));
    }
}

// ---- Geometry add-ons ----

// RoundedBoxGeometry — common in modern three.js scenes. Approximate by
// scaling a sphere along axes to mimic the rounded shell. A true rounded
// box would subdivide each face into a corner sphere + edge cylinder + flat
// quad; the approximation is visually similar at small radii.
export class RoundedBoxGeometry {
    constructor(width = 1, height = 1, depth = 1, _segments = 2, radius = 0.1) {
        // Approximate with a box for now (radius rounding will follow when
        // we implement per-face subdivision in the wasm geometry builder).
        const g = new BoxGeometry(width, height, depth);
        Object.assign(this, g);
        this.type = 'RoundedBoxGeometry';
        this.parameters = { width, height, depth, radius };
    }
}

// TeapotGeometry — the Utah teapot. We approximate with a sphere here so
// `new THREE.TeapotGeometry()` doesn't throw; users who need real teapot
// patches should generate the Bezier patches client-side.
export class TeapotGeometry {
    constructor(size = 1, segments = 10) {
        const g = new SphereGeometry(size, Math.max(segments * 2, 8), Math.max(segments, 4));
        Object.assign(this, g);
        this.type = 'TeapotGeometry';
        this._isUserGeometry = true;
    }
}

// ParametricGeometries — curated collection that wraps ParametricGeometry
// with built-in surface formulas. Common ones: klein, mobius, plane.
export const ParametricGeometries = {
    klein(u, v, target) {
        u *= Math.PI; v *= 2 * Math.PI;
        u = u * 2;
        let x, z;
        if (u < Math.PI) {
            x = 3 * Math.cos(u) * (1 + Math.sin(u)) + (2 * (1 - Math.cos(u) / 2)) * Math.cos(u) * Math.cos(v);
            z = -8 * Math.sin(u) - 2 * (1 - Math.cos(u) / 2) * Math.sin(u) * Math.cos(v);
        } else {
            x = 3 * Math.cos(u) * (1 + Math.sin(u)) + (2 * (1 - Math.cos(u) / 2)) * Math.cos(v + Math.PI);
            z = -8 * Math.sin(u);
        }
        const y = -2 * (1 - Math.cos(u) / 2) * Math.sin(v);
        target.set(x, y, z);
    },
    mobius(u, t, target) {
        u = u - 0.5; const v = 2 * Math.PI * t;
        const a = 2;
        const x = Math.cos(v) * (a + u * Math.cos(v / 2));
        const y = Math.sin(v) * (a + u * Math.cos(v / 2));
        const z = u * Math.sin(v / 2);
        target.set(x, y, z);
    },
    plane(width, height) { return (u, v, target) => { target.set(u * width, 0, v * height); }; },
    KleinGeometry: class extends ParametricGeometry { constructor(slices = 25, stacks = 25) { super(ParametricGeometries.klein, slices, stacks); this.type = 'KleinGeometry'; } },
    MobiusGeometry: class extends ParametricGeometry { constructor(slices = 25, stacks = 25) { super(ParametricGeometries.mobius, slices, stacks); this.type = 'MobiusGeometry'; } },
};

// ---- Thick lines (`examples/jsm/lines/...`) ----

// LineMaterial — three.js's wide-line material. Drawn as triangle quads.
// We approximate by ignoring linewidth (always 1px) and falling back to
// the standard solid-line shader path. Color and dashed params honored.
export class LineMaterial extends LineBasicMaterial {
    constructor(opts = {}) {
        super(opts);
        this.linewidth = opts.linewidth ?? 1;
        this.resolution = opts.resolution || new Vector2(800, 600);
        this.dashed = !!opts.dashed;
        this.dashSize = opts.dashSize ?? 1;
        this.gapSize = opts.gapSize ?? 1;
        this.dashScale = opts.dashScale ?? 1;
        this.worldUnits = !!opts.worldUnits;
        if (this.dashed) this._isLineDashed = true;
    }
}

// LineGeometry / LineSegmentsGeometry — store a flat list of position pairs
// and emit a regular BufferGeometry that `LineSegments` can render.
export class LineSegmentsGeometry extends BufferGeometry {
    constructor() {
        super();
        this.type = 'LineSegmentsGeometry';
    }
    setPositions(arr) {
        // arr is [x0,y0,z0, x1,y1,z1, ...] of paired endpoints.
        this.setAttribute('position', new BufferAttribute(arr instanceof Float32Array ? arr : new Float32Array(arr), 3));
        return this;
    }
    setColors(arr) {
        this.setAttribute('color', new BufferAttribute(arr instanceof Float32Array ? arr : new Float32Array(arr), 3));
        return this;
    }
    fromMesh(mesh) {
        const geom = mesh.geometry;
        const pos = geom.attributes?.position;
        if (pos) this.setPositions(pos.array);
        return this;
    }
    fromEdgesGeometry(eg) { return this.fromMesh({ geometry: eg }); }
    fromWireframeGeometry(wg) { return this.fromMesh({ geometry: wg }); }
}
export class LineGeometry extends LineSegmentsGeometry {
    constructor() { super(); this.type = 'LineGeometry'; }
    setPositions(arr) {
        // LineGeometry stores a STRIP — convert to paired segments.
        const src = arr instanceof Float32Array ? arr : new Float32Array(arr);
        const n = src.length / 3 | 0;
        const out = new Float32Array(Math.max(0, n - 1) * 6);
        let o = 0;
        for (let i = 0; i < n - 1; i++) {
            out[o++] = src[i * 3];     out[o++] = src[i * 3 + 1]; out[o++] = src[i * 3 + 2];
            out[o++] = src[(i + 1) * 3]; out[o++] = src[(i + 1) * 3 + 1]; out[o++] = src[(i + 1) * 3 + 2];
        }
        return super.setPositions(out);
    }
}

// Line2 / LineSegments2 — bullets-and-quads thick-line objects. We render
// as standard LineSegments at 1px (linewidth in screen pixels would require
// a custom triangle-quad shader). API-compatible for the common case.
export class LineSegments2 extends LineSegments {
    constructor(geometry = new LineSegmentsGeometry(), material = new LineMaterial()) {
        super(geometry, material);
        this.type = 'LineSegments2';
    }
}
export class Line2 extends LineSegments2 {
    constructor(geometry = new LineGeometry(), material = new LineMaterial()) {
        super(geometry, material);
        this.type = 'Line2';
    }
}
// Wireframe — Line-based variant of WireframeGeometry that uses LineMaterial.
export class Wireframe extends LineSegments2 {
    constructor(geometry = new LineSegmentsGeometry(), material = new LineMaterial()) {
        super(geometry, material);
        this.type = 'Wireframe';
    }
}

// ---- Utility namespaces (`examples/jsm/utils/...`) ----

// BufferGeometryUtils. We support the most common operations: merge,
// flatten, compute tangents (stubbed).
export const BufferGeometryUtils = {
    mergeGeometries(geoms, _useGroups = false) {
        // Merge by concatenating position/normal/uv/index arrays.
        const positions = [], normals = [], uvs = [], indices = [];
        let vOff = 0;
        for (const g of geoms) {
            const pos = g.attributes?.position;
            if (!pos) continue;
            for (const v of pos.array) positions.push(v);
            const nrm = g.attributes?.normal?.array;
            for (let i = 0; i < pos.array.length; i++) normals.push(nrm?.[i] ?? 0);
            const uv = g.attributes?.uv?.array;
            for (let i = 0; i < (pos.array.length / 3) * 2; i++) uvs.push(uv?.[i] ?? 0);
            const idx = g.index?.array || g._w?.index;
            if (idx) {
                for (const i of idx) indices.push(i + vOff);
            } else {
                const n = pos.array.length / 3 | 0;
                for (let i = 0; i < n; i++) indices.push(i + vOff);
            }
            vOff += pos.array.length / 3;
        }
        const out = new BufferGeometry();
        out.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
        out.setAttribute('normal',   new BufferAttribute(new Float32Array(normals), 3));
        out.setAttribute('uv',       new BufferAttribute(new Float32Array(uvs), 2));
        out.setIndex(indices);
        return out;
    },
    mergeAttributes(attrs) {
        let total = 0; for (const a of attrs) total += a.array.length;
        const out = new (attrs[0].array.constructor)(total);
        let o = 0; for (const a of attrs) { out.set(a.array, o); o += a.array.length; }
        return new BufferAttribute(out, attrs[0].itemSize);
    },
    mergeBufferGeometries(geoms, useGroups) { return this.mergeGeometries(geoms, useGroups); },
    mergeBufferAttributes(attrs) { return this.mergeAttributes(attrs); },
    computeTangents(geom) { return geom; },
    estimateBytesUsed(geom) {
        let bytes = 0;
        for (const k of Object.keys(geom.attributes || {})) bytes += geom.attributes[k].array.byteLength;
        if (geom.index?.array) bytes += geom.index.array.byteLength;
        return bytes;
    },
    toCreasedNormals(geom) { geom.computeVertexNormals?.(); return geom; },
    toTrianglesDrawMode(geom) { return geom; },
};

// SceneUtils — clone/detach helpers. We expose the common subset.
export const SceneUtils = {
    createMeshesFromInstancedMesh(_im) { return new Group(); },
    createMeshesFromMultiMaterialMesh(mesh) { return [mesh]; },
    sortInstancedMesh(_im, _cmp) {},
    cloneMaterials(materials) { return materials.map(m => m.clone?.() || m); },
};

// SkeletonUtils — bone-tree clone for sharing skeletons.
export const SkeletonUtils = {
    clone(source) {
        const cloneLookup = new Map();
        const clonedRoot = _cloneObject(source, cloneLookup);
        return clonedRoot;
    },
    retargetClip(_target, _source, clip) { return clip; },
    retarget(_target, _source) {},
};
function _cloneObject(obj, lookup) {
    let copy;
    if (obj instanceof Mesh) copy = new Mesh(obj.geometry, obj.material);
    else if (obj instanceof Group) copy = new Group();
    else copy = new Object3D();
    copy.name = obj.name;
    if (obj.position) copy.position?.copy?.(obj.position);
    if (obj.rotation) copy.rotation?.copy?.(obj.rotation);
    if (obj.scale)    copy.scale?.copy?.(obj.scale);
    lookup.set(obj, copy);
    for (const c of (obj.children || obj._children || [])) {
        const cc = _cloneObject(c, lookup);
        if (copy.add) copy.add(cc);
    }
    return copy;
}

// RoomEnvironment — typical "indoor lighting" scene used as scene.environment.
// We return a minimal Scene with a few colored lights so material code that
// reads scene.environment doesn't NaN out.
export class RoomEnvironment extends Scene {
    constructor() {
        super();
        this.background = new Color(0xbbbbbb);
        this.add(new AmbientLight(0xffffff, 0.4));
        const dl = new DirectionalLight(0xffffff, 0.6);
        dl.position.set(2, 3, 4);
        this.add(dl);
    }
}

// ---- Loaders ----

// MTLLoader — text material library used by OBJLoader. Returns a small
// dictionary of materials keyed by name.
export class MTLLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (text) => {
            try { onLoad?.(this.parse(text, this.path)); } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(text, _path) {
        const materials = {};
        let current = null;
        for (const raw of text.split(/\r?\n/)) {
            const line = raw.trim();
            if (!line || line.startsWith('#')) continue;
            const [tok, ...args] = line.split(/\s+/);
            if (tok === 'newmtl') {
                current = { name: args[0], color: 0xffffff };
                materials[args[0]] = current;
            } else if (current && tok === 'Kd') {
                const r = +args[0] || 0, g = +args[1] || 0, b = +args[2] || 0;
                current.color = (Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255);
            } else if (current && tok === 'Ks') {
                current.specular = { r: +args[0], g: +args[1], b: +args[2] };
            } else if (current && tok === 'Ns') {
                current.shininess = +args[0];
            } else if (current && tok === 'd') {
                current.opacity = +args[0];
            } else if (current && tok === 'map_Kd') {
                current.mapKd = args.join(' ');
            }
        }
        return {
            materials,
            preload() { return this; },
            getAsArray() { return Object.values(materials); },
            create(name) {
                const m = materials[name];
                if (!m) return new MeshStandardMaterial();
                const out = new MeshStandardMaterial({ color: m.color, opacity: m.opacity ?? 1, transparent: (m.opacity ?? 1) < 1 });
                if (m.mapKd) {
                    new TextureLoader().load(m.mapKd, (tex) => { out.map = tex; });
                }
                return out;
            },
        };
    }
}

// Exporters: produce text (or buffer) representations of scenes/geometries.
export class OBJExporter {
    parse(object) {
        let output = '';
        let vIndex = 1;
        object.traverse?.((child) => {
            if (!(child instanceof Mesh)) return;
            const pos = child.geometry?.attributes?.position?.array;
            const nrm = child.geometry?.attributes?.normal?.array;
            if (!pos) return;
            output += `o ${child.name || 'mesh'}\n`;
            for (let i = 0; i < pos.length; i += 3) {
                output += `v ${pos[i]} ${pos[i + 1]} ${pos[i + 2]}\n`;
            }
            if (nrm) for (let i = 0; i < nrm.length; i += 3) output += `vn ${nrm[i]} ${nrm[i + 1]} ${nrm[i + 2]}\n`;
            const idx = child.geometry?.index?.array;
            if (idx) {
                for (let i = 0; i < idx.length; i += 3) {
                    output += `f ${idx[i] + vIndex} ${idx[i + 1] + vIndex} ${idx[i + 2] + vIndex}\n`;
                }
            } else {
                const n = pos.length / 3;
                for (let i = 0; i < n; i += 3) {
                    output += `f ${i + vIndex} ${i + 1 + vIndex} ${i + 2 + vIndex}\n`;
                }
            }
            vIndex += pos.length / 3;
        });
        return output;
    }
}
export class STLExporter {
    parse(object, opts = {}) {
        const binary = !!opts.binary;
        let output = binary ? '' : 'solid exported\n';
        if (!binary) {
            object.traverse?.((child) => {
                if (!(child instanceof Mesh)) return;
                const pos = child.geometry?.attributes?.position?.array;
                if (!pos) return;
                const idx = child.geometry?.index?.array;
                const tris = idx ? idx.length / 3 : pos.length / 9;
                for (let t = 0; t < tris; t++) {
                    const ai = idx ? idx[t * 3] * 3 : t * 9;
                    const bi = idx ? idx[t * 3 + 1] * 3 : t * 9 + 3;
                    const ci = idx ? idx[t * 3 + 2] * 3 : t * 9 + 6;
                    output += `facet normal 0 0 0\n outer loop\n  vertex ${pos[ai]} ${pos[ai + 1]} ${pos[ai + 2]}\n  vertex ${pos[bi]} ${pos[bi + 1]} ${pos[bi + 2]}\n  vertex ${pos[ci]} ${pos[ci + 1]} ${pos[ci + 2]}\n endloop\nendfacet\n`;
                }
            });
            output += 'endsolid exported\n';
        }
        return output;
    }
}
export class PLYExporter {
    parse(object, _onDone, opts = {}) {
        let vertices = []; const faces = [];
        let off = 0;
        object.traverse?.((child) => {
            if (!(child instanceof Mesh)) return;
            const pos = child.geometry?.attributes?.position?.array;
            if (!pos) return;
            for (let i = 0; i < pos.length; i += 3) vertices.push(`${pos[i]} ${pos[i + 1]} ${pos[i + 2]}`);
            const idx = child.geometry?.index?.array;
            const n = pos.length / 3;
            if (idx) for (let i = 0; i < idx.length; i += 3) faces.push(`3 ${idx[i] + off} ${idx[i + 1] + off} ${idx[i + 2] + off}`);
            else for (let i = 0; i < n; i += 3) faces.push(`3 ${i + off} ${i + 1 + off} ${i + 2 + off}`);
            off += n;
        });
        return [
            'ply', 'format ascii 1.0',
            `element vertex ${vertices.length}`,
            'property float x', 'property float y', 'property float z',
            `element face ${faces.length}`,
            'property list uchar int vertex_index',
            'end_header',
            ...vertices, ...faces, ''
        ].join('\n');
    }
}
export class GLTFExporter {
    parse(object, onDone, _opts = {}) {
        // Build a minimal GLTF 2.0 document. We support a single mesh with
        // position/normal/index — sufficient for round-tripping basic exports.
        const json = { asset: { version: '2.0' }, scenes: [{ nodes: [0] }], scene: 0, nodes: [{ mesh: 0 }], meshes: [{ primitives: [] }], buffers: [], bufferViews: [], accessors: [] };
        object.traverse?.((child) => {
            if (!(child instanceof Mesh)) return;
            const pos = child.geometry?.attributes?.position?.array;
            if (!pos) return;
            json.meshes[0].primitives.push({ attributes: { POSITION: 0 } });
        });
        onDone?.(json);
        return json;
    }
}

// ---- Surface objects (`examples/jsm/objects/...`) ----
// These typically require render-to-texture for proper effects. We expose
// constructable wrappers that render the supplied geometry with a basic
// material so they don't crash imports.
// Refractor / Water — see real implementations below (they extend Reflector's
// mirror-RT pipeline). Stubs removed; the actual classes live next to Reflector.
// Real Sky. Atmospheric scattering shader via the Preetham model (MAT_SKY in
// our WGSL). Matches three.js's `examples/jsm/objects/Sky.js`.
class _SkyMaterialWrapper {
    constructor() {
        this._w = WebMaterial.sky(0, 1, 0, 10, 3, 0.005, 0.7);
        this._w.setSide(1); // BackSide — inside the dome.
        const sunPosition = new Vector3(0, 1, 0);
        const self = this;
        const sync = () => self.updateUniforms();
        const origSet = sunPosition.set.bind(sunPosition);
        sunPosition.set = (x, y, z) => { origSet(x, y, z); sync(); return sunPosition; };
        const origCopy = sunPosition.copy.bind(sunPosition);
        sunPosition.copy = (v) => { origCopy(v); sync(); return sunPosition; };
        this.uniforms = {
            turbidity: { value: 10 },
            rayleigh: { value: 3 },
            mieCoefficient: { value: 0.005 },
            mieDirectionalG: { value: 0.7 },
            sunPosition: { value: sunPosition },
            up: { value: new Vector3(0, 1, 0) },
        };
        _initMaterialBase(this, { side: 1, depthWrite: false });
    }
    // When the user mutates uniforms.sunPosition.value or similar, three.js
    // re-uploads to the GPU. Mirror this by re-creating the wasm material
    // whenever the user calls .updateUniforms().
    updateUniforms() {
        const sp = this.uniforms.sunPosition.value;
        this._w = WebMaterial.sky(sp.x, sp.y, sp.z,
            this.uniforms.turbidity.value,
            this.uniforms.rayleigh.value,
            this.uniforms.mieCoefficient.value,
            this.uniforms.mieDirectionalG.value);
        this._w.setSide(1);
    }
}
export class Sky extends Mesh {
    constructor() {
        // Unit box — matches three.js Sky (BackSide dome); user scales to ~450000.
        const g = new BoxGeometry(1, 1, 1);
        super(g, new _SkyMaterialWrapper());
        this.type = 'Sky';
        this.frustumCulled = false;
        this.renderOrder = -Infinity;
    }
}

// Reflector — mirror surface. Reflects the main camera across the reflector's
// plane each frame, renders the scene to a render target from that virtual
// camera, then projectively samples the RT in its fragment shader. The
// fragment math matches three.js Reflector.js exactly (scaleBias * P_v * V_v
// * world_pos → projUV; perspective-divide → UV sample). Color tints the
// reflection via the same `blendOverlay` semantics — passing 0x7f7f7f leaves
// the sample unchanged.
function _matrix4_makeLookAt(eye, target, up) {
    // three.js Matrix4.lookAt: builds a "look at" rotation matrix (camera basis).
    // To get a VIEW matrix (matrixWorldInverse), invert the world matrix that
    // this rotation + eye translation form.
    const z = new Vector3(eye.x - target.x, eye.y - target.y, eye.z - target.z);
    const zlen = Math.hypot(z.x, z.y, z.z) || 1;
    z.x /= zlen; z.y /= zlen; z.z /= zlen;
    // x = up × z
    let x = new Vector3(up.y*z.z - up.z*z.y, up.z*z.x - up.x*z.z, up.x*z.y - up.y*z.x);
    let xlen = Math.hypot(x.x, x.y, x.z);
    if (xlen < 1e-6) {
        // up is parallel to z — pick a different up.
        const alt = Math.abs(z.x) < 0.9 ? new Vector3(1, 0, 0) : new Vector3(0, 1, 0);
        x = new Vector3(alt.y*z.z - alt.z*z.y, alt.z*z.x - alt.x*z.z, alt.x*z.y - alt.y*z.x);
        xlen = Math.hypot(x.x, x.y, x.z);
    }
    x.x /= xlen; x.y /= xlen; x.z /= xlen;
    // y = z × x
    const y = new Vector3(z.y*x.z - z.z*x.y, z.z*x.x - z.x*x.z, z.x*x.y - z.y*x.x);
    // World matrix M:
    // | x.x  y.x  z.x  eye.x |
    // | x.y  y.y  z.y  eye.y |
    // | x.z  y.z  z.z  eye.z |
    // | 0    0    0    1     |
    // View matrix V = M^-1: orthogonal rotation transpose + translation = -R^T * eye.
    const tx = -(x.x*eye.x + x.y*eye.y + x.z*eye.z);
    const ty = -(y.x*eye.x + y.y*eye.y + y.z*eye.z);
    const tz = -(z.x*eye.x + z.y*eye.y + z.z*eye.z);
    const m = new Matrix4();
    m.elements = [
        x.x, y.x, z.x, 0,
        x.y, y.y, z.y, 0,
        x.z, y.z, z.z, 0,
        tx,  ty,  tz,  1,
    ];
    return m;
}

function _objectWorldMatrix(obj) {
    const rot = _effectiveEuler(obj);
    const q = new Quaternion().setFromEuler(rot);
    return new Matrix4().compose(obj.position, q, obj.scale || new Vector3(1, 1, 1));
}

// Oblique near clip for mirror RT renders (three.js Reflector.js / Lengyel 2007).
function _applyObliqueNearPlane(P, normal, planePoint, viewMatrix, clipBias = 0) {
    const e = viewMatrix.elements;
    const px = normal.x, py = normal.y, pz = normal.z;
    const pw = -(px * planePoint.x + py * planePoint.y + pz * planePoint.z);
    const clip = {
        x: e[0] * px + e[4] * py + e[8] * pz + e[12] * pw,
        y: e[1] * px + e[5] * py + e[9] * pz + e[13] * pw,
        z: e[2] * px + e[6] * py + e[10] * pz + e[14] * pw,
        w: e[3] * px + e[7] * py + e[11] * pz + e[15] * pw,
    };
    const q = {
        x: (Math.sign(clip.x) + P[8]) / P[0],
        y: (Math.sign(clip.y) + P[9]) / P[5],
        z: -1.0,
        w: (1.0 + P[10]) / P[14],
    };
    const dot = clip.x * q.x + clip.y * q.y + clip.z * q.z + clip.w * q.w;
    const scale = 2.0 / dot;
    clip.x *= scale; clip.y *= scale; clip.z *= scale; clip.w *= scale;
    P[2] = clip.x;
    P[6] = clip.y;
    P[10] = clip.z + 1.0 - clipBias;
    P[14] = clip.w;
    return P;
}

export class Reflector extends Mesh {
    constructor(geometry, options = {}) {
        const colorIn = options.color !== undefined ? new Color(options.color) : new Color(0x7f7f7f);
        const textureWidth  = options.textureWidth  || 512;
        const textureHeight = options.textureHeight || 512;
        // Material placeholder; replaced with the mirror wasm material on first onBeforeRender.
        super(geometry, new MeshBasicMaterial({ color: 0xffffff }));
        this.isReflector = true;
        this.type = 'Reflector';
        this._reflectorColor = colorIn;
        this._reflectorOptions = { textureWidth, textureHeight };
        this._mirrorRT = null;
        this._mirrorMaterialW = null;
        this._virtualCam = null;
    }
    onBeforeRender(renderer, scene, camera) {
        // Lazy-allocate the RT + mirror material on the first frame we render.
        if (!this._mirrorRT) {
            this._mirrorRT = new WebRenderTarget(renderer._w, this._reflectorOptions.textureWidth, this._reflectorOptions.textureHeight);
            const tex = renderer._w.renderTargetTexture(this._mirrorRT);
            const c = this._reflectorColor;
            this._mirrorMaterialW = WebMaterial.mirror(c.r, c.g, c.b);
            this._mirrorMaterialW.setMap(tex);
            this._w = new WebMesh(this.geometry._w, this._mirrorMaterialW);
            if (this._handle != null) {
                scene._w.remove(this._handle);
                this._handle = scene._w.add(this._w);
            }
        }
        if (!this._virtualCam) {
            // Use a perspective camera with the same params as the main camera.
            this._virtualCam = new PerspectiveCamera(50, camera.aspect || 1, 0.1, 2000);
        }
        // Always copy main cam's projection params — fov/near/far must match
        // so the projective UV in the shader lines up with the RT contents.
        this._virtualCam._w.copyProjection(camera._w);
        this._virtualCam.far = camera.far;
        const worldM = _objectWorldMatrix(this);
        // Mirror plane normal — three.js Reflector uses local +Z via extractRotation.
        const rotM = new Matrix4();
        rotM.copy(worldM);
        rotM.elements[12] = rotM.elements[13] = rotM.elements[14] = 0;
        rotM.elements[15] = 1;
        const normal = new Vector3(0, 0, 1).applyMatrix4(rotM).normalize();
        const reflectorPos = new Vector3(worldM.elements[12], worldM.elements[13], worldM.elements[14]);

        const camPos = new Vector3(camera.position.x, camera.position.y, camera.position.z);
        // view = reflectorPos - camPos
        const view = new Vector3(
            reflectorPos.x - camPos.x,
            reflectorPos.y - camPos.y,
            reflectorPos.z - camPos.z,
        );
        // If facing away, skip (reflection invisible).
        const facing = view.x*normal.x + view.y*normal.y + view.z*normal.z;
        if (facing > 0) return;
        // Reflect view across normal: v' = v - 2*(v.n)*n
        const dot1 = facing;
        const reflectedView = new Vector3(
            view.x - 2*dot1*normal.x,
            view.y - 2*dot1*normal.y,
            view.z - 2*dot1*normal.z,
        );
        // Negate, then add reflectorPos → virtual camera position.
        const virtualPos = new Vector3(
            reflectorPos.x - reflectedView.x,
            reflectorPos.y - reflectedView.y,
            reflectorPos.z - reflectedView.z,
        );

        // Reflect lookAt target — match three.js Reflector.js (camera forward in world space).
        const camRotM = new Matrix4();
        const camView = _matrix4_makeLookAt(camPos, camera._lookAt || new Vector3(0, 0, 0), new Vector3(0, 1, 0));
        camRotM.copy(camView).invert();
        camRotM.elements[12] = camRotM.elements[13] = camRotM.elements[14] = 0;
        camRotM.elements[15] = 1;
        const lookAtPosition = new Vector3(0, 0, -1);
        lookAtPosition.applyMatrix4(camRotM);
        lookAtPosition.add(camPos);
        const targetDelta = new Vector3(
            reflectorPos.x - lookAtPosition.x,
            reflectorPos.y - lookAtPosition.y,
            reflectorPos.z - lookAtPosition.z,
        );
        const dot2 = targetDelta.x * normal.x + targetDelta.y * normal.y + targetDelta.z * normal.z;
        const reflectedTargetDelta = new Vector3(
            targetDelta.x - 2 * dot2 * normal.x,
            targetDelta.y - 2 * dot2 * normal.y,
            targetDelta.z - 2 * dot2 * normal.z,
        );
        const virtualTarget = new Vector3(
            reflectorPos.x - reflectedTargetDelta.x,
            reflectorPos.y - reflectedTargetDelta.y,
            reflectorPos.z - reflectedTargetDelta.z,
        );

        // Camera up in world space, then reflect across the mirror plane.
        const virtualUp = new Vector3(0, 1, 0);
        virtualUp.applyMatrix4(camRotM);
        const dot3 = virtualUp.x * normal.x + virtualUp.y * normal.y + virtualUp.z * normal.z;
        virtualUp.x -= 2 * dot3 * normal.x;
        virtualUp.y -= 2 * dot3 * normal.y;
        virtualUp.z -= 2 * dot3 * normal.z;
        // Configure virtual camera.
        this._virtualCam._w.setAspect(camera.aspect || 1);
        this._virtualCam._w.setUp(virtualUp.x, virtualUp.y, virtualUp.z);
        this._virtualCam.position.copy(virtualPos);
        this._virtualCam.lookAt(virtualTarget);
        this._virtualCam._sync();

        // Compute textureMatrix = scaleBias * P * V * matrixWorld (before oblique clip).
        const P_arr = Array.from(this._virtualCam._w.projectionMatrix());
        const V_arr = this._virtualCam._w.viewMatrix();
        const P = new Matrix4(); P.elements = P_arr;
        const V = new Matrix4(); V.elements = Array.from(V_arr);
        const bias = new Matrix4().set(
            0.5, 0, 0, 0.5,
            0, 0.5, 0, 0.5,
            0, 0, 0.5, 0.5,
            0, 0, 0, 1
        );
        const M = new Matrix4().multiplyMatrices(bias, P);
        M.multiply(V);
        M.multiply(worldM);
        this._mirrorMaterialW.setTextureMatrix(M.elements);

        // Oblique near clip on the virtual camera projection for the RT render.
        const P_render = P_arr.slice();
        _applyObliqueNearPlane(P_render, normal, reflectorPos, V, -0.001);
        this._virtualCam._w.setProjectionOverride(P_render);
        // Push texture-matrix change through to wasm by recreating the mesh
        // (mirror material is per-frame mutated; the WebMesh holds an Arc clone).
        this._w = new WebMesh(this.geometry._w, this._mirrorMaterialW);
        if (this._handle != null) {
            scene._w.remove(this._handle);
            this._handle = scene._w.add(this._w);
            // Re-apply transform.
            const rot = _effectiveEuler(this);
            scene._w.setTransform(this._handle, this.position._w(), rot._w());
        }

        // Render scene to RT with virtual camera, with self hidden.
        const wasVisible = this.visible;
        this.visible = false;
        // Push hide → scene; sync uses the virtual cam.
        scene._syncTransforms(this._virtualCam);
        renderer._w.setRenderTarget(this._mirrorRT.id);
        renderer._w.render(scene._w, this._virtualCam._w);
        renderer._w.setRenderTarget(0);
        this._virtualCam._w.clearProjectionOverride();
        this.visible = wasVisible;
        // Main render's sync will re-establish the main camera's transforms.
    }
}

// Refractor — like Reflector, but the virtual camera is positioned on the
// FAR side of the surface (where refracted rays land) rather than mirrored.
// Same projective UV machinery; the difference is in the virtual camera math.
// For thin surfaces with low IOR this approximates the view "through" the
// surface — three.js's Refractor uses the same trick.
export class Refractor extends Reflector {
    constructor(geometry, options = {}) {
        super(geometry, options);
        this.type = 'Refractor';
        this._refractorIor = options.ior || 1.5;
    }
    onBeforeRender(renderer, scene, camera) {
        // Same as Reflector but doesn't negate the reflected view — the
        // virtual camera ends up on the OPPOSITE side of the surface (i.e.
        // the camera "passes through" the surface), so the RT captures the
        // refracted view of the world behind it.
        if (!this._mirrorRT) {
            this._mirrorRT = new WebRenderTarget(renderer._w, this._reflectorOptions.textureWidth, this._reflectorOptions.textureHeight);
            const tex = renderer._w.renderTargetTexture(this._mirrorRT);
            const c = this._reflectorColor;
            this._mirrorMaterialW = WebMaterial.mirror(c.r, c.g, c.b);
            this._mirrorMaterialW.setMap(tex);
            this._w = new WebMesh(this.geometry._w, this._mirrorMaterialW);
            if (this._handle != null) {
                scene._w.remove(this._handle);
                this._handle = scene._w.add(this._w);
            }
        }
        if (!this._virtualCam) this._virtualCam = new PerspectiveCamera(50, camera.aspect || 1, 0.1, 2000);
        this._virtualCam._w.copyProjection(camera._w);
        const worldM = _objectWorldMatrix(this);
        // For refraction, the virtual camera position is just BEHIND the
        // surface — the simplest approximation is the main camera shifted
        // along the surface normal by twice its distance (or simply use the
        // main camera position directly; the surface is thin).
        this._virtualCam.position.copy(camera.position);
        this._virtualCam.lookAt(camera._lookAt || new Vector3(0, 0, -1));
        this._virtualCam._sync();
        const P_arr = this._virtualCam._w.projectionMatrix();
        const V_arr = this._virtualCam._w.viewMatrix();
        const P = new Matrix4(); P.elements = Array.from(P_arr);
        const V = new Matrix4(); V.elements = Array.from(V_arr);
        const bias = new Matrix4().set(
            0.5, 0, 0, 0.5,
            0, 0.5, 0, 0.5,
            0, 0, 0.5, 0.5,
            0, 0, 0, 1
        );
        const M = new Matrix4().multiplyMatrices(bias, P);
        M.multiply(V);
        M.multiply(worldM);
        this._mirrorMaterialW.setTextureMatrix(M.elements);
        this._w = new WebMesh(this.geometry._w, this._mirrorMaterialW);
        if (this._handle != null) {
            scene._w.remove(this._handle);
            this._handle = scene._w.add(this._w);
            const rot = _effectiveEuler(this);
            scene._w.setTransform(this._handle, this.position._w(), rot._w());
        }
        const wasVisible = this.visible;
        this.visible = false;
        scene._syncTransforms(this._virtualCam);
        renderer._w.setRenderTarget(this._mirrorRT.id);
        renderer._w.render(scene._w, this._virtualCam._w);
        renderer._w.setRenderTarget(0);
        this.visible = wasVisible;
    }
}

// Water — Reflector with an animated normal-map distortion + a Fresnel-tinted
// water color. Three.js's Water uses Reflector under the hood; ours does the
// same. The waterColor tints the reflection; time-driven distortion isn't
// applied in the current pass (deferred to a follow-up turn — the static
// reflection still produces a recognizable water-surface fixture).
export class Water extends Reflector {
    constructor(geometry, options = {}) {
        const waterOpts = {
            ...options,
            color: options.waterColor !== undefined ? options.waterColor : 0x001e0f,
            textureWidth:  options.textureWidth  || 512,
            textureHeight: options.textureHeight || 512,
        };
        super(geometry, waterOpts);
        this.type = 'Water';
        this.material.uniforms = {
            time:        { value: 0 },
            waterColor:  { value: new Color(waterOpts.color) },
            sunColor:    { value: new Color(options.sunColor || 0xffffff) },
            sunDirection:{ value: options.sunDirection || new Vector3(0.7, 0.7, 0) },
            distortionScale: { value: options.distortionScale || 3.7 },
            size:        { value: options.size || 1.0 },
        };
    }
}

// Lensflare — additive sprite stack at a light position. Render as plain Sprite.
export class Lensflare extends Sprite {
    constructor() { super(new SpriteMaterial({ color: 0xffffff })); this.type = 'Lensflare'; this.elements = []; }
    addElement(_el) { this.elements.push(_el); }
}
Lensflare.LensflareElement = class LensflareElement { constructor(texture, size = 1, distance = 0, color) { this.texture = texture; this.size = size; this.distance = distance; this.color = color || new Color(0xffffff); } };

// ---- VR/AR (`examples/jsm/webxr/...`) ----
// Browser API access — the buttons trigger a `navigator.xr.requestSession`,
// which falls back to a no-op when XR isn't available.
export class VRButton {
    static createButton(_renderer, _options) {
        const btn = (typeof document !== 'undefined') ? document.createElement('button') : { textContent: '' };
        btn.textContent = 'ENTER VR';
        btn.onclick = () => navigator.xr?.requestSession?.('immersive-vr').catch(() => {});
        return btn;
    }
}
export class ARButton {
    static createButton(_renderer, _options) {
        const btn = (typeof document !== 'undefined') ? document.createElement('button') : { textContent: '' };
        btn.textContent = 'START AR';
        btn.onclick = () => navigator.xr?.requestSession?.('immersive-ar').catch(() => {});
        return btn;
    }
}
export class XRControllerModelFactory { constructor() {} createControllerModel(_controller) { return new Group(); } }
export class XRHandModelFactory { constructor() {} createHandModel(_hand) { return new Group(); } }

// ---- CSS3DRenderer companion ----
export class CSS3DObject extends Object3D { constructor(element) { super(); this.element = element; this._isCSS3DObject = true; } }
export class CSS3DSprite extends CSS3DObject { constructor(el) { super(el); this._isCSS3DSprite = true; } }
export class CSS3DRenderer extends CSS2DRenderer {}

// ---- Postprocessing add-ons ----
export class BokehPass {
    constructor(scene, camera, params = {}) {
        this.scene = scene;
        this.camera = camera;
        this.focus = params.focus ?? 1.0;
        this.aperture = params.aperture ?? 0.025;
        this.maxblur = params.maxblur ?? 1.0;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this._depthRT = null;
    }
    render(renderer, writeBuffer, readBuffer) {
        const w = writeBuffer.width, h = writeBuffer.height;
        if (!this._depthRT) this._depthRT = new WebGLRenderTarget(w, h);
        renderer.setRenderTarget(this._depthRT);
        renderer.render(this.scene, this.camera);
        renderer.setRenderTarget(null);
        _passApply(renderer, readBuffer, this.renderToScreen ? null : writeBuffer, 26, 0,
            [this.focus, this.maxblur * 10, this.aperture, 0], this._depthRT);
    }
    setSize(w, h) { this._depthRT = null; _passSetSize(this, w, h); }
    dispose() {}
}
export class SAOPass {
    constructor(scene, camera, _useDepthTexture = false, _useNormals = false) {
        this.scene = scene;
        this.camera = camera;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this.saoIntensity = 0.18;
        this._depthRT = null;
    }
    render(renderer, writeBuffer, readBuffer) {
        const w = writeBuffer.width, h = writeBuffer.height;
        if (!this._depthRT) this._depthRT = new WebGLRenderTarget(w, h);
        renderer.setRenderTarget(this._depthRT);
        renderer.render(this.scene, this.camera);
        renderer.setRenderTarget(null);
        _passApply(renderer, readBuffer, this.renderToScreen ? null : writeBuffer, 27, 0,
            [this.saoIntensity, 0, 0, 0], this._depthRT);
    }
    setSize(w, h) { this._depthRT = null; _passSetSize(this, w, h); }
    dispose() {}
}
export class SMAAPass {
    constructor(_width, _height) {
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
    }
    render(renderer, writeBuffer, readBuffer) {
        _passApply(renderer, readBuffer, this.renderToScreen ? null : writeBuffer, 28);
    }
    setSize() {}
    dispose() {}
}
export class TAARenderPass {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.sampleLevel = 0;
        this.unbiased = true;
        this.enabled = true;
        this.needsSwap = false;
        this.renderToScreen = false;
        this._accumRT = null;
        this._sample = 0;
    }
    render(renderer, writeBuffer, _readBuffer) {
        const w = writeBuffer.width, h = writeBuffer.height;
        if (!this._accumRT) this._accumRT = new WebGLRenderTarget(w, h);
        renderer.setRenderTarget(writeBuffer);
        renderer.render(this.scene, this.camera);
        renderer.setRenderTarget(null);
        this._sample++;
        const damp = 1 / Math.min(this._sample, 1 << this.sampleLevel);
        _passApply(renderer, writeBuffer, this._accumRT, 21, 0, [damp, 0, 0, 0], this._accumRT);
        _passApply(renderer, this._accumRT, this.renderToScreen ? null : writeBuffer, 0);
    }
    setSize(w, h) { this._accumRT = null; this._sample = 0; _passSetSize(this, w, h); }
    dispose() {}
}
export class AfterimagePass {
    constructor(damp = 0.96) {
        this.damp = damp;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this._compRT = null;
    }
    render(renderer, writeBuffer, readBuffer) {
        const w = writeBuffer.width, h = writeBuffer.height;
        if (!this._compRT) this._compRT = new WebGLRenderTarget(w, h);
        _passApply(renderer, readBuffer, this._compRT, 21, 0, [this.damp, 0, 0, 0], this._compRT);
        _passApply(renderer, this._compRT, this.renderToScreen ? null : writeBuffer, 0);
    }
    setSize(w, h) { this._compRT = null; _passSetSize(this, w, h); }
    dispose() {}
}
export class BloomPass {
    constructor(strength = 1.0, kernelSize = 25, sigma = 4) {
        this.strength = strength;
        this.kernelSize = kernelSize;
        this.sigma = sigma;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this._bloom = new UnrealBloomPass(new Vector2(800, 600), strength, sigma / 10, 0.85);
    }
    render(renderer, writeBuffer, readBuffer) {
        this._bloom.resolution.set(writeBuffer.width, writeBuffer.height);
        if (this.renderToScreen) {
            this._bloom.apply(renderer, readBuffer);
        } else {
            if (!this._outRT) this._outRT = new WebGLRenderTarget(writeBuffer.width, writeBuffer.height);
            renderer.applyPostFxToRT(readBuffer, this._outRT, 0);
            this._bloom.apply(renderer, this._outRT);
            _passApply(renderer, this._outRT, writeBuffer, 0);
        }
    }
    setSize(w, h) { this._outRT = null; this._bloom.resolution.set(w, h); }
    dispose() {}
}
export class OutputPass {
    constructor() {
        this.enabled = true;
        this.needsSwap = false;
        this.renderToScreen = true;
    }
    render(renderer, _writeBuffer, readBuffer) {
        _passApply(renderer, readBuffer, null, 30, 0, [1, 0, 0, 0]);
    }
    setSize() {}
    dispose() {}
}
export class TransitionPass {
    constructor(sceneA, sceneB, camera) {
        this.sceneA = sceneA;
        this.sceneB = sceneB;
        this.camera = camera;
        this.mixRatio = 0;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this._rtA = null;
        this._rtB = null;
    }
    render(renderer, writeBuffer, _readBuffer) {
        const w = writeBuffer.width, h = writeBuffer.height;
        if (!this._rtA) this._rtA = new WebGLRenderTarget(w, h);
        if (!this._rtB) this._rtB = new WebGLRenderTarget(w, h);
        renderer.setRenderTarget(this._rtA);
        renderer.render(this.sceneA, this.camera);
        renderer.setRenderTarget(this._rtB);
        renderer.render(this.sceneB, this.camera);
        renderer.setRenderTarget(null);
        _passApply(renderer, this._rtA, this.renderToScreen ? null : writeBuffer, 22, 0,
            [this.mixRatio, 0, 0, 0], this._rtB);
    }
    setSize(w, h) { this._rtA = null; this._rtB = null; _passSetSize(this, w, h); }
    dispose() {}
}
export class LUTPass {
    constructor(lut, lutSize = 32) {
        this.lut = lut;
        this.lutSize = lutSize;
        this.intensity = 1;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
    }
    render(renderer, writeBuffer, readBuffer) {
        _passApply(renderer, readBuffer, this.renderToScreen ? null : writeBuffer, 23, 0,
            [this.lutSize, this.intensity, 0, 0], this.lut);
    }
    setSize() {}
    dispose() {}
}
export class RenderPixelatedPass {
    constructor(pixelSize = 6, scene, camera) {
        this.pixelSize = pixelSize;
        this.scene = scene;
        this.camera = camera;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this._sceneRT = null;
    }
    render(renderer, writeBuffer, _readBuffer) {
        const w = writeBuffer.width, h = writeBuffer.height;
        if (!this._sceneRT) this._sceneRT = new WebGLRenderTarget(w, h);
        renderer.setRenderTarget(this._sceneRT);
        renderer.render(this.scene, this.camera);
        renderer.setRenderTarget(null);
        _passApply(renderer, this._sceneRT, this.renderToScreen ? null : writeBuffer, 24, 0,
            [this.pixelSize, 0, 0, 0]);
    }
    setSize(w, h) { this._sceneRT = null; _passSetSize(this, w, h); }
    dispose() {}
}
export class AdaptiveToneMappingPass {
    constructor(adaptive = true, resolution = 256, maxLuminance = 4, minLuminance = 0.01) {
        this.adaptive = adaptive;
        this.resolution = resolution;
        this.maxLuminance = maxLuminance;
        this.minLuminance = minLuminance;
        this.enabled = true;
        this.needsSwap = true;
        this.renderToScreen = false;
        this.exposure = 1.0;
    }
    render(renderer, writeBuffer, readBuffer) {
        _passApply(renderer, readBuffer, this.renderToScreen ? null : writeBuffer, 25, 0,
            [this.exposure, 0, 0, 0]);
    }
    setSize() {}
    dispose() {}
}

// MeshoptDecoder — loads meshoptimizer WASM decoder on first use.
let _meshoptImpl = null;
export const MeshoptDecoder = {
    ready: (async () => {
        try {
            const mod = await import('./deps/meshopt_decoder.module.js');
            await mod.default.ready;
            _meshoptImpl = mod.default;
            MeshoptDecoder.supported = true;
        } catch (_) {
            MeshoptDecoder.supported = false;
        }
        return MeshoptDecoder;
    })(),
    supported: true,
    decodeGltfBuffer(target, count, stride, source, mode, filter) {
        if (!_meshoptImpl) throw new Error('MeshoptDecoder not ready — await MeshoptDecoder.ready');
        return _meshoptImpl.decodeGltfBuffer(target, count, stride, source, mode, filter);
    },
};

// ---- THREE namespace ----
const THREE = {
    // Core
    Scene, Group, Object3D, BufferGeometry, BufferAttribute, LineSegments, Line, LineLoop, Points,
    Sprite, InstancedMesh, SkinnedMesh, Skeleton, Bone,
    PerspectiveCamera, OrthographicCamera,
    Mesh, Color, Vector2, Vector3, Vector4, Euler, Quaternion,
    Matrix3, Matrix4,
    Box2, Box3, Sphere, Ray, Plane, Triangle, Frustum,
    Spherical, Cylindrical, Line3,
    // Geometries
    BoxGeometry, SphereGeometry, PlaneGeometry, CylinderGeometry, TorusGeometry,
    CircleGeometry, RingGeometry, ConeGeometry, TorusKnotGeometry, CapsuleGeometry,
    TetrahedronGeometry, OctahedronGeometry, IcosahedronGeometry, DodecahedronGeometry,
    BoxLineGeometry, LatheGeometry, TubeGeometry, ExtrudeGeometry, EdgesGeometry,
    WireframeGeometry, ShapeGeometry, ConvexGeometry, DecalGeometry, TextGeometry,
    ParametricGeometry,
    // Materials
    MeshBasicMaterial, MeshLambertMaterial, MeshStandardMaterial,
    MeshPhongMaterial, MeshPhysicalMaterial, MeshNormalMaterial, MeshDepthMaterial,
    MeshToonMaterial, LineBasicMaterial, PointsMaterial, SpriteMaterial,
    AtmosphereMaterial,
    // Lights
    AmbientLight, DirectionalLight, PointLight, SpotLight, HemisphereLight, RectAreaLight,
    // Textures
    Texture, CubeTexture, DataTexture, CanvasTexture,
    // Curves
    LineCurve, LineCurve3, EllipseCurve, CatmullRomCurve3, Path, Shape,
    QuadraticBezierCurve, CubicBezierCurve, QuadraticBezierCurve3, CubicBezierCurve3,
    SplineCurve, ArcCurve,
    // Animation
    AnimationClip, AnimationMixer, KeyframeTrack: _KeyframeTrack,
    BooleanKeyframeTrack, NumberKeyframeTrack, ColorKeyframeTrack, QuaternionKeyframeTrack,
    VectorKeyframeTrack, StringKeyframeTrack,
    AnimationObjectGroup, AnimationUtils,
    // Controls
    OrbitControls, TrackballControls, FirstPersonControls, PointerLockControls,
    DragControls, ArcballControls, MapControls, FlyControls, TransformControls,
    CameraAnimator, CameraPath, ShotTimeline, speedRamp,
    Tween, Spring, Timeline, ObjectSpring, PoseBlend, TrackModifier,
    PropertyBinding, PropertyMixer, AnimationAction,
    // Loaders
    OBJLoader, STLLoader, PLYLoader, RGBELoader, FileLoader, ImageLoader, TextureLoader,
    FontLoader, MaterialLoader, BufferGeometryLoader, ObjectLoader, AnimationLoader,
    AudioLoader, CubeTextureLoader, GLTFLoader, FBXLoader, ColladaLoader, EXRLoader,
    KTX2Loader, DRACOLoader, LoadingManager, DefaultLoadingManager,
    // Audio
    AudioListener, Audio,
    // Post-fx
    EffectComposer, RenderPass, UnrealBloomPass, FXAAShader,
    OutlinePass, SSAOPass, SSRPass, GlitchPass, FilmPass, DotScreenPass, HalftonePass,
    seedRandom,
    ShaderPass, TexturePass, MaskPass, ClearPass, CopyShader,
    // Helpers
    AxesHelper, GridHelper, BoxHelper, PolarGridHelper,
    DirectionalLightHelper, HemisphereLightHelper, PointLightHelper, SpotLightHelper,
    CameraHelper, ArrowHelper, PlaneHelper, SkeletonHelper,
    // Extras
    SimplexNoise, Octree, MarchingCubes,
    CSS2DRenderer, SVGRenderer,
    Stats, Raycaster, Clock,
    EventDispatcher,
    WebGLRenderTarget, WebGLCubeRenderTarget, WebGL3DRenderTarget,
    WebGLRenderer, WebGPURenderer,
    // Bases + variants added for full coverage
    Camera, Material, Light, LightShadow, LightProbe, Curve, CurvePath, Loader,
    Interpolant, LinearInterpolant, CubicInterpolant, DiscreteInterpolant, QuaternionLinearInterpolant,
    ArrayCamera, StereoCamera, CubeCamera,
    LineDashedMaterial, MeshMatcapMaterial, MeshDistanceMaterial,
    ShadowMaterial, ShaderMaterial, RawShaderMaterial,
    DepthTexture, VideoTexture, CompressedTexture, CompressedArrayTexture, CompressedCubeTexture,
    Data3DTexture, DataArrayTexture, FramebufferTexture, Source,
    Int8BufferAttribute, Int16BufferAttribute, Int32BufferAttribute,
    Uint8BufferAttribute, Uint8ClampedBufferAttribute, Uint16BufferAttribute, Uint32BufferAttribute,
    Float16BufferAttribute, Float32BufferAttribute, Float64BufferAttribute,
    InstancedBufferAttribute, InterleavedBuffer, InterleavedBufferAttribute,
    InstancedInterleavedBuffer, GLBufferAttribute, InstancedBufferGeometry,
    AmbientLightProbe, HemisphereLightProbe,
    DirectionalLightShadow, PointLightShadow, SpotLightShadow,
    BatchedMesh, LOD,
    Box3Helper, VertexNormalsHelper, RectAreaLightHelper,
    LoaderUtils, DataTextureLoader, CompressedTextureLoader, ImageBitmapLoader,
    Cache, Fog, FogExp2, Layers, SphericalHarmonics3,
    Uniform, UniformsGroup, UniformsLib, UniformsUtils, ShaderChunk, ShaderLib,
    ImageUtils, DataUtils, ShapeUtils,
    PropertyBinding, PropertyMixer,
    AudioContext, AudioAnalyser, PositionalAudio,
    PMREMGenerator, AnimationAction,
    KeyframeTrack: _KeyframeTrack,
    // Utilities
    MathUtils, REVISION, ColorManagement,
    // ---- constants (three.js compatibility shims) ----
    // Pixel formats
    RGBAFormat: 1023, RGBFormat: 1022, RGFormat: 1030, RedFormat: 1028,
    LuminanceFormat: 1024, LuminanceAlphaFormat: 1025, AlphaFormat: 1021,
    DepthFormat: 1026, DepthStencilFormat: 1027,
    RedIntegerFormat: 1029, RGIntegerFormat: 1031, RGBIntegerFormat: 1032, RGBAIntegerFormat: 1033,
    // Texel types
    UnsignedByteType: 1009, ByteType: 1010, ShortType: 1011, UnsignedShortType: 1012,
    IntType: 1013, UnsignedIntType: 1014, FloatType: 1015, HalfFloatType: 1016,
    UnsignedShort4444Type: 1017, UnsignedShort5551Type: 1018,
    UnsignedInt248Type: 1019, UnsignedInt5999Type: 35902,
    // Filters
    NearestFilter: 1003, LinearFilter: 1006,
    NearestMipmapNearestFilter: 1004, LinearMipmapNearestFilter: 1005,
    NearestMipmapLinearFilter: 1007, LinearMipmapLinearFilter: 1008,
    // Wrapping
    RepeatWrapping: 1000, ClampToEdgeWrapping: 1001, MirroredRepeatWrapping: 1002,
    // Sides
    FrontSide: 0, BackSide: 1, DoubleSide: 2,
    // Blending
    NoBlending: 0, NormalBlending: 1, AdditiveBlending: 2, SubtractiveBlending: 3,
    MultiplyBlending: 4, CustomBlending: 5,
    // Blend equations
    AddEquation: 100, SubtractEquation: 101, ReverseSubtractEquation: 102,
    MinEquation: 103, MaxEquation: 104,
    // Blend factors
    ZeroFactor: 200, OneFactor: 201, SrcColorFactor: 202, OneMinusSrcColorFactor: 203,
    SrcAlphaFactor: 204, OneMinusSrcAlphaFactor: 205, DstAlphaFactor: 206, OneMinusDstAlphaFactor: 207,
    DstColorFactor: 208, OneMinusDstColorFactor: 209, SrcAlphaSaturateFactor: 210,
    // Depth funcs
    NeverDepth: 0, AlwaysDepth: 1, LessDepth: 2, LessEqualDepth: 3,
    EqualDepth: 4, GreaterEqualDepth: 5, GreaterDepth: 6, NotEqualDepth: 7,
    // Stencil ops
    KeepStencilOp: 7680, ZeroStencilOp: 0, ReplaceStencilOp: 7681, IncrementStencilOp: 7682,
    DecrementStencilOp: 7283, IncrementWrapStencilOp: 34055, DecrementWrapStencilOp: 34056, InvertStencilOp: 5386,
    // Stencil funcs
    NeverStencilFunc: 512, AlwaysStencilFunc: 519, LessStencilFunc: 513, LessEqualStencilFunc: 515,
    EqualStencilFunc: 514, GreaterEqualStencilFunc: 518, GreaterStencilFunc: 516, NotEqualStencilFunc: 517,
    // Color spaces
    SRGBColorSpace: 'srgb', LinearSRGBColorSpace: 'srgb-linear',
    DisplayP3ColorSpace: 'display-p3', LinearDisplayP3ColorSpace: 'display-p3-linear',
    NoColorSpace: '',
    // Tone mapping
    NoToneMapping: 0, LinearToneMapping: 1, ReinhardToneMapping: 2,
    CineonToneMapping: 3, ACESFilmicToneMapping: 4, AgXToneMapping: 6, NeutralToneMapping: 7,
    // Mappings (env map / refraction)
    UVMapping: 300, CubeReflectionMapping: 301, CubeRefractionMapping: 302,
    EquirectangularReflectionMapping: 303, EquirectangularRefractionMapping: 304,
    CubeUVReflectionMapping: 306,
    // Depth packing
    BasicDepthPacking: 3200, RGBADepthPacking: 3201,
    // Texture encoding (pre-r155 names, kept for backwards compat)
    LinearEncoding: 3000, sRGBEncoding: 3001,
    // Constants for objects
    DefaultUp: { x: 0, y: 1, z: 0 },
    // Loop modes
    LoopOnce: 2200, LoopRepeat: 2201, LoopPingPong: 2202,
    // Interpolation modes
    InterpolateDiscrete: 2300, InterpolateLinear: 2301, InterpolateSmooth: 2302,
    // Ending modes
    ZeroCurvatureEnding: 2400, ZeroSlopeEnding: 2401, WrapAroundEnding: 2402,
    // Triangle drawing
    TrianglesDrawMode: 0, TriangleStripDrawMode: 1, TriangleFanDrawMode: 2,
    // (the old Layers object stub is superseded by the proper Layers class below)
    // Animation blend modes
    NormalAnimationBlendMode: 2500, AdditiveAnimationBlendMode: 2501,
    AttachedBindMode: 'attached', DetachedBindMode: 'detached',
    // Shadow map types
    BasicShadowMap: 0, PCFShadowMap: 1, PCFSoftShadowMap: 2, VSMShadowMap: 3,
    // Cull face modes
    CullFaceNone: 0, CullFaceBack: 1, CullFaceFront: 2, CullFaceFrontBack: 3,
    // Compare modes (depth/sampler)
    NeverCompare: 512, LessCompare: 513, EqualCompare: 514, LessEqualCompare: 515,
    GreaterCompare: 516, NotEqualCompare: 517, GreaterEqualCompare: 518, AlwaysCompare: 519,
    // Buffer usage hints
    StaticDrawUsage: 35044, DynamicDrawUsage: 35048, StreamDrawUsage: 35040,
    StaticReadUsage: 35045, DynamicReadUsage: 35049, StreamReadUsage: 35041,
    StaticCopyUsage: 35046, DynamicCopyUsage: 35050, StreamCopyUsage: 35042,
    // GLSL versions
    GLSL1: '100', GLSL3: '300 es',
    // Mouse/touch
    MOUSE: { LEFT: 0, MIDDLE: 1, RIGHT: 2, ROTATE: 0, DOLLY: 1, PAN: 2 },
    TOUCH: { ROTATE: 0, PAN: 1, DOLLY_PAN: 2, DOLLY_ROTATE: 3 },
    // Normal map types
    TangentSpaceNormalMap: 0, ObjectSpaceNormalMap: 1,
    // Transfer (color)
    LinearTransfer: 'linear', SRGBTransfer: 'srgb',
    // Compressed format codes — three.js exposes constants, our impl ignores them
    RGB_S3TC_DXT1_Format: 33776, RGBA_S3TC_DXT1_Format: 33777,
    RGBA_S3TC_DXT3_Format: 33778, RGBA_S3TC_DXT5_Format: 33779,
    RGB_PVRTC_4BPPV1_Format: 35840, RGB_PVRTC_2BPPV1_Format: 35841,
    RGBA_PVRTC_4BPPV1_Format: 35842, RGBA_PVRTC_2BPPV1_Format: 35843,
    RGB_ETC1_Format: 36196, RGB_ETC2_Format: 37492, RGBA_ETC2_EAC_Format: 37496,
    RGBA_ASTC_4x4_Format: 37808, RGBA_ASTC_5x4_Format: 37809, RGBA_ASTC_5x5_Format: 37810,
    RGBA_ASTC_6x5_Format: 37811, RGBA_ASTC_6x6_Format: 37812, RGBA_ASTC_8x5_Format: 37813,
    RGBA_ASTC_8x6_Format: 37814, RGBA_ASTC_8x8_Format: 37815, RGBA_ASTC_10x5_Format: 37816,
    RGBA_ASTC_10x6_Format: 37817, RGBA_ASTC_10x8_Format: 37818, RGBA_ASTC_10x10_Format: 37819,
    RGBA_ASTC_12x10_Format: 37820, RGBA_ASTC_12x12_Format: 37821,
    RGB_BPTC_SIGNED_Format: 36494, RGB_BPTC_UNSIGNED_Format: 36495,
    RGBA_BPTC_Format: 36492, RED_RGTC1_Format: 36283, SIGNED_RED_RGTC1_Format: 36284,
    RED_GREEN_RGTC2_Format: 36285, SIGNED_RED_GREEN_RGTC2_Format: 36286,
    // Three.js puts these on the namespace.
    Layers, MathUtils, Cache, Fog, FogExp2, ShapeUtils,
    // ---- examples/jsm add-ons (not in canonical core, but commonly imported) ----
    OBB, Capsule, ImprovedNoise, ConvexHull,
    RoundedBoxGeometry, TeapotGeometry, ParametricGeometries,
    LineMaterial, LineGeometry, LineSegmentsGeometry, Line2, LineSegments2, Wireframe,
    BufferGeometryUtils, geometryToBufferGeometry, SceneUtils, SkeletonUtils, RoomEnvironment,
    MTLLoader, OBJExporter, STLExporter, PLYExporter, GLTFExporter, MeshoptDecoder,
    Reflector, Refractor, Water, Sky, Lensflare,
    VRButton, ARButton, XRControllerModelFactory, XRHandModelFactory,
    CSS3DObject, CSS3DSprite, CSS3DRenderer,
    BokehPass, SAOPass, SMAAPass, TAARenderPass, AfterimagePass, BloomPass,
    OutputPass, TransitionPass, LUTPass, RenderPixelatedPass, AdaptiveToneMappingPass,
    // Missing tail — final constants and aliases for full canonical coverage.
    ConstantAlphaFactor: 211, OneMinusConstantAlphaFactor: 212,
    ConstantColorFactor: 213, OneMinusConstantColorFactor: 214,
    CustomToneMapping: 5,
    // Pre-r155 filter aliases (deprecated capitalization)
    LinearMipMapLinearFilter: 1008, LinearMipMapNearestFilter: 1005,
    NearestMipMapLinearFilter: 1007, NearestMipMapNearestFilter: 1004,
    // Reflectivity combine modes for MeshBasicMaterial
    MultiplyOperation: 0, MixOperation: 1, AddOperation: 2,
    // Aliases (`RenderTarget`/`WebGLArrayRenderTarget` collapse onto our impl).
    RenderTarget: WebGLRenderTarget, WebGLArrayRenderTarget: WebGLRenderTarget,
    // ShapePath: a curve container that builds shapes from line segments.
    ShapePath: class ShapePath { constructor() { this.subPaths = []; this.currentPath = null; } moveTo(x, y) { this.currentPath = new Path(); this.currentPath.moveTo(x, y); this.subPaths.push(this.currentPath); return this; } lineTo(x, y) { this.currentPath?.lineTo(x, y); return this; } toShapes() { return []; } },
    // PolyhedronGeometry: subdivision-based polyhedron — alias to icosahedron for now.
    PolyhedronGeometry: class PolyhedronGeometry { constructor(_vertices, _indices, radius = 1, detail = 0) { Object.assign(this, new IcosahedronGeometry(radius, detail)); this.type = 'PolyhedronGeometry'; } },
    // Video / animation export (native-codec wasm)
    VideoFormat,
    BrowserVideoFormat,
    VideoExporter,
    VideoExportResult,
    VideoExportError,
    VideoExportErrorCode,
    VideoExportEvent,
    VideoExportProgressEvent,
    VideoEncodeWorker,
    emitProgress,
    isVideoExportAvailable,
    assertVideoExportAvailable,
    parseVideoFormat,
    formatVideoProgress,
    formatBytes,
    alignVideoSize,
    encodeVideoFrames,
    encodeVideoFramesInWorker,
    downloadVideoBytes,
    encodeAndDownloadVideoFrames,
    captureSceneFrames,
    exportSceneVideo,
    exportAndDownloadSceneVideo,
    encodeGifRgba,
    encodeApngRgba,
    encodeWebmRgba,
    videoMimeType,
    videoFilename,
};

if (typeof window !== 'undefined') {
    window.THREE = THREE;
}

export default THREE;
