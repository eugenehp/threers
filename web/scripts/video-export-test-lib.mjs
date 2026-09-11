/**
 * Shared helpers for browser video-export / H.264 parity tests.
 * FNV-1a and frame generators must stay in sync with `tests/h264_animation_parity.rs`.
 */

/** @typedef {import('../video-export.d.ts').VideoFormatName} VideoFormatName */

export function fnv1a(buf) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < buf.length; i++) {
    h ^= buf[i];
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

export function fnv1aNum(buf) {
  return Number.parseInt(fnv1a(buf), 16) >>> 0;
}

export function solidFrame(w, h, r, g, b, a = 255) {
  const out = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = a;
  }
  return out;
}

export function gradientFrame(w, h) {
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      out[i] = Math.floor((x * 255) / Math.max(1, w));
      out[i + 1] = Math.floor((y * 255) / Math.max(1, h));
      out[i + 2] = ((x ^ y) * 37) & 0xFF;
      out[i + 3] = 255;
    }
  }
  return out;
}

export function checkerFrame(w, h, seed = 0) {
  const out = new Uint8Array(w * h * 4);
  const s = (seed * 17) >>> 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const v = ((x * 37) ^ (y * 101) ^ s) & 0xFF;
      out[i] = v;
      out[i + 1] = (v + 40) & 0xFF;
      out[i + 2] = (v + 80) & 0xFF;
      out[i + 3] = 255;
    }
  }
  return out;
}

export function grayRampFrame(w, h) {
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const g = Math.floor(((x + y) * 255) / Math.max(1, w + h));
      out[i] = g;
      out[i + 1] = g;
      out[i + 2] = g;
      out[i + 3] = 255;
    }
  }
  return out;
}

export function lowBitnessFrame(w, h) {
  const out = solidFrame(w, h, 0, 0, 0);
  for (let i = 0; i < w * h; i++) {
    const q = (i % 4) * 64;
    out[i * 4] = q;
    out[i * 4 + 1] = (q + 16) & 0xFF;
    out[i * 4 + 2] = (q + 32) & 0xFF;
  }
  return out;
}

export function syntheticSolidFrames(w, h) {
  return [
    solidFrame(w, h, 255, 0, 0),
    solidFrame(w, h, 0, 255, 0),
    solidFrame(w, h, 0, 0, 255),
    solidFrame(w, h, 255, 255, 0),
  ];
}

export function framesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) return false;
    for (let j = 0; j < a[i].length; j++) {
      if (a[i][j] !== b[i][j]) return false;
    }
  }
  return true;
}

export function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export function assertMp4(bytes, label = 'mp4') {
  if (!(bytes?.length > 12)) throw new Error(`${label}: too small`);
  if (bytes[4] !== 0x66 || bytes[5] !== 0x74 || bytes[6] !== 0x79 || bytes[7] !== 0x70) {
    throw new Error(`${label}: missing ftyp`);
  }
}

export function assertGif(bytes) {
  if (bytes[0] !== 0x47 || bytes[1] !== 0x49 || bytes[2] !== 0x46) {
    throw new Error('missing GIF magic');
  }
}

export function assertApng(bytes) {
  if (bytes[0] !== 0x89 || bytes[1] !== 0x50) throw new Error('missing PNG magic');
}

export function assertWebm(bytes) {
  if (bytes[0] !== 0x1a || bytes[1] !== 0x45 || bytes[2] !== 0xdf || bytes[3] !== 0xa3) {
    throw new Error('missing WebM magic');
  }
}

/** Golden MP4 checksums from `tests/h264_animation_parity.rs`. */
export const GOLDEN_MP4 = Object.freeze({
  solid64x48_4f_10fps: 'f5ca7103',
  gradient128x72_3f_30fps: '1c091ef4',
  checker320x240_2f_60fps: '06597030',
});

export function workerUrls(origin = location.origin) {
  return {
    wasmUrl: new URL('/web/pkg/threers_bg.wasm', origin).href,
    workerUrl: new URL('/web/video-export-worker.js', origin).href,
  };
}

/**
 * Encode on main thread and worker; assert byte-identical output.
 * @param {object} deps
 * @param {typeof import('../video-export.js').VideoExporter} deps.VideoExporter
 * @param {typeof import('../video-export.js').VideoEncodeWorker} deps.VideoEncodeWorker
 * @param {Uint8Array[]} frames
 * @param {import('../video-export.d.ts').VideoEncodeOptions} opts
 * @param {string} label
 */
export async function assertMainWorkerParity(deps, frames, opts, label) {
  const main = deps.VideoExporter.encode(frames, opts);
  const worker = new deps.VideoEncodeWorker(workerUrls());
  try {
    const fromWorker = await worker.encode(frames, opts);
    if (main.byteLength !== fromWorker.byteLength) {
      throw new Error(`${label}: size ${fromWorker.byteLength} != main ${main.byteLength}`);
    }
    if (fnv1a(main.bytes) !== fnv1a(fromWorker.bytes)) {
      throw new Error(`${label}: worker bytes differ from main thread`);
    }
    return { main, worker: fromWorker };
  } finally {
    worker.terminate();
  }
}

/**
 * @param {Uint8Array[]} frames
 * @param {'uint8'|'arraybuffer'|'subarray'} mode
 */
export function rebufferFrames(frames, mode) {
  if (mode === 'uint8') return frames;
  if (mode === 'arraybuffer') {
    return frames.map((u8) => u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
  }
  if (mode === 'subarray') {
    return frames.map((u8) => new Uint8Array(u8.buffer, u8.byteOffset, u8.byteLength));
  }
  throw new Error(`unknown buffer mode: ${mode}`);
}

/**
 * @param {() => Promise<void>|void} fn
 * @returns {Promise<number>} elapsed ms
 */
export async function timeAsync(fn) {
  const t0 = performance.now();
  await fn();
  return performance.now() - t0;
}
