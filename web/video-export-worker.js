/**
 * Web Worker that loads threers wasm (`native-codec`) and encodes GIF/APNG/WebM/MP4.
 *
 * Main thread captures frames (optionally pipelined); this worker encodes so the
 * UI stays responsive.
 *
 * Messages in:
 *   { id, type: 'init', wasmUrl }
 *   { id, type: 'encode', wasmUrl?, format, width, height, fps, transparent, gifColors, frames: ArrayBuffer[] }
 *   { id, type: 'streamBegin', streamId, wasmUrl?, format, width, height, fps, transparent, gifColors }
 *   { type: 'streamPush', streamId, frames: ArrayBuffer[] }
 *   { id, type: 'streamFinish', streamId }
 *
 * Messages out:
 *   { id, ok: true }
 *   { id, ok: true, bytes: ArrayBuffer, mime, format }
 *   { id, ok: false, error: string }
 */

import init, {
  encodeGifRgba,
  encodeApngRgba,
  encodeWebmRgba,
  encodeMp4Rgba,
} from './pkg/threers.js';

let initPromise = null;
let wasmUrlCached = null;

/** @type {Map<number, { format: string, width: number, height: number, fps: number, transparent: boolean, gifColors: number, frames: Uint8Array[] }>} */
const streams = new Map();

function ensureInit(wasmUrl) {
  const url = wasmUrl || wasmUrlCached;
  if (!url) {
    return Promise.reject(new Error('video-export-worker: wasmUrl required on first message'));
  }
  if (!initPromise || wasmUrlCached !== url) {
    wasmUrlCached = url;
    initPromise = init({ module_or_path: url });
  }
  return initPromise;
}

function asU8(bytes) {
  return bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
}

function frameViews(buffers) {
  return (buffers || []).map((buf) => {
    if (buf instanceof Uint8Array) return buf;
    if (buf instanceof ArrayBuffer) return new Uint8Array(buf);
    if (ArrayBuffer.isView(buf)) {
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    }
    throw new Error('frame must be ArrayBuffer or TypedArray');
  });
}

function encodeFormat(format, width, height, fps, transparent, gifColors, frames) {
  if (format === 'gif') {
    return encodeGifRgba(width, height, fps, gifColors, transparent, frames);
  }
  if (format === 'apng') {
    return encodeApngRgba(width, height, fps, transparent, frames);
  }
  if (format === 'webm') {
    return encodeWebmRgba(width, height, fps, transparent, frames);
  }
  if (format === 'mp4') {
    if (transparent) {
      throw new Error('MP4/H.264 does not support transparency');
    }
    if (width % 2 !== 0 || height % 2 !== 0) {
      throw new Error(`MP4/H.264 requires even width and height (got ${width}×${height})`);
    }
    return encodeMp4Rgba(width, height, fps, frames);
  }
  throw new Error(`unsupported format: ${format}`);
}

function mimeFor(format) {
  if (format === 'gif') return 'image/gif';
  if (format === 'apng') return 'image/png';
  if (format === 'mp4') return 'video/mp4';
  return 'video/webm';
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  const { id, type } = msg;
  try {
    if (type === 'init') {
      await ensureInit(msg.wasmUrl);
      if (typeof encodeGifRgba !== 'function') {
        throw new Error('native-codec bindings missing — rebuild with NATIVE_CODEC=1');
      }
      if (typeof encodeMp4Rgba !== 'function') {
        throw new Error('encodeMp4Rgba missing — rebuild with NATIVE_CODEC=1');
      }
      self.postMessage({ id, ok: true, type: 'init' });
      return;
    }

    if (type === 'streamBegin') {
      await ensureInit(msg.wasmUrl);
      const streamId = msg.streamId | 0;
      if (!streamId) throw new Error('streamBegin: streamId required');
      streams.set(streamId, {
        format: String(msg.format || 'gif').toLowerCase(),
        width: msg.width | 0,
        height: msg.height | 0,
        fps: Math.max(1, msg.fps | 0 || 30),
        transparent: !!msg.transparent,
        gifColors: Math.min(256, Math.max(2, msg.gifColors | 0 || 256)),
        frames: [],
      });
      self.postMessage({ id, ok: true, type: 'streamBegin' });
      return;
    }

    if (type === 'streamPush') {
      const streamId = msg.streamId | 0;
      const stream = streams.get(streamId);
      if (!stream) throw new Error(`streamPush: unknown stream ${streamId}`);
      for (const view of frameViews(msg.frames)) {
        stream.frames.push(view);
      }
      return;
    }

    if (type === 'streamFinish') {
      const streamId = msg.streamId | 0;
      const stream = streams.get(streamId);
      if (!stream) throw new Error(`streamFinish: unknown stream ${streamId}`);
      streams.delete(streamId);
      const encoded = encodeFormat(
        stream.format,
        stream.width,
        stream.height,
        stream.fps,
        stream.transparent,
        stream.gifColors,
        stream.frames,
      );
      stream.frames = [];
      const u8 = asU8(encoded);
      const copy = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
      self.postMessage({
        id,
        ok: true,
        type: 'streamFinish',
        bytes: copy,
        mime: mimeFor(stream.format),
        format: stream.format,
      }, [copy]);
      return;
    }

    if (type === 'encode') {
      await ensureInit(msg.wasmUrl);
      const width = msg.width | 0;
      const height = msg.height | 0;
      const fps = Math.max(1, msg.fps | 0 || 30);
      const transparent = !!msg.transparent;
      const gifColors = Math.min(256, Math.max(2, msg.gifColors | 0 || 256));
      const format = String(msg.format || 'gif').toLowerCase();
      const frames = frameViews(msg.frames);

      const encoded = encodeFormat(format, width, height, fps, transparent, gifColors, frames);

      const u8 = asU8(encoded);
      const copy = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
      self.postMessage({ id, ok: true, type: 'encode', bytes: copy, mime: mimeFor(format), format }, [copy]);
      return;
    }

    throw new Error(`unknown worker message type: ${type}`);
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      type: type || 'error',
      error: String(err?.message || err),
    });
  }
};
