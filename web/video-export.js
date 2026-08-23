/**
 * Browser video / animated-image export (requires wasm built with `native-codec`).
 *
 * Preferred entry point: {@link VideoExporter}.
 *
 * @example
 * ```js
 * import { VideoExporter, VideoFormat } from './video-export.js';
 * const result = VideoExporter.encode(frames, {
 *   width: 320, height: 240, fps: 15, format: VideoFormat.Gif, transparent: true,
 * }).download('cube');
 * // result.bytes / result.toBlob() / result.summary
 * ```
 */

import * as _wasm from './pkg/threers.js';

/** @typedef {'gif' | 'apng' | 'webm' | 'mp4'} VideoFormatName */

/**
 * Browser-friendly containers encoded in-process (no ffmpeg).
 * @readonly
 * @enum {VideoFormatName}
 */
export const VideoFormat = Object.freeze({
  Gif: 'gif',
  Apng: 'apng',
  Webm: 'webm',
  Mp4: 'mp4',
});

/** @deprecated Use {@link VideoFormat}. */
export const BrowserVideoFormat = VideoFormat;

/** Stable {@link VideoExportError} codes. */
export const VideoExportErrorCode = Object.freeze({
  Unavailable: 'Unavailable',
  InvalidFormat: 'InvalidFormat',
  InvalidOption: 'InvalidOption',
  InvalidDimensions: 'InvalidDimensions',
  EmptyFrames: 'EmptyFrames',
  InvalidFrame: 'InvalidFrame',
  FrameSize: 'FrameSize',
  EncodeFailed: 'EncodeFailed',
  Aborted: 'Aborted',
});

/**
 * Event names dispatched by {@link VideoExporter}.
 * @readonly
 */
export const VideoExportEvent = Object.freeze({
  /** Export / encode is about to begin. */
  Start: 'start',
  /** Any progress tick (`detail` is {@link VideoExportProgress}). */
  Progress: 'progress',
  /** Capture-phase progress (scene path). */
  Capture: 'capture',
  /** Encode-phase progress. */
  Encode: 'encode',
  /** Finished successfully (`detail.result` is {@link VideoExportResult}). */
  Complete: 'complete',
  /** Failed (`detail.error`). */
  Error: 'error',
  /** Aborted via {@link AbortSignal} (`detail.error`). */
  Abort: 'abort',
});

/**
 * Progress CustomEvent — also exposes `phase` / `frame` / `ratio` / `message` on the event.
 */
export class VideoExportProgressEvent extends CustomEvent {
  /**
   * @param {string} type
   * @param {import('./video-export.d.ts').VideoExportProgress} progress
   */
  constructor(type, progress) {
    super(type, { detail: progress });
    this.progress = progress;
    this.phase = progress.phase;
    this.frame = progress.frame;
    this.frames = progress.frames;
    this.ratio = progress.ratio;
    this.format = progress.format;
    this.message = progress.message;
  }
}

/**
 * Structured export output.
 */
export class VideoExportResult {
  /**
   * @param {{
   *   bytes: Uint8Array,
   *   format: VideoFormatName,
   *   width: number,
   *   height: number,
   *   fps: number,
   *   frameCount: number,
   *   transparent: boolean,
   *   filename: string,
   * }} init
   */
  constructor(init) {
    this.bytes = init.bytes;
    this.format = init.format;
    this.width = init.width;
    this.height = init.height;
    this.fps = init.fps;
    this.frameCount = init.frameCount;
    this.transparent = init.transparent;
    this.filename = init.filename;
    this.mime = videoMimeType(init.format);
    /** @type {string|null} */
    this._objectUrl = null;
  }

  /** Duration in seconds (`frameCount / fps`). */
  get duration() {
    return this.frameCount / Math.max(1, this.fps);
  }

  /** Byte length of the encoded payload. */
  get byteLength() {
    return this.bytes.byteLength;
  }

  /** e.g. `"320×240"`. */
  get sizeLabel() {
    return `${this.width}×${this.height}`;
  }

  /** Short human summary for logs / UI status. */
  get summary() {
    return `${this.filename} · ${this.sizeLabel} · ${this.frameCount}f @ ${this.fps}fps · ${formatBytes(this.byteLength)}`;
  }

  /** Fresh Blob for this payload. */
  toBlob() {
    return new Blob([this.bytes], { type: this.mime });
  }

  /**
   * Object URL — reuse until {@link revokeObjectURL} / next {@link download}.
   * Caller may also `URL.revokeObjectURL` themselves.
   */
  toObjectURL() {
    if (!this._objectUrl) {
      this._objectUrl = URL.createObjectURL(this.toBlob());
    }
    return this._objectUrl;
  }

  /** Drop a cached object URL created by {@link toObjectURL}. */
  revokeObjectURL() {
    if (this._objectUrl) {
      URL.revokeObjectURL(this._objectUrl);
      this._objectUrl = null;
    }
    return this;
  }

  /**
   * Trigger a browser download.
   * Extension is added automatically when omitted (`"cube"` → `"cube.gif"`).
   * @param {string} [filename]
   * @returns {this}
   */
  download(filename) {
    const name = ensureFilename(
      filename || this.filename,
      this.format,
      this.transparent,
    );
    const url = this.toObjectURL();
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    this.filename = name;
    // Keep URL alive briefly so the download can start, then revoke.
    setTimeout(() => this.revokeObjectURL(), 0);
    return this;
  }

  /**
   * Build an `<img>` (gif/apng) or `<video>` (webm/mp4) for quick preview.
   * @param {ParentNode|null} [parent] append target
   * @returns {HTMLImageElement|HTMLVideoElement}
   */
  preview(parent = null) {
    const url = this.toObjectURL();
    /** @type {HTMLImageElement|HTMLVideoElement} */
    let el;
    if (this.format === 'webm' || this.format === 'mp4') {
      el = document.createElement('video');
      el.src = url;
      el.controls = true;
      el.loop = true;
      el.playsInline = true;
      el.muted = true;
      void el.play?.().catch(() => {});
    } else {
      el = document.createElement('img');
      el.src = url;
      el.alt = this.filename;
    }
    el.dataset.threersPreview = this.format;
    if (parent) parent.appendChild(el);
    return el;
  }
}

/**
 * Fluent encoder for GIF / APNG / WebM / MP4 (`EventTarget`).
 *
 * Buffer path: `VideoExporter.encode(frames, opts)` or `new VideoExporter(opts).encode(frames)`.
 * Scene path (shim): `VideoExporter.from(renderer, scene, camera).gif().frames(30).download('cube')`.
 *
 * Trace progress with events:
 * ```js
 * exporter.on(VideoExportEvent.Progress, (e) => console.log(e.message, e.ratio));
 * exporter.on(VideoExportEvent.Complete, ({ detail }) => console.log(detail.result.summary));
 * ```
 */
export class VideoExporter extends EventTarget {
  /**
   * @param {Partial<import('./video-export.d.ts').VideoEncodeOptions>} [options]
   */
  constructor(options = {}) {
    super();
    this._opts = {
      width: options.width,
      height: options.height,
      format: options.format ?? VideoFormat.Gif,
      fps: options.fps ?? 30,
      transparent: options.transparent ?? false,
      gifColors: options.gifColors ?? 256,
      filename: options.filename,
      basename: options.basename ?? 'export',
      strictSize: options.strictSize ?? false,
    };
    this._onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
    this._signal = options.signal ?? null;
    this._mapFrame = typeof options.mapFrame === 'function' ? options.mapFrame : null;
    this._forwardTarget = options.eventTarget && options.eventTarget !== this
      ? options.eventTarget
      : null;
  }

  /**
   * Chainable `addEventListener`.
   * @param {string} type
   * @param {EventListenerOrEventListenerObject} listener
   * @param {boolean|AddEventListenerOptions} [options]
   */
  on(type, listener, options) {
    this.addEventListener(type, listener, options);
    return this;
  }

  /**
   * Chainable `removeEventListener`.
   * @param {string} type
   * @param {EventListenerOrEventListenerObject} listener
   * @param {boolean|EventListenerOptions} [options]
   */
  off(type, listener, options) {
    this.removeEventListener(type, listener, options);
    return this;
  }

  /**
   * Chainable one-shot listener.
   * @param {string} type
   * @param {EventListenerOrEventListenerObject} listener
   */
  once(type, listener) {
    this.addEventListener(type, listener, { once: true });
    return this;
  }

  /** @internal */
  _emitTargets() {
    return this._forwardTarget ? [this, this._forwardTarget] : [this];
  }

  /** @internal */
  _emit(type, detail = {}) {
    const event = new CustomEvent(type, { detail });
    for (const t of this._emitTargets()) {
      try { t.dispatchEvent(event); } catch { /* ignore listener errors */ }
    }
    return event;
  }

  /** @internal */
  _emitProgress(info) {
    return emitProgress(this._onProgress, enrichProgress(info), this);
  }

  /** @internal */
  _emitLifecycleError(err) {
    const aborted = VideoExportError.is(err) && err.code === VideoExportErrorCode.Aborted;
    this._emit(aborted ? VideoExportEvent.Abort : VideoExportEvent.Error, { error: err });
  }


  /**
   * Set several options at once (same keys as the constructor).
   * @param {Partial<import('./video-export.d.ts').VideoEncodeOptions>} options
   */
  configure(options = {}) {
    if (options.width != null || options.height != null) {
      this._opts.width = options.width ?? this._opts.width;
      this._opts.height = options.height ?? this._opts.height;
    }
    if (options.format != null) this._opts.format = parseVideoFormat(options.format).format;
    if (options.fps != null) this._opts.fps = options.fps;
    if (options.transparent != null) this._opts.transparent = !!options.transparent;
    if (options.gifColors != null) this._opts.gifColors = options.gifColors;
    if (options.filename != null) this._opts.filename = options.filename;
    if (options.basename != null) this._opts.basename = options.basename;
    if (options.strictSize != null) this._opts.strictSize = !!options.strictSize;
    if (typeof options.onProgress === 'function') this._onProgress = options.onProgress;
    if (options.signal !== undefined) this._signal = options.signal ?? null;
    if (typeof options.mapFrame === 'function') this._mapFrame = options.mapFrame;
    // Accept shorthand aliases from parseVideoFormat for transparent webm etc.
    if (options.format != null) {
      const parsed = parseVideoFormat(options.format);
      if (parsed.transparent) this._opts.transparent = true;
    }
    return this;
  }

  /** @param {VideoFormatName|string} format */
  format(format) {
    const parsed = parseVideoFormat(format);
    this._opts.format = parsed.format;
    if (parsed.transparent) this._opts.transparent = true;
    return this;
  }

  /** GIF preset (`transparent` optional). */
  gif(options = {}) {
    this._opts.format = VideoFormat.Gif;
    if (options.transparent != null) this._opts.transparent = !!options.transparent;
    if (options.colors != null) this._opts.gifColors = options.colors;
    if (options.gifColors != null) this._opts.gifColors = options.gifColors;
    return this;
  }

  /** APNG preset. */
  apng(options = {}) {
    this._opts.format = VideoFormat.Apng;
    if (options.transparent != null) this._opts.transparent = !!options.transparent;
    return this;
  }

  /** WebM/VP9 preset. Pass `{ alpha: true }` for transparent WebM. */
  webm(options = {}) {
    this._opts.format = VideoFormat.Webm;
    if (options.alpha != null) this._opts.transparent = !!options.alpha;
    if (options.transparent != null) this._opts.transparent = !!options.transparent;
    return this;
  }

  /** H.264 MP4 preset (even width/height; no alpha). */
  mp4() {
    this._opts.format = VideoFormat.Mp4;
    this._opts.transparent = false;
    return this;
  }

  /** @param {number} width @param {number} height */
  size(width, height) {
    this._opts.width = width;
    this._opts.height = height;
    return this;
  }

  /** @param {number} fps */
  fps(fps) {
    this._opts.fps = fps;
    return this;
  }

  /** @param {boolean} [on=true] */
  transparent(on = true) {
    this._opts.transparent = !!on;
    return this;
  }

  /** GIF palette size (`2..=256`). */
  gifColors(n) {
    this._opts.gifColors = n;
    return this;
  }

  /**
   * Download / result filename. Extension optional (`"cube"` → `"cube.gif"`).
   * @param {string} name
   */
  filename(name) {
    this._opts.filename = name;
    return this;
  }

  basename(name) {
    this._opts.basename = name;
    return this;
  }

  /**
   * When true, WebM/MP4 reject misaligned sizes instead of snapping (scene path).
   * @param {boolean} [on=true]
   */
  strictSize(on = true) {
    this._opts.strictSize = !!on;
    return this;
  }

  /** @param {(info: import('./video-export.d.ts').VideoExportProgress) => void} cb */
  onProgress(cb) {
    this._onProgress = cb;
    return this;
  }

  /** @param {AbortSignal|null|undefined} signal */
  signal(signal) {
    this._signal = signal ?? null;
    return this;
  }

  /**
   * Abort encode/capture after `ms` milliseconds (`AbortSignal.timeout`).
   * @param {number} ms
   */
  timeout(ms) {
    const n = Number(ms);
    if (!Number.isFinite(n) || n <= 0) {
      throw new VideoExportError('timeout must be a positive number of ms', VideoExportErrorCode.InvalidOption);
    }
    if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function') {
      this._signal = AbortSignal.timeout(n);
    } else {
      const ctrl = new AbortController();
      setTimeout(() => ctrl.abort(new DOMException('TimeoutError', 'TimeoutError')), n);
      this._signal = ctrl.signal;
    }
    return this;
  }

  /**
   * Optional per-frame RGBA transform (e.g. alpha punch).
   * @param {(rgba: Uint8Array, frameIndex: number, frameCount: number) => Uint8Array} fn
   */
  mapFrame(fn) {
    this._mapFrame = fn;
    return this;
  }

  /**
   * Encode RGBA frames to a {@link VideoExportResult}.
   * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
   * @returns {VideoExportResult}
   */
  encode(frames) {
    const opts = this._resolvedEncodeOptions();
    this._emit(VideoExportEvent.Start, {
      phase: 'encode',
      format: opts.format,
      frames: frames?.length ?? 0,
    });
    try {
      const result = encodeWithOptions(
        frames,
        opts,
        this._onProgress,
        this._signal,
        this._mapFrame,
        this,
      );
      this._emit(VideoExportEvent.Complete, { result });
      return result;
    } catch (err) {
      this._emitLifecycleError(err);
      throw err;
    }
  }

  /**
   * Encode and download.
   * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
   * @param {string} [filename]
   * @returns {VideoExportResult}
   */
  encodeAndDownload(frames, filename) {
    return this.encode(frames).download(filename);
  }

  /**
   * One-shot encode from RGBA frames (no fluent chain required).
   * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
   * @param {import('./video-export.d.ts').VideoEncodeOptions} options
   * @returns {VideoExportResult}
   */
  static encode(frames, options) {
    return new VideoExporter(options).encode(frames);
  }

  /**
   * One-shot encode + download.
   * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
   * @param {import('./video-export.d.ts').VideoEncodeOptions} options
   * @returns {VideoExportResult}
   */
  static encodeAndDownload(frames, options) {
    return new VideoExporter(options).encodeAndDownload(frames, options?.filename);
  }

  /**
   * Encode on a Web Worker (keeps the UI thread free).
   * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
   * @param {import('./video-export.d.ts').VideoEncodeOptions} options
   * @param {import('./video-export.d.ts').VideoEncodeWorkerOptions} [workerOptions]
   * @returns {Promise<VideoExportResult>}
   */
  static encodeInWorker(frames, options, workerOptions) {
    return encodeVideoFramesInWorker(frames, options, workerOptions);
  }

  /**
   * Instance helper — encode current options on a worker.
   * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
   * @param {import('./video-export.d.ts').VideoEncodeWorkerOptions} [workerOptions]
   * @returns {Promise<VideoExportResult>}
   */
  async encodeInWorker(frames, workerOptions) {
    const opts = this._resolvedEncodeOptions();
    this._emit(VideoExportEvent.Start, {
      phase: 'encode',
      format: opts.format,
      frames: frames?.length ?? 0,
    });
    try {
      const result = await encodeVideoFramesInWorker(frames, opts, {
        ...workerOptions,
        onProgress: workerOptions?.onProgress ?? this._onProgress,
        signal: workerOptions?.signal ?? this._signal,
        mapFrame: workerOptions?.mapFrame ?? this._mapFrame,
        eventTarget: workerOptions?.eventTarget ?? this,
      });
      this._emit(VideoExportEvent.Complete, { result });
      return result;
    } catch (err) {
      this._emitLifecycleError(err);
      throw err;
    }
  }

  _resolvedEncodeOptions() {
    const width = u32(this._opts.width, 'width');
    const height = u32(this._opts.height, 'height');
    const fps = Math.max(1, u32(this._opts.fps ?? 30, 'fps'));
    const format = normalizeFormat(this._opts.format);
    const transparent = !!this._opts.transparent;
    const gifColors = clamp(this._opts.gifColors ?? 256, 2, 256);
    const filename = ensureFilename(
      this._opts.filename || this._opts.basename || 'export',
      format,
      transparent,
    );
    return {
      width,
      height,
      fps,
      format,
      transparent,
      gifColors,
      filename,
      strictSize: !!this._opts.strictSize,
    };
  }
}

/**
 * True when the loaded wasm exposes native-codec encode bindings.
 * @returns {boolean}
 */
export function isVideoExportAvailable() {
  return typeof _wasm.encodeGifRgba === 'function'
    && typeof _wasm.encodeApngRgba === 'function'
    && typeof _wasm.encodeWebmRgba === 'function'
    && typeof _wasm.encodeMp4Rgba === 'function';
}

/**
 * Throw a helpful {@link VideoExportError} when native-codec wasm is missing.
 * @returns {void}
 */
export function assertVideoExportAvailable() {
  assertAvailable();
}

/**
 * Parse UI / user format strings: `"gif"`, `"GIF"`, `"webm-alpha"`, `"apng.png"`.
 * @param {string|VideoFormatName} input
 * @returns {{ format: VideoFormatName, transparent: boolean }}
 */
export function parseVideoFormat(input) {
  const raw = String(input ?? '').trim().toLowerCase();
  if (!raw) {
    throw new VideoExportError('video format is required', VideoExportErrorCode.InvalidFormat);
  }
  if (raw === 'gif' || raw === 'image/gif') {
    return { format: 'gif', transparent: false };
  }
  if (raw === 'apng' || raw === 'apng.png' || raw === 'png' || raw === 'image/png' || raw === 'image/apng') {
    return { format: 'apng', transparent: false };
  }
  if (raw === 'webm' || raw === 'video/webm' || raw === 'vp9') {
    return { format: 'webm', transparent: false };
  }
  if (raw === 'webm-alpha' || raw === 'webm_alpha' || raw === 'webm+alpha' || raw === 'vp9-alpha') {
    return { format: 'webm', transparent: true };
  }
  if (raw === 'mp4' || raw === 'video/mp4' || raw === 'h264' || raw === 'avc1') {
    return { format: 'mp4', transparent: false };
  }
  throw new VideoExportError(
    `unsupported video format "${input}" (use gif | apng | webm | webm-alpha | mp4)`,
    VideoExportErrorCode.InvalidFormat,
  );
}

/**
 * Ready-to-display progress line, e.g. `"Rendering 12/30 (40%)"`.
 * @param {import('./video-export.d.ts').VideoExportProgress} info
 * @returns {string}
 */
export function formatVideoProgress(info) {
  if (!info || typeof info !== 'object') return '';
  const pct = Math.round(clamp(info.ratio ?? 0, 0, 1) * 100);
  if (info.phase === 'capture') {
    return `Rendering ${info.frame}/${info.frames} (${pct}%)`;
  }
  if (info.phase === 'encode') {
    return `Encoding ${String(info.format || '').toUpperCase()}… (${pct}%)`;
  }
  if (info.phase === 'done') {
    return `Done (${info.frames} frames)`;
  }
  return `${info.phase || 'export'} ${pct}%`;
}

/**
 * Snap width/height down to multiples of 8 (VP9). Returns whether values changed.
 * @param {number} width
 * @param {number} height
 * @returns {{ width: number, height: number, snapped: boolean }}
 */
export function alignVideoSize(width, height) {
  const w = Math.max(8, Math.floor(Number(width) / 8) * 8);
  const h = Math.max(8, Math.floor(Number(height) / 8) * 8);
  return { width: w, height: h, snapped: w !== width || h !== height };
}

/**
 * Snap width/height down to even values (H.264 YUV420). Returns whether values changed.
 * @param {number} width
 * @param {number} height
 * @returns {{ width: number, height: number, snapped: boolean }}
 */
export function alignVideoSizeForMp4(width, height) {
  const w = Math.max(2, Math.floor(Number(width) / 2) * 2);
  const h = Math.max(2, Math.floor(Number(height) / 2) * 2);
  return { width: w, height: h, snapped: w !== width || h !== height };
}

/**
 * @param {VideoFormatName} format
 * @returns {string}
 */
export function videoMimeType(format) {
  switch (normalizeFormat(format)) {
    case 'gif': return 'image/gif';
    case 'apng': return 'image/png';
    case 'webm': return 'video/webm';
    case 'mp4': return 'video/mp4';
    default: throw new VideoExportError(`unknown video format: ${format}`, VideoExportErrorCode.InvalidFormat);
  }
}

/**
 * @param {VideoFormatName} format
 * @param {string} [basename='export']
 * @param {boolean} [transparent=false]
 * @returns {string}
 */
export function videoFilename(format, basename = 'export', transparent = false) {
  const base = String(basename).replace(/\.(gif|png|apng|webm)$/i, '');
  switch (normalizeFormat(format)) {
    case 'gif': return `${base}.gif`;
    case 'apng': return `${base}.apng.png`;
    case 'webm': return transparent ? `${base}-alpha.webm` : `${base}.webm`;
    case 'mp4': return `${base}.mp4`;
    default: throw new VideoExportError(`unknown video format: ${format}`, VideoExportErrorCode.InvalidFormat);
  }
}

/**
 * Encode tightly packed RGBA8 frames into GIF, APNG, WebM, or MP4 bytes.
 * Prefer {@link VideoExporter.encode} for a structured result.
 *
 * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
 * @param {import('./video-export.d.ts').VideoEncodeOptions} options
 * @returns {Uint8Array}
 */
export function encodeVideoFrames(frames, options) {
  return VideoExporter.encode(frames, options).bytes;
}

/**
 * @param {Uint8Array|ArrayBuffer} bytes
 * @param {import('./video-export.d.ts').VideoDownloadOptions} options
 * @returns {string} filename used
 */
export function downloadVideoBytes(bytes, options) {
  const format = normalizeFormat(options?.format);
  const transparent = !!options?.transparent;
  const filename = ensureFilename(
    options?.filename || options?.basename || 'export',
    format,
    transparent,
  );
  const result = new VideoExportResult({
    bytes: bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes),
    format,
    width: options?.width ?? 0,
    height: options?.height ?? 0,
    fps: options?.fps ?? 0,
    frameCount: options?.frameCount ?? 0,
    transparent,
    filename,
  });
  result.download(filename);
  return filename;
}

/**
 * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
 * @param {import('./video-export.d.ts').VideoEncodeOptions} options
 * @returns {VideoExportResult}
 */
export function encodeAndDownloadVideoFrames(frames, options) {
  return VideoExporter.encodeAndDownload(frames, options);
}

/**
 * Pool / single-shot Web Worker encoder (wasm `native-codec` runs off the main thread).
 */

/**
 * Transfer an RGBA frame to a worker. When `move` is true, detaches the source
 * buffer (for streaming capture). Batch `encode()` always copies.
 * @param {Uint8Array|ArrayBuffer|ArrayBufferView} frame
 * @param {{ move?: boolean }} [options]
 * @returns {{ buffer: ArrayBuffer, owned: boolean }}
 */
export function transferFrameBuffer(frame, options = {}) {
  const move = options.move === true;
  let u8 = frame;
  if (frame instanceof ArrayBuffer) {
    if (move) return { buffer: frame, owned: true };
    return { buffer: frame.slice(0), owned: false };
  }
  if (ArrayBuffer.isView(frame)) {
    u8 = new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
  } else if (!(frame instanceof Uint8Array)) {
    u8 = new Uint8Array(frame);
  }
  if (move && u8.byteOffset === 0 && u8.byteLength === u8.buffer.byteLength) {
    return { buffer: u8.buffer, owned: true };
  }
  return {
    buffer: u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength),
    owned: false,
  };
}

export class VideoEncodeWorker {
  /**
   * @param {import('./video-export.d.ts').VideoEncodeWorkerOptions} [options]
   */
  constructor(options = {}) {
    this._wasmUrl = options.wasmUrl || new URL('./pkg/threers_bg.wasm', import.meta.url).href;
    this._workerUrl = options.workerUrl || new URL('./video-export-worker.js', import.meta.url).href;
    this._worker = null;
    this._ready = null;
    this._seq = 0;
    /** @type {Map<number, { resolve: Function, reject: Function }>} */
    this._pending = new Map();
    /** @type {number|null} */
    this._activeStreamId = null;
    /** @type {number} */
    this._streamSeq = 0;
  }

  /** @returns {Worker} */
  _ensureWorker() {
    if (this._worker) return this._worker;
    const worker = new Worker(this._workerUrl, { type: 'module' });
    worker.onmessage = (event) => {
      const msg = event.data || {};
      const pending = this._pending.get(msg.id);
      if (!pending) return;
      this._pending.delete(msg.id);
      if (msg.ok) pending.resolve(msg);
      else pending.reject(new VideoExportError(msg.error || 'worker encode failed', VideoExportErrorCode.EncodeFailed));
    };
    worker.onerror = (event) => {
      const err = new VideoExportError(
        event.message || 'video encode worker error',
        VideoExportErrorCode.EncodeFailed,
      );
      for (const [, pending] of this._pending) pending.reject(err);
      this._pending.clear();
    };
    this._worker = worker;
    return worker;
  }

  /** @returns {Promise<void>} */
  init() {
    if (this._ready) return this._ready;
    const worker = this._ensureWorker();
    this._ready = this._call(worker, { type: 'init', wasmUrl: this._wasmUrl }).then(() => undefined);
    return this._ready;
  }

  /**
   * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
   * @param {import('./video-export.d.ts').VideoEncodeOptions} options
   * @returns {Promise<VideoExportResult>}
   */
  async encode(frames, options) {
    await this.init();
    throwIfAborted(options?.signal);
    const resolved = new VideoExporter(options)._resolvedEncodeOptions();
    let list = normalizeFrameList(frames, resolved.width * resolved.height * 4);
    if (typeof options?.mapFrame === 'function') {
      list = list.map((rgba, i) => {
        const next = options.mapFrame(rgba, i, list.length);
        if (!(next instanceof Uint8Array) || next.length !== rgba.length) {
          throw new VideoExportError(`mapFrame returned invalid buffer for frame ${i}`, VideoExportErrorCode.InvalidFrame);
        }
        return next;
      });
    }

    emitProgress(options?.onProgress, enrichProgress({
      phase: 'encode',
      frame: 0,
      frames: list.length,
      ratio: 0,
      format: resolved.format,
    }), options?.eventTarget);

    const transfers = [];
    const buffers = list.map((u8) => {
      const { buffer } = transferFrameBuffer(u8, { move: false });
      transfers.push(buffer);
      return buffer;
    });

    const msg = await this._call(this._ensureWorker(), {
      type: 'encode',
      wasmUrl: this._wasmUrl,
      format: resolved.format,
      width: resolved.width,
      height: resolved.height,
      fps: resolved.fps,
      transparent: resolved.transparent,
      gifColors: resolved.gifColors,
      frames: buffers,
    }, transfers);

    throwIfAborted(options?.signal);
    emitProgress(options?.onProgress, enrichProgress({
      phase: 'done',
      frame: list.length,
      frames: list.length,
      ratio: 1,
      format: resolved.format,
    }), options?.eventTarget);

    return new VideoExportResult({
      bytes: new Uint8Array(msg.bytes),
      format: resolved.format,
      width: resolved.width,
      height: resolved.height,
      fps: resolved.fps,
      frameCount: list.length,
      transparent: resolved.transparent,
      filename: resolved.filename,
    });
  }

  /**
   * Stream RGBA frames to the worker as they are captured (pipelined export).
   * @param {import('./video-export.d.ts').VideoEncodeOptions} options
   * @param {import('./video-export.d.ts').VideoEncodeStreamOptions} [streamOptions]
   * @returns {Promise<{ push: (frame: Uint8Array) => Promise<void>, finish: () => Promise<VideoExportResult>, streamId: number }>}
   */
  async beginStream(options, streamOptions = {}) {
    await this.init();
    throwIfAborted(streamOptions.signal ?? options?.signal);
    const resolved = new VideoExporter(options)._resolvedEncodeOptions();
    const batchSize = Math.max(1, streamOptions.batchSize ?? 4);
    const streamId = (this._streamSeq = (this._streamSeq || 0) + 1);
    this._activeStreamId = streamId;
    const worker = this._ensureWorker();

    await this._call(worker, {
      type: 'streamBegin',
      streamId,
      wasmUrl: this._wasmUrl,
      format: resolved.format,
      width: resolved.width,
      height: resolved.height,
      fps: resolved.fps,
      transparent: resolved.transparent,
      gifColors: resolved.gifColors,
    });

    /** @type {ArrayBuffer[]} */
    let batch = [];
    /** @type {Transferable[]} */
    let batchTransfers = [];
    let frameCount = 0;
    /** @type {Promise<void>} */
    let pushChain = Promise.resolve();

    const flush = () => {
      if (batch.length === 0) return pushChain;
      const frames = batch;
      const transfers = batchTransfers;
      batch = [];
      batchTransfers = [];
      pushChain = pushChain.then(() => new Promise((resolve, reject) => {
        try {
          worker.postMessage({ type: 'streamPush', streamId, frames }, transfers);
          resolve();
        } catch (err) {
          reject(err);
        }
      }));
      return pushChain;
    };

    const push = async (frame) => {
      throwIfAborted(streamOptions.signal ?? options?.signal);
      const { buffer } = transferFrameBuffer(frame, { move: true });
      batch.push(buffer);
      batchTransfers.push(buffer);
      frameCount += 1;
      if (batch.length >= batchSize) await flush();
    };

    const finish = async () => {
      try {
        await flush();
        await pushChain;
        throwIfAborted(streamOptions.signal ?? options?.signal);
        emitProgress(streamOptions.onProgress ?? options?.onProgress, enrichProgress({
          phase: 'encode',
          frame: frameCount,
          frames: frameCount,
          ratio: 0.95,
          format: resolved.format,
        }), streamOptions.eventTarget ?? options?.eventTarget);

        const msg = await this._call(worker, { type: 'streamFinish', streamId });

        emitProgress(streamOptions.onProgress ?? options?.onProgress, enrichProgress({
          phase: 'done',
          frame: frameCount,
          frames: frameCount,
          ratio: 1,
          format: resolved.format,
        }), streamOptions.eventTarget ?? options?.eventTarget);

        return new VideoExportResult({
          bytes: new Uint8Array(msg.bytes),
          format: resolved.format,
          width: resolved.width,
          height: resolved.height,
          fps: resolved.fps,
          frameCount,
          transparent: resolved.transparent,
          filename: resolved.filename,
        });
      } finally {
        if (this._activeStreamId === streamId) this._activeStreamId = null;
      }
    };

    return { push, finish, streamId };
  }

  terminate() {
    if (this._worker) {
      this._worker.terminate();
      this._worker = null;
    }
    this._ready = null;
    for (const [, pending] of this._pending) {
      pending.reject(new VideoExportError('worker terminated', VideoExportErrorCode.Aborted));
    }
    this._pending.clear();
  }

  /**
   * @param {Worker} worker
   * @param {object} payload
   * @param {Transferable[]} [transfer]
   */
  _call(worker, payload, transfer = []) {
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      try {
        worker.postMessage({ ...payload, id }, transfer);
      } catch (err) {
        this._pending.delete(id);
        reject(err);
      }
    });
  }
}

let sharedEncodeWorker = null;

/**
 * Shared warm worker instance (wasm stays loaded between exports).
 * @param {import('./video-export.d.ts').VideoEncodeWorkerOptions} [options]
 * @returns {VideoEncodeWorker}
 */
export function getSharedEncodeWorker(options = {}) {
  if (!sharedEncodeWorker) {
    sharedEncodeWorker = new VideoEncodeWorker(options);
  }
  return sharedEncodeWorker;
}

/**
 * Encode RGBA frames in a shared Web Worker.
 * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
 * @param {import('./video-export.d.ts').VideoEncodeOptions} options
 * @param {import('./video-export.d.ts').VideoEncodeWorkerOptions} [workerOptions]
 * @returns {Promise<VideoExportResult>}
 */
export async function encodeVideoFramesInWorker(frames, options, workerOptions = {}) {
  let worker = workerOptions.worker;
  if (!worker) {
    worker = getSharedEncodeWorker(workerOptions);
  }
  return worker.encode(frames, {
    ...options,
    onProgress: workerOptions.onProgress ?? options?.onProgress,
    signal: workerOptions.signal ?? options?.signal,
    mapFrame: workerOptions.mapFrame ?? options?.mapFrame,
    eventTarget: workerOptions.eventTarget ?? options?.eventTarget,
  });
}

/** Raw wasm bindings (present only with `native-codec`). */
export const encodeGifRgba = (...args) => {
  assertAvailable();
  return asU8(_wasm.encodeGifRgba(...args));
};
export const encodeApngRgba = (...args) => {
  assertAvailable();
  return asU8(_wasm.encodeApngRgba(...args));
};
export const encodeWebmRgba = (...args) => {
  assertAvailable();
  return asU8(_wasm.encodeWebmRgba(...args));
};
export const encodeMp4Rgba = (...args) => {
  assertAvailable();
  return asU8(_wasm.encodeMp4Rgba(...args));
};

export class VideoExportError extends Error {
  /**
   * @param {string} message
   * @param {string} [code]
   * @param {{ cause?: unknown, hint?: string }} [options]
   */
  constructor(message, code = 'VideoExportError', options = {}) {
    const text = options.hint ? `${message} — ${options.hint}` : message;
    super(text);
    this.name = 'VideoExportError';
    this.code = code;
    if (options.hint) this.hint = options.hint;
    if (options.cause !== undefined) {
      try {
        this.cause = options.cause;
      } catch {
        /* ignore non-writable cause on older engines */
      }
    }
  }

  /** @param {unknown} err */
  static is(err) {
    return err instanceof VideoExportError
      || (Boolean(err) && typeof err === 'object' && /** @type {{ name?: string }} */ (err).name === 'VideoExportError');
  }
}

/**
 * @param {ArrayLike<Uint8Array|ArrayBuffer|ArrayBufferView>} frames
 * @param {{
 *   width: number, height: number, fps: number, format: VideoFormatName,
 *   transparent: boolean, gifColors: number, filename: string, strictSize?: boolean,
 * }} opts
 * @param {((info: object) => void)|null} onProgress
 * @param {AbortSignal|null} signal
 * @param {((rgba: Uint8Array, i: number, n: number) => Uint8Array)|null} [mapFrame]
 * @returns {VideoExportResult}
 */
function encodeWithOptions(frames, opts, onProgress, signal, mapFrame = null, eventTarget = null) {
  assertAvailable();
  throwIfAborted(signal);
  const { width, height, fps, format, transparent, gifColors, filename } = opts;

  if (format === 'webm' && (width % 8 !== 0 || height % 8 !== 0)) {
    const aligned = alignVideoSize(width, height);
    throw new VideoExportError(
      `WebM/VP9 requires width and height multiples of 8 (got ${width}×${height})`,
      VideoExportErrorCode.InvalidDimensions,
      { hint: `use ${aligned.width}×${aligned.height}, or export via VideoExporter.from() which snaps capture size` },
    );
  }

  if (format === 'mp4' && transparent) {
    throw new VideoExportError(
      'MP4/H.264 does not support transparency',
      VideoExportErrorCode.InvalidOption,
    );
  }

  if (format === 'mp4' && (width % 2 !== 0 || height % 2 !== 0)) {
    const aligned = alignVideoSizeForMp4(width, height);
    throw new VideoExportError(
      `MP4/H.264 requires even width and height (got ${width}×${height})`,
      VideoExportErrorCode.InvalidDimensions,
      { hint: `use ${aligned.width}×${aligned.height}, or export via VideoExporter.from() which snaps capture size` },
    );
  }

  let list = normalizeFrameList(frames, width * height * 4);

  if (typeof mapFrame === 'function') {
    list = list.map((rgba, i) => {
      const next = mapFrame(rgba, i, list.length);
      if (!(next instanceof Uint8Array) || next.length !== rgba.length) {
        throw new VideoExportError(`mapFrame returned invalid buffer for frame ${i}`, VideoExportErrorCode.InvalidFrame);
      }
      return next;
    });
  }

  const total = list.length;
  emitProgress(onProgress, enrichProgress({
    phase: 'encode',
    frame: 0,
    frames: total,
    ratio: 0,
    format,
  }), eventTarget);
  throwIfAborted(signal);

  let bytes;
  try {
    if (format === 'gif') {
      bytes = _wasm.encodeGifRgba(width, height, fps, gifColors, transparent, list);
    } else if (format === 'apng') {
      bytes = _wasm.encodeApngRgba(width, height, fps, transparent, list);
    } else if (format === 'webm') {
      bytes = _wasm.encodeWebmRgba(width, height, fps, transparent, list);
    } else {
      bytes = _wasm.encodeMp4Rgba(width, height, fps, list);
    }
  } catch (e) {
    if (VideoExportError.is(e)) throw e;
    throw new VideoExportError(String(/** @type {{ message?: string }} */ (e)?.message || e), VideoExportErrorCode.EncodeFailed, { cause: e });
  }

  emitProgress(onProgress, enrichProgress({
    phase: 'done',
    frame: total,
    frames: total,
    ratio: 1,
    format,
  }), eventTarget);

  return new VideoExportResult({
    bytes: asU8(bytes),
    format,
    width,
    height,
    fps,
    frameCount: total,
    transparent,
    filename,
  });
}

function enrichProgress(info) {
  return { ...info, message: formatVideoProgress(info) };
}

/**
 * Fire callback + EventTarget progress events (`progress`, plus `capture`/`encode`).
 * @param {((info: object) => void)|null|undefined} cb
 * @param {object} info
 * @param {EventTarget|null|undefined} eventTarget
 */
export function emitProgress(cb, info, eventTarget = null) {
  const enriched = info?.message != null ? info : enrichProgress(info || {});
  if (typeof cb === 'function') cb(enriched);
  if (eventTarget && typeof eventTarget.dispatchEvent === 'function') {
    const targets = typeof eventTarget._emitTargets === 'function'
      ? eventTarget._emitTargets()
      : [eventTarget];
    for (const t of targets) {
      t.dispatchEvent(new VideoExportProgressEvent(VideoExportEvent.Progress, enriched));
      if (enriched.phase === 'capture') {
        t.dispatchEvent(new VideoExportProgressEvent(VideoExportEvent.Capture, enriched));
      } else if (enriched.phase === 'encode') {
        t.dispatchEvent(new VideoExportProgressEvent(VideoExportEvent.Encode, enriched));
      }
    }
  }
  return enriched;
}

/** @deprecated Use {@link emitProgress}. */
function report(cb, info, eventTarget = null) {
  return emitProgress(cb, info, eventTarget);
}

function throwIfAborted(signal) {
  if (signal?.aborted) {
    throw new VideoExportError('Video export aborted', VideoExportErrorCode.Aborted, { cause: signal.reason });
  }
}

function assertAvailable() {
  if (!isVideoExportAvailable()) {
    throw new VideoExportError(
      'Video export unavailable',
      VideoExportErrorCode.Unavailable,
      { hint: 'rebuild wasm with native-codec (NATIVE_CODEC=1 web/build.sh)' },
    );
  }
}

function normalizeFormat(format) {
  return parseVideoFormat(format).format;
}

/**
 * @param {string|undefined|null} name
 * @param {VideoFormatName} format
 * @param {boolean} transparent
 */
function ensureFilename(name, format, transparent) {
  const raw = String(name || 'export').trim() || 'export';
  if (/\.(gif|png|apng|webm|mp4)$/i.test(raw)) return raw;
  return videoFilename(format, raw, transparent);
}

function normalizeFrameList(frames, expected) {
  if (!frames || typeof frames.length !== 'number' || frames.length < 1) {
    throw new VideoExportError('need at least one RGBA frame', VideoExportErrorCode.EmptyFrames);
  }
  const out = [];
  for (let i = 0; i < frames.length; i++) {
    const raw = frames[i];
    let u8;
    if (raw instanceof Uint8Array) u8 = raw;
    else if (ArrayBuffer.isView(raw)) u8 = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
    else if (raw instanceof ArrayBuffer) u8 = new Uint8Array(raw);
    else throw new VideoExportError(`frame ${i}: expected Uint8Array`, VideoExportErrorCode.InvalidFrame);
    if (u8.length !== expected) {
      throw new VideoExportError(
        `frame ${i}: got ${u8.length} bytes, expected ${expected} (${expected / 4} pixels)`,
        VideoExportErrorCode.FrameSize,
      );
    }
    out.push(u8);
  }
  return out;
}

function u32(v, name) {
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0 || (n | 0) !== n) {
    throw new VideoExportError(`${name} must be a positive integer`, VideoExportErrorCode.InvalidOption);
  }
  return n;
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, Number(v) || lo));
}

function asU8(bytes) {
  return bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
}

/** @param {number} n */
export function formatBytes(n) {
  const v = Number(n) || 0;
  if (v < 1024) return `${v} B`;
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(v < 10 * 1024 ? 1 : 0)} KB`;
  return `${(v / (1024 * 1024)).toFixed(v < 10 * 1024 * 1024 ? 2 : 1)} MB`;
}
