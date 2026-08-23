/**
 * Fast video → texture upload for canvas-3D backdrop mode.
 *
 * Uses requestVideoFrameCallback when the decode clock matches scroll time;
 * falls back to canvas + replaceRgba (no per-frame WebDataTexture alloc).
 */

import THREE from '../threejs-shim.js';
import { drawImageCover } from './video-utils.js';

function isSafariBrowser() {
    if (typeof navigator === 'undefined') return false;
    return /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
}

/**
 * Upload frames from an HTMLVideoElement into a CanvasTexture-backed plane map.
 */
export class VideoElementTexture {
    constructor(video, width, height, opts) {
        opts = opts || {};
        this.video = video;
        this.maxWidth = opts.maxWidth || 1280;
        this.maxHeight = opts.maxHeight || 720;
        const fit = this._fitSize(width || this.maxWidth, height || this.maxHeight);
        this.width = fit.w;
        this.height = fit.h;
        this._canvas = document.createElement('canvas');
        this._canvas.width = this.width;
        this._canvas.height = this.height;
        this._ctx = this._canvas.getContext('2d', {
            alpha: false,
            desynchronized: true,
            willReadFrequently: true,
        });
        if (this._ctx) {
            this._ctx.imageSmoothingEnabled = true;
            if (this._ctx.imageSmoothingQuality) {
                this._ctx.imageSmoothingQuality = 'high';
            }
        }
        this.texture = new THREE.CanvasTexture(this._canvas);
        this.texture.magFilter = 1006;
        this.texture.minFilter = 1008;
        if (typeof this.texture._syncFilters === 'function') {
            this.texture._syncFilters();
        }
        this._planeMat = null;
        this._lastUploadT = -1;
        this._materialBound = false;
        this._rvfcId = null;
        this._rvfcPending = false;
        this._rvfcCallback = null;
    }

    bindMaterial(mat) {
        this._planeMat = mat;
        this._materialBound = false;
    }

    _fitSize(w, h) {
        if (!w || !h) return { w: this.maxWidth, h: this.maxHeight };
        if (w <= this.maxWidth && h <= this.maxHeight) return { w, h };
        const scale = Math.min(this.maxWidth / w, this.maxHeight / h);
        return {
            w: Math.max(2, Math.round(w * scale)),
            h: Math.max(2, Math.round(h * scale)),
        };
    }

    cancelRvfc() {
        const video = this.video;
        if (video && video.cancelVideoFrameCallback && this._rvfcId != null) {
            try { video.cancelVideoFrameCallback(this._rvfcId); } catch (_err) { /* ignore */ }
        }
        this._rvfcId = null;
        this._rvfcPending = false;
        this._rvfcCallback = null;
    }

    /**
     * Schedule upload on the next decoded frame (lower CPU when not scrubbing).
     * @param {number} t
     * @param {(t: number, fallbackPaint?: Function) => void} onFrame
     * @returns {boolean}
     */
    scheduleRvfc(t, onFrame) {
        const video = this.video;
        if (!video || !video.requestVideoFrameCallback) return false;
        if (video.readyState < 2) return false;
        if (Math.abs(video.currentTime - t) > 0.05) return false;
        this._rvfcCallback = onFrame;
        if (this._rvfcPending) return true;
        this._rvfcPending = true;
        this._rvfcId = video.requestVideoFrameCallback(() => {
            this._rvfcId = null;
            this._rvfcPending = false;
            const cb = this._rvfcCallback;
            this._rvfcCallback = null;
            if (cb) cb(t);
        });
        return true;
    }

    /**
     * @param {number} t
     * @param {(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => void} [fallbackPaint]
     * @param {boolean} [force]
     */
    update(t, fallbackPaint, force) {
        const ctx = this._ctx;
        const source = this.video;
        const hasFrame = source && source.readyState >= 2;

        if (!force && !fallbackPaint && this._lastUploadT >= 0
            && Math.abs(t - this._lastUploadT) < 0.0005) {
            return { texture: this.texture, dirty: false };
        }

        let sourceW = this.width;
        let sourceH = this.height;
        if (hasFrame && source.videoWidth > 0 && source.videoHeight > 0) {
            sourceW = source.videoWidth;
            sourceH = source.videoHeight;
        }
        const fit = this._fitSize(sourceW, sourceH);
        const w = fit.w;
        const h = fit.h;
        const resized = this._canvas.width !== w || this._canvas.height !== h;
        if (resized) {
            this._canvas.width = w;
            this._canvas.height = h;
        }
        this.width = w;
        this.height = h;

        if (!hasFrame) {
            if (fallbackPaint) {
                fallbackPaint(ctx, w, h, t);
            } else {
                ctx.fillStyle = '#0a0c10';
                ctx.fillRect(0, 0, w, h);
            }
        }

        if (hasFrame) {
            try {
                drawImageCover(ctx, source, w, h);
            } catch (_err) {
                if (fallbackPaint) {
                    fallbackPaint(ctx, w, h, t);
                }
            }
        }

        this.texture.update();
        if (typeof this.texture._syncFilters === 'function') {
            this.texture._syncFilters();
        }
        if (this._planeMat) {
            this._planeMat.map = this.texture;
        }
        this._lastUploadT = t;
        return { texture: this.texture, dirty: true, resized };
    }
}

export { isSafariBrowser };
