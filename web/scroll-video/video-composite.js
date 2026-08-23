/**
 * Composite an HTMLVideoElement with a threers scene.
 *
 * Display modes:
 * - canvas-3d: video sampled into a screen-filling 3D plane behind the overlay
 * - html-layer: visible HTML <video> under a transparent WebGL canvas
 */

import THREE from '../threejs-shim.js';
import { VideoElementTexture, isSafariBrowser } from './video-element-texture.js';
import { drawImageCover } from './video-utils.js';

export const DisplayMode = {
    /** Flat fullscreen video plane behind 3D overlay (locked camera). */
    CANVAS_3D: 'canvas-3d',
    /** Visible HTML video under transparent WebGL. */
    HTML_LAYER: 'html-layer',
};

/** World-space Z for the flat backdrop plane (behind overlay objects). */
const BACKDROP_PLANE_Z = -8;

/** Absolute cap — keeps Safari memory bounded. */
const ABS_MAX_TEX_W = 1280;
const ABS_MAX_TEX_H = 720;

function cappedPixelRatio(max) {
    const raw = typeof devicePixelRatio !== 'undefined' ? devicePixelRatio : 1;
    return Math.min(raw, max);
}

/**
 * Target video texture resolution — match stage pixels (capped).
 * @param {HTMLCanvasElement} canvas
 * @param {number} dpr
 */
export function textureBudgetForCanvas(canvas, dpr) {
    const cw = canvas && canvas.clientWidth ? canvas.clientWidth : 960;
    const ch = canvas && canvas.clientHeight ? canvas.clientHeight : 540;
    const w = Math.min(ABS_MAX_TEX_W, Math.ceil(cw * dpr));
    const h = Math.min(ABS_MAX_TEX_H, Math.ceil(ch * dpr));
    return {
        w: Math.max(480, w),
        h: Math.max(270, h),
    };
}

/**
 * Video + 3D compositor driven by video playback time.
 */
export class VideoSceneComposite {
    constructor(options) {
        options = options || {};
        this.canvas = options.canvas;
        this.video = options.video;
        this.displayMode = options.displayMode || DisplayMode.CANVAS_3D;
        this.maxTextureWidth = options.maxTextureWidth || ABS_MAX_TEX_W;
        this.maxTextureHeight = options.maxTextureHeight || ABS_MAX_TEX_H;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.plane = null;
        this.planeMat = null;
        this._videoTex = null;
        this._mixers = [];
        this._texW = 0;
        this._texH = 0;
        this._pendingRvfc = false;
    }

    _pixelRatio() {
        return cappedPixelRatio(isSafariBrowser() ? 1.5 : 2);
    }

    _updateTextureBudget() {
        if (!this.canvas) return;
        const budget = textureBudgetForCanvas(this.canvas, this._pixelRatio());
        this.maxTextureWidth = budget.w;
        this.maxTextureHeight = budget.h;
        if (this._videoTex) {
            this._videoTex.maxWidth = budget.w;
            this._videoTex.maxHeight = budget.h;
        }
    }

    _msaaSamples() {
        if (this.usesHtmlLayer()) return 1;
        return isSafariBrowser() ? 2 : 4;
    }

    _applyMsaa() {
        if (!this.renderer || !this.renderer._w || !this.renderer._w.setMsaa) return;
        this.renderer._w.setMsaa(this._msaaSamples());
    }

    async initRenderer() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        this.renderer = await THREE.WebGLRenderer.create(this.canvas);
        const dpr = this._pixelRatio();
        const w = Math.floor(this.canvas.clientWidth * dpr) || 640;
        const h = Math.floor(this.canvas.clientHeight * dpr) || 360;
        this.renderer.setSize(w, h, false);
        this._updateTextureBudget();
        this._applyDisplayMode();
        return this;
    }

    usesHtmlLayer() {
        return this.displayMode === DisplayMode.HTML_LAYER;
    }

    usesCanvas3d() {
        return !this.usesHtmlLayer();
    }

    /** @deprecated alias */
    usesBackdrop() {
        return this.usesCanvas3d();
    }

    _setSceneTransparent(transparent) {
        if (!this.scene) return;
        if (this.scene._w && this.scene._w.setBackgroundAlpha) {
            this.scene.backgroundAlpha = transparent ? 0 : 1;
            this.scene._w.setBackgroundAlpha(transparent ? 0 : 1);
        }
        if (!transparent) {
            this.scene.background = new THREE.Color(0x000000);
        }
    }

    _syncHtmlVideoVisibility() {
        const video = this.video;
        if (!video) return;
        if (this.usesHtmlLayer()) {
            video.style.removeProperty('display');
            video.style.removeProperty('visibility');
            video.style.removeProperty('opacity');
            video.style.removeProperty('z-index');
        } else {
            video.style.removeProperty('display');
            video.style.visibility = 'hidden';
            video.style.opacity = '0';
            video.style.zIndex = '-1';
        }
    }

    _ensureVideoPlane() {
        if (this.plane) return;
        this._videoTex = new VideoElementTexture(this.video, this.maxTextureWidth, this.maxTextureHeight, {
            maxWidth: this.maxTextureWidth,
            maxHeight: this.maxTextureHeight,
        });
        this.planeMat = new THREE.MeshBasicMaterial({
            map: this._videoTex.texture,
            toneMapped: false,
            side: THREE.DoubleSide,
            depthWrite: true,
        });
        this._videoTex.bindMaterial(this.planeMat);
        this.plane = new THREE.Mesh(new THREE.PlaneGeometry(16, 9), this.planeMat);
        this.plane.renderOrder = -1000;
        this.scene.add(this.plane);
        this._layoutVideoPlane();
        this._texW = this._videoTex.width;
        this._texH = this._videoTex.height;
    }

    _syncMapToWasm() {
        if (!this.planeMat || !this.planeMat._w || !this._videoTex) return;
        const tex = this._videoTex.texture;
        if (tex && tex._w && this.planeMat._w.setMapData) {
            this.planeMat._w.setMapData(tex._w);
        }
    }

    _rebindVideoPlaneMaterial(force) {
        if (!this.plane || !this.planeMat || !this.scene || !this.scene._w) return;
        if (!force && this._videoTex && this._videoTex._materialBound) return;
        if (this.plane._handle && this.scene._w.setMeshMaterial) {
            this.scene._w.setMeshMaterial(this.plane._handle, this.planeMat._w);
            if (this._videoTex) this._videoTex._materialBound = true;
        }
    }

    _removeVideoPlane() {
        if (this._videoTex) this._videoTex.cancelRvfc();
        if (this.plane && this.scene) {
            if (this.plane._handle && this.scene._w && this.scene._w.remove) {
                this.scene._w.remove(this.plane._handle);
            }
            this.scene.remove(this.plane);
            this.plane._handle = null;
        }
        this.plane = null;
        this.planeMat = null;
        this._videoTex = null;
        this._texW = 0;
        this._texH = 0;
        this._pendingRvfc = false;
    }

    fitBackdropPlane(camera) {
        if (!this.plane || !camera) return;
        this._layoutVideoPlane(camera);
    }

    _layoutVideoPlane(camera) {
        if (!this.plane) return;
        const cam = camera || this.camera;
        if (this.usesCanvas3d() && cam) {
            const planeZ = BACKDROP_PLANE_Z;
            const dist = cam.position.z - planeZ;
            if (dist > 0) {
                const vFov = ((cam.fov || 40) * Math.PI) / 180;
                const visibleHeight = 2 * Math.tan(vFov / 2) * dist;
                const visibleWidth = visibleHeight * (cam.aspect || 1);
                this.plane.scale.set(visibleWidth / 16, visibleHeight / 9, 1);
                this.plane.position.set(0, 0, planeZ);
                this.plane.rotation.set(0, 0, 0);
                return;
            }
        }
        this.plane.scale.set(1, 1, 1);
        this.plane.position.set(0, 0, -1.5);
    }

    _applyDisplayMode() {
        if (this.usesHtmlLayer()) {
            this._setSceneTransparent(true);
            this._removeVideoPlane();
        } else {
            this._setSceneTransparent(false);
            this._ensureVideoPlane();
            this.fitBackdropPlane(this.camera);
        }
        this._applyMsaa();
        this._syncHtmlVideoVisibility();
    }

    /** @param {string} mode DisplayMode value */
    setDisplayMode(mode) {
        this.displayMode = mode === DisplayMode.HTML_LAYER
            ? DisplayMode.HTML_LAYER
            : DisplayMode.CANVAS_3D;
        this._applyDisplayMode();
    }

    setCamera(camera) {
        this.camera = camera;
        if (this.usesCanvas3d()) {
            this.fitBackdropPlane(camera);
        }
    }

    addAnimationMixer(root) {
        const mixer = new THREE.AnimationMixer(root);
        this._mixers.push(mixer);
        return mixer;
    }

    syncAnimationsToTime(t) {
        for (let i = 0; i < this._mixers.length; i++) {
            const mixer = this._mixers[i];
            const actions = mixer.actions || [];
            for (let j = 0; j < actions.length; j++) {
                const action = actions[j];
                if (!action.enabled) continue;
                if (!action.isRunning) action.play();
                action.paused = true;
                action.time = t;
            }
            mixer.update(0);
        }
    }

    _finishVideoUpload(t, fallbackPaint) {
        if (!this._videoTex) return;
        const result = this._videoTex.update(t, fallbackPaint);
        if (!result || !result.dirty) return;
        if (result.resized) {
            this._texW = this._videoTex.width;
            this._texH = this._videoTex.height;
            this._videoTex._materialBound = false;
        }
        this._syncMapToWasm();
        this._rebindVideoPlaneMaterial(true);
    }

    /**
     * Upload the current video frame into the 3D backdrop plane texture.
     * @param {number} t
     * @param {(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => void} [fallbackPaint]
     */
    updateVideoFrame(t, fallbackPaint) {
        if (this.usesHtmlLayer()) return;
        if (!this._videoTex) {
            this._ensureVideoPlane();
        }
        if (!this._videoTex) return;
        if (this._videoTex) this._videoTex.cancelRvfc();
        this._finishVideoUpload(t, fallbackPaint);
    }

    render() {
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    resize() {
        if (!this.renderer || !this.camera || !this.canvas) return;
        const dpr = this._pixelRatio();
        const w = Math.floor(this.canvas.clientWidth * dpr);
        const h = Math.floor(this.canvas.clientHeight * dpr);
        if (w < 16 || h < 16) return;
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this._updateTextureBudget();
        if (this.usesCanvas3d()) {
            this.fitBackdropPlane(this.camera);
        }
    }
}

/**
 * Read playback time from a video element (clamped).
 * @param {HTMLVideoElement} video
 * @returns {number}
 */
export function videoPlaybackTime(video) {
    if (!video) return 0;
    const time = video.currentTime;
    return typeof time === 'number' && isFinite(time) ? time : 0;
}

export { drawImageCover };
