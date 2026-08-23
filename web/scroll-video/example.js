/**
 * Minimal scroll-driven video + threers overlay (production template).
 */

import THREE, { initThreers } from '../threejs-shim.js';
import { ScrollVideoSync } from './scroll-sync.js';
import { VideoSceneComposite } from './video-composite.js';
import { loadStageVideo } from './clip-loader.js';
import { paintClipFrame, DEFAULT_CLIP_DURATION } from './paint-clip.js';

const videoEl = document.getElementById('v');
const canvas = document.getElementById('c');
const hudEl = document.getElementById('hud');
const errEl = document.getElementById('err');

/** Default mapping — replace with sync.toJSON() from calibration. */
const SCROLL_CONFIG = {
    mode: 'progress',
    scrollStart: 0,
    scrollEnd: 3000,
    videoDuration: DEFAULT_CLIP_DURATION,
    clipIn: 0,
    clipOut: DEFAULT_CLIP_DURATION,
    pxPerSecond: 900,
    velocityGain: 1,
    smoothing: 0.18,
    easing: false,
    markers: [],
};

let sync = ScrollVideoSync.fromJSON(SCROLL_CONFIG);
let composite = null;
let camera = null;

function resize() {
    if (!composite || !camera || !canvas) return;
    composite.resize();
}

async function main() {
    try {
        await initThreers({ module_or_path: '../pkg/threers_bg.wasm' });
        const duration = await loadStageVideo(videoEl);
        sync.videoDuration = duration;
        sync.clipOut = duration;

        composite = new VideoSceneComposite({
            canvas,
            video: videoEl,
        });
        await composite.initRenderer();

        composite.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const key = new THREE.DirectionalLight(0xffffff, 1.2);
        key.position.set(2, 4, 6);
        composite.scene.add(key);

        const ring = new THREE.Mesh(
            new THREE.TorusGeometry(1.2, 0.06, 12, 64),
            new THREE.MeshStandardMaterial({
                color: 0x6fb3ff,
                emissive: 0x143350,
                emissiveIntensity: 0.5,
                metalness: 0.6,
                roughness: 0.25,
            }),
        );
        ring.position.set(0, 0.2, 1);
        composite.scene.add(ring);

        const dpr = Math.min(devicePixelRatio || 1, 2);
        const w = Math.floor(canvas.clientWidth * dpr) || 1280;
        const h = Math.floor(canvas.clientHeight * dpr) || 720;
        camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
        camera.position.set(0, 0, 6.8);
        camera.lookAt(0, 0, 0);
        composite.setCamera(camera);
        resize();
        addEventListener('resize', resize);

        function frame(now) {
            requestAnimationFrame(frame);
            const scrollY = window.scrollY;
            sync.update({ scrollY, now });
            sync.applyToVideo(videoEl);

            const t = sync.time;
            const durationSec = sync.videoDuration || DEFAULT_CLIP_DURATION;
            const needsFallback = videoEl.readyState < 2 || !videoEl.videoWidth;

            composite.updateVideoFrame(t, needsFallback
                ? function (ctx, cw, ch, time) {
                    paintClipFrame(ctx, cw, ch, time, durationSec);
                }
                : undefined);
            ring.rotation.x = t * 0.6;
            ring.rotation.y = t * 0.9;

            if (hudEl) hudEl.textContent = `${t.toFixed(1)} s`;
            composite.render();
        }
        requestAnimationFrame(frame);
    } catch (e) {
        if (errEl) {
            errEl.hidden = false;
            errEl.textContent = (e && e.stack) || String(e);
        }
        console.error(e);
    }
}

void main();
