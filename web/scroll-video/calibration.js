// Scroll-synced video with threers overlay — calibration playground.
//
// The video sits on a fullscreen plane; 3D annotations float above it.
// Tune mapping in the sidebar, export JSON when markers line up.

import THREE, { initThreers } from '../threejs-shim.js';
import {
    ScrollVideoSync,
    autoFitScrollRange,
    readScrollMarkers,
    attachMarkerResizeObserver,
} from './scroll-sync.js';
import {
    DisplayMode,
    VideoSceneComposite,
} from './video-composite.js';
import { paintClipFrame } from './paint-clip.js';
import { isSafariBrowser } from './video-element-texture.js';
import { loadStageVideo, loadUserVideoFile } from './clip-loader.js';
import { drawImageCover } from './video-utils.js';

const errEl = document.getElementById('err');
const statusEl = document.getElementById('status');
const videoEl = document.getElementById('v');
const previewCanvasEl = document.getElementById('v-preview-canvas');
const canvas = document.getElementById('c');
const exportEl = document.getElementById('export');
const stageEl = document.getElementById('stage');
const videoFallbackEl = document.getElementById('video-fallback');
const copyScrollEl = document.getElementById('copy-scroll');
const stageWrapEl = document.getElementById('stage-wrap');

const markerReadOpts = function () {
    return { scrollRoot: copyScrollEl };
};

function getScrollY() {
    return copyScrollEl ? copyScrollEl.scrollTop : window.scrollY;
}

function getScrollSpan() {
    if (!copyScrollEl) return Math.max(document.documentElement.scrollHeight - window.innerHeight, 1);
    return Math.max(copyScrollEl.scrollHeight - copyScrollEl.clientHeight, 1);
}

const ui = {
    displayMode: document.getElementById('display-mode'),
    mode: document.getElementById('mode'),
    easing: document.getElementById('easing'),
    scrollStart: document.getElementById('scroll-start'),
    scrollEnd: document.getElementById('scroll-end'),
    pxPerSec: document.getElementById('px-per-sec'),
    velocityGain: document.getElementById('velocity-gain'),
    smoothing: document.getElementById('smoothing'),
    clipIn: document.getElementById('clip-in'),
    clipOut: document.getElementById('clip-out'),
    fitMarkers: document.getElementById('fit-markers'),
    resetRange: document.getElementById('reset-range'),
    copyJson: document.getElementById('copy-json'),
    importJson: document.getElementById('import-json'),
    videoFile: document.getElementById('video-file'),
    liveScroll: document.getElementById('live-scroll'),
    liveSpeed: document.getElementById('live-speed'),
    liveTime: document.getElementById('live-time'),
    liveRate: document.getElementById('live-rate'),
    markerErrors: document.getElementById('marker-errors'),
    timelineFill: document.getElementById('timeline-fill'),
    timeBadge: document.getElementById('time-badge'),
    modeChip: document.getElementById('mode-chip'),
    scrollProgress: document.getElementById('scroll-progress'),
    vStart: document.getElementById('v-start'),
    vEnd: document.getElementById('v-end'),
    vPxps: document.getElementById('v-pxps'),
    vGain: document.getElementById('v-gain'),
    vSmooth: document.getElementById('v-smooth'),
    vIn: document.getElementById('v-in'),
    vOut: document.getElementById('v-out'),
};

let sync = new ScrollVideoSync({ videoDuration: 10, scrollEnd: 4000 });

let composite = null;
let camera = null;
const CAMERA_REST = { x: 0, y: 0, z: 6.8, lookX: 0, lookY: 0, lookZ: 0 };
let ring = null;
let torus = null;
let progressFill = null;
let markerPins = [];
let cornerOrbs = [];
let useFallbackVideo = false;
let exportDirty = true;
let markerPanelDirty = true;
let lastActiveMarkerIdx = -1;
let lastMarkerPanelAt = 0;

let lastPreviewAt = 0;

/** Pick a MediaRecorder mime the browser actually supports. */
function pickRecorderMime() {
    const candidates = [
        'video/webm;codecs=vp9',
        'video/webm;codecs=vp8',
        'video/webm',
    ];
    for (const mime of candidates) {
        if (MediaRecorder.isTypeSupported(mime)) return mime;
    }
    return '';
}

/**
 * Record a short labelled test clip — no external assets required.
 *
 * @param {number} durationSec
 * @returns {Promise<string>} object URL
 */
function recordTestClip(durationSec = 10) {
    const w = 854;
    const h = 480;
    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    const ctx = off.getContext('2d');
    if (!ctx) return Promise.reject(new Error('2d context unavailable'));

    const fps = 24;
    const stream = off.captureStream(fps);
    const mime = pickRecorderMime();
    if (!mime) return Promise.reject(new Error('MediaRecorder not supported in this browser'));

    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 1500000 });
    const chunks = [];

    return new Promise((resolve, reject) => {
        rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
        rec.onerror = () => reject(rec.error || new Error('MediaRecorder failed'));
        rec.onstop = () => {
            if (!chunks.length) {
                reject(new Error('recorder produced no data'));
                return;
            }
            const blob = new Blob(chunks, { type: mime });
            resolve(URL.createObjectURL(blob));
        };

        let frame = 0;
        const totalFrames = Math.ceil(durationSec * fps);
        const tick = () => {
            const t = frame / fps;
            paintClipFrame(ctx, w, h, t, durationSec);
            frame += 1;
            if (frame <= totalFrames) {
                setTimeout(tick, 1000 / fps);
            } else {
                rec.stop();
            }
        };

        rec.start(100);
        tick();
    });
}

function paintFallbackFrame(t, duration) {
    duration = duration != null ? duration : sync.videoDuration || 10;
    if (composite && composite.usesCanvas3d()) {
        composite.updateVideoFrame(t, function (ctx, w, h, time) {
            paintClipFrame(ctx, w, h, time, duration);
        });
        return;
    }
    if (!videoFallbackEl) return;
    const ctx = videoFallbackEl.getContext('2d');
    if (!ctx) return;
    paintClipFrame(ctx, videoFallbackEl.width, videoFallbackEl.height, t, duration);
}

function updatePreviewCanvas(force) {
    if (!previewCanvasEl) return;
    const now = performance.now();
    if (!force && now - lastPreviewAt < 300) return;
    lastPreviewAt = now;
    const ctx = previewCanvasEl.getContext('2d');
    if (!ctx) return;
    if (videoEl.readyState < 2) {
        paintClipFrame(ctx, previewCanvasEl.width, previewCanvasEl.height, sync.time, sync.videoDuration || 10);
        return;
    }
    try {
        drawImageCover(ctx, videoEl, previewCanvasEl.width, previewCanvasEl.height);
    } catch (_err) { /* ignore */ }
}

async function loadTestClip(url) {
    await loadStageVideo(videoEl, url);
    useFallbackVideo = false;
    if (stageEl) stageEl.classList.remove('use-fallback');
    updatePreviewCanvas(true);
}

function setFallbackVideo(enabled) {
    useFallbackVideo = enabled;
    if (stageEl) stageEl.classList.toggle('use-fallback', enabled);
}

function syncFromUi() {
    sync.mode = ui.mode.value;
    sync.easing = !!(ui.easing && ui.easing.checked);
    sync.scrollStart = Number(ui.scrollStart.value);
    sync.scrollEnd = Number(ui.scrollEnd.value);
    sync.pxPerSecond = Number(ui.pxPerSec.value);
    sync.velocityGain = Number(ui.velocityGain.value);
    sync.smoothing = Number(ui.smoothing.value);
    sync.clipIn = Number(ui.clipIn.value);
    sync.clipOut = Number(ui.clipOut.value);
    sync.videoDuration = Math.max(Number(ui.clipOut.value), videoEl.duration || 10);
}

function updateLiveUi() {
    ui.vStart.textContent = `${sync.scrollStart | 0} px`;
    ui.vEnd.textContent = `${sync.scrollEnd | 0} px`;
    ui.vPxps.textContent = String(sync.pxPerSecond | 0);
    ui.vGain.textContent = `${sync.velocityGain.toFixed(2)}×`;
    ui.vSmooth.textContent = sync.smoothing.toFixed(2);
    ui.vIn.textContent = `${sync.clipIn.toFixed(1)} s`;
    const outLabel = sync.clipOut != null ? sync.clipOut.toFixed(1) : '—';
    ui.vOut.textContent = `${outLabel} s`;
}

function refreshExport() {
    if (!exportDirty) return;
    exportEl.value = JSON.stringify(sync.toJSON(), null, 2);
    exportDirty = false;
}

function applySyncToUi() {
    ui.mode.value = sync.mode;
    if (ui.easing) ui.easing.checked = !!sync.easing;
    ui.scrollStart.value = String(sync.scrollStart | 0);
    ui.scrollEnd.value = String(sync.scrollEnd | 0);
    ui.pxPerSec.value = String(sync.pxPerSecond | 0);
    ui.velocityGain.value = String(sync.velocityGain);
    ui.smoothing.value = String(sync.smoothing);
    ui.clipIn.value = String(sync.clipIn);
    if (sync.clipOut != null) ui.clipOut.value = String(sync.clipOut);
    updateLiveUi();
    exportDirty = true;
    refreshExport();
    markerPanelDirty = true;
}

function readUiIntoSync() {
    syncFromUi();
    updateLiveUi();
    refreshExport();
}

function refreshMarkerPanel(force) {
    if (!force && !markerPanelDirty) return;
    markerPanelDirty = false;
    lastMarkerPanelAt = performance.now();
    const rows = sync.markerErrors();
    if (!rows.length) {
        ui.markerErrors.textContent = '—';
        return;
    }
    ui.markerErrors.innerHTML = rows.map((r) => {
        const ms = (r.error * 1000).toFixed(0);
        const cls = Math.abs(r.error) < 0.08 ? 'ok' : 'warn';
        return `<div class="${cls}">${r.label || r.time + 's'}: ${ms} ms</div>`;
    }).join('');
}

function updateActiveMarker(scrollY) {
    const markers = sync.markers;
    if (!markers.length) return;
    let idx = 0;
    for (let i = 0; i < markers.length; i++) {
        if (scrollY + 80 >= markers[i].scrollY) idx = i;
    }
    if (idx === lastActiveMarkerIdx) return;
    lastActiveMarkerIdx = idx;
    document.querySelectorAll('section.marker[data-scroll-marker]').forEach(function (el, i) {
        el.classList.toggle('active', i === idx);
    });
}

function updateScrollRail(scrollY) {
    if (!ui.scrollProgress || !copyScrollEl) return;
    const span = getScrollSpan();
    const pct = Math.min(Math.max(scrollY / span, 0), 1);
    ui.scrollProgress.style.height = `${pct * copyScrollEl.clientHeight}px`;
}

function updateModeChip() {
    if (!ui.modeChip) return;
    const ease = sync.easing && sync.mode === 'progress' ? ' · eased' : '';
    const display = composite && composite.usesHtmlLayer() ? 'html layer' : 'canvas 3D';
    ui.modeChip.textContent = `${sync.mode} · ${display}${ease}`;
}

function applyDisplayMode() {
    if (!composite) return;
    const mode = ui.displayMode && ui.displayMode.value === DisplayMode.HTML_LAYER
        ? DisplayMode.HTML_LAYER
        : DisplayMode.CANVAS_3D;
    composite.setDisplayMode(mode);
    if (stageEl) {
        stageEl.classList.toggle('canvas-3d-mode', mode === DisplayMode.CANVAS_3D);
        stageEl.classList.toggle('html-layer-mode', mode === DisplayMode.HTML_LAYER);
    }
    if (stageWrapEl) {
        stageWrapEl.classList.toggle('html-layer-mode', mode === DisplayMode.HTML_LAYER);
    }
    if (mode === DisplayMode.HTML_LAYER && videoEl) {
        if (videoEl.src && videoEl.readyState < 2) {
            try { videoEl.load(); } catch (_err) { /* ignore */ }
        }
        try {
            videoEl.currentTime = sync.time;
        } catch (_err) { /* ignore */ }
    }
    if (mode === DisplayMode.CANVAS_3D) {
        setFallbackVideo(false);
        if (camera) {
            camera.position.set(CAMERA_REST.x, CAMERA_REST.y, CAMERA_REST.z);
            camera.lookAt(CAMERA_REST.lookX, CAMERA_REST.lookY, CAMERA_REST.lookZ);
            composite.fitBackdropPlane(camera);
        }
        paintFallbackFrame(sync.time, sync.videoDuration);
    }
    updateModeChip();
}

function setupKeyframeAnimation() {
    if (!torus || !composite) return;
    const dur = Math.max(
        sync.effectiveDuration(),
        sync.videoDuration || 10,
        sync.clipOut || 0,
    );
    if (dur <= 0) return;

    const clip = new THREE.AnimationClip('torus-demo', dur, [
        new THREE.NumberKeyframeTrack('.position[y]', [0, dur * 0.5, dur], [-0.45, 0.2, -0.45]),
        new THREE.NumberKeyframeTrack('.rotation[y]', [0, dur], [0, Math.PI * 2]),
    ]);
    const mixer = composite.addAnimationMixer(torus);
    const action = mixer.clipAction(clip);
    action.play();
    action.paused = true;
}

function importSyncJson(raw) {
    const data = typeof raw === 'string' ? JSON.parse(raw) : raw;
    sync = ScrollVideoSync.fromJSON(data);
    if (data.markers && data.markers.length) {
        sync.markers = data.markers;
    } else {
        sync.markers = readScrollMarkers(document, markerReadOpts());
    }
    applySyncToUi();
    refreshMarkerPanel(true);
    updateModeChip();
    rebuildMarkerPins();
    statusEl.textContent = 'imported scroll mapping JSON';
}

async function loadUserVideo(file) {
    if (!file) return;
    try {
        await loadUserVideoFile(videoEl, file);
        sync.videoDuration = videoEl.duration || sync.videoDuration;
        ui.clipOut.max = String(Math.max(12, sync.videoDuration + 2));
        ui.clipOut.value = String(sync.videoDuration.toFixed(2));
        exportDirty = true;
        syncFromUi();
        updateLiveUi();
        refreshExport();
        markerPanelDirty = true;
        statusEl.textContent = `loaded ${file.name}`;
    } catch (e) {
        setFallbackVideo(true);
        statusEl.textContent = `video load failed — procedural fallback (${file.name})`;
        console.warn(e);
    }
}

function bindScrollDriving() {
    if (!copyScrollEl) return;
    copyScrollEl.addEventListener('scroll', function () {
        markerPanelDirty = true;
    }, { passive: true });

    if (stageWrapEl) {
        stageWrapEl.addEventListener('wheel', function (e) {
            if (!copyScrollEl) return;
            copyScrollEl.scrollTop += e.deltaY;
            e.preventDefault();
        }, { passive: false });
    }
}

function bindControls() {
    const onChange = () => {
        syncFromUi();
        updateLiveUi();
        exportDirty = true;
        refreshExport();
        markerPanelDirty = true;
        updateModeChip();
    };
    for (const el of [
        ui.mode, ui.easing, ui.scrollStart, ui.scrollEnd, ui.pxPerSec,
        ui.velocityGain, ui.smoothing, ui.clipIn, ui.clipOut,
    ]) {
        if (el) el.addEventListener('input', onChange);
    }
    if (ui.displayMode) {
        ui.displayMode.addEventListener('change', () => {
            applyDisplayMode();
            onChange();
        });
    }

    if (ui.fitMarkers) {
        ui.fitMarkers.addEventListener('click', () => {
            const markers = readScrollMarkers(document, markerReadOpts());
            autoFitScrollRange(sync, markers);
            applySyncToUi();
            refreshMarkerPanel(true);
            rebuildMarkerPins();
        });
    }

    if (ui.resetRange) {
        ui.resetRange.addEventListener('click', () => {
            ui.scrollStart.value = '0';
            ui.scrollEnd.value = '4000';
            ui.mode.value = 'progress';
            if (ui.easing) ui.easing.checked = false;
            ui.pxPerSec.value = '900';
            ui.velocityGain.value = '1';
            ui.smoothing.value = '0.18';
            ui.clipIn.value = '0';
            ui.clipOut.value = String(Math.max(10, videoEl.duration || 10).toFixed(2));
            sync.easing = false;
            onChange();
            refreshMarkerPanel(true);
        });
    }

    if (ui.copyJson) {
        ui.copyJson.addEventListener('click', async () => {
            exportDirty = true;
            refreshExport();
            try {
                await navigator.clipboard.writeText(exportEl.value);
                statusEl.textContent = 'JSON copied';
            } catch (_err) {
                exportEl.select();
                document.execCommand('copy');
                statusEl.textContent = 'JSON copied';
            }
        });
    }

    if (ui.importJson) {
        ui.importJson.addEventListener('click', () => {
            const raw = window.prompt('Paste scroll-video JSON:', exportEl.value);
            if (raw == null || !raw.trim()) return;
            try {
                importSyncJson(raw.trim());
            } catch (e) {
                statusEl.textContent = 'invalid JSON';
                console.error(e);
            }
        });
    }

    if (ui.videoFile) {
        ui.videoFile.addEventListener('change', () => {
            const file = ui.videoFile.files && ui.videoFile.files[0];
            void loadUserVideo(file);
        });
    }
}

function overlayMaterial(color, emissive, opts) {
    opts = opts || {};
    return new THREE.MeshStandardMaterial({
        color: color,
        emissive: emissive,
        emissiveIntensity: opts.emissiveIntensity != null ? opts.emissiveIntensity : 0.45,
        metalness: opts.metalness != null ? opts.metalness : 0.45,
        roughness: opts.roughness != null ? opts.roughness : 0.28,
        transparent: opts.transparent || false,
        opacity: opts.opacity != null ? opts.opacity : 1,
        depthWrite: opts.depthWrite != null ? opts.depthWrite : true,
    });
}

/**
 * 3D annotations composited in front of the video.
 */
function buildOverlay(targetScene) {
    cornerOrbs = [];

    const frame = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(15.6, 8.75, 0.06)),
        new THREE.LineBasicMaterial({ color: 0x6fb3ff, transparent: true, opacity: 0.92 }),
    );
    frame.renderOrder = 10;
    targetScene.add(frame);

    const cornerPositions = [
        [-7.1, 3.75, 0.55],
        [7.1, 3.75, 0.55],
        [-7.1, -3.75, 0.55],
        [7.1, -3.75, 0.55],
    ];
    for (let i = 0; i < cornerPositions.length; i++) {
        const orb = new THREE.Mesh(
            new THREE.SphereGeometry(0.24, 16, 12),
            overlayMaterial(0x7ddc9a, 0x1f5c38, { emissiveIntensity: 0.65 }),
        );
        orb.position.set(cornerPositions[i][0], cornerPositions[i][1], cornerPositions[i][2]);
        orb.renderOrder = 12;
        targetScene.add(orb);
        cornerOrbs.push(orb);
    }

    ring = new THREE.Mesh(
        new THREE.TorusGeometry(1.45, 0.07, 12, 64),
        overlayMaterial(0x6fb3ff, 0x143350, {
            emissiveIntensity: 0.55,
            metalness: 0.72,
            roughness: 0.18,
            transparent: true,
            opacity: 0.94,
            depthWrite: false,
        }),
    );
    ring.position.set(0, 0.35, 1.1);
    ring.renderOrder = 11;
    targetScene.add(ring);

    torus = new THREE.Mesh(
        new THREE.TorusKnotGeometry(0.7, 0.2, 64, 16),
        overlayMaterial(0xffb454, 0x4a2808, { metalness: 0.55, roughness: 0.25 }),
    );
    torus.position.set(2.4, -0.45, 1.65);
    torus.renderOrder = 11;
    targetScene.add(torus);

    const progressTrack = new THREE.Mesh(
        new THREE.BoxGeometry(12.4, 0.1, 0.06),
        overlayMaterial(0x1a2030, 0x080a10, { metalness: 0.15, roughness: 0.85 }),
    );
    progressTrack.position.set(0, -3.55, 0.95);
    progressTrack.renderOrder = 10;
    targetScene.add(progressTrack);

    progressFill = new THREE.Mesh(
        new THREE.BoxGeometry(0.35, 0.16, 0.1),
        overlayMaterial(0x7ddc9a, 0x2d6b45, { emissiveIntensity: 0.6 }),
    );
    progressFill.position.set(-6.0, -3.55, 1.1);
    progressFill.renderOrder = 12;
    targetScene.add(progressFill);

    const badge = new THREE.Mesh(
        new THREE.BoxGeometry(1.6, 0.55, 0.12),
        overlayMaterial(0x2f6fd0, 0x0a1a30, {
            transparent: true,
            opacity: 0.88,
            depthWrite: false,
        }),
    );
    badge.position.set(-5.2, 2.6, 1.35);
    badge.renderOrder = 11;
    targetScene.add(badge);
}

function rebuildMarkerPins() {
    const scene = composite ? composite.scene : null;
    if (!scene) return;
    for (let i = 0; i < markerPins.length; i++) {
        scene.remove(markerPins[i]);
    }
    markerPins = [];
    const d = sync.effectiveDuration();
    if (d <= 0) return;
    for (let i = 0; i < sync.markers.length; i++) {
        const m = sync.markers[i];
        const u = (m.time - sync.clipIn) / d;
        const pin = new THREE.Mesh(
            new THREE.BoxGeometry(0.08, 0.28, 0.08),
            overlayMaterial(0x3a86e0, 0x0a1a30, {
                emissiveIntensity: 0.35,
                transparent: true,
                opacity: 0.85,
            }),
        );
        pin.position.set(-6.0 + u * 12.0, -3.55, 1.05);
        pin.renderOrder = 11;
        scene.add(pin);
        markerPins.push(pin);
    }
}

async function setupScene() {
    await initThreers({ module_or_path: '../pkg/threers_bg.wasm' });
    statusEl.textContent = 'wasm ready — loading clip…';

    try {
        await loadStageVideo(videoEl);
    } catch (e) {
        console.warn('bundled clip failed:', e);
        if (!isSafariBrowser()) {
            try {
                const clipUrl = await recordTestClip(10);
                await loadTestClip(clipUrl);
            } catch (recErr) {
                console.warn('MediaRecorder clip failed:', recErr);
                setFallbackVideo(true);
                statusEl.textContent = 'procedural test frames (no clip)';
            }
        } else {
            setFallbackVideo(true);
            statusEl.textContent = 'procedural test frames (clip load failed)';
        }
    }

    sync.videoDuration = videoEl.duration || 10;
    ui.clipOut.max = String(Math.max(12, sync.videoDuration + 2));
    ui.clipOut.value = String(sync.videoDuration.toFixed(2));
    syncFromUi();
    updateLiveUi();
    exportDirty = true;
    refreshExport();

    composite = new VideoSceneComposite({
        canvas,
        video: videoEl,
        displayMode: DisplayMode.CANVAS_3D,
    });
    await composite.initRenderer();

    buildOverlay(composite.scene);
    setupKeyframeAnimation();

    composite.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(2.5, 4, 6);
    composite.scene.add(key);
    const rim = new THREE.DirectionalLight(0x8ec8ff, 0.85);
    rim.position.set(-4, 1, 3);
    composite.scene.add(rim);

    const dpr = Math.min(devicePixelRatio || 1, isSafariBrowser() ? 1.5 : 2);
    const w = Math.floor(canvas.clientWidth * dpr) || 1280;
    const h = Math.floor(canvas.clientHeight * dpr) || 720;
    camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
    camera.position.set(0, 0, 6.8);
    camera.lookAt(0, 0, 0);
    composite.setCamera(camera);

    applyDisplayMode();
    paintFallbackFrame(0, sync.videoDuration);

    statusEl.textContent = 'scroll — video + 3D in one canvas';
}

function resize() {
    if (composite) {
        composite.resize();
        if (camera && composite.usesCanvas3d()) composite.fitBackdropPlane(camera);
    }
}

function updateOverlayFromTime(t) {
    const u = sync.effectiveDuration() > 0
        ? (t - sync.clipIn) / sync.effectiveDuration()
        : 0;

    if (ring) {
        ring.rotation.x = t * 0.85;
        ring.rotation.y = t * 1.15;
        ring.scale.setScalar(0.9 + u * 0.4);
        ring.material.color.setHSL(0.58 - u * 0.2, 0.7, 0.62);
    }
    if (torus) {
        torus.rotation.x = t * 1.5;
        torus.position.z = 1.65 + u * 0.35;
    }
    if (progressFill) {
        progressFill.position.x = -6.0 + u * 12.0;
        progressFill.scale.x = 0.6 + u * 2.8;
    }
    for (let i = 0; i < cornerOrbs.length; i++) {
        const orb = cornerOrbs[i];
        const pulse = 1 + Math.sin(t * 2.2 + i * 1.2) * 0.12;
        orb.scale.setScalar(pulse);
        orb.position.z = 0.55 + Math.sin(t * 1.8 + i) * 0.12;
    }
    if (camera) {
        if (composite && composite.usesCanvas3d()) {
            camera.position.set(CAMERA_REST.x, CAMERA_REST.y, CAMERA_REST.z);
            camera.lookAt(CAMERA_REST.lookX, CAMERA_REST.lookY, CAMERA_REST.lookZ);
        } else {
            camera.position.z = CAMERA_REST.z - u * 0.9;
            camera.position.x = Math.sin(t * 0.4) * 0.35;
            camera.lookAt(0, 0.05, 0.5);
        }
    }
}

async function frame(now) {
    requestAnimationFrame(frame);

    const scrollY = getScrollY();
    sync.update({ scrollY, now });

    sync.applyToVideo(videoEl);

    const t = sync.time;
    const duration = sync.videoDuration || 10;

    if (composite) {
        if (composite.usesHtmlLayer()) {
            if (videoEl.readyState < 2 || !videoEl.videoWidth) {
                setFallbackVideo(true);
                paintFallbackFrame(t, duration);
            } else {
                setFallbackVideo(false);
            }
        } else {
            setFallbackVideo(false);
            const paintProc = function (ctx, w, h, time) {
                paintClipFrame(ctx, w, h, time, duration);
            };
            const needsFallback = videoEl.readyState < 2 || !videoEl.videoWidth;
            composite.updateVideoFrame(t, needsFallback ? paintProc : undefined);
        }
        composite.syncAnimationsToTime(t);
    }

    updateOverlayFromTime(t);

    updatePreviewCanvas();

    ui.liveScroll.textContent = String(scrollY | 0);
    ui.liveSpeed.textContent = String(sync.scrollSpeed.toFixed(0));
    ui.liveTime.textContent = t.toFixed(2);
    ui.liveRate.textContent = sync.playbackRate.toFixed(2);
    if (ui.timeBadge) ui.timeBadge.textContent = t.toFixed(1);
    if (ui.timelineFill) {
        ui.timelineFill.style.width = `${Math.min(Math.max(sync.progress, 0), 1) * 100}%`;
    }

    updateActiveMarker(scrollY);
    updateScrollRail(scrollY);
    if (markerPanelDirty || now - lastMarkerPanelAt > 250) {
        refreshMarkerPanel(true);
    }

    if (composite) {
        composite.render();
    }
}

async function main() {
    try {
        bindControls();
        bindScrollDriving();
        attachMarkerResizeObserver(sync, copyScrollEl, document, markerReadOpts());
        addEventListener('resize', resize);
        await setupScene();
        sync.markers = readScrollMarkers(document, markerReadOpts());
        if (sync.markers.length >= 2) {
            autoFitScrollRange(sync, sync.markers);
            applySyncToUi();
        } else {
            syncFromUi();
            exportDirty = true;
            refreshExport();
        }
        rebuildMarkerPins();
        refreshMarkerPanel(true);
        updateActiveMarker(getScrollY());
        updateScrollRail(getScrollY());
        resize();
        requestAnimationFrame(frame);
    } catch (e) {
        if (errEl) errEl.textContent = (e && e.stack) || String(e);
        if (statusEl) statusEl.textContent = 'failed';
        console.error(e);
    }
}

void main();
