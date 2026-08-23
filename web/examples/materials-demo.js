// Live gallery of every `materials::presets` entry plus the extended
// MeshPhysicalMaterial layers, rendered in the browser by threers/wgpu.
//
// Everything is procedural — the environment cubemap and the foil crinkle
// normal map are both generated here in JS, so the page needs no asset
// downloads.

// Relative paths so the page works under any static server root — served
// from the repo root, from web/, or from web/examples/ all resolve.
import THREE, { initThreers } from '../threejs-shim.js';

const errEl = document.getElementById('err');
const statusEl = document.getElementById('status');

// ---------------------------------------------------------------------------
// Procedural noise — mirrors examples/spacecraft_common.inc so the browser
// gallery and the native examples show the same materials.
// ---------------------------------------------------------------------------

function hash2(x, y) {
    let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263)) | 0;
    h = Math.imul(h ^ (h >> 13), 1274126177) | 0;
    return ((h ^ (h >> 16)) >>> 0) / 4294967295;
}

const smooth = (t) => t * t * (3 - 2 * t);

function valueNoise(x, y) {
    const xi = Math.floor(x), yi = Math.floor(y);
    const tx = smooth(x - xi), ty = smooth(y - yi);
    const a = hash2(xi, yi), b = hash2(xi + 1, yi);
    const c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
    const top = a + (b - a) * tx;
    const bot = c + (d - c) * tx;
    return top + (bot - top) * ty;
}

// Ridged noise: creases rather than blobs, which is what reads as *crumpled*.
function ridgedFbm(x, y, octaves) {
    let sum = 0, amp = 0.5, freq = 1, norm = 0;
    for (let i = 0; i < octaves; i++) {
        const n = valueNoise(x * freq, y * freq) * 2 - 1;
        sum += (1 - Math.abs(n)) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.07;
    }
    return sum / Math.max(norm, 1e-6);
}

/** Tangent-space normal map of crumpled foil — the thing that makes MLI read as foil. */
function crinkleNormalMap(size, scale, strength) {
    const h = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            h[y * size + x] = ridgedFbm((x / size) * scale, (y / size) * scale, 5);
        }
    }
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const xm = (x + size - 1) % size, xp = (x + 1) % size;
            const ym = (y + size - 1) % size, yp = (y + 1) % size;
            const dx = (h[y * size + xp] - h[y * size + xm]) * strength;
            const dy = (h[yp * size + x] - h[ym * size + x]) * strength;
            const len = Math.sqrt(dx * dx + dy * dy + 1);
            const i = (y * size + x) * 4;
            data[i] = ((-dx / len) * 0.5 + 0.5) * 255;
            data[i + 1] = ((-dy / len) * 0.5 + 0.5) * 255;
            data[i + 2] = ((1 / len) * 0.5 + 0.5) * 255;
            data[i + 3] = 255;
        }
    }
    const tex = new THREE.DataTexture(data, size, size);
    tex.magFilter = 1006; // LinearFilter — a nearest-sampled normal map facets badly
    tex.minFilter = 1006;
    return tex;
}

/** Solar-cell grid: dark cells with busbar / interconnect lines. */
function solarCellMap(size, cells) {
    const data = new Uint8Array(size * size * 4);
    const cellPx = Math.max(size / cells, 2);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const cx = (x % cellPx) / cellPx;
            const cy = (y % cellPx) / cellPx;
            const gap = cx < 0.03 || cy < 0.03;
            const busbar = Math.abs(cx - 0.25) < 0.012
                || Math.abs(cx - 0.5) < 0.012
                || Math.abs(cx - 0.75) < 0.012;
            let r, g, b;
            if (gap) { r = 0.42; g = 0.44; b = 0.48; }
            else if (busbar) { r = 0.62; g = 0.63; b = 0.66; }
            else {
                const v = 0.9 + 0.1 * hash2(Math.floor(x / cellPx), Math.floor(y / cellPx));
                r = 0.16 * v; g = 0.20 * v; b = 0.34 * v;
            }
            const i = (y * size + x) * 4;
            data[i] = r * 255; data[i + 1] = g * 255; data[i + 2] = b * 255; data[i + 3] = 255;
        }
    }
    const tex = new THREE.DataTexture(data, size, size);
    tex.magFilter = 1006; tex.minFilter = 1006;
    return tex;
}

/** Checker card used behind transmissive tiles so refraction is legible. */
function checkerMap(size, squares) {
    const data = new Uint8Array(size * size * 4);
    const cell = size / squares;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const on = ((Math.floor(x / cell) + Math.floor(y / cell)) & 1) === 0;
            const i = (y * size + x) * 4;
            const v = on ? 232 : 44;
            const w = on ? 236 : 58;
            data[i] = v; data[i + 1] = v; data[i + 2] = w; data[i + 3] = 255;
        }
    }
    const tex = new THREE.DataTexture(data, size, size);
    tex.magFilter = 1006; tex.minFilter = 1006;
    return tex;
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

const SUN_DIR = (() => {
    const d = [0.55, 0.42, 0.72];
    const l = Math.hypot(...d);
    return d.map((c) => c / l);
})();

function cubeDir(face, u, v) {
    switch (face) {
        case 0: return [1, -v, -u];
        case 1: return [-1, -v, u];
        case 2: return [u, 1, v];
        case 3: return [u, -1, -v];
        case 4: return [u, -v, 1];
        default: return [-u, -v, -1];
    }
}

/**
 * Low-Earth-orbit environment: dark sky, stars, a *tight* sun disc, and a blue
 * Earth glow from below.
 *
 * This is not decoration — metals have no diffuse term, so without an
 * environment every metalness=1 sphere on this page would render near-black.
 * The sun disc is deliberately small: an 8-bit cubemap caps at 1.0, so a large
 * bright disc gets smeared across tens of degrees by the PMREM prefilter and
 * every metal ends up reflecting the same grey wash. The DirectionalLight
 * supplies the highlight instead.
 */
function spaceEnvironment(size) {
    const faces = [];
    for (let face = 0; face < 6; face++) {
        const data = new Uint8Array(size * size * 4);
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const u = ((x + 0.5) / size) * 2 - 1;
                const v = ((y + 0.5) / size) * 2 - 1;
                // X is flipped on purpose: the PMREM atlas builder samples the
                // source cube at (-x, y, z), so a feature authored at cube
                // direction d is seen at world direction (-d.x, d.y, d.z).
                const d0 = cubeDir(face, u, v);
                const l = Math.hypot(...d0);
                const d = [-d0[0] / l, d0[1] / l, d0[2] / l];

                let r = 0.006, g = 0.008, b = 0.014;

                const earth = Math.pow(Math.max(-d[1], 0), 2.0);
                r += earth * 0.055; g += earth * 0.080; b += earth * 0.115;
                const limb = Math.max(1 - Math.abs(d[1] + 0.12) * 4.5, 0);
                r += limb * 0.048; g += limb * 0.070; b += limb * 0.100;

                const cosSun = d[0] * SUN_DIR[0] + d[1] * SUN_DIR[1] + d[2] * SUN_DIR[2];
                if (cosSun > 0.99985) { r = 1; g = 1; b = 0.98; }
                else {
                    const bloom = Math.pow(Math.max((cosSun - 0.999) / 0.001, 0), 2);
                    r += bloom * 0.55; g += bloom * 0.52; b += bloom * 0.45;
                }

                const s = hash2(Math.imul(Math.trunc(d[0] * 400), 31) + Math.trunc(d[2] * 400),
                                Math.trunc(d[1] * 400));
                if (s > 0.9985) {
                    const mag = (s - 0.9985) / 0.0015;
                    r += mag * 0.8; g += mag * 0.8; b += mag * 0.9;
                }

                const i = (y * size + x) * 4;
                data[i] = Math.pow(Math.min(r, 1), 1 / 2.2) * 255;
                data[i + 1] = Math.pow(Math.min(g, 1), 1 / 2.2) * 255;
                data[i + 2] = Math.pow(Math.min(b, 1), 1 / 2.2) * 255;
                data[i + 3] = 255;
            }
        }
        faces.push(Array.from(data));
    }
    return new THREE.CubeTexture(size, ...faces);
}

// ---------------------------------------------------------------------------
// The gallery
// ---------------------------------------------------------------------------

const PI = Math.PI;

/** One entry per tile. `build` returns a MeshPhysicalMaterial. */
function tileSpecs(crinkle, cells) {
    const foil = (name) => THREE.MeshPhysicalMaterial.preset(name, 0, {
        normalMap: crinkle,
        normalScale: { x: 1.4, y: 1.4 },
    });

    return [
        // --- the presets -----------------------------------------------
        { label: 'gold foil', note: 'MLI blanket · F0 (1.00, 0.77, 0.34)', build: () => foil('gold_foil') },
        { label: 'silver foil', note: 'aluminized kapton', build: () => foil('silver_foil') },
        { label: 'aluminium', note: 'bare structure', build: () => THREE.MeshPhysicalMaterial.preset('aluminum') },
        { label: 'brushed aluminium', note: 'anisotropy 0.8', build: () => THREE.MeshPhysicalMaterial.preset('brushed_aluminum', 0) },
        { label: 'titanium', note: 'F0 (0.54, 0.50, 0.45)', build: () => THREE.MeshPhysicalMaterial.preset('titanium') },
        { label: 'solar cell', note: 'clearcoat cover glass', build: () => THREE.MeshPhysicalMaterial.preset('solar_cell', 0, { map: cells }) },
        { label: 'white paint', note: 'thermal radiator', build: () => THREE.MeshPhysicalMaterial.preset('white_thermal_paint') },
        { label: 'black kapton', note: 'sheen 0.35', build: () => THREE.MeshPhysicalMaterial.preset('black_kapton') },

        // --- iridescence: hue is set by film thickness, not by color ----
        { label: 'iridescence 180nm', note: 'thin film', build: () => THREE.MeshPhysicalMaterial.preset('anodized_titanium', 180) },
        { label: 'iridescence 300nm', note: 'thin film', build: () => THREE.MeshPhysicalMaterial.preset('anodized_titanium', 300) },
        { label: 'iridescence 420nm', note: 'thin film', build: () => THREE.MeshPhysicalMaterial.preset('anodized_titanium', 420) },
        { label: 'iridescence 560nm', note: 'thin film', build: () => THREE.MeshPhysicalMaterial.preset('anodized_titanium', 560) },

        // --- anisotropy rotation: same material, rotated streak ---------
        { label: 'anisotropy 0°', note: 'streak along U', build: () => THREE.MeshPhysicalMaterial.preset('brushed_aluminum', 0) },
        { label: 'anisotropy 45°', note: 'anisotropyRotation', build: () => THREE.MeshPhysicalMaterial.preset('brushed_aluminum', PI / 4) },
        { label: 'anisotropy 90°', note: 'anisotropyRotation', build: () => THREE.MeshPhysicalMaterial.preset('brushed_aluminum', PI / 2) },
        { label: 'anisotropy 135°', note: 'anisotropyRotation', build: () => THREE.MeshPhysicalMaterial.preset('brushed_aluminum', (3 * PI) / 4) },

        // --- glass + the raw three.js option API -----------------------
        {
            label: 'optical glass', note: 'refract · IOR 1.52', backdrop: true,
            build: () => THREE.MeshPhysicalMaterial.preset('optical_glass'),
        },
        {
            label: 'tinted glass', note: 'Beer-Lambert absorption', backdrop: true,
            build: () => new THREE.MeshPhysicalMaterial({
                color: 0xffffff, roughness: 0.03, metalness: 0.0,
                transmission: 1.0, ior: 1.5, thickness: 0.6, dispersion: 0.2,
                attenuationColor: 0x66ddaa, attenuationDistance: 0.8,
                transparencyMode: 'refract',
            }),
        },
        {
            label: 'clearcoat over red', note: 'clearcoat 1.0',
            build: () => new THREE.MeshPhysicalMaterial({
                color: 0xaa1620, roughness: 0.45, metalness: 0.0,
                clearcoat: 1.0, clearcoatRoughness: 0.03,
            }),
        },
        {
            label: 'velvet sheen', note: 'sheen 1.0 · Charlie lobe',
            build: () => new THREE.MeshPhysicalMaterial({
                color: 0x2e1c5c, roughness: 0.9, metalness: 0.0,
                sheen: 1.0, sheenColor: 0x9a6cff, sheenRoughness: 0.35,
            }),
        },
    ];
}

// ---------------------------------------------------------------------------

// Wrapped in an async IIFE rather than using top-level `await`, which is
// ES2022 module syntax and needs Safari 15. A syntax error is not recoverable —
// the module never loads and the page reports one error a long way from the
// cause — so the cost of the wrapper is worth the floor it buys.
(async () => {
    try {
        statusEl && (statusEl.textContent = 'loading wasm…');
        await initThreers({ module_or_path: new URL('../pkg/threers_bg.wasm', import.meta.url) });

        const canvas = document.getElementById('canvas');
        const grid = document.getElementById('grid');

        statusEl && (statusEl.textContent = 'building environment…');
        const crinkle = crinkleNormalMap(256, 9.0, 2.2);
        const cells = solarCellMap(256, 6);
        const checker = checkerMap(256, 8);

        const renderer = await THREE.WebGLRenderer.create(canvas);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x05060a);

        // Prefilter the cubemap into a roughness-indexed mip chain. Without this
        // step every metal samples a mirror-sharp environment whatever its
        // roughness, and the roughness slider does nothing.
        const pmrem = new THREE.PMREMGenerator(renderer);
        scene.environment = pmrem.fromCubemap(spaceEnvironment(128)).texture;

        scene.add(new THREE.AmbientLight(0x223044, 0.06));
        const sun = new THREE.DirectionalLight(0xfff6e8, 2.7);
        sun.position.set(SUN_DIR[0] * 12, SUN_DIR[1] * 12, SUN_DIR[2] * 12);
        scene.add(sun);
        const bounce = new THREE.DirectionalLight(0x4a7fc8, 0.35);
        bounce.position.set(-3.6, -12, 2.4);
        scene.add(bounce);

        statusEl && (statusEl.textContent = 'building materials…');
        const specs = tileSpecs(crinkle, cells);

        const COLS = 5;
        const rows = Math.ceil(specs.length / COLS);
        const SPACING = 2.4;
        const sphere = new THREE.SphereGeometry(1, 64, 32);

        specs.forEach((spec, i) => {
            const mesh = new THREE.Mesh(sphere, spec.build());
            const col = i % COLS;
            const row = Math.floor(i / COLS);
            mesh.position.set(
                (col - (COLS - 1) / 2) * SPACING,
                ((rows - 1) / 2 - row) * SPACING,
                0,
            );
            scene.add(mesh);

            // Screen-space refraction can only show what is actually behind the
            // surface. Against an empty black background transmissive materials
            // read as flat grey blobs, so give those tiles a lit card to refract.
            if (spec.backdrop) {
                const card = new THREE.Mesh(
                    new THREE.PlaneGeometry(2.2, 2.2),
                    new THREE.MeshBasicMaterial({ map: checker }),
                );
                card.position.set(mesh.position.x, mesh.position.y, -1.6);
                scene.add(card);
            }

            // Caption cell, laid out by CSS to line up under its sphere.
            const cell = document.createElement('figure');
            cell.className = 'tile';
            cell.innerHTML = `<figcaption><b>${spec.label}</b><span>${spec.note}</span></figcaption>`;
            grid.appendChild(cell);
        });
        grid.style.setProperty('--cols', COLS);
        grid.style.setProperty('--rows', rows);

        // Match the canvas aspect to the sphere grid so the camera fits both axes
        // identically (no letterboxing), then inset the caption grid by the same
        // margin the camera leaves around the spheres. Without this the labels
        // drift relative to the spheres they name.
        const FIT_MARGIN = 1.06;
        canvas.parentElement.style.aspectRatio = `${COLS} / ${rows}`;
        grid.style.inset = `${(((1 - 1 / FIT_MARGIN) / 2) * 100).toFixed(2)}%`;

        // Frame the whole grid: fit the wider of the two axes.
        // A narrow FOV keeps the projection close to orthographic, so spheres at the
        // edges of the grid stay round instead of smearing into ellipses.
        const camera = new THREE.PerspectiveCamera(24, 1, 0.1, 400);
        const halfW = (COLS * SPACING) / 2;
        const halfH = (rows * SPACING) / 2;

        function resize() {
            const dpr = Math.min(devicePixelRatio || 1, 2);
            if (!canvas.parentElement) return;
            const rect = canvas.parentElement.getBoundingClientRect();
            const w = Math.max(2, Math.floor(rect.width * dpr));
            const h = Math.max(2, Math.floor(rect.height * dpr));
            canvas.width = w; canvas.height = h;
            renderer.setSize(w, h);
            camera.aspect = w / h;
            // Distance that fits both grid extents inside the frustum.
            const vFov = (camera.fov * Math.PI) / 180;
            const distH = halfH / Math.tan(vFov / 2);
            const distW = halfW / (Math.tan(vFov / 2) * camera.aspect);
            camera.position.set(0, 0, Math.max(distH, distW) * FIT_MARGIN);
            camera.lookAt(0, 0, 0);
            camera.updateProjectionMatrix();
        }
        resize();
        addEventListener('resize', resize);

        statusEl && (statusEl.textContent = `${specs.length} materials · live wgpu`);

        // Render + read back in one synchronous block. A WebGPU canvas' contents are
        // not guaranteed to survive past presentation, so a headless screenshot can
        // come back blank even though the frame drew fine; grabbing here instead
        // captures the real pixels.
        window.__THREERS_GRAB__ = () => {
            renderer.render(scene, camera);
            return canvas.toDataURL('image/png');
        };

        let frames = 0;
        function frame() {
            renderer.render(scene, camera);
            frames++;
            // Signal readiness for the screenshot harness once a few frames land.
            if (frames === 3) window.__THREERS_MATERIALS_READY__ = { ok: true, count: specs.length };
            requestAnimationFrame(frame);
        }
        frame();
    } catch (e) {
        const msg = (e && (e.stack || e.message)) || String(e);
        if (errEl) { errEl.style.display = 'block'; errEl.textContent = msg; }
        window.__THREERS_MATERIALS_READY__ = { ok: false, error: msg };
        console.error(e);
    }
})();
