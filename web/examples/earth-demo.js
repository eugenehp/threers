// Earth lit by the sun, with a lens flare — the browser port of
// `examples/earth_sun_flare.rs`.
//
// Same construction as the native example: five equirectangular maps derived
// from one shared elevation field, noise sampled in 3D on the sphere so there
// is neither a date-line seam nor pole smearing, and the flare composited in
// screen space. The maps are generated in a Web Worker so the page stays
// responsive while a 4K set is built.

import THREE, { initThreers } from '../threejs-shim.js';

const statusEl = document.getElementById('status');
const errEl = document.getElementById('err');
const canvas = document.getElementById('c');
const overlay = document.getElementById('flare');
const octx = overlay.getContext('2d');

const say = (t) => { statusEl.textContent = t; };

// ---------------------------------------------------------------------------
// Map generation, off the main thread
// ---------------------------------------------------------------------------

/** Ask the worker for a map set at `size × size/2`. */
function generateMaps(size, onProgress) {
    return new Promise((resolve, reject) => {
        const worker = new Worker('./earth-worker.js', { type: 'module' });
        worker.onmessage = (e) => {
            if (e.data.progress !== undefined) {
                onProgress?.(e.data.progress, e.data.stage);
                return;
            }
            worker.terminate();
            resolve(e.data);
        };
        worker.onerror = (e) => {
            worker.terminate();
            reject(new Error(e.message || 'map worker failed'));
        };
        worker.postMessage({ size });
    });
}

// ---------------------------------------------------------------------------
// Real imagery, when it has been fetched
// ---------------------------------------------------------------------------
//
// `scripts/fetch-earth-textures.sh` puts NASA's public-domain Blue Marble,
// Black Marble and MODIS cloud products in `web/assets/earth/`. When they are
// there we use them; when they are not the procedural planet stands in, so the
// page works from a clean checkout.

const ASSETS = '../assets/earth';

/**
 * Arctic imagery, decoded at most once and kept: the whole-globe day map and
 * every 500 m north tile need the same splice, and the baked path skips the
 * day map entirely but still streams tiles.
 *
 * Small on purpose. It is only ever read above 72°N, where its own sea-ice mask
 * is blocky at about a quarter of a degree, so a 2048-wide copy costs 8 MB and
 * throws away nothing.
 */
let iceCap = null;
let iceCapTried = false;

async function ensureIceCap() {
    if (!iceCapTried) {
        iceCapTried = true;
        try {
            iceCap = await decode(`${ASSETS}/ice_cap.png`, 2048, 1024);
        } catch (e) {
            console.warn('no Arctic imagery, the north pole will be dark:', e.message);
        }
    }
    return iceCap;
}

/**
 * Flip an RGBA buffer top-to-bottom, in place.
 *
 * `getImageData` hands back the top row first, but a `DataTexture` is uploaded
 * verbatim — `flipY` is `false` for data textures, in three.js and here — and
 * UV space puts `v = 0` at the bottom. On a `SphereGeometry` the north pole is
 * at `uv.y = 1`, so an unflipped equirectangular map puts Antarctica at the
 * north pole and renders the whole planet upside down.
 */
function flipRows(data, width, height) {
    const stride = width * 4;
    const row = new Uint8Array(stride);
    for (let y = 0; y < (height >> 1); y++) {
        const top = y * stride;
        const bottom = (height - 1 - y) * stride;
        row.set(data.subarray(top, top + stride));
        data.copyWithin(top, bottom, bottom + stride);
        data.set(row, bottom);
    }
    return data;
}

/** The mean colour of every row of a map, as a flat RGBA array. */
function zonalMeans(m) {
    const { width: w, height: h, data } = m;
    const means = new Float32Array(h * 4);
    for (let y = 0; y < h; y++) {
        const base = y * w * 4;
        let r = 0, g = 0, b = 0, a = 0;
        for (let x = 0; x < w; x++) {
            const o = base + x * 4;
            r += data[o]; g += data[o + 1]; b += data[o + 2]; a += data[o + 3];
        }
        const o = y * 4;
        means[o] = r / w; means[o + 1] = g / w; means[o + 2] = b / w; means[o + 3] = a / w;
    }
    return means;
}

/**
 * Flatten the last few rows to their zonal mean.
 *
 * A UV sphere's pole is a fan of triangles whose apex vertices all sit at the
 * same point but carry different `u`. Anything reading the map *per vertex* —
 * displacement above all — gets a different answer at each of them, and the
 * pole tears open into a star. Making the top and bottom rows constant along
 * longitude closes it, and a constant is the one thing that survives the pole's
 * convergence unchanged.
 *
 * This used to also blur along longitude by up to 24 texels, on the theory that
 * the map is oversampled by `1/cos(latitude)` near the pole and the surplus
 * gets stretched back out as a fan of streaks. It is not: `tests/polar_sampling.rs`
 * renders a latitude-only map straight down the pole and finds it rotationally
 * symmetric to 0.05/255. What the blur did do was smear features *around* the
 * pole — a vortex, plainly visible as a spiral in the cloud deck over the
 * Arctic — and flattening from 80° washed ten degrees of real imagery into its
 * own average, which is what made the ice cap a featureless disc.
 *
 * The mean has to come from the whole latitude circle, which is why it is
 * passed in: a 90°-wide tile averaged on its own would converge to a different
 * colour from its three neighbours, and all four meet at the pole.
 *
 * `latTop`/`latBottom` are the latitudes of this map's first and last rows, so
 * the same function serves whole globes and single tiles.
 */
function easePoles(m, means, meanRows, latTop = 90, latBottom = -90) {
    const { width: w, height: h, data } = m;
    const smooth = (a, b, x) => {
        const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
        return t * t * (3 - 2 * t);
    };
    for (let y = 0; y < h; y++) {
        const lat = latTop + (latBottom - latTop) * ((y + 0.5) / h);
        const cos = Math.max(Math.abs(Math.cos(lat * Math.PI / 180)), 1e-6);
        // 30x oversampling is 88.1 degrees of latitude, 90x is 89.4. Bilinear
        // sampling reaches a row either side of the pole row, so the ramp
        // covers a few rows rather than snapping on at the last one.
        const wgt = smooth(30, 90, 1 / cos);
        if (wgt <= 0.001) continue;
        const base = y * w * 4;
        const gy = Math.min(meanRows - 1, Math.max(0, Math.round((90 - lat) / 180 * meanRows)));
        const mb = gy * 4;
        for (let x = 0; x < w; x++) {
            const o = base + x * 4;
            for (let c = 0; c < 4; c++) {
                data[o + c] = Math.round(data[o + c] + (means[mb + c] - data[o + c]) * wgt);
            }
        }
    }
    return m;
}

/** Decode an image to RGBA pixels at a given size. */
async function decode(url, width, height) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    const bitmap = await createImageBitmap(await res.blob());
    const w = width || bitmap.width;
    const h = height || bitmap.height;
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, w, h);
    bitmap.close();
    return { data: new Uint8Array(ctx.getImageData(0, 0, w, h).data.buffer), width: w, height: h };
}

/** Are the downloaded textures present? */
async function haveAssets() {
    try {
        const r = await fetch(`${ASSETS}/day.jpg`, { method: 'HEAD' });
        return r.ok;
    } catch { return false; }
}

/** Column letters run west→east from 180°W; rows run north→south. */
const TILE_COLS = ['A', 'B', 'C', 'D'];
const TILE_ROWS = ['1', '2'];

/** Are the 500 m tiles on disk? */
async function haveTiles() {
    return fetch(`${ASSETS}/tiles/day.A1.4096.jpg`, { method: 'HEAD' })
        .then(r => r.ok).catch(() => false);
}

/**
 * One NASA tile, decoded at `edge` px square.
 *
 * The originals are 21600² — 1.9 GB of RGBA each, well past what a browser will
 * decode — so the fetch script resamples them to 4096 and we take them down
 * further for the coarse level.
 */
async function loadTile(col, row, edge) {
    return decode(`${ASSETS}/tiles/day.${col}${row}.4096.jpg`, edge, edge);
}

/** The single 5400×2700 file, for when the tiles were never fetched. */
async function loadWholeDayMap() {
    const day = await decode(`${ASSETS}/day.jpg`);
    // Blue Marble has no Arctic. It is built from visible-light MODIS passes
    // and the region inside the Arctic Circle spends months in polar night, so
    // every monthly mosaic pads the top of the map with the same near-black
    // fill — the December and July releases are byte-identical down to about
    // 82.6°N. On a globe that is a black cap ringed by a feathered fringe where
    // the fill meets real coastline, which is what the Arctic "pinwheel"
    // actually was; no amount of polar filtering fixes it, because there is
    // nothing there to filter. The 2002 Blue Marble has the sea ice.
    const cap = await ensureIceCap();
    if (cap) blendPolarCap(day, cap, 72, 82);
    return day;
}

/**
 * Splice another equirectangular map into a latitude cap, feathered.
 *
 * The mix has to happen in *linear* light, or every partial blend of white ice
 * over dark sea comes out too dark and the join reads as a grey ring.
 *
 * `fromLat`/`toLat` are degrees north; pass both negative, descending, for the
 * south cap. Antarctica needs no such help: it is land, it is imaged, and it
 * renders correctly as it is.
 *
 * The bounds say which patch of globe `base` covers, so the same function
 * serves the whole map and a single 90°×90° tile. `cap` is always a whole
 * globe and is sampled bilinearly, so it need not match `base`'s resolution.
 */
function blendPolarCap(base, cap, fromLat, toLat,
                       latTop = 90, latBottom = -90, lonLeft = -180, lonRight = 180) {
    const { width: w, height: h, data } = base;
    const { width: cw, height: ch, data: cd } = cap;
    if (toLat === fromLat || cw < 2 || ch < 2) return base;
    const toLinear = (b) => {
        const v = b / 255;
        return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    const toSrgb = (v) => {
        const c = Math.min(Math.max(v, 0), 1);
        const e = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
        return Math.round(e * 255);
    };
    for (let y = 0; y < h; y++) {
        const lat = latTop + (latBottom - latTop) * ((y + 0.5) / h);
        const t = Math.min(Math.max((lat - fromLat) / (toLat - fromLat), 0), 1);
        const mix = t * t * (3 - 2 * t);
        if (mix <= 0) continue;
        const cy = Math.min(Math.max((90 - lat) / 180 * ch - 0.5, 0), ch - 1);
        const y0 = Math.floor(cy), fy = cy - y0, y1 = Math.min(y0 + 1, ch - 1);
        for (let x = 0; x < w; x++) {
            const lon = lonLeft + (lonRight - lonLeft) * ((x + 0.5) / w);
            const cx = (((lon + 180) / 360 * cw - 0.5) % cw + cw) % cw;
            const x0 = Math.floor(cx), fx = cx - x0, x1 = (x0 + 1) % cw;
            const o = (y * w + x) * 4;
            for (let c = 0; c < 3; c++) {
                const s = (xi, yi) => toLinear(cd[(yi * cw + xi) * 4 + c]);
                const top = s(x0, y0) * (1 - fx) + s(x1, y0) * fx;
                const bot = s(x0, y1) * (1 - fx) + s(x1, y1) * fx;
                const b = toLinear(data[o + c]);
                data[o + c] = toSrgb(b + (top * (1 - fy) + bot * fy - b) * mix);
            }
        }
    }
    return base;
}

/**
 * Relief and water from the elevation map.
 *
 * NASA does not ship a specular mask with Blue Marble, but the bathymetry in
 * the elevation product says where the water is — below sea level is ocean, and
 * ocean is smooth.
 */
function reliefFromElevation(elev) {
    const { width: w, height: h, data } = elev;
    const normal = new Uint8Array(w * h * 4);
    const rough = new Uint8Array(w * h * 4);
    const at = (x, y) => data[((Math.min(Math.max(y, 0), h - 1)) * w + (((x % w) + w) % w)) * 4] / 255;
    // GEBCO's ramp puts sea level near the middle of the range.
    const SEA = 0.5;
    for (let y = 0; y < h; y++) {
        const lat = ((y + 0.5) / h - 0.5) * Math.PI;
        const scale = 1 / Math.max(Math.cos(lat), 0.2);
        for (let x = 0; x < w; x++) {
            const e = at(x, y);
            const land = e > SEA;
            const strength = land ? 9.0 : 0.6;
            const dx = (at(x + 1, y) - at(x - 1, y)) * 0.5 * scale * strength;
            const dy = (at(x, y + 1) - at(x, y - 1)) * 0.5 * strength;
            const len = Math.sqrt(dx * dx + dy * dy + 1);
            const o = (y * w + x) * 4;
            normal[o] = (-dx / len * 0.5 + 0.5) * 255;
            normal[o + 1] = (-dy / len * 0.5 + 0.5) * 255;
            normal[o + 2] = (1 / len * 0.5 + 0.5) * 255;
            normal[o + 3] = 255;
            // Water is rougher than it looks. Cox & Munk (1954) measured the
            // sea surface's slope distribution from photographs of sun glitter:
            // mean square slope = 0.003 + 0.00512 W, for W the wind at 10 m in
            // m/s. GGX's alpha is the RMS microfacet slope and the shader takes
            // alpha = roughness², so roughness = mss^(1/4) — 0.45 at the global
            // mean ocean wind of about 7 m/s, for an 11° lobe.
            //
            // This was 0.28, which is a wind of 0.6 m/s: a dead calm, which
            // squeezes the sun's reflection into a hard bright dot instead of
            // the broad glint you see from orbit.
            const r = land ? 0.85 : 0.45;
            rough[o] = r * 255; rough[o + 1] = r * 255; rough[o + 2] = r * 255; rough[o + 3] = 255;
        }
    }
    return { normal: { data: normal, width: w, height: h }, roughness: { data: rough, width: w, height: h } };
}

/** NASA's cloud composite is greyscale; opacity comes from its brightness. */
function cloudsFromGrey(img) {
    const out = new Uint8Array(img.data.length);
    for (let i = 0; i < out.length; i += 4) {
        out[i] = 255; out[i + 1] = 255; out[i + 2] = 255;
        out[i + 3] = img.data[i];
    }
    return { data: out, width: img.width, height: img.height };
}

/**
 * The Moon, from the CGI Moon Kit.
 *
 * Decoded well below the source resolution by default. The colour map is
 * 8192x4096 — 134 MB of RGBA once decoded — and the Moon is thirty pixels
 * across until you fly to it, so paying that up front is a poor trade. `?moon=`
 * raises it.
 */
async function loadMoonMaps(edge) {
    // Baked, if it is there: albedo already scaled to the Moon's real
    // reflectance, relief already differentiated out of LOLA, poles already
    // eased. See `examples/bake_planet_maps.rs`.
    // In the worker: three 4096x2048 maps is a hundred megabytes of RGBA, and
    // decoding them on the main thread was a single 4.4 second freeze.
    const relief = Math.min(edge, 5760);
    const baked = await decodeInWorker({
        colour: { url: `${ASSETS}/baked/moon_albedo.png`, width: edge, height: edge / 2 },
        normal: { url: `${ASSETS}/baked/moon_normal.png`, width: relief, height: relief / 2 },
        displacement: { url: `${ASSETS}/baked/moon_height.png`, width: relief, height: relief / 2 },
    });
    if (baked && baked.colour) return baked;

    const colour = await decode(`${ASSETS}/sky/moon_color.png`, edge, edge / 2)
        .catch(() => decode(`${ASSETS}/sky/moon_color_4k.png`, edge, edge / 2));
    // LOLA is 5760 px wide, so half the colour edge is about its native
    // resolution — decoding it larger would only interpolate.
    const height = await decode(`${ASSETS}/sky/moon_height.png`,
        Math.min(edge, 5760), Math.min(edge / 2, 2880));
    const m = { colour: scaleAlbedo(colour, 0.55), ...moonRelief(height) };
    // The Moon pinwheels at its poles exactly as Earth does.
    for (const k of ['colour', 'normal', 'displacement']) {
        if (m[k]) easePoles(m[k], zonalMeans(m[k]), m[k].height);
    }
    return m;
}

/**
 * Scale a colour map's brightness, in linear light.
 *
 * LRO's mosaic is brightness-stretched for legibility, but the Moon's
 * geometric albedo is about 0.12 — worn asphalt. Left alone it renders as a
 * white disc next to a correctly-exposed Earth. Scaling the sRGB bytes directly
 * would crush the midtones far more than the ends, hence the round trip.
 */
function scaleAlbedo(img, factor) {
    const toLinear = (b) => { const v = b / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    const toSrgb = (v) => {
        const c = Math.min(Math.max(v, 0), 1);
        return Math.round((c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055) * 255);
    };
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
        d[i] = toSrgb(toLinear(d[i]) * factor);
        d[i + 1] = toSrgb(toLinear(d[i + 1]) * factor);
        d[i + 2] = toSrgb(toLinear(d[i + 2]) * factor);
    }
    return img;
}

/**
 * A normal map and a height map from LOLA elevation.
 *
 * The height is stretched to the full range first: LOLA fills only 7..161 of
 * its 255 levels, so used as-is it throws away two thirds of the precision and
 * adds a constant swell to whatever it displaces. Craters are the Moon's whole
 * character, so the relief takes a heavier hand than Earth's.
 */
function moonRelief(elev) {
    const { width: w, height: h, data } = elev;
    let lo = 255, hi = 0;
    for (let i = 0; i < data.length; i += 4) { const v = data[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = Math.max(hi - lo, 1);
    const at = (x, y) =>
        (data[((Math.min(Math.max(y, 0), h - 1)) * w + (((x % w) + w) % w)) * 4] - lo) / span;

    const normal = new Uint8Array(w * h * 4);
    const disp = new Uint8Array(w * h * 4);
    // Slope is measured per texel, not per unit of ground, so the same terrain
    // at half the resolution shows twice the height change between neighbours.
    // Without scaling by width a downsampled map turns into a field of facets.
    // 14 is tuned for LOLA's `ldem_16` at 5760 px.
    const strength = 14.0 * w / 5760;
    for (let y = 0; y < h; y++) {
        const lat = ((y + 0.5) / h - 0.5) * Math.PI;
        const scale = 1 / Math.max(Math.cos(lat), 0.2);
        for (let x = 0; x < w; x++) {
            const dx = (at(x + 1, y) - at(x - 1, y)) * 0.5 * scale * strength;
            const dy = (at(x, y + 1) - at(x, y - 1)) * 0.5 * strength;
            const len = Math.sqrt(dx * dx + dy * dy + 1);
            const o = (y * w + x) * 4;
            normal[o] = (-dx / len * 0.5 + 0.5) * 255;
            normal[o + 1] = (-dy / len * 0.5 + 0.5) * 255;
            normal[o + 2] = (1 / len * 0.5 + 0.5) * 255;
            normal[o + 3] = 255;
            const v = at(x, y) * 255;
            disp[o] = v; disp[o + 1] = v; disp[o + 2] = v; disp[o + 3] = 255;
        }
    }
    return {
        normal: { data: normal, width: w, height: h },
        displacement: { data: disp, width: w, height: h },
    };
}

/**
 * Load maps that were precomputed by `examples/bake_planet_maps.rs`, or return
 * null if they are not there.
 *
 * Everything between the NASA downloads and a texture the GPU can take is a
 * pure function of files on disk: splicing the Arctic in, easing the poles,
 * differentiating GEBCO into a normal map, turning its bathymetry into ocean
 * roughness, reshaping the cloud photo into an alpha mask. Doing that here
 * costs a few seconds of single-threaded JavaScript over 30-odd million texels,
 * every load, on every machine, to arrive at bytes that never differ. Baked,
 * the same maps are just PNG decodes the browser does off-thread.
 *
 * `?baked=0` forces the slow path, which is what to reach for when changing the
 * pipeline — otherwise the bake has to be rerun to see any of it.
 */
/**
 * Decode a set of images in the Web Worker and get their pixels back.
 *
 * `createImageBitmap` already decompresses off the main thread, but the
 * draw-and-read-back that turns a bitmap into the RGBA bytes wasm wants is
 * synchronous, and at 4096x2048 that is 33 MB a map. Doing it on the main
 * thread froze the page for four and a half seconds while the Moon loaded —
 * one unbroken block, measured. The buffers come back transferred, so moving
 * them costs nothing, and a URL that fails is simply left out.
 *
 * Entries are `url` or `{ url, width, height }` to decode at a chosen size.
 */
function decodeInWorker(entries) {
    return new Promise((resolve) => {
        let worker;
        try {
            worker = new Worker('./earth-worker.js', { type: 'module' });
        } catch {
            resolve(null);
            return;
        }
        worker.onmessage = (e) => {
            if (!e.data || !('decoded' in e.data)) return;
            worker.terminate();
            resolve(e.data.decoded);
        };
        worker.onerror = () => { worker.terminate(); resolve(null); };
        worker.postMessage({ decode: entries });
    });
}

async function loadBakedMaps(say, edge = 0) {
    say('loading baked maps…');
    // `edge` of 0 means "whatever the file is"; anything else caps it, which is
    // how a phone asks for a quarter of the pixels.
    const at = (file, edge) => (edge
        ? { url: `${ASSETS}/baked/${file}`, width: edge, height: edge / 2 }
        : `${ASSETS}/baked/${file}`);
    const maps = await decodeInWorker({
        albedo: at('albedo.png', edge),
        normal: at('normal.png', edge),
        roughness: at('roughness.png', edge),
        night: at('night.png', edge ? edge : 0),
        clouds: at('clouds.png', edge && Math.min(edge, 2048)),
        displacement: at('height.png', edge),
    });
    // Albedo is the one that cannot be substituted; without it there is nothing
    // to bake against and the raw pipeline is the honest answer.
    if (!maps || !maps.albedo) return null;
    // Clouds bake with cover already in alpha, so cloudsFromGrey must not run
    // again — it would overwrite the shape with a flat white sheet.
    return { ...maps, real: true, baked: true };
}

/** Load the shared (non-tiled) NASA maps, or throw. */
async function loadNasaMaps(edge, say) {
    say('loading day map…');
    const day = await loadWholeDayMap();
    say('loading night lights…');
    const night = await decode(`${ASSETS}/night.jpg`);
    say('loading clouds…');
    const clouds = cloudsFromGrey(await decode(`${ASSETS}/clouds.jpg`));
    say('deriving relief from elevation…');
    const elev = await decode(`${ASSETS}/elevation.4096.png`).catch(
        () => decode(`${ASSETS}/elevation.png`, 4096, 2048));
    const { normal, roughness } = reliefFromElevation(elev);
    // `elev` doubles as the height map: red is elevation, which is exactly what
    // the vertex stage samples.
    return { day, night, clouds, normal, roughness, displacement: elev, real: true };
}

// ---------------------------------------------------------------------------
// Lens flare, drawn on a 2D canvas over the WebGPU one
// ---------------------------------------------------------------------------
//
// A flare is a property of the camera, not the scene, so it belongs in screen
// space. On the web a stacked 2D canvas is the cheapest place to put it: no
// readback of the WebGPU target, and the compositor does the blending.

// Each ghost is one internal reflection: a position along the axis from the
// sun through the frame centre, a radius, a tint and a strength.
// Each entry is one internal reflection: how far along the axis from the sun
// through the frame centre it lands, its radius as a fraction of the short
// side, its tint, and its strength. Kept small and faint — a flare is
// something you notice at the edge of attention, not a row of discs.
const GHOSTS = [
    [0.30, 0.013, '90, 140, 255', 0.13],
    [0.55, 0.022, '255, 185, 90', 0.10],
    [0.80, 0.009, '140, 255, 180', 0.14],
    [1.00, 0.030, '100, 150, 255', 0.055],
    [1.25, 0.015, '255, 140, 115', 0.09],
    [1.55, 0.038, '115, 190, 255', 0.040],
    [1.85, 0.011, '255, 215, 130', 0.08],
    [2.10, 0.020, '180, 130, 255', 0.045],
];

/**
 * A radial falloff with enough stops to have no visible edge.
 *
 * `createRadialGradient` interpolates linearly between stops, so a two- or
 * three-stop gradient has a kink in its slope at each one — and the eye finds
 * those kinks and reads them as rings. Sampling a smooth curve at a dozen
 * points keeps the slope continuous, which is the difference between light and
 * a drawn circle. `shape(u)` returns brightness at radius fraction `u`.
 */
function radial(ctx, x, y, r, rgb, alpha, shape) {
    if (r <= 0.5 || alpha <= 0) return;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    const STOPS = 14;
    for (let i = 0; i <= STOPS; i++) {
        const u = i / STOPS;
        g.addColorStop(u, `rgba(${rgb}, ${(alpha * shape(u)).toFixed(4)})`);
    }
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
}

/** A soft blob: bright in the middle, gone by the rim. */
function disc(ctx, x, y, r, rgb, alpha) {
    radial(ctx, x, y, r, rgb, alpha, (u) => Math.exp(-u * u * 5.5) * (1 - u));
}

/**
 * An iris ghost.
 *
 * A real one is an out-of-focus image of the aperture and does carry a brighter
 * rim — but at these sizes and opacities the rim is most of what you see, and
 * it reads as a drawn circle rather than as a reflection. A plain soft blob is
 * the honest choice here: slightly wrong optics, visibly better picture.
 */
function ghost(ctx, x, y, r, rgb, alpha) {
    radial(ctx, x, y, r, rgb, alpha, (u) => Math.exp(-u * u * 3.4) * (1 - u * u));
}

/** The glow hugging the planet's limb, brightest where the sun is behind it. */
function drawAtmosphere(ctx, w, h, cx, cy, radius, sx, sy, strength) {
    if (radius <= 2 || strength <= 0) return;
    const band = Math.max(radius * 0.05, 1.5);
    const outer = radius + band * 2.4;
    const ring = (alpha, colour) => {
        const g = ctx.createRadialGradient(cx, cy, Math.max(radius - band * 0.8, 0), cx, cy, outer);
        g.addColorStop(0, 'rgba(120, 175, 255, 0)');
        g.addColorStop(0.34, `rgba(${colour}, ${alpha})`);
        g.addColorStop(1, 'rgba(120, 175, 255, 0)');
        return g;
    };
    // A dim ring all the way round…
    ctx.fillStyle = ring(0.20 * strength, '140, 185, 255');
    ctx.beginPath();
    ctx.arc(cx, cy, outer, 0, Math.PI * 2);
    ctx.fill();

    // …and a soft bloom where the sun sits behind the limb, since that is where
    // the air forward-scatters. A clipped wedge would do it too, but the clip
    // boundary shows as a hard edge across the ring.
    const ang = Math.atan2(sy - cy, sx - cx);
    const lx = cx + Math.cos(ang) * radius;
    const ly = cy + Math.sin(ang) * radius;
    disc(ctx, lx, ly, radius * 0.85, '190, 218, 255', 0.28 * strength);
    disc(ctx, lx, ly, radius * 0.30, '225, 238, 255', 0.30 * strength);
}

function drawFlare(ctx, w, h, sx, sy, strength) {
    if (strength <= 0.002) return;
    const cx = w * 0.5, cy = h * 0.5;
    const dx = cx - sx, dy = cy - sy;
    // Scale everything off the *shorter* side. Using the diagonal made the glow
    // and halo so large they stopped reading as optics and just sat over the
    // scene as arbitrary discs.
    const span = Math.min(w, h);

    // Veiling glare: the broad wash a bright source throws across the whole
    // frame by scattering inside the lens barrel. It is what lifts the blacks
    // near the sun, and its absence is part of why a synthetic flare reads as
    // decals on a clean image.
    disc(ctx, sx, sy, span * 0.95, '175, 195, 230', 0.030 * strength);

    // The sun's own glow and core.
    disc(ctx, sx, sy, span * 0.13, '255, 238, 200', 0.16 * strength);
    disc(ctx, sx, sy, span * 0.030, '255, 250, 235', 0.85 * strength);

    // Diffraction starburst. Light bending round the aperture blades throws
    // spikes; an n-blade iris gives n spikes for odd n and n for even, so a
    // six-blade one gives six long and six short between them.
    const BLADES = 6;
    ctx.save();
    ctx.translate(sx, sy);
    for (let i = 0; i < BLADES * 2; i++) {
        const long = i % 2 === 0;
        const len = span * (long ? 0.20 : 0.085);
        const half = Math.max(0.6, span * (long ? 0.0016 : 0.0011));
        ctx.save();
        ctx.rotate((i / (BLADES * 2)) * Math.PI * 2 + 0.35);
        const g = ctx.createLinearGradient(0, 0, len, 0);
        g.addColorStop(0, `rgba(255, 244, 220, ${(long ? 0.30 : 0.18) * strength})`);
        g.addColorStop(0.35, `rgba(255, 240, 215, ${(long ? 0.10 : 0.06) * strength})`);
        g.addColorStop(1, 'rgba(255, 240, 215, 0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.moveTo(0, -half);
        ctx.lineTo(len, 0);
        ctx.lineTo(0, half);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }
    ctx.restore();

    // Anamorphic streak. Drawn as stacked strips with the alpha falling off
    // away from the axis: a single rectangle has hard top and bottom edges, and
    // a 4-pixel bar with a crisp border reads as a drawn line rather than as
    // light spilling across the sensor.
    const reach = span * 0.42;
    const halfHeight = Math.max(2, span * 0.006);
    for (let i = -6; i <= 6; i++) {
        const k = i / 6;
        const fall = Math.exp(-k * k * 3.2);
        const streak = ctx.createLinearGradient(sx - reach, sy, sx + reach, sy);
        streak.addColorStop(0, 'rgba(140, 185, 255, 0)');
        streak.addColorStop(0.5, `rgba(210, 230, 255, ${0.16 * fall * strength})`);
        streak.addColorStop(1, 'rgba(140, 185, 255, 0)');
        ctx.fillStyle = streak;
        ctx.fillRect(sx - reach, sy + k * halfHeight - 1, reach * 2, 2);
    }

    // Ghosts, with chromatic fringing: the glass disperses, so each reflection
    // is really three slightly different-sized images. Drawing the warm and
    // cool ends offset along the flare axis gives the colour separation without
    // reintroducing a rim.
    const axis = Math.hypot(dx, dy) || 1;
    for (const [t, size, rgb, gain] of GHOSTS) {
        const gx = sx + dx * 2 * t, gy = sy + dy * 2 * t;
        const r = span * size;
        const split = r * 0.10;
        ghost(ctx, gx, gy, r, rgb, gain * 0.7 * strength);
        ghost(ctx, gx + dx / axis * split, gy + dy / axis * split, r * 1.03, '255, 150, 110', gain * 0.22 * strength);
        ghost(ctx, gx - dx / axis * split, gy - dy / axis * split, r * 0.97, '110, 165, 255', gain * 0.22 * strength);
    }

    // The halo: the last internal reflection, landing about the point
    // diametrically opposite the sun.
    //
    // A soft annulus, not a stroked circle. `stroke` gives a band of constant
    // width with two hard edges, which reads as a drawn ring sitting on top of
    // the scene rather than as light — the radial gradient below peaks at the
    // ring radius and fades to nothing on both sides, which is what the real
    // artefact looks like.
    // Broad and faint enough to read as a warm wash rather than an outline: a
    // narrow band, however smoothly it is drawn, is still a ring.
    radial(ctx, sx + dx * 2, sy + dy * 2, span * 0.16, '225, 185, 145',
        0.035 * strength, (u) => Math.exp(-((u - 0.72) ** 2) / 0.16) * (1 - u ** 4));
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

// Everything is in Earth radii (6371 km), with Earth at the origin.
const EARTH_R = 1.0;

// The Sun is really 23,481 Earth radii away. Putting it there would leave the
// whole scene inside a rounding error of the depth buffer, so it sits at 250
// with a radius that preserves its *apparent* size: the Sun subtends 0.53° from
// Earth, so radius = 250 * tan(0.265°).
const SUN_DISTANCE = 250.0;
const SUN_R = SUN_DISTANCE * Math.tan(0.265 * Math.PI / 180);

/**
 * The Moon's orbit, from its actual elements.
 *
 * Distances in Earth radii, angles in radians, time in days. The semi-major
 * axis is 384,400 km — 60.3 Earth radii, so the Moon really is that far away
 * and really is that small. It is about four pixels from a view that frames
 * Earth, which is correct and also why the scene opens next to it instead.
 */
const MOON = {
    radius: 1737.4 / 6371.0,          // 0.2727
    a: 384400 / 6371.0,               // 60.336
    e: 0.0549,                        // perigee 57.0, apogee 63.7
    inclination: 5.145 * Math.PI / 180,
    period: 27.321661,                // sidereal month, days
    node: 1.15,                       // longitude of ascending node — a phase
    periapsis: 0.62,                  // argument of periapsis — likewise
};

/** Earth's sidereal day, in days. Not 1: that is the *solar* day. */
const EARTH_SIDEREAL_DAY = 0.9972696;

/**
 * Earth's axial tilt, and the flattening that goes with spinning.
 *
 * The tilt matters for more than seasons here: the Moon's 5.145° inclination is
 * measured to the *ecliptic*, so leaving Earth's poles perpendicular to it puts
 * the two bodies in inconsistent frames.
 *
 * Flattening is 1/298.257 — equatorial 6378.137 km against polar 6356.752, a
 * difference of 21 km. That is about a pixel on a 700-pixel globe: barely
 * visible, but it is one number and the shape is not a sphere.
 */
const EARTH_TILT = 23.4392811 * Math.PI / 180;
const EARTH_POLAR_SCALE = 6356.752 / 6378.137;

/**
 * The Sun's direction at a given instant, from its apparent geocentric
 * position.
 *
 * The low-precision solar formulae from the Astronomical Almanac: mean
 * longitude and anomaly linear in the Julian date, one equation-of-centre term,
 * then ecliptic to equatorial through the obliquity. Good to about a
 * hundredth of a degree over a century either side of 2000, which is far
 * tighter than anything you can see here — but it means the tilt, the seasons
 * and the terminator are all in the right place for a real date instead of
 * being a slider someone dragged.
 *
 * Returned in the scene's frame, which is Earth-centred with +Y along the
 * rotation axis before the tilt is applied — so the obliquity shows up as the
 * Sun climbing above and below the equator through the year, which is exactly
 * what the seasons are.
 */
function sunDirection(julianDay) {
    const n = julianDay - 2451545.0;             // days from J2000.0
    const L = (280.460 + 0.9856474 * n) * Math.PI / 180;   // mean longitude
    const g = (357.528 + 0.9856003 * n) * Math.PI / 180;   // mean anomaly
    // Ecliptic longitude: the equation of centre, to two terms.
    const lambda = L + (1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * Math.PI / 180;
    const eps = (23.439 - 0.0000004 * n) * Math.PI / 180;  // obliquity
    return {
        x: Math.cos(lambda),
        y: Math.sin(lambda) * Math.sin(eps),
        z: Math.sin(lambda) * Math.cos(eps),
    };
}

/** Julian day for a JS Date. */
function julianDay(date) {
    return date.getTime() / 86400000 + 2440587.5;
}

/**
 * Greenwich mean sidereal time, in radians — how far Earth has turned.
 *
 * Sidereal, not solar: this is the angle to the stars, which is what puts a
 * given meridian under the Sun at the right moment.
 */
function gmst(julianDay) {
    const n = julianDay - 2451545.0;
    return ((18.697374558 + 24.06570982441908 * n) % 24) * Math.PI / 12;
}

/**
 * Where the Moon is, `days` into the simulation.
 *
 * A real two-body solution rather than a circle: solve Kepler's equation for
 * the eccentric anomaly, place the point on the ellipse, then rotate it by the
 * argument of periapsis, the inclination and the ascending node. At e = 0.055
 * Newton-Raphson converges in three iterations; six is free and covers the
 * whole range.
 */
function moonState(days) {
    const { a, e, inclination, period, node, periapsis } = MOON;
    const M = (days / period) * Math.PI * 2;
    let E = M;
    for (let i = 0; i < 6; i++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));

    // In the orbital plane, periapsis along +X.
    const px = a * (Math.cos(E) - e);
    const pz = a * Math.sqrt(1 - e * e) * Math.sin(E);

    // Argument of periapsis, in-plane.
    const cw = Math.cos(periapsis), sw = Math.sin(periapsis);
    let x = px * cw - pz * sw;
    let z = px * sw + pz * cw;
    // Inclination, about X.
    const ci = Math.cos(inclination), si = Math.sin(inclination);
    let y = z * si;
    z = z * ci;
    // Ascending node, about Y.
    const cn = Math.cos(node), sn = Math.sin(node);
    [x, z] = [x * cn + z * sn, -x * sn + z * cn];

    return { x, y, z, distance: Math.hypot(x, y, z) };
}

/**
 * The Moon's spin, so the near side faces Earth — it is tidally locked, which
 * is why we only ever see one face of it.
 *
 * The colour map puts lunar longitude 0 (the middle of the near side) at
 * `u = 0.5`, and `SphereGeometry` puts that at local +X. Rotating by `a` about
 * Y sends local +X to `(cos a, 0, -sin a)`; we need it pointing back at Earth,
 * `-(cos λ, 0, sin λ)` for `λ = atan2(z, x)`. That gives `a = π - λ`.
 */
function moonSpin(m) {
    return Math.PI - Math.atan2(m.z, m.x);
}

/**
 * Project a world point to pixels from the camera's own basis.
 *
 * Done by hand rather than through `Vector3.project`, which wants
 * `matrixWorldInverse` / `projectionMatrix` on the JS camera — the wasm side
 * owns those here. Everything needed is already known: eye, target, fov.
 */
function project(v, view, w, h) {
    const { eye, forward, right, up, fov, aspect } = view;
    const d = v.clone().sub(eye);
    const z = d.dot(forward);
    if (z <= 1e-4) return null; // behind the camera
    const tanV = Math.tan((fov * Math.PI) / 360);
    const ndcX = d.dot(right) / z / (tanV * aspect);
    const ndcY = d.dot(up) / z / tanV;
    return [(ndcX * 0.5 + 0.5) * w, (1 - (ndcY * 0.5 + 0.5)) * h];
}

/** How much of the sun clears a sphere of `radius` at the origin, 0..1. */
function sunVisibility(eye, sun, radius) {
    const dir = sun.clone().sub(eye).normalize();
    const toCenter = eye.clone().multiplyScalar(-1);
    const along = toCenter.dot(dir);
    if (along <= 0) return 1;
    const miss = toCenter.clone().sub(dir.clone().multiplyScalar(along)).length();
    return Math.min(Math.max((miss - radius) / (radius * 0.1), 0), 1);
}

async function main() {
    say('loading wasm…');
    await initThreers();

    const params = new URLSearchParams(location.search);
    const debug = params.get('debug') === '1';
    // How much texture a device can be asked to hold.
    //
    // A phone is not a small desktop. Safari on iOS kills a tab that grows past
    // a few hundred megabytes, and this scene decodes to 347 MB of RGBA at desk
    // sizes — six 4096x2048 Earth maps, three for the Moon and a half-float sky
    // — before mip chains add a third on top. That is not a slow load on a
    // phone, it is a blank tab.
    //
    // So halve every map, drop the Moon to 1024, generate the sky rather than
    // decoding a 67 MB one, and leave the 500 m tiles alone. About 87 MB, which
    // fits. `?full=1` overrides for a tablet that can take it.
    const compact =
        params.get('full') !== '1' &&
        (params.get('compact') === '1' ||
            Math.min(screen.width, screen.height) <= 500 ||
            matchMedia('(pointer: coarse)').matches);
    const budget = compact
        ? { earth: 2048, moon: 1024, sky: 'procedural', tiles: false }
        : { earth: 0, moon: 2048, sky: 'exr', tiles: true };
    const size = Number(params.get('texture')) || (compact ? 1024 : 2048);
    // Per-tile edge when composing the eight NASA tiles: 4096 → a 16384×8192
    // globe, 2048 → 8192×4096 for devices with a smaller texture limit.
    const edge = Number(params.get('tile')) || 2048;
    const forceProcedural = params.get('procedural') === '1';
    const flagProcedural = forceProcedural;

    const t0 = performance.now();
    let maps;
    if (!forceProcedural && await haveAssets()) {
        try {
            if (params.get('baked') !== '0') maps = await loadBakedMaps(say, budget.earth);
            if (!maps) maps = await loadNasaMaps(edge, say);
        } catch (e) {
            console.warn('falling back to the procedural planet:', e);
            maps = null;
        }
    }
    if (!maps) {
        try {
            say(`generating ${size}×${size / 2} maps…`);
            maps = await generateMaps(size, (p, stage) => {
                say(`generating ${size}×${size / 2} maps… ${stage} ${Math.round(p * 100)}%`);
            });
        } catch (e) {
            // Almost always one thing: the page was opened from `file://`,
            // where a browser refuses to load a module worker at all and
            // reports it as an access-control failure. Say so, because the
            // alternative is an unhandled rejection naming a line that is
            // fine.
            const hint = location.protocol === 'file:'
                ? 'open it over http:// — a module worker cannot load from file://'
                : e.message;
            errEl.textContent = `could not build the planet maps: ${hint}`;
            throw e;
        }
    }
    // The procedural worker calls its colour map `albedo`; NASA's is `day`.
    if (maps.day) maps.albedo = maps.day;
    // The real sky, if it has been fetched: NASA's Deep Star Maps, 1.7 billion
    // stars from Hipparcos-2, Tycho-2 and Gaia DR2. Decoded as half-float,
    // because over half of it sits below a hundredth of full scale and 8 bits
    // would leave the galaxy as a few grey dots.
    if (maps.real && !forceProcedural && budget.sky === 'exr') {
        // 4k by default: the 8k is 124 MB, and the sky is a backdrop.
        const which = params.get('sky') === '8k' ? 'starmap_8k' : 'starmap_4k';
        try {
            say('loading the sky…');
            const res = await fetch(`${ASSETS}/sky/${which}.exr`);
            if (res.ok) {
                const bytes = new Uint8Array(await res.arrayBuffer());
                maps.skyTexture = new THREE.EXRLoader().parse(bytes);
            }
        } catch (e) {
            console.warn('no Deep Star Map, falling back to generated stars:', e.message);
        }
    }
    // Only when there is no real sky. `maps.stars` is a *fallback* — the one
    // consumer takes `skyTexture ?? stars` — and reaching it means generating a
    // whole procedural planet, elevation field and all six maps, because that is
    // what the worker's one message produces. Doing that while holding NASA's
    // Deep Star Map in hand cost seconds of the load for something nothing
    // would ever sample.
    if (!maps.stars && !maps.skyTexture) {
        const generated = await generateMaps(1024, () => {});
        maps.stars = generated.stars;
    }
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    const px = (maps.albedo.width * maps.albedo.height / 1e6).toFixed(1);
    say(`${maps.real ? 'NASA imagery' : 'procedural maps'} · ${maps.albedo.width}×${maps.albedo.height} (${px} Mtexel) · ${secs}s`);

    // The Moon is optional and independent of Earth's imagery.
    if (maps.real && !forceProcedural && document.getElementById('moon-on')?.checked !== false) {
        try {
            say('loading the Moon…');
            // Level 0 only has to hold until the tiles arrive, and they carry
            // 4096 px *per patch* — sixteen times this. Decoding the whole-globe
            // map at 4096 meant slicing eight 1024x1024 patches and uploading
            // twenty-four textures, which was over a second of frozen main
            // thread for detail that is replaced within seconds. `?moon=` still
            // raises it.
            maps.moon = await loadMoonMaps(Number(params.get('moon')) || budget.moon);
        } catch (e) {
            console.warn('no CGI Moon Kit imagery:', e.message);
        }
    }

    // Take the pinwheel out of both poles before anything else touches the
    // maps — slices, tiles and cloud-shadow lookups all inherit it.
    //
    // `dayMeans` is computed either way: streamed tiles are never baked, so
    // they are still eased as they arrive, against the *whole* globe's zonal
    // means. A 90°-wide tile averaged on its own would converge to a different
    // colour from its three neighbours, and all four meet at the pole.
    const dayMeans = zonalMeans(maps.albedo);
    if (!maps.baked) {
        say('easing the poles…');
        for (const key of ['albedo', 'normal', 'roughness', 'night', 'clouds', 'displacement']) {
            const m = maps[key];
            if (m) easePoles(m, key === 'albedo' ? dayMeans : zonalMeans(m), m.height);
        }
    }

    // Tiled detail is only available with the 500 m NASA tiles on disk.
    const tiled = budget.tiles && maps.real && !flagProcedural && await haveTiles();
    // `?tile=` caps the per-tile edge, for devices that cannot spare the memory
    // (four 4096² RGBA tiles is 268 MB before mips).
    const capEdge = Number(params.get('tile')) || 4096;
    const maxLevel = capEdge >= 4096 ? 3 : capEdge >= 2048 ? 2 : 1;

    const renderer = await THREE.WebGLRenderer.create(canvas);
    // The sun is hundreds of times brighter than the Earth it lights; without a
    // filmic curve everything facing it clips to flat white.
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    const resize = () => {
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.round(rect.width * dpr));
        const h = Math.max(1, Math.round(rect.height * dpr));
        canvas.width = w; canvas.height = h;
        // The flare canvas does not need the scene's resolution. It is cleared
        // and refilled with radial gradients every frame, on the CPU, and at
        // full-window Retina that is five to eight megapixels of 2D work per
        // frame — which is a cost that arrived the moment this demo went
        // full-window. Nothing it draws has an edge in it: capping the backing
        // store and letting CSS scale it up is free of visible difference and
        // cuts the work by the square of the ratio.
        const FLARE_MAX = 1280;
        // Against the larger dimension, or a portrait phone escapes the cap.
        const fs = Math.min(1, FLARE_MAX / Math.max(w, h, 1));
        overlay.width = Math.max(1, Math.round(w * fs));
        overlay.height = Math.max(1, Math.round(h * fs));
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    };

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x01020a);

    // Colour maps are sRGB-encoded (every JPEG is); normal and roughness maps
    // are data and must stay linear. Getting this backwards is what washed the
    // planet out earlier.
    const sharedTexCache = new Map();
    /** One upload per map, for maps several materials read. */
    const shared = (m, colour = true) => {
        if (!sharedTexCache.has(m)) sharedTexCache.set(m, tex(m, colour));
        return sharedTexCache.get(m);
    };
    const tex = (m, colour = true) => {
        // Every producer here — the NASA decode, the tile loader and the
        // procedural worker — writes the north pole in row 0, the way an
        // equirectangular image is stored. A DataTexture is uploaded verbatim,
        // so flip once, here, where the pixels become a texture.
        flipRows(m.data, m.width, m.height);
        const t = colour
            ? new THREE.DataTexture(m.data, m.width, m.height, 'srgb')
            : new THREE.DataTexture(m.data, m.width, m.height);
        // Equirectangular maps repeat in longitude and clamp at the poles, and
        // want trilinear + anisotropic filtering; the wasm side builds the mip
        // chain on upload.
        t.wrapS = 1000; t.wrapT = 1001;
        t.magFilter = 1006; t.minFilter = 1008;
        return t;
    };

    // ---- the globe, as eight tiles ----
    //
    // One sphere would need one texture, and WebGPU guarantees only 8192 px —
    // so a single-texture globe tops out at 8192x4096 no matter how much
    // imagery you have. Eight patches carry eight textures: at 4096 each that
    // is a 16384x8192 globe, and it is the only way the 500 m NASA tiles are
    // reachable at all.
    //
    // The shared maps (relief, roughness, city lights) stay whole and each
    // patch samples its own quarter through the texture's offset/repeat, so
    // they are uploaded once rather than eight times.
    // Cut the quarter of a whole-globe map that a patch covers, with a wide
    // border of texels carried over from its neighbours.
    //
    // A patch's UVs run 0..1, so the filter reaches past the outer texel centre
    // and clamps — and the two patches meeting there clamp to different values,
    // which draws a line down the join. The border gives the filter real data
    // on both sides instead.
    //
    // It has to be *wide*, not just a texel or two. The mip chain halves the
    // border at every level, so a 2-texel bleed is gone by mip 3, and 16x
    // anisotropic filtering at a grazing angle reaches about 16 texels sideways
    // — which is why a narrow bleed still seamed near the limb while looking
    // clean face-on. 32 covers the anisotropic reach at mip 0 and still leaves
    // a texel at mip 5, by which point a patch is a few dozen pixels across.
    //
    // The cost is 13% more pixels per slice. Longitude wraps, so the side
    // borders come from the far edge of the map; latitude does not, so the top
    // and bottom clamp.
    const BLEED = 32;
    const sliceMap = (m, col, row) => {
        const sw = Math.floor(m.width / 4), sh = Math.floor(m.height / 2);
        const ow = sw + BLEED * 2, oh = sh + BLEED * 2;
        const out = new Uint8Array(ow * oh * 4);
        for (let y = 0; y < oh; y++) {
            const sy = Math.min(Math.max(row * sh + y - BLEED, 0), m.height - 1);
            const rowBase = sy * m.width;
            const src = (rowBase + col * sw) * 4;
            out.set(m.data.subarray(src, src + sw * 4), (y * ow + BLEED) * 4);
            for (let b = 0; b < BLEED; b++) {
                for (const [dst, sxRaw] of [[b, col * sw - BLEED + b], [ow - BLEED + b, col * sw + sw + b]]) {
                    const sx = ((sxRaw % m.width) + m.width) % m.width;
                    const s = (rowBase + sx) * 4, d = (y * ow + dst) * 4;
                    out[d] = m.data[s]; out[d + 1] = m.data[s + 1];
                    out[d + 2] = m.data[s + 2]; out[d + 3] = m.data[s + 3];
                }
            }
        }
        return { data: out, width: ow, height: oh, inset: [BLEED / ow, BLEED / oh, sw / ow, sh / oh] };
    };
    const subTex = (m, colour, col, row) => {
        const sliced = sliceMap(m, col, row);
        const t = tex(sliced, colour);
        const [ox, oy, rx, ry] = sliced.inset;
        t.offset = { x: ox, y: oy };
        t.repeat = { x: rx, y: ry };
        return t;
    };
    /**
     * A per-tile texture, inset by half a texel.
     *
     * NASA ships the tiles as exact 90° blocks with no overlap, so there is no
     * neighbour to bleed from — but stopping at the outer texel *centres*
     * instead of the outer edges is enough to stop the clamp, and adjacent
     * tiles then meet on adjacent samples.
     */
    const tileTex = (m, col, row, globe) => {
        // The tiles are separate files with no neighbour to bleed from — but
        // the whole-globe map is always loaded and covers the same ground, so
        // the border comes from there. Both sides of a join then read the same
        // source and agree, which is what stops the seam; that the border is
        // coarser than the tile's interior does not show over 32 texels.
        const b = Math.max(4, Math.round(m.width / 128));
        const ow = m.width + b * 2, oh = m.height + b * 2;
        const out = new Uint8Array(ow * oh * 4);
        const g = globe, gsw = g.width / 4, gsh = g.height / 2;
        for (let y = 0; y < oh; y++) {
            if (y >= b && y < b + m.height) {
                const src = (y - b) * m.width * 4;
                out.set(m.data.subarray(src, src + m.width * 4), (y * ow + b) * 4);
            }
            for (let x = 0; x < ow; x++) {
                if (x >= b && x < b + m.width && y >= b && y < b + m.height) continue;
                // Same point on the globe, read out of the whole-globe map.
                const gx = Math.min(Math.max(Math.round(col * gsw + (x - b) / m.width * gsw), 0), g.width - 1);
                const gy = Math.min(Math.max(Math.round(row * gsh + (y - b) / m.height * gsh), 0), g.height - 1);
                const s = (gy * g.width + ((gx % g.width) + g.width) % g.width) * 4, d = (y * ow + x) * 4;
                out[d] = g.data[s]; out[d + 1] = g.data[s + 1];
                out[d + 2] = g.data[s + 2]; out[d + 3] = g.data[s + 3];
            }
        }
        const t = tex({ data: out, width: ow, height: oh });
        t.offset = { x: b / ow, y: b / oh };
        t.repeat = { x: m.width / ow, y: m.height / oh };
        return t;
    };

    const HALF_PI = Math.PI / 2;
    const tiles = [];
    const makeTile = (col, row, dayMap, segs) => {
        const mat = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            map: dayMap,
            normalMap: subTex(maps.normal, false, col, row),
            roughnessMap: subTex(maps.roughness, false, col, row),
            emissiveMap: subTex(maps.night, true, col, row),
            emissive: 0xffffff,
            emissiveIntensity: 2.3,
            roughness: 1.0,
            metalness: 0.0,
            // The deck overhead, casting onto the ground. Whole-globe, not the
            // patch's quarter: the shader works out for itself where the sun
            // ray crosses the shell, which can be under a different patch.
            cloudShadowMap: shared(maps.clouds, true),
            cloudHeight: 0.012,
            cloudShadow: 0.55,
            // Narrow: this is light on the *ground*, which dies within a few
            // degrees of the terminator. The broad glow round the night limb is
            // the atmosphere shell's, not this.
            twilight: 0.10,
            twilightColor: 0xff8c52,
            ...(maps.displacement
                ? { displacementMap: subTex(maps.displacement, false, col, row), displacementScale: 0 }
                : {}),
        });
        const geo = new THREE.SphereGeometry(
            EARTH_R, segs, segs,
            col * HALF_PI, HALF_PI,   // longitude quarter
            row * HALF_PI, HALF_PI,   // latitude half
        );
        const mesh = new THREE.Mesh(geo, mat);
        // Euler XYZ applies Z, then Y, then X — so the spin happens about the
        // body's own axis and the tilt leans that axis over, which is the order
        // we want.
        mesh.rotation.x = EARTH_TILT;
        mesh.scale.set(1, EARTH_POLAR_SCALE, 1);
        scene.add(mesh);
        return { col, row, mat, mesh, level: 0 };
    };

    // Level 0 is the whole-globe 5400x2700 file, split across the eight patches
    // by offset/repeat — so a planet is on screen before a single tile has been
    // fetched, and the tiles upgrade it in place.
    for (let row = 0; row < 2; row++) {
        for (let col = 0; col < 4; col++) {
            tiles.push(makeTile(col, row, subTex(maps.albedo, true, col, row), 96));
        }
    }

    const cloudMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        map: shared(maps.clouds, true),
        roughness: 0.95,
        metalness: 0.0,
        // Below 1 so the shell lands on the alpha pipeline; the map's own alpha
        // shapes it from there.
        opacity: 0.96,
        transparent: true,
    });
    const clouds = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R * 1.012, 144, 72), cloudMat);
    clouds.rotation.x = EARTH_TILT;
    clouds.scale.set(1, EARTH_POLAR_SCALE, 1);
    scene.add(clouds);

    // ---- level of detail ----
    //
    // Three levels, each a bigger decode of the same NASA tile:
    //   0  the 5400x2700 whole-globe file, already on screen
    //   1  1024 px per tile   →  4096x2048 globe
    //   2  2048 px per tile   →  8192x4096
    //   3  4096 px per tile   → 16384x8192, the full 500 m product
    //
    // A tile upgrades when the camera is close enough that its texels would
    // otherwise be visible, and only if it is facing us — there is no point
    // paying 64 MB to sharpen the far side of the planet. One decode at a time,
    // so streaming never stalls the frame loop.
    // Level 0 is not "no texture" — it is the whole-globe map split across the
    // same eight patches, so its effective edge is a quarter of the map's
    // width. Recording that is what stops the streamer fetching 1024 px tiles
    // to replace 1024 px it already had, which is what it was doing at the
    // default camera distance: seven tiles decoded, uploaded and worth nothing.
    const LOD_EDGES = [Math.round((maps.albedo?.width ?? 5400) / 4), 1024, 2048, 4096];
    let lodBusy = false;
    // Which level a distance deserves, worked out rather than tabulated.
    //
    // Distance alone cannot answer it: the same camera position wants four
    // times the texels on a 4K display as on a laptop, and half of them at a
    // narrow field of view. What actually matters is how many texels of the map
    // land on one pixel, so this projects the globe, asks how wide it is on
    // screen, and picks the smallest tile that still has a texel per pixel.
    //
    // A globe of radius 1 at distance d subtends `asin(1/d)`; its screen radius
    // follows from the projection. Going the other way, a tile spans 90 degrees
    // of longitude, so the equator needs about `2*pi*screenRadius` texels all
    // the way round and a quarter of that per tile.
    const wantedLevel = (distance) => {
        if (!tiled || distance <= 1.0) return maxLevel;
        const h = canvas.height;
        const halfFov = (camera.fov * Math.PI) / 360;
        const screenRadius = (h / 2) * Math.tan(Math.asin(1 / distance)) / Math.tan(halfFov);
        const needed = (Math.PI / 2) * screenRadius;
        // Smallest edge that covers it; `maxLevel` still caps for memory.
        for (let i = 0; i < LOD_EDGES.length; i++) {
            if (LOD_EDGES[i] >= needed) return i;
        }
        return LOD_EDGES.length - 1;
    };
    async function updateLod(eye) {
        if (lodBusy || !tiled) return;
        const dist = Math.hypot(eye.x, eye.y, eye.z);
        const want = Math.min(wantedLevel(dist), maxLevel);
        // Nearest under-detailed tile that is actually facing the camera.
        let best = null;
        for (const tile of tiles) {
            if (tile.level >= want) continue;
            // Patch centre, in the rotated frame the meshes are drawn in.
            const lon = (tile.col + 0.5) * HALF_PI + tile.mesh.rotation.y;
            const lat = Math.PI / 2 - (tile.row + 0.5) * HALF_PI;
            const n = {
                x: -Math.cos(lon) * Math.cos(lat),
                y: Math.sin(lat),
                z: Math.sin(lon) * Math.cos(lat),
            };
            const facing = (n.x * eye.x + n.y * eye.y + n.z * eye.z) / dist;
            if (facing < 0.1) continue;
            if (!best || facing > best.facing) best = { tile, facing };
        }
        if (!best) return;
        lodBusy = true;
        const { tile } = best;
        const level = want;
        try {
            const img = await loadTile(TILE_COLS[tile.col], TILE_ROWS[tile.row], LOD_EDGES[level]);
            // A north tile spans 90°..0°, a south one 0°..-90°.
            const latTop = tile.row === 0 ? 90 : 0;
            const latBottom = tile.row === 0 ? 0 : -90;
            // The 500 m tiles are the same product as the whole-globe map, so
            // the north row carries the same missing Arctic. Splice before
            // easing, for the same reason: easing a gap only spreads the gap.
            const cap = tile.row === 0 ? await ensureIceCap() : null;
            if (cap) {
                blendPolarCap(img, cap, 72, 82, latTop, latBottom,
                    tile.col * 90 - 180, (tile.col + 1) * 90 - 180);
            }
            easePoles(img, dayMeans, maps.albedo.height, latTop, latBottom);
            // Each patch's UVs span its own 0..1, so a tile texture needs no
            // offset/repeat — unlike the shared maps beside it.
            tile.mat.map = tileTex(img, tile.col, tile.row, maps.albedo);
            tile.level = level;
            lodLoaded++;
            say(`NASA imagery · ${lodLoaded} tile${lodLoaded === 1 ? '' : 's'} at ` +
                `${LOD_EDGES[level]}px · ${(tiles.reduce((a, t) => a + LOD_EDGES[t.level] ** 2, 0) / 1e6).toFixed(0)} Mtexel`);
        } catch (e) {
            console.warn('tile load failed', tile.col, tile.row, e);
            tile.level = level; // do not retry in a loop
        } finally {
            lodBusy = false;
        }
    }
    let lodLoaded = 0;

    // Are the 16k Moon tiles on disk? Same question the globe asks of its own.
    // Gated by the same budget as the Earth tiles: three 4096 px Moon
    // patches is two hundred megabytes of RGBA, which is most of a
    // phone's whole allowance for a body that is a few hundred pixels
    // across on its screen.
    const moonTiles = budget.tiles && maps.moon
        ? await fetch(`${ASSETS}/sky/tiles/moon.A1.png`, { method: 'HEAD' })
            .then((r) => r.ok).catch(() => false)
        : false;
    const moonMeans = maps.moon ? zonalMeans(maps.moon.colour) : null;

    // ---- the Moon's level of detail ----
    //
    // The same shape as Earth's, on its own budget: the 16k map is split into
    // eight 4096² tiles, and a patch upgrades when the camera is close enough
    // that its texels would show and it is facing us. Distances are in Moon
    // radii from the body, not Earth radii from the origin, because that is
    // what decides how big it is on screen.
    const MOON_TILE_EDGE = 4096;
    let moonBusy = false;
    let moonLoaded = 0;
    async function updateMoonLod(eye, m) {
        if (moonBusy || !moonPatches.length || !moonTiles) return;
        const dx = eye.x - m.x, dy = eye.y - m.y, dz = eye.z - m.z;
        const dist = Math.hypot(dx, dy, dz) / MOON_R;   // in Moon radii
        if (dist > 12) return;                          // too far to be worth it
        let best = null;
        for (const q of moonPatches) {
            if (q.level >= 1) continue;
            // Patch centre in the rotated frame, then toward the camera?
            const lon = (q.col + 0.5) * HALF_PI + moonGroupRot;
            const lat = Math.PI / 2 - (q.row + 0.5) * HALF_PI;
            const n = {
                x: -Math.cos(lon) * Math.cos(lat),
                y: Math.sin(lat),
                z: Math.sin(lon) * Math.cos(lat),
            };
            const facing = (n.x * dx + n.y * dy + n.z * dz) / (dist * MOON_R);
            if (facing < 0.1) continue;
            if (!best || facing > best.facing) best = { q, facing };
        }
        if (!best) return;
        moonBusy = true;
        const { q } = best;
        try {
            const img = await decode(
                `${ASSETS}/sky/tiles/moon.${TILE_COLS[q.col]}${TILE_ROWS[q.row]}.png`,
                MOON_TILE_EDGE, MOON_TILE_EDGE);
            scaleAlbedo(img, 0.55);
            // The poles pinwheel on the Moon exactly as on Earth, and a tile
            // spans only 90° of longitude — so the mean has to come from the
            // whole globe, or the four tiles meeting at a pole would each
            // settle on a different colour.
            easePoles(img, moonMeans, maps.moon.colour.height,
                q.row === 0 ? 90 : 0, q.row === 0 ? 0 : -90);
            q.mat.map = tileTex(img, q.col, q.row, maps.moon.colour);
            q.level = 1;
            moonLoaded++;
            say(`NASA imagery · ${lodLoaded} Earth tiles · ${moonLoaded} Moon tiles at ${MOON_TILE_EDGE}px`);
        } catch (e) {
            console.warn('moon tile failed', q.col, q.row, e.message);
            q.level = 1;
        } finally {
            moonBusy = false;
        }
    }

    // Atmosphere: one shell, shaded analytically.
    //
    // `AtmosphereMaterial` is not a surface material — the shader intersects
    // each view ray with the planet and with this shell and integrates the air
    // between them. That is what makes the limb thicken on its own, the day
    // side glow, the terminator redden, and the planet occlude it correctly.
    const AIR = 1.045;
    const airMat = new THREE.AtmosphereMaterial({
        planetRadius: EARTH_R,
        atmosphereRadius: EARTH_R * AIR,
        color: 0x4d8bff,
        sunsetColor: 0xff7a33,
        intensity: 2.6,
        falloff: 2.3,
        // Oxygen at ~90 km. Faint, and only visible tracing the night limb.
        airglow: 0.55,
        airglowColor: 0x4dff8a,
    });
    const air = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R * AIR, 128, 64), airMat);
    scene.add(air);

    // ---- the Moon ----
    //
    // Real geometry, not a sprite: it takes the same light as Earth, occludes
    // and is occluded, and carries LOLA relief as displacement so crater rims
    // break its silhouette. Radius is to scale (0.2727 Earth radii); the orbit
    // is not — 60 Earth radii would make it four pixels.
    const MOON_R = MOON.radius;
    // ---- the Moon, in eight patches like Earth ----
    //
    // The CGI Moon Kit's 16k colour map is 16384x8192, past what a single
    // texture can be, so the body is built the same way the globe is: four
    // columns by two rows of sphere patches, each carrying its own tile. The
    // relief maps stay whole and are sliced per patch, since LOLA is only
    // 5760 px wide and fits comfortably.
    const moonPatches = [];
    let moon = null;          // the first patch, for anything that needs one
    let moonMat = null;
    let moonGroupRot = 0;
    if (maps.moon) {
        const mm = maps.moon;
        // Each patch addresses its own quarter of the *whole* map through a
        // view, rather than getting a sliced copy of it.
        //
        // The patch geometry's own UVs run 0..1 across the patch, so the view's
        // job is local -> global: a quarter of the width per column, half the
        // height per row. Latitude runs the other way to the row index, because
        // `SphereGeometry` puts v = 1 at the north pole and row 0 is the
        // northern half.
        //
        // Three uploads for the three maps, shared by all eight patches, in
        // place of twenty-four slices uploaded separately — which was over a
        // second of frozen main thread on the first frame. Tiles are unaffected:
        // a tile is a 0..1 map for one patch and keeps its own transform, so it
        // simply replaces the view when it arrives.
        const patchView = (m, colour, col, row) =>
            shared(m, colour).view(col * 0.25, (1 - row) * 0.5, 0.25, 0.5);
        for (let row = 0; row < 2; row++) {
            for (let col = 0; col < 4; col++) {
                const mat = new THREE.MeshStandardMaterial({
                    color: 0xffffff,
                    // Level 0 is the whole-globe 8k map, split across the same
                    // patches; the 16k tiles upgrade it in place.
                    map: patchView(mm.colour, true, col, row),
                    normalMap: patchView(mm.normal, false, col, row),
                    displacementMap: patchView(mm.displacement, false, col, row),
                    // Centred: the stretched height map has no natural zero, so
                    // half the amplitude goes below the mean radius and half
                    // above rather than inflating the whole body.
                    displacementScale: MOON_R * 0.03,
                    displacementBias: -MOON_R * 0.015,
                    roughness: 0.95,
                    metalness: 0.0,
                });
                mat.eclipseOccluder = [0, 0, 0, EARTH_R];
                // Displacement can only be as detailed as the mesh carrying it.
                const mesh = new THREE.Mesh(new THREE.SphereGeometry(
                    MOON_R, 128, 128,
                    col * HALF_PI, HALF_PI,
                    row * HALF_PI, HALF_PI,
                ), mat);
                scene.add(mesh);
                moonPatches.push({ col, row, mat, mesh, level: 0 });
            }
        }
        moon = moonPatches[0].mesh;
        moonMat = moonPatches[0].mat;
    }

    /** Put every Moon patch where the body is, facing the way it faces. */
    const placeMoon = (m) => {
        moonGroupRot = moonSpin(m);
        for (const q of moonPatches) {
            q.mesh.position.set(m.x, m.y, m.z);
            q.mesh.rotation.y = moonGroupRot;
            q.mesh.visible = ui.moon ? ui.moon.checked : true;
        }
    };

    // Stars: unlit, so they hold still in world space as the camera orbits and
    // take no light — and left in half-float where the map is, so the
    // renderer's tone mapping decides what the galaxy looks like rather than an
    // 8-bit clip deciding for it.
    //
    // Whichever sky is in play, the Deep Star Map or the generated one. The
    // generated one used to stay on a sphere because only the EXR path had
    // `cubeFaces`, which left its pole exactly as it was.
    const skyTex = maps.skyTexture ?? tex(maps.stars);
    const skyFaces = skyTex?.cubeFaces?.(0) ?? null;
    const starMat = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        map: skyTex,
        side: 1,
    });
    // Beyond the Sun, so the Sun is inside the sky rather than hidden by it.
    const SKY_R = 400;
    // Six quads if the map can be resampled onto cube faces, a sphere if not.
    //
    // An equirectangular map on a UV sphere converges every longitude onto one
    // texel at each pole, and anything with width near there fans out radially
    // when it is wrapped. A cube has no such point. It is also twelve triangles
    // against four thousand, and it has no date line, because no face wraps.
    //
    // The basis that generated each face places its quad, so a face texel and
    // the direction it was sampled from line up with no rotation bookkeeping.
    const SKY_FACES = [
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    ];
    let starfield;
    if (skyFaces) {
        starfield = new THREE.Group();
        skyFaces.forEach((face, i) => {
            const [f, right, up] = SKY_FACES[i];
            const pos = [];
            for (const [x, y] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
                pos.push(
                    (f[0] + x * right[0] + y * up[0]) * SKY_R,
                    (f[1] + x * right[1] + y * up[1]) * SKY_R,
                    (f[2] + x * right[2] + y * up[2]) * SKY_R,
                );
            }
            const g = new THREE.BufferGeometry();
            g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
            g.setAttribute('normal', new THREE.BufferAttribute(
                new Float32Array([-f[0], -f[1], -f[2], -f[0], -f[1], -f[2],
                                  -f[0], -f[1], -f[2], -f[0], -f[1], -f[2]]), 3));
            g.setAttribute('uv', new THREE.BufferAttribute(
                new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), 2));
            g.setIndex([0, 1, 2, 2, 1, 3]);
            // Double-sided: which way a face winds depends on the handedness of
            // its basis, and being inside the cube means it has to show either
            // way.
            starfield.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({
                color: 0xffffff, map: face, side: 2,
            })));
        });
    } else {
        starfield = new THREE.Mesh(new THREE.SphereGeometry(SKY_R, 64, 32), starMat);
    }
    scene.add(starfield);

    // The sun: a very bright body plus the light it casts.
    const sunDir = new THREE.Vector3(1.0, 0.18, 0.55).normalize();
    const sunPos = sunDir.clone().multiplyScalar(SUN_DISTANCE);
    const sun = new THREE.Mesh(
        new THREE.SphereGeometry(SUN_R, 32, 16),
        new THREE.MeshStandardMaterial({
            color: 0xfff6e0, emissive: 0xfff4d6, emissiveIntensity: 90,
        }),
    );
    sun.position.set(sunPos.x, sunPos.y, sunPos.z);
    scene.add(sun);

    const key = new THREE.DirectionalLight(0xfff6ea, 3.1);
    // The shim's light position is a plain {x,y,z,set()} — not a Vector3.
    key.position.set(sunPos.x, sunPos.y, sunPos.z);
    scene.add(key);
    scene.add(new THREE.AmbientLight(0x0a1428, 0.35));

    // Earthshine. Earth is 1.9° across seen from the Moon and returns about
    // 80 ppm of the sunlight falling on it — faint, but it is why the old moon
    // shows in the new moon's arms.
    //
    // A point light at Earth's centre is the right source and costs nothing
    // extra: Earth's own surface normals point away from it, so it lights the
    // Moon and leaves the planet alone. Decay 0 because the falloff over a
    // fixed 60-radius orbit is a constant, folded into the intensity.
    const earthshine = new THREE.PointLight(0x9ec4ff, 0.06, 0, 0);
    earthshine.position.set(0, 0, 0);
    scene.add(earthshine);

    const camera = new THREE.PerspectiveCamera(56, 16 / 9, 0.01, 900);
    camera.position.set(2.2, 0.7, 2.4);
    camera.lookAt(0, 0, 0);
    resize();
    window.addEventListener('resize', resize);

    // Drag to rotate, wheel to zoom, right-drag to pan.
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.minDistance = 1.35;   // just clear of the surface
    // Far enough out to see the Moon's orbit whole.
    controls.maxDistance = 140;
    // Any interaction takes over from the automatic orbit.
    for (const ev of ['pointerdown', 'wheel']) {
        canvas.addEventListener(ev, () => { manual = true; ui.spin.checked = false; }, { passive: true });
    }

    let manual = params.has('t');
    // Controls
    const ui = {
        spin: document.getElementById('spin'),
        flare: document.getElementById('flare-on'),
        clouds: document.getElementById('clouds-on'),
        moon: document.getElementById('moon-on'),
        orbit: document.getElementById('orbit'),
        sunLon: document.getElementById('sun-lon'),
        sunLat: document.getElementById('sun-lat'),
        night: document.getElementById('night'),
        air: document.getElementById('air'),
        cloudOp: document.getElementById('cloud-op'),
        relief: document.getElementById('relief'),
        rate: document.getElementById('rate'),
    };

    // Days elapsed since the epoch below. Declared here because the controls
    // read it through `nowJd` before the frame loop ever runs.
    let simDays = 0;

    // ---- the epoch ----
    //
    // `?date=` sets the instant the scene starts from; the clock runs on from
    // there. With a real date the Sun's declination, Earth's rotation and the
    // Moon's place in its orbit are all the ones that actually obtained, so the
    // terminator falls where it fell and the seasons are the right way round.
    const startJd = julianDay(
        params.has('date') ? new Date(params.get('date')) : new Date(),
    );
    const nowJd = () => startJd + simDays;

    // ---- live scene controls ----
    //
    // Moving the sun is what actually controls day and night: the terminator is
    // where the light stops reaching, the night map is an emissive that only
    // shows where the surface is dark, and the atmosphere reads the same light
    // for its day side and its sunset band. So one slider drives all three,
    // rather than each being faked separately.
    const applySun = () => {
        // The sliders are an offset on the real direction, not a replacement:
        // dragging them walks the Sun away from where it actually is, and
        // leaving them at zero leaves the ephemeris in charge.
        const s = sunDirection(nowJd());
        const base = Math.atan2(s.z, s.x);
        const baseLat = Math.asin(Math.min(Math.max(s.y, -1), 1));
        const lon = base + Number(ui.sunLon.value) * Math.PI / 180;
        const lat = baseLat + Number(ui.sunLat.value) * Math.PI / 180;
        const d = new THREE.Vector3(
            Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon),
        ).normalize();
        sunDir.set(d.x, d.y, d.z);
        sunPos.set(d.x * SUN_DISTANCE, d.y * SUN_DISTANCE, d.z * SUN_DISTANCE);
        sun.position.set(sunPos.x, sunPos.y, sunPos.z);
        // The shim's light position is a plain {x,y,z,set()}; a directional
        // light aims from its position at the origin.
        key.position.set(sunPos.x, sunPos.y, sunPos.z);
        window.__sun = { dir: [+d.x.toFixed(3), +d.y.toFixed(3), +d.z.toFixed(3)],
            keyPos: [+key.position.x.toFixed(1), +key.position.y.toFixed(1), +key.position.z.toFixed(1)] };
    };
    const applyMaterials = () => {
        const night = Number(ui.night.value);
        const relief = Number(ui.relief.value);
        for (const tile of tiles) {
            tile.mat.emissiveIntensity = night;
            if (maps.displacement) tile.mat.displacementScale = relief;
        }
        cloudMat.opacity = Number(ui.cloudOp.value);
        airMat.intensity = Number(ui.air.value);
    };
    for (const [el, fn] of [
        [ui.sunLon, applySun], [ui.sunLat, applySun],
        [ui.night, applyMaterials], [ui.air, applyMaterials],
        [ui.cloudOp, applyMaterials], [ui.relief, applyMaterials],
    ]) {
        el?.addEventListener('input', fn);
    }
    applySun();
    applyMaterials();
    let scrub = false;
    ui.orbit.addEventListener('input', () => { scrub = true; manual = false; ui.spin.checked = false; });

    let t = params.has('t') ? Number(params.get('t')) : 0.62;
    if (manual) ui.orbit.value = String(t);
    let last = performance.now();

    // ---- the simulation clock ----
    //
    // Both bodies read this, which is the only way their relative motion means
    // anything — but it also means you cannot have both at a comfortable speed.
    // Earth turns 27.3 times per lunar month, so a rate that makes the orbit
    // worth watching spins the planet like a top. The default is set for the
    // planet (a rotation every 80 seconds) and the slider is there for when you
    // want the orbit instead: at the top of its range the Moon comes round in
    // half a minute.
    const hoursPerSecond = () => Number(ui.rate.value);


    // ---- the opening flyover ----
    //
    // At its true distance the Moon is four pixels from anywhere that frames
    // Earth, so the scene opens beside it and falls inward instead. The path is
    // interpolated in spherical coordinates about Earth — radius easing down
    // from lunar distance while the direction slerps round — which reads as an
    // orbital transfer rather than a dolly, and keeps Earth in frame the whole
    // way in. Any interaction hands the camera straight to OrbitControls.
    const FLY_SECONDS = 18;
    let flyElapsed = moon && params.get('flyover') !== '0' && !params.has('t') ? 0 : null;

    document.getElementById('replay')?.addEventListener('click', () => {
        flyElapsed = 0;
        manual = false;
        scrub = false;
    });

    const smootherstep = (x) => { const c = Math.min(Math.max(x, 0), 1); return c * c * c * (c * (c * 6 - 15) + 10); };
    /** Great-circle interpolation between two unit vectors. */
    const slerp = (a, b, u) => {
        const dot = Math.min(Math.max(a.x * b.x + a.y * b.y + a.z * b.z, -1), 1);
        const theta = Math.acos(dot);
        if (theta < 1e-4) return { x: b.x, y: b.y, z: b.z };
        const s = Math.sin(theta);
        const wa = Math.sin((1 - u) * theta) / s, wb = Math.sin(u * theta) / s;
        return { x: a.x * wa + b.x * wb, y: a.y * wa + b.y * wb, z: a.z * wa + b.z * wb };
    };

    // The scene is up, so the loading cover can go. Left in the DOM and faded
    // rather than removed, because `say` keeps writing tile counts to it.
    statusEl?.setAttribute('data-done', '');
    say('running — WebGPU');

    // `?fps=1` — a rolling frame time, for when the answer to "is it choppy"
    // has to come from the machine that thinks so. Off by default; a permanent
    // readout is exactly the sort of text this page went full-bleed to lose.
    const showFps = params.get('fps') === '1';
    let fpsEl = null;
    if (showFps) {
        fpsEl = document.createElement('div');
        fpsEl.style.cssText =
            'position:fixed;top:8px;left:12px;z-index:5;font:12px ui-monospace,monospace;' +
            'color:#8e98ad;background:rgba(9,11,16,.7);padding:4px 8px;border-radius:5px;' +
            'pointer-events:none;white-space:pre';
        document.body.appendChild(fpsEl);
    }
    const fpsWindow = [];

    function frame(now) {
        const dt = Math.min((now - last) / 1000, 0.1);
        if (fpsEl) {
            fpsWindow.push(now - last);
            if (fpsWindow.length > 120) fpsWindow.shift();
            if (fpsWindow.length > 8 && (fpsWindow.length & 7) === 0) {
                const sorted = [...fpsWindow].sort((a, b) => a - b);
                const p50 = sorted[sorted.length >> 1];
                const p95 = sorted[Math.floor(sorted.length * 0.95)];
                fpsEl.textContent =
                    `${(1000 / p50).toFixed(0)} fps   p50 ${p50.toFixed(1)}ms   ` +
                    `p95 ${p95.toFixed(1)}ms   ${canvas.width}x${canvas.height}`;
            }
        }
        last = now;
        simDays += dt * hoursPerSecond() / 24;
        if (!manual && ui.spin.checked) t += dt * 0.02;
        if (scrub) t = Number(ui.orbit.value);
        else if (!manual) ui.orbit.value = String(t % 1);
        if (flyElapsed !== null && (manual || scrub)) flyElapsed = null;

        let eye, look;
        let flying = false;
        // Exposed so tests and screenshots can key off the intro's progress
        // rather than guessing at wall-clock timings.
        window.__flyover = flyElapsed === null ? 1 : flyElapsed / FLY_SECONDS;
        if (flyElapsed !== null) {
            flyElapsed += dt;
            const u = flyElapsed / FLY_SECONDS;
            if (u >= 1) {
                flyElapsed = null;
            } else {
                const m = moonState(simDays);
                const md = Math.hypot(m.x, m.y, m.z);
                // Start off the Moon's sunward shoulder, close enough that it
                // fills a good part of the frame.
                const side = { x: -m.z / md, y: 0, z: m.x / md };
                const start = {
                    x: m.x + side.x * MOON.radius * 3.2 + m.x / md * MOON.radius * 1.5,
                    y: m.y + MOON.radius * 1.1,
                    z: m.z + side.z * MOON.radius * 3.2 + m.z / md * MOON.radius * 1.5,
                };
                const sd = Math.hypot(start.x, start.y, start.z);
                const dir0 = { x: start.x / sd, y: start.y / sd, z: start.z / sd };
                // Where the automatic tour will pick up, so the handover does
                // not jump.
                const yaw = t * Math.PI * 2;
                const end = { x: Math.cos(yaw) * 3.5, y: 0.45 + Math.sin(yaw) * 0.30, z: Math.sin(yaw) * 3.5 };
                const ed = Math.hypot(end.x, end.y, end.z);
                const dir1 = { x: end.x / ed, y: end.y / ed, z: end.z / ed };

                // Hold station by the Moon for the first fifth, then fall
                // inward. The radius interpolates geometrically — the distance
                // spans a factor of sixteen, and a linear ramp would spend the
                // whole shot in the empty middle of it — and is eased twice, so
                // the descent starts and ends gently instead of braking.
                const hold = Math.min(Math.max((u - 0.18) / 0.82, 0), 1);
                const ePos = smootherstep(smootherstep(hold));
                const r = sd * Math.pow(ed / sd, ePos);
                const d = slerp(dir0, dir1, smootherstep(hold));
                eye = new THREE.Vector3(d.x * r, d.y * r, d.z * r);
                camera.position.set(eye.x, eye.y, eye.z);

                // Hold on the Moon for the first third, then swing to Earth.
                //
                // Swung as a *direction* from the camera, not as a point
                // lerped between the two bodies: the space between the Moon and
                // Earth is empty, so interpolating the target position aims the
                // shot at nothing for most of the trip.
                // Turn to Earth early, while the Moon is still ahead of us:
                // once the camera is *between* the two, "toward the Moon" is
                // backwards, and slerping across that half-turn sends the shot
                // spinning through empty sky.
                const eLook = smootherstep(Math.min(Math.max((u - 0.12) / 0.24, 0), 1));
                const norm = (v) => { const l = Math.hypot(v.x, v.y, v.z) || 1; return { x: v.x / l, y: v.y / l, z: v.z / l }; };
                const toMoon = norm({ x: m.x - eye.x, y: m.y - eye.y, z: m.z - eye.z });
                const toEarth = norm({ x: -eye.x, y: -eye.y, z: -eye.z });
                const ld = slerp(toMoon, toEarth, eLook);
                look = new THREE.Vector3(eye.x + ld.x * r, eye.y + ld.y * r, eye.z + ld.z * r);
                camera.lookAt(look.x, look.y, look.z);
                window.__flydbg = { r: +r.toFixed(2), sd: +sd.toFixed(2), ed: +ed.toFixed(2),
                    md: +md.toFixed(2), eye: [+eye.x.toFixed(2), +eye.y.toFixed(2), +eye.z.toFixed(2)] };
                flying = true;
            }
        }
        if (flying) {
            // eye/look already set by the flyover.
        } else if (manual) {
            // The user is driving: OrbitControls owns the camera.
            controls.update?.();
            eye = new THREE.Vector3(camera.position.x, camera.position.y, camera.position.z);
            look = new THREE.Vector3(0, 0, 0);
        } else {
            // Automatic tour, aiming between the planet and the sun so both stay
            // in shot once the sun comes round the limb.
            const yaw = t * Math.PI * 2;
            eye = new THREE.Vector3(
                Math.cos(yaw) * 3.5,
                0.45 + Math.sin(yaw) * 0.30,
                Math.sin(yaw) * 3.5,
            );
            camera.position.set(eye.x, eye.y, eye.z);
            const toEarth = eye.clone().multiplyScalar(-1).normalize();
            const toSun = sunPos.clone().sub(eye).normalize();
            const ahead = Math.min(Math.max((toEarth.dot(toSun) + 0.17) / 0.67, 0), 1);
            const bias = 0.45 * ahead;
            look = toEarth.clone().multiplyScalar(1 - bias)
                .add(toSun.clone().multiplyScalar(bias)).normalize()
                .multiplyScalar(eye.length()).add(eye);
            camera.lookAt(look.x, look.y, look.z);
        }

        const forward = look.clone().sub(eye).normalize();
        let right = forward.clone().cross(new THREE.Vector3(0, 1, 0));
        if (right.length() < 1e-4) right = new THREE.Vector3(1, 0, 0);
        right.normalize();
        const camUp = right.clone().cross(forward).normalize();
        const view = { eye, forward, right, up: camUp, fov: camera.fov, aspect: camera.aspect };

        // Earth on the same clock as the Moon — a sidereal day, not the solar
        // one, because that is the turn a distant observer sees.
        // Sidereal time rather than a bare turn count, so a given meridian
        // faces the Sun at the moment it really does.
        const spin = gmst(nowJd());
        for (const tile of tiles) tile.mesh.rotation.y = spin;
        // The atmosphere shell stays a sphere: its shader intersects the view
        // ray with two spheres analytically, and flattening it by a third of a
        // percent would cost that exactness for a change of about a pixel.
        // The Sun moves with the date — a degree a day, which is what carries
        // the seasons round.
        if (hoursPerSecond() > 0) applySun();

        // The Moon on its real ellipse, tidally locked.
        if (moon) {
            const m = moonState(simDays);
            placeMoon(m);
            updateMoonLod(eye, m);

            // Eclipses. Each body is told the other is out there and how big,
            // and the shader works out how much of the Sun's half-degree disc
            // it covers from each shaded point — so a solar eclipse lands as a
            // small dark patch with a soft edge, and a lunar one dims the whole
            // Moon as it slides into Earth's shadow. The orbit is real enough
            // to produce both without being told when.
            const occ = moon.visible ? [m.x, m.y, m.z, MOON_R] : [0, 0, 0, 0];
            for (const tile of tiles) tile.mat.eclipseOccluder = occ;
        }
        // Stream sharper tiles for whatever the camera is looking at.
        updateLod(eye);
        // Clouds drift a little faster than the ground beneath them — and the
        // shadow has to drift with them, so the surface materials are told how
        // far the deck has run ahead, in turns.
        clouds.rotation.y = spin * 1.06;
        const cloudLead = -(clouds.rotation.y - spin) / (Math.PI * 2);
        for (const tile of tiles) tile.mat.cloudRotation = cloudLead;
        clouds.visible = ui.clouds.checked;

        renderer.render(scene, camera);

        // Screen-space pass on the stacked 2D canvas.
        const w = overlay.width, h = overlay.height;
        octx.clearRect(0, 0, w, h);
        octx.globalCompositeOperation = 'lighter';
        const sunPx = project(sunPos.clone(), view, w, h);
        const centre = project(new THREE.Vector3(0, 0, 0), view, w, h);
        let limb = null;
        if (sunPx && centre) {
            // A point on the limb, to measure the planet's projected radius.
            limb = project(right.clone().multiplyScalar(EARTH_R), view, w, h);
            // (The atmosphere is real geometry now — see the shells above.)
            if (ui.flare.checked && !debug) {
                const visible = sunVisibility(eye, sunPos, EARTH_R * 1.02);
                const inside = (v, s) => Math.min(Math.max(1 - Math.max(-v, v - s) / (s * 0.35), 0), 1);
                drawFlare(octx, w, h, sunPx[0], sunPx[1],
                    visible * inside(sunPx[0], w) * inside(sunPx[1], h));
            }
            if (debug) {
                // Where the projection *thinks* things are, so it can be
                // compared against where they actually rendered.
                octx.globalCompositeOperation = 'source-over';
                octx.strokeStyle = '#ff2d55'; octx.lineWidth = 2;
                octx.beginPath();
                octx.moveTo(sunPx[0] - 30, sunPx[1]); octx.lineTo(sunPx[0] + 30, sunPx[1]);
                octx.moveTo(sunPx[0], sunPx[1] - 30); octx.lineTo(sunPx[0], sunPx[1] + 30);
                octx.stroke();
                octx.strokeStyle = '#32d74b';
                octx.beginPath();
                octx.moveTo(centre[0] - 20, centre[1]); octx.lineTo(centre[0] + 20, centre[1]);
                octx.moveTo(centre[0], centre[1] - 20); octx.lineTo(centre[0], centre[1] + 20);
                octx.stroke();
                if (limb) {
                    const r = Math.hypot(limb[0] - centre[0], limb[1] - centre[1]);
                    octx.strokeStyle = '#0a84ff';
                    octx.beginPath(); octx.arc(centre[0], centre[1], r, 0, Math.PI * 2); octx.stroke();
                }
                octx.globalCompositeOperation = 'lighter';
            }
        }
        octx.globalCompositeOperation = 'source-over';

        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch((e) => {
    errEl.textContent = String(e?.stack || e);
    say('failed');
});
