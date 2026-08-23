// Procedural Earth maps, generated off the main thread.
//
// A direct port of `build_earth_maps` in `examples/earth_sun_flare.rs`: one
// shared elevation field, sampled from 3D value noise *on the sphere* so an
// equirectangular map comes out with no date-line seam and no pole smearing,
// and five maps derived from it so their features agree.
//
// It runs in a Worker because a 4K set is 8.4 M texels per map — enough to
// freeze a tab for seconds if it ran inline.

/**
 * Fetch and decode a set of images to raw RGBA, off the main thread.
 *
 * `entries` is `{ key: url }`; a key whose image is missing is simply left out,
 * so a partial bake degrades to the slow path rather than failing the load.
 */
async function decodeAll(entries) {
    const maps = {};
    const buffers = [];
    await Promise.all(Object.entries(entries).map(async ([key, entry]) => {
        // An entry is a URL, or `{ url, width, height }` to decode at a size
        // other than the file's own — which is most of the point for the Moon,
        // whose colour map is 4096 wide on disk and wanted smaller than that
        // until you fly to it.
        const url = typeof entry === 'string' ? entry : entry.url;
        try {
            const res = await fetch(url);
            if (!res.ok) return;
            const bitmap = await createImageBitmap(await res.blob());
            const w = (typeof entry === 'object' && entry.width) || bitmap.width;
            const h = (typeof entry === 'object' && entry.height) || bitmap.height;
            const canvas = new OffscreenCanvas(w, h);
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(bitmap, 0, 0, w, h);
            bitmap.close();
            const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
            maps[key] = { data: img.data, width: canvas.width, height: canvas.height };
            buffers.push(img.data.buffer);
        } catch { /* leave the key out */ }
    }));
    return { maps, buffers };
}

// ---------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------

// Mixed with `>>>`, deliberately: JavaScript's `>>` is sign-propagating, so
// `h ^ (h >> 16)` always clears the top bit and the hash never exceeds 0.5 —
// which would put this entire planet below sea level.
function hash3(x, y, z, seed) {
    let h = (Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(z, 2246822519)
        ^ Math.imul(seed, 1274126177)) >>> 0;
    h = (h ^ (h >>> 13)) >>> 0;
    h = Math.imul(h, 1274126177) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    return h / 4294967295;
}

const smooth = (t) => t * t * (3 - 2 * t);

function valueNoise(px, py, pz, seed) {
    const xi = Math.floor(px), yi = Math.floor(py), zi = Math.floor(pz);
    const fx = smooth(px - xi), fy = smooth(py - yi), fz = smooth(pz - zi);
    const c = (dx, dy, dz) => hash3(xi + dx, yi + dy, zi + dz, seed);
    const lerp = (a, b, t) => a + (b - a) * t;
    const x00 = lerp(c(0, 0, 0), c(1, 0, 0), fx);
    const x10 = lerp(c(0, 1, 0), c(1, 1, 0), fx);
    const x01 = lerp(c(0, 0, 1), c(1, 0, 1), fx);
    const x11 = lerp(c(0, 1, 1), c(1, 1, 1), fx);
    return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz);
}

function fbm(p, frequency, octaves, seed) {
    let sum = 0, amp = 0.5, norm = 0, f = frequency;
    for (let o = 0; o < octaves; o++) {
        sum += valueNoise(p[0] * f, p[1] * f, p[2] * f, seed + o * 977) * amp;
        norm += amp; amp *= 0.5; f *= 2;
    }
    return sum / Math.max(norm, 1e-6);
}

function ridged(p, frequency, octaves, seed) {
    let sum = 0, amp = 0.5, norm = 0, f = frequency;
    for (let o = 0; o < octaves; o++) {
        const n = valueNoise(p[0] * f, p[1] * f, p[2] * f, seed + o * 313) * 2 - 1;
        sum += (1 - Math.abs(n)) * amp;
        norm += amp; amp *= 0.5; f *= 2;
    }
    return sum / Math.max(norm, 1e-6);
}

/**
 * sRGB → linear.
 *
 * `WebDataTexture` uploads as `Rgba8Unorm`, i.e. the sampler reads the bytes as
 * linear values. The native example uses an `Rgba8UnormSrgb` map and lets the
 * hardware decode, so to land on the same colour the colour maps have to be
 * written already-linearised. Without this everything comes out washed out —
 * a dark ocean byte of 12 reads as 0.047 instead of 0.0037.
 */
function srgbToLinear(v) {
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

/** The surface point for an equirectangular texel. */
/**
 * Linear sRGB for a blackbody at `kelvin`, normalised to unit luminance.
 *
 * Star colour is temperature and nothing else, so it belongs on the hue and not
 * on the brightness — an M dwarf is not a dim white star, it is an orange one,
 * and how bright it looks is the magnitude's business.
 *
 * The Planckian locus is Kim et al.'s cubic fit, good from 1667 K to 25000 K,
 * which spans an M dwarf to a hot B star.
 */
function blackbodyRgb(kelvin) {
    const t = Math.min(Math.max(kelvin, 1667), 25000);
    const t1 = 1 / t, t2 = t1 * t1, t3 = t2 * t1;
    const x = t <= 4000
        ? -0.2661239e9 * t3 - 0.2343589e6 * t2 + 0.8776956e3 * t1 + 0.179910
        : -3.0258469e9 * t3 + 2.1070379e6 * t2 + 0.2226347e3 * t1 + 0.240390;
    const x2 = x * x, x3 = x2 * x;
    let y;
    if (t <= 2222) y = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683;
    else if (t <= 4000) y = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
    else y = 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
    y = Math.max(y, 1e-4);
    // xyY with Y = 1 into XYZ, then the sRGB primaries.
    const xx = x / y, zz = (1 - x - y) / y;
    const rgb = [
        Math.max(3.2404542 * xx - 1.5371385 - 0.4985314 * zz, 0),
        Math.max(-0.9692660 * xx + 1.8760108 + 0.0415560 * zz, 0),
        Math.max(0.0556434 * xx - 0.2040259 + 1.0572252 * zz, 0),
    ];
    // Clipping to the gamut costs luminance, so renormalise after.
    const luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    if (luma <= 1e-6) return [1, 1, 1];
    return [rgb[0] / luma, rgb[1] / luma, rgb[2] / luma];
}

function direction(u, v) {
    const lon = (u - 0.5) * Math.PI * 2;
    const lat = (0.5 - v) * Math.PI;
    const cl = Math.cos(lat), sl = Math.sin(lat);
    return [cl * Math.cos(lon), sl, cl * Math.sin(lon)];
}

/** Elevation in -1..1. Below 0 is ocean. Every map derives from this. */
function elevation(p) {
    const continents = fbm(p, 1.5, 8, 11);
    const warp = fbm(p, 4.3, 4, 71) - 0.5;
    const mass = continents + warp * 0.22;
    const land = mass - 0.545;
    if (land <= 0) {
        const floor = fbm(p, 6.0, 4, 401);
        return Math.max(land * 2.4 - floor * 0.1, -1);
    }
    const mountains = ridged(p, 7.5, 5, 907);
    const relief = fbm(p, 18.0, 4, 55);
    return Math.min(land * 3 + mountains * mountains * 0.6 * Math.min(land * 7, 1) + relief * 0.07, 1);
}

// ---------------------------------------------------------------------------

self.onmessage = (e) => {
    // Decoding a set of baked maps, rather than generating from noise.
    //
    // `createImageBitmap` already decompresses off the main thread, but the
    // draw-and-read-back that turns a bitmap into the RGBA bytes wasm wants is
    // synchronous, and at 4096x2048 that is 33 MB per map. Doing it here keeps
    // the page interactive while it happens, and the buffers come back
    // transferred rather than copied, so moving them costs nothing.
    if (e.data && e.data.decode) {
        decodeAll(e.data.decode).then(
            ({ maps, buffers }) => self.postMessage({ decoded: maps }, buffers),
            (err) => self.postMessage({ decoded: null, error: String(err && err.message || err) }),
        );
        return;
    }
    const size = Math.max(256, Math.min(4096, e.data.size | 0 || 2048));
    const w = size, h = size / 2;
    const post = (progress, stage) => self.postMessage({ progress, stage });

    // --- shared elevation field ---
    const field = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
        const v = (y + 0.5) / h;
        for (let x = 0; x < w; x++) {
            field[y * w + x] = elevation(direction((x + 0.5) / w, v));
        }
        if ((y & 63) === 0) post(y / h, 'terrain');
    }
    const at = (x, y) => field[Math.min(Math.max(y, 0), h - 1) * w + ((x % w) + w) % w];

    const albedo = new Uint8Array(w * h * 4);
    const normal = new Uint8Array(w * h * 4);
    const rough = new Uint8Array(w * h * 4);
    const night = new Uint8Array(w * h * 4);
    const clouds = new Uint8Array(w * h * 4);

    for (let y = 0; y < h; y++) {
        const v = (y + 0.5) / h;
        const lat = (v - 0.5) * Math.PI;
        const polarBase = (Math.abs(lat) - 1.02) / 0.34;
        const habitable = Math.min(Math.max(1 - Math.min(Math.max((Math.abs(lat) - 0.30) / 0.95, 0), 1), 0), 1);
        const nScale = 1 / Math.max(Math.cos(lat), 0.2);
        const band = 0.55 + 0.45 * Math.cos(lat * 6) * Math.max(1 - Math.abs(lat) / 1.7, 0);

        for (let x = 0; x < w; x++) {
            const u = (x + 0.5) / w;
            const d = direction(u, v);
            const e = field[y * w + x];
            const o = (y * w + x) * 4;
            const polar = Math.min(Math.max(polarBase + (fbm(d, 14.0, 3, 4242) - 0.5) * 0.55, 0), 1);

            // ---- albedo ----
            let rgb;
            if (e <= 0) {
                const depth = Math.min(Math.max(-e / 0.6, 0), 1);
                rgb = [0.05 + (0.008 - 0.05) * depth, 0.30 + (0.035 - 0.30) * depth, 0.36 + (0.13 - 0.36) * depth];
            } else {
                const jitter = fbm(d, 9.0, 3, 1234);
                const warmth = Math.min(Math.max(1 - Math.abs(lat) / 1.4, 0), 1) + (jitter - 0.5) * 0.28;
                const belt = Math.min(Math.max(1 - Math.abs(Math.abs(lat) - 0.46) / 0.30, 0), 1);
                const dryness = fbm(d, 5.5, 4, 6060);
                const desert = Math.min(Math.max(belt * 1.25 * (dryness - 0.34) * 3, 0), 1);
                const base = [0.34 + (0.06 - 0.34) * warmth, 0.33 + (0.24 - 0.33) * warmth, 0.28 + (0.07 - 0.28) * warmth];
                const green = [base[0] * 0.5 + 0.045, base[1] * 0.5 + 0.10, base[2] * 0.5 + 0.04];
                const land = [
                    green[0] + (0.62 - green[0]) * desert,
                    green[1] + (0.51 - green[1]) * desert,
                    green[2] + (0.31 - green[2]) * desert,
                ];
                const snowLine = 0.78 - 0.62 * Math.pow(Math.abs(lat) / 1.5, 2);
                const alpine = Math.min(Math.max((e - snowLine) / 0.14, 0), 1);
                const snow = Math.max(polar, alpine);
                rgb = [
                    land[0] + (0.92 - land[0]) * snow,
                    land[1] + (0.94 - land[1]) * snow,
                    land[2] + (0.97 - land[2]) * snow,
                ];
            }
            if (e <= 0 && polar > 0.35) {
                const t = Math.min(Math.max((polar - 0.35) / 0.4, 0), 1);
                rgb = [rgb[0] + (0.88 - rgb[0]) * t, rgb[1] + (0.91 - rgb[1]) * t, rgb[2] + (0.95 - rgb[2]) * t];
            }
            albedo[o] = srgbToLinear(rgb[0]) * 255;
            albedo[o + 1] = srgbToLinear(rgb[1]) * 255;
            albedo[o + 2] = srgbToLinear(rgb[2]) * 255;
            albedo[o + 3] = 255;

            // ---- normal, from the elevation field (Sobel) ----
            const strength = e > 0 ? 3.2 : 0.25;
            const dx = (at(x + 1, y) - at(x - 1, y)) * 0.5 * nScale * strength;
            const dy = (at(x, y + 1) - at(x, y - 1)) * 0.5 * strength;
            const len = Math.sqrt(dx * dx + dy * dy + 1);
            normal[o] = (-dx / len * 0.5 + 0.5) * 255;
            normal[o + 1] = (-dy / len * 0.5 + 0.5) * 255;
            normal[o + 2] = (1 / len * 0.5 + 0.5) * 255;
            normal[o + 3] = 255;

            // ---- roughness: water is a mirror, land is not ----
            const r = e <= 0 ? 0.12 + 0.7 * polar : 0.82 + Math.min(e * 0.15, 0.12);
            const rv = Math.min(Math.max(r, 0), 1) * 255;
            rough[o] = rv; rough[o + 1] = rv; rough[o + 2] = rv; rough[o + 3] = 255;

            // ---- night lights ----
            if (e <= 0.005) {
                night[o] = 0; night[o + 1] = 0; night[o + 2] = 0; night[o + 3] = 255;
            } else {
                const region = fbm(d, 7.0, 4, 5150);
                const towns = fbm(d, 44.0, 3, 8191);
                const lowland = Math.pow(1 - Math.min(Math.max(e / 0.45, 0), 1), 1.5);
                let lit = Math.max(region - 0.52, 0) * 4 * lowland * habitable;
                lit *= Math.min(Math.max((towns - 0.45) * 3, 0), 1);
                lit = Math.pow(Math.min(Math.max(lit, 0), 1), 1.6);
                const core = Math.max(lit - 0.55, 0) / 0.45;
                night[o] = srgbToLinear(lit) * 255;
                night[o + 1] = srgbToLinear(lit * (0.78 + 0.22 * core)) * 255;
                night[o + 2] = srgbToLinear(lit * (0.43 + 0.57 * core)) * 255;
                night[o + 3] = 255;
            }

            // ---- clouds: streaked along longitude ----
            const n = fbm([d[0], d[1] * 2.6, d[2]], 6.0, 6, 31337);
            const a = Math.min(Math.max((n * band - 0.42) * 3.4, 0), 1);
            clouds[o] = 255; clouds[o + 1] = 255; clouds[o + 2] = 255; clouds[o + 3] = a * 255;
        }
        if ((y & 31) === 0) post(y / h, 'maps');
    }

    // ---- stars: a background sphere's worth of sky ----
    //
    // Point sources are what makes a sky hard. A star's flux is fixed, so a
    // camera concentrates it into about one pixel however wide the field of
    // view — brightness lives in the *value*, not in how many texels the star
    // covers. Three things follow, and they are the same three the native
    // generator in `src/planet/maps.rs` applies:
    //
    // - Counts from `N(<m) proportional to 10^(0.6 m)`, the distribution a
    //   uniform spread of stars in space produces, with flux `10^(-0.4 m)` per
    //   magnitude. A few obvious stars, many faint ones, the right ratio.
    // - One fixed sub-texel Gaussian, energy-conserving, stretched by
    //   `1/cos(latitude)` so it stays round on the sphere instead of becoming a
    //   radial streak near the poles. Bright stars then look bigger for free,
    //   because the same kernel's tail stays above the visible threshold
    //   further out — which is why they look bigger through a real lens. The
    //   old map faked it by painting a plus sign, which read as a cross.
    // - Colour from blackbody temperature, normalised to unit luminance so a
    //   cool star is orange rather than dim.
    //
    // Unlike the native path this stays 8-bit, so the very brightest stars
    // clip. That is what a real exposure does too, and the browser's HDR sky is
    // the Deep Star Map EXR when the assets are there; this is the fallback.
    // Scaled with the request rather than fixed: a phone asking for 1024 maps
    // does not then want a 33 MB sky, and 4096 was hardcoded here regardless.
    const sw = Math.min(4096, Math.max(1024, size * 2)), sh = sw / 2;
    const starF = new Float32Array(sw * sh * 3);

    // Galactic pole and centre, in the frame `direction` returns.
    const GPOLE = [-0.868, 0.198, 0.456];
    const GCENTRE = [-0.055, -0.874, -0.483];
    for (let y = 0; y < sh; y++) {
        const v = (y + 0.5) / sh;
        for (let x = 0; x < sw; x++) {
            const d = direction((x + 0.5) / sw, v);
            const sinB = d[0] * GPOLE[0] + d[1] * GPOLE[1] + d[2] * GPOLE[2];
            const toward = d[0] * GCENTRE[0] + d[1] * GCENTRE[1] + d[2] * GCENTRE[2];
            // The bulge is a fat lens, the outer disc a thin one. Clamped
            // because two unit vectors can dot to a hair past -1.
            const b = Math.min(Math.max((toward + 1) * 0.5, 0), 1);
            const bulge = b * b * b;
            const thickness = 0.055 + 0.16 * bulge;
            const glow = Math.exp(-((sinB / thickness) ** 2)) * (0.05 + 0.95 * Math.sqrt(bulge));
            if (glow <= 1e-4) continue;
            const clumps = 0.45 + 0.55 * fbm(d, 14.0, 5, 771);
            // Dust sits in the plane, so it only bites where the glow is, and
            // it multiplies: the Great Rift is cold dust in front of stars, not
            // a gap between them.
            const lane = Math.exp(-((sinB / (thickness * 0.55)) ** 2));
            const dust = Math.min(Math.max(fbm(d, 26.0, 4, 313) * 1.7 - 0.45, 0), 1) * lane;
            const m = glow * clumps * (1 - 0.85 * dust) * 0.05;
            const warmth = 1 + 0.35 * bulge + 0.5 * dust;
            const o = (y * sw + x) * 3;
            starF[o] = m * Math.min(0.72 * warmth, 1.3);
            starF[o + 1] = m * 0.80;
            starF[o + 2] = m * Math.max(0.95 / Math.max(warmth, 1), 0.55);
        }
        if ((y & 255) === 0) post(y / sh, 'stars');
    }

    // A bare LCG will not do: its consecutive outputs lie on a lattice, and
    // five draws go into every star, so that structure becomes a correlation
    // between where a star is and how bright it is.
    let seed = 12345;
    const rnd = () => {
        seed = (Math.imul(seed, 747796405) + 2891336453) >>> 0;
        let x = seed;
        x ^= x >>> 16; x = Math.imul(x, 2246822519);
        x ^= x >>> 13; x = Math.imul(x, 3266489917);
        x ^= x >>> 16;
        return (x >>> 8) / 16777216;
    };
    // Spectral classes in roughly the proportions the naked-eye sky shows,
    // which is far hotter than the true stellar population — the cool dwarfs
    // that dominate it by number are all too faint to see.
    const CLASSES = [
        [0.08, 12000, 18000], [0.22, 7800, 10000], [0.20, 6200, 7500],
        [0.18, 5300, 6000], [0.22, 4000, 5200], [0.10, 2900, 3900],
    ];
    // Two thirds of a texel: at half a texel the sampling grid's own variance is
    // a quarter of the kernel's again, and it lands on latitude only, leaving
    // every star slightly taller than it is wide — a fan of streaks pointing at
    // the pole. See `splat` in `src/planet/maps.rs`.
    const M_LIM = 8.0, M_MIN = -1.5, SIGMA = 0.65, VAR_GRID = 1 / 12;
    for (let i = 0; i < 26000; i++) {
        const z = rnd() * 2 - 1;
        const lon = rnd() * Math.PI * 2;
        // `N(<m) proportional to 10^(0.6 m)`, inverted.
        const u = Math.max(rnd(), 1e-6);
        const mag = Math.max(M_LIM + Math.log10(u) / 0.6, M_MIN);
        const flux = Math.pow(10, -0.4 * (mag - 6)) * 0.55;

        let pick = rnd(), acc = 0, temp = 5800;
        for (const [share, lo, hi] of CLASSES) {
            acc += share;
            if (pick <= acc) { temp = lo + (hi - lo) * rnd(); break; }
        }
        const tint = blackbodyRgb(temp);

        const lat = Math.asin(z);
        const sx = lon / (Math.PI * 2) * sw;
        const sy = (0.5 - lat / Math.PI) * sh;
        // Round on the sphere, so stretched in longitude by 1/cos(latitude) —
        // capped at half the width, which already covers every longitude.
        const stretch = Math.min(1 / Math.max(Math.abs(Math.cos(lat)), 1e-3), sw / 2);
        // The sigma whose *sampled* width is `stretch` times the sampled height,
        // rather than simply `stretch` times the sigma.
        const varV = SIGMA * SIGMA + VAR_GRID;
        const sigmaU = Math.sqrt(Math.max(stretch * stretch * varV - VAR_GRID, SIGMA * SIGMA));
        const rx = Math.max(Math.ceil(sigmaU * 2.5), 1), ry = 2;
        const fx0 = Math.floor(sx), fy0 = Math.floor(sy);
        // Two passes, so the kernel normalises to exactly one even when it is
        // clipped against the top or bottom row.
        let total = 0;
        for (let dy = -ry; dy <= ry; dy++) {
            const yy = fy0 + dy;
            if (yy < 0 || yy >= sh) continue;
            const gy = (yy + 0.5 - sy) / SIGMA;
            for (let dx = -rx; dx <= rx; dx++) {
                const gx = (fx0 + dx + 0.5 - sx) / sigmaU;
                total += Math.exp(-0.5 * (gx * gx + gy * gy));
            }
        }
        if (total <= 1e-9) continue;
        for (let dy = -ry; dy <= ry; dy++) {
            const yy = fy0 + dy;
            if (yy < 0 || yy >= sh) continue;
            const gy = (yy + 0.5 - sy) / SIGMA;
            for (let dx = -rx; dx <= rx; dx++) {
                const xi = fx0 + dx;
                const gx = (xi + 0.5 - sx) / sigmaU;
                const wgt = Math.exp(-0.5 * (gx * gx + gy * gy)) / total;
                const o = (yy * sw + ((xi % sw) + sw) % sw) * 3;
                starF[o] += flux * tint[0] * wgt;
                starF[o + 1] += flux * tint[1] * wgt;
                starF[o + 2] += flux * tint[2] * wgt;
            }
        }
    }

    // Out to bytes. The buffer is already linear light, which is what
    // `Rgba8Unorm` means to the sampler — no sRGB encode on the way out.
    const stars = new Uint8Array(sw * sh * 4);
    for (let i = 0, o = 0; i < starF.length; i += 3, o += 4) {
        stars[o] = Math.min(starF[i], 1) * 255;
        stars[o + 1] = Math.min(starF[i + 1], 1) * 255;
        stars[o + 2] = Math.min(starF[i + 2], 1) * 255;
        stars[o + 3] = 255;
    }

    const map = (data) => ({ data, width: w, height: h });
    self.postMessage(
        {
            albedo: map(albedo), normal: map(normal), roughness: map(rough),
            night: map(night), clouds: map(clouds),
            stars: { data: stars, width: sw, height: sh },
        },
        [albedo.buffer, normal.buffer, rough.buffer, night.buffer, clouds.buffer, stars.buffer],
    );
};
