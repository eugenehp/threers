// What the gummy creatures demo draws, and how: lights, the table, the gel,
// and every mesh, carried along each frame to the simulation's latest
// snapshot (jelly-sim.js) — which may have come from a worker.
//
//   gel       MeshPhysicalMaterial in 'refract' mode: the renderer measures
//             the path through the jelly from its own back faces and absorbs
//             along it (attenuationColor), so a thin tentacle glows and a thick
//             body goes deep
//   insides   opaque surfaces bound to the body's lattice, seen through the gel
//   chains    tapered capsules rebuilt each frame from the particle chains
//   sugar     the sour coating: crystals bound to the lattice like the skin
import THREE from '../threejs-shim.js';
import { noise3 } from './creatures.js';

export const FINISHES = {
    glossy:  { roughness: 0.05, sheen: 0, sheenRoughness: 0.3, grain: 0, grainRoughness: 0.2, transmission: 1 },
    sour:    { roughness: 0.3, sheen: 0.7, sheenRoughness: 0.35, grain: 1, grainRoughness: 0.2, transmission: 0.92 },
    frosted: { roughness: 0.42, sheen: 1.0, sheenRoughness: 0.55, grain: 0.4, grainRoughness: 0.6, transmission: 0.86 },
};
export const TABLES = {
    paper:  { color: '#e4e1dc' },
    mat:    { color: '#2c5a4b' },
    marble: { color: '#ebeae7' },
    night:  { color: '#101218' },
};

const lin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const smoothstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
// the looks are linear, which is what Color.setRGB takes
const colorOf = (rgb) => new THREE.Color().setRGB(...rgb);

function tableTexture(name) {
    const S = 1024, cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const g = cv.getContext('2d', { willReadFrequently: name === 'marble' });
    g.fillStyle = TABLES[name].color; g.fillRect(0, 0, S, S);
    if (name === 'mat') {
        // one tile is 5 units: a minor line every 0.5, a major every 2.5, a diagonal
        const px = S / 5;
        g.strokeStyle = 'rgba(150, 200, 160, 0.35)'; g.lineWidth = 2;
        for (let i = 0; i <= 10; i++) { const p = i * px / 2; g.beginPath(); g.moveTo(p, 0); g.lineTo(p, S); g.moveTo(0, p); g.lineTo(S, p); g.stroke(); }
        g.strokeStyle = 'rgba(150, 200, 160, 0.75)'; g.lineWidth = 4;
        for (let i = 0; i <= 2; i++) { const p = i * px * 2.5; g.beginPath(); g.moveTo(p, 0); g.lineTo(p, S); g.moveTo(0, p); g.lineTo(S, p); g.stroke(); }
        g.strokeStyle = 'rgba(150, 200, 160, 0.4)'; g.lineWidth = 3;
        g.beginPath(); g.moveTo(0, 0); g.lineTo(S, S); g.stroke();
    } else if (name === 'marble') {
        const img = g.getImageData(0, 0, S, S), d = img.data;
        const fbm = (x, y) => { let v = 0, a = 0.5; for (let o = 0; o < 5; o++) { v += a * noise3(x, y, 0.5); x = x * 2.03 + 1.7; y = y * 2.03 + 9.2; a *= 0.5; } return v; };
        for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
            const X = i / S * 5, Y = j / S * 5;
            const v = Math.abs(Math.sin(X * 1.4 + Y * 0.55 + fbm(X * 0.8, Y * 0.8) * 6));
            const vein = (1 - smoothstep(0, 0.04, v)) * 0.45 + (1 - smoothstep(0, 0.3, v)) * 0.1;
            const k = 1 - vein * 0.3, o = 4 * (j * S + i);
            d[o] *= k; d[o + 1] *= k * 0.995; d[o + 2] *= k * 0.985;
        }
        g.putImageData(img, 0, 0);
    } else if (name === 'night') {
        const r = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
        r.addColorStop(0, '#3a3d47'); r.addColorStop(1, '#101218');
        g.fillStyle = r; g.fillRect(0, 0, S, S);
    }
    const t = new THREE.CanvasTexture(cv);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 8;
    if (name !== 'night') { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(8, 8); }
    return t;
}

function geometry(pos, nrm, idx, withColor) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(pos), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(Float32Array.from(nrm), 3));
    if (withColor) g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(pos.length).fill(1), 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    return g;
}

// capsule template: rings round a unit segment, hemispherical caps at both
// ends; (cos·ring, sin·ring, t along the segment, cap offset) per vertex
function capsuleTemplate(sides, capRings) {
    const vtx = [], idx = [], rings = [];
    for (let r = capRings; r >= 1; r--) rings.push([0, -Math.sin(r / capRings * Math.PI / 2), Math.cos(r / capRings * Math.PI / 2)]);
    rings.push([0, 0, 1], [1, 0, 1]);
    for (let r = 1; r <= capRings; r++) rings.push([1, Math.sin(r / capRings * Math.PI / 2), Math.cos(r / capRings * Math.PI / 2)]);
    for (const [t, cap, rad] of rings) for (let s = 0; s < sides; s++) {
        const a = s / sides * Math.PI * 2;
        vtx.push(Math.cos(a) * rad, Math.sin(a) * rad, t, cap);
    }
    for (let r = 0; r < rings.length - 1; r++) for (let s = 0; s < sides; s++) {
        const a = r * sides + s, b = r * sides + (s + 1) % sides, c = a + sides, d = b + sides;
        idx.push(a, c, b, b, c, d);
    }
    return { vtx: new Float32Array(vtx), idx, nv: vtx.length / 4 };
}

/// Everything drawn for one creature, from what its build reported (`info`,
/// see jelly-sim.js). `sync(snapshot)` each frame carries it along.
export function createView(renderer, creature, info) {
    const scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new THREE.RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.55;

    // light from the original's direction; the shadow frustum hugs the table
    const lightDir = (() => { const d = [-0.42, 1.0, -0.55]; const l = Math.hypot(...d); return d.map((v) => v / l); })();
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(lightDir[0] * 6, 0.2 + lightDir[1] * 6, 0.1 + lightDir[2] * 6);
    sun.target.position.set(0, 0.2, 0.1);
    scene.add(sun.target);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -2.6, right: 2.6, top: 2.6, bottom: -2.6, near: 0.5, far: 14 });
    sun.shadow.radius = 6;
    sun.shadow.bias = -0.0005;
    scene.add(sun);
    const fill = new THREE.HemisphereLight(0xffffff, 0xd8d2c8, 0.9);
    scene.add(fill);

    // the table: an unlit floor in exactly the page colour, the shadows laid
    // over it on a ShadowMaterial tinted by the creature's colour
    const textures = {};
    const floorMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), floorMat);
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);
    const shadowCatcher = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShadowMaterial({ opacity: 0.8 }));
    shadowCatcher.rotation.x = -Math.PI / 2;
    shadowCatcher.position.y = 0.0005;
    shadowCatcher.receiveShadow = true;
    scene.add(shadowCatcher);

    // ── the gel: one mesh per block (the body; three slots per part) ──
    const gelMats = [];
    const gelMaterial = () => {
        const m = new THREE.MeshPhysicalMaterial({
            color: 0xffffff, roughness: 0.05, metalness: 0, ior: 1.42,
            transmission: 1, thickness: 0.4, attenuationColor: 0xff8899, attenuationDistance: 0.24,
            specularIntensity: 1, vertexColors: true, transparencyMode: 'refract',
        });
        gelMats.push(m);
        return m;
    };
    const skins = info.skins;
    const topoOf = (b) => (b.part < 0 ? 0 : b.part + 1);
    const blockMeshes = info.blocks.map((b) => {
        const s = skins[topoOf(b)];
        const m = new THREE.Mesh(geometry(s.position, s.normal, s.index, true), gelMaterial());
        m.castShadow = true;
        m.frustumCulled = false;
        m.visible = b.active;
        m.userData.emissive = -1;
        scene.add(m);
        return m;
    });

    // ── insides ──
    const milkMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.55 });
    const eyeMat = new THREE.MeshPhysicalMaterial({ color: 0x050608, roughness: 0.15, clearcoat: 1 });
    const mouthMat = new THREE.MeshStandardMaterial({ color: 0x401018, roughness: 0.6 });
    const insideMats = { milk: milkMat, eye: eyeMat, mouth: mouthMat };
    const insideMeshes = info.insides.map(({ surf, kind }) => {
        const mesh = new THREE.Mesh(geometry(surf.pos, surf.nrm, surf.idx, false), insideMats[kind]);
        mesh.frustumCulled = false;
        scene.add(mesh);
        return mesh;
    });

    // ── chains: a tapered capsule per segment ──
    const segCount = info.segCount;
    const cap = capsuleTemplate(7, 2);
    const chainPos = new Float32Array(segCount * cap.nv * 3), chainNrm = new Float32Array(segCount * cap.nv * 3);
    const chainIdx = new Uint32Array(segCount * cap.idx.length);
    for (let s = 0; s < segCount; s++) for (let k = 0; k < cap.idx.length; k++) chainIdx[s * cap.idx.length + k] = cap.idx[k] + s * cap.nv;
    const chainGeo = geometry(chainPos, chainNrm, chainIdx, true);
    const chainMat = gelMaterial();
    const chains = new THREE.Mesh(chainGeo, chainMat);
    chains.castShadow = true;
    chains.frustumCulled = false;
    if (segCount) scene.add(chains);
    function buildChains(S) {
        const V = cap.vtx, P = chainGeo.attributes.position.array, N = chainGeo.attributes.normal.array;
        let o = 0;
        for (let s = 0; s < segCount; s++) {
            const ax = S[8 * s], ay = S[8 * s + 1], az = S[8 * s + 2], ra = S[8 * s + 3];
            const bx = S[8 * s + 4], by = S[8 * s + 5], bz = S[8 * s + 6], rb = S[8 * s + 7];
            let dx = bx - ax, dy = by - ay, dz = bz - az; const L = Math.hypot(dx, dy, dz) || 1e-5; dx /= L; dy /= L; dz /= L;
            const hx = Math.abs(dy) > 0.9 ? 1 : 0, hy = Math.abs(dy) > 0.9 ? 0 : 1;
            let e1x = -dz * hy, e1y = dz * hx, e1z = dx * hy - dy * hx;
            const l1 = Math.hypot(e1x, e1y, e1z); e1x /= l1; e1y /= l1; e1z /= l1;
            const e2x = dy * e1z - dz * e1y, e2y = dz * e1x - dx * e1z, e2z = dx * e1y - dy * e1x;
            for (let v = 0; v < cap.nv; v++) {
                const cx = V[4 * v], cy = V[4 * v + 1], t = V[4 * v + 2], cw = V[4 * v + 3];
                const r = ra + (rb - ra) * t;
                const ox = e1x * cx + e2x * cy + dx * cw, oy = e1y * cx + e2y * cy + dy * cw, oz = e1z * cx + e2z * cy + dz * cw;
                const ol = Math.hypot(ox, oy, oz) || 1;
                P[o] = ax + (bx - ax) * t + ox * r; P[o + 1] = ay + (by - ay) * t + oy * r; P[o + 2] = az + (bz - az) * t + oz * r;
                N[o] = ox / ol; N[o + 1] = oy / ol; N[o + 2] = oz / ol;
                o += 3;
            }
        }
        chainGeo.attributes.position.needsUpdate = true;
        chainGeo.attributes.normal.needsUpdate = true;
    }

    // ── goo ──
    const dropMat = gelMaterial();
    const drops = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2), dropMat, 120);
    drops.castShadow = true;
    drops.frustumCulled = false;
    drops.count = 0;
    scene.add(drops);

    // ── sugar: one instanced mesh for every block's crystals; the simulation
    // carries them, each keeps its own turn and size here ──
    let seed = 11;
    const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const sugarCount = (b) => info.sugarCounts[topoOf(b)];
    const sugarTotal = info.blocks.reduce((a, b) => a + sugarCount(b), 0);
    const sugarMat = new THREE.MeshPhysicalMaterial({ color: 0xfbf8f2, roughness: 0.2, transmission: 0.35, thickness: 0.01, ior: 1.55 });
    const sugar = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), sugarMat, Math.max(1, sugarTotal));
    sugar.frustumCulled = false;
    sugar.visible = false;
    scene.add(sugar);
    const sugarRot = [], sugarSize = new Float32Array(sugarTotal);
    for (let i = 0; i < sugarTotal; i++) {
        sugarRot.push(new THREE.Quaternion().setFromEuler(new THREE.Euler(rand() * 6.3, rand() * 6.3, rand() * 6.3)));
        sugarSize[i] = 0.007 + 0.008 * rand() * rand();
    }
    const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3(), tmpP = new THREE.Vector3();
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    let grain = 0;
    function placeSugar(blocks, pos) {
        let k = 0, o = 0;
        blocks.forEach((b) => {
            const n = sugarCount(b);
            if (!b.active) { for (let i = 0; i < n; i++) sugar.setMatrixAt(k++, zero); return; }
            const sc = (b.part >= 0 ? b.scale : 1) * grain;
            for (let i = 0; i < n; i++, k++, o += 3) {
                tmpP.set(pos[o], pos[o + 1], pos[o + 2]);
                tmpS.setScalar(sugarSize[k] * sc);
                tmpM.compose(tmpP, sugarRot[k], tmpS);
                sugar.setMatrixAt(k, tmpM);
            }
        });
        sugar.instanceMatrix.needsUpdate = true;
    }

    // ── the lattice overlay ──
    const lineGeo = new THREE.BufferGeometry();
    const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ color: 0x401018, transparent: true, opacity: 0.25, depthTest: false }));
    lines.frustumCulled = false;
    lines.visible = false;
    lines.renderOrder = 10;
    scene.add(lines);
    function buildLines(L) {
        const a = lineGeo.attributes.position;
        if (a && a.array.length === L.length) { a.array.set(L); a.needsUpdate = true; }
        else lineGeo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(L), 3));
    }

    // ── looks ──
    let look = null;
    function paintSkins() {
        info.blocks.forEach((b, i) => {
            const g = blockMeshes[i].geometry, C = g.attributes.color.array, P = skins[topoOf(b)].position;
            for (let v = 0; v < P.length / 3; v++) {
                const c = creature.paint(look, P[3 * v], P[3 * v + 1], P[3 * v + 2], topoOf(b));
                C[3 * v] = c[0]; C[3 * v + 1] = c[1]; C[3 * v + 2] = c[2];
            }
            g.attributes.color.needsUpdate = true;
        });
    }
    let table = 'paper';
    function tintShadow() {
        const hex = TABLES[table].color;
        const base = [1, 3, 5].map((i) => lin(parseInt(hex.slice(i, i + 2), 16) / 255));
        shadowCatcher.material.color.copy(colorOf(base.map((c, i) => c * look.deep[i])));
    }
    function setLook(L, F) {
        look = L;
        // three.js keeps attenuationColor^(thickness/attenuationDistance) of
        // the light; the surface filters by the tint once more for the light
        // the gel scatters. A rainbow's hue is per vertex, over a near-grey
        // absorption so the thick body still goes deeper.
        const bodyAtt = L.rainbow ? [0.8, 0.76, 0.8] : L.tint, surfTint = L.rainbow ? [1, 1, 1] : L.tint;
        for (const m of gelMats) {
            const isChain = m === chainMat;
            m.attenuationColor.copy(colorOf(isChain ? L.gill : bodyAtt));
            m.attenuationDistance = 1 / (isChain ? L.gillDepth : L.depth);
            m.color.copy(colorOf(isChain ? L.gill : surfTint));
            Object.assign(m, { roughness: F.roughness, sheen: F.sheen, sheenRoughness: F.sheenRoughness, transmission: F.transmission });
            m.sheenColor.set(0xffffff);
            m.needsUpdate = true;
        }
        milkMat.color.copy(colorOf(L.milk));
        eyeMat.color.copy(L.darkEyes !== 0 ? colorOf([0.006, 0.007, 0.012]) : colorOf([0.5, 0.12, 0.14]));
        mouthMat.color.copy(colorOf(L.tint.map((c) => c * c * 0.25)));
        paintSkins();
        grain = F.grain;
        sugar.visible = F.grain > 0;
        sugarMat.roughness = F.grainRoughness;
        sugarMat.transmission = F.grain < 1 ? 0 : 0.35;
        tintShadow();
    }
    function setTable(name) {
        table = name;
        floorMat.map = textures[name] ??= tableTexture(name);
        floorMat.needsUpdate = true;
        scene.background = new THREE.Color(TABLES[name].color);
        const dark = name === 'night' || name === 'mat';
        shadowCatcher.material.opacity = dark ? 0.6 : 0.8;
        fill.intensity = name === 'night' ? 0.35 : 0.9;
        sun.intensity = name === 'night' ? 1.6 : 2.2;
        if (look) tintShadow();
    }

    // ── per frame ──
    // A block's glow: pale while a bud regrows, a soft light under the
    // pointer, and a flush that deepens as a held part nears tearing.
    function emissiveOf(b, bi, hover, held) {
        let r = 0, g = 0, bl = 0;
        if (b.part >= 0 && b.slot === 0 && b.grow < 1) { const k = 1 - b.grow; r += 0.35 * k; g += 0.29 * k; bl += 0.28 * k; }
        if (bi === hover && !held) { r += 0.06; g += 0.06; bl += 0.07; }
        if (held && held.block === bi && held.ratio > 0) { const k = Math.min(1, held.ratio) ** 2; r += 0.45 * k; g += 0.05 * k; bl += 0.08 * k; }
        return [r, g, bl];
    }
    let shown = null;
    /// carry everything along to `snap` (from the simulation); the hover and
    /// the held part's nearness to tearing light it
    function sync(snap, { hoverBlock = -1, heldBlock = null } = {}) {
        const blocks = snap.blocks;
        let o = 0;
        blocks.forEach((b, bi) => {
            const m = blockMeshes[bi];
            m.visible = b.active;
            if (b.active && snap !== shown) {
                const g = m.geometry, len = skins[topoOf(b)].position.length;
                g.attributes.position.array.set(snap.skinPos.subarray(o, o + len));
                g.attributes.normal.array.set(snap.skinNrm.subarray(o, o + len));
                g.attributes.position.needsUpdate = true;
                g.attributes.normal.needsUpdate = true;
                o += len;
            }
            if (!b.active) return;
            const e = emissiveOf(b, bi, hoverBlock, heldBlock), key = e.map((v) => v.toFixed(3)).join();
            if (m.userData.emissive !== key) { m.userData.emissive = key; m.material.emissive.setRGB(...e); }
        });
        if (snap === shown) return;
        shown = snap;
        o = 0;
        for (const mesh of insideMeshes) {
            const g = mesh.geometry, len = g.attributes.position.array.length;
            g.attributes.position.array.set(snap.insidePos.subarray(o, o + len));
            g.attributes.normal.array.set(snap.insideNrm.subarray(o, o + len));
            g.attributes.position.needsUpdate = true;
            g.attributes.normal.needsUpdate = true;
            o += len;
        }
        if (segCount) buildChains(snap.segments);
        const D = snap.drops, nd = D.length / 4;
        for (let i = 0; i < nd; i++) {
            tmpP.set(D[4 * i], D[4 * i + 1], D[4 * i + 2]);
            tmpS.setScalar(D[4 * i + 3]);
            tmpM.compose(tmpP, tmpQ.identity(), tmpS);
            drops.setMatrixAt(i, tmpM);
        }
        drops.count = nd;
        if (nd) drops.instanceMatrix.needsUpdate = true;
        if (sugar.visible && snap.sugar) placeSugar(blocks, snap.sugar);
        if (lines.visible && snap.lines) buildLines(snap.lines);
    }

    const facts = `${skins.reduce((a, s) => a + s.index.length / 3, 0).toLocaleString()} skin triangles · ${segCount} chain segments · ${sugarTotal.toLocaleString()} sugar crystals`;
    return { scene, sync, setLook, setTable, lines, sugarVisible: () => sugar.visible, facts, sun, floor, shadowCatcher };
}
