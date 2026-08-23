// Live IK + actuator plant with compact joint modules.
//
// Mechanisms sized like published hardware (not exploded toy cans):
//   gear     → onero/HEBI-class sealed Φ57 can (no rear stub)
//   hydraulic→ short vane cartridge, hoses along the link
//   tendon   → Ø28 pulley, cables along the link, winch at base
// Refs: OpenWonderLabs/RobotDoc, HEBI X5 dims, ALTO CSF+NEMA layouts.

import THREE, { initThreers } from '../threejs-shim.js';
import { WebIkServoPlant } from '../pkg/threers.js';

const statusEl = document.getElementById('status');
const errEl = document.getElementById('err');
const hudEl = document.getElementById('hud');

const SCALE = 1 / 120;
const DOWN = [0, 0, -1];
const SEED = [-2.8, 88.9, 93.9];
const BODY_STRIDE = 15;
const JOINT_STRIDE = 13;
const CONTACT_STRIDE = 7;
const PART_STRIDE = 16;
const LINK_R_SCALE = 0.55;

const ROLE = {
    motor: 0, gear: 1, flange: 2, shaft: 3, harmonic: 4,
    vane: 5, port: 6, hose: 7, reservoir: 8,
    pulley: 9, cable: 10, winch: 11, sheath: 12, bearing: 13, encoder: 14,
    housing: 15,
};

const KINDS = [
    { kind: 'direct', className: 'direct', title: 'direct drive', accent: 0x59c0f2 },
    { kind: 'qdd', className: 'qdd', title: 'QDD 15:1', accent: 0x73d98a },
    { kind: 'high', className: 'high', title: 'servo 288:1', accent: 0xf28b5a },
    { kind: 'hydraulic', className: 'hydraulic', title: 'hydraulic', accent: 0xc9a227 },
    { kind: 'tendon', className: 'tendon', title: 'tendon / cable', accent: 0xc084fc },
];

const MAT_NAMES = ['aluminum', 'steel', 'plastic', 'rubber'];

function v(p) {
    return new THREE.Vector3(p[0] * SCALE, p[1] * SCALE, p[2] * SCALE);
}

function goalAt(t) {
    const u = Math.min(1, t / 0.15);
    return { at: [280 + 140 * u, 0, 220], along: DOWN };
}

function quatAlignY(dir) {
    const len = Math.hypot(dir.x, dir.y, dir.z) || 1e-6;
    const ux = dir.x / len, uy = dir.y / len, uz = dir.z / len;
    const dot = uy;
    if (Math.abs(dot - 1) < 1e-5) return new THREE.Quaternion();
    if (Math.abs(dot + 1) < 1e-5) {
        return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
    }
    const ax = uz, ay = 0, az = -ux;
    const alen = Math.hypot(ax, ay, az) || 1;
    const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
    return new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(ax / alen, ay / alen, az / alen),
        angle,
    );
}

function placeCapsule(mesh, a, b, radiusScene) {
    const mid = a.clone().add(b).multiplyScalar(0.5);
    const d = b.clone().sub(a);
    const len = Math.max(1e-4, d.length());
    mesh.position.copy(mid);
    mesh.scale.set(radiusScene * 2, len, radiusScene * 2);
    mesh.quaternion.copy(quatAlignY(d));
}

function placePart(mesh, ox, oy, oz, ax, ay, az, radiusMm, lengthMm, shape) {
    const o = v([ox, oy, oz]);
    const axis = new THREE.Vector3(ax, ay, az).normalize();
    const half = (lengthMm * SCALE) * 0.5;
    const r = Math.max(0.004, radiusMm * SCALE);
    if (shape === 1) {
        // Segment: origin → origin + axis * length
        const end = o.clone().add(axis.multiplyScalar(lengthMm * SCALE));
        placeCapsule(mesh, o, end, r);
    } else {
        // Capsule / disk along axis, centered at origin
        placeCapsule(
            mesh,
            o.clone().add(axis.clone().multiplyScalar(-half)),
            o.clone().add(axis.clone().multiplyScalar(half)),
            r,
        );
    }
}

function matSteel(hex, metal = 0.9, rough = 0.3) {
    return new THREE.MeshStandardMaterial({ color: hex, metalness: metal, roughness: rough });
}

function makePartMesh() {
    return new THREE.Mesh(
        new THREE.CylinderGeometry(0.5, 0.5, 1, 18),
        matSteel(0x888888, 0.7, 0.4),
    );
}

function makeLinkBeam() {
    return new THREE.Mesh(
        new THREE.CylinderGeometry(0.5, 0.5, 1, 16),
        matSteel(0x88a0b8, 0.55, 0.45),
    );
}

function makeContactMarker() {
    const m = new THREE.Mesh(
        new THREE.SphereGeometry(1, 12, 10),
        new THREE.MeshStandardMaterial({
            color: 0xff5533,
            metalness: 0.1,
            roughness: 0.4,
            emissive: 0x331100,
        }),
    );
    m.visible = false;
    return m;
}

function linkEndpoints(origin, distal, clear = 0.12) {
    const d = distal.clone().sub(origin);
    const len = d.length();
    const u = d.multiplyScalar(1 / Math.max(len, 1e-6));
    const endClear = clear;
    if (len < clear + endClear + 0.05) {
        return { a: origin.clone(), b: distal.clone() };
    }
    return {
        a: origin.clone().add(u.clone().multiplyScalar(clear)),
        b: distal.clone().add(u.clone().multiplyScalar(-endClear)),
    };
}

function roleName(role) {
    return Object.keys(ROLE).find((k) => ROLE[k] === role) || `r${role}`;
}

class ArmView {
    constructor(scene, x, spec) {
        this.spec = spec;
        this.root = new THREE.Group();
        this.root.position.set(x, 0, 0);
        this.root.rotation.x = -Math.PI / 2;
        scene.add(this.root);

        this.plant = new WebIkServoPlant(spec.kind);
        this.plant.seed(new Float64Array(SEED));
        this.tipErr = 0;

        const n = this.plant.bodyCount;
        this.partMeshes = [];
        this.beams = [];
        this.ghostBeams = [];
        for (let i = 0; i < n; i++) {
            const beam = makeLinkBeam();
            const ghost = makeLinkBeam();
            ghost.material = matSteel(0xb8c4d0, 0.15, 0.7);
            this.root.add(beam, ghost);
            this.beams.push(beam);
            this.ghostBeams.push(ghost);
        }

        this.tip = new THREE.Mesh(
            new THREE.SphereGeometry(0.1, 16, 12),
            new THREE.MeshStandardMaterial({ color: spec.accent, metalness: 0.35, roughness: 0.35 }),
        );
        this.target = new THREE.Mesh(
            new THREE.SphereGeometry(0.12, 16, 12),
            new THREE.MeshStandardMaterial({ color: 0xf25932, metalness: 0.2, roughness: 0.45 }),
        );
        this.root.add(this.tip, this.target);

        this.contactMarkers = [];
        for (let i = 0; i < 8; i++) {
            const m = makeContactMarker();
            this.root.add(m);
            this.contactMarkers.push(m);
        }

        const ped = new THREE.Mesh(
            new THREE.BoxGeometry(2.6, 0.1, 2.6),
            new THREE.MeshStandardMaterial({ color: 0x1a1e28, roughness: 0.85, metalness: 0.05 }),
        );
        ped.position.set(0, 0, -0.05);
        this.root.add(ped);

        this.baseCol = new THREE.Mesh(
            new THREE.CylinderGeometry(0.22, 0.28, 0.35, 20),
            matSteel(0x3d4450, 0.7, 0.45),
        );
        this.root.add(this.baseCol);

        const wb = this.plant.worldBounds();
        const mn = v([wb[0], wb[1], wb[2]]);
        const mx = v([wb[3], wb[4], wb[5]]);
        const size = mx.clone().sub(mn);
        const center = mn.clone().add(mx).multiplyScalar(0.5);
        this.obstacle = new THREE.Mesh(
            new THREE.BoxGeometry(Math.abs(size.x), Math.abs(size.y), Math.abs(size.z)),
            new THREE.MeshStandardMaterial({ color: 0x6b5a3e, metalness: 0.1, roughness: 0.75 }),
        );
        this.obstacle.position.copy(center);
        this.root.add(this.obstacle);

        this.stats = { bodies: n, joints: n, contacts: 0, materials: [], parts: 0, bom: [] };
    }

    ensureParts(count) {
        while (this.partMeshes.length < count) {
            const m = makePartMesh();
            this.root.add(m);
            this.partMeshes.push(m);
        }
        for (let i = 0; i < this.partMeshes.length; i++) {
            this.partMeshes[i].visible = i < count;
        }
    }

    update(goal, dt) {
        this.tipErr = this.plant.step(
            goal.at[0], goal.at[1], goal.at[2],
            goal.along[0], goal.along[1], goal.along[2],
            dt,
        );

        const bodies = this.plant.bodiesFlat();
        const joints = this.plant.jointsFlat();
        const contacts = this.plant.contactsFlat();
        const skC = this.plant.skeletonCmd();
        const parts = this.plant.designPartsFlat();
        const names = this.plant.bodyNames().split('\n').filter(Boolean);
        const n = this.beams.length;
        const nParts = (parts.length / PART_STRIDE) | 0;

        this.stats.contacts = this.plant.contactCount;
        this.stats.parts = nParts;
        this.stats.materials = [];
        this.stats.bom = [];

        {
            const j0 = v([joints[0], joints[1], joints[2]]);
            this.baseCol.position.set(j0.x, j0.y, j0.z - 0.22);
            this.baseCol.quaternion.copy(quatAlignY(new THREE.Vector3(0, 0, 1)));
        }

        // --- mechanism BOM from Rust design ---
        this.ensureParts(nParts);
        const roleCounts = {};
        for (let i = 0; i < nParts; i++) {
            const o = i * PART_STRIDE;
            const role = parts[o + 1] | 0;
            const shape = parts[o + 15] | 0;
            const mesh = this.partMeshes[i];
            placePart(
                mesh,
                parts[o + 2], parts[o + 3], parts[o + 4],
                parts[o + 5], parts[o + 6], parts[o + 7],
                parts[o + 8], parts[o + 9],
                shape,
            );
            mesh.material.color.setRGB(parts[o + 10], parts[o + 11], parts[o + 12]);
            mesh.material.metalness = parts[o + 13];
            mesh.material.roughness = parts[o + 14];
            // Sheaths slightly transparent so cable shows through.
            mesh.material.transparent = role === ROLE.sheath;
            mesh.material.opacity = role === ROLE.sheath ? 0.45 : 1.0;
            const rn = roleName(role);
            roleCounts[rn] = (roleCounts[rn] || 0) + 1;
        }
        this.stats.bom = Object.entries(roleCounts).map(([k, c]) => `${c}× ${k}`);

        // --- link beams between joints ---
        for (let i = 0; i < n; i++) {
            const o = i * BODY_STRIDE;
            const origin = v([bodies[o], bodies[o + 1], bodies[o + 2]]);
            const distal = v([bodies[o + 3], bodies[o + 4], bodies[o + 5]]);
            const radiusMm = bodies[o + 6];
            const matId = bodies[o + 9] | 0;
            const j = i * JOINT_STRIDE;

            const beam = linkEndpoints(origin, distal, 0.12);
            const linkR = radiusMm * SCALE * LINK_R_SCALE;
            this.beams[i].material.color.setRGB(bodies[o + 10], bodies[o + 11], bodies[o + 12]);
            this.beams[i].material.metalness = bodies[o + 13];
            this.beams[i].material.roughness = bodies[o + 14];
            placeCapsule(this.beams[i], beam.a, beam.b, linkR);

            const c0 = v([skC[i * 3], skC[i * 3 + 1], skC[i * 3 + 2]]);
            const c1 = v([skC[(i + 1) * 3], skC[(i + 1) * 3 + 1], skC[(i + 1) * 3 + 2]]);
            const g = linkEndpoints(c0, c1, 0.12);
            placeCapsule(this.ghostBeams[i], g.a, g.b, linkR * 0.7);

            this.stats.materials.push({
                name: names[i] || `link${i + 1}`,
                mat: MAT_NAMES[matId] || '?',
                mass: bodies[o + 7],
                angle: joints[j + 6],
                torque: joints[j + 11],
            });
        }

        this.tip.position.copy(v(this.plant.tipAct()));
        this.target.position.copy(v(goal.at));

        for (let i = 0; i < this.contactMarkers.length; i++) {
            const m = this.contactMarkers[i];
            if (i * CONTACT_STRIDE + 6 < contacts.length) {
                const o = i * CONTACT_STRIDE;
                m.visible = true;
                m.position.copy(v([contacts[o], contacts[o + 1], contacts[o + 2]]));
                const s = Math.min(0.2, 0.06 + contacts[o + 6] * SCALE * 2);
                m.scale.set(s, s, s);
            } else {
                m.visible = false;
            }
        }
    }

    hudHtml() {
        const mats = this.stats.materials
            .map(
                (m) =>
                    `J/${m.name} · ${m.mat} · ${m.mass.toFixed(2)}kg · ∠${m.angle.toFixed(1)}° · τ ${m.torque.toFixed(2)}`,
            )
            .join('<br>');
        const pairs = this.plant.contactPairs();
        const hit = pairs
            ? pairs.split('\n').filter(Boolean).map((p) => p.replace('|', '↔')).join(', ')
            : 'none';
        const matName = this.plant.materialName();
        const temp = this.plant.tempC().toFixed(0);
        let extra = `${matName} · ${temp} °C`;
        if (this.spec.kind === 'hydraulic') {
            extra += ` · ν=${this.plant.fluidNuCst().toFixed(1)} cSt`;
        } else if (this.spec.kind === 'tendon') {
            const ten = this.plant.tensions0();
            if (ten.length >= 3) {
                extra += ` · T± ${ten[0].toFixed(0)}/${ten[1].toFixed(0)} N · T0 ${ten[2].toFixed(0)}`;
            }
        } else if (this.spec.kind === 'direct' || this.spec.kind === 'qdd' || this.spec.kind === 'high') {
            extra = `N=${this.plant.ratio().toFixed(0)} · ${extra}`;
        }
        const bom = this.stats.bom.slice(0, 8).join(', ');
        return `<div class="card ${this.spec.className}">
            <strong>${this.spec.title}</strong>
            <div>${this.plant.designLabel()}</div>
            <div class="muted">${extra}</div>
            <div>BOM <b>${this.stats.parts}</b> parts · tip <b>${this.tipErr.toFixed(1)} mm</b> · ${this.plant.meanJointErr().toFixed(2)}°</div>
            <div class="muted">${bom}</div>
            <div class="muted">${mats}</div>
            <div class="muted">hits: ${hit}</div>
        </div>`;
    }
}

async function main() {
    await initThreers({ module_or_path: new URL('../pkg/threers_bg.wasm', import.meta.url) });
    statusEl.textContent = 'wasm ready';

    const canvas = document.getElementById('c');
    const renderer = await THREE.WebGLRenderer.create(canvas);
    renderer.setSize(canvas.width, canvas.height, false);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1018);
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const key = new THREE.DirectionalLight(0xfff4e6, 2.2);
    key.position.set(-0.4, 0.85, 0.35);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x7390e8, 0.55);
    fill.position.set(0.7, 0.1, 0.6);
    scene.add(fill);

    const camera = new THREE.PerspectiveCamera(42, canvas.width / canvas.height, 0.1, 100);
    camera.position.set(0, 3.2, 14);
    camera.lookAt(0, 2.0, 1);
    const controls = new THREE.OrbitControls(camera, canvas);
    if (typeof controls._w?.setTarget === 'function') controls._w.setTarget(0, 2.0, 1);

    const span = 3.4;
    const origin = -0.5 * (KINDS.length - 1) * span;
    const arms = KINDS.map((spec, i) => new ArmView(scene, origin + i * span, spec));

    const tempEl = document.getElementById('temp');
    const tempVal = document.getElementById('tempVal');
    const fluidEl = document.getElementById('fluid');
    const tendonEl = document.getElementById('tendon');

    function applyEnv() {
        const temp = Number(tempEl.value);
        tempVal.textContent = String(temp);
        for (const a of arms) {
            a.plant.setTemp(temp);
            if (a.spec.kind === 'hydraulic') a.plant.setFluid(fluidEl.value);
            if (a.spec.kind === 'tendon') a.plant.setTendon(tendonEl.value);
        }
    }
    tempEl.addEventListener('input', applyEnv);
    fluidEl.addEventListener('change', applyEnv);
    tendonEl.addEventListener('change', applyEnv);
    applyEnv();

    let t = 0;
    let last = performance.now();
    const LOOP = 1.1;

    function frame(now) {
        const dtWall = Math.min(0.05, (now - last) / 1000);
        last = now;
        const simDt = 1 / 240;
        let budget = dtWall;
        while (budget > 1e-6) {
            const h = Math.min(simDt, budget);
            t += h;
            if (t > LOOP) {
                t = 0;
                for (const a of arms) a.plant.seed(new Float64Array(SEED));
            }
            const goal = goalAt(t);
            for (const a of arms) a.update(goal, h);
            budget -= h;
        }
        hudEl.innerHTML = arms.map((a) => a.hudHtml()).join('');
        const high = arms.find((a) => a.spec.kind === 'high') || arms[2];
        const parts = arms.reduce((n, a) => n + a.stats.parts, 0);
        statusEl.textContent = `t=${t.toFixed(2)}s · ${tempEl.value} °C · ${parts} mech parts · high tip ${high.tipErr.toFixed(1)} mm`;
        controls.update();
        renderer.render(scene, camera);
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch((e) => {
    errEl.textContent = e && e.stack ? e.stack : String(e);
    statusEl.textContent = 'failed';
});
