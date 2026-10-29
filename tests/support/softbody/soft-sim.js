// Ported from Axolotl Jelly (https://github.com/saulfloresjr/axolotl-jelly),
// unchanged apart from the module export at the end.
//
// MIT License
//
// Copyright (c) 2026 Dream Meridian
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of the experiments in this folder (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies of the
// Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ─────────────────────────────────────────────────────────────────────────────
// sim.js — XPBD for a body made of parts that can come apart and grow back.
//
// All particles live in one big array. It is cut into *blocks*: the body,
// and for each leg one attached slot plus two "loose" slots for torn-off legs.
// A block shares its topology (tets, edges, rest lengths) with every other
// block of the same part; only its offset into the particle array differs.
//
//   glue      distance constraints from leg root nodes to nearby body nodes
//   tearing   when a held leg's glue is stretched past a limit, the leg's
//             particles are copied into a loose slot and the glue lets go
//   regrowth  the attached slot is reset as a tiny bud at the socket and its
//             rest lengths and volumes are scaled from 5 % back up to 100 %
//   muscles   body edges change rest length in a travelling wave: the tail sways
//   gills     particle chains pulled toward their rest shape (rotated with the
//             head), so they hold their form but flutter as the body moves
// ─────────────────────────────────────────────────────────────────────────────
class AxoSim {
  constructor(geo) {
    this.geo = geo;
    const mkTopo = (p) => {
      const n = p.nodes.length / 3, mass = new Float32Array(n), nodeFw = new Float32Array(n);
      for (let t = 0; t < p.restVol.length; t++) for (let c = 0; c < 4; c++) { mass[p.tets[4*t+c]] += p.restVol[t] / 4; nodeFw[p.tets[4*t+c]] += p.restVol[t]; }
      const restLen = new Float32Array(p.edges.length / 2);
      for (let e = 0; e < restLen.length; e++) {
        const a = 3 * p.edges[2*e], b = 3 * p.edges[2*e+1];
        restLen[e] = Math.hypot(p.nodes[a]-p.nodes[b], p.nodes[a+1]-p.nodes[b+1], p.nodes[a+2]-p.nodes[b+2]);
      }
      return { n, rest: p.nodes, tets: p.tets, edges: p.edges, restLen, restVol: p.restVol, DmInv: p.DmInv,
               w: mass.map(m => (m > 0 ? 1 / m : 0)), nodeFw, boundary: p.boundary, radius: p.radius,
               totalVol: p.restVol.reduce((a, b) => a + b, 0), mod: new Float32Array(restLen.length) };
    };
    this.bodyTopo = mkTopo(geo.body);
    this.legTopo = geo.legs.map(mkTopo);

    // blocks
    this.blocks = [];
    let off = 0;
    const add = (topo, extra) => { const b = { topo, off, active: true, scale: 1, grow: 1, twitch: 0, born: 0, ...extra }; this.blocks.push(b); off += topo.n; return b; };
    this.body = add(this.bodyTopo, { leg: -1, slot: 0 });
    this.legSlots = geo.legs.map((_, i) => [0, 1, 2].map(slot => add(this.legTopo[i], { leg: i, slot, active: slot === 0 })));
    this.N = off;

    this.x = new Float32Array(this.N * 3); this.prev = new Float32Array(this.N * 3); this.v = new Float32Array(this.N * 3);
    this.w = new Float32Array(this.N); this.rad = new Float32Array(this.N);
    this.nodeF = new Float32Array(this.N * 12);
    for (const b of this.blocks) { this.w.set(b.topo.w, b.off); this.rad.set(b.topo.radius, b.off); }

    // body muscles: lengthwise edges on each flank, weighted toward the tail
    const T = this.bodyTopo; this.muscle = new Float32Array(T.restLen.length); this.muscleX = new Float32Array(T.restLen.length);
    for (let e = 0; e < T.restLen.length; e++) {
      const a = 3 * T.edges[2*e], b = 3 * T.edges[2*e+1], R = T.rest;
      const mx = (R[a] + R[b]) / 2, mz = (R[a+2] + R[b+2]) / 2, ax = Math.abs(R[a] - R[b]) / T.restLen[e];
      const tail = Math.min(1, Math.max(0, (0.25 - mx) / 0.9));
      this.muscle[e] = Math.sign(mz) * Math.min(1, Math.abs(mz) / 0.06) * ax * tail;
      this.muscleX[e] = mx;
    }
    // loose-leg twitch: phase by distance from the stump
    this.legPhase = geo.legs.map((L, i) => {
      const t = this.legTopo[i], ph = new Float32Array(t.restLen.length);
      for (let e = 0; e < ph.length; e++) {
        const a = 3 * t.edges[2*e];
        ph[e] = Math.hypot(t.rest[a] - L.pivot[0], t.rest[a+1] - L.pivot[1], t.rest[a+2] - L.pivot[2]) * 30;
      }
      return ph;
    });

    // gills
    const G = geo.gills;
    this.gn = G.rest.length / 3;
    this.gx = new Float32Array(G.rest); this.gprev = new Float32Array(G.rest); this.gv = new Float32Array(this.gn * 3);
    this.gRestLen = new Float32Array(this.gn);
    for (let i = 0; i < this.gn; i++) {
      const p = G.parent[i];
      if (p >= 0) this.gRestLen[i] = Math.hypot(G.rest[3*i]-G.rest[3*p], G.rest[3*i+1]-G.rest[3*p+1], G.rest[3*i+2]-G.rest[3*p+2]);
    }
    this.headRest = this.frameOf(geo.body.nodes, 0);
    this.R = [1,0,0, 0,1,0, 0,0,1];

    // drops (goo flung off when a leg tears; you can throw them too)
    this.maxDrops = 120;
    this.dp = new Float32Array(this.maxDrops * 3); this.dprev = new Float32Array(this.maxDrops * 3);
    this.dv = new Float32Array(this.maxDrops * 3); this.dr = new Float32Array(this.maxDrops);
    this.dCount = 0; this.dNext = 0;
    this.dTouch = new Uint8Array(this.maxDrops);   // bit 1: on the jelly, bit 2: against another drop

    this.params = { firmness: 0.5, damping: 0.4, regrow: 0.5, substeps: 12, gravity: 9.8 };
    this.time = 0;
    this.startle = 0;
    this.grab = null; this.dropGrab = null;
    this.events = [];
    this.regrown = 0;
    this.hash = new SpatialHash(0.12, this.N);
    this.activeIdx = new Int32Array(this.N);
    this.stats = { solveMs: 0 };
    this.reset();
  }

  reset() {
    for (const b of this.blocks) {
      b.active = b.slot === 0; b.scale = 1; b.grow = 1; b.twitch = 0;
      this.x.set(b.topo.rest, 3 * b.off);
    }
    this.prev.set(this.x); this.v.fill(0);
    this.gx.set(this.geo.gills.rest); this.gprev.set(this.gx); this.gv.fill(0);
    this.dCount = 0; this.dNext = 0; this.grab = null; this.dropGrab = null; this.regrown = 0;
    this.updateNodeF();
  }

  get edgeCompliance() { return Math.pow(10, 0.6 - 1.6 * this.params.firmness); }
  get dampRate() { return 0.4 + 22 * Math.pow(this.params.damping, 1.6); }
  get growSeconds() { return 14 - 11 * this.params.regrow; }

  // ── frame ─────────────────────────────────────────────────────────────────
  step(frameDt) {
    const t0 = performance.now();
    const S = this.params.substeps, dt = frameDt / S;
    this.time += frameDt;
    this.impact = 0;
    this.startle = Math.max(0, this.startle - frameDt / 1.4);
    this.updateGrowth(frameDt);
    this.updateMuscles();
    this.headFrame();
    let na = 0;
    for (const b of this.blocks) if (b.active) for (let i = 0; i < b.topo.n; i++) this.activeIdx[na++] = b.off + i;
    this.nActive = na;
    this.hash.build(this.x, this.activeIdx, na);
    for (let s = 0; s < S; s++) this.substep(dt);
    this.checkTear();
    this.updateNodeF();
    this.stats.solveMs = performance.now() - t0;
  }

  substep(dt) {
    const { x, prev, v, w } = this, g = this.params.gravity;
    for (const b of this.blocks) {
      if (!b.active) continue;
      for (let i = b.off, e = b.off + b.topo.n; i < e; i++) {
        v[3*i+1] -= g * dt;
        prev[3*i] = x[3*i]; prev[3*i+1] = x[3*i+1]; prev[3*i+2] = x[3*i+2];
        x[3*i] += v[3*i] * dt; x[3*i+1] += v[3*i+1] * dt; x[3*i+2] += v[3*i+2] * dt;
      }
    }
    this.solveGrab();
    const alpha = this.edgeCompliance / (dt * dt);
    for (const b of this.blocks) if (b.active) { this.solveEdges(b, b === this.body ? alpha : alpha * 0.3); this.solveVolumes(b, 1e-8 / (dt * dt)); }
    for (let i = 0; i < this.legSlots.length; i++) this.solveGlue(i, alpha * 3);
    this.solveFloor(dt);
    this.solveWalls(this.activeIdx, this.nActive);
    const inv = 1 / dt, k = Math.exp(-this.dampRate * dt);
    for (const b of this.blocks) {
      if (!b.active) continue;
      let mx = 0, my = 0, mz = 0;
      const s = b.off, e = b.off + b.topo.n;
      for (let i = s; i < e; i++) {
        v[3*i] = (x[3*i] - prev[3*i]) * inv; v[3*i+1] = (x[3*i+1] - prev[3*i+1]) * inv; v[3*i+2] = (x[3*i+2] - prev[3*i+2]) * inv;
        mx += v[3*i]; my += v[3*i+1]; mz += v[3*i+2];
      }
      const n = b.topo.n; mx /= n; my /= n; mz /= n;
      for (let i = s; i < e; i++) { v[3*i] = mx + (v[3*i] - mx) * k; v[3*i+1] = my + (v[3*i+1] - my) * k; v[3*i+2] = mz + (v[3*i+2] - mz) * k; }
    }
    this.substepGills(dt);
    this.substepDrops(dt);
  }

  solveEdges(b, alpha) {
    const { x, w } = this, T = b.topo, E = T.edges, L = T.restLen, M = T.mod, o = b.off, s = b.scale;
    for (let e = 0, m = L.length; e < m; e++) {
      const a = o + E[2*e], c = o + E[2*e+1];
      const wa = w[a], wc = w[c], ws = wa + wc;
      const dx = x[3*a]-x[3*c], dy = x[3*a+1]-x[3*c+1], dz = x[3*a+2]-x[3*c+2];
      const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (len === 0 || ws === 0) continue;
      const rest = L[e] * s * (b === this.body ? 1 + M[e] : (b.twitch ? 1 + b.twitch * Math.sin(this.time * 17 + this.legPhase[b.leg][e]) * 0.22 : 1));
      const k = -(len - rest) / (ws + alpha) / len;
      x[3*a] += dx * k * wa; x[3*a+1] += dy * k * wa; x[3*a+2] += dz * k * wa;
      x[3*c] -= dx * k * wc; x[3*c+1] -= dy * k * wc; x[3*c+2] -= dz * k * wc;
    }
  }

  solveVolumes(b, alpha) {
    const { x, w } = this, T = b.topo, tets = T.tets, RV = T.restVol, o = b.off, s3 = b.scale ** 3;
    const g = this._g || (this._g = new Float32Array(12));
    for (let t = 0, m = RV.length; t < m; t++) {
      const n0 = o + tets[4*t], n1 = o + tets[4*t+1], n2 = o + tets[4*t+2], n3 = o + tets[4*t+3];
      const i0 = 3*n0, i1 = 3*n1, i2 = 3*n2, i3 = 3*n3;
      const ax = x[i1]-x[i0], ay = x[i1+1]-x[i0+1], az = x[i1+2]-x[i0+2];
      const bx = x[i2]-x[i0], by = x[i2+1]-x[i0+1], bz = x[i2+2]-x[i0+2];
      const cx = x[i3]-x[i0], cy = x[i3+1]-x[i0+1], cz = x[i3+2]-x[i0+2];
      g[3] = (by*cz-bz*cy)/6; g[4] = (bz*cx-bx*cz)/6; g[5] = (bx*cy-by*cx)/6;
      g[6] = (cy*az-cz*ay)/6; g[7] = (cz*ax-cx*az)/6; g[8] = (cx*ay-cy*ax)/6;
      g[9] = (ay*bz-az*by)/6; g[10] = (az*bx-ax*bz)/6; g[11] = (ax*by-ay*bx)/6;
      g[0] = -(g[3]+g[6]+g[9]); g[1] = -(g[4]+g[7]+g[10]); g[2] = -(g[5]+g[8]+g[11]);
      const vol = ax*g[3] + ay*g[4] + az*g[5];
      const w0 = w[n0], w1 = w[n1], w2 = w[n2], w3 = w[n3];
      const ws = w0*(g[0]*g[0]+g[1]*g[1]+g[2]*g[2]) + w1*(g[3]*g[3]+g[4]*g[4]+g[5]*g[5]) + w2*(g[6]*g[6]+g[7]*g[7]+g[8]*g[8]) + w3*(g[9]*g[9]+g[10]*g[10]+g[11]*g[11]);
      if (ws === 0) continue;
      const k = -(vol - RV[t] * s3) / (ws + alpha);
      x[i0] += g[0]*k*w0; x[i0+1] += g[1]*k*w0; x[i0+2] += g[2]*k*w0;
      x[i1] += g[3]*k*w1; x[i1+1] += g[4]*k*w1; x[i1+2] += g[5]*k*w1;
      x[i2] += g[6]*k*w2; x[i2+1] += g[7]*k*w2; x[i2+2] += g[8]*k*w2;
      x[i3] += g[9]*k*w3; x[i3+1] += g[10]*k*w3; x[i3+2] += g[11]*k*w3;
    }
  }

  // Glue rest lengths follow the bud as it grows: the leg's rest shape is
  // scaled about its pivot, the body end stays put.
  glueRest(i, g, s) {
    const L = this.geo.legs[i], LR = this.legTopo[i].rest, BR = this.bodyTopo.rest;
    const a = 3 * L.glue.leg[g], b = 3 * L.glue.body[g], P = L.pivot;
    return Math.hypot(P[0] + s * (LR[a] - P[0]) - BR[b], P[1] + s * (LR[a+1] - P[1]) - BR[b+1], P[2] + s * (LR[a+2] - P[2]) - BR[b+2]);
  }
  solveGlue(i, alpha) {
    const B = this.legSlots[i][0], L = this.geo.legs[i], { x, w } = this;
    const GL = L.glue.leg, GB = L.glue.body, s = B.scale;
    const rest = B._glueRest && B._glueScale === s ? B._glueRest : (B._glueRest = Float32Array.from(GL, (_, g) => this.glueRest(i, g, s)));
    B._glueScale = s;
    let stretch = 0;
    for (let g = 0; g < GL.length; g++) {
      const a = B.off + GL[g], c = GB[g], wa = w[a], wc = w[c], ws = wa + wc;
      const dx = x[3*a]-x[3*c], dy = x[3*a+1]-x[3*c+1], dz = x[3*a+2]-x[3*c+2];
      const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (len === 0) continue;
      stretch = Math.max(stretch, len - rest[g]);
      const k = -(len - rest[g]) / (ws + alpha) / len;
      x[3*a] += dx * k * wa; x[3*a+1] += dy * k * wa; x[3*a+2] += dz * k * wa;
      x[3*c] -= dx * k * wc; x[3*c+1] -= dy * k * wc; x[3*c+2] -= dz * k * wc;
    }
    B.strain = stretch;
  }

  solveFloor(dt) {
    const { x, prev } = this;
    for (let q = 0; q < this.nActive; q++) {
      const i = this.activeIdx[q];
      if (x[3*i+1] >= 0) continue;
      // how fast it came down, if it was airborne a moment ago (for the thud)
      if (prev[3*i+1] > 0.004) this.impact = Math.max(this.impact, (prev[3*i+1] - x[3*i+1]) / dt);
      const pen = -x[3*i+1];
      x[3*i+1] = 0;
      const tx = x[3*i] - prev[3*i], tz = x[3*i+2] - prev[3*i+2], tl = Math.hypot(tx, tz);
      if (tl > 0) { const k = Math.min(1, 0.7 * pen / tl + 0.35); x[3*i] -= tx * k; x[3*i+2] -= tz * k; }
    }
  }

  // Soft walls at the edges of the screen: four planes from the camera, set
  // each frame by the page. A node outside is eased back in a little each
  // substep (moving prev half as far), so it bumps rather than ricochets.
  solveWalls(idx, count) {
    const W = this.walls; if (!W) return;
    const { x, prev } = this;
    for (let q = 0; q < count; q++) {
      const i = idx ? idx[q] : q;
      for (let k = 0; k < W.length; k += 4) {
        const s = W[k]*x[3*i] + W[k+1]*x[3*i+1] + W[k+2]*x[3*i+2] + W[k+3];
        if (s >= 0) continue;
        const m = Math.min(-s, 0.006);
        for (let c = 0; c < 3; c++) { x[3*i+c] += W[k+c] * m; prev[3*i+c] += W[k+c] * m * 0.5; }
      }
    }
  }
  wallsDrop(i) {
    const W = this.walls; if (!W) return;
    const { dp, dprev } = this, r = this.dr[i];
    for (let k = 0; k < W.length; k += 4) {
      const s = W[k]*dp[3*i] + W[k+1]*dp[3*i+1] + W[k+2]*dp[3*i+2] + W[k+3] - r;
      if (s >= 0) continue;
      for (let c = 0; c < 3; c++) { dp[3*i+c] -= W[k+c] * s; dprev[3*i+c] -= W[k+c] * s * 0.6; }
    }
  }

  // ── muscles: a slow idle sway, and a fast flick when startled ─────────────
  updateMuscles() {
    const M = this.bodyTopo.mod, m = this.muscle, X = this.muscleX;
    const st = this.startle * this.startle;
    const ampIdle = 0.03 * (1 - st), ampFlick = 0.2 * st;
    const t = this.time;
    for (let e = 0; e < M.length; e++) {
      if (m[e] === 0) { M[e] = 0; continue; }
      M[e] = m[e] * (ampIdle * Math.sin(t * 1.7 + X[e] * 3.2) + ampFlick * Math.sin(t * 13 + X[e] * 4.5));
    }
  }

  // ── grabbing ──────────────────────────────────────────────────────────────
  startGrab(hit, block) {
    const radius = block === this.body ? 0.2 : 0.16;
    const nodes = [], weights = [], offsets = [];
    for (let i = block.off; i < block.off + block.topo.n; i++) {
      const dx = this.x[3*i]-hit[0], dy = this.x[3*i+1]-hit[1], dz = this.x[3*i+2]-hit[2];
      const d = Math.hypot(dx, dy, dz);
      if (d > radius) continue;
      const f = 1 - d / radius;
      nodes.push(i); weights.push(f * f); offsets.push(dx, dy, dz);
    }
    this.grab = { block, nodes, weights, offsets, target: hit.slice(), angle: 0, axis: [0, 0, 1] };
    if (block === this.body) this.poke();
  }
  solveGrab() {
    const G = this.grab; if (!G) return;
    const x = this.x, c = Math.cos(G.angle), s = Math.sin(G.angle), [ax, ay, az] = G.axis;
    for (let k = 0; k < G.nodes.length; k++) {
      const i = G.nodes[k], ox = G.offsets[3*k], oy = G.offsets[3*k+1], oz = G.offsets[3*k+2];
      const d = ax*ox + ay*oy + az*oz;
      const rx = ox*c + (ay*oz - az*oy)*s + ax*d*(1-c), ry = oy*c + (az*ox - ax*oz)*s + ay*d*(1-c), rz = oz*c + (ax*oy - ay*ox)*s + az*d*(1-c);
      const k2 = 0.5 * G.weights[k];
      x[3*i] += (G.target[0] + rx - x[3*i]) * k2; x[3*i+1] += (G.target[1] + ry - x[3*i+1]) * k2; x[3*i+2] += (G.target[2] + rz - x[3*i+2]) * k2;
    }
  }
  poke() { this.startle = Math.max(this.startle, 0.55); }
  flick() { this.startle = 1; }

  // ── tearing and regrowth ──────────────────────────────────────────────────
  checkTear() {
    const G = this.grab;
    if (!G || G.block.leg < 0 || G.block.slot !== 0 || G.block.grow < 1) return;
    if (G.block.strain > 0.07) this.tear(G.block.leg);
  }
  tear(i) {
    const A = this.legSlots[i][0], n = A.topo.n;
    const free = this.legSlots[i].slice(1).sort((a, b) => (a.active - b.active) || (a.born - b.born))[0];
    for (const arr of [this.x, this.prev, this.v]) arr.copyWithin(3 * free.off, 3 * A.off, 3 * (A.off + n));
    Object.assign(free, { active: true, scale: 1, grow: 1, twitch: 1, born: this.time });
    if (this.grab && this.grab.block === A) {
      this.grab.nodes = this.grab.nodes.map(g => g - A.off + free.off);
      this.grab.block = free;
    }
    // a burst of goo from the stump
    const p = this.pivotNow(i);
    for (let k = 0; k < 14; k++) {
      const a = Math.random() * Math.PI * 2, u = Math.random();
      this.addDrop([p[0] + Math.cos(a) * 0.04, p[1] + 0.06 + u * 0.05, p[2] + Math.sin(a) * 0.04],
                   [Math.cos(a) * (0.6 + u), 1.4 + Math.random() * 1.6, Math.sin(a) * (0.6 + u)], 0.014 + Math.random() * 0.02);
    }
    A.grow = 0; A.scale = 0.05;
    this.placeBud(i);
    this.poke();
    this.events.push({ type: 'tear', leg: i });
  }
  // rotation of the body near the socket, from the deformation gradient
  socketRotation(i) {
    const a = this.geo.legs[i].anchor, F = this.nodeF, o = 12 * a;
    let c0 = [F[o], F[o+1], F[o+2]], c1 = [F[o+4], F[o+5], F[o+6]];
    const l0 = Math.hypot(...c0) || 1; c0 = c0.map(v => v / l0);
    const d = c0[0]*c1[0] + c0[1]*c1[1] + c0[2]*c1[2];
    c1 = [c1[0]-c0[0]*d, c1[1]-c0[1]*d, c1[2]-c0[2]*d];
    const l1 = Math.hypot(...c1) || 1; c1 = c1.map(v => v / l1);
    const c2 = [c0[1]*c1[2]-c0[2]*c1[1], c0[2]*c1[0]-c0[0]*c1[2], c0[0]*c1[1]-c0[1]*c1[0]];
    return [...c0, ...c1, ...c2];   // column-major
  }
  pivotNow(i) {
    const L = this.geo.legs[i], a = L.anchor, R = this.socketRotation(i), BR = this.bodyTopo.rest;
    const r = [L.pivot[0] - BR[3*a], L.pivot[1] - BR[3*a+1], L.pivot[2] - BR[3*a+2]];
    return [this.x[3*a] + R[0]*r[0] + R[3]*r[1] + R[6]*r[2], this.x[3*a+1] + R[1]*r[0] + R[4]*r[1] + R[7]*r[2], this.x[3*a+2] + R[2]*r[0] + R[5]*r[1] + R[8]*r[2]];
  }
  placeBud(i) {
    const A = this.legSlots[i][0], L = this.geo.legs[i], R = this.socketRotation(i), P = this.pivotNow(i), rest = A.topo.rest, s = A.scale;
    for (let k = 0; k < A.topo.n; k++) {
      const r = [(rest[3*k] - L.pivot[0]) * s, (rest[3*k+1] - L.pivot[1]) * s, (rest[3*k+2] - L.pivot[2]) * s];
      const j = 3 * (A.off + k);
      this.x[j] = P[0] + R[0]*r[0] + R[3]*r[1] + R[6]*r[2];
      this.x[j+1] = P[1] + R[1]*r[0] + R[4]*r[1] + R[7]*r[2];
      this.x[j+2] = P[2] + R[2]*r[0] + R[5]*r[1] + R[8]*r[2];
    }
    this.prev.copyWithin(3 * A.off, 3 * A.off, 3 * (A.off + A.topo.n));
    this.prev.set(this.x.subarray(3 * A.off, 3 * (A.off + A.topo.n)), 3 * A.off);
    this.v.fill(0, 3 * A.off, 3 * (A.off + A.topo.n));
  }
  updateGrowth(dt) {
    for (const slots of this.legSlots) {
      const A = slots[0];
      if (A.grow < 1) {
        A.grow = Math.min(1, A.grow + dt / this.growSeconds);
        // ease out with a little overshoot, like a bud swelling then settling
        const t = A.grow - 1, c1 = 1.5, c3 = c1 + 1;
        A.scale = A.grow >= 1 ? 1 : 0.05 + 0.95 * (1 + c3 * t * t * t + c1 * t * t);
        if (A.grow >= 1) { this.regrown++; this.events.push({ type: 'regrown', leg: A.leg }); }
      }
      for (const L of slots.slice(1)) if (L.active) {
        const age = this.time - L.born;
        L.twitch = Math.max(0, 1 - age / 9) * (0.55 + 0.45 * Math.sin(age * 2.1));
      }
    }
  }

  // ── gills ─────────────────────────────────────────────────────────────────
  frameOf(P, off) {
    const [a, b, c] = this.geo.landmarks.head.map(i => 3 * (off + i));
    let e1 = [P[b]-P[a], P[b+1]-P[a+1], P[b+2]-P[a+2]]; const l1 = Math.hypot(...e1); e1 = e1.map(v => v / l1);
    let e2 = [P[c]-P[a], P[c+1]-P[a+1], P[c+2]-P[a+2]]; const d = e1[0]*e2[0] + e1[1]*e2[1] + e1[2]*e2[2];
    e2 = [e2[0]-e1[0]*d, e2[1]-e1[1]*d, e2[2]-e1[2]*d]; const l2 = Math.hypot(...e2); e2 = e2.map(v => v / l2);
    const e3 = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
    return [e1, e2, e3];
  }
  headFrame() {
    const now = this.frameOf(this.x, 0), rest = this.headRest, R = this.R;
    // R = now · restᵀ (row-major 3×3)
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
      R[3*r+c] = now[0][r]*rest[0][c] + now[1][r]*rest[1][c] + now[2][r]*rest[2][c];
  }
  substepGills(dt) {
    const G = this.geo.gills, { gx, gprev, gv } = this, x = this.x, R = this.R;
    const T = this.bodyTopo.tets, E = G.rootEmb;
    // roots ride on the head
    for (let r = 0; r < G.roots.length; r++) {
      const i = G.roots[r], t = E.tetOf[r], b1 = E.bary[3*r], b2 = E.bary[3*r+1], b3 = E.bary[3*r+2], b0 = 1 - b1 - b2 - b3;
      for (let c = 0; c < 3; c++) gx[3*i+c] = b0*x[3*T[4*t]+c] + b1*x[3*T[4*t+1]+c] + b2*x[3*T[4*t+2]+c] + b3*x[3*T[4*t+3]+c];
    }
    const t = this.time, breath = 1 + 0.07 * Math.sin(t * 2.2) + 0.12 * this.startle;
    for (let i = 0; i < this.gn; i++) {
      if (G.parent[i] < 0) continue;
      gv[3*i+1] -= this.params.gravity * 0.3 * dt;
      gv[3*i] += Math.sin(t * 3.1 + i * 0.7) * 0.6 * dt; gv[3*i+2] += Math.cos(t * 2.6 + i * 1.3) * 0.6 * dt;
      gprev[3*i] = gx[3*i]; gprev[3*i+1] = gx[3*i+1]; gprev[3*i+2] = gx[3*i+2];
      gx[3*i] += gv[3*i] * dt; gx[3*i+1] += gv[3*i+1] * dt; gx[3*i+2] += gv[3*i+2] * dt;
    }
    for (let i = 0; i < this.gn; i++) {     // parents come before children in the array
      const p = G.parent[i]; if (p < 0) continue;
      const a = G.anchor[i], S = G.rest;
      // soft pull toward the rest pose, carried round by the head's rotation
      const rx = (S[3*i] - S[3*a]) * breath, ry = (S[3*i+1] - S[3*a+1]) * breath, rz = (S[3*i+2] - S[3*a+2]) * breath;
      const k = G.stiff[i];
      gx[3*i]   += (gx[3*a]   + R[0]*rx + R[1]*ry + R[2]*rz - gx[3*i])   * k;
      gx[3*i+1] += (gx[3*a+1] + R[3]*rx + R[4]*ry + R[5]*rz - gx[3*i+1]) * k;
      gx[3*i+2] += (gx[3*a+2] + R[6]*rx + R[7]*ry + R[8]*rz - gx[3*i+2]) * k;
      // inextensible link to the parent (only the child moves)
      const dx = gx[3*i]-gx[3*p], dy = gx[3*i+1]-gx[3*p+1], dz = gx[3*i+2]-gx[3*p+2];
      const len = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1, f = this.gRestLen[i] * breath / len;
      gx[3*i] = gx[3*p] + dx * f; gx[3*i+1] = gx[3*p+1] + dy * f; gx[3*i+2] = gx[3*p+2] + dz * f;
      if (gx[3*i+1] < G.radius[i]) gx[3*i+1] = G.radius[i];
    }
    const inv = 1 / dt, kd = Math.exp(-3 * dt);
    for (let i = 0; i < this.gn; i++) for (let c = 0; c < 3; c++) gv[3*i+c] = (gx[3*i+c] - gprev[3*i+c]) * inv * kd;
  }

  // ── drops ─────────────────────────────────────────────────────────────────
  addDrop(p, vel, r) {
    const i = this.dNext;
    this.dNext = (this.dNext + 1) % this.maxDrops; this.dCount = Math.min(this.dCount + 1, this.maxDrops);
    this.dp.set(p, 3*i); this.dprev.set(p, 3*i); this.dv.set(vel, 3*i); this.dr[i] = r;
    if (this.dropGrab && this.dropGrab.index === i) this.dropGrab = null;
  }
  substepDrops(dt) {
    const { dp, dprev, dv, dr, x, w } = this, g = this.params.gravity, m = this.dCount;
    for (let i = 0; i < m; i++) {
      dv[3*i+1] -= g * dt;
      dprev[3*i] = dp[3*i]; dprev[3*i+1] = dp[3*i+1]; dprev[3*i+2] = dp[3*i+2];
      dp[3*i] += dv[3*i]*dt; dp[3*i+1] += dv[3*i+1]*dt; dp[3*i+2] += dv[3*i+2]*dt;
    }
    if (this.dropGrab) { const i = this.dropGrab.index, T = this.dropGrab.target; for (let c = 0; c < 3; c++) dp[3*i+c] += (T[c] - dp[3*i+c]) * 0.35; }
    const touch = this._touch || (this._touch = new Uint8Array(this.maxDrops));
    touch.fill(0);
    for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) {
      const dx = dp[3*i]-dp[3*j], dy = dp[3*i+1]-dp[3*j+1], dz = dp[3*i+2]-dp[3*j+2], R = dr[i] + dr[j], d2 = dx*dx + dy*dy + dz*dz;
      if (d2 >= R*R || d2 === 0) continue;
      if (!(this.dTouch[i] & 2) && !(touch[i] & 2)) {
        const rv = Math.hypot(dv[3*i]-dv[3*j], dv[3*i+1]-dv[3*j+1], dv[3*i+2]-dv[3*j+2]);
        if (rv > 0.6 && this.events.length < 40) this.events.push({ type: 'tick', speed: rv });
      }
      touch[i] |= 2; touch[j] |= 2;
      const d = Math.sqrt(d2), s = 0.5 * Math.min(R - d, 0.01) / d, k = 0.8 * s;
      dp[3*i] += dx*s; dp[3*i+1] += dy*s; dp[3*i+2] += dz*s; dp[3*j] -= dx*s; dp[3*j+1] -= dy*s; dp[3*j+2] -= dz*s;
      dprev[3*i] += dx*k; dprev[3*i+1] += dy*k; dprev[3*i+2] += dz*k; dprev[3*j] -= dx*k; dprev[3*j+1] -= dy*k; dprev[3*j+2] -= dz*k;
    }
    const H = this.hash, ids = H.queryIds;
    for (let i = 0; i < m; i++) {
      const r = dr[i], wd = 1 / (r * r * r * 4.2);
      const cnt = H.query(dp[3*i], dp[3*i+1], dp[3*i+2]);
      for (let q = 0; q < cnt; q++) {
        const n = ids[q], R = r + this.rad[n];
        if (R <= 0) continue;
        const dx = dp[3*i]-x[3*n], dy = dp[3*i+1]-x[3*n+1], dz = dp[3*i+2]-x[3*n+2], d2 = dx*dx + dy*dy + dz*dz;
        if (d2 >= R*R || d2 === 0) continue;
        if (!(this.dTouch[i] & 1) && !(touch[i] & 1)) {
          const sp = Math.hypot(dv[3*i], dv[3*i+1], dv[3*i+2]);
          if (sp > 0.8 && this.events.length < 40) this.events.push({ type: 'plop', r, speed: sp });
        }
        touch[i] |= 1;
        const d = Math.sqrt(d2), C = Math.min(R - d, 0.01) / d, wn = w[n], ws = wd + wn, kd = 0.6 * C * wd / ws;
        dp[3*i] += dx*C*wd/ws; dp[3*i+1] += dy*C*wd/ws; dp[3*i+2] += dz*C*wd/ws;
        dprev[3*i] += dx*kd; dprev[3*i+1] += dy*kd; dprev[3*i+2] += dz*kd;
        x[3*n] -= dx*C*wn/ws; x[3*n+1] -= dy*C*wn/ws; x[3*n+2] -= dz*C*wn/ws;
      }
    }
    this.dTouch.set(touch);
    const inv = 1 / dt;
    for (let i = 0; i < m; i++) {
      const r = dr[i];
      if (dp[3*i+1] < r) {
        if (dprev[3*i+1] > r + 0.002 && dv[3*i+1] < -0.8 && this.events.length < 40) this.events.push({ type: 'plip', r, speed: -dv[3*i+1] });
        dp[3*i+1] = r; dp[3*i] -= (dp[3*i] - dprev[3*i]) * 0.3; dp[3*i+2] -= (dp[3*i+2] - dprev[3*i+2]) * 0.3; }
      this.wallsDrop(i);
      dv[3*i] = (dp[3*i]-dprev[3*i])*inv; dv[3*i+1] = (dp[3*i+1]-dprev[3*i+1])*inv; dv[3*i+2] = (dp[3*i+2]-dprev[3*i+2])*inv;
      const Rr = Math.hypot(dp[3*i], dp[3*i+2]);
      if (Rr > 4) { const f = 4 / Rr; dp[3*i] *= f; dp[3*i+2] *= f; dv[3*i] *= -0.3; dv[3*i+2] *= -0.3; }
    }
  }

  // ── per-node deformation gradient, for normals on the GPU ─────────────────
  updateNodeF() {
    const { x, nodeF } = this;
    const acc = this._acc || (this._acc = new Float32Array(this.N * 9));
    const F = this._F || (this._F = new Float32Array(9));
    for (const b of this.blocks) {
      if (!b.active) continue;
      const T = b.topo, D = T.DmInv, o = b.off;
      acc.fill(0, 9 * o, 9 * (o + T.n));
      for (let t = 0, m = T.restVol.length; t < m; t++) {
        const i0 = 3 * (o + T.tets[4*t]), d = 9*t, e = this._e || (this._e = new Float32Array(9));
        for (let c = 1; c <= 3; c++) { const ic = 3 * (o + T.tets[4*t+c]); e[3*c-3] = x[ic]-x[i0]; e[3*c-2] = x[ic+1]-x[i0+1]; e[3*c-1] = x[ic+2]-x[i0+2]; }
        for (let col = 0; col < 3; col++) for (let row = 0; row < 3; row++)
          F[3*col+row] = e[row]*D[d+3*col] + e[3+row]*D[d+3*col+1] + e[6+row]*D[d+3*col+2];
        const wv = T.restVol[t];
        for (let c = 0; c < 4; c++) { const a = 9 * (o + T.tets[4*t+c]); for (let k = 0; k < 9; k++) acc[a+k] += F[k] * wv; }
      }
      for (let i = 0; i < T.n; i++) {
        const s = 1 / T.nodeFw[i], a = 9 * (o + i), q = 12 * (o + i);
        nodeF[q] = acc[a]*s; nodeF[q+1] = acc[a+1]*s; nodeF[q+2] = acc[a+2]*s;
        nodeF[q+4] = acc[a+3]*s; nodeF[q+5] = acc[a+4]*s; nodeF[q+6] = acc[a+5]*s;
        nodeF[q+8] = acc[a+6]*s; nodeF[q+9] = acc[a+7]*s; nodeF[q+10] = acc[a+8]*s;
      }
    }
  }

  measure() {
    const { x } = this, T = this.bodyTopo;
    let V = 0;
    for (let t = 0; t < T.restVol.length; t++) {
      const a = 3*T.tets[4*t], b = 3*T.tets[4*t+1], c = 3*T.tets[4*t+2], d = 3*T.tets[4*t+3];
      const ux = x[b]-x[a], uy = x[b+1]-x[a+1], uz = x[b+2]-x[a+2], vx = x[c]-x[a], vy = x[c+1]-x[a+1], vz = x[c+2]-x[a+2], wx = x[d]-x[a], wy = x[d+1]-x[a+1], wz = x[d+2]-x[a+2];
      V += ((uy*vz-uz*vy)*wx + (uz*vx-ux*vz)*wy + (ux*vy-uy*vx)*wz) / 6;
    }
    const L = this.geo.landmarks;
    const len = Math.hypot(x[3*L.snout]-x[3*L.tail], x[3*L.snout+1]-x[3*L.tail+1], x[3*L.snout+2]-x[3*L.tail+2]);
    const attached = this.legSlots.filter(s => s[0].grow >= 1).length;
    const growing = this.legSlots.map(s => s[0]).filter(b => b.grow < 1);
    return { volume: V / T.totalVol, length: len, attached, growing, loose: this.legSlots.flat().filter(b => b.slot > 0 && b.active).length };
  }

  raycast(o, d) {
    let best = { t: Infinity, kind: null };
    const x = this.x;
    for (const blk of this.blocks) {
      if (!blk.active) continue;
      const B = blk.topo.boundary, off = blk.off;
      for (let f = 0; f < B.length; f += 3) {
        const a = 3*(off+B[f]), b = 3*(off+B[f+1]), c = 3*(off+B[f+2]);
        const e1x = x[b]-x[a], e1y = x[b+1]-x[a+1], e1z = x[b+2]-x[a+2], e2x = x[c]-x[a], e2y = x[c+1]-x[a+1], e2z = x[c+2]-x[a+2];
        const px = d[1]*e2z-d[2]*e2y, py = d[2]*e2x-d[0]*e2z, pz = d[0]*e2y-d[1]*e2x, det = e1x*px + e1y*py + e1z*pz;
        if (Math.abs(det) < 1e-10) continue;
        const inv = 1 / det, tx = o[0]-x[a], ty = o[1]-x[a+1], tz = o[2]-x[a+2];
        const u = (tx*px + ty*py + tz*pz) * inv; if (u < 0 || u > 1) continue;
        const qx = ty*e1z-tz*e1y, qy = tz*e1x-tx*e1z, qz = tx*e1y-ty*e1x;
        const v = (d[0]*qx + d[1]*qy + d[2]*qz) * inv; if (v < 0 || u + v > 1) continue;
        const t = (e2x*qx + e2y*qy + e2z*qz) * inv;
        // legs win close calls, so a click on a foot grabs the foot
        const tt = blk === this.body ? t + 0.03 : t;
        if (t > 0 && tt < best.t) best = { t: tt, kind: 'part', block: blk };
      }
    }
    for (let i = 0; i < this.dCount; i++) {
      const r = this.dr[i] * 1.8, ox = o[0]-this.dp[3*i], oy = o[1]-this.dp[3*i+1], oz = o[2]-this.dp[3*i+2];
      const b = ox*d[0] + oy*d[1] + oz*d[2], c = ox*ox + oy*oy + oz*oz - r*r, h = b*b - c;
      if (h < 0) continue;
      const t = -b - Math.sqrt(h);
      if (t > 0 && t < best.t) best = { t, kind: 'drop', index: i };
    }
    if (best.kind) best.point = [o[0]+d[0]*best.t, o[1]+d[1]*best.t, o[2]+d[2]*best.t];
    return best;
  }
}

class SpatialHash {
  constructor(spacing, maxItems) {
    this.s = spacing; this.size = 2 * maxItems;
    this.start = new Int32Array(this.size + 1); this.entries = new Int32Array(maxItems); this.queryIds = new Int32Array(maxItems);
  }
  key(xi, yi, zi) { return Math.abs((xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)) % this.size; }
  build(P, idx, n) {
    this.start.fill(0);
    const k = (i) => this.key(Math.floor(P[3*i]/this.s), Math.floor(P[3*i+1]/this.s), Math.floor(P[3*i+2]/this.s));
    for (let q = 0; q < n; q++) this.start[k(idx[q])]++;
    let acc = 0;
    for (let i = 0; i < this.size; i++) { acc += this.start[i]; this.start[i] = acc; }
    this.start[this.size] = acc;
    for (let q = 0; q < n; q++) this.entries[--this.start[k(idx[q])]] = idx[q];
  }
  query(x, y, z) {
    const xi = Math.floor(x/this.s), yi = Math.floor(y/this.s), zi = Math.floor(z/this.s);
    let n = 0;
    for (let a = xi - 1; a <= xi + 1; a++) for (let b = yi - 1; b <= yi + 1; b++) for (let c = zi - 1; c <= zi + 1; c++) {
      const h = this.key(a, b, c);
      for (let k = this.start[h]; k < this.start[h + 1] && n < this.queryIds.length; k++) this.queryIds[n++] = this.entries[k];
    }
    return n;
  }
}

export { AxoSim, SpatialHash };
