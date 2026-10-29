// From Axolotl Jelly (https://github.com/saulfloresjr/axolotl-jelly): its
// tetrahedral lattice, rest-state precomputation and point embedding, as written
// there, with the creature removed and buildPart taking a prebuilt skin.
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
// soft-lattice.js — the lattice a soft part is simulated on.
//
//   tetLattice   cubes over the part's field, each split into five tetrahedra
//   buildPart    a skin (positions, normals, material channels, triangles)
//                plus its lattice, rest volumes, edges, boundary and the
//                binding of every skin vertex to the tetrahedron it lies in
//   embedPoints  that binding, for any other points (insides, chain roots)
// ─────────────────────────────────────────────────────────────────────────────
const SoftLattice = (() => {
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const TET_EVEN = [[1,2,4,7],[0,1,2,4],[3,1,2,7],[5,1,4,7],[6,2,4,7]];
  const TET_ODD  = [[0,3,5,6],[1,0,3,5],[2,0,3,6],[4,0,5,6],[7,3,5,6]];
  function tetVolume(P, T, t) {
    const a = 3*T[t], b = 3*T[t+1], c = 3*T[t+2], d = 3*T[t+3];
    const ux = P[b]-P[a], uy = P[b+1]-P[a+1], uz = P[b+2]-P[a+2], vx = P[c]-P[a], vy = P[c+1]-P[a+1], vz = P[c+2]-P[a+2], wx = P[d]-P[a], wy = P[d+1]-P[a+1], wz = P[d+2]-P[a+2];
    return ((uy*vz-uz*vy)*wx + (uz*vx-ux*vz)*wy + (ux*vy-uy*vx)*wz) / 6;
  }
  function invert3(m, out, o) {
    const [a,b,c,d,e,f,g,h,i] = m, A = e*i-f*h, B = -(d*i-f*g), C = d*h-e*g, r = 1 / (a*A + b*B + c*C);
    out[o] = A*r; out[o+1] = -(b*i-c*h)*r; out[o+2] = (b*f-c*e)*r; out[o+3] = B*r; out[o+4] = (a*i-c*g)*r; out[o+5] = -(a*f-c*d)*r; out[o+6] = C*r; out[o+7] = -(a*h-b*g)*r; out[o+8] = (a*e-b*d)*r;
  }
  function tetLattice(f, B, h) {
    const o = [B.min[0], 0, B.min[2]];
    const nx = Math.ceil((B.max[0] - o[0]) / h), ny = Math.ceil((B.max[1] - o[1]) / h), nz = Math.ceil((B.max[2] - o[2]) / h);
    const nodeId = new Map(), pos = [], tets = [], cellTets = new Map();
    const node = (i, j, k) => {
      const key = (k * (ny + 1) + j) * (nx + 1) + i;
      let n = nodeId.get(key);
      if (n === undefined) { n = pos.length / 3; nodeId.set(key, n); pos.push(o[0] + i * h, o[1] + j * h, o[2] + k * h); }
      return n;
    };
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (f(o[0] + (i + 0.5) * h, o[1] + (j + 0.5) * h, o[2] + (k + 0.5) * h) > h * 0.6) continue;
      const c = [];
      for (let b = 0; b < 8; b++) c.push(node(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)));
      const list = [];
      for (const t of ((i + j + k) & 1) ? TET_ODD : TET_EVEN) { list.push(tets.length / 4); tets.push(c[t[0]], c[t[1]], c[t[2]], c[t[3]]); }
      cellTets.set((k * ny + j) * nx + i, list);
    }
    const P = new Float32Array(pos), T = new Uint32Array(tets);
    for (let t = 0; t < T.length; t += 4) if (tetVolume(P, T, t) < 0) { const s = T[t+2]; T[t+2] = T[t+3]; T[t+3] = s; }
    return { pos: P, tets: T, cellTets, o, h, nx, ny, nz };
  }
  // bind each point to the tetrahedron that contains it (or the nearest one)
  function embedPoints(points, lat, DmInv) {
    const nv = points.length / 3, tetOf = new Uint32Array(nv), bary = new Float32Array(nv * 3);
    const { o, h, nx, ny, cellTets, pos: X, tets: T } = lat;
    for (let v = 0; v < nv; v++) {
      const px = points[3*v], py = points[3*v+1], pz = points[3*v+2];
      const ci = Math.floor((px - o[0]) / h), cj = Math.floor((py - o[1]) / h), ck = Math.floor((pz - o[2]) / h);
      let best = 0, bestScore = -Infinity, bb = [0, 0, 0];
      for (let r = 0; r <= 3 && bestScore < -1e-4; r++)
        for (let dk = -r; dk <= r; dk++) for (let dj = -r; dj <= r; dj++) for (let di = -r; di <= r; di++) {
          if (Math.max(Math.abs(di), Math.abs(dj), Math.abs(dk)) !== r) continue;
          const list = cellTets.get(((ck + dk) * ny + (cj + dj)) * nx + (ci + di));
          if (!list) continue;
          for (const t of list) {
            const a = 3 * T[4*t], m = 9 * t, rx = px - X[a], ry = py - X[a+1], rz = pz - X[a+2];
            const b1 = DmInv[m]*rx + DmInv[m+3]*ry + DmInv[m+6]*rz, b2 = DmInv[m+1]*rx + DmInv[m+4]*ry + DmInv[m+7]*rz, b3 = DmInv[m+2]*rx + DmInv[m+5]*ry + DmInv[m+8]*rz;
            const score = Math.min(1 - b1 - b2 - b3, b1, b2, b3);
            if (score > bestScore) { bestScore = score; best = t; bb = [b1, b2, b3]; }
          }
        }
      tetOf[v] = best; bary.set(bb, 3 * v);
    }
    return { tetOf, bary };
  }

  // One soft part: skin + lattice + everything the solver and renderer need
  function buildPart(f, B, h, partId, surf) {
    for (let i = 0; i < surf.mat.length / 4; i++) surf.mat[4 * i + 3] = partId;
    const lat = tetLattice(f, B, h);
    const nT = lat.tets.length / 4, nN = lat.pos.length / 3;
    const DmInv = new Float32Array(nT * 9), restVol = new Float32Array(nT), edgeSet = new Map(), faces = new Map();
    for (let t = 0; t < nT; t++) {
      const T = lat.tets, P = lat.pos, a = 3 * T[4*t], m = [];
      for (let c = 1; c <= 3; c++) { const b = 3 * T[4*t+c]; m.push(P[b]-P[a], P[b+1]-P[a+1], P[b+2]-P[a+2]); }
      invert3(m, DmInv, 9 * t);
      restVol[t] = tetVolume(P, T, 4 * t);
      const q = [T[4*t], T[4*t+1], T[4*t+2], T[4*t+3]];
      for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) { const lo = Math.min(q[i], q[j]), hi = Math.max(q[i], q[j]); edgeSet.set(lo * nN + hi, [lo, hi]); }
      for (const fc of [[0,2,1],[0,1,3],[0,3,2],[1,2,3]]) {
        const tri = [q[fc[0]], q[fc[1]], q[fc[2]]], key = [...tri].sort((x, y) => x - y).join(',');
        const e = faces.get(key); if (e) e.n++; else faces.set(key, { n: 1, tri });
      }
    }
    const edges = new Uint32Array(edgeSet.size * 2); let ei = 0;
    for (const [, [a, b]] of edgeSet) { edges[ei++] = a; edges[ei++] = b; }
    const boundary = [];
    for (const [, fc] of faces) if (fc.n === 1) boundary.push(...fc.tri);
    const radius = new Float32Array(nN);
    for (let i = 0; i < nN; i++) radius[i] = clamp(0.35 * h - f(lat.pos[3*i], lat.pos[3*i+1], lat.pos[3*i+2]), 0, h * 1.1);
    const emb = embedPoints(surf.pos, lat, DmInv);
    return { surf, emb, lat, nodes: lat.pos, tets: lat.tets, edges, boundary: new Uint32Array(boundary), DmInv, restVol, radius, h };
  }


  return { buildPart, embedPoints, tetLattice };
})();

export { SoftLattice };
