// threers-fem through Node: FEM against beam theory, BEM against a sphere.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { FemModel, FemSurface, femShapes } = require("../index.js");

test("cantilever matches beam theory", { skip: !FemModel && "built without fem" }, () => {
  // 10 × 1 × 1, E = 1000, tip load 1: FL³/3EI = 4, plus a little shear.
  const m = FemModel.boxGrid([0, 0, 0], [10, 1, 1], [20, 2, 2]);
  m.setMaterial(1000, 0.3);
  m.fixBox([-1, -1, -1], [0, 2, 2]);
  m.forceBox([10, -1, -1], [11, 2, 2], [0, -1, 0]);
  const r = m.solve();
  assert.ok(r.maxDisplacement > 3.9 && r.maxDisplacement < 4.2, String(r.maxDisplacement));
  assert.ok(r.displacement instanceof Float64Array);
  assert.equal(r.displacement.length, 3 * m.nodeCount);
  assert.ok(Math.abs(r.reaction[1] - 1) < 1e-9);
});

test("modes and heat on a built-in shape", { skip: !FemModel && "built without fem" }, () => {
  assert.ok(femShapes().some((s) => s.startsWith("bracket")));
  const m = FemModel.shape("beam", 0.5);
  m.setMaterial(200e3, 0.3, 7.85e-9);
  m.fixBox([-1, -1, -1], [0, 2, 2]);
  const { frequencies, shapes } = m.modes(2);
  const want = 0.5596 * Math.sqrt(200e3 / 12 / (7.85e-9 * 1e4));
  assert.ok(Math.abs(frequencies[0] / want - 1) < 0.03, `${frequencies[0]} vs ${want}`);
  assert.equal(shapes[0].length, 3 * m.nodeCount);
  m.clear();
  m.temperatureBox([-1, -1, -1], [0, 2, 2], 100);
  m.temperatureBox([10, -1, -1], [11, 2, 2], 0);
  const h = m.solveHeat(1);
  assert.ok(Math.abs(h.max - 100) < 1e-9 && Math.abs(h.min) < 1e-9);
});

test("errors are thrown with the solver's message", { skip: !FemModel && "built without fem" }, () => {
  const m = FemModel.boxGrid([0, 0, 0], [1, 1, 1], [2, 2, 2]);
  m.setMaterial(1, 0.3);
  m.fixBox([5, 5, 5], [6, 6, 6]);
  assert.throws(() => m.solve(), /support/);
  assert.throws(() => FemModel.shape("teapot", 1), /unknown shape/);
});

test("sphere capacitance and a pulsating sphere", { skip: !FemSurface && "built without fem" }, () => {
  const s = FemSurface.sphere(1, 3);
  assert.ok(Math.abs(s.capacitance() / (4 * Math.PI) - 1) < 0.03);
  // ∂p/∂n = iωρ v; v = −i/(ωρ) gives ∂p/∂n = 1, and p = 1/(i − 1) at k = 1.
  const k = 1, c = 343, rho = 1.204, omega = k * c;
  const n = s.triangleCount;
  const r = s.radiate(omega / (2 * Math.PI), new Array(n).fill(0), new Array(n).fill(-1 / (omega * rho)), c, rho);
  const re = r.pressureRe()[0], im = r.pressureIm()[0];
  // 1 / (−1 + i) = (−1 − i) / 2
  assert.ok(Math.hypot(re + 0.5, im + 0.5) < 0.05 * Math.SQRT1_2, `${re} ${im}`);
});
