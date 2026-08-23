/**
 * nurbs implementation — thin wrappers over the wasm `WebNurbs*` bindings.
 * Loaded only when features.nurbs is true (NURBS=1 web/build.sh).
 *
 * The three.js `NURBSCurve` in the examples is an evaluator: you get points out
 * and nothing else. These carry derivatives, knot operations and a tolerance-
 * driven tessellator, so a surface reaches a mesh with analytic normals rather
 * than a segment count and averaged face normals.
 */
import { isNurbsEnabled } from '/web/features.js';
import * as wasm from '/web/pkg/threers.js';

const _FEATURE_ERR = 'nurbs feature is disabled. Rebuild with: NURBS=1 web/build.sh';

if (!isNurbsEnabled()) {
    throw new Error(_FEATURE_ERR);
}

export { isNurbsEnabled };

/** Flatten `[[x,y,z], …]` or accept an already-flat array. */
function flatten(points, stride) {
    if (points.length && Array.isArray(points[0])) {
        return Float64Array.from(points.flat());
    }
    if (points.length % stride !== 0) {
        throw new Error(`expected a multiple of ${stride} coordinates, got ${points.length}`);
    }
    return Float64Array.from(points);
}

function chunk3(flat) {
    const out = [];
    for (let i = 0; i < flat.length; i += 3) out.push([flat[i], flat[i + 1], flat[i + 2]]);
    return out;
}

export class NurbsCurve {
    /** @param {number} degree @param {number[]} knots @param {number[][]|number[]} points */
    constructor(degree, knots, points, weights = null) {
        this._inner = new wasm.WebNurbsCurve(
            degree,
            Float64Array.from(knots),
            flatten(points, 3),
            weights ? Float64Array.from(weights) : undefined,
        );
    }

    /** Control points already in homogeneous `(w·x, w·y, w·z, w)` form. */
    static fromHomogeneous(degree, knots, control) {
        const c = Object.create(NurbsCurve.prototype);
        c._inner = wasm.WebNurbsCurve.fromHomogeneous(
            degree,
            Float64Array.from(knots),
            flatten(control, 4),
        );
        return c;
    }

    static _wrap(inner) {
        const c = Object.create(NurbsCurve.prototype);
        c._inner = inner;
        return c;
    }

    get degree() {
        return this._inner.degree();
    }
    get knots() {
        return Array.from(this._inner.knots());
    }
    get controlCount() {
        return this._inner.controlCount();
    }
    /** `[uMin, uMax]` */
    get domain() {
        return Array.from(this._inner.domain());
    }

    point(u) {
        return Array.from(this._inner.point(u));
    }
    /** Point at normalized `t ∈ [0, 1]`, matching three.js's `getPoint`. */
    getPoint(t) {
        return Array.from(this._inner.pointAt(t));
    }
    /** Derivatives 0..=k as `[[C], [C'], …]`. */
    derivatives(u, k = 1) {
        return chunk3(this._inner.derivatives(u, k));
    }
    /** Unit tangent, or `null` at a cusp. */
    tangent(u) {
        const t = this._inner.tangent(u);
        return t.length ? Array.from(t) : null;
    }
    /** Adaptive polyline honouring `tolerance` (model units). */
    tessellate(tolerance = 1e-3) {
        return chunk3(this._inner.tessellate(tolerance));
    }
}

export class NurbsSurface {
    /**
     * @param {number[][]|number[]} points u-major control grid, index `i * nV + j`
     */
    constructor(degreeU, degreeV, knotsU, knotsV, nU, nV, points, weights = null) {
        this._inner = new wasm.WebNurbsSurface(
            degreeU,
            degreeV,
            Float64Array.from(knotsU),
            Float64Array.from(knotsV),
            nU,
            nV,
            flatten(points, 3),
            weights ? Float64Array.from(weights) : undefined,
        );
    }

    static _wrap(inner) {
        const s = Object.create(NurbsSurface.prototype);
        s._inner = inner;
        return s;
    }

    get degreeU() {
        return this._inner.degreeU();
    }
    get degreeV() {
        return this._inner.degreeV();
    }
    /** `[uMin, uMax, vMin, vMax]` */
    get domain() {
        return Array.from(this._inner.domain());
    }
    get isRational() {
        return this._inner.isRational();
    }

    point(u, v) {
        return Array.from(this._inner.point(u, v));
    }
    /** Point at normalized `(s, t) ∈ [0, 1]²`. */
    getPoint(s, t) {
        return Array.from(this._inner.pointAt(s, t));
    }
    /** Analytic unit normal — defined at poles too, unlike `Sᵤ × Sᵥ` alone. */
    normal(u, v) {
        const n = this._inner.normal(u, v);
        return n.length ? Array.from(n) : null;
    }

    /** `BufferGeometry` at the given chord tolerance, with analytic normals. */
    toGeometry(tolerance = 1e-3) {
        return this._inner.tessellate(tolerance);
    }
    /** `BufferGeometry` on a fixed grid, when a predictable vertex count matters. */
    toGeometryGrid(uSegments, vSegments) {
        return this._inner.tessellateGrid(uSegments, vSegments);
    }
    /**
     * Did the sampler reach `tolerance`, or did its sample ceiling bind first?
     * A truncated grid is indistinguishable from a converged one by eye.
     */
    meetsTolerance(tolerance = 1e-3) {
        return this._inner.meetsTolerance(tolerance);
    }
}

export const circle = (center, radius) =>
    NurbsCurve._wrap(wasm.nurbsCircle(center[0], center[1], center[2], radius));

export const arc = (center, radius, start, end) =>
    NurbsCurve._wrap(wasm.nurbsArc(center[0], center[1], center[2], radius, start, end));

export const sphere = (center, radius) =>
    NurbsSurface._wrap(wasm.nurbsSphere(center[0], center[1], center[2], radius));

export const cylinder = (base, radius, height) =>
    NurbsSurface._wrap(wasm.nurbsCylinder(base[0], base[1], base[2], radius, height));

export const cone = (apex, baseRadius, height) =>
    NurbsSurface._wrap(wasm.nurbsCone(apex[0], apex[1], apex[2], baseRadius, height));

export const torus = (center, major, minor) =>
    NurbsSurface._wrap(wasm.nurbsTorus(center[0], center[1], center[2], major, minor));

export const revolve = (profile, axisPoint, axisDir, angle = 0) =>
    NurbsSurface._wrap(
        wasm.nurbsRevolve(
            profile._inner,
            axisPoint[0], axisPoint[1], axisPoint[2],
            axisDir[0], axisDir[1], axisDir[2],
            angle,
        ),
    );

export const extrude = (profile, dir) =>
    NurbsSurface._wrap(wasm.nurbsExtrude(profile._inner, dir[0], dir[1], dir[2]));

export const ruled = (a, b) => NurbsSurface._wrap(wasm.nurbsRuled(a._inner, b._inner));

/** Attach the NURBS types to a `THREE`-shaped namespace. */
export function installNurbs(THREE = {}) {
    if (!isNurbsEnabled()) throw new Error(_FEATURE_ERR);
    THREE.NurbsCurve = NurbsCurve;
    THREE.NurbsSurface = NurbsSurface;
    return { enabled: true };
}
