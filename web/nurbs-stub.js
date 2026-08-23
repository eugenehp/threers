/**
 * Stub nurbs addon — active when wasm was built without NURBS=1.
 */
import { features, isNurbsEnabled } from './features.js';

const ERR = 'nurbs feature is disabled. Rebuild wasm with: NURBS=1 web/build.sh';

export { features, isNurbsEnabled };

export class NurbsCurve {
    constructor() {
        throw new Error(`NurbsCurve: ${ERR}`);
    }
}
export class NurbsSurface {
    constructor() {
        throw new Error(`NurbsSurface: ${ERR}`);
    }
}

function disabled(name) {
    const err = () => {
        throw new Error(`${name}: ${ERR}`);
    };
    err.enabled = false;
    return err;
}

export const circle = disabled('circle');
export const arc = disabled('arc');
export const sphere = disabled('sphere');
export const cylinder = disabled('cylinder');
export const cone = disabled('cone');
export const torus = disabled('torus');
export const revolve = disabled('revolve');
export const extrude = disabled('extrude');
export const ruled = disabled('ruled');

/** Mirrors the impl's shape so callers can branch without a try/catch. */
export function installNurbs() {
    return { enabled: false };
}
