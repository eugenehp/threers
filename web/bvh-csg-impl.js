/**
 * three-bvh-csg@0.0.16 for the shim (loaded when features.bvhCsg is true —
 * BVH_CSG=1 or VARIANT=full web/build.sh). Entry point: bvh-csg-addon.js.
 *
 * The boolean itself — the BVH intersection, triangle splitting and the
 * inside/outside classification — runs in wasm (`csgEvaluate`, the Rust port in
 * `src/csg/`). The rest is three-bvh-csg's own JavaScript, kept here as the
 * library has it: `Brush`, `Operation` and `OperationGroup` as scene-graph
 * nodes with their dirty tracking, and `Evaluator`'s attribute selection,
 * group and material bookkeeping and `evaluateHierarchy`.
 */

import { isBvhCsgEnabled } from '/web/features.js';
import { installMeshBvh } from '/web/mesh-bvh-impl.js';
import { Mesh, Group, Matrix4, BufferGeometry, BufferAttribute, Vector3 } from '/web/threejs-shim.js';
import { csgEvaluate } from '/web/pkg/threers.js';

const _FEATURE_ERR = 'bvh-csg feature is disabled. Rebuild with: BVH_CSG=1 web/build.sh';

if (!isBvhCsgEnabled()) {
    throw new Error(_FEATURE_ERR);
}

// ---- constants (three-bvh-csg) ----
export const ADDITION = 0;
export const SUBTRACTION = 1;
export const REVERSE_SUBTRACTION = 2;
export const INTERSECTION = 3;
export const DIFFERENCE = 4;
export const HOLLOW_SUBTRACTION = 5;
export const HOLLOW_INTERSECTION = 6;

function _matricesDiffer(a, b) {
    const ae = a.elements, be = b.elements;
    for (let i = 0; i < 16; i++) {
        if (ae[i] !== be[i]) return true;
    }
    return false;
}

/** three-bvh-csg `Brush`: a Mesh the evaluator can take as an operand. */
export class Brush extends Mesh {
    constructor(...args) {
        super(...args);
        this.isBrush = true;
        this._previousMatrix = new Matrix4();
        this._previousMatrix.elements.fill(0);
    }

    markUpdated() {
        this._previousMatrix.copy(this.matrix);
    }

    isDirty() {
        return _matricesDiffer(this.matrix, this._previousMatrix);
    }

    /// The BVH and half-edge structures are built in wasm for each evaluation;
    /// nothing is cached on the geometry.
    prepareGeometry() {}

    disposeCacheData() {}
}

/** three-bvh-csg `Operation`: a Brush applied to its parent with `operation`. */
export class Operation extends Brush {
    constructor(...args) {
        super(...args);
        this.isOperation = true;
        this.operation = ADDITION;
        this._cachedGeometry = new BufferGeometry();
        this._cachedMaterials = null;
        this._previousOperation = null;
    }

    markUpdated() {
        super.markUpdated();
        this._previousOperation = this.operation;
    }

    isDirty() {
        return this.operation !== this._previousOperation || super.isDirty();
    }

    insertBefore(brush) {
        const parent = this.parent;
        const index = parent.children.indexOf(this);
        parent.children.splice(index, 0, brush);
    }

    insertAfter(brush) {
        const parent = this.parent;
        const index = parent.children.indexOf(this);
        parent.children.splice(index + 1, 0, brush);
    }
}

/** three-bvh-csg `OperationGroup`: children that apply to the enclosing operation. */
export class OperationGroup extends Group {
    constructor() {
        super();
        this.isOperationGroup = true;
        this._previousMatrix = new Matrix4();
        this._previousMatrix.elements.fill(0);
    }

    markUpdated() {
        this._previousMatrix.copy(this.matrix);
    }

    isDirty() {
        return _matricesDiffer(this.matrix, this._previousMatrix);
    }
}

// The plain description `csgEvaluate` reads.
function _describe(brush) {
    const geometry = brush.geometry;
    const attributes = {};
    for (const key in geometry.attributes) {
        const attr = geometry.attributes[key];
        if (attr.isInterleavedBufferAttribute) {
            throw new Error('Brush: InterleavedBufferAttributes are not supported.');
        }
        const array = attr.array instanceof Float32Array ? attr.array : Float32Array.from(attr.array);
        attributes[key] = { array, itemSize: attr.itemSize };
    }
    const index = geometry.index ? Uint32Array.from(geometry.index.array) : null;
    return {
        attributes,
        index,
        groups: geometry.groups.map(g => ({ start: g.start, count: g.count, materialIndex: g.materialIndex ?? 0 })),
        matrixWorld: Float64Array.from(brush.matrixWorld.elements),
    };
}

// three-bvh-csg `getMaterialList`.
function getMaterialList(groups, materials) {
    let result = materials;
    if (!Array.isArray(materials)) {
        result = [];
        groups.forEach(g => {
            result[g.materialIndex] = materials;
        });
    }
    return result;
}

// three-bvh-csg `joinGroups`: merge neighbouring groups of one material.
function joinGroups(groups) {
    for (let i = 0; i < groups.length - 1; i++) {
        const group = groups[i];
        const nextGroup = groups[i + 1];
        if (group.materialIndex === nextGroup.materialIndex) {
            const start = group.start;
            const end = nextGroup.start + nextGroup.count;
            nextGroup.start = start;
            nextGroup.count = end - start;
            groups.splice(i, 1);
            i--;
        }
    }
}

// three-bvh-csg `assignBufferData`, into a fresh geometry: the groups'
// triangles concatenated in `groupOrder`, a group for each non-empty one.
function assignBufferData(geometry, groupData, attributes, itemSizes, groupOrder) {
    const keys = attributes.filter(k => groupData.some(g => g && g[k]));
    const order = groupOrder.slice(0, groupData.length);
    for (const key of keys) {
        let length = 0;
        for (const { index } of order) length += groupData[index]?.[key]?.length ?? 0;
        const array = new Float32Array(length);
        let offset = 0;
        for (const { index } of order) {
            const part = groupData[index]?.[key];
            if (!part) continue;
            array.set(part, offset);
            offset += part.length;
        }
        geometry.setAttribute(key, new BufferAttribute(array, itemSizes[key]));
    }
    geometry.clearGroups();
    let groupOffset = 0;
    for (const { index, materialIndex } of order) {
        const pos = groupData[index]?.position;
        const vertCount = pos ? pos.length / 3 : 0;
        if (vertCount !== 0) {
            geometry.addGroup(groupOffset, vertCount, materialIndex);
            groupOffset += vertCount;
        }
    }
    geometry.setDrawRange(0, groupOffset);
    geometry.boundsTree = null;
}

/** three-bvh-csg `Evaluator`. */
export class Evaluator {
    constructor() {
        this.attributes = ['position', 'uv', 'normal'];
        this.useGroups = true;
        this.consolidateGroups = true;
        this.debug = { enabled: false, init() {}, complete() {} };
    }

    getGroupRanges(geometry) {
        return !this.useGroups || geometry.groups.length === 0
            ? [{ start: 0, count: Infinity, materialIndex: 0 }]
            : geometry.groups.map(group => ({ ...group }));
    }

    evaluate(a, b, operations, targetBrushes = new Brush()) {
        let wasArray = true;
        if (!Array.isArray(operations)) operations = [operations];
        if (!Array.isArray(targetBrushes)) {
            targetBrushes = [targetBrushes];
            wasArray = false;
        }
        if (targetBrushes.length !== operations.length) {
            throw new Error('Evaluator: operations and target array passed as different sizes.');
        }

        a.prepareGeometry();
        b.prepareGeometry();
        const { attributes, useGroups, consolidateGroups } = this;
        // Every attribute the evaluator carries must be on `a`, as in three-bvh-csg.
        for (const key of attributes) {
            if (!a.geometry.attributes[key]) {
                throw new TypeError(`Evaluator: brush geometry has no "${key}" attribute`);
            }
        }
        // `itemSize` of each carried attribute, from `a`.
        const sizes = Object.fromEntries(attributes.map(k => [k, a.geometry.attributes[k].itemSize]));
        const results = csgEvaluate(_describe(a), _describe(b), Uint8Array.from(operations), useGroups, attributes);

        // The materials and group ranges.
        const aGroups = this.getGroupRanges(a.geometry);
        const aMaterials = getMaterialList(aGroups, a.material);
        const bGroups = this.getGroupRanges(b.geometry);
        const bMaterials = getMaterialList(bGroups, b.material);
        bGroups.forEach(g => (g.materialIndex += aMaterials.length));

        let groups = [...aGroups, ...bGroups].map((group, index) => ({ ...group, index }));

        if (useGroups) {
            const allMaterials = [...aMaterials, ...bMaterials];
            if (consolidateGroups) {
                groups = groups
                    .map(group => {
                        const mat = allMaterials[group.materialIndex];
                        group.materialIndex = allMaterials.indexOf(mat);
                        return group;
                    })
                    .sort((x, y) => x.materialIndex - y.materialIndex);
            }
            // Map old to new indices and drop the materials no group uses.
            const finalMaterials = [];
            for (let i = 0, l = allMaterials.length; i < l; i++) {
                let foundGroup = false;
                for (let g = 0, lg = groups.length; g < lg; g++) {
                    const group = groups[g];
                    if (group.materialIndex === i) {
                        foundGroup = true;
                        group.materialIndex = finalMaterials.length;
                    }
                }
                if (foundGroup) finalMaterials.push(allMaterials[i]);
            }
            targetBrushes.forEach(tb => {
                tb.material = finalMaterials;
            });
        } else {
            groups = [{ start: 0, count: Infinity, index: 0, materialIndex: 0 }];
            targetBrushes.forEach(tb => {
                tb.material = aMaterials[0];
            });
        }

        targetBrushes.forEach((brush, i) => {
            const geometry = new BufferGeometry();
            assignBufferData(geometry, results[i], attributes, sizes, groups);
            if (consolidateGroups) joinGroups(geometry.groups);
            brush.geometry = geometry;
        });

        return wasArray ? targetBrushes : targetBrushes[0];
    }

    evaluateHierarchy(root, target = new Brush()) {
        root.updateMatrixWorld(true);

        const flatTraverse = (obj, cb) => {
            const children = obj.children;
            for (let i = 0, l = children.length; i < l; i++) {
                const child = children[i];
                if (child.isOperationGroup) {
                    flatTraverse(child, cb);
                } else {
                    cb(child);
                }
            }
        };

        const traverse = brush => {
            const children = brush.children;
            let didChange = false;
            for (let i = 0, l = children.length; i < l; i++) {
                const child = children[i];
                didChange = traverse(child) || didChange;
            }
            const isDirty = brush.isDirty();
            if (isDirty) brush.markUpdated();
            if (didChange && !brush.isOperationGroup) {
                let result;
                flatTraverse(brush, child => {
                    if (!result) {
                        result = this.evaluate(brush, child, child.operation);
                    } else {
                        result = this.evaluate(result, child, child.operation);
                    }
                });
                brush._cachedGeometry = result.geometry;
                brush._cachedMaterials = result.material;
                return true;
            }
            return didChange || isDirty;
        };

        traverse(root);
        target.geometry = root._cachedGeometry;
        target.material = root._cachedMaterials;
        return target;
    }

    reset() {}
}

/** three-bvh-csg `computeMeshVolume`: the signed volume a closed mesh encloses, in world space. */
export function computeMeshVolume(mesh) {
    mesh.updateMatrixWorld?.();
    const geometry = mesh.geometry ?? mesh;
    const matrix = mesh.matrixWorld;
    const pos = geometry.attributes.position;
    const index = geometry.index;
    const count = index ? index.count : pos.count;
    const a = new Vector3(), b = new Vector3(), c = new Vector3();
    let volume = 0;
    for (let i = 0; i < count; i += 3) {
        const i0 = index ? index.getX(i) : i;
        const i1 = index ? index.getX(i + 1) : i + 1;
        const i2 = index ? index.getX(i + 2) : i + 2;
        a.fromBufferAttribute(pos, i0);
        b.fromBufferAttribute(pos, i1);
        c.fromBufferAttribute(pos, i2);
        if (matrix) {
            a.applyMatrix4(matrix);
            b.applyMatrix4(matrix);
            c.applyMatrix4(matrix);
        }
        volume += a.dot(b.cross(c)) / 6;
    }
    return volume;
}

/** Patch threers for three-bvh-csg compatibility (implies installMeshBvh). */
export function installBvhCsg(THREE = {}) {
    installMeshBvh(THREE);
    return { enabled: true };
}

export default { installBvhCsg };
