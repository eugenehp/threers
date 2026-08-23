#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f threejs-shim.js ]; then
    echo "ERROR: web/threejs-shim.js is missing."
    echo "       Restore it from git:  git checkout HEAD -- web/threejs-shim.js"
    exit 1
fi

node -e '
const fs = require("fs");
const path = "pkg/threers.js";
let src = fs.readFileSync(path, "utf8");
let changed = false;

const requestDeviceRe = /(__wbg_requestDevice_[a-f0-9]+: function\(arg0, arg1\) \{)[\s\S]*?const ret = arg0\.requestDevice\(arg1\);/;
const requestDevicePatch = `$1
            if (arg1) {
                const desc = { ...arg1 };
                if (desc.requiredLimits) {
                    const _rl = {};
                    for (const [_k, _v] of Object.entries(desc.requiredLimits)) {
                        if (_k !== "maxInterStageShaderComponents") _rl[_k] = _v;
                    }
                    if (Object.keys(_rl).length) desc.requiredLimits = _rl;
                    else delete desc.requiredLimits;
                }
                delete desc.maxInterStageShaderComponents;
                arg1 = desc;
            }
            const ret = arg0.requestDevice(arg1);`;
if (requestDeviceRe.test(src)) {
    src = src.replace(requestDeviceRe, requestDevicePatch);
    changed = true;
}

if (src.includes("({module} = module)")) {
    src = src.replace(/\(\{module\} = module\)/g, "module = module.module;");
    changed = true;
}
if (src.includes("({module_or_path} = module_or_path)")) {
    src = src.replace(
        /\(\{module_or_path\} = module_or_path\)/g,
        "module_or_path = module_or_path.module_or_path;"
    );
    changed = true;
}

if (!changed) {
    console.log("==> threers.js already patched");
} else {
    fs.writeFileSync(path, src);
    console.log("==> patched threers.js (requestDevice + Safari init glue)");
}
'

node scripts/generate-shim-types.mjs

MESH_BVH="${MESH_BVH:-}" BVH_CSG="${BVH_CSG:-}" node scripts/generate-features-addon.mjs

# Dev parity UI reads this to bust browser cache after rebuilds.
date +%s > pkg/build-id.txt
echo "==> wrote pkg/build-id.txt"
