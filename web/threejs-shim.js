import init, {
    // renderer + scene + camera
    WebRenderer, WebRenderTarget, WebCubeRenderTarget, WebScene, WebCamera, WebObjectHandle,
    // geometries
    WebGeometry, WebBufferGeometry, WebBufferAttribute,
    // materials
    WebMaterial,
    WebShaderMaterial,
    // lights
    WebLight,
    // math
    WebColor, WebVector2, WebVector3, WebVector4, WebBox3,
    // textures
    WebTexture, WebCubeTexture, WebDataTexture,
    // curves
    WebCurve,
    WebShapeCurvePath, WebShapeList, WebFont, shapeUtilsArea, shapeUtilsTriangulate,
    // animation
    WebAnimClip, WebAnimTrack, WebInterpolant, WebAnimationMixer, WebAnimationObjectGroup, WebPropertyMixer,
    setAnimationCallbacks, setAnimationNodeUuid, dropAnimationNode, takeAnimationWrites, takeAnimationStringWrites,
    takeAnimationReleased, animationWritesPending, trackModifierRemapTime, trackModifierApplyValue,
    parseTrackName, sanitizeNodeName, getKeyframeOrder,
    // controls
    WebOrbitControls, WebTrackballControls, WebFirstPersonControls,
    WebPointerLockControls, WebArcballControls,
    // helpers
    WebAxesHelper, WebGridHelper, WebPolarGridHelper,
    // debug helpers, three.js r165 geometry (src/wasm/extras.rs)
    helperDirectionalLight, helperHemisphereLight, helperPointLight, helperSpotLight,
    helperPerspectiveProjection, helperOrthographicProjection, helperCamera,
    helperArrowQuaternion, helperArrow, helperPlane, helperSkeleton, helperBoxOfObjects,
    helperBox3, helperVertexNormals, helperRectAreaLight, helperRectAreaLightColor,
    // loaders
    WebObjLoader, WebStlLoader, WebPlyLoader, WebHdrLoader,
    WebFbxLoader, WebColladaLoader, WebExrLoader,
    WebAlembicLoader, WebKtx2Loader, WebDracoLoader,
    // exporters (src/wasm/exporters.rs)
    WebSceneExporter,
    // audio
    WebAudio, WebAudioListener,
    // post-fx
    WebEffectComposer, WebPass, WebPmremGenerator,
    // extras
    WebOctree, WebThreeSimplexNoise, WebMarchingCubes,
    // alt renderers
    WebCss2dRenderer, WebSvgRenderer,
    // stats
    WebStats,
    // raycaster + clock
    WebRaycaster, WebClock,
    // geometry generators (src/wasm/generators.rs)
    WebConvexHull,
} from './pkg/threers.js';
// Bindings that exist only when the wasm bundle was built with the matching
// Cargo feature. A named import of one of these would fail at load time for
// every build that lacks it, taking the whole shim with it, so they are looked
// up on the namespace instead.
import * as _optional from './pkg/threers.js';
import {
    BrowserVideoFormat,
    VideoEncodeWorker,
    VideoExportError,
    VideoExportErrorCode,
    VideoExportEvent,
    VideoExportProgressEvent,
    VideoExporter,
    VideoExportResult,
    VideoFormat,
    alignVideoSize,
    assertVideoExportAvailable,
    downloadVideoBytes,
    emitProgress,
    encodeAndDownloadVideoFrames,
    encodeApngRgba as _encodeApngRgba,
    encodeGifRgba as _encodeGifRgba,
    encodeVideoFrames,
    encodeVideoFramesInWorker,
    encodeWebmRgba as _encodeWebmRgba,
    formatBytes,
    formatVideoProgress,
    isVideoExportAvailable,
    parseVideoFormat,
    videoFilename,
    videoMimeType,
} from './video-export.js';

export {
    BrowserVideoFormat,
    VideoEncodeWorker,
    VideoExportError,
    VideoExportErrorCode,
    VideoExportEvent,
    VideoExportProgressEvent,
    VideoExporter,
    VideoExportResult,
    VideoFormat,
    alignVideoSize,
    assertVideoExportAvailable,
    downloadVideoBytes,
    emitProgress,
    encodeAndDownloadVideoFrames,
    encodeVideoFrames,
    encodeVideoFramesInWorker,
    formatBytes,
    formatVideoProgress,
    isVideoExportAvailable,
    parseVideoFormat,
    videoFilename,
    videoMimeType,
};

/** Present when wasm was built with `--features native-codec` (throws if missing). */
export const encodeGifRgba = _encodeGifRgba;
export const encodeApngRgba = _encodeApngRgba;
export const encodeWebmRgba = _encodeWebmRgba;

/** Browsers reject wgpu's legacy `maxInterStageShaderComponents` limit name. */
const STRIP_WEBGPU_LIMITS = new Set(['maxInterStageShaderComponents']);

function filterWebGpuRequiredLimits(requiredLimits) {
    if (!requiredLimits) return requiredLimits;
    const out = {};
    for (const [k, v] of Object.entries(requiredLimits)) {
        if (!STRIP_WEBGPU_LIMITS.has(k)) out[k] = v;
    }
    return out;
}

function patchWebGpuDeviceLimits() {
    if (globalThis.__threersWebGpuPatched) return true;
    const Adapter = globalThis.GPUAdapter;
    if (!Adapter?.prototype?.requestDevice) return false;
    const orig = Adapter.prototype.requestDevice;
    Adapter.prototype.requestDevice = function (descriptor) {
        if (descriptor?.requiredLimits) {
            descriptor = {
                ...descriptor,
                requiredLimits: filterWebGpuRequiredLimits(descriptor.requiredLimits),
            };
        }
        return orig.call(this, descriptor);
    };
    globalThis.__threersWebGpuPatched = true;
    return true;
}
patchWebGpuDeviceLimits();

let _wasmReady = null;
export async function initThreers(wasmUrl) {
    if (!patchWebGpuDeviceLimits()) {
        for (let i = 0; i < 100 && !patchWebGpuDeviceLimits(); i++) {
            await new Promise((r) => setTimeout(r, 10));
        }
    }
    if (_wasmReady) return _wasmReady;
    const opts = (wasmUrl != null && typeof wasmUrl === 'object')
        ? wasmUrl
        : { module_or_path: wasmUrl };
    _wasmReady = init(opts);
    return _wasmReady;
}

// Material.side (exported early — csg/mesh-bvh import these by name).
export const FrontSide = 0;
export const BackSide = 1;
export const DoubleSide = 2;
// Material.depthFunc (three.js constants).
export const NeverDepth = 0;
export const AlwaysDepth = 1;
export const LessDepth = 2;
export const LessEqualDepth = 3;
export const EqualDepth = 4;
export const GreaterEqualDepth = 5;
export const GreaterDepth = 6;
export const NotEqualDepth = 7;

// three.js texture type constants (also exported on default THREE object below).
const UnsignedByteType = 1009;
const HalfFloatType = 1016;

// ---- Math helpers ----
/// A wasm `WebColor` for a three.js colour argument, made at the point a
/// material, light, fog or background consumes it. A `Color` holds linear
/// working-space floats (three.js r165), which is what `WebColor`'s
/// constructor takes; a hex number or CSS string goes through `Color` so it
/// is decoded from sRGB exactly as three.js decodes it.
function _color(input) {
    if (input && input.isColor) return new WebColor(input.r, input.g, input.b);
    if (typeof input === 'number' || typeof input === 'string') {
        const c = _colorScratch.set(input);
        return new WebColor(c.r, c.g, c.b);
    }
    if (typeof input === 'object' && input && 'r' in input) return new WebColor(input.r, input.g, input.b);
    return new WebColor(1, 1, 1);
}

// ---- Color + ColorManagement: three.js r165, pure JavaScript ----
const _CM_SRGB = 'srgb', _CM_LINEAR_SRGB = 'srgb-linear';
const _CM_P3 = 'display-p3', _CM_LINEAR_P3 = 'display-p3-linear';
/// three.js `SRGBToLinear` / `LinearToSRGB` (ColorManagement.js).
function SRGBToLinear( c ) {
    return ( c < 0.04045 ) ? c * 0.0773993808 : Math.pow( c * 0.9478672986 + 0.0521327014, 2.4 );
}
function LinearToSRGB( c ) {
    return ( c < 0.0031308 ) ? c * 12.92 : 1.055 * ( Math.pow( c, 0.41666 ) ) - 0.055;
}
// P3 <-> Rec. 709 primaries, column-major as `Matrix3.set` stores them.
// Applied inline so ColorManagement needs nothing defined later in the module.
const _CM_LINEAR_SRGB_TO_LINEAR_P3 = [ 0.8224621, 0.0331941, 0.0170827, 0.177538, 0.9668058, 0.0723974, 0.0, 0.0, 0.9105199 ];
const _CM_LINEAR_P3_TO_LINEAR_SRGB = [ 1.2249401, - 0.0420569, - 0.0196376, - 0.2249404, 1.0420571, - 0.0786361, 0.0, 0.0, 1.0982735 ];
const _cmApply = ( color, elements ) => color.applyMatrix3( { elements } );
const _CM_COLOR_SPACES = {
    [ _CM_LINEAR_SRGB ]: {
        transfer: 'linear', primaries: 'rec709',
        toReference: ( color ) => color,
        fromReference: ( color ) => color,
    },
    [ _CM_SRGB ]: {
        transfer: 'srgb', primaries: 'rec709',
        toReference: ( color ) => color.convertSRGBToLinear(),
        fromReference: ( color ) => color.convertLinearToSRGB(),
    },
    [ _CM_LINEAR_P3 ]: {
        transfer: 'linear', primaries: 'p3',
        toReference: ( color ) => _cmApply( color, _CM_LINEAR_P3_TO_LINEAR_SRGB ),
        fromReference: ( color ) => _cmApply( color, _CM_LINEAR_SRGB_TO_LINEAR_P3 ),
    },
    [ _CM_P3 ]: {
        transfer: 'srgb', primaries: 'p3',
        toReference: ( color ) => _cmApply( color.convertSRGBToLinear(), _CM_LINEAR_P3_TO_LINEAR_SRGB ),
        fromReference: ( color ) => _cmApply( color, _CM_LINEAR_SRGB_TO_LINEAR_P3 ).convertLinearToSRGB(),
    },
};
const _CM_SUPPORTED_WORKING_COLOR_SPACES = new Set( [ _CM_LINEAR_SRGB, _CM_LINEAR_P3 ] );
/// three.js r165 `ColorManagement`.
export const ColorManagement = {
    enabled: true,
    _workingColorSpace: _CM_LINEAR_SRGB,
    get workingColorSpace() {
        return this._workingColorSpace;
    },
    set workingColorSpace( colorSpace ) {
        if ( ! _CM_SUPPORTED_WORKING_COLOR_SPACES.has( colorSpace ) ) {
            throw new Error( `Unsupported working color space, "${ colorSpace }".` );
        }
        this._workingColorSpace = colorSpace;
    },
    convert: function ( color, sourceColorSpace, targetColorSpace ) {
        if ( this.enabled === false || sourceColorSpace === targetColorSpace || ! sourceColorSpace || ! targetColorSpace ) {
            return color;
        }
        const sourceToReference = _CM_COLOR_SPACES[ sourceColorSpace ].toReference;
        const targetFromReference = _CM_COLOR_SPACES[ targetColorSpace ].fromReference;
        return targetFromReference( sourceToReference( color ) );
    },
    fromWorkingColorSpace: function ( color, targetColorSpace ) {
        return this.convert( color, this._workingColorSpace, targetColorSpace );
    },
    toWorkingColorSpace: function ( color, sourceColorSpace ) {
        return this.convert( color, sourceColorSpace, this._workingColorSpace );
    },
    getPrimaries: function ( colorSpace ) {
        return _CM_COLOR_SPACES[ colorSpace ].primaries;
    },
    getTransfer: function ( colorSpace ) {
        if ( colorSpace === '' ) return 'linear';
        return _CM_COLOR_SPACES[ colorSpace ].transfer;
    },
};

const _colorKeywords = { 'aliceblue': 0xF0F8FF, 'antiquewhite': 0xFAEBD7, 'aqua': 0x00FFFF, 'aquamarine': 0x7FFFD4, 'azure': 0xF0FFFF,
    'beige': 0xF5F5DC, 'bisque': 0xFFE4C4, 'black': 0x000000, 'blanchedalmond': 0xFFEBCD, 'blue': 0x0000FF, 'blueviolet': 0x8A2BE2,
    'brown': 0xA52A2A, 'burlywood': 0xDEB887, 'cadetblue': 0x5F9EA0, 'chartreuse': 0x7FFF00, 'chocolate': 0xD2691E, 'coral': 0xFF7F50,
    'cornflowerblue': 0x6495ED, 'cornsilk': 0xFFF8DC, 'crimson': 0xDC143C, 'cyan': 0x00FFFF, 'darkblue': 0x00008B, 'darkcyan': 0x008B8B,
    'darkgoldenrod': 0xB8860B, 'darkgray': 0xA9A9A9, 'darkgreen': 0x006400, 'darkgrey': 0xA9A9A9, 'darkkhaki': 0xBDB76B, 'darkmagenta': 0x8B008B,
    'darkolivegreen': 0x556B2F, 'darkorange': 0xFF8C00, 'darkorchid': 0x9932CC, 'darkred': 0x8B0000, 'darksalmon': 0xE9967A, 'darkseagreen': 0x8FBC8F,
    'darkslateblue': 0x483D8B, 'darkslategray': 0x2F4F4F, 'darkslategrey': 0x2F4F4F, 'darkturquoise': 0x00CED1, 'darkviolet': 0x9400D3,
    'deeppink': 0xFF1493, 'deepskyblue': 0x00BFFF, 'dimgray': 0x696969, 'dimgrey': 0x696969, 'dodgerblue': 0x1E90FF, 'firebrick': 0xB22222,
    'floralwhite': 0xFFFAF0, 'forestgreen': 0x228B22, 'fuchsia': 0xFF00FF, 'gainsboro': 0xDCDCDC, 'ghostwhite': 0xF8F8FF, 'gold': 0xFFD700,
    'goldenrod': 0xDAA520, 'gray': 0x808080, 'green': 0x008000, 'greenyellow': 0xADFF2F, 'grey': 0x808080, 'honeydew': 0xF0FFF0, 'hotpink': 0xFF69B4,
    'indianred': 0xCD5C5C, 'indigo': 0x4B0082, 'ivory': 0xFFFFF0, 'khaki': 0xF0E68C, 'lavender': 0xE6E6FA, 'lavenderblush': 0xFFF0F5, 'lawngreen': 0x7CFC00,
    'lemonchiffon': 0xFFFACD, 'lightblue': 0xADD8E6, 'lightcoral': 0xF08080, 'lightcyan': 0xE0FFFF, 'lightgoldenrodyellow': 0xFAFAD2, 'lightgray': 0xD3D3D3,
    'lightgreen': 0x90EE90, 'lightgrey': 0xD3D3D3, 'lightpink': 0xFFB6C1, 'lightsalmon': 0xFFA07A, 'lightseagreen': 0x20B2AA, 'lightskyblue': 0x87CEFA,
    'lightslategray': 0x778899, 'lightslategrey': 0x778899, 'lightsteelblue': 0xB0C4DE, 'lightyellow': 0xFFFFE0, 'lime': 0x00FF00, 'limegreen': 0x32CD32,
    'linen': 0xFAF0E6, 'magenta': 0xFF00FF, 'maroon': 0x800000, 'mediumaquamarine': 0x66CDAA, 'mediumblue': 0x0000CD, 'mediumorchid': 0xBA55D3,
    'mediumpurple': 0x9370DB, 'mediumseagreen': 0x3CB371, 'mediumslateblue': 0x7B68EE, 'mediumspringgreen': 0x00FA9A, 'mediumturquoise': 0x48D1CC,
    'mediumvioletred': 0xC71585, 'midnightblue': 0x191970, 'mintcream': 0xF5FFFA, 'mistyrose': 0xFFE4E1, 'moccasin': 0xFFE4B5, 'navajowhite': 0xFFDEAD,
    'navy': 0x000080, 'oldlace': 0xFDF5E6, 'olive': 0x808000, 'olivedrab': 0x6B8E23, 'orange': 0xFFA500, 'orangered': 0xFF4500, 'orchid': 0xDA70D6,
    'palegoldenrod': 0xEEE8AA, 'palegreen': 0x98FB98, 'paleturquoise': 0xAFEEEE, 'palevioletred': 0xDB7093, 'papayawhip': 0xFFEFD5, 'peachpuff': 0xFFDAB9,
    'peru': 0xCD853F, 'pink': 0xFFC0CB, 'plum': 0xDDA0DD, 'powderblue': 0xB0E0E6, 'purple': 0x800080, 'rebeccapurple': 0x663399, 'red': 0xFF0000, 'rosybrown': 0xBC8F8F,
    'royalblue': 0x4169E1, 'saddlebrown': 0x8B4513, 'salmon': 0xFA8072, 'sandybrown': 0xF4A460, 'seagreen': 0x2E8B57, 'seashell': 0xFFF5EE,
    'sienna': 0xA0522D, 'silver': 0xC0C0C0, 'skyblue': 0x87CEEB, 'slateblue': 0x6A5ACD, 'slategray': 0x708090, 'slategrey': 0x708090, 'snow': 0xFFFAFA,
    'springgreen': 0x00FF7F, 'steelblue': 0x4682B4, 'tan': 0xD2B48C, 'teal': 0x008080, 'thistle': 0xD8BFD8, 'tomato': 0xFF6347, 'turquoise': 0x40E0D0,
    'violet': 0xEE82EE, 'wheat': 0xF5DEB3, 'white': 0xFFFFFF, 'whitesmoke': 0xF5F5F5, 'yellow': 0xFFFF00, 'yellowgreen': 0x9ACD32 };
const _hslA = { h: 0, s: 0, l: 0 };
const _hslB = { h: 0, s: 0, l: 0 };
const _colorClamp = ( value, min, max ) => Math.max( min, Math.min( max, value ) );
const _colorEuclideanModulo = ( n, m ) => ( ( n % m ) + m ) % m;
const _colorLerp = ( x, y, t ) => ( 1 - t ) * x + t * y;
function _hue2rgb( p, q, t ) {
    if ( t < 0 ) t += 1;
    if ( t > 1 ) t -= 1;
    if ( t < 1 / 6 ) return p + ( q - p ) * 6 * t;
    if ( t < 1 / 2 ) return q;
    if ( t < 2 / 3 ) return p + ( q - p ) * 6 * ( 2 / 3 - t );
    return p;
}
/// three.js r165 `Color`: linear working-space floats in `r`/`g`/`b`; hex and
/// CSS inputs are sRGB and decoded on the way in, `getHex`/`getStyle` encode.
/// A plain value type — nothing here touches wasm. Materials, lights, fog and
/// backgrounds read `r`/`g`/`b` when they consume it (see `_color`).
export class Color {
    constructor( r, g, b ) {
        this.isColor = true;
        this.r = 1;
        this.g = 1;
        this.b = 1;
        return this.set( r, g, b );
    }
    set( r, g, b ) {
        if ( g === undefined && b === undefined ) {
            // r is THREE.Color, hex or string
            const value = r;
            if ( value && value.isColor ) {
                this.copy( value );
            } else if ( typeof value === 'number' ) {
                this.setHex( value );
            } else if ( typeof value === 'string' ) {
                this.setStyle( value );
            }
        } else {
            this.setRGB( r, g, b );
        }
        return this;
    }
    setScalar( scalar ) {
        this.r = scalar;
        this.g = scalar;
        this.b = scalar;
        return this;
    }
    setHex( hex, colorSpace = _CM_SRGB ) {
        hex = Math.floor( hex );
        this.r = ( hex >> 16 & 255 ) / 255;
        this.g = ( hex >> 8 & 255 ) / 255;
        this.b = ( hex & 255 ) / 255;
        ColorManagement.toWorkingColorSpace( this, colorSpace );
        return this;
    }
    setRGB( r, g, b, colorSpace = ColorManagement.workingColorSpace ) {
        this.r = r;
        this.g = g;
        this.b = b;
        ColorManagement.toWorkingColorSpace( this, colorSpace );
        return this;
    }
    setHSL( h, s, l, colorSpace = ColorManagement.workingColorSpace ) {
        // h,s,l ranges are in 0.0 - 1.0
        h = _colorEuclideanModulo( h, 1 );
        s = _colorClamp( s, 0, 1 );
        l = _colorClamp( l, 0, 1 );
        if ( s === 0 ) {
            this.r = this.g = this.b = l;
        } else {
            const p = l <= 0.5 ? l * ( 1 + s ) : l + s - ( l * s );
            const q = ( 2 * l ) - p;
            this.r = _hue2rgb( q, p, h + 1 / 3 );
            this.g = _hue2rgb( q, p, h );
            this.b = _hue2rgb( q, p, h - 1 / 3 );
        }
        ColorManagement.toWorkingColorSpace( this, colorSpace );
        return this;
    }
    setStyle( style, colorSpace = _CM_SRGB ) {
        function handleAlpha( string ) {
            if ( string === undefined ) return;
            if ( parseFloat( string ) < 1 ) {
                console.warn( 'THREE.Color: Alpha component of ' + style + ' will be ignored.' );
            }
        }
        let m;
        if ( m = /^(\w+)\(([^\)]*)\)/.exec( style ) ) {
            // rgb / hsl
            let color;
            const name = m[ 1 ];
            const components = m[ 2 ];
            switch ( name ) {
                case 'rgb':
                case 'rgba':
                    if ( color = /^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec( components ) ) {
                        // rgb(255,0,0) rgba(255,0,0,0.5)
                        handleAlpha( color[ 4 ] );
                        return this.setRGB(
                            Math.min( 255, parseInt( color[ 1 ], 10 ) ) / 255,
                            Math.min( 255, parseInt( color[ 2 ], 10 ) ) / 255,
                            Math.min( 255, parseInt( color[ 3 ], 10 ) ) / 255,
                            colorSpace
                        );
                    }
                    if ( color = /^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec( components ) ) {
                        // rgb(100%,0%,0%) rgba(100%,0%,0%,0.5)
                        handleAlpha( color[ 4 ] );
                        return this.setRGB(
                            Math.min( 100, parseInt( color[ 1 ], 10 ) ) / 100,
                            Math.min( 100, parseInt( color[ 2 ], 10 ) ) / 100,
                            Math.min( 100, parseInt( color[ 3 ], 10 ) ) / 100,
                            colorSpace
                        );
                    }
                    break;
                case 'hsl':
                case 'hsla':
                    if ( color = /^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec( components ) ) {
                        // hsl(120,50%,50%) hsla(120,50%,50%,0.5)
                        handleAlpha( color[ 4 ] );
                        return this.setHSL(
                            parseFloat( color[ 1 ] ) / 360,
                            parseFloat( color[ 2 ] ) / 100,
                            parseFloat( color[ 3 ] ) / 100,
                            colorSpace
                        );
                    }
                    break;
                default:
                    // threers: the rest of CSS Color 4 (`hwb`, `lab`, `lch`,
                    // `oklab`, `oklch`, `color()`, space-separated syntax).
                    // three.js's own models in capitals stay unknown, as there.
                    if ( ! /^(rgba?|hsla?)$/i.test( name ) && this._setCss( style ) ) return this;
                    console.warn( 'THREE.Color: Unknown color model ' + style );
            }
        } else if ( m = /^\#([A-Fa-f\d]+)$/.exec( style ) ) {
            // hex color
            const hex = m[ 1 ];
            const size = hex.length;
            if ( size === 3 ) {
                // #ff0
                return this.setRGB(
                    parseInt( hex.charAt( 0 ), 16 ) / 15,
                    parseInt( hex.charAt( 1 ), 16 ) / 15,
                    parseInt( hex.charAt( 2 ), 16 ) / 15,
                    colorSpace
                );
            } else if ( size === 6 ) {
                // #ff0000
                return this.setHex( parseInt( hex, 16 ), colorSpace );
            } else if ( ! this._setCss( style ) ) {
                // threers: `#rgba` and `#rrggbbaa` (CSS Color 4) parse too.
                console.warn( 'THREE.Color: Invalid hex color ' + style );
            }
        } else if ( style && style.length > 0 ) {
            return this.setColorName( style, colorSpace );
        }
        return this;
    }
    /// threers: any CSS Color 4 string three.js's own parser does not take,
    /// through the renderer's (`WebColor.fromCss`, linear RGB). False if it
    /// is not a colour either.
    _setCss( style ) {
        const c = WebColor.fromCss?.( style );
        if ( ! c ) return false;
        this.r = c.r(); this.g = c.g(); this.b = c.b();
        return true;
    }
    setColorName( style, colorSpace = _CM_SRGB ) {
        // color keywords
        const hex = _colorKeywords[ style.toLowerCase() ];
        if ( hex !== undefined ) {
            // red
            this.setHex( hex, colorSpace );
        } else {
            // unknown color
            console.warn( 'THREE.Color: Unknown color ' + style );
        }
        return this;
    }
    clone() {
        return new this.constructor( this.r, this.g, this.b );
    }
    copy( color ) {
        this.r = color.r;
        this.g = color.g;
        this.b = color.b;
        return this;
    }
    copySRGBToLinear( color ) {
        this.r = SRGBToLinear( color.r );
        this.g = SRGBToLinear( color.g );
        this.b = SRGBToLinear( color.b );
        return this;
    }
    copyLinearToSRGB( color ) {
        this.r = LinearToSRGB( color.r );
        this.g = LinearToSRGB( color.g );
        this.b = LinearToSRGB( color.b );
        return this;
    }
    convertSRGBToLinear() {
        this.copySRGBToLinear( this );
        return this;
    }
    convertLinearToSRGB() {
        this.copyLinearToSRGB( this );
        return this;
    }
    getHex( colorSpace = _CM_SRGB ) {
        ColorManagement.fromWorkingColorSpace( _colorScratch.copy( this ), colorSpace );
        return Math.round( _colorClamp( _colorScratch.r * 255, 0, 255 ) ) * 65536 + Math.round( _colorClamp( _colorScratch.g * 255, 0, 255 ) ) * 256 + Math.round( _colorClamp( _colorScratch.b * 255, 0, 255 ) );
    }
    getHexString( colorSpace = _CM_SRGB ) {
        return ( '000000' + this.getHex( colorSpace ).toString( 16 ) ).slice( - 6 );
    }
    getHSL( target, colorSpace = ColorManagement.workingColorSpace ) {
        // h,s,l ranges are in 0.0 - 1.0
        ColorManagement.fromWorkingColorSpace( _colorScratch.copy( this ), colorSpace );
        const r = _colorScratch.r, g = _colorScratch.g, b = _colorScratch.b;
        const max = Math.max( r, g, b );
        const min = Math.min( r, g, b );
        let hue, saturation;
        const lightness = ( min + max ) / 2.0;
        if ( min === max ) {
            hue = 0;
            saturation = 0;
        } else {
            const delta = max - min;
            saturation = lightness <= 0.5 ? delta / ( max + min ) : delta / ( 2 - max - min );
            switch ( max ) {
                case r: hue = ( g - b ) / delta + ( g < b ? 6 : 0 ); break;
                case g: hue = ( b - r ) / delta + 2; break;
                case b: hue = ( r - g ) / delta + 4; break;
            }
            hue /= 6;
        }
        target.h = hue;
        target.s = saturation;
        target.l = lightness;
        return target;
    }
    getRGB( target, colorSpace = ColorManagement.workingColorSpace ) {
        ColorManagement.fromWorkingColorSpace( _colorScratch.copy( this ), colorSpace );
        target.r = _colorScratch.r;
        target.g = _colorScratch.g;
        target.b = _colorScratch.b;
        return target;
    }
    getStyle( colorSpace = _CM_SRGB ) {
        ColorManagement.fromWorkingColorSpace( _colorScratch.copy( this ), colorSpace );
        const r = _colorScratch.r, g = _colorScratch.g, b = _colorScratch.b;
        if ( colorSpace !== _CM_SRGB ) {
            // Requires CSS Color Module Level 4 (https://www.w3.org/TR/css-color-4/).
            return `color(${ colorSpace } ${ r.toFixed( 3 ) } ${ g.toFixed( 3 ) } ${ b.toFixed( 3 ) })`;
        }
        return `rgb(${ Math.round( r * 255 ) },${ Math.round( g * 255 ) },${ Math.round( b * 255 ) })`;
    }
    offsetHSL( h, s, l ) {
        this.getHSL( _hslA );
        return this.setHSL( _hslA.h + h, _hslA.s + s, _hslA.l + l );
    }
    add( color ) {
        this.r += color.r;
        this.g += color.g;
        this.b += color.b;
        return this;
    }
    addColors( color1, color2 ) {
        this.r = color1.r + color2.r;
        this.g = color1.g + color2.g;
        this.b = color1.b + color2.b;
        return this;
    }
    addScalar( s ) {
        this.r += s;
        this.g += s;
        this.b += s;
        return this;
    }
    sub( color ) {
        this.r = Math.max( 0, this.r - color.r );
        this.g = Math.max( 0, this.g - color.g );
        this.b = Math.max( 0, this.b - color.b );
        return this;
    }
    multiply( color ) {
        this.r *= color.r;
        this.g *= color.g;
        this.b *= color.b;
        return this;
    }
    multiplyScalar( s ) {
        this.r *= s;
        this.g *= s;
        this.b *= s;
        return this;
    }
    lerp( color, alpha ) {
        this.r += ( color.r - this.r ) * alpha;
        this.g += ( color.g - this.g ) * alpha;
        this.b += ( color.b - this.b ) * alpha;
        return this;
    }
    lerpColors( color1, color2, alpha ) {
        this.r = color1.r + ( color2.r - color1.r ) * alpha;
        this.g = color1.g + ( color2.g - color1.g ) * alpha;
        this.b = color1.b + ( color2.b - color1.b ) * alpha;
        return this;
    }
    lerpHSL( color, alpha ) {
        this.getHSL( _hslA );
        color.getHSL( _hslB );
        const h = _colorLerp( _hslA.h, _hslB.h, alpha );
        const s = _colorLerp( _hslA.s, _hslB.s, alpha );
        const l = _colorLerp( _hslA.l, _hslB.l, alpha );
        this.setHSL( h, s, l );
        return this;
    }
    setFromVector3( v ) {
        this.r = v.x;
        this.g = v.y;
        this.b = v.z;
        return this;
    }
    applyMatrix3( m ) {
        const r = this.r, g = this.g, b = this.b;
        const e = m.elements;
        this.r = e[ 0 ] * r + e[ 3 ] * g + e[ 6 ] * b;
        this.g = e[ 1 ] * r + e[ 4 ] * g + e[ 7 ] * b;
        this.b = e[ 2 ] * r + e[ 5 ] * g + e[ 8 ] * b;
        return this;
    }
    equals( c ) {
        return ( c.r === this.r ) && ( c.g === this.g ) && ( c.b === this.b );
    }
    fromArray( array, offset = 0 ) {
        this.r = array[ offset ];
        this.g = array[ offset + 1 ];
        this.b = array[ offset + 2 ];
        return this;
    }
    toArray( array = [], offset = 0 ) {
        array[ offset ] = this.r;
        array[ offset + 1 ] = this.g;
        array[ offset + 2 ] = this.b;
        return array;
    }
    fromBufferAttribute( attribute, index ) {
        this.r = attribute.getX( index );
        this.g = attribute.getY( index );
        this.b = attribute.getZ( index );
        return this;
    }
    toJSON() {
        return this.getHex();
    }
    *[ Symbol.iterator ]() {
        yield this.r;
        yield this.g;
        yield this.b;
    }
}
const _colorScratch = /*@__PURE__*/ new Color();
Color.NAMES = _colorKeywords;

// ---- Math value types: three.js r165, pure JavaScript ----
// Vector2/3/4, Matrix3, Box2/3, Sphere, Ray, Plane, Triangle, Frustum,
// Spherical, Cylindrical and Line3 are three.js r165's own code, so their
// results are three.js's to the last bit. Their module-level scratch values
// sit together after Line3. Only what needs scene data — a geometry's or an
// object's bounds — goes to Rust (`src/wasm/math.rs`).
export class Vector2 {
    constructor( x = 0, y = 0 ) {
        Vector2.prototype.isVector2 = true;
        this.x = x;
        this.y = y;
    }
    get width() {
        return this.x;
    }
    set width( value ) {
        this.x = value;
    }
    get height() {
        return this.y;
    }
    set height( value ) {
        this.y = value;
    }
    set( x, y ) {
        this.x = x;
        this.y = y;
        return this;
    }
    setScalar( scalar ) {
        this.x = scalar;
        this.y = scalar;
        return this;
    }
    setX( x ) {
        this.x = x;
        return this;
    }
    setY( y ) {
        this.y = y;
        return this;
    }
    setComponent( index, value ) {
        switch ( index ) {
            case 0: this.x = value; break;
            case 1: this.y = value; break;
            default: throw new Error( 'index is out of range: ' + index );
        }
        return this;
    }
    getComponent( index ) {
        switch ( index ) {
            case 0: return this.x;
            case 1: return this.y;
            default: throw new Error( 'index is out of range: ' + index );
        }
    }
    clone() {
        return new this.constructor( this.x, this.y );
    }
    copy( v ) {
        this.x = v.x;
        this.y = v.y;
        return this;
    }
    add( v ) {
        this.x += v.x;
        this.y += v.y;
        return this;
    }
    addScalar( s ) {
        this.x += s;
        this.y += s;
        return this;
    }
    addVectors( a, b ) {
        this.x = a.x + b.x;
        this.y = a.y + b.y;
        return this;
    }
    addScaledVector( v, s ) {
        this.x += v.x * s;
        this.y += v.y * s;
        return this;
    }
    sub( v ) {
        this.x -= v.x;
        this.y -= v.y;
        return this;
    }
    subScalar( s ) {
        this.x -= s;
        this.y -= s;
        return this;
    }
    subVectors( a, b ) {
        this.x = a.x - b.x;
        this.y = a.y - b.y;
        return this;
    }
    multiply( v ) {
        this.x *= v.x;
        this.y *= v.y;
        return this;
    }
    multiplyScalar( scalar ) {
        this.x *= scalar;
        this.y *= scalar;
        return this;
    }
    divide( v ) {
        this.x /= v.x;
        this.y /= v.y;
        return this;
    }
    divideScalar( scalar ) {
        return this.multiplyScalar( 1 / scalar );
    }
    applyMatrix3( m ) {
        const x = this.x, y = this.y;
        const e = m.elements;
        this.x = e[ 0 ] * x + e[ 3 ] * y + e[ 6 ];
        this.y = e[ 1 ] * x + e[ 4 ] * y + e[ 7 ];
        return this;
    }
    min( v ) {
        this.x = Math.min( this.x, v.x );
        this.y = Math.min( this.y, v.y );
        return this;
    }
    max( v ) {
        this.x = Math.max( this.x, v.x );
        this.y = Math.max( this.y, v.y );
        return this;
    }
    clamp( min, max ) {
        // assumes min < max, componentwise
        this.x = Math.max( min.x, Math.min( max.x, this.x ) );
        this.y = Math.max( min.y, Math.min( max.y, this.y ) );
        return this;
    }
    clampScalar( minVal, maxVal ) {
        this.x = Math.max( minVal, Math.min( maxVal, this.x ) );
        this.y = Math.max( minVal, Math.min( maxVal, this.y ) );
        return this;
    }
    clampLength( min, max ) {
        const length = this.length();
        return this.divideScalar( length || 1 ).multiplyScalar( Math.max( min, Math.min( max, length ) ) );
    }
    floor() {
        this.x = Math.floor( this.x );
        this.y = Math.floor( this.y );
        return this;
    }
    ceil() {
        this.x = Math.ceil( this.x );
        this.y = Math.ceil( this.y );
        return this;
    }
    round() {
        this.x = Math.round( this.x );
        this.y = Math.round( this.y );
        return this;
    }
    roundToZero() {
        this.x = Math.trunc( this.x );
        this.y = Math.trunc( this.y );
        return this;
    }
    negate() {
        this.x = - this.x;
        this.y = - this.y;
        return this;
    }
    dot( v ) {
        return this.x * v.x + this.y * v.y;
    }
    cross( v ) {
        return this.x * v.y - this.y * v.x;
    }
    lengthSq() {
        return this.x * this.x + this.y * this.y;
    }
    length() {
        return Math.sqrt( this.x * this.x + this.y * this.y );
    }
    manhattanLength() {
        return Math.abs( this.x ) + Math.abs( this.y );
    }
    normalize() {
        return this.divideScalar( this.length() || 1 );
    }
    angle() {
        // computes the angle in radians with respect to the positive x-axis
        const angle = Math.atan2( - this.y, - this.x ) + Math.PI;
        return angle;
    }
    angleTo( v ) {
        const denominator = Math.sqrt( this.lengthSq() * v.lengthSq() );
        if ( denominator === 0 ) return Math.PI / 2;
        const theta = this.dot( v ) / denominator;
        // clamp, to handle numerical problems
        return Math.acos( MathUtils.clamp( theta, - 1, 1 ) );
    }
    distanceTo( v ) {
        return Math.sqrt( this.distanceToSquared( v ) );
    }
    distanceToSquared( v ) {
        const dx = this.x - v.x, dy = this.y - v.y;
        return dx * dx + dy * dy;
    }
    manhattanDistanceTo( v ) {
        return Math.abs( this.x - v.x ) + Math.abs( this.y - v.y );
    }
    setLength( length ) {
        return this.normalize().multiplyScalar( length );
    }
    lerp( v, alpha ) {
        this.x += ( v.x - this.x ) * alpha;
        this.y += ( v.y - this.y ) * alpha;
        return this;
    }
    lerpVectors( v1, v2, alpha ) {
        this.x = v1.x + ( v2.x - v1.x ) * alpha;
        this.y = v1.y + ( v2.y - v1.y ) * alpha;
        return this;
    }
    equals( v ) {
        return ( ( v.x === this.x ) && ( v.y === this.y ) );
    }
    fromArray( array, offset = 0 ) {
        this.x = array[ offset ];
        this.y = array[ offset + 1 ];
        return this;
    }
    toArray( array = [], offset = 0 ) {
        array[ offset ] = this.x;
        array[ offset + 1 ] = this.y;
        return array;
    }
    fromBufferAttribute( attribute, index ) {
        this.x = attribute.getX( index );
        this.y = attribute.getY( index );
        return this;
    }
    rotateAround( center, angle ) {
        const c = Math.cos( angle ), s = Math.sin( angle );
        const x = this.x - center.x;
        const y = this.y - center.y;
        this.x = x * c - y * s + center.x;
        this.y = x * s + y * c + center.y;
        return this;
    }
    random() {
        this.x = Math.random();
        this.y = Math.random();
        return this;
    }
    *[ Symbol.iterator ]() {
        yield this.x;
        yield this.y;
    }
    /// The wasm `WebVector2` — for the bindings that take one.
    _w() { return new WebVector2(this.x, this.y); }
}
export class Vector3 {
    constructor( x = 0, y = 0, z = 0 ) {
        Vector3.prototype.isVector3 = true;
        this.x = x;
        this.y = y;
        this.z = z;
    }
    set( x, y, z ) {
        if ( z === undefined ) z = this.z; // sprite.scale.set(x,y)
        this.x = x;
        this.y = y;
        this.z = z;
        return this;
    }
    setScalar( scalar ) {
        this.x = scalar;
        this.y = scalar;
        this.z = scalar;
        return this;
    }
    setX( x ) {
        this.x = x;
        return this;
    }
    setY( y ) {
        this.y = y;
        return this;
    }
    setZ( z ) {
        this.z = z;
        return this;
    }
    setComponent( index, value ) {
        switch ( index ) {
            case 0: this.x = value; break;
            case 1: this.y = value; break;
            case 2: this.z = value; break;
            default: throw new Error( 'index is out of range: ' + index );
        }
        return this;
    }
    getComponent( index ) {
        switch ( index ) {
            case 0: return this.x;
            case 1: return this.y;
            case 2: return this.z;
            default: throw new Error( 'index is out of range: ' + index );
        }
    }
    clone() {
        return new this.constructor( this.x, this.y, this.z );
    }
    copy( v ) {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
        return this;
    }
    add( v ) {
        this.x += v.x;
        this.y += v.y;
        this.z += v.z;
        return this;
    }
    addScalar( s ) {
        this.x += s;
        this.y += s;
        this.z += s;
        return this;
    }
    addVectors( a, b ) {
        this.x = a.x + b.x;
        this.y = a.y + b.y;
        this.z = a.z + b.z;
        return this;
    }
    addScaledVector( v, s ) {
        this.x += v.x * s;
        this.y += v.y * s;
        this.z += v.z * s;
        return this;
    }
    sub( v ) {
        this.x -= v.x;
        this.y -= v.y;
        this.z -= v.z;
        return this;
    }
    subScalar( s ) {
        this.x -= s;
        this.y -= s;
        this.z -= s;
        return this;
    }
    subVectors( a, b ) {
        this.x = a.x - b.x;
        this.y = a.y - b.y;
        this.z = a.z - b.z;
        return this;
    }
    multiply( v ) {
        this.x *= v.x;
        this.y *= v.y;
        this.z *= v.z;
        return this;
    }
    multiplyScalar( scalar ) {
        this.x *= scalar;
        this.y *= scalar;
        this.z *= scalar;
        return this;
    }
    multiplyVectors( a, b ) {
        this.x = a.x * b.x;
        this.y = a.y * b.y;
        this.z = a.z * b.z;
        return this;
    }
    applyEuler( euler ) {
        return this.applyQuaternion( _v3_quaternion.setFromEuler( euler ) );
    }
    applyAxisAngle( axis, angle ) {
        return this.applyQuaternion( _v3_quaternion.setFromAxisAngle( axis, angle ) );
    }
    applyMatrix3( m ) {
        const x = this.x, y = this.y, z = this.z;
        const e = m.elements;
        this.x = e[ 0 ] * x + e[ 3 ] * y + e[ 6 ] * z;
        this.y = e[ 1 ] * x + e[ 4 ] * y + e[ 7 ] * z;
        this.z = e[ 2 ] * x + e[ 5 ] * y + e[ 8 ] * z;
        return this;
    }
    applyNormalMatrix( m ) {
        return this.applyMatrix3( m ).normalize();
    }
    applyMatrix4( m ) {
        const x = this.x, y = this.y, z = this.z;
        const e = m.elements;
        const w = 1 / ( e[ 3 ] * x + e[ 7 ] * y + e[ 11 ] * z + e[ 15 ] );
        this.x = ( e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z + e[ 12 ] ) * w;
        this.y = ( e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z + e[ 13 ] ) * w;
        this.z = ( e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z + e[ 14 ] ) * w;
        return this;
    }
    applyQuaternion( q ) {
        // quaternion q is assumed to have unit length
        const vx = this.x, vy = this.y, vz = this.z;
        const qx = q.x, qy = q.y, qz = q.z, qw = q.w;
        // t = 2 * cross( q.xyz, v );
        const tx = 2 * ( qy * vz - qz * vy );
        const ty = 2 * ( qz * vx - qx * vz );
        const tz = 2 * ( qx * vy - qy * vx );
        // v + q.w * t + cross( q.xyz, t );
        this.x = vx + qw * tx + qy * tz - qz * ty;
        this.y = vy + qw * ty + qz * tx - qx * tz;
        this.z = vz + qw * tz + qx * ty - qy * tx;
        return this;
    }
    project( camera ) {
        return this.applyMatrix4( camera.matrixWorldInverse ).applyMatrix4( camera.projectionMatrix );
    }
    unproject( camera ) {
        return this.applyMatrix4( camera.projectionMatrixInverse ).applyMatrix4( camera.matrixWorld );
    }
    transformDirection( m ) {
        // input: THREE.Matrix4 affine matrix
        // vector interpreted as a direction
        const x = this.x, y = this.y, z = this.z;
        const e = m.elements;
        this.x = e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z;
        this.y = e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z;
        this.z = e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z;
        return this.normalize();
    }
    divide( v ) {
        this.x /= v.x;
        this.y /= v.y;
        this.z /= v.z;
        return this;
    }
    divideScalar( scalar ) {
        return this.multiplyScalar( 1 / scalar );
    }
    min( v ) {
        this.x = Math.min( this.x, v.x );
        this.y = Math.min( this.y, v.y );
        this.z = Math.min( this.z, v.z );
        return this;
    }
    max( v ) {
        this.x = Math.max( this.x, v.x );
        this.y = Math.max( this.y, v.y );
        this.z = Math.max( this.z, v.z );
        return this;
    }
    clamp( min, max ) {
        // assumes min < max, componentwise
        this.x = Math.max( min.x, Math.min( max.x, this.x ) );
        this.y = Math.max( min.y, Math.min( max.y, this.y ) );
        this.z = Math.max( min.z, Math.min( max.z, this.z ) );
        return this;
    }
    clampScalar( minVal, maxVal ) {
        this.x = Math.max( minVal, Math.min( maxVal, this.x ) );
        this.y = Math.max( minVal, Math.min( maxVal, this.y ) );
        this.z = Math.max( minVal, Math.min( maxVal, this.z ) );
        return this;
    }
    clampLength( min, max ) {
        const length = this.length();
        return this.divideScalar( length || 1 ).multiplyScalar( Math.max( min, Math.min( max, length ) ) );
    }
    floor() {
        this.x = Math.floor( this.x );
        this.y = Math.floor( this.y );
        this.z = Math.floor( this.z );
        return this;
    }
    ceil() {
        this.x = Math.ceil( this.x );
        this.y = Math.ceil( this.y );
        this.z = Math.ceil( this.z );
        return this;
    }
    round() {
        this.x = Math.round( this.x );
        this.y = Math.round( this.y );
        this.z = Math.round( this.z );
        return this;
    }
    roundToZero() {
        this.x = Math.trunc( this.x );
        this.y = Math.trunc( this.y );
        this.z = Math.trunc( this.z );
        return this;
    }
    negate() {
        this.x = - this.x;
        this.y = - this.y;
        this.z = - this.z;
        return this;
    }
    dot( v ) {
        return this.x * v.x + this.y * v.y + this.z * v.z;
    }
    // TODO lengthSquared?
    lengthSq() {
        return this.x * this.x + this.y * this.y + this.z * this.z;
    }
    length() {
        return Math.sqrt( this.x * this.x + this.y * this.y + this.z * this.z );
    }
    manhattanLength() {
        return Math.abs( this.x ) + Math.abs( this.y ) + Math.abs( this.z );
    }
    normalize() {
        return this.divideScalar( this.length() || 1 );
    }
    setLength( length ) {
        return this.normalize().multiplyScalar( length );
    }
    lerp( v, alpha ) {
        this.x += ( v.x - this.x ) * alpha;
        this.y += ( v.y - this.y ) * alpha;
        this.z += ( v.z - this.z ) * alpha;
        return this;
    }
    lerpVectors( v1, v2, alpha ) {
        this.x = v1.x + ( v2.x - v1.x ) * alpha;
        this.y = v1.y + ( v2.y - v1.y ) * alpha;
        this.z = v1.z + ( v2.z - v1.z ) * alpha;
        return this;
    }
    cross( v ) {
        return this.crossVectors( this, v );
    }
    crossVectors( a, b ) {
        const ax = a.x, ay = a.y, az = a.z;
        const bx = b.x, by = b.y, bz = b.z;
        this.x = ay * bz - az * by;
        this.y = az * bx - ax * bz;
        this.z = ax * by - ay * bx;
        return this;
    }
    projectOnVector( v ) {
        const denominator = v.lengthSq();
        if ( denominator === 0 ) return this.set( 0, 0, 0 );
        const scalar = v.dot( this ) / denominator;
        return this.copy( v ).multiplyScalar( scalar );
    }
    projectOnPlane( planeNormal ) {
        _v3_vector.copy( this ).projectOnVector( planeNormal );
        return this.sub( _v3_vector );
    }
    reflect( normal ) {
        // reflect incident vector off plane orthogonal to normal
        // normal is assumed to have unit length
        return this.sub( _v3_vector.copy( normal ).multiplyScalar( 2 * this.dot( normal ) ) );
    }
    angleTo( v ) {
        const denominator = Math.sqrt( this.lengthSq() * v.lengthSq() );
        if ( denominator === 0 ) return Math.PI / 2;
        const theta = this.dot( v ) / denominator;
        // clamp, to handle numerical problems
        return Math.acos( MathUtils.clamp( theta, - 1, 1 ) );
    }
    distanceTo( v ) {
        return Math.sqrt( this.distanceToSquared( v ) );
    }
    distanceToSquared( v ) {
        const dx = this.x - v.x, dy = this.y - v.y, dz = this.z - v.z;
        return dx * dx + dy * dy + dz * dz;
    }
    manhattanDistanceTo( v ) {
        return Math.abs( this.x - v.x ) + Math.abs( this.y - v.y ) + Math.abs( this.z - v.z );
    }
    setFromSpherical( s ) {
        return this.setFromSphericalCoords( s.radius, s.phi, s.theta );
    }
    setFromSphericalCoords( radius, phi, theta ) {
        const sinPhiRadius = Math.sin( phi ) * radius;
        this.x = sinPhiRadius * Math.sin( theta );
        this.y = Math.cos( phi ) * radius;
        this.z = sinPhiRadius * Math.cos( theta );
        return this;
    }
    setFromCylindrical( c ) {
        return this.setFromCylindricalCoords( c.radius, c.theta, c.y );
    }
    setFromCylindricalCoords( radius, theta, y ) {
        this.x = radius * Math.sin( theta );
        this.y = y;
        this.z = radius * Math.cos( theta );
        return this;
    }
    setFromMatrixPosition( m ) {
        const e = m.elements;
        this.x = e[ 12 ];
        this.y = e[ 13 ];
        this.z = e[ 14 ];
        return this;
    }
    setFromMatrixScale( m ) {
        const sx = this.setFromMatrixColumn( m, 0 ).length();
        const sy = this.setFromMatrixColumn( m, 1 ).length();
        const sz = this.setFromMatrixColumn( m, 2 ).length();
        this.x = sx;
        this.y = sy;
        this.z = sz;
        return this;
    }
    setFromMatrixColumn( m, index ) {
        return this.fromArray( m.elements, index * 4 );
    }
    setFromMatrix3Column( m, index ) {
        return this.fromArray( m.elements, index * 3 );
    }
    setFromEuler( e ) {
        this.x = e._x;
        this.y = e._y;
        this.z = e._z;
        return this;
    }
    setFromColor( c ) {
        this.x = c.r;
        this.y = c.g;
        this.z = c.b;
        return this;
    }
    equals( v ) {
        return ( ( v.x === this.x ) && ( v.y === this.y ) && ( v.z === this.z ) );
    }
    fromArray( array, offset = 0 ) {
        this.x = array[ offset ];
        this.y = array[ offset + 1 ];
        this.z = array[ offset + 2 ];
        return this;
    }
    toArray( array = [], offset = 0 ) {
        array[ offset ] = this.x;
        array[ offset + 1 ] = this.y;
        array[ offset + 2 ] = this.z;
        return array;
    }
    fromBufferAttribute( attribute, index ) {
        this.x = attribute.getX( index );
        this.y = attribute.getY( index );
        this.z = attribute.getZ( index );
        return this;
    }
    random() {
        this.x = Math.random();
        this.y = Math.random();
        this.z = Math.random();
        return this;
    }
    randomDirection() {
        // https://mathworld.wolfram.com/SpherePointPicking.html
        const theta = Math.random() * Math.PI * 2;
        const u = Math.random() * 2 - 1;
        const c = Math.sqrt( 1 - u * u );
        this.x = c * Math.cos( theta );
        this.y = u;
        this.z = c * Math.sin( theta );
        return this;
    }
    *[ Symbol.iterator ]() {
        yield this.x;
        yield this.y;
        yield this.z;
    }
    /// The wasm `WebVector3` — for the bindings that take one.
    _w() { return new WebVector3(this.x, this.y, this.z); }
}
export class Vector4 {
    constructor( x = 0, y = 0, z = 0, w = 1 ) {
        Vector4.prototype.isVector4 = true;
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }
    get width() {
        return this.z;
    }
    set width( value ) {
        this.z = value;
    }
    get height() {
        return this.w;
    }
    set height( value ) {
        this.w = value;
    }
    set( x, y, z, w ) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
        return this;
    }
    setScalar( scalar ) {
        this.x = scalar;
        this.y = scalar;
        this.z = scalar;
        this.w = scalar;
        return this;
    }
    setX( x ) {
        this.x = x;
        return this;
    }
    setY( y ) {
        this.y = y;
        return this;
    }
    setZ( z ) {
        this.z = z;
        return this;
    }
    setW( w ) {
        this.w = w;
        return this;
    }
    setComponent( index, value ) {
        switch ( index ) {
            case 0: this.x = value; break;
            case 1: this.y = value; break;
            case 2: this.z = value; break;
            case 3: this.w = value; break;
            default: throw new Error( 'index is out of range: ' + index );
        }
        return this;
    }
    getComponent( index ) {
        switch ( index ) {
            case 0: return this.x;
            case 1: return this.y;
            case 2: return this.z;
            case 3: return this.w;
            default: throw new Error( 'index is out of range: ' + index );
        }
    }
    clone() {
        return new this.constructor( this.x, this.y, this.z, this.w );
    }
    copy( v ) {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
        this.w = ( v.w !== undefined ) ? v.w : 1;
        return this;
    }
    add( v ) {
        this.x += v.x;
        this.y += v.y;
        this.z += v.z;
        this.w += v.w;
        return this;
    }
    addScalar( s ) {
        this.x += s;
        this.y += s;
        this.z += s;
        this.w += s;
        return this;
    }
    addVectors( a, b ) {
        this.x = a.x + b.x;
        this.y = a.y + b.y;
        this.z = a.z + b.z;
        this.w = a.w + b.w;
        return this;
    }
    addScaledVector( v, s ) {
        this.x += v.x * s;
        this.y += v.y * s;
        this.z += v.z * s;
        this.w += v.w * s;
        return this;
    }
    sub( v ) {
        this.x -= v.x;
        this.y -= v.y;
        this.z -= v.z;
        this.w -= v.w;
        return this;
    }
    subScalar( s ) {
        this.x -= s;
        this.y -= s;
        this.z -= s;
        this.w -= s;
        return this;
    }
    subVectors( a, b ) {
        this.x = a.x - b.x;
        this.y = a.y - b.y;
        this.z = a.z - b.z;
        this.w = a.w - b.w;
        return this;
    }
    multiply( v ) {
        this.x *= v.x;
        this.y *= v.y;
        this.z *= v.z;
        this.w *= v.w;
        return this;
    }
    multiplyScalar( scalar ) {
        this.x *= scalar;
        this.y *= scalar;
        this.z *= scalar;
        this.w *= scalar;
        return this;
    }
    applyMatrix4( m ) {
        const x = this.x, y = this.y, z = this.z, w = this.w;
        const e = m.elements;
        this.x = e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z + e[ 12 ] * w;
        this.y = e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z + e[ 13 ] * w;
        this.z = e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z + e[ 14 ] * w;
        this.w = e[ 3 ] * x + e[ 7 ] * y + e[ 11 ] * z + e[ 15 ] * w;
        return this;
    }
    divideScalar( scalar ) {
        return this.multiplyScalar( 1 / scalar );
    }
    setAxisAngleFromQuaternion( q ) {
        // http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
        // q is assumed to be normalized
        this.w = 2 * Math.acos( q.w );
        const s = Math.sqrt( 1 - q.w * q.w );
        if ( s < 0.0001 ) {
            this.x = 1;
            this.y = 0;
            this.z = 0;
        } else {
            this.x = q.x / s;
            this.y = q.y / s;
            this.z = q.z / s;
        }
        return this;
    }
    setAxisAngleFromRotationMatrix( m ) {
        // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
        // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)
        let angle, x, y, z; // variables for result
        const epsilon = 0.01,		// margin to allow for rounding errors
            epsilon2 = 0.1,		// margin to distinguish between 0 and 180 degrees
            te = m.elements,
            m11 = te[ 0 ], m12 = te[ 4 ], m13 = te[ 8 ],
            m21 = te[ 1 ], m22 = te[ 5 ], m23 = te[ 9 ],
            m31 = te[ 2 ], m32 = te[ 6 ], m33 = te[ 10 ];
        if ( ( Math.abs( m12 - m21 ) < epsilon ) &&
             ( Math.abs( m13 - m31 ) < epsilon ) &&
             ( Math.abs( m23 - m32 ) < epsilon ) ) {
            // singularity found
            // first check for identity matrix which must have +1 for all terms
            // in leading diagonal and zero in other terms
            if ( ( Math.abs( m12 + m21 ) < epsilon2 ) &&
                 ( Math.abs( m13 + m31 ) < epsilon2 ) &&
                 ( Math.abs( m23 + m32 ) < epsilon2 ) &&
                 ( Math.abs( m11 + m22 + m33 - 3 ) < epsilon2 ) ) {
                // this singularity is identity matrix so angle = 0
                this.set( 1, 0, 0, 0 );
                return this; // zero angle, arbitrary axis
            }
            // otherwise this singularity is angle = 180
            angle = Math.PI;
            const xx = ( m11 + 1 ) / 2;
            const yy = ( m22 + 1 ) / 2;
            const zz = ( m33 + 1 ) / 2;
            const xy = ( m12 + m21 ) / 4;
            const xz = ( m13 + m31 ) / 4;
            const yz = ( m23 + m32 ) / 4;
            if ( ( xx > yy ) && ( xx > zz ) ) {
                // m11 is the largest diagonal term
                if ( xx < epsilon ) {
                    x = 0;
                    y = 0.707106781;
                    z = 0.707106781;
                } else {
                    x = Math.sqrt( xx );
                    y = xy / x;
                    z = xz / x;
                }
            } else if ( yy > zz ) {
                // m22 is the largest diagonal term
                if ( yy < epsilon ) {
                    x = 0.707106781;
                    y = 0;
                    z = 0.707106781;
                } else {
                    y = Math.sqrt( yy );
                    x = xy / y;
                    z = yz / y;
                }
            } else {
                // m33 is the largest diagonal term so base result on this
                if ( zz < epsilon ) {
                    x = 0.707106781;
                    y = 0.707106781;
                    z = 0;
                } else {
                    z = Math.sqrt( zz );
                    x = xz / z;
                    y = yz / z;
                }
            }
            this.set( x, y, z, angle );
            return this; // return 180 deg rotation
        }
        // as we have reached here there are no singularities so we can handle normally
        let s = Math.sqrt( ( m32 - m23 ) * ( m32 - m23 ) +
            ( m13 - m31 ) * ( m13 - m31 ) +
            ( m21 - m12 ) * ( m21 - m12 ) ); // used to normalize
        if ( Math.abs( s ) < 0.001 ) s = 1;
        // prevent divide by zero, should not happen if matrix is orthogonal and should be
        // caught by singularity test above, but I've left it in just in case
        this.x = ( m32 - m23 ) / s;
        this.y = ( m13 - m31 ) / s;
        this.z = ( m21 - m12 ) / s;
        this.w = Math.acos( ( m11 + m22 + m33 - 1 ) / 2 );
        return this;
    }
    min( v ) {
        this.x = Math.min( this.x, v.x );
        this.y = Math.min( this.y, v.y );
        this.z = Math.min( this.z, v.z );
        this.w = Math.min( this.w, v.w );
        return this;
    }
    max( v ) {
        this.x = Math.max( this.x, v.x );
        this.y = Math.max( this.y, v.y );
        this.z = Math.max( this.z, v.z );
        this.w = Math.max( this.w, v.w );
        return this;
    }
    clamp( min, max ) {
        // assumes min < max, componentwise
        this.x = Math.max( min.x, Math.min( max.x, this.x ) );
        this.y = Math.max( min.y, Math.min( max.y, this.y ) );
        this.z = Math.max( min.z, Math.min( max.z, this.z ) );
        this.w = Math.max( min.w, Math.min( max.w, this.w ) );
        return this;
    }
    clampScalar( minVal, maxVal ) {
        this.x = Math.max( minVal, Math.min( maxVal, this.x ) );
        this.y = Math.max( minVal, Math.min( maxVal, this.y ) );
        this.z = Math.max( minVal, Math.min( maxVal, this.z ) );
        this.w = Math.max( minVal, Math.min( maxVal, this.w ) );
        return this;
    }
    clampLength( min, max ) {
        const length = this.length();
        return this.divideScalar( length || 1 ).multiplyScalar( Math.max( min, Math.min( max, length ) ) );
    }
    floor() {
        this.x = Math.floor( this.x );
        this.y = Math.floor( this.y );
        this.z = Math.floor( this.z );
        this.w = Math.floor( this.w );
        return this;
    }
    ceil() {
        this.x = Math.ceil( this.x );
        this.y = Math.ceil( this.y );
        this.z = Math.ceil( this.z );
        this.w = Math.ceil( this.w );
        return this;
    }
    round() {
        this.x = Math.round( this.x );
        this.y = Math.round( this.y );
        this.z = Math.round( this.z );
        this.w = Math.round( this.w );
        return this;
    }
    roundToZero() {
        this.x = Math.trunc( this.x );
        this.y = Math.trunc( this.y );
        this.z = Math.trunc( this.z );
        this.w = Math.trunc( this.w );
        return this;
    }
    negate() {
        this.x = - this.x;
        this.y = - this.y;
        this.z = - this.z;
        this.w = - this.w;
        return this;
    }
    dot( v ) {
        return this.x * v.x + this.y * v.y + this.z * v.z + this.w * v.w;
    }
    lengthSq() {
        return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
    }
    length() {
        return Math.sqrt( this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w );
    }
    manhattanLength() {
        return Math.abs( this.x ) + Math.abs( this.y ) + Math.abs( this.z ) + Math.abs( this.w );
    }
    normalize() {
        return this.divideScalar( this.length() || 1 );
    }
    setLength( length ) {
        return this.normalize().multiplyScalar( length );
    }
    lerp( v, alpha ) {
        this.x += ( v.x - this.x ) * alpha;
        this.y += ( v.y - this.y ) * alpha;
        this.z += ( v.z - this.z ) * alpha;
        this.w += ( v.w - this.w ) * alpha;
        return this;
    }
    lerpVectors( v1, v2, alpha ) {
        this.x = v1.x + ( v2.x - v1.x ) * alpha;
        this.y = v1.y + ( v2.y - v1.y ) * alpha;
        this.z = v1.z + ( v2.z - v1.z ) * alpha;
        this.w = v1.w + ( v2.w - v1.w ) * alpha;
        return this;
    }
    equals( v ) {
        return ( ( v.x === this.x ) && ( v.y === this.y ) && ( v.z === this.z ) && ( v.w === this.w ) );
    }
    fromArray( array, offset = 0 ) {
        this.x = array[ offset ];
        this.y = array[ offset + 1 ];
        this.z = array[ offset + 2 ];
        this.w = array[ offset + 3 ];
        return this;
    }
    toArray( array = [], offset = 0 ) {
        array[ offset ] = this.x;
        array[ offset + 1 ] = this.y;
        array[ offset + 2 ] = this.z;
        array[ offset + 3 ] = this.w;
        return array;
    }
    fromBufferAttribute( attribute, index ) {
        this.x = attribute.getX( index );
        this.y = attribute.getY( index );
        this.z = attribute.getZ( index );
        this.w = attribute.getW( index );
        return this;
    }
    random() {
        this.x = Math.random();
        this.y = Math.random();
        this.z = Math.random();
        this.w = Math.random();
        return this;
    }
    *[ Symbol.iterator ]() {
        yield this.x;
        yield this.y;
        yield this.z;
        yield this.w;
    }
    /// The wasm `WebVector4` — for the bindings that take one.
    _w() { return new WebVector4(this.x, this.y, this.z, this.w); }
}

/// three.js r165 `Euler`: angles in `order`, kept in step with the owning
/// object's quaternion through `_onChange` (as three.js's Object3D does).
export class Euler {
    constructor( x = 0, y = 0, z = 0, order = Euler.DEFAULT_ORDER ) {
        this.isEuler = true;
        this._x = x;
        this._y = y;
        this._z = z;
        this._order = order;
    }
    get x() {
        return this._x;
    }
    set x( value ) {
        this._x = value;
        this._onChangeCallback();
    }
    get y() {
        return this._y;
    }
    set y( value ) {
        this._y = value;
        this._onChangeCallback();
    }
    get z() {
        return this._z;
    }
    set z( value ) {
        this._z = value;
        this._onChangeCallback();
    }
    get order() {
        return this._order;
    }
    set order( value ) {
        this._order = value;
        this._onChangeCallback();
    }
    set( x, y, z, order = this._order ) {
        this._x = x;
        this._y = y;
        this._z = z;
        this._order = order;
        this._onChangeCallback();
        return this;
    }
    clone() {
        return new this.constructor( this._x, this._y, this._z, this._order );
    }
    copy( euler ) {
        // A plain {x, y, z} keeps this order.
        this._x = euler._x ?? euler.x;
        this._y = euler._y ?? euler.y;
        this._z = euler._z ?? euler.z;
        this._order = euler._order ?? euler.order ?? this._order;
        this._onChangeCallback();
        return this;
    }
    setFromRotationMatrix( m, order = this._order, update = true ) {
        // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)
        const te = m.elements;
        const m11 = te[ 0 ], m12 = te[ 4 ], m13 = te[ 8 ];
        const m21 = te[ 1 ], m22 = te[ 5 ], m23 = te[ 9 ];
        const m31 = te[ 2 ], m32 = te[ 6 ], m33 = te[ 10 ];
        switch ( order ) {
            case 'XYZ':
                this._y = Math.asin( _clamp( m13, - 1, 1 ) );
                if ( Math.abs( m13 ) < 0.9999999 ) {
                    this._x = Math.atan2( - m23, m33 );
                    this._z = Math.atan2( - m12, m11 );
                } else {
                    this._x = Math.atan2( m32, m22 );
                    this._z = 0;
                }
                break;
            case 'YXZ':
                this._x = Math.asin( - _clamp( m23, - 1, 1 ) );
                if ( Math.abs( m23 ) < 0.9999999 ) {
                    this._y = Math.atan2( m13, m33 );
                    this._z = Math.atan2( m21, m22 );
                } else {
                    this._y = Math.atan2( - m31, m11 );
                    this._z = 0;
                }
                break;
            case 'ZXY':
                this._x = Math.asin( _clamp( m32, - 1, 1 ) );
                if ( Math.abs( m32 ) < 0.9999999 ) {
                    this._y = Math.atan2( - m31, m33 );
                    this._z = Math.atan2( - m12, m22 );
                } else {
                    this._y = 0;
                    this._z = Math.atan2( m21, m11 );
                }
                break;
            case 'ZYX':
                this._y = Math.asin( - _clamp( m31, - 1, 1 ) );
                if ( Math.abs( m31 ) < 0.9999999 ) {
                    this._x = Math.atan2( m32, m33 );
                    this._z = Math.atan2( m21, m11 );
                } else {
                    this._x = 0;
                    this._z = Math.atan2( - m12, m22 );
                }
                break;
            case 'YZX':
                this._z = Math.asin( _clamp( m21, - 1, 1 ) );
                if ( Math.abs( m21 ) < 0.9999999 ) {
                    this._x = Math.atan2( - m23, m22 );
                    this._y = Math.atan2( - m31, m11 );
                } else {
                    this._x = 0;
                    this._y = Math.atan2( m13, m33 );
                }
                break;
            case 'XZY':
                this._z = Math.asin( - _clamp( m12, - 1, 1 ) );
                if ( Math.abs( m12 ) < 0.9999999 ) {
                    this._x = Math.atan2( m32, m22 );
                    this._y = Math.atan2( m13, m11 );
                } else {
                    this._x = Math.atan2( - m23, m33 );
                    this._y = 0;
                }
                break;
            default:
                console.warn( 'THREE.Euler: .setFromRotationMatrix() encountered an unknown order: ' + order );
        }
        this._order = order;
        if ( update === true ) this._onChangeCallback();
        return this;
    }
    setFromQuaternion( q, order, update ) {
        _eulerMatrix.makeRotationFromQuaternion( q );
        return this.setFromRotationMatrix( _eulerMatrix, order, update );
    }
    setFromVector3( v, order = this._order ) {
        return this.set( v.x, v.y, v.z, order );
    }
    reorder( newOrder ) {
        // WARNING: this discards revolution information -bhouston
        _eulerQuaternion.setFromEuler( this );
        return this.setFromQuaternion( _eulerQuaternion, newOrder );
    }
    equals( euler ) {
        return ( euler._x === this._x ) && ( euler._y === this._y ) && ( euler._z === this._z ) && ( euler._order === this._order );
    }
    fromArray( array ) {
        this._x = array[ 0 ];
        this._y = array[ 1 ];
        this._z = array[ 2 ];
        if ( array[ 3 ] !== undefined ) this._order = array[ 3 ];
        this._onChangeCallback();
        return this;
    }
    toArray( array = [], offset = 0 ) {
        array[ offset ] = this._x;
        array[ offset + 1 ] = this._y;
        array[ offset + 2 ] = this._z;
        array[ offset + 3 ] = this._order;
        return array;
    }
    _onChange( callback ) {
        this._onChangeCallback = callback;
        return this;
    }
    _onChangeCallback() {}
    *[ Symbol.iterator ]() {
        yield this._x;
        yield this._y;
        yield this._z;
        yield this._order;
    }
}
Euler.DEFAULT_ORDER = 'XYZ';

/// three.js r165 `Quaternion`, with `_onChange` notification.
export class Quaternion {
    constructor( x = 0, y = 0, z = 0, w = 1 ) {
        this.isQuaternion = true;
        this._x = x;
        this._y = y;
        this._z = z;
        this._w = w;
    }
    static slerpFlat( dst, dstOffset, src0, srcOffset0, src1, srcOffset1, t ) {
        // fuzz-free, array-based Quaternion SLERP operation
        let x0 = src0[ srcOffset0 + 0 ],
            y0 = src0[ srcOffset0 + 1 ],
            z0 = src0[ srcOffset0 + 2 ],
            w0 = src0[ srcOffset0 + 3 ];
        const x1 = src1[ srcOffset1 + 0 ],
            y1 = src1[ srcOffset1 + 1 ],
            z1 = src1[ srcOffset1 + 2 ],
            w1 = src1[ srcOffset1 + 3 ];
        if ( t === 0 ) {
            dst[ dstOffset + 0 ] = x0;
            dst[ dstOffset + 1 ] = y0;
            dst[ dstOffset + 2 ] = z0;
            dst[ dstOffset + 3 ] = w0;
            return;
        }
        if ( t === 1 ) {
            dst[ dstOffset + 0 ] = x1;
            dst[ dstOffset + 1 ] = y1;
            dst[ dstOffset + 2 ] = z1;
            dst[ dstOffset + 3 ] = w1;
            return;
        }
        if ( w0 !== w1 || x0 !== x1 || y0 !== y1 || z0 !== z1 ) {
            let s = 1 - t;
            const cos = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1,
                dir = ( cos >= 0 ? 1 : - 1 ),
                sqrSin = 1 - cos * cos;
            // Skip the Slerp for tiny steps to avoid numeric problems:
            if ( sqrSin > Number.EPSILON ) {
                const sin = Math.sqrt( sqrSin ),
                    len = Math.atan2( sin, cos * dir );
                s = Math.sin( s * len ) / sin;
                t = Math.sin( t * len ) / sin;
            }
            const tDir = t * dir;
            x0 = x0 * s + x1 * tDir;
            y0 = y0 * s + y1 * tDir;
            z0 = z0 * s + z1 * tDir;
            w0 = w0 * s + w1 * tDir;
            // Normalize in case we just did a lerp:
            if ( s === 1 - t ) {
                const f = 1 / Math.sqrt( x0 * x0 + y0 * y0 + z0 * z0 + w0 * w0 );
                x0 *= f;
                y0 *= f;
                z0 *= f;
                w0 *= f;
            }
        }
        dst[ dstOffset ] = x0;
        dst[ dstOffset + 1 ] = y0;
        dst[ dstOffset + 2 ] = z0;
        dst[ dstOffset + 3 ] = w0;
    }
    static multiplyQuaternionsFlat( dst, dstOffset, src0, srcOffset0, src1, srcOffset1 ) {
        const x0 = src0[ srcOffset0 ];
        const y0 = src0[ srcOffset0 + 1 ];
        const z0 = src0[ srcOffset0 + 2 ];
        const w0 = src0[ srcOffset0 + 3 ];
        const x1 = src1[ srcOffset1 ];
        const y1 = src1[ srcOffset1 + 1 ];
        const z1 = src1[ srcOffset1 + 2 ];
        const w1 = src1[ srcOffset1 + 3 ];
        dst[ dstOffset ] = x0 * w1 + w0 * x1 + y0 * z1 - z0 * y1;
        dst[ dstOffset + 1 ] = y0 * w1 + w0 * y1 + z0 * x1 - x0 * z1;
        dst[ dstOffset + 2 ] = z0 * w1 + w0 * z1 + x0 * y1 - y0 * x1;
        dst[ dstOffset + 3 ] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1;
        return dst;
    }
    get x() {
        return this._x;
    }
    set x( value ) {
        this._x = value;
        this._onChangeCallback();
    }
    get y() {
        return this._y;
    }
    set y( value ) {
        this._y = value;
        this._onChangeCallback();
    }
    get z() {
        return this._z;
    }
    set z( value ) {
        this._z = value;
        this._onChangeCallback();
    }
    get w() {
        return this._w;
    }
    set w( value ) {
        this._w = value;
        this._onChangeCallback();
    }
    set( x, y, z, w ) {
        this._x = x;
        this._y = y;
        this._z = z;
        this._w = w;
        this._onChangeCallback();
        return this;
    }
    clone() {
        return new this.constructor( this._x, this._y, this._z, this._w );
    }
    copy( quaternion ) {
        this._x = quaternion.x;
        this._y = quaternion.y;
        this._z = quaternion.z;
        this._w = quaternion.w;
        this._onChangeCallback();
        return this;
    }
    setFromEuler( euler, update = true ) {
        // A plain {x, y, z} reads as an XYZ Euler.
        const x = euler._x ?? euler.x, y = euler._y ?? euler.y, z = euler._z ?? euler.z, order = euler._order ?? euler.order ?? 'XYZ';
        // http://www.mathworks.com/matlabcentral/fileexchange/
        // 	20696-function-to-convert-between-dcm-euler-angles-quaternions-and-euler-vectors/
        //	content/SpinCalc.m
        const cos = Math.cos;
        const sin = Math.sin;
        const c1 = cos( x / 2 );
        const c2 = cos( y / 2 );
        const c3 = cos( z / 2 );
        const s1 = sin( x / 2 );
        const s2 = sin( y / 2 );
        const s3 = sin( z / 2 );
        switch ( order ) {
            case 'XYZ':
                this._x = s1 * c2 * c3 + c1 * s2 * s3;
                this._y = c1 * s2 * c3 - s1 * c2 * s3;
                this._z = c1 * c2 * s3 + s1 * s2 * c3;
                this._w = c1 * c2 * c3 - s1 * s2 * s3;
                break;
            case 'YXZ':
                this._x = s1 * c2 * c3 + c1 * s2 * s3;
                this._y = c1 * s2 * c3 - s1 * c2 * s3;
                this._z = c1 * c2 * s3 - s1 * s2 * c3;
                this._w = c1 * c2 * c3 + s1 * s2 * s3;
                break;
            case 'ZXY':
                this._x = s1 * c2 * c3 - c1 * s2 * s3;
                this._y = c1 * s2 * c3 + s1 * c2 * s3;
                this._z = c1 * c2 * s3 + s1 * s2 * c3;
                this._w = c1 * c2 * c3 - s1 * s2 * s3;
                break;
            case 'ZYX':
                this._x = s1 * c2 * c3 - c1 * s2 * s3;
                this._y = c1 * s2 * c3 + s1 * c2 * s3;
                this._z = c1 * c2 * s3 - s1 * s2 * c3;
                this._w = c1 * c2 * c3 + s1 * s2 * s3;
                break;
            case 'YZX':
                this._x = s1 * c2 * c3 + c1 * s2 * s3;
                this._y = c1 * s2 * c3 + s1 * c2 * s3;
                this._z = c1 * c2 * s3 - s1 * s2 * c3;
                this._w = c1 * c2 * c3 - s1 * s2 * s3;
                break;
            case 'XZY':
                this._x = s1 * c2 * c3 - c1 * s2 * s3;
                this._y = c1 * s2 * c3 - s1 * c2 * s3;
                this._z = c1 * c2 * s3 + s1 * s2 * c3;
                this._w = c1 * c2 * c3 + s1 * s2 * s3;
                break;
            default:
                console.warn( 'THREE.Quaternion: .setFromEuler() encountered an unknown order: ' + order );
        }
        if ( update === true ) this._onChangeCallback();
        return this;
    }
    setFromAxisAngle( axis, angle ) {
        // http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
        // assumes axis is normalized
        const halfAngle = angle / 2, s = Math.sin( halfAngle );
        this._x = axis.x * s;
        this._y = axis.y * s;
        this._z = axis.z * s;
        this._w = Math.cos( halfAngle );
        this._onChangeCallback();
        return this;
    }
    setFromRotationMatrix( m ) {
        // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
        // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)
        const te = m.elements,
            m11 = te[ 0 ], m12 = te[ 4 ], m13 = te[ 8 ],
            m21 = te[ 1 ], m22 = te[ 5 ], m23 = te[ 9 ],
            m31 = te[ 2 ], m32 = te[ 6 ], m33 = te[ 10 ],
            trace = m11 + m22 + m33;
        if ( trace > 0 ) {
            const s = 0.5 / Math.sqrt( trace + 1.0 );
            this._w = 0.25 / s;
            this._x = ( m32 - m23 ) * s;
            this._y = ( m13 - m31 ) * s;
            this._z = ( m21 - m12 ) * s;
        } else if ( m11 > m22 && m11 > m33 ) {
            const s = 2.0 * Math.sqrt( 1.0 + m11 - m22 - m33 );
            this._w = ( m32 - m23 ) / s;
            this._x = 0.25 * s;
            this._y = ( m12 + m21 ) / s;
            this._z = ( m13 + m31 ) / s;
        } else if ( m22 > m33 ) {
            const s = 2.0 * Math.sqrt( 1.0 + m22 - m11 - m33 );
            this._w = ( m13 - m31 ) / s;
            this._x = ( m12 + m21 ) / s;
            this._y = 0.25 * s;
            this._z = ( m23 + m32 ) / s;
        } else {
            const s = 2.0 * Math.sqrt( 1.0 + m33 - m11 - m22 );
            this._w = ( m21 - m12 ) / s;
            this._x = ( m13 + m31 ) / s;
            this._y = ( m23 + m32 ) / s;
            this._z = 0.25 * s;
        }
        this._onChangeCallback();
        return this;
    }
    setFromUnitVectors( vFrom, vTo ) {
        // assumes direction vectors vFrom and vTo are normalized
        let r = vFrom.dot( vTo ) + 1;
        if ( r < Number.EPSILON ) {
            // vFrom and vTo point in opposite directions
            r = 0;
            if ( Math.abs( vFrom.x ) > Math.abs( vFrom.z ) ) {
                this._x = - vFrom.y;
                this._y = vFrom.x;
                this._z = 0;
                this._w = r;
            } else {
                this._x = 0;
                this._y = - vFrom.z;
                this._z = vFrom.y;
                this._w = r;
            }
        } else {
            // crossVectors( vFrom, vTo ); // inlined to avoid cyclic dependency on Vector3
            this._x = vFrom.y * vTo.z - vFrom.z * vTo.y;
            this._y = vFrom.z * vTo.x - vFrom.x * vTo.z;
            this._z = vFrom.x * vTo.y - vFrom.y * vTo.x;
            this._w = r;
        }
        return this.normalize();
    }
    angleTo( q ) {
        return 2 * Math.acos( Math.abs( _clamp( this.dot( q ), - 1, 1 ) ) );
    }
    rotateTowards( q, step ) {
        const angle = this.angleTo( q );
        if ( angle === 0 ) return this;
        const t = Math.min( 1, step / angle );
        this.slerp( q, t );
        return this;
    }
    identity() {
        return this.set( 0, 0, 0, 1 );
    }
    invert() {
        // quaternion is assumed to have unit length
        return this.conjugate();
    }
    conjugate() {
        this._x *= - 1;
        this._y *= - 1;
        this._z *= - 1;
        this._onChangeCallback();
        return this;
    }
    dot( v ) {
        return this._x * v._x + this._y * v._y + this._z * v._z + this._w * v._w;
    }
    lengthSq() {
        return this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w;
    }
    length() {
        return Math.sqrt( this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w );
    }
    normalize() {
        let l = this.length();
        if ( l === 0 ) {
            this._x = 0;
            this._y = 0;
            this._z = 0;
            this._w = 1;
        } else {
            l = 1 / l;
            this._x = this._x * l;
            this._y = this._y * l;
            this._z = this._z * l;
            this._w = this._w * l;
        }
        this._onChangeCallback();
        return this;
    }
    multiply( q ) {
        return this.multiplyQuaternions( this, q );
    }
    premultiply( q ) {
        return this.multiplyQuaternions( q, this );
    }
    multiplyQuaternions( a, b ) {
        // from http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/code/index.htm
        const qax = a._x, qay = a._y, qaz = a._z, qaw = a._w;
        const qbx = b._x, qby = b._y, qbz = b._z, qbw = b._w;
        this._x = qax * qbw + qaw * qbx + qay * qbz - qaz * qby;
        this._y = qay * qbw + qaw * qby + qaz * qbx - qax * qbz;
        this._z = qaz * qbw + qaw * qbz + qax * qby - qay * qbx;
        this._w = qaw * qbw - qax * qbx - qay * qby - qaz * qbz;
        this._onChangeCallback();
        return this;
    }
    slerp( qb, t ) {
        if ( t === 0 ) return this;
        if ( t === 1 ) return this.copy( qb );
        const x = this._x, y = this._y, z = this._z, w = this._w;
        // http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/
        let cosHalfTheta = w * qb._w + x * qb._x + y * qb._y + z * qb._z;
        if ( cosHalfTheta < 0 ) {
            this._w = - qb._w;
            this._x = - qb._x;
            this._y = - qb._y;
            this._z = - qb._z;
            cosHalfTheta = - cosHalfTheta;
        } else {
            this.copy( qb );
        }
        if ( cosHalfTheta >= 1.0 ) {
            this._w = w;
            this._x = x;
            this._y = y;
            this._z = z;
            return this;
        }
        const sqrSinHalfTheta = 1.0 - cosHalfTheta * cosHalfTheta;
        if ( sqrSinHalfTheta <= Number.EPSILON ) {
            const s = 1 - t;
            this._w = s * w + t * this._w;
            this._x = s * x + t * this._x;
            this._y = s * y + t * this._y;
            this._z = s * z + t * this._z;
            this.normalize(); // normalize calls _onChangeCallback()
            return this;
        }
        const sinHalfTheta = Math.sqrt( sqrSinHalfTheta );
        const halfTheta = Math.atan2( sinHalfTheta, cosHalfTheta );
        const ratioA = Math.sin( ( 1 - t ) * halfTheta ) / sinHalfTheta,
            ratioB = Math.sin( t * halfTheta ) / sinHalfTheta;
        this._w = ( w * ratioA + this._w * ratioB );
        this._x = ( x * ratioA + this._x * ratioB );
        this._y = ( y * ratioA + this._y * ratioB );
        this._z = ( z * ratioA + this._z * ratioB );
        this._onChangeCallback();
        return this;
    }
    slerpQuaternions( qa, qb, t ) {
        return this.copy( qa ).slerp( qb, t );
    }
    random() {
        // sets this quaternion to a uniform random unit quaternnion
        // Ken Shoemake
        // Uniform random rotations
        // D. Kirk, editor, Graphics Gems III, pages 124-132. Academic Press, New York, 1992.
        const theta1 = 2 * Math.PI * Math.random();
        const theta2 = 2 * Math.PI * Math.random();
        const x0 = Math.random();
        const r1 = Math.sqrt( 1 - x0 );
        const r2 = Math.sqrt( x0 );
        return this.set(
            r1 * Math.sin( theta1 ),
            r1 * Math.cos( theta1 ),
            r2 * Math.sin( theta2 ),
            r2 * Math.cos( theta2 ),
        );
    }
    equals( quaternion ) {
        return ( quaternion._x === this._x ) && ( quaternion._y === this._y ) && ( quaternion._z === this._z ) && ( quaternion._w === this._w );
    }
    fromArray( array, offset = 0 ) {
        this._x = array[ offset ];
        this._y = array[ offset + 1 ];
        this._z = array[ offset + 2 ];
        this._w = array[ offset + 3 ];
        this._onChangeCallback();
        return this;
    }
    toArray( array = [], offset = 0 ) {
        array[ offset ] = this._x;
        array[ offset + 1 ] = this._y;
        array[ offset + 2 ] = this._z;
        array[ offset + 3 ] = this._w;
        return array;
    }
    fromBufferAttribute( attribute, index ) {
        this._x = attribute.getX( index );
        this._y = attribute.getY( index );
        this._z = attribute.getZ( index );
        this._w = attribute.getW( index );
        this._onChangeCallback();
        return this;
    }
    toJSON() {
        return this.toArray();
    }
    _onChange( callback ) {
        this._onChangeCallback = callback;
        return this;
    }
    _onChangeCallback() {}
    *[ Symbol.iterator ]() {
        yield this._x;
        yield this._y;
        yield this._z;
        yield this._w;
    }
}

export class Matrix3 {
    constructor( n11, n12, n13, n21, n22, n23, n31, n32, n33 ) {
        Matrix3.prototype.isMatrix3 = true;
        this.elements = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ];
        if ( n11 !== undefined ) {
            this.set( n11, n12, n13, n21, n22, n23, n31, n32, n33 );
        }
    }
    set( n11, n12, n13, n21, n22, n23, n31, n32, n33 ) {
        const te = this.elements;
        te[ 0 ] = n11; te[ 1 ] = n21; te[ 2 ] = n31;
        te[ 3 ] = n12; te[ 4 ] = n22; te[ 5 ] = n32;
        te[ 6 ] = n13; te[ 7 ] = n23; te[ 8 ] = n33;
        return this;
    }
    identity() {
        this.set(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        );
        return this;
    }
    copy( m ) {
        const te = this.elements;
        const me = m.elements;
        te[ 0 ] = me[ 0 ]; te[ 1 ] = me[ 1 ]; te[ 2 ] = me[ 2 ];
        te[ 3 ] = me[ 3 ]; te[ 4 ] = me[ 4 ]; te[ 5 ] = me[ 5 ];
        te[ 6 ] = me[ 6 ]; te[ 7 ] = me[ 7 ]; te[ 8 ] = me[ 8 ];
        return this;
    }
    extractBasis( xAxis, yAxis, zAxis ) {
        xAxis.setFromMatrix3Column( this, 0 );
        yAxis.setFromMatrix3Column( this, 1 );
        zAxis.setFromMatrix3Column( this, 2 );
        return this;
    }
    setFromMatrix4( m ) {
        const me = m.elements;
        this.set(
            me[ 0 ], me[ 4 ], me[ 8 ],
            me[ 1 ], me[ 5 ], me[ 9 ],
            me[ 2 ], me[ 6 ], me[ 10 ]
        );
        return this;
    }
    multiply( m ) {
        return this.multiplyMatrices( this, m );
    }
    premultiply( m ) {
        return this.multiplyMatrices( m, this );
    }
    multiplyMatrices( a, b ) {
        const ae = a.elements;
        const be = b.elements;
        const te = this.elements;
        const a11 = ae[ 0 ], a12 = ae[ 3 ], a13 = ae[ 6 ];
        const a21 = ae[ 1 ], a22 = ae[ 4 ], a23 = ae[ 7 ];
        const a31 = ae[ 2 ], a32 = ae[ 5 ], a33 = ae[ 8 ];
        const b11 = be[ 0 ], b12 = be[ 3 ], b13 = be[ 6 ];
        const b21 = be[ 1 ], b22 = be[ 4 ], b23 = be[ 7 ];
        const b31 = be[ 2 ], b32 = be[ 5 ], b33 = be[ 8 ];
        te[ 0 ] = a11 * b11 + a12 * b21 + a13 * b31;
        te[ 3 ] = a11 * b12 + a12 * b22 + a13 * b32;
        te[ 6 ] = a11 * b13 + a12 * b23 + a13 * b33;
        te[ 1 ] = a21 * b11 + a22 * b21 + a23 * b31;
        te[ 4 ] = a21 * b12 + a22 * b22 + a23 * b32;
        te[ 7 ] = a21 * b13 + a22 * b23 + a23 * b33;
        te[ 2 ] = a31 * b11 + a32 * b21 + a33 * b31;
        te[ 5 ] = a31 * b12 + a32 * b22 + a33 * b32;
        te[ 8 ] = a31 * b13 + a32 * b23 + a33 * b33;
        return this;
    }
    multiplyScalar( s ) {
        const te = this.elements;
        te[ 0 ] *= s; te[ 3 ] *= s; te[ 6 ] *= s;
        te[ 1 ] *= s; te[ 4 ] *= s; te[ 7 ] *= s;
        te[ 2 ] *= s; te[ 5 ] *= s; te[ 8 ] *= s;
        return this;
    }
    determinant() {
        const te = this.elements;
        const a = te[ 0 ], b = te[ 1 ], c = te[ 2 ],
            d = te[ 3 ], e = te[ 4 ], f = te[ 5 ],
            g = te[ 6 ], h = te[ 7 ], i = te[ 8 ];
        return a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g;
    }
    invert() {
        const te = this.elements,
            n11 = te[ 0 ], n21 = te[ 1 ], n31 = te[ 2 ],
            n12 = te[ 3 ], n22 = te[ 4 ], n32 = te[ 5 ],
            n13 = te[ 6 ], n23 = te[ 7 ], n33 = te[ 8 ],
            t11 = n33 * n22 - n32 * n23,
            t12 = n32 * n13 - n33 * n12,
            t13 = n23 * n12 - n22 * n13,
            det = n11 * t11 + n21 * t12 + n31 * t13;
        if ( det === 0 ) return this.set( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
        const detInv = 1 / det;
        te[ 0 ] = t11 * detInv;
        te[ 1 ] = ( n31 * n23 - n33 * n21 ) * detInv;
        te[ 2 ] = ( n32 * n21 - n31 * n22 ) * detInv;
        te[ 3 ] = t12 * detInv;
        te[ 4 ] = ( n33 * n11 - n31 * n13 ) * detInv;
        te[ 5 ] = ( n31 * n12 - n32 * n11 ) * detInv;
        te[ 6 ] = t13 * detInv;
        te[ 7 ] = ( n21 * n13 - n23 * n11 ) * detInv;
        te[ 8 ] = ( n22 * n11 - n21 * n12 ) * detInv;
        return this;
    }
    transpose() {
        let tmp;
        const m = this.elements;
        tmp = m[ 1 ]; m[ 1 ] = m[ 3 ]; m[ 3 ] = tmp;
        tmp = m[ 2 ]; m[ 2 ] = m[ 6 ]; m[ 6 ] = tmp;
        tmp = m[ 5 ]; m[ 5 ] = m[ 7 ]; m[ 7 ] = tmp;
        return this;
    }
    getNormalMatrix( matrix4 ) {
        return this.setFromMatrix4( matrix4 ).invert().transpose();
    }
    transposeIntoArray( r ) {
        const m = this.elements;
        r[ 0 ] = m[ 0 ];
        r[ 1 ] = m[ 3 ];
        r[ 2 ] = m[ 6 ];
        r[ 3 ] = m[ 1 ];
        r[ 4 ] = m[ 4 ];
        r[ 5 ] = m[ 7 ];
        r[ 6 ] = m[ 2 ];
        r[ 7 ] = m[ 5 ];
        r[ 8 ] = m[ 8 ];
        return this;
    }
    setUvTransform( tx, ty, sx, sy, rotation, cx, cy ) {
        const c = Math.cos( rotation );
        const s = Math.sin( rotation );
        this.set(
            sx * c, sx * s, - sx * ( c * cx + s * cy ) + cx + tx,
            - sy * s, sy * c, - sy * ( - s * cx + c * cy ) + cy + ty,
            0, 0, 1
        );
        return this;
    }
    //
    scale( sx, sy ) {
        this.premultiply( _m3_m3.makeScale( sx, sy ) );
        return this;
    }
    rotate( theta ) {
        this.premultiply( _m3_m3.makeRotation( - theta ) );
        return this;
    }
    translate( tx, ty ) {
        this.premultiply( _m3_m3.makeTranslation( tx, ty ) );
        return this;
    }
    // for 2D Transforms
    makeTranslation( x, y ) {
        if ( x.isVector2 ) {
            this.set(
                1, 0, x.x,
                0, 1, x.y,
                0, 0, 1
            );
        } else {
            this.set(
                1, 0, x,
                0, 1, y,
                0, 0, 1
            );
        }
        return this;
    }
    makeRotation( theta ) {
        // counterclockwise
        const c = Math.cos( theta );
        const s = Math.sin( theta );
        this.set(
            c, - s, 0,
            s, c, 0,
            0, 0, 1
        );
        return this;
    }
    makeScale( x, y ) {
        this.set(
            x, 0, 0,
            0, y, 0,
            0, 0, 1
        );
        return this;
    }
    //
    equals( matrix ) {
        const te = this.elements;
        const me = matrix.elements;
        for ( let i = 0; i < 9; i ++ ) {
            if ( te[ i ] !== me[ i ] ) return false;
        }
        return true;
    }
    fromArray( array, offset = 0 ) {
        for ( let i = 0; i < 9; i ++ ) {
            this.elements[ i ] = array[ i + offset ];
        }
        return this;
    }
    toArray( array = [], offset = 0 ) {
        const te = this.elements;
        array[ offset ] = te[ 0 ];
        array[ offset + 1 ] = te[ 1 ];
        array[ offset + 2 ] = te[ 2 ];
        array[ offset + 3 ] = te[ 3 ];
        array[ offset + 4 ] = te[ 4 ];
        array[ offset + 5 ] = te[ 5 ];
        array[ offset + 6 ] = te[ 6 ];
        array[ offset + 7 ] = te[ 7 ];
        array[ offset + 8 ] = te[ 8 ];
        return array;
    }
    clone() {
        return new this.constructor().fromArray( this.elements );
    }
}

export class Matrix4 {
    constructor() {
        this.elements = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
    }
    set(n11, n12, n13, n14, n21, n22, n23, n24, n31, n32, n33, n34, n41, n42, n43, n44) {
        const e = this.elements;
        e[0]=n11; e[4]=n12; e[8]=n13; e[12]=n14;
        e[1]=n21; e[5]=n22; e[9]=n23; e[13]=n24;
        e[2]=n31; e[6]=n32; e[10]=n33; e[14]=n34;
        e[3]=n41; e[7]=n42; e[11]=n43; e[15]=n44;
        return this;
    }
    identity() { return this.set(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1); }
    setPosition(x, y, z) {
        if (typeof x === 'object') { y = x.y; z = x.z; x = x.x; }
        this.elements[12] = x; this.elements[13] = y; this.elements[14] = z;
        return this;
    }
    copy(m) { for (let i = 0; i < 16; i++) this.elements[i] = m.elements[i]; return this; }
    clone() { return new Matrix4().fromArray(this.elements); }
    fromArray(a, off = 0) { for (let i = 0; i < 16; i++) this.elements[i] = a[off + i]; return this; }
    toArray(a = [], off = 0) { for (let i = 0; i < 16; i++) a[off + i] = this.elements[i]; return a; }
    makeTranslation(x, y, z) {
        if (typeof x === 'object') { y = x.y; z = x.z; x = x.x; }
        return this.set(1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1);
    }
    makeScale(x, y, z) { return this.set(x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1); }
    makeRotationX(t) { const c = Math.cos(t), s = Math.sin(t); return this.set(1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1); }
    makeRotationY(t) { const c = Math.cos(t), s = Math.sin(t); return this.set(c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1); }
    makeRotationZ(t) { const c = Math.cos(t), s = Math.sin(t); return this.set(c,-s,0,0, s,c,0,0, 0,0,1,0, 0,0,0,1); }
    makeRotationAxis(axis, angle) {
        const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
        const x = axis.x, y = axis.y, z = axis.z;
        return this.set(
            t*x*x + c,    t*x*y - s*z,  t*x*z + s*y,  0,
            t*x*y + s*z,  t*y*y + c,    t*y*z - s*x,  0,
            t*x*z - s*y,  t*y*z + s*x,  t*z*z + c,    0,
            0,            0,            0,            1,
        );
    }
    makeRotationFromQuaternion(q) {
        const x = q.x, y = q.y, z = q.z, w = q.w;
        const x2 = x + x, y2 = y + y, z2 = z + z;
        const xx = x * x2, xy = x * y2, xz = x * z2;
        const yy = y * y2, yz = y * z2, zz = z * z2;
        const wx = w * x2, wy = w * y2, wz = w * z2;
        return this.set(
            1 - (yy + zz), xy - wz,       xz + wy,       0,
            xy + wz,       1 - (xx + zz), yz - wx,       0,
            xz - wy,       yz + wx,       1 - (xx + yy), 0,
            0,             0,             0,             1
        );
    }
    makePerspective(fovDeg_or_left, aspect_or_right, near_or_top, far_or_bottom, near_, far_) {
        // Support both signatures: (fov, aspect, near, far) and (left, right, top, bottom, near, far).
        if (arguments.length === 4) {
            const fov = fovDeg_or_left, aspect = aspect_or_right, near = near_or_top, far = far_or_bottom;
            const f = 1 / Math.tan(fov * 0.5 * Math.PI / 180);
            return this.set(
                f/aspect, 0, 0, 0,
                0,        f, 0, 0,
                0,        0, (far+near)/(near-far), (2*far*near)/(near-far),
                0,        0, -1, 0,
            );
        }
        // 6-arg form
        const l = fovDeg_or_left, r = aspect_or_right, t = near_or_top, b = far_or_bottom, n = near_, fa = far_;
        return this.set(
            2*n/(r-l), 0, (r+l)/(r-l), 0,
            0, 2*n/(t-b), (t+b)/(t-b), 0,
            0, 0, -(fa+n)/(fa-n), -2*fa*n/(fa-n),
            0, 0, -1, 0,
        );
    }
    makeOrthographic(l, r, t, b, n, f) {
        return this.set(
            2/(r-l), 0, 0, -(r+l)/(r-l),
            0, 2/(t-b), 0, -(t+b)/(t-b),
            0, 0, -2/(f-n), -(f+n)/(f-n),
            0, 0, 0, 1,
        );
    }
    multiply(m) { return this.multiplyMatrices(this, m); }
    premultiply(m) { return this.multiplyMatrices(m, this); }
    multiplyMatrices(a, b) {
        const ae = a.elements, be = b.elements, te = this.elements;
        const a11=ae[0], a12=ae[4], a13=ae[8], a14=ae[12];
        const a21=ae[1], a22=ae[5], a23=ae[9], a24=ae[13];
        const a31=ae[2], a32=ae[6], a33=ae[10],a34=ae[14];
        const a41=ae[3], a42=ae[7], a43=ae[11],a44=ae[15];
        const b11=be[0], b12=be[4], b13=be[8], b14=be[12];
        const b21=be[1], b22=be[5], b23=be[9], b24=be[13];
        const b31=be[2], b32=be[6], b33=be[10],b34=be[14];
        const b41=be[3], b42=be[7], b43=be[11],b44=be[15];
        te[0]  = a11*b11 + a12*b21 + a13*b31 + a14*b41;
        te[4]  = a11*b12 + a12*b22 + a13*b32 + a14*b42;
        te[8]  = a11*b13 + a12*b23 + a13*b33 + a14*b43;
        te[12] = a11*b14 + a12*b24 + a13*b34 + a14*b44;
        te[1]  = a21*b11 + a22*b21 + a23*b31 + a24*b41;
        te[5]  = a21*b12 + a22*b22 + a23*b32 + a24*b42;
        te[9]  = a21*b13 + a22*b23 + a23*b33 + a24*b43;
        te[13] = a21*b14 + a22*b24 + a23*b34 + a24*b44;
        te[2]  = a31*b11 + a32*b21 + a33*b31 + a34*b41;
        te[6]  = a31*b12 + a32*b22 + a33*b32 + a34*b42;
        te[10] = a31*b13 + a32*b23 + a33*b33 + a34*b43;
        te[14] = a31*b14 + a32*b24 + a33*b34 + a34*b44;
        te[3]  = a41*b11 + a42*b21 + a43*b31 + a44*b41;
        te[7]  = a41*b12 + a42*b22 + a43*b32 + a44*b42;
        te[11] = a41*b13 + a42*b23 + a43*b33 + a44*b43;
        te[15] = a41*b14 + a42*b24 + a43*b34 + a44*b44;
        return this;
    }
    transpose() {
        const e = this.elements; let t;
        t = e[1]; e[1] = e[4]; e[4] = t;
        t = e[2]; e[2] = e[8]; e[8] = t;
        t = e[6]; e[6] = e[9]; e[9] = t;
        t = e[3]; e[3] = e[12]; e[12] = t;
        t = e[7]; e[7] = e[13]; e[13] = t;
        t = e[11]; e[11] = e[14]; e[14] = t;
        return this;
    }
    invert() {
        // From three.js (Mesa GLU). Returns identity if non-invertible.
        const e = this.elements;
        const n11=e[0], n21=e[1], n31=e[2], n41=e[3];
        const n12=e[4], n22=e[5], n32=e[6], n42=e[7];
        const n13=e[8], n23=e[9], n33=e[10],n43=e[11];
        const n14=e[12],n24=e[13],n34=e[14],n44=e[15];
        const t11 = n23*n34*n42 - n24*n33*n42 + n24*n32*n43 - n22*n34*n43 - n23*n32*n44 + n22*n33*n44;
        const t12 = n14*n33*n42 - n13*n34*n42 - n14*n32*n43 + n12*n34*n43 + n13*n32*n44 - n12*n33*n44;
        const t13 = n13*n24*n42 - n14*n23*n42 + n14*n22*n43 - n12*n24*n43 - n13*n22*n44 + n12*n23*n44;
        const t14 = n14*n23*n32 - n13*n24*n32 - n14*n22*n33 + n12*n24*n33 + n13*n22*n34 - n12*n23*n34;
        const det = n11*t11 + n21*t12 + n31*t13 + n41*t14;
        if (det === 0) return this.identity();
        const di = 1 / det;
        e[0] = t11 * di; e[4] = t12 * di; e[8] = t13 * di; e[12] = t14 * di;
        e[1] = (n24*n33*n41 - n23*n34*n41 - n24*n31*n43 + n21*n34*n43 + n23*n31*n44 - n21*n33*n44) * di;
        e[5] = (n13*n34*n41 - n14*n33*n41 + n14*n31*n43 - n11*n34*n43 - n13*n31*n44 + n11*n33*n44) * di;
        e[9] = (n14*n23*n41 - n13*n24*n41 - n14*n21*n43 + n11*n24*n43 + n13*n21*n44 - n11*n23*n44) * di;
        e[13] = (n13*n24*n31 - n14*n23*n31 + n14*n21*n33 - n11*n24*n33 - n13*n21*n34 + n11*n23*n34) * di;
        e[2] = (n22*n34*n41 - n24*n32*n41 + n24*n31*n42 - n21*n34*n42 - n22*n31*n44 + n21*n32*n44) * di;
        e[6] = (n14*n32*n41 - n12*n34*n41 - n14*n31*n42 + n11*n34*n42 + n12*n31*n44 - n11*n32*n44) * di;
        e[10] = (n12*n24*n41 - n14*n22*n41 + n14*n21*n42 - n11*n24*n42 - n12*n21*n44 + n11*n22*n44) * di;
        e[14] = (n14*n22*n31 - n12*n24*n31 - n14*n21*n32 + n11*n24*n32 + n12*n21*n34 - n11*n22*n34) * di;
        e[3] = (n23*n32*n41 - n22*n33*n41 - n23*n31*n42 + n21*n33*n42 + n22*n31*n43 - n21*n32*n43) * di;
        e[7] = (n12*n33*n41 - n13*n32*n41 + n13*n31*n42 - n11*n33*n42 - n12*n31*n43 + n11*n32*n43) * di;
        e[11] = (n13*n22*n41 - n12*n23*n41 - n13*n21*n42 + n11*n23*n42 + n12*n21*n43 - n11*n22*n43) * di;
        e[15] = (n12*n23*n31 - n13*n22*n31 + n13*n21*n32 - n11*n23*n32 - n12*n21*n33 + n11*n22*n33) * di;
        return this;
    }
    compose(pos, quat, scl) {
        const e = this.elements;
        const x = quat.x, y = quat.y, z = quat.z, w = quat.w;
        const x2 = x+x, y2 = y+y, z2 = z+z;
        const xx = x*x2, xy = x*y2, xz = x*z2;
        const yy = y*y2, yz = y*z2, zz = z*z2;
        const wx = w*x2, wy = w*y2, wz = w*z2;
        const sx = scl.x, sy = scl.y, sz = scl.z;
        e[0] = (1-(yy+zz))*sx; e[1] = (xy+wz)*sx;    e[2] = (xz-wy)*sx;    e[3] = 0;
        e[4] = (xy-wz)*sy;     e[5] = (1-(xx+zz))*sy; e[6] = (yz+wx)*sy;   e[7] = 0;
        e[8] = (xz+wy)*sz;     e[9] = (yz-wx)*sz;    e[10] = (1-(xx+yy))*sz; e[11] = 0;
        e[12] = pos.x; e[13] = pos.y; e[14] = pos.z; e[15] = 1;
        return this;
    }
    _determinant() {
        const e = this.elements;
        const n11 = e[0], n21 = e[1], n31 = e[2], n41 = e[3];
        const n12 = e[4], n22 = e[5], n32 = e[6], n42 = e[7];
        const n13 = e[8], n23 = e[9], n33 = e[10], n43 = e[11];
        const n14 = e[12], n24 = e[13], n34 = e[14], n44 = e[15];
        return (
            n41 * (+n14*n23*n32 - n13*n24*n32 - n14*n22*n33 + n12*n24*n33 + n13*n22*n34 - n12*n23*n34) +
            n42 * (+n11*n23*n34 - n11*n24*n33 + n14*n21*n33 - n13*n21*n34 + n13*n24*n31 - n14*n23*n31) +
            n43 * (+n11*n24*n32 - n11*n22*n34 - n14*n21*n32 + n12*n21*n34 + n14*n22*n31 - n12*n24*n31) +
            n44 * (-n13*n22*n31 - n11*n23*n32 + n11*n22*n33 + n13*n21*n32 - n12*n21*n33 + n12*n23*n31)
        );
    }
    determinant() { return this._determinant(); }
    multiplyScalar( s ) {
        const te = this.elements;
        te[ 0 ] *= s; te[ 4 ] *= s; te[ 8 ] *= s; te[ 12 ] *= s;
        te[ 1 ] *= s; te[ 5 ] *= s; te[ 9 ] *= s; te[ 13 ] *= s;
        te[ 2 ] *= s; te[ 6 ] *= s; te[ 10 ] *= s; te[ 14 ] *= s;
        te[ 3 ] *= s; te[ 7 ] *= s; te[ 11 ] *= s; te[ 15 ] *= s;
        return this;
    }
    // ---- three.js r165 ----
    extractBasis( xAxis, yAxis, zAxis ) {
        xAxis.setFromMatrixColumn( this, 0 );
        yAxis.setFromMatrixColumn( this, 1 );
        zAxis.setFromMatrixColumn( this, 2 );
        return this;
    }
    makeBasis( xAxis, yAxis, zAxis ) {
        this.set(
            xAxis.x, yAxis.x, zAxis.x, 0,
            xAxis.y, yAxis.y, zAxis.y, 0,
            xAxis.z, yAxis.z, zAxis.z, 0,
            0, 0, 0, 1
        );
        return this;
    }
    extractRotation( m ) {
        // this method does not support reflection matrices
        const te = this.elements;
        const me = m.elements;
        const scaleX = 1 / _m4V1.setFromMatrixColumn( m, 0 ).length();
        const scaleY = 1 / _m4V1.setFromMatrixColumn( m, 1 ).length();
        const scaleZ = 1 / _m4V1.setFromMatrixColumn( m, 2 ).length();
        te[ 0 ] = me[ 0 ] * scaleX;
        te[ 1 ] = me[ 1 ] * scaleX;
        te[ 2 ] = me[ 2 ] * scaleX;
        te[ 3 ] = 0;
        te[ 4 ] = me[ 4 ] * scaleY;
        te[ 5 ] = me[ 5 ] * scaleY;
        te[ 6 ] = me[ 6 ] * scaleY;
        te[ 7 ] = 0;
        te[ 8 ] = me[ 8 ] * scaleZ;
        te[ 9 ] = me[ 9 ] * scaleZ;
        te[ 10 ] = me[ 10 ] * scaleZ;
        te[ 11 ] = 0;
        te[ 12 ] = 0;
        te[ 13 ] = 0;
        te[ 14 ] = 0;
        te[ 15 ] = 1;
        return this;
    }
    makeRotationFromEuler( euler ) {
        const te = this.elements;
        const x = euler.x, y = euler.y, z = euler.z;
        const a = Math.cos( x ), b = Math.sin( x );
        const c = Math.cos( y ), d = Math.sin( y );
        const e = Math.cos( z ), f = Math.sin( z );
        if ( euler.order === 'XYZ' ) {
            const ae = a * e, af = a * f, be = b * e, bf = b * f;
            te[ 0 ] = c * e;
            te[ 4 ] = - c * f;
            te[ 8 ] = d;
            te[ 1 ] = af + be * d;
            te[ 5 ] = ae - bf * d;
            te[ 9 ] = - b * c;
            te[ 2 ] = bf - ae * d;
            te[ 6 ] = be + af * d;
            te[ 10 ] = a * c;
        } else if ( euler.order === 'YXZ' ) {
            const ce = c * e, cf = c * f, de = d * e, df = d * f;
            te[ 0 ] = ce + df * b;
            te[ 4 ] = de * b - cf;
            te[ 8 ] = a * d;
            te[ 1 ] = a * f;
            te[ 5 ] = a * e;
            te[ 9 ] = - b;
            te[ 2 ] = cf * b - de;
            te[ 6 ] = df + ce * b;
            te[ 10 ] = a * c;
        } else if ( euler.order === 'ZXY' ) {
            const ce = c * e, cf = c * f, de = d * e, df = d * f;
            te[ 0 ] = ce - df * b;
            te[ 4 ] = - a * f;
            te[ 8 ] = de + cf * b;
            te[ 1 ] = cf + de * b;
            te[ 5 ] = a * e;
            te[ 9 ] = df - ce * b;
            te[ 2 ] = - a * d;
            te[ 6 ] = b;
            te[ 10 ] = a * c;
        } else if ( euler.order === 'ZYX' ) {
            const ae = a * e, af = a * f, be = b * e, bf = b * f;
            te[ 0 ] = c * e;
            te[ 4 ] = be * d - af;
            te[ 8 ] = ae * d + bf;
            te[ 1 ] = c * f;
            te[ 5 ] = bf * d + ae;
            te[ 9 ] = af * d - be;
            te[ 2 ] = - d;
            te[ 6 ] = b * c;
            te[ 10 ] = a * c;
        } else if ( euler.order === 'YZX' ) {
            const ac = a * c, ad = a * d, bc = b * c, bd = b * d;
            te[ 0 ] = c * e;
            te[ 4 ] = bd - ac * f;
            te[ 8 ] = bc * f + ad;
            te[ 1 ] = f;
            te[ 5 ] = a * e;
            te[ 9 ] = - b * e;
            te[ 2 ] = - d * e;
            te[ 6 ] = ad * f + bc;
            te[ 10 ] = ac - bd * f;
        } else if ( euler.order === 'XZY' ) {
            const ac = a * c, ad = a * d, bc = b * c, bd = b * d;
            te[ 0 ] = c * e;
            te[ 4 ] = - f;
            te[ 8 ] = d * e;
            te[ 1 ] = ac * f + bd;
            te[ 5 ] = a * e;
            te[ 9 ] = ad * f - bc;
            te[ 2 ] = bc * f - ad;
            te[ 6 ] = b * e;
            te[ 10 ] = bd * f + ac;
        }
        // bottom row
        te[ 3 ] = 0;
        te[ 7 ] = 0;
        te[ 11 ] = 0;
        // last column
        te[ 12 ] = 0;
        te[ 13 ] = 0;
        te[ 14 ] = 0;
        te[ 15 ] = 1;
        return this;
    }
    lookAt( eye, target, up ) {
        const te = this.elements;
        _m4Z.subVectors( eye, target );
        if ( _m4Z.lengthSq() === 0 ) {
            // eye and target are in the same position
            _m4Z.z = 1;
        }
        _m4Z.normalize();
        _m4X.crossVectors( up, _m4Z );
        if ( _m4X.lengthSq() === 0 ) {
            // up and z are parallel
            if ( Math.abs( up.z ) === 1 ) {
                _m4Z.x += 0.0001;
            } else {
                _m4Z.z += 0.0001;
            }
            _m4Z.normalize();
            _m4X.crossVectors( up, _m4Z );
        }
        _m4X.normalize();
        _m4Y.crossVectors( _m4Z, _m4X );
        te[ 0 ] = _m4X.x; te[ 4 ] = _m4Y.x; te[ 8 ] = _m4Z.x;
        te[ 1 ] = _m4X.y; te[ 5 ] = _m4Y.y; te[ 9 ] = _m4Z.y;
        te[ 2 ] = _m4X.z; te[ 6 ] = _m4Y.z; te[ 10 ] = _m4Z.z;
        return this;
    }
    scale( v ) {
        const te = this.elements;
        const x = v.x, y = v.y, z = v.z;
        te[ 0 ] *= x; te[ 4 ] *= y; te[ 8 ] *= z;
        te[ 1 ] *= x; te[ 5 ] *= y; te[ 9 ] *= z;
        te[ 2 ] *= x; te[ 6 ] *= y; te[ 10 ] *= z;
        te[ 3 ] *= x; te[ 7 ] *= y; te[ 11 ] *= z;
        return this;
    }
    getMaxScaleOnAxis() {
        const te = this.elements;
        const scaleXSq = te[ 0 ] * te[ 0 ] + te[ 1 ] * te[ 1 ] + te[ 2 ] * te[ 2 ];
        const scaleYSq = te[ 4 ] * te[ 4 ] + te[ 5 ] * te[ 5 ] + te[ 6 ] * te[ 6 ];
        const scaleZSq = te[ 8 ] * te[ 8 ] + te[ 9 ] * te[ 9 ] + te[ 10 ] * te[ 10 ];
        return Math.sqrt( Math.max( scaleXSq, scaleYSq, scaleZSq ) );
    }
    makeShear( xy, xz, yx, yz, zx, zy ) {
        this.set(
            1, yx, zx, 0,
            xy, 1, zy, 0,
            xz, yz, 1, 0,
            0, 0, 0, 1
        );
        return this;
    }
    decompose( position, quaternion, scale ) {
        const te = this.elements;
        let sx = _m4V1.set( te[ 0 ], te[ 1 ], te[ 2 ] ).length();
        const sy = _m4V1.set( te[ 4 ], te[ 5 ], te[ 6 ] ).length();
        const sz = _m4V1.set( te[ 8 ], te[ 9 ], te[ 10 ] ).length();
        // if determine is negative, we need to invert one scale
        const det = this.determinant();
        if ( det < 0 ) sx = - sx;
        position.x = te[ 12 ];
        position.y = te[ 13 ];
        position.z = te[ 14 ];
        // scale the rotation part
        _m4M1.copy( this );
        const invSX = 1 / sx;
        const invSY = 1 / sy;
        const invSZ = 1 / sz;
        _m4M1.elements[ 0 ] *= invSX;
        _m4M1.elements[ 1 ] *= invSX;
        _m4M1.elements[ 2 ] *= invSX;
        _m4M1.elements[ 4 ] *= invSY;
        _m4M1.elements[ 5 ] *= invSY;
        _m4M1.elements[ 6 ] *= invSY;
        _m4M1.elements[ 8 ] *= invSZ;
        _m4M1.elements[ 9 ] *= invSZ;
        _m4M1.elements[ 10 ] *= invSZ;
        quaternion.setFromRotationMatrix( _m4M1 );
        scale.x = sx;
        scale.y = sy;
        scale.z = sz;
        return this;
    }
    equals( matrix ) {
        const te = this.elements;
        const me = matrix.elements;
        for ( let i = 0; i < 16; i ++ ) {
            if ( te[ i ] !== me[ i ] ) return false;
        }
        return true;
    }
    copyPosition( m ) {
        const te = this.elements, me = m.elements;
        te[ 12 ] = me[ 12 ];
        te[ 13 ] = me[ 13 ];
        te[ 14 ] = me[ 14 ];
        return this;
    }
    setFromMatrix3( m ) {
        const me = m.elements;
        this.set(
            me[ 0 ], me[ 3 ], me[ 6 ], 0,
            me[ 1 ], me[ 4 ], me[ 7 ], 0,
            me[ 2 ], me[ 5 ], me[ 8 ], 0,
            0, 0, 0, 1
        );
        return this;
    }
}
// Scratch values for Matrix4 / Euler (three.js keeps the same module-level ones).
const _m4V1 = new Vector3(), _m4X = new Vector3(), _m4Y = new Vector3(), _m4Z = new Vector3();
const _m4M1 = new Matrix4();
const _eulerMatrix = new Matrix4();
const _eulerQuaternion = new Quaternion();
function _clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
// three.js's type flags, which three.js code (and addons) test.
Vector2.prototype.isVector2 = true;
Vector3.prototype.isVector3 = true;
Vector4.prototype.isVector4 = true;
Matrix3.prototype.isMatrix3 = true;
Matrix4.prototype.isMatrix4 = true;
Color.prototype.isColor = true;

export class Box2 {
    constructor( min = new Vector2( + Infinity, + Infinity ), max = new Vector2( - Infinity, - Infinity ) ) {
        this.isBox2 = true;
        this.min = min;
        this.max = max;
    }
    set( min, max ) {
        this.min.copy( min );
        this.max.copy( max );
        return this;
    }
    setFromPoints( points ) {
        this.makeEmpty();
        for ( let i = 0, il = points.length; i < il; i ++ ) {
            this.expandByPoint( points[ i ] );
        }
        return this;
    }
    setFromCenterAndSize( center, size ) {
        const halfSize = _box2_vector.copy( size ).multiplyScalar( 0.5 );
        this.min.copy( center ).sub( halfSize );
        this.max.copy( center ).add( halfSize );
        return this;
    }
    clone() {
        return new this.constructor().copy( this );
    }
    copy( box ) {
        this.min.copy( box.min );
        this.max.copy( box.max );
        return this;
    }
    makeEmpty() {
        this.min.x = this.min.y = + Infinity;
        this.max.x = this.max.y = - Infinity;
        return this;
    }
    isEmpty() {
        // this is a more robust check for empty than ( volume <= 0 ) because volume can get positive with two negative axes
        return ( this.max.x < this.min.x ) || ( this.max.y < this.min.y );
    }
    getCenter( target ) {
        return this.isEmpty() ? target.set( 0, 0 ) : target.addVectors( this.min, this.max ).multiplyScalar( 0.5 );
    }
    getSize( target ) {
        return this.isEmpty() ? target.set( 0, 0 ) : target.subVectors( this.max, this.min );
    }
    expandByPoint( point ) {
        this.min.min( point );
        this.max.max( point );
        return this;
    }
    expandByVector( vector ) {
        this.min.sub( vector );
        this.max.add( vector );
        return this;
    }
    expandByScalar( scalar ) {
        this.min.addScalar( - scalar );
        this.max.addScalar( scalar );
        return this;
    }
    containsPoint( point ) {
        return point.x < this.min.x || point.x > this.max.x ||
            point.y < this.min.y || point.y > this.max.y ? false : true;
    }
    containsBox( box ) {
        return this.min.x <= box.min.x && box.max.x <= this.max.x &&
            this.min.y <= box.min.y && box.max.y <= this.max.y;
    }
    getParameter( point, target ) {
        // This can potentially have a divide by zero if the box
        // has a size dimension of 0.
        return target.set(
            ( point.x - this.min.x ) / ( this.max.x - this.min.x ),
            ( point.y - this.min.y ) / ( this.max.y - this.min.y )
        );
    }
    intersectsBox( box ) {
        // using 4 splitting planes to rule out intersections
        return box.max.x < this.min.x || box.min.x > this.max.x ||
            box.max.y < this.min.y || box.min.y > this.max.y ? false : true;
    }
    clampPoint( point, target ) {
        return target.copy( point ).clamp( this.min, this.max );
    }
    distanceToPoint( point ) {
        return this.clampPoint( point, _box2_vector ).distanceTo( point );
    }
    intersect( box ) {
        this.min.max( box.min );
        this.max.min( box.max );
        if ( this.isEmpty() ) this.makeEmpty();
        return this;
    }
    union( box ) {
        this.min.min( box.min );
        this.max.max( box.max );
        return this;
    }
    translate( offset ) {
        this.min.add( offset );
        this.max.add( offset );
        return this;
    }
    equals( box ) {
        return box.min.equals( this.min ) && box.max.equals( this.max );
    }
}
export class Box3 {
    constructor( min = new Vector3( + Infinity, + Infinity, + Infinity ), max = new Vector3( - Infinity, - Infinity, - Infinity ) ) {
        this.isBox3 = true;
        this.min = min;
        this.max = max;
    }
    set( min, max ) {
        this.min.copy( min );
        this.max.copy( max );
        return this;
    }
    setFromArray( array ) {
        this.makeEmpty();
        for ( let i = 0, il = array.length; i < il; i += 3 ) {
            this.expandByPoint( _box3_vector.fromArray( array, i ) );
        }
        return this;
    }
    setFromBufferAttribute( attribute ) {
        this.makeEmpty();
        for ( let i = 0, il = attribute.count; i < il; i ++ ) {
            this.expandByPoint( _box3_vector.fromBufferAttribute( attribute, i ) );
        }
        return this;
    }
    setFromPoints( points ) {
        this.makeEmpty();
        for ( let i = 0, il = points.length; i < il; i ++ ) {
            this.expandByPoint( points[ i ] );
        }
        return this;
    }
    setFromCenterAndSize( center, size ) {
        const halfSize = _box3_vector.copy( size ).multiplyScalar( 0.5 );
        this.min.copy( center ).sub( halfSize );
        this.max.copy( center ).add( halfSize );
        return this;
    }
    setFromObject( object, precise = false ) {
        this.makeEmpty();
        return this.expandByObject( object, precise );
    }
    clone() {
        return new this.constructor().copy( this );
    }
    copy( box ) {
        this.min.copy( box.min );
        this.max.copy( box.max );
        return this;
    }
    makeEmpty() {
        this.min.x = this.min.y = this.min.z = + Infinity;
        this.max.x = this.max.y = this.max.z = - Infinity;
        return this;
    }
    isEmpty() {
        // this is a more robust check for empty than ( volume <= 0 ) because volume can get positive with two negative axes
        return ( this.max.x < this.min.x ) || ( this.max.y < this.min.y ) || ( this.max.z < this.min.z );
    }
    getCenter( target ) {
        return this.isEmpty() ? target.set( 0, 0, 0 ) : target.addVectors( this.min, this.max ).multiplyScalar( 0.5 );
    }
    getSize( target ) {
        return this.isEmpty() ? target.set( 0, 0, 0 ) : target.subVectors( this.max, this.min );
    }
    expandByPoint( point ) {
        this.min.min( point );
        this.max.max( point );
        return this;
    }
    expandByVector( vector ) {
        this.min.sub( vector );
        this.max.add( vector );
        return this;
    }
    expandByScalar( scalar ) {
        this.min.addScalar( - scalar );
        this.max.addScalar( scalar );
        return this;
    }
    expandByObject( object, precise = false ) {
        // three.js r165's walk over the object and its children. The bounds
        // themselves are Rust's: the geometry's box (`_geometryBoundingBox`),
        // or each vertex of the mirrored node — morphed and skinned as
        // `getVertexPosition` poses it — placed by this `matrixWorld`.
        object.updateWorldMatrix( false, false );
        const geometry = _boundsGeometryOf( object );
        if ( geometry !== undefined ) {
            if ( precise === true && object.isInstancedMesh !== true && _boundsHavePositions( geometry ) ) {
                const b = _objectVertexBounds( object, 0, object.matrixWorld );
                if ( b ) this.union( _box3_box.set( _box3_vector.set( b[ 0 ], b[ 1 ], b[ 2 ] ), _box3_v0.set( b[ 3 ], b[ 4 ], b[ 5 ] ) ) );
            } else {
                if ( object.boundingBox !== undefined ) {
                    // object-level bounding box
                    if ( object.boundingBox === null ) {
                        object.computeBoundingBox();
                    }
                    _box3_box.copy( object.boundingBox );
                } else {
                    // geometry-level bounding box
                    _box3_box.copy( _geometryBoundingBox( geometry ) );
                }
                _box3_box.applyMatrix4( object.matrixWorld );
                this.union( _box3_box );
            }
        }
        const children = object.children;
        for ( let i = 0, l = children.length; i < l; i ++ ) {
            this.expandByObject( children[ i ], precise );
        }
        return this;
    }
    containsPoint( point ) {
        return point.x < this.min.x || point.x > this.max.x ||
            point.y < this.min.y || point.y > this.max.y ||
            point.z < this.min.z || point.z > this.max.z ? false : true;
    }
    containsBox( box ) {
        return this.min.x <= box.min.x && box.max.x <= this.max.x &&
            this.min.y <= box.min.y && box.max.y <= this.max.y &&
            this.min.z <= box.min.z && box.max.z <= this.max.z;
    }
    getParameter( point, target ) {
        // This can potentially have a divide by zero if the box
        // has a size dimension of 0.
        return target.set(
            ( point.x - this.min.x ) / ( this.max.x - this.min.x ),
            ( point.y - this.min.y ) / ( this.max.y - this.min.y ),
            ( point.z - this.min.z ) / ( this.max.z - this.min.z )
        );
    }
    intersectsBox( box ) {
        // using 6 splitting planes to rule out intersections.
        return box.max.x < this.min.x || box.min.x > this.max.x ||
            box.max.y < this.min.y || box.min.y > this.max.y ||
            box.max.z < this.min.z || box.min.z > this.max.z ? false : true;
    }
    intersectsSphere( sphere ) {
        // Find the point on the AABB closest to the sphere center.
        this.clampPoint( sphere.center, _box3_vector );
        // If that point is inside the sphere, the AABB and sphere intersect.
        return _box3_vector.distanceToSquared( sphere.center ) <= ( sphere.radius * sphere.radius );
    }
    intersectsPlane( plane ) {
        // We compute the minimum and maximum dot product values. If those values
        // are on the same side (back or front) of the plane, then there is no intersection.
        let min, max;
        if ( plane.normal.x > 0 ) {
            min = plane.normal.x * this.min.x;
            max = plane.normal.x * this.max.x;
        } else {
            min = plane.normal.x * this.max.x;
            max = plane.normal.x * this.min.x;
        }
        if ( plane.normal.y > 0 ) {
            min += plane.normal.y * this.min.y;
            max += plane.normal.y * this.max.y;
        } else {
            min += plane.normal.y * this.max.y;
            max += plane.normal.y * this.min.y;
        }
        if ( plane.normal.z > 0 ) {
            min += plane.normal.z * this.min.z;
            max += plane.normal.z * this.max.z;
        } else {
            min += plane.normal.z * this.max.z;
            max += plane.normal.z * this.min.z;
        }
        return ( min <= - plane.constant && max >= - plane.constant );
    }
    intersectsTriangle( triangle ) {
        if ( this.isEmpty() ) {
            return false;
        }
        // compute box center and extents
        this.getCenter( _box3_center );
        _box3_extents.subVectors( this.max, _box3_center );
        // translate triangle to aabb origin
        _box3_v0.subVectors( triangle.a, _box3_center );
        _box3_v1.subVectors( triangle.b, _box3_center );
        _box3_v2.subVectors( triangle.c, _box3_center );
        // compute edge vectors for triangle
        _box3_f0.subVectors( _box3_v1, _box3_v0 );
        _box3_f1.subVectors( _box3_v2, _box3_v1 );
        _box3_f2.subVectors( _box3_v0, _box3_v2 );
        // test against axes that are given by cross product combinations of the edges of the triangle and the edges of the aabb
        // make an axis testing of each of the 3 sides of the aabb against each of the 3 sides of the triangle = 9 axis of separation
        // axis_ij = u_i x f_j (u0, u1, u2 = face normals of aabb = x,y,z axes vectors since aabb is axis aligned)
        let axes = [
            0, - _box3_f0.z, _box3_f0.y, 0, - _box3_f1.z, _box3_f1.y, 0, - _box3_f2.z, _box3_f2.y,
            _box3_f0.z, 0, - _box3_f0.x, _box3_f1.z, 0, - _box3_f1.x, _box3_f2.z, 0, - _box3_f2.x,
            - _box3_f0.y, _box3_f0.x, 0, - _box3_f1.y, _box3_f1.x, 0, - _box3_f2.y, _box3_f2.x, 0
        ];
        if ( ! _box3_satForAxes( axes, _box3_v0, _box3_v1, _box3_v2, _box3_extents ) ) {
            return false;
        }
        // test 3 face normals from the aabb
        axes = [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ];
        if ( ! _box3_satForAxes( axes, _box3_v0, _box3_v1, _box3_v2, _box3_extents ) ) {
            return false;
        }
        // finally testing the face normal of the triangle
        // use already existing triangle edge vectors here
        _box3_triangleNormal.crossVectors( _box3_f0, _box3_f1 );
        axes = [ _box3_triangleNormal.x, _box3_triangleNormal.y, _box3_triangleNormal.z ];
        return _box3_satForAxes( axes, _box3_v0, _box3_v1, _box3_v2, _box3_extents );
    }
    clampPoint( point, target ) {
        return target.copy( point ).clamp( this.min, this.max );
    }
    distanceToPoint( point ) {
        return this.clampPoint( point, _box3_vector ).distanceTo( point );
    }
    getBoundingSphere( target ) {
        if ( this.isEmpty() ) {
            target.makeEmpty();
        } else {
            this.getCenter( target.center );
            target.radius = this.getSize( _box3_vector ).length() * 0.5;
        }
        return target;
    }
    intersect( box ) {
        this.min.max( box.min );
        this.max.min( box.max );
        // ensure that if there is no overlap, the result is fully empty, not slightly empty with non-inf/+inf values that will cause subsequence intersects to erroneously return valid values.
        if ( this.isEmpty() ) this.makeEmpty();
        return this;
    }
    union( box ) {
        this.min.min( box.min );
        this.max.max( box.max );
        return this;
    }
    applyMatrix4( matrix ) {
        // transform of empty box is an empty box.
        if ( this.isEmpty() ) return this;
        // NOTE: I am using a binary pattern to specify all 2^3 combinations below
        _box3_points[ 0 ].set( this.min.x, this.min.y, this.min.z ).applyMatrix4( matrix ); // 000
        _box3_points[ 1 ].set( this.min.x, this.min.y, this.max.z ).applyMatrix4( matrix ); // 001
        _box3_points[ 2 ].set( this.min.x, this.max.y, this.min.z ).applyMatrix4( matrix ); // 010
        _box3_points[ 3 ].set( this.min.x, this.max.y, this.max.z ).applyMatrix4( matrix ); // 011
        _box3_points[ 4 ].set( this.max.x, this.min.y, this.min.z ).applyMatrix4( matrix ); // 100
        _box3_points[ 5 ].set( this.max.x, this.min.y, this.max.z ).applyMatrix4( matrix ); // 101
        _box3_points[ 6 ].set( this.max.x, this.max.y, this.min.z ).applyMatrix4( matrix ); // 110
        _box3_points[ 7 ].set( this.max.x, this.max.y, this.max.z ).applyMatrix4( matrix ); // 111
        this.setFromPoints( _box3_points );
        return this;
    }
    translate( offset ) {
        this.min.add( offset );
        this.max.add( offset );
        return this;
    }
    equals( box ) {
        return box.min.equals( this.min ) && box.max.equals( this.max );
    }
}
export class Sphere {
    constructor( center = new Vector3(), radius = - 1 ) {
        this.isSphere = true;
        this.center = center;
        this.radius = radius;
    }
    set( center, radius ) {
        this.center.copy( center );
        this.radius = radius;
        return this;
    }
    setFromPoints( points, optionalCenter ) {
        const center = this.center;
        if ( optionalCenter !== undefined ) {
            center.copy( optionalCenter );
        } else {
            _sphere_box.setFromPoints( points ).getCenter( center );
        }
        let maxRadiusSq = 0;
        for ( let i = 0, il = points.length; i < il; i ++ ) {
            maxRadiusSq = Math.max( maxRadiusSq, center.distanceToSquared( points[ i ] ) );
        }
        this.radius = Math.sqrt( maxRadiusSq );
        return this;
    }
    copy( sphere ) {
        this.center.copy( sphere.center );
        this.radius = sphere.radius;
        return this;
    }
    isEmpty() {
        return ( this.radius < 0 );
    }
    makeEmpty() {
        this.center.set( 0, 0, 0 );
        this.radius = - 1;
        return this;
    }
    containsPoint( point ) {
        return ( point.distanceToSquared( this.center ) <= ( this.radius * this.radius ) );
    }
    distanceToPoint( point ) {
        return ( point.distanceTo( this.center ) - this.radius );
    }
    intersectsSphere( sphere ) {
        const radiusSum = this.radius + sphere.radius;
        return sphere.center.distanceToSquared( this.center ) <= ( radiusSum * radiusSum );
    }
    intersectsBox( box ) {
        return box.intersectsSphere( this );
    }
    intersectsPlane( plane ) {
        return Math.abs( plane.distanceToPoint( this.center ) ) <= this.radius;
    }
    clampPoint( point, target ) {
        const deltaLengthSq = this.center.distanceToSquared( point );
        target.copy( point );
        if ( deltaLengthSq > ( this.radius * this.radius ) ) {
            target.sub( this.center ).normalize();
            target.multiplyScalar( this.radius ).add( this.center );
        }
        return target;
    }
    getBoundingBox( target ) {
        if ( this.isEmpty() ) {
            // Empty sphere produces empty bounding box
            target.makeEmpty();
            return target;
        }
        target.set( this.center, this.center );
        target.expandByScalar( this.radius );
        return target;
    }
    applyMatrix4( matrix ) {
        this.center.applyMatrix4( matrix );
        this.radius = this.radius * matrix.getMaxScaleOnAxis();
        return this;
    }
    translate( offset ) {
        this.center.add( offset );
        return this;
    }
    expandByPoint( point ) {
        if ( this.isEmpty() ) {
            this.center.copy( point );
            this.radius = 0;
            return this;
        }
        _sphere_v1.subVectors( point, this.center );
        const lengthSq = _sphere_v1.lengthSq();
        if ( lengthSq > ( this.radius * this.radius ) ) {
            // calculate the minimal sphere
            const length = Math.sqrt( lengthSq );
            const delta = ( length - this.radius ) * 0.5;
            this.center.addScaledVector( _sphere_v1, delta / length );
            this.radius += delta;
        }
        return this;
    }
    union( sphere ) {
        if ( sphere.isEmpty() ) {
            return this;
        }
        if ( this.isEmpty() ) {
            this.copy( sphere );
            return this;
        }
        if ( this.center.equals( sphere.center ) === true ) {
             this.radius = Math.max( this.radius, sphere.radius );
        } else {
            _sphere_v2.subVectors( sphere.center, this.center ).setLength( sphere.radius );
            this.expandByPoint( _sphere_v1.copy( sphere.center ).add( _sphere_v2 ) );
            this.expandByPoint( _sphere_v1.copy( sphere.center ).sub( _sphere_v2 ) );
        }
        return this;
    }
    equals( sphere ) {
        return sphere.center.equals( this.center ) && ( sphere.radius === this.radius );
    }
    clone() {
        return new this.constructor().copy( this );
    }
}
export class Ray {
    constructor( origin = new Vector3(), direction = new Vector3( 0, 0, - 1 ) ) {
        this.origin = origin;
        this.direction = direction;
    }
    set( origin, direction ) {
        this.origin.copy( origin );
        this.direction.copy( direction );
        return this;
    }
    copy( ray ) {
        this.origin.copy( ray.origin );
        this.direction.copy( ray.direction );
        return this;
    }
    at( t, target ) {
        return target.copy( this.origin ).addScaledVector( this.direction, t );
    }
    lookAt( v ) {
        this.direction.copy( v ).sub( this.origin ).normalize();
        return this;
    }
    recast( t ) {
        this.origin.copy( this.at( t, _ray_vector ) );
        return this;
    }
    closestPointToPoint( point, target ) {
        target.subVectors( point, this.origin );
        const directionDistance = target.dot( this.direction );
        if ( directionDistance < 0 ) {
            return target.copy( this.origin );
        }
        return target.copy( this.origin ).addScaledVector( this.direction, directionDistance );
    }
    distanceToPoint( point ) {
        return Math.sqrt( this.distanceSqToPoint( point ) );
    }
    distanceSqToPoint( point ) {
        const directionDistance = _ray_vector.subVectors( point, this.origin ).dot( this.direction );
        // point behind the ray
        if ( directionDistance < 0 ) {
            return this.origin.distanceToSquared( point );
        }
        _ray_vector.copy( this.origin ).addScaledVector( this.direction, directionDistance );
        return _ray_vector.distanceToSquared( point );
    }
    distanceSqToSegment( v0, v1, optionalPointOnRay, optionalPointOnSegment ) {
        // from https://github.com/pmjoniak/GeometricTools/blob/master/GTEngine/Include/Mathematics/GteDistRaySegment.h
        // It returns the min distance between the ray and the segment
        // defined by v0 and v1
        // It can also set two optional targets :
        // - The closest point on the ray
        // - The closest point on the segment
        _ray_segCenter.copy( v0 ).add( v1 ).multiplyScalar( 0.5 );
        _ray_segDir.copy( v1 ).sub( v0 ).normalize();
        _ray_diff.copy( this.origin ).sub( _ray_segCenter );
        const segExtent = v0.distanceTo( v1 ) * 0.5;
        const a01 = - this.direction.dot( _ray_segDir );
        const b0 = _ray_diff.dot( this.direction );
        const b1 = - _ray_diff.dot( _ray_segDir );
        const c = _ray_diff.lengthSq();
        const det = Math.abs( 1 - a01 * a01 );
        let s0, s1, sqrDist, extDet;
        if ( det > 0 ) {
            // The ray and segment are not parallel.
            s0 = a01 * b1 - b0;
            s1 = a01 * b0 - b1;
            extDet = segExtent * det;
            if ( s0 >= 0 ) {
                if ( s1 >= - extDet ) {
                    if ( s1 <= extDet ) {
                        // region 0
                        // Minimum at interior points of ray and segment.
                        const invDet = 1 / det;
                        s0 *= invDet;
                        s1 *= invDet;
                        sqrDist = s0 * ( s0 + a01 * s1 + 2 * b0 ) + s1 * ( a01 * s0 + s1 + 2 * b1 ) + c;
                    } else {
                        // region 1
                        s1 = segExtent;
                        s0 = Math.max( 0, - ( a01 * s1 + b0 ) );
                        sqrDist = - s0 * s0 + s1 * ( s1 + 2 * b1 ) + c;
                    }
                } else {
                    // region 5
                    s1 = - segExtent;
                    s0 = Math.max( 0, - ( a01 * s1 + b0 ) );
                    sqrDist = - s0 * s0 + s1 * ( s1 + 2 * b1 ) + c;
                }
            } else {
                if ( s1 <= - extDet ) {
                    // region 4
                    s0 = Math.max( 0, - ( - a01 * segExtent + b0 ) );
                    s1 = ( s0 > 0 ) ? - segExtent : Math.min( Math.max( - segExtent, - b1 ), segExtent );
                    sqrDist = - s0 * s0 + s1 * ( s1 + 2 * b1 ) + c;
                } else if ( s1 <= extDet ) {
                    // region 3
                    s0 = 0;
                    s1 = Math.min( Math.max( - segExtent, - b1 ), segExtent );
                    sqrDist = s1 * ( s1 + 2 * b1 ) + c;
                } else {
                    // region 2
                    s0 = Math.max( 0, - ( a01 * segExtent + b0 ) );
                    s1 = ( s0 > 0 ) ? segExtent : Math.min( Math.max( - segExtent, - b1 ), segExtent );
                    sqrDist = - s0 * s0 + s1 * ( s1 + 2 * b1 ) + c;
                }
            }
        } else {
            // Ray and segment are parallel.
            s1 = ( a01 > 0 ) ? - segExtent : segExtent;
            s0 = Math.max( 0, - ( a01 * s1 + b0 ) );
            sqrDist = - s0 * s0 + s1 * ( s1 + 2 * b1 ) + c;
        }
        if ( optionalPointOnRay ) {
            optionalPointOnRay.copy( this.origin ).addScaledVector( this.direction, s0 );
        }
        if ( optionalPointOnSegment ) {
            optionalPointOnSegment.copy( _ray_segCenter ).addScaledVector( _ray_segDir, s1 );
        }
        return sqrDist;
    }
    intersectSphere( sphere, target ) {
        _ray_vector.subVectors( sphere.center, this.origin );
        const tca = _ray_vector.dot( this.direction );
        const d2 = _ray_vector.dot( _ray_vector ) - tca * tca;
        const radius2 = sphere.radius * sphere.radius;
        if ( d2 > radius2 ) return null;
        const thc = Math.sqrt( radius2 - d2 );
        // t0 = first intersect point - entrance on front of sphere
        const t0 = tca - thc;
        // t1 = second intersect point - exit point on back of sphere
        const t1 = tca + thc;
        // test to see if t1 is behind the ray - if so, return null
        if ( t1 < 0 ) return null;
        // test to see if t0 is behind the ray:
        // if it is, the ray is inside the sphere, so return the second exit point scaled by t1,
        // in order to always return an intersect point that is in front of the ray.
        if ( t0 < 0 ) return this.at( t1, target );
        // else t0 is in front of the ray, so return the first collision point scaled by t0
        return this.at( t0, target );
    }
    intersectsSphere( sphere ) {
        return this.distanceSqToPoint( sphere.center ) <= ( sphere.radius * sphere.radius );
    }
    distanceToPlane( plane ) {
        const denominator = plane.normal.dot( this.direction );
        if ( denominator === 0 ) {
            // line is coplanar, return origin
            if ( plane.distanceToPoint( this.origin ) === 0 ) {
                return 0;
            }
            // Null is preferable to undefined since undefined means.... it is undefined
            return null;
        }
        const t = - ( this.origin.dot( plane.normal ) + plane.constant ) / denominator;
        // Return if the ray never intersects the plane
        return t >= 0 ? t : null;
    }
    intersectPlane( plane, target ) {
        const t = this.distanceToPlane( plane );
        if ( t === null ) {
            return null;
        }
        return this.at( t, target );
    }
    intersectsPlane( plane ) {
        // check if the ray lies on the plane first
        const distToPoint = plane.distanceToPoint( this.origin );
        if ( distToPoint === 0 ) {
            return true;
        }
        const denominator = plane.normal.dot( this.direction );
        if ( denominator * distToPoint < 0 ) {
            return true;
        }
        // ray origin is behind the plane (and is pointing behind it)
        return false;
    }
    intersectBox( box, target ) {
        let tmin, tmax, tymin, tymax, tzmin, tzmax;
        const invdirx = 1 / this.direction.x,
            invdiry = 1 / this.direction.y,
            invdirz = 1 / this.direction.z;
        const origin = this.origin;
        if ( invdirx >= 0 ) {
            tmin = ( box.min.x - origin.x ) * invdirx;
            tmax = ( box.max.x - origin.x ) * invdirx;
        } else {
            tmin = ( box.max.x - origin.x ) * invdirx;
            tmax = ( box.min.x - origin.x ) * invdirx;
        }
        if ( invdiry >= 0 ) {
            tymin = ( box.min.y - origin.y ) * invdiry;
            tymax = ( box.max.y - origin.y ) * invdiry;
        } else {
            tymin = ( box.max.y - origin.y ) * invdiry;
            tymax = ( box.min.y - origin.y ) * invdiry;
        }
        if ( ( tmin > tymax ) || ( tymin > tmax ) ) return null;
        if ( tymin > tmin || isNaN( tmin ) ) tmin = tymin;
        if ( tymax < tmax || isNaN( tmax ) ) tmax = tymax;
        if ( invdirz >= 0 ) {
            tzmin = ( box.min.z - origin.z ) * invdirz;
            tzmax = ( box.max.z - origin.z ) * invdirz;
        } else {
            tzmin = ( box.max.z - origin.z ) * invdirz;
            tzmax = ( box.min.z - origin.z ) * invdirz;
        }
        if ( ( tmin > tzmax ) || ( tzmin > tmax ) ) return null;
        if ( tzmin > tmin || tmin !== tmin ) tmin = tzmin;
        if ( tzmax < tmax || tmax !== tmax ) tmax = tzmax;
        //return point closest to the ray (positive side)
        if ( tmax < 0 ) return null;
        return this.at( tmin >= 0 ? tmin : tmax, target );
    }
    intersectsBox( box ) {
        return this.intersectBox( box, _ray_vector ) !== null;
    }
    intersectTriangle( a, b, c, backfaceCulling, target ) {
        // Compute the offset origin, edges, and normal.
        // from https://github.com/pmjoniak/GeometricTools/blob/master/GTEngine/Include/Mathematics/GteIntrRay3Triangle3.h
        _ray_edge1.subVectors( b, a );
        _ray_edge2.subVectors( c, a );
        _ray_normal.crossVectors( _ray_edge1, _ray_edge2 );
        // Solve Q + t*D = b1*E1 + b2*E2 (Q = kDiff, D = ray direction,
        // E1 = kEdge1, E2 = kEdge2, N = Cross(E1,E2)) by
        //   |Dot(D,N)|*b1 = sign(Dot(D,N))*Dot(D,Cross(Q,E2))
        //   |Dot(D,N)|*b2 = sign(Dot(D,N))*Dot(D,Cross(E1,Q))
        //   |Dot(D,N)|*t = -sign(Dot(D,N))*Dot(Q,N)
        let DdN = this.direction.dot( _ray_normal );
        let sign;
        if ( DdN > 0 ) {
            if ( backfaceCulling ) return null;
            sign = 1;
        } else if ( DdN < 0 ) {
            sign = - 1;
            DdN = - DdN;
        } else {
            return null;
        }
        _ray_diff.subVectors( this.origin, a );
        const DdQxE2 = sign * this.direction.dot( _ray_edge2.crossVectors( _ray_diff, _ray_edge2 ) );
        // b1 < 0, no intersection
        if ( DdQxE2 < 0 ) {
            return null;
        }
        const DdE1xQ = sign * this.direction.dot( _ray_edge1.cross( _ray_diff ) );
        // b2 < 0, no intersection
        if ( DdE1xQ < 0 ) {
            return null;
        }
        // b1+b2 > 1, no intersection
        if ( DdQxE2 + DdE1xQ > DdN ) {
            return null;
        }
        // Line intersects triangle, check if ray does.
        const QdN = - sign * _ray_diff.dot( _ray_normal );
        // t < 0, no intersection
        if ( QdN < 0 ) {
            return null;
        }
        // Ray intersects triangle.
        return this.at( QdN / DdN, target );
    }
    applyMatrix4( matrix4 ) {
        this.origin.applyMatrix4( matrix4 );
        this.direction.transformDirection( matrix4 );
        return this;
    }
    equals( ray ) {
        return ray.origin.equals( this.origin ) && ray.direction.equals( this.direction );
    }
    clone() {
        return new this.constructor().copy( this );
    }
}
export class Plane {
    constructor( normal = new Vector3( 1, 0, 0 ), constant = 0 ) {
        this.isPlane = true;
        // normal is assumed to be normalized
        this.normal = normal;
        this.constant = constant;
    }
    set( normal, constant ) {
        this.normal.copy( normal );
        this.constant = constant;
        return this;
    }
    setComponents( x, y, z, w ) {
        this.normal.set( x, y, z );
        this.constant = w;
        return this;
    }
    setFromNormalAndCoplanarPoint( normal, point ) {
        this.normal.copy( normal );
        this.constant = - point.dot( this.normal );
        return this;
    }
    setFromCoplanarPoints( a, b, c ) {
        const normal = _plane_vector1.subVectors( c, b ).cross( _plane_vector2.subVectors( a, b ) ).normalize();
        // Q: should an error be thrown if normal is zero (e.g. degenerate plane)?
        this.setFromNormalAndCoplanarPoint( normal, a );
        return this;
    }
    copy( plane ) {
        this.normal.copy( plane.normal );
        this.constant = plane.constant;
        return this;
    }
    normalize() {
        // Note: will lead to a divide by zero if the plane is invalid.
        const inverseNormalLength = 1.0 / this.normal.length();
        this.normal.multiplyScalar( inverseNormalLength );
        this.constant *= inverseNormalLength;
        return this;
    }
    negate() {
        this.constant *= - 1;
        this.normal.negate();
        return this;
    }
    distanceToPoint( point ) {
        return this.normal.dot( point ) + this.constant;
    }
    distanceToSphere( sphere ) {
        return this.distanceToPoint( sphere.center ) - sphere.radius;
    }
    projectPoint( point, target ) {
        return target.copy( point ).addScaledVector( this.normal, - this.distanceToPoint( point ) );
    }
    intersectLine( line, target ) {
        const direction = line.delta( _plane_vector1 );
        const denominator = this.normal.dot( direction );
        if ( denominator === 0 ) {
            // line is coplanar, return origin
            if ( this.distanceToPoint( line.start ) === 0 ) {
                return target.copy( line.start );
            }
            // Unsure if this is the correct method to handle this case.
            return null;
        }
        const t = - ( line.start.dot( this.normal ) + this.constant ) / denominator;
        if ( t < 0 || t > 1 ) {
            return null;
        }
        return target.copy( line.start ).addScaledVector( direction, t );
    }
    intersectsLine( line ) {
        // Note: this tests if a line intersects the plane, not whether it (or its end-points) are coplanar with it.
        const startSign = this.distanceToPoint( line.start );
        const endSign = this.distanceToPoint( line.end );
        return ( startSign < 0 && endSign > 0 ) || ( endSign < 0 && startSign > 0 );
    }
    intersectsBox( box ) {
        return box.intersectsPlane( this );
    }
    intersectsSphere( sphere ) {
        return sphere.intersectsPlane( this );
    }
    coplanarPoint( target ) {
        return target.copy( this.normal ).multiplyScalar( - this.constant );
    }
    applyMatrix4( matrix, optionalNormalMatrix ) {
        const normalMatrix = optionalNormalMatrix || _plane_normalMatrix.getNormalMatrix( matrix );
        const referencePoint = this.coplanarPoint( _plane_vector1 ).applyMatrix4( matrix );
        const normal = this.normal.applyMatrix3( normalMatrix ).normalize();
        this.constant = - referencePoint.dot( normal );
        return this;
    }
    translate( offset ) {
        this.constant -= offset.dot( this.normal );
        return this;
    }
    equals( plane ) {
        return plane.normal.equals( this.normal ) && ( plane.constant === this.constant );
    }
    clone() {
        return new this.constructor().copy( this );
    }
}
export class Triangle {
    constructor( a = new Vector3(), b = new Vector3(), c = new Vector3() ) {
        this.a = a;
        this.b = b;
        this.c = c;
    }
    static getNormal( a, b, c, target ) {
        target.subVectors( c, b );
        _tri_v0.subVectors( a, b );
        target.cross( _tri_v0 );
        const targetLengthSq = target.lengthSq();
        if ( targetLengthSq > 0 ) {
            return target.multiplyScalar( 1 / Math.sqrt( targetLengthSq ) );
        }
        return target.set( 0, 0, 0 );
    }
    // static/instance method to calculate barycentric coordinates
    // based on: http://www.blackpawn.com/texts/pointinpoly/default.html
    static getBarycoord( point, a, b, c, target ) {
        _tri_v0.subVectors( c, a );
        _tri_v1.subVectors( b, a );
        _tri_v2.subVectors( point, a );
        const dot00 = _tri_v0.dot( _tri_v0 );
        const dot01 = _tri_v0.dot( _tri_v1 );
        const dot02 = _tri_v0.dot( _tri_v2 );
        const dot11 = _tri_v1.dot( _tri_v1 );
        const dot12 = _tri_v1.dot( _tri_v2 );
        const denom = ( dot00 * dot11 - dot01 * dot01 );
        // collinear or singular triangle
        if ( denom === 0 ) {
            target.set( 0, 0, 0 );
            return null;
        }
        const invDenom = 1 / denom;
        const u = ( dot11 * dot02 - dot01 * dot12 ) * invDenom;
        const v = ( dot00 * dot12 - dot01 * dot02 ) * invDenom;
        // barycentric coordinates must always sum to 1
        return target.set( 1 - u - v, v, u );
    }
    static containsPoint( point, a, b, c ) {
        // if the triangle is degenerate then we can't contain a point
        if ( this.getBarycoord( point, a, b, c, _tri_v3 ) === null ) {
            return false;
        }
        return ( _tri_v3.x >= 0 ) && ( _tri_v3.y >= 0 ) && ( ( _tri_v3.x + _tri_v3.y ) <= 1 );
    }
    static getInterpolation( point, p1, p2, p3, v1, v2, v3, target ) {
        if ( this.getBarycoord( point, p1, p2, p3, _tri_v3 ) === null ) {
            target.x = 0;
            target.y = 0;
            if ( 'z' in target ) target.z = 0;
            if ( 'w' in target ) target.w = 0;
            return null;
        }
        target.setScalar( 0 );
        target.addScaledVector( v1, _tri_v3.x );
        target.addScaledVector( v2, _tri_v3.y );
        target.addScaledVector( v3, _tri_v3.z );
        return target;
    }
    static isFrontFacing( a, b, c, direction ) {
        _tri_v0.subVectors( c, b );
        _tri_v1.subVectors( a, b );
        // strictly front facing
        return ( _tri_v0.cross( _tri_v1 ).dot( direction ) < 0 ) ? true : false;
    }
    set( a, b, c ) {
        this.a.copy( a );
        this.b.copy( b );
        this.c.copy( c );
        return this;
    }
    setFromPointsAndIndices( points, i0, i1, i2 ) {
        this.a.copy( points[ i0 ] );
        this.b.copy( points[ i1 ] );
        this.c.copy( points[ i2 ] );
        return this;
    }
    setFromAttributeAndIndices( attribute, i0, i1, i2 ) {
        this.a.fromBufferAttribute( attribute, i0 );
        this.b.fromBufferAttribute( attribute, i1 );
        this.c.fromBufferAttribute( attribute, i2 );
        return this;
    }
    clone() {
        return new this.constructor().copy( this );
    }
    copy( triangle ) {
        this.a.copy( triangle.a );
        this.b.copy( triangle.b );
        this.c.copy( triangle.c );
        return this;
    }
    getArea() {
        _tri_v0.subVectors( this.c, this.b );
        _tri_v1.subVectors( this.a, this.b );
        return _tri_v0.cross( _tri_v1 ).length() * 0.5;
    }
    getMidpoint( target ) {
        return target.addVectors( this.a, this.b ).add( this.c ).multiplyScalar( 1 / 3 );
    }
    getNormal( target ) {
        return Triangle.getNormal( this.a, this.b, this.c, target );
    }
    getPlane( target ) {
        return target.setFromCoplanarPoints( this.a, this.b, this.c );
    }
    getBarycoord( point, target ) {
        return Triangle.getBarycoord( point, this.a, this.b, this.c, target );
    }
    getInterpolation( point, v1, v2, v3, target ) {
        return Triangle.getInterpolation( point, this.a, this.b, this.c, v1, v2, v3, target );
    }
    containsPoint( point ) {
        return Triangle.containsPoint( point, this.a, this.b, this.c );
    }
    isFrontFacing( direction ) {
        return Triangle.isFrontFacing( this.a, this.b, this.c, direction );
    }
    intersectsBox( box ) {
        return box.intersectsTriangle( this );
    }
    closestPointToPoint( p, target ) {
        const a = this.a, b = this.b, c = this.c;
        let v, w;
        // algorithm thanks to Real-Time Collision Detection by Christer Ericson,
        // published by Morgan Kaufmann Publishers, (c) 2005 Elsevier Inc.,
        // under the accompanying license; see chapter 5.1.5 for detailed explanation.
        // basically, we're distinguishing which of the voronoi regions of the triangle
        // the point lies in with the minimum amount of redundant computation.
        _tri_vab.subVectors( b, a );
        _tri_vac.subVectors( c, a );
        _tri_vap.subVectors( p, a );
        const d1 = _tri_vab.dot( _tri_vap );
        const d2 = _tri_vac.dot( _tri_vap );
        if ( d1 <= 0 && d2 <= 0 ) {
            // vertex region of A; barycentric coords (1, 0, 0)
            return target.copy( a );
        }
        _tri_vbp.subVectors( p, b );
        const d3 = _tri_vab.dot( _tri_vbp );
        const d4 = _tri_vac.dot( _tri_vbp );
        if ( d3 >= 0 && d4 <= d3 ) {
            // vertex region of B; barycentric coords (0, 1, 0)
            return target.copy( b );
        }
        const vc = d1 * d4 - d3 * d2;
        if ( vc <= 0 && d1 >= 0 && d3 <= 0 ) {
            v = d1 / ( d1 - d3 );
            // edge region of AB; barycentric coords (1-v, v, 0)
            return target.copy( a ).addScaledVector( _tri_vab, v );
        }
        _tri_vcp.subVectors( p, c );
        const d5 = _tri_vab.dot( _tri_vcp );
        const d6 = _tri_vac.dot( _tri_vcp );
        if ( d6 >= 0 && d5 <= d6 ) {
            // vertex region of C; barycentric coords (0, 0, 1)
            return target.copy( c );
        }
        const vb = d5 * d2 - d1 * d6;
        if ( vb <= 0 && d2 >= 0 && d6 <= 0 ) {
            w = d2 / ( d2 - d6 );
            // edge region of AC; barycentric coords (1-w, 0, w)
            return target.copy( a ).addScaledVector( _tri_vac, w );
        }
        const va = d3 * d6 - d5 * d4;
        if ( va <= 0 && ( d4 - d3 ) >= 0 && ( d5 - d6 ) >= 0 ) {
            _tri_vbc.subVectors( c, b );
            w = ( d4 - d3 ) / ( ( d4 - d3 ) + ( d5 - d6 ) );
            // edge region of BC; barycentric coords (0, 1-w, w)
            return target.copy( b ).addScaledVector( _tri_vbc, w ); // edge region of BC
        }
        // face region
        const denom = 1 / ( va + vb + vc );
        // u = va * denom
        v = vb * denom;
        w = vc * denom;
        return target.copy( a ).addScaledVector( _tri_vab, v ).addScaledVector( _tri_vac, w );
    }
    equals( triangle ) {
        return triangle.a.equals( this.a ) && triangle.b.equals( this.b ) && triangle.c.equals( this.c );
    }
}
export class Frustum {
    constructor( p0 = new Plane(), p1 = new Plane(), p2 = new Plane(), p3 = new Plane(), p4 = new Plane(), p5 = new Plane() ) {
        this.planes = [ p0, p1, p2, p3, p4, p5 ];
    }
    set( p0, p1, p2, p3, p4, p5 ) {
        const planes = this.planes;
        planes[ 0 ].copy( p0 );
        planes[ 1 ].copy( p1 );
        planes[ 2 ].copy( p2 );
        planes[ 3 ].copy( p3 );
        planes[ 4 ].copy( p4 );
        planes[ 5 ].copy( p5 );
        return this;
    }
    copy( frustum ) {
        const planes = this.planes;
        for ( let i = 0; i < 6; i ++ ) {
            planes[ i ].copy( frustum.planes[ i ] );
        }
        return this;
    }
    setFromProjectionMatrix( m, coordinateSystem = 2000 ) {
        const planes = this.planes;
        const me = m.elements;
        const me0 = me[ 0 ], me1 = me[ 1 ], me2 = me[ 2 ], me3 = me[ 3 ];
        const me4 = me[ 4 ], me5 = me[ 5 ], me6 = me[ 6 ], me7 = me[ 7 ];
        const me8 = me[ 8 ], me9 = me[ 9 ], me10 = me[ 10 ], me11 = me[ 11 ];
        const me12 = me[ 12 ], me13 = me[ 13 ], me14 = me[ 14 ], me15 = me[ 15 ];
        planes[ 0 ].setComponents( me3 - me0, me7 - me4, me11 - me8, me15 - me12 ).normalize();
        planes[ 1 ].setComponents( me3 + me0, me7 + me4, me11 + me8, me15 + me12 ).normalize();
        planes[ 2 ].setComponents( me3 + me1, me7 + me5, me11 + me9, me15 + me13 ).normalize();
        planes[ 3 ].setComponents( me3 - me1, me7 - me5, me11 - me9, me15 - me13 ).normalize();
        planes[ 4 ].setComponents( me3 - me2, me7 - me6, me11 - me10, me15 - me14 ).normalize();
        if ( coordinateSystem === 2000 ) {
            planes[ 5 ].setComponents( me3 + me2, me7 + me6, me11 + me10, me15 + me14 ).normalize();
        } else if ( coordinateSystem === 2001 ) {
            planes[ 5 ].setComponents( me2, me6, me10, me14 ).normalize();
        } else {
            throw new Error( 'THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: ' + coordinateSystem );
        }
        return this;
    }
    intersectsObject( object ) {
        if ( object.boundingSphere !== undefined ) {
            if ( object.boundingSphere === null ) object.computeBoundingSphere();
            _frustum_sphere.copy( object.boundingSphere ).applyMatrix4( object.matrixWorld );
        } else {
            _frustum_sphere.copy( _geometryBoundingSphere( _boundsGeometryOf( object ) ) ).applyMatrix4( object.matrixWorld );
        }
        return this.intersectsSphere( _frustum_sphere );
    }
    intersectsSprite( sprite ) {
        _frustum_sphere.center.set( 0, 0, 0 );
        _frustum_sphere.radius = 0.7071067811865476;
        _frustum_sphere.applyMatrix4( sprite.matrixWorld );
        return this.intersectsSphere( _frustum_sphere );
    }
    intersectsSphere( sphere ) {
        const planes = this.planes;
        const center = sphere.center;
        const negRadius = - sphere.radius;
        for ( let i = 0; i < 6; i ++ ) {
            const distance = planes[ i ].distanceToPoint( center );
            if ( distance < negRadius ) {
                return false;
            }
        }
        return true;
    }
    intersectsBox( box ) {
        const planes = this.planes;
        for ( let i = 0; i < 6; i ++ ) {
            const plane = planes[ i ];
            // corner at max distance
            _frustum_vector.x = plane.normal.x > 0 ? box.max.x : box.min.x;
            _frustum_vector.y = plane.normal.y > 0 ? box.max.y : box.min.y;
            _frustum_vector.z = plane.normal.z > 0 ? box.max.z : box.min.z;
            if ( plane.distanceToPoint( _frustum_vector ) < 0 ) {
                return false;
            }
        }
        return true;
    }
    containsPoint( point ) {
        const planes = this.planes;
        for ( let i = 0; i < 6; i ++ ) {
            if ( planes[ i ].distanceToPoint( point ) < 0 ) {
                return false;
            }
        }
        return true;
    }
    clone() {
        return new this.constructor().copy( this );
    }
}
export class Spherical {
    constructor( radius = 1, phi = 0, theta = 0 ) {
        this.radius = radius;
        this.phi = phi; // polar angle
        this.theta = theta; // azimuthal angle
        return this;
    }
    set( radius, phi, theta ) {
        this.radius = radius;
        this.phi = phi;
        this.theta = theta;
        return this;
    }
    copy( other ) {
        this.radius = other.radius;
        this.phi = other.phi;
        this.theta = other.theta;
        return this;
    }
    // restrict phi to be between EPS and PI-EPS
    makeSafe() {
        const EPS = 0.000001;
        this.phi = Math.max( EPS, Math.min( Math.PI - EPS, this.phi ) );
        return this;
    }
    setFromVector3( v ) {
        return this.setFromCartesianCoords( v.x, v.y, v.z );
    }
    setFromCartesianCoords( x, y, z ) {
        this.radius = Math.sqrt( x * x + y * y + z * z );
        if ( this.radius === 0 ) {
            this.theta = 0;
            this.phi = 0;
        } else {
            this.theta = Math.atan2( x, z );
            this.phi = Math.acos( MathUtils.clamp( y / this.radius, - 1, 1 ) );
        }
        return this;
    }
    clone() {
        return new this.constructor().copy( this );
    }
}
export class Cylindrical {
    constructor( radius = 1, theta = 0, y = 0 ) {
        this.radius = radius; // distance from the origin to a point in the x-z plane
        this.theta = theta; // counterclockwise angle in the x-z plane measured in radians from the positive z-axis
        this.y = y; // height above the x-z plane
        return this;
    }
    set( radius, theta, y ) {
        this.radius = radius;
        this.theta = theta;
        this.y = y;
        return this;
    }
    copy( other ) {
        this.radius = other.radius;
        this.theta = other.theta;
        this.y = other.y;
        return this;
    }
    setFromVector3( v ) {
        return this.setFromCartesianCoords( v.x, v.y, v.z );
    }
    setFromCartesianCoords( x, y, z ) {
        this.radius = Math.sqrt( x * x + z * z );
        this.theta = Math.atan2( x, z );
        this.y = y;
        return this;
    }
    clone() {
        return new this.constructor().copy( this );
    }
}
export class Line3 {
    constructor( start = new Vector3(), end = new Vector3() ) {
        this.start = start;
        this.end = end;
    }
    set( start, end ) {
        this.start.copy( start );
        this.end.copy( end );
        return this;
    }
    copy( line ) {
        this.start.copy( line.start );
        this.end.copy( line.end );
        return this;
    }
    getCenter( target ) {
        return target.addVectors( this.start, this.end ).multiplyScalar( 0.5 );
    }
    delta( target ) {
        return target.subVectors( this.end, this.start );
    }
    distanceSq() {
        return this.start.distanceToSquared( this.end );
    }
    distance() {
        return this.start.distanceTo( this.end );
    }
    at( t, target ) {
        return this.delta( target ).multiplyScalar( t ).add( this.start );
    }
    closestPointToPointParameter( point, clampToLine ) {
        _line3_startP.subVectors( point, this.start );
        _line3_startEnd.subVectors( this.end, this.start );
        const startEnd2 = _line3_startEnd.dot( _line3_startEnd );
        const startEnd_startP = _line3_startEnd.dot( _line3_startP );
        let t = startEnd_startP / startEnd2;
        if ( clampToLine ) {
            t = MathUtils.clamp( t, 0, 1 );
        }
        return t;
    }
    closestPointToPoint( point, clampToLine, target ) {
        const t = this.closestPointToPointParameter( point, clampToLine );
        return this.delta( target ).multiplyScalar( t ).add( this.start );
    }
    applyMatrix4( matrix ) {
        this.start.applyMatrix4( matrix );
        this.end.applyMatrix4( matrix );
        return this;
    }
    equals( line ) {
        return line.start.equals( this.start ) && line.end.equals( this.end );
    }
    clone() {
        return new this.constructor().copy( this );
    }
}
// Scratch values of the math classes above (three.js keeps the same ones at
// module level), declared once every class they construct exists.
const _v3_vector = new Vector3();
const _v3_quaternion = new Quaternion();
const _m3_m3 = new Matrix3();
const _box2_vector = new Vector2();
const _box3_points = [
    new Vector3(),
    new Vector3(),
    new Vector3(),
    new Vector3(),
    new Vector3(),
    new Vector3(),
    new Vector3(),
    new Vector3()
];

const _box3_vector = new Vector3();

const _box3_box = new Box3();

// triangle centered vertices

const _box3_v0 = new Vector3();
const _box3_v1 = new Vector3();
const _box3_v2 = new Vector3();

// triangle edge vectors

const _box3_f0 = new Vector3();
const _box3_f1 = new Vector3();
const _box3_f2 = new Vector3();

const _box3_center = new Vector3();
const _box3_extents = new Vector3();
const _box3_triangleNormal = new Vector3();
const _box3_testAxis = new Vector3();

function _box3_satForAxes( axes, v0, v1, v2, extents ) {

    for ( let i = 0, j = axes.length - 3; i <= j; i += 3 ) {

        _box3_testAxis.fromArray( axes, i );
        // project the aabb onto the separating axis
        const r = extents.x * Math.abs( _box3_testAxis.x ) + extents.y * Math.abs( _box3_testAxis.y ) + extents.z * Math.abs( _box3_testAxis.z );
        // project all 3 vertices of the triangle onto the separating axis
        const p0 = v0.dot( _box3_testAxis );
        const p1 = v1.dot( _box3_testAxis );
        const p2 = v2.dot( _box3_testAxis );
        // actual test, basically see if either of the most extreme of the triangle points intersects r
        if ( Math.max( - Math.max( p0, p1, p2 ), Math.min( p0, p1, p2 ) ) > r ) {

            // points of the projected triangle are outside the projected half-length of the aabb
            // the axis is separating and we can exit
            return false;

        }

    }

    return true;

}
const _sphere_box = new Box3();
const _sphere_v1 = new Vector3();
const _sphere_v2 = new Vector3();
const _ray_vector = new Vector3();
const _ray_segCenter = new Vector3();
const _ray_segDir = new Vector3();
const _ray_diff = new Vector3();

const _ray_edge1 = new Vector3();
const _ray_edge2 = new Vector3();
const _ray_normal = new Vector3();
const _plane_vector1 = new Vector3();
const _plane_vector2 = new Vector3();
const _plane_normalMatrix = new Matrix3();
const _tri_v0 = new Vector3();
const _tri_v1 = new Vector3();
const _tri_v2 = new Vector3();
const _tri_v3 = new Vector3();

const _tri_vab = new Vector3();
const _tri_vac = new Vector3();
const _tri_vbc = new Vector3();
const _tri_vap = new Vector3();
const _tri_vbp = new Vector3();
const _tri_vcp = new Vector3();
const _frustum_sphere = new Sphere();
const _frustum_vector = new Vector3();
const _line3_startP = new Vector3();
const _line3_startEnd = new Vector3();

// ---- Bounds from scene data (Rust: `src/wasm/math.rs`) ----
/// three.js's shared Sprite geometry (a unit quad), for the bounds of a
/// sprite — the shim's Sprite draws it in Rust and has no geometry of its own.
let _spriteBoundsGeometry = null;
/// The geometry three.js would read an object's bounds from.
function _boundsGeometryOf( object ) {
    if ( object.geometry !== undefined ) return object.geometry;
    if ( object.isSprite !== true ) return undefined;
    if ( _spriteBoundsGeometry === null ) {
        _spriteBoundsGeometry = new BufferGeometry();
        _spriteBoundsGeometry.setAttribute( 'position', new BufferAttribute( new Float32Array( [ - 0.5, - 0.5, 0, 0.5, - 0.5, 0, 0.5, 0.5, 0, - 0.5, 0.5, 0 ] ), 3 ) );
    }
    return _spriteBoundsGeometry;
}
/// `geometry.getAttribute( 'position' ) !== undefined`, for a built-in whose
/// positions live in Rust too.
function _boundsHavePositions( geometry ) {
    if ( geometry === null ) return false;
    if ( geometry.attributes !== undefined ) return geometry.attributes.position !== undefined;
    return _wasmGeometryOf( geometry ) !== null;
}
/// `BufferGeometry.computeBoundingBox()` over the geometry's Rust build.
function _computeGeometryBox( geometry, target ) {
    const b = _wasmGeometryOf( geometry )?.threeBoundingBox();
    if ( b === undefined ) return target.makeEmpty();
    target.min.set( b[ 0 ], b[ 1 ], b[ 2 ] );
    target.max.set( b[ 3 ], b[ 4 ], b[ 5 ] );
    return target;
}
/// `BufferGeometry.computeBoundingSphere()` over the geometry's Rust build
/// (the sphere is left as it is without positions, as in three.js).
function _computeGeometrySphere( geometry, target ) {
    const s = _wasmGeometryOf( geometry )?.threeBoundingSphere();
    if ( s !== undefined ) {
        target.center.set( s[ 0 ], s[ 1 ], s[ 2 ] );
        target.radius = s[ 3 ];
    }
    return target;
}
/// `geometry.boundingBox`, computed when null (as three.js does before
/// reading it); a built-in geometry without the method caches one too.
function _geometryBoundingBox( geometry ) {
    if ( geometry.boundingBox === null || geometry.boundingBox === undefined ) {
        if ( typeof geometry.computeBoundingBox === 'function' ) geometry.computeBoundingBox();
        else geometry.boundingBox = _computeGeometryBox( geometry, new Box3() );
    }
    return geometry.boundingBox;
}
function _geometryBoundingSphere( geometry ) {
    if ( geometry.boundingSphere === null || geometry.boundingSphere === undefined ) {
        if ( typeof geometry.computeBoundingSphere === 'function' ) geometry.computeBoundingSphere();
        else geometry.boundingSphere = _computeGeometrySphere( geometry, new Sphere() );
    }
    return geometry.boundingSphere;
}
const _skinBoundsMatrix = new Matrix4();
// InstancedMesh.computeBoundingBox / computeBoundingSphere scratch.
const _instanceLocalMatrix = new Matrix4();
const _instanceBox = new Box3();
const _instanceSphere = new Sphere();
/// The bounds of an object's vertices one by one, from its mirrored node
/// (`WebScene.vertexBounds`): `mode` 0 a box `[min, max]`, 1 a sphere
/// `[center, radius]`; placed by `matrix` when given. A mesh's vertices are
/// three.js's `getVertexPosition`: morphed by `morphTargetInfluences` and,
/// for a SkinnedMesh, skinned by its bones' current `matrixWorld`.
/// Undefined when the node has no positions.
function _objectVertexBounds( object, mode, matrix ) {
    const handle = object._handle;
    if ( ! handle ) return undefined;
    const elements = matrix ? Float64Array.from( matrix.elements ) : undefined;
    let influences, skin;
    if ( object.isMesh === true ) {
        const mi = object.morphTargetInfluences;
        if ( mi ) influences = Float64Array.from( mi );
        const skeleton = object.isSkinnedMesh === true ? object.skeleton : undefined;
        if ( skeleton ) {
            const bones = skeleton.bones;
            skin = new Float64Array( 32 + bones.length * 16 );
            skin.set( object.bindMatrix.elements, 0 );
            skin.set( object.bindMatrixInverse.elements, 16 );
            for ( let i = 0; i < bones.length; i ++ ) {
                _skinBoundsMatrix.multiplyMatrices( bones[ i ].matrixWorld, skeleton.boneInverses[ i ] );
                skin.set( _skinBoundsMatrix.elements, 32 + i * 16 );
            }
        }
    }
    return object._m.graph.w.vertexBounds( handle, mode, elements, influences, skin );
}


// ---- Geometries ----
export class BoxLineGeometry { constructor(w = 1, h = 1, d = 1) { this._w = WebGeometry.boxLine(w, h, d); } }

// ---- Materials ----
function _matColor(opts) {
    if (typeof opts === 'object' && opts !== null && 'color' in opts) return _color(opts.color);
    return _color(opts);
}
// Apply `opts.map` (a Texture / DataTexture) to a WebMaterial. three.js's
// MeshBasic/Standard/Physical/Sprite all accept `{ map: tex }`; we route to
// either setMap (Texture) or setMapData (DataTexture) based on which wasm
// handle the JS class is wrapping.
function _applyMap(w, opts) {
    if (!opts || !opts.map) return;
    const t = opts.map;
    _applyTextureColorSpace(t);
    // Sync the texture's filter/wrap and uv matrix to the wasm Texture before
    // binding it.
    _syncTextureFilters(t);
    _syncTextureUv(t);
    // CanvasTexture also wraps a WebDataTexture; route either via setMapData.
    if (t?._w?.constructor?.name === 'WebDataTexture' || t instanceof DataTexture || t instanceof CanvasTexture) w.setMapData(t._w);
    else if (t instanceof Texture) w.setMap(t._w);
}
// Translate three.js's numeric filter/wrap constants to our compact
// (mag, min, wrap_s, wrap_t) enum and push to the wasm Texture.
function _syncTextureFilters(t) {
    if (!t?._w?.setFilters) return false;
    // three.js's own constants: wasm reads the filters (all six `minFilter`s,
    // mipmapped or not) and wraps as three.js defines them.
    const codes = [t.magFilter ?? 1006, t.minFilter ?? 1008, t.wrapS ?? 1001, t.wrapT ?? 1001];
    const last = t._filtersSynced;
    if (last && last.w === t._w && codes.every((v, i) => v === last.codes[i])) return false;
    t._filtersSynced = { w: t._w, codes };
    t._w.setFilters(...codes);
    return true;
}
// The top two rows of three.js's `Texture.matrix` for `t`: `offset`, `repeat`,
// `rotation` and `center` through `Matrix3.setUvTransform`, or `t.matrix`
// itself with `matrixAutoUpdate` off. Canvas and video pixels arrive top row
// first, so with `flipY` (their default) v is read the other way up — what
// three.js's flip on upload does.
function _textureUvRows(t) {
    let a, b, c, d, e, f;
    const m = t.matrix?.elements;
    if (t.matrixAutoUpdate === false && m) {
        [a, b, c, d, e, f] = [m[0], m[3], m[6], m[1], m[4], m[7]];
    } else {
        const tx = t.offset?.x ?? 0, ty = t.offset?.y ?? 0;
        const sx = t.repeat?.x ?? 1, sy = t.repeat?.y ?? 1;
        const cx = t.center?.x ?? 0, cy = t.center?.y ?? 0;
        const r = t.rotation ?? 0, cs = Math.cos(r), sn = Math.sin(r);
        a = sx * cs; b = sx * sn; c = -sx * (cs * cx + sn * cy) + cx + tx;
        d = -sy * sn; e = sy * cs; f = -sy * (-sn * cx + cs * cy) + cy + ty;
    }
    if (t._topRowFirst && t.flipY !== false) { d = -d; e = -e; f = 1 - f; }
    return [a, b, c, d, e, f];
}
// Push `t`'s uv matrix to its wasm texture if it changed since the last push
// (or the wasm texture was replaced); whether it did.
function _syncTextureUv(t) {
    const w = t?._w;
    if (!w?.setUvMatrix) return false;
    const rows = _textureUvRows(t);
    const last = t._uvSynced;
    if (last && last.w === w && rows.every((v, i) => v === last.rows[i])) return false;
    t._uvSynced = { w, rows };
    w.setUvMatrix(...rows);
    // A replaced wasm texture (a loader's image arriving) starts on `uv`.
    if (t._channel) w.setChannel?.(t._channel);
    return true;
}
// three.js reads every texture's matrix each frame, so `map.offset.x += 0.01`
// scrolls a map with no `needsUpdate`; and it sets a texture's filter and wrap
// when it first uploads it, so changes made before the first render apply
// without one. Before each render: any map of `material` whose matrix moved,
// or that has not been drawn yet and whose sampler state changed, is pushed
// and re-bound wherever it is held.
function _syncMaterialTextures(material) {
    if (_syncEnvRotation(material)) _materialChanged(material);
    const keys = material?._mapKeys;
    if (!keys) return;
    for (const key of keys) {
        const tex = material[key];
        if (!tex) continue;
        let changed = _syncTextureUv(tex);
        if (!tex._drawn) {
            changed = _syncTextureFilters(tex) || changed;
            tex._drawn = true;
        }
        if (changed) _rebindTexture(tex);
    }
}

// Apply the cross-material `transparent`/`opacity`/`wireframe`/`emissive`
// flags, mirroring three.js's MeshXxxMaterial constructor options.
function _applyCommon(w, opts) {
    if (!opts || typeof opts !== 'object') return;
    if (typeof opts.opacity === 'number') w.setOpacity(opts.opacity);
    if (opts.transparent === true) w.setTransparent(true);
    if (opts.wireframe === true && w.setWireframe) w.setWireframe(true);
    if ('emissive' in opts && w.setEmissive) w.setEmissive(_color(opts.emissive));
    if (typeof opts.emissiveIntensity === 'number' && w.setEmissiveIntensity) {
        w.setEmissiveIntensity(opts.emissiveIntensity);
    }
}
// Mix Material's cross-cutting fields (onBeforeCompile, userData, uuid, …)
// into any concrete material class. Idempotent.
function _initMaterialBase(self, opts = {}) {
    // Push side through to wasm so BackSide / DoubleSide route to the no-cull
    // triangle pipeline. (FrontSide is the default; no setter needed.)
    const sideVal = opts.side ?? 0;
    if (sideVal !== 0 && self._w?.setSide) {
        self._w.setSide(sideVal);
    }
    self.uuid = self.uuid || MathUtils.generateUUID();
    self.name = opts.name ?? self.name ?? '';
    self.userData = self.userData || {};
    // three.js clipping, read by the renderer every frame (`_syncClipPlane`);
    // `clipIntersection` / `clipShadows` are kept for code that reads them.
    if (self.clippingPlanes === undefined) self.clippingPlanes = opts.clippingPlanes ?? null;
    if (self.clipIntersection === undefined) self.clipIntersection = opts.clipIntersection ?? false;
    if (self.clipShadows === undefined) self.clipShadows = opts.clipShadows ?? false;
    self.onBeforeCompile = self.onBeforeCompile || null;
    // three.js `alphaTest` and `transparent`, live: the wasm material takes
    // them when they change. `transparent` always goes across as three.js's
    // flag, so an opaque material at `opacity < 1` stays opaque, as in three.js.
    if (!Object.getOwnPropertyDescriptor(self, 'alphaTest')?.get) {
        let alphaTest = Number(opts.alphaTest ?? self.alphaTest ?? 0) || 0;
        Object.defineProperty(self, 'alphaTest', {
            get: () => alphaTest,
            set: (v) => {
                const next = Number(v) || 0;
                if (next === alphaTest) return;
                alphaTest = next;
                self._w?.setAlphaTest?.(alphaTest);
                _materialChanged(self);
            },
            enumerable: true, configurable: true,
        });
        if (alphaTest > 0) self._w?.setAlphaTest?.(alphaTest);
    }
    if (!Object.getOwnPropertyDescriptor(self, 'transparent')?.get) {
        let transparent = !!(opts.transparent ?? (typeof self.transparent === 'boolean' ? self.transparent : false));
        Object.defineProperty(self, 'transparent', {
            get: () => transparent,
            set: (v) => {
                const next = !!v;
                if (next === transparent) return;
                transparent = next;
                self._w?.setTransparent?.(transparent);
                _materialChanged(self);
            },
            enumerable: true, configurable: true,
        });
        self._w?.setTransparent?.(transparent);
    }
    const initColor = (typeof opts === 'object' && opts && 'color' in opts) ? opts.color : 0xffffff;
    // three.js copies a Color it is given (`material.color.set(color)`).
    self.color = initColor?.isColor ? new Color(initColor.r, initColor.g, initColor.b) : new Color(initColor);
    // three.js `emissive`, on the materials whose wasm side takes one.
    if (self.emissive === undefined && (self instanceof MeshStandardMaterial || self instanceof MeshPhysicalMaterial
        || self instanceof MeshLambertMaterial)) {
        self.emissive = new Color(('emissive' in opts && opts.emissive != null) ? opts.emissive : 0x000000);
    }
    // `color` / `emissive` are plain three.js Colors, mutated in place by user
    // code; what the wasm material was last given is remembered here and
    // compared at each render (`_syncMaterialColors`), as three.js re-reads
    // its uniforms every frame.
    self._wColors = [['color', 'setColor'], ['emissive', 'setEmissive']]
        .filter(([key]) => self[key]?.isColor)
        .map(([key, setter]) => { const c = self[key]; return { key, setter, obj: c, r: c.r, g: c.g, b: c.b }; });
    // three.js `polygonOffset` / `polygonOffsetFactor` / `polygonOffsetUnits`,
    // forwarded to the wasm material whenever one of them changes.
    if (!Object.getOwnPropertyDescriptor(self, 'polygonOffset')) {
        const po = { on: opts.polygonOffset === true, factor: opts.polygonOffsetFactor ?? 0, units: opts.polygonOffsetUnits ?? 0 };
        const push = () => self._w?.setPolygonOffset?.(po.on, po.factor, po.units);
        for (const [k, f] of [['polygonOffset', 'on'], ['polygonOffsetFactor', 'factor'], ['polygonOffsetUnits', 'units']]) {
            Object.defineProperty(self, k, {
                get: () => po[f],
                set: (v) => { po[f] = f === 'on' ? !!v : Number(v) || 0; push(); },
                enumerable: true, configurable: true,
            });
        }
        if (po.on) push();
    }
    // three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`: the
    // wasm material's pipeline state, pushed whenever one of them changes
    // (and again by anything that rebuilds `_w`, via `_pushRenderState`).
    // three.js's defaults go across as they are — a transparent material
    // writes depth unless `depthWrite` is false, as in three.js.
    if (!Object.getOwnPropertyDescriptor(self, 'depthTest')) {
        const rs = {
            depthTest: opts.depthTest ?? true,
            depthWrite: opts.depthWrite ?? true,
            depthFunc: opts.depthFunc ?? LessEqualDepth,
            colorWrite: opts.colorWrite ?? true,
        };
        self._pushRenderState = () => {
            self._w?.setRenderState?.(rs.depthTest, rs.depthWrite, rs.depthFunc, rs.colorWrite);
            // A rebuilt wasm material starts from defaults: carry these too.
            self._w?.setAlphaTest?.(self.alphaTest ?? 0);
            if (typeof self.transparent === 'boolean') self._w?.setTransparent?.(self.transparent);
        };
        for (const k of Object.keys(rs)) {
            Object.defineProperty(self, k, {
                get: () => rs[k],
                set: (v) => {
                    const next = k === 'depthFunc' ? Number(v) : !!v;
                    if (next === rs[k]) return;
                    rs[k] = next;
                    self._pushRenderState();
                    _materialChanged(self);
                },
                enumerable: true, configurable: true,
            });
        }
        self._pushRenderState();
    }
    // three.js `toneMapped`, forwarded to the wasm material when it changes.
    if (!Object.getOwnPropertyDescriptor(self, 'toneMapped')) {
        let toneMapped = opts.toneMapped ?? true;
        Object.defineProperty(self, 'toneMapped', {
            get: () => toneMapped,
            set: (v) => { toneMapped = !!v; self._w?.setToneMapped?.(toneMapped); _materialChanged(self); },
            enumerable: true, configurable: true,
        });
        if (!toneMapped) self._w?.setToneMapped?.(false);
    }
    self.blending = opts.blending ?? 1;
    self.side = opts.side ?? 0;
    self.opacity = opts.opacity ?? 1;
    self.visible = opts.visible ?? true;
    self.wireframe = opts.wireframe ?? false;
    self.version = 0;
    // three.js `material.needsUpdate = true`: the objects drawn with it take its
    // current wasm state at the next sync.
    Object.defineProperty(self, 'needsUpdate', {
        get() { return false; },
        set(v) {
            if (v !== true) return;
            self.version++;
            // `opacity` is a plain field (an animation track writes it, then
            // sets needsUpdate): the material re-reads it here.
            if (typeof self.opacity === 'number' && self._w?.setOpacity && !Object.getOwnPropertyDescriptor(self, 'opacity')?.set) {
                self._w.setOpacity(self.opacity);
            }
            _materialChanged(self);
        },
        configurable: true,
    });
    self._compiledOnce = false;
    // Provide `_runOnBeforeCompile` and `customProgramCacheKey` so calling
    // them on any material is safe regardless of class inheritance.
    if (!self._runOnBeforeCompile) {
        self._runOnBeforeCompile = function () {
            if (typeof this.onBeforeCompile !== 'function' || this._compiledOnce) return;
            const shader = { uniforms: { ...this.uniforms || {} }, vertexShader: '/* threers wgsl */', fragmentShader: '/* threers wgsl */', defines: {} };
            try { this.onBeforeCompile(shader, this); } catch (_e) {}
            this.userData.shader = shader;
            this.uniforms = shader.uniforms;
            this._compiledOnce = true;
        };
    }
    if (!self.customProgramCacheKey) self.customProgramCacheKey = () => self.uuid;
    if (!self.dispose) self.dispose = () => {};
    return self;
}
/**
 * A custom-shader material.
 *
 * three.js's `ShaderMaterial` takes GLSL; this takes **WGSL**, compiled against
 * the renderer's own preamble, so `VsOut`, the `frame` and `mesh` uniforms,
 * `sample_env_cube`, `apply_tone_mapping` and `framebuffer_encode` are all in
 * scope. That is not a drop-in for a three.js shader — the language is
 * different — but it is the same place in the API, and until now there was no
 * place at all.
 *
 *     const m = new THREE.ShaderMaterial({
 *       fragment: `
 *         @fragment
 *         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
 *           let tint = u_data.data[0].rgb;
 *           return vec4<f32>(framebuffer_encode(apply_tone_mapping(tint)), 1.0);
 *         }`,
 *       uniforms: [1.0, 0.3, 0.1, 1.0],
 *     });
 *     scene.add(new THREE.Mesh(geometry, m));
 *
 * `vertex` replaces the vertex stage with `@vertex fn vs_custom`, for anything
 * whose position has to be computed rather than read. `storage0`..`storage2`
 * are flat float arrays, read in the shader as `s0(i)`/`s1(i)`/`s2(i)` — use
 * those rather than `u_s0[i]`, because WebGL2 has no storage buffers and the
 * accessors fall back to a texture there while a raw index will not compile.
 */
export class ShaderMaterial {
    constructor(opts = {}) {
        // A bare string is the common case and reads better than `{fragment}`.
        const o = typeof opts === 'string' ? { fragment: opts } : (opts || {});
        this._sm = new WebShaderMaterial(o.fragment ?? o.fragmentShader ?? '');
        if (o.vertex ?? o.vertexShader) this._sm.setVertex(o.vertex ?? o.vertexShader);
        // `uniforms` as three.js spells it is `{ name: { value } }`, which has
        // no meaning against a WGSL uniform block — there are no names to bind
        // to. An array is this renderer's spelling and goes straight in.
        if (Array.isArray(o.uniforms) || ArrayBuffer.isView(o.uniforms)) {
            this._sm.setData(Float32Array.from(o.uniforms));
        }
        for (let i = 0; i < 3; i++) {
            const data = o['storage' + i];
            if (data) this._sm.setStorage(i, Float32Array.from(data));
        }
        if (o.transparent) this._sm.setTransparent(true);
        if (o.opacity !== undefined) this._sm.setOpacity(o.opacity);
        if (o.side) this._sm.setSide(o.side);
        if (o.depthWrite !== undefined) this._sm.setDepthWrite(!!o.depthWrite);
        if (o.screenSpace) this._sm.setScreenSpace(true);
        // three.js `alphaToCoverage` (the WGSL returns `coverage_alpha(c)`,
        // or its samples through `frag_out`).
        if (o.alphaToCoverage) this._sm.setAlphaToCoverage(true);
        // three.js Data3DTextures for u_vol0 / u_vol1 (uploaded on the GPU the
        // renderer opened — at once if it exists, else on the first render)
        this._volumes = [null, null];
        this._volumeSent = [null, null];
        (o.volumes ?? []).slice(0, 2).forEach((t, i) => { this._volumes[i] = t ?? null; });
        this._w = this._sm.material();
        // The surface three.js code reads off a material, kept so existing code
        // does not throw on a property access.
        this.uniforms = o.uniforms ?? {};
        this.vertexShader = o.vertex ?? o.vertexShader ?? '';
        this.fragmentShader = o.fragment ?? o.fragmentShader ?? '';
        this.defines = o.defines ?? {};
        this.transparent = !!o.transparent;
        _initMaterialBase(this, o);
        this._syncVolumes();
    }

    /// Bind a `Data3DTexture` as `u_vol0` (`slot` 0) or `u_vol1` (1); `null`
    /// unbinds it. 8-bit red or RGBA, or one-channel float (kept as half
    /// float, the float format every GPU filters).
    setVolume(slot, texture) {
        if (slot !== 0 && slot !== 1) throw new RangeError('ShaderMaterial volume slots are 0 and 1');
        this._volumes[slot] = texture ?? null;
        this._syncVolumes();
        return this;
    }

    /// Upload what changed since the last call: a new texture in a slot, a
    /// `needsUpdate`, or a slot cleared. Run every render by the renderer, so
    /// a volume set before the renderer existed lands on its first frame.
    _syncVolumes() {
        let changed = false;
        for (let i = 0; i < 2; i++) {
            const t = this._volumes[i];
            const sent = this._volumeSent[i];
            if (!t) {
                if (sent) { this._sm.clearVolume(i); this._volumeSent[i] = null; changed = true; }
                continue;
            }
            if (sent && sent.tex === t && sent.version === t.version) continue;
            const { data, width, height, depth } = t.image ?? {};
            try {
                if (t.type === 1015 || data instanceof Float32Array) {
                    this._sm.setVolumeF32(i, data instanceof Float32Array ? data : Float32Array.from(data), width, height, depth);
                } else {
                    const u8 = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : Uint8Array.from(data);
                    this._sm.setVolume(i, u8, width, height, depth, t.format === 1023 ? 4 : 1);
                }
            } catch (e) {
                // no renderer yet: try again next render; anything else is the
                // data's fault and would only repeat, so say it once
                if (!/create a WebGLRenderer/.test(String(e))) {
                    console.error('[threers] Data3DTexture upload failed:', e);
                    this._volumeSent[i] = { tex: t, version: t.version };
                }
                continue;
            }
            this._volumeSent[i] = { tex: t, version: t.version };
            changed = true;
        }
        if (changed) {
            this._w = this._sm.material();
            this._pushRenderState?.();
            _materialChanged(this);
        }
    }

    /**
     * Compile-check the shader. Returns the error, or `null`.
     *
     * Always `null` in a browser — see `ShaderMaterial.validationAvailable()`.
     * Native builds of the crate do check, so the useful place for this is a
     * Node test rather than a page.
     */
    validate() { return this._sm.validate() ?? null; }

    static validationAvailable() { return WebShaderMaterial.validationAvailable(); }

    /**
     * Re-read the uniform block. The material has to be rebuilt for the
     * renderer to see it, which is why this exists rather than a plain setter.
     */
    setUniforms(values) {
        this._sm.setData(Float32Array.from(values));
        this._w = this._sm.material();
        this._pushRenderState?.();
        _materialChanged(this);
        return this;
    }

    setStorage(slot, values) {
        this._sm.setStorage(slot, Float32Array.from(values));
        this._w = this._sm.material();
        this._pushRenderState?.();
        _materialChanged(this);
        return this;
    }
}

/** three.js distinguishes these by whether it injects its own preamble; this
 *  renderer always does, so there is nothing to distinguish. */
export class RawShaderMaterial extends ShaderMaterial {}

export class MeshBasicMaterial {
    constructor(opts = {}) {
        this._w = WebMaterial.basic(_matColor(opts));
        _applyCommon(this._w, opts); _initMaterialBase(this, opts);
        _bindMapSlot(this, 'map', 'setMap', 'setMapData', opts.map);
        _bindMapSlot(this, 'alphaMap', 'setAlphaMap', 'setAlphaMapData', opts.alphaMap);
        _bindLightBumpSpecular(this, opts, { bump: false, specular: true });
        _bindEnvMap(this, opts);
    }
}
export class MeshLambertMaterial { constructor(opts = {}) { this._w = WebMaterial.lambert(_matColor(opts)); _applyCommon(this._w, opts); _applyLitMaps(this, opts); _initMaterialBase(this, opts); } }
// `setMap` / `setMapData`: the wasm setter's name, or a function of the wasm
// material and texture.
function _bindDataTex(w, tex, setMap, setMapData) {
    if (!tex) return;
    _applyTextureColorSpace(tex);
    _syncTextureFilters(tex);
    _syncTextureUv(tex);
    const call = (set, t) => { if (set) (typeof set === 'function' ? set(w, t) : w[set](t)); };
    if (tex?._w?.constructor?.name === 'WebDataTexture' || tex instanceof DataTexture || tex instanceof CanvasTexture) {
        call(setMapData, tex._w);
    } else if (tex instanceof Texture) {
        call(setMap, tex._w);
    } else if (tex?._w) {
        call(setMapData, tex._w);
    }
}

/** Planet / PBR map options used by the earth demo. */
/// three.js `roughness` / `metalness` as live properties of a standard or
/// physical material.
function _pbrScalars(mat, roughness, metalness) {
    let r = roughness, m = metalness;
    Object.defineProperty(mat, 'roughness', {
        get() { return r; },
        set(v) { r = Number(v); mat._w?.setRoughness?.(r); _materialChanged(mat); },
        configurable: true, enumerable: true,
    });
    Object.defineProperty(mat, 'metalness', {
        get() { return m; },
        set(v) { m = Number(v); mat._w?.setMetalness?.(m); _materialChanged(mat); },
        configurable: true, enumerable: true,
    });
}
// A live three.js map slot (`mat.map = next` rebinds the GPU texture). The
// texture remembers the slot, so a change to it — its image arriving, its
// `colorSpace`, `needsUpdate` — binds it again (`_rebindTexture`).
function _bindMapSlot(mat, key, setMap, setMapData, value) {
    const bind = (tex) => {
        if (key === 'map') {
            _applyMap(mat._w, { map: tex });
        } else {
            _bindDataTex(mat._w, tex, setMap, setMapData);
        }
    };
    (mat._mapKeys ??= new Set()).add(key);
    Object.defineProperty(mat, key, {
        get() { return mat[`_${key}`]; },
        set(tex) {
            const had = mat[`_${key}`];
            mat[`_${key}`] = tex ?? null;
            if (!tex) {
                // three.js `material.map = null`: the slot empties.
                if (had && mat._w?.clearMap?.(key)) _materialChanged(mat);
                return;
            }
            bind(tex);
            const ref = mat._wref ??= new WeakRef(mat);
            const slots = tex._boundIn ??= new Map();
            slots.set(`${key}:${mat.uuid ?? ''}`, { ref, key, bind });
            _materialChanged(mat);
        },
        configurable: true,
        enumerable: true,
    });
    mat[key] = value ?? null;
}
// A texture changed under the materials it is bound into: bind it again in
// each slot that still holds it (dropping the slots that moved on).
function _rebindTexture(tex) {
    const slots = tex?._boundIn;
    if (!slots) return;
    for (const [id, b] of slots) {
        const mat = b.ref.deref();
        if (!mat || mat[b.key] !== tex) { slots.delete(id); continue; }
        b.bind(tex);
        _materialChanged(mat);
    }
}
// three.js `texture.colorSpace`: an sRGB colour map is decoded by the sampler;
// none (the default) or linear is read as it is.
function _applyTextureColorSpace(t) {
    if (t && typeof t.colorSpace === 'string') t._w?.setSrgb?.(t.colorSpace === 'srgb');
}
// three.js `aoMapIntensity` (default 1), live.
function _bindAoIntensity(mat, opts) {
    let v = typeof opts?.aoMapIntensity === 'number' ? opts.aoMapIntensity : 1;
    Object.defineProperty(mat, 'aoMapIntensity', {
        get() { return v; },
        set(x) { v = Number(x); mat._w?.setAoMapIntensity?.(v); _materialChanged(mat); },
        configurable: true, enumerable: true,
    });
    if (v !== 1) mat._w?.setAoMapIntensity?.(v);
}
// A numeric material property three.js has, live: `key` (default `def`) pushed
// through `setter` on change.
function _bindScalar(mat, key, setter, def, opts) {
    let v = typeof opts?.[key] === 'number' ? opts[key] : def;
    Object.defineProperty(mat, key, {
        get() { return v; },
        set(x) { v = Number(x); mat._w?.[setter]?.(v); _materialChanged(mat); },
        configurable: true, enumerable: true,
    });
    if (v !== def) mat._w?.[setter]?.(v);
}
// three.js `envMap`, `combine`, `reflectivity`, `refractionRatio` and
// `envMapRotation` on Basic, Lambert and Phong, live. The rotation is an Euler
// edited in place, so it is pushed before each render when it moved
// (`_syncMaterialTextures`).
function _bindEnvMap(mat, opts) {
    _bindMapSlot(mat, 'envMap', null, 'setEnvMap', opts.envMap);
    _bindScalar(mat, 'combine', 'setCombine', 0, opts);
    _bindScalar(mat, 'reflectivity', 'setReflectivity', 1, opts);
    _bindScalar(mat, 'refractionRatio', 'setRefractionRatio', 0.98, opts);
    const r = opts.envMapRotation;
    mat.envMapRotation = r?.isEuler ? r : new Euler(r?.x ?? 0, r?.y ?? 0, r?.z ?? 0, r?.order ?? 'XYZ');
    _syncEnvRotation(mat);
}
// three.js `envMap`, `envMapIntensity` and `envMapRotation` on Standard and
// Physical: the material's own image-based light, in place of
// `scene.environment`.
function _bindPbrEnvMap(mat, opts) {
    _bindMapSlot(mat, 'envMap', null, 'setEnvMap', opts.envMap);
    _bindScalar(mat, 'envMapIntensity', 'setEnvMapIntensity', 1, opts);
    const r = opts.envMapRotation;
    mat.envMapRotation = r?.isEuler ? r : new Euler(r?.x ?? 0, r?.y ?? 0, r?.z ?? 0, r?.order ?? 'XYZ');
    _syncEnvRotation(mat);
}
function _syncEnvRotation(mat) {
    const e = mat.envMapRotation;
    if (!e || !mat._w?.setEnvMapRotation) return false;
    const last = mat._envRotSynced;
    if (last && last[0] === e.x && last[1] === e.y && last[2] === e.z && last[3] === e.order) return false;
    mat._envRotSynced = [e.x, e.y, e.z, e.order];
    mat._w.setEnvMapRotation(e.x, e.y, e.z, e.order ?? 'XYZ');
    return true;
}
// three.js `lightMap` / `lightMapIntensity`, `bumpMap` / `bumpScale` and
// `specularMap`, live, on the materials that have each.
function _bindLightBumpSpecular(mat, opts, { light = true, bump = true, specular = false } = {}) {
    if (light) {
        _bindMapSlot(mat, 'lightMap', 'setLightMap', 'setLightMapData', opts.lightMap);
        _bindScalar(mat, 'lightMapIntensity', 'setLightMapIntensity', 1, opts);
    }
    if (bump) {
        _bindMapSlot(mat, 'bumpMap', 'setBumpMap', 'setBumpMapData', opts.bumpMap);
        _bindScalar(mat, 'bumpScale', 'setBumpScale', 1, opts);
    }
    if (specular) _bindMapSlot(mat, 'specularMap', 'setSpecularMap', 'setSpecularMapData', opts.specularMap);
}
// The maps three.js's Lambert, Phong and Toon materials take, live, with
// `aoMapIntensity`, `normalScale`, `displacementScale` / `displacementBias`.
function _applyLitMaps(mat, opts = {}) {
    _bindMapSlot(mat, 'map', 'setMap', 'setMapData', opts.map);
    _bindMapSlot(mat, 'emissiveMap', 'setEmissiveMap', 'setEmissiveMapData', opts.emissiveMap);
    _bindMapSlot(mat, 'normalMap', 'setNormalMap', 'setNormalMapData', opts.normalMap);
    _bindMapSlot(mat, 'aoMap', 'setAoMap', 'setAoMapData', opts.aoMap);
    _bindMapSlot(mat, 'displacementMap', 'setDisplacementMap', 'setDisplacementMapData', opts.displacementMap);
    _bindMapSlot(mat, 'alphaMap', 'setAlphaMap', 'setAlphaMapData', opts.alphaMap);
    _bindLightBumpSpecular(mat, opts, { specular: !(mat instanceof MeshToonMaterial) });
    if (!(mat instanceof MeshToonMaterial)) _bindEnvMap(mat, opts);
    _bindAoIntensity(mat, opts);
    const ns = opts.normalScale;
    mat.normalScale = ns?.isVector2 ? ns.clone() : new Vector2(ns?.x ?? 1, ns?.y ?? 1);
    if (mat.normalScale.x !== 1 || mat.normalScale.y !== 1) mat._w?.setNormalScale?.(mat.normalScale.x, mat.normalScale.y);
    mat.displacementScale = opts.displacementScale ?? 1;
    mat.displacementBias = opts.displacementBias ?? 0;
    mat._w?.setDisplacement?.(mat.displacementScale, mat.displacementBias);
}
function _applyStandardExtras(mat, opts) {
    if (!opts || typeof opts !== 'object') return;
    const w = mat._w;
    if (typeof opts.displacementScale === 'number' || typeof opts.displacementBias === 'number') {
        w.setDisplacement?.(opts.displacementScale ?? 1, opts.displacementBias ?? 0);
    }
    const cloudH = typeof opts.cloudHeight === 'number' ? opts.cloudHeight : 0.012;
    const cloudS = typeof opts.cloudShadow === 'number' ? opts.cloudShadow : 0.55;
    const twilight = typeof opts.twilight === 'number' ? opts.twilight : 0.1;
    if (opts.cloudShadowMap || 'twilight' in opts || 'cloudHeight' in opts) {
        w.setAtmosphereShading?.(cloudH, cloudS, 0, twilight);
    }
    if ('twilightColor' in opts && w.setTwilightColor) {
        w.setTwilightColor(_color(opts.twilightColor));
    }

    mat._cloudHeight = cloudH;
    mat._cloudShadow = cloudS;
    mat._cloudRotation = 0;
    mat._twilight = twilight;
    mat._dispScale = opts.displacementScale ?? 0;
    mat._dispBias = opts.displacementBias ?? 0;
    mat._eclipse = null;

    // Live setters so LOD / progressive upgrades (`mat.map = next`) rebind GPU.
    _bindMapSlot(mat, 'map', 'setMap', 'setMapData', opts.map);
    _bindMapSlot(mat, 'normalMap', 'setNormalMap', 'setNormalMapData', opts.normalMap);
    _bindMapSlot(mat, 'roughnessMap', 'setRoughnessMap', 'setRoughnessMapData', opts.roughnessMap);
    _bindMapSlot(mat, 'metalnessMap', 'setMetalnessMap', 'setMetalnessMapData', opts.metalnessMap);
    _bindMapSlot(mat, 'emissiveMap', 'setEmissiveMap', 'setEmissiveMapData', opts.emissiveMap);
    _bindMapSlot(mat, 'aoMap', 'setAoMap', 'setAoMapData', opts.aoMap);
    _bindMapSlot(mat, 'displacementMap', 'setDisplacementMap', 'setDisplacementMapData', opts.displacementMap);
    _bindMapSlot(mat, 'cloudShadowMap', null, 'setCloudShadowMapData', opts.cloudShadowMap);
    _bindMapSlot(mat, 'alphaMap', 'setAlphaMap', 'setAlphaMapData', opts.alphaMap);
    _bindLightBumpSpecular(mat, opts);
    _bindPbrEnvMap(mat, opts);
    _bindAoIntensity(mat, opts);

    Object.defineProperty(mat, 'emissiveIntensity', {
        get() { return mat._emissiveIntensity ?? 1; },
        set(v) {
            mat._emissiveIntensity = Number(v);
            mat._w?.setEmissiveIntensity?.(mat._emissiveIntensity);
        },
        configurable: true, enumerable: true,
    });
    if (typeof opts.emissiveIntensity === 'number') mat.emissiveIntensity = opts.emissiveIntensity;

    Object.defineProperty(mat, 'emissiveNightSide', {
        get() { return mat._emissiveNightSide ?? 0; },
        set(v) {
            mat._emissiveNightSide = Number(v);
            mat._w?.setEmissiveNightSide?.(mat._emissiveNightSide);
        },
        configurable: true, enumerable: true,
    });
    if (typeof opts.emissiveNightSide === 'number') mat.emissiveNightSide = opts.emissiveNightSide;

    Object.defineProperty(mat, 'displacementScale', {
        get() { return mat._dispScale; },
        set(v) {
            mat._dispScale = Number(v);
            mat._w?.setDisplacement?.(mat._dispScale, mat._dispBias);
        },
        configurable: true, enumerable: true,
    });
    Object.defineProperty(mat, 'displacementBias', {
        get() { return mat._dispBias; },
        set(v) {
            mat._dispBias = Number(v);
            mat._w?.setDisplacement?.(mat._dispScale, mat._dispBias);
        },
        configurable: true, enumerable: true,
    });
    Object.defineProperty(mat, 'cloudRotation', {
        get() { return mat._cloudRotation; },
        set(v) {
            mat._cloudRotation = Number(v);
            mat._w?.setAtmosphereShading?.(mat._cloudHeight, mat._cloudShadow, mat._cloudRotation, mat._twilight);
        },
        configurable: true, enumerable: true,
    });
    // Earth demo: `mat.eclipseOccluder = [x,y,z,r]` — sun angular radius ~0.00465 rad.
    Object.defineProperty(mat, 'eclipseOccluder', {
        get() { return mat._eclipse; },
        set(v) {
            mat._eclipse = v;
            if (!mat._w?.setEclipse) return;
            if (!v || !v[3]) mat._w.setEclipse(0, 0, 0, 0, 0.00465);
            else mat._w.setEclipse(v[0], v[1], v[2], v[3], 0.00465);
        },
        configurable: true, enumerable: true,
    });
}

export class MeshStandardMaterial {
    constructor(opts = {}) {
        const c = _matColor(opts);
        const r = (typeof opts === 'object' && opts && 'roughness' in opts) ? opts.roughness : 1.0;
        const m = (typeof opts === 'object' && opts && 'metalness' in opts) ? opts.metalness : 0.0;
        this._w = WebMaterial.standard(c, r, m);
        _applyCommon(this._w, opts); _initMaterialBase(this, opts);
        _applyStandardExtras(this, opts);
        _pbrScalars(this, r, m);
    }
}
export class MeshPhongMaterial { constructor(opts = {}) { this._w = WebMaterial.phong(_matColor(opts)); _applyCommon(this._w, opts); _applyLitMaps(this, opts); _initMaterialBase(this, opts); } }
export class MeshPhysicalMaterial {
    constructor(opts = {}) {
        const o = (typeof opts === 'object' && opts) ? opts : {};
        const r  = 'roughness' in o ? o.roughness : 1.0;
        const m  = 'metalness' in o ? o.metalness : 0.0;
        const cc = 'clearcoat' in o ? o.clearcoat : 0.0;
        const cr = 'clearcoatRoughness' in o ? o.clearcoatRoughness : 0.0;
        this._w = WebMaterial.physical(_matColor(opts), r, m, cc, cr);
        _applyCommon(this._w, opts); _initMaterialBase(this, opts);
        _applyStandardExtras(this, opts);
        _pbrScalars(this, r, m);
        _physicalProps(this, o);
    }
}
// three.js MeshPhysicalMaterial's own properties, live, with three.js's
// defaults. The scalars travel together (`setPhysical`); `sheenColor`,
// `specularColor` and `attenuationColor` are Colors mutated in place, so they
// join the per-render colour sync (`_syncMaterialColors`). `reflectivity` is
// three.js's other view of `ior`.
function _physicalProps(mat, o) {
    const v = {
        clearcoat: 0, clearcoatRoughness: 0, ior: 1.5, specularIntensity: 1,
        sheen: 0, sheenRoughness: 1, iridescence: 0, iridescenceIOR: 1.3,
        iridescenceThicknessRange: [100, 400], anisotropy: 0, anisotropyRotation: 0,
        transmission: 0, thickness: 0, attenuationDistance: Infinity, dispersion: 0,
    };
    const push = () => {
        const [tmin, tmax] = v.iridescenceThicknessRange;
        mat._w?.setPhysical?.(v.clearcoat, v.clearcoatRoughness, v.ior, v.specularIntensity,
            v.sheen, v.sheenRoughness, v.iridescence, v.iridescenceIOR, tmin, tmax,
            v.anisotropy, v.anisotropyRotation, v.transmission, v.thickness,
            v.attenuationDistance, v.dispersion);
        _materialChanged(mat);
    };
    for (const key of Object.keys(v)) {
        if (key in o) v[key] = key === 'iridescenceThicknessRange' ? [...o[key]] : Number(o[key]);
        Object.defineProperty(mat, key, {
            get() { return v[key]; },
            set(x) { v[key] = key === 'iridescenceThicknessRange' ? [...x] : Number(x); push(); },
            configurable: true, enumerable: true,
        });
    }
    Object.defineProperty(mat, 'reflectivity', {
        get() { return Math.min(Math.max(2.5 * (v.ior - 1) / (v.ior + 1), 0), 1); },
        set(r) { mat.ior = (1 + 0.4 * r) / (1 - 0.4 * r); },
        configurable: true, enumerable: true,
    });
    if ('reflectivity' in o && !('ior' in o)) {
        const r = Number(o.reflectivity);
        v.ior = (1 + 0.4 * r) / (1 - 0.4 * r);
    }
    push();
    // MeshPhysicalMaterial's own maps, live, through the named-slot setters.
    for (const key of ['specularIntensityMap', 'specularColorMap', 'clearcoatMap', 'clearcoatRoughnessMap',
        'clearcoatNormalMap', 'iridescenceMap', 'iridescenceThicknessMap', 'sheenColorMap', 'sheenRoughnessMap',
        'transmissionMap', 'thicknessMap', 'anisotropyMap']) {
        _bindMapSlot(mat, key, (w, t) => w.setNamedMap?.(key, t), (w, t) => w.setNamedMapData?.(key, t), o[key]);
    }
    // `clearcoatNormalScale`, a Vector2 edited in place: pushed whenever it
    // has changed, before each render (`_syncMaterialColors`).
    mat.clearcoatNormalScale = o.clearcoatNormalScale?.isVector2 ? o.clearcoatNormalScale.clone()
        : new Vector2(o.clearcoatNormalScale?.x ?? 1, o.clearcoatNormalScale?.y ?? 1);
    (mat._wVec2 ??= []).push({ key: 'clearcoatNormalScale', setter: 'setClearcoatNormalScale', x: NaN, y: NaN });
    const colours = [['sheenColor', 'setSheenColor', 0x000000], ['specularColor', 'setSpecularColor', 0xffffff],
        ['attenuationColor', 'setAttenuationColor', 0xffffff]];
    for (const [key, setter, def] of colours) {
        mat[key] = new Color(key in o && o[key] != null ? o[key] : def);
        const c = mat[key];
        mat._w?.[setter]?.(new WebColor(c.r, c.g, c.b));
        mat._wColors.push({ key, setter, obj: c, r: c.r, g: c.g, b: c.b });
    }
    // threers's glass modes (not three.js): how a transmissive surface is
    // drawn — 'blend' (three.js's transmission, the default), 'refract'
    // (screen-space refraction), 'glass' (one blended layer) or 'oit'
    // (order-independent; 'glass' where the device cannot). See
    // `docs/materials.md`.
    const modes = { blend: 0, glass: 1, oit: 2, refract: 3 };
    let mode = o.transparencyMode in modes ? o.transparencyMode : 'blend';
    Object.defineProperty(mat, 'transparencyMode', {
        get() { return mode; },
        set(m) {
            if (!(m in modes)) {
                console.warn(`threers: transparencyMode '${m}' is not one of ${Object.keys(modes).join(', ')}`);
                return;
            }
            mode = m;
            mat._w?.setTransparencyMode?.(modes[m]);
            _materialChanged(mat);
        },
        configurable: true, enumerable: true,
    });
    if (mode !== 'blend') mat._w?.setTransparencyMode?.(modes[mode]);
}
export class MeshNormalMaterial { constructor(opts = {}) { this._w = WebMaterial.normalMat(); _initMaterialBase(this, opts); } }
export class MeshDepthMaterial { constructor(opts = {}) { this._w = WebMaterial.depth(); _initMaterialBase(this, opts); } }
export class MeshToonMaterial {
    constructor(opts = {}) {
        this._w = WebMaterial.toon(_matColor(opts));
        _applyCommon(this._w, opts);
        _applyLitMaps(this, opts);
        // three.js `gradientMap`: the ramp each light is read through
        // (`getGradientIrradiance`); null is the two-tone default.
        _bindMapSlot(this, 'gradientMap', (w, t) => w.setNamedMap?.('gradientMap', t),
            (w, t) => w.setNamedMapData?.('gradientMap', t), opts.gradientMap);
        _initMaterialBase(this, opts);
        this.isMeshToonMaterial = true;
    }
}
export class LineBasicMaterial {
    constructor(opts = {}) {
        this._w = WebMaterial.line(_matColor(opts));
        // three.js `toneMapped` (every helper passes `false`).
        if (opts?.toneMapped === false) this._w.setToneMapped?.(false);
        // The Material fields three.js has on every material (color,
        // opacity, transparent, depthTest, …).
        _initMaterialBase(this, opts);
    }
}
export class PointsMaterial {
    constructor(opts = {}) {
        opts = (typeof opts === 'object' && opts) ? opts : {};
        this._w = WebMaterial.points(_matColor(opts), opts.size ?? 1);
        this.isPointsMaterial = true;
        _bindScalar(this, 'size', 'setPointSize', 1, opts);
        // three.js's default is `true`; the wasm material's too.
        let attenuate = opts.sizeAttenuation !== false;
        Object.defineProperty(this, 'sizeAttenuation', {
            get() { return attenuate; },
            set(v) { attenuate = v !== false; this._w.setSizeAttenuation(attenuate); _materialChanged(this); },
            configurable: true, enumerable: true,
        });
        if (!attenuate) this._w.setSizeAttenuation(false);
        _bindMapSlot(this, 'map', 'setMap', 'setMapData', opts.map);
        _bindMapSlot(this, 'alphaMap', 'setAlphaMap', 'setAlphaMapData', opts.alphaMap);
        _initMaterialBase(this, opts);
    }
}
export class SpriteMaterial {
    constructor(opts = {}) {
        this._w = WebMaterial.sprite(_matColor(opts));
        _bindMapSlot(this, 'map', 'setMap', 'setMapData', opts.map);
        _bindMapSlot(this, 'alphaMap', 'setAlphaMap', 'setAlphaMapData', opts.alphaMap);
        _applyCommon(this._w, opts); _initMaterialBase(this, opts);
        this.isSpriteMaterial = true;
        this._rotation = 0;
        this._sizeAttenuation = true;
        this.rotation = opts.rotation ?? 0;
        this.sizeAttenuation = opts.sizeAttenuation ?? true;
    }
    /// three.js `rotation`: the quad's turn in the view plane (drawn and hit).
    get rotation() { return this._rotation; }
    set rotation(v) { this._rotation = Number(v) || 0; this._w.setSpriteRotation?.(this._rotation); _materialChanged(this); }
    /// three.js `sizeAttenuation`: false keeps a constant on-screen size.
    get sizeAttenuation() { return this._sizeAttenuation; }
    set sizeAttenuation(v) { this._sizeAttenuation = v !== false; this._w.setSizeAttenuation?.(this._sizeAttenuation); _materialChanged(this); }
}

/** Analytic planetary atmosphere shell — see `WebMaterial.atmosphere`. */
export class AtmosphereMaterial {
    constructor(opts = {}) {
        const o = (typeof opts === 'object' && opts) ? opts : {};
        const planetR = typeof o.planetRadius === 'number' ? o.planetRadius : 1;
        const atmoR = typeof o.atmosphereRadius === 'number' ? o.atmosphereRadius : planetR * 1.025;
        this._w = WebMaterial.atmosphere(planetR, atmoR);
        this.planetRadius = planetR;
        this.atmosphereRadius = atmoR;
        this._sunset = new Color(o.sunsetColor ?? 0xff7a33);
        this._intensity = typeof o.intensity === 'number' ? o.intensity : 1.5;
        this._falloff = typeof o.falloff === 'number' ? o.falloff : 3.0;
        this._opacity = typeof o.opacity === 'number' ? o.opacity : 1.0;
        this._airglow = typeof o.airglow === 'number' ? o.airglow : 0;
        this._airglowColor = new Color(o.airglowColor ?? 0x4dff8a);
        this._syncAtmosphere();
        this._syncAirglow();
        _initMaterialBase(this, o);
        // setColor does not yet reach Atmosphere in wasm; keep a local tint anyway.
        if ('color' in o) this.color = new Color(o.color);
        const self = this;
        // After _initMaterialBase (it writes plain `opacity`), bind live setters.
        Object.defineProperty(this, 'intensity', {
            get() { return self._intensity; },
            set(v) { self._intensity = Number(v); self._syncAtmosphere(); },
            configurable: true,
            enumerable: true,
        });
        Object.defineProperty(this, 'falloff', {
            get() { return self._falloff; },
            set(v) { self._falloff = Number(v); self._syncAtmosphere(); },
            configurable: true,
            enumerable: true,
        });
        Object.defineProperty(this, 'opacity', {
            get() { return self._opacity; },
            set(v) { self._opacity = Number(v); self._syncAtmosphere(); },
            configurable: true,
            enumerable: true,
        });
        Object.defineProperty(this, 'airglow', {
            get() { return self._airglow; },
            set(v) { self._airglow = Number(v); self._syncAirglow(); },
            configurable: true,
            enumerable: true,
        });
    }
    _syncAtmosphere() {
        this._w.setAtmosphere(_color(this._sunset), this._intensity, this._falloff, this._opacity);
    }
    _syncAirglow() {
        this._w.setAirglow(this._airglow, _color(this._airglowColor));
    }
}

// ---- Scene graph: Object3D and its wasm mirror ----
//
// Every Object3D has one node in a wasm scene graph (a `WebScene`): the graph
// of the Scene it is in, or — for an object outside any Scene — a hidden graph
// it is mirrored into when something needs it there (a raycast, a control).
// The node has the same parent, the same children in the same order, and the
// object's own state: position / quaternion / scale (or `matrix`, with
// `matrixAutoUpdate` off), visibility, layers, render order and shadow flags,
// geometry (groups, draw range, morph attributes, BVH), material or material
// array, morph influences, instance data, skeleton and bind matrices, light
// parameters. The renderer, the raycaster and the controls read only that
// graph, so it is the one scene graph; JS keeps three.js's objects in front of
// it.
//
// Changes reach it through dirty flags. The math types an object owns notify
// it (`position` and `scale` are observed vectors, `rotation` and
// `quaternion` call back through three.js's `_onChange`), and so do the
// accessors three.js leaves plain — `visible`, `layers.mask`, `renderOrder`,
// `castShadow`, `geometry`, `material`, `matrixAutoUpdate`, `children` (add,
// remove, attach) — and a geometry, attribute, material or instance buffer
// tells the objects using it. A dirty object goes on one queue; a sync (before
// a render, a raycast, `scene.updateMatrixWorld()`, or `object._syncNow()`)
// pushes only what is queued, so its wasm calls are O(changed). What three.js
// lets code change with no setter at all — a matrix written with
// `matrixAutoUpdate` off, `morphTargetInfluences`, a light's colour, a
// sprite's `center` — is compared at each sync, for the few objects that have
// such state.

const _D_TRS = 1, _D_VISIBLE = 2, _D_LAYERS = 4, _D_DRAW = 8;
const _D_GEOMETRY = 16, _D_MATERIAL = 32, _D_DATA = 64, _D_CHILDREN = 128;
/// `name` and `uuid`, which animation tracks find nodes by.
const _D_NAME = 256;
const _D_ALL = 511;
/// Objects that have a node and changes not yet pushed to it.
const _mirrorQueue = new Set();
let _object3DId = 1;

/// One wasm scene graph and the JS objects mirrored into it.
class _SceneGraph {
    constructor(w, scene) {
        this.w = w;
        /// The Scene this graph mirrors; null for the hidden graph.
        this.scene = scene;
        this.root = w.rootHandle();
        this.rootId = this.root.id;
        /// node id → WeakRef(object), for hits and control events.
        this.objects = new Map();
        /// Objects whose unobservable state is compared at each sync.
        this.polled = new Set();
        /// Hidden graph: ids of the detached roots mirrored under its root.
        this.detachedIds = new Set();
    }
    /// The object a node id stands for here, or null.
    objectOf(id) {
        const o = this.objects.get(id)?.deref();
        return o?._m?.graph === this && o._m.id === id ? o : null;
    }
}
let _hiddenGraphInstance = null;
/// The graph objects outside any Scene are mirrored into on demand.
function _hiddenGraph() {
    return _hiddenGraphInstance ??= new _SceneGraph(new WebScene(), null);
}
/// A node goes when its object is collected.
const _nodeFinalizer = typeof FinalizationRegistry === 'function'
    ? new FinalizationRegistry(({ graph, id }) => {
        graph.objects.delete(id);
        graph.detachedIds.delete(id);
        const h = graph.w.handleOf(id);
        if (h) graph.w.deleteNode(h);
    })
    : null;

function _mirrorRoot(obj) {
    let o = obj;
    while (o.parent) o = o.parent;
    return o;
}
function _graphOfRoot(root) {
    return root.isScene === true && root._graph ? root._graph : null;
}
/// The object's node in `graph`, of the kind it needs now — created (or,
/// from another graph, moved there: the node is the object's, whichever
/// graph has it) with its whole state and its subtree when it is not.
function _mirrorEnsure(obj, graph) {
    if (obj._graph === graph) return graph.root;
    const kind = obj._mirrorKind();
    const m = obj._m;
    if (m && m.graph === graph && m.kind === kind) return m.handle;
    const moved = m && m.kind === kind ? _mirrorMove(obj, graph) : null;
    if (!moved && obj._m) _mirrorDrop(obj);
    const handle = moved ?? obj._mirrorCreate(graph.w);
    const rec = { graph, handle, id: handle.id, kind, parentId: undefined, childIds: null };
    obj._m = rec;
    graph.objects.set(rec.id, obj._wref ??= new WeakRef(obj));
    _nodeFinalizer?.register(obj, { graph, id: rec.id }, rec);
    _mirrorQueue.delete(obj);
    _mirrorPush(obj, _D_ALL | obj._dirtyFlags);
    return handle;
}
/// Move the object's node out of the graph it is in into `graph` (its
/// children follow as they are linked there). Null when it cannot move.
function _mirrorMove(obj, graph) {
    const m = obj._m;
    if (obj.isScene || typeof graph.w.adoptNode !== 'function' || !m.graph.w.hasNode(m.handle)) return null;
    _nodeFinalizer?.unregister(m);
    m.graph.objects.delete(m.id);
    m.graph.polled.delete(obj);
    m.graph.detachedIds.delete(m.id);
    obj._m = null;
    return graph.w.adoptNode(m.graph.w, m.handle) ?? null;
}
/// Forget the object's node (and delete it from its graph).
function _mirrorDrop(obj) {
    const m = obj._m;
    if (!m || obj.isScene) return;
    _nodeFinalizer?.unregister(m);
    m.graph.objects.delete(m.id);
    m.graph.polled.delete(obj);
    m.graph.detachedIds.delete(m.id);
    if (m.graph.w.hasNode(m.handle)) m.graph.w.deleteNode(m.handle);
    obj._m = null;
}
/// Push the state `flags` name to the object's node.
function _mirrorPush(obj, flags) {
    const m = obj._m;
    const w = m.graph.w, h = m.handle;
    obj._dirtyFlags &= ~flags;
    if (flags & _D_TRS) obj._mirrorTransform(w, h, m);
    // three.js draws nothing for a single material with `visible: false`
    // (TransformControls' drag plane); casts still see it.
    if (flags & _D_VISIBLE) w.setVisible(h, obj.visible !== false && obj.material?.visible !== false);
    if (flags & _D_NAME) w.setNameUuid(h, String(obj.name ?? ''), String(obj.uuid ?? ''));
    if (flags & _D_LAYERS) w.setLayersMask(h, (obj.layers?.mask ?? 1) >>> 0);
    if (flags & _D_DRAW) {
        const order = Number(obj.renderOrder) || 0;
        w.setRenderOrder(h, Number.isFinite(order) ? order | 0 : (order < 0 ? -2147483648 : 2147483647));
        w.setObjectCastShadow(h, obj.castShadow === true);
        w.setObjectReceiveShadow(h, obj.receiveShadow === true);
    }
    if (flags & (_D_GEOMETRY | _D_MATERIAL | _D_DATA)) obj._mirrorContent(w, h, flags, m);
    if (flags & _D_CHILDREN) _mirrorChildren(obj, m.graph);
    if (obj._mirrorPolled()) m.graph.polled.add(obj);
    else m.graph.polled.delete(obj);
}
/// The node's children: the object's, in order (and any internal ones, such
/// as a BatchedMesh's slots), each mirrored first.
function _mirrorChildren(obj, graph) {
    const m = obj._m;
    const ids = [];
    const take = (c) => {
        if (!(c instanceof Object3D) || c.isScene) return;
        _mirrorEnsure(c, graph);
        const cm = c._m;
        if (cm.parentId !== m.id) {
            // Linked here now: catch up on what changed while it was elsewhere.
            cm.parentId = m.id;
            _mirrorCatchUp(c, graph);
        }
        ids.push(cm.id);
    };
    for (const c of obj.children) take(c);
    const extra = obj._mirrorExtraChildren?.();
    if (extra) for (const c of extra) take(c);
    if (m.childIds) {
        for (const id of m.childIds) {
            if (ids.includes(id)) continue;
            const gone = graph.objectOf(id);
            if (gone?._m?.parentId === m.id) gone._m.parentId = undefined;
        }
    }
    m.childIds = ids;
    graph.w.setChildren(m.handle, new Float64Array(ids));
}
/// Push whatever is pending anywhere in `obj`'s subtree (mirrored in `graph`).
function _mirrorCatchUp(obj, graph) {
    const visit = (o) => {
        if (!(o instanceof Object3D) || o._m?.graph !== graph) return;
        if (o._dirtyFlags) {
            _mirrorQueue.delete(o);
            _mirrorSyncObject(o, graph);
        } else if (o._mirrorPolled()) {
            graph.polled.add(o);
        }
        for (const c of o.children) visit(c);
        const extra = o._mirrorExtraChildren?.();
        if (extra) for (const c of extra) visit(c);
    };
    visit(obj);
}
/// Bring one queued object's node up to date in `graph`.
function _mirrorSyncObject(obj, graph) {
    _mirrorQueue.delete(obj);
    if (obj.isScene) {
        if (obj._dirtyFlags) _mirrorPush(obj, obj._dirtyFlags);
        return;
    }
    const m = obj._m;
    if (!m || m.graph !== graph || m.kind !== obj._mirrorKind()) {
        const parent = obj.parent;
        const parentHere = parent && (parent._graph === graph || parent._m?.graph === graph);
        _mirrorEnsure(obj, graph);
        // A new node: re-link it under its parent.
        if (parentHere) _mirrorChildren(parent, graph);
        return;
    }
    if (obj._dirtyFlags) _mirrorPush(obj, obj._dirtyFlags);
}
/// Objects whose per-sync work runs before the queue (helpers recomputing
/// their lines), then the queue, then the compared state.
function _mirrorPreSync(graph) {
    for (const obj of graph.polled) obj._mirrorBeforeSync?.();
}
function _mirrorPoll(graph) {
    for (const obj of graph.polled) {
        const m = obj._m;
        if (m?.graph !== graph) { graph.polled.delete(obj); continue; }
        // Only what is in the graph's tree (a removed object stops being compared
        // until it is linked again).
        const root = _mirrorRoot(obj);
        const inTree = graph.scene ? root === graph.scene : graph.detachedIds.has(root._m?.id);
        if (!inTree) { graph.polled.delete(obj); continue; }
        obj._mirrorPoll(graph.w, m.handle, m);
        if (obj._dirtyFlags) _mirrorSyncObject(obj, graph);
    }
}
/// Bring a Scene's graph up to date: everything queued for it, then the
/// compared state. Queued objects outside any Scene are left for when they
/// are used (they catch up when linked or cast).
function _mirrorSyncScene(scene) {
    const graph = scene._graph;
    _mirrorPreSync(graph);
    if (_mirrorQueue.size) {
        for (const obj of Array.from(_mirrorQueue)) {
            if (!_mirrorQueue.has(obj)) continue;
            const g = _graphOfRoot(_mirrorRoot(obj));
            if (g === graph) _mirrorSyncObject(obj, graph);
            else if (g === null) _mirrorQueue.delete(obj);
        }
    }
    _mirrorPoll(graph);
}
/// Mirror a root that is in no Scene into the hidden graph, up to date.
function _mirrorSyncDetached(root) {
    const graph = _hiddenGraph();
    _mirrorEnsure(root, graph);
    const m = root._m;
    if (!graph.detachedIds.has(m.id) || m.parentId !== graph.rootId) {
        graph.detachedIds.add(m.id);
        m.parentId = graph.rootId;
        const ids = [];
        for (const id of graph.detachedIds) {
            const o = graph.objectOf(id);
            if (o && o.parent === null) ids.push(id);
            else graph.detachedIds.delete(id);
        }
        graph.w.setChildren(graph.root, new Float64Array(ids));
    }
    _mirrorPreSync(graph);
    _mirrorCatchUp(root, graph);
    _mirrorPoll(graph);
}
/// Bring the graph `obj` is in (its Scene's, or the hidden one) up to date;
/// returns its node record.
function _mirrorSync(obj) {
    const root = _mirrorRoot(obj);
    if (root.isScene === true && root._graph) _mirrorSyncScene(root);
    else _mirrorSyncDetached(root);
    return obj._m;
}
/// What a scene's wasm fog was made from (null: no fog).
function _sceneFogState(f) {
    if (!f) return null;
    const c = f.color;
    return [c?.r ?? 1, c?.g ?? 1, c?.b ?? 1, f.near ?? 1, f.far ?? 1000, f.density ?? 0, f.isFogExp2 ? 1 : 0];
}
function _sameNumbers(a, b, n = b.length) {
    if (!a || a.length !== n) return false;
    for (let i = 0; i < n; i++) if (a[i] !== b[i]) return false;
    return true;
}

/// Users of a geometry, material or skeleton (WeakRefs to objects), told
/// when it changes.
function _useBy(thing, obj) {
    if (!thing || typeof thing !== 'object') return;
    const users = thing._users ??= new Set();
    users.add(obj._wref ??= new WeakRef(obj));
}
function _notifyUsers(thing, flag, still) {
    const users = thing?._users;
    if (!users) return;
    for (const ref of users) {
        const o = ref.deref();
        if (!o || !still(o)) { users.delete(ref); continue; }
        o._dirty(flag);
    }
}
/// A material's wasm state changed (a setter, `needsUpdate`, a rebuilt
/// uniform block): the objects drawn with it pick it up at the next sync.
function _materialChanged(material) {
    _notifyUsers(material, _D_MATERIAL, (o) => o.material === material
        || (Array.isArray(o.material) && o.material.includes(material)));
}
/// Push a material's `color` / `emissive` to its wasm material when user code
/// changed them (in place, or by assigning another Color) since the last push.
/// Called for every drawn material before each render.
function _syncMaterialColors(material) {
    if (!material?._wColors && !material?._wVec2) return;
    let changed = false;
    for (const slot of material._wVec2 ?? []) {
        const v = material[slot.key];
        if (!v || (v.x === slot.x && v.y === slot.y)) continue;
        slot.x = v.x; slot.y = v.y;
        material._w?.[slot.setter]?.(v.x, v.y);
        changed = true;
    }
    for (const slot of material._wColors ?? []) {
        const c = material[slot.key];
        if (!c?.isColor) continue;
        if (c === slot.obj && c.r === slot.r && c.g === slot.g && c.b === slot.b) continue;
        slot.obj = c; slot.r = c.r; slot.g = c.g; slot.b = c.b;
        material._w?.[slot.setter]?.(new WebColor(c.r, c.g, c.b));
        changed = true;
    }
    if (changed) _materialChanged(material);
}

/// The wasm geometry for a JS one: a BufferGeometry's current build, a
/// built-in's (or a loader's) `WebGeometry`; null when there is none to draw.
function _wasmGeometryOf(g) {
    if (!g) return null;
    if (typeof g._wasmGeometry === 'function') return g._wasmGeometry();
    if (g instanceof WebGeometry) return g;
    if (g._w instanceof WebGeometry) return g._w;
    if (g._w instanceof WebBufferGeometry) return g._wShared ??= WebGeometry.shareBufferGeometry(g._w);
    return null;
}
/// What changing a geometry's wasm build is keyed on.
function _geometryKey(g) {
    return g && typeof g._wasmGeometry === 'function' ? `${g._version}:${g._rangeVersion}` : g;
}
let _fallbackMaterialW = null;
/// The wasm material for a JS one (a plain white basic material for none).
function _wasmMaterialOf(material) {
    const w = material?._w;
    if (w instanceof WebMaterial) return w;
    return _fallbackMaterialW ??= WebMaterial.basic(WebColor.fromHex(0xffffff));
}
/// Push a single material or a material array (three.js `mesh.material`).
function _mirrorMaterials(w, h, material, obj, single = obj._wasmMaterialOverride?.()) {
    w.clearNodeMaterials(h);
    if (Array.isArray(material)) {
        if (material.length) w.setNodeMaterial(h, _wasmMaterialOf(material[0]));
        for (const mat of material) {
            w.pushNodeMaterial(h, _wasmMaterialOf(mat));
            _useBy(mat, obj);
        }
    } else {
        w.setNodeMaterial(h, single ?? _wasmMaterialOf(material));
        _useBy(material, obj);
    }
}

/// Layers with an owner: a changed mask reaches the owner's node.
export class Layers {
    constructor() { this._mask = 1 | 0; this._owner = null; }
    get mask() { return this._mask; }
    set mask(v) {
        v |= 0;
        if (v === this._mask) return;
        this._mask = v;
        this._owner?._dirty(_D_LAYERS);
    }
    set(channel) { this.mask = (1 << channel) | 0; }
    enable(channel) { this.mask = this._mask | (1 << channel) | 0; }
    enableAll() { this.mask = 0xffffffff | 0; }
    toggle(channel) { this.mask = this._mask ^ (1 << channel) | 0; }
    disable(channel) { this.mask = this._mask & ~(1 << channel); }
    disableAll() { this.mask = 0; }
    test(layers) { return (this._mask & layers.mask) !== 0; }
    isEnabled(channel) { return (this._mask & (1 << channel) | 0) !== 0; }
}

/// A Vector3 that tells its owner when it changes (an object's `position`
/// and `scale`).
class _ObservedVector3 extends Vector3 {
    constructor(x, y, z, onChange) {
        super(x, y, z);
        this._onChange = onChange;
    }
    get x() { return this._x; }
    set x(v) { this._x = v; this._onChange?.(); }
    get y() { return this._y; }
    set y(v) { this._y = v; this._onChange?.(); }
    get z() { return this._z; }
    set z(v) { this._z = v; this._onChange?.(); }
    set(x, y, z) {
        if (z === undefined) z = this._z;
        this._x = x; this._y = y; this._z = z;
        this._onChange?.();
        return this;
    }
    copy(v) {
        this._x = v.x; this._y = v.y; this._z = v.z;
        this._onChange?.();
        return this;
    }
    toJSON() { return { x: this._x, y: this._y, z: this._z }; }
}

const _o3dQ = new Quaternion();
const _o3dV = new Vector3(), _o3dTarget = new Vector3(), _o3dPosition = new Vector3(), _o3dScale = new Vector3();
const _o3dM = new Matrix4();
const _o3dQuat = new Quaternion();
const _xAxis = new Vector3(1, 0, 0), _yAxis = new Vector3(0, 1, 0), _zAxis = new Vector3(0, 0, 1);

/// three.js r165 `Object3D`, mirrored (see above).
export class Object3D {
    constructor() {
        this.isObject3D = true;
        this.id = _object3DId++;
        this.uuid = MathUtils.generateUUID();
        this.name = '';
        this.type = 'Object3D';
        this.parent = null;
        this.children = [];
        this.up = Object3D.DEFAULT_UP.clone();
        this._m = null;
        this._dirtyFlags = 0;
        const changed = () => this._dirty(_D_TRS);
        const rotation = new Euler();
        const quaternion = new Quaternion();
        rotation._onChange(() => { quaternion.setFromEuler(rotation, false); changed(); });
        quaternion._onChange(() => { rotation.setFromQuaternion(quaternion, undefined, false); changed(); });
        this._position = new _ObservedVector3(0, 0, 0, changed);
        this._rotation = rotation;
        this._quaternion = quaternion;
        this._scale = new _ObservedVector3(1, 1, 1, changed);
        this.modelViewMatrix = new Matrix4();
        this.normalMatrix = new Matrix3();
        this.matrix = new Matrix4();
        this.matrixWorld = new Matrix4();
        this._matrixAutoUpdate = Object3D.DEFAULT_MATRIX_AUTO_UPDATE;
        this.matrixWorldAutoUpdate = Object3D.DEFAULT_MATRIX_WORLD_AUTO_UPDATE;
        this.matrixWorldNeedsUpdate = false;
        this._layers = new Layers();
        this._layers._owner = this;
        this._visible = true;
        this._castShadow = false;
        this._receiveShadow = false;
        this.frustumCulled = true;
        this._renderOrder = 0;
        this.animations = [];
        this.userData = {};
    }

    // three.js makes these four non-writable; assigning copies instead.
    get position() { return this._position; }
    set position(v) { this._position.copy(v); }
    get rotation() { return this._rotation; }
    set rotation(v) { this._rotation.copy(v); }
    get quaternion() { return this._quaternion; }
    set quaternion(v) { this._quaternion.copy(v); }
    get scale() { return this._scale; }
    set scale(v) { this._scale.copy(v); }
    get layers() { return this._layers; }
    set layers(v) { this._layers.mask = v?.mask ?? 1; }
    get name() { return this._name; }
    set name(v) { if (v !== this._name) { this._name = v; this._dirty(_D_NAME); } }
    get uuid() { return this._uuid; }
    set uuid(v) { if (v !== this._uuid) { this._uuid = v; this._dirty(_D_NAME); } }
    get visible() { return this._visible; }
    set visible(v) { v = !!v; if (v !== this._visible) { this._visible = v; this._dirty(_D_VISIBLE); } }
    get castShadow() { return this._castShadow; }
    set castShadow(v) { v = !!v; if (v !== this._castShadow) { this._castShadow = v; this._dirty(_D_DRAW); } }
    get receiveShadow() { return this._receiveShadow; }
    set receiveShadow(v) { v = !!v; if (v !== this._receiveShadow) { this._receiveShadow = v; this._dirty(_D_DRAW); } }
    get renderOrder() { return this._renderOrder; }
    set renderOrder(v) { if (v !== this._renderOrder) { this._renderOrder = v; this._dirty(_D_DRAW); } }
    get matrixAutoUpdate() { return this._matrixAutoUpdate; }
    set matrixAutoUpdate(v) { v = !!v; if (v !== this._matrixAutoUpdate) { this._matrixAutoUpdate = v; this._dirty(_D_TRS); } }
    /// The object's node handle in its wasm graph — mirrored (with its graph
    /// brought up to date) when asked for, so code that addresses nodes by id
    /// (post-processing selections) sees every object it is given.
    get _handle() {
        if (!this._m || this._dirtyFlags) _mirrorSync(this);
        return this._m?.handle ?? null;
    }
    set _handle(_v) { /* the mirror owns it */ }

    /// Mark state for the next sync (see the section notes).
    _dirty(flags) {
        this._dirtyFlags |= flags;
        if (this._m) _mirrorQueue.add(this);
    }
    /// Push this object's graph (its Scene's, or the hidden one) to wasm now.
    _syncNow() { _mirrorSync(this); return this; }

    // ---- mirror hooks, overridden per kind ----
    /// What node this object needs; a change re-creates it.
    _mirrorKind() { return this.isGroup === true ? 'group:g' : 'group'; }
    _mirrorCreate(w) { return w.createGroup(this.isGroup === true); }
    _mirrorTransform(w, h, m) {
        if (this._matrixAutoUpdate) {
            const p = this._position, q = this._quaternion, s = this._scale;
            w.setTRS(h, p.x, p.y, p.z, q._x, q._y, q._z, q._w, s.x, s.y, s.z);
            m.matrix = null;
        } else {
            const e = this.matrix.elements;
            w.setLocalMatrix(h, Float32Array.from(e));
            m.matrix = Float64Array.from(e);
        }
    }
    _mirrorContent(_w, _h, _flags, _m) {}
    /// Whether the object has state compared at each sync.
    _mirrorPolled() { return !this._matrixAutoUpdate; }
    _mirrorPoll(_w, _h, m) {
        if (!this._matrixAutoUpdate && !_sameNumbers(m.matrix, this.matrix.elements, 16)) this._dirty(_D_TRS);
    }

    onBeforeShadow() {}
    onAfterShadow() {}
    onBeforeRender() {}
    onAfterRender() {}

    applyMatrix4(matrix) {
        if (this.matrixAutoUpdate) this.updateMatrix();
        this.matrix.premultiply(matrix);
        this.matrix.decompose(this.position, this.quaternion, this.scale);
        if (!this.matrixAutoUpdate) this._dirty(_D_TRS);
        return this;
    }
    applyQuaternion(q) { this.quaternion.premultiply(q); return this; }
    setRotationFromAxisAngle(axis, angle) { this.quaternion.setFromAxisAngle(axis, angle); }
    setRotationFromEuler(euler) { this.quaternion.setFromEuler(euler, true); }
    setRotationFromMatrix(m) { this.quaternion.setFromRotationMatrix(m); }
    setRotationFromQuaternion(q) { this.quaternion.copy(q); }
    rotateOnAxis(axis, angle) { _o3dQ.setFromAxisAngle(axis, angle); this.quaternion.multiply(_o3dQ); return this; }
    rotateOnWorldAxis(axis, angle) { _o3dQ.setFromAxisAngle(axis, angle); this.quaternion.premultiply(_o3dQ); return this; }
    rotateX(angle) { return this.rotateOnAxis(_xAxis, angle); }
    rotateY(angle) { return this.rotateOnAxis(_yAxis, angle); }
    rotateZ(angle) { return this.rotateOnAxis(_zAxis, angle); }
    translateOnAxis(axis, distance) {
        _o3dV.copy(axis).applyQuaternion(this.quaternion);
        this.position.add(_o3dV.multiplyScalar(distance));
        return this;
    }
    translateX(distance) { return this.translateOnAxis(_xAxis, distance); }
    translateY(distance) { return this.translateOnAxis(_yAxis, distance); }
    translateZ(distance) { return this.translateOnAxis(_zAxis, distance); }
    localToWorld(vector) { this.updateWorldMatrix(true, false); return vector.applyMatrix4(this.matrixWorld); }
    worldToLocal(vector) { this.updateWorldMatrix(true, false); return vector.applyMatrix4(_o3dM.copy(this.matrixWorld).invert()); }
    /// three.js `lookAt`: cameras and lights turn -Z to the target, the rest +Z;
    /// a parent's rotation is divided out.
    lookAt(x, y, z) {
        if (x?.isVector3 || (x && typeof x === 'object')) _o3dTarget.copy(x);
        else _o3dTarget.set(x, y, z);
        const parent = this.parent;
        this.updateWorldMatrix(true, false);
        _o3dPosition.setFromMatrixPosition(this.matrixWorld);
        if (this.isCamera || this.isLight) _o3dM.lookAt(_o3dPosition, _o3dTarget, this.up);
        else _o3dM.lookAt(_o3dTarget, _o3dPosition, this.up);
        this.quaternion.setFromRotationMatrix(_o3dM);
        if (parent) {
            _o3dM.extractRotation(parent.matrixWorld);
            _o3dQ.setFromRotationMatrix(_o3dM);
            this.quaternion.premultiply(_o3dQ.invert());
        }
        if (this._lookAt) this._lookAt.copy(_o3dTarget);
    }
    add(object) {
        if (arguments.length > 1) {
            for (let i = 0; i < arguments.length; i++) this.add(arguments[i]);
            return this;
        }
        if (Array.isArray(object)) { for (const o of object) this.add(o); return this; }
        if (object === this) {
            console.error('THREE.Object3D.add: object can\'t be added as a child of itself.', object);
            return this;
        }
        if (object && typeof object === 'object') {
            object.removeFromParent?.();
            if (object.parent) object.parent.remove?.(object);
            object.parent = this;
            this.children.push(object);
            this._dirty(_D_CHILDREN);
            object.dispatchEvent?.({ type: 'added' });
            this.dispatchEvent?.({ type: 'childadded', child: object });
        } else {
            console.error('THREE.Object3D.add: object not an instance of THREE.Object3D.', object);
        }
        return this;
    }
    remove(object) {
        if (arguments.length > 1) {
            for (let i = 0; i < arguments.length; i++) this.remove(arguments[i]);
            return this;
        }
        if (Array.isArray(object)) { for (const o of object) this.remove(o); return this; }
        const index = this.children.indexOf(object);
        if (index !== -1) {
            object.parent = null;
            this.children.splice(index, 1);
            this._dirty(_D_CHILDREN);
            object.dispatchEvent?.({ type: 'removed' });
            this.dispatchEvent?.({ type: 'childremoved', child: object });
        }
        return this;
    }
    removeFromParent() {
        const parent = this.parent;
        if (parent !== null) parent.remove(this);
        return this;
    }
    clear() { return this.remove(...this.children); }
    /// three.js `attach`: re-parent keeping the world transform.
    attach(object) {
        this.updateWorldMatrix(true, false);
        _o3dM.copy(this.matrixWorld).invert();
        if (object.parent !== null) {
            object.parent.updateWorldMatrix(true, false);
            _o3dM.multiply(object.parent.matrixWorld);
        }
        object.applyMatrix4(_o3dM);
        object.removeFromParent();
        object.parent = this;
        this.children.push(object);
        this._dirty(_D_CHILDREN);
        object.updateWorldMatrix(false, true);
        object.dispatchEvent?.({ type: 'added' });
        this.dispatchEvent?.({ type: 'childadded', child: object });
        return this;
    }
    getObjectById(id) { return this.getObjectByProperty('id', id); }
    getObjectByName(name) { return this.getObjectByProperty('name', name); }
    getObjectByProperty(name, value) {
        if (this[name] === value) return this;
        for (const c of this.children) {
            const found = c.getObjectByProperty?.(name, value);
            if (found !== undefined) return found;
        }
        return undefined;
    }
    getObjectsByProperty(name, value, result = []) {
        if (this[name] === value) result.push(this);
        for (const c of this.children) c.getObjectsByProperty?.(name, value, result);
        return result;
    }
    getWorldPosition(target = new Vector3()) {
        this.updateWorldMatrix(true, false);
        return target.setFromMatrixPosition(this.matrixWorld);
    }
    getWorldQuaternion(target = new Quaternion()) {
        this.updateWorldMatrix(true, false);
        this.matrixWorld.decompose(_o3dPosition, target, _o3dScale);
        return target;
    }
    getWorldScale(target = new Vector3()) {
        this.updateWorldMatrix(true, false);
        this.matrixWorld.decompose(_o3dPosition, _o3dQuat, target);
        return target;
    }
    getWorldDirection(target = new Vector3()) {
        this.updateWorldMatrix(true, false);
        const e = this.matrixWorld.elements;
        return target.set(e[8], e[9], e[10]).normalize();
    }
    /// The base kind has nothing to hit; Mesh, Line, Points and Sprite cast in
    /// Rust, and an object's own `raycast` (three.js lets code replace it) is
    /// called as three.js calls it.
    raycast() {}
    traverse(callback) {
        callback(this);
        for (const c of this.children) c.traverse?.(callback);
    }
    traverseVisible(callback) {
        if (this.visible === false) return;
        callback(this);
        for (const c of this.children) c.traverseVisible?.(callback);
    }
    traverseAncestors(callback) {
        const parent = this.parent;
        if (parent !== null) { callback(parent); parent.traverseAncestors(callback); }
    }
    updateMatrix() {
        this.matrix.compose(this.position, this.quaternion, this.scale);
        this.matrixWorldNeedsUpdate = true;
    }
    updateMatrixWorld(force) {
        if (this.matrixAutoUpdate) this.updateMatrix();
        if (this.matrixWorldNeedsUpdate || force) {
            if (this.matrixWorldAutoUpdate === true) {
                if (this.parent === null) this.matrixWorld.copy(this.matrix);
                else this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
            }
            this.matrixWorldNeedsUpdate = false;
            force = true;
        }
        for (const c of this.children) c.updateMatrixWorld?.(force);
    }
    updateWorldMatrix(updateParents, updateChildren) {
        const parent = this.parent;
        if (updateParents === true && parent !== null) parent.updateWorldMatrix?.(true, false);
        if (this.matrixAutoUpdate) this.updateMatrix();
        if (this.matrixWorldAutoUpdate === true) {
            if (parent === null || !parent.matrixWorld) this.matrixWorld.copy(this.matrix);
            else this.matrixWorld.multiplyMatrices(parent.matrixWorld, this.matrix);
        }
        if (updateChildren === true) for (const c of this.children) c.updateWorldMatrix?.(false, true);
    }
    toJSON() {
        return {
            metadata: { version: 4.6, type: 'Object', generator: 'threers' },
            object: {
                uuid: this.uuid, type: this.type || this.constructor.name, name: this.name,
                matrix: this.matrix.toArray(),
                position: [this.position.x, this.position.y, this.position.z],
                rotation: [this.rotation.x, this.rotation.y, this.rotation.z],
                scale: [this.scale.x, this.scale.y, this.scale.z],
                children: this.children.map((c) => c.toJSON?.()?.object).filter(Boolean),
            },
        };
    }
    clone(recursive) { return new this.constructor().copy(this, recursive); }
    copy(source, recursive = true) {
        this.name = source.name;
        this.up.copy(source.up);
        this.position.copy(source.position);
        this.rotation.order = source.rotation.order;
        this.quaternion.copy(source.quaternion);
        this.scale.copy(source.scale);
        this.matrix.copy(source.matrix);
        this.matrixWorld.copy(source.matrixWorld);
        this.matrixAutoUpdate = source.matrixAutoUpdate;
        this.matrixWorldAutoUpdate = source.matrixWorldAutoUpdate;
        this.matrixWorldNeedsUpdate = source.matrixWorldNeedsUpdate;
        this.layers.mask = source.layers.mask;
        this.visible = source.visible;
        this.castShadow = source.castShadow;
        this.receiveShadow = source.receiveShadow;
        this.frustumCulled = source.frustumCulled;
        this.renderOrder = source.renderOrder;
        this.animations = source.animations.slice();
        this.userData = JSON.parse(JSON.stringify(source.userData ?? {}));
        if (recursive === true) {
            for (const child of source.children) this.add(child.clone());
        }
        return this;
    }
}
Object3D.DEFAULT_UP = new Vector3(0, 1, 0);
Object3D.DEFAULT_MATRIX_AUTO_UPDATE = true;
Object3D.DEFAULT_MATRIX_WORLD_AUTO_UPDATE = true;

/// three.js `Group`.
export class Group extends Object3D {
    constructor() {
        super();
        this.isGroup = true;
        this.type = 'Group';
    }
}

/// three.js `Bone`: a joint in a skeleton's hierarchy.
export class Bone extends Object3D {
    constructor() {
        super();
        this.isBone = true;
        this.type = 'Bone';
    }
}

/// The mesh-kinds' shared mirror: geometry, material(s), morph influences.
function _mirrorMeshContent(obj, w, h, flags, m) {
    if (flags & _D_GEOMETRY) {
        const key = _geometryKey(obj.geometry);
        const g = _wasmGeometryOf(obj.geometry);
        if (g && m.geometry !== key) {
            w.setNodeGeometry(h, g, undefined);
            m.geometry = key;
        }
        _useBy(obj.geometry, obj);
        // three.js reads `morphTargetInfluences` per draw; the geometry may have
        // brought (or dropped) targets.
        flags |= _D_DATA;
    }
    if (flags & _D_MATERIAL) _mirrorMaterials(w, h, obj.material, obj);
    if (flags & _D_DATA) {
        const inf = obj.morphTargetInfluences;
        if (inf?.length) {
            w.setMorphTargetInfluences(h, Float32Array.from(inf));
            m.morph = Float64Array.from(inf);
        } else if (m.morph) {
            w.setMorphTargetInfluences(h, new Float32Array(0));
            m.morph = null;
        }
    }
}

/// three.js r165 `Mesh`.
export class Mesh extends Object3D {
    constructor(geometry = new BufferGeometry(), material = new MeshBasicMaterial()) {
        super();
        this.isMesh = true;
        this.type = 'Mesh';
        this._geometry = geometry;
        this._material = material;
        this.morphTargetDictionary = undefined;
        this.morphTargetInfluences = undefined;
        this.updateMorphTargets();
    }
    get geometry() { return this._geometry; }
    set geometry(g) { if (g !== this._geometry) { this._geometry = g; this._dirty(_D_GEOMETRY); } }
    get material() { return this._material; }
    set material(m) { this._material = m; this._dirty(_D_MATERIAL); }
    /// three.js: sized from the geometry's first morph attribute, named from
    /// its targets.
    updateMorphTargets() {
        const morphAttributes = this.geometry?.morphAttributes;
        const keys = morphAttributes ? Object.keys(morphAttributes) : [];
        if (keys.length > 0) {
            const morphAttribute = morphAttributes[keys[0]];
            if (morphAttribute !== undefined) {
                this.morphTargetInfluences = [];
                this.morphTargetDictionary = {};
                for (let m = 0, ml = morphAttribute.length; m < ml; m++) {
                    const name = morphAttribute[m].name || String(m);
                    this.morphTargetInfluences.push(0);
                    this.morphTargetDictionary[name] = m;
                }
            }
        }
        this._dirty(_D_DATA);
    }
    copy(source, recursive) {
        super.copy(source, recursive);
        if (source.morphTargetInfluences !== undefined) this.morphTargetInfluences = source.morphTargetInfluences.slice();
        if (source.morphTargetDictionary !== undefined) this.morphTargetDictionary = Object.assign({}, source.morphTargetDictionary);
        this.material = Array.isArray(source.material) ? source.material.slice() : source.material;
        this.geometry = source.geometry;
        return this;
    }
    raycast(raycaster, intersects) { _rcCast(raycaster, this, intersects); }
    _mirrorKind() { return _wasmGeometryOf(this.geometry) ? 'mesh' : 'group'; }
    _mirrorCreate(w) {
        const g = _wasmGeometryOf(this.geometry);
        if (!g) return w.createGroup(false);
        const mat = Array.isArray(this.material) ? this.material[0] : this.material;
        return w.createMesh(g, this._wasmMaterialOverride?.() ?? _wasmMaterialOf(mat));
    }
    _mirrorContent(w, h, flags, m) {
        if (m.kind === 'group') return;
        _mirrorMeshContent(this, w, h, flags, m);
    }
    _mirrorPolled() { return super._mirrorPolled() || this.morphTargetInfluences?.length > 0; }
    _mirrorPoll(w, h, m) {
        super._mirrorPoll(w, h, m);
        const inf = this.morphTargetInfluences;
        if (inf?.length && !_sameNumbers(m.morph, inf)) this._dirty(_D_DATA);
    }
}

/// three.js `Sprite`: a camera-facing quad.
export class Sprite extends Object3D {
    constructor(material = new SpriteMaterial()) {
        super();
        this.isSprite = true;
        this.type = 'Sprite';
        this._material = material;
        this.center = new Vector2(0.5, 0.5);
    }
    get material() { return this._material; }
    set material(m) { this._material = m; this._dirty(_D_MATERIAL); }
    /// three.js needs `raycaster.camera` (set by `setFromCamera`) for sprites.
    raycast(raycaster, intersects) {
        if (raycaster.camera === null || raycaster.camera === undefined) {
            console.error('THREE.Sprite: "Raycaster.camera" needs to be set in order to raycast against sprites.');
            return;
        }
        _rcCast(raycaster, this, intersects);
    }
    copy(source, recursive) {
        super.copy(source, recursive);
        if (source.center !== undefined) this.center.copy(source.center);
        this.material = source.material;
        return this;
    }
    _mirrorKind() { return 'sprite'; }
    _mirrorCreate(w) { return w.createSprite(_wasmMaterialOf(this.material)); }
    _mirrorContent(w, h, flags, m) {
        if (flags & _D_MATERIAL) _mirrorMaterials(w, h, this.material, this);
        if (flags & _D_DATA) {
            w.setSpriteCenter(h, this.center.x, this.center.y);
            m.center = [this.center.x, this.center.y];
        }
    }
    _mirrorPolled() { return true; }
    _mirrorPoll(w, h, m) {
        super._mirrorPoll(w, h, m);
        if (!_sameNumbers(m.center, [this.center.x, this.center.y])) this._dirty(_D_DATA);
    }
}

/// three.js r165 `Skeleton`. Rendering and raycasting take their bone
/// matrices from the mirrored bones in Rust; `update()` fills
/// `boneMatrices` for code that reads it.
export class Skeleton {
    constructor(bones = [], boneInverses = []) {
        this.uuid = MathUtils.generateUUID();
        this.bones = bones.slice(0);
        this.boneInverses = boneInverses.slice(0);
        this.boneMatrices = null;
        this.boneTexture = null;
        this.init();
    }
    init() {
        const bones = this.bones, boneInverses = this.boneInverses;
        this.boneMatrices = new Float32Array(bones.length * 16);
        if (boneInverses.length === 0) {
            this.calculateInverses();
        } else if (bones.length !== boneInverses.length) {
            console.warn('THREE.Skeleton: Number of inverse bone matrices does not match amount of bones.');
            this.boneInverses = [];
            for (let i = 0, il = this.bones.length; i < il; i++) this.boneInverses.push(new Matrix4());
        }
    }
    calculateInverses() {
        this.boneInverses.length = 0;
        for (let i = 0, il = this.bones.length; i < il; i++) {
            const inverse = new Matrix4();
            if (this.bones[i]) inverse.copy(this.bones[i].matrixWorld).invert();
            this.boneInverses.push(inverse);
        }
        _notifyUsers(this, _D_DATA, (o) => o.skeleton === this);
    }
    pose() {
        for (let i = 0, il = this.bones.length; i < il; i++) {
            const bone = this.bones[i];
            if (bone) bone.matrixWorld.copy(this.boneInverses[i]).invert();
        }
        for (let i = 0, il = this.bones.length; i < il; i++) {
            const bone = this.bones[i];
            if (bone) {
                if (bone.parent && bone.parent.isBone) {
                    bone.matrix.copy(bone.parent.matrixWorld).invert();
                    bone.matrix.multiply(bone.matrixWorld);
                } else {
                    bone.matrix.copy(bone.matrixWorld);
                }
                bone.matrix.decompose(bone.position, bone.quaternion, bone.scale);
            }
        }
    }
    update() {
        const bones = this.bones, boneInverses = this.boneInverses, boneMatrices = this.boneMatrices;
        for (let i = 0, il = bones.length; i < il; i++) {
            const matrix = bones[i] ? bones[i].matrixWorld : _o3dM.identity();
            new Matrix4().multiplyMatrices(matrix, boneInverses[i]).toArray(boneMatrices, i * 16);
        }
    }
    clone() { return new Skeleton(this.bones, this.boneInverses); }
    getBoneByName(name) {
        for (let i = 0, il = this.bones.length; i < il; i++) if (this.bones[i].name === name) return this.bones[i];
        return undefined;
    }
    dispose() {}
}

const AttachedBindMode = 'attached';
const DetachedBindMode = 'detached';

/// three.js r165 `SkinnedMesh`. The bones are mirrored nodes; Rust computes
/// the skinning palette from them (with the bind matrix and mode) at every
/// world update, for the renderers and the raycaster alike.
export class SkinnedMesh extends Mesh {
    constructor(geometry, material) {
        super(geometry, material);
        this.isSkinnedMesh = true;
        this.type = 'SkinnedMesh';
        this._bindMode = AttachedBindMode;
        this.bindMatrix = new Matrix4();
        this.bindMatrixInverse = new Matrix4();
        this.boundingBox = null;
        this.boundingSphere = null;
        this.skeleton = undefined;
    }
    get bindMode() { return this._bindMode; }
    set bindMode(v) { if (v !== this._bindMode) { this._bindMode = v; this._dirty(_D_DATA); } }
    bind(skeleton, bindMatrix) {
        this.skeleton = skeleton;
        if (bindMatrix === undefined) {
            this.updateMatrixWorld(true);
            this.skeleton.calculateInverses();
            bindMatrix = this.matrixWorld;
        }
        this.bindMatrix.copy(bindMatrix);
        this.bindMatrixInverse.copy(bindMatrix).invert();
        _useBy(skeleton, this);
        this._dirty(_D_DATA);
    }
    pose() { this.skeleton?.pose(); }
    /// three.js r165: the box / sphere of every vertex as posed now
    /// (`getVertexPosition`), walked in Rust over the mirrored node.
    computeBoundingBox() {
        if (this.boundingBox === null) this.boundingBox = new Box3();
        this.boundingBox.makeEmpty();
        const b = _objectVertexBounds(this, 0, null);
        if (b) this.boundingBox.set(_o3dV.set(b[0], b[1], b[2]), _o3dTarget.set(b[3], b[4], b[5]));
    }
    computeBoundingSphere() {
        if (this.boundingSphere === null) this.boundingSphere = new Sphere();
        this.boundingSphere.makeEmpty();
        const s = _objectVertexBounds(this, 1, null);
        if (s) { this.boundingSphere.center.set(s[0], s[1], s[2]); this.boundingSphere.radius = s[3]; }
    }
    normalizeSkinWeights() {
        const vector = new Vector4();
        const skinWeight = this.geometry.attributes.skinWeight;
        for (let i = 0, l = skinWeight.count; i < l; i++) {
            vector.fromBufferAttribute(skinWeight, i);
            const scale = 1.0 / (Math.abs(vector.x) + Math.abs(vector.y) + Math.abs(vector.z) + Math.abs(vector.w));
            if (scale !== Infinity) vector.multiplyScalar(scale);
            else vector.set(1, 0, 0, 0);
            const o = i * skinWeight.itemSize;
            skinWeight.array[o] = vector.x; skinWeight.array[o + 1] = vector.y;
            skinWeight.array[o + 2] = vector.z; skinWeight.array[o + 3] = vector.w;
        }
        skinWeight.needsUpdate = true;
    }
    updateMatrixWorld(force) {
        super.updateMatrixWorld(force);
        if (this.bindMode === AttachedBindMode) this.bindMatrixInverse.copy(this.matrixWorld).invert();
        else if (this.bindMode === DetachedBindMode) this.bindMatrixInverse.copy(this.bindMatrix).invert();
    }
    copy(source, recursive) {
        super.copy(source, recursive);
        this.bindMode = source.bindMode;
        this.bindMatrix.copy(source.bindMatrix);
        this.bindMatrixInverse.copy(source.bindMatrixInverse);
        this.skeleton = source.skeleton;
        this._dirty(_D_DATA);
        return this;
    }
    _mirrorKind() { return _wasmGeometryOf(this.geometry) ? 'skinned' : 'group'; }
    _mirrorCreate(w) {
        const g = _wasmGeometryOf(this.geometry);
        if (!g) return w.createGroup(false);
        const mat = Array.isArray(this.material) ? this.material[0] : this.material;
        return w.createSkinnedMesh(g, _wasmMaterialOf(mat));
    }
    _mirrorContent(w, h, flags, m) {
        if (m.kind === 'group') return;
        _mirrorMeshContent(this, w, h, flags, m);
        if (flags & (_D_DATA | _D_GEOMETRY)) {
            const skeleton = this.skeleton;
            const bones = skeleton?.bones ?? [];
            const ids = new Float64Array(bones.length);
            const inverses = new Float32Array(bones.length * 16);
            bones.forEach((bone, i) => {
                if (bone instanceof Object3D) {
                    _mirrorEnsure(bone, m.graph);
                    ids[i] = bone._m.id;
                }
                (skeleton.boneInverses[i] ?? _o3dM.identity()).toArray(inverses, i * 16);
            });
            w.setSkeleton(h, ids, inverses);
            w.setBindMatrix(h, Float32Array.from(this.bindMatrix.elements), this.bindMode === DetachedBindMode);
        }
    }
}

/// three.js r165 `InstancedMesh`. `instanceMatrix` / `instanceColor` reach
/// the node when flagged `needsUpdate` (and when the mesh is first
/// mirrored), as three.js uploads them; a raycast reads the matrices as they
/// are, as three.js's does.
/// threers extension (not in three.js): shell fur grown on `geometry`
/// (`threers::fur`). Options, all optional: `length`, `density` (strands per
/// UV unit), `thickness`, `taper`, `gravity` (Vector3 or [x, y, z]), `curl`,
/// `clumping`, `clumpSize`, `lengthVariation`, `seed`, `shells`, `rootColor`,
/// `tipColor`, `occlusion`, `specular`, `specularExponent`, `specularShift`,
/// `wind` (Vector3 or array), `windFrequency`, `flutter`, `stiffness`,
/// `damping`, `response`, `maxLag`, and `placement: 'surface'` with `cell` to
/// root strands over the surface rather than the UVs. The renderer animates
/// its wind, flutter and inertia at every `render`.
export class Fur extends Object3D {
    constructor(geometry, options = {}) {
        super();
        this.isFur = true;
        this.type = 'Fur';
        this.geometry = geometry;
        this.options = { ...options };
        // Fur casts and receives shadows by default — the coat shadowing what
        // is under it and itself; `shadowHeight: 0` casts none.
        this.castShadow = options.shadowHeight !== 0;
        this.receiveShadow = true;
    }
    _furMaterial() {
        const F = _optional.WebFurMaterial;
        if (!F) return null;
        const o = this.options;
        const f = new F();
        const v3 = (v, d) => (v == null ? d : Array.isArray(v) ? v : [v.x, v.y, v.z]);
        if (o.length != null) f.setLength(o.length);
        if (o.density != null) f.setDensity(o.density);
        if (o.thickness != null) f.setThickness(o.thickness);
        if (o.taper != null) f.setTaper(o.taper);
        if (o.gravity != null) f.setGravity(...v3(o.gravity));
        if (o.curl != null) f.setCurl(o.curl);
        if (o.clumping != null || o.clumpSize != null) f.setClumping(o.clumping ?? 0.3, o.clumpSize ?? 4);
        if (o.lengthVariation != null) f.setLengthVariation(o.lengthVariation);
        if (o.seed != null) f.setSeed(o.seed >>> 0);
        if (o.shells != null) f.setShells(o.shells);
        if (o.rootColor != null || o.tipColor != null) {
            f.setColors(_color(o.rootColor ?? 0x2b1d10), _color(o.tipColor ?? 0xb08a5a));
        }
        if (o.occlusion != null) f.setOcclusion(o.occlusion);
        if (o.specular != null || o.specularExponent != null || o.specularShift != null) {
            f.setSpecular(o.specular ?? 0.25, o.specularExponent ?? 60, o.specularShift ?? 0.1);
        }
        if (o.wind != null || o.windFrequency != null) f.setWind(...v3(o.wind, [0, 0, 0]), o.windFrequency ?? 0.6);
        if (o.flutter != null) f.setFlutter(o.flutter);
        if (o.stiffness != null || o.damping != null || o.response != null || o.maxLag != null) {
            f.setDynamics(o.stiffness ?? 80, o.damping ?? 0.35, o.response ?? 1, o.maxLag ?? 1.2);
        }
        if (o.placement === 'surface') f.setSurfacePlacement(o.cell ?? 0.02);
        if (o.shadowHeight != null) f.setShadowHeight(o.shadowHeight);
        if (o.lod != null) f.setLod(!!o.lod);
        if (o.drag != null || o.turbulence != null) f.setAir?.(o.drag ?? 0.15, o.turbulence ?? 0.35);
        return f;
    }
    /// A hand in the coat: within `radius` of `point` (world), strands are
    /// pushed by `push` (a Vector3 or [x, y, z]: a pull per unit of their
    /// length, along the hand's motion) and parted away from it by `part`.
    /// Call it as the hand moves; let go and the strands spring back.
    stroke(point, push = [0, 0, 0], radius = 0.3, part = 0.5) {
        const m = this._m;
        if (!m?.graph?.w?.strokeFur) return;
        const p = Array.isArray(push) ? push : [push.x, push.y, push.z];
        m.graph.w.strokeFur(m.handle, point.x, point.y, point.z, radius, p[0], p[1], p[2], part);
    }
    _mirrorKind() { return _wasmGeometryOf(this.geometry) && _optional.WebFurMaterial ? 'fur' : 'group'; }
    _mirrorCreate(w) {
        const g = _wasmGeometryOf(this.geometry);
        const f = g && w.createFur ? this._furMaterial() : null;
        return f ? w.createFur(g, f) : w.createGroup(false);
    }
}

/// Strand hair (a threers extension): `count` strands grown on `geometry`,
/// drawn as camera-facing ribbons. Takes `Fur`'s options (shells, density,
/// thickness and placement aside) plus `count`, `segments`, `width` and
/// `shadowStrands` (the share of strands that cast shadows, 0.25; 0 for none).
///
/// Given a `SkinnedMesh` instead of a geometry, the hair grows on its mesh,
/// is added as its child, and follows its animation: each strand carried by
/// its triangle and trailing its root on a spring (`stiffness`, `damping`).
/// Leave its transform at the identity.
export class Hair extends Fur {
    constructor(geometryOrSkinnedMesh, options = {}) {
        const skinned = geometryOrSkinnedMesh?.isSkinnedMesh ? geometryOrSkinnedMesh : null;
        super(skinned ? null : geometryOrSkinnedMesh, options);
        this.isHair = true;
        this.type = 'Hair';
        // Hair casts by default (its tubes are made for it); `shadowStrands: 0`
        // makes none, `castShadow = false` turns them off.
        this.castShadow = options.shadowStrands !== 0;
        this.skinnedMesh = skinned;
        skinned?.add(this);
    }
    _mirrorKind() {
        const grows = this.skinnedMesh || _wasmGeometryOf(this.geometry);
        return grows && _optional.WebFurMaterial ? 'hair' : 'group';
    }
    _mirrorCreate(w) {
        const o = this.options;
        const args = [(o.count ?? 4000) >>> 0, (o.segments ?? 8) >>> 0, o.width ?? 0.006, o.shadowStrands];
        // Skinned: on the mesh's node in this graph (made before its
        // children are), while this is still its child.
        const sm = this.skinnedMesh;
        if (sm) {
            const m = sm._m;
            const f = m && m.graph.w === w && this.parent === sm && w.createSkinnedHair ? this._furMaterial() : null;
            return f ? w.createSkinnedHair(m.handle, f, ...args) : w.createGroup(false);
        }
        const g = _wasmGeometryOf(this.geometry);
        const f = g && w.createHair ? this._furMaterial() : null;
        return f ? w.createHair(g, f, ...args) : w.createGroup(false);
    }
}

/// Particles (a threers extension, `threers::particles`): a system made from
/// a preset — `'fire'`, `'embers'`, `'heatHaze'`, `'smoke'`, `'sparks'`, `'rain'`, `'snowfall'`,
/// `'sandStream'`, `'dust'`, `'fountain'` — or `'custom'`, simulated with
/// gravity, air drag set by each particle's terminal speed, wind and
/// turbulence, and drawn as one instanced draw. Its emitters sit where this
/// object is: move it and they follow (particles already out stay where they
/// are). The renderer steps it at every `render`.
///
/// Preset options: `radius` (fire, smoke), `size` ([x, z] metres: rain,
/// snowfall), `height` (rain, snowfall: how far above the object they start;
/// fountain: how high it throws), `rate` (per second), `depthPerSecond`
/// (snowfall: metres of snow a second), `flow` (sandStream: m³ a second),
/// `count` (dust), `wind` ([x, y, z] m/s). Any system: `gravity`,
/// `turbulence` ({ strength, scale, churn }), `color` ([r, g, b, a], linear),
/// `colorOverLife` ([[t, r, g, b, a], …]), `sizeOverLife` ([[t, s], …]),
/// `particleSize` ([min, max] metres), `lifetime` ([min, max] s), `speed`,
/// `terminalSpeed`, `look` ({ kind, blend, softness, shutter, intensity,
/// lit, detail, stretch }), `flutter` (m/s²), `cooling` ({ time,
/// coldGravity }), `boundaryLayer` ({ ground, roughness, reference }),
/// `spray` ({ count, speed, angle, lifetime, size }), `shading` ({ scattering,
/// selfShadow, glint, flake, castsShadow }), `temperatureOverLife`
/// ([[t, kelvin], …]), `maxParticles`, `seed`,
/// and `settleOn` — a `SnowCover` or `SandBed` to land in.
export class Particles extends Object3D {
    constructor(preset = 'custom', options = {}) {
        super();
        this.isParticles = true;
        this.type = 'Particles';
        this.preset = preset;
        this.options = { ...options };
        this.frustumCulled = false;
        this.receiveShadow = true;
        if (options.settleOn) this.settleOn(options.settleOn);
    }
    _system() {
        const P = _optional.WebParticleSystem;
        if (!P) return null;
        const o = this.options;
        const v3 = (v, d) => (v == null ? d : Array.isArray(v) ? v : [v.x, v.y, v.z]);
        const wind = v3(o.wind, [0, 0, 0]);
        const size = o.size ?? [8, 8];
        let s;
        switch (this.preset) {
            case 'fire': s = P.fire(0, 0, 0, o.radius ?? 0.3); break;
            case 'smoke': s = P.smoke(0, 0, 0, o.radius ?? 0.25); break;
            case 'sparks': s = P.sparks(0, 0, 0); break;
            case 'embers': s = P.embers(0, 0, 0, o.radius ?? 0.25); break;
            case 'heatHaze': s = P.heatHaze(0, 0, 0, o.radius ?? 0.25); break;
            case 'rain': s = P.rain(0, 0, 0, size[0], size[1], o.height ?? 6, o.rate ?? 4000, ...wind); break;
            case 'snowfall':
                s = P.snowfall(0, 0, 0, size[0], size[1], o.height ?? 5, o.rate ?? 4000, o.depthPerSecond ?? 0.002, ...wind);
                break;
            case 'sandStream': s = P.sandStream(0, 0, 0, o.rate ?? 4000, o.flow ?? 0.001); break;
            case 'dust': s = P.dust(0, 0, 0, o.count ?? 120); break;
            case 'fountain': s = P.fountain(0, 0, 0, o.height ?? 1.5); break;
            default:
                s = new P();
                s.setEmitterPoint(0, 0, 0);
        }
        if (o.wind != null && this.preset !== 'rain' && this.preset !== 'snowfall') s.setWind(...wind);
        if (o.gravity != null) s.setGravity(...v3(o.gravity));
        if (o.turbulence) s.setTurbulence(o.turbulence.strength ?? 0.3, o.turbulence.scale ?? 1, o.turbulence.churn ?? 0.3);
        if (o.look) {
            const l = o.look;
            s.setLook(l.kind ?? 'billboard', l.blend ?? 'alpha', l.softness ?? 1, l.shutter ?? 0.02, l.intensity ?? 1, !!l.lit,
                l.detail ?? 0, l.stretch ?? 0);
        }
        if (o.rate != null && this.preset === 'custom') s.setRate(o.rate);
        if (o.particleSize) s.setSize(o.particleSize[0], o.particleSize[1] ?? o.particleSize[0]);
        if (o.lifetime) s.setLifetime(o.lifetime[0], o.lifetime[1] ?? o.lifetime[0]);
        if (o.speed) s.setSpeed(o.speed[0], o.speed[1] ?? o.speed[0]);
        if (o.terminalSpeed) s.setTerminalSpeed(o.terminalSpeed[0], o.terminalSpeed[1] ?? o.terminalSpeed[0]);
        if (o.color) s.setColor(o.color[0], o.color[1], o.color[2], o.color[3] ?? 1);
        if (o.colorOverLife) s.setColorOverLife(Float32Array.from(o.colorOverLife.flat()));
        if (o.sizeOverLife) s.setSizeOverLife(Float32Array.from(o.sizeOverLife.flat()));
        if (o.shading) {
            const h = o.shading;
            s.setShading(h.scattering ?? 0, h.selfShadow ?? 0, h.glint ?? 0, !!h.flake, !!h.castsShadow);
        }
        if (o.temperatureOverLife) s.setTemperatureOverLife(Float32Array.from(o.temperatureOverLife.flat()));
        if (o.flutter != null) s.setFlutter(o.flutter);
        if (o.cooling) s.setCooling(o.cooling.time ?? 1, ...v3(o.cooling.coldGravity, [0, 0, 0]));
        if (o.boundaryLayer) {
            const b = o.boundaryLayer;
            s.setBoundaryLayer(b.ground ?? 0, b.roughness ?? 0.01, b.reference ?? 10);
        }
        if (o.spray) {
            const p = o.spray;
            s.setSpray(...(p.count ?? [1, 3]), ...(p.speed ?? [0.5, 1.5]), p.angle ?? 1, ...(p.lifetime ?? [0.2, 0.5]), p.size ?? 0.6);
        }
        if (o.maxParticles != null) s.setMaxParticles(o.maxParticles >>> 0);
        if (o.seed != null) s.setSeed(o.seed >>> 0);
        return s;
    }
    /// Drive `light` (a PointLight or SpotLight in the same scene): it sits
    /// at this system's glowing centre with about `intensity`, flickering as
    /// the flames come and go.
    lights(light, intensity = 6) {
        this._light = [light, intensity];
        _particleLinks.add(this);
    }
    /// Land in `layer` (a `SnowCover` or `SandBed`) and pile up there.
    settleOn(layer) {
        this._settleOn = layer;
        _particleLinks.add(this);
    }
    /// Particles alive now.
    get count() {
        const m = this._m;
        return m?.graph?.w?.particleCount ? m.graph.w.particleCount(m.handle) : 0;
    }
    /// `count` particles at once.
    burst(count) {
        const m = this._m;
        m?.graph?.w?.burstParticles?.(m.handle, count >>> 0);
    }
    /// Particles a second: 0 stops the flow.
    setRate(rate) {
        const m = this._m;
        m?.graph?.w?.setParticleRate?.(m.handle, rate);
    }
    setWind(x, y, z) {
        const m = this._m;
        m?.graph?.w?.setParticleWind?.(m.handle, x, y, z);
    }
    _mirrorKind() { return _optional.WebParticleSystem ? 'particles' : 'group'; }
    _mirrorCreate(w) {
        const s = w.createParticles ? this._system() : null;
        return s ? w.createParticles(s) : w.createGroup(false);
    }
}

/// Snow lying in the scene (a threers extension): a grid `size` ([x, z]
/// metres, centred on `center` [x, z]) of `resolution` cells along its longer
/// side, resting on whatever is below it — found by rays cast down from
/// `castFrom` metres the first time the scene is drawn. Snowfall settled on
/// it (`Particles.settleOn`) piles up, and slumps where steeper than snow
/// holds. Leave its transform at the identity.
export class SnowCover extends Object3D {
    constructor(options = {}) {
        super();
        this.isSnowCover = true;
        this.type = 'SnowCover';
        this.options = { size: [16, 16], center: [0, 0], resolution: 256, castFrom: 30, ...options };
        this.castShadow = true;
        this.receiveShadow = true;
    }
    /// A wind drifting the lying snow toward `direction` (`'+x'`, `'-x'`,
    /// `'+z'`, `'-z'`): `slabsPerSecond` small slabs lifted and carried
    /// downwind, piling in the lee of whatever stands in it.
    drift(direction = '+x', slabsPerSecond = 20000) {
        this._wind = [direction, slabsPerSecond];
        const m = this._m;
        if (m?.graph?.w?.setLayerWind) m.graph.w.setLayerWind(m.handle, direction, slabsPerSecond);
    }
    /// Snow lying now, m³.
    get volume() {
        const m = this._m;
        return m?.graph?.w?.layerVolume ? m.graph.w.layerVolume(m.handle) : 0;
    }
    _mirrorKind() { return _optional.WebParticleSystem ? 'snowcover' : 'group'; }
    _mirrorCreate(w) {
        const o = this.options;
        if (!w.createSnowCover) return w.createGroup(false);
        const h = w.createSnowCover(o.center[0], o.center[1], o.size[0], o.size[1], o.resolution >>> 0, o.castFrom);
        if (this._wind) w.setLayerWind(h, ...this._wind);
        return h;
    }
}

/// Sand lying in the scene (a threers extension): as `SnowCover`, piling
/// poured sand at its angle of repose (34°). `spread(depth, roughness)` lays
/// a sheet; `blow(direction, slabsPerSecond)` sets a wind (`'+x'`, `'-x'`,
/// `'+z'`, `'-z'`) that shapes it into dunes.
export class SandBed extends SnowCover {
    constructor(options = {}) {
        super(options);
        this.isSnowCover = false;
        this.isSandBed = true;
        this.type = 'SandBed';
    }
    spread(depth, roughness = 0) {
        this._spread = [depth, roughness];
        const m = this._m;
        if (m?.graph?.w?.spreadSand) m.graph.w.spreadSand(m.handle, depth, roughness);
    }
    /// Shape the sand now: `slabs` slab moves of wind toward `direction`
    /// (dunes from a flat sheet take a few million on a 200-cell grid).
    shape(direction = '+x', slabs = 2000000) {
        this._shape = [direction, slabs >>> 0];
        const m = this._m;
        if (m?.graph?.w?.blowSand) m.graph.w.blowSand(m.handle, direction, slabs >>> 0);
    }
    blow(direction = '+x', slabsPerSecond = 20000) {
        this._wind = [direction, slabsPerSecond];
        const m = this._m;
        if (m?.graph?.w?.setLayerWind) m.graph.w.setLayerWind(m.handle, direction, slabsPerSecond);
    }
    /// Grow a dune field on it (replacing its sand): `wavelength` and
    /// `height` in metres, crests drawn out `stretch` times across the wind,
    /// worked by the wind toward `wind` (`'+x'`, …) for `shaping` slab moves
    /// a cell.
    dunes(options = {}) {
        const o = { wavelength: 110, height: 14, stretch: 2.2, shaping: 24, seed: 7, wind: '+x', ...options };
        this._dunes = [o.wavelength, o.height, o.stretch, o.shaping, o.seed >>> 0, o.wind];
        const m = this._m;
        if (m?.graph?.w?.shapeDunes) m.graph.w.shapeDunes(m.handle, ...this._dunes);
    }
    /// Its look: `color` and `shadowColor` ([r, g, b] linear), `ripples`
    /// (0–1), `rippleWavelength` (m), `streaks` (sand streaking up the
    /// windward faces, 0 for none), `stylize` (Journey's high-contrast
    /// shading, 0–1), `fogGlow` (the haze brighter toward the sun).
    look(options = {}) {
        const o = {
            color: [0.72, 0.42, 0.18], shadowColor: [0.34, 0.22, 0.13], ripples: 0.7, rippleWavelength: 0.08,
            streaks: 0, stylize: 0, fogGlow: 0, ...this._look, ...options,
        };
        this._look = o;
        const m = this._m;
        if (m?.graph?.w?.setSandLook) m.graph.w.setSandLook(m.handle, ...this._lookArgs());
    }
    _lookArgs() {
        const o = this._look;
        return [...o.color, ...o.shadowColor, o.ripples, o.rippleWavelength, o.streaks, o.stylize, o.fogGlow];
    }
    _mirrorKind() { return _optional.WebParticleSystem ? 'sandbed' : 'group'; }
    _mirrorCreate(w) {
        const o = this.options;
        if (!w.createSandBed) return w.createGroup(false);
        const h = w.createSandBed(o.center[0], o.center[1], o.size[0], o.size[1], o.resolution >>> 0, o.castFrom);
        if (o.ripples != null || o.rippleWind) {
            const rw = o.rippleWind ?? [1, 0, 0];
            w.setSandRipples(h, o.ripples ?? 0.7, rw[0], rw[2] ?? rw[1] ?? 0);
        }
        if (this._spread) w.spreadSand(h, ...this._spread);
        if (this._dunes && w.shapeDunes) w.shapeDunes(h, ...this._dunes);
        if (this._shape) w.blowSand(h, ...this._shape);
        if (this._wind) w.setLayerWind(h, ...this._wind);
        if (this._look && w.setSandLook) w.setSandLook(h, ...this._lookArgs());
        return h;
    }
}

/// A time of day over the desert (a threers extension): `'dawn'`, `'day'`,
/// `'sunset'` or `'night'` — the sun, the sky, the haze and the sand, for
/// lighting a scene. Colours are linear.
export function desertTime(name = 'day') {
    const f = _optional.desertTime ? _optional.desertTime(name) : null;
    if (!f) return null;
    const c = (i) => [f[i], f[i + 1], f[i + 2]];
    return {
        sun: c(0), sunColor: c(3), sunIntensity: f[6], zenith: c(7), horizon: c(10), haze: f[13], glow: f[14],
        skyLight: c(15), groundLight: c(18), skyIntensity: f[21], sand: c(22), shadow: c(25), stars: f[28],
    };
}

/// The desert's sky (a threers extension): a dome `radius` metres across for
/// a time of day, its horizon the haze the sand fades into. Keep it on the
/// camera (`sky.position.copy(camera.position)`) and the camera's far plane
/// beyond it.
export class DesertSky extends Object3D {
    constructor(time = 'day', radius = 2200) {
        super();
        this.isDesertSky = true;
        this.type = 'DesertSky';
        this.time = time;
        this.radius = radius;
        this.frustumCulled = false;
    }
    setTime(time) {
        this.time = time;
        const m = this._m;
        m?.graph?.w?.setDesertSky?.(m.handle, time);
    }
    _mirrorKind() { return _optional.WebParticleSystem ? 'desertsky' : 'group'; }
    _mirrorCreate(w) {
        return w.createDesertSky ? w.createDesertSky(this.time, this.radius) : w.createGroup(false);
    }
}

/// Sand blown off the crests of the dunes on `bed` (a `SandBed` with
/// `dunes()`, added to the scene first) into `wind` ([x, y, z] m/s): `rate`
/// puffs a second, in `color` ([r, g, b] linear).
export class Spindrift extends Object3D {
    constructor(bed, options = {}) {
        super();
        this.isSpindrift = true;
        this.type = 'Spindrift';
        this.bed = bed;
        this.options = { wind: [9, 0, 1.5], rate: 3000, color: [0.8, 0.5, 0.25], ...options };
        this.frustumCulled = false;
        this.receiveShadow = true;
    }
    _mirrorKind() { return _optional.WebParticleSystem ? 'particles' : 'group'; }
    _mirrorCreate(w) {
        const b = this.bed?._m;
        const o = this.options;
        if (!w.createSpindrift || !b) return w.createGroup(false);
        return w.createSpindrift(b.handle, ...o.wind, o.rate, ...o.color);
    }
}

/// Snow or rain filling a box of air round the camera (a threers
/// extension): `count` flakes or drops placed by the GPU each frame, so a
/// hundred thousand cost nothing on the CPU. They do not land — for snow
/// that piles up, add a few thousand `Particles('snowfall')` settling on a
/// `SnowCover` as well. Options: `halfWidth`, `halfDepth`, `height`,
/// `floor` (metres) and `wind` ([x, y, z] m/s).
export class Weather extends Object3D {
    constructor(kind = 'snow', count = 50000, options = {}) {
        super();
        this.isWeather = true;
        this.type = 'Weather';
        this.kind = kind;
        this.options = { count, halfWidth: 15, halfDepth: 15, height: 12, floor: 0, wind: [0.4, 0, 0.1], ...options };
        this.frustumCulled = false;
    }
    setWind(x, y, z) {
        this.options.wind = [x, y, z];
        const m = this._m;
        m?.graph?.w?.setWeatherWind?.(m.handle, x, y, z);
    }
    _mirrorKind() { return _optional.WebParticleSystem ? 'weather' : 'group'; }
    _mirrorCreate(w) {
        const o = this.options;
        if (!w.createWeather) return w.createGroup(false);
        return w.createWeather(this.kind, o.count >>> 0, o.halfWidth, o.halfDepth, o.height, o.floor, ...o.wind);
    }
}

/// Particles waiting to be linked to the layer they settle on: both have to
/// be in the same wasm scene first.
const _particleLinks = new Set();
function _linkParticles() {
    for (const p of _particleLinks) {
        const a = p._m;
        if (!a) continue;
        let done = true;
        if (p._settleOn) {
            const b = p._settleOn._m;
            if (b && a.graph === b.graph && a.graph.w.settleParticles) {
                a.graph.w.settleParticles(a.handle, b.handle);
                p._settleOn = null;
            } else {
                done = false;
            }
        }
        if (p._light) {
            const l = p._light[0]._m;
            if (l && a.graph === l.graph && a.graph.w.lightParticles) {
                a.graph.w.lightParticles(a.handle, l.handle, p._light[1]);
                p._light = null;
            } else {
                done = false;
            }
        }
        if (done) _particleLinks.delete(p);
    }
}

export class InstancedMesh extends Mesh {
    constructor(geometry, material, count) {
        super(geometry, material);
        this.isInstancedMesh = true;
        this._instanceMatrix = null;
        this._instanceColor = null;
        this.instanceMatrix = new InstancedBufferAttribute(new Float32Array((count | 0) * 16), 16);
        this.morphTexture = null;
        this._count = count | 0;
        this.boundingBox = null;
        this.boundingSphere = null;
        const id = new Matrix4();
        for (let i = 0; i < this._count; i++) this.setMatrixAt(i, id);
    }
    get count() { return this._count; }
    set count(v) { if (v !== this._count) { this._count = v; this._dirty(_D_DATA); } }
    get instanceMatrix() { return this._instanceMatrix; }
    set instanceMatrix(a) { this._instanceMatrix = a; a?._owners?.add(this); this._dirty(_D_DATA); }
    get instanceColor() { return this._instanceColor; }
    set instanceColor(a) { this._instanceColor = a; a?._owners?.add(this); this._dirty(_D_DATA); }
    /// An instance buffer was flagged `needsUpdate`.
    _attributeChanged() { this._dirty(_D_DATA); }
    getMatrixAt(index, matrix) { matrix.fromArray(this.instanceMatrix.array, index * 16); }
    setMatrixAt(index, matrix) { matrix.toArray(this.instanceMatrix.array, index * 16); }
    getColorAt(index, color) { color.fromArray(this.instanceColor.array, index * 3); }
    setColorAt(index, color) {
        if (this.instanceColor === null) {
            this.instanceColor = new InstancedBufferAttribute(new Float32Array(this.instanceMatrix.count * 3).fill(1), 3);
        }
        color.toArray(this.instanceColor.array, index * 3);
    }
    computeBoundingBox() {
        const geometry = this.geometry;
        const count = this.count;
        if (this.boundingBox === null) this.boundingBox = new Box3();
        const box = _geometryBoundingBox(geometry);
        this.boundingBox.makeEmpty();
        for (let i = 0; i < count; i++) {
            this.getMatrixAt(i, _instanceLocalMatrix);
            _instanceBox.copy(box).applyMatrix4(_instanceLocalMatrix);
            this.boundingBox.union(_instanceBox);
        }
    }
    computeBoundingSphere() {
        const geometry = this.geometry;
        const count = this.count;
        if (this.boundingSphere === null) this.boundingSphere = new Sphere();
        const sphere = _geometryBoundingSphere(geometry);
        this.boundingSphere.makeEmpty();
        for (let i = 0; i < count; i++) {
            this.getMatrixAt(i, _instanceLocalMatrix);
            _instanceSphere.copy(sphere).applyMatrix4(_instanceLocalMatrix);
            this.boundingSphere.union(_instanceSphere);
        }
    }
    copy(source, recursive) {
        super.copy(source, recursive);
        this.instanceMatrix =new InstancedBufferAttribute(Float32Array.from(source.instanceMatrix.array), 16);
        if (source.instanceColor !== null) this.instanceColor = new InstancedBufferAttribute(Float32Array.from(source.instanceColor.array), 3);
        this.count = source.count;
        return this;
    }
    dispose() { this.dispatchEvent?.({ type: 'dispose' }); }
    _mirrorKind() { return _wasmGeometryOf(this.geometry) ? 'instanced' : 'group'; }
    _mirrorCreate(w) {
        const g = _wasmGeometryOf(this.geometry);
        if (!g) return w.createGroup(false);
        const mat = Array.isArray(this.material) ? this.material[0] : this.material;
        return w.createInstancedMesh(g, _wasmMaterialOf(mat), this.count);
    }
    _mirrorContent(w, h, flags, m) {
        if (m.kind === 'group') return;
        _mirrorMeshContent(this, w, h, flags, m);
        if (flags & _D_DATA) {
            const n = Math.max(0, Math.min(this.count, this.instanceMatrix.count));
            const matrices = this.instanceMatrix.array.subarray(0, n * 16);
            w.setInstanceMatrices(h, Float32Array.from(matrices));
            m.instances = Float32Array.from(matrices);
            const c = this.instanceColor;
            w.setInstanceColors(h, c ? Float32Array.from(c.array.subarray(0, n * 3)) : new Float32Array(0));
        }
    }
    /// Before a cast: matrices edited in place reach the node (three.js's
    /// raycast reads the array, flagged or not).
    _rcPrepare() {
        const m = this._m;
        if (!m || m.kind === 'group') return;
        const n = Math.max(0, Math.min(this.count, this.instanceMatrix.count));
        if (!_sameNumbers(m.instances, this.instanceMatrix.array, n * 16)) _mirrorPush(this, _D_DATA);
    }
}

/// three.js r165 `BatchedMesh` (the r165 API: `addGeometry` gives the id each
/// draw is addressed by). Each geometry is drawn by an internal mesh node
/// under this one — not one of its `children` — placed by its matrix.
export class BatchedMesh extends Mesh {
    constructor(maxInstanceCount, _maxVertexCount, _maxIndexCount, material) {
        if (_maxIndexCount && typeof _maxIndexCount !== 'number') material = _maxIndexCount;
        super(new BufferGeometry(), material);
        this.isBatchedMesh = true;
        this.type = 'BatchedMesh';
        this._maxInstanceCount = maxInstanceCount;
        this._slots = [];
    }
    addGeometry(geometry) {
        const id = this._slots.length;
        const mesh = new Mesh(geometry, this.material);
        mesh.matrixAutoUpdate = false;
        mesh.parent = this;
        this._slots.push(mesh);
        this._dirty(_D_CHILDREN);
        return id;
    }
    setMatrixAt(id, matrix) {
        const s = this._slots[id];
        if (!s) return this;
        s.matrix.copy(matrix);
        s._dirty(_D_TRS);
        return this;
    }
    getMatrixAt(id, matrix) { const s = this._slots[id]; return s ? matrix.copy(s.matrix) : null; }
    setVisibleAt(id, value) { const s = this._slots[id]; if (s) s.visible = value; return this; }
    getVisibleAt(id) { return this._slots[id]?.visible ?? false; }
    set material(m) { this._material = m; for (const s of this._slots ?? []) s.material = m; this._dirty(_D_MATERIAL); }
    get material() { return this._material; }
    _mirrorExtraChildren() { return this._slots; }
    _mirrorKind() { return 'group'; }
    _mirrorCreate(w) { return w.createGroup(false); }
    /// Each visible draw, hits reported on this mesh with their `batchId`.
    raycast(raycaster, intersects) {
        this._slots.forEach((s, batchId) => {
            if (!s.visible) return;
            // three.js casts each draw at `matrixWorld · instanceMatrix`, from
            // this mesh's matrixWorld as it stands.
            s.matrixWorld.multiplyMatrices(this.matrixWorld, s.matrix);
            _rcCast(raycaster, s, intersects, false, (h) => { h.object = this; h.batchId = batchId; });
        });
    }
}

export class BufferAttribute {
    constructor(array, itemSize) {
        // wasm-bindgen Vec<f32> expects a regular Float32Array (or Array).
        const arr = (array instanceof Float32Array) ? array : new Float32Array(array);
        this._w = new WebBufferAttribute(arr, itemSize);
        this.array = arr;
        this.itemSize = itemSize;
        this.count = arr.length / itemSize | 0;
        this.normalized = false;
        this.version = 0;
        /// What uses this attribute (geometries, an InstancedMesh), told when
        /// it is flagged `needsUpdate`.
        this._owners = new Set();
    }
    /// three.js: flag edits made in place to `array` — they reach the wasm
    /// objects drawn with it at the next sync.
    set needsUpdate(value) {
        if (value !== true) return;
        this.version++;
        for (const owner of this._owners) owner._attributeChanged?.(this);
    }
    getX(index) { return this.array[index * this.itemSize]; }
    getY(index) { return this.array[index * this.itemSize + 1]; }
    getZ(index) { return this.array[index * this.itemSize + 2]; }
    getW(index) { return this.array[index * this.itemSize + 3]; }
    fromBufferAttribute(attr, index) {
        const i = index * attr.itemSize;
        for (let k = 0; k < attr.itemSize; k++) this.array[k] = attr.array[i + k];
        return this;
    }
}
/// three.js `geometry.drawRange`, telling its geometry when it changes.
class _DrawRange {
    constructor(geometry) { this._g = geometry; this._start = 0; this._count = Infinity; }
    get start() { return this._start; }
    set start(v) { if (v !== this._start) { this._start = v; this._g._bumpRange(); } }
    get count() { return this._count; }
    set count(v) { if (v !== this._count) { this._count = v; this._g._bumpRange(); } }
    toJSON() { return { start: this._start, count: this._count }; }
}
export class BufferGeometry {
    constructor() {
        this.isBufferGeometry = true;
        this._isUserGeometry = true;
        this.uuid = MathUtils.generateUUID();
        this.name = '';
        this.type = 'BufferGeometry';
        this.attributes = {};
        /// Bumped by every change the wasm build depends on; the meshes, lines
        /// and points drawn with this geometry re-take it at the next sync.
        this._version = 0;
        this._rangeVersion = 0;
        this._drawRange = new _DrawRange(this);
        this._groups = [];
        this._morphAttributes = {};
        this._morphTargetsRelative = false;
        this._boundsTree = null;
        this.boundingBox = null;
        this.boundingSphere = null;
        this.userData = {};
    }
    get drawRange() { return this._drawRange; }
    set drawRange(r) { this._drawRange.start = r?.start ?? 0; this._drawRange.count = r?.count ?? Infinity; }
    get groups() { return this._groups; }
    set groups(g) { this._groups = g ?? []; this._bump(); }
    get morphAttributes() { return this._morphAttributes; }
    set morphAttributes(m) { this._morphAttributes = m ?? {}; this._bump(); }
    get morphTargetsRelative() { return this._morphTargetsRelative; }
    set morphTargetsRelative(v) { v = !!v; if (v !== this._morphTargetsRelative) { this._morphTargetsRelative = v; this._bump(); } }
    /// three-mesh-bvh's `boundsTree`: the Rust raycaster walks it.
    get boundsTree() { return this._boundsTree; }
    set boundsTree(t) { if (t !== this._boundsTree) { this._boundsTree = t ?? null; this._bump(); } }
    /// A change the wasm build depends on.
    _bump() {
        this._version++;
        _notifyUsers(this, _D_GEOMETRY, (o) => o.geometry === this);
    }
    _bumpRange() {
        this._rangeVersion++;
        _notifyUsers(this, _D_GEOMETRY, (o) => o.geometry === this);
    }
    /// An attribute was flagged `needsUpdate`.
    _attributeChanged() { this._bump(); }
    /// The wasm geometry nodes are drawn with: attributes, index, groups,
    /// polygons, draw range, morph attributes and BVH. Rebuilt only when
    /// something it reads changed; a draw-range change alone re-ranges the
    /// existing build. Null without positions.
    _wasmGeometry() {
        if (!this.attributes.position?.array?.length) return null;
        if (this._geom && this._geomVersion === this._version) {
            if (this._geomRange !== this._rangeVersion) {
                this._geom.setDrawRange(this._drawRange.start, this._drawRange.count);
                this._geomRange = this._rangeVersion;
            }
            return this._geom;
        }
        const g = WebGeometry.shareBufferGeometry(this._syncWasmFromJs());
        const dr = this._drawRange;
        if (dr.start !== 0 || dr.count !== Infinity) g.setDrawRange(dr.start, dr.count);
        for (const name of ['position', 'normal', 'color']) {
            const list = this._morphAttributes?.[name];
            if (!list?.length) continue;
            const packed = list.map(_attrPacked);
            const n = packed[0].length;
            const flat = new Float32Array(n * list.length);
            packed.forEach((a, t) => flat.set(a.length === n ? a : new Float32Array(n), t * n));
            g.setMorphAttributeList(name, flat, list.length, list[0].itemSize || 3);
        }
        if (this._morphTargetsRelative) g.setMorphTargetsRelative(true);
        if (this._boundsTree?._w && typeof g.setBoundsTree === 'function') g.setBoundsTree(this._boundsTree._w);
        this._geom = g;
        this._geomVersion = this._version;
        this._geomRange = this._rangeVersion;
        return g;
    }
    _syncWasmFromJs() {
        this._w = new WebBufferGeometry();
        const uploaded = new Set();
        for (const [name, attr] of Object.entries(this.attributes)) {
            // Interleaved: this attribute's components out of the shared buffer.
            const a = _attrPacked(attr);
            if (!a?.length) continue;
            if (attr.isInterleavedBufferAttribute) uploaded.add(attr.data);
            const arr = a instanceof Float32Array ? a : new Float32Array(a);
            this._w.setAttribute(name, new WebBufferAttribute(arr, attr.itemSize || 3));
            // three.js's skinning attributes, under the names the Rust skinning reads.
            if (name === 'skinIndex') this._w.setAttribute('joint', new WebBufferAttribute(arr, attr.itemSize || 4));
            if (name === 'skinWeight') this._w.setAttribute('weight', new WebBufferAttribute(arr, attr.itemSize || 4));
        }
        // The whole buffer went over, so no range is still pending (three.js
        // clears them after its upload too).
        for (const data of uploaded) data.clearUpdateRanges?.();
        const idx = this._indexAttr?.array || this.index?.array;
        if (idx?.length) {
            this._w.setIndex(idx instanceof Uint32Array ? idx : new Uint32Array(idx));
        }
        if (this.groups?.length && this._w.setGroups) {
            this._w.setGroups(new Uint32Array(this.groups.flatMap((g) => [g.start, g.count, g.materialIndex ?? 0])));
        }
        // After the index: wasm drops the grouping on `setIndex`, as here.
        if (this._polygons?.length && this._w.setPolygons) {
            try {
                this._w.setPolygons(this._polygons);
            } catch (e) {
                // Stale against the current triangles (a draw-range or an
                // array edited in place): draw as plain triangles, once warned.
                if (!this._warnedPolygons) {
                    console.warn(`BufferGeometry: ignoring polygons — ${e?.message ?? e}`);
                    this._warnedPolygons = true;
                }
            }
        }
        delete this._geom;
        return this._w;
    }
    setAttribute(name, attr) {
        // A new vertex count is new topology for unindexed geometry, so the
        // polygon grouping goes — the same rule as the Rust BufferGeometry.
        if (name === 'position' && this._polygons) {
            const count = (a) => a?.isInterleavedBufferAttribute ? a.count : a?.array ? a.array.length / (a.itemSize || 3) : undefined;
            if (count(this.attributes.position) !== count(attr)) this._polygons = null;
        }
        this.attributes[name] = attr;
        attr?._owners?.add(this);
        this._bump();
        return this;
    }
    getAttribute(name) { return this.attributes[name]; }
    hasAttribute(name) { return this.attributes[name] !== undefined; }
    /// threers extension: group triangles into polygons. `counts[i]` is how
    /// many consecutive triangles (in index order) polygon `i` became — `2`
    /// for a quad. Wireframes (`wireframe: true`, `WireframeGeometry`) then
    /// outline polygons instead of drawing fan diagonals, and `EdgesGeometry`
    /// never emits an edge inside one. Call after `setIndex` and the last
    /// `position` change, both of which drop it. Pass null to remove.
    /// Throws unless every count is ≥ 1 and they sum to the triangle count.
    setPolygons(counts) {
        if (counts == null) { this._polygons = null; return this; }
        const u32 = counts instanceof Uint32Array ? counts : Uint32Array.from(counts);
        const tris = this._triangleCount();
        let sum = 0;
        for (let i = 0; i < u32.length; i++) {
            if (u32[i] === 0) throw new Error(`setPolygons: polygon ${i} has zero triangles`);
            sum += u32[i];
        }
        if (sum !== tris) {
            throw new Error(`setPolygons: counts sum to ${sum} triangles but the geometry draws ${tris}`);
        }
        this._polygons = u32;
        this._warnedPolygons = false;
        return this;
    }
    /// The polygon grouping (`Uint32Array` of triangles per polygon), or null.
    get polygons() { return this._polygons ?? null; }
    /// threers extension: pair adjacent triangles back into quads (Blender's
    /// "Tris to Quads") — for glTF/STL imports, which carry triangles only.
    /// Reorders the index so each quad's two triangles are consecutive and
    /// sets `polygons`; vertices are untouched. Angles in radians (40° each by
    /// default): the most two triangles may fold, and the most any quad
    /// corner may be off 90°. Returns how many quads were formed.
    inferQuads(maxFaceAngle = 40 * Math.PI / 180, maxShapeAngle = maxFaceAngle) {
        const w = this._syncWasmFromJs();
        if (typeof w?.inferQuads !== 'function') return 0;
        const quads = w.inferQuads(maxFaceAngle, maxShapeAngle);
        if (quads > 0) {
            this.setIndex(new Uint32Array(w.getIndexArray()));
            this._polygons = new Uint32Array(w.polygonCounts());
            this._warnedPolygons = false;
        }
        return quads;
    }
    /// threers extension: Catmull–Clark subdivision (three.js shipped this as
    /// the `SubdivisionModifier` addon, Loop and triangles only). `levels`
    /// rounds, 0–6, each multiplying the face count by about 4. Returns a new
    /// indexed BufferGeometry whose faces are all quads, carried as `polygons`
    /// so wireframes, exporters and `polygonIndex` see them, with `position`
    /// and smooth `normal` only (UVs and other attributes are dropped). The
    /// cage is this geometry's polygons — its triangles where it has none —
    /// welded by position. `options.sharpBoundary` (default true) holds open
    /// borders in place. This geometry is untouched.
    subdivide(levels = 1, options = {}) {
        const w = this._syncWasmFromJs();
        if (typeof w?.subdivide !== 'function') {
            throw new Error('BufferGeometry.subdivide: this wasm build has no subdivision');
        }
        const sub = w.subdivide(levels, options?.sharpBoundary ?? true);
        const out = new BufferGeometry();
        for (const name of ['position', 'normal']) {
            const arr = sub.getAttributeArray(name);
            if (arr?.length) {
                out.setAttribute(name, new BufferAttribute(new Float32Array(arr), sub.getAttributeItemSize(name) || 3));
            }
        }
        const index = sub.getIndexArray();
        if (index?.length) out.setIndex(new Uint32Array(index));
        const counts = sub.polygonCounts();
        if (counts?.length) out._polygons = new Uint32Array(counts);
        sub.free?.();
        return out;
    }
    _triangleCount() {
        const idx = this._indexAttr?.array;
        if (idx) return Math.floor(idx.length / 3);
        const p = this.attributes.position;
        return p?.array ? Math.floor(p.array.length / (p.itemSize || 3) / 3) : 0;
    }
    setIndex(index) {
        this._polygons = null;
        let u32;
        let indexAttr;
        if (index?.array) {
            const arr = index.array;
            u32 = arr instanceof Uint32Array ? arr : new Uint32Array(arr);
            indexAttr = index;
        } else {
            u32 = index instanceof Uint32Array ? index : new Uint32Array(index);
            indexAttr = new BufferAttribute(u32, 1);
        }
        this._indexAttr = indexAttr;
        indexAttr?._owners?.add(this);
        this._bump();
        return this;
    }
    setDrawRange(start, count) {
        this._drawRange.start = start;
        this._drawRange.count = count;
        return this;
    }
    deleteAttribute(name) {
        delete this.attributes[name];
        this._bump();
        return this;
    }
    /// three.js r165, over the Rust build of this geometry (positions and
    /// position morph targets), in f64 as three.js computes it.
    computeBoundingBox() {
        if (this.boundingBox === null) this.boundingBox = new Box3();
        _computeGeometryBox(this, this.boundingBox);
    }
    computeBoundingSphere() {
        if (this.boundingSphere === null) this.boundingSphere = new Sphere();
        _computeGeometrySphere(this, this.boundingSphere);
    }
    dispose() {
        this.boundsTree = null;
        this.halfEdges = null;
        this.groupIndices = null;
        return this;
    }
    translate(x, y, z) {
        const p = this.attributes.position;
        if (!p) return this;
        if (this._hasInterleaved()) return this._applyMatrixInPlace(new Matrix4().makeTranslation(x, y, z));
        const a = p.array;
        for (let i = 0; i < a.length; i += 3) {
            a[i] += x; a[i+1] += y; a[i+2] += z;
        }
        // Push the mutated array back into the wasm BufferAttribute by re-setting
        // the attribute (the wasm side took a copy at original construction).
        this.setAttribute('position', new BufferAttribute(a, p.itemSize));
        return this;
    }
    addGroup(start, count, materialIndex = 0) {
        this._groups.push({ start, count, materialIndex });
        this._bump();
        return this;
    }
    clearGroups() { this.groups = []; return this; }
    scale(x, y, z) {
        const p = this.attributes.position;
        if (!p) return this;
        if (this._hasInterleaved()) return this._applyMatrixInPlace(new Matrix4().makeScale(x, y, z));
        const a = p.array;
        for (let i = 0; i < a.length; i += 3) {
            a[i] *= x; a[i+1] *= y; a[i+2] *= z;
        }
        this.setAttribute('position', new BufferAttribute(a, p.itemSize));
        return this;
    }
    rotateX(angle) { return this._applyAxis(angle, 0); }
    rotateY(angle) { return this._applyAxis(angle, 1); }
    rotateZ(angle) { return this._applyAxis(angle, 2); }
    _applyAxis(angle, axis) {
        const p = this.attributes.position;
        if (!p) return this;
        if (this._hasInterleaved()) {
            const m = new Matrix4();
            return this._applyMatrixInPlace(axis === 0 ? m.makeRotationX(angle) : axis === 1 ? m.makeRotationY(angle) : m.makeRotationZ(angle));
        }
        const a = p.array;
        const c = Math.cos(angle), s = Math.sin(angle);
        for (let i = 0; i < a.length; i += 3) {
            const x = a[i], y = a[i+1], z = a[i+2];
            if (axis === 0)      { a[i+1] = y*c - z*s; a[i+2] = y*s + z*c; }
            else if (axis === 1) { a[i]   = x*c + z*s; a[i+2] = -x*s + z*c; }
            else                 { a[i]   = x*c - y*s; a[i+1] = x*s + y*c; }
        }
        this.setAttribute('position', new BufferAttribute(a, p.itemSize));
        return this;
    }
    computeVertexNormals() {
        this._syncWasmFromJs();
        this._w.computeVertexNormals();
        const normal = this._w.getAttributeArray?.('normal');
        if (normal?.length) {
            this.setAttribute('normal', new BufferAttribute(new Float32Array(normal), 3));
        }
        return this;
    }
    /// Whether position, normal or tangent is an `InterleavedBufferAttribute`,
    /// which the transforms below cannot treat as a packed array.
    _hasInterleaved() {
        const at = this.attributes;
        return !!(at.position?.isInterleavedBufferAttribute || at.normal?.isInterleavedBufferAttribute
            || at.tangent?.isInterleavedBufferAttribute);
    }
    /// three.js r165 `BufferGeometry.applyMatrix4`, in place on the
    /// attributes whatever their storage: position by the matrix, normal by
    /// its normal matrix, tangent as a direction; bounds recomputed if set.
    _applyMatrixInPlace(matrix) {
        const { position, normal, tangent } = this.attributes;
        // Per vertex, through the attribute's own accessors where it is
        // interleaved, straight on the packed array where it is not.
        const each = (attr, f) => {
            if (attr === undefined) return;
            const v = new Vector3();
            for (let i = 0, l = attr.count; i < l; i++) {
                v.set(attr.getX(i), attr.getY(i), attr.getZ(i));
                f(v);
                if (attr.isInterleavedBufferAttribute) attr.setXYZ(i, v.x, v.y, v.z);
                else { const o = i * attr.itemSize; attr.array[o] = v.x; attr.array[o + 1] = v.y; attr.array[o + 2] = v.z; }
            }
            attr.needsUpdate = true;
        };
        const nm = new Matrix3().getNormalMatrix(matrix);
        each(position, (v) => v.applyMatrix4(matrix));
        each(normal, (v) => v.applyNormalMatrix(nm));
        each(tangent, (v) => v.transformDirection(matrix));
        this._bump();
        if (this.boundingBox !== null) this.computeBoundingBox();
        if (this.boundingSphere !== null) this.computeBoundingSphere();
        return this;
    }
    applyMatrix4(m) {
        const p = this.attributes.position;
        if (!p) return this;
        if (this._hasInterleaved()) return this._applyMatrixInPlace(m);
        const a = p.array;
        const e = m.elements || m._w?.elements?.() || m;
        for (let i = 0; i < a.length; i += 3) {
            const x = a[i], y = a[i + 1], z = a[i + 2];
            a[i]     = e[0] * x + e[4] * y + e[8]  * z + e[12];
            a[i + 1] = e[1] * x + e[5] * y + e[9]  * z + e[13];
            a[i + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
        }
        this.setAttribute('position', new BufferAttribute(a, p.itemSize));
        const n = this.attributes.normal;
        if (n) {
            // For normals, multiply by the upper 3x3 of the matrix (ignoring
            // translation) and renormalize. For non-uniform scale this should
            // be the inverse-transpose, but for the common rotate+uniform-scale
            // case this matches three.js's behavior in BufferGeometry.applyMatrix4.
            const na = n.array;
            for (let i = 0; i < na.length; i += 3) {
                const x = na[i], y = na[i + 1], z = na[i + 2];
                let nx = e[0] * x + e[4] * y + e[8]  * z;
                let ny = e[1] * x + e[5] * y + e[9]  * z;
                let nz = e[2] * x + e[6] * y + e[10] * z;
                const len = Math.hypot(nx, ny, nz) || 1;
                na[i] = nx / len; na[i + 1] = ny / len; na[i + 2] = nz / len;
            }
            this.setAttribute('normal', new BufferAttribute(na, n.itemSize));
        }
        return this;
    }
    applyQuaternion(q) {
        const m = new Matrix4().makeRotationFromQuaternion(q);
        return this.applyMatrix4(m);
    }
    center() {
        const p = this.attributes.position;
        if (!p) return this;
        if (p.isInterleavedBufferAttribute) {
            this.computeBoundingBox();
            const b = this.boundingBox;
            return this.translate(-(b.min.x + b.max.x) * 0.5, -(b.min.y + b.max.y) * 0.5, -(b.min.z + b.max.z) * 0.5);
        }
        let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
        const a = p.array;
        for (let i = 0; i < a.length; i += 3) {
            if (a[i]     < mnx) mnx = a[i];     if (a[i]     > mxx) mxx = a[i];
            if (a[i + 1] < mny) mny = a[i + 1]; if (a[i + 1] > mxy) mxy = a[i + 1];
            if (a[i + 2] < mnz) mnz = a[i + 2]; if (a[i + 2] > mxz) mxz = a[i + 2];
        }
        return this.translate(-(mnx + mxx) * 0.5, -(mny + mxy) * 0.5, -(mnz + mxz) * 0.5);
    }
    /// three.js r165 `toNonIndexed`: every attribute and morph attribute
    /// expanded per index, `morphTargetsRelative` and the groups kept (they
    /// count indices, which are now vertices).
    toNonIndexed() {
        const idx = this.index?.array;
        if (!idx) return this;
        const expand = (src) => {
            const a = _attrPacked(src);
            const itemSize = src.itemSize;
            const dst = new a.constructor(idx.length * itemSize);
            for (let i = 0; i < idx.length; i++) {
                const j = idx[i] * itemSize;
                for (let k = 0; k < itemSize; k++) dst[i * itemSize + k] = a[j + k];
            }
            return new BufferAttribute(dst, itemSize, src.normalized === true);
        };
        const out = new BufferGeometry();
        for (const name of Object.keys(this.attributes)) out.setAttribute(name, expand(this.attributes[name]));
        for (const name of Object.keys(this.morphAttributes ?? {})) {
            out.morphAttributes[name] = this.morphAttributes[name].map(expand);
        }
        out.morphTargetsRelative = this.morphTargetsRelative;
        for (const g of this.groups ?? []) out.addGroup(g.start, g.count, g.materialIndex);
        // Unindexing keeps triangle order, so the grouping still holds.
        if (this._polygons) out._polygons = new Uint32Array(this._polygons);
        return out;
    }
    toJSON() {
        const out = { metadata: { version: 4.6, type: 'BufferGeometry', generator: 'threers' }, type: 'BufferGeometry', uuid: this.uuid || MathUtils.generateUUID() };
        const data = { attributes: {} };
        for (const name of Object.keys(this.attributes)) {
            const a = this.attributes[name];
            // Interleaved: de-interleaved, as three.js's `toJSON()` without
            // a shared `data` writes it.
            data.attributes[name] = a.isInterleavedBufferAttribute ? a.toJSON()
                : { itemSize: a.itemSize, type: a.array.constructor.name, array: Array.from(a.array), normalized: !!a.normalized };
        }
        if (this.index) {
            data.index = { type: this.index.array.constructor.name, array: Array.from(this.index.array) };
        }
        out.data = data;
        return out;
    }
    get index() {
        // Surface a three.js-shaped { array } accessor so user code that reads
        // geometry.index.array works. The wasm geometry stores indices but we
        // also mirror them on the JS attribute when set via `setIndex`.
        return this._indexAttr || null;
    }
    clone() { return new BufferGeometry().copy(this); }
    /// three.js `copy`: attributes, index, groups, draw range, morph
    /// attributes (and the polygon grouping).
    copy(other) {
        for (const name of Object.keys(other.attributes)) {
            const src = other.attributes[name];
            // An interleaved attribute is cloned de-interleaved (three.js's
            // `clone()` with no shared `data`).
            this.setAttribute(name, src.isInterleavedBufferAttribute ? src.clone()
                : new BufferAttribute(new src.array.constructor(src.array), src.itemSize));
        }
        if (other.index) this.setIndex(new Uint32Array(other.index.array));
        this.groups = (other.groups ?? []).map((g) => ({ ...g }));
        if (other.drawRange) this.setDrawRange(other.drawRange.start, other.drawRange.count);
        const morph = {};
        for (const [name, list] of Object.entries(other.morphAttributes ?? {})) {
            morph[name] = list.map((a) => a.isInterleavedBufferAttribute ? a.clone() : new BufferAttribute(new a.array.constructor(a.array), a.itemSize));
        }
        this.morphAttributes = morph;
        this.morphTargetsRelative = !!other.morphTargetsRelative;
        this._polygons = other._polygons ? new Uint32Array(other._polygons) : null;
        return this;
    }
    // Convert to the WebGeometry that Mesh/etc expect via `geom._w`. Made
    // available as a plain method (not just a getter) so subclasses that
    // copy via Object.assign still find it.
    get _asGeometry() {
        this._syncWasmFromJs();
        this._geom = WebGeometry.fromBufferGeometry(this._w);
        return this._geom;
    }
}

/// threers extension (not in three.js): a quad from four arbitrary corners in
/// winding order, meshed as a bilinear patch — `PlaneGeometry` for corners
/// that need not form an axis-aligned rectangle. `corners` is four `Vector3`s
/// (or `[x, y, z]` arrays), or twelve numbers. Corners 0..3 get UVs (0,0),
/// (1,0), (1,1), (0,1); passing a plane's corners counter-clockwise from
/// bottom-left reproduces `PlaneGeometry`. Each grid cell is one polygon, so a
/// wireframe draws the cell outlines and not their diagonals.
export class QuadGeometry extends BufferGeometry {
    constructor(corners = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]],
                widthSegments = 1, heightSegments = 1) {
        super();
        const flat = [];
        for (const c of corners) {
            if (typeof c === 'number') flat.push(c);
            else if (Array.isArray(c) || ArrayBuffer.isView(c)) flat.push(c[0], c[1], c[2]);
            else flat.push(c.x, c.y, c.z);
        }
        const ws = Math.max(1, Math.floor(widthSegments));
        const hs = Math.max(1, Math.floor(heightSegments));
        const bg = WebGeometry.quad(new Float32Array(flat), ws, hs).toBufferGeometry();
        for (const name of ['position', 'normal', 'uv']) {
            const a = bg.getAttributeArray(name);
            if (a) this.setAttribute(name, new BufferAttribute(new Float32Array(a), bg.getAttributeItemSize(name)));
        }
        const index = bg.getIndexArray();
        if (index) this.setIndex(new Uint32Array(index));
        const counts = bg.polygonCounts();
        if (counts) this.setPolygons(counts);
        this.type = 'QuadGeometry';
        this.parameters = { corners: flat, widthSegments: ws, heightSegments: hs };
    }
}

/// Copy a geometry the Rust side generated into the JS BufferGeometry
/// `target` — every attribute, the index, `groups` and the polygon grouping —
/// so a generator class can be a thin `extends BufferGeometry` wrapper whose
/// buffers three.js code can still read. Takes a `WebGeometry` or a
/// `WebBufferGeometry`; frees the handle it was given. Returns `target`.
function _fillFromWasm(target, wasmGeometry) {
    const bg = typeof wasmGeometry.toBufferGeometry === 'function'
        ? wasmGeometry.toBufferGeometry() : wasmGeometry;
    for (const name of bg.attributeNames()) {
        const a = bg.getAttributeArray(name);
        if (a) target.setAttribute(name, new BufferAttribute(new Float32Array(a), bg.getAttributeItemSize(name)));
    }
    const index = bg.getIndexArray();
    if (index?.length) target.setIndex(new Uint32Array(index));
    const groups = bg.groupsArray();
    target.groups = [];
    for (let i = 0; i < groups.length; i += 3) target.addGroup(groups[i], groups[i + 1], groups[i + 2]);
    const counts = bg.polygonCounts();
    if (counts?.length) target._polygons = new Uint32Array(counts);
    if (bg !== wasmGeometry) bg.free?.();
    wasmGeometry.free?.();
    return target;
}

/** Convert built-in wasm geometry or BufferGeometry to non-indexed BufferGeometry (for CSG). */
export function geometryToBufferGeometry(geometry) {
    if (geometry?.attributes?.position) {
        return geometry.index ? geometry.toNonIndexed() : geometry;
    }
    if (geometry?._w?.toBufferGeometry) {
        const wasm = geometry._w.toBufferGeometry();
        const out = new BufferGeometry();
        const pos = wasm.getAttributeArray('position');
        if (pos) {
            const itemSize = wasm.getAttributeItemSize('position') || 3;
            out.setAttribute('position', new BufferAttribute(new Float32Array(pos), itemSize));
        }
        const normal = wasm.getAttributeArray('normal');
        if (normal) {
            const itemSize = wasm.getAttributeItemSize('normal') || 3;
            out.setAttribute('normal', new BufferAttribute(new Float32Array(normal), itemSize));
        }
        const uv = wasm.getAttributeArray('uv');
        if (uv) {
            const itemSize = wasm.getAttributeItemSize('uv') || 2;
            out.setAttribute('uv', new BufferAttribute(new Float32Array(uv), itemSize));
        }
        const index = wasm.getIndexArray?.();
        if (index?.length) out.setIndex(new Uint32Array(index));
        return out.toNonIndexed();
    }
    throw new Error('geometryToBufferGeometry: unsupported geometry');
}

function _wrapWebTexture(wtex) {
    const t = Object.create(Texture.prototype);
    t._w = wtex;
    t.image = { width: wtex.width(), height: wtex.height() };
    t.magFilter = 1006; t.minFilter = 1008;
    t.wrapS = 1001; t.wrapT = 1001;
    // EXR / equirect sky → six faces (earth-demo starfield cube).
    if (typeof wtex.cubeFaces === 'function') {
        t.cubeFaces = (faceSize = 0) => {
            const raw = wtex.cubeFaces(faceSize);
            const out = [];
            for (let i = 0; i < raw.length; i++) out.push(_wrapWebTexture(raw[i]));
            return out;
        };
    }
    return t;
}

function _wrapLoaderGeometry(wgeom) {
    return { _w: wgeom, geometry: wgeom };
}

/// three.js r165 `Line`: a vertex strip (`LineSegments` and `LineLoop` below).
/// The node draws the strip expanded into segments in Rust, and the
/// raycaster walks the strip, so a hit's `index` is the strip vertex.
export class Line extends Object3D {
    constructor(geometry = new BufferGeometry(), material = new LineBasicMaterial()) {
        super();
        this.isLine = true;
        this.type = 'Line';
        this._geometry = geometry;
        this._material = material;
    }
    get geometry() { return this._geometry; }
    set geometry(g) { if (g !== this._geometry) { this._geometry = g; this._dirty(_D_GEOMETRY); } }
    get material() { return this._material; }
    set material(m) { this._material = m; this._dirty(_D_MATERIAL); }
    /// three.js: the distance along the strip to each vertex, as `lineDistance`.
    computeLineDistances() {
        const geometry = this.geometry;
        const pos = geometry.attributes?.position;
        if (!pos) return this;
        const a = _attrPacked(pos), n = pos.count;
        const d = new Float32Array(n);
        for (let i = 1; i < n; i++) {
            d[i] = d[i - 1] + Math.hypot(a[i * 3] - a[i * 3 - 3], a[i * 3 + 1] - a[i * 3 - 2], a[i * 3 + 2] - a[i * 3 - 1]);
        }
        _setLineDistances(this, d);
        return this;
    }
    copy(source, recursive) {
        super.copy(source, recursive);
        this.material = Array.isArray(source.material) ? source.material.slice() : source.material;
        this.geometry = source.geometry;
        return this;
    }
    raycast(raycaster, intersects) { _rcCast(raycaster, this, intersects); }
    /// 0 segments, 1 strip, 2 loop (`WebScene.createLine`).
    _lineMode() { return 1; }
    _mirrorKind() { return _wasmGeometryOf(this.geometry) ? `line:${this._lineMode()}` : 'group'; }
    _mirrorCreate(w) {
        const g = _wasmGeometryOf(this.geometry);
        if (!g) return w.createGroup(false);
        return w.createLine(g, _wasmMaterialOf(this.material), this._lineMode());
    }
    _mirrorContent(w, h, flags, m) {
        if (m.kind === 'group') return;
        if (flags & _D_GEOMETRY) {
            const key = _geometryKey(this.geometry);
            const g = _wasmGeometryOf(this.geometry);
            if (g && m.geometry !== key) { w.setNodeGeometry(h, g, this._lineMode()); m.geometry = key; }
            _useBy(this.geometry, this);
        }
        if (flags & _D_MATERIAL) _mirrorMaterials(w, h, this.material, this);
    }
}
/// `lineDistance` on a line's geometry, and — for the dashed material, which
/// reads it from there — as the first uv coordinate.
function _setLineDistances(line, d) {
    const geom = line.geometry;
    geom.setAttribute('lineDistance', new BufferAttribute(d, 1));
    if (line.material?._isLineDashed) {
        const uvs = new Float32Array(d.length * 2);
        for (let i = 0; i < d.length; i++) uvs[i * 2] = d[i];
        geom.setAttribute('uv', new BufferAttribute(uvs, 2));
    }
}
/// three.js `LineSegments`: vertex pairs.
export class LineSegments extends Line {
    constructor(geometry, material) {
        super(geometry, material);
        this.isLineSegments = true;
        this.type = 'LineSegments';
    }
    /// three.js: each segment's distances start from the previous segment's end.
    computeLineDistances() {
        const pos = this.geometry.attributes?.position;
        if (!pos) return this;
        const a = _attrPacked(pos), n = pos.count;
        const d = new Float32Array(n);
        for (let i = 0; i + 1 < n; i += 2) {
            d[i] = i === 0 ? 0 : d[i - 1];
            d[i + 1] = d[i] + Math.hypot(a[i * 3 + 3] - a[i * 3], a[i * 3 + 4] - a[i * 3 + 1], a[i * 3 + 5] - a[i * 3 + 2]);
        }
        _setLineDistances(this, d);
        return this;
    }
    _lineMode() { return 0; }
}
/// three.js `LineLoop`: a closed strip.
export class LineLoop extends Line {
    constructor(geometry, material) {
        super(geometry, material);
        this.isLineLoop = true;
        this.type = 'LineLoop';
    }
    _lineMode() { return 2; }
}
/// three.js `Points`: each vertex a point.
export class Points extends Object3D {
    constructor(geometry = new BufferGeometry(), material = new PointsMaterial()) {
        super();
        this.isPoints = true;
        this.type = 'Points';
        this._geometry = geometry;
        this._material = material;
    }
    get geometry() { return this._geometry; }
    set geometry(g) { if (g !== this._geometry) { this._geometry = g; this._dirty(_D_GEOMETRY); } }
    get material() { return this._material; }
    set material(m) { this._material = m; this._dirty(_D_MATERIAL); }
    copy(source, recursive) {
        super.copy(source, recursive);
        this.material = Array.isArray(source.material) ? source.material.slice() : source.material;
        this.geometry = source.geometry;
        return this;
    }
    raycast(raycaster, intersects) { _rcCast(raycaster, this, intersects); }
    _mirrorKind() { return _wasmGeometryOf(this.geometry) ? 'points' : 'group'; }
    _mirrorCreate(w) {
        const g = _wasmGeometryOf(this.geometry);
        if (!g) return w.createGroup(false);
        return w.createPoints(g, _wasmMaterialOf(this.material));
    }
    _mirrorContent(w, h, flags, m) {
        if (m.kind === 'group') return;
        if (flags & _D_GEOMETRY) {
            const key = _geometryKey(this.geometry);
            const g = _wasmGeometryOf(this.geometry);
            if (g && m.geometry !== key) { w.setNodeGeometry(h, g, undefined); m.geometry = key; }
            _useBy(this.geometry, this);
        }
        if (flags & _D_MATERIAL) _mirrorMaterials(w, h, this.material, this);
    }
}

// ---- Lights ----
/// A light's shadow: three.js's `light.shadow` — `camera` (the shadow
/// frustum), `mapSize`, `bias`, `normalBias`, `radius` — kept on the wasm
/// light, which the node is re-sent with whenever the frustum changes (and,
/// for `mapSize`, `bias`, `normalBias` and `radius`, polled in `_lightState`).
function _makeLightShadow(light) {
    const store = { left: -5, right: 5, top: 5, bottom: -5, near: 0.1, far: 500 };
    const sync = () => {
        light?._w?.setShadowCamera?.(store.left, store.right, store.top, store.bottom, store.near, store.far);
        light?._dirty?.(_D_DATA);
    };
    const camera = {};
    for (const key of ['left', 'right', 'top', 'bottom', 'near', 'far']) {
        Object.defineProperty(camera, key, {
            get() { return store[key]; },
            set(v) { store[key] = v; sync(); },
            enumerable: true,
            configurable: true,
        });
    }
    const mapSize = { width: 512, height: 512, set(w, h) { this.width = w; this.height = h; } };
    return { camera, mapSize, bias: 0, normalBias: 0, radius: 1, _sync: sync };
}
/// three.js r165 `Light`: `color` and `intensity` on an Object3D. The wasm
/// light (`_w`) is the node's light; its parameters are compared at each sync.
export class Light extends Object3D {
    constructor(color, intensity = 1) {
        super();
        this.isLight = true;
        this._isLight = true;
        this.type = 'Light';
        this.color = new Color(color);
        this._intensity = intensity;
    }
    get intensity() { return this._intensity; }
    set intensity(v) { this._intensity = Number(v); this._dirty(_D_DATA); }
    get castShadow() { return this._castShadow; }
    set castShadow(v) {
        v = !!v;
        if (v === this._castShadow) return;
        this._castShadow = v;
        this._w?.setCastShadow?.(v);
        this.shadow?._sync?.();
        this._dirty(_D_DATA | _D_DRAW);
    }
    dispose() {}
    copy(source, recursive) {
        super.copy(source, recursive);
        this.color.copy(source.color);
        this.intensity = source.intensity;
        return this;
    }
    /// The light's shining direction, for `setDirection`, in the light's own
    /// frame (the renderers carry it by the node's world rotation): toward
    /// `target`'s world position. Null for lights without a target.
    _lightDirection() {
        if (!this.target) return null;
        this.updateWorldMatrix(true, false);
        this.target.updateWorldMatrix(true, false);
        const e = this.matrixWorld.elements, t = this.target.matrixWorld.elements;
        const dx = t[12] - e[12], dy = t[13] - e[13], dz = t[14] - e[14];
        // World → light frame: the inverse of the world matrix's linear part.
        const inv = _o3dM.copy(this.matrixWorld).invert().elements;
        return [inv[0] * dx + inv[4] * dy + inv[8] * dz, inv[1] * dx + inv[5] * dy + inv[9] * dz, inv[2] * dx + inv[6] * dy + inv[10] * dz];
    }
    /// What the wasm light is made of, compared at each sync.
    _lightState() {
        const c = this.color, g = this.groundColor;
        return [c.r, c.g, c.b, this.intensity, this.distance ?? 0, this.decay ?? 0, this.angle ?? 0,
            this.penumbra ?? 0, this.width ?? 0, this.height ?? 0, g?.r ?? 0, g?.g ?? 0, g?.b ?? 0,
            ...(this._lightDirection() ?? [0, 0, 0]),
            this.shadow?.bias ?? 0, this.shadow?.normalBias ?? 0, this.shadow?.radius ?? 1,
            this.shadow?.mapSize?.width ?? 0, this.shadow?.mapSize?.height ?? 0];
    }
    _mirrorKind() { return this._w instanceof WebLight ? 'light' : 'group'; }
    _mirrorCreate(w) { return this._w instanceof WebLight ? w.createLight(this._w) : w.createGroup(false); }
    _mirrorContent(w, h, flags, m) {
        if (!(flags & _D_DATA) || m.kind !== 'light') return;
        const lw = this._w;
        const s = this._lightState();
        lw.setColor?.(_color(this.color));
        lw.setIntensity?.(this.intensity);
        if (this.groundColor) lw.setGroundColor?.(_color(this.groundColor));
        if (this.distance !== undefined) lw.setRange?.(this.distance, this.decay ?? 2);
        if (this.angle !== undefined) lw.setCone?.(this.angle, this.penumbra ?? 0);
        if (this.width !== undefined) lw.setRectSize?.(this.width, this.height);
        const d = this._lightDirection();
        if (d) lw.setDirection?.(d[0], d[1], d[2]);
        if (this.shadow) lw.setShadowParams?.(this.shadow.bias ?? 0, this.shadow.normalBias ?? 0, this.shadow.radius ?? 1);
        if (this.shadow?.mapSize) lw.setShadowMapSize?.(this.shadow.mapSize.width, this.shadow.mapSize.height);
        w.setNodeLight(h, lw);
        m.light = s;
    }
    _mirrorPolled() { return true; }
    _mirrorPoll(w, h, m) {
        super._mirrorPoll(w, h, m);
        if (!_sameNumbers(m.light, this._lightState())) this._dirty(_D_DATA);
    }
}
export class AmbientLight extends Light {
    constructor(color = 0xffffff, intensity = 1) {
        super(color, intensity);
        this.isAmbientLight = true;
        this.type = 'AmbientLight';
        this._w = WebLight.ambient(_color(color), intensity);
    }
}
export class DirectionalLight extends Light {
    constructor(color = 0xffffff, intensity = 1) {
        super(color, intensity);
        this.isDirectionalLight = true;
        this.type = 'DirectionalLight';
        this._w = WebLight.directional(_color(color), intensity);
        this.position.copy(Object3D.DEFAULT_UP);
        this.target = new Object3D();
        this.shadow = _makeLightShadow(this);
    }
    copy(source) {
        super.copy(source);
        this.target = source.target.clone();
        return this;
    }
}
export class PointLight extends Light {
    constructor(color = 0xffffff, intensity = 1, distance = 0, decay = 2) {
        super(color, intensity);
        this.isPointLight = true;
        this.type = 'PointLight';
        this._w = WebLight.point(_color(color), intensity, distance, decay);
        this.distance = distance;
        this.decay = decay;
        this.shadow = _makeLightShadow(this);
    }
    get power() { return this.intensity * 4 * Math.PI; }
    set power(p) { this.intensity = p / (4 * Math.PI); }
}
export class SpotLight extends Light {
    constructor(color = 0xffffff, intensity = 1, distance = 0, angle = Math.PI / 3, penumbra = 0, decay = 2) {
        super(color, intensity);
        this.isSpotLight = true;
        this.type = 'SpotLight';
        this._w = WebLight.spot(_color(color), intensity, distance, angle, penumbra, decay);
        this.position.copy(Object3D.DEFAULT_UP);
        this.target = new Object3D();
        this.distance = distance;
        this.angle = angle;
        this.penumbra = penumbra;
        this.decay = decay;
        this.map = null;
        this.shadow = _makeLightShadow(this);
    }
    get power() { return this.intensity * Math.PI; }
    set power(p) { this.intensity = p / Math.PI; }
}
export class HemisphereLight extends Light {
    constructor(skyColor = 0xffffff, groundColor = 0xffffff, intensity = 1) {
        super(skyColor, intensity);
        this.isHemisphereLight = true;
        this.type = 'HemisphereLight';
        this._w = WebLight.hemisphere(_color(skyColor), _color(groundColor), intensity);
        this.position.copy(Object3D.DEFAULT_UP);
        this.groundColor = new Color(groundColor);
    }
}
export class RectAreaLight extends Light {
    constructor(color = 0xffffff, intensity = 1, width = 10, height = 10) {
        super(color, intensity);
        this.isRectAreaLight = true;
        this.type = 'RectAreaLight';
        this._w = WebLight.rectArea(_color(color), intensity, width, height);
        this.width = width;
        this.height = height;
    }
    get power() { return this.intensity * this.width * this.height * Math.PI; }
    set power(p) { this.intensity = p / (this.width * this.height * Math.PI); }
}
// three.js's `RectAreaLightUniformsLib` (`three/addons/lights/`): `init()`
// uploads the area lights' LTC tables before a RectAreaLight can shade. The
// renderer carries those tables already (extracted from three.js r165), so
// `init` has nothing left to do — but three.js code calls it, and a missing
// export would fail the import. `RectAreaLightTexturesLib` is its r166 name.
export const RectAreaLightUniformsLib = { init() {} };
export const RectAreaLightTexturesLib = { init() {} };

// ---- Textures ----
export class Texture {
    constructor(width, height, data) {
        // three.js permits `new Texture()` with no args (the image is filled
        // in later via .image / .needsUpdate). Fall back to a 1×1 white pixel.
        if (width === undefined) {
            this._w = new WebTexture(1, 1, new Uint8Array([255, 255, 255, 255]));
            this.image = null;
        } else {
            this._w = new WebTexture(width, height, data);
            this.image = { width, height, data };
        }
        this.needsUpdate = false;
        this.magFilter = 1006; this.minFilter = 1008;
        this.wrapS = 1001; this.wrapT = 1001;
        this.format = 1023; this.type = 1009;
        this.repeat = new Vector2(1, 1);
        this.offset = new Vector2(0, 0);
        this.center = new Vector2(0, 0);
        this.rotation = 0;
    }
    static solid(r, g, b, a) { const t = Object.create(Texture.prototype); t._w = WebTexture.solid(r, g, b, a); return t; }
}
// three.js `colorSpace` and `needsUpdate` on a texture: either re-binds it in
// every material slot holding it, so a change made after binding — the usual
// `texture.colorSpace = SRGBColorSpace` right after `loader.load()` — reaches
// the GPU. `version` counts `needsUpdate`, as in three.js.
function _defineTextureState(proto) {
    // `offset`, `repeat` and `center` are Vector2s; a plain `{ x, y }` assigned
    // to one becomes one. Edits in place reach the GPU at the next render
    // (`_syncMaterialTextures`), as three.js reads the matrix every frame.
    for (const [key, def] of [['offset', 0], ['repeat', 1], ['center', 0]]) {
        Object.defineProperty(proto, key, {
            get() { return this[`_${key}`]; },
            set(v) {
                this[`_${key}`] = v instanceof Vector2 ? v
                    : new Vector2(v?.x ?? def, v?.y ?? def);
            },
            configurable: true, enumerable: true,
        });
    }
    // three.js `matrix` (a Matrix3, rebuilt from the above each frame while
    // `matrixAutoUpdate` is on) and `matrixAutoUpdate` (default true).
    Object.defineProperty(proto, 'matrix', {
        get() { return this._matrix ??= new Matrix3(); },
        set(m) { this._matrix = m; },
        configurable: true, enumerable: true,
    });
    Object.defineProperty(proto, 'matrixAutoUpdate', {
        get() { return this._matrixAutoUpdate ?? true; },
        set(v) { this._matrixAutoUpdate = v !== false; },
        configurable: true, enumerable: true,
    });
    // three.js `updateMatrix()`: the matrix from offset, repeat, rotation and
    // center.
    if (!proto.updateMatrix) {
        proto.updateMatrix = function () {
            this.matrix.setUvTransform(this.offset.x, this.offset.y, this.repeat.x, this.repeat.y,
                this.rotation ?? 0, this.center.x, this.center.y);
        };
    }
    // three.js `channel`: which uv set the texture reads (0 `uv`, 1 `uv1`, …).
    Object.defineProperty(proto, 'channel', {
        get() { return this._channel ?? 0; },
        set(v) {
            const next = Number(v) | 0;
            if (next === (this._channel ?? 0)) return;
            this._channel = next;
            this._w?.setChannel?.(next);
            _rebindTexture(this);
        },
        configurable: true, enumerable: true,
    });
        Object.defineProperty(proto, 'colorSpace', {
        get() { return this._colorSpace ?? ''; },
        set(v) {
            const next = v ?? '';
            if (next === this._colorSpace) return;
            this._colorSpace = next;
            _rebindTexture(this);
        },
        configurable: true, enumerable: true,
    });
    Object.defineProperty(proto, 'needsUpdate', {
        get() { return false; },
        set(v) {
            if (v !== true) return;
            this.version = (this.version ?? 0) + 1;
            // Filter and wrap changes take effect here, as in three.js.
            _syncTextureFilters(this);
            _syncTextureUv(this);
            _rebindTexture(this);
        },
        configurable: true, enumerable: true,
    });
}
// three.js `CubeTexture(images, mapping, wrapS, wrapT, magFilter, minFilter,
// format, type, anisotropy, colorSpace)`: six face images, +X −X +Y −Y +Z −Z,
// uploaded when they are all there — on construction or on `needsUpdate`, as
// `CubeTextureLoader` sets it. (threers's own form, `(size, px, nx, py, ny,
// pz, nz)` with RGBA bytes, still works.)
export class CubeTexture {
    constructor(...args) {
        this.isCubeTexture = true;
        this.flipY = false;
        if (typeof args[0] === 'number') {
            this.images = [];
            this._w = new WebCubeTexture(...args);
            this._colorSpace = 'srgb';
            return;
        }
        this._w = null;
        this.images = args[0] ?? [];
        this._mapping = args[1] ?? 301;
        this.colorSpace = args[9] ?? '';
        if (this.images.length === 6) this._upload();
    }
    get image() { return this.images; }
    set image(v) { this.images = v; }
    /// three.js `CubeReflectionMapping` (301) or `CubeRefractionMapping` (302).
    get mapping() { return this._mapping ?? 301; }
    set mapping(v) {
        this._mapping = v;
        this._w?.setRefraction?.(v === 302);
        _rebindTexture(this);
    }
    get colorSpace() { return this._colorSpace ?? ''; }
    set colorSpace(v) {
        if ((v ?? '') === this._colorSpace) return;
        this._colorSpace = v ?? '';
        this._w?.setSrgb?.(this._colorSpace === 'srgb');
        _rebindTexture(this);
    }
    get needsUpdate() { return false; }
    set needsUpdate(v) {
        if (v !== true) return;
        this.version = (this.version ?? 0) + 1;
        this._upload();
        _rebindTexture(this);
    }
    /// three.js `clone()`: the same images and settings, uploaded on its own.
    clone() {
        const t = new CubeTexture(this.images.slice(), this.mapping);
        t.colorSpace = this.colorSpace;
        if (this.images.length !== 6 && this._w) t._w = this._w;
        return t;
    }
    /// The six images' pixels, each drawn through a 2D canvas, into a wasm
    /// cube (square faces, all the size of the first).
    _upload() {
        const imgs = this.images;
        if (!Array.isArray(imgs) || imgs.length !== 6 || imgs.some((i) => !i)) return;
        const size = imgs[0].width || imgs[0].naturalWidth || 1;
        const faces = imgs.map((img) => {
            if (img.data && img.width === size) return new Uint8Array(img.data.buffer ?? img.data);
            const c = typeof OffscreenCanvas !== 'undefined'
                ? new OffscreenCanvas(size, size)
                : Object.assign(document.createElement('canvas'), { width: size, height: size });
            const ctx = c.getContext('2d');
            ctx.drawImage(img, 0, 0, size, size);
            return new Uint8Array(ctx.getImageData(0, 0, size, size).data.buffer);
        });
        this._w = new WebCubeTexture(size, ...faces);
        this._w.setSrgb?.(this.colorSpace === 'srgb');
        this._w.setFilters?.(this.magFilter ?? 1006, this.minFilter ?? 1008);
        this._w.setRefraction?.(this.mapping === 302);
    }
}
export class DataTexture {
    constructor(...args) {
        // three.js: (data, width, height, format?, type?)
        // threers earth demo: (data, width, height, 'srgb') for colour maps
        // legacy:   (width, height, data)
        let data, width, height, colourSpace;
        if (typeof args[0] === 'number') {
            [width, height, data] = args;
        } else {
            [data, width, height, colourSpace] = args;
        }
        const u8 = (() => {
            // Always copy into a tight buffer. ImageData / worker views onto
            // pooled ArrayBuffers scramble uploads if we pass .buffer raw.
            if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
            if (data instanceof ArrayBuffer) return new Uint8Array(data).slice();
            return new Uint8Array(data);
        })();
        this._w = colourSpace === 'srgb'
            ? WebDataTexture.newSrgb(width, height, u8)
            : new WebDataTexture(width, height, u8);
        this.image = { data: u8, width, height };
        this.needsUpdate = false;
        this.magFilter = 1006; this.minFilter = 1008;
        this.wrapS = 1001; this.wrapT = 1001;
        this.format = 1023; this.type = 1009;
        this.colorSpace = colourSpace === 'srgb' ? 'srgb' : '';
        this.center = new Vector2(0, 0);
        this.rotation = 0;
        this.offset = new Vector2(0, 0);
        this.repeat = new Vector2(1, 1);
        this._syncFilters();
    }
    _syncFilters() {
        _syncTextureFilters(this);
    }
    _syncUv() {
        if (_syncTextureUv(this)) _rebindTexture(this);
    }
    /** Second handle on the same pixels with its own UV sub-rect (atlas view). */
    view(ox, oy, rx, ry) {
        if (!this._w || typeof this._w.view !== 'function') {
            throw new Error('DataTexture.view: underlying WebDataTexture.view is missing');
        }
        const t = Object.create(DataTexture.prototype);
        t._w = this._w.view(ox, oy, rx, ry);
        t.image = this.image;
        t.needsUpdate = false;
        t.magFilter = this.magFilter; t.minFilter = this.minFilter;
        t.wrapS = this.wrapS; t.wrapT = this.wrapT;
        t.format = this.format; t.type = this.type;
        t.colorSpace = this.colorSpace;
        t.offset = new Vector2(ox, oy);
        t.repeat = new Vector2(rx, ry);
        t.center = new Vector2(0, 0);
        t.rotation = 0;
        t._syncFilters = DataTexture.prototype._syncFilters;
        t._syncUv = DataTexture.prototype._syncUv;
        return t;
    }
}

// CanvasTexture — reads pixels from a 2D canvas / OffscreenCanvas into a
// DataTexture. Mirrors three.js's CanvasTexture; supports `needsUpdate` to
// re-upload after the source canvas changes.
export class CanvasTexture {
    constructor(canvas) {
        this.image = canvas;
        this.format = 1023; this.type = 1009;
        this.magFilter = 1006; this.minFilter = 1008;
        this.wrapS = 1001; this.wrapT = 1001;
        // Canvas pixels arrive top row first; `flipY` (three.js's default
        // for an image) reads them the right way up — see `_textureUvRows`.
        this.flipY = true;
        this._topRowFirst = true;
        this.offset = new Vector2(0, 0);
        this.repeat = new Vector2(1, 1);
        this.center = new Vector2(0, 0);
        this.rotation = 0;
        this._upload();
        this.needsUpdate = false;
    }
    _syncUv() {
        if (_syncTextureUv(this)) _rebindTexture(this);
    }
    _upload() {
        const c = this.image;
        if (!c) {
            this._w = new WebDataTexture(1, 1, new Uint8Array([255, 255, 255, 255]));
            this._syncUv();
            return;
        }
        const ctx = c.getContext('2d');
        const w = c.width, h = c.height;
        const data = ctx.getImageData(0, 0, w, h).data;
        const byteLen = data.length;
        if (!this._buf || this._buf.length !== byteLen) {
            this._buf = new Uint8Array(byteLen);
        }
        this._buf.set(data);
        if (this._w && this._lw === w && this._lh === h && this._w.replaceRgba) {
            if (this._w.replaceRgba(this._buf)) {
                return;
            }
        }
        this._w = new WebDataTexture(w, h, this._buf);
        this._lw = w;
        this._lh = h;
        this._syncUv();
        this._syncFilters();
    }
    _syncFilters() {
        _syncTextureFilters(this);
    }
    update() { this._upload(); }
}
_defineTextureState(Texture.prototype);
_defineTextureState(DataTexture.prototype);
_defineTextureState(CanvasTexture.prototype);
// A CanvasTexture's `needsUpdate` re-reads its canvas first, as three.js
// re-uploads it.
Object.defineProperty(CanvasTexture.prototype, 'needsUpdate', {
    get() { return false; },
    set(v) {
        if (v !== true) return;
        if (this.image) this._upload();
        this.version = (this.version ?? 0) + 1;
        _syncTextureFilters(this);
        _syncTextureUv(this);
        _rebindTexture(this);
    },
    configurable: true, enumerable: true,
});

// ---- Curves (three.js r165, over `src/curves/three.rs`) ----
// Every evaluation runs in Rust, in f64 as three.js computes it: getPoint,
// the arc-length table (`getLengths`, cached with `needsUpdate` exactly as
// three.js caches it), `getUtoTmapping`, `getPointAt`, tangents, Frenet frames
// and CurvePath's walk. These classes keep three.js's properties (`v1`,
// `points`, `closed`, …) and push them to their `WebCurve` before each call.
//
// A subclass that overrides `getPoint` (or `getTangent`) — a user's
// `class MyCurve extends Curve` — runs on a custom `WebCurve` whose Rust
// base-class algorithms call back into that JS method, so getPointAt,
// getSpacedPoints, computeFrenetFrames, TubeGeometry and ExtrudeGeometry's
// `extrudePath` all work on it.
const _CURVE_NATIVE = Symbol('threers.curveNative');
const _curveWeakRef = (o) => (typeof WeakRef === 'function' ? new WeakRef(o) : { deref: () => o });
/// A `WebCurve` calling back into `curve.getPoint` (and `getTangent`, when
/// overridden). Held weakly, so the Rust side does not keep the curve alive.
function _customWebCurve(curve, withTangent) {
    const ref = _curveWeakRef(curve);
    const out = new Float64Array(3);
    const write = (p) => {
        if (!p) return false;
        out[0] = p.x; out[1] = p.y; out[2] = p.z ?? 0;
        return true;
    };
    const point = (t) => write(ref.deref()?.getPoint(t));
    const tangent = withTangent ? (t) => write(ref.deref()?.getTangent(t)) : undefined;
    const p0 = curve.getPoint(0);
    const dims = p0?.isVector2 ? 2 : 3;
    return { rs: WebCurve.custom(point, out, tangent, dims, 0, 0), dims };
}
/// three.js's flags, as a custom `WebCurve`'s role (see `CustomRole`).
function _curveRole(c) {
    if (c.isLineCurve || c.isLineCurve3) return [1, 0];
    if (c.isEllipseCurve) return [2, 0];
    if (c.isSplineCurve) return [3, c.points?.length ?? 0];
    return [0, 0];
}
const _foreignCurves = new WeakMap();
/// The `WebCurve` of any curve: a shim `Curve` (synced), or anything else
/// with a `getPoint` — three.js's own curves included — through callbacks.
function _webCurveOf(c) {
    if (c instanceof Curve) return c._curve();
    let e = _foreignCurves.get(c);
    if (!e) {
        e = _customWebCurve(c, typeof c.getTangent === 'function');
        _foreignCurves.set(c, e);
    }
    const [role, n] = _curveRole(c);
    e.rs.setCustomRole(e.dims, role, n);
    e.rs.arcLengthDivisions = c.arcLengthDivisions ?? 200;
    return e.rs;
}
function _curveVec(dims, a, i, target) {
    const v = target ?? (dims === 2 ? new Vector2() : new Vector3());
    return v.isVector2 ? v.set(a[i], a[i + 1]) : v.set(a[i], a[i + 1], a[i + 2]);
}
function _curveVecs(dims, flat) {
    const out = [];
    for (let i = 0; i < flat.length; i += 3) out.push(_curveVec(dims, flat, i));
    return out;
}

export class Curve {
    constructor() {
        this.type = 'Curve';
        this.arcLengthDivisions = 200;
    }
    /// The `WebCurve` with this curve's own parameters (built-in classes).
    _nativeCurve() {
        const rs = this._rs ??= new WebCurve();
        rs.arcLengthDivisions = this.arcLengthDivisions;
        this._syncCurve(rs);
        return rs;
    }
    _syncCurve(_rs) {}
    /// The `WebCurve` the base-class methods run on: the native one, or a
    /// custom one when `getPoint` / `getTangent` are overridden.
    _curve() {
        const gp = this.getPoint, gt = this.getTangent;
        if (gp[_CURVE_NATIVE] && gt[_CURVE_NATIVE]) return this._nativeCurve();
        if (!this._rsCustom || this._rsGp !== gp || this._rsGt !== gt) {
            this._rsCustom?.rs.free();
            // Any getTangent but the base class's is called back (a line
            // subclass keeps LineCurve's exact direction).
            this._rsCustom = _customWebCurve(this, gt !== Curve.prototype.getTangent);
            this._rsGp = gp; this._rsGt = gt;
        }
        const { rs, dims } = this._rsCustom;
        const [role, n] = _curveRole(this);
        rs.setCustomRole(dims, role, n);
        rs.arcLengthDivisions = this.arcLengthDivisions;
        return rs;
    }
    /// 2 when this curve's points are `Vector2`s.
    _dims() {
        if (this.getPoint[_CURVE_NATIVE] && this.getTangent[_CURVE_NATIVE]) return this._nativeDims ?? this._nativeCurve().dims;
        this._curve();
        return this._rsCustom.dims;
    }
    _vec(a, target) { return a.length === 0 ? null : _curveVec(this._dims(), a, 0, target); }
    get needsUpdate() { return this._curve().needsUpdate; }
    set needsUpdate(v) { this._curve().needsUpdate = !!v; }
    get cacheArcLengths() { const c = this._curve().cacheArcLengths(); return c && Array.from(c); }
    getPoint() {
        console.warn('THREE.Curve: .getPoint() not implemented.');
        return null;
    }
    getPointAt(u, optionalTarget) { return this._vec(this._curve().getPointAt(u), optionalTarget); }
    getPoints(divisions = 5) { return _curveVecs(this._dims(), this._curve().getPoints(divisions)); }
    getSpacedPoints(divisions = 5) { return _curveVecs(this._dims(), this._curve().getSpacedPoints(divisions)); }
    getLength() { return this._curve().getLength(); }
    getLengths(divisions = this.arcLengthDivisions) { return Array.from(this._curve().getLengths(divisions)); }
    updateArcLengths() { this._curve().updateArcLengths(); }
    getUtoTmapping(u, distance) { return this._curve().getUtoTmapping(u, distance); }
    getTangent(t, optionalTarget) { return this._vec(this._curve().getBaseTangent(t), optionalTarget); }
    getTangentAt(u, optionalTarget) { return this._vec(this._curve().getTangentAt(u), optionalTarget); }
    computeFrenetFrames(segments, closed) {
        const f = this._curve().computeFrenetFrames(segments, closed);
        const n = segments + 1;
        const at = (k) => Array.from({ length: n }, (_, i) => new Vector3(f[(k * n + i) * 3], f[(k * n + i) * 3 + 1], f[(k * n + i) * 3 + 2]));
        return { tangents: at(0), normals: at(1), binormals: at(2) };
    }
    clone() { return new this.constructor().copy(this); }
    copy(source) {
        this.arcLengthDivisions = source.arcLengthDivisions;
        return this;
    }
    toJSON() {
        return {
            metadata: { version: 4.6, type: 'Curve', generator: 'Curve.toJSON' },
            arcLengthDivisions: this.arcLengthDivisions,
            type: this.type,
        };
    }
    fromJSON(json) {
        this.arcLengthDivisions = json.arcLengthDivisions;
        return this;
    }
}
Curve.prototype.getTangent[_CURVE_NATIVE] = true;

const _xy = (v) => [v.x, v.y];
const _xyz = (v) => [v.x, v.y, v.z];
const _curveVecJSON = {
    toJSON(c, keys, data) { for (const k of keys) data[k] = c[k].toArray(); return data; },
    fromJSON(c, keys, json) { for (const k of keys) c[k].fromArray(json[k]); return c; },
    copy(c, keys, source) { for (const k of keys) c[k].copy(source[k]); return c; },
};
/// A curve defined by control vectors (lines and Béziers).
class _VectorCurve extends Curve {
    _syncCurve(rs) {
        const flat = this._nativeDims === 2 ? this._keys.flatMap((k) => _xy(this[k])) : this._keys.flatMap((k) => _xyz(this[k]));
        rs[this._setter](flat);
    }
    getPoint(t, optionalTarget) { return _curveVec(this._nativeDims, this._nativeCurve().getPoint(t), 0, optionalTarget); }
    copy(source) { super.copy(source); return _curveVecJSON.copy(this, this._keys, source); }
    toJSON() { return _curveVecJSON.toJSON(this, this._keys, super.toJSON()); }
    fromJSON(json) { super.fromJSON(json); return _curveVecJSON.fromJSON(this, this._keys, json); }
}
_VectorCurve.prototype.getPoint[_CURVE_NATIVE] = true;
/// LineCurve / LineCurve3: linear in arc length, so `getPointAt` is
/// `getPoint` and the tangent is the direction.
class _LineCurveBase extends _VectorCurve {
    getPointAt(u, optionalTarget) { return this.getPoint(u, optionalTarget); }
    getTangent(_t, optionalTarget) { return _curveVec(this._nativeDims, this._nativeCurve().getTangent(0), 0, optionalTarget); }
    getTangentAt(u, optionalTarget) { return this.getTangent(u, optionalTarget); }
}
_LineCurveBase.prototype.getTangent[_CURVE_NATIVE] = true;

export class LineCurve extends _LineCurveBase {
    constructor(v1 = new Vector2(), v2 = new Vector2()) {
        super();
        this.isLineCurve = true;
        this.type = 'LineCurve';
        this.v1 = v1;
        this.v2 = v2;
    }
}
export class LineCurve3 extends _LineCurveBase {
    constructor(v1 = new Vector3(), v2 = new Vector3()) {
        super();
        this.isLineCurve3 = true;
        this.type = 'LineCurve3';
        this.v1 = v1;
        this.v2 = v2;
    }
}
export class QuadraticBezierCurve extends _VectorCurve {
    constructor(v0 = new Vector2(), v1 = new Vector2(), v2 = new Vector2()) {
        super();
        this.isQuadraticBezierCurve = true;
        this.type = 'QuadraticBezierCurve';
        this.v0 = v0; this.v1 = v1; this.v2 = v2;
    }
}
export class QuadraticBezierCurve3 extends _VectorCurve {
    constructor(v0 = new Vector3(), v1 = new Vector3(), v2 = new Vector3()) {
        super();
        this.isQuadraticBezierCurve3 = true;
        this.type = 'QuadraticBezierCurve3';
        this.v0 = v0; this.v1 = v1; this.v2 = v2;
    }
}
export class CubicBezierCurve extends _VectorCurve {
    constructor(v0 = new Vector2(), v1 = new Vector2(), v2 = new Vector2(), v3 = new Vector2()) {
        super();
        this.isCubicBezierCurve = true;
        this.type = 'CubicBezierCurve';
        this.v0 = v0; this.v1 = v1; this.v2 = v2; this.v3 = v3;
    }
}
export class CubicBezierCurve3 extends _VectorCurve {
    constructor(v0 = new Vector3(), v1 = new Vector3(), v2 = new Vector3(), v3 = new Vector3()) {
        super();
        this.isCubicBezierCurve3 = true;
        this.type = 'CubicBezierCurve3';
        this.v0 = v0; this.v1 = v1; this.v2 = v2; this.v3 = v3;
    }
}
for (const [C, dims, setter, keys] of [
    [LineCurve, 2, 'setLine', ['v1', 'v2']],
    [LineCurve3, 3, 'setLine', ['v1', 'v2']],
    [QuadraticBezierCurve, 2, 'setQuadraticBezier', ['v0', 'v1', 'v2']],
    [QuadraticBezierCurve3, 3, 'setQuadraticBezier', ['v0', 'v1', 'v2']],
    [CubicBezierCurve, 2, 'setCubicBezier', ['v0', 'v1', 'v2', 'v3']],
    [CubicBezierCurve3, 3, 'setCubicBezier', ['v0', 'v1', 'v2', 'v3']],
]) {
    Object.defineProperties(C.prototype, {
        _nativeDims: { value: dims }, _setter: { value: setter }, _keys: { value: keys },
    });
}

/// SplineCurve / CatmullRomCurve3: a list of points.
function _pointsJSON(c, data) { data.points = c.points.map((p) => p.toArray()); return data; }
export class SplineCurve extends Curve {
    constructor(points = []) {
        super();
        this.isSplineCurve = true;
        this.type = 'SplineCurve';
        this.points = points;
    }
    get _nativeDims() { return 2; }
    _syncCurve(rs) { rs.setSpline(this.points.flatMap(_xy)); }
    getPoint(t, optionalTarget) { return _curveVec(2, this._nativeCurve().getPoint(t), 0, optionalTarget); }
    copy(source) { super.copy(source); this.points = source.points.map((p) => p.clone()); return this; }
    toJSON() { return _pointsJSON(this, super.toJSON()); }
    fromJSON(json) { super.fromJSON(json); this.points = json.points.map((p) => new Vector2().fromArray(p)); return this; }
}
SplineCurve.prototype.getPoint[_CURVE_NATIVE] = true;
export class CatmullRomCurve3 extends Curve {
    constructor(points = [], closed = false, curveType = 'centripetal', tension = 0.5) {
        super();
        this.isCatmullRomCurve3 = true;
        this.type = 'CatmullRomCurve3';
        this.points = points;
        this.closed = closed;
        this.curveType = curveType;
        this.tension = tension;
    }
    get _nativeDims() { return 3; }
    _syncCurve(rs) { rs.setCatmullRom(this.points.flatMap(_xyz), !!this.closed, String(this.curveType), this.tension); }
    getPoint(t, optionalTarget) { return _curveVec(3, this._nativeCurve().getPoint(t), 0, optionalTarget); }
    copy(source) {
        super.copy(source);
        this.points = source.points.map((p) => p.clone());
        this.closed = source.closed;
        this.curveType = source.curveType;
        this.tension = source.tension;
        return this;
    }
    toJSON() {
        const data = _pointsJSON(this, super.toJSON());
        data.closed = this.closed;
        data.curveType = this.curveType;
        data.tension = this.tension;
        return data;
    }
    fromJSON(json) {
        super.fromJSON(json);
        this.points = json.points.map((p) => new Vector3().fromArray(p));
        this.closed = json.closed;
        this.curveType = json.curveType;
        this.tension = json.tension;
        return this;
    }
}
CatmullRomCurve3.prototype.getPoint[_CURVE_NATIVE] = true;

const _ELLIPSE_KEYS = ['aX', 'aY', 'xRadius', 'yRadius', 'aStartAngle', 'aEndAngle', 'aClockwise', 'aRotation'];
export class EllipseCurve extends Curve {
    constructor(aX = 0, aY = 0, xRadius = 1, yRadius = 1, aStartAngle = 0, aEndAngle = Math.PI * 2, aClockwise = false, aRotation = 0) {
        super();
        this.isEllipseCurve = true;
        this.type = 'EllipseCurve';
        this.aX = aX;
        this.aY = aY;
        this.xRadius = xRadius;
        this.yRadius = yRadius;
        this.aStartAngle = aStartAngle;
        this.aEndAngle = aEndAngle;
        this.aClockwise = aClockwise;
        this.aRotation = aRotation;
    }
    get _nativeDims() { return 2; }
    _syncCurve(rs) {
        rs.setEllipse(this.aX, this.aY, this.xRadius, this.yRadius, this.aStartAngle, this.aEndAngle, this.aClockwise === true, this.aRotation);
    }
    getPoint(t, optionalTarget) { return _curveVec(2, this._nativeCurve().getPoint(t), 0, optionalTarget); }
    copy(source) { super.copy(source); for (const k of _ELLIPSE_KEYS) this[k] = source[k]; return this; }
    toJSON() { const data = super.toJSON(); for (const k of _ELLIPSE_KEYS) data[k] = this[k]; return data; }
    fromJSON(json) { super.fromJSON(json); for (const k of _ELLIPSE_KEYS) this[k] = json[k]; return this; }
}
EllipseCurve.prototype.getPoint[_CURVE_NATIVE] = true;
export class ArcCurve extends EllipseCurve {
    constructor(aX, aY, aRadius, aStartAngle, aEndAngle, aClockwise) {
        super(aX, aY, aRadius, aRadius, aStartAngle, aEndAngle, aClockwise);
        this.isArcCurve = true;
        this.type = 'ArcCurve';
    }
}

/// three.js's `Curves` namespace, for `CurvePath.fromJSON`.
const _Curves = {
    ArcCurve, CatmullRomCurve3, CubicBezierCurve, CubicBezierCurve3, EllipseCurve,
    LineCurve, LineCurve3, QuadraticBezierCurve, QuadraticBezierCurve3, SplineCurve,
};
export class CurvePath extends Curve {
    constructor() {
        super();
        this.type = 'CurvePath';
        // A `Path` keeps these in Rust behind accessors.
        if (!('curves' in this)) this.curves = [];
        if (!('autoClose' in this)) this.autoClose = false;
    }
    _syncCurve(rs) {
        rs.setPath(!!this.autoClose);
        for (const c of this.curves) rs.addCurve(_webCurveOf(c));
    }
    _dims() {
        if (this.getPoint[_CURVE_NATIVE] && this.getTangent[_CURVE_NATIVE]) {
            const c = this.curves[0];
            return c instanceof Curve ? c._dims() : c?.getPoint?.(0)?.isVector2 ? 2 : 3;
        }
        return super._dims();
    }
    add(curve) { this.curves.push(curve); }
    closePath() {
        // Adds a line curve to the JS list, so it is done here, as three.js does.
        const startPoint = this.curves[0].getPoint(0);
        const endPoint = this.curves[this.curves.length - 1].getPoint(1);
        if (!startPoint.equals(endPoint)) {
            this.curves.push(startPoint.isVector2 === true ? new LineCurve(endPoint, startPoint) : new LineCurve3(endPoint, startPoint));
        }
        return this;
    }
    getPoint(t, optionalTarget) { return this._vec(this._nativeCurve().getPoint(t), optionalTarget); }
    getLength() { return this._nativeCurve().getLength(); }
    updateArcLengths() { this._nativeCurve().updateArcLengths(); }
    getCurveLengths() { return Array.from(this._nativeCurve().getCurveLengths()); }
    getSpacedPoints(divisions = 40) { return _curveVecs(this._dims(), this._nativeCurve().getSpacedPoints(divisions)); }
    getPoints(divisions = 12) { return _curveVecs(this._dims(), this._nativeCurve().getPoints(divisions)); }
    copy(source) {
        super.copy(source);
        this.curves = source.curves.map((c) => c.clone());
        this.autoClose = source.autoClose;
        return this;
    }
    toJSON() {
        const data = super.toJSON();
        data.autoClose = this.autoClose;
        data.curves = this.curves.map((c) => c.toJSON());
        return data;
    }
    fromJSON(json) {
        super.fromJSON(json);
        this.autoClose = json.autoClose;
        this.curves = [...this.curves, ...json.curves.map((c) => new _Curves[c.type]().fromJSON(c))];
        return this;
    }
}
CurvePath.prototype.getPoint[_CURVE_NATIVE] = true;
// ---- Path / Shape / ShapePath (three.js r165, over `src/curves/shapes`) ----
// Sampling, arc length and triangulation all run in Rust, in f64 as three.js
// computes them; these classes hold a `WebShapeCurvePath` and translate
// Vector2 arrays to and from flat Float64Arrays.
function _flat2(points) {
    const out = new Float64Array(points.length * 2);
    for (let i = 0; i < points.length; i++) { out[i * 2] = points[i].x; out[i * 2 + 1] = points[i].y; }
    return out;
}
function _vec2s(flat) {
    const out = [];
    for (let i = 0; i < flat.length; i += 2) out.push(new Vector2(flat[i], flat[i + 1]));
    return out;
}
const _v2 = (d, i) => new Vector2(d[i], d[i + 1]);
/// `Path.curves`, read back as three.js-shaped curve descriptions.
function _shapeCurveFromData(d) {
    switch (d[0]) {
        case 0: return new LineCurve(_v2(d, 1), _v2(d, 3));
        case 1: return new QuadraticBezierCurve(_v2(d, 1), _v2(d, 3), _v2(d, 5));
        case 2: return new CubicBezierCurve(_v2(d, 1), _v2(d, 3), _v2(d, 5), _v2(d, 7));
        case 3: return new SplineCurve(_vec2s(d.subarray(1)));
        default: return new EllipseCurve(d[1], d[2], d[3], d[4], d[5], d[6], d[7] !== 0, d[8]);
    }
}
/// `CurvePath.add(curve)` for the 2D curve types three.js's Path is made of.
function _addShapeCurve(w, c) {
    if (c.isEllipseCurve || c.aX !== undefined) {
        const rx = c.xRadius ?? c.aRadius ?? 1, ry = c.yRadius ?? c.aRadius ?? 1;
        w.addEllipseCurve(c.aX ?? 0, c.aY ?? 0, rx, ry, c.aStartAngle ?? 0, c.aEndAngle ?? Math.PI * 2, !!c.aClockwise, c.aRotation ?? 0);
    } else if (c.isSplineCurve || Array.isArray(c.points)) {
        w.addSplineCurve(_flat2(c.points));
    } else if (c.v3) {
        w.addCubicCurve(c.v0.x, c.v0.y, c.v1.x, c.v1.y, c.v2.x, c.v2.y, c.v3.x, c.v3.y);
    } else if (c.v0 && c.v2) {
        w.addQuadraticCurve(c.v0.x, c.v0.y, c.v1.x, c.v1.y, c.v2.x, c.v2.y);
    } else if (c.v1 && c.v2) {
        w.addLineCurve(c.v1.x, c.v1.y, c.v2.x, c.v2.y);
    } else {
        throw new TypeError(`Path.add: unsupported curve ${c?.type ?? c}`);
    }
}
export class Path extends CurvePath {
    constructor(points) {
        super();
        this._w = new WebShapeCurvePath();
        this.type = 'Path';
        this.arcLengthDivisions = 200;
        if (points) this.setFromPoints(points);
    }
    get currentPoint() { return new Vector2(this._w.currentX, this._w.currentY); }
    set currentPoint(v) { this._w.setCurrentPoint(v.x, v.y); }
    get autoClose() { return this._w.autoClose; }
    set autoClose(v) { this._w.autoClose = !!v; }
    /// A snapshot: edit a path through its methods, or assign a new array.
    get curves() {
        const out = [];
        for (let i = 0, n = this._w.curveCount(); i < n; i++) out.push(_shapeCurveFromData(this._w.curveData(i)));
        return out;
    }
    set curves(list) { this._w.clearCurves(); for (const c of list) _addShapeCurve(this._w, c); }
    add(curve) { _addShapeCurve(this._w, curve); }
    setFromPoints(points) { this._w.setFromPoints(_flat2(points)); return this; }
    moveTo(x, y) { this._w.moveTo(x, y); return this; }
    lineTo(x, y) { this._w.lineTo(x, y); return this; }
    quadraticCurveTo(cx, cy, x, y) { this._w.quadraticCurveTo(cx, cy, x, y); return this; }
    bezierCurveTo(c1x, c1y, c2x, c2y, x, y) { this._w.bezierCurveTo(c1x, c1y, c2x, c2y, x, y); return this; }
    splineThru(pts) { this._w.splineThru(_flat2(pts)); return this; }
    arc(x, y, r, a0, a1, cw) { this._w.arc(x, y, r, a0, a1, !!cw); return this; }
    absarc(x, y, r, a0, a1, cw) { this._w.absarc(x, y, r, a0, a1, !!cw); return this; }
    ellipse(x, y, rx, ry, a0, a1, cw, rot) { this._w.ellipse(x, y, rx, ry, a0, a1, !!cw, rot ?? 0); return this; }
    absellipse(x, y, rx, ry, a0, a1, cw, rot) { this._w.absellipse(x, y, rx, ry, a0, a1, !!cw, rot ?? 0); return this; }
    closePath() { this._w.closePath(); return this; }
    getPoints(divisions = 12) { return _vec2s(this._w.getPoints(divisions)); }
    getSpacedPoints(divisions = 40) { return _vec2s(this._w.getSpacedPoints(divisions)); }
    getLength() { return this._w.getLength(); }
    getCurveLengths() { return Array.from(this._w.getCurveLengths()); }
    getLengths(divisions = this.arcLengthDivisions) { return Array.from(this._w.getLengths(divisions)); }
    updateArcLengths() {}
    getPoint(t, target = new Vector2()) { const p = this._w.getPoint(t); return p ? target.set(p[0], p[1]) : null; }
    getPointAt(u, target = new Vector2()) { const p = this._w.getPointAt(u); return p ? target.set(p[0], p[1]) : null; }
    getTangent(t, target = new Vector2()) { const p = this._w.getTangent(t); return target.set(p[0], p[1]); }
    getTangentAt(u, target = new Vector2()) { const p = this._w.getTangentAt(u); return target.set(p[0], p[1]); }
    copy(source) { this._w = source._w.clone(); this.arcLengthDivisions = source.arcLengthDivisions; return this; }
    clone() { return new this.constructor().copy(this); }
    toJSON() {
        const data = super.toJSON();
        data.currentPoint = this.currentPoint.toArray();
        return data;
    }
    fromJSON(json) {
        super.fromJSON(json);
        this.currentPoint = new Vector2().fromArray(json.currentPoint);
        return this;
    }
}
export class Shape extends Path {
    constructor(points) {
        super(points);
        this.uuid = MathUtils.generateUUID();
        this.type = 'Shape';
        this.holes = [];
    }
    getPointsHoles(divisions) { return this.holes.map((h) => h.getPoints(divisions)); }
    extractPoints(divisions) {
        return { shape: this.getPoints(divisions), holes: this.getPointsHoles(divisions) };
    }
    copy(source) { super.copy(source); this.holes = (source.holes ?? []).map((h) => h.clone()); return this; }
    toJSON() {
        const data = super.toJSON();
        data.uuid = this.uuid;
        data.holes = this.holes.map((h) => h.toJSON());
        return data;
    }
    fromJSON(json) {
        super.fromJSON(json);
        this.uuid = json.uuid;
        this.holes = json.holes.map((h) => new Path().fromJSON(h));
        return this;
    }
}
/// A `WebShapeList` (shapes + holes) as JS Shapes.
function _shapesFromWasmList(list) {
    const out = [];
    for (let i = 0; i < list.length; i++) {
        const s = new Shape();
        s._w.free();
        s._w = list.outline(i);
        for (let j = 0, n = list.holeCount(i); j < n; j++) {
            const h = new Path();
            h._w.free();
            h._w = list.hole(i, j);
            s.holes.push(h);
        }
        out.push(s);
    }
    return out;
}
/// JS Shapes (or one Shape) as a `WebShapeList`; a plain point array is
/// taken as an outline. The caller frees it.
function _wasmShapeList(shapes) {
    const pathOf = (p) => {
        if (p?._w instanceof WebShapeCurvePath) return { w: p._w, own: false };
        const w = new WebShapeCurvePath();
        w.setFromPoints(_flat2(Array.isArray(p) ? p : p.getPoints()));
        return { w, own: true };
    };
    const list = new WebShapeList();
    for (const s of (Array.isArray(shapes) ? shapes : [shapes])) {
        const o = pathOf(s);
        list.addShape(o.w);
        if (o.own) o.w.free();
        for (const hole of s.holes ?? []) {
            const h = pathOf(hole);
            list.addHole(h.w);
            if (h.own) h.w.free();
        }
    }
    return list;
}
export class ShapePath {
    constructor() {
        this.type = 'ShapePath';
        this.color = new Color();
        this.subPaths = [];
        this.currentPath = null;
    }
    moveTo(x, y) {
        this.currentPath = new Path();
        this.subPaths.push(this.currentPath);
        this.currentPath.moveTo(x, y);
        return this;
    }
    lineTo(x, y) { this.currentPath.lineTo(x, y); return this; }
    quadraticCurveTo(cx, cy, x, y) { this.currentPath.quadraticCurveTo(cx, cy, x, y); return this; }
    bezierCurveTo(c1x, c1y, c2x, c2y, x, y) { this.currentPath.bezierCurveTo(c1x, c1y, c2x, c2y, x, y); return this; }
    splineThru(pts) { this.currentPath.splineThru(pts); return this; }
    toShapes(isCCW) {
        const subs = new WebShapeList();
        for (const p of this.subPaths) subs.addShape(p._w);
        const shapes = subs.toShapes(!!isCCW);
        subs.free();
        const out = _shapesFromWasmList(shapes);
        shapes.free();
        return out;
    }
}

// ---- Animation ----
//
// three.js r165's animation system. The mixer, its actions, the property
// mixers and their blending, every interpolant, track and clip operation,
// track-name parsing and node lookup run in Rust (`src/animation/three`,
// bound in `src/wasm/animation.rs`). This is the three.js API over them, and
// the JS objects' side of `PropertyBinding`: three callbacks the Rust mixer
// calls — find a node (in the Rust scene graph, through the mirror), bind a
// property on the JS object three.js code expects to see animated, read a
// property's value — and the store that writes the values a mixer computed
// into the JS objects, once per update, through their own setters (so the
// scene-graph mirror carries them to the Rust nodes at the next sync).
//
// Track and clip data stay in the JS arrays three.js code reads and writes
// (`track.times`, `track.values`); a clip is copied to Rust when an action is
// made from it (and again if its tracks were edited through their methods).

export const LoopOnce = 2200, LoopRepeat = 2201, LoopPingPong = 2202;
export const NormalAnimationBlendMode = 2500, AdditiveAnimationBlendMode = 2501;
export const InterpolateDiscrete = 2300, InterpolateLinear = 2301, InterpolateSmooth = 2302;
export const ZeroCurvatureEnding = 2400, ZeroSlopeEnding = 2401, WrapAroundEnding = 2402;

/// Objects the mixer addresses — roots, nodes found, group members, groups —
/// numbered once each.
let _animNextKey = 1;
const _animKeys = new WeakMap();
const _animRefs = new Map();
const _animKeyFinalizer = typeof FinalizationRegistry === 'function'
    ? new FinalizationRegistry((key) => { _animRefs.delete(key); dropAnimationNode(key); })
    : null;
function _animKey(obj) {
    let k = _animKeys.get(obj);
    if (k === undefined) {
        k = _animNextKey++;
        _animKeys.set(obj, k);
        _animRefs.set(k, new WeakRef(obj));
        _animKeyFinalizer?.register(obj, k);
    }
    return k;
}
function _animObject(key) { return _animRefs.get(key)?.deref(); }
/// A root's number, with its current uuid (what three.js keys caches by).
function _animRootKey(obj) {
    const k = _animKey(obj);
    if (!obj.isAnimationObjectGroup) setAnimationNodeUuid(k, String(obj.uuid));
    return k;
}

/// three.js `PropertyBinding.findNode` for a root that is no Object3D
/// (three.js lets any object with `children` / `skeleton` be a root).
function _animFindPlain(root, nodeName) {
    if (root.skeleton) {
        const bone = root.skeleton.getBoneByName?.(nodeName);
        if (bone !== undefined) return bone;
    }
    const search = (children) => {
        for (const c of children || []) {
            if (c.name === nodeName || c.uuid === nodeName) return c;
            const r = search(c.children);
            if (r) return r;
        }
        return null;
    };
    return search(root.children);
}

/// Callback: `PropertyBinding.findNode(root, nodeName)` — the root itself, or
/// a node of the Rust scene graph the root is mirrored in.
function _animFindNode(rootKey, nodeName) {
    const root = _animObject(rootKey);
    if (!root) return undefined;
    if (nodeName === undefined || nodeName === '' || nodeName === '.' || nodeName === -1
        || nodeName === root.name || nodeName === root.uuid) return rootKey;
    let found = null;
    if (root instanceof Object3D) {
        root._syncNow();
        const m = root.isScene ? root._m : root._m;
        if (m) {
            const id = m.graph.w.findAnimationNode(m.id, nodeName);
            if (id !== undefined) found = m.graph.objectOf(id) ?? (id === m.graph.rootId ? m.graph.scene : null);
        }
    } else {
        found = _animFindPlain(root, nodeName);
    }
    return found ? _animKey(found) : undefined;
}

/// Bound properties, by the number the Rust side gave each binding.
const _animTargets = new Map();

/// three.js `PropertyBinding.bind()` past `findNode`, on a JS object: the
/// object (`material`, `materials`, `bones`, `map`, any `node[name]`), the
/// property and its binding type and versioning. Returns `{ get, set, width }`
/// or null (unavailable, as three.js reports it).
function _animAccessor(node, objectName, objectIndex, propertyName, propertyIndex, typeName, valueSize, path, nodeName) {
    let targetObject = node;
    if (!targetObject) {
        console.warn('THREE.PropertyBinding: No target node found for track: ' + path + '.');
        return null;
    }
    if (objectName) {
        switch (objectName) {
            case 'materials':
                if (!targetObject.material) { console.error('THREE.PropertyBinding: Can not bind to material as node does not have a material.', path); return null; }
                if (!targetObject.material.materials) { console.error('THREE.PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.', path); return null; }
                targetObject = targetObject.material.materials;
                break;
            case 'bones':
                if (!targetObject.skeleton) { console.error('THREE.PropertyBinding: Can not bind to bones as node does not have a skeleton.', path); return null; }
                targetObject = targetObject.skeleton.bones;
                for (let i = 0; i < targetObject.length; i++) {
                    if (targetObject[i].name === objectIndex) { objectIndex = i; break; }
                }
                break;
            case 'map':
                if ('map' in targetObject) { targetObject = targetObject.map; break; }
                if (!targetObject.material) { console.error('THREE.PropertyBinding: Can not bind to material as node does not have a material.', path); return null; }
                if (!targetObject.material.map) { console.error('THREE.PropertyBinding: Can not bind to material.map as node.material does not have a map.', path); return null; }
                targetObject = targetObject.material.map;
                break;
            default:
                if (targetObject[objectName] === undefined) { console.error('THREE.PropertyBinding: Can not bind to objectName of node undefined.', path); return null; }
                targetObject = targetObject[objectName];
        }
        if (objectIndex !== undefined) {
            if (targetObject[objectIndex] === undefined) { console.error('THREE.PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.', path); return null; }
            targetObject = targetObject[objectIndex];
        }
    }
    const nodeProperty = targetObject[propertyName];
    if (nodeProperty === undefined) {
        console.error('THREE.PropertyBinding: Trying to update property for track: ' + nodeName + '.' + propertyName + ' but it wasn\'t found.', targetObject);
        return null;
    }
    // versioning: a material re-reads itself, a node its world matrix
    const t = targetObject;
    const version = t.needsUpdate !== undefined ? () => { t.needsUpdate = true; }
        : t.matrixWorldNeedsUpdate !== undefined ? () => { t.matrixWorldNeedsUpdate = true; }
        : null;
    const asBool = typeName === 'bool';
    if (propertyIndex !== undefined) {
        if (propertyName === 'morphTargetInfluences') {
            if (!t.geometry) { console.error('THREE.PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.', path); return null; }
            if (!t.geometry.morphAttributes) { console.error('THREE.PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.', path); return null; }
            if (t.morphTargetDictionary?.[propertyIndex] !== undefined) propertyIndex = t.morphTargetDictionary[propertyIndex];
        }
        const arr = nodeProperty, idx = propertyIndex;
        return {
            width: 1,
            get: () => arr[idx],
            set: (v, o) => { arr[idx] = asBool ? v[o] !== 0 : v[o]; version?.(); },
        };
    }
    if (nodeProperty.fromArray !== undefined && nodeProperty.toArray !== undefined) {
        const p = nodeProperty;
        // An Euler's fromArray reads from the start of the array (and takes
        // a fourth value as its order); set its angles instead.
        const set = p.isEuler
            ? (v, o) => { p.set(v[o], v[o + 1], v[o + 2]); version?.(); }
            : (v, o) => { p.fromArray(v, o); version?.(); };
        return { width: Math.min(valueSize, p.toArray().length), get: () => p.toArray(), set };
    }
    if (Array.isArray(nodeProperty)) {
        const dest = nodeProperty;
        return {
            width: Math.max(1, dest.length),
            get: () => dest,
            set: (v, o, n) => {
                for (let i = 0, m = Math.min(dest.length, n); i !== m; ++i) dest[i] = v[o + i];
                version?.();
            },
        };
    }
    return {
        width: 1,
        get: () => t[propertyName],
        set: (v, o) => { t[propertyName] = asBool ? v[o] !== 0 : v[o]; version?.(); },
        setString: (v) => { t[propertyName] = v[0]; version?.(); },
    };
}

/// Callback: bind `target` to a property of node `nodeKey`; returns how many
/// values its setter takes (0: unavailable).
function _animBind(target, nodeKey, objectName, objectIndex, propertyName, propertyIndex, typeName, valueSize, path, nodeName) {
    const acc = _animAccessor(_animObject(nodeKey), objectName, objectIndex, propertyName, propertyIndex,
        typeName, valueSize, path, nodeName);
    if (!acc) return 0;
    _animTargets.set(target, acc);
    return acc.width;
}

/// Callback: a bound property's current value (number, boolean, string or array).
function _animRead(target) {
    return _animTargets.get(target)?.get();
}

let _animCallbacksSet = false;
function _animInit() {
    if (_animCallbacksSet) return;
    setAnimationCallbacks(_animFindNode, _animBind, _animRead);
    _animCallbacksSet = true;
}

/// Store what the mixers wrote since the last call into the JS objects.
function _animFlush() {
    if (!animationWritesPending()) return;
    const w = takeAnimationWrites();
    for (let i = 0, n = w.length; i < n;) {
        const count = w[i + 1];
        _animTargets.get(w[i])?.set(w, i + 2, count);
        i += 2 + count;
    }
    const s = takeAnimationStringWrites();
    for (let i = 0; i < s.length; i += 2) {
        const acc = _animTargets.get(s[i]);
        if (acc?.setString) acc.setString(s[i + 1]);
    }
    for (const t of takeAnimationReleased()) _animTargets.delete(t);
}

// ---- Interpolants ----

const _INTERPOLANT_KIND = new Map();

/// three.js `Interpolant`: the seek and the built-in interpolations run in
/// Rust over a copy of the samples (taken again when `parameterPositions` or
/// `sampleValues` is replaced). A subclass that overrides `interpolate_`
/// still gets the Rust seek, then its own methods.
export class Interpolant {
    constructor(parameterPositions, sampleValues, sampleSize, resultBuffer) {
        this.parameterPositions = parameterPositions;
        this._cachedIndex = 0;
        this.resultBuffer = resultBuffer !== undefined ? resultBuffer : new sampleValues.constructor(sampleSize);
        this.sampleValues = sampleValues;
        this.valueSize = sampleSize;
        this.settings = null;
        this.DefaultSettings_ = {};
        this._w = null;
    }
    _wasm(kind) {
        if (!this._w || this._wTimes !== this.parameterPositions || this._wValues !== this.sampleValues) {
            this._w?.free?.();
            this._w = new WebInterpolant(kind, Float32Array.from(this.parameterPositions),
                Float32Array.from(this.sampleValues, Number), this.valueSize);
            this._wTimes = this.parameterPositions;
            this._wValues = this.sampleValues;
            this._wSettings = null;
        }
        const s = this.settings;
        const key = s ? `${s.endingStart},${s.endingEnd}` : null;
        if (key !== this._wSettings) {
            if (s) this._w.setSettings(s.endingStart, s.endingEnd); else this._w.clearSettings();
            this._wSettings = key;
        }
        return this._w;
    }
    evaluate(t) {
        const kind = _INTERPOLANT_KIND.get(this.constructor);
        const result = this.resultBuffer;
        if (kind !== undefined && this.interpolate_ === Interpolant.prototype.interpolate_) {
            const w = this._wasm(kind);
            const r = result instanceof Float64Array || Array.isArray(result) ? w.evaluate64(t) : w.evaluate32(t);
            for (let i = 0, n = r.length; i !== n; ++i) result[i] = r[i];
            return result;
        }
        // A JS subclass: the seek here, its template methods for the rest.
        const s = this._wasm(1).seek(t);
        this._cachedIndex = s[4];
        if (s[0] === 0) return this.copySampleValue_(s[1]);
        if (s[0] === 2) this.intervalChanged_(s[1], s[2], s[3]);
        return this.interpolate_(s[1], s[2], t, s[3]);
    }
    getSettings_() { return this.settings || this.DefaultSettings_; }
    copySampleValue_(index) {
        const result = this.resultBuffer, values = this.sampleValues, stride = this.valueSize, offset = index * stride;
        for (let i = 0; i !== stride; ++i) result[i] = values[offset + i];
        return result;
    }
    interpolate_() { throw new Error('call to abstract method'); }
    intervalChanged_() {}
}
export class DiscreteInterpolant extends Interpolant {}
export class LinearInterpolant extends Interpolant {}
export class CubicInterpolant extends Interpolant {
    constructor(parameterPositions, sampleValues, sampleSize, resultBuffer) {
        super(parameterPositions, sampleValues, sampleSize, resultBuffer);
        this.DefaultSettings_ = { endingStart: ZeroCurvatureEnding, endingEnd: ZeroCurvatureEnding };
    }
}
export class QuaternionLinearInterpolant extends Interpolant {}
_INTERPOLANT_KIND.set(DiscreteInterpolant, 0).set(LinearInterpolant, 1).set(CubicInterpolant, 2)
    .set(QuaternionLinearInterpolant, 3);

// ---- Keyframe tracks ----

let _trackVersion = 0;

/// three.js `KeyframeTrack`. Its arrays are the ones three.js code reads;
/// `shift`, `scale`, `trim`, `optimize` and `validate` run in Rust.
export class KeyframeTrack {
    constructor(name, times, values, interpolation) {
        if (name === undefined) throw new Error('THREE.KeyframeTrack: track name is undefined');
        if (times === undefined || times.length === 0) throw new Error('THREE.KeyframeTrack: no keyframes in track named ' + name);
        this.name = name;
        this.times = AnimationUtils.convertArray(times, this.TimeBufferType);
        this.values = AnimationUtils.convertArray(values, this.ValueBufferType);
        this._version = ++_trackVersion;
        this.setInterpolation(interpolation || this.DefaultInterpolation);
    }
    static toJSON(track) {
        const trackType = track.constructor;
        if (trackType.toJSON !== KeyframeTrack.toJSON && typeof trackType.toJSON === 'function') {
            const json = trackType.toJSON(track);
            json.type = track.ValueTypeName;
            return json;
        }
        const w = track._toWasm();
        const json = JSON.parse(w.toJSON());
        w.free();
        return json;
    }
    InterpolantFactoryMethodDiscrete(result) { return new DiscreteInterpolant(this.times, this.values, this.getValueSize(), result); }
    InterpolantFactoryMethodLinear(result) { return new LinearInterpolant(this.times, this.values, this.getValueSize(), result); }
    InterpolantFactoryMethodSmooth(result) { return new CubicInterpolant(this.times, this.values, this.getValueSize(), result); }
    setInterpolation(interpolation) {
        let factoryMethod;
        switch (interpolation) {
            case InterpolateDiscrete: factoryMethod = this.InterpolantFactoryMethodDiscrete; break;
            case InterpolateLinear: factoryMethod = this.InterpolantFactoryMethodLinear; break;
            case InterpolateSmooth: factoryMethod = this.InterpolantFactoryMethodSmooth; break;
        }
        if (factoryMethod === undefined) {
            const message = 'unsupported interpolation for ' + this.ValueTypeName + ' keyframe track named ' + this.name;
            if (this.createInterpolant === undefined) {
                if (interpolation !== this.DefaultInterpolation) this.setInterpolation(this.DefaultInterpolation);
                else throw new Error(message);
            }
            console.warn('THREE.KeyframeTrack:', message);
            return this;
        }
        this.createInterpolant = factoryMethod;
        this._gltfCubic = false;
        this._version = ++_trackVersion;
        return this;
    }
    getInterpolation() {
        switch (this.createInterpolant) {
            case this.InterpolantFactoryMethodDiscrete: return InterpolateDiscrete;
            case this.InterpolantFactoryMethodLinear: return InterpolateLinear;
            case this.InterpolantFactoryMethodSmooth: return InterpolateSmooth;
        }
    }
    getValueSize() { return this.values.length / this.times.length; }
    /// The track as Rust data (the caller frees it).
    _toWasm() {
        const type = this.ValueTypeName;
        const v = this.values;
        const interp = this.createInterpolant?.isInterpolantFactoryMethodGLTFCubicSpline || this._gltfCubic
            ? 1 : (this.getInterpolation() ?? 0);
        const w = new WebAnimTrack(type, String(this.name), Float32Array.from(this.times, Number),
            type === 'bool' || type === 'string' ? undefined : (v instanceof Float32Array ? v : Float32Array.from(v, Number)),
            type === 'bool' ? Uint8Array.from(v, (x) => (x ? 1 : 0)) : undefined,
            type === 'string' ? Array.from(v, (x) => String(x)) : undefined,
            interp);
        if (this.modifiers?.length) w.setModifiers(...TrackModifier._encode(this.modifiers));
        return w;
    }
    /// Take back the arrays of a Rust track (after an edit there).
    _fromWasm(w, inPlace) {
        const type = this.ValueTypeName;
        const times = w.times;
        if (inPlace && this.times.length === times.length && ArrayBuffer.isView(this.times)) this.times.set(times);
        else this.times = AnimationUtils.convertArray(times, this.TimeBufferType);
        const values = type === 'bool' ? Array.from(w.bools, (b) => b !== 0)
            : type === 'string' ? w.strings : w.values;
        if (inPlace && this.values.length === values.length && ArrayBuffer.isView(this.values)) this.values.set(values);
        else this.values = AnimationUtils.convertArray(values, this.ValueBufferType);
        this._version = ++_trackVersion;
    }
    _edit(op, inPlace) {
        const w = this._toWasm();
        op(w);
        this._fromWasm(w, inPlace);
        w.free();
        return this;
    }
    shift(timeOffset) { return timeOffset !== 0.0 ? this._edit((w) => w.shift(timeOffset), true) : this; }
    scale(timeScale) { return timeScale !== 1.0 ? this._edit((w) => w.scale(timeScale), true) : this; }
    trim(startTime, endTime) { return this._edit((w) => w.trim(startTime, endTime), false); }
    validate() {
        const w = this._toWasm();
        const ok = w.validate();
        w.free();
        return ok;
    }
    optimize() { return this._edit((w) => w.optimize(), false); }
    clone() {
        const track = new this.constructor(this.name, this.times.slice(), this.values.slice());
        track.createInterpolant = this.createInterpolant;
        if (this.modifiers) track.modifiers = this.modifiers.slice();
        return track;
    }
}
KeyframeTrack.prototype.TimeBufferType = Float32Array;
KeyframeTrack.prototype.ValueBufferType = Float32Array;
KeyframeTrack.prototype.DefaultInterpolation = InterpolateLinear;

export class BooleanKeyframeTrack extends KeyframeTrack { constructor(name, times, values) { super(name, times, values); } }
BooleanKeyframeTrack.prototype.ValueTypeName = 'bool';
BooleanKeyframeTrack.prototype.ValueBufferType = Array;
BooleanKeyframeTrack.prototype.DefaultInterpolation = InterpolateDiscrete;
BooleanKeyframeTrack.prototype.InterpolantFactoryMethodLinear = undefined;
BooleanKeyframeTrack.prototype.InterpolantFactoryMethodSmooth = undefined;
export class ColorKeyframeTrack extends KeyframeTrack {}
ColorKeyframeTrack.prototype.ValueTypeName = 'color';
export class NumberKeyframeTrack extends KeyframeTrack {}
NumberKeyframeTrack.prototype.ValueTypeName = 'number';
export class QuaternionKeyframeTrack extends KeyframeTrack {
    InterpolantFactoryMethodLinear(result) { return new QuaternionLinearInterpolant(this.times, this.values, this.getValueSize(), result); }
}
QuaternionKeyframeTrack.prototype.ValueTypeName = 'quaternion';
QuaternionKeyframeTrack.prototype.InterpolantFactoryMethodSmooth = undefined;
export class StringKeyframeTrack extends KeyframeTrack { constructor(name, times, values) { super(name, times, values); } }
StringKeyframeTrack.prototype.ValueTypeName = 'string';
StringKeyframeTrack.prototype.ValueBufferType = Array;
StringKeyframeTrack.prototype.DefaultInterpolation = InterpolateDiscrete;
StringKeyframeTrack.prototype.InterpolantFactoryMethodLinear = undefined;
StringKeyframeTrack.prototype.InterpolantFactoryMethodSmooth = undefined;
export class VectorKeyframeTrack extends KeyframeTrack {}
VectorKeyframeTrack.prototype.ValueTypeName = 'vector';

const _TRACK_TYPES = {
    number: NumberKeyframeTrack, vector: VectorKeyframeTrack, quaternion: QuaternionKeyframeTrack,
    color: ColorKeyframeTrack, bool: BooleanKeyframeTrack, string: StringKeyframeTrack,
};

/// A JS track for a Rust one (`like`: a track whose `createInterpolant` a
/// glTF cubic track should keep).
function _trackFromWasm(w, type, like) {
    const C = _TRACK_TYPES[type] || NumberKeyframeTrack;
    const values = type === 'bool' ? Array.from(w.bools, (b) => b !== 0) : type === 'string' ? w.strings : w.values;
    const interp = w.interpolation;
    const track = C === BooleanKeyframeTrack || C === StringKeyframeTrack
        ? new C(w.name, w.times, values) : new C(w.name, w.times, values, interp || undefined);
    if (interp === 0 && type !== 'bool' && type !== 'string') {
        if (like?.createInterpolant?.isInterpolantFactoryMethodGLTFCubicSpline) track.createInterpolant = like.createInterpolant;
        else track._gltfCubic = true;
    }
    return track;
}

/// Procedural modifiers on keyframe tracks (noise / cycles / stepped /
/// limit), Blender-style — set `track.modifiers`; the mixer applies them in
/// Rust when it samples the track.
export class TrackModifier {
    static noise({ amplitude = 0.05, frequency = 1, seed = 0 } = {}) { return { kind: 'noise', amplitude, frequency, seed }; }
    static cycles({ before = Infinity, after = Infinity } = {}) { return { kind: 'cycles', before, after }; }
    static stepped({ stepSize = 0.1 } = {}) { return { kind: 'stepped', stepSize }; }
    static limit({ min = -Infinity, max = Infinity } = {}) { return { kind: 'limit', min, max }; }
    static _encode(modifiers) {
        const kinds = [], params = [];
        for (const m of modifiers || []) {
            switch (m.kind) {
                case 'noise': kinds.push(0); params.push(m.amplitude, m.frequency, m.seed); break;
                case 'cycles': kinds.push(1); params.push(m.before, m.after, 0); break;
                case 'stepped': kinds.push(2); params.push(m.stepSize, 0, 0); break;
                case 'limit': kinds.push(3); params.push(m.min, m.max, 0); break;
            }
        }
        return [Uint8Array.from(kinds), Float64Array.from(params)];
    }
    static remapTime(modifiers, t, track) {
        if (!modifiers?.length) return t;
        const [k, p] = TrackModifier._encode(modifiers);
        return trackModifierRemapTime(k, p, t, track?.times?.[track.times.length - 1] || 0);
    }
    static applyValue(modifiers, values, t) {
        if (!modifiers?.length) return values;
        const [k, p] = TrackModifier._encode(modifiers);
        const r = trackModifierApplyValue(k, p, Float64Array.from(values), t);
        for (let i = 0; i < values.length; i++) values[i] = r[i];
        return values;
    }
}

// ---- Clips ----

/// The Rust copy of a clip, made again when the clip or its tracks changed.
function _clipWasm(clip) {
    const tracks = clip.tracks;
    const c = clip._wasm;
    if (c && c.tracks === tracks && c.n === tracks.length && c.uuid === clip.uuid && c.name === clip.name
        && c.duration === clip.duration && c.blendMode === clip.blendMode) {
        let same = true;
        for (let i = 0; i < tracks.length && same; i++) {
            const t = tracks[i], s = c.sig[i];
            same = s[0] === t && s[1] === t.times && s[2] === t.values && s[3] === t._version && s[4] === t.name
                && s[5] === t.createInterpolant && s[6] === t.modifiers;
        }
        if (same) return c.w;
    }
    c?.w.free();
    const w = new WebAnimClip(String(clip.name), clip.duration, clip.blendMode, String(clip.uuid));
    for (const t of tracks) {
        const tw = t._toWasm ? t._toWasm() : KeyframeTrack.prototype._toWasm.call(t);
        w.addTrack(tw);
        tw.free();
    }
    clip._wasm = {
        w, tracks, n: tracks.length, uuid: clip.uuid, name: clip.name, duration: clip.duration, blendMode: clip.blendMode,
        sig: tracks.map((t) => [t, t.times, t.values, t._version, t.name, t.createInterpolant, t.modifiers]),
    };
    return w;
}

/// A JS clip for a Rust one (`like`: the clip whose tracks it came from).
function _clipFromWasm(w, ClipClass = AnimationClip, like = null) {
    const tracks = [];
    for (let i = 0; i < w.trackCount; i++) {
        const tw = w.track(i);
        const prior = like?.tracks.find((t) => t.name === tw.name);
        tracks.push(_trackFromWasm(tw, w.trackType(i), prior));
        tw.free();
    }
    const clip = new ClipClass(w.name, w.duration, tracks, w.blendMode);
    clip.uuid = w.uuid || undefined;
    return clip;
}

/// three.js `AnimationClip`.
export class AnimationClip {
    constructor(name = '', duration = -1, tracks = [], blendMode = NormalAnimationBlendMode) {
        this.name = name;
        this.tracks = tracks;
        this.duration = duration;
        this.blendMode = blendMode;
        this.uuid = MathUtils.generateUUID();
        if (this.duration < 0) this.resetDuration();
    }
    static parse(json) {
        const w = WebAnimClip.parse(JSON.stringify(json));
        const clip = _clipFromWasm(w, this);
        clip.uuid = json.uuid;
        w.free();
        return clip;
    }
    static toJSON(clip) {
        return JSON.parse(_clipWasm(clip).toJSON());
    }
    static CreateFromMorphTargetSequence(name, morphTargetSequence, fps, noLoop) {
        const w = WebAnimClip.createFromMorphTargetSequence(String(name), morphTargetSequence.map((m) => String(m.name)), fps, !!noLoop);
        const clip = _clipFromWasm(w, this);
        w.free();
        return clip;
    }
    static findByName(objectOrClipArray, name) {
        let clipArray = objectOrClipArray;
        if (!Array.isArray(objectOrClipArray)) {
            const o = objectOrClipArray;
            clipArray = o.geometry && o.geometry.animations || o.animations;
        }
        for (let i = 0; i < clipArray.length; i++) if (clipArray[i].name === name) return clipArray[i];
        return null;
    }
    static CreateClipsFromMorphTargetSequences(morphTargets, fps, noLoop) {
        const list = WebAnimClip.createClipsFromMorphTargetSequences(morphTargets.map((m) => String(m.name)), fps, !!noLoop);
        return list.map((w) => { const c = _clipFromWasm(w, this); w.free(); return c; });
    }
    static parseAnimation(animation, bones) {
        if (!animation) {
            console.error('THREE.AnimationClip: No animation in JSONLoader data.');
            return null;
        }
        const w = WebAnimClip.parseAnimation(JSON.stringify(animation), (bones || []).map((b) => String(b?.name ?? '')));
        if (!w) return null;
        const clip = _clipFromWasm(w, this);
        w.free();
        return clip;
    }
    resetDuration() {
        const w = _clipWasm(this);
        this.duration = w.resetDuration();
        this._wasm.duration = this.duration;
        return this;
    }
    trim() {
        for (let i = 0; i < this.tracks.length; i++) this.tracks[i].trim(0, this.duration);
        return this;
    }
    validate() {
        let valid = true;
        for (let i = 0; i < this.tracks.length; i++) valid = valid && this.tracks[i].validate();
        return valid;
    }
    optimize() {
        for (let i = 0; i < this.tracks.length; i++) this.tracks[i].optimize();
        return this;
    }
    clone() {
        const tracks = [];
        for (let i = 0; i < this.tracks.length; i++) tracks.push(this.tracks[i].clone());
        return new this.constructor(this.name, this.duration, tracks, this.blendMode);
    }
    toJSON() { return this.constructor.toJSON(this); }
}

// ---- AnimationUtils ----

export const AnimationUtils = {
    /// JS array-type plumbing, as three.js has it.
    convertArray(array, type, forceClone) {
        if (!array || (!forceClone && array.constructor === type)) return array;
        if (typeof type.BYTES_PER_ELEMENT === 'number') return new type(array);
        return Array.prototype.slice.call(array);
    },
    isTypedArray(object) { return ArrayBuffer.isView(object) && !(object instanceof DataView); },
    getKeyframeOrder(times) { return Array.from(getKeyframeOrder(Float64Array.from(times, Number))); },
    sortedArray(values, stride, order) {
        const nValues = values.length;
        const result = new values.constructor(nValues);
        for (let i = 0, dstOffset = 0; dstOffset !== nValues; ++i) {
            const srcOffset = order[i] * stride;
            for (let j = 0; j !== stride; ++j) result[dstOffset++] = values[srcOffset + j];
        }
        return result;
    },
    /// The AOS key flattener (keys may hold three.js math objects).
    flattenJSON(jsonKeys, times, values, valuePropertyName) {
        let i = 1, key = jsonKeys[0];
        while (key !== undefined && key[valuePropertyName] === undefined) key = jsonKeys[i++];
        if (key === undefined) return;
        let value = key[valuePropertyName];
        if (value === undefined) return;
        const push = Array.isArray(value) ? (v) => values.push(...v)
            : value.toArray !== undefined ? (v) => v.toArray(values, values.length)
            : (v) => values.push(v);
        do {
            value = key[valuePropertyName];
            if (value !== undefined) { times.push(key.time); push(value); }
            key = jsonKeys[i++];
        } while (key !== undefined);
    },
    subclip(sourceClip, name, startFrame, endFrame, fps = 30) {
        const w = _clipWasm(sourceClip).subclip(String(name), startFrame, endFrame, fps);
        const clip = _clipFromWasm(w, sourceClip.constructor, sourceClip);
        w.free();
        return clip;
    },
    makeClipAdditive(targetClip, referenceFrame = 0, referenceClip = targetClip, fps = 30) {
        const w = _clipWasm(targetClip);
        if (referenceClip === targetClip) w.makeClipAdditive(referenceFrame, fps);
        else w.makeClipAdditiveFrom(referenceFrame, _clipWasm(referenceClip), fps);
        // three.js edits the target's value arrays in place
        for (let i = 0; i < targetClip.tracks.length; i++) {
            const t = targetClip.tracks[i];
            const tw = w.track(i);
            if (t.ValueTypeName !== 'bool' && t.ValueTypeName !== 'string') {
                const v = tw.values;
                if (ArrayBuffer.isView(t.values) && t.values.length === v.length) t.values.set(v);
                else for (let k = 0; k < v.length; k++) t.values[k] = v[k];
            }
            tw.free();
        }
        targetClip.blendMode = AdditiveAnimationBlendMode;
        targetClip._wasm = null;
        w.free();
        return targetClip;
    },
};

// ---- PropertyBinding / PropertyMixer ----

/// three.js `PropertyBinding`: the parse and the node lookup in Rust, the
/// property on the JS object.
export class PropertyBinding {
    constructor(rootNode, path, parsedPath) {
        _animInit();
        this.path = path;
        this.parsedPath = parsedPath || PropertyBinding.parseTrackName(path);
        this.node = PropertyBinding.findNode(rootNode, this.parsedPath.nodeName);
        this.rootNode = rootNode;
        this._acc = undefined;
    }
    static create(root, path, parsedPath) {
        return root && root.isAnimationObjectGroup
            ? new PropertyBinding.Composite(root, path, parsedPath)
            : new PropertyBinding(root, path, parsedPath);
    }
    static sanitizeNodeName(name) { return sanitizeNodeName(String(name)); }
    static parseTrackName(trackName) {
        const r = parseTrackName(String(trackName));
        return { nodeName: r[1], objectName: r[2], objectIndex: r[3], propertyName: r[4], propertyIndex: r[5] };
    }
    static findNode(root, nodeName) {
        _animInit();
        const k = _animFindNode(_animKey(root), nodeName);
        return k === undefined ? null : _animObject(k);
    }
    bind() {
        const p = this.parsedPath;
        if (!this.node) this.node = PropertyBinding.findNode(this.rootNode, p.nodeName);
        this._acc = _animAccessor(this.node, p.objectName, p.objectIndex, p.propertyName, p.propertyIndex,
            'number', Infinity, this.path, p.nodeName);
        this.targetObject = this._acc ? this.node : undefined;
    }
    unbind() { this.node = null; this._acc = undefined; }
    getValue(buffer, offset) {
        if (this._acc === undefined) this.bind();
        const v = this._acc?.get();
        if (v === undefined) return;
        if (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string') buffer[offset] = v;
        else for (let i = 0; i < v.length; i++) buffer[offset + i] = v[i];
    }
    setValue(buffer, offset) {
        if (this._acc === undefined) this.bind();
        this._acc?.set(buffer, offset, this._acc.width);
    }
}
PropertyBinding.Composite = class Composite {
    constructor(targetGroup, path, optionalParsedPath) {
        const parsedPath = optionalParsedPath || PropertyBinding.parseTrackName(path);
        this._targetGroup = targetGroup;
        this._path = path;
        this._parsedPath = parsedPath;
        this._bindings = new Map();
    }
    _active() {
        const g = this._targetGroup;
        return g._objects.slice(g.nCachedObjects_).map((o) => {
            let b = this._bindings.get(o);
            if (!b) this._bindings.set(o, b = new PropertyBinding(o, this._path, this._parsedPath));
            return b;
        });
    }
    getValue(array, offset) { this._active()[0]?.getValue(array, offset); }
    setValue(array, offset) { for (const b of this._active()) b.setValue(array, offset); }
    bind() { for (const b of this._active()) b.bind(); }
    unbind() { for (const b of this._active()) b.unbind(); }
};

/// three.js `PropertyMixer` on its own (the mixer has its own, in Rust):
/// the accumulation and blending run in Rust; `buffer` is this side's copy,
/// passed over and back around each call.
export class PropertyMixer {
    constructor(binding, typeName, valueSize) {
        _animInit();
        this.binding = binding;
        this.valueSize = valueSize;
        const root = binding.rootNode ?? binding._targetGroup;
        this._w = new WebPropertyMixer(_animRootKey(root), String(binding.path), String(typeName), valueSize);
        this.buffer = typeName === 'string' || typeName === 'bool'
            ? new Array(valueSize * 5) : new Float64Array(valueSize * (typeName === 'quaternion' ? 6 : 5));
    }
    _call(f) {
        if (this.buffer instanceof Float64Array) this._w.setBuffer(this.buffer);
        const r = f(this._w);
        if (this.buffer instanceof Float64Array) this.buffer.set(this._w.buffer());
        _animFlush();
        return r;
    }
    get cumulativeWeight() { return this._w.cumulativeWeight; }
    get cumulativeWeightAdditive() { return this._w.cumulativeWeightAdditive; }
    get useCount() { return this._w.useCount; }
    set useCount(v) { this._w.useCount = v; }
    get referenceCount() { return this._w.referenceCount; }
    set referenceCount(v) { this._w.referenceCount = v; }
    accumulate(accuIndex, weight) { this._call((w) => w.accumulate(accuIndex, weight)); }
    accumulateAdditive(weight) { this._call((w) => w.accumulateAdditive(weight)); }
    apply(accuIndex) { this._call((w) => w.apply(accuIndex)); }
    saveOriginalState() { this._call((w) => w.saveOriginalState()); }
    restoreOriginalState() { this._call((w) => w.restoreOriginalState()); }
}

// ---- AnimationObjectGroup ----

/// three.js `AnimationObjectGroup`: the bookkeeping and per-object bindings
/// in Rust.
export class AnimationObjectGroup {
    constructor(...objects) {
        _animInit();
        this.isAnimationObjectGroup = true;
        this._uuid = MathUtils.generateUUID();
        this._w = new WebAnimationObjectGroup(_animKey(this), this._uuid, Uint32Array.from(objects, _animRootKey));
        const scope = this;
        this.stats = {
            objects: {
                get total() { return scope._w.stats()[0]; },
                get inUse() { return scope._w.stats()[1]; },
            },
            get bindingsPerObject() { return scope._w.stats()[2]; },
        };
    }
    get uuid() { return this._uuid; }
    set uuid(v) { this._uuid = v; this._w.uuid = String(v); }
    get nCachedObjects_() { return this._w.nCachedObjects; }
    get _objects() { return Array.from(this._w.objects(), _animObject); }
    add(...objects) { this._w.add(Uint32Array.from(objects, _animRootKey)); }
    remove(...objects) { this._w.remove(Uint32Array.from(objects, _animRootKey)); }
    uncache(...objects) { this._w.uncache(Uint32Array.from(objects, _animRootKey)); }
}

// ---- AnimationAction / AnimationMixer ----

const _A_TIME = 0, _A_TIME_SCALE = 1, _A_WEIGHT = 2, _A_ENABLED = 3, _A_PAUSED = 4, _A_LOOP = 5;
const _A_REPETITIONS = 6, _A_CLAMP = 7, _A_ZERO_START = 8, _A_ZERO_END = 9, _A_BLEND = 10;
const _A_EFFECTIVE_WEIGHT = 11, _A_EFFECTIVE_TIME_SCALE = 12, _A_RUNNING = 13, _A_SCHEDULED = 14;

/// Actions JS no longer holds are dropped on the Rust side too.
const _actionFinalizer = typeof FinalizationRegistry === 'function'
    ? new FinalizationRegistry(({ w, id }) => { try { w.forgetAction(id); } catch (_) { /* mixer gone */ } })
    : null;

/// three.js `AnimationAction`: state and logic in the Rust mixer; this holds
/// its number. Made by `mixer.clipAction`.
export class AnimationAction {
    constructor(mixer, clip, localRoot = null, blendMode = clip.blendMode, _id = undefined) {
        this._mixer = mixer;
        this._clip = clip;
        this._localRoot = localRoot;
        this._id = _id ?? mixer._make(clip, localRoot, blendMode);
    }
    get _w() { return this._mixer._w; }
    _get(f) { return this._w.actionGet(this._id, f); }
    _set(f, v) { this._w.actionSet(this._id, f, Number(v)); }
    get time() { return this._get(_A_TIME); }
    set time(v) { this._set(_A_TIME, v); }
    get timeScale() { return this._get(_A_TIME_SCALE); }
    set timeScale(v) { this._set(_A_TIME_SCALE, v); }
    get weight() { return this._get(_A_WEIGHT); }
    set weight(v) { this._set(_A_WEIGHT, v); }
    get enabled() { return this._get(_A_ENABLED) !== 0; }
    set enabled(v) { this._set(_A_ENABLED, v ? 1 : 0); }
    get paused() { return this._get(_A_PAUSED) !== 0; }
    set paused(v) { this._set(_A_PAUSED, v ? 1 : 0); }
    get loop() { return this._get(_A_LOOP); }
    set loop(v) { this._set(_A_LOOP, v); }
    get repetitions() { return this._get(_A_REPETITIONS); }
    set repetitions(v) { this._set(_A_REPETITIONS, v); }
    get clampWhenFinished() { return this._get(_A_CLAMP) !== 0; }
    set clampWhenFinished(v) { this._set(_A_CLAMP, v ? 1 : 0); }
    get zeroSlopeAtStart() { return this._get(_A_ZERO_START) !== 0; }
    set zeroSlopeAtStart(v) { this._set(_A_ZERO_START, v ? 1 : 0); }
    get zeroSlopeAtEnd() { return this._get(_A_ZERO_END) !== 0; }
    set zeroSlopeAtEnd(v) { this._set(_A_ZERO_END, v ? 1 : 0); }
    get blendMode() { return this._get(_A_BLEND); }
    set blendMode(v) { this._set(_A_BLEND, v); }
    get _startTime() { const t = this._w.startTime(this._id); return Number.isNaN(t) ? null : t; }
    play() { this._mixer._hold(this); this._w.play(this._id); _animFlush(); return this; }
    stop() { this._w.stop(this._id); _animFlush(); return this; }
    reset() { this._w.reset(this._id); return this; }
    isRunning() { return this._get(_A_RUNNING) !== 0; }
    isScheduled() { return this._get(_A_SCHEDULED) !== 0; }
    startAt(time) { this._w.startAt(this._id, time); return this; }
    setLoop(mode, repetitions) { this.loop = mode; this.repetitions = repetitions; return this; }
    setEffectiveWeight(weight) { this._w.setEffectiveWeight(this._id, weight); return this; }
    getEffectiveWeight() { return this._get(_A_EFFECTIVE_WEIGHT); }
    fadeIn(duration) { this._w.fadeIn(this._id, duration); return this; }
    fadeOut(duration) { this._w.fadeOut(this._id, duration); return this; }
    crossFadeFrom(fadeOutAction, duration, warp) { this._w.crossFadeFrom(this._id, fadeOutAction._id, duration, !!warp); return this; }
    crossFadeTo(fadeInAction, duration, warp) { fadeInAction.crossFadeFrom(this, duration, warp); return this; }
    stopFading() { this._w.stopFading(this._id); return this; }
    setEffectiveTimeScale(timeScale) { this._w.setEffectiveTimeScale(this._id, timeScale); return this; }
    getEffectiveTimeScale() { return this._get(_A_EFFECTIVE_TIME_SCALE); }
    setDuration(duration) { this._w.setDuration(this._id, duration); return this; }
    syncWith(action) { this._w.syncWith(this._id, action._id); return this; }
    halt(duration) { this._w.halt(this._id, duration); return this; }
    warp(startTimeScale, endTimeScale, duration) { this._w.warp(this._id, startTimeScale, endTimeScale, duration); return this; }
    stopWarping() { this._w.stopWarping(this._id); return this; }
    getMixer() { return this._mixer; }
    getClip() { return this._clip; }
    getRoot() { return this._localRoot || this._mixer._root; }
}

/// three.js `AnimationMixer`: runs in Rust; events reach listeners at the
/// point in the update three.js dispatches them.
export class AnimationMixer {
    constructor(root) {
        _animInit();
        this._root = root;
        this._w = new WebAnimationMixer(_animRootKey(root));
        /// Cached actions (held), and others JS may still hold.
        this._actions = new Map();
        this._loose = new Map();
        const scope = this;
        const s = (i) => scope._w.stats()[i];
        this.stats = {
            actions: { get total() { return s(0); }, get inUse() { return s(1); } },
            bindings: { get total() { return s(2); }, get inUse() { return s(3); } },
            controlInterpolants: { get total() { return s(4); }, get inUse() { return s(5); } },
        };
    }
    get time() { return this._w.time; }
    set time(v) { this._w.time = v; }
    get timeScale() { return this._w.timeScale; }
    set timeScale(v) { this._w.timeScale = v; }
    _rootArg(optionalRoot) {
        _animRootKey(this._root);
        return optionalRoot ? _animRootKey(optionalRoot) : NaN;
    }
    _make(clip, localRoot, blendMode) {
        const id = this._w.clipAction(_clipWasm(clip), this._rootArg(localRoot), blendMode ?? NaN);
        return id;
    }
    _hold(action) {
        if (this._actions.get(action._id) === action) return;
        this._loose.delete(action._id);
        this._actions.set(action._id, action);
    }
    /// After an uncache: actions the Rust mixer forgot are only weakly held.
    _reconcile() {
        const cached = new Set(this._w.cachedActions());
        for (const [id, a] of this._actions) {
            if (cached.has(id)) continue;
            this._actions.delete(id);
            this._loose.set(id, new WeakRef(a));
            _actionFinalizer?.register(a, { w: this._w, id });
        }
    }
    _action(id, clip, localRoot) {
        if (id < 0) return null;
        let a = this._actions.get(id) ?? this._loose.get(id)?.deref();
        if (!a) a = new AnimationAction(this, clip, localRoot, undefined, id);
        this._hold(a);
        return a;
    }
    _clipOfUuid(uuid) {
        for (const a of this._actions.values()) if (a._clip?.uuid === uuid) return a._clip;
        return null;
    }
    clipAction(clip, optionalRoot, blendMode) {
        const root = optionalRoot || this._root;
        const clipObject = typeof clip === 'string' ? AnimationClip.findByName(root, clip) : clip;
        const rootKey = this._rootArg(optionalRoot);
        const blend = blendMode === undefined ? NaN : blendMode;
        if (clipObject !== null) {
            return this._action(this._w.clipAction(_clipWasm(clipObject), rootKey, blend), clipObject, optionalRoot ?? null);
        }
        const id = this._w.clipActionByUuid(String(clip), rootKey, blend);
        return this._action(id, this._clipOfUuid(String(clip)), optionalRoot ?? null);
    }
    existingAction(clip, optionalRoot) {
        const root = optionalRoot || this._root;
        const clipObject = typeof clip === 'string' ? AnimationClip.findByName(root, clip) : clip;
        const uuid = clipObject ? clipObject.uuid : clip;
        const id = this._w.existingAction(String(uuid), this._rootArg(optionalRoot));
        if (id < 0) return null;
        return this._actions.get(id) ?? this._loose.get(id)?.deref() ?? null;
    }
    stopAllAction() { this._w.stopAllAction(); _animFlush(); return this; }
    _dispatchFrom(e) {
        const action = this._actions.get(e[1]) ?? this._loose.get(e[1])?.deref();
        if (e[0] === 1) this.dispatchEvent({ type: 'finished', action, direction: e[2] });
        else this.dispatchEvent({ type: 'loop', action, loopDelta: e[2] });
    }
    _run() {
        const w = this._w;
        for (let e = w.resume(); e[0] !== 0; e = w.resume()) this._dispatchFrom(e);
        _animFlush();
    }
    update(deltaTime) {
        const l = this._listeners;
        if (l && (l.finished?.length || l.loop?.length)) {
            this._w.beginUpdate(deltaTime);
            this._run();
        } else {
            this._w.updateAll(deltaTime);
            _animFlush();
        }
        return this;
    }
    setTime(timeInSeconds) {
        this._w.beginSetTime(timeInSeconds);
        this._run();
        return this;
    }
    getRoot() { return this._root; }
    uncacheClip(clip) { this._w.uncacheClip(String(clip.uuid)); _animFlush(); this._reconcile(); }
    uncacheRoot(root) { this._w.uncacheRoot(String(root.uuid)); _animFlush(); this._reconcile(); }
    uncacheAction(clip, optionalRoot) {
        const root = optionalRoot || this._root;
        const clipObject = typeof clip === 'string' ? AnimationClip.findByName(root, clip) : clip;
        const uuid = clipObject ? clipObject.uuid : clip;
        this._w.uncacheAction(String(uuid), this._rootArg(optionalRoot));
        _animFlush();
        this._reconcile();
    }
    addEventListener(type, listener) { EventDispatcherMixin.addEventListener.call(this, type, listener); }
    hasEventListener(type, listener) { return EventDispatcherMixin.hasEventListener.call(this, type, listener); }
    removeEventListener(type, listener) { EventDispatcherMixin.removeEventListener.call(this, type, listener); }
    dispatchEvent(event) { EventDispatcherMixin.dispatchEvent.call(this, event); }
}


// ---- Controls ----
/// A control moved the wasm camera: pose the JS camera to match (position,
/// up, and the rotation that looks at the wasm camera's target), so the
/// next render's `camera._sync()` pushes the same pose back.
function _cameraFromWasm(camera) {
    const w = camera?._w;
    if (!w?.readPosition) return;
    const p = w.readPosition(), t = w.readTarget();
    camera.position.set(p.x, p.y, p.z);
    const u = w.readUp?.();
    if (u) camera.up.set(u.x, u.y, u.z);
    camera.lookAt(t.x, t.y, t.z);
}
function _controlViewport(domElement) {
    const w = domElement?.clientWidth || (typeof window !== 'undefined' ? window.innerWidth : 800);
    const h = domElement?.clientHeight || (typeof window !== 'undefined' ? window.innerHeight : 600);
    return [Math.max(1, w), Math.max(1, h)];
}

export class OrbitControls {
    constructor(camera, domElement) {
        this._camera = camera;
        this.domElement = domElement;
        this.enabled = true;
        this.enableDamping = false;
        this.dampingFactor = 0.05;
        this.autoRotate = false;
        this.autoRotateSpeed = 2.0;
        // three.js-compatible public knobs (synced into wasm each update).
        this.target = camera?._lookAt ? camera._lookAt.clone() : new Vector3();
        // three.js's button and touch mapping, and pan in the screen plane
        // (MapControls swaps the buttons and pans over the ground).
        this.screenSpacePanning = true;
        this.mouseButtons = { LEFT: _MOUSE.ROTATE, MIDDLE: _MOUSE.DOLLY, RIGHT: _MOUSE.PAN };
        this.touches = { ONE: TOUCH.ROTATE, TWO: TOUCH.DOLLY_PAN };
        this.minDistance = 0.1;
        this.maxDistance = Infinity;
        this.minPolarAngle = 0;
        this.maxPolarAngle = Math.PI;
        this.minAzimuthAngle = -Infinity;
        this.maxAzimuthAngle = Infinity;
        this.enableZoom = true;
        this.enableRotate = true;
        this.enablePan = true;
        this.rotateSpeed = 1.0;
        this.zoomSpeed = 1.0;
        this.panSpeed = 1.0;
        this._rotating = false;
        this._panning = false;
        this._lastX = 0;
        this._lastY = 0;
        this._dragPointerId = null;
        this._lastT = (typeof performance !== 'undefined' ? performance.now() : 0);
        this._w = null;
        if (camera) {
            if (typeof camera._sync === 'function') camera._sync();
            // `_sync` carries the camera's matrix, whose y axis is not three.js's
            // world `up` once the camera tilts; the orbit turns about `up`.
            camera._w.setUp?.(camera.up.x, camera.up.y, camera.up.z);
            if (camera._lookAt) camera._w.lookAt(camera._lookAt.x, camera._lookAt.y, camera._lookAt.z);
            this._w = new WebOrbitControls(camera._w);
            if (typeof this._w.setTarget === 'function') {
                this._w.setTarget(this.target.x, this.target.y, this.target.z);
            }
            camera._orbitControlled = true;
        }

        this._onContextMenu = (e) => e.preventDefault();
        this._onPointerDown = (e) => {
            if (!this.enabled) return;
            const root = this.domElement;
            if (!root) return;
            // three.js's `onMouseDown`: the button's action, with
            // ctrl / meta / shift swapping rotate and pan.
            const action = e.button === 0 ? this.mouseButtons.LEFT
                : e.button === 1 ? this.mouseButtons.MIDDLE
                : e.button === 2 ? this.mouseButtons.RIGHT : -1;
            const swap = e.ctrlKey || e.metaKey || e.shiftKey;
            let rotate = false, pan = false;
            if (action === _MOUSE.ROTATE) { if (swap) pan = this.enablePan; else rotate = this.enableRotate; }
            else if (action === _MOUSE.PAN) { if (swap) rotate = this.enableRotate; else pan = this.enablePan; }
            if (!rotate && !pan) return;
            e.preventDefault();
            this._dragPointerId = e.pointerId;
            this._rotating = rotate;
            this._panning = pan;
            this.dispatchEvent({ type: 'start' });
            this._lastX = e.clientX;
            this._lastY = e.clientY;
            root.setPointerCapture?.(e.pointerId);
            const doc = root.ownerDocument || (typeof document !== 'undefined' ? document : null);
            if (doc) {
                doc.addEventListener('pointermove', this._onPointerMove);
                doc.addEventListener('pointerup', this._onPointerUp);
                doc.addEventListener('pointercancel', this._onPointerUp);
            }
        };
        this._onPointerMove = (e) => {
            if (!this.enabled || e.pointerId !== this._dragPointerId) return;
            if (!this._rotating && !this._panning) return;
            const dx = e.clientX - this._lastX;
            const dy = e.clientY - this._lastY;
            this._lastX = e.clientX;
            this._lastY = e.clientY;
            if (dx || dy) this.update(dx, dy, 0, this._rotating, this._panning);
        };
        this._onPointerUp = (e) => {
            if (this._dragPointerId != null && e.pointerId !== this._dragPointerId) return;
            this._endDrag();
        };
        this._onWheel = (e) => {
            if (!this.enabled || !this.enableZoom) return;
            e.preventDefault();
            this.update(0, 0, e.deltaY, false, false);
        };

        if (domElement?.addEventListener) {
            domElement.addEventListener('contextmenu', this._onContextMenu);
            domElement.addEventListener('pointerdown', this._onPointerDown);
            domElement.addEventListener('wheel', this._onWheel, { passive: false });
            if (!domElement.style.touchAction) domElement.style.touchAction = 'none';
        }
    }
    _endDrag() {
        const root = this.domElement;
        const dragging = this._dragPointerId != null;
        if (root && dragging) {
            root.releasePointerCapture?.(this._dragPointerId);
            const doc = root.ownerDocument || (typeof document !== 'undefined' ? document : null);
            if (doc) {
                doc.removeEventListener('pointermove', this._onPointerMove);
                doc.removeEventListener('pointerup', this._onPointerUp);
                doc.removeEventListener('pointercancel', this._onPointerUp);
            }
        }
        this._dragPointerId = null;
        this._rotating = false;
        this._panning = false;
        if (dragging) this.dispatchEvent({ type: 'end' });
    }
    dispose() {
        this._endDrag();
        const root = this.domElement;
        if (root?.removeEventListener) {
            root.removeEventListener('contextmenu', this._onContextMenu);
            root.removeEventListener('pointerdown', this._onPointerDown);
            root.removeEventListener('wheel', this._onWheel);
        }
        if (this._camera) this._camera._orbitControlled = false;
    }
    _syncLimits() {
        if (typeof this._w.setLimits === 'function') {
            const maxD = Number.isFinite(this.maxDistance) ? this.maxDistance : 1e6;
            this._w.setLimits(
                Math.max(0, this.minDistance),
                Math.max(this.minDistance, maxD),
                this.minPolarAngle,
                this.maxPolarAngle,
                this.rotateSpeed,
                this.zoomSpeed,
                this.panSpeed,
                this.enableDamping ? (this.dampingFactor ?? 0.05) : 0,
            );
        }
        if (typeof this._w.setTarget === 'function') {
            this._w.setTarget(this.target.x, this.target.y, this.target.z);
        }
        this._w.setScreenSpacePanning?.(this.screenSpacePanning !== false);
    }
    _syncMotion() {
        if (typeof this._w.setMotion === 'function') {
            const radPerSec = (this.autoRotateSpeed * Math.PI) / 30;
            this._w.setMotion(
                !!this.enableDamping,
                this.dampingFactor ?? 0.05,
                !!this.autoRotate,
                radPerSec,
            );
        }
    }
    update(dx = 0, dy = 0, wheel = 0, rotating = false, panning = false) {
        if (!this._w) return false;
        const p0 = this._camera.position, x0 = p0.x, y0 = p0.y, z0 = p0.z;
        const t0 = this.target, tx = t0.x, ty = t0.y, tz = t0.z;
        // three.js re-reads the camera's position and `target` at every
        // update, so either set from outside (an animation, a saved view)
        // moves the orbit. Push them into wasm when they changed since the
        // controls last wrote them.
        const s = this._synced;
        if (!s || s[0] !== x0 || s[1] !== y0 || s[2] !== z0 || s[3] !== tx || s[4] !== ty || s[5] !== tz) {
            // The pose as three.js holds it — position, target and the
            // world `up` the orbit turns about — not the camera's matrix,
            // whose own y axis is not `up` once it tilts.
            const up = this._camera.up;
            this._camera._w.setUp?.(up.x, up.y, up.z);
            this._camera._w.setPosition?.(x0, y0, z0);
            this._camera._w.lookAt?.(tx, ty, tz);
            this._camera._lookAt?.set(tx, ty, tz);
            this._w.setTarget?.(tx, ty, tz);
            this._w.reseedFromCamera?.(this._camera._w);
        }
        this._syncLimits();
        this._syncMotion();
        const [w, h] = _controlViewport(this.domElement);
        const now = (typeof performance !== 'undefined' ? performance.now() : this._lastT + 16.6);
        const dt = Math.min(0.1, Math.max(0.0, (now - this._lastT) / 1000));
        this._lastT = now;
        if (typeof this._w.updateDt === 'function') {
            this._w.updateDt(this._camera._w, dx, dy, wheel, rotating, panning, w, h, dt || 1 / 60);
        } else {
            this._w.update(this._camera._w, dx, dy, wheel, rotating, panning, w, h);
        }
        this._syncFromWasm();
        const p = this._camera.position, t = this.target;
        const changed = p.x !== x0 || p.y !== y0 || p.z !== z0 || t.x !== tx || t.y !== ty || t.z !== tz;
        if (changed) this.dispatchEvent({ type: 'change' });
        return changed;
    }
    /// The wasm camera's new pose onto the JS camera: position, and the
    /// rotation that looks at the orbit target (its `lookAt`).
    _syncFromWasm() {
        _cameraFromWasm(this._camera);
        const t = this._camera._w.readTarget();
        this.target.set(t.x, t.y, t.z);
        const p = this._camera.position;
        this._synced = [p.x, p.y, p.z, t.x, t.y, t.z];
    }
    resetFromCamera() {
        if (typeof this._camera._sync === 'function') this._camera._sync();
        if (this._camera._lookAt) this.target.copy(this._camera._lookAt);
        if (typeof this._w?.reseedFromCamera === 'function') {
            this._w.reseedFromCamera(this._camera._w);
        } else {
            this._w = new WebOrbitControls(this._camera._w);
        }
    }
}

// ---------------------------------------------------------------------------
// Cinematic camera helpers — additive; do NOT replace OrbitControls /
// AnimationMixer. Mirrors threers-animation SpeedRamp / CameraPath / shots.
// ---------------------------------------------------------------------------
const _CAM_EASINGS = {
    linear: (t) => t,
    quadIn: (t) => t * t,
    quadOut: (t) => t * (2 - t),
    cubicIn: (t) => t * t * t,
    cubicOut: (t) => 1 - Math.pow(1 - t, 3),
    cubicInOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    smooth: (t) => t * t * (3 - 2 * t),
    smootherstep: (t) => t * t * t * (t * (t * 6 - 15) + 10),
};

/** Map linear 0..1 → shaped parameter (ease / cruise / smootherstep). */
export function speedRamp(kind, t, opts = {}) {
    t = Math.min(1, Math.max(0, t));
    if (kind == null || kind === 'linear') return t;
    if (typeof kind === 'function') return kind(t);
    if (kind === 'smooth') return _CAM_EASINGS.smooth(t);
    if (kind === 'smoother' || kind === 'smootherstep') return _CAM_EASINGS.smootherstep(t);
    if (kind === 'inOut' || kind === 'ramp') {
        const a = Math.min(1, Math.max(0, opts.easeIn ?? 0.25));
        const b = Math.min(1 - a, Math.max(0, opts.easeOut ?? 0.25));
        const mid = 1 - a - b;
        if (t <= a) {
            if (a <= 1e-8) return 0;
            const u = t / a;
            return a * _CAM_EASINGS.smooth(u);
        }
        if (t >= 1 - b) {
            if (b <= 1e-8) return 1;
            const u = (t - (1 - b)) / b;
            return a + mid + b * _CAM_EASINGS.smooth(u);
        }
        return a + mid * ((t - a) / Math.max(1e-8, mid));
    }
    const ease = _CAM_EASINGS[kind] || _CAM_EASINGS.cubicInOut;
    return ease(t);
}

function _v3(p) {
    if (p instanceof Vector3) return p.clone();
    if (Array.isArray(p)) return new Vector3(p[0], p[1], p[2]);
    if (p && typeof p === 'object') return new Vector3(p.x ?? 0, p.y ?? 0, p.z ?? 0);
    return new Vector3();
}

function _sampleLinear(points, u) {
    const n = points.length - 1;
    const f = u * n;
    const i = Math.min(n - 1, Math.floor(f));
    const t = f - i;
    return new Vector3().lerpVectors(points[i], points[i + 1], t);
}

function _sampleBezier(points, u) {
    const segments = Math.floor((points.length - 1) / 3);
    if (segments <= 0) return _sampleLinear(points, u);
    const f = u * segments;
    const s = Math.min(segments - 1, Math.floor(f));
    const t = f - s;
    const i = s * 3;
    const p0 = points[i], c0 = points[i + 1], c1 = points[i + 2], p1 = points[i + 3];
    const omt = 1 - t;
    return new Vector3(
        omt * omt * omt * p0.x + 3 * omt * omt * t * c0.x + 3 * omt * t * t * c1.x + t * t * t * p1.x,
        omt * omt * omt * p0.y + 3 * omt * omt * t * c0.y + 3 * omt * t * t * c1.y + t * t * t * p1.y,
        omt * omt * omt * p0.z + 3 * omt * omt * t * c0.z + 3 * omt * t * t * c1.z + t * t * t * p1.z,
    );
}

function _sampleCatmull(points, u) {
    const n = points.length - 1;
    const f = u * n;
    const i = Math.min(n - 1, Math.floor(f));
    const t = f - i;
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[Math.min(points.length - 1, i + 1)];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    const t2 = t * t, t3 = t2 * t;
    return new Vector3(
        (-0.5 * t3 + t2 - 0.5 * t) * p0.x + (1.5 * t3 - 2.5 * t2 + 1) * p1.x + (-1.5 * t3 + 2 * t2 + 0.5 * t) * p2.x + (0.5 * t3 - 0.5 * t2) * p3.x,
        (-0.5 * t3 + t2 - 0.5 * t) * p0.y + (1.5 * t3 - 2.5 * t2 + 1) * p1.y + (-1.5 * t3 + 2 * t2 + 0.5 * t) * p2.y + (0.5 * t3 - 0.5 * t2) * p3.y,
        (-0.5 * t3 + t2 - 0.5 * t) * p0.z + (1.5 * t3 - 2.5 * t2 + 1) * p1.z + (-1.5 * t3 + 2 * t2 + 0.5 * t) * p2.z + (0.5 * t3 - 0.5 * t2) * p3.z,
    );
}

function _sampleCurve(points, kind, u) {
    if (!points || points.length === 0) return new Vector3();
    if (points.length === 1) return points[0].clone();
    u = Math.min(1, Math.max(0, u));
    if (kind === 'bezier') return _sampleBezier(points, u);
    if (kind === 'linear') return _sampleLinear(points, u);
    return _sampleCatmull(points, u);
}

/** World-space eye path (+ optional interest) with speed ramp + arc-length. */
export class CameraPath {
    constructor({
        points = [],
        kind = 'catmull',
        interest = [],
        interestKind = 'catmull',
        lookAhead = 1,
        fixedTarget = null,
        ramp = 'smoother',
        rampOpts = {},
        fov = null,
    } = {}) {
        this.points = points.map(_v3);
        this.kind = kind;
        this.interest = interest.map(_v3);
        this.interestKind = interestKind;
        this.lookAhead = lookAhead;
        this.fixedTarget = fixedTarget ? _v3(fixedTarget) : null;
        this.ramp = ramp;
        this.rampOpts = rampOpts;
        this.fov = fov;
        this._lengths = [];
        this.totalLength = 0;
        this.rebuildArcLength();
    }
    static catmullRom(points, opts = {}) {
        return new CameraPath({ ...opts, points, kind: 'catmull' });
    }
    static bezier(points, opts = {}) {
        return new CameraPath({ ...opts, points, kind: 'bezier' });
    }
    static linear(points, opts = {}) {
        return new CameraPath({ ...opts, points, kind: 'linear' });
    }
    withInterest(points, kind) {
        this.interest = points.map(_v3);
        if (kind) this.interestKind = kind;
        return this;
    }
    withRamp(ramp, rampOpts = {}) {
        this.ramp = ramp;
        this.rampOpts = rampOpts;
        return this;
    }
    withFixedTarget(target) {
        this.fixedTarget = _v3(target);
        this.lookAhead = 0;
        return this;
    }
    rebuildArcLength() {
        const SAMPLES = 64;
        this._lengths = [0];
        if (this.points.length < 2) {
            this.totalLength = 0;
            return this;
        }
        let prev = _sampleCurve(this.points, this.kind, 0);
        let acc = 0;
        for (let i = 1; i <= SAMPLES; i++) {
            const p = _sampleCurve(this.points, this.kind, i / SAMPLES);
            acc += p.distanceTo(prev);
            this._lengths.push(acc);
            prev = p;
        }
        this.totalLength = acc;
        return this;
    }
    _arcParameter(t) {
        if (this._lengths.length < 2 || this.totalLength <= 1e-8) return t;
        const want = t * this.totalLength;
        let lo = 0, hi = this._lengths.length - 1;
        while (lo + 1 < hi) {
            const mid = (lo + hi) >> 1;
            if (this._lengths[mid] < want) lo = mid;
            else hi = mid;
        }
        const l0 = this._lengths[lo], l1 = this._lengths[hi];
        const local = (want - l0) / Math.max(1e-8, l1 - l0);
        const u0 = lo / (this._lengths.length - 1);
        const u1 = hi / (this._lengths.length - 1);
        return u0 + (u1 - u0) * local;
    }
    sampleEye(t) {
        const u = this._arcParameter(speedRamp(this.ramp, t, this.rampOpts));
        return _sampleCurve(this.points, this.kind, u);
    }
    sampleTarget(t) {
        t = Math.min(1, Math.max(0, t));
        const u = this._arcParameter(speedRamp(this.ramp, t, this.rampOpts));
        if (this.interest.length >= 2) {
            return _sampleCurve(this.interest, this.interestKind, u);
        }
        if (this.fixedTarget) return this.fixedTarget.clone();
        if (this.lookAhead > 0 && this.totalLength > 1e-6) {
            const ahead = Math.min(1, u + this.lookAhead / this.totalLength);
            return _sampleCurve(this.points, this.kind, ahead);
        }
        const eye = _sampleCurve(this.points, this.kind, u);
        const next = _sampleCurve(this.points, this.kind, Math.min(1, u + 0.01));
        const dir = next.clone().sub(eye);
        if (dir.lengthSq() < 1e-10) return eye.clone().add(new Vector3(0, 0, -1));
        return eye.clone().add(dir.normalize());
    }
}

function _applyCamView(cam, pos, look, fov) {
    cam.position.copy(pos);
    cam.lookAt?.(look);
    if (cam.isPerspectiveCamera && fov != null) {
        cam.fov = fov;
        cam.updateProjectionMatrix?.();
    }
    cam._sync?.();
}

/**
 * Additive cinematic helper for a three.js-shaped PerspectiveCamera.
 * @example
 * const anim = new THREE.CameraAnimator(camera);
 * anim.flyTo({ position, target, duration: 2, ramp: 'inOut', easeIn: 0.2, easeOut: 0.3 });
 * anim.followPath({ path, duration: 4 });
 * anim.update(dt);
 */
export class CameraAnimator {
    constructor(camera) {
        this.camera = camera;
        this._mode = 'idle'; // idle | fly | path
        this._from = { pos: new Vector3(), target: new Vector3(), fov: 50 };
        this._to = { pos: new Vector3(), target: new Vector3(), fov: 50 };
        this._path = null;
        this._elapsed = 0;
        this._duration = 0;
        this._ramp = 'cubicInOut';
        this._rampOpts = {};
        this._active = false;
    }
    stop() {
        this._active = false;
        this._mode = 'idle';
        return this;
    }
    flyTo({
        position, target, fov, duration = 1,
        easing, ramp, easeIn, easeOut,
    } = {}) {
        const cam = this.camera;
        this._from.pos.copy(cam.position);
        this._from.target.copy(cam._lookAt || new Vector3());
        this._from.fov = cam.fov ?? cam._fov ?? 50;
        this._to.pos.copy(position ? _v3(position) : cam.position);
        this._to.target.copy(target ? _v3(target) : this._from.target);
        this._to.fov = fov ?? this._from.fov;
        this._duration = Math.max(0, duration);
        this._elapsed = 0;
        this._ramp = ramp || easing || 'cubicInOut';
        this._rampOpts = { easeIn, easeOut };
        this._mode = 'fly';
        this._active = true;
        return this;
    }
    /** Spherical orbit around the current look-at by azimuth/elevation deltas (radians). */
    orbitBy({
        azimuth = 0, elevation = 0, radiusDelta = 0,
        duration = 1, easing, ramp, easeIn, easeOut, fov,
    } = {}) {
        const cam = this.camera;
        const eye = cam.position.clone();
        const tgt = (cam._lookAt || new Vector3()).clone();
        const off = eye.clone().sub(tgt);
        const r0 = Math.max(1e-4, off.length());
        const phi0 = Math.acos(Math.min(1, Math.max(-1, off.y / r0)));
        const theta0 = Math.atan2(off.x, off.z);
        const r1 = Math.max(1e-4, r0 + radiusDelta);
        const phi1 = Math.min(Math.PI - 1e-3, Math.max(1e-3, phi0 + elevation));
        const theta1 = theta0 + azimuth;
        const toPos = new Vector3(
            tgt.x + r1 * Math.sin(phi1) * Math.sin(theta1),
            tgt.y + r1 * Math.cos(phi1),
            tgt.z + r1 * Math.sin(phi1) * Math.cos(theta1),
        );
        return this.flyTo({
            position: toPos, target: tgt, fov, duration,
            easing, ramp, easeIn, easeOut,
        });
    }
    /** Retarget look-at; set `holdEye: true` to keep eye and rebuild framing. */
    lookAt({ target, duration = 1, holdEye = false, easing, ramp, easeIn, easeOut } = {}) {
        const cam = this.camera;
        const tgt = _v3(target);
        if (holdEye) {
            return this.flyTo({
                position: cam.position.clone(), target: tgt, duration,
                easing, ramp, easeIn, easeOut,
            });
        }
        return this.flyTo({
            position: cam.position.clone(), target: tgt, duration,
            easing, ramp, easeIn, easeOut,
        });
    }
    /** Animate FOV only (push-in / pull-out). */
    setFov({ fov, duration = 1, easing, ramp, easeIn, easeOut } = {}) {
        const cam = this.camera;
        return this.flyTo({
            position: cam.position.clone(),
            target: (cam._lookAt || new Vector3()).clone(),
            fov, duration, easing, ramp, easeIn, easeOut,
        });
    }
    /**
     * Dolly-zoom / Vertigo: move along view axis while compensating FOV so
     * the subject framing stays approximately constant.
     */
    vertigo({
        targetFov, duration = 1, easing, ramp, easeIn, easeOut,
    } = {}) {
        const cam = this.camera;
        const eye = cam.position.clone();
        const tgt = (cam._lookAt || new Vector3()).clone();
        const fov0 = cam.fov ?? cam._fov ?? 50;
        const fov1 = targetFov ?? fov0;
        const dist0 = Math.max(1e-4, eye.distanceTo(tgt));
        // similar triangles: dist * tan(fov/2) ≈ constant framing size
        const half0 = (fov0 * Math.PI) / 360;
        const half1 = (fov1 * Math.PI) / 360;
        const dist1 = dist0 * Math.tan(half0) / Math.max(1e-6, Math.tan(half1));
        const dir = eye.clone().sub(tgt).normalize();
        const toPos = tgt.clone().add(dir.multiplyScalar(dist1));
        return this.flyTo({
            position: toPos, target: tgt, fov: fov1, duration,
            easing, ramp, easeIn, easeOut,
        });
    }
    /** Follow a {@link CameraPath} (or `{ points, kind, … }` options). */
    followPath({ path, duration = 1, ramp, easeIn, easeOut } = {}) {
        this._path = path instanceof CameraPath ? path : new CameraPath(path || {});
        if (ramp != null) this._path.withRamp(ramp, { easeIn, easeOut });
        this._duration = Math.max(0, duration);
        this._elapsed = 0;
        this._ramp = this._path.ramp;
        this._rampOpts = this._path.rampOpts || {};
        this._mode = 'path';
        this._active = true;
        return this;
    }
    get isActive() { return this._active; }
    update(dt) {
        if (!this._active) return false;
        this._elapsed += Math.max(0, dt);
        let u = this._duration <= 0 ? 1 : Math.min(1, this._elapsed / this._duration);
        const cam = this.camera;
        if (this._mode === 'path' && this._path) {
            const eye = this._path.sampleEye(u);
            const look = this._path.sampleTarget(u);
            const fov = this._path.fov != null ? this._path.fov : (cam.fov ?? cam._fov);
            _applyCamView(cam, eye, look, fov);
        } else {
            const s = speedRamp(this._ramp, u, this._rampOpts);
            const pos = new Vector3().lerpVectors(this._from.pos, this._to.pos, s);
            const look = new Vector3().lerpVectors(this._from.target, this._to.target, s);
            const fov = this._from.fov + (this._to.fov - this._from.fov) * s;
            _applyCamView(cam, pos, look, fov);
        }
        if (u >= 1) {
            this._active = false;
            this._mode = 'idle';
        }
        return this._active;
    }
}

/**
 * Editorial shot list: hold / fly / path clips with cut or blend transitions.
 * Apply with `timeline.apply(camera)` each frame after `timeline.update(dt)`.
 */
export class ShotTimeline {
    constructor() {
        this.shots = [];
        this.time = 0;
        this.looping = true;
    }
    push(shot) {
        this.shots.push(shot);
        return this.shots.length - 1;
    }
    hold(name, { position, target, fov = 50, duration = 1, transition } = {}) {
        return this.push({
            name, duration, kind: 'hold',
            position: _v3(position), target: _v3(target), fov,
            transition: transition || { type: 'cut' },
        });
    }
    fly(name, {
        from, to, duration = 1, ramp = 'inOut', easeIn = 0.2, easeOut = 0.3, transition,
    } = {}) {
        return this.push({
            name, duration, kind: 'fly',
            from: {
                position: _v3(from.position),
                target: _v3(from.target),
                fov: from.fov ?? 50,
            },
            to: {
                position: _v3(to.position),
                target: _v3(to.target),
                fov: to.fov ?? from.fov ?? 50,
            },
            ramp, easeIn, easeOut,
            transition: transition || { type: 'cut' },
        });
    }
    path(name, { path, duration = 1, transition } = {}) {
        return this.push({
            name, duration, kind: 'path',
            path: path instanceof CameraPath ? path : new CameraPath(path || {}),
            transition: transition || { type: 'cut' },
        });
    }
    duration() {
        return this.shots.reduce((s, sh) => s + Math.max(0, sh.duration || 0), 0);
    }
    seek(time) {
        const total = this.duration();
        if (total <= 0) { this.time = 0; return this; }
        let t = Math.max(0, time);
        if (this.looping) t %= total;
        else t = Math.min(total, t);
        this.time = t;
        return this;
    }
    update(dt) {
        if (Number.isFinite(dt)) this.seek(this.time + dt);
        return this;
    }
    _shotAt(time) {
        let acc = 0;
        for (let i = 0; i < this.shots.length; i++) {
            const end = acc + this.shots[i].duration;
            if (time < end || i + 1 === this.shots.length) {
                return { index: i, local: Math.max(0, time - acc) };
            }
            acc = end;
        }
        return { index: 0, local: 0 };
    }
    _sampleShot(shot, local) {
        const u = shot.duration <= 0 ? 1 : Math.min(1, Math.max(0, local / shot.duration));
        if (shot.kind === 'hold') {
            return {
                position: shot.position.clone(),
                target: shot.target.clone(),
                fov: shot.fov,
            };
        }
        if (shot.kind === 'path') {
            return {
                position: shot.path.sampleEye(u),
                target: shot.path.sampleTarget(u),
                fov: shot.path.fov ?? 50,
            };
        }
        // fly
        const s = speedRamp(shot.ramp, u, { easeIn: shot.easeIn, easeOut: shot.easeOut });
        return {
            position: new Vector3().lerpVectors(shot.from.position, shot.to.position, s),
            target: new Vector3().lerpVectors(shot.from.target, shot.to.target, s),
            fov: shot.from.fov + (shot.to.fov - shot.from.fov) * s,
        };
    }
    sample() {
        if (!this.shots.length) {
            return { position: new Vector3(), target: new Vector3(), fov: 50, shotIndex: -1, shotName: '' };
        }
        const { index, local } = this._shotAt(this.time);
        const cur = this.shots[index];
        let pose = this._sampleShot(cur, local);
        if (index > 0 && cur.transition && cur.transition.type === 'blend') {
            const dur = Math.min(cur.duration, Math.max(0, cur.transition.duration ?? 0.5));
            if (dur > 0 && local < dur) {
                const prev = this.shots[index - 1];
                const prevPose = this._sampleShot(prev, prev.duration);
                const bu = speedRamp(
                    cur.transition.ramp || 'cubicInOut',
                    local / dur,
                    { easeIn: cur.transition.easeIn, easeOut: cur.transition.easeOut },
                );
                pose = {
                    position: new Vector3().lerpVectors(prevPose.position, pose.position, bu),
                    target: new Vector3().lerpVectors(prevPose.target, pose.target, bu),
                    fov: prevPose.fov + (pose.fov - prevPose.fov) * bu,
                };
            }
        }
        pose.shotIndex = index;
        pose.shotName = cur.name || '';
        return pose;
    }
    apply(camera) {
        const pose = this.sample();
        _applyCamView(camera, pose.position, pose.target, pose.fov);
        return pose;
    }
}

// ---------------------------------------------------------------------------
// General motion primitives (mirror threers-animation Tween / Spring / Timeline)
// ---------------------------------------------------------------------------
const _EASING_FNS = {
    linear: (t) => t,
    quadIn: (t) => t * t,
    quadOut: (t) => t * (2 - t),
    quadInOut: (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
    cubicIn: (t) => t * t * t,
    cubicOut: (t) => { const u = 1 - t; return 1 - u * u * u; },
    cubicInOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    smooth: (t) => t * t * (3 - 2 * t),
    smoother: (t) => t * t * t * (t * (t * 6 - 15) + 10),
    backOut: (t) => { const c = 1.70158; const u = t - 1; return 1 + u * u * ((c + 1) * u + c); },
};

function _ease(name, t) {
    if (typeof name === 'function') return name(t);
    return (_EASING_FNS[name] || _EASING_FNS.linear)(Math.min(1, Math.max(0, t)));
}

function _lerpScalar(a, b, t) { return a + (b - a) * t; }
function _lerpAny(a, b, t) {
    if (typeof a === 'number') return _lerpScalar(a, b, t);
    if (a?.isQuaternion || (a?.w != null && typeof a.slerp === 'function')) {
        const out = a.clone?.() || new Quaternion(a.x, a.y, a.z, a.w);
        return out.slerp(b, t);
    }
    if (a?.x != null && b?.x != null) {
        return new Vector3(
            _lerpScalar(a.x, b.x, t),
            _lerpScalar(a.y, b.y, t),
            _lerpScalar(a.z ?? 0, b.z ?? 0, t),
        );
    }
    return b;
}

/** A→B over a known duration. */
export class Tween {
    constructor(from, to, duration = 1) {
        this.from = from;
        this.to = to;
        this.duration = Math.max(0, duration);
        this.delay = 0;
        this.easing = 'cubicOut';
        this.repeat = 0;
        this.yoyo = false;
        this.elapsed = 0;
        this._value = from?.clone?.() ?? from;
        this.finished = false;
        this.onUpdate = null;
        this.onComplete = null;
    }
    setEasing(e) { this.easing = e; return this; }
    setDelay(d) { this.delay = Math.max(0, d); return this; }
    setRepeat(n, yoyo = false) { this.repeat = n; this.yoyo = yoyo; return this; }
    value() { return this._value; }
    isFinished() { return this.finished; }
    update(dt) {
        if (this.finished || !Number.isFinite(dt)) return this._value;
        this.elapsed += Math.max(0, dt);
        const active = this.elapsed - this.delay;
        if (active < 0) return this._value;
        const dur = this.duration || 1e-8;
        let cycle = Math.floor(active / dur);
        let u = (active % dur) / dur;
        const maxCycles = this.repeat === Infinity ? Infinity : (this.repeat + 1);
        if (cycle >= maxCycles) {
            u = 1;
            this.finished = true;
            cycle = maxCycles - 1;
        }
        if (this.yoyo && (cycle % 2 === 1)) u = 1 - u;
        const e = _ease(this.easing, u);
        this._value = _lerpAny(this.from, this.to, e);
        this.onUpdate?.(this._value, e);
        if (this.finished) this.onComplete?.(this._value);
        return this._value;
    }
}

/** Damped spring chasing a moving target (no fixed duration). */
export class Spring {
    constructor(value, angularFrequency = 8, dampingRatio = 1) {
        this.value = typeof value === 'number' ? value : (value.clone?.() || value);
        this.velocity = typeof value === 'number' ? 0 : new Vector3();
        this.target = typeof value === 'number' ? value : (value.clone?.() || value);
        this.angularFrequency = Math.max(0, angularFrequency);
        this.dampingRatio = Math.max(0, dampingRatio);
        this.restThreshold = 1e-3;
    }
    static criticallyDamped(value, frequency = 8) {
        return new Spring(value, frequency, 1);
    }
    setTarget(t) {
        this.target = typeof t === 'number' ? t : (t.clone?.() || t);
        return this;
    }
    isSettled() {
        if (typeof this.value === 'number') {
            return Math.abs(this.value - this.target) < this.restThreshold
                && Math.abs(this.velocity) < this.restThreshold;
        }
        const d = this.value.distanceTo?.(this.target) ?? 0;
        const v = this.velocity.length?.() ?? 0;
        return d < this.restThreshold && v < this.restThreshold;
    }
    update(dt) {
        if (!Number.isFinite(dt) || dt <= 0) return this.value;
        const w = this.angularFrequency;
        const z = this.dampingRatio;
        if (typeof this.value === 'number') {
            const x = this.value - this.target;
            const a = -2 * z * w * this.velocity - w * w * x;
            this.velocity += a * dt;
            this.value += this.velocity * dt;
            if (this.isSettled()) { this.value = this.target; this.velocity = 0; }
            return this.value;
        }
        const x = this.value.clone().sub(this.target);
        const a = this.velocity.clone().multiplyScalar(-2 * z * w).addScaledVector
            ? this.velocity.clone().multiplyScalar(-2 * z * w).add(x.multiplyScalar(-w * w))
            : this.velocity.clone().multiplyScalar(-2 * z * w).add(x.multiplyScalar(-w * w));
        this.velocity.add(a.multiplyScalar(dt));
        this.value.add(this.velocity.clone().multiplyScalar(dt));
        if (this.isSettled()) {
            this.value.copy(this.target);
            this.velocity.set(0, 0, 0);
        }
        return this.value;
    }
}

/** Shared clock with tracks, markers, and optional time remap. */
export class Timeline {
    constructor() {
        this.tracks = [];
        this.markers = [];
        this.time = 0;
        this.elapsed = 0;
        this.duration = 0;
        this.speed = 1;
        this.playing = true;
        this.repeat = 0;
        this.timeRemap = null;
    }
    add(start, duration, easing = 'linear') {
        const id = this.tracks.length;
        this.tracks.push({ start, duration: Math.max(0, duration), easing });
        this.duration = Math.max(this.duration, start + duration);
        return id;
    }
    then(duration, easing = 'linear') {
        return this.add(this.duration, duration, easing);
    }
    addMarker(time, name, data = null) {
        this.markers.push({ time, name, data });
        this.markers.sort((a, b) => a.time - b.time);
        return this;
    }
    setTimeRemap(fnOrSamples) {
        this.timeRemap = fnOrSamples;
        return this;
    }
    _remap(t) {
        if (!this.timeRemap) return t;
        if (typeof this.timeRemap === 'function') return this.timeRemap(t);
        const s = this.timeRemap;
        if (!s.length) return t;
        if (t <= s[0].t) return s[0].v;
        if (t >= s[s.length - 1].t) return s[s.length - 1].v;
        for (let i = 0; i < s.length - 1; i++) {
            if (t >= s[i].t && t <= s[i + 1].t) {
                const u = (t - s[i].t) / (s[i + 1].t - s[i].t || 1);
                return s[i].v + (s[i + 1].v - s[i].v) * u;
            }
        }
        return t;
    }
    update(dt) {
        if (!this.playing || !Number.isFinite(dt)) return this;
        this.seek(this.elapsed + dt * this.speed);
        return this;
    }
    seek(time) {
        this.elapsed = time;
        let t = time;
        if (this.duration > 0) {
            if (this.repeat === Infinity) t = ((t % this.duration) + this.duration) % this.duration;
            else t = Math.min(this.duration, Math.max(0, t));
        }
        this.time = this._remap(t);
        return this;
    }
    progressOf(id) {
        const tr = this.tracks[id];
        if (!tr || tr.duration <= 0) return 0;
        const u = (this.time - tr.start) / tr.duration;
        return _ease(tr.easing, Math.min(1, Math.max(0, u)));
    }
    valueOf(id, from, to) {
        return _lerpAny(from, to, this.progressOf(id));
    }
    markerAt(time = this.time) {
        let hit = null;
        for (const m of this.markers) {
            if (m.time <= time) hit = m;
            else break;
        }
        return hit;
    }
}

/** Spring-lag follow for any Object3D position (secondary motion). */
export class ObjectSpring {
    constructor(object, { frequency = 6, damping = 1, property = 'position' } = {}) {
        this.object = object;
        this.property = property;
        const cur = object[property];
        this.spring = Spring.criticallyDamped(
            cur?.clone?.() || (typeof cur === 'number' ? cur : new Vector3()),
            frequency,
        );
        this.spring.dampingRatio = damping;
    }
    setTarget(target) {
        if (target?.position) this.spring.setTarget(target.position.clone());
        else this.spring.setTarget(target?.clone?.() || target);
        return this;
    }
    update(dt) {
        const v = this.spring.update(dt);
        const prop = this.object[this.property];
        if (typeof prop === 'number') this.object[this.property] = v;
        else prop?.copy?.(v);
        this.object._sync?.();
        return v;
    }
}

/**
 * Lightweight animated↔simulated pose blend (JS stand-in for PhysicsBlend).
 * Freezes the pose at limp/recover so a continuing clip cannot yank the fade.
 */
export class PoseBlend {
    constructor(target, duration = 0.25) {
        this.target = target;
        this.duration = Math.max(0, duration);
        this.state = 'animated';
        this.elapsed = 0;
        this._anchorPos = target.position.clone();
        this._anchorQuat = target.quaternion?.clone?.() || new Quaternion();
        this._lastAnimPos = target.position.clone();
        this._lastAnimQuat = target.quaternion?.clone?.() || new Quaternion();
    }
    goLimp() {
        this.state = 'toSimulated';
        this.elapsed = 0;
        this._anchorPos.copy(this._lastAnimPos);
        this._anchorQuat.copy(this._lastAnimQuat);
        return this;
    }
    getUp() {
        this.state = 'toAnimated';
        this.elapsed = 0;
        this._anchorPos.copy(this.target.position);
        if (this.target.quaternion) this._anchorQuat.copy(this.target.quaternion);
        return this;
    }
    update(dt, animated, simulated) {
        const aPos = animated?.position || animated;
        const sPos = simulated?.position || simulated;
        const aQuat = animated?.quaternion;
        const sQuat = simulated?.quaternion;
        if (aPos?.clone) this._lastAnimPos.copy(aPos);
        else if (aPos) this._lastAnimPos.set(aPos.x, aPos.y, aPos.z);
        if (aQuat) this._lastAnimQuat.copy(aQuat);

        if (this.state === 'animated') {
            this.target.position.copy(aPos);
            if (aQuat && this.target.quaternion) this.target.quaternion.copy(aQuat);
        } else if (this.state === 'simulated') {
            this.target.position.copy(sPos);
            if (sQuat && this.target.quaternion) this.target.quaternion.copy(sQuat);
        } else {
            this.elapsed += dt;
            const u = this.duration <= 0 ? 1 : Math.min(1, this.elapsed / this.duration);
            const e = _ease('cubicInOut', u);
            if (this.state === 'toSimulated') {
                this.target.position.lerpVectors(this._anchorPos, sPos, e);
                if (sQuat && this.target.quaternion) {
                    this.target.quaternion.copy(this._anchorQuat).slerp(sQuat, e);
                }
                if (u >= 1) this.state = 'simulated';
            } else {
                this.target.position.lerpVectors(this._anchorPos, aPos, e);
                if (aQuat && this.target.quaternion) {
                    this.target.quaternion.copy(this._anchorQuat).slerp(aQuat, e);
                }
                if (u >= 1) this.state = 'animated';
            }
        }
        this.target._sync?.();
        return this.state;
    }
}

export class TrackballControls {
    constructor(camera, domElement) {
        this._w = new WebTrackballControls(camera._w);
        this._camera = camera;
        this.domElement = domElement;
        this.enabled = true;
        this._rotating = false;
        this._panning = false;
        this._lastX = 0; this._lastY = 0;
        if (domElement?.addEventListener) {
            domElement.addEventListener('pointerdown', (e) => { this._rotating = e.button === 0; this._panning = e.button === 2; this._lastX = e.clientX; this._lastY = e.clientY; });
            domElement.addEventListener('pointermove', (e) => {
                if (!this.enabled || (!this._rotating && !this._panning)) return;
                const dx = e.clientX - this._lastX, dy = e.clientY - this._lastY;
                this._lastX = e.clientX; this._lastY = e.clientY;
                this.update(dx, dy, 0, this._rotating, this._panning);
            });
            domElement.addEventListener('pointerup', () => { this._rotating = false; this._panning = false; });
            domElement.addEventListener('wheel', (e) => { if (this.enabled) this.update(0, 0, e.deltaY, false, false); });
        }
    }
    update(dx = 0, dy = 0, wheel = 0, rotating = false, panning = false) {
        const w = this.domElement?.clientWidth || (typeof window !== 'undefined' ? window.innerWidth : 800);
        const h = this.domElement?.clientHeight || (typeof window !== 'undefined' ? window.innerHeight : 600);
        this._w.update(this._camera._w, dx, dy, wheel, rotating, panning, w, h);
        _cameraFromWasm(this._camera);
    }
}
export class FirstPersonControls {
    constructor(camera, domElement) {
        this._w = new WebFirstPersonControls(camera._w);
        this._camera = camera;
        this.domElement = domElement;
        this.enabled = true;
        this._rotating = false;
        this._keys = {};
        if (domElement?.addEventListener) {
            domElement.addEventListener('pointerdown', () => { this._rotating = true; });
            domElement.addEventListener('pointerup', () => { this._rotating = false; });
            domElement.addEventListener('pointermove', (e) => {
                if (!this.enabled || !this._rotating) return;
                this.update(e.movementX, e.movementY, 1 / 60, true);
            });
            domElement.addEventListener('keydown', (e) => { this._keys[e.code] = true; });
            domElement.addEventListener('keyup', (e) => { this._keys[e.code] = false; });
        }
    }
    update(dx = 0, dy = 0, dt = 1 / 60, rotating = false) {
        const fwd = (this._keys['KeyW'] || this._keys['ArrowUp']) ? 1 : 0;
        const back = (this._keys['KeyS'] || this._keys['ArrowDown']) ? 1 : 0;
        const left = (this._keys['KeyA'] || this._keys['ArrowLeft']) ? 1 : 0;
        const right = (this._keys['KeyD'] || this._keys['ArrowRight']) ? 1 : 0;
        this._w.setMoveInput(fwd - back, right - left, 0);
        this._w.update(this._camera._w, dx, dy, dt, rotating);
        _cameraFromWasm(this._camera);
    }
}
export class PointerLockControls { constructor(camera) { this._w = new WebPointerLockControls(camera._w); } }

// ---- Loaders ----
export class OBJLoader {
    constructor() { this._w = new WebObjLoader(); }
    parse(src) { const g = this._w.parse(src); return { geometry: g }; }
}
export class STLLoader {
    constructor() { this._w = new WebStlLoader(); }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return this._w.parse(u8);
    }
}
export class PLYLoader {
    constructor() { this._w = new WebPlyLoader(); }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return this._w.parse(u8);
    }
}
export class RGBELoader {
    constructor() { this._w = new WebHdrLoader(); }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return this._w.parse(u8);
    }
}

// ---- Audio ----
export class AudioListener { constructor() { this._w = new WebAudioListener(); } setMasterVolume(v) { this._w.setMasterVolume(v); } }
export class Audio extends Object3D {
    constructor(listener) { super(); this.type = 'Audio'; this.listener = listener; this._w = new WebAudio(); }
    setVolume(v) { this._w.setVolume(v); return this; }
    setLoop(l) { this._w.setLoop(l); return this; }
    play() { this._w.play(); return this; }
    stop() { this._w.stop(); return this; }
}

// ---- Post-FX ----
// three.js's EffectComposer and passes. Each class is a thin wrapper over a
// Rust pass in `crate::postprocessing` (`WebPass`); the composer's sequencing
// — read/write buffers, renderToScreen, swaps, masks — and every pass's own
// (mip chains, prepasses, accumulation) run in Rust. Properties and uniforms
// forward to the Rust pass by name, from the schema it describes: numbers and
// booleans on every assignment, vectors and colours as live three.js objects
// pushed before each render (so `pass.uniforms.center.value.set(x, y)` works
// as it does in three.js), textures as their wasm handles.

// Default projection uniforms for `WebGLRenderer.applyPostFx`.
function _defaultPostFxCam() {
    if (!_defaultPostFxCam._cache) {
        const proj = new Matrix4().identity();
        _defaultPostFxCam._cache = {
            near: 0.1, far: 100.0, kernelRadius: 8, kernelSize: 32,
            proj: proj.elements, invProj: proj.elements,
        };
    }
    return _defaultPostFxCam._cache;
}
function _postFxCameraFrom(camera, kernelRadius = 8, kernelSize = 32) {
    camera._sync();
    const proj = Array.from(camera._w.projectionMatrix());
    const inv = new Matrix4().fromArray(proj).invert();
    return {
        near: camera.near ?? 0.1,
        far: camera.far ?? 100.0,
        kernelRadius,
        kernelSize,
        proj,
        invProj: inv.elements,
    };
}

/** Seed `Math.random` and the passes' random stream (GlitchPass, SSAOPass)
 *  with the parity LCG: one stream, drawn from by JavaScript and Rust alike,
 *  as three.js's code all draws from `Math.random`. */
const _ppNativeRandom = Math.random;
export function seedRandom(seed) {
    if (typeof _optional.seedRandom === 'function') {
        _optional.seedRandom(seed >>> 0);
        Math.random = () => _optional.random();
        return;
    }
    let s = seed >>> 0;
    Math.random = () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}
export function unseedRandom() {
    _optional.unseedRandom?.();
    Math.random = _ppNativeRandom;
}

// Live three.js objects for the vector and colour parameters.
function _ppMakeLive(kind, arr) {
    const a = Array.isArray(arr) ? arr : [];
    if (kind === 'vec2') return new Vector2(a[0] ?? 0, a[1] ?? 0);
    if (kind === 'vec3') return new Vector3(a[0] ?? 0, a[1] ?? 0, a[2] ?? 0);
    if (kind === 'vec4') return new Vector4(a[0] ?? 0, a[1] ?? 0, a[2] ?? 0, a[3] ?? 0);
    if (kind === 'color') return new Color(a[0] ?? 0, a[1] ?? 0, a[2] ?? 0);
    return a.slice();
}
function _ppToArray(kind, v) {
    if (v == null) return [];
    if (typeof v === 'number' || typeof v === 'boolean') return v;
    if (kind === 'color') {
        const c = (v.isColor || v instanceof Color) ? v : new Color(v);
        // The linear working-space value, which is what the renderer uses.
        return [c.r, c.g, c.b];
    }
    if (Array.isArray(v) || ArrayBuffer.isView(v)) return Array.from(v);
    if (typeof v.x === 'number') {
        if (kind === 'vec2') return [v.x, v.y];
        if (kind === 'vec3') return [v.x, v.y, v.z];
        return [v.x, v.y, v.z ?? 0, v.w ?? 0];
    }
    return [];
}
// Wasm object ids for `objs` and everything under them.
function _ppObjectIds(objs) {
    const ids = [];
    const visit = (o) => {
        if (!o) return;
        if (o._handle?.id !== undefined) ids.push(o._handle.id);
        for (const c of o.children || []) visit(c);
    };
    for (const o of objs || []) visit(o);
    return ids;
}
function _ppTextureHandle(t) {
    return t?._w ?? t?.texture?._w ?? null;
}

// Define `pass.uniforms[name].value` / `pass[name]` accessors over `w`.
function _ppForward(pass, w, uniforms) {
    pass.uniforms = uniforms || {};
    uniforms = pass.uniforms;
    pass._ppLive = {};
    pass._ppHeld = {};
    for (const { name, kind, uniform } of w.paramSchema()) {
        const holder = uniform ? (uniforms[name] ??= {}) : pass;
        const key = uniform ? 'value' : name;
        const prior = Object.getOwnPropertyDescriptor(holder, key);
        if (prior && !prior.configurable) continue;
        const isVector = kind === 'vec2' || kind === 'vec3' || kind === 'vec4' || kind === 'color';
        Object.defineProperty(holder, key, {
            configurable: true,
            enumerable: true,
            get() {
                if (isVector || kind === 'floats') {
                    return (pass._ppLive[name] ??= _ppMakeLive(kind, w.get(name)));
                }
                if (kind === 'texture' || kind === 'cubeTexture' || kind === 'objects') {
                    return pass._ppHeld[name] ?? null;
                }
                const v = w.get(name);
                return kind === 'bool' ? !!v : v;
            },
            set(v) {
                if (kind === 'texture' || kind === 'cubeTexture') {
                    pass._ppHeld[name] = v;
                    const h = _ppTextureHandle(v);
                    if (!h) w.clearTexture(name);
                    else if (kind === 'cubeTexture') w.setCubeTexture(name, h);
                    else w.setTexture(name, h);
                    return;
                }
                if (kind === 'objects') {
                    pass._ppHeld[name] = v;
                    w.setObjects(name, new Float64Array(_ppObjectIds(v)));
                    return;
                }
                if (isVector) {
                    if (v != null && typeof v === 'object' && (v.isColor || v instanceof Color || typeof v.x === 'number')) {
                        pass._ppLive[name] = v;
                    } else {
                        delete pass._ppLive[name];
                    }
                    w.set(name, _ppToArray(kind, v));
                    return;
                }
                if (kind === 'floats') {
                    pass._ppLive[name] = v;
                    w.set(name, _ppToArray(kind, v));
                    return;
                }
                w.set(name, v == null ? [] : (kind === 'bool' ? !!v : Number(v)));
            },
        });
    }
}
// Push the live objects — mutated in place by the page — to the Rust pass.
function _ppSync(pass) {
    const w = pass._w;
    for (const name in pass._ppLive) {
        const v = pass._ppLive[name];
        const kind = v?.isColor || v instanceof Color ? 'color'
            : (v && typeof v.w === 'number') ? 'vec4'
            : (v && typeof v.z === 'number') ? 'vec3'
            : (v && typeof v.x === 'number') ? 'vec2' : 'floats';
        w.set(name, _ppToArray(kind, v));
    }
    for (const name in pass._ppHeld) {
        const v = pass._ppHeld[name];
        if (Array.isArray(v)) w.setObjects(name, new Float64Array(_ppObjectIds(v)));
    }
}

/** The base of every pass: `enabled`, `needsSwap`, `clear`,
 *  `renderToScreen` live on the Rust pass. */
class _Pass {
    constructor(w, uniforms) {
        this._w = w;
        this.isPass = true;
        _ppForward(this, w, uniforms);
    }
    get enabled() { return this._w.enabled; }
    set enabled(v) { this._w.enabled = !!v; }
    get needsSwap() { return this._w.needsSwap; }
    set needsSwap(v) { this._w.needsSwap = !!v; }
    get clear() { return this._w.clear; }
    set clear(v) { this._w.clear = !!v; }
    get renderToScreen() { return this._w.renderToScreen; }
    set renderToScreen(v) { this._w.renderToScreen = !!v; }
    setSize(width, height) { this._w.setSize(Math.max(1, width | 0), Math.max(1, height | 0)); }
    /** `[scene, camera]` for each scene reference, in the Rust pass's order. */
    _sceneRefs() { return []; }
    _beforeRender(_renderer) {}
    render(renderer, writeBuffer, readBuffer, deltaTime) {
        throw new Error(`${this.constructor.name}.render: passes run inside an EffectComposer`);
    }
    dispose() {}
}

function _ppWrapTarget(renderer, wrt, cache) {
    if (!wrt) return null;
    const hit = cache.get(wrt.id);
    if (hit) { wrt.free?.(); return hit; }
    const rt = new WebGLRenderTarget(wrt.width, wrt.height, { type: HalfFloatType });
    rt._w = wrt;
    if (renderer?._w?.renderTargetTexture) {
        rt.texture._w = renderer._w.renderTargetTexture(wrt);
        rt.texture._w_isExternal = true;
        rt.texture.isRenderTargetTexture = true;
    }
    cache.set(wrt.id, rt);
    return rt;
}

let _ppEmptyScene = null;
// The chain runs in Rust, so a pass has to be one of the ported ones.
function _ppCheckPass(pass) {
    if (!(pass?._w instanceof WebPass)) {
        throw new Error('EffectComposer: only the built-in passes can be added — a pass '
            + 'implemented in JavaScript cannot run inside the Rust composer');
    }
}

/** three.js `EffectComposer(renderer, renderTarget)`. */
export class EffectComposer {
    constructor(renderer, renderTarget) {
        this.renderer = renderer;
        const pr = renderer?.getPixelRatio?.() ?? 1;
        const w = renderer?._width ?? renderer?._canvas?.width ?? 800;
        const h = renderer?._height ?? renderer?._canvas?.height ?? 600;
        if (renderTarget?._w) {
            this._w = WebEffectComposer.withTarget(renderTarget._w);
        } else {
            this._w = new WebEffectComposer(w, h, pr);
        }
        this._targets = new Map();
        this.passes = [];
        // three.js's: `render()` with no delta passes the time since the last.
        this.clock = new Clock();
    }
    get renderToScreen() { return this._w.renderToScreen; }
    set renderToScreen(v) { this._w.renderToScreen = !!v; }
    get readBuffer() { return _ppWrapTarget(this.renderer, this._w.readBuffer(), this._targets); }
    get writeBuffer() { return _ppWrapTarget(this.renderer, this._w.writeBuffer(), this._targets); }
    get renderTarget1() { return this.writeBuffer; }
    get renderTarget2() { return this.readBuffer; }
    addPass(pass) {
        _ppCheckPass(pass);
        this.passes.push(pass);
        this._w.addPass(pass._w);
    }
    insertPass(pass, index) {
        _ppCheckPass(pass);
        this.passes.splice(index, 0, pass);
        this._w.insertPass(pass._w, index);
    }
    removePass(pass) {
        const i = this.passes.indexOf(pass);
        if (i !== -1) {
            this.passes.splice(i, 1);
            this._w.removePass(pass._w);
        }
    }
    isLastEnabledPass(passIndex) {
        for (let i = passIndex + 1; i < this.passes.length; i++) {
            if (this.passes[i].enabled) return false;
        }
        return true;
    }
    swapBuffers() { this._w.swapBuffers(); }
    setSize(width, height) { this._w.setSize(Math.max(1, width | 0), Math.max(1, height | 0)); }
    setPixelRatio(ratio) { this._w.setPixelRatio(ratio); }
    reset(renderTarget) {
        if (renderTarget?._w) {
            const old = this._w;
            this._w = WebEffectComposer.withTarget(renderTarget._w);
            for (const p of this.passes) this._w.addPass(p._w);
            old.free?.();
        } else {
            this._w.reset();
        }
        this._targets.clear();
    }
    render(deltaTime) {
        if (deltaTime === undefined) deltaTime = this.clock.getDelta();
        const renderer = this.renderer;
        // Scenes and cameras by slot: the Rust passes name them by index.
        const scenes = [];
        const cameras = [];
        const pairs = [];
        for (const p of this.passes) {
            _ppSync(p);
            p._beforeRender(renderer);
            if (!p.enabled) continue;
            p._sceneRefs().forEach(([scene, camera], n) => {
                let si = scene ? scenes.indexOf(scene) : 0;
                if (scene && si === -1) { si = scenes.length; scenes.push(scene); }
                let ci = camera ? cameras.indexOf(camera) : 0;
                if (camera && ci === -1) { ci = cameras.length; cameras.push(camera); }
                p._w.setSceneSlot(n, Math.max(0, si), Math.max(0, ci));
                if (scene && camera) pairs.push([scene, camera]);
            });
        }
        if (scenes.length > 2 || cameras.length > 4) {
            throw new Error('EffectComposer: at most two scenes and four cameras per composer');
        }
        for (const [scene, camera] of pairs) renderer._prepareRender(scene, camera);
        for (const c of cameras) c._sync?.();
        const r = renderer._w;
        if (cameras.length === 0) {
            this._w.render0(r, deltaTime);
            return;
        }
        const c = [0, 1, 2, 3].map((i) => (cameras[i] ?? cameras[0])._w);
        if (scenes.length === 0) {
            _ppEmptyScene ??= new WebScene();
            this._w.render1(r, _ppEmptyScene, c[0], c[1], c[2], c[3], deltaTime);
        } else if (scenes.length === 1) {
            this._w.render1(r, scenes[0]._w, c[0], c[1], c[2], c[3], deltaTime);
        } else {
            this._w.render2(r, scenes[0]._w, scenes[1]._w, c[0], c[1], c[2], c[3], deltaTime);
        }
    }
    dispose() { this._w.free?.(); }
}

export class RenderPass extends _Pass {
    constructor(scene, camera, overrideMaterial = null, clearColor = null, clearAlpha = null) {
        super(WebPass.renderPass());
        this.scene = scene;
        this.camera = camera;
        this.overrideMaterial = overrideMaterial;
        if (clearColor !== null) this.clearColor = clearColor;
        if (clearAlpha !== null) this.clearAlpha = clearAlpha;
    }
    get overrideMaterial() { return this._override ?? null; }
    set overrideMaterial(m) {
        this._override = m;
        if (m?._w) this._w.setOverrideMaterial(m._w);
        else this._w.clearOverrideMaterial();
    }
    _sceneRefs() { return [[this.scene, this.camera]]; }
}

// The shaders ShaderPass can run: three.js's `name` and uniforms, for the
// shaders the Rust side ports (a ShaderPass with another copies).
const _ppShader = (name, uniforms) => ({ name, uniforms, vertexShader: '', fragmentShader: '' });
export const CopyShader = _ppShader('CopyShader', { tDiffuse: { value: null }, opacity: { value: 1.0 } });
export const FXAAShader = _ppShader('FXAAShader', {
    tDiffuse: { value: null },
    resolution: { value: new Vector2(1 / 1024, 1 / 512) },
});
export const FilmShader = _ppShader('FilmShader', {
    tDiffuse: { value: null }, time: { value: 0.0 }, intensity: { value: 0.5 }, grayscale: { value: false },
});
export const DotScreenShader = _ppShader('DotScreenShader', {
    tDiffuse: { value: null }, tSize: { value: new Vector2(256, 256) },
    center: { value: new Vector2(0.5, 0.5) }, angle: { value: 1.57 }, scale: { value: 1.0 },
});
export const HalftoneShader = _ppShader('HalftoneShader', {
    tDiffuse: { value: null }, shape: { value: 1 }, radius: { value: 4 },
    rotateR: { value: Math.PI / 12 * 1 }, rotateG: { value: Math.PI / 12 * 2 }, rotateB: { value: Math.PI / 12 * 3 },
    scatter: { value: 0 }, width: { value: 1 }, height: { value: 1 }, blending: { value: 1 },
    blendingMode: { value: 1 }, greyscale: { value: false }, disable: { value: false },
});
export const DigitalGlitch = _ppShader('DigitalGlitch', {
    tDiffuse: { value: null }, tDisp: { value: null }, byp: { value: 0 }, amount: { value: 0.08 },
    angle: { value: 0.02 }, seed: { value: 0.02 }, seed_x: { value: 0.02 }, seed_y: { value: 0.02 },
    distortion_x: { value: 0.5 }, distortion_y: { value: 0.6 }, col_s: { value: 0.05 },
});
export const LuminosityShader = _ppShader('LuminosityShader', { tDiffuse: { value: null } });
export const LuminosityHighPassShader = _ppShader('LuminosityHighPassShader', {
    tDiffuse: { value: null }, luminosityThreshold: { value: 1.0 }, smoothWidth: { value: 1.0 },
    // `null`: a Color here would need the wasm module at import time; the
    // pass's own default is black.
    defaultColor: { value: null }, defaultOpacity: { value: 0.0 },
});
export const OutputShader = _ppShader('OutputShader', { tDiffuse: { value: null }, toneMappingExposure: { value: 1 } });

const _ppWarned = new Set();
export class ShaderPass extends _Pass {
    constructor(shader, textureID = 'tDiffuse') {
        const name = shader?.name ?? 'unspecified';
        if (!WebPass.isKnownShader(name) && !_ppWarned.has(name)) {
            _ppWarned.add(name);
            console.warn(`ShaderPass: no port of "${name}"; the pass copies its input`);
        }
        // three.js clones the shader's uniforms; the ported ones forward.
        const uniforms = {};
        for (const k in shader?.uniforms || {}) {
            const v = shader.uniforms[k].value;
            uniforms[k] = { value: v?.clone ? v.clone() : v };
        }
        const initial = {};
        for (const k in uniforms) initial[k] = uniforms[k].value;
        super(WebPass.shaderPass(name), uniforms);
        for (const k in initial) {
            if (initial[k] != null && Object.getOwnPropertyDescriptor(uniforms[k], 'value')?.set) {
                uniforms[k].value = initial[k];
            }
        }
        this.textureID = textureID;
        this.material = { uniforms: this.uniforms, name };
    }
}

export class TexturePass extends _Pass {
    constructor(map, opacity = 1.0) {
        super(WebPass.texturePass(opacity));
        if (map) this.map = map;
    }
}

export class CubeTexturePass extends _Pass {
    constructor(camera, tCube, opacity = 1) {
        super(WebPass.cubeTexturePass(opacity));
        this.camera = camera;
        if (tCube) this.tCube = tCube;
    }
    _sceneRefs() { return [[null, this.camera]]; }
}

export class ClearPass extends _Pass {
    constructor(clearColor = 0x000000, clearAlpha = 0) {
        const c = new Color(clearColor);
        super(WebPass.clearPass(c.r, c.g, c.b, !!clearColor, clearAlpha));
    }
}

export class MaskPass extends _Pass {
    constructor(scene, camera) {
        super(WebPass.maskPass());
        this.scene = scene;
        this.camera = camera;
    }
    _sceneRefs() { return [[this.scene, this.camera]]; }
}
export class ClearMaskPass extends _Pass {
    constructor() { super(WebPass.clearMaskPass()); }
}

export class OutputPass extends _Pass {
    constructor() { super(WebPass.outputPass()); }
    // three.js takes these from the renderer at render time.
    _beforeRender(renderer) {
        this._w.set('toneMapping', renderer?._toneMapping ?? renderer?.toneMapping ?? 0);
        this._w.set('toneMappingExposure', renderer?._toneMappingExposure ?? renderer?.toneMappingExposure ?? 1);
        this._w.set('srgbTransfer', (renderer?.outputColorSpace ?? 'srgb') !== 'srgb-linear');
    }
}

export class UnrealBloomPass extends _Pass {
    constructor(resolution = new Vector2(256, 256), strength = 1, radius = 0, threshold = 0) {
        super(WebPass.unrealBloomPass(resolution.x | 0, resolution.y | 0, strength ?? 1, radius ?? 0, threshold ?? 0));
    }
    // three.js keeps the tints as `bloomTintColors`, an array of Vector3.
    get bloomTintColors() {
        if (!this._tints) {
            const f = this._w.get('bloomTintColors') || [];
            this._tints = [0, 1, 2, 3, 4].map((i) => new Vector3(f[3 * i], f[3 * i + 1], f[3 * i + 2]));
        }
        return this._tints;
    }
    set bloomTintColors(v) { this._tints = v; }
    _beforeRender() {
        if (this._tints) this._w.set('bloomTintColors', this._tints.flatMap((t) => [t.x, t.y, t.z]));
    }
}

export class BloomPass extends _Pass {
    constructor(strength = 1, kernelSize = 25, sigma = 4) {
        super(WebPass.bloomPass(strength, kernelSize, sigma));
    }
}

export class FilmPass extends _Pass {
    constructor(intensity = 0.5, grayscale = false) {
        super(WebPass.filmPass(intensity, grayscale));
    }
}

export class DotScreenPass extends _Pass {
    constructor(center, angle, scale) {
        super(WebPass.dotScreenPass(center?.x ?? 0.5, center?.y ?? 0.5, angle ?? 1.57, scale ?? 1.0));
    }
}

export class HalftonePass extends _Pass {
    constructor(width, height, params = {}) {
        super(WebPass.halftonePass(width | 0, height | 0));
        for (const key in params) {
            if (this.uniforms[key]) this.uniforms[key].value = params[key];
        }
    }
}

export class GlitchPass extends _Pass {
    constructor(dt_size = 64) {
        super(WebPass.glitchPass(dt_size));
    }
    // threers extension: draw without DigitalGlitch's per-pixel snow.
    get _skipSnow() { return !this._w.get('snow'); }
    set _skipSnow(v) { this._w.set('snow', !v); }
}

export class AfterimagePass extends _Pass {
    constructor(damp = 0.96) { super(WebPass.afterimagePass(damp)); }
}

/** three.js `LUTPass({ lut, intensity })`; `lut` a `Data3DTexture` (or the
 *  `texture3D` a LUT loader returns). */
export class LUTPass extends _Pass {
    constructor(options = {}, legacySize) {
        // The shim once took `(lut, lutSize)`.
        const opts = (options && (options.isTexture || options.image || options.isData3DTexture))
            ? { lut: options, lutSize: legacySize } : options;
        super(WebPass.lutPass('intensity' in opts ? opts.intensity : 1));
        this.lut = opts.lut || null;
    }
    get lut() { return this._lut ?? null; }
    set lut(v) {
        this._lut = v;
        const img = v?.image;
        const size = img?.width ?? 0;
        const data = img?.data;
        if (!v || !size || !data) { this._w.setLut(0, undefined); return; }
        const n = size * size * size;
        const channels = Math.round(data.length / n);
        const scale = data instanceof Uint8Array || data instanceof Uint8ClampedArray ? 1 / 255 : 1;
        const rgba = new Float32Array(n * 4);
        for (let i = 0; i < n; i++) {
            for (let c = 0; c < 4; c++) {
                rgba[i * 4 + c] = c < channels ? data[i * channels + c] * scale : 1;
            }
        }
        this._w.setLut(size, rgba);
    }
}

export class SMAAPass extends _Pass {
    constructor(width = 1, height = 1) { super(WebPass.smaaPass(width | 0, height | 0)); }
}

export class BokehPass extends _Pass {
    constructor(scene, camera, params = {}) {
        super(WebPass.bokehPass(params.focus ?? 1.0, params.aperture ?? 0.025, params.maxblur ?? 1.0));
        this.scene = scene;
        this.camera = camera;
        this.uniforms.aspect.value = camera?.aspect ?? 1;
    }
    _sceneRefs() { return [[this.scene, this.camera]]; }
}

export class SSAOPass extends _Pass {
    constructor(scene, camera, width = 512, height = 512, kernelSize = 32) {
        super(WebPass.ssaoPass(width | 0, height | 0, kernelSize));
        this.scene = scene;
        this.camera = camera;
    }
    get kernel() {
        const k = this._w.ssaoKernel();
        const out = [];
        for (let i = 0; i < k.length; i += 3) out.push(new Vector3(k[i], k[i + 1], k[i + 2]));
        return out;
    }
    generateSampleKernel(kernelSize) { this._w.ssaoRegenerate(kernelSize); }
    _sceneRefs() { return [[this.scene, this.camera]]; }
}
SSAOPass.OUTPUT = { Default: 0, SSAO: 1, Blur: 2, Depth: 3, Normal: 4 };

export class SAOPass extends _Pass {
    constructor(scene, camera, resolution = new Vector2(256, 256)) {
        super(WebPass.saoPass(resolution.x | 0, resolution.y | 0));
        this.scene = scene;
        this.camera = camera;
        // three.js keeps the tunables in `params`; each forwards.
        const w = this._w;
        this.params = {};
        for (const name of ['output', 'saoBias', 'saoIntensity', 'saoScale', 'saoKernelRadius',
            'saoMinResolution', 'saoBlur', 'saoBlurRadius', 'saoBlurStdDev', 'saoBlurDepthCutoff']) {
            Object.defineProperty(this.params, name, {
                enumerable: true,
                get: () => w.get(name),
                set: (v) => w.set(name, typeof v === 'boolean' ? v : Number(v)),
            });
            delete this[name];
        }
    }
    _sceneRefs() { return [[this.scene, this.camera]]; }
}
SAOPass.OUTPUT = { Default: 0, SAO: 1, Normal: 2 };

export class SSRPass extends _Pass {
    constructor({ renderer, scene, camera, width = 512, height = 512, selects, bouncing = false, groundReflector } = {}) {
        super(WebPass.ssrPass(width | 0, height | 0));
        this.renderer = renderer;
        this.scene = scene;
        this.camera = camera;
        this.groundReflector = groundReflector;
        if (selects) this.selects = selects;
        this.bouncing = bouncing;
    }
    get selective() { return Array.isArray(this.selects); }
    _sceneRefs() { return [[this.scene, this.camera]]; }
}
SSRPass.OUTPUT = { Default: 0, SSR: 1, Beauty: 3, Depth: 4, Normal: 5, Metalness: 7 };

export class OutlinePass extends _Pass {
    constructor(resolution = new Vector2(256, 256), scene, camera, selectedObjects = []) {
        super(WebPass.outlinePass(resolution.x | 0, resolution.y | 0));
        this.renderScene = scene;
        this.renderCamera = camera;
        this.selectedObjects = selectedObjects;
    }
    // The shim's older names for the scene and camera.
    get scene() { return this.renderScene; }
    set scene(v) { this.renderScene = v; }
    get camera() { return this.renderCamera; }
    set camera(v) { this.renderCamera = v; }
    _sceneRefs() { return [[this.renderScene, this.renderCamera]]; }
}
OutlinePass.BlurDirectionX = new Vector2(1.0, 0.0);
OutlinePass.BlurDirectionY = new Vector2(0.0, 1.0);

export class SSAARenderPass extends _Pass {
    constructor(scene, camera, clearColor = 0x000000, clearAlpha = 0) {
        super(WebPass.ssaaRenderPass());
        this.scene = scene;
        this.camera = camera;
        this.clearColor = clearColor;
        this.clearAlpha = clearAlpha;
    }
    _sceneRefs() { return [[this.scene, this.camera]]; }
}
export class TAARenderPass extends _Pass {
    constructor(scene, camera, clearColor = 0x000000, clearAlpha = 0) {
        super(WebPass.taaRenderPass());
        this.scene = scene;
        this.camera = camera;
        this.clearColor = clearColor;
        this.clearAlpha = clearAlpha;
    }
    _sceneRefs() { return [[this.scene, this.camera]]; }
}

export class RenderPixelatedPass extends _Pass {
    constructor(pixelSize, scene, camera, options = {}) {
        super(WebPass.renderPixelatedPass(Math.max(1, pixelSize | 0)));
        this.scene = scene;
        this.camera = camera;
        this.normalEdgeStrength = options.normalEdgeStrength || 0.3;
        this.depthEdgeStrength = options.depthEdgeStrength || 0.4;
    }
    setPixelSize(pixelSize) { this.pixelSize = pixelSize; }
    _sceneRefs() { return [[this.scene, this.camera]]; }
}

export class RenderTransitionPass extends _Pass {
    constructor(sceneA, cameraA, sceneB, cameraB) {
        super(WebPass.renderTransitionPass());
        this.sceneA = sceneA;
        this.cameraA = cameraA;
        this.sceneB = sceneB;
        this.cameraB = cameraB;
    }
    setTransition(value) { this.uniforms.mixRatio.value = value; }
    useTexture(value) { this.uniforms.useTexture.value = value ? 1 : 0; }
    setTexture(value) { this.uniforms.tMixTexture.value = value; }
    setTextureThreshold(value) { this.uniforms.threshold.value = value; }
    _sceneRefs() { return [[this.sceneA, this.cameraA], [this.sceneB, this.cameraB]]; }
}
/** threers's older transition: `mixRatio` 0 shows `sceneA`, 1 `sceneB` — a
 *  RenderTransitionPass with the two swapped and no mix texture. */
export class TransitionPass extends RenderTransitionPass {
    constructor(sceneA, sceneB, camera) {
        super(sceneB, camera, sceneA, camera);
        this.useTexture(false);
    }
    get mixRatio() { return this.uniforms.mixRatio.value; }
    set mixRatio(v) { this.uniforms.mixRatio.value = v; }
}

/** AdaptiveToneMappingPass as three.js shipped it until r152. */
export class AdaptiveToneMappingPass extends _Pass {
    constructor(adaptive, resolution) {
        super(WebPass.adaptiveToneMappingPass(adaptive !== undefined ? !!adaptive : true, resolution ?? 256));
    }
    reset() { this._w.reset(); }
    setAdaptive(adaptive) { this.adaptive = !!adaptive; }
    setAdaptionRate(rate) { if (rate) this.adaptionRate = Math.abs(rate); }
    setMinLuminance(minLum) { if (minLum) this.minLuminance = minLum; }
    setMaxLuminance(maxLum) { if (maxLum) this.maxLuminance = maxLum; }
    setAverageLuminance(avgLum) { if (avgLum) this.averageLuminance = avgLum; }
    setMiddleGrey(middleGrey) { if (middleGrey) this.middleGrey = middleGrey; }
}

// ---- Helpers ----
/// three.js's `AxesHelper` / `GridHelper` / `PolarGridHelper`: line objects
/// built in Rust (`src/helpers`), mirrored as a copy of that object.
class _RustHelperObject extends Object3D {
    constructor(w, kind, type) {
        super();
        this._w = w;
        this._helperKind = kind;
        this.isLine = true;
        this.isLineSegments = true;
        this.type = type;
    }
    raycast(raycaster, intersects) { _rcCast(raycaster, this, intersects); }
    dispose() {}
    _mirrorKind() { return this._helperKind; }
    _mirrorCreate(w) {
        if (this._helperKind === 'axes') return w.createAxesHelper(this._w);
        if (this._helperKind === 'grid') return w.createGridHelper(this._w);
        return w.createPolarGridHelper(this._w);
    }
}
export class AxesHelper extends _RustHelperObject {
    constructor(size = 1) {
        super(new WebAxesHelper(size), 'axes', 'AxesHelper');
        this._isHelper = 'axes';
    }
}
export class GridHelper extends _RustHelperObject {
    constructor(size = 10, divisions = 10, color1 = 0x444444, color2 = 0x888888) {
        super(new WebGridHelper(size, divisions, color1, color2), 'grid', 'GridHelper');
        this._isHelper = 'grid';
    }
}
export class PolarGridHelper extends _RustHelperObject {
    constructor(radius = 10, segments = 16, circles = 8) {
        super(new WebPolarGridHelper(radius, segments, circles), 'polar', 'PolarGridHelper');
        this._isHelper = 'polar';
    }
}

// ---- Extras ----
/// three.js's `SimplexNoise` (examples/jsm/math): the permutation is drawn
/// from `r.random()` here, exactly as three.js draws it, and the noise is
/// computed by `WebThreeSimplexNoise` (src/extras/three_simplex_noise.rs).
export class SimplexNoise {
    constructor(r = Math) {
        const p = new Uint8Array(256);
        for (let i = 0; i < 256; i++) p[i] = Math.floor(r.random() * 256);
        this._w = new WebThreeSimplexNoise(p);
    }
    noise(x, y) { return this._w.noise(x, y); }
    noise3d(x, y, z) { return this._w.noise3d(x, y, z); }
    noise4d(x, y, z, w) { return this._w.noise4d(x, y, z, w); }
}
export class Octree {
    constructor(bb, maxDepth = 8, maxPoints = 8) {
        this._w = new WebOctree(bb._w, maxDepth, maxPoints);
    }
    insert(p) { this._w.insert(p._w()); }
}
export const MarchingCubes = class { constructor() { this._w = new WebMarchingCubes(); } };

// ---- Alt renderers ----
export class CSS2DRenderer {
    constructor() {
        this._w = new WebCss2dRenderer(800, 600);
        this.domElement = (typeof document !== 'undefined') ? document.createElement('div') : null;
    }
    setSize(w, h) { this._w = new WebCss2dRenderer(w, h); }
}
// three.js's SVGRenderer draws into an `<svg>` element you append to the page,
// the same way WebGLRenderer draws into a canvas — so `domElement` and
// `render()` are the whole of how you turn it on. Without them the renderer can
// only hand back a string, which is an export path and not a renderer.
export class SVGRenderer {
    constructor() {
        this._w = new WebSvgRenderer(800, 600);
        this.domElement = (typeof document !== 'undefined')
            ? document.createElementNS('http://www.w3.org/2000/svg', 'svg')
            : null;
        this.autoClear = true;
        this.info = { render: { vertices: 0, faces: 0 } };
        this._applySize(800, 600);
    }

    setSize(w, h) {
        // Resize in place. Rebuilding the renderer here would silently throw
        // away every option set before it, which is the sort of thing that only
        // shows up as "my SVG lost its background" much later.
        this._w.setSize(w, h);
        this._applySize(w, h);
    }

    _applySize(w, h) {
        if (!this.domElement) return;
        this.domElement.setAttribute('width', String(w));
        this.domElement.setAttribute('height', String(h));
        this.domElement.setAttribute('viewBox', `0 0 ${w} ${h}`);
    }

    /// Draw into `domElement`. The markup is a whole `<svg>` document, so its
    /// children and attributes are transplanted onto the live element rather
    /// than nesting one document inside another.
    render(scene, camera) {
        const markup = this.renderToString(scene, camera);
        this.info.render.faces = (markup.match(/<path/g) || []).length;
        this.info.render.vertices = this.info.render.faces * 3;
        if (!this.domElement) return markup;
        if (this.autoClear) this.clear();
        const parsed = new DOMParser().parseFromString(markup, 'image/svg+xml').documentElement;
        for (const attr of Array.from(parsed.attributes)) {
            this.domElement.setAttribute(attr.name, attr.value);
        }
        while (parsed.firstChild) this.domElement.appendChild(parsed.firstChild);
        return markup;
    }

    /// The document as markup, for writing to a file or embedding directly.
    ///
    /// Pushes the scene graph and camera into wasm first, exactly as
    /// `WebGLRenderer.render` does. Without it the renderer draws whatever pose
    /// was last uploaded, so moving a mesh or the camera from JavaScript would
    /// change nothing and the picture would simply look stuck.
    renderToString(scene, camera) {
        scene._syncTransforms?.(camera);
        if (!camera?._orbitControlled) camera._sync?.();
        this._syncBackground(scene);
        return this._w.renderToString(scene._w, camera._w);
    }

    /// Take the background from the scene, the way three.js reads it.
    ///
    /// `Scene.background` is stored unconverted, because WebGL's clear colour
    /// is written verbatim and three.js's `Color.r/g/b` are raw bytes over 255.
    /// This renderer works in linear light and encodes to sRGB on the way out,
    /// so handing it those same numbers renders `#101820` as a far lighter
    /// `#475663`. Round-tripping the authored hex through `setClearColor`,
    /// which decodes it, is what makes the document come out the colour that
    /// was asked for.
    _syncBackground(scene) {
        if (this._clearColorSet) return;
        const bg = scene?._background;
        const hex = (bg && typeof bg.getHex === 'function')
            ? bg.getHex()
            : (typeof bg === 'number' ? bg : null);
        if (hex == null || hex === this._sceneBackgroundHex) return;
        this._sceneBackgroundHex = hex;
        this._w.setClearColor(hex >>> 0, scene._backgroundAlpha ?? 1);
    }

    clear() {
        if (!this.domElement) return;
        while (this.domElement.firstChild) {
            this.domElement.removeChild(this.domElement.firstChild);
        }
    }

    // ---- three.js SVGRenderer surface ----

    /// `0xRRGGBB` or a `Color`, plus an alpha. Overrides `scene.background`.
    setClearColor(color, alpha = 1) {
        const hex = (color && typeof color.getHex === 'function')
            ? color.getHex()
            : (typeof color === 'string' ? parseInt(color.replace('#', ''), 16) : Number(color));
        if (!Number.isFinite(hex)) return;
        // An explicit colour wins over the scene's from here on.
        this._clearColorSet = true;
        this._w.setClearColor(hex >>> 0, alpha);
    }
    /// Decimal places kept on coordinates. `null` restores the default.
    setPrecision(precision) { this._w.setPrecision(precision == null ? 2 : precision); }
    /// three.js's coarse/fine switch. 'low' drops the curves and the seam
    /// hairline, which is most of what makes a document big.
    setQuality(quality) {
        const low = quality === 'low';
        this._w.setCurveTolerance(low ? 0 : 0.15);
        this._w.setSeamStroke(low ? 0 : 0.8);
        this._w.setPrecision(low ? 1 : 2);
    }
    /// No-op: an SVG has no pixels to scale. Present so a three.js render loop
    /// that sets it does not throw.
    setPixelRatio(_) {}

    // ---- threers extras ----

    /// 0 lit (default), 1 flat material colour, 2 wireframe.
    setShading(mode) { this._w.setShading(mode); }
    /// Match the WebGL renderer's tone mapping so the two agree on colour.
    /// three.js constants: 0 NoToneMapping, 1 Linear, 4 ACESFilmic, 6 AgX, 7 Neutral.
    setToneMapping(mode, exposure = 1) { this._w.setToneMapping(mode, exposure); }
    /// Draw the background rect at all. Off gives a transparent document.
    setBackground(on) { this._w.setBackground(!!on); }
    setCullBackfaces(on) { this._w.setCullBackfaces(!!on); }
    /// Hairline per face, in pixels, that hides SVG's anti-aliasing seams.
    setSeamStroke(px) { this._w.setSeamStroke(px); }
    /// Bend edges onto the real surface past this pixel error. 0 keeps every
    /// edge straight.
    setCurveTolerance(px) { this._w.setCurveTolerance(px); }
    /// Split faces too deep for one depth to sort. 0 disables.
    setDepthSplit(tolerance) { this._w.setDepthSplit(tolerance); }
    setSort(on) { this._w.setSort(!!on); }
}

// ---- Stats ----
export class Stats {
    constructor() { this._w = new WebStats(); this.dom = (typeof document !== 'undefined') ? document.createElement('div') : null; }
    begin() { this._w.begin(); }
    end() { this._w.end(); }
    update() { this.end(); this.begin(); }
    get fps() { return this._w.fps; }
}

// ---- Raycaster ----
// three.js r165's `Raycaster` over the Rust one (`src/core/raycaster.rs`).
// The walk is three.js's `intersect`: each object's layers are tested against
// the raycaster's and its own `raycast` is called — so code that replaces an
// object's `raycast` is honoured as in three.js, and a `false` return stops
// the descent — and then its children, when recursive. Visibility is not
// consulted, as in three.js. The built-in kinds' `raycast` (Mesh,
// SkinnedMesh, InstancedMesh, Line, LineSegments, LineLoop, Points, Sprite)
// do not cast there: they put the object's node on the cast's batch, and one
// Rust call per scene graph casts the batch — morph targets, skinning,
// instances, groups and material arrays, draw ranges, line thresholds,
// sprites and three-mesh-bvh's `boundsTree` all handled there — after the
// graphs involved are synced. Hits come back three.js-shaped, plus the
// threers `polygonIndex` and `barycoord` on triangle hits.
//
// Rust casts in f32 where three.js uses f64: distances, points, uvs and
// normals agree to f32 precision (relative error ~1e-6 of the scene's
// extent), not bit for bit.

/// three.js's `Raycaster.intersect` walk.
function _rcIntersect(object, raycaster, intersects, recursive) {
    if (!object) return;
    let propagate = true;
    const layers = object.layers;
    if (!layers || layers.test(raycaster.layers)) {
        const result = object.raycast?.(raycaster, intersects);
        if (result === false) propagate = false;
    }
    if (propagate === true && recursive === true) {
        const children = object.children ?? [];
        for (let i = 0, l = children.length; i < l; i++) _rcIntersect(children[i], raycaster, intersects, true);
    }
}
function _rcAscSort(a, b) { return a.distance - b.distance; }
/// A built-in kind's cast: onto the raycaster's batch (inside
/// `intersectObject(s)`), else cast now. `firstOnly` keeps only the
/// object's nearest hit (three-mesh-bvh's `firstHitOnly`); `remap` edits
/// each hit before it lands in `intersects`.
function _rcCast(raycaster, obj, intersects, firstOnly = false, remap = null) {
    if (typeof raycaster?._beginBatch !== 'function') {
        // A raycaster-shaped object (three.js code builds them): cast with a real one.
        const rc = new Raycaster(raycaster?.ray?.origin, raycaster?.ray?.direction, raycaster?.near ?? 0, raycaster?.far ?? Infinity);
        if (raycaster?.params) rc.params = raycaster.params;
        if (raycaster?.camera) rc.camera = raycaster.camera;
        if (raycaster?.layers) rc.layers.mask = raycaster.layers.mask;
        raycaster = rc;
    }
    if (raycaster._batch) {
        _rcEnqueue(raycaster._batch, obj, intersects, firstOnly, remap);
        return;
    }
    const batch = raycaster._beginBatch();
    try {
        _rcEnqueue(batch, obj, intersects, firstOnly, remap);
        raycaster._castBatch(batch);
    } finally {
        raycaster._batch = batch.outer;
    }
}
function _rcEnqueue(batch, obj, intersects, firstOnly, remap) {
    const root = _mirrorRoot(obj);
    if (!batch.synced.has(root)) {
        batch.synced.add(root);
        _mirrorSync(root);
    }
    obj._rcPrepare?.();
    const m = obj._m;
    if (!m || m.kind === 'group') return;
    let e = batch.graphs.get(m.graph);
    if (!e) batch.graphs.set(m.graph, e = { ids: [], first: [], worlds: [], entries: new Map() });
    e.ids.push(m.id);
    e.first.push(firstOnly ? 1 : 0);
    // three.js casts against `matrixWorld` as it stands — not updated here.
    for (const v of obj.matrixWorld.elements) e.worlds.push(v);
    e.entries.set(m.id, { obj, intersects, remap });
}
/// A Rust hit as three.js reports it for `obj`'s kind.
function _rcRecord(raw, obj) {
    const p = raw.point;
    const hit = { distance: raw.distance, point: new Vector3(p.x, p.y, p.z), object: obj };
    if (raw.face) {
        if (raw.uv) hit.uv = new Vector2(raw.uv.x, raw.uv.y);
        if (raw.uv1) hit.uv1 = new Vector2(raw.uv1.x, raw.uv1.y);
        if (raw.normal) hit.normal = new Vector3(raw.normal.x, raw.normal.y, raw.normal.z);
        const f = raw.face;
        hit.face = { a: f.a, b: f.b, c: f.c, normal: new Vector3(f.normal.x, f.normal.y, f.normal.z), materialIndex: f.materialIndex };
        hit.faceIndex = raw.faceIndex;
        if (raw.instanceId !== undefined) hit.instanceId = raw.instanceId;
        // threers extensions.
        hit.polygonIndex = raw.polygonIndex ?? null;
        if (raw.barycoord) hit.barycoord = new Vector3(raw.barycoord.x, raw.barycoord.y, raw.barycoord.z);
    } else if (obj.isPoints) {
        hit.distanceToRay = raw.distanceToRay;
        hit.index = raw.index;
        hit.face = null;
    } else if (obj.isLine) {
        hit.index = raw.index;
        hit.face = null;
        hit.faceIndex = null;
    } else {
        // Sprite.
        if (raw.uv) hit.uv = new Vector2(raw.uv.x, raw.uv.y);
        hit.face = null;
    }
    return hit;
}

/// three.js r165 `Raycaster`, casting in Rust (see above).
export class Raycaster {
    constructor(origin, direction, near = 0, far = Infinity) {
        this.ray = new Ray(origin, direction);
        this.near = near;
        this.far = far;
        this.camera = null;
        this.layers = new Layers();
        this.params = { Mesh: {}, Line: { threshold: 1 }, LOD: {}, Points: { threshold: 1 }, Sprite: {} };
        /// three-mesh-bvh: a mesh with a `boundsTree` reports only its nearest
        /// hit (needs `acceleratedRaycast` installed, as in three.js).
        this.firstHitOnly = false;
        this._w = new WebRaycaster();
        // three.js's rule: invisible objects are hit.
        this._w.includeInvisible = true;
        this._batch = null;
    }
    /// The direction is assumed normalised, as in three.js.
    set(origin, direction) {
        this.ray.set(origin, direction);
        return this;
    }
    /// three.js `setFromCamera`, from the camera's `matrixWorld` and
    /// `projectionMatrixInverse` (in Rust). The camera's world matrix is
    /// brought up to date first — the shim's cameras are posed from it.
    setFromCamera(coords, camera) {
        const ortho = camera?.isOrthographicCamera === true;
        if (!ortho && camera?.isPerspectiveCamera !== true) {
            console.error('THREE.Raycaster: Unsupported camera type: ' + camera?.type);
            return this;
        }
        camera.updateWorldMatrix(true, false);
        this._w.setFromCameraMatrices(coords.x, coords.y, Float32Array.from(camera.matrixWorld.elements),
            Float32Array.from(camera.projectionMatrixInverse.elements), ortho, camera.near, camera.far);
        const r = this._w.rayArray();
        this.ray.origin.set(r[0], r[1], r[2]);
        this.ray.direction.set(r[3], r[4], r[5]);
        this.camera = camera;
        return this;
    }
    /// three.js `setFromXRController`: from the controller's position along its -Z.
    setFromXRController(controller) {
        controller.updateWorldMatrix(true, false);
        const m = new Matrix4().extractRotation(controller.matrixWorld);
        this.ray.origin.setFromMatrixPosition(controller.matrixWorld);
        this.ray.direction.set(0, 0, -1).applyMatrix4(m);
        return this;
    }
    intersectObject(object, recursive = true, intersects = []) {
        return this.intersectObjects([object], recursive, intersects);
    }
    intersectObjects(objects, recursive = true, intersects = []) {
        const batch = this._beginBatch();
        try {
            for (let i = 0, l = objects.length; i < l; i++) _rcIntersect(objects[i], this, intersects, recursive);
            this._castBatch(batch);
        } finally {
            this._batch = batch.outer;
        }
        intersects.sort(_rcAscSort);
        return intersects;
    }
    _beginBatch() {
        const batch = { outer: this._batch, synced: new Set(), graphs: new Map() };
        this._batch = batch;
        return batch;
    }
    /// The ray, range, thresholds and camera, then one Rust cast per graph.
    _castBatch(batch) {
        if (!batch.graphs.size) return;
        const w = this._w, o = this.ray.origin, d = this.ray.direction;
        w.setRay(o.x, o.y, o.z, d.x, d.y, d.z);
        w.near = this.near;
        w.far = this.far;
        w.setLineThreshold(this.params.Line?.threshold ?? 1);
        w.setPointsThreshold(this.params.Points?.threshold ?? 1);
        const cam = this.camera;
        if (cam?.matrixWorld) {
            cam.updateWorldMatrix?.(true, false);
            w.setCameraMatrices(Float32Array.from(cam.matrixWorld.elements), cam.isPerspectiveCamera === true);
        } else {
            w.setCameraMatrices(undefined, false);
        }
        for (const [graph, e] of batch.graphs) {
            const hits = w.intersectEach(graph.w, new Float64Array(e.ids), new Uint8Array(e.first), new Float32Array(e.worlds));
            for (const raw of hits) {
                const entry = e.entries.get(raw.objectId);
                if (!entry) continue;
                const hit = _rcRecord(raw, entry.obj);
                entry.remap?.(hit);
                entry.intersects.push(hit);
            }
        }
    }
}
/// For web/mesh-bvh-impl.js's `acceleratedRaycast`: the built-in cast, with
/// three-mesh-bvh's per-mesh `firstHitOnly`.
export const _meshRaycastInternals = Object.freeze({ cast: _rcCast });

// ---- Clock ----
export class Clock {
    constructor() { this._w = new WebClock(); }
    getDelta() { return this._w.getDelta(); }
    getElapsedTime() { return this._w.getElapsedTime(); }
}

// ---- Scene ----
/// three.js `Scene`: the root of a wasm scene graph. Its children, and theirs,
/// are mirrored into `_w` (see the scene-graph notes at `Object3D`); a render
/// or raycast syncs it, and so does `updateMatrixWorld()` / `_syncNow()`.
export class Scene extends Object3D {
    constructor() {
        super();
        this.isScene = true;
        this.type = 'Scene';
        this._w = new WebScene();
        this._graph = new _SceneGraph(this._w, this);
        this._m = { graph: this._graph, handle: this._graph.root, id: this._graph.rootId, kind: 'scene', parentId: undefined, childIds: null };
        this._background = null;
        this._backgroundAlpha = 1;
        this._fog = null;
        this._environment = null;
        this.backgroundBlurriness = 0;
        this.backgroundIntensity = 1;
        this.environmentIntensity = 1;
        this.overrideMaterial = null;
    }
    set environment(env) {
        this._environment = env;
        this._bindEnvironment(env);
        // A cube whose images arrive later (`CubeTextureLoader`) binds again
        // when they do, through the texture's re-bind list.
        if (env && !env._cubeRTId) {
            const slots = env._boundIn ??= new Map();
            const ref = this._wref ??= new WeakRef(this);
            slots.set(`environment:${this.uuid ?? ''}`, {
                ref, key: 'environment', bind: (t) => this._bindEnvironment(t),
            });
        }
    }
    _bindEnvironment(env) {
        const cubeRTId = env?._cubeRTId;
        if (cubeRTId) {
            this._w.setEnvironmentCube(cubeRTId);
        } else if (env?._w) {
            this._w.setEnvironmentMap(env._w);
        } else {
            this._w.clearEnvironment();
        }
    }
    get environment() { return this._environment; }
    // three.js `scene.environmentIntensity` (default 1).
    get environmentIntensity() { return this._environmentIntensity ?? 1; }
    set environmentIntensity(v) {
        this._environmentIntensity = Number(v);
        this._w.setEnvironmentIntensity(this._environmentIntensity);
    }
    // three.js `scene.environmentRotation`, an Euler edited in place: pushed
    // before each render when it moved.
    get environmentRotation() { return this._environmentRotation ??= new Euler(0, 0, 0); }
    set environmentRotation(e) { this._environmentRotation = e; }
    _syncEnvironmentRotation() {
        const e = this._environmentRotation;
        if (!e) return;
        const last = this._envRotSynced;
        if (last && last[0] === e.x && last[1] === e.y && last[2] === e.z && last[3] === e.order) return;
        this._envRotSynced = [e.x, e.y, e.z, e.order];
        this._w.setEnvironmentRotation(e.x, e.y, e.z, e.order ?? 'XYZ');
    }
    set fog(f) {
        this._fog = f;
        this._fogState = _sceneFogState(f);
        if (!f) { this._w.setFog(new WebColor(1, 1, 1), 1, 1000, 0, 0); return; }
        // Linear Fog: { color, near, far } → mode 1
        // FogExp2:    { color, density }   → mode 2
        const c = f.color || new Color(0xffffff);
        const wc = new WebColor(c.r, c.g, c.b);
        if (f.isFogExp2) this._w.setFog(wc, 0, 0, f.density ?? 0, 2);
        else             this._w.setFog(wc, f.near ?? 1, f.far ?? 1000, 0, 1);
    }
    get fog() { return this._fog; }
    set background(c) {
        this._background = c;
        // The renderer takes the background in linear light, like every other
        // colour, and encodes it for the canvas — three.js's WebGLBackground
        // does the same conversion before `gl.clearColor`. So hand it the
        // linear WebColor (`fromHex` decodes), and `0x20262c` lands on the
        // canvas as `#20262c` on an sRGB or a plain `bgra8unorm` canvas alike.
        this._w.background = _color(c);
        this._backgroundState = c?.isColor ? [c.r, c.g, c.b] : null;
    }
    get background() { return this._background; }
    /// three.js reads `scene.fog` and a Color `scene.background` every frame,
    /// so a change made in place after they were set (`fog.color.setHex`,
    /// `fog.near = …`, `background.lerp(…)`) reaches wasm here, before each
    /// render.
    _syncColors() {
        const fog = _sceneFogState(this._fog);
        if (fog ? !_sameNumbers(this._fogState, fog) : this._fogState != null) this.fog = this._fog;
        const bg = this._background;
        if (bg?.isColor && !_sameNumbers(this._backgroundState, [bg.r, bg.g, bg.b])) this.background = bg;
    }
    set backgroundAlpha(a) {
        this._backgroundAlpha = Number(a);
        if (this._w.setBackgroundAlpha) this._w.setBackgroundAlpha(this._backgroundAlpha);
    }
    get backgroundAlpha() { return this._backgroundAlpha != null ? this._backgroundAlpha : 1; }
    /// Push every pending change of this scene's graph to wasm.
    _syncNow() { _mirrorSyncScene(this); return this; }
    /// Older callers' name for `_syncNow` (the camera argument is unused: the
    /// layers are the camera's own, in Rust).
    _syncTransforms(_camera) { _mirrorSyncScene(this); }
    /// three.js `updateMatrixWorld`, then a sync of the wasm graph.
    updateMatrixWorld(force) {
        super.updateMatrixWorld(force);
        _mirrorSyncScene(this);
    }
    _mirrorTransform(w, h, m) {
        super._mirrorTransform(w, h, m);
        w.setMatrixWorldAutoUpdate(h, this.matrixWorldAutoUpdate !== false);
    }
    copy(source, recursive) {
        super.copy(source, recursive);
        if (source.background !== null) this.background = source.background.clone?.() ?? source.background;
        if (source.environment !== null) this.environment = source.environment;
        if (source.fog !== null) this.fog = source.fog.clone?.() ?? source.fog;
        this.backgroundBlurriness = source.backgroundBlurriness;
        this.backgroundIntensity = source.backgroundIntensity;
        this.overrideMaterial = source.overrideMaterial;
        return this;
    }
}

// ---- Cameras ----
/// three.js `Camera`: an Object3D looking down its -Z, with the projection
/// matrices three.js keeps. The wasm camera (`_w`) draws from the camera's
/// world matrix (`_sync`), so a camera can be parented, rotated or animated
/// like any object.
export class Camera extends Object3D {
    constructor() {
        super();
        this.isCamera = true;
        this.type = 'Camera';
        this.matrixWorldInverse = new Matrix4();
        this.projectionMatrix = new Matrix4();
        this.projectionMatrixInverse = new Matrix4();
        this.coordinateSystem = 2000; // WebGLCoordinateSystem
        /// Where `lookAt` last aimed it (world space) — read by controls that
        /// orbit a target.
        this._lookAt = new Vector3();
    }
    copy(source, recursive) {
        super.copy(source, recursive);
        this.matrixWorldInverse.copy(source.matrixWorldInverse);
        this.projectionMatrix.copy(source.projectionMatrix);
        this.projectionMatrixInverse.copy(source.projectionMatrixInverse);
        this.coordinateSystem = source.coordinateSystem;
        if (source._lookAt) this._lookAt.copy(source._lookAt);
        return this;
    }
    getWorldDirection(target = new Vector3()) { return super.getWorldDirection(target).negate(); }
    updateMatrixWorld(force) {
        super.updateMatrixWorld(force);
        this.matrixWorldInverse.copy(this.matrixWorld).invert();
    }
    updateWorldMatrix(updateParents, updateChildren) {
        super.updateWorldMatrix(updateParents, updateChildren);
        this.matrixWorldInverse.copy(this.matrixWorld).invert();
    }
    clone() { return new this.constructor().copy(this); }
    /// Pose the wasm camera from this camera's world matrix (and layers).
    _sync() {
        if (!this._w) return;
        this.updateWorldMatrix(true, false);
        this._w.setMatrixWorld(Float32Array.from(this.matrixWorld.elements));
        this._w.setLayersMask?.(this.layers.mask >>> 0);
        if (this.focus != null) this._w.setFocusDistance?.(this.focus);
    }
}

/// three.js r165 `PerspectiveCamera` (plus the cinematic extras the wasm
/// camera has: `focus`, `filmOffset` as a lens shift).
export class PerspectiveCamera extends Camera {
    constructor(fov = 50, aspect = 1, near = 0.1, far = 2000) {
        super();
        this.isPerspectiveCamera = true;
        this.type = 'PerspectiveCamera';
        this._w = WebCamera.perspective(fov, aspect, near, far);
        this._fov = fov;
        this.zoom = 1;
        this.near = near;
        this.far = far;
        this.focus = 10;
        this.aspect = aspect;
        this.view = null;
        this.filmGauge = 35;
        this.filmOffset = 0;
        this.updateProjectionMatrix();
    }
    get fov() { return this._fov; }
    set fov(v) {
        this._fov = Number(v);
        this._w.setFov?.(this._fov);
    }
    copy(source, recursive) {
        super.copy(source, recursive);
        this.fov = source.fov;
        this.zoom = source.zoom;
        this.near = source.near;
        this.far = source.far;
        this.focus = source.focus;
        this.aspect = source.aspect;
        this.view = source.view === null ? null : Object.assign({}, source.view);
        this.filmGauge = source.filmGauge;
        this.filmOffset = source.filmOffset;
        this.updateProjectionMatrix();
        return this;
    }
    setFocalLength(focalLength) {
        const vExtentSlope = 0.5 * this.getFilmHeight() / focalLength;
        this.fov = MathUtils.RAD2DEG * 2 * Math.atan(vExtentSlope);
        this.updateProjectionMatrix();
    }
    getFocalLength() {
        const vExtentSlope = Math.tan(MathUtils.DEG2RAD * 0.5 * this.fov);
        return 0.5 * this.getFilmHeight() / vExtentSlope;
    }
    getEffectiveFOV() {
        return MathUtils.RAD2DEG * 2 * Math.atan(Math.tan(MathUtils.DEG2RAD * 0.5 * this.fov) / this.zoom);
    }
    getFilmWidth() { return this.filmGauge * Math.min(this.aspect, 1); }
    getFilmHeight() { return this.filmGauge / Math.max(this.aspect, 1); }
    setViewOffset(fullWidth, fullHeight, x, y, width, height) {
        this.aspect = fullWidth / fullHeight;
        if (this.view === null) this.view = { enabled: true, fullWidth: 1, fullHeight: 1, offsetX: 0, offsetY: 0, width: 1, height: 1 };
        Object.assign(this.view, { enabled: true, fullWidth, fullHeight, offsetX: x, offsetY: y, width, height });
        this.updateProjectionMatrix();
    }
    clearViewOffset() {
        if (this.view !== null) this.view.enabled = false;
        this.updateProjectionMatrix();
    }
    /// three.js's frustum (zoom, view offset, film offset) into
    /// `projectionMatrix`; the wasm camera takes fov, aspect, near/far and the
    /// film offset as a lens shift.
    updateProjectionMatrix() {
        const near = this.near;
        let top = near * Math.tan(MathUtils.DEG2RAD * 0.5 * this.fov) / this.zoom;
        let height = 2 * top;
        let width = this.aspect * height;
        let left = -0.5 * width;
        const view = this.view;
        if (this.view !== null && this.view.enabled) {
            const fullWidth = view.fullWidth, fullHeight = view.fullHeight;
            left += view.offsetX * width / fullWidth;
            top -= view.offsetY * height / fullHeight;
            width *= view.width / fullWidth;
            height *= view.height / fullHeight;
        }
        const skew = this.filmOffset;
        if (skew !== 0) left += near * skew / this.getFilmWidth();
        this.projectionMatrix.makePerspective(left, left + width, top, top - height, near, this.far);
        this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
        this._w.setAspect(this.aspect);
        this._w.setFov?.(this._fov);
        this._w.setFrustum?.(0, 0, 0, 0, this.near, this.far, this.zoom);
        if (typeof this._w.setShift === 'function') {
            this._w.setShift(this.filmGauge ? (this.filmOffset / this.filmGauge) : 0, 0);
        }
        return this;
    }
}

/// three.js r165 `OrthographicCamera`.
export class OrthographicCamera extends Camera {
    constructor(left = -1, right = 1, top = 1, bottom = -1, near = 0.1, far = 2000) {
        super();
        this.isOrthographicCamera = true;
        this.type = 'OrthographicCamera';
        this._w = WebCamera.orthographic(left, right, top, bottom, near, far);
        this.zoom = 1;
        this.view = null;
        this.left = left;
        this.right = right;
        this.top = top;
        this.bottom = bottom;
        this.near = near;
        this.far = far;
        this.updateProjectionMatrix();
    }
    copy(source, recursive) {
        super.copy(source, recursive);
        this.left = source.left; this.right = source.right; this.top = source.top; this.bottom = source.bottom;
        this.near = source.near; this.far = source.far; this.zoom = source.zoom;
        this.view = source.view === null ? null : Object.assign({}, source.view);
        this.updateProjectionMatrix();
        return this;
    }
    setViewOffset(fullWidth, fullHeight, x, y, width, height) {
        if (this.view === null) this.view = { enabled: true, fullWidth: 1, fullHeight: 1, offsetX: 0, offsetY: 0, width: 1, height: 1 };
        Object.assign(this.view, { enabled: true, fullWidth, fullHeight, offsetX: x, offsetY: y, width, height });
        this.updateProjectionMatrix();
    }
    clearViewOffset() {
        if (this.view !== null) this.view.enabled = false;
        this.updateProjectionMatrix();
    }
    updateProjectionMatrix() {
        const dx = (this.right - this.left) / (2 * this.zoom);
        const dy = (this.top - this.bottom) / (2 * this.zoom);
        const cx = (this.right + this.left) / 2;
        const cy = (this.top + this.bottom) / 2;
        let left = cx - dx, right = cx + dx, top = cy + dy, bottom = cy - dy;
        if (this.view !== null && this.view.enabled) {
            const scaleW = (this.right - this.left) / this.view.fullWidth / this.zoom;
            const scaleH = (this.top - this.bottom) / this.view.fullHeight / this.zoom;
            left += scaleW * this.view.offsetX;
            right = left + scaleW * this.view.width;
            top -= scaleH * this.view.offsetY;
            bottom = top - scaleH * this.view.height;
        }
        this.projectionMatrix.makeOrthographic(left, right, top, bottom, this.near, this.far);
        this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
        this._w.setFrustum?.(left, right, top, bottom, this.near, this.far, 1);
        return this;
    }
}

// ---- Renderer ----
/// The background a `scene.background = null` scene gets from wasm
/// (`Scene::new`), restored when a clear colour stops applying.
const _DEFAULT_SCENE_BACKGROUND = 0x111111;
/// `0xRRGGBB` from a three.js colour argument — hex number, CSS string or
/// `Color` — or null. Hex rather than r/g/b because the shim's `Color` keeps
/// authored bytes, and the hex round-trip is what decodes them to linear.
function _clearColorHex(color) {
    if (color && typeof color.getHex === 'function') return color.getHex() >>> 0;
    if (typeof color === 'number' && Number.isFinite(color)) return color >>> 0;
    if (typeof color === 'string') return new Color(color).getHex() >>> 0;
    if (color && typeof color === 'object' && 'r' in color) {
        const b = (v) => Math.max(0, Math.min(255, Math.round(Number(v) * 255)));
        return ((b(color.r) << 16) | (b(color.g) << 8) | b(color.b)) >>> 0;
    }
    return null;
}
/// three.js clipping, resolved to the one world-space plane threers clips an
/// object by (`Object3D::clip_plane`, discarded per fragment in every colour
/// pass): `renderer.clippingPlanes` apply to every object, a material's
/// `clippingPlanes` to its own objects when `renderer.localClippingEnabled`.
/// Planes are re-read every frame — three.js does, and apps move a section
/// plane by assigning `plane.constant` without telling anyone.
///
/// One plane per object: with more, the renderer's first (else the
/// material's first) is used and a warning says so once. `clipIntersection`
/// only matters with several planes, so it is not consulted.
let _clipWarned = false;
function _clipPlaneOf(renderer, o) {
    const mat = Array.isArray(o.material) ? o.material[0] : o.material;
    const local = renderer.localClippingEnabled && mat?.clippingPlanes?.length ? mat.clippingPlanes : null;
    const global = renderer.clippingPlanes?.length ? renderer.clippingPlanes : null;
    if (!local && !global) return null;
    if ((local?.length ?? 0) + (global?.length ?? 0) > 1 && !_clipWarned) {
        _clipWarned = true;
        console.warn('[threers] one clipping plane per object is applied; the others are ignored');
    }
    return (global ?? local)[0];
}
function _syncClipPlane(renderer, o) {
    const m = o._m;
    if (!m || typeof m.graph.w.setObjectClipPlane !== 'function') return;
    const p = _clipPlaneOf(renderer, o);
    const s = m.clip;
    if (!p) {
        if (s) m.graph.w.setObjectClipPlane(m.handle, 0, 0, 0, 0, false);
        m.clip = null;
        return;
    }
    const n = p.normal;
    if (s && s[0] === n.x && s[1] === n.y && s[2] === n.z && s[3] === p.constant) return;
    m.clip = [n.x, n.y, n.z, p.constant];
    m.graph.w.setObjectClipPlane(m.handle, n.x, n.y, n.z, p.constant, true);
}

export class WebGLRenderer {
    constructor() {
        throw new Error("Use 'await WebGLRenderer.create(canvas)' (async on wasm).");
    }
    /// three.js `new WebGLRenderer(parameters)`, async: `create(canvas,
    /// parameters)` or `create({ canvas, antialias, … })`. `antialias`
    /// (default false, as in three.js) turns on 4× MSAA.
    static async create(canvasOrParameters, parameters = {}) {
        let canvas = canvasOrParameters;
        // A parameters object rather than a canvas: no size of its own.
        if (canvasOrParameters && typeof canvasOrParameters === 'object' && !('width' in canvasOrParameters)) {
            parameters = canvasOrParameters;
            canvas = parameters.canvas;
        }
        parameters = parameters ?? {};
        if (!_wasmReady) await initThreers();
        const w = await WebRenderer.new(canvas);
        const r = Object.create(WebGLRenderer.prototype);
        r._w = w;
        r._canvas = canvas;
        // three.js `shadowMap`; autoUpdate / needsUpdate live in wasm (a
        // render with autoUpdate off redraws the maps only when needsUpdate is
        // set, and clears it).
        let shadowType = 1;
        w.setShadowType?.(shadowType);
        r.shadowMap = {
            enabled: true,
            // three.js's filter: Basic, PCF (the default), PCFSoft, and VSM —
            // drawn as percentage-closer soft shadows (see `setShadowType`).
            get type() { return shadowType; },
            set type(v) { shadowType = Number(v) || 0; w.setShadowType?.(shadowType); },
            get autoUpdate() { return w.shadowMapAutoUpdate ?? true; },
            set autoUpdate(v) { if ('shadowMapAutoUpdate' in w) w.shadowMapAutoUpdate = !!v; },
            get needsUpdate() { return w.shadowMapNeedsUpdate ?? false; },
            set needsUpdate(v) { if ('shadowMapNeedsUpdate' in w) w.shadowMapNeedsUpdate = !!v; },
        };
        // three.js clipping: the renderer's planes clip everything, a
        // material's only with localClippingEnabled (see _syncClipPlane).
        r.localClippingEnabled = false;
        r.clippingPlanes = [];
        // Push toneMapping / exposure into wasm — assigning the JS fields alone
        // left the GPU on NoToneMapping, so sunlit Earth clipped to flat white.
        r._toneMapping = 0;
        r._toneMappingExposure = 1;
        Object.defineProperty(r, 'toneMapping', {
            get() { return r._toneMapping; },
            set(v) {
                r._toneMapping = Number(v) || 0;
                r._w.setToneMapping?.(r._toneMapping, r._toneMappingExposure);
            },
            configurable: true, enumerable: true,
        });
        Object.defineProperty(r, 'toneMappingExposure', {
            get() { return r._toneMappingExposure; },
            set(v) {
                r._toneMappingExposure = Number(v);
                r._w.setToneMapping?.(r._toneMapping, r._toneMappingExposure);
            },
            configurable: true, enumerable: true,
        });
        // three.js `antialias`: 4× MSAA on the forward pass, off by default.
        r._antialias = parameters.antialias === true;
        r._w.setMsaa?.(r._antialias ? 4 : 1);
        // threers extras, both off as three.js renders: geometric specular
        // antialiasing in place of `geometryRoughness`, and a transparent
        // Standard/Physical surface's alpha raised by its reflectance.
        r._specularAntialiasing = false;
        r._reflectiveTransparency = false;
        Object.defineProperty(r, 'specularAntialiasing', {
            get() { return r._specularAntialiasing; },
            set(v) { r._specularAntialiasing = !!v; r._w.setSpecularAntialiasing?.(r._specularAntialiasing); },
            configurable: true, enumerable: true,
        });
        Object.defineProperty(r, 'reflectiveTransparency', {
            get() { return r._reflectiveTransparency; },
            set(v) { r._reflectiveTransparency = !!v; r._w.setReflectiveTransparency?.(r._reflectiveTransparency); },
            configurable: true, enumerable: true,
        });
        // three.js sizing state: `_width`/`_height` are logical (CSS) pixels,
        // the canvas holds `floor(size × pixelRatio)` device pixels. Like
        // three.js, the starting size is whatever the canvas already has.
        r._pixelRatio = 1;
        r._width = canvas.width;
        r._height = canvas.height;
        r._viewport = new Vector4(0, 0, r._width, r._height);
        r._scissor = new Vector4(0, 0, r._width, r._height);
        r._scissorTest = false;
        // three.js's default clear colour (black, opaque). It only becomes the
        // background of a `scene.background = null` render once set
        // explicitly — until then such scenes keep threers' own default, so
        // pages that never call setClearColor look exactly as before.
        r._clearColor = new Color(0x000000);
        r._clearAlpha = 1;
        r._clearColorSet = false;
        return r;
    }
    get domElement() { return this._canvas; }
    /// three.js `setSize`: `width`/`height` are CSS pixels; the drawing buffer
    /// becomes `floor(size × pixelRatio)`. `updateStyle` (default true) also
    /// sets the canvas's CSS size. Resets the viewport to the whole canvas.
    setSize(w, h, updateStyle = true) {
        this._width = w;
        this._height = h;
        this._canvas.width = Math.floor(w * this._pixelRatio);
        this._canvas.height = Math.floor(h * this._pixelRatio);
        if (updateStyle !== false && this._canvas.style) {
            this._canvas.style.width = `${w}px`;
            this._canvas.style.height = `${h}px`;
        }
        this._w.setSize(this._canvas.width, this._canvas.height);
        this._resetViewport(w, h);
    }
    getSize(target = new Vector2()) { return target.set(this._width, this._height); }
    getPixelRatio() { return this._pixelRatio; }
    /// three.js `setPixelRatio`: re-sizes the drawing buffer to keep the CSS
    /// size, and scales viewport/scissor rectangles from here on. Pages that
    /// size the canvas in device pixels themselves never call this and keep
    /// a ratio of 1.
    setPixelRatio(value) {
        if (value === undefined || !Number.isFinite(Number(value)) || Number(value) <= 0) return;
        this._pixelRatio = Number(value);
        this._w.setPixelRatio?.(this._pixelRatio);
        this.setSize(this._width, this._height, false);
    }
    getDrawingBufferSize(target = new Vector2()) {
        return target.set(Math.floor(this._width * this._pixelRatio), Math.floor(this._height * this._pixelRatio));
    }
    setDrawingBufferSize(width, height, pixelRatio) {
        this._width = width;
        this._height = height;
        this._pixelRatio = pixelRatio;
        this._w.setPixelRatio?.(pixelRatio);
        this._canvas.width = Math.floor(width * pixelRatio);
        this._canvas.height = Math.floor(height * pixelRatio);
        this._w.setSize(this._canvas.width, this._canvas.height);
        this._resetViewport(width, height);
    }
    /// Viewport back to the whole canvas. `resetViewport` rather than
    /// `setViewport(0, 0, w, h)`: an explicit viewport is what tells the WebGL2
    /// backend a page composes frames from several draws, and a resize is not that.
    _resetViewport(w, h) {
        this._viewport.set(0, 0, w, h);
        if (this._w.resetViewport) this._w.resetViewport();
        else this._w.setViewport?.(0, 0, w, h);
    }
    /// three.js `setViewport` / `setScissor` (bottom-left origin, CSS pixels,
    /// accepts a Vector4 too) and `setScissorTest`. Rendering several views
    /// into one canvas per frame works as in three.js:
    ///
    ///     renderer.autoClear = false; renderer.setScissorTest(true);
    ///     for each view: setViewport; setScissor; setClearColor; clear(); render
    setViewport(x, y, width, height) {
        if (x && typeof x === 'object') ({ x, y, z: width, w: height } = x);
        this._viewport.set(x, y, width, height);
        this._w.setViewport?.(x, y, width, height);
    }
    getViewport(target = new Vector4()) { return target.copy(this._viewport); }
    setScissor(x, y, width, height) {
        if (x && typeof x === 'object') ({ x, y, z: width, w: height } = x);
        this._scissor.set(x, y, width, height);
        this._w.setScissor?.(x, y, width, height);
    }
    getScissor(target = new Vector4()) { return target.copy(this._scissor); }
    setScissorTest(enabled) { this._scissorTest = !!enabled; this._w.setScissorTest?.(!!enabled); }
    getScissorTest() { return this._scissorTest; }
    get autoClear() { return this._autoClear ?? true; }
    set autoClear(v) { this._autoClear = !!v; this._w.setAutoClear?.(!!v); }
    /// three.js `setClearColor(color, alpha = 1)`: what `clear()` fills with,
    /// and the background of renders whose `scene.background` is null.
    setClearColor(color, alpha = 1) {
        const hex = _clearColorHex(color);
        if (hex == null) return;
        this._clearColor = new Color(hex);
        this._clearAlpha = Number.isFinite(Number(alpha)) ? Number(alpha) : 1;
        this._clearColorSet = true;
    }
    getClearColor(target = new Color()) { return target.setHex(this._clearColor.getHex()); }
    getClearAlpha() { return this._clearAlpha; }
    setClearAlpha(alpha) {
        this._clearAlpha = Number(alpha);
        this._clearColorSet = true;
    }
    /// three.js `clear(color, depth, stencil)`. Colour fills with the clear
    /// colour — only inside the scissor while the scissor test is on — on the
    /// canvas or the active render target. Depth needs no work here: every
    /// render clears depth itself; there is no stencil.
    clear(color = true, _depth = true, _stencil = true) {
        if (color) this._w.clear?.(_color(this._clearColor.getHex()), this._clearAlpha);
    }
    clearColor() { this.clear(true, false, false); }
    clearDepth() {}
    clearStencil() {}
    /// Point a null-background scene at the clear colour once one was set, and
    /// back at threers' default otherwise (the scene may be shared with a
    /// renderer that never set one).
    _syncClearBackground(scene) {
        if (!scene?._w || scene._background != null) return;
        if (this._clearColorSet) {
            scene._w.background = _color(this._clearColor.getHex());
            scene._w.setBackgroundAlpha?.(this._clearAlpha);
            scene._bgFromClearColor = true;
        } else if (scene._bgFromClearColor) {
            scene._w.background = _color(_DEFAULT_SCENE_BACKGROUND);
            scene._w.setBackgroundAlpha?.(scene._backgroundAlpha ?? 1);
            scene._bgFromClearColor = false;
        }
    }
    setRenderTarget(target) {
        if (target === null || target === undefined) {
            this._w.setRenderTarget(0);
            this._currentRenderTarget = null;
            return;
        }
        if (!target._w) {
            // Lazy-allocate the wgpu render target now that we have the renderer.
            target._w = (target._halfFloat || target.type === HalfFloatType) && WebRenderTarget.newHalfFloat
                ? WebRenderTarget.newHalfFloat(this._w, target.width, target.height)
                : new WebRenderTarget(this._w, target.width, target.height);
        }
        // Make sure target.texture._w is a Texture handle backed by the RT view.
        if (target.texture && !target.texture._w_isExternal) {
            const tex_w = this._w.renderTargetTexture(target._w);
            target.texture._w = tex_w;
            target.texture._w_isExternal = true;
        }
        _syncTargetColorSpace(target);
        this._w.setRenderTarget(target._w.id);
        this._currentRenderTarget = target;
    }
    getRenderTarget() { return this._currentRenderTarget || null; }
    /// Apply one of the built-in post-fx passes: read from `inputRT`, write to
    /// the canvas surface. `effectKind`: 0=copy/passthrough, 1=fxaa, 2=film,
    /// 3=dotscreen, 4=halftone, 5=glitch. `time` is seconds (for time-tied
    /// effects); `p2` is a `[x, y, z, w]` array of per-effect tuning values.
    /// For SSAO (kind 11), pass `depthRT`, `normalRT`, and `cam` from
    /// `_postFxCameraFrom()`.
    applyPostFx(inputRT, effectKind, time = 0, p2 = [0, 0, 0, 0], additive = false, depthRT = null, normalRT = null, cam = null, p3 = [0, 0, 0, 0]) {
        if (!inputRT?._w) return;
        const depthId = depthRT?._w?.id ?? 0;
        const normalId = normalRT?._w?.id ?? 0;
        const c = cam || _defaultPostFxCam();
        this._w.applyPostFx(
            inputRT._w.id, depthId, normalId, effectKind, time,
            p2[0], p2[1], p2[2], p2[3],
            p3[0], p3[1], p3[2], p3[3],
            additive ? 1 : 0,
            c.near, c.far, c.kernelRadius, c.kernelSize, c.proj, c.invProj,
        );
    }
    applyPostFxToRT(inputRT, outputRT, effectKind, time = 0, p2 = [0, 0, 0, 0], depthRT = null, normalRT = null, cam = null, p3 = [0, 0, 0, 0]) {
        if (!inputRT?._w) return;
        if (!outputRT._w) {
            outputRT._w = (outputRT._halfFloat || outputRT.type === HalfFloatType) && WebRenderTarget.newHalfFloat
                ? WebRenderTarget.newHalfFloat(this._w, outputRT.width, outputRT.height)
                : new WebRenderTarget(this._w, outputRT.width, outputRT.height);
        }
        if (outputRT.texture && !outputRT.texture._w_isExternal) {
            const tex_w = this._w.renderTargetTexture(outputRT._w);
            outputRT.texture._w = tex_w;
            outputRT.texture._w_isExternal = true;
        }
        const depthId = depthRT?._w?.id ?? 0;
        const normalId = normalRT?._w?.id ?? 0;
        const c = cam || _defaultPostFxCam();
        this._w.applyPostFxToRT(
            inputRT._w.id, outputRT._w.id, depthId, normalId, effectKind, time,
            p2[0], p2[1], p2[2], p2[3],
            p3[0], p3[1], p3[2], p3[3],
            c.near, c.far, c.kernelRadius, c.kernelSize, c.proj, c.invProj,
        );
    }
    async readRenderTargetPixels(target, x, y, width, height, buffer) {
        if (!target?._w) return;
        // three.js reads a `HalfFloatType` target into a `Uint16Array` of half
        // floats, and an 8-bit one as its colour space says: linear for the
        // default `NoColorSpace`, encoded for `SRGBColorSpace`.
        if (target._halfFloat) {
            const raw = await this._w.readRenderTargetF16(target._w.id, x, y, width, height);
            const halves = raw && new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength >> 1);
            if (buffer && halves) buffer.set(halves);
            return halves;
        }
        _syncTargetColorSpace(target);
        const bytes = await this._w.readRenderTargetPixels(target._w.id, x, y, width, height);
        if (buffer && bytes) buffer.set(bytes);
        return bytes;
    }
    async readRenderTargetF16(target, x = 0, y = 0, w, h) {
        if (!target?._w || !this._w?.readRenderTargetF16) return null;
        const width = w ?? target.width ?? this._canvas?.width ?? 800;
        const height = h ?? target.height ?? this._canvas?.height ?? 600;
        return this._w.readRenderTargetF16(target._w.id, x, y, width, height);
    }
    render(scene, camera) {
        if (scene && !scene.isScene && scene instanceof Object3D) {
            this._renderRoot(scene, camera);
            return;
        }
        this._prepareRender(scene, camera);
        this._stepFur(scene._w, camera);
        this._w.render(scene._w, camera._w);
    }
    /// threers: advance a scene's fur (wind, flutter, inertia) on this
    /// renderer's clock — the frame step clamped so a paused tab does not
    /// fling it.
    _stepFur(w, camera) {
        if (!w?.stepFur) return;
        const now = (globalThis.performance?.now?.() ?? Date.now()) / 1000;
        if (this._furStart === undefined) {
            this._furStart = now;
            this._furLast = now;
        }
        const dt = Math.min(Math.max(now - this._furLast, 1 / 240), 0.1);
        this._furLast = now;
        w.stepFur(now - this._furStart, dt);
        // Particles, snow and sand, on the same clock; alpha-blended
        // particles sorted from the camera.
        if (w.stepParticles) {
            _linkParticles();
            const e = camera?.matrixWorld?.elements;
            w.stepParticles(dt, e ? e[12] : 0, e ? e[13] : 0, e ? e[14] : 0);
        }
    }
    /// three.js draws any Object3D as the root (`renderer.render(gltf.scene,
    /// camera)` with the `Group` a GLTFLoader gives). An object outside every
    /// Scene is mirrored into the hidden graph, so that is what is drawn — with
    /// only this object's tree under its root for the draw, on the background
    /// a Scene without one gets. (An object inside a Scene draws that Scene.)
    _renderRoot(root, camera) {
        this._prepareRender(root, camera);
        const top = _mirrorRoot(root);
        if (top.isScene) {
            this._stepFur(top._w, camera);
            this._w.render(top._w, camera._w);
            return;
        }
        top._syncNow();
        const graph = _hiddenGraph();
        const w = graph.w;
        this._stepFur(w, camera);
        w.background = _color(this._clearColorSet ? this._clearColor.getHex() : _DEFAULT_SCENE_BACKGROUND);
        w.setBackgroundAlpha?.(this._clearColorSet ? this._clearAlpha : 1);
        const others = Array.from(graph.detachedIds);
        w.setChildren(graph.root, new Float64Array([top._m.id]));
        try {
            this._w.render(w, camera._w);
        } finally {
            w.setChildren(graph.root, new Float64Array(others));
        }
    }
    /// Everything `render` does before the draw: three.js's world-matrix
    /// update (which syncs the scene's wasm graph), material hooks,
    /// onBeforeRender, the camera pose and the clear-colour background.
    /// Shared with `QuadView.render`, which draws the same scene several times.
    _prepareRender(scene, camera) {
        // Fire material.onBeforeCompile once per material on first render, and
        // give materials a chance to re-upload mutated uniforms (e.g. Sky).
        const hooks = [];
        const clipped = [];
        const globalClip = this.clippingPlanes?.length > 0;
        scene._syncColors?.();
        scene._syncEnvironmentRotation?.();
        scene.traverse?.((o) => {
            const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
            for (const m of mats) {
                _syncMaterialColors(m);
                _syncMaterialTextures(m);
                m._syncVolumes?.();
                m._runOnBeforeCompile?.();
                if (m.updateUniforms) {
                    m.updateUniforms();
                    _materialChanged(m);
                }
            }
            if (o.onBeforeRender && o.onBeforeRender !== Object3D.prototype.onBeforeRender) hooks.push(o);
            // anything with a plane now, or one sent before that may need clearing
            if (mats.length && (o._m?.clip || globalClip || (this.localClippingEnabled && mats[0].clippingPlanes?.length))) clipped.push(o);
        });
        // Per-object onBeforeRender hooks (Reflector / Refractor / Water etc).
        // Guarded against recursion: each hook can render to its own RT but
        // sets _inRender to prevent infinite re-entry through this same path.
        if (!this._inRender && hooks.length) {
            this._inRender = true;
            camera?.updateWorldMatrix?.(true, false);
            for (const o of hooks) o.onBeforeRender(this, scene, camera);
            this._inRender = false;
        }
        // three.js: `scene.updateMatrixWorld()` and `camera.updateMatrixWorld()`;
        // the scene's also brings its wasm graph up to date.
        if (scene.matrixWorldAutoUpdate !== false) scene.updateMatrixWorld?.();
        else scene._syncNow?.();
        // after the sync, so objects added this frame have their nodes
        for (const o of clipped) _syncClipPlane(this, o);
        // Orbit/trackball controls own the wasm camera pose — don't push stale JS over it.
        if (!camera?._orbitControlled) camera?._sync?.();
        this._syncClearBackground(scene);
    }
}
export const WebGPURenderer = WebGLRenderer;

// A render target's `texture.colorSpace`, to the wasm target that decides
// what its read-back returns, and its `samples`, to the one that decides
// whether it is drawn multisampled.
function _syncTargetColorSpace(target) {
    target?._w?.setSrgbColorSpace?.(target.texture?.colorSpace === 'srgb');
    target?._w?.setSamples?.(target.samples | 0);
}

/// threers extension (not in three.js): a CAD-style quad viewport — top,
/// front and side orthographic panes (Y-up, third-angle) plus your
/// perspective camera, on one canvas. Pane names are `'top' | 'front' |
/// 'side' | 'perspective'`; rectangles are three.js's (bottom-left origin,
/// CSS pixels), so flip a DOM mouse y (`height - y`) before `paneAt`.
///
///     const quad = new QuadView();
///     quad.frameScene(scene);
///     function frame() { quad.render(renderer, scene, camera); }
export class QuadView {
    constructor() {
        const C = _optional.WebQuadView;
        if (!C) throw new Error('QuadView: this wasm build has no WebQuadView binding');
        this._w = new C();
    }
    /// Panes in grid order: top-left, top-right, bottom-left, bottom-right.
    get layout() { return this._w.layout(); }
    set layout(panes) { this._w.setLayout(panes[0], panes[1], panes[2], panes[3]); }
    get gap() { return this._w.gap; }
    set gap(v) { this._w.gap = v; }
    get border() { return this._w.border; }
    set border(v) { this._w.border = v; }
    get margin() { return this._w.margin; }
    set margin(v) { this._w.margin = v; }
    /// Colour of the gap and border; null leaves them untouched.
    set gapColor(c) {
        if (c == null) this._w.clearGapColor();
        else this._w.setGapColor(_color(_clearColorHex(c)));
    }
    /// Background for one pane, overriding `scene.background`; null removes it.
    setPaneBackground(pane, color) {
        if (color == null) this._w.clearPaneBackground(pane);
        else this._w.setPaneBackground(pane, _color(_clearColorHex(color)));
        return this;
    }
    /// The pane shown full-size, or null for all four.
    get maximized() { return this._w.maximized() ?? null; }
    set maximized(pane) { this._w.setMaximized(pane ?? undefined); }
    toggleMaximize(pane) { this._w.toggleMaximize(pane); return this; }
    /// Aim the orthographic cameras at a `Box3`.
    frameBox(box) {
        const b = box?._w ?? new WebBox3(
            new WebVector3(box.min.x, box.min.y, box.min.z),
            new WebVector3(box.max.x, box.max.y, box.max.z));
        this._w.frameBox(b);
        return this;
    }
    /// Frame every visible mesh; false when there is nothing to frame.
    frameScene(scene) {
        scene._syncTransforms?.();
        return this._w.frameScene(scene._w);
    }
    /// The framed box's centre — a natural `OrbitControls.target`.
    get target() { const t = this._w.target(); return new Vector3(t[0], t[1], t[2]); }
    /// `pane`'s rectangle as a `Vector4(x, y, width, height)` on a
    /// `width`×`height` canvas (CSS pixels; defaults to `renderer.getSize()`
    /// when a renderer is passed instead), or null when hidden.
    paneRect(pane, width, height) {
        if (width && typeof width === 'object') ({ x: width, y: height } = width.getSize(new Vector2()));
        const r = this._w.paneRect(pane, width, height);
        return r ? new Vector4(r[0], r[1], r[2], r[3]) : null;
    }
    /// The pane under `(x, y)` (bottom-left origin), or null over a gap.
    paneAt(x, y, width, height) {
        if (width && typeof width === 'object') ({ x: width, y: height } = width.getSize(new Vector2()));
        return this._w.paneAt(x, y, width, height) ?? null;
    }
    /// Fit the orthographic frusta to a canvas size (render does this itself).
    fit(width, height) { this._w.fit(width, height); return this; }
    /// The wasm orthographic camera for `'top'`, `'front'` or `'side'`, as
    /// last fitted — a `WebCamera` handle (readPosition / readTarget).
    getOrthoCamera(pane) { return this._w.orthoCamera(pane); }
    /// Draw all visible panes with `renderer` (to the canvas, or its active
    /// render target). `camera` must be a PerspectiveCamera; its aspect is set
    /// per pane on a copy. Renderer viewport/scissor/autoClear are restored.
    render(renderer, scene, camera) {
        this._renderer = renderer;
        this._camera ??= camera;
        renderer._prepareRender(scene, camera);
        this._w.render(renderer._w, scene._w, camera._w);
    }
    /// Magnification of the orthographic panes (one for all three); framing
    /// resets it to 1.
    get zoom() { return this._w.zoom; }
    set zoom(v) { this._w.zoom = v; }
    /// Move what the orthographic panes are centred on (world units).
    panBy(x, y, z) { this._w.panBy(x, y, z); return this; }
    /// Draw the shadow maps once per render rather than once per pane
    /// (default true; identical pictures).
    get shareShadows() { return this._w.shareShadows; }
    set shareShadows(v) { this._w.shareShadows = !!v; }
    /// Frame every visible mesh in every pane: orthographic panes at zoom 1,
    /// `camera` moved along its line of sight to fit.
    frameAll(scene, camera = this._camera) {
        scene._syncTransforms?.();
        if (!camera) return this.frameScene(scene);
        camera._sync?.();
        const ok = this._w.frameAll(scene._w, camera._w);
        this._readCamera(camera);
        return ok;
    }
    _readCamera(camera) {
        if (!camera?._w?.readPosition) return;
        const p = camera._w.readPosition();
        const t = camera._w.readTarget();
        camera.position.set(p.x, p.y, p.z);
        camera.lookAt?.(t.x, t.y, t.z);
    }
    /// Size the panes are laid out at: the renderer's (as `render` uses),
    /// else the element's CSS size.
    _viewSize(el) {
        const r = this._renderer;
        if (r?.getSize) { const s = r.getSize(new Vector2()); if (s.x > 0 && s.y > 0) return [s.x, s.y]; }
        return [Math.max(1, el?.clientWidth || 1), Math.max(1, el?.clientHeight || 1)];
    }
    /// Mouse/touch interaction on `domElement`: drag in the perspective pane
    /// orbits `camera` (right/middle drag pans), drag in an orthographic pane
    /// pans all three together, the wheel zooms (one scale for the three
    /// orthographic panes, about the cursor), double-click maximises or
    /// restores a pane. `onChange` (if set) is called after anything moves.
    connect(domElement, camera = this._camera) {
        this.disconnect();
        this._dom = domElement;
        if (camera) this._camera = camera;
        const at = (e) => {
            const rect = domElement.getBoundingClientRect();
            const [w, h] = this._viewSize(domElement);
            const sx = rect.width > 0 ? w / rect.width : 1;
            const sy = rect.height > 0 ? h / rect.height : 1;
            return [(e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy, w, h];
        };
        const changed = () => { this._readCamera(this._camera); this.onChange?.(); };
        const push = () => {
            const cam = this._camera;
            if (cam && !cam._orbitControlled) cam._sync?.();
        };
        const doc = domElement.ownerDocument || (typeof document !== 'undefined' ? document : null);
        let pointerId = null;
        const move = (e) => {
            if (e.pointerId !== pointerId || !this._camera) return;
            const [x, y] = at(e);
            if (this._w.pointerMove(this._camera._w, x, y)) changed();
        };
        const up = (e) => {
            if (pointerId == null || e.pointerId !== pointerId) return;
            try { domElement.releasePointerCapture?.(pointerId); } catch (_) { /* already released */ }
            pointerId = null;
            this._w.pointerUp();
            doc?.removeEventListener('pointermove', move);
            doc?.removeEventListener('pointerup', up);
            doc?.removeEventListener('pointercancel', up);
        };
        const down = (e) => {
            if (!this._camera || pointerId != null) return;
            push();
            const [x, y, w, h] = at(e);
            const pane = this._w.pointerDown(this._camera._w, x, y, e.button, w, h);
            if (pane == null) return;
            e.preventDefault();
            pointerId = e.pointerId;
            domElement.setPointerCapture?.(pointerId);
            doc?.addEventListener('pointermove', move);
            doc?.addEventListener('pointerup', up);
            doc?.addEventListener('pointercancel', up);
        };
        const wheel = (e) => {
            if (!this._camera) return;
            push();
            const [x, y, w, h] = at(e);
            if (this._w.wheel(this._camera._w, x, y, e.deltaY, w, h)) {
                e.preventDefault();
                changed();
            }
        };
        const dbl = (e) => {
            const [x, y, w, h] = at(e);
            if (this._w.dblclick(x, y, w, h) != null) { e.preventDefault(); this.onChange?.(); }
        };
        const menu = (e) => e.preventDefault();
        domElement.addEventListener('pointerdown', down);
        domElement.addEventListener('wheel', wheel, { passive: false });
        domElement.addEventListener('dblclick', dbl);
        domElement.addEventListener('contextmenu', menu);
        if (domElement.style && !domElement.style.touchAction) domElement.style.touchAction = 'none';
        this._listeners = () => {
            up({ pointerId });
            domElement.removeEventListener('pointerdown', down);
            domElement.removeEventListener('wheel', wheel);
            domElement.removeEventListener('dblclick', dbl);
            domElement.removeEventListener('contextmenu', menu);
        };
        return this;
    }
    /// Remove what `connect` added.
    disconnect() {
        this._listeners?.();
        this._listeners = null;
        this._dom = null;
        return this;
    }
}

/**
 * Render frames of `scene` through `renderer` into RGBA buffers.
 *
 * Size defaults to the canvas drawing-buffer size when `width`/`height` omitted.
 * Prefer `frames` or `duration` (seconds × fps) for length.
 * WebM capture sizes snap down to multiples of 8 unless `strictSize: true`.
 *
 * Set `parallel: true` (or `concurrency: N`) to pipeline GPU readbacks across
 * multiple render targets so the next frame can render while a prior map_async
 * completes. Default is sequential (`concurrency: 1`).
 *
 * @returns {Promise<Uint8Array[]>}
 */
export async function captureSceneFrames(renderer, scene, camera, options = {}) {
    const canvas = renderer?.domElement;
    let width = (options.width | 0) || (canvas?.width | 0);
    let height = (options.height | 0) || (canvas?.height | 0);
    const fps = Math.max(1, (options.fps | 0) || 30);
    let frameCount = options.frameCount | 0;
    if (!frameCount && options.frames) frameCount = options.frames | 0;
    if (!frameCount && options.duration) {
        frameCount = Math.max(1, Math.round(Number(options.duration) * fps));
    }

    const formatInfo = options.format != null
        ? parseVideoFormat(options.format)
        : { format: VideoFormat.Gif, transparent: false };
    const format = formatInfo.format;

    if (format === VideoFormat.Webm && width > 0 && height > 0) {
        const aligned = alignVideoSize(width, height);
        if (aligned.snapped) {
            if (options.strictSize) {
                throw new VideoExportError(
                    `WebM/VP9 requires width and height multiples of 8 (got ${width}×${height})`,
                    VideoExportErrorCode.InvalidDimensions,
                    { hint: `use ${aligned.width}×${aligned.height}` },
                );
            }
            width = aligned.width;
            height = aligned.height;
        }
    }

    if (width <= 0 || height <= 0 || frameCount <= 0) {
        const missing = [];
        if (width <= 0) missing.push('width');
        if (height <= 0) missing.push('height');
        if (frameCount <= 0) missing.push('frames|duration');
        throw new VideoExportError(
            `captureSceneFrames: missing ${missing.join(', ')}`,
            VideoExportErrorCode.InvalidOption,
            { hint: 'pass size()/frames() or ensure the canvas has a drawing buffer' },
        );
    }

    const concurrency = resolveCaptureConcurrency(options, frameCount);
    const signal = options.signal ?? null;
    const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
    let mapFrame = typeof options.mapFrame === 'function' ? options.mapFrame : null;
    if (!mapFrame && options.transparentCornerPunch) {
        const punchSize = options.transparentCornerPunch === true
            ? Math.min(40, width, height)
            : Math.min(Math.max(1, options.transparentCornerPunch | 0), width, height);
        mapFrame = (rgba) => {
            const copy = new Uint8Array(rgba);
            for (let y = 0; y < punchSize; y++) {
                for (let x = 0; x < punchSize; x++) {
                    copy[(y * width + x) * 4 + 3] = 0;
                }
            }
            return copy;
        };
    }

    const ownedTargets = [];
    /** @type {InstanceType<typeof WebGLRenderTarget>[]} */
    let targets;
    if (Array.isArray(options.renderTargets) && options.renderTargets.length > 0) {
        targets = options.renderTargets.slice(0, concurrency);
        while (targets.length < concurrency) {
            const rt = new WebGLRenderTarget(width, height);
            ownedTargets.push(rt);
            targets.push(rt);
        }
    } else if (options.renderTarget && concurrency === 1) {
        targets = [options.renderTarget];
    } else {
        targets = [];
        for (let i = 0; i < concurrency; i++) {
            const rt = (i === 0 && options.renderTarget)
                ? options.renderTarget
                : new WebGLRenderTarget(width, height);
            if (rt !== options.renderTarget) ownedTargets.push(rt);
            targets.push(rt);
        }
    }

    const prev = renderer.getRenderTarget?.() ?? null;
    /** @type {(Uint8Array|undefined)[]} */
    const frames = new Array(frameCount);
    /** @type {({ index: number, promise: Promise<Uint8Array> }|null)[]} */
    const slots = new Array(concurrency).fill(null);
    const done = new Array(frameCount).fill(false);
    let reported = 0;

    const throwIfAborted = () => {
        if (signal?.aborted) {
            throw new VideoExportError('Video export aborted', VideoExportErrorCode.Aborted, {
                cause: signal.reason,
            });
        }
    };

    const eventTarget = options.eventTarget
        || (options.emitter && typeof options.emitter.dispatchEvent === 'function' ? options.emitter : null);

    const reportThrough = () => {
        while (reported < frameCount && done[reported]) {
            reported += 1;
            const info = {
                phase: 'capture',
                frame: reported,
                frames: frameCount,
                ratio: reported / (frameCount + 1),
                format,
            };
            info.message = formatVideoProgress(info);
            emitProgress(onProgress, info, eventTarget);
        }
    };

    const settleSlot = async (slot) => {
        const job = slots[slot];
        if (!job) return;
        slots[slot] = null;
        const rgba = await job.promise;
        frames[job.index] = rgba;
        done[job.index] = true;
        if (typeof options.onFrame === 'function') options.onFrame(job.index, frameCount);
        reportThrough();
    };

    try {
        if (eventTarget) {
            eventTarget.dispatchEvent(new CustomEvent(VideoExportEvent.Start, {
                detail: { phase: 'capture', frames: frameCount, format, concurrency },
            }));
        }
        for (let i = 0; i < frameCount; i++) {
            throwIfAborted();
            const slot = i % concurrency;
            if (slots[slot]) await settleSlot(slot);

            if (typeof options.update === 'function') options.update(i, frameCount);
            const rt = targets[slot];
            renderer.setRenderTarget(rt);
            renderer.render(scene, camera);

            // Kick off async GPU readback without awaiting — next iterations can
            // render into other targets while this map_async completes.
            const index = i;
            const promise = Promise.resolve(
                renderer.readRenderTargetPixels(rt, 0, 0, width, height),
            ).then((raw) => {
                let rgba = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
                if (mapFrame) rgba = mapFrame(rgba, index, frameCount);
                return rgba;
            });
            slots[slot] = { index, promise };

            if (options.yield !== false) {
                await new Promise((r) => setTimeout(r, 0));
            }
        }

        for (let slot = 0; slot < concurrency; slot++) {
            if (slots[slot]) await settleSlot(slot);
        }
    } catch (err) {
        if (eventTarget) {
            const aborted = VideoExportError.is(err) && err.code === VideoExportErrorCode.Aborted;
            eventTarget.dispatchEvent(new CustomEvent(
                aborted ? VideoExportEvent.Abort : VideoExportEvent.Error,
                { detail: { error: err } },
            ));
        }
        throw err;
    } finally {
        renderer.setRenderTarget(prev);
        for (const rt of ownedTargets) {
            try { rt.dispose?.(); } catch { /* ignore */ }
        }
    }

    return frames;
}

/** @param {object} options @param {number} frameCount */
function resolveCaptureConcurrency(options, frameCount) {
    let n = 1;
    if (options.concurrency != null) {
        n = Number(options.concurrency) || 1;
    } else if (options.parallel === true) {
        n = 2;
    } else if (typeof options.parallel === 'number') {
        n = options.parallel;
    } else if (options.parallel) {
        n = 2;
    }
    n = Math.max(1, Math.min(8, n | 0));
    return Math.min(n, Math.max(1, frameCount | 0));
}

/**
 * Capture a scene animation and encode it to GIF / APNG / WebM.
 * @returns {Promise<import('./video-export.js').VideoExportResult>}
 */
export async function exportSceneVideo(renderer, scene, camera, options = {}) {
    const canvas = renderer?.domElement;
    let width = (options.width | 0) || (canvas?.width | 0);
    let height = (options.height | 0) || (canvas?.height | 0);
    const fps = Math.max(1, (options.fps | 0) || 30);
    const parsed = options.format != null
        ? parseVideoFormat(options.format)
        : { format: VideoFormat.Gif, transparent: false };
    const format = parsed.format;
    const transparent = options.transparent != null ? !!options.transparent : parsed.transparent;

    if (format === VideoFormat.Webm && width > 0 && height > 0) {
        const aligned = alignVideoSize(width, height);
        if (aligned.snapped && !options.strictSize) {
            width = aligned.width;
            height = aligned.height;
        }
    }

    const eventTarget = options.eventTarget
        || (options.emitter && typeof options.emitter.dispatchEvent === 'function' ? options.emitter : null);

    const frames = await captureSceneFrames(renderer, scene, camera, {
        ...options,
        width,
        height,
        fps,
        format,
        transparent,
        eventTarget,
    });
    const exporter = (eventTarget instanceof VideoExporter)
        ? eventTarget.configure({
            width,
            height,
            fps,
            format,
            transparent,
            gifColors: options.gifColors,
            filename: options.filename,
            basename: options.basename,
            onProgress: options.onProgress,
            signal: options.signal,
            strictSize: options.strictSize,
        })
        : new VideoExporter({
            width,
            height,
            fps,
            format,
            transparent,
            gifColors: options.gifColors,
            filename: options.filename,
            basename: options.basename,
            onProgress: options.onProgress,
            signal: options.signal,
            strictSize: options.strictSize,
            eventTarget,
        });
    if (options.worker || options.encodeInWorker) {
        return exporter.encodeInWorker(frames, {
            wasmUrl: options.wasmUrl,
            workerUrl: options.workerUrl,
            worker: options.worker instanceof VideoEncodeWorker ? options.worker : undefined,
            onProgress: options.onProgress,
            signal: options.signal,
            eventTarget: exporter,
        });
    }
    return exporter.encode(frames);
}

/**
 * Capture, encode, and trigger a file download.
 * @returns {Promise<import('./video-export.js').VideoExportResult>}
 */
export async function exportAndDownloadSceneVideo(renderer, scene, camera, options = {}) {
    const result = await exportSceneVideo(renderer, scene, camera, options);
    return result.download(options.filename);
}

/**
 * Bind a renderer/scene/camera and configure a fluent scene export.
 *
 * @example
 * ```js
 * await VideoExporter.from(renderer, scene, camera)
 *   .gif({ transparent: true })
 *   .parallel(3)
 *   .frames(30)
 *   .update((i, n) => { cube.rotation.y = (i / n) * Math.PI * 2; })
 *   .onProgress(({ message }) => setStatus(message))
 *   .download('cube');
 * ```
 */
VideoExporter.from = function from(renderer, scene, camera, options = {}) {
    const api = new VideoExporter(options);
    let frameCount = options.frameCount || options.frames || 0;
    let duration = options.duration || 0;
    let update = typeof options.update === 'function' ? options.update : null;
    let yieldBetween = options.yield !== false;
    let cornerPunch = options.transparentCornerPunch || false;
    let concurrency = resolveCaptureConcurrency(options, 1);

    api.frames = function frames(n) {
        frameCount = n | 0;
        return api;
    };
    api.duration = function durationSec(seconds) {
        duration = Number(seconds) || 0;
        return api;
    };
    api.update = function updateFn(fn) {
        update = fn;
        return api;
    };
    api.yieldBetweenFrames = function yieldBetweenFrames(on = true) {
        yieldBetween = !!on;
        return api;
    };
    api.transparentCornerPunch = function transparentCornerPunch(on = true) {
        cornerPunch = on;
        return api;
    };
    /**
     * Pipeline GPU readbacks across N render targets (1–8).
     * `true` → 2; a number sets concurrency explicitly.
     * @param {boolean|number} [n=true]
     */
    api.parallel = function parallel(n = true) {
        concurrency = resolveCaptureConcurrency({ parallel: n }, 1);
        return api;
    };
    /** Alias for {@link parallel}. @param {number} n */
    api.concurrency = function concurrencyFn(n) {
        concurrency = resolveCaptureConcurrency({ concurrency: n }, 1);
        return api;
    };
    let useWorker = !!(options.worker || options.encodeInWorker);
    api.worker = function worker(on = true) {
        useWorker = !!on;
        return api;
    };

    async function run(doDownload, filenameOverride) {
        const canvas = renderer?.domElement;
        let width = (api._opts.width | 0) || (canvas?.width | 0);
        let height = (api._opts.height | 0) || (canvas?.height | 0);
        if (api._opts.format === VideoFormat.Webm && width > 0 && height > 0) {
            const aligned = alignVideoSize(width, height);
            if (aligned.snapped && !api._opts.strictSize) {
                width = aligned.width;
                height = aligned.height;
            }
        }
        api.size(width, height);
        if (!frameCount && !duration) {
            frameCount = 30; // sensible default for demos / quick exports
        }
        const opts = {
            width,
            height,
            fps: api._opts.fps,
            format: api._opts.format,
            transparent: api._opts.transparent,
            gifColors: api._opts.gifColors,
            filename: filenameOverride || api._opts.filename,
            basename: api._opts.basename,
            frameCount,
            duration,
            frames: frameCount,
            update,
            mapFrame: api._mapFrame,
            transparentCornerPunch: cornerPunch,
            onProgress: api._onProgress,
            signal: api._signal,
            yield: yieldBetween,
            strictSize: api._opts.strictSize,
            concurrency,
            parallel: concurrency > 1,
            encodeInWorker: useWorker,
            wasmUrl: options.wasmUrl,
            workerUrl: options.workerUrl,
            eventTarget: api,
        };
        const result = await exportSceneVideo(renderer, scene, camera, opts);
        return doDownload ? result.download(opts.filename) : result;
    }

    api.export = (overrides) => {
        if (overrides && typeof overrides === 'object') {
            if (overrides.frames != null) frameCount = overrides.frames | 0;
            if (overrides.frameCount != null) frameCount = overrides.frameCount | 0;
            if (overrides.duration != null) duration = Number(overrides.duration) || 0;
            if (typeof overrides.update === 'function') update = overrides.update;
            if (overrides.parallel != null || overrides.concurrency != null) {
                concurrency = resolveCaptureConcurrency(overrides, 1);
            }
            api.configure(overrides);
        }
        return run(false);
    };
    api.download = (filenameOrOverrides) => {
        if (filenameOrOverrides && typeof filenameOrOverrides === 'object') {
            const o = filenameOrOverrides;
            if (o.frames != null) frameCount = o.frames | 0;
            if (o.frameCount != null) frameCount = o.frameCount | 0;
            if (o.duration != null) duration = Number(o.duration) || 0;
            if (typeof o.update === 'function') update = o.update;
            if (o.parallel != null || o.concurrency != null) {
                concurrency = resolveCaptureConcurrency(o, 1);
            }
            api.configure(o);
            return run(true, o.filename);
        }
        return run(true, typeof filenameOrOverrides === 'string' ? filenameOrOverrides : undefined);
    };
    return api;
};

// ---- Pass 5: remaining geometries (LatheGeometry, TubeGeometry,
// ExtrudeGeometry, EdgesGeometry, WireframeGeometry, ShapeGeometry,
// ConvexGeometry, DecalGeometry, TextGeometry, ParametricGeometry).
//
// Each is a thin `extends BufferGeometry` wrapper over the Rust generator
// (src/geometries, bound in src/wasm/generators.rs), whose buffers match
// three.js r165's; `_fillFromWasm` copies them back so three.js code can read
// `attributes` / `index` / `groups`.

/// Flatten `Vector2`s / `Vector3`s (or `[x, y(, z)]` arrays) for a binding.
function _flatPoints(points, dims) {
    const out = new Float64Array(points.length * dims);
    for (let i = 0; i < points.length; i++) {
        const p = points[i];
        out[i * dims] = p.x ?? p[0];
        out[i * dims + 1] = p.y ?? p[1];
        if (dims === 3) out[i * dims + 2] = p.z ?? p[2];
    }
    return out;
}
/// The `WebBufferGeometry` of any geometry the shim knows — a JS
/// BufferGeometry (synced), a built-in (`_w`), or a raw wasm geometry — as
/// `{ wasm, owned }`, `owned` when the caller must free it; null if none.
/// A shim `BoxGeometry` is read as three.js builds it (triangles, no quad
/// polygons), so `EdgesGeometry` / `WireframeGeometry` of one match three.js.
function _wasmSourceOf(geometry) {
    if (!geometry) return null;
    if (geometry instanceof BoxGeometry) {
        const p = geometry.parameters ?? {};
        const w = WebGeometry.threeBox(p.width ?? 1, p.height ?? 1, p.depth ?? 1,
            p.widthSegments ?? 1, p.heightSegments ?? 1, p.depthSegments ?? 1);
        const wasm = w.toBufferGeometry();
        w.free?.();
        return { wasm, owned: true };
    }
    if (typeof geometry._syncWasmFromJs === 'function') {
        if (!geometry.attributes?.position?.array?.length) return null;
        return { wasm: geometry._syncWasmFromJs(), owned: false };
    }
    if (geometry._w?.toBufferGeometry) return { wasm: geometry._w.toBufferGeometry(), owned: true };
    if (geometry.toBufferGeometry) return { wasm: geometry.toBufferGeometry(), owned: true };
    return null;
}

// CapsuleGeometry — three.js r165 (a lathe of two arcs), generated in Rust.
export class CapsuleGeometry extends BufferGeometry {
    constructor(radius = 1, length = 1, capSegments = 4, radialSegments = 8) {
        super();
        this.type = 'CapsuleGeometry';
        this.parameters = { radius, length, capSegments, radialSegments };
        _fillFromWasm(this, WebGeometry.capsule(radius, length, Math.floor(capSegments), Math.floor(radialSegments)));
    }
}

// LatheGeometry — three.js r165 `new LatheGeometry(points, segments, phiStart, phiLength)`.
// three.js r165's primitives: each a BufferGeometry with three.js's
// constructor, `type` and `parameters`, its index, attributes and groups
// built in Rust from the same numbers, bit for bit as three.js builds them
// (`tests/primitives_upstream.rs`).
function _primitive(target, type, parameters, args) {
    target.type = type;
    target.parameters = parameters;
    return _fillFromWasm(target, WebGeometry.threePrimitive(type, Float64Array.from(args, Number)));
}
export class BoxGeometry extends BufferGeometry {
    constructor(width = 1, height = 1, depth = 1, widthSegments = 1, heightSegments = 1, depthSegments = 1) {
        super();
        _primitive(this, 'BoxGeometry', { width, height, depth, widthSegments, heightSegments, depthSegments },
            [width, height, depth, widthSegments, heightSegments, depthSegments]);
    }
}
export class SphereGeometry extends BufferGeometry {
    constructor(radius = 1, widthSegments = 32, heightSegments = 16, phiStart = 0, phiLength = Math.PI * 2,
                thetaStart = 0, thetaLength = Math.PI) {
        super();
        _primitive(this, 'SphereGeometry',
            { radius, widthSegments, heightSegments, phiStart, phiLength, thetaStart, thetaLength },
            [radius, widthSegments, heightSegments, phiStart, phiLength, thetaStart, thetaLength]);
    }
}
export class PlaneGeometry extends BufferGeometry {
    constructor(width = 1, height = 1, widthSegments = 1, heightSegments = 1) {
        super();
        _primitive(this, 'PlaneGeometry', { width, height, widthSegments, heightSegments },
            [width, height, widthSegments, heightSegments]);
    }
}
export class CylinderGeometry extends BufferGeometry {
    constructor(radiusTop = 1, radiusBottom = 1, height = 1, radialSegments = 32, heightSegments = 1,
                openEnded = false, thetaStart = 0, thetaLength = Math.PI * 2) {
        super();
        _primitive(this, 'CylinderGeometry',
            { radiusTop, radiusBottom, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength },
            [radiusTop, radiusBottom, height, radialSegments, heightSegments, openEnded ? 1 : 0, thetaStart, thetaLength]);
    }
}
export class ConeGeometry extends BufferGeometry {
    constructor(radius = 1, height = 1, radialSegments = 32, heightSegments = 1, openEnded = false,
                thetaStart = 0, thetaLength = Math.PI * 2) {
        super();
        // three.js keeps CylinderGeometry's parameter names on a cone.
        _primitive(this, 'ConeGeometry',
            { radiusTop: 0, radiusBottom: radius, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength },
            [radius, height, radialSegments, heightSegments, openEnded ? 1 : 0, thetaStart, thetaLength]);
        this.parameters.radius = radius;
    }
}
export class TorusGeometry extends BufferGeometry {
    constructor(radius = 1, tube = 0.4, radialSegments = 12, tubularSegments = 48, arc = Math.PI * 2) {
        super();
        _primitive(this, 'TorusGeometry', { radius, tube, radialSegments, tubularSegments, arc },
            [radius, tube, radialSegments, tubularSegments, arc]);
    }
}
export class CircleGeometry extends BufferGeometry {
    constructor(radius = 1, segments = 32, thetaStart = 0, thetaLength = Math.PI * 2) {
        super();
        _primitive(this, 'CircleGeometry', { radius, segments, thetaStart, thetaLength },
            [radius, segments, thetaStart, thetaLength]);
    }
}
export class RingGeometry extends BufferGeometry {
    constructor(innerRadius = 0.5, outerRadius = 1, thetaSegments = 32, phiSegments = 1, thetaStart = 0,
                thetaLength = Math.PI * 2) {
        super();
        _primitive(this, 'RingGeometry',
            { innerRadius, outerRadius, thetaSegments, phiSegments, thetaStart, thetaLength },
            [innerRadius, outerRadius, thetaSegments, phiSegments, thetaStart, thetaLength]);
    }
}
export class TorusKnotGeometry extends BufferGeometry {
    constructor(radius = 1, tube = 0.4, tubularSegments = 64, radialSegments = 8, p = 2, q = 3) {
        super();
        _primitive(this, 'TorusKnotGeometry', { radius, tube, tubularSegments, radialSegments, p, q },
            [radius, tube, tubularSegments, radialSegments, p, q]);
    }
}
export class TetrahedronGeometry extends BufferGeometry {
    constructor(radius = 1, detail = 0) {
        super();
        _primitive(this, 'TetrahedronGeometry', { radius, detail }, [radius, detail]);
    }
}
export class OctahedronGeometry extends BufferGeometry {
    constructor(radius = 1, detail = 0) {
        super();
        _primitive(this, 'OctahedronGeometry', { radius, detail }, [radius, detail]);
    }
}
export class IcosahedronGeometry extends BufferGeometry {
    constructor(radius = 1, detail = 0) {
        super();
        _primitive(this, 'IcosahedronGeometry', { radius, detail }, [radius, detail]);
    }
}
export class DodecahedronGeometry extends BufferGeometry {
    constructor(radius = 1, detail = 0) {
        super();
        _primitive(this, 'DodecahedronGeometry', { radius, detail }, [radius, detail]);
    }
}
export class LatheGeometry extends BufferGeometry {
    constructor(points = [new Vector2(0, -0.5), new Vector2(0.5, 0), new Vector2(0, 0.5)],
                segments = 12, phiStart = 0, phiLength = Math.PI * 2) {
        super();
        this.type = 'LatheGeometry';
        this.parameters = { points, segments, phiStart, phiLength };
        _fillFromWasm(this, WebGeometry.lathe(_flatPoints(points, 2), Math.floor(segments), phiStart, phiLength));
    }
}

// TubeGeometry — three.js r165. The path is any curve (a JS `Curve` subclass
// included, called back from Rust); sampling, Frenet frames and the tube are
// all built in Rust.
export class TubeGeometry extends BufferGeometry {
    constructor(path = new QuadraticBezierCurve3(new Vector3(-1, -1, 0), new Vector3(-1, 1, 0), new Vector3(1, 1, 0)),
                tubularSegments = 64, radius = 1, radialSegments = 8, closed = false) {
        super();
        this.type = 'TubeGeometry';
        this.parameters = { path, tubularSegments, radius, radialSegments, closed };
        const result = WebGeometry.tubeFromCurve(_webCurveOf(path), tubularSegments, radius, radialSegments, closed);
        // three.js exposes the frames it built from.
        const f = result.frames(), n = f.length / 9;
        const at = (k) => Array.from({ length: n }, (_, i) => new Vector3(f[(k * n + i) * 3], f[(k * n + i) * 3 + 1], f[(k * n + i) * 3 + 2]));
        this.tangents = at(0);
        this.normals = at(1);
        this.binormals = at(2);
        _fillFromWasm(this, result.geometry());
        result.free();
    }
}

// ShapeGeometry / ExtrudeGeometry — three.js r165, generated in Rust
// (`geometries::{ShapeGeometry, ExtrudeGeometry}` over `curves::shapes`).
export class ShapeGeometry extends BufferGeometry {
    constructor(shapes = new Shape([new Vector2(0, 0.5), new Vector2(-0.5, -0.5), new Vector2(0.5, -0.5)]),
                curveSegments = 12) {
        super();
        this.type = 'ShapeGeometry';
        this.parameters = { shapes, curveSegments };
        const list = _wasmShapeList(shapes);
        _fillFromWasm(this, WebGeometry.shapeGeometry(list, curveSegments, Array.isArray(shapes)));
        list.free();
    }
}

/// A custom `options.UVGenerator`: three.js calls it per face on the
/// unindexed vertices as they are built. Rust builds them; this replays the
/// calls over the finished vertices in the same order (cap triangles, then
/// side quads, per shape) and replaces the world UVs.
function _extrudeCustomUVs(geometry, result, uvgen) {
    // Grown face by face, as three.js's `verticesArray` is when it calls.
    const all = result.positions();
    const vertices = [];
    const uv = [];
    let n = 0;
    for (const kind of result.faceKinds()) {
        const add = kind === 3 ? 3 : 6;
        for (let k = n * 3, end = (n + add) * 3; k < end; k++) vertices.push(all[k]);
        if (kind === 3) {
            n += 3;
            for (const p of uvgen.generateTopUV(geometry, vertices, n - 3, n - 2, n - 1)) uv.push(p.x, p.y);
        } else {
            n += 6;
            const q = uvgen.generateSideWallUV(geometry, vertices, n - 6, n - 3, n - 2, n - 1);
            for (const k of [0, 1, 3, 1, 2, 3]) uv.push(q[k].x, q[k].y);
        }
    }
    geometry.setAttribute('uv', new BufferAttribute(new Float32Array(uv), 2));
}

export class ExtrudeGeometry extends BufferGeometry {
    constructor(shapes = new Shape([
        new Vector2(0.5, 0.5), new Vector2(-0.5, 0.5),
        new Vector2(-0.5, -0.5), new Vector2(0.5, -0.5),
    ]), options = {}) {
        super();
        this.type = 'ExtrudeGeometry';
        this.parameters = { shapes, options };
        const steps = options.steps ?? 1;
        const list = _wasmShapeList(shapes);
        // An `extrudePath` is any curve (a JS `Curve` subclass included);
        // Rust samples it as three.js does and builds the sweep.
        const result = options.extrudePath
            ? WebGeometry.extrudeAlongCurve(list, options.curveSegments ?? 12, steps, options.bevelThickness ?? 0.2,
                _webCurveOf(options.extrudePath))
            : WebGeometry.extrude(
                list,
                options.curveSegments ?? 12, steps, options.depth ?? 1,
                options.bevelEnabled ?? true, options.bevelThickness ?? 0.2,
                options.bevelSize ?? NaN, options.bevelOffset ?? 0, options.bevelSegments ?? 3,
            );
        list.free();
        _fillFromWasm(this, result.geometry());
        if (options.UVGenerator) _extrudeCustomUVs(this, result, options.UVGenerator);
        result.free();
    }
}

// ConvexHull — three.js r165 addon (examples/jsm/math/ConvexHull.js). The
// quickhull runs in Rust (`WebConvexHull`, a faithful port, so faces come out
// in three.js's order); this class reads the result back into three.js-shaped
// objects: `vertices` (VertexNode-like `{ point }`), and `faces` with
// `normal`, `midpoint`, `area`, `constant`, `mark` and a half-edge loop
// (`edge`, `next`, `prev`, `twin`, `head()`, `tail()`).
export class ConvexHull {
    constructor() {
        this.tolerance = -1;
        this.faces = [];
        this.newFaces = [];
        this.vertices = [];
        this._w = new WebConvexHull();
    }
    setFromPoints(points) {
        if (points.length >= 4) {
            this._w.setFromPoints(_flatPoints(points, 3));
            this.vertices = points.map((point) => ({ point, prev: null, next: null, face: null }));
            this._readFaces();
        }
        return this;
    }
    setFromObject(object) {
        const points = [];
        object.updateMatrixWorld?.(true);
        const visit = (node) => {
            const attribute = node.geometry?.attributes?.position;
            if (attribute) {
                const a = _attrPacked(attribute), s = attribute.itemSize || 3, e = node.matrixWorld.elements;
                for (let i = 0, l = attribute.count; i < l; i++) {
                    const x = a[i * s], y = a[i * s + 1], z = a[i * s + 2];
                    const w = 1 / (e[3] * x + e[7] * y + e[11] * z + e[15]);
                    points.push(new Vector3(
                        (e[0] * x + e[4] * y + e[8] * z + e[12]) * w,
                        (e[1] * x + e[5] * y + e[9] * z + e[13]) * w,
                        (e[2] * x + e[6] * y + e[10] * z + e[14]) * w));
                }
            }
            for (const c of node.children ?? []) visit(c);
        };
        visit(object);
        return this.setFromPoints(points);
    }
    containsPoint(point) { return this._w.containsPoint(point.x, point.y, point.z); }
    intersectRay(ray, target) {
        const o = ray.origin, d = ray.direction;
        const t = this._w.intersectRay(o.x, o.y, o.z, d.x, d.y, d.z);
        return t === undefined ? null : ray.at(t, target);
    }
    intersectsRay(ray) { return this.intersectRay(ray, new Vector3()) !== null; }
    makeEmpty() { this.faces = []; this.vertices = []; this._w.makeEmpty(); return this; }
    _readFaces() {
        this.tolerance = this._w.tolerance;
        const loops = this._w.faceVertexIndices();
        const data = this._w.faceData();
        const faces = [];
        const byEnds = new Map();
        const halfEdge = (vertex, face) => ({
            vertex, prev: null, next: null, twin: null, face,
            head() { return this.vertex; },
            tail() { return this.prev ? this.prev.vertex : null; },
            length() { const t = this.tail(); return t ? t.point.distanceTo(this.vertex.point) : -1; },
            lengthSquared() { const t = this.tail(); return t ? t.point.distanceToSquared(this.vertex.point) : -1; },
            setTwin(e) { this.twin = e; e.twin = this; return this; },
        });
        for (let i = 0, f = 0; i < loops.length; f++) {
            const n = loops[i++];
            const d = f * 8;
            const face = {
                normal: new Vector3(data[d], data[d + 1], data[d + 2]),
                midpoint: new Vector3(data[d + 3], data[d + 4], data[d + 5]),
                constant: data[d + 6], area: data[d + 7],
                outside: null, mark: 0, edge: null,
                distanceToPoint(p) { return this.normal.dot(p) - this.constant; },
                getEdge(k) {
                    let e = this.edge;
                    while (k > 0) { e = e.next; k--; }
                    while (k < 0) { e = e.prev; k++; }
                    return e;
                },
            };
            const edges = [];
            for (let k = 0; k < n; k++) edges.push(halfEdge(this.vertices[loops[i + k]], face));
            for (let k = 0; k < n; k++) {
                edges[k].next = edges[(k + 1) % n];
                edges[(k + 1) % n].prev = edges[k];
            }
            face.edge = edges[0];
            for (let k = 0; k < n; k++) {
                const tail = loops[i + (k + n - 1) % n], head = loops[i + k];
                const twin = byEnds.get(`${head},${tail}`);
                if (twin) edges[k].setTwin(twin);
                else byEnds.set(`${tail},${head}`, edges[k]);
            }
            faces.push(face);
            i += n;
        }
        this.faces = faces;
    }
}

// ConvexGeometry — three.js r165 addon: one triangle per hull face, unindexed,
// face normals; built in Rust from the same quickhull.
export class ConvexGeometry extends BufferGeometry {
    constructor(points = []) {
        super();
        _fillFromWasm(this, WebGeometry.convex(_flatPoints(points, 3)));
    }
}

// ParametricGeometry — sample a u/v function (R²→R³) on a slice×stack grid
// and emit a triangle mesh. Mirrors three.js's signature
// `new ParametricGeometry(func, slices, stacks)`.
/// three.js r165 `ParametricGeometry` (examples/jsm/geometries): the surface
/// `func(u, v, target)` sampled on a `slices` × `stacks` grid, normals from
/// finite differences. The callback is user code, so this runs in JS.
export class ParametricGeometry extends BufferGeometry {
    constructor(func = (u, v, target) => target.set(u, v, Math.cos(u) * Math.sin(v)), slices = 8, stacks = 8) {
        super();
        this.type = 'ParametricGeometry';
        this.parameters = { func, slices, stacks };
        const indices = [], vertices = [], normals = [], uvs = [];
        const EPS = 0.00001;
        const normal = new Vector3(), p0 = new Vector3(), p1 = new Vector3(), pu = new Vector3(), pv = new Vector3();
        const sliceCount = slices + 1;
        for (let i = 0; i <= stacks; i++) {
            const v = i / stacks;
            for (let j = 0; j <= slices; j++) {
                const u = j / slices;
                func(u, v, p0);
                vertices.push(p0.x, p0.y, p0.z);
                if (u - EPS >= 0) { func(u - EPS, v, p1); pu.subVectors(p0, p1); } else { func(u + EPS, v, p1); pu.subVectors(p1, p0); }
                if (v - EPS >= 0) { func(u, v - EPS, p1); pv.subVectors(p0, p1); } else { func(u, v + EPS, p1); pv.subVectors(p1, p0); }
                normal.crossVectors(pu, pv).normalize();
                normals.push(normal.x, normal.y, normal.z);
                uvs.push(u, v);
            }
        }
        for (let i = 0; i < stacks; i++) {
            for (let j = 0; j < slices; j++) {
                const a = i * sliceCount + j;
                const b = i * sliceCount + j + 1;
                const c = (i + 1) * sliceCount + j + 1;
                const d = (i + 1) * sliceCount + j;
                indices.push(a, b, d, b, c, d);
            }
        }
        this.setIndex(new Uint32Array(indices));
        this.setAttribute('position', new BufferAttribute(new Float32Array(vertices), 3));
        this.setAttribute('normal', new BufferAttribute(new Float32Array(normals), 3));
        this.setAttribute('uv', new BufferAttribute(new Float32Array(uvs), 2));
    }
}

// DecalGeometry — three.js r165 addon: the mesh's triangles inside the box
// projector, clipped to it, in world space with box-face UVs. Clipping and
// projection run in Rust (`WebGeometry.decal`).
export class DecalGeometry extends BufferGeometry {
    constructor(mesh, position = new Vector3(), orientation = new Euler(), size = new Vector3(1, 1, 1)) {
        super();
        const src = _wasmSourceOf(mesh?.geometry);
        if (!src) return;
        const mw = mesh.matrixWorld?.elements ?? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        try {
            _fillFromWasm(this, WebGeometry.decal(src.wasm, Float64Array.from(mw),
                new Float64Array([position.x, position.y, position.z]),
                new Float64Array([orientation.x, orientation.y, orientation.z]),
                orientation.order ?? 'XYZ',
                new Float64Array([size.x, size.y, size.z])));
        } finally {
            if (src.owned) src.wasm.free?.();
        }
    }
}

// Font (examples/jsm/loaders/FontLoader.js) — a typeface.js font, parsed and
// laid out in Rust (`loaders::typeface`); `generateShapes` returns Shapes.
export class Font {
    constructor(data) {
        this.isFont = true;
        this.type = 'Font';
        this.data = data;
        this._w = null;
    }
    _wasm() {
        return this._w ??= new WebFont(typeof this.data === 'string' ? this.data : JSON.stringify(this.data));
    }
    generateShapes(text, size = 100) {
        const list = this._wasm().generateShapes(String(text), size);
        const shapes = _shapesFromWasmList(list);
        list.free();
        return shapes;
    }
}

// TextGeometry (examples/jsm/geometries/TextGeometry.js): the font's shapes,
// extruded — three.js's own defaults on top of ExtrudeGeometry's.
export class TextGeometry extends ExtrudeGeometry {
    constructor(text, parameters = {}) {
        const font = parameters.font;
        if (font === undefined) {
            super();
        } else {
            const shapes = font.generateShapes(text, parameters.size);
            if (parameters.depth === undefined && parameters.height !== undefined) {
                console.warn('THREE.TextGeometry: .height is now depreciated. Please use .depth instead');
            }
            parameters.depth = parameters.depth ?? parameters.height ?? 50;
            if (parameters.bevelThickness === undefined) parameters.bevelThickness = 10;
            if (parameters.bevelSize === undefined) parameters.bevelSize = 8;
            if (parameters.bevelEnabled === undefined) parameters.bevelEnabled = false;
            super(shapes, parameters);
        }
        this.type = 'TextGeometry';
    }
}

// EdgesGeometry / WireframeGeometry — three.js r165, generated in Rust
// (src/geometries/edges_geometry.rs) from any source geometry. threers
// extension: a source carrying polygons (`setPolygons`, `inferQuads`,
// `subdivide`) gets its polygon outlines from WireframeGeometry, and
// EdgesGeometry never draws an edge inside one polygon.
function _edgesFromWasm(target, geometry, build) {
    const src = _wasmSourceOf(geometry);
    if (!src) return target;
    try {
        _fillFromWasm(target, build(src.wasm));
    } finally {
        if (src.owned) src.wasm.free?.();
    }
    return target;
}
export class EdgesGeometry extends BufferGeometry {
    constructor(geometry = null, thresholdAngle = 1) {
        super();
        this.type = 'EdgesGeometry';
        this.parameters = { geometry, thresholdAngle };
        if (geometry !== null) _edgesFromWasm(this, geometry, (w) => WebGeometry.edges(w, thresholdAngle));
    }
}
export class WireframeGeometry extends BufferGeometry {
    constructor(geometry = null) {
        super();
        this.type = 'WireframeGeometry';
        this.parameters = { geometry };
        if (geometry !== null) _edgesFromWasm(this, geometry, (w) => WebGeometry.wireframe(w));
    }
}
// ---- Debug helpers (three.js r165) ----
// Each helper's lines — and the filled cone / quad three.js draws for some —
// come from the Rust `helper*` bindings (`src/helpers/three_lines.rs`), already
// in world space: the helper's node is placed so that its world matrix is the
// identity (under its parent's inverse when it has a parent other than the
// Scene), and its geometry is recomputed from the light / camera / object it
// shows, on `update()` and, for the helpers three.js refreshes every frame, at
// each sync.
const _HELPER_IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
/// An object's world matrix as 16 numbers (identity for none).
function _helperWorld(obj) {
    if (!obj?.matrixWorld) return _HELPER_IDENTITY;
    obj.updateWorldMatrix(true, false);
    return Array.from(obj.matrixWorld.elements);
}
function _helperWorldPosition(obj) {
    const e = _helperWorld(obj);
    return [e[12], e[13], e[14]];
}
function _helperSameArray(a, b) {
    if (!a || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}
/// Base: a LineSegments drawing world-space lines from Rust. `_helperCompute()`
/// returns a `WebHelperGeometry` (or null); `fill` options make a translucent
/// / back-faced Mesh part for the triangles, an internal child of the helper.
class _RustLineHelper extends LineSegments {
    constructor(color, fill = null) {
        // three.js's helpers are all `toneMapped: false`.
        super(new BufferGeometry(), new LineBasicMaterial({ color: color ?? 0xffffff, toneMapped: false }));
        this.matrixAutoUpdate = false;
        this._worldSpaceLines = true;
        this._helperParts = [];
        this._fillOptions = fill;
    }
    _helperApply(hg) {
        const lines = hg ? new Float32Array(hg.lines) : new Float32Array(0);
        const colors = hg ? new Float32Array(hg.colors) : new Float32Array(0);
        const mesh = hg ? new Float32Array(hg.mesh) : new Float32Array(0);
        hg?.free?.();
        const g = this.geometry;
        const sameLines = _helperSameArray(g.attributes.position?.array, lines)
            && (colors.length ? _helperSameArray(g.attributes.color?.array, colors) : !g.attributes.color);
        if (!sameLines) {
            g.setAttribute('position', new BufferAttribute(lines, 3));
            if (colors.length) g.setAttribute('color', new BufferAttribute(colors, 3));
            else g.deleteAttribute('color');
        }
        if (this._fillOptions && mesh.length) {
            const fill = this._helperParts[0];
            if (!fill) {
                const fg = new BufferGeometry();
                fg.setAttribute('position', new BufferAttribute(mesh, 3));
                const part = new Mesh(fg, new MeshBasicMaterial(this._fillOptions));
                part.parent = this;
                this._helperParts.push(part);
                this._dirty(_D_CHILDREN);
            } else if (!_helperSameArray(fill.geometry.attributes.position.array, mesh)) {
                fill.geometry.setAttribute('position', new BufferAttribute(mesh, 3));
            }
        }
    }
    _helperCompute() { return null; }
    update() { this._helperApply(this._helperCompute()); return this; }
    /// The helpers three.js recomputes in `updateMatrixWorld`, or whose matrix
    /// *is* their subject's, are recomputed at each sync.
    _helperSync() { if (this._autoUpdate) this.update(); }
    dispose() {}
    // ---- mirror: world-space lines ----
    _mirrorExtraChildren() { return this._helperParts; }
    /// World identity: the inverse of the parent's world matrix (identity
    /// under the Scene, or with no parent).
    _helperLocal() {
        const p = this.parent;
        if (!p || p.isScene || !p.matrixWorld) return _HELPER_IDENTITY;
        p.updateWorldMatrix(true, false);
        return Array.from(_o3dM.copy(p.matrixWorld).invert().elements);
    }
    _mirrorTransform(w, h, m) {
        const local = this._helperLocal();
        w.setLocalMatrix(h, Float32Array.from(local));
        m.matrix = local;
    }
    _mirrorPolled() { return true; }
    _mirrorBeforeSync() { this._helperSync(); }
    _mirrorPoll(_w, _h, m) {
        if (!_sameNumbers(m.matrix, this._helperLocal())) this._dirty(_D_TRS);
    }
}
function _helperLightColor(light) {
    return light?.color ?? 0xffffff;
}

export class DirectionalLightHelper extends _RustLineHelper {
    constructor(light, size = 1, color) {
        super(color ?? _helperLightColor(light));
        this.light = light;
        this.size = size;
        this.color = color;
        this.type = 'DirectionalLightHelper';
        this._autoUpdate = true;
        this.update();
    }
    _helperCompute() {
        if (!this.light) return null;
        const [tx, ty, tz] = _helperWorldPosition(this.light.target);
        return helperDirectionalLight(new Float64Array(_helperWorld(this.light)), tx, ty, tz, this.size);
    }
}
export class HemisphereLightHelper extends _RustLineHelper {
    constructor(light, size = 1, color) {
        super(0xffffff);
        this.light = light;
        this.size = size;
        this.color = color;
        this.type = 'HemisphereLightHelper';
        this._autoUpdate = true;
        this.update();
    }
    _helperCompute() {
        if (!this.light) return null;
        const sky = _color(this.color ?? this.light.color ?? 0xffffff);
        const ground = _color(this.color ?? this.light.groundColor ?? 0xffffff);
        return helperHemisphereLight(new Float64Array(_helperWorld(this.light)), this.size, sky, ground);
    }
}
export class PointLightHelper extends _RustLineHelper {
    constructor(light, sphereSize = 1, color) {
        super(color ?? _helperLightColor(light));
        this.light = light;
        this.sphereSize = sphereSize;
        this.color = color;
        this.type = 'PointLightHelper';
        this._autoUpdate = true;
        this.update();
    }
    _helperCompute() {
        if (!this.light) return null;
        return helperPointLight(new Float64Array(_helperWorld(this.light)), this.sphereSize);
    }
}
export class SpotLightHelper extends _RustLineHelper {
    constructor(light, color) {
        super(color ?? _helperLightColor(light));
        this.light = light;
        this.color = color;
        this.type = 'SpotLightHelper';
        this._autoUpdate = true;
        this.update();
    }
    _helperCompute() {
        if (!this.light) return null;
        const [tx, ty, tz] = _helperWorldPosition(this.light.target);
        return helperSpotLight(new Float64Array(_helperWorld(this.light)), tx, ty, tz,
            this.light.distance ?? 0, this.light.angle ?? Math.PI / 3);
    }
}
/// A camera's projection matrix, rebuilt from its parameters the way its
/// `updateProjectionMatrix` does (the shim's cameras keep none in JS).
function _helperProjection(camera) {
    const view = camera.view?.enabled
        ? new Float64Array([camera.view.fullWidth, camera.view.fullHeight, camera.view.offsetX,
            camera.view.offsetY, camera.view.width, camera.view.height])
        : undefined;
    if (camera.isOrthographicCamera || (camera.left !== undefined && camera.fov === undefined)) {
        return helperOrthographicProjection(camera.left, camera.right, camera.top, camera.bottom,
            camera.near, camera.far, camera.zoom ?? 1, view);
    }
    const filmWidth = (camera.filmGauge ?? 35) * Math.min(camera.aspect ?? 1, 1);
    return helperPerspectiveProjection(camera.fov ?? 50, camera.aspect ?? 1, camera.near ?? 0.1,
        camera.far ?? 2000, camera.zoom ?? 1, camera.filmOffset ?? 0, filmWidth, view);
}
export class CameraHelper extends _RustLineHelper {
    constructor(camera) {
        super(0xffffff);
        this.camera = camera;
        this.type = 'CameraHelper';
        this._autoUpdate = true;
        this._colors = [new Color(0xffaa00), new Color(0xff0000), new Color(0x00aaff),
            new Color(0xffffff), new Color(0x333333)];
        this.update();
    }
    setColors(frustum, cone, up, target, cross) {
        this._colors = [frustum, cone, up, target, cross];
        return this.update();
    }
    _helperCompute() {
        if (!this.camera) return null;
        const [f, c, u, t, x] = this._colors.map((col) => _color(col));
        return helperCamera(_helperProjection(this.camera), new Float64Array(_helperWorld(this.camera)), f, c, u, t, x);
    }
}
export class ArrowHelper extends _RustLineHelper {
    constructor(dir = new Vector3(0, 0, 1), origin = new Vector3(0, 0, 0), length = 1, color = 0xffff00,
                headLength = length * 0.2, headWidth = headLength * 0.2) {
        super(color, { color, toneMapped: false });
        this.type = 'ArrowHelper';
        this.matrixAutoUpdate = true;
        this.position.copy(origin);
        this._autoUpdate = true;
        this.setDirection(dir);
        this.setLength(length, headLength, headWidth);
    }
    setDirection(dir) {
        const [x, y, z, w] = helperArrowQuaternion(dir.x, dir.y, dir.z);
        this.quaternion.set(x, y, z, w);
        return this.update();
    }
    setLength(length, headLength = length * 0.2, headWidth = headLength * 0.2) {
        this._length = length; this._headLength = headLength; this._headWidth = headWidth;
        return this.update();
    }
    setColor(color) {
        this.material = new LineBasicMaterial({ color, toneMapped: false });
        this._fillOptions = { color, toneMapped: false };
        if (this._helperParts[0]) this._helperParts[0].material = new MeshBasicMaterial(this._fillOptions);
        return this;
    }
    _helperCompute() {
        if (this._length === undefined) return null;
        return helperArrow(new Float64Array(_helperWorld(this)), this._length, this._headLength, this._headWidth);
    }
}
export class PlaneHelper extends _RustLineHelper {
    constructor(plane, size = 1, hex = 0xffff00) {
        super(hex, { color: hex, opacity: 0.2, transparent: true, depthWrite: false });
        this.type = 'PlaneHelper';
        this.plane = plane;
        this.size = size;
        this._autoUpdate = true;
        this.update();
    }
    _helperCompute() {
        if (!this.plane) return null;
        const n = this.plane.normal;
        return helperPlane(n.x, n.y, n.z, this.plane.constant, this.size);
    }
}
/// three.js `getBoneList`: `object` and its descendants that are bones.
function _helperBoneList(object) {
    const out = [];
    if (object?.isBone === true) out.push(object);
    for (const c of object?.children ?? []) out.push(..._helperBoneList(c));
    return out;
}
export class SkeletonHelper extends _RustLineHelper {
    constructor(object) {
        super(0xffffff);
        this.isSkeletonHelper = true;
        this.type = 'SkeletonHelper';
        this.root = object;
        this.bones = _helperBoneList(object);
        this._autoUpdate = true;
        this.update();
    }
    _helperCompute() {
        if (!this.root) return null;
        const worlds = new Float64Array(this.bones.length * 16);
        this.bones.forEach((b, i) => worlds.set(_helperWorld(b), i * 16));
        const parents = new Int32Array(this.bones.map((b) => (b.parent?.isBone ? this.bones.indexOf(b.parent) : -1)));
        return helperSkeleton(new Float64Array(_helperWorld(this.root)), worlds, parents);
    }
}
export class BoxHelper extends _RustLineHelper {
    constructor(object, color = 0xffff00) {
        super(color);
        this.object = object;
        this.type = 'BoxHelper';
        this.update();
    }
    setFromObject(object) { this.object = object; return this.update(); }
    _helperCompute() {
        if (!this.object) return null;
        const worlds = [], positions = [], counts = [];
        const visit = (node) => {
            const g = node.geometry;
            if (g && !node._worldSpaceLines) {
                const pos = _attrPacked(g.attributes?.position)
                    ?? (g._w?.toBufferGeometry ? g._w.toBufferGeometry().getAttributeArray('position') : null);
                if (pos?.length) {
                    worlds.push(..._helperWorld(node));
                    for (let i = 0; i < pos.length; i++) positions.push(pos[i]);
                    counts.push(pos.length);
                }
            }
            for (const c of node.children ?? []) visit(c);
        };
        visit(this.object);
        return helperBoxOfObjects(new Float64Array(worlds), new Float32Array(positions), new Uint32Array(counts));
    }
}
export class Box3Helper extends _RustLineHelper {
    constructor(box, color = 0xffff00) {
        super(color);
        this.box = box;
        this.type = 'Box3Helper';
        this._autoUpdate = true;
        this.update();
    }
    _helperCompute() {
        const b = this.box;
        if (!b || b.isEmpty?.()) return null;
        // The shim's Box3 keeps its corners in wasm; read them back through it.
        let min = b.min, max = b.max;
        if (!min || !max) {
            const c = b._w.center(), s = b._w.size();
            min = { x: c.x - s.x / 2, y: c.y - s.y / 2, z: c.z - s.z / 2 };
            max = { x: c.x + s.x / 2, y: c.y + s.y / 2, z: c.z + s.z / 2 };
        }
        return helperBox3(new Float64Array([min.x, min.y, min.z]), new Float64Array([max.x, max.y, max.z]));
    }
}
export class VertexNormalsHelper extends _RustLineHelper {
    constructor(object, size = 1, color = 0xff0000) {
        super(color);
        this.object = object;
        this.size = size;
        this.type = 'VertexNormalsHelper';
        this.update();
    }
    _helperCompute() {
        const g = this.object?.geometry;
        const src = g?.attributes?.position ? g : (g ? geometryToBufferGeometry(g) : null);
        const pos = _attrPacked(src?.attributes?.position), nrm = _attrPacked(src?.attributes?.normal);
        if (!pos || !nrm) return null;
        return helperVertexNormals(new Float64Array(_helperWorld(this.object)), new Float32Array(pos), new Float32Array(nrm), this.size);
    }
}
export class RectAreaLightHelper extends _RustLineHelper {
    constructor(light, color) {
        super(0xffffff, { side: BackSide });
        this.light = light;
        this.color = color;
        this.type = 'RectAreaLightHelper';
        this._autoUpdate = true;
        this._lastColor = null;
        this.update();
    }
    update() {
        if (this.light) {
            const c = this.color !== undefined
                ? this.color
                : (() => { const [r, g, b] = helperRectAreaLightColor(_color(this.light.color ?? 0xffffff), this.light.intensity ?? 1); return { r, g, b }; })();
            const key = typeof c === 'object' ? `${c.r},${c.g},${c.b}` : String(c);
            if (key !== this._lastColor) {
                this._lastColor = key;
                this.material = new LineBasicMaterial({ color: c });
                this._fillOptions = { side: BackSide, color: c };
                if (this._helperParts[0]) this._helperParts[0].material = new MeshBasicMaterial(this._fillOptions);
            }
        }
        return super.update();
    }
    _helperCompute() {
        if (!this.light) return null;
        return helperRectAreaLight(new Float64Array(_helperWorld(this.light)), this.light.width ?? 10, this.light.height ?? 10);
    }
}

// ---- Pass 6: loaders + EventDispatcher mixin ----
//
// EventDispatcher: three.js's tiny pub/sub. Used by loaders for progress, by
// the renderer for context-lost events, by Object3D subclasses for selection
// events, etc. Apply via Object.assign(MyClass.prototype, EventDispatcherMixin).

export const EventDispatcherMixin = {
    addEventListener(type, listener) {
        if (!this._listeners) this._listeners = {};
        (this._listeners[type] = this._listeners[type] || []).push(listener);
    },
    hasEventListener(type, listener) {
        return !!(this._listeners && this._listeners[type] && this._listeners[type].includes(listener));
    },
    removeEventListener(type, listener) {
        if (!this._listeners || !this._listeners[type]) return;
        this._listeners[type] = this._listeners[type].filter(l => l !== listener);
    },
    dispatchEvent(event) {
        if (!this._listeners || !this._listeners[event.type]) return;
        const ls = this._listeners[event.type].slice();
        event.target = this;
        for (const l of ls) l.call(this, event);
    },
};
export class EventDispatcher {}
Object.assign(EventDispatcher.prototype, EventDispatcherMixin);

// LoadingManager: tracks in-flight requests + global progress callbacks.
export class LoadingManager {
    constructor(onLoad, onProgress, onError) {
        this.onLoad = onLoad; this.onProgress = onProgress; this.onError = onError;
        this._loading = 0; this._loaded = 0; this._total = 0;
    }
    itemStart(url) { this._loading++; this._total++; this.onProgress?.(url, this._loaded, this._total); }
    itemEnd(url)   { this._loading--; this._loaded++; this.onProgress?.(url, this._loaded, this._total); if (this._loading === 0) this.onLoad?.(); }
    itemError(url) { this.onError?.(url); }
}
const DefaultLoadingManager = new LoadingManager();
export { DefaultLoadingManager };

// Tiny shared Loader base — provides .load(), .loadAsync(), .setPath(), etc.
class _Loader {
    constructor(manager = DefaultLoadingManager) {
        this.manager = manager;
        this.path = ''; this.resourcePath = ''; this.crossOrigin = 'anonymous';
        this.withCredentials = false;
        this.requestHeader = {};
    }
    setWithCredentials(v) { this.withCredentials = v; return this; }
    setPath(p) { this.path = p; return this; }
    setResourcePath(p) { this.resourcePath = p; return this; }
    setCrossOrigin(co) { this.crossOrigin = co; return this; }
    setRequestHeader(h) { this.requestHeader = h; return this; }
    loadAsync(url, onProgress) {
        return new Promise((resolve, reject) => this.load(url, resolve, onProgress, reject));
    }
}

export class FileLoader extends _Loader {
    constructor(manager) { super(manager); this.responseType = ''; this.mimeType = ''; }
    setResponseType(t) { this.responseType = t; return this; }
    setMimeType(t) { this.mimeType = t; return this; }
    async load(url, onLoad, _onProgress, onError) {
        try {
            this.manager.itemStart(url);
            const res = await fetch(this.path + url, {
                headers: this.requestHeader,
                credentials: this.withCredentials ? 'include' : 'same-origin',
            });
            if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
            let data;
            if (this.responseType === 'arraybuffer') data = await res.arrayBuffer();
            else if (this.responseType === 'blob')   data = await res.blob();
            else if (this.responseType === 'json')   data = await res.json();
            else data = await res.text();
            onLoad?.(data);
            this.manager.itemEnd(url);
        } catch (e) {
            onError?.(e); this.manager.itemError(url); this.manager.itemEnd(url);
        }
    }
}

export class ImageLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        this.manager.itemStart(url);
        const img = new Image();
        img.crossOrigin = this.crossOrigin;
        img.onload = () => { onLoad?.(img); this.manager.itemEnd(url); };
        img.onerror = (e) => { onError?.(e); this.manager.itemError(url); this.manager.itemEnd(url); };
        img.src = this.path + url;
        return img;
    }
}

export class TextureLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        const tex = new Texture();
        // We delegate image fetching to ImageLoader and copy the decoded image
        // bytes into a DataTexture once it lands. For now this returns the
        // Texture eagerly so user code that does `material.map = loader.load(...)`
        // works — the texture's bytes get populated asynchronously.
        new ImageLoader(this.manager).load(url, (img) => {
            try {
                const c = document.createElement('canvas');
                c.width = img.width; c.height = img.height;
                const ctx = c.getContext('2d');
                ctx.drawImage(img, 0, 0);
                const data = ctx.getImageData(0, 0, img.width, img.height).data;
                tex._w = new WebDataTexture(img.width, img.height, new Uint8Array(data));
                tex.image = img;
                tex.needsUpdate = true;
                onLoad?.(tex);
            } catch (e) { onError?.(e); }
        }, undefined, onError);
        return tex;
    }
}

// Minimal JSON loaders so user code calling `.load(...)` returns sensibly.
export class FontLoader extends _Loader {
    load(url, onLoad, onProgress, onError) {
        const loader = new FileLoader(this.manager);
        loader.setPath(this.path);
        loader.setRequestHeader(this.requestHeader);
        loader.load(url, (text) => {
            let font;
            try { font = this.parse(typeof text === 'string' ? JSON.parse(text) : text); } catch (e) { onError?.(e); return; }
            onLoad?.(font);
        }, onProgress, onError);
    }
    parse(json) { return new Font(json); }
}
export class MaterialLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) { new FileLoader(this.manager).load(url, onLoad, undefined, onError); }
}
// Real BufferGeometryLoader. Parses the three.js BufferGeometry JSON shape:
//   { data: { attributes: { position: {itemSize, type, array}, ... },
//             index: { type, array } } }
// Returns a BufferGeometry instance suitable for `new Mesh(geom, mat)`.
export class BufferGeometryLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (text) => {
            try {
                const json = (typeof text === 'string') ? JSON.parse(text) : text;
                onLoad?.(this.parse(json));
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(json) {
        const g = new BufferGeometry();
        const data = json.data || json;
        const attrs = data.attributes || {};
        for (const name of Object.keys(attrs)) {
            const a = attrs[name];
            const Ctor = _typedArrayByName(a.type) || Float32Array;
            const arr = new Ctor(a.array);
            g.setAttribute(name, new BufferAttribute(arr, a.itemSize));
        }
        if (data.index) {
            const Ctor = _typedArrayByName(data.index.type) || Uint32Array;
            const idxArr = new Ctor(data.index.array);
            g.setIndex(new Uint32Array(idxArr));
        }
        if (json.uuid) g.uuid = json.uuid;
        if (json.name) g.name = json.name;
        return g;
    }
}
function _typedArrayByName(name) {
    switch (name) {
        case 'Int8Array': return Int8Array;
        case 'Uint8Array': return Uint8Array;
        case 'Uint8ClampedArray': return Uint8ClampedArray;
        case 'Int16Array': return Int16Array;
        case 'Uint16Array': return Uint16Array;
        case 'Int32Array': return Int32Array;
        case 'Uint32Array': return Uint32Array;
        case 'Float32Array': return Float32Array;
        case 'Float64Array': return Float64Array;
        default: return null;
    }
}
// Real ObjectLoader. Parses three.js's Scene JSON: geometries[], materials[],
// images[]/textures[], object{ children[] }. We support a useful subset:
// BufferGeometry data, common materials (basic/standard/lambert/phong/normal),
// and Mesh/Group/Object3D nodes with position/rotation/scale/quaternion.
export class ObjectLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (text) => {
            try {
                const json = (typeof text === 'string') ? JSON.parse(text) : text;
                onLoad?.(this.parse(json));
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(json) {
        const geomById = {};
        const bgLoader = new BufferGeometryLoader(this.manager);
        for (const gd of (json.geometries || [])) {
            // three.js wraps the actual attribute shape in `data` for BufferGeometry.
            geomById[gd.uuid] = bgLoader.parse(gd);
        }
        const matById = {};
        for (const md of (json.materials || [])) {
            matById[md.uuid] = _parseMaterialJson(md);
        }
        const root = _parseObjectJson(json.object || json, geomById, matById);
        return root;
    }
}
function _parseMaterialJson(md) {
    const opts = {};
    if (md.color !== undefined) opts.color = md.color;
    if (md.emissive !== undefined) opts.emissive = md.emissive;
    if (md.roughness !== undefined) opts.roughness = md.roughness;
    if (md.metalness !== undefined) opts.metalness = md.metalness;
    if (md.opacity !== undefined) opts.opacity = md.opacity;
    if (md.transparent !== undefined) opts.transparent = md.transparent;
    if (md.side !== undefined) opts.side = md.side;
    if (md.wireframe !== undefined) opts.wireframe = md.wireframe;
    if (md.vertexColors !== undefined) opts.vertexColors = md.vertexColors;
    switch (md.type) {
        case 'MeshBasicMaterial': return new MeshBasicMaterial(opts);
        case 'MeshLambertMaterial': return new MeshLambertMaterial(opts);
        case 'MeshPhongMaterial': return new MeshPhongMaterial(opts);
        case 'MeshStandardMaterial': return new MeshStandardMaterial(opts);
        case 'MeshPhysicalMaterial': return new MeshPhysicalMaterial(opts);
        case 'MeshNormalMaterial': return new MeshNormalMaterial();
        case 'MeshDepthMaterial': return new MeshDepthMaterial();
        case 'MeshToonMaterial': return new MeshToonMaterial(opts);
        case 'LineBasicMaterial': return new LineBasicMaterial(opts);
        case 'PointsMaterial': return new PointsMaterial(opts);
        default: return new MeshStandardMaterial(opts);
    }
}
function _parseObjectJson(od, geomById, matById) {
    let obj;
    const matRef = (typeof od.material === 'string') ? matById[od.material]
                 : Array.isArray(od.material) ? od.material.map(u => matById[u])
                 : null;
    const geomRef = od.geometry ? geomById[od.geometry] : null;
    switch (od.type) {
        case 'Mesh':
            obj = new Mesh(geomRef, matRef || new MeshStandardMaterial());
            break;
        case 'Group':
            obj = new Group();
            break;
        case 'Scene':
            obj = new Scene();
            break;
        case 'PerspectiveCamera':
            obj = new PerspectiveCamera(od.fov, od.aspect, od.near, od.far);
            break;
        case 'OrthographicCamera':
            obj = new OrthographicCamera(od.left, od.right, od.top, od.bottom, od.near, od.far);
            break;
        case 'AmbientLight':
            obj = new AmbientLight(od.color, od.intensity);
            break;
        case 'DirectionalLight':
            obj = new DirectionalLight(od.color, od.intensity);
            break;
        case 'PointLight':
            obj = new PointLight(od.color, od.intensity, od.distance, od.decay);
            break;
        case 'Object3D':
        default:
            obj = new Object3D();
            break;
    }
    if (od.name) obj.name = od.name;
    // three.js serializes object transforms as a flat 4×4 matrix (column-major),
    // decomposed as three.js's ObjectLoader does.
    if (Array.isArray(od.matrix) && od.matrix.length === 16 && obj.matrix) {
        obj.matrix.fromArray(od.matrix);
        if (od.matrixAutoUpdate !== undefined) obj.matrixAutoUpdate = od.matrixAutoUpdate;
        if (obj.matrixAutoUpdate) obj.matrix.decompose(obj.position, obj.quaternion, obj.scale);
    } else {
        if (od.position) obj.position.fromArray(od.position);
        if (od.rotation) obj.rotation.fromArray(od.rotation);
        if (od.quaternion) obj.quaternion.fromArray(od.quaternion);
        if (od.scale) obj.scale.fromArray(od.scale);
    }
    if (od.visible !== undefined) obj.visible = od.visible;
    if (od.layers !== undefined) obj.layers.mask = od.layers;
    for (const cd of (od.children || [])) {
        const c = _parseObjectJson(cd, geomById, matById);
        if (obj instanceof Scene) obj.add(c);
        else if (obj instanceof Group) obj.add(c);
        else if (obj instanceof Object3D) obj.add(c);
    }
    return obj;
}
export class AnimationLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) { new FileLoader(this.manager).load(url, onLoad, undefined, onError); }
}
export class AudioLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        const fl = new FileLoader(this.manager); fl.setResponseType('arraybuffer');
        fl.load(url, onLoad, undefined, onError);
    }
}
export class CubeTextureLoader extends _Loader {
    load(urls, onLoad, _onProgress, onError) {
        if (!Array.isArray(urls)) urls = [urls, urls, urls, urls, urls, urls];
        const tex = new CubeTexture();
        let done = 0;
        const imgs = new Array(6);
        urls.forEach((u, i) => {
            new ImageLoader(this.manager).load(u, (img) => {
                imgs[i] = img; done++;
                // three.js's `CubeTextureLoader` marks the result sRGB.
                if (done === 6) { tex.images = imgs; tex.colorSpace = 'srgb'; tex.needsUpdate = true; onLoad?.(tex); }
            }, undefined, onError);
        });
        return tex;
    }
}
// ---- GLTFLoader ----
//
// three.js r165 `GLTFLoader`. The parser is Rust's (`WebGltfParser`: three.js's
// GLTFParser ported to `src/loaders/gltf`), and it builds every object as a
// node of the scene graph the shim mirrors into — the hidden graph, which a
// node leaves (moved, not copied) when its object is added to a Scene. It
// decodes Draco, meshopt, quantized and sparse accessors, PNG and KTX2 too.
// This side fetches what the file names by URI, decodes other images with the
// browser's decoder, and wraps the nodes the parser made in three.js objects:
// `Group`/`Mesh`/`SkinnedMesh`/`Bone`/cameras/lights/`InstancedMesh`, the
// materials and textures they use, and the `AnimationClip`s.

/// three.js `InterpolateLinear` / `InterpolateDiscrete` (CUBICSPLINE tracks
/// keep the default and get three.js's own interpolant).
const _GLTF_INTERPOLATION = { LINEAR: 2301, STEP: 2300, CUBICSPLINE: 2301 };

/// The extensions three.js's parser lists as `parser.extensions` (plugins
/// registered by default, plus those enabled by `extensionsUsed`).
const _GLTF_PLUGIN_NAMES = [
    'KHR_materials_clearcoat', 'KHR_materials_dispersion', 'KHR_texture_basisu', 'EXT_texture_webp',
    'EXT_texture_avif', 'KHR_materials_sheen', 'KHR_materials_transmission', 'KHR_materials_volume',
    'KHR_materials_ior', 'KHR_materials_emissive_strength', 'KHR_materials_specular',
    'KHR_materials_iridescence', 'KHR_materials_anisotropy', 'EXT_materials_bump', 'KHR_lights_punctual',
    'EXT_meshopt_compression', 'EXT_mesh_gpu_instancing',
];
const _GLTF_USED_EXTENSIONS = ['KHR_materials_unlit', 'KHR_draco_mesh_compression', 'KHR_texture_transform', 'KHR_mesh_quantization'];

/// three.js r165's defaults for what a material class has and a glTF material
/// may leave unset (the shim's material classes define fewer properties).
const _GLTF_BASE_DEFAULTS = {
    side: 0, opacity: 1, transparent: false, alphaTest: 0, depthTest: true, depthWrite: true, colorWrite: true,
    visible: true, blending: 1, toneMapped: true, vertexColors: false, alphaHash: false, alphaToCoverage: false, premultipliedAlpha: false,
    dithering: false, forceSinglePass: false, shadowSide: null, clippingPlanes: null,
    clipIntersection: false, clipShadows: false, precision: null, fog: true,
};
const _GLTF_MATERIAL_DEFAULTS = {
    MeshStandardMaterial: {
        roughness: 1, metalness: 0, map: null, lightMap: null, lightMapIntensity: 1, aoMap: null, aoMapIntensity: 1,
        emissive: () => new Color(0, 0, 0), emissiveIntensity: 1, emissiveMap: null, bumpMap: null, bumpScale: 1,
        normalMap: null, normalMapType: 0, normalScale: () => new Vector2(1, 1), displacementMap: null,
        displacementScale: 1, displacementBias: 0, roughnessMap: null, metalnessMap: null, alphaMap: null,
        envMap: null, envMapRotation: () => new Euler(0, 0, 0), envMapIntensity: 1, wireframeLinewidth: 1,
        wireframeLinecap: 'round', wireframeLinejoin: 'round', flatShading: false,
    },
    MeshPhysicalMaterial: {
        clearcoat: 0, clearcoatMap: null, clearcoatRoughness: 0, clearcoatRoughnessMap: null,
        clearcoatNormalScale: () => new Vector2(1, 1), clearcoatNormalMap: null, ior: 1.5, reflectivity: 0.5,
        dispersion: 0, iridescence: 0, iridescenceMap: null, iridescenceIOR: 1.3,
        iridescenceThicknessRange: () => [100, 400], iridescenceThicknessMap: null, sheen: 0,
        sheenColor: () => new Color(0, 0, 0), sheenColorMap: null, sheenRoughness: 1, sheenRoughnessMap: null,
        transmission: 0, transmissionMap: null, thickness: 0, thicknessMap: null, attenuationDistance: Infinity,
        attenuationColor: () => new Color(1, 1, 1), specularIntensity: 1, specularIntensityMap: null,
        specularColor: () => new Color(1, 1, 1), specularColorMap: null, anisotropy: 0, anisotropyRotation: 0,
        anisotropyMap: null,
    },
    MeshBasicMaterial: {
        map: null, lightMap: null, lightMapIntensity: 1, aoMap: null, aoMapIntensity: 1, specularMap: null,
        alphaMap: null, envMap: null, envMapRotation: () => new Euler(0, 0, 0), combine: 0, reflectivity: 1,
        refractionRatio: 0.98, wireframeLinewidth: 1, wireframeLinecap: 'round', wireframeLinejoin: 'round',
    },
    PointsMaterial: { map: null, alphaMap: null, size: 1, sizeAttenuation: true },
    LineBasicMaterial: { map: null, linewidth: 1, linecap: 'round', linejoin: 'round' },
};

/// three.js r165's `GLTFCubicSplineInterpolant` (and its quaternion twin,
/// which normalizes): values are `[in, value, out]` per key. Made on first
/// use — `Interpolant` is defined further down this file.
let _gltfInterpolants = null;
function _gltfCubicInterpolant(quaternion) {
    _gltfInterpolants ??= (() => {
        class GLTFCubicSplineInterpolant extends Interpolant {
            constructor(parameterPositions, sampleValues, sampleSize, resultBuffer) {
                super(parameterPositions, sampleValues, sampleSize, resultBuffer);
                this.parameterPositions = parameterPositions;
                this.sampleValues = sampleValues;
                this.valueSize = sampleSize;
                this.resultBuffer = resultBuffer ?? new sampleValues.constructor(sampleSize);
            }
            copySampleValue_(index) {
                const result = this.resultBuffer, values = this.sampleValues, valueSize = this.valueSize;
                const offset = index * valueSize * 3 + valueSize;
                for (let i = 0; i !== valueSize; i++) result[i] = values[offset + i];
                return result;
            }
            interpolate_(i1, t0, t, t1) {
                const result = this.resultBuffer, values = this.sampleValues, stride = this.valueSize;
                const stride2 = stride * 2, stride3 = stride * 3, td = t1 - t0;
                const p = (t - t0) / td, pp = p * p, ppp = pp * p;
                const offset1 = i1 * stride3, offset0 = offset1 - stride3;
                const s2 = -2 * ppp + 3 * pp, s3 = ppp - pp, s0 = 1 - s2, s1 = s3 - pp + p;
                for (let i = 0; i !== stride; i++) {
                    const p0 = values[offset0 + i + stride], m0 = values[offset0 + i + stride2] * td;
                    const p1 = values[offset1 + i + stride], m1 = values[offset1 + i] * td;
                    result[i] = s0 * p0 + s1 * m0 + s2 * p1 + s3 * m1;
                }
                return result;
            }
            evaluate(t) {
                const pp = this.parameterPositions, n = pp.length;
                if (n === 0) return this.resultBuffer;
                if (t <= pp[0]) return this.copySampleValue_(0);
                if (t >= pp[n - 1]) return this.copySampleValue_(n - 1);
                let lo = 1, hi = n - 1;
                while (lo < hi) {
                    const mid = (lo + hi) >> 1;
                    if (t < pp[mid]) hi = mid; else lo = mid + 1;
                }
                return this.interpolate_(lo, pp[lo - 1], t, pp[lo]);
            }
        }
        class GLTFCubicSplineQuaternionInterpolant extends GLTFCubicSplineInterpolant {
            interpolate_(i1, t0, t, t1) {
                const r = super.interpolate_(i1, t0, t, t1);
                const len = Math.hypot(r[0], r[1], r[2], r[3]) || 1;
                r[0] /= len; r[1] /= len; r[2] /= len; r[3] /= len;
                return r;
            }
        }
        return { GLTFCubicSplineInterpolant, GLTFCubicSplineQuaternionInterpolant };
    })();
    return quaternion ? _gltfInterpolants.GLTFCubicSplineQuaternionInterpolant : _gltfInterpolants.GLTFCubicSplineInterpolant;
}

/// Whether the browser decodes WebP and AVIF (three.js's `detectSupport`).
let _gltfSupport = null;
function _gltfImageSupport() {
    return _gltfSupport ??= (async () => {
        const probe = async (type, b64) => {
            if (typeof createImageBitmap !== 'function') return false;
            try {
                const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
                const bmp = await createImageBitmap(new Blob([bytes], { type }));
                const ok = bmp.height === 1;
                bmp.close?.();
                return ok;
            } catch (_) { return false; }
        };
        const [webp, avif] = await Promise.all([
            probe('image/webp', 'UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEADsD+JaQAA3AAAAAA'),
            probe('image/avif', 'AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADybWV0YQAAAAAAAAAoaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAGxpYmF2aWYAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABGgAAABcAAAAoaWluZgAAAAAAAQAAABppbmZlAgAAAAABAABhdjAxQ29sb3IAAAAAamlwcnAAAABLaXBjbwAAABRpc3BlAAAAAAAAAAEAAAABAAAAEHBpeGkAAAAAAwgICAAAAAxhdjFDgQAMAAAAABNjb2xybmNseAACAAIABoAAAAAXaXBtYQAAAAAAAAABAAEEAQKDBAAAAB9tZGF0EgAKCBgABogQEDQgMgkQAAAAB8dSLfI='),
        ]);
        return { webp, avif };
    })();
}

/// An image the Rust decoders do not take (JPEG, WebP, AVIF…), through the
/// browser's: RGBA rows top first, or null without one.
async function _gltfDecodeImage(bytes, mimeType) {
    if (typeof createImageBitmap !== 'function') return null;
    const bmp = await createImageBitmap(new Blob([bytes], { type: mimeType }),
        { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    try {
        const { width, height } = bmp;
        const canvas = typeof OffscreenCanvas === 'function'
            ? new OffscreenCanvas(width, height)
            : Object.assign(document.createElement('canvas'), { width, height });
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(bmp, 0, 0);
        const data = ctx.getImageData(0, 0, width, height).data;
        return { width, height, data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) };
    } finally {
        bmp.close?.();
    }
}

/// Fetch a resource the file names (relative to the resource path), with the
/// loader's manager, headers and credentials.
function _gltfFetch(options, uri) {
    return new Promise((resolve, reject) => {
        const fl = new FileLoader(options.manager);
        fl.setResponseType('arraybuffer');
        fl.setRequestHeader(options.requestHeader);
        fl.setWithCredentials(options.crossOrigin === 'use-credentials');
        fl.load(LoaderUtils.resolveURL(uri, options.path), resolve, undefined, reject);
    });
}

/// Whether a Rust node (`nodeInfo`) is the node a JS object's mirror kind wants.
function _gltfSameKind(kind, info) {
    if (!info) return false;
    const [base, mode] = kind.split(':');
    if (base !== info.kind) return false;
    if (base === 'group') return info.isGroup === (mode === 'g');
    if (base === 'line') return info.lineMode === Number(mode);
    return true;
}

function _gltfWarn(p) {
    for (const w of p.takeWarnings()) console.warn(w);
}

/// three.js's `parser`: `json`, `associations`, `options`, `extensions`,
/// `plugins`, and `getDependency` / `getDependencies` (with the `load*`
/// names), resolved from what the Rust parser made — or loaded from it, for
/// what no scene used.
class _GLTFParser {
    constructor(state, json, options) {
        this._state = state;
        this.json = json;
        this.options = options;
        this.extensions = {};
        this.plugins = {};
        this.associations = new Map();
        this.fileLoader = new FileLoader(options.manager).setResponseType('arraybuffer');
        this.textureLoader = new TextureLoader(options.manager);
    }
    setExtensions(extensions) { this.extensions = extensions; }
    setPlugins(plugins) { this.plugins = plugins; }
    getDependency(type, index) { return this._state.dependency(type, index); }
    getDependencies(type) {
        const defs = this.json[type + (type === 'mesh' ? 'es' : 's')] || [];
        return Promise.all(defs.map((_, i) => this.getDependency(type, i)));
    }
    loadScene(i) { return this.getDependency('scene', i); }
    loadNode(i) { return this.getDependency('node', i); }
    loadMesh(i) { return this.getDependency('mesh', i); }
    loadMaterial(i) { return this.getDependency('material', i); }
    loadTexture(i) { return this.getDependency('texture', i); }
    loadCamera(i) { return this.getDependency('camera', i); }
    loadSkin(i) { return this.getDependency('skin', i); }
    loadAnimation(i) { return this.getDependency('animation', i); }
    loadAccessor(i) { return this.getDependency('accessor', i); }
    loadBufferView(i) { return this.getDependency('bufferView', i); }
    loadBuffer(i) { return this.getDependency('buffer', i); }
}

/// One load: the Rust parser, and the three.js objects wrapping what it made.
class _GLTFState {
    constructor(p, json, options) {
        this.p = p;
        this.graph = _hiddenGraph();
        this.options = options;
        this.parser = new _GLTFParser(this, json, options);
        this.objects = []; this.geometries = []; this.materials = []; this.textures = [];
        this.skeletons = []; this.clips = []; this.records = [];
        this.cache = new Map();
    }
    async fetchBuffers() {
        const p = this.p;
        await Promise.all(Array.from(p.externalBuffers(), async ([i, uri]) => {
            try {
                p.setBuffer(i, new Uint8Array(await _gltfFetch(this.options, uri)));
            } catch (_) {
                p.setBufferError(i, `THREE.GLTFLoader: Failed to load buffer "${uri}".`);
            }
        }));
    }
    /// Supply the images `next()` asks for until it asks for none.
    async fetchImages(next) {
        for (let round = 0; round < 8; round++) {
            const reqs = next();
            if (!reqs.length) return;
            await Promise.all(reqs.map((r) => this._image(r)));
        }
    }
    async _image(r) {
        const p = this.p;
        try {
            const bytes = r.bytes ?? new Uint8Array(await _gltfFetch(this.options, r.uri));
            if (p.setImageBytes(r.index, bytes)) return;
            const img = await _gltfDecodeImage(bytes, r.mimeType);
            if (img) p.setImageRgba(r.index, img.width, img.height, img.data);
            else p.setImageFailed(r.index, `no image decoder for ${r.mimeType}`);
        } catch (e) {
            console.error('THREE.GLTFLoader: Couldn\'t load texture', r.uri ?? `image ${r.index}`);
            p.setImageFailed(r.index, String(e?.message ?? e));
        }
    }
    /// `parser.getDependency`, cached per type and index as three.js caches it.
    dependency(type, index) {
        const key = type + ':' + index;
        let dep = this.cache.get(key);
        if (!dep) {
            dep = this._dependency(type, index);
            this.cache.set(key, dep);
        }
        return dep;
    }
    async _dependency(type, index) {
        const p = this.p;
        switch (type) {
            case 'accessor': {
                const a = p.accessor(index);
                return new BufferAttribute(a.array, a.itemSize);
            }
            case 'bufferView': return p.bufferView(index).buffer;
            case 'buffer': return p.buffer(index).buffer;
            case 'scene': case 'node': case 'mesh': case 'material': case 'texture':
            case 'camera': case 'skin': case 'animation': case 'light': {
                await this.fetchImages(() => p.pendingImagesFor(type, index));
                const d = p.load(this.graph.w, type, index);
                _gltfWarn(p);
                this.absorb(d);
                if (d.result < 0) return type === 'texture' ? null : undefined;
                const list = type === 'material' ? this.materials : type === 'texture' ? this.textures
                    : type === 'skin' ? this.skeletons : type === 'animation' ? this.clips : this.objects;
                return list[d.result];
            }
            default: {
                for (const plugin of Object.values(this.parser.plugins)) {
                    const dep = plugin.getDependency?.(type, index);
                    if (dep) return dep;
                }
                throw new Error('Unknown type: ' + type);
            }
        }
    }

    /// Wrap everything new in `d` (from `parse` or `load`).
    absorb(d) {
        d.records.forEach((r, i) => {
            const rec = this.records[i] ??= {};
            Object.assign(rec, r);
        });
        for (const t of d.textures) this.textures[t.index] = this._texture(t);
        for (const g of d.geometries) this.geometries[g.index] = this._geometry(g);
        for (const m of d.materials) this.materials[m.index] = this._material(m);
        const fresh = d.objects.map((o) => {
            const obj = this._object(o);
            this.objects[o.index] = obj;
            return [o, obj];
        });
        for (const [o, obj] of fresh) {
            if (o.lightTarget !== undefined) obj.target = this.objects[o.lightTarget];
        }
        for (const s of d.skeletons) {
            this.skeletons[s.index] = new Skeleton(
                s.bones.map((b) => this.objects[b]),
                s.bones.map((_, i) => new Matrix4().fromArray(s.inverses, i * 16)),
            );
        }
        for (const [o, obj] of fresh) {
            // `mesh.bind(skeleton, identity)`.
            if (o.skeleton !== undefined) obj.bind(this.skeletons[o.skeleton], new Matrix4());
        }
        d.children.forEach((kids, i) => {
            const parent = this.objects[i];
            if (!parent) return;
            for (const k of kids) {
                const c = this.objects[k];
                if (c && c.parent !== parent) parent.add(c);
            }
        });
        const assoc = this.parser.associations;
        const record = (a) => (a === -1 ? undefined : this.records[a]);
        for (const t of d.textures) if (t.association !== undefined) assoc.set(this.textures[t.index], record(t.association));
        for (const m of d.materials) if (m.association !== undefined) assoc.set(this.materials[m.index], record(m.association));
        for (const [o, obj] of fresh) {
            if (o.association !== undefined && o.association >= 0 && this.p.objectAssociated(o.index)) {
                assoc.set(obj, this.records[o.association]);
            }
        }
        this._adopt(fresh);
        for (const c of d.clips) this.clips[c.index] = this._clip(c);
    }

    _texture(t) {
        const tex = new Texture();
        tex._w = t.w;
        tex.isTexture = true;
        tex.uuid = MathUtils.generateUUID();
        tex.image = { width: t.width, height: t.height };
        tex.name = t.name;
        tex.flipY = false;
        tex.magFilter = t.magFilter; tex.minFilter = t.minFilter;
        tex.wrapS = t.wrapS; tex.wrapT = t.wrapT;
        tex.channel = t.channel;
        tex.offset = new Vector2(t.offset[0], t.offset[1]);
        tex.repeat = new Vector2(t.repeat[0], t.repeat[1]);
        tex.rotation = t.rotation;
        tex.center = new Vector2(0, 0);
        tex.colorSpace = t.srgb ? 'srgb' : '';
        tex.userData = t.userData;
        return tex;
    }

    _geometry(g) {
        const geo = new BufferGeometry();
        for (const a of g.attributes) geo.setAttribute(a.name, new BufferAttribute(a.array, a.itemSize));
        if (g.indices) geo.setIndex(g.indices);
        if (g.morphAttributes.length) {
            const morph = {};
            for (const e of g.morphAttributes) morph[e.name] = e.targets.map((t) => new BufferAttribute(t.array, t.itemSize));
            geo.morphAttributes = morph;
            geo.morphTargetsRelative = g.morphTargetsRelative;
        }
        geo.userData = g.userData;
        if (g.boxMin) {
            geo.boundingBox = new Box3(new Vector3(...g.boxMin), new Vector3(...g.boxMax));
            geo.boundingSphere = new Sphere(new Vector3(...g.sphereCenter), g.sphereRadius);
        }
        // The wasm geometry is the parser's — attributes, index and morph
        // targets as these arrays hold them — until an edit rebuilds it.
        geo._w = g.wBuffer;
        geo._geom = g.w;
        geo._geomVersion = geo._version;
        geo._geomRange = geo._rangeVersion;
        return geo;
    }

    _material(m) {
        const Ctor = { MeshStandardMaterial, MeshPhysicalMaterial, MeshBasicMaterial, PointsMaterial, LineBasicMaterial }[m.type];
        const P = m.params;
        const scalar = {};
        for (const k of ['roughness', 'metalness', 'clearcoat', 'clearcoatRoughness', 'opacity', 'transparent', 'side', 'alphaTest', 'depthWrite', 'depthTest']) {
            if (typeof P[k] === 'number' || typeof P[k] === 'boolean') scalar[k] = P[k];
        }
        const mat = new Ctor(scalar);
        // Parameters first, while the setters reach only the placeholder wasm
        // material the constructor made; then the parser's own material.
        const textures = [];
        for (const [k, v] of Object.entries(P)) {
            if (v && typeof v === 'object' && 'texture' in v) { textures.push([k, this.textures[v.texture]]); continue; }
            if (v && typeof v === 'object' && 'color' in v) {
                if (k === 'color' && typeof mat.color?.setRGB === 'function') mat.color.setRGB(...v.color);
                else mat[k] = new Color(...v.color);
                continue;
            }
            if (v && typeof v === 'object' && 'vec2' in v) { mat[k] = new Vector2(...v.vec2); continue; }
            if (Array.isArray(v)) { mat[k] = v.slice(); continue; }
            mat[k] = v;
        }
        const defaults = {
            ..._GLTF_BASE_DEFAULTS,
            ...(_GLTF_MATERIAL_DEFAULTS[m.type === 'MeshPhysicalMaterial' ? 'MeshStandardMaterial' : m.type] ?? {}),
            ...(m.type === 'MeshPhysicalMaterial' ? _GLTF_MATERIAL_DEFAULTS.MeshPhysicalMaterial : {}),
        };
        for (const [k, def] of Object.entries(defaults)) {
            if (!(k in P) && mat[k] === undefined) mat[k] = typeof def === 'function' ? def() : def;
        }
        if (!(mat.color instanceof Color)) mat.color = new Color(1, 1, 1);
        if (m.w) mat._w = m.w;
        for (const [k, tex] of textures) {
            // A map slot with a setter keeps its texture in `_<slot>`; set that,
            // so the wasm material keeps the texture view the parser chose.
            const d = Object.getOwnPropertyDescriptor(mat, k);
            if (d?.set && ('_' + k) in mat) mat['_' + k] = tex;
            else if (d?.set) Object.defineProperty(mat, k, { value: tex, writable: true, enumerable: true, configurable: true });
            else mat[k] = tex;
        }
        mat.name = m.name;
        mat.userData = m.userData;
        mat.type = m.type;
        mat['is' + m.type] = true;
        if (m.type === 'MeshPhysicalMaterial') mat.isMeshStandardMaterial = true;
        mat.isMaterial = true;
        mat.uuid ??= MathUtils.generateUUID();
        return mat;
    }

    _object(o) {
        const geo = o.geometry !== undefined ? this.geometries[o.geometry] : undefined;
        const mat = o.material !== undefined ? this.materials[o.material] : undefined;
        let obj;
        switch (o.type) {
            case 'Group': obj = new Group(); break;
            case 'Bone': obj = new Bone(); break;
            case 'Mesh': obj = new Mesh(geo, mat); break;
            case 'SkinnedMesh': obj = new SkinnedMesh(geo, mat); break;
            case 'InstancedMesh':
                obj = new InstancedMesh(geo, mat, o.count);
                obj.instanceMatrix.array.set(o.instanceMatrix);
                if (o.instanceColor) obj.instanceColor = new InstancedBufferAttribute(o.instanceColor, o.instanceColorSize);
                break;
            case 'Line': obj = new Line(geo, mat); break;
            case 'LineSegments': obj = new LineSegments(geo, mat); break;
            case 'LineLoop': obj = new LineLoop(geo, mat); break;
            case 'Points': obj = new Points(geo, mat); break;
            case 'PerspectiveCamera': obj = new PerspectiveCamera(...o.camera); break;
            case 'OrthographicCamera': obj = new OrthographicCamera(...o.camera); break;
            case 'DirectionalLight': obj = new DirectionalLight(); break;
            case 'PointLight': obj = new PointLight(); break;
            case 'SpotLight': obj = new SpotLight(); break;
            default: obj = new Object3D();
        }
        if (o.light) {
            const l = o.light;
            obj.color.setRGB(...l.color);
            obj.intensity = l.intensity;
            obj.decay = l.decay;
            if (o.type !== 'DirectionalLight') obj.distance = l.distance;
            if (o.type === 'SpotLight') { obj.angle = l.angle; obj.penumbra = l.penumbra; }
            if (o.w) obj._w = o.w;
        }
        obj.name = o.name;
        obj.userData = o.userData;
        if (o.position) {
            obj.position.set(...o.position);
            obj.quaternion.set(...o.quaternion);
            obj.scale.set(...o.scale);
        }
        if (o.morphTargetInfluences) {
            obj.morphTargetInfluences = o.morphTargetInfluences.slice();
            obj.morphTargetDictionary = { ...o.morphTargetDictionary };
        }
        return obj;
    }

    /// Give each new object the node the parser made for it (rather than a
    /// new one), and push its state as the mirror does for a node it creates.
    _adopt(fresh) {
        const graph = this.graph, w = graph.w;
        const adopted = [];
        for (const [o, obj] of fresh) {
            const handle = w.handleOf(o.id);
            if (!handle) continue;
            const kind = obj._mirrorKind();
            if (!_gltfSameKind(kind, w.nodeInfo(handle))) {
                // Not the node this object needs: let the mirror make that.
                console.warn(`THREE.GLTFLoader: node for ${o.type} "${o.name}" is a ${w.nodeInfo(handle)?.kind}; rebuilt`);
                w.deleteNode(handle);
                continue;
            }
            const rec = { graph, handle, id: handle.id, kind, parentId: undefined, childIds: null };
            obj._m = rec;
            graph.objects.set(rec.id, obj._wref ??= new WeakRef(obj));
            _nodeFinalizer?.register(obj, { graph, id: rec.id }, rec);
            _mirrorQueue.delete(obj);
            adopted.push(obj);
        }
        for (const obj of adopted) if (obj._m) _mirrorPush(obj, _D_ALL | obj._dirtyFlags);
    }

    _clip(c) {
        const tracks = c.tracks.map((t) => {
            const target = this.objects[t.target];
            const name = (target.name ? target.name : target.uuid) + '.' + t.property;
            const Track = t.kind === 'quaternion' ? QuaternionKeyframeTrack
                : t.kind === 'number' ? NumberKeyframeTrack : VectorKeyframeTrack;
            const track = new Track(name, t.times, t.values, _GLTF_INTERPOLATION[t.interpolation]);
            if (t.interpolation === 'CUBICSPLINE') {
                const quaternion = t.kind === 'quaternion';
                track.createInterpolant = function InterpolantFactoryMethodGLTFCubicSpline(result) {
                    const Interp = _gltfCubicInterpolant(quaternion);
                    return new Interp(this.times, this.values, this.getValueSize() / 3, result);
                };
                track.createInterpolant.isInterpolantFactoryMethodGLTFCubicSpline = true;
            }
            return track;
        });
        return new AnimationClip(c.name, -1, tracks);
    }
}

/// `GLTFLoader.parse`, to the three.js result.
async function _gltfParse(loader, data, path) {
    const Binding = _optional.WebGltfParser;
    if (!Binding) throw new Error('THREE.GLTFLoader: this wasm build has no glTF parser (rebuild web/pkg)');
    let bytes;
    if (typeof data === 'string') bytes = new TextEncoder().encode(data);
    else if (data instanceof ArrayBuffer) bytes = new Uint8Array(data);
    else if (ArrayBuffer.isView(data)) bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    else bytes = new TextEncoder().encode(JSON.stringify(data));
    const p = new Binding(bytes);
    const isObject = data && typeof data === 'object' && !(data instanceof ArrayBuffer) && !ArrayBuffer.isView(data);
    const json = isObject ? data : JSON.parse(p.jsonText());
    const options = {
        path: path || loader.resourcePath || '',
        crossOrigin: loader.crossOrigin,
        requestHeader: loader.requestHeader,
        manager: loader.manager,
        ktx2Loader: loader.ktx2Loader,
        meshoptDecoder: loader.meshoptDecoder,
    };
    const state = new _GLTFState(p, json, options);
    const parser = state.parser;
    const extensions = {};
    for (const name of _GLTF_PLUGIN_NAMES) extensions[name] = true;
    for (const name of json.extensionsUsed || []) if (_GLTF_USED_EXTENSIONS.includes(name)) extensions[name] = { name };
    const plugins = {};
    for (const callback of loader.pluginCallbacks) {
        const plugin = callback(parser);
        if (!plugin.name) console.error('THREE.GLTFLoader: Invalid plugin found: missing name');
        plugins[plugin.name] = plugin;
        extensions[plugin.name] = true;
        for (const hook of ['loadMesh', 'loadNode', 'loadMaterial', 'loadTexture', 'loadBufferView', 'loadAnimation',
            'createNodeMesh', 'createNodeAttachment', 'getMaterialType', 'extendMaterialParams']) {
            if (typeof plugin[hook] === 'function') {
                console.warn(`THREE.GLTFLoader: plugin "${plugin.name}" implements ${hook}, which this loader does not call (only beforeRoot, afterRoot and getDependency are).`);
            }
        }
    }
    parser.setExtensions(extensions);
    parser.setPlugins(plugins);
    _gltfWarn(p);
    const used = json.extensionsUsed || [];
    if (used.includes('EXT_texture_webp') || used.includes('EXT_texture_avif')) {
        const support = await _gltfImageSupport();
        p.setImageSupport(support.webp, support.avif);
    }
    await state.fetchBuffers();
    await Promise.all(Object.values(plugins).map((pl) => pl.beforeRoot?.()));
    await state.fetchImages(() => p.pendingImages());
    const d = p.parse(state.graph.w);
    _gltfWarn(p);
    state.absorb(d);
    // What `getDependency` would load now resolves to what `parse` made.
    d.scenes.forEach((s, i) => s >= 0 && state.cache.set('scene:' + i, Promise.resolve(state.objects[s])));
    d.animations.forEach((a, i) => a >= 0 && state.cache.set('animation:' + i, Promise.resolve(state.clips[a])));
    const result = {
        scene: d.scene === undefined ? undefined : state.objects[d.scene],
        scenes: d.scenes.map((s) => state.objects[s]),
        animations: d.animations.map((a) => state.clips[a]),
        cameras: d.cameras.map((c) => (c < 0 ? undefined : state.objects[c])),
        asset: json.asset,
        parser,
        userData: d.userData,
    };
    await Promise.all(Object.values(plugins).map((pl) => pl.afterRoot?.(result)));
    for (const scene of result.scenes) scene.updateMatrixWorld();
    return result;
}

/// three.js r165 `GLTFLoader`: `load` / `loadAsync` / `parse` / `parseAsync`,
/// the resource path and request options, and the decoder setters (accepted;
/// Draco, meshopt and KTX2 are decoded in Rust whichever is set).
export class GLTFLoader extends _Loader {
    constructor(manager) {
        super(manager);
        this.dracoLoader = null;
        this.ktx2Loader = null;
        this.meshoptDecoder = null;
        this.pluginCallbacks = [];
    }
    load(url, onLoad, onProgress, onError) {
        let resourcePath;
        if (this.resourcePath !== '') resourcePath = this.resourcePath;
        else if (this.path !== '') resourcePath = LoaderUtils.resolveURL(LoaderUtils.extractUrlBase(url), this.path);
        else resourcePath = LoaderUtils.extractUrlBase(url);
        this.manager.itemStart(url);
        const _onError = (e) => {
            if (onError) onError(e); else console.error(e);
            this.manager.itemError(url);
            this.manager.itemEnd(url);
        };
        const loader = new FileLoader(this.manager);
        loader.setPath(this.path);
        loader.setResponseType('arraybuffer');
        loader.setRequestHeader(this.requestHeader);
        loader.setWithCredentials(this.withCredentials);
        loader.load(url, (data) => {
            this.parse(data, resourcePath, (gltf) => {
                onLoad(gltf);
                this.manager.itemEnd(url);
            }, _onError);
        }, onProgress, _onError);
    }
    setDRACOLoader(dracoLoader) { this.dracoLoader = dracoLoader; return this; }
    setDDSLoader() {
        throw new Error('THREE.GLTFLoader: "MSFT_texture_dds" no longer supported. Please update to "KHR_texture_basisu".');
    }
    setKTX2Loader(ktx2Loader) { this.ktx2Loader = ktx2Loader; return this; }
    setMeshoptDecoder(meshoptDecoder) { this.meshoptDecoder = meshoptDecoder; return this; }
    register(callback) {
        if (this.pluginCallbacks.indexOf(callback) === -1) this.pluginCallbacks.push(callback);
        return this;
    }
    unregister(callback) {
        const i = this.pluginCallbacks.indexOf(callback);
        if (i !== -1) this.pluginCallbacks.splice(i, 1);
        return this;
    }
    parse(data, path, onLoad, onError) {
        // As three.js's: an error thrown by `onLoad` reaches `onError` too.
        _gltfParse(this, data, path)
            .then((gltf) => onLoad(gltf))
            .catch((e) => { if (onError) onError(e); else console.error(e); });
    }
    parseAsync(data, path) {
        return new Promise((resolve, reject) => this.parse(data, path, resolve, reject));
    }
}
/**
 * Alembic (`.abc`) geometry caches.
 *
 * The Rust side reads the Ogawa container and the `PolyMesh` geometry in it.
 * `parse` gives the first mesh; `meshNames` says what else is in there, because
 * a cache usually holds a scene rather than a model.
 */
export class AlembicLoader extends _Loader {
    constructor(manager) { super(manager); this._w = new WebAlembicLoader(); }
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (buf) => {
            try {
                const u8 = buf instanceof ArrayBuffer ? new Uint8Array(buf) : buf;
                const group = new Group();
                group.add(new Mesh(_wrapLoaderGeometry(this._w.parse(u8)), new MeshStandardMaterial()));
                onLoad?.(group);
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return _wrapLoaderGeometry(this._w.parse(u8));
    }
    /** The names of every PolyMesh in the archive. */
    meshNames(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return this._w.mesh_names(u8);
    }
}

/**
 * USD — `.usda`, `.usdc` and `.usdz`.
 *
 * Available only when the wasm bundle was built with the `usd` feature, which
 * is why the binding is looked up rather than imported: a build without it
 * should fail with a clear message here, not at import time for everyone.
 */
export class USDLoader extends _Loader {
    constructor(manager) {
        super(manager);
        const Binding = _optional.WebUsdLoader;
        if (!Binding) {
            throw new Error('USDLoader: this build has no USD support (rebuild with --features usd)');
        }
        this._w = new Binding();
    }
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (buf) => {
            try {
                const u8 = buf instanceof ArrayBuffer ? new Uint8Array(buf) : buf;
                const group = new Group();
                group.add(new Mesh(_wrapLoaderGeometry(this._w.parse(u8)), new MeshStandardMaterial()));
                onLoad?.(group);
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return _wrapLoaderGeometry(this._w.parse(u8));
    }
    /** Which of the three forms the bytes are — USD has no useful extension. */
    sniff(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return this._w.sniff(u8);
    }
}

export class FBXLoader extends _Loader {
    constructor(manager) { super(manager); this._w = new WebFbxLoader(); }
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (buf) => {
            try {
                const u8 = buf instanceof ArrayBuffer ? new Uint8Array(buf) : buf;
                const geom = this._w.parse(u8);
                const group = new Group();
                group.add(new Mesh(_wrapLoaderGeometry(geom), new MeshStandardMaterial()));
                onLoad?.(group);
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        const geom = this._w.parse(u8);
        return _wrapLoaderGeometry(geom);
    }
}
export class ColladaLoader extends _Loader {
    constructor(manager) { super(manager); this._w = new WebColladaLoader(); }
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (text) => {
            try {
                const geom = this._w.parse(text);
                const scene = new Scene();
                scene.add(new Mesh(_wrapLoaderGeometry(geom), new MeshStandardMaterial()));
                onLoad?.({ scene });
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(text) {
        const geom = this._w.parse(text);
        const scene = new Scene();
        scene.add(new Mesh(_wrapLoaderGeometry(geom), new MeshStandardMaterial()));
        return { scene };
    }
}
export class EXRLoader extends _Loader {
    constructor(manager) { super(manager); this._w = new WebExrLoader(); }
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (buf) => {
            try {
                const u8 = buf instanceof ArrayBuffer ? new Uint8Array(buf) : buf;
                onLoad?.(_wrapWebTexture(this._w.parse(u8)));
            } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        return _wrapWebTexture(this._w.parse(u8));
    }
}

// KTX2Loader — the container, zlib supercompression, BGRA and BC1/BC7 blocks
// read by `WebKtx2Loader` (src/loaders/ktx2.rs); Basis Universal and zstd
// payloads are refused by name, as there is no transcoder here.
export class KTX2Loader extends _Loader {
    constructor(manager) {
        super(manager);
        this._w = new WebKtx2Loader();
        this.transcoderPath = '';
    }
    setTranscoderPath(path) { this.transcoderPath = path; return this; }
    setWorkerLimit() { return this; }
    detectSupport() { return this; }
    dispose() { return this; }
    _parse(buffer) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
        const texture = _wrapWebTexture(this._w.parseTexture(u8));
        texture.flipY = false;
        texture.generateMipmaps = false;
        texture.needsUpdate = true;
        return texture;
    }
    load(url, onLoad, onProgress, onError) {
        new FileLoader(this.manager).load(url, (buf) => {
            try { onLoad?.(this._parse(buf)); } catch (e) { if (onError) onError(e); else throw e; }
        }, onProgress, onError);
    }
    parse(buffer, onLoad, onError) {
        try {
            const texture = this._parse(buffer);
            onLoad?.(texture);
            return texture;
        } catch (e) {
            if (onError) onError(e); else throw e;
        }
    }
}

// DRACOLoader — `.drc` meshes decoded by the crate's Draco decoder
// (src/loaders/draco/), not Google's: `WebDracoLoader.decode` returns what
// three.js's DRACOLoader builds (position / normal / color / uv + index).
export class DRACOLoader extends _Loader {
    constructor(manager) {
        super(manager);
        this._w = new WebDracoLoader();
        this.decoderPath = '';
        this.decoderConfig = {};
        this.workerLimit = 4;
    }
    setDecoderPath(path) { this.decoderPath = path; return this; }
    setDecoderConfig(config) { this.decoderConfig = config; return this; }
    setWorkerLimit(workerLimit) { this.workerLimit = workerLimit; return this; }
    preload() { return this; }
    dispose() { return this; }
    _decode(buffer, srgbColors) {
        const u8 = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : new Uint8Array(buffer.buffer ?? buffer, buffer.byteOffset ?? 0, buffer.byteLength);
        return _fillFromWasm(new BufferGeometry(), this._w.decode(u8, srgbColors));
    }
    /// three.js: `parse` decodes vertex colours as sRGB.
    parse(buffer, onLoad, onError = () => {}) {
        try { onLoad?.(this._decode(buffer, true)); } catch (e) { onError(e); }
    }
    /// three.js: linear vertex colours unless `vertexColorSpace` is sRGB.
    decodeDracoFile(buffer, callback, _attributeIDs, _attributeTypes, vertexColorSpace = 'srgb-linear', onError = () => {}) {
        return Promise.resolve()
            .then(() => {
                const geometry = this._decode(buffer, vertexColorSpace === 'srgb');
                callback?.(geometry);
                return geometry;
            })
            .catch(onError);
    }
    load(url, onLoad, onProgress, onError) {
        new FileLoader(this.manager).load(url, (buf) => this.parse(buf, onLoad, onError), onProgress, onError);
    }
}

// ---- Pass 7: more controls + post-fx + WebGLRenderer extras ----
function _noopControls(type) {
    return { _w: null, type, enabled: true, update() {}, dispose() {}, addEventListener(){}, removeEventListener(){}, dispatchEvent(){} };
}

// DragControls — three.js r165's, over `controls/drag.rs`: the cast is the
// shim's Raycaster (objects' own `raycast` included, as in three.js), the
// drag itself — the plane facing the camera through the picked object, the
// parent's transform divided out, `transformGroup`, rotate mode — runs in
// Rust on the scene node, which the object then reads back. Dispatches
// `hoveron` / `hoveroff` / `dragstart` / `drag` / `dragend` and sets the
// cursor as three.js does.
export class DragControls extends EventDispatcher {
    constructor(objects = [], camera, domElement) {
        super();
        this._objects = objects;
        this._camera = camera;
        this._dom = domElement ?? null;
        this._w = new _optional.WebDragControls();
        this._raycaster = new Raycaster();
        this._pointer = new Vector2();
        this._graph = null;
        this.recursive = true;
        if (this._dom?.style) this._dom.style.touchAction = 'none';
        this._onMove = (e) => this._pointerMove(e);
        this._onDown = (e) => this._pointerDown(e);
        this._onCancel = () => this._pointerCancel();
        this.activate();
    }
    get enabled() { return this._w.enabled; }
    set enabled(v) { this._w.enabled = !!v; }
    get transformGroup() { return this._w.transformGroup; }
    set transformGroup(v) { this._w.transformGroup = !!v; }
    get mode() { return this._w.mode; }
    set mode(v) { this._w.mode = String(v); }
    get rotateSpeed() { return this._w.rotateSpeed; }
    set rotateSpeed(v) { this._w.rotateSpeed = Number(v); }
    get objects() { return this._objects; }
    activate() {
        const d = this._dom;
        if (!d?.addEventListener) return;
        d.addEventListener('pointermove', this._onMove);
        d.addEventListener('pointerdown', this._onDown);
        d.addEventListener('pointerup', this._onCancel);
        d.addEventListener('pointerleave', this._onCancel);
    }
    deactivate() {
        const d = this._dom;
        if (!d?.removeEventListener) return;
        d.removeEventListener('pointermove', this._onMove);
        d.removeEventListener('pointerdown', this._onDown);
        d.removeEventListener('pointerup', this._onCancel);
        d.removeEventListener('pointerleave', this._onCancel);
        if (d.style) d.style.cursor = '';
    }
    dispose() { this.deactivate(); }
    getObjects() { return this._objects; }
    setObjects(objects) { this._objects = objects; }
    getRaycaster() { return this._raycaster; }
    _updatePointer(event) {
        const rect = this._dom.getBoundingClientRect();
        this._pointer.x = (event.clientX - rect.left) / rect.width * 2 - 1;
        this._pointer.y = -(event.clientY - rect.top) / rect.height * 2 + 1;
    }
    /// The nearest hit among the draggables, three.js's cast.
    _pick() {
        const hits = this._raycaster.intersectObjects(this._objects, this.recursive, []);
        return hits.length ? hits[0].object : null;
    }
    _cameraArgs() {
        const c = this._camera;
        c.updateWorldMatrix(true, false);
        const q = c.quaternion;
        return [Float32Array.from(c.matrixWorld.elements), Float32Array.from([q.x, q.y, q.z, q.w])];
    }
    /// Rust's events and cursor, onto the objects: a dragged object takes the
    /// node's new position (or rotation), then each event is dispatched.
    _apply(outcome, graph) {
        for (const ev of outcome.events) {
            const object = graph.objectOf(ev.objectId);
            if (ev.type === 'drag' && object?._m) {
                const t = graph.w.getTRS(object._m.handle);
                if (this.mode === 'rotate') object.quaternion.set(t[3], t[4], t[5], t[6]);
                else object.position.set(t[0], t[1], t[2]);
            }
            this.dispatchEvent({ type: ev.type, object });
        }
        if (outcome.cursor !== undefined && this._dom?.style) this._dom.style.cursor = outcome.cursor;
    }
    _pointerDown(event) {
        if (this.enabled === false) return;
        this._updatePointer(event);
        this._raycaster.setFromCamera(this._pointer, this._camera);
        const hit = this._pick();
        const graph = hit?._m?.graph ?? this._graph ?? _hiddenGraph();
        this._graph = graph;
        const [cw, cq] = this._cameraArgs();
        this._apply(this._w.pointerDown(graph.w, this._pointer.x, this._pointer.y, this._raycaster._w,
            hit?._m?.id, cw, cq), graph);
    }
    _pointerMove(event) {
        if (this.enabled === false) return;
        this._updatePointer(event);
        this._raycaster.setFromCamera(this._pointer, this._camera);
        let graph = this._graph ?? _hiddenGraph();
        let hover = false, hitId;
        const selected = this._w.selected;
        if (selected === undefined) {
            if (event.pointerType === 'mouse' || event.pointerType === 'pen') {
                hover = true;
                const hit = this._pick();
                if (hit?._m) { graph = hit._m.graph; hitId = hit._m.id; }
            }
        } else {
            // The dragged node as the scene has it now.
            const obj = graph.objectOf(selected);
            if (obj) _mirrorSync(obj);
        }
        this._graph = graph;
        const [cw, cq] = this._cameraArgs();
        this._apply(this._w.pointerMove(graph.w, this._pointer.x, this._pointer.y, this._raycaster._w,
            hover, hitId, cw, cq), graph);
    }
    _pointerCancel() {
        if (this.enabled === false) return;
        this._apply(this._w.pointerCancel(), this._graph ?? _hiddenGraph());
    }
    update() {}
}

// FlyControls — three.js r165 (examples/jsm/controls/FlyControls.js) over
// `controls/fly.rs`: the move state, the movement / rotation vectors and
// `update(delta)`'s translate-and-turn run in Rust; this class forwards the
// DOM events (keys on `window`, the pointer on the element) and applies the
// pose to the object.
const _flyMoveKeys = ['up', 'down', 'left', 'right', 'forward', 'back', 'pitchUp', 'pitchDown',
    'yawLeft', 'yawRight', 'rollLeft', 'rollRight'];
export class FlyControls extends EventDispatcher {
    constructor(object, domElement) {
        super();
        this.object = object;
        this.domElement = domElement;
        this._w = new _optional.WebFlyControls();
        this.tmpQuaternion = new Quaternion();
        this.moveVector = new Vector3();
        this.rotationVector = new Vector3();
        // `moveState` reads and writes through to Rust, so code that sets
        // `moveState.forward = 1` and calls `updateMovementVector()` works.
        const state = {};
        for (const [i, key] of _flyMoveKeys.entries()) {
            Object.defineProperty(state, key, {
                enumerable: true,
                get: () => this._w.moveState()[i],
                set: (v) => { const s = this._w.moveState(); s[i] = Number(v) || 0; this._w.setMoveState(s); },
            });
        }
        this.moveState = state;
        this.keydown = (event) => { this._w.keyDown(event.code, !!event.altKey); this._readVectors(); };
        this.keyup = (event) => { this._w.keyUp(event.code); this._readVectors(); };
        this.pointerdown = (event) => { this._w.pointerDown(event.button); this._readVectors(); };
        this.pointermove = (event) => {
            const c = this.getContainerDimensions();
            this._w.pointerMove(event.pageX, event.pageY, c.size[0], c.size[1], c.offset[0], c.offset[1]);
            this._readVectors();
        };
        this.pointerup = (event) => { this._w.pointerUp(event.button); this._readVectors(); };
        this.pointercancel = () => { this._w.pointerCancel(); this._readVectors(); };
        this.contextMenu = (event) => { if (this.enabled === false) return; event.preventDefault(); };
        const d = domElement;
        if (d?.addEventListener) {
            d.addEventListener('contextmenu', this.contextMenu);
            d.addEventListener('pointerdown', this.pointerdown);
            d.addEventListener('pointermove', this.pointermove);
            d.addEventListener('pointerup', this.pointerup);
            d.addEventListener('pointercancel', this.pointercancel);
            if (typeof window !== 'undefined') {
                window.addEventListener('keydown', this.keydown);
                window.addEventListener('keyup', this.keyup);
            }
        }
        this._readVectors();
    }
    get enabled() { return this._w.enabled; }
    set enabled(v) { this._w.enabled = !!v; }
    get movementSpeed() { return this._w.movementSpeed; }
    set movementSpeed(v) { this._w.movementSpeed = Number(v); }
    get rollSpeed() { return this._w.rollSpeed; }
    set rollSpeed(v) { this._w.rollSpeed = Number(v); }
    get dragToLook() { return this._w.dragToLook; }
    set dragToLook(v) { this._w.dragToLook = !!v; }
    get autoForward() { return this._w.autoForward; }
    set autoForward(v) { this._w.autoForward = !!v; }
    get movementSpeedMultiplier() { return this._w.movementSpeedMultiplier; }
    get status() { return this._w.status; }
    _readVectors() {
        const v = this._w.vectors();
        this.moveVector.set(v[0], v[1], v[2]);
        this.rotationVector.set(v[3], v[4], v[5]);
    }
    updateMovementVector() { this._w.updateMovementVector(); this._readVectors(); }
    updateRotationVector() { this._w.updateRotationVector(); this._readVectors(); }
    getContainerDimensions() {
        if (typeof document === 'undefined' || this.domElement != document) {
            return {
                size: [this.domElement.offsetWidth, this.domElement.offsetHeight],
                offset: [this.domElement.offsetLeft, this.domElement.offsetTop],
            };
        }
        return { size: [window.innerWidth, window.innerHeight], offset: [0, 0] };
    }
    update(delta) {
        const o = this.object;
        if (!o) return;
        const p = o.position, q = o.quaternion;
        const r = this._w.update(delta, Float64Array.of(p.x, p.y, p.z, q.x, q.y, q.z, q.w));
        if (!r) return;
        p.set(r[0], r[1], r[2]);
        q.set(r[3], r[4], r[5], r[6]);
        if (r[7]) this.dispatchEvent({ type: 'change' });
    }
    dispose() {
        const d = this.domElement;
        if (d?.removeEventListener) {
            d.removeEventListener('contextmenu', this.contextMenu);
            d.removeEventListener('pointerdown', this.pointerdown);
            d.removeEventListener('pointermove', this.pointermove);
            d.removeEventListener('pointerup', this.pointerup);
            d.removeEventListener('pointercancel', this.pointercancel);
        }
        if (typeof window !== 'undefined') {
            window.removeEventListener('keydown', this.keydown);
            window.removeEventListener('keyup', this.keyup);
        }
    }
}

// The actions the controls' `mouseButtons` map buttons to, by three.js's
// numbers. Internal: threers' API has no `MOUSE`.
const _MOUSE = { LEFT: 0, MIDDLE: 1, RIGHT: 2, ROTATE: 0, DOLLY: 1, PAN: 2 };
// three.js's `TOUCH` (the controls' `touches`).
export const TOUCH = { ROTATE: 0, PAN: 1, DOLLY_PAN: 2, DOLLY_ROTATE: 3 };

// MapControls — three.js r165: OrbitControls with the left button panning,
// the right rotating, and panning over the plane orthogonal to the camera's
// up (`screenSpacePanning = false`), all in the Rust orbit controls.
export class MapControls extends OrbitControls {
    constructor(object, domElement) {
        super(object, domElement);
        this.screenSpacePanning = false;
        this.mouseButtons = { LEFT: _MOUSE.PAN, MIDDLE: _MOUSE.DOLLY, RIGHT: _MOUSE.ROTATE };
        this.touches = { ONE: TOUCH.PAN, TWO: TOUCH.DOLLY_ROTATE };
    }
}

// ArcballControls is a full-featured UI in three.js (gizmos, arcs); this
// provides working camera-orbit behaviour so user code that calls .update()
// doesn't crash.
export class ArcballControls {
    constructor(camera, domElement) {
        this._w = camera?._w ? new WebArcballControls(camera._w) : null;
        this.camera = camera || null;
        this.domElement = domElement || (typeof document !== 'undefined' ? document.body : null);
        this.enabled = true;
        this.target = new Vector3();
        this._rotating = false;
        this._ndc = new Vector2();
        Object.assign(this, EventDispatcherMixin);
        if (this.domElement?.addEventListener) {
            this._onDown = (e) => { if (e.button === 0) this._rotating = true; this._updateNdc(e); };
            this._onMove = (e) => { if (this._rotating) this._updateNdc(e); };
            this._onUp = () => { this._rotating = false; };
            this._onWheel = (e) => { if (this.enabled && this._w && this.camera?._w) { this._w.update(this.camera._w, this._ndc.x, this._ndc.y, e.deltaY, false); _cameraFromWasm(this.camera); } };
            this.domElement.addEventListener('pointerdown', this._onDown);
            this.domElement.addEventListener('pointermove', this._onMove);
            this.domElement.addEventListener('pointerup', this._onUp);
            this.domElement.addEventListener('wheel', this._onWheel);
        }
    }
    _updateNdc(e) {
        const rect = this.domElement.getBoundingClientRect();
        this._ndc.set(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -(((e.clientY - rect.top) / rect.height) * 2 - 1),
        );
        if (this.enabled && this._w) {
            this._w.setTarget(this.target.x, this.target.y, this.target.z);
            this._w.update(this.camera._w, this._ndc.x, this._ndc.y, 0, this._rotating);
            _cameraFromWasm(this.camera);
        }
    }
    update() {}
    dispose() {
        if (this.domElement?.removeEventListener) {
            this.domElement.removeEventListener('pointerdown', this._onDown);
            this.domElement.removeEventListener('pointermove', this._onMove);
            this.domElement.removeEventListener('pointerup', this._onUp);
            this.domElement.removeEventListener('wheel', this._onWheel);
        }
    }
    reset() { this.target.set(0, 0, 0); }
}
// TransformControls — three.js r165 (examples/jsm/controls/TransformControls.js)
// over `controls/transform` (Rust): the gizmo's handles (geometry baked as
// three.js's `setupGizmo` bakes it, its shared materials), their placement,
// hiding and highlighting each `updateMatrixWorld`, the drag plane, and the
// translate / rotate / scale math run there in f64. This class is the DOM and
// scene glue: it is the Object3D (with `_gizmo` and `_plane` children) the
// app adds to its scene, it casts at the pickers with the (Rust) Raycaster,
// hands the object's and camera's matrices over, and applies the poses and
// dispatches the events that come back in three.js's order.
class TransformControlsGizmo extends Object3D {
    constructor() {
        super();
        this.isTransformControlsGizmo = true;
        this.type = 'TransformControlsGizmo';
        this.gizmo = {};
        this.picker = {};
        this.helper = {};
    }
}
class TransformControlsPlane extends Mesh {
    constructor(geometry) {
        super(geometry, new MeshBasicMaterial({
            visible: false, wireframe: true, side: DoubleSide, transparent: true, opacity: 0.1, toneMapped: false,
        }));
        this.isTransformControlsPlane = true;
        this.type = 'TransformControlsPlane';
    }
}
/// A Rust-made geometry (`{ position, normal?, uv?, index? }`) as a BufferGeometry.
function _tcGeometry(g) {
    const geo = new BufferGeometry();
    geo.setAttribute('position', new BufferAttribute(g.position, 3));
    if (g.normal) geo.setAttribute('normal', new BufferAttribute(g.normal, 3));
    if (g.uv) geo.setAttribute('uv', new BufferAttribute(g.uv, 2));
    if (g.index) geo.setIndex(new BufferAttribute(g.index, 1));
    return geo;
}
const _tcRustProps = ['enabled', 'axis', 'mode', 'translationSnap', 'rotationSnap', 'scaleSnap',
    'space', 'size', 'dragging', 'showX', 'showY', 'showZ', 'rotationAngle'];
const _tcMatrix = new Matrix4();
export class TransformControls extends Object3D {
    constructor(camera, domElement) {
        super();
        if (domElement === undefined) {
            console.warn('THREE.TransformControls: The second parameter "domElement" is now mandatory.');
            domElement = typeof document !== 'undefined' ? document : null;
        }
        this.isTransformControls = true;
        this.visible = false;
        this.domElement = domElement;
        if (domElement?.style) domElement.style.touchAction = 'none';
        const W = _optional.WebTransformControls;
        this._w = new W();
        this._camera = camera;
        this._object = undefined;
        this._raycaster = new Raycaster();

        // The gizmo: materials shared as three.js shares them, handles in the
        // order `setupGizmo` adds them, groups in the order they are created.
        const materials = [];
        this._materialStates = new Float64Array(W.materialCount() * 2);
        for (let i = 0, n = W.materialCount(); i < n; i++) {
            const d = W.materialInfo(i);
            this._materialStates[i * 2] = d.color;
            this._materialStates[i * 2 + 1] = d.opacity;
            const params = { depthTest: false, depthWrite: false, fog: false, toneMapped: false, transparent: true,
                color: d.color, opacity: d.opacity };
            materials.push(d.line ? new LineBasicMaterial(params) : new MeshBasicMaterial(params));
        }
        this._materials = materials;
        const gizmo = new TransformControlsGizmo();
        this._handles = [];
        for (let i = 0, n = this._w.handleCount(); i < n; i++) {
            const info = this._w.handleInfo(i);
            let group = gizmo[info.group][info.mode];
            if (!group) {
                group = gizmo[info.group][info.mode] = new Object3D();
                if (info.group === 'picker') group.visible = false;
                gizmo.add(group);
            }
            const geometry = _tcGeometry(this._w.handleGeometry(i));
            const object = info.line ? new Line(geometry, materials[info.material]) : new Mesh(geometry, materials[info.material]);
            object.name = info.name;
            object.tag = info.tag;
            object.renderOrder = Infinity;
            group.add(object);
            this._handles.push(object);
        }
        this._gizmo = gizmo;
        this.add(gizmo);
        this._plane = new TransformControlsPlane(_tcGeometry(W.planeGeometry()));
        this.add(this._plane);
        this._handleStates = null;

        // three.js's public working vectors, refreshed from Rust.
        this.worldPosition = new Vector3();
        this.worldPositionStart = new Vector3();
        this.worldQuaternion = new Quaternion();
        this.worldQuaternionStart = new Quaternion();
        this.cameraPosition = new Vector3();
        this.cameraQuaternion = new Quaternion();
        this.pointStart = new Vector3();
        this.pointEnd = new Vector3();
        this.rotationAxis = new Vector3();
        this.eye = new Vector3();

        this._getPointer = (event) => {
            if (this.domElement?.ownerDocument?.pointerLockElement) return { x: 0, y: 0, button: event.button };
            const rect = this.domElement.getBoundingClientRect();
            return {
                x: (event.clientX - rect.left) / rect.width * 2 - 1,
                y: -(event.clientY - rect.top) / rect.height * 2 + 1,
                button: event.button,
            };
        };
        this._onPointerHover = (event) => {
            if (!this.enabled) return;
            if (event.pointerType === 'mouse' || event.pointerType === 'pen') this.pointerHover(this._getPointer(event));
        };
        this._onPointerDown = (event) => {
            if (!this.enabled) return;
            if (!(typeof document !== 'undefined' && document.pointerLockElement)) this.domElement.setPointerCapture?.(event.pointerId);
            this.domElement.addEventListener('pointermove', this._onPointerMove);
            this.pointerHover(this._getPointer(event));
            this.pointerDown(this._getPointer(event));
        };
        this._onPointerMove = (event) => {
            if (!this.enabled) return;
            this.pointerMove(this._getPointer(event));
        };
        this._onPointerUp = (event) => {
            if (!this.enabled) return;
            this.domElement.releasePointerCapture?.(event.pointerId);
            this.domElement.removeEventListener('pointermove', this._onPointerMove);
            this.pointerUp(this._getPointer(event));
        };
        if (domElement?.addEventListener) {
            domElement.addEventListener('pointerdown', this._onPointerDown);
            domElement.addEventListener('pointermove', this._onPointerHover);
            domElement.addEventListener('pointerup', this._onPointerUp);
        }
    }

    get camera() { return this._camera; }
    set camera(v) {
        if (this._camera === v) return;
        this._camera = v;
        this._emitProp('camera', v);
    }
    get object() { return this._object; }
    set object(v) {
        if (this._object === v) return;
        this._object = v;
        this._w?.setHasObject(v !== undefined && v !== null);
        this._emitProp('object', v);
    }

    /// Dispatch `<name>-changed` then `change`, as three.js's setters do.
    _emitProp(name, value) {
        this.dispatchEvent({ type: `${name}-changed`, value });
        this.dispatchEvent({ type: 'change' });
    }
    /// Rust's events, in order: property changes, the object's new pose, the
    /// control's own events.
    _emit(events) {
        const o = this._object;
        for (const e of events) {
            switch (e.type) {
                case 'prop': this._emitProp(e.name, e.value); break;
                case 'position': o?.position.set(e.value[0], e.value[1], e.value[2]); break;
                case 'quaternion': o?.quaternion.set(e.value[0], e.value[1], e.value[2], e.value[3]); break;
                case 'scale': o?.scale.set(e.value[0], e.value[1], e.value[2]); break;
                case 'mouseDown': case 'mouseUp': this.dispatchEvent({ type: e.type, mode: e.mode }); break;
                default: this.dispatchEvent({ type: e.type });
            }
        }
        this._readWorld();
    }
    _readWorld() {
        const s = this._w.worldState();
        this.worldPosition.set(s[0], s[1], s[2]);
        this.worldPositionStart.set(s[3], s[4], s[5]);
        this.worldQuaternion.set(s[6], s[7], s[8], s[9]);
        this.worldQuaternionStart.set(s[10], s[11], s[12], s[13]);
        this.cameraPosition.set(s[14], s[15], s[16]);
        this.cameraQuaternion.set(s[17], s[18], s[19], s[20]);
        this.pointStart.set(s[21], s[22], s[23]);
        this.pointEnd.set(s[24], s[25], s[26]);
        this.rotationAxis.set(s[27], s[28], s[29]);
        this.eye.set(s[30], s[31], s[32]);
    }
    /// `[position, quaternion, scale, matrixWorld, hasParent, parent.matrixWorld]`.
    _objectState() {
        const o = this._object;
        if (o === undefined || o === null) return undefined;
        const a = new Float64Array(43);
        const p = o.position, q = o.quaternion, s = o.scale;
        a[0] = p.x; a[1] = p.y; a[2] = p.z;
        a[3] = q.x; a[4] = q.y; a[5] = q.z; a[6] = q.w;
        a[7] = s.x; a[8] = s.y; a[9] = s.z;
        a.set(o.matrixWorld.elements, 10);
        if (o.parent) { a[26] = 1; a.set(o.parent.matrixWorld.elements, 27); }
        return a;
    }
    /// `[matrixWorld, projectionMatrixInverse, ortho, near, far, fov, zoom, top, bottom]`.
    _cameraState() {
        const c = this._camera;
        const a = new Float64Array(39);
        if (!c) return a;
        a.set(c.matrixWorld.elements, 0);
        a.set(c.projectionMatrixInverse.elements, 16);
        a[32] = c.isOrthographicCamera ? 1 : 0;
        a[33] = c.near; a[34] = c.far; a[35] = c.fov ?? 50; a[36] = c.zoom ?? 1;
        a[37] = c.top ?? 1; a[38] = c.bottom ?? -1;
        return a;
    }

    updateMatrixWorld(force) {
        if (this._w && this._camera) {
            const object = this._object;
            if (object !== undefined && object !== null) {
                object.updateMatrixWorld();
                if (object.parent === null) console.error('TransformControls: The attached 3D object must be a part of the scene graph.');
            }
            this._camera.updateMatrixWorld();
            if (this.matrixAutoUpdate) this.updateMatrix();
            const world = this.parent ? _tcMatrix.multiplyMatrices(this.parent.matrixWorld, this.matrix) : this.matrix;
            this._w.updateMatrixWorld(this._objectState(), this._cameraState(), Float64Array.from(world.elements));
            this._readWorld();
            this._applyGizmo();
        }
        super.updateMatrixWorld(force);
    }
    /// The handles' and materials' state from the last update onto the
    /// objects (only what changed), and the plane's pose.
    _applyGizmo() {
        // Only the current mode's gizmo and helper show.
        const mode = this.mode, g = this._gizmo;
        for (const m of ['translate', 'rotate', 'scale']) {
            if (g.gizmo[m]) g.gizmo[m].visible = mode === m;
            if (g.helper[m]) g.helper[m].visible = mode === m;
        }
        const s = this._w.handleStates();
        const last = this._handleStates;
        for (let i = 0, n = this._handles.length; i < n; i++) {
            const k = i * 11;
            if (last) {
                let same = true;
                for (let j = 0; j < 11; j++) if (s[k + j] !== last[k + j]) { same = false; break; }
                if (same) continue;
            }
            const h = this._handles[i];
            h.position.set(s[k], s[k + 1], s[k + 2]);
            h.quaternion.set(s[k + 3], s[k + 4], s[k + 5], s[k + 6]);
            h.scale.set(s[k + 7], s[k + 8], s[k + 9]);
            h.visible = s[k + 10] !== 0;
        }
        this._handleStates = s;
        const m = this._w.materialStates();
        const lastM = this._materialStates;
        for (let i = 0, n = this._materials.length; i < n; i++) {
            if (lastM[i * 2] === m[i * 2] && lastM[i * 2 + 1] === m[i * 2 + 1]) continue;
            const mat = this._materials[i];
            mat.color?.setHex(m[i * 2]);
            mat.opacity = m[i * 2 + 1];
            mat._w?.setOpacity?.(mat.opacity);
            _materialChanged(mat);
        }
        this._materialStates = m;
        const p = this._w.planeTransform();
        this._plane.position.set(p[0], p[1], p[2]);
        this._plane.quaternion.set(p[3], p[4], p[5], p[6]);
    }

    pointerHover(pointer) {
        if (!this._w.wantsHover()) return;
        if (pointer !== null) this._raycaster.setFromCamera(pointer, this._camera);
        const picker = this._gizmo.picker[this.mode];
        let name;
        if (picker) {
            for (const hit of this._raycaster.intersectObject(picker, true)) {
                if (hit.object.visible) { name = hit.object.name; break; }
            }
        }
        this._emit(this._w.pointerHover(name));
    }
    pointerDown(pointer) {
        const o = this._object;
        if (o !== undefined && o !== null && !this.dragging && (pointer == null || pointer.button === 0) && this.axis !== null) {
            if (pointer !== null) this._raycaster.setFromCamera(pointer, this._camera);
            o.updateMatrixWorld();
            o.parent?.updateMatrixWorld();
        }
        this._emit(this._w.pointerDown(pointer != null, pointer?.x ?? 0, pointer?.y ?? 0, pointer?.button ?? 0,
            this._objectState(), this._cameraState()));
    }
    pointerMove(pointer) {
        if (pointer !== null && this.dragging && this.axis !== null && pointer.button === -1) {
            this._raycaster.setFromCamera(pointer, this._camera);
        }
        this._emit(this._w.pointerMove(pointer != null, pointer?.x ?? 0, pointer?.y ?? 0, pointer?.button ?? -1,
            this._objectState(), this._cameraState()));
    }
    pointerUp(pointer) {
        this._emit(this._w.pointerUp(pointer != null, pointer?.button ?? 0));
    }
    dispose() {
        const d = this.domElement;
        if (d?.removeEventListener) {
            d.removeEventListener('pointerdown', this._onPointerDown);
            d.removeEventListener('pointermove', this._onPointerHover);
            d.removeEventListener('pointermove', this._onPointerMove);
            d.removeEventListener('pointerup', this._onPointerUp);
        }
        this.traverse((child) => {
            child.geometry?.dispose?.();
            child.material?.dispose?.();
        });
    }
    attach(object) {
        this.object = object;
        this.visible = true;
        return this;
    }
    detach() {
        this.object = undefined;
        this.visible = false;
        this.axis = null;
        return this;
    }
    reset() {
        if (!this.enabled) return;
        this._emit(this._w.reset());
    }
    getRaycaster() { return this._raycaster; }
    /// three.js r169+ splits the gizmo off as `getHelper()`; in r165 the
    /// control is the scene object itself, so both spellings work.
    getHelper() { return this; }
    getMode() { return this.mode; }
    setMode(mode) { this.mode = mode; }
    setTranslationSnap(translationSnap) { this.translationSnap = translationSnap; }
    setRotationSnap(rotationSnap) { this.rotationSnap = rotationSnap; }
    setScaleSnap(scaleSnap) { this.scaleSnap = scaleSnap; }
    setSize(size) { this.size = size; }
    setSpace(space) { this.space = space; }
}
// The properties three.js defines with `defineProperty`: held in Rust, a
// `<name>-changed` and a `change` event when one changes.
for (const name of _tcRustProps) {
    Object.defineProperty(TransformControls.prototype, name, {
        get() { return this._w ? this._w.getProperty(name) : undefined; },
        set(v) { if (this._w) this._emit(this._w.setProperty(name, v)); },
        configurable: true,
    });
}
export { TransformControlsGizmo, TransformControlsPlane };

// Render targets — a stub class so user code that calls
// `renderer.setRenderTarget(rt); renderer.render(...); renderer.setRenderTarget(null);`
// doesn't throw. Future: route to a wgpu off-screen Texture.
export class WebGLRenderTarget {
    constructor(width = 1, height = 1, opts = {}) {
        this.width = width; this.height = height;
        this.type = opts.type ?? UnsignedByteType;
        this._halfFloat = this.type === HalfFloatType;
        this.texture = new Texture();
        // three.js: `options.colorSpace` is the texture's (default none).
        if (opts.colorSpace !== undefined) this.texture.colorSpace = opts.colorSpace;
        this.depthBuffer = opts.depthBuffer !== false;
        this.stencilBuffer = opts.stencilBuffer ?? false;
        // three.js: MSAA for this target, whatever the renderer's `antialias`.
        this.samples = opts.samples ?? 0;
        // _w is the wgpu-backed RenderTarget. It's created lazily on first
        // setRenderTarget call because we need the renderer's device to
        // allocate the matching wgpu texture format.
        this._w = null;
    }
    // three.js reallocates at the new size; the wasm target keeps its id, so
    // `texture` — and every material sampling it — follows.
    setSize(w, h) {
        this.width = w;
        this.height = h;
        this._w?.setSize(Math.max(1, w | 0), Math.max(1, h | 0));
    }
    clone() {
        return new WebGLRenderTarget(this.width, this.height, {
            depthBuffer: this.depthBuffer,
            stencilBuffer: this.stencilBuffer,
            type: this.type,
            colorSpace: this.texture?.colorSpace,
            samples: this.samples,
        });
    }
    dispose() { this._w = null; }
}
export class WebGLCubeRenderTarget extends WebGLRenderTarget {
    constructor(size = 256, opts = {}) {
        super(size, size, opts);
        this.isWebGLCubeRenderTarget = true;
        // Will be lazy-allocated on first CubeCamera.update via the renderer.
        this._w_cube = null;
        this._cubeSide = size;
        // The .texture is a CubeTexture sampled from this RT. We construct it
        // with a 1×1 white placeholder; the real cube view binds at sample time.
        const w = new Uint8Array([255, 255, 255, 255]);
        this.texture = new CubeTexture(1, w, w, w, w, w, w);
    }
}
export class WebGL3DRenderTarget extends WebGLRenderTarget {}

// ---- Pass 8: animation tracks + curves + misc ----

// Side constants re-exported at top of module (see FrontSide/BackSide/DoubleSide above).

// ---- Pass 9: constants + MathUtils ----
export const MathUtils = {
    DEG2RAD: Math.PI / 180, RAD2DEG: 180 / Math.PI,
    generateUUID: (() => {
        const hex = []; for (let i = 0; i < 256; i++) hex[i] = (i < 16 ? '0' : '') + i.toString(16);
        return function () {
            const a = Math.random() * 0xffffffff | 0, b = Math.random() * 0xffffffff | 0,
                  c = Math.random() * 0xffffffff | 0, d = Math.random() * 0xffffffff | 0;
            return (hex[a & 0xff] + hex[(a >> 8) & 0xff] + hex[(a >> 16) & 0xff] + hex[(a >> 24) & 0xff] + '-' +
                    hex[b & 0xff] + hex[(b >> 8) & 0xff] + '-' +
                    hex[((b >> 16) & 0x0f) | 0x40] + hex[(b >> 24) & 0xff] + '-' +
                    hex[(c & 0x3f) | 0x80] + hex[(c >> 8) & 0xff] + '-' +
                    hex[(c >> 16) & 0xff] + hex[(c >> 24) & 0xff] + hex[d & 0xff] + hex[(d >> 8) & 0xff] + hex[(d >> 16) & 0xff] + hex[(d >> 24) & 0xff]).toUpperCase();
        };
    })(),
    clamp: (v, mn, mx) => Math.max(mn, Math.min(mx, v)),
    euclideanModulo: (n, m) => ((n % m) + m) % m,
    mapLinear: (x, a1, a2, b1, b2) => b1 + (x - a1) * (b2 - b1) / (a2 - a1),
    inverseLerp: (x, y, v) => x !== y ? (v - x) / (y - x) : 0,
    lerp: (x, y, t) => (1 - t) * x + t * y,
    damp: (x, y, lambda, dt) => MathUtils.lerp(x, y, 1 - Math.exp(-lambda * dt)),
    pingpong: (x, length = 1) => length - Math.abs(MathUtils.euclideanModulo(x, length * 2) - length),
    smoothstep: (x, mn, mx) => { if (x <= mn) return 0; if (x >= mx) return 1; x = (x - mn) / (mx - mn); return x * x * (3 - 2 * x); },
    smootherstep: (x, mn, mx) => { if (x <= mn) return 0; if (x >= mx) return 1; x = (x - mn) / (mx - mn); return x * x * x * (x * (x * 6 - 15) + 10); },
    randInt: (low, high) => low + Math.floor(Math.random() * (high - low + 1)),
    randFloat: (low, high) => low + Math.random() * (high - low),
    randFloatSpread: (range) => range * (0.5 - Math.random()),
    seededRandom: (s) => { s = s ?? Math.random() * 1e9; let t = s + 0x6D2B79F5; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return (((t ^ (t >>> 14)) >>> 0) / 4294967296); },
    degToRad: (d) => d * (Math.PI / 180),
    radToDeg: (r) => r * (180 / Math.PI),
    isPowerOfTwo: (v) => (v & (v - 1)) === 0 && v !== 0,
    ceilPowerOfTwo: (v) => Math.pow(2, Math.ceil(Math.log(v) / Math.LN2)),
    floorPowerOfTwo: (v) => Math.pow(2, Math.floor(Math.log(v) / Math.LN2)),
    setQuaternionFromProperEuler(q, a, b, c, order) { q.setFromEuler(new Euler(a, b, c, order)); },
    normalize: _normalize,
    denormalize: _denormalize,
};
export const REVISION = '165';
// ColorManagement: three.js r165's, defined with Color near the top.

// Object3D (defined with the scene graph, above) dispatches three.js's events.
Object.assign(Object3D.prototype, EventDispatcherMixin);
// OrbitControls (and MapControls) dispatch `start` / `change` / `end`.
Object.assign(OrbitControls.prototype, EventDispatcherMixin);

// ---- Stubs for the remaining three.js API surface ----
// Base classes (used by `instanceof` checks in user code).
export class Material {
    constructor() {
        this.type = 'Material';
        this.uuid = MathUtils.generateUUID();
        this.opacity = 1; this.transparent = false; this.side = 0;
        this.visible = true; this.uniforms = {};
        this.userData = {};
        this.alphaTest = 0; this.alphaToCoverage = false;
        this.blending = 1; // NormalBlending
        this.depthTest = true; this.depthWrite = true;
        this.colorWrite = true; this.toneMapped = true;
        this.wireframe = false;
        this.needsUpdate = false;
        this.version = 0;
        // onBeforeCompile is called once per shader build by three.js. Our wgpu
        // pipeline doesn't recompile shaders from GLSL strings, but we still
        // invoke the callback with a fake `shader` object so user code that
        // sets `shader.uniforms.foo` for later reading works correctly.
        this.onBeforeCompile = null;
    }
    clone() { return new this.constructor().copy(this); }
    copy(other) {
        for (const k of ['opacity','transparent','side','visible','alphaTest','blending','depthTest','depthWrite','colorWrite','toneMapped','wireframe']) {
            if (other[k] !== undefined) this[k] = other[k];
        }
        this.uniforms = { ...other.uniforms };
        this.userData = JSON.parse(JSON.stringify(other.userData || {}));
        return this;
    }
    dispose() {}
    setValues(v) { Object.assign(this, v); return this; }
    customProgramCacheKey() { return this.uuid; }
    // Run user's onBeforeCompile if set, with a shader-shaped object they
    // can mutate. Their string-replacements on vertexShader/fragmentShader
    // are captured but don't recompile our WGSL pipeline; their uniform
    // additions ARE captured on `material.userData.shader` for later read.
    _runOnBeforeCompile() {
        if (typeof this.onBeforeCompile !== 'function' || this._compiledOnce) return;
        const shader = {
            uniforms: { ...this.uniforms },
            vertexShader: '/* threers wgsl pipeline */',
            fragmentShader: '/* threers wgsl pipeline */',
            defines: {},
        };
        try { this.onBeforeCompile(shader, this); } catch (_e) {}
        this.userData.shader = shader;
        this.uniforms = shader.uniforms;
        this._compiledOnce = true;
    }
}
export class LightShadow { constructor(camera) { this.camera = camera; this.bias = 0; this.normalBias = 0; this.radius = 1; this.blurSamples = 8; this.mapSize = new Vector2(512, 512); this.map = null; this.matrix = new Matrix4(); } }
export class LightProbe extends Light { constructor() { super(); this.sh = new SphericalHarmonics3(); } }

// Cameras.
export class ArrayCamera extends PerspectiveCamera { constructor(cameras = []) { super(); this.cameras = cameras; this.isArrayCamera = true; } }
export class StereoCamera {
    constructor() {
        this.type = 'StereoCamera';
        this.aspect = 1;
        this.eyeSep = 0.064;
        this.cameraL = new PerspectiveCamera();
        this.cameraR = new PerspectiveCamera();
    }
    /** three.js StereoCamera.update(camera) — builds left/right from a master cam. */
    update(camera) {
        if (!camera) return;
        if (typeof camera._sync === 'function') camera._sync();
        camera.updateWorldMatrix?.(true, false);
        const eye = new Vector3().setFromMatrixPosition(camera.matrixWorld);
        const forward = camera.getWorldDirection(new Vector3());
        const target = eye.clone().add(forward);
        const up = new Vector3(0, 1, 0).applyQuaternion(camera.getWorldQuaternion(new Quaternion()));
        let right = new Vector3().crossVectors(forward, up);
        if (right.lengthSq() < 1e-10) right.set(1, 0, 0);
        else right.normalize();
        const half = this.eyeSep * 0.5;
        const apply = (cam, sign) => {
            cam.fov = camera.fov ?? camera._fov ?? 50;
            cam.aspect = (camera.aspect || 1) * 0.5 * this.aspect;
            cam.near = camera.near;
            cam.far = camera.far;
            cam.position.set(
                eye.x + right.x * half * sign,
                eye.y + right.y * half * sign,
                eye.z + right.z * half * sign,
            );
            cam.lookAt(target.x, target.y, target.z);
            cam.updateProjectionMatrix?.();
            cam._sync?.();
        };
        apply(this.cameraL, -1);
        apply(this.cameraR, +1);
    }
}
export class CubeCamera extends Object3D {
    constructor(near = 0.1, far = 1000, renderTarget = null) {
        super();
        this.near = near; this.far = far;
        this.renderTarget = renderTarget;
        this.coordinateSystem = 2000;
        // 6 perspective cameras pointing in the cube face directions.
        const mk = (lookAt, up) => {
            const cam = new PerspectiveCamera(90, 1, near, far);
            cam._lookAt = new Vector3(lookAt[0], lookAt[1], lookAt[2]);
            cam.up = new Vector3(up[0], up[1], up[2]);
            return cam;
        };
        this.cameraPX = mk([ 1, 0, 0], [0,-1, 0]);
        this.cameraNX = mk([-1, 0, 0], [0,-1, 0]);
        this.cameraPY = mk([ 0, 1, 0], [0, 0, 1]);
        this.cameraNY = mk([ 0,-1, 0], [0, 0,-1]);
        this.cameraPZ = mk([ 0, 0, 1], [0,-1, 0]);
        this.cameraNZ = mk([ 0, 0,-1], [0,-1, 0]);
    }
    update(renderer, scene) {
        const rt = this.renderTarget;
        if (!rt) return;
        if (!rt._w_cube) {
            rt._w_cube = new WebCubeRenderTarget(renderer._w, rt._cubeSide);
            // Tag the texture so `scene.environment = rt.texture` finds the cube RT id.
            rt.texture._cubeRTId = rt._w_cube.id;
        }
        const cams = [this.cameraPX, this.cameraNX, this.cameraPY, this.cameraNY, this.cameraPZ, this.cameraNZ];
        // Rendered from the cube camera's world position, as three.js's are.
        this.updateWorldMatrix(true, false);
        const p = new Vector3().setFromMatrixPosition(this.matrixWorld);
        for (let i = 0; i < 6; i++) {
            const cam = cams[i];
            cam.position.copy(p);
            const t = new Vector3(p.x + cam._lookAt.x, p.y + cam._lookAt.y, p.z + cam._lookAt.z);
            cam._lookAtTarget = t;
            cam._w.setPosition(p.x, p.y, p.z);
            // Push the face camera's specific up vector before lookAt; +Y/-Y
            // views require non-(0,1,0) ups or the view matrix is degenerate.
            cam._w.setUp(cam.up.x, cam.up.y, cam.up.z);
            cam._w.lookAt(t.x, t.y, t.z);
            scene._syncTransforms(cam);
            renderer._w.renderToCubeFace(scene._w, cam._w, rt._w_cube, i);
        }
    }
}

// Material variants (stubs that map onto closest existing material).
export class LineDashedMaterial extends LineBasicMaterial {
    constructor(opts = {}) {
        super(opts);
        this.dashSize = opts.dashSize ?? 3;
        this.gapSize = opts.gapSize ?? 1;
        this.scale = opts.scale ?? 1;
        this._isLineDashed = true;
        this._w = WebMaterial.lineDashed(_matColor(opts), this.scale, this.dashSize, this.gapSize);
    }
}
export class MeshMatcapMaterial {
    constructor(opts = {}) {
        this._w = WebMaterial.matcap(_matColor(opts));
        _applyCommon(this._w, opts);
        _bindMapSlot(this, 'matcap', 'setMatcap', 'setMatcapData', opts.matcap);
        _bindMapSlot(this, 'map', 'setMap', 'setMapData', opts.map);
        _bindMapSlot(this, 'alphaMap', 'setAlphaMap', 'setAlphaMapData', opts.alphaMap);
        _initMaterialBase(this, opts);
    }
}
export class MeshDistanceMaterial {
    constructor(opts = {}) {
        this.referencePosition = opts?.referencePosition || new Vector3();
        this.nearDistance = opts?.nearDistance ?? 1;
        this.farDistance = opts?.farDistance ?? 1000;
        this._w = WebMaterial.distance(
            this.referencePosition.x, this.referencePosition.y, this.referencePosition.z,
            this.nearDistance, this.farDistance
        );
        _initMaterialBase(this, opts);
    }
}
export class ShadowMaterial {
    constructor(opts = {}) {
        this._w = WebMaterial.shadow(opts?.opacity ?? 1);
        this.transparent = true;
        this.opacity = opts?.opacity ?? 1;
        _initMaterialBase(this, { ...opts, color: 0x000000 });
    }
}

// Textures + variants.
export class DepthTexture extends Texture { constructor(w, h) { super(w, h, new Uint8Array(w * h * 4)); this.isDepthTexture = true; } }
export class VideoTexture extends Texture {
    constructor(video) {
        super();
        this.image = video;
        this.isVideoTexture = true;
        // Frames are drawn through a canvas, top row first — see CanvasTexture.
        this.flipY = true;
        this._topRowFirst = true;
        this._canvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;
        this._ctx = this._canvas ? this._canvas.getContext('2d') : null;
        this.needsUpdate = false;
    }
    _syncUv() {
        if (_syncTextureUv(this)) _rebindTexture(this);
    }
    update() {
        const video = this.image;
        const canvas = this._canvas;
        const ctx = this._ctx;
        if (!video || !canvas || !ctx) return;
        const w = video.videoWidth || 1280;
        const h = video.videoHeight || 720;
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== h) canvas.height = h;
        if (video.readyState >= 2) {
            try { ctx.drawImage(video, 0, 0, w, h); } catch (_err) { /* seek */ }
        } else {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, w, h);
        }
        const data = ctx.getImageData(0, 0, w, h).data;
        const byteLen = data.length;
        if (!this._buf || this._buf.length !== byteLen) {
            this._buf = new Uint8Array(byteLen);
        }
        this._buf.set(data);
        if (this._w && this._lw === w && this._lh === h && this._w.replaceRgba) {
            if (this._w.replaceRgba(this._buf)) {
                this.needsUpdate = true;
                return;
            }
        }
        this._w = new WebDataTexture(w, h, this._buf);
        this._lw = w;
        this._lh = h;
        this._syncUv();
        this.needsUpdate = true;
    }
}
export class CompressedTexture extends Texture { constructor(mipmaps = [], w = 1, h = 1) { super(); this.mipmaps = mipmaps; this.image = { width: w, height: h }; this.isCompressedTexture = true; } }
export class CompressedArrayTexture extends CompressedTexture {}
export class CompressedCubeTexture extends CompressedTexture {}
/// three.js `Data3DTexture`: voxels, x fastest, then y, then z. Bound to a
/// `ShaderMaterial` (`volumes`, `setVolume`) as `u_vol0` / `u_vol1`, a WGSL
/// `texture_3d<f32>` sampled with `u_vol_samp`.
///
/// Not a `DataTexture`: it used to be one, which handed `w·h·d` one-channel
/// voxels to the 2-D path as a `w×h` RGBA image. Defaults are three.js's:
/// `RedFormat`, `UnsignedByteType`, nearest filtering, clamped in all three
/// axes; `needsUpdate = true` re-uploads (it bumps `version`).
export class Data3DTexture {
    constructor(data = null, width = 1, height = 1, depth = 1) {
        this.isTexture = true;
        this.isData3DTexture = true;
        this.uuid = MathUtils.generateUUID();
        this.name = '';
        this.image = { data, width, height, depth };
        this.format = 1028; // RedFormat
        this.type = data instanceof Float32Array ? 1015 : 1009; // FloatType : UnsignedByteType
        this.magFilter = 1003; this.minFilter = 1003; // NearestFilter
        this.wrapS = 1001; this.wrapT = 1001; this.wrapR = 1001; // ClampToEdgeWrapping
        this.generateMipmaps = false;
        this.flipY = false;
        this.unpackAlignment = 1;
        this.userData = {};
        this.version = 0;
    }
    set needsUpdate(v) { if (v) this.version++; }
    get needsUpdate() { return false; }
    clone() {
        const t = new Data3DTexture(this.image.data, this.image.width, this.image.height, this.image.depth);
        Object.assign(t, { format: this.format, type: this.type, magFilter: this.magFilter, minFilter: this.minFilter });
        return t;
    }
    dispose() {}
}
export class DataArrayTexture extends DataTexture {}
export class FramebufferTexture extends Texture { constructor(w, h) { super(w, h, new Uint8Array(w * h * 4)); this.isFramebufferTexture = true; } }
export class Source { constructor(data = null) { this.data = data; this.uuid = String(Math.random()).slice(2); } toJSON() { return { uuid: this.uuid }; } }

// Buffer-attribute typed subclasses (all behave the same — just enforce a typed array on creation).
function _typedBufferAttr(Ctor) { return class extends BufferAttribute { constructor(arr, itemSize, normalized = false) { super(arr instanceof Ctor ? arr : new Ctor(arr), itemSize); this.normalized = normalized; } }; }
export class Int8BufferAttribute    extends _typedBufferAttr(Int8Array)    {}
export class Int16BufferAttribute   extends _typedBufferAttr(Int16Array)   {}
export class Int32BufferAttribute   extends _typedBufferAttr(Int32Array)   {}
export class Uint8BufferAttribute   extends _typedBufferAttr(Uint8Array)   {}
export class Uint8ClampedBufferAttribute extends _typedBufferAttr(Uint8ClampedArray) {}
export class Uint16BufferAttribute  extends _typedBufferAttr(Uint16Array)  {}
export class Uint32BufferAttribute  extends _typedBufferAttr(Uint32Array)  {}
export class Float16BufferAttribute extends _typedBufferAttr(Uint16Array)  {}
export class Float32BufferAttribute extends _typedBufferAttr(Float32Array) {}
export class Float64BufferAttribute extends _typedBufferAttr(Float64Array) {}
export class InstancedBufferAttribute extends BufferAttribute { constructor(arr, itemSize, normalized = false, meshPerAttribute = 1) { super(arr, itemSize); this.meshPerAttribute = meshPerAttribute; } }
// three.js r165 `MathUtils.denormalize` / `normalize`: an integer component
// of a `normalized` attribute to its float value, and back.
function _denormalize(value, array) {
    switch (array.constructor) {
        case Float32Array: return value;
        case Uint32Array: return value / 4294967295.0;
        case Uint16Array: return value / 65535.0;
        case Uint8Array: return value / 255.0;
        case Int32Array: return Math.max(value / 2147483647.0, -1.0);
        case Int16Array: return Math.max(value / 32767.0, -1.0);
        case Int8Array: return Math.max(value / 127.0, -1.0);
        default: throw new Error('Invalid component type.');
    }
}
function _normalize(value, array) {
    switch (array.constructor) {
        case Float32Array: return value;
        case Uint32Array: return Math.round(value * 4294967295.0);
        case Uint16Array: return Math.round(value * 65535.0);
        case Uint8Array: return Math.round(value * 255.0);
        case Int32Array: return Math.round(value * 2147483647.0);
        case Int16Array: return Math.round(value * 32767.0);
        case Int8Array: return Math.round(value * 127.0);
        default: throw new Error('Invalid component type.');
    }
}
/// three.js r165 `InterleavedBuffer`: one array holding several attributes,
/// `stride` elements a vertex. The renderer's vertex buffers are its own
/// packing (see `GpuMesh`), so this is de-interleaved where geometry crosses
/// into Rust (`_attrPacked`); `needsUpdate` re-syncs every geometry drawing
/// any attribute of the buffer.
export class InterleavedBuffer {
    constructor(array, stride) {
        this.isInterleavedBuffer = true;
        this.array = array;
        this.stride = stride;
        this.count = array !== undefined ? array.length / stride : 0;
        this.usage = 35044; // StaticDrawUsage
        this._updateRange = { offset: 0, count: -1 };
        this.updateRanges = [];
        this.version = 0;
        this.uuid = MathUtils.generateUUID();
        /// The geometries (and instanced meshes) drawing an attribute of this
        /// buffer, told when it is flagged `needsUpdate`.
        this._owners = new Set();
    }
    onUploadCallback() {}
    set needsUpdate(value) {
        if (value !== true) return;
        this.version++;
        for (const owner of this._owners) owner._attributeChanged?.(this);
    }
    get updateRange() { return this._updateRange; }
    setUsage(value) { this.usage = value; return this; }
    addUpdateRange(start, count) { this.updateRanges.push({ start, count }); }
    clearUpdateRanges() { this.updateRanges.length = 0; }
    copy(source) {
        this.array = new source.array.constructor(source.array);
        this.count = source.count;
        this.stride = source.stride;
        this.usage = source.usage;
        return this;
    }
    copyAt(index1, attribute, index2) {
        index1 *= this.stride;
        index2 *= attribute.stride;
        for (let i = 0, l = this.stride; i < l; i++) this.array[index1 + i] = attribute.array[index2 + i];
        return this;
    }
    set(value, offset = 0) { this.array.set(value, offset); return this; }
    clone(data) {
        if (data.arrayBuffers === undefined) data.arrayBuffers = {};
        if (this.array.buffer._uuid === undefined) this.array.buffer._uuid = MathUtils.generateUUID();
        if (data.arrayBuffers[this.array.buffer._uuid] === undefined) {
            data.arrayBuffers[this.array.buffer._uuid] = this.array.slice(0).buffer;
        }
        const array = new this.array.constructor(data.arrayBuffers[this.array.buffer._uuid]);
        const ib = new this.constructor(array, this.stride);
        ib.setUsage(this.usage);
        return ib;
    }
    onUpload(callback) { this.onUploadCallback = callback; return this; }
    toJSON(data) {
        if (data.arrayBuffers === undefined) data.arrayBuffers = {};
        if (this.array.buffer._uuid === undefined) this.array.buffer._uuid = MathUtils.generateUUID();
        if (data.arrayBuffers[this.array.buffer._uuid] === undefined) {
            data.arrayBuffers[this.array.buffer._uuid] = Array.from(new Uint32Array(this.array.buffer));
        }
        return { uuid: this.uuid, buffer: this.array.buffer._uuid, type: this.array.constructor.name, stride: this.stride };
    }
}
const _ibaVector = new Vector3();
/// three.js r165 `InterleavedBufferAttribute`: `itemSize` components at
/// `offset` within each `stride` of `data`.
export class InterleavedBufferAttribute {
    constructor(interleavedBuffer, itemSize, offset, normalized = false) {
        this.isInterleavedBufferAttribute = true;
        this.name = '';
        this.data = interleavedBuffer;
        this.itemSize = itemSize;
        this.offset = offset;
        this.normalized = normalized;
    }
    get count() { return this.data.count; }
    get array() { return this.data.array; }
    set needsUpdate(value) { this.data.needsUpdate = value; }
    /// Who to tell on `needsUpdate`: the buffer's, since that is shared.
    get _owners() { return this.data._owners; }
    applyMatrix4(m) {
        for (let i = 0, l = this.data.count; i < l; i++) {
            _ibaVector.fromBufferAttribute(this, i);
            _ibaVector.applyMatrix4(m);
            this.setXYZ(i, _ibaVector.x, _ibaVector.y, _ibaVector.z);
        }
        return this;
    }
    applyNormalMatrix(m) {
        for (let i = 0, l = this.count; i < l; i++) {
            _ibaVector.fromBufferAttribute(this, i);
            _ibaVector.applyNormalMatrix(m);
            this.setXYZ(i, _ibaVector.x, _ibaVector.y, _ibaVector.z);
        }
        return this;
    }
    transformDirection(m) {
        for (let i = 0, l = this.count; i < l; i++) {
            _ibaVector.fromBufferAttribute(this, i);
            _ibaVector.transformDirection(m);
            this.setXYZ(i, _ibaVector.x, _ibaVector.y, _ibaVector.z);
        }
        return this;
    }
    getComponent(index, component) {
        let value = this.array[index * this.data.stride + this.offset + component];
        if (this.normalized) value = _denormalize(value, this.array);
        return value;
    }
    setComponent(index, component, value) {
        if (this.normalized) value = _normalize(value, this.array);
        this.data.array[index * this.data.stride + this.offset + component] = value;
        return this;
    }
    setX(index, x) { return this.setComponent(index, 0, x); }
    setY(index, y) { return this.setComponent(index, 1, y); }
    setZ(index, z) { return this.setComponent(index, 2, z); }
    setW(index, w) { return this.setComponent(index, 3, w); }
    getX(index) { return this.getComponent(index, 0); }
    getY(index) { return this.getComponent(index, 1); }
    getZ(index) { return this.getComponent(index, 2); }
    getW(index) { return this.getComponent(index, 3); }
    setXY(index, x, y) {
        index = index * this.data.stride + this.offset;
        if (this.normalized) { x = _normalize(x, this.array); y = _normalize(y, this.array); }
        this.data.array[index + 0] = x;
        this.data.array[index + 1] = y;
        return this;
    }
    setXYZ(index, x, y, z) {
        index = index * this.data.stride + this.offset;
        if (this.normalized) { x = _normalize(x, this.array); y = _normalize(y, this.array); z = _normalize(z, this.array); }
        this.data.array[index + 0] = x;
        this.data.array[index + 1] = y;
        this.data.array[index + 2] = z;
        return this;
    }
    setXYZW(index, x, y, z, w) {
        index = index * this.data.stride + this.offset;
        if (this.normalized) {
            x = _normalize(x, this.array); y = _normalize(y, this.array);
            z = _normalize(z, this.array); w = _normalize(w, this.array);
        }
        this.data.array[index + 0] = x;
        this.data.array[index + 1] = y;
        this.data.array[index + 2] = z;
        this.data.array[index + 3] = w;
        return this;
    }
    /// The raw (still normalized-integer) components, packed `itemSize` a vertex.
    _deinterleaved() {
        const array = [];
        for (let i = 0; i < this.count; i++) {
            const index = i * this.data.stride + this.offset;
            for (let j = 0; j < this.itemSize; j++) array.push(this.data.array[index + j]);
        }
        return array;
    }
    clone(data) {
        if (data === undefined) {
            const out = new BufferAttribute(new this.array.constructor(this._deinterleaved()), this.itemSize, this.normalized);
            out.normalized = this.normalized;
            return out;
        }
        if (data.interleavedBuffers === undefined) data.interleavedBuffers = {};
        if (data.interleavedBuffers[this.data.uuid] === undefined) {
            data.interleavedBuffers[this.data.uuid] = this.data.clone(data);
        }
        return new InterleavedBufferAttribute(data.interleavedBuffers[this.data.uuid], this.itemSize, this.offset, this.normalized);
    }
    toJSON(data) {
        if (data === undefined) {
            return { itemSize: this.itemSize, type: this.array.constructor.name, array: this._deinterleaved(), normalized: this.normalized };
        }
        if (data.interleavedBuffers === undefined) data.interleavedBuffers = {};
        if (data.interleavedBuffers[this.data.uuid] === undefined) {
            data.interleavedBuffers[this.data.uuid] = this.data.toJSON(data);
        }
        return { isInterleavedBufferAttribute: true, itemSize: this.itemSize, data: this.data.uuid, offset: this.offset, normalized: this.normalized };
    }
}
export class InstancedInterleavedBuffer extends InterleavedBuffer {
    constructor(array, stride, meshPerAttribute = 1) {
        super(array, stride);
        this.isInstancedInterleavedBuffer = true;
        this.meshPerAttribute = meshPerAttribute;
    }
    copy(source) { super.copy(source); this.meshPerAttribute = source.meshPerAttribute; return this; }
    clone(data) { const ib = super.clone(data); ib.meshPerAttribute = this.meshPerAttribute; return ib; }
    toJSON(data) {
        const json = super.toJSON(data);
        json.isInstancedInterleavedBuffer = true;
        json.meshPerAttribute = this.meshPerAttribute;
        return json;
    }
}
/// An attribute's components packed `itemSize` a vertex, as the Rust
/// geometry takes them: the attribute's own array, or for an
/// `InterleavedBufferAttribute` its components gathered out of the shared
/// buffer — converted to float as the GPU's vertex fetch would
/// (`normalized` integers to [0, 1] / [-1, 1]). Plain attributes are returned
/// as they are; their arrays are already packed.
function _attrPacked(attr) {
    if (!attr?.isInterleavedBufferAttribute) return attr?.array;
    const { data, itemSize, offset } = attr;
    const src = data.array, stride = data.stride, n = Math.floor(data.count);
    const out = new Float32Array(n * itemSize);
    for (let i = 0; i < n; i++) {
        const b = i * stride + offset;
        for (let j = 0; j < itemSize; j++) {
            const v = src[b + j];
            out[i * itemSize + j] = attr.normalized ? _denormalize(v, src) : v;
        }
    }
    return out;
}
export class GLBufferAttribute { constructor(buf, type, itemSize, elementSize, count) { this.buffer = buf; this.type = type; this.itemSize = itemSize; this.elementSize = elementSize; this.count = count; } }
export class InstancedBufferGeometry extends BufferGeometry { constructor() { super(); this.instanceCount = 0; } }

// Lights.
export class AmbientLightProbe extends LightProbe { constructor(color = 0xffffff, intensity = 1) { super(); this.color = new Color(color); this.intensity = intensity; this._isLight = true; this._w = WebLight.ambient(_color(color), intensity); } }
export class HemisphereLightProbe extends LightProbe { constructor(sky = 0xffffff, ground = 0xffffff, intensity = 1) { super(); this.color = new Color(sky); this.groundColor = new Color(ground); this.intensity = intensity; this._isLight = true; this._w = WebLight.hemisphere(_color(sky), _color(ground), intensity); } }
export class DirectionalLightShadow extends LightShadow { constructor() { super(new OrthographicCamera(-5, 5, 5, -5, 0.5, 500)); } }
export class PointLightShadow extends LightShadow { constructor() { super(new PerspectiveCamera(90, 1, 0.5, 500)); } }
export class SpotLightShadow extends LightShadow { constructor() { super(new PerspectiveCamera(90, 1, 0.5, 500)); this.focus = 1; } }

// Objects. BatchedMesh is the real impl above (per-geom InstancedMesh pool).
export class LOD extends Object3D { constructor() { super(); this.levels = []; this.autoUpdate = true; } addLevel(obj, distance = 0) { this.levels.push({ object: obj, distance }); this.add(obj); return this; } getCurrentLevel() { return 0; } }

// Helpers.

// Loaders.
export class Loader extends _Loader {}
export class LoaderUtils { static decodeText(buf) { return new TextDecoder().decode(buf); } static extractUrlBase(url) { const i = url.lastIndexOf('/'); return i < 0 ? './' : url.slice(0, i + 1); } static resolveURL(url, path) { if (typeof url !== 'string' || url === '') return ''; if (/^https?:\/\//i.test(path) && /^\//.test(url)) path = path.replace(/(^https?:\/\/[^/]+).*/i, '$1'); if (/^(https?:)?\/\//i.test(url)) return url; if (/^data:.*,.*$/i.test(url)) return url; if (/^blob:.*$/i.test(url)) return url; return path + url; } }
export class DataTextureLoader extends _Loader { load(url, onLoad, _onProgress, onError) { new FileLoader(this.manager).setResponseType('arraybuffer').load(url, (buf) => { const tex = new DataTexture(new Uint8Array(buf), 1, 1); onLoad?.(tex); }, undefined, onError); } }
export class CompressedTextureLoader extends _Loader { load(url, onLoad, _onProgress, onError) { new FileLoader(this.manager).setResponseType('arraybuffer').load(url, () => onLoad?.(new CompressedTexture()), undefined, onError); } }
export class ImageBitmapLoader extends _Loader { setOptions(opts) { this.options = opts; return this; } load(url, onLoad, _onProgress, onError) { fetch(this.path + url).then(r => r.blob()).then(b => createImageBitmap(b, this.options)).then(onLoad).catch(onError); } }

// Misc.
export class Cache { static enabled = false; static files = {}; static add(k, v) { this.files[k] = v; } static get(k) { return this.files[k]; } static remove(k) { delete this.files[k]; } static clear() { this.files = {}; } }
export class Fog { constructor(color = 0xffffff, near = 1, far = 1000) { this.color = new Color(color); this.near = near; this.far = far; this.isFog = true; } clone() { return new Fog(this.color.getHex(), this.near, this.far); } }
export class FogExp2 { constructor(color = 0xffffff, density = 0.00025) { this.color = new Color(color); this.density = density; this.isFogExp2 = true; } }
export class SphericalHarmonics3 { constructor() { this.coefficients = []; for (let i = 0; i < 9; i++) this.coefficients.push(new Vector3()); } set(coeffs) { for (let i = 0; i < 9; i++) this.coefficients[i].copy(coeffs[i]); return this; } zero() { for (const c of this.coefficients) c.set(0, 0, 0); return this; } }
export class Uniform { constructor(value) { this.value = value; } clone() { return new Uniform(this.value?.clone ? this.value.clone() : this.value); } }
export class UniformsGroup { constructor() { this.uniforms = []; this.name = ''; } add(u) { this.uniforms.push(u); return this; } setName(n) { this.name = n; return this; } }
export const UniformsLib = { common: {}, lights: {}, points: {}, sprite: {}, shadowmap: {}, fog: {} };
export const UniformsUtils = { clone(uniforms) { const out = {}; for (const k of Object.keys(uniforms)) out[k] = new Uniform(uniforms[k].value); return out; }, merge(list) { const out = {}; for (const u of list) Object.assign(out, u); return out; } };
export const ShaderChunk = {};
export const ShaderLib = { common: { uniforms: {}, vertexShader: '', fragmentShader: '' }, basic: { uniforms: {}, vertexShader: '', fragmentShader: '' }, lambert: { uniforms: {}, vertexShader: '', fragmentShader: '' }, phong: { uniforms: {}, vertexShader: '', fragmentShader: '' }, standard: { uniforms: {}, vertexShader: '', fragmentShader: '' }, physical: { uniforms: {}, vertexShader: '', fragmentShader: '' }, points: { uniforms: {}, vertexShader: '', fragmentShader: '' } };
export const ImageUtils = { getDataURL(_img) { return ''; }, sRGBToLinear(img) { return img; } };
export const DataUtils = { fromHalfFloat: (v) => v, toHalfFloat: (v) => v };
// three.js r165 ShapeUtils over Rust (`curves::shapes::shape_utils` and its
// exact Earcut port).
export const ShapeUtils = {
    area(contour) { return shapeUtilsArea(_flat2(contour)); },
    isClockWise(pts) { return ShapeUtils.area(pts) < 0; },
    triangulateShape(contour, holes = []) {
        const lengths = new Uint32Array(holes.map((h) => h.length));
        const r = shapeUtilsTriangulate(_flat2(contour), _flat2(holes.flat()), lengths);
        // three.js drops a repeated closing point from the caller's arrays.
        if (r[0]) contour.pop();
        holes.forEach((h, i) => { if (r[i + 1]) h.pop(); });
        const faces = [];
        for (let i = holes.length + 1; i < r.length; i += 3) faces.push([r[i], r[i + 1], r[i + 2]]);
        return faces;
    },
};
// (MathUtils + ColorManagement already declared above; not re-declared here.)
// PropertyBinding / PropertyMixer / AnimationAction defined with AnimationMixer above.
export class AudioContext { static getContext() { if (!this._ctx && typeof window !== 'undefined' && window.AudioContext) this._ctx = new (window.AudioContext || window.webkitAudioContext)(); return this._ctx; } static setContext(c) { this._ctx = c; } }
export class AudioAnalyser { constructor(audio, fftSize = 2048) { this.analyser = AudioContext.getContext()?.createAnalyser?.(); this.data = new Uint8Array(fftSize / 2); } getFrequencyData() { this.analyser?.getByteFrequencyData?.(this.data); return this.data; } getAverageFrequency() { let s = 0; for (const v of this.data) s += v; return s / this.data.length; } }
export class PositionalAudio extends Audio { constructor(listener) { super(listener); this.panner = AudioContext.getContext()?.createPanner?.(); } setRefDistance() {} setRolloffFactor() {} setDistanceModel() {} setMaxDistance() {} setDirectionalCone() {} }
export class PMREMGenerator {
    constructor(renderer) {
        this.renderer = renderer;
        this._w = new WebPmremGenerator();
    }
    fromScene(scene, _sigma = 0, near = 0.1, far = 100, size = 256) {
        // three.js r16x passes `{ size, position }` as the fourth-onward options
        if (size && typeof size === 'object') size = size.size ?? 256;
        // The room every three.js app lights with, built exactly in wasm (HDR,
        // ray-cast) rather than rasterised through a cube camera.
        if (scene?.isRoomEnvironment && typeof WebPmremGenerator.roomEnvironment === 'function') {
            const texOut = Object.create(CubeTexture.prototype);
            texOut._w = WebPmremGenerator.roomEnvironment(size);
            return { texture: texOut, dispose() {} };
        }
        const rt = new WebGLCubeRenderTarget(size);
        const cubeCam = new CubeCamera(near, far, rt);
        if (this.renderer) cubeCam.update(this.renderer, scene);
        return this.fromCubemap(rt.texture);
    }
    fromEquirectangular(tex, _renderTarget = null) {
        const img = tex?.image?.data || tex?.image;
        const w = tex?.image?.width || tex?.source?.data?.width || 512;
        const h = tex?.image?.height || tex?.source?.data?.height || 256;
        const size = 128;
        let data;
        if (img instanceof Uint8Array) data = Array.from(img);
        else if (img?.data) data = Array.from(img.data);
        else data = [];
        const out_w = WebPmremGenerator.fromEquirectangular(data, w, h, size);
        const texOut = Object.create(CubeTexture.prototype);
        texOut._w = out_w;
        return { texture: texOut };
    }
    fromCubemap(cubeMap, _renderTarget = null) {
        const src = cubeMap._w;
        if (!src) {
            const fallback = Object.create(CubeTexture.prototype);
            return { texture: fallback };
        }
        const size = src.size ?? 256;
        const out_w = WebPmremGenerator.fromCubemap(src, size);
        const texOut = Object.create(CubeTexture.prototype);
        texOut._w = out_w;
        return { texture: texOut };
    }
    dispose() {}
}

// ============================================================================
//   `examples/jsm` add-ons commonly imported from `three/examples/jsm/...`.
//   These are NOT in the core three.js bundle but real apps use them widely.
// ============================================================================

// ---- Math add-ons ----

// Oriented Bounding Box: three.js r165 `examples/jsm/math/OBB.js`.
const _obb_a = {
    c: null, // center
    u: [ new Vector3(), new Vector3(), new Vector3() ], // basis vectors
    e: [] // half width
};

const _obb_b = {
    c: null, // center
    u: [ new Vector3(), new Vector3(), new Vector3() ], // basis vectors
    e: [] // half width
};

const _obb_R = [[], [], []];
const _obb_AbsR = [[], [], []];
const _obb_t = [];

const _obb_xAxis = new Vector3();
const _obb_yAxis = new Vector3();
const _obb_zAxis = new Vector3();
const _obb_v1 = new Vector3();
const _obb_size = new Vector3();
const _obb_closestPoint = new Vector3();
const _obb_rotationMatrix = new Matrix3();
const _obb_aabb = new Box3();
const _obb_matrix = new Matrix4();
const _obb_inverse = new Matrix4();
const _obb_localRay = new Ray();

export class OBB {
    constructor( center = new Vector3(), halfSize = new Vector3(), rotation = new Matrix3() ) {
        this.center = center;
        this.halfSize = halfSize;
        this.rotation = rotation;
    }
    set( center, halfSize, rotation ) {
        this.center = center;
        this.halfSize = halfSize;
        this.rotation = rotation;
        return this;
    }
    copy( _obb_obb ) {
        this.center.copy( _obb_obb.center );
        this.halfSize.copy( _obb_obb.halfSize );
        this.rotation.copy( _obb_obb.rotation );
        return this;
    }
    clone() {
        return new this.constructor().copy( this );
    }
    getSize( result ) {
        return result.copy( this.halfSize ).multiplyScalar( 2 );
    }
    /**
    * Reference: Closest Point on OBB to Point in Real-Time Collision Detection
    * by Christer Ericson (chapter 5.1.4)
    */
    clampPoint( point, result ) {
        const halfSize = this.halfSize;
        _obb_v1.subVectors( point, this.center );
        this.rotation.extractBasis( _obb_xAxis, _obb_yAxis, _obb_zAxis );
        // start at the center position of the OBB
        result.copy( this.center );
        // project the target onto the OBB axes and walk towards that point
        const x = MathUtils.clamp( _obb_v1.dot( _obb_xAxis ), - halfSize.x, halfSize.x );
        result.add( _obb_xAxis.multiplyScalar( x ) );
        const y = MathUtils.clamp( _obb_v1.dot( _obb_yAxis ), - halfSize.y, halfSize.y );
        result.add( _obb_yAxis.multiplyScalar( y ) );
        const z = MathUtils.clamp( _obb_v1.dot( _obb_zAxis ), - halfSize.z, halfSize.z );
        result.add( _obb_zAxis.multiplyScalar( z ) );
        return result;
    }
    containsPoint( point ) {
        _obb_v1.subVectors( point, this.center );
        this.rotation.extractBasis( _obb_xAxis, _obb_yAxis, _obb_zAxis );
        // project _obb_v1 onto each axis and check if these points lie inside the OBB
        return Math.abs( _obb_v1.dot( _obb_xAxis ) ) <= this.halfSize.x &&
                Math.abs( _obb_v1.dot( _obb_yAxis ) ) <= this.halfSize.y &&
                Math.abs( _obb_v1.dot( _obb_zAxis ) ) <= this.halfSize.z;
    }
    intersectsBox3( box3 ) {
        return this.intersectsOBB( _obb_obb.fromBox3( box3 ) );
    }
    intersectsSphere( sphere ) {
        // find the point on the OBB closest to the sphere center
        this.clampPoint( sphere.center, _obb_closestPoint );
        // if that point is inside the sphere, the OBB and sphere intersect
        return _obb_closestPoint.distanceToSquared( sphere.center ) <= ( sphere.radius * sphere.radius );
    }
    /**
    * Reference: OBB-OBB Intersection in Real-Time Collision Detection
    * by Christer Ericson (chapter 4.4.1)
    *
    */
    intersectsOBB( _obb_obb, epsilon = Number.EPSILON ) {
        // prepare data structures (the code uses the same nomenclature like the reference)
        _obb_a.c = this.center;
        _obb_a.e[ 0 ] = this.halfSize.x;
        _obb_a.e[ 1 ] = this.halfSize.y;
        _obb_a.e[ 2 ] = this.halfSize.z;
        this.rotation.extractBasis( _obb_a.u[ 0 ], _obb_a.u[ 1 ], _obb_a.u[ 2 ] );
        _obb_b.c = _obb_obb.center;
        _obb_b.e[ 0 ] = _obb_obb.halfSize.x;
        _obb_b.e[ 1 ] = _obb_obb.halfSize.y;
        _obb_b.e[ 2 ] = _obb_obb.halfSize.z;
        _obb_obb.rotation.extractBasis( _obb_b.u[ 0 ], _obb_b.u[ 1 ], _obb_b.u[ 2 ] );
        // compute rotation _obb_matrix expressing _obb_b in _obb_a's coordinate frame
        for ( let i = 0; i < 3; i ++ ) {
            for ( let j = 0; j < 3; j ++ ) {
                _obb_R[ i ][ j ] = _obb_a.u[ i ].dot( _obb_b.u[ j ] );
            }
        }
        // compute translation vector
        _obb_v1.subVectors( _obb_b.c, _obb_a.c );
        // bring translation into _obb_a's coordinate frame
        _obb_t[ 0 ] = _obb_v1.dot( _obb_a.u[ 0 ] );
        _obb_t[ 1 ] = _obb_v1.dot( _obb_a.u[ 1 ] );
        _obb_t[ 2 ] = _obb_v1.dot( _obb_a.u[ 2 ] );
        // compute common subexpressions. Add in an epsilon term to
        // counteract arithmetic errors when two edges are parallel and
        // their cross product is (near) null
        for ( let i = 0; i < 3; i ++ ) {
            for ( let j = 0; j < 3; j ++ ) {
                _obb_AbsR[ i ][ j ] = Math.abs( _obb_R[ i ][ j ] ) + epsilon;
            }
        }
        let ra, rb;
        // test axes L = A0, L = A1, L = A2
        for ( let i = 0; i < 3; i ++ ) {
            ra = _obb_a.e[ i ];
            rb = _obb_b.e[ 0 ] * _obb_AbsR[ i ][ 0 ] + _obb_b.e[ 1 ] * _obb_AbsR[ i ][ 1 ] + _obb_b.e[ 2 ] * _obb_AbsR[ i ][ 2 ];
            if ( Math.abs( _obb_t[ i ] ) > ra + rb ) return false;
        }
        // test axes L = B0, L = B1, L = B2
        for ( let i = 0; i < 3; i ++ ) {
            ra = _obb_a.e[ 0 ] * _obb_AbsR[ 0 ][ i ] + _obb_a.e[ 1 ] * _obb_AbsR[ 1 ][ i ] + _obb_a.e[ 2 ] * _obb_AbsR[ 2 ][ i ];
            rb = _obb_b.e[ i ];
            if ( Math.abs( _obb_t[ 0 ] * _obb_R[ 0 ][ i ] + _obb_t[ 1 ] * _obb_R[ 1 ][ i ] + _obb_t[ 2 ] * _obb_R[ 2 ][ i ] ) > ra + rb ) return false;
        }
        // test axis L = A0 x B0
        ra = _obb_a.e[ 1 ] * _obb_AbsR[ 2 ][ 0 ] + _obb_a.e[ 2 ] * _obb_AbsR[ 1 ][ 0 ];
        rb = _obb_b.e[ 1 ] * _obb_AbsR[ 0 ][ 2 ] + _obb_b.e[ 2 ] * _obb_AbsR[ 0 ][ 1 ];
        if ( Math.abs( _obb_t[ 2 ] * _obb_R[ 1 ][ 0 ] - _obb_t[ 1 ] * _obb_R[ 2 ][ 0 ] ) > ra + rb ) return false;
        // test axis L = A0 x B1
        ra = _obb_a.e[ 1 ] * _obb_AbsR[ 2 ][ 1 ] + _obb_a.e[ 2 ] * _obb_AbsR[ 1 ][ 1 ];
        rb = _obb_b.e[ 0 ] * _obb_AbsR[ 0 ][ 2 ] + _obb_b.e[ 2 ] * _obb_AbsR[ 0 ][ 0 ];
        if ( Math.abs( _obb_t[ 2 ] * _obb_R[ 1 ][ 1 ] - _obb_t[ 1 ] * _obb_R[ 2 ][ 1 ] ) > ra + rb ) return false;
        // test axis L = A0 x B2
        ra = _obb_a.e[ 1 ] * _obb_AbsR[ 2 ][ 2 ] + _obb_a.e[ 2 ] * _obb_AbsR[ 1 ][ 2 ];
        rb = _obb_b.e[ 0 ] * _obb_AbsR[ 0 ][ 1 ] + _obb_b.e[ 1 ] * _obb_AbsR[ 0 ][ 0 ];
        if ( Math.abs( _obb_t[ 2 ] * _obb_R[ 1 ][ 2 ] - _obb_t[ 1 ] * _obb_R[ 2 ][ 2 ] ) > ra + rb ) return false;
        // test axis L = A1 x B0
        ra = _obb_a.e[ 0 ] * _obb_AbsR[ 2 ][ 0 ] + _obb_a.e[ 2 ] * _obb_AbsR[ 0 ][ 0 ];
        rb = _obb_b.e[ 1 ] * _obb_AbsR[ 1 ][ 2 ] + _obb_b.e[ 2 ] * _obb_AbsR[ 1 ][ 1 ];
        if ( Math.abs( _obb_t[ 0 ] * _obb_R[ 2 ][ 0 ] - _obb_t[ 2 ] * _obb_R[ 0 ][ 0 ] ) > ra + rb ) return false;
        // test axis L = A1 x B1
        ra = _obb_a.e[ 0 ] * _obb_AbsR[ 2 ][ 1 ] + _obb_a.e[ 2 ] * _obb_AbsR[ 0 ][ 1 ];
        rb = _obb_b.e[ 0 ] * _obb_AbsR[ 1 ][ 2 ] + _obb_b.e[ 2 ] * _obb_AbsR[ 1 ][ 0 ];
        if ( Math.abs( _obb_t[ 0 ] * _obb_R[ 2 ][ 1 ] - _obb_t[ 2 ] * _obb_R[ 0 ][ 1 ] ) > ra + rb ) return false;
        // test axis L = A1 x B2
        ra = _obb_a.e[ 0 ] * _obb_AbsR[ 2 ][ 2 ] + _obb_a.e[ 2 ] * _obb_AbsR[ 0 ][ 2 ];
        rb = _obb_b.e[ 0 ] * _obb_AbsR[ 1 ][ 1 ] + _obb_b.e[ 1 ] * _obb_AbsR[ 1 ][ 0 ];
        if ( Math.abs( _obb_t[ 0 ] * _obb_R[ 2 ][ 2 ] - _obb_t[ 2 ] * _obb_R[ 0 ][ 2 ] ) > ra + rb ) return false;
        // test axis L = A2 x B0
        ra = _obb_a.e[ 0 ] * _obb_AbsR[ 1 ][ 0 ] + _obb_a.e[ 1 ] * _obb_AbsR[ 0 ][ 0 ];
        rb = _obb_b.e[ 1 ] * _obb_AbsR[ 2 ][ 2 ] + _obb_b.e[ 2 ] * _obb_AbsR[ 2 ][ 1 ];
        if ( Math.abs( _obb_t[ 1 ] * _obb_R[ 0 ][ 0 ] - _obb_t[ 0 ] * _obb_R[ 1 ][ 0 ] ) > ra + rb ) return false;
        // test axis L = A2 x B1
        ra = _obb_a.e[ 0 ] * _obb_AbsR[ 1 ][ 1 ] + _obb_a.e[ 1 ] * _obb_AbsR[ 0 ][ 1 ];
        rb = _obb_b.e[ 0 ] * _obb_AbsR[ 2 ][ 2 ] + _obb_b.e[ 2 ] * _obb_AbsR[ 2 ][ 0 ];
        if ( Math.abs( _obb_t[ 1 ] * _obb_R[ 0 ][ 1 ] - _obb_t[ 0 ] * _obb_R[ 1 ][ 1 ] ) > ra + rb ) return false;
        // test axis L = A2 x B2
        ra = _obb_a.e[ 0 ] * _obb_AbsR[ 1 ][ 2 ] + _obb_a.e[ 1 ] * _obb_AbsR[ 0 ][ 2 ];
        rb = _obb_b.e[ 0 ] * _obb_AbsR[ 2 ][ 1 ] + _obb_b.e[ 1 ] * _obb_AbsR[ 2 ][ 0 ];
        if ( Math.abs( _obb_t[ 1 ] * _obb_R[ 0 ][ 2 ] - _obb_t[ 0 ] * _obb_R[ 1 ][ 2 ] ) > ra + rb ) return false;
        // since no separating axis is found, the OBBs must be intersecting
        return true;
    }
    /**
    * Reference: Testing Box Against Plane in Real-Time Collision Detection
    * by Christer Ericson (chapter 5.2.3)
    */
    intersectsPlane( plane ) {
        this.rotation.extractBasis( _obb_xAxis, _obb_yAxis, _obb_zAxis );
        // compute the projection interval radius of this OBB onto L(_obb_t) = this->center + _obb_t * p.normal;
        const r = this.halfSize.x * Math.abs( plane.normal.dot( _obb_xAxis ) ) +
                this.halfSize.y * Math.abs( plane.normal.dot( _obb_yAxis ) ) +
                this.halfSize.z * Math.abs( plane.normal.dot( _obb_zAxis ) );
        // compute distance of the OBB's center from the plane
        const d = plane.normal.dot( this.center ) - plane.constant;
        // Intersection occurs when distance d falls within [-r,+r] interval
        return Math.abs( d ) <= r;
    }
    /**
    * Performs _obb_a ray/OBB intersection test and stores the intersection point
    * to the given 3D vector. If no intersection is detected, *null* is returned.
    */
    intersectRay( ray, result ) {
        // the idea is to perform the intersection test in the local space
        // of the OBB.
        this.getSize( _obb_size );
        _obb_aabb.setFromCenterAndSize( _obb_v1.set( 0, 0, 0 ), _obb_size );
        // create _obb_a 4x4 transformation _obb_matrix
        _obb_matrix.setFromMatrix3( this.rotation );
        _obb_matrix.setPosition( this.center );
        // transform ray to the local space of the OBB
        _obb_inverse.copy( _obb_matrix ).invert();
        _obb_localRay.copy( ray ).applyMatrix4( _obb_inverse );
        // perform ray <-> AABB intersection test
        if ( _obb_localRay.intersectBox( _obb_aabb, result ) ) {
            // transform the intersection point back to world space
            return result.applyMatrix4( _obb_matrix );
        } else {
            return null;
        }
    }
    /**
    * Performs _obb_a ray/OBB intersection test. Returns either true or false if
    * there is _obb_a intersection or not.
    */
    intersectsRay( ray ) {
        return this.intersectRay( ray, _obb_v1 ) !== null;
    }
    fromBox3( box3 ) {
        box3.getCenter( this.center );
        box3.getSize( this.halfSize ).multiplyScalar( 0.5 );
        this.rotation.identity();
        return this;
    }
    equals( _obb_obb ) {
        return _obb_obb.center.equals( this.center ) &&
            _obb_obb.halfSize.equals( this.halfSize ) &&
            _obb_obb.rotation.equals( this.rotation );
    }
    applyMatrix4( _obb_matrix ) {
        const e = _obb_matrix.elements;
        let sx = _obb_v1.set( e[ 0 ], e[ 1 ], e[ 2 ] ).length();
        const sy = _obb_v1.set( e[ 4 ], e[ 5 ], e[ 6 ] ).length();
        const sz = _obb_v1.set( e[ 8 ], e[ 9 ], e[ 10 ] ).length();
        const det = _obb_matrix.determinant();
        if ( det < 0 ) sx = - sx;
        _obb_rotationMatrix.setFromMatrix4( _obb_matrix );
        const invSX = 1 / sx;
        const invSY = 1 / sy;
        const invSZ = 1 / sz;
        _obb_rotationMatrix.elements[ 0 ] *= invSX;
        _obb_rotationMatrix.elements[ 1 ] *= invSX;
        _obb_rotationMatrix.elements[ 2 ] *= invSX;
        _obb_rotationMatrix.elements[ 3 ] *= invSY;
        _obb_rotationMatrix.elements[ 4 ] *= invSY;
        _obb_rotationMatrix.elements[ 5 ] *= invSY;
        _obb_rotationMatrix.elements[ 6 ] *= invSZ;
        _obb_rotationMatrix.elements[ 7 ] *= invSZ;
        _obb_rotationMatrix.elements[ 8 ] *= invSZ;
        this.rotation.multiply( _obb_rotationMatrix );
        this.halfSize.x *= sx;
        this.halfSize.y *= sy;
        this.halfSize.z *= sz;
        _obb_v1.setFromMatrixPosition( _obb_matrix );
        this.center.add( _obb_v1 );
        return this;
    }
}
const _obb_obb = new OBB();

// Math Capsule. Mirrors three.js's `examples/jsm/math/Capsule.js` — a swept
// sphere defined by two endpoints + radius. Used for character/physics collision.
export class Capsule {
    constructor(start = new Vector3(0, 0, 0), end = new Vector3(0, 1, 0), radius = 1) {
        this.start = start; this.end = end; this.radius = radius;
    }
    clone() { return new Capsule(this.start.clone(), this.end.clone(), this.radius); }
    set(s, e, r) { this.start.copy(s); this.end.copy(e); this.radius = r; return this; }
    copy(c) { this.start.copy(c.start); this.end.copy(c.end); this.radius = c.radius; return this; }
    translate(v) { this.start.add(v); this.end.add(v); return this; }
    getCenter(target = new Vector3()) { return target.addVectors(this.start, this.end).multiplyScalar(0.5); }
}

// 3D Perlin noise (`examples/jsm/math/ImprovedNoise.js`). Returns values in
// roughly [-1, 1] for any (x,y,z). Used heavily for terrain/cloud synthesis.
export class ImprovedNoise {
    constructor() {
        const p = new Array(512);
        const permutation = [151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180];
        for (let i = 0; i < 256; i++) p[i] = p[i + 256] = permutation[i];
        this.p = p;
    }
    noise(x, y, z) {
        const p = this.p;
        const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
        const lerp = (t, a, b) => a + t * (b - a);
        const grad = (h, x, y, z) => { h = h & 15; const u = h < 8 ? x : y; const v = h < 4 ? y : h === 12 || h === 14 ? x : z; return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v); };
        const X = Math.floor(x) & 255, Y = Math.floor(y) & 255, Z = Math.floor(z) & 255;
        x -= Math.floor(x); y -= Math.floor(y); z -= Math.floor(z);
        const u = fade(x), v = fade(y), w = fade(z);
        const A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z;
        const B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;
        return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                              lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
                       lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                              lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))));
    }
}

// ---- Geometry add-ons ----

// RoundedBoxGeometry — three.js r165 addon, generated in Rust: a unit
// BoxGeometry with `2·segments + 1` divisions bent onto the rounded box,
// unindexed, with the box's six groups. Like three.js's (which extends
// BoxGeometry) its `type` and `parameters` are the underlying box's.
export class RoundedBoxGeometry extends BufferGeometry {
    constructor(width = 1, height = 1, depth = 1, segments = 2, radius = 0.1) {
        super();
        const s = segments * 2 + 1;
        this.type = 'BoxGeometry';
        this.parameters = { width: 1, height: 1, depth: 1, widthSegments: s, heightSegments: s, depthSegments: s };
        _fillFromWasm(this, WebGeometry.roundedBox(width, height, depth, Math.max(0, Math.floor(segments)), radius));
    }
}

// TeapotGeometry — three.js r165 addon: Newell's teapot patches tessellated
// in Rust, same options and defaults.
export class TeapotGeometry extends BufferGeometry {
    constructor(size = 50, segments = 10, bottom = true, lid = true, body = true, fitLid = true, blinn = true) {
        super();
        _fillFromWasm(this, WebGeometry.teapot(size, Math.max(2, Math.floor(segments)), !!bottom, !!lid, !!body, !!fitLid, !!blinn));
    }
}

// ParametricGeometries — curated collection that wraps ParametricGeometry
// with built-in surface formulas. Common ones: klein, mobius, plane.
export const ParametricGeometries = {
    klein(u, v, target) {
        u *= Math.PI; v *= 2 * Math.PI;
        u = u * 2;
        let x, z;
        if (u < Math.PI) {
            x = 3 * Math.cos(u) * (1 + Math.sin(u)) + (2 * (1 - Math.cos(u) / 2)) * Math.cos(u) * Math.cos(v);
            z = -8 * Math.sin(u) - 2 * (1 - Math.cos(u) / 2) * Math.sin(u) * Math.cos(v);
        } else {
            x = 3 * Math.cos(u) * (1 + Math.sin(u)) + (2 * (1 - Math.cos(u) / 2)) * Math.cos(v + Math.PI);
            z = -8 * Math.sin(u);
        }
        const y = -2 * (1 - Math.cos(u) / 2) * Math.sin(v);
        target.set(x, y, z);
    },
    mobius(u, t, target) {
        u = u - 0.5; const v = 2 * Math.PI * t;
        const a = 2;
        const x = Math.cos(v) * (a + u * Math.cos(v / 2));
        const y = Math.sin(v) * (a + u * Math.cos(v / 2));
        const z = u * Math.sin(v / 2);
        target.set(x, y, z);
    },
    plane(width, height) { return (u, v, target) => { target.set(u * width, 0, v * height); }; },
    KleinGeometry: class extends ParametricGeometry { constructor(slices = 25, stacks = 25) { super(ParametricGeometries.klein, slices, stacks); this.type = 'KleinGeometry'; } },
    MobiusGeometry: class extends ParametricGeometry { constructor(slices = 25, stacks = 25) { super(ParametricGeometries.mobius, slices, stacks); this.type = 'MobiusGeometry'; } },
};

// ---- Thick lines (`examples/jsm/lines/...`) ----

// LineMaterial — three.js's wide-line material. Drawn as triangle quads.
// We approximate by ignoring linewidth (always 1px) and falling back to
// the standard solid-line shader path. Color and dashed params honored.
export class LineMaterial extends LineBasicMaterial {
    constructor(opts = {}) {
        super(opts);
        this.linewidth = opts.linewidth ?? 1;
        this.resolution = opts.resolution || new Vector2(800, 600);
        this.dashed = !!opts.dashed;
        this.dashSize = opts.dashSize ?? 1;
        this.gapSize = opts.gapSize ?? 1;
        this.dashScale = opts.dashScale ?? 1;
        this.worldUnits = !!opts.worldUnits;
        if (this.dashed) this._isLineDashed = true;
    }
}

// LineGeometry / LineSegmentsGeometry — store a flat list of position pairs
// and emit a regular BufferGeometry that `LineSegments` can render.
export class LineSegmentsGeometry extends BufferGeometry {
    constructor() {
        super();
        this.type = 'LineSegmentsGeometry';
    }
    setPositions(arr) {
        // arr is [x0,y0,z0, x1,y1,z1, ...] of paired endpoints.
        this.setAttribute('position', new BufferAttribute(arr instanceof Float32Array ? arr : new Float32Array(arr), 3));
        return this;
    }
    setColors(arr) {
        this.setAttribute('color', new BufferAttribute(arr instanceof Float32Array ? arr : new Float32Array(arr), 3));
        return this;
    }
    fromMesh(mesh) {
        const geom = mesh.geometry;
        const pos = geom.attributes?.position;
        if (pos) this.setPositions(_attrPacked(pos));
        return this;
    }
    fromEdgesGeometry(eg) { return this.fromMesh({ geometry: eg }); }
    fromWireframeGeometry(wg) { return this.fromMesh({ geometry: wg }); }
}
export class LineGeometry extends LineSegmentsGeometry {
    constructor() { super(); this.type = 'LineGeometry'; }
    setPositions(arr) {
        // LineGeometry stores a STRIP — convert to paired segments.
        const src = arr instanceof Float32Array ? arr : new Float32Array(arr);
        const n = src.length / 3 | 0;
        const out = new Float32Array(Math.max(0, n - 1) * 6);
        let o = 0;
        for (let i = 0; i < n - 1; i++) {
            out[o++] = src[i * 3];     out[o++] = src[i * 3 + 1]; out[o++] = src[i * 3 + 2];
            out[o++] = src[(i + 1) * 3]; out[o++] = src[(i + 1) * 3 + 1]; out[o++] = src[(i + 1) * 3 + 2];
        }
        return super.setPositions(out);
    }
}

// Line2 / LineSegments2 — bullets-and-quads thick-line objects. We render
// as standard LineSegments at 1px (linewidth in screen pixels would require
// a custom triangle-quad shader). API-compatible for the common case.
export class LineSegments2 extends LineSegments {
    constructor(geometry = new LineSegmentsGeometry(), material = new LineMaterial()) {
        super(geometry, material);
        this.type = 'LineSegments2';
    }
}
export class Line2 extends LineSegments2 {
    constructor(geometry = new LineGeometry(), material = new LineMaterial()) {
        super(geometry, material);
        this.type = 'Line2';
    }
}
// Wireframe — Line-based variant of WireframeGeometry that uses LineMaterial.
export class Wireframe extends LineSegments2 {
    constructor(geometry = new LineSegmentsGeometry(), material = new LineMaterial()) {
        super(geometry, material);
        this.type = 'Wireframe';
    }
}

// ---- Utility namespaces (`examples/jsm/utils/...`) ----

// BufferGeometryUtils. We support the most common operations: merge,
// flatten, compute tangents (stubbed).
export const BufferGeometryUtils = {
    mergeGeometries(geoms, _useGroups = false) {
        // Merge by concatenating position/normal/uv/index arrays.
        const positions = [], normals = [], uvs = [], indices = [];
        let vOff = 0;
        for (const g of geoms) {
            const posAttr = g.attributes?.position;
            if (!posAttr) continue;
            // Packed views, so interleaved attributes merge too.
            const pos = { array: _attrPacked(posAttr) };
            for (const v of pos.array) positions.push(v);
            const nrm = _attrPacked(g.attributes?.normal);
            for (let i = 0; i < pos.array.length; i++) normals.push(nrm?.[i] ?? 0);
            const uv = _attrPacked(g.attributes?.uv);
            for (let i = 0; i < (pos.array.length / 3) * 2; i++) uvs.push(uv?.[i] ?? 0);
            const idx = g.index?.array || g._w?.index;
            if (idx) {
                for (const i of idx) indices.push(i + vOff);
            } else {
                const n = pos.array.length / 3 | 0;
                for (let i = 0; i < n; i++) indices.push(i + vOff);
            }
            vOff += pos.array.length / 3;
        }
        const out = new BufferGeometry();
        out.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
        out.setAttribute('normal',   new BufferAttribute(new Float32Array(normals), 3));
        out.setAttribute('uv',       new BufferAttribute(new Float32Array(uvs), 2));
        out.setIndex(indices);
        return out;
    },
    mergeAttributes(attrs) {
        // An interleaved attribute contributes its own components, not the
        // whole shared buffer (three.js reads those with `getComponent`).
        const raw = (a) => a.isInterleavedBufferAttribute ? new a.array.constructor(a._deinterleaved()) : a.array;
        let total = 0; for (const a of attrs) total += raw(a).length;
        const out = new (attrs[0].array.constructor)(total);
        let o = 0; for (const a of attrs) { const r = raw(a); out.set(r, o); o += r.length; }
        const result = new BufferAttribute(out, attrs[0].itemSize);
        result.normalized = !!attrs[0].normalized;
        return result;
    },
    /// three.js r165: one `InterleavedBuffer` holding all of `attributes`
    /// (same array type), returned as an `InterleavedBufferAttribute` each.
    interleaveAttributes(attributes) {
        let TypedArray, arrayLength = 0, stride = 0;
        for (const attribute of attributes) {
            if (TypedArray === undefined) TypedArray = attribute.array.constructor;
            if (TypedArray !== attribute.array.constructor) {
                console.error('AttributeBuffers of different types cannot be interleaved');
                return null;
            }
            arrayLength += attribute.array.length;
            stride += attribute.itemSize;
        }
        const interleavedBuffer = new InterleavedBuffer(new TypedArray(arrayLength), stride);
        let offset = 0;
        const res = [];
        const getters = ['getX', 'getY', 'getZ', 'getW'];
        const setters = ['setX', 'setY', 'setZ', 'setW'];
        for (const attribute of attributes) {
            const itemSize = attribute.itemSize, count = attribute.count;
            const iba = new InterleavedBufferAttribute(interleavedBuffer, itemSize, offset, attribute.normalized);
            res.push(iba);
            offset += itemSize;
            for (let c = 0; c < count; c++) {
                for (let k = 0; k < itemSize; k++) iba[setters[k]](c, attribute[getters[k]](c));
            }
        }
        return res;
    },
    /// three.js r165: a plain copy of an interleaved attribute.
    deinterleaveAttribute(attribute) {
        const out = attribute.clone();
        if (attribute.data?.isInstancedInterleavedBuffer) {
            const inst = new InstancedBufferAttribute(out.array, out.itemSize, out.normalized, attribute.data.meshPerAttribute);
            inst.normalized = out.normalized;
            return inst;
        }
        return out;
    },
    /// three.js r165: replace every interleaved attribute of `geometry`
    /// (and morph target) with a plain one.
    deinterleaveGeometry(geometry) {
        const map = new Map();
        const swap = (attr) => {
            if (!attr?.isInterleavedBufferAttribute) return attr;
            if (!map.has(attr)) map.set(attr, this.deinterleaveAttribute(attr));
            return map.get(attr);
        };
        for (const key of Object.keys(geometry.attributes)) geometry.setAttribute(key, swap(geometry.attributes[key]));
        const morph = geometry.morphAttributes ?? {};
        let changed = false;
        for (const key of Object.keys(morph)) {
            morph[key] = morph[key].map((a) => { const b = swap(a); changed ||= b !== a; return b; });
        }
        if (changed) geometry.morphAttributes = morph;
    },
    mergeBufferGeometries(geoms, useGroups) { return this.mergeGeometries(geoms, useGroups); },
    mergeBufferAttributes(attrs) { return this.mergeAttributes(attrs); },
    computeTangents(geom) { return geom; },
    estimateBytesUsed(geom) {
        let bytes = 0;
        for (const k of Object.keys(geom.attributes || {})) {
            const a = geom.attributes[k];
            // three.js: itemSize × count × element size, so an interleaved
            // attribute counts its own share of the buffer.
            bytes += a.isInterleavedBufferAttribute ? a.count * a.itemSize * a.array.BYTES_PER_ELEMENT : a.array.byteLength;
        }
        if (geom.index?.array) bytes += geom.index.array.byteLength;
        return bytes;
    },
    toCreasedNormals(geom) { geom.computeVertexNormals?.(); return geom; },
    toTrianglesDrawMode(geom) { return geom; },
};

// SceneUtils — clone/detach helpers. We expose the common subset.
export const SceneUtils = {
    createMeshesFromInstancedMesh(_im) { return new Group(); },
    createMeshesFromMultiMaterialMesh(mesh) { return [mesh]; },
    sortInstancedMesh(_im, _cmp) {},
    cloneMaterials(materials) { return materials.map(m => m.clone?.() || m); },
};

// SkeletonUtils — bone-tree clone for sharing skeletons.
export const SkeletonUtils = {
    clone(source) {
        const cloneLookup = new Map();
        const clonedRoot = _cloneObject(source, cloneLookup);
        return clonedRoot;
    },
    retargetClip(_target, _source, clip) { return clip; },
    retarget(_target, _source) {},
};
function _cloneObject(obj, lookup) {
    let copy;
    if (obj instanceof Mesh) copy = new Mesh(obj.geometry, obj.material);
    else if (obj instanceof Group) copy = new Group();
    else copy = new Object3D();
    copy.name = obj.name;
    if (obj.position) copy.position?.copy?.(obj.position);
    if (obj.rotation) copy.rotation?.copy?.(obj.rotation);
    if (obj.scale)    copy.scale?.copy?.(obj.scale);
    lookup.set(obj, copy);
    for (const c of (obj.children || obj._children || [])) {
        const cc = _cloneObject(c, lookup);
        if (copy.add) copy.add(cc);
    }
    return copy;
}

// RoomEnvironment — three.js's neutral studio room, used as
// `scene.environment = pmrem.fromScene(new RoomEnvironment()).texture`. That
// call is what matters, and PMREMGenerator.fromScene answers it with the exact
// room from wasm (`extras::RoomEnvironment`: the boxes, the point light, the
// six emissive panels, in HDR). The scene below is only what renders if the
// room is drawn directly, which nothing does.
export class RoomEnvironment extends Scene {
    constructor() {
        super();
        this.name = 'RoomEnvironment';
        this.isRoomEnvironment = true;
        this.background = new Color(0xbbbbbb);
        this.add(new AmbientLight(0xffffff, 0.4));
        const dl = new DirectionalLight(0xffffff, 0.6);
        dl.position.set(2, 3, 4);
        this.add(dl);
    }
}

// ---- Loaders ----

// MTLLoader — text material library used by OBJLoader. Returns a small
// dictionary of materials keyed by name.
export class MTLLoader extends _Loader {
    load(url, onLoad, _onProgress, onError) {
        new FileLoader(this.manager).load(url, (text) => {
            try { onLoad?.(this.parse(text, this.path)); } catch (e) { onError?.(e); }
        }, undefined, onError);
    }
    parse(text, _path) {
        const materials = {};
        let current = null;
        for (const raw of text.split(/\r?\n/)) {
            const line = raw.trim();
            if (!line || line.startsWith('#')) continue;
            const [tok, ...args] = line.split(/\s+/);
            if (tok === 'newmtl') {
                current = { name: args[0], color: 0xffffff };
                materials[args[0]] = current;
            } else if (current && tok === 'Kd') {
                const r = +args[0] || 0, g = +args[1] || 0, b = +args[2] || 0;
                current.color = (Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255);
            } else if (current && tok === 'Ks') {
                current.specular = { r: +args[0], g: +args[1], b: +args[2] };
            } else if (current && tok === 'Ns') {
                current.shininess = +args[0];
            } else if (current && tok === 'd') {
                current.opacity = +args[0];
            } else if (current && tok === 'map_Kd') {
                current.mapKd = args.join(' ');
            }
        }
        return {
            materials,
            preload() { return this; },
            getAsArray() { return Object.values(materials); },
            create(name) {
                const m = materials[name];
                if (!m) return new MeshStandardMaterial();
                const out = new MeshStandardMaterial({ color: m.color, opacity: m.opacity ?? 1, transparent: (m.opacity ?? 1) < 1 });
                if (m.mapKd) {
                    new TextureLoader().load(m.mapKd, (tex) => { out.map = tex; });
                }
                return out;
            },
        };
    }
}

// ---- Exporters ----
// three.js r165's OBJExporter / STLExporter / PLYExporter / GLTFExporter,
// written in Rust (`WebSceneExporter`, over `src/exporters` and
// `src/loaders/gltf_write.rs`). The JS side only walks the scene graph — its
// objects live here — handing each object's geometry, transforms and material
// over in `traverse` order. OBJ, STL and PLY come out byte for byte as
// three.js writes them. threers extension: OBJExporter and PLYExporter take
// `{ polygons: true }` to write a geometry's polygons (`BufferGeometry.polygons`,
// e.g. from `inferQuads` or `subdivide`) as n-gon faces.

/// An object's rotation as `[x, y, z, w]` (its quaternion, which its Euler
/// rotation is kept in step with, as in three.js).
function _exportQuaternion(o) {
    const q = o.quaternion;
    return q ? [q.x, q.y, q.z, q.w] : [0, 0, 0, 1];
}
/// An object's world matrix as three.js's `updateWorldMatrix(true, false)`
/// leaves it, in f64.
function _exportWorldMatrix(o) {
    if (typeof o.updateWorldMatrix === 'function') {
        o.updateWorldMatrix(true, false);
        return o.matrixWorld.elements;
    }
    return new Matrix4().elements;
}
function _exportKind(o) {
    if (o instanceof LineLoop) return 'lineLoop';
    if (o instanceof LineSegments) return 'lineSegments';
    if (o instanceof Line) return 'line';
    if (o instanceof Points || o._isPoints) return 'points';
    if (o.isMesh === true || o instanceof Mesh) return 'mesh';
    return 'node';
}
/// Feed `root` and its descendants (all of them, or the visible ones) to a
/// `WebSceneExporter`. `skipRoot` hands over only the children — a Scene, which
/// glTF writes as `scenes[0]` rather than as a node.
function _exportScene(roots, options) {
    const visibleOnly = !!options?.visibleOnly, skipRoot = !!options?.skipRoot;
    const ex = new WebSceneExporter();
    const materials = new Map();
    const material = (m) => {
        if (!m) return -1;
        if (materials.has(m)) return materials.get(m);
        const kind = (m instanceof MeshStandardMaterial || m instanceof MeshPhysicalMaterial) ? 'standard'
            : m instanceof MeshBasicMaterial ? 'basic' : 'other';
        const c = m.color ?? { r: 1, g: 1, b: 1 };
        const e = m.emissive ?? { r: 0, g: 0, b: 0 };
        const id = ex.addMaterial(kind, String(m.name ?? ''), new Float32Array([c.r, c.g, c.b]),
            m.opacity ?? 1, !!m.transparent, m.metalness ?? 0, m.roughness ?? 1,
            new Float32Array([e.r, e.g, e.b]), m.side ?? 0, m.alphaTest ?? 0);
        materials.set(m, id);
        return id;
    };
    const visit = (o, parent) => {
        if (visibleOnly && o.visible === false) return;
        const kind = _exportKind(o);
        const p = o.position ?? { x: 0, y: 0, z: 0 };
        const q = _exportQuaternion(o);
        const s = o.scale ?? { x: 1, y: 1, z: 1 };
        const index = ex.addObject(parent, String(o.name ?? ''), kind,
            new Float64Array([p.x, p.y, p.z]), new Float64Array(q),
            new Float64Array([s.x, s.y, s.z]), new Float64Array(_exportWorldMatrix(o)));
        if (kind !== 'node') {
            const g = o.geometry;
            if (g?._isUserGeometry || !g?._w) {
                const w = g?.attributes?.position && typeof g._syncWasmFromJs === 'function'
                    ? g._syncWasmFromJs() : new WebBufferGeometry();
                ex.setBufferGeometry(index, w);
            } else if (g._w instanceof WebGeometry) {
                ex.setGeometry(index, g._w);
            } else if (g._w instanceof WebBufferGeometry) {
                ex.setBufferGeometry(index, g._w);
            }
            const m = material(Array.isArray(o.material) ? o.material[0] : o.material);
            if (m >= 0) ex.setMaterial(index, m);
        }
        for (const c of o.children ?? []) visit(c, index);
    };
    for (const r of roots) {
        if (skipRoot) for (const c of r.children ?? []) visit(c, -1);
        else visit(r, -1);
    }
    return ex;
}
export class OBJExporter {
    parse(object, options = {}) {
        const ex = _exportScene([object]);
        try { return ex.obj(!!options?.polygons); } finally { ex.free(); }
    }
}
export class STLExporter {
    parse(scene, options = {}) {
        const ex = _exportScene([scene]);
        try {
            if (options?.binary === true) return new DataView(new Uint8Array(ex.stlBinary()).buffer);
            return ex.stlAscii();
        } finally { ex.free(); }
    }
}
export class PLYExporter {
    parse(object, onDone, options = {}) {
        const o = { binary: false, excludeAttributes: [], littleEndian: false, ...options };
        const ex = _exportScene([object]);
        let result = null;
        try {
            const bytes = ex.ply(!!o.binary, !!o.littleEndian, o.excludeAttributes.map(String), !!o.polygons);
            if (bytes === undefined) {
                console.error('PLYExporter: Failed to generate a valid PLY file with triangle indices because the ' +
                    'number of indices is not divisible by 3.');
                return null;
            }
            result = o.binary ? new Uint8Array(bytes).buffer : new TextDecoder().decode(bytes);
        } finally { ex.free(); }
        if (typeof onDone === 'function') {
            const later = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : (f) => setTimeout(f, 0);
            later(() => onDone(result));
        }
        return result;
    }
}
export class GLTFExporter {
    constructor() { this.pluginCallbacks = []; }
    register(callback) { this.pluginCallbacks.push(callback); return this; }
    unregister(callback) { this.pluginCallbacks = this.pluginCallbacks.filter((c) => c !== callback); return this; }
    /// `parse(input, onDone, onError, options)`: `input` an object, a Scene or
    /// an array of them; `options.binary` gives a GLB `ArrayBuffer`, otherwise
    /// the glTF JSON object with its buffer inline. Honours `trs` and
    /// `onlyVisible` (default true).
    parse(input, onDone, onError, options = {}) {
        const o = { binary: false, trs: false, onlyVisible: true, ...options };
        try {
            const inputs = Array.isArray(input) ? input : [input];
            const scenes = inputs.filter((i) => i instanceof Scene);
            const others = inputs.filter((i) => !(i instanceof Scene));
            // three.js writes one scene; objects outside a Scene go into an "AuxScene".
            const ex = scenes.length
                ? _exportScene(scenes.slice(0, 1), { visibleOnly: o.onlyVisible, skipRoot: true })
                : _exportScene(others, { visibleOnly: o.onlyVisible });
            const name = scenes.length ? (scenes[0].name || undefined) : 'AuxScene';
            let result;
            try {
                result = o.binary ? new Uint8Array(ex.glb(!!o.trs, name)).buffer : JSON.parse(ex.gltf(!!o.trs, name));
            } finally { ex.free(); }
            onDone?.(result);
            return result;
        } catch (e) {
            if (typeof onError === 'function') { onError(e); return undefined; }
            throw e;
        }
    }
    parseAsync(input, options) {
        return new Promise((resolve, reject) => this.parse(input, resolve, reject, options));
    }
}

// ---- Surface objects (`examples/jsm/objects/...`) ----
// These typically require render-to-texture for proper effects. We expose
// constructable wrappers that render the supplied geometry with a basic
// material so they don't crash imports.
// Refractor / Water — see real implementations below (they extend Reflector's
// mirror-RT pipeline). Stubs removed; the actual classes live next to Reflector.
// Real Sky. Atmospheric scattering shader via the Preetham model (MAT_SKY in
// our WGSL). Matches three.js's `examples/jsm/objects/Sky.js`.
class _SkyMaterialWrapper {
    constructor() {
        this._w = WebMaterial.sky(0, 1, 0, 10, 3, 0.005, 0.7);
        this._w.setSide(1); // BackSide — inside the dome.
        const sunPosition = new Vector3(0, 1, 0);
        const self = this;
        const sync = () => self.updateUniforms();
        const origSet = sunPosition.set.bind(sunPosition);
        sunPosition.set = (x, y, z) => { origSet(x, y, z); sync(); return sunPosition; };
        const origCopy = sunPosition.copy.bind(sunPosition);
        sunPosition.copy = (v) => { origCopy(v); sync(); return sunPosition; };
        this.uniforms = {
            turbidity: { value: 10 },
            rayleigh: { value: 3 },
            mieCoefficient: { value: 0.005 },
            mieDirectionalG: { value: 0.7 },
            sunPosition: { value: sunPosition },
            up: { value: new Vector3(0, 1, 0) },
        };
        _initMaterialBase(this, { side: 1, depthWrite: false });
    }
    // When the user mutates uniforms.sunPosition.value or similar, three.js
    // re-uploads to the GPU. Mirror this by re-creating the wasm material
    // whenever the user calls .updateUniforms().
    updateUniforms() {
        const sp = this.uniforms.sunPosition.value;
        this._w = WebMaterial.sky(sp.x, sp.y, sp.z,
            this.uniforms.turbidity.value,
            this.uniforms.rayleigh.value,
            this.uniforms.mieCoefficient.value,
            this.uniforms.mieDirectionalG.value);
        this._w.setSide(1);
    }
}
export class Sky extends Mesh {
    constructor() {
        // Unit box — matches three.js Sky (BackSide dome); user scales to ~450000.
        const g = new BoxGeometry(1, 1, 1);
        super(g, new _SkyMaterialWrapper());
        this.type = 'Sky';
        this.frustumCulled = false;
        this.renderOrder = -Infinity;
    }
}

// Reflector — mirror surface. Reflects the main camera across the reflector's
// plane each frame, renders the scene to a render target from that virtual
// camera, then projectively samples the RT in its fragment shader. The
// fragment math matches three.js Reflector.js exactly (scaleBias * P_v * V_v
// * world_pos → projUV; perspective-divide → UV sample). Color tints the
// reflection via the same `blendOverlay` semantics — passing 0x7f7f7f leaves
// the sample unchanged.
// Oblique near clip for mirror RT renders (three.js Reflector.js / Lengyel 2007).
function _applyObliqueNearPlane(P, normal, planePoint, viewMatrix, clipBias = 0) {
    const e = viewMatrix.elements;
    const px = normal.x, py = normal.y, pz = normal.z;
    const pw = -(px * planePoint.x + py * planePoint.y + pz * planePoint.z);
    const clip = {
        x: e[0] * px + e[4] * py + e[8] * pz + e[12] * pw,
        y: e[1] * px + e[5] * py + e[9] * pz + e[13] * pw,
        z: e[2] * px + e[6] * py + e[10] * pz + e[14] * pw,
        w: e[3] * px + e[7] * py + e[11] * pz + e[15] * pw,
    };
    const q = {
        x: (Math.sign(clip.x) + P[8]) / P[0],
        y: (Math.sign(clip.y) + P[9]) / P[5],
        z: -1.0,
        w: (1.0 + P[10]) / P[14],
    };
    const dot = clip.x * q.x + clip.y * q.y + clip.z * q.z + clip.w * q.w;
    const scale = 2.0 / dot;
    clip.x *= scale; clip.y *= scale; clip.z *= scale; clip.w *= scale;
    P[2] = clip.x;
    P[6] = clip.y;
    P[10] = clip.z + 1.0 - clipBias;
    P[14] = clip.w;
    return P;
}

export class Reflector extends Mesh {
    constructor(geometry, options = {}) {
        const colorIn = options.color !== undefined ? new Color(options.color) : new Color(0x7f7f7f);
        const textureWidth  = options.textureWidth  || 512;
        const textureHeight = options.textureHeight || 512;
        // Material placeholder; replaced with the mirror wasm material on first onBeforeRender.
        super(geometry, new MeshBasicMaterial({ color: 0xffffff }));
        this.isReflector = true;
        this.type = 'Reflector';
        this._reflectorColor = colorIn;
        this._reflectorOptions = { textureWidth, textureHeight };
        this._mirrorRT = null;
        this._mirrorMaterialW = null;
        this._virtualCam = null;
    }
    /// The mirror material, once made: what the node is drawn with.
    _wasmMaterialOverride() { return this._mirrorMaterialW ?? null; }
    _mirrorSetup(renderer, camera) {
        if (!this._mirrorRT) {
            this._mirrorRT = new WebRenderTarget(renderer._w, this._reflectorOptions.textureWidth, this._reflectorOptions.textureHeight);
            const tex = renderer._w.renderTargetTexture(this._mirrorRT);
            const c = this._reflectorColor;
            this._mirrorMaterialW = WebMaterial.mirror(c.r, c.g, c.b);
            this._mirrorMaterialW.setMap(tex);
            this._dirty(_D_MATERIAL);
        }
        // Same parameters as the main camera, so the projective UV in the
        // shader lines up with the RT contents.
        if (!this._virtualCam) this._virtualCam = new PerspectiveCamera(50, camera.aspect || 1, 0.1, 2000);
        this._virtualCam._w.copyProjection(camera._w);
        this._virtualCam.far = camera.far;
    }
    /// Texture matrix (scaleBias · P · V · matrixWorld) for the mirror
    /// material, then the scene drawn into the RT from the virtual camera with
    /// this surface hidden.
    _mirrorRender(renderer, scene, normal, planePoint, oblique) {
        const P_arr = Array.from(this._virtualCam._w.projectionMatrix());
        const V = new Matrix4().fromArray(Array.from(this._virtualCam._w.viewMatrix()));
        const P = new Matrix4().fromArray(P_arr);
        const bias = new Matrix4().set(
            0.5, 0, 0, 0.5,
            0, 0.5, 0, 0.5,
            0, 0, 0.5, 0.5,
            0, 0, 0, 1
        );
        const M = new Matrix4().multiplyMatrices(bias, P).multiply(V).multiply(this.matrixWorld);
        this._mirrorMaterialW.setTextureMatrix(M.elements);
        // `setTextureMatrix` gave the wasm material a new state: re-send it.
        this._dirty(_D_MATERIAL);
        if (oblique) {
            const P_render = P_arr.slice();
            _applyObliqueNearPlane(P_render, normal, planePoint, V, -0.001);
            this._virtualCam._w.setProjectionOverride(P_render);
        }
        const wasVisible = this.visible;
        this.visible = false;
        scene._syncNow();
        renderer._w.setRenderTarget(this._mirrorRT.id);
        renderer._w.render(scene._w, this._virtualCam._w);
        renderer._w.setRenderTarget(0);
        if (oblique) this._virtualCam._w.clearProjectionOverride();
        this.visible = wasVisible;
    }
    /// three.js Reflector.js's virtual camera: the main camera reflected in
    /// the surface's plane (local +Z), looking at the reflected target.
    onBeforeRender(renderer, scene, camera) {
        this._mirrorSetup(renderer, camera);
        this.updateWorldMatrix(true, false);
        camera.updateWorldMatrix(true, false);
        const reflectorPos = new Vector3().setFromMatrixPosition(this.matrixWorld);
        const camPos = new Vector3().setFromMatrixPosition(camera.matrixWorld);
        const rot = new Matrix4().extractRotation(this.matrixWorld);
        const normal = new Vector3(0, 0, 1).applyMatrix4(rot);
        const view = new Vector3().subVectors(reflectorPos, camPos);
        // Facing away: the reflection is not visible.
        if (view.dot(normal) > 0) return;
        const reflect = (v) => v.sub(new Vector3().copy(normal).multiplyScalar(2 * v.dot(normal)));
        reflect(view).negate();
        view.add(reflectorPos);
        const camRot = new Matrix4().extractRotation(camera.matrixWorld);
        const lookAtPosition = new Vector3(0, 0, -1).applyMatrix4(camRot).add(camPos);
        const target = new Vector3().subVectors(reflectorPos, lookAtPosition);
        reflect(target).negate();
        target.add(reflectorPos);
        const vc = this._virtualCam;
        vc.position.copy(view);
        vc.up.set(0, 1, 0).applyMatrix4(camRot);
        reflect(vc.up);
        vc.lookAt(target);
        vc._w.setAspect(camera.aspect || 1);
        vc._sync();
        this._mirrorRender(renderer, scene, normal, reflectorPos, true);
    }
}

// Refractor — like Reflector, but the virtual camera is positioned on the
// FAR side of the surface (where refracted rays land) rather than mirrored.
// Same projective UV machinery; the difference is in the virtual camera math.
// For thin surfaces with low IOR this approximates the view "through" the
// surface — three.js's Refractor uses the same trick.
export class Refractor extends Reflector {
    constructor(geometry, options = {}) {
        super(geometry, options);
        this.type = 'Refractor';
        this._refractorIor = options.ior || 1.5;
    }
    /// The virtual camera is the main camera (the surface is thin): what is
    /// behind the surface, projected onto it.
    onBeforeRender(renderer, scene, camera) {
        this._mirrorSetup(renderer, camera);
        this.updateWorldMatrix(true, false);
        camera.updateWorldMatrix(true, false);
        const vc = this._virtualCam;
        vc.position.setFromMatrixPosition(camera.matrixWorld);
        vc.quaternion.setFromRotationMatrix(new Matrix4().extractRotation(camera.matrixWorld));
        vc._sync();
        this._mirrorRender(renderer, scene, null, null, false);
    }
}

// Water — Reflector with an animated normal-map distortion + a Fresnel-tinted
// water color. Three.js's Water uses Reflector under the hood; ours does the
// same. The waterColor tints the reflection; time-driven distortion isn't
// applied in the current pass (deferred to a follow-up turn — the static
// reflection still produces a recognizable water-surface fixture).
export class Water extends Reflector {
    constructor(geometry, options = {}) {
        const waterOpts = {
            ...options,
            color: options.waterColor !== undefined ? options.waterColor : 0x001e0f,
            textureWidth:  options.textureWidth  || 512,
            textureHeight: options.textureHeight || 512,
        };
        super(geometry, waterOpts);
        this.type = 'Water';
        this.material.uniforms = {
            time:        { value: 0 },
            waterColor:  { value: new Color(waterOpts.color) },
            sunColor:    { value: new Color(options.sunColor || 0xffffff) },
            sunDirection:{ value: options.sunDirection || new Vector3(0.7, 0.7, 0) },
            distortionScale: { value: options.distortionScale || 3.7 },
            size:        { value: options.size || 1.0 },
        };
    }
}

// Lensflare — additive sprite stack at a light position. Render as plain Sprite.
export class Lensflare extends Sprite {
    constructor() { super(new SpriteMaterial({ color: 0xffffff })); this.type = 'Lensflare'; this.elements = []; }
    addElement(_el) { this.elements.push(_el); }
}
Lensflare.LensflareElement = class LensflareElement { constructor(texture, size = 1, distance = 0, color) { this.texture = texture; this.size = size; this.distance = distance; this.color = color || new Color(0xffffff); } };

// ---- VR/AR (`examples/jsm/webxr/...`) ----
// Browser API access — the buttons trigger a `navigator.xr.requestSession`,
// which falls back to a no-op when XR isn't available.
export class VRButton {
    static createButton(_renderer, _options) {
        const btn = (typeof document !== 'undefined') ? document.createElement('button') : { textContent: '' };
        btn.textContent = 'ENTER VR';
        btn.onclick = () => navigator.xr?.requestSession?.('immersive-vr').catch(() => {});
        return btn;
    }
}
export class ARButton {
    static createButton(_renderer, _options) {
        const btn = (typeof document !== 'undefined') ? document.createElement('button') : { textContent: '' };
        btn.textContent = 'START AR';
        btn.onclick = () => navigator.xr?.requestSession?.('immersive-ar').catch(() => {});
        return btn;
    }
}
export class XRControllerModelFactory { constructor() {} createControllerModel(_controller) { return new Group(); } }
export class XRHandModelFactory { constructor() {} createHandModel(_hand) { return new Group(); } }

// ---- CSS3DRenderer companion ----
export class CSS3DObject extends Object3D { constructor(element) { super(); this.element = element; this._isCSS3DObject = true; } }
export class CSS3DSprite extends CSS3DObject { constructor(el) { super(el); this._isCSS3DSprite = true; } }
export class CSS3DRenderer extends CSS2DRenderer {}

// MeshoptDecoder — loads meshoptimizer WASM decoder on first use.
let _meshoptImpl = null;
export const MeshoptDecoder = {
    ready: (async () => {
        try {
            const mod = await import('./deps/meshopt_decoder.module.js');
            await mod.default.ready;
            _meshoptImpl = mod.default;
            MeshoptDecoder.supported = true;
        } catch (_) {
            MeshoptDecoder.supported = false;
        }
        return MeshoptDecoder;
    })(),
    supported: true,
    decodeGltfBuffer(target, count, stride, source, mode, filter) {
        if (!_meshoptImpl) throw new Error('MeshoptDecoder not ready — await MeshoptDecoder.ready');
        return _meshoptImpl.decodeGltfBuffer(target, count, stride, source, mode, filter);
    },
};

// ---- THREE namespace ----
const THREE = {
    // Core
    Scene, Group, Object3D, BufferGeometry, BufferAttribute, LineSegments, Line, LineLoop, Points,
    Sprite, InstancedMesh, SkinnedMesh, Skeleton, Bone,
    PerspectiveCamera, OrthographicCamera,
    Mesh, Color, Vector2, Vector3, Vector4, Euler, Quaternion,
    Matrix3, Matrix4,
    Box2, Box3, Sphere, Ray, Plane, Triangle, Frustum,
    Spherical, Cylindrical, Line3,
    // Geometries
    BoxGeometry, SphereGeometry, PlaneGeometry, CylinderGeometry, TorusGeometry,
    CircleGeometry, RingGeometry, ConeGeometry, TorusKnotGeometry, CapsuleGeometry,
    TetrahedronGeometry, OctahedronGeometry, IcosahedronGeometry, DodecahedronGeometry,
    BoxLineGeometry, LatheGeometry, TubeGeometry, ExtrudeGeometry, EdgesGeometry,
    WireframeGeometry, ShapeGeometry, ConvexGeometry, DecalGeometry, TextGeometry,
    ParametricGeometry,
    QuadGeometry, // threers extension
    // Materials
    MeshBasicMaterial, MeshLambertMaterial, MeshStandardMaterial,
    MeshPhongMaterial, MeshPhysicalMaterial, MeshNormalMaterial, MeshDepthMaterial,
    MeshToonMaterial, LineBasicMaterial, PointsMaterial, SpriteMaterial,
    AtmosphereMaterial,
    // Lights
    AmbientLight, DirectionalLight, PointLight, SpotLight, HemisphereLight, RectAreaLight,
    // Textures
    Texture, CubeTexture, DataTexture, CanvasTexture,
    // Curves
    LineCurve, LineCurve3, EllipseCurve, CatmullRomCurve3, Path, Shape,
    QuadraticBezierCurve, CubicBezierCurve, QuadraticBezierCurve3, CubicBezierCurve3,
    SplineCurve, ArcCurve,
    // Animation
    AnimationClip, AnimationMixer, KeyframeTrack,
    BooleanKeyframeTrack, NumberKeyframeTrack, ColorKeyframeTrack, QuaternionKeyframeTrack,
    VectorKeyframeTrack, StringKeyframeTrack,
    AnimationObjectGroup, AnimationUtils,
    // Controls
    OrbitControls, TrackballControls, FirstPersonControls, PointerLockControls,
    DragControls, ArcballControls, MapControls, FlyControls, TransformControls,
    TransformControlsGizmo, TransformControlsPlane, TOUCH,
    CameraAnimator, CameraPath, ShotTimeline, speedRamp,
    Tween, Spring, Timeline, ObjectSpring, PoseBlend, TrackModifier,
    PropertyBinding, PropertyMixer, AnimationAction,
    // Loaders
    OBJLoader, STLLoader, PLYLoader, RGBELoader, FileLoader, ImageLoader, TextureLoader,
    FontLoader, Font, MaterialLoader, BufferGeometryLoader, ObjectLoader, AnimationLoader,
    AudioLoader, CubeTextureLoader, GLTFLoader, FBXLoader, ColladaLoader, EXRLoader,
    KTX2Loader, DRACOLoader, LoadingManager, DefaultLoadingManager,
    // Audio
    AudioListener, Audio,
    // Post-fx
    EffectComposer, RenderPass, UnrealBloomPass, FXAAShader,
    OutlinePass, SSAOPass, SSRPass, GlitchPass, FilmPass, DotScreenPass, HalftonePass,
    seedRandom, unseedRandom,
    ShaderPass, TexturePass, CubeTexturePass, MaskPass, ClearMaskPass, ClearPass, CopyShader,
    FilmShader, DotScreenShader, HalftoneShader, DigitalGlitch, LuminosityShader,
    LuminosityHighPassShader, OutputShader,
    // Helpers
    AxesHelper, GridHelper, BoxHelper, PolarGridHelper,
    DirectionalLightHelper, HemisphereLightHelper, PointLightHelper, SpotLightHelper,
    CameraHelper, ArrowHelper, PlaneHelper, SkeletonHelper,
    QuadView, // threers extension
    Fur, // threers extension
    Hair, // threers extension
    Particles, // threers extension
    SnowCover, // threers extension
    SandBed, // threers extension
    Weather, // threers extension
    DesertSky, // threers extension
    Spindrift, // threers extension
    desertTime, // threers extension
    // Extras
    SimplexNoise, Octree, MarchingCubes,
    CSS2DRenderer, SVGRenderer,
    Stats, Raycaster, Clock,
    EventDispatcher,
    WebGLRenderTarget, WebGLCubeRenderTarget, WebGL3DRenderTarget,
    WebGLRenderer, WebGPURenderer,
    // Bases + variants added for full coverage
    Camera, Material, Light, LightShadow, LightProbe, Curve, CurvePath, Loader,
    Interpolant, LinearInterpolant, CubicInterpolant, DiscreteInterpolant, QuaternionLinearInterpolant,
    ArrayCamera, StereoCamera, CubeCamera,
    LineDashedMaterial, MeshMatcapMaterial, MeshDistanceMaterial,
    ShadowMaterial, ShaderMaterial, RawShaderMaterial,
    DepthTexture, VideoTexture, CompressedTexture, CompressedArrayTexture, CompressedCubeTexture,
    Data3DTexture, DataArrayTexture, FramebufferTexture, Source,
    Int8BufferAttribute, Int16BufferAttribute, Int32BufferAttribute,
    Uint8BufferAttribute, Uint8ClampedBufferAttribute, Uint16BufferAttribute, Uint32BufferAttribute,
    Float16BufferAttribute, Float32BufferAttribute, Float64BufferAttribute,
    InstancedBufferAttribute, InterleavedBuffer, InterleavedBufferAttribute,
    InstancedInterleavedBuffer, GLBufferAttribute, InstancedBufferGeometry,
    AmbientLightProbe, HemisphereLightProbe,
    DirectionalLightShadow, PointLightShadow, SpotLightShadow,
    BatchedMesh, LOD,
    Box3Helper, VertexNormalsHelper, RectAreaLightHelper,
    RectAreaLightUniformsLib, RectAreaLightTexturesLib,
    LoaderUtils, DataTextureLoader, CompressedTextureLoader, ImageBitmapLoader,
    Cache, Fog, FogExp2, Layers, SphericalHarmonics3,
    Uniform, UniformsGroup, UniformsLib, UniformsUtils, ShaderChunk, ShaderLib,
    ImageUtils, DataUtils, ShapeUtils,
    AudioContext, AudioAnalyser, PositionalAudio,
    PMREMGenerator,
    // Utilities
    MathUtils, REVISION, ColorManagement,
    // ---- constants (three.js compatibility shims) ----
    // Pixel formats
    RGBAFormat: 1023, RGBFormat: 1022, RGFormat: 1030, RedFormat: 1028,
    LuminanceFormat: 1024, LuminanceAlphaFormat: 1025, AlphaFormat: 1021,
    DepthFormat: 1026, DepthStencilFormat: 1027,
    RedIntegerFormat: 1029, RGIntegerFormat: 1031, RGBIntegerFormat: 1032, RGBAIntegerFormat: 1033,
    // Texel types
    UnsignedByteType: 1009, ByteType: 1010, ShortType: 1011, UnsignedShortType: 1012,
    IntType: 1013, UnsignedIntType: 1014, FloatType: 1015, HalfFloatType: 1016,
    UnsignedShort4444Type: 1017, UnsignedShort5551Type: 1018,
    UnsignedInt248Type: 1019, UnsignedInt5999Type: 35902,
    // Filters
    NearestFilter: 1003, LinearFilter: 1006,
    NearestMipmapNearestFilter: 1004, LinearMipmapNearestFilter: 1007,
    NearestMipmapLinearFilter: 1005, LinearMipmapLinearFilter: 1008,
    // Wrapping
    RepeatWrapping: 1000, ClampToEdgeWrapping: 1001, MirroredRepeatWrapping: 1002,
    // Sides
    FrontSide: 0, BackSide: 1, DoubleSide: 2,
    // Blending
    NoBlending: 0, NormalBlending: 1, AdditiveBlending: 2, SubtractiveBlending: 3,
    MultiplyBlending: 4, CustomBlending: 5,
    // Blend equations
    AddEquation: 100, SubtractEquation: 101, ReverseSubtractEquation: 102,
    MinEquation: 103, MaxEquation: 104,
    // Blend factors
    ZeroFactor: 200, OneFactor: 201, SrcColorFactor: 202, OneMinusSrcColorFactor: 203,
    SrcAlphaFactor: 204, OneMinusSrcAlphaFactor: 205, DstAlphaFactor: 206, OneMinusDstAlphaFactor: 207,
    DstColorFactor: 208, OneMinusDstColorFactor: 209, SrcAlphaSaturateFactor: 210,
    // Depth funcs
    NeverDepth: 0, AlwaysDepth: 1, LessDepth: 2, LessEqualDepth: 3,
    EqualDepth: 4, GreaterEqualDepth: 5, GreaterDepth: 6, NotEqualDepth: 7,
    // Stencil ops
    KeepStencilOp: 7680, ZeroStencilOp: 0, ReplaceStencilOp: 7681, IncrementStencilOp: 7682,
    DecrementStencilOp: 7283, IncrementWrapStencilOp: 34055, DecrementWrapStencilOp: 34056, InvertStencilOp: 5386,
    // Stencil funcs
    NeverStencilFunc: 512, AlwaysStencilFunc: 519, LessStencilFunc: 513, LessEqualStencilFunc: 515,
    EqualStencilFunc: 514, GreaterEqualStencilFunc: 518, GreaterStencilFunc: 516, NotEqualStencilFunc: 517,
    // Color spaces
    SRGBColorSpace: 'srgb', LinearSRGBColorSpace: 'srgb-linear',
    DisplayP3ColorSpace: 'display-p3', LinearDisplayP3ColorSpace: 'display-p3-linear',
    NoColorSpace: '',
    // Tone mapping
    NoToneMapping: 0, LinearToneMapping: 1, ReinhardToneMapping: 2,
    CineonToneMapping: 3, ACESFilmicToneMapping: 4, AgXToneMapping: 6, NeutralToneMapping: 7,
    // Mappings (env map / refraction)
    UVMapping: 300, CubeReflectionMapping: 301, CubeRefractionMapping: 302,
    EquirectangularReflectionMapping: 303, EquirectangularRefractionMapping: 304,
    CubeUVReflectionMapping: 306,
    // Depth packing
    BasicDepthPacking: 3200, RGBADepthPacking: 3201,
    // Texture encoding (pre-r155 names, kept for backwards compat)
    LinearEncoding: 3000, sRGBEncoding: 3001,
    // Constants for objects
    DefaultUp: { x: 0, y: 1, z: 0 },
    // Loop modes
    LoopOnce: 2200, LoopRepeat: 2201, LoopPingPong: 2202,
    // Interpolation modes
    InterpolateDiscrete: 2300, InterpolateLinear: 2301, InterpolateSmooth: 2302,
    // Ending modes
    ZeroCurvatureEnding: 2400, ZeroSlopeEnding: 2401, WrapAroundEnding: 2402,
    // Triangle drawing
    TrianglesDrawMode: 0, TriangleStripDrawMode: 1, TriangleFanDrawMode: 2,
    // (the old Layers object stub is superseded by the proper Layers class below)
    // Animation blend modes
    NormalAnimationBlendMode: 2500, AdditiveAnimationBlendMode: 2501,
    AttachedBindMode: 'attached', DetachedBindMode: 'detached',
    // Shadow map types
    BasicShadowMap: 0, PCFShadowMap: 1, PCFSoftShadowMap: 2, VSMShadowMap: 3,
    // Cull face modes
    CullFaceNone: 0, CullFaceBack: 1, CullFaceFront: 2, CullFaceFrontBack: 3,
    // Compare modes (depth/sampler)
    NeverCompare: 512, LessCompare: 513, EqualCompare: 514, LessEqualCompare: 515,
    GreaterCompare: 516, NotEqualCompare: 517, GreaterEqualCompare: 518, AlwaysCompare: 519,
    // Buffer usage hints
    StaticDrawUsage: 35044, DynamicDrawUsage: 35048, StreamDrawUsage: 35040,
    StaticReadUsage: 35045, DynamicReadUsage: 35049, StreamReadUsage: 35041,
    StaticCopyUsage: 35046, DynamicCopyUsage: 35050, StreamCopyUsage: 35042,
    // GLSL versions
    GLSL1: '100', GLSL3: '300 es',
    // Normal map types
    TangentSpaceNormalMap: 0, ObjectSpaceNormalMap: 1,
    // Transfer (color)
    LinearTransfer: 'linear', SRGBTransfer: 'srgb',
    // Compressed format codes — three.js exposes constants, our impl ignores them
    RGB_S3TC_DXT1_Format: 33776, RGBA_S3TC_DXT1_Format: 33777,
    RGBA_S3TC_DXT3_Format: 33778, RGBA_S3TC_DXT5_Format: 33779,
    RGB_PVRTC_4BPPV1_Format: 35840, RGB_PVRTC_2BPPV1_Format: 35841,
    RGBA_PVRTC_4BPPV1_Format: 35842, RGBA_PVRTC_2BPPV1_Format: 35843,
    RGB_ETC1_Format: 36196, RGB_ETC2_Format: 37492, RGBA_ETC2_EAC_Format: 37496,
    RGBA_ASTC_4x4_Format: 37808, RGBA_ASTC_5x4_Format: 37809, RGBA_ASTC_5x5_Format: 37810,
    RGBA_ASTC_6x5_Format: 37811, RGBA_ASTC_6x6_Format: 37812, RGBA_ASTC_8x5_Format: 37813,
    RGBA_ASTC_8x6_Format: 37814, RGBA_ASTC_8x8_Format: 37815, RGBA_ASTC_10x5_Format: 37816,
    RGBA_ASTC_10x6_Format: 37817, RGBA_ASTC_10x8_Format: 37818, RGBA_ASTC_10x10_Format: 37819,
    RGBA_ASTC_12x10_Format: 37820, RGBA_ASTC_12x12_Format: 37821,
    RGB_BPTC_SIGNED_Format: 36494, RGB_BPTC_UNSIGNED_Format: 36495,
    RGBA_BPTC_Format: 36492, RED_RGTC1_Format: 36283, SIGNED_RED_RGTC1_Format: 36284,
    RED_GREEN_RGTC2_Format: 36285, SIGNED_RED_GREEN_RGTC2_Format: 36286,
    // ---- examples/jsm add-ons (not in canonical core, but commonly imported) ----
    OBB, Capsule, ImprovedNoise, ConvexHull,
    RoundedBoxGeometry, TeapotGeometry, ParametricGeometries,
    LineMaterial, LineGeometry, LineSegmentsGeometry, Line2, LineSegments2, Wireframe,
    BufferGeometryUtils, geometryToBufferGeometry, SceneUtils, SkeletonUtils, RoomEnvironment,
    MTLLoader, OBJExporter, STLExporter, PLYExporter, GLTFExporter, MeshoptDecoder,
    Reflector, Refractor, Water, Sky, Lensflare,
    VRButton, ARButton, XRControllerModelFactory, XRHandModelFactory,
    CSS3DObject, CSS3DSprite, CSS3DRenderer,
    BokehPass, SAOPass, SMAAPass, TAARenderPass, SSAARenderPass, AfterimagePass, BloomPass,
    OutputPass, TransitionPass, RenderTransitionPass, LUTPass, RenderPixelatedPass,
    AdaptiveToneMappingPass,
    // Missing tail — final constants and aliases for full canonical coverage.
    ConstantAlphaFactor: 211, OneMinusConstantAlphaFactor: 212,
    ConstantColorFactor: 213, OneMinusConstantColorFactor: 214,
    CustomToneMapping: 5,
    // Pre-r155 filter aliases (deprecated capitalization)
    LinearMipMapLinearFilter: 1008, LinearMipMapNearestFilter: 1007,
    NearestMipMapLinearFilter: 1005, NearestMipMapNearestFilter: 1004,
    // Reflectivity combine modes for MeshBasicMaterial
    MultiplyOperation: 0, MixOperation: 1, AddOperation: 2,
    // Aliases (`RenderTarget`/`WebGLArrayRenderTarget` collapse onto our impl).
    RenderTarget: WebGLRenderTarget, WebGLArrayRenderTarget: WebGLRenderTarget,
    // ShapePath: subpaths sorted into shapes and holes (in Rust).
    ShapePath,
    // PolyhedronGeometry: subdivision-based polyhedron — alias to icosahedron for now.
    PolyhedronGeometry: class PolyhedronGeometry { constructor(_vertices, _indices, radius = 1, detail = 0) { Object.assign(this, new IcosahedronGeometry(radius, detail)); this.type = 'PolyhedronGeometry'; } },
    // Video / animation export (native-codec wasm)
    VideoFormat,
    BrowserVideoFormat,
    VideoExporter,
    VideoExportResult,
    VideoExportError,
    VideoExportErrorCode,
    VideoExportEvent,
    VideoExportProgressEvent,
    VideoEncodeWorker,
    emitProgress,
    isVideoExportAvailable,
    assertVideoExportAvailable,
    parseVideoFormat,
    formatVideoProgress,
    formatBytes,
    alignVideoSize,
    encodeVideoFrames,
    encodeVideoFramesInWorker,
    downloadVideoBytes,
    encodeAndDownloadVideoFrames,
    captureSceneFrames,
    exportSceneVideo,
    exportAndDownloadSceneVideo,
    encodeGifRgba,
    encodeApngRgba,
    encodeWebmRgba,
    videoMimeType,
    videoFilename,
};

if (typeof window !== 'undefined') {
    window.THREE = THREE;
}

export default THREE;
