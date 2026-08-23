/** Load the stage decode video from a URL or bundled asset. */

export const BUNDLED_CLIP_PATH = './assets/calibration-clip.mp4';

/**
 * @param {HTMLVideoElement} video
 * @param {number} time
 */
export function seekVideo(video, time) {
    return new Promise((resolve) => {
        if (!video || video.readyState < 1) {
            resolve();
            return;
        }
        const onSeeked = () => {
            video.removeEventListener('seeked', onSeeked);
            resolve();
        };
        video.addEventListener('seeked', onSeeked);
        try {
            video.currentTime = time;
        } catch (_err) {
            video.removeEventListener('seeked', onSeeked);
            resolve();
            return;
        }
        if (Math.abs(video.currentTime - time) < 0.001) {
            video.removeEventListener('seeked', onSeeked);
            resolve();
        }
    });
}

/**
 * Prime a hidden video element so Safari decodes the first frame.
 * @param {HTMLVideoElement} video
 */
export async function primeVideoDecode(video) {
    if (!video) return;
    try {
        const playPromise = video.play();
        if (playPromise && playPromise.then) await playPromise.catch(() => {});
        video.pause();
    } catch (_err) { /* ignore */ }
}

let activeClipSrc = BUNDLED_CLIP_PATH;
let activeObjectUrl = null;

/**
 * @param {string} [path]
 */
export function setBundledClipPath(path) {
    activeClipSrc = path || BUNDLED_CLIP_PATH;
}

/**
 * @returns {string}
 */
export function getBundledClipPath() {
    return activeClipSrc;
}

/**
 * Load clip into a single decode video element.
 * @param {HTMLVideoElement} video
 * @param {string} [url]
 * @returns {Promise<number>} duration seconds
 */
export async function loadStageVideo(video, url) {
    if (!video) throw new Error('video element required');
    const src = url || activeClipSrc;
    if (activeObjectUrl && activeObjectUrl !== src) {
        URL.revokeObjectURL(activeObjectUrl);
        activeObjectUrl = null;
    }
    if (src.startsWith('blob:')) activeObjectUrl = src;

    video.src = src;
    await new Promise((resolve, reject) => {
        function onMeta() {
            video.removeEventListener('error', onErr);
            resolve();
        }
        function onErr() {
            video.removeEventListener('loadedmetadata', onMeta);
            reject(new Error('video failed to load'));
        }
        video.addEventListener('loadedmetadata', onMeta);
        video.addEventListener('error', onErr);
    });
    video.pause();
    await seekVideo(video, 0);
    await primeVideoDecode(video);
    return video.duration || 10;
}

/**
 * @param {HTMLVideoElement} video
 * @param {File} file
 * @returns {Promise<number>}
 */
export async function loadUserVideoFile(video, file) {
    const url = URL.createObjectURL(file);
    return loadStageVideo(video, url);
}
