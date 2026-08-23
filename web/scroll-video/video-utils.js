/**
 * Draw a video frame (or image) centered with object-fit: cover into a 2D canvas.
 * @param {CanvasRenderingContext2D} ctx
 * @param {CanvasImageSource} source
 * @param {number} cw
 * @param {number} ch
 */
export function drawImageCover(ctx, source, cw, ch) {
    const iw = source.videoWidth || source.width || cw;
    const ih = source.videoHeight || source.height || ch;
    if (!iw || !ih) return;
    const scale = Math.max(cw / iw, ch / ih);
    const dw = iw * scale;
    const dh = ih * scale;
    const dx = (cw - dw) / 2;
    const dy = (ch - dh) / 2;
    ctx.drawImage(source, dx, dy, dw, dh);
}
