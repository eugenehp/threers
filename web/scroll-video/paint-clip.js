/** Procedural calibration clip frames (no external video asset). */

export const DEFAULT_CLIP_DURATION = 10;

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} w
 * @param {number} h
 * @param {number} t
 * @param {number} duration
 */
export function paintClipFrame(ctx, w, h, t, duration) {
    const u = duration > 0 ? Math.min(Math.max(t / duration, 0), 1) : 0;
    const hue = 210 - u * 120;
    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, `hsl(${hue}, 55%, 18%)`);
    grad.addColorStop(1, `hsl(${hue + 40}, 45%, 28%)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(255,255,255,.08)';
    ctx.lineWidth = 1;
    const grid = 48;
    for (let x = 0; x < w; x += grid) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = 0; y < h; y += grid) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    ctx.fillStyle = 'rgba(255,255,255,.92)';
    ctx.font = '700 96px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(t.toFixed(1), w / 2, h * 0.42);

    ctx.font = '500 28px ui-monospace, Menlo, monospace';
    ctx.fillStyle = 'rgba(255,255,255,.55)';
    ctx.fillText('scroll-video calibration clip', w / 2, h * 0.58);

    const barW = w * 0.6;
    const barX = (w - barW) / 2;
    const barY = h * 0.78;
    ctx.fillStyle = 'rgba(0,0,0,.35)';
    ctx.fillRect(barX, barY, barW, 10);
    ctx.fillStyle = `hsl(${hue + 80}, 70%, 55%)`;
    ctx.fillRect(barX, barY, barW * u, 10);
}
