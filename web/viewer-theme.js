// Shared theme manager for the threers viewer pages. Cycles System → Light → Dark
// (default System = follow the OS), persists the choice, and calls `onResolve`
// with the effective 'light'|'dark' so the caller can set the 3D scene background.
const KEY = 'threers-viewer-theme';
const ORDER = ['system', 'light', 'dark'];
const LABEL = { system: '◑ system', light: '☀ light', dark: '☾ dark' };

/** Read the CSS `--scene` colour for the current theme as a 0xRRGGBB number. */
export function sceneColor() {
  const c = getComputedStyle(document.documentElement).getPropertyValue('--scene').trim() || '#14161c';
  const m = c.match(/#?([0-9a-f]{6})/i);
  return m ? parseInt(m[1], 16) : 0x14161c;
}

/** Wire up the `#theme-btn` toggle. `onResolve(theme, colorInt)` fires on init and
 *  on every change (including live OS-preference changes while in System mode). */
export function initTheme(onResolve) {
  const btn = document.getElementById('theme-btn');
  const mql = matchMedia('(prefers-color-scheme: light)');
  let mode = localStorage.getItem(KEY) || 'system';

  const resolved = () => (mode === 'system' ? (mql.matches ? 'light' : 'dark') : mode);
  function apply() {
    if (mode === 'system') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', mode);
    if (btn) btn.textContent = LABEL[mode];
    // Defer so the browser recomputes CSS variables before we read --scene.
    requestAnimationFrame(() => onResolve?.(resolved(), sceneColor()));
  }
  if (btn) {
    btn.addEventListener('click', () => {
      mode = ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length];
      localStorage.setItem(KEY, mode);
      apply();
    });
  }
  mql.addEventListener?.('change', () => { if (mode === 'system') apply(); });
  apply();
  return { resolved };
}
