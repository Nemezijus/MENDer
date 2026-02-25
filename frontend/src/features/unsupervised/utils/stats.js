export function toFiniteNumbers(arr) {
  if (!Array.isArray(arr)) return [];
  return arr
    .map((v) => (typeof v === 'number' ? v : Number(v)))
    .filter((v) => Number.isFinite(v));
}

export function histogram(values, nBins = 30) {
  const xs = toFiniteNumbers(values);
  if (!xs.length) return null;

  const min = Math.min(...xs);
  const max = Math.max(...xs);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  if (min === max) return { x: [min], y: [xs.length] };

  const bins = Math.max(5, Math.min(nBins, Math.floor(Math.sqrt(xs.length) * 2)));
  const width = (max - min) / bins;
  const counts = new Array(bins).fill(0);

  xs.forEach((v) => {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / width)));
    counts[idx] += 1;
  });

  const centers = Array.from({ length: bins }, (_, i) => min + (i + 0.5) * width);
  return { x: centers, y: counts };
}

export function hexToRgba(hex, alpha = 0.25) {
  if (typeof hex !== 'string' || !hex.startsWith('#') || (hex.length !== 7 && hex.length !== 4)) {
    return `rgba(0,0,0,${alpha})`;
  }

  let r;
  let g;
  let b;

  if (hex.length === 4) {
    r = parseInt(hex[1] + hex[1], 16);
    g = parseInt(hex[2] + hex[2], 16);
    b = parseInt(hex[3] + hex[3], 16);
  } else {
    r = parseInt(hex.slice(1, 3), 16);
    g = parseInt(hex.slice(3, 5), 16);
    b = parseInt(hex.slice(5, 7), 16);
  }

  if (![r, g, b].every((v) => Number.isFinite(v))) return `rgba(0,0,0,${alpha})`;
  return `rgba(${r},${g},${b},${alpha})`;
}

/**
 * Build Lorenz curve + Gini coefficient from cluster sizes.
 *
 * @param {{size:any}[]} sizes
 */
export function lorenzFromSizes(sizes) {
  const vals = (Array.isArray(sizes) ? sizes : [])
    .map((r) => Number(r?.size))
    .filter((v) => Number.isFinite(v) && v >= 0)
    .sort((a, b) => a - b);

  const n = vals.length;
  if (n < 1) return null;
  const total = vals.reduce((s, v) => s + v, 0);
  if (!Number.isFinite(total) || total <= 0) return null;

  const x = [0];
  const y = [0];
  let cum = 0;
  for (let i = 0; i < n; i += 1) {
    cum += vals[i];
    x.push((i + 1) / n);
    y.push(cum / total);
  }

  // Gini from Lorenz via trapezoid area.
  let area = 0;
  for (let i = 1; i < x.length; i += 1) {
    const dx = x[i] - x[i - 1];
    const yAvg = (y[i] + y[i - 1]) / 2;
    area += dx * yAvg;
  }
  const gini = 1 - 2 * area;
  return { x, y, gini: Number.isFinite(gini) ? gini : null };
}

/**
 * Returns a (rough) ellipse outline in 2D from mean + covariance.
 * scale=2 roughly corresponds to ~95% for Gaussian if cov is well-behaved.
 */
export function ellipsePoints(mean, cov, n = 100, scale = 2) {
  const mx = mean?.[0];
  const my = mean?.[1];
  const a = cov?.[0]?.[0];
  const b = cov?.[0]?.[1];
  const c = cov?.[1]?.[0];
  const d = cov?.[1]?.[1];
  if (![mx, my, a, b, c, d].every((v) => typeof v === 'number' && Number.isFinite(v))) return null;

  // eigen-decomposition for 2x2
  const tr = a + d;
  const det = a * d - b * c;
  const disc = Math.max(tr * tr - 4 * det, 0);
  const s = Math.sqrt(disc);
  const l1 = (tr + s) / 2;
  const l2 = (tr - s) / 2;
  if (!(l1 > 0) || !(l2 > 0)) return null;

  // eigenvector for l1
  let vx = b;
  let vy = l1 - a;
  if (Math.abs(vx) + Math.abs(vy) < 1e-12) {
    vx = l1 - d;
    vy = c;
  }
  const norm = Math.hypot(vx, vy) || 1;
  vx /= norm;
  vy /= norm;

  const wx = -vy;
  const wy = vx;

  const rx = Math.sqrt(l1) * scale;
  const ry = Math.sqrt(l2) * scale;

  const xs = [];
  const ys = [];

  for (let i = 0; i <= n; i += 1) {
    const t = (i / n) * Math.PI * 2;
    const ct = Math.cos(t);
    const st = Math.sin(t);
    const px = mx + rx * ct * vx + ry * st * wx;
    const py = my + rx * ct * vy + ry * st * wy;
    xs.push(px);
    ys.push(py);
  }

  return { x: xs, y: ys };
}
