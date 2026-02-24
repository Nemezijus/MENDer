// Tiny numeric helpers shared across results visualizations.

export function mean(vals) {
  const xs = Array.isArray(vals)
    ? vals.filter((v) => typeof v === 'number' && Number.isFinite(v))
    : [];
  if (!xs.length) return null;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

export function median(vals) {
  const xs = Array.isArray(vals)
    ? vals.filter((v) => typeof v === 'number' && Number.isFinite(v))
    : [];
  if (!xs.length) return null;
  const a = [...xs].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

export function safeMean(vals) {
  return mean(vals);
}

export function safeWeightedMean(vals, weights) {
  const pairs = (Array.isArray(vals) ? vals : [])
    .map((v, i) => [v, Array.isArray(weights) ? weights[i] : undefined])
    .filter(
      ([v, w]) =>
        typeof v === 'number' &&
        Number.isFinite(v) &&
        typeof w === 'number' &&
        Number.isFinite(w) &&
        w > 0,
    );
  if (!pairs.length) return null;
  const num = pairs.reduce((acc, [v, w]) => acc + v * w, 0);
  const den = pairs.reduce((acc, [, w]) => acc + w, 0);
  if (!den) return null;
  return num / den;
}
