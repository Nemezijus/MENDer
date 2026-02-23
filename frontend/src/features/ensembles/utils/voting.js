/**
 * Voting-ensemble specific helpers.
 */

export function dedupeWarning(estimators) {
  const algos = (estimators || []).map((s) => s?.model?.algo).filter(Boolean);
  const set = new Set();
  const dup = new Set();
  for (const a of algos) {
    if (set.has(a)) dup.add(a);
    set.add(a);
  }
  return dup.size ? Array.from(dup) : null;
}

export function normalizeWeight(v) {
  if (v === '' || v == null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}
