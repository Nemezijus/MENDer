export function truncateListForDisplay(items, maxItems = 10) {
  if (!Array.isArray(items)) return { text: '—', truncated: false };

  const arr = items.map((v) => String(v));
  if (arr.length <= maxItems) {
    return { text: arr.join(', '), truncated: false };
  }

  // show first (maxItems - 1), ellipsis, last
  const head = arr.slice(0, Math.max(1, maxItems - 1));
  const tail = arr[arr.length - 1];
  return { text: `${head.join(', ')}, …, ${tail}`, truncated: true };
}

export function computeImbalanceRatioFromCounts(counts) {
  if (!Array.isArray(counts) || counts.length < 2) return null;
  const nonZero = counts.filter((c) => typeof c === 'number' && c > 0);
  if (nonZero.length < 2) return null;
  const maxV = Math.max(...nonZero);
  const minV = Math.min(...nonZero);
  if (!minV) return null;
  return maxV / minV;
}

export function describeImbalanceRatio(ratio) {
  if (typeof ratio !== 'number' || !Number.isFinite(ratio)) return null;
  if (ratio <= 1.5) return 'Labels are well balanced.';
  if (ratio <= 3) return 'Labels are mildly imbalanced.';
  if (ratio <= 5) return 'Labels are moderately imbalanced.';
  return 'Labels are strongly imbalanced (majority label dominates).';
}

export function formatNumber(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return String(v);
  if (v == null) return '—';
  return String(v);
}
