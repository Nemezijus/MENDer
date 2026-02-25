function isEmptyCell(v) {
  if (v === null || v === undefined) return true;
  if (typeof v === 'string') return v.trim() === '';
  return false;
}

/**
 * Selects a stable column order for the preview table.
 *
 * - Includes preferred keys first if present.
 * - Drops columns that are entirely empty (except 'index').
 */
export function pickColumns(rows) {
  if (!rows || rows.length === 0) return [];

  const keys = new Set();
  rows.forEach((r) => Object.keys(r || {}).forEach((k) => keys.add(k)));

  const nonEmptyKeys = new Set(
    [...keys].filter((k) => k === 'index' || rows.some((r) => !isEmptyCell(r?.[k]))),
  );

  const preferred = [
    'index',
    'cluster_id',
    'is_noise',
    'is_core',
    'distance_to_center',
    'max_membership_prob',
    'log_likelihood',
  ];

  const rest = [...nonEmptyKeys]
    .filter((k) => !preferred.includes(k))
    .sort();

  const out = [];
  preferred.forEach((k) => nonEmptyKeys.has(k) && out.push(k));
  out.push(...rest);
  return out;
}
