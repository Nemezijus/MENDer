/**
 * Shared helpers for building lightweight "preview" tables.
 *
 * These are used by both supervised and unsupervised results cards.
 */

export function isEmptyCell(v) {
  if (v === null || v === undefined) return true;
  if (typeof v === 'string') return v.trim() === '';
  return false;
}

/**
 * Pick stable, human-friendly columns for a preview table.
 *
 * @param {Array<object>} rows
 * @param {object} [opts]
 * @param {string[]} [opts.preferred=[]] - keys that should appear first (if present)
 * @param {string[]} [opts.groupPrefixes=[]] - keep columns with these prefixes together (in this order)
 * @param {string[]} [opts.alwaysInclude=['index']] - include these keys even if empty
 */
export function pickColumns(rows, opts = {}) {
  const {
    preferred = [],
    groupPrefixes = [],
    alwaysInclude = ['index'],
  } = opts;

  if (!rows || rows.length === 0) return [];

  const keys = new Set();
  rows.forEach((r) => Object.keys(r || {}).forEach((k) => keys.add(k)));

  const nonEmptyKeys = new Set(
    [...keys].filter(
      (k) =>
        (alwaysInclude || []).includes(k) ||
        rows.some((r) => !isEmptyCell(r?.[k])),
    ),
  );

  const preferredSet = new Set(preferred);

  const out = [];
  (preferred || []).forEach((k) => nonEmptyKeys.has(k) && out.push(k));

  // Group known prefixes (e.g. p_* then score_*), keeping them stable.
  const used = new Set(out);
  (groupPrefixes || []).forEach((prefix) => {
    const cols = [...nonEmptyKeys]
      .filter((k) => !used.has(k) && !preferredSet.has(k) && String(k).startsWith(prefix))
      .sort();
    cols.forEach((k) => used.add(k));
    out.push(...cols);
  });

  const rest = [...nonEmptyKeys]
    .filter((k) => !used.has(k))
    .sort();

  out.push(...rest);
  return out;
}
