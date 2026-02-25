/**
 * Shallow equality for plain objects.
 *
 * Used to avoid re-setting model defaults when the merged result matches.
 */
export function shallowEqual(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  if (typeof a !== 'object' || typeof b !== 'object') return false;

  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;

  for (let i = 0; i < ka.length; i += 1) {
    const k = ka[i];
    if (a[k] !== b[k]) return false;
  }

  return true;
}
