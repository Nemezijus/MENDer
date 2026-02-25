/**
 * Zustand store helper "landing zone".
 *
 * As we refactor features one-by-one, many stores repeat patterns:
 * - resettable initial state
 * - shallow-merge setters for nested slices
 * - array item update/remove helpers
 */

/**
 * @param {object} initialState
 * @param {(next: any) => void} set
 */
export function makeReset(initialState, set) {
  return () => set({ ...(initialState || {}) });
}

/**
 * Simple setter for a single key.
 *
 * Keeps stores consistent and avoids repeating `(value) => set({ key: value })`.
 *
 * @param {string} key
 * @param {(updater: any) => void} set
 */
export function makeKeySetter(key, set) {
  return (value) => set({ [key]: value });
}

/**
 * Shallow-merge update for a nested slice.
 *
 * @param {string} key
 * @param {(updater: any) => void} set
 */
export function makeShallowMergeSetter(key, set) {
  return (partial) =>
    set((state) => ({
      [key]: { ...(state?.[key] || {}), ...(partial || {}) },
    }));
}

/**
 * Update an item within an array slice.
 *
 * @param {string} key
 * @param {(updater: any) => void} set
 */
export function makeArrayItemUpdater(key, set) {
  return (idx, patch) =>
    set((state) => {
      const cur = Array.isArray(state?.[key]) ? state[key] : [];
      const next = cur.map((it, i) =>
        i === idx ? { ...(it || {}), ...(patch || {}) } : it,
      );
      return { [key]: next };
    });
}

/**
 * Remove an item within an array slice.
 *
 * @param {string} key
 * @param {(updater: any) => void} set
 * @param {object} [opts]
 * @param {number} [opts.minLength=0]
 */
export function makeArrayItemRemover(key, set, opts = {}) {
  const { minLength = 0 } = opts;
  return (idx) =>
    set((state) => {
      const cur = Array.isArray(state?.[key]) ? state[key] : [];
      if (cur.length <= minLength) return state;
      const next = cur.filter((_, i) => i !== idx);
      return { [key]: next };
    });
}

/**
 * Update an item within an array nested under a slice.
 *
 * Example:
 *   makeNestedArrayItemUpdater('voting', 'estimators', set)
 */
export function makeNestedArrayItemUpdater(sliceKey, arrayKey, set) {
  return (idx, patch) =>
    set((state) => {
      const slice = state?.[sliceKey] || {};
      const cur = Array.isArray(slice?.[arrayKey]) ? slice[arrayKey] : [];
      const next = cur.map((it, i) =>
        i === idx ? { ...(it || {}), ...(patch || {}) } : it,
      );
      return { [sliceKey]: { ...(slice || {}), [arrayKey]: next } };
    });
}

/**
 * Remove an item within an array nested under a slice.
 */
export function makeNestedArrayItemRemover(sliceKey, arrayKey, set, opts = {}) {
  const { minLength = 0 } = opts;
  return (idx) =>
    set((state) => {
      const slice = state?.[sliceKey] || {};
      const cur = Array.isArray(slice?.[arrayKey]) ? slice[arrayKey] : [];
      if (cur.length <= minLength) return state;
      const next = cur.filter((_, i) => i !== idx);
      return { [sliceKey]: { ...(slice || {}), [arrayKey]: next } };
    });
}
