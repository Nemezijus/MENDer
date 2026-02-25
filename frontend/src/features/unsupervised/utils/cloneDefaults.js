/**
 * Best-effort deep clone for schema defaults.
 *
 * We use this to avoid mutating schema-provided defaults while still allowing
 * local edits in the UI (ModelSelectionCard writes into the model object).
 */
export function cloneDefaults(obj) {
  if (obj == null) return obj;

  // Modern browsers.
  if (typeof structuredClone === 'function') {
    try {
      return structuredClone(obj);
    } catch {
      // fall through
    }
  }

  // JSON fallback for plain data.
  try {
    return JSON.parse(JSON.stringify(obj));
  } catch {
    return obj;
  }
}
