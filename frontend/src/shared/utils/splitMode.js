/**
 * Determine the schema-owned default split mode for the UI.
 *
 * The Engine exposes per-mode defaults under:
 *   split.holdout.defaults
 *   split.kfold.defaults
 *
 * The frontend should not hardcode a global default (e.g. "holdout").
 * Instead, when the user has not selected a split mode override, we pick the
 * schema default that is compatible with the caller's allowedModes.
 *
 * If schema is not yet loaded, we fall back to the first allowed mode.
 */
export function getDefaultSplitMode({ split, allowedModes } = {}) {
  const modes = Array.isArray(allowedModes) && allowedModes.length > 0
    ? allowedModes.map(String)
    : ['holdout', 'kfold'];

  const holdoutMode = split?.holdout?.defaults?.mode;
  const kfoldMode = split?.kfold?.defaults?.mode;

  if (holdoutMode != null && modes.includes(String(holdoutMode))) return String(holdoutMode);
  if (kfoldMode != null && modes.includes(String(kfoldMode))) return String(kfoldMode);

  return modes[0];
}
