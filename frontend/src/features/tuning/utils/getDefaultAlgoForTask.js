/**
 * Schema-owned default algo selection for tuning panels.
 *
 * We prefer an explicit engine-provided mapping (models.default_algo_by_task)
 * and only fall back to the first compatible algo as a backwards-compatible
 * behavior for older schema bundles.
 */

export function getDefaultAlgoForTask({ models, task }) {
  if (!models) return null;

  const taskKey = task === 'regression' ? 'regression' : 'classification';

  const byTask = models?.default_algo_by_task ?? null;
  const preferred = byTask?.[taskKey];
  if (preferred) return preferred;

  // Back-compat fallback: derive from meta.task.
  const defaults = models?.defaults ?? {};
  const meta = models?.meta ?? {};

  const all = Object.keys(defaults);
  if (!all.length) return null;

  const compatible = all.filter((algo) => {
    const m = meta?.[algo];
    if (!m) return true; // if no meta, don't exclude
    const t = m.task;
    if (!t) return true;
    if (t === 'clustering' && taskKey === 'unsupervised') return true;
    if (Array.isArray(t)) return t.includes(taskKey);
    return t === taskKey;
  });

  return compatible[0] ?? all[0] ?? null;
}
