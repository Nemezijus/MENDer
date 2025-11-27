import { Card, Stack, Group, Text, Divider, Select, NumberInput, Checkbox, SimpleGrid } from '@mantine/core';
import { useEffect } from 'react';
import { useDataStore } from '../state/useDataStore.js';

/** ---------------- helpers (unchanged) ---------------- **/

function resolveRef(schema, ref) {
  if (!schema || !ref || typeof ref !== 'string') return null;
  const prefix = '#/$defs/';
  if (!ref.startsWith(prefix)) return null;
  const key = ref.slice(prefix.length);
  return schema?.$defs?.[key] ?? null;
}

function getAlgoSchema(schema, algo) {
  if (!schema || !algo) return null;

  const mapping = schema?.discriminator?.mapping;
  if (mapping && mapping[algo]) {
    const target = resolveRef(schema, mapping[algo]);
    if (target) return target;
  }

  const variants = schema?.oneOf || schema?.anyOf || [];
  for (const entry of variants) {
    const target = entry?.$ref ? resolveRef(schema, entry.$ref) : entry;
    const alg = target?.properties?.algo?.const ?? target?.properties?.algo?.default;
    if (alg === algo) return target || null;
  }

  return null;
}

function enumFromSubSchema(sub, key, fallback) {
  try {
    const p = sub?.properties?.[key];
    if (!p) return fallback;
    if (Array.isArray(p.enum)) return p.enum;
    const list = (p.anyOf ?? p.oneOf ?? [])
      .flatMap((x) => {
        if (Array.isArray(x.enum)) return x.enum;
        if (x.const != null) return [x.const];
        if (x.type === 'null') return [null];
        return [];
      });
    return list.length ? list : fallback;
  } catch {
    return fallback;
  }
}

function toSelectData(enums, { includeNoneLabel = false } = {}) {
  const out = [];
  let hasNull = false;
  for (const v of enums ?? []) {
    if (v === null) { hasNull = true; continue; }
    out.push({ value: String(v), label: String(v) });
  }
  if (hasNull && includeNoneLabel) out.unshift({ value: 'none', label: 'none' });
  return out;
}

function fromSelectNullable(v) {
  return v === 'none' ? null : v;
}

function maxFeatToModeVal(v) {
  if (v == null) return { mode: 'none', value: null };
  if (v === 'sqrt' || v === 'log2') return { mode: v, value: null };
  if (typeof v === 'number' && Number.isFinite(v)) {
    return { mode: Number.isInteger(v) ? 'int' : 'float', value: v };
  }
  return { mode: 'none', value: null };
}
function modeValToMaxFeat(mode, value) {
  if (mode === 'none') return null;
  if (mode === 'sqrt' || mode === 'log2') return mode;
  if (mode === 'int' || mode === 'float') {
    if (value == null || value === '') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function algoListFromSchema(schema) {
  if (!schema) return null;

  const mapping = schema?.discriminator?.mapping;
  if (mapping && typeof mapping === 'object') {
    const keys = Object.keys(mapping);
    if (keys.length) return keys;
  }

  const variants = schema?.oneOf || schema?.anyOf || [];
  const set = new Set();
  for (const entry of variants) {
    const target = entry?.$ref ? resolveRef(schema, entry.$ref) : entry;
    const alg = target?.properties?.algo?.const ?? target?.properties?.algo?.default;
    if (alg) set.add(String(alg));
  }
  if (set.size) return Array.from(set);

  return null;
}

/** --------------- component --------------- **/

export default function ModelSelectionCard({ model, onChange, schema, enums, models }) {
  const m = model || {};
  const set = (patch) => onChange?.({ ...m, ...patch });

  const effectiveTask = useDataStore(
    (s) => s.taskSelected || s.inspectReport?.task_inferred || null,
  );

  // Prefer schema -> defaults -> known fallback
  const schemaAlgos = algoListFromSchema(schema);
  const defaultsAlgos = models?.defaults ? Object.keys(models.defaults) : null;
  const available = new Set(schemaAlgos || defaultsAlgos || ['logreg','svm','tree','forest','knn','linreg']);

  // Preferred ordering, append any extras after
  const preferred = ['logreg','svm','tree','forest','knn','linreg'];
  const orderedAll = [
    ...preferred.filter(a => available.has(a)),
    ...Array.from(available).filter(a => !preferred.includes(a)),
  ];

  // NEW: filter by task using backend-provided meta[algo].task
  const meta = models?.meta || {};
  const matchesTask = (algo) => {
    if (!effectiveTask) return true;                     // no filter if task unknown
    const t = meta[algo]?.task;                          // 'classification' | 'regression' | undefined
    if (!t) return true;                                 // if backend didn’t annotate, don’t hide it
    return t === effectiveTask;
  };
  const ordered = orderedAll.filter(matchesTask);

  // If current selection is incompatible with task, auto-switch to first compatible
  useEffect(() => {
    if (!m?.algo) return;
    if (ordered.length === 0) return;                    // nothing to switch to
    if (!matchesTask(m.algo)) {
      set({ algo: ordered[0] });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveTask, JSON.stringify(orderedAll), JSON.stringify(meta)]);

  const algoData = toSelectData(ordered.length ? ordered : orderedAll);

  const sub = getAlgoSchema(schema, m.algo);

  // Logistic Regression enums
  const lrPenalty     = toSelectData(enumFromSubSchema(sub, 'penalty', enums?.PenaltyName));
  const lrSolver      = toSelectData(enumFromSubSchema(sub, 'solver',  enums?.LogRegSolver));
  const lrClassWeight = toSelectData(
    enumFromSubSchema(sub, 'class_weight', enums?.ClassWeightBalanced ?? ['balanced', null]),
    { includeNoneLabel: true }
  );

  // SVM enums
  const svmKernel         = toSelectData(enumFromSubSchema(sub, 'kernel', enums?.SVMKernel));
  const svmDecisionShape  = toSelectData(enumFromSubSchema(sub, 'decision_function_shape', enums?.SVMDecisionShape));
  const svmClassWeight    = toSelectData(
    enumFromSubSchema(sub, 'class_weight', enums?.ClassWeightBalanced ?? ['balanced', null]),
    { includeNoneLabel: true }
  );

  // Tree enums
  const treeCriterion   = toSelectData(enumFromSubSchema(sub, 'criterion', enums?.TreeCriterion));
  const treeSplitter    = toSelectData(enumFromSubSchema(sub, 'splitter',  enums?.TreeSplitter));
  const treeClassWeight = toSelectData(
    enumFromSubSchema(sub, 'class_weight', enums?.ClassWeightBalanced ?? ['balanced', null]),
    { includeNoneLabel: true }
  );

  // Forest enums
  const forestCriterion   = toSelectData(enumFromSubSchema(sub, 'criterion', enums?.TreeCriterion));
  const forestClassWeight = toSelectData(
    enumFromSubSchema(sub, 'class_weight', enums?.ForestClassWeight ?? ['balanced', 'balanced_subsample', null]),
    { includeNoneLabel: true }
  );

  // KNN enums
  const knnWeights   = toSelectData(enumFromSubSchema(sub, 'weights',   enums?.KNNWeights));
  const knnAlgorithm = toSelectData(enumFromSubSchema(sub, 'algorithm', enums?.KNNAlgorithm));
  const knnMetric    = toSelectData(enumFromSubSchema(sub, 'metric',    enums?.KNNMetric));

  const gammaMode = typeof m.gamma === 'number' ? 'numeric' : (m.gamma ?? 'scale');
  const gammaValue = typeof m.gamma === 'number' ? m.gamma : 0.1;

  const tMF = maxFeatToModeVal(m.max_features);
  const fMF = maxFeatToModeVal(m.max_features);

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        <Group justify="space-between"><Text fw={500}>Model</Text></Group>

        <Select
          label="Algorithm"
          data={algoData}
          value={m.algo ?? (algoData[0]?.value ?? 'logreg')}
          onChange={(algo) => { if (algo) set({ algo }); }}
        />

        {/* Logistic Regression */}
        {m.algo === 'logreg' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <NumberInput label="C" value={m.C ?? 1.0} onChange={(v) => set({ C: v })} min={0} step={0.1} />
              <Select label="penalty" data={lrPenalty} value={m.penalty ?? 'l2'} onChange={(v)=>set({ penalty: v })} />
              <Select label="solver" data={lrSolver} value={m.solver ?? 'lbfgs'} onChange={(v)=>set({ solver: v })} />
              <NumberInput label="max_iter" value={m.max_iter ?? 1000} onChange={(v)=>set({ max_iter: v })} allowDecimal={false} min={1} />
              <Select
                label="class_weight"
                data={lrClassWeight}
                value={m.class_weight == null ? 'none' : String(m.class_weight)}
                onChange={(v)=>set({ class_weight: fromSelectNullable(v) })}
              />
              {m.penalty === 'elasticnet' && (
                <NumberInput label="l1_ratio" value={m.l1_ratio ?? 0.5} onChange={(v)=>set({ l1_ratio: v })} min={0} max={1} step={0.01} />
              )}
            </SimpleGrid>
          </Stack>
        )}

        {/* SVM */}
        {m.algo === 'svm' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <Select label="kernel" data={svmKernel} value={m.kernel ?? 'rbf'} onChange={(v)=>set({ kernel: v })} />
              <NumberInput label="C" value={m.C ?? 1.0} onChange={(v)=>set({ C: v })} min={0} step={0.1} />
              <NumberInput label="degree (poly)" value={m.degree ?? 3} onChange={(v)=>set({ degree: v })} allowDecimal={false} min={1} />
              <Select
                label="gamma"
                data={[
                  { value: 'scale', label: 'scale' },
                  { value: 'auto', label: 'auto' },
                  { value: 'numeric', label: 'numeric' },
                ]}
                value={gammaMode}
                onChange={(mode) => {
                  if (mode === 'numeric') set({ gamma: typeof gammaValue === 'number' ? gammaValue : 0.1 });
                  else set({ gamma: mode });
                }}
              />
              {gammaMode === 'numeric' && (
                <NumberInput label="gamma value" value={gammaValue} onChange={(v)=>set({ gamma: v })} min={0} step={0.001} />
              )}
              <NumberInput label="coef0" value={m.coef0 ?? 0.0} onChange={(v)=>set({ coef0: v })} step={0.001} />
              <Checkbox label="shrinking" checked={!!m.shrinking} onChange={(e)=>set({ shrinking: e.currentTarget.checked })} />
              <Checkbox label="probability" checked={!!m.probability} onChange={(e)=>set({ probability: e.currentTarget.checked })} />
              <NumberInput label="tol" value={m.tol ?? 1e-3} onChange={(v)=>set({ tol: v })} step={0.0001} />
              <NumberInput label="cache_size" value={m.cache_size ?? 200.0} onChange={(v)=>set({ cache_size: v })} />
              <Select
                label="class_weight"
                data={svmClassWeight}
                value={m.class_weight == null ? 'none' : String(m.class_weight)}
                onChange={(v)=>set({ class_weight: fromSelectNullable(v) })}
              />
              <NumberInput label="max_iter" value={m.max_iter ?? -1} onChange={(v)=>set({ max_iter: v })} allowDecimal={false} />
              <Select label="decision_function_shape" data={svmDecisionShape}
                      value={m.decision_function_shape ?? 'ovr'} onChange={(v)=>set({ decision_function_shape: v })} />
              <Checkbox label="break_ties" checked={!!m.break_ties} onChange={(e)=>set({ break_ties: e.currentTarget.checked })} />
            </SimpleGrid>
          </Stack>
        )}

        {/* Decision Tree */}
        {m.algo === 'tree' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <Select label="criterion" data={treeCriterion} value={m.criterion ?? 'gini'} onChange={(v)=>set({ criterion: v })} />
              <Select label="splitter" data={treeSplitter} value={m.splitter ?? 'best'} onChange={(v)=>set({ splitter: v })} />
              <NumberInput label="max_depth" value={m.max_depth ?? null} onChange={(v)=>set({ max_depth: v })} />
              <NumberInput label="min_samples_split" value={m.min_samples_split ?? 2} onChange={(v)=>set({ min_samples_split: v })} />
              <NumberInput label="min_samples_leaf" value={m.min_samples_leaf ?? 1} onChange={(v)=>set({ min_samples_leaf: v })} />
              <NumberInput label="min_weight_fraction_leaf" value={m.min_weight_fraction_leaf ?? 0.0} onChange={(v)=>set({ min_weight_fraction_leaf: v })} step={0.01} min={0} max={1} />
              <Select
                label="max_features (mode)"
                data={[
                  { value: 'sqrt', label: 'sqrt' },
                  { value: 'log2', label: 'log2' },
                  { value: 'int', label: 'int' },
                  { value: 'float', label: 'float' },
                  { value: 'none', label: 'none' },
                ]}
                value={tMF.mode}
                onChange={(mode)=>set({ max_features: modeValToMaxFeat(mode, tMF.value) })}
              />
              {(tMF.mode === 'int' || tMF.mode === 'float') && (
                <NumberInput
                  label="max_features (value)"
                  value={tMF.value ?? null}
                  onChange={(v)=>set({ max_features: modeValToMaxFeat(tMF.mode, v) })}
                  step={tMF.mode === 'int' ? 1 : 0.01}
                  allowDecimal={tMF.mode === 'float'}
                />
              )}
              <NumberInput label="max_leaf_nodes" value={m.max_leaf_nodes ?? null} onChange={(v)=>set({ max_leaf_nodes: v })} allowDecimal={false} />
              <NumberInput label="min_impurity_decrease" value={m.min_impurity_decrease ?? 0.0} onChange={(v)=>set({ min_impurity_decrease: v })} step={0.0001} />
              <Select
                label="class_weight"
                data={treeClassWeight}
                value={m.class_weight == null ? 'none' : String(m.class_weight)}
                onChange={(v)=>set({ class_weight: fromSelectNullable(v) })}
              />
              <NumberInput label="ccp_alpha" value={m.ccp_alpha ?? 0.0} onChange={(v)=>set({ ccp_alpha: v })} step={0.0001} />
            </SimpleGrid>
          </Stack>
        )}

        {/* Random Forest */}
        {m.algo === 'forest' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <NumberInput label="n_estimators" value={m.n_estimators ?? 100} onChange={(v)=>set({ n_estimators: v })} allowDecimal={false} min={1} />
              <Select label="criterion" data={forestCriterion} value={m.criterion ?? 'gini'} onChange={(v)=>set({ criterion: v })} />
              <NumberInput label="max_depth" value={m.max_depth ?? null} onChange={(v)=>set({ max_depth: v })} />
              <NumberInput label="min_samples_split" value={m.min_samples_split ?? 2} onChange={(v)=>set({ min_samples_split: v })} />
              <NumberInput label="min_samples_leaf" value={m.min_samples_leaf ?? 1} onChange={(v)=>set({ min_samples_leaf: v })} />
              <NumberInput label="min_weight_fraction_leaf" value={m.min_weight_fraction_leaf ?? 0.0} onChange={(v)=>set({ min_weight_fraction_leaf: v })} step={0.01} min={0} max={1} />
              <Select
                label="max_features (mode)"
                data={[
                  { value: 'sqrt', label: 'sqrt' },
                  { value: 'log2', label: 'log2' },
                  { value: 'int', label: 'int' },
                  { value: 'float', label: 'float' },
                  { value: 'none', label: 'none' },
                ]}
                value={fMF.mode}
                onChange={(mode)=>set({ max_features: modeValToMaxFeat(mode, fMF.value) })}
              />
              {(fMF.mode === 'int' || fMF.mode === 'float') && (
                <NumberInput
                  label="max_features (value)"
                  value={fMF.value ?? null}
                  onChange={(v)=>set({ max_features: modeValToMaxFeat(fMF.mode, v) })}
                  step={fMF.mode === 'int' ? 1 : 0.01}
                  allowDecimal={fMF.mode === 'float'}
                />
              )}
              <NumberInput label="max_leaf_nodes" value={m.max_leaf_nodes ?? null} onChange={(v)=>set({ max_leaf_nodes: v })} allowDecimal={false} />
              <NumberInput label="min_impurity_decrease" value={m.min_impurity_decrease ?? 0.0} onChange={(v)=>set({ min_impurity_decrease: v })} step={0.0001} />
              <Select
                label="class_weight"
                data={forestClassWeight}
                value={m.class_weight == null ? 'none' : String(m.class_weight)}
                onChange={(v)=>set({ class_weight: fromSelectNullable(v) })}
              />
              <Checkbox label="bootstrap" checked={!!m.bootstrap} onChange={(e)=>set({ bootstrap: e.currentTarget.checked })} />
              <Checkbox label="oob_score" checked={!!m.oob_score} onChange={(e)=>set({ oob_score: e.currentTarget.checked })} />
              <NumberInput label="n_jobs" value={m.n_jobs ?? null} onChange={(v)=>set({ n_jobs: v })} allowDecimal={false} />
              <NumberInput label="ccp_alpha" value={m.ccp_alpha ?? 0.0} onChange={(v)=>set({ ccp_alpha: v })} step={0.0001} />
              <Checkbox label="warm_start" checked={!!m.warm_start} onChange={(e)=>set({ warm_start: e.currentTarget.checked })} />
              <NumberInput label="random_state" value={m.random_state ?? null} onChange={(v)=>set({ random_state: v })} allowDecimal={false} />
              <NumberInput label="max_samples" value={m.max_samples ?? null} onChange={(v)=>set({ max_samples: v })} />
            </SimpleGrid>
          </Stack>
        )}

        {/* KNN */}
        {m.algo === 'knn' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <NumberInput label="n_neighbors" value={m.n_neighbors ?? 5} onChange={(v)=>set({ n_neighbors: v })} allowDecimal={false} min={1} />
              <Select label="weights" data={knnWeights} value={m.weights ?? 'uniform'} onChange={(v)=>set({ weights: v })} />
              <Select label="algorithm" data={knnAlgorithm} value={m.algorithm ?? 'auto'} onChange={(v)=>set({ algorithm: v })} />
              <NumberInput label="leaf_size" value={m.leaf_size ?? 30} onChange={(v)=>set({ leaf_size: v })} allowDecimal={false} min={1} />
              <NumberInput label="p" value={m.p ?? 2} onChange={(v)=>set({ p: v })} allowDecimal={false} min={1} />
              <Select label="metric" data={knnMetric} value={m.metric ?? 'minkowski'} onChange={(v)=>set({ metric: v })} />
              <NumberInput label="n_jobs" value={m.n_jobs ?? null} onChange={(v)=>set({ n_jobs: v })} allowDecimal={false} />
            </SimpleGrid>
          </Stack>
        )}

        {/* Linear Regression */}
        {m.algo === 'linreg' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <Checkbox
                label="fit_intercept"
                checked={!!m.fit_intercept}
                onChange={(e)=>set({ fit_intercept: e.currentTarget.checked })}
              />
              <Checkbox
                label="copy_X"
                checked={m.copy_X ?? true}
                onChange={(e)=>set({ copy_X: e.currentTarget.checked })}
              />
              <NumberInput
                label="n_jobs"
                value={m.n_jobs ?? null}
                onChange={(v)=>set({ n_jobs: v })}
                allowDecimal={false}
              />
              <Checkbox
                label="positive"
                checked={!!m.positive}
                onChange={(e)=>set({ positive: e.currentTarget.checked })}
              />
            </SimpleGrid>
          </Stack>
        )}

        <Divider my="xs" />
        <Text size="xs" c="dimmed">
          Algorithms are filtered by dataset task via <code>models.meta[algo].task</code> and your selected task in the Data panel.
        </Text>
      </Stack>
    </Card>
  );
}
