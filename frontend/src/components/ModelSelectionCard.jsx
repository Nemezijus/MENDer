import { Card, Stack, Group, Text, Divider, Select, NumberInput, Checkbox, SimpleGrid } from '@mantine/core';

// --- helpers ---------------------------------------------------------------

// tree/forest max_features adapters (unchanged)
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

// extract enum values from JSON Schema for a property name
function enumFromSchema(schema, key, fallback) {
  try {
    const p = schema?.properties?.[key];
    if (!p) return fallback;
    // direct enum
    if (Array.isArray(p.enum)) return p.enum;

    // union (anyOf / oneOf) with enums and/or nulls
    const list = (p.anyOf ?? p.oneOf ?? [])
      .map((x) => (Array.isArray(x.enum) ? x.enum : (x.const != null ? [x.const] : [])))
      .flat();
    return list.length ? list : fallback;
  } catch {
    return fallback;
  }
}

// Turn an enum array into Mantine <Select> data, optionally injecting a "none" option for null
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

// Map "none" UI choice to null, otherwise keep as string
function fromSelectNullable(v) {
  return v === 'none' ? null : v;
}

// --- component -------------------------------------------------------------

export default function ModelSelectionCard({ model, onChange, schema }) {
  const m = model || {};
  const set = (patch) => onChange?.({ ...m, ...patch });

  // SVM gamma UI adapter
  const gammaMode =
    typeof m.svm_gamma === 'number'
      ? 'numeric'
      : (m.svm_gamma ?? 'scale'); // 'scale' | 'auto' | 'numeric'
  const gammaValue = typeof m.svm_gamma === 'number' ? m.svm_gamma : 0.1;

  // tree/forest max_features adapters
  const tMF = maxFeatToModeVal(m.tree_max_features);
  const fMF = maxFeatToModeVal(m.forest_max_features);

  // --- schema-powered enums (with safe fallbacks) ---
  const algoData = toSelectData(
    enumFromSchema(schema, 'algo', ['logreg', 'svm', 'tree', 'forest', 'knn'])
  );

  const lrPenalty = toSelectData(
    enumFromSchema(schema, 'penalty', ['l2', 'l1', 'elasticnet', 'none'])
  );
  const lrSolver = toSelectData(
    enumFromSchema(schema, 'solver', ['lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'])
  );
  const lrClassWeight = toSelectData(
    enumFromSchema(schema, 'class_weight', ['balanced', null]),
    { includeNoneLabel: true }
  );

  const svmKernel = toSelectData(
    enumFromSchema(schema, 'svm_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
  );
  const svmDecisionShape = toSelectData(
    enumFromSchema(schema, 'svm_decision_function_shape', ['ovr', 'ovo'])
  );
  const svmClassWeight = toSelectData(
    enumFromSchema(schema, 'svm_class_weight', ['balanced', null]),
    { includeNoneLabel: true }
  );

  const treeCriterion = toSelectData(
    enumFromSchema(schema, 'tree_criterion', ['gini', 'entropy', 'log_loss'])
  );
  const treeSplitter = toSelectData(
    enumFromSchema(schema, 'tree_splitter', ['best', 'random'])
  );
  const treeClassWeight = toSelectData(
    enumFromSchema(schema, 'tree_class_weight', ['balanced', null]),
    { includeNoneLabel: true }
  );

  const forestCriterion = toSelectData(
    enumFromSchema(schema, 'forest_criterion', ['gini', 'entropy', 'log_loss'])
  );
  const forestClassWeight = toSelectData(
    enumFromSchema(schema, 'forest_class_weight', ['balanced', 'balanced_subsample', null]),
    { includeNoneLabel: true }
  );

  const knnWeights = toSelectData(
    enumFromSchema(schema, 'knn_weights', ['uniform', 'distance'])
  );
  const knnAlgorithm = toSelectData(
    enumFromSchema(schema, 'knn_algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
  );
  const knnMetric = toSelectData(
    enumFromSchema(schema, 'knn_metric', ['minkowski', 'euclidean', 'manhattan', 'chebyshev'])
  );

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        <Group justify="space-between"><Text fw={500}>Model</Text></Group>

        <Select
          label="Algorithm"
          data={algoData}
          value={m.algo ?? 'logreg'}
          onChange={(v) => set({ algo: v })}
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
              <Select label="kernel" data={svmKernel} value={m.svm_kernel ?? 'rbf'} onChange={(v)=>set({ svm_kernel: v })} />
              <NumberInput label="C" value={m.svm_C ?? 1.0} onChange={(v)=>set({ svm_C: v })} min={0} step={0.1} />
              <NumberInput label="degree (poly)" value={m.svm_degree ?? 3} onChange={(v)=>set({ svm_degree: v })} allowDecimal={false} min={1} />
              <Select
                label="gamma"
                data={[
                  { value: 'scale', label: 'scale' },
                  { value: 'auto', label: 'auto' },
                  { value: 'numeric', label: 'numeric' },
                ]}
                value={gammaMode}
                onChange={(mode) => {
                  if (mode === 'numeric') set({ svm_gamma: typeof gammaValue === 'number' ? gammaValue : 0.1 });
                  else set({ svm_gamma: mode });
                }}
              />
              {gammaMode === 'numeric' && (
                <NumberInput label="gamma value" value={gammaValue} onChange={(v)=>set({ svm_gamma: v })} min={0} step={0.001} />
              )}
              <NumberInput label="coef0" value={m.svm_coef0 ?? 0.0} onChange={(v)=>set({ svm_coef0: v })} step={0.001} />
              <Checkbox label="shrinking" checked={!!m.svm_shrinking} onChange={(e)=>set({ svm_shrinking: e.currentTarget.checked })} />
              <Checkbox label="probability" checked={!!m.svm_probability} onChange={(e)=>set({ svm_probability: e.currentTarget.checked })} />
              <NumberInput label="tol" value={m.svm_tol ?? 1e-3} onChange={(v)=>set({ svm_tol: v })} step={0.0001} />
              <NumberInput label="cache_size" value={m.svm_cache_size ?? 200.0} onChange={(v)=>set({ svm_cache_size: v })} />
              <Select
                label="class_weight"
                data={svmClassWeight}
                value={m.svm_class_weight == null ? 'none' : String(m.svm_class_weight)}
                onChange={(v)=>set({ svm_class_weight: fromSelectNullable(v) })}
              />
              <NumberInput label="max_iter" value={m.svm_max_iter ?? -1} onChange={(v)=>set({ svm_max_iter: v })} allowDecimal={false} />
              <Select label="decision_function_shape" data={svmDecisionShape}
                      value={m.svm_decision_function_shape ?? 'ovr'} onChange={(v)=>set({ svm_decision_function_shape: v })} />
              <Checkbox label="break_ties" checked={!!m.svm_break_ties} onChange={(e)=>set({ svm_break_ties: e.currentTarget.checked })} />
            </SimpleGrid>
          </Stack>
        )}

        {/* Decision Tree */}
        {m.algo === 'tree' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <Select label="criterion" data={treeCriterion} value={m.tree_criterion ?? 'gini'} onChange={(v)=>set({ tree_criterion: v })} />
              <Select label="splitter" data={treeSplitter} value={m.tree_splitter ?? 'best'} onChange={(v)=>set({ tree_splitter: v })} />
              <NumberInput label="max_depth" value={m.tree_max_depth ?? null} onChange={(v)=>set({ tree_max_depth: v })} />
              <NumberInput label="min_samples_split" value={m.tree_min_samples_split ?? 2} onChange={(v)=>set({ tree_min_samples_split: v })} />
              <NumberInput label="min_samples_leaf" value={m.tree_min_samples_leaf ?? 1} onChange={(v)=>set({ tree_min_samples_leaf: v })} />
              <NumberInput label="min_weight_fraction_leaf" value={m.tree_min_weight_fraction_leaf ?? 0.0} onChange={(v)=>set({ tree_min_weight_fraction_leaf: v })} step={0.01} min={0} max={1} />
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
                onChange={(mode)=>set({ tree_max_features: modeValToMaxFeat(mode, tMF.value) })}
              />
              {(tMF.mode === 'int' || tMF.mode === 'float') && (
                <NumberInput
                  label="max_features (value)"
                  value={tMF.value ?? null}
                  onChange={(v)=>set({ tree_max_features: modeValToMaxFeat(tMF.mode, v) })}
                  step={tMF.mode === 'int' ? 1 : 0.01}
                  allowDecimal={tMF.mode === 'float'}
                />
              )}
              <NumberInput label="max_leaf_nodes" value={m.tree_max_leaf_nodes ?? null} onChange={(v)=>set({ tree_max_leaf_nodes: v })} allowDecimal={false} />
              <NumberInput label="min_impurity_decrease" value={m.tree_min_impurity_decrease ?? 0.0} onChange={(v)=>set({ tree_min_impurity_decrease: v })} step={0.0001} />
              <Select
                label="class_weight"
                data={treeClassWeight}
                value={m.tree_class_weight == null ? 'none' : String(m.tree_class_weight)}
                onChange={(v)=>set({ tree_class_weight: fromSelectNullable(v) })}
              />
              <NumberInput label="ccp_alpha" value={m.tree_ccp_alpha ?? 0.0} onChange={(v)=>set({ tree_ccp_alpha: v })} step={0.0001} />
            </SimpleGrid>
          </Stack>
        )}

        {/* Random Forest */}
        {m.algo === 'forest' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <NumberInput label="n_estimators" value={m.forest_n_estimators ?? 100} onChange={(v)=>set({ forest_n_estimators: v })} allowDecimal={false} min={1} />
              <Select label="criterion" data={forestCriterion} value={m.forest_criterion ?? 'gini'} onChange={(v)=>set({ forest_criterion: v })} />
              <NumberInput label="max_depth" value={m.forest_max_depth ?? null} onChange={(v)=>set({ forest_max_depth: v })} />
              <NumberInput label="min_samples_split" value={m.forest_min_samples_split ?? 2} onChange={(v)=>set({ forest_min_samples_split: v })} />
              <NumberInput label="min_samples_leaf" value={m.forest_min_samples_leaf ?? 1} onChange={(v)=>set({ forest_min_samples_leaf: v })} />
              <NumberInput label="min_weight_fraction_leaf" value={m.forest_min_weight_fraction_leaf ?? 0.0} onChange={(v)=>set({ forest_min_weight_fraction_leaf: v })} step={0.01} min={0} max={1} />
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
                onChange={(mode)=>set({ forest_max_features: modeValToMaxFeat(mode, fMF.value) })}
              />
              {(fMF.mode === 'int' || fMF.mode === 'float') && (
                <NumberInput
                  label="max_features (value)"
                  value={fMF.value ?? null}
                  onChange={(v)=>set({ forest_max_features: modeValToMaxFeat(fMF.mode, v) })}
                  step={fMF.mode === 'int' ? 1 : 0.01}
                  allowDecimal={fMF.mode === 'float'}
                />
              )}
              <NumberInput label="max_leaf_nodes" value={m.forest_max_leaf_nodes ?? null} onChange={(v)=>set({ forest_max_leaf_nodes: v })} allowDecimal={false} />
              <NumberInput label="min_impurity_decrease" value={m.forest_min_impurity_decrease ?? 0.0} onChange={(v)=>set({ forest_min_impurity_decrease: v })} step={0.0001} />
              <Select
                label="class_weight"
                data={forestClassWeight}
                value={m.forest_class_weight == null ? 'none' : String(m.forest_class_weight)}
                onChange={(v)=>set({ forest_class_weight: fromSelectNullable(v) })}
              />
              <Checkbox label="bootstrap" checked={!!m.forest_bootstrap} onChange={(e)=>set({ forest_bootstrap: e.currentTarget.checked })} />
              <Checkbox label="oob_score" checked={!!m.forest_oob_score} onChange={(e)=>set({ forest_oob_score: e.currentTarget.checked })} />
              <NumberInput label="n_jobs" value={m.forest_n_jobs ?? null} onChange={(v)=>set({ forest_n_jobs: v })} allowDecimal={false} />
              <NumberInput label="ccp_alpha" value={m.forest_ccp_alpha ?? 0.0} onChange={(v)=>set({ forest_ccp_alpha: v })} step={0.0001} />
              <Checkbox label="warm_start" checked={!!m.forest_warm_start} onChange={(e)=>set({ forest_warm_start: e.currentTarget.checked })} />
              <NumberInput label="random_state" value={m.forest_random_state ?? null} onChange={(v)=>set({ forest_random_state: v })} allowDecimal={false} />
              <NumberInput label="max_samples" value={m.forest_max_samples ?? null} onChange={(v)=>set({ forest_max_samples: v })} />
            </SimpleGrid>
          </Stack>
        )}

        {/* KNN */}
        {m.algo === 'knn' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <NumberInput label="n_neighbors" value={m.knn_n_neighbors ?? 5} onChange={(v)=>set({ knn_n_neighbors: v })} allowDecimal={false} min={1} />
              <Select label="weights" data={knnWeights} value={m.knn_weights ?? 'uniform'} onChange={(v)=>set({ knn_weights: v })} />
              <Select label="algorithm" data={knnAlgorithm} value={m.knn_algorithm ?? 'auto'} onChange={(v)=>set({ knn_algorithm: v })} />
              <NumberInput label="leaf_size" value={m.knn_leaf_size ?? 30} onChange={(v)=>set({ knn_leaf_size: v })} allowDecimal={false} min={1} />
              <NumberInput label="p" value={m.knn_p ?? 2} onChange={(v)=>set({ knn_p: v })} allowDecimal={false} min={1} />
              <Select label="metric" data={knnMetric} value={m.knn_metric ?? 'minkowski'} onChange={(v)=>set({ knn_metric: v })} />
              <NumberInput label="n_jobs" value={m.knn_n_jobs ?? null} onChange={(v)=>set({ knn_n_jobs: v })} allowDecimal={false} />
            </SimpleGrid>
          </Stack>
        )}

        <Divider my="xs" />
        <Text size="xs" c="dimmed">Choices come from backend schema when available.</Text>
      </Stack>
    </Card>
  );
}
