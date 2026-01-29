import {
  Card,
  Stack,
  Group,
  Text,
  Divider,
  Select,
  NumberInput,
  Checkbox,
  SimpleGrid,
  Box,
  Button,
  TextInput,
} from '@mantine/core';
import { useEffect, useMemo, useState } from 'react';
import { useDataStore } from '../state/useDataStore.js';
import ModelHelpText, {
  ModelIntroText,
} from './helpers/helpTexts/ModelHelpText.jsx';

/** ---------------- helpers ---------------- **/

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
    if (Array.isArray(p.enum)) return fallback && fallback.length ? fallback : p.enum;
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
    if (v === null) {
      hasNull = true;
      continue;
    }
    out.push({ value: String(v), label: String(v) });
  }
  if (hasNull && includeNoneLabel) out.unshift({ value: 'none', label: 'none' });
  return out;
}

function fromSelectNullable(v) {
  return v === 'none' ? null : v;
}

function parseCsvFloats(s) {
  if (s == null) return null;
  const txt = String(s).trim();
  if (!txt) return null;
  const parts = txt
    .split(',')
    .map((x) => x.trim())
    .filter((x) => x.length);
  if (!parts.length) return null;
  const vals = parts
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x));
  return vals.length ? vals : null;
}

function formatCsvFloats(arr) {
  if (!Array.isArray(arr) || !arr.length) return '';
  return arr.map((x) => String(x)).join(', ');
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

// User-friendly names for the Algorithm dropdown.
// Values must remain the internal algo keys used by the backend.
const ALGO_LABELS = {
  // -------- classifiers --------
  logreg: 'Logistic Regression',
  ridge: 'Ridge Classifier',
  sgd: 'SGD Classifier',
  svm: 'Support Vector Machine (SVC)',
  tree: 'Decision Tree',
  forest: 'Random Forest',
  extratrees: 'Extra Trees Classifier',
  hgb: 'Histogram Gradient Boosting',
  knn: 'k-Nearest Neighbors',
  gnb: 'Gaussian Naive Bayes',

  // -------- regressors --------
  linreg: 'Linear Regression',
  ridgereg: 'Ridge Regression',
  ridgecv: 'Ridge Regression (CV)',
  enet: 'Elastic Net',
  enetcv: 'Elastic Net (CV)',
  lasso: 'Lasso',
  lassocv: 'Lasso (CV)',
  bayridge: 'Bayesian Ridge',
  svr: 'Support Vector Regression (SVR)',
  linsvr: 'Linear SVR',
  knnreg: 'k-Nearest Neighbors Regressor',
  treereg: 'Decision Tree Regressor',
  rfreg: 'Random Forest Regressor',

  // -------- unsupervised --------
  kmeans: 'K-Means',
  dbscan: 'DBSCAN',
  spectral: 'Spectral Clustering',
  agglo: 'Agglomerative Clustering',
  gmm: 'Gaussian Mixture',
  bgmm: 'Bayesian Gaussian Mixture',
  meanshift: 'MeanShift',
  birch: 'Birch',
};

function algoKeyToLabel(algo) {
  if (!algo) return '';
  return ALGO_LABELS[algo] ?? String(algo);
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

export default function ModelSelectionCard({
  model,
  onChange,
  schema,
  enums,
  models,
  taskOverride = null,
  showHelp = false,
}) {
  const m = model || {};
  // IMPORTANT:
  // - set(patch) merges into the current model.
  // - replace(next) fully replaces the model.
  //   We use replace() when switching algorithms to avoid "parameter leakage"
  //   across models that share field names (e.g. LogisticRegression.solver -> RidgeClassifier.solver,
  //   or Forest.max_features -> HistGradientBoosting.max_features).
  const set = (patch) => onChange?.({ ...m, ...patch });
  const replace = (next) => onChange?.({ ...(next || {}) });

  const cloneDefaults = (obj) => {
    if (!obj) return obj;
    // Prefer structuredClone when available (keeps numbers as numbers), else JSON clone.
    if (typeof structuredClone === 'function') return structuredClone(obj);
    return JSON.parse(JSON.stringify(obj));
  };

  const applyAlgo = (algo) => {
    const def = models?.defaults?.[algo];
    if (def && typeof def === 'object') {
      replace(cloneDefaults(def));
    } else {
      replace({ algo });
    }
  };

  const inferredTask = useDataStore(
    (s) => s.taskSelected || s.inspectReport?.task_inferred || null,
  );
  const effectiveTask = taskOverride ?? inferredTask;

  // Prefer schema -> defaults -> known fallback
  const schemaAlgos = algoListFromSchema(schema);
  const defaultsAlgos = models?.defaults ? Object.keys(models.defaults) : null;
  const available = new Set(
    schemaAlgos ||
      defaultsAlgos ||
      [
        // classifiers
        'logreg',
        'ridge',
        'sgd',
        'svm',
        'tree',
        'forest',
        'extratrees',
        'hgb',
        'knn',
        'gnb',
        // regressors
        'linreg',
        'ridgereg',
        'ridgecv',
        'enet',
        'enetcv',
        'lasso',
        'lassocv',
        'bayridge',
        'svr',
        'linsvr',
        'knnreg',
        'treereg',
        'rfreg',

        // unsupervised
        'kmeans',
        'dbscan',
        'spectral',
        'agglo',
        'gmm',
        'bgmm',
        'meanshift',
        'birch',
      ],
  );

  // Preferred ordering (task-aware), append any extras after
  const preferredByTask = {
    classification: [
      'logreg',
      'ridge',
      'sgd',
      'svm',
      'tree',
      'forest',
      'extratrees',
      'hgb',
      'knn',
      'gnb',
    ],
    regression: [
      'linreg',
      'ridgereg',
      'ridgecv',
      'lasso',
      'lassocv',
      'enet',
      'enetcv',
      'bayridge',
      'svr',
      'linsvr',
      'knnreg',
      'treereg',
      'rfreg',
    ],
    unsupervised: [
      'kmeans',
      'gmm',
      'bgmm',
      'agglo',
      'spectral',
      'birch',
      'meanshift',
      'dbscan',
    ],
  };

  const preferred =
    preferredByTask[effectiveTask] ||
    [
      ...preferredByTask.classification,
      ...preferredByTask.regression,
      ...preferredByTask.unsupervised,
    ];

  const orderedAll = [
    ...preferred.filter((a) => available.has(a)),
    ...Array.from(available).filter((a) => !preferred.includes(a)),
  ];

  // Filter by task using backend-provided meta[algo].task
  const meta = models?.meta || {};
  const matchesTask = (algo) => {
    if (!effectiveTask) return true; // no filter if task unknown
    let t = meta[algo]?.task; // 'classification' | 'regression' | 'unsupervised' | ...
    if (!t) return true; // if backend didn’t annotate, don’t hide it

    // Backwards-compat: older backends may use "clustering".
    if (t === 'clustering') t = 'unsupervised';

    if (Array.isArray(t)) return t.includes(effectiveTask);
    return t === effectiveTask;
  };
  const ordered = orderedAll.filter(matchesTask);

  // Final visible algo list for this card
  const visibleAlgos = ordered.length ? ordered : orderedAll;
  const visibleKey = useMemo(
    () => (visibleAlgos && visibleAlgos.length ? visibleAlgos.join('|') : ''),
    [visibleAlgos],
  );
  const algoData = (visibleAlgos ?? []).map((a) => ({
    value: String(a),
    label: algoKeyToLabel(a),
  }));

  // Ensure selection is valid when task / availability changes
  useEffect(() => {
    if (!visibleAlgos.length) return;
    const current = m.algo;

    // If nothing matches the task filter (e.g. older meta), fall back to "available" only.
    const hasTaskFilteredList = ordered.length > 0;

    const isValid =
      current &&
      visibleAlgos.includes(current) &&
      (hasTaskFilteredList ? matchesTask(current) : true);
    if (!isValid) {
      applyAlgo(visibleAlgos[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveTask, visibleKey, meta, m.algo]);

  const sub = getAlgoSchema(schema, m.algo);

  // Logistic Regression enums
  const lrPenalty = toSelectData(
    enumFromSubSchema(sub, 'penalty', enums?.PenaltyName),
  );
  const lrSolver = toSelectData(
    enumFromSubSchema(sub, 'solver', enums?.LogRegSolver),
  );
  const lrClassWeight = toSelectData(
    enumFromSubSchema(
      sub,
      'class_weight',
      enums?.ClassWeightBalanced ?? ['balanced', null],
    ),
    { includeNoneLabel: true },
  );

  // SVM enums
  const svmKernel = toSelectData(
    enumFromSubSchema(sub, 'kernel', enums?.SVMKernel),
  );
  const svmDecisionShape = toSelectData(
    enumFromSubSchema(sub, 'decision_function_shape', enums?.SVMDecisionShape),
  );
  const svmClassWeight = toSelectData(
    enumFromSubSchema(
      sub,
      'class_weight',
      enums?.ClassWeightBalanced ?? ['balanced', null],
    ),
    { includeNoneLabel: true },
  );

  // Tree enums
  const treeCriterion = toSelectData(
    enumFromSubSchema(sub, 'criterion', enums?.TreeCriterion),
  );
  const treeSplitter = toSelectData(
    enumFromSubSchema(sub, 'splitter', enums?.TreeSplitter),
  );
  const treeClassWeight = toSelectData(
    enumFromSubSchema(
      sub,
      'class_weight',
      enums?.ClassWeightBalanced ?? ['balanced', null],
    ),
    { includeNoneLabel: true },
  );

  // Forest enums
  const forestCriterion = toSelectData(
    enumFromSubSchema(sub, 'criterion', enums?.TreeCriterion),
  );
  const forestClassWeight = toSelectData(
    enumFromSubSchema(
      sub,
      'class_weight',
      enums?.ForestClassWeight ?? ['balanced', 'balanced_subsample', null],
    ),
    { includeNoneLabel: true },
  );

  // KNN enums
  const knnWeights = toSelectData(
    enumFromSubSchema(sub, 'weights', enums?.KNNWeights),
  );
  const knnAlgorithm = toSelectData(
    enumFromSubSchema(sub, 'algorithm', enums?.KNNAlgorithm),
  );
  const knnMetric = toSelectData(
    enumFromSubSchema(sub, 'metric', enums?.KNNMetric),
  );

  // GaussianNB (no enums, but keep consistent)

  // RidgeClassifier enums
  const ridgeSolver = toSelectData(
    enumFromSubSchema(sub, 'solver', enums?.RidgeSolver),
  );
  const ridgeClassWeight = toSelectData(
    enumFromSubSchema(
      sub,
      'class_weight',
      enums?.ClassWeightBalanced ?? ['balanced', null],
    ),
    { includeNoneLabel: true },
  );

  // SGDClassifier enums
  const sgdLoss = toSelectData(enumFromSubSchema(sub, 'loss', enums?.SGDLoss));
  const sgdPenalty = toSelectData(
    enumFromSubSchema(sub, 'penalty', enums?.SGDPenalty),
  );
  const sgdLR = toSelectData(
    enumFromSubSchema(sub, 'learning_rate', enums?.SGDLearningRate),
  );
  const sgdClassWeight = toSelectData(
    enumFromSubSchema(
      sub,
      'class_weight',
      enums?.ClassWeightBalanced ?? ['balanced', null],
    ),
    { includeNoneLabel: true },
  );

  // HistGradientBoostingClassifier enums
  const hgbLoss = toSelectData(enumFromSubSchema(sub, 'loss', enums?.HGBLoss));

  // Regressor enums
  const regTreeCriterion = toSelectData(
    enumFromSubSchema(sub, 'criterion', enums?.RegTreeCriterion),
  );
  const cdSelection = toSelectData(
    enumFromSubSchema(sub, 'selection', enums?.CoordinateDescentSelection),
  );
  const linsvrLoss = toSelectData(
    enumFromSubSchema(sub, 'loss', enums?.LinearSVRLoss),
  );
  const ridgecvGcvMode = toSelectData(
    enumFromSubSchema(
      sub,
      'gcv_mode',
      ['auto', 'svd', 'eigen', null],
    ),
    { includeNoneLabel: true },
  );


  const gammaMode =
    typeof m.gamma === 'number' ? 'numeric' : (m.gamma ?? 'scale');
  const gammaValue = typeof m.gamma === 'number' ? m.gamma : 0.1;

  const tMF = maxFeatToModeVal(m.max_features);
  const fMF = maxFeatToModeVal(m.max_features);

  const sgdAvgMode =
    typeof m.average === 'number' ? 'int' : m.average ? 'true' : 'false';
  const sgdAvgValue = typeof m.average === 'number' ? m.average : 10;

  const hgbES =
    m.early_stopping === 'auto' ? 'auto' : m.early_stopping ? 'true' : 'false';
  const hgbMaxFeaturesValue =
    typeof m.max_features === 'number' ? m.max_features : 1.0;

  // Toggle for showing/hiding detailed help (block C) when help is enabled
  const [showDetails, setShowDetails] = useState(false);

  // Controls + parameter editors
  const controlsBody = (
    <>
      <Select
        label="Algorithm"
        data={algoData}
        value={m.algo ?? (algoData[0]?.value ?? 'logreg')}
        onChange={(algo) => {
          if (algo) applyAlgo(algo);
        }}
      />

      {/* Logistic Regression */}
      {m.algo === 'logreg' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="C (strength)"
              value={m.C ?? 1.0}
              onChange={(v) => set({ C: v })}
              min={0}
              step={0.1}
            />
            <Select
              label="Penalty"
              data={lrPenalty}
              value={m.penalty ?? 'l2'}
              onChange={(v) => set({ penalty: v })}
            />
            <Select
              label="Solver"
              data={lrSolver}
              value={m.solver ?? 'lbfgs'}
              onChange={(v) => set({ solver: v })}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? 1000}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Class weight"
              data={lrClassWeight}
              value={m.class_weight == null ? 'none' : String(m.class_weight)}
              onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
            />
            {m.penalty === 'elasticnet' && (
              <NumberInput
                label="L1 ratio"
                value={m.l1_ratio ?? 0.5}
                onChange={(v) => set({ l1_ratio: v })}
                min={0}
                max={1}
                step={0.01}
              />
            )}
          </SimpleGrid>
        </Stack>
      )}

      {/* SVM */}
      {m.algo === 'svm' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <Select
              label="Kernel"
              data={svmKernel}
              value={m.kernel ?? 'rbf'}
              onChange={(v) => set({ kernel: v })}
            />
            <NumberInput
              label="C (penalty)"
              value={m.C ?? 1.0}
              onChange={(v) => set({ C: v })}
              min={0}
              step={0.1}
            />
            <NumberInput
              label="Degree (poly)"
              value={m.degree ?? 3}
              onChange={(v) => set({ degree: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Gamma mode"
              data={[
                { value: 'scale', label: 'scale' },
                { value: 'auto', label: 'auto' },
                { value: 'numeric', label: 'numeric' },
              ]}
              value={gammaMode}
              onChange={(mode) => {
                if (mode === 'numeric') {
                  set({
                    gamma:
                      typeof gammaValue === 'number' ? gammaValue : 0.1,
                  });
                } else {
                  set({ gamma: mode });
                }
              }}
            />
            {gammaMode === 'numeric' && (
              <NumberInput
                label="Gamma value"
                value={gammaValue}
                onChange={(v) => set({ gamma: v })}
                min={0}
                step={0.001}
              />
            )}
            <NumberInput
              label="Coef0"
              value={m.coef0 ?? 0.0}
              onChange={(v) => set({ coef0: v })}
              step={0.001}
            />
            <Checkbox
              label="Use shrinking"
              checked={!!m.shrinking}
              onChange={(e) =>
                set({ shrinking: e.currentTarget.checked })
              }
            />
            <Checkbox
              label="Enable probability"
              checked={!!m.probability}
              onChange={(e) =>
                set({ probability: e.currentTarget.checked })
              }
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-3}
              onChange={(v) => set({ tol: v })}
              step={0.0001}
            />
            <NumberInput
              label="Cache size (MB)"
              value={m.cache_size ?? 200.0}
              onChange={(v) => set({ cache_size: v })}
            />
            <Select
              label="Class weight"
              data={svmClassWeight}
              value={m.class_weight == null ? 'none' : String(m.class_weight)}
              onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? -1}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
            />
            <Select
              label="Decision shape"
              data={svmDecisionShape}
              value={m.decision_function_shape ?? 'ovr'}
              onChange={(v) =>
                set({ decision_function_shape: v })
              }
            />
            <Checkbox
              label="Break ties"
              checked={!!m.break_ties}
              onChange={(e) =>
                set({ break_ties: e.currentTarget.checked })
              }
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Decision Tree */}
      {m.algo === 'tree' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <Select
              label="Criterion"
              data={treeCriterion}
              value={m.criterion ?? 'gini'}
              onChange={(v) => set({ criterion: v })}
            />
            <Select
              label="Splitter"
              data={treeSplitter}
              value={m.splitter ?? 'best'}
              onChange={(v) => set({ splitter: v })}
            />
            <NumberInput
              label="Max depth"
              value={m.max_depth ?? null}
              onChange={(v) => set({ max_depth: v })}
            />
            <NumberInput
              label="Min samples split"
              value={m.min_samples_split ?? 2}
              onChange={(v) => set({ min_samples_split: v })}
            />
            <NumberInput
              label="Min samples leaf"
              value={m.min_samples_leaf ?? 1}
              onChange={(v) => set({ min_samples_leaf: v })}
            />
            <NumberInput
              label="Min weight fraction"
              value={m.min_weight_fraction_leaf ?? 0.0}
              onChange={(v) =>
                set({ min_weight_fraction_leaf: v })
              }
              step={0.01}
              min={0}
              max={1}
            />
            <Select
              label="Max features mode"
              data={[
                { value: 'sqrt', label: 'sqrt' },
                { value: 'log2', label: 'log2' },
                { value: 'int', label: 'int' },
                { value: 'float', label: 'float' },
                { value: 'none', label: 'none' },
              ]}
              value={tMF.mode}
              onChange={(mode) =>
                set({ max_features: modeValToMaxFeat(mode, tMF.value) })
              }
            />
            {(tMF.mode === 'int' || tMF.mode === 'float') && (
              <NumberInput
                label="Max features value"
                value={tMF.value ?? null}
                onChange={(v) =>
                  set({
                    max_features: modeValToMaxFeat(tMF.mode, v),
                  })
                }
                step={tMF.mode === 'int' ? 1 : 0.01}
                allowDecimal={tMF.mode === 'float'}
              />
            )}
            <NumberInput
              label="Max leaf nodes"
              value={m.max_leaf_nodes ?? null}
              onChange={(v) => set({ max_leaf_nodes: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Min impurity decrease"
              value={m.min_impurity_decrease ?? 0.0}
              onChange={(v) =>
                set({ min_impurity_decrease: v })
              }
              step={0.0001}
            />
            <Select
              label="Class weight"
              data={treeClassWeight}
              value={m.class_weight == null ? 'none' : String(m.class_weight)}
              onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
            />
            <NumberInput
              label="CCP alpha"
              value={m.ccp_alpha ?? 0.0}
              onChange={(v) => set({ ccp_alpha: v })}
              step={0.0001}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Random Forest */}
      {m.algo === 'forest' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Trees (n_estimators)"
              value={m.n_estimators ?? 100}
              onChange={(v) => set({ n_estimators: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Criterion"
              data={forestCriterion}
              value={m.criterion ?? 'gini'}
              onChange={(v) => set({ criterion: v })}
            />
            <NumberInput
              label="Max depth"
              value={m.max_depth ?? null}
              onChange={(v) => set({ max_depth: v })}
            />
            <NumberInput
              label="Min samples split"
              value={m.min_samples_split ?? 2}
              onChange={(v) => set({ min_samples_split: v })}
            />
            <NumberInput
              label="Min samples leaf"
              value={m.min_samples_leaf ?? 1}
              onChange={(v) => set({ min_samples_leaf: v })}
            />
            <NumberInput
              label="Min weight fraction"
              value={m.min_weight_fraction_leaf ?? 0.0}
              onChange={(v) =>
                set({ min_weight_fraction_leaf: v })
              }
              step={0.01}
              min={0}
              max={1}
            />
            <Select
              label="Max features mode"
              data={[
                { value: 'sqrt', label: 'sqrt' },
                { value: 'log2', label: 'log2' },
                { value: 'int', label: 'int' },
                { value: 'float', label: 'float' },
                { value: 'none', label: 'none' },
              ]}
              value={fMF.mode}
              onChange={(mode) =>
                set({ max_features: modeValToMaxFeat(mode, fMF.value) })
              }
            />
            {(fMF.mode === 'int' || fMF.mode === 'float') && (
              <NumberInput
                label="Max features value"
                value={fMF.value ?? null}
                onChange={(v) =>
                  set({
                    max_features: modeValToMaxFeat(fMF.mode, v),
                  })
                }
                step={fMF.mode === 'int' ? 1 : 0.01}
                allowDecimal={fMF.mode === 'float'}
              />
            )}
            <NumberInput
              label="Max leaf nodes"
              value={m.max_leaf_nodes ?? null}
              onChange={(v) => set({ max_leaf_nodes: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Min impurity decrease"
              value={m.min_impurity_decrease ?? 0.0}
              onChange={(v) =>
                set({ min_impurity_decrease: v })
              }
              step={0.0001}
            />
            <Select
              label="Class weight"
              data={forestClassWeight}
              value={m.class_weight == null ? 'none' : String(m.class_weight)}
              onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
            />
            <Checkbox
              label="Use bootstrap"
              checked={!!m.bootstrap}
              onChange={(e) =>
                set({ bootstrap: e.currentTarget.checked })
              }
            />
            <Checkbox
              label="OOB score"
              checked={!!m.oob_score}
              onChange={(e) =>
                set({ oob_score: e.currentTarget.checked })
              }
            />
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
            <Checkbox
              label="Warm start"
              checked={!!m.warm_start}
              onChange={(e) =>
                set({ warm_start: e.currentTarget.checked })
              }
            />
            <NumberInput
              label="CCP alpha"
              value={m.ccp_alpha ?? 0.0}
              onChange={(v) => set({ ccp_alpha: v })}
              step={0.0001}
            />
            <NumberInput
              label="Max samples"
              value={m.max_samples ?? null}
              onChange={(v) => set({ max_samples: v })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* KNN */}
      {m.algo === 'knn' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Neighbours"
              value={m.n_neighbors ?? 5}
              onChange={(v) => set({ n_neighbors: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Weights"
              data={knnWeights}
              value={m.weights ?? 'uniform'}
              onChange={(v) => set({ weights: v })}
            />
            <Select
              label="Algorithm"
              data={knnAlgorithm}
              value={m.algorithm ?? 'auto'}
              onChange={(v) => set({ algorithm: v })}
            />
            <NumberInput
              label="Leaf size (leaf_size)"
              value={m.leaf_size ?? 30}
              onChange={(v) => set({ leaf_size: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="p"
              value={m.p ?? 2}
              onChange={(v) => set({ p: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Metric"
              data={knnMetric}
              value={m.metric ?? 'minkowski'}
              onChange={(v) => set({ metric: v })}
            />
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Gaussian Naive Bayes */}
      {m.algo === 'gnb' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Variance smoothing"
              value={m.var_smoothing ?? 1e-9}
              onChange={(v) => set({ var_smoothing: v })}
              step={1e-9}
              min={0}
            />
            <TextInput
              label="Priors (comma-separated)"
              placeholder="e.g. 0.2, 0.8"
              value={formatCsvFloats(m.priors)}
              onChange={(e) => set({ priors: parseCsvFloats(e.currentTarget.value) })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Ridge Classifier */}
      {m.algo === 'ridge' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Alpha (regularization)"
              value={m.alpha ?? 1.0}
              onChange={(v) => set({ alpha: v })}
              min={0}
              step={0.1}
            />
            <Select
              label="Solver"
              data={ridgeSolver}
              value={m.solver ?? 'auto'}
              onChange={(v) => set({ solver: v })}
            />
            <Checkbox
              label="Fit intercept"
              checked={m.fit_intercept ?? true}
              onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
            />
            <Select
              label="Class weight"
              data={ridgeClassWeight}
              value={m.class_weight == null ? 'none' : String(m.class_weight)}
              onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? null}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-4}
              onChange={(v) => set({ tol: v })}
              step={1e-5}
              min={0}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* SGD Classifier */}
      {m.algo === 'sgd' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <Select
              label="Loss"
              data={sgdLoss}
              value={m.loss ?? 'hinge'}
              onChange={(v) => set({ loss: v })}
            />
            <Select
              label="Penalty"
              data={sgdPenalty}
              value={m.penalty ?? 'l2'}
              onChange={(v) => set({ penalty: v })}
            />
            <NumberInput
              label="Alpha"
              value={m.alpha ?? 0.0001}
              onChange={(v) => set({ alpha: v })}
              min={0}
              step={0.0001}
            />
            <NumberInput
              label="L1 ratio"
              value={m.l1_ratio ?? 0.15}
              onChange={(v) => set({ l1_ratio: v })}
              min={0}
              max={1}
              step={0.01}
            />
            <Checkbox
              label="Fit intercept"
              checked={m.fit_intercept ?? true}
              onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? 1000}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-3}
              onChange={(v) => set({ tol: v })}
              step={0.0001}
              min={0}
            />
            <Checkbox
              label="Shuffle"
              checked={m.shuffle ?? true}
              onChange={(e) => set({ shuffle: e.currentTarget.checked })}
            />
            <Select
              label="Learning rate"
              data={sgdLR}
              value={m.learning_rate ?? 'optimal'}
              onChange={(v) => set({ learning_rate: v })}
            />
            <NumberInput
              label="Eta0"
              value={m.eta0 ?? 0.0}
              onChange={(v) => set({ eta0: v })}
              min={0}
              step={0.01}
            />
            <NumberInput
              label="Power t"
              value={m.power_t ?? 0.5}
              onChange={(v) => set({ power_t: v })}
              step={0.01}
            />
            <Checkbox
              label="Early stopping"
              checked={!!m.early_stopping}
              onChange={(e) => set({ early_stopping: e.currentTarget.checked })}
            />
            <NumberInput
              label="Validation fraction"
              value={m.validation_fraction ?? 0.1}
              onChange={(v) => set({ validation_fraction: v })}
              min={0}
              max={1}
              step={0.01}
            />
            <NumberInput
              label="No-change rounds"
              value={m.n_iter_no_change ?? 5}
              onChange={(v) => set({ n_iter_no_change: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Class weight"
              data={sgdClassWeight}
              value={m.class_weight == null ? 'none' : String(m.class_weight)}
              onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
            />
            <Select
              label="Average"
              data={[
                { value: 'false', label: 'false' },
                { value: 'true', label: 'true' },
                { value: 'int', label: 'int' },
              ]}
              value={sgdAvgMode}
              onChange={(mode) => {
                if (mode === 'int') set({ average: sgdAvgValue });
                else if (mode === 'true') set({ average: true });
                else set({ average: false });
              }}
            />
            {sgdAvgMode === 'int' && (
              <NumberInput
                label="Average window"
                value={sgdAvgValue}
                onChange={(v) => set({ average: v })}
                allowDecimal={false}
                min={1}
              />
            )}
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Extra Trees */}
      {m.algo === 'extratrees' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Trees (n_estimators)"
              value={m.n_estimators ?? 100}
              onChange={(v) => set({ n_estimators: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Criterion"
              data={forestCriterion}
              value={m.criterion ?? 'gini'}
              onChange={(v) => set({ criterion: v })}
            />
            <NumberInput
              label="Max depth"
              value={m.max_depth ?? null}
              onChange={(v) => set({ max_depth: v })}
            />
            <NumberInput
              label="Min samples split"
              value={m.min_samples_split ?? 2}
              onChange={(v) => set({ min_samples_split: v })}
            />
            <NumberInput
              label="Min samples leaf"
              value={m.min_samples_leaf ?? 1}
              onChange={(v) => set({ min_samples_leaf: v })}
            />
            <Select
              label="Max features mode"
              data={[
                { value: 'sqrt', label: 'sqrt' },
                { value: 'log2', label: 'log2' },
                { value: 'int', label: 'int' },
                { value: 'float', label: 'float' },
                { value: 'none', label: 'none' },
              ]}
              value={fMF.mode}
              onChange={(mode) => set({ max_features: modeValToMaxFeat(mode, fMF.value) })}
            />
            {(fMF.mode === 'int' || fMF.mode === 'float') && (
              <NumberInput
                label="Max features value"
                value={fMF.value ?? null}
                onChange={(v) => set({ max_features: modeValToMaxFeat(fMF.mode, v) })}
                step={fMF.mode === 'int' ? 1 : 0.01}
                allowDecimal={fMF.mode === 'float'}
              />
            )}
            <NumberInput
              label="Max leaf nodes"
              value={m.max_leaf_nodes ?? null}
              onChange={(v) => set({ max_leaf_nodes: v })}
              allowDecimal={false}
            />
            <Select
              label="Class weight"
              data={forestClassWeight}
              value={m.class_weight == null ? 'none' : String(m.class_weight)}
              onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
            />
            <Checkbox
              label="Use bootstrap"
              checked={!!m.bootstrap}
              onChange={(e) => set({ bootstrap: e.currentTarget.checked })}
            />
            <Checkbox
              label="OOB score"
              checked={!!m.oob_score}
              onChange={(e) => set({ oob_score: e.currentTarget.checked })}
            />
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* HistGradientBoosting */}
      {m.algo === 'hgb' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <Select
              label="Loss"
              data={hgbLoss}
              value={m.loss ?? 'log_loss'}
              onChange={(v) => set({ loss: v })}
            />
            <NumberInput
              label="Learning rate"
              value={m.learning_rate ?? 0.1}
              onChange={(v) => set({ learning_rate: v })}
              min={0}
              step={0.01}
            />
            <NumberInput
              label="Iterations (max_iter)"
              value={m.max_iter ?? 100}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Max leaf nodes"
              value={m.max_leaf_nodes ?? 31}
              onChange={(v) => set({ max_leaf_nodes: v })}
              allowDecimal={false}
              min={2}
            />
            <NumberInput
              label="Max depth"
              value={m.max_depth ?? null}
              onChange={(v) => set({ max_depth: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Min samples leaf"
              value={m.min_samples_leaf ?? 20}
              onChange={(v) => set({ min_samples_leaf: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="L2 regularization"
              value={m.l2_regularization ?? 0.0}
              onChange={(v) => set({ l2_regularization: v })}
              min={0}
              step={0.01}
            />
            <NumberInput
              label="Max features (fraction)"
              value={hgbMaxFeaturesValue}
              onChange={(v) => {
                // Mantine NumberInput may provide a string; backend expects a number.
                if (v == null || v === '') set({ max_features: null });
                else set({ max_features: typeof v === 'number' ? v : Number(v) });
              }}
              min={0}
              max={1}
              step={0.01}
            />
            <NumberInput
              label="Max bins"
              value={m.max_bins ?? 255}
              onChange={(v) => set({ max_bins: v })}
              allowDecimal={false}
              min={2}
            />
            <Select
              label="Early stopping"
              data={[
                { value: 'auto', label: 'auto' },
                { value: 'true', label: 'true' },
                { value: 'false', label: 'false' },
              ]}
              value={hgbES}
              onChange={(v) => {
                if (v === 'auto') set({ early_stopping: 'auto' });
                else if (v === 'true') set({ early_stopping: true });
                else set({ early_stopping: false });
              }}
            />
            <TextInput
              label="Scoring"
              placeholder="loss"
              value={m.scoring ?? 'loss'}
              onChange={(e) => {
                const t = e.currentTarget.value;
                set({ scoring: t === '' ? null : t });
              }}
            />
            <NumberInput
              label="Validation fraction"
              value={m.validation_fraction ?? 0.1}
              onChange={(v) => set({ validation_fraction: v })}
              min={0}
              max={1}
              step={0.01}
            />
            <NumberInput
              label="No-change rounds"
              value={m.n_iter_no_change ?? 10}
              onChange={(v) => set({ n_iter_no_change: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Tolerance"
              value={m.tol ?? 1e-7}
              onChange={(v) => set({ tol: v })}
              step={1e-7}
              min={0}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Linear Regression */}
      {m.algo === 'linreg' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <Checkbox
              label="Fit intercept"
              checked={!!m.fit_intercept}
              onChange={(e) =>
                set({ fit_intercept: e.currentTarget.checked })
              }
            />
            <Checkbox
              label="Copy X"
              checked={m.copy_X ?? true}
              onChange={(e) =>
                set({ copy_X: e.currentTarget.checked })
              }
            />
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
            <Checkbox
              label="Positive coefficients"
              checked={!!m.positive}
              onChange={(e) =>
                set({ positive: e.currentTarget.checked })
              }
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Ridge Regression */}
      {m.algo === 'ridgereg' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Alpha (regularization)"
              value={m.alpha ?? 1.0}
              onChange={(v) => set({ alpha: v })}
              min={0}
              step={0.1}
            />
            <Select
              label="Solver"
              data={ridgeSolver}
              value={m.solver ?? 'auto'}
              onChange={(v) => set({ solver: v })}
            />
            <Checkbox
              label="Fit intercept"
              checked={m.fit_intercept ?? true}
              onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? null}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-4}
              onChange={(v) => set({ tol: v })}
              step={1e-5}
              min={0}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
            <Checkbox
              label="Positive coefficients"
              checked={!!m.positive}
              onChange={(e) => set({ positive: e.currentTarget.checked })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Ridge Regression (CV) */}
      {m.algo === 'ridgecv' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <TextInput
              label="Alphas (comma-separated)"
              placeholder="e.g. 0.1, 1.0, 10.0"
              value={formatCsvFloats(m.alphas)}
              onChange={(e) => set({ alphas: parseCsvFloats(e.currentTarget.value) || [0.1, 1.0, 10.0] })}
            />
            <Checkbox
              label="Fit intercept"
              checked={m.fit_intercept ?? true}
              onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
            />
            <TextInput
              label="Scoring"
              placeholder="(optional)"
              value={m.scoring ?? ''}
              onChange={(e) => {
                const t = e.currentTarget.value;
                set({ scoring: t === '' ? null : t });
              }}
            />
            <NumberInput
              label="CV folds"
              value={m.cv ?? null}
              onChange={(v) => set({ cv: v })}
              allowDecimal={false}
              min={2}
            />
            <Select
              label="GCV mode"
              data={ridgecvGcvMode}
              value={m.gcv_mode == null ? 'none' : String(m.gcv_mode)}
              onChange={(v) => set({ gcv_mode: fromSelectNullable(v) })}
            />
            <Checkbox
              label="Alpha per target"
              checked={!!m.alpha_per_target}
              onChange={(e) => set({ alpha_per_target: e.currentTarget.checked })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Elastic Net */}
      {m.algo === 'enet' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Alpha"
              value={m.alpha ?? 1.0}
              onChange={(v) => set({ alpha: v })}
              min={0}
              step={0.1}
            />
            <NumberInput
              label="L1 ratio"
              value={m.l1_ratio ?? 0.5}
              onChange={(v) => set({ l1_ratio: v })}
              min={0}
              max={1}
              step={0.05}
            />
            <Checkbox
              label="Fit intercept"
              checked={m.fit_intercept ?? true}
              onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
            />
            <Select
              label="Selection"
              data={cdSelection}
              value={m.selection ?? 'cyclic'}
              onChange={(v) => set({ selection: v })}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? 1000}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-4}
              onChange={(v) => set({ tol: v })}
              step={1e-5}
              min={0}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
            <Checkbox
              label="Positive coefficients"
              checked={!!m.positive}
              onChange={(e) => set({ positive: e.currentTarget.checked })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Elastic Net (CV) */}
      {m.algo === 'enetcv' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <TextInput
              label="L1 ratio list (comma-separated)"
              placeholder="e.g. 0.1, 0.5, 0.9"
              value={formatCsvFloats(m.l1_ratio)}
              onChange={(e) => {
                const vals = parseCsvFloats(e.currentTarget.value);
                set({ l1_ratio: vals || [0.1, 0.5, 0.9] });
              }}
            />
            <NumberInput
              label="eps"
              value={m.eps ?? 1e-3}
              onChange={(v) => set({ eps: v })}
              min={0}
              step={1e-4}
            />
            <NumberInput
              label="n_alphas"
              value={m.n_alphas ?? 100}
              onChange={(v) => set({ n_alphas: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="CV folds"
              value={m.cv ?? 5}
              onChange={(v) => set({ cv: v })}
              allowDecimal={false}
              min={2}
            />
            <Checkbox
              label="Fit intercept"
              checked={m.fit_intercept ?? true}
              onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
            />
            <Select
              label="Selection"
              data={cdSelection}
              value={m.selection ?? 'cyclic'}
              onChange={(v) => set({ selection: v })}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? 1000}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-4}
              onChange={(v) => set({ tol: v })}
              step={1e-5}
              min={0}
            />
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
            <Checkbox
              label="Positive coefficients"
              checked={!!m.positive}
              onChange={(e) => set({ positive: e.currentTarget.checked })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Lasso */}
      {m.algo === 'lasso' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Alpha"
              value={m.alpha ?? 1.0}
              onChange={(v) => set({ alpha: v })}
              min={0}
              step={0.1}
            />
            <Checkbox
              label="Fit intercept"
              checked={m.fit_intercept ?? true}
              onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
            />
            <Select
              label="Selection"
              data={cdSelection}
              value={m.selection ?? 'cyclic'}
              onChange={(v) => set({ selection: v })}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? 1000}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-4}
              onChange={(v) => set({ tol: v })}
              step={1e-5}
              min={0}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
            <Checkbox
              label="Positive coefficients"
              checked={!!m.positive}
              onChange={(e) => set({ positive: e.currentTarget.checked })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Lasso (CV) */}
      {m.algo === 'lassocv' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="eps"
              value={m.eps ?? 1e-3}
              onChange={(v) => set({ eps: v })}
              min={0}
              step={1e-4}
            />
            <NumberInput
              label="n_alphas"
              value={m.n_alphas ?? 100}
              onChange={(v) => set({ n_alphas: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="CV folds"
              value={m.cv ?? 5}
              onChange={(v) => set({ cv: v })}
              allowDecimal={false}
              min={2}
            />
            <Checkbox
              label="Fit intercept"
              checked={m.fit_intercept ?? true}
              onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
            />
            <Select
              label="Selection"
              data={cdSelection}
              value={m.selection ?? 'cyclic'}
              onChange={(v) => set({ selection: v })}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? 1000}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-4}
              onChange={(v) => set({ tol: v })}
              step={1e-5}
              min={0}
            />
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
            <Checkbox
              label="Positive coefficients"
              checked={!!m.positive}
              onChange={(e) => set({ positive: e.currentTarget.checked })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Bayesian Ridge */}
      {m.algo === 'bayridge' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Iterations (n_iter)"
              value={m.n_iter ?? 300}
              onChange={(v) => set({ n_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-3}
              onChange={(v) => set({ tol: v })}
              step={1e-4}
              min={0}
            />
            <NumberInput
              label="alpha_1"
              value={m.alpha_1 ?? 1e-6}
              onChange={(v) => set({ alpha_1: v })}
              step={1e-6}
              min={0}
            />
            <NumberInput
              label="alpha_2"
              value={m.alpha_2 ?? 1e-6}
              onChange={(v) => set({ alpha_2: v })}
              step={1e-6}
              min={0}
            />
            <NumberInput
              label="lambda_1"
              value={m.lambda_1 ?? 1e-6}
              onChange={(v) => set({ lambda_1: v })}
              step={1e-6}
              min={0}
            />
            <NumberInput
              label="lambda_2"
              value={m.lambda_2 ?? 1e-6}
              onChange={(v) => set({ lambda_2: v })}
              step={1e-6}
              min={0}
            />
            <Checkbox
              label="Compute score"
              checked={!!m.compute_score}
              onChange={(e) => set({ compute_score: e.currentTarget.checked })}
            />
            <Checkbox
              label="Fit intercept"
              checked={m.fit_intercept ?? true}
              onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
            />
            <Checkbox
              label="Copy X"
              checked={m.copy_X ?? true}
              onChange={(e) => set({ copy_X: e.currentTarget.checked })}
            />
            <Checkbox
              label="Verbose"
              checked={!!m.verbose}
              onChange={(e) => set({ verbose: e.currentTarget.checked })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Support Vector Regression */}
      {m.algo === 'svr' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="C"
              value={m.C ?? 1.0}
              onChange={(v) => set({ C: v })}
              min={0}
              step={0.1}
            />
            <Select
              label="Kernel"
              data={svmKernel}
              value={m.kernel ?? 'rbf'}
              onChange={(v) => set({ kernel: v })}
            />
            <NumberInput
              label="Degree"
              value={m.degree ?? 3}
              onChange={(v) => set({ degree: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Gamma mode"
              data={[
                { value: 'scale', label: 'scale' },
                { value: 'auto', label: 'auto' },
                { value: 'numeric', label: 'numeric' },
              ]}
              value={gammaMode}
              onChange={(mode) => {
                if (mode === 'numeric') set({ gamma: gammaValue });
                else set({ gamma: mode });
              }}
            />
            {gammaMode === 'numeric' && (
              <NumberInput
                label="Gamma value"
                value={gammaValue}
                onChange={(v) => set({ gamma: v })}
                min={0}
                step={0.01}
              />
            )}
            <NumberInput
              label="Coef0"
              value={m.coef0 ?? 0.0}
              onChange={(v) => set({ coef0: v })}
              step={0.1}
            />
            <Checkbox
              label="Shrinking"
              checked={m.shrinking ?? true}
              onChange={(e) => set({ shrinking: e.currentTarget.checked })}
            />
            <NumberInput
              label="Epsilon"
              value={m.epsilon ?? 0.1}
              onChange={(v) => set({ epsilon: v })}
              min={0}
              step={0.01}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-3}
              onChange={(v) => set({ tol: v })}
              step={1e-4}
              min={0}
            />
            <NumberInput
              label="Cache size (MB)"
              value={m.cache_size ?? 200.0}
              onChange={(v) => set({ cache_size: v })}
              min={0}
              step={10}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? -1}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Linear SVR */}
      {m.algo === 'linsvr' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="C"
              value={m.C ?? 1.0}
              onChange={(v) => set({ C: v })}
              min={0}
              step={0.1}
            />
            <Select
              label="Loss"
              data={linsvrLoss}
              value={m.loss ?? 'epsilon_insensitive'}
              onChange={(v) => set({ loss: v })}
            />
            <NumberInput
              label="Epsilon"
              value={m.epsilon ?? 0.0}
              onChange={(v) => set({ epsilon: v })}
              min={0}
              step={0.01}
            />
            <Checkbox
              label="Fit intercept"
              checked={m.fit_intercept ?? true}
              onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
            />
            <NumberInput
              label="Intercept scaling"
              value={m.intercept_scaling ?? 1.0}
              onChange={(v) => set({ intercept_scaling: v })}
              min={0}
              step={0.1}
            />
            <Checkbox
              label="Dual"
              checked={m.dual ?? true}
              onChange={(e) => set({ dual: e.currentTarget.checked })}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-4}
              onChange={(v) => set({ tol: v })}
              step={1e-5}
              min={0}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? 1000}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* KNN Regressor */}
      {m.algo === 'knnreg' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Neighbours"
              value={m.n_neighbors ?? 5}
              onChange={(v) => set({ n_neighbors: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Weights"
              data={knnWeights}
              value={m.weights ?? 'uniform'}
              onChange={(v) => set({ weights: v })}
            />
            <Select
              label="Algorithm"
              data={knnAlgorithm}
              value={m.algorithm ?? 'auto'}
              onChange={(v) => set({ algorithm: v })}
            />
            <NumberInput
              label="Leaf size (leaf_size)"
              value={m.leaf_size ?? 30}
              onChange={(v) => set({ leaf_size: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="p"
              value={m.p ?? 2}
              onChange={(v) => set({ p: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Metric"
              data={knnMetric}
              value={m.metric ?? 'minkowski'}
              onChange={(v) => set({ metric: v })}
            />
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Decision Tree Regressor */}
      {m.algo === 'treereg' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <Select
              label="Criterion"
              data={regTreeCriterion}
              value={m.criterion ?? 'squared_error'}
              onChange={(v) => set({ criterion: v })}
            />
            <Select
              label="Splitter"
              data={treeSplitter}
              value={m.splitter ?? 'best'}
              onChange={(v) => set({ splitter: v })}
            />
            <NumberInput
              label="Max depth"
              value={m.max_depth ?? null}
              onChange={(v) => set({ max_depth: v })}
            />
            <NumberInput
              label="Min samples split"
              value={m.min_samples_split ?? 2}
              onChange={(v) => set({ min_samples_split: v })}
            />
            <NumberInput
              label="Min samples leaf"
              value={m.min_samples_leaf ?? 1}
              onChange={(v) => set({ min_samples_leaf: v })}
            />
            <NumberInput
              label="Min weight fraction"
              value={m.min_weight_fraction_leaf ?? 0.0}
              onChange={(v) => set({ min_weight_fraction_leaf: v })}
              step={0.01}
              min={0}
              max={1}
            />
            <Select
              label="Max features mode"
              data={[
                { value: 'sqrt', label: 'sqrt' },
                { value: 'log2', label: 'log2' },
                { value: 'int', label: 'int' },
                { value: 'float', label: 'float' },
                { value: 'none', label: 'none' },
              ]}
              value={tMF.mode}
              onChange={(mode) => set({ max_features: modeValToMaxFeat(mode, tMF.value) })}
            />
            {(tMF.mode === 'int' || tMF.mode === 'float') && (
              <NumberInput
                label="Max features value"
                value={tMF.value ?? null}
                onChange={(v) => set({ max_features: modeValToMaxFeat(tMF.mode, v) })}
                step={tMF.mode === 'int' ? 1 : 0.01}
                allowDecimal={tMF.mode === 'float'}
              />
            )}
            <NumberInput
              label="Max leaf nodes"
              value={m.max_leaf_nodes ?? null}
              onChange={(v) => set({ max_leaf_nodes: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Min impurity decrease"
              value={m.min_impurity_decrease ?? 0.0}
              onChange={(v) => set({ min_impurity_decrease: v })}
              step={0.0001}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="CCP alpha"
              value={m.ccp_alpha ?? 0.0}
              onChange={(v) => set({ ccp_alpha: v })}
              step={0.0001}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Random Forest Regressor */}
      {m.algo === 'rfreg' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Trees (n_estimators)"
              value={m.n_estimators ?? 100}
              onChange={(v) => set({ n_estimators: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Criterion"
              data={regTreeCriterion}
              value={m.criterion ?? 'squared_error'}
              onChange={(v) => set({ criterion: v })}
            />
            <NumberInput
              label="Max depth"
              value={m.max_depth ?? null}
              onChange={(v) => set({ max_depth: v })}
            />
            <NumberInput
              label="Min samples split"
              value={m.min_samples_split ?? 2}
              onChange={(v) => set({ min_samples_split: v })}
            />
            <NumberInput
              label="Min samples leaf"
              value={m.min_samples_leaf ?? 1}
              onChange={(v) => set({ min_samples_leaf: v })}
            />
            <NumberInput
              label="Min weight fraction"
              value={m.min_weight_fraction_leaf ?? 0.0}
              onChange={(v) => set({ min_weight_fraction_leaf: v })}
              step={0.01}
              min={0}
              max={1}
            />
            <Select
              label="Max features mode"
              data={[
                { value: 'sqrt', label: 'sqrt' },
                { value: 'log2', label: 'log2' },
                { value: 'int', label: 'int' },
                { value: 'float', label: 'float' },
                { value: 'none', label: 'none' },
              ]}
              value={fMF.mode}
              onChange={(mode) => set({ max_features: modeValToMaxFeat(mode, fMF.value) })}
            />
            {(fMF.mode === 'int' || fMF.mode === 'float') && (
              <NumberInput
                label="Max features value"
                value={fMF.value ?? null}
                onChange={(v) => set({ max_features: modeValToMaxFeat(fMF.mode, v) })}
                step={fMF.mode === 'int' ? 1 : 0.01}
                allowDecimal={fMF.mode === 'float'}
              />
            )}
            <NumberInput
              label="Max leaf nodes"
              value={m.max_leaf_nodes ?? null}
              onChange={(v) => set({ max_leaf_nodes: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Min impurity decrease"
              value={m.min_impurity_decrease ?? 0.0}
              onChange={(v) => set({ min_impurity_decrease: v })}
              step={0.0001}
            />
            <Checkbox
              label="Use bootstrap"
              checked={m.bootstrap ?? true}
              onChange={(e) => set({ bootstrap: e.currentTarget.checked })}
            />
            <Checkbox
              label="OOB score"
              checked={!!m.oob_score}
              onChange={(e) => set({ oob_score: e.currentTarget.checked })}
            />
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
            <Checkbox
              label="Warm start"
              checked={!!m.warm_start}
              onChange={(e) => set({ warm_start: e.currentTarget.checked })}
            />
            <NumberInput
              label="CCP alpha"
              value={m.ccp_alpha ?? 0.0}
              onChange={(v) => set({ ccp_alpha: v })}
              step={0.0001}
            />
            <NumberInput
              label="Max samples"
              value={m.max_samples ?? null}
              onChange={(v) => set({ max_samples: v })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* KMeans */}
      {m.algo === 'kmeans' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Clusters (n_clusters)"
              value={m.n_clusters ?? 8}
              onChange={(v) => set({ n_clusters: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Init"
              data={toSelectData(enumFromSubSchema(sub, 'init', ['k-means++', 'random']))}
              value={m.init ?? 'k-means++'}
              onChange={(v) => set({ init: v })}
            />
            <TextInput
              label="n_init (auto or int)"
              value={m.n_init == null ? 'auto' : String(m.n_init)}
              onChange={(e) => {
                const raw = e.currentTarget.value;
                const t = String(raw ?? '').trim();
                if (!t || t.toLowerCase() === 'auto') {
                  set({ n_init: 'auto' });
                  return;
                }
                const n = Number(t);
                set({ n_init: Number.isFinite(n) ? Math.max(1, Math.trunc(n)) : 'auto' });
              }}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? 300}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-4}
              onChange={(v) => set({ tol: v })}
              step={1e-5}
              min={0}
            />
            <NumberInput
              label="Verbose"
              value={m.verbose ?? 0}
              onChange={(v) => set({ verbose: v })}
              allowDecimal={false}
              min={0}
            />
            <Select
              label="Algorithm"
              data={toSelectData(enumFromSubSchema(sub, 'algorithm', ['lloyd', 'elkan', 'auto']))}
              value={m.algorithm ?? 'lloyd'}
              onChange={(v) => set({ algorithm: v })}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* DBSCAN */}
      {m.algo === 'dbscan' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Epsilon (eps)"
              value={m.eps ?? 0.5}
              onChange={(v) => set({ eps: v })}
              step={0.01}
              min={0}
            />
            <NumberInput
              label="Minimum samples (min_samples)"
              value={m.min_samples ?? 5}
              onChange={(v) => set({ min_samples: v })}
              allowDecimal={false}
              min={1}
            />
            <TextInput
              label="Distance metric (metric)"
              value={m.metric ?? 'euclidean'}
              onChange={(e) => set({ metric: e.currentTarget.value })}
            />
            <Select
              label="Search algorithm (algorithm)"
              data={toSelectData(enumFromSubSchema(sub, 'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']))}
              value={m.algorithm ?? 'auto'}
              onChange={(v) => set({ algorithm: v })}
            />
            <NumberInput
              label="Leaf size (leaf_size)"
              value={m.leaf_size ?? 30}
              onChange={(v) => set({ leaf_size: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Minkowski power (p)"
              value={m.p ?? null}
              onChange={(v) => set({ p: v })}
            />
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Spectral Clustering */}
      {m.algo === 'spectral' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Clusters (n_clusters)"
              value={m.n_clusters ?? 8}
              onChange={(v) => set({ n_clusters: v })}
              allowDecimal={false}
              min={2}
            />
            <Select
              label="Affinity"
              data={toSelectData(enumFromSubSchema(sub, 'affinity', ['rbf', 'nearest_neighbors']))}
              value={m.affinity ?? 'rbf'}
              onChange={(v) => set({ affinity: v })}
            />
            <Select
              label="Assign labels"
              data={toSelectData(enumFromSubSchema(sub, 'assign_labels', ['kmeans', 'discretize', 'cluster_qr']))}
              value={m.assign_labels ?? 'kmeans'}
              onChange={(v) => set({ assign_labels: v })}
            />
            <NumberInput
              label="Initializations (n_init)"
              value={m.n_init ?? 10}
              onChange={(v) => set({ n_init: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Gamma"
              value={m.gamma ?? 1.0}
              onChange={(v) => set({ gamma: v })}
              step={0.1}
              min={0}
            />
            <NumberInput
              label="Neighbours (n_neighbors)"
              value={m.n_neighbors ?? 10}
              onChange={(v) => set({ n_neighbors: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Agglomerative Clustering */}
      {m.algo === 'agglo' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Clusters (n_clusters)"
              value={m.n_clusters ?? 2}
              onChange={(v) => set({ n_clusters: v })}
              allowDecimal={false}
              min={2}
            />
            <Select
              label="Linkage"
              data={toSelectData(enumFromSubSchema(sub, 'linkage', ['ward', 'complete', 'average', 'single']))}
              value={m.linkage ?? 'ward'}
              onChange={(v) => set({ linkage: v })}
            />
            <TextInput
              label="Distance metric (metric)"
              value={m.metric ?? 'euclidean'}
              onChange={(e) => set({ metric: e.currentTarget.value })}
            />
            <NumberInput
              label="Distance threshold"
              value={m.distance_threshold ?? null}
              onChange={(v) => set({ distance_threshold: v })}
              min={0}
            />
            <Select
              label="Compute full tree"
              data={[
                { value: 'auto', label: 'auto' },
                { value: 'true', label: 'true' },
                { value: 'false', label: 'false' },
              ]}
              value={
                m.compute_full_tree === true
                  ? 'true'
                  : m.compute_full_tree === false
                  ? 'false'
                  : 'auto'
              }
              onChange={(v) => {
                if (v === 'true') set({ compute_full_tree: true });
                else if (v === 'false') set({ compute_full_tree: false });
                else set({ compute_full_tree: 'auto' });
              }}
            />
            <Checkbox
              label="Compute distances"
              checked={!!m.compute_distances}
              onChange={(e) => set({ compute_distances: e.currentTarget.checked })}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Gaussian Mixture */}
      {m.algo === 'gmm' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Components (n_components)"
              value={m.n_components ?? 1}
              onChange={(v) => set({ n_components: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Covariance type"
              data={toSelectData(enumFromSubSchema(sub, 'covariance_type', ['full', 'tied', 'diag', 'spherical']))}
              value={m.covariance_type ?? 'full'}
              onChange={(v) => set({ covariance_type: v })}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-3}
              onChange={(v) => set({ tol: v })}
              step={1e-4}
              min={0}
            />
            <NumberInput
              label="Regularization (reg_covar)"
              value={m.reg_covar ?? 1e-6}
              onChange={(v) => set({ reg_covar: v })}
              step={1e-6}
              min={0}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? 100}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Initializations (n_init)"
              value={m.n_init ?? 1}
              onChange={(v) => set({ n_init: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Init params"
              data={toSelectData(enumFromSubSchema(sub, 'init_params', ['kmeans', 'k-means++', 'random', 'random_from_data']))}
              value={m.init_params ?? 'kmeans'}
              onChange={(v) => set({ init_params: v })}
            />
            <Checkbox
              label="Warm start"
              checked={!!m.warm_start}
              onChange={(e) => set({ warm_start: e.currentTarget.checked })}
            />
            <NumberInput
              label="Verbose"
              value={m.verbose ?? 0}
              onChange={(v) => set({ verbose: v })}
              allowDecimal={false}
              min={0}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Bayesian Gaussian Mixture */}
      {m.algo === 'bgmm' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Components (n_components)"
              value={m.n_components ?? 1}
              onChange={(v) => set({ n_components: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Covariance type"
              data={toSelectData(enumFromSubSchema(sub, 'covariance_type', ['full', 'tied', 'diag', 'spherical']))}
              value={m.covariance_type ?? 'full'}
              onChange={(v) => set({ covariance_type: v })}
            />
            <NumberInput
              label="Tolerance (tol)"
              value={m.tol ?? 1e-3}
              onChange={(v) => set({ tol: v })}
              step={1e-4}
              min={0}
            />
            <NumberInput
              label="Regularization (reg_covar)"
              value={m.reg_covar ?? 1e-6}
              onChange={(v) => set({ reg_covar: v })}
              step={1e-6}
              min={0}
            />
            <NumberInput
              label="Max iterations"
              value={m.max_iter ?? 100}
              onChange={(v) => set({ max_iter: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="Initializations (n_init)"
              value={m.n_init ?? 1}
              onChange={(v) => set({ n_init: v })}
              allowDecimal={false}
              min={1}
            />
            <Select
              label="Init params"
              data={toSelectData(enumFromSubSchema(sub, 'init_params', ['kmeans', 'k-means++', 'random', 'random_from_data']))}
              value={m.init_params ?? 'kmeans'}
              onChange={(v) => set({ init_params: v })}
            />
            <Select
              label="Weight concentration prior type"
              data={toSelectData(enumFromSubSchema(sub, 'weight_concentration_prior_type', ['dirichlet_process', 'dirichlet_distribution']))}
              value={m.weight_concentration_prior_type ?? 'dirichlet_process'}
              onChange={(v) => set({ weight_concentration_prior_type: v })}
            />
            <Checkbox
              label="Warm start"
              checked={!!m.warm_start}
              onChange={(e) => set({ warm_start: e.currentTarget.checked })}
            />
            <NumberInput
              label="Verbose"
              value={m.verbose ?? 0}
              onChange={(v) => set({ verbose: v })}
              allowDecimal={false}
              min={0}
            />
            <NumberInput
              label="Random state"
              value={m.random_state ?? null}
              onChange={(v) => set({ random_state: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* MeanShift */}
      {m.algo === 'meanshift' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Bandwidth (optional)"
              value={m.bandwidth ?? null}
              onChange={(v) => set({ bandwidth: v })}
              min={0}
            />
            <Checkbox
              label="Bin seeding"
              checked={!!m.bin_seeding}
              onChange={(e) => set({ bin_seeding: e.currentTarget.checked })}
            />
            <NumberInput
              label="Min bin freq"
              value={m.min_bin_freq ?? 1}
              onChange={(v) => set({ min_bin_freq: v })}
              allowDecimal={false}
              min={1}
            />
            <Checkbox
              label="Cluster all"
              checked={m.cluster_all ?? true}
              onChange={(e) => set({ cluster_all: e.currentTarget.checked })}
            />
            <NumberInput
              label="Jobs (n_jobs)"
              value={m.n_jobs ?? null}
              onChange={(v) => set({ n_jobs: v })}
              allowDecimal={false}
            />
          </SimpleGrid>
        </Stack>
      )}

      {/* Birch */}
      {m.algo === 'birch' && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
            <NumberInput
              label="Threshold"
              value={m.threshold ?? 0.5}
              onChange={(v) => set({ threshold: v })}
              step={0.01}
              min={0}
            />
            <NumberInput
              label="Branching factor"
              value={m.branching_factor ?? 50}
              onChange={(v) => set({ branching_factor: v })}
              allowDecimal={false}
              min={1}
            />
            <NumberInput
              label="n_clusters (optional)"
              value={m.n_clusters ?? 3}
              onChange={(v) => set({ n_clusters: v })}
              allowDecimal={false}
              min={1}
            />
            <Checkbox
              label="Compute labels"
              checked={m.compute_labels ?? true}
              onChange={(e) => set({ compute_labels: e.currentTarget.checked })}
            />
            <Checkbox
              label="Copy"
              checked={m.copy ?? true}
              onChange={(e) => set({ copy: e.currentTarget.checked })}
            />
          </SimpleGrid>
        </Stack>
      )}


      <Divider my="xs" />
      <Text size="xs" c="dimmed">
        Algorithms are filtered by dataset task via{' '}
        <code>models.meta[algo].task</code> and your selected task in the Data
        panel.
      </Text>
    </>
  );

  // --- Render paths ---

  if (!showHelp) {
    // Original layout (no help text)
    return (
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between">
            <Text fw={500}>Model</Text>
          </Group>

          <Stack gap="md">{controlsBody}</Stack>
        </Stack>
      </Card>
    );
  }

  // Layout with A/B/C help structure
  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        {/* A title, centered and slightly larger */}
        <Text fw={700} size="lg" align="center">
          Model
        </Text>

        {/* A + B row: controls on the left, short intro on the right */}
        <Group align="flex-start" gap="xl" grow wrap="nowrap">
          {/* Block A: controls */}
          <Box style={{ flex: 1, minWidth: 0 }}>
            <Stack gap="md">{controlsBody}</Stack>
          </Box>

          {/* Block B: short intro + toggle */}
          <Box
            style={{
              flex: 1,
              minWidth: 220,
            }}
          >
            <Stack gap="xs">
              <ModelIntroText />
              <Button
                size="xs"
                variant="subtle"
                onClick={() => setShowDetails((prev) => !prev)}
              >
                {showDetails ? 'Show less' : 'Show more'}
              </Button>
            </Stack>
          </Box>
        </Group>

        {/* Block C: full-width detailed help text, toggled */}
        {showDetails && (
          <Box mt="md">
            <ModelHelpText
              selectedAlgo={m.algo}
              effectiveTask={effectiveTask}
              visibleAlgos={visibleAlgos}
            />
          </Box>
        )}
      </Stack>
    </Card>
  );
}
