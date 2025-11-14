import {
  Card, Stack, Select, NumberInput, Checkbox, Divider, Text, Group, SimpleGrid, Tooltip
} from '@mantine/core';

export default function ModelSelectionCard({ value, onChange, title = 'Model' }) {
  const { algo, logreg, svm, tree, forest, knn } = value;

  const setAlgo = (next) => onChange({ ...value, algo: next || 'logreg' });
  const setLogReg = (k, v) => onChange({ ...value, logreg: { ...logreg, [k]: v } });
  const setSVM = (k, v) => onChange({ ...value, svm: { ...svm, [k]: v } });
  const setTree = (k, v) => onChange({ ...value, tree: { ...tree, [k]: v } });
  const setForest = (k, v) => onChange({ ...value, forest: { ...forest, [k]: v } });
  const setKNN = (k, v) => onChange({ ...value, knn: { ...knn, [k]: v } });

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        <Group justify="space-between" wrap="nowrap">
          <Text fw={500}>{title}</Text>
        </Group>

        <Select
          label="Algorithm"
          data={[
            { value: 'logreg', label: 'Logistic Regression' },
            { value: 'svm', label: 'SVM (SVC)' },
            { value: 'tree', label: 'Decision Tree' },
            { value: 'forest', label: 'Random Forest' },
            { value: 'knn', label: 'K-Nearest Neighbors' },
          ]}
          value={algo}
          onChange={setAlgo}
        />

        {/* Logistic Regression */}
        {algo === 'logreg' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <NumberInput
                label="C (inverse regularization)"
                value={logreg?.C ?? 1.0}
                onChange={(v) => setLogReg('C', v)}
                min={0}
                step={0.1}
              />
              <Select
                label="penalty"
                data={[
                  { value: 'l2', label: 'l2' },
                  { value: 'l1', label: 'l1' },
                  { value: 'elasticnet', label: 'elasticnet' },
                  { value: 'none', label: 'none' },
                ]}
                value={logreg?.penalty ?? 'l2'}
                onChange={(v) => setLogReg('penalty', v)}
              />
              {logreg?.penalty === 'elasticnet' && (
                <NumberInput
                  label="l1_ratio (0–1)"
                  value={logreg?.l1_ratio ?? 0.5}
                  onChange={(v) => setLogReg('l1_ratio', v)}
                  min={0}
                  max={1}
                  step={0.05}
                />
              )}
              <Select
                label="solver"
                data={['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'].map((s) => ({ value: s, label: s }))}
                value={logreg?.solver ?? 'lbfgs'}
                onChange={(v) => setLogReg('solver', v)}
              />
              <NumberInput
                label="max_iter"
                value={logreg?.max_iter ?? 1000}
                onChange={(v) => setLogReg('max_iter', v)}
                step={50}
                min={1}
              />
              <Select
                label="class_weight"
                data={[
                  { value: 'none', label: 'none' },
                  { value: 'balanced', label: 'balanced' },
                ]}
                value={logreg?.class_weight ?? 'none'}
                onChange={(v) => setLogReg('class_weight', v === 'none' ? null : v)}
              />
            </SimpleGrid>
          </Stack>
        )}

        {/* SVM */}
        {algo === 'svm' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <Select
                label="kernel"
                data={['linear', 'poly', 'rbf', 'sigmoid'].map((k) => ({ value: k, label: k }))}
                value={svm?.kernel ?? 'rbf'}
                onChange={(v) => setSVM('kernel', v)}
              />
              <NumberInput
                label="C"
                value={svm?.C ?? 1.0}
                onChange={(v) => setSVM('C', v)}
                min={0}
                step={0.1}
              />
              <NumberInput
                label="degree (poly)"
                value={svm?.degree ?? 3}
                onChange={(v) => setSVM('degree', v)}
                allowDecimal={false}
                min={1}
              />
              <Select
                label="gamma"
                data={[
                  { value: 'scale', label: 'scale' },
                  { value: 'auto', label: 'auto' },
                  { value: 'numeric', label: 'numeric' },
                ]}
                value={svm?.gammaMode ?? 'scale'}
                onChange={(v) => setSVM('gammaMode', v)}
              />
              {svm?.gammaMode === 'numeric' && (
                <NumberInput
                  label="gamma value"
                  value={svm?.gammaValue ?? 0.1}
                  onChange={(v) => setSVM('gammaValue', v)}
                  min={0}
                  step={0.001}
                />
              )}
              <NumberInput
                label="coef0 (poly/sigmoid)"
                value={svm?.coef0 ?? 0.0}
                onChange={(v) => setSVM('coef0', v)}
                step={0.1}
              />
              <Checkbox
                label="shrinking"
                checked={!!svm?.shrinking}
                onChange={(e) => setSVM('shrinking', e.currentTarget.checked)}
              />
              <Checkbox
                label="probability"
                checked={!!svm?.probability}
                onChange={(e) => setSVM('probability', e.currentTarget.checked)}
              />
              <NumberInput
                label="tol"
                value={svm?.tol ?? 0.001}
                onChange={(v) => setSVM('tol', v)}
                min={0}
                step={0.0001}
              />
              <NumberInput
                label="cache_size (MB)"
                value={svm?.cache_size ?? 200.0}
                onChange={(v) => setSVM('cache_size', v)}
                min={1}
                step={10}
              />
              <Select
                label="class_weight"
                data={[
                  { value: 'none', label: 'none' },
                  { value: 'balanced', label: 'balanced' },
                ]}
                value={svm?.class_weight ?? 'none'}
                onChange={(v) => setSVM('class_weight', v === 'none' ? null : v)}
              />
              <NumberInput
                label="max_iter"
                value={svm?.max_iter ?? -1}
                onChange={(v) => setSVM('max_iter', v)}
                allowDecimal={false}
              />
              <Select
                label="decision_function_shape"
                data={['ovr', 'ovo'].map((k) => ({ value: k, label: k }))}
                value={svm?.decision_function_shape ?? 'ovr'}
                onChange={(v) => setSVM('decision_function_shape', v)}
              />
              <Checkbox
                label="break_ties (ovr)"
                checked={!!svm?.break_ties}
                onChange={(e) => setSVM('break_ties', e.currentTarget.checked)}
              />
            </SimpleGrid>

            <Tooltip label="For RBF/poly/sigmoid: use scale/auto, or select numeric and enter a value.">
              <Text size="xs" c="dimmed">Gamma hints: scale = 1 / (n_features · Var(X))</Text>
            </Tooltip>
          </Stack>
        )}

        {/* Decision Tree */}
        {algo === 'tree' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <Select
                label="criterion"
                data={['gini', 'entropy', 'log_loss'].map((c) => ({ value: c, label: c }))}
                value={tree?.criterion ?? 'gini'}
                onChange={(v) => setTree('criterion', v)}
              />
              <Select
                label="splitter"
                data={['best', 'random'].map((s) => ({ value: s, label: s }))}
                value={tree?.splitter ?? 'best'}
                onChange={(v) => setTree('splitter', v)}
              />
              <NumberInput label="max_depth" value={tree?.max_depth ?? null} onChange={(v) => setTree('max_depth', v)} />
              <NumberInput label="min_samples_split" value={tree?.min_samples_split ?? 2} onChange={(v) => setTree('min_samples_split', v)} />
              <NumberInput label="min_samples_leaf" value={tree?.min_samples_leaf ?? 1} onChange={(v) => setTree('min_samples_leaf', v)} />
              <NumberInput label="min_weight_fraction_leaf" value={tree?.min_weight_fraction_leaf ?? 0.0} onChange={(v) => setTree('min_weight_fraction_leaf', v)} step={0.01} min={0} max={1} />
              <Select
                label="max_features (mode)"
                data={[
                  { value: 'auto', label: 'auto' },
                  { value: 'sqrt', label: 'sqrt' },
                  { value: 'log2', label: 'log2' },
                  { value: 'int', label: 'int' },
                  { value: 'float', label: 'float' },
                  { value: 'none', label: 'none' },
                ]}
                value={tree?.max_featuresMode ?? 'auto'}
                onChange={(v) => setTree('max_featuresMode', v)}
              />
              {(tree?.max_featuresMode === 'int' || tree?.max_featuresMode === 'float') && (
                <NumberInput
                  label="max_features (value)"
                  value={tree?.max_featuresValue ?? null}
                  onChange={(v) => setTree('max_featuresValue', v)}
                  step={tree?.max_featuresMode === 'int' ? 1 : 0.01}
                  allowDecimal={tree?.max_featuresMode === 'float'}
                />
              )}
              <NumberInput label="random_state" value={tree?.random_state ?? null} onChange={(v) => setTree('random_state', v)} allowDecimal={false} />
              <NumberInput label="max_leaf_nodes" value={tree?.max_leaf_nodes ?? null} onChange={(v) => setTree('max_leaf_nodes', v)} allowDecimal={false} />
              <NumberInput label="min_impurity_decrease" value={tree?.min_impurity_decrease ?? 0.0} onChange={(v) => setTree('min_impurity_decrease', v)} step={0.0001} />
              <Select
                label="class_weight"
                data={[
                  { value: 'none', label: 'none' },
                  { value: 'balanced', label: 'balanced' },
                ]}
                value={tree?.class_weight ?? 'none'}
                onChange={(v) => setTree('class_weight', v === 'none' ? null : v)}
              />
              <NumberInput label="ccp_alpha" value={tree?.ccp_alpha ?? 0.0} onChange={(v) => setTree('ccp_alpha', v)} step={0.0001} />
            </SimpleGrid>
          </Stack>
        )}

        {/* Random Forest */}
        {algo === 'forest' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <NumberInput label="n_estimators" value={forest?.n_estimators ?? 100} onChange={(v) => setForest('n_estimators', v)} allowDecimal={false} min={1} />
              <Select
                label="criterion"
                data={['gini', 'entropy', 'log_loss'].map((c) => ({ value: c, label: c }))}
                value={forest?.criterion ?? 'gini'}
                onChange={(v) => setForest('criterion', v)}
              />
              <NumberInput label="max_depth" value={forest?.max_depth ?? null} onChange={(v) => setForest('max_depth', v)} />
              <NumberInput label="min_samples_split" value={forest?.min_samples_split ?? 2} onChange={(v) => setForest('min_samples_split', v)} />
              <NumberInput label="min_samples_leaf" value={forest?.min_samples_leaf ?? 1} onChange={(v) => setForest('min_samples_leaf', v)} />
              <NumberInput label="min_weight_fraction_leaf" value={forest?.min_weight_fraction_leaf ?? 0.0} onChange={(v) => setForest('min_weight_fraction_leaf', v)} step={0.01} min={0} max={1} />
              <Select
                label="max_features (mode)"
                data={[
                  { value: 'sqrt', label: 'sqrt' },
                  { value: 'log2', label: 'log2' },
                  { value: 'int', label: 'int' },
                  { value: 'float', label: 'float' },
                  { value: 'none', label: 'none' },
                ]}
                value={forest?.max_featuresMode ?? 'sqrt'}
                onChange={(v) => setForest('max_featuresMode', v)}
              />
              {(forest?.max_featuresMode === 'int' || forest?.max_featuresMode === 'float') && (
                <NumberInput
                  label="max_features (value)"
                  value={forest?.max_featuresValue ?? null}
                  onChange={(v) => setForest('max_featuresValue', v)}
                  step={forest?.max_featuresMode === 'int' ? 1 : 0.01}
                  allowDecimal={forest?.max_featuresMode === 'float'}
                />
              )}
              <NumberInput label="max_leaf_nodes" value={forest?.max_leaf_nodes ?? null} onChange={(v) => setForest('max_leaf_nodes', v)} allowDecimal={false} />
              <NumberInput label="min_impurity_decrease" value={forest?.min_impurity_decrease ?? 0.0} onChange={(v) => setForest('min_impurity_decrease', v)} step={0.0001} />
              <Select
                label="class_weight"
                data={[
                  { value: 'none', label: 'none' },
                  { value: 'balanced', label: 'balanced' },
                  { value: 'balanced_subsample', label: 'balanced_subsample' },
                ]}
                value={forest?.class_weight ?? 'none'}
                onChange={(v) => setForest('class_weight', v === 'none' ? null : v)}
              />
              <Checkbox label="bootstrap" checked={!!forest?.bootstrap} onChange={(e) => setForest('bootstrap', e.currentTarget.checked)} />
              <Checkbox label="oob_score" checked={!!forest?.oob_score} onChange={(e) => setForest('oob_score', e.currentTarget.checked)} />
              <NumberInput label="n_jobs" value={forest?.n_jobs ?? null} onChange={(v) => setForest('n_jobs', v)} allowDecimal={false} />
              <NumberInput label="ccp_alpha" value={forest?.ccp_alpha ?? 0.0} onChange={(v) => setForest('ccp_alpha', v)} step={0.0001} />
              <Checkbox label="warm_start" checked={!!forest?.warm_start} onChange={(e) => setForest('warm_start', e.currentTarget.checked)} />
            </SimpleGrid>
          </Stack>
        )}

        {/* KNN */}
        {algo === 'knn' && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <NumberInput
                label="n_neighbors"
                value={knn?.n_neighbors ?? 5}
                onChange={(v) => setKNN('n_neighbors', v)}
                allowDecimal={false}
                min={1}
              />
              <Select
                label="weights"
                data={['uniform', 'distance'].map((w) => ({ value: w, label: w }))}
                value={knn?.weights ?? 'uniform'}
                onChange={(v) => setKNN('weights', v)}
              />
              <Select
                label="algorithm"
                data={['auto', 'ball_tree', 'kd_tree', 'brute'].map((a) => ({ value: a, label: a }))}
                value={knn?.algorithm ?? 'auto'}
                onChange={(v) => setKNN('algorithm', v)}
              />
              <NumberInput
                label="leaf_size"
                value={knn?.leaf_size ?? 30}
                onChange={(v) => setKNN('leaf_size', v)}
                allowDecimal={false}
                min={1}
              />
              <NumberInput
                label="p"
                value={knn?.p ?? 2}
                onChange={(v) => setKNN('p', v)}
                allowDecimal={false}
                min={1}
              />
              <Select
                label="metric"
                data={['minkowski', 'euclidean', 'manhattan', 'chebyshev'].map((m) => ({ value: m, label: m }))}
                value={knn?.metric ?? 'minkowski'}
                onChange={(v) => setKNN('metric', v)}
              />
            </SimpleGrid>
          </Stack>
        )}

        <Divider my="xs" />
        <Text size="xs" c="dimmed">
          Tip: leave fields empty to keep defaults.
        </Text>
      </Stack>
    </Card>
  );
}
