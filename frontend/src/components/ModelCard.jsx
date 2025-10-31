// src/components/ModelCard.jsx
import React from 'react';
import {
  Stack, Select, NumberInput, Checkbox, Divider, Text, Group
} from '@mantine/core';

export default function ModelCard({ value, onChange, title = 'Model' }) {
  const { algo, logreg, svm, tree, forest } = value;

  const setAlgo = (next) => onChange({ ...value, algo: next || 'logreg' });
  const setLogReg = (k, v) => onChange({ ...value, logreg: { ...logreg, [k]: v } });
  const setSVM = (k, v) => onChange({ ...value, svm: { ...svm, [k]: v } });
  const setTree = (k, v) => onChange({ ...value, tree: { ...tree, [k]: v } });
  const setForest = (k, v) => onChange({ ...value, forest: { ...forest, [k]: v } });

  return (
    <Stack gap="sm">
      <Select
        label={title}
        data={[
          { value: 'logreg', label: 'Logistic Regression' },
          { value: 'svm', label: 'SVM (SVC)' },
          { value: 'tree', label: 'Decision Tree' },
          { value: 'forest', label: 'Random Forest' },
        ]}
        value={algo}
        onChange={setAlgo}
      />

      {/* Logistic Regression */}
      {algo === 'logreg' && (
        <Stack gap={6} ml="xs">
          <NumberInput label="C (inverse regularization)" value={logreg.C} onChange={(v) => setLogReg('C', v)} min={0} step={0.1} />
          <Select
            label="penalty"
            data={[
              { value: 'l2', label: 'l2' },
              { value: 'l1', label: 'l1' },
              { value: 'elasticnet', label: 'elasticnet' },
              { value: 'none', label: 'none' },
            ]}
            value={logreg.penalty}
            onChange={(v) => setLogReg('penalty', v)}
          />
          {logreg.penalty === 'elasticnet' && (
            <NumberInput label="l1_ratio (0–1)" value={logreg.l1_ratio ?? 0.5} onChange={(v) => setLogReg('l1_ratio', v)} min={0} max={1} step={0.05} />
          )}
          <Select
            label="solver"
            data={['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'].map(s => ({ value: s, label: s }))}
            value={logreg.solver}
            onChange={(v) => setLogReg('solver', v)}
          />
          <NumberInput label="max_iter" value={logreg.max_iter} onChange={(v) => setLogReg('max_iter', v)} step={50} min={1} />
          <Select
            label="class_weight"
            data={[{ value: 'none', label: 'none' }, { value: 'balanced', label: 'balanced' }]}
            value={logreg.class_weight ?? 'none'}
            onChange={(v) => setLogReg('class_weight', v === 'none' ? null : 'balanced')}
          />
          <Text size="xs" c="dimmed">l1/elasticnet need compatible solvers (saga; l1 also liblinear).</Text>
        </Stack>
      )}

      {/* SVM */}
      {algo === 'svm' && (
        <Stack gap={6} ml="xs">
          <NumberInput label="C" value={svm.C} onChange={(v) => setSVM('C', v)} min={0} step={0.1} />
          <Select label="kernel" data={['linear', 'poly', 'rbf', 'sigmoid'].map(k => ({ value: k, label: k }))} value={svm.kernel} onChange={(v) => setSVM('kernel', v)} />
          {svm.kernel === 'poly' && <NumberInput label="degree" value={svm.degree} onChange={(v) => setSVM('degree', v)} min={1} />}
          <Select
            label="gamma"
            data={[{ value: 'scale', label: 'scale' }, { value: 'auto', label: 'auto' }, { value: 'numeric', label: 'numeric' }]}
            value={svm.gammaMode}
            onChange={(v) => setSVM('gammaMode', v || 'scale')}
          />
          {svm.gammaMode === 'numeric' && <NumberInput label="gamma (numeric)" value={svm.gammaValue} onChange={(v) => setSVM('gammaValue', v)} min={0} step={0.01} />}
          {(svm.kernel === 'poly' || svm.kernel === 'sigmoid') && (
            <NumberInput label="coef0" value={svm.coef0} onChange={(v) => setSVM('coef0', v)} step={0.1} />
          )}
          <Checkbox label="shrinking" checked={svm.shrinking} onChange={(e) => setSVM('shrinking', e.currentTarget.checked)} />
          <Checkbox label="probability (slower)" checked={svm.probability} onChange={(e) => setSVM('probability', e.currentTarget.checked)} />
          <NumberInput label="tol" value={svm.tol} onChange={(v) => setSVM('tol', v)} min={0} step={0.0001} />
          <NumberInput label="cache_size (MB)" value={svm.cache_size} onChange={(v) => setSVM('cache_size', v)} min={0} step={50} />
          <Select
            label="class_weight"
            data={[{ value: 'none', label: 'none' }, { value: 'balanced', label: 'balanced' }]}
            value={svm.class_weight ?? 'none'}
            onChange={(v) => setSVM('class_weight', v === 'none' ? null : 'balanced')}
          />
          <NumberInput label="max_iter (-1 = unlimited)" value={svm.max_iter} onChange={(v) => setSVM('max_iter', v)} step={100} />
          <Select
            label="decision_function_shape"
            data={[{ value: 'ovr', label: 'ovr' }, { value: 'ovo', label: 'ovo' }]}
            value={svm.decision_function_shape}
            onChange={(v) => setSVM('decision_function_shape', v)}
          />
          <Checkbox label="break_ties" checked={svm.break_ties} onChange={(e) => setSVM('break_ties', e.currentTarget.checked)} />
          <Text size="xs" c="dimmed">Tip: SVMs usually benefit from standardization.</Text>
        </Stack>
      )}

      {/* Decision Tree */}
      {algo === 'tree' && (
        <Stack gap={6} ml="xs">
          <Select
            label="criterion"
            data={['gini', 'entropy', 'log_loss'].map(c => ({ value: c, label: c }))}
            value={tree.criterion}
            onChange={(v) => setTree('criterion', v)}
          />
          <Select
            label="splitter"
            data={['best', 'random'].map(s => ({ value: s, label: s }))}
            value={tree.splitter}
            onChange={(v) => setTree('splitter', v)}
          />
          <NumberInput label="max_depth" value={tree.max_depth} onChange={(v) => setTree('max_depth', v)} min={1} />
          <NumberInput label="min_samples_split (int/float)" value={tree.min_samples_split} onChange={(v) => setTree('min_samples_split', v)} min={0} step={0.1} />
          <NumberInput label="min_samples_leaf (int/float)" value={tree.min_samples_leaf} onChange={(v) => setTree('min_samples_leaf', v)} min={0} step={0.1} />
          <NumberInput label="min_weight_fraction_leaf" value={tree.min_weight_fraction_leaf} onChange={(v) => setTree('min_weight_fraction_leaf', v)} min={0} max={0.5} step={0.01} />
          <Select
            label="max_features"
            data={[
              { value: 'none', label: 'all' },
              { value: 'sqrt', label: 'sqrt' },
              { value: 'log2', label: 'log2' },
              { value: 'int', label: 'int' },
              { value: 'float', label: 'float' },
            ]}
            value={tree.max_features_mode}
            onChange={(v) => setTree('max_features_mode', v || 'none')}
          />
          {tree.max_features_mode === 'int' && (
            <NumberInput label="max_features (int)" value={tree.max_features_value} onChange={(v) => setTree('max_features_value', v)} min={1} />
          )}
          {tree.max_features_mode === 'float' && (
            <NumberInput label="max_features (0–1 float)" value={tree.max_features_value} onChange={(v) => setTree('max_features_value', v)} min={0} max={1} step={0.05} />
          )}
          <NumberInput label="max_leaf_nodes" value={tree.max_leaf_nodes} onChange={(v) => setTree('max_leaf_nodes', v)} min={2} />
          <NumberInput label="min_impurity_decrease" value={tree.min_impurity_decrease} onChange={(v) => setTree('min_impurity_decrease', v)} min={0} step={0.0001} />
          <Select
            label="class_weight"
            data={[{ value: 'none', label: 'none' }, { value: 'balanced', label: 'balanced' }]}
            value={tree.class_weight ?? 'none'}
            onChange={(v) => setTree('class_weight', v === 'none' ? null : 'balanced')}
          />
          <NumberInput label="ccp_alpha" value={tree.ccp_alpha} onChange={(v) => setTree('ccp_alpha', v)} min={0} step={0.0001} />
        </Stack>
      )}

      {/* Random Forest */}
      {algo === 'forest' && (
        <Stack gap={6} ml="xs">
          <NumberInput label="n_estimators" value={forest.n_estimators} onChange={(v) => setForest('n_estimators', v)} min={1} step={10} />
          <Select label="criterion" data={['gini', 'entropy', 'log_loss'].map(c => ({ value: c, label: c }))} value={forest.criterion} onChange={(v) => setForest('criterion', v)} />
          <NumberInput label="max_depth" value={forest.max_depth} onChange={(v) => setForest('max_depth', v)} min={1} />
          <NumberInput label="min_samples_split (int/float)" value={forest.min_samples_split} onChange={(v) => setForest('min_samples_split', v)} min={0} step={0.1} />
          <NumberInput label="min_samples_leaf (int/float)" value={forest.min_samples_leaf} onChange={(v) => setForest('min_samples_leaf', v)} min={0} step={0.1} />
          <NumberInput label="min_weight_fraction_leaf" value={forest.min_weight_fraction_leaf} onChange={(v) => setForest('min_weight_fraction_leaf', v)} min={0} max={0.5} step={0.01} />
          <Select
            label="max_features"
            data={[
              { value: 'sqrt', label: 'sqrt' },
              { value: 'log2', label: 'log2' },
              { value: 'none', label: 'all' },
              { value: 'int', label: 'int' },
              { value: 'float', label: 'float' },
            ]}
            value={forest.max_features_mode}
            onChange={(v) => setForest('max_features_mode', v || 'sqrt')}
          />
          {forest.max_features_mode === 'int' && (
            <NumberInput label="max_features (int)" value={forest.max_features_value} onChange={(v) => setForest('max_features_value', v)} min={1} />
          )}
          {forest.max_features_mode === 'float' && (
            <NumberInput label="max_features (0–1 float)" value={forest.max_features_value} onChange={(v) => setForest('max_features_value', v)} min={0} max={1} step={0.05} />
          )}
          <NumberInput label="max_leaf_nodes" value={forest.max_leaf_nodes} onChange={(v) => setForest('max_leaf_nodes', v)} min={2} />
          <NumberInput label="min_impurity_decrease" value={forest.min_impurity_decrease} onChange={(v) => setForest('min_impurity_decrease', v)} min={0} step={0.0001} />
          <Checkbox label="bootstrap" checked={forest.bootstrap} onChange={(e) => setForest('bootstrap', e.currentTarget.checked)} />
          <Checkbox label="oob_score" checked={forest.oob_score} onChange={(e) => setForest('oob_score', e.currentTarget.checked)} />
          <NumberInput label="n_jobs" value={forest.n_jobs} onChange={(v) => setForest('n_jobs', v)} step={1} />
          <Select
            label="class_weight"
            data={[
              { value: 'none', label: 'none' },
              { value: 'balanced', label: 'balanced' },
              { value: 'balanced_subsample', label: 'balanced_subsample' },
            ]}
            value={forest.class_weight ?? 'none'}
            onChange={(v) => setForest('class_weight', v === 'none' ? null : v)}
          />
          <NumberInput label="ccp_alpha" value={forest.ccp_alpha} onChange={(v) => setForest('ccp_alpha', v)} min={0} step={0.0001} />
          <Checkbox label="warm_start" checked={forest.warm_start} onChange={(e) => setForest('warm_start', e.currentTarget.checked)} />
        </Stack>
      )}

      <Divider my={4} />
    </Stack>
  );
}
