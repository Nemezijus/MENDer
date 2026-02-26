import {
  Stack,
  Select,
  NumberInput,
  Checkbox,
  TextInput,
  Tooltip,
  Box,
} from '@mantine/core';
import { useSchemaDefaults } from '../../schema/SchemaDefaultsContext.jsx';
import { useFeatureStore } from '../../state/useFeatureStore.js';
import FeatureHelpText, {
  FeatureIntroText,
} from '../../content/help/FeatureHelpText.jsx';
import { getVariantSchema, enumFromSubSchema, toSelectData, listDiscriminatorValues } from '../../utils/schema/jsonSchema.js';
import ConfigCardShell from './common/ConfigCardShell.jsx';

/** -------- helpers to read enums from the Features schema -------- **/

/** ---------------- component ---------------- **/

export default function FeatureCard({ title = 'Features' }) {
  const {
    method,
    setMethod,
    // PCA
    pca_n,
    setPcaN,
    pca_var,
    setPcaVar,
    pca_whiten,
    setPcaWhiten,
    // LDA
    lda_n,
    setLdaN,
    lda_solver,
    setLdaSolver,
    lda_shrinkage,
    setLdaShrinkage,
    lda_tol,
    setLdaTol,
    // SFS
    sfs_k,
    setSfsK,
    sfs_direction,
    setSfsDirection,
    sfs_cv,
    setSfsCv,
    sfs_n_jobs,
    setSfsNJobs,
  } = useFeatureStore();

  const { features, enums } = useSchemaDefaults();

  // Store is overrides-only. Display schema defaults when unset.
  const baseDefaults = features?.defaults ?? {};
  const defaultMethod = baseDefaults?.method;
  const effectiveMethod = method ?? defaultMethod ?? null;

  const sub = getVariantSchema(features?.schema, 'method', effectiveMethod);

  const propDefault = (key) => {
    const d = sub?.properties?.[key]?.default;
    if (d !== undefined) return d;
    return baseDefaults?.[key];
  };

  const numOrUndef = (v) => (v === null || v === '' || v === undefined ? undefined : v);
  const toIntOrUndef = (v) => {
    if (v === null || v === '' || v === undefined) return undefined;
    const n = Number(v);
    if (!Number.isFinite(n)) return undefined;
    return Math.trunc(n);
  };
  const toFloatOrUndef = (v) => {
    if (v === null || v === '' || v === undefined) return undefined;
    const n = Number(v);
    if (!Number.isFinite(n)) return undefined;
    return n;
  };

  const clearIfDefault = (next, def) => {
    // When schema default is unknown, we cannot safely decide what is "default".
    // In that case, only clear on empty.
    if (next === undefined) return true;
    if (def === undefined) return false;
    return String(next) === String(def);
  };

  const methodEnum =
    (Array.isArray(enums?.FeatureName) && enums.FeatureName.length
      ? enums.FeatureName
      : null) ??
    listDiscriminatorValues(features?.schema, 'method') ??
    [];
  const methods = toSelectData(methodEnum);
  const methodsUnavailable = methods.length === 0;

  const ldaSolverData = toSelectData(enumFromSubSchema(sub, 'lda_solver') ?? []);
  const ldaSolverUnavailable = ldaSolverData.length === 0;
  const sfsDirectionData = toSelectData(enumFromSubSchema(sub, 'sfs_direction') ?? []);
  const sfsDirectionUnavailable = sfsDirectionData.length === 0;

  // Effective (displayed) values
  const effectivePcaN = numOrUndef(pca_n ?? propDefault('pca_n'));
  const effectivePcaVar = numOrUndef(pca_var ?? propDefault('pca_var'));
  const effectivePcaWhiten = Boolean(
    pca_whiten ?? propDefault('pca_whiten') ?? false,
  );

  const effectiveLdaN = numOrUndef(lda_n ?? propDefault('lda_n'));
  const effectiveLdaSolver = String(
    lda_solver ?? propDefault('lda_solver') ?? '',
  );
  const effectiveLdaShrinkage = numOrUndef(
    lda_shrinkage ?? propDefault('lda_shrinkage'),
  );
  const effectiveLdaTol = numOrUndef(lda_tol ?? propDefault('lda_tol'));

  const effectiveSfsK =
    sfs_k ?? propDefault('sfs_k') ?? '';
  const effectiveSfsDirection = String(
    sfs_direction ?? propDefault('sfs_direction') ?? '',
  );
  const effectiveSfsCv = numOrUndef(sfs_cv ?? propDefault('sfs_cv'));
  const effectiveSfsNJobs = numOrUndef(
    sfs_n_jobs ?? propDefault('sfs_n_jobs'),
  );

  // Handlers: keep the store as overrides-only (clear if set to schema default).
  const handleMethodChange = (v) => {
    const next = v || undefined;
    if (clearIfDefault(next, defaultMethod)) setMethod(undefined);
    else setMethod(next);
  };

  return (
    <ConfigCardShell
      title={title}
      left={(
        <Stack gap="sm">
          <Select
            label="Feature method"
            data={methods}
            value={effectiveMethod}
            onChange={handleMethodChange}
            disabled={methodsUnavailable}
            placeholder={methodsUnavailable ? 'Schema enums unavailable' : undefined}
            description={methodsUnavailable ? 'Schema did not provide feature method options.' : undefined}
            styles={{
              input: {
                borderWidth: 2,
                borderColor: '#5c94ccff',
              },
            }}
          />
        </Stack>
      )}
      right={<FeatureIntroText />}
      help={<FeatureHelpText selectedMethod={effectiveMethod} />}
    >
      {/* Method-specific controls (grow downward, help text stays relatively stable) */}
      <Box mt="md">
        {/* PCA controls */}
        {effectiveMethod === 'pca' && (
          <Stack gap="sm">
            <Tooltip label="Number of components (optional). Leave empty for variance-based.">
              <NumberInput
                label="pca_n"
                value={effectivePcaN}
                onChange={(v) => {
                  const next = toIntOrUndef(v);
                  if (clearIfDefault(next, propDefault('pca_n'))) setPcaN(undefined);
                  else setPcaN(next);
                }}
                placeholder="empty = auto by variance"
                allowDecimal={false}
                min={1}
              />
            </Tooltip>
            <Tooltip label="Retained variance if pca_n is empty (e.g., 0.95)">
              <NumberInput
                label="pca_var"
                value={effectivePcaVar}
                onChange={(v) => {
                  const next = toFloatOrUndef(v);
                  if (clearIfDefault(next, propDefault('pca_var'))) setPcaVar(undefined);
                  else setPcaVar(next);
                }}
                min={0.0}
                max={1.0}
                step={0.01}
              />
            </Tooltip>
            <Checkbox
              label="pca_whiten"
              checked={effectivePcaWhiten}
              onChange={(e) => {
                const next = Boolean(e.currentTarget.checked);
                if (clearIfDefault(next, propDefault('pca_whiten'))) setPcaWhiten(undefined);
                else setPcaWhiten(next);
              }}
            />
          </Stack>
        )}

        {/* LDA controls */}
        {effectiveMethod === 'lda' && (
          <Stack gap="sm">
            <Tooltip label="Target number of components (<= n_classes - 1). Leave empty to infer.">
              <NumberInput
                label="lda_n"
                value={effectiveLdaN}
                onChange={(v) => {
                  const next = toIntOrUndef(v);
                  if (clearIfDefault(next, propDefault('lda_n'))) setLdaN(undefined);
                  else setLdaN(next);
                }}
                allowDecimal={false}
                min={1}
              />
            </Tooltip>
            <Select
              label="lda_solver"
              data={ldaSolverData}
              value={effectiveLdaSolver || null}
              disabled={ldaSolverUnavailable}
              placeholder={ldaSolverUnavailable ? 'Schema enums unavailable' : undefined}
              description={ldaSolverUnavailable ? 'Schema did not provide lda_solver options.' : undefined}
              onChange={(v) => {
                const next = v || undefined;
                if (clearIfDefault(next, propDefault('lda_solver'))) setLdaSolver(undefined);
                else setLdaSolver(next);
              }}
            />
            <Tooltip label="Only used with 'lsqr' or 'eigen'. Leave empty for None.">
              <NumberInput
                label="lda_shrinkage"
                value={effectiveLdaShrinkage}
                onChange={(v) => {
                  const next = toFloatOrUndef(v);
                  if (clearIfDefault(next, propDefault('lda_shrinkage'))) setLdaShrinkage(undefined);
                  else setLdaShrinkage(next);
                }}
                step={0.1}
                min={0}
              />
            </Tooltip>
            <NumberInput
              label="lda_tol"
              value={effectiveLdaTol}
              onChange={(v) => {
                const next = toFloatOrUndef(v);
                if (clearIfDefault(next, propDefault('lda_tol'))) setLdaTol(undefined);
                else setLdaTol(next);
              }}
              step={1e-4}
              precision={6}
              min={0}
            />
          </Stack>
        )}

        {/* SFS controls */}
        {effectiveMethod === 'sfs' && (
          <Stack gap="sm">
            <Tooltip label="Number of selected features; 'auto' or integer">
              <TextInput
                label="sfs_k"
                value={effectiveSfsK == null ? '' : String(effectiveSfsK)}
                onChange={(e) => {
                  const v = e.currentTarget.value.trim();
                  const def = propDefault('sfs_k');
                  if (v === '') {
                    setSfsK(undefined);
                    return;
                  }
                  if (v.toLowerCase() === 'auto') {
                    if (clearIfDefault('auto', def)) setSfsK(undefined);
                    else setSfsK('auto');
                    return;
                  }
                  const digits = v.replace(/\D/g, '');
                  if (digits === '') {
                    setSfsK(undefined);
                    return;
                  }
                  if (clearIfDefault(digits, def)) setSfsK(undefined);
                  else setSfsK(digits);
                }}
                placeholder="auto or integer"
              />
            </Tooltip>
            <Select
              label="sfs_direction"
              data={sfsDirectionData}
              value={effectiveSfsDirection || null}
              disabled={sfsDirectionUnavailable}
              placeholder={sfsDirectionUnavailable ? 'Schema enums unavailable' : undefined}
              description={
                sfsDirectionUnavailable ? 'Schema did not provide sfs_direction options.' : undefined
              }
              onChange={(v) => {
                const next = v || undefined;
                if (clearIfDefault(next, propDefault('sfs_direction'))) setSfsDirection(undefined);
                else setSfsDirection(next);
              }}
            />
            <NumberInput
              label="sfs_cv"
              value={effectiveSfsCv}
              onChange={(v) => {
                const next = toIntOrUndef(v);
                if (clearIfDefault(next, propDefault('sfs_cv'))) setSfsCv(undefined);
                else setSfsCv(next);
              }}
              allowDecimal={false}
              min={2}
              max={20}
            />
            <Tooltip label="Number of parallel jobs for SFS; empty = use default">
              <NumberInput
                label="sfs_n_jobs"
                value={effectiveSfsNJobs}
                onChange={(v) => {
                  const next = toIntOrUndef(v);
                  if (clearIfDefault(next, propDefault('sfs_n_jobs'))) setSfsNJobs(undefined);
                  else setSfsNJobs(next);
                }}
                allowDecimal={false}
                min={-1}
              />
            </Tooltip>
          </Stack>
        )}
      </Box>
    </ConfigCardShell>
  );
}
