import {
  Card,
  Stack,
  Group,
  Text,
  Select,
  NumberInput,
  Checkbox,
  TextInput,
  Tooltip,
  Box,
} from '@mantine/core';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';
import { useFeatureStore } from '../state/useFeatureStore.js';
import FeatureHelpText, {
  FeatureIntroText,
} from './helpers/helpTexts/FeatureHelpText.jsx';

/** -------- helpers to read enums from the Features schema -------- **/

function resolveRef(schema, ref) {
  if (!schema || !ref || typeof ref !== 'string') return null;
  const prefix = '#/$defs/';
  if (!ref.startsWith(prefix)) return null;
  const key = ref.slice(prefix.length);
  return schema?.$defs?.[key] ?? null;
}

function getMethodSchema(schema, method) {
  if (!schema || !method) return null;

  // try discriminator mapping
  const mapping = schema?.discriminator?.mapping;
  if (mapping && mapping[method]) {
    const target = resolveRef(schema, mapping[method]);
    if (target) return target;
  }

  // fallback: scan oneOf/anyOf and check properties.method const/default
  const variants = schema?.oneOf || schema?.anyOf || [];
  for (const entry of variants) {
    const target = entry?.$ref ? resolveRef(schema, entry.$ref) : entry;
    const m = target?.properties?.method?.const ?? target?.properties?.method?.default;
    if (m === method) return target || null;
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

function toSelectData(values, { includeNone = false } = {}) {
  const out = [];
  let sawNull = false;
  for (const v of values ?? []) {
    if (v === null) {
      sawNull = true;
      continue;
    }
    out.push({ value: String(v), label: String(v) });
  }
  if (includeNone && sawNull) out.unshift({ value: 'none', label: 'none' });
  return out;
}

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

  const methods = toSelectData(enums?.FeatureName ?? ['none', 'pca', 'lda', 'sfs']);
  const sub = getMethodSchema(features?.schema, method);

  const ldaSolverData = toSelectData(
    enumFromSubSchema(sub, 'lda_solver', ['svd', 'lsqr', 'eigen']),
  );
  const sfsDirectionData = toSelectData(
    enumFromSubSchema(sub, 'sfs_direction', ['forward', 'backward']),
  );

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        {/* Centered, larger title */}
        <Text fw={700} size="lg" align="center">
          {title}
        </Text>

        {/* Row A/B: dropdown + short intro */}
        <Group align="flex-start" gap="xl" grow wrap="nowrap">
          <Box style={{ flex: 1, minWidth: 0 }}>
            <Stack gap="sm">
              <Select
                label="Feature method"
                data={methods}
                value={method}
                onChange={setMethod}
                styles={{
                  input: {
                    borderWidth: 2,
                    borderColor: '#5c94ccff',
                  },
                }}
              />
            </Stack>
          </Box>

          <Box
            style={{
              flex: 1,
              minWidth: 220,
            }}
          >
            <FeatureIntroText />
          </Box>
        </Group>

        {/* Block C: full-width detailed help text, with selected method highlighted
            and parameter descriptions only for the active method */}
        <Box mt="md">
          <FeatureHelpText selectedMethod={method} />
        </Box>

        {/* Method-specific controls (grow downward, help text stays relatively stable) */}
        <Box mt="md">
          {/* PCA controls */}
          {method === 'pca' && (
            <Stack gap="sm">
              <Tooltip label="Number of components (optional). Leave empty for variance-based.">
                <NumberInput
                  label="pca_n"
                  value={pca_n}
                  onChange={setPcaN}
                  placeholder="empty = auto by variance"
                  allowDecimal={false}
                  min={1}
                />
              </Tooltip>
              <Tooltip label="Retained variance if pca_n is empty (e.g., 0.95)">
                <NumberInput
                  label="pca_var"
                  value={pca_var}
                  onChange={setPcaVar}
                  min={0.0}
                  max={1.0}
                  step={0.01}
                />
              </Tooltip>
              <Checkbox
                label="pca_whiten"
                checked={pca_whiten}
                onChange={(e) => setPcaWhiten(e.currentTarget.checked)}
              />
            </Stack>
          )}

          {/* LDA controls */}
          {method === 'lda' && (
            <Stack gap="sm">
              <Tooltip label="Target number of components (<= n_classes - 1). Leave empty to infer.">
                <NumberInput
                  label="lda_n"
                  value={lda_n}
                  onChange={setLdaN}
                  allowDecimal={false}
                  min={1}
                />
              </Tooltip>
              <Select
                label="lda_solver"
                data={ldaSolverData}
                value={lda_solver}
                onChange={setLdaSolver}
              />
              <Tooltip label="Only used with 'lsqr' or 'eigen'. Leave empty for None.">
                <NumberInput
                  label="lda_shrinkage"
                  value={lda_shrinkage}
                  onChange={setLdaShrinkage}
                  step={0.1}
                  min={0}
                />
              </Tooltip>
              <NumberInput
                label="lda_tol"
                value={lda_tol}
                onChange={setLdaTol}
                step={1e-4}
                precision={6}
                min={0}
              />
            </Stack>
          )}

          {/* SFS controls */}
          {method === 'sfs' && (
            <Stack gap="sm">
              <Tooltip label="Number of selected features; 'auto' or integer">
                <TextInput
                  label="sfs_k"
                  value={String(sfs_k)}
                  onChange={(e) => {
                    const v = e.currentTarget.value.trim();
                    if (v === '' || v.toLowerCase() === 'auto') setSfsK('auto');
                    else setSfsK(v.replace(/\D/g, ''));
                  }}
                  placeholder="auto or integer"
                />
              </Tooltip>
              <Select
                label="sfs_direction"
                data={sfsDirectionData}
                value={sfs_direction}
                onChange={setSfsDirection}
              />
              <NumberInput
                label="sfs_cv"
                value={sfs_cv}
                onChange={setSfsCv}
                allowDecimal={false}
                min={2}
                max={20}
              />
              <Tooltip label="Number of parallel jobs for SFS; empty = use default">
                <NumberInput
                  label="sfs_n_jobs"
                  value={sfs_n_jobs}
                  onChange={setSfsNJobs}
                  allowDecimal={false}
                  min={-1}
                />
              </Tooltip>
            </Stack>
          )}
        </Box>
      </Stack>
    </Card>
  );
}
