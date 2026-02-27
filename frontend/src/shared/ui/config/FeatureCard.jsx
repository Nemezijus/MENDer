import {
  Stack,
  Select,
  Box,
} from '@mantine/core';

import { useSchemaDefaults } from '../../schema/SchemaDefaultsContext.jsx';
import { useFeatureStore } from '../../state/useFeatureStore.js';
import FeatureHelpText, {
  FeatureIntroText,
} from '../../content/help/FeatureHelpText.jsx';
import {
  getVariantSchema,
  enumFromSubSchema,
  toSelectData,
  listDiscriminatorValues,
} from '../../utils/schema/jsonSchema.js';
import { numOrUndef } from '../../utils/coerce.js';
import ConfigCardShell from './common/ConfigCardShell.jsx';
import { PcaControls, LdaControls, SfsControls } from './FeatureMethodControls.jsx';

import '../styles/forms.css';

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

  const effectiveSfsK = sfs_k ?? propDefault('sfs_k') ?? '';
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
            classNames={{
              input: 'configSelectInput',
            }}
          />
        </Stack>
      )}
      right={<FeatureIntroText />}
      help={<FeatureHelpText selectedMethod={effectiveMethod} />}
    >
      {/* Method-specific controls (grow downward, help text stays relatively stable) */}
      <Box mt="md">
        {effectiveMethod === 'pca' && (
          <PcaControls
            effectivePcaN={effectivePcaN}
            effectivePcaVar={effectivePcaVar}
            effectivePcaWhiten={effectivePcaWhiten}
            propDefault={propDefault}
            clearIfDefault={clearIfDefault}
            setPcaN={setPcaN}
            setPcaVar={setPcaVar}
            setPcaWhiten={setPcaWhiten}
          />
        )}

        {effectiveMethod === 'lda' && (
          <LdaControls
            effectiveLdaN={effectiveLdaN}
            effectiveLdaSolver={effectiveLdaSolver}
            effectiveLdaShrinkage={effectiveLdaShrinkage}
            effectiveLdaTol={effectiveLdaTol}
            ldaSolverData={ldaSolverData}
            ldaSolverUnavailable={ldaSolverUnavailable}
            propDefault={propDefault}
            clearIfDefault={clearIfDefault}
            setLdaN={setLdaN}
            setLdaSolver={setLdaSolver}
            setLdaShrinkage={setLdaShrinkage}
            setLdaTol={setLdaTol}
          />
        )}

        {effectiveMethod === 'sfs' && (
          <SfsControls
            effectiveSfsK={effectiveSfsK}
            effectiveSfsDirection={effectiveSfsDirection}
            effectiveSfsCv={effectiveSfsCv}
            effectiveSfsNJobs={effectiveSfsNJobs}
            sfsDirectionData={sfsDirectionData}
            sfsDirectionUnavailable={sfsDirectionUnavailable}
            propDefault={propDefault}
            clearIfDefault={clearIfDefault}
            setSfsK={setSfsK}
            setSfsDirection={setSfsDirection}
            setSfsCv={setSfsCv}
            setSfsNJobs={setSfsNJobs}
          />
        )}
      </Box>
    </ConfigCardShell>
  );
}
