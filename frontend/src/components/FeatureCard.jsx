import { Card, Stack, Group, Text, Select, NumberInput, Checkbox, TextInput, Tooltip } from '@mantine/core';
import { useFeatureCtx } from '../state/FeatureContext.jsx';

export default function FeatureCard({ title = 'Features' }) {
  const {
    method, setMethod,
    // PCA
    pca_n, setPcaN,
    pca_var, setPcaVar,
    pca_whiten, setPcaWhiten,
    // LDA
    lda_n, setLdaN,
    lda_solver, setLdaSolver,
    lda_shrinkage, setLdaShrinkage,
    lda_tol, setLdaTol,
    // SFS
    sfs_k, setSfsK,
    sfs_direction, setSfsDirection,
    sfs_cv, setSfsCv,
    sfs_n_jobs, setSfsNJobs,
  } = useFeatureCtx();

  const methodSelect = (
    <Select
      label="Feature method"
      data={[
        { value: 'none', label: 'none' },
        { value: 'pca', label: 'pca' },
        { value: 'lda', label: 'lda' },
        { value: 'sfs', label: 'sfs' },
      ]}
      value={method}
      onChange={setMethod}
    />
  );

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        <Group justify="space-between" wrap="nowrap">
          <Text fw={500}>{title}</Text>
        </Group>

        {methodSelect}

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
              data={[
                { value: 'svd', label: 'svd' },
                { value: 'lsqr', label: 'lsqr' },
                { value: 'eigen', label: 'eigen' },
              ]}
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
                  // allow 'auto' or integer-like
                  if (v === '' || v.toLowerCase() === 'auto') setSfsK('auto');
                  else setSfsK(v.replace(/\D/g, '')); // keep digits only
                }}
                placeholder="auto or integer"
              />
            </Tooltip>
            <Select
              label="sfs_direction"
              data={[
                { value: 'forward', label: 'forward' },
                { value: 'backward', label: 'backward' },
              ]}
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
      </Stack>
    </Card>
  );
}
