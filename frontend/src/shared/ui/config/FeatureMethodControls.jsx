import {
  Stack,
  Select,
  NumberInput,
  Checkbox,
  TextInput,
  Tooltip,
} from '@mantine/core';

import { numOrUndef, intOrUndef } from '../../utils/coerce.js';

export function PcaControls({
  effectivePcaN,
  effectivePcaVar,
  effectivePcaWhiten,
  propDefault,
  clearIfDefault,
  setPcaN,
  setPcaVar,
  setPcaWhiten,
}) {
  return (
    <Stack gap="sm">
      <Tooltip label="Number of components (optional). Leave empty for variance-based.">
        <NumberInput
          label="pca_n"
          value={effectivePcaN}
          onChange={(v) => {
            const next = intOrUndef(v);
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
            const next = numOrUndef(v);
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
  );
}

export function LdaControls({
  effectiveLdaN,
  effectiveLdaSolver,
  effectiveLdaShrinkage,
  effectiveLdaTol,
  ldaSolverData,
  ldaSolverUnavailable,
  propDefault,
  clearIfDefault,
  setLdaN,
  setLdaSolver,
  setLdaShrinkage,
  setLdaTol,
}) {
  return (
    <Stack gap="sm">
      <Tooltip label="Target number of components (<= n_classes - 1). Leave empty to infer.">
        <NumberInput
          label="lda_n"
          value={effectiveLdaN}
          onChange={(v) => {
            const next = intOrUndef(v);
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
            const next = numOrUndef(v);
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
          const next = numOrUndef(v);
          if (clearIfDefault(next, propDefault('lda_tol'))) setLdaTol(undefined);
          else setLdaTol(next);
        }}
        step={1e-4}
        precision={6}
        min={0}
      />
    </Stack>
  );
}

export function SfsControls({
  effectiveSfsK,
  effectiveSfsDirection,
  effectiveSfsCv,
  effectiveSfsNJobs,
  sfsDirectionData,
  sfsDirectionUnavailable,
  propDefault,
  clearIfDefault,
  setSfsK,
  setSfsDirection,
  setSfsCv,
  setSfsNJobs,
}) {
  return (
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
          const next = intOrUndef(v);
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
            const next = intOrUndef(v);
            if (clearIfDefault(next, propDefault('sfs_n_jobs'))) setSfsNJobs(undefined);
            else setSfsNJobs(next);
          }}
          allowDecimal={false}
          min={-1}
        />
      </Tooltip>
    </Stack>
  );
}
