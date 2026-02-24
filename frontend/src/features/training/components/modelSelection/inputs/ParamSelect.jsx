import { Select } from '@mantine/core';

/**
 * ParamSelect
 * - Wraps Mantine Select.
 * - Normalizes null/empty selection into undefined by default.
 * - If emptyToUndefined={false}, forwards Mantine's raw value (string | null).
 */
export default function ParamSelect({ value, onChange, emptyToUndefined = true, ...props }) {
  const selectValue = value == null ? null : value;

  return (
    <Select
      {...props}
      value={selectValue}
      onChange={(v) => {
        if (!onChange) return;

        if (!emptyToUndefined) {
          onChange(v);
          return;
        }

        onChange(v ?? undefined);
      }}
    />
  );
}
