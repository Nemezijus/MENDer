import { Select } from '@mantine/core';

/**
 * ParamSelect
 * - Wraps Mantine Select.
 * - Normalizes onChange into (string | undefined) by default.
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
        onChange(v || undefined);
      }}
    />
  );
}
