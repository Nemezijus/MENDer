import { Checkbox } from '@mantine/core';

/**
 * ParamCheckbox
 * - Wraps Mantine Checkbox.
 * - Normalizes onChange into (boolean).
 */
export default function ParamCheckbox({ checked, onChange, ...props }) {
  return (
    <Checkbox
      {...props}
      checked={Boolean(checked)}
      onChange={(e) => {
        if (!onChange) return;
        onChange(e.currentTarget.checked);
      }}
    />
  );
}
