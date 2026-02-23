import { Switch } from '@mantine/core';

/**
 * ParamSwitch
 * - Wraps Mantine Switch.
 * - Normalizes onChange into (boolean).
 */
export default function ParamSwitch({ checked, onChange, ...props }) {
  return (
    <Switch
      {...props}
      checked={Boolean(checked)}
      onChange={(e) => {
        if (!onChange) return;
        onChange(e.currentTarget.checked);
      }}
    />
  );
}
