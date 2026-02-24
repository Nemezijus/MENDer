import { NumberInput } from '@mantine/core';

/**
 * ParamNumber
 * - Wraps Mantine NumberInput.
 * - Normalizes onChange into (number | undefined) by default.
 * - If emptyToUndefined={false}, forwards Mantine's raw value.
 */
export default function ParamNumber({ value, onChange, emptyToUndefined = true, ...props }) {
  return (
    <NumberInput
      {...props}
      value={value}
      onChange={(v) => {
        if (!onChange) return;

        if (!emptyToUndefined) {
          onChange(v);
          return;
        }

        if (v === '' || v == null) {
          onChange(undefined);
          return;
        }

        const n = typeof v === 'number' ? v : Number(v);
        onChange(Number.isFinite(n) ? n : undefined);
      }}
    />
  );
}
