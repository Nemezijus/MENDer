import { NumberInput } from '@mantine/core';

/**
 * NumberOverrideInput
 *
 * Pattern used throughout tuning panels:
 * - store keeps overrides-only (valueOverride)
 * - UI shows effective value = override ?? default
 * - when user sets it back to default, clear override (set undefined)
 */
export default function NumberOverrideInput({
  label,
  description,
  min,
  max,
  step,
  precision,
  valueOverride,
  defaultValue,
  onChangeOverride,
  ...props
}) {
  const value = valueOverride ?? defaultValue;

  return (
    <NumberInput
      {...props}
      label={label}
      description={description}
      min={min}
      max={max}
      step={step}
      precision={precision}
      value={value}
      onChange={(next) => {
        if (!onChangeOverride) return;
        const v = next === '' || next == null ? undefined : next;
        if (defaultValue != null && v === defaultValue) {
          onChangeOverride(undefined);
          return;
        }
        onChangeOverride(v);
      }}
    />
  );
}
