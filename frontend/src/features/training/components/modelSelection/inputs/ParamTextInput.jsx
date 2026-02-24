import { TextInput } from '@mantine/core';

/**
 * ParamTextInput
 * - Wraps Mantine TextInput.
 * - Normalizes onChange into (string | undefined) when emptyToUndefined is true.
 * - This is optional; sections with custom parsing can keep TextInput directly.
 */
export default function ParamTextInput({ value, onChange, emptyToUndefined = true, ...props }) {
  const v = value == null ? '' : String(value);

  return (
    <TextInput
      {...props}
      value={v}
      onChange={(e) => {
        if (!onChange) return;
        const t = e.currentTarget.value;
        if (emptyToUndefined && t === '') {
          onChange(undefined);
          return;
        }
        onChange(t);
      }}
    />
  );
}
