import { Select } from '@mantine/core';

export default function ParamNameSelect({ label, options, value, onChange }) {
  const data = Array.isArray(options) ? options : [];

  return (
    <Select
      label={label || 'Hyperparameter to vary'}
      placeholder={
        data.length
          ? 'Pick a parameter (e.g. C, max_depth)'
          : 'No parameters available for this model'
      }
      data={data}
      value={value || null}
      onChange={onChange}
      searchable
      clearable
    />
  );
}
