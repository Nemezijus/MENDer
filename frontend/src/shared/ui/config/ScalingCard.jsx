import { Stack, Select } from '@mantine/core';
import { useSchemaDefaults } from '../../schema/SchemaDefaultsContext.jsx';
import { enumFromSubSchema } from '../../utils/schema/jsonSchema.js';
import ScalingHelpText, {
  ScalingIntroText,
} from '../../content/help/ScalingHelpText.jsx';
import { makeScaleOptionLabel } from '../../constants/scaling.js';
import ConfigCardShell from './common/ConfigCardShell.jsx';

import '../styles/forms.css';

export default function ScalingCard({ value, onChange, title = 'Scaling' }) {
  const { enums, scale } = useSchemaDefaults();

  const rawScaleNames =
    (Array.isArray(enums?.ScaleName) && enums.ScaleName.length
      ? enums.ScaleName
      : null) ??
    (enumFromSubSchema(scale?.schema, 'method') ?? []);
  const scaleOptions = (rawScaleNames ?? [])
    .filter((v) => v != null)
    .map((v) => ({
      value: String(v),
      label: makeScaleOptionLabel(v),
    }));

  const optionsUnavailable = scaleOptions.length === 0;

  return (
    <ConfigCardShell
      title={title}
      left={(
        <Stack gap="sm">
          <Select
            label="Scaling method"
            data={scaleOptions}
            value={value}
            onChange={onChange}
            disabled={optionsUnavailable}
            placeholder={optionsUnavailable ? 'Schema enums unavailable' : undefined}
            description={optionsUnavailable ? 'Schema did not provide scaling options.' : undefined}
            classNames={{
              input: 'configSelectInput',
            }}
          />
        </Stack>
      )}
      right={<ScalingIntroText />}
      help={<ScalingHelpText selectedScaling={value} />}
    />
  );
}
