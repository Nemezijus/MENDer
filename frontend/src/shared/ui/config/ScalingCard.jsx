import { Stack, Select } from '@mantine/core';
import { useSchemaDefaults } from '../../schema/SchemaDefaultsContext.jsx';
import ScalingHelpText, {
  ScalingIntroText,
} from '../../content/help/ScalingHelpText.jsx';
import ConfigCardShell from './common/ConfigCardShell.jsx';

export default function ScalingCard({ value, onChange, title = 'Scaling' }) {
  const { enums } = useSchemaDefaults();

  const scaleOptions = (
    enums?.ScaleName ?? ['none', 'standard', 'robust', 'minmax', 'maxabs', 'quantile']
  ).map((v) => {
    const labelBase =
      v === 'none'
        ? 'None'
        : String(v).charAt(0).toUpperCase() + String(v).slice(1);
    const suffix =
      v === 'none'
        ? ''
        : String(v).toLowerCase().endsWith('abs')
        ? ' Scaler'
        : v === 'quantile'
        ? ' Transformer'
        : ' Scaler';

    return {
      value: String(v),
      label: `${labelBase}${suffix}`,
    };
  });

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
            styles={{
              input: {
                borderWidth: 2,
                borderColor: '#5c94ccff',
              },
            }}
          />
        </Stack>
      )}
      right={<ScalingIntroText />}
      help={<ScalingHelpText selectedScaling={value} />}
    />
  );
}
