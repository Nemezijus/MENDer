import { Card, Stack, Text, Select, Group, Box } from '@mantine/core';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';
import ScalingHelpText, {
  ScalingIntroText,
} from './helpers/helpTexts/ScalingHelpText.jsx';

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
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        {/* Centered, larger title */}
        <Text fw={700} size="lg" align="center">
          {title}
        </Text>

        {/* A + B: controls on the left, short intro on the right */}
        <Group align="flex-start" gap="xl" grow wrap="nowrap">
          <Box style={{ flex: 1, minWidth: 0 }}>
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
          </Box>

          <Box
            style={{
              flex: 1,
              minWidth: 220,
            }}
          >
            <ScalingIntroText />
          </Box>
        </Group>

        {/* C: full-width detailed help text, with selected scaling highlighted */}
        <Box mt="md">
          <ScalingHelpText selectedScaling={value} />
        </Box>
      </Stack>
    </Card>
  );
}
