import { Box, Group } from '@mantine/core';

import XGBLearningCurvesSection from './XGBLearningCurvesSection.jsx';
import XGBFeatureImportanceSection from './XGBFeatureImportanceSection.jsx';

export default function XGBInsightsRow({ report }) {
  return (
    <Box style={{ overflowX: 'auto' }}>
      <Group align="stretch" wrap="nowrap" gap="md" style={{ flexWrap: 'nowrap' }}>
        <Box style={{ flex: '1 1 0', minWidth: 250 }}>
          <XGBLearningCurvesSection report={report} />
        </Box>

        <Box style={{ flex: '1 1 0', minWidth: 250 }}>
          <XGBFeatureImportanceSection report={report} />
        </Box>
      </Group>
    </Box>
  );
}
