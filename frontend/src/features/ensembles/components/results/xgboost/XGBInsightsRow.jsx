import { Box, Group } from '@mantine/core';

import XGBLearningCurvesSection from './XGBLearningCurvesSection.jsx';
import XGBFeatureImportanceSection from './XGBFeatureImportanceSection.jsx';

export default function XGBInsightsRow({ report }) {
  return (
    <Box className="ensOverflowXAuto">
      <Group align="stretch" wrap="nowrap" gap="md">
        <Box className="ensFlexMin250">
          <XGBLearningCurvesSection report={report} />
        </Box>

        <Box className="ensFlexMin250">
          <XGBFeatureImportanceSection report={report} />
        </Box>
      </Group>
    </Box>
  );
}
