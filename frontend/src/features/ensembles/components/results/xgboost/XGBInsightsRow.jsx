import { Box, Group } from '@mantine/core';

import XGBLearningCurvesSection from './XGBLearningCurvesSection.jsx';
import XGBFeatureImportanceSection from './XGBFeatureImportanceSection.jsx';

export default function XGBInsightsRow({ report }) {
  return (
    <Box className="ensScrollX">
      <Group align="stretch" wrap="nowrap" gap="md">
        <Box className="xgbInsightCol">
          <XGBLearningCurvesSection report={report} />
        </Box>

        <Box className="xgbInsightCol">
          <XGBFeatureImportanceSection report={report} />
        </Box>
      </Group>
    </Box>
  );
}
