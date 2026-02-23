import { Box, Stack, Button } from '@mantine/core';

import EnsembleHelpText, {
  XGBoostIntroText,
} from '../../../../shared/content/help/EnsembleHelpText.jsx';

export default function XGBoostHelpPane({ showHelp, onToggleHelp }) {
  return (
    <Box style={{ flex: 1, minWidth: 260 }}>
      <Stack gap="xs">
        <XGBoostIntroText />
        <Button size="xs" variant="subtle" onClick={onToggleHelp}>
          {showHelp ? 'Show less' : 'Show more'}
        </Button>
        {showHelp && <EnsembleHelpText kind="xgboost" />}
      </Stack>
    </Box>
  );
}
