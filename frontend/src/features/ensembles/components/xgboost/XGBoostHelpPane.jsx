import { Stack, Button } from '@mantine/core';

import EnsembleHelpText, { XGBoostIntroText } from '../../../../shared/content/help/EnsembleHelpText.jsx';

export default function XGBoostHelpPane({ showHelp, onToggleHelp }) {
  return (
    <Stack gap="xs">
      <XGBoostIntroText />
      <Button size="xs" variant="subtle" onClick={onToggleHelp}>
        {showHelp ? 'Show less' : 'Show more'}
      </Button>
      {showHelp && <EnsembleHelpText kind="xgboost" />}
    </Stack>
  );
}
