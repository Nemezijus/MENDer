import { Stack, Button } from '@mantine/core';

import EnsembleHelpText, { AdaBoostIntroText } from '../../../../shared/content/help/EnsembleHelpText.jsx';

export default function AdaBoostHelpPane({ showHelp, onToggleHelp }) {
  return (
    <Stack gap="xs">
      <AdaBoostIntroText />
      <Button size="xs" variant="subtle" onClick={onToggleHelp}>
        {showHelp ? 'Show less' : 'Show more'}
      </Button>
      {showHelp && <EnsembleHelpText kind="adaboost" />}
    </Stack>
  );
}
