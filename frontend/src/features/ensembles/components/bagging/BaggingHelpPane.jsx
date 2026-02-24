import { Stack, Button } from '@mantine/core';

import EnsembleHelpText, { BaggingIntroText } from '../../../../shared/content/help/EnsembleHelpText.jsx';

export default function BaggingHelpPane({ showHelp, onToggleHelp }) {
  return (
    <Stack gap="xs">
      <BaggingIntroText />
      <Button size="xs" variant="subtle" onClick={onToggleHelp}>
        {showHelp ? 'Show less' : 'Show more'}
      </Button>
      {showHelp && <EnsembleHelpText kind="bagging" />}
    </Stack>
  );
}
