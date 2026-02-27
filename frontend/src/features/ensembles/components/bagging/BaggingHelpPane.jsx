import { Stack, Button, Text } from '@mantine/core';
import { lazy, Suspense } from 'react';

import { BaggingIntroText } from '../../../../shared/content/help/EnsembleIntroText.jsx';

const LazyEnsembleHelpText = lazy(() =>
  import('../../../../shared/content/help/EnsembleHelpText.jsx')
);

export default function BaggingHelpPane({ showHelp, onToggleHelp }) {
  return (
    <Stack gap="xs">
      <BaggingIntroText />
      <Button size="xs" variant="subtle" onClick={onToggleHelp}>
        {showHelp ? 'Show less' : 'Show more'}
      </Button>
      {showHelp && (
        <Suspense
          fallback={
            <Text size="xs" c="dimmed">
              Loading help…
            </Text>
          }
        >
          <LazyEnsembleHelpText kind="bagging" />
        </Suspense>
      )}
    </Stack>
  );
}
