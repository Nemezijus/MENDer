import { Stack, Button, Text } from '@mantine/core';
import { lazy, Suspense } from 'react';

import { XGBoostIntroText } from '../../../../shared/content/help/EnsembleIntroText.jsx';

const LazyEnsembleHelpText = lazy(() =>
  import('../../../../shared/content/help/EnsembleHelpText.jsx')
);

export default function XGBoostHelpPane({ showHelp, onToggleHelp }) {
  return (
    <Stack gap="xs">
      <XGBoostIntroText />
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
          <LazyEnsembleHelpText kind="xgboost" />
        </Suspense>
      )}
    </Stack>
  );
}
