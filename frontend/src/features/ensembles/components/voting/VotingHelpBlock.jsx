import { Box, Text } from '@mantine/core';
import { lazy, Suspense } from 'react';

const LazyEnsembleHelpText = lazy(() =>
  import('../../../../shared/content/help/EnsembleHelpText.jsx')
);

export default function VotingHelpBlock({ effectiveTask, votingType, mode }) {
  return (
    <Box>
      <Suspense
        fallback={
          <Text size="xs" c="dimmed">
            Loading help…
          </Text>
        }
      >
        <LazyEnsembleHelpText
          kind="voting"
          effectiveTask={effectiveTask}
          votingType={votingType}
          mode={mode}
        />
      </Suspense>
    </Box>
  );
}
