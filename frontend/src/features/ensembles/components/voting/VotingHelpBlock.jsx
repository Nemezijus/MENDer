import { Box } from '@mantine/core';

import EnsembleHelpText from '../../../../shared/content/help/EnsembleHelpText.jsx';

export default function VotingHelpBlock({ effectiveTask, votingType, mode }) {
  return (
    <Box>
      <EnsembleHelpText kind="voting" effectiveTask={effectiveTask} votingType={votingType} mode={mode} />
    </Box>
  );
}
