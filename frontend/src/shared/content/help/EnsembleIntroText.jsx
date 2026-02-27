import { Stack, Text } from '@mantine/core';

// NOTE:
// This file intentionally contains only the short "preview" intro components.
// The expanded help (EnsembleHelpText.jsx) is lazy-loaded so the large help
// content does not inflate the initial bundle when help is collapsed.

export function VotingIntroText({ effectiveTask, votingType }) {
  const isReg = effectiveTask === 'regression';

  return (
    <Stack gap={6}>
      <Text fw={600} size="sm">
        Voting ensemble
      </Text>

      <Text size="xs" c="dimmed">
        Trains multiple base models on the same split and combines their predictions.
      </Text>

      <Text size="xs" c="dimmed">
        {isReg
          ? 'Regression: averages predictions (VotingRegressor).'
          : votingType === 'soft'
          ? 'Soft voting averages predicted probabilities.'
          : 'Hard voting chooses the majority class.'}
      </Text>
    </Stack>
  );
}

export function BaggingIntroText({ effectiveTask }) {
  const isReg = effectiveTask === 'regression';

  return (
    <Stack gap={6}>
      <Text fw={600} size="sm">
        Bagging ensemble
      </Text>

      <Text size="xs" c="dimmed">
        Trains many copies of the same estimator on resampled data and averages/votes.
      </Text>

      <Text size="xs" c="dimmed">
        {isReg
          ? 'Regression: averages predictions (BaggingRegressor).'
          : 'Classification: majority vote across estimators (BaggingClassifier).'}
      </Text>
    </Stack>
  );
}

export function AdaBoostIntroText({ effectiveTask }) {
  const isReg = effectiveTask === 'regression';

  return (
    <Stack gap={6}>
      <Text fw={600} size="sm">
        AdaBoost ensemble
      </Text>

      <Text size="xs" c="dimmed">
        Adds weak learners sequentially, focusing more on the samples it previously got wrong.
      </Text>

      <Text size="xs" c="dimmed">
        {isReg
          ? 'Regression: weighted ensemble of weak regressors (AdaBoostRegressor).'
          : 'Classification: boosts weak classifiers (AdaBoostClassifier).'}
      </Text>
    </Stack>
  );
}

export function XGBoostIntroText({ effectiveTask }) {
  const isReg = effectiveTask === 'regression';

  return (
    <Stack gap={6}>
      <Text fw={600} size="sm">
        XGBoost
      </Text>

      <Text size="xs" c="dimmed">
        Gradient boosted decision trees (high-performance for tabular data).
      </Text>

      <Text size="xs" c="dimmed">
        {isReg ? 'Regression: XGBRegressor.' : 'Classification: XGBClassifier.'}
      </Text>
    </Stack>
  );
}
