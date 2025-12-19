import { Stack, Text, List } from '@mantine/core';

export function EnsembleIntroText() {
  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        What is an ensemble?
      </Text>

      <Text size="xs" c="dimmed">
        An ensemble combines multiple models to make a single prediction. The
        goal is usually better accuracy and stability than any single model,
        especially when base models make different kinds of mistakes.
      </Text>
    </Stack>
  );
}

export function VotingIntroText({ effectiveTask, votingType }) {
  const isReg = effectiveTask === 'regression';

  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        Voting ensemble
      </Text>

      <Text size="xs" c="dimmed">
        Voting trains several base models on the same data split and combines
        their predictions. It’s a simple way to get a more robust result.
      </Text>

      <Text size="xs" c="dimmed">
        {isReg
          ? 'For regression, MENDer uses VotingRegressor (averaging predictions).'
          : votingType === 'soft'
          ? 'Soft voting averages predicted probabilities (requires predict_proba on every estimator).'
          : 'Hard voting chooses the class by majority vote.'}
      </Text>
    </Stack>
  );
}

function VotingDetailsText({ effectiveTask, votingType, mode }) {
  const isReg = effectiveTask === 'regression';

  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        How to choose settings
      </Text>

      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>Simple vs Advanced</Text> – Simple mode uses each
          model’s default hyperparameters. Advanced mode lets you tune each
          estimator and optionally set weights.
        </List.Item>

        {!isReg && (
          <List.Item>
            <Text span fw={600}>Hard vs Soft</Text> – Hard voting combines class
            labels. Soft voting averages probabilities and often performs better
            when base models are well-calibrated.
          </List.Item>
        )}

        {!isReg && votingType === 'soft' && (
          <List.Item>
            <Text span fw={600}>Soft voting requirement</Text> – every estimator
            must support <Text span fw={600}>predict_proba</Text>. For example,
            an SVM typically needs <Text span fw={600}>probability=true</Text>.
          </List.Item>
        )}

        <List.Item>
          <Text span fw={600}>Prefer diversity</Text> – mixing different model
          families (linear + tree + kernel) is usually better than repeating the
          same algorithm.
        </List.Item>

        <List.Item>
          <Text span fw={600}>Duplicates</Text> – repeating the same model type
          can act like implicit weighting (especially if hyperparameters are
          identical). If you really want to emphasize one estimator, prefer
          explicit weights in Advanced mode.
        </List.Item>

        {!isReg && (
          <List.Item>
            <Text span fw={600}>Odd number of models</Text> – for hard voting,
            3 or 5 estimators often avoids ties. (Ties can still happen with
            class imbalance or multi-class problems.)
          </List.Item>
        )}

        <List.Item>
          <Text span fw={600}>What is not included (yet)</Text> – stacking /
          blending and per-estimator data splits are intentionally not part of
          Voting. Those belong to more advanced ensemble paradigms.
        </List.Item>
      </List>

      <Text size="xs" c="dimmed">
        Tip: if Soft voting fails, switch to Hard voting or adjust estimators so
        they support probabilities.
      </Text>
    </Stack>
  );
}

export default function EnsembleHelpText({ kind, effectiveTask, votingType, mode }) {
  if (kind === 'voting') {
    return (
      <Stack gap="sm">
        <VotingDetailsText
          effectiveTask={effectiveTask}
          votingType={votingType}
          mode={mode}
        />
      </Stack>
    );
  }

  return (
    <Text size="xs" c="dimmed">
      Help text for this ensemble type is not available yet.
    </Text>
  );
}
