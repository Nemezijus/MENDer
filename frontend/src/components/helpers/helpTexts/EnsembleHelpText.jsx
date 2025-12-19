import { Stack, Text, List } from '@mantine/core';

/* ---------- short previews (used in the “C” area) ---------- */

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
        Trains many copies of the same estimator on bootstrap samples and averages/votes.
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
        Sequentially adds weak learners, focusing more on previous errors.
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

/* ---------- expanded help ---------- */

function VotingDetailsText({ effectiveTask, votingType }) {
  const isReg = effectiveTask === 'regression';

  return (
    <Stack gap="xs">
      <Text fw={600} size="sm">
        How to choose settings
      </Text>

      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>Simple vs Advanced</Text> – Simple uses default hyperparameters. Advanced lets you tune
          estimators and optionally set weights.
        </List.Item>

        {!isReg && (
          <List.Item>
            <Text span fw={600}>Hard vs Soft</Text> – Hard voting combines labels. Soft voting averages probabilities.
          </List.Item>
        )}

        {!isReg && votingType === 'soft' && (
          <List.Item>
            <Text span fw={600}>Soft voting requirement</Text> – all estimators must support{' '}
            <Text span fw={600}>predict_proba</Text>.
          </List.Item>
        )}

        <List.Item>
          <Text span fw={600}>Prefer diversity</Text> – mixing model families often improves results.
        </List.Item>

        <List.Item>
          <Text span fw={600}>Duplicates</Text> – identical estimators act like implicit weighting; prefer explicit weights.
        </List.Item>
      </List>
    </Stack>
  );
}

function BaggingDetailsText() {
  return (
    <Stack gap="xs">
      <Text fw={600} size="sm">
        How to choose settings
      </Text>

      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>n_estimators</Text> – more estimators usually increases stability but costs time.
        </List.Item>
        <List.Item>
          <Text span fw={600}>max_samples / max_features</Text> – smaller values add randomness and can reduce overfitting.
        </List.Item>
        <List.Item>
          <Text span fw={600}>bootstrap</Text> – classic bagging uses bootstrap sampling.
        </List.Item>
        <List.Item>
          <Text span fw={600}>oob_score</Text> – out-of-bag scoring can estimate generalization without a separate split.
        </List.Item>
      </List>
    </Stack>
  );
}

function AdaBoostDetailsText({ effectiveTask }) {
  const isReg = effectiveTask === 'regression';

  return (
    <Stack gap="xs">
      <Text fw={600} size="sm">
        How to choose settings
      </Text>

      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>Base estimator</Text> – usually a weak learner (e.g. shallow trees). Too-strong learners can
          overfit quickly.
        </List.Item>
        <List.Item>
          <Text span fw={600}>n_estimators</Text> – more rounds can improve performance but increase overfitting risk.
        </List.Item>
        <List.Item>
          <Text span fw={600}>learning_rate</Text> – smaller values are more conservative; you may need more estimators.
        </List.Item>
        {!isReg && (
          <List.Item>
            <Text span fw={600}>algorithm</Text> – leave default unless you specifically need SAMME/SAMME.R.
          </List.Item>
        )}
      </List>
    </Stack>
  );
}

function XGBoostDetailsText() {
  return (
    <Stack gap="xs">
      <Text fw={600} size="sm">
        How to choose settings
      </Text>

      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>n_estimators + learning_rate</Text> – classic tradeoff: smaller learning_rate usually needs
          more trees.
        </List.Item>
        <List.Item>
          <Text span fw={600}>max_depth</Text> – deeper trees fit more complex patterns but can overfit.
        </List.Item>
        <List.Item>
          <Text span fw={600}>subsample / colsample_bytree</Text> – subsampling can improve generalization.
        </List.Item>
        <List.Item>
          <Text span fw={600}>reg_alpha / reg_lambda</Text> – L1/L2 regularization; increase to reduce overfitting.
        </List.Item>
        <List.Item>
          <Text span fw={600}>gamma / min_child_weight</Text> – control split conservativeness and minimum leaf “strength”.
        </List.Item>
      </List>

      <Text size="xs" c="dimmed">
        Note: XGBoost requires the xgboost Python package to be installed in the backend environment.
      </Text>
    </Stack>
  );
}

/* ---------- router ---------- */

export default function EnsembleHelpText({ kind, effectiveTask, votingType }) {
  if (kind === 'voting') {
    return <VotingDetailsText effectiveTask={effectiveTask} votingType={votingType} />;
  }
  if (kind === 'bagging') {
    return <BaggingDetailsText />;
  }
  if (kind === 'adaboost') {
    return <AdaBoostDetailsText effectiveTask={effectiveTask} />;
  }
  if (kind === 'xgboost') {
    return <XGBoostDetailsText />;
  }

  return (
    <Text size="xs" c="dimmed">
      Help text for this ensemble type is not available yet.
    </Text>
  );
}
