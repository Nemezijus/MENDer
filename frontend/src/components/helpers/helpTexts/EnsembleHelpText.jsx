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
          <Text span fw={600}>
            Simple vs Advanced
          </Text>{' '}
          – Simple uses default hyperparameters. Advanced lets you tune estimators and optionally set weights.
        </List.Item>

        {!isReg && (
          <List.Item>
            <Text span fw={600}>
              Hard vs Soft
            </Text>{' '}
            – Hard voting combines labels. Soft voting averages probabilities.
          </List.Item>
        )}

        {!isReg && votingType === 'soft' && (
          <List.Item>
            <Text span fw={600}>
              Soft voting requirement
            </Text>{' '}
            – all estimators must support <Text span fw={600}>predict_proba</Text>.
          </List.Item>
        )}

        <List.Item>
          <Text span fw={600}>
            Prefer diversity
          </Text>{' '}
          – mixing model families often improves results.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Duplicates
          </Text>{' '}
          – identical estimators act like implicit weighting; prefer explicit weights.
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

      <List spacing={6} size="xs">
        <List.Item>
          <Text span fw={600}>
            Number of estimators
          </Text>{' '}
          – how many base models you train. More estimators usually reduce variance and make results more stable, but
          increase training time. Typical range: <Text span fw={600}>25–200</Text>. If results look noisy, increase this.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Max samples (fraction)
          </Text>{' '}
          – fraction of the training fold used to train each estimator.{' '}
          <Text span fw={600}>1.0</Text> means “same size as the training fold” (with replacement if Bootstrap is on).
          Smaller values (e.g. <Text span fw={600}>0.5–0.9</Text>) increase diversity between estimators and can reduce
          overfitting, but each estimator learns from less data.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Max features (fraction)
          </Text>{' '}
          – fraction of input features used per estimator (feature subsampling / random subspace). Lower values increase
          estimator diversity and can improve generalization in high-dimensional problems, but may reduce accuracy if too
          low. Common starting points: <Text span fw={600}>0.5–1.0</Text>.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Bootstrap
          </Text>{' '}
          – when enabled, each estimator trains on a bootstrap sample (sampling <Text span fw={600}>with replacement</Text>
          ). This is classic bagging and adds randomness. If disabled, sampling is{' '}
          <Text span fw={600}>without replacement</Text>. With Bootstrap off and Max samples = 1.0, every estimator sees
          the same rows (so only Max features adds randomness).
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Bootstrap features
          </Text>{' '}
          – when enabled, each estimator also trains on a resampled subset of features. This can further increase
          diversity, especially when you have many correlated features. If you already use Max features &lt; 1.0, this may
          be redundant.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Out-of-bag score
          </Text>{' '}
          – only meaningful when <Text span fw={600}>Bootstrap</Text> is enabled. For each estimator, some training samples
          are not selected into its bootstrap sample (“out-of-bag” samples). The out-of-bag score evaluates predictions on
          those left-out samples and gives a built-in generalization estimate without creating a separate validation set.
          This is most useful for quick feedback and sanity checks; for reporting, prefer your chosen holdout / k-fold
          split.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Balanced bagging
          </Text>{' '}
          – uses an imbalanced-learn variant that tries to reduce class imbalance inside each estimator’s training sample.
          Recommended when classes are noticeably imbalanced or when you see bagging failures due to class sparsity. If
          your dataset is only mildly imbalanced, you usually don’t need it.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Sampling strategy (Balanced bagging)
          </Text>{' '}
          – controls how classes are balanced inside each bag.{' '}
          <Text span fw={600}>Auto</Text> is the safe default. Options like “majority”, “not minority”, etc. decide which
          classes are down-sampled. If you’re unsure, keep <Text span fw={600}>Auto</Text>.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Replacement (Balanced bagging)
          </Text>{' '}
          – whether the class-balancing sampler is allowed to sample with replacement. Turning this on can help when some
          classes have few samples, but may increase duplicate rows inside a bag.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Number of jobs
          </Text>{' '}
          – parallelism. Higher values use more CPU cores to train estimators faster. If supported,{' '}
          <Text span fw={600}>-1</Text> means “use all cores”. If your machine becomes unresponsive, reduce this.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Random state
          </Text>{' '}
          – seed for reproducibility. Set it to a fixed value (e.g. 42) to make results repeatable across runs.
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

      <List spacing={6} size="xs">
        <List.Item>
          <Text span fw={600}>
            Base estimator
          </Text>{' '}
          – AdaBoost works best with <Text span fw={600}>weak learners</Text>. The classic choice is a decision stump
          (a very shallow tree). If you use a strong learner (deep trees, complex models), AdaBoost can overfit quickly
          and become unstable.
          {!isReg && (
            <Text size="xs" c="dimmed">
              Tip: for classification, start with a shallow tree-like base learner.
            </Text>
          )}
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Number of estimators
          </Text>{' '}
          – number of boosting rounds (how many weak learners are added sequentially). Higher values can improve
          performance, but also increase training time and overfitting risk. Typical range:{' '}
          <Text span fw={600}>50–500</Text>. If you reduce the learning rate, you usually need more estimators.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Learning rate
          </Text>{' '}
          – scales how much each new learner contributes. Smaller values make boosting more conservative and often
          improve generalization, but you typically need more estimators. Good starting points:{' '}
          <Text span fw={600}>0.05–0.5</Text>. If results are unstable, try lowering it.
        </List.Item>

        {!isReg && (
          <List.Item>
            <Text span fw={600}>
              Algorithm
            </Text>{' '}
            – controls the boosting variant. If you’re unsure, keep the default. Older sklearn versions used
            SAMME/SAMME.R; newer versions have changed defaults and may deprecate some options. Only change this if you
            know you need it.
          </List.Item>
        )}

        <List.Item>
          <Text span fw={600}>
            Random state
          </Text>{' '}
          – seed for reproducibility. Set to a fixed value (e.g. 42) to make results repeatable across runs (especially
          with stochastic base estimators).
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Practical tuning recipe
          </Text>{' '}
          – if you see overfitting: reduce the learning rate, reduce base estimator complexity, and/or add more data.
          If you see underfitting: increase estimators and/or allow slightly stronger base learners.
        </List.Item>
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

      <List spacing={6} size="xs">
        <List.Item>
          <Text span fw={600}>
            Number of estimators
          </Text>{' '}
          – number of boosted trees. More trees can improve performance, but increase training time and overfitting risk.
          Typical range: <Text span fw={600}>200–2000</Text> (depends heavily on learning rate and dataset size).
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Learning rate
          </Text>{' '}
          – how aggressively each tree updates the model. Smaller values are safer and often generalize better, but need
          more trees. Common starting points: <Text span fw={600}>0.03–0.2</Text>.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Max depth
          </Text>{' '}
          – maximum depth of each tree. Deeper trees can capture more complex patterns but overfit more easily. Typical
          range: <Text span fw={600}>3–10</Text>. If you see overfitting, reduce this.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Subsample
          </Text>{' '}
          – fraction of rows used to grow each tree. Values &lt; 1.0 add randomness and often improve generalization
          (especially on noisy data). Try <Text span fw={600}>0.6–0.9</Text> as a starting range.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Column sample by tree
          </Text>{' '}
          – fraction of features considered per tree. Reducing this (e.g. <Text span fw={600}>0.5–0.9</Text>) can reduce
          overfitting and help with high-dimensional inputs.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            L2 regularization (lambda)
          </Text>{' '}
          – larger values penalize large weights and can reduce overfitting. If your model overfits, try increasing it.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            L1 regularization (alpha)
          </Text>{' '}
          – encourages sparsity in leaf weights. Can help when many features are noisy or redundant.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Min child weight
          </Text>{' '}
          – minimum “amount of information” needed in a leaf. Higher values make the algorithm more conservative (fewer,
          simpler splits), which can help reduce overfitting.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Gamma
          </Text>{' '}
          – minimum loss reduction required to make a split. Higher values make splitting more conservative (often helps
          with overfitting).
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Number of jobs
          </Text>{' '}
          – parallelism. Higher values use more CPU cores. If supported, <Text span fw={600}>-1</Text> means “use all
          cores”.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Random state
          </Text>{' '}
          – seed for reproducibility. Set to a fixed value (e.g. 42) for repeatable runs.
        </List.Item>

        <List.Item>
          <Text span fw={600}>
            Practical tuning recipe
          </Text>{' '}
          – start with learning_rate 0.1, max_depth 4–6, subsample/colsample 0.8, then tune depth/regularization to
          control overfitting. If underfitting, add trees or increase depth slightly.
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
