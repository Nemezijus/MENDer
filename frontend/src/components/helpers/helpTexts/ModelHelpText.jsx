import { Stack, Text, List } from '@mantine/core';

export function ModelIntroText() {
  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        What is a model?
      </Text>

      <Text size="xs" c="dimmed">
        A model is the algorithm that learns patterns from your training data
        and makes predictions on new data. Different models make different
        assumptions and trade off accuracy, interpretability, robustness, and
        training speed.
      </Text>
    </Stack>
  );
}

export function ModelDetailsText({
  selectedAlgo,
  effectiveTask,
  visibleAlgos,
}) {
  const algo = selectedAlgo || null;
  const allowed = new Set(visibleAlgos || []);

  const isVisible = (name) =>
    allowed.size === 0 || allowed.has(name);

  const isSelected = (name) => algo === name;

  const labelStyle = (name) => ({
    fw: isSelected(name) ? 700 : 600,
    c: isSelected(name) ? 'blue' : undefined,
  });

  const taskNote =
    effectiveTask === 'classification'
      ? 'You are working on a classification task, so classification algorithms are most relevant.'
      : effectiveTask === 'regression'
      ? 'You are working on a regression task, so regression algorithms are most relevant.'
      : 'If the task is not set yet, you can still explore models. They will be filtered once the task is known.';

  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        Choosing a model
      </Text>

      <Text size="xs" c="dimmed">
        No single model is best for all problems. Simpler models are easier to
        interpret and faster to train, while more flexible models can capture
        complex patterns but may overfit.
      </Text>

      <Text size="xs" c="dimmed">
        {taskNote}
      </Text>

      <List spacing={4} size="xs" mt="xs">
        {isVisible('logreg') && (
          <List.Item>
            <Text span {...labelStyle('logreg')}>
              Logistic regression
            </Text>{' '}
            – linear classifier that estimates class probabilities. A strong,
            interpretable baseline when relationships are approximately linear.
          </List.Item>
        )}

        {isVisible('svm') && (
          <List.Item>
            <Text span {...labelStyle('svm')}>
              Support Vector Machine (SVM)
            </Text>{' '}
            – powerful classifier that can model complex boundaries using
            kernels. Works well on medium-sized datasets.
          </List.Item>
        )}

        {isVisible('tree') && (
          <List.Item>
            <Text span {...labelStyle('tree')}>
              Decision tree
            </Text>{' '}
            – rule-based model that splits data by feature thresholds. Easy to
            interpret but prone to overfitting if unconstrained.
          </List.Item>
        )}

        {isVisible('forest') && (
          <List.Item>
            <Text span {...labelStyle('forest')}>
              Random forest
            </Text>{' '}
            – ensemble of decision trees that improves robustness and accuracy.
            Often a strong default choice.
          </List.Item>
        )}

        {isVisible('knn') && (
          <List.Item>
            <Text span {...labelStyle('knn')}>
              k-Nearest Neighbours (kNN)
            </Text>{' '}
            – predicts based on similarity to nearby samples. Simple but
            sensitive to feature scaling and dataset size.
          </List.Item>
        )}

        {isVisible('linreg') && (
          <List.Item>
            <Text span {...labelStyle('linreg')}>
              Linear regression
            </Text>{' '}
            – models a linear relationship between features and target. Useful
            as a fast, interpretable regression baseline.
          </List.Item>
        )}
      </List>
    </Stack>
  );
}

export function ModelParamsText({ selectedAlgo }) {
  if (!selectedAlgo) {
    return (
      <Text size="xs" c="dimmed">
        Select an algorithm above to see a short description of its key
        parameters.
      </Text>
    );
  }

  switch (selectedAlgo) {
    case 'logreg':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Logistic regression parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>C</Text>{' '}
              – inverse regularisation strength. Smaller values enforce stronger
              regularisation and reduce overfitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Penalty</Text>{' '}
              – type of regularisation applied to coefficients (L1, L2, or
              elastic net).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Solver</Text>{' '}
              – optimisation algorithm used to fit the model. Some solvers only
              support certain penalties.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max iterations</Text>{' '}
              – maximum number of optimisation steps before stopping. Increase
              if the model does not converge.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – adjusts the importance of classes to compensate for imbalance
              (e.g. balanced).
            </List.Item>
            <List.Item>
              <Text span fw={600}>L1 ratio</Text>{' '}
              – controls the mix between L1 and L2 regularisation when using
              elastic net.
            </List.Item>
          </List>
        </Stack>
      );

    case 'svm':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            SVM parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Kernel</Text>{' '}
              – defines the shape of the decision boundary. Linear is fastest;
              RBF and polynomial capture non-linear patterns.
            </List.Item>
            <List.Item>
              <Text span fw={600}>C</Text>{' '}
              – penalty for misclassification. Larger values fit training data
              more closely but may overfit.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Degree</Text>{' '}
              – degree of the polynomial kernel. Higher values increase model
              complexity and training time.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Gamma</Text>{' '}
              – controls how far the influence of a single sample reaches.
              Larger values lead to tighter, more complex boundaries.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Coef0</Text>{' '}
              – constant term used by polynomial and sigmoid kernels. Affects
              how strongly higher-order terms contribute.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Shrinking</Text>{' '}
              – enables a heuristic that can speed up optimisation on large
              datasets.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Probability</Text>{' '}
              – enables probability estimates via additional calibration, which
              slows training and prediction.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Tolerance</Text>{' '}
              – stopping threshold for optimisation. Smaller values increase
              precision but slow training.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Cache size</Text>{' '}
              – memory (in MB) used to cache kernel values. Larger caches can
              improve speed at the cost of RAM.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – adjusts class importance to mitigate imbalance.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max iterations</Text>{' '}
              – upper limit on optimisation steps. Use higher values if
              convergence warnings appear.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Decision shape</Text>{' '}
              – strategy for multi-class problems (e.g. one-vs-rest).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Break ties</Text>{' '}
              – refines tie-breaking between classes in multi-class settings,
              at a small computational cost.
            </List.Item>
          </List>
        </Stack>
      );

    case 'tree':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Decision tree parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Criterion</Text>{' '}
              – measure of split quality (e.g. gini or entropy). Affects how
              class purity is evaluated.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Splitter</Text>{' '}
              – strategy for choosing splits. Random splitting can reduce
              variance at the cost of interpretability.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max depth</Text>{' '}
              – maximum tree depth. Larger values increase model complexity and
              risk overfitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples split</Text>{' '}
              – minimum samples required to split a node. Larger values make the
              tree more conservative.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples leaf</Text>{' '}
              – minimum samples per leaf. Higher values smooth predictions and
              reduce overfitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max features</Text>{' '}
              – number of features considered at each split. Smaller values
              increase randomness and reduce correlation between splits.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max leaf nodes</Text>{' '}
              – maximum number of terminal nodes. Larger values allow finer
              decision regions but increase overfitting risk.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min impurity decrease</Text>{' '}
              – required reduction in impurity to allow a split. Larger values
              prevent weak, noisy splits.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – balances class importance, useful for imbalanced datasets.
            </List.Item>
            <List.Item>
              <Text span fw={600}>CCP alpha</Text>{' '}
              – pruning strength. Higher values produce simpler, more robust
              trees.
            </List.Item>
          </List>
        </Stack>
      );

    case 'forest':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Random forest parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Trees (n_estimators)</Text>{' '}
              – number of trees in the forest. More trees improve stability but
              increase training time.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max depth</Text>{' '}
              – limits depth of each tree. Smaller values reduce overfitting but
              may underfit.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples split</Text>{' '}
              – minimum samples required to split a node. Higher values make the
              forest more conservative.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples leaf</Text>{' '}
              – minimum samples per leaf. Helps smooth predictions.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max features</Text>{' '}
              – number of features tried at each split. Smaller values increase
              tree diversity.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max leaf nodes</Text>{' '}
              – caps tree complexity. Larger values allow more detailed trees
              but increase variance.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Bootstrap</Text>{' '}
              – whether each tree is trained on a random sample with
              replacement. Improves robustness.
            </List.Item>
            <List.Item>
              <Text span fw={600}>OOB score</Text>{' '}
              – estimates generalisation error using unused samples during
              training.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – adjusts class importance for imbalanced datasets.
            </List.Item>
            <List.Item>
              <Text span fw={600}>CCP alpha</Text>{' '}
              – pruning strength applied to each tree to control overfitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max samples</Text>{' '}
              – number or fraction of samples used per tree. Smaller values
              increase randomness.
            </List.Item>
          </List>
        </Stack>
      );

    case 'knn':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            kNN parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Neighbours</Text>{' '}
              – number of nearest samples considered. Small values are sensitive
              to noise; large values smooth predictions.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Weights</Text>{' '}
              – how neighbours contribute (uniform or distance-based).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Algorithm</Text>{' '}
              – strategy used to search for neighbours (auto, ball-tree, kd-tree).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Leaf size</Text>{' '}
              – affects speed and memory usage of tree-based searches.
            </List.Item>
            <List.Item>
              <Text span fw={600}>p</Text>{' '}
              – power parameter of the Minkowski distance (p=2 corresponds to
              Euclidean distance).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Metric</Text>{' '}
              – distance function used to compare samples.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Jobs (n_jobs)</Text>{' '}
              – number of CPU cores used for neighbour searches.
            </List.Item>
          </List>
        </Stack>
      );

    case 'linreg':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Linear regression parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Fit intercept</Text>{' '}
              – whether to include an intercept term. Disable only if data is
              already centred.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Copy X</Text>{' '}
              – copies the input matrix before fitting to avoid modifying it.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Jobs (n_jobs)</Text>{' '}
              – number of CPU cores used during fitting, if supported.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Positive coefficients</Text>{' '}
              – constrains coefficients to be non-negative, useful for some
              physical or interpretability constraints.
            </List.Item>
          </List>
        </Stack>
      );

    default:
      return (
        <Text size="xs" c="dimmed">
          No parameter help is defined for this algorithm yet.
        </Text>
      );
  }
}

export default function ModelHelpText({
  selectedAlgo,
  effectiveTask,
  visibleAlgos,
}) {
  return (
    <Stack gap="sm">
      <ModelDetailsText
        selectedAlgo={selectedAlgo}
        effectiveTask={effectiveTask}
        visibleAlgos={visibleAlgos}
      />
      <ModelParamsText selectedAlgo={selectedAlgo} />
    </Stack>
  );
}
