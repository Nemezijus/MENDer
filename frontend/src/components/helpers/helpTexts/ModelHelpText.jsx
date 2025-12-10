// frontend/src/components/helpers/helpTexts/ModelHelpText.jsx
import { Stack, Text, List } from '@mantine/core';

export function ModelIntroText() {
  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        What is a model?
      </Text>

      <Text size="xs" c="dimmed">
        A model is the algorithm that learns patterns from your training data
        and makes predictions on new data. Different models have different
        assumptions, strengths and trade-offs in terms of accuracy,
        interpretability, and training time.
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
            – linear model for classification. Works well when classes are
            roughly linearly separable and you want interpretable weights.
          </List.Item>
        )}

        {isVisible('svm') && (
          <List.Item>
            <Text span {...labelStyle('svm')}>
              Support Vector Machine (SVM)
            </Text>{' '}
            – flexible classifier with different kernels. Good for
            medium-sized datasets and non-linear decision boundaries.
          </List.Item>
        )}

        {isVisible('tree') && (
          <List.Item>
            <Text span {...labelStyle('tree')}>
              Decision tree
            </Text>{' '}
            – interpretable tree that splits the data by feature thresholds.
            Can capture non-linear interactions but may overfit if not
            regularised.
          </List.Item>
        )}

        {isVisible('forest') && (
          <List.Item>
            <Text span {...labelStyle('forest')}>
              Random forest
            </Text>{' '}
            – an ensemble of decision trees. More robust than a single tree,
            often a strong default for many classification problems.
          </List.Item>
        )}

        {isVisible('knn') && (
          <List.Item>
            <Text span {...labelStyle('knn')}>
              k-Nearest Neighbours (kNN)
            </Text>{' '}
            – compares new examples to the closest training samples. Simple and
            non-parametric, but can be slow and sensitive to feature scaling.
          </List.Item>
        )}

        {isVisible('linreg') && (
          <List.Item>
            <Text span {...labelStyle('linreg')}>
              Linear regression
            </Text>{' '}
            – standard baseline for regression. Assumes a linear relationship
            between features and target.
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
              – inverse regularisation strength. Smaller values mean stronger
              regularisation.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Penalty</Text>{' '}
              – type of regularisation (e.g. L2, L1, elastic net).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Solver</Text>{' '}
              – optimisation algorithm. Some solvers support only specific
              penalties.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max iterations</Text>{' '}
              – maximum number of optimisation iterations before stopping.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – reweights classes (e.g.{' '}
              <Text span fw={600}>balanced</Text>
              ) to handle class imbalance.
            </List.Item>
            <List.Item>
              <Text span fw={600}>L1 ratio</Text>{' '}
              – balance between L1 and L2 when using elastic-net penalty.
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
              – shape of the decision function (linear, RBF, polynomial, etc.).
            </List.Item>
            <List.Item>
              <Text span fw={600}>C</Text>{' '}
              – penalty for misclassification. Larger values fit the training
              data more closely.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Degree</Text>{' '}
              – polynomial degree for the polynomial kernel.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Gamma</Text>{' '}
              – controls the influence of individual training points for RBF
              and related kernels.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Coef0</Text>{' '}
              – independent term added in polynomial / sigmoid kernels.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Shrinking</Text>{' '}
              – whether to use the shrinking heuristic for speed.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Probability</Text>{' '}
              – enables probability estimates (slower training and inference).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Tolerance</Text>{' '}
              – stopping criterion for optimisation (smaller = more precise,
              slower).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Cache size</Text>{' '}
              – size of the kernel cache in MB. Larger values may speed up
              training.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – balances classes in the loss function (e.g.{' '}
              <Text span fw={600}>balanced</Text>).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max iterations</Text>{' '}
              – maximum number of iterations, or -1 for no explicit limit.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Decision shape</Text>{' '}
              – multi-class decision function shape (e.g.{' '}
              <Text span fw={600}>ovr</Text>).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Break ties</Text>{' '}
              – when enabled, breaks ties between classes more carefully in
              multi-class problems.
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
              – measure used to split nodes (e.g. gini, entropy).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Splitter</Text>{' '}
              – strategy used to choose the split at each node (best vs.
              random).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max depth</Text>{' '}
              – maximum depth of the tree. Smaller values reduce overfitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples split</Text>{' '}
              – minimum number of samples needed to split an internal node.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples leaf</Text>{' '}
              – minimum number of samples required in a leaf node.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min weight fraction</Text>{' '}
              – minimum weighted fraction of the input samples required at a
              leaf node.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max features</Text>{' '}
              – number (or fraction) of features considered at each split
              (int, float, sqrt, log2 or none).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max leaf nodes</Text>{' '}
              – maximum number of leaf nodes. Limits model complexity.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min impurity decrease</Text>{' '}
              – minimum impurity decrease required for a split.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – reweights classes to reduce bias for imbalanced data.
            </List.Item>
            <List.Item>
              <Text span fw={600}>CCP alpha</Text>{' '}
              – complexity parameter used for minimal cost-complexity pruning.
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
              – number of trees in the forest.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Criterion</Text>{' '}
              – split quality measure (gini, entropy, etc.).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max depth</Text>{' '}
              – maximum depth of each tree.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples split</Text>{' '}
              – minimum samples required to split an internal node.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples leaf</Text>{' '}
              – minimum samples required in a leaf node.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min weight fraction</Text>{' '}
              – minimum weighted fraction of samples at a leaf node.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max features</Text>{' '}
              – number (or fraction) of features considered at each split.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max leaf nodes</Text>{' '}
              – maximum number of leaf nodes per tree.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min impurity decrease</Text>{' '}
              – minimum impurity decrease required for a split.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Bootstrap</Text>{' '}
              – whether each tree is trained on a bootstrap sample of the data.
            </List.Item>
            <List.Item>
              <Text span fw={600}>OOB score</Text>{' '}
              – uses out-of-bag samples to estimate generalisation performance.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Jobs (n_jobs)</Text>{' '}
              – number of CPU cores used for fitting (if supported).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Random state</Text>{' '}
              – random seed for reproducible results.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Warm start</Text>{' '}
              – when enabled, allows adding more trees to an existing forest.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – balances classes (e.g.{' '}
              <Text span fw={600}>balanced</Text> or{' '}
              <Text span fw={600}>balanced_subsample</Text>).
            </List.Item>
            <List.Item>
              <Text span fw={600}>CCP alpha</Text>{' '}
              – complexity parameter for cost-complexity pruning.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max samples</Text>{' '}
              – if set, number (or fraction) of samples used to train each
              tree.
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
              – number of neighbours used to make a prediction.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Weights</Text>{' '}
              – how neighbours are weighted (uniform vs. distance).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Algorithm</Text>{' '}
              – search strategy for nearest neighbours (auto, ball-tree, etc.).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Leaf size</Text>{' '}
              – tree leaf size parameter for tree-based search methods.
            </List.Item>
            <List.Item>
              <Text span fw={600}>p</Text>{' '}
              – power parameter for the Minkowski metric (e.g. p=2 is Euclidean).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Metric</Text>{' '}
              – distance measure (Minkowski, Euclidean, Manhattan, Chebyshev).
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
              – whether to fit an intercept term.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Copy X</Text>{' '}
              – whether to copy the input matrix or overwrite it during
              fitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Jobs (n_jobs)</Text>{' '}
              – number of CPU cores used for fitting (if supported).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Positive coefficients</Text>{' '}
              – if enabled, forces regression coefficients to be non-negative.
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
