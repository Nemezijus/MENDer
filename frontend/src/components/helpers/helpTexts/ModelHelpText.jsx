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
      : effectiveTask === 'unsupervised'
      ? 'You are working on an unsupervised task, so clustering / mixture models are most relevant.'
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

        {isVisible('ridge') && (
          <List.Item>
            <Text span {...labelStyle('ridge')}>
              Ridge classifier
            </Text>{' '}
            – linear classifier with L2 regularisation. Often a strong, stable baseline
            when you want something fast and robust.
          </List.Item>
        )}

        {isVisible('sgd') && (
          <List.Item>
            <Text span {...labelStyle('sgd')}>
              SGD classifier
            </Text>{' '}
            – fast linear model trained with stochastic gradient descent. Useful for
            larger datasets and hyperparameter sweeps.
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

        {isVisible('extratrees') && (
          <List.Item>
            <Text span {...labelStyle('extratrees')}>
              Extra Trees
            </Text>{' '}
            – like random forest, but with extra randomness in splits. Often very strong
            on tabular numeric data.
          </List.Item>
        )}

        {isVisible('hgb') && (
          <List.Item>
            <Text span {...labelStyle('hgb')}>
              HistGradientBoosting
            </Text>{' '}
            – modern boosting model for tabular data. Captures non-linear patterns and
            interactions with strong default performance.
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

        {isVisible('gnb') && (
          <List.Item>
            <Text span {...labelStyle('gnb')}>
              Gaussian Naive Bayes
            </Text>{' '}
            – simple probabilistic classifier that assumes features are (approximately)
            normally distributed within each class. Very fast and can work surprisingly well
            on some lab-style numeric data.
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

        {isVisible('ridgereg') && (
          <List.Item>
            <Text span {...labelStyle('ridgereg')}>
              Ridge regression
            </Text>{' '}
            – linear regression with L2 regularisation. A stable, strong default for continuous targets.
          </List.Item>
        )}

        {isVisible('ridgecv') && (
          <List.Item>
            <Text span {...labelStyle('ridgecv')}>
              Ridge regression (CV)
            </Text>{' '}
            – ridge regression that selects the regularisation strength using internal cross-validation.
          </List.Item>
        )}

        {isVisible('lasso') && (
          <List.Item>
            <Text span {...labelStyle('lasso')}>
              Lasso
            </Text>{' '}
            – linear regression with L1 regularisation. Encourages sparse (feature-selecting) solutions.
          </List.Item>
        )}

        {isVisible('lassocv') && (
          <List.Item>
            <Text span {...labelStyle('lassocv')}>
              Lasso (CV)
            </Text>{' '}
            – lasso with internal cross-validation to pick the regularisation strength.
          </List.Item>
        )}

        {isVisible('enet') && (
          <List.Item>
            <Text span {...labelStyle('enet')}>
              Elastic Net
            </Text>{' '}
            – combines L1 and L2 regularisation. Useful with correlated features and when you want controlled sparsity.
          </List.Item>
        )}

        {isVisible('enetcv') && (
          <List.Item>
            <Text span {...labelStyle('enetcv')}>
              Elastic Net (CV)
            </Text>{' '}
            – elastic net with internal cross-validation over regularisation settings.
          </List.Item>
        )}

        {isVisible('bayridge') && (
          <List.Item>
            <Text span {...labelStyle('bayridge')}>
              Bayesian Ridge
            </Text>{' '}
            – probabilistic linear regression with automatic regularisation. Often robust with limited data.
          </List.Item>
        )}

        {isVisible('svr') && (
          <List.Item>
            <Text span {...labelStyle('svr')}>
              Support Vector Regression (SVR)
            </Text>{' '}
            – kernel-based regressor that can model non-linear relationships. Typically needs feature scaling.
          </List.Item>
        )}

        {isVisible('linsvr') && (
          <List.Item>
            <Text span {...labelStyle('linsvr')}>
              Linear SVR
            </Text>{' '}
            – faster linear variant of SVR. Useful on larger datasets than kernel SVR.
          </List.Item>
        )}

        {isVisible('knnreg') && (
          <List.Item>
            <Text span {...labelStyle('knnreg')}>
              kNN regressor
            </Text>{' '}
            – predicts by averaging nearby samples. Simple baseline, sensitive to scaling and dataset size.
          </List.Item>
        )}

        {isVisible('treereg') && (
          <List.Item>
            <Text span {...labelStyle('treereg')}>
              Decision tree regressor
            </Text>{' '}
            – threshold-based regression tree. Easy to interpret but can overfit without constraints.
          </List.Item>
        )}

        {isVisible('rfreg') && (
          <List.Item>
            <Text span {...labelStyle('rfreg')}>
              Random forest regressor
            </Text>{' '}
            – ensemble of regression trees. Strong default for tabular regression problems.
          </List.Item>
        )}

        {isVisible('kmeans') && (
          <List.Item>
            <Text span {...labelStyle('kmeans')}>
              K-Means
            </Text>{' '}
            – partitions data into a fixed number of clusters by minimising within-cluster distance. Fast, strong baseline when clusters are roughly spherical.
          </List.Item>
        )}

        {isVisible('dbscan') && (
          <List.Item>
            <Text span {...labelStyle('dbscan')}>
              DBSCAN
            </Text>{' '}
            – density-based clustering that can find arbitrary-shaped clusters and label outliers as noise. Does not require choosing the number of clusters.
          </List.Item>
        )}

        {isVisible('spectral') && (
          <List.Item>
            <Text span {...labelStyle('spectral')}>
              Spectral clustering
            </Text>{' '}
            – graph-based clustering that can capture complex cluster shapes. Often slower than K-Means on large datasets.
          </List.Item>
        )}

        {isVisible('agglo') && (
          <List.Item>
            <Text span {...labelStyle('agglo')}>
              Agglomerative clustering
            </Text>{' '}
            – hierarchical clustering that merges samples step-by-step. Useful when you want linkage-based structure.
          </List.Item>
        )}

        {isVisible('gmm') && (
          <List.Item>
            <Text span {...labelStyle('gmm')}>
              Gaussian mixture model
            </Text>{' '}
            – probabilistic clustering that models each cluster as a Gaussian component. Can provide soft assignments.
          </List.Item>
        )}

        {isVisible('bgmm') && (
          <List.Item>
            <Text span {...labelStyle('bgmm')}>
              Bayesian Gaussian mixture
            </Text>{' '}
            – mixture model with Bayesian regularisation that can shrink unused components. Often more stable with limited data.
          </List.Item>
        )}

        {isVisible('meanshift') && (
          <List.Item>
            <Text span {...labelStyle('meanshift')}>
              MeanShift
            </Text>{' '}
            – mode-seeking clustering that can discover the number of clusters based on bandwidth. Can be expensive on large datasets.
          </List.Item>
        )}

        {isVisible('birch') && (
          <List.Item>
            <Text span {...labelStyle('birch')}>
              Birch
            </Text>{' '}
            – scalable clustering based on a compact tree representation. Useful for larger datasets.
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
              elastic net (0 = pure L2, 1 = pure L1). Ignored for other penalties.
            </List.Item>
          </List>
        </Stack>
      );

    case 'ridge':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Ridge classifier parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Alpha</Text>{' '}
              – regularisation strength. Larger values shrink coefficients more and
              can improve generalisation, but may underfit.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Fit intercept</Text>{' '}
              – include a bias/intercept term. Disable only if data is already centred.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – balances class importance for imbalanced datasets.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Solver</Text>{' '}
              – method used to fit the model. "auto" chooses a reasonable default.
              Iterative solvers can be faster on large datasets; direct solvers are often
              more accurate for smaller problems.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max iterations</Text>{' '}
              – only used by iterative solvers. Increase if you see convergence warnings.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Tolerance</Text>{' '}
              – stopping threshold for iterative solvers. Smaller values can yield a more
              accurate solution but may take longer.
            </List.Item>
          </List>
        </Stack>
      );

    case 'sgd':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            SGD classifier parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Loss</Text>{' '}
              – the objective (e.g. hinge for linear SVM, log_loss for logistic-style probabilities).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Penalty</Text>{' '}
              – regularisation type (L2/L1/elasticnet). Helps prevent overfitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Alpha</Text>{' '}
              – regularisation strength. Larger values make the model more conservative.
            </List.Item>
            <List.Item>
              <Text span fw={600}>L1 ratio</Text>{' '}
              – only used for elasticnet penalty. Higher values favour L1 (sparser coefficients);
              lower values favour L2 (more stable coefficients).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Fit intercept</Text>{' '}
              – include a bias term. Disable only if your features are already centred.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Learning rate / eta0</Text>{' '}
              – controls the step size schedule.
              <Text span fw={600}> Learning rate</Text> chooses how the step size changes over time;
              <Text span fw={600}> eta0</Text> is the initial step size for some schedules.
              Too high can diverge; too low can train slowly.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Power t</Text>{' '}
              – used for the "invscaling" learning-rate schedule. Larger values decrease the
              step size faster.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max iterations / tolerance</Text>{' '}
              – training stops when improvement stalls.
              <Text span fw={600}> Max iterations</Text> is the maximum number of passes;
              <Text span fw={600}> tolerance</Text> sets how small the improvement must be before stopping.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Shuffle</Text>{' '}
              – shuffles training data each epoch. Usually improves convergence; disable for reproducibility
              experiments.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Verbose</Text>{' '}
              – prints training progress. Useful for debugging but can be noisy.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Epsilon</Text>{' '}
              – only relevant for some loss functions (e.g. Huber / epsilon-insensitive). Controls the width
              of the “no-penalty” region around the margin.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Jobs (n_jobs)</Text>{' '}
              – number of CPU cores used where supported.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Early stopping</Text>{' '}
              – uses a validation split to stop automatically if performance stops improving.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Validation fraction</Text>{' '}
              – fraction of training data held out for early stopping.
            </List.Item>
            <List.Item>
              <Text span fw={600}>n_iter_no_change</Text>{' '}
              – how many epochs with no improvement are allowed before stopping (when early stopping is on).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – balances class importance for imbalanced datasets.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Warm start</Text>{' '}
              – continues training from the previous solution when refitting. Useful for iterative workflows,
              but can be confusing if you expect a fresh fit.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Average</Text>{' '}
              – averaged SGD weights can reduce variance and improve stability.
              Can be <Text span fw={600}>true</Text> (start averaging immediately) or an integer (start averaging
              after that many updates).
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

    case 'kmeans':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            K-Means parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>n_clusters</Text>{' '}
              – number of clusters to form.
            </List.Item>
            <List.Item>
              <Text span fw={600}>init</Text>{' '}
              – initialisation method (k-means++ is a strong default).
            </List.Item>
            <List.Item>
              <Text span fw={600}>n_init</Text>{' '}
              – number of initialisations. More initialisations can improve stability.
            </List.Item>
            <List.Item>
              <Text span fw={600}>max_iter</Text>{' '}
              – maximum iterations per initialisation.
            </List.Item>
          </List>
        </Stack>
      );

    case 'dbscan':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            DBSCAN parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>eps</Text>{' '}
              – neighbourhood radius. Most important parameter.
            </List.Item>
            <List.Item>
              <Text span fw={600}>min_samples</Text>{' '}
              – minimum neighbours needed to be a core point. Higher values make clustering more conservative.
            </List.Item>
            <List.Item>
              <Text span fw={600}>metric</Text>{' '}
              – distance metric.
            </List.Item>
          </List>
        </Stack>
      );

    case 'spectral':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Spectral clustering parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>n_clusters</Text>{' '}
              – number of clusters.
            </List.Item>
            <List.Item>
              <Text span fw={600}>affinity</Text>{' '}
              – how to build the similarity graph (rbf, nearest_neighbors, ...).
            </List.Item>
            <List.Item>
              <Text span fw={600}>gamma</Text>{' '}
              – used for rbf affinity; controls how local the similarity is.
            </List.Item>
          </List>
        </Stack>
      );

    case 'agglo':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Agglomerative clustering parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>n_clusters</Text>{' '}
              – number of clusters.
            </List.Item>
            <List.Item>
              <Text span fw={600}>linkage</Text>{' '}
              – how clusters are merged (ward, complete, average, single).
            </List.Item>
            <List.Item>
              <Text span fw={600}>metric</Text>{' '}
              – distance metric (for non-ward linkages).
            </List.Item>
          </List>
        </Stack>
      );

    case 'gmm':
    case 'bgmm':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            {selectedAlgo === 'bgmm' ? 'Bayesian Gaussian Mixture' : 'Gaussian Mixture'} parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>n_components</Text>{' '}
              – maximum number of mixture components.
            </List.Item>
            <List.Item>
              <Text span fw={600}>covariance_type</Text>{' '}
              – shape of covariance (full, diag, tied, spherical).
            </List.Item>
            <List.Item>
              <Text span fw={600}>reg_covar</Text>{' '}
              – adds a small value to the diagonal for numerical stability.
            </List.Item>
            <List.Item>
              <Text span fw={600}>max_iter</Text>{' '}
              – EM iterations.
            </List.Item>
          </List>
        </Stack>
      );

    case 'meanshift':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            MeanShift parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>bandwidth</Text>{' '}
              – kernel bandwidth. If not set, it can be estimated but may be slow.
            </List.Item>
            <List.Item>
              <Text span fw={600}>bin_seeding</Text>{' '}
              – speed optimisation that seeds from binned points.
            </List.Item>
          </List>
        </Stack>
      );

    case 'birch':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Birch parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>threshold</Text>{' '}
              – radius threshold for creating new subclusters.
            </List.Item>
            <List.Item>
              <Text span fw={600}>branching_factor</Text>{' '}
              – maximum number of subclusters in each node.
            </List.Item>
            <List.Item>
              <Text span fw={600}>n_clusters</Text>{' '}
              – optional final clustering step (e.g. an integer number of clusters).
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
              <Text span fw={600}>Min weight fraction leaf</Text>{' '}
              – minimum weighted fraction of the total sample weight required at a leaf.
              Mostly relevant when using sample weights; larger values make the tree more conservative.
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
              <Text span fw={600}>Criterion</Text>{' '}
              – measure used to evaluate split quality (e.g. gini/entropy/log_loss).
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
              <Text span fw={600}>Min weight fraction leaf</Text>{' '}
              – minimum weighted fraction of total sample weight at a leaf (mostly relevant with sample weights).
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
              <Text span fw={600}>Min impurity decrease</Text>{' '}
              – minimum impurity improvement required to split. Larger values prevent weak/noisy splits.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Bootstrap</Text>{' '}
              – whether each tree is trained on a random sample with
              replacement. Improves robustness.
            </List.Item>
            <List.Item>
              <Text span fw={600}>OOB score</Text>{' '}
              – estimates generalisation error using unused samples during
              training (requires bootstrap=true).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Jobs (n_jobs)</Text>{' '}
              – number of CPU cores used for training/prediction.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Random state</Text>{' '}
              – seed for reproducibility.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Warm start</Text>{' '}
              – reuses the existing fitted forest and adds more trees when refitting.
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
              increase randomness (only used when bootstrap=true).
            </List.Item>
          </List>
        </Stack>
      );

    case 'extratrees':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Extra Trees parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Trees (n_estimators)</Text>{' '}
              – number of trees. More trees improve stability but increase training time.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Criterion</Text>{' '}
              – measure used to evaluate split quality (e.g. gini/entropy/log_loss).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max depth</Text>{' '}
              – limits depth of each tree. Smaller values reduce overfitting but may underfit.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples split / min samples leaf</Text>{' '}
              – minimum samples required to split a node / to form a leaf. Larger values make trees more conservative.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min weight fraction leaf</Text>{' '}
              – minimum weighted fraction of total sample weight at a leaf (mostly relevant with sample weights).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max features</Text>{' '}
              – number of features tried at each split. Smaller values increase randomness/diversity.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max leaf nodes</Text>{' '}
              – caps tree complexity. Larger values allow finer partitions but increase variance.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min impurity decrease</Text>{' '}
              – minimum impurity improvement required to split. Larger values prevent weak/noisy splits.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Bootstrap</Text>{' '}
              – optional sampling with replacement (default is usually off for Extra Trees).
            </List.Item>
            <List.Item>
              <Text span fw={600}>OOB score</Text>{' '}
              – out-of-bag score estimate using unused samples (requires bootstrap=true).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Jobs (n_jobs)</Text>{' '}
              – number of CPU cores used for training/prediction.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Random state</Text>{' '}
              – seed for reproducibility.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Warm start</Text>{' '}
              – reuses the existing fitted ensemble and adds more trees when refitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – balances class importance for imbalanced datasets.
            </List.Item>
            <List.Item>
              <Text span fw={600}>CCP alpha</Text>{' '}
              – pruning strength applied to each tree.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max samples</Text>{' '}
              – number or fraction of samples used per tree (only used when bootstrap=true).
            </List.Item>
          </List>
        </Stack>
      );

    case 'hgb':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            HistGradientBoosting parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Loss</Text>{' '}
              – objective function. For classification this is typically log_loss (cross-entropy).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Learning rate</Text>{' '}
              – step size of boosting. Smaller values often generalise better but need more iterations.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Iterations (max_iter)</Text>{' '}
              – number of boosting stages. More stages can improve fit but may overfit.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max leaf nodes / max depth</Text>{' '}
              – controls complexity of each tree. Larger values capture interactions but increase overfitting risk.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples leaf</Text>{' '}
              – minimum samples per leaf. Larger values smooth the model and reduce overfitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max features (fraction)</Text>{' '}
              – fraction of features used per split (0–1]. Smaller values add randomness.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max bins</Text>{' '}
              – number of discrete bins used when histogram-binning continuous features.
              More bins can capture finer detail but increase memory/time.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Early stopping</Text>{' '}
              – stops training when the validation score stops improving.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Scoring</Text>{' '}
              – metric used for early stopping / validation monitoring. "loss" means stop when validation loss stops improving.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Validation fraction</Text>{' '}
              – fraction of training data held out for early stopping.
            </List.Item>
            <List.Item>
              <Text span fw={600}>No-change rounds</Text>{' '}
              – how many iterations without improvement are allowed before stopping.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Tolerance</Text>{' '}
              – minimum improvement considered “progress” for early stopping. Smaller values make stopping less sensitive.
            </List.Item>
            <List.Item>
              <Text span fw={600}>L2 regularisation</Text>{' '}
              – shrinks leaf values to reduce overfitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Verbose</Text>{' '}
              – prints training progress.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Random state</Text>{' '}
              – seed for reproducibility.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Class weight</Text>{' '}
              – balances class importance for imbalanced datasets.
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

    case 'gnb':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Gaussian Naive Bayes parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Variance smoothing</Text>{' '}
              – adds a small value to variances for numerical stability. Increase if you see numerical issues.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Priors</Text>{' '}
              – optional class prior probabilities. Leave empty to estimate from the training data.
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



    case 'ridgereg':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Ridge regression parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Alpha</Text>{' '}
              – L2 regularisation strength. Larger values shrink coefficients more.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Solver</Text>{' '}
              – numerical method used to solve the ridge system.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Positive</Text>{' '}
              – constrains coefficients to be non-negative.
            </List.Item>
          </List>
        </Stack>
      );

    case 'ridgecv':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Ridge regression (CV) parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Alphas</Text>{' '}
              – list of candidate regularisation strengths tested internally.
            </List.Item>
            <List.Item>
              <Text span fw={600}>CV folds</Text>{' '}
              – number of folds used for internal validation (if set).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Scoring</Text>{' '}
              – metric used to pick the best alpha (optional).
            </List.Item>
          </List>
        </Stack>
      );

    case 'enet':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Elastic Net parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Alpha</Text>{' '}
              – overall regularisation strength.
            </List.Item>
            <List.Item>
              <Text span fw={600}>L1 ratio</Text>{' '}
              – mix between L1 and L2 (0 = ridge-like, 1 = lasso-like).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Selection</Text>{' '}
              – coordinate descent update strategy (cyclic or random).
            </List.Item>
          </List>
        </Stack>
      );

    case 'enetcv':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Elastic Net (CV) parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>L1 ratio list</Text>{' '}
              – candidate L1 ratios tested internally.
            </List.Item>
            <List.Item>
              <Text span fw={600}>n_alphas</Text>{' '}
              – number of alphas along the regularisation path.
            </List.Item>
            <List.Item>
              <Text span fw={600}>CV folds</Text>{' '}
              – number of folds used for internal validation.
            </List.Item>
          </List>
        </Stack>
      );

    case 'lasso':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Lasso parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Alpha</Text>{' '}
              – L1 regularisation strength. Larger values yield sparser solutions.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Selection</Text>{' '}
              – coordinate descent update strategy (cyclic or random).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Positive</Text>{' '}
              – constrains coefficients to be non-negative.
            </List.Item>
          </List>
        </Stack>
      );

    case 'lassocv':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Lasso (CV) parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>n_alphas</Text>{' '}
              – number of alphas tested along the regularisation path.
            </List.Item>
            <List.Item>
              <Text span fw={600}>CV folds</Text>{' '}
              – number of folds used for internal validation.
            </List.Item>
            <List.Item>
              <Text span fw={600}>eps</Text>{' '}
              – controls the range of alphas searched.
            </List.Item>
          </List>
        </Stack>
      );

    case 'bayridge':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Bayesian Ridge parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>n_iter</Text>{' '}
              – maximum number of update iterations.
            </List.Item>
            <List.Item>
              <Text span fw={600}>alpha_1 / alpha_2</Text>{' '}
              – hyperpriors for the noise precision.
            </List.Item>
            <List.Item>
              <Text span fw={600}>lambda_1 / lambda_2</Text>{' '}
              – hyperpriors for the weights precision.
            </List.Item>
          </List>
        </Stack>
      );

    case 'svr':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            SVR parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Kernel</Text>{' '}
              – shape of the regression function (RBF, polynomial, etc.).
            </List.Item>
            <List.Item>
              <Text span fw={600}>C</Text>{' '}
              – penalty for errors. Larger values fit the training data more closely.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Epsilon</Text>{' '}
              – width of the insensitive zone around the target values.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Gamma</Text>{' '}
              – controls influence of individual samples (RBF/poly/sigmoid).
            </List.Item>
          </List>
        </Stack>
      );

    case 'linsvr':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Linear SVR parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>C</Text>{' '}
              – regularisation strength (inverse). Larger values fit the training data more closely.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Loss</Text>{' '}
              – epsilon-insensitive or squared epsilon-insensitive loss.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Dual</Text>{' '}
              – whether to solve the dual optimisation problem.
            </List.Item>
          </List>
        </Stack>
      );

    case 'knnreg':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            kNN regressor parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Neighbours</Text>{' '}
              – number of nearest samples used to compute the prediction.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Weights</Text>{' '}
              – uniform or distance-weighted averaging of neighbours.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Metric</Text>{' '}
              – distance function used to compare samples.
            </List.Item>
          </List>
        </Stack>
      );

    case 'treereg':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Decision tree regressor parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>Criterion</Text>{' '}
              – how split quality is measured (e.g. squared error).
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max depth</Text>{' '}
              – limits tree depth to reduce overfitting.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Min samples leaf</Text>{' '}
              – minimum samples required in a leaf node.
            </List.Item>
          </List>
        </Stack>
      );

    case 'rfreg':
      return (
        <Stack gap={4}>
          <Text fw={500} size="sm">
            Random forest regressor parameters
          </Text>
          <List spacing={4} size="xs">
            <List.Item>
              <Text span fw={600}>n_estimators</Text>{' '}
              – number of trees in the forest.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Max features</Text>{' '}
              – how many features are considered at each split.
            </List.Item>
            <List.Item>
              <Text span fw={600}>Bootstrap / OOB</Text>{' '}
              – sampling with replacement and optional out-of-bag scoring.
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
