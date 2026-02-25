export const MODEL_OVERVIEW_ENTRIES = [
  {
    algo: 'logreg',
    label: 'Logistic regression',
    summary:
      'linear classifier that estimates class probabilities. A strong, interpretable baseline when relationships are approximately linear.',
  },
  {
    algo: 'ridge',
    label: 'Ridge classifier',
    summary:
      'linear classifier with L2 regularisation. Often a strong, stable baseline when you want something fast and robust.',
  },
  {
    algo: 'sgd',
    label: 'SGD classifier',
    summary:
      'fast linear model trained with stochastic gradient descent. Useful for larger datasets and hyperparameter sweeps.',
  },
  {
    algo: 'svm',
    label: 'Support Vector Machine (SVM)',
    summary:
      'powerful classifier that can model complex boundaries using kernels. Works well on medium-sized datasets.',
  },
  {
    algo: 'tree',
    label: 'Decision tree',
    summary:
      'rule-based model that splits data by feature thresholds. Easy to interpret but prone to overfitting if unconstrained.',
  },
  {
    algo: 'forest',
    label: 'Random forest',
    summary:
      'ensemble of decision trees that improves robustness and accuracy. Often a strong default choice.',
  },
  {
    algo: 'extratrees',
    label: 'Extra Trees',
    summary:
      'like random forest, but with extra randomness in splits. Often very strong on tabular numeric data.',
  },
  {
    algo: 'hgb',
    label: 'HistGradientBoosting',
    summary:
      'modern boosting model for tabular data. Captures non-linear patterns and interactions with strong default performance.',
  },
  {
    algo: 'knn',
    label: 'k-Nearest Neighbours (kNN)',
    summary:
      'predicts based on similarity to nearby samples. Simple but sensitive to feature scaling and dataset size.',
  },
  {
    algo: 'gnb',
    label: 'Gaussian Naive Bayes',
    summary:
      'simple probabilistic classifier that assumes features are (approximately) normally distributed within each class. Very fast and can work surprisingly well on some lab-style numeric data.',
  },
  {
    algo: 'linreg',
    label: 'Linear regression',
    summary:
      'models a linear relationship between features and target. Useful as a fast, interpretable regression baseline.',
  },
  {
    algo: 'ridgereg',
    label: 'Ridge regression',
    summary:
      'linear regression with L2 regularisation. A stable, strong default for continuous targets.',
  },
  {
    algo: 'ridgecv',
    label: 'Ridge regression (CV)',
    summary:
      'ridge regression that selects the regularisation strength using internal cross-validation.',
  },
  {
    algo: 'lasso',
    label: 'Lasso',
    summary:
      'linear regression with L1 regularisation. Encourages sparse (feature-selecting) solutions.',
  },
  {
    algo: 'lassocv',
    label: 'Lasso (CV)',
    summary:
      'lasso with internal cross-validation to pick the regularisation strength.',
  },
  {
    algo: 'enet',
    label: 'Elastic Net',
    summary:
      'combines L1 and L2 regularisation. Useful with correlated features and when you want controlled sparsity.',
  },
  {
    algo: 'enetcv',
    label: 'Elastic Net (CV)',
    summary:
      'elastic net with internal cross-validation over regularisation settings.',
  },
  {
    algo: 'bayridge',
    label: 'Bayesian Ridge',
    summary:
      'probabilistic linear regression with automatic regularisation. Often robust with limited data.',
  },
  {
    algo: 'svr',
    label: 'Support Vector Regression (SVR)',
    summary:
      'kernel-based regressor that can model non-linear relationships. Typically needs feature scaling.',
  },
  {
    algo: 'linsvr',
    label: 'Linear SVR',
    summary:
      'faster linear variant of SVR. Useful on larger datasets than kernel SVR.',
  },
  {
    algo: 'knnreg',
    label: 'kNN regressor',
    summary:
      'predicts by averaging nearby samples. Simple baseline, sensitive to scaling and dataset size.',
  },
  {
    algo: 'treereg',
    label: 'Decision tree regressor',
    summary:
      'threshold-based regression tree. Easy to interpret but can overfit without constraints.',
  },
  {
    algo: 'rfreg',
    label: 'Random forest regressor',
    summary:
      'ensemble of regression trees. Strong default for tabular regression problems.',
  },
  {
    algo: 'kmeans',
    label: 'K-Means',
    summary:
      'partitions data into a fixed number of clusters by minimising within-cluster distance. Fast, strong baseline when clusters are roughly spherical.',
  },
  {
    algo: 'dbscan',
    label: 'DBSCAN',
    summary:
      'density-based clustering that can find arbitrary-shaped clusters and label outliers as noise. Does not require choosing the number of clusters.',
  },
  {
    algo: 'spectral',
    label: 'Spectral clustering',
    summary:
      'graph-based clustering that can capture complex cluster shapes. Often slower than K-Means on large datasets.',
  },
  {
    algo: 'agglo',
    label: 'Agglomerative clustering',
    summary:
      'hierarchical clustering that merges samples step-by-step. Useful when you want linkage-based structure.',
  },
  {
    algo: 'gmm',
    label: 'Gaussian mixture model',
    summary:
      'probabilistic clustering that models each cluster as a Gaussian component. Can provide soft assignments.',
  },
  {
    algo: 'bgmm',
    label: 'Bayesian Gaussian mixture',
    summary:
      'mixture model with Bayesian regularisation that can shrink unused components. Often more stable with limited data.',
  },
  {
    algo: 'meanshift',
    label: 'MeanShift',
    summary:
      'mode-seeking clustering that can discover the number of clusters based on bandwidth. Can be expensive on large datasets.',
  },
  {
    algo: 'birch',
    label: 'Birch',
    summary:
      'scalable clustering based on a compact tree representation. Useful for larger datasets.',
  },
];
