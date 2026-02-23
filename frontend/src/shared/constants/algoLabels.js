/**
 * Shared algorithm naming.
 *
 * Keys must match backend "algo" identifiers.
 * This file is a landing zone so we can remove duplicated maps scattered
 * across panels.
 */

export const ALGO_LABELS = {
  // -------- classifiers --------
  logreg: 'Logistic Regression',
  ridge: 'Ridge Classifier',
  sgd: 'SGD Classifier',
  svm: 'Support Vector Machine (SVC)',
  tree: 'Decision Tree',
  forest: 'Random Forest',
  extratrees: 'Extra Trees Classifier',
  hgb: 'Histogram Gradient Boosting',
  knn: 'k-Nearest Neighbors',
  gnb: 'Gaussian Naive Bayes',

  // -------- regressors --------
  linreg: 'Linear Regression',
  ridgereg: 'Ridge Regression',
  ridgecv: 'Ridge Regression (CV)',
  enet: 'Elastic Net',
  enetcv: 'Elastic Net (CV)',
  lasso: 'Lasso',
  lassocv: 'Lasso (CV)',
  bayridge: 'Bayesian Ridge',
  svr: 'Support Vector Regression (SVR)',
  linsvr: 'Linear SVR',
  knnreg: 'k-Nearest Neighbors Regressor',
  treereg: 'Decision Tree Regressor',
  rfreg: 'Random Forest Regressor',

  // -------- unsupervised --------
  kmeans: 'K-Means',
  dbscan: 'DBSCAN',
  spectral: 'Spectral Clustering',
  agglo: 'Agglomerative Clustering',
  gmm: 'Gaussian Mixture',
  bgmm: 'Bayesian Gaussian Mixture',
  meanshift: 'MeanShift',
  birch: 'Birch',

  // -------- special --------
  xgboost: 'XGBoost',
};

export const ALGO_ABBREV = {
  logreg: 'LR',
  ridge: 'Ridge',
  sgd: 'SGD',
  svm: 'SVM',
  forest: 'RF',
  extratrees: 'ET',
  hgb: 'HGB',
  knn: 'kNN',
  gnb: 'GNB',
  linreg: 'LinReg',
  ridgereg: 'Ridge',
  ridgecv: 'RidgeCV',
  enet: 'EN',
  enetcv: 'ENCV',
  lassocv: 'LassoCV',
  bayridge: 'BayRidge',
  svr: 'SVR',
  linsvr: 'LinSVR',
  rfreg: 'RF',
  xgboost: 'XGB',
};

export function getAlgoLabel(algo) {
  if (!algo) return '';
  return ALGO_LABELS[algo] ?? String(algo);
}

export function getAlgoAbbrev(algo) {
  if (!algo) return '';
  return ALGO_ABBREV[algo] ?? ALGO_LABELS[algo] ?? String(algo);
}
