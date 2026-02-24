// Small guards to keep Results panels readable and consistent.

export function getTask(trainResult) {
  return trainResult?.task || trainResult?.artifact?.kind || null;
}

export function isUnsupervisedTask(task) {
  return task === 'unsupervised';
}

export function isCVResult(trainResult) {
  return !!trainResult && Array.isArray(trainResult.fold_scores);
}

export function hasConfusionMatrix(trainResult) {
  const confusion = trainResult?.confusion;
  return (
    !!confusion &&
    Array.isArray(confusion.matrix) &&
    confusion.matrix.length > 0 &&
    Array.isArray(confusion.labels) &&
    confusion.labels.length > 0
  );
}

export function isClassificationResult(trainResult) {
  const task = getTask(trainResult);
  return task === 'classification' || hasConfusionMatrix(trainResult);
}

export function isRegressionResult(trainResult) {
  const task = getTask(trainResult);
  return task === 'regression' || (trainResult?.regression && typeof trainResult.regression === 'object');
}
