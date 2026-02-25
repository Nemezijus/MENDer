import { CLASSIFICATION_PARAM_BLOCKS } from './classificationParams.jsx';
import { REGRESSION_PARAM_BLOCKS } from './regressionParams.jsx';
import { UNSUPERVISED_PARAM_BLOCKS } from './unsupervisedParams.jsx';

export const MODEL_PARAM_BLOCKS = {
  ...CLASSIFICATION_PARAM_BLOCKS,
  ...REGRESSION_PARAM_BLOCKS,
  ...UNSUPERVISED_PARAM_BLOCKS,
};
