import { Stack, Text, List } from '@mantine/core';
import '../../../styles/help.css';

export const REGRESSION_PARAM_BLOCKS = {
  linreg: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Linear regression parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Fit intercept
          </Text>{' '}
          – whether to include an intercept term. Disable only if data is
          already centred.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Copy X
          </Text>{' '}
          – copies the input matrix before fitting to avoid modifying it.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Jobs (n_jobs)
          </Text>{' '}
          – number of CPU cores used during fitting, if supported.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Positive coefficients
          </Text>{' '}
          – constrains coefficients to be non-negative, useful for some
          physical or interpretability constraints.
        </List.Item>
      </List>
    </Stack>
  ),

  ridgereg: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Ridge regression parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Alpha
          </Text>{' '}
          – L2 regularisation strength. Larger values shrink coefficients more.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Solver
          </Text>{' '}
          – numerical method used to solve the ridge system.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Positive
          </Text>{' '}
          – constrains coefficients to be non-negative.
        </List.Item>
      </List>
    </Stack>
  ),

  ridgecv: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Ridge regression (CV) parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Alphas
          </Text>{' '}
          – list of candidate regularisation strengths tested internally.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            CV folds
          </Text>{' '}
          – number of folds used for internal validation (if set).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Scoring
          </Text>{' '}
          – metric used to pick the best alpha (optional).
        </List.Item>
      </List>
    </Stack>
  ),

  enet: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Elastic Net parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Alpha
          </Text>{' '}
          – overall regularisation strength.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            L1 ratio
          </Text>{' '}
          – mix between L1 and L2 (0 = ridge-like, 1 = lasso-like).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Selection
          </Text>{' '}
          – coordinate descent update strategy (cyclic or random).
        </List.Item>
      </List>
    </Stack>
  ),

  enetcv: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Elastic Net (CV) parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            L1 ratio list
          </Text>{' '}
          – candidate L1 ratios tested internally.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            n_alphas
          </Text>{' '}
          – number of alphas along the regularisation path.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            CV folds
          </Text>{' '}
          – number of folds used for internal validation.
        </List.Item>
      </List>
    </Stack>
  ),

  lasso: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Lasso parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Alpha
          </Text>{' '}
          – L1 regularisation strength. Larger values yield sparser solutions.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Selection
          </Text>{' '}
          – coordinate descent update strategy (cyclic or random).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Positive
          </Text>{' '}
          – constrains coefficients to be non-negative.
        </List.Item>
      </List>
    </Stack>
  ),

  lassocv: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Lasso (CV) parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            n_alphas
          </Text>{' '}
          – number of alphas tested along the regularisation path.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            CV folds
          </Text>{' '}
          – number of folds used for internal validation.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            eps
          </Text>{' '}
          – controls the range of alphas searched.
        </List.Item>
      </List>
    </Stack>
  ),

  bayridge: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Bayesian Ridge parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            n_iter
          </Text>{' '}
          – maximum number of update iterations.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            alpha_1 / alpha_2
          </Text>{' '}
          – hyperpriors for the noise precision.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            lambda_1 / lambda_2
          </Text>{' '}
          – hyperpriors for the weights precision.
        </List.Item>
      </List>
    </Stack>
  ),

  svr: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        SVR parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Kernel
          </Text>{' '}
          – shape of the regression function (RBF, polynomial, etc.).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            C
          </Text>{' '}
          – penalty for errors. Larger values fit the training data more
          closely.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Epsilon
          </Text>{' '}
          – width of the insensitive zone around the target values.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Gamma
          </Text>{' '}
          – controls influence of individual samples (RBF/poly/sigmoid).
        </List.Item>
      </List>
    </Stack>
  ),

  linsvr: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Linear SVR parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            C
          </Text>{' '}
          – regularisation strength (inverse). Larger values fit the training
          data more closely.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Loss
          </Text>{' '}
          – epsilon-insensitive or squared epsilon-insensitive loss.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Dual
          </Text>{' '}
          – whether to solve the dual optimisation problem.
        </List.Item>
      </List>
    </Stack>
  ),

  knnreg: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        kNN regressor parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Neighbours
          </Text>{' '}
          – number of nearest samples used to compute the prediction.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Weights
          </Text>{' '}
          – uniform or distance-weighted averaging of neighbours.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Metric
          </Text>{' '}
          – distance function used to compare samples.
        </List.Item>
      </List>
    </Stack>
  ),

  treereg: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Decision tree regressor parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Criterion
          </Text>{' '}
          – how split quality is measured (e.g. squared error).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Max depth
          </Text>{' '}
          – limits tree depth to reduce overfitting.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Min samples leaf
          </Text>{' '}
          – minimum samples required in a leaf node.
        </List.Item>
      </List>
    </Stack>
  ),

  rfreg: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Random forest regressor parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            n_estimators
          </Text>{' '}
          – number of trees in the forest.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Max features
          </Text>{' '}
          – how many features are considered at each split.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Bootstrap / OOB
          </Text>{' '}
          – sampling with replacement and optional out-of-bag scoring.
        </List.Item>
      </List>
    </Stack>
  ),
};
