import { Stack, Text, List } from '@mantine/core';

export const REGRESSION_PARAM_BLOCKS = {
  linreg: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Linear regression parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Fit intercept
          </Text>{' '}
          – whether to include an intercept term. Disable only if data is
          already centred.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Copy X
          </Text>{' '}
          – copies the input matrix before fitting to avoid modifying it.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Jobs (n_jobs)
          </Text>{' '}
          – number of CPU cores used during fitting, if supported.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Positive coefficients
          </Text>{' '}
          – constrains coefficients to be non-negative, useful for some
          physical or interpretability constraints.
        </List.Item>
      </List>
    </Stack>
  ),

  ridgereg: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Ridge regression parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Alpha
          </Text>{' '}
          – L2 regularisation strength. Larger values shrink coefficients more.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Solver
          </Text>{' '}
          – numerical method used to solve the ridge system.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Positive
          </Text>{' '}
          – constrains coefficients to be non-negative.
        </List.Item>
      </List>
    </Stack>
  ),

  ridgecv: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Ridge regression (CV) parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Alphas
          </Text>{' '}
          – list of candidate regularisation strengths tested internally.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            CV folds
          </Text>{' '}
          – number of folds used for internal validation (if set).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Scoring
          </Text>{' '}
          – metric used to pick the best alpha (optional).
        </List.Item>
      </List>
    </Stack>
  ),

  enet: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Elastic Net parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Alpha
          </Text>{' '}
          – overall regularisation strength.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            L1 ratio
          </Text>{' '}
          – mix between L1 and L2 (0 = ridge-like, 1 = lasso-like).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Selection
          </Text>{' '}
          – coordinate descent update strategy (cyclic or random).
        </List.Item>
      </List>
    </Stack>
  ),

  enetcv: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Elastic Net (CV) parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            L1 ratio list
          </Text>{' '}
          – candidate L1 ratios tested internally.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            n_alphas
          </Text>{' '}
          – number of alphas along the regularisation path.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            CV folds
          </Text>{' '}
          – number of folds used for internal validation.
        </List.Item>
      </List>
    </Stack>
  ),

  lasso: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Lasso parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Alpha
          </Text>{' '}
          – L1 regularisation strength. Larger values yield sparser solutions.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Selection
          </Text>{' '}
          – coordinate descent update strategy (cyclic or random).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Positive
          </Text>{' '}
          – constrains coefficients to be non-negative.
        </List.Item>
      </List>
    </Stack>
  ),

  lassocv: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Lasso (CV) parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            n_alphas
          </Text>{' '}
          – number of alphas tested along the regularisation path.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            CV folds
          </Text>{' '}
          – number of folds used for internal validation.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            eps
          </Text>{' '}
          – controls the range of alphas searched.
        </List.Item>
      </List>
    </Stack>
  ),

  bayridge: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Bayesian Ridge parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            n_iter
          </Text>{' '}
          – maximum number of update iterations.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            alpha_1 / alpha_2
          </Text>{' '}
          – hyperpriors for the noise precision.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            lambda_1 / lambda_2
          </Text>{' '}
          – hyperpriors for the weights precision.
        </List.Item>
      </List>
    </Stack>
  ),

  svr: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        SVR parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Kernel
          </Text>{' '}
          – shape of the regression function (RBF, polynomial, etc.).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            C
          </Text>{' '}
          – penalty for errors. Larger values fit the training data more
          closely.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Epsilon
          </Text>{' '}
          – width of the insensitive zone around the target values.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Gamma
          </Text>{' '}
          – controls influence of individual samples (RBF/poly/sigmoid).
        </List.Item>
      </List>
    </Stack>
  ),

  linsvr: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Linear SVR parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            C
          </Text>{' '}
          – regularisation strength (inverse). Larger values fit the training
          data more closely.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Loss
          </Text>{' '}
          – epsilon-insensitive or squared epsilon-insensitive loss.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Dual
          </Text>{' '}
          – whether to solve the dual optimisation problem.
        </List.Item>
      </List>
    </Stack>
  ),

  knnreg: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        kNN regressor parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Neighbours
          </Text>{' '}
          – number of nearest samples used to compute the prediction.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Weights
          </Text>{' '}
          – uniform or distance-weighted averaging of neighbours.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Metric
          </Text>{' '}
          – distance function used to compare samples.
        </List.Item>
      </List>
    </Stack>
  ),

  treereg: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Decision tree regressor parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Criterion
          </Text>{' '}
          – how split quality is measured (e.g. squared error).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Max depth
          </Text>{' '}
          – limits tree depth to reduce overfitting.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Min samples leaf
          </Text>{' '}
          – minimum samples required in a leaf node.
        </List.Item>
      </List>
    </Stack>
  ),

  rfreg: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Random forest regressor parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            n_estimators
          </Text>{' '}
          – number of trees in the forest.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Max features
          </Text>{' '}
          – how many features are considered at each split.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Bootstrap / OOB
          </Text>{' '}
          – sampling with replacement and optional out-of-bag scoring.
        </List.Item>
      </List>
    </Stack>
  ),
};
