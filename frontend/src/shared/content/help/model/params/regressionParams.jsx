import { Stack, Text, List } from "@mantine/core";
import "../../../styles/help.css";
export const REGRESSION_PARAM_BLOCKS = {
  linreg: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Linear regression parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Fit intercept{" "}
          </Text>{" "}
          – whether to include an intercept term. Disable only if data is
          already centred.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Copy X{" "}
          </Text>{" "}
          – copies the input matrix before fitting to avoid modifying it.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Jobs (n_jobs){" "}
          </Text>{" "}
          – number of CPU cores used during fitting, if supported.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Positive coefficients{" "}
          </Text>{" "}
          – constrains coefficients to be non-negative, useful for some physical
          or interpretability constraints.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  ridgereg: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Ridge regression parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Alpha{" "}
          </Text>{" "}
          – L2 regularisation strength. Larger values shrink coefficients
          more.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Solver{" "}
          </Text>{" "}
          – numerical method used to solve the ridge system.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Positive{" "}
          </Text>{" "}
          – constrains coefficients to be non-negative.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  ridgecv: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Ridge regression (CV) parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Alphas{" "}
          </Text>{" "}
          – list of candidate regularisation strengths tested internally.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            CV folds{" "}
          </Text>{" "}
          – number of folds used for internal validation (if set).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Scoring{" "}
          </Text>{" "}
          – metric used to pick the best alpha (optional).{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  enet: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Elastic Net parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Alpha{" "}
          </Text>{" "}
          – overall regularisation strength.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            L1 ratio{" "}
          </Text>{" "}
          – mix between L1 and L2 (0 = ridge-like, 1 = lasso-like).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Selection{" "}
          </Text>{" "}
          – coordinate descent update strategy (cyclic or random).{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  enetcv: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Elastic Net (CV) parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            L1 ratio list{" "}
          </Text>{" "}
          – candidate L1 ratios tested internally.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            n_alphas{" "}
          </Text>{" "}
          – number of alphas along the regularisation path.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            CV folds{" "}
          </Text>{" "}
          – number of folds used for internal validation.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  lasso: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Lasso parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Alpha{" "}
          </Text>{" "}
          – L1 regularisation strength. Larger values yield sparser
          solutions.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Selection{" "}
          </Text>{" "}
          – coordinate descent update strategy (cyclic or random).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Positive{" "}
          </Text>{" "}
          – constrains coefficients to be non-negative.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  lassocv: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Lasso (CV) parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            n_alphas{" "}
          </Text>{" "}
          – number of alphas tested along the regularisation path.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            CV folds{" "}
          </Text>{" "}
          – number of folds used for internal validation.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            eps{" "}
          </Text>{" "}
          – controls the range of alphas searched.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  bayridge: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Bayesian Ridge parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            n_iter{" "}
          </Text>{" "}
          – maximum number of update iterations.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            alpha_1 / alpha_2{" "}
          </Text>{" "}
          – hyperpriors for the noise precision.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            lambda_1 / lambda_2{" "}
          </Text>{" "}
          – hyperpriors for the weights precision.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  svr: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> SVR parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Kernel{" "}
          </Text>{" "}
          – shape of the regression function (RBF, polynomial, etc.).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            C{" "}
          </Text>{" "}
          – penalty for errors. Larger values fit the training data more
          closely.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Epsilon{" "}
          </Text>{" "}
          – width of the insensitive zone around the target values.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Gamma{" "}
          </Text>{" "}
          – controls influence of individual samples (RBF/poly/sigmoid).{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  linsvr: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Linear SVR parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            C{" "}
          </Text>{" "}
          – regularisation strength (inverse). Larger values fit the training
          data more closely.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Loss{" "}
          </Text>{" "}
          – epsilon-insensitive or squared epsilon-insensitive loss.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Dual{" "}
          </Text>{" "}
          – whether to solve the dual optimisation problem.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  knnreg: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> kNN regressor parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Neighbours{" "}
          </Text>{" "}
          – number of nearest samples used to compute the prediction.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Weights{" "}
          </Text>{" "}
          – uniform or distance-weighted averaging of neighbours.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Metric{" "}
          </Text>{" "}
          – distance function used to compare samples.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  treereg: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle">
        {" "}
        Decision tree regressor parameters{" "}
      </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Criterion{" "}
          </Text>{" "}
          – how split quality is measured (e.g. squared error).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max depth{" "}
          </Text>{" "}
          – limits tree depth to reduce overfitting.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min samples leaf{" "}
          </Text>{" "}
          – minimum samples required in a leaf node.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  rfreg: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle">
        {" "}
        Random forest regressor parameters{" "}
      </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            n_estimators{" "}
          </Text>{" "}
          – number of trees in the forest.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max features{" "}
          </Text>{" "}
          – how many features are considered at each split.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Bootstrap / OOB{" "}
          </Text>{" "}
          – sampling with replacement and optional out-of-bag scoring.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
};
