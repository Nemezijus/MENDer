import { Stack, Text, List } from "@mantine/core";
import "../../../styles/help.css";
export const CLASSIFICATION_PARAM_BLOCKS = {
  logreg: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Logistic regression parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            C{" "}
          </Text>{" "}
          – inverse regularisation strength. Smaller values enforce stronger
          regularisation and reduce overfitting.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Penalty{" "}
          </Text>{" "}
          – type of regularisation applied to coefficients (L1, L2, or elastic
          net).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Solver{" "}
          </Text>{" "}
          – optimisation algorithm used to fit the model. Some solvers only
          support certain penalties.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max iterations{" "}
          </Text>{" "}
          – maximum number of optimisation steps before stopping. Increase if
          the model does not converge.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Class weight{" "}
          </Text>{" "}
          – adjusts the importance of classes to compensate for imbalance (e.g.
          balanced).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            L1 ratio{" "}
          </Text>{" "}
          – controls the mix between L1 and L2 regularisation when using elastic
          net (0 = pure L2, 1 = pure L1). Ignored for other penalties.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  ridge: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Ridge classifier parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Alpha{" "}
          </Text>{" "}
          – regularisation strength. Larger values shrink coefficients more and
          can improve generalisation, but may underfit.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Fit intercept{" "}
          </Text>{" "}
          – include a bias/intercept term. Disable only if data is already
          centred.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Class weight{" "}
          </Text>{" "}
          – balances class importance for imbalanced datasets.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Solver{" "}
          </Text>{" "}
          – method used to fit the model."auto" chooses a reasonable default.
          Iterative solvers can be faster on large datasets; direct solvers are
          often more accurate for smaller problems.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max iterations{" "}
          </Text>{" "}
          – only used by iterative solvers. Increase if you see convergence
          warnings.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Tolerance{" "}
          </Text>{" "}
          – stopping threshold for iterative solvers. Smaller values can yield a
          more accurate solution but may take longer.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  sgd: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> SGD classifier parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Loss{" "}
          </Text>{" "}
          – the objective (e.g. hinge for linear SVM, log_loss for
          logistic-style probabilities).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Penalty{" "}
          </Text>{" "}
          – regularisation type (L2/L1/elasticnet). Helps prevent
          overfitting.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Alpha{" "}
          </Text>{" "}
          – regularisation strength. Larger values make the model more
          conservative.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            L1 ratio{" "}
          </Text>{" "}
          – only used for elasticnet penalty. Higher values favour L1 (sparser
          coefficients); lower values favour L2 (more stable coefficients).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Fit intercept{" "}
          </Text>{" "}
          – include a bias term. Disable only if your features are already
          centred.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Learning rate / eta0{" "}
          </Text>{" "}
          – controls the step size schedule.{" "}
          <Text span className="helpLabel">
            {" "}
            Learning rate{" "}
          </Text>{" "}
          chooses how the step size changes over time;{" "}
          <Text span className="helpLabel">
            {" "}
            eta0{" "}
          </Text>{" "}
          is the initial step size for some schedules. Too high can diverge; too
          low can train slowly.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Power t{" "}
          </Text>{" "}
          – used for the"invscaling" learning-rate schedule. Larger values
          decrease the step size faster.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max iterations / tolerance{" "}
          </Text>{" "}
          – training stops when improvement stalls.{" "}
          <Text span className="helpLabel">
            {" "}
            Max iterations{" "}
          </Text>{" "}
          is the maximum number of passes;{" "}
          <Text span className="helpLabel">
            {" "}
            tolerance{" "}
          </Text>{" "}
          sets how small the improvement must be before stopping.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Shuffle{" "}
          </Text>{" "}
          – shuffles training data each epoch. Usually improves convergence;
          disable for reproducibility experiments.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Verbose{" "}
          </Text>{" "}
          – prints training progress. Useful for debugging but can be
          noisy.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Epsilon{" "}
          </Text>{" "}
          – only relevant for some loss functions (e.g. Huber /
          epsilon-insensitive). Controls the width of the “no-penalty” region
          around the margin.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Jobs (n_jobs){" "}
          </Text>{" "}
          – number of CPU cores used where supported.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Early stopping{" "}
          </Text>{" "}
          – uses a validation split to stop automatically if performance stops
          improving.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Validation fraction{" "}
          </Text>{" "}
          – fraction of training data held out for early stopping.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            n_iter_no_change{" "}
          </Text>{" "}
          – how many epochs with no improvement are allowed before stopping
          (when early stopping is on).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Class weight{" "}
          </Text>{" "}
          – balances class importance for imbalanced datasets.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Warm start{" "}
          </Text>{" "}
          – continues training from the previous solution when refitting. Useful
          for iterative workflows, but can be confusing if you expect a fresh
          fit.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Average{" "}
          </Text>{" "}
          – averaged SGD weights can reduce variance and improve stability. Can
          be{" "}
          <Text span className="helpLabel">
            {" "}
            true{" "}
          </Text>{" "}
          (start averaging immediately) or an integer (start averaging after
          that many updates).{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  svm: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> SVM parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Kernel{" "}
          </Text>{" "}
          – defines the shape of the decision boundary. Linear is fastest; RBF
          and polynomial capture non-linear patterns.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            C{" "}
          </Text>{" "}
          – penalty for misclassification. Larger values fit training data more
          closely but may overfit.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Degree{" "}
          </Text>{" "}
          – degree of the polynomial kernel. Higher values increase model
          complexity and training time.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Gamma{" "}
          </Text>{" "}
          – controls how far the influence of a single sample reaches. Larger
          values lead to tighter, more complex boundaries.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Coef0{" "}
          </Text>{" "}
          – constant term used by polynomial and sigmoid kernels. Affects how
          strongly higher-order terms contribute.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Shrinking{" "}
          </Text>{" "}
          – enables a heuristic that can speed up optimisation on large
          datasets.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Probability{" "}
          </Text>{" "}
          – enables probability estimates via additional calibration, which
          slows training and prediction.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Tolerance{" "}
          </Text>{" "}
          – stopping threshold for optimisation. Smaller values increase
          precision but slow training.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Cache size{" "}
          </Text>{" "}
          – memory (in MB) used to cache kernel values. Larger caches can
          improve speed at the cost of RAM.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Class weight{" "}
          </Text>{" "}
          – adjusts class importance to mitigate imbalance.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max iterations{" "}
          </Text>{" "}
          – upper limit on optimisation steps. Use higher values if convergence
          warnings appear.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Decision shape{" "}
          </Text>{" "}
          – strategy for multi-class problems (e.g. one-vs-rest).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Break ties{" "}
          </Text>{" "}
          – refines tie-breaking between classes in multi-class settings, at a
          small computational cost.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  tree: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Decision tree parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Criterion{" "}
          </Text>{" "}
          – measure of split quality (e.g. gini or entropy). Affects how class
          purity is evaluated.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Splitter{" "}
          </Text>{" "}
          – strategy for choosing splits. Random splitting can reduce variance
          at the cost of interpretability.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max depth{" "}
          </Text>{" "}
          – maximum tree depth. Larger values increase model complexity and risk
          overfitting.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min samples split{" "}
          </Text>{" "}
          – minimum samples required to split a node. Larger values make the
          tree more conservative.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min samples leaf{" "}
          </Text>{" "}
          – minimum samples per leaf. Higher values smooth predictions and
          reduce overfitting.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min weight fraction leaf{" "}
          </Text>{" "}
          – minimum weighted fraction of the total sample weight required at a
          leaf. Mostly relevant when using sample weights; larger values make
          the tree more conservative.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max features{" "}
          </Text>{" "}
          – number of features considered at each split. Smaller values increase
          randomness and reduce correlation between splits.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max leaf nodes{" "}
          </Text>{" "}
          – maximum number of terminal nodes. Larger values allow finer decision
          regions but increase overfitting risk.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min impurity decrease{" "}
          </Text>{" "}
          – required reduction in impurity to allow a split. Larger values
          prevent weak, noisy splits.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Class weight{" "}
          </Text>{" "}
          – balances class importance, useful for imbalanced datasets.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            CCP alpha{" "}
          </Text>{" "}
          – pruning strength. Higher values produce simpler, more robust
          trees.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  forest: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Random forest parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Trees (n_estimators){" "}
          </Text>{" "}
          – number of trees in the forest. More trees improve stability but
          increase training time.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Criterion{" "}
          </Text>{" "}
          – measure used to evaluate split quality (e.g.
          gini/entropy/log_loss).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max depth{" "}
          </Text>{" "}
          – limits depth of each tree. Smaller values reduce overfitting but may
          underfit.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min samples split{" "}
          </Text>{" "}
          – minimum samples required to split a node. Higher values make the
          forest more conservative.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min samples leaf{" "}
          </Text>{" "}
          – minimum samples per leaf. Helps smooth predictions.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min weight fraction leaf{" "}
          </Text>{" "}
          – minimum weighted fraction of total sample weight at a leaf (mostly
          relevant with sample weights).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max features{" "}
          </Text>{" "}
          – number of features tried at each split. Smaller values increase tree
          diversity.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max leaf nodes{" "}
          </Text>{" "}
          – caps tree complexity. Larger values allow more detailed trees but
          increase variance.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min impurity decrease{" "}
          </Text>{" "}
          – minimum impurity improvement required to split. Larger values
          prevent weak/noisy splits.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Bootstrap{" "}
          </Text>{" "}
          – whether each tree is trained on a random sample with replacement.
          Improves robustness.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            OOB score{" "}
          </Text>{" "}
          – estimates generalisation error using unused samples during training
          (requires bootstrap=true).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Jobs (n_jobs){" "}
          </Text>{" "}
          – number of CPU cores used for training/prediction.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Random state{" "}
          </Text>{" "}
          – seed for reproducibility.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Warm start{" "}
          </Text>{" "}
          – reuses the existing fitted forest and adds more trees when
          refitting.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Class weight{" "}
          </Text>{" "}
          – adjusts class importance for imbalanced datasets.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            CCP alpha{" "}
          </Text>{" "}
          – pruning strength applied to each tree to control overfitting.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max samples{" "}
          </Text>{" "}
          – number or fraction of samples used per tree. Smaller values increase
          randomness (only used when bootstrap=true).{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  extratrees: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Extra Trees parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Trees (n_estimators){" "}
          </Text>{" "}
          – number of trees. More trees improve stability but increase training
          time.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Criterion{" "}
          </Text>{" "}
          – measure used to evaluate split quality (e.g.
          gini/entropy/log_loss).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max depth{" "}
          </Text>{" "}
          – limits depth of each tree. Smaller values reduce overfitting but may
          underfit.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min samples split / min samples leaf{" "}
          </Text>{" "}
          – minimum samples required to split a node / to form a leaf. Larger
          values make trees more conservative.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min weight fraction leaf{" "}
          </Text>{" "}
          – minimum weighted fraction of total sample weight at a leaf (mostly
          relevant with sample weights).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max features{" "}
          </Text>{" "}
          – number of features tried at each split. Smaller values increase
          randomness/diversity.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max leaf nodes{" "}
          </Text>{" "}
          – caps tree complexity. Larger values allow finer partitions but
          increase variance.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min impurity decrease{" "}
          </Text>{" "}
          – minimum impurity improvement required to split. Larger values
          prevent weak/noisy splits.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Bootstrap{" "}
          </Text>{" "}
          – optional sampling with replacement (default is usually off for Extra
          Trees).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            OOB score{" "}
          </Text>{" "}
          – out-of-bag score estimate using unused samples (requires
          bootstrap=true).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Jobs (n_jobs){" "}
          </Text>{" "}
          – number of CPU cores used for training/prediction.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Random state{" "}
          </Text>{" "}
          – seed for reproducibility.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Warm start{" "}
          </Text>{" "}
          – reuses the existing fitted ensemble and adds more trees when
          refitting.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Class weight{" "}
          </Text>{" "}
          – balances class importance for imbalanced datasets.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            CCP alpha{" "}
          </Text>{" "}
          – pruning strength applied to each tree.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max samples{" "}
          </Text>{" "}
          – number or fraction of samples used per tree (only used when
          bootstrap=true).{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  hgb: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> HistGradientBoosting parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Loss{" "}
          </Text>{" "}
          – objective function. For classification this is typically log_loss
          (cross-entropy).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Learning rate{" "}
          </Text>{" "}
          – step size of boosting. Smaller values often generalise better but
          need more iterations.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Iterations (max_iter){" "}
          </Text>{" "}
          – number of boosting stages. More stages can improve fit but may
          overfit.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max leaf nodes / max depth{" "}
          </Text>{" "}
          – controls complexity of each tree. Larger values capture interactions
          but increase overfitting risk.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Min samples leaf{" "}
          </Text>{" "}
          – minimum samples per leaf. Larger values smooth the model and reduce
          overfitting.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max features (fraction){" "}
          </Text>{" "}
          – fraction of features used per split (0–1]. Smaller values add
          randomness.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Max bins{" "}
          </Text>{" "}
          – number of discrete bins used when histogram-binning continuous
          features. More bins can capture finer detail but increase
          memory/time.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Early stopping{" "}
          </Text>{" "}
          – stops training when the validation score stops improving.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Scoring{" "}
          </Text>{" "}
          – metric used for early stopping / validation monitoring."loss" means
          stop when validation loss stops improving.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Validation fraction{" "}
          </Text>{" "}
          – fraction of training data held out for early stopping.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            No-change rounds{" "}
          </Text>{" "}
          – how many iterations without improvement are allowed before
          stopping.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Tolerance{" "}
          </Text>{" "}
          – minimum improvement considered “progress” for early stopping.
          Smaller values make stopping less sensitive.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            L2 regularisation{" "}
          </Text>{" "}
          – shrinks leaf values to reduce overfitting.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Verbose{" "}
          </Text>{" "}
          – prints training progress.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Random state{" "}
          </Text>{" "}
          – seed for reproducibility.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Class weight{" "}
          </Text>{" "}
          – balances class importance for imbalanced datasets.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  knn: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> kNN parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Neighbours{" "}
          </Text>{" "}
          – number of nearest samples considered. Small values are sensitive to
          noise; large values smooth predictions.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Weights{" "}
          </Text>{" "}
          – how neighbours contribute (uniform or distance-based).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Algorithm{" "}
          </Text>{" "}
          – strategy used to search for neighbours (auto, ball-tree,
          kd-tree).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Leaf size{" "}
          </Text>{" "}
          – affects speed and memory usage of tree-based searches.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            p{" "}
          </Text>{" "}
          – power parameter of the Minkowski distance (p=2 corresponds to
          Euclidean distance).{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Metric{" "}
          </Text>{" "}
          – distance function used to compare samples.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Jobs (n_jobs){" "}
          </Text>{" "}
          – number of CPU cores used for neighbour searches.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
  gnb: (
    <Stack className="helpParamSection">
      {" "}
      <Text className="helpTitle"> Gaussian Naive Bayes parameters </Text>{" "}
      <List className="helpOptionList">
        {" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Variance smoothing{" "}
          </Text>{" "}
          – adds a small value to variances for numerical stability. Increase if
          you see numerical issues.{" "}
        </List.Item>{" "}
        <List.Item>
          {" "}
          <Text span className="helpLabel">
            {" "}
            Priors{" "}
          </Text>{" "}
          – optional class prior probabilities. Leave empty to estimate from the
          training data.{" "}
        </List.Item>{" "}
      </List>{" "}
    </Stack>
  ),
};
