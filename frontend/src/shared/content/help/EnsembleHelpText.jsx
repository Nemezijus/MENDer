import "../styles/help.css";
/* ---------- short previews ---------- */ export function VotingIntroText({
  effectiveTask,
  votingType,
}) {
  const isReg = effectiveTask === "regression";
  return (
    <div className="helpSectionPreview">
      {" "}
      <div className="helpTitleStrong">Voting ensemble</div>{" "}
      <p className="helpBody">
        {" "}
        Trains multiple base models on the same split and combines their
        predictions.{" "}
      </p>{" "}
      <p className="helpBody">
        {" "}
        {isReg
          ? "Regression: averages predictions (VotingRegressor)."
          : votingType === "soft"
            ? "Soft voting averages predicted probabilities."
            : "Hard voting chooses the majority class."}{" "}
      </p>{" "}
    </div>
  );
}
export function BaggingIntroText({ effectiveTask }) {
  const isReg = effectiveTask === "regression";
  return (
    <div className="helpSectionPreview">
      {" "}
      <div className="helpTitleStrong">Bagging ensemble</div>{" "}
      <p className="helpBody">
        {" "}
        Trains many copies of the same estimator on resampled data and
        averages/votes.{" "}
      </p>{" "}
      <p className="helpBody">
        {" "}
        {isReg
          ? "Regression: averages predictions (BaggingRegressor)."
          : "Classification: majority vote across estimators (BaggingClassifier)."}{" "}
      </p>{" "}
    </div>
  );
}
export function AdaBoostIntroText({ effectiveTask }) {
  const isReg = effectiveTask === "regression";
  return (
    <div className="helpSectionPreview">
      {" "}
      <div className="helpTitleStrong">AdaBoost ensemble</div>{" "}
      <p className="helpBody">
        {" "}
        Adds weak learners sequentially, focusing more on the samples it
        previously got wrong.{" "}
      </p>{" "}
      <p className="helpBody">
        {" "}
        {isReg
          ? "Regression: weighted ensemble of weak regressors (AdaBoostRegressor)."
          : "Classification: boosts weak classifiers (AdaBoostClassifier)."}{" "}
      </p>{" "}
    </div>
  );
}
export function XGBoostIntroText({ effectiveTask }) {
  const isReg = effectiveTask === "regression";
  return (
    <div className="helpSectionPreview">
      {" "}
      <div className="helpTitleStrong">XGBoost</div>{" "}
      <p className="helpBody">
        {" "}
        Gradient boosted decision trees (high-performance for tabular
        data).{" "}
      </p>{" "}
      <p className="helpBody">
        {" "}
        {isReg
          ? "Regression: XGBRegressor."
          : "Classification: XGBClassifier."}{" "}
      </p>{" "}
    </div>
  );
}
/* ---------- expanded help ---------- */ function VotingDetailsText({
  effectiveTask,
  votingType,
}) {
  const isReg = effectiveTask === "regression";
  return (
    <div className="helpSection">
      {" "}
      <div className="helpTitleStrong">How to choose settings</div>{" "}
      <ul className="helpOptionList">
        {" "}
        <li>
          {" "}
          <span className="helpLabel">Simple vs Advanced</span> – Simple uses
          default hyperparameters. Advanced lets you tune estimators and
          optionally set weights.{" "}
        </li>{" "}
        {!isReg && (
          <li>
            {" "}
            <span className="helpLabel">Hard vs Soft</span> – Hard voting
            combines labels. Soft voting averages probabilities.{" "}
          </li>
        )}{" "}
        {!isReg && votingType === "soft" && (
          <li>
            {" "}
            <span className="helpLabel">Soft voting requirement</span> – all
            estimators must support{" "}
            <code className="helpCode">predict_proba</code>.{" "}
          </li>
        )}{" "}
        <li>
          {" "}
          <span className="helpLabel">Prefer diversity</span> – mixing model
          families often improves results.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Duplicates</span> – identical estimators
          act like implicit weighting; prefer explicit weights.{" "}
        </li>{" "}
      </ul>{" "}
    </div>
  );
}
function BaggingDetailsText() {
  return (
    <div className="helpSection">
      {" "}
      <div className="helpTitleStrong">How to choose settings</div>{" "}
      <ul className="helpGuideList">
        {" "}
        <li>
          {" "}
          <span className="helpLabel">Number of estimators</span> – how many
          base models you train. More estimators usually reduce variance and
          make results more stable, but increase training time. Typical range:{" "}
          <span className="helpStrong">25–200</span>. If results look noisy,
          increase this.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Max samples (fraction)</span> – fraction
          of the training fold used to train each estimator.{" "}
          <span className="helpStrong">1.0</span> means “same size as the
          training fold” (with replacement if Bootstrap is on). Smaller values
          (e.g. <span className="helpStrong">0.5–0.9</span>) increase diversity
          between estimators and can reduce overfitting, but each estimator
          learns from less data.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Max features (fraction)</span> – fraction
          of input features used per estimator (feature subsampling / random
          subspace). Lower values increase estimator diversity and can improve
          generalization in high-dimensional problems, but may reduce accuracy
          if too low. Common starting points:{" "}
          <span className="helpStrong">0.5–1.0</span>.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Bootstrap</span> – when enabled, each
          estimator trains on a bootstrap sample (sampling{" "}
          <span className="helpStrong">with replacement</span>). This is classic
          bagging and adds randomness. If disabled, sampling is{" "}
          <span className="helpStrong">without replacement</span>. With
          Bootstrap off and Max samples = 1.0, every estimator sees the same
          rows (so only Max features adds randomness).{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Bootstrap features</span> – when enabled,
          each estimator also trains on a resampled subset of features. This can
          further increase diversity, especially when you have many correlated
          features. If you already use Max features &lt; 1.0, this may be
          redundant.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Out-of-bag score</span> – only meaningful
          when <span className="helpStrong">Bootstrap</span> is enabled. For
          each estimator, some training samples are not selected into its
          bootstrap sample (“out-of-bag” samples). The out-of-bag score
          evaluates predictions on those left-out samples and gives a built-in
          generalization estimate without creating a separate validation set.
          This is most useful for quick feedback and sanity checks; for
          reporting, prefer your chosen holdout / k-fold split.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Balanced bagging</span> – uses an
          imbalanced-learn variant that tries to reduce class imbalance inside
          each estimator’s training sample. Recommended when classes are
          noticeably imbalanced or when you see bagging failures due to class
          sparsity. If your dataset is only mildly imbalanced, you usually don’t
          need it.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">
            Sampling strategy (Balanced bagging)
          </span>{" "}
          – controls how classes are balanced inside each bag.{" "}
          <span className="helpStrong">Auto</span> is the safe default. Options
          like “majority”, “not minority”, etc. decide which classes are
          down-sampled. If you’re unsure, keep{" "}
          <span className="helpStrong">Auto</span>.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Replacement (Balanced bagging)</span> –
          whether the class-balancing sampler is allowed to sample with
          replacement. Turning this on can help when some classes have few
          samples, but may increase duplicate rows inside a bag.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Number of jobs</span> – parallelism.
          Higher values use more CPU cores to train estimators faster. If
          supported, <span className="helpStrong">-1</span> means “use all
          cores”. If your machine becomes unresponsive, reduce this.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Random state</span> – seed for
          reproducibility. Set it to a fixed value (e.g. 42) to make results
          repeatable across runs.{" "}
        </li>{" "}
      </ul>{" "}
    </div>
  );
}
function AdaBoostDetailsText({ effectiveTask }) {
  const isReg = effectiveTask === "regression";
  return (
    <div className="helpSection">
      {" "}
      <div className="helpTitleStrong">How to choose settings</div>{" "}
      <ul className="helpGuideList">
        {" "}
        <li>
          {" "}
          <span className="helpLabel">Base estimator</span> – AdaBoost works
          best with <span className="helpStrong">weak learners</span>. The
          classic choice is a decision stump (a very shallow tree). If you use a
          strong learner (deep trees, complex models), AdaBoost can overfit
          quickly and become unstable.{" "}
          {!isReg && (
            <p className="helpBody">
              Tip: for classification, start with a shallow tree-like base
              learner.
            </p>
          )}{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Number of estimators</span> – number of
          boosting rounds (how many weak learners are added sequentially).
          Higher values can improve performance, but also increase training time
          and overfitting risk. Typical range:{" "}
          <span className="helpStrong">50–500</span>. If you reduce the learning
          rate, you usually need more estimators.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Learning rate</span> – scales how much
          each new learner contributes. Smaller values make boosting more
          conservative and often improve generalization, but you typically need
          more estimators. Good starting points:{" "}
          <span className="helpStrong">0.05–0.5</span>. If results are unstable,
          try lowering it.{" "}
        </li>{" "}
        {!isReg && (
          <li>
            {" "}
            <span className="helpLabel">Algorithm</span> – controls the boosting
            variant. If you’re unsure, keep the default. Older sklearn versions
            used SAMME/SAMME.R; newer versions have changed defaults and may
            deprecate some options. Only change this if you know you need
            it.{" "}
          </li>
        )}{" "}
        <li>
          {" "}
          <span className="helpLabel">Random state</span> – seed for
          reproducibility. Set to a fixed value (e.g. 42) to make results
          repeatable across runs (especially with stochastic base
          estimators).{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Practical tuning recipe</span> – if you
          see overfitting: reduce the learning rate, reduce base estimator
          complexity, and/or add more data. If you see underfitting: increase
          estimators and/or allow slightly stronger base learners.{" "}
        </li>{" "}
      </ul>{" "}
    </div>
  );
}
function XGBoostDetailsText() {
  return (
    <div className="helpSection">
      {" "}
      <div className="helpTitleStrong">How to choose settings</div>{" "}
      <ul className="helpGuideList">
        {" "}
        <li>
          {" "}
          <span className="helpLabel">Number of estimators</span> – number of
          boosted trees. More trees can improve performance, but increase
          training time and overfitting risk. Typical range:{" "}
          <span className="helpStrong">200–2000</span> (depends heavily on
          learning rate and dataset size).{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Learning rate</span> – how aggressively
          each tree updates the model. Smaller values are safer and often
          generalize better, but need more trees. Common starting points:{" "}
          <span className="helpStrong">0.03–0.2</span>.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Max depth</span> – maximum depth of each
          tree. Deeper trees can capture more complex patterns but overfit more
          easily. Typical range: <span className="helpStrong">3–10</span>. If
          you see overfitting, reduce this.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Subsample</span> – fraction of rows used
          to grow each tree. Values &lt; 1.0 add randomness and often improve
          generalization (especially on noisy data). Try{" "}
          <span className="helpStrong">0.6–0.9</span> as a starting range.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Column sample by tree</span> – fraction of
          features considered per tree. Reducing this (e.g.{" "}
          <span className="helpStrong">0.5–0.9</span>) can reduce overfitting
          and help with high-dimensional inputs.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">L2 regularization (lambda)</span> – larger
          values penalize large weights and can reduce overfitting. If your
          model overfits, try increasing it.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">L1 regularization (alpha)</span> –
          encourages sparsity in leaf weights. Can help when many features are
          noisy or redundant.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Min child weight</span> – minimum “amount
          of information” needed in a leaf. Higher values make the algorithm
          more conservative (fewer, simpler splits), which can help reduce
          overfitting.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Gamma</span> – minimum loss reduction
          required to make a split. Higher values make splitting more
          conservative (often helps with overfitting).{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Use early stopping (Advanced)</span> –
          enables an{" "}
          <span className="helpStrong">internal validation split</span> from the
          training data, and stops adding trees once the validation metric stops
          improving. This is also what allows MENDer to show{" "}
          <span className="helpStrong">learning curves</span> (training vs
          validation over boosting rounds). Turning this off is fine if you only
          care about the final held-out / k-fold metric, but then “best
          iteration/score” and learning curves may be unavailable.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Early stopping rounds (patience)</span> –
          how many rounds XGBoost will wait without improvement before stopping.
          Smaller values stop sooner (faster, less overfitting risk); larger
          values are more permissive. If left blank, MENDer chooses a reasonable
          default.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Eval set fraction (Advanced)</span> –
          fraction of the training fold reserved for internal early-stopping
          evaluation (typical range: <span className="helpStrong">0.1–0.3</span>
          ). Higher values give a more stable validation signal but leave fewer
          samples to fit the trees. This does{" "}
          <span className="helpStrong">not</span> change your main train/test
          split (holdout/k-fold).{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">
            Internal eval metric vs final metric
          </span>{" "}
          – learning curves and “best score” are based on an internal training
          metric (e.g. logloss/mlogloss/rmse), chosen for stability. Your model
          card and confusion/ROC use your selected final metric (e.g.
          accuracy).{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Number of jobs</span> – parallelism.
          Higher values use more CPU cores. If supported,{" "}
          <span className="helpStrong">-1</span> means “use all cores”.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Random state</span> – seed for
          reproducibility. Set to a fixed value (e.g. 42) for repeatable
          runs.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Practical tuning recipe</span> – start
          with learning_rate 0.1, max_depth 4–6, subsample/colsample 0.8, then
          tune depth/regularization to control overfitting. If underfitting, add
          trees or increase depth slightly.{" "}
        </li>{" "}
      </ul>{" "}
      <p className="helpBody">
        {" "}
        Note: XGBoost requires the xgboost Python package to be installed in the
        backend environment.{" "}
      </p>{" "}
    </div>
  );
}
/* ---------- router ---------- */ export default function EnsembleHelpText({
  kind,
  effectiveTask,
  votingType,
}) {
  if (kind === "voting") {
    return (
      <VotingDetailsText
        effectiveTask={effectiveTask}
        votingType={votingType}
      />
    );
  }
  if (kind === "bagging") {
    return <BaggingDetailsText />;
  }
  if (kind === "adaboost") {
    return <AdaBoostDetailsText effectiveTask={effectiveTask} />;
  }
  if (kind === "xgboost") {
    return <XGBoostDetailsText />;
  }
  return (
    <p className="helpBody">
      Help text for this ensemble type is not available yet.
    </p>
  );
}
