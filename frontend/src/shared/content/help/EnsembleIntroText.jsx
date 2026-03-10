import "../styles/help.css";

// NOTE:
// This file intentionally contains only the short "preview" intro components.
// The expanded help (EnsembleHelpText.jsx) is lazy-loaded so the large help
// content does not inflate the initial bundle when help is collapsed.

export function VotingIntroText({ effectiveTask, votingType }) {
  const isReg = effectiveTask === "regression";

  return (
    <div className="helpSectionPreview">
      <div className="helpTitleStrong">Voting ensemble</div>

      <p className="helpBody">
        Trains multiple base models on the same split and combines their
        predictions.
      </p>

      <p className="helpBody">
        {isReg
          ? "Regression: averages predictions (VotingRegressor)."
          : votingType === "soft"
            ? "Soft voting averages predicted probabilities."
            : "Hard voting chooses the majority class."}
      </p>
    </div>
  );
}

export function BaggingIntroText({ effectiveTask }) {
  const isReg = effectiveTask === "regression";

  return (
    <div className="helpSectionPreview">
      <div className="helpTitleStrong">Bagging ensemble</div>

      <p className="helpBody">
        Trains many copies of the same estimator on resampled data and
        averages/votes.
      </p>

      <p className="helpBody">
        {isReg
          ? "Regression: averages predictions (BaggingRegressor)."
          : "Classification: majority vote across estimators (BaggingClassifier)."}
      </p>
    </div>
  );
}

export function AdaBoostIntroText({ effectiveTask }) {
  const isReg = effectiveTask === "regression";

  return (
    <div className="helpSectionPreview">
      <div className="helpTitleStrong">AdaBoost ensemble</div>

      <p className="helpBody">
        Adds weak learners sequentially, focusing more on the samples it
        previously got wrong.
      </p>

      <p className="helpBody">
        {isReg
          ? "Regression: weighted ensemble of weak regressors (AdaBoostRegressor)."
          : "Classification: boosts weak classifiers (AdaBoostClassifier)."}
      </p>
    </div>
  );
}

export function XGBoostIntroText({ effectiveTask }) {
  const isReg = effectiveTask === "regression";

  return (
    <div className="helpSectionPreview">
      <div className="helpTitleStrong">XGBoost</div>

      <p className="helpBody">
        Gradient boosted decision trees (high-performance for tabular data).
      </p>

      <p className="helpBody">
        {isReg ? "Regression: XGBRegressor." : "Classification: XGBClassifier."}
      </p>
    </div>
  );
}
