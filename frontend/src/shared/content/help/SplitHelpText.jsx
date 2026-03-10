import "../styles/help.css";
export function SplitIntroText() {
  return (
    <div className="helpSection">
      {" "}
      <div className="helpTitle">What is a data split?</div>{" "}
      <p className="helpBody">
        {" "}
        A data split defines how the dataset is divided into training and
        validation/test parts. It affects how reliable your performance estimate
        is and how much data the model sees during training.{" "}
      </p>{" "}
    </div>
  );
}
export function SplitDetailsText({ selectedMode, allowStratified }) {
  const mode = selectedMode || "holdout";
  const isSelected = (name) => mode === name;
  const labelClassName = (name) =>
    isSelected(name)
      ? "helpChoiceLabel helpChoiceLabelSelected"
      : "helpChoiceLabel";
  return (
    <div className="helpSection">
      {" "}
      <div className="helpTitle">Split strategies</div>{" "}
      <ul className="helpOptionList">
        {" "}
        <li>
          {" "}
          <span className={labelClassName("holdout")}>Hold-out</span> – one
          fixed split into train and test (e.g. 75% / 25%). Simple and fast,
          good for quick experiments or large datasets.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className={labelClassName("kfold")}>
            K-fold cross-validation
          </span>{" "}
          – splits the data into <span className="helpStrong">K</span> folds and
          rotates which fold is used as validation. More expensive, but gives a
          more stable estimate on smaller datasets.{" "}
        </li>{" "}
      </ul>{" "}
      <div className="helpSubsectionTitle">Important options</div>{" "}
      <ul className="helpOptionList">
        {" "}
        <li>
          {" "}
          <span className="helpLabel">Train fraction</span> – used in{" "}
          <span className="helpLabel">hold-out</span>. Controls how much of the
          data is used for training versus testing (e.g. 0.75 → 75% train, 25%
          test).{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">K-Fold (n_splits)</span> – used in{" "}
          <span className="helpLabel">k-fold</span>. Higher values give more
          validation folds but increase runtime.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Stratified</span> – keeps the class
          proportions similar in each split. This is helpful for classification,
          especially with imbalanced classes, but is not used for regression.{" "}
          {!allowStratified && (
            <p className="helpBody">
              (Disabled here because the current task is regression.)
            </p>
          )}{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Shuffle split</span> – randomly shuffles
          the data before splitting. This helps avoid artifacts from ordering
          (e.g. time, blocks in acquisition), but should be used with care for
          time-series.{" "}
        </li>{" "}
        <li>
          {" "}
          <span className="helpLabel">Seed</span> – controls the random
          shuffling. Using a fixed seed makes the split reproducible; changing
          it gives a different random split.{" "}
        </li>{" "}
      </ul>{" "}
      <p className="helpBody">
        {" "}
        Rough rule of thumb: use <span className="helpStrong">
          hold-out
        </span>{" "}
        for fast prototyping or large datasets, and{" "}
        <span className="helpStrong">k-fold</span> when you need a more reliable
        estimate on limited data.{" "}
      </p>{" "}
    </div>
  );
} // Default export: just the detailed part (for the full-width C block)
export default function SplitHelpText({ selectedMode, allowStratified }) {
  return (
    <SplitDetailsText
      selectedMode={selectedMode}
      allowStratified={allowStratified}
    />
  );
}
