import "../styles/help.css";
import { MODEL_OVERVIEW_ENTRIES } from "./model/overviewEntries.js";
import { MODEL_PARAM_BLOCKS } from "./model/params/index.js";
export function ModelIntroText() {
  return (
    <div className="helpSection">
      {" "}
      <div className="helpTitle">What is a model?</div>{" "}
      <p className="helpBody">
        {" "}
        A model is the algorithm that learns patterns from your training data
        and makes predictions on new data. Different models make different
        assumptions and trade off accuracy, interpretability, robustness, and
        training speed.{" "}
      </p>{" "}
    </div>
  );
}
export function ModelDetailsText({
  selectedAlgo,
  effectiveTask,
  visibleAlgos,
}) {
  const algo = selectedAlgo || null;
  const allowed = new Set(visibleAlgos || []);
  const isVisible = (name) => allowed.size === 0 || allowed.has(name);
  const isSelected = (name) => algo === name;
  const labelClassName = (name) =>
    isSelected(name)
      ? "helpChoiceLabel helpChoiceLabelSelected"
      : "helpChoiceLabel";
  const taskNote =
    effectiveTask === "classification"
      ? "You are working on a classification task, so classification algorithms are most relevant."
      : effectiveTask === "regression"
        ? "You are working on a regression task, so regression algorithms are most relevant."
        : effectiveTask === "unsupervised"
          ? "You are working on an unsupervised task, so clustering / mixture models are most relevant."
          : "If the task is not set yet, you can still explore models. They will be filtered once the task is known.";
  return (
    <div className="helpSection">
      {" "}
      <div className="helpTitle">Choosing a model</div>{" "}
      <p className="helpBody">
        {" "}
        No single model is best for all problems. Simpler models are easier to
        interpret and faster to train, while more flexible models can capture
        complex patterns but may overfit.{" "}
      </p>{" "}
      <p className="helpBody">{taskNote}</p>{" "}
      <ul className="helpOptionList helpSubsectionList">
        {" "}
        {MODEL_OVERVIEW_ENTRIES.filter((entry) => isVisible(entry.algo)).map(
          (entry) => (
            <li key={entry.algo}>
              {" "}
              <span className={labelClassName(entry.algo)}>
                {entry.label}
              </span>{" "}
              – {entry.summary}{" "}
            </li>
          ),
        )}{" "}
      </ul>{" "}
    </div>
  );
}
export function ModelParamsText({ selectedAlgo }) {
  if (!selectedAlgo) {
    return (
      <p className="helpBody">
        {" "}
        Select an algorithm above to see a short description of its key
        parameters.{" "}
      </p>
    );
  }
  const block = MODEL_PARAM_BLOCKS[selectedAlgo];
  if (block) {
    return block;
  }
  return (
    <p className="helpBody">
      No parameter help is defined for this algorithm yet.
    </p>
  );
}
export default function ModelHelpText({
  selectedAlgo,
  effectiveTask,
  visibleAlgos,
}) {
  return (
    <div className="helpSectionPanel">
      {" "}
      <ModelDetailsText
        selectedAlgo={selectedAlgo}
        effectiveTask={effectiveTask}
        visibleAlgos={visibleAlgos}
      />{" "}
      <ModelParamsText selectedAlgo={selectedAlgo} />{" "}
    </div>
  );
}
