import "../styles/help.css";

// NOTE:
// This component is intentionally split out from ModelHelpText.jsx so that
// the large model parameter help blocks can be lazy-loaded only when the user
// expands the help section.

export function ModelIntroText() {
  return (
    <div className="helpSection">
      <div className="helpTitle">What is a model?</div>

      <p className="helpBody">
        A model is the algorithm that learns patterns from your training data
        and makes predictions on new data. Different models make different
        assumptions and trade off accuracy, interpretability, robustness, and
        training speed.
      </p>
    </div>
  );
}

export default ModelIntroText;
