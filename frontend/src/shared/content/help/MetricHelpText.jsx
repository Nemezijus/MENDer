import "../styles/help.css";
export function MetricIntroText() {
  return (
    <div className="helpSection">
      {" "}
      <div className="helpTitle">What is a metric?</div>{" "}
      <p className="helpBody">
        {" "}
        A metric is used to judge how well your model performs on validation or
        test data. Different metrics highlight different aspects of performance,
        such as overall accuracy, how well all classes are treated, or how good
        the predicted probabilities are.{" "}
      </p>{" "}
    </div>
  );
}
export function MetricDetailsText({ selectedMetric }) {
  // Helper to check if a given metric is currently selected const isSelected = (name) => selectedMetric === name; const labelClassName = (name) => isSelected(name) ? 'helpChoiceLabel helpChoiceLabelSelected' : 'helpChoiceLabel'; return ( <div className="helpSection"> <div className="helpTitle">Classification metrics</div> <ul className="helpOptionList"> <li> <span className={labelClassName('accuracy')}>accuracy</span> – fraction of all predictions that are correct. Simple and intuitive, but can be misleading when classes are very imbalanced. </li> <li> <span className={labelClassName('balanced_accuracy')}> balanced_accuracy </span>{' '} – average of recall over classes. Each class contributes equally, so it is more informative when some classes are rare. </li> <li> <span className={labelClassName('f1_macro')}>f1_macro</span> – F1 score (harmonic mean of precision and recall) averaged equally over classes. Good when you care about performance on all classes, including minority ones. </li> <li> <span className={labelClassName('f1_micro')}>f1_micro</span> – F1 computed globally over all samples. Weights classes by how often they appear and behaves similarly to accuracy, but still reflects both precision and recall. </li> <li> <span className={labelClassName('f1_weighted')}>f1_weighted</span> – F1 averaged over classes, weighted by class support. Large classes influence this more than rare ones. </li> <li> <span className={labelClassName('precision_macro')}> precision_macro </span>{' '} – average precision over classes. Focuses on how often predicted labels are correct and is sensitive to false positives. </li> <li> <span className={labelClassName('recall_macro')}>recall_macro</span> – average recall (TPR) over classes. Focuses on how many true examples are detected and is sensitive to false negatives. </li> <li> <span className={labelClassName('log_loss')}>log_loss</span> – penalises wrong and overconfident probability predictions. Useful when you care about well-calibrated probabilistic outputs, not just hard labels. </li> <li> <span className={labelClassName('roc_auc_ovr')}>roc_auc_ovr</span> – area under the ROC curve in a one-vs-rest setting. Measures how well the model ranks true class vs others across thresholds, robust to class imbalance. </li> <li> <span className={labelClassName('roc_auc_ovo')}>roc_auc_ovo</span> – area under ROC curves averaged over all one-vs-one class pairs. Highlights how well each pair of classes can be separated. </li> <li> <span className={labelClassName('avg_precision_macro')}> avg_precision_macro </span>{' '} – macro-averaged area under the precision–recall curve. Emphasises performance on positive examples and is useful under heavy class imbalance. </li> </ul> <div className="helpSubsectionTitle">Regression metrics</div> <ul className="helpOptionList"> <li> <span className={labelClassName('r2')}>r2</span> – proportion of variance in the target explained by the model. 1 is perfect, 0 means no better than a constant baseline. </li> <li> <span className={labelClassName('explained_variance')}> explained_variance </span>{' '} – similar to R², focusing on the variance of the prediction error. </li> <li> <span className={labelClassName('mse')}>mse</span> – mean squared error. Large errors are penalised more strongly, which makes it sensitive to outliers. </li> <li> <span className={labelClassName('rmse')}>rmse</span> – square root of MSE. Same units as the target, so it is easier to interpret while still penalising large errors. </li> <li> <span className={labelClassName('mae')}>mae</span> – mean absolute error. Measures average absolute deviation and is more robust to outliers than MSE. </li> <li> <span className={labelClassName('mape')}>mape</span> – mean absolute percentage error. Expresses error as a percentage of the true value, but can be unstable when targets are close to zero. </li> </ul> <p className="helpSubsectionText"> As a rough guideline, for balanced classification you can start with{' '} <span className="helpStrong">accuracy</span> or{' '} <span className="helpStrong">f1_macro</span>. For imbalanced multi-class problems, <span className="helpStrong">balanced_accuracy</span>{' '} or <span className="helpStrong">f1_macro</span> are often more informative than plain accuracy. For regression tasks, common choices are <span className="helpStrong">r2</span> or{' '} <span className="helpStrong">rmse</span>. </p> </div> );
}
export default function MetricHelpText({ selectedMetric }) {
  return (
    <div className="helpSectionPanel">
      {" "}
      <MetricDetailsText selectedMetric={selectedMetric} />{" "}
    </div>
  );
}
