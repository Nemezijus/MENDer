export function getClassificationMetricTooltips({ isMulticlass }) {
  const precisionTooltip = isMulticlass
    ? 'For a given class, precision answers: “Of all samples the model predicted as this class, what fraction really belong to it?” High precision means the model rarely assigns this label incorrectly.'
    : 'Precision answers: “Of all samples predicted as positive, what fraction really are positive?” High precision means the model rarely raises false alarms.';

  const recallTooltip = isMulticlass
    ? 'For a given class, recall (TPR) answers: “Of all true samples of this class, what fraction did the model correctly label as this class?” High recall means the model rarely misses this class.'
    : 'Recall (TPR) answers: “Of all true positive samples, what fraction did the model correctly detect?” High recall means the model rarely misses positive cases.';

  const fprTooltip = isMulticlass
    ? 'For a given class, FPR is the fraction of samples from other classes that were incorrectly given this label. Lower FPR means fewer impostors being mistaken for this class.'
    : 'FPR is the fraction of true negative samples that were incorrectly predicted as positive. Lower FPR means fewer false alarms.';

  const tnrTooltip = isMulticlass
    ? 'For a given class, TNR (specificity) is the fraction of samples from other classes that correctly do not receive this label. Higher TNR means the model usually avoids assigning this class when it should not.'
    : 'TNR (specificity) is the fraction of true negative samples that the model correctly keeps as negative. Higher TNR means fewer false positives.';

  const fnrTooltip = isMulticlass
    ? 'For a given class, FNR is the fraction of true samples of this class that the model failed to label as such. Lower FNR means fewer missed examples of this class.'
    : 'FNR is the fraction of true positive samples that the model missed. Lower FNR means fewer missed positive cases.';

  const f1Tooltip =
    'F1 score is the harmonic mean of precision and recall. It is high only when both precision and recall are high.';

  const mccTooltip =
    'Matthews Correlation Coefficient (MCC) is a balanced measure of classification quality. It ranges from -1 (total disagreement) through 0 (random) to 1 (perfect prediction).';

  const macroTooltip =
    'Macro average treats each class equally, regardless of how many samples each class has. Good for seeing performance on rare classes.';

  const weightedTooltip =
    'Weighted average takes into account how many samples each class has. Large classes influence this more than small ones.';

  return {
    precisionTooltip,
    recallTooltip,
    fprTooltip,
    tnrTooltip,
    fnrTooltip,
    f1Tooltip,
    mccTooltip,
    macroTooltip,
    weightedTooltip,
  };
}
