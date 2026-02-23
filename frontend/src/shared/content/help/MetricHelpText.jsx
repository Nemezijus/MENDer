import { Stack, Text, List } from '@mantine/core';

export function MetricIntroText() {
  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        What is a metric?
      </Text>

      <Text size="xs" c="dimmed">
        A metric is used to judge how well your model performs on validation or
        test data. Different metrics highlight different aspects of performance,
        such as overall accuracy, how well all classes are treated, or how good
        the predicted probabilities are.
      </Text>
    </Stack>
  );
}

export function MetricDetailsText({ selectedMetric }) {
  // Helper to check if a given metric is currently selected
  const isSelected = (name) => selectedMetric === name;

  // Small style helpers for the metric name label
  const labelStyle = (name) => ({
    fw: isSelected(name) ? 700 : 600,
    c: isSelected(name) ? 'blue' : undefined,
  });

  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        Classification metrics
      </Text>

      <List spacing={4} size="xs">
        <List.Item>
          <Text span {...labelStyle('accuracy')}>
            accuracy
          </Text>{' '}
          – fraction of all predictions that are correct. Simple and intuitive,
          but can be misleading when classes are very imbalanced.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('balanced_accuracy')}>
            balanced_accuracy
          </Text>{' '}
          – average of recall over classes. Each class contributes equally, so
          it is more informative when some classes are rare.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('f1_macro')}>
            f1_macro
          </Text>{' '}
          – F1 score (harmonic mean of precision and recall) averaged equally
          over classes. Good when you care about performance on all classes,
          including minority ones.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('f1_micro')}>
            f1_micro
          </Text>{' '}
          – F1 computed globally over all samples. Weights classes by how often
          they appear and behaves similarly to accuracy, but still reflects both
          precision and recall.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('f1_weighted')}>
            f1_weighted
          </Text>{' '}
          – F1 averaged over classes, weighted by class support. Large classes
          influence this more than rare ones.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('precision_macro')}>
            precision_macro
          </Text>{' '}
          – average precision over classes. Focuses on how often predicted
          labels are correct and is sensitive to false positives.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('recall_macro')}>
            recall_macro
          </Text>{' '}
          – average recall (TPR) over classes. Focuses on how many true
          examples are detected and is sensitive to false negatives.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('log_loss')}>
            log_loss
          </Text>{' '}
          – penalises wrong and overconfident probability predictions. Useful
          when you care about well-calibrated probabilistic outputs, not just
          hard labels.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('roc_auc_ovr')}>
            roc_auc_ovr
          </Text>{' '}
          – area under the ROC curve in a one-vs-rest setting. Measures how well
          the model ranks true class vs others across thresholds, robust to
          class imbalance.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('roc_auc_ovo')}>
            roc_auc_ovo
          </Text>{' '}
          – area under ROC curves averaged over all one-vs-one class pairs.
          Highlights how well each pair of classes can be separated.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('avg_precision_macro')}>
            avg_precision_macro
          </Text>{' '}
          – macro-averaged area under the precision–recall curve. Emphasises
          performance on positive examples and is useful under heavy class
          imbalance.
        </List.Item>
      </List>

      <Text fw={500} size="sm" mt="xs">
        Regression metrics
      </Text>

      <List spacing={4} size="xs">
        <List.Item>
          <Text span {...labelStyle('r2')}>
            r2
          </Text>{' '}
          – proportion of variance in the target explained by the model. 1 is
          perfect, 0 means no better than a constant baseline.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('explained_variance')}>
            explained_variance
          </Text>{' '}
          – similar to R², focusing on the variance of the prediction error.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('mse')}>
            mse
          </Text>{' '}
          – mean squared error. Large errors are penalised more strongly, which
          makes it sensitive to outliers.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('rmse')}>
            rmse
          </Text>{' '}
          – square root of MSE. Same units as the target, so it is easier to
          interpret while still penalising large errors.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('mae')}>
            mae
          </Text>{' '}
          – mean absolute error. Measures average absolute deviation and is more
          robust to outliers than MSE.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('mape')}>
            mape
          </Text>{' '}
          – mean absolute percentage error. Expresses error as a percentage of
          the true value, but can be unstable when targets are close to zero.
        </List.Item>
      </List>

      <Text size="xs" c="dimmed" mt="xs">
        As a rough guideline, for balanced classification you can start with{' '}
        <Text span fw={600}>
          accuracy
        </Text>{' '}
        or{' '}
        <Text span fw={600}>
          f1_macro
        </Text>
        . For imbalanced multi-class problems,{' '}
        <Text span fw={600}>
          balanced_accuracy
        </Text>{' '}
        or{' '}
        <Text span fw={600}>
          f1_macro
        </Text>{' '}
        are often more informative than plain accuracy. For regression tasks,
        common choices are{' '}
        <Text span fw={600}>
          r2
        </Text>{' '}
        or{' '}
        <Text span fw={600}>
          rmse
        </Text>
        .
      </Text>
    </Stack>
  );
}

export default function MetricHelpText({ selectedMetric }) {
  return (
    <Stack gap="sm">
      <MetricDetailsText selectedMetric={selectedMetric} />
    </Stack>
  );
}
