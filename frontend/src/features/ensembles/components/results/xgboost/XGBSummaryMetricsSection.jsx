import { Box, Text } from '@mantine/core';

import MetricPairsTable from '../common/MetricPairsTable.jsx';
import SectionTitle from '../common/SectionTitle.jsx';

import { fmt as fmtBase, safeNum } from '../../../utils/resultsFormat.js';

export default function XGBSummaryMetricsSection({ report }) {
  if (!report || report.kind !== 'xgboost') return null;

  const metricName = report.metric_name || '';
  const trainEvalMetric = report.train_eval_metric || '';

  const xgb = report.xgboost || {};
  const params = xgb.params || {};

  const bestIterMean = safeNum(xgb.best_iteration_mean);
  const bestIterStd = safeNum(xgb.best_iteration_std);
  const bestScoreMean = safeNum(xgb.best_score_mean);
  const bestScoreStd = safeNum(xgb.best_score_std);

  const summaryItems = [
    {
      label: 'Best iteration (mean)',
      value: bestIterMean == null ? 'N/A' : fmtBase(bestIterMean, 1, 'N/A'),
      tooltip:
        'Average best boosting round (only available when early stopping / eval sets are used).',
    },
    {
      label: 'Best iteration (std)',
      value: bestIterStd == null ? 'N/A' : fmtBase(bestIterStd, 1, 'N/A'),
      tooltip: 'Standard deviation of best iteration across folds.',
    },
    {
      label: 'Best score (mean)',
      value: bestScoreMean == null ? 'N/A' : fmtBase(bestScoreMean, 5, 'N/A'),
      tooltip:
        'Average best evaluation score reported by XGBoost during training (depends on eval metric).',
    },
    {
      label: 'Best score (std)',
      value: bestScoreStd == null ? 'N/A' : fmtBase(bestScoreStd, 5, 'N/A'),
      tooltip: 'Standard deviation of best score across folds.',
    },
  ];

  const rows = [
    [summaryItems[0], summaryItems[1]],
    [summaryItems[2], summaryItems[3]],
  ];

  const usedParams = [
    'n_estimators',
    'max_depth',
    'learning_rate',
    'subsample',
    'colsample_bytree',
    'reg_alpha',
    'reg_lambda',
    'gamma',
    'min_child_weight',
  ]
    .filter((k) => params && Object.prototype.hasOwnProperty.call(params, k))
    .map((k) => `${k}=${params[k]}`)
    .slice(0, 6);

  return (
    <Box>
      <SectionTitle title="Summary metrics" />
      <MetricPairsTable rows={rows} tooltipMaw={360} />

      <Text size="sm" c="dimmed" mt="xs" align="center">
        <b>Final metric:</b> {metricName || 'N/A'} • <b>Training eval metric:</b> {trainEvalMetric || 'N/A'}
        {usedParams.length > 0 ? (
          <>
            {' '}
            • <b>Params:</b> {usedParams.join(' • ')}
          </>
        ) : null}
      </Text>
    </Box>
  );
}
