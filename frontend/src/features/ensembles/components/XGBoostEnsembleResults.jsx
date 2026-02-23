import { Divider } from '@mantine/core';

import ResultsCardShell from './results/common/ResultsCardShell.jsx';
import XGBSummaryMetricsSection from './results/xgboost/XGBSummaryMetricsSection.jsx';
import XGBInsightsRow from './results/xgboost/XGBInsightsRow.jsx';

export default function XGBoostEnsembleResults({ report }) {
  if (!report || report.kind !== 'xgboost') return null;

  return (
    <ResultsCardShell title="XGBoost ensemble insights">
      <XGBSummaryMetricsSection report={report} />
      <Divider my="xs" />
      <XGBInsightsRow report={report} />
    </ResultsCardShell>
  );
}
