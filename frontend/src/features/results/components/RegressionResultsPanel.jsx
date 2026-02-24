import { Card, Divider, Stack, Text } from '@mantine/core';
import { binCenters, parseNumber } from '../utils/formatNumbers.js';
import RegressionMetricsTable from './regression/RegressionMetricsTable.jsx';
import FoldScoreDistribution from './regression/FoldScoreDistribution.jsx';
import PredVsTruePlot from './regression/PredVsTruePlot.jsx';
import ResidualHistogram from './regression/ResidualHistogram.jsx';
import ResidualsVsPredPlot from './regression/ResidualsVsPredPlot.jsx';
import ErrorByTrueBinPlot from './regression/ErrorByTrueBinPlot.jsx';

export default function RegressionResultsPanel({ trainResult }) {
  if (!trainResult) return null;

  const nSplits = trainResult?.n_splits;
  const isKFold = typeof nSplits === 'number' && Number.isFinite(nSplits) && nSplits > 1;

  const metricNameRaw = trainResult?.metric_name;
  const metricName =
    typeof metricNameRaw === 'string' && metricNameRaw.trim() !== ''
      ? metricNameRaw.replaceAll('_', ' ')
      : 'score';

  const foldScores = Array.isArray(trainResult?.fold_scores)
    ? trainResult.fold_scores.filter((v) => typeof v === 'number' && Number.isFinite(v))
    : [];
  const hasFoldScores = isKFold && foldScores.length > 1;

  const diag = trainResult.regression || null;
  const summary = diag?.summary || null;
  if (!summary || typeof summary !== 'object') return null;

  const predVsTrue = diag?.pred_vs_true || null;
  const residualHist = diag?.residual_hist || null;
  const residualsVsPred = diag?.residuals_vs_pred || null;
  const errorByBin = diag?.error_by_true_bin || null;
  const idealLine = diag?.ideal_line || null;

  // Residual histogram (edges -> bin centers).
  const histEdges = residualHist?.edges;
  const histCounts = residualHist?.counts;
  const hasHist =
    Array.isArray(histEdges) &&
    Array.isArray(histCounts) &&
    histEdges.length >= 2 &&
    histCounts.length === histEdges.length - 1;

  const histX = hasHist ? binCenters(histEdges) : [];
  const histW = hasHist
    ? histEdges.slice(0, -1).map((e, i) => {
        const a = parseNumber(e);
        const b = parseNumber(histEdges[i + 1]);
        return a !== null && b !== null ? Math.abs(b - a) : 0;
      })
    : [];

  // Error by target magnitude (quantile bins).
  const hasBinned =
    errorByBin &&
    Array.isArray(errorByBin.edges) &&
    Array.isArray(errorByBin.mae) &&
    Array.isArray(errorByBin.rmse) &&
    errorByBin.edges.length >= 2 &&
    errorByBin.mae.length === errorByBin.edges.length - 1 &&
    errorByBin.rmse.length === errorByBin.edges.length - 1;

  const binnedX = hasBinned ? binCenters(errorByBin.edges) : [];

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={600} size="xl" align="center">
          Regression diagnostics
        </Text>

        <Stack gap={6}>
          <Text size="sm" c="dimmed">
            {isKFold
              ? `Summary metrics computed on pooled out-of-fold predictions (concatenated across ${nSplits} folds).`
              : 'Summary metrics computed on the evaluation (test) set.'}
          </Text>
        </Stack>

        <RegressionMetricsTable summary={summary} />

        <Divider />

        {hasFoldScores && (
          <>
            <FoldScoreDistribution
              foldScores={foldScores}
              metricName={metricName}
              metricNameRaw={metricNameRaw}
            />
            <Divider />
          </>
        )}

        <PredVsTruePlot points={predVsTrue} idealLine={idealLine} />

        {hasHist && (
          <>
            <Divider />
            <ResidualHistogram x={histX} counts={histCounts} widths={histW} />
          </>
        )}

        <ResidualsVsPredPlot points={residualsVsPred} />

        {hasBinned && (
          <>
            <Divider />
            <ErrorByTrueBinPlot x={binnedX} mae={errorByBin.mae} rmse={errorByBin.rmse} />
          </>
        )}
      </Stack>
    </Card>
  );
}
