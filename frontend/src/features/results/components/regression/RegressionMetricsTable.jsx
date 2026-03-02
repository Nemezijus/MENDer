import { Fragment } from 'react';
import { Table, Text, Tooltip } from '@mantine/core';
import { chunk, fmt3, rangeText } from '../../utils/formatNumbers.js';

export default function RegressionMetricsTable({ summary }) {
  if (!summary || typeof summary !== 'object') return null;

  const headerTextStyle = 'resultsHeaderTextCompact';

  const metricPairs = [
    { k: 'RMSE', v: summary.rmse, tip: 'Root mean squared error. Smaller is better.' },
    { k: 'MAE', v: summary.mae, tip: 'Mean absolute error. Smaller is better.' },
    {
      k: 'Median AE',
      v: summary.median_ae,
      tip: 'Median absolute error (robust to outliers). Smaller is better.',
    },
    { k: 'R²', v: summary.r2, tip: 'Coefficient of determination. Larger is better (1 is perfect).' },
    {
      k: 'Explained variance',
      v: summary.explained_variance,
      tip: 'Explained variance score. Larger is better (1 is perfect).',
    },
    {
      k: 'Bias (pred − true)',
      v: summary.bias,
      tip: 'Mean(predicted − true). Values > 0 indicate overestimation on average.',
    },
    {
      k: 'Residual std',
      v: summary.residual_std,
      tip: 'Standard deviation of residuals (predicted − true). Smaller is better.',
    },
    {
      k: 'Pearson r',
      v: summary.pearson_r,
      tip: 'Linear correlation between predictions and ground truth.',
    },
    {
      k: 'Spearman ρ',
      v: summary.spearman_r,
      tip: 'Rank correlation between predictions and ground truth.',
    },
    {
      k: 'NRMSE',
      v: summary.nrmse,
      tip: 'Normalized RMSE (e.g., divided by std(y_true)). Smaller is better.',
    },
    { k: 'N samples', v: summary.n, tip: 'Number of evaluation samples used for these metrics.' },
    {
      k: 'True range',
      v: rangeText(summary.y_true_min, summary.y_true_max),
      tip: 'Range of ground-truth values in the evaluation set.',
      isText: true,
    },
    {
      k: 'Pred range',
      v: rangeText(summary.y_pred_min, summary.y_pred_max),
      tip: 'Range of predicted values in the evaluation set.',
      isText: true,
    },
  ];

  const metricRows = chunk(metricPairs, 3);

  return (
    <Table
      withTableBorder={false}
      withColumnBorders={false}
      horizontalSpacing="xs"
      verticalSpacing="xs"
      striped
    >
      <Table.Thead>
        <Table.Tr className="resultsDarkHeaderRow">
          {Array.from({ length: 3 }).map((_, i) => (
            <Fragment key={`hdr-${i}`}>
              <Table.Th className="tableCellCenterNowrap">
                <Text size="xs" fw={600} c="white" className={headerTextStyle}>
                  Metric
                </Text>
              </Table.Th>
              <Table.Th className="tableCellCenterNowrap">
                <Text size="xs" fw={600} c="white" className={headerTextStyle}>
                  Value
                </Text>
              </Table.Th>
            </Fragment>
          ))}
        </Table.Tr>
      </Table.Thead>

      <Table.Tbody>
        {metricRows.map((row, idx) => (
          <Table.Tr key={idx}>
            {row.map((r) => (
              <Fragment key={r.k}>
                <Table.Td className="resultsTdLeft">
                  <Tooltip label={r.tip} multiline maw={320} withArrow>
                    <Text size="sm" fw={500} className="resultsCursorHelp">
                      {r.k}
                    </Text>
                  </Tooltip>
                </Table.Td>
                <Table.Td className="resultsTdCenter">
                  <Text size="sm" fw={700}>
                    {r.isText ? String(r.v ?? '—') : fmt3(r.v)}
                  </Text>
                </Table.Td>
              </Fragment>
            ))}

            {row.length < 3 &&
              Array.from({ length: 3 - row.length }).map((_, p) => (
                <Fragment key={`pad-${p}`}>
                  <Table.Td />
                  <Table.Td />
                </Fragment>
              ))}
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}
