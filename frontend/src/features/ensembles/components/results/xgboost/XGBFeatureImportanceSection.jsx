import { Box, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';
import { cmBlue } from '../../../utils/resultsFormat.js';

export default function XGBFeatureImportanceSection({ report }) {
  if (!report || report.kind !== 'xgboost') return null;

  const feat = report.feature_importance || null;

  const topFeatures = Array.isArray(feat?.top_features) ? feat.top_features : [];
  const featNames = topFeatures.map((d) => String(d.name));
  const featVals = topFeatures.map((d) => Number(d.importance) || 0);

  const featTrace =
    topFeatures.length > 0
      ? {
          type: 'bar',
          orientation: 'h',
          y: featNames.slice().reverse(),
          x: featVals.slice().reverse(),
          marker: {
            color: featVals
              .slice()
              .reverse()
              .map((_, i) =>
                cmBlue(0.55 + 0.35 * (i / Math.max(1, featVals.length - 1))),
              ),
          },
          hovertemplate: '%{y}<br>importance: %{x:.5f}<extra></extra>',
        }
      : null;

  return (
    <Box>
      <SectionTitle
        title="Feature importance"
        tooltip="Top feature importances aggregated across folds (normalized)."
        maw={420}
      />

      {featTrace ? (
        <Plot
          data={[featTrace]}
          layout={{
            autosize: true,
            height: 360,
            margin: { l: 60, r: 12, t: 10, b: 50 },
            xaxis: {
              title: { text: 'Importance (normalized)' },
              automargin: true,
              showgrid: true,
              zeroline: false,
            },
            yaxis: {
              automargin: true,
              showgrid: false,
              zeroline: false,
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff',
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
        />
      ) : (
        <Text size="sm" c="dimmed" align="center">
          No feature importance available (backend not provided yet).
        </Text>
      )}
    </Box>
  );
}
