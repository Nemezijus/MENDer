import { Box, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';
import { safeNum } from '../../../utils/resultsFormat.js';

function pickTopKeys(obj, maxKeys = 3) {
  if (!obj || typeof obj !== 'object') return [];
  const keys = Object.keys(obj);
  // Prefer validation metrics over training if both exist
  const score = (k) => {
    const kk = k.toLowerCase();
    if (kk.includes('validation_1')) return 3;
    if (kk.includes('validation_0')) return 2;
    if (kk.includes('train')) return 1;
    return 0;
  };
  return keys
    .sort((a, b) => score(b) - score(a) || a.localeCompare(b))
    .slice(0, maxKeys);
}

function prettyCurveName(key) {
  const k = String(key || '');

  // Expected patterns from backend aggregation (e.g. "validation_0:mlogloss")
  const m = k.match(/^validation_(\d+):(.+)$/);
  if (!m) return k;

  const idx = Number(m[1]);
  const metric = m[2];

  let setLabel = `Eval set ${idx}`;
  if (idx === 0) setLabel = 'Train (internal)';
  if (idx === 1) setLabel = 'Validation (internal)';

  return `${setLabel} • ${metric}`;
}

export default function XGBLearningCurvesSection({ report }) {
  if (!report || report.kind !== 'xgboost') return null;

  const curves = report.learning_curves || null;

  const curveKeys = pickTopKeys(curves, 3);

  const curveTraces = [];
  if (curves && curveKeys.length > 0) {
    curveKeys.forEach((k) => {
      const item = curves[k];
      const mean = Array.isArray(item?.mean) ? item.mean : null;
      const std = Array.isArray(item?.std) ? item.std : null;
      if (!mean || mean.length === 0) return;

      const x = Array.from({ length: mean.length }, (_, i) => i + 1);
      const prettyName = prettyCurveName(k);

      curveTraces.push({
        type: 'scatter',
        mode: 'lines',
        name: prettyName,
        x,
        y: mean,
        line: { width: 2 },
        hovertemplate: `round: %{x}<br>${prettyName}: %{y:.5f}<extra></extra>`,
      });

      // Optional std band
      if (std && std.length === mean.length) {
        const upper = mean.map((v, i) => safeNum(v) + (safeNum(std[i]) ?? 0));
        const lower = mean.map((v, i) => safeNum(v) - (safeNum(std[i]) ?? 0));

        curveTraces.push({
          type: 'scatter',
          mode: 'lines',
          name: `${prettyName} +std`,
          x,
          y: upper,
          line: { width: 0 },
          showlegend: false,
          hoverinfo: 'skip',
        });
        curveTraces.push({
          type: 'scatter',
          mode: 'lines',
          name: `${prettyName} -std`,
          x,
          y: lower,
          line: { width: 0 },
          fill: 'tonexty',
          fillcolor: 'rgba(0,0,0,0.08)',
          showlegend: false,
          hoverinfo: 'skip',
        });
      }
    });
  }

  return (
    <Box>
      <SectionTitle
        title="Learning curves"
        tooltip="Mean learning curves aggregated across folds (aligned to shortest curve). Shaded area indicates ±1 std."
        maw={420}
      />

      {curveTraces.length > 0 ? (
        <Plot
          data={curveTraces}
          layout={{
            autosize: true,
            height: 360,
            margin: { l: 60, r: 12, t: 10, b: 60 },
            xaxis: {
              title: { text: 'Boosting round' },
              automargin: true,
              showgrid: false,
              zeroline: false,
            },
            yaxis: {
              title: { text: 'Metric' },
              automargin: true,
              showgrid: true,
              zeroline: false,
            },
            legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.2 },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff',
          }}
          config={{ displayModeBar: false, responsive: true }}
          className="ensPlotFullWidth"
        />
      ) : (
        <Text size="sm" c="dimmed" align="center">
          No learning curves available (evals_result not provided by backend).
        </Text>
      )}
    </Box>
  );
}
