import { Box, Group, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import {
  HEATMAP_COLORSCALE,
  fmt,
  makeUniqueLabels,
  niceEstimatorLabel,
  prettyEstimatorName,
  safeNum,
} from '../../../utils/resultsFormat.js';

function algoKeyToAbbrev(algoKey) {
  if (!algoKey) return 'UNK';
  return prettyEstimatorName(String(algoKey), { task: 'regression' });
}

function buildLegendLines(estimators) {
  const seen = new Map(); // abbrev -> full label
  (estimators || []).forEach((e) => {
    const key = String(e?.algo || '').toLowerCase();
    if (!key) return;
    const ab = algoKeyToAbbrev(key);
    const full = niceEstimatorLabel({ name: e?.name, algo: e?.algo }, { fallback: key });
    if (!seen.has(ab)) seen.set(ab, full);
  });

  const entries = Array.from(seen.entries())
    .map(([ab, full]) => ({ ab, full }))
    .sort((a, b) => a.full.localeCompare(b.full));

  return entries;
}

export default function VotingRegPairwiseMatricesSection({ report }) {
  if (!report || report.kind !== 'voting') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const estimators = Array.isArray(report.estimators) ? report.estimators : [];

  const baseAbbrev = estimators.map((e) => algoKeyToAbbrev(e?.algo));
  const labelsPretty = makeUniqueLabels(baseAbbrev);

  const similarity = report.similarity || {};
  const corrRaw = Array.isArray(similarity.pairwise_corr) ? similarity.pairwise_corr : null;
  const absRaw = Array.isArray(similarity.pairwise_absdiff) ? similarity.pairwise_absdiff : null;

  const legendEntries = buildLegendLines(estimators);

  const corrTrace = (() => {
    if (!corrRaw) return null;
    const zColor = corrRaw.map((row) =>
      row.map((v) => {
        const n = safeNum(v);
        if (n == null) return null;
        return (n + 1) / 2;
      }),
    );

    const text = corrRaw.map((row) =>
      row.map((v) => {
        const n = safeNum(v);
        return n == null ? '' : n.toFixed(2);
      }),
    );

    return {
      type: 'heatmap',
      x: labelsPretty,
      y: labelsPretty,
      z: zColor,
      zmin: 0,
      zmax: 1,
      colorscale: HEATMAP_COLORSCALE,
      showscale: true,
      colorbar: {
        x: 1.02,
        xanchor: 'left',
        xpad: 0,
        thickness: 12,
        len: 0.92,
        outlinewidth: 0,
        tickvals: [0, 0.5, 1],
        ticktext: ['-1', '0', '1'],
      },
      text,
      texttemplate: '%{text}',
      hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>corr: %{text}<extra></extra>',
    };
  })();

  const absTrace = (() => {
    if (!absRaw) return null;
    const flat = absRaw
      .flat()
      .map((v) => safeNum(v))
      .filter((v) => v != null);
    const zmax = flat.length ? Math.max(...flat) : 1;

    const text = absRaw.map((row) =>
      row.map((v) => {
        const n = safeNum(v);
        return n == null ? '' : n.toFixed(2);
      }),
    );

    return {
      type: 'heatmap',
      x: labelsPretty,
      y: labelsPretty,
      z: absRaw,
      zmin: 0,
      zmax: zmax || 1,
      colorscale: HEATMAP_COLORSCALE,
      showscale: true,
      colorbar: {
        x: 1.02,
        xanchor: 'left',
        xpad: 0,
        thickness: 12,
        len: 0.92,
        outlinewidth: 0,
      },
      text,
      texttemplate: '%{text}',
      hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>|Δ|: %{z:.4f}<extra></extra>',
    };
  })();

  return (
    <Box>
      <SectionTitle
        title="Pairwise prediction structure"
        tooltip="Pairwise relationships between base estimators. Correlation close to 1 means very similar predictions; |Δ| highlights how far predictions differ on average."
        maw={420}
      />

      <Group align="stretch" grow wrap="wrap">
        <Box style={{ flex: 1, minWidth: 320 }}>
          <Text size="md" fw={500} align="center" mb={6}>
            Prediction similarity (corr)
          </Text>

          {corrTrace ? (
            <Plot
              data={[corrTrace]}
              layout={{
                autosize: true,
                height: 360,
                margin: { l: 80, r: 36, t: 10, b: 90 },
                xaxis: {
                  title: { text: 'Estimator' },
                  tickangle: -30,
                  side: 'top',
                  automargin: true,
                  showgrid: false,
                  zeroline: false,
                  constrain: 'domain',
                },
                yaxis: {
                  title: { text: 'Estimator' },
                  autorange: 'reversed',
                  automargin: true,
                  showgrid: false,
                  zeroline: false,
                  scaleanchor: 'x',
                  scaleratio: 1,
                  constrain: 'domain',
                },
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#ffffff',
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          ) : (
            <Text size="sm" c="dimmed" align="center">
              Correlation matrix unavailable.
            </Text>
          )}
        </Box>

        <Box style={{ flex: 1, minWidth: 320 }}>
          <Text size="md" fw={500} align="center" mb={6}>
            Absolute prediction differences (|Δ|)
          </Text>

          {absTrace ? (
            <Plot
              data={[absTrace]}
              layout={{
                autosize: true,
                height: 360,
                margin: { l: 80, r: 36, t: 10, b: 90 },
                xaxis: {
                  title: { text: 'Estimator' },
                  tickangle: -30,
                  side: 'top',
                  automargin: true,
                  showgrid: false,
                  zeroline: false,
                  constrain: 'domain',
                },
                yaxis: {
                  title: { text: 'Estimator' },
                  autorange: 'reversed',
                  automargin: true,
                  showgrid: false,
                  zeroline: false,
                  scaleanchor: 'x',
                  scaleratio: 1,
                  constrain: 'domain',
                },
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#ffffff',
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          ) : (
            <Text size="sm" c="dimmed" align="center">
              Absolute-difference matrix unavailable.
            </Text>
          )}
        </Box>
      </Group>

      {legendEntries.length > 0 && (
        <Box mt="xs">
          <Text size="sm" c="dimmed" align="center">
            {legendEntries.map((e, idx) => (
              <span key={e.ab}>
                <b>{e.ab}</b> — {e.full}
                {idx < legendEntries.length - 1 ? '   •   ' : ''}
              </span>
            ))}
          </Text>
        </Box>
      )}
    </Box>
  );
}
