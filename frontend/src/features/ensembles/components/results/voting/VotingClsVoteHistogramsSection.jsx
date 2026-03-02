import { Card, Group, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import SectionTitle from '../common/SectionTitle.jsx';

import {
  cmBlue,
  rebinHistogram,
  safeNum,
} from '../../../utils/resultsFormat.js';

export default function VotingClsVoteHistogramsSection({ report }) {
  if (!report || report.kind !== 'voting') return null;

  const vote = report.vote || {};
  const marginHist = vote.margin_hist || {};
  const strengthHist = vote.strength_hist || {};

  const marginPlot = (() => {
    const edges = Array.isArray(marginHist.edges) ? marginHist.edges : null;
    const counts = Array.isArray(marginHist.counts) ? marginHist.counts : null;
    if (!edges || !counts || edges.length < 2 || counts.length < 1) return null;

    const e = edges.map((v) => safeNum(v)).filter((v) => v != null);
    if (e.length !== edges.length) return null;

    const binW = e.length >= 2 ? e[1] - e[0] : null;
    const mids = e.slice(0, -1).map((a, i) => (a + e[i + 1]) / 2);

    const trace = {
      type: 'bar',
      x: mids,
      y: counts,
      marker: { color: cmBlue(0.75) },
      width:
        typeof binW === 'number' && Number.isFinite(binW)
          ? mids.map(() => binW * 0.9)
          : undefined,
      hovertemplate: 'margin bin: %{x}<br>count: %{y}<extra></extra>',
    };

    const axis = {
      tickmode: 'linear',
      tick0: typeof binW === 'number' && Number.isFinite(binW) ? mids[0] : undefined,
      dtick: typeof binW === 'number' && Number.isFinite(binW) ? binW : undefined,
    };

    return { trace, axis, xRange: [e[0], e[e.length - 1]] };
  })();

  const strengthPlot = (() => {
    const edges = Array.isArray(strengthHist.edges) ? strengthHist.edges : null;
    const counts = Array.isArray(strengthHist.counts) ? strengthHist.counts : null;
    if (!edges || !counts || edges.length < 2 || counts.length < 1) return null;

    const e = edges.map((v) => safeNum(v));
    if (e.some((v) => v == null)) return null;

    const binCount = counts.length;
    const wantRebin = binCount <= 6;

    let edgesUse = e;
    let countsUse = counts;

    if (wantRebin) {
      const step = 0.05;
      const newEdges = [];
      for (let x = 0; x <= 1 + 1e-9; x += step) newEdges.push(Number(x.toFixed(10)));
      countsUse = rebinHistogram(e, counts, newEdges);
      edgesUse = newEdges;
    }

    const binW = edgesUse.length >= 2 ? edgesUse[1] - edgesUse[0] : null;
    const mids = edgesUse.slice(0, -1).map((a, i) => (a + edgesUse[i + 1]) / 2);

    const N = Number(report.n_estimators) || 0;

    const kOutOfN = mids.map((s) => {
      if (!N) return '—';
      const k = Math.max(0, Math.min(N, Math.round(s * N)));
      return `${k} votes out of ${N}`;
    });

    const trace = {
      type: 'bar',
      x: mids,
      y: countsUse,
      customdata: kOutOfN,
      marker: { color: cmBlue(0.75) },
      width:
        typeof binW === 'number' && Number.isFinite(binW)
          ? mids.map(() => binW * 0.9)
          : undefined,
      hovertemplate:
        '%{customdata}<br>strength: %{x:.3f}<br>count: %{y:.2f}<extra></extra>',
    };

    return { trace, binW };
  })();

  return (
    <Group align="stretch" grow wrap="wrap">
      <Card withBorder={false} radius="md" p="sm" className="ensPlotCol">
        <SectionTitle
          title="Vote margins"
          tooltip="Distribution of vote margins (top votes − runner-up votes). Larger margins mean clearer majorities."
          maw={320}
        />

        {marginPlot?.trace ? (
          <Plot
            data={[marginPlot.trace]}
            layout={{
              autosize: true,
              height: 220,
              margin: { l: 60, r: 12, t: 10, b: 60 },
              xaxis: {
                title: { text: 'Margin (top − runner-up)' },
                automargin: true,
                showgrid: false,
                zeroline: false,
                ...(marginPlot.axis || {}),
                range: marginPlot.xRange || undefined,
              },
              yaxis: {
                title: { text: 'Count' },
                automargin: true,
                showgrid: true,
                zeroline: false,
              },
              bargap: 0.05,
              plot_bgcolor: '#ffffff',
              paper_bgcolor: '#ffffff',
            }}
            config={{ displayModeBar: false, responsive: true }}
            className="ensPlotFullWidth"
          />
        ) : (
          <Text size="sm" c="dimmed" align="center">
            Margin histogram unavailable.
          </Text>
        )}
      </Card>

      <Card withBorder={false} radius="md" p="sm" className="ensPlotCol">
        <SectionTitle
          title="Vote strength"
          tooltip="Distribution of vote strengths (top votes / total estimators). Values closer to 1 mean stronger consensus."
          maw={340}
        />

        {strengthPlot?.trace ? (
          <Plot
            data={[strengthPlot.trace]}
            layout={{
              autosize: true,
              height: 220,
              margin: { l: 60, r: 12, t: 10, b: 60 },
              xaxis: {
                title: { text: 'Strength (top / total)' },
                automargin: true,
                showgrid: false,
                zeroline: false,
                range: [0, 1],
                tickmode: 'linear',
                tick0: 0,
                dtick: 0.1,
              },
              yaxis: {
                title: { text: 'Count' },
                automargin: true,
                showgrid: true,
                zeroline: false,
              },
              bargap: 0.05,
              plot_bgcolor: '#ffffff',
              paper_bgcolor: '#ffffff',
            }}
            config={{ displayModeBar: false, responsive: true }}
            className="ensPlotFullWidth"
          />
        ) : (
          <Text size="sm" c="dimmed" align="center">
            Strength histogram unavailable.
          </Text>
        )}
      </Card>
    </Group>
  );
}
