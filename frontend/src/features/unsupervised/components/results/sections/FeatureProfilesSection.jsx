import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';

import {
  AXIS_TITLE,
  MENDER_BLUE_SCALE,
  PLOT_BG,
  PLOT_MARGIN_STD,
} from '../../../utils/plotly.js';



export default function FeatureProfilesSection({ centroids, clusterLabel }) {
  if (!centroids) return null;
  const clusterIds = centroids?.cluster_ids;
  const featIdx = centroids?.feature_idx;
  const z = centroids?.values;
  if (!Array.isArray(clusterIds) || !Array.isArray(featIdx) || !Array.isArray(z) || !z.length) return null;

  const featureLabels = featIdx.map((i) => `f${i}`);
  const showFeatureTicks = featureLabels.length <= 25;

  return (
    <Stack gap={4} className="unsupTileStack">
      <PlotHeader title="Per-cluster feature profiles" help="Mean feature values per cluster (shown for top-variance features)." />

      <div className="unsupSquareOuter">
        <div className="unsupSquareInner">
          <Plot
            data={[
              {
                type: 'heatmap',
                z,
                x: featureLabels,
                y: clusterIds.map((c) => clusterLabel(c)),
                hovertemplate: 'Cluster=%{y}<br>Feature=%{x}<br>Value=%{z:.3f}<extra></extra>',
                colorscale: MENDER_BLUE_SCALE,
                showscale: false,
              },
            ]}
            layout={{
              margin: { ...PLOT_MARGIN_STD, l: 60, t: 14, b: 50 },
              xaxis: {
                title: AXIS_TITLE('Features'),
                tickfont: { size: 10 },
                showgrid: false,
                zeroline: false,
                showline: true,
                mirror: true,
                linecolor: '#e5e7eb',
                linewidth: 1,
                ticks: 'outside',
                showticklabels: showFeatureTicks,
                tickangle: showFeatureTicks ? -45 : 0,
                automargin: true,
              },
              yaxis: {
                title: { text: '' },
                tickfont: { size: 11 },
                showgrid: false,
                zeroline: false,
                showline: true,
                mirror: true,
                linecolor: '#e5e7eb',
                linewidth: 1,
                ticks: 'outside',
                ticklen: 3,
                showticklabels: true,
                automargin: true,
              },
              ...PLOT_BG,
            }}
            config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
            className="unsupPlotFill"
          />
        </div>
      </div>
    </Stack>
  );
}
