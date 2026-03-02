import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';

import { AXIS_TICK, AXIS_TITLE, PLOT_BG, PLOT_MARGIN_STD } from '../../../utils/plotly.js';
import { LEGEND_TOP_TIGHT } from '../common/styles.js';

export default function DendrogramSection({ dendrogram, clusterLabel }) {
  if (!dendrogram || !Array.isArray(dendrogram?.segments) || !dendrogram.segments.length) return null;

  const traces = dendrogram.segments.map((seg, i) => ({
    type: 'scatter',
    mode: 'lines',
    x: seg.x,
    y: seg.y,
    hoverinfo: 'skip',
    showlegend: false,
    line: { width: 1 },
    name: `seg-${i}`,
  }));

  // Leaf markers (hover + cluster color)
  const leafOrder = Array.isArray(dendrogram?.leaf_order) ? dendrogram.leaf_order : [];
  const leafLabels = Array.isArray(dendrogram?.leaf_labels) ? dendrogram.leaf_labels : leafOrder.map((v) => String(v));
  const leafX = Array.isArray(dendrogram?.leaf_x) ? dendrogram.leaf_x : [];
  const leafClusterIds = Array.isArray(dendrogram?.leaf_cluster_ids) ? dendrogram.leaf_cluster_ids : [];

  const leafCount = Math.min(leafLabels.length, leafX.length);
  if (leafCount > 0 && leafClusterIds.length >= leafCount) {
    const byCluster = new Map();
    for (let i = 0; i < leafCount; i += 1) {
      const cid = Number(leafClusterIds[i]);
      if (!byCluster.has(cid)) byCluster.set(cid, []);
      byCluster.get(cid).push(i);
    }

    Array.from(byCluster.entries())
      .sort((a, b) => a[0] - b[0])
      .forEach(([cid, idxs]) => {
        traces.push({
          type: 'scatter',
          mode: 'markers',
          x: idxs.map((j) => leafX[j]),
          y: idxs.map(() => 0),
          text: idxs.map((j) => leafLabels[j]),
          name: clusterLabel(cid),
          marker: { size: 7, opacity: 0.9 },
          hovertemplate: `Leaf=%{text}<br>Cluster=${clusterLabel(cid)}<extra></extra>`,
        });
      });
  }

  return (
    <Stack gap={4} className="unsupTileStack">
      <PlotHeader
        title="Dendrogram"
        help="Hierarchical clustering dendrogram (AgglomerativeClustering). Leaf markers are colored by the assigned cluster label."
      />
      <Plot
        data={traces}
        layout={{
          margin: { ...PLOT_MARGIN_STD, t: 30, l: 50 },
          xaxis: {
            title: AXIS_TITLE('Leaves'),
            tickfont: { size: 10 },
            showgrid: false,
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
            showticklabels: false,
          },
          yaxis: {
            title: AXIS_TITLE('Distance'),
            tickfont: AXIS_TICK,
            showgrid: true,
            gridcolor: 'rgba(200,200,200,0.4)',
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          ...PLOT_BG,
          hovermode: 'closest',
          legend: LEGEND_TOP_TIGHT,
        }}
        config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
        className="unsupPlotLg"
      />
    </Stack>
  );
}
