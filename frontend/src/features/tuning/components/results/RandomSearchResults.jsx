// frontend/src/components/visualizations/RandomSearchResults.jsx
import Plot from 'react-plotly.js';

function isNumericArray(arr) {
  return (
    Array.isArray(arr) &&
    arr.length > 0 &&
    arr.every((v) => typeof v === 'number' && Number.isFinite(v))
  );
}

export default function RandomSearchResults({
  param1Name,
  param2Name,
  param1Samples,
  param2Samples,
  scores,
  bestPoint,
  textColor,
  gridColor,
  axisColor,
  metricLabel = 'Score',
}) {
  if (
    !param1Samples ||
    !param2Samples ||
    !scores ||
    !param1Samples.length ||
    !param2Samples.length ||
    param1Samples.length !== param2Samples.length ||
    param1Samples.length !== scores.length
  ) {
    return null;
  }

  const numericX = isNumericArray(param1Samples);
  const numericY = isNumericArray(param2Samples);

  let x = param1Samples;
  let y = param2Samples;

  const xaxis = {
    title: { text: param1Name },
    tickfont: { color: textColor },
    titlefont: { color: textColor },
    gridcolor: gridColor,
    linecolor: axisColor,
    mirror: true,
    automargin: true,
  };

  const yaxis = {
    title: { text: param2Name },
    tickfont: { color: textColor },
    titlefont: { color: textColor },
    gridcolor: gridColor,
    linecolor: axisColor,
    mirror: true,
    automargin: true,
  };

  // For categorical axes, map to indices and label ticks
  if (!numericX) {
    const xIdx = param1Samples.map((_, i) => i);
    x = xIdx;
    xaxis.tickmode = 'array';
    xaxis.tickvals = xIdx;
    xaxis.ticktext = param1Samples.map((v) => String(v));
  }

  if (!numericY) {
    const yIdx = param2Samples.map((_, i) => i);
    y = yIdx;
    yaxis.tickmode = 'array';
    yaxis.tickvals = yIdx;
    yaxis.ticktext = param2Samples.map((v) => String(v));
  }

  const scatterTrace = {
    x,
    y,
    mode: 'markers',
    type: 'scatter',
    name: 'Samples',
    marker: {
      size: 9,
      color: scores,
      colorbar: { title: metricLabel },
      showscale: true,
    },
    hovertemplate:
      `${param1Name}: %{customdata[0]}<br>` +
      `${param2Name}: %{customdata[1]}<br>` +
      `${metricLabel}: %{customdata[2]:.3f}<extra></extra>`,
    customdata: param1Samples.map((v1, i) => [
      String(v1),
      String(param2Samples[i]),
      scores[i],
    ]),
  };

  let bestTrace = null;
  if (bestPoint && bestPoint.index != null) {
    const i = bestPoint.index;
    if (i >= 0 && i < param1Samples.length) {
      const bx = numericX ? param1Samples[i] : i;
      const by = numericY ? param2Samples[i] : i;
      bestTrace = {
        x: [bx],
        y: [by],
        type: 'scatter',
        mode: 'markers',
        marker: {
          size: 12,
          symbol: 'x',
          color: 'red',
        },
        name: 'Best example',
        hovertemplate:
          `Best<br>${param1Name}: ${String(param1Samples[i])}<br>` +
          `${param2Name}: ${String(param2Samples[i])}<extra></extra>`,
      };
    }
  }

  const data = bestTrace ? [scatterTrace, bestTrace] : [scatterTrace];

  return (
    <Plot
      data={data}
      layout={{
        title: {
          text: `Randomized search — ${metricLabel} over ${param1Name} × ${param2Name}`,
          font: { color: textColor },
          y: 0.97,
        },
        font: { color: textColor },
        xaxis,
        yaxis,
        margin: { l: 10, r: 10, t: 90, b: 90 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        autosize: true,
        legend: {
          x: 0,
          y: 1.1,
          xanchor: 'left',
          yanchor: 'bottom',
          orientation: 'h',
        },
      }}
      config={{ displaylogo: false, responsive: true }}
      style={{ width: '100%', height: '460px' }}
      useResizeHandler
    />
  );
}
