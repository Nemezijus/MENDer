import Plot from 'react-plotly.js';

export default function ValidationCurveResults({
  plotTraces,
  textColor,
  gridColor,
  axisColor,
  metricLabel = 'Metric',
  paramName,
}) {
  // Ensure both mean curves have markers, and only the key items show in legend
  const adjustedTraces = (plotTraces || []).map((trace) => {
    const t = { ...trace };

    if (t.name === 'Train (mean)' || t.name === 'Validation (mean)') {
      t.mode = 'lines+markers';
    }

    if (['Train (mean)', 'Validation (mean)', 'Recommended value'].includes(t.name)) {
      t.showlegend = true;
    } else {
      t.showlegend = false;
    }

    return t;
  });

  const xTitle = paramName || 'Parameter value';
  const curveLabel = paramName || 'parameter value';
  const yTitle = `${metricLabel} (mean ± SEM)`;

  return (
    <Plot
      data={adjustedTraces}
      layout={{
        title: {
          text: `Validation Curve — ${metricLabel} vs. ${curveLabel}`,
          font: { color: textColor },
          y: 0.97,
        },
        font: { color: textColor },
        xaxis: {
          title: { text: xTitle },
          tickfont: { color: textColor },
          titlefont: { color: textColor },
          gridcolor: gridColor,
          linecolor: axisColor,
          mirror: true,
          automargin: true,
        },
        yaxis: {
          title: { text: yTitle },
          tickfont: { color: textColor },
          titlefont: { color: textColor },
          gridcolor: gridColor,
          linecolor: axisColor,
          mirror: true,
          automargin: true,
        },
        legend: {
          orientation: 'h',
          x: 0.5,
          y: 1,
          xanchor: 'center',
          yanchor: 'bottom',
          font: { color: textColor },
        },
        margin: { l: 10, r: 10, t: 90, b: 90 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        autosize: true,
      }}
      config={{ displaylogo: false, responsive: true }}
      style={{ width: '100%', height: '460px' }}
      useResizeHandler
    />
  );
}
