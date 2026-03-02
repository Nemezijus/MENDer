import Plot from 'react-plotly.js';

export default function LearningCurveResults({
  plotTraces,
  textColor,
  gridColor,
  axisColor,
  metricLabel = 'Accuracy',
}) {
  // Ensure both mean curves have markers, and only the 3 main items appear in legend
  const adjustedTraces = (plotTraces || []).map((trace) => {
    const t = { ...trace };

    // Make sure both means use lines+markers
    if (t.name === 'Train (mean)' || t.name === 'Validation (mean)') {
      t.mode = 'lines+markers';
    }

    // Legend control: show only these three
    if (['Train (mean)', 'Validation (mean)', 'Recommended size'].includes(t.name)) {
      t.showlegend = true;
    } else {
      t.showlegend = false;
    }

    return t;
  });

  const yTitle = `${metricLabel} (mean ± SEM)`;

  return (
    <Plot
      data={adjustedTraces}
      layout={{
        title: {
          text: `Learning Curve — ${metricLabel} vs. Training Set Size`,
          font: { color: textColor },
          // Slightly lower in the figure to separate from legend
          y: 0.97,
        },
        font: { color: textColor },
        xaxis: {
          title: { text: 'Training size (samples)' },
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
          // range removed so Plotly can auto-range for arbitrary metrics
        },
        legend: {
          orientation: 'h',
          x: 0.5,
          y: 1,             // move legend a bit higher above plot
          xanchor: 'center',
          yanchor: 'bottom',
          font: { color: textColor },
        },
        // More top margin so title + legend have breathing room
        margin: { l: 10, r: 10, t: 90, b: 90 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        autosize: true,
      }}
      config={{ displaylogo: false, responsive: true }}
      className="tuningPlotlyChart"
      useResizeHandler
    />
  );
}
