import Plot from 'react-plotly.js';

function isNumericArray(arr) {
  return (
    Array.isArray(arr) &&
    arr.length > 0 &&
    arr.every((v) => typeof v === 'number' && Number.isFinite(v))
  );
}

export default function GridSearchResults({
  param1Name,
  param2Name,
  param1Values,
  param2Values,
  meanScores,
  bestPoint,
  textColor,
  gridColor,
  axisColor,
  metricLabel = 'Score',
}) {
  if (
    !param1Values ||
    !param2Values ||
    !meanScores ||
    !param1Values.length ||
    !param2Values.length
  ) {
    return null;
  }

  const numericX = isNumericArray(param1Values);
  const numericY = isNumericArray(param2Values);

  // Default axis configs
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

  let dataTrace;
  let bestTrace = null;

  if (numericX && numericY) {
    // --- TRUE CONTOUR PLOT (both axes numeric) ---
    dataTrace = {
      x: param1Values,
      y: param2Values,
      z: meanScores,
      type: 'contour',
      contours: {
        coloring: 'heatmap',
        showlabels: true,
        showlines: true,
      },
      colorbar: {
        title: metricLabel,
      },
      hovertemplate:
        `${param1Name}: %{x}<br>` +
        `${param2Name}: %{y}<br>` +
        `${metricLabel}: %{z:.3f}<extra></extra>`,
    };

    if (bestPoint && bestPoint.x != null && bestPoint.y != null) {
      bestTrace = {
        x: [bestPoint.x],
        y: [bestPoint.y],
        type: 'scatter',
        mode: 'markers',
        name: 'Best',
        marker: {
          size: 10,
          symbol: 'x',
        },
        hovertemplate:
          `Best<br>${param1Name}: %{x}<br>${param2Name}: %{y}<extra></extra>`,
      };
    }
  } else {
    // --- FALLBACK: HEATMAP OVER INDEXED CATEGORIES ---
    const xIdx = param1Values.map((_, i) => i);
    const yIdx = param2Values.map((_, i) => i);

    dataTrace = {
      x: xIdx,
      y: yIdx,
      z: meanScores,
      type: 'heatmap',
      colorbar: {
        title: metricLabel,
      },
      hovertemplate:
        `${param1Name}: %{customdata[0]}<br>` +
        `${param2Name}: %{customdata[1]}<br>` +
        `${metricLabel}: %{z:.3f}<extra></extra>`,
      customdata: meanScores.map((row, j) =>
        row.map((_, i) => [
          String(param1Values[i]),
          String(param2Values[j]),
        ]),
      ),
    };

    // categorical ticks
    xaxis.tickmode = 'array';
    xaxis.tickvals = xIdx;
    xaxis.ticktext = param1Values.map((v) => String(v));

    yaxis.tickmode = 'array';
    yaxis.tickvals = yIdx;
    yaxis.ticktext = param2Values.map((v) => String(v));

    if (bestPoint && bestPoint.x != null && bestPoint.y != null) {
      const bx = param1Values.findIndex(
        (v) => String(v) === String(bestPoint.x),
      );
      const by = param2Values.findIndex(
        (v) => String(v) === String(bestPoint.y),
      );
      if (bx !== -1 && by !== -1) {
        bestTrace = {
          x: [bx],
          y: [by],
          type: 'scatter',
          mode: 'markers',
          name: 'Best',
          marker: {
            size: 10,
            symbol: 'x',
          },
          hovertemplate:
            `Best<br>${param1Name}: ${String(bestPoint.x)}<br>` +
            `${param2Name}: ${String(bestPoint.y)}<extra></extra>`,
        };
      }
    }
  }

  const data = bestTrace ? [dataTrace, bestTrace] : [dataTrace];

  return (
    <Plot
      data={data}
      layout={{
        title: {
          text: `Grid search — ${metricLabel} over ${param1Name} × ${param2Name}`,
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
      }}
      config={{ displaylogo: false, responsive: true }}
      style={{ width: '100%', height: '460px' }}
      useResizeHandler
    />
  );
}
