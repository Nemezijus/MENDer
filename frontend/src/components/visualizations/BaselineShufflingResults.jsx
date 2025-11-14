import { Stack, Text, Card } from '@mantine/core';
import Plot from 'react-plotly.js';

export default function BaselineShufflingResults({
    title = 'Shuffle-label baseline',
    metricName,
    referenceLabel = 'real',
    referenceValue,
    shuffledScores,
    pValue,
    digits = 3,
}) {
    if (!Array.isArray(shuffledScores) || shuffledScores.length === 0) {
        return null;
    }

    const isNumber = (v) => typeof v === 'number';
    const fmt = (v) => (isNumber(v) ? v.toFixed(digits) : v);

    return (
        <Card
            withBorder
            radius="md"
            padding="sm"
            style={{ borderStyle: 'solid', borderWidth: 0 }}
        >
            <Stack gap="xs">
                <Text fw={500} size="sm">{title}</Text>
                <Plot
                    data={[
                        {
                            type: 'histogram',
                            x: shuffledScores,
                            opacity: 0.75,
                            name: 'Shuffled',
                        },
                    ]}
                    layout={{
                        bargap: 0.05,
                        xaxis: { title: { text: metricName }, automargin: true, },
                        yaxis: { title: { text: 'Count' }, automargin: true, },
                        autosize: true,
                        shapes: [
                            {
                                type: 'line',
                                x0: referenceValue,
                                x1: referenceValue,
                                y0: 0,
                                y1: 1,
                                yref: 'paper',
                                line: { width: 2 },
                            },
                        ],
                        annotations:
                            pValue != null
                                ? [
                                    {
                                        x: referenceValue,
                                        y: 1,
                                        yref: 'paper',
                                        text: `${referenceLabel} = ${fmt(referenceValue)} · p≈${Number(pValue).toFixed(digits)}`,
                                        showarrow: false,
                                        xanchor: 'left',
                                        align: 'left',
                                    },
                                ]
                                : [],
                        margin: { t: 10, r: 10, b: 20, l: 50 },
                        height: 260,

                        legend: {
                            x: 0.5,       // 0 = left, 1 = right
                            y: 1.15,      // 0 = bottom, 1 = top, >1 = above plot
                            xanchor: 'center',
                            yanchor: 'bottom',
                            orientation: 'h', // 'h' or 'v'
                        },
                    }}
                    config={{ displayModeBar: false }}
                    style={{ width: '100%', maxWidth: 520 }}
                />
            </Stack>
        </Card>
    );
}
