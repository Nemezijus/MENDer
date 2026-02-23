import { Stack, Text, Table, Group, Card } from '@mantine/core';
import Plot from 'react-plotly.js';

export default function KFoldResults({
    title,
    foldScores,
    metricName,
    meanScore,
    stdScore,
}) {
    if (!Array.isArray(foldScores) || foldScores.length === 0) {
        return null;
    }

    const folds = foldScores;
    const idxs = folds.map((_, i) => i + 1);
    const isNumber = (v) => typeof v === 'number';
    const fmt = (v) => (isNumber(v) ? v.toFixed(3) : v);

    return (
        <Card
            withBorder
            radius="md"
            padding="sm"
            style={{ borderStyle: 'solid', borderWidth: 0 }}
        >
            <Stack gap="xs">
                <Text fw={500} size="sm">{title}</Text>
                <Group align="flex-start" grow wrap="nowrap">
                    {/* ----- TABLE ----- */}
                    <Table
                        withTableBorder
                        withColumnBorders
                        striped
                        maw={260}
                        style={{ width: '20%', flexShrink: 0 }}
                    >
                        <Table.Thead>
                            <Table.Tr>
                                <Table.Th style={{ width: '35%' }}>#</Table.Th>
                                <Table.Th style={{ width: '65%' }}>score</Table.Th>
                            </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                            {folds.map((s, i) => (
                                <Table.Tr key={i}>
                                    <Table.Td>{i + 1}</Table.Td>
                                    <Table.Td>{fmt(s)}</Table.Td>
                                </Table.Tr>
                            ))}
                        </Table.Tbody>
                    </Table>

                    {/* ----- BAR PLOT ----- */}
                    <Plot
                        data={[
                            { type: 'bar', x: idxs, y: folds, name: 'Fold score' },
                            {
                                type: 'scatter',
                                mode: 'lines',
                                x: [0, idxs.length + 1],
                                y: [meanScore, meanScore],
                                name: 'Mean',
                            },
                        ]}
                        layout={{
                            title: 'Fold scores',
                            margin: { l: 50, r: 10, b: 20, t: 10 },
                            xaxis: { title: { text: 'Fold' }, automargin: true, },
                            yaxis: { title: { text: metricName }, automargin: true, },
                            autosize: true,
                            legend: {
                                x: 0.5,       // 0 = left, 1 = right
                                y: 1.15,      // 0 = bottom, 1 = top, >1 = above plot
                                xanchor: 'center',
                                yanchor: 'bottom',
                                orientation: 'h', // 'h' or 'v'
                            },

                        }}
                        style={{ width: '80%', height: 300, maxWidth: 520 }}
                        config={{ displayModeBar: false, responsive: true }}
                    />
                </Group>
            </Stack>
        </Card>
    );
}
