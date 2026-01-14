import { useMemo, useState } from 'react';
import {
  Card,
  Stack,
  Text,
  Group,
  Select,
  Button,
  Table,
  ScrollArea,
  Tooltip,
} from '@mantine/core';

import { downloadBlob } from '../../api/models';

function toCsvValue(v) {
  if (v === null || v === undefined) return '';
  const s = String(v);
  if (/[",\n\r]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function mean(vals) {
  if (!vals.length) return null;
  const s = vals.reduce((a, b) => a + b, 0);
  return s / vals.length;
}

function median(vals) {
  if (!vals.length) return null;
  const a = [...vals].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  if (a.length % 2) return a[mid];
  return (a[mid - 1] + a[mid]) / 2;
}

function parseNumber(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const x = Number(v);
    if (Number.isFinite(x)) return x;
  }
  return null;
}

function fmt3(v) {
  if (typeof v === 'number' && Number.isFinite(v)) {
    // Keep integers clean
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(3);
  }
  return '—';
}

function buildCsv(rows, columns) {
  const header = columns.map(toCsvValue).join(',');
  const lines = rows.map((r) =>
    columns.map((c) => toCsvValue(r?.[c])).join(','),
  );
  return [header, ...lines].join('\n');
}

function pickColumns(rows) {
  if (!rows || rows.length === 0) return [];
  const keys = new Set();
  rows.forEach((r) => {
    Object.keys(r || {}).forEach((k) => keys.add(k));
  });

  const preferred = ['index', 'trial_id', 'y_true', 'y_pred', 'correct', 'margin', 'decoder_score'];

  const pCols = [...keys].filter((k) => k.startsWith('p_')).sort();
  const scoreCols = [...keys].filter((k) => k.startsWith('score_')).sort();

  const rest = [...keys]
    .filter(
      (k) =>
        !preferred.includes(k) &&
        !k.startsWith('p_') &&
        !k.startsWith('score_'),
    )
    .sort();

  const out = [];
  preferred.forEach((k) => {
    if (keys.has(k)) out.push(k);
  });
  out.push(...pCols);
  out.push(...scoreCols);
  out.push(...rest);

  return out;
}

function prettifyHeader(key) {
  if (!key) return '';
  // Dynamic probability / score columns
  if (key.startsWith('p_')) return `p=${key.slice(2)}`;
  if (key.startsWith('score_')) return `score=${key.slice(6)}`;
  // Otherwise: underscores -> spaces
  return key.replaceAll('_', ' ');
}

function buildHeaderTooltip(key) {
  if (key === 'index') return 'Row index within the preview (often corresponds to test sample order).';
  if (key === 'trial_id') return 'Optional trial identifier (if provided).';
  if (key === 'y_true') return 'Ground-truth label for this sample.';
  if (key === 'y_pred') return 'Model-predicted label for this sample.';
  if (key === 'correct') return 'Whether prediction matches the ground truth.';
  if (key === 'decoder_score')
    return 'Binary decision value (e.g., decision_function). More positive typically means stronger evidence for the positive class.';
  if (key === 'margin')
    return 'Confidence proxy: for multiclass, usually (top score − runner-up score). Larger margin = more confident.';
  if (key.startsWith('p_')) {
    const c = key.slice(2);
    return `Predicted probability for class ${c}. (Rows sum to ~1 across all p=... columns.)`;
  }
  if (key.startsWith('score_')) {
    const c = key.slice(6);
    return `Raw decision score (logit / decision value) for class ${c}. Higher score usually means higher probability.`;
  }
  return null;
}

export default function DecoderOutputsResults({ trainResult }) {
  if (!trainResult) return null;

  const decoder = trainResult.decoder_outputs || trainResult.decoderOutputs || null;
  const preview = decoder?.preview_rows || decoder?.previewRows || [];

  if (!decoder || !Array.isArray(preview) || preview.length === 0) {
    return null;
  }

  const classes = decoder.classes || [];
  const hasDecisionScores = Boolean(
    decoder.has_decision_scores ?? decoder.hasDecisionScores,
  );
  const hasProbabilities = Boolean(
    decoder.has_proba ??
      decoder.hasProbabilities ??
      decoder.has_probabilities ??
      decoder.hasProba,
  );

  const classOptions = useMemo(() => {
    const opts = [];
    if (Array.isArray(classes)) {
      for (const c of classes) {
        opts.push({ value: String(c), label: String(c) });
      }
    }
    return opts;
  }, [classes]);

  // Default selection: positive class label if present, otherwise first class
  const defaultSelected =
    decoder.positive_class_label != null
      ? String(decoder.positive_class_label)
      : classOptions.length
        ? classOptions[0].value
        : null;

  const [selectedClass, setSelectedClass] = useState(defaultSelected);
  const selectedProbKey = selectedClass ? `p_${selectedClass}` : null;

  const probStats = useMemo(() => {
    if (!hasProbabilities || !selectedProbKey)
      return { mean: null, median: null, n: 0 };

    const vals = preview
      .map((r) => parseNumber(r?.[selectedProbKey]))
      .filter((v) => v !== null);

    return {
      mean: mean(vals),
      median: median(vals),
      n: vals.length,
    };
  }, [hasProbabilities, preview, selectedProbKey]);

  const columns = useMemo(() => pickColumns(preview), [preview]);

  const handleExportPreview = () => {
    const csv = buildCsv(preview, columns);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    downloadBlob(blob, 'decoder_outputs_preview.csv');
  };

  const previewN = Array.isArray(preview) ? preview.length : 0;
  const totalN =
    typeof decoder.n_rows_total === 'number' && Number.isFinite(decoder.n_rows_total)
      ? decoder.n_rows_total
      : null;

  const renderCell = (col, value) => {
    if (value === null || value === undefined) return '—';

    if (col === 'correct') {
      const isTrue = value === true || value === 'true';
      return isTrue ? 'true' : 'false';
    }

    const num = parseNumber(value);
    if (num !== null) return fmt3(num);

    if (typeof value === 'boolean') return value ? 'true' : 'false';
    return String(value);
  };

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500} size="xl" align="center">
          Decoder outputs
        </Text>

        {/* Export button above table, left-aligned */}
        <Group justify="flex-start">
          <Button size="xs" variant="light" onClick={handleExportPreview}>
            Export preview CSV
          </Button>
        </Group>

        {/* Summary lines (similar vibe to ClassificationMetricResults) */}
        <Stack gap={0}>
          <Text size="sm">
            <Text span fw={500}>
              Classes:{' '}
            </Text>
            <Text span fw={700}>
              {Array.isArray(classes) ? classes.length : 0}
            </Text>
          </Text>

          <Text size="sm">
            <Text span fw={500}>
              decision_function:{' '}
            </Text>
            <Text span fw={700}>
              {hasDecisionScores ? 'yes' : 'no'}
            </Text>
          </Text>

          <Text size="sm">
            <Text span fw={500}>
              predict_proba:{' '}
            </Text>
            <Text span fw={700}>
              {hasProbabilities ? 'yes' : 'no'}
            </Text>
          </Text>

          <Text size="sm">
            <Text span fw={500}>
              Rows:{' '}
            </Text>
            <Text span fw={700}>
              {totalN != null ? `${previewN} / ${totalN} previewed` : `${previewN} previewed`}
            </Text>
            <Text span size="xs" c="dimmed">
              {' '}
              (preview only)
            </Text>
          </Text>
        </Stack>

        {Array.isArray(decoder.notes) && decoder.notes.length > 0 && (
          <Stack gap={4}>
            <Text size="sm" fw={500}>
              Notes
            </Text>
            <ul style={{ marginTop: 0 }}>
              {decoder.notes.map((n, i) => (
                <li key={i}>
                  <Text size="sm">{n}</Text>
                </li>
              ))}
            </ul>
          </Stack>
        )}

        {/* Probability summary (preview only) */}
        {hasProbabilities && classOptions.length > 0 && (
          <Card withBorder radius="md" padding="sm">
            <Stack gap="xs">
              <Group justify="space-between" align="flex-end">
                <Select
                  label="Class to summarize"
                  data={classOptions}
                  value={selectedClass}
                  onChange={setSelectedClass}
                  w={240}
                />

                <Stack gap={0} style={{ textAlign: 'right' }}>
                  <Text size="sm">
                    <Text span fw={500}>
                      Mean:{' '}
                    </Text>
                    <Text span fw={700}>
                      {probStats.mean != null ? probStats.mean.toFixed(3) : '—'}
                    </Text>
                  </Text>
                  <Text size="sm">
                    <Text span fw={500}>
                      Median:{' '}
                    </Text>
                    <Text span fw={700}>
                      {probStats.median != null ? probStats.median.toFixed(3) : '—'}
                    </Text>
                  </Text>
                  <Text size="sm" c="dimmed">
                    n: {probStats.n}
                  </Text>
                </Stack>
              </Group>

              <Text size="xs" c="dimmed">
                Summary is computed from the preview rows only.
              </Text>
            </Stack>
          </Card>
        )}

        <ScrollArea h={320} type="auto">
          <Table
            withTableBorder={false}
            withColumnBorders={false}
            horizontalSpacing="xs"
            verticalSpacing="xs"
          >
            <Table.Thead>
              <Table.Tr style={{ backgroundColor: 'var(--mantine-color-gray-8)' }}>
                {columns.map((c) => {
                  const tip = buildHeaderTooltip(c);
                  const label = prettifyHeader(c);
                  return (
                    <Table.Th key={c} style={{ textAlign: 'center' }}>
                      {tip ? (
                        <Tooltip label={tip} multiline maw={260} withArrow>
                          <Text size="xs" fw={600} c="white">
                            {label}
                          </Text>
                        </Tooltip>
                      ) : (
                        <Text size="xs" fw={600} c="white">
                          {label}
                        </Text>
                      )}
                    </Table.Th>
                  );
                })}
              </Table.Tr>
            </Table.Thead>

            <Table.Tbody>
              {preview.map((row, idx) => {
                const isStriped = idx % 2 === 1;
                return (
                  <Table.Tr
                    key={row?.index ?? idx}
                    style={{
                      backgroundColor: isStriped
                        ? 'var(--mantine-color-gray-1)'
                        : 'white',
                    }}
                  >
                    {columns.map((c) => {
                      const val = row?.[c];
                      const isCorrectCol = c === 'correct';
                      const isFalse =
                        isCorrectCol && (val === false || val === 'false');

                      return (
                        <Table.Td
                          key={c}
                          style={{
                            textAlign: 'center',
                            backgroundColor: isFalse
                              ? 'var(--mantine-color-red-1)'
                              : undefined,
                          }}
                        >
                          <Text size="sm">{renderCell(c, val)}</Text>
                        </Table.Td>
                      );
                    })}
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      </Stack>
    </Card>
  );
}
