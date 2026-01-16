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
  Divider,
} from '@mantine/core';

import { downloadBlob } from '../../api/models';

function toCsvValue(v) {
  if (v === null || v === undefined) return '';
  const s = String(v);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function mean(vals) {
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function median(vals) {
  if (!vals.length) return null;
  const a = [...vals].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
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
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(3);
  }
  return '—';
}

function fmtMaybe3(v) {
  if (v === null || v === undefined) return '—';
  const num = parseNumber(v);
  if (num === null) return String(v);
  return fmt3(num);
}

function fmtMaybePct(v) {
  const num = parseNumber(v);
  if (num === null) return '—';
  return `${(num * 100).toFixed(1)}%`;
}

function buildCsv(rows, columns) {
  const header = columns.map(toCsvValue).join(',');
  const lines = rows.map((r) => columns.map((c) => toCsvValue(r?.[c])).join(','));
  return [header, ...lines].join('\n');
}

function pickColumns(rows) {
  if (!rows || rows.length === 0) return [];
  const keys = new Set();
  rows.forEach((r) => Object.keys(r || {}).forEach((k) => keys.add(k)));

  // Keep key columns first; fold_id (when present) should appear right after index.
  const preferred = [
    'index',
    'fold_id',
    'trial_id',
    'y_true',
    'y_pred',
    'correct',
    'margin',
    'decoder_score',
  ];

  const pCols = [...keys].filter((k) => k.startsWith('p_')).sort();
  const scoreCols = [...keys].filter((k) => k.startsWith('score_')).sort();
  const rest = [...keys]
    .filter((k) => !preferred.includes(k) && !k.startsWith('p_') && !k.startsWith('score_'))
    .sort();

  const out = [];
  preferred.forEach((k) => keys.has(k) && out.push(k));
  out.push(...pCols, ...scoreCols, ...rest);
  return out;
}

function prettifyHeader(key) {
  if (!key) return '';
  if (key.startsWith('p_')) return `p=${key.slice(2)}`;
  if (key.startsWith('score_')) return `score=${key.slice(6)}`;
  return key.replaceAll('_', ' ');
}

function buildHeaderTooltip(key) {
  if (key === 'index') return 'Row index within the preview (often corresponds to test sample order).';
  if (key === 'fold_id') return 'Fold index for this row in k-fold CV (1..K).';
  if (key === 'trial_id') return 'Optional trial identifier (if provided).';
  if (key === 'y_true') return 'Ground-truth label for this sample.';
  if (key === 'y_pred') return 'Model-predicted label for this sample.';
  if (key === 'correct') return 'Whether prediction matches the ground truth.';
  if (key === 'decoder_score')
    return 'Binary decision value (decision_function). More positive typically means stronger evidence for the positive class.';
  if (key === 'margin')
    return 'Confidence proxy: usually (top score − runner-up score). Larger margin = more confident.';
  if (key.startsWith('p_')) {
    const c = key.slice(2);
    return `Predicted probability for class ${c}. (Rows sum to ~1 across all p=... columns.)`;
  }
  if (key.startsWith('score_')) {
    const c = key.slice(6);
    return `Raw decision score (logit/decision value) for class ${c}. Higher score usually means higher probability.`;
  }
  return null;
}

function getSummary(decoder) {
  if (!decoder) return null;
  return (
    decoder.summary ||
    decoder.decoder_summary ||
    decoder.decoderSummary ||
    decoder.decoderSummaries ||
    null
  );
}

function firstKey(obj, candidates) {
  if (!obj) return null;
  for (const k of candidates) {
    if (Object.prototype.hasOwnProperty.call(obj, k)) return k;
  }
  return null;
}

export default function DecoderOutputsResults({ trainResult }) {
  if (!trainResult) return null;

  const decoder = trainResult.decoder_outputs || trainResult.decoderOutputs || null;
  const preview = decoder?.preview_rows || decoder?.previewRows || [];

  if (!decoder || !Array.isArray(preview) || preview.length === 0) return null;

  const classes = decoder.classes || [];
  const hasDecisionScores = Boolean(decoder.has_decision_scores ?? decoder.hasDecisionScores);
  const hasProbabilities = Boolean(
    decoder.has_proba ??
      decoder.hasProbabilities ??
      decoder.has_probabilities ??
      decoder.hasProba,
  );
  const probaSource =
    decoder.proba_source ?? decoder.probaSource ?? decoder.proba_source ?? null;

  const isVoteShare = String(probaSource || '').toLowerCase() === 'vote_share';
  const summary = useMemo(() => getSummary(decoder), [decoder]);

  // De-dupe decoder notes against global trainResult.notes (often prefixed "Decoder outputs: ...")
  const decoderNotes = useMemo(() => {
    const dn = Array.isArray(decoder.notes) ? decoder.notes : [];
    const tn = Array.isArray(trainResult.notes) ? trainResult.notes : [];
    return dn.filter((n) => !tn.includes(n) && !tn.includes(`Decoder outputs: ${n}`));
  }, [decoder.notes, trainResult.notes]);

  const classOptions = useMemo(() => {
    if (!Array.isArray(classes)) return [];
    return classes.map((c) => ({ value: String(c), label: String(c) }));
  }, [classes]);

  const defaultSelected =
    decoder.positive_class_label != null
      ? String(decoder.positive_class_label)
      : classOptions.length
        ? classOptions[0].value
        : null;

  const [selectedClass, setSelectedClass] = useState(defaultSelected);
  const selectedProbKey = selectedClass ? `p_${selectedClass}` : null;

  const probStats = useMemo(() => {
    if (!hasProbabilities || !selectedProbKey) return { mean: null, median: null, n: 0 };
    const vals = preview.map((r) => parseNumber(r?.[selectedProbKey])).filter((v) => v !== null);
    return { mean: mean(vals), median: median(vals), n: vals.length };
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

  const stickyThStyle = {
    position: 'sticky',
    top: 0,
    zIndex: 2,
    backgroundColor: 'var(--mantine-color-gray-8)',
    textAlign: 'center',
    whiteSpace: 'nowrap',
  };

  const cellNowrap = { whiteSpace: 'nowrap' };
  const headerTextStyle = { ...cellNowrap, lineHeight: 1.1 };

  // Pick “nice” threshold keys if present (decoder_summaries may emit multiple)
  const marginFracKey = summary
    ? firstKey(summary, ['margin_frac_lt_0_1', 'margin_frac_lt_0_05'])
    : null;
  const maxProbaFracKey = summary
    ? firstKey(summary, ['max_proba_frac_ge_0_9', 'max_proba_frac_ge_0_8'])
    : null;

  // Calibration fields
  const eceBins =
    summary && Array.isArray(summary.reliability_bins) ? summary.reliability_bins : null;

  const nonEmptyBins = useMemo(() => {
    if (!eceBins) return [];
    return eceBins.filter((b) => b && typeof b.count === 'number' && b.count > 0);
  }, [eceBins]);

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500} size="xl" align="center">
          Decoder outputs
        </Text>

        {/* Basic summary */}
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
              {hasProbabilities && isVoteShare ? (
                <Text span fw={500} c="dimmed">
                  {' '}
                  (vote share)
                </Text>
              ) : null}
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

        {/* Global decoder summary (from backend, typically OOF when CV is used) */}
        {summary && typeof summary === 'object' && (
          <Card withBorder radius="md" padding="sm">
            <Stack gap="xs">
              <Group justify="space-between" align="center">
                <Text size="sm" fw={600}>
                  Global decoder summary
                </Text>
                <Text size="xs" c="dimmed">
                  (computed from full decoder outputs)
                </Text>
              </Group>

              <Divider />

              <Group justify="space-between" wrap="wrap" gap="xs">
                {'n_samples' in summary && (
                  <Text size="sm">
                    <Text span fw={500}>
                      n:
                    </Text>{' '}
                    <Text span fw={700}>
                      {summary.n_samples}
                    </Text>
                  </Text>
                )}

                {'log_loss' in summary && (
                  <Tooltip
                    label="Cross-entropy (log loss). Sensitive to confidence changes even when accuracy is similar."
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span fw={500}>
                        log loss:
                      </Text>{' '}
                      <Text span fw={700}>
                        {fmtMaybe3(summary.log_loss)}
                      </Text>
                    </Text>
                  </Tooltip>
                )}

                {'brier' in summary && (
                  <Tooltip
                    label="Brier score (mean squared error of probabilities vs one-hot truth). Lower is better."
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span fw={500}>
                        brier:
                      </Text>{' '}
                      <Text span fw={700}>
                        {fmtMaybe3(summary.brier)}
                      </Text>
                    </Text>
                  </Tooltip>
                )}

                {'ece' in summary && (
                  <Tooltip
                    label="Expected Calibration Error (ECE) using top-1 confidence. Lower is better; 0 means perfect calibration."
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span fw={500}>
                        ECE:
                      </Text>{' '}
                      <Text span fw={700}>
                        {fmtMaybe3(summary.ece)}
                      </Text>
                    </Text>
                  </Tooltip>
                )}

                {'mce' in summary && (
                  <Tooltip
                    label="Maximum Calibration Error (MCE): the largest |accuracy - confidence| among bins."
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span fw={500}>
                        MCE:
                      </Text>{' '}
                      <Text span fw={700}>
                        {fmtMaybe3(summary.mce)}
                      </Text>
                    </Text>
                  </Tooltip>
                )}

                {'ece_n_bins' in summary && (
                  <Text size="sm">
                    <Text span fw={500}>
                      bins:
                    </Text>{' '}
                    <Text span fw={700}>
                      {summary.ece_n_bins}
                    </Text>
                  </Text>
                )}

                {'margin_mean' in summary && (
                  <Tooltip
                    label="Mean margin (top1 − top2). Larger means more separation/confidence."
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span fw={500}>
                        margin mean:
                      </Text>{' '}
                      <Text span fw={700}>
                        {fmtMaybe3(summary.margin_mean)}
                      </Text>
                    </Text>
                  </Tooltip>
                )}

                {'margin_median' in summary && (
                  <Text size="sm">
                    <Text span fw={500}>
                      margin median:
                    </Text>{' '}
                    <Text span fw={700}>
                      {fmtMaybe3(summary.margin_median)}
                    </Text>
                  </Text>
                )}

                {marginFracKey && (
                  <Tooltip
                    label={`Fraction of samples with margin below threshold (${marginFracKey
                      .replace('margin_frac_lt_', '')
                      .replaceAll('_', '.')}).`}
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span fw={500}>
                        low margin:
                      </Text>{' '}
                      <Text span fw={700}>
                        {fmtMaybePct(summary[marginFracKey])}
                      </Text>
                    </Text>
                  </Tooltip>
                )}

                {'max_proba_mean' in summary && (
                  <Tooltip
                    label="Mean of max predicted probability per sample (confidence proxy)."
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span fw={500}>
                        max p mean:
                      </Text>{' '}
                      <Text span fw={700}>
                        {fmtMaybe3(summary.max_proba_mean)}
                      </Text>
                    </Text>
                  </Tooltip>
                )}

                {maxProbaFracKey && (
                  <Tooltip
                    label={`Fraction of samples with max probability ≥ threshold (${maxProbaFracKey
                      .replace('max_proba_frac_ge_', '')
                      .replaceAll('_', '.')}).`}
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span fw={500}>
                        high conf:
                      </Text>{' '}
                      <Text span fw={700}>
                        {fmtMaybePct(summary[maxProbaFracKey])}
                      </Text>
                    </Text>
                  </Tooltip>
                )}
              </Group>

              {/* Reliability bins (compact) */}
              {nonEmptyBins.length > 0 && (
                <>
                  <Divider my="xs" />

                  <Group justify="space-between" align="center">
                    <Text size="sm" fw={600}>
                      Calibration bins (top-1 confidence)
                    </Text>
                    <Text size="xs" c="dimmed">
                      Shows only non-empty bins
                    </Text>
                  </Group>

                  <ScrollArea h={160} type="auto" offsetScrollbars>
                    <Table
                      withTableBorder={false}
                      withColumnBorders={false}
                      horizontalSpacing="xs"
                      verticalSpacing="xs"
                    >
                      <Table.Thead>
                        <Table.Tr>
                          <Table.Th>
                            <Text size="xs" fw={600}>
                              bin
                            </Text>
                          </Table.Th>
                          <Table.Th>
                            <Text size="xs" fw={600}>
                              range
                            </Text>
                          </Table.Th>
                          <Table.Th style={{ textAlign: 'right' }}>
                            <Text size="xs" fw={600}>
                              n
                            </Text>
                          </Table.Th>
                          <Table.Th style={{ textAlign: 'right' }}>
                            <Text size="xs" fw={600}>
                              conf
                            </Text>
                          </Table.Th>
                          <Table.Th style={{ textAlign: 'right' }}>
                            <Text size="xs" fw={600}>
                              acc
                            </Text>
                          </Table.Th>
                          <Table.Th style={{ textAlign: 'right' }}>
                            <Text size="xs" fw={600}>
                              gap
                            </Text>
                          </Table.Th>
                        </Table.Tr>
                      </Table.Thead>
                      <Table.Tbody>
                        {nonEmptyBins.map((b) => (
                          <Table.Tr key={b.bin}>
                            <Table.Td>
                              <Text size="sm">{b.bin}</Text>
                            </Table.Td>
                            <Table.Td>
                              <Text size="sm">
                                {fmtMaybe3(b.bin_lo)}–{fmtMaybe3(b.bin_hi)}
                              </Text>
                            </Table.Td>
                            <Table.Td style={{ textAlign: 'right' }}>
                              <Text size="sm">{b.count}</Text>
                            </Table.Td>
                            <Table.Td style={{ textAlign: 'right' }}>
                              <Text size="sm">{fmtMaybe3(b.avg_confidence)}</Text>
                            </Table.Td>
                            <Table.Td style={{ textAlign: 'right' }}>
                              <Text size="sm">{fmtMaybe3(b.accuracy)}</Text>
                            </Table.Td>
                            <Table.Td style={{ textAlign: 'right' }}>
                              <Text size="sm">{fmtMaybe3(b.gap)}</Text>
                            </Table.Td>
                          </Table.Tr>
                        ))}
                      </Table.Tbody>
                    </Table>
                  </ScrollArea>
                </>
              )}
            </Stack>
          </Card>
        )}

        {/* Notes (deduped) */}
        {decoderNotes.length > 0 && (
          <Stack gap={4}>
            <Text size="sm" fw={500}>
              Notes
            </Text>
            <ul style={{ marginTop: 0 }}>
              {decoderNotes.map((n, i) => (
                <li key={i}>
                  <Text size="sm">{n}</Text>
                </li>
              ))}
            </ul>
          </Stack>
        )}

        {/* Probability summary (preview-only) */}
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
                This class-summary is computed from the preview rows only.
              </Text>
            </Stack>
          </Card>
        )}

        {/* Export button */}
        <Group justify="flex-start">
          <Button size="xs" variant="light" onClick={handleExportPreview}>
            Export preview CSV
          </Button>
        </Group>

        <ScrollArea
          h={320}
          type="auto"
          offsetScrollbars
          styles={{ viewport: { paddingRight: 8, paddingBottom: 6 }, scrollbar: { zIndex: 5 } }}
        >
          <Table
            withTableBorder={false}
            withColumnBorders={false}
            horizontalSpacing="xs"
            verticalSpacing="xs"
          >
            <Table.Thead>
              <Table.Tr>
                {columns.map((c) => {
                  const tip = buildHeaderTooltip(c);
                  const label = prettifyHeader(c);
                  const minW =
                    c === 'index'
                      ? 40
                      : c === 'fold_id'
                        ? 40
                        : c === 'y_true' || c === 'y_pred'
                          ? 40
                          : c === 'correct'
                            ? 40
                            : undefined;
                  const thStyle = minW ? { ...stickyThStyle, minWidth: minW } : stickyThStyle;

                  return (
                    <Table.Th key={c} style={thStyle}>
                      {tip ? (
                        <Tooltip label={tip} multiline maw={260} withArrow>
                          <Text size="xs" fw={600} c="white" style={headerTextStyle}>
                            {label}
                          </Text>
                        </Tooltip>
                      ) : (
                        <Text size="xs" fw={600} c="white" style={headerTextStyle}>
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

                const curFold = row?.fold_id;
                const prevFold = idx > 0 ? preview[idx - 1]?.fold_id : null;
                const isFoldBoundary =
                  curFold != null && prevFold != null && String(curFold) !== String(prevFold);

                return (
                  <Table.Tr
                    key={row?.index ?? idx}
                    style={{
                      backgroundColor: isStriped ? 'var(--mantine-color-gray-1)' : 'white',
                      borderTop: isFoldBoundary ? '3px solid var(--mantine-color-gray-4)' : undefined,
                    }}
                  >
                    {columns.map((c) => {
                      const val = row?.[c];
                      const isCorrectCol = c === 'correct';
                      const isFalse = isCorrectCol && (val === false || val === 'false');

                      return (
                        <Table.Td
                          key={c}
                          style={{
                            textAlign: 'center',
                            backgroundColor: isFalse ? 'var(--mantine-color-red-1)' : undefined,
                            whiteSpace: 'nowrap',
                          }}
                        >
                          <Text size="sm" style={cellNowrap}>
                            {renderCell(c, val)}
                          </Text>
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
