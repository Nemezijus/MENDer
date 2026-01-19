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
  SimpleGrid,
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
  if (key === 'index') return 'Row index within the preview.';
  if (key === 'fold_id') return 'Fold index for this row in k-fold CV (1..K).';
  if (key === 'trial_id') return 'Optional trial identifier (if provided).';
  if (key === 'y_true') return 'Ground-truth label for this sample.';
  if (key === 'y_pred') return 'Model-predicted label for this sample.';
  if (key === 'correct') return 'Whether prediction matches the ground truth.';
  if (key === 'decoder_score')
    return 'Binary decision value (decision_function). More positive usually means stronger evidence for the positive class.';
  if (key === 'margin')
    return 'Confidence proxy: usually (top score − runner-up score) or (top probability − runner-up). Larger margin = more confident.';
  if (key.startsWith('p_')) {
    const c = key.slice(2);
    return `Predicted probability for class ${c}. (Rows sum to ~1 across all p=... columns.)`;
  }
  if (key.startsWith('score_')) {
    const c = key.slice(6);
    return `Raw decision score for class ${c}. Higher score usually means higher probability.`;
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

function detectSplitType(trainResult) {
  // Best-effort: normalize various possible places where split info could live.
  const splitCandidates = [
    trainResult?.split,
    trainResult?.data_split,
    trainResult?.dataSplit,
    trainResult?.split_type,
    trainResult?.splitType,
    trainResult?.eval?.split,
    trainResult?.eval?.split_type,
    trainResult?.eval?.splitType,
    trainResult?.eval?.split_method,
    trainResult?.eval?.splitMethod,
    trainResult?.model_card?.split,
    trainResult?.modelCard?.split,
  ];

  const raw = splitCandidates.find((x) => typeof x === 'string' && x.trim().length > 0);
  const s = String(raw || '').toLowerCase();

  if (s.includes('kfold') || s.includes('k-fold') || s.includes('cv')) return 'kfold';
  if (s.includes('hold') || s.includes('test') || s.includes('split')) return 'holdout';

  return 'unknown';
}

function KeyValueBlock({ title, titleTooltip, items }) {
  const visible = (items || []).filter((x) => x && x.value !== null && x.value !== undefined);
  if (visible.length === 0) return null;

  return (
    <Stack gap={6}>
      {titleTooltip ? (
        <Tooltip label={titleTooltip} multiline maw={360} withArrow>
          <Text size="sm" fw={600} style={{ width: 'fit-content' }}>
            {title}
          </Text>
        </Tooltip>
      ) : (
        <Text size="sm" fw={600}>
          {title}
        </Text>
      )}

      <Table
        withTableBorder={false}
        withColumnBorders={false}
        horizontalSpacing="xs"
        verticalSpacing={4}
        style={{ tableLayout: 'fixed' }}
      >
        <Table.Tbody>
          {visible.map((it) => (
            <Table.Tr key={it.key}>
              <Table.Td style={{ paddingLeft: 0, width: '70%' }}>
                {it.tooltip ? (
                  <Tooltip label={it.tooltip} multiline maw={360} withArrow>
                    <Text size="sm" c="dimmed">
                      {it.key}
                    </Text>
                  </Tooltip>
                ) : (
                  <Text size="sm" c="dimmed">
                    {it.key}
                  </Text>
                )}
              </Table.Td>
              <Table.Td style={{ paddingRight: 0, textAlign: 'left' }}>
                <Text size="sm" fw={700}>
                  {it.format === 'pct' ? fmtMaybePct(it.value) : fmtMaybe3(it.value)}
                </Text>
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Stack>
  );
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

  const probaSource = decoder.proba_source ?? decoder.probaSource ?? null;
  const isVoteShare = String(probaSource || '').toLowerCase() === 'vote_share';

  const summary = useMemo(() => getSummary(decoder), [decoder]);

  const splitType = useMemo(() => detectSplitType(trainResult), [trainResult]);
  const isKfold = splitType === 'kfold';
  const isHoldout = splitType === 'holdout';

  // Notes: remove duplicated “Decoder summary: …” line if we already display the structured summary.
  const decoderNotes = useMemo(() => {
    const dnRaw = Array.isArray(decoder.notes) ? decoder.notes : [];
    const tn = Array.isArray(trainResult.notes) ? trainResult.notes : [];

    const dn = dnRaw.filter((n) => !tn.includes(n) && !tn.includes(`Decoder outputs: ${n}`));

    const hasStructuredSummary = summary && typeof summary === 'object';
    const filtered = hasStructuredSummary
      ? dn.filter((n) => !String(n).toLowerCase().startsWith('decoder summary:'))
      : dn;

    return filtered.filter((n) => String(n).trim().length > 0);
  }, [decoder.notes, trainResult.notes, summary]);

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

  const marginFracKey = summary
    ? firstKey(summary, ['margin_frac_lt_0_1', 'margin_frac_lt_0_05'])
    : null;
  const maxProbaFracKey = summary
    ? firstKey(summary, ['max_proba_frac_ge_0_9', 'max_proba_frac_ge_0_8'])
    : null;

  const eceBins =
    summary && Array.isArray(summary.reliability_bins) ? summary.reliability_bins : null;

  const nonEmptyBins = useMemo(() => {
    if (!eceBins) return [];
    return eceBins.filter((b) => b && typeof b.count === 'number' && b.count > 0);
  }, [eceBins]);

  const showCalibrationBins = nonEmptyBins.length > 0;

  const showClassSummary = hasProbabilities && classOptions.length > 0;

  // ---------- Summary blocks (structured) ----------
  const evalSamplesTooltip = isKfold
    ? 'Number of evaluation samples pooled across folds using out-of-fold (OOF) predictions.'
    : isHoldout
      ? 'Number of samples in the held-out test split.'
      : 'Number of evaluation samples for this run.';

  const dataParamsItems = [
    {
      key: 'Evaluation samples',
      value: summary?.n_samples ?? totalN ?? null,
      tooltip: evalSamplesTooltip,
    },
    {
      key: 'Number of classes',
      value: summary?.n_classes ?? (Array.isArray(classes) ? classes.length : null),
      tooltip: 'Number of unique class labels.',
    },
    {
      key: 'Calibration bins',
      value: summary?.ece_n_bins ?? null,
      tooltip:
        'Number of confidence bins used to compute calibration (ECE/MCE). More bins = finer detail, but fewer samples per bin.',
    },
  ];

  const lossCalItems = [
    {
      key: 'Log loss',
      value: summary?.log_loss ?? null,
      tooltip:
        'Cross-entropy loss. Smaller is better. Minimum is 0; there is no fixed upper bound.',
    },
    {
      key: 'Brier score',
      value: summary?.brier ?? null,
      tooltip:
        'Probability error (mean squared error vs one-hot truth). Smaller is better. Typical range is 0 to 1.',
    },
    {
      key: 'Expected calibration error (ECE)',
      value: summary?.ece ?? null,
      tooltip:
        'Calibration error using top-1 confidence. Smaller is better. Typical range is 0 to 1 (0 means perfectly calibrated).',
    },
    {
      key: 'Maximum calibration error (MCE)',
      value: summary?.mce ?? null,
      tooltip:
        'Worst-bin calibration gap |accuracy − confidence|. Smaller is better. Typical range is 0 to 1.',
    },
  ];

  const confidenceItems = [
    {
      key: 'Margin (mean)',
      value: summary?.margin_mean ?? null,
      tooltip:
        'Average top1−top2 separation (score or probability). Larger usually means more confident predictions.',
    },
    {
      key: 'Margin (median)',
      value: summary?.margin_median ?? null,
      tooltip:
        'Median top1−top2 separation (score or probability). Larger usually means more confident predictions.',
    },
    marginFracKey
      ? {
          key: 'Low margin (fraction)',
          value: summary?.[marginFracKey] ?? null,
          format: 'pct',
          tooltip:
            'Fraction of samples with small top1−top2 separation. Smaller is better (fewer uncertain predictions).',
        }
      : null,
    {
      key: 'Max probability (mean)',
      value: summary?.max_proba_mean ?? null,
      tooltip:
        'Average of max predicted probability per sample. Higher means the model is more confident (not necessarily more accurate). Range 0–1.',
    },
    maxProbaFracKey
      ? {
          key: 'High confidence (fraction)',
          value: summary?.[maxProbaFracKey] ?? null,
          format: 'pct',
          tooltip:
            'Fraction of samples with high max probability. Higher means more confident predictions (not necessarily more accurate).',
        }
      : null,
  ].filter(Boolean);

  const datasetTitleTip =
    'Quick context about the evaluation set used for decoder outputs. In k-fold CV this is typically pooled out-of-fold (OOF) predictions; in hold-out it is the held-out test split.';
  const lossTitleTip =
    'Loss and calibration describe probability quality. Log loss and Brier measure probability error (smaller is better). ECE/MCE describe how well confidence matches accuracy (smaller is better; 0 is ideal).';
  const confTitleTip =
    'Confidence describes how decisive predictions are. Margin reflects top-1 vs runner-up separation (larger is typically more confident). Max probability is a confidence proxy (higher means more confident, not necessarily more accurate).';

  // ---------- Calibration bins table styling ----------
  const calStickyThStyle = {
    ...stickyThStyle,
    backgroundColor: 'var(--mantine-color-gray-8)',
    textAlign: 'center',
  };

  const calHeaderTip = (label, tip) => (
    <Tooltip label={tip} multiline maw={280} withArrow>
      <Text size="xs" fw={600} c="white" style={headerTextStyle}>
        {label}
      </Text>
    </Tooltip>
  );

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500} size="xl" align="center">
          Decoder outputs
        </Text>

        {/* Overview */}
        <Stack gap={6}>
          <Text size="sm" c="dimmed">
            {isKfold ? (
              <>
                Per-sample decoder outputs on the evaluation set (
                <Tooltip
                  label="Out-of-fold (OOF) predictions are generated for samples that were not used to train the model in that fold. Pooling OOF predictions gives an unbiased estimate of performance."
                  multiline
                  maw={380}
                  withArrow
                >
                  <Text span fw={600} style={{ cursor: 'help' }}>
                    out-of-fold (OOF)
                  </Text>
                </Tooltip>{' '}
                pooled across folds).
              </>
            ) : isHoldout ? (
              <>Per-sample decoder outputs on the held-out test split.</>
            ) : (
              <>Per-sample decoder outputs on the evaluation set.</>
            )}
          </Text>

          <Group gap="md" wrap="wrap">
            <Tooltip
              label="Whether decision_function outputs exist (used to build score_* columns)."
              multiline
              maw={320}
              withArrow
            >
              <Text size="sm">
                <Text span c="dimmed">
                  Decision scores:{' '}
                </Text>
                <Text span fw={700}>
                  {hasDecisionScores ? 'Available' : 'Not available'}
                </Text>
              </Text>
            </Tooltip>

            <Tooltip
              label="Whether probability columns exist. For hard voting, these may be vote shares (not calibrated probabilities)."
              multiline
              maw={340}
              withArrow
            >
              <Text size="sm">
                <Text span c="dimmed">
                  Probabilities:{' '}
                </Text>
                <Text span fw={700}>
                  {hasProbabilities ? 'Available' : 'Not available'}
                </Text>
                {hasProbabilities && isVoteShare ? (
                  <Text span fw={500} c="dimmed">
                    {' '}
                    (vote share)
                  </Text>
                ) : null}
              </Text>
            </Tooltip>

            <Tooltip
              label="Number of rows rendered in the table. Preview may be capped for performance."
              multiline
              maw={320}
              withArrow
            >
              <Text size="sm">
                <Text span c="dimmed">
                  Previewed samples:{' '}
                </Text>
                <Text span fw={700}>
                  {totalN != null ? `${previewN} / ${totalN}` : `${previewN}`}
                </Text>
              </Text>
            </Tooltip>
          </Group>
        </Stack>

        <Divider />

        {/* Decoder summary */}
        {summary && typeof summary === 'object' && (
          <Stack gap="sm">
            <Group justify="space-between" align="center">
              <Stack gap={0}>
                <Text size="sm" fw={600}>
                  Decoder summary
                </Text>
                <Text size="xs" c="dimmed">
                  Computed from the full evaluation set (not just the preview).
                </Text>
              </Stack>
            </Group>

            <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
              <KeyValueBlock title="Dataset" titleTooltip={datasetTitleTip} items={dataParamsItems} />
              <KeyValueBlock
                title="Loss & calibration"
                titleTooltip={lossTitleTip}
                items={lossCalItems}
              />
              <KeyValueBlock title="Confidence" titleTooltip={confTitleTip} items={confidenceItems} />
            </SimpleGrid>

            {/* Calibration bins */}
            {showCalibrationBins && (
              <Stack gap="xs">
                <Group justify="space-between" align="center">
                  <Stack gap={0}>
                    <Text size="sm" fw={600}>
                      Calibration bins
                    </Text>
                    <Text size="xs" c="dimmed">
                      Top-1 confidence reliability (non-empty bins).
                    </Text>
                  </Stack>
                </Group>

                <ScrollArea h={180} type="auto" offsetScrollbars>
                  <Table
                    withTableBorder={false}
                    withColumnBorders={false}
                    horizontalSpacing="xs"
                    verticalSpacing="xs"
                  >
                    <Table.Thead>
                      <Table.Tr>
                        <Table.Th style={{ ...calStickyThStyle, minWidth: 50 }}>
                          {calHeaderTip('Bin', 'Bin index (0..B-1).')}
                        </Table.Th>
                        <Table.Th style={{ ...calStickyThStyle, minWidth: 110 }}>
                          {calHeaderTip('Range', 'Confidence range covered by this bin.')}
                        </Table.Th>
                        <Table.Th style={{ ...calStickyThStyle, minWidth: 60 }}>
                          {calHeaderTip('N', 'Number of samples in this bin.')}
                        </Table.Th>
                        <Table.Th style={{ ...calStickyThStyle, minWidth: 90 }}>
                          {calHeaderTip('Confidence', 'Mean top-1 confidence in this bin.')}
                        </Table.Th>
                        <Table.Th style={{ ...calStickyThStyle, minWidth: 90 }}>
                          {calHeaderTip('Accuracy', 'Fraction correct in this bin.')}
                        </Table.Th>
                        <Table.Th style={{ ...calStickyThStyle, minWidth: 70 }}>
                          {calHeaderTip('Gap', '|accuracy − confidence| for this bin.')}
                        </Table.Th>
                      </Table.Tr>
                    </Table.Thead>

                    <Table.Tbody>
                      {nonEmptyBins.map((b, i) => (
                        <Table.Tr
                          key={b.bin}
                          style={{
                            backgroundColor: i % 2 === 1 ? 'var(--mantine-color-gray-1)' : 'white',
                          }}
                        >
                          <Table.Td style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                            <Text size="sm">{b.bin}</Text>
                          </Table.Td>
                          <Table.Td style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                            <Text size="sm">
                              {fmtMaybe3(b.bin_lo)}–{fmtMaybe3(b.bin_hi)}
                            </Text>
                          </Table.Td>
                          <Table.Td style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                            <Text size="sm">{b.count}</Text>
                          </Table.Td>
                          <Table.Td style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                            <Text size="sm">{fmtMaybe3(b.avg_confidence)}</Text>
                          </Table.Td>
                          <Table.Td style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                            <Text size="sm">{fmtMaybe3(b.accuracy)}</Text>
                          </Table.Td>
                          <Table.Td style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                            <Text size="sm">{fmtMaybe3(b.gap)}</Text>
                          </Table.Td>
                        </Table.Tr>
                      ))}
                    </Table.Tbody>
                  </Table>
                </ScrollArea>
              </Stack>
            )}
          </Stack>
        )}

        {/* Only render class-summary dividers if that section exists */}
        {showClassSummary ? (
          <>
            <Divider />

            <Stack gap="xs">
              <Stack gap={0}>
                <Text size="sm" fw={600}>
                  Class probability summary
                </Text>
                <Text size="xs" c="dimmed">
                  Mean/median of p(class) across the preview rows (quick inspection).
                </Text>
              </Stack>

              <Group justify="space-between" align="flex-end" wrap="wrap">
                <Select
                  label="Class"
                  data={classOptions}
                  value={selectedClass}
                  onChange={setSelectedClass}
                  w={240}
                />

                <Group gap="lg" wrap="wrap">
                  <Tooltip
                    label="Average p(selected class) over preview rows. Range 0–1."
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span c="dimmed">
                        Mean:{' '}
                      </Text>
                      <Text span fw={700}>
                        {probStats.mean != null ? probStats.mean.toFixed(3) : '—'}
                      </Text>
                    </Text>
                  </Tooltip>

                  <Tooltip
                    label="Median p(selected class) over preview rows. Range 0–1."
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span c="dimmed">
                        Median:{' '}
                      </Text>
                      <Text span fw={700}>
                        {probStats.median != null ? probStats.median.toFixed(3) : '—'}
                      </Text>
                    </Text>
                  </Tooltip>

                  <Tooltip
                    label="Number of preview rows used for these summary values."
                    multiline
                    maw={280}
                    withArrow
                  >
                    <Text size="sm">
                      <Text span c="dimmed">
                        Rows used:{' '}
                      </Text>
                      <Text span fw={700}>
                        {probStats.n}
                      </Text>
                    </Text>
                  </Tooltip>
                </Group>
              </Group>
            </Stack>

            <Divider />
          </>
        ) : (
          <Divider />
        )}

        {/* Export + main table */}
        <Group justify="space-between" align="center">
          <Text size="sm" fw={600}>
            Preview table
          </Text>
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
                      ? 55
                      : c === 'fold_id'
                        ? 70
                        : c === 'y_true' || c === 'y_pred'
                          ? 80
                          : c === 'correct'
                            ? 80
                            : c === 'margin'
                              ? 80
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
                          <Text size="sm" style={{ whiteSpace: 'nowrap' }}>
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

        {/* Notes */}
        {decoderNotes.length > 0 && (
          <Stack gap={4}>
            <Text size="sm" fw={600}>
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
      </Stack>
    </Card>
  );
}
