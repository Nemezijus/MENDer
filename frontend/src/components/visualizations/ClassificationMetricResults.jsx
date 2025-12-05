import { Stack, Text, Table, Tooltip } from '@mantine/core';

export default function ClassificationMetricResults({
  confusion,
  metricName,
}) {
  if (!confusion) return null;

  const { overall, macro_avg, weighted_avg, per_class } = confusion;

  if (!Array.isArray(per_class) || per_class.length === 0) {
    return null;
  }

  const fmt = (v) =>
    typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : '—';

  const fmtRatio = (v) =>
    typeof v === 'number' && Number.isFinite(v) ? v.toFixed(2) : '—';

  const supports = per_class.map((c) => c.support ?? 0);
  const totalSupport = supports.reduce((acc, s) => acc + s, 0);

  // ---- class imbalance (max/min support, ignoring zeros) ----
  const nonZeroSupports = supports.filter((s) => s > 0);
  const imbalanceRatio =
    nonZeroSupports.length >= 2
      ? Math.max(...nonZeroSupports) / Math.min(...nonZeroSupports)
      : null;

  let imbalanceDesc = null;
  if (typeof imbalanceRatio === 'number' && Number.isFinite(imbalanceRatio)) {
    if (imbalanceRatio <= 1.5) {
      imbalanceDesc = 'Classes are well balanced.';
    } else if (imbalanceRatio <= 3) {
      imbalanceDesc = 'Classes are mildly imbalanced.';
    } else if (imbalanceRatio <= 5) {
      imbalanceDesc = 'Classes are moderately imbalanced.';
    } else {
      imbalanceDesc =
        'Classes are strongly imbalanced (majority class dominates the minority).';
    }
  }

  const safeMean = (vals) => {
    const arr = vals.filter((v) => typeof v === 'number' && Number.isFinite(v));
    if (!arr.length) return null;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  };

  const safeWeightedMean = (vals, weights) => {
    const pairs = vals
      .map((v, i) => [v, weights[i]])
      .filter(
        ([v, w]) =>
          typeof v === 'number' &&
          Number.isFinite(v) &&
          typeof w === 'number' &&
          w > 0,
      );
    if (!pairs.length) return null;
    const num = pairs.reduce((acc, [v, w]) => acc + v * w, 0);
    const den = pairs.reduce((acc, [, w]) => acc + w, 0);
    if (!den) return null;
    return num / den;
  };

  // Macro values (precision/recall/f1 from backend if present; otherwise from per_class)
  const macroPrecision =
    macro_avg?.precision ??
    safeMean(per_class.map((c) => c.precision));
  const macroRecall =
    macro_avg?.recall ??
    safeMean(per_class.map((c) => c.recall ?? c.tpr));
  const macroF1 =
    macro_avg?.f1 ??
    safeMean(per_class.map((c) => c.f1));

  const macroTPR = safeMean(per_class.map((c) => c.tpr));
  const macroFPR = safeMean(per_class.map((c) => c.fpr));
  const macroTNR = safeMean(per_class.map((c) => c.tnr));
  const macroFNR = safeMean(per_class.map((c) => c.fnr));
  const macroMCC =
    macro_avg?.mcc ??
    safeMean(per_class.map((c) => c.mcc));

  // Weighted values (precision/recall/f1 from backend if present; otherwise weighted by support)
  const weightedPrecision =
    weighted_avg?.precision ??
    (totalSupport
      ? safeWeightedMean(per_class.map((c) => c.precision), supports)
      : null);
  const weightedRecall =
    weighted_avg?.recall ??
    (totalSupport
      ? safeWeightedMean(
          per_class.map((c) => c.recall ?? c.tpr),
          supports,
        )
      : null);
  const weightedF1 =
    weighted_avg?.f1 ??
    (totalSupport
      ? safeWeightedMean(per_class.map((c) => c.f1), supports)
      : null);

  const weightedTPR = totalSupport
    ? safeWeightedMean(per_class.map((c) => c.tpr), supports)
    : null;
  const weightedFPR = totalSupport
    ? safeWeightedMean(per_class.map((c) => c.fpr), supports)
    : null;
  const weightedTNR = totalSupport
    ? safeWeightedMean(per_class.map((c) => c.tnr), supports)
    : null;
  const weightedFNR = totalSupport
    ? safeWeightedMean(per_class.map((c) => c.fnr), supports)
    : null;
  const weightedMCC =
    weighted_avg?.mcc ??
    (totalSupport
      ? safeWeightedMean(per_class.map((c) => c.mcc), supports)
      : null);

  return (
    <Stack gap="xs" mt={2}>
      <Text fw={500}  size="xl" align="center">
        Summary of metrics
      </Text>

      {overall && (
        <Stack gap={0}>
          <Text size="sm">
            <Text span fw={500}>
              Accuracy:{' '}
            </Text>
            <Text span fw={700}>
              {fmt(overall.accuracy)}
            </Text>
          </Text>

          <Text size="sm">
            <Text span fw={500}>
              Balanced accuracy:{' '}
            </Text>
            <Text span fw={700}>
              {fmt(overall.balanced_accuracy)}
            </Text>
          </Text>

          <Text size="sm">
            <Text span fw={500}>
              Class imbalance (max / min support):{' '}
            </Text>
            <Text span fw={700}>
              {fmtRatio(imbalanceRatio)}
            </Text>
            {imbalanceDesc && (
              <>
                {' '}
                <Text span size="xs" c="dimmed">
                  {imbalanceDesc}
                </Text>
              </>
            )}
          </Text>
        </Stack>
      )}

      <Table
        withTableBorder={false}
        withColumnBorders={false}
        horizontalSpacing="xs"
        verticalSpacing="xs"
      >
        <Table.Thead>
          <Table.Tr
            style={{
              backgroundColor: 'var(--mantine-color-gray-8)',
            }}
          >
            <Table.Th style={{ textAlign: 'center' }}>
              <Text size="xs" fw={600} c="white">
                Class / aggregate
              </Text>
            </Table.Th>

            <Table.Th style={{ textAlign: 'center' }}>
              <Tooltip
                label="Precision measures how many of the predicted positives are actually correct. High precision means few false positives."
                multiline
                maw={260}
                withArrow
              >
                <Text size="xs" fw={600} c="white">
                  Precision
                </Text>
              </Tooltip>
            </Table.Th>

            <Table.Th style={{ textAlign: 'center' }}>
              <Tooltip
                label="Recall, or True Positive Rate (TPR), measures how many of the true positives the model correctly finds. A recall of 1 means every true case was detected."
                multiline
                maw={260}
                withArrow
              >
                <Text size="xs" fw={600} c="white">
                  Recall (TPR)
                </Text>
              </Tooltip>
            </Table.Th>

            <Table.Th style={{ textAlign: 'center' }}>
              <Tooltip
                label="False Positive Rate (FPR) measures how many of the true negatives were incorrectly predicted as positive. Lower values are better."
                multiline
                maw={260}
                withArrow
              >
                <Text size="xs" fw={600} c="white">
                  FPR
                </Text>
              </Tooltip>
            </Table.Th>

            <Table.Th style={{ textAlign: 'center' }}>
              <Tooltip
                label="True Negative Rate (TNR), or specificity, measures how many of the true negatives were correctly predicted as negative. Higher values are better."
                multiline
                maw={260}
                withArrow
              >
                <Text size="xs" fw={600} c="white">
                  TNR
                </Text>
              </Tooltip>
            </Table.Th>

            <Table.Th style={{ textAlign: 'center' }}>
              <Tooltip
                label="False Negative Rate (FNR) measures how many of the true positives were missed by the model. Lower values are better."
                multiline
                maw={260}
                withArrow
              >
                <Text size="xs" fw={600} c="white">
                  FNR
                </Text>
              </Tooltip>
            </Table.Th>

            <Table.Th style={{ textAlign: 'center' }}>
              <Tooltip
                label="F1 score is the harmonic mean of precision and recall. It is high only when both precision and recall are high."
                multiline
                maw={260}
                withArrow
              >
                <Text size="xs" fw={600} c="white">
                  F1
                </Text>
              </Tooltip>
            </Table.Th>

            <Table.Th style={{ textAlign: 'center' }}>
              <Tooltip
                label="Matthews Correlation Coefficient (MCC) is a balanced measure of classification quality. It ranges from -1 (total disagreement) through 0 (random) to 1 (perfect prediction)."
                multiline
                maw={260}
                withArrow
              >
                <Text size="xs" fw={600} c="white">
                  MCC
                </Text>
              </Tooltip>
            </Table.Th>
          </Table.Tr>
        </Table.Thead>

        <Table.Tbody>
          {/* Per-class rows (zebra striping) */}
          {per_class.map((c, idx) => {
            const isStriped = idx % 2 === 1;
            return (
              <Table.Tr
                key={idx}
                style={{
                  backgroundColor: isStriped
                    ? 'var(--mantine-color-gray-1)'
                    : 'white',
                }}
              >
                <Table.Td style={{ textAlign: 'center' }}>
                  <Text size="sm" fw={600}>
                    {String(c.label)}
                  </Text>
                </Table.Td>
                <Table.Td style={{ textAlign: 'center' }}>{fmt(c.precision)}</Table.Td>
                <Table.Td style={{ textAlign: 'center' }}>{fmt(c.tpr)}</Table.Td>
                <Table.Td style={{ textAlign: 'center' }}>{fmt(c.fpr)}</Table.Td>
                <Table.Td style={{ textAlign: 'center' }}>{fmt(c.tnr)}</Table.Td>
                <Table.Td style={{ textAlign: 'center' }}>{fmt(c.fnr)}</Table.Td>
                <Table.Td style={{ textAlign: 'center' }}>{fmt(c.f1)}</Table.Td>
                <Table.Td style={{ textAlign: 'center' }}>{fmt(c.mcc)}</Table.Td>
              </Table.Tr>
            );
          })}

          {/* Spacer / separator before aggregates */}
          <Table.Tr
            style={{
              borderTop: '2px solid var(--mantine-color-gray-4)',
              height: 4,
            }}
          >
            <Table.Td colSpan={8} style={{ padding: 0 }} />
          </Table.Tr>

          {/* Macro avg row */}
          <Table.Tr
            style={{
              backgroundColor: 'var(--mantine-color-gray-2)',
              borderTop: '1px solid var(--mantine-color-gray-4)',
            }}
          >
            <Table.Td style={{ textAlign: 'left' }}>
              <Tooltip
                label="Macro average treats each class equally, regardless of how many samples each class has. Good for seeing performance on rare classes."
                multiline
                maw={260}
                withArrow
              >
                <Text size="sm" fw={600}>
                  Macro avg
                </Text>
              </Tooltip>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(macroPrecision)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(macroTPR ?? macroRecall)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(macroFPR)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(macroTNR)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(macroFNR)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(macroF1)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(macroMCC)}
              </Text>
            </Table.Td>
          </Table.Tr>

          {/* Weighted avg row */}
          <Table.Tr
            style={{
              backgroundColor: 'var(--mantine-color-gray-3)',
              borderTop: '1px solid var(--mantine-color-gray-4)',
            }}
          >
            <Table.Td style={{ textAlign: 'left' }}>
              <Tooltip
                label="Weighted average takes into account how many samples each class has. Large classes influence this more than small ones."
                multiline
                maw={260}
                withArrow
              >
                <Text size="sm" fw={600}>
                  Weighted avg
                </Text>
              </Tooltip>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(weightedPrecision)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(weightedTPR ?? weightedRecall)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(weightedFPR)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(weightedTNR)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(weightedFNR)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(weightedF1)}
              </Text>
            </Table.Td>
            <Table.Td style={{ textAlign: 'center' }}>
              <Text size="sm" fw={600}>
                {fmt(weightedMCC)}
              </Text>
            </Table.Td>
          </Table.Tr>
        </Table.Tbody>
      </Table>
    </Stack>
  );
}
