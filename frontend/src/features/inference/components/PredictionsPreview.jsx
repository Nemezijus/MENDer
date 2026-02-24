import { ScrollArea, Stack, Table, Text, Tooltip } from '@mantine/core';

import {
  buildHeaderTooltip,
  pickPreviewColumns,
  prettifyHeader,
  renderPreviewCell,
} from '../utils/predictionsPreview.js';

export function PredictionsPreview({ applyResult }) {
  if (!applyResult) return null;

  const {
    n_samples,
    n_features,
    task,
    metric_name,
    metric_value,
    preview,
    decoder_outputs,
  } = applyResult;

  // Prefer decoder preview rows when available so the production preview matches
  // the decoder outputs/export columns (score_*, p_*, margin, etc.).
  const decoderPreview = Array.isArray(decoder_outputs?.preview_rows)
    ? decoder_outputs.preview_rows
    : [];
  const basePreview = Array.isArray(preview) ? preview : [];

  const rows = decoderPreview.length > 0 ? decoderPreview : basePreview;
  const columnsRaw = pickPreviewColumns(rows);

  // Hide columns that are entirely empty in the preview (common in classification: residual/abs_error).
  const alwaysKeep = new Set(['index', 'trial_id', 'y_true', 'y_pred', 'correct']);
  const columns = columnsRaw.filter((c) => {
    if (alwaysKeep.has(c)) return true;
    return rows.some((r) => {
      const v = r?.[c];
      return v !== null && v !== undefined && String(v) !== '';
    });
  });

  const stickyThStyle = {
    position: 'sticky',
    top: 0,
    zIndex: 2,
    backgroundColor: 'var(--mantine-color-gray-8)',
    textAlign: 'center',
  };

  return (
    <Stack gap="xs">
      <Text size="sm" fw={500}>
        Prediction summary
      </Text>

      <Text size="xs" c="dimmed">
        Samples: {n_samples} · Features: {n_features} · Task: {task}
      </Text>

      {metric_name && metric_value != null && (
        <Text size="xs" c="dimmed">
          {metric_name}: {Number(metric_value).toFixed(4)} (on uploaded labels)
        </Text>
      )}

      {rows.length > 0 && columns.length > 0 && (
        <ScrollArea h={240} type="auto">
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
                  return (
                    <Table.Th key={c} style={stickyThStyle}>
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
              {rows.map((row, idx) => {
                const isStriped = idx % 2 === 1;
                return (
                  <Table.Tr
                    key={row?.index ?? idx}
                    style={{
                      backgroundColor: isStriped ? 'var(--mantine-color-gray-1)' : 'white',
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
                          }}
                        >
                          <Text size="sm">{renderPreviewCell(c, val)}</Text>
                        </Table.Td>
                      );
                    })}
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      )}
    </Stack>
  );
}
