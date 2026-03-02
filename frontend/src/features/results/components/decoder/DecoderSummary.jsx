import {
  Stack,
  Group,
  Text,
  Tooltip,
  Table,
  ScrollArea,
  SimpleGrid,
} from '@mantine/core';

import { fmtMaybe3, fmtMaybePct } from '../../utils/formatNumbers.js';

function KeyValueBlock({ title, titleTooltip, items }) {
  const visible = (items || []).filter((x) => x && x.value !== null && x.value !== undefined);
  if (visible.length === 0) return null;

  return (
    <Stack gap={6}>
      {titleTooltip ? (
        <Tooltip label={titleTooltip} multiline maw={360} withArrow>
          <Text size="sm" fw={600} className="resultsFitContent">
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
        className="resultsTableFixed"
      >
        <Table.Tbody>
          {visible.map((it) => (
            <Table.Tr key={it.key}>
              <Table.Td className="decoderKvKeyTd">
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
              <Table.Td className="decoderKvValTd">
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

export default function DecoderSummary({
  isRegression,
  summary,
  regPerfItems,
  regDataItems,
  dataParamsItems,
  lossCalItems,
  confidenceItems,
  datasetTitleTip,
  regTitleTip,
  lossTitleTip,
  confTitleTip,
  showCalibrationBins,
  nonEmptyBins,
}) {
  if (!summary || typeof summary !== 'object') return null;

  const stickyBg = { backgroundColor: 'var(--mantine-color-gray-8)' };

  const calHeaderTip = (label, tip) => (
    <Tooltip label={tip} multiline maw={280} withArrow>
      <Text size="xs" fw={600} c="white" className="decoderHeaderText">
        {label}
      </Text>
    </Tooltip>
  );

  return (
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

      {isRegression ? (
        <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
          <KeyValueBlock title="Performance" titleTooltip={regTitleTip} items={regPerfItems} />
          <KeyValueBlock title="Dataset" titleTooltip={datasetTitleTip} items={regDataItems} />
        </SimpleGrid>
      ) : (
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
          <KeyValueBlock title="Dataset" titleTooltip={datasetTitleTip} items={dataParamsItems} />
          <KeyValueBlock title="Loss & calibration" titleTooltip={lossTitleTip} items={lossCalItems} />
          <KeyValueBlock title="Confidence" titleTooltip={confTitleTip} items={confidenceItems} />
        </SimpleGrid>
      )}

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
              striped
            >
              <Table.Thead>
                <Table.Tr>
                  <Table.Th style={{ ...stickyBg, minWidth: 50 }} className="decoderStickyTh">
                    {calHeaderTip('Bin', 'Bin index (0..B-1).')}
                  </Table.Th>
                  <Table.Th style={{ ...stickyBg, minWidth: 110 }} className="decoderStickyTh">
                    {calHeaderTip('Range', 'Confidence range covered by this bin.')}
                  </Table.Th>
                  <Table.Th style={{ ...stickyBg, minWidth: 60 }} className="decoderStickyTh">
                    {calHeaderTip('N', 'Number of samples in this bin.')}
                  </Table.Th>
                  <Table.Th style={{ ...stickyBg, minWidth: 90 }} className="decoderStickyTh">
                    {calHeaderTip('Confidence', 'Mean top-1 confidence in this bin.')}
                  </Table.Th>
                  <Table.Th style={{ ...stickyBg, minWidth: 90 }} className="decoderStickyTh">
                    {calHeaderTip('Accuracy', 'Fraction correct in this bin.')}
                  </Table.Th>
                  <Table.Th style={{ ...stickyBg, minWidth: 70 }} className="decoderStickyTh">
                    {calHeaderTip('Gap', '|accuracy − confidence| for this bin.')}
                  </Table.Th>
                </Table.Tr>
              </Table.Thead>

              <Table.Tbody>
                {nonEmptyBins.map((b, i) => (
                  <Table.Tr key={b.bin}>
                    <Table.Td className="decoderPreviewTd">
                      <Text size="sm">{b.bin}</Text>
                    </Table.Td>
                    <Table.Td className="decoderPreviewTd">
                      <Text size="sm">
                        {fmtMaybe3(b.bin_lo)}–{fmtMaybe3(b.bin_hi)}
                      </Text>
                    </Table.Td>
                    <Table.Td className="decoderPreviewTd">
                      <Text size="sm">{b.count}</Text>
                    </Table.Td>
                    <Table.Td className="decoderPreviewTd">
                      <Text size="sm">{fmtMaybe3(b.avg_confidence)}</Text>
                    </Table.Td>
                    <Table.Td className="decoderPreviewTd">
                      <Text size="sm">{fmtMaybe3(b.accuracy)}</Text>
                    </Table.Td>
                    <Table.Td className="decoderPreviewTd">
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
  );
}
