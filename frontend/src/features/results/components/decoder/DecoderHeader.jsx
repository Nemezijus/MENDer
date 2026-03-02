import { Stack, Text, Group, Tooltip } from '@mantine/core';

export default function DecoderHeader({
  isKfold,
  isHoldout,
  nSplits,
  isRegression,
  hasDecisionScores,
  hasProbabilities,
  isVoteShare,
  previewN,
  totalN,
}) {
  return (
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
              <Text span fw={600} className="resultsCursorHelp">
                out-of-fold (OOF)
              </Text>
            </Tooltip>{' '}
            pooled (concatenated) across {nSplits || 'multiple'} folds).
          </>
        ) : isHoldout ? (
          <>Per-sample decoder outputs on the held-out test split.</>
        ) : (
          <>Per-sample decoder outputs on the evaluation set.</>
        )}
      </Text>

      <Group gap="md" wrap="wrap">
        {!isRegression && (
          <>
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
          </>
        )}

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
  );
}
