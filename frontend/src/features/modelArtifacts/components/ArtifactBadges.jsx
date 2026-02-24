import { Badge, Group, Tooltip } from '@mantine/core';

export default function ArtifactBadges({
  primaryLabel,
  isEnsemble,
  kind,
  nSplits,
  isKFold,
}) {
  return (
    <Group gap="sm">
      <Badge variant="light">{primaryLabel}</Badge>

      {isEnsemble && (
        <Badge color="cyan" variant="light">
          ensemble
        </Badge>
      )}

      {kind && (
        <Badge color="gray" variant="light">
          {kind}
        </Badge>
      )}

      {nSplits ? (
        <Badge color="grape" variant="light">
          {nSplits} splits
        </Badge>
      ) : null}

      {isKFold ? (
        <Tooltip
          label="Out-of-fold evaluation (pooled across folds): each sample is predicted by a model that did not train on it."
          withArrow
        >
          <Badge color="teal" variant="light">
            OOF
          </Badge>
        </Tooltip>
      ) : null}
    </Group>
  );
}
