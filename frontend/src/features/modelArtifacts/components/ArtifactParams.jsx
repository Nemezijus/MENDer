import { Text } from '@mantine/core';

function HyperparamSummary({ modelCfg }) {
  if (!modelCfg || typeof modelCfg !== 'object') {
    return (
      <Text size="sm" c="dimmed">
        No hyperparameters recorded.
      </Text>
    );
  }

  const entries = Object.entries(modelCfg)
    .filter(([k, v]) => !['algo', 'ensemble_kind', 'ensemble'].includes(k) && v !== undefined && v !== null)
    .slice(0, 6);

  if (!entries.length) {
    return (
      <Text size="sm" c="dimmed">
        No hyperparameters recorded.
      </Text>
    );
  }

  return (
    <Text size="sm">
      {entries.map(([k, v], idx) => (
        <span key={k}>
          {k}={String(v)}
          {idx < entries.length - 1 ? ', ' : ''}
        </span>
      ))}
    </Text>
  );
}

function StepParamSummary({ step }) {
  const params = step?.params && typeof step.params === 'object' ? step.params : null;
  if (!params) {
    return (
      <Text size="sm" c="dimmed">
        No estimator parameters recorded.
      </Text>
    );
  }

  const entries = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null)
    .slice(0, 6);

  if (!entries.length) {
    return (
      <Text size="sm" c="dimmed">
        No estimator parameters recorded.
      </Text>
    );
  }

  return (
    <Text size="sm">
      {entries.map(([k, v], idx) => (
        <span key={k}>
          {k}={String(v)}
          {idx < entries.length - 1 ? ', ' : ''}
        </span>
      ))}
    </Text>
  );
}

export default function ArtifactParams({ artifact, lastPipelineStep }) {
  if (!artifact) return null;
  return artifact?.model ? (
    <HyperparamSummary modelCfg={artifact.model} />
  ) : (
    <StepParamSummary step={lastPipelineStep} />
  );
}
