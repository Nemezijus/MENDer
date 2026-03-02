import { Text } from '@mantine/core';

import { friendlyClassName } from '../utils/artifactUtils.js';

export default function ArtifactPipeline({ artifact }) {
  if (!artifact) return null;

  return Array.isArray(artifact.pipeline) && artifact.pipeline.length > 0 ? (
    <ul className="artifactPipelineList">
      {artifact.pipeline.map((s, i) => (
        <li key={`${s.name}-${i}`}>
          <Text size="sm">
            <strong>{s.name}</strong>: {friendlyClassName(s.class_path)}
          </Text>
        </li>
      ))}
    </ul>
  ) : (
    <Text size="sm" c="dimmed">
      No pipeline details.
    </Text>
  );
}
