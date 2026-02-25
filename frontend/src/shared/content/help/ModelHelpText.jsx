import { Stack, Text, List } from '@mantine/core';

import { MODEL_OVERVIEW_ENTRIES } from './model/overviewEntries.js';
import { MODEL_PARAM_BLOCKS } from './model/params/index.js';

export function ModelIntroText() {
  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        What is a model?
      </Text>

      <Text size="xs" c="dimmed">
        A model is the algorithm that learns patterns from your training data
        and makes predictions on new data. Different models make different
        assumptions and trade off accuracy, interpretability, robustness, and
        training speed.
      </Text>
    </Stack>
  );
}

export function ModelDetailsText({ selectedAlgo, effectiveTask, visibleAlgos }) {
  const algo = selectedAlgo || null;
  const allowed = new Set(visibleAlgos || []);

  const isVisible = (name) => allowed.size === 0 || allowed.has(name);

  const isSelected = (name) => algo === name;

  const labelStyle = (name) => ({
    fw: isSelected(name) ? 700 : 600,
    c: isSelected(name) ? 'blue' : undefined,
  });

  const taskNote =
    effectiveTask === 'classification'
      ? 'You are working on a classification task, so classification algorithms are most relevant.'
      : effectiveTask === 'regression'
      ? 'You are working on a regression task, so regression algorithms are most relevant.'
      : effectiveTask === 'unsupervised'
      ? 'You are working on an unsupervised task, so clustering / mixture models are most relevant.'
      : 'If the task is not set yet, you can still explore models. They will be filtered once the task is known.';

  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        Choosing a model
      </Text>

      <Text size="xs" c="dimmed">
        No single model is best for all problems. Simpler models are easier to
        interpret and faster to train, while more flexible models can capture
        complex patterns but may overfit.
      </Text>

      <Text size="xs" c="dimmed">
        {taskNote}
      </Text>

      <List spacing={4} size="xs" mt="xs">
        {MODEL_OVERVIEW_ENTRIES.filter((entry) => isVisible(entry.algo)).map(
          (entry) => (
            <List.Item key={entry.algo}>
              <Text span {...labelStyle(entry.algo)}>
                {entry.label}
              </Text>{' '}
              – {entry.summary}
            </List.Item>
          )
        )}
      </List>
    </Stack>
  );
}

export function ModelParamsText({ selectedAlgo }) {
  if (!selectedAlgo) {
    return (
      <Text size="xs" c="dimmed">
        Select an algorithm above to see a short description of its key
        parameters.
      </Text>
    );
  }

  const block = MODEL_PARAM_BLOCKS[selectedAlgo];

  if (block) {
    return block;
  }

  return (
    <Text size="xs" c="dimmed">
      No parameter help is defined for this algorithm yet.
    </Text>
  );
}

export default function ModelHelpText({
  selectedAlgo,
  effectiveTask,
  visibleAlgos,
}) {
  return (
    <Stack gap="sm">
      <ModelDetailsText
        selectedAlgo={selectedAlgo}
        effectiveTask={effectiveTask}
        visibleAlgos={visibleAlgos}
      />
      <ModelParamsText selectedAlgo={selectedAlgo} />
    </Stack>
  );
}
