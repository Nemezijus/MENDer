import { Stack, Text, List } from '@mantine/core';

export function SplitIntroText() {
  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        What is a data split?
      </Text>

      <Text size="xs" c="dimmed">
        A data split defines how the dataset is divided into training and
        validation/test parts. It affects how reliable your performance
        estimate is and how much data the model sees during training.
      </Text>
    </Stack>
  );
}

export function SplitDetailsText({ selectedMode, allowStratified }) {
  const mode = selectedMode || 'holdout';

  const isSelected = (name) => mode === name;

  const labelStyle = (name) => ({
    fw: isSelected(name) ? 700 : 600,
    c: isSelected(name) ? 'blue' : undefined,
  });

  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        Split strategies
      </Text>

      <List spacing={4} size="xs">
        <List.Item>
          <Text span {...labelStyle('holdout')}>
            Hold-out
          </Text>{' '}
          – one fixed split into train and test (e.g. 75% / 25%). Simple and
          fast, good for quick experiments or large datasets.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('kfold')}>
            K-fold cross-validation
          </Text>{' '}
          – splits the data into <Text span fw={600}>K</Text> folds and rotates
          which fold is used as validation. More expensive, but gives a more
          stable estimate on smaller datasets.
        </List.Item>
      </List>

      <Text fw={500} size="sm" mt="xs">
        Important options
      </Text>

      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>Train fraction</Text>{' '}
          – used in <Text span fw={600}>hold-out</Text>. Controls how much of
          the data is used for training versus testing (e.g. 0.75 → 75% train,
          25% test).
        </List.Item>

        <List.Item>
          <Text span fw={600}>K-Fold (n_splits)</Text>{' '}
          – used in <Text span fw={600}>k-fold</Text>. Higher values give more
          validation folds but increase runtime.
        </List.Item>

        <List.Item>
          <Text span fw={600}>Stratified</Text>{' '}
          – keeps the class proportions similar in each split. This is helpful
          for classification, especially with imbalanced classes, but is not
          used for regression.
          {!allowStratified && (
            <Text size="xs" c="dimmed">
              (Disabled here because the current task is regression.)
            </Text>
          )}
        </List.Item>

        <List.Item>
          <Text span fw={600}>Shuffle split</Text>{' '}
          – randomly shuffles the data before splitting. This helps avoid
          artifacts from ordering (e.g. time, blocks in acquisition), but
          should be used with care for time-series.
        </List.Item>

        <List.Item>
          <Text span fw={600}>Seed</Text>{' '}
          – controls the random shuffling. Using a fixed seed makes the split
          reproducible; changing it gives a different random split.
        </List.Item>
      </List>

      <Text size="xs" c="dimmed">
        Rough rule of thumb: use{' '}
        <Text span fw={600}>hold-out</Text> for fast prototyping or large
        datasets, and <Text span fw={600}>k-fold</Text> when you need a more
        reliable estimate on limited data.
      </Text>
    </Stack>
  );
}

// Default export: just the detailed part (for the full-width C block)
export default function SplitHelpText({ selectedMode, allowStratified }) {
  return (
    <SplitDetailsText
      selectedMode={selectedMode}
      allowStratified={allowStratified}
    />
  );
}
