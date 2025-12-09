import { Stack, Text, List } from '@mantine/core';

export function ScalingIntroText() {
  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        What is scaling?
      </Text>

      <Text size="xs" c="dimmed">
        Scaling changes the numerical range and distribution of your features
        before they are given to the model. This often makes optimisation
        easier and prevents some features from dominating others simply because
        they have larger numeric values.
      </Text>
    </Stack>
  );
}

export function ScalingDetailsText({ selectedScaling }) {
  // Normalise selected scaling value (we use the raw value, not the label)
  const selectedKey = selectedScaling
    ? String(selectedScaling).toLowerCase()
    : null;

  const isSelected = (name) => selectedKey === name;

  const labelStyle = (name) => ({
    fw: isSelected(name) ? 700 : 600,
    c: isSelected(name) ? 'blue' : undefined,
  });

  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        When to use each option
      </Text>

      <List spacing={4} size="xs">
        <List.Item>
          <Text span {...labelStyle('none')}>
            None
          </Text>{' '}
          – keep raw feature scales. Useful for tree-based models or when your
          features are already on comparable scales.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('standard')}>
            Standard
          </Text>{' '}
          – subtract mean and divide by standard deviation. Works well for many
          linear models, SVMs and neural networks.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('robust')}>
            Robust
          </Text>{' '}
          – uses median and interquartile range instead of mean and standard
          deviation. Prefer this when your features have strong outliers.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('minmax')}>
            MinMax
          </Text>{' '}
          – rescales each feature to a fixed range (commonly [0, 1]). Useful
          when you want all inputs to be strictly bounded.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('maxabs')}>
            MaxAbs
          </Text>{' '}
          – scales features to lie within [-1, 1] based on their maximum
          absolute value. Handy for sparse data where you do not want to
          destroy sparsity.
        </List.Item>

        <List.Item>
          <Text span {...labelStyle('quantile')}>
            Quantile
          </Text>{' '}
          – transforms each feature to follow a target distribution (e.g.
          uniform or normal). Can make highly skewed features more comparable,
          but is more aggressive than simple scaling.
        </List.Item>
      </List>

      <Text size="xs" c="dimmed">
        If you are unsure, a good default for many models is{' '}
        <Text span fw={600}>
          Standard
        </Text>{' '}
        scaling, unless you know your data have heavy outliers, in which case{' '}
        <Text span fw={600}>
          Robust
        </Text>{' '}
        is often safer.
      </Text>
    </Stack>
  );
}

export default function ScalingHelpText({ selectedScaling }) {
  return (
    <Stack gap="sm">
      <ScalingDetailsText selectedScaling={selectedScaling} />
    </Stack>
  );
}
