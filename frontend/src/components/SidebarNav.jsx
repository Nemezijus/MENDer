// frontend/src/components/SidebarNav.jsx
import { Stack, Text, NavLink, Divider } from '@mantine/core';

const SECTION_GROUPS = [
  {
    label: 'DATA',
    items: [
      {
        id: 'data',
        label: 'Data & files',
        description: 'Upload and inspect training data, manage files',
      },
    ],
  },
  {
    label: 'SETTINGS',
    items: [
      {
        id: 'settings',
        label: 'Settings',
        description: 'Scaling, features, metric',
      },
    ],
  },
  {
    label: 'MODEL TRAINING',
    items: [
      {
        id: 'train',
        label: 'Train a model',
        description: 'Choose algorithm and fit on current data',
      },
    ],
  },
  {
    label: 'TUNING',
    items: [
      {
        id: 'learning-curve',
        label: 'Learning curve',
        description: 'Explore sample size vs performance',
      },
      {
        id: 'validation-curve',
        label: 'Validation curve',
        description: 'Explore hyperparameter vs performance',
      },
      {
        id: 'grid-search',
        label: 'Grid search',
        description: 'Exhaustive search over two hyperparameters',
      },
      {
        id: 'random-search',
        label: 'Randomized search',
        description: 'Random sampling over two hyperparameters',
      },
    ],
  },
  {
    label: 'RESULTS',
    items: [
      {
        id: 'results',
        label: 'Results',
        description: 'Graphs, tables, and summaries',
      },
    ],
  },
  {
    label: 'PREDICTIONS',
    items: [
      {
        id: 'predictions',
        label: 'Predictions',
        description: 'Apply model to production data',
      },
    ],
  },
];

export default function SidebarNav({ active, onChange }) {
  return (
    <Stack gap="xs">
      {SECTION_GROUPS.map((group, groupIdx) => (
        <Stack gap={4} key={group.label}>
          <Text size="xs" fw={700} c="dimmed" tt="uppercase" pl="xs">
            {group.label}
          </Text>

          {group.items.map((item) => (
            <NavLink
              key={item.id}
              label={item.label}
              description={item.description}
              active={active === item.id}
              onClick={() => onChange(item.id)}
              variant="light"
              radius="sm"
            />
          ))}

          {groupIdx < SECTION_GROUPS.length - 1 && <Divider my="xs" />}
        </Stack>
      ))}
    </Stack>
  );
}
