import { Stack, SimpleGrid, NumberInput, Select } from '@mantine/core';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function KnnSection({ m, set, sub, enums }) {
  const knnWeights = makeSelectData(sub, 'weights', enums?.KNNWeights);
  const knnAlgorithm = makeSelectData(sub, 'algorithm', enums?.KNNAlgorithm);
  const knnMetric = makeSelectData(sub, 'metric', enums?.KNNMetric);
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Neighbours"
          value={m.n_neighbors ?? 5}
          onChange={(v) => set({ n_neighbors: v })}
          allowDecimal={false}
          min={1}
        />
        <Select
          label="Weights"
          data={knnWeights}
          value={m.weights ?? 'uniform'}
          onChange={(v) => set({ weights: v })}
        />
        <Select
          label="Algorithm"
          data={knnAlgorithm}
          value={m.algorithm ?? 'auto'}
          onChange={(v) => set({ algorithm: v })}
        />
        <NumberInput
          label="Leaf size (leaf_size)"
          value={m.leaf_size ?? 30}
          onChange={(v) => set({ leaf_size: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="p"
          value={m.p ?? 2}
          onChange={(v) => set({ p: v })}
          allowDecimal={false}
          min={1}
        />
        <Select
          label="Metric"
          data={knnMetric}
          value={m.metric ?? 'minkowski'}
          onChange={(v) => set({ metric: v })}
        />
        <NumberInput
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
        />
      </SimpleGrid>
    </Stack>
  );
}
