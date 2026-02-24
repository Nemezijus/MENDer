import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function KnnregSection({ m, set, sub, enums }) {
  const knnWeights = makeSelectData(sub, 'weights', enums?.KNNWeights);
  const knnAlgorithm = makeSelectData(sub, 'algorithm', enums?.KNNAlgorithm);
  const knnMetric = makeSelectData(sub, 'metric', enums?.KNNMetric);
  return (
    <ParamGrid>
        <ParamNumber
          label="Neighbours"
          value={m.n_neighbors ?? 5}
          onChange={(v) => set({ n_neighbors: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Weights"
          data={knnWeights}
          value={m.weights ?? 'uniform'}
          onChange={(v) => set({ weights: v })}
        />
        <ParamSelect
          label="Algorithm"
          data={knnAlgorithm}
          value={m.algorithm ?? 'auto'}
          onChange={(v) => set({ algorithm: v })}
        />
        <ParamNumber
          label="Leaf size (leaf_size)"
          value={m.leaf_size ?? 30}
          onChange={(v) => set({ leaf_size: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="p"
          value={m.p ?? 2}
          onChange={(v) => set({ p: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Metric"
          data={knnMetric}
          value={m.metric ?? 'minkowski'}
          onChange={(v) => set({ metric: v })}
        />
        <ParamNumber
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
