import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function KnnregSection({ m, set, sub, enums, d }) {
  const knnWeights = makeSelectData(sub, 'weights', enums?.KNNWeights);
  const knnAlgorithm = makeSelectData(sub, 'algorithm', enums?.KNNAlgorithm);
  const knnMetric = makeSelectData(sub, 'metric', enums?.KNNMetric);
  return (
    <ParamGrid>
        <ParamNumber
          label="Neighbours"
          value={m.n_neighbors}

          placeholder={defaultPlaceholder(d?.n_neighbors)}
          onChange={(v) => set({ n_neighbors: overrideOrUndef(v, d?.n_neighbors) })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Weights"
          data={knnWeights}
          value={m.weights}

          placeholder={defaultPlaceholder(d?.weights)}
          onChange={(v) => set({ weights: overrideOrUndef(v, d?.weights) })}
        />
        <ParamSelect
          label="Algorithm"
          data={knnAlgorithm}
          value={m.algorithm}

          placeholder={defaultPlaceholder(d?.algorithm)}
          onChange={(v) => set({ algorithm: overrideOrUndef(v, d?.algorithm) })}
        />
        <ParamNumber
          label="Leaf size (leaf_size)"
          value={m.leaf_size}

          placeholder={defaultPlaceholder(d?.leaf_size)}
          onChange={(v) => set({ leaf_size: overrideOrUndef(v, d?.leaf_size) })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="p"
          value={m.p}

          placeholder={defaultPlaceholder(d?.p)}
          onChange={(v) => set({ p: overrideOrUndef(v, d?.p) })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Metric"
          data={knnMetric}
          value={m.metric}

          placeholder={defaultPlaceholder(d?.metric)}
          onChange={(v) => set({ metric: overrideOrUndef(v, d?.metric) })}
        />
        <ParamNumber
          label="Jobs (n_jobs)"
          value={m.n_jobs}

          placeholder={defaultPlaceholder(d?.n_jobs)}
          onChange={(v) => set({ n_jobs: overrideOrUndef(v, d?.n_jobs) })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
