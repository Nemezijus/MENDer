import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';

export default function BgmmSection({ m, set, sub, enums }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Components (n_components)"
          value={m.n_components ?? 1}
          onChange={(v) => set({ n_components: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Covariance type"
          data={toSelectData(enumFromSubSchema(sub, 'covariance_type', ['full', 'tied', 'diag', 'spherical']))}
          value={m.covariance_type ?? 'full'}
          onChange={(v) => set({ covariance_type: v })}
        />
        <ParamNumber
          label="Tolerance (tol)"
          value={m.tol ?? 1e-3}
          onChange={(v) => set({ tol: v })}
          step={1e-4}
          min={0}
        />
        <ParamNumber
          label="Regularization (reg_covar)"
          value={m.reg_covar ?? 1e-6}
          onChange={(v) => set({ reg_covar: v })}
          step={1e-6}
          min={0}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter ?? 100}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="Initializations (n_init)"
          value={m.n_init ?? 1}
          onChange={(v) => set({ n_init: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Init params"
          data={toSelectData(enumFromSubSchema(sub, 'init_params', ['kmeans', 'k-means++', 'random', 'random_from_data']))}
          value={m.init_params ?? 'kmeans'}
          onChange={(v) => set({ init_params: v })}
        />
        <ParamSelect
          label="Weight concentration prior type"
          data={toSelectData(enumFromSubSchema(sub, 'weight_concentration_prior_type', ['dirichlet_process', 'dirichlet_distribution']))}
          value={m.weight_concentration_prior_type ?? 'dirichlet_process'}
          onChange={(v) => set({ weight_concentration_prior_type: v })}
        />
        <ParamCheckbox
          label="Warm start"
          checked={!!m.warm_start}
          onChange={(checked) => set({ warm_start: checked })}
        />
        <ParamNumber
          label="Verbose"
          value={m.verbose ?? 0}
          onChange={(v) => set({ verbose: v })}
          allowDecimal={false}
          min={0}
        />
        <ParamNumber
          label="Random state"
          value={m.random_state ?? null}
          onChange={(v) => set({ random_state: v })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
