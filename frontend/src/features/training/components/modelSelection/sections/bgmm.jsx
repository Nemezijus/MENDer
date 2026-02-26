import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function BgmmSection({ m, set, sub, enums, d }) {
  const covarianceTypeData = toSelectData(enumFromSubSchema(sub, 'covariance_type'));
  const initParamsData = toSelectData(enumFromSubSchema(sub, 'init_params'));
  const weightConcentrationPriorTypeData = toSelectData(
    enumFromSubSchema(sub, 'weight_concentration_prior_type'),
  );

  const covarianceTypeUnavailable = covarianceTypeData.length === 0;
  const initParamsUnavailable = initParamsData.length === 0;
  const weightPriorTypeUnavailable = weightConcentrationPriorTypeData.length === 0;

  return (
    <ParamGrid>
      <ParamNumber
        label="Components (n_components)"
        value={m.n_components}
        placeholder={defaultPlaceholder(d?.n_components)}
        onChange={(v) => set({ n_components: overrideOrUndef(v, d?.n_components) })}
        allowDecimal={false}
        min={1}
      />

      <ParamSelect
        label="Covariance type"
        data={covarianceTypeData}
        value={m.covariance_type}
        disabled={covarianceTypeUnavailable}
        placeholder={
          covarianceTypeUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.covariance_type)
        }
        description={
          covarianceTypeUnavailable ? 'Schema did not provide covariance_type options.' : undefined
        }
        onChange={(v) => set({ covariance_type: overrideOrUndef(v, d?.covariance_type) })}
      />

      <ParamNumber
        label="Tolerance (tol)"
        value={m.tol}
        placeholder={defaultPlaceholder(d?.tol)}
        onChange={(v) => set({ tol: overrideOrUndef(v, d?.tol) })}
        step={1e-4}
        min={0}
      />

      <ParamNumber
        label="Regularization (reg_covar)"
        value={m.reg_covar}
        placeholder={defaultPlaceholder(d?.reg_covar)}
        onChange={(v) => set({ reg_covar: overrideOrUndef(v, d?.reg_covar) })}
        step={1e-6}
        min={0}
      />

      <ParamNumber
        label="Max iterations"
        value={m.max_iter}
        placeholder={defaultPlaceholder(d?.max_iter)}
        onChange={(v) => set({ max_iter: overrideOrUndef(v, d?.max_iter) })}
        allowDecimal={false}
        min={1}
      />

      <ParamNumber
        label="Initializations (n_init)"
        value={m.n_init}
        placeholder={defaultPlaceholder(d?.n_init)}
        onChange={(v) => set({ n_init: overrideOrUndef(v, d?.n_init) })}
        allowDecimal={false}
        min={1}
      />

      <ParamSelect
        label="Init params"
        data={initParamsData}
        value={m.init_params}
        disabled={initParamsUnavailable}
        placeholder={initParamsUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.init_params)}
        description={initParamsUnavailable ? 'Schema did not provide init_params options.' : undefined}
        onChange={(v) => set({ init_params: overrideOrUndef(v, d?.init_params) })}
      />

      <ParamSelect
        label="Weight concentration prior type"
        data={weightConcentrationPriorTypeData}
        value={m.weight_concentration_prior_type}
        disabled={weightPriorTypeUnavailable}
        placeholder={
          weightPriorTypeUnavailable
            ? 'Schema enums unavailable'
            : defaultPlaceholder(d?.weight_concentration_prior_type)
        }
        description={
          weightPriorTypeUnavailable
            ? 'Schema did not provide weight_concentration_prior_type options.'
            : undefined
        }
        onChange={(v) =>
          set({
            weight_concentration_prior_type: overrideOrUndef(v, d?.weight_concentration_prior_type),
          })
        }
      />

      <ParamCheckbox
        label="Warm start"
        checked={effectiveValue(m.warm_start, d?.warm_start)}
        onChange={(checked) => set({ warm_start: overrideOrUndef(checked, d?.warm_start) })}
      />

      <ParamNumber
        label="Verbose"
        value={m.verbose}
        placeholder={defaultPlaceholder(d?.verbose)}
        onChange={(v) => set({ verbose: overrideOrUndef(v, d?.verbose) })}
        allowDecimal={false}
        min={0}
      />

      <ParamNumber
        label="Random state"
        value={m.random_state}
        placeholder={defaultPlaceholder(d?.random_state)}
        onChange={(v) => set({ random_state: overrideOrUndef(v, d?.random_state) })}
        allowDecimal={false}
      />
    </ParamGrid>
  );
}
