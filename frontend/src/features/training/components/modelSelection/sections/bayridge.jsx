import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';

export default function BayridgeSection({ m, set, sub, enums }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Iterations (n_iter)"
          value={m.n_iter ?? 300}
          onChange={(v) => set({ n_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="Tolerance (tol)"
          value={m.tol ?? 1e-3}
          onChange={(v) => set({ tol: v })}
          step={1e-4}
          min={0}
        />
        <ParamNumber
          label="alpha_1"
          value={m.alpha_1 ?? 1e-6}
          onChange={(v) => set({ alpha_1: v })}
          step={1e-6}
          min={0}
        />
        <ParamNumber
          label="alpha_2"
          value={m.alpha_2 ?? 1e-6}
          onChange={(v) => set({ alpha_2: v })}
          step={1e-6}
          min={0}
        />
        <ParamNumber
          label="lambda_1"
          value={m.lambda_1 ?? 1e-6}
          onChange={(v) => set({ lambda_1: v })}
          step={1e-6}
          min={0}
        />
        <ParamNumber
          label="lambda_2"
          value={m.lambda_2 ?? 1e-6}
          onChange={(v) => set({ lambda_2: v })}
          step={1e-6}
          min={0}
        />
        <ParamCheckbox
          label="Compute score"
          checked={!!m.compute_score}
          onChange={(checked) => set({ compute_score: checked })}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(checked) => set({ fit_intercept: checked })}
        />
        <ParamCheckbox
          label="Copy X"
          checked={m.copy_X ?? true}
          onChange={(checked) => set({ copy_X: checked })}
        />
        <ParamCheckbox
          label="Verbose"
          checked={!!m.verbose}
          onChange={(checked) => set({ verbose: checked })}
        />
      </ParamGrid>
  );
}
