import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';

export default function LinregSection({ m, set, sub, enums }) {  return (
    <ParamGrid>
        <ParamCheckbox
          label="Fit intercept"
          checked={!!m.fit_intercept}
          onChange={(checked) =>
            set({ fit_intercept: checked })
          }
        />
        <ParamCheckbox
          label="Copy X"
          checked={m.copy_X ?? true}
          onChange={(checked) =>
            set({ copy_X: checked })
          }
        />
        <ParamNumber
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
        />
        <ParamCheckbox
          label="Positive coefficients"
          checked={!!m.positive}
          onChange={(checked) =>
            set({ positive: checked })
          }
        />
      </ParamGrid>
  );
}
