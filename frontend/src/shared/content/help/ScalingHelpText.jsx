import '../styles/help.css';

export function ScalingIntroText() {
  return (
    <div className="helpStack helpStackXs">
      <div className="helpTitle">What is scaling?</div>

      <p className="helpTextBodyXs">
        Scaling changes the numerical range and distribution of your features
        before they are given to the model. This often makes optimisation
        easier and prevents some features from dominating others simply because
        they have larger numeric values.
      </p>
    </div>
  );
}

export function ScalingDetailsText({ selectedScaling }) {
  // Normalise selected scaling value (we use the raw value, not the label)
  const selectedKey = selectedScaling
    ? String(selectedScaling).toLowerCase()
    : null;

  const isSelected = (name) => selectedKey === name;

  const labelClassName = (name) =>
    isSelected(name)
      ? 'helpAlgoLabel helpAlgoLabelSelected'
      : 'helpAlgoLabel';

  return (
    <div className="helpStack helpStackXs">
      <div className="helpTitle">When to use each option</div>

      <ul className="helpList helpListXs helpListTight">
        <li>
          <span className={labelClassName('none')}>None</span> – keep raw feature
          scales. Useful for tree-based models or when your features are already
          on comparable scales.
        </li>

        <li>
          <span className={labelClassName('standard')}>Standard</span> – subtract
          mean and divide by standard deviation. Works well for many linear
          models, SVMs and neural networks.
        </li>

        <li>
          <span className={labelClassName('robust')}>Robust</span> – uses median
          and interquartile range instead of mean and standard deviation. Prefer
          this when your features have strong outliers.
        </li>

        <li>
          <span className={labelClassName('minmax')}>MinMax</span> – rescales each
          feature to a fixed range (commonly [0, 1]). Useful when you want all
          inputs to be strictly bounded.
        </li>

        <li>
          <span className={labelClassName('maxabs')}>MaxAbs</span> – scales features
          to lie within [-1, 1] based on their maximum absolute value. Handy for
          sparse data where you do not want to destroy sparsity.
        </li>

        <li>
          <span className={labelClassName('quantile')}>Quantile</span> – transforms
          each feature to follow a target distribution (e.g. uniform or normal).
          Can make highly skewed features more comparable, but is more aggressive
          than simple scaling.
        </li>
      </ul>

      <p className="helpTextBodyXs">
        If you are unsure, a good default for many models is{' '}
        <span className="helpInlineStrong">Standard</span> scaling, unless you
        know your data have heavy outliers, in which case{' '}
        <span className="helpInlineStrong">Robust</span> is often safer.
      </p>
    </div>
  );
}

export default function ScalingHelpText({ selectedScaling }) {
  return (
    <div className="helpStack helpStackSm">
      <ScalingDetailsText selectedScaling={selectedScaling} />
    </div>
  );
}
