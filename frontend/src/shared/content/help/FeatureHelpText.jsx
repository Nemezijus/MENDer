import '../styles/help.css';

export function FeatureIntroText() {
  return (
    <div className="helpStack helpStackXs">
      <div className="helpTitle">What is feature extraction / selection?</div>

      <p className="helpTextBodyXs">
        Feature extraction or selection transforms or reduces your original
        feature space before fitting the model. It can make models simpler,
        improve generalisation, and sometimes reveal more interpretable
        structure in the data.
      </p>
    </div>
  );
}

export function FeatureDetailsText({ selectedMethod }) {
  const selectedKey = selectedMethod ? String(selectedMethod).toLowerCase() : null;

  const isSelected = (name) => selectedKey === name;

  const labelClassName = (name) =>
    isSelected(name)
      ? 'helpAlgoLabel helpAlgoLabelSelected'
      : 'helpAlgoLabel';

  return (
    <div className="helpStack helpStackXs">
      <div className="helpTitle">Available feature methods</div>

      <ul className="helpList helpListXs helpListTight">
        <li>
          <span className={labelClassName('none')}>none</span> – use all original
          features without any dimensionality reduction or selection. A good
          baseline, and often sufficient for tree-based models.
        </li>

        <li>
          <span className={labelClassName('pca')}>pca</span> – Principal
          Component Analysis. Creates orthogonal components that capture as much
          variance in the data as possible, often used to reduce dimensionality
          while keeping most of the signal.

          {isSelected('pca') && (
            <div className="helpDetailBlock">
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">pca_n</span> – optional target
                number of components. When left empty, the number of components
                is derived from the variance threshold.
              </p>
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">pca_var</span> – fraction of
                total variance to retain (e.g. 0.95). Used when{' '}
                <span className="helpInlineLabel">pca_n</span> is empty.
              </p>
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">pca_whiten</span> – if enabled,
                components are scaled to have unit variance. This can make
                models more sensitive to noise but sometimes helps optimisation.
              </p>
            </div>
          )}
        </li>

        <li>
          <span className={labelClassName('lda')}>lda</span> – Linear Discriminant
          Analysis. Finds directions that best separate the classes. Can both
          reduce dimensionality and act as a supervised projection.

          {isSelected('lda') && (
            <div className="helpDetailBlock">
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">lda_n</span> – optional target
                number of components (at most{' '}
                <span className="helpInlineLabel">n_classes − 1</span>). Leave
                empty to infer from the data.
              </p>
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">lda_solver</span> – numerical
                solver used by LDA (e.g.{' '}
                <span className="helpInlineLabel">svd</span>,{' '}
                <span className="helpInlineLabel">lsqr</span>,{' '}
                <span className="helpInlineLabel">eigen</span>). Some options
                support shrinkage.
              </p>
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">lda_shrinkage</span> – optional
                shrinkage parameter used with{' '}
                <span className="helpInlineLabel">lsqr</span> or{' '}
                <span className="helpInlineLabel">eigen</span> solvers. Leave
                empty for no shrinkage.
              </p>
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">lda_tol</span> – tolerance used
                in internal optimisation. Smaller values are stricter but may
                slow down or increase numerical sensitivity.
              </p>
            </div>
          )}
        </li>

        <li>
          <span className={labelClassName('sfs')}>sfs</span> – Sequential Feature
          Selection. Iteratively adds or removes features to find a subset that
          works well with the chosen model. This can be more expensive but
          sometimes yields very compact feature sets.

          {isSelected('sfs') && (
            <div className="helpDetailBlock">
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">sfs_k</span> – number of
                features to keep. Can be an integer or{' '}
                <span className="helpInlineLabel">auto</span> to let the
                algorithm search for a good subset size.
              </p>
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">sfs_direction</span> – search
                direction: <span className="helpInlineLabel">forward</span> adds
                features one by one, while{' '}
                <span className="helpInlineLabel">backward</span> starts with all
                features and removes them.
              </p>
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">sfs_cv</span> – number of
                cross-validation folds used to evaluate each candidate subset.
                Higher values are more reliable but slower.
              </p>
              <p className="helpTextBodyXs">
                <span className="helpInlineLabel">sfs_n_jobs</span> – number of
                parallel jobs for SFS. <span className="helpInlineLabel">-1</span>{' '}
                uses all available cores; leaving it empty uses the default.
              </p>
            </div>
          )}
        </li>
      </ul>

      <p className="helpTextBodyXs">
        If you are unsure, a good starting point is to use{' '}
        <span className="helpInlineStrong">none</span> for a baseline, then try{' '}
        <span className="helpInlineStrong">pca</span> or{' '}
        <span className="helpInlineStrong">sfs</span> if you suspect many
        features are redundant or noisy.
      </p>
    </div>
  );
}

export default function FeatureHelpText({ selectedMethod }) {
  return (
    <div className="helpStack helpStackSm">
      <FeatureDetailsText selectedMethod={selectedMethod} />
    </div>
  );
}
