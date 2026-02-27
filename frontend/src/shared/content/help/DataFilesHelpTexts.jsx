import '../styles/help.css';

const SupportedFormatsText = () => (
  <p className="helpTextBodySm">
    Supported formats:{' '}
    <code className="helpCodeInline">.mat</code>,{' '}
    <code className="helpCodeInline">.npz</code>,{' '}
    <code className="helpCodeInline">.npy</code>,{' '}
    <code className="helpCodeInline">.csv</code>,{' '}
    <code className="helpCodeInline">.tsv</code>,{' '}
    <code className="helpCodeInline">.txt</code>,{' '}
    <code className="helpCodeInline">.h5/.hdf5</code>,{' '}
    <code className="helpCodeInline">.xlsx</code>.
  </p>
);

// -------------------------
// Training data
// -------------------------
export function TrainingDataIntroText() {
  return (
    <div className="helpStack helpStackTight">
      <div className="helpTitleStrong">Training data</div>
      <p className="helpTextBodySm">
        Training data is used to fit your model and estimate how well it generalizes.
        It usually consists of <b className="helpInlineStrong">features</b> (inputs) and{' '}
        <b className="helpInlineStrong">labels</b> (targets).
      </p>
      <p className="helpTextBodySm">
        <b className="helpInlineStrong">Features (X)</b> are the measurements you provide to the model
        (e.g. neuron responses, pixel intensities, sensor readings).{' '}
        <b className="helpInlineStrong">Labels (y)</b> are what you want the model to predict
        (e.g. class IDs for classification or a continuous value for regression).
      </p>
      <SupportedFormatsText />
      <p className="helpTextBodySm">
        For tabular files (<code className="helpCodeInline">.csv/.tsv/.txt/.xlsx</code>), a first-row header is allowed.
        If present, MENDer will strip it from the numeric matrix and preserve the column titles as
        feature names for future visualizations.
      </p>
    </div>
  );
}

export function TrainingIndividualFilesText() {
  return (
    <p className="helpTextBodySm">
      Use this option when features and labels are stored in{' '}
      <b className="helpInlineStrong">separate files</b>. You can paste a file path or browse and upload the file(s).
      For HDF5 (<code className="helpCodeInline">.h5/.hdf5</code>), use the{' '}
      <b className="helpInlineStrong">X key</b>/<b className="helpInlineStrong">y key</b> fields to specify the dataset names.
    </p>
  );
}

export function TrainingCompoundFileText() {
  return (
    <p className="helpTextBodySm">
      Use this option when your dataset is stored in a{' '}
      <b className="helpInlineStrong">single compound file</b>. For{' '}
      <code className="helpCodeInline">.npz</code> and HDF5 (<code className="helpCodeInline">.h5/.hdf5</code>),
      set <b className="helpInlineStrong">X key</b> and <b className="helpInlineStrong">y key</b> to the array/dataset names.
      For <code className="helpCodeInline">.xlsx</code>, keys can refer to sheet names (or indices).
    </p>
  );
}

// -------------------------
// Production data
// -------------------------
export function ProductionDataIntroText() {
  return (
    <div className="helpStack helpStackTight">
      <div className="helpTitleStrong">Production data</div>
      <p className="helpTextBodySm">
        Production (unseen) data is used to generate predictions with a trained or loaded model.
        Typically you provide <b className="helpInlineStrong">features (X)</b>.{' '}
        <b className="helpInlineStrong">Labels (y)</b> are optional—if present, you can score predictions later.
      </p>
      <SupportedFormatsText />
      <p className="helpTextBodySm">
        For tabular files (<code className="helpCodeInline">.csv/.tsv/.txt/.xlsx</code>), a first-row header is allowed and will be
        preserved as feature names.
      </p>
    </div>
  );
}

export function ProductionIndividualFilesText() {
  return (
    <p className="helpTextBodySm">
      Use this option when production features (and optionally labels) are stored in{' '}
      <b className="helpInlineStrong">separate files</b>. Paste a file path or browse and upload.
      For HDF5 (<code className="helpCodeInline">.h5/.hdf5</code>), use the{' '}
      <b className="helpInlineStrong">X key</b>/<b className="helpInlineStrong">y key</b> fields to specify the dataset names.
    </p>
  );
}

export function ProductionCompoundFileText() {
  return (
    <p className="helpTextBodySm">
      Use this option when production inputs are stored in a{' '}
      <b className="helpInlineStrong">single compound file</b>. For{' '}
      <code className="helpCodeInline">.npz</code> and HDF5 (<code className="helpCodeInline">.h5/.hdf5</code>), keys select the arrays/datasets.
      For <code className="helpCodeInline">.xlsx</code>, keys can refer to sheet names (or indices).
      This is convenient for deployment and helps avoid mismatched file paths.
    </p>
  );
}
