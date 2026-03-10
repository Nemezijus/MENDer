import "../styles/help.css";
const SupportedFormatsText = () => (
  <p className="helpLead">
    {" "}
    Supported formats: <code className="helpCode">.mat</code>,{" "}
    <code className="helpCode">.npz</code>,{" "}
    <code className="helpCode">.npy</code>,{" "}
    <code className="helpCode">.csv</code>,{" "}
    <code className="helpCode">.tsv</code>,{" "}
    <code className="helpCode">.txt</code>,{" "}
    <code className="helpCode">.h5/.hdf5</code>,{" "}
    <code className="helpCode">.xlsx</code>.{" "}
  </p>
); // -------------------------
// Training data
// -------------------------
export function TrainingDataIntroText() {
  return (
    <div className="helpSectionPreview">
      {" "}
      <div className="helpTitleStrong">Training data</div>{" "}
      <p className="helpLead">
        {" "}
        Training data is used to fit your model and estimate how well it
        generalizes. It usually consists of{" "}
        <b className="helpStrong">features</b> (inputs) and{" "}
        <b className="helpStrong">labels</b> (targets).{" "}
      </p>{" "}
      <p className="helpLead">
        {" "}
        <b className="helpStrong">Features (X)</b> are the measurements you
        provide to the model (e.g. neuron responses, pixel intensities, sensor
        readings). <b className="helpStrong">Labels (y)</b> are what you want
        the model to predict (e.g. class IDs for classification or a continuous
        value for regression).{" "}
      </p>{" "}
      <SupportedFormatsText />{" "}
      <p className="helpLead">
        {" "}
        For tabular files (
        <code className="helpCode">.csv/.tsv/.txt/.xlsx</code>), a first-row
        header is allowed. If present, MENDer will strip it from the numeric
        matrix and preserve the column titles as feature names for future
        visualizations.{" "}
      </p>{" "}
    </div>
  );
}
export function TrainingIndividualFilesText() {
  return (
    <p className="helpLead">
      {" "}
      Use this option when features and labels are stored in{" "}
      <b className="helpStrong">separate files</b>. You can paste a file path or
      browse and upload the file(s). For HDF5 (
      <code className="helpCode">.h5/.hdf5</code>), use the{" "}
      <b className="helpStrong">X key</b>/<b className="helpStrong">y key</b>{" "}
      fields to specify the dataset names.{" "}
    </p>
  );
}
export function TrainingCompoundFileText() {
  return (
    <p className="helpLead">
      {" "}
      Use this option when your dataset is stored in a{" "}
      <b className="helpStrong">single compound file</b>. For{" "}
      <code className="helpCode">.npz</code> and HDF5 (
      <code className="helpCode">.h5/.hdf5</code>), set{" "}
      <b className="helpStrong">X key</b> and{" "}
      <b className="helpStrong">y key</b> to the array/dataset names. For{" "}
      <code className="helpCode">.xlsx</code>, keys can refer to sheet names (or
      indices).{" "}
    </p>
  );
} // -------------------------
// Production data
// -------------------------
export function ProductionDataIntroText() {
  return (
    <div className="helpSectionPreview">
      {" "}
      <div className="helpTitleStrong">Production data</div>{" "}
      <p className="helpLead">
        {" "}
        Production (unseen) data is used to generate predictions with a trained
        or loaded model. Typically you provide{" "}
        <b className="helpStrong">features (X)</b>.{" "}
        <b className="helpStrong">Labels (y)</b> are optional—if present, you
        can score predictions later.{" "}
      </p>{" "}
      <SupportedFormatsText />{" "}
      <p className="helpLead">
        {" "}
        For tabular files (
        <code className="helpCode">.csv/.tsv/.txt/.xlsx</code>), a first-row
        header is allowed and will be preserved as feature names.{" "}
      </p>{" "}
    </div>
  );
}
export function ProductionIndividualFilesText() {
  return (
    <p className="helpLead">
      {" "}
      Use this option when production features (and optionally labels) are
      stored in <b className="helpStrong">separate files</b>. Paste a file path
      or browse and upload. For HDF5 (
      <code className="helpCode">.h5/.hdf5</code>), use the{" "}
      <b className="helpStrong">X key</b>/<b className="helpStrong">y key</b>{" "}
      fields to specify the dataset names.{" "}
    </p>
  );
}
export function ProductionCompoundFileText() {
  return (
    <p className="helpLead">
      {" "}
      Use this option when production inputs are stored in a{" "}
      <b className="helpStrong">single compound file</b>. For{" "}
      <code className="helpCode">.npz</code> and HDF5 (
      <code className="helpCode">.h5/.hdf5</code>), keys select the
      arrays/datasets. For <code className="helpCode">.xlsx</code>, keys can
      refer to sheet names (or indices). This is convenient for deployment and
      helps avoid mismatched file paths.{" "}
    </p>
  );
}
