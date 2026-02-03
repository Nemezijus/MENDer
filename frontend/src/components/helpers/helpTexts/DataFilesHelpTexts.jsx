import { Stack, Text } from '@mantine/core';

const SupportedFormatsText = () => (
  <Text size="sm" c="dimmed">
    Supported formats: <code>.mat</code>, <code>.npz</code>, <code>.npy</code>, <code>.csv</code>, <code>.tsv</code>, <code>.txt</code>, <code>.h5/.hdf5</code>, <code>.xlsx</code>.
  </Text>
);

// -------------------------
// Training data
// -------------------------
export function TrainingDataIntroText() {
  return (
    <Stack gap={6}>
      <Text size="sm" fw={600}>
        Training data
      </Text>
      <Text size="sm" c="dimmed">
        Training data is used to fit your model and estimate how well it generalizes.
        It usually consists of <b>features</b> (inputs) and <b>labels</b> (targets).
      </Text>
      <Text size="sm" c="dimmed">
        <b>Features (X)</b> are the measurements you provide to the model (e.g. neuron responses,
        pixel intensities, sensor readings). <b>Labels (y)</b> are what you want the model to predict
        (e.g. class IDs for classification or a continuous value for regression).
      </Text>
      <SupportedFormatsText />
      <Text size="sm" c="dimmed">
        For tabular files (<code>.csv/.tsv/.txt/.xlsx</code>), a first-row header is allowed.
        If present, MENDer will strip it from the numeric matrix and preserve the column titles as
        feature names for future visualizations.
      </Text>
    </Stack>
  );
}

export function TrainingIndividualFilesText() {
  return (
    <Text size="sm" c="dimmed">
      Use this option when features and labels are stored in <b>separate files</b>.
      You can paste a file path or browse and upload the file(s).
      For HDF5 (<code>.h5/.hdf5</code>), use the <b>X key</b>/<b>y key</b> fields to specify the dataset names.
    </Text>
  );
}

export function TrainingCompoundFileText() {
  return (
    <Text size="sm" c="dimmed">
      Use this option when your dataset is stored in a <b>single compound file</b>.
      For <code>.npz</code> and HDF5 (<code>.h5/.hdf5</code>), set <b>X key</b> and <b>y key</b> to the array/dataset names.
      For <code>.xlsx</code>, keys can refer to sheet names (or indices).
    </Text>
  );
}

// -------------------------
// Production data
// -------------------------
export function ProductionDataIntroText() {
  return (
    <Stack gap={6}>
      <Text size="sm" fw={600}>
        Production data
      </Text>
      <Text size="sm" c="dimmed">
        Production (unseen) data is used to generate predictions with a trained or loaded model.
        Typically you provide <b>features (X)</b>. <b>Labels (y)</b> are optional—if present, you can score
        predictions later.
      </Text>
      <SupportedFormatsText />
      <Text size="sm" c="dimmed">
        For tabular files (<code>.csv/.tsv/.txt/.xlsx</code>), a first-row header is allowed and will be
        preserved as feature names.
      </Text>
    </Stack>
  );
}

export function ProductionIndividualFilesText() {
  return (
    <Text size="sm" c="dimmed">
      Use this option when production features (and optionally labels) are stored in <b>separate files</b>.
      Paste a file path or browse and upload.
      For HDF5 (<code>.h5/.hdf5</code>), use the <b>X key</b>/<b>y key</b> fields to specify the dataset names.
    </Text>
  );
}

export function ProductionCompoundFileText() {
  return (
    <Text size="sm" c="dimmed">
      Use this option when production inputs are stored in a <b>single compound file</b>.
      For <code>.npz</code> and HDF5 (<code>.h5/.hdf5</code>), keys select the arrays/datasets.
      For <code>.xlsx</code>, keys can refer to sheet names (or indices).
      This is convenient for deployment and helps avoid mismatched file paths.
    </Text>
  );
}
