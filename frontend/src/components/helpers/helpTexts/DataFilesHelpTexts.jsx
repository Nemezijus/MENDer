import { Stack, Text } from '@mantine/core';

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
    </Stack>
  );
}

export function TrainingIndividualFilesText() {
  return (
    <Text size="sm" c="dimmed">
      Use this option when features and labels are stored in <b>separate files</b>.
      You can paste a file path or browse and upload the file(s).
    </Text>
  );
}

export function TrainingCompoundFileText() {
  return (
    <Text size="sm" c="dimmed">
      Use this option when your dataset is stored in a <b>single compound file</b> (e.g. <code>.npz</code>).
      You may need to specify <b>X key</b> and <b>y key</b> to identify which arrays contain features and labels.
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
    </Stack>
  );
}

export function ProductionIndividualFilesText() {
  return (
    <Text size="sm" c="dimmed">
      Use this option when production features (and optionally labels) are stored in <b>separate files</b>.
      Paste a file path or browse and upload.
    </Text>
  );
}

export function ProductionCompoundFileText() {
  return (
    <Text size="sm" c="dimmed">
      Use this option when production inputs are stored in a <b>single compound file</b> (e.g. <code>.npz</code>).
      This is convenient for deployment and avoids mismatched file paths.
    </Text>
  );
}
