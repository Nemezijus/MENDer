import { Stack, Text, List } from '@mantine/core';

export function FeatureIntroText() {
  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        What is feature extraction / selection?
      </Text>

      <Text size="xs" c="dimmed">
        Feature extraction or selection transforms or reduces your original
        feature space before fitting the model. It can make models simpler,
        improve generalisation, and sometimes reveal more interpretable
        structure in the data.
      </Text>
    </Stack>
  );
}

export function FeatureDetailsText({ selectedMethod }) {
  const selectedKey = selectedMethod ? String(selectedMethod).toLowerCase() : null;

  const isSelected = (name) => selectedKey === name;

  const labelStyle = (name) => ({
    fw: isSelected(name) ? 700 : 600,
    c: isSelected(name) ? 'blue' : undefined,
  });

  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        Available feature methods
      </Text>

      <List spacing={4} size="xs">
        {/* none */}
        <List.Item>
          <Text span {...labelStyle('none')}>
            none
          </Text>{' '}
          – use all original features without any dimensionality reduction or
          selection. A good baseline, and often sufficient for tree-based
          models.
        </List.Item>

        {/* PCA */}
        <List.Item>
          <Text span {...labelStyle('pca')}>
            pca
          </Text>{' '}
          – Principal Component Analysis. Creates orthogonal components that
          capture as much variance in the data as possible, often used to
          reduce dimensionality while keeping most of the signal.
          {isSelected('pca') && (
            <Stack gap={2} mt={4}>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>pca_n</Text> – optional target number of
                components. When left empty, the number of components is
                derived from the variance threshold.
              </Text>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>pca_var</Text> – fraction of total variance
                to retain (e.g. 0.95). Used when <Text span fw={600}>pca_n</Text>{' '}
                is empty.
              </Text>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>pca_whiten</Text> – if enabled, components
                are scaled to have unit variance. This can make models more
                sensitive to noise but sometimes helps optimisation.
              </Text>
            </Stack>
          )}
        </List.Item>

        {/* LDA */}
        <List.Item>
          <Text span {...labelStyle('lda')}>
            lda
          </Text>{' '}
          – Linear Discriminant Analysis. Finds directions that best separate
          the classes. Can both reduce dimensionality and act as a supervised
          projection.
          {isSelected('lda') && (
            <Stack gap={2} mt={4}>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>lda_n</Text> – optional target number of
                components (at most <Text span fw={600}>n_classes − 1</Text>).
                Leave empty to infer from the data.
              </Text>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>lda_solver</Text> – numerical solver used by
                LDA (e.g. <Text span fw={600}>svd</Text>,{' '}
                <Text span fw={600}>lsqr</Text>, <Text span fw={600}>eigen</Text>
                ). Some options support shrinkage.
              </Text>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>lda_shrinkage</Text> – optional shrinkage
                parameter used with{' '}
                <Text span fw={600}>lsqr</Text> or <Text span fw={600}>eigen</Text>{' '}
                solvers. Leave empty for no shrinkage.
              </Text>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>lda_tol</Text> – tolerance used in internal
                optimisation. Smaller values are stricter but may slow down or
                increase numerical sensitivity.
              </Text>
            </Stack>
          )}
        </List.Item>

        {/* SFS */}
        <List.Item>
          <Text span {...labelStyle('sfs')}>
            sfs
          </Text>{' '}
          – Sequential Feature Selection. Iteratively adds or removes features
          to find a subset that works well with the chosen model. This can be
          more expensive but sometimes yields very compact feature sets.
          {isSelected('sfs') && (
            <Stack gap={2} mt={4}>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>sfs_k</Text> – number of features to keep.
                Can be an integer or <Text span fw={600}>auto</Text> to let the
                algorithm search for a good subset size.
              </Text>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>sfs_direction</Text> – search direction:{' '}
                <Text span fw={600}>forward</Text> adds features one by one,
                while <Text span fw={600}>backward</Text> starts with all
                features and removes them.
              </Text>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>sfs_cv</Text> – number of cross-validation
                folds used to evaluate each candidate subset. Higher values are
                more reliable but slower.
              </Text>
              <Text size="xs" c="dimmed">
                <Text span fw={600}>sfs_n_jobs</Text> – number of parallel jobs
                for SFS. <Text span fw={600}>-1</Text> uses all available cores;
                leaving it empty uses the default.
              </Text>
            </Stack>
          )}
        </List.Item>
      </List>

      <Text size="xs" c="dimmed">
        If you are unsure, a good starting point is to use{' '}
        <Text span fw={600}>none</Text> for a baseline, then try{' '}
        <Text span fw={600}>pca</Text> or{' '}
        <Text span fw={600}>sfs</Text> if you suspect many features are
        redundant or noisy.
      </Text>
    </Stack>
  );
}

export default function FeatureHelpText({ selectedMethod }) {
  return (
    <Stack gap="sm">
      <FeatureDetailsText selectedMethod={selectedMethod} />
    </Stack>
  );
}
