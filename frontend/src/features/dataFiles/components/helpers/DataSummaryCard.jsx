import { Card, Stack, Text, Alert, Group, Box, Divider } from '@mantine/core';

const SHORT_HASH_LEN = 7;

function formatMaybePath(p) {
  if (!p) return '—';
  return String(p);
}

function basename(path) {
  if (!path) return '';
  const p = String(path).replaceAll('\\', '/');
  const i = p.lastIndexOf('/');
  return i >= 0 ? p.slice(i + 1) : p;
}

function shortHashFromCanonicalFilename(fileName, n = SHORT_HASH_LEN) {
  // canonical stored uploads look like "<hex>.<ext>" where hex is sha256 (64),
  // but we allow 40+ to be tolerant.
  const m = String(fileName).match(/^([0-9a-f]{40,64})\.[A-Za-z0-9]+$/i);
  if (!m) return null;
  return m[1].slice(0, n);
}

function formatFallbackUploadLabelFromPath(path) {
  // If we only have a backend path, try to show a short hash + the canonical filename.
  // Example: "E:\...\uploads\<sha>.mat" -> "[04012af] <sha>.mat"
  if (!path) return '—';
  const file = basename(path);
  const sh = shortHashFromCanonicalFilename(file);
  return sh ? `[${sh}] ${file}` : String(path);
}

function formatClasses(classes, limit = 10) {
  if (!Array.isArray(classes)) return '—';
  if (classes.length <= limit) return classes.join(', ');

  // first 9, ellipsis, last 1  => total displayed = 11 tokens incl. "…"
  const head = classes.slice(0, limit - 1);
  const tail = classes[classes.length - 1];
  return `${head.join(', ')}, …, ${tail}`;
}

function describeImbalance(ratio) {
  // ratio = max/min
  if (!Number.isFinite(ratio)) return 'extremely imbalanced';
  if (ratio < 1.5) return 'roughly balanced';
  if (ratio < 3) return 'mildly imbalanced';
  if (ratio < 10) return 'moderately imbalanced';
  return 'severely imbalanced';
}

function computeImbalance(classCounts) {
  if (!classCounts || typeof classCounts !== 'object') return null;

  const vals = Object.values(classCounts)
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v) && v >= 0);

  if (!vals.length) return null;

  const maxC = Math.max(...vals);
  const minC = Math.min(...vals);

  if (maxC === 0) return null;
  const ratio = minC === 0 ? Infinity : maxC / minC;

  return { minC, maxC, ratio, descriptor: describeImbalance(ratio) };
}

function KeyValue({ k, v }) {
  return (
    <Group justify="space-between" gap="md" wrap="nowrap">
      <Text size="sm" c="dimmed">
        {k}
      </Text>
      <Text size="sm" fw={600} style={{ textAlign: 'right' }}>
        {v}
      </Text>
    </Group>
  );
}

function ClassificationDetails({ inspectReport }) {
  const classes = inspectReport?.classes ?? [];
  const nClasses = Array.isArray(classes) ? classes.length : 0;

  // class_counts may be legacy top-level OR inside y_summary depending on backend versions
  const classCounts =
    inspectReport?.class_counts ||
    inspectReport?.y_summary?.class_counts ||
    null;

  const imb = computeImbalance(classCounts);

  return (
    <Stack gap={6}>
      <Text size="sm" fw={700}>
        Classification details
      </Text>

      <KeyValue k="Number of labels" v={nClasses || '—'} />
      <KeyValue k="Labels (preview)" v={formatClasses(classes, 10)} />

      {imb && (
        <Text size="xs" c="dimmed">
          Class balance: min {imb.minC}, max {imb.maxC}
          {Number.isFinite(imb.ratio)
            ? ` (max/min ≈ ${imb.ratio.toFixed(2)}; ${imb.descriptor})`
            : ` (some classes have 0 samples; ${imb.descriptor})`}
        </Text>
      )}
    </Stack>
  );
}

function RegressionDetails({ inspectReport }) {
  const ySum = inspectReport?.y_summary || null;

  return (
    <Stack gap={6}>
      <Text size="sm" fw={700}>
        Regression details
      </Text>

      {!ySum ? (
        <Text size="sm" c="dimmed">
          No regression summary available.
        </Text>
      ) : (
        <>
          <KeyValue
            k="Target summary"
            v={`n=${ySum.n ?? '—'}, unique=${ySum.n_unique ?? '—'}`}
          />
          <KeyValue
            k="Min / Max"
            v={`${ySum.min ?? '—'} / ${ySum.max ?? '—'}`}
          />
          <KeyValue
            k="Mean ± Std"
            v={`${ySum.mean ?? '—'} ± ${ySum.std ?? '—'}`}
          />
        </>
      )}
    </Stack>
  );
}

function buildCompatibilityWarnings({ modelArtifact, inspectReport, effectiveTask }) {
  const warnings = [];
  if (!modelArtifact || !inspectReport) return warnings;

  const modelKind = modelArtifact.kind ?? null;
  const dataKind = effectiveTask ?? inspectReport?.task_inferred ?? null;

  if (modelKind && dataKind && modelKind !== dataKind) {
    warnings.push(
      `Task mismatch: model is "${modelKind}", but the data looks like "${dataKind}".`
    );
  }

  const modelNFeat = modelArtifact.n_features_in ?? null;
  const dataNFeat = inspectReport.n_features ?? null;
  if (
    Number.isFinite(modelNFeat) &&
    Number.isFinite(dataNFeat) &&
    modelNFeat !== dataNFeat
  ) {
    warnings.push(
      `Feature mismatch: model expects ${modelNFeat} features, but this data has ${dataNFeat}.`
    );
  }

  // Only check classes if data labels exist (production labels are optional)
  const modelClasses = modelArtifact.classes;
  const dataClasses = inspectReport.classes;
  if (
    modelKind === 'classification' &&
    Array.isArray(modelClasses) &&
    Array.isArray(dataClasses) &&
    dataClasses.length > 0 &&
    modelClasses.length !== dataClasses.length
  ) {
    warnings.push(
      `Label mismatch: model was trained with ${modelClasses.length} classes, but these labels contain ${dataClasses.length}.`
    );
  }

  return warnings;
}

export default function DataSummaryCard({
  inspectReport,
  effectiveTask,
  // optional: pass these from Training/Production stores
  xPath = null,
  yPath = null,
  npzPath = null,

  // NEW (persisted display names from stores)
  xDisplay = '',
  yDisplay = '',
  npzDisplay = '',

  // existing:
  showSuggestion = true,
  modelArtifact = null,
}) {
  const taskInferred = inspectReport?.task_inferred || null;
  const taskShown = effectiveTask || taskInferred || '—';

  const nSamples = inspectReport?.n_samples ?? null;
  const nFeatures = inspectReport?.n_features ?? null;

  const missingTotal = inspectReport?.missingness?.total ?? 0;

  const shouldSuggestFeatureReduction =
    Number.isFinite(nSamples) &&
    Number.isFinite(nFeatures) &&
    nSamples != null &&
    nFeatures != null &&
    nFeatures > 2 * nSamples;

  const compatWarnings = buildCompatibilityWarnings({
    modelArtifact,
    inspectReport,
    effectiveTask: taskShown === '—' ? null : taskShown,
  });

  // Prefer friendly display strings (persisted), otherwise fall back to a readable label from the backend path.
  const xShown = xDisplay?.trim()
    ? xDisplay.trim()
    : xPath
      ? formatFallbackUploadLabelFromPath(xPath)
      : '—';

  const yShown = yDisplay?.trim()
    ? yDisplay.trim()
    : yPath
      ? formatFallbackUploadLabelFromPath(yPath)
      : '—';

  const npzShown = npzDisplay?.trim()
    ? npzDisplay.trim()
    : npzPath
      ? formatFallbackUploadLabelFromPath(npzPath)
      : '—';

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        {/* Centered title */}
        <Group justify="space-between" align="center">
          <Box style={{ width: 90 }} />
          <Text fw={700} size="lg" align="center" style={{ flex: 1 }}>
            Summary
          </Text>
          <Box style={{ width: 90 }} />
        </Group>

        {!inspectReport && (
          <Text size="sm" c="dimmed">
            Run “Upload & Inspect” to see a summary of the loaded data.
          </Text>
        )}

        {inspectReport && (
          <>
            {/* Compatibility warning (production use-case) */}
            {compatWarnings.length > 0 && (
              <Alert color="orange" variant="light" title="Model compatibility warning">
                <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
                  {compatWarnings.join('\n')}
                </Text>
              </Alert>
            )}

            {/* 1) File paths */}
            <Stack gap={6}>
              <Text size="sm" fw={700}>
                Loaded files
              </Text>

              {npzPath ? (
                <>
                  <KeyValue k="Compound file" v={formatMaybePath(npzShown)} />
                  <Text size="xs" c="dimmed">
                    (This file is expected to contain both features (X) and labels (y), using the selected keys.)
                  </Text>
                </>
              ) : (
                <>
                  <KeyValue k="Features (X)" v={formatMaybePath(xShown)} />
                  <KeyValue k="Labels (y)" v={formatMaybePath(yShown)} />
                  <Text size="xs" c="dimmed">
                    (Labels are optional for production/unseen data.)
                  </Text>
                </>
              )}
            </Stack>

            <Divider />

            {/* 2) General info */}
            <Stack gap={6}>
              <Text size="sm" fw={700}>
                General
              </Text>

              <KeyValue
                k="Number of samples"
                v={Number.isFinite(nSamples) ? String(nSamples) : '—'}
              />
              <KeyValue
                k="Number of features"
                v={Number.isFinite(nFeatures) ? String(nFeatures) : '—'}
              />
              {missingTotal > 0 && (
                <KeyValue k="Missing values (total)" v={String(missingTotal)} />
              )}

              <KeyValue k="Recommended model type" v={taskShown} />
            </Stack>

            <Divider />

            {/* 3) Task-specific details */}
            {taskShown === 'classification' ? (
              <ClassificationDetails inspectReport={inspectReport} />
            ) : taskShown === 'regression' ? (
              <RegressionDetails inspectReport={inspectReport} />
            ) : (
              <Text size="sm" c="dimmed">
                Task-specific details will appear after task inference.
              </Text>
            )}

            {/* Suggestion (hide for production by passing showSuggestion={false}) */}
            {showSuggestion && shouldSuggestFeatureReduction && (
              <Alert color="blue" variant="light" mt="sm">
                <Text size="sm">
                  Suggestion:{' '}
                  {`Because the number of features (${nFeatures}) is much larger than the number of samples (${nSamples}), consider feature reduction (e.g. SFS, PCA, or LDA) or stronger regularization.`}
                </Text>
              </Alert>
            )}
          </>
        )}
      </Stack>
    </Card>
  );
}
