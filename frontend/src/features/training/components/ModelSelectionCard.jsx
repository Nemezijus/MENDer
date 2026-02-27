import { Card, Stack, Group, Text, Divider, Box, Button } from '@mantine/core';
import { useEffect, useMemo, useState, lazy, Suspense } from 'react';
import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { ModelIntroText } from '../../../shared/content/help/ModelIntroText.jsx';
import { getAlgoLabel } from '../../../shared/constants/algoLabels.js';
import {
  getVariantSchema,
  listDiscriminatorValues,
} from '../../../shared/utils/schema/jsonSchema.js';
import AlgoSelect from './modelSelection/AlgoSelect.jsx';
import AlgoParamsSwitch from './modelSelection/AlgoParamsSwitch.jsx';

const LazyModelHelpText = lazy(() =>
  import('../../../shared/content/help/ModelHelpText.jsx')
);

function algoKeyToLabel(algo) {
  if (!algo) return '';
  return getAlgoLabel(algo);
}

/** --------------- component --------------- **/

export default function ModelSelectionCard({
  model,
  onChange,
  schema,
  enums,
  models,
  taskOverride = null,
  showHelp = false,
}) {
  const m = model || {};
  // IMPORTANT:
  // - set(patch) merges into the current model.
  // - replace(next) fully replaces the model.
  //   We use replace() when switching algorithms to avoid "parameter leakage"
  //   across models that share field names (e.g. LogisticRegression.solver -> RidgeClassifier.solver,
  //   or Forest.max_features -> HistGradientBoosting.max_features).
  const set = (patch) => onChange?.({ ...m, ...patch });
  const replace = (next) => onChange?.({ ...(next || {}) });

  // IMPORTANT (Patch 1A3):
  // The frontend must never clone engine defaults into state.
  // State stores overrides only; defaults are displayed from schema.
  const applyAlgo = (algo) => {
    replace({ algo });
  };

  const inferredTask = useDataStore(
    (s) => s.taskSelected || s.inspectReport?.task_inferred || null,
  );
  const effectiveTask = taskOverride ?? inferredTask;

  // IMPORTANT: This card must not be the authority for "what models exist".
  // Available algorithms must come from the engine schema bundle.
  const schemaAlgos = listDiscriminatorValues(schema, 'algo');
  const defaultsAlgos = models?.defaults ? Object.keys(models.defaults) : null;
  const availableAlgos = (schemaAlgos || defaultsAlgos || []).map((a) => String(a));
  const hasInventory = availableAlgos.length > 0;

  // Filter by task using backend-provided meta[algo].task
  const meta = models?.meta || {};
  const matchesTask = (algo) => {
    if (!effectiveTask) return true; // no filter if task unknown
    let t = meta[algo]?.task; // 'classification' | 'regression' | 'unsupervised' | ...
    if (!t) return true; // if backend didn’t annotate, don’t hide it

    // Backwards-compat: older backends may use "clustering".
    if (t === 'clustering') t = 'unsupervised';

    if (Array.isArray(t)) return t.includes(effectiveTask);
    return t === effectiveTask;
  };

  // Deterministic ordering that does not enumerate algorithm inventories in the UI.
  // We sort by task group (when task is not locked) and then by display label.
  const taskRank = (t) => {
    if (t === 'classification') return 0;
    if (t === 'regression') return 1;
    if (t === 'unsupervised') return 2;
    return 9;
  };

  const taskOfAlgo = (algo) => {
    let t = meta[algo]?.task;
    if (!t) return null;
    if (t === 'clustering') t = 'unsupervised';
    if (Array.isArray(t)) return t[0] ?? null;
    return t;
  };

  const sortKey = (algo) => {
    const label = algoKeyToLabel(algo).toLowerCase();
    const t = taskOfAlgo(algo);
    return {
      rank: effectiveTask ? 0 : taskRank(t),
      label,
      algo,
    };
  };

  const visibleAlgos = (hasInventory ? availableAlgos : [])
    .filter((a) => (effectiveTask ? matchesTask(a) : true))
    .slice()
    .sort((a, b) => {
      const ka = sortKey(a);
      const kb = sortKey(b);
      if (ka.rank !== kb.rank) return ka.rank - kb.rank;
      if (ka.label < kb.label) return -1;
      if (ka.label > kb.label) return 1;
      return ka.algo < kb.algo ? -1 : ka.algo > kb.algo ? 1 : 0;
    });
  const visibleKey = useMemo(
    () => (visibleAlgos && visibleAlgos.length ? visibleAlgos.join('|') : ''),
    [visibleAlgos],
  );
  const algoData = (visibleAlgos ?? []).map((a) => ({
    value: String(a),
    label: algoKeyToLabel(a),
  }));

  const algoDataForSelect =
    algoData.length > 0
      ? algoData
      : m.algo
        ? [{ value: String(m.algo), label: algoKeyToLabel(m.algo) }]
        : [];

  // Ensure selection is valid when task / availability changes
  useEffect(() => {
    if (!hasInventory) return;
    if (!visibleAlgos.length) return;
    const current = m.algo;

    const isValid = current && visibleAlgos.includes(current);
    if (!isValid) {
      applyAlgo(visibleAlgos[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveTask, visibleKey, meta, m.algo, hasInventory]);

  const sub = getVariantSchema(schema, 'algo', m.algo);

  // Toggle for showing/hiding detailed help (block C) when help is enabled
  const [showDetails, setShowDetails] = useState(false);

  const controlsBody = (
    <>
      <AlgoSelect
        algoDataForSelect={algoDataForSelect}
        value={m.algo ?? null}
        hasInventory={hasInventory}
        onChange={applyAlgo}
      />

      <AlgoParamsSwitch algo={m.algo} m={m} set={set} sub={sub} enums={enums} d={models?.defaults?.[m.algo]} />

      <Divider my="xs" />
      <Text size="xs" c="dimmed">
        Algorithms are filtered by dataset task via <code>models.meta[algo].task</code>{' '}
        and your selected task in the Data panel.
      </Text>
    </>
  );

  // --- Render paths ---

  if (!showHelp) {
    return (
      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between">
            <Text fw={500}>Model</Text>
          </Group>

          <Stack gap="md">{controlsBody}</Stack>
        </Stack>
      </Card>
    );
  }

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        <Text fw={700} size="lg" align="center">
          Model
        </Text>

        <Group align="flex-start" gap="xl" grow wrap="nowrap">
          <Box style={{ flex: 1, minWidth: 0 }}>
            <Stack gap="md">{controlsBody}</Stack>
          </Box>

          <Box style={{ flex: 1, minWidth: 220 }}>
            <Stack gap="xs">
              <ModelIntroText />
              <Button
                size="xs"
                variant="subtle"
                onClick={() => setShowDetails((prev) => !prev)}
              >
                {showDetails ? 'Show less' : 'Show more'}
              </Button>
            </Stack>
          </Box>
        </Group>

        {showDetails && (
          <Box mt="md">
            <Suspense
              fallback={
                <Text size="xs" c="dimmed">
                  Loading help…
                </Text>
              }
            >
              <LazyModelHelpText
                selectedAlgo={m.algo}
                effectiveTask={effectiveTask}
                visibleAlgos={visibleAlgos}
              />
            </Suspense>
          </Box>
        )}
      </Stack>
    </Card>
  );
}
