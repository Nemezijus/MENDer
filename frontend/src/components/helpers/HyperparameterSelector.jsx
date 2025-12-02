import { useEffect, useMemo, useState } from 'react';
import {
  Stack,
  Group,
  Select,
  Text,
  NumberInput,
  Textarea,
  Checkbox,
  SegmentedControl,
  Box,
} from '@mantine/core';

/**
 * Helpers adapted from ModelSelectionCard
 */

function resolveRef(schema, ref) {
  if (!schema || !ref || typeof ref !== 'string') return null;
  const prefix = '#/$defs/';
  if (!ref.startsWith(prefix)) return null;
  const key = ref.slice(prefix.length);
  return schema?.$defs?.[key] ?? null;
}

function getAlgoSchema(schema, algo) {
  if (!schema || !algo) return null;

  const mapping = schema?.discriminator?.mapping;
  if (mapping && mapping[algo]) {
    const target = resolveRef(schema, mapping[algo]);
    if (target) return target;
  }

  const variants = schema?.oneOf || schema?.anyOf || [];
  for (const entry of variants) {
    const target = entry?.$ref ? resolveRef(schema, entry.$ref) : entry;
    const alg = target?.properties?.algo?.const ?? target?.properties?.algo?.default;
    if (alg === algo) return target || null;
  }

  return null;
}

function getParamSchema(sub, key) {
  if (!sub || !key) return null;
  return sub.properties?.[key] ?? null;
}

function collectTypes(p) {
  const out = new Set();
  if (!p) return out;
  const direct = Array.isArray(p.type) ? p.type : (p.type ? [p.type] : []);
  for (const t of direct) out.add(t);
  const unions = [...(p.anyOf ?? []), ...(p.oneOf ?? [])];
  for (const u of unions) {
    const ts = Array.isArray(u.type) ? u.type : (u.type ? [u.type] : []);
    for (const t of ts) out.add(t);
  }
  return out;
}

function collectEnumValues(p) {
  if (!p) return null;
  if (Array.isArray(p.enum)) return p.enum;

  const list = (p.anyOf ?? p.oneOf ?? []).flatMap((x) => {
    if (Array.isArray(x.enum)) return x.enum;
    if (x.const != null) return [x.const];
    if (x.type === 'null') return [null];
    return [];
  });
  return list.length ? list : null;
}

function getParamInfo(sub, key) {
  const p = getParamSchema(sub, key);
  if (!p) return { kind: 'other', allowedValues: null };

  const enums = collectEnumValues(p);
  const types = collectTypes(p);

  // Boolean by explicit type or enum {true,false}
  if (types.has('boolean')) {
    return { kind: 'boolean', allowedValues: [true, false] };
  }
  if (enums) {
    const uniq = Array.from(new Set(enums));
    if (
      uniq.length === 2 &&
      uniq.includes(true) &&
      uniq.includes(false)
    ) {
      return { kind: 'boolean', allowedValues: [true, false] };
    }
    return { kind: 'enum', allowedValues: enums };
  }

  if (types.has('number') || types.has('integer')) {
    return { kind: 'numeric', allowedValues: null };
  }

  return { kind: 'other', allowedValues: null };
}

function parseScalar(raw) {
  const lower = raw.toLowerCase();

  if (lower === 'true' || lower === 'yes') return true;
  if (lower === 'false' || lower === 'no') return false;
  if (lower === 'none' || lower === 'null') return null;

  if (/^-?\d+$/.test(raw)) {
    const v = parseInt(raw, 10);
    return Number.isNaN(v) ? raw : v;
  }
  if (/^-?\d*\.\d+$/.test(raw)) {
    const v = parseFloat(raw);
    return Number.isNaN(v) ? raw : v;
  }
  return raw;
}

function countDecimals(str) {
  const m = String(str).match(/\.(\d+)/);
  return m ? m[1].length : 0;
}

/**
 * HyperparameterSelector
 *
 * Props:
 *  - schema: models.schema from backend
 *  - model: current union model config (must contain .algo)
 *  - value: { paramName: string | null, values: any[] }
 *  - onChange: ({ paramName, values }) => void
 */
export default function HyperparameterSelector({
  schema,
  model,
  value,
  onChange,
  label,
}) {
  const algo = model?.algo ?? null;
  const [mode, setMode] = useState('range'); // 'range' | 'list'
  const [rangeFrom, setRangeFrom] = useState('');
  const [rangeTo, setRangeTo] = useState('');
  const [rangeStep, setRangeStep] = useState('');
  const [rawList, setRawList] = useState('');
  const [enumSelection, setEnumSelection] = useState([]);

  const [helperError, setHelperError] = useState('');
  const [displayPrecision, setDisplayPrecision] = useState(null);

  const sub = useMemo(() => getAlgoSchema(schema, algo), [schema, algo]);

  const paramOptions = useMemo(() => {
    const props = sub?.properties ?? {};
    return Object.keys(props)
      .filter((key) => key !== 'algo')
      .map((key) => ({ value: key, label: key }));
  }, [sub]);

  const selectedName = value?.paramName ?? '';

  const paramInfo = useMemo(
    () => (selectedName ? getParamInfo(sub, selectedName) : { kind: 'other', allowedValues: null }),
    [sub, selectedName],
  );

  // Reset local controls when parameter changes
  useEffect(() => {
    setRangeFrom('');
    setRangeTo('');
    setRangeStep('');
    setRawList('');
    setHelperError('');
    setDisplayPrecision(null);

    if (paramInfo.kind === 'enum') {
      const allowed = paramInfo.allowedValues ?? [];
      const initial =
        Array.isArray(value?.values) && value.values.length
          ? value.values
          : allowed;
      setEnumSelection(initial);
      if (initial.length >= 2) {
        onChange?.({ paramName: selectedName, values: initial });
      }
    } else if (paramInfo.kind === 'boolean') {
      const vals = [true, false];
      setEnumSelection(vals);
      onChange?.({ paramName: selectedName, values: vals });
    } else {
      setEnumSelection([]);
      if (selectedName) {
        onChange?.({ paramName: selectedName, values: value?.values ?? [] });
      } else {
        onChange?.({ paramName: '', values: [] });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedName, paramInfo.kind]);

  function handleParamChange(name) {
    const paramName = name || '';
    onChange?.({ paramName, values: [] });
  }

  function updateFromRange(fromStr, toStr, stepStr) {
    setRangeFrom(fromStr);
    setRangeTo(toStr);
    setRangeStep(stepStr);
    setHelperError('');

    if (!fromStr || !toStr) {
      setHelperError('Specify both "from" and "to" for a numeric range.');
      onChange?.({ paramName: selectedName, values: [] });
      return;
    }

    const from = Number(fromStr);
    const to = Number(toStr);
    if (!Number.isFinite(from) || !Number.isFinite(to)) {
      setHelperError('Range bounds must be numeric.');
      onChange?.({ paramName: selectedName, values: [] });
      return;
    }
    if (from === to) {
      setHelperError('Range bounds must not be equal.');
      onChange?.({ paramName: selectedName, values: [] });
      return;
    }
    if (from > to) {
      setHelperError('"from" should be less than "to".');
      onChange?.({ paramName: selectedName, values: [] });
      return;
    }

    let step;
    if (!stepStr) {
      const steps = 10;
      step = (to - from) / (steps - 1);
    } else {
      step = Number(stepStr);
      if (!Number.isFinite(step) || step <= 0) {
        setHelperError('"step" must be a positive number.');
        onChange?.({ paramName: selectedName, values: [] });
        return;
      }
    }

    // Precision: highest decimals from from/to/step
    let maxDec = 0;
    [fromStr, toStr, stepStr].forEach((s) => {
      if (!s) return;
      const d = countDecimals(s);
      if (d > maxDec) maxDec = d;
    });
    setDisplayPrecision(maxDec || null);

    const values = [];
    let v = from;
    const epsilon = (to - from) * 1e-9;
    while (v <= to + epsilon) {
      values.push(v);
      v += step;
      if (values.length > 1000) break; // safety
    }

    if (values.length < 2) {
      setHelperError('Need at least two points in the range.');
    }
    onChange?.({ paramName: selectedName, values });
  }

  function updateFromList(raw) {
    setRawList(raw);
    setHelperError('');

    const tokens = raw
      .split(/[, \t\n\r]+/)
      .map((s) => s.trim())
      .filter(Boolean);

    if (!tokens.length) {
      onChange?.({ paramName: selectedName, values: [] });
      return;
    }

    // Precision: highest decimals in numeric tokens
    let maxDec = 0;
    tokens.forEach((tok) => {
      if (/^-?\d*\.?\d+$/.test(tok)) {
        const d = countDecimals(tok);
        if (d > maxDec) maxDec = d;
      }
    });
    setDisplayPrecision(maxDec || null);

    const values = tokens.map(parseScalar);
    if (values.length < 2) {
      setHelperError('Provide at least two values.');
    }
    onChange?.({ paramName: selectedName, values });
  }

  function toggleEnumValue(v) {
    const current = Array.isArray(enumSelection) ? enumSelection : [];
    let next;
    if (current.includes(v)) {
      next = current.filter((x) => x !== v);
    } else {
      next = [...current, v];
    }
    setEnumSelection(next);
    if (next.length < 2) {
      setHelperError('Select at least two options.');
    } else {
      setHelperError('');
    }
    onChange?.({ paramName: selectedName, values: next });
    setDisplayPrecision(null);
  }

  const showNumericControls = paramInfo.kind === 'numeric' || paramInfo.kind === 'other';
  const showEnumControls = paramInfo.kind === 'enum';
  const showBooleanNote = paramInfo.kind === 'boolean';

  function formatValueForDisplay(v) {
    if (typeof v === 'number' && Number.isFinite(v) && displayPrecision != null) {
      return v.toFixed(displayPrecision);
    }
    return String(v);
  }

  const effectiveValues = Array.isArray(value?.values) ? value.values : [];
  let displayValues = effectiveValues;

  if (effectiveValues.length > 10) {
    displayValues = [
      ...effectiveValues.slice(0, 9),
      '…',
      effectiveValues[effectiveValues.length - 1],
    ];
  }

  return (
    <Stack gap="sm">
      <Select
        label={
          label ||
          (paramOptions.length
            ? 'Hyperparameter to vary'
            : 'Hyperparameter to vary')
        }
        placeholder={
          paramOptions.length
            ? 'Pick a parameter (e.g. C, max_depth)'
            : 'No parameters available for this model'
        }
        data={paramOptions}
        value={selectedName || null}
        onChange={handleParamChange}
        searchable
        clearable
      />

      {selectedName && showBooleanNote && (
        <Text size="sm" c="dimmed">
          This is a boolean parameter. The validation curve will automatically
          evaluate both <Text span fw={500}>true</Text> and <Text span fw={500}>false</Text>.
        </Text>
      )}

      {selectedName && showEnumControls && Array.isArray(paramInfo.allowedValues) && (
        <Box>
          <Text size="sm" fw={500} mb={4}>
            Values to include
          </Text>
          <Stack gap={4}>
            {paramInfo.allowedValues.map((v) => (
              <Checkbox
                key={String(v)}
                label={String(v)}
                checked={enumSelection.includes(v)}
                onChange={() => toggleEnumValue(v)}
              />
            ))}
          </Stack>
        </Box>
      )}

      {selectedName && showNumericControls && (
        <Stack gap="xs">
          <SegmentedControl
            size="xs"
            value={mode}
            onChange={setMode}
            data={[
              { value: 'range', label: 'Range' },
              { value: 'list', label: 'Explicit values' },
            ]}
          />

          {mode === 'range' && (
            <Group align="flex-end">
                <Box style={{ flex: 1 }}>
                <NumberInput
                    label="From"
                    value={rangeFrom}
                    onChange={(v) => updateFromRange(String(v ?? ''), rangeTo, rangeStep)}
                />
                </Box>
                <Box style={{ flex: 1 }}>
                <NumberInput
                    label="Step"
                    placeholder="auto"
                    value={rangeStep}
                    onChange={(v) => updateFromRange(rangeFrom, rangeTo, String(v ?? ''))}
                />
                </Box>
                <Box style={{ flex: 1 }}>
                <NumberInput
                    label="To"
                    value={rangeTo}
                    onChange={(v) => updateFromRange(rangeFrom, String(v ?? ''), rangeStep)}
                />
                </Box>
            </Group>
            )}


          {mode === 'list' && (
            <Textarea
              label="Parameter values"
              description="Whitespace or comma-separated. Examples: 0.01 0.1 1 10   or   linear rbf poly"
              minRows={2}
              autosize
              value={rawList}
              onChange={(e) => updateFromList(e.currentTarget.value)}
              placeholder="e.g. 0.01 0.1 1 10"
            />
          )}
        </Stack>
      )}

      {helperError && (
        <Text size="xs" c="red">
          {helperError}
        </Text>
      )}

      {selectedName && effectiveValues.length >= 2 && (
        <Text size="xs" c="dimmed">
          Effective values:{' '}
          {displayValues.map((v) => formatValueForDisplay(v)).join(', ')}
        </Text>
      )}
    </Stack>
  );
}
