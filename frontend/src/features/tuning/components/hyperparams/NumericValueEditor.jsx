import { useEffect, useMemo, useState } from 'react';
import { Box, Group, NumberInput, SegmentedControl, Stack, Textarea } from '@mantine/core';
import { countDecimals, parseScalar } from '../../utils/hyperparamUtils.js';

function computePrecision(parts) {
  let maxDec = 0;
  for (const s of parts) {
    if (!s) continue;
    const d = countDecimals(s);
    if (d > maxDec) maxDec = d;
  }
  return maxDec || null;
}

export default function NumericValueEditor({ initialMode = 'range', onValuesChange }) {
  const [mode, setMode] = useState(initialMode); // 'range' | 'list'
  const [rangeFrom, setRangeFrom] = useState('');
  const [rangeTo, setRangeTo] = useState('');
  const [rangeStep, setRangeStep] = useState('');
  const [rawList, setRawList] = useState('');

  const emit = (values, { error = '', precision = null } = {}) => {
    onValuesChange?.({ values, error, precision });
  };

  const updateFromRange = (fromStr, toStr, stepStr) => {
    setRangeFrom(fromStr);
    setRangeTo(toStr);
    setRangeStep(stepStr);

    if (!fromStr || !toStr) {
      emit([], { error: 'Specify both "from" and "to" for a numeric range.' });
      return;
    }

    const from = Number(fromStr);
    const to = Number(toStr);
    if (!Number.isFinite(from) || !Number.isFinite(to)) {
      emit([], { error: 'Range bounds must be numeric.' });
      return;
    }
    if (from === to) {
      emit([], { error: 'Range bounds must not be equal.' });
      return;
    }
    if (from > to) {
      emit([], { error: '"from" should be less than "to".' });
      return;
    }

    let step;
    if (!stepStr) {
      const steps = 10;
      step = (to - from) / (steps - 1);
    } else {
      step = Number(stepStr);
      if (!Number.isFinite(step) || step <= 0) {
        emit([], { error: '"step" must be a positive number.' });
        return;
      }
    }

    const precision = computePrecision([fromStr, toStr, stepStr]);

    const values = [];
    let v = from;
    const epsilon = (to - from) * 1e-9;
    while (v <= to + epsilon) {
      values.push(v);
      v += step;
      if (values.length > 1000) break; // safety
    }

    const error = values.length < 2 ? 'Need at least two points in the range.' : '';
    emit(values, { error, precision });
  };

  const updateFromList = (raw) => {
    setRawList(raw);

    const tokens = raw
      .split(/[, \t\n\r]+/)
      .map((s) => s.trim())
      .filter(Boolean);

    if (!tokens.length) {
      emit([], { error: '' });
      return;
    }

    const precision = computePrecision(tokens.filter((tok) => /^-?\d*\.?\d+$/.test(tok)));
    const values = tokens.map(parseScalar);
    const error = values.length < 2 ? 'Provide at least two values.' : '';
    emit(values, { error, precision });
  };

  // When switching modes, clear errors/values until user enters new data.
  useEffect(() => {
    emit([], { error: '' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const modeOptions = useMemo(
    () => [
      { value: 'range', label: 'Range' },
      { value: 'list', label: 'Explicit values' },
    ],
    [],
  );

  return (
    <Stack gap="xs">
      <SegmentedControl size="xs" value={mode} onChange={setMode} data={modeOptions} />

      {mode === 'range' && (
        <Group align="flex-end">
          <Box className="tuningNumericCol">
            <NumberInput
              label="From"
              value={rangeFrom}
              onChange={(v) => updateFromRange(String(v ?? ''), rangeTo, rangeStep)}
            />
          </Box>
          <Box className="tuningNumericCol">
            <NumberInput
              label="Step"
              placeholder="auto"
              value={rangeStep}
              onChange={(v) => updateFromRange(rangeFrom, rangeTo, String(v ?? ''))}
            />
          </Box>
          <Box className="tuningNumericCol">
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
  );
}
