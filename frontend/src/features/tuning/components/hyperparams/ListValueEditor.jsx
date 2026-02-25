import { useEffect, useState } from 'react';
import { Textarea } from '@mantine/core';
import { countDecimals, parseScalar } from '../../utils/hyperparamUtils.js';

function computePrecision(tokens) {
  let maxDec = 0;
  tokens.forEach((tok) => {
    if (/^-?\d*\.?\d+$/.test(tok)) {
      const d = countDecimals(tok);
      if (d > maxDec) maxDec = d;
    }
  });
  return maxDec || null;
}

export default function ListValueEditor({
  label = 'Parameter values',
  description = 'Whitespace or comma-separated. Examples: sag saga lsqr   or   sqrt log2',
  placeholder = 'e.g. value1 value2 value3',
  onValuesChange,
}) {
  const [rawList, setRawList] = useState('');

  const emit = (values, { error = '', precision = null } = {}) => {
    onValuesChange?.({ values, error, precision });
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

    const precision = computePrecision(tokens);
    const values = tokens.map(parseScalar);
    const error = values.length < 2 ? 'Provide at least two values.' : '';
    emit(values, { error, precision });
  };

  // Ensure parent gets cleared state on mount.
  useEffect(() => {
    emit([], { error: '' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Textarea
      label={label}
      description={description}
      minRows={2}
      autosize
      value={rawList}
      onChange={(e) => updateFromList(e.currentTarget.value)}
      placeholder={placeholder}
    />
  );
}
