import { useEffect, useMemo, useState } from 'react';
import { Stack, Text } from '@mantine/core';
import { getVariantSchema } from '../../../../shared/utils/schema/jsonSchema.js';
import {
  formatValueForDisplay,
  getParamInfo,
  humanizeParamName,
  summarizeValues,
} from '../../utils/hyperparamUtils.js';
import BooleanNote from '../hyperparams/BooleanNote.jsx';
import EnumValueSelector from '../hyperparams/EnumValueSelector.jsx';
import ListValueEditor from '../hyperparams/ListValueEditor.jsx';
import NumericValueEditor from '../hyperparams/NumericValueEditor.jsx';
import ParamNameSelect from '../hyperparams/ParamNameSelect.jsx';

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
  const [enumSelection, setEnumSelection] = useState([]);

  const [helperError, setHelperError] = useState('');
  const [displayPrecision, setDisplayPrecision] = useState(null);

  const sub = useMemo(() => getVariantSchema(schema, 'algo', algo), [schema, algo]);

  const paramOptions = useMemo(() => {
    const props = sub?.properties ?? {};
    return Object.keys(props)
      .filter((key) => key !== 'algo')
      .map((key) => ({ value: key, label: humanizeParamName(key) }));
  }, [sub]);

  const selectedName = value?.paramName ?? '';

  const paramInfo = useMemo(
    () =>
      selectedName
        ? getParamInfo(sub, selectedName)
        : { kind: 'other', allowedValues: null },
    [sub, selectedName],
  );

  // Reset local controls when parameter changes
  useEffect(() => {
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

  const showNumericControls = paramInfo.kind === 'numeric';
  const showListOnlyControls = paramInfo.kind === 'other';
  const showEnumControls = paramInfo.kind === 'enum';
  const showBooleanNote = paramInfo.kind === 'boolean';

  const effectiveValues = Array.isArray(value?.values) ? value.values : [];
  const displayValues = summarizeValues(effectiveValues, 10);

  const handleEditorValues = ({ values, error, precision }) => {
    setHelperError(error || '');
    setDisplayPrecision(precision ?? null);
    onChange?.({ paramName: selectedName, values: Array.isArray(values) ? values : [] });
  };

  return (
    <Stack gap="sm">
      <ParamNameSelect
        label={label || 'Hyperparameter to vary'}
        options={paramOptions}
        value={selectedName}
        onChange={handleParamChange}
      />

      {selectedName && showBooleanNote && <BooleanNote />}

      {selectedName && showEnumControls && (
        <EnumValueSelector
          allowedValues={paramInfo.allowedValues}
          selectedValues={enumSelection}
          onToggle={toggleEnumValue}
        />
      )}

      {selectedName && showNumericControls && (
        <NumericValueEditor
          key={`num-${selectedName}`}
          onValuesChange={handleEditorValues}
        />
      )}

      {selectedName && showListOnlyControls && (
        <ListValueEditor
          key={`list-${selectedName}`}
          onValuesChange={handleEditorValues}
        />
      )}

      {helperError && (
        <Text size="xs" c="red">
          {helperError}
        </Text>
      )}

      {selectedName && effectiveValues.length >= 2 && (
        <Text size="xs" c="dimmed">
          Effective values: {displayValues
            .map((v) => formatValueForDisplay(v, displayPrecision))
            .join(', ')}
        </Text>
      )}
    </Stack>
  );
}
