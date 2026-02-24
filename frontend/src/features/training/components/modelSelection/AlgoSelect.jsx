import { Select, Text } from '@mantine/core';

export default function AlgoSelect({
  algoDataForSelect,
  value,
  onChange,
  hasInventory,
}) {
  return (
    <>
      <Select
        label="Algorithm"
        data={algoDataForSelect}
        value={value ?? null}
        disabled={!hasInventory}
        placeholder={hasInventory ? 'Select an algorithm' : 'Schema not loaded'}
        onChange={(algo) => {
          if (algo) onChange?.(algo);
        }}
      />

      {!hasInventory && (
        <Text size="sm" c="dimmed">
          Model inventory is unavailable. Load the schema bundle to select a model.
        </Text>
      )}
    </>
  );
}
