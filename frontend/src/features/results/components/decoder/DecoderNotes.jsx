import { Stack, Text } from '@mantine/core';

export default function DecoderNotes({ notes }) {
  const xs = Array.isArray(notes) ? notes.filter((n) => String(n).trim().length > 0) : [];
  if (!xs.length) return null;

  return (
    <Stack gap={4}>
      <Text size="sm" fw={600}>
        Notes
      </Text>
      <ul className="decoderNotesList">
        {xs.map((n, i) => (
          <li key={i}>
            <Text size="sm">{n}</Text>
          </li>
        ))}
      </ul>
    </Stack>
  );
}
