import { Stack, Title } from '@mantine/core';

export default function SectionShell({ title, children }) {
  return (
    <Stack gap="md">
      <Title order={3}>{title}</Title>
      {children}
    </Stack>
  );
}
