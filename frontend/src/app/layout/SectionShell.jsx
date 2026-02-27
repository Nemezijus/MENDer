import { Stack, Title } from '@mantine/core';

import '../styles/layout.css';

export default function SectionShell({ title, children }) {
  return (
    <Stack gap="md">
      <Title order={3} className="sectionShellTitle">
        {title}
      </Title>
      {children}
    </Stack>
  );
}
