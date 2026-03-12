import { Stack, Title } from '@mantine/core';

export default function SectionShell({ title, children, className }) {
  return (
    <Stack gap="md" className={className}>
      <Title order={3} className="sectionShellTitle">
        {title}
      </Title>
      {children}
    </Stack>
  );
}
