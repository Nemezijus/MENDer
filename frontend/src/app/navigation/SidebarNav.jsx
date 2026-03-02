import { Stack, Text, NavLink, Divider } from '@mantine/core';

import { SECTION_GROUPS } from './sections.js';

export default function SidebarNav({ active, onChange }) {
  return (
    <Stack gap="xs">
      {SECTION_GROUPS.map((group, groupIdx) => (
        <Stack gap={4} key={group.groupLabel}>
          <Text size="xs" fw={700} c="dimmed" tt="uppercase" pl="xs">
            {group.groupLabel}
          </Text>

          {group.items.map((item) => (
            <NavLink
              key={item.id}
              label={item.navLabel}
              description={item.description}
              active={active === item.id}
              onClick={() => onChange(item.id)}
              variant="light"
              radius="sm"
            />
          ))}

          {groupIdx < SECTION_GROUPS.length - 1 && <Divider my="xs" />}
        </Stack>
      ))}
    </Stack>
  );
}
