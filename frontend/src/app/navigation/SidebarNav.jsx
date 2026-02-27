import { Stack, Text, NavLink, Divider } from '@mantine/core';

import '../styles/navigation.css';

import { SECTION_GROUPS } from './sections.js';

export default function SidebarNav({ active, onChange }) {
  return (
    <Stack gap="xs">
      {SECTION_GROUPS.map((group, groupIdx) => (
        <Stack gap={4} key={group.groupLabel}>
          <Text className="sidebarGroupLabel">
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

          {groupIdx < SECTION_GROUPS.length - 1 && (
            <Divider className="sidebarGroupDivider" />
          )}
        </Stack>
      ))}
    </Stack>
  );
}
