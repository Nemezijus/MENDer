import { Code, ScrollArea, Text, Group } from '@mantine/core';

import '../styles/navigation.css';

import SidebarNavGroup from './SidebarNavGroup.jsx';
import { NAV_SECTIONS, SECTION_META_BY_ID } from './sections.js';

export default function SidebarNav({ active, onChange }) {
  const activeMeta = SECTION_META_BY_ID[active] ?? null;

  return (
    <nav className="sidebarNav">
      <div className="sidebarNavHeader">
        <Group justify="space-between" align="center" className="sidebarNavHeaderInner">
          <div className="sidebarNavBrandWrap">
            <Text className="sidebarNavBrand">
              NAVIGATION
            </Text>
          </div>
        </Group>
      </div>

      <ScrollArea className="sidebarNavLinks">
        <div className="sidebarNavLinksInner">
          {NAV_SECTIONS.map((section) => (
            <SidebarNavGroup
              key={section.id ?? section.navLabel}
              section={section}
              active={active}
              onChange={onChange}
            />
          ))}
        </div>
      </ScrollArea>

      <div className="sidebarNavFooter">
        <Text className="sidebarNavFooterText">
          Major version: 2.0
        </Text>
      </div>
    </nav>
  );
}
