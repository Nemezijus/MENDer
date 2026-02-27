import '../styles/navigation.css';

import { SECTION_GROUPS } from './sections.js';

export default function SidebarNav({ active, onChange }) {
  return (
    <nav className="sidebarNavRoot" aria-label="App navigation">
      {SECTION_GROUPS.map((group, groupIdx) => (
        <div className="sidebarNavGroup" key={group.groupLabel}>
          <div className="sidebarNavGroupLabel">{group.groupLabel}</div>

          {group.items.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`sidebarNavItem ${active === item.id ? 'sidebarNavItemActive' : ''}`}
              onClick={() => onChange(item.id)}
              aria-current={active === item.id ? 'page' : undefined}
            >
              <div className="sidebarNavItemLabel">{item.navLabel}</div>
              {item.description ? (
                <div className="sidebarNavItemDescription">{item.description}</div>
              ) : null}
            </button>
          ))}

          {groupIdx < SECTION_GROUPS.length - 1 ? (
            <hr className="sidebarNavDivider" />
          ) : null}
        </div>
      ))}
    </nav>
  );
}
