import { NavLink } from '@mantine/core';

function renderIcon(Icon) {
  if (!Icon) return null;
  return <Icon size={18} stroke={1.8} />;
}

export default function SidebarNavGroup({ section, active, onChange }) {
  const hasChildren = Array.isArray(section.items) && section.items.length > 0;

  if (!hasChildren) {
    return (
      <NavLink
        label={section.navLabel}
        description={section.description}
        leftSection={renderIcon(section.icon)}
        active={active === section.id}
        onClick={() => onChange(section.id)}
        className="sidebarNavItem"
        classNames={{
          body: 'sidebarNavItemBody',
          label: 'sidebarNavItemLabel',
          description: 'sidebarNavItemDescription',
          section: 'sidebarNavItemSection',
        }}
      />
    );
  }

  const hasActiveChild = section.items.some((item) => item.id === active);

  return (
    <NavLink
      label={section.navLabel}
      description={section.description}
      leftSection={renderIcon(section.icon)}
      defaultOpened={section.initiallyOpened}
      childrenOffset={8}
      active={hasActiveChild}
      className="sidebarNavItem"
      classNames={{
        body: 'sidebarNavItemBody',
        label: 'sidebarNavItemLabel',
        description: 'sidebarNavItemDescription',
        section: 'sidebarNavItemSection',
      }}
    >
      <div className="sidebarNavChildLinks">
        {section.items.map((item) => (
          <NavLink
            key={item.id}
            label={item.navLabel}
            active={active === item.id}
            onClick={() => onChange(item.id)}
            className="sidebarNavChildLink"
            classNames={{
              body: 'sidebarNavChildBody',
              label: 'sidebarNavChildLabel',
              section: 'sidebarNavChildSection',
            }}
          />
        ))}
      </div>
    </NavLink>
  );
}
