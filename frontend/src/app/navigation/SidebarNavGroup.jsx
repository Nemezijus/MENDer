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
        variant="light"
        className="sidebarNavItem"
        classNames={{ description: 'sidebarNavItemDescription' }}
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
      variant="subtle"
      active={hasActiveChild}
      className="sidebarNavItem"
      classNames={{ description: 'sidebarNavItemDescription' }}
    >
      <div className="sidebarNavChildLinks">
        {section.items.map((item) => (
          <NavLink
            key={item.id}
            label={item.navLabel}
            active={active === item.id}
            onClick={() => onChange(item.id)}
            variant="light"
            className="sidebarNavChildLink"
          />
        ))}
      </div>
    </NavLink>
  );
}
