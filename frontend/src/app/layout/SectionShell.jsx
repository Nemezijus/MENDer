import { Text } from '@mantine/core';


export default function SectionShell({ title, children, className }) {
  return (
    <section className={className ? `sectionShell ${className}` : 'sectionShell'}>
      <div className="appColumnHeader">
        <Text className="sectionShellTitle">{title}</Text>
      </div>

      <div className="sectionShellBody">
        {children}
      </div>
    </section>
  );
}
