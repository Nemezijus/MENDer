import { Stack, Title } from '@mantine/core';

export default function SectionShell({ title, children, className }) {
  return (
    <section className={className ? `sectionShell ${className}` : 'sectionShell'}>
      <div className="sectionShellHeader">
        <Title order={3} className="sectionShellTitle">
          {title}
        </Title>
      </div>

      <div className="sectionShellBody">
        {children}
      </div>
    </section>
  );
}
