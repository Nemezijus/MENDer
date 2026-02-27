import '../styles/layout.css';

export default function SectionShell({ title, children }) {
  return (
    <div className="sectionShell">
      <h3 className="sectionShellTitle">{title}</h3>
      {children}
    </div>
  );
}
