import '../styles/guards.css';

import { useDataStore } from '../../features/dataFiles/state/useDataStore.js';

export default function DataGuard({ children }) {
  const dataReady = useDataStore(
    (s) => !!s.inspectReport && s.inspectReport.n_samples > 0,
  );

  if (!dataReady) {
    return (
      <div className="dataGuardAlert" role="alert">
        <div className="dataGuardTitle">No inspected training data yet.</div>
        <div className="dataGuardText">
          Please upload and inspect your training data in the{' '}
          <strong>Data &amp; files</strong> section before using this panel.
        </div>
      </div>
    );
  }

  return children;
}
