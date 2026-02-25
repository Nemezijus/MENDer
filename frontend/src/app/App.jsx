import { useState } from 'react';
import { Container, Box } from '@mantine/core';

import SidebarNav from './navigation/SidebarNav.jsx';

import { DEFAULT_SECTION_ID, SECTION_META_BY_ID } from './navigation/sections.js';
import DataGuard from './guards/DataGuard.jsx';
import SectionShell from './layout/SectionShell.jsx';

import DataFilesPanel from '../features/dataFiles/components/DataFilesPanel.jsx';
import SingleModelTrainingPanel from '../features/training/components/SingleModelTrainingPanel.jsx';
import LearningCurvePanel from '../features/tuning/components/LearningCurvePanel.jsx';
import ModelArtifactCard from '../features/modelArtifacts/components/ModelArtifactCard.jsx';
import ResultsPanel from '../features/results/components/ResultsPanel.jsx';
import ApplyModelPanel from '../features/inference/components/ApplyModelPanel.jsx';
import SettingsPanel from '../features/settings/components/SettingsPanel.jsx';
import ValidationCurvePanel from '../features/tuning/components/ValidationCurvePanel.jsx';
import GridSearchPanel from '../features/tuning/components/GridSearchPanel.jsx';
import RandomSearchPanel from '../features/tuning/components/RandomSearchPanel.jsx';
import EnsembleTrainingPanel from '../features/ensembles/components/EnsembleTrainingPanel.jsx';
import UnsupervisedTrainingPanel from '../features/unsupervised/components/UnsupervisedTrainingPanel.jsx';

const COLUMN_GAP = 'var(--mantine-spacing-lg)';

const SECTION_COMPONENTS = {
  data: DataFilesPanel,
  settings: SettingsPanel,
  train: SingleModelTrainingPanel,
  'train-ensemble': EnsembleTrainingPanel,
  'train-unsupervised': UnsupervisedTrainingPanel,
  'learning-curve': LearningCurvePanel,
  'validation-curve': ValidationCurvePanel,
  'grid-search': GridSearchPanel,
  'random-search': RandomSearchPanel,
  results: ResultsPanel,
  predictions: ApplyModelPanel,
};

export default function App() {
  const [activeSection, setActiveSection] = useState(DEFAULT_SECTION_ID);

  function renderMain() {
    const meta = SECTION_META_BY_ID[activeSection];
    const Panel = SECTION_COMPONENTS[activeSection];

    if (!meta || !Panel) return null;

    const content = <Panel />;

    return (
      <SectionShell title={meta.title}>
        {meta.requiresTrainingData ? <DataGuard>{content}</DataGuard> : content}
      </SectionShell>
    );
  }

  return (
    <Container fluid pt="xl" pb="md">
      <Box mx="auto" style={{ width: '80vw' }}>
        <Box
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: COLUMN_GAP,
          }}
        >
          {/* Left: sidebar */}
          <Box style={{ flex: 2, minWidth: 0 }}>
            <SidebarNav active={activeSection} onChange={setActiveSection} />
          </Box>

          {/* Center: main content */}
          <Box style={{ flex: 5, minWidth: 0 }}>
            {renderMain()}
          </Box>

          {/* Right: persistent model card */}
          <Box style={{ flex: 3, minWidth: 0 }}>
            <ModelArtifactCard />
          </Box>
        </Box>
      </Box>
    </Container>
  );
}