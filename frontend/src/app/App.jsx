import { useState } from 'react';
import { Box, Text } from '@mantine/core';

import './styles/App.css';

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

const SECTION_COMPONENTS = {
  data: DataFilesPanel,
  'settings-scaling': SettingsPanel,
  'settings-metric': SettingsPanel,
  'settings-features': SettingsPanel,
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

    const content = <Panel key={activeSection} {...(meta.panelProps ?? {})} />;

    return (
      <SectionShell title={meta.title}>
        {meta.requiresTrainingData ? <DataGuard>{content}</DataGuard> : content}
      </SectionShell>
    );
  }

  const activeMeta = SECTION_META_BY_ID[activeSection] ?? null;

  return (
    <Box className="appPage">
      <Box className="appShell">
        <Box className="appHeader">
          <div className="appHeaderTitleWrap">
            <Text className="appHeaderEyebrow">Machine learning simplified</Text>
            <Text className="appHeaderTitle">MENDer</Text>
          </div>
        </Box>

        <Box className="appSurface">
          <Box className="appSidebarColumn">
            <SidebarNav active={activeSection} onChange={setActiveSection} />
          </Box>

          <Box className="appMainColumn">
            {renderMain()}
          </Box>

          <Box className="appArtifactColumn">
            <div className="appArtifactShell">
              <div className="appColumnHeader">
                <Text className="appArtifactTitle">Current Model</Text>
              </div>

              <div className="appArtifactBody">
                <ModelArtifactCard />
              </div>
            </div>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
