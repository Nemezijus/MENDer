import { useState } from 'react';
import { Container, Box, Stack, Alert, Text, Title } from '@mantine/core';

import SidebarNav from './navigation/SidebarNav.jsx';


import DataFilesPanel from '../features/dataFiles/components/DataFilesPanel.jsx';
import SingleModelTrainingPanel from '../features/training/components/SingleModelTrainingPanel.jsx';
import LearningCurvePanel from '../features/tuning/components/LearningCurvePanel.jsx';
import ModelArtifactCard from '../features/modelArtifacts/components/ModelArtifactCard.jsx';
import ResultsPanel from '../features/results/components/ResultsPanel.jsx';
import ApplyModelPanel from '../features/inference/components/ApplyModelPanel.jsx';
import SettingsPanel from '../features/settings/components/SettingsPanel.jsx';

import { useDataStore } from '../features/dataFiles/state/useDataStore.js';
import ValidationCurvePanel from '../features/tuning/components/ValidationCurvePanel.jsx';
import GridSearchPanel from '../features/tuning/components/GridSearchPanel.jsx';
import RandomSearchPanel from '../features/tuning/components/RandomSearchPanel.jsx';
import EnsembleTrainingPanel from '../features/ensembles/components/EnsembleTrainingPanel.jsx';
import UnsupervisedTrainingPanel from '../features/unsupervised/components/UnsupervisedTrainingPanel.jsx';

const COLUMN_GAP = 'var(--mantine-spacing-lg)';

function DataGuard({ children }) {
  const dataReady = useDataStore(
    (s) => !!s.inspectReport && s.inspectReport.n_samples > 0,
  );

  if (!dataReady) {
    return (
      <Alert color="yellow" variant="light">
        <Text fw={500}>No inspected training data yet.</Text>
        <Text size="sm">
          Please upload and inspect your training data in the{' '}
          <strong>Data &amp; files</strong> section before using this panel.
        </Text>
      </Alert>
    );
  }

  return children;
}

export default function App() {
  const [activeSection, setActiveSection] = useState('data');

  function renderMain() {
    switch (activeSection) {
      case 'data':
        return (
          <Stack gap="md">
            <Title order={3} align="center">
              Upload data & models
            </Title>
            <DataFilesPanel />
          </Stack>
        );

      case 'settings':
        return (
          <Stack gap="md">
            <Title order={3} align="center">Specify global settings</Title>
            <SettingsPanel />
          </Stack>
        );

      case 'train':
        return (
          <Stack gap="md">
            <Title order={3} align="center">
              Model training
            </Title>
            <DataGuard>
              <SingleModelTrainingPanel />
            </DataGuard>
          </Stack>
        );
      case 'train-ensemble':
        return (
          <Stack gap="md">
            <Title order={3} align="center">
              Ensemble training
            </Title>
            <DataGuard>
              <EnsembleTrainingPanel />
            </DataGuard>
          </Stack>
        );

      case 'train-unsupervised':
        return (
          <Stack gap="md">
            <Title order={3} align="center">
              Unsupervised learning
            </Title>
            <DataGuard>
              <UnsupervisedTrainingPanel />
            </DataGuard>
          </Stack>
        );

      case 'learning-curve':
        return (
          <Stack gap="md">
            <Title order={3} align="center">Find optimal data split</Title>
            <DataGuard>
              <LearningCurvePanel />
            </DataGuard>
          </Stack>
        );

      case 'validation-curve':
        return (
          <Stack gap="md">
            <Title order={3} align="center">Find the best parameter value</Title>
            <DataGuard>
              <ValidationCurvePanel />
            </DataGuard>
          </Stack>
        );

      case 'grid-search':
        return (
          <Stack gap="md">
            <Title order={3} align="center">Grid search</Title>
            <DataGuard>
              <GridSearchPanel />
            </DataGuard>
          </Stack>
        );

      case 'random-search':
        return (
          <Stack gap="md">
            <Title order={3} align="center">Randomized search</Title>
            <DataGuard>
              <RandomSearchPanel />
            </DataGuard>
          </Stack>
        );

      case 'results':
        return (
          <Stack gap="md">
            <Title order={3} align="center">View results</Title>
            <DataGuard>
              <ResultsPanel />
            </DataGuard>
          </Stack>
        );

      case 'predictions':
        return (
          <Stack gap="md">
            <Title order={3} align="center">Run predictions</Title>
            <ApplyModelPanel />
          </Stack>
        );

      default:
        return null;
    }
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