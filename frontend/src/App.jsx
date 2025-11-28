import { useState } from 'react';
import { Container, Box, Stack, Alert, Text, Title } from '@mantine/core';

import SidebarNav from './components/SidebarNav.jsx';

import DataSidebar from './components/DataSidebar.jsx';
import RunModelPanel from './components/RunModelPanel.jsx';
import LearningCurvePanel from './components/LearningCurvePanel.jsx';
import ModelCard from './components/ModelCard.jsx';
import ModelTrainingResultsPanel from './components/ModelTrainingResultsPanel.jsx';
import LearningCurveResultsPanel from './components/LearningCurveResultsPanel.jsx';
import ApplyModelCard from './components/ApplyModelCard.jsx';
import SettingsPanel from './components/SettingsPanel.jsx';
import ModelLoadCard from './components/ModelLoadCard.jsx';
import ProductionDataUploadCard from './components/ProductionDataUploadCard.jsx';

import { useDataStore } from './state/useDataStore.js';

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
          Please upload and inspect your training data in the <strong>Data &amp; files</strong> section
          before using this panel.
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
            <Title order={3}>Upload your data</Title>

            {/* Middle area: left (training data + summary), right (load model + production data) */}
            <Box
              style={{
                display: 'flex',
                gap: COLUMN_GAP,
                alignItems: 'flex-start', // don't stretch right column to full left height
              }}
            >
              {/* Left side: Training Data + Summary (inside DataSidebar) */}
              <Box style={{ flex: 11, minWidth: 0 }}>
                <DataSidebar />
              </Box>

              {/* Right side: Load a saved model, then Production data */}
              <Box style={{ flex: 9, minWidth: 0 }}>
                <Stack gap="md">
                  <ModelLoadCard />
                  <ProductionDataUploadCard />
                </Stack>
              </Box>
            </Box>
          </Stack>
        );

      case 'settings':
        return (
          <Stack gap="md">
            <Title order={3}>Specify global settings</Title>
            <SettingsPanel />
          </Stack>
        );

      case 'train':
        return (
          <Stack gap="md">
            <Title order={3}>Train a model</Title>
            <DataGuard>
              <RunModelPanel />
            </DataGuard>
          </Stack>
        );

      case 'tuning':
        return (
          <Stack gap="md">
            <Title order={3}>Tune and diagnose</Title>
            <DataGuard>
              <LearningCurvePanel />
            </DataGuard>
          </Stack>
        );

      case 'results':
        return (
          <Stack gap="md">
            <Title order={3}>View results</Title>
            <DataGuard>
              <Stack gap="lg">
                <ModelTrainingResultsPanel />
                <LearningCurveResultsPanel />
              </Stack>
            </DataGuard>
          </Stack>
        );

      case 'predictions':
        return (
          <Stack gap="md">
            <Title order={3}>Run predictions</Title>
            <ApplyModelCard />
          </Stack>
        );

      default:
        return null;
    }
  }

  return (
    <Container fluid py="md">
      {/* W = 80% of screen width */}
      <Box mx="auto" style={{ width: '80vw' }}>
        <Box
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: COLUMN_GAP, // same spacing between sidebar, main, and model card
          }}
        >
          {/* Sidebar / navbar (≈20%) */}
          <Box style={{ flex: 2, minWidth: 0 }}>
            <SidebarNav active={activeSection} onChange={setActiveSection} />
          </Box>

          {/* Main content (≈50%) */}
          <Box style={{ flex: 5, minWidth: 0 }}>
            {renderMain()}
          </Box>

          {/* Persistent ModelCard (≈30%) */}
          <Box style={{ flex: 3, minWidth: 0 }}>
            <ModelCard />
          </Box>
        </Box>
      </Box>
    </Container>
  );
}
