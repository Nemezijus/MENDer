// src/App.jsx
import React from 'react';
import { Container, Grid, Tabs } from '@mantine/core';
import { DataProvider } from './state/DataContext.jsx';
import DataSidebar from './components/DataSidebar.jsx';
import TrainPanel from './components/TrainPanel.jsx';
import CrossValPanel from './components/CrossValPanel.jsx';
import { useDataCtx } from './state/DataContext.jsx';

function TabsWithGuard() {
  const { dataReady } = useDataCtx();
  return (
    <Tabs defaultValue="holdout" variant="outline">
      <Tabs.List>
        <Tabs.Tab value="holdout" disabled={!dataReady}>Hold-out</Tabs.Tab>
        <Tabs.Tab value="cv" disabled={!dataReady}>Cross-Validation</Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="holdout" pt="md">
        <TrainPanel />
      </Tabs.Panel>

      <Tabs.Panel value="cv" pt="md">
        <CrossValPanel />
      </Tabs.Panel>
    </Tabs>
  );
}

export default function App() {
  return (
    <DataProvider>
      <Container size="xl" my="lg">
        <Grid gutter="lg">
          <Grid.Col span={{ base: 12, md: 4, lg: 3 }}>
            <DataSidebar />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 8, lg: 9 }}>
            <TabsWithGuard />
          </Grid.Col>
        </Grid>
      </Container>
    </DataProvider>
  );
}
