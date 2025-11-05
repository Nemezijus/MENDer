import { Container, Grid, Tabs } from '@mantine/core';
import { DataProvider } from './state/DataContext.jsx';
import { FeatureProvider } from './state/FeatureContext.jsx';
import DataSidebar from './components/DataSidebar.jsx';
import { useDataCtx } from './state/DataContext.jsx';
import RunModelPanel from './components/RunModelPanel.jsx';

function TabsWithGuard() {
  const { dataReady } = useDataCtx();
  return (
    <Tabs defaultValue="holdout" variant="outline">
      <Tabs.List>
        <Tabs.Tab value="runModel" disabled={!dataReady}>Train a model</Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="runModel" pt="md">
        <RunModelPanel />
      </Tabs.Panel>
    </Tabs>
  );
}

export default function App() {
  return (
    <DataProvider>
      <FeatureProvider>
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
      </FeatureProvider>
    </DataProvider>
  );
}
