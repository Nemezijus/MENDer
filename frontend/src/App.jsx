import { Container, Grid, Tabs, Stack } from '@mantine/core';
import { DataProvider, useDataCtx } from './state/DataContext.jsx';
import { FeatureProvider } from './state/FeatureContext.jsx';
import { RunModelResultsProvider } from './state/RunModelResultsContext.jsx';
import { LearningCurveResultsProvider } from './state/LearningCurveResultsContext.jsx';
import { ModelArtifactProvider } from "./state/ModelArtifactContext";
import { SchemaDefaultsProvider } from './state/SchemaDefaultsContext.jsx';
import { ProductionDataProvider } from './state/ProductionDataContext.jsx';
import { ProductionResultsProvider } from './state/ProductionResultsContext.jsx';

import DataSidebar from './components/DataSidebar.jsx';
import RunModelPanel from './components/RunModelPanel.jsx';
import LearningCurvePanel from './components/LearningCurvePanel.jsx';
import ModelCard from './components/ModelCard.jsx';
import ModelTrainingResultsPanel from './components/ModelTrainingResultsPanel.jsx';
import LearningCurveResultsPanel from './components/LearningCurveResultsPanel.jsx';
import ApplyModelCard from './components/ApplyModelCard.jsx';

function TabsWithGuard() {
  const { dataReady } = useDataCtx();

  return (
    <Tabs defaultValue="runModel" variant="pills">
      <Tabs.List grow>
        <Tabs.Tab value="runModel" disabled={!dataReady}>Train a model</Tabs.Tab>
        <Tabs.Tab value="learningCurve" disabled={!dataReady}>Learning Curve</Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="runModel" pt="md">
        <RunModelPanel />
      </Tabs.Panel>
      <Tabs.Panel value="learningCurve" pt="md">
        <LearningCurvePanel />
      </Tabs.Panel>
    </Tabs>
  );
}

export default function App() {
  return (
    <SchemaDefaultsProvider>
      <DataProvider>
        <FeatureProvider>
          <RunModelResultsProvider>
            <LearningCurveResultsProvider>
              <ModelArtifactProvider>
                <ProductionDataProvider>
                  <ProductionResultsProvider>
                    <Container size="xl" my="lg">
                      <Grid gutter="lg">
                        {/* Left: data sidebar */}
                        <Grid.Col span={{ base: 12, md: 4, lg: 3 }}>
                          <DataSidebar />
                        </Grid.Col>

                        {/* Middle: main tabs */}
                        <Grid.Col span={{ base: 12, md: 8, lg: 5 }}>
                          <TabsWithGuard />
                        </Grid.Col>

                        {/* Right: model + results + prediction */}
                        <Grid.Col span={{ base: 12, md: 12, lg: 4 }}>
                          <Stack gap="lg">
                            <ModelCard />
                            <ApplyModelCard />
                            <ModelTrainingResultsPanel />
                            <LearningCurveResultsPanel />
                          </Stack>
                        </Grid.Col>
                      </Grid>
                    </Container>
                  </ProductionResultsProvider>
                </ProductionDataProvider>
              </ModelArtifactProvider>
            </LearningCurveResultsProvider>
          </RunModelResultsProvider>
        </FeatureProvider>
      </DataProvider>
    </SchemaDefaultsProvider>
  );
}