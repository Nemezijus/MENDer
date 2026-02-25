import { useMemo } from 'react';
import { Divider, SimpleGrid, Stack, Text } from '@mantine/core';

import { histogram, lorenzFromSizes } from '../../utils/stats.js';
import { parseClusterSizes } from '../../utils/clusterSummary.js';

import EmbeddingScatterSection from './sections/EmbeddingScatterSection.jsx';
import ClusterSizeDistributionSection from './sections/ClusterSizeDistributionSection.jsx';
import ClusterSizeInequalitySection from './sections/ClusterSizeInequalitySection.jsx';
import DistanceToCenterHistogramSection from './sections/DistanceToCenterHistogramSection.jsx';
import SilhouetteSection from './sections/SilhouetteSection.jsx';
import FeatureProfilesSection from './sections/FeatureProfilesSection.jsx';
import SeparationMatrixSection from './sections/SeparationMatrixSection.jsx';

import ElbowCurveSection from './sections/ElbowCurveSection.jsx';
import CompactnessSeparationSection from './sections/CompactnessSeparationSection.jsx';
import KDistanceSection from './sections/KDistanceSection.jsx';
import CoreBorderNoiseCountsSection from './sections/CoreBorderNoiseCountsSection.jsx';
import SpectralEigenvaluesSection from './sections/SpectralEigenvaluesSection.jsx';
import DendrogramSection from './sections/DendrogramSection.jsx';

function normalizeClusterSizes(clusterSummary) {
  const pairs = parseClusterSizes(clusterSummary?.cluster_sizes);
  return pairs
    .map((r) => ({ cluster_id: Number(r.cluster_id), size: Number(r.size) }))
    .filter((r) => Number.isFinite(r.cluster_id) && Number.isFinite(r.size))
    .sort((a, b) => a.cluster_id - b.cluster_id);
}

export default function UnsupervisedTrainingResults({ trainResult }) {
  const clusterSummary = trainResult?.cluster_summary || {};
  const diag = trainResult?.diagnostics || {};
  const plotData = diag?.plot_data || {};

  const sizes = useMemo(() => normalizeClusterSizes(clusterSummary), [clusterSummary]);
  const lorenz = useMemo(() => lorenzFromSizes(sizes), [sizes]);

  const embedding = diag?.embedding_2d || null;

  const distanceToCenter = plotData?.distance_to_center || null;
  const distanceHist = useMemo(() => histogram(distanceToCenter?.values), [distanceToCenter]);

  const centroids = plotData?.centroids || null;
  const sepMatrix = plotData?.separation_matrix || null;
  const silhouette = plotData?.silhouette || null;

  // Model-specific plot payloads
  const compactSep = plotData?.compactness_separation || null;
  const elbow = plotData?.elbow_curve || null;
  const kdist = plotData?.k_distance || null;
  const coreCounts = plotData?.core_border_noise_counts || null;
  const spectral = plotData?.spectral_eigenvalues || null;
  const gmmEllipses = plotData?.gmm_ellipses || null;
  const dendrogram = plotData?.dendrogram || null;

  const clusterLabel = (cid) => {
    const c = Number(cid);
    if (!Number.isFinite(c)) return String(cid);
    if (c === -1) return 'Noise';
    return `C${c + 1}`;
  };

  const hasAnyGlobal = Boolean(
    embedding ||
      sizes.length ||
      (distanceHist && Array.isArray(distanceHist?.x) && Array.isArray(distanceHist?.y)) ||
      centroids ||
      sepMatrix ||
      silhouette,
  );

  const hasAnyModelSpecific = Boolean(elbow || compactSep || kdist || coreCounts || spectral || gmmEllipses || dendrogram);

  if (!hasAnyGlobal && !hasAnyModelSpecific) {
    return (
      <Text size="sm" c="dimmed">
        No visualization payload was returned for this run.
      </Text>
    );
  }

  return (
    <Stack gap="md">
      {hasAnyGlobal ? (
        <>
          <Text fw={700} size="xl" ta="center">
            Unsupervised model results
          </Text>
          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md" style={{ alignItems: 'stretch' }}>
            <EmbeddingScatterSection
              embedding={embedding}
              gmmEllipses={gmmEllipses}
              clusterLabel={clusterLabel}
            />
            <ClusterSizeDistributionSection sizes={sizes} />
            <ClusterSizeInequalitySection lorenz={lorenz} />
            <DistanceToCenterHistogramSection distanceHist={distanceHist} />
            <SilhouetteSection silhouette={silhouette} clusterLabel={clusterLabel} />
            <FeatureProfilesSection centroids={centroids} clusterLabel={clusterLabel} />
            <SeparationMatrixSection sepMatrix={sepMatrix} clusterLabel={clusterLabel} />
          </SimpleGrid>
        </>
      ) : null}

      {hasAnyModelSpecific ? (
        <>
          <Divider my="sm" />
          <Text fw={700} size="xl" ta="center">
            Model-specific plots
          </Text>
          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md" style={{ alignItems: 'stretch' }}>
            <ElbowCurveSection elbow={elbow} />
            <CompactnessSeparationSection compactSep={compactSep} />
            <KDistanceSection kdist={kdist} />
            <CoreBorderNoiseCountsSection coreCounts={coreCounts} />
            <SpectralEigenvaluesSection spectral={spectral} />
          </SimpleGrid>

          <DendrogramSection dendrogram={dendrogram} clusterLabel={clusterLabel} />

          {gmmEllipses && gmmEllipses?.components?.length ? (
            <Text size="xs" c="dimmed">
              Note: covariance ellipses are overlaid on the embedding scatter when available.
            </Text>
          ) : null}
        </>
      ) : null}
    </Stack>
  );
}
