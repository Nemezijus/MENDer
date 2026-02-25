import { Stack, Text, List } from '@mantine/core';

export const UNSUPERVISED_PARAM_BLOCKS = {
  kmeans: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        K-Means parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Clusters
          </Text>{' '}
          (n_clusters) – number of clusters to form.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Initialization
          </Text>{' '}
          (init) – how initial centroids are chosen (k-means++ is a strong
          default).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Initializations
          </Text>{' '}
          (n_init) – number of restarts; more can improve stability.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Max iterations
          </Text>{' '}
          (max_iter) – iterations per restart.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Tolerance
          </Text>{' '}
          (tol) – convergence threshold.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Algorithm
          </Text>{' '}
          (algorithm) – optimization variant (lloyd/elkan).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Verbose
          </Text>{' '}
          (verbose) – logging verbosity.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Random state
          </Text>{' '}
          (random_state) – reproducible initialization.
        </List.Item>
      </List>
    </Stack>
  ),

  dbscan: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        DBSCAN parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Epsilon
          </Text>{' '}
          (eps) – neighbourhood radius (most important parameter).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Minimum samples
          </Text>{' '}
          (min_samples) – neighbours required to be a core point.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Distance metric
          </Text>{' '}
          (metric) – distance function (e.g., euclidean).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Search algorithm
          </Text>{' '}
          (algorithm) – neighbour search backend.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Leaf size
          </Text>{' '}
          (leaf_size) – tree leaf size for ball/kd trees.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Minkowski power
          </Text>{' '}
          (p) – only used for minkowski metric.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Jobs
          </Text>{' '}
          (n_jobs) – parallelism for neighbour search.
        </List.Item>
      </List>
    </Stack>
  ),

  spectral: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Spectral clustering parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Clusters
          </Text>{' '}
          (n_clusters) – number of clusters.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Affinity
          </Text>{' '}
          (affinity) – similarity graph (rbf, nearest_neighbors, ...).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Assign labels
          </Text>{' '}
          (assign_labels) – final discretization method.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Initializations
          </Text>{' '}
          (n_init) – restarts for k-means discretization.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Gamma
          </Text>{' '}
          (gamma) – used for rbf affinity; controls locality.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Neighbours
          </Text>{' '}
          (n_neighbors) – used for nearest_neighbors affinity.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Random state
          </Text>{' '}
          (random_state) – reproducible runs.
        </List.Item>
      </List>
    </Stack>
  ),

  agglo: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Agglomerative clustering parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Clusters
          </Text>{' '}
          (n_clusters) – number of clusters (ignored if distance_threshold is
          set).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Linkage
          </Text>{' '}
          (linkage) – how clusters are merged (ward, complete, average, single).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Distance metric
          </Text>{' '}
          (metric) – only for non-ward linkage.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Distance threshold
          </Text>{' '}
          (distance_threshold) – stop merging when distances exceed this.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Compute full tree
          </Text>{' '}
          (compute_full_tree) – controls dendrogram computation.
        </List.Item>
      </List>
    </Stack>
  ),

  gmm: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Gaussian Mixture parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Components
          </Text>{' '}
          (n_components) – number of mixture components.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Covariance type
          </Text>{' '}
          (covariance_type) – shape of covariance (full, diag, tied, spherical).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Regularization
          </Text>{' '}
          (reg_covar) – stabilizes covariance estimates.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Max iterations
          </Text>{' '}
          (max_iter) – EM iterations.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Initializations
          </Text>{' '}
          (n_init) – restarts for EM.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Init params
          </Text>{' '}
          (init_params) – how to initialize responsibilities / means.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Random state
          </Text>{' '}
          (random_state) – reproducible initialization.
        </List.Item>
      </List>
    </Stack>
  ),

  bgmm: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Bayesian Gaussian Mixture parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Components
          </Text>{' '}
          (n_components) – number of mixture components.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Covariance type
          </Text>{' '}
          (covariance_type) – shape of covariance (full, diag, tied, spherical).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Regularization
          </Text>{' '}
          (reg_covar) – stabilizes covariance estimates.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Max iterations
          </Text>{' '}
          (max_iter) – EM iterations.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Initializations
          </Text>{' '}
          (n_init) – restarts for EM.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Init params
          </Text>{' '}
          (init_params) – how to initialize responsibilities / means.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Random state
          </Text>{' '}
          (random_state) – reproducible initialization.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Weight prior type
          </Text>{' '}
          (weight_concentration_prior_type) – controls component sparsity.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Weight prior
          </Text>{' '}
          (weight_concentration_prior) – strength of the prior.
        </List.Item>
      </List>
    </Stack>
  ),

  meanshift: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        MeanShift parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Bandwidth
          </Text>{' '}
          (bandwidth) – kernel bandwidth. If not set, it may be estimated (can
          be slow).
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Bin seeding
          </Text>{' '}
          (bin_seeding) – speed optimization for seeding.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Cluster all
          </Text>{' '}
          (cluster_all) – whether to assign all points to clusters.
        </List.Item>
      </List>
    </Stack>
  ),

  birch: (
    <Stack gap={4}>
      <Text fw={500} size="sm">
        Birch parameters
      </Text>
      <List spacing={4} size="xs">
        <List.Item>
          <Text span fw={600}>
            Threshold
          </Text>{' '}
          (threshold) – radius threshold for subcluster merging.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Branching factor
          </Text>{' '}
          (branching_factor) – maximum subclusters per node.
        </List.Item>
        <List.Item>
          <Text span fw={600}>
            Clusters
          </Text>{' '}
          (n_clusters) – optional final clustering step.
        </List.Item>
      </List>
    </Stack>
  ),
};
