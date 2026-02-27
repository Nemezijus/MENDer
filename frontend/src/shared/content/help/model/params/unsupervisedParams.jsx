import { Stack, Text, List } from '@mantine/core';
import '../../../styles/help.css';

export const UNSUPERVISED_PARAM_BLOCKS = {
  kmeans: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        K-Means parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Clusters
          </Text>{' '}
          (n_clusters) – number of clusters to form.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Initialization
          </Text>{' '}
          (init) – how initial centroids are chosen (k-means++ is a strong
          default).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Initializations
          </Text>{' '}
          (n_init) – number of restarts; more can improve stability.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Max iterations
          </Text>{' '}
          (max_iter) – iterations per restart.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Tolerance
          </Text>{' '}
          (tol) – convergence threshold.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Algorithm
          </Text>{' '}
          (algorithm) – optimization variant (lloyd/elkan).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Verbose
          </Text>{' '}
          (verbose) – logging verbosity.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Random state
          </Text>{' '}
          (random_state) – reproducible initialization.
        </List.Item>
      </List>
    </Stack>
  ),

  dbscan: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        DBSCAN parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Epsilon
          </Text>{' '}
          (eps) – neighbourhood radius (most important parameter).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Minimum samples
          </Text>{' '}
          (min_samples) – neighbours required to be a core point.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Distance metric
          </Text>{' '}
          (metric) – distance function (e.g., euclidean).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Search algorithm
          </Text>{' '}
          (algorithm) – neighbour search backend.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Leaf size
          </Text>{' '}
          (leaf_size) – tree leaf size for ball/kd trees.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Minkowski power
          </Text>{' '}
          (p) – only used for minkowski metric.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Jobs
          </Text>{' '}
          (n_jobs) – parallelism for neighbour search.
        </List.Item>
      </List>
    </Stack>
  ),

  spectral: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Spectral clustering parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Clusters
          </Text>{' '}
          (n_clusters) – number of clusters.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Affinity
          </Text>{' '}
          (affinity) – similarity graph (rbf, nearest_neighbors, ...).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Assign labels
          </Text>{' '}
          (assign_labels) – final discretization method.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Initializations
          </Text>{' '}
          (n_init) – restarts for k-means discretization.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Gamma
          </Text>{' '}
          (gamma) – used for rbf affinity; controls locality.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Neighbours
          </Text>{' '}
          (n_neighbors) – used for nearest_neighbors affinity.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Random state
          </Text>{' '}
          (random_state) – reproducible runs.
        </List.Item>
      </List>
    </Stack>
  ),

  agglo: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Agglomerative clustering parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Clusters
          </Text>{' '}
          (n_clusters) – number of clusters (ignored if distance_threshold is
          set).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Linkage
          </Text>{' '}
          (linkage) – how clusters are merged (ward, complete, average, single).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Distance metric
          </Text>{' '}
          (metric) – only for non-ward linkage.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Distance threshold
          </Text>{' '}
          (distance_threshold) – stop merging when distances exceed this.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Compute full tree
          </Text>{' '}
          (compute_full_tree) – controls dendrogram computation.
        </List.Item>
      </List>
    </Stack>
  ),

  gmm: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Gaussian Mixture parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Components
          </Text>{' '}
          (n_components) – number of mixture components.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Covariance type
          </Text>{' '}
          (covariance_type) – shape of covariance (full, diag, tied, spherical).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Regularization
          </Text>{' '}
          (reg_covar) – stabilizes covariance estimates.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Max iterations
          </Text>{' '}
          (max_iter) – EM iterations.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Initializations
          </Text>{' '}
          (n_init) – restarts for EM.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Init params
          </Text>{' '}
          (init_params) – how to initialize responsibilities / means.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Random state
          </Text>{' '}
          (random_state) – reproducible initialization.
        </List.Item>
      </List>
    </Stack>
  ),

  bgmm: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Bayesian Gaussian Mixture parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Components
          </Text>{' '}
          (n_components) – number of mixture components.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Covariance type
          </Text>{' '}
          (covariance_type) – shape of covariance (full, diag, tied, spherical).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Regularization
          </Text>{' '}
          (reg_covar) – stabilizes covariance estimates.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Max iterations
          </Text>{' '}
          (max_iter) – EM iterations.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Initializations
          </Text>{' '}
          (n_init) – restarts for EM.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Init params
          </Text>{' '}
          (init_params) – how to initialize responsibilities / means.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Random state
          </Text>{' '}
          (random_state) – reproducible initialization.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Weight prior type
          </Text>{' '}
          (weight_concentration_prior_type) – controls component sparsity.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Weight prior
          </Text>{' '}
          (weight_concentration_prior) – strength of the prior.
        </List.Item>
      </List>
    </Stack>
  ),

  meanshift: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        MeanShift parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Bandwidth
          </Text>{' '}
          (bandwidth) – kernel bandwidth. If not set, it may be estimated (can
          be slow).
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Bin seeding
          </Text>{' '}
          (bin_seeding) – speed optimization for seeding.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Cluster all
          </Text>{' '}
          (cluster_all) – whether to assign all points to clusters.
        </List.Item>
      </List>
    </Stack>
  ),

  birch: (
    <Stack className="helpStack helpStackGap4">
      <Text className="helpTitle">
        Birch parameters
      </Text>
      <List className="helpList helpListXs helpListTight">
        <List.Item>
          <Text span className="helpInlineLabel">
            Threshold
          </Text>{' '}
          (threshold) – radius threshold for subcluster merging.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Branching factor
          </Text>{' '}
          (branching_factor) – maximum subclusters per node.
        </List.Item>
        <List.Item>
          <Text span className="helpInlineLabel">
            Clusters
          </Text>{' '}
          (n_clusters) – optional final clustering step.
        </List.Item>
      </List>
    </Stack>
  ),
};
