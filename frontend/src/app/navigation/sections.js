/**
 * Single source of truth for app sections.
 *
 * SidebarNav and App both consume this file to avoid drift between:
 * - section IDs
 * - nav labels / descriptions
 * - section titles
 * - guard requirements
 */

export const DEFAULT_SECTION_ID = 'data';

export const SECTION_GROUPS = [
  {
    groupLabel: 'DATA',
    items: [
      {
        id: 'data',
        navLabel: 'Data & files',
        title: 'Upload data & models',
        description: 'Upload and inspect training data, manage files',
        requiresTrainingData: false,
      },
    ],
  },
  {
    groupLabel: 'SETTINGS',
    items: [
      {
        id: 'settings',
        navLabel: 'Settings',
        title: 'Specify global settings',
        description: 'Scaling, features, metric',
        requiresTrainingData: false,
      },
    ],
  },
  {
    groupLabel: 'MODEL TRAINING',
    items: [
      {
        id: 'train',
        navLabel: 'Train a model',
        title: 'Model training',
        description: 'Choose algorithm and fit on current data',
        requiresTrainingData: true,
      },
      {
        id: 'train-unsupervised',
        navLabel: 'Unsupervised learning',
        title: 'Unsupervised learning',
        description: 'Clustering models (X-only)',
        requiresTrainingData: true,
      },
      {
        id: 'train-ensemble',
        navLabel: 'Train an ensemble',
        title: 'Ensemble training',
        description: 'Perform ensemble model training',
        requiresTrainingData: true,
      },
    ],
  },
  {
    groupLabel: 'TUNING',
    items: [
      {
        id: 'learning-curve',
        navLabel: 'Learning curve',
        title: 'Find optimal data split',
        description: 'Explore sample size vs performance',
        requiresTrainingData: true,
      },
      {
        id: 'validation-curve',
        navLabel: 'Validation curve',
        title: 'Find the best parameter value',
        description: 'Explore hyperparameter vs performance',
        requiresTrainingData: true,
      },
      {
        id: 'grid-search',
        navLabel: 'Grid search',
        title: 'Grid search',
        description: 'Exhaustive search over two hyperparameters',
        requiresTrainingData: true,
      },
      {
        id: 'random-search',
        navLabel: 'Randomized search',
        title: 'Randomized search',
        description: 'Random sampling over two hyperparameters',
        requiresTrainingData: true,
      },
    ],
  },
  {
    groupLabel: 'RESULTS',
    items: [
      {
        id: 'results',
        navLabel: 'Results',
        title: 'View results',
        description: 'Graphs, tables, and summaries',
        requiresTrainingData: true,
      },
    ],
  },
  {
    groupLabel: 'PREDICTIONS',
    items: [
      {
        id: 'predictions',
        navLabel: 'Predictions',
        title: 'Run predictions',
        description: 'Apply model to production data',
        requiresTrainingData: false,
      },
    ],
  },
];

export const SECTION_META_BY_ID = Object.fromEntries(
  SECTION_GROUPS.flatMap((g) => g.items).map((item) => [item.id, item]),
);
