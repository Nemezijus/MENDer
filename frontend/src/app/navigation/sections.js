import {
  IconAdjustments,
  IconCalendarStats,
  IconFileAnalytics,
  IconGauge,
  IconNotes,
  IconPresentationAnalytics,
} from '@tabler/icons-react';

/**
 * Single source of truth for app sections.
 *
 * The sidebar consumes NAV_SECTIONS to render top-level items and nested links.
 * App consumes SECTION_META_BY_ID to resolve the visible panel and its title.
 */

export const DEFAULT_SECTION_ID = 'data';

export const NAV_SECTIONS = [
  {
    id: 'data',
    navLabel: 'Data & Files',
    title: 'Upload data & files',
    description: 'Upload training and production data',
    icon: IconNotes,
    requiresTrainingData: false,
  },
  {
    navLabel: 'Settings',
    description: 'Global modelling defaults',
    icon: IconAdjustments,
    initiallyOpened: false,
    items: [
      {
        id: 'settings-scaling',
        navLabel: 'Scaling',
        title: 'Settings · Scaling',
        requiresTrainingData: false,
        panelProps: { initialTab: 'scaling' },
      },
      {
        id: 'settings-metric',
        navLabel: 'Metric',
        title: 'Settings · Metric',
        requiresTrainingData: false,
        panelProps: { initialTab: 'metric' },
      },
      {
        id: 'settings-features',
        navLabel: 'Features',
        title: 'Settings · Features',
        requiresTrainingData: false,
        panelProps: { initialTab: 'features' },
      },
    ],
  },
  {
    navLabel: 'Training',
    description: 'Single models, clustering, and ensembles',
    icon: IconPresentationAnalytics,
    initiallyOpened: true,
    items: [
      {
        id: 'train',
        navLabel: 'Train a model',
        title: 'Model training',
        requiresTrainingData: true,
      },
      {
        id: 'train-unsupervised',
        navLabel: 'Unsupervised learning',
        title: 'Unsupervised learning',
        requiresTrainingData: true,
      },
      {
        id: 'train-ensemble',
        navLabel: 'Train an ensemble',
        title: 'Ensemble training',
        requiresTrainingData: true,
      },
    ],
  },
  {
    id: 'results',
    navLabel: 'Results',
    title: 'View results',
    description: 'Graphs, tables, and summaries',
    icon: IconFileAnalytics,
    requiresTrainingData: true,
  },
  {
    id: 'predictions',
    navLabel: 'Predictions',
    title: 'Run predictions',
    description: 'Apply a saved model to production data',
    icon: IconGauge,
    requiresTrainingData: false,
  },
  {
    navLabel: 'Tuning',
    description: 'Curves and search workflows',
    icon: IconCalendarStats,
    initiallyOpened: false,
    items: [
      {
        id: 'learning-curve',
        navLabel: 'Learning curve',
        title: 'Find optimal data split',
        requiresTrainingData: true,
      },
      {
        id: 'validation-curve',
        navLabel: 'Validation curve',
        title: 'Find the best parameter value',
        requiresTrainingData: true,
      },
      {
        id: 'grid-search',
        navLabel: 'Grid search',
        title: 'Grid search',
        requiresTrainingData: true,
      },
      {
        id: 'random-search',
        navLabel: 'Randomized search',
        title: 'Randomized search',
        requiresTrainingData: true,
      },
    ],
  },
];

function flattenLeafSections(sections) {
  return sections.flatMap((section) => (Array.isArray(section.items) ? section.items : [section]));
}

export const SECTION_META_BY_ID = Object.fromEntries(
  flattenLeafSections(NAV_SECTIONS).map((item) => [item.id, item]),
);
