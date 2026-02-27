import { pickColumns as pickColumnsBase } from '../../../shared/utils/previewTable.js';

/**
 * Selects a stable column order for the preview table.
 *
 * - Includes preferred keys first if present.
 * - Drops columns that are entirely empty (except 'index').
 */
export function pickColumns(rows) {
  const preferred = [
    'index',
    'cluster_id',
    'is_noise',
    'is_core',
    'distance_to_center',
    'max_membership_prob',
    'log_likelihood',
  ];

  return pickColumnsBase(rows, {
    preferred,
    alwaysInclude: ['index'],
  });
}
