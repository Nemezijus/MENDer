/**
 * Re-export shared coercion helpers.
 *
 * Keeping this file avoids churn across existing ensemble imports while ensuring
 * there is a single source of truth.
 */

export {
  numOrUndef,
  intOrUndef,
  boolOrUndef,
} from '../../../shared/utils/coerce.js';
