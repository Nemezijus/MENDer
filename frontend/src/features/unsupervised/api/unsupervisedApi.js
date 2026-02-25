import { postJson } from '../../../shared/api/http.js';

/**
 * Run an unsupervised (clustering) training request.
 *
 * Backend uses the shared train endpoint:
 *   POST /api/v1/train
 */
export async function runUnsupervisedTrainRequest(payload) {
  return await postJson('/train', payload);
}
