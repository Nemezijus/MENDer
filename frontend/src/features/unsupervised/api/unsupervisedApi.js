import api from './client.js';

/**
 * Run an unsupervised (clustering) training request.
 *
 * Backend uses the same endpoint as supervised training:
 *   POST /api/v1/train
 */
export async function runUnsupervisedTrainRequest(payload) {
  const { data } = await api.post('/train', payload);
  return data;
}
