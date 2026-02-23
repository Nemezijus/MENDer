// frontend/src/api/ensembles.js
import api from '../../../shared/api/client.js';

export async function runEnsembleTrainRequest(payload) {
  // POST /api/v1/ensembles/train
  const { data } = await api.post('/ensembles/train', payload);
  return data;
}
