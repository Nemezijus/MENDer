// frontend/src/features/ensembles/api/ensemblesApi.js
import { postJson } from '../../../shared/api/http.js';

export async function runEnsembleTrainRequest(payload) {
  // POST /api/v1/ensembles/train
  return await postJson('/ensembles/train', payload);
}
