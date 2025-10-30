// MENDer/frontend/src/api/train.js
import api from './client.js';

export async function runTrainRequest(payload) {
  // payload should match TrainRequest (data/split/scale/features/model/eval)
  const response = await api.post('/train', payload);
  return response.data;
}
