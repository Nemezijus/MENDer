import api from '../../../shared/api/client.js';

export async function runTrainRequest(payload) {
  const { data } = await api.post('/train', payload);
  return data;
}
