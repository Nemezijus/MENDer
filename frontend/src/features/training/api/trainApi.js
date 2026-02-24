import { postJson } from '../../../shared/api/http.js';

export async function runTrainRequest(payload) {
  return await postJson('/train', payload);
}
