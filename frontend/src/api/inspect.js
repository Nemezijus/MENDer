// MENDer/frontend/src/api/inspect.js
import api from './client.js';

export async function runInspectRequest(payload) {
  // payload: { x_path, y_path } or { npz_path, x_key, y_key }
  const response = await api.post('/data/inspect', payload);
  return response.data;
}
