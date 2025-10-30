// src/api/cv.js
import api from './client';

export async function runCrossvalRequest(payload) {
  const { data } = await api.post('/crossval', payload);
  return data;
}
