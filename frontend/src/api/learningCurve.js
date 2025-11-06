import api from './client';

export async function requestLearningCurve(payload) {
  const { data } = await api.post('/learning-curve', payload);
  return data;
}
