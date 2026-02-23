import api from './client';

// Learning curve
export async function requestLearningCurve(payload) {
  // POST /api/v1/tuning/learning-curve
  const { data } = await api.post('/tuning/learning-curve', payload);
  return data;
}

// Validation curve
export async function requestValidationCurve(payload) {
  // POST /api/v1/tuning/validation-curve
  const { data } = await api.post('/tuning/validation-curve', payload);
  return data;
}

// Grid search (GridSearchCV)
export async function requestGridSearch(payload) {
  // POST /api/v1/tuning/grid-search
  const { data } = await api.post('/tuning/grid-search', payload);
  return data;
}

// Randomized search (RandomizedSearchCV)
export async function requestRandomSearch(payload) {
  // POST /api/v1/tuning/random-search
  const { data } = await api.post('/tuning/random-search', payload);
  return data;
}
