import { postJson } from '../../../shared/api/http.js';

// Learning curve
export async function requestLearningCurve(payload) {
  return await postJson('/tuning/learning-curve', payload);
}

// Validation curve
export async function requestValidationCurve(payload) {
  return await postJson('/tuning/validation-curve', payload);
}

// Grid search (GridSearchCV)
export async function requestGridSearch(payload) {
  return await postJson('/tuning/grid-search', payload);
}

// Randomized search (RandomizedSearchCV)
export async function requestRandomSearch(payload) {
  return await postJson('/tuning/random-search', payload);
}
