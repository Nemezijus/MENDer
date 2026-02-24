import { getJson } from '../../../shared/api/http.js';

export async function fetchProgress(progressId) {
  return await getJson(`/progress/${encodeURIComponent(progressId)}`);
}
