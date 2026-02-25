import { postFormData, postJson } from '../../../shared/api/http.js';

// existing inspect endpoint (reuse)
export async function inspectData(payload) {
  return postJson('/data/inspect', payload);
}

export async function inspectProductionData(payload) {
  return postJson('/data/inspect_production', payload);
}

// new upload endpoint (multipart form)
export async function uploadData(formData) {
  return postFormData('/data/upload', formData);
}
