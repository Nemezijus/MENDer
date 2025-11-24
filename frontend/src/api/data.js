import api from './client';

// existing inspect endpoint (reuse)
export async function inspectData(payload) {
  const { data } = await api.post('/data/inspect', payload);
  return data;
}

// new upload endpoint (multipart form)
export async function uploadData(formData) {
  const { data } = await api.post('/data/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}
