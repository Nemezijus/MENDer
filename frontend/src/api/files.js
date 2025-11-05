import api from './client'; // baseURL '/api/v1'

export async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  const { data } = await api.post('/files/upload', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data; // { path: "/data/uploads/uuid.ext", original_name: "..." }
}

export async function listUploads() {
  const { data } = await api.get('/files/list');
  return data;
}
