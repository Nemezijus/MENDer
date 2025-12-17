import api from './client'; // baseURL '/api/v1'

export async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  const { data } = await api.post('/files/upload', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  // Backend returns:
  // {
  //   path: "/uploads/<sha256>.<ext>",        // canonical stored path
  //   original_name: "data_mean.mat",        // user-friendly name (for UI)
  //   saved_name: "<sha256>.<ext>"           // stable content-addressed identifier
  // }
  return data;
}

export async function listUploads() {
  const { data } = await api.get('/files/list');
  // Returns: Array<{ path, original_name, saved_name }>
  // Note: original_name is a display name derived from the JSON index when available.
  return data;
}
