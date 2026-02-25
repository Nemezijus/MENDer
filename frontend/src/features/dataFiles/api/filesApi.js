import { getJson, postFormData } from '../../../shared/api/http.js'; // baseURL '/api/v1'

export async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);

  // Backend returns:
  // {
  //   path: "/uploads/<sha256>.<ext>",        // canonical stored path
  //   original_name: "data_mean.mat",        // user-friendly name (for UI)
  //   saved_name: "<sha256>.<ext>"           // stable content-addressed identifier
  // }
  return postFormData('/files/upload', fd);
}

export async function listUploads() {
  // Returns: Array<{ path, original_name, saved_name }>
  // Note: original_name is a display name derived from the JSON index when available.
  return getJson('/files/list');
}

export async function getFilesConstraints() {
  // Returns:
  // {
  //   upload_dir: string,
  //   allowed_exts: string[],
  //   data_default_keys: { x_key: string, y_key: string }
  // }
  return getJson('/files/constraints');
}
