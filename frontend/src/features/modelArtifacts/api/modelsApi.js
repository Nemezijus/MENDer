// API helpers for saving/loading model artifacts and applying them to production data.

import { postBlob, postFormData, postJson } from '../../../shared/api/http.js';
import { getFilenameFromContentDisposition } from '../../../shared/utils/download.js';

/**
 * Save the last trained model.
 * @param {{ artifactUid: string, artifactMeta: object, filename?: string }} params
 * @returns {Promise<{ blob: Blob, filename: string, sha256?: string, size?: number }>}
 */
export async function saveModel({ artifactUid, artifactMeta, filename }) {
  const { blob, headers } = await postBlob('/api/v1/models/save', {
    artifact_uid: artifactUid,
    artifact_meta: artifactMeta,
    filename,
  });

  const cd = headers['content-disposition'] || headers['Content-Disposition'];
  const serverName = getFilenameFromContentDisposition(cd);
  const sha256 = headers['x-mender-sha256'] || headers['X-MENDER-SHA256'] || undefined;
  const size = headers['x-mender-size'] || headers['X-MENDER-Size'] || undefined;

  return {
    blob,
    filename: serverName || filename || 'model.mend',
    sha256,
    size: size ? Number(size) : undefined,
  };
}

/**
 * Load a saved model artifact file.
 * @param {File} file
 * @returns {Promise<{ artifact: object }>} with artifact meta
 */
export async function loadModel(file) {
  const fd = new FormData();
  fd.append('file', file);

  return await postFormData('/api/v1/models/load', fd);
}

/**
 * Apply an existing model to a new dataset (production data).
 * @param {{ artifactUid: string, artifactMeta: object, data: object }} params
 *   - artifactUid: current model artifact UID
 *   - artifactMeta: current artifact meta (ModelArtifactMeta as plain object)
 *   - data: DataInspectRequest-like object { x_path?, y_path?, npz_path?, x_key, y_key }
 * @returns {Promise<object>} ApplyModelResponse
 */
export async function applyModelToData({ artifactUid, artifactMeta, data }) {
  return await postJson('/api/v1/models/apply', {
    artifact_uid: artifactUid,
    artifact_meta: artifactMeta,
    data,
  });
}

/**
 * Export predictions as CSV for an applied model.
 * This uses a streaming endpoint that returns text/csv.
 *
 * @param {{ artifactUid: string, artifactMeta: object, data: object, filename?: string }} params
 *   - artifactUid: current model artifact UID
 *   - artifactMeta: current artifact meta (ModelArtifactMeta as plain object)
 *   - data: DataInspectRequest-like object { x_path?, y_path?, npz_path?, x_key, y_key }
 *   - filename: optional client-suggested filename (without or with .csv)
 * @returns {Promise<{ blob: Blob, filename: string }>}
 */
export async function exportPredictions({ artifactUid, artifactMeta, data, filename }) {
  const { blob, headers } = await postBlob('/api/v1/models/apply/export', {
    artifact_uid: artifactUid,
    artifact_meta: artifactMeta,
    data,
    filename,
  });

  const cd = headers['content-disposition'] || headers['Content-Disposition'];
  const serverName = getFilenameFromContentDisposition(cd);

  return {
    blob,
    filename: serverName || filename || 'predictions.csv',
  };
}

/**
 * Export cached decoder/evaluation outputs as CSV (from a training run).
 *
 * @param {{ artifactUid: string, filename?: string }} params
 * @returns {Promise<{ blob: Blob, filename: string }>} 
 */
export async function exportDecoderOutputs({ artifactUid, filename }) {
  const { blob, headers } = await postBlob('/api/v1/decoder/export', {
    artifact_uid: artifactUid,
    filename,
  });

  const cd = headers['content-disposition'] || headers['Content-Disposition'];
  const serverName = getFilenameFromContentDisposition(cd);

  return {
    blob,
    filename: serverName || filename || 'decoder_outputs.csv',
  };
}

// NOTE: download helpers live in src/shared/utils/download.js
