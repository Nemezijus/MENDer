// API helpers for saving and loading model artifacts using the exact backend routes:
//   POST /api/v1/save  -> binary .mend
//   POST /api/v1/load  -> JSON { artifact: ... }

function parseContentDispositionFilename(header) {
  if (!header) return null;
  const match = header.match(/filename\*?=(?:UTF-8''|")?([^\";]+)/i);
  if (match && match[1]) return decodeURIComponent(match[1].replace(/"/g, ''));
  return null;
}

/**
 * Save the last trained model.
 * @param {{ artifactUid: string, artifactMeta: object, filename?: string }} params
 * @returns {Promise<{ blob: Blob, filename: string, sha256?: string, size?: number }>}
 */
export async function saveModel({ artifactUid, artifactMeta, filename }) {
  const resp = await fetch('/api/v1/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      artifact_uid: artifactUid,
      artifact_meta: artifactMeta,
      filename,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(text || `Save failed with status ${resp.status}`);
  }

  const blob = await resp.blob();
  const cd = resp.headers.get('Content-Disposition');
  const serverName = parseContentDispositionFilename(cd);
  const sha256 = resp.headers.get('X-MENDER-SHA256') || undefined;
  const size = resp.headers.get('X-MENDER-Size') || undefined;

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

  const resp = await fetch('/api/v1/load', {
    method: 'POST',
    body: fd,
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(text || `Load failed with status ${resp.status}`);
  }

  return await resp.json();
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
  const resp = await fetch('/api/v1/models/apply', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      artifact_uid: artifactUid,
      artifact_meta: artifactMeta,
      data,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(text || `Apply model failed with status ${resp.status}`);
  }

  return await resp.json();
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
  const resp = await fetch('/api/v1/models/apply/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      artifact_uid: artifactUid,
      artifact_meta: artifactMeta,
      data,
      filename,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(text || `Export predictions failed with status ${resp.status}`);
  }

  const blob = await resp.blob();
  const cd = resp.headers.get('Content-Disposition');
  const serverName = parseContentDispositionFilename(cd);

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
  const resp = await fetch('/api/v1/decoder/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      artifact_uid: artifactUid,
      filename,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(text || `Export decoder outputs failed with status ${resp.status}`);
  }

  const blob = await resp.blob();
  const cd = resp.headers.get('Content-Disposition');
  const serverName = parseContentDispositionFilename(cd);

  return {
    blob,
    filename: serverName || filename || 'decoder_outputs.csv',
  };
}

/** Fallback download if interactive save is not supported. */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || 'model.mend';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Interactive save via the File System Access API (Chromium). */
export async function saveBlobInteractive(blob, suggestedName = 'model.mend') {
  const supportsFS = typeof window !== 'undefined' && 'showSaveFilePicker' in window;
  if (!supportsFS) {
    downloadBlob(blob, suggestedName);
    return;
  }
  try {
    const handle = await window.showSaveFilePicker({
      suggestedName,
      types: [{ description: 'MENDer model artifact', accept: { 'application/octet-stream': ['.mend'] } }],
    });
    const writable = await handle.createWritable();
    await writable.write(blob);
    await writable.close();
  } catch (e) {
    if (e && e.name === 'AbortError') {
      // user canceled -> fall back to normal download
      downloadBlob(blob, suggestedName);
      return false;
    }
    throw e;
  }
}
