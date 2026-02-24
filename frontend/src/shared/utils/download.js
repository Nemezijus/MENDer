/**
 * Shared download utilities.
 *
 * Several panels need to download blobs (CSV exports, saved model artifacts).
 * Keep the DOM/File System Access API logic here so features don't depend on
 * each other for basic utilities.
 */

/**
 * Best-effort filename extraction from Content-Disposition.
 *
 * @param {string | null | undefined} header
 * @returns {string | null}
 */
export function getFilenameFromContentDisposition(header) {
  if (!header) return null;
  const match = header.match(/filename\*?=(?:UTF-8''|")?([^";]+)/i);
  if (match && match[1]) {
    try {
      return decodeURIComponent(match[1].replace(/"/g, ''));
    } catch {
      return match[1].replace(/"/g, '');
    }
  }
  return null;
}

/**
 * Trigger a browser download for a Blob.
 *
 * @param {Blob} blob
 * @param {string} filename
 */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || 'download';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/**
 * Interactive save via the File System Access API (Chromium).
 * Falls back to a normal download if unavailable.
 *
 * @param {Blob} blob
 * @param {string} suggestedName
 * @returns {Promise<boolean | void>} false when user cancels
 */
export async function saveBlobInteractive(blob, suggestedName = 'download') {
  const supportsFS = typeof window !== 'undefined' && 'showSaveFilePicker' in window;
  if (!supportsFS) {
    downloadBlob(blob, suggestedName);
    return;
  }

  try {
    const handle = await window.showSaveFilePicker({
      suggestedName,
    });
    const writable = await handle.createWritable();
    await writable.write(blob);
    await writable.close();
    return true;
  } catch (e) {
    if (e && e.name === 'AbortError') {
      return false;
    }
    throw e;
  }
}
