const SHORT_HASH_LEN = 7;

/**
 * @param {File | null | undefined} file
 */
export function displayLocalFilePath(file) {
  if (!file) return '';
  const rel =
    file.webkitRelativePath && file.webkitRelativePath.length > 0
      ? file.webkitRelativePath
      : file.name;
  return `local://${rel}`;
}

/**
 * Extract a short hash prefix from a canonical saved upload name.
 *
 * Canonical uploads are typically "<sha>.<ext>".
 *
 * @param {string | null | undefined} savedName
 * @param {number} [n]
 */
export function shortHashFromSavedName(savedName, n = SHORT_HASH_LEN) {
  if (!savedName) return '';
  const dot = savedName.lastIndexOf('.');
  const base = dot >= 0 ? savedName.slice(0, dot) : savedName;
  return base.slice(0, n);
}

/**
 * Human-friendly label for a backend upload response.
 *
 * @param {{ saved_name?: string, original_name?: string } | null | undefined} up
 * @param {number} [n]
 */
export function formatDisplayNameFromUpload(up, n = SHORT_HASH_LEN) {
  if (!up) return '';
  const sh = shortHashFromSavedName(up.saved_name, n);
  const name = up.original_name || up.saved_name;
  return sh ? `[${sh}] ${name}` : String(name || '');
}

/**
 * @param {string | null | undefined} path
 */
export function basename(path) {
  if (!path) return '';
  const p = String(path).replaceAll('\\', '/');
  const i = p.lastIndexOf('/');
  return i >= 0 ? p.slice(i + 1) : p;
}

/**
 * Canonical stored uploads look like "<hex>.<ext>" where hex is sha256 (64),
 * but we allow 40+ to be tolerant.
 *
 * @param {string | null | undefined} fileName
 * @param {number} [n]
 */
export function shortHashFromCanonicalFilename(fileName, n = SHORT_HASH_LEN) {
  if (!fileName) return null;
  const m = String(fileName).match(/^([0-9a-f]{40,64})\.[A-Za-z0-9]+$/i);
  if (!m) return null;
  return m[1].slice(0, n);
}

/**
 * If we only have a backend path, try to show a short hash + the canonical filename.
 * Example: "E:\\...\\uploads\\<sha>.mat" -> "[04012af] <sha>.mat"
 *
 * @param {string | null | undefined} path
 */
export function formatFallbackUploadLabelFromPath(path) {
  if (!path) return '—';
  const file = basename(path);
  const sh = shortHashFromCanonicalFilename(file);
  return sh ? `[${sh}] ${file}` : String(path);
}
