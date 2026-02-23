import { compactPayload } from '../compactPayload.js';

/**
 * Build the "data" section of request payloads.
 *
 * Convention:
 * - Overrides only (omit unset fields).
 * - If npz_path is provided, omit x_path/y_path so the backend can infer.
 */
export function buildDataPayload({ xPath, yPath, npzPath, xKey, yKey } = {}) {
  return compactPayload({
    x_path: npzPath ? undefined : xPath,
    y_path: npzPath ? undefined : yPath,
    npz_path: npzPath,
    x_key: xKey,
    y_key: yKey,
  });
}
