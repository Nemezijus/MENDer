/**
 * Minimal HTTP "landing zone".
 *
 * The repo currently mixes axios and raw fetch across panels.
 * This module provides a single place to standardize:
 * - request helpers (json vs formdata)
 * - error shaping
 *
 * For now this is intentionally small and not yet wired into every panel.
 */

import api from './client.js';
import { toErrorText } from '../utils/errors.js';

function rethrowNormalized(e) {
  // Preserve the original error on the thrown Error so callers can inspect it.
  const err = new Error(toErrorText(e));
  err.cause = e;
  throw err;
}

/**
 * When axios is configured with responseType='blob', error responses may
 * also arrive as a Blob (e.g. FastAPI JSON errors). Convert them to text
 * so users don't see "[object Blob]".
 *
 * @param {any} e
 * @returns {Promise<never>}
 */
async function rethrowNormalizedBlobAware(e) {
  const data = e?.response?.data;
  // Blob exists in browsers; guard for non-browser environments.
  const isBlob = typeof Blob !== 'undefined' && data instanceof Blob;

  if (isBlob) {
    try {
      const text = await data.text();
      // Try to rehydrate JSON so toErrorText can pick {detail: ...}
      try {
        const parsed = JSON.parse(text);
        const err = new Error(toErrorText({ ...e, response: { ...(e.response || {}), data: parsed } }));
        err.cause = e;
        throw err;
      } catch {
        const err = new Error(text || toErrorText(e));
        err.cause = e;
        throw err;
      }
    } catch {
      // Fall back to the generic formatter.
      rethrowNormalized(e);
    }
  }

  rethrowNormalized(e);
}

/**
 * @template T
 * @param {string} url
 * @param {import('axios').AxiosRequestConfig} [config]
 * @returns {Promise<T>}
 */
export async function getJson(url, config) {
  try {
    const res = await api.get(url, config);
    return res.data;
  } catch (e) {
    rethrowNormalized(e);
  }
}

/**
 * @template T
 * @param {string} url
 * @param {any} body
 * @param {import('axios').AxiosRequestConfig} [config]
 * @returns {Promise<T>}
 */
export async function postJson(url, body, config) {
  try {
    const res = await api.post(url, body, config);
    return res.data;
  } catch (e) {
    rethrowNormalized(e);
  }
}

/**
 * @template T
 * @param {string} url
 * @param {any} body
 * @param {import('axios').AxiosRequestConfig} [config]
 * @returns {Promise<T>}
 */
export async function putJson(url, body, config) {
  try {
    const res = await api.put(url, body, config);
    return res.data;
  } catch (e) {
    rethrowNormalized(e);
  }
}

/**
 * @template T
 * @param {string} url
 * @param {import('axios').AxiosRequestConfig} [config]
 * @returns {Promise<T>}
 */
export async function delJson(url, config) {
  try {
    const res = await api.delete(url, config);
    return res.data;
  } catch (e) {
    rethrowNormalized(e);
  }
}

/**
 * POST multipart/form-data.
 *
 * @template T
 * @param {string} url
 * @param {FormData} formData
 * @param {import('axios').AxiosRequestConfig} [config]
 * @returns {Promise<T>}
 */
export async function postFormData(url, formData, config) {
  try {
    const res = await api.post(url, formData, {
      ...config,
      headers: {
        ...(config?.headers || {}),
        // Let the browser set the correct boundary.
      },
    });
    return res.data;
  } catch (e) {
    rethrowNormalized(e);
  }
}

/**
 * POST and return a Blob (also returns response headers).
 *
 * Useful for streaming endpoints like:
 * - /api/v1/models/save
 * - /api/v1/models/apply/export
 * - /api/v1/decoder/export
 *
 * @param {string} url
 * @param {any} body
 * @param {import('axios').AxiosRequestConfig} [config]
 * @returns {Promise<{ blob: Blob, headers: Record<string, string> }>} 
 */
export async function postBlob(url, body, config) {
  try {
    const res = await api.post(url, body, {
      ...config,
      responseType: 'blob',
    });
    return { blob: res.data, headers: res.headers || {} };
  } catch (e) {
    await rethrowNormalizedBlobAware(e);
  }
}
