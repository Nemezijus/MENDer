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
