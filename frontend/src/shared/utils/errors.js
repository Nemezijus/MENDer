/**
 * Centralized, UI-friendly error normalization.
 *
 * Goal:
 * - Convert a wide variety of error shapes (axios, fetch, FastAPI/Pydantic, strings)
 *   into a single readable string for <Alert> components.
 * - Keep the output stable so different panels don't implement slightly
 *   different error formatters.
 */

function isPlainObject(v) {
  return (
    v !== null &&
    typeof v === 'object' &&
    (v.constructor === Object || Object.getPrototypeOf(v) === null)
  );
}

function safeStringify(v) {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

function pydanticItemToLine(it) {
  if (typeof it === 'string') return it;
  if (!it || typeof it !== 'object') return String(it);

  const loc = Array.isArray(it.loc) ? it.loc.join('.') : it.loc;
  const msg = it.msg ?? it.message ?? null;
  if (msg) return loc ? `${loc}: ${msg}` : String(msg);
  return safeStringify(it);
}

function extractDetail(data) {
  if (data == null) return null;

  // FastAPI often returns { detail: ... }
  const detail = isPlainObject(data) ? (data.detail ?? data.message) : null;
  const payload = detail ?? data;

  if (typeof payload === 'string') return payload;

  if (Array.isArray(payload)) {
    return payload.map(pydanticItemToLine).join('\n');
  }

  if (isPlainObject(payload)) {
    // If it's an object, try to pick a common message field.
    const msg = payload.msg ?? payload.message;
    if (typeof msg === 'string') return msg;
    return safeStringify(payload);
  }

  return String(payload);
}

/**
 * @param {any} err
 * @param {object} [opts]
 * @param {string} [opts.fallback='Unexpected error']
 */
export function toErrorText(err, opts = {}) {
  const { fallback = 'Unexpected error' } = opts;

  if (err == null) return fallback;
  if (typeof err === 'string') return err;

  // axios error shape: { response: { data, status, ... }, message, ... }
  const axiosData = err?.response?.data;
  const detail = extractDetail(axiosData) ?? extractDetail(err?.detail);
  if (detail) return detail;

  const msg = err?.message;
  if (typeof msg === 'string' && msg.trim()) return msg;

  // Some code paths throw raw Response-like objects.
  const respText = extractDetail(err);
  if (respText && respText !== '[object Object]') return respText;

  return fallback;
}

/**
 * Best-effort extraction of structured details (e.g. Pydantic validation errors).
 * Returns an array of readable lines.
 *
 * @param {any} err
 * @returns {string[]}
 */
export function toErrorLines(err) {
  if (err == null) return [];

  const axiosData = err?.response?.data;
  const detail = axiosData?.detail ?? err?.detail;
  const payload = detail ?? axiosData;

  if (Array.isArray(payload)) return payload.map(pydanticItemToLine);

  const text = toErrorText(err);
  return text ? String(text).split('\n').filter(Boolean) : [];
}
