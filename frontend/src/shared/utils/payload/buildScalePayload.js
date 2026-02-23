import { compactPayload } from '../compactPayload.js';

/**
 * Build the "scale" section of request payloads.
 */
export function buildScalePayload({ method } = {}) {
  return compactPayload({
    method,
  });
}
