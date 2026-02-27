/**
 * Shared UI label shaping for scaling methods.
 *
 * NOTE: This is presentation-only; the backend/engine owns scaling behavior.
 */

export function makeScaleOptionLabel(v) {
  if (v == null) return '';
  const name = String(v);

  const labelBase =
    name === 'none'
      ? 'None'
      : name.charAt(0).toUpperCase() + name.slice(1);

  const suffix =
    name === 'none'
      ? ''
      : name.toLowerCase().endsWith('abs')
      ? ' Scaler'
      : name === 'quantile'
      ? ' Transformer'
      : ' Scaler';

  return `${labelBase}${suffix}`;
}
