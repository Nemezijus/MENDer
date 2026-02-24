/**
 * JSON Schema helpers.
 *
 * Landing zone for schema navigation utilities used across features.
 * We keep these small and resilient because backend schemas may evolve.
 */

/**
 * Resolve a $ref of the form '#/$defs/<Key>' within a schema bundle.
 *
 * @param {any} schema
 * @param {string} ref
 * @returns {any | null}
 */
export function resolveRef(schema, ref) {
  if (!schema || !ref || typeof ref !== 'string') return null;
  const prefix = '#/$defs/';
  if (!ref.startsWith(prefix)) return null;
  const key = ref.slice(prefix.length);
  return schema?.$defs?.[key] ?? null;
}

/**
 * Given a discriminated union schema (Pydantic), return the variant schema
 * matching the provided discriminator value.
 *
 * Works for both models (discriminatorKey='algo') and features
 * (discriminatorKey='method').
 *
 * @param {any} schema
 * @param {string} discriminatorKey
 * @param {string | null | undefined} discriminatorValue
 * @returns {any | null}
 */
export function getVariantSchema(schema, discriminatorKey, discriminatorValue) {
  if (!schema || !discriminatorKey || !discriminatorValue) return null;

  // Prefer explicit discriminator.mapping when provided.
  const mapping = schema?.discriminator?.mapping;
  if (mapping && mapping[discriminatorValue]) {
    const target = resolveRef(schema, mapping[discriminatorValue]);
    if (target) return target;
  }

  // Fallback: scan oneOf/anyOf entries.
  const variants = schema?.oneOf || schema?.anyOf || [];
  for (const entry of variants) {
    const target = entry?.$ref ? resolveRef(schema, entry.$ref) : entry;
    const v =
      target?.properties?.[discriminatorKey]?.const ??
      target?.properties?.[discriminatorKey]?.default;
    if (v === discriminatorValue) return target || null;
  }

  return null;
}

/**
 * Extract enum-like values from a sub-schema property.
 *
 * Handles common Pydantic shapes:
 * - { enum: [...] }
 * - { anyOf: [{const: ...}, {type:'null'}] }
 *
 * @param {any} sub
 * @param {string} key
 * @param {any[] | undefined | null} fallback
 * @returns {any[] | undefined | null}
 */
export function enumFromSubSchema(sub, key, fallback) {
  try {
    const p = sub?.properties?.[key];
    if (!p) return fallback;

    if (Array.isArray(p.enum)) {
      // If a higher-level enums bundle was provided (engine enums), prefer it to keep
      // ordering consistent across the UI.
      if (Array.isArray(fallback) && fallback.length) return fallback;
      return p.enum;
    }

    const list = (p.anyOf ?? p.oneOf ?? []).flatMap((x) => {
      if (Array.isArray(x.enum)) return x.enum;
      if (x.const != null) return [x.const];
      if (x.type === 'null') return [null];
      return [];
    });

    return list.length ? list : fallback;
  } catch {
    return fallback;
  }
}

/**
 * Convert a list of enum values into Mantine Select "data" items.
 *
 * By convention we represent null as the option { value: 'none', label: 'none' }
 * when includeNoneLabel is true.
 *
 * @param {any[] | undefined | null} enums
 * @param {{ includeNoneLabel?: boolean }} [opts]
 * @returns {{ value: string, label: string }[]}
 */
export function toSelectData(enums, opts = {}) {
  const { includeNoneLabel = false } = opts;

  const out = [];
  let hasNull = false;

  for (const v of enums ?? []) {
    if (v === null) {
      hasNull = true;
      continue;
    }
    out.push({ value: String(v), label: String(v) });
  }

  if (hasNull && includeNoneLabel) out.unshift({ value: 'none', label: 'none' });
  return out;
}

/**
 * Convert Select value back to nullable value.
 *
 * @param {string | null} v
 * @returns {string | null}
 */
export function fromSelectNullable(v) {
  return v === 'none' ? null : v;
}

/**
 * List discriminator values present in the union schema.
 *
 * @param {any} schema
 * @param {string} discriminatorKey
 * @returns {string[] | null}
 */
export function listDiscriminatorValues(schema, discriminatorKey) {
  if (!schema || !discriminatorKey) return null;

  const mapping = schema?.discriminator?.mapping;
  if (mapping && typeof mapping === 'object') {
    const keys = Object.keys(mapping);
    if (keys.length) return keys.map(String);
  }

  const variants = schema?.oneOf || schema?.anyOf || [];
  const set = new Set();
  for (const entry of variants) {
    const target = entry?.$ref ? resolveRef(schema, entry.$ref) : entry;
    const v =
      target?.properties?.[discriminatorKey]?.const ??
      target?.properties?.[discriminatorKey]?.default;
    if (v != null) set.add(String(v));
  }

  return set.size ? Array.from(set) : null;
}
