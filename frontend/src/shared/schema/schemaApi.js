import { getJson } from '../api/http.js';

export async function getModelSchema() {
  // returns: { schema, defaults }
  return getJson('/api/v1/schema/model');
}

export async function getEnums() {
  return getJson('/api/v1/schema/enums');
}

export async function getAllDefaults() {
  return getJson('/api/v1/schema/defaults');
}

export async function getTuningDefaults() {
  return getJson('/api/v1/schema/tuning-defaults');
}
