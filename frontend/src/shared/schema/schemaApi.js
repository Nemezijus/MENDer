import { getJson } from '../api/http.js';

export async function getModelSchema() {
  // returns: { schema, defaults }
  // NOTE: shared/api/client.js already sets baseURL = '/api/v1'
  // so all paths here should be relative to that base.
  return getJson('/schema/model');
}

export async function getEnums() {
  return getJson('/schema/enums');
}

export async function getAllDefaults() {
  return getJson('/schema/defaults');
}

export async function getTuningDefaults() {
  return getJson('/schema/tuning-defaults');
}
