export async function getModelSchema() {
  const res = await fetch('/api/v1/schema/model');
  if (!res.ok) throw new Error(`Schema fetch failed: ${res.status}`);
  return res.json(); // { schema, defaults }
}

export async function getEnums() {
  const res = await fetch('/api/v1/schema/enums');
  if (!res.ok) throw new Error(`Schema fetch failed: ${res.status}`);
  return res.json();
}