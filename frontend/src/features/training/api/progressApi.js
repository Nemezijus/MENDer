export async function fetchProgress(progressId) {
  const res = await fetch(`/api/v1/progress/${encodeURIComponent(progressId)}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Failed to fetch progress (${res.status})`);
  }
  return res.json();
}
