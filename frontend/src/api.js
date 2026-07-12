const API_BASE = 'http://localhost:8000';

export async function predictImage(file, model) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', model);

  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || 'Prediction failed');
  }

  return response.json();
}
