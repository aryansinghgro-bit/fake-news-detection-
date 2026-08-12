const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000/api';

export interface PredictionResponse {
  prediction: 'REAL' | 'FAKE';
  confidence: number;
  probabilities: { REAL: number; FAKE: number };
}

export interface ModelInfo {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
  train_samples: number;
  test_samples: number;
  total_samples: number;
  model_type: string;
  max_features: number;
  ngram_range: number[];
}

interface ApiErrorPayload { error?: string }

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${API_URL}${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...options?.headers },
    });
    const payload = await response.json() as T & ApiErrorPayload;
    if (!response.ok) throw new Error(payload.error || 'The analysis server returned an error.');
    return payload;
  } catch (error) {
    if (error instanceof TypeError) throw new Error('Unable to connect to the analysis server. Please try again.');
    throw error;
  }
}

export function predictNews(title: string, text: string): Promise<PredictionResponse> {
  return request<PredictionResponse>('/predict', {
    method: 'POST',
    body: JSON.stringify({ title, text }),
  });
}

export function getHealth(): Promise<{ status: string; service: string; model_loaded: boolean }> {
  return request('/health');
}

export function getModelInfo(): Promise<ModelInfo> {
  return request('/model-info');
}
