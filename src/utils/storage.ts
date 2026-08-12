import type { PredictionResponse } from '@/services/api';

export interface HistoryItem extends PredictionResponse {
  id: string;
  title: string;
  createdAt: string;
}

const HISTORY_KEY = 'truthscan-history';

export function getHistory(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) as HistoryItem[] : [];
  } catch {
    return [];
  }
}

export function saveHistory(item: Omit<HistoryItem, 'id' | 'createdAt'>): HistoryItem {
  const saved: HistoryItem = { ...item, id: crypto.randomUUID(), createdAt: new Date().toISOString() };
  const next = [saved, ...getHistory()].slice(0, 50);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
  return saved;
}

export function clearHistory(): void {
  localStorage.removeItem(HISTORY_KEY);
}
