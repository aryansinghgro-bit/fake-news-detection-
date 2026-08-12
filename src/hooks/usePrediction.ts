import { useCallback, useState } from 'react';
import { predictNews, type PredictionResponse } from '@/services/api';
import { saveHistory } from '@/utils/storage';

export function usePrediction() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const analyze = useCallback(async (title: string, text: string) => {
    setIsLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await predictNews(title, text);
      setResult(response);
      saveHistory({ title, ...response });
      return response;
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : 'Something went wrong while analyzing the article. Please try again.');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { result, isLoading, error, analyze, clearResult: () => setResult(null) };
}
