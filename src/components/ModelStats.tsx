import { useEffect, useState } from 'react';
import { BarChart3, Database, Gauge, Layers3 } from 'lucide-react';
import { getModelInfo, type ModelInfo } from '@/services/api';

export function ModelStats() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  useEffect(() => { getModelInfo().then(setInfo).catch(() => undefined); }, []);
  const stats = info ? [{ label: 'Accuracy', value: info.accuracy, icon: Gauge }, { label: 'Precision', value: info.precision, icon: BarChart3 }, { label: 'Recall', value: info.recall, icon: Layers3 }, { label: 'F1 score', value: info.f1, icon: Database }] : [];
  return <div className="stats-grid">{stats.map(({ label, value, icon: Icon }) => <div className="stat-card" key={label}><Icon size={17} /><span>{label}</span><strong>{value.toFixed(2)}%</strong></div>)}</div>;
}
