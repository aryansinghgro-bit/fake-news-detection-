import { BookOpen, CalendarDays, Search, Trash2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { getHistory, clearHistory, type HistoryItem } from '@/utils/storage';

function formatDate(value: string): string { return new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(new Date(value)); }
export function History() {
  const [items, setItems] = useState<HistoryItem[]>(getHistory());
  const [query, setQuery] = useState('');
  const filtered = useMemo(() => items.filter((item) => item.title.toLowerCase().includes(query.toLowerCase())), [items, query]);
  const clear = () => { clearHistory(); setItems([]); };
  return <main className="page shell"><div className="page-heading"><div><p className="eyebrow">Your workspace</p><h1>Analysis <em>history</em></h1><p>Recent predictions saved locally in this browser.</p></div>{items.length > 0 && <button className="button button-ghost danger-button" onClick={clear}><Trash2 size={15} /> Clear history</button>}</div>{items.length > 0 && <div className="search-field"><Search size={17} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search headlines" /></div>}{filtered.length === 0 ? <div className="empty-state"><BookOpen size={28} /><h2>{items.length ? 'No matches found' : 'No analyses yet'}</h2><p>{items.length ? 'Try a different headline search.' : 'Run your first analysis and it will appear here.'}</p></div> : <div className="history-list">{filtered.map((item) => <article className="history-item" key={item.id}><div className={`history-status ${item.prediction.toLowerCase()}`}><span>{item.prediction}</span><strong>{item.confidence.toFixed(2)}%</strong></div><div className="history-copy"><h3>{item.title}</h3><p><CalendarDays size={14} /> {formatDate(item.createdAt)} <span>•</span> model confidence</p></div></article>)}</div>}</main>;
}
