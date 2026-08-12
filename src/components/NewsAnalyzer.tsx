import { FormEvent, useState } from 'react';
import { ArrowRight, RotateCcw, ScanSearch, LoaderCircle } from 'lucide-react';
import { usePrediction } from '@/hooks/usePrediction';
import { PredictionResult } from './PredictionResult';

export function NewsAnalyzer() {
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const { result, isLoading, error, analyze } = usePrediction();
  const submit = async (event: FormEvent) => { event.preventDefault(); await analyze(title, text); };
  const clear = () => { setTitle(''); setText(''); };
  return <div className="analyzer-layout"><form className="analyzer-card" onSubmit={submit}>
    <div className="section-heading"><div className="icon-box"><ScanSearch size={20} /></div><div><p className="eyebrow">Analysis workspace</p><h2>Inspect a story</h2></div><span className="status-pill"><i /> Model online</span></div>
    <div className="field"><label htmlFor="headline">News headline <span>required</span></label><input id="headline" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Enter the article headline" maxLength={240} /></div>
    <div className="field"><div className="label-row"><label htmlFor="article">News article <span>required</span></label><small>{text.length.toLocaleString()} / 50,000</small></div><textarea id="article" value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste the full article content here for a more useful signal..." maxLength={50000} rows={9} /></div>
    {error && <div className="error-message" role="alert">{error}</div>}
    <div className="form-actions"><button type="button" className="button button-ghost" onClick={clear}><RotateCcw size={16} /> Clear</button><button type="submit" className="button button-accent button-grow" disabled={isLoading}>{isLoading ? <><LoaderCircle className="spin" size={17} /> Analyzing...</> : <>Analyze news <ArrowRight size={17} /></>}</button></div>
  </form>{result && <PredictionResult result={result} />}</div>;
}
