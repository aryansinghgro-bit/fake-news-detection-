import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

interface Props { real: number; fake: number }
export function ConfidenceChart({ real, fake }: Props) {
  const data = [{ name: 'REAL', value: real }, { name: 'FAKE', value: fake }];
  return <div className="chart-wrap"><ResponsiveContainer width="100%" height={190}><PieChart><Pie data={data} dataKey="value" nameKey="name" innerRadius={58} outerRadius={78} paddingAngle={4} stroke="none"><Cell fill="#48d597" /><Cell fill="#ff6b73" /></Pie><Tooltip formatter={(value) => [`${Number(value).toFixed(2)}%`, 'Model score']} contentStyle={{ background: '#111927', border: '1px solid #26354b', borderRadius: 10, color: '#f4f7fb' }} /></PieChart></ResponsiveContainer><div className="chart-center"><strong>{Math.max(real, fake).toFixed(2)}%</strong><span>confidence</span></div></div>;
}
