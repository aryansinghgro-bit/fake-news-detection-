import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { Navbar } from '@/components/Navbar';
import { Footer } from '@/components/Footer';
import { Home } from '@/pages/Home';
import { Analyzer } from '@/pages/Analyzer';
import { History } from '@/pages/History';
import { HowItWorks } from '@/pages/HowItWorks';
import { About } from '@/pages/About';

function App() {
  return <BrowserRouter><Navbar /><Routes><Route path="/" element={<Home />} /><Route path="/analyzer" element={<Analyzer />} /><Route path="/history" element={<History />} /><Route path="/how-it-works" element={<HowItWorks />} /><Route path="/about" element={<About />} /></Routes><Footer /></BrowserRouter>;
}
export default App;
