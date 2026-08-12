import { useState } from 'react';
import { ArrowUpRight, Menu, ShieldCheck, X } from 'lucide-react';
import { NavLink, useNavigate } from 'react-router-dom';
import { NAV_ITEMS } from '@/constants';

export function Navbar() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  return <header className="topbar">
    <div className="shell topbar-inner">
      <NavLink to="/" className="brand" onClick={() => setOpen(false)}>
        <span className="brand-mark"><ShieldCheck size={18} strokeWidth={2.5} /></span>
        <span>TRUTHSCAN <b>AI</b></span>
      </NavLink>
      <nav className={`desktop-nav ${open ? 'mobile-open' : ''}`} aria-label="Primary navigation">
        {NAV_ITEMS.map(({ label, href, icon: Icon }) => <NavLink key={href} to={href} onClick={() => setOpen(false)} className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}><Icon size={15} />{label}</NavLink>)}
      </nav>
      <button className="button button-small button-accent nav-cta" onClick={() => { navigate('/analyzer'); setOpen(false); }}>Start analysis <ArrowUpRight size={15} /></button>
      <button className="menu-button" aria-label={open ? 'Close menu' : 'Open menu'} onClick={() => setOpen(!open)}>{open ? <X /> : <Menu />}</button>
    </div>
  </header>;
}
