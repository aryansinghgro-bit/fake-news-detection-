import { Activity, BookOpen, Compass, Home, Info } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export interface NavItem { label: string; href: string; icon: LucideIcon }

export const NAV_ITEMS: NavItem[] = [
  { label: 'Home', href: '/', icon: Home },
  { label: 'Analyzer', href: '/analyzer', icon: Activity },
  { label: 'History', href: '/history', icon: BookOpen },
  { label: 'How it works', href: '/how-it-works', icon: Compass },
  { label: 'About', href: '/about', icon: Info },
];

export const MODEL_STEPS = [
  { number: '01', label: 'Input', description: 'Headline and article content enter the analysis workspace.' },
  { number: '02', label: 'Normalize', description: 'Text is cleaned and normalized for consistent processing.' },
  { number: '03', label: 'TF-IDF', description: 'Language is translated into weighted numerical features.' },
  { number: '04', label: 'Classify', description: 'Logistic Regression evaluates learned language patterns.' },
  { number: '05', label: 'Result', description: 'A classification and model confidence score are returned.' },
];
