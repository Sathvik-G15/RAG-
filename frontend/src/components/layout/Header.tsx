import { NavLink } from 'react-router-dom';
import { useThemeContext } from '../../contexts/ThemeProvider';
import { 
  Sun, Moon, Monitor, HeartPulse
} from 'lucide-react';

function ThemeToggle() {
  const { theme, setTheme } = useThemeContext();

  const themes: Array<{ value: 'light' | 'dark' | 'system'; icon: JSX.Element; label: string }> = [
    { value: 'light', icon: <Sun className="h-4 w-4" />, label: 'Light mode' },
    { value: 'dark', icon: <Moon className="h-4 w-4" />, label: 'Dark mode' },
    { value: 'system', icon: <Monitor className="h-4 w-4" />, label: 'System theme' },
  ];

  const next = theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light';
  const current = themes.find((t) => t.value === theme)!;

  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      className="rounded-lg p-2 text-medical-neutral-500 hover:text-medical-neutral-900 hover:bg-medical-neutral-100 dark:text-medical-neutral-400 dark:hover:text-medical-neutral-100 dark:hover:bg-medical-neutral-800 transition-colors"
      aria-label={current.label}
      title={current.label}
    >
      {current.icon}
    </button>
  );
}

export function Header() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-medical-neutral-200 dark:border-medical-neutral-800 bg-white/95 dark:bg-medical-neutral-900/95 backdrop-blur-md shadow-sm">
      <div className="container-page">
        <div className="flex h-16 items-center justify-between gap-4">
          {/* Logo & Brand */}
          <NavLink
            to="/"
            className="flex items-center gap-3 text-xl font-bold text-medical-primary-600 dark:text-medical-primary-400 shrink-0 group"
            aria-label="CAAR-CDSS Home"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-medical-primary-500 to-medical-primary-700 text-white shadow-md shadow-medical-primary-500/20 group-hover:scale-105 transition-transform">
              <svg
                className="h-6 w-6"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M12 2v20M2 12h20M7 7l10 10M17 7L7 17" />
              </svg>
            </div>
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <span className="font-extrabold tracking-tight text-medical-neutral-900 dark:text-medical-neutral-100 text-lg">
                  CAAR<span className="text-medical-primary-600 dark:text-medical-primary-400">-CDSS</span>
                </span>
                <span className="inline-flex items-center rounded-full bg-emerald-100 dark:bg-emerald-950/60 px-2 py-0.5 text-[11px] font-semibold text-emerald-800 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse mr-1.5" />
                  Live 18.5k Guidelines
                </span>
              </div>
              <span className="text-[11px] text-medical-neutral-500 dark:text-medical-neutral-400 font-medium">
                Patient Symptom Checker & Clinical Decision Support
              </span>
            </div>
          </NavLink>

          {/* Right actions: Portal Badge + Theme Toggle */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-medical-primary-50 dark:bg-medical-primary-950/50 border border-medical-primary-200 dark:border-medical-primary-800 text-xs font-medium text-medical-primary-700 dark:text-medical-primary-300">
              <HeartPulse className="h-3.5 w-3.5 text-medical-primary-600 dark:text-medical-primary-400" />
              <span>Patient Portal Active</span>
            </div>

            <ThemeToggle />
          </div>
        </div>
      </div>
    </header>
  );
}
