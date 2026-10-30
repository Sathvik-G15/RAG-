import { NavLink } from 'react-router-dom';
import { clsx } from 'clsx';
import { useThemeContext } from '../../contexts/ThemeProvider';
import { useAuthContext } from '../../contexts/AuthProvider';
import { Button } from '../ui/Button';
import { Sun, Moon, Monitor, LogOut } from 'lucide-react';

const navigation = [
  { name: 'Patient', href: '/' },
] as const;

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
  const { isAuthenticated, user, logout } = useAuthContext();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-medical-neutral-200 dark:border-medical-neutral-700 bg-white/80 dark:bg-medical-neutral-900/80 backdrop-blur-sm">
      <div className="container-page">
        <div className="flex h-16 items-center justify-between gap-4">
          {/* Logo */}
          <NavLink
            to="/"
            className="flex items-center gap-2 text-xl font-bold text-medical-primary-600 dark:text-medical-primary-400 shrink-0"
            aria-label="CAAR-CDSS Home"
          >
            <svg
              className="h-8 w-8"
              viewBox="0 0 32 32"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <rect width="32" height="32" rx="8" className="fill-medical-primary-600" />
              <path
                d="M8 16L14 22L24 10"
                stroke="white"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <span className="hidden sm:inline">CAAR-CDSS</span>
          </NavLink>

          {/* Nav */}
          <nav className="flex items-center gap-1 overflow-x-auto" aria-label="Main navigation">
            {navigation.map((item) => (
              <NavLink
                key={item.name}
                to={item.href}
                end={item.href === '/'}
                className={({ isActive }) =>
                  clsx(
                    'px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap',
                    isActive
                      ? 'bg-medical-primary-50 dark:bg-medical-primary-900/30 text-medical-primary-700 dark:text-medical-primary-300'
                      : 'text-medical-neutral-600 dark:text-medical-neutral-400 hover:text-medical-neutral-900 dark:hover:text-medical-neutral-100 hover:bg-medical-neutral-100 dark:hover:bg-medical-neutral-800'
                  )
                }
              >
                {item.name}
              </NavLink>
            ))}
          </nav>

          {/* Right actions */}
          <div className="flex items-center gap-2 shrink-0">
            <ThemeToggle />

            {isAuthenticated && (
              <Button
                variant="ghost"
                size="sm"
                onClick={logout}
                title={`Sign out${user ? ` (${user.username})` : ''}`}
                aria-label="Sign out"
                className="hidden sm:flex"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden md:inline">Sign out</span>
              </Button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}