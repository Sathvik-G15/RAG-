import { NavLink } from 'react-router-dom';
import { clsx } from 'clsx';

const navigation = [
  { name: 'Patient', href: '/' },
  { name: 'Doctor', href: '/doctor' },
  { name: 'Admin', href: '/admin' },
  { name: 'Corpus', href: '/corpus' },
] as const;

export function Header() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-medical-neutral-200 dark:border-medical-neutral-700 bg-white/80 dark:bg-medical-neutral-900/80 backdrop-blur-sm">
      <div className="container-page">
        <div className="flex h-16 items-center justify-between">
          <NavLink
            to="/"
            className="flex items-center gap-2 text-xl font-bold text-medical-primary-600 dark:text-medical-primary-400"
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
            <span>CAAR-CDSS</span>
          </NavLink>

          <nav className="flex items-center gap-1" aria-label="Main navigation">
            {navigation.map((item) => (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) =>
                  clsx(
                    'px-3 py-2 rounded-lg text-sm font-medium transition-colors',
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
        </div>
      </div>
    </header>
  );
}