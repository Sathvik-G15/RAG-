import { ButtonHTMLAttributes, forwardRef } from 'react';
import { twMerge } from 'tailwind-merge';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
}

const variantStyles = {
  primary: 'bg-medical-primary-600 text-white hover:bg-medical-primary-700 active:bg-medical-primary-800 focus:ring-medical-primary-500',
  secondary: 'bg-medical-neutral-100 text-medical-neutral-900 hover:bg-medical-neutral-200 active:bg-medical-neutral-300 focus:ring-medical-neutral-400 dark:bg-medical-neutral-800 dark:text-medical-neutral-100 dark:hover:bg-medical-neutral-700 dark:active:bg-medical-neutral-600',
  ghost: 'text-medical-neutral-700 hover:bg-medical-neutral-100 active:bg-medical-neutral-200 focus:ring-medical-neutral-400 dark:text-medical-neutral-300 dark:hover:bg-medical-neutral-800 dark:active:bg-medical-neutral-700',
  danger: 'bg-medical-danger-600 text-white hover:bg-medical-danger-700 active:bg-medical-danger-800 focus:ring-medical-danger-500',
  outline: 'border border-medical-neutral-300 bg-white hover:bg-medical-neutral-50 active:bg-medical-neutral-100 focus:ring-medical-neutral-400 dark:border-medical-neutral-600 dark:bg-medical-neutral-800 dark:hover:bg-medical-neutral-700 dark:active:bg-medical-neutral-600',
};

const sizeStyles = {
  sm: 'px-3 py-1.5 text-xs gap-1.5',
  md: 'px-4 py-2.5 text-sm gap-2',
  lg: 'px-6 py-3 text-base gap-2.5',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', loading, disabled, children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={twMerge(
          'btn-base',
          variantStyles[variant],
          sizeStyles[size],
          loading && 'cursor-wait',
          className
        )}
        disabled={disabled || loading}
        aria-busy={loading}
        {...props}
      >
        {loading && (
          <svg
            className="animate-spin h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';