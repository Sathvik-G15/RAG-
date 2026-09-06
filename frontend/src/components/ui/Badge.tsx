import { twMerge } from 'tailwind-merge';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'outline';
  size?: 'sm' | 'md';
  dot?: boolean;
}

const variantStyles = {
  default: 'bg-medical-neutral-100 text-medical-neutral-800 dark:bg-medical-neutral-800 dark:text-medical-neutral-200',
  success: 'bg-medical-success-100 text-medical-success-800 dark:bg-medical-success-900/30 dark:text-medical-success-400',
  warning: 'bg-medical-warning-100 text-medical-warning-800 dark:bg-medical-warning-900/30 dark:text-medical-warning-400',
  danger: 'bg-medical-danger-100 text-medical-danger-800 dark:bg-medical-danger-900/30 dark:text-medical-danger-400',
  info: 'bg-medical-primary-100 text-medical-primary-800 dark:bg-medical-primary-900/30 dark:text-medical-primary-400',
  outline: 'border border-medical-neutral-300 bg-transparent text-medical-neutral-700 dark:border-medical-neutral-600 dark:text-medical-neutral-300',
};

const sizeStyles = {
  sm: 'px-2 py-0.5 text-xs gap-1',
  md: 'px-2.5 py-0.5 text-xs gap-1.5',
};

const dotColors = {
  default: 'bg-medical-neutral-500',
  success: 'bg-medical-success-500',
  warning: 'bg-medical-warning-500',
  danger: 'bg-medical-danger-500',
  info: 'bg-medical-primary-500',
  outline: 'bg-medical-neutral-500',
};

export function Badge({
  className,
  variant = 'default',
  size = 'md',
  dot,
  children,
  ...props
}: BadgeProps) {
  return (
    <span
      className={twMerge(
        'badge-base',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      {...props}
    >
      {dot && (
        <span
          className={twMerge('h-1.5 w-1.5 rounded-full flex-shrink-0', dotColors[variant])}
          aria-hidden="true"
        />
      )}
      {children}
    </span>
  );
}