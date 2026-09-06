import { clsx } from 'clsx';
import { motion } from 'framer-motion';

interface LoadingOverlayProps {
  isLoading: boolean;
  children: React.ReactNode;
  message?: string;
  fullScreen?: boolean;
}

export function LoadingOverlay({
  isLoading,
  children,
  message = 'Loading...',
  fullScreen = false,
}: LoadingOverlayProps) {
  return (
    <div className={clsx('relative', fullScreen && 'fixed inset-0 z-50')}>
      {children}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className={clsx(
            'flex items-center justify-center bg-white/80 dark:bg-medical-neutral-900/80 backdrop-blur-sm transition-opacity',
            fullScreen
              ? 'fixed inset-0 z-50'
              : 'absolute inset-0 rounded-xl'
          )}
          role="status"
          aria-live="polite"
          aria-label={message}
        >
          <div className="flex flex-col items-center gap-4 p-6">
            <div className="relative">
              <svg className="h-12 w-12 text-medical-primary-600 animate-spin" viewBox="0 0 24 24" aria-hidden="true">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
            </div>
            <p className="text-sm font-medium text-medical-neutral-700 dark:text-medical-neutral-300">{message}</p>
          </div>
        </motion.div>
      )}
    </div>
  );
}