import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { cn } from '../../lib/utils';

export function DisclaimerBanner({ onDismiss }: { onDismiss?: () => void }) {
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('disclaimerDismissed');
    if (stored) setDismissed(true);
  }, []);

  if (dismissed) return null;

  const handleDismiss = () => {
    setDismissed(true);
    localStorage.setItem('disclaimerDismissed', 'true');
    onDismiss?.();
  };

  return (
    <div
      className={cn(
        'fixed bottom-4 right-4 z-50 max-w-md animate-slide-up',
        'bg-medical-warning-50 dark:bg-medical-warning-900/30 border border-medical-warning-200 dark:border-medical-warning-800 rounded-xl shadow-medical-lg p-4'
      )}
      role="alert"
    >
      <div className="flex gap-3">
        <div className="flex-shrink-0">
          <svg
            className="h-5 w-5 text-medical-warning-600 dark:text-medical-warning-400"
            fill="currentColor"
            viewBox="0 0 20 20"
            aria-hidden="true"
          >
            <path
              fillRule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-medical-warning-800 dark:text-medical-warning-200">
            Research Prototype
          </p>
          <p className="mt-1 text-sm text-medical-warning-700 dark:text-medical-warning-300">
            CAAR-CDSS is a research prototype for academic use only. Not a medical device.
            Do not use for clinical decision making without clinician review.
          </p>
        </div>
        <button
          type="button"
          onClick={handleDismiss}
          className="flex-shrink-0 rounded-lg p-1 text-medical-warning-500 hover:text-medical-warning-700 hover:bg-medical-warning-100 dark:hover:bg-medical-warning-800 transition-colors"
          aria-label="Dismiss disclaimer"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}