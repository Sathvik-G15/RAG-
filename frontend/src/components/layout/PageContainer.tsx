import { ReactNode } from 'react';

interface PageContainerProps {
  children: ReactNode;
  className?: string;
  title?: string;
  description?: string;
  action?: ReactNode;
}

export function PageContainer({
  children,
  className,
  title,
  description,
  action,
}: PageContainerProps) {
  return (
    <main className={className} id="main-content" role="main">
      <div className="container-page py-8">
        {(title || description || action) && (
          <div className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              {title && (
                <h1 className="text-2xl sm:text-3xl font-bold text-medical-neutral-900 dark:text-medical-neutral-100">
                  {title}
                </h1>
              )}
              {description && (
                <p className="mt-1 text-medical-neutral-600 dark:text-medical-neutral-400">
                  {description}
                </p>
              )}
            </div>
            {action && <div className="flex-shrink-0">{action}</div>}
          </div>
        )}
        <div className="animate-fade-in">{children}</div>
      </div>
    </main>
  );
}