import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import App from './App';
import './styles/globals.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5,
      gcTime: 1000 * 60 * 30,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
        <Toaster
          position="top-right"
          toastOptions={{
            classNames: {
              toast: 'rounded-lg border bg-white dark:bg-medical-neutral-800 text-medical-neutral-900 dark:text-medical-neutral-100 shadow-medical-lg',
              description: 'text-medical-neutral-600 dark:text-medical-neutral-400',
              actionButton: 'bg-medical-primary-600 hover:bg-medical-primary-700',
              cancelButton: 'bg-medical-neutral-200 hover:bg-medical-neutral-300 dark:bg-medical-neutral-700 dark:hover:bg-medical-neutral-600',
            },
          }}
        />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);