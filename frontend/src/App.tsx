import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthProvider';
import { ThemeProvider } from './contexts/ThemeProvider';
import { Header, Footer } from './components/layout';
import { ErrorBoundary } from './components/feedback/ErrorBoundary';
import { PatientPortal } from './pages/PatientPortal';

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<PatientPortal />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function AppLayout() {
  return (
    <div className="min-h-screen flex flex-col bg-medical-neutral-50 dark:bg-medical-neutral-900 text-medical-neutral-900 dark:text-medical-neutral-100">
      <ErrorBoundary>
        <Header />
      </ErrorBoundary>
      <main className="flex-1 container-page py-6 sm:py-8" id="main-content" aria-label="Main content">
        <ErrorBoundary>
          <AppRoutes />
        </ErrorBoundary>
      </main>
      <Footer />
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <AppLayout />
      </AuthProvider>
    </ThemeProvider>
  );
}
