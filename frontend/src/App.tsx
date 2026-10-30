import { Routes, Route, Navigate, Outlet } from 'react-router-dom';
import { AuthProvider, useAuthContext } from './contexts/AuthProvider';
import { ThemeProvider } from './contexts/ThemeProvider';
import { Header, Footer } from './components/layout';
import { ErrorBoundary } from './components/feedback/ErrorBoundary';
import { PatientPortal, LoginPage } from './pages';

function ProtectedRoute() {
  const { isAuthenticated, isLoading } = useAuthContext();

  if (isLoading) {
    return (
      <div className="flex min-h-[400px] items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-medical-primary-600 border-t-transparent" aria-label="Loading" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: { pathname: '/' }} } />;
  }

  return <Outlet />;
}

function PublicRoute() {
  const { isAuthenticated, isLoading } = useAuthContext();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-medical-primary-600 border-t-transparent" aria-label="Loading" />
      </div>
    );
  }

  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return <Outlet />;
}

function AppRoutes() {
  return (
    <Routes>
      <Route element={<PublicRoute />}>
        <Route path="/login" element={<LoginPage />} />
      </Route>

      <Route element={<ProtectedRoute />}>
        <Route path="/" element={<PatientPortal />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function AppLayout() {
  return (
    <div className="min-h-screen flex flex-col bg-medical-neutral-50 dark:bg-medical-neutral-900">
      <ErrorBoundary>
        <Header />
      </ErrorBoundary>
      <main className="flex-1 container-page py-8" id="main-content" aria-label="Main content">
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