import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { loginSchema, type LoginRequest } from '../lib/validations';
import { Button, Input, Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui';
import { useAuth } from '../hooks/useAuth';
import { Stethoscope, ShieldCheck, Lock, Activity, Sparkles } from 'lucide-react';

export function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, isLoading } = useAuth();
  const [error, setError] = useState('');

  const from = location.state?.from?.pathname || '/';

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<LoginRequest>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      username: 'doctor',
      password: 'x',
    },
  });

  const onSubmit = async (data: LoginRequest) => {
    setError('');
    try {
      await login(data);
      navigate(from, { replace: true });
    } catch {
      setError('Invalid clinical credentials');
    }
  };

  const handleQuickLogin = async (username: string) => {
    setValue('username', username);
    setValue('password', 'dev-password');
    setError('');
    try {
      await login({ username, password: 'dev-password' });
      navigate(from, { replace: true });
    } catch {
      setError('Authentication failed');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-medical-neutral-50 via-white to-medical-primary-50/30 dark:from-medical-neutral-950 dark:via-medical-neutral-900 dark:to-medical-primary-950/20 px-4 py-12">
      <div className="w-full max-w-md space-y-6">
        <Card variant="elevated" padding="lg" className="border-medical-neutral-200 dark:border-medical-neutral-800 shadow-xl backdrop-blur-sm">
          <CardHeader className="text-center pb-2">
            <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-medical-primary-500 to-medical-primary-700 text-white shadow-lg shadow-medical-primary-500/25">
              <svg
                className="h-8 w-8"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M12 2v20M2 12h20M7 7l10 10M17 7L7 17" />
              </svg>
            </div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-medical-primary-50 dark:bg-medical-primary-950/60 text-medical-primary-700 dark:text-medical-primary-300 text-[11px] font-semibold mb-2 border border-medical-primary-200 dark:border-medical-primary-800">
              <Activity className="h-3 w-3" />
              <span>CAAR Clinical Decision Support</span>
            </div>
            <CardTitle className="text-2xl font-extrabold tracking-tight">
              Clinician Sign In
            </CardTitle>
            <CardDescription className="text-xs">
              Access the Adaptive Evidence Budgeting & Triage System
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-5 pt-2">
            {error && (
              <div
                className="p-3 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200 text-xs font-medium"
                role="alert"
              >
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <Input
                label="Staff Username / ID"
                placeholder="e.g. doctor"
                {...register('username')}
                error={errors.username?.message}
                disabled={isLoading}
              />

              <Input
                label="Access Key / Password"
                type="password"
                placeholder="••••••••"
                {...register('password')}
                error={errors.password?.message}
                disabled={isLoading}
              />

              <Button type="submit" className="w-full shadow-md shadow-medical-primary-500/20" loading={isLoading} size="lg">
                Sign In to Portal
              </Button>
            </form>

            {/* Quick Demo Sign-in */}
            <div className="pt-3 border-t border-medical-neutral-200 dark:border-medical-neutral-800 space-y-2">
              <div className="flex items-center gap-1.5 text-[11px] font-semibold text-medical-neutral-500 dark:text-medical-neutral-400">
                <Sparkles className="h-3 w-3 text-amber-500" />
                <span>Quick Evaluation Demo Access:</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => handleQuickLogin('doctor')}
                  disabled={isLoading}
                  className="flex items-center justify-center gap-1.5 p-2 rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800 hover:bg-medical-primary-50 dark:hover:bg-medical-primary-950/40 text-medical-neutral-700 dark:text-medical-neutral-300 hover:text-medical-primary-700 dark:hover:text-medical-primary-300 border border-medical-neutral-200 dark:border-medical-neutral-700 text-xs font-medium transition-colors"
                >
                  <Stethoscope className="h-3.5 w-3.5 text-medical-primary-600" />
                  <span>Clinician Demo</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleQuickLogin('admin')}
                  disabled={isLoading}
                  className="flex items-center justify-center gap-1.5 p-2 rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800 hover:bg-medical-primary-50 dark:hover:bg-medical-primary-950/40 text-medical-neutral-700 dark:text-medical-neutral-300 hover:text-medical-primary-700 dark:hover:text-medical-primary-300 border border-medical-neutral-200 dark:border-medical-neutral-700 text-xs font-medium transition-colors"
                >
                  <Lock className="h-3.5 w-3.5 text-medical-primary-600" />
                  <span>Admin Demo</span>
                </button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Security / Compliance footer */}
        <div className="flex items-center justify-center gap-2 text-xs text-medical-neutral-500 dark:text-medical-neutral-400">
          <ShieldCheck className="h-4 w-4 text-emerald-600" />
          <span>PHI Safe · De-identified Clinical Environment</span>
        </div>
      </div>
    </div>
  );
}
