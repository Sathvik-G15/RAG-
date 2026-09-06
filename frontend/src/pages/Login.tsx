import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { loginSchema, type LoginRequest } from '../lib/validations';
import { Button, Input, Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui';
import { useAuth } from '../hooks/useAuth';

export function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, isLoading } = useAuth();
  const [error, setError] = useState('');

  const from = location.state?.from?.pathname || '/';

  const {
    register,
    handleSubmit,
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
      setError('Invalid username or password');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-medical-neutral-50 dark:bg-medical-neutral-900 px-4">
      <Card variant="elevated" padding="lg" className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-medical-primary-100 dark:bg-medical-primary-900/30">
            <svg
              className="h-7 w-7 text-medical-primary-600 dark:text-medical-primary-400"
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
          </div>
          <CardTitle className="text-2xl">Welcome to CAAR-CDSS</CardTitle>
          <CardDescription>
            Sign in to access the Clinical Decision Support System
          </CardDescription>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            {error && (
              <div
                className="p-3 rounded-lg bg-medical-danger-50 dark:bg-medical-danger-900/30 border border-medical-danger-200 dark:border-medical-danger-800 text-medical-danger-700 dark:text-medical-danger-300 text-sm"
                role="alert"
              >
                {error}
              </div>
            )}

            <Input
              label="Username"
              placeholder="Enter your username"
              {...register('username')}
              error={errors.username?.message}
              disabled={isLoading}
            />

            <Input
              label="Password"
              type="password"
              placeholder="Enter your password"
              {...register('password')}
              error={errors.password?.message}
              disabled={isLoading}
            />

            <Button type="submit" className="w-full" loading={isLoading} size="lg">
              Sign In
            </Button>
          </form>

          <div className="mt-6 p-3 rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800">
            <p className="text-xs text-medical-neutral-600 dark:text-medical-neutral-400 text-center">
              Demo credentials: <code className="font-mono">doctor</code> / <code className="font-mono">x</code>
            </p>
          </div>
        </CardContent>
      </Card>

      <p className="mt-6 text-center text-xs text-medical-neutral-500 dark:text-medical-neutral-400">
        Research prototype for academic use. Not a medical device.
      </p>
    </div>
  );
}