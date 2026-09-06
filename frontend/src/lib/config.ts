import { z } from 'zod';

const envSchema = z.object({
  VITE_API_URL: z.string().url().default('http://localhost:8000'),
  VITE_AUTH_MODE: z.enum(['memory', 'cookie']).default('memory'),
  VITE_APP_NAME: z.string().default('CAAR-CDSS'),
  VITE_APP_VERSION: z.string().default('0.1.0'),
  VITE_ENABLE_DEVTOOLS: z.string().transform((v) => v === 'true').default('false'),
  VITE_SENTRY_DSN: z.string().url().optional(),
});

export type EnvConfig = z.infer<typeof envSchema>;

let cachedConfig: EnvConfig | null = null;

export function getConfig(): EnvConfig {
  if (cachedConfig) return cachedConfig;

  const parsed = envSchema.safeParse(import.meta.env);
  if (!parsed.success) {
    console.error('Invalid environment configuration:', parsed.error.flatten().fieldErrors);
    throw new Error('Invalid environment configuration');
  }

  cachedConfig = parsed.data;
  return cachedConfig;
}

export const config = getConfig();