import { config } from './config';

const TOKEN_KEY = 'caar_cdss_token';
const EXPIRY_BUFFER_MS = 60 * 1000; // treat token as expired 1 minute early

export interface TokenPayload {
  sub: string;
  exp: number;
  iat: number;
  [key: string]: unknown;
}

let memoryToken: string | null = null;

export function parseJwt(token: string): TokenPayload | null {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  } catch {
    return null;
  }
}

function isTokenExpired(token: string): boolean {
  const payload = parseJwt(token);
  if (!payload?.exp) return true;
  return Date.now() + EXPIRY_BUFFER_MS >= payload.exp * 1000;
}

export async function getAuthToken(): Promise<string | null> {
  if (config.VITE_AUTH_MODE === 'cookie') {
    return null;
  }

  // Use in-memory token if still valid
  if (memoryToken && !isTokenExpired(memoryToken)) {
    return memoryToken;
  }

  // Fall back to session storage
  const stored = sessionStorage.getItem(TOKEN_KEY);
  if (stored && !isTokenExpired(stored)) {
    memoryToken = stored;
    return memoryToken;
  }

  // Auto-acquire dev token seamlessly for the patient portal if missing or expired
  try {
    const baseUrl = config.VITE_API_URL.replace(/\/+$/, '');
    const res = await fetch(`${baseUrl}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'patient', password: 'dev' }),
    });
    if (res.ok) {
      const data = await res.json();
      if (data?.access_token) {
        setAuthToken(data.access_token);
        return data.access_token;
      }
    }
  } catch {
    // Backend offline or mock test environment
  }

  // Token is expired or missing — clear and return null
  clearAuthToken();
  return null;
}

export function setAuthToken(token: string): void {
  memoryToken = token;
  if (config.VITE_AUTH_MODE === 'memory') {
    sessionStorage.setItem(TOKEN_KEY, token);
  }
}

export function clearAuthToken(): void {
  memoryToken = null;
  sessionStorage.removeItem(TOKEN_KEY);
}

export function getAuthMode(): AuthMode {
  return config.VITE_AUTH_MODE;
}

export type AuthMode = 'memory' | 'cookie';