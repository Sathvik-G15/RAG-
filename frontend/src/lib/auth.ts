import { config } from './config';

const TOKEN_KEY = 'caar_cdss_token';
const REFRESH_THRESHOLD = 60 * 1000; // 1 minute before expiry

interface TokenPayload {
  sub: string;
  exp: number;
  iat: number;
  [key: string]: unknown;
}

let memoryToken: string | null = null;
let refreshPromise: Promise<string> | null = null;

function parseJwt(token: string): TokenPayload | null {
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

function isTokenExpiringSoon(token: string): boolean {
  const payload = parseJwt(token);
  if (!payload?.exp) return true;
  const expiryTime = payload.exp * 1000;
  return Date.now() + REFRESH_THRESHOLD >= expiryTime;
}

async function doRefreshToken(): Promise<string> {
  const response = await fetch(`${config.VITE_API_URL}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'doctor', password: 'x' }),
    credentials: config.VITE_AUTH_MODE === 'cookie' ? 'include' : 'omit',
  });

  if (!response.ok) {
    throw new Error('Failed to refresh token');
  }

  const data = await response.json();
  return data.access_token;
}

async function refreshToken(): Promise<string> {
  if (refreshPromise) return refreshPromise;

  refreshPromise = doRefreshToken().finally(() => {
    refreshPromise = null;
  });

  try {
    return await refreshPromise;
  } catch {
    clearAuthToken();
    throw new Error('Token refresh failed');
  }
}

export async function getAuthToken(): Promise<string | null> {
  if (config.VITE_AUTH_MODE === 'cookie') {
    return null;
  }

  if (memoryToken && !isTokenExpiringSoon(memoryToken)) {
    return memoryToken;
  }

  if (memoryToken) {
    try {
      memoryToken = await refreshToken();
      return memoryToken;
    } catch {
      return null;
    }
  }

  const stored = sessionStorage.getItem(TOKEN_KEY);
  if (stored && !isTokenExpiringSoon(stored)) {
    memoryToken = stored;
    return memoryToken;
  }

  if (stored) {
    try {
      memoryToken = await refreshToken();
      sessionStorage.setItem(TOKEN_KEY, memoryToken);
      return memoryToken;
    } catch {
      return null;
    }
  }

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