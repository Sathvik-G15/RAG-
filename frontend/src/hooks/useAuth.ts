import { useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { api, type LoginRequest } from '../lib';
import { getAuthToken, setAuthToken, clearAuthToken, getAuthMode } from '../lib/auth';
import { toast } from 'sonner';

export function useAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  const checkAuth = useCallback(async () => {
    try {
      const token = await getAuthToken();
      setIsAuthenticated(!!token);
    } catch {
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const login = async (credentials: LoginRequest) => {
    try {
      const response = await api.post<{ access_token: string }>('/login', credentials);
      setAuthToken(response.access_token);
      setIsAuthenticated(true);
      toast.success('Welcome back!');

      const from = location.state?.from?.pathname || '/';
      navigate(from, { replace: true });
    } catch (error) {
      toast.error('Invalid credentials');
      throw error;
    }
  };

  const logout = () => {
    clearAuthToken();
    setIsAuthenticated(false);
    toast.success('Logged out successfully');
    navigate('/login', { replace: true });
  };

  return {
    isAuthenticated,
    isLoading,
    login,
    logout,
    checkAuth,
    authMode: getAuthMode(),
  };
}