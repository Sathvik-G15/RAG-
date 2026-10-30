import { useNavigate, useLocation } from 'react-router-dom';
import { useAuthContext } from '../contexts/AuthProvider';
import { type LoginRequest } from '../lib';
import { toast } from 'sonner';

export function useAuth() {
  const context = useAuthContext();
  const navigate = useNavigate();
  const location = useLocation();

  const login = async (credentials: LoginRequest) => {
    try {
      await context.login(credentials.username, credentials.password);
      toast.success('Welcome back!');

      const from = (location.state as { from?: { pathname?: string } })?.from?.pathname || '/';
      navigate(from, { replace: true });
    } catch (error) {
      toast.error('Invalid credentials');
      throw error;
    }
  };

  const logout = () => {
    context.logout();
    toast.success('Logged out successfully');
    navigate('/login', { replace: true });
  };

  return {
    isAuthenticated: context.isAuthenticated,
    isLoading: context.isLoading,
    user: context.user,
    login,
    logout,
    checkAuth: async () => {},
    authMode: context.authMode,
  };
}