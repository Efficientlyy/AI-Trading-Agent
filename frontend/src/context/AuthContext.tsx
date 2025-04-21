import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { AuthState, AuthContextType } from '../types';

// API URL from environment variables
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Provider component
export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    user: null,
    token: null,
    isLoading: true,
    error: null
  });

  // Logout function - defined early so it can be used in other functions
  const logout = useCallback(() => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    localStorage.removeItem('email');
    setAuthState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null
    });
  }, []);

  // Initialize auth state from localStorage (runs only once on mount to prevent infinite update loop)
  useEffect(() => {
    const initializeAuth = async () => {
      const token = localStorage.getItem('token');
      
      if (!token) {
        setAuthState(prev => ({ ...prev, isLoading: false }));
        return;
      }
      
      try {
        // In development mode, just set isAuthenticated to true if token exists
        if (process.env.NODE_ENV === 'development') {
          setAuthState({
            isAuthenticated: true,
            user: { 
              id: '1', 
              username: localStorage.getItem('username') || 'testuser',
              email: localStorage.getItem('email') || 'testuser@example.com',
              role: 'user'
            },
            token,
            isLoading: false,
            error: null
          });
          return;
        }

        // In production, validate token with backend
        // Use /auth/me to validate token and get user info
        const response = await axios.get(`${API_URL}/auth/me`, { headers: { Authorization: `Bearer ${token}` } });
        
        setAuthState({
          user: response.data.user,
          token,
          isAuthenticated: true,
          isLoading: false,
          error: null
        });
      } catch (error) {
        localStorage.removeItem('token');
        setAuthState({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
          error: 'Session expired, please login again'
        });
      }
    };

    initializeAuth();
  }, []);

  // Refresh token periodically
  const refreshToken = useCallback(async () => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const response = await axios.post(`${API_URL}/auth/refresh`, {}, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        
        localStorage.setItem('token', response.data.token);
        setAuthState(prev => ({
          ...prev,
          token: response.data.token
        }));
      } catch (error) {
        // If refresh fails, log the user out
        logout();
      }
    }
  }, [logout]);

  // Set up token refresh interval
  useEffect(() => {
    // Only set up refresh in production
    if (process.env.NODE_ENV !== 'development' && authState.isAuthenticated) {
      const interval = setInterval(refreshToken, 15 * 60 * 1000); // Refresh every 15 minutes
      return () => clearInterval(interval);
    }
  }, [authState.isAuthenticated, refreshToken]);

  // Login function
  const login = async (email: string, password: string) => {
    setAuthState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      // Mock authentication for development
      if (process.env.NODE_ENV === 'development') {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Mock credentials check
        if ((email === 'testuser@example.com' && password === 'Password123!') || 
            (email === 'admin' && password === 'admin123')) {
          const mockToken = 'mock-jwt-token-for-development';
          const mockUser = {
            id: '1', 
            username: email === 'admin' ? 'admin' : 'testuser',
            email,
            role: email === 'admin' ? 'admin' : 'user'
          };
          
          localStorage.setItem('token', mockToken);
          localStorage.setItem('username', mockUser.username);
          localStorage.setItem('email', mockUser.email);
          
          setAuthState({
            user: mockUser,
            token: mockToken,
            isAuthenticated: true,
            isLoading: false,
            error: null
          });
          return;
        } else {
          throw new Error('Invalid credentials');
        }
      }
      
      // Real authentication for production
      const response = await axios.post(`${API_URL}/auth/login`, { email, password });
      
      localStorage.setItem('token', response.data.token);
      
      setAuthState({
        user: response.data.user,
        token: response.data.token,
        isAuthenticated: true,
        isLoading: false,
        error: null
      });
    } catch (error) {
      setAuthState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Login failed. Please try again.'
      }));
    }
  };

  // Register function
  const register = async (username: string, email: string, password: string) => {
    setAuthState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      // Mock registration for development
      if (process.env.NODE_ENV === 'development') {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const mockToken = 'mock-jwt-token-for-development';
        const mockUser = {
          id: '1', 
          username,
          email,
          role: 'user'
        };
        
        localStorage.setItem('token', mockToken);
        localStorage.setItem('username', mockUser.username);
        localStorage.setItem('email', mockUser.email);
        
        setAuthState({
          user: mockUser,
          token: mockToken,
          isAuthenticated: true,
          isLoading: false,
          error: null
        });
        return;
      }
      
      // Real registration for production
      const response = await axios.post(`${API_URL}/auth/register`, { username, email, password });
      
      localStorage.setItem('token', response.data.token);
      
      setAuthState({
        user: response.data.user,
        token: response.data.token,
        isAuthenticated: true,
        isLoading: false,
        error: null
      });
    } catch (error) {
      setAuthState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Registration failed. Please try again.'
      }));
    }
  };

  // Clear error
  const clearError = useCallback(() => {
    setAuthState(prev => ({ ...prev, error: null }));
  }, []);

  const contextValue = React.useMemo(() => ({
    authState,
    login,
    register,
    logout,
    clearError
  }), [authState, login, register, logout, clearError]);

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = () => useContext(AuthContext);
