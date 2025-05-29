/**
 * API Client for the AI Trading Agent application
 * Centralizes API request handling with consistent error handling and authentication
 */
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

// Base URL for the API
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000, // 15 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding authentication token
apiClient.interceptors.request.use(
  (config: AxiosRequestConfig) => {
    const token = localStorage.getItem('auth_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle specific error cases
    if (error.response) {
      // Server responded with error status
      console.error('API Error Response:', error.response.status, error.response.data);
      
      // Handle authentication errors
      if (error.response.status === 401) {
        // Clear token and redirect to login if unauthorized
        localStorage.removeItem('auth_token');
        // We could redirect to login page here if needed
      }
    } else if (error.request) {
      // Request was made but no response received
      console.error('API No Response Error:', error.request);
    } else {
      // Error in setting up the request
      console.error('API Configuration Error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

export default apiClient;
