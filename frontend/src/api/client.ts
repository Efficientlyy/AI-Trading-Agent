import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

// Create a base API client
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const createApiClient = (token?: string): AxiosInstance => {
  const config: AxiosRequestConfig = {
    baseURL: API_URL,
    headers: {
      'Content-Type': 'application/json',
    },
  };

  if (token) {
    config.headers = {
      ...config.headers,
      Authorization: `Bearer ${token}`,
    };
  }

  const client = axios.create(config);

  // Add a response interceptor for error handling
  client.interceptors.response.use(
    (response) => response,
    (error) => {
      // Handle token expiration
      if (error.response && error.response.status === 401) {
        // Clear local storage and redirect to login
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
      return Promise.reject(error);
    }
  );

  return client;
};

// Create a default client without authentication
export const apiClient = createApiClient();

// Create an authenticated client with the token
export const createAuthenticatedClient = (): AxiosInstance => {
  const token = localStorage.getItem('token');
  return token ? createApiClient(token) : apiClient;
};
