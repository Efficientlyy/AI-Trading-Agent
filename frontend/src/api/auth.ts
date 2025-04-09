import { apiClient, createAuthenticatedClient } from './client';

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterCredentials extends LoginCredentials {
  email?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

export const authApi = {
  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    const formData = new FormData();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);
    
    const response = await apiClient.post<AuthResponse>('/auth/login', formData);
    return response.data;
  },
  
  register: async (credentials: RegisterCredentials): Promise<{ msg: string }> => {
    const response = await apiClient.post<{ msg: string }>('/auth/register', null, {
      params: {
        username: credentials.username,
        password: credentials.password,
      }
    });
    return response.data;
  },
  
  refreshToken: async (token: string): Promise<AuthResponse> => {
    const response = await apiClient.post<AuthResponse>('/auth/refresh', { token });
    return response.data;
  },
  
  resetPassword: async (username: string, newPassword: string): Promise<{ msg: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.post<{ msg: string }>('/auth/reset-password', null, {
      params: {
        username,
        new_password: newPassword,
      }
    });
    return response.data;
  },
};
