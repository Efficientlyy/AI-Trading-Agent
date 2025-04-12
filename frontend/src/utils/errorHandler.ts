import React from 'react';
import axios, { AxiosError } from 'axios';

// Define error types
export enum ErrorType {
  NETWORK = 'network',
  AUTHENTICATION = 'authentication',
  AUTHORIZATION = 'authorization',
  VALIDATION = 'validation',
  SERVER = 'server',
  NOT_FOUND = 'not_found',
  TIMEOUT = 'timeout',
  UNKNOWN = 'unknown'
}

// Error response structure
export interface ErrorResponse {
  type: ErrorType;
  message: string;
  details?: string | Record<string, any>;
  statusCode?: number;
  path?: string;
}

// Default error messages
const DEFAULT_ERROR_MESSAGES: Record<ErrorType, string> = {
  [ErrorType.NETWORK]: 'Network connection error. Please check your internet connection.',
  [ErrorType.AUTHENTICATION]: 'Authentication failed. Please log in again.',
  [ErrorType.AUTHORIZATION]: 'You do not have permission to perform this action.',
  [ErrorType.VALIDATION]: 'Invalid data provided. Please check your inputs.',
  [ErrorType.SERVER]: 'Server error occurred. Please try again later.',
  [ErrorType.NOT_FOUND]: 'The requested resource was not found.',
  [ErrorType.TIMEOUT]: 'Request timed out. Please try again.',
  [ErrorType.UNKNOWN]: 'An unexpected error occurred. Please try again.'
};

/**
 * Parses an error from any source into a standardized ErrorResponse
 */
export function parseError(error: any): ErrorResponse {
  // Handle Axios errors
  if (axios.isAxiosError(error)) {
    return parseAxiosError(error);
  }
  
  // Handle standard Error objects
  if (error instanceof Error) {
    return {
      type: ErrorType.UNKNOWN,
      message: error.message || DEFAULT_ERROR_MESSAGES[ErrorType.UNKNOWN],
      details: error.stack
    };
  }
  
  // Handle string errors
  if (typeof error === 'string') {
    return {
      type: ErrorType.UNKNOWN,
      message: error
    };
  }
  
  // Handle unknown errors
  return {
    type: ErrorType.UNKNOWN,
    message: DEFAULT_ERROR_MESSAGES[ErrorType.UNKNOWN],
    details: JSON.stringify(error)
  };
}

/**
 * Parse Axios errors into standardized format
 */
function parseAxiosError(error: AxiosError): ErrorResponse {
  const statusCode = error.response?.status;
  const responseData = error.response?.data as any;
  
  // Network errors (no response)
  if (!error.response) {
    return {
      type: ErrorType.NETWORK,
      message: DEFAULT_ERROR_MESSAGES[ErrorType.NETWORK],
      details: error.message
    };
  }
  
  // Map status codes to error types
  let errorType: ErrorType;
  switch (statusCode) {
    case 401:
      errorType = ErrorType.AUTHENTICATION;
      break;
    case 403:
      errorType = ErrorType.AUTHORIZATION;
      break;
    case 404:
      errorType = ErrorType.NOT_FOUND;
      break;
    case 422:
      errorType = ErrorType.VALIDATION;
      break;
    case 408:
      errorType = ErrorType.TIMEOUT;
      break;
    case 500:
    case 502:
    case 503:
    case 504:
      errorType = ErrorType.SERVER;
      break;
    default:
      errorType = ErrorType.UNKNOWN;
  }
  
  // Extract error message from response if available
  let message = DEFAULT_ERROR_MESSAGES[errorType];
  let details = undefined;
  
  if (responseData) {
    if (typeof responseData === 'string') {
      message = responseData;
    } else if (responseData.message) {
      message = responseData.message;
      details = responseData.details || responseData.error;
    } else if (responseData.error) {
      message = typeof responseData.error === 'string' 
        ? responseData.error 
        : DEFAULT_ERROR_MESSAGES[errorType];
      details = responseData;
    }
  }
  
  return {
    type: errorType,
    message,
    details,
    statusCode,
    path: error.config?.url
  };
}

/**
 * Log errors to console in development and potentially to a monitoring service in production
 */
export function logError(error: ErrorResponse): void {
  // Always log to console in development
  if (process.env.NODE_ENV === 'development') {
    console.error('Error:', error);
  }
  
  // In production, you could send to a monitoring service like Sentry
  if (process.env.NODE_ENV === 'production') {
    // Example: Sentry.captureException(error);
    // For now, just log critical errors
    if (
      error.type === ErrorType.SERVER || 
      error.type === ErrorType.UNKNOWN
    ) {
      console.error('Critical Error:', error);
    }
  }
}

/**
 * Handle specific error types with appropriate actions
 */
export function handleError(error: ErrorResponse): void {
  logError(error);
  
  // Handle authentication errors (redirect to login)
  if (error.type === ErrorType.AUTHENTICATION) {
    // Clear auth tokens
    localStorage.removeItem('auth-token');
    
    // Redirect to login page if not already there
    if (!window.location.pathname.includes('/login')) {
      window.location.href = '/login?session=expired';
    }
  }
  
  // Other specific error handling can be added here
}

/**
 * Create an error boundary component wrapper
 */
export function withErrorBoundary<P>(
  Component: React.ComponentType<P>,
  FallbackComponent: React.ComponentType<{ error: Error }>
): React.ComponentType<P> {
  return class ErrorBoundary extends React.Component<P, { hasError: boolean; error: Error | null }> {
    constructor(props: P) {
      super(props);
      this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error: Error) {
      return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
      logError({
        type: ErrorType.UNKNOWN,
        message: error.message,
        details: errorInfo
      });
    }

    render() {
      if (this.state.hasError && this.state.error) {
        return <FallbackComponent error={this.state.error} />;
      }

      return <Component {...this.props} />;
    }
  };
}
