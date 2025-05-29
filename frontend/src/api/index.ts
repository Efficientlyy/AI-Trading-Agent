// Export all API modules for easier imports
export * from './alphaVantage';
export * from './apiClient';
export * from './apiUtils';
export * from './auth';
export * from './backtest';
export * from './backtestApi';
export * from './client';
export * from './market';
export * from './orders';
export * from './paperTrading';
export * from './performance';
export * from './portfolio';
// Import and re-export sentiment modules with explicit naming to avoid conflicts
import * as sentimentModule from './sentiment';
import * as sentimentApiModule from './sentimentApi';
export { sentimentModule, sentimentApiModule };
export * from './strategies';
export * from './systemControl';
export * from './trades';
