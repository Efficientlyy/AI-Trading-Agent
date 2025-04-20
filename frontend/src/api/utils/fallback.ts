import { recordApiCall } from './monitoring';
import { recordEnhancedApiCall } from './enhancedMonitoring';

// Define fallback status types
type FallbackStatus = 'fallback_attempt' | 'fallback_success' | 'fallback_failure' | 
                     'fallback_cache_hit' | 'fallback_internal_cache_hit' | 'fallback_secondary_success';

/**
 * Fallback options interface for tiered fallback strategy
 */
export interface FallbackOptions<T> {
  /** Primary fallback function */
  primary?: () => Promise<T>;
  /** Secondary fallback function (used if primary fails) */
  secondary?: () => Promise<T>;
  /** Cache-based fallback function */
  cache?: () => Promise<T | null>;
  /** Validation function to ensure fallback data meets requirements */
  validator?: (result: T) => boolean;
  /** Maximum age of cached data in milliseconds */
  maxCacheAgeMs?: number;
  /** Context information for smart fallback selection */
  context?: {
    /** Is this a critical operation that requires data even if stale? */
    isCritical?: boolean;
    /** Error that triggered the fallback */
    error?: Error;
    /** Exchange name */
    exchange: string;
    /** Method name */
    method: string;
  };
}

/**
 * Cache for storing fallback data
 */
interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

const fallbackCache: Record<string, CacheEntry<any>> = {};

/**
 * Store data in the fallback cache
 * @param key Cache key (typically exchange:method)
 * @param data Data to cache
 */
export const cacheForFallback = <T>(key: string, data: T): void => {
  fallbackCache[key] = {
    data,
    timestamp: Date.now(),
  };
};

/**
 * Get data from the fallback cache if it's not too old
 * @param key Cache key
 * @param maxAgeMs Maximum age in milliseconds
 * @returns Cached data or null if not found or too old
 */
export const getFromFallbackCache = <T>(key: string, maxAgeMs: number = 5 * 60 * 1000): T | null => {
  const entry = fallbackCache[key];
  if (!entry) return null;
  
  const age = Date.now() - entry.timestamp;
  if (age > maxAgeMs) return null;
  
  return entry.data;
};

/**
 * Execute tiered fallback strategy
 * @param options Fallback options
 * @returns Result from the first successful fallback
 * @throws Error if all fallbacks fail
 */
export const executeFallback = async <T>(options: FallbackOptions<T>): Promise<T> => {
  const { primary, secondary, cache, validator, maxCacheAgeMs = 5 * 60 * 1000, context } = options;
  
  // Log fallback attempt
  if (context) {
    console.warn(`Executing fallback for ${context.exchange} ${context.method}`);
    recordEnhancedApiCall(context.exchange, context.method, 'fallback_attempt');
  }
  
  // Try primary fallback
  if (primary) {
    try {
      const result = await primary();
      
      // Validate result if validator provided
      if (validator && !validator(result)) {
        throw new Error('Primary fallback data failed validation');
      }
      
      // Cache successful result for future fallbacks
      if (context) {
        const cacheKey = `${context.exchange}:${context.method}`;
        cacheForFallback(cacheKey, result);
        recordEnhancedApiCall(context.exchange, context.method, 'fallback_success');
      }
      
      return result;
    } catch (error) {
      console.error('Primary fallback failed:', error);
      // Continue to next fallback
    }
  }
  
  // Try cache fallback
  if (cache) {
    try {
      const cachedResult = await cache();
      
      if (cachedResult !== null) {
        // Validate cached result
        if (validator && !validator(cachedResult)) {
          throw new Error('Cached fallback data failed validation');
        }
        
        if (context) {
          recordEnhancedApiCall(context.exchange, context.method, 'fallback_cache_hit');
        }
        
        return cachedResult;
      }
    } catch (error) {
      console.error('Cache fallback failed:', error);
      // Continue to next fallback
    }
  }
  
  // Try from internal cache
  if (context) {
    const cacheKey = `${context.exchange}:${context.method}`;
    const cachedData = getFromFallbackCache<T>(cacheKey, maxCacheAgeMs);
    
    if (cachedData !== null) {
      // Validate cached result
      if (validator && !validator(cachedData)) {
        console.warn('Internal cached fallback data failed validation');
      } else {
        recordEnhancedApiCall(context.exchange, context.method, 'fallback_internal_cache_hit');
        return cachedData;
      }
    }
  }
  
  // Try secondary fallback
  if (secondary) {
    try {
      const result = await secondary();
      
      // Validate result if validator provided
      if (validator && !validator(result)) {
        throw new Error('Secondary fallback data failed validation');
      }
      
      // Cache successful result for future fallbacks
      if (context) {
        const cacheKey = `${context.exchange}:${context.method}`;
        cacheForFallback(cacheKey, result);
        recordEnhancedApiCall(context.exchange, context.method, 'fallback_secondary_success');
      }
      
      return result;
    } catch (error) {
      console.error('Secondary fallback failed:', error);
      // All fallbacks failed
    }
  }
  
  // All fallbacks failed
  if (context) {
    recordEnhancedApiCall(context.exchange, context.method, 'fallback_failure');
  }
  
  throw new Error('All fallbacks failed');
};

/**
 * Smart fallback selection based on error type and context
 * @param error Error that triggered the fallback
 * @param options Available fallback options
 * @returns Selected fallback strategy
 */
export const selectFallbackStrategy = <T>(
  error: Error,
  options: {
    networkError?: () => Promise<T>;
    timeoutError?: () => Promise<T>;
    authError?: () => Promise<T>;
    rateLimitError?: () => Promise<T>;
    serverError?: () => Promise<T>;
    defaultFallback?: () => Promise<T>;
  }
): (() => Promise<T>) | undefined => {
  // Check error type and select appropriate fallback
  if (error.name === 'NetworkError' || error.message.includes('network')) {
    return options.networkError || options.defaultFallback;
  }
  
  if (error.message.includes('timeout') || error.name === 'TimeoutError') {
    return options.timeoutError || options.defaultFallback;
  }
  
  if (error.message.includes('authentication') || error.message.includes('unauthorized')) {
    return options.authError || options.defaultFallback;
  }
  
  if (error.message.includes('rate limit') || error.message.includes('too many requests')) {
    return options.rateLimitError || options.defaultFallback;
  }
  
  if (error.message.includes('server error') || error.message.includes('500')) {
    return options.serverError || options.defaultFallback;
  }
  
  // Default fallback
  return options.defaultFallback;
};
