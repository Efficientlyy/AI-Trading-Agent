/**
 * Simple client-side cache utility for storing and retrieving data with expiration
 */

interface CacheItem<T> {
  data: T;
  timestamp: number;
  expiry: number; // Expiry time in milliseconds
}

class ClientCache {
  private static instance: ClientCache;
  private cache: Map<string, CacheItem<any>>;
  
  private constructor() {
    this.cache = new Map();
  }
  
  /**
   * Get singleton instance of ClientCache
   */
  public static getInstance(): ClientCache {
    if (!ClientCache.instance) {
      ClientCache.instance = new ClientCache();
    }
    return ClientCache.instance;
  }
  
  /**
   * Set data in cache with expiration
   * @param key Cache key
   * @param data Data to store
   * @param expiryMs Time in milliseconds until the cache item expires (default: 5 minutes)
   */
  public set<T>(key: string, data: T, expiryMs: number = 5 * 60 * 1000): void {
    const timestamp = Date.now();
    this.cache.set(key, {
      data,
      timestamp,
      expiry: expiryMs
    });
  }
  
  /**
   * Get data from cache if it exists and hasn't expired
   * @param key Cache key
   * @returns The cached data or null if not found or expired
   */
  public get<T>(key: string): T | null {
    const item = this.cache.get(key);
    
    if (!item) return null;
    
    const now = Date.now();
    const expired = now - item.timestamp > item.expiry;
    
    if (expired) {
      this.cache.delete(key);
      return null;
    }
    
    return item.data as T;
  }
  
  /**
   * Check if cache has a non-expired item for the given key
   * @param key Cache key
   * @returns True if item exists and has not expired
   */
  public has(key: string): boolean {
    const item = this.cache.get(key);
    
    if (!item) return false;
    
    const now = Date.now();
    const expired = now - item.timestamp > item.expiry;
    
    if (expired) {
      this.cache.delete(key);
      return false;
    }
    
    return true;
  }
  
  /**
   * Remove item from cache
   * @param key Cache key
   */
  public remove(key: string): void {
    this.cache.delete(key);
  }
  
  /**
   * Clear all items from cache
   */
  public clear(): void {
    this.cache.clear();
  }
  
  /**
   * Get cache size
   * @returns Number of items in cache
   */
  public size(): number {
    return this.cache.size;
  }
  
  /**
   * Remove all expired items from cache
   * @returns Number of items removed
   */
  public cleanup(): number {
    const now = Date.now();
    let count = 0;
    
    // Convert Map entries to array to avoid issues with downlevelIteration
    Array.from(this.cache.entries()).forEach(([key, item]) => {
      const expired = now - item.timestamp > item.expiry;
      if (expired) {
        this.cache.delete(key);
        count++;
      }
    });
    
    return count;
  }
}

export default ClientCache.getInstance();
