import { useCallback, useEffect, useState } from 'react';

const TOKEN_KEY = 'auth_token';

export const useAuthToken = () => {
    const [token, setTokenState] = useState<string | null>(() => {
        return localStorage.getItem(TOKEN_KEY);
    });

    const setToken = useCallback((newToken: string | null) => {
        setTokenState(newToken);

        if (newToken) {
            localStorage.setItem(TOKEN_KEY, newToken);
        } else {
            localStorage.removeItem(TOKEN_KEY);
        }
    }, []);

    const getToken = useCallback(() => {
        return localStorage.getItem(TOKEN_KEY);
    }, []);

    const clearToken = useCallback(() => {
        setTokenState(null);
        localStorage.removeItem(TOKEN_KEY);
    }, []);

    // Check token expiration
    useEffect(() => {
        const checkTokenExpiration = () => {
            const currentToken = getToken();
            if (currentToken) {
                try {
                    // JWT tokens are in the format: header.payload.signature
                    const payload = JSON.parse(atob(currentToken.split('.')[1]));
                    const expirationTime = payload.exp * 1000; // Convert to milliseconds

                    if (Date.now() >= expirationTime) {
                        // Token has expired
                        clearToken();
                    }
                } catch (error) {
                    console.error('Error checking token expiration:', error);
                    // If we can't parse the token, better to clear it
                    clearToken();
                }
            }
        };

        // Check on mount
        checkTokenExpiration();

        // Set up interval to check periodically
        const interval = setInterval(checkTokenExpiration, 60000); // Check every minute

        return () => clearInterval(interval);
    }, [clearToken, getToken]);

    return { token, setToken, getToken, clearToken };
};