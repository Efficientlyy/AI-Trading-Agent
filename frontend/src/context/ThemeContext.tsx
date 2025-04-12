import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Define theme types
export type ThemeMode = 'light' | 'dark' | 'system';
export type ThemeAccent = 'blue' | 'green' | 'purple' | 'orange' | 'red';

interface ThemeContextType {
  mode: ThemeMode;
  accent: ThemeAccent;
  setMode: (mode: ThemeMode) => void;
  setAccent: (accent: ThemeAccent) => void;
  toggleMode: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Default values
const DEFAULT_MODE: ThemeMode = 'system';
const DEFAULT_ACCENT: ThemeAccent = 'blue';

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider = ({ children }: ThemeProviderProps) => {
  // Initialize state from localStorage or defaults
  const [mode, setModeState] = useState<ThemeMode>(() => {
    const savedMode = localStorage.getItem('theme-mode');
    return (savedMode as ThemeMode) || DEFAULT_MODE;
  });
  
  const [accent, setAccentState] = useState<ThemeAccent>(() => {
    const savedAccent = localStorage.getItem('theme-accent');
    return (savedAccent as ThemeAccent) || DEFAULT_ACCENT;
  });

  // Update localStorage when theme changes
  useEffect(() => {
    localStorage.setItem('theme-mode', mode);
    
    // Apply theme to document
    const isDark = mode === 'dark' || (mode === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.classList.toggle('dark', isDark);
    
    // Listen for system theme changes if in system mode
    if (mode === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const handleChange = (e: MediaQueryListEvent) => {
        document.documentElement.classList.toggle('dark', e.matches);
      };
      
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
  }, [mode]);

  // Update accent color
  useEffect(() => {
    localStorage.setItem('theme-accent', accent);
    document.documentElement.setAttribute('data-accent', accent);
  }, [accent]);

  // Toggle between light and dark (skipping system)
  const toggleMode = () => {
    setModeState(prev => prev === 'light' ? 'dark' : 'light');
  };

  // Wrapper functions to update state
  const setMode = (newMode: ThemeMode) => setModeState(newMode);
  const setAccent = (newAccent: ThemeAccent) => setAccentState(newAccent);

  return (
    <ThemeContext.Provider value={{ mode, accent, setMode, setAccent, toggleMode }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Custom hook for using the theme context
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
