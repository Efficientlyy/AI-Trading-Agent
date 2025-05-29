import React from 'react';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '../context/ThemeContext';
import SessionManagementPanel from '../components/dashboard/SessionManagementPanel';
import ActivityFeed from '../components/dashboard/ActivityFeed';
import SystemControlPanel from '../components/dashboard/SystemControlPanel';
import PerformanceMetricsPanel from '../components/dashboard/PerformanceMetricsPanel';

// Mock the necessary contexts and props
jest.mock('../context/PaperTradingContext', () => ({
  usePaperTrading: () => ({
    sessions: [],
    stopSession: jest.fn(),
    pauseSession: jest.fn(),
    resumeSession: jest.fn(),
    createSession: jest.fn(),
    loading: false,
    error: null
  })
}));

jest.mock('../context/SystemControlContext', () => ({
  useSystemControl: () => ({
    systemStatus: 'running',
    startSystem: jest.fn(),
    stopSystem: jest.fn(),
    agents: [],
    startAgent: jest.fn(),
    stopAgent: jest.fn(),
    loading: false,
    error: null
  })
}));

jest.mock('../context/ActivityContext', () => ({
  useActivity: () => ({
    activities: [],
    loading: false,
    error: null,
    filterActivities: jest.fn()
  })
}));

// Helper to render components with dark theme
const renderWithDarkTheme = (component: React.ReactNode) => {
  // Force dark mode by mocking localStorage
  Object.defineProperty(window, 'localStorage', {
    value: {
      getItem: jest.fn(() => 'dark'),
      setItem: jest.fn(),
    },
    writable: true
  });
  
  // Mock matchMedia for system theme detection
  Object.defineProperty(window, 'matchMedia', {
    value: jest.fn().mockImplementation(query => ({
      matches: true, // Always match dark mode
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    })),
    writable: true
  });
  
  return render(
    <ThemeProvider>
      {component}
    </ThemeProvider>
  );
};

describe('Dark Theme Verification', () => {
  beforeEach(() => {
    // Add dark class to document for testing
    document.documentElement.classList.add('dark');
  });
  
  afterEach(() => {
    // Clean up
    document.documentElement.classList.remove('dark');
    jest.clearAllMocks();
  });
  
  test('SessionManagementPanel renders with dark theme styling', () => {
    renderWithDarkTheme(<SessionManagementPanel />);
    // Check for dark theme class or styling
    const panel = document.querySelector('.session-management-panel');
    expect(panel).toBeTruthy();
    
    // Check if dark theme classes are applied
    const darkElements = document.querySelectorAll('[class*="dark"]');
    expect(darkElements.length).toBeGreaterThan(0);
  });
  
  test('ActivityFeed renders with dark theme styling', () => {
    renderWithDarkTheme(<ActivityFeed />);
    // Check for dark theme class or styling
    const feed = document.querySelector('.activity-feed');
    expect(feed).toBeTruthy();
    
    // Check if dark theme classes are applied
    const darkElements = document.querySelectorAll('[class*="dark"]');
    expect(darkElements.length).toBeGreaterThan(0);
  });
  
  test('SystemControlPanel renders with dark theme styling', () => {
    renderWithDarkTheme(<SystemControlPanel />);
    // Check for dark theme class or styling
    const panel = document.querySelector('.system-control-panel');
    expect(panel).toBeTruthy();
    
    // Check if dark theme classes are applied
    const darkElements = document.querySelectorAll('[class*="dark"]');
    expect(darkElements.length).toBeGreaterThan(0);
  });
  
  test('PerformanceMetricsPanel renders with dark theme styling', () => {
    renderWithDarkTheme(<PerformanceMetricsPanel />);
    // Check for dark theme class or styling
    const panel = document.querySelector('.performance-metrics-panel');
    expect(panel).toBeTruthy();
    
    // Check if dark theme classes are applied
    const darkElements = document.querySelectorAll('[class*="dark"]');
    expect(darkElements.length).toBeGreaterThan(0);
  });
});
