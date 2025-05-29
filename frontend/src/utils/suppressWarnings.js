// This file contains utilities to suppress specific React warnings that can't be fixed directly

// Suppress the defaultProps warning for function components
// This is particularly useful for third-party libraries that haven't updated yet
export const suppressDefaultPropsWarning = () => {
  // Store the original console.error
  const originalConsoleError = console.error;
  
  // Override console.error to filter out specific warnings
  console.error = function(...args) {
    // Check if this is the defaultProps warning
    if (args[0] && typeof args[0] === 'string' && 
        args[0].includes('Support for defaultProps will be removed from function components')) {
      // Don't log this warning
      return;
    }
    
    // Call the original console.error for all other cases
    return originalConsoleError.apply(console, args);
  };
  
  // Return a function to restore the original console.error if needed
  return () => {
    console.error = originalConsoleError;
  };
};
