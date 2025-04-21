import { useRef } from 'react';

/**
 * Logs a message to the console every time the component renders.
 * Optionally logs the props or any dependencies.
 * Only logs in development mode.
 *
 * @param name - Name of the component for logging
 * @param deps - Optional dependencies to log (e.g., props)
 */
export function useRenderLogger(name: string, deps?: unknown) {
  const renderCount = useRef(1);
  if (process.env.NODE_ENV === 'development') {
    // eslint-disable-next-line no-console
    console.log(`%c[Render] ${name} (render #${renderCount.current})`, 'color: #1e90ff; font-weight: bold;', deps);
    renderCount.current += 1;
  }
}
