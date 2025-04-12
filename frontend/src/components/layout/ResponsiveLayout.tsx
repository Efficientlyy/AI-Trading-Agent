import React, { useState, useEffect } from 'react';

// Define breakpoints for responsive design
export enum Breakpoint {
  XS = 'xs',   // Extra small devices (phones, less than 576px)
  SM = 'sm',   // Small devices (landscape phones, 576px and up)
  MD = 'md',   // Medium devices (tablets, 768px and up)
  LG = 'lg',   // Large devices (desktops, 992px and up)
  XL = 'xl',   // Extra large devices (large desktops, 1200px and up)
  XXL = 'xxl'  // Extra extra large devices (larger desktops, 1400px and up)
}

// Breakpoint pixel values
const breakpointValues = {
  [Breakpoint.XS]: 0,
  [Breakpoint.SM]: 576,
  [Breakpoint.MD]: 768,
  [Breakpoint.LG]: 992,
  [Breakpoint.XL]: 1200,
  [Breakpoint.XXL]: 1400
};

// Hook to get current breakpoint
export function useBreakpoint() {
  const [breakpoint, setBreakpoint] = useState<Breakpoint>(() => {
    // Initial determination based on window width
    const width = window.innerWidth;
    if (width >= breakpointValues[Breakpoint.XXL]) return Breakpoint.XXL;
    if (width >= breakpointValues[Breakpoint.XL]) return Breakpoint.XL;
    if (width >= breakpointValues[Breakpoint.LG]) return Breakpoint.LG;
    if (width >= breakpointValues[Breakpoint.MD]) return Breakpoint.MD;
    if (width >= breakpointValues[Breakpoint.SM]) return Breakpoint.SM;
    return Breakpoint.XS;
  });
  
  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      if (width >= breakpointValues[Breakpoint.XXL]) setBreakpoint(Breakpoint.XXL);
      else if (width >= breakpointValues[Breakpoint.XL]) setBreakpoint(Breakpoint.XL);
      else if (width >= breakpointValues[Breakpoint.LG]) setBreakpoint(Breakpoint.LG);
      else if (width >= breakpointValues[Breakpoint.MD]) setBreakpoint(Breakpoint.MD);
      else if (width >= breakpointValues[Breakpoint.SM]) setBreakpoint(Breakpoint.SM);
      else setBreakpoint(Breakpoint.XS);
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  return breakpoint;
}

// Helper functions to check breakpoints
export function isBreakpointUp(currentBreakpoint: Breakpoint, minBreakpoint: Breakpoint): boolean {
  return breakpointValues[currentBreakpoint] >= breakpointValues[minBreakpoint];
}

export function isBreakpointDown(currentBreakpoint: Breakpoint, maxBreakpoint: Breakpoint): boolean {
  return breakpointValues[currentBreakpoint] <= breakpointValues[maxBreakpoint];
}

// Component that renders different content based on breakpoint
interface ResponsiveProps {
  children: React.ReactNode;
  breakpointUp?: Breakpoint;
  breakpointDown?: Breakpoint;
}

export const Responsive: React.FC<ResponsiveProps> = ({ 
  children, 
  breakpointUp, 
  breakpointDown 
}) => {
  const currentBreakpoint = useBreakpoint();
  
  // Determine if component should render based on current breakpoint
  const shouldRender = () => {
    if (breakpointUp && !isBreakpointUp(currentBreakpoint, breakpointUp)) {
      return false;
    }
    
    if (breakpointDown && !isBreakpointDown(currentBreakpoint, breakpointDown)) {
      return false;
    }
    
    return true;
  };
  
  return shouldRender() ? <>{children}</> : null;
};

// Grid component for responsive layouts
interface GridProps {
  children: React.ReactNode;
  columns?: {
    xs?: number;
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
    xxl?: number;
  };
  gap?: string;
  className?: string;
}

export const ResponsiveGrid: React.FC<GridProps> = ({ 
  children, 
  columns = { xs: 1, sm: 2, md: 3, lg: 4, xl: 4, xxl: 6 }, 
  gap = '1rem',
  className = ''
}) => {
  const breakpoint = useBreakpoint();
  
  // Determine number of columns based on current breakpoint
  const getColumns = () => {
    switch (breakpoint) {
      case Breakpoint.XXL:
        return columns.xxl || columns.xl || columns.lg || columns.md || columns.sm || columns.xs || 6;
      case Breakpoint.XL:
        return columns.xl || columns.lg || columns.md || columns.sm || columns.xs || 4;
      case Breakpoint.LG:
        return columns.lg || columns.md || columns.sm || columns.xs || 4;
      case Breakpoint.MD:
        return columns.md || columns.sm || columns.xs || 3;
      case Breakpoint.SM:
        return columns.sm || columns.xs || 2;
      case Breakpoint.XS:
      default:
        return columns.xs || 1;
    }
  };
  
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: `repeat(${getColumns()}, 1fr)`,
    gap
  };
  
  return (
    <div style={gridStyle} className={className}>
      {children}
    </div>
  );
};

// Container component with responsive padding/margins
interface ContainerProps {
  children: React.ReactNode;
  fluid?: boolean;
  className?: string;
}

export const Container: React.FC<ContainerProps> = ({ 
  children, 
  fluid = false,
  className = ''
}) => {
  const containerClass = fluid 
    ? 'w-full px-4' 
    : 'w-full px-4 mx-auto max-w-7xl';
    
  return (
    <div className={`${containerClass} ${className}`}>
      {children}
    </div>
  );
};

export default { Responsive, ResponsiveGrid, Container, useBreakpoint };
