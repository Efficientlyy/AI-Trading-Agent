import React, { useState, useRef } from 'react';
import { CSSTransition, TransitionGroup } from 'react-transition-group';

// Animation types
export enum AnimationType {
  FADE = 'fade',
  SLIDE_UP = 'slide-up',
  SLIDE_DOWN = 'slide-down',
  SLIDE_LEFT = 'slide-left',
  SLIDE_RIGHT = 'slide-right',
  ZOOM = 'zoom',
  NONE = 'none'
}

// Animation durations
export enum AnimationDuration {
  FAST = 150,
  NORMAL = 250,
  SLOW = 350
}

interface AnimatedTransitionProps {
  children: React.ReactNode;
  type?: AnimationType;
  duration?: AnimationDuration;
  className?: string;
  show?: boolean;
  unmountOnExit?: boolean;
}

// Component for single element animations
export const AnimatedTransition: React.FC<AnimatedTransitionProps> = ({
  children,
  type = AnimationType.FADE,
  duration = AnimationDuration.NORMAL,
  className = '',
  show = true,
  unmountOnExit = true
}) => {
  const nodeRef = useRef<HTMLDivElement | null>(null);
  
  // Generate CSS classes based on animation type
  const getClassNames = () => {
    switch (type) {
      case AnimationType.FADE:
        return {
          enter: 'opacity-0',
          enterActive: 'opacity-100 transition-opacity',
          exit: 'opacity-100',
          exitActive: 'opacity-0 transition-opacity'
        };
      case AnimationType.SLIDE_UP:
        return {
          enter: 'opacity-0 transform translate-y-4',
          enterActive: 'opacity-100 transform translate-y-0 transition-all',
          exit: 'opacity-100 transform translate-y-0',
          exitActive: 'opacity-0 transform translate-y-4 transition-all'
        };
      case AnimationType.SLIDE_DOWN:
        return {
          enter: 'opacity-0 transform -translate-y-4',
          enterActive: 'opacity-100 transform translate-y-0 transition-all',
          exit: 'opacity-100 transform translate-y-0',
          exitActive: 'opacity-0 transform -translate-y-4 transition-all'
        };
      case AnimationType.SLIDE_LEFT:
        return {
          enter: 'opacity-0 transform translate-x-4',
          enterActive: 'opacity-100 transform translate-x-0 transition-all',
          exit: 'opacity-100 transform translate-x-0',
          exitActive: 'opacity-0 transform translate-x-4 transition-all'
        };
      case AnimationType.SLIDE_RIGHT:
        return {
          enter: 'opacity-0 transform -translate-x-4',
          enterActive: 'opacity-100 transform translate-x-0 transition-all',
          exit: 'opacity-100 transform translate-x-0',
          exitActive: 'opacity-0 transform -translate-x-4 transition-all'
        };
      case AnimationType.ZOOM:
        return {
          enter: 'opacity-0 transform scale-95',
          enterActive: 'opacity-100 transform scale-100 transition-all',
          exit: 'opacity-100 transform scale-100',
          exitActive: 'opacity-0 transform scale-95 transition-all'
        };
      case AnimationType.NONE:
      default:
        return {
          enter: '',
          enterActive: '',
          exit: '',
          exitActive: ''
        };
    }
  };
  
  const classNames = getClassNames();
  
  return (
    <CSSTransition
      in={show}
      nodeRef={nodeRef as React.RefObject<HTMLElement>}
      timeout={duration}
      classNames={classNames}
      unmountOnExit={unmountOnExit}
    >
      <div ref={nodeRef} className={className} style={{ transitionDuration: `${duration}ms` }}>
        {children}
      </div>
    </CSSTransition>
  );
};

// Component for animating between different elements/pages
interface AnimatedSwitchProps {
  children: React.ReactNode;
  type?: AnimationType;
  duration?: AnimationDuration;
  className?: string;
}

export const AnimatedSwitch: React.FC<AnimatedSwitchProps> = ({
  children,
  type = AnimationType.FADE,
  duration = AnimationDuration.NORMAL,
  className = ''
}) => {
  // Use React.Children.map to add keys if they don't exist
  const childrenWithKeys = React.Children.map(children, (child, index) => {
    if (React.isValidElement(child)) {
      return React.cloneElement(child, { key: child.key || `animated-child-${index}` });
    }
    return child;
  });
  
  // Generate CSS classes based on animation type
  const getClassNames = () => {
    switch (type) {
      case AnimationType.FADE:
        return {
          enter: 'opacity-0',
          enterActive: 'opacity-100 transition-opacity',
          exit: 'opacity-100',
          exitActive: 'opacity-0 transition-opacity'
        };
      case AnimationType.SLIDE_UP:
        return {
          enter: 'opacity-0 transform translate-y-4',
          enterActive: 'opacity-100 transform translate-y-0 transition-all',
          exit: 'opacity-100 transform translate-y-0',
          exitActive: 'opacity-0 transform translate-y-4 transition-all'
        };
      case AnimationType.SLIDE_DOWN:
        return {
          enter: 'opacity-0 transform -translate-y-4',
          enterActive: 'opacity-100 transform translate-y-0 transition-all',
          exit: 'opacity-100 transform translate-y-0',
          exitActive: 'opacity-0 transform -translate-y-4 transition-all'
        };
      case AnimationType.SLIDE_LEFT:
        return {
          enter: 'opacity-0 transform translate-x-4',
          enterActive: 'opacity-100 transform translate-x-0 transition-all',
          exit: 'opacity-100 transform translate-x-0',
          exitActive: 'opacity-0 transform translate-x-4 transition-all'
        };
      case AnimationType.SLIDE_RIGHT:
        return {
          enter: 'opacity-0 transform -translate-x-4',
          enterActive: 'opacity-100 transform translate-x-0 transition-all',
          exit: 'opacity-100 transform translate-x-0',
          exitActive: 'opacity-0 transform -translate-x-4 transition-all'
        };
      case AnimationType.ZOOM:
        return {
          enter: 'opacity-0 transform scale-95',
          enterActive: 'opacity-100 transform scale-100 transition-all',
          exit: 'opacity-100 transform scale-100',
          exitActive: 'opacity-0 transform scale-95 transition-all'
        };
      case AnimationType.NONE:
      default:
        return {
          enter: '',
          enterActive: '',
          exit: '',
          exitActive: ''
        };
    }
  };
  
  return (
    <TransitionGroup className={className}>
      {React.Children.map(childrenWithKeys, (child) => {
        if (React.isValidElement(child)) {
          const nodeRef = React.createRef<HTMLDivElement>();
          
          return (
            <CSSTransition
              key={child.key}
              nodeRef={nodeRef as React.RefObject<HTMLElement>}
              timeout={duration}
              classNames={getClassNames()}
            >
              <div ref={nodeRef} style={{ transitionDuration: `${duration}ms` }}>
                {child}
              </div>
            </CSSTransition>
          );
        }
        return child;
      })}
    </TransitionGroup>
  );
};

// Hook for triggering animations
export function useAnimation(
  initialState = false,
  type = AnimationType.FADE,
  duration = AnimationDuration.NORMAL
) {
  const [isVisible, setIsVisible] = useState(initialState);
  const [hasAnimatedIn, setHasAnimatedIn] = useState(initialState);
  
  const show = () => {
    setIsVisible(true);
    setHasAnimatedIn(true);
  };
  
  const hide = () => {
    setIsVisible(false);
  };
  
  const toggle = () => {
    setIsVisible(prev => !prev);
    if (!hasAnimatedIn) setHasAnimatedIn(true);
  };
  
  return {
    isVisible,
    hasAnimatedIn,
    show,
    hide,
    toggle,
    animationProps: {
      type,
      duration,
      show: isVisible
    }
  };
}


