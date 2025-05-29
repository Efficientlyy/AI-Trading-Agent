import React, { useState, useRef } from 'react';
import { CSSTransition, TransitionGroup } from 'react-transition-group';
import { Location } from 'react-router-dom';

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

// Hoisted getClassNames function
const getClassNames = (type: AnimationType) => {
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
        exitActive: 'opacity-0 transform -translate-x-4 transition-all'
      };
    case AnimationType.SLIDE_RIGHT:
      return {
        enter: 'opacity-0 transform -translate-x-4',
        enterActive: 'opacity-100 transform translate-x-0 transition-all',
        exit: 'opacity-100 transform translate-x-0',
        exitActive: 'opacity-0 transform translate-x-4 transition-all'
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
  
  return (
    <CSSTransition
      in={show}
      nodeRef={nodeRef as React.RefObject<HTMLElement>}
      timeout={duration}
      classNames={getClassNames(type)}
      unmountOnExit={unmountOnExit}
    >
      <div ref={nodeRef} className={className} style={{ transitionDuration: `${duration}ms` }}>
        {children}
      </div>
    </CSSTransition>
  );
};

interface AnimatedSwitchProps {
  children: React.ReactNode;
  animationType?: AnimationType;
  animationDuration?: AnimationDuration;
  className?: string;
  location: Location;
}

// Component for animating transitions between different elements/pages (routes)
export const AnimatedSwitch: React.FC<AnimatedSwitchProps> = ({
  children, 
  animationType = AnimationType.FADE,
  animationDuration = AnimationDuration.NORMAL,
  className = '',
  location
}) => {
  const nodeRef = useRef<HTMLDivElement | null>(null); // useRef for CSSTransition's nodeRef

  return (
    <TransitionGroup className={className} component={null}> {/* component={null} avoids an extra div from TransitionGroup */}
      <CSSTransition
        key={location.pathname} // Key the transition on the path. This is crucial.
        nodeRef={nodeRef}       // Pass the ref to CSSTransition.
        timeout={animationDuration}
        classNames={getClassNames(animationType)}
        unmountOnExit           // Unmount the old route's component after animation.
        appear                  // Allow animation on initial mount if desired.
      >
        {/* CSSTransition expects a single child that can accept a ref.
            We wrap the 'children' (which is <Routes>) in a div that takes the nodeRef.
        */}
        <div 
          ref={nodeRef}
          style={{ width: '100%', height: '100%', position: 'relative' }} // Added styling
        >
          {children}
        </div>
      </CSSTransition>
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
