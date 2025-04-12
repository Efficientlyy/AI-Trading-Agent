declare module 'react-transition-group' {
  import * as React from 'react';

  interface CSSTransitionClassNames {
    appear?: string;
    appearActive?: string;
    appearDone?: string;
    enter?: string;
    enterActive?: string;
    enterDone?: string;
    exit?: string;
    exitActive?: string;
    exitDone?: string;
  }

  interface CSSTransitionProps {
    in?: boolean;
    mountOnEnter?: boolean;
    unmountOnExit?: boolean;
    appear?: boolean;
    enter?: boolean;
    exit?: boolean;
    timeout: number | { enter?: number; exit?: number; appear?: number };
    classNames: string | CSSTransitionClassNames;
    children: React.ReactElement;
    nodeRef?: React.RefObject<HTMLElement>;
    addEndListener?: (done: () => void) => void;
    onEnter?: (node: HTMLElement, isAppearing: boolean) => void;
    onEntering?: (node: HTMLElement, isAppearing: boolean) => void;
    onEntered?: (node: HTMLElement, isAppearing: boolean) => void;
    onExit?: (node: HTMLElement) => void;
    onExiting?: (node: HTMLElement) => void;
    onExited?: (node: HTMLElement) => void;
  }

  interface TransitionGroupProps {
    component?: React.ElementType | null;
    children?: React.ReactNode;
    className?: string;
    appear?: boolean;
    enter?: boolean;
    exit?: boolean;
  }

  export class CSSTransition extends React.Component<CSSTransitionProps> {}
  export class TransitionGroup extends React.Component<TransitionGroupProps> {}
}
