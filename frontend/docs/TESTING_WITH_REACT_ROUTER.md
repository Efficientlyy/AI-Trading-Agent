# Testing with React Router in AI Trading Agent

This document explains how to test components that use React Router in the AI Trading Agent project.

## Background

We encountered an issue where Jest couldn't resolve `react-router-dom` in tests, even though it worked fine in the application code. This was due to a compatibility issue between Jest's module resolution system and React Router DOM v7.5.1's modern ES module format.

## Solution: Mock Implementation

We created a mock implementation of React Router DOM for Jest to use during tests. This is located in:

```
src/__mocks__/react-router-dom.ts
```

This mock provides all the commonly used components and hooks from React Router, such as:
- `BrowserRouter`, `MemoryRouter`, `Routes`, `Route`
- `Link`, `NavLink`, `Navigate`, `Outlet`
- `useNavigate`, `useParams`, `useLocation`, `useRoutes`

## How to Use in Tests

### Basic Component Testing

For basic component testing, you can simply import from 'react-router-dom' as usual:

```tsx
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import YourComponent from './YourComponent';

test('your component renders correctly', () => {
  const { getByText } = render(
    <MemoryRouter>
      <YourComponent />
    </MemoryRouter>
  );
  
  expect(getByText('Expected Text')).toBeInTheDocument();
});
```

### Testing Components that Use Context Providers

If your component uses context providers (like SelectedAssetProvider), make sure to wrap your component in all necessary providers:

```tsx
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { SelectedAssetProvider } from '../context/SelectedAssetContext';
import YourComponent from './YourComponent';

test('your component with context', () => {
  const { getByText } = render(
    <SelectedAssetProvider>
      <MemoryRouter>
        <YourComponent />
      </MemoryRouter>
    </SelectedAssetProvider>
  );
  
  // Your assertions here
});
```

### Testing Navigation

To test navigation, you can use the `userEvent` library to interact with links:

```tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import YourComponent from './YourComponent';

test('navigation works', async () => {
  render(
    <MemoryRouter>
      <YourComponent />
    </MemoryRouter>
  );
  
  // Find a link and click it
  const link = screen.getByText('Go to Dashboard');
  await userEvent.click(link);
  
  // Assert that navigation occurred
  // Note: Since we're using a mock, you'll need to check for side effects
  // rather than actual URL changes
});
```

## Extending the Mock

If you need to add more components or hooks from React Router to the mock, edit the `src/__mocks__/react-router-dom.ts` file and add your new components or hooks.

## Troubleshooting

If you encounter issues with the React Router mock:

1. Make sure you're importing from 'react-router-dom' directly, not from a specific path like '../../node_modules/react-router-dom'
2. Check that you're wrapping your components in the necessary providers
3. Ensure you've imported the Jest types with `import { jest } from '@jest/globals'` in your test files

## Example Tests

See the following test files for examples:
- `src/router-test.test.tsx` - Basic React Router component testing
- `src/pages/__integration__.test.tsx` - Testing components that share context across multiple components
- `src/context/SelectedAssetContext.test.tsx` - Testing context providers with React Router
