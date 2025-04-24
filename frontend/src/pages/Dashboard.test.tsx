import '@testing-library/jest-dom';
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Dashboard from './Dashboard'; // Assuming Dashboard component exists

describe('Dashboard Page', () => {
  it('renders without crashing', () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    );
    // You might add a simple assertion here later, e.g., checking for a specific piece of text
    expect(true).toBe(true); // Placeholder assertion
  });
});
