// Simple mock for react-router-dom
const mockNavigate = jest.fn();

// Mock components
const BrowserRouter = ({ children }: any) => children;
const MemoryRouter = ({ children }: any) => children;
const Routes = ({ children }: any) => children;
const Route = ({ element }: any) => element;
const Link = ({ children }: any) => children;
const NavLink = ({ children }: any) => children;
const Navigate = () => null;
const Outlet = () => null;

// Mock hooks
const useNavigate = () => mockNavigate;
const useParams = () => ({});
const useLocation = () => ({ pathname: '/', search: '', hash: '', state: null });
const useRoutes = () => null;

export {
  BrowserRouter,
  MemoryRouter,
  Routes,
  Route,
  Link,
  NavLink,
  Navigate,
  Outlet,
  useNavigate,
  useParams,
  useLocation,
  useRoutes
};

module.exports = {
  BrowserRouter,
  MemoryRouter,
  Routes,
  Route,
  Link,
  NavLink,
  Navigate,
  Outlet,
  useNavigate,
  useParams,
  useLocation,
  useRoutes
};

// Default export for compatibility
export default {
  BrowserRouter,
  Routes,
  Route,
  Link,
  NavLink,
  Navigate,
  Outlet,
  MemoryRouter,
  useNavigate,
  useParams,
  useLocation,
  useRoutes
};
