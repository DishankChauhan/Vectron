import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Models from './pages/Models'
import Collections from './pages/Collections'
import Playground from './pages/Playground'
import Login from './pages/Login'
import { login } from './services/api'
import type { User, LoginResponse } from './types'
import './App.css'

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
});

// Protected route component
interface ProtectedRouteProps {
  isAuthenticated: boolean;
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ isAuthenticated, children }) => {
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
};

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [initialCheckDone, setInitialCheckDone] = useState<boolean>(false);

  useEffect(() => {
    // Check for stored token on app load
    const storedToken = localStorage.getItem('token');
    const storedUser = localStorage.getItem('user');
    
    if (storedToken && storedUser) {
      setToken(storedToken);
      setUser(JSON.parse(storedUser));
      setIsAuthenticated(true);
    }
    
    setInitialCheckDone(true);
  }, []);

  const handleLogin = async (username: string, password: string) => {
    try {
      const response = await login(username, password);
      const data = response as LoginResponse;
      
      if (data.access_token) {
        setToken(data.access_token);
        localStorage.setItem('token', data.access_token);
        
        // For now, we'll mock the user - in a real app, you'd fetch the user profile
        const mockUser = {
          username,
          email: `${username}@example.com`,
          role: username === 'admin' ? 'admin' : 'user' as 'admin' | 'user'
        };
        
        setUser(mockUser);
        localStorage.setItem('user', JSON.stringify(mockUser));
        setIsAuthenticated(true);
        
        return true;
      }
      return false;
    } catch (error) {
      console.error('Login failed:', error);
      return false;
    }
  };

  const handleLogout = () => {
    setUser(null);
    setToken(null);
    setIsAuthenticated(false);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  };

  // Wait for initial authentication check before rendering
  if (!initialCheckDone) {
    return <div>Loading...</div>;
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/login" element={
            isAuthenticated ? 
            <Navigate to="/" replace /> : 
            <Login onLogin={handleLogin} />
          } />
          
          <Route path="/" element={
            <ProtectedRoute isAuthenticated={isAuthenticated}>
              <Layout user={user} onLogout={handleLogout}>
                <Dashboard />
              </Layout>
            </ProtectedRoute>
          } />
          
          <Route path="/models" element={
            <ProtectedRoute isAuthenticated={isAuthenticated}>
              <Layout user={user} onLogout={handleLogout}>
                <Models />
              </Layout>
            </ProtectedRoute>
          } />
          
          <Route path="/collections" element={
            <ProtectedRoute isAuthenticated={isAuthenticated}>
              <Layout user={user} onLogout={handleLogout}>
                <Collections />
              </Layout>
            </ProtectedRoute>
          } />
          
          <Route path="/playground" element={
            <ProtectedRoute isAuthenticated={isAuthenticated}>
              <Layout user={user} onLogout={handleLogout}>
                <Playground />
              </Layout>
            </ProtectedRoute>
          } />
          
          {/* Add more protected routes as you create more pages */}
          
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>
    </ThemeProvider>
  )
}

export default App 