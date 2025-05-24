import React, { useState } from 'react';
import { Box, CssBaseline, Toolbar } from '@mui/material';
import Header from './Header';
import Sidebar from './Sidebar';
import { User } from '../types';

interface LayoutProps {
  children: React.ReactNode;
  user: User | null;
  onLogout: () => void;
}

const Layout: React.FC<LayoutProps> = ({ children, user, onLogout }) => {
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <CssBaseline />
      
      <Header 
        onMenuClick={handleDrawerToggle} 
        user={user} 
        onLogout={onLogout} 
      />
      
      <Sidebar 
        isAdmin={user?.role === 'admin'} 
        open={mobileOpen} 
        onClose={handleDrawerToggle} 
        variant="temporary"
      />
      
      <Sidebar 
        isAdmin={user?.role === 'admin'} 
        open={true} 
        onClose={() => {}} 
        variant="permanent"
      />
      
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - 240px)` },
          mt: 8
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
};

export default Layout; 