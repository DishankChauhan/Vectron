import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Box, 
  Drawer, 
  List, 
  ListItem, 
  ListItemButton, 
  ListItemIcon, 
  ListItemText,
  Divider,
  IconButton,
  Typography
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Code as CodeIcon,
  Storage as StorageIcon,
  Settings as SettingsIcon,
  Api as ApiIcon,
  SupervisorAccount as AdminIcon,
  Menu as MenuIcon,
  Close as CloseIcon
} from '@mui/icons-material';

const DRAWER_WIDTH = 240;

interface SidebarItem {
  text: string;
  icon: JSX.Element;
  path: string;
  adminOnly?: boolean;
}

const menuItems: SidebarItem[] = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Models', icon: <CodeIcon />, path: '/models' },
  { text: 'Collections', icon: <StorageIcon />, path: '/collections' },
  { text: 'API Playground', icon: <ApiIcon />, path: '/playground' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
  { text: 'Admin', icon: <AdminIcon />, path: '/admin', adminOnly: true },
];

interface SidebarProps {
  isAdmin?: boolean;
  open: boolean;
  onClose: () => void;
  variant: 'permanent' | 'temporary';
}

const Sidebar = ({ isAdmin = false, open, onClose, variant }: SidebarProps) => {
  const location = useLocation();
  
  const drawer = (
    <>
      <Box sx={{ display: 'flex', alignItems: 'center', p: 2, justifyContent: 'space-between' }}>
        <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 'bold' }}>
          Vectron UI
        </Typography>
        {variant === 'temporary' && (
          <IconButton onClick={onClose}>
            <CloseIcon />
          </IconButton>
        )}
      </Box>
      <Divider />
      <List>
        {menuItems
          .filter(item => !item.adminOnly || (item.adminOnly && isAdmin))
          .map((item) => (
            <ListItem key={item.text} disablePadding>
              <ListItemButton 
                component={Link} 
                to={item.path}
                selected={location.pathname === item.path}
                onClick={variant === 'temporary' ? onClose : undefined}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItemButton>
            </ListItem>
          ))}
      </List>
    </>
  );

  return (
    <Box
      component="nav"
      sx={{ width: { sm: DRAWER_WIDTH }, flexShrink: { sm: 0 } }}
    >
      {/* Mobile drawer */}
      {variant === 'temporary' && (
        <Drawer
          variant="temporary"
          open={open}
          onClose={onClose}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: DRAWER_WIDTH },
          }}
        >
          {drawer}
        </Drawer>
      )}

      {/* Desktop drawer */}
      {variant === 'permanent' && (
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: DRAWER_WIDTH },
          }}
          open
        >
          {drawer}
        </Drawer>
      )}
    </Box>
  );
};

export default Sidebar; 