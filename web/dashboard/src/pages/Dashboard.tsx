import React, { useEffect, useState } from 'react';
import { Box, Card, CardContent, Typography, Grid, CircularProgress } from '@mui/material';
import { getHealthStatus } from '../services/api';
import { HealthResponse } from '../types';

const Dashboard: React.FC = () => {
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealthData = async () => {
      try {
        setLoading(true);
        const response = await getHealthStatus();
        setHealthData(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch service health data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchHealthData();
    // Poll health data every 30 seconds
    const interval = setInterval(fetchHealthData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !healthData) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error && !healthData) {
    return (
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Typography color="error" variant="h6">
          {error}
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Service Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Service Status */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Service Status
              </Typography>
              <Typography variant="h5" component="div" sx={{ 
                color: healthData?.status === 'OK' ? 'success.main' : 'error.main' 
              }}>
                {healthData?.status || 'Unknown'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Uptime: {healthData ? Math.floor(healthData.uptime / 60) : 0} minutes
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Version: {healthData?.version || 'Unknown'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Models */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Models
              </Typography>
              <Typography variant="h5" component="div">
                {healthData?.models?.length || 0} Available
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {healthData?.models?.filter(m => m.is_loaded).length || 0} Loaded
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* API Keys */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                API Provider Status
              </Typography>
              {healthData?.provider_statuses ? (
                <Box>
                  {Object.entries(healthData.provider_statuses).map(([provider, status]) => (
                    <Typography key={provider} variant="body2" sx={{ 
                      color: status.is_valid ? 'success.main' : 'error.main'
                    }}>
                      {provider}: {status.is_valid ? 'Valid' : status.has_key ? 'Invalid' : 'Missing'}
                    </Typography>
                  ))}
                </Box>
              ) : (
                <Typography variant="body2" color="textSecondary">
                  No provider data available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Model List */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Available Models
              </Typography>
              <Grid container spacing={2}>
                {healthData?.models?.map((model) => (
                  <Grid item xs={12} sm={6} md={4} key={model.id}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6">{model.name}</Typography>
                        <Typography variant="body2" color="textSecondary">
                          ID: {model.id}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Provider: {model.provider}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Dimensions: {model.dimensions}
                        </Typography>
                        <Typography 
                          variant="body2" 
                          sx={{ color: model.is_loaded ? 'success.main' : 'warning.main' }}
                        >
                          Status: {model.is_loaded ? 'Loaded' : 'Not Loaded'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 