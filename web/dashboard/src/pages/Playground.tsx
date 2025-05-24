import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Divider,
  Tab,
  Tabs,
  CircularProgress
} from '@mui/material';
import { getModels } from '../services/api';
import { Model } from '../types';

interface TabPanelProps {
  children: React.ReactNode;
  value: number;
  index: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`playground-tabpanel-${index}`}
      aria-labelledby={`playground-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const Playground: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [singleText, setSingleText] = useState<string>('');
  const [batchTexts, setBatchTexts] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await getModels();
      setModels(response.data);
      if (response.data.length > 0) {
        setSelectedModel(response.data[0].id);
      }
    } catch (err) {
      setError('Failed to fetch models');
      console.error(err);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setResult('');
  };

  const handleModelChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setSelectedModel(event.target.value as string);
  };

  const generateCurlCommand = (endpoint: string, data: Record<string, any>) => {
    const headers = apiKey 
      ? `-H "X-API-Key: ${apiKey}"` 
      : '';
    
    const jsonData = JSON.stringify(data, null, 2);
    
    return `curl -X POST "http://localhost:8001${endpoint}" \\
  -H "Content-Type: application/json" \\
  ${headers} \\
  -d '${jsonData}'`;
  };

  const handleSingleEmbedding = () => {
    if (!singleText.trim()) {
      setError('Please enter text to embed');
      return;
    }

    const data = {
      text: singleText,
      model_id: selectedModel
    };

    setResult(generateCurlCommand('/embed', data));
    setError(null);
  };

  const handleBatchEmbedding = () => {
    if (!batchTexts.trim()) {
      setError('Please enter texts to embed');
      return;
    }

    try {
      const texts = batchTexts.split('\n').filter(t => t.trim().length > 0);
      
      if (texts.length === 0) {
        setError('Please enter at least one text to embed');
        return;
      }
      
      const data = {
        texts: texts,
        model_id: selectedModel
      };
      
      setResult(generateCurlCommand('/batch-embed', data));
      setError(null);
    } catch (err) {
      setError('Failed to parse input texts');
      console.error(err);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        API Playground
      </Typography>
      
      <Typography paragraph>
        Test the Vectron API endpoints with these code examples.
      </Typography>
      
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Authentication
          </Typography>
          
          <TextField
            label="API Key (Optional)"
            variant="outlined"
            fullWidth
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Enter your API key for authenticated requests"
            sx={{ mb: 2 }}
          />
          
          <Paper variant="outlined" sx={{ p: 2, backgroundColor: '#f5f5f5' }}>
            <Typography variant="body2" component="div" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
              # To get an API key, use:
              {'\n'}
              curl -X POST "http://localhost:8001/users/{'{username}'}/api-key" \
              {'\n'}
              -H "Authorization: Bearer {'{your_jwt_token}'}"
            </Typography>
          </Paper>
        </CardContent>
      </Card>
      
      <Box sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab label="Single Embedding" id="playground-tab-0" />
            <Tab label="Batch Embedding" id="playground-tab-1" />
          </Tabs>
        </Box>
        
        {error && (
          <Typography color="error" sx={{ mt: 2 }}>
            {error}
          </Typography>
        )}
        
        <TabPanel value={tabValue} index={0}>
          <Card>
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel id="model-select-label">Model</InputLabel>
                    <Select
                      labelId="model-select-label"
                      id="model-select"
                      value={selectedModel}
                      label="Model"
                      onChange={handleModelChange}
                    >
                      {models.map((model) => (
                        <MenuItem key={model.id} value={model.id}>
                          {model.name} ({model.dimensions} dimensions)
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    label="Text to Embed"
                    multiline
                    rows={4}
                    variant="outlined"
                    fullWidth
                    value={singleText}
                    onChange={(e) => setSingleText(e.target.value)}
                    placeholder="Enter text to generate embedding..."
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Button 
                    variant="contained" 
                    color="primary" 
                    onClick={handleSingleEmbedding}
                    disabled={!singleText.trim() || !selectedModel}
                  >
                    Generate curl Command
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <Card>
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel id="batch-model-select-label">Model</InputLabel>
                    <Select
                      labelId="batch-model-select-label"
                      id="batch-model-select"
                      value={selectedModel}
                      label="Model"
                      onChange={handleModelChange}
                    >
                      {models.map((model) => (
                        <MenuItem key={model.id} value={model.id}>
                          {model.name} ({model.dimensions} dimensions)
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    label="Texts to Embed (One per line)"
                    multiline
                    rows={6}
                    variant="outlined"
                    fullWidth
                    value={batchTexts}
                    onChange={(e) => setBatchTexts(e.target.value)}
                    placeholder="Enter texts to embed (one per line)..."
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Button 
                    variant="contained" 
                    color="primary" 
                    onClick={handleBatchEmbedding}
                    disabled={!batchTexts.trim() || !selectedModel}
                  >
                    Generate curl Command
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </TabPanel>
        
        {result && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Generated Command:
            </Typography>
            
            <Paper variant="outlined" sx={{ p: 2, backgroundColor: '#f5f5f5' }}>
              <Typography variant="body2" component="div" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
                {result}
              </Typography>
            </Paper>
            
            <Typography variant="body2" sx={{ mt: 2 }}>
              Copy this command and run it in your terminal to test the API.
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default Playground; 