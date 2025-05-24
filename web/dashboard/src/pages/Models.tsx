import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  Card, 
  CardContent, 
  TextField, 
  Button, 
  CircularProgress,
  Paper,
  Chip,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import { getModels, createEmbedding } from '../services/api';
import { Model, EmbeddingResponse } from '../types';

const Models: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [inputText, setInputText] = useState<string>('');
  const [embedding, setEmbedding] = useState<number[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await getModels();
        setModels(response.data);
        if (response.data.length > 0) {
          // Select the first model by default
          setSelectedModel(response.data[0].id);
        }
      } catch (err) {
        setError('Failed to fetch models');
        console.error(err);
      }
    };
    
    fetchModels();
  }, []);

  const handleModelChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setSelectedModel(event.target.value as string);
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputText(event.target.value);
  };

  const handleGenerateEmbedding = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to embed');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const response = await createEmbedding(inputText, selectedModel);
      const embeddingData = response.data as EmbeddingResponse;
      
      setEmbedding(embeddingData.embedding);
    } catch (err) {
      setError('Failed to generate embedding');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Embedding Models
      </Typography>
      
      <Grid container spacing={3}>
        {/* Models List */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Available Models
              </Typography>
              <Grid container spacing={2}>
                {models.map((model) => (
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
                        <Box mt={1}>
                          <Chip 
                            label={model.is_remote ? "Remote API" : "Local"} 
                            color={model.is_remote ? "secondary" : "primary"} 
                            size="small" 
                          />
                          {' '}
                          <Chip 
                            label={model.is_loaded ? "Loaded" : "Not Loaded"} 
                            color={model.is_loaded ? "success" : "warning"} 
                            size="small" 
                          />
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Embedding Generation */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Test Embedding Generation
              </Typography>
              
              <Grid container spacing={2}>
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
                    value={inputText}
                    onChange={handleInputChange}
                    placeholder="Enter text to generate embeddings..."
                  />
                </Grid>

                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={handleGenerateEmbedding}
                    disabled={loading || !selectedModel}
                  >
                    {loading ? <CircularProgress size={24} /> : 'Generate Embedding'}
                  </Button>
                </Grid>

                {error && (
                  <Grid item xs={12}>
                    <Typography color="error">{error}</Typography>
                  </Grid>
                )}

                {embedding && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom>
                      Embedding Result ({embedding.length} dimensions):
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2, maxHeight: 200, overflow: 'auto' }}>
                      <Typography variant="body2" component="div" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                        [{embedding.slice(0, 10).map(n => n.toFixed(6)).join(', ')}
                        {embedding.length > 10 ? ', ...' : ''}]
                      </Typography>
                    </Paper>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Models; 