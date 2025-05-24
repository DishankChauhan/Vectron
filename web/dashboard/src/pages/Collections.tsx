import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  CircularProgress,
  Divider,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tab,
  Tabs,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import {
  Add as AddIcon,
  Search as SearchIcon,
  Delete as DeleteIcon,
  Save as SaveIcon
} from '@mui/icons-material';
import { getCollections, createCollection, storeVector, createEmbedding, searchCollection } from '../services/api';
import type { Collection, Model, QueryResult } from '../types';

interface TabPanelProps {
  children: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`collection-tabpanel-${index}`}
      aria-labelledby={`collection-tab-${index}`}
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

const Collections: React.FC = () => {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState('');
  const [newCollectionDimension, setNewCollectionDimension] = useState(384);
  const [newCollectionMetric, setNewCollectionMetric] = useState('cosine');
  const [newCollectionIndex, setNewCollectionIndex] = useState('flat');
  const [tabValue, setTabValue] = useState(0);
  const [selectedCollection, setSelectedCollection] = useState<string>('');
  const [storeText, setStoreText] = useState('');
  const [queryText, setQueryText] = useState('');
  const [queryResults, setQueryResults] = useState<QueryResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [queryTime, setQueryTime] = useState<number | null>(null);

  useEffect(() => {
    fetchCollections();
  }, []);

  const fetchCollections = async () => {
    try {
      setLoading(true);
      const response = await getCollections();
      setCollections(response.data);
      
      if (response.data.length > 0 && !selectedCollection) {
        setSelectedCollection(response.data[0].name);
      }
    } catch (err) {
      setError('Failed to fetch collections');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleCreateCollection = async () => {
    if (!newCollectionName.trim()) {
      setError('Collection name is required');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      await createCollection({
        name: newCollectionName,
        dimension: newCollectionDimension,
        distance_metric: newCollectionMetric as 'cosine' | 'euclidean' | 'dot',
        index_type: newCollectionIndex as 'flat' | 'hnsw'
      });
      
      // Reset form
      setNewCollectionName('');
      setCreateDialogOpen(false);
      
      // Refresh collections list
      fetchCollections();
    } catch (err) {
      setError('Failed to create collection');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleStoreVector = async () => {
    if (!storeText.trim() || !selectedCollection) {
      setError('Text and collection are required');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      await storeVector(selectedCollection, storeText);
      setStoreText('');
      
      // Refresh collections to update counts
      fetchCollections();
    } catch (err) {
      setError('Failed to store vector');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!queryText.trim() || !selectedCollection) {
      setError('Query text and collection are required');
      return;
    }

    try {
      setSearchLoading(true);
      setError(null);
      
      // First generate embedding for the query text
      const embeddingResponse = await createEmbedding(queryText);
      const embedding = embeddingResponse.data.embedding;
      
      // Then search using the embedding
      const searchResponse = await searchCollection(selectedCollection, embedding);
      setQueryResults(searchResponse.data.results);
      setQueryTime(searchResponse.data.query_time_ms);
    } catch (err) {
      setError('Failed to perform search');
      console.error(err);
    } finally {
      setSearchLoading(false);
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Vector Collections
        </Typography>
        
        <Button 
          variant="contained" 
          color="primary" 
          startIcon={<AddIcon />}
          onClick={() => setCreateDialogOpen(true)}
        >
          Create Collection
        </Button>
      </Box>
      
      {error && (
        <Typography color="error" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}
      
      {loading && !collections.length ? (
        <CircularProgress />
      ) : (
        <>
          {collections.length === 0 ? (
            <Card>
              <CardContent>
                <Typography>
                  No collections found. Create your first collection to get started.
                </Typography>
              </CardContent>
            </Card>
          ) : (
            <>
              <Box sx={{ mb: 3 }}>
                <FormControl fullWidth>
                  <InputLabel id="collection-select-label">Collection</InputLabel>
                  <Select
                    labelId="collection-select-label"
                    id="collection-select"
                    value={selectedCollection}
                    label="Collection"
                    onChange={(e) => setSelectedCollection(e.target.value as string)}
                  >
                    {collections.map((collection) => (
                      <MenuItem key={collection.name} value={collection.name}>
                        {collection.name} ({collection.count} vectors, {collection.dimension}d)
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>
              
              <Box sx={{ width: '100%' }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                  <Tabs value={tabValue} onChange={handleTabChange}>
                    <Tab label="Collection Details" id="collection-tab-0" aria-controls="collection-tabpanel-0" />
                    <Tab label="Add Vectors" id="collection-tab-1" aria-controls="collection-tabpanel-1" />
                    <Tab label="Search" id="collection-tab-2" aria-controls="collection-tabpanel-2" />
                  </Tabs>
                </Box>
                
                <TabPanel value={tabValue} index={0}>
                  {selectedCollection && (
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Collection Details
                        </Typography>
                        
                        {collections.filter(c => c.name === selectedCollection).map(collection => (
                          <Box key={collection.name}>
                            <Grid container spacing={2}>
                              <Grid item xs={12} sm={6}>
                                <Typography variant="subtitle2">Name</Typography>
                                <Typography>{collection.name}</Typography>
                              </Grid>
                              <Grid item xs={12} sm={6}>
                                <Typography variant="subtitle2">Dimension</Typography>
                                <Typography>{collection.dimension}</Typography>
                              </Grid>
                              <Grid item xs={12} sm={6}>
                                <Typography variant="subtitle2">Vector Count</Typography>
                                <Typography>{collection.count}</Typography>
                              </Grid>
                              <Grid item xs={12} sm={6}>
                                <Typography variant="subtitle2">Index Type</Typography>
                                <Typography>{collection.index_type}</Typography>
                              </Grid>
                              <Grid item xs={12} sm={6}>
                                <Typography variant="subtitle2">Distance Metric</Typography>
                                <Typography>{collection.distance_metric}</Typography>
                              </Grid>
                              <Grid item xs={12} sm={6}>
                                <Typography variant="subtitle2">Created At</Typography>
                                <Typography>{new Date(collection.created_at).toLocaleString()}</Typography>
                              </Grid>
                            </Grid>
                          </Box>
                        ))}
                      </CardContent>
                    </Card>
                  )}
                </TabPanel>
                
                <TabPanel value={tabValue} index={1}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Add Vector to Collection
                      </Typography>
                      
                      <TextField
                        label="Text to Embed"
                        multiline
                        rows={4}
                        variant="outlined"
                        fullWidth
                        value={storeText}
                        onChange={(e) => setStoreText(e.target.value)}
                        placeholder="Enter text to embed and store in the collection..."
                        sx={{ mb: 2 }}
                      />
                      
                      <Button 
                        variant="contained" 
                        color="primary"
                        startIcon={<SaveIcon />}
                        onClick={handleStoreVector}
                        disabled={loading || !storeText.trim() || !selectedCollection}
                      >
                        {loading ? <CircularProgress size={24} /> : 'Store Vector'}
                      </Button>
                    </CardContent>
                  </Card>
                </TabPanel>
                
                <TabPanel value={tabValue} index={2}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Search Collection
                      </Typography>
                      
                      <TextField
                        label="Query Text"
                        multiline
                        rows={2}
                        variant="outlined"
                        fullWidth
                        value={queryText}
                        onChange={(e) => setQueryText(e.target.value)}
                        placeholder="Enter text to search for similar vectors..."
                        sx={{ mb: 2 }}
                      />
                      
                      <Button 
                        variant="contained" 
                        color="primary"
                        startIcon={<SearchIcon />}
                        onClick={handleSearch}
                        disabled={searchLoading || !queryText.trim() || !selectedCollection}
                      >
                        {searchLoading ? <CircularProgress size={24} /> : 'Search'}
                      </Button>
                      
                      {queryResults.length > 0 && (
                        <Box sx={{ mt: 3 }}>
                          <Typography variant="subtitle1">
                            Search Results {queryTime !== null && `(${queryTime.toFixed(2)}ms)`}:
                          </Typography>
                          
                          <List>
                            {queryResults.map((result, index) => (
                              <ListItem 
                                key={result.id} 
                                divider={index < queryResults.length - 1}
                                sx={{ 
                                  backgroundColor: index === 0 ? 'rgba(25, 118, 210, 0.08)' : 'inherit',
                                  borderRadius: index === 0 ? 1 : 0
                                }}
                              >
                                <ListItemText
                                  primary={`ID: ${result.id}`}
                                  secondary={`Score: ${result.score.toFixed(4)}${result.metadata ? ' • ' + JSON.stringify(result.metadata) : ''}`}
                                />
                              </ListItem>
                            ))}
                          </List>
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </TabPanel>
              </Box>
            </>
          )}
        </>
      )}
      
      {/* Create Collection Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)}>
        <DialogTitle>Create New Collection</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Collection Name"
            fullWidth
            variant="outlined"
            value={newCollectionName}
            onChange={(e) => setNewCollectionName(e.target.value)}
            sx={{ mb: 2 }}
          />
          
          <TextField
            margin="dense"
            label="Dimension"
            type="number"
            fullWidth
            variant="outlined"
            value={newCollectionDimension}
            onChange={(e) => setNewCollectionDimension(parseInt(e.target.value))}
            sx={{ mb: 2 }}
          />
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="metric-select-label">Distance Metric</InputLabel>
            <Select
              labelId="metric-select-label"
              value={newCollectionMetric}
              label="Distance Metric"
              onChange={(e) => setNewCollectionMetric(e.target.value as string)}
            >
              <MenuItem value="cosine">Cosine</MenuItem>
              <MenuItem value="euclidean">Euclidean</MenuItem>
              <MenuItem value="dot">Dot Product</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl fullWidth>
            <InputLabel id="index-select-label">Index Type</InputLabel>
            <Select
              labelId="index-select-label"
              value={newCollectionIndex}
              label="Index Type"
              onChange={(e) => setNewCollectionIndex(e.target.value as string)}
            >
              <MenuItem value="flat">Flat</MenuItem>
              <MenuItem value="hnsw">HNSW</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateCollection} variant="contained" color="primary">
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Collections; 