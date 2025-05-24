import axios from 'axios';

const API_URL = import.meta.env.PROD 
  ? 'http://localhost:8000'  // Production build
  : 'http://localhost:8000'; // Development

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Authentication
export const login = async (username: string, password: string) => {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('password', password);
  
  const response = await axios.post(`${API_URL}/token`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const createUser = async (userData: any) => {
  return api.post('/users', userData);
};

export const generateApiKey = async (username: string) => {
  return api.post(`/users/${username}/api-key`);
};

// Models
export const getModels = async () => {
  return api.get('/models');
};

// Embeddings
export const createEmbedding = async (text: string, modelId = 'minilm') => {
  return api.post('/embed', { text, model_id: modelId });
};

export const batchEmbedding = async (texts: string[], modelId = 'minilm') => {
  return api.post('/batch-embed', { texts, model_id: modelId });
};

// Collections
export const createCollection = async (collectionData: any) => {
  return api.post('/collections', collectionData);
};

export const getCollections = async () => {
  return api.get('/collections');
};

export const searchCollection = async (name: string, queryVector: number[], topK = 5) => {
  return api.post(`/collections/${name}/search`, { query_vector: queryVector, top_k: topK });
};

export const storeVector = async (collectionName: string, text: string, modelId = 'minilm', metadata?: any) => {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('collection_name', collectionName);
  formData.append('model_id', modelId);
  
  if (metadata) {
    formData.append('metadata', JSON.stringify(metadata));
  }
  
  return api.post('/embed-and-store', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// Health
export const getHealthStatus = async () => {
  return api.get('/health');
};

export default api; 