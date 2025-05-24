// Authentication types
export interface User {
  username: string;
  email: string;
  role: 'user' | 'admin';
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  error: string | null;
  loading: boolean;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

// Embedding types
export interface EmbeddingRequest {
  text: string;
  model_id?: string;
}

export interface BatchEmbeddingRequest {
  texts: string[];
  model_id?: string;
}

export interface EmbeddingResponse {
  embedding: number[];
  dimensions: number;
  model_id: string;
  generated_at: string;
}

export interface BatchEmbeddingResponse {
  embeddings: number[][];
  dimensions: number;
  model_id: string;
  generated_at: string;
}

// Model types
export interface Model {
  id: string;
  name: string;
  dimensions: number;
  is_remote: boolean;
  is_loaded: boolean;
  provider: string;
}

// Collection types
export interface Collection {
  name: string;
  dimension: number;
  count: number;
  index_type: string;
  distance_metric: string;
  created_at: string;
}

export interface CollectionCreateRequest {
  name: string;
  dimension: number;
  distance_metric: 'cosine' | 'euclidean' | 'dot';
  index_type: 'flat' | 'hnsw';
}

export interface QueryResult {
  id: string;
  score: number;
  metadata?: Record<string, any>;
}

export interface SearchResponse {
  results: QueryResult[];
  query_time_ms: number;
}

// Health check types
export interface ProviderStatus {
  has_key: boolean;
  is_valid: boolean;
}

export interface ModelStatus {
  id: string;
  name: string;
  is_loaded: boolean;
  dimensions: number;
  provider: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime: number;
  provider_statuses: Record<string, ProviderStatus>;
  models: ModelStatus[];
}

// Service metrics types
export interface MetricPoint {
  timestamp: number;
  value: number;
}

export interface PerformanceMetrics {
  request_count: MetricPoint[];
  average_response_time: MetricPoint[];
  error_rate: MetricPoint[];
} 