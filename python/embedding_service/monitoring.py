"""
Monitoring module for the embedding service.
Provides Prometheus metrics and logging capabilities.
"""

import time
import logging
import os
from typing import Callable, Dict, Any, Optional, List
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Summary, Info, start_http_server
import psutil
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics settings
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8002"))
ENABLE_METRICS = os.environ.get("ENABLE_METRICS", "true").lower() == "true"

# Metrics
REQUEST_COUNT = Counter('embedding_request_count', 'Total number of requests', ['endpoint', 'model_id', 'status'])
REQUEST_LATENCY = Histogram('embedding_request_latency_seconds', 'Request latency in seconds', 
                           ['endpoint', 'model_id'], buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0))
TOKEN_COUNT = Counter('embedding_token_count', 'Total number of tokens processed', ['model_id'])
BATCH_SIZE = Histogram('embedding_batch_size', 'Batch sizes of embedding requests', 
                      ['model_id'], buckets=(1, 5, 10, 25, 50, 100, 250, 500))
MODEL_LOAD_TIME = Histogram('embedding_model_load_time_seconds', 'Time to load a model in seconds',
                           ['model_id'], buckets=(0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0))
MODEL_LOADED = Gauge('embedding_model_loaded', 'Whether a model is currently loaded', ['model_id'])
API_KEY_STATUS = Gauge('embedding_api_key_status', 'Status of provider API keys', ['provider'])
SYSTEM_MEMORY_USAGE = Gauge('embedding_system_memory_percent', 'System memory usage percentage')
SYSTEM_CPU_USAGE = Gauge('embedding_system_cpu_percent', 'System CPU usage percentage')
SYSTEM_DISK_USAGE = Gauge('embedding_system_disk_percent', 'System disk usage percentage')
SERVICE_INFO = Info('embedding_service', 'Embedding service information')

# Initialize metrics server
def initialize_metrics_server():
    """Initialize the Prometheus metrics server if enabled."""
    if ENABLE_METRICS:
        try:
            start_http_server(METRICS_PORT)
            logger.info(f"Prometheus metrics server started on port {METRICS_PORT}")
            
            # Initialize service info
            service_info = {
                'version': os.environ.get("SERVICE_VERSION", "0.1.0"),
                'python_version': os.environ.get("PYTHON_VERSION", "3.10"),
                'host': os.environ.get("HOSTNAME", "unknown")
            }
            SERVICE_INFO.info(service_info)
            
            # Schedule periodic system metrics collection
            import threading
            threading.Thread(target=collect_system_metrics, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    else:
        logger.info("Prometheus metrics server disabled")


def collect_system_metrics(interval=15):
    """Collect system metrics periodically."""
    while True:
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.percent)
            
            # CPU usage
            cpu = psutil.cpu_percent(interval=None)
            SYSTEM_CPU_USAGE.set(cpu)
            
            # Disk usage for main partition
            disk = psutil.disk_usage('/')
            SYSTEM_DISK_USAGE.set(disk.percent)
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        time.sleep(interval)


def track_request_time(endpoint: str, model_id: str = "unknown"):
    """Decorator to track request time for an endpoint."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                logger.error(f"Error in {endpoint} with model {model_id}: {str(e)}")
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_LATENCY.labels(endpoint=endpoint, model_id=model_id).observe(duration)
                REQUEST_COUNT.labels(endpoint=endpoint, model_id=model_id, status=status).inc()
                
        return wrapper
    return decorator


def track_model_load_time(model_id: str):
    """Decorator to track model loading time."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            MODEL_LOADED.labels(model_id=model_id).set(0)
            
            try:
                result = func(*args, **kwargs)
                MODEL_LOADED.labels(model_id=model_id).set(1)
                return result
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {str(e)}")
                raise
            finally:
                duration = time.time() - start_time
                MODEL_LOAD_TIME.labels(model_id=model_id).observe(duration)
                
        return wrapper
    return decorator


def track_batch_size(model_id: str, batch_size: int):
    """Track the batch size for a model."""
    BATCH_SIZE.labels(model_id=model_id).observe(batch_size)


def track_token_count(model_id: str, token_count: int):
    """Track the number of tokens processed by a model."""
    TOKEN_COUNT.labels(model_id=model_id).inc(token_count)


def set_api_key_status(provider: str, status: bool):
    """Set the API key status for a provider."""
    API_KEY_STATUS.labels(provider=provider).set(1 if status else 0)


def log_model_info(model_id: str, dimensions: int, provider: str = "local"):
    """Log information about a model."""
    logger.info(f"Model {model_id} from {provider} with {dimensions} dimensions")


def log_request(endpoint: str, model_id: str, input_size: int, status_code: int, duration: float):
    """Log information about a request."""
    logger.info(f"{endpoint} with model {model_id} for {input_size} inputs: {status_code} in {duration:.2f}s")


def calculate_embedding_stats(embeddings: List[List[float]]) -> Dict[str, Any]:
    """Calculate statistics for a batch of embeddings."""
    if not embeddings:
        return {}
        
    embeddings_np = np.array(embeddings)
    
    # Compute statistics
    stats = {
        "mean": float(np.mean(embeddings_np)),
        "std": float(np.std(embeddings_np)),
        "min": float(np.min(embeddings_np)),
        "max": float(np.max(embeddings_np)),
        "l2_norm_mean": float(np.mean(np.linalg.norm(embeddings_np, axis=1))),
        "dimension": embeddings_np.shape[1]
    }
    
    return stats


# Initialize metrics on module load
initialize_metrics_server() 