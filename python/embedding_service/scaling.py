"""
Scaling module for the embedding service.
Handles caching, batching, and load balancing for improved performance.
"""

import os
import time
import logging
import json
import threading
import hashlib
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic, Union
from datetime import datetime, timedelta
import redis
from fastapi import BackgroundTasks
from queue import PriorityQueue, Empty
import numpy as np
import asyncio
import functools
import random
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for generics
T = TypeVar('T')
U = TypeVar('U')

# Redis configuration for caching
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_ENABLED = os.environ.get("REDIS_ENABLED", "false").lower() == "true"
REDIS_TTL = int(os.environ.get("REDIS_TTL", "3600"))  # 1 hour

# Thread pool settings
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
THREAD_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Batch processing settings
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "100"))
BATCH_WAIT_TIME = float(os.environ.get("BATCH_WAIT_TIME", "0.1"))  # seconds

# Load balancing settings
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "20"))


class Cache:
    """Cache interface for storing and retrieving data."""
    
    def __init__(self):
        """Initialize the cache."""
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection if enabled."""
        self.redis = None
        if REDIS_ENABLED:
            try:
                self.redis = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    password=REDIS_PASSWORD,
                    decode_responses=False
                )
                logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Convert args and kwargs to a string and hash it
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[bytes]:
        """Get a value from the cache."""
        if not self.redis:
            return None
        
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: bytes, ttl: int = REDIS_TTL) -> bool:
        """Set a value in the cache with a TTL."""
        if not self.redis:
            return False
        
        try:
            return self.redis.set(key, value, ex=ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        if not self.redis:
            return False
        
        try:
            return self.redis.delete(key) > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def flush(self) -> bool:
        """Flush the entire cache."""
        if not self.redis:
            return False
        
        try:
            return self.redis.flushdb()
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False


# Global cache instance
cache = Cache()


def cache_result(ttl: int = REDIS_TTL, prefix: str = "embedding"):
    """Decorator to cache function results."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Skip caching if Redis is not available
            if not cache.redis:
                return await func(*args, **kwargs)
            
            # Generate cache key
            skip_cache = kwargs.pop("skip_cache", False)
            cache_key = f"{prefix}:{cache._generate_key(*args, **kwargs)}"
            
            # Try to get from cache if not skipping
            if not skip_cache:
                cached_result = cache.get(cache_key)
                if cached_result:
                    try:
                        return json.loads(cached_result)
                    except Exception as e:
                        logger.error(f"Error deserializing cached result: {e}")
            
            # Execute function if not in cache or skip_cache
            result = await func(*args, **kwargs)
            
            # Store in cache
            try:
                cache.set(cache_key, json.dumps(result).encode(), ttl)
            except Exception as e:
                logger.error(f"Error caching result: {e}")
            
            return result
        return wrapper
    return decorator


class BatchProcessor(Generic[T, U]):
    """Process items in batches for improved performance."""
    
    def __init__(self, process_func: Callable[[List[T]], List[U]], 
                 max_batch_size: int = MAX_BATCH_SIZE,
                 max_wait_time: float = BATCH_WAIT_TIME):
        """Initialize the batch processor."""
        self.process_func = process_func
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.items: List[T] = []
        self.results: Dict[int, U] = {}
        self.last_processed = time.time()
        self.lock = threading.Lock()
        self.processing = False
        self.next_item_id = 0
        
        # Start background processing
        self.stop_event = threading.Event()
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
    
    def _process_loop(self):
        """Background loop to process batches."""
        while not self.stop_event.is_set():
            # Check if we should process a batch
            with self.lock:
                current_time = time.time()
                time_since_last = current_time - self.last_processed
                should_process = (
                    len(self.items) >= self.max_batch_size or
                    (len(self.items) > 0 and time_since_last >= self.max_wait_time)
                )
                
                if should_process and not self.processing:
                    # Get items to process
                    items_to_process = self.items.copy()
                    item_ids = list(range(
                        self.next_item_id - len(items_to_process),
                        self.next_item_id
                    ))
                    self.items = []
                    self.processing = True
                else:
                    items_to_process = []
                    item_ids = []
            
            # Process items if needed
            if items_to_process:
                try:
                    # Process the batch
                    results = self.process_func(items_to_process)
                    
                    # Store results
                    with self.lock:
                        for i, result in zip(item_ids, results):
                            self.results[i] = result
                        
                        self.last_processed = time.time()
                        self.processing = False
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    
                    # Mark as failed
                    with self.lock:
                        for i in item_ids:
                            self.results[i] = None
                        
                        self.last_processed = time.time()
                        self.processing = False
            
            # Sleep briefly
            time.sleep(0.01)
    
    def add_item(self, item: T) -> int:
        """Add an item to the batch and return its ID."""
        with self.lock:
            self.items.append(item)
            item_id = self.next_item_id
            self.next_item_id += 1
            return item_id
    
    def get_result(self, item_id: int, timeout: float = 30.0) -> Optional[U]:
        """Get the result for an item, waiting if necessary."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if item_id in self.results:
                    return self.results.pop(item_id)
            
            # Sleep briefly
            time.sleep(0.05)
        
        # Timeout
        logger.warning(f"Timeout waiting for item {item_id}")
        return None
    
    def stop(self):
        """Stop the batch processor."""
        self.stop_event.set()
        self.process_thread.join(timeout=1.0)


class RequestLimiter:
    """Limit the number of concurrent requests."""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_REQUESTS):
        """Initialize the request limiter."""
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def acquire(self):
        """Acquire a request slot."""
        await self.semaphore.acquire()
    
    def release(self):
        """Release a request slot."""
        self.semaphore.release()


# Global request limiter
request_limiter = RequestLimiter()


async def with_concurrency_limit(func, *args, **kwargs):
    """Execute a function with concurrency limiting."""
    await request_limiter.acquire()
    try:
        return await func(*args, **kwargs)
    finally:
        request_limiter.release()


class ModelLoadBalancer:
    """Load balancer for distributing requests across model replicas."""
    
    def __init__(self, model_id: str, create_replica_func: Callable, max_replicas: int = 3):
        """Initialize the load balancer."""
        self.model_id = model_id
        self.create_replica_func = create_replica_func
        self.max_replicas = max_replicas
        self.replicas = []
        self.request_counts = []
        self.lock = threading.RLock()
    
    def initialize(self):
        """Initialize with one replica."""
        with self.lock:
            if not self.replicas:
                replica = self.create_replica_func()
                self.replicas.append(replica)
                self.request_counts.append(0)
    
    def get_replica(self):
        """Get a replica using least connections algorithm."""
        with self.lock:
            if not self.replicas:
                self.initialize()
            
            # Find the replica with the fewest requests
            min_requests = min(self.request_counts)
            candidates = [i for i, count in enumerate(self.request_counts) if count == min_requests]
            
            # Choose randomly from candidates
            replica_idx = random.choice(candidates)
            
            # Increment request count
            self.request_counts[replica_idx] += 1
            
            # Lazily scale up if all replicas are busy
            if (
                min_requests > 5 and 
                len(self.replicas) < self.max_replicas and
                all(count > 0 for count in self.request_counts)
            ):
                try:
                    logger.info(f"Scaling up {self.model_id} with new replica")
                    new_replica = self.create_replica_func()
                    self.replicas.append(new_replica)
                    self.request_counts.append(0)
                except Exception as e:
                    logger.error(f"Error creating new replica: {e}")
            
            return self.replicas[replica_idx], replica_idx
    
    def release_replica(self, replica_idx: int):
        """Release a replica after use."""
        with self.lock:
            if 0 <= replica_idx < len(self.request_counts):
                self.request_counts[replica_idx] = max(0, self.request_counts[replica_idx] - 1)
    
    def scale_down(self):
        """Scale down by removing idle replicas."""
        with self.lock:
            if len(self.replicas) <= 1:
                return  # Keep at least one replica
            
            # Find idle replicas
            idle_indices = [i for i, count in enumerate(self.request_counts) if count == 0]
            
            # Remove one idle replica if there are any
            if idle_indices:
                idx_to_remove = idle_indices[0]
                logger.info(f"Scaling down {self.model_id} by removing idle replica")
                del self.replicas[idx_to_remove]
                del self.request_counts[idx_to_remove]


# Initialize a thread pool
def run_in_thread_pool(func, *args, **kwargs):
    """Run a function in the thread pool."""
    return THREAD_POOL.submit(func, *args, **kwargs)


# Batch embedding function creator
def create_batch_embedder(embed_func):
    """Create a batch embedder that processes items in batches."""
    processor = BatchProcessor(embed_func)
    
    async def batch_embed(texts, model_id=None):
        """Add texts to the batch processor and wait for results."""
        item_id = processor.add_item(texts)
        return processor.get_result(item_id)
    
    return batch_embed 