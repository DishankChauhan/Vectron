"""
Vector database module for storing and retrieving embeddings.
Supports CRUD operations and similarity search.
"""

import logging
import os
import json
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import uuid
import sqlite3
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """Types of vector indices supported."""
    FLAT = "flat"  # Exact nearest neighbors
    IVF = "ivf"    # Inverted file index for faster but approximate search
    HNSW = "hnsw"  # Hierarchical Navigable Small World for very fast search


class DistanceMetric(str, Enum):
    """Distance metrics for similarity search."""
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


@dataclass
class VectorCollectionConfig:
    """Configuration for a vector collection."""
    name: str
    dimension: int
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    index_type: IndexType = IndexType.FLAT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorItem:
    """A single vector item stored in the database."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    collection: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class VectorStore:
    """Vector database for storing and retrieving embeddings."""
    
    def __init__(self, data_dir: str = None):
        """Initialize the vector store."""
        self.data_dir = Path(data_dir or os.environ.get("VECTOR_DB_PATH", "./data/vectors"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create SQLite database for metadata
        self.db_path = self.data_dir / "vector_metadata.db"
        self._init_db()
        
        # In-memory cache of vector collections
        self.collections: Dict[str, Dict[str, VectorItem]] = {}
        self.configs: Dict[str, VectorCollectionConfig] = {}
        self.indices: Dict[str, Any] = {}
        
        # Load existing collections
        self._load_collections()
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create collections table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS collections (
            name TEXT PRIMARY KEY,
            config TEXT NOT NULL
        )
        ''')
        
        # Create vectors table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vectors (
            id TEXT PRIMARY KEY,
            collection TEXT NOT NULL,
            metadata TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (collection) REFERENCES collections(name)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_collections(self):
        """Load all collections from disk."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load collection configs
        cursor.execute("SELECT name, config FROM collections")
        for name, config_json in cursor.fetchall():
            config = VectorCollectionConfig(**json.loads(config_json))
            self.configs[name] = config
            self.collections[name] = {}
            
            # Load vectors
            vectors_path = self.data_dir / name / "vectors.npy"
            if vectors_path.exists():
                # Load vector data
                try:
                    vector_file = self.data_dir / name / "vectors.npy"
                    vector_data = np.load(vector_file, allow_pickle=True).item()
                    
                    # Convert to VectorItem objects
                    cursor.execute("SELECT id, metadata, created_at FROM vectors WHERE collection = ?", (name,))
                    for vector_id, metadata_json, created_at in cursor.fetchall():
                        if vector_id in vector_data:
                            vector = vector_data[vector_id].tolist()
                            metadata = json.loads(metadata_json)
                            self.collections[name][vector_id] = VectorItem(
                                id=vector_id,
                                vector=vector,
                                metadata=metadata,
                                collection=name,
                                created_at=created_at
                            )
                    
                    # Build index
                    self._build_index(name)
                    logger.info(f"Loaded collection {name} with {len(self.collections[name])} vectors")
                except Exception as e:
                    logger.error(f"Error loading collection {name}: {e}")
        
        conn.close()
    
    def _build_index(self, collection_name: str):
        """Build a search index for a collection."""
        with self.lock:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist")
            
            config = self.configs[collection_name]
            vectors = self.collections[collection_name]
            
            if not vectors:
                logger.info(f"No vectors in collection {collection_name}, skipping index build")
                return
            
            # Extract vectors as numpy array
            vector_ids = list(vectors.keys())
            vector_array = np.array([vectors[vid].vector for vid in vector_ids])
            
            # Build index based on type
            if config.index_type == IndexType.FLAT:
                # Simple in-memory numpy index
                self.indices[collection_name] = {
                    "type": IndexType.FLAT,
                    "vectors": vector_array,
                    "ids": vector_ids,
                    "metric": config.distance_metric
                }
                logger.info(f"Built FLAT index for collection {collection_name}")
            
            elif config.index_type == IndexType.IVF:
                # For IVF, we'd normally use FAISS here
                # This is a simplified version
                self.indices[collection_name] = {
                    "type": IndexType.IVF,
                    "vectors": vector_array,
                    "ids": vector_ids,
                    "metric": config.distance_metric
                }
                logger.info(f"Built IVF index for collection {collection_name}")
            
            elif config.index_type == IndexType.HNSW:
                # For HNSW, we'd normally use hnswlib or FAISS here
                # This is a simplified version
                self.indices[collection_name] = {
                    "type": IndexType.HNSW,
                    "vectors": vector_array,
                    "ids": vector_ids,
                    "metric": config.distance_metric
                }
                logger.info(f"Built HNSW index for collection {collection_name}")
    
    def _save_collection(self, collection_name: str):
        """Save a collection to disk."""
        with self.lock:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist")
            
            # Create collection directory
            collection_dir = self.data_dir / collection_name
            collection_dir.mkdir(exist_ok=True)
            
            # Save vectors
            vectors = self.collections[collection_name]
            vector_data = {vid: np.array(v.vector) for vid, v in vectors.items()}
            np.save(collection_dir / "vectors.npy", vector_data)
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save or update collection config
            config_json = json.dumps(asdict(self.configs[collection_name]))
            cursor.execute(
                "INSERT OR REPLACE INTO collections (name, config) VALUES (?, ?)",
                (collection_name, config_json)
            )
            
            # Save or update vector metadata
            for vector_id, vector_item in vectors.items():
                metadata_json = json.dumps(vector_item.metadata)
                cursor.execute(
                    "INSERT OR REPLACE INTO vectors (id, collection, metadata, created_at) VALUES (?, ?, ?, ?)",
                    (vector_id, collection_name, metadata_json, vector_item.created_at)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved collection {collection_name} with {len(vectors)} vectors")
    
    def create_collection(self, name: str, dimension: int, 
                         distance_metric: DistanceMetric = DistanceMetric.COSINE,
                         index_type: IndexType = IndexType.FLAT,
                         metadata: Dict[str, Any] = None) -> VectorCollectionConfig:
        """Create a new vector collection."""
        with self.lock:
            if name in self.collections:
                raise ValueError(f"Collection {name} already exists")
            
            # Create config
            config = VectorCollectionConfig(
                name=name,
                dimension=dimension,
                distance_metric=distance_metric,
                index_type=index_type,
                metadata=metadata or {}
            )
            
            # Initialize collection
            self.configs[name] = config
            self.collections[name] = {}
            
            # Save to disk
            self._save_collection(name)
            
            return config
    
    def list_collections(self) -> List[VectorCollectionConfig]:
        """List all vector collections."""
        return list(self.configs.values())
    
    def get_collection(self, name: str) -> Optional[VectorCollectionConfig]:
        """Get a collection by name."""
        return self.configs.get(name)
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        with self.lock:
            if name not in self.collections:
                return False
            
            # Delete from memory
            del self.collections[name]
            del self.configs[name]
            if name in self.indices:
                del self.indices[name]
            
            # Delete from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vectors WHERE collection = ?", (name,))
            cursor.execute("DELETE FROM collections WHERE name = ?", (name,))
            conn.commit()
            conn.close()
            
            # Delete files
            collection_dir = self.data_dir / name
            if collection_dir.exists():
                import shutil
                shutil.rmtree(collection_dir)
            
            logger.info(f"Deleted collection {name}")
            return True
    
    def add_item(self, collection_name: str, vector: List[float], 
                metadata: Dict[str, Any] = None, item_id: str = None) -> str:
        """Add an item to a collection."""
        with self.lock:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist")
            
            config = self.configs[collection_name]
            if len(vector) != config.dimension:
                raise ValueError(f"Vector dimension {len(vector)} does not match collection dimension {config.dimension}")
            
            # Generate ID if not provided
            if item_id is None:
                item_id = str(uuid.uuid4())
            
            # Create vector item
            vector_item = VectorItem(
                id=item_id,
                vector=vector,
                metadata=metadata or {},
                collection=collection_name
            )
            
            # Add to collection
            self.collections[collection_name][item_id] = vector_item
            
            # Update index
            self._build_index(collection_name)
            
            # Save to disk
            self._save_collection(collection_name)
            
            return item_id
    
    def add_items(self, collection_name: str, vectors: List[List[float]], 
                 metadatas: List[Dict[str, Any]] = None, 
                 ids: List[str] = None) -> List[str]:
        """Add multiple items to a collection."""
        with self.lock:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist")
            
            if not vectors:
                return []
            
            # Normalize inputs
            metadatas = metadatas or [{}] * len(vectors)
            ids = ids or [str(uuid.uuid4()) for _ in range(len(vectors))]
            
            if len(vectors) != len(metadatas) or len(vectors) != len(ids):
                raise ValueError("Vectors, metadatas, and ids must have the same length")
            
            # Add each item
            for i, (vector, metadata, item_id) in enumerate(zip(vectors, metadatas, ids)):
                self.add_item(collection_name, vector, metadata, item_id)
            
            return ids
    
    def get_item(self, collection_name: str, item_id: str) -> Optional[VectorItem]:
        """Get an item by ID."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        
        return self.collections[collection_name].get(item_id)
    
    def update_item(self, collection_name: str, item_id: str, 
                   vector: List[float] = None, 
                   metadata: Dict[str, Any] = None) -> bool:
        """Update an item."""
        with self.lock:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist")
            
            if item_id not in self.collections[collection_name]:
                return False
            
            item = self.collections[collection_name][item_id]
            
            # Update vector if provided
            if vector is not None:
                config = self.configs[collection_name]
                if len(vector) != config.dimension:
                    raise ValueError(f"Vector dimension {len(vector)} does not match collection dimension {config.dimension}")
                item.vector = vector
            
            # Update metadata if provided
            if metadata is not None:
                item.metadata = metadata
            
            # Update index
            self._build_index(collection_name)
            
            # Save to disk
            self._save_collection(collection_name)
            
            return True
    
    def delete_item(self, collection_name: str, item_id: str) -> bool:
        """Delete an item."""
        with self.lock:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist")
            
            if item_id not in self.collections[collection_name]:
                return False
            
            # Remove from collection
            del self.collections[collection_name][item_id]
            
            # Update index
            self._build_index(collection_name)
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vectors WHERE id = ?", (item_id,))
            conn.commit()
            conn.close()
            
            # Save to disk
            self._save_collection(collection_name)
            
            return True
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray, metric: DistanceMetric) -> float:
        """Compute similarity between two vectors."""
        if metric == DistanceMetric.COSINE:
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return np.dot(vec1, vec2) / (norm1 * norm2)
        
        elif metric == DistanceMetric.DOT:
            # Dot product
            return np.dot(vec1, vec2)
        
        elif metric == DistanceMetric.EUCLIDEAN:
            # Euclidean distance (converted to similarity)
            dist = np.linalg.norm(vec1 - vec2)
            return 1.0 / (1.0 + dist)
        
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    def search(self, collection_name: str, query_vector: List[float], 
              top_k: int = 10, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        with self.lock:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist")
            
            if collection_name not in self.indices:
                logger.warning(f"No index for collection {collection_name}")
                return []
            
            config = self.configs[collection_name]
            if len(query_vector) != config.dimension:
                raise ValueError(f"Query vector dimension {len(query_vector)} does not match collection dimension {config.dimension}")
            
            # Get index
            index = self.indices[collection_name]
            metric = index["metric"]
            
            # Convert query to numpy
            query_np = np.array(query_vector)
            
            # Compute similarities
            vectors = index["vectors"]
            ids = index["ids"]
            
            if len(vectors) == 0:
                return []
            
            similarities = []
            for i, vec in enumerate(vectors):
                item_id = ids[i]
                item = self.collections[collection_name][item_id]
                
                # Apply filter if provided
                if filter and not self._matches_filter(item.metadata, filter):
                    continue
                
                # Compute similarity
                similarity = self._compute_similarity(query_np, vec, metric)
                similarities.append((item_id, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Limit to top_k
            similarities = similarities[:top_k]
            
            # Format results
            results = []
            for item_id, similarity in similarities:
                item = self.collections[collection_name][item_id]
                results.append({
                    "id": item.id,
                    "metadata": item.metadata,
                    "similarity": float(similarity)
                })
            
            return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches a filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                # If filter value is a list, check if metadata value is in the list
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        
        return True
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        
        vectors = self.collections[collection_name]
        config = self.configs[collection_name]
        
        return {
            "name": collection_name,
            "vector_count": len(vectors),
            "dimension": config.dimension,
            "distance_metric": config.distance_metric,
            "index_type": config.index_type,
            "created_at": config.created_at,
            "metadata": config.metadata
        }
    

# Create a global instance for use in the application
vector_store = VectorStore() 