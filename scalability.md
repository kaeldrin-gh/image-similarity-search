# Scalability Considerations

This document outlines strategies for scaling the image similarity search system for Phase 2 and beyond.

## Current Architecture Limitations

### Phase 1 (MVP) Constraints
- Single-threaded embedding extraction
- In-memory FAISS index (limited by RAM)
- No distributed processing
- ResNet-50 only (large model)
- No caching mechanism

## Phase 2 Scaling Strategies

### 1. Distributed Computing

#### Embedding Extraction
```python
# Parallel processing with multiprocessing
from multiprocessing import Pool
from functools import partial

def extract_embeddings_parallel(image_paths, num_workers=4):
    chunk_size = len(image_paths) // num_workers
    chunks = [image_paths[i:i+chunk_size] for i in range(0, len(image_paths), chunk_size)]
    
    with Pool(num_workers) as pool:
        embeddings = pool.map(extract_chunk, chunks)
    
    return np.vstack(embeddings)
```

#### Ray for Distributed Processing
```python
import ray

@ray.remote
class EmbeddingExtractor:
    def __init__(self, model_name):
        self.extractor = EmbeddingExtractor(model_name)
    
    def extract_batch(self, image_paths):
        return self.extractor.extract_embeddings(image_paths)

# Usage
ray.init()
workers = [EmbeddingExtractor.remote("resnet50") for _ in range(4)]
futures = [worker.extract_batch.remote(chunk) for worker, chunk in zip(workers, chunks)]
results = ray.get(futures)
```

### 2. Memory Optimization

#### Embedding Quantization
```python
import faiss

def build_quantized_index(embeddings, num_bits=8):
    """Build quantized FAISS index to reduce memory usage."""
    d = embeddings.shape[1]
    
    # Create PQ quantizer
    pq = faiss.IndexPQ(d, d // 4, num_bits)
    pq.train(embeddings)
    pq.add(embeddings)
    
    return pq
```

#### Streaming Processing
```python
def process_images_streaming(image_paths, batch_size=1000):
    """Process images in streaming batches to reduce memory usage."""
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        embeddings = extract_embeddings(batch)
        yield embeddings
```

### 3. Performance Optimization

#### GPU Acceleration
```python
def build_gpu_index(embeddings):
    """Build GPU-accelerated FAISS index."""
    res = faiss.StandardGpuResources()
    
    # Build flat index on GPU
    index_flat = faiss.IndexFlatL2(embeddings.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    
    gpu_index.add(embeddings)
    return gpu_index
```

#### Model Optimization
```python
# EfficientNet variants for better speed/accuracy trade-off
from torchvision.models import efficientnet_b0, efficientnet_b3

def get_efficient_model(variant="b0"):
    """Get EfficientNet model variant."""
    if variant == "b0":
        return efficientnet_b0(weights="IMAGENET1K_V1")
    elif variant == "b3":
        return efficientnet_b3(weights="IMAGENET1K_V1")
```

### 4. Caching Strategy

#### Redis Integration
```python
import redis
import pickle

class CachedEmbeddingExtractor:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.extractor = EmbeddingExtractor()
    
    def extract_with_cache(self, image_path):
        # Check cache first
        cached = self.redis_client.get(image_path)
        if cached:
            return pickle.loads(cached)
        
        # Extract and cache
        embedding = self.extractor.extract_single_embedding(image_path)
        self.redis_client.setex(image_path, 3600, pickle.dumps(embedding))
        return embedding
```

## Phase 3 Advanced Scaling

### 1. Microservices Architecture

```yaml
# docker-compose.yml
version: '3.8'
services:
  embedding-service:
    build: ./embedding-service
    ports:
      - "8001:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
  
  index-service:
    build: ./index-service
    ports:
      - "8002:8000"
    depends_on:
      - redis
  
  api-gateway:
    build: ./api-gateway
    ports:
      - "8000:8000"
    depends_on:
      - embedding-service
      - index-service
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### 2. Vector Database Integration

#### Milvus Integration
```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

def create_milvus_collection(collection_name, dim=2048):
    """Create Milvus collection for embeddings."""
    connections.connect("default", host="localhost", port="19530")
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500)
    ]
    
    schema = CollectionSchema(fields)
    collection = Collection(collection_name, schema)
    
    # Create index
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index("embedding", index_params)
    
    return collection
```

### 3. Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: embedding-service
        image: embedding-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
```

## Monitoring and Observability

### 1. Metrics Collection

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Metrics
EMBEDDING_REQUESTS = Counter('embedding_requests_total', 'Total embedding requests')
EMBEDDING_DURATION = Histogram('embedding_duration_seconds', 'Time spent extracting embeddings')
INDEX_SIZE = Gauge('index_size_total', 'Total number of vectors in index')

def extract_embeddings_with_metrics(image_paths):
    EMBEDDING_REQUESTS.inc()
    
    with EMBEDDING_DURATION.time():
        embeddings = extract_embeddings(image_paths)
    
    INDEX_SIZE.set(len(embeddings))
    return embeddings
```

### 2. Logging Strategy

```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def extract_embeddings_with_logging(image_paths):
    logger.info("Starting embedding extraction", num_images=len(image_paths))
    
    try:
        embeddings = extract_embeddings(image_paths)
        logger.info("Embedding extraction completed", 
                   num_images=len(image_paths), 
                   embedding_shape=embeddings.shape)
        return embeddings
    except Exception as e:
        logger.error("Embedding extraction failed", error=str(e))
        raise
```

## Performance Benchmarks

### Target Metrics (Phase 2)

| Metric | Current | Target |
|--------|---------|---------|
| Embedding Extraction | 100 images/min | 1000 images/min |
| Index Build Time | 10 min (100K images) | 2 min (100K images) |
| Query Response Time | 100ms | 10ms |
| Memory Usage | 8GB (100K images) | 2GB (100K images) |
| Throughput | 10 QPS | 1000 QPS |

### Optimization Strategies

1. **Batching**: Process images in optimal batch sizes
2. **Model Quantization**: Reduce model size while maintaining accuracy
3. **Index Optimization**: Use approximate indices for faster search
4. **Caching**: Cache frequently accessed embeddings
5. **Load Balancing**: Distribute requests across multiple instances

## Cost Optimization

### 1. Cloud Resource Management

```python
# Auto-scaling configuration
import boto3

def setup_auto_scaling(service_name, min_capacity=1, max_capacity=10):
    """Setup auto-scaling for ECS service."""
    client = boto3.client('application-autoscaling')
    
    # Register scalable target
    client.register_scalable_target(
        ServiceNamespace='ecs',
        ResourceId=f'service/{service_name}',
        ScalableDimension='ecs:service:DesiredCount',
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity
    )
    
    # Create scaling policy
    client.put_scaling_policy(
        PolicyName=f'{service_name}-cpu-scaling',
        ServiceNamespace='ecs',
        ResourceId=f'service/{service_name}',
        ScalableDimension='ecs:service:DesiredCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 70.0,
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'ECSServiceAverageCPUUtilization'
            }
        }
    )
```

### 2. Cost Monitoring

```python
def estimate_processing_cost(num_images, gpu_hours_per_1k_images=0.1):
    """Estimate cost for processing images."""
    gpu_cost_per_hour = 0.90  # AWS p3.2xlarge
    
    total_gpu_hours = (num_images / 1000) * gpu_hours_per_1k_images
    total_cost = total_gpu_hours * gpu_cost_per_hour
    
    return {
        'total_images': num_images,
        'gpu_hours': total_gpu_hours,
        'estimated_cost': total_cost
    }
```

## Next Steps

1. **Implement distributed processing** using Ray or Dask
2. **Add GPU support** for faster embedding extraction
3. **Integrate vector databases** like Milvus or Pinecone
4. **Implement caching layer** with Redis
5. **Add monitoring and alerting** with Prometheus/Grafana
6. **Deploy on Kubernetes** for better scalability
7. **Implement cost optimization** strategies

This scalability roadmap provides a clear path from the current MVP to a production-ready, scalable system capable of handling millions of images and thousands of queries per second.
