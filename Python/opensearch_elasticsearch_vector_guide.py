"""
OpenSearch & Elasticsearch for Vector Search - Complete Guide
Learn how to use OpenSearch and Elasticsearch for similarity search
Covers setup, indexing, querying, and production deployment
"""

print("=" * 80)
print("OPENSEARCH & ELASTICSEARCH - VECTOR SEARCH MASTERY")
print("=" * 80)

# ============================================================================
# SECTION 1: OPENSEARCH FUNDAMENTALS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 1: OPENSEARCH FUNDAMENTALS")
print("=" * 60)

# Question 1: What is OpenSearch and Why Use It?
print("\n1. What is OpenSearch and why use it for vector search?")
print("-" * 59)
print("""
OPENSEARCH OVERVIEW:
OpenSearch is an open-source search and analytics engine derived from 
Elasticsearch. It provides powerful full-text search, analytics, and 
vector search capabilities.

KEY FEATURES:
🔍 Full-text Search: Traditional keyword-based search
🧠 Vector Search: k-NN similarity search with multiple algorithms
📊 Analytics: Real-time data analytics and aggregations
🔄 Real-time: Near real-time indexing and search
🎯 Multi-modal: Support for text, images, and other data types
⚡ Performance: Distributed, scalable architecture

VECTOR SEARCH CAPABILITIES:
• k-NN Plugin: Built-in vector similarity search
• Multiple Algorithms: Exact, LSH, HNSW, IVF
• Distance Metrics: Cosine, Euclidean, Manhattan, Hamming
• Filtering: Combine vector search with traditional filters
• Hybrid Search: Mix vector and text search results

OPENSEARCH vs ELASTICSEARCH:
Feature               | OpenSearch    | Elasticsearch
---------------------|---------------|---------------
License              | Apache 2.0    | Elastic License
Vector Search        | Built-in      | Dense Vector (paid)
Community            | Growing       | Established
Commercial Support   | AWS           | Elastic
Cost                 | Free          | Paid features

WHY CHOOSE OPENSEARCH FOR VECTOR SEARCH?
✅ Free and open-source
✅ AWS managed service available
✅ Strong vector search capabilities
✅ Active development community
✅ Enterprise-grade features
✅ Excellent documentation

REAL-WORLD USE CASES:
🛍️ E-commerce: Product recommendations
🎵 Media: Content similarity matching
📚 Knowledge Management: Document search
🔍 Search Engines: Semantic search
🏥 Healthcare: Similar case finding
🔒 Security: Anomaly detection

INTERVIEW QUESTIONS:
Q: "Difference between OpenSearch and Elasticsearch?"
A: "OpenSearch is open-source fork of Elasticsearch 7.10.2, 
   maintained by AWS, with free vector search capabilities."

Q: "When would you choose OpenSearch over PostgreSQL+pgvector?"
A: "When you need full-text search combined with vector search,
   real-time analytics, or managing very large datasets with
   horizontal scaling requirements."
""")

def demonstrate_opensearch_concepts():
    """Demonstrate OpenSearch vector search concepts"""
    print("OpenSearch Vector Search Concepts Demo:")
    
    print("\n1. OpenSearch Architecture Overview:")
    print("-" * 40)
    print("""
    OPENSEARCH CLUSTER ARCHITECTURE:
    
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │   Master Node   │  │   Data Node 1   │  │   Data Node 2   │
    │                 │  │                 │  │                 │
    │ • Cluster state │  │ • Store data    │  │ • Store data    │
    │ • Index mgmt    │  │ • Execute       │  │ • Execute       │
    │ • Node mgmt     │  │   queries       │  │   queries       │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                    ┌─────────────────────────────┐
                    │      Load Balancer          │
                    │                             │
                    │ • Route requests            │
                    │ • Health monitoring         │
                    └─────────────────────────────┘
    
    VECTOR SEARCH FLOW:
    1. Documents indexed with vectors
    2. Vector embeddings stored in specialized fields
    3. k-NN queries executed across shards
    4. Results merged and ranked
    5. Combined with traditional search if needed
    """)
    
    print("\n2. Index Structure for Vector Search:")
    print("-" * 42)
    print("""
    INDEX MAPPING EXAMPLE:
    {
      "mappings": {
        "properties": {
          "title": {"type": "text"},
          "content": {"type": "text"},
          "category": {"type": "keyword"},
          "timestamp": {"type": "date"},
          "embedding": {
            "type": "knn_vector",
            "dimension": 384,
            "method": {
              "name": "hnsw",
              "space_type": "cosinesimil",
              "engine": "nmslib",
              "parameters": {
                "ef_construction": 128,
                "m": 24
              }
            }
          }
        }
      }
    }
    
    KEY COMPONENTS:
    • knn_vector: Special field type for vectors
    • dimension: Vector dimensionality (must match embeddings)
    • method: Algorithm for similarity search
    • space_type: Distance metric (cosinesimil, l2, l1, etc.)
    • engine: Vector library (nmslib, faiss, lucene)
    """)

demonstrate_opensearch_concepts()

# Question 2: OpenSearch Setup and Configuration
print("\n2. How to set up OpenSearch for vector search?")
print("-" * 50)
print("""
OPENSEARCH SETUP GUIDE:

INSTALLATION OPTIONS:
1. Docker Compose (Development)
2. AWS OpenSearch Service (Managed)
3. Self-hosted cluster (Production)
4. Local development setup

SYSTEM REQUIREMENTS:
• RAM: Minimum 4GB, recommended 16GB+
• CPU: Multi-core recommended
• Storage: SSD for better performance
• Network: Low latency between nodes

CONFIGURATION FILES:
• opensearch.yml: Main configuration
• jvm.options: JVM heap settings
• log4j2.properties: Logging configuration

SECURITY SETUP:
• Enable security plugin
• Configure authentication
• Set up TLS/SSL
• Define role-based access

PERFORMANCE TUNING:
• Heap size: 50% of available RAM
• Thread pools: CPU core count
• Refresh interval: Based on use case
• Replica settings: For fault tolerance
""")

def demonstrate_opensearch_setup():
    """Demonstrate OpenSearch setup and configuration"""
    print("OpenSearch Setup and Configuration Demo:")
    
    print("\n1. Docker Compose Setup:")
    print("-" * 30)
    print("""
# docker-compose.yml
version: '3.7'

services:
  opensearch-node1:
    image: opensearchproject/opensearch:2.11.0
    container_name: opensearch-node1
    environment:
      - cluster.name=opensearch-cluster
      - node.name=opensearch-node1
      - discovery.seed_hosts=opensearch-node1,opensearch-node2
      - cluster.initial_cluster_state=cluster_bootstrap_timeout=30s
      - bootstrap.memory_lock=true
      - "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"
      - "DISABLE_INSTALL_DEMO_CONFIG=true"
      - "DISABLE_SECURITY_PLUGIN=true"
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536
    volumes:
      - opensearch-data1:/usr/share/opensearch/data
    ports:
      - 9200:9200
      - 9600:9600
    networks:
      - opensearch-net

  opensearch-node2:
    image: opensearchproject/opensearch:2.11.0
    container_name: opensearch-node2
    environment:
      - cluster.name=opensearch-cluster
      - node.name=opensearch-node2
      - discovery.seed_hosts=opensearch-node1,opensearch-node2
      - cluster.initial_cluster_state=cluster_bootstrap_timeout=30s
      - bootstrap.memory_lock=true
      - "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"
      - "DISABLE_INSTALL_DEMO_CONFIG=true"
      - "DISABLE_SECURITY_PLUGIN=true"
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536
    volumes:
      - opensearch-data2:/usr/share/opensearch/data
    networks:
      - opensearch-net

  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:2.11.0
    container_name: opensearch-dashboards
    ports:
      - 5601:5601
    expose:
      - "5601"
    environment:
      - 'OPENSEARCH_HOSTS=["http://opensearch-node1:9200","http://opensearch-node2:9200"]'
      - "DISABLE_SECURITY_DASHBOARDS_PLUGIN=true"
    networks:
      - opensearch-net

volumes:
  opensearch-data1:
  opensearch-data2:

networks:
  opensearch-net:
    """)
    
    print("\n2. Python Client Setup:")
    print("-" * 28)
    print("""
# Installation
pip install opensearch-py sentence-transformers

# Basic client configuration
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import json

class OpenSearchVectorClient:
    def __init__(self, host='localhost', port=9200):
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Model embedding dimension
    
    def create_index(self, index_name):
        '''Create index with vector field mapping'''
        mapping = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "category": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    }
                }
            }
        }
        
        try:
            response = self.client.indices.create(
                index=index_name,
                body=mapping
            )
            print(f"Index '{index_name}' created successfully")
            return response
        except Exception as e:
            print(f"Error creating index: {e}")
            return None
    
    def index_document(self, index_name, doc_id, title, content, category=None):
        '''Index a document with its vector embedding'''
        # Generate embedding
        embedding = self.model.encode(content).tolist()
        
        document = {
            "title": title,
            "content": content,
            "category": category,
            "timestamp": "now",
            "embedding": embedding
        }
        
        try:
            response = self.client.index(
                index=index_name,
                id=doc_id,
                body=document
            )
            return response
        except Exception as e:
            print(f"Error indexing document: {e}")
            return None
    
    def search_similar(self, index_name, query_text, k=10, filter_category=None):
        '''Search for similar documents using vector similarity'''
        # Generate query embedding
        query_embedding = self.model.encode(query_text).tolist()
        
        # Build search query
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [{
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": k
                            }
                        }
                    }]
                }
            }
        }
        
        # Add category filter if specified
        if filter_category:
            search_body["query"]["bool"]["filter"] = [{
                "term": {"category": filter_category}
            }]
        
        try:
            response = self.client.search(
                index=index_name,
                body=search_body
            )
            return response
        except Exception as e:
            print(f"Error searching: {e}")
            return None
    
    def hybrid_search(self, index_name, query_text, k=10):
        '''Combine vector search with text search'''
        query_embedding = self.model.encode(query_text).tolist()
        
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": k,
                                    "boost": 1.0
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["title^2", "content"],
                                "boost": 0.5
                            }
                        }
                    ]
                }
            }
        }
        
        return self.client.search(index=index_name, body=search_body)
    """)

demonstrate_opensearch_setup()

# ============================================================================
# SECTION 2: ELASTICSEARCH VECTOR SEARCH
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 2: ELASTICSEARCH VECTOR SEARCH")
print("=" * 60)

# Question 3: Elasticsearch vs OpenSearch
print("\n3. How does Elasticsearch compare to OpenSearch for vector search?")
print("-" * 70)
print("""
ELASTICSEARCH VECTOR SEARCH:

DENSE VECTOR FIELD TYPE:
Elasticsearch introduced dense_vector field type for vector search,
but advanced features require paid subscription.

ELASTICSEARCH VERSIONS:
• 7.x: Basic dense_vector support
• 8.x: Enhanced vector search capabilities
• 8.8+: Advanced k-NN algorithms (paid)

VECTOR SEARCH FEATURES:
Free Tier:
✅ Basic dense_vector field
✅ Script-based similarity search
✅ Cosine similarity queries
❌ Optimized k-NN algorithms
❌ HNSW indexing
❌ Advanced distance metrics

Paid Tier (Platinum/Enterprise):
✅ Optimized k-NN search
✅ HNSW algorithm
✅ Multiple distance metrics
✅ Vector similarity functions
✅ k-NN aggregations

COMPARISON MATRIX:
Feature                 | OpenSearch | Elasticsearch Free | Elasticsearch Paid
-----------------------|------------|-------------------|-------------------
Vector Field Type      | knn_vector | dense_vector      | dense_vector
k-NN Algorithms        | ✅ Multiple | ❌ Script only     | ✅ HNSW
Distance Metrics       | ✅ All      | ✅ Cosine only    | ✅ All
Performance            | ✅ High     | ⭐⭐ Slow         | ✅ High
Cost                   | Free       | Free              | Paid subscription
Scalability           | ✅ Excellent| ⭐⭐ Limited      | ✅ Excellent

WHEN TO CHOOSE ELASTICSEARCH:
✅ Already using Elasticsearch stack
✅ Budget for paid license
✅ Need Elastic Cloud managed service
✅ Require enterprise support
✅ Integration with other Elastic products

WHEN TO CHOOSE OPENSEARCH:
✅ Cost-sensitive projects
✅ Open-source requirement
✅ AWS ecosystem integration
✅ Advanced vector search without licensing
✅ Community-driven development
""")

def demonstrate_elasticsearch_vector_search():
    """Demonstrate Elasticsearch vector search implementation"""
    print("Elasticsearch Vector Search Implementation:")
    
    print("\n1. Elasticsearch Free Tier Implementation:")
    print("-" * 44)
    print("""
from elasticsearch import Elasticsearch
import numpy as np

class ElasticsearchVectorSearch:
    def __init__(self, host='localhost', port=9200):
        self.client = Elasticsearch([f'http://{host}:{port}'])
        
    def create_index_free_tier(self, index_name):
        '''Create index for free tier vector search'''
        mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "category": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384
                    }
                }
            }
        }
        
        return self.client.indices.create(index=index_name, body=mapping)
    
    def search_vector_free_tier(self, index_name, query_vector, k=10):
        '''Vector search using script scoring (free tier)'''
        search_body = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }
        
        return self.client.search(index=index_name, body=search_body)
    
    def create_index_paid_tier(self, index_name):
        '''Create index for paid tier with k-NN'''
        mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "category": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        
        return self.client.indices.create(index=index_name, body=mapping)
    
    def search_vector_paid_tier(self, index_name, query_vector, k=10):
        '''Vector search using k-NN (paid tier)'''
        search_body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": 100
            }
        }
        
        return self.client.search(index=index_name, body=search_body)
    """)
    
    print("\n2. Performance Comparison:")
    print("-" * 32)
    print("""
    ELASTICSEARCH VECTOR SEARCH PERFORMANCE:
    
    Method              | Query Time | Accuracy | Memory Usage | Cost
    -------------------|------------|----------|--------------|--------
    Script Score       | ~200ms     | 100%     | High         | Free
    k-NN (Paid)        | ~20ms      | 99%+     | Optimized    | $$$$
    OpenSearch k-NN    | ~25ms      | 99%+     | Optimized    | Free
    
    SCALABILITY COMPARISON:
    Dataset Size     | Script Score | k-NN (Paid) | OpenSearch k-NN
    ----------------|--------------|--------------|----------------
    10K vectors     | 50ms         | 5ms          | 8ms
    100K vectors    | 500ms        | 15ms         | 20ms
    1M vectors      | 5000ms       | 50ms         | 60ms
    10M vectors     | Timeout      | 200ms        | 250ms
    
    OBSERVATION:
    Script-based search doesn't scale well beyond 100K vectors.
    For production vector search, use optimized k-NN algorithms.
    """)

demonstrate_elasticsearch_vector_search()

# ============================================================================
# SECTION 3: PRODUCTION DEPLOYMENT PATTERNS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 3: PRODUCTION DEPLOYMENT PATTERNS")
print("=" * 60)

# Question 4: Production Architecture
print("\n4. How to architect OpenSearch/Elasticsearch for production vector search?")
print("-" * 79)
print("""
PRODUCTION ARCHITECTURE BEST PRACTICES:

CLUSTER DESIGN:
🏗️ Node Types:
• Master Nodes: Cluster management (3 nodes for HA)
• Data Nodes: Store and search data (scale horizontally)
• Ingest Nodes: Data preprocessing (optional)
• Coordinating Nodes: Query routing (load balancers)

📊 Capacity Planning:
• Vector Storage: ~4KB per 1000-dim vector
• Memory: 64GB+ per data node recommended
• CPU: 16+ cores for vector computations
• Storage: NVMe SSD for optimal performance

🔄 Data Architecture:
• Hot-Warm-Cold tiers for time-based data
• Index templates for consistent mappings
• Rollover policies for index management
• Replica configuration for fault tolerance

SCALING STRATEGIES:
📈 Horizontal Scaling:
• Add data nodes to increase capacity
• Distribute shards across nodes evenly
• Use routing for data locality
• Monitor shard sizes (aim for 10-50GB)

⚡ Performance Optimization:
• Optimize JVM heap (50% of RAM max)
• Tune refresh intervals based on use case
• Use bulk operations for indexing
• Implement caching strategies

🔒 Security & Monitoring:
• Enable authentication and authorization
• Set up TLS/SSL encryption
• Monitor cluster health and performance
• Implement alerting for critical metrics

DEPLOYMENT OPTIONS:
1. Self-managed clusters
2. AWS OpenSearch Service
3. Elastic Cloud (for Elasticsearch)
4. Kubernetes deployments
""")

def demonstrate_production_architecture():
    """Demonstrate production architecture and deployment"""
    print("Production Architecture and Deployment:")
    
    print("\n1. Production Cluster Configuration:")
    print("-" * 42)
    print("""
# opensearch.yml for production
cluster.name: "production-vector-search"
node.name: "data-node-1"
node.roles: ["data", "ingest"]

# Network settings
network.host: 0.0.0.0
http.port: 9200
transport.port: 9300

# Discovery settings
discovery.seed_hosts: ["10.0.1.1", "10.0.1.2", "10.0.1.3"]
cluster.initial_master_nodes: ["master-1", "master-2", "master-3"]

# Memory and performance
bootstrap.memory_lock: true
indices.memory.index_buffer_size: 30%
indices.breaker.total.limit: 95%

# Vector search specific settings
knn.algo_param.ef_search: 100
knn.memory.circuit_breaker.enabled: true
knn.memory.circuit_breaker.limit: 50%

# Security
plugins.security.disabled: false
plugins.security.ssl.http.enabled: true
plugins.security.ssl.transport.enabled: true
    """)
    
    print("\n2. Index Templates for Vector Data:")
    print("-" * 38)
    print("""
PUT _index_template/vector_documents
{
  "index_patterns": ["documents-*"],
  "template": {
    "settings": {
      "number_of_shards": 3,
      "number_of_replicas": 1,
      "index.knn": true,
      "index.knn.algo_param.ef_search": 100,
      "refresh_interval": "5s"
    },
    "mappings": {
      "properties": {
        "title": {
          "type": "text",
          "analyzer": "standard"
        },
        "content": {
          "type": "text",
          "analyzer": "standard"
        },
        "category": {
          "type": "keyword"
        },
        "timestamp": {
          "type": "date"
        },
        "embedding": {
          "type": "knn_vector",
          "dimension": 384,
          "method": {
            "name": "hnsw",
            "space_type": "cosinesimil",
            "engine": "nmslib",
            "parameters": {
              "ef_construction": 128,
              "m": 24
            }
          }
        },
        "tags": {
          "type": "keyword"
        }
      }
    }
  },
  "priority": 500
}
    """)
    
    print("\n3. Monitoring and Alerting:")
    print("-" * 34)
    print("""
# Key metrics to monitor
CLUSTER HEALTH METRICS:
• Cluster status (green/yellow/red)
• Node availability and connectivity
• Shard allocation and balance
• JVM heap usage (<85%)
• CPU utilization (<80%)
• Disk space usage (<85%)

VECTOR SEARCH METRICS:
• k-NN query latency (p95, p99)
• k-NN memory usage
• Vector indexing rate
• Search throughput (QPS)
• Cache hit ratios

ALERTING RULES:
• Cluster status != green for >5 minutes
• JVM heap usage >90% for >2 minutes
• Search latency p95 >500ms for >5 minutes
• Available disk space <15%
• Node disconnection

# Prometheus alerting example
groups:
- name: opensearch-alerts
  rules:
  - alert: OpenSearchClusterRed
    expr: opensearch_cluster_status != 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "OpenSearch cluster status is red"
      
  - alert: OpenSearchHighHeapUsage
    expr: opensearch_jvm_heap_used_percent > 90
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "OpenSearch JVM heap usage is high"
    """)

demonstrate_production_architecture()

# ============================================================================
# SECTION 4: ADVANCED VECTOR SEARCH PATTERNS
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 4: ADVANCED VECTOR SEARCH PATTERNS")
print("=" * 60)

# Question 5: Advanced Use Cases
print("\n5. What are advanced vector search patterns in OpenSearch/Elasticsearch?")
print("-" * 77)
print("""
ADVANCED VECTOR SEARCH PATTERNS:

🔍 HYBRID SEARCH:
Combine vector similarity with traditional text search for better results.
Use cases: E-commerce product search, document retrieval

🎯 FILTERED VECTOR SEARCH:
Apply filters while maintaining vector search performance.
Techniques: Pre-filtering, post-filtering, hybrid approaches

📊 MULTI-VECTOR SEARCH:
Search across multiple embedding types (text, image, audio).
Implementation: Multiple vector fields in same document

🔄 REAL-TIME VECTOR UPDATES:
Handle dynamic vector updates with minimal performance impact.
Strategies: Incremental updates, versioning, background reindexing

📈 APPROXIMATE SEARCH TUNING:
Balance accuracy vs performance for specific use cases.
Parameters: ef_search, ef_construction, m parameter

🧮 VECTOR AGGREGATIONS:
Aggregate and analyze vector search results.
Use cases: Clustering, trend analysis, recommendation scoring

⚡ PERFORMANCE OPTIMIZATION:
Advanced techniques for large-scale vector search.
Methods: Sharding strategies, caching, query optimization
""")

def demonstrate_advanced_patterns():
    """Demonstrate advanced vector search patterns"""
    print("Advanced Vector Search Patterns Implementation:")
    
    print("\n1. Hybrid Search Implementation:")
    print("-" * 38)
    print("""
class AdvancedVectorSearch:
    def __init__(self, client, model):
        self.client = client
        self.model = model
    
    def hybrid_search(self, index_name, query_text, 
                     vector_weight=0.7, text_weight=0.3, k=10):
        '''Combine vector and text search with custom weights'''
        query_embedding = self.model.encode(query_text).tolist()
        
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": k * 2,  # Get more candidates
                                    "boost": vector_weight
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["title^3", "content^1"],
                                "type": "best_fields",
                                "boost": text_weight
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["title", "content", "category", "timestamp"]
        }
        
        return self.client.search(index=index_name, body=search_body)
    
    def filtered_vector_search(self, index_name, query_text, 
                              filters=None, k=10):
        '''Vector search with efficient filtering'''
        query_embedding = self.model.encode(query_text).tolist()
        
        # Build filter conditions
        filter_conditions = []
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    filter_conditions.append({
                        "terms": {field: value}
                    })
                else:
                    filter_conditions.append({
                        "term": {field: value}
                    })
        
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [{
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": k * 3  # Compensate for filtering
                            }
                        }
                    }],
                    "filter": filter_conditions
                }
            }
        }
        
        return self.client.search(index=index_name, body=search_body)
    
    def multi_vector_search(self, index_name, text_query=None, 
                           image_vector=None, k=10):
        '''Search using multiple vector types'''
        must_clauses = []
        
        if text_query:
            text_embedding = self.model.encode(text_query).tolist()
            must_clauses.append({
                "knn": {
                    "text_embedding": {
                        "vector": text_embedding,
                        "k": k,
                        "boost": 1.0
                    }
                }
            })
        
        if image_vector:
            must_clauses.append({
                "knn": {
                    "image_embedding": {
                        "vector": image_vector,
                        "k": k,
                        "boost": 1.2
                    }
                }
            })
        
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "should": must_clauses,
                    "minimum_should_match": 1
                }
            }
        }
        
        return self.client.search(index=index_name, body=search_body)
    
    def vector_aggregation_search(self, index_name, query_text, k=100):
        '''Vector search with aggregations for analysis'''
        query_embedding = self.model.encode(query_text).tolist()
        
        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            },
            "aggs": {
                "categories": {
                    "terms": {
                        "field": "category",
                        "size": 10
                    }
                },
                "similarity_stats": {
                    "stats": {
                        "script": {
                            "source": "_score"
                        }
                    }
                },
                "time_histogram": {
                    "date_histogram": {
                        "field": "timestamp",
                        "calendar_interval": "day"
                    }
                }
            }
        }
        
        return self.client.search(index=index_name, body=search_body)
    """)
    
    print("\n2. Performance Optimization Techniques:")
    print("-" * 44)
    print("""
class VectorSearchOptimizer:
    def __init__(self, client):
        self.client = client
    
    def optimize_index_settings(self, index_name):
        '''Optimize index for vector search performance'''
        settings = {
            "index": {
                "refresh_interval": "30s",  # Slower refresh for better indexing
                "number_of_replicas": 0,    # Temporarily disable replicas
                "translog.durability": "async",
                "knn.algo_param.ef_search": 50,  # Lower for faster search
                "merge.policy.max_merge_at_once": 2,
                "merge.policy.segments_per_tier": 2
            }
        }
        
        return self.client.indices.put_settings(
            index=index_name,
            body=settings
        )
    
    def bulk_index_vectors(self, index_name, documents, batch_size=100):
        '''Efficient bulk indexing of vector documents'''
        from elasticsearch.helpers import bulk
        
        def generate_docs():
            for i, doc in enumerate(documents):
                yield {
                    "_index": index_name,
                    "_id": doc.get("id", i),
                    "_source": doc
                }
        
        # Bulk index with optimized settings
        success, failed = bulk(
            self.client,
            generate_docs(),
            chunk_size=batch_size,
            request_timeout=60,
            max_retries=3,
            initial_backoff=2,
            max_backoff=600
        )
        
        return success, failed
    
    def cache_frequent_queries(self, index_name, queries, ttl=3600):
        '''Implement query result caching'''
        import redis
        import json
        import hashlib
        
        cache = redis.Redis(host='localhost', port=6379, db=0)
        
        cached_results = {}
        for query in queries:
            # Create cache key
            query_hash = hashlib.md5(
                json.dumps(query, sort_keys=True).encode()
            ).hexdigest()
            
            cache_key = f"vector_search:{index_name}:{query_hash}"
            
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result:
                cached_results[query_hash] = json.loads(cached_result)
            else:
                # Execute search and cache result
                result = self.client.search(index=index_name, body=query)
                cache.setex(
                    cache_key,
                    ttl,
                    json.dumps(result, default=str)
                )
                cached_results[query_hash] = result
        
        return cached_results
    """)

demonstrate_advanced_patterns()

print("\n" + "=" * 80)
print("OPENSEARCH/ELASTICSEARCH INTERVIEW QUESTIONS")
print("=" * 80)

print("""
TOP INTERVIEW QUESTIONS & ANSWERS:

Q1: "What's the difference between OpenSearch and Elasticsearch vector search?"
A: "OpenSearch provides free k-NN algorithms with HNSW support, while 
   Elasticsearch requires paid license for optimized vector search.
   OpenSearch uses knn_vector field, Elasticsearch uses dense_vector."

Q2: "How do you handle vector search at scale?"
A: "Use proper sharding strategy, optimize JVM heap, implement caching,
   use appropriate k-NN parameters (ef_search, m), and monitor performance
   metrics like query latency and memory usage."

Q3: "Explain hybrid search implementation."
A: "Combine vector similarity search with traditional text search using
   bool queries with should clauses. Weight vector and text scores
   appropriately based on use case requirements."

Q4: "How do you optimize k-NN search performance?"
A: "Tune ef_search parameter (lower = faster, less accurate),
   use proper indexing settings, implement result caching,
   and consider pre-filtering vs post-filtering strategies."

Q5: "What are the key monitoring metrics for vector search?"
A: "Monitor k-NN query latency, memory usage, indexing throughput,
   cluster health, JVM heap usage, and cache hit ratios.
   Set up alerts for performance degradation."

Q6: "How do you handle real-time vector updates?"
A: "Use incremental indexing with proper refresh intervals,
   implement versioning for conflict resolution, consider
   hot-warm architecture for time-based data."

KEY TAKEAWAYS:
✅ OpenSearch offers better value for vector search (free k-NN)
✅ Proper cluster architecture is crucial for production
✅ Hybrid search combines best of vector and text search
✅ Performance tuning requires balancing accuracy vs speed
✅ Monitoring and alerting are essential for production systems

NEXT STEPS:
🔗 Set up OpenSearch cluster for hands-on practice
🔗 Implement hybrid search for your use case
🔗 Benchmark different k-NN parameters
🔗 Practice production deployment scenarios
🔗 Study cost optimization strategies
""")

print("\n" + "=" * 80)
print("END OF OPENSEARCH & ELASTICSEARCH GUIDE")
print("Master enterprise-grade vector search with confidence!")
print("=" * 80)
